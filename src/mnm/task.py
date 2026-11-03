"""Training objectives.

* :class:`MaskedFluxPretrainTask` - masked-flux reconstruction on the FBA corpus.
* :class:`FermentationTask` - rate prediction integrated to cumulative
  trajectories, RMSE against the measured cumulative assays.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from .registry import Registry


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

class _FluxLogTransform:
    """v -> (log10(v + eps) - log10(eps)) / (log10(v_ref + eps) - log10(eps)), in [0, 1] for v <= v_ref."""

    def __init__(self, log_eps: float, reference_max: float = 1000.0):
        self.log_eps = float(log_eps)
        self._log10_eps = math.log10(self.log_eps)
        self._range = math.log10(float(reference_max) + self.log_eps) - self._log10_eps
        self._range_inv = 1.0 / self._range

    def forward(self, values: Tensor) -> Tensor:
        return (torch.log10(torch.clamp(values, min=0.0) + self.log_eps) - self._log10_eps) * self._range_inv

    def scalar(self, value: float) -> float:
        return (math.log10(max(float(value), 0.0) + self.log_eps) - self._log10_eps) * self._range_inv


@Registry.register("task.masked_flux_pretrain")
class MaskedFluxPretrainTask(nn.Module):
    """Masked flux reconstruction.

    A fraction ``mask_rate`` of the half-nodes whose flux exceeds
    ``mask_positive_eps`` is masked. Each node's adapter output (flux
    statistics + metabolite embedding + missing-gene embedding) is concatenated
    with its visible (log-scaled) flux value and mask flag and projected back
    to ``d_model`` (``_flux_proj``); the encoder output is decoded by a linear
    head. Loss: bin-weighted MAE on masked nodes with target magnitude
    ``>= loss_eps`` plus ``zero_flux_loss_weight`` times a hinge penalty on
    predictions above the zero threshold where the target is (near) zero.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_rate: float = 0.15,
        mask_positive_eps: float = 1e-3,
        log_eps: float = 1e-3,
        zero_flux_loss_weight: float = 1.0,
        zero_flux_eps: float = -1,
        loss_eps: float = 1e-4,
        active_eps: float = 0.1,
        loss_bin_boundaries: Optional[List[float]] = None,
        loss_bin_weights: Optional[List[float]] = None,
        eval_mask_seed: Optional[int] = 0,
        global_salt: str = "TAFLP_v1",
        logger: Any = None,
    ):
        super().__init__()
        self.model = model
        self.logger = logger
        self.mask_rate = float(mask_rate)
        self.mask_positive_eps = float(mask_positive_eps)
        self.zero_flux_loss_weight = float(zero_flux_loss_weight)
        # -1 means "same threshold as the masking threshold"
        self.zero_flux_eps = float(mask_positive_eps) if float(zero_flux_eps) == -1.0 else float(zero_flux_eps)
        self.loss_eps = float(loss_eps)
        self.active_eps = float(active_eps)
        self._transform = _FluxLogTransform(log_eps)
        bounds = [float(b) for b in (loss_bin_boundaries or [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1000.0])]
        weights = [float(w) for w in (loss_bin_weights or [1.0] * (len(bounds) - 1))]
        if len(weights) != len(bounds) - 1:
            raise ValueError("loss_bin_weights needs len(loss_bin_boundaries) - 1 entries")
        self._bin_bounds = bounds
        self._bin_weight_values = [w / sum(weights) for w in weights]
        self.eval_mask_seed = eval_mask_seed
        self.global_salt = str(global_salt)
        d_model = getattr(model, "d_model", 128)
        self._flux_proj = nn.Linear(d_model + 2, d_model)
        self._reg_head: Optional[nn.Linear] = None
        self._model_components_initialized = False

    def parameters(self, recurse: bool = True):
        params = list(self.model.parameters())
        if self._reg_head is not None:
            params.extend(self._reg_head.parameters())
        params.extend(self._flux_proj.parameters())
        return iter(params)

    # -- masking ----------------------------------------------------------------
    def _mask_seed(self, sample_index: int, phase: str) -> int:
        if self.eval_mask_seed is not None:
            return int(self.eval_mask_seed)
        h = hashlib.sha256(f"{self.global_salt}_{phase}_{sample_index}".encode())
        return int(h.hexdigest(), 16) % (2 ** 32)

    def _select_mask_single(self, flux: Tensor, sample_index: int, phase: str) -> Tensor:
        mask = torch.zeros(flux.shape[0], dtype=torch.bool, device=flux.device)
        if self.mask_rate <= 0.0:
            return mask
        valid_idx = (flux > self.mask_positive_eps).nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            return mask
        n_mask = max(1, int(round(self.mask_rate * valid_idx.numel())))
        generator = None
        if phase != "train":  # fixed masks for validation / test
            generator = torch.Generator(device=flux.device)
            generator.manual_seed(self._mask_seed(sample_index, phase))
        perm = torch.randperm(valid_idx.numel(), device=flux.device, generator=generator)
        mask[valid_idx[perm[:n_mask]]] = True
        return mask

    def _prepare_batch(self, batch: Dict[str, Any], phase: str):
        flux = batch["sampled_flux_batch"]
        B, N = flux.shape
        base = batch["node_feature_base"].to(flux.device).unsqueeze(0).expand(B, -1, -1).clone()
        idx = batch.get("sample_indices")
        idx = idx if idx is not None else torch.arange(B, device=flux.device)
        mask = torch.zeros_like(flux, dtype=torch.bool)
        for b in range(B):
            mask[b] = self._select_mask_single(flux[b], int(idx[b].item()), phase)
        masked = flux.clone()
        masked[mask] = 0.0
        indicator = torch.zeros_like(flux)
        indicator[mask] = 1.0
        feats = torch.cat([base, masked.unsqueeze(-1), indicator.unsqueeze(-1)], dim=2)
        feats[:, :, -2] = self._transform.forward(feats[:, :, -2].clone())
        data = {"node_feature": feats, "edge_index": batch["edge_index"], "meta": batch.get("meta", {})}
        return data, self._transform.forward(flux), mask, flux

    # -- encoder ----------------------------------------------------------------
    def _ensure_model_components(self, meta: Dict[str, Any], device: torch.device) -> None:
        if self._model_components_initialized:
            return
        N = 2 * len(meta.get("reaction_ids", []))
        dummy = {
            "node_feature_base": torch.zeros(N, 1, device=device),
            "meta": meta,
            "batch_size": 1,
            "gpr_gene_vocab": ["gene_gpr_0"],
            "rna_gene_vocab": ["gene_rna_0"],
            "rna": {},
        }
        self.model._ensure_built(dummy)
        self._model_components_initialized = True

    def _encode(self, data: Dict[str, Any]) -> Tensor:
        feats = data["node_feature"]
        edge_index = data["edge_index"]
        meta = data["meta"]
        B, N, _ = feats.shape
        device = feats.device
        masked_flux, indicator = feats[:, :, -2], feats[:, :, -1]
        self._ensure_model_components(meta, device)
        model = self.model
        z_met = model.reaction_to_nodes(model.metabolites_encoder(meta, device))
        z_flux = model.flux_encoder(meta, device) if "flux_statistics_scaled" in meta else None
        z_gene = torch.zeros(N, model.gene_encoder.embedding_dim, device=device)
        z = model.adapter(z_flux, z_met, z_gene, m_flux=True, m_gene=False)
        z = torch.cat([z.unsqueeze(0).expand(B, -1, -1).clone(), masked_flux.unsqueeze(-1),
                       indicator.unsqueeze(-1)], dim=-1)
        self._flux_proj = self._flux_proj.to(device)
        z = self._flux_proj(z)
        film = torch.zeros(B, model.film_dim, device=device)

        def backbone(x, f):
            return model.gnn_backbone(x, edge_index, film_condition=f)

        if model.gnn_grad_checkpoint and torch.is_grad_enabled():
            step = model.gnn_phase_chunk_size or B
            h = torch.cat([torch.utils.checkpoint.checkpoint(backbone, z[i:i + step], film[i:i + step],
                                                             use_reentrant=False)
                           for i in range(0, B, step)], dim=0)
        else:
            h = backbone(z, film)
        if self._reg_head is None:
            self._reg_head = nn.Linear(int(h.shape[-1]), 1).to(device)
        return self._reg_head(h).squeeze(-1)

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        data, _, _, _ = self._prepare_batch(batch, phase="train")
        return self._encode(data)

    # -- loss -------------------------------------------------------------------
    def _sample_weights(self, target_raw: Tensor) -> Tensor:
        upper = target_raw.new_tensor(self._bin_bounds[1:])
        w = target_raw.new_tensor(self._bin_weight_values)
        idx = torch.clamp(torch.bucketize(target_raw.abs(), upper, right=False), 0, w.shape[0] - 1)
        return w[idx]

    def _losses(self, pred: Tensor, target: Tensor, mask: Tensor, target_raw: Tensor) -> Dict[str, Tensor]:
        raw_abs = target_raw.abs()
        loss_mask = mask & (raw_abs >= max(self.loss_eps, 0.0))
        active_mask = mask & (raw_abs >= max(self.active_eps, 0.0))
        sw = self._sample_weights(target_raw)
        lw = sw[loss_mask]
        count = lw.sum()
        if lw.numel() > 0 and float(count.detach().cpu().item()) > 0.0:
            diff = pred[loss_mask] - target[loss_mask]
            masked_loss = (lw * diff.abs()).sum() / count
            rmse = torch.sqrt((lw * diff ** 2).sum() / count + 1e-12)
        else:
            masked_loss = rmse = pred.new_tensor(0.0)
        zero_thr = self._transform.scalar(self.zero_flux_eps)
        zero_mask = target < zero_thr
        zero_loss = torch.relu(pred[zero_mask] - zero_thr).mean() if zero_mask.any() else pred.new_tensor(0.0)
        aw = sw[active_mask]
        a_count = aw.sum()
        if aw.numel() > 0 and float(a_count.detach().cpu().item()) > 0.0:
            active_mae = (aw * (pred[active_mask] - target[active_mask]).abs()).sum() / a_count
        else:
            active_mae = pred.new_tensor(0.0)
        return {
            "loss": masked_loss + self.zero_flux_loss_weight * zero_loss,
            "loss_count": count,
            "loss_masked_raw": masked_loss,
            "loss_masked_raw_count": count,
            "loss_zero_reg_raw": zero_loss,
            "mae_masked": masked_loss,
            "mae_masked_count": count,
            "rmse_masked": rmse,
            "rmse_masked_count": count,
            "active_mae": active_mae,
            "active_mae_count": a_count,
        }

    def _step(self, batch: Dict[str, Any], phase: str) -> Dict[str, Tensor]:
        data, target, mask, target_raw = self._prepare_batch(batch, phase)
        return self._losses(self._encode(data), target, mask, target_raw)

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def test_step(self, batch):
        return self._step(batch, "test")


# ---------------------------------------------------------------------------
# Finetuning
# ---------------------------------------------------------------------------

def _first_valid_index(target: Tensor) -> Tensor:
    """First index per row that is non-NaN and |x| > 1e-6 (0 if none)."""
    B = target.shape[0]
    valid = ~torch.isnan(target) & (target.abs() > 1e-6)
    out = torch.zeros(B, dtype=torch.long, device=target.device)
    for b in range(B):
        idx = torch.where(valid[b])[0]
        out[b] = idx[0] if len(idx) > 0 else 0
    return out


def _monotonicity_mask(target: Tensor, tolerance: float = 0.0) -> Tensor:
    """False where a cumulative measurement drops below (1 - tolerance) times
    the previous valid measurement (NaN and ~0 values are skipped)."""
    B, T = target.shape
    mask = torch.ones_like(target, dtype=torch.bool)
    for b in range(B):
        last = None
        for t in range(T):
            if torch.isnan(target[b, t]):
                continue
            v = target[b, t].item()
            if abs(v) < 1e-6:
                continue
            if last is not None and v < last * (1.0 - tolerance):
                mask[b, t] = False
                continue
            last = v
    return mask


@Registry.register("task.fermentation")
class FermentationTask(nn.Module):
    """Integrate predicted rates to cumulative trajectories; RMSE loss.

    * ``biomass_rate``: forward-Euler integral from 0 (rates clipped to +-5),
      compared with the measured biomass at valid points.
    * other readouts: specific rates (``readout_is_specific_rate``) are scaled
      by the adjacent-time-point biomass average (predicted biomass unless
      ``readout_specific_rate_use_measured_biomass``; detached when
      ``specific_rate_detach_biomass``), then integrated from the readout's
      first valid measured value over ``[start, T-2]``.
    * ``growth_rate`` (graph-baseline configs): log-biomass increments,
      optionally teacher-forced from the measured biomass.

    Missing (NaN) and non-monotone cumulative measurements are excluded.
    The loss is the sum of the per-readout losses.
    """

    def __init__(
        self,
        model: nn.Module,
        readout_to_target_mapping: Dict[str, str],
        loss_type: str = "RMSE",
        monotonicity_tolerance: float = 0.0,
        readout_is_specific_rate: bool = True,
        readout_specific_rate_use_measured_biomass: bool = False,
        specific_rate_detach_biomass: bool = False,
        specific_rate_biomass_target: Optional[str] = None,
        use_teacher_forcing_for_growth_rate: bool = False,
        growth_rate_min_biomass_threshold: float = 0.1,
        rate_clip_max: float = 5.0,
        logger: Any = None,
    ):
        super().__init__()
        self.model = model
        self.logger = logger
        self.readout_to_target_mapping = dict(readout_to_target_mapping)
        self.loss_type = loss_type.upper()
        self.monotonicity_tolerance = float(monotonicity_tolerance)
        self.readout_is_specific_rate = bool(readout_is_specific_rate)
        self.readout_specific_rate_use_measured_biomass = bool(readout_specific_rate_use_measured_biomass)
        self.specific_rate_detach_biomass = bool(specific_rate_detach_biomass)
        # biomass column scaling the specific rates when the model has no biomass readout
        self.specific_rate_biomass_target = specific_rate_biomass_target
        self.use_teacher_forcing_for_growth_rate = bool(use_teacher_forcing_for_growth_rate)
        self.growth_rate_min_biomass_threshold = float(growth_rate_min_biomass_threshold)
        self.rate_clip_max = float(rate_clip_max)

    def _loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "RMSE":
            return torch.sqrt(F.mse_loss(pred, target) + 1e-8)
        if self.loss_type == "MAE":
            return F.l1_loss(pred, target)
        raise ValueError(f"Unknown loss type {self.loss_type}")

    def _integrate(self, rates: Tensor, times: Tensor, C_0: Tensor, start_idx: Tensor) -> Tensor:
        B, T = rates.shape
        inc = torch.clamp(rates, -self.rate_clip_max, self.rate_clip_max)[:, :-1] * (times[:, 1:] - times[:, :-1])
        rows = []
        for b in range(B):
            s = int(start_idx[b].item())
            row = torch.zeros(T, dtype=torch.float32, device=rates.device)
            row[s] = C_0[b]
            if s < T - 1:
                row[s + 1:] = C_0[b] + torch.cumsum(inc[b, s:], dim=0)
            rows.append(row)
        return torch.stack(rows, dim=0)

    @staticmethod
    def _adjacent_biomass(measured: Tensor, predicted: Optional[Tensor], use_measured: bool) -> Tensor:
        """Average biomass over each interval [t, t+1]; with ``use_measured``
        missing values fall back to t-1 / t+2, then to the prediction."""
        B, T = measured.shape
        avg = torch.zeros(B, T - 1, dtype=torch.float32, device=measured.device)
        for b in range(B):
            for t in range(T - 1):
                bt = btn = None
                if use_measured:
                    if not torch.isnan(measured[b, t]):
                        bt = measured[b, t]
                    elif t - 1 >= 0 and not torch.isnan(measured[b, t - 1]):
                        bt = measured[b, t - 1]
                    if not torch.isnan(measured[b, t + 1]):
                        btn = measured[b, t + 1]
                    elif t + 2 < T and not torch.isnan(measured[b, t + 2]):
                        btn = measured[b, t + 2]
                    if bt is None and predicted is not None:
                        bt = predicted[b, t]
                    if btn is None and predicted is not None:
                        btn = predicted[b, t + 1]
                elif predicted is not None:
                    bt, btn = predicted[b, t], predicted[b, t + 1]
                if bt is not None and btn is not None:
                    avg[b, t] = (bt + btn) / 2.0
                elif bt is not None:
                    avg[b, t] = bt
                elif btn is not None:
                    avg[b, t] = btn
                else:
                    avg[b, t] = 1.0
        return avg

    def compute_batch(self, batch: Dict[str, Any], capture: bool = False
                      ) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Any]]]:
        """Loss terms and, with ``capture``, the predicted rates / trajectories."""
        times = batch["times"]
        assay = batch["assay_metrics"]
        names = batch["dataset_assay_metrics"]
        device = times.device
        predictions = self.model(batch=batch, query_times=times,
                                 initial_condition=batch["initial_condition"].to(device),
                                 sensor_stats=batch["sensor_stats"])
        B, T = times.shape
        losses: Dict[str, Tensor] = {}
        total = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
        captured = {"metadata": batch["metadata_list"], "times": times.detach().cpu(), "readouts": {}} if capture \
            else None

        predicted_biomass = biomass_measured = None
        if "biomass_rate" in predictions:
            target_name = self.readout_to_target_mapping.get("biomass_rate")
            if target_name is not None:
                biomass_measured = assay[:, :, names.index(target_name)]
                inc = torch.clamp(predictions["biomass_rate"][:, :-1], -self.rate_clip_max, self.rate_clip_max) \
                    * (times[:, 1:] - times[:, :-1])
                predicted_biomass = torch.cat([torch.zeros(B, 1, dtype=inc.dtype, device=device),
                                               torch.cumsum(inc, dim=1)], dim=1)
        if biomass_measured is None and self.specific_rate_biomass_target is not None:
            biomass_measured = assay[:, :, names.index(self.specific_rate_biomass_target)]

        order: List[str] = []
        for name in predictions:
            if name == "biomass_rate":
                order.insert(0, name)
            elif name == "growth_rate" and "biomass_rate" in order:
                order.insert(order.index("biomass_rate") + 1, name)
            else:
                order.append(name)

        for name in order:
            rates = predictions[name]
            target_name = self.readout_to_target_mapping.get(name)
            if target_name is None:
                continue
            target = assay[:, :, names.index(target_name)]

            if name == "biomass_rate":
                if predicted_biomass is None or biomass_measured is None:
                    continue
                mono = _monotonicity_mask(biomass_measured, self.monotonicity_tolerance)
                valid = ~torch.isnan(biomass_measured) & mono
                if valid.any():
                    loss = self._loss(predicted_biomass[valid], biomass_measured[valid])
                    if torch.isfinite(loss):
                        total = total + loss
                        losses[f"loss_{name}"] = loss.detach()
                if capture:
                    captured["readouts"][name] = {"predicted_cumulative": predicted_biomass.detach().cpu(),
                                                  "measured_cumulative": biomass_measured.detach().cpu(),
                                                  "predicted_rates": rates.detach().cpu(), "target_name": target_name}
                continue

            if name == "growth_rate":
                if predicted_biomass is None:
                    warnings.warn("growth_rate needs a biomass_rate readout")
                    continue
                eps = 1e-6
                tf = self.use_teacher_forcing_for_growth_rate and biomass_measured is not None
                ref = biomass_measured if tf else predicted_biomass
                ln_ref = torch.log(torch.clamp(ref, min=eps))
                ln_pred = torch.zeros_like(ln_ref)
                for b in range(B):
                    ln_pred[b, 0] = ln_ref[b, 0]
                    for t in range(T - 1):
                        dt = times[b, t + 1] - times[b, t]
                        g = torch.clamp(rates[b, t], -self.rate_clip_max, self.rate_clip_max)
                        if self.use_teacher_forcing_for_growth_rate and biomass_measured is not None \
                                and not torch.isnan(biomass_measured[b, t]):
                            start = torch.log(torch.clamp(biomass_measured[b, t], min=eps))
                        else:
                            start = ln_pred[b, t]
                        ln_pred[b, t + 1] = start + g * dt
                thr = self.growth_rate_min_biomass_threshold
                if tf:
                    mono = _monotonicity_mask(biomass_measured, self.monotonicity_tolerance)
                    valid = ~torch.isnan(biomass_measured) & mono & (biomass_measured > 1e-3) & (biomass_measured > thr)
                    ln_target = torch.log(torch.clamp(biomass_measured, min=eps))
                else:
                    valid = (predicted_biomass > 1e-3) & (predicted_biomass > thr)
                    ln_target = torch.log(torch.clamp(predicted_biomass, min=eps))
                if valid.any():
                    loss = self._loss(ln_pred[valid], ln_target[valid])
                    if torch.isfinite(loss):
                        total = total + loss
                        losses[f"loss_{name}"] = loss.detach()
                if capture:
                    captured["readouts"][name] = {"predicted_rates": rates.detach().cpu(), "target_name": target_name}
                continue

            to_integrate = rates
            if self.readout_is_specific_rate:
                if predicted_biomass is None and biomass_measured is None:
                    warnings.warn(f"No biomass available to scale {name}; treated as an absolute rate")
                else:
                    pb = predicted_biomass.detach() if (self.specific_rate_detach_biomass
                                                        and predicted_biomass is not None) else predicted_biomass
                    avg = self._adjacent_biomass(
                        biomass_measured if biomass_measured is not None else torch.zeros_like(predicted_biomass),
                        pb, use_measured=self.readout_specific_rate_use_measured_biomass)
                    scaled = torch.zeros_like(rates)
                    scaled[:, :-1] = rates[:, :-1] * avg
                    scaled[:, -1] = rates[:, -1]
                    to_integrate = scaled
            start = _first_valid_index(target)
            C_0 = torch.zeros(B, dtype=torch.float32, device=device)
            for b in range(B):
                C_0[b] = target[b, int(start[b].item())]
            pred_cum = self._integrate(to_integrate, times, C_0, start)
            in_range = torch.zeros_like(target, dtype=torch.bool)
            for b in range(B):
                s = int(start[b].item())
                if s < T - 1:
                    in_range[b, s:T - 1] = True
            mono = _monotonicity_mask(target, self.monotonicity_tolerance)
            valid = in_range & ~torch.isnan(target) & mono
            if valid.any():
                loss = self._loss(pred_cum[valid], target[valid])
                if torch.isfinite(loss):
                    total = total + loss
                    losses[f"loss_{name}"] = loss.detach()
            if capture:
                captured["readouts"][name] = {"predicted_cumulative": pred_cum.detach().cpu(),
                                              "measured_cumulative": target.detach().cpu(),
                                              "predicted_rates": rates.detach().cpu(), "target_name": target_name}
        return {"loss": total, **losses}, captured

    def forward(self, batch):
        return self.compute_batch(batch)[0]

    def training_step(self, batch):
        return self.compute_batch(batch)[0]

    def validation_step(self, batch):
        with torch.no_grad():
            return self.compute_batch(batch)[0]

    def test_step(self, batch):
        with torch.no_grad():
            return self.compute_batch(batch)[0]
