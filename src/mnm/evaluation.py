"""Canonical, pure-NumPy evaluation shared by every method (GNN, Ridge, XGBoost).

Semantics mirror the training loss in ``mnm.task.FermentationTask``:
- biomass_rate: integrate from zero (cumsum) over the whole run
- other readouts: integrate from the readout's first valid measured value
  over [start_idx, T-2]
- specific rates scaled to absolute rates by the adjacent-timepoint biomass average
- rate clip +-5, monotonicity mask on the target, NaN-safe

The reported metric is the mean over runs of the per-run RMSE; the composite
sums the per-readout means. ``tests/test_evaluation_parity.py`` checks agreement
with the torch loss to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any

import numpy as np


@dataclass
class EvaluationConfig:
    readout_to_target_mapping: Dict[str, str]
    readouts_to_aggregate: List[str]
    specific_rate_readouts: Set[str]
    biomass_readout: str = 'biomass_rate'
    biomass_target: str = 'interpolated_biomass'
    specific_rate_use_measured_biomass: bool = True
    rate_clip_max: float = 5.0
    monotonicity_tolerance: float = 0.0
    loss_type: str = 'RMSE'

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['specific_rate_readouts'] = sorted(list(self.specific_rate_readouts))
        return d


@dataclass
class EvaluationResult:
    per_sample: Dict[str, Dict[str, Dict[int, float]]]
    per_readout_mean: Dict[str, Dict[str, float]]
    per_readout_std: Dict[str, Dict[str, float]]
    aggregated_sum: Dict[str, float]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            'per_sample': {
                split: {r: {str(k): v for k, v in per_s.items()} for r, per_s in readouts.items()}
                for split, readouts in self.per_sample.items()
            },
            'per_readout_mean': self.per_readout_mean,
            'per_readout_std': self.per_readout_std,
            'aggregated_sum': self.aggregated_sum,
            'config_snapshot': self.config_snapshot,
        }


def find_start_idx(target: np.ndarray) -> int:
    """First index where target is non-NaN and |target| > 1e-6.

    Returns 0 if no such index exists (as in training).
    """
    valid = (~np.isnan(target)) & (np.abs(target) > 1e-6)
    idxs = np.where(valid)[0]
    return int(idxs[0]) if idxs.size > 0 else 0


def create_mono_mask(target: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """Mask non-monotonic measurements.

    A timepoint is invalid (False) if its value decreases below
    `last_valid * (1 - tolerance)`. NaNs and near-zeros are skipped
    (remain True but don't update last_valid).
    """
    T = target.shape[0]
    mask = np.ones(T, dtype=bool)
    last_valid: Optional[float] = None

    for t in range(T):
        v = target[t]
        if np.isnan(v):
            continue
        if abs(v) < 1e-6:
            continue
        if last_valid is not None:
            threshold = last_valid * (1.0 - tolerance)
            if v < threshold:
                mask[t] = False
                continue
        last_valid = float(v)

    return mask


def integrate_rates(
    rates: np.ndarray,
    times: np.ndarray,
    C_0: float,
    start_idx: int,
    rate_clip_max: float = 5.0,
) -> np.ndarray:
    """Integrate rates into cumulative trajectory from start_idx.

    Matches FermentationTask._integrate (single-sample, float64 internally).
    """
    T = rates.shape[0]
    rates = rates.astype(np.float64)
    times = times.astype(np.float64)
    rates_clipped = np.clip(rates, -rate_clip_max, rate_clip_max)
    dt = times[1:] - times[:-1]
    increments = rates_clipped[:-1] * dt

    row = np.zeros(T, dtype=np.float64)
    if start_idx >= T:
        raise ValueError(f"start_idx {start_idx} >= T {T}")

    if start_idx < T - 1:
        inc_from_start = increments[start_idx:]
        cum = np.cumsum(inc_from_start)
        row[start_idx] = C_0
        row[start_idx + 1:] = C_0 + cum
    else:
        row[start_idx] = C_0

    return row


def integrate_from_zero(
    rates: np.ndarray,
    times: np.ndarray,
    rate_clip_max: float = 5.0,
) -> np.ndarray:
    """Biomass-rate style: start from 0, cumsum full trajectory.

    Matches FermentationTask:
        cumulative = cumsum(clip(rate[:-1], ±5) * dt)
        predicted = cat([0], cumulative)
    """
    T = rates.shape[0]
    rates = rates.astype(np.float64)
    times = times.astype(np.float64)
    rates_clipped = np.clip(rates[:-1], -rate_clip_max, rate_clip_max)
    dt = times[1:] - times[:-1]
    increments = rates_clipped * dt
    cumulative = np.cumsum(increments)
    out = np.zeros(T, dtype=np.float64)
    out[1:] = cumulative
    return out


def adjacent_biomass_average(
    biomass_measured: np.ndarray,
    biomass_predicted: Optional[np.ndarray],
    use_measured: bool,
) -> np.ndarray:
    """Per-interval biomass average with NaN fallback chain.

    Matches FermentationTask._adjacent_biomass. Returns [T-1] array.

    For each interval (t, t+1):
    - use_measured=True:  measured[t] | measured[t-1] | predicted[t]
                          measured[t+1] | measured[t+2] | predicted[t+1]
    - use_measured=False: predicted[t], predicted[t+1]
    - default if both unknown: 1.0
    """
    T = biomass_measured.shape[0]
    avg = np.zeros(T - 1, dtype=np.float64)

    for t in range(T - 1):
        bt: Optional[float] = None
        btn: Optional[float] = None

        if use_measured:
            if not np.isnan(biomass_measured[t]):
                bt = float(biomass_measured[t])
            elif t - 1 >= 0 and not np.isnan(biomass_measured[t - 1]):
                bt = float(biomass_measured[t - 1])

            if not np.isnan(biomass_measured[t + 1]):
                btn = float(biomass_measured[t + 1])
            elif t + 2 < T and not np.isnan(biomass_measured[t + 2]):
                btn = float(biomass_measured[t + 2])

            if bt is None and biomass_predicted is not None:
                bt = float(biomass_predicted[t])
            if btn is None and biomass_predicted is not None:
                btn = float(biomass_predicted[t + 1])
        else:
            if biomass_predicted is not None:
                bt = float(biomass_predicted[t])
                btn = float(biomass_predicted[t + 1])

        if bt is not None and btn is not None:
            avg[t] = (bt + btn) / 2.0
        elif bt is not None:
            avg[t] = bt
        elif btn is not None:
            avg[t] = btn
        else:
            avg[t] = 1.0

    return avg


def compute_loss(pred: np.ndarray, target: np.ndarray, loss_type: str) -> float:
    """Pointwise loss on flat 1-D arrays. Matches FermentationTask._loss.

    RMSE uses `sqrt(mse + 1e-8)` to match torch-side stability epsilon —
    required for the parity test to pass at 1e-6.
    """
    if pred.size == 0:
        return float('nan')
    diff = pred.astype(np.float64) - target.astype(np.float64)
    if loss_type == 'MSE':
        return float(np.mean(diff * diff))
    if loss_type == 'RMSE':
        return float(np.sqrt(np.mean(diff * diff) + 1e-8))
    if loss_type == 'MAE':
        return float(np.mean(np.abs(diff)))
    if loss_type == 'HUBER':
        a = np.abs(diff)
        quadratic = 0.5 * diff * diff
        linear = a - 0.5
        return float(np.mean(np.where(a < 1.0, quadratic, linear)))
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _to_numpy(x: Any) -> np.ndarray:
    """Accept torch tensor or ndarray; return ndarray on CPU."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_assay_metric_names(samples: List[Dict], provided: Optional[List[str]]) -> List[str]:
    if provided is not None:
        return list(provided)
    for s in samples:
        names = s.get('dataset_assay_metrics')
        if names is not None:
            return list(names)
    raise ValueError(
        "dataset_assay_metrics not found on any sample; pass it explicitly."
    )


def _evaluate_one_sample(
    sample: Dict[str, Any],
    rates_per_readout: Dict[str, np.ndarray],
    override_per_readout: Dict[str, np.ndarray],
    cfg: EvaluationConfig,
    assay_metric_names: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {readout: {'pred': array, 'target': array}} of valid-masked pairs.

    Returns raw (pred, target) arrays per readout so the caller can aggregate
    as pooled RMSE or mean-of-per-sample RMSE.
    """
    times = _to_numpy(sample['times']).astype(np.float64)
    assay = _to_numpy(sample['assay_metrics']).astype(np.float64)
    T = times.shape[0]
    out: Dict[str, Dict[str, np.ndarray]] = {}

    biomass_col = assay_metric_names.index(cfg.biomass_target)
    biomass_measured = assay[:, biomass_col]

    biomass_predicted: Optional[np.ndarray] = None
    bio_override = override_per_readout.get(cfg.biomass_readout)
    if bio_override is not None:
        biomass_predicted = _to_numpy(bio_override).astype(np.float64)
    elif cfg.biomass_readout in rates_per_readout:
        biomass_predicted = integrate_from_zero(
            _to_numpy(rates_per_readout[cfg.biomass_readout]).astype(np.float64),
            times,
            cfg.rate_clip_max,
        )

    _empty = {'pred': np.array([], dtype=np.float64),
              'target': np.array([], dtype=np.float64)}

    for readout in cfg.readouts_to_aggregate:
        target_name = cfg.readout_to_target_mapping.get(readout)
        if target_name is None or target_name not in assay_metric_names:
            out[readout] = _empty
            continue
        target_col = assay_metric_names.index(target_name)
        target = assay[:, target_col]
        non_nan = ~np.isnan(target)
        mono = create_mono_mask(target, cfg.monotonicity_tolerance)

        if readout == cfg.biomass_readout:
            if biomass_predicted is None:
                out[readout] = _empty
                continue
            valid = non_nan & mono
            if not valid.any():
                out[readout] = _empty
                continue
            out[readout] = {'pred': biomass_predicted[valid], 'target': target[valid]}
            continue

        override_cum = override_per_readout.get(readout)
        if override_cum is not None:
            pred_cum = _to_numpy(override_cum).astype(np.float64)
        else:
            if readout not in rates_per_readout:
                out[readout] = _empty
                continue
            rate_pred = _to_numpy(rates_per_readout[readout]).astype(np.float64)
            start_idx = find_start_idx(target)
            C_0 = float(target[start_idx]) if start_idx < T else 0.0

            if readout in cfg.specific_rate_readouts:
                avg_bio = adjacent_biomass_average(
                    biomass_measured,
                    biomass_predicted,
                    use_measured=cfg.specific_rate_use_measured_biomass,
                )
                scaled = rate_pred.copy()
                scaled[:-1] = rate_pred[:-1] * avg_bio
                rates_to_integrate = scaled
            else:
                rates_to_integrate = rate_pred

            pred_cum = integrate_rates(
                rates_to_integrate, times, C_0, start_idx, cfg.rate_clip_max
            )

        start_idx = find_start_idx(target)
        range_mask = np.zeros(T, dtype=bool)
        if start_idx < T - 1:
            range_mask[start_idx:T - 1] = True
        valid = range_mask & non_nan & mono
        if not valid.any():
            out[readout] = _empty
            continue
        out[readout] = {'pred': pred_cum[valid], 'target': target[valid]}

    return out


def evaluate_predictions(
    predicted_rates: Dict[int, Dict[str, np.ndarray]],
    samples: List[Dict[str, Any]],
    split_indices: Dict[str, List[int]],
    cfg: EvaluationConfig,
    cumulative_override: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    dataset_assay_metrics: Optional[List[str]] = None,
) -> EvaluationResult:
    """Canonical evaluation across methods. Pure numpy, deterministic."""
    cumulative_override = cumulative_override or {}
    assay_names = _get_assay_metric_names(samples, dataset_assay_metrics)

    per_sample: Dict[str, Dict[str, Dict[int, float]]] = {}
    per_readout_mean: Dict[str, Dict[str, float]] = {}
    per_readout_std: Dict[str, Dict[str, float]] = {}
    aggregated_sum: Dict[str, float] = {}

    for split, indices in split_indices.items():
        per_sample[split] = {r: {} for r in cfg.readouts_to_aggregate}
        # Collect all (pred, target) pairs across samples for pooled RMSE
        pooled_pairs: Dict[str, Dict[str, list]] = {
            r: {'pred': [], 'target': []} for r in cfg.readouts_to_aggregate
        }

        for si in indices:
            if si not in predicted_rates and si not in cumulative_override:
                raise KeyError(
                    f"No predictions for sample_idx {si} in split '{split}'. "
                    "Every sample in a split must have either rates or cumulative_override."
                )
            rates = predicted_rates.get(si, {})
            override = cumulative_override.get(si, {})
            pairs = _evaluate_one_sample(samples[si], rates, override, cfg, assay_names)
            for r, pt in pairs.items():
                p, t = pt['pred'], pt['target']
                if p.size > 0:
                    pooled_pairs[r]['pred'].append(p)
                    pooled_pairs[r]['target'].append(t)
                    # Per-sample RMSE for diagnostics
                    per_sample[split][r][si] = compute_loss(p, t, cfg.loss_type)
                else:
                    per_sample[split][r][si] = float('nan')

        # Mean-of-per-sample RMSE: one RMSE per run, then the mean across runs
        # (the training loss at batch size 1); pooled RMSE would weight long runs more.
        per_readout_mean[split] = {}
        per_readout_std[split] = {}
        for r in cfg.readouts_to_aggregate:
            sample_vals = np.array(
                [v for v in per_sample[split][r].values() if not np.isnan(v)],
                dtype=np.float64,
            )
            if sample_vals.size > 0:
                per_readout_mean[split][r] = float(sample_vals.mean())
                per_readout_std[split][r] = float(sample_vals.std(ddof=0))
            else:
                per_readout_mean[split][r] = float('nan')
                per_readout_std[split][r] = float('nan')
        aggregated_sum[split] = float(
            sum(v for v in per_readout_mean[split].values() if not np.isnan(v))
        )

    return EvaluationResult(
        per_sample=per_sample,
        per_readout_mean=per_readout_mean,
        per_readout_std=per_readout_std,
        aggregated_sum=aggregated_sum,
        config_snapshot=cfg.as_dict(),
    )
