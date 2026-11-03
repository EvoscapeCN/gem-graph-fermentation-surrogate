"""The surrogate: GEM-graph encoder + temporal kernel + GRU readout + biomass head."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from .layers import (
    FiLMGenerator,
    FluxModalityEncoder,
    GeneCountsModalityEncoder,
    KernelGating,
    MetabolitesModalityEncoder,
    ModalityAdapter,
    _causal_admissible,
)
from .registry import Registry


class LagPhaseGating(nn.Module):
    """r(t) = sigmoid(a (t - tau)) * r_raw(t), with learnable a and tau."""

    def __init__(self, tau_initial: float = 6.0, a_initial: float = 1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau_initial)))
        self.a = nn.Parameter(torch.tensor(float(a_initial)))

    def forward(self, rates: Tensor, times: Tensor) -> Tensor:
        return torch.sigmoid(self.a * (times - self.tau)) * rates


class BiomassHead(nn.Module):
    """Absolute biomass rate from global features, bypassing the graph's
    ``biomass_C`` node; output clipped to ``[-rate_clip_max, rate_clip_max]``
    and passed through a lag-phase gate."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1,
                 lag_tau_initial: float = 6.0, lag_a_initial: float = 1.0, rate_clip_max: float = 5.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.rate_clip_max = float(rate_clip_max)
        self.lag_gating = LagPhaseGating(lag_tau_initial, lag_a_initial)
        self.mlp: Optional[nn.Sequential] = None  # built on first call

    def _ensure_built(self, in_dim: int, device: torch.device) -> None:
        if self.mlp is not None:
            return
        layers: List[nn.Module] = [nn.Linear(in_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)])
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers).to(device)

    def forward(self, global_features: Tensor, times: Tensor) -> Tensor:
        self._ensure_built(global_features.shape[-1], global_features.device)
        B, T, Fdim = global_features.shape
        raw = self.mlp(global_features.reshape(B * T, Fdim)).reshape(B, T)
        raw = torch.clamp(raw, -self.rate_clip_max, self.rate_clip_max)
        return self.lag_gating(raw, times)


@Registry.register("model.gru_readout")
class GRUReadout(nn.Module):
    """Autoregressive GRU over query times with one MLP head per readout.

    At step t the input is ``[features_t, y_{t-1}]``; the initial hidden state
    is an MLP of ``[h_global(t_0), initial_condition]``.
    """

    def __init__(self, readout_names: List[str], hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.0,
                 head_hidden_dim: int = 64, head_layers: int = 2, head_dropout: float = 0.1,
                 head_final_activation: str = "none"):
        super().__init__()
        self.readout_names = list(readout_names)
        self.num_readouts = len(self.readout_names)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_layers = int(head_layers)
        self.head_dropout = float(head_dropout)
        self.head_final_activation = str(head_final_activation or "none").lower()
        self.gru: Optional[nn.GRU] = None
        self.h0_mlp: Optional[nn.Sequential] = None
        self.readout_heads = nn.ModuleDict()

    def _ensure_built(self, in_dim: int, h0_in_dim: Optional[int], device: torch.device) -> None:
        if self.gru is None:
            self.gru = nn.GRU(input_size=in_dim + self.num_readouts, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, dropout=self.dropout if self.num_layers > 1 else 0.0,
                              batch_first=True).to(device)
        if h0_in_dim is not None and self.h0_mlp is None:
            self.h0_mlp = nn.Sequential(nn.Linear(h0_in_dim, self.hidden_dim), nn.ReLU()).to(device)
        if len(self.readout_heads) == 0:
            for name in self.readout_names:
                if self.head_layers <= 1:
                    layers: List[nn.Module] = [nn.Linear(self.hidden_dim, 1)]
                else:
                    layers = [nn.Linear(self.hidden_dim, self.head_hidden_dim), nn.ReLU(),
                              nn.Dropout(self.head_dropout), nn.Linear(self.head_hidden_dim, 1)]
                if self.head_final_activation == "relu":
                    layers.append(nn.ReLU())
                elif self.head_final_activation == "softplus":
                    layers.append(nn.Softplus())
                elif self.head_final_activation != "none":
                    raise ValueError(f"Unknown head_final_activation {self.head_final_activation!r}")
                self.readout_heads[name] = nn.Sequential(*layers).to(device)

    def forward(self, base_features: Tensor, h0_features: Optional[Tensor] = None) -> Dict[str, Tensor]:
        B, T, Fdim = base_features.shape
        device = base_features.device
        self._ensure_built(Fdim, h0_features.shape[-1] if h0_features is not None else None, device)
        if h0_features is not None:
            h0 = self.h0_mlp(h0_features)
        else:
            h0 = torch.zeros(B, self.hidden_dim, dtype=base_features.dtype, device=device)
        h = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        prev_y = torch.zeros(B, self.num_readouts, dtype=base_features.dtype, device=device)
        outputs: Dict[str, List[Tensor]] = {name: [] for name in self.readout_names}
        for t in range(T):
            out_t, h = self.gru(torch.cat([base_features[:, t, :], prev_y], dim=-1).unsqueeze(1), h)
            h_t = out_t[:, 0, :]
            ys = []
            for name in self.readout_names:
                y = self.readout_heads[name](h_t).squeeze(-1)
                outputs[name].append(y)
                ys.append(y)
            prev_y = torch.stack(ys, dim=-1)
        return {name: torch.stack(v, dim=1) for name, v in outputs.items()}


class ExpressionReadoutAdapter(nn.Module):
    """Whole-transcriptome readout features.

    Maps each snapshot's expression features (by default its 16 whitened
    train-split principal components) through ``in -> hidden -> out`` and
    interpolates the per-snapshot embeddings to the query times with its own
    Gaussian kernel. Bypasses the graph encoder.
    """

    def __init__(self, in_dim: int, out_dim: int = 16, hidden_dim: Optional[int] = None, dropout: float = 0.1,
                 gamma_initial: float = 0.01, causal_mask: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.causal_mask = bool(causal_mask)
        hidden_dim = hidden_dim or out_dim * 2
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        self.gamma_raw = nn.Parameter(torch.tensor(float(math.log(math.exp(gamma_initial) - 1.0))))
        self.eps = 1e-6

    def forward(self, features: Tensor, rna_times: Tensor, query_times: Tensor, rna_mask: Tensor) -> Tensor:
        """features [B, K, G], rna_times [B, K], query_times [B, T], rna_mask [B, K] -> [B, T, out]."""
        B, K, G = features.shape
        z = self.mlp(features.reshape(B * K, G)).reshape(B, K, self.out_dim)
        gamma = F.softplus(self.gamma_raw) + self.eps
        out = []
        for b in range(B):
            t_k, t_q, mask_b = rna_times[b], query_times[b], rna_mask[b]
            dist_sq = (t_q.unsqueeze(1) - t_k.unsqueeze(0)) ** 2
            admissible = mask_b.unsqueeze(0).expand_as(dist_sq)
            if self.causal_mask:
                admissible = _causal_admissible(admissible, t_k, t_q, mask_b)
            w = torch.softmax((-gamma * dist_sq).masked_fill(~admissible, float("-inf")), dim=-1)
            out.append(w @ z[b])
        return torch.stack(out, dim=0)


_FREEZE_ATTR = {
    "gnn": "gnn_backbone",
    "adapter": "adapter",
    "metabolites_encoder": "metabolites_encoder",
    "flux_encoder": "flux_encoder",
    "flux_proj": "flux_proj",
}


@Registry.register("model.metabolic_gnn")
class MetabolicGNN(nn.Module):
    """GEM-graph surrogate (pretrain and finetune modes share all encoders).

    Finetune forward, per run b and RNA snapshot k:
      node input  = adapter([e_flux_missing, metabolite emb., GPR gene emb.(expression_k)])
      h_k         = GNN(node input)                                      [N, d]
      h(t)        = sum_k kernel_k(t) h_k                                [T, N, d]
    Readout input at each query time: per-readout pooled embeddings over the
    reactions in ``readout_reaction_ids``, global mean pooling, sensor window
    statistics, initial conditions, (biomass head output) and the optional
    whole-transcriptome readout features; a GRU emits the rates.

    Args (selected):
        gnn_backbone: graph encoder (``model.gat_encoder`` / ``model.node_mlp_encoder``);
            rebuilt with in/out width ``d_model``.
        readout_reaction_ids: readout name -> GEM reaction ids pooled for it.
        biomass_head: ``{enabled, hidden_dim, num_layers, dropout}``; when enabled,
            ``biomass_rate`` comes from :class:`BiomassHead` and enters the GRU
            (detached) as a side input.
        expression_readout_adapter: ``{enabled, input, hidden_dim, out_dim, dropout}``;
            ``input`` is a key of ``batch['rna']`` (``rna_gene_pcs`` in the paper).
        use_film: FiLM-modulate encoder layers with a projection of all RNA genes (appendix).
        reuse_pretrain_flux_proj: keep the pretraining flux projection at finetune,
            feeding constant (v, m) = (flux_proj_value, flux_proj_mask_value) (appendix).
        kernel_causal_mask: only snapshots at or before t contribute (appendix).
        checkpoint_path / checkpoint_load_modules / checkpoint_strict: pretrained
            encoder to load (``gnn``, ``adapter``, ``metabolites_encoder``,
            ``flux_encoder``, ``flux_proj``).
        freeze_modules / freeze_epochs: modules frozen for the first epochs
            (``[pretrain]`` = the loaded modules).
        metabolites_shuffle_seed: permute the metabolite embedding rows after
            loading (structural control).
    """

    def __init__(
        self,
        d_model: int = 128,
        mode: str = "finetune",
        gnn_backbone: Optional[nn.Module] = None,
        adapter: Optional[Dict[str, Any]] = None,
        input_heads: Optional[Dict[str, Any]] = None,
        readout_heads: Optional[Dict[str, Any]] = None,
        readout_reaction_ids: Optional[Dict[str, List[str]]] = None,
        biomass_head: Optional[Dict[str, Any]] = None,
        expression_readout_adapter: Optional[Dict[str, Any]] = None,
        gamma_initial: float = 0.01,
        kernel_causal_mask: bool = False,
        use_film: bool = False,
        film_dim: int = 128,
        reuse_pretrain_flux_proj: bool = False,
        flux_proj_value: float = 0.0,
        flux_proj_mask_value: float = 1.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_load_modules: Optional[List[str]] = None,
        checkpoint_strict: bool = True,
        freeze_modules: Optional[List[str]] = None,
        freeze_epochs: int = 0,
        metabolites_shuffle_seed: Optional[int] = None,
        gnn_phase_chunk_size: Optional[int] = None,
        gnn_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mode = mode.lower()
        assert self.mode in {"pretrain", "finetune"}
        self.use_film = bool(use_film)
        self.film_dim = int(film_dim)
        self.reuse_pretrain_flux_proj = bool(reuse_pretrain_flux_proj)
        self.flux_proj_value = float(flux_proj_value)
        self.flux_proj_mask_value = float(flux_proj_mask_value)
        self.flux_proj: Optional[nn.Linear] = None
        self._expr_adapter_cfg = expression_readout_adapter
        self.checkpoint_path = checkpoint_path
        self.checkpoint_load_modules = list(checkpoint_load_modules or [])
        self.checkpoint_strict = bool(checkpoint_strict)
        self.gamma_initial = float(gamma_initial)
        self.kernel_causal_mask = bool(kernel_causal_mask)
        self.gnn_phase_chunk_size = int(gnn_phase_chunk_size) if gnn_phase_chunk_size else None
        self.gnn_grad_checkpoint = bool(gnn_grad_checkpoint)
        self.metabolites_shuffle_seed = int(metabolites_shuffle_seed) if metabolites_shuffle_seed is not None else None

        self.freeze_epochs = int(freeze_epochs)
        freeze = [str(m) for m in (freeze_modules or [])]
        self.freeze_module_names = list(self.checkpoint_load_modules) if freeze == ["pretrain"] else freeze
        self._modules_frozen = False

        if not readout_reaction_ids:
            raise ValueError("readout_reaction_ids is required")
        self.readout_reaction_ids = dict(readout_reaction_ids)
        self.readout_names = list(self.readout_reaction_ids.keys())

        self._biomass_head_cfg = biomass_head
        self._biomass_head_enabled = bool(biomass_head and biomass_head.get("enabled", False))

        self.flux_encoder: Optional[nn.Module] = None
        self.metabolites_encoder: Optional[nn.Module] = None
        self.gene_encoder: Optional[nn.Module] = None
        self.adapter: Optional[nn.Module] = None
        self.kernel_gating: Optional[nn.Module] = None
        self.readout_gru: Optional[GRUReadout] = None
        self.biomass_head_module: Optional[BiomassHead] = None
        self.expression_adapter: Optional[ExpressionReadoutAdapter] = None
        # BiomassHead carries its own lag gate; otherwise biomass_rate from the GRU is gated here
        self.lag_phase_gating = None if self._biomass_head_enabled else LagPhaseGating(6.0, 1.0)

        if gnn_backbone is None or not hasattr(gnn_backbone, "init_kwargs"):
            raise ValueError("gnn_backbone must be a graph encoder config")
        self._gnn_cls = type(gnn_backbone)
        self._gnn_kwargs = dict(gnn_backbone.init_kwargs)
        self.gnn_backbone: Optional[nn.Module] = None
        self._adapter_cfg = adapter or {}
        self._input_heads_cfg = input_heads or {}
        head = (readout_heads or {}).get("all_readouts")
        if not isinstance(head, GRUReadout):
            raise ValueError("readout_heads.all_readouts must be a model.gru_readout config")
        self._pending = {"readout_head": head}  # plain dict: registered as readout_gru when built
        self._reaction_to_node_indices: Dict[str, List[int]] = {}
        self._is_built = False

    @staticmethod
    def reaction_to_nodes(z: Tensor) -> Tensor:
        """[m, ...] per-reaction features -> [2m, ...] in the [fwd, rev] half-node layout."""
        return torch.cat([z, z], dim=0)

    # -- lazy construction ----------------------------------------------------
    def _ensure_built(self, batch: Dict[str, Any]) -> None:
        if self._is_built:
            return
        device = batch["node_feature_base"].device
        meta = batch.get("meta", {})
        num_gpr_genes = len(batch.get("gpr_gene_vocab", []))
        num_rna_genes = len(batch.get("rna_gene_vocab", []))
        reaction_ids = meta.get("reaction_ids", [])
        half = meta.get("half_node_reaction_ids", [])
        for rid in reaction_ids:
            self._reaction_to_node_indices[rid] = [i for i, h in enumerate(half) if h == rid]

        self.flux_encoder = FluxModalityEncoder().to(device)
        flux_dim = FluxModalityEncoder.output_dim(meta)
        met_cfg = self._input_heads_cfg.get("metabolites", {})
        met_dim = met_cfg.get("embedding_dim", 300)
        self.metabolites_encoder = MetabolitesModalityEncoder(
            embedding_dim=met_dim, mode=met_cfg.get("mode", "mean"),
            use_stoichiometry_weight=met_cfg.get("use_stoichiometry_weight", True)).to(device)
        gene_cfg = self._input_heads_cfg.get("gene_counts", {})
        gene_dim = gene_cfg.get("embedding_dim", 128)
        self.gene_encoder = GeneCountsModalityEncoder(embedding_dim=gene_dim,
                                                      mode=gene_cfg.get("mode", "rel_mean")).to(device)
        self.adapter = ModalityAdapter(flux_dim=flux_dim, metabolites_dim=met_dim, gene_dim=gene_dim,
                                       out_dim=self.d_model, dropout=self._adapter_cfg.get("dropout", 0.1)).to(device)
        self.gnn_backbone = self._gnn_cls(in_dim=self.d_model, out_dim=self.d_model, **self._gnn_kwargs).to(device)
        if self.reuse_pretrain_flux_proj:
            self.flux_proj = nn.Linear(self.d_model + 2, self.d_model).to(device)
        self.kernel_gating = KernelGating(gamma_initial=self.gamma_initial,
                                          causal_mask=self.kernel_causal_mask).to(device)
        if self.use_film:
            self.rna_projection = nn.Sequential(nn.Linear(num_rna_genes, self.film_dim),
                                                nn.LayerNorm(self.film_dim), nn.ReLU())
            self.rna_film_generator = FiLMGenerator(cond_dim=self.film_dim, hidden_dim=self.gnn_backbone.hidden_dim)
            self.rna_projection.to(device)
            self.rna_film_generator.to(device)
            self.gnn_backbone.film_generator = self.rna_film_generator
            self.gnn_backbone.enable_film = True
        else:
            self.rna_projection = None
            self.rna_film_generator = None
            self.gnn_backbone.enable_film = False

        cfg = self._expr_adapter_cfg
        if cfg and cfg.get("enabled", False):
            self._expr_input = cfg.get("input", "rna_gene_pcs")
            self.expression_adapter = ExpressionReadoutAdapter(
                in_dim=batch["rna"][self._expr_input].shape[-1], out_dim=cfg.get("out_dim", 16),
                hidden_dim=cfg.get("hidden_dim"), dropout=cfg.get("dropout", 0.1),
                gamma_initial=self.gamma_initial, causal_mask=self.kernel_causal_mask).to(device)
        if self._biomass_head_enabled:
            bh = self._biomass_head_cfg
            self.biomass_head_module = BiomassHead(
                hidden_dim=bh.get("hidden_dim", 128), num_layers=bh.get("num_layers", 2),
                dropout=bh.get("dropout", 0.1), lag_tau_initial=bh.get("lag_tau_initial", 6.0),
                lag_a_initial=bh.get("lag_a_initial", 1.0), rate_clip_max=bh.get("rate_clip_max", 5.0)).to(device)
        self.readout_gru = self._pending.pop("readout_head").to(device)

        if self.checkpoint_path and self.checkpoint_load_modules:
            if "metabolites_encoder" in self.checkpoint_load_modules:
                _ = self.metabolites_encoder(meta, device)  # build the embedding table before loading
            self._load_checkpoint(device)
        if self.metabolites_shuffle_seed is not None:
            self._shuffle_metabolite_embeddings()
        self._is_built = True

    def _shuffle_metabolite_embeddings(self) -> None:
        """Permute metabolite embedding rows (fixed seed): keeps the multiset of
        vectors and the stoichiometric weighting, destroys which species owns
        which vector. Applied after the pretrained table is loaded."""
        bag = self.metabolites_encoder.embedding_bag
        if bag is None:
            raise RuntimeError("metabolite embedding table not built")
        n = bag.weight.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.metabolites_shuffle_seed))
        with torch.no_grad():
            bag.weight.copy_(bag.weight[perm.to(bag.weight.device)])

    def _load_checkpoint(self, device: torch.device) -> None:
        """Load pretrained sub-modules from a checkpoint holding a task state
        dict under ``'task'`` (keys ``model.<module>.*``; the pretraining
        flux projection is ``_flux_proj.*``)."""
        raw = torch.load(self.checkpoint_path, map_location=device)
        state = raw.get("task", raw) if isinstance(raw, dict) else raw

        def extract(prefix: str) -> Dict[str, Tensor]:
            out = {}
            for k, v in state.items():
                if k.startswith(f"model.{prefix}."):
                    out[k[len(prefix) + 7:]] = v
                elif k.startswith(f"{prefix}."):
                    out[k[len(prefix) + 1:]] = v
            return out

        for name in self.checkpoint_load_modules:
            if name == "gnn":
                sub = extract("gnn_backbone")
                if not self.use_film:
                    sub = {k: v for k, v in sub.items() if not k.startswith("film_generator.")}
                if sub:
                    self.gnn_backbone.load_state_dict(sub, strict=self.checkpoint_strict)
            elif name == "adapter":
                sub = extract("adapter")
                if sub:
                    self.adapter.load_state_dict(sub, strict=self.checkpoint_strict)
            elif name == "metabolites_encoder":
                sub = extract("metabolites_encoder")
                if sub:
                    self.metabolites_encoder.load_state_dict(sub, strict=self.checkpoint_strict)
            elif name == "flux_encoder":
                pass  # parameter-free
            elif name == "flux_proj":
                sub = extract("_flux_proj")
                if sub and self.flux_proj is not None:
                    self.flux_proj.load_state_dict(sub, strict=self.checkpoint_strict)
            else:
                raise ValueError(f"Unknown checkpoint module {name!r}")
        print(f"[MetabolicGNN] loaded {self.checkpoint_load_modules} from {self.checkpoint_path}", flush=True)

    # -- freeze schedule ------------------------------------------------------
    def _freeze_targets(self) -> List[nn.Module]:
        mods = [getattr(self, _FREEZE_ATTR.get(n, n), None) for n in self.freeze_module_names]
        return [m for m in mods if isinstance(m, nn.Module)]

    def freeze_pretrained_modules(self) -> int:
        if self._modules_frozen:
            return 0
        n = 0
        for mod in self._freeze_targets():
            for p in mod.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n += p.numel()
        self._modules_frozen = True
        return n

    def unfreeze_all_modules(self) -> int:
        if not self._modules_frozen:
            return 0
        n = 0
        for mod in self._freeze_targets():
            for p in mod.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    n += p.numel()
        self._modules_frozen = False
        return n

    # -- forward ----------------------------------------------------------------
    def _backbone(self, x: Tensor, edge_index: Tensor, film: Optional[Tensor]) -> Tensor:
        film = film if self.use_film else None
        if self.gnn_grad_checkpoint and torch.is_grad_enabled():
            # non-reentrant checkpointing keeps the RNG state, so dropout masks match on recompute
            return torch.utils.checkpoint.checkpoint(
                lambda xx, ff: self.gnn_backbone(xx, edge_index, film_condition=ff), x, film, use_reentrant=False)
        return self.gnn_backbone(x, edge_index, film_condition=film)

    def forward(self, batch: Dict[str, Any], query_times: Tensor, initial_condition: Tensor,
                sensor_stats: Tensor) -> Dict[str, Tensor]:
        """Returns readout name -> predicted rates [B, T]."""
        self._ensure_built(batch)
        B = batch["batch_size"]
        device = batch["node_feature_base"].device
        meta = batch["meta"]
        rna = batch["rna"]
        rna_times = rna["rna_times"]
        rna_gene_counts = rna["rna_gene_counts"]
        rna_mask = rna["rna_mask"]
        reaction_gpr = batch.get("reaction_gpr_gene_indices", [])
        K = rna_times.shape[1]
        m = len(meta.get("reaction_ids", []))
        N = 2 * m

        z_met_nodes = self.reaction_to_nodes(self.metabolites_encoder(meta, device))
        ref = rna["gpr_gene_raw_mean"]
        if ref.dim() > 1:
            ref = ref[0]

        node_feats, film_conds = [], []
        for b in range(B):
            phases, conds = [], []
            for k in range(K):
                if not rna_mask[b, k]:
                    phases.append(torch.zeros(N, self.d_model, dtype=torch.float32, device=device))
                    conds.append(torch.zeros(self.film_dim, dtype=torch.float32, device=device))
                    continue
                if self.gene_encoder.mode == "rel_mean":
                    counts_k = rna["gpr_gene_ratio"][b, k] * ref
                    z_gene = self.gene_encoder(counts_k, reaction_gpr, device, ref_counts=ref)
                else:
                    z_gene = self.gene_encoder(rna["gpr_gene_counts"][b, k], reaction_gpr, device)
                z = self.adapter(None, z_met_nodes, self.reaction_to_nodes(z_gene), m_flux=False, m_gene=True)
                if self.flux_proj is not None:
                    v_col = z.new_full((z.shape[0], 1), self.flux_proj_value)
                    m_col = z.new_full((z.shape[0], 1), self.flux_proj_mask_value)
                    z = self.flux_proj(torch.cat([z, v_col, m_col], dim=-1))
                if self.use_film:
                    conds.append(self.rna_projection(rna_gene_counts[b, k]))
                else:
                    conds.append(torch.zeros(self.film_dim, dtype=torch.float32, device=device))
                phases.append(z)
            node_feats.append(torch.stack(phases, dim=0))
            film_conds.append(torch.stack(conds, dim=0))
        x = torch.stack(node_feats, dim=0).reshape(B * K, N, self.d_model)
        film = torch.stack(film_conds, dim=0).reshape(B * K, self.film_dim)

        edge_index = batch["edge_index"]
        chunk = self.gnn_phase_chunk_size
        if chunk is None or chunk >= B * K:
            h_flat = self._backbone(x, edge_index, film)
        else:
            # the B*K snapshots are independent over the shared graph; chunking only saves memory
            h_flat = torch.cat([self._backbone(x[s:s + chunk], edge_index, film[s:s + chunk])
                                for s in range(0, B * K, chunk)], dim=0)
        h_phases = h_flat.reshape(B, K, N, self.d_model)
        h_query = torch.stack([self.kernel_gating(h_phases[b], rna_times[b], query_times[b], rna_mask[b])
                               for b in range(B)], dim=0)  # [B, T, N, d]

        T = query_times.shape[1]
        h_global = h_query.mean(dim=2)
        z_expr = None
        if self.expression_adapter is not None:
            z_expr = self.expression_adapter(rna[self._expr_input], rna_times, query_times, rna_mask)
        sensor_flat = sensor_stats.reshape(B, T, -1)
        init_exp = initial_condition.unsqueeze(1).expand(-1, T, -1)

        pooled = []
        for name in self.readout_names:
            idx: List[int] = []
            for rid in self.readout_reaction_ids[name]:
                idx.extend(self._reaction_to_node_indices.get(rid, []))
            if not idx:
                raise ValueError(f"No graph nodes for readout {name} ({self.readout_reaction_ids[name]})")
            pooled.append(h_query[:, :, idx, :].mean(dim=2))
        readout_concat = torch.cat(pooled, dim=-1)
        h0_features = torch.cat([h_global[:, 0, :], initial_condition], dim=-1)

        if self.biomass_head_module is not None:
            bio_in = [h_global, sensor_flat, init_exp] + ([z_expr] if z_expr is not None else [])
            biomass_rate = self.biomass_head_module(torch.cat(bio_in, dim=-1), query_times)
            parts = [readout_concat, h_global, sensor_flat, init_exp, biomass_rate.detach().unsqueeze(-1)]
            if z_expr is not None:
                parts.append(z_expr)
            predictions = self.readout_gru(torch.cat(parts, dim=-1), h0_features=h0_features)
            predictions["biomass_rate"] = biomass_rate
        else:
            parts = [readout_concat, h_global, sensor_flat, init_exp] + ([z_expr] if z_expr is not None else [])
            predictions = self.readout_gru(torch.cat(parts, dim=-1), h0_features=h0_features)
            if "biomass_rate" in predictions and self.lag_phase_gating is not None:
                predictions["biomass_rate"] = self.lag_phase_gating(predictions["biomass_rate"], query_times)
        return predictions
