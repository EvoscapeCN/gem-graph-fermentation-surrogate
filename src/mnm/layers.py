"""Building blocks: modality encoders, fusion adapter, FiLM, temporal kernel,
and the batched GATv2 / per-node-MLP graph encoders."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax

from .registry import Registry


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------

class FluxModalityEncoder(nn.Module):
    """Flux-distribution statistics per half-node (identity map).

    Reads ``meta['flux_statistics_scaled']`` (``{stat: [2m]}``) and stacks the
    statistics in sorted-key order into a ``[2m, n_stats]`` tensor. Only the
    pretraining dataset provides these statistics; at finetuning the flux
    modality is replaced by a learned missing-modality embedding and only its
    width (``meta['flux_stat_dim']``) is needed.
    """

    def forward(self, meta: Dict[str, Any], device: torch.device) -> Tensor:
        stats = meta.get("flux_statistics_scaled")
        if stats is None:
            raise KeyError("flux_statistics_scaled not found in meta")
        keys = sorted(stats.keys())
        return torch.tensor([stats[k] for k in keys], dtype=torch.float32, device=device).T

    @staticmethod
    def output_dim(meta: Dict[str, Any]) -> int:
        stats = meta.get("flux_statistics_scaled")
        if stats is not None:
            return len(stats)
        return int(meta["flux_stat_dim"])


class MetabolitesModalityEncoder(nn.Module):
    """Stoichiometry-weighted mean of learnable metabolite embeddings per reaction."""

    def __init__(self, embedding_dim: int = 300, mode: str = "mean", use_stoichiometry_weight: bool = True):
        super().__init__()
        assert mode in {"sum", "mean"}
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.use_stoichiometry_weight = bool(use_stoichiometry_weight)
        self.embedding_bag: Optional[nn.EmbeddingBag] = None

    def _ensure_built(self, num_embeddings: int, device: torch.device) -> None:
        if self.embedding_bag is None or self.embedding_bag.num_embeddings != num_embeddings:
            # mode='sum' is required for per_sample_weights; the mean is taken below
            self.embedding_bag = nn.EmbeddingBag(num_embeddings, self.embedding_dim, mode="sum").to(device)

    def forward(self, meta: Dict[str, Any], device: torch.device) -> Tensor:
        vocab = meta["metabolites_vocab"]
        ohe_indices = meta["node_metabolites_ohe_indices"]
        weights_list = meta.get("node_metabolites_weights")
        m = len(ohe_indices) // 2  # both half-nodes of a reaction share its metabolites
        flat_idx: List[int] = []
        flat_w: List[float] = []
        offsets = [0]
        for i, idx_list in enumerate(ohe_indices[:m]):
            if idx_list:
                flat_idx.extend(idx_list)
                if self.use_stoichiometry_weight and weights_list is not None and i < len(weights_list):
                    flat_w.extend(weights_list[i][: len(idx_list)])
                else:
                    flat_w.extend([1.0] * len(idx_list))
            offsets.append(len(flat_idx))
        self._ensure_built(len(vocab), device)
        if not flat_idx:
            return torch.zeros(m, self.embedding_dim, dtype=torch.float32, device=device)
        idx_t = torch.tensor(flat_idx, dtype=torch.long, device=device)
        off_t = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        psw = None
        if self.use_stoichiometry_weight and flat_w:
            psw = torch.tensor(flat_w, dtype=torch.float32, device=device)
        embeds = self.embedding_bag(idx_t, off_t, per_sample_weights=psw)
        if self.mode == "mean":
            if psw is not None:
                sums = []
                for i in range(len(offsets) - 1):
                    s, e = offsets[i], offsets[i + 1]
                    if s < e:
                        w = sum(flat_w[s:e])
                        sums.append(w if w > 0 else 1.0)
                    else:
                        sums.append(1.0)
            else:
                sums = [max(offsets[i + 1] - offsets[i], 1) for i in range(len(offsets) - 1)]
            embeds = embeds / torch.tensor(sums, dtype=torch.float32, device=device).unsqueeze(1)
        return embeds


class GeneCountsModalityEncoder(nn.Module):
    """GPR-routed gene-expression encoder: per reaction, an expression-weighted
    mixture of learnable gene embeddings.

    ``mode='rel_mean'`` (paper): weights are raw-scale expression
    ``rho_j * mean_j`` and the divisor is the summed training-mean expression
    of the reaction's genes. ``mode='mean'`` divides by the summed weights.
    Reactions without a GPR rule get a learnable fallback vector.
    """

    def __init__(self, embedding_dim: int = 128, mode: str = "rel_mean"):
        super().__init__()
        assert mode in {"mean", "rel_mean"}
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.gene_embedding: Optional[nn.Embedding] = None
        self.no_gpr_embedding: Optional[nn.Parameter] = None
        self.embedding_bag: Optional[nn.EmbeddingBag] = None

    def _ensure_built(self, num_genes: int, device: torch.device) -> None:
        if self.gene_embedding is None or self.gene_embedding.num_embeddings != num_genes:
            self.gene_embedding = nn.Embedding(num_genes, self.embedding_dim)
            self.no_gpr_embedding = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.float32) * 0.01)
            self.embedding_bag = nn.EmbeddingBag(num_genes, self.embedding_dim, mode="sum")
            self.embedding_bag.weight = self.gene_embedding.weight  # shared table
            self.gene_embedding.to(device)
            self.embedding_bag.to(device)
            self.no_gpr_embedding.data = self.no_gpr_embedding.data.to(device)

    def forward(
        self,
        gene_counts: Tensor,
        reaction_gene_indices: List[List[int]],
        device: torch.device,
        ref_counts: Optional[Tensor] = None,
    ) -> Tensor:
        G = gene_counts.shape[0]
        m = len(reaction_gene_indices)
        self._ensure_built(G, device)
        flat_idx: List[int] = []
        flat_w: List[float] = []
        offsets = [0]
        no_gpr = []
        for gene_indices in reaction_gene_indices:
            if not gene_indices:
                no_gpr.append(True)
            else:
                no_gpr.append(False)
                flat_idx.extend(gene_indices)
                for g in gene_indices:
                    flat_w.append(float(gene_counts[g].item()))
            offsets.append(len(flat_idx))
        if not flat_idx:
            return self.no_gpr_embedding.unsqueeze(0).expand(m, -1)
        idx_t = torch.tensor(flat_idx, dtype=torch.long, device=device)
        off_t = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        w_t = torch.tensor(flat_w, dtype=torch.float32, device=device)
        embeds = self.embedding_bag(idx_t, off_t, per_sample_weights=w_t)
        if self.mode == "mean":
            sums = []
            for i in range(len(offsets) - 1):
                s, e = offsets[i], offsets[i + 1]
                if s < e:
                    w = sum(flat_w[s:e])
                    sums.append(w if w > 0 else 1.0)
                else:
                    sums.append(1.0)
            embeds = embeds / torch.tensor(sums, dtype=torch.float32, device=device).unsqueeze(1)
        else:  # rel_mean
            ref = ref_counts.detach().to("cpu", torch.float32).tolist()
            denoms = []
            for i in range(len(offsets) - 1):
                d = sum(ref[g] for g in flat_idx[offsets[i]:offsets[i + 1]])
                denoms.append(d if d > 0 else 1.0)
            embeds = embeds / torch.tensor(denoms, dtype=torch.float32, device=device).unsqueeze(1)
        no_gpr_t = torch.tensor(no_gpr, dtype=torch.bool, device=device)
        if no_gpr_t.any():
            embeds[no_gpr_t] = self.no_gpr_embedding.unsqueeze(0).expand(int(no_gpr_t.sum()), -1)
        return embeds


class ModalityAdapter(nn.Module):
    """Concatenate [flux, metabolites, gene] per node and project to ``out_dim``.

    A missing modality is replaced by a learnable embedding of matching width,
    so the same weights serve pretraining (no genes) and finetuning (no flux).
    """

    def __init__(self, flux_dim: int, metabolites_dim: int, gene_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.e_missing_flux = nn.Parameter(torch.randn(flux_dim) * 0.01)
        self.e_missing_gene = nn.Parameter(torch.randn(gene_dim) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(flux_dim + metabolites_dim + gene_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_flux: Optional[Tensor], z_met: Tensor, z_gene: Optional[Tensor],
                m_flux: bool = True, m_gene: bool = True) -> Tensor:
        n = z_met.shape[0]
        device = z_met.device
        if not m_flux or z_flux is None:
            z_flux = self.e_missing_flux.unsqueeze(0).expand(n, -1).to(device)
        if not m_gene or z_gene is None:
            z_gene = self.e_missing_gene.unsqueeze(0).expand(n, -1).to(device)
        return self.mlp(torch.cat([z_flux, z_met, z_gene], dim=-1))


class FiLMGenerator(nn.Module):
    """(gamma, beta) from a conditioning vector, initialised to identity."""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: Tensor) -> Tuple[Tensor, Tensor]:
        gamma, beta = self.mlp(c).chunk(2, dim=-1)
        return gamma + 1.0, beta


def _causal_admissible(admissible: Tensor, t_k: Tensor, t_q: Tensor, mask: Tensor) -> Tensor:
    """Keep snapshots with t_k <= t; query times before the first valid
    snapshot fall back to the earliest valid one."""
    admissible = admissible & (t_k.unsqueeze(0) <= t_q.unsqueeze(1))
    empty_rows = ~admissible.any(dim=1)
    if bool(empty_rows.any()):
        earliest = int(torch.argmin(t_k.masked_fill(~mask, float("inf"))).item())
        fallback = torch.zeros_like(admissible)
        fallback[:, earliest] = True
        admissible = torch.where(empty_rows.unsqueeze(1), fallback, admissible)
    return admissible


class KernelGating(nn.Module):
    """Gaussian temporal kernel over the K RNA snapshots of a run:
    ``w_k(t) = softmax_k(-gamma (t - t_k)^2)``, gamma = softplus(raw) > 0.
    With ``causal_mask`` only snapshots at or before t contribute."""

    def __init__(self, gamma_initial: float = 0.01, causal_mask: bool = False):
        super().__init__()
        self.gamma_raw = nn.Parameter(torch.tensor(float(math.log(math.exp(gamma_initial) - 1.0))))
        self.eps = 1e-6
        self.causal_mask = bool(causal_mask)

    def forward(self, h_phases: Tensor, rna_times: Tensor, query_times: Tensor, rna_mask: Tensor) -> Tensor:
        """h_phases [K, N, D], rna_times [K], query_times [T], rna_mask [K] -> [T, N, D]."""
        T = query_times.shape[0]
        gamma = F.softplus(self.gamma_raw) + self.eps
        t_query = query_times.unsqueeze(1)
        t_rna = rna_times.unsqueeze(0)
        logits = -gamma * (t_query - t_rna) ** 2
        admissible = rna_mask.unsqueeze(0).expand(T, -1)
        if self.causal_mask:
            admissible = _causal_admissible(admissible, rna_times, query_times, rna_mask)
        weights = torch.softmax(logits.masked_fill(~admissible, float("-inf")), dim=1)
        return torch.einsum("tk,knd->tnd", weights, h_phases)


# ---------------------------------------------------------------------------
# Graph encoders (inputs [B, N, F] over one shared static graph)
# ---------------------------------------------------------------------------

def _batched_group_softmax(scores: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    """Softmax of [B, H, E] attention scores grouped by (batch, head, dst)."""
    if scores.numel() == 0:
        return scores
    B, H, E = scores.shape
    dst = dst.to(torch.long)
    index = dst.view(1, 1, E).expand(B, H, E)
    offsets = torch.arange(B * H, device=scores.device, dtype=dst.dtype).view(B, H, 1) * num_nodes
    flat_index = (index + offsets).reshape(-1)
    return softmax(scores.reshape(-1), flat_index, num_nodes=B * H * num_nodes).view(B, H, E)


class GATv2ConvBatched(nn.Module):
    """GATv2 convolution for a batch of node-feature matrices on one graph.

    Messages flow ``edge_index[0] -> edge_index[1]``; heads are averaged.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 dropout: float = 0.0, add_self_loops: bool = True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = float(dropout)
        self.add_self_loops = bool(add_self_loops)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.att = nn.Parameter(torch.randn(heads, out_channels))
        self.negative_slope = 0.2
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        B, N, _ = x.shape
        edge_index = edge_index.to(x.device)
        if self.add_self_loops:
            loops = torch.arange(N, device=x.device, dtype=edge_index.dtype)
            edge_index = torch.cat([edge_index, torch.stack([loops, loops], dim=0)], dim=1)
        proj = self.lin(x).view(B, N, self.heads, self.out_channels)
        src_feat = proj[:, edge_index[0]]
        dst_feat = proj[:, edge_index[1]]
        alpha = F.leaky_relu(dst_feat + src_feat, negative_slope=self.negative_slope)
        alpha = (alpha * self.att.view(1, 1, self.heads, self.out_channels)).sum(dim=-1)  # [B, E, H]
        dst = edge_index[1]
        alpha = _batched_group_softmax(alpha.permute(0, 2, 1), dst, N).permute(0, 2, 1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = src_feat * alpha.unsqueeze(-1)
        out = x.new_zeros(B, N, self.heads, self.out_channels)
        scatter_index = dst.view(1, -1, 1, 1).expand(B, -1, self.heads, self.out_channels)
        out.scatter_add_(1, scatter_index, msg.to(out.dtype))  # AMP may promote alpha to fp32
        return out.mean(dim=2) + self.bias.view(1, 1, -1)


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU()}[name]


def _norm(name: str, dim: int) -> nn.Module:
    return {"layer": nn.LayerNorm(dim), "none": nn.Identity()}[name]


class _EncoderBase(nn.Module):
    """Shared FiLM plumbing; the model rebuilds the encoder from ``init_kwargs``
    with ``in_dim``/``out_dim`` set to the model width."""

    film_generator: Optional[nn.Module]
    enable_film: bool

    def _film(self, film_condition: Optional[Tensor]):
        if self.enable_film and film_condition is not None and self.film_generator is not None:
            return self.film_generator(film_condition)
        return None, None


@Registry.register("model.gat_encoder")
class GATEncoderBatched(_EncoderBase):
    """Stack of GATv2 layers, each followed by norm, activation, optional FiLM
    (``gamma * h + beta``) and dropout; a final linear maps to ``out_dim``."""

    def __init__(self, in_dim: Optional[int] = None, hidden_dim: int = 400, num_layers: int = 3, heads: int = 2,
                 attn_dropout: float = 0.1, feat_dropout: float = 0.1, activation: str = "relu",
                 norm: str = "layer", add_self_loops: bool = True, out_dim: Optional[int] = None):
        super().__init__()
        self.init_kwargs = dict(hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, attn_dropout=attn_dropout,
                                feat_dropout=feat_dropout, activation=activation, norm=norm,
                                add_self_loops=add_self_loops)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.attn_dropout = float(attn_dropout)
        self.activation_name = activation.lower()
        self.norm_type = norm.lower()
        self.add_self_loops = bool(add_self_loops)
        self.out_dim = int(out_dim) if out_dim is not None else None
        self.enable_film = False
        self.film_generator = None
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropout = nn.Dropout(float(feat_dropout))
        self.final_proj: Optional[nn.Linear] = None
        if in_dim is not None:
            self._build(int(in_dim))

    def _build(self, in_dim: int) -> None:
        last = in_dim
        for _ in range(self.num_layers):
            self.layers.append(GATv2ConvBatched(last, self.hidden_dim, heads=self.heads, dropout=self.attn_dropout,
                                                add_self_loops=self.add_self_loops))
            self.norms.append(_norm(self.norm_type, self.hidden_dim))
            self.acts.append(_activation(self.activation_name))
            last = self.hidden_dim
        if self.out_dim is not None and last != self.out_dim:
            self.final_proj = nn.Linear(last, self.out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, film_condition: Optional[Tensor] = None) -> Tensor:
        gamma, beta = self._film(film_condition)
        h = x
        for conv, norm, act in zip(self.layers, self.norms, self.acts):
            y = act(norm(conv(h, edge_index)))
            if gamma is not None:
                y = gamma.unsqueeze(1) * y + beta.unsqueeze(1)
            h = self.dropout(y)
        if self.final_proj is not None:
            h = self.final_proj(h)
        return h


@Registry.register("model.node_mlp_encoder")
class NodeMLPEncoderBatched(_EncoderBase):
    """Per-node MLP with the GAT encoder's interface and no message passing
    (the -GNN ablation)."""

    def __init__(self, in_dim: Optional[int] = None, hidden_dim: int = 570, num_layers: int = 3,
                 feat_dropout: float = 0.1, activation: str = "relu", norm: str = "layer",
                 out_dim: Optional[int] = None):
        super().__init__()
        self.init_kwargs = dict(hidden_dim=hidden_dim, num_layers=num_layers, feat_dropout=feat_dropout,
                                activation=activation, norm=norm)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation_name = activation.lower()
        self.norm_type = norm.lower()
        self.out_dim = int(out_dim) if out_dim is not None else None
        self.enable_film = False
        self.film_generator = None
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropout = nn.Dropout(float(feat_dropout))
        self.final_proj: Optional[nn.Linear] = None
        if in_dim is not None:
            self._build(int(in_dim))

    def _build(self, in_dim: int) -> None:
        last = in_dim
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(last, self.hidden_dim))
            self.norms.append(_norm(self.norm_type, self.hidden_dim))
            self.acts.append(_activation(self.activation_name))
            last = self.hidden_dim
        if self.out_dim is not None and last != self.out_dim:
            self.final_proj = nn.Linear(last, self.out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, film_condition: Optional[Tensor] = None) -> Tensor:
        gamma, beta = self._film(film_condition)
        h = x
        for linear, norm, act in zip(self.layers, self.norms, self.acts):
            y = act(norm(linear(h)))
            if gamma is not None:
                y = gamma.unsqueeze(1) * y + beta.unsqueeze(1)
            h = self.dropout(y)
        if self.final_proj is not None:
            h = self.final_proj(h)
        return h
