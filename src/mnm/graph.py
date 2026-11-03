"""Mass Flow Graph (MFG) construction from an SBML genome-scale model.

Each reaction ``r`` contributes a forward and a reverse half-node, laid out in
two blocks ``[fwd R1..Rm, rev R1..Rm]`` (``|V| = 2m``). A directed edge
``a -> b`` exists when half-node ``a`` produces a metabolite that half-node
``b`` consumes (Beguerisse-Diaz et al. 2018). Edges are computed from the
stoichiometry alone (unit flux on every admissible half-node).

The module also loads the FBA flux corpus used for pretraining and computes
the per-half-node distributional statistics consumed by the flux encoder.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cobra
import lmoments3 as lm
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.stattools import medcouple

#: Order in which the per-half-node flux statistics are computed. The flux
#: encoder consumes them in sorted-key order, giving a 10-dim feature.
FLUX_STAT_KEYS = [
    "percentage_of_zero",
    "median",
    "MAD",
    "IQR",
    "medcouple",
    "trimmed_kurtosis",
    "L1",
    "L2",
    "L3_L2",
    "L4_L2",
]

# Columns of a flux corpus file that are not reactions.
CORPUS_NON_REACTION_COLUMNS = ("time", "near_opt_frac")


_SBML_CACHE: Dict[str, cobra.Model] = {}


def read_sbml(sbml_path: str) -> cobra.Model:
    """Parse an SBML model once per process (callers must not modify it)."""
    key = os.path.abspath(sbml_path)
    if key not in _SBML_CACHE:
        logging.getLogger("cobra").setLevel(logging.ERROR)
        _SBML_CACHE[key] = cobra.io.read_sbml_model(sbml_path)
    return _SBML_CACHE[key]


def extract_gpr_from_sbml(sbml_path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """GPR gene vocabulary (sorted) and reaction id -> gene indices.

    Boolean AND/OR structure is flattened: a reaction maps to every gene that
    appears in its rule.
    """
    model = read_sbml(sbml_path)
    gene_vocab = sorted(g.id for g in model.genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_vocab)}
    reaction_to_genes: Dict[str, List[int]] = {}
    for rxn in model.reactions:
        genes = sorted(g.id for g in rxn.genes)
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if idx:
            reaction_to_genes[rxn.id] = idx
    return gene_vocab, reaction_to_genes


def _stoichiometry(model: cobra.Model) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    S_df = cobra.util.array.create_stoichiometric_matrix(model, array_type="DataFrame")
    reaction_ids = list(S_df.columns)
    metabolite_ids = list(S_df.index)
    S = np.nan_to_num(S_df.values.astype(float))
    r = np.array([1 if rxn.reversibility else 0 for rxn in model.reactions], dtype=float)
    return S, reaction_ids, r, metabolite_ids


def _mfg_support_edges(S: np.ndarray, r: np.ndarray, edge_threshold: float = 1e-12) -> np.ndarray:
    """Non-zero pattern of the MFG adjacency at unit flux.

    Uses the 2m-unfolded stoichiometry ``S_2m = [S, -S] diag(1, rev)``; the MFG
    matrix is ``(S+ V)^T J^+ (S- V)`` with ``V = diag(v_2m)`` and
    ``J = diag(S+ v_2m)``. Returns ``(rows, cols)`` of entries above threshold.
    """
    S = np.nan_to_num(S)
    m = S.shape[1]
    A = np.block([[np.eye(m), np.zeros((m, m))], [np.zeros((m, m)), np.diag(r)]])
    S_2m = np.block([S, -S]).dot(A)
    S_plus = (np.abs(S_2m) + S_2m) / 2
    S_minus = (np.abs(S_2m) - S_2m) / 2

    v_2m = np.vstack((np.ones(m), np.where(r > 0.5, 1.0, 0.0))).flatten()
    V = np.diag(v_2m)
    J_pinv = np.linalg.pinv(np.diag(S_plus.dot(v_2m)))
    M = (S_plus.dot(V)).T.dot(J_pinv).dot(S_minus.dot(V))
    return np.where(M > edge_threshold)


_GRAPH_CACHE: Dict[str, Dict[str, Any]] = {}


def build_mfg(sbml_path: str) -> Dict[str, Any]:
    """Build the static MFG of an SBML model (cached per process).

    Returns ``edge_index [2, E]``, ``num_nodes`` and ``meta`` with the reaction
    ids, the half-node layout and each half-node's metabolites with absolute
    stoichiometric coefficients (the input of the metabolite encoder).
    """
    key = os.path.abspath(sbml_path)
    if key in _GRAPH_CACHE:
        cached = _GRAPH_CACHE[key]
        return {
            "edge_index": cached["edge_index"].clone(),
            "num_nodes": cached["num_nodes"],
            "meta": {k: (list(v) if isinstance(v, list) else v) for k, v in cached["meta"].items()},
        }
    model = read_sbml(sbml_path)
    S, reaction_ids, r, metabolite_ids = _stoichiometry(model)
    m = S.shape[1]
    rows, cols = _mfg_support_edges(S, r)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)

    met_idx: List[List[int]] = []
    met_w: List[List[float]] = []
    for i in range(m):
        nz = np.where(S[:, i] != 0)[0]
        met_idx.append([int(x) for x in nz.tolist()])
        met_w.append([float(x) for x in np.abs(S[nz, i]).tolist()])

    meta = {
        "reaction_ids": reaction_ids,
        "half_node_reaction_ids": reaction_ids + reaction_ids,
        "half_node_dirs": ["fwd"] * m + ["rev"] * m,
        "reversibility": [int(x) for x in r.tolist()],
        "metabolites_vocab": metabolite_ids,
        "node_metabolites_ohe_indices": met_idx + met_idx,
        "node_metabolites_weights": met_w + met_w,
        "num_reactions": m,
        "num_metabolites": int(S.shape[0]),
        "flux_stat_dim": len(FLUX_STAT_KEYS),
    }
    graph = {"edge_index": edge_index, "num_nodes": 2 * m, "meta": meta}
    _GRAPH_CACHE[key] = graph
    return build_mfg(sbml_path)


# ---------------------------------------------------------------------------
# Flux corpus
# ---------------------------------------------------------------------------

def load_flux_corpus(
    flux_path: str,
    reaction_ids: List[str],
    min_near_opt_frac: Optional[float] = None,
    max_near_opt_frac: Optional[float] = None,
) -> pd.DataFrame:
    """Read a flux corpus (CSV or Parquet; one row per flux vector).

    Required columns: one per SBML reaction id. Optional: ``time`` (h, used
    for time stratification) and ``near_opt_frac`` (rows outside
    ``[min_near_opt_frac, max_near_opt_frac]`` are dropped). See
    ``docs/data_format.md``.
    """
    ext = os.path.splitext(flux_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(flux_path)
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(flux_path)
    else:
        raise ValueError(f"Unsupported flux corpus extension {ext!r} (use .csv or .parquet)")
    if "near_opt_frac" in df.columns and min_near_opt_frac is not None and max_near_opt_frac is not None:
        df = df[(df["near_opt_frac"] >= float(min_near_opt_frac)) & (df["near_opt_frac"] <= float(max_near_opt_frac))]
        if len(df) == 0:
            raise ValueError("No rows remaining after near_opt_frac filtering")
    missing = [rid for rid in reaction_ids if rid not in df.columns]
    if missing:
        raise ValueError(f"Flux corpus is missing reaction columns, e.g. {missing[:5]}")
    return df


def _robust_statistics(x: np.ndarray) -> Dict[str, float]:
    """Robust distributional statistics of one half-node's (log10) flux samples."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return {k: 0.0 for k in FLUX_STAT_KEYS if k != "percentage_of_zero"}

    median_val = float(np.median(x))
    MAD_val = float(1.4826 * np.median(np.abs(x - median_val)))
    q25, q75 = np.percentile(x, [25, 75])
    IQR_val = float(q75 - q25)
    medc_val = float(medcouple(x))

    trim = int(0.05 * len(x))
    xt = np.sort(x)[trim: len(x) - trim] if len(x) > 20 else x
    if xt.size < 4 or np.std(xt) < 1e-12 or np.unique(xt).size < 2:
        trimmed_kurtosis_val = 0.0
    else:
        tk = st.kurtosis(xt, fisher=True, bias=False)
        trimmed_kurtosis_val = float(tk) if np.isfinite(tk) else 0.0

    unique_count = int(np.unique(x).size)
    near_zero_spread = (abs(IQR_val) < 1e-12) or (np.std(x) < 1e-12)
    if unique_count < 2 or near_zero_spread:
        L1_val, L2_val, L3_L2_val, L4_L2_val = float(np.mean(x)), 0.0, 0.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmoms = lm.lmom_ratios(x)
        if isinstance(lmoms, dict):
            L1_val = float(lmoms.get("l1", np.mean(x)))
            L2_val = float(lmoms.get("l2", 0.0))
            L3_L2_val = float(lmoms.get("t3", 0.0))
            L4_L2_val = float(lmoms.get("t4", 0.0))
        else:
            seq = list(lmoms)
            L1_val = float(seq[0]) if len(seq) > 0 and np.isfinite(seq[0]) else float(np.mean(x))
            L2_val = float(seq[1]) if len(seq) > 1 and np.isfinite(seq[1]) else 0.0
            L3_L2_val = float(seq[2]) if len(seq) > 2 and np.isfinite(seq[2]) else 0.0
            L4_L2_val = float(seq[3]) if len(seq) > 3 and np.isfinite(seq[3]) else 0.0
        if (not np.isfinite(L2_val)) or L2_val < 1e-8:
            L3_L2_val, L4_L2_val = 0.0, 0.0
            if not np.isfinite(L2_val):
                L2_val = 0.0
        elif L2_val < 1e-6:
            scale = float(L2_val / (L2_val + 1e-6))
            L3_L2_val *= scale
            L4_L2_val *= scale

    result = {
        "median": median_val,
        "MAD": MAD_val,
        "IQR": IQR_val,
        "medcouple": medc_val,
        "trimmed_kurtosis": trimmed_kurtosis_val,
        "L1": L1_val,
        "L2": L2_val,
        "L3_L2": L3_L2_val,
        "L4_L2": L4_L2_val,
    }
    bad = [k for k, v in result.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"Non-finite statistics: {bad}")
    return result


def compute_flux_statistics(
    df: pd.DataFrame,
    reaction_ids: List[str],
    reversibility: List[int],
    eps: float = 1e-4,
) -> Dict[str, List[float]]:
    """Per-half-node flux statistics, robust-scaled across half-nodes.

    Forward half-node: statistics of log10(v + eps) over samples with
    ``v > eps``; reverse half-node: the same over ``|v|`` for ``v < -eps``.
    ``percentage_of_zero`` is the share of samples not in that direction.
    Irreversible reactions get an all-zero reverse half-node (zero share 1).
    The 2m x 10 matrix is then scaled column-wise with a RobustScaler.
    Returns ``{stat_name: [2m values]}`` in the ``[fwd..., rev...]`` layout.
    """
    keys = FLUX_STAT_KEYS
    zero_template = _robust_statistics(np.zeros(0))
    fwd_rows: List[List[float]] = []
    rev_rows: List[List[float]] = []
    rev_arr = np.asarray(reversibility, dtype=float)
    for idx, rid in enumerate(reaction_ids):
        samples = df[rid].to_numpy(dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            pos_vals = np.zeros(0)
            neg_abs = np.zeros(0)
            zero_count = 0
        else:
            pos_mask = samples > eps
            neg_mask = samples < -eps
            pos_vals = samples[pos_mask]
            neg_abs = np.abs(samples[neg_mask])
            zero_count = int(np.count_nonzero(~(pos_mask | neg_mask)))
        n_pos, n_neg = int(pos_vals.size), int(neg_abs.size)
        f_zero, r_zero = float(n_neg + zero_count), float(n_pos + zero_count)
        f_total, r_total = float(n_pos) + f_zero, float(n_neg) + r_zero
        f_pct = f_zero / f_total if f_total > 0 else 1.0
        r_pct = r_zero / r_total if r_total > 0 else 1.0

        f_stats = _robust_statistics(np.log10(pos_vals + float(eps)) if n_pos > 0 else np.zeros(0))
        if rev_arr[idx] > 0.5:
            r_stats = _robust_statistics(np.log10(neg_abs + float(eps)) if n_neg > 0 else np.zeros(0))
        else:
            r_pct = 1.0
            r_stats = dict(zero_template)
        f_row = {"percentage_of_zero": float(f_pct), **f_stats}
        r_row = {"percentage_of_zero": float(r_pct), **r_stats}
        fwd_rows.append([f_row[k] for k in keys])
        rev_rows.append([r_row[k] for k in keys])

    X = np.asarray(fwd_rows + rev_rows, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = float(np.nanmedian(col)) if np.any(~np.isfinite(col)) else float(np.median(col))
        col[~np.isfinite(col)] = med
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X)
    if not np.all(np.isfinite(X_scaled)):
        raise ValueError("Non-finite values after robust scaling")
    return {k: [float(v) for v in X_scaled[:, i]] for i, k in enumerate(keys)}
