"""Flat-ML baselines (Ridge, XGBoost): per-time-point regression of rates.

One row per (run, assay interval [t, t+1]): sensor window statistics, the
run-level initial conditions, the time t and the gene-expression vector of the
snapshot nearest to t. Targets are finite-difference rates of the cumulative
assays (divided by the interval's mean measured biomass for specific-rate
targets). Predicted rates are integrated by :mod:`mnm.evaluation` exactly like
the GNN's.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def find_nearest_rna_snapshot(query_time: float, rna_times: np.ndarray, rna_counts: np.ndarray,
                              rna_mask: np.ndarray) -> np.ndarray:
    valid = np.where(rna_mask)[0]
    if len(valid) == 0:
        return np.zeros(rna_counts.shape[1])
    return rna_counts[valid[np.argmin(np.abs(rna_times[valid] - query_time))]]


def build_feature_matrix(
    samples: List[Dict[str, Any]],
    sample_indices: List[int],
    assay_metric_names: List[str],
    target_readout_mapping: Dict[str, str],
    specific_rate: bool = False,
    biomass_column: str = "interpolated_biomass",
    use_gpr_genes: bool = False,
    gene_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Returns ``X [rows, F]``, ``{readout: target rates [rows]}`` and per-row
    ``{sample_idx, t_idx, time}``. ``use_gpr_genes`` uses the GPR genes only;
    ``gene_transform`` maps the ``[K, G]`` gene matrix first (e.g. onto the
    training-split principal components)."""
    X_rows = []
    targets: Dict[str, list] = {name: [] for name in target_readout_mapping}
    row_info = []
    specific = {r for r in target_readout_mapping if r not in ("biomass_rate", "growth_rate")} if specific_rate \
        else set()
    bio_col = assay_metric_names.index(biomass_column) if biomass_column in assay_metric_names else None

    for si in sample_indices:
        s = samples[si]
        times = s["times"].numpy()
        sensor = s["sensor_stats"].numpy()
        assay = s["assay_metrics"].numpy()
        rna = s["rna"]
        genes = rna["gpr_gene_counts" if use_gpr_genes else "rna_gene_counts"].numpy()
        if gene_transform is not None:
            genes = gene_transform(genes)
        init = s["initial_condition"].numpy()
        biomass = assay[:, bio_col] if (specific_rate and bio_col is not None) else None
        for t in range(len(times) - 1):
            dt = times[t + 1] - times[t]
            if dt <= 0:
                continue
            snap = find_nearest_rna_snapshot(times[t], rna["rna_times"].numpy(), genes, rna["rna_mask"].numpy())
            X_rows.append(np.concatenate([sensor[t].flatten(), init, [times[t]], snap]))
            avg_bio = 1.0
            if biomass is not None:
                bt = biomass[t] if not np.isnan(biomass[t]) else 0.0
                btn = biomass[t + 1] if not np.isnan(biomass[t + 1]) else 0.0
                avg_bio = (bt + btn) / 2.0 if (bt > 0 and btn > 0) else max(bt, btn, 1e-6)
            for readout, metric in target_readout_mapping.items():
                if metric not in assay_metric_names:
                    targets[readout].append(np.nan)
                    continue
                j = assay_metric_names.index(metric)
                a, b = assay[t, j], assay[t + 1, j]
                if np.isnan(a) or np.isnan(b):
                    targets[readout].append(np.nan)
                elif readout == "growth_rate":
                    targets[readout].append((np.log(b) - np.log(a)) / dt if (a > 1e-6 and b > 1e-6) else np.nan)
                else:
                    rate = (b - a) / dt
                    targets[readout].append(rate / avg_bio if readout in specific else rate)
            row_info.append({"sample_idx": si, "t_idx": t, "time": times[t]})
    X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, 0), dtype=np.float32)
    return X, {k: np.array(v, dtype=np.float32) for k, v in targets.items()}, row_info


def restructure_flat_to_per_sample(preds: Dict[str, np.ndarray], row_info: List[Dict[str, Any]],
                                   samples: List[Dict[str, Any]]) -> Dict[int, Dict[str, np.ndarray]]:
    """Flat row predictions -> ``{sample_idx: {readout: rates[T]}}`` (last entry 0)."""
    groups: Dict[int, List[int]] = {}
    for i, info in enumerate(row_info):
        groups.setdefault(info["sample_idx"], []).append(i)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for si, rows in groups.items():
        T = samples[si]["times"].shape[0]
        out[si] = {}
        for readout, vals in preds.items():
            rates = np.zeros(T, dtype=np.float64)
            for i in rows:
                v = float(vals[i])
                rates[row_info[i]["t_idx"]] = 0.0 if np.isnan(v) else v
            out[si][readout] = rates
    return out


XGB_DEFAULTS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def train_flat_model(model_type: str, X_train: np.ndarray, y_train: Dict[str, np.ndarray], X_val: np.ndarray,
                     X_test: np.ndarray, y_val: Dict[str, np.ndarray], readouts: List[str],
                     random_state: int = 42, xgb_params: Optional[Dict[str, Any]] = None):
    """Fit one regressor per readout; returns (val predictions, test predictions).

    Ridge: features standardised on the training rows; ``alpha`` is chosen
    from ``RIDGE_ALPHAS`` by the mean validation MSE over readouts.
    XGBoost: fixed ``XGB_DEFAULTS`` (overridable by ``xgb_params``).
    """
    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr, Xv, Xte = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
        best = None
        for alpha in RIDGE_ALPHAS:
            v, t = {}, {}
            total, n = 0.0, 0
            for r in readouts:
                y = y_train[r]
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    v[r], t[r] = np.zeros(len(Xv)), np.zeros(len(Xte))
                    continue
                m = Ridge(alpha=alpha).fit(Xtr[mask], y[mask])
                v[r], t[r] = m.predict(Xv), m.predict(Xte)
                vm = ~np.isnan(y_val[r])
                total += np.mean((v[r][vm] - y_val[r][vm]) ** 2) if vm.any() else 0
                n += 1
            score = total / max(n, 1)
            if best is None or score < best[0]:
                best = (score, v, t)
        return best[1], best[2]
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        v, t = {}, {}
        for r in readouts:
            y = y_train[r]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                v[r], t[r] = np.zeros(len(X_val)), np.zeros(len(X_test))
                continue
            m = XGBRegressor(**{**XGB_DEFAULTS, "random_state": random_state, "verbosity": 0, **(xgb_params or {})})
            m.fit(X_train[mask], y[mask])
            v[r], t[r] = m.predict(X_val), m.predict(X_test)
        return v, t
    raise ValueError(f"Unknown flat model {model_type!r}")
