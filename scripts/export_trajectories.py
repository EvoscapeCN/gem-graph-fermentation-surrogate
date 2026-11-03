#!/usr/bin/env python
"""Held-out cumulative trajectories of a method (appendix trajectory figure).

Every run is a test run in exactly one fold; it gets a prediction from each
seed's checkpoint of that fold. Per run the across-seed median and SD of the
integrated cumulative trajectories are stored. Three runs are then selected by
MEASURED final astaxanthin only (10th / 50th / 90th percentile), never by fit.

    python scripts/export_trajectories.py --results outputs/experiments/main_comparison/results.json \
        --yaml configs/experiments/main_comparison.yaml --base main --out outputs/figures/trajectories
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_experiment import build_eval_config, fold_config, load_method_config, predict_rates  # noqa: E402
from mnm.evaluation import (EvaluationConfig, _to_numpy, adjacent_biomass_average, create_mono_mask,  # noqa: E402
                            find_start_idx, integrate_from_zero, integrate_rates)

READOUT_TO_KEY = {"biomass_rate": "biomass", "o2_consumption_rate": "o2", "sugar_consumption_rate": "sugar",
                  "carotenoids_rate": "carotenoids", "astaxanthin_rate": "astaxanthin"}


def cumulative_trajectories(sample: dict, rates: Dict[str, np.ndarray], cfg: EvaluationConfig,
                            names: List[str]) -> Dict[str, np.ndarray]:
    """Integrate a run's predicted rates exactly as the evaluation does."""
    times = _to_numpy(sample["times"]).astype(np.float64).flatten()
    assay = _to_numpy(sample["assay_metrics"]).astype(np.float64)
    T = len(times)
    bio_meas = assay[:, names.index(cfg.biomass_target)]
    bio_pred = integrate_from_zero(rates[cfg.biomass_readout].astype(np.float64), times, cfg.rate_clip_max) \
        if cfg.biomass_readout in rates else bio_meas.copy()
    out = {"times": times}
    for readout, key in READOUT_TO_KEY.items():
        target_name = cfg.readout_to_target_mapping.get(readout)
        if target_name is None or target_name not in names:
            continue
        target = assay[:, names.index(target_name)]
        masked = target.copy()
        masked[~create_mono_mask(target, cfg.monotonicity_tolerance)] = np.nan
        out[f"{key}_meas"] = masked
        if readout == cfg.biomass_readout:
            out[f"{key}_pred"] = bio_pred
            continue
        if readout not in rates:
            out[f"{key}_pred"] = np.full(T, np.nan)
            continue
        r = rates[readout].astype(np.float64)
        s = find_start_idx(target)
        if readout in cfg.specific_rate_readouts:
            r = r.copy()
            r[:-1] = r[:-1] * adjacent_biomass_average(bio_meas, bio_pred, cfg.specific_rate_use_measured_biomass)
        out[f"{key}_pred"] = integrate_rates(r, times, float(target[s]) if s < T else 0.0, s, cfg.rate_clip_max)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--base", default="main")
    ap.add_argument("--out", default="outputs/figures/trajectories")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    res_all = json.load(open(args.results))
    res = res_all["methods"]
    man = yaml.safe_load(open(args.yaml))
    shared = man["shared"]
    exp = man[[k for k in man if k != "shared"][0]]
    ecfg = build_eval_config(shared, exp)
    seeds = sorted(int(k.rsplit("_seed", 1)[1]) for k in res if k.rsplit("_seed", 1)[0] == args.base)
    method_cfg = exp["methods"][f"{args.base}_seed{seeds[0]}"]
    overrides = dict((kv.split("=", 1)[0], yaml.safe_load(kv.split("=", 1)[1])) for kv in args.set)
    base_cfg = load_method_config(method_cfg, overrides)
    folds = sorted(int(f[4:]) for f in res[f"{args.base}_seed{seeds[0]}"]["fold_results"])
    n_folds = res_all.get("n_folds", shared["n_folds"])
    per: Dict[int, Dict[int, dict]] = {}
    for fold in folds:
        cfg = fold_config(base_cfg, fold, n_folds, method_cfg.get("config_overrides"))
        for s in seeds:
            ck = res[f"{args.base}_seed{s}"]["fold_results"][f"fold{fold}"]["best_checkpoint"]
            rates, samples, split, names = predict_rates(cfg, ck)
            for si in split["test"]:
                per.setdefault(si, {})[s] = cumulative_trajectories(samples[si], rates.get(si, {}), ecfg, names)
    runs = sorted(per)
    maxT = max(len(per[si][seeds[0]]["times"]) for si in runs)
    B = {"times": np.full((len(runs), maxT), np.nan, np.float32), "sample_ids": np.array(runs, np.int32)}
    for key in READOUT_TO_KEY.values():
        for suf in ("measured", "pred", "pred_std"):
            B[f"{key}_{suf}"] = np.full((len(runs), maxT), np.nan, np.float32)
    for row, si in enumerate(runs):
        first = per[si][seeds[0]]
        T = len(first["times"])
        B["times"][row, :T] = first["times"]
        for key in READOUT_TO_KEY.values():
            if f"{key}_meas" not in first:
                continue
            B[f"{key}_measured"][row, :T] = first[f"{key}_meas"]
            P = np.stack([per[si][s][f"{key}_pred"] for s in seeds])
            B[f"{key}_pred"][row, :T] = np.nanmedian(P, 0)
            B[f"{key}_pred_std"][row, :T] = np.nanstd(P, 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out + "_all.npz", **B)
    final = np.array([np.nanmax(B["astaxanthin_measured"][r]) if np.isfinite(B["astaxanthin_measured"][r]).any()
                      else np.nan for r in range(len(runs))])
    ok = np.where(np.isfinite(final))[0]
    order = ok[np.argsort(final[ok])]
    pick = [order[int(round(q * (len(order) - 1)))] for q in (0.1, 0.5, 0.9)]
    np.savez(args.out + "_selected.npz",
             **{k: (v[pick] if isinstance(v, np.ndarray) and v.shape[:1] == (len(runs),) else v) for k, v in B.items()})
    print("selected runs (10th/50th/90th percentile of measured final astaxanthin):", [int(runs[p]) for p in pick])


if __name__ == "__main__":
    main()
