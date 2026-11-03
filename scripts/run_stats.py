#!/usr/bin/env python
"""Run-level statistics for the paper's tables (fermentation run = unit of analysis).

For every method: seed-level composite mean +- SD and per-readout means
(cross-fold means of the per-run test RMSE). For every method against the main
model: seeds are averaged within each run, runs are paired across methods (a
run's composite sums the readouts it has; missing readouts are skipped), and a
cluster bootstrap over runs (10,000 resamples) gives the 95% CI of the mean
difference; Cohen's d, the share of runs favouring the main model and a
power analysis are reported too.

    python scripts/run_stats.py --results-dir outputs/experiments --out outputs/stats.json [--macros outputs/macros.tex]

Methods whose results are missing are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

READOUTS = ["biomass_rate", "o2_consumption_rate", "sugar_consumption_rate", "carotenoids_rate", "astaxanthin_rate"]

# name -> (experiment directory, method base name in that experiment)
METHODS = {
    "ours": ("main_comparison", "main"),
    "gnn": ("main_comparison", "gnn"),
    "ridge": ("main_comparison", "ridge"),
    "xgb": ("main_comparison", "xgb"),
    "ridge_gpr": ("input_matched_baselines", "ridge_gpr"),
    "xgb_gpr": ("input_matched_baselines", "xgb_gpr"),
    "ridge_pca16": ("input_matched_baselines", "ridge_pca16"),
    "xgb_pca16": ("input_matched_baselines", "xgb_pca16"),
    "no_pretrain": ("ablations", "no_pretrain"),
    "no_expr": ("ablations", "no_expr"),
    "no_pretrain_no_expr": ("ablations", "no_pretrain_no_expr"),
    "no_freeze": ("ablations", "no_freeze"),
    "no_gnn": ("ablations", "no_gnn"),
    "ctl_degperm": ("controls", "edge_shuffle"),
    "ctl_randedge": ("controls", "random_edges"),
    "ctl_gprshuffle": ("controls", "gpr_shuffle"),
    "ctl_metshuffle": ("controls", "metabolite_shuffle"),
    "ctl_bothpaths": ("controls", "no_prior_both_paths"),
    "film": ("appendix", "with_film"),
    "reuse": ("appendix", "reuse_flux_proj"),
    "causal": ("appendix", "causal_kernel"),
}


def long_form(results: dict, method_base: str, seeds: List[int]) -> pd.DataFrame:
    """One row per (run, seed, fold) with per-readout test RMSE and the composite
    (sum over the readouts the run has; NaN readouts are skipped, as in evaluation)."""
    rows = []
    for seed in seeds:
        for fold, fr in results["methods"][f"{method_base}_seed{seed}"]["fold_results"].items():
            per = fr["per_sample"]["test"]
            ids = set()
            for r in READOUTS:
                ids |= set(per.get(r, {}))
            for sid in ids:
                vals = {r: per[r][sid] for r in READOUTS if sid in per.get(r, {}) and np.isfinite(per[r][sid])}
                if vals:
                    rows.append({"run": int(sid), "seed": seed, "fold": fold, "composite": float(sum(vals.values())),
                                 **{f"rmse_{r}": float(v) for r, v in vals.items()}})
    return pd.DataFrame(rows)


def load(results_dir: str, name: str):
    exp, base = METHODS[name]
    path = os.path.join(results_dir, exp, "results.json")
    if not os.path.exists(path):
        return None
    res = json.load(open(path))
    m = res["methods"]
    seeds = sorted(int(k.rsplit("_seed", 1)[1]) for k in m if k.rsplit("_seed", 1)[0] == base)
    seeds = [s for s in seeds if m[f"{base}_seed{s}"]["n_folds_succeeded"] > 0 and
             m[f"{base}_seed{s}"]["n_folds_failed"] == 0]
    if not seeds:
        return None
    seed_comp = [m[f"{base}_seed{s}"]["cross_fold_mean"]["aggregated_sum"]["test"] for s in seeds]
    per_ro = {r: float(np.mean([m[f"{base}_seed{s}"]["cross_fold_mean"]["per_readout"]["test"][r] for s in seeds]))
              for r in READOUTS}
    return {"seeds": seeds, "seed_comp": seed_comp, "per_readout": per_ro, "df": long_form(res, base, seeds)}


def run_means(d, value="composite"):
    return d["df"].pivot_table(index="run", columns="seed", values=value).mean(axis=1)


def contrast(a, b, value="composite", rng_seed=0, n_boot=10000):
    """Per-run b - a (positive = b worse than a), cluster bootstrap over runs."""
    ra, rb = run_means(a, value), run_means(b, value)
    runs = ra.index.intersection(rb.index)
    ra, rb = ra.loc[runs], rb.loc[runs]
    ok = ra.notna() & rb.notna()
    d = (rb[ok] - ra[ok]).to_numpy()
    n = len(d)
    rng = np.random.default_rng(rng_seed)
    boot = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = d.std(ddof=1) if n > 1 else float("nan")
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"n": n, "delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "d": float(d.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
            "frac_b_worse": float((d > 0).mean()), "boot_frac_b_worse": float((boot > 0).mean()), "wilcoxon_p": p}


def n_required(dz, alpha=0.05, power=0.8):
    """Smallest n for a two-sided paired t-test at effect dz (noncentral t)."""
    for n in range(5, 100000):
        tc = stats.t.ppf(1 - alpha / 2, n - 1)
        nc = abs(dz) * np.sqrt(n)
        if 1 - stats.nct.cdf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc) >= power:
            return n
    return None


def detectable_d(n, alpha=0.05, power=0.8):
    """Smallest dz detectable by a two-sided paired t-test (noncentral t)."""
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tc = stats.t.ppf(1 - alpha / 2, n - 1)
        pw = 1 - stats.nct.cdf(tc, n - 1, mid * np.sqrt(n)) + stats.nct.cdf(-tc, n - 1, mid * np.sqrt(n))
        lo, hi = (lo, mid) if pw >= power else (mid, hi)
    return hi


# ---------------------------------------------------------------- macros ----
RO_TEX = dict(zip(READOUTS, ["Biomass", "Oxygen", "Sugar", "Pigment", "AXT"]))


def _f3(x):
    return f"{x:.3f}"


def _pm(m):
    return f"\\ensuremath{{{m['mean']:.3f} \\pm {m['sd']:.3f}}}"


def _ci(lo, hi):
    return f"\\ensuremath{{[{lo:+.3f},\\,{hi:+.3f}]}}"


def macros(out) -> Dict[str, str]:
    """LaTeX macros of the main tables for whatever methods are present."""
    M, C, S = out["methods"], out["contrasts"], out["spread"]
    mac: Dict[str, str] = {}
    for key, tex in [("ours", "ours"), ("gnn", "gnnNoTF"), ("ridge", "ridge"),
                     ("xgb", "xgboost")]:
        if key in M:
            mac[f"{tex}SumTest"] = _pm(M[key])
            for r in READOUTS:
                mac[f"{tex}{RO_TEX[r]}"] = _f3(M[key]["per_readout"][r])
    for key, tex in [("gnn", "GNN"), ("comp", "Comp"), ("ridge", "Ridge"), ("xgb", "XGB")]:
        if key in C:  # ours - other (negative favours ours)
            c = C[key]
            mac[f"deltaRun{tex}"] = f"\\ensuremath{{{-c['delta']:+.3f}}}"
            mac[f"ciRun{tex}"] = _ci(-c["hi"], -c["lo"])
            mac[f"dRun{tex}"] = f"\\ensuremath{{{-c['d']:+.2f}}}"
            mac[f"runsFav{tex}"] = f"{round(c['frac_b_worse'] * 100)}\\%"
    for key, tex in [("ours", "Ours"), ("gnn", "GNN"), ("comp", "Comp"), ("xgb", "XGB"), ("ridge", "Ridge")]:
        if key in S:
            mac[f"median{tex}"] = _f3(S[key]["median"])
            mac[f"pNinety{tex}"] = _f3(S[key]["p90"])
            mac[f"nAboveOne{tex}"] = str(S[key]["n_above_1"])
    for key, tex in [("no_pretrain", "NoPretrain"), ("no_expr", "NoRNA"), ("no_gnn", "NoGNN"),
                     ("no_freeze", "NoFreeze"), ("no_pretrain_no_expr", "NoPretrainNoRNA")]:
        if key in M:
            mac[f"abl{tex}Test"] = _pm(M[key])
            mac[f"ci{tex}"] = _ci(C[key]["lo"], C[key]["hi"])
    for key, tex in [("ctl_degperm", "DegPerm"), ("ctl_randedge", "RandEdge"), ("ctl_gprshuffle", "GPRShuffle"),
                     ("ctl_metshuffle", "MetShuffle"), ("ctl_bothpaths", "BothPaths")]:
        if key in M:
            mac[f"ctl{tex}"] = f"\\ensuremath{{{M[key]['mean']:.4f} \\pm {M[key]['sd']:.3f}}}"
            mac[f"ctl{tex}Delta"] = f"\\ensuremath{{{M[key]['mean'] - M['ours']['mean']:+.4f}}}"
            mac[f"ctl{tex}CI"] = _ci(C[key]["lo"], C[key]["hi"])
    if "causal" in M:
        mac["causalRerunComposite"] = _pm(M["causal"])
    return mac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/experiments")
    ap.add_argument("--out", default="outputs/stats.json")
    ap.add_argument("--macros", default=None, help="also write LaTeX \\newcommand definitions here")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    D = {k: v for k in METHODS if (v := load(args.results_dir, k)) is not None}
    if "ours" not in D:
        raise SystemExit(f"main model results not found under {args.results_dir}/main_comparison")
    out = {"methods": {}, "contrasts": {}, "readout_contrasts": {}, "spread": {}}
    for k, d in D.items():
        sc = np.array(d["seed_comp"])
        out["methods"][k] = {"seeds": d["seeds"], "mean": float(sc.mean()),
                             "sd": float(sc.std(ddof=1)) if len(sc) > 1 else 0.0,
                             "per_seed": [float(x) for x in sc], "per_readout": d["per_readout"]}
        rm = run_means(d).dropna()
        out["spread"][k] = {"n_runs": int(len(rm)), "median": float(rm.median()), "p90": float(np.percentile(rm, 90)),
                            "n_above_1": int((rm > 1.0).sum()), "run_sd": float(rm.std(ddof=1)) if len(rm) > 1 else 0.0}
    for k in D:
        if k != "ours":
            out["contrasts"][k] = contrast(D["ours"], D[k], n_boot=args.n_boot)
            out["readout_contrasts"][k] = {r: contrast(D["ours"], D[k], value=f"rmse_{r}", n_boot=args.n_boot)
                                           for r in READOUTS if f"rmse_{r}" in D[k]["df"].columns}
    if out["contrasts"]:
        n = max(c["n"] for c in out["contrasts"].values())
        z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        out["power"] = {"n": n, "detectable_d": detectable_d(n) if n > 2 else float("nan"),
                        "n_required": {k: n_required(c["d"]) for k, c in out["contrasts"].items()
                                       if k in ("ridge", "xgb") and np.isfinite(c["d"]) and c["d"]},
                        "n_required_z_approx": {k: round((z / c["d"]) ** 2) for k, c in out["contrasts"].items()
                                                if k in ("ridge", "xgb") and np.isfinite(c["d"]) and c["d"]}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"{'method':22s} {'mean':>7s} {'sd':>6s}  " + " ".join(f"{r[:7]:>7s}" for r in READOUTS))
    for k, m in out["methods"].items():
        print(f"{k:22s} {m['mean']:7.3f} {m['sd']:6.3f}  " + " ".join(f"{m['per_readout'][r]:7.3f}" for r in READOUTS))
    if out["contrasts"]:
        print("\nrun-level contrasts (other - ours; positive = ours better):")
        for k, c in out["contrasts"].items():
            print(f"{k:22s} n={c['n']} delta={c['delta']:+.3f} CI[{c['lo']:+.3f},{c['hi']:+.3f}] d={c['d']:+.2f} "
                  f"runs ours better={c['frac_b_worse'] * 100:.0f}%")
        print("power:", out["power"])
    if args.macros:
        with open(args.macros, "w") as f:
            for k, v in macros(out).items():
                f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
