#!/usr/bin/env python
"""Generate a small synthetic dataset in the exact format the code reads.

Everything is drawn from parametric curves and random numbers; nothing is
derived from real measurements. Files (see docs/data_format.md):

  metadata.csv, time_series.csv, gene_expression.csv, culture_medium_reference.csv
      fermentation runs for finetuning (dataset.fermentation)
  flux_corpus.parquet
      flux vectors for the pretraining demo (dataset.flux_corpus), sampled with
      cobrapy's OptGP sampler from the GEM (scripts/build_gem.py) at the SBML's own bounds,
      with biomass >= 95% of its optimum. Sample times are assigned at random;
      unlike the paper's corpus, the bounds do not vary with time.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SBML = REPO / "data" / "gem" / "iyli21_modified.xml"

ASSAYS = ["cumulated_o2_consumption", "cumulated_sugar_consumption", "interpolated_biomass", "carotenoids",
          "astaxanthin"]
SENSORS = ["agitation_speed", "temp", "feed_temp", "ph", "do"]
MEDIUM = [f"M{i:02d}" for i in range(1, 14)]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def logistic(t, top, rate, mid):
    return top / (1.0 + np.exp(-rate * (t - mid)))


def make_runs(out: Path, n_runs: int, n_extra_genes: int, rng: np.random.Generator) -> None:
    import cobra
    logging.getLogger("cobra").setLevel(logging.ERROR)
    gpr_genes = sorted(g.id for g in cobra.io.read_sbml_model(str(SBML)).genes)

    # medium reference recipe (arbitrary positive amounts)
    reference = np.round(rng.uniform(0.5, 50.0, len(MEDIUM)), 2)
    pd.DataFrame({"component": MEDIUM, "reference": reference}).to_csv(out / "culture_medium_reference.csv",
                                                                       index=False)

    genes = [f"G{i:05d}" for i in range(len(gpr_genes) + n_extra_genes)]
    xml_ids = [f"G_{g}" for g in gpr_genes] + [""] * n_extra_genes
    G = len(genes)
    base = rng.uniform(0.5, 8.0, G)                 # log1p-scale mean expression
    slope = rng.normal(0.0, 0.4, G)                 # drift over the run
    loading = rng.normal(0.0, 1.0, G) * (rng.random(G) < 0.1)  # genes tied to the run's growth rate

    meta_rows, ts_rows, expr = [], [], {}
    for i in range(n_runs):
        task, reactor = f"SYN{i // 4 + 1:02d}", f"R{i % 4 + 1:02d}"
        ferm_id = f"{task}-{reactor}"
        mu = rng.uniform(0.12, 0.30)                # growth rate (1/h)
        mid = rng.uniform(18.0, 28.0)               # mid-exponential time (h)
        top = rng.uniform(0.6, 1.0)                 # final biomass (scaled)
        y_o2, y_s = rng.uniform(0.6, 0.9), rng.uniform(0.7, 1.0)
        pig = rng.uniform(0.3, 0.9)
        t_end = rng.uniform(60.0, 72.0)
        k = int(rng.integers(3, 6))
        rna_times = sorted(rng.choice([12, 18, 24, 30, 36, 42], size=k, replace=False).tolist())
        changes = rng.choice(MEDIUM, size=int(rng.integers(0, 3)), replace=False)
        change_str = ", ".join(f"{c}: {round(float(reference[MEDIUM.index(c)] * rng.uniform(0.5, 1.5)), 2)}"
                               for c in changes)
        meta_rows.append({"ferm_id": ferm_id, "task": task, "reactor_id": reactor, "strain_id": "strain_a",
                          "rna_times": ", ".join(str(t) for t in rna_times),
                          "initial_od": round(float(rng.uniform(0.05, 0.2)), 4), "culture_medium_change": change_str})
        for t in rna_times:
            z = base + slope * (t - 24.0) / 24.0 + loading * (mu - 0.21) * 10.0 + rng.normal(0, 0.3, G)
            expr[f"{task}-{reactor}-{t}"] = np.clip(z, 0.0, None)

        # sensor rows every 0.5 h; assay rows every ~3 h early, ~6 h late
        t_sensor = np.arange(0.0, t_end, 0.5)
        t_assay = [0.0]
        while t_assay[-1] < t_end - 3:
            t_assay.append(t_assay[-1] + (3.0 if t_assay[-1] < 30 else 6.0) + rng.uniform(-0.4, 0.4))
        t_assay = np.round(np.array(t_assay), 2)
        X = lambda t: logistic(t, top, mu, mid)  # noqa: E731
        car = lambda t: pig * logistic(t, 1.0, 0.15, mid + 12.0)  # noqa: E731
        integral = lambda t: np.array([_trapz(X(np.linspace(0, s, 50)), np.linspace(0, s, 50)) for s in t])  # noqa: E731

        def curves(t):
            o2 = y_o2 * X(t) + 0.004 * integral(t)
            sugar = y_s * X(t) + 0.006 * integral(t)
            return {"cumulated_o2_consumption": o2, "cumulated_sugar_consumption": sugar,
                    "interpolated_biomass": X(t), "carotenoids": car(t), "astaxanthin": 0.6 * car(t - 4.0)}

        assay_vals = curves(t_assay)
        for name in ASSAYS:
            assay_vals[name] = np.clip(assay_vals[name] * (1 + rng.normal(0, 0.03, len(t_assay))), 0.0, None)
            assay_vals[name][0] = 0.0 if name != "interpolated_biomass" else assay_vals[name][0]
        if rng.random() < 0.15:  # some runs miss pigment readings
            assay_vals["carotenoids"][:] = np.nan
        def sensors(t):
            growth = X(t) / top
            return {"agitation_speed": 0.4 + 0.4 * growth + rng.normal(0, 0.01, len(t)),
                    "temp": 0.5 + rng.normal(0, 0.01, len(t)),
                    "feed_temp": 0.3 + 0.05 * np.sin(t / 10.0) + rng.normal(0, 0.01, len(t)),
                    "ph": 0.55 - 0.1 * growth + rng.normal(0, 0.01, len(t)),
                    "do": 0.9 - 0.6 * growth * np.exp(-np.clip(t - mid - 10, 0, None) / 10) + rng.normal(0, 0.02, len(t))}
        s_sensor, s_assay = sensors(t_sensor), sensors(t_assay)
        for j, t in enumerate(t_sensor):
            ts_rows.append({"ferm_id": ferm_id, "time": float(t), "is_original_assay": False,
                            **{c: float(s_sensor[c][j]) for c in SENSORS}, **{a: np.nan for a in ASSAYS}})
        for j, t in enumerate(t_assay):
            ts_rows.append({"ferm_id": ferm_id, "time": float(t), "is_original_assay": True,
                            **{c: float(s_assay[c][j]) for c in SENSORS},
                            **{a: float(assay_vals[a][j]) for a in ASSAYS}})

    pd.DataFrame(meta_rows).to_csv(out / "metadata.csv", index=False)
    ts = pd.DataFrame(ts_rows).sort_values(["ferm_id", "time", "is_original_assay"], kind="stable")
    ts.round(5).to_csv(out / "time_series.csv", index=False)
    ge = pd.DataFrame({"Gene_ID": genes, "XML_ID": xml_ids})
    ge = pd.concat([ge, pd.DataFrame({k: np.round(v, 4) for k, v in expr.items()})], axis=1)
    ge.to_csv(out / "gene_expression.csv", index=False)
    print(f"{n_runs} runs, {G} genes ({len(gpr_genes)} GPR), {len(expr)} expression snapshots -> {out}")


def make_flux_corpus(out: Path, n_samples: int, seed: int, rng: np.random.Generator) -> None:
    import cobra
    from cobra.sampling import OptGPSampler
    logging.getLogger("cobra").setLevel(logging.ERROR)
    model = cobra.io.read_sbml_model(str(SBML))
    opt = model.slim_optimize()
    model.reactions.get_by_id("biomass_C").lower_bound = 0.95 * opt  # near-optimal growth
    print(f"sampling {n_samples} flux vectors (biomass optimum {opt:.4f}) ...", flush=True)
    df = OptGPSampler(model, processes=1, thinning=100, seed=seed).sample(n_samples)
    df.insert(0, "near_opt_frac", 0.95)
    df.insert(0, "time", rng.choice(np.arange(6.0, 49.0, 3.0), size=len(df)))
    df.reset_index(drop=True).to_parquet(out / "flux_corpus.parquet", index=False)
    print(f"flux corpus {df.shape} -> {out / 'flux_corpus.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "synthetic"))
    ap.add_argument("--n-runs", type=int, default=24)
    ap.add_argument("--n-extra-genes", type=int, default=400, help="genes without a GPR association")
    ap.add_argument("--flux-samples", type=int, default=150)
    ap.add_argument("--skip-flux", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    make_runs(out, args.n_runs, args.n_extra_genes, rng)
    if not args.skip_flux:
        make_flux_corpus(out, args.flux_samples, args.seed, rng)


if __name__ == "__main__":
    main()
