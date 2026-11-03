#!/usr/bin/env python
"""Temporal-kernel bandwidth and non-causality statistics (Sec. 4.3, appendix).

gamma = softplus(gamma_raw) + 1e-6 of the encoder's kernel and of the readout
expression adapter's kernel, over all best.pt checkpoints of a method. Future
weight: for every run and assay query time t, the share of kernel weight on
snapshots with t_k > t, at the median gamma.

    python scripts/kernel_stats.py --results outputs/experiments/main_comparison/results.json --base main
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import mnm  # noqa: E402,F401
from mnm.registry import build_from_config, load_config, resolve_variables  # noqa: E402

KERNELS = {"encoder": "model.kernel_gating.gamma_raw", "readout_adapter": "model.expression_adapter.gamma_raw"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default="main")
    ap.add_argument("--config", default=str(REPO / "configs" / "finetune_main.yaml"))
    ap.add_argument("--out", default="outputs/kernel_stats.json")
    args = ap.parse_args()
    methods = json.load(open(args.results))["methods"]
    g = {k: [] for k in KERNELS}
    for name, m in methods.items():
        if name.rsplit("_seed", 1)[0] != args.base:
            continue
        for fr in m["fold_results"].values():
            st = torch.load(fr["best_checkpoint"], map_location="cpu", weights_only=False)["task"]
            for k, key in KERNELS.items():
                if key in st:
                    g[k].append(float(F.softplus(st[key]) + 1e-6))
    out = {k: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}
           for k, v in g.items() if v}
    cfg = resolve_variables(load_config(args.config))
    ds = build_from_config({"dataset": cfg["dataset"]})["dataset"]
    for k in out:
        gam = out[k]["median"]
        fw, pre = [], []
        for s in ds.samples:
            tk = s["rna"]["rna_times"].numpy().astype(float)
            for t in s["times"].numpy().astype(float):
                w = np.exp(-gam * (t - tk) ** 2)
                w = w / w.sum()
                fw.append(w[tk > t].sum())
                pre.append(t < tk.min())
        fw = np.array(fw)
        out[k].update(mean_future_weight=float(fw.mean()), frac_queries_gt10pct_future=float((fw > 0.1).mean()),
                      frac_queries_before_first_snapshot=float(np.mean(pre)), length_scale_h=float(1 / np.sqrt(gam)),
                      n_queries=len(fw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
