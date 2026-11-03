#!/usr/bin/env python
"""Appendix trajectory figure from export_trajectories.py output.

    python scripts/plot_trajectories.py outputs/figures/trajectories_selected.npz outputs/figures/trajectories
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

B = np.load(sys.argv[1])
out = sys.argv[2]
RO = [("biomass", "Biomass"), ("o2", "O$_2$ consumed"), ("sugar", "Glucose consumed"),
      ("carotenoids", "Total carotenoids"), ("astaxanthin", "Astaxanthin")]
COL = {"biomass": "#4C72B0", "o2": "#55A868", "sugar": "#C44E52", "carotenoids": "#DD8452", "astaxanthin": "#8172B3"}
ROWS = ["low titre (10th pct.)", "median titre", "high titre (90th pct.)"]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
n = B["times"].shape[0]
fig, axes = plt.subplots(n, 5, figsize=(7.2, 1.33 * n), squeeze=False)
for r in range(n):
    t = B["times"][r]
    for c, (k, name) in enumerate(RO):
        ax = axes[r][c]
        m, p, sd = B[f"{k}_measured"][r], B[f"{k}_pred"][r], B[f"{k}_pred_std"][r]
        okp, okm = np.isfinite(t) & np.isfinite(p), np.isfinite(t) & np.isfinite(m)
        if k != "biomass":  # specific-rate readouts are integrated from their first non-zero measurement
            nz = np.where(okm & (np.abs(np.nan_to_num(m)) > 1e-6))[0]
            if nz.size:
                okp &= np.arange(len(t)) >= nz[0]
        ax.fill_between(t[okp], (p - sd)[okp], (p + sd)[okp], color=COL[k], alpha=0.18, lw=0)
        ax.plot(t[okp], p[okp], color=COL[k], lw=1.2, ls="--", label="predicted")
        ax.plot(t[okm], m[okm], color="0.15", lw=0.9, marker="o", ms=2.2, label="measured")
        if r == 0:
            ax.set_title(name, fontsize=7.5, pad=3)
        if r == n - 1:
            ax.set_xlabel("time (h)")
        else:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel((ROWS[r] if n == 3 else f"run {r}") + "\ncumulative (norm.)", fontsize=6.5)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.8")
h, lab = axes[0][0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf")
