"""Phase 3 of the LLM-judge experiment: join judge_ratings.json to judge_key.json,
report per-SAE stats with significance, and plot the abstractness/breadth/coherence
distributions for full / hybrid / outbias.

Pure local analysis — no API, no GPU. Run on the Mac:
  python3 analyze_judge.py
Writes plots/judge_axes.png and plots/judge_abstractness_dist.png.
"""
import json
import os
from math import erf, sqrt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")

SAES = ["full", "hybrid", "outbias"]
AXES = ["breadth", "coherence", "abstractness"]
BLOCKS = ["peak", "typical"]
# Okabe-Ito, colorblind-safe, fixed SAE order (validated: worst-pair ΔE 51.6)
COL = {"full": "#0072B2", "hybrid": "#E69F00", "outbias": "#009E73"}

# ---- load + join ------------------------------------------------------------
R = json.load(open(os.path.join(BASE, "judge_ratings.json")))
K = json.load(open(os.path.join(BASE, "judge_key.json")))
vals = {s: {b: {a: [] for a in AXES} for b in BLOCKS} for s in SAES}
for aid, meta in K.items():
    if aid not in R:
        continue
    r = R[aid]
    for b in BLOCKS:
        for a in AXES:
            vals[meta["sae"]][b][a].append(r[f"{b}_{a}"])
for s in SAES:
    for b in BLOCKS:
        for a in AXES:
            vals[s][b][a] = np.array(vals[s][b][a], dtype=float)


def mean_ci(x):
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def ztest(x, y):
    """Two-sample mean-difference z-test (n~944, CLT holds). Returns (Δ, z, p)."""
    d = x.mean() - y.mean()
    se = sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    z = d / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return d, z, p


# ---- stats to stdout --------------------------------------------------------
for b in BLOCKS:
    print(f"\n===== {b.upper()} block — mean [95% CI] =====")
    for a in AXES:
        cells = "  ".join(f"{s} {vals[s][b][a].mean():.2f}±{mean_ci(vals[s][b][a])[1]:.2f}" for s in SAES)
        print(f"  {a:13} {cells}")
    print(f"  -- differences vs FULL ({b}) --")
    for a in AXES:
        for s in ("hybrid", "outbias"):
            d, z, p = ztest(vals[s][b][a], vals["full"][b][a])
            sig = "SIG" if p < 0.05 else "ns "
            print(f"     {a:13} {s:8} Δ={d:+.2f}  z={z:+5.1f}  p={p:.1e}  [{sig}]")


# ---- figure A: three-axis signature ----------------------------------------
def grouped(ax, groups, series_vals, series_err=None, ylabel="", labels=None):
    x = np.arange(len(groups)); w = 0.26
    for i, s in enumerate(SAES):
        err = series_err[s] if series_err else None
        ax.bar(x + (i - 1) * w, series_vals[s], w, yerr=err, color=COL[s], label=s,
               capsize=3, edgecolor="white", linewidth=0.8, error_kw=dict(lw=1, ecolor="#555"))
    ax.set_xticks(x); ax.set_xticklabels(labels or groups)
    ax.grid(axis="y", lw=0.5, color="#e8e8e8"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, b in zip(axs, BLOCKS):
    m = {s: [vals[s][b][a].mean() for a in AXES] for s in SAES}
    e = {s: [mean_ci(vals[s][b][a])[1] for a in AXES] for s in SAES}
    grouped(ax, AXES, m, e, ylabel="mean rating (1–5)" if b == "peak" else "")
    ax.set_ylim(0, 5); ax.set_title(f"{b} block", fontsize=11)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("Judge ratings by SAE:  breadth ↑,  abstractness flat,  coherence ↓",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "judge_axes.png"), dpi=150, bbox_inches="tight")

# ---- figure B: abstractness distribution -----------------------------------
levels = [1, 2, 3, 4, 5]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, b in zip(axs, BLOCKS):
    frac = {s: [100 * (vals[s][b]["abstractness"] == L).mean() for L in levels] for s in SAES}
    grouped(ax, levels, frac, None, ylabel="% of features" if b == "peak" else "",
            labels=["1\nsurface", "2", "3\ntopic", "4", "5\nabstract"])
    ax.set_title(f"{b} block", fontsize=11)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("Abstractness rating distribution by SAE (frequency-matched) — distributions overlap",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "judge_abstractness_dist.png"), dpi=150, bbox_inches="tight")

print(f"\nwrote plots/judge_axes.png and plots/judge_abstractness_dist.png")
