"""Phase 3 of the LLM-judge experiment: join judge_ratings.json to judge_key.json,
report per-SAE stats with significance, and plot the distributions for full / hybrid
/ outbias.

Two of the three axes come from the LLM (coherence, abstractness). BREADTH is NOT
LLM-judged — it's the Python "effective #words" metric (exp entropy) computed in
eval_trivial.py and stored per feature in stats_*.pt:
  peak breadth   <- stats["eff_peak"]   (peak sample)
  typical breadth <- stats["eff"]        (activation-weighted sample)
So the LLM only rates what needs judgment; breadth is an exact count over ALL firings.

Pure local analysis — no API, no GPU. Run on the Mac:
  python3 analyze_judge.py
Expects stats_*.pt scp'd from the box into this folder (or set STATS_DIR).
Writes plots/judge_axes.png, plots/judge_abstractness_dist.png, plots/judge_breadth.png.
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
STATS_DIR = os.environ.get("STATS_DIR", BASE)   # where stats_{sae}.pt live (scp'd from the box)

SAES = ["full", "hybrid", "outbias"]
LLM_AXES = ["coherence", "abstractness"]         # rated by the LLM judge
AXES = ["breadth", "coherence", "abstractness"]  # breadth sourced from Python eff_words
BLOCKS = ["peak", "typical"]
# Okabe-Ito, colorblind-safe, fixed SAE order (validated: worst-pair ΔE 51.6)
COL = {"full": "#0072B2", "hybrid": "#E69F00", "outbias": "#009E73"}


# ---- breadth from Python eff_words (per feature, over ALL firings) -----------
def load_stats():
    """Return {sae: {'eff': arr, 'eff_peak': arr}} indexed by feature id, or None
    if the stats files aren't present locally (breadth then skipped gracefully)."""
    try:
        import torch
    except ImportError:
        return None
    out = {}
    for s in SAES:
        p = os.path.join(STATS_DIR, f"stats_{s}.pt")
        if not os.path.exists(p):
            return None
        d = torch.load(p, map_location="cpu")
        if "eff" not in d:      # old stats without the eff_words fields
            return None
        out[s] = {"eff": d["eff"].numpy(), "eff_peak": d["eff_peak"].numpy()}
    return out


STATS = load_stats()
HAVE_BREADTH = STATS is not None
if not HAVE_BREADTH:
    print("(stats_*.pt with 'eff' not found in STATS_DIR — skipping breadth; "
          "scp the box's stats_*.pt here or set STATS_DIR to enable it)")
REPORT_AXES = AXES if HAVE_BREADTH else LLM_AXES

# ---- load + join ------------------------------------------------------------
R = json.load(open(os.path.join(BASE, "judge_ratings.json")))
K = json.load(open(os.path.join(BASE, "judge_key.json")))
vals = {s: {b: {a: [] for a in AXES} for b in BLOCKS} for s in SAES}
for aid, meta in K.items():
    if aid not in R:
        continue
    r, s, feat = R[aid], meta["sae"], meta["feat"]
    for b in BLOCKS:
        for a in LLM_AXES:
            vals[s][b][a].append(r[f"{b}_{a}"])
    if HAVE_BREADTH:                                  # peak <- eff_peak, typical <- eff (weighted)
        vals[s]["peak"]["breadth"].append(float(STATS[s]["eff_peak"][feat]))
        vals[s]["typical"]["breadth"].append(float(STATS[s]["eff"][feat]))
for s in SAES:
    for b in BLOCKS:
        for a in REPORT_AXES:
            vals[s][b][a] = np.array(vals[s][b][a], dtype=float)


def mean_ci(x):
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def ztest(x, y):
    """Two-sample mean-difference z-test (n large, CLT holds). Returns (Δ, z, p)."""
    d = x.mean() - y.mean()
    se = sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    z = d / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return d, z, p


# ---- stats to stdout --------------------------------------------------------
# abstractness == 0 means "no coherent pattern" (N/A), NOT "less abstract than 1".
# Exclude those from abstractness means/tests; report them separately as incoherent %.
def axis_vals(s, b, a):
    x = vals[s][b][a]
    return x[x > 0] if a == "abstractness" else x

for b in BLOCKS:
    print(f"\n===== {b.upper()} block — mean [95% CI] =====   (breadth = effective #words; coherence 1-5; abstractness 1-5)")
    for a in REPORT_AXES:
        cells = "  ".join(f"{s} {axis_vals(s, b, a).mean():.2f}±{mean_ci(axis_vals(s, b, a))[1]:.2f}" for s in SAES)
        print(f"  {a:13} {cells}")
    inc = "  ".join(f"{s} {100 * (vals[s][b]['abstractness'] == 0).mean():.0f}%" for s in SAES)
    print(f"  {'incoherent':13} {inc}   (abstractness=0; excluded from the abstractness mean above)")
    print(f"  -- differences vs FULL ({b}) --")
    for a in REPORT_AXES:
        for s in ("hybrid", "outbias"):
            d, z, p = ztest(axis_vals(s, b, a), axis_vals("full", b, a))
            sig = "SIG" if p < 0.05 else "ns "
            print(f"     {a:13} {s:8} Δ={d:+.2f}  z={z:+5.1f}  p={p:.1e}  [{sig}]")


# ---- shared grouped-bar helper ----------------------------------------------
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


# ---- figure A: LLM axes (coherence, abstractness) on the 0-5 scale ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True)
for ax, b in zip(axs, BLOCKS):
    m = {s: [axis_vals(s, b, a).mean() for a in LLM_AXES] for s in SAES}
    e = {s: [mean_ci(axis_vals(s, b, a))[1] for a in LLM_AXES] for s in SAES}
    grouped(ax, LLM_AXES, m, e, ylabel="mean rating (1–5)" if b == "peak" else "")
    ax.set_ylim(0, 5); ax.set_title(f"{b} block", fontsize=11)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("LLM-judge ratings by SAE (frequency-matched): coherence & abstractness",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "judge_axes.png"), dpi=150, bbox_inches="tight")

# ---- figure B: abstractness distribution -----------------------------------
levels = [0, 1, 2, 3, 4, 5]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, b in zip(axs, BLOCKS):
    frac = {s: [100 * (vals[s][b]["abstractness"] == L).mean() for L in levels] for s in SAES}
    grouped(ax, levels, frac, None, ylabel="% of features" if b == "peak" else "",
            labels=["0\nnoise", "1\nsurface", "2", "3\ntopic", "4", "5\nabstract"])
    ax.set_title(f"{b} block", fontsize=11)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("Abstractness rating distribution by SAE (frequency-matched)",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "judge_abstractness_dist.png"), dpi=150, bbox_inches="tight")

# ---- figure C: breadth (Python eff_words, its own scale) --------------------
made = "plots/judge_axes.png and plots/judge_abstractness_dist.png"
if HAVE_BREADTH:
    fig, ax = plt.subplots(figsize=(6, 4.3))
    m = {s: [vals[s][b]["breadth"].mean() for b in BLOCKS] for s in SAES}
    e = {s: [mean_ci(vals[s][b]["breadth"])[1] for b in BLOCKS] for s in SAES}
    grouped(ax, BLOCKS, m, e, ylabel="effective # words (breadth)")
    ax.set_title("Breadth by SAE — Python eff_words (frequency-matched)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "judge_breadth.png"), dpi=150, bbox_inches="tight")
    made += " and plots/judge_breadth.png"

print(f"\nwrote {made}")
