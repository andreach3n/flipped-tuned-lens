"""THE FIGURE: the trained-vs-random AUROC gap as a function of how often the latent fires.

This is the one plot that carries the whole Pythia result. It makes three points at once, none
of which survive in the aggregate number everyone reports:

  1. The gap is MONOTONE in firing frequency -- random-model latents win among rare latents,
     trained-model latents tie or win among frequent ones.
  2. It CROSSES ZERO. So "are random-model SAE features as interpretable as trained ones?" has
     no single answer; it depends entirely on which part of the dictionary you look at.
  3. The shape REPLICATES ACROSS LEARNING RATES, while the aggregate does not: naive mean AUROC
     gives -0.183 (z=-17) at LR 1e-3 and +0.015 (z=1.2) at 2e-3 -- opposite conclusions from
     the same models. The frequency-resolved structure is the stable object.

Why a gap-vs-bin line and not grouped bars of the raw AUROCs: the quantity of interest is a
DIFFERENCE with a meaningful zero (polarity), and how that difference moves along an ordered
covariate. Bars of raw values would show eight numbers and hide the trend and the crossing.

Error bars are +/- 1 SE of the within-bin difference, latents being the sampling unit. Bins
thin in either arm are dropped by matched_comparison.py's rule and simply do not appear.

    RESULTS_DIR=/dev/shm/delphi_run/results python -u plot_frequency_gap.py
"""
import glob
import json
import math
import os
import re

import numpy as np
from safetensors.numpy import load_file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT     = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
LAYER    = int(os.environ.get("LAYER", 8))
LRS      = os.environ.get("LRS", "1e-3,2e-3").split(",")
SCORER   = os.environ.get("SCORER", "fuzz")
CELL_FMT = os.environ.get("CELL_FMT", "pythia1b_{arm}_L{layer}_lr{lr}")
MIN_PER_BIN = int(os.environ.get("MIN_PER_BIN", 5))
EDGES = [0, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, float("inf")]

# Okabe-Ito, validated for this use: adjacent-pair CVD separation dE 29.2 (protan) / 30.9
# (tritan), normal-vision 36.2 -- all far above the 8 floor. The low-contrast warning on the
# orange is discharged by the legend plus the numeric table this script prints.
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def per_latent_auroc(cell):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{SCORER}/*.txt")):
        nums = re.findall(r"\d+", os.path.basename(f))
        if not nums:
            continue
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r["probability"] for r in recs if r["activating"] and r.get("probability") is not None]
        neg = [r["probability"] for r in recs if not r["activating"] and r.get("probability") is not None]
        a = auroc(pos, neg)
        if a is not None:
            out[int(nums[-1])] = a
    return out


def firing_counts(cell):
    counts = {}
    for s in sorted(glob.glob(f"{ROOT}/{cell}/latents/layers.{LAYER}/*.safetensors")):
        start = int(os.path.basename(s).split("_")[0])
        idx = load_file(s)["locations"][:, 2].astype(np.int64) + start
        for lid, c in zip(*np.unique(idx, return_counts=True)):
            counts[int(lid)] = counts.get(int(lid), 0) + int(c)
    return counts


def stats(xs):
    if len(xs) < 2:
        return float("nan"), float("nan"), len(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


def label(lo, hi):
    # One decimal above 1000: int(1600/1000) would render 1600 as "1k" and 3200 as "3k",
    # which mislabels the bins by up to 60%.
    def s(v):
        if v == float("inf"):
            return "∞"
        if v >= 1000:
            return f"{v/1000:.1f}k".replace(".0k", "k")
        return str(int(v))
    return f"<{s(hi)}" if lo == 0 else (f">{s(lo)}" if hi == float("inf") else f"{s(lo)}–{s(hi)}")


# ---------------------------------------------------------------- compute
series = {}
print(f"\nAUROC gap (trained - random), scorer={SCORER}, layer {LAYER}\n")
for lr in LRS:
    pts = {}
    for arm in ("trained", "rand"):
        cell = CELL_FMT.format(arm=arm, layer=LAYER, lr=lr)
        a, fc = per_latent_auroc(cell), firing_counts(cell)
        pts[arm] = [(fc[i], a[i]) for i in sorted(a) if i in fc]
    rows = []
    print(f"  LR {lr}:")
    for bi, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        seg = {arm: [v for f, v in pts[arm] if lo <= f < hi] for arm in ("trained", "rand")}
        if len(seg["trained"]) < MIN_PER_BIN or len(seg["rand"]) < MIN_PER_BIN:
            continue
        mt, set_, nt = stats(seg["trained"])
        mr, ser, nr = stats(seg["rand"])
        g, gse = mt - mr, math.sqrt(set_ ** 2 + ser ** 2)
        rows.append((bi, g, gse, nt, nr))
        print(f"    {label(lo, hi):>10}  gap {g:+.3f} +/- {gse:.3f}  (n {nt} vs {nr})")
    series[lr] = rows

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
used = sorted({bi for rows in series.values() for bi, *_ in rows})
xmap = {bi: k for k, bi in enumerate(used)}

ax.axhline(0, lw=1.4, color=INK, zorder=4)

for lr, colour in zip(LRS, (BLUE, ORANGE)):
    rows = series[lr]
    xs = [xmap[bi] for bi, *_ in rows]
    ys = [g for _, g, _, _, _ in rows]
    es = [e for _, _, e, _, _ in rows]
    ax.errorbar(xs, ys, yerr=es, marker="o", ms=7, lw=2.2, capsize=4, color=colour,
                ecolor=colour, elinewidth=1.3, label=f"LR {lr}", zorder=5)

ax.set_xticks(range(len(used)), [label(EDGES[bi], EDGES[bi + 1]) for bi in used], fontsize=9)
ax.set_xlabel("latent firing count on the scoring corpus (30M tokens)", fontsize=10)
ax.set_ylabel(f"{SCORER} AUROC gap  (trained − randomized)", fontsize=10)
# Scale to the DATA, then shade the positive half. Shading before setting limits would drag
# the axis to the top of the span and squash the series into a corner.
lo_y, hi_y = ax.get_ylim()
pad = (hi_y - lo_y) * 0.16
ax.set_ylim(lo_y - pad, hi_y + pad)
ax.axhspan(0, ax.get_ylim()[1], color=GRID, alpha=0.28, zorder=0)
# Annotations right-aligned, legend upper-left: the data occupies the lower-left (rare latents,
# large negative gap) and the right edge, so the upper-left corner is the only empty region.
ax.text(0.985, 0.96, "trained more interpretable", transform=ax.transAxes,
        fontsize=9, color=MUTED, va="top", ha="right")
ax.text(0.985, 0.04, "randomized more interpretable", transform=ax.transAxes,
        fontsize=9, color=MUTED, va="bottom", ha="right")
ax.set_title("Auto-interp separates trained from random only among FREQUENT latents\n"
             "pythia-1b layer 8 · delphi + Llama-3.1-70B · per-latent AUROC, ±1 SE over latents",
             fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=1)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=9)

fig.text(0.5, -0.02,
         "The aggregate gap is a mixture over this curve, so it inherits the dictionary's "
         "frequency profile: naive mean AUROC gives −0.183 (z=−17)\nat LR 1e-3 and +0.015 "
         "(z=1.2) at 2e-3 — opposite conclusions from the same models. The curve itself is "
         "stable across both.",
         ha="center", fontsize=8, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   f"pythia1b_L{LAYER}_freq_gap_{SCORER}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved {out}")
