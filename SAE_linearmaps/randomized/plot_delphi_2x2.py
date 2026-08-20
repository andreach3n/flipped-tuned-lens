"""Figure: delphi + local Llama-3.1-70B across the full 2x2 — plain vs skip-embed, trained vs random.

THE FIRST DELPHI NUMBERS IN THIS PROJECT THAT ARE TRUSTWORTHY. Every earlier delphi run went
through `convert_sae_to_sparsify.py`, which had two bugs (b_dec applied twice, because sparsify's
own `SparseCoder.encode` subtracts it AND the converter folded it into `encoder.bias`; and the
decoder-norm scaling that `sae_lens`' encode applies was never folded in). On top of that, delphi's
`filter_bos` strips BOS from the corpus while our SAEs were trained on BOS-prefixed contexts, so
the activations were out of distribution too -- |h| ~141 instead of ~174. All three are silent:
each produces a complete set of plausible numbers.

These cells instead come from `write_delphi_cache.py`, which writes delphi's latent cache directly
from `sae_lens`' own `encode` -- bypassing the conversion entirely -- and prepends BOS for the
forward pass. That writer was validated to Jaccard 1.000000 against delphi's own archived cache
(see `diff_delphi_cache.py`). delphi still does all the scoring-relevant work unchanged: example
construction, quantile sampling, prompts, and the judge.

WHAT THE FIGURE HAS TO SHOW. The scientific claim is about a DIFFERENCE OF DIFFERENCES -- whether
skip-embed separates trained from random better than a plain SAE does. So the gap (trained - rand)
is annotated per SAE type rather than left for the reader to subtract off two bar heights, and its
standard error is propagated so "bigger" can be judged against noise. Bars start at chance (0.5),
which is the meaningful zero for both AUROC and balanced accuracy; raw values are labelled on each
bar so nothing depends on reading a shifted axis.

Rows are the two metrics because they answer different questions and disagreed historically:
AUROC is rank-based and bias-free (does the judge DISCRIMINATE), balanced accuracy folds in the
judge's response bias (does it ANSWER WELL). A gap that appears in one and not the other is a
statement about the judge, not the SAE -- which is why TPR/TNR are printed underneath.

Everything is clustered BY LATENT, the unit delphi samples. Pooling the ~40k individual scoring
decisions would treat ~400 latents as ~40000 independent trials and understate the errors ~10x.

    RESULTS_DIR=/dev/shm/delphi_run/results python randomized/plot_delphi_2x2.py
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")

# (cell dir, SAE type, arm) -- the dir names written by write_delphi_cache.py / delphi --name
CELLS = [("trained_full", "full", "trained"), ("rand_full", "full", "rand"),
         ("trained_resid", "resid", "trained"), ("rand_resid", "resid", "rand")]
SAE_TYPES = [("full", "plain top-k SAE"), ("resid", "skip-embed SAE")]
ARMS = ["trained", "rand"]
SCORERS = ["detection", "fuzz"]

# Okabe-Ito, the project's existing pair. Validated for categorical use: adjacent CVD dE 29.2
# (protan) / 30.9 (tritan), normal-vision 36.2. Orange falls below 3:1 against the surface, which
# the per-bar value labels discharge -- identity never rests on colour alone here.
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARM_LABEL = {"trained": "trained gemma", "rand": "randomized gemma"}
CHANCE = 0.5


def auroc(pos, neg):
    """P(pos > neg) + 0.5*P(pos == neg) -- the exact Mann-Whitney U statistic.

    Written as the pairwise sum rather than via ranks because it is the definition, it handles ties
    without a correction term, and at ~50x50 pairs per latent the cost is irrelevant.
    """
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def per_latent(cell, scorer):
    """(AUROC, balanced accuracy, TPR, TNR) per latent for one cell."""
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
        # A latent contributes to AUROC only if the judge returned probabilities for BOTH classes;
        # a parse failure drops that example, not the latent's accuracy numbers.
        a = auroc([r["probability"] for r in pos if r.get("probability") is not None],
                  [r["probability"] for r in neg if r.get("probability") is not None])
        rows.append((a, (tpr + tnr) / 2, tpr, tnr))
    return rows


def stats(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (float("nan"), float("nan"), len(xs))
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


D = {(c, s): per_latent(c, s) for c, _, _ in CELLS for s in SCORERS}
N = {c: len(D[(c, "detection")]) for c, _, _ in CELLS}
missing = [c for c, n in N.items() if n == 0]
if missing:
    raise SystemExit(f"no scores found for {missing} under {ROOT!r} -- check RESULTS_DIR")


def cell_of(sae, arm):
    return next(c for c, s, a in CELLS if s == sae and a == arm)


def summary(sae, arm, scorer, idx):
    return stats([r[idx] for r in D[(cell_of(sae, arm), scorer)]])


fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), dpi=200)
W = 0.34

for row, (idx, metric) in enumerate(((0, "AUROC"), (1, "class-balanced accuracy"))):
    for col, scorer in enumerate(SCORERS):
        ax = axes[row][col]
        for i, arm in enumerate(ARMS):
            xs = [j + (i - 0.5) * W for j in range(len(SAE_TYPES))]
            ms = [summary(sae, arm, scorer, idx) for sae, _ in SAE_TYPES]
            ax.bar(xs, [m - CHANCE for m, _, _ in ms], W, yerr=[e for _, e, _ in ms],
                   error_kw=dict(ecolor=INK, capsize=4, elinewidth=1.4), zorder=3,
                   color=ARM_COLOUR[arm], label=ARM_LABEL[arm] if (row, col) == (0, 0) else None)
            for x, (m, e, _) in zip(xs, ms):
                off = (e + 0.008) if m >= CHANCE else -(e + 0.026)
                ax.text(x, m - CHANCE + off, f"{m:.3f}", ha="center", fontsize=8.5, color=MUTED)

        # The difference of differences is the actual claim, so state each gap rather than leaving
        # it to be eyeballed off two bar tops.
        top = ax.get_ylim()[1]
        for j, (sae, _) in enumerate(SAE_TYPES):
            (mt, et, _) = summary(sae, "trained", scorer, idx)
            (mr, er, _) = summary(sae, "rand", scorer, idx)
            d, se = mt - mr, math.sqrt(et ** 2 + er ** 2)
            ax.text(j, top * 0.94, f"gap {d:+.3f} ± {se:.3f}\nz = {d / se:.1f}",
                    ha="center", va="top", fontsize=8, color=INK)

        ax.axhline(0, lw=1, color=MUTED, zorder=2)
        ax.set_xticks(range(len(SAE_TYPES)), [lab for _, lab in SAE_TYPES])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(f"{metric} above chance")
        ax.set_title(f"{'AB'[row]}{col + 1}  {metric} — {scorer}", fontsize=10, loc="left",
                     color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)

axes[0][0].legend(frameon=False, fontsize=9, loc="upper right")

ns = " / ".join(f"{c.split('_')[1]}-{c.split('_')[0]} {N[c]}" for c, _, _ in CELLS)
fig.suptitle("delphi + local Llama-3.1-70B, gemma-2-2b layer 13 — plain vs skip-embed SAEs",
             fontsize=11.5, color=INK, y=0.985)
fig.text(0.5, 0.952, f"latents surviving min_examples=200:  {ns}", ha="center", fontsize=8.5,
         color=MUTED)
fig.text(0.5, 0.006,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars ±1 SE over "
         "LATENTS, the unit delphi samples. AUROC is the exact Mann-Whitney statistic per latent. "
         "Caches written by write_delphi_cache.py (validated to Jaccard 1.000000 against delphi's "
         "own), bypassing the sparsify conversion; BOS prepended to match the SAEs' training regime.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.945))

out = os.path.join(HERE, "plots", "delphi_L13_2x2.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}\n")

# ---- the table, including the response-bias columns the figure cannot carry -------------------
print(f"{'cell':>15} {'scorer':>10} {'AUROC':>15} {'bal acc':>15} {'TPR':>15} {'TNR':>15} {'n':>5}")
for c, _, _ in CELLS:
    for sc in SCORERS:
        cols = [stats([r[i] for r in D[(c, sc)]]) for i in range(4)]
        print(f"{c:>15} {sc:>10} " + " ".join(f"{m:8.3f}±{e:.3f}" for m, e, _ in cols)
              + f" {cols[0][2]:5d}")

print("\nGaps (trained - rand), clustered by latent:")
for idx, lab in ((0, "AUROC"), (1, "bal acc")):
    for sae, sae_lab in SAE_TYPES:
        for sc in SCORERS:
            (mt, et, nt) = summary(sae, "trained", sc, idx)
            (mr, er, nr) = summary(sae, "rand", sc, idx)
            d, se = mt - mr, math.sqrt(et ** 2 + er ** 2)
            print(f"  {lab:>8} {sae_lab:>18} {sc:>9}: {d:+.3f} ± {se:.3f}  z = {d / se:5.2f}"
                  f"  (n={nt} vs {nr})")

# The difference of differences -- does skip-embed separate the arms better than a plain SAE?
print("\nDifference of differences (skip-embed gap - plain gap):")
for idx, lab in ((0, "AUROC"), (1, "bal acc")):
    for sc in SCORERS:
        gaps, ses = [], []
        for sae, _ in SAE_TYPES:
            (mt, et, _) = summary(sae, "trained", sc, idx)
            (mr, er, _) = summary(sae, "rand", sc, idx)
            gaps.append(mt - mr)
            ses.append(math.sqrt(et ** 2 + er ** 2))
        dd = gaps[1] - gaps[0]
        sdd = math.sqrt(ses[0] ** 2 + ses[1] ** 2)
        print(f"  {lab:>8} {sc:>9}: {dd:+.3f} ± {sdd:.3f}  z = {dd / sdd:5.2f}")
