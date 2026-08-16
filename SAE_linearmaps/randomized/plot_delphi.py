"""Figure: delphi (the paper's own pipeline) on our trained vs randomized gemma, layer 13.

Everything is computed from the PER-LATENT score files, not from delphi's pooled confusion matrix.
That distinction is the whole point of this script. delphi's summary pools every scoring decision
(~3.5k trained, ~2.3k random) and a standard error over those pretends they are independent. They
are not: they come from only 35 and 23 latents. Clustering by latent -- the unit that was actually
sampled -- multiplies the standard errors by ~4 and takes the trained-vs-random gap from an
apparent 6 sigma down to ~1.5-1.8. The gap is real in POINT ESTIMATE and agrees with our own
pipeline, but this delphi run ALONE is underpowered to establish it.

Provenance: delphi @ EleutherAI, explainer meta-llama/llama-3.1-70b-instruct via OpenRouter (the
paper's default), SAEs = ours retrained with SAE_ARCH=topk (the paper's per-token TopK), layer 13,
100M tokens, k=32, d_sae=73728, --max_latents 100, openwebtext. delphi produced only 35 / 23
explanations despite max_latents=100 -- most latents fail its min_examples=200 filter.

Reads the archived run from HF (delphi_results_L13_topk.tar.gz) unless RESULTS_DIR is set.

    python randomized/plot_delphi.py
"""
import glob
import json
import math
import os
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0"
ARCHIVE = "delphi_results_L13_topk.tar.gz"
ARMS = [("trained", "trained gemma"), ("rand", "randomized gemma")]
SCORERS = ["detection", "fuzz"]
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
CHANCE = 0.5


def results_dir():
    d = os.environ.get("RESULTS_DIR")
    if d:
        return d
    cache = os.path.join(HERE, ".delphi_archive")
    if not os.path.isdir(os.path.join(cache, "results")):
        from huggingface_hub import hf_hub_download
        os.makedirs(cache, exist_ok=True)
        tarfile.open(hf_hub_download(REPO, ARCHIVE, repo_type="model")).extractall(cache)
    return os.path.join(cache, "results")


ROOT = results_dir()


def per_latent(arm, scorer):
    """(balanced accuracy, TPR, TNR) per latent. The LATENT is the sampling unit, not the example."""
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{arm}-L13/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
        rows.append(((tpr + tnr) / 2, tpr, tnr))
    return rows


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


DATA = {(a, s): per_latent(a, s) for a, _ in ARMS for s in SCORERS}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
W, EB = 0.34, dict(ecolor=INK, capsize=4, elinewidth=1.4)

# ---- A: balanced accuracy, as distance above chance -----------------------------------------
# Chance is the meaningful zero for these metrics, so bars start there: bar length then encodes
# discriminative signal and the baseline is honest rather than a truncated axis.
for i, sc in enumerate(SCORERS):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    ms = [stats([r[0] for r in DATA[(a, sc)]]) for a, _ in ARMS]
    axA.bar(xs, [m - CHANCE for m, _, _ in ms], W, yerr=[e for _, e, _ in ms], error_kw=EB,
            zorder=3, color=(BLUE if sc == "detection" else ORANGE), label=sc)
    for x, (m, e, _) in zip(xs, ms):
        axA.text(x, m - CHANCE + e + 0.008, f"{m:.3f}", ha="center", fontsize=8.5, color=MUTED)

axA.axhline(0, lw=1, color=MUTED, zorder=2)
axA.set_xticks(range(len(ARMS)), [lab for _, lab in ARMS])
axA.set_xlim(-0.6, 1.6)
axA.set_ylabel("class-balanced accuracy above chance")
axA.set_ylim(0, 0.26)
axA.set_title("A  the arms separate — but not significantly at n = 35 / 23 latents",
              fontsize=10, loc="left", color=INK)
axA.legend(frameon=False, fontsize=9, loc="lower right")
_a = axA.twinx(); _a.set_ylim(CHANCE, CHANCE + 0.26)
_a.set_ylabel("(raw balanced accuracy)", color=MUTED, fontsize=9)
_a.tick_params(colors=MUTED, labelsize=8); _a.spines["top"].set_visible(False)

for sc, y in zip(SCORERS, (0.246, 0.231)):
    (mt, et, _), (mr, er, _) = (stats([r[0] for r in DATA[(a, sc)]]) for a, _ in ARMS)
    d, s = mt - mr, math.sqrt(et ** 2 + er ** 2)
    axA.text(0.5, y, f"{sc}: gap {d:+.3f} ± {s:.3f}   z = {d / s:.1f}",
             ha="center", fontsize=8, color=MUTED)

# ---- B: TPR vs TNR -- why the random arm clears chance at all --------------------------------
for i, (lab, idx) in enumerate([("true positive rate", 1), ("true negative rate", 2)]):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    ms = [stats([r[idx] for r in DATA[(a, "detection")]]) for a, _ in ARMS]
    axB.bar(xs, [m - CHANCE for m, _, _ in ms], W, yerr=[e for _, e, _ in ms], error_kw=EB,
            zorder=3, color=(BLUE if idx == 1 else ORANGE), label=lab)
    for x, (m, e, _) in zip(xs, ms):
        off = (e + 0.012) if m >= CHANCE else -(e + 0.034)
        axB.text(x, m - CHANCE + off, f"{m:.3f}", ha="center", fontsize=8.5, color=MUTED)

axB.axhline(0, lw=1, color=MUTED, zorder=2)
axB.set_xticks(range(len(ARMS)), [lab for _, lab in ARMS])
axB.set_xlim(-0.6, 1.6)
axB.set_ylabel("rate above chance (detection scorer)")
axB.set_ylim(-0.20, 0.42)
axB.set_title("B  on the random arm the judge agrees rather than discriminates",
              fontsize=10, loc="left", color=INK)
axB.legend(frameon=False, fontsize=9, loc="upper left")
axB.annotate("below chance — a vacuous explanation\n(“unrelated texts”: 57% of this arm)\nmatches every example shown",
             xy=(1 + 0.5 * W, -0.05), xytext=(-0.05, -0.185), fontsize=8, color=MUTED,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
_b = axB.twinx(); _b.set_ylim(CHANCE - 0.20, CHANCE + 0.42)
_b.set_ylabel("(raw rate)", color=MUTED, fontsize=9)
_b.tick_params(colors=MUTED, labelsize=8); _b.spines["top"].set_visible(False)

for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

fig.suptitle("delphi + Llama-3.1-70B on gemma-2-2b layer 13 (TopK SAEs, 100M tokens)",
             fontsize=11, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars: ±1 SE over "
         "LATENTS (35 trained / 23 random) — not over pooled scoring decisions, which would understate them ~4×.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

out = os.path.join(HERE, "plots", "delphi_L13_topk.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}\n")

print(f"{'arm':>8} {'scorer':>10} {'bal acc':>16} {'TPR':>16} {'TNR':>16} {'n':>4}")
for a, _ in ARMS:
    for sc in SCORERS:
        (m, e, n), (tp, te, _), (tn, te2, _) = (
            stats([r[i] for r in DATA[(a, sc)]]) for i in (0, 1, 2))
        print(f"{a:>8} {sc:>10} {m:9.3f}±{e:.3f} {tp:9.3f}±{te:.3f} {tn:9.3f}±{te2:.3f} {n:4d}")
for sc in SCORERS:
    (mt, et, nt), (mr, er, nr) = (stats([r[0] for r in DATA[(a, sc)]]) for a, _ in ARMS)
    d, s = mt - mr, math.sqrt(et ** 2 + er ** 2)
    print(f"  gap {sc:>9}: {d:+.3f} ± {s:.3f}  z = {d / s:.2f}  (clustered by latent, n={nt} vs {nr})")
