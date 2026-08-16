"""Figure: delphi (the paper's own pipeline) on our trained vs randomized gemma, layer 13.

Counts are transcribed from the delphi `log_results` summaries -- everything else (rates, balanced
accuracy, standard errors) is DERIVED here, so the figure cannot silently disagree with the source.

Provenance: delphi @ EleutherAI, explainer meta-llama/llama-3.1-70b-instruct via OpenRouter
(the paper's default explainer), SAEs = ours retrained with SAE_ARCH=topk (per-token TopK, the
paper's architecture), layer 13, 100M tokens, k=32, d_sae=73728, --max_latents 100, openwebtext.
Runs: results/trained-L13 (2026-08-16 21:24), results/rand-L13 (2026-08-16 22:36).

Two panels because the headline and the mechanism are different claims:
  A  balanced accuracy -- the arms separate, in THEIR pipeline with THEIR judge.
  B  TPR vs TNR -- why. On the random arm Llama's true-negative rate is BELOW chance: it is not
     discriminating, it is agreeing. Its apparent accuracy there is response bias.

    python randomized/plot_delphi.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# arm -> scorer -> (TP, FN, TN, FP)
COUNTS = {
    "trained": {"detection": (1267, 470, 1103, 630), "fuzz": (1184, 515, 1089, 607)},
    "random":  {"detection": (883,  261, 510,  636), "fuzz": (836,  272, 469,  648)},
}
ARMS = ["trained", "random"]
SCORERS = ["detection", "fuzz"]

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"


def rates(tp, fn, tn, fp):
    """TPR, TNR, balanced accuracy and their standard errors from raw counts.

    SE treats each scoring decision as independent. Decisions CLUSTER within latents (one latent
    contributes many examples), so the true uncertainty is larger -- roughly 1.5-2x. Stated on the
    figure rather than buried, because it is the difference between ~6 sigma and ~3-4 sigma.
    """
    npos, nneg = tp + fn, tn + fp
    tpr, tnr = tp / npos, tn / nneg
    se_tpr = math.sqrt(tpr * (1 - tpr) / npos)
    se_tnr = math.sqrt(tnr * (1 - tnr) / nneg)
    return tpr, se_tpr, tnr, se_tnr, (tpr + tnr) / 2, math.sqrt(se_tpr**2 + se_tnr**2) / 2


# Bars of (metric - 0.5), i.e. ABOVE CHANCE. Two problems solved at once: chance is the real zero
# for these metrics, so bar length now encodes something (discriminative signal) and the baseline is
# honest rather than truncated; and the axis shrinks ~5x, which is what makes an SE of ~0.008
# visible at all. The twin axis on the right restores the raw values people quote -- that is a unit
# conversion of one scale, not a second measure, so it is not a dual-axis chart.
CHANCE = 0.5
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
W, EB = 0.34, dict(ecolor=INK, capsize=4, elinewidth=1.4)

# ---- A: balanced accuracy, grouped by arm ---------------------------------------------------
for i, sc in enumerate(SCORERS):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    raw = [rates(*COUNTS[a][sc])[4] for a in ARMS]
    es = [rates(*COUNTS[a][sc])[5] for a in ARMS]
    axA.bar(xs, [y - CHANCE for y in raw], W, yerr=es, error_kw=EB, zorder=3,
            color=(BLUE if sc == "detection" else ORANGE), label=sc)
    for x, y, e in zip(xs, raw, es):               # raw value: what gets quoted, and contrast relief
        axA.text(x, y - CHANCE + e + 0.006, f"{y:.3f}", ha="center", fontsize=8.5, color=MUTED)

axA.axhline(0, lw=1, color=MUTED, zorder=2)
axA.set_xticks(range(len(ARMS)), ["trained gemma", "randomized gemma"])
axA.set_xlim(-0.6, 1.6)
axA.set_ylabel("class-balanced accuracy above chance")
axA.set_ylim(0, 0.215)
_a = axA.twinx()
_a.set_ylim(CHANCE, CHANCE + 0.215)
_a.set_ylabel("(raw balanced accuracy)", color=MUTED, fontsize=9)
_a.tick_params(colors=MUTED, labelsize=8)
for sp in ("top",):
    _a.spines[sp].set_visible(False)
axA.set_title("A  the arms separate in delphi's own pipeline", fontsize=10, loc="left", color=INK)
axA.legend(frameon=False, fontsize=9, loc="upper right")

# ---- B: TPR vs TNR -- the response-bias mechanism --------------------------------------------
for i, (lab, idx) in enumerate([("true positive rate", 0), ("true negative rate", 2)]):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    r = [rates(*COUNTS[a]["detection"]) for a in ARMS]
    raw, es = [v[idx] for v in r], [v[idx + 1] for v in r]
    axB.bar(xs, [y - CHANCE for y in raw], W, yerr=es, error_kw=EB, zorder=3,
            color=(BLUE if idx == 0 else ORANGE), label=lab)
    for x, y, e in zip(xs, raw, es):
        off = (e + 0.008) if y >= CHANCE else -(e + 0.022)
        axB.text(x, y - CHANCE + off, f"{y:.3f}", ha="center", fontsize=8.5, color=MUTED)

axB.axhline(0, lw=1, color=MUTED, zorder=2)
axB.set_xticks(range(len(ARMS)), ["trained gemma", "randomized gemma"])
axB.set_xlim(-0.6, 1.6)
axB.set_ylabel("rate above chance (detection scorer)")
axB.set_ylim(-0.12, 0.32)
axB.annotate("below chance:\nthe judge is agreeing,\nnot discriminating",
             xy=(1 + 0.5 * W, -0.055), xytext=(0.30, -0.095), fontsize=8, color=MUTED,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
_b = axB.twinx()
_b.set_ylim(CHANCE - 0.12, CHANCE + 0.32)
_b.set_ylabel("(raw rate)", color=MUTED, fontsize=9)
_b.tick_params(colors=MUTED, labelsize=8)
_b.spines["top"].set_visible(False)
axB.set_title("B  on the random arm the judge agrees rather than discriminates",
              fontsize=10, loc="left", color=INK)
axB.legend(frameon=False, fontsize=9, loc="upper left")

for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

fig.suptitle("delphi + Llama-3.1-70B on gemma-2-2b layer 13 (TopK SAEs, 100M tokens, 100 latents/arm)",
             fontsize=11, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars: ±1 SE over "
         "scoring decisions \u2014 decisions cluster within latents, so true uncertainty is ~1.5\u20132\u00d7 larger.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "delphi_L13_topk.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")

print(f"\n{'arm':>10} {'scorer':>10} {'TPR':>16} {'TNR':>16} {'bal acc':>16}")
for a in ARMS:
    for sc in SCORERS:
        tpr, se1, tnr, se2, ba, seb = rates(*COUNTS[a][sc])
        print(f"{a:>10} {sc:>10} {tpr:>9.3f}±{se1:.3f} {tnr:>9.3f}±{se2:.3f} {ba:>9.3f}±{seb:.3f}")
for sc in SCORERS:
    t, r = rates(*COUNTS["trained"][sc]), rates(*COUNTS["random"][sc])
    d, sd = t[4] - r[4], math.sqrt(t[5] ** 2 + r[5] ** 2)
    print(f"  gap {sc:>10}: {d:+.3f} ± {sd:.3f}   z = {d / sd:.1f}  (independence-assuming)")
