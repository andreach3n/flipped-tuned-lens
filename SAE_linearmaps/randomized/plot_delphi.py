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


# DOTS, not bars. These are bounded metrics whose meaningful floor is 0.5 (chance), not 0, so the
# axis has to be truncated to read the differences -- and a truncated BAR lies, because bar length
# encodes magnitude. Position-encoded marks make the same axis honest, and put the error bars
# (the point of the figure) in the foreground.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=200)
W, MS = 0.16, 9

# ---- A: balanced accuracy, grouped by arm ---------------------------------------------------
for i, sc in enumerate(SCORERS):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    ys = [rates(*COUNTS[a][sc])[4] for a in ARMS]
    es = [rates(*COUNTS[a][sc])[5] for a in ARMS]
    c = BLUE if sc == "detection" else ORANGE
    axA.errorbar(xs, ys, yerr=es, fmt="o", ms=MS, color=c, ecolor=c,
                 elinewidth=1.6, capsize=4, zorder=3, label=sc,
                 markeredgecolor="white", markeredgewidth=1.2)
    for x, y in zip(xs, ys):                       # visible labels: the contrast relief
        axA.text(x + 0.10, y, f"{y:.3f}", ha="left", va="center", fontsize=8.5, color=MUTED)

axA.axhline(0.5, ls="--", lw=1, color=MUTED, zorder=2)
axA.text(1.57, 0.507, "chance", ha="right", fontsize=8, color=MUTED)
axA.set_xticks(range(len(ARMS)), ["trained gemma", "randomized gemma"])
axA.set_xlim(-0.5, 1.6)
axA.set_ylabel("class-balanced accuracy")
axA.set_ylim(0.4, 0.78)
axA.set_title("A  the arms separate in delphi's own pipeline", fontsize=10, loc="left", color=INK)
axA.legend(frameon=False, fontsize=9, loc="lower left")

# ---- B: TPR vs TNR -- the response-bias mechanism --------------------------------------------
for i, (lab, idx) in enumerate([("true positive rate", 0), ("true negative rate", 2)]):
    xs = [j + (i - 0.5) * W for j in range(len(ARMS))]
    r = [rates(*COUNTS[a]["detection"]) for a in ARMS]
    ys, es = [v[idx] for v in r], [v[idx + 1] for v in r]
    c = BLUE if idx == 0 else ORANGE
    axB.errorbar(xs, ys, yerr=es, fmt="o", ms=MS, color=c, ecolor=c,
                 elinewidth=1.6, capsize=4, zorder=3, label=lab,
                 markeredgecolor="white", markeredgewidth=1.2)
    for x, y in zip(xs, ys):
        axB.text(x + 0.10, y, f"{y:.3f}", ha="left", va="center", fontsize=8.5, color=MUTED)

axB.axhline(0.5, ls="--", lw=1, color=MUTED, zorder=2)
axB.text(1.57, 0.507, "chance", ha="right", fontsize=8, color=MUTED)
axB.set_xticks(range(len(ARMS)), ["trained gemma", "randomized gemma"])
axB.set_xlim(-0.5, 1.6)
axB.set_ylabel("rate (detection scorer)")
axB.set_ylim(0.3, 0.92)
axB.set_title("B  on the random arm the judge agrees rather than discriminates",
              fontsize=10, loc="left", color=INK)
axB.legend(frameon=False, fontsize=9, loc="lower left")

for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

fig.suptitle("delphi + Llama-3.1-70B on gemma-2-2b layer 13 (TopK SAEs, 100M tokens, 100 latents/arm)",
             fontsize=11, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Error bars: ±1 SE over scoring decisions. Decisions cluster within latents, so true "
         "uncertainty is ~1.5–2× larger.",
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
