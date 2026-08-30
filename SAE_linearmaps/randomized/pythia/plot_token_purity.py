"""THE MECHANISM FIGURE: why dictionary size decides whether delphi separates trained from random.

Reads the analysis straight out of token_purity.py -- importing it runs and PRINTS the full table,
which is intended: the figure and the numbers behind it should never drift apart.

WHAT THE READER IS SUPPOSED TO SEE. Panels A and B are the same measurement at two dictionary
sizes: for every scored latent, how often its 50 activating examples peak on the SAME token. Far
right = a pure single-word detector. In A (d_sae 131,072) the re-randomized model's curve is shoved
to the right -- 10.5% of its latents are pure detectors and the mean share is 0.509. In B
(d_sae 16,384) that same curve has collapsed onto the left wall, 2.3% and 0.184, while the trained
curve has barely moved (0.253 -> 0.262, z = -1.05). The orange curve moving and the blue one
staying put IS the result.

WHY THAT EXPLAINS THE SCORES. Heap et al.'s randomization keeps the embedding table, so token
identity is close to the only structure a re-randomized pythia has left. A 131,072-latent
dictionary at k=32 has room to spend roughly one latent per token, and a single-token detector is
honestly describable and trivially scorable -- a randomly drawn non-activating window almost never
contains the token. So delphi scoring the random model 0.81 there is the metric working CORRECTLY.
At 16,384 there is no room, its latents become mixtures, and the score falls to 0.63. The trained
model's purity does not move because its latents were never token detectors to begin with.

Panel C is the join: one point per cell, mean purity against mean fuzz AUROC. Across the seven
RANDOM-arm cells these are near rank-identical (Spearman +0.929), with all four wide-dictionary
cells above all three narrow ones on both axes.

THE HONEST LIMITS, and they are in the caption because they change what may be claimed:
  - purity does NOT explain the TRAINED arm's AUROC spread (0.606-0.866 at a flat ~0.25), so
    something else is running that, and this figure does not say what;
  - peak-token purity only sees the argmax token, so "impure" means "not a single-token detector",
    NOT "uninterpretable" -- a bigram or context detector reads as impure here;
  - n = 7 cells per arm in panel C, so the rank correlation is suggestive, not an estimate.

Curves are % of latents per bin so cells of different n compare directly. Colour means the ARM,
as everywhere else in this project; Okabe-Ito, validated (adjacent-pair CVD dE 29.2 protan / 30.9
tritan, normal-vision 36.2). In panel C dictionary size is carried by marker FILL, not by colour,
so no reader has to distinguish four hues.

    RESULTS_DIR=delphi_results python3 -u plot_token_purity.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import token_purity as tp  # noqa: E402  (import runs + prints the analysis; that is deliberate)

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARM_LABEL = {"trained": "trained", "rand": "re-randomized"}
NB = 20                                          # bins of width 0.05 over [0, 1]


def pooled(arm, dsae, idx):
    return [x[idx] for lab, _, d in tp.CELLS if d == dsae for x in tp.data[(lab, arm)].values()]


def hist_pct(vals):
    h = [0] * NB
    for v in vals:
        h[min(NB - 1, int(v * NB))] += 1
    return [100 * c / len(vals) for c in h]


fig = plt.figure(figsize=(15.4, 6.4), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[5, 4], hspace=0.42, wspace=0.19)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])

EDGES = [i / NB for i in range(NB + 1)]

for ax, dsae, tag, note in (
        (axA, 131072, "A   wide dictionary — d_sae 131,072  (expansion 64×)",
         "random latent 489:  ' moment' on 50 of 50 examples"),
        (axB, 16384, "B   narrow dictionary — d_sae 16,384  (8×)",
         "random latent  74:  50 different tokens in 50 examples")):
    for arm in ("trained", "rand"):
        v = pooled(arm, dsae, 2)
        y = hist_pct(v)
        # Step outline plus a light fill: two series per panel, so the overlap stays legible
        # without the four-way mud a single combined panel would give.
        ax.stairs(y, EDGES, color=ARM_COLOUR[arm], lw=2.0, zorder=4,
                  label=f"{ARM_LABEL[arm]}  (mean {sum(v)/len(v):.3f})")
        ax.stairs(y, EDGES, color=ARM_COLOUR[arm], fill=True, alpha=0.20, zorder=3)
    ax.set_title(tag, fontsize=10.5, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 42)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("% of latents", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right", handlelength=1.4)
    ax.text(0.985, 0.40, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.2, color=MUTED, family="monospace")

axB.set_xlabel("how often a latent's 50 activating examples peak on the SAME token\n"
               "0 = a different token every time          1 = a pure single-word detector",
               fontsize=9.5, color=INK)
axA.tick_params(labelbottom=False)

# ---------------------------------------------------------------- panel C
XLIM, YLIM = (0.10, 0.61), (0.48, 0.93)

pts = []
for arm in ("trained", "rand"):
    for lab, _, dsae in tp.CELLS:
        v = list(tp.data[(lab, arm)].values())
        pts.append((sum(t[2] for t in v) / len(v), sum(t[0] for t in v) / len(v), lab, arm, dsae))
for x, y, lab, arm, dsae in pts:
    axC.scatter(x, y, s=110, zorder=4,
                facecolor=ARM_COLOUR[arm] if dsae == 131072 else "white",
                edgecolor=ARM_COLOUR[arm], linewidth=2.0)

# Label the GROUPS, not the fourteen points. The four wide random cells sit inside
# (0.48-0.55, 0.81-0.83) and the three narrow ones inside (0.14-0.30, 0.63-0.66); per-point labels
# overprint at any readable font size, and cell identity is already carried by the other figure.
# What panel C has to say is which CLUSTER a cell falls in, so that is what gets named. Leader
# lines rather than coloured text, so identity stays with the marks (the legend) and not the ink.
LEAD = dict(arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                            shrinkA=2, shrinkB=6, connectionstyle="arc3,rad=0.12"))
for tx, ty, ax_, ay_, txt, ha in (
        (0.545, 0.895, 0.515, 0.828,
         "re-randomized, WIDE dictionary\nmost token-pure, highest-scoring", "center"),
        (0.132, 0.545, 0.170, 0.622,
         "re-randomized, NARROW\nleast pure, near the trained arm's floor", "left"),
        (0.455, 0.705, 0.330, 0.740,
         "trained: purity barely moves,\nscore spans 0.61–0.87\n(what purity does NOT explain)",
         "left")):
    axC.annotate(txt, xy=(ax_, ay_), xytext=(tx, ty), ha=ha, va="center",
                 fontsize=8.2, color=INK, zorder=6, **LEAD)

# Legend built by hand: colour = arm, fill = dictionary size. Two encodings, four combinations,
# and nobody has to tell four hues apart.
for arm in ("trained", "rand"):
    axC.scatter([], [], s=110, facecolor=ARM_COLOUR[arm], edgecolor=ARM_COLOUR[arm],
                linewidth=2.0, label=f"{ARM_LABEL[arm]} · d_sae 131,072")
    axC.scatter([], [], s=110, facecolor="white", edgecolor=ARM_COLOUR[arm],
                linewidth=2.0, label=f"{ARM_LABEL[arm]} · d_sae 16,384")
axC.axhline(0.5, lw=1.2, color=INK, zorder=2)
axC.text(0.012, 0.505, "chance", fontsize=8, color=MUTED)
axC.set_xlabel("mean top-token share  (how token-pure the cell's latents are)", fontsize=9.5)
axC.set_ylabel("fuzz AUROC", fontsize=9.5)
axC.set_title("C   purity tracks the score — but only for the random arm\n"
              "      Spearman across cells: random +0.929, trained +0.714",
              fontsize=10.5, color=INK, loc="left")
axC.set_xlim(0.10, 0.61)
axC.set_ylim(0.48, 0.93)
axC.grid(color=GRID, lw=0.6, zorder=0)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
axC.tick_params(colors=MUTED, labelsize=9)
axC.legend(frameon=False, fontsize=8.6, loc="lower right", handletextpad=0.5)

st, se_t = tp.mean_se(pooled("trained", 131072, 2))
st2, se_t2 = tp.mean_se(pooled("trained", 16384, 2))
sr, se_r = tp.mean_se(pooled("rand", 131072, 2))
sr2, se_r2 = tp.mean_se(pooled("rand", 16384, 2))
zt = (st - st2) / math.sqrt(se_t ** 2 + se_t2 ** 2)
zr = (sr - sr2) / math.sqrt(se_r ** 2 + se_r2 ** 2)

fig.suptitle("A big SAE dictionary turns a random transformer into single-word detectors — "
             "which is why auto-interp calls it interpretable",
             fontsize=12.5, color=INK, y=1.015)
fig.text(0.5, -0.075,
         f"Shrinking the dictionary collapses the RANDOM model's token purity ({sr:.3f} → {sr2:.3f}, "
         f"z = {zr:.1f}; pure detectors 10.5% → 2.3%; peak entropy 2.34 → 4.55 bits, about 5 "
         f"effective tokens to about 23) and leaves the TRAINED model's untouched "
         f"({st:.3f} → {st2:.3f}, z = {zt:.2f}).\nHeap et al.'s randomization keeps the embedding "
         "table, so token identity is nearly all a random pythia has left; a 131,072-latent "
         "dictionary has room to spend one latent per token, and those detectors are honestly "
         "describable and trivially scorable. Every latent is measured on exactly 50 activating "
         "examples\n(delphi fixes n_examples_test=50), so the small-sample purity bias cannot "
         "operate. Purity also predicts AUROC WITHIN all 14 cells (Spearman +0.295…+0.761, every "
         "one positive).   LIMITS: purity does NOT explain the trained arm's own AUROC spread "
         "(0.606–0.866 at a flat ~0.25),\nand peak-token purity only sees the argmax token — "
         "'impure' means 'not a single-token detector', not 'uninterpretable'. n = 7 cells per arm "
         "in C.   pythia-1b layer 8 · delphi + Llama-3.1-70B · 30M scoring tokens · "
         "case/whitespace-folded",
         ha="center", fontsize=8.4, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   "pythia1b_L8_token_purity.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved {out}")
