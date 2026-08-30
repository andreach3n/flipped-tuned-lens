"""THE MECHANISM FIGURE, entropy version: HOW MANY different words does each latent fire on?

WHY THIS EXISTS ALONGSIDE plot_token_purity.py. That figure puts the top-token SHARE on the x-axis
-- how often a latent's 50 examples peak on the same token. It is the more immediately obvious
statistic and it is the one the headline number is quoted in, but it cannot tell two very different
latents apart. At share 0.64 the random arm contains BOTH

    latent 294   'refuses' x32, 'bold' x18, nothing else          <- a union of two detectors
    latent 108   'tech'    x32 plus 17 other tokens, singletons   <- one detector with a noise tail

Peak entropy separates them (0.94 bits vs 2.40) because it does not privilege the mode. So entropy
is the better statistic, and the only reason share leads elsewhere is that "how often the same
word" needs no explaining.

THE FIX THAT MAKES ENTROPY READABLE: 2**H is the EFFECTIVE NUMBER of distinct peak tokens, so the
top axis is labelled in words rather than bits. 0 bits = always the same word. 1 bit = two words.
~2.3 bits = five. log2(50) = 5.64 bits = a different word every single time, the hard ceiling at
50 examples. A reader who does not think in bits can read the top axis alone.

WHAT THE FIGURE SAYS. Panels A and B are one measurement at two dictionary sizes. At d_sae 131,072
the re-randomized model's latents sit at 2.34 bits -- about 5 effective words each -- and 10.5% are
pure single-word detectors. At 16,384 they have spread to 4.55 bits, about 23 words, while the
trained model barely moves (4.11 -> 4.05). Panel C joins that to the score: across the seven
random-arm cells, fewer effective words means a higher auto-interp score, near rank-perfectly.

READ IT AS: a wide dictionary has room to give a random transformer one latent per token, and a
one-word latent is honestly describable and trivially scorable. Narrow the dictionary and there is
no room, so its latents blur across ~23 unrelated words and the score falls. The trained model is
unaffected because its latents were never word detectors.

LIMITS, same as the share figure: this does not explain the TRAINED arm's own AUROC spread
(0.606-0.866 at a flat ~4.1 bits), and peak entropy only sees the argmax token per example, so a
bigram or context detector reads as high-entropy without being uninterpretable.

    RESULTS_DIR=delphi_results python3 -u plot_token_entropy.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import token_purity as tp  # noqa: E402  (import runs + prints the analysis; deliberate)

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARM_LABEL = {"trained": "trained", "rand": "re-randomized"}

HMAX = math.log2(50)                       # 5.64 bits: a different token in all 50 examples
NB = 19
EDGES = [i * HMAX / NB for i in range(NB + 1)]


def pooled(arm, dsae, idx=4):              # idx 4 = folded peak entropy
    return [x[idx] for lab, _, d in tp.CELLS if d == dsae for x in tp.data[(lab, arm)].values()]


def hist_pct(vals):
    h = [0] * NB
    for v in vals:
        h[min(NB - 1, int(v / HMAX * NB))] += 1
    return [100 * c / len(vals) for c in h]


fig = plt.figure(figsize=(15.4, 6.6), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[5, 4], hspace=0.95, wspace=0.19)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])

for ax, dsae, tag, ty in (
        (axA, 131072, "A   wide dictionary — d_sae 131,072  (expansion 64×)", 1.42),
        (axB, 16384, "B   narrow dictionary — d_sae 16,384  (8×)", 1.20)):
    for arm in ("trained", "rand"):
        v = pooled(arm, dsae)
        m = sum(v) / len(v)
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], lw=2.0, zorder=4,
                  label=f"{ARM_LABEL[arm]}  (mean {m:.2f} bits ≈ {2**m:.0f} words)")
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], fill=True, alpha=0.20, zorder=3)
    # Title placed by hand above the secondary axis and its label; set_title(pad=...) collided
    # with the suptitle once the top axis was added.
    ax.text(0.0, ty, tag, transform=ax.transAxes, fontsize=10.5, color=INK, ha="left")
    ax.set_xlim(0, HMAX)
    ax.set_ylim(0, 26)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("% of latents", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left", handlelength=1.4)

    # The whole point: a second axis in WORDS, so nobody has to think in bits.
    top = ax.secondary_xaxis("top", functions=(lambda h: h, lambda h: h))
    top.set_xticks([0, 1, 2, 3, 4, 5], ["1", "2", "4", "8", "16", "32"], fontsize=9)
    top.tick_params(colors=MUTED)
    top.spines["top"].set_color(GRID)
    if ax is axA:
        top.set_xlabel("effective number of different words the latent fires on   (= 2 ^ bits)",
                       fontsize=9.5, color=INK, labelpad=6)

axB.set_xlabel("peak-token entropy (bits)      0 = always the same word · "
               f"{HMAX:.2f} = a different word every time", fontsize=9.5, color=INK)
axA.tick_params(labelbottom=False)

# ---------------------------------------------------------------- panel C
pts = []
for arm in ("trained", "rand"):
    for lab, _, dsae in tp.CELLS:
        v = list(tp.data[(lab, arm)].values())
        pts.append((sum(t[4] for t in v) / len(v), sum(t[0] for t in v) / len(v), arm, dsae))
for x, y, arm, dsae in pts:
    axC.scatter(x, y, s=110, zorder=4,
                facecolor=ARM_COLOUR[arm] if dsae == 131072 else "white",
                edgecolor=ARM_COLOUR[arm], linewidth=2.0)

# Groups, not points: the four wide random cells overlap almost exactly, as do the three narrow.
LEAD = dict(arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6,
                            connectionstyle="arc3,rad=-0.15"))
for tx, ty, ax_, ay_, txt, ha in (
        (2.30, 0.900, 2.30, 0.830, "re-randomized, WIDE dictionary\n~5 words per latent, "
         "highest-scoring", "left"),
        (3.55, 0.545, 4.60, 0.622, "re-randomized, NARROW — ~23 words per latent,\n"
         "down at the trained arm's floor", "left"),
        (2.08, 0.700, 3.62, 0.742, "trained: ~17 words at BOTH sizes,\n"
         "score still spans 0.61–0.87\n(what this does NOT explain)", "left")):
    axC.annotate(txt, xy=(ax_, ay_), xytext=(tx, ty), ha=ha, va="center",
                 fontsize=8.2, color=INK, zorder=6, **LEAD)

for arm in ("trained", "rand"):
    axC.scatter([], [], s=110, facecolor=ARM_COLOUR[arm], edgecolor=ARM_COLOUR[arm],
                linewidth=2.0, label=f"{ARM_LABEL[arm]} · d_sae 131,072")
    axC.scatter([], [], s=110, facecolor="white", edgecolor=ARM_COLOUR[arm],
                linewidth=2.0, label=f"{ARM_LABEL[arm]} · d_sae 16,384")
axC.axhline(0.5, lw=1.2, color=INK, zorder=2)
axC.text(4.72, 0.507, "chance", fontsize=8, color=MUTED)
axC.set_xlabel("mean peak-token entropy (bits)  —  fewer words to the LEFT", fontsize=9.5)
axC.set_ylabel("fuzz AUROC", fontsize=9.5)
axC.set_title("C   the fewer words a latent fires on, the better it scores\n"
              "      Spearman across cells: random −0.929, trained −0.714",
              fontsize=10.5, color=INK, loc="left")
axC.set_xlim(2.0, 5.0)
axC.set_ylim(0.48, 0.93)
axC.grid(color=GRID, lw=0.6, zorder=0)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
axC.tick_params(colors=MUTED, labelsize=9)
axC.legend(frameon=False, fontsize=8.6, loc="lower left", handletextpad=0.5,
           bbox_to_anchor=(0.0, 0.02))

hr, hr_se = tp.mean_se(pooled("rand", 131072))
hr2, hr2_se = tp.mean_se(pooled("rand", 16384))
ht, ht_se = tp.mean_se(pooled("trained", 131072))
ht2, ht2_se = tp.mean_se(pooled("trained", 16384))

fig.suptitle("A wide SAE dictionary lets a random transformer spend one latent per word — "
             "and that is what auto-interp scores as interpretable",
             fontsize=12.5, color=INK, y=1.10)
fig.text(0.5, -0.11,
         f"Each latent is summarised by the entropy of the token it peaks on across its 50 "
         f"activating examples; 2^entropy is the effective number of different words; random latent 12 fires on "
         f"' w' and nothing else (0 bits) while random latent 74 hits 50 different words in 50 examples (5.64). Shrinking "
         f"the dictionary spreads the RANDOM model's latents from {hr:.2f} bits (~{2**hr:.0f} "
         f"words) to {hr2:.2f} (~{2**hr2:.0f}),\nand leaves the TRAINED model where it was "
         f"({ht:.2f} → {ht2:.2f}). Heap et al.'s randomization keeps the embedding table, so token "
         "identity is nearly all a random pythia has left, and a one-word latent is honestly "
         "describable and trivially scorable — a randomly drawn\nnon-activating window almost "
         "never contains the word. Entropy is preferred to the top-token share because the share "
         "cannot separate a two-word union ('refuses'×32 + 'bold'×18, 0.94 bits) from a one-word "
         "detector with a noise tail\n('tech'×32 + 17 singletons, 2.40 bits) — both score 0.64 on "
         "share.   LIMITS: this does not explain the trained arm's own 0.61–0.87 spread at a flat "
         "~4.1 bits, and only the argmax token per example is used, so a bigram or context\n"
         "detector reads as high-entropy without being uninterpretable. n = 7 cells per arm in C.  "
         " pythia-1b layer 8 · delphi + Llama-3.1-70B · 30M scoring tokens · case/whitespace-folded",
         ha="center", fontsize=8.4, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   "pythia1b_L8_token_entropy.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved {out}")
