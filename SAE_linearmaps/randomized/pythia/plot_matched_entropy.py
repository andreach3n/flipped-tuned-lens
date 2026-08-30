"""THE CONTROLLED ENTROPY FIGURE: one recipe, two dictionary sizes, no pooling.

Companion to plot_matched_purity.py, same two cells and same structure, with peak-token ENTROPY on
the x-axis instead of the top-token share. plot_token_entropy.py averages four SAEs per arm at
d_sae 131,072 against three at 16,384, and those pools differ in architecture, codebase, LR,
context length and position-0 masking as well as width -- broad evidence, but not a controlled
comparison. This uses the only pair that isolates the dictionary:

    top-k R=64, lr 2e-3   d_sae 131,072   sparsify, per-token TopK, k=32, ctx 2048, no p0 mask
    top-k R=8,  lr 2e-3   d_sae  16,384   sparsify, per-token TopK, k=32, ctx 2048, no p0 mask

Same recipe; R is the expansion factor and d_sae = R * d_model with d_model = 2048.

WHY ENTROPY AS WELL AS SHARE. The share cannot separate a two-word union from a one-word detector
with a noise tail -- 'refuses' x32 + 'bold' x18 and 'tech' x32 + 17 singletons both score 0.64.
Entropy does (0.94 bits vs 2.40), because it does not privilege the mode. And 2**H is the effective
number of distinct peak words, so the top axis reads in words rather than bits.

TWO CAVEATS, BOTH REAL, AND THEY PULL IN OPPOSITE DIRECTIONS.

1. ENTROPY FROM 50 EXAMPLES IS A FLOOR. You cannot observe more distinct words than you have
   draws, so log2(50) = 5.64 bits is a hard ceiling. Rarefaction (token_purity.py) shows the share
   has converged by n=50 (+/-0.004 from n=40) while entropy is still climbing +0.07..+0.20 per ten
   examples, steepest where entropy is highest. So every word count here is a LOWER BOUND, loosest
   at the scattered end -- which means the true R=64-vs-R=8 gap is WIDER than drawn, not narrower.

2. THE TRAINED SIDE OF PANEL A IS A SURVIVORSHIP SAMPLE. That SAE returned only 192 of 500 latents
   because the rest fired under delphi's min_examples=200, so the scored 192 are its highest-firing
   third. Panel C measures what that does. Truncating the uncontaminated 1e-3 wide cell to the same
   top 38% moves it 4.36 -> 3.71 bits, which is 58% of the distance to the 3.24 actually observed --
   so most, though not all, of the trained arm's low entropy at R=64 is selection. Do not quote the
   trained change from this figure. The random arm needs no such warning: it barely moves under the
   same truncation (+0.04 bits at R=64, +0.18 at R=8) and had 82% and 96% survival anyway.
   Note also that the two trained curves run in OPPOSITE directions -- the wide cell falls 0.65 bits
   and the narrow one RISES 0.41 -- so this is a property of particular cells, not of "trained".

WHAT PANEL B SAYS, and it is the most interesting cell here: at R=8 the two arms have essentially
the SAME entropy (3.76 vs 3.73 bits, ~13-14 effective words) while their fuzz AUROC differs by
0.206 (0.866 vs 0.660). Word-concentration explains the random arm's score and explains nothing
about the trained arm's.

    RESULTS_DIR=delphi_results python3 -u plot_matched_entropy.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch as t  # noqa: E402

import token_purity as tp  # noqa: E402  (import runs + prints the full analysis; deliberate)

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARM_LABEL = {"trained": "trained", "rand": "re-randomized"}
ROOT = os.environ.get("RESULTS_DIR", "delphi_results")

WIDE, NARROW, REF = "top-k R64 2e-3", "top-k R8 2e-3", "top-k R64 1e-3"
CELLFILE = {(WIDE, "trained"): "pythia1b_trained_L8_lr2e-3",
            (WIDE, "rand"): "pythia1b_rand_L8_lr2e-3",
            (NARROW, "trained"): "pythia1b_trained_R8_L8_lr2e-3",
            (NARROW, "rand"): "pythia1b_rand_R8_L8_lr2e-3",
            (REF, "trained"): "pythia1b_trained_L8_lr1e-3"}

HMAX = math.log2(50)                     # 5.64 bits: a different word in all 50 examples
NB = 16
EDGES = [i * HMAX / NB for i in range(NB + 1)]


def ents(cell, arm):                     # index 4 = folded peak entropy
    return [x[4] for x in tp.data[(cell, arm)].values()]


def hist_pct(v):
    h = [0] * NB
    for x in v:
        h[min(NB - 1, int(x / HMAX * NB))] += 1
    return [100 * c / len(v) for c in h]


def firing_curve(cell, arm, qs):
    """Mean entropy when only the top q fraction of latents BY FIRING RATE is kept."""
    o = t.load(f"{ROOT}/{CELLFILE[(cell, arm)]}/log/hookpoint_firing_counts.pt",
               weights_only=True, map_location="cpu")
    c = (list(o.values())[0] if isinstance(o, dict) else o).tolist()
    rows = sorted(((c[lid], v[4]) for lid, v in tp.data[(cell, arm)].items()
                   if lid < len(c) and c[lid] > 0), key=lambda r: -r[0])
    return [tp.mean_se([e for _, e in rows[:max(2, round(q * len(rows)))]])[0] for q in qs]


fig = plt.figure(figsize=(15.4, 6.6), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[5, 4], hspace=0.95, wspace=0.20)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])

for ax, cell, tag, ty in (
        (axA, WIDE, "A   R=64  →  d_sae 131,072", 1.42),
        (axB, NARROW, "B   R=8  →  d_sae 16,384   —   same recipe, only the expansion factor differs",
         1.20)):
    for arm in ("trained", "rand"):
        v = ents(cell, arm)
        m = sum(v) / len(v)
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], lw=2.0, zorder=4,
                  label=f"{ARM_LABEL[arm]}  {m:.2f} bits, ≥{2**m:.0f} words  (n={len(v)})")
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], fill=True, alpha=0.20, zorder=3)
    # Title placed by hand above the secondary axis and its label -- set_title(pad=) collides with
    # the suptitle once a top axis exists.
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

    top = ax.secondary_xaxis("top", functions=(lambda h: h, lambda h: h))
    top.set_xticks([0, 1, 2, 3, 4, 5], ["1", "2", "4", "8", "16", "32"], fontsize=9)
    top.tick_params(colors=MUTED)
    top.spines["top"].set_color(GRID)
    if ax is axA:
        top.set_xlabel("effective number of different words the latent fires on   "
                       "(= 2 ^ bits; a FLOOR — see caption)", fontsize=9.5, color=INK, labelpad=6)

axB.set_xlabel(f"peak-token entropy (bits)      0 = always the same word · "
               f"{HMAX:.2f} = a different word every time", fontsize=9.5, color=INK)
axA.tick_params(labelbottom=False)

# ---------------------------------------------------------------- panel C, survivorship control
QS = [1.00, 0.75, 0.50, 0.38, 0.25, 0.15]
DASH = (0, (5, 2))
for cell, arm, col, ls, lab in (
        (REF, "trained", BLUE, "-", "trained, R=64 (1e-3, 79% survived)"),
        (NARROW, "trained", BLUE, DASH, "trained, R=8 (91% survived)"),
        (WIDE, "rand", ORANGE, "-", "random, R=64 (82% survived)"),
        (NARROW, "rand", ORANGE, DASH, "random, R=8 (96% survived)")):
    axC.plot([100 * q for q in QS], firing_curve(cell, arm, QS),
             color=col, ls=ls, lw=2.0, marker="o", ms=5, zorder=4, label=lab)

obs = sum(ents(WIDE, "trained")) / len(ents(WIDE, "trained"))
axC.scatter([38], [obs], s=150, marker="*", color=BLUE, zorder=6,
            edgecolor="white", linewidth=0.8)
axC.annotate(f"R=64 trained actually scored\n{obs:.2f} bits — but only its top\n38% by firing rate "
             "was ever\nscored (192 of 500)",
             xy=(38, obs), xytext=(62, 4.30), ha="left", va="center", fontsize=8.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=8,
                             connectionstyle="arc3,rad=-0.2"))
axC.invert_xaxis()
axC.set_xlabel("keep only the top % of latents by firing rate", fontsize=9.5)
axC.set_ylabel("mean peak-token entropy (bits)", fontsize=9.5)
axC.set_title("C   why the trained arm's change is not trustworthy\n"
              "      truncating to the top 38% moves the WIDE trained cell −0.65 bits; the random arm moves +0.04",
              fontsize=10.5, color=INK, loc="left")
axC.grid(color=GRID, lw=0.6, zorder=0)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
axC.tick_params(colors=MUTED, labelsize=9)
axC.legend(frameon=False, fontsize=8.4, loc="lower left", handlelength=3.4)

mw, sw = tp.mean_se(ents(WIDE, "rand"))
mn, sn = tp.mean_se(ents(NARROW, "rand"))
tw, tws = tp.mean_se(ents(WIDE, "trained"))
tn, tns = tp.mean_se(ents(NARROW, "trained"))
zr = (mn - mw) / math.sqrt(sw ** 2 + sn ** 2)
zt = (tn - tw) / math.sqrt(tws ** 2 + tns ** 2)

fig.suptitle("Controlled for everything but dictionary size: shrinking it spreads the RANDOM "
             "model's latents across many more words", fontsize=12.5, color=INK, y=1.10)
fig.text(0.5, -0.115,
         f"The only pair in this project that varies d_sae alone — both are stock sparsify plain "
         f"top-k, k=32, 100M tokens, lr 2e-3, ctx 2048, no position-0 mask; R is the expansion "
         f"factor and d_sae = R × 2048.\nRandom arm {mw:.2f} → {mn:.2f} bits, about "
         f"{2**mw:.0f} → {2**mn:.0f} effective words (z = {zr:.1f}); trained arm {tw:.2f} → "
         f"{tn:.2f} (z = {zt:.1f}).   WORD COUNTS ARE FLOORS: entropy from 50 draws cannot exceed "
         f"log2(50) = 5.64 bits, and rarefaction shows\nthe share converges by n=50 while entropy "
         f"is still rising +0.07…+0.20 per ten examples, steepest where entropy is highest — so "
         f"the true gap is WIDER than drawn.   THE TRAINED NUMBER IS NOT SAFE TO QUOTE: its R=64 "
         f"cell returned\nonly 192 of 500 latents (the rest fired under delphi's min_examples=200), and truncating the uncontaminated 1e-3 wide cell to that same top 38% moves it 4.36 → 3.71 bits — 58% of the way to the observed 3.24.\nThe random arm barely moves under the same truncation (+0.04 bits at R=64, +0.18 at R=8).   NOTE PANEL B: at R=8 the two arms have\nessentially "
         f"the same entropy ({tn:.2f} vs {mn:.2f}) yet their fuzz AUROC differs by 0.206 (0.866 vs "
         f"0.660) — word-concentration explains the random arm's score and nothing about the "
         f"trained arm's.   pythia-1b layer 8 · delphi + Llama-3.1-70B",
         ha="center", fontsize=8.4, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   "pythia1b_L8_matched_entropy.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved {out}")
