"""THE CONTROLLED VERSION: token purity for the ONE pair that varies dictionary size and nothing else.

WHY THIS EXISTS. plot_token_purity.py pools four SAEs per arm at d_sae 131,072 against three at
16,384, and those pools are not matched: two of the four wide SAEs are skip-embed, two of the three
narrow ones come from a different codebase entirely (BatchTopK + Matryoshka, ctx 256, position-0
masked, lr 3e-4), and the learning rates differ across the whole set. So the pooled panels confound
dictionary size with architecture, codebase, LR, context length and p0 masking. They are evidence
that the pattern holds broadly; they are not a controlled comparison.

Exactly one pair in the project isolates d_sae:

    top-k R=64, lr 2e-3   d_sae 131,072   sparsify, per-token TopK, k=32, ctx 2048, no p0 mask
    top-k R=8,  lr 2e-3   d_sae  16,384   sparsify, per-token TopK, k=32, ctx 2048, no p0 mask

Both are the SAME recipe. R is the expansion factor and d_sae = R * d_model with d_model = 2048,
so R=64 gives 131,072 latents and R=8 gives 16,384. Nothing else differs -- which is why the old
labels ("plain top-k 2e-3" vs "sparsify R=8") were a mistake: they named one cell by architecture
and the other by codebase+width, making one recipe at two sizes read as two different things.

WHAT CHANGES WHEN YOU CONTROL. The random-arm collapse survives easily -- top-token share
0.525 -> 0.298, z = 14.1 -- so the headline holds. But the TRAINED arm is no longer flat: it drops
0.381 -> 0.298, z = 3.6, where the pooled figure showed z = -1.05. That flatness was an artifact of
LR composition, since the wide pool is half 1e-3 cells (trained purity ~0.20) and half 2e-3 (~0.35),
averaging to the narrow pool's value by coincidence.

WHY PANEL C IS NOT OPTIONAL. The trained side of this pair is the survivorship cell: its SAE
returned only 192 of 500 latents because the rest fired under delphi's min_examples=200, so the
192 scored are the highest-firing third. Panel C shows why that matters -- purity rises with
firing rate on the trained arm about 5x faster than on the random one over the range in question
(0.203 -> 0.323 vs 0.525 -> 0.547 down to the top 38%), so a top-third sample is inflated by
construction. That covers most of the distance to the 0.381 actually observed, so the trained
arm's apparent drop is probably smaller than 0.083 and may be nothing at all: this figure must
NOT be used to claim the trained arm moves. Both random cells rise under truncation too -- do not
call them flat -- but they barely needed truncating (82% and 96% survival) and rise far less.

THE OTHER THING PANEL B SAYS, and it is not incidental: at d_sae 16,384 the two arms have
IDENTICAL purity, 0.298 vs 0.298, while their fuzz AUROC differs by 0.206 (0.866 vs 0.660). So
word-concentration explains the RANDOM arm's score and explains nothing at all about the trained
arm's. Whatever makes a trained SAE's latents scoreable, it is not this.

    RESULTS_DIR=delphi_results python3 -u plot_matched_purity.py
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

WIDE, NARROW = "top-k R64 2e-3", "top-k R8 2e-3"
CELLFILE = {("top-k R64 2e-3", "trained"): "pythia1b_trained_L8_lr2e-3",
            ("top-k R64 2e-3", "rand"): "pythia1b_rand_L8_lr2e-3",
            ("top-k R8 2e-3", "trained"): "pythia1b_trained_R8_L8_lr2e-3",
            ("top-k R8 2e-3", "rand"): "pythia1b_rand_R8_L8_lr2e-3",
            ("top-k R64 1e-3", "trained"): "pythia1b_trained_L8_lr1e-3"}
NB = 16
EDGES = [i / NB for i in range(NB + 1)]


def shares(cell, arm):
    return [x[2] for x in tp.data[(cell, arm)].values()]


def hist_pct(v):
    h = [0] * NB
    for x in v:
        h[min(NB - 1, int(x * NB))] += 1
    return [100 * c / len(v) for c in h]


def firing_curve(cell, arm, qs):
    """Mean purity when only the top q fraction of latents BY FIRING RATE is kept."""
    p = f"{ROOT}/{CELLFILE[(cell, arm)]}/log/hookpoint_firing_counts.pt"
    o = t.load(p, weights_only=True, map_location="cpu")
    c = (list(o.values())[0] if isinstance(o, dict) else o).tolist()
    rows = sorted(((c[lid], v[2]) for lid, v in tp.data[(cell, arm)].items()
                   if lid < len(c) and c[lid] > 0), key=lambda r: -r[0])
    out = []
    for q in qs:
        k = max(2, round(q * len(rows)))
        out.append(tp.mean_se([s for _, s in rows[:k]])[0])
    return out, len(rows)


fig = plt.figure(figsize=(15.2, 6.3), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[5, 4], hspace=0.46, wspace=0.20)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])

for ax, cell, tag in ((axA, WIDE, "A   R=64  →  d_sae 131,072"),
                      (axB, NARROW, "B   R=8  →  d_sae 16,384   —   same recipe, only the expansion factor differs")):
    for arm in ("trained", "rand"):
        v = shares(cell, arm)
        m, se = tp.mean_se(v)
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], lw=2.0, zorder=4,
                  label=f"{ARM_LABEL[arm]}  mean {m:.3f}  (n={len(v)})")
        ax.stairs(hist_pct(v), EDGES, color=ARM_COLOUR[arm], fill=True, alpha=0.20, zorder=3)
    ax.set_title(tag, fontsize=10.5, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 34)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("% of latents", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right", handlelength=1.4)

axB.set_xlabel("how often a latent's 50 activating examples peak on the SAME word\n"
               "0 = a different word every time                    1 = a pure single-word detector",
               fontsize=9.5, color=INK)
axA.tick_params(labelbottom=False)

# ---------------------------------------------------------------- panel C, the survivorship control
QS = [1.00, 0.75, 0.50, 0.38, 0.25, 0.15]
DASH = (0, (5, 2))
CURVES = [("top-k R64 1e-3", "trained", BLUE, "-", "trained, WIDE (1e-3, 79% survived)"),
          ("top-k R8 2e-3", "trained", BLUE, DASH, "trained, narrow (91% survived)"),
          ("top-k R64 2e-3", "rand", ORANGE, "-", "random, WIDE (82% survived)"),
          ("top-k R8 2e-3", "rand", ORANGE, DASH, "random, narrow (96% survived)")]
for cell, arm, col, ls, lab in CURVES:
    y, n = firing_curve(cell, arm, QS)
    axC.plot([100 * q for q in QS], y, color=col, ls=ls, lw=2.0, marker="o", ms=5,
             zorder=4, label=lab)

# The observed value of the contaminated cell, and where selection alone would put it.
obs = tp.mean_se(shares(WIDE, "trained"))[0]
axC.scatter([38], [obs], s=150, marker="*", color=BLUE, zorder=6,
            edgecolor="white", linewidth=0.8)
axC.annotate("R=64 trained actually scored\n0.381 — but only its top 38%\nby firing rate was ever\n"
             "scored (192 of 500)",
             xy=(38, obs), xytext=(60, 0.437), ha="left", va="center", fontsize=8.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                             shrinkA=2, shrinkB=8, connectionstyle="arc3,rad=0.2"))
axC.invert_xaxis()
axC.set_xlabel("keep only the top % of latents by firing rate", fontsize=9.5)
axC.set_ylabel("mean top-token share", fontsize=9.5)
axC.set_title("C   why the trained arm's drop is not trustworthy\n"
              "      over the range that matters the trained arm rises 5× faster than the random one",
              fontsize=10.5, color=INK, loc="left")
axC.grid(color=GRID, lw=0.6, zorder=0)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
axC.tick_params(colors=MUTED, labelsize=9)
axC.legend(frameon=False, fontsize=8.4, loc="upper left", handlelength=3.4)

mw, sw = tp.mean_se(shares(WIDE, "rand"))
mn, sn = tp.mean_se(shares(NARROW, "rand"))
tw, tws = tp.mean_se(shares(WIDE, "trained"))
tn, tns = tp.mean_se(shares(NARROW, "trained"))
zr = (mw - mn) / math.sqrt(sw ** 2 + sn ** 2)
zt = (tw - tn) / math.sqrt(tws ** 2 + tns ** 2)

fig.suptitle("Controlled for everything but dictionary size: shrinking it collapses the RANDOM "
             "model's single-word latents", fontsize=12.5, color=INK, y=1.02)
fig.text(0.5, -0.115,
         f"The only pair in this project that varies d_sae alone — both are stock sparsify plain "
         f"top-k, k=32, 100M tokens, lr 2e-3, ctx 2048, no position-0 mask, same corpus and same "
         f"scoring config; R is the expansion factor and d_sae = R × 2048.\nRandom arm {mw:.3f} → {mn:.3f} (z = {zr:.1f}); trained arm "
         f"{tw:.3f} → {tn:.3f} (z = {zt:.1f}).   THE TRAINED NUMBER IS NOT SAFE TO QUOTE: its wide "
         f"cell returned only 192 of 500 latents (the rest fired under delphi's\n"
         f"min_examples=200), so it is a top-third-by-firing sample, and panel C shows purity rises "
         f"steeply with firing rate on the trained arm — truncating the uncontaminated 1e-3 cell to "
         f"the same 38% moves it 0.203 → 0.323,\ncovering most of the gap to 0.381. The random arm rises too but 5× less over that range (0.525 → 0.547), and it barely needed truncating: 82% and 96% survival.\nNOTE PANEL B: at 16,384 the two arms have IDENTICAL purity (0.298 vs 0.298) yet their fuzz AUROC differs by 0.206 (0.866 vs 0.660) — so word-concentration does not explain the trained arm's score at all.\n   Compare plot_token_purity.py, which pools four "
         f"SAEs per arm against "
         f"three and so confounds d_sae with architecture, codebase, LR, context length and p0 "
         f"masking — broader evidence, weaker control.   pythia-1b layer 8 · delphi + Llama-3.1-70B",
         ha="center", fontsize=8.4, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   "pythia1b_L8_matched_purity.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"\nsaved {out}")
