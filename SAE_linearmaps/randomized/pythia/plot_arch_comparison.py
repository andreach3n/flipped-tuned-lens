"""THE FIGURE: the trained-vs-random gap is set by DICTIONARY SIZE, not by the SAE codebase.

WHAT THIS SHOWS, AND WHAT IT REPLACES. The earlier version of this figure said the flip came from
switching codebases -- sparsify's plain top-k scored the RANDOMIZED pythia-1b as more interpretable
than the trained one, while the temporal/non-temporal SAEs from AI4LIFE-GROUP scored it the other
way. Six things differed between those two settings at once (BatchTopK vs per-token TopK,
Matryoshka, d_sae 16384 vs 131072, ctx 256, position-0 masking, the trainer itself) and the
non-temporal control only ruled out the seventh, the contrastive loss.

Training a sparsify top-k SAE at d_sae=16384 collapses all six to one. That run has none of the
other five properties -- stock sparsify, per-token TopK, no Matryoshka, ctx 2048, no position-0
mask -- and it flips harder than the temporal SAE did (+0.206). Sorting the panel by d_sae makes
the pattern unmissable: the RANDOM arm sits at 0.808-0.826 in all four wide cells and at
0.626-0.660 in all three narrow ones, across two independent codebases, while the trained arm
stays at 0.80-0.87 wherever the SAE is not degenerate. Dictionary size moves the random
transformer's apparent interpretability by ~0.17 AUROC; nothing else here moves it at all.

WHY PANEL B STAYS. It was the control that killed the temporal explanation and it is still the
cleanest single refutation of it: the contrastively-regularised Matryoshka group and the
unregularised one separate the arms equally (+0.176 vs +0.175).

THE ASTERISKED BAR. plain 2e-3 is the one cell that reproduces Heap et al. (+0.015), and it is
not trustworthy: its trained SAE returned only 192 of 500 latents, because the other 308 fired too
rarely to build windows (median 29 firings against 1,937 for the survivors). Since AUROC rises
with firing rate in the trained arm, scoring only the healthiest third inflates it -- truncating
the plain 1e-3 SAE to the same 38% moves it 0.626 -> 0.779, and to 25% moves it to 0.855, while
the random arm is flat under identical truncation (0.808 -> 0.804). So that bar is survivorship,
and it is marked rather than dropped because it is the cell the literature would quote.

Bars grow from 0.5 because that is chance for an AUROC. Error bars are +/- 1 SE over LATENTS --
the unit delphi samples. That is the WRONG denominator for a claim about trained-vs-random, which
needs a between-SAE component this project does not yet have; see the caption.

Colour means the ARM, as in every other figure here. Okabe-Ito, validated: adjacent-pair CVD
separation dE 29.2 (protan) / 30.9 (tritan), normal-vision 36.2. The orange is below 3:1 against
the surface, so every bar carries a visible value label -- required relief, not decoration.

    RESULTS_DIR=delphi_results python3 -u plot_arch_comparison.py
"""
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT   = os.environ.get("RESULTS_DIR", "delphi_results")
SCORER = os.environ.get("SCORER", "fuzz")
LAYER  = int(os.environ.get("LAYER", 8))
SPLIT  = int(os.environ.get("SPLIT", 500))     # delphi latent < SPLIT is high-level

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARMS = [("trained", "trained"), ("rand", "re-randomized")]

# Ordered by dictionary size, which is the axis the result lives on. Within a block, ordered by
# codebase then LR. The LR is in every label because the temporal and non-temporal runs are stuck
# at 3e-4 (hardcoded in their trainer, line 182), so they are not LR-matched to the sparsify cells
# and that should be visible rather than buried.
ARCHS = [
    ("plain top-k\n1e-3",   f"pythia1b_{{arm}}_L{LAYER}_lr1e-3",       131072),
    ("plain top-k\n2e-3 *", f"pythia1b_{{arm}}_L{LAYER}_lr2e-3",       131072),
    ("skip-embed\n1e-3",    f"pythia1b_{{arm}}_resid_L{LAYER}_lr1e-3", 131072),
    ("skip-embed\n2e-3",    f"pythia1b_{{arm}}_resid_L{LAYER}_lr2e-3", 131072),
    ("plain top-k\n2e-3",   f"pythia1b_{{arm}}_R8_L{LAYER}_lr2e-3",     16384),
    ("temporal\n3e-4",      f"pythia1b_{{arm}}_tsae_L{LAYER}",          16384),
    ("non-temporal\n3e-4",  f"pythia1b_{{arm}}_base_L{LAYER}",          16384),
]
NWIDE = sum(1 for *_, d in ARCHS if d == 131072)      # where the d_sae divider goes
TSAE = f"pythia1b_{{arm}}_tsae_L{LAYER}"


def auroc(pos, neg):
    """Exact Mann-Whitney: P(pos > neg) + 0.5 P(pos == neg). Same statistic as report_auroc.py."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def per_latent(cell):
    """{latent id: AUROC} for one cell."""
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
    if not out:
        raise SystemExit(f"no {SCORER} scores under {ROOT}/{cell}/scores/{SCORER}/")
    return out


def stats(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


# ---------------------------------------------------------------- compute
A = {}                                    # panel A: (arch, arm) -> (mean, se, n)
for name, fmt, _ in ARCHS:
    for arm, _ in ARMS:
        A[(name, arm)] = stats(list(per_latent(fmt.format(arm=arm)).values()))

B = {}                                    # panel B: (group, arm) -> (mean, se, n)
for arm, _ in ARMS:
    d = per_latent(TSAE.format(arm=arm))
    B[("high-level", arm)] = stats([v for k, v in d.items() if k < SPLIT])
    B[("low-level", arm)] = stats([v for k, v in d.items() if k >= SPLIT])

print(f"\n{SCORER} AUROC, per-latent, clustered by latent (+/- 1 SE)\n")
for label, D, keys in (("A  architectures, by dictionary size", A, [n for n, *_ in ARCHS]),
                       ("B  temporal SAE, by Matryoshka group", B, ["high-level", "low-level"])):
    print(f"  {label}")
    for k in keys:
        mt, st, nt = D[(k, "trained")]
        mr, sr, nr = D[(k, "rand")]
        g, gse = mt - mr, math.sqrt(st ** 2 + sr ** 2)
        print(f"    {k.replace(chr(10), ' '):<18} trained {mt:.3f}+/-{st:.3f} (n={nt:>3})  "
              f"random {mr:.3f}+/-{sr:.3f} (n={nr:>3})  gap {g:+.3f}+/-{gse:.3f}  z={g/gse:6.2f}")
    print()

# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(17.2, 6.0), dpi=200, sharey=True,
    gridspec_kw={"width_ratios": [7, 2], "wspace": 0.05})

W = 0.32


def draw(ax, D, keys, title):
    for i, key in enumerate(keys):
        for j, (arm, arm_label) in enumerate(ARMS):
            m, se, n = D[(key, arm)]
            x = i + (j - 0.5) * W
            # 2px surface gap between adjacent bars: width*0.88 leaves it at this scale.
            ax.bar(x, m - 0.5, W * 0.88, bottom=0.5, color=ARM_COLOUR[arm], zorder=3,
                   label=arm_label if i == 0 and ax is axA else None)
            ax.errorbar(x, m, yerr=se, color=INK, lw=1.2, capsize=3, zorder=4)
            # Value label on every bar -- the required relief for the orange's sub-3:1 contrast.
            ax.text(x, m + se + 0.010, f"{m:.3f}", ha="center", fontsize=9, color=INK)
            ax.text(x, 0.508, f"n={n}", ha="center", fontsize=7.5, color="white", zorder=5)

        mt, st, _ = D[(key, "trained")]
        mr, sr, _ = D[(key, "rand")]
        g, gse = mt - mr, math.sqrt(st ** 2 + sr ** 2)
        ax.text(i, 0.452, f"gap {g:+.3f}\n± {gse:.3f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold" if abs(g) > 5 * gse else "normal")

    ax.axhline(0.5, lw=1.3, color=INK, zorder=2)
    ax.set_xticks(range(len(keys)), keys, fontsize=9.5)
    ax.set_xlim(-0.62, len(keys) - 0.38)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


draw(axA, A, [n for n, *_ in ARCHS], "")
draw(axB, B, ["high-level", "low-level"], "")
axB.spines["left"].set_visible(False)
axB.tick_params(left=False)

# The d_sae divider -- the only structure in panel A that matters. Headers sit ABOVE the axes in
# blended coords (x in data, y in axes) so they cannot collide with the tallest bars' value labels,
# which is exactly what happened when they lived at y=0.878 in data space.
axA.axvline(NWIDE - 0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
blend = matplotlib.transforms.blended_transform_factory(axA.transData, axA.transAxes)
for lo, hi, txt in ((-0.5, NWIDE - 0.5, "d_sae 131,072   (expansion 64×)  ·  sparsify"),
                    (NWIDE - 0.5, len(ARCHS) - 0.5,
                     "d_sae 16,384   (8×)  ·  sparsify | their trainer")):
    axA.text((lo + hi) / 2, 1.015, txt, ha="center", va="bottom", fontsize=9.5, color=MUTED,
             transform=blend, clip_on=False,
             fontweight="bold" if "16,384" in txt else "normal")

axA.text(0.0, 1.075, "A   every SAE trained on pythia-1b layer 8, ordered by dictionary size",
         transform=axA.transAxes, fontsize=10.5, color=INK, ha="left")
axB.text(0.0, 1.075, "B   temporal SAE, split by Matryoshka group",
         transform=axB.transAxes, fontsize=10.5, color=INK, ha="left")

axA.set_ylabel(f"{SCORER} AUROC   (0.5 = chance)", fontsize=10)
axA.set_ylim(0.44, 0.90)
axA.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.0, 0.97))

fig.suptitle("Dictionary size — not the codebase, not the temporal loss — decides whether delphi "
             "separates a trained transformer from a random one",
             fontsize=12.5, color=INK, y=1.09, x=0.5)
fig.text(0.5, -0.10,
         "The re-randomized arm scores 0.808–0.826 in all four d_sae=131,072 cells and 0.626–0.660 "
         "in all three d_sae=16,384 cells, across two independent codebases; the trained arm stays "
         "0.80–0.87 wherever the SAE is not degenerate.\nThe narrow sparsify run is stock "
         "per-token TopK with no Matryoshka, no BatchTopK, no contrastive term, ctx 2048 and no "
         "position-0 mask, so it rules out every difference the non-temporal control left open. "
         "Panel B rules out the last one:\nthe contrastively-regularised Matryoshka half and the "
         "unregularised half separate the arms equally (+0.176 vs +0.175).   "
         "* plain top-k 2e-3 is the one cell reproducing Heap et al. (+0.015) and is survivorship: "
         "its trained SAE\nreturned 192/500 latents (dropped ones fired a median 29 times vs 1,937 "
         "for survivors), and truncating the 1e-3 SAE to the same 38% moves it 0.626→0.779 while "
         "the random arm is unmoved.   CAVEAT: error bars are ±1 SE over latents\nwithin a single "
         "SAE per arm — they do not contain SAE-seed or randomization-seed variance, so they "
         "understate the uncertainty on every gap shown.   pythia-1b layer 8 · delphi + "
         "Llama-3.1-70B · 30M scoring tokens",
         ha="center", fontsize=8.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   f"pythia1b_L{LAYER}_arch_comparison_{SCORER}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
