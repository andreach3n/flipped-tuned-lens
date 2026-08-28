"""THE FIGURE: three SAE architectures on the same model, and the control that undercuts one.

WHAT THIS SHOWS. On pythia-1b layer 8 a plain top-k SAE and a skip-embed SAE both score the
RANDOMIZED transformer as more interpretable than the trained one -- a reversal of Heap et al.,
who report the two as indistinguishable. A temporal SAE flips the sign: trained 0.801 vs random
0.626. That is the first architecture here that separates the arms in the direction the whole
diagnostic programme assumes.

WHY PANEL B EXISTS, AND WHY IT IS NOT AN AFTERTHOUGHT. The T-SAE is a Matryoshka SAE with a
contrastive temporal loss applied to its HIGH-LEVEL group (latents [0..3275]); the low-level group
[3276..] carries no such term. If the flip were caused by the temporal loss, the high-level half
should separate the arms and the low-level half should not. Scoring 500 latents from each group --
via a column permutation in delphi_tsae.py, since delphi's only selection mechanism is
torch.arange(max_latents) -- gives +0.176 and +0.175. Indistinguishable. So the effect belongs to
something the whole dictionary shares (BatchTopK vs per-token TopK, Matryoshka, d_sae 16384 vs
131072, their trainer, or the position-0 masking) and NOT to the contrastive term.

Reporting panel A without panel B would be the single most misleading thing this project could
publish, which is why they are one figure rather than two.

Bars grow from 0.5 because that is chance for an AUROC; from zero, a 0.52 latent would look most
of the way to a 0.85 one. Error bars are +/- 1 SE over LATENTS, the unit delphi samples -- not
over individual judge decisions, which would understate them several-fold by treating ~100
correlated decisions per latent as independent.

Colour means the ARM, the same as in every other figure in this project. Okabe-Ito, validated:
adjacent-pair CVD separation dE 29.2 (protan) / 30.9 (tritan), normal-vision 36.2, all far above
the 8 floor. The orange sits below 3:1 against the surface, so every bar carries a visible value
label -- that is the required relief, not decoration.

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
LR     = os.environ.get("LR", "1e-3")          # the clean cell for plain / skip-embed
SPLIT  = int(os.environ.get("SPLIT", 500))     # delphi latent < SPLIT is high-level

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}
ARMS = [("trained", "trained"), ("rand", "re-randomized")]

# BOTH learning rates for plain and skip-embed, not just the clean one. At 2e-3 the plain gap is
# +0.015 -- i.e. Heap et al. REPLICATING -- and showing only 1e-3's -0.183 would be displaying the
# most anti-Heap cell available. The 2e-3 plain cell is separately known to be contaminated (192
# trained latents vs 410 random, a selection artifact), but that belongs in the caption, not in a
# decision about which bars to draw. The temporal result sits outside both, so nothing is lost.
#
# The LR is in every label because the temporal SAE runs at 3e-4 -- hardcoded in their trainer,
# line 182 -- so it is not LR-matched to anything here, and that should be visible rather than
# buried.
ARCHS = [
    ("plain top-k\n1e-3",  f"pythia1b_{{arm}}_L{LAYER}_lr1e-3"),
    ("plain top-k\n2e-3",  f"pythia1b_{{arm}}_L{LAYER}_lr2e-3"),
    ("skip-embed\n1e-3",   f"pythia1b_{{arm}}_resid_L{LAYER}_lr1e-3"),
    ("skip-embed\n2e-3",   f"pythia1b_{{arm}}_resid_L{LAYER}_lr2e-3"),
    ("temporal\n3e-4",     f"pythia1b_{{arm}}_tsae_L{LAYER}"),
]
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
for name, fmt in ARCHS:
    for arm, _ in ARMS:
        A[(name, arm)] = stats(list(per_latent(fmt.format(arm=arm)).values()))

B = {}                                    # panel B: (group, arm) -> (mean, se, n)
for arm, _ in ARMS:
    d = per_latent(TSAE.format(arm=arm))
    B[("high-level", arm)] = stats([v for k, v in d.items() if k < SPLIT])
    B[("low-level", arm)] = stats([v for k, v in d.items() if k >= SPLIT])

print(f"\n{SCORER} AUROC, per-latent, clustered by latent (+/- 1 SE)\n")
for label, D, keys in (("A  architectures", A, [n for n, _ in ARCHS]),
                       ("B  temporal SAE, by Matryoshka group", B, ["high-level", "low-level"])):
    print(f"  {label}")
    for k in keys:
        mt, st, nt = D[(k, "trained")]
        mr, sr, nr = D[(k, "rand")]
        g, gse = mt - mr, math.sqrt(st ** 2 + sr ** 2)
        print(f"    {k:<12} trained {mt:.3f}+/-{st:.3f} (n={nt:>3})  "
              f"random {mr:.3f}+/-{sr:.3f} (n={nr:>3})  gap {g:+.3f}+/-{gse:.3f}  z={g/gse:6.2f}")
    print()

# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(14.0, 5.6), dpi=200, sharey=True,
    gridspec_kw={"width_ratios": [5, 2], "wspace": 0.06})

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


draw(axA, A, [n for n, _ in ARCHS], "A   three SAE architectures, both learning rates where available")
draw(axB, B, ["high-level", "low-level"],
     "B   temporal SAE, split by Matryoshka group")
axB.spines["left"].set_visible(False)
axB.tick_params(left=False)

axA.set_ylabel(f"{SCORER} AUROC   (0.5 = chance)", fontsize=10)
axA.set_ylim(0.44, 0.90)
axA.legend(frameon=False, fontsize=10, loc="upper left")

fig.suptitle("A temporal SAE flips the trained-vs-random gap on pythia-1b — but not because of "
             "the temporal loss",
             fontsize=12, color=INK, y=1.00, x=0.5)
fig.text(0.5, -0.07,
         "Panel B is the control: the contrastive temporal loss acts only on the high-level "
         "group, yet both groups separate the arms equally (+0.176 vs +0.175).\n"
         "The plain 2e-3 cell is the one where Heap et al. replicates (gap +0.015) — but it "
         "scored 192 trained latents against 410 random, a selection artifact.\n"
         "So the flip comes from something the whole dictionary shares — BatchTopK vs per-token "
         "TopK, Matryoshka, d_sae 16384 vs 131072, or the trainer — not from temporality.\n"
         "pythia-1b layer 8 · delphi + Llama-3.1-70B · per-latent AUROC, ±1 SE over latents · "
         "30M scoring tokens",
         ha="center", fontsize=8.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   f"pythia1b_L{LAYER}_arch_comparison_{SCORER}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
