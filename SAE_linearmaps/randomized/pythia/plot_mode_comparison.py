"""Plain top-k vs skip-embed, side by side, one panel per learning rate.

THE QUESTION THIS ANSWERS. Skip-embed is meant to be a DIAGNOSTIC: subtract the token-identity
component P[tok] and a metric that could not tell a trained transformer from a random one should
start to. The quantity that tests it is not either arm's AUROC but the DIFFERENCE OF DIFFERENCES

    dd = (trained - random | skip-embed) - (trained - random | plain top-k)

because any effect that hits both arms of a mode -- learning rate, target scale, dictionary
health -- cancels inside the within-mode gap. On gemma-2-2b this came out +0.051 +/- 0.016 on
fuzz (z=3.2) and a non-significant +0.011 on detection, so the effect was real but fuzz-specific.
This is the pythia-1b restatement, on the paper's own model family.

WHY TWO PANELS RATHER THAN ONE AVERAGED NUMBER. The plain run's two learning rates produced
opposite headline conclusions (-0.183 at 1e-3, +0.015 at 2e-3), and the reason was selection:
at 2e-3 the plain trained arm kept only 192 latents against the random arm's 410, so the arms
were not comparable populations. Averaging over LRs would bury exactly the instability that
makes the aggregate untrustworthy. Each panel is a separate experiment; agreement between them
is the evidence, not their mean.

READ THE n's ON THE BARS. They are the number of latents delphi could score (>= min_examples 200
firings), and they are a result in themselves: a mode or LR that scores far fewer latents is
reporting on a self-selected, more-active, easier-to-describe subsample. The plain 2e-3 cell's
192-vs-410 split is the cautionary case.

Bars start at 0.5 because that is chance for an AUROC; a bar from zero would make a 0.52 latent
look most of the way to a 0.85 one. Error bars are +/- 1 SE over LATENTS, the unit delphi
samples -- not over individual judge decisions, which would understate them several-fold by
treating ~100 correlated decisions per latent as independent.

    RESULTS_DIR=/dev/shm/delphi_run/results python -u plot_mode_comparison.py
    SCORER=detection python -u plot_mode_comparison.py
"""
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT   = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
LAYER  = int(os.environ.get("LAYER", 8))
LRS    = os.environ.get("LRS", "1e-3,2e-3").split(",")
SCORER = os.environ.get("SCORER", "fuzz")
MODES = [
    ("plain top-k SAE", os.environ.get("FULL_FMT", "pythia1b_{arm}_L{layer}_lr{lr}")),
    ("skip-embed SAE",  os.environ.get("RESID_FMT", "pythia1b_{arm}_resid_L{layer}_lr{lr}")),
]
ARMS = [("trained", "trained"), ("rand", "re-randomized")]

# Okabe-Ito. Colour means the ARM here, the same as in every other figure in this project --
# plot_firing_distribution.py, plot_frequency_gap.py -- so the two can sit on one page without
# the reader having to re-learn the encoding.
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARM_COLOUR = {"trained": BLUE, "rand": ORANGE}


def auroc(pos, neg):
    """Exact Mann-Whitney: P(pos > neg) + 0.5 P(pos == neg). Same statistic as report_auroc.py."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def per_latent(cell):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{SCORER}/*.txt")):
        if not re.findall(r"\d+", os.path.basename(f)):
            continue
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r["probability"] for r in recs if r["activating"] and r.get("probability") is not None]
        neg = [r["probability"] for r in recs if not r["activating"] and r.get("probability") is not None]
        a = auroc(pos, neg)
        if a is not None:
            out.append(a)
    return out


def stats(xs):
    if len(xs) < 2:
        return float("nan"), float("nan"), len(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


# ---------------------------------------------------------------- compute
D = {}
for lr in LRS:
    for mode_name, fmt in MODES:
        for arm, _ in ARMS:
            cell = fmt.format(arm=arm, layer=LAYER, lr=lr)
            vals = per_latent(cell)
            if not vals:
                raise SystemExit(f"no {SCORER} scores under {ROOT}/{cell}/scores/{SCORER}/")
            D[(lr, mode_name, arm)] = stats(vals)

print(f"\n{SCORER} AUROC, per-latent, clustered by latent (+/- 1 SE)\n")
dd_rows = []
for lr in LRS:
    print(f"  LR {lr}")
    gaps = {}
    for mode_name, _ in MODES:
        mt, set_, nt = D[(lr, mode_name, "trained")]
        mr, ser, nr = D[(lr, mode_name, "rand")]
        g, gse = mt - mr, math.sqrt(set_ ** 2 + ser ** 2)
        gaps[mode_name] = (g, gse)
        print(f"    {mode_name:<16} trained {mt:.3f}+/-{set_:.3f} (n={nt:>3})   "
              f"random {mr:.3f}+/-{ser:.3f} (n={nr:>3})   gap {g:+.3f}+/-{gse:.3f}")
    (gf, sf), (gr, sr) = gaps[MODES[0][0]], gaps[MODES[1][0]]
    dd, dse = gr - gf, math.sqrt(sf ** 2 + sr ** 2)
    dd_rows.append((lr, dd, dse))
    print(f"    {'difference of diffs':<16} {dd:+.3f} +/- {dse:.3f}   z = {dd / dse:+.2f}"
          f"   (skip-embed gap - plain gap)\n")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(LRS), figsize=(5.2 * len(LRS), 5.0), dpi=200, sharey=True)
axes = [axes] if len(LRS) == 1 else list(axes)

W = 0.34
for ax, lr in zip(axes, LRS):
    for i, (mode_name, _) in enumerate(MODES):
        for j, (arm, arm_label) in enumerate(ARMS):
            m, se, n = D[(lr, mode_name, arm)]
            x = i + (j - 0.5) * W
            # Bars grow from 0.5, the chance floor for an AUROC.
            ax.bar(x, m - 0.5, W * 0.92, bottom=0.5, color=ARM_COLOUR[arm], zorder=3,
                   label=arm_label if i == 0 else None)
            ax.errorbar(x, m, yerr=se, color=INK, lw=1.2, capsize=3, zorder=4)
            ax.text(x, m + se + 0.012, f"{m:.3f}", ha="center", fontsize=8.5, color=INK)
            ax.text(x, 0.508, f"n={n}", ha="center", fontsize=7.5, color="white", zorder=5)

        g = D[(lr, mode_name, "trained")][0] - D[(lr, mode_name, "rand")][0]
        gse = math.sqrt(D[(lr, mode_name, "trained")][1] ** 2 + D[(lr, mode_name, "rand")][1] ** 2)
        ax.text(i, 0.462, f"gap {g:+.3f}\n± {gse:.3f}", ha="center", fontsize=8.5,
                color=INK if abs(g) > 2 * gse else MUTED)

    ax.axhline(0.5, lw=1.2, color=INK, zorder=2)
    ax.set_xticks(range(len(MODES)), [m for m, _ in MODES], fontsize=10)
    ax.set_xlim(-0.62, len(MODES) - 0.38)
    ax.set_title(f"LR {lr}", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

axes[0].set_ylabel(f"{SCORER} AUROC  (0.5 = chance)", fontsize=10)
axes[0].set_ylim(0.44, 0.90)
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")

dd_txt = " · ".join(f"LR {lr}: {dd:+.3f} ± {dse:.3f} (z={dd/dse:+.1f})" for lr, dd, dse in dd_rows)
fig.suptitle("Does skip-embed restore the trained-vs-random distinction? "
             "pythia-1b layer 8, delphi + Llama-3.1-70B",
             fontsize=11.5, color=INK, y=1.02)
fig.text(0.5, -0.06,
         f"Difference of differences (skip-embed gap − plain gap) — {dd_txt}.\n"
         "Bars grow from chance (0.5); n is the number of latents delphi could score. "
         "Error bars ±1 SE over LATENTS, not over judge decisions.",
         ha="center", fontsize=8.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   f"pythia1b_L{LAYER}_mode_comparison_{SCORER}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
