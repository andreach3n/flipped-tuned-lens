"""Figure: how often does a latent actually fire? The frequency distribution behind min_examples=200.

WHY THIS EXISTS. delphi drops any latent with fewer than 200 activating examples, and ~20% of a
73,728-latent dictionary fails that over 30M tokens — which is surprising until you look at the
distribution, because the MEAN firing count is 13,020. That mean is arithmetic, not a fact about
the SAE: top-k fixes the total number of firings at k * n_tokens, so every cell has the identical
mean by construction (32 * 29,999,104 / 73,728 = 13,020). Everything informative lives in the
shape, and the shape is extreme — medians run 8-35x below the mean, while the single most active
latent in the trained/plain cell fires on 99.4% of all tokens.

A log x-axis is not a stylistic choice here. The data spans ~1 to ~3e7; on a linear axis every cell
collapses into the first bin and the figure says nothing.

Counts come from `log/hookpoint_firing_counts.pt`, which the cache writer accumulates over the FULL
dictionary regardless of MAX_LATENTS — so this describes all 73,728 latents, not just the 500
delphi scored. That is what makes it usable as a check on whether `--max_latents 500` (which is
`arange(500)`, the first 500 indices) draws a representative sample.

    RESULTS_DIR=<dir with the extracted result tarballs> python randomized/plot_firing_distribution.py
"""
import os

import numpy as np
import torch as t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
MIN_EXAMPLES = 200

# Okabe-Ito, the project's pair, reused with the SAME meaning as the 2x2 figure: colour is the ARM.
# Validated for categorical use (adjacent CVD dE 29.2 protan / 30.9 tritan, normal-vision 36.2).
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
ARMS = [("trained", BLUE, "trained gemma"), ("rand", ORANGE, "randomized gemma")]
PANELS = [("full", "plain top-k SAE"), ("resid", "skip-embed SAE")]


def counts(cell):
    p = f"{ROOT}/{cell}/log/hookpoint_firing_counts.pt"
    if not os.path.exists(p):
        return None
    return list(t.load(p, weights_only=True).values())[0].float().numpy()


data = {}
for arm, _, _ in ARMS:
    for sae, _ in PANELS:
        c = counts(f"{arm}_{sae}")
        if c is not None:
            data[(arm, sae)] = c
if not data:
    raise SystemExit(f"no hookpoint_firing_counts.pt under {ROOT!r} -- set RESULTS_DIR")

BINS = np.logspace(0, np.log10(3e7), 61)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=200, sharey=True)

for ax, (sae, sae_label) in zip(axes, PANELS):
    for arm, colour, arm_label in ARMS:
        c = data.get((arm, sae))
        if c is None:
            continue
        # log bins cannot show zero; report the dead separately rather than silently dropping them.
        dead = int((c == 0).sum())
        ax.hist(np.clip(c, 1, None), bins=BINS, histtype="step", lw=2, color=colour,
                label=f"{arm_label}  (median {int(np.median(c)):,}"
                      + (f", {dead} dead)" if dead else ")"))

    ax.axvline(MIN_EXAMPLES, color=INK, lw=1.2, ls="--", zorder=5)

    # One annotation block in the empty left region, rather than a label beside the line: an
    # inline label at the top of the axes lands underneath the legend, which is only visible once
    # the figure is rendered.
    lines = []
    for arm, _, arm_label in ARMS:
        c = data.get((arm, sae))
        if c is not None:
            lines.append(f"  {arm_label.split()[0]:<10s} {(c < MIN_EXAMPLES).mean() * 100:.1f}%")
    ax.text(0.02, 0.72, f"dashed line: delphi's min_examples = {MIN_EXAMPLES}\n"
                        f"latents dropped before explanation:\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top", linespacing=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("times the latent fired across 30M tokens")
    ax.set_title(sae_label, fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

axes[0].set_ylabel("number of latents")

fig.suptitle("Latent firing frequency, gemma-2-2b layer 13 — all 73,728 latents per SAE",
             fontsize=11.5, color=INK, y=1.0)
fig.text(0.5, -0.02,
         "Log x-axis: firing counts span 1 to ~3e7. The MEAN is 13,020 in every cell by "
         "construction (top-k fixes total firings at k x n_tokens),\nso only the shape is "
         "informative — and medians run 8-35x below that mean. Latents left of the dashed line are "
         "dropped by delphi before explanation.",
         ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
fig.tight_layout()

out = os.path.join(HERE, "plots", "firing_distribution_L13.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}\n")

print(f"{'cell':>15} {'median':>10} {'p90':>12} {'max':>12} {'max % of tokens':>16} {'<200':>8}")
for arm, _, _ in ARMS:
    for sae, _ in PANELS:
        c = data.get((arm, sae))
        if c is None:
            continue
        print(f"{arm + '_' + sae:>15} {int(np.median(c)):>10,} {int(np.percentile(c, 90)):>12,} "
              f"{int(c.max()):>12,} {c.max() / 29999104 * 100:>15.1f}% "
              f"{(c < MIN_EXAMPLES).mean() * 100:>7.1f}%")
