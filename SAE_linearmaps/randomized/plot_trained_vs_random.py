"""The Tier-1 headline figure: real SAE FVU vs its matched-Gaussian floor, trained vs random.

One bar pair per (arm x SAE): the solid bar is the real SAE's FVU, the hatched bar is the FVU of an
IDENTICAL SAE trained on N(mu, Sigma) fitted to the same target -- the "no structure beyond
covariance" floor. The annotation is 1 - real/null: how far below its own floor each SAE lands.
The story is the right-most pair: on a random transformer, the skip-embed SAE sits ON its floor.

Numbers are the verified 2026-08-04 results (centred convention, 2M held-out tokens, k=64,
20M-token SAEs, seed-0 rand_all). Provenance -- per-arm HF repos:
  fvu_{arm}_s0_k64.json           <- eval_fvu.py    (…-trained-20m / …-rand-all-s0)
  gauss_null_{arm}_s0_{mode}_k64.json <- gauss_null.py
Run:  python plot_trained_vs_random.py   (writes plots/trained_vs_random_fvu.png)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# (arm label, SAE label, real centred FVU, null centred FVU)
DATA = [
    ("trained", "plain\n(topk SAE)",        0.2535, 0.4789),
    ("trained", "skip-embed\n(resid SAE)",  0.3337, 0.5318),
    ("random",  "plain\n(topk SAE)",        0.5344, 0.6250),
    ("random",  "skip-embed\n(resid SAE)",  0.6694, 0.6486),
]

BLUE  = "#0072B2"   # Okabe-Ito blue (project standard) -- the real SAE
GRAY  = "#5B5B5B"   # baseline ink; hatch = secondary encoding so identity is not color-alone
INK, MUTED = "#1a1a1a", "#666666"

fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
centers = [0.0, 1.0, 2.5, 3.5]          # extra gap separates the trained / random clusters
W = 0.34

for cx, (arm, sae, real, null) in zip(centers, DATA):
    ax.bar(cx - W/2, real, W, color=BLUE, zorder=3)
    ax.bar(cx + W/2, null, W, facecolor="white", edgecolor=GRAY,
           hatch="///", linewidth=1.0, zorder=3)
    # value labels at each bar end (muted -- selective, small)
    ax.text(cx - W/2, real + 0.012, f"{real:.2f}", ha="center", va="bottom",
            fontsize=8, color=MUTED)
    ax.text(cx + W/2, null + 0.012, f"{null:.2f}", ha="center", va="bottom",
            fontsize=8, color=MUTED)
    # the statistic: % below own floor
    pct = 100 * (1 - real / null)
    label = f"−{pct:.0f}%" if pct > 0 else f"+{-pct:.1f}% (≈ 0)"
    ax.text(cx, max(real, null) + 0.055, label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=INK)

ax.set_xticks(centers)
ax.set_xticklabels([d[1] for d in DATA], fontsize=9, color=INK)
for x, lbl in [(0.5, "trained Gemma-2-2b"), (3.0, "re-randomized (incl. embeddings, seed 0)")]:
    ax.text(x, -0.135, lbl, transform=ax.get_xaxis_transform(),
            ha="center", fontsize=10, fontweight="bold", color=INK)

ax.set_ylabel("FVU on target (centred)  —  lower = more structure found", fontsize=9, color=INK)
ax.set_ylim(0, 0.82)
ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(colors=MUTED, labelsize=8)

ax.set_title("How far does each SAE beat its own noise floor?",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=28)
ax.text(0, 1.06, "% = FVU reduction vs an identical SAE trained on N(μ, Σ) matched to the same target "
                 "(layer 13, k=64, 20M-token SAEs, 2M held-out tokens)",
        transform=ax.transAxes, fontsize=8, color=MUTED)

handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
           plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY, hatch="///")]
ax.legend(handles, ["real SAE", "matched-Gaussian null (floor)"],
          loc="upper left", frameon=False, fontsize=8.5)

PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS, exist_ok=True)
out = f"{PLOTS}/trained_vs_random_fvu.png"
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
