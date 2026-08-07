"""Companion to plot_trained_vs_random.py: every system reconstructs the SAME target h.

The skip-embed column here is the COMPOSITE system (frozen map + resid SAE, reassembled as
P[tok] + r_hat), so the blue bars are directly comparable across all four groups -- unlike the
main figure, where plain is scored on h and skip-embed on r.

The composite's floor is DERIVED, not separately trained: eval_fvu.py verifies the exact identity
    FVU_h(composite) = FVU_r(resid SAE) * Var(r)/Var(h)   [centred SS ratio]
so the floor for "map + noise-level SAE" is  null_r * SS_r/SS_h.  Because real and null are scaled
by the same ratio, the %-below-null annotations are IDENTICAL to the residual-space figure --
the statistic is invariant to which space you measure in.

Numbers: 2026-08-04 verified results (centred), same provenance as plot_trained_vs_random.py,
plus var_ratio_r_over_h (centred) from fvu_{arm}_s0_k64.json: trained 0.7477, rand_all 0.7740.
Run:  python plot_trained_vs_random_h.py   (writes plots/trained_vs_random_fvu_h.png)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# (arm, SAE label, real FVU on h, null FVU on h)
# composite nulls: 0.5318*0.7477 = 0.3976 (trained), 0.6486*0.7740 = 0.5020 (rand_all)
DATA = [
    ("trained", "plain\n(topk SAE)",              0.2535, 0.4789),
    ("trained", "skip-embed\n(map + resid SAE)",  0.2495, 0.3976),
    ("random",  "plain\n(topk SAE)",              0.5344, 0.6250),
    ("random",  "skip-embed\n(map + resid SAE)",  0.5182, 0.5020),
]

BLUE  = "#0072B2"
GRAY  = "#5B5B5B"
INK, MUTED = "#1a1a1a", "#666666"

fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
centers = [0.0, 1.0, 2.5, 3.5]
W = 0.34

for cx, (arm, sae, real, null) in zip(centers, DATA):
    ax.bar(cx - W/2, real, W, color=BLUE, zorder=3)
    ax.bar(cx + W/2, null, W, facecolor="white", edgecolor=GRAY,
           hatch="///", linewidth=1.0, zorder=3)
    ax.text(cx - W/2, real + 0.012, f"{real:.2f}", ha="center", va="bottom",
            fontsize=8, color=MUTED)
    ax.text(cx + W/2, null + 0.012, f"{null:.2f}", ha="center", va="bottom",
            fontsize=8, color=MUTED)
    pct = 100 * (1 - real / null)
    label = f"−{pct:.0f}%" if pct > 0 else f"+{-pct:.1f}% (≈ 0)"
    ax.text(cx, max(real, null) + 0.055, label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=INK)

ax.set_xticks(centers)
ax.set_xticklabels([d[1] for d in DATA], fontsize=9, color=INK)
for x, lbl in [(0.5, "trained Gemma-2-2b"), (3.0, "re-randomized (incl. embeddings, seed 0)")]:
    ax.text(x, -0.135, lbl, transform=ax.get_xaxis_transform(),
            ha="center", fontsize=10, fontweight="bold", color=INK)

ax.set_ylabel("FVU on h (centred)  —  lower = more structure found", fontsize=9, color=INK)
ax.set_ylim(0, 0.82)
ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(colors=MUTED, labelsize=8)

ax.set_title("Same question, one shared target: reconstructing h",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=28)
ax.text(0, 1.06, "skip-embed = composite (map + resid SAE) on h; its floor is map + covariance-only SAE, "
                 "via the exact identity FVU_h = FVU_r · Var(r)/Var(h)",
        transform=ax.transAxes, fontsize=8, color=MUTED)

handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
           plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY, hatch="///")]
ax.legend(handles, ["real system", "matched-Gaussian null (floor)"],
          loc="upper left", frameon=False, fontsize=8.5)

PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS, exist_ok=True)
out = f"{PLOTS}/trained_vs_random_fvu_h.png"
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
