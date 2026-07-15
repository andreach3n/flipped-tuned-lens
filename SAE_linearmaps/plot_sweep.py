"""Reconstruction-vs-sparsity curve: FVU on h (y) vs measured L0 (x), full vs hybrid.
Reads sweep_results.pt (from eval_sweep.py). CPU / seconds.

The horizontal gap between the curves at a fixed FVU = hybrid's win in "extra active-feature
equivalents." NOTE: hybrid's L0 counts only its SPARSE features -- the dense linear map is
off-axis, so the gap is what the map buys in feature-equivalents, not a free lunch.
"""
import torch as t
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "/workspace/sae_cache_layer13"
results = t.load(f"{CACHE_DIR}/sweep_results.pt")

STYLE = {"full": ("#4553c9", "o"), "hybrid": ("#2c885f", "*")}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for mode, (color, marker) in STYLE.items():
    pts = sorted([r for r in results if r["mode"] == mode], key=lambda r: r["L0"])
    if not pts:
        continue
    xs = [r["L0"] for r in pts]; ys = [r["FVU"] for r in pts]
    ax.plot(xs, ys, "-" + marker, color=color, ms=11, lw=2, label=mode)
    for r in pts:                                              # annotate each point with its k
        ax.annotate(f"k={r['k']}", (r["L0"], r["FVU"]), fontsize=7.5, color=color,
                    textcoords="offset points", xytext=(5, 5))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Sparsity  (L0 = active features per token)")
ax.set_ylabel("FVU on h  (normalized reconstruction error)")
ax.set_title("Reconstruction vs sparsity — full vs hybrid\n"
             "(hybrid's L0 excludes the dense linear map)", fontsize=12)
ax.legend()
ax.annotate("better", xy=(0.08, 0.08), xycoords="axes fraction", color="#888", fontsize=11)
ax.annotate("", xy=(0.03, 0.03), xytext=(0.13, 0.13), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#888"))
fig.tight_layout()
out = f"{CACHE_DIR}/sweep_recon_vs_sparsity.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved -> {out}")
