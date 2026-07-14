"""Single-bucket, EQUAL-WEIGHT distribution of the max-20 unique-word count.

Every alive feature counts ONCE -- a "per-dictionary-row" view (each row of the dictionary gets
equal say), with NO frequency stratification and NO frequency weighting. This is the deliberately
naive view: it does not control for firing frequency, so it will re-surface the aggregate
"resid looks more trivial" effect that the stratified / heatmap plots correct for (resid holds
more rare features, and rare features are trivially single-token).

Reads stats_*.pt (nd_peak = # distinct words in the MAX-20 examples), so it's CPU / seconds.
"""
import torch as t
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "/workspace/sae_cache_layer13"
TOPK = 20

# (label, color, stats file) -- hybrid is skipped automatically if eval_trivial hasn't produced it
SERIES = [("full",   "#4553c9", "stats_full.pt"),
          ("resid",  "#b5762e", "stats_resid.pt"),
          ("hybrid", "#2c885f", "stats_hybrid.pt")]

fig, ax = plt.subplots(figsize=(9, 5.5))
ubins = np.arange(0.5, TOPK + 1.5, 1)              # integer-centered bins 1..20
for label, color, fname in SERIES:
    path = f"{CACHE_DIR}/{fname}"
    if not os.path.exists(path):
        continue
    s = t.load(path)
    nd = s["nd_peak"].float()[s["alive"]].numpy()  # # distinct words in max-20, alive features, ONE vote each
    ax.hist(nd, bins=ubins, density=True, alpha=.5, color=color,
            label=f"{label}  (n={len(nd)}, mean {nd.mean():.1f}, single-word {(nd == 1).mean()*100:.1f}%)")

ax.set_xlabel(f"# distinct words in the max-{TOPK} examples")
ax.set_ylabel("fraction of dictionary features")
ax.set_title("Dictionary complexity — one bucket, every feature weighted equally\n"
             "(each dictionary row counts once; NOT frequency-controlled)")
ax.legend()
fig.tight_layout()
out = f"{CACHE_DIR}/dict_equal_weight.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved plot -> {out}")
