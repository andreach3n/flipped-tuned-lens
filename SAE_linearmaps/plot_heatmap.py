"""Fast standalone: 2D heatmap of feature frequency vs # distinct words in max-20.
Reads the saved per-feature stats (stats_full.pt / stats_resid.pt) so it does NOT re-run the
4M-token eval -- pure CPU, a few seconds. (The same plot is also emitted by eval_trivial.py on a
full rerun.) Uses the MAX/peak word-count ('nd_peak'), the metric interp dashboards show in practice.
"""
import torch as t
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "/workspace/sae_cache_layer13"
TOPK = 20

full  = t.load(f"{CACHE_DIR}/stats_full.pt")
resid = t.load(f"{CACHE_DIR}/stats_resid.pt")

# per-feature: firing frequency, # distinct words in max-20 (peak metric), alive mask
fqF, ndF, alF = full["freq"].float(),  full["nd_peak"].float(),  full["alive"]
fqR, ndR, alR = resid["freq"].float(), resid["nd_peak"].float(), resid["alive"]

fmax = float(max(fqF[alF].max(), fqR[alR].max()))
xedges = 10 ** np.arange(np.log10(TOPK), np.log10(fmax) + 0.5, 0.5)   # half-order-of-magnitude bins
yedges = np.arange(0.5, TOPK + 1.5, 1)                               # integer word-count bins 1..K

def col_norm_hist(nd, fq, alive):
    """Column-normalized 2D histogram: each frequency slice sums to 1 -> P(#words | freq)."""
    H, _, _ = np.histogram2d(fq[alive].numpy(), nd[alive].numpy(), bins=[xedges, yedges])
    col = H.sum(axis=1, keepdims=True)                              # features per frequency column
    return np.divide(H, col, out=np.zeros_like(H), where=col > 0)

Hf = col_norm_hist(ndF, fqF, alF)
Hr = col_norm_hist(ndR, fqR, alR)
X, Y = np.meshgrid(xedges, yedges)
vtop = max(Hf.max(), Hr.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, H, ttl in [(axes[0], Hf, "FULL"), (axes[1], Hr, "RESID")]:
    pc = ax.pcolormesh(X, Y, H.T, cmap="magma", vmin=0, vmax=vtop)
    ax.set_xscale("log"); ax.set_title(f"{ttl}   ·   P(#words | freq)")
    ax.set_xlabel("feature firing frequency (log)"); ax.set_ylabel("# distinct words in max-20")
    fig.colorbar(pc, ax=ax, fraction=.046)

D = (Hr - Hf).T                                                     # resid - full
vmax = float(np.abs(D).max()) or 1e-9
pc = axes[2].pcolormesh(X, Y, D, cmap="RdBu", vmin=-vmax, vmax=vmax)  # blue=resid more, red=full more
axes[2].set_xscale("log"); axes[2].set_title("RESID − FULL   (blue = resid more, red = full more)")
axes[2].set_xlabel("feature firing frequency (log)"); axes[2].set_ylabel("# distinct words in max-20")
fig.colorbar(pc, ax=axes[2], fraction=.046)

fig.suptitle("Feature frequency vs max-20 word count  (column-normalized: P(#words | freq))", fontsize=14)
fig.tight_layout()
out = f"{CACHE_DIR}/heatmap_freq_vs_words.png"
fig.savefig(out, dpi=130, bbox_inches="tight")

# also report the bucket sizes the mentor asked about, per half-OOM frequency bin
counts_f, _ = np.histogram(fqF[alF].numpy(), bins=xedges)
counts_r, _ = np.histogram(fqR[alR].numpy(), bins=xedges)
print("freq bin (half-OOM)      full   resid")
for i in range(len(xedges) - 1):
    print(f"  {xedges[i]:8.0f}-{xedges[i+1]:<8.0f} {int(counts_f[i]):6d} {int(counts_r[i]):6d}")
print(f"\nsaved plot -> {out}")
