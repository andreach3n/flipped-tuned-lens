"""Standalone bar-chart re-render of the k-sparse probing results (Figure-3 style, arXiv:2511.05541).

Reads ksparse_probe.json (written by probe_sparse.py) and draws GROUPED BARS: one panel per label
type, k on the x-axis, one colored bar per method. Plot-only -- no GPU, no re-run, just matplotlib.

    python plot_probe.py                      # auto-finds ksparse_probe.json in the usual spots
    PROBE_JSON=/path/to/ksparse_probe.json python plot_probe.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# find the JSON: env override, else the usual locations (repo root / SAE dir / pod out dir)
DEFAULT_LOCS = ["ksparse_probe.json", "plots/ksparse_probe.json",
                "SAE_linearmaps/plots/ksparse_probe.json", "/workspace/out/ksparse_probe.json"]
PROBE_JSON = os.environ.get("PROBE_JSON") or next((p for p in DEFAULT_LOCS if os.path.exists(p)), DEFAULT_LOCS[0])
OUT_PNG    = os.environ.get("OUT_PNG") or os.path.join(os.path.dirname(PROBE_JSON) or ".", "ksparse_probe_bars.png")

with open(PROBE_JSON) as f:
    results = json.load(f)
print(f"loaded {PROBE_JSON}")

METHOD_ORDER = ["raw-h", "full", "resid", "hybrid", "outbias"]      # fixed order + colors as elsewhere
COLORS = {"raw-h": "#888888", "full": "#4553c9", "resid": "#b5762e",
          "hybrid": "#2c885f", "outbias": "#a0439c"}
methods = [m for m in METHOD_ORDER if m in results]
panels  = [p for p in ("semantic", "contextual", "syntactic") if p in results[methods[0]]]
ks      = sorted(int(k) for k in results[methods[0]][panels[0]].keys())

fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
x     = np.arange(len(ks))
width = 0.8 / len(methods)
offs  = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width   # center each k-group
for ax, panel in zip(axes[0], panels):
    for i, m in enumerate(methods):
        ys = [results[m][panel][str(k)] for k in ks]
        ax.bar(x + offs[i], ys, width, color=COLORS.get(m, "#333"),
               label=m, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="#ccc", lw=.8, ls=":")                     # chance (balanced accuracy)
    ax.set_title(panel); ax.set_xlabel("k (probe sparsity)"); ax.set_ylabel("balanced accuracy")
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_ylim(0.5, 0.95)                                           # floor at chance to show above-chance signal
axes[0][0].legend(fontsize=7, loc="lower left", ncol=2)
fig.suptitle("k-sparse probing of SAE features  —  MMLU, Gemma-2-2b L13")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"saved -> {OUT_PNG}")

# optionally push to HF (no-op on your Mac / when hf_io or a token isn't available)
try:
    from hf_io import push
    push(OUT_PNG)
except Exception as e:
    print(f"(not pushed to HF: {e})")
