import torch as t
import torch.nn as nn
import glob
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D_MODEL = 2304
LAYERS = [1, 5, 9, 13, 17, 21, 25]

def log(msg):
    print(msg, flush=True)

log("Loading linear maps...")
linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))
log(f"Loaded maps for layers: {sorted(linear_map.keys())}")

log("Loading day_embeddings.pt...")
day_embds = t.load("/workspace/day_embeddings.pt", weights_only=False)
log(f"  day_embds shape: {day_embds.shape}")

log("Loading day_token_ids.pt...")
day_ids = t.load("/workspace/day_token_ids.pt", weights_only=False)
log(f"  day_ids shape: {day_ids.shape}")

log("Loading days_token_map.pt...")
days_map = t.load("/workspace/days_token_map.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")

log("Loading day layer activations...")
day_h = {}
for l in LAYERS:
    day_h[l] = t.load(f"/workspace/day_layer_{l}.pt", weights_only=False)
    log(f"  layer {l}: {day_h[l].shape}")

def returnMatrices(l):
    H = day_h[l].float()
    with t.no_grad():
        H_hat = linear_map[l](day_embds.float())
    residual = H - H_hat
    return H, H_hat, residual

log("Running PCA...")
results = {}
for l in LAYERS:
    log(f"  Layer {l}...")
    H, H_hat, residual = returnMatrices(l)
    log(f"    H: {H.shape}, H_hat: {H_hat.shape}, residual: {residual.shape}")
    mats = {
        "H": H.detach().cpu().numpy(),
        "H_hat": H_hat.detach().cpu().numpy(),
        "E": residual.detach().cpu().numpy(),
    }
    results[l] = {}
    for piece, mat in mats.items():
        log(f"    PCA on {piece}...")
        pca = PCA(n_components=2).fit(mat)
        results[l][piece] = {
            "proj": pca.transform(mat),
            "var": pca.explained_variance_ratio_,
        }
log("PCA done.")

# ── Per-point colors (one per day) ────────────────────────────────────────
log("Building plot colors...")
PIECES = ["H", "H_hat", "E"]
PIECE_LABELS = {"H": r"Full $h^l$", "H_hat": r"Predicted $\hat{h}^l$", "E": r"Residual $e^l$"}

day_ids_list = sorted(days_map.keys())                  # 7 token IDs
id_to_idx = {tid: i for i, tid in enumerate(day_ids_list)}
cmap = plt.get_cmap("tab10")
point_colors = np.array([cmap(id_to_idx[tid.item()]) for tid in day_ids])
log(f"  point_colors shape: {point_colors.shape}")

# ── Plot 3 x 7 grid ───────────────────────────────────────────────────────
log("Plotting...")
n_rows, n_cols = len(PIECES), len(LAYERS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

for col, l in enumerate(LAYERS):
    for row, piece in enumerate(PIECES):
        ax = axes[row, col]
        proj = results[l][piece]["proj"]
        var = results[l][piece]["var"]

        ax.scatter(proj[:, 0], proj[:, 1], c=point_colors, s=10, alpha=0.7, edgecolors="none")

        if row == 0:
            ax.set_title(f"Layer {l}", fontsize=11)
        if col == 0:
            ax.set_ylabel(PIECE_LABELS[piece], fontsize=11)

        ax.text(
            0.02, 0.98,
            f"PC1: {var[0]*100:.1f}%\nPC2: {var[1]*100:.1f}%",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

log("Saving figure...")
# ── Shared legend ─────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=cmap(i), markersize=8,
           label=days_map[tid].strip())
    for i, tid in enumerate(day_ids_list)
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=7,
    fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle("PCA of day-of-week token representations across layers", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig("/workspace/pca_days.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved to /workspace/pca_days.png")
