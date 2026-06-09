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
import os
os.makedirs("/workspace/pca_plots", exist_ok=True)

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
day_embds = t.load("/workspace/day_embeddings_1.pt", weights_only=False)
log(f"  day_embds shape: {day_embds.shape}")

log("Loading day_token_ids.pt...")
day_ids = t.load("/workspace/day_token_ids_1.pt", weights_only=False)
log(f"  day_ids shape: {day_ids.shape}")

log("Loading days_token_map.pt...")
days_map = t.load("/workspace/days_token_map_1.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")
day_ids_list = sorted(days_map.keys())

log("Loading day layer activations...")
day_h = {}
for l in LAYERS:
    day_h[l] = t.load(f"/workspace/day_layer_{l}_1.pt", weights_only=False)
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
        pca = PCA(n_components=3).fit(mat)
        means = np.zeros((len(day_ids_list), mat.shape[1]))

        for i, tid in enumerate(day_ids_list):
            mask = (day_ids == tid).numpy()
            means[i] = mat[mask].mean(axis=0)

        results[l][piece] = {
            "proj": pca.transform(mat),
            "means_proj": pca.transform(means),
            "var": pca.explained_variance_ratio_,
        }
log("PCA done.")

# ── Per-point colors (one per day) ────────────────────────────────────────
log("Building plot colors...")
PIECES = ["H", "H_hat", "E"]
PIECE_LABELS = {"H": r"Full $h^l$", "H_hat": r"Predicted $\hat{h}^l$", "E": r"Residual $e^l$"}

# day_ids_list = sorted(days_map.keys())                  # 7 token IDs
id_to_idx = {tid: i for i, tid in enumerate(day_ids_list)}
cmap = plt.get_cmap("tab10")
point_colors = np.array([cmap(id_to_idx[tid.item()]) for tid in day_ids])
log(f"  point_colors shape: {point_colors.shape}")

def make_plot(pc_x, pc_y, filename):
    log(f"Plotting PC{pc_x+1}/PC{pc_y+1}...")
    n_rows, n_cols = len(PIECES), len(LAYERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for col, l in enumerate(LAYERS):
        for row, piece in enumerate(PIECES):
            ax = axes[row, col]
            proj = results[l][piece]["proj"]
            var = results[l][piece]["var"]
            mp = results[l][piece]["means_proj"]

            # Scatter of all points
            # ax.scatter(proj[:, pc_x], proj[:, pc_y], c=point_colors,
            #            s=10, alpha=0.7, edgecolors="none")
            # Overlay of per-day means
            mean_colors = [cmap(i) for i in range(len(day_ids_list))]
            ax.scatter(mp[:, pc_x], mp[:, pc_y], c=mean_colors, s=200,
                       edgecolors="black", linewidths=1.5, marker="*", zorder=10)
            ax.set_aspect("equal", adjustable="datalim")

            if row == 0:
                ax.set_title(f"Layer {l}", fontsize=11)
            if col == 0:
                ax.set_ylabel(PIECE_LABELS[piece], fontsize=11)

            ax.text(
                0.02, 0.98,
                f"PC{pc_x+1}: {var[pc_x]*100:.1f}%\nPC{pc_y+1}: {var[pc_y]*100:.1f}%",
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.7, edgecolor="none"),
            )
            ax.set_xticks([])
            ax.set_yticks([])

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

    fig.suptitle("PCA of day-of-week mean representations across layers",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved to {filename}")


make_plot(0, 1, "/workspace/pca_plots/pca_days_2_pc12_means_only.png")
make_plot(1, 2, "/workspace/pca_plots/pca_days_2_pc23_means_only.png")
