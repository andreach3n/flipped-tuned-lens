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

log("Loading 'after' dataset...")
day_embds_after = t.load("/workspace/day_embeddings_after.pt", weights_only=False)
day_ids_after = t.load("/workspace/day_token_ids_after.pt", weights_only=False)
day_h_after = {}
for l in LAYERS:
    day_h_after[l] = t.load(f"/workspace/day_layer_{l}_after.pt", weights_only=False)
    log(f"  after layer {l}: {day_h_after[l].shape}")

log("Loading 'simple' dataset...")
day_embds_simple = t.load("/workspace/day_embeddings_simple.pt", weights_only=False)
day_ids_simple = t.load("/workspace/day_token_ids_simple.pt", weights_only=False)
day_h_simple = {}
for l in LAYERS:
    day_h_simple[l] = t.load(f"/workspace/day_layer_{l}_simple.pt", weights_only=False)
    log(f"  simple layer {l}: {day_h_simple[l].shape}")

log("Loading days_token_map...")
days_map = t.load("/workspace/days_token_map_after.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")
day_ids_list = sorted(days_map.keys())

def get_matrices(l):
    H_after = day_h_after[l].float()
    H_simple = day_h_simple[l].float()
    with t.no_grad():
        H_hat_after = linear_map[l](day_embds_after.float())
        H_hat_simple = linear_map[l](day_embds_simple.float())
    E_after = H_after - H_hat_after
    E_simple = H_simple - H_hat_simple
    return {
        "H":     (H_after.detach().cpu().numpy(),     H_simple.detach().cpu().numpy()),
        "H_hat": (H_hat_after.detach().cpu().numpy(), H_hat_simple.detach().cpu().numpy()),
        "E":     (E_after.detach().cpu().numpy(),     E_simple.detach().cpu().numpy()),
    }

log("Running PCA...")
results = {}
for l in LAYERS:
    log(f"  Layer {l}...")
    mats = get_matrices(l)
    results[l] = {}
    for piece, (mat_after, mat_simple) in mats.items():
        combined = np.vstack([mat_after, mat_simple])
        pca = PCA(n_components=3).fit(combined)

        means_after  = np.zeros((len(day_ids_list), mat_after.shape[1]))
        means_simple = np.zeros((len(day_ids_list), mat_simple.shape[1]))
        for i, tid in enumerate(day_ids_list):
            means_after[i]  = mat_after [(day_ids_after  == tid).numpy()].mean(axis=0)
            means_simple[i] = mat_simple[(day_ids_simple == tid).numpy()].mean(axis=0)

        results[l][piece] = {
            "means_after_proj":  pca.transform(means_after),
            "means_simple_proj": pca.transform(means_simple),
            "var": pca.explained_variance_ratio_,
        }
log("PCA done.")

cmap = plt.get_cmap("tab10")
PIECES = ["H", "H_hat", "E"]
PIECE_LABELS = {"H": r"Full $h^l$", "H_hat": r"Predicted $\hat{h}^l$", "E": r"Residual $e^l$"}

def make_plot(pc_x, pc_y, filename):
    log(f"Plotting PC{pc_x+1}/PC{pc_y+1}...")
    n_rows, n_cols = len(PIECES), len(LAYERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for col, l in enumerate(LAYERS):
        for row, piece in enumerate(PIECES):
            ax = axes[row, col]
            var          = results[l][piece]["var"]
            mp_after     = results[l][piece]["means_after_proj"]
            mp_simple    = results[l][piece]["means_simple_proj"]
            mean_colors  = [cmap(i) for i in range(len(day_ids_list))]

            # Simple templates as circles (anchors)
            ax.scatter(mp_simple[:, pc_x], mp_simple[:, pc_y], c=mean_colors,
                       s=120, edgecolors="black", linewidths=1.0,
                       marker="o", zorder=9, alpha=0.85)
            # Day-after templates as stars
            ax.scatter(mp_after[:, pc_x], mp_after[:, pc_y], c=mean_colors,
                       s=200, edgecolors="black", linewidths=1.5,
                       marker="*", zorder=10)
            # Arrows from simple (circle) to day-after (star)
            for i in range(len(day_ids_list)):
                ax.annotate("",
                    xy=(mp_after[i, pc_x], mp_after[i, pc_y]),
                    xytext=(mp_simple[i, pc_x], mp_simple[i, pc_y]),
                    arrowprops=dict(arrowstyle="->", color=mean_colors[i],
                                   lw=1.5, mutation_scale=10),
                    zorder=8)

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

    day_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=8,
               label=days_map[tid].strip())
        for i, tid in enumerate(day_ids_list)
    ]
    type_handles = [
        Line2D([0], [0], marker="o", color="gray", markersize=8,
               linestyle="None", label="Simple (anchor)"),
        Line2D([0], [0], marker="*", color="gray", markersize=10,
               linestyle="None", label="Day-after"),
    ]
    fig.legend(
        handles=day_handles + type_handles, loc="lower center", ncol=9,
        fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("PCA of day-of-week representations: simple (○) vs day-after (★)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved to {filename}")


make_plot(0, 1, "/workspace/pca_plots/pca_days_pc12_combined.png")
make_plot(1, 2, "/workspace/pca_plots/pca_days_pc23_combined.png")
