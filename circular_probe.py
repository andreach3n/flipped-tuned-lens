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

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1
D_MODEL = 2304

def log(msg):
    print(msg, flush=True)

log("Loading days_token_map...")
days_map = t.load("/workspace/days_token_map_simple.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")
day_ids_list = sorted(days_map.keys())

log("Loading 'after' dataset...")
day_embds_after = t.load("/workspace/day_embeddings_simple.pt", weights_only=False)
day_ids_after = t.load("/workspace/day_token_ids_simple.pt", weights_only=False)
day_h_after = {}
for l in LAYERS:
    day_h_after[l] = t.load(f"/workspace/day_layer_{l}_simple.pt", weights_only=False)
    log(f"  after layer {l}: {day_h_after[l].shape}")

log("Loading linear maps...")
linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))
log(f"Loaded maps for layers: {sorted(linear_map.keys())}")

pca_H = {}
pca_H_hat = {}
pca_E = {}

Z_H = {}
Z_H_hat = {}
Z_E = {}
for l in LAYERS:
    pca_H[l] = PCA(n_components=5).fit(day_h_after[l].float().numpy())
    Z_H[l] = pca_H[l].transform(day_h_after[l].float().numpy())

    with t.no_grad():
        H_hat = linear_map[l](day_embds_after.float())
        pca_H_hat[l] = PCA(n_components=5).fit(H_hat.float().numpy())
        Z_H_hat[l] = pca_H_hat[l].transform(H_hat.float().numpy())

    E = day_h_after[l] - H_hat
    pca_E[l] = PCA(n_components=5).fit(E.float().numpy())
    Z_E[l] = pca_E[l].transform(E.float().numpy())

# label each sample 0-6
day_order = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
labels = [day_order.index(days_map[tid.item()]) for tid in day_ids_after]

pca_embds = PCA(n_components=5).fit(day_embds_after.float().numpy())
Z_embds = pca_embds.transform(day_embds_after.float().numpy())

targets = np.array([[np.cos(i * 2 * np.pi / 7), np.sin(i * 2 * np.pi / 7)] for i in labels])

residuals_H = {}
residuals_H_hat = {}
residuals_E = {}
for l in LAYERS:
    _, residuals_H[l], _, _ = np.linalg.lstsq(Z_H[l], targets, rcond=None)
    _, residuals_H_hat[l], _, _ = np.linalg.lstsq(Z_H_hat[l], targets, rcond=None)
    _, residuals_E[l], _, _ = np.linalg.lstsq(Z_E[l], targets, rcond=None)

mse_H = {l: residuals_H[l].sum() / len(labels) for l in LAYERS}
mse_H_hat = {l: residuals_H_hat[l].sum() / len(labels) for l in LAYERS}
mse_E = {l: residuals_E[l].sum() / len(labels) for l in LAYERS}

# plot
x = LAYERS
y_H     = [mse_H[l]     for l in LAYERS]
y_H_hat = [mse_H_hat[l] for l in LAYERS]
y_E     = [mse_E[l]     for l in LAYERS]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y_H,     marker="o", label=r"$H$ (full)")
ax.plot(x, y_H_hat, marker="o", label=r"$\hat{H}$ (predicted)")
ax.plot(x, y_E,     marker="o", label=r"$E$ (residual)")
ax.set_xlabel("Layer")
ax.set_ylabel("Circular probe MSE")
ax.set_title("Circularity of day-of-week representations across layers")
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.savefig("/workspace/circular_probe_loss_simple.png", dpi=150)
log("Saved circular_probe_loss.png")
