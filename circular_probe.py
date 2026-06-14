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

def log(msg):
    print(msg, flush=True)

log("Loading days_token_map...")
days_map = t.load("/workspace/days_token_map_after.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")
day_ids_list = sorted(days_map.keys())

log("Loading 'after' dataset...")
day_embds_after = t.load("/workspace/day_embeddings_after.pt", weights_only=False)
day_ids_after = t.load("/workspace/day_token_ids_after.pt", weights_only=False)
day_h_after = {}
for l in LAYERS:
    day_h_after[l] = t.load(f"/workspace/day_layer_{l}_after.pt", weights_only=False)
    log(f"  after layer {l}: {day_h_after[l].shape}")

# label each sample 0-6
day_order = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
labels = [day_order.index(days_map[tid.item()]) for tid in day_ids_after]

pca_embds = PCA(n_components=5).fit(day_embds_after.float().numpy())
Z_embds = pca_embds.transform(day_embds_after.float().numpy())

pca = {}
Z = {}
for l in LAYERS:
    pca[l] = PCA(n_components=5).fit(day_h_after[l].float().numpy())
    Z[l] = pca[l].transform(day_h_after[l].float().numpy())

targets = np.array([[np.cos(i * 2 * np.pi / 7), np.sin(i * 2 * np.pi / 7)] for i in labels])


_, residuals_embds, _, _ = np.linalg.lstsq(Z_embds, targets, rcond=None)
residuals = {}
for l in LAYERS:
    _, residuals[l], _, _ = np.linalg.lstsq(Z[l], targets, rcond=None)

mse_embds = residuals_embds.sum()/len(labels)
mse_layers = {l: residuals[l].sum()/len(labels) for l in LAYERS}

# plot 
x = [0] + LAYERS
y = [mse_embds] + [mse_layers[l] for l in LAYERS]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, marker="o")
ax.set_xlabel("Layer (0 = embedding)")
ax.set_ylabel("Circular probe MSE")
ax.set_title("Circularity of day-of-week representations across layers")
ax.set_xticks(x)
plt.tight_layout()
plt.savefig("/workspace/circular_probe_loss.png", dpi=150)
log("Saved circular_probe_loss.png")
