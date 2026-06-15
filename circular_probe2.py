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
from sklearn.model_selection import train_test_split
import numpy as np


MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1
D_MODEL = 2304

def log(msg):
    print(msg, flush=True)

log("Loading days_token_map...")
days_map = t.load("/workspace/days_token_map.pt", weights_only=False)
log(f"  days_map keys: {list(days_map.keys())}")
day_ids_list = sorted(days_map.keys())

log("Loading OWT dataset...")
day_embds = t.load("/workspace/day_embeddings.pt", weights_only=False)
day_ids = t.load("/workspace/day_token_ids.pt", weights_only=False)
day_h = {}
for l in LAYERS:
    day_h[l] = t.load(f"/workspace/day_layer_{l}.pt", weights_only=False)
    log(f"  after layer {l}: {day_h[l].shape}")

log("Loading linear maps...")
linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))
log(f"Loaded maps for layers: {sorted(linear_map.keys())}")

day_order = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
labels = np.array([(day_order.index(days_map[tid.item()])) for tid in day_ids])
targets = np.array([[np.cos(i * 2 * np.pi / 7), np.sin(i * 2 * np.pi / 7)] for i in labels])

indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(indices, test_size=0.2)

labels_train, labels_test = labels[train_idx], labels[test_idx]
targets_train, targets_test = targets[train_idx], targets[test_idx]

H = {}
H_hat = {}
E = {}
for l in LAYERS:
    H[l] = day_h[l]
    with t.no_grad():
        H_hat[l] = linear_map[l](day_embds.float())
    E[l] = H[l] - H_hat[l]

mse_H = {}
mse_H_hat = {}
mse_E = {}
for l in LAYERS:
    P_H, _, _, _ = np.linalg.lstsq(H[l][train_idx].float().numpy(), targets_train, rcond=None)
    preds_H = H[l][test_idx].float().numpy() @ P_H
    mse_H[l] = np.mean((preds_H-targets_test) ** 2)

    P_H_hat, _, _, _ = np.linalg.lstsq(H_hat[l][train_idx].detach().float().numpy(), targets_train, rcond=None)
    preds_H_hat = H_hat[l][test_idx].detach().float().numpy() @ P_H_hat
    mse_H_hat[l] = np.mean((preds_H_hat - targets_test) ** 2)

    P_E, _, _, _ = np.linalg.lstsq(E[l][train_idx].detach().float().numpy(), targets_train, rcond=None)
    preds_E = E[l][test_idx].detach().float().numpy() @ P_E
    mse_E[l] = np.mean((preds_E - targets_test) ** 2)


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
ax.set_title("Circularity of day-of-week representations (OWT)")
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.savefig("/workspace/circular_probe2_loss.png", dpi=150)
log("Saved circular_probe2_loss.png")
