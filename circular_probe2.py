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
N_EPOCHS = 20000

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
train_val_idx, test_idx = train_test_split(indices, test_size=0.15)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15/0.85)

labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]
targets_train, targets_val, targets_test = targets[train_idx], targets[val_idx], targets[test_idx]

targets_train_t = t.tensor(targets_train, dtype=t.float32)
targets_val_t   = t.tensor(targets_val,   dtype=t.float32)
targets_test_t  = t.tensor(targets_test,  dtype=t.float32)

H = {}
H_hat = {}
E = {}
for l in LAYERS:
    H[l] = day_h[l]
    with t.no_grad():
        H_hat[l] = linear_map[l](day_embds.float())
    E[l] = H[l].float() - H_hat[l]

mse_H = {}
mse_H_hat = {}
mse_E = {}

PATIENCE = 50

def train_probe(acts_train, acts_val, acts_test, label):
    probe = nn.Linear(D_MODEL, 2, bias=False)
    optimizer = t.optim.Adam(probe.parameters())

    mean = acts_train.mean(axis=0)
    std  = acts_train.std(axis=0) + 1e-8
    acts_train = (acts_train - mean) / std
    acts_val   = (acts_val   - mean) / std
    acts_test  = (acts_test  - mean) / std

    acts_train_t = t.tensor(acts_train, dtype=t.float32)
    acts_val_t   = t.tensor(acts_val,   dtype=t.float32)
    acts_test_t  = t.tensor(acts_test,  dtype=t.float32)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        loss = t.nn.functional.mse_loss(probe(acts_train_t), targets_train_t)
        loss.backward()
        optimizer.step()

        with t.no_grad():
            val_loss = t.nn.functional.mse_loss(probe(acts_val_t), targets_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            log(f"  {label} epoch {epoch}: val loss {val_loss:.4f}")

        if epochs_without_improvement >= PATIENCE:
            log(f"  {label} early stopping at epoch {epoch}")
            break

    with t.no_grad():
        test_mse = t.nn.functional.mse_loss(probe(acts_test_t), targets_test_t).item()
    return test_mse

for l in LAYERS:
    log(f"Layer {l}...")
    mse_H[l] = train_probe(H[l][train_idx].float().numpy(), H[l][val_idx].float().numpy(), H[l][test_idx].float().numpy(), f"H layer {l}")
    mse_H_hat[l] = train_probe(H_hat[l][train_idx].detach().float().numpy(), H_hat[l][val_idx].detach().float().numpy(), H_hat[l][test_idx].detach().float().numpy(), f"H_hat layer {l}")
    mse_E[l] = train_probe(E[l][train_idx].detach().float().numpy(), E[l][val_idx].detach().float().numpy(), E[l][test_idx].detach().float().numpy(), f"E layer {l}")

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
plt.savefig("/workspace/circular_probe2_loss_OWT_gd_1.png", dpi=150)
log("Saved circular_probe2_loss.png")
