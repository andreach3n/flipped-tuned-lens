import torch as t
import torch.nn as nn
from transformer_lens import (
    HookedTransformer,
)
import glob
from transformer_lens.hook_points import HookPoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1
D_MODEL = 2304

# df = pd.read_csv("/Users/andrea/Documents/flipped tuned lens/truth_datasets/cities.csv")
df = pd.read_csv("/workspace/cities.csv")
statements = df["statement"].tolist()
labels = np.array(df["label"].tolist())

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))

embeddings = []
layer_activations = {l: [] for l in LAYERS}
names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in layer_activations]

for data in statements:
    tokens = model.to_tokens(data).to(device)
    tokens = tokens[:, :512]

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    embeddings.append(cache["hook_embed"].squeeze(0).mean(dim=0).cpu())
    for l in layer_activations:
        layer_activations[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).mean(dim=0).cpu()) #last token position [-1], now mean across activations
    del cache, logits

embeddings = t.stack(embeddings)
np_layer_activations = {l: None for l in LAYERS}
for l in LAYERS:
    np_layer_activations[l] = t.stack(layer_activations[l]).float().numpy()

accuracy_full = {}
accuracy_emb = {}
accuracy_res = {}

all_emb_contrib = {}
all_res_contrib = {}
all_test_labels = {}


for l in LAYERS:
    with t.no_grad():
        linear_map[l].eval()
        h_emb = linear_map[l](embeddings.float()).numpy()
    h_res = np_layer_activations[l] - h_emb

    idx = np.arange(len(np_layer_activations[l]))
    train_idx, test_idx = train_test_split(idx, test_size=0.2)

    train_activations, test_activations = np_layer_activations[l][train_idx], np_layer_activations[l][test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    train_h_emb, test_h_emb = h_emb[train_idx], h_emb[test_idx]
    train_h_res, test_h_res = h_res[train_idx], h_res[test_idx]

    # probe for full
    scaler_activations = StandardScaler()
    train_activations = scaler_activations.fit_transform(train_activations)
    test_activations = scaler_activations.transform(test_activations)

    probe_full = LogisticRegression(max_iter=1000)
    probe_full.fit(train_activations, train_labels)

    pred_activations = probe_full.predict(test_activations)
    accuracy_full[l] = accuracy_score(test_labels, pred_activations)

    # probe for embeddings part
    scaler_emb = StandardScaler()
    train_h_emb_scaled, test_h_emb_scaled = scaler_emb.fit_transform(train_h_emb), scaler_emb.transform(test_h_emb)
    probe_emb = LogisticRegression(max_iter=1000)
    probe_emb.fit(train_h_emb_scaled, train_labels)

    pred_emb = probe_emb.predict(test_h_emb_scaled)
    accuracy_emb[l] = accuracy_score(test_labels, pred_emb)

    # probe for residuals
    scaler_res = StandardScaler()
    train_h_res_scaled, test_h_res_scaled = scaler_res.fit_transform(train_h_res), scaler_res.transform(test_h_res)
    probe_res = LogisticRegression(max_iter=1000)
    probe_res.fit(train_h_res_scaled, train_labels)

    pred_res = probe_res.predict(test_h_res_scaled)
    accuracy_res[l] = accuracy_score(test_labels, pred_res)
    print(f"Layer {l} — full: {accuracy_full[l]:.4f}, emb: {accuracy_emb[l]:.4f}, res: {accuracy_res[l]:.4f}")

    test_h_emb_scaled_full = scaler_activations.transform(test_h_emb)
    test_h_res_scaled_full = scaler_activations.transform(test_h_res)

    all_emb_contrib[l] = (probe_full.coef_ @ test_h_emb_scaled_full.T).flatten()
    all_res_contrib[l] = (probe_full.coef_ @ test_h_res_scaled_full.T).flatten()
    all_test_labels[l] = test_labels

# accuracy plot
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(LAYERS, [accuracy_full[l] for l in LAYERS], marker='o', label='full')
ax1.plot(LAYERS, [accuracy_emb[l] for l in LAYERS], marker='o', label='embedding')
ax1.plot(LAYERS, [accuracy_res[l] for l in LAYERS], marker='o', label='residual')
ax1.set_xlabel("Layer")
ax1.set_ylabel("Accuracy")
ax1.set_title("Probe Accuracy by Component")
ax1.legend()
ax1.set_xticks(LAYERS)
ax1.set_ylim(0, 1)
fig1.tight_layout()
fig1.savefig("/workspace/truth_probe_accuracy_by_layer_cities_mean.png", dpi=150)

# contribution scatter plot
fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, l in zip(axes.flatten(), LAYERS):
    colors = ['red' if label == 0 else 'blue' for label in all_test_labels[l]]
    ax.scatter(all_emb_contrib[l], all_res_contrib[l], c=colors, alpha=0.5, s=10)
    ax.set_xlabel("emb contrib")
    ax.set_ylabel("res contrib")
    ax.set_title(f"Layer {l}")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
axes.flatten()[-1].set_visible(False)
fig2.suptitle("Embedding vs Residual Contribution to Probe Score")
fig2.tight_layout()
fig2.savefig("/workspace/truth_probe_contributions_cities_mean.png", dpi=150)
