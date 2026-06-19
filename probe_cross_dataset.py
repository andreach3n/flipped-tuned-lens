import torch as t
import torch.nn as nn
from transformer_lens import (
    HookedTransformer,
)
import glob
from transformer_lens.hook_points import HookPoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1
D_MODEL = 2304

df_train = pd.read_csv("/workspace/cities.csv")
statements_train = df_train["statement"].tolist()
labels_train = np.array(df_train["label"].tolist())

df_test = pd.read_csv("/workspace/neg_cities.csv")
statements_test = df_test["statement"].tolist()
labels_test = np.array(df_test["label"].tolist())

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))

train_embeddings, test_embeddings = [], []
train_activations_by_layer, test_activations_by_layer= {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in LAYERS]

for data in statements_train:
    tokens = model.to_tokens(data).to(device)
    tokens = tokens[:, :512]

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    train_embeddings.append(cache["hook_embed"].squeeze(0).mean(dim=0).cpu())
    for l in train_activations_by_layer:
        train_activations_by_layer[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).mean(dim=0).cpu()) #last token position [-1], now mean across activations mean(dim=0)
    del cache, logits

for data in statements_test:
    tokens = model.to_tokens(data).to(device)
    tokens = tokens[:, :512]

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    test_embeddings.append(cache["hook_embed"].squeeze(0).mean(dim=0).cpu())
    for l in test_activations_by_layer:
        test_activations_by_layer[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).mean(dim=0).cpu()) #last token position [-1], now mean across activations mean(dim=0)
    del cache, logits

train_embeddings, test_embeddings = t.stack(train_embeddings), t.stack(test_embeddings)
train_np_activations_by_layer, test_np_activations_by_layer = {l: None for l in LAYERS}, {l: None for l in LAYERS}
for l in LAYERS:
    train_np_activations_by_layer[l] = t.stack(train_activations_by_layer[l]).float().numpy()
    test_np_activations_by_layer[l] = t.stack(test_activations_by_layer[l]).float().numpy()

accuracy_full = {}
accuracy_emb = {}
accuracy_res = {}

all_emb_contrib = {}
all_res_contrib = {}

for l in LAYERS:
    with t.no_grad():
        linear_map[l].eval()
        train_h_emb = linear_map[l](train_embeddings.float()).numpy()
        test_h_emb = linear_map[l](test_embeddings.float()).numpy()
    train_h_res = train_np_activations_by_layer[l] - train_h_emb
    test_h_res = test_np_activations_by_layer[l] - test_h_emb

    # probe for full
    scaler_activations = StandardScaler()
    train_activations = scaler_activations.fit_transform(train_np_activations_by_layer[l])
    test_activations = scaler_activations.transform(test_np_activations_by_layer[l])

    probe_full = LogisticRegression(max_iter=1000, C=0.1)
    probe_full.fit(train_activations, labels_train)

    pred_activations = probe_full.predict(test_activations)
    accuracy_full[l] = accuracy_score(labels_test, pred_activations)
    train_acc_full = probe_full.score(train_activations, labels_train)

    # probe for embeddings part
    scaler_emb = StandardScaler()
    train_h_emb_scaled, test_h_emb_scaled = scaler_emb.fit_transform(train_h_emb), scaler_emb.transform(test_h_emb)
    probe_emb = LogisticRegression(max_iter=1000, C=0.1)
    probe_emb.fit(train_h_emb_scaled, labels_train)

    pred_emb = probe_emb.predict(test_h_emb_scaled)
    accuracy_emb[l] = accuracy_score(labels_test, pred_emb)
    train_acc_emb = probe_emb.score(train_h_emb_scaled, labels_train)

    # probe for residuals
    scaler_res = StandardScaler()
    train_h_res_scaled, test_h_res_scaled = scaler_res.fit_transform(train_h_res), scaler_res.transform(test_h_res)
    probe_res = LogisticRegression(max_iter=1000, C=0.1)
    probe_res.fit(train_h_res_scaled, labels_train)

    pred_res = probe_res.predict(test_h_res_scaled)
    accuracy_res[l] = accuracy_score(labels_test, pred_res)
    train_acc_res = probe_res.score(train_h_res_scaled, labels_train)
    print(f"Layer {l} — full: {accuracy_full[l]:.4f} (train {train_acc_full:.4f}), emb: {accuracy_emb[l]:.4f} (train {train_acc_emb:.4f}), res: {accuracy_res[l]:.4f} (train {train_acc_res:.4f})")

    test_h_emb_scaled_full = scaler_activations.transform(test_h_emb)
    test_h_res_scaled_full = scaler_activations.transform(test_h_res)

    all_emb_contrib[l] = (probe_full.coef_ @ test_h_emb_scaled_full.T).flatten()
    all_res_contrib[l] = (probe_full.coef_ @ test_h_res_scaled_full.T).flatten()

# accuracy plot
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(LAYERS, [accuracy_full[l] for l in LAYERS], marker='o', label='full')
ax1.plot(LAYERS, [accuracy_emb[l] for l in LAYERS], marker='o', label='embedding')
ax1.plot(LAYERS, [accuracy_res[l] for l in LAYERS], marker='o', label='residual')
ax1.set_xlabel("Layer")
ax1.set_ylabel("Accuracy")
ax1.set_title("Probe Accuracy by Component (train: cities, test: neg_cities)")
ax1.legend()
ax1.set_xticks(LAYERS)
ax1.set_ylim(0, 1)
fig1.tight_layout()
fig1.savefig("/workspace/truth_probe_accuracy_cross_dataset_cities.png", dpi=150)

# contribution scatter plot
fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, l in zip(axes.flatten(), LAYERS):
    colors = ['red' if label == 0 else 'blue' for label in labels_test]
    ax.scatter(all_emb_contrib[l], all_res_contrib[l], c=colors, alpha=0.5, s=10)
    ax.set_xlabel("emb contrib")
    ax.set_ylabel("res contrib")
    ax.set_title(f"Layer {l}")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
axes.flatten()[-1].set_visible(False)
fig2.suptitle("Embedding vs Residual Contribution (train: cities, test: neg_cities)")
fig2.tight_layout()
fig2.savefig("/workspace/truth_probe_contributions_cross_dataset_cities.png", dpi=150)
