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
LAYERS = [5, 13, 21]
STOP_LAYER = max(LAYERS) + 1
D_MODEL = 2304

df_cities_neg = pd.concat([
    pd.read_csv("/workspace/cities.csv"),
    pd.read_csv("/workspace/neg_cities.csv")
], ignore_index=True)

df_confounded = pd.concat([
    pd.read_csv("/workspace/cities.csv").query("label == 1"),
    pd.read_csv("/workspace/neg_cities.csv").query("label == 0")
], ignore_index=True)

df_cities = pd.read_csv("/workspace/cities.csv")
df_neg_cities = pd.read_csv("/workspace/neg_cities.csv")

df_cities_true = df_cities[df_cities["label"] == 1]
df_cities_false = df_cities[df_cities["label"] == 0]
df_neg_cities_true = df_neg_cities[df_neg_cities["label"] == 1]
df_neg_cities_false = df_neg_cities[df_neg_cities["label"] == 0]
# DATASETS = {
#     "cities": pd.read_csv("/workspace/cities.csv"),
#     "neg_cities": pd.read_csv("/workspace/neg_cities.csv"),
#     "cities+neg_cities": df_cities_neg,
#     "common_claim": pd.read_csv("/workspace/common_claim_true_false.csv"),
#     "companies": pd.read_csv("/workspace/companies_true_false.csv"),
#     "smaller_than": pd.read_csv("/workspace/smaller_than.csv"),
#     "sp_en_trans": pd.read_csv("/workspace/sp_en_trans.csv"),
# }

TRAIN_DATASETS = {
    "confounded": df_confounded,
    "cities": df_cities,
    "neg_cities": df_neg_cities,
}

TEST_DATASETS = {
    "confounded": df_confounded,
    "cities": df_cities,
    "neg_cities": df_neg_cities,
    "cities_true": df_cities_true,
    "cities_false": df_cities_false,
    "neg_cities_true": df_neg_cities_true,
    "neg_cities_false": df_neg_cities_false,
}

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(D_MODEL, D_MODEL)
    linear_map[l].load_state_dict(t.load(path, weights_only=False))

names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in LAYERS]

dataset_data = {}
for dataset in TEST_DATASETS:
    statements = TEST_DATASETS[dataset]["statement"].tolist()
    labels = np.array(TEST_DATASETS[dataset]["label"].tolist())

    embeddings = []
    activations_by_layer = {l: [] for l in LAYERS}
    for statement in statements:
        tokens = model.to_tokens(statement).to(device)
        tokens = tokens[:, :512]

        with t.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

        embeddings.append(cache["hook_embed"].squeeze(0).mean(dim=0).cpu())
        for l in activations_by_layer:
            activations_by_layer[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).mean(dim=0).cpu())
        del cache, logits

    embeddings_stacked = t.stack(embeddings)
    np_activations = {l: t.stack(activations_by_layer[l]).float().numpy() for l in LAYERS}
    dataset_data[dataset] = (embeddings_stacked, np_activations, labels)

train_accuracy_full = {}
train_accuracy_emb = {}
train_accuracy_res = {}

test_accuracy_full = {}
test_accuracy_emb = {}
test_accuracy_res = {}

for train_dataset in TRAIN_DATASETS:
    for test_dataset in TEST_DATASETS:
        for l in LAYERS:
            with t.no_grad():
                linear_map[l].eval()
                train_h_emb = linear_map[l](dataset_data[train_dataset][0].float()).numpy()
                test_h_emb = linear_map[l](dataset_data[test_dataset][0].float()).numpy()
            train_h_res = dataset_data[train_dataset][1][l] - train_h_emb
            test_h_res = dataset_data[test_dataset][1][l] - test_h_emb

            # full probe
            scaler_activations = StandardScaler()
            train_activations = scaler_activations.fit_transform(dataset_data[train_dataset][1][l])
            test_activations = scaler_activations.transform(dataset_data[test_dataset][1][l])

            probe_full = LogisticRegression(max_iter=1000, C=0.1)
            probe_full.fit(train_activations, dataset_data[train_dataset][2])

            pred_activations = probe_full.predict(test_activations)
            test_accuracy_full[(train_dataset, test_dataset, l)] = accuracy_score(dataset_data[test_dataset][2], pred_activations)
            train_accuracy_full[(train_dataset, test_dataset, l)] = probe_full.score(train_activations, dataset_data[train_dataset][2])

            # probe for embeddings part
            scaler_emb = StandardScaler()
            train_h_emb_scaled, test_h_emb_scaled = scaler_emb.fit_transform(train_h_emb), scaler_emb.transform(test_h_emb)
            probe_emb = LogisticRegression(max_iter=1000, C=0.1)
            probe_emb.fit(train_h_emb_scaled, dataset_data[train_dataset][2])

            pred_emb = probe_emb.predict(test_h_emb_scaled)
            test_accuracy_emb[(train_dataset, test_dataset, l)] = accuracy_score(dataset_data[test_dataset][2], pred_emb)
            train_accuracy_emb[(train_dataset, test_dataset, l)] = probe_emb.score(train_h_emb_scaled, dataset_data[train_dataset][2])

            # probe for residuals
            scaler_res = StandardScaler()
            train_h_res_scaled, test_h_res_scaled = scaler_res.fit_transform(train_h_res), scaler_res.transform(test_h_res)
            probe_res = LogisticRegression(max_iter=1000, C=0.1)
            probe_res.fit(train_h_res_scaled, dataset_data[train_dataset][2])

            pred_res = probe_res.predict(test_h_res_scaled)
            test_accuracy_res[(train_dataset, test_dataset, l)] = accuracy_score(dataset_data[test_dataset][2], pred_res)
            train_accuracy_res[(train_dataset, test_dataset, l)] = probe_res.score(train_h_res_scaled, dataset_data[train_dataset][2])

# dataset_names = list(DATASETS.keys())
train_names = list(TRAIN_DATASETS.keys())
test_names = list(TEST_DATASETS.keys())

for component, acc_dict in [("full", test_accuracy_full), ("embedding", test_accuracy_emb), ("residual", test_accuracy_res)]:
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(6 * len(LAYERS), 5))
    if len(LAYERS) == 1:
        axes = [axes]
    for ax, l in zip(axes, LAYERS):
        grid = np.array([
            [acc_dict[(train, test, l)] for test in test_names]
            for train in train_names
        ])
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(test_names)))
        ax.set_yticks(range(len(train_names)))
        ax.set_xticklabels(test_names, rotation=45, ha="right")
        ax.set_yticklabels(train_names)
        ax.set_xlabel("test dataset")
        ax.set_ylabel("train dataset")
        ax.set_title(f"Layer {l}")
        for i in range(len(train_names)):
            for j in range(len(test_names)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Cross-dataset probe accuracy — {component}")
    fig.tight_layout()
    fig.savefig(f"/workspace/probe_grid_{component}_confounded.png", dpi=150)
