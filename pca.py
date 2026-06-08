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

day_embds = t.load("/workspace/day_embeddings.pt", weights_only=False)
day_ids = t.load("/workspace/day_token_ids.pt", weights_only=False)
days_map = t.load("/workspace/days_token_map.pt", weights_only=False)

day_h = {}
for l in LAYERS:
    day_h[l] = t.load(f"/workspace/day_layer_{l}.pt", weights_only=False)

def returnMatrices(l):
    H = day_h[l].float()
    with t.no_grad():
        H_hat = linear_map[l](day_embds.float())
    residual = H - H_hat
    return H, H_hat, residual

pca = PCA(n_components=2)

for l in LAYERS:
    H, H_hat, residual = returnMatrices(l)
    H_2d = pca.fit_transform(H)
    H_hat_2d = pca.fit_transform(H_hat)
    residual_2d = pca.fit_transform(residual)
