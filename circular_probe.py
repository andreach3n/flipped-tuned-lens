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

