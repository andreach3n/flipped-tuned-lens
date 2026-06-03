import torch as t
import torch.nn as nn
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset

MODEL_NAME = "google/gemma-2-2b"
MAX_TOKENS = 1280000
LAYERS = [1, 5, 9, 13, 17, 21, 25]

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
print(f"Layers: {model.cfg.n_layers}")

# loading openwebtext
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

count = 0
embeddings = []
layer_activations = {l: [] for l in LAYERS}
names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in layer_activations]

for data in ds:
    tokens = model.to_tokens(data["text"]).to(device)
    tokens = tokens[:, :model.cfg.n_ctx] # crop to context window

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=26)

    embeddings.append(cache["hook_embed"].squeeze(0).cpu())
    for l in layer_activations:
        layer_activations[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).cpu())

    count+=tokens.shape[1]
    if count >= MAX_TOKENS: # currently 1M tokens
        break

t.save(embeddings, "/workspace/embeddings.pt")
for l in LAYERS:
    t.save(layer_activations[l], f"/workspace/layer_{l}.pt")
