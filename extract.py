import os
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
# MAX_TOKENS = 1280000
MAX_TOKENS = 50_000_000          # SAE training budget
# LAYERS = [1, 5, 9, 17, 21, 25]
LAYERS = [13]                    # only the layer we're training the SAE on
STOP_LAYER = max(LAYERS) + 1
# [1, 5, 9, 13, 17, 21, 25]
CHUNK_TOKENS = 1_000_000         # save a chunk every ~1M tokens (~50 files total)
OUT_DIR = "/workspace/sae_cache_layer13"   # namespaced dir -> does NOT touch the old 1.28M cache
os.makedirs(OUT_DIR, exist_ok=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
print(f"Layers: {model.cfg.n_layers}")

# loading openwebtext
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

count = 0
# embeddings = []
token_ids = []                                   # keep token ids -> lets us residualize (r = h - P[tok]) later
layer_activations = {l: [] for l in LAYERS}
# names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in layer_activations]
names_filter = [f"blocks.{l}.hook_resid_post" for l in layer_activations]
chunk = 0

print("Starting data loop...")
for data in ds:
    tokens = model.to_tokens(data["text"]).to(device)
    # tokens = tokens[:, :model.cfg.n_ctx] # crop to context window
    tokens = tokens[:, :512]

    if tokens.shape[1] < 10:
        continue

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    # embeddings.append(cache["hook_embed"].squeeze(0).cpu())
    # drop position 0 (BOS) everywhere -> its massive activation would dominate the SAE
    token_ids.append(tokens.squeeze(0)[1:].cpu())
    for l in layer_activations:
        # layer_activations[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0).cpu())
        layer_activations[l].append(cache[f"blocks.{l}.hook_resid_post"].squeeze(0)[1:].cpu())

    del cache, logits
    t.cuda.empty_cache()

    # count+=tokens.shape[1]
    count += tokens.shape[1] - 1                  # count the tokens we actually kept (BOS dropped)

    # if count // 100000 > chunk:
    #     chunk = count // 100000
    if count // CHUNK_TOKENS > chunk:
        chunk = count // CHUNK_TOKENS
        print(f"Saving checkpoint at {count} tokens...")
        # t.save(embeddings, f"/workspace/embeddings_chunk_{chunk}.pt")
        # t.save(token_ids, f"/workspace/tokens_chunk_{chunk}.pt")
        t.save(token_ids, f"{OUT_DIR}/tokens_chunk_{chunk}.pt")
        for l in LAYERS:
            # t.save(layer_activations[l], f"/workspace/layer_{l}_chunk_{chunk}.pt")
            t.save(layer_activations[l], f"{OUT_DIR}/layer_{l}_chunk_{chunk}.pt")
        # embeddings = []
        token_ids = []
        layer_activations = {l: [] for l in LAYERS}

    if count >= MAX_TOKENS:
        break

# this was needed when i didnt chunk save tokens
# t.save(embeddings, "/workspace/embeddings.pt")
# for l in LAYERS:
#     t.save(layer_activations[l], f"/workspace/layer_{l}.pt")
