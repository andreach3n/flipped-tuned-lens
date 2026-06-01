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

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"Layers: {model.cfg.n_layers}")

# loading openwebtext
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

count = 0
embeddings = []
middle_layer = []

for data in ds:
    tokens = model.to_tokens(data["text"]).to(device)
    tokens = tokens[:, :model.cfg.n_ctx] # crop to context window

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens)

    embeddings.append(cache["hook_embed"].squeeze(0).cpu())
    middle_layer.append(cache["blocks.13.hook_resid_post"].squeeze(0).cpu())

    count+=tokens.shape[1]
    if count >= 100000:
        break

t.save(embeddings, "./embeddings.pt")
t.save(middle_layer, "./middle_layer.pt")
