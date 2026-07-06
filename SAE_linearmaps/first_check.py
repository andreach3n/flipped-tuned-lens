from sae_lens import SAE
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
LAYERS = [1, 5, 9, 13, 17, 21, 25]

device = t.device("cuda" if t.cuda.is_available() else "cpu")
sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",   # residual-stream SAEs
    sae_id="layer_13/width_16k/canonical",
    device="cuda",
)
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

embeddings = []
tokens = []
activations = []
names_filter = ["hook_embed"] + [f"blocks.{13}.hook_resid_post"]
count = 0

for data in ds:
    curr_tokens = model.to_tokens(data["text"]).to(device)
    curr_tokens = curr_tokens[:, :512]

    if curr_tokens.shape[1] < 10:
        continue

    with t.no_grad():
        logits, cache = model.run_with_cache(curr_tokens, names_filter=names_filter, stop_at_layer=14)
    tokens.append(curr_tokens.squeeze(0)[1:].cpu()) # drop the BOS token
    embeddings.append(cache["hook_embed"].squeeze(0)[1:].cpu())
    activations.append(cache["blocks.13.hook_resid_post"].squeeze(0)[1:].cpu())

    count+=curr_tokens.shape[1]
    if count >= 100000:
        break

tokens_flat = t.cat(tokens, dim=0)          # (N,)
emb_flat    = t.cat(embeddings, dim=0).float()   # (N, d_model)  -> upcast from bf16
act_flat    = t.cat(activations, dim=0).float()  # (N, d_act)

