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
import glob

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
D_MODEL = 2304

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
emb_flat    = t.cat(embeddings, dim=0).float().to(device)   # (N, d_model)  -> upcast from bf16
act_flat    = t.cat(activations, dim=0).float().to(device)  # (N, d_act)

linear_map = nn.Linear(D_MODEL, D_MODEL).to(device)
linear_map.load_state_dict(t.load("/workspace/linear_map_layer_13.pt", weights_only=False))
linear_map.eval()

with t.no_grad():
    p = linear_map(emb_flat)
    r = act_flat - p

    a = sae.encode(act_flat)
    # compute triviality metric
    vals, idx = a.topk(100, dim=0)
    top_tokens = tokens_flat.to(idx.device)[idx]
    modal_tok, _ = t.mode(top_tokens, dim=0)
    modal_frac = (top_tokens == modal_tok).float().mean(dim=0)

    W_enc = sae.W_enc.float()
    b_dec = sae.b_dec.float()

    tok_drive = (p - b_dec) @ W_enc
    res_drive = r @ W_enc

    var_tok = tok_drive.var(dim=0)   # (d_sae,)
    var_res = res_drive.var(dim=0)   # (d_sae,)
    token_frac = var_tok / (var_tok + var_res)   # (d_sae,) in [0,1]

    alive = (var_tok + var_res) > 1e-8          # avoid 0/0 -> NaN for silent features
    fires = (a > 0).sum(dim=0) >= 100           # enough activations for modal_frac to mean anything
    mask  = alive & fires

    tf = token_frac[mask].cpu()
    mf = modal_frac[mask].cpu()

import matplotlib.pyplot as plt
plt.scatter(tf, mf, s=4, alpha=0.3)
plt.xlabel("token_frac  (drive from embedding-predictable part)")
plt.ylabel("modal-token fraction  (triviality)")
plt.title("Are trivial features the linearly-predictable ones?")
plt.savefig("/workspace/first_check.png", dpi=150)

# the actual numbers that decide it
trivial     = mask & (modal_frac > 0.8)
nontrivial  = mask & (modal_frac < 0.3)
print("median token_frac — trivial features:   ", token_frac[trivial].median().item())
print("median token_frac — non-trivial features:", token_frac[nontrivial].median().item())
