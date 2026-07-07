from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput
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

CACHE_DIR   = "/workspace/sae_cache_layer13"
FULL_PATH   = f"{CACHE_DIR}/sae_full_final.pt"
RESID_PATH  = f"{CACHE_DIR}/sae_resid_final.pt"
P_PATH      = f"{CACHE_DIR}/P.pt"
EVAL_N      = 100_000     # subset of tokens for the eval (memory-bounded)
TOPK        = 100         # top activating examples per feature
TRIVIAL_THRESH = 0.8      # modal-token fraction above this = "trivial"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

h = t.cat(t.load(f"{CACHE_DIR}/layer_13_chunk_1.pt"), dim=0)[:EVAL_N]   # (100k, 2304)
tok = t.cat(t.load(f"{CACHE_DIR}/tokens_chunk_1.pt"),   dim=0)[:EVAL_N]   # (100k,)
P = t.load(P_PATH, map_location=device)   # (V, 2304)

def load_sae(path):
    ckpt = t.load(path)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])   # cfg was saved in the checkpoint
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]                  # <-- you MUST return the scale

sae_full,  scale_full  = load_sae(FULL_PATH)
sae_resid, scale_resid = load_sae(RESID_PATH)

def feature_acts(sae, scale, mode):
    x = (h - P[tok]) if mode == "resid" else h     # same transform as training
    x = x / scale
    a = sae.encode(x)   # (N, 16384) — do it in batches (see memory note)

# item 4: feature activations for each SAE
a_full  = feature_acts(sae_full,  scale_full,  "full")    # (N, 16384)
a_resid = feature_acts(sae_resid, scale_resid, "resid")

def trivial_fraction(a, tok):
    vals, idx = a.topk(TOPK, dim=0)
    top_tokens = tok.to(idx.device)[idx]
    modal_tok, _ = t.mode(top_tokens, dim=0)
    modal_frac = (top_tokens == modal_tok).float().mean(dim=0)
    alive = (a>0).sum(dim=0) >= TOPK
    trivial = alive & (modal_frac > TRIVIAL_THRESH)
    return (trivial.sum()/alive.sum()).item()

# item 5: the headline numbers
print("full  trivial fraction:", trivial_fraction(a_full,  tok))
print("resid trivial fraction:", trivial_fraction(a_resid, tok))
