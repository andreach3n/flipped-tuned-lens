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
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])   # cfg was saved in the checkpoint
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]                  # <-- you MUST return the scale

sae_full,  scale_full  = load_sae(FULL_PATH)
sae_resid, scale_resid = load_sae(RESID_PATH)

def feature_acts(sae, scale, mode, bs=8192):
    outs = []
    with t.no_grad():
        for start in range(0, h.shape[0], bs):         # encode in batches — BatchTopK makes
            hh = h[start:start+bs].float().to(device)  # several full-size copies internally
            tt = tok[start:start+bs].to(device)
            x = (hh - P[tt]) if mode == "resid" else hh
            a = sae.encode(x / scale)                  # (bs, 16384)
            outs.append(a.cpu())                       # accumulate on CPU to free GPU
    return t.cat(outs, dim=0)                          # (N, 16384) on CPU

# item 4: feature activations for each SAE
a_full  = feature_acts(sae_full,  scale_full,  "full")    # (N, 16384)
a_resid = feature_acts(sae_resid, scale_resid, "resid")

def triviality(a, tok):
    """Per-feature modal-token fraction + the alive mask (fired >= TOPK times)."""
    vals, idx = a.topk(TOPK, dim=0)
    top_tokens = tok.to(idx.device)[idx]
    modal_tok, _ = t.mode(top_tokens, dim=0)
    modal_frac = (top_tokens == modal_tok).float().mean(dim=0)   # (16384,)
    alive = (a > 0).sum(dim=0) >= TOPK
    return modal_frac, alive

def report(name, a, tok):
    modal_frac, alive = triviality(a, tok)
    mf = modal_frac[alive]                    # distribution over ALIVE features only
    print(f"\n=== {name} ===")
    print(f"alive features: {int(alive.sum())} / {alive.numel()}")
    print(f"modal_frac  mean {mf.mean().item():.4f} | median {mf.median().item():.4f}")
    # threshold sweep: does a gap appear at any cutoff?
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"  frac trivial @ {thr:.1f}: {(mf > thr).float().mean().item():.4f}")
    return modal_frac, alive

mf_full,  al_full  = report("FULL",  a_full,  tok)
mf_resid, al_resid = report("RESID", a_resid, tok)

# side-by-side shift in the distribution (over each SAE's own alive features)
print("\n=== shift (resid - full) ===")
print(f"mean modal_frac:   full {mf_full[al_full].mean().item():.4f}  "
      f"resid {mf_resid[al_resid].mean().item():.4f}")
print(f"median modal_frac: full {mf_full[al_full].median().item():.4f}  "
      f"resid {mf_resid[al_resid].median().item():.4f}")
