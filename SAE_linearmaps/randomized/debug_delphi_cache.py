"""Localize a write_delphi_cache.py vs delphi disagreement: is it our h, or our encode?

diff_delphi_cache.py tells you THAT the caches disagree. This tells you WHERE. There are only
three places the two pipelines can diverge, and each has a different fix:

  (A) the ACTIVATION h        -- we take blocks.13.hook_resid_post from TransformerLens; delphi
                                 takes layers.13 from HF AutoModel.
  (B) the ENCODE path         -- we call sae_lens' sae.encode(h/scale); delphi calls a sparsify
                                 SparseCoder whose weights were FOLDED by convert_sae_to_sparsify.py.
                                 The fold is only correct if sae.encode does nothing beyond
                                 (x - b_dec) @ W_enc + b_enc -- e.g. no input normalization.
  (C) the CHECKPOINT          -- the archived cache may simply not have come from the SAE we think.

The test is a chain of comparisons on ONE batch. Each link isolates one hypothesis:

  1. pre_manual   = (h/scale - b_dec) @ W_enc + b_enc        our SAE, formula the CONVERTER assumed
  2. pre_encode   = whatever sae.encode(h/scale) actually computes (recovered from its support)
  3. pre_sparsify = h @ enc.weight.T + enc.bias              the WEIGHTS DELPHI ACTUALLY USED
  4. the reference cache's own firings for the same rows

  1 vs 3 disagree  -> (C): the sparsify dir was folded from a different checkpoint. Nothing about
                     our writer is wrong; find the checkpoint that produced it.
  1 vs 2 disagree  -> (B): sae_lens' encode does something the fold did not model, which would mean
                     the EXISTING delphi arms were scored with a subtly wrong SAE, not just these.
  1,2,3 all agree but 4 disagrees -> (A): our h differs from delphi's. That is the TL-vs-HF path,
                     and it would have to be far larger than check_hookpoint.py's cosine 0.999992
                     to explain a real disagreement.

Deliberately dependency-free on `sparsify`: the SAE directory is read as raw safetensors and the
encoder applied by hand, so this cannot drag torch forward in the training venv.

    VARIANT=trained HF_REPO=$TRAINED_REPO SAE_NAME=sae_full_k32_d73728_100M_topk_final.pt \
      REF_CACHE=/dev/shm/ref_trained/delphi_latents_L13_30M/layers.13 \
      python -u randomized/debug_delphi_cache.py

Env: SAE_NAME, REF_CACHE, SPARSIFY_REPO_DIR (default: downloads sparsify_topk_L13/ from HF_REPO),
     ROWS (default 32), MAX_LATENTS (default 500, only for the reference comparison).
"""
import glob
import json
import os
import sys

import numpy as np
import torch as t
from safetensors.numpy import load_file as load_np
from safetensors.torch import load_file as load_pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import D_IN, HOOK, LAYER, VARIANT, load_model
from hf_io import HF_REPO, pull
from sae_arch import arch_of
from sae_arch import load_sae as _rebuild_sae

SAE_NAME  = os.environ.get("SAE_NAME", "sae_full_k32_d73728_100M_topk_final.pt")
REF_CACHE = os.environ.get("REF_CACHE", "/dev/shm/ref_trained/delphi_latents_L13_30M/layers.13")
SP_DIR    = os.environ.get("SPARSIFY_REPO_DIR", "")
ROWS      = int(os.environ.get("ROWS", 32))
MAX_LATENTS = int(os.environ.get("MAX_LATENTS", 500))

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# ---- 0. what does the reference cache SAY it is? ----------------------------------------------
with open(f"{REF_CACHE}/config.json") as f:
    ref_cfg = json.load(f)
print("[debug] reference cache config.json:")
for k, v in ref_cfg.items():
    print(f"    {k}: {v}")

ref_shards = sorted(glob.glob(f"{REF_CACHE}/*.safetensors"))
tokens_np = load_np(ref_shards[0])["tokens"]
CTX_LEN = tokens_np.shape[1]
tok = t.from_numpy(tokens_np[:ROWS].astype(np.int64))
print(f"\n[debug] using rows [0, {ROWS}) x {CTX_LEN} = {ROWS * CTX_LEN:,} tokens")

# ---- 1. our SAE ------------------------------------------------------------------------------
ckpt = t.load(pull(SAE_NAME), weights_only=False)
sae = _rebuild_sae(ckpt)
sae.to(device).eval()
scale = float(ckpt["scale"])
sd = {k: v.detach().float() for k, v in sae.state_dict().items()}
W_enc, b_enc, b_dec = sd["W_enc"].to(device), sd["b_enc"].to(device), sd["b_dec"].to(device)
K = int(sae.cfg.k)
D_SAE = int(sae.cfg.d_sae)
print(f"[debug] {SAE_NAME}: arch={arch_of(ckpt)} d_sae={D_SAE} k={K} scale={scale:.6f} "
      f"mode={ckpt.get('mode')} apply_b_dec_to_input={getattr(sae.cfg, 'apply_b_dec_to_input', None)}")
for attr in ("normalize_activations", "normalize_sae_decoder", "scale_sparsity_penalty_by_decoder_norm"):
    if hasattr(sae.cfg, attr):
        print(f"    cfg.{attr} = {getattr(sae.cfg, attr)!r}")

# ---- 2. the sparsify weights delphi ACTUALLY used ----------------------------------------------
if not SP_DIR:
    from huggingface_hub import snapshot_download
    local = snapshot_download(HF_REPO, allow_patterns="sparsify_topk_L13/*",
                              local_dir="/dev/shm/sparsify_ref")
    SP_DIR = local
hits = glob.glob(f"{SP_DIR}/**/sae.safetensors", recursive=True) + \
       glob.glob(f"{SP_DIR}/**/*.safetensors", recursive=True)
if not hits:
    sys.exit(f"no safetensors under {SP_DIR!r} -- pass SPARSIFY_REPO_DIR explicitly")
sp_path = hits[0]
sp = {k: v.float().to(device) for k, v in load_pt(sp_path).items()}
print(f"\n[debug] sparsify weights: {sp_path}")
print(f"    keys: {sorted(sp.keys())}")
enc_w = sp.get("encoder.weight")
enc_b = sp.get("encoder.bias")
if enc_w is None:
    sys.exit(f"no encoder.weight in {sp_path} -- keys are {sorted(sp.keys())}")

# The fold the converter applied, recomputed here. If the sparsify dir came from THIS checkpoint,
# these must match to float tolerance -- that is exactly what convert_sae_to_sparsify.py asserted.
fold_w = (W_enc / scale).T
fold_b = b_enc - b_dec @ W_enc
dw = (enc_w - fold_w).abs().max().item() / max(fold_w.abs().max().item(), 1e-9)
db = (enc_b - fold_b).abs().max().item() / max(fold_b.abs().max().item(), 1e-9)
print(f"    encoder.weight vs our fold: max rel {dw:.3e}   {'OK' if dw < 1e-3 else 'MISMATCH'}")
print(f"    encoder.bias   vs our fold: max rel {db:.3e}   {'OK' if db < 1e-3 else 'MISMATCH'}")
if dw >= 1e-3 or db >= 1e-3:
    print("    -> (C) THE SPARSIFY DIR WAS NOT FOLDED FROM THIS CHECKPOINT. The archived cache "
          "came from a different SAE; our writer is not implicated. Find which .pt matches.")

# ---- 3. activations --------------------------------------------------------------------------
model = load_model(device)
with t.no_grad():
    _, cache = model.run_with_cache(tok.to(device), names_filter=[HOOK], stop_at_layer=LAYER + 1)
h = cache[HOOK].reshape(-1, D_IN).float()
print(f"\n[debug] h: {tuple(h.shape)} | mean |h| {h.norm(dim=-1).mean():.2f} | var {h.var():.4f}")

with t.no_grad():
    pre_manual = (h / scale - b_dec) @ W_enc + b_enc          # the converter's assumed formula
    acts_encode = sae.encode(h / scale)                        # what our writer actually calls
    pre_sparsify = h @ enc_w.T + enc_b                         # delphi's actual computation

# ---- 4. link by link ---------------------------------------------------------------------------
def support(x, k=K):
    return t.topk(x, k, dim=-1).indices.sort(dim=-1).values


sup_manual = support(pre_manual)
sup_sparsify = support(pre_sparsify)
enc_nz = (acts_encode.abs() > 1e-5)
enc_count = enc_nz.sum(-1).float()
print(f"\n[debug] sae.encode support size: mean {enc_count.mean():.2f} (expect {K})")

# 1 vs 3
rel13 = ((pre_manual - pre_sparsify).abs().max() / pre_manual.abs().max()).item()
ov13 = (sup_manual == sup_sparsify).all(-1).float().mean().item()
print(f"\n  [1 vs 3] our formula vs sparsify weights: max rel {rel13:.3e} | "
      f"identical top-{K} sets on {ov13:.4f} of tokens")

# 1 vs 2 -- does sae.encode agree with the formula the fold assumed?
sup_from_encode = t.zeros_like(sup_manual)
for i in range(acts_encode.shape[0]):
    idx = t.nonzero(enc_nz[i]).flatten()
    if idx.numel() == K:
        sup_from_encode[i] = idx.sort().values
ov12 = (sup_manual == sup_from_encode).all(-1).float().mean().item()
print(f"  [1 vs 2] our formula vs sae.encode:       identical top-{K} sets on {ov12:.4f} of tokens")
if ov12 < 0.99:
    print("    -> (B) sae_lens' encode is NOT the formula convert_sae_to_sparsify.py folded. "
          "That would mean the EXISTING delphi arms were scored with a mis-folded SAE too.")

# 4 -- the reference cache's own firings for these rows
ref_loc, ref_act = [], []
for p in ref_shards:
    first = int(os.path.basename(p).split("_")[0])
    d = load_np(p)
    loc = d["locations"].astype(np.int64)
    loc[:, 2] += first
    m = (loc[:, 0] < ROWS) & (loc[:, 2] < MAX_LATENTS)
    ref_loc.append(loc[m])
    ref_act.append(d["activations"].astype(np.float32)[m])
ref_loc, ref_act = np.concatenate(ref_loc), np.concatenate(ref_act)
print(f"\n  [4] reference firings in these rows, latents <{MAX_LATENTS}: {len(ref_act):,}")

flat = ref_loc[:, 0] * CTX_LEN + ref_loc[:, 1]
ref_keys = set(zip(flat.tolist(), ref_loc[:, 2].tolist()))


def ours_keys(sup):
    out = set()
    s = sup.cpu().numpy()
    for i in range(s.shape[0]):
        for j in s[i]:
            if j < MAX_LATENTS:
                out.add((i, int(j)))
    return out


for label, sup in (("our formula", sup_manual), ("sparsify weights on OUR h", sup_sparsify)):
    ok = ours_keys(sup)
    inter = len(ref_keys & ok)
    print(f"      {label:26s}: {len(ok):,} firings | {inter:,} shared with reference "
          f"({inter / max(len(ref_keys | ok), 1):.4f} Jaccard)")

print("\nREAD IT AS: if [1 vs 3] and [1 vs 2] are both clean but [4] is not, the divergence is in "
      "h (TL vs HF). If [1 vs 3] is a MISMATCH, the archived cache came from another checkpoint.")
