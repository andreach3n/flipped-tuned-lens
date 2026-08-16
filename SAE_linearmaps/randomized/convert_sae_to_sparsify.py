"""Convert a sae_lens TopK / BatchTopK SAE into the sparsify format delphi loads.

Delphi (`python -m delphi <model> <sae_dir> --hookpoints ...`) calls
SparseCoder.load_from_disk(<sae_dir>/<hookpoint>), i.e. it wants a directory per hookpoint holding
cfg.json + sae.safetensors. Our SAEs are sae_lens checkpoints, so they have to be re-expressed. Three things have to be folded in, and getting any of them wrong produces
plausible-looking numbers rather than an error:

1. SCALE. Our SAEs were trained on x = h / scale, but delphi feeds the RAW activation h. The
   scalar therefore has to move into the weights.
2. apply_b_dec_to_input=True. Our encoder sees (x - b_dec), which is an extra bias term delphi
   knows nothing about.
3. TOP-K SEMANTICS -- exact for SAE_ARCH=topk, approximate for batchtopk. sparsify only offers
   per-token topk (and groupmax); there is no threshold/jumprelu option. A topk checkpoint
   therefore converts EXACTLY. A batchtopk one does not: sae_lens flattens the batch and keeps
   the top k*n_tokens overall (verified -- BatchTopK.forward has no self.training branch, so this
   holds at eval too), letting a token in a dull batch fire more than k. The agreement is printed
   below. Prefer training with SAE_ARCH=topk for anything destined for delphi: it is also the
   architecture Heap et al. used, so it removes two divergences at once.

The algebra, with `enc`/`dec` in sae_lens orientation (W_enc (d_in,d_sae), W_dec (d_sae,d_in)):
    pre  = (h/scale - b_dec) @ W_enc + b_enc  =  h @ (W_enc/scale) + (b_enc - b_dec @ W_enc)
    h^   = (acts @ W_dec + b_dec) * scale     =  acts @ (W_dec*scale) + (b_dec*scale)
so encoder.weight = (W_enc/scale).T, encoder.bias = b_enc - b_dec @ W_enc,
   W_dec *= scale, b_dec *= scale.

VERIFICATION is not optional here: the script checks the converted PRE-ACTIVATIONS match the
originals to float tolerance on random inputs. That isolates the folding math (which must be
exact) from the top-k semantics (which cannot be), and it is the only way to catch a silent
transpose or a dropped scale.

    HF_REPO=<arm repo> SAE_NAME=sae_full_k32_d73728_100M_final.pt \
      HOOKPOINT=model.layers.13 OUT_DIR=/dev/shm/sparsify_sae \
      python -u randomized/convert_sae_to_sparsify.py
"""
import json
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_io import pull
from sae_arch import load_sae as _rebuild_sae, arch_of

SAE_NAME  = os.environ.get("SAE_NAME", "sae_full_k32_d73728_100M_final.pt")
HOOKPOINT = os.environ.get("HOOKPOINT", "model.layers.13")
OUT_DIR   = os.environ.get("OUT_DIR", "/workspace/sparsify_sae")
TOL       = float(os.environ.get("TOL", 2e-3))     # relative, on pre-activations

# ---- load ours ------------------------------------------------------------------------------

ckpt = t.load(pull(SAE_NAME), weights_only=False)
sae = _rebuild_sae(ckpt)
sae.eval()
scale = float(ckpt["scale"])
sd = sae.state_dict()
W_enc, b_enc = sd["W_enc"], sd["b_enc"]          # (d_in, d_sae), (d_sae,)
W_dec, b_dec = sd["W_dec"], sd["b_dec"]          # (d_sae, d_in), (d_in,)
d_in, d_sae = W_enc.shape
k = int(sae.cfg.k)
assert bool(getattr(sae.cfg, "apply_b_dec_to_input", False)), \
    "this conversion assumes apply_b_dec_to_input=True, as all our runs used"
print(f"[convert] {SAE_NAME}: d_in={d_in} d_sae={d_sae} k={k} scale={scale:.6f}")

# ---- fold ----------------------------------------------------------------------------------
enc_W = (W_enc / scale).T.contiguous()           # (d_sae, d_in)
enc_b = (b_enc - b_dec @ W_enc).contiguous()     # (d_sae,)
dec_W = (W_dec * scale).contiguous()             # (d_sae, d_in)
dec_b = (b_dec * scale).contiguous()             # (d_in,)

# ---- build the sparsify object and let IT define the shapes ---------------------------------
# Assigning into a real SparseCoder (rather than hand-writing safetensors) means sparsify's own
# orientation conventions are authoritative -- a shape mismatch raises here instead of silently
# transposing a 73728x2304 matrix.
from sparsify import SparseCoder, SparseCoderConfig

sc = SparseCoder(d_in=d_in, cfg=SparseCoderConfig(num_latents=d_sae, k=k), device="cpu")
with t.no_grad():
    assert sc.encoder.weight.shape == enc_W.shape, (sc.encoder.weight.shape, enc_W.shape)
    assert sc.encoder.bias.shape == enc_b.shape, (sc.encoder.bias.shape, enc_b.shape)
    sc.encoder.weight.copy_(enc_W)
    sc.encoder.bias.copy_(enc_b)
    # W_dec orientation differs between versions; pick whichever matches and say which.
    if sc.W_dec.shape == dec_W.shape:
        sc.W_dec.copy_(dec_W);        print("  W_dec: (d_sae, d_in)")
    elif sc.W_dec.shape == dec_W.T.shape:
        sc.W_dec.copy_(dec_W.T);      print("  W_dec: (d_in, d_sae) -- transposed on write")
    else:
        sys.exit(f"cannot place W_dec: sparsify wants {tuple(sc.W_dec.shape)}, ours is {tuple(dec_W.shape)}")
    sc.b_dec.copy_(dec_b)

# ---- verify the folding is EXACT on pre-activations -----------------------------------------
# Pre-activations are the part that must match bit-for-bit-ish; the top-k mask is where the two
# libraries legitimately differ, so we measure that separately rather than assert on it.
g = t.Generator().manual_seed(0)
h = t.randn(256, d_in, generator=g) * 30.0                      # ~the scale of real gemma acts
with t.no_grad():
    ours = (h / scale - b_dec) @ W_enc + b_enc
    theirs = h @ sc.encoder.weight.T + sc.encoder.bias
rel = ((ours - theirs).abs().max() / ours.abs().max()).item()
print(f"  pre-activation max rel err: {rel:.2e}  ({'OK' if rel < TOL else 'FAIL'})")
if rel >= TOL:
    sys.exit("folding is wrong -- do NOT use this checkpoint")

# how much does the top-k convention actually change? report, do not gate.
with t.no_grad():
    flat = ours.flatten()
    thr = flat.topk(k * ours.shape[0]).values.min()             # sae_lens batch-topk threshold
    batch_mask = ours >= thr
    tok_mask = t.zeros_like(ours, dtype=t.bool)
    tok_mask.scatter_(1, ours.topk(k, dim=-1).indices, True)    # sparsify per-token topk
    agree = (batch_mask & tok_mask).sum().item() / max(tok_mask.sum().item(), 1)
print(f"  top-k mask agreement batch-vs-per-token: {100*agree:.1f}% of fired latents")

os.makedirs(f"{OUT_DIR}/{HOOKPOINT}", exist_ok=True)
sc.save_to_disk(f"{OUT_DIR}/{HOOKPOINT}")
print(f"  saved -> {OUT_DIR}/{HOOKPOINT}  (delphi: --hookpoints {HOOKPOINT})")
print(f"  contents: {sorted(os.listdir(f'{OUT_DIR}/{HOOKPOINT}'))}")
