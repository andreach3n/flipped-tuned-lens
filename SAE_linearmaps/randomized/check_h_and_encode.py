"""Is our h the same tensor delphi cached, on REAL cache rows? And what does sae.encode compute?

debug_delphi_cache.py narrowed a cache disagreement to two things. This settles both.

WHY check_hookpoint.py DID NOT ALREADY SETTLE THE FIRST. That script compares TransformerLens
`blocks.13.hook_resid_post` against HF `layers.13` on a sentence built with `tl.to_tokens(TEXT)` --
which PREPENDS BOS. delphi's cache rows contain no BOS at all: `filter_bos` deletes every BOS from
the flattened token stream and re-chunks to 256. Gemma-2 has a strong BOS attention sink, so a
BOS-free 256-token row is a different regime from the one that check passed in, and the cosine
0.999992 it reported does not transfer. Measured here on the actual rows instead.

THE DECISIVE COMPARISON is not h itself but what the SAE does with it. Two activations can look
close and still select different top-k latents, because top-k is a RANKING. So this applies the
SAME sparsify encoder delphi used to both TL's h and HF's h, and scores each against the reference
cache's own firings:

    TL  h -> sparsify encoder -> top-k  vs  reference   (are WE right?)
    HF  h -> sparsify encoder -> top-k  vs  reference   (is delphi's path reproducible at all?)

If HF matches the reference and TL does not, the divergence is the activation source, and patching
gemma to emit h - P[tok] (so delphi computes h itself) is the correct route rather than writing the
cache from TransformerLens activations.
If NEITHER matches, the disagreement is not the model backend and something else is wrong -- most
likely the row indexing convention, which this also prints enough to spot.

SECOND QUESTION: sae_lens' encode. convert_sae_to_sparsify.py verified its fold against a
hand-written (x - b_dec) @ W_enc + b_enc, never against sae.encode. If those differ, the sparsify
SAE is not the trained SAE and the EXISTING delphi arms inherit the problem. Printed as raw
indices and values for one token, which is unambiguous in a way a set-overlap statistic is not.

    VARIANT=trained HF_REPO=$TRAINED_REPO SAE_NAME=sae_full_k32_d73728_100M_topk_final.pt \
      REF_CACHE=/dev/shm/ref_trained/delphi_latents_L13_30M/layers.13 \
      SPARSIFY_REPO_DIR=/dev/shm/sparsify_ref \
      python -u randomized/check_h_and_encode.py

Env: SAE_NAME, REF_CACHE, SPARSIFY_REPO_DIR, ROWS (default 32), MAX_LATENTS (default 500).
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
from activations import D_IN, HOOK, LAYER, MODEL_NAME, VARIANT, load_model
from hf_io import HF_REPO, pull
from sae_arch import load_sae as _rebuild_sae

SAE_NAME    = os.environ.get("SAE_NAME", "sae_full_k32_d73728_100M_topk_final.pt")
REF_CACHE   = os.environ.get("REF_CACHE", "/dev/shm/ref_trained/delphi_latents_L13_30M/layers.13")
SP_DIR      = os.environ.get("SPARSIFY_REPO_DIR", "/dev/shm/sparsify_ref")
ROWS        = int(os.environ.get("ROWS", 32))
MAX_LATENTS = int(os.environ.get("MAX_LATENTS", 500))

device = t.device("cuda" if t.cuda.is_available() else "cpu")

with open(f"{REF_CACHE}/config.json") as f:
    ref_cfg = json.load(f)
ref_model = ref_cfg.get("model_name", MODEL_NAME)
ref_shards = sorted(glob.glob(f"{REF_CACHE}/*.safetensors"))
tokens_np = load_np(ref_shards[0])["tokens"]
CTX_LEN = tokens_np.shape[1]
tok = t.from_numpy(tokens_np[:ROWS].astype(np.int64)).to(device)
print(f"[check] rows [0,{ROWS}) x {CTX_LEN} | cache model_name={ref_model}")

# BOS presence in the rows -- if filter_bos did its job there should be none at all.
sae_ckpt = t.load(pull(SAE_NAME), weights_only=False)
sae = _rebuild_sae(sae_ckpt); sae.to(device).eval()
scale = float(sae_ckpt["scale"])
sd = {k: v.detach().float().to(device) for k, v in sae.state_dict().items()}
W_enc, b_enc, b_dec = sd["W_enc"], sd["b_enc"], sd["b_dec"]
K = int(sae.cfg.k)

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(ref_model)
n_bos = int((tok == tk.bos_token_id).sum())
print(f"  BOS id {tk.bos_token_id} appears {n_bos} times in these rows (expect 0 -- filter_bos)")

# ---- our h: TransformerLens -------------------------------------------------------------------
tl = load_model(device)
with t.no_grad():
    _, cache = tl.run_with_cache(tok, names_filter=[HOOK], stop_at_layer=LAYER + 1)
h_tl = cache[HOOK].reshape(-1, D_IN).float()
del cache
print(f"\n  TL h: mean |h| {h_tl.norm(dim=-1).mean():.2f} | var {h_tl.var():.4f}")

# ---- delphi's h: HF AutoModel, hooked exactly as delphi hooks it -------------------------------
# delphi's load_artifacts uses AutoModel (not ForCausalLM) in bf16 on cuda, and its hook takes
# output[0] when the module returns a tuple -- which decoder layers do.
from transformers import AutoModel
hf = AutoModel.from_pretrained(ref_model, torch_dtype=t.bfloat16).to(device).eval()
grabbed = {}


def _hook(_m, _i, output):
    grabbed["x"] = output[0] if isinstance(output, tuple) else output


target = [n for n, _ in hf.named_modules() if n.endswith(f"layers.{LAYER}")][0]
handle = dict(hf.named_modules())[target].register_forward_hook(_hook)
with t.no_grad():
    hf(tok)
handle.remove()
h_hf = grabbed["x"].reshape(-1, D_IN).float()
print(f"  HF h ({target}): mean |h| {h_hf.norm(dim=-1).mean():.2f} | var {h_hf.var():.4f}")

cos = t.nn.functional.cosine_similarity(h_tl.flatten(), h_hf.flatten(), dim=0).item()
rel = ((h_tl - h_hf).abs().max() / h_tl.abs().max()).item()
percos = t.nn.functional.cosine_similarity(h_tl, h_hf, dim=-1)
print(f"  TL vs HF: cosine {cos:.6f} | max rel {rel:.3e} | per-token cosine "
      f"min {percos.min():.6f} mean {percos.mean():.6f}")

# ---- the sparsify encoder delphi actually ran ---------------------------------------------------
hits = glob.glob(f"{SP_DIR}/**/sae.safetensors", recursive=True)
if not hits:
    sys.exit(f"no sae.safetensors under {SP_DIR!r}")
sp = {k: v.float().to(device) for k, v in load_pt(hits[0]).items()}
enc_w, enc_b = sp["encoder.weight"], sp["encoder.bias"]
print(f"\n  sparsify encoder: {hits[0]}")


sp_b_dec = sp["b_dec"]


def firings(h, subtract_b_dec, k=K):
    """(flat_pos, latent) pairs for these activations, restricted to <MAX_LATENTS.

    `subtract_b_dec` is the whole question. sparsify's SparseCoder.encode does
        x = x - self.b_dec ; fused_encoder(x, encoder.weight, encoder.bias, k, act)
    i.e. it applies the b_dec correction ITSELF. convert_sae_to_sparsify.py additionally folded
    b_dec into encoder.bias (b_enc - b_dec @ W_enc), so delphi subtracts it TWICE. The converter
    never caught this because it verified `h @ encoder.weight.T + encoder.bias` by hand rather
    than calling encode.
    """
    with t.no_grad():
        x = (h - sp_b_dec) if subtract_b_dec else h
        pre = x @ enc_w.T + enc_b
        idx = t.topk(pre, k, dim=-1).indices
    out = set()
    a = idx.cpu().numpy()
    for i in range(a.shape[0]):
        for j in a[i]:
            if j < MAX_LATENTS:
                out.add((i, int(j)))
    return out


# ---- the reference's own firings for these rows -------------------------------------------------
ref = set()
for p in ref_shards:
    first = int(os.path.basename(p).split("_")[0])
    d = load_np(p)
    loc = d["locations"].astype(np.int64)
    loc[:, 2] += first
    m = (loc[:, 0] < ROWS) & (loc[:, 2] < MAX_LATENTS)
    sel = loc[m]
    for r, c, l in sel:
        ref.add((int(r) * CTX_LEN + int(c), int(l)))

print(f"\n  reference firings (rows<{ROWS}, latents<{MAX_LATENTS}): {len(ref):,}")
best = None
for label, h in (("TL h", h_tl), ("HF h", h_hf)):
    for sub in (False, True):
        ours = firings(h, sub)
        inter = len(ref & ours)
        jac = inter / max(len(ref | ours), 1)
        tag = "encode() incl. its own -b_dec" if sub else "encoder(h) as the converter assumed"
        print(f"    {label} | {tag:36s}: {len(ours):,} firings | {inter:,} shared | Jaccard {jac:.4f}")
        if best is None or jac > best[2]:
            best = (label, sub, jac)

print(f"\n  VERDICT: best is {best[0]} with subtract_b_dec={best[1]} (Jaccard {best[2]:.4f})")
if best[2] > 0.95 and best[1]:
    print("    -> CONFIRMED: delphi applies b_dec TWICE, because convert_sae_to_sparsify.py folded "
          "it into encoder.bias AND left it in b_dec, while sparsify's encode subtracts it itself.\n"
          "    -> The fix is encoder.bias = b_enc (drop the -b_dec @ W_enc term).\n"
          "    -> EVERY delphi result in this project so far was produced with this mis-biased "
          "encoder and needs rerunning.")
elif best[2] > 0.95:
    print("    -> that configuration reproduces delphi's cache; the converter's assumption held.")
else:
    print("    -> still no match. The disagreement is neither TL-vs-HF nor the b_dec convention; "
          "suspect the row/position indexing or the tokens<->locations alignment.")

# ---- second question: what does sae_lens' encode actually compute? -----------------------------
print("\n  [encode] one token (flat position 0), top-8 by each route:")
with t.no_grad():
    pre_manual = (h_tl[:1] / scale - b_dec) @ W_enc + b_enc
    acts = sae.encode(h_tl[:1] / scale)
mv, mi = t.topk(pre_manual[0], 8)
nz = t.nonzero(acts[0].abs() > 1e-5).flatten()
av, ai = t.topk(acts[0], 8)
print(f"    manual preacts   idx {mi.tolist()}")
print(f"                     val {[round(v, 3) for v in mv.tolist()]}")
print(f"    sae.encode       idx {ai.tolist()}")
print(f"                     val {[round(v, 3) for v in av.tolist()]}")
print(f"    sae.encode nonzeros: {nz.numel()} (k={K})")
same = set(mi.tolist()) == set(ai.tolist())
print(f"    same top-8 latents: {same}")

# The values differ even when the latents agree, so encode applies a PER-LATENT factor. Identify
# it: sae_lens variants rescale feature acts by the decoder column norm. If the ratio tracks
# ||W_dec_j|| (or its reciprocal), that is the whole story -- and it is harmless downstream,
# because delphi ranks examples WITHIN a latent, where a positive per-latent factor cancels.
# What it does change is which latents win the top-k at a token, i.e. the selection.
with t.no_grad():
    ratio = (acts[0][mi] / pre_manual[0][mi]).cpu()
    dnorm = sd["W_dec"][mi].norm(dim=-1).cpu()
print(f"    encode/manual ratio: {[round(v, 4) for v in ratio.tolist()]}")
print(f"    ||W_dec_j||        : {[round(v, 4) for v in dnorm.tolist()]}")
print(f"    ratio * ||W_dec||  : {[round(v, 4) for v in (ratio * dnorm).tolist()]}")
print("    (a constant row in the last line means encode = manual / ||W_dec_j||)")
