"""Does the SAE delphi actually ran reconstruct gemma at all? One number, no judge, no GPU-hours.

THE ARGUMENT THIS CLOSES. The case that the pre-2026-08-19 delphi numbers are invalid currently
rests on two things: the bug is provable from sparsify's source (its `encode` subtracts `b_dec`,
which `convert_sae_to_sparsify.py` ALSO folded into `encoder.bias`), and the corrected pipeline
agrees with our own GPT-judge pipeline where the broken one did not. Both are good arguments, but
they are arguments. This turns the claim into a measurement: an autoencoder whose encoder no longer
matches its decoder cannot RECONSTRUCT, and reconstruction error needs no judge and no opinion.

Three variants of the same checkpoint, on the same activations:

  A  sae_lens                 the trained object, as `eval_fvu.py` evaluates it
  B  sparsify AS ARCHIVED     exactly what delphi ran: relu/topk((h - b_dec) @ enc.W^T + enc.b),
                              decode acts @ W_dec + b_dec, with the weights from the HF repo
  C  sparsify, FIXED FOLD     the corrected conversion recomputed here from the sae_lens
                              checkpoint -- encoder.bias = b_enc * ||W_dec||, decoder norms folded

C exists as a control: it must reproduce A to float tolerance. If it does, the fold is right and
any gap between A and B is the damage, not an artifact of this script.

Both activation regimes are reported, because delphi cached BOS-free rows while our SAEs were
trained on BOS-prefixed contexts:

  BOS on   the regime the SAE was trained in -- the fair test of the dictionary
  BOS off  the regime delphi actually cached in -- what its scores were really computed over

FVU conventions follow eval_fvu.py: `flat` divides by the variance about a single scalar mean (the
project's historical convention), `centred` by the trace of the covariance (the statistically
correct fraction of variance, and the one to quote across arms).

    VARIANT=trained HF_REPO=$TRAINED_REPO SAE_NAME=sae_full_k32_d73728_100M_topk_final.pt \
      REF_CACHE=<any cache dir with tokens+config> \
      SPARSIFY_DIR=<dir containing sae.safetensors> \
      python -u randomized/fvu_converted_sae.py

Env: SAE_NAME, REF_CACHE, SPARSIFY_DIR, ROWS (default 2000 rows = 512k tokens), BATCH_ROWS.
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
from hf_io import pull
from sae_arch import arch_of
from sae_arch import load_sae as _rebuild_sae

SAE_NAME   = os.environ.get("SAE_NAME", "sae_full_k32_d73728_100M_topk_final.pt")
REF_CACHE  = os.environ.get("REF_CACHE", "/dev/shm/ref_small/delphi_latents_L13_30M_rand_full_writer/layers.13")
SP_DIR     = os.environ.get("SPARSIFY_DIR", "/dev/shm/sparsify_rand/sparsify_topk_L13/layers.13")
ROWS       = int(os.environ.get("ROWS", 2000))
BATCH_ROWS = int(os.environ.get("BATCH_ROWS", 32))

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# ---- tokens, straight out of a delphi cache (same text every arm is scored on) -----------------
shards = sorted(glob.glob(f"{REF_CACHE}/*.safetensors"))
if not shards:
    sys.exit(f"no shards in REF_CACHE={REF_CACHE!r}")
tokens_np = load_np(shards[0])["tokens"]
CTX = tokens_np.shape[1]
N_ROWS = min(ROWS, tokens_np.shape[0])
tokens = t.from_numpy(tokens_np[:N_ROWS].astype(np.int64))
with open(f"{REF_CACHE}/config.json") as f:
    ref_model = json.load(f).get("model_name", "google/gemma-2-2b")
from transformers import AutoTokenizer
BOS_ID = AutoTokenizer.from_pretrained(ref_model).bos_token_id
print(f"[fvu] {N_ROWS:,} rows x {CTX} = {N_ROWS * CTX:,} tokens | VARIANT={VARIANT} | BOS id {BOS_ID}")

# ---- the trained object -----------------------------------------------------------------------
ckpt = t.load(pull(SAE_NAME), weights_only=False)
sae = _rebuild_sae(ckpt); sae.to(device).eval()
scale = float(ckpt["scale"])
sd = {k: v.detach().float().to(device) for k, v in sae.state_dict().items()}
W_enc, b_enc, W_dec, b_dec = sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"]
K = int(sae.cfg.k)
print(f"  {SAE_NAME}: arch={arch_of(ckpt)} d_sae={int(sae.cfg.d_sae)} k={K} scale={scale:.6f} "
      f"mode={ckpt.get('mode')}")
if ckpt.get("mode") != "full":
    sys.exit("this comparison only makes sense for MODE=full -- sparsify cannot express resid")

# ---- the weights delphi actually ran ------------------------------------------------------------
hits = glob.glob(f"{SP_DIR}/**/sae.safetensors", recursive=True) or \
       glob.glob(f"{SP_DIR}/sae.safetensors")
if not hits:
    sys.exit(f"no sae.safetensors under SPARSIFY_DIR={SP_DIR!r}")
sp = {k: v.float().to(device) for k, v in load_pt(hits[0]).items()}
sp_encw, sp_encb, sp_Wdec, sp_bdec = sp["encoder.weight"], sp["encoder.bias"], sp["W_dec"], sp["b_dec"]
if sp_Wdec.shape != (W_dec.shape[0], W_dec.shape[1]):
    sp_Wdec = sp_Wdec.T                       # orientation differs between sparsify versions
print(f"  archived sparsify: {hits[0]}")

# ---- the corrected fold, recomputed here as a control -------------------------------------------
dnorm = W_dec.norm(dim=1)
fix_encw = ((W_enc / scale) * dnorm).T.contiguous()
fix_encb = (b_enc * dnorm).contiguous()
fix_Wdec = (W_dec / dnorm[:, None] * scale).contiguous()
fix_bdec = (b_dec * scale).contiguous()
print(f"  decoder norms: min {dnorm.min():.4f} max {dnorm.max():.4f} mean {dnorm.mean():.4f}")


def sparsify_recon(h, encw, encb, Wdec, bdec):
    """sparsify's own forward: it subtracts b_dec ITSELF, then top-k, then decode + b_dec."""
    pre = (h - bdec) @ encw.T + encb
    vals, idx = t.topk(pre, K, dim=-1)
    vals = vals.clamp(min=0)                   # sparsify applies relu in its activation
    acts = t.zeros_like(pre).scatter_(-1, idx, vals)
    return acts @ Wdec + bdec, idx


def saelens_recon(h):
    acts = sae.encode(h / scale)
    return sae.decode(acts) * scale, acts


model = load_model(device)
rows_per = BATCH_ROWS

results = {}
for bos in (True, False):
    sse = {"A_saelens": 0.0, "B_archived": 0.0, "C_fixed": 0.0}
    h_sum = t.zeros(D_IN, dtype=t.float64, device=device)
    h_sumsq, n = 0.0, 0
    overlap, nb = 0.0, 0
    with t.no_grad():
        for s in range(0, N_ROWS, rows_per):
            tok = tokens[s:s + rows_per].to(device)
            inp = t.cat([t.full((tok.shape[0], 1), BOS_ID, device=device, dtype=tok.dtype), tok],
                        dim=1) if bos else tok
            _, cache = model.run_with_cache(inp, names_filter=[HOOK], stop_at_layer=LAYER + 1)
            h3 = cache[HOOK]
            if bos:
                h3 = h3[:, 1:]
            h = h3.reshape(-1, D_IN).float()

            hA, actsA = saelens_recon(h)
            hB, idxB = sparsify_recon(h, sp_encw, sp_encb, sp_Wdec, sp_bdec)
            hC, idxC = sparsify_recon(h, fix_encw, fix_encb, fix_Wdec, fix_bdec)

            sse["A_saelens"] += ((h - hA) ** 2).double().sum().item()
            sse["B_archived"] += ((h - hB) ** 2).double().sum().item()
            sse["C_fixed"] += ((h - hC) ** 2).double().sum().item()
            h_sum += h.double().sum(0)
            h_sumsq += (h ** 2).double().sum().item()
            n += h.shape[0]

            # how often the archived conversion picks the SAME latents as the trained SAE
            a = t.topk(actsA, K, dim=-1).indices.sort(-1).values
            overlap += (a == idxB.sort(-1).values).all(-1).float().mean().item()
            nb += 1

    n_elem = n * D_IN
    var_flat = h_sumsq / n_elem - (h_sum.sum().item() / n_elem) ** 2
    ss_cent = h_sumsq - (h_sum ** 2).sum().item() / n
    results[bos] = {k: (v / (n_elem * var_flat), v / ss_cent) for k, v in sse.items()}
    results[bos]["_overlap"] = overlap / nb
    print(f"\n=== BOS {'ON (the SAE training regime)' if bos else 'OFF (what delphi cached)'} "
          f"— {n:,} tokens ===")
    print(f"{'variant':>22} {'FVU flat':>10} {'FVU centred':>12}")
    for k in ("A_saelens", "B_archived", "C_fixed"):
        f_, c_ = results[bos][k]
        print(f"{k:>22} {f_:10.4f} {c_:12.4f}")
    print(f"  archived conversion picks the trained SAE's exact top-{K} set on "
          f"{results[bos]['_overlap']:.4f} of tokens")

print("\nHOW TO READ THIS")
print("  C_fixed must match A_saelens -- that is the control proving the corrected fold is right.")
print("  B_archived is what delphi ran. An FVU near or above 1.0 means it explained no more of the")
print("  activation than predicting the mean, i.e. it was not a working dictionary of gemma at all,")
print("  and its 'features' were not features of any trained SAE.")
