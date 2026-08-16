"""Streaming FVU evaluation for ONE arm -- ported off the /workspace/sae_cache_layer13 cache.

Activations are generated on the fly (VARIANT-aware, so this works on trained and random arms
alike) and every artifact is pulled from that arm's HF_REPO. Run it once per arm and compare the
JSONs; nothing here reads the other arm, so the env vars fully determine what is measured:

    VARIANT=trained  HF_REPO=<trained-repo>  OUT_DIR=... python -u eval_fvu.py
    VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-repo> OUT_DIR=... python -u eval_fvu.py

Why not just read the training logs: their FVU is a single-batch estimate at whatever step the run
happened to end on. This is a clean pass over ~2M HELD-OUT tokens (a different stream seed than
training's 0), which is the number to quote.

TWO FVU CONVENTIONS ARE REPORTED, and the difference matters here:
  flat     -- divide by Var over the FLATTENED tensor (h.var()), i.e. spread about a single SCALAR
              mean. This is what train_sae_res.py / fit_map.py have always used, so it is the one
              comparable to the project's existing 0.14 / 0.55 numbers.
  centred  -- divide by the trace of the covariance, i.e. spread about the per-DIMENSION mean.
              This is the statistically correct fraction-of-variance, and it is the one to trust
              when comparing ACROSS arms: the flat denominator absorbs the cross-dimension spread
              of the mean vector, which differs between a trained and a random model and would
              otherwise leak into the comparison.
context_var.py reports the centred convention only; see its docstring.

WHICH CHECKPOINT: K + SUFFIX + CKPT only pick a FILENAME (`sae_{mode}_k{K}{SUFFIX}_{CKPT}.pt`).
Defaults reproduce the historical 20M/16k artifacts, so old invocations are unchanged. For the
100M / R=32 fleet, and for the milestone ladder that gives FVU-vs-training-tokens:

    # trained arm, final
    VARIANT=trained HF_REPO=<trained-repo> K=32 SUFFIX=_d73728_100M \
      OUT_DIR=/dev/shm/out_eval python -u eval_fvu.py
    # random arm, final
    VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-repo> K=32 SUFFIX=_d73728_100M \
      OUT_DIR=/dev/shm/out_eval python -u eval_fvu.py
    # the ladder (per arm): CKPT=t20M, t40M, t60M, t80M
    ... K=32 SUFFIX=_d73728_100M CKPT=t40M python -u eval_fvu.py

Every run echoes the LOADED d_sae/k, so a filename that resolved to the wrong config is visible
before the 2M-token pass starts. Remember raw FVU is uninterpretable across configs without a
matched-Gaussian floor -- gauss_null.py must be re-run at whatever config you evaluate here.

Env: K, SUFFIX, CKPT, N_TOKENS, BATCH, SEED, OUT_DIR (+ VARIANT / INIT_SEED / HF_REPO from the arm).
"""
import os
import json
import torch as t
import torch.nn as nn

from activations import (load_model, activation_stream, D_IN, LAYER, VARIANT, INIT_SEED, MAP_FILE)
from hf_io import push, pull
from sae_arch import load_sae as _rebuild_sae, arch_of

K_SAE    = int(os.environ.get("K", 64))
SUFFIX   = os.environ.get("SUFFIX", "")                 # tagged runs, e.g. _d73728_100M (see train_sae_res.py)
CKPT     = os.environ.get("CKPT", "final")              # "final" | "t20M" | "t40M" | ... (milestone ladder)
N_TOKENS = int(os.environ.get("N_TOKENS", 2_000_000))   # FVU converges fast; 2M is plenty
BATCH    = int(os.environ.get("BATCH", 8192))
SEED     = int(os.environ.get("SEED", 7))               # != training's 0 -> effectively held-out docs
OUT_DIR  = os.environ.get("OUT_DIR", "/workspace/out")
os.makedirs(OUT_DIR, exist_ok=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model  = load_model(device)
V      = model.cfg.d_vocab
print(f"[eval_fvu] VARIANT={VARIANT} INIT_SEED={INIT_SEED} layer={LAYER} K={K_SAE} "
      f"SUFFIX={SUFFIX!r} CKPT={CKPT} N_TOKENS={N_TOKENS:,} stream_seed={SEED}")


def load_sae(name):
    """Pull a checkpoint from this arm's HF repo and rebuild the SAE + its activation scale."""
    ckpt = t.load(pull(name), weights_only=False)
    sae = _rebuild_sae(ckpt)
    sae.to(device).eval()
    # Echo the geometry actually loaded. K/SUFFIX only pick a FILENAME -- this is the cheap check
    # that the file holds the config you think it does (e.g. d_sae 73728, not the old 16384).
    print(f"  {name}: arch={arch_of(ckpt)} d_sae={sae.cfg.d_sae} k={sae.cfg.k} step={ckpt.get('step')} "
          f"tokens={ckpt.get('tokens', 'final')}")
    return sae, float(ckpt["scale"]), ckpt


def try_load(name):
    try:
        return load_sae(name)
    except Exception as e:
        print(f"  [skip] {name} not in this arm's repo ({type(e).__name__})")
        return None


sae_full,  scale_full,  _ = load_sae(f"sae_full_k{K_SAE}{SUFFIX}_{CKPT}.pt")
sae_resid, scale_resid, _ = load_sae(f"sae_resid_k{K_SAE}{SUFFIX}_{CKPT}.pt")

# The frozen greedy map, rebuilt into a (V, 2304) lookup exactly as train_sae_res.py does.
# P.pt is NOT read from disk -- it is pure scratch there and gets deleted between runs.
linear_map = nn.Linear(D_IN, D_IN).to(device)
linear_map.load_state_dict(t.load(pull(MAP_FILE), weights_only=False))
linear_map.eval()
with t.no_grad():
    embed_table = model.embed(t.arange(V, device=device)).float()
    P = linear_map(embed_table)                       # (V, 2304)

# hybrid / outbias are optional -- included only if this arm has them (same guarded pattern as before).
# Their P table comes from the checkpoint's JOINTLY-TRAINED map, not the frozen one above.
_hyb = try_load(f"sae_hybrid_k{K_SAE}{SUFFIX}_{CKPT}.pt")
_out = try_load(f"sae_outbias_k{K_SAE}{SUFFIX}_{CKPT}.pt")


def _trained_P(ckpt, emb):
    """Rebuild a jointly-trained map's (V, 2304) table. `emb` is passed in, not closed over,
    so this stays correct no matter where the embed table is freed."""
    lm = nn.Linear(D_IN, D_IN).to(device)
    lm.load_state_dict(ckpt["linear_map"])
    lm.eval()
    with t.no_grad():
        return lm(emb)


HAVE_HYBRID = _hyb is not None
if HAVE_HYBRID:
    sae_hybrid, scale_hybrid, _ck = _hyb
    P_hybrid = _trained_P(_ck, embed_table)
    print("hybrid artifacts found -> including HYBRID FVU")

HAVE_OUTBIAS = _out is not None
if HAVE_OUTBIAS:
    sae_outbias, scale_outbias, _ck = _out
    P_outbias = _trained_P(_ck, embed_table)
    print("outbias artifacts found -> including OUTBIAS FVU")

del embed_table
t.cuda.empty_cache()


def recon_full(hh):
    a = sae_full.encode(hh / scale_full)
    return sae_full.decode(a) * scale_full            # back to raw units


def recon_resid(hh, tt):
    p = P[tt]                                         # (b, 2304) map's prediction
    r = hh - p
    r_hat = sae_resid.decode(sae_resid.encode(r / scale_resid)) * scale_resid
    return p + r_hat, r, r_hat                        # composite h_hat, plus r/r_hat for the side-metric


def recon_hybrid(hh, tt):
    p = P_hybrid[tt]                                  # the JOINTLY-TRAINED prediction
    a = sae_hybrid.encode((hh - p) / scale_hybrid)
    return p + sae_hybrid.decode(a) * scale_hybrid


def recon_outbias(hh, tt):
    a = sae_outbias.encode(hh / scale_outbias)        # encoder sees the FULL h (token NOT removed)
    return P_outbias[tt] + sae_outbias.decode(a) * scale_outbias   # map added at the OUTPUT


# float64 accumulators: summing ~5e9 float32 squares drifts; .double() first avoids it.
# Per-DIMENSION sums (not just a grand total) are what make the centred convention available.
sse_full = sse_comp = sse_r = sse_hybrid = sse_outbias = 0.0
h_sum = t.zeros(D_IN, dtype=t.float64, device=device)
r_sum = t.zeros(D_IN, dtype=t.float64, device=device)
h_sumsq = r_sumsq = 0.0
n_rows = 0

with t.no_grad():
    for step, (hh, tt) in enumerate(activation_stream(
            model, device, batch=BATCH, seed=SEED, max_tokens=N_TOKENS)):
        h_hat_full = recon_full(hh)
        h_hat_comp, r, r_hat = recon_resid(hh, tt)

        sse_full += ((hh - h_hat_full) ** 2).double().sum().item()
        sse_comp += ((hh - h_hat_comp) ** 2).double().sum().item()
        sse_r    += ((r - r_hat) ** 2).double().sum().item()
        if HAVE_HYBRID:
            sse_hybrid += ((hh - recon_hybrid(hh, tt)) ** 2).double().sum().item()
        if HAVE_OUTBIAS:
            sse_outbias += ((hh - recon_outbias(hh, tt)) ** 2).double().sum().item()

        h_sum += hh.double().sum(0);  h_sumsq += (hh ** 2).double().sum().item()
        r_sum += r.double().sum(0);   r_sumsq += (r ** 2).double().sum().item()
        n_rows += hh.shape[0]
        if step % 50 == 0:
            print(f"  {n_rows:,}/{N_TOKENS:,} tokens")

n_elem = n_rows * D_IN

# flat: spread about the SCALAR grand mean (the project's historical convention)
var_h_flat = h_sumsq / n_elem - (h_sum.sum().item() / n_elem) ** 2
var_r_flat = r_sumsq / n_elem - (r_sum.sum().item() / n_elem) ** 2
# centred: trace of the covariance -- sum_i ||x_i - xbar||^2 with xbar the per-DIM mean vector
ss_h_cent = h_sumsq - (h_sum ** 2).sum().item() / n_rows
ss_r_cent = r_sumsq - (r_sum ** 2).sum().item() / n_rows


def fvu(sse, flat_var, cent_ss):
    return sse / (n_elem * flat_var), sse / cent_ss


full_flat,  full_cent  = fvu(sse_full, var_h_flat, ss_h_cent)
comp_flat,  comp_cent  = fvu(sse_comp, var_h_flat, ss_h_cent)
residr_flat, residr_cent = fvu(sse_r,  var_r_flat, ss_r_cent)

print(f"\n=== {VARIANT} (seed {INIT_SEED}), {n_rows:,} tokens ===")
print(f"{'':32s} {'flat':>9s} {'centred':>9s}")
print(f"{'Var(r)/Var(h)':32s} {var_r_flat/var_h_flat:>9.4f} {ss_r_cent/ss_h_cent:>9.4f}")
print(f"{'FVU on h  - full SAE':32s} {full_flat:>9.4f} {full_cent:>9.4f}")
print(f"{'FVU on h  - map + resid SAE':32s} {comp_flat:>9.4f} {comp_cent:>9.4f}")
print(f"{'FVU on r  - resid SAE alone':32s} {residr_flat:>9.4f} {residr_cent:>9.4f}")
# h - (p + r_hat) == r - r_hat identically, so this must hold to floating-point noise
print(f"identity check: {residr_flat * var_r_flat / var_h_flat:.6f} == {comp_flat:.6f} (composite)")

results = {
    "variant": VARIANT, "init_seed": INIT_SEED, "layer": LAYER, "k": K_SAE,
    "suffix": SUFFIX, "ckpt": CKPT, "d_sae": int(sae_full.cfg.d_sae),   # provenance: WHICH SAEs these are
    "n_tokens": n_rows, "stream_seed": SEED,
    "var_ratio_r_over_h": {"flat": var_r_flat / var_h_flat, "centred": ss_r_cent / ss_h_cent},
    "fvu_h_full":      {"flat": full_flat,  "centred": full_cent},
    "fvu_h_composite": {"flat": comp_flat,  "centred": comp_cent},
    "fvu_r_resid":     {"flat": residr_flat, "centred": residr_cent},
}
if HAVE_HYBRID:
    hf_, hc_ = fvu(sse_hybrid, var_h_flat, ss_h_cent)
    print(f"{'FVU on h  - hybrid (joint)':32s} {hf_:>9.4f} {hc_:>9.4f}")
    results["fvu_h_hybrid"] = {"flat": hf_, "centred": hc_}
if HAVE_OUTBIAS:
    of_, oc_ = fvu(sse_outbias, var_h_flat, ss_h_cent)
    print(f"{'FVU on h  - outbias':32s} {of_:>9.4f} {oc_:>9.4f}")
    results["fvu_h_outbias"] = {"flat": of_, "centred": oc_}

_tag = "" if CKPT == "final" else f"_{CKPT}"   # keep historical filenames unchanged
path = f"{OUT_DIR}/fvu_{VARIANT}_s{INIT_SEED}_k{K_SAE}{SUFFIX}{_tag}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {path}")
push(path)
if device.type == "cuda":
    print(f"peak GPU memory: {t.cuda.max_memory_allocated()/2**30:.1f} GiB")
