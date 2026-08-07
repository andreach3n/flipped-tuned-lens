"""Tier-0 of the trained-vs-random experiment: how much of h_13 is NOT explained by token identity?

No SAE required. This measures the CONTEXTUAL VARIANCE FRACTION

    rho = E_tok[ Var(h | tok) ] / Var(h)

i.e. the share of layer-13 residual variance that survives a per-token lookup table. Because the
lookup table is the BEST possible token-static predictor, 1 - rho upper-bounds the linear map's R^2
(the map W·embed[tok] + b is just a rank-2304-constrained lookup table). So this one number says how
much contextual structure there is to find, before spending a single GPU-hour on SAE training.

The premise of the whole thread is  rho_trained >> rho_random:  a random stack barely mixes context,
so nearly all of its h_13 is a function of the current token. Run both arms and compare:

    VARIANT=trained  OUT_DIR=/workspace/out_trained  python -u context_var.py
    VARIANT=rand_all INIT_SEED=0 HF_REPO=<per-arm-repo> OUT_DIR=/workspace/out_rand0 python -u context_var.py

Implementation: one streaming pass accumulating per-token sums, using the within-group
sum-of-squares identity

    sum_tok sum_i ||h_i - hbar_tok||^2  =  sum_i ||h_i||^2  -  sum_tok n_tok ||hbar_tok||^2

so nothing is stored per-example. All quantities are TRACE variances (summed over the 2304 dims),
which is what makes the ratio the fraction-of-variance interpretation.

Env: N_TOKENS, MIN_COUNTS, ACC_DTYPE, OUT_DIR (+ VARIANT / INIT_SEED / HF_REPO from the arm).
"""
import os
import json
import torch as t

from activations import load_model, activation_stream, D_IN, LAYER, VARIANT, INIT_SEED
from hf_io import push, pull

N_TOKENS  = int(os.environ.get("N_TOKENS", 20_000_000))
BATCH     = int(os.environ.get("BATCH", 8192))
SEED      = int(os.environ.get("SEED", 0))
OUT_DIR   = os.environ.get("OUT_DIR", "/workspace/out")
# tokens seen only a handful of times contribute ~0 within-token variance and bias rho DOWNWARD,
# so we report rho over several minimum-count thresholds plus the coverage each one retains.
MIN_COUNTS = [int(x) for x in os.environ.get("MIN_COUNTS", "1,2,10,30,100").split(",")]
# float64 accumulators: a frequent token can absorb ~1e6 additions, where float32's ~7 digits
# would visibly erode the per-token means. (V, D) in float64 is ~4.7 GB -- fits alongside gemma.
ACC_DTYPE = getattr(t, os.environ.get("ACC_DTYPE", "float64"))
os.makedirs(OUT_DIR, exist_ok=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model  = load_model(device)
V      = model.cfg.d_vocab

print(f"[context_var] VARIANT={VARIANT} INIT_SEED={INIT_SEED} layer={LAYER} "
      f"N_TOKENS={N_TOKENS:,} vocab={V:,} acc={ACC_DTYPE}")

# ---- streaming accumulators ----
sum_h     = t.zeros(V, D_IN, dtype=ACC_DTYPE, device=device)   # per-token sum of h
sumsq_tok = t.zeros(V,       dtype=t.float64, device=device)   # per-token sum of ||h||^2
count     = t.zeros(V,       dtype=t.float64, device=device)   # per-token occurrence count
seen = 0

with t.no_grad():
    for step, (h_b, tok_b) in enumerate(activation_stream(
            model, device, batch=BATCH, seed=SEED, max_tokens=N_TOKENS)):
        h = h_b.to(ACC_DTYPE)
        # index_add_ scatters each row of h into the row of sum_h named by its token id,
        # accumulating (not overwriting) when a token repeats within the batch.
        sum_h.index_add_(0, tok_b, h)
        sumsq_tok.index_add_(0, tok_b, (h.double() ** 2).sum(-1))
        count.index_add_(0, tok_b, t.ones_like(tok_b, dtype=t.float64))
        seen += h_b.shape[0]
        if step % 200 == 0:
            print(f"  {seen:,}/{N_TOKENS:,} tokens")

print(f"  done: {seen:,} tokens over {int((count > 0).sum()):,} distinct token ids")

# ---- exact rho for any token subset, straight from the accumulators ----
# For a subset S of token ids:
#   N_S      = sum_{tok in S} n_tok
#   total_S  = sum_{tok in S} sumsq_tok  -  ||sum_{tok in S} sum_h[tok]||^2 / N_S
#   within_S = sum_{tok in S} sumsq_tok  -  sum_{tok in S} ||sum_h[tok]||^2 / n_tok
# total_S is the variance about the GLOBAL mean; within_S is about each token's OWN mean, so
# their ratio is exactly the fraction of variance the per-token lookup table cannot explain.
def rho_for(mask):
    n_tok  = count[mask]
    s_tok  = sum_h[mask]
    ss     = sumsq_tok[mask].sum()
    N_S    = n_tok.sum()
    grand  = s_tok.sum(0)
    total  = ss - (grand.double() ** 2).sum() / N_S
    within = ss - ((s_tok.double() ** 2).sum(-1) / n_tok).sum()
    return (within / total).item(), int(N_S.item()), int(mask.sum().item())


results = {}
for mc in MIN_COUNTS:
    mask = count >= mc
    if int(mask.sum()) == 0:
        continue
    rho, n_pos, n_types = rho_for(mask)
    coverage = n_pos / seen
    results[str(mc)] = {"rho": rho, "lookup_r2": 1.0 - rho,
                        "n_positions": n_pos, "n_token_types": n_types, "coverage": coverage}
    print(f"  min_count>={mc:4d}: rho={rho:.4f}  lookup-table R^2={1-rho:.4f}  "
          f"({n_types:,} types, {coverage:.1%} of positions)")

# ---- optional context: the LINEAR map's R^2 on the same stream, if this arm has one fitted ----
# The lookup table upper-bounds it, so linear_r2 <= lookup_r2 is a built-in consistency check.
linear_r2 = None
VOCAB_CHUNK = int(os.environ.get("VOCAB_CHUNK", 8192))
try:
    import torch.nn as nn
    lm = nn.Linear(D_IN, D_IN).to(device)
    lm.load_state_dict(t.load(pull("linear_map_layer_13.pt"), weights_only=False))
    lm.eval()
    with t.no_grad():
        # SSE = sum_i ||h_i - P[tok_i]||^2, expanded so it needs only the accumulators:
        #     = sum ||h||^2 - 2 <sum_h, P> + sum n_tok ||P||^2
        # Done in VOCAB CHUNKS: materializing the full (V, D) float64 P alongside sum_h -- plus the
        # same-size temporary that sum_h * P would create -- peaks near 22 GB and OOMs a 24 GB card
        # at the very end of the run. Chunked, the largest temporary is VOCAB_CHUNK x D (~150 MB).
        alive = count > 0
        sse = t.zeros((), dtype=t.float64, device=device)
        for s in range(0, V, VOCAB_CHUNK):
            e = min(s + VOCAB_CHUNK, V)
            m = alive[s:e]
            if not bool(m.any()):
                continue
            emb = model.embed(t.arange(s, e, device=device)).float()   # same call fit_map.py uses
            Pc  = lm(emb).double()[m]                                  # (chunk_alive, D)
            sh  = sum_h[s:e][m].double()
            nc  = count[s:e][m]
            sse += (sumsq_tok[s:e][m].sum()
                    - 2.0 * (sh * Pc).sum()
                    + (nc * (Pc ** 2).sum(-1)).sum())
            del emb, Pc, sh, nc
        grand = sum_h[alive].sum(0).double()
        total = sumsq_tok[alive].sum() - (grand ** 2).sum() / count[alive].sum()
        linear_r2 = (1.0 - sse / total).item()
    print(f"  linear map R^2 on the same stream: {linear_r2:.4f}  "
          f"(must be <= lookup-table R^2 above)")
except Exception as e:
    print(f"  [skip] no linear map for this arm yet ({type(e).__name__}: {e})")

if device.type == "cuda":
    print(f"  peak GPU memory: {t.cuda.max_memory_allocated()/2**30:.1f} GiB")

out = {"variant": VARIANT, "init_seed": INIT_SEED, "layer": LAYER, "n_tokens": seen,
       "by_min_count": results, "linear_map_r2": linear_r2}
path = f"{OUT_DIR}/context_var_{VARIANT}_s{INIT_SEED}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved {path}")
push(path)
