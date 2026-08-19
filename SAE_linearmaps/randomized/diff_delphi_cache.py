"""Validate write_delphi_cache.py against delphi's OWN cache, before trusting it on skip-embed.

THE ARGUMENT THIS SCRIPT EXISTS TO MAKE. Writing delphi's cache ourselves is only defensible if we
can show our writer reproduces delphi's. We can: for the PLAIN topk SAE, delphi's assumption (the
SAE reads the raw hookpoint) is true, so delphi already produced a cache for exactly that SAE and
archived it as `delphi_latents_L13_30M/`. Run write_delphi_cache.py with MODE=full against the same
SAE, diff the two here, and the claim becomes "we reproduced delphi's caching step and then
extended it to an SAE whose input their loader cannot express" -- rather than "we replaced their
caching step". Only after this passes should the writer be pointed at MODE=resid.

WHAT AGREEMENT TO EXPECT, AND WHY IT IS NOT 100%. We take h_13 from TransformerLens
(`blocks.13.hook_resid_post`) because P was fit against TL's embedding table; delphi takes it from
HF `AutoModel`'s `layers.13`. check_hookpoint.py established these are the same tensor (cosine
0.999992), but two bf16 execution paths disagree at ~1e-2 relative. Top-k is a RANKING, so that
noise flips latents sitting near the k-th boundary: a token whose 32nd and 33rd pre-activations are
nearly tied can select either. Those flips are real and expected.

So read the numbers in this order:
  1. tokens identical              -- MUST be exact. We copied the array; anything else is a bug.
  2. per-token top-k set agreement -- the headline. Expect >99%. This is "do we select the same
                                      latents", which is what the judge ultimately sees.
  3. activation agreement          -- on shared locations, expect ~1e-2 relative, matching the
                                      hookpoint check. float16 storage alone costs ~1e-3.
  4. firing-count correlation      -- a global check that no systematic offset crept in.
A LOW number 2 with a HIGH number 3 means an indexing bug (right values, wrong places). The reverse
means a dtype or scale problem. Both low means the wrong SAE or the wrong arm.

    REF_CACHE=/dev/shm/ref/delphi_latents_L13_30M/layers.13 \
      OURS=/dev/shm/delphi_run/results/trained_full/latents/layers.13 \
      python -u randomized/diff_delphi_cache.py

Env: REF_CACHE, OURS, MAX_LATENTS (defaults to whatever range OURS actually covers).
"""
import glob
import os
import sys

import numpy as np
from safetensors.numpy import load_file

REF  = os.environ.get("REF_CACHE", "/dev/shm/ref/delphi_latents_L13_30M/layers.13")
OURS = os.environ.get("OURS", "/dev/shm/delphi_run/results/trained_full/latents/layers.13")
_ml  = os.environ.get("MAX_LATENTS", "").strip()
MAX_LATENTS = int(_ml) if _ml else None


def read_cache(d):
    """Load every shard of a delphi cache into global (row, pos, latent) + value arrays.

    The per-shard latent offset has to be added back from the FILENAME -- that is how delphi's
    TensorBuffer.load does it, and a shard read without it silently reports latent 0..14744 for
    every split.
    """
    shards = sorted(glob.glob(f"{d}/*.safetensors"))
    if not shards:
        sys.exit(f"no shards in {d!r}")
    locs, acts, tokens = [], [], None
    for path in shards:
        first_latent = int(os.path.basename(path).split("_")[0])
        sd = load_file(path)
        loc = sd["locations"].astype(np.int64)
        loc[:, 2] += first_latent
        locs.append(loc)
        acts.append(sd["activations"].astype(np.float32))
        if tokens is None and "tokens" in sd:
            tokens = sd["tokens"]
    return np.concatenate(locs), np.concatenate(acts), tokens, [os.path.basename(p) for p in shards]


ref_loc, ref_act, ref_tok, ref_files = read_cache(REF)
our_loc, our_act, our_tok, our_files = read_cache(OURS)
print(f"[diff] reference {REF}\n         shards: {ref_files}")
print(f"       ours      {OURS}\n         shards: {our_files}")

# ---- 1. tokens -------------------------------------------------------------------------------
if ref_tok is None or our_tok is None:
    sys.exit("one of the caches has no `tokens` array -- delphi needs it, and so does the judge")
same_shape = ref_tok.shape == our_tok.shape
identical = same_shape and bool((ref_tok == our_tok).all())
print(f"\n1. tokens: ref {ref_tok.shape} vs ours {our_tok.shape} -> "
      f"{'IDENTICAL' if identical else 'DIFFERENT -- stop here, the judge would read different text'}")
if not identical:
    sys.exit(1)

N_ROWS, CTX_LEN = ref_tok.shape

# Restrict to the region ours actually covers, on BOTH axes. write_delphi_cache.py is normally
# run with MAX_LATENTS=500 (delphi only reads latents 0..499) and, for a quick validation, with
# MAX_ROWS set. Comparing outside that region would count rows and latents we deliberately never
# encoded as disagreements and drown the signal.
hi = MAX_LATENTS if MAX_LATENTS is not None else int(our_loc[:, 2].max()) + 1
row_hi = int(our_loc[:, 0].max()) + 1
if row_hi < N_ROWS:
    print(f"   ours covers only rows [0, {row_hi:,}) of {N_ROWS:,} -- a MAX_ROWS validation cache. "
          f"Restricting the comparison to those rows.")
    print(f"   NOTE: this validates the WRITER, not a scoreable cache. Re-run without MAX_ROWS "
          f"before scoring anything.")
ref_m = (ref_loc[:, 2] < hi) & (ref_loc[:, 0] < row_hi)
ref_loc, ref_act = ref_loc[ref_m], ref_act[ref_m]
our_m = (our_loc[:, 2] < hi) & (our_loc[:, 0] < row_hi)
our_loc, our_act = our_loc[our_m], our_act[our_m]
print(f"   comparing latents [0, {hi}) x rows [0, {row_hi:,}):  "
      f"ref {len(ref_act):,} firings | ours {len(our_act):,}")

# ---- 2. location agreement -------------------------------------------------------------------
# One int64 key per (row, position, latent). Max key ~2.2e12, comfortably inside int64.
D_SPAN = int(max(ref_loc[:, 2].max(), our_loc[:, 2].max())) + 1


def keys(loc):
    return (loc[:, 0].astype(np.int64) * CTX_LEN + loc[:, 1]) * D_SPAN + loc[:, 2]


ref_k, our_k = keys(ref_loc), keys(our_loc)
common, ref_i, our_i = np.intersect1d(ref_k, our_k, return_indices=True)
union = len(ref_k) + len(our_k) - len(common)
print(f"\n2. locations: {len(common):,} shared | {len(ref_k) - len(common):,} ref-only | "
      f"{len(our_k) - len(common):,} ours-only")
print(f"   Jaccard: {len(common) / union:.6f}")

# The number that matters is per-TOKEN: did we select the same top-k set at each position?
tok_key_ref = ref_k // D_SPAN
tok_key_our = our_k // D_SPAN
disagreeing = np.union1d(tok_key_ref[np.isin(ref_k, common, invert=True)],
                         tok_key_our[np.isin(our_k, common, invert=True)])
n_tokens_seen = len(np.union1d(tok_key_ref, tok_key_our))
agree_frac = 1 - len(disagreeing) / max(n_tokens_seen, 1)
print(f"   tokens with an IDENTICAL selected-latent set: {agree_frac:.6f} "
      f"({n_tokens_seen - len(disagreeing):,}/{n_tokens_seen:,})")
print(f"   -> expect >0.99; TL-vs-HF bf16 noise flips latents tied near the k-th boundary")

# ---- 3. activation agreement on shared locations ----------------------------------------------
a, b = ref_act[ref_i], our_act[our_i]
denom = np.maximum(np.abs(a), 1e-6)
rel = np.abs(a - b) / denom
print(f"\n3. activations on shared locations: max |rel| {rel.max():.2e} | "
      f"median {np.median(rel):.2e} | p99 {np.percentile(rel, 99):.2e}")
print(f"   corr {np.corrcoef(a, b)[0, 1]:.8f}  (float16 storage alone costs ~1e-3)")

# ---- 4. per-latent firing counts ---------------------------------------------------------------
ref_counts = np.bincount(ref_loc[:, 2], minlength=hi)
our_counts = np.bincount(our_loc[:, 2], minlength=hi)
both_alive = (ref_counts > 0) | (our_counts > 0)
corr = np.corrcoef(ref_counts[both_alive], our_counts[both_alive])[0, 1]
worst = np.argsort(-np.abs(ref_counts.astype(np.int64) - our_counts))[:5]
print(f"\n4. per-latent firing counts over {int(both_alive.sum()):,} live latents: corr {corr:.6f}")
print(f"   largest disagreements (latent: ref vs ours): "
      + ", ".join(f"{i}: {ref_counts[i]:,} vs {our_counts[i]:,}" for i in worst))

# min_examples=200 is delphi's survival filter, so a latent near that line is where a small
# location disagreement could actually change WHICH latents get explained.
ref_surv = int((ref_counts >= 200).sum())
our_surv = int((our_counts >= 200).sum())
print(f"   latents clearing min_examples=200: ref {ref_surv:,} | ours {our_surv:,}")
print(f"   (delphi filters on constructed EXAMPLES, not raw firings, so this is indicative "
      f"rather than the exact survivor count)")

ok = identical and agree_frac > 0.99 and corr > 0.99
print(f"\nVERDICT: {'writer reproduces delphi -- safe to run MODE=resid' if ok else 'DO NOT PROCEED -- see which check failed above'}")
sys.exit(0 if ok else 1)
