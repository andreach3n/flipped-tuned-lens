"""Write a delphi latent cache OURSELVES, so a skip-embed SAE can be scored by delphi's pipeline.

WHY THIS EXISTS. delphi assumes the SAE reads the hookpoint activation directly: it hooks
`layers.13`, hands the tensor to a sparsify SparseCoder, and caches whatever fires. Our `resid`
(skip-embed) SAE does not read that tensor -- it reads  h_13 - P[tok],  the residual after the
frozen linear map's per-token prediction is removed. That subtraction CANNOT be folded into the
sparsify format the way convert_sae_to_sparsify.py folds `scale` and `b_dec`: those are constants,
while P[tok] is a (V, 2304) lookup, and SparseCoder has exactly one bias vector. So a converted
skip-embed SAE would be fed the wrong input and produce plausible, wrong latents -- silently.

The fix is to replace exactly ONE of delphi's three stages. delphi does cache -> explain -> score,
and only `cache` touches the SAE; `explain` and `score` read the on-disk cache and talk to the
judge. So we compute the cache here (where subtracting P[tok] is trivial), write it in delphi's
format, and let delphi do everything that affects the score -- example construction, quantile
sampling, prompts, the Llama-70B judge -- completely unmodified.

    Stage 1 (cache)     <- THIS SCRIPT
    Stage 2 (explain)   <- delphi, untouched
    Stage 3 (score)     <- delphi, untouched

DO NOT REGENERATE THE TOKENS. The single biggest correctness trap here is the token matrix.
delphi's `filter_bos` (default True) does not simply drop position 0: it tokenizes the corpus to
(n, 256), FLATTENS it, deletes every BOS occurrence wherever it appears, truncates to a multiple
of 256 and reshapes. Rows therefore do not align with documents, and row boundaries are shifted
relative to the raw tokenization. On top of that the corpus is shuffled with seed 22 and truncated
to whole batches (n_rows = (n_tokens // 256) // 32 * 32 = 117,184 for 30M tokens).

Rather than reproduce all of that, we READ the `tokens` array out of an existing delphi cache and
run the model over exactly those rows, in order. Every one of those settings is then inherited by
construction rather than by matching parameters. It also means all arms are scored on identical
text. delphi never re-tokenizes at read time (LatentDataset.load_tokens is dead code behind a
`hasattr` guard that is always true), so the array in our shards IS the text the judge sees.

    python -c "from huggingface_hub import snapshot_download as d; \
      d('andreayhchen/gemma2-2b-linearmap-saes-trained-20m', allow_patterns='delphi_latents_L13_30M/*', \
        local_dir='/dev/shm/ref')"

USAGE (per arm; ~2.5 h of GPU for the model pass, same as delphi's own caching):

    # skip-embed, trained arm
    VARIANT=trained HF_REPO=<trained-repo> MODE=resid \
      SAE_NAME=sae_resid_k32_d73728_100M_topk_final.pt \
      REF_CACHE=/dev/shm/ref/delphi_latents_L13_30M/layers.13 \
      OUT_DIR=/dev/shm/delphi_run/results/trained_resid \
      python -u randomized/write_delphi_cache.py

    # skip-embed, random arm
    VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-repo> MODE=resid ... (same shape)

    # VALIDATION RUN -- do this FIRST (see diff_delphi_cache.py). Keep MAX_LATENTS=500: that is
    # exactly the range delphi reads, so the diff is both cheap and the one that matters.
    VARIANT=trained HF_REPO=<trained-repo> MODE=full \
      SAE_NAME=<the topk full SAE that produced the reference cache> MAX_LATENTS=500 ...

CONFIRM WHICH SAE MADE THE REFERENCE CACHE before validating. The archived cache was built from
the topk RETRAIN (the delphi arms were rerun with SAE_ARCH=topk, which tags `_topk` into the
artifact name), not from the batchtopk 100M fleet whose name convert_sae_to_sparsify.py defaults
to. Diffing against the wrong checkpoint fails loudly but for an uninteresting reason, and it is
easy to misread as a bug in the writer. Check the arm repo's file list first.

Then point delphi at OUT_DIR's parent with --name <that dir's name>; it will find the cache and
log `Files found in ..., skipping...` instead of recaching.

MAX_LATENTS IS A REAL SAVING, AND IT IS NOT WHAT THE PROJECT NOTES SAY. delphi's --max_latents N
is `torch.arange(N)` (__main__.py) -- the FIRST N latent indices, NOT the top-firing N. Our notes
record it as a top-firing default; that is wrong. Consequence: with --max_latents 500 only latents
0..499 are ever read, so we write only those and the cache is ~1% the size (~150 MB vs ~14 GB).
Set MAX_LATENTS= (blank) to write the full dictionary. If you set it, delphi MUST be run with
--max_latents <= that value, or it will silently find no data for the latents above it.

WHAT WILL NOT MATCH BIT-FOR-BIT. We take h_13 from TransformerLens (`blocks.13.hook_resid_post`),
because P was fit against TL's embedding table in fit_map.py and must stay consistent with it.
delphi takes it from HF `AutoModel`'s `layers.13`. check_hookpoint.py proves these are the same
tensor (cosine 0.999992) but two bf16 execution paths differ at ~1e-2 relative, which can flip
latents sitting near the top-k boundary. Expect the validation diff to agree on the vast majority
of locations, not all of them -- see diff_delphi_cache.py, which quantifies exactly that.

Env: MODE, SAE_NAME, REF_CACHE, OUT_DIR, MAX_LATENTS, ROWS, N_SPLITS, HOOKPOINT
     (+ VARIANT / INIT_SEED / HF_REPO, which select the arm, exactly as in eval_fvu.py).
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch as t
import torch.nn as nn
from safetensors.numpy import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import D_IN, HOOK, INIT_SEED, LAYER, MAP_FILE, VARIANT, load_model
from hf_io import pull
from sae_arch import arch_of
from sae_arch import load_sae as _rebuild_sae

MODE      = os.environ.get("MODE", "resid")            # "resid" (skip-embed) | "full" (plain, for validation)
SAE_NAME  = os.environ.get("SAE_NAME", f"sae_{MODE}_k32_d73728_100M_topk_final.pt")
REF_CACHE = os.environ.get("REF_CACHE", "/dev/shm/ref/delphi_latents_L13_30M/layers.13")
OUT_DIR   = os.environ.get("OUT_DIR", "/dev/shm/delphi_run/results/skipembed")
HOOKPOINT = os.environ.get("HOOKPOINT", f"layers.{LAYER}")   # delphi resolves WITHOUT the wrapper prefix
N_SPLITS  = int(os.environ.get("N_SPLITS", 5))
ROWS      = int(os.environ.get("ROWS", 32))            # rows per forward pass; delphi's cache batch_size
_ml       = os.environ.get("MAX_LATENTS", "500").strip()
MAX_LATENTS = int(_ml) if _ml else None
# MAX_ROWS exists for the VALIDATION run only: diffing the writer against delphi's cache does not
# need all 117,184 rows, and 2,000 rows turns a 2.5 h pass into a few minutes. The saved `tokens`
# array stays the FULL matrix either way (delphi indexes into it), so a truncated cache is still
# structurally valid -- but it is NOT a real run. Leave blank for anything you intend to score.
_mr       = os.environ.get("MAX_ROWS", "").strip()
MAX_ROWS  = int(_mr) if _mr else None
# VALIDATION-ONLY knobs. EMULATE_SPARSIFY reproduces delphi's exact computation -- including the
# two conversion bugs found 2026-08-19 -- so diff_delphi_cache.py can check this writer's SHARDING,
# INDEXING and DTYPES against a real delphi cache. It does NOT produce a scientifically usable
# cache: the point of the science runs is to use the trained SAE, which delphi's converted one is
# not. BACKEND=hf uses AutoModel the way delphi does; TransformerLens differs by bf16 path noise
# (cosine 0.9997) which flips ~8% of firings at the k-boundary, so exact reproduction needs hf.
EMULATE = os.environ.get("EMULATE_SPARSIFY", "").strip()
BACKEND = os.environ.get("BACKEND", "tl").strip().lower()
# PREPEND_BOS -- NOT cosmetic, and the reason a first attempt produced a NEGATIVE linear-map
# explained variance (2026-08-19). activations.py builds every training activation with
# `model.to_tokens(text)`, which PREPENDS BOS, then drops position 0. So the SAEs and the linear
# map only ever saw tokens in a BOS-prefixed context. delphi's rows have BOS stripped entirely
# (filter_bos deletes it from the FLATTENED stream), and gemma-2 leans hard on the BOS attention
# sink: |h| falls from the ~174 implied by the SAE's own scale to ~141. P then overshoots, and
# h - P[tok] has MORE variance than h. We therefore run [BOS] + row and drop the BOS position,
# which restores the training regime while leaving the stored tokens and position indices
# untouched -- delphi cannot tell the difference.
# This is a DELIBERATE divergence from delphi's own caching, and it applies to the plain SAE too:
# the pre-2026-08-19 delphi runs fed out-of-distribution activations for the same reason.
PREPEND_BOS = os.environ.get("PREPEND_BOS", "1").strip() not in ("", "0", "false", "no")
# Emulation defaults to BOS-off so that EMULATE_SPARSIFY alone reproduces delphi exactly. But the
# two defects are INDEPENDENT and the ablation that attributes the false null needs all four
# combinations, so an EXPLICIT PREPEND_BOS always wins.
if EMULATE and PREPEND_BOS and "PREPEND_BOS" not in os.environ:
    PREPEND_BOS = False
    print("  EMULATE_SPARSIFY set -> PREPEND_BOS defaulted off (exact delphi reproduction); "
          "set PREPEND_BOS=1 explicitly to isolate the conversion defect on its own")
if BACKEND not in ("tl", "hf"):
    sys.exit(f"BACKEND={BACKEND!r} -- use 'tl' (default, matches how the SAEs were trained) or 'hf'")
if BACKEND == "hf" and MODE == "resid":
    sys.exit("BACKEND=hf is validation-only: MODE=resid needs P, which fit_map.py built from "
             "TransformerLens' embedding table. Keep all science runs on BACKEND=tl so the four "
             "cells share one activation pipeline.")

if MODE not in ("resid", "full"):
    sys.exit(f"MODE={MODE!r} -- this script handles 'resid' (the point of it) and 'full' (validation)")

# delphi resolves `results/<name>` relative to CWD, so OUT_DIR has to BE that directory. A path
# whose parent is not `results` will produce a perfectly good cache that delphi then ignores --
# and it recaches from scratch with the wrong SAE instead of erroring.
if os.path.basename(os.path.dirname(os.path.abspath(OUT_DIR.rstrip("/")))) != "results":
    print(f"  WARNING: OUT_DIR={OUT_DIR!r} -- delphi expects <somewhere>/results/<name>. As given, "
          f"it will not find this cache and will silently recache.")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"[cache] VARIANT={VARIANT} INIT_SEED={INIT_SEED} MODE={MODE} layer={LAYER} "
      f"hookpoint={HOOKPOINT} max_latents={MAX_LATENTS} rows/batch={ROWS}")

# ---- the reference cache: tokens + config come from delphi's own output ----------------------
ref_shards = sorted(glob.glob(f"{REF_CACHE}/*.safetensors"))
if not ref_shards:
    sys.exit(f"no .safetensors in REF_CACHE={REF_CACHE!r} -- snapshot_download the arm's "
             f"delphi_latents_L13_30M/ first (see the module docstring)")
if not os.path.exists(f"{REF_CACHE}/config.json"):
    sys.exit(f"{REF_CACHE}/config.json missing -- delphi refuses a cache without it")

tokens_np = load_file(ref_shards[0])["tokens"]          # (n_rows, ctx_len); identical in all 5 shards
N_ROWS, CTX_LEN = tokens_np.shape
tokens_t = t.from_numpy(tokens_np.astype(np.int64))
N_PASS = min(N_ROWS, MAX_ROWS) if MAX_ROWS else N_ROWS
print(f"  reference tokens: {N_ROWS:,} rows x {CTX_LEN} = {N_ROWS * CTX_LEN:,} tokens "
      f"(from {os.path.basename(ref_shards[0])})")
if N_PASS != N_ROWS:
    print(f"  MAX_ROWS={MAX_ROWS}: encoding only the first {N_PASS:,} rows. VALIDATION ONLY -- "
          f"do not score this cache.")

BOS_ID = None
if PREPEND_BOS:
    from transformers import AutoTokenizer
    with open(f"{REF_CACHE}/config.json") as _f:
        _refm = json.load(_f).get("model_name", "google/gemma-2-2b")
    BOS_ID = AutoTokenizer.from_pretrained(_refm).bos_token_id
    if BOS_ID is None:
        sys.exit(f"{_refm} has no bos_token_id but PREPEND_BOS is on")
    n_bos = int((tokens_t[:N_PASS] == BOS_ID).sum())
    print(f"  PREPEND_BOS: prefixing every row with token {BOS_ID} for the forward pass, then "
          f"dropping it ({n_bos} BOS already present in the rows -- expect 0)")

# ---- the model and the SAE, built exactly as eval_fvu.py builds them --------------------------
model = load_model(device)
V = model.cfg.d_vocab

ckpt = t.load(pull(SAE_NAME), weights_only=False)
sae = _rebuild_sae(ckpt)
sae.to(device).eval()
scale = float(ckpt["scale"])
D_SAE = int(sae.cfg.d_sae)
K = int(sae.cfg.k)
print(f"  {SAE_NAME}: arch={arch_of(ckpt)} d_sae={D_SAE} k={K} scale={scale:.6f} "
      f"mode={ckpt.get('mode')} tokens={ckpt.get('tokens', 'final')}")
if ckpt.get("mode") != MODE:
    sys.exit(f"checkpoint says mode={ckpt.get('mode')!r} but MODE={MODE!r} -- the encoder input "
             f"would be wrong in exactly the silent way this script exists to prevent")
if arch_of(ckpt) != "topk":
    print(f"  WARNING: arch is {arch_of(ckpt)}, not topk. BatchTopK's firing set depends on what "
          f"else is in the batch, so this cache depends on ROWS. Retrain with SAE_ARCH=topk for "
          f"anything meant to sit next to the existing delphi arms.")

# The frozen greedy map, rebuilt into a (V, 2304) lookup exactly as train_sae_res.py / eval_fvu.py
# do. It MUST come from model.embed -- fit_map.py fit the map against that table, and a differently
# scaled embedding source would produce a wrong P with no error anywhere.
P = None
if MODE == "resid":
    linear_map = nn.Linear(D_IN, D_IN).to(device)
    linear_map.load_state_dict(t.load(pull(MAP_FILE), weights_only=False))
    linear_map.eval()
    with t.no_grad():
        P = linear_map(model.embed(t.arange(V, device=device)).float())      # (V, 2304)
    print(f"  rebuilt P from {MAP_FILE}: {tuple(P.shape)}")

# ---- the activation source --------------------------------------------------------------------
hf_model, _grab = None, {}
if BACKEND == "hf":
    # delphi's load_artifacts: AutoModel (not ForCausalLM), bf16, cuda; its hook takes output[0].
    from transformers import AutoModel
    _src = os.environ.get("HF_MODEL_PATH", "google/gemma-2-2b")
    hf_model = AutoModel.from_pretrained(_src, torch_dtype=t.bfloat16).to(device).eval()
    _target = [n for n, _ in hf_model.named_modules() if n.endswith(f"layers.{LAYER}")][0]
    dict(hf_model.named_modules())[_target].register_forward_hook(
        lambda _m, _i, o: _grab.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    print(f"  BACKEND=hf: {_src} hooked at {_target} (validation only)")

# The sparsify encoder, loaded only to EMULATE delphi. Its encode subtracts b_dec itself, which is
# precisely what convert_sae_to_sparsify.py used to double-count.
sp_w = sp_b = sp_bdec = None
if EMULATE:
    from safetensors.torch import load_file as _load_pt
    _hits = glob.glob(f"{EMULATE}/**/sae.safetensors", recursive=True)
    if not _hits:
        sys.exit(f"EMULATE_SPARSIFY={EMULATE!r}: no sae.safetensors under it")
    _sp = {k: v.float().to(device) for k, v in _load_pt(_hits[0]).items()}
    sp_w, sp_b, sp_bdec = _sp["encoder.weight"], _sp["encoder.bias"], _sp["b_dec"]
    print(f"  EMULATE_SPARSIFY: {_hits[0]}\n"
          f"    reproducing delphi's computation as archived. Validation only -- if the sparsify "
          f"dir predates the 2026-08-19 converter fix, this deliberately reproduces its bugs.")


@t.no_grad()
def encode_rows(tok_rows):
    """(rows, ctx_len) token ids -> (rows*ctx_len, d_sae) dense latents.

    Equivalent to delphi's `sae_dense_latents`: a zero buffer with the top-k values scattered in.
    sae_lens' encode already returns exactly that shape, with zeros off the top-k support.
    """
    tok = tok_rows.to(device)
    # [BOS] + row, then drop the BOS column, so each of the row's 256 tokens is encoded in the
    # BOS-prefixed context the SAEs were trained in. Shapes downstream are unchanged.
    inp = t.cat([t.full((tok.shape[0], 1), BOS_ID, device=device, dtype=tok.dtype), tok], dim=1) \
        if PREPEND_BOS else tok
    if BACKEND == "hf":
        hf_model(inp)
        h3 = _grab["x"]
    else:
        _, cache = model.run_with_cache(inp, names_filter=[HOOK], stop_at_layer=LAYER + 1)
        h3 = cache[HOOK]                                       # (rows, ctx[+1], 2304)
    if PREPEND_BOS:
        h3 = h3[:, 1:]
    h = h3.reshape(-1, D_IN).float()                           # (rows*ctx, 2304)
    flat_tok = tok.reshape(-1)
    r = (h - P[flat_tok]) if MODE == "resid" else h

    if EMULATE:
        pre = (r - sp_bdec) @ sp_w.T + sp_b
        vals, idx = t.topk(pre, K, dim=-1)
        acts = t.zeros_like(pre).scatter_(-1, idx, vals)
        return acts, h, r / scale
    return sae.encode(r / scale), h, r / scale


# ---- the pass ---------------------------------------------------------------------------------
# Accumulate as int32/float16 on CPU. delphi keeps the same thing in RAM (InMemoryCache) and its
# full-dictionary cache is ~14 GB; with MAX_LATENTS=500 this is ~1% of that.
loc_chunks, act_chunks = [], []
firing_counts = t.zeros(D_SAE, dtype=t.int64)      # delphi's log/hookpoint_firing_counts.pt
n_kept = 0
var_h = var_r = 0.0

with t.no_grad():
    for start in range(0, N_PASS, ROWS):
        rows = tokens_t[start:start + ROWS]
        acts, h, x = encode_rows(rows)

        firing_counts += (acts > 0).sum((0,)).cpu().to(t.int64)
        if start == 0 and MODE == "resid":
            # Cheap smoke test that P is the right table: train_sae_res.py reports linear-map
            # explained variance 0.5616 (trained) / 0.3212 (rand_all s0). A wrongly-scaled or
            # wrong-arm embedding source lands nowhere near those.
            var_h, var_r = h.var().item(), (x * scale).var().item()
            ev = 1 - var_r / var_h
            # NB: in resid mode `scale` normalizes r, not h -- so the |h| expectation comes from
            # the FULL SAE's scale (3.6328 * sqrt(2304) ~ 174), not this checkpoint's.
            print(f"  linear-map explained variance on batch 0: {ev:.4f} "
                  f"(expect ~0.56 trained / ~0.32 rand_all)")
            print(f"  mean |h| {h.norm(dim=-1).mean():.2f} (expect ~174 in the training regime; "
                  f"~141 means BOS is missing) | mean |r| {(x * scale).norm(dim=-1).mean():.2f} "
                  f"(expect ~{scale * D_IN ** 0.5:.0f} from this SAE's scale)")
            # HARD GATE. A wrong activation regime (e.g. PREPEND_BOS off) makes P overshoot and
            # produces a residual the SAE never saw -- 2.5 h of plausible, useless cache. Die now.
            if ev < 0.1 and os.environ.get("ALLOW_BAD_MAP", "") != "1":
                sys.exit(f"\nABORT: explained variance {ev:.4f} means P does not fit these "
                         f"activations, so h - P[tok] is out of distribution for this SAE.\n"
                         f"  Most likely PREPEND_BOS is off: the map and the SAE were fit on "
                         f"BOS-prefixed contexts (activations.py uses to_tokens, which prepends "
                         f"BOS, then drops position 0), while delphi's rows carry no BOS.\n"
                         f"  Set ALLOW_BAD_MAP=1 only if you intend this.")

        # nonzero on the FLAT (rows*ctx, d_sae) tensor returns rows ascending, then latent
        # ascending -- the same lexicographic order delphi's 3D nonzero produces, so a byte-level
        # diff against its shards is meaningful.
        nz = t.nonzero(acts.abs() > 1e-5)                       # (n, 2): [flat_position, latent]
        if MAX_LATENTS is not None:
            nz = nz[nz[:, 1] < MAX_LATENTS]
        vals = acts[nz[:, 0], nz[:, 1]]

        flat = nz[:, 0]
        loc = t.stack([
            (start + flat // CTX_LEN).to(t.int32),              # row index, ABSOLUTE across the run
            (flat % CTX_LEN).to(t.int32),                       # position within the row
            nz[:, 1].to(t.int32),                               # latent index
        ], dim=1)
        loc_chunks.append(loc.cpu())
        act_chunks.append(vals.half().cpu())
        n_kept += loc.shape[0]

        if (start // ROWS) % 200 == 0:
            done = min(start + ROWS, N_PASS)
            print(f"  {done:,}/{N_PASS:,} rows ({done * CTX_LEN:,} tokens), {n_kept:,} firings kept")

locations = t.cat(loc_chunks).numpy()
activations = t.cat(act_chunks).numpy()
del loc_chunks, act_chunks
print(f"  total firings kept: {locations.shape[0]:,}  "
      f"(mean {locations.shape[0] / (N_PASS * CTX_LEN):.2f} per token over the encoded rows; k={K}, "
      f"of which {MAX_LATENTS or D_SAE}/{D_SAE} latents are kept)")

# ---- write the shards, mirroring LatentCache.save_splits --------------------------------------
# Boundaries are torch.linspace(0, width, n_splits+1).long() with INCLUSIVE ranges (start, end-1),
# and column 2 has `start` subtracted so a smaller dtype fits; the loader adds it back from the
# filename. Every shard carries the FULL token matrix -- duplicated, not shared. That looks
# wasteful and is what delphi does.
latents_dir = f"{OUT_DIR}/latents/{HOOKPOINT}"
os.makedirs(latents_dir, exist_ok=True)
boundaries = t.linspace(0, D_SAE, steps=N_SPLITS + 1).long()
written, skipped = [], []

for lo, hi in zip(boundaries[:-1].tolist(), (boundaries[1:] - 1).tolist()):
    mask = (locations[:, 2] >= lo) & (locations[:, 2] <= hi)
    if not mask.any():
        # With MAX_LATENTS set, every shard above the first is empty. delphi's LatentDataset
        # discovers shards by globbing, and bucketizes selected latents against whatever it finds,
        # so omitting empty shards is safe -- and safer than writing zero-row safetensors.
        skipped.append(f"{lo}_{hi}")
        continue

    shard_loc = locations[mask].copy()
    shard_loc[:, 2] -= lo
    if shard_loc[:, 2].max() < 2**16 and shard_loc[:, 0].max() < 2**16:
        shard_loc = shard_loc.astype(np.uint16)
    else:
        shard_loc = shard_loc.astype(np.uint32)

    path = f"{latents_dir}/{lo}_{hi}.safetensors"
    save_file({"locations": shard_loc,
               "activations": activations[mask],
               "tokens": tokens_np}, path)
    written.append(f"{lo}_{hi} ({mask.sum():,} rows, {shard_loc.dtype})")

shutil.copyfile(f"{REF_CACHE}/config.json", f"{latents_dir}/config.json")
print(f"\n  wrote {latents_dir}")
for w in written:
    print(f"    {w}")
if skipped:
    print(f"    (empty, not written: {', '.join(skipped)})")
print(f"    config.json copied from the reference cache (field names drifted between delphi "
      f"versions -- do not hand-write it)")

# delphi's own summary step reads this and dies with KeyError: 'firing_count' when it is missing.
# That crash is harmless -- it happens after all scores are on disk -- but writing the file costs
# nothing and lets log_results actually run.
log_dir = f"{OUT_DIR}/log"
os.makedirs(log_dir, exist_ok=True)
t.save({HOOKPOINT: firing_counts}, f"{log_dir}/hookpoint_firing_counts.pt")
alive = int((firing_counts > 0).sum())
print(f"  wrote {log_dir}/hookpoint_firing_counts.pt  ({alive:,}/{D_SAE:,} latents alive)")

meta = {"variant": VARIANT, "init_seed": INIT_SEED, "mode": MODE, "sae": SAE_NAME,
        "arch": arch_of(ckpt), "d_sae": D_SAE, "k": K, "scale": scale, "layer": LAYER,
        "hookpoint": HOOKPOINT, "n_rows": N_ROWS, "n_rows_encoded": N_PASS, "ctx_len": CTX_LEN,
        "max_latents": MAX_LATENTS, "n_firings": int(locations.shape[0]),
        "ref_cache": REF_CACHE}
with open(f"{OUT_DIR}/write_delphi_cache.json", "w") as f:
    json.dump(meta, f, indent=2)

# delphi writes and reads `results/<name>` RELATIVE TO CWD, so the directory to cd into is the
# parent of `results/`, not OUT_DIR itself.
_out_abs = os.path.abspath(OUT_DIR.rstrip("/"))
name = os.path.basename(_out_abs)
run_dir = os.path.dirname(os.path.dirname(_out_abs))
print(f"\nnow run delphi against this cache (it should log 'Files found in ..., skipping...'):")
print(f"  cd {run_dir}")
print(f"  $V -m delphi google/gemma-2-2b <any_sparsify_dir> --hookpoints {HOOKPOINT} "
      f"--scorers fuzz detection --log_probs "
      + (f"--max_latents {MAX_LATENTS} " if MAX_LATENTS else "")
      + f"--n_tokens {N_ROWS * CTX_LEN} --dataset_repo Skylion007/openwebtext "
      f"--dataset_split 'train[:3%]' --name {name}")
print("  (<any_sparsify_dir> only has to RESOLVE the hookpoint -- with the cache present its "
      "weights are never read. Verify that on the pod before trusting it.)")
