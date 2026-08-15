"""On-the-fly layer-13 activations for gemma-2-2b over openwebtext.

Replaces the old /workspace/sae_cache_layer13 files: nothing is cached to disk, so the
training pod needs NO network volume -- just the model (pulled from HF) and the streamed
corpus (also from HF). Each item is a (h, tok) pair for one token position:
    h   = blocks.13.hook_resid_post   (the residual-stream activation, d=2304)
    tok = the token id at that position
The BOS token (position 0) is dropped everywhere, matching the old extract.py.

TRAINED-vs-RANDOM ARMS (the skip-embed diagnostic thread)
--------------------------------------------------------
Set VARIANT to swap in a randomly-initialized gemma of the SAME architecture:

    VARIANT=trained        the real pretrained weights (DEFAULT -- unchanged behaviour)
    VARIANT=rand_all       every weight re-sampled, embeddings included
    VARIANT=rand_nonembed  every weight re-sampled EXCEPT embed/unembed -- isolates
                           "learned computation" from "learned representation"

Randomization follows Heap et al. (arXiv:2501.17727): each tensor is re-sampled from a
Gaussian matched to THAT tensor's own trained mean and std. That is why the pretrained
weights are still downloaded on a random arm -- we need their moments. Default init would
change the activation scale across 13 layers and confound the whole comparison.

INIT_SEED picks the random draw (report >=3 seeds; one draw is one sample).
"""
import os
import torch as t
from transformer_lens import HookedTransformer
from datasets import load_dataset

MODEL_NAME = "google/gemma-2-2b"
# LAYER is env-driven for the depth sweep (gemma-2-2b has 26 blocks, 0..25). Everything derived
# from it lives HERE so no script can disagree about which layer it is on -- including MAP_FILE,
# because the linear map is layer-specific and a stale one is silently wrong, not an error.
# The default 13 reproduces every historical artifact name byte-for-byte.
LAYER    = int(os.environ.get("LAYER", 13))
HOOK     = f"blocks.{LAYER}.hook_resid_post"
MAP_FILE = f"linear_map_layer_{LAYER}.pt"
SEQ_LEN  = 512
D_IN     = 2304

VARIANT   = os.environ.get("VARIANT", "trained").strip() or "trained"   # trained | rand_all | rand_nonembed
INIT_SEED = int(os.environ.get("INIT_SEED", 0))
_VARIANTS = ("trained", "rand_all", "rand_nonembed")
if VARIANT not in _VARIANTS:
    raise ValueError(f"VARIANT={VARIANT!r} not in {_VARIANTS}")

# gemma-2 ties lm_head to embed_tokens, so named_parameters() (which de-duplicates shared
# tensors) yields the embedding exactly once -- skipping this one name covers both roles.
_EMBED_NAMES = ("embed_tokens", "lm_head")


@t.no_grad()
def _randomize_(hf_model, keep_embeddings, seed):
    """In-place moment-matched re-initialization of every parameter (see module docstring).

    For each tensor: draw randn * std(w) + mean(w), using the TRAINED tensor's own moments.
    Stats are computed in float32 -- mean/std over a bf16 tensor is badly imprecise.
    RMSNorm gains are re-sampled too, matching the paper's "all model parameters"; moment
    matching keeps them near their trained scale, but they are the parameter class most
    likely to surprise us, so the norm check in the caller is worth watching.
    """
    g = t.Generator().manual_seed(seed)
    kept, redrawn = [], 0
    for name, p in hf_model.named_parameters():
        if keep_embeddings and any(s in name for s in _EMBED_NAMES):
            kept.append(name)
            continue
        w = p.detach().float()
        mu, sd = w.mean().item(), w.std().item()
        new = t.randn(p.shape, generator=g, dtype=t.float32) * sd + mu
        p.copy_(new.to(p.dtype))
        redrawn += 1
    print(f"  re-initialized {redrawn} parameter tensors (seed={seed})"
          + (f"; kept {kept}" if kept else "; embeddings included"))


def load_model(device):
    if VARIANT == "trained":
        model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
        model.eval()
        return model

    # --- random arm: build the HF model, re-sample its weights, hand it to TransformerLens ---
    # from_pretrained_no_processing forwards **from_pretrained_kwargs to from_pretrained,
    # which accepts `hf_model` -- so TL imports OUR weights instead of re-downloading them.
    from transformers import AutoModelForCausalLM
    print(f"[activations] VARIANT={VARIANT} INIT_SEED={INIT_SEED}: randomizing gemma weights")
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.bfloat16)
    _randomize_(hf, keep_embeddings=(VARIANT == "rand_nonembed"), seed=INIT_SEED)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, hf_model=hf, dtype=t.bfloat16,
    ).to(device)
    model.eval()
    del hf
    t.cuda.empty_cache()

    # Sanity: a random 13-layer stack can explode or collapse. Print the scale of h_13 so a
    # pathological arm is caught BEFORE hours of SAE training, not after.
    _h, _ = take_sample(model, device, n_tokens=20_000, seed=INIT_SEED)
    _h = _h.float()
    print(f"  h_{LAYER} check: |h| mean {_h.norm(dim=-1).mean():.2f} | "
          f"var {_h.var().item():.4f} | max|dim var| {_h.var(0).max().item():.4f}")
    del _h
    return model


def _docs(seed):
    """Infinite stream of openwebtext documents: shuffled, and looped when exhausted."""
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)   # shuffle document order too, not just tokens
    while True:
        for data in ds:
            yield data["text"]


@t.no_grad()
def _forward(model, device, text):
    """Run one document through gemma; return (h, tok) on CPU with BOS dropped, or None if too short."""
    tokens = model.to_tokens(text)[:, :SEQ_LEN].to(device)
    if tokens.shape[1] < 10:
        return None
    _, cache = model.run_with_cache(tokens, names_filter=[HOOK], stop_at_layer=LAYER + 1)
    h   = cache[HOOK].squeeze(0)[1:].cpu()   # (T-1, 2304) bf16
    tok = tokens.squeeze(0)[1:].cpu()        # (T-1,)
    return h, tok


@t.no_grad()
def activation_stream(model, device, batch=4096, buffer_tokens=1_000_000, seed=0, max_tokens=None):
    """Yield (h_batch, tok_batch) on `device`, h upcast to float32.

    Mirrors the old cache path's per-chunk shuffle: fill a ~buffer_tokens buffer by running
    gemma over documents, shuffle it (so each batch mixes many contexts -- SAE training needs
    decorrelated activations, not one document at a time), emit BATCH-sized slices, then refill.
    Stops after `max_tokens` have been emitted (None = run forever).
    """
    docs = _docs(seed)
    served, chunk_id = 0, 0
    while True:
        # ---- fill the buffer by generating activations ----
        hs, toks, filled = [], [], 0
        while filled < buffer_tokens:
            out = _forward(model, device, next(docs))
            if out is None:
                continue
            h, tok = out
            hs.append(h); toks.append(tok); filled += h.shape[0]
        H, TOK = t.cat(hs), t.cat(toks)

        # ---- shuffle the buffer (deterministic per buffer, for reproducible runs) ----
        g = t.Generator().manual_seed(seed * 100003 + chunk_id)
        perm = t.randperm(H.shape[0], generator=g)
        H, TOK = H[perm], TOK[perm]
        chunk_id += 1

        # ---- emit batches ----
        for start in range(0, H.shape[0], batch):
            h_b   = H[start:start + batch].float().to(device)   # bf16 buffer -> float32 batch (as the old code did)
            tok_b = TOK[start:start + batch].to(device)
            yield h_b, tok_b
            served += h_b.shape[0]
            if max_tokens is not None and served >= max_tokens:
                return


@t.no_grad()
def take_sample(model, device, n_tokens, seed=0):
    """Collect ~n_tokens (h, tok) pairs into CPU tensors -- used for the one-off scale/sanity check."""
    docs = _docs(seed + 777)      # offset the seed so this sample != the first training buffer
    hs, toks, filled = [], [], 0
    while filled < n_tokens:
        out = _forward(model, device, next(docs))
        if out is None:
            continue
        h, tok = out
        hs.append(h); toks.append(tok); filled += h.shape[0]
    return t.cat(hs)[:n_tokens], t.cat(toks)[:n_tokens]
