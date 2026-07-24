"""On-the-fly layer-13 activations for gemma-2-2b over openwebtext.

Replaces the old /workspace/sae_cache_layer13 files: nothing is cached to disk, so the
training pod needs NO network volume -- just the model (pulled from HF) and the streamed
corpus (also from HF). Each item is a (h, tok) pair for one token position:
    h   = blocks.13.hook_resid_post   (the residual-stream activation, d=2304)
    tok = the token id at that position
The BOS token (position 0) is dropped everywhere, matching the old extract.py.
"""
import torch as t
from transformer_lens import HookedTransformer
from datasets import load_dataset

MODEL_NAME = "google/gemma-2-2b"
LAYER   = 13
HOOK    = f"blocks.{LAYER}.hook_resid_post"
SEQ_LEN = 512
D_IN    = 2304


def load_model(device):
    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
    model.eval()
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
