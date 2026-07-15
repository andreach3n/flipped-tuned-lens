from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput
import torch as t
import torch.nn as nn
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset
import glob
import os

MODEL_NAME = "google/gemma-2-2b"
LAYER=13
D_IN=2304
D_SAE=16384
K    = int(os.environ.get("K", 64))        # sweep: K=32 MODE=full python train_sae_res.py
LR=4e-4
CACHE_DIR="/workspace/sae_cache_layer13"
BATCH = 4096
MODE = os.environ.get("MODE", "hybrid")    # "full" | "resid" | "hybrid"
SEED = int(os.environ.get("SEED", 0))      # reproducibility across the sweep fleet
TRAIN_TOKENS = int(os.environ.get("TRAIN_TOKENS", 20_000_000))  # cap training length (full cache ~50M; FVU converges by ~20M)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

cfg = BatchTopKTrainingSAEConfig(
    d_in=D_IN, d_sae=D_SAE, k=K,
    dtype="float32", device=str(device),
    apply_b_dec_to_input=True,
    normalize_activations="none",   # you normalize in iter_batches
)
t.manual_seed(SEED)                        # deterministic SAE init (reproducible fleet)
sae = BatchTopKTrainingSAE(cfg).to(device)

linear_map = nn.Linear(2304, 2304).to(device)
linear_map.load_state_dict(t.load("/workspace/linear_map_layer_13.pt", weights_only=False))
linear_map.eval()

with t.no_grad():
    embed_table = model.embed(t.arange(model.cfg.d_vocab, device=device))  # (V, 2304): the input to the linear map
    P = linear_map(embed_table.float())                                    # (V, 2304): frozen greedy prediction table
    if not os.path.exists(f"{CACHE_DIR}/P.pt"):   # don't clobber an existing P.pt (deterministic anyway)
        t.save(P, f"{CACHE_DIR}/P.pt")

# hybrid needs the embedding table kept around to recompute P[token] WITH gradient each step
embed_W = embed_table.detach() if MODE in ("hybrid", "outbias") else None
del embed_table

del model                # free ~5 GB — not needed after P is built
t.cuda.empty_cache()

SANITY_N = 100_000   # subsample for the check/scale — a full 1M-token chunk on GPU OOMs
chunk_h   = t.cat(t.load(f"{CACHE_DIR}/layer_13_chunk_1.pt"), dim=0)[:SANITY_N].float().to(device)
chunk_tok = t.cat(t.load(f"{CACHE_DIR}/tokens_chunk_1.pt"),   dim=0)[:SANITY_N].to(device)
r = chunk_h - P[chunk_tok]
explained = 1 - (r**2).mean() / chunk_h.var()
print(explained.item())   # want ≈ 0.66

# from chunk_1 (already loaded above as chunk_h / chunk_tok):
sample = (chunk_h - P[chunk_tok]) if MODE in ("resid", "hybrid") else chunk_h
scale  = sample.norm(dim=-1).mean() / (D_IN ** 0.5)   # a single scalar
del chunk_h, chunk_tok, r, sample    # free the ~9 GB sanity-check tensors before training
t.cuda.empty_cache()

# yield raw (activation, token) pairs; the residual / normalization is built per-step in the loop,
# because in hybrid mode P[token] is recomputed each step through the TRAINABLE linear map.
def iter_batches():
    layer_files = sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt"))
    for ci, lf in enumerate(layer_files):
        tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
        h = t.cat(t.load(lf), dim=0)
        tok = t.cat(t.load(tf), dim=0)

        g = t.Generator().manual_seed(SEED * 1000 + ci)   # deterministic data order, per chunk
        shuffled_idx = t.randperm(h.shape[0], generator=g)
        h = h[shuffled_idx]
        tok = tok[shuffled_idx]

        for start in range(0, h.shape[0], BATCH):
            h_batch = h[start: start+BATCH].float().to(device)
            tok_batch = tok[start: start+BATCH].to(device)
            yield h_batch, tok_batch

# hybrid jointly trains the linear map (warm-started from the fitted map) alongside the SAE
params = list(sae.parameters()) + (list(linear_map.parameters()) if MODE in ("hybrid", "outbias") else [])
opt = t.optim.Adam(params, lr=LR)
n_since_fired = t.zeros(D_SAE, device=device)
DEAD_WINDOW = 200
N = 100
seen_tokens = 0
for step, (h_batch, tok_batch) in enumerate(iter_batches()):
    dead = n_since_fired > DEAD_WINDOW

    if MODE == "hybrid":
        P_batch = linear_map(embed_W[tok_batch].float())   # per-token bias SUBTRACTED from the encoder input
        x = (h_batch - P_batch) / scale
    elif MODE == "outbias":
        P_batch = linear_map(embed_W[tok_batch].float())   # per-token bias added at the OUTPUT only
        x = h_batch / scale                                # encoder sees the FULL activation (token NOT removed)
    elif MODE == "resid":
        x = (h_batch - P[tok_batch]) / scale               # frozen greedy map
    else:
        x = h_batch / scale                                # full activation

    out = sae.training_forward_pass(TrainStepInput(sae_in=x, coefficients={}, dead_neuron_mask=dead, n_training_steps=step, is_logging_step=False))
    if MODE == "outbias":
        # encoder saw the full h; the decoder must output the residual so that P + decode = h.
        # reconstruction loss on h, map added at the OUTPUT and jointly trained. Divide by scale so
        # it matches the SAE's native (scaled-space) loss magnitude -- otherwise the raw-unit loss is
        # ~scale^2 too large and the effective LR blows up (features die, FVU stalls).
        # NOTE: still bypasses the BatchTopK dead-neuron aux term -- watch the dead-feature count.
        loss = (((P_batch + out.sae_out * scale - h_batch) / scale) ** 2).mean()
    else:
        loss = out.loss
    opt.zero_grad()
    loss.backward()
    t.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    fired = (out.feature_acts > 0).any(0)             # which features fired this batch
    n_since_fired += 1
    n_since_fired[fired] = 0 # reset the ones that just fired back to 0

    if step % 100 == 0:
        fvu = ((out.sae_out - out.sae_in)**2).mean() / out.sae_in.var()   # FVU in the SAE's input space
        l0 = (out.feature_acts > 0).float().sum(-1).mean()
        n_dead = int(dead.sum())
        msg = f"step {step:6d} | loss {out.loss.item():.3f} | FVU {fvu.item():.3f} | L0 {l0.item():.1f} | dead {n_dead}"
        if MODE in ("hybrid", "outbias"):
            with t.no_grad():   # FVU on the FULL activation h -- the number to compare against the full SAE's 0.143
                fvu_h = ((h_batch - (P_batch + out.sae_out * scale))**2).mean() / h_batch.var()
            msg += f" | FVU_h {fvu_h.item():.3f}"
        print(msg)

    # rolling resume checkpoint (overwrites the same file each time)
    if step % 2000 == 0:
        ckpt = {
            "sae": sae.state_dict(),
            "cfg": sae.cfg,
            "scale": scale,
            "step": step,
            "mode": MODE,
            "opt": opt.state_dict(),
            "n_since_fired": n_since_fired,
        }
        if MODE in ("hybrid", "outbias"):
            ckpt["linear_map"] = linear_map.state_dict()   # JOINTLY-TRAINED map; differs from the frozen P.pt
        t.save(ckpt, f"{CACHE_DIR}/sae_{MODE}_k{K}_latest.pt")

    seen_tokens += h_batch.shape[0]
    if seen_tokens >= TRAIN_TOKENS:        # stop once the token budget is hit
        break

# final artifact for the trivial-fraction eval (weights + cfg + scale; no optimizer needed)
final_ckpt = {
    "sae": sae.state_dict(),
    "cfg": sae.cfg,
    "scale": scale,
    "step": step,
    "mode": MODE,
}
if MODE in ("hybrid", "outbias"):
    final_ckpt["linear_map"] = linear_map.state_dict()     # needed at eval to rebuild the trained P[token]

# save the JOINTLY-TRAINED prediction table so eval can add/subtract it (hybrid & outbias)
if MODE in ("hybrid", "outbias"):
    with t.no_grad():
        t.save(linear_map(embed_W.float()), f"{CACHE_DIR}/P_{MODE}_k{K}.pt")
t.save(final_ckpt, f"{CACHE_DIR}/sae_{MODE}_k{K}_final.pt")
print(f"saved final SAE -> {CACHE_DIR}/sae_{MODE}_k{K}_final.pt")
