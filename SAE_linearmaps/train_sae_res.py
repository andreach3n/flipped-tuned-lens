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

MODEL_NAME = "google/gemma-2-2b"
LAYER=13
D_IN=2304
D_SAE=16384
K=64
LR=4e-4
CACHE_DIR="/workspace/sae_cache_layer13"
BATCH = 4096
MODE = "full"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

cfg = BatchTopKTrainingSAEConfig(
    d_in=D_IN, d_sae=D_SAE, k=K,
    dtype="float32", device=str(device),
    apply_b_dec_to_input=True,
    normalize_activations="none",   # you normalize in iter_batches
)
sae = BatchTopKTrainingSAE(cfg).to(device)

linear_map = nn.Linear(2304, 2304).to(device)
linear_map.load_state_dict(t.load("/workspace/linear_map_layer_13.pt", weights_only=False))
linear_map.eval()

with t.no_grad():
    P = linear_map(model.embed(t.arange(model.cfg.d_vocab, device=device)).float())  # (V, 2304)
    t.save(P, f"{CACHE_DIR}/P.pt")

del model                # free ~5 GB — not needed after P is built
t.cuda.empty_cache()

SANITY_N = 100_000   # subsample for the check/scale — a full 1M-token chunk on GPU OOMs
chunk_h   = t.cat(t.load(f"{CACHE_DIR}/layer_13_chunk_1.pt"), dim=0)[:SANITY_N].float().to(device)
chunk_tok = t.cat(t.load(f"{CACHE_DIR}/tokens_chunk_1.pt"),   dim=0)[:SANITY_N].to(device)
r = chunk_h - P[chunk_tok]
explained = 1 - (r**2).mean() / chunk_h.var()
print(explained.item())   # want ≈ 0.66

# from chunk_1 (already loaded above as chunk_h / chunk_tok):
sample = (chunk_h - P[chunk_tok]) if MODE == "resid" else chunk_h
scale  = sample.norm(dim=-1).mean() / (D_IN ** 0.5)   # a single scalar
del chunk_h, chunk_tok, r, sample    # free the ~9 GB sanity-check tensors before training
t.cuda.empty_cache()

def iter_batches(mode, scale):
    layer_files = sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt"))
    for lf in layer_files:
        tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
        h = t.cat(t.load(lf), dim=0)
        tok = t.cat(t.load(tf), dim=0)

        shuffled_idx = t.randperm(h.shape[0])
        h = h[shuffled_idx]
        tok = tok[shuffled_idx]

        for start in range(0, h.shape[0], BATCH):
            h_batch = h[start: start+BATCH].float().to(device)
            tok_batch = tok[start: start+BATCH].to(device)
            res_batch = h_batch - P[tok_batch] if MODE == "resid" else h_batch
            yield res_batch / scale

opt = t.optim.Adam(sae.parameters(), lr=LR)
n_since_fired = t.zeros(D_SAE, device=device)
DEAD_WINDOW = 200
N = 100
for step, x in enumerate(iter_batches(MODE, scale)):
    dead = n_since_fired > DEAD_WINDOW

    out = sae.training_forward_pass(TrainStepInput(sae_in=x, coefficients={}, dead_neuron_mask=dead, n_training_steps=step, is_logging_step=False))
    opt.zero_grad()
    out.loss.backward()
    t.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    opt.step()

    fired = (out.feature_acts > 0).any(0)             # which features fired this batch
    n_since_fired += 1
    n_since_fired[fired] = 0 # reset the ones that just fired back to 0

    if step % 100 == 0:
        fvu = ((out.sae_out - out.sae_in)**2).mean() / out.sae_in.var()
        l0 = (out.feature_acts > 0).float().sum(-1).mean()
        n_dead = int(dead.sum())
        print(f"step {step:6d} | loss {out.loss.item():.3f} | FVU {fvu.item():.3f} | L0 {l0.item():.1f} | dead {n_dead}")

    # rolling resume checkpoint (overwrites the same file each time)
    if step % 2000 == 0:
        t.save({
            "sae": sae.state_dict(),
            "cfg": sae.cfg,
            "scale": scale,
            "step": step,
            "mode": MODE,
            "opt": opt.state_dict(),
            "n_since_fired": n_since_fired,
        }, f"{CACHE_DIR}/sae_{MODE}_latest.pt")

# final artifact for the trivial-fraction eval (weights + cfg + scale; no optimizer needed)
t.save({
    "sae": sae.state_dict(),
    "cfg": sae.cfg,
    "scale": scale,
    "step": step,
    "mode": MODE,
}, f"{CACHE_DIR}/sae_{MODE}_final.pt")
print(f"saved final SAE -> {CACHE_DIR}/sae_{MODE}_final.pt")
