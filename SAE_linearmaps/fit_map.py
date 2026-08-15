"""Refit the embedding -> layer-13 residual linear map ON THE FLY and push it to HF.

This regenerates linear_map_layer_13.pt, which used to live only on the network volume
(the old train.py fit it from cached embedding/activation chunks). We no longer store
activations: we stream gemma over openwebtext, and because a token's embedding is a
deterministic lookup, the map's INPUT is just embed_table[tok] -- so we only need (tok, h)
pairs, exactly what activation_stream yields.

The map is the least-squares (ridge) solution, solved in closed form from streamed Gram
matrices -- deterministic, no learning-rate / epoch tuning. It minimizes  ||W·embed[tok] + b - h||^2.
Run this ONCE; every SAE run then just pulls the result from HF.

    HF_TOKEN=hf_xxx python fit_map.py
"""
import os
import torch as t
import torch.nn as nn
from activations import (load_model, activation_stream, take_sample, D_IN, VARIANT,
                         INIT_SEED, LAYER, MAP_FILE)
from hf_io import push

MAP_TOKENS = int(os.environ.get("MAP_TOKENS", 20_000_000))   # plenty to fit a 2304x2304 map
RIDGE      = float(os.environ.get("RIDGE", 1e-2))            # tiny L2 for a numerically stable solve
BATCH      = 8192
OUT_DIR    = os.environ.get("OUT_DIR", "/workspace/out")
os.makedirs(OUT_DIR, exist_ok=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model  = load_model(device)

# embedding lookup table: row t = the (frozen) input the map sees for token t
with t.no_grad():
    embed_table = model.embed(t.arange(model.cfg.d_vocab, device=device)).float()  # (V, 2304)

# ---- stream (tok, h) and accumulate the normal-equation Gram matrices ----
# augment the input with a constant 1 so the bias is fit too:  x_aug = [embed[tok], 1]
# A = sum x_aug x_aug^T  (D+1, D+1) ;  Bmat = sum x_aug h^T  (D+1, D) ;  then solve  A · beta = Bmat
D = D_IN
A    = t.zeros(D + 1, D + 1, dtype=t.float64, device=device)
Bmat = t.zeros(D + 1, D,     dtype=t.float64, device=device)
seen = 0
for step, (h_b, tok_b) in enumerate(activation_stream(model, device, batch=BATCH, seed=0, max_tokens=MAP_TOKENS)):
    x     = embed_table[tok_b]                                            # (b, D)
    x_aug = t.cat([x, t.ones(x.shape[0], 1, device=device)], dim=1)       # (b, D+1)
    A    += (x_aug.T @ x_aug).double()     # per-batch matmul in float32, accumulated in float64
    Bmat += (x_aug.T @ h_b).double()
    seen += h_b.shape[0]
    if step % 200 == 0:
        print(f"  accumulated {seen:,}/{MAP_TOKENS:,} tokens")

# ---- closed-form ridge solve ----
A[:D, :D] += RIDGE * t.eye(D, dtype=t.float64, device=device)   # regularize the weight block, not the bias
beta = t.linalg.solve(A, Bmat)                  # (D+1, D)
W = beta[:D, :].T.contiguous().float()          # (D_out, D_in): the nn.Linear weight
b = beta[D, :].contiguous().float()             # (D_out,):      the nn.Linear bias

linear_map = nn.Linear(D, D)
with t.no_grad():
    linear_map.weight.copy_(W)
    linear_map.bias.copy_(b)

# ---- report fit quality on a FRESH sample ----
# Expect ~0.56 TRAINED, ~0.32 RANDOM (measured 2026-08-10: 0.5616 / 0.3212 on the 100k-token
# startup sample in train_sae_res.py; the trained value cross-checks against eval_fvu's flat
# Var(r)/Var(h)=0.4411 -> 0.5589. An older "~0.66 on either arm" note here was stale on BOTH
# counts). The arm gap is NOT a sign randomization failed or that the
# random model is "less token-static". The token-static share itself is ~0.28 for BOTH arms
# (context_var.py, 2026-08-02: lookup-table R^2 0.2743 trained vs 0.2877 rand_all, min_count>=30,
# 20M tokens) -- random attention is DIFFUSE, so every position is a broad random mixture of its
# context, producing just as much context-dependent variance as real computation, merely unstructured.
# The arms differ HERE only because this R^2 divides by Hs.var(), a FLAT variance about one scalar
# mean. Trained gemma's massive-activation dims give its mean VECTOR a large cross-dimension spread,
# which inflates that denominator and which the map predicts trivially -> 0.56. A random arm has no
# massive activations (|h| mean == sqrt(d * var), i.e. per-element mean ~0), so flat == centred and
# the number collapses onto the true ~0.29. See context_var.py's docstring on the two conventions.
Hs, Ts = take_sample(model, device, n_tokens=200_000, seed=123)
Hs, Ts = Hs.float().to(device), Ts.to(device)
with t.no_grad():
    pred = linear_map.to(device)(embed_table[Ts])
    r2 = 1 - ((Hs - pred) ** 2).mean() / Hs.var()
print(f"linear map R^2 on held-out sample: {r2.item():.4f}   (expect ~0.66 trained / ~0.30 random)")
print(f"  VARIANT={VARIANT} INIT_SEED={INIT_SEED}")

path = f"{OUT_DIR}/{MAP_FILE}"
t.save(linear_map.cpu().state_dict(), path)
push(path, MAP_FILE)
print(f"saved {path} and pushed to HF")
