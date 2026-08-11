"""Matched-covariance Gaussian NULL for the FVU comparison.

An SAE reconstructs pure Gaussian noise respectably at k=64 -- second-order structure alone buys a
lot. So "the random arm's resid SAE reaches FVU 0.45" is meaningless on its own; the question is
whether it beats what the SAME architecture achieves on data with the SAME mean and covariance but
NO higher-order structure at all. That is this script.

    fit mu, Sigma of the target on real activations
      -> train an identical BatchTopK SAE on x = mu + L z   (L L^T = Sigma, z ~ N(0, I))
      -> report its FVU on fresh synthetic samples

Read the result as a FLOOR. Interpretation:
    real FVU  <<  null FVU   -> the target has non-Gaussian structure the SAE is exploiting
    real FVU  ~=  null FVU   -> the SAE is only reproducing the covariance; nothing else is there

MODE picks which target to match:
    full   -> Sigma of h                     (null for sae_full)
    resid  -> Sigma of r = h - P[tok]        (null for sae_resid)   <- the one the hypothesis needs

Run it per arm, with that arm's VARIANT / HF_REPO / OUT_DIR:
    MODE=resid VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-repo> OUT_DIR=... python -u gauss_null.py

Cheap compared to a real run: fitting Sigma needs one short pass through gemma, and training then
needs NO forward passes at all -- just a matmul per batch -- so the SAE trains in minutes.

Env: MODE, K, COV_TOKENS, TRAIN_TOKENS, EVAL_TOKENS, RIDGE, SEED, OUT_DIR.
"""
import os
import json
import torch as t
import torch.nn as nn
from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput

import sys
# this experiment lives in SAE_linearmaps/randomized/ -- the shared modules
# (activations, hf_io, sae_lens loaders) live one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import load_model, activation_stream, D_IN, LAYER, VARIANT, INIT_SEED
from hf_io import push, pull

MODE         = os.environ.get("MODE", "resid")                        # "full" | "resid"
K            = int(os.environ.get("K", 64))
D_SAE        = int(os.environ.get("D_SAE", 16384))                    # MATCH the real run being nulled
LR           = float(os.environ.get("LR", 4e-4))                      # ditto
BATCH        = 4096
COV_TOKENS   = int(os.environ.get("COV_TOKENS", 5_000_000))           # n >> d=2304 for a stable Sigma
TRAIN_TOKENS = int(os.environ.get("TRAIN_TOKENS", 20_000_000))        # MATCH the real runs
EVAL_TOKENS  = int(os.environ.get("EVAL_TOKENS", 2_000_000))
RIDGE        = float(os.environ.get("RIDGE", 1e-6))                   # keeps Sigma positive-definite
SEED         = int(os.environ.get("SEED", 0))
OUT_DIR      = os.environ.get("OUT_DIR", "/workspace/out")
assert MODE in ("full", "resid"), f"MODE={MODE!r} must be 'full' or 'resid'"
os.makedirs(OUT_DIR, exist_ok=True)

# A null is only a floor for a run with the SAME architecture and budget, so non-default settings
# are tagged into the artifact names exactly as train_sae_res.py tags the real runs -- otherwise a
# k=32/d73728 null would overwrite the k=64/16k null it is NOT comparable to. Historical defaults
# reproduce the historical names.
SUFFIX = ""
if D_SAE != 16384:             SUFFIX += f"_d{D_SAE}"
if LR != 4e-4:                 SUFFIX += f"_lr{LR:g}"
if TRAIN_TOKENS != 20_000_000: SUFFIX += f"_{TRAIN_TOKENS // 1_000_000}M"
print(f"[gauss_null] MODE={MODE} K={K} D_SAE={D_SAE} TRAIN_TOKENS={TRAIN_TOKENS:,} SUFFIX={SUFFIX!r}")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model  = load_model(device)
V      = model.cfg.d_vocab
print(f"[gauss_null] VARIANT={VARIANT} INIT_SEED={INIT_SEED} MODE={MODE} K={K} "
      f"cov={COV_TOKENS:,} train={TRAIN_TOKENS:,}")

# resid's target needs the frozen greedy map, rebuilt exactly as train_sae_res.py does
P = None
if MODE == "resid":
    lm = nn.Linear(D_IN, D_IN).to(device)
    lm.load_state_dict(t.load(pull("linear_map_layer_13.pt"), weights_only=False))
    lm.eval()
    with t.no_grad():
        embed_table = model.embed(t.arange(V, device=device)).float()
        P = lm(embed_table)
        del embed_table
    t.cuda.empty_cache()

# ---- pass 1: accumulate mean and covariance of the TARGET ----
# Sigma = E[x x^T] - mu mu^T, from streamed sums. (D, D) float64 = 42 MB, trivial.
S1 = t.zeros(D_IN, dtype=t.float64, device=device)
S2 = t.zeros(D_IN, D_IN, dtype=t.float64, device=device)
n = 0
with t.no_grad():
    for step, (hh, tt) in enumerate(activation_stream(
            model, device, batch=8192, seed=SEED, max_tokens=COV_TOKENS)):
        x = (hh - P[tt]) if MODE == "resid" else hh
        xd = x.double()
        S1 += xd.sum(0)
        S2 += xd.T @ xd
        n  += x.shape[0]
        if step % 100 == 0:
            print(f"  covariance: {n:,}/{COV_TOKENS:,} tokens")

mu    = (S1 / n).float()
Sigma = (S2 / n - t.outer(S1 / n, S1 / n))
del S1, S2
# symmetrize (kills accumulated asymmetry) and ridge the diagonal so cholesky cannot fail
Sigma = 0.5 * (Sigma + Sigma.T)
Sigma += RIDGE * t.diag(Sigma).mean() * t.eye(D_IN, dtype=t.float64, device=device)
L = t.linalg.cholesky(Sigma).float()               # (D, D) lower-triangular: L L^T = Sigma
trace_var = t.diag(Sigma).sum().item() / D_IN      # mean per-dim variance, for reporting
print(f"  Sigma fitted on {n:,} tokens | mean per-dim var {trace_var:.4f} | "
      f"||mu|| {mu.norm().item():.3f}")

# gemma is no longer needed -- synthetic sampling is a matmul, no forward passes
del model
if MODE == "resid":
    del P, lm
t.cuda.empty_cache()

gen = t.Generator(device=device).manual_seed(SEED)


def synth(bs):
    """One batch from N(mu, Sigma): x = mu + z L^T with z ~ N(0, I)  =>  Cov(x) = L L^T = Sigma."""
    z = t.randn(bs, D_IN, generator=gen, device=device)
    return mu + z @ L.T


# ---- the SAE: identical config, init and optimizer to the real runs ----
cfg = BatchTopKTrainingSAEConfig(
    d_in=D_IN, d_sae=D_SAE, k=K,
    dtype="float32", device=str(device),
    apply_b_dec_to_input=True,
    normalize_activations="none",
)
t.manual_seed(SEED)
sae = BatchTopKTrainingSAE(cfg).to(device)

# same scalar normalization the real runs apply, measured the same way
with t.no_grad():
    scale = (synth(100_000).norm(dim=-1).mean() / (D_IN ** 0.5)).item()
print(f"  scale {scale:.4f}")

opt = t.optim.Adam(sae.parameters(), lr=LR)
n_since_fired = t.zeros(D_SAE, device=device)
DEAD_WINDOW = 200
steps = TRAIN_TOKENS // BATCH

for step in range(steps):
    dead = n_since_fired > DEAD_WINDOW
    x = synth(BATCH) / scale
    out = sae.training_forward_pass(TrainStepInput(
        sae_in=x, coefficients={}, dead_neuron_mask=dead,
        n_training_steps=step, is_logging_step=False))
    opt.zero_grad()
    out.loss.backward()
    t.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    opt.step()

    fired = (out.feature_acts > 0).any(0)
    n_since_fired += 1
    n_since_fired[fired] = 0

    if step % 500 == 0:
        fvu_s = ((out.sae_out - out.sae_in) ** 2).mean() / out.sae_in.var()
        l0    = (out.feature_acts > 0).float().sum(-1).mean()
        print(f"step {step:6d}/{steps} | loss {out.loss.item():.3f} | FVU {fvu_s.item():.3f} | "
              f"L0 {l0.item():.1f} | dead {int(dead.sum())}")

# ---- evaluate on FRESH synthetic samples, in both conventions (matching eval_fvu.py) ----
sae.eval()
sse = 0.0
x_sum = t.zeros(D_IN, dtype=t.float64, device=device)
x_sumsq = 0.0
n_rows = 0
with t.no_grad():
    for _ in range(EVAL_TOKENS // 8192):
        x = synth(8192)
        x_hat = sae.decode(sae.encode(x / scale)) * scale
        sse     += ((x - x_hat) ** 2).double().sum().item()
        x_sum   += x.double().sum(0)
        x_sumsq += (x ** 2).double().sum().item()
        n_rows  += x.shape[0]

n_elem   = n_rows * D_IN
var_flat = x_sumsq / n_elem - (x_sum.sum().item() / n_elem) ** 2
ss_cent  = x_sumsq - (x_sum ** 2).sum().item() / n_rows
null_flat, null_cent = sse / (n_elem * var_flat), sse / ss_cent

print(f"\n=== GAUSSIAN NULL | {VARIANT} (seed {INIT_SEED}) | target={MODE} | {n_rows:,} eval tokens ===")
print(f"  null FVU   flat {null_flat:.4f}   centred {null_cent:.4f}")
print(f"  compare against eval_fvu.py's "
      f"{'fvu_h_full' if MODE == 'full' else 'fvu_r_resid'} for this arm:")
print("    real << null -> genuine non-Gaussian structure;  real ~= null -> covariance only")

results = {"variant": VARIANT, "init_seed": INIT_SEED, "layer": LAYER, "k": K, "target": MODE,
           "cov_tokens": n, "train_tokens": steps * BATCH, "eval_tokens": n_rows,
           "scale": scale, "mean_per_dim_var": trace_var,
           "null_fvu": {"flat": null_flat, "centred": null_cent}}
path = f"{OUT_DIR}/gauss_null_{VARIANT}_s{INIT_SEED}_{MODE}_k{K}{SUFFIX}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {path}")
push(path)

# The LOCAL name carries the arm, the REMOTE name does not: repos are already per-arm, but two
# arms run concurrently from one OUT_DIR would otherwise write the same path -- and since save is
# immediately followed by push, the loser's weights get uploaded to the winner's repo.
ck_name = f"sae_gauss_{MODE}_k{K}{SUFFIX}_final.pt"                       # name on HF (unchanged)
ck_path = f"{OUT_DIR}/{VARIANT}_s{INIT_SEED}_{ck_name}"                   # arm-specific local path
t.save({"sae": sae.state_dict(), "cfg": sae.cfg, "scale": scale,
        "step": steps, "mode": f"gauss_{MODE}", "variant": VARIANT}, ck_path)
push(ck_path, name=ck_name)
