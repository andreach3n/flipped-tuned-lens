"""Fit the frozen embedding -> layer-L linear map P for pythia-1b. RUN THIS ONCE PER ARM.

P[t] = W @ embed_in[t] + b, fit by ridge least squares to minimise ||P[tok] - h||^2. The
skip-embed SAE then encodes r = h - P[tok]; `resid` keeps this map FROZEN (unlike `hybrid`, which
trains it jointly), so it adds no trainable parameters to the SAE run and the capacity comparison
against the plain arm stays exactly matched.

WHY PER ARM, AND WHY THIS IS THE MOST DANGEROUS FILE HERE. The map is a property of a specific
model's embedding table and a specific layer. Fitting it on trained pythia and reusing it on the
randomized checkpoint would produce a `resid` cell that trains, gates, scores and reports without
a single error -- and means nothing. ../../fit_map.py carries the same warning for Gemma; this is
the Pythia restatement of it. The saved file records the arm and an embedding fingerprint, and
skipembed.load_beta() refuses to attach a map whose fingerprint does not match the live model.

SOLVED IN CLOSED FORM from streamed Gram matrices -- deterministic, no learning rate, no epochs,
identical on every rerun. Augmenting the input with a constant 1 fits the bias in the same solve:

    A = sum x_aug x_aug^T   (D+1, D+1)      x_aug = [embed_in[tok], 1]
    B = sum x_aug h^T       (D+1, D)
    beta = (A + ridge*I)^-1 B               (D+1, D)   -- trailing row is the bias

The residual sum of squares comes out of the SAME accumulators, so the fit is scored without a
second pass over the corpus:

    SSE = S_hh - 2*<beta, B> + tr(beta^T A beta)

We report explained variance BOTH flat and centred. On Gemma the two disagreed by a lot on the
trained arm and barely at all on the random one -- not a bug, an artifact of the flat denominator
being inflated by the mean vector's cross-dimension spread, which a trained model has (massive
activations) and a random one does not. Expect the same asymmetry on Pythia: verify_randomization
measured max-dim-var-ratio 879.19 trained vs 1.55 random at L8. Quote the CENTRED number.

    ARM=trained python -u fit_map_pythia.py
    ARM=rand RAND_MODEL=/dev/shm/pythia1b_rand_s0 python -u fit_map_pythia.py
"""
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import LAYER, MODEL_NAME  # noqa: E402
from skipembed import embed_fingerprint         # noqa: E402

ARM        = os.environ.get("ARM", "trained")
RAND_MODEL = os.environ.get("RAND_MODEL", "/dev/shm/pythia1b_rand_s0")
DATASET    = os.environ.get("DATASET", "Skylion007/openwebtext")
# The SAME slice train_saes.sh trains on. The map is model-side preprocessing, not a held-out
# estimate, so fitting it on the training distribution is the correct choice -- and it keeps the
# map from being fit on a distribution the SAE never sees.
SPLIT      = os.environ.get("SPLIT", "train[:3%]")
MAP_TOKENS = int(os.environ.get("MAP_TOKENS", 20_000_000))   # matches the Gemma precedent
RIDGE      = float(os.environ.get("RIDGE", 1e-2))            # tiny L2, for a stable solve only
CTX        = int(os.environ.get("CTX", 2048))
BATCH      = int(os.environ.get("FIT_BATCH", 8))             # sequences per forward
OUT_DIR    = os.environ.get("OUT_DIR", "/dev/shm/maps")
OUT        = os.environ.get("MAP_OUT", os.path.join(OUT_DIR, f"P_pythia1b_L{LAYER}_{ARM}.pt"))

from datasets import load_dataset                        # noqa: E402
from transformers import AutoModel, AutoTokenizer        # noqa: E402

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model_path = MODEL_NAME if ARM == "trained" else RAND_MODEL
if ARM not in ("trained", "rand"):
    raise SystemExit("ARM must be 'trained' or 'rand'")
if ARM == "rand" and not os.path.exists(os.path.join(model_path, "config.json")):
    raise SystemExit(f"no checkpoint at {model_path} -- run randomize_pythia.py first")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(f"[fit_map] ARM={ARM} model={model_path} layer={LAYER} device={device}")

# torch_dtype=, not dtype=: the latter only exists from transformers 4.57 and we pin 4.56.1.
# Keep this consistent with randomize_pythia.py and check_saes.py.
model = AutoModel.from_pretrained(model_path, torch_dtype=t.bfloat16).to(device).eval()
tokzr = AutoTokenizer.from_pretrained(MODEL_NAME)

D = model.config.hidden_size
V = model.get_input_embeddings().weight.shape[0]
fp = embed_fingerprint(model.get_input_embeddings().weight)
print(f"  d_model={D} vocab={V} embed_fingerprint={fp}")

embed_table = model.get_input_embeddings().weight.detach().float()   # (V, D)

captured = {}
model.layers[LAYER].register_forward_hook(
    lambda _m, _i, out: captured.__setitem__(
        "h", (out[0] if isinstance(out, tuple) else out).detach())
)

A    = t.zeros(D + 1, D + 1, dtype=t.float64, device=device)
Bmat = t.zeros(D + 1, D,     dtype=t.float64, device=device)
S_hh = 0.0
seen = 0

ds = load_dataset(DATASET, split=SPLIT, streaming=True)
buf: list[int] = []
with t.no_grad():
    rows: list[list[int]] = []
    for rec in ds:
        buf.extend(tokzr(rec["text"])["input_ids"])
        while len(buf) >= CTX:
            rows.append(buf[:CTX])
            buf = buf[CTX:]
        if len(rows) < BATCH:
            continue

        toks = t.tensor(rows[:BATCH], device=device)
        rows = rows[BATCH:]
        model(toks)
        h = captured["h"].reshape(-1, D).float()                     # (b*CTX, D)
        tok_flat = toks.reshape(-1)

        x = embed_table[tok_flat]                                    # (b*CTX, D)
        x_aug = t.cat([x, t.ones(x.shape[0], 1, device=device)], dim=1)
        A    += (x_aug.T @ x_aug).double()
        Bmat += (x_aug.T @ h).double()
        S_hh += float((h.double() ** 2).sum())
        seen += h.shape[0]

        if (seen // (BATCH * CTX)) % 100 == 0:
            print(f"  accumulated {seen:,}/{MAP_TOKENS:,} tokens", flush=True)
        if seen >= MAP_TOKENS:
            break

print(f"  done streaming: {seen:,} tokens")

# ---- solve -----------------------------------------------------------------------------
eye = t.eye(D + 1, dtype=t.float64, device=device)
eye[-1, -1] = 0.0                     # do not penalise the bias
beta = t.linalg.solve(A + RIDGE * eye, Bmat)                          # (D+1, D)

# ---- score, from the same accumulators (no second pass) --------------------------------
sse = S_hh - 2.0 * float((beta * Bmat).sum()) + float(t.einsum("jd,jk,kd->", beta, A, beta))
n = float(A[-1, -1])
mu = Bmat[-1] / n                                                     # sum h / n
sst_flat = S_hh
sst_cent = S_hh - n * float((mu ** 2).sum())

ev_flat, ev_cent = 1.0 - sse / sst_flat, 1.0 - sse / sst_cent
print()
print(f"  explained variance  flat     {ev_flat:.4f}")
print(f"  explained variance  centred  {ev_cent:.4f}   <- quote this one")
print(f"  Var(r)/Var(h)       centred  {sse / sst_cent:.4f}")
print()

if not (0.0 < ev_cent < 0.999):
    raise SystemExit(f"FAIL: centred explained variance {ev_cent:.4f} is not in (0, 0.999) -- a "
                     f"map that explains everything or nothing means the hookpoint, the "
                     f"embedding table or the solve is wrong. Refusing to save.")

t.save({
    "beta": beta.float().cpu(),
    "embed_fingerprint": fp,
    "arm": ARM,
    "model": model_path,
    "layer": LAYER,
    "d_model": D,
    "vocab": V,
    "tokens": seen,
    "ridge": RIDGE,
    "dataset": DATASET,
    "split": SPLIT,
    "explained_variance_flat": ev_flat,
    "explained_variance_centred": ev_cent,
}, OUT)
print(f"saved {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
print("\nPush this with the SAEs. Without the exact map, the resid checkpoint's encoder input "
      "cannot be reproduced and the SAE is uninterpretable.")
