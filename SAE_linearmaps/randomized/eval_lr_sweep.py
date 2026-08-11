"""LR sweep screen: do 1e-4 / 4e-4 / 1e-3 give materially different SAEs at matched 20M tokens?

Context: the Heap et al. auto-interp result does not replicate in our pipeline. Possibility 2 is
that our SAEs are simply undertrained -- token count (handled by the 100M runs) and LR (this).
This is the CHEAP SCREEN: if all three LRs land in the same place, LR is not the explanation and
no 100M LR run is warranted; if they diverge, the winner earns one.

The 4e-4 midpoint is FREE -- it is the t20M milestone of the 100M default-LR run. That is a fair
comparison because train_sae_res.py uses a CONSTANT LR (plain Adam, no scheduler), so "20M tokens
into a 100M run" and "a 20M run" are the same optimization state at the same LR.

Metrics, in increasing order of relevance to the actual question:
  FVU (flat + centred)  -- reconstruction; the conventions are eval_fvu.py's, see its docstring
  L0                    -- sanity, must be ~k
  alive fraction        -- THE one to watch. Auto-interp samples latents; a dictionary where most
                           latents never fire has far fewer real features to explain, which depresses
                           interpretability scores for reasons that have nothing to do with the model.
                           If LR moves anything here, it moves the replication.

Run on a free GPU of the training pod (needs the random arm's repo):

    CUDA_VISIBLE_DEVICES=4 VARIANT=rand_all INIT_SEED=0 \
      HF_REPO=andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0 \
      OUT_DIR=/dev/shm/out_lrsweep python -u randomized/eval_lr_sweep.py
"""
import os
import json
import torch as t
from sae_lens import BatchTopKTrainingSAE

from activations import load_model, activation_stream, D_IN, LAYER, VARIANT, INIT_SEED
from hf_io import push, pull

N_TOKENS = int(os.environ.get("N_TOKENS", 2_000_000))   # FVU converges fast; 2M is plenty
BATCH    = int(os.environ.get("BATCH", 8192))
SEED     = int(os.environ.get("SEED", 7))               # != training's 0 -> effectively held-out docs
OUT_DIR  = os.environ.get("OUT_DIR", "/workspace/out")
os.makedirs(OUT_DIR, exist_ok=True)

# label -> artifact in the random arm's HF repo. All three are MODE=full, k=32, d_sae=73728, 20M tokens.
RUNS = [
    ("1e-4", "sae_full_k32_d73728_lr0.0001_final.pt"),
    ("4e-4", "sae_full_k32_d73728_100M_t20M.pt"),      # milestone of the default-LR 100M run
    ("1e-3", "sae_full_k32_d73728_lr0.001_final.pt"),
]

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model  = load_model(device)
print(f"[eval_lr_sweep] VARIANT={VARIANT} INIT_SEED={INIT_SEED} layer={LAYER} "
      f"N_TOKENS={N_TOKENS:,} stream_seed={SEED}")

saes = []
for label, name in RUNS:
    ckpt = t.load(pull(name), weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    d_sae = sae.cfg.d_sae
    saes.append({
        "label": label, "file": name, "sae": sae, "scale": float(ckpt["scale"]),
        "step": int(ckpt.get("step", -1)), "d_sae": d_sae,
        "sse": 0.0, "l0_sum": 0.0,
        "fires": t.zeros(d_sae, dtype=t.float64, device=device),
    })
    print(f"  loaded {label:5s} d_sae={d_sae} step={ckpt.get('step')} scale={float(ckpt['scale']):.4f}")

# float64 accumulators, per-DIMENSION for h so both FVU conventions are available (as eval_fvu.py).
h_sum   = t.zeros(D_IN, dtype=t.float64, device=device)
h_sumsq = 0.0
n_rows  = 0

with t.no_grad():
    for step, (hh, _tt) in enumerate(activation_stream(
            model, device, batch=BATCH, seed=SEED, max_tokens=N_TOKENS)):
        for r in saes:
            acts  = r["sae"].encode(hh / r["scale"])
            recon = r["sae"].decode(acts) * r["scale"]
            r["sse"]    += ((hh - recon) ** 2).double().sum().item()
            fired        = acts > 0
            r["l0_sum"] += fired.sum().double().item()
            r["fires"]  += fired.sum(0).double()

        h_sum += hh.double().sum(0)
        h_sumsq += (hh ** 2).double().sum().item()
        n_rows += hh.shape[0]
        if step % 50 == 0:
            print(f"  {n_rows:,}/{N_TOKENS:,} tokens")

n_elem     = n_rows * D_IN
var_h_flat = h_sumsq / n_elem - (h_sum.sum().item() / n_elem) ** 2
ss_h_cent  = h_sumsq - (h_sum ** 2).sum().item() / n_rows

print(f"\n=== LR sweep | {VARIANT} seed {INIT_SEED} | {n_rows:,} held-out tokens ===")
print(f"{'LR':>6s} {'FVU flat':>9s} {'FVU cent':>9s} {'L0':>6s} "
      f"{'alive':>8s} {'alive%':>7s} {'>=10x':>8s}")
results = []
for r in saes:
    flat  = r["sse"] / (n_elem * var_h_flat)
    cent  = r["sse"] / ss_h_cent
    l0    = r["l0_sum"] / n_rows
    alive = int((r["fires"] > 0).sum().item())          # fired at least once in N_TOKENS
    solid = int((r["fires"] >= 10).sum().item())        # fired >=10x: enough examples to explain
    pct   = 100.0 * alive / r["d_sae"]
    print(f"{r['label']:>6s} {flat:>9.4f} {cent:>9.4f} {l0:>6.1f} "
          f"{alive:>8d} {pct:>6.1f}% {solid:>8d}")
    results.append({
        "lr": r["label"], "file": r["file"], "step": r["step"], "d_sae": r["d_sae"],
        "fvu_h": {"flat": flat, "centred": cent}, "l0": l0,
        "alive": alive, "alive_frac": alive / r["d_sae"], "alive_ge10": solid,
    })

print("\nRead: if these rows are within noise of each other, LR is NOT the non-replication's cause")
print("and no 100M LR run is warranted. Divergence in `alive%` matters more than in FVU.")

path = f"{OUT_DIR}/lr_sweep_{VARIANT}_s{INIT_SEED}.json"
with open(path, "w") as f:
    json.dump({"variant": VARIANT, "init_seed": INIT_SEED, "layer": LAYER,
               "n_tokens": n_rows, "stream_seed": SEED, "runs": results}, f, indent=2)
print(f"\nsaved {path}")
push(path)
