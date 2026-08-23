"""SAE health gate: run this between training and delphi, on both arms.

delphi costs GPU-hours and judge time, and it only scores latents with >= min_examples (200)
activations. A dictionary that is half dead still produces a complete, plausible set of AUROC
numbers -- it just silently changes WHICH latents got sampled, differently per arm. That is
the same shape of failure as the conversion bug in ../DELPHI_SETUP.md: no error, wrong answer.
So the numbers that decide whether the run is usable get measured here, before any judge runs.

Reports, on held-out tokens the SAE never saw (a disjoint slice of the corpus):
    FVU flat / centred   reconstruction quality. Centred is the honest one -- flat FVU is
                         inflated by the mean vector's cross-dimension spread, which a trained
                         model has (massive activations) and a random one does not, so the flat
                         number flatters the random arm for a reason that is not about features.
    L0                   should be exactly k; anything else means a config mismatch.
    alive %              fraction of latents that fire at least once, and at least 10 times.
                         THE LOAD-BEARING NUMBER. With sparsify's auxk_alpha=0 default nothing
                         revives dead latents.
    firing rate          mean firings per token among latents 0..499 -- the exact latents
                         delphi will score, since --max_latents N is torch.arange(N), the FIRST
                         N indices, not the top-firing N.

MODE=resid changes what "reconstruction" means: the SAE's input and target are both
r = h - P[tok], so FVU is measured against Var(r), not Var(h). The two are not comparable --
r carries roughly half the variance and is the harder half to sparse-code, so a resid FVU will
look worse than a full FVU on the same model and that is expected, not a regression. The script
additionally reports Var(r)/Var(h) so the map's share is on the record next to the run.

    SAE_DIR=/dev/shm/saes/pythia1b_trained_L8_R64_k32_100M ARM=trained python -u check_saes.py
    SAE_DIR=... ARM=rand RAND_MODEL=/dev/shm/pythia1b_rand_s0 python -u check_saes.py
    MODE=resid MAP=/dev/shm/maps/P_pythia1b_L8_trained.pt SAE_DIR=... ARM=trained \
      python -u check_saes.py
"""
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import LAYER, MODEL_NAME  # noqa: E402
from skipembed import PROBE_NAME, attach, hookpoint, load_beta  # noqa: E402

SAE_DIR    = os.environ["SAE_DIR"]
ARM        = os.environ.get("ARM", "trained")
MODE       = os.environ.get("MODE", "full")        # "full" | "resid"
MAP        = os.environ.get("MAP")
RAND_MODEL = os.environ.get("RAND_MODEL", "/dev/shm/pythia1b_rand_s0")
DATASET    = os.environ.get("DATASET", "Skylion007/openwebtext")
# Held out from training by construction: train_saes.sh trains on `train[:3%]`, this reads a
# slice starting at 3%. Evaluating on training data would hide exactly the memorization gap
# that showed up on the Gemma random arm (train 0.379 vs held-out 0.4725).
SPLIT      = os.environ.get("EVAL_SPLIT", "train[3%:4%]")
N_TOKENS   = int(os.environ.get("N_EVAL_TOKENS", 2_000_000))
CTX        = int(os.environ.get("CTX", 2048))
BATCH      = int(os.environ.get("EVAL_BATCH", 8))
N_HEAD     = int(os.environ.get("N_HEAD_LATENTS", 500))   # what delphi's --max_latents 500 sees

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model_path = MODEL_NAME if ARM == "trained" else RAND_MODEL

from datasets import load_dataset                      # noqa: E402
from sparsify import SparseCoder                       # noqa: E402
from transformers import AutoModel, AutoTokenizer      # noqa: E402

if MODE not in ("full", "resid"):
    raise SystemExit("MODE must be 'full' or 'resid'")
if MODE == "resid" and not MAP:
    raise SystemExit("MODE=resid needs MAP=<path to this ARM's P_pythia1b_L*.pt>")

HP = hookpoint(LAYER) if MODE == "resid" else f"layers.{LAYER}"
print(f"[check_saes] ARM={ARM} MODE={MODE} model={model_path} sae={SAE_DIR} "
      f"hookpoint={HP} device={device}")
sae = SparseCoder.load_from_disk(os.path.join(SAE_DIR, HP), device=str(device))
d_sae, k = sae.num_latents, sae.cfg.k
print(f"  d_sae={d_sae} k={k} activation={sae.cfg.activation} "
      f"normalize_decoder={sae.cfg.normalize_decoder}")

# `torch_dtype=`, not `dtype=`: the latter only exists from transformers 4.57, and we pin
# 4.56.1 so that the image's torch 2.4.1 stays usable. Keep this consistent with
# randomize_pythia.py or the two will disagree about precision on different versions.
model = AutoModel.from_pretrained(model_path, torch_dtype=t.bfloat16).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

ds = load_dataset(DATASET, split=SPLIT).shuffle(seed=7)
ids = []
for row in ds:
    ids.extend(tok(row["text"])["input_ids"])
    if len(ids) >= N_TOKENS:
        break
n_seq = len(ids) // CTX
toks = t.tensor(ids[: n_seq * CTX]).view(n_seq, CTX)
print(f"  {n_seq} x {CTX} = {toks.numel()} held-out tokens from {DATASET} [{SPLIT}]")

captured = {}
model.layers[LAYER].register_forward_hook(
    lambda _m, _i, out: captured.__setitem__("h", (out[0] if isinstance(out, tuple) else out).detach())
)

# For MODE=resid, attach the SAME probe training and delphi use, and read the SAE's input off it
# rather than recomputing h - P[tok] here. Recomputing would be a second implementation of the
# thing under test, which is precisely how the conversion bug survived: it verified its own
# restatement instead of the library's actual call.
if MODE == "resid":
    beta, map_meta = load_beta(MAP, model)
    if map_meta.get("arm") != ARM or map_meta.get("layer") != LAYER:
        raise SystemExit(f"map is arm={map_meta.get('arm')} layer={map_meta.get('layer')}, "
                         f"this run is arm={ARM} layer={LAYER}")
    attach(model, beta, LAYER)
    getattr(model.layers[LAYER], PROBE_NAME).register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("r", out.detach()))
    print(f"  map {MAP}: ev_centred={map_meta.get('explained_variance_centred'):.4f} "
          f"fit on {map_meta.get('tokens'):,} tokens")

fired = t.zeros(d_sae, dtype=t.long, device=device)
sse = sst_flat = sst_cent = 0.0
n_tok = 0
l0_sum = 0.0
# Two passes would be needed for an exact centred denominator; instead accumulate sum and
# sum-of-squares and form the variance at the end, which is exact and single-pass.
D = model.config.hidden_size
# `x` is what the SAE actually consumes and reconstructs: h for full, r for resid. FVU must be
# normalised against ITS variance, not h's.
x_sum = t.zeros(D, dtype=t.float64, device=device)
x_sqsum = t.zeros(D, dtype=t.float64, device=device)
h_sum = t.zeros(D, dtype=t.float64, device=device)
h_sqsum = t.zeros(D, dtype=t.float64, device=device)

with t.no_grad():
    for i in range(0, n_seq, BATCH):
        batch = toks[i : i + BATCH].to(device)
        model(batch)
        h = captured["h"].reshape(-1, D).float()
        x = captured["r"].reshape(-1, D).float() if MODE == "resid" else h
        out = sae(x)
        recon = out.sae_out.float()

        sse += float(((x - recon) ** 2).sum())
        x_sum += x.sum(0).double()
        x_sqsum += (x.double() ** 2).sum(0)
        h_sum += h.sum(0).double()
        h_sqsum += (h.double() ** 2).sum(0)
        n_tok += x.shape[0]

        idx = out.latent_indices.reshape(-1)
        fired.index_add_(0, idx, t.ones_like(idx, dtype=t.long))
        l0_sum += float((out.latent_acts.abs() > 1e-5).float().sum())

mu = x_sum / n_tok
var_cent = float((x_sqsum / n_tok - mu ** 2).sum())
sst_cent = var_cent * n_tok
sst_flat = float((x_sqsum).sum())

mu_h = h_sum / n_tok
var_h_cent = float((h_sqsum / n_tok - mu_h ** 2).sum())

alive = int((fired > 0).sum())
alive10 = int((fired >= 10).sum())
head_rate = float(fired[:N_HEAD].sum()) / n_tok

print()
tgt = "r = h - P[tok]" if MODE == "resid" else "h"
print(f"  target        {tgt}")
if MODE == "resid":
    print(f"  Var(r)/Var(h) {var_cent / var_h_cent:.4f} centred   (map explains "
          f"{1 - var_cent / var_h_cent:.1%})")
    print("                -- NOT comparable to a full-mode FVU: r is the harder half.")
print(f"  FVU flat      {sse / sst_flat:.4f}")
print(f"  FVU centred   {sse / sst_cent:.4f}")
print(f"  L0            {l0_sum / n_tok:.2f}   (must equal k={k})")
print(f"  alive         {alive}/{d_sae} = {alive / d_sae:.1%}   (>=10 firings: "
      f"{alive10}/{d_sae} = {alive10 / d_sae:.1%})")
print(f"  latents 0..{N_HEAD-1} firing rate  {head_rate:.3f} per token   "
      f"(uniform would be {k * N_HEAD / d_sae:.3f})")
print()

bad = []
if abs(l0_sum / n_tok - k) > 0.5:
    bad.append(f"L0 {l0_sum / n_tok:.2f} != k {k}")
if alive / d_sae < 0.5:
    bad.append(f"only {alive / d_sae:.1%} of latents alive -- rerun BOTH arms with AUXK=0.03125")
if bad:
    print("FAIL: " + "; ".join(bad))
    sys.exit(1)
print("OK -- cleared for delphi. Record these numbers; both arms must be compared at the same "
      "alive% and L0, or the AUROC difference is confounded by dictionary health.")
