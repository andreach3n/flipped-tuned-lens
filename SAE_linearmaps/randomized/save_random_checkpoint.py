"""Materialize the randomized gemma as a REAL HF checkpoint, then push it.

Why this exists: activations.py randomizes gemma IN MEMORY at load time, which is fine for our
own scripts but useless to any tool that takes a model by name -- delphi in particular
(`python -m delphi <model> <sparse_model> ...`). To run their pipeline on our random arm the arm
has to exist as a checkpoint on disk / the Hub.

THE ONE THING THAT MATTERS: this must produce EXACTLY the model our SAEs were trained on, or the
comparison is meaningless and nothing will error. So it does NOT reimplement the randomization --
it imports `_randomize_` from activations.py and calls it with the same (keep_embeddings, seed).
Determinism holds because that function drives a single torch.Generator over
hf_model.named_parameters() in order, and we hand it the same freshly-loaded model.

It then VERIFIES against the numbers recorded for VARIANT=rand_all INIT_SEED=0:
    288 tensors re-initialized;  h_13: |h| mean 317.00, var 43.6235, max|dim var| 64.3964
A mismatch means the saved checkpoint is NOT the arm the SAEs were trained on -- stop and debug.

    VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/rand_ckpt \
      PUSH_REPO=andreayhchen/gemma2-2b-rand-all-s0 python -u randomized/save_random_checkpoint.py

Omit PUSH_REPO to write locally only. Env: VARIANT, INIT_SEED, OUT_DIR, PUSH_REPO, LAYER.
"""
import os
import sys

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import MODEL_NAME, LAYER, HOOK, VARIANT, INIT_SEED, _randomize_, take_sample

OUT_DIR   = os.environ.get("OUT_DIR", "/workspace/rand_ckpt")
PUSH_REPO = os.environ.get("PUSH_REPO", "")
os.makedirs(OUT_DIR, exist_ok=True)

if VARIANT == "trained":
    sys.exit("VARIANT=trained has nothing to materialize -- point delphi at google/gemma-2-2b.")

print(f"[save_random_checkpoint] VARIANT={VARIANT} INIT_SEED={INIT_SEED} -> {OUT_DIR}")
hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.bfloat16)
_randomize_(hf, keep_embeddings=(VARIANT == "rand_nonembed"), seed=INIT_SEED)

hf.save_pretrained(OUT_DIR)
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(OUT_DIR)   # delphi needs it alongside
print(f"  saved model + tokenizer to {OUT_DIR}")

# ---- verify the SAVED checkpoint reproduces the arm the SAEs were trained on ----------------
# Round-trip through disk on purpose: this checks serialization too, not just the randomization.
del hf
from transformer_lens import HookedTransformer
device = t.device("cuda" if t.cuda.is_available() else "cpu")
reloaded = AutoModelForCausalLM.from_pretrained(OUT_DIR, torch_dtype=t.bfloat16)
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME, hf_model=reloaded, dtype=t.bfloat16).to(device)
model.eval()

h, _ = take_sample(model, device, n_tokens=20_000, seed=INIT_SEED)
h = h.float()
mean_norm, var, maxdim = h.norm(dim=-1).mean().item(), h.var().item(), h.var(0).max().item()
print(f"  h_{LAYER} check: |h| mean {mean_norm:.2f} | var {var:.4f} | max|dim var| {maxdim:.4f}")

if VARIANT == "rand_all" and INIT_SEED == 0 and LAYER == 13:
    REF = (317.00, 43.6235, 64.3964)      # recorded for this arm on 2026-08-03
    ok = (abs(mean_norm - REF[0]) < 0.5 and abs(var - REF[1]) < 0.05 and abs(maxdim - REF[2]) < 0.5)
    print(f"  vs reference {REF}: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("saved checkpoint does NOT reproduce the trained-against arm -- do not use it.")

if PUSH_REPO:
    from huggingface_hub import HfApi, create_repo
    create_repo(PUSH_REPO, repo_type="model", exist_ok=True, private=True)
    HfApi().upload_folder(folder_path=OUT_DIR, repo_id=PUSH_REPO, repo_type="model")
    print(f"  pushed -> {PUSH_REPO}  (delphi: python -m delphi {PUSH_REPO} <sae> "
          f"--hookpoints {HOOK.replace('blocks.', 'model.layers.').replace('.hook_resid_post', '')})")
