"""Prove delphi's hookpoint captures the SAME tensor our SAEs were trained on.

Our SAEs were trained on TransformerLens `blocks.{LAYER}.hook_resid_post`. Delphi resolves a
hookpoint against HF `model.named_modules()` and its hook takes the module OUTPUT (`output[0]`
when it is a tuple, which decoder layers return). For HF gemma-2 that should make
`model.layers.{LAYER}` the post-block residual -- the same tensor. SHOULD.

If it is off by a block, or grabs a pre-norm tensor, delphi produces a complete set of plausible
numbers and nothing raises. This script is the cheap way to never wonder: run one text through
both stacks and compare the tensors elementwise.

    LAYER=13 python -u randomized/check_hookpoint.py            # trained arm
    VARIANT=rand_all INIT_SEED=0 python -u randomized/check_hookpoint.py

Prints the matching module names too, so you can see what to pass to --hookpoints on this model.
"""
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import MODEL_NAME, LAYER, HOOK, SEQ_LEN, VARIANT, INIT_SEED, load_model

TEXT = os.environ.get("TEXT", "The capital of France is Paris, and the capital of Japan is Tokyo.")
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# ---- our side: TransformerLens, exactly as activations.py builds it -------------------------
tl = load_model(device)
tokens = tl.to_tokens(TEXT)[:, :SEQ_LEN].to(device)
with t.no_grad():
    _, cache = tl.run_with_cache(tokens, names_filter=[HOOK], stop_at_layer=LAYER + 1)
ours = cache[HOOK].squeeze(0).float().cpu()
print(f"[check] {HOOK}: {tuple(ours.shape)}")

# The HF model TL is wrapping. Reaching through TL guarantees we compare the SAME weights --
# reloading from the Hub would silently compare the trained arm against itself on a random run.
hf = getattr(tl, "hf_model", None)
if hf is None:
    from transformers import AutoModelForCausalLM
    src = os.environ.get("HF_MODEL_PATH", MODEL_NAME)
    print(f"  (TL kept no hf_model; loading {src} -- make sure this is the RIGHT ARM)")
    hf = AutoModelForCausalLM.from_pretrained(src, torch_dtype=t.bfloat16)
hf = hf.to(device).eval()

names = [n for n, _ in hf.named_modules()]
cands = [n for n in names if n.endswith(f"layers.{LAYER}")]
print(f"  candidate hookpoints on this model: {cands}")
if not cands:
    sys.exit("no module named *.layers.{LAYER} -- inspect named_modules() by hand")

# ---- their side: delphi's rule, replicated exactly ------------------------------------------
grabbed = {}


def hook(_mod, _inp, output):
    grabbed["x"] = output[0] if isinstance(output, tuple) else output


for name, mod in hf.named_modules():
    if name == cands[0]:
        h = mod.register_forward_hook(hook)
        break
with t.no_grad():
    hf(tokens)
h.remove()
theirs = grabbed["x"].squeeze(0).float().cpu()
print(f"  {cands[0]} output[0]: {tuple(theirs.shape)}")

# ---- compare -------------------------------------------------------------------------------
if ours.shape != theirs.shape:
    sys.exit(f"SHAPE MISMATCH {ours.shape} vs {theirs.shape} -- wrong hookpoint")
den = ours.abs().max().item()
rel = (ours - theirs).abs().max().item() / max(den, 1e-9)
cos = t.nn.functional.cosine_similarity(ours.flatten(), theirs.flatten(), dim=0).item()
print(f"  max rel diff {rel:.2e} | cosine {cos:.6f} | VARIANT={VARIANT} s{INIT_SEED}")
# bf16 forward, two different execution paths -> expect ~1e-2 relative, not 1e-7. Cosine is the
# discriminating number: a wrong-block tensor lands far below 0.999.
print("  VERDICT:", "SAME TENSOR" if cos > 0.999 else "DIFFERENT -- do not run delphi on this hookpoint")
