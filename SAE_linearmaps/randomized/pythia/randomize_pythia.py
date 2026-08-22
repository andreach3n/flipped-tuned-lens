"""Moment-matched re-randomization of Pythia, and materialization as an HF checkpoint.

This is the Pythia arm of the Heap et al. replication (arXiv:2501.17727 / OpenReview
USyGD0eUod, "Automated Interpretability Metrics Do Not Distinguish Trained and Random
Transformers"). Their definition, quoted from S3:

    Re-randomized incl. embeddings: "All the model parameters, including the embeddings,
    are re-initialized by sampling Gaussian noise with mean and variance equal to the
    values for each of the original, trained weight matrices."

    Re-randomized excl. embeddings: "As above, except the embedding and unembedding
    weight matrices are not re-initialized."

So: per-tensor N(mean(W), std(W)^2), where the moments come from THAT tensor's own trained
values. That is why the pretrained weights are still downloaded on a random arm -- we need
their moments. A default init would change the activation scale across 8+ layers and
confound everything downstream.

WHAT IS DIFFERENT FROM THE GEMMA ARM (../../activations.py), and why this is a fresh module
rather than a parameter of that one:
  * Pythia is GPT-NeoX, and `tie_word_embeddings` is FALSE. Gemma-2 ties lm_head to
    embed_tokens, so skipping one name covered both roles. Here `embed_in` and `embed_out`
    are two independent tensors and BOTH must be re-drawn for "incl. embeddings" -- a silent
    single-name skip would leave a trained unembedding in a "random" model.
  * LayerNorm here has a BIAS as well as a gain; Gemma's RMSNorm has neither a bias nor a
    post-feedforward-norm equivalent in the same count. The tensor inventory is therefore
    12*n_layers + 4, not gemma's 11*n_layers + 2.
  * No TransformerLens anywhere. sparsify and delphi are both plain-HF, so keeping this
    pipeline plain-HF removes the TL-vs-HF activation divergence (cosine 0.9997, ~8% of
    firings flipped at the top-k boundary) that DELPHI_SETUP.md had to legislate around.

DETERMINISM. The draw is driven by a single CPU torch.Generator walked over
`named_parameters()` in order, with the model on CPU. Same seed + same transformers version
=> bit-identical weights on any machine. The ordering is load-bearing, so
verify_randomization.py hashes the ordered name list and prints it; if a transformers
upgrade reorders parameters, that hash moves and the arms are no longer comparable to
previously trained ones.

Usage -- materialize the random arm as a real checkpoint (sparsify and delphi both take a
model by path/name, so it has to exist on disk):

    VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/pythia1b_rand_s0 \
      PUSH_REPO=andreayhchen/pythia-1b-rand-all-s0 python -u randomize_pythia.py

Omit PUSH_REPO to write locally only. Run verify_randomization.py BEFORE training on it.
"""
import os
import sys

import torch as t

MODEL_NAME = os.environ.get("MODEL_NAME", "EleutherAI/pythia-1b")
LAYER      = int(os.environ.get("LAYER", 8))          # pythia-1b has 16 blocks; the paper
                                                       # trains every 2nd layer, 8 is the middle
HOOKPOINT  = f"layers.{LAYER}"                         # what sparsify and delphi both call it

VARIANT   = os.environ.get("VARIANT", "trained").strip() or "trained"
INIT_SEED = int(os.environ.get("INIT_SEED", 0))
_VARIANTS = ("trained", "rand_all", "rand_nonembed")
if VARIANT not in _VARIANTS:
    raise ValueError(f"VARIANT={VARIANT!r} not in {_VARIANTS}")

# Pythia does NOT tie these -- both names must match for "excl. embeddings" to mean what the
# paper says ("the embedding and unembedding weight matrices").
EMBED_NAMES = ("embed_in", "embed_out")

# Storage dtype. The Hub ships pythia-1b as float16; we keep the random arm in float16 too so
# both arms hit sparsify/delphi in the SAME format and neither gets a precision advantage.
# Moments and the draw itself are computed in float32 -- mean/std over a float16 tensor is
# badly imprecise, and that error would propagate into the matched variance.
STORE_DTYPE = t.float16


def load_trained(dtype=STORE_DTYPE):
    """The trained model, on CPU, in the Hub's own dtype."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)


@t.no_grad()
def randomize_(model, keep_embeddings=False, seed=0, report=None):
    """In-place moment-matched re-initialization of every parameter tensor.

    For each tensor: draw randn * std(W) + mean(W) from the TRAINED tensor's own moments.
    LayerNorm gains and biases are re-sampled too -- the paper says "all the model
    parameters", and a gain whose trained std is an appreciable fraction of its mean will
    produce some NEGATIVE gains. That is faithful, not a bug, but it is the parameter class
    most likely to surprise us, so verify_randomization.py reports the negative fraction.

    `report`, if given a list, is appended one dict per tensor recording the old and new
    moments and the old-vs-new correlation on a fixed subsample. Collecting it here rather
    than by diffing two full models keeps peak memory at one model instead of two.
    """
    g = t.Generator().manual_seed(seed)
    kept, redrawn = [], 0
    for name, p in model.named_parameters():
        if keep_embeddings and any(s in name for s in EMBED_NAMES):
            kept.append(name)
            continue
        w = p.detach().float()
        mu, sd = w.mean().item(), w.std().item()
        new = t.randn(p.shape, generator=g, dtype=t.float32) * sd + mu
        if report is not None:
            report.append(_row(name, w, new, mu, sd))
        p.copy_(new.to(p.dtype))
        redrawn += 1
    print(f"  re-initialized {redrawn} parameter tensors (seed={seed})"
          + (f"; kept {kept}" if kept else "; embeddings included"))
    return redrawn, kept


def _row(name, old, new, mu, sd, n_sub=4096):
    """Per-tensor record: moments before/after plus a decorrelation check.

    The correlation is taken on a fixed stride-subsample rather than the whole tensor so the
    cost is O(n_sub) on a 103M-element embedding table. A fresh Gaussian draw is independent
    of the trained values, so |corr| should sit at the sampling noise floor ~1/sqrt(n_sub)
    ~ 0.016; anything appreciably larger means a tensor was not actually redrawn.
    """
    flat_o, flat_n = old.flatten(), new.flatten()
    step = max(1, flat_o.numel() // n_sub)
    a, b = flat_o[::step][:n_sub], flat_n[::step][:n_sub]
    if a.numel() > 2 and a.std() > 0 and b.std() > 0:
        corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std(unbiased=False) * b.std(unbiased=False)))
    else:
        corr = float("nan")
    return dict(name=name, numel=old.numel(), shape=tuple(old.shape),
                old_mean=mu, old_std=sd,
                new_mean=float(new.mean()), new_std=float(new.std()), corr=corr)


def build_model(variant=None, seed=None, report=None):
    """Trained or randomized model, on CPU. The single entry point every other script uses."""
    variant = VARIANT if variant is None else variant
    seed = INIT_SEED if seed is None else seed
    model = load_trained()
    if variant != "trained":
        print(f"[randomize_pythia] VARIANT={variant} INIT_SEED={seed}: randomizing {MODEL_NAME}")
        randomize_(model, keep_embeddings=(variant == "rand_nonembed"), seed=seed, report=report)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# CLI: materialize the arm as a checkpoint on disk (and optionally on the Hub)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR   = os.environ.get("OUT_DIR", f"/workspace/pythia1b_{VARIANT}_s{INIT_SEED}")
    PUSH_REPO = os.environ.get("PUSH_REPO", "")

    if VARIANT == "trained":
        sys.exit(f"VARIANT=trained has nothing to materialize -- point sparsify/delphi at {MODEL_NAME}.")

    from transformers import AutoTokenizer
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()
    model.save_pretrained(OUT_DIR)
    # delphi resolves the tokenizer from the model directory, so it has to be there. Pythia's
    # tokenizer is identical across arms -- we never touch it, only the weights.
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(OUT_DIR)
    print(f"  saved model + tokenizer to {OUT_DIR}")
    print(f"  NEXT: run verify_randomization.py against this checkpoint BEFORE training on it.")

    if PUSH_REPO:
        from huggingface_hub import HfApi, create_repo
        create_repo(PUSH_REPO, repo_type="model", exist_ok=True)
        HfApi().upload_folder(folder_path=OUT_DIR, repo_id=PUSH_REPO, repo_type="model")
        print(f"  pushed -> {PUSH_REPO}")
