"""THE GATE: prove the randomized Pythia is what Heap et al. define, before any SAE touches it.

Nothing downstream errors if this is wrong. A model that is 95% randomized, or that kept a
trained unembedding, or whose activations quietly exploded, trains an SAE perfectly happily
and produces a complete set of plausible AUROC numbers. That failure mode has already cost
this project one full re-run cycle (see ../DELPHI_SETUP.md), so the randomization is gated
here rather than trusted.

Eight checks, each a hard gate:

  G0  parameter inventory      every tensor classified; count == 12*n_layers + 4; the ordered
                               name list is hashed and printed, because the draw walks
                               named_parameters() in order -- if a transformers upgrade
                               reorders them, the same seed gives a DIFFERENT model and arms
                               trained at different times stop being comparable.
  G1  embeddings included      embed_in AND embed_out both re-drawn. Pythia does not tie them
                               (Gemma does), so this is the check that would have caught a
                               copy-paste of the Gemma recipe leaving a trained unembedding
                               inside a "fully random" model.
  G2  moments matched          per tensor, |new_mean - old_mean| and new_std/old_std inside
                               the 6-sigma sampling band for that tensor's size. This is the
                               actual content of "mean and variance equal to the values for
                               each of the original, trained weight matrices".
  G3  decorrelated             |corr(old, new)| at the noise floor for every tensor -- i.e.
                               the values really were replaced, not perturbed.
  G4  determinism              same seed -> bit-identical weights; different seed -> different.
  G5  disk round-trip          the SAVED checkpoint is bit-identical to what we verified.
                               sparsify and delphi read the file, not this process's memory.
  G6  hookpoint identity       the tensor sparsify/delphi hook as `layers.N` is exactly
                               hidden_states[N+1]. Cheap, and it pins the one thing that
                               silently mismatches across libraries.
  G7  activation health        layer-N residual is finite and sanely scaled; per-dim variance
                               spread reported (a random net should have NO massive-activation
                               dims); next-token loss MEASURED on both arms.

  On G7's loss: do NOT expect a random transformer to sit at uniform-chance ln(V) = 10.83.
  It is not a uniform predictor -- it is sharply peaked and confidently wrong, which scores
  WORSE than chance. Gemma-2 measured 22.6 against its ln(256000)=12.45. Pythia has no logit
  softcapping, so its value is unbounded in principle; this script reports it rather than
  asserting a range, and only fails on non-finite.

Run (CPU is fine -- pythia-1b is 2 GB; ~5 min on a Mac, no GPU needed):

    VARIANT=rand_all INIT_SEED=0 OUT_DIR=/tmp/pythia1b_rand_s0 python -u verify_randomization.py

Env: VARIANT, INIT_SEED, LAYER, OUT_DIR, N_CHECK_TOKENS, DATASET, SKIP_TRAINED_BASELINE.
Exit code 0 == every gate passed == cleared for training.
"""
import hashlib
import math
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import (  # noqa: E402
    EMBED_NAMES, LAYER, MODEL_NAME, STORE_DTYPE, build_model, load_trained, randomize_,
)

OUT_DIR    = os.environ.get("OUT_DIR", "")
VARIANT    = os.environ.get("VARIANT", "rand_all")
INIT_SEED  = int(os.environ.get("INIT_SEED", 0))
N_TOKENS   = int(os.environ.get("N_CHECK_TOKENS", 4096))
SEQ_LEN    = int(os.environ.get("CHECK_SEQ_LEN", 512))
# CORPUS -- a documented deviation from the paper, forced and then chosen.
# Heap et al. train on "100M tokens from the RedPajama dataset". That dataset no longer
# exists: `togethercomputer/RedPajama-Data-1T-Sample` 404s even with a token (so does
# cerebras/SlimPajama-627B), and the surviving parent repo is a loading SCRIPT, which
# datasets>=3 refuses to execute. Only community re-uploads remain.
# Given that the paper's own corpus was already off-distribution for Pythia (which was
# pretrained on the Pile), and that every Gemma cell in this project used openwebtext, we
# use openwebtext -- which makes the Pythia arms directly comparable to the existing 2x2.
# Both arms see identical text either way, so this shifts absolute AUROC, not the
# trained-vs-random contrast the paper's claim rests on.
DATASET    = os.environ.get("DATASET", "Skylion007/openwebtext")
SKIP_BASE  = os.environ.get("SKIP_TRAINED_BASELINE", "0") == "1"

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def fingerprint(model):
    """{param name: sha256 of its raw bytes}. Bit-level, so it catches a change of any size."""
    fp = {}
    for name, p in model.named_parameters():
        fp[name] = hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return fp


def classify(name):
    """Bucket a GPT-NeoX parameter name so G0 can assert the inventory is complete."""
    tail = name.split("layers.")[-1]
    tail = tail.split(".", 1)[1] if tail[0].isdigit() else name
    for key in ("embed_in", "embed_out", "final_layer_norm", "input_layernorm",
                "post_attention_layernorm", "attention.query_key_value", "attention.dense",
                "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"):
        if key in name:
            return key + ("" if key.startswith("embed") else (".bias" if name.endswith("bias") else ".weight"))
    return f"UNCLASSIFIED:{tail}"


# =======================================================================================
print(f"\n=== verify_randomization: {MODEL_NAME} VARIANT={VARIANT} INIT_SEED={INIT_SEED} "
      f"LAYER={LAYER} device={DEVICE} ===\n")

if VARIANT == "trained":
    sys.exit("VARIANT=trained has no randomization to verify.")

report = []
model = build_model(variant=VARIANT, seed=INIT_SEED, report=report)
cfg = model.config
n_layers = cfg.num_hidden_layers
all_names = [n for n, _ in model.named_parameters()]

# --- G0 inventory -----------------------------------------------------------------------
print("\nG0  parameter inventory")
order_hash = hashlib.sha256("\n".join(all_names).encode()).hexdigest()[:16]
buckets = {}
for n in all_names:
    buckets[classify(n)] = buckets.get(classify(n), 0) + 1
for k in sorted(buckets):
    print(f"       {buckets[k]:>4} x {k}")
expected = 12 * n_layers + 4
gate("no unclassified parameters", not any(k.startswith("UNCLASSIFIED") for k in buckets),
     ",".join(k for k in buckets if k.startswith("UNCLASSIFIED")) or "all named")
gate(f"tensor count == 12*{n_layers}+4 == {expected}", len(all_names) == expected,
     f"got {len(all_names)}")
gate("every tensor re-drawn", len(report) == expected, f"redrawn {len(report)} of {len(all_names)}")
print(f"       ordered-name hash {order_hash}  (transformers reorder detector -- record this)")

# --- G1 embeddings ----------------------------------------------------------------------
print("\nG1  embeddings included (pythia does NOT tie embed_in/embed_out)")
gate("tie_word_embeddings is False", cfg.tie_word_embeddings is False, str(cfg.tie_word_embeddings))
redrawn_names = {r["name"] for r in report}
for e in EMBED_NAMES:
    hit = [n for n in redrawn_names if e in n]
    gate(f"{e} re-drawn", len(hit) == 1, hit[0] if hit else "MISSING -- trained weights survive!")

# --- G2 moments -------------------------------------------------------------------------
print("\nG2  moments matched per tensor (6-sigma sampling band)")
worst_mu = worst_sd = (0.0, "")
n_bad = 0
for r in report:
    n = r["numel"]
    if r["old_std"] == 0:
        continue
    z_mu = abs(r["new_mean"] - r["old_mean"]) / (r["old_std"] / math.sqrt(n))
    z_sd = abs(r["new_std"] / r["old_std"] - 1.0) * math.sqrt(2 * n)
    worst_mu = max(worst_mu, (z_mu, r["name"]))
    worst_sd = max(worst_sd, (z_sd, r["name"]))
    if z_mu > 6 or z_sd > 6:
        n_bad += 1
        print(f"       OUT OF BAND {r['name']}: z_mean={z_mu:.2f} z_std={z_sd:.2f}")
gate("all tensors within 6 sigma on mean and std", n_bad == 0, f"{n_bad} out of band")
print(f"       worst z_mean {worst_mu[0]:.2f} ({worst_mu[1]})")
print(f"       worst z_std  {worst_sd[0]:.2f} ({worst_sd[1]})")

# --- G3 decorrelation -------------------------------------------------------------------
print("\nG3  old-vs-new decorrelated (values replaced, not perturbed)")
corrs = [(abs(r["corr"]), r["name"]) for r in report if not math.isnan(r["corr"])]
worst_corr = max(corrs) if corrs else (0.0, "")
gate("|corr| < 0.08 for every tensor", worst_corr[0] < 0.08,
     f"worst |corr| {worst_corr[0]:.4f} ({worst_corr[1]})")

# LayerNorm gains: report, do not gate. Moment matching CAN produce negative gains and that
# is faithful to "all the model parameters"; we just want to know if it happened.
ln_neg = [(r["name"], r["old_mean"], r["old_std"])
          for r in report if "layernorm" in r["name"].lower() and r["name"].endswith("weight")]
if ln_neg:
    # Expected share of re-drawn gains that land negative, = Phi(-mu/sd) per tensor. Reported,
    # never gated: a sign flip on a LayerNorm gain is what "all the model parameters" implies,
    # but it is the class most likely to surprise us, so the number goes in the log.
    snrs = sorted(mu / sd for _, mu, sd in ln_neg if sd > 0)
    neg = [0.5 * math.erfc(z / math.sqrt(2)) for z in snrs]
    print(f"       LN gains: {len(ln_neg)} tensors, median mean/std {snrs[len(snrs)//2]:+.2f}, "
          f"expected negative-gain share {sum(neg)/len(neg):.1%} (max {max(neg):.1%}) -- "
          f"faithful to 'all the model parameters', not a fault")

fp_a = fingerprint(model)
del model

# --- G4 determinism ---------------------------------------------------------------------
print("\nG4  determinism")
m_b = build_model(variant=VARIANT, seed=INIT_SEED)
fp_b = fingerprint(m_b)
del m_b
gate("same seed -> bit-identical", fp_a == fp_b,
     "identical" if fp_a == fp_b else f"{sum(1 for k in fp_a if fp_a[k] != fp_b[k])} tensors differ")

m_c = build_model(variant=VARIANT, seed=INIT_SEED + 1)
fp_c = fingerprint(m_c)
del m_c
n_same = sum(1 for k in fp_a if fp_a[k] == fp_c[k])
gate("different seed -> different weights", n_same == 0, f"{n_same} tensors identical across seeds")

# --- G5 disk round-trip -----------------------------------------------------------------
print("\nG5  disk round-trip (sparsify/delphi read the FILE, not this process)")
if not OUT_DIR:
    gate("OUT_DIR provided", False, "set OUT_DIR=<dir> so the saved checkpoint can be verified")
    reloaded = None
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not os.path.exists(os.path.join(OUT_DIR, "config.json")):
        print(f"       {OUT_DIR} has no checkpoint -- writing one now")
        os.makedirs(OUT_DIR, exist_ok=True)
        m = build_model(variant=VARIANT, seed=INIT_SEED)
        m.save_pretrained(OUT_DIR)
        AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(OUT_DIR)
        del m
    reloaded = AutoModelForCausalLM.from_pretrained(OUT_DIR, torch_dtype=STORE_DTYPE)
    fp_d = fingerprint(reloaded)
    diff = [k for k in fp_a if fp_a.get(k) != fp_d.get(k)]
    gate("saved checkpoint == verified weights", not diff,
         "bit-identical" if not diff else f"{len(diff)} tensors differ, e.g. {diff[:3]}")

# --- G6 / G7 forward-pass checks --------------------------------------------------------
print("\nG6/G7  hookpoint identity and activation health")


def get_tokens():
    """A fixed slab of tokens from the corpus the SAEs will be trained on.

    Shuffled, not taken head-first: a streamed corpus hands back shard 0 in file order, which
    for most Hub corpora is a single homogeneous source. Activation scale and next-token loss
    both move with genre, so an unshuffled head would make the two arms' numbers depend on
    whatever happens to sit at the front of the dataset.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=INIT_SEED, buffer_size=1000)
    ids, need = [], N_TOKENS
    for row in ds:
        ids.extend(tok(row["text"])["input_ids"])
        if len(ids) >= need:
            break
    n_seq = max(1, len(ids) // SEQ_LEN)
    return t.tensor(ids[: n_seq * SEQ_LEN]).view(n_seq, SEQ_LEN)


@t.no_grad()
def health(model, toks, label):
    """Layer-LAYER residual stats + next-token loss, and the G6 hook identity check."""
    model = model.to(t.float32).to(DEVICE).eval()
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.gpt_neox.layers[LAYER].register_forward_hook(hook)
    out = model(toks.to(DEVICE), labels=toks.to(DEVICE), output_hidden_states=True)
    handle.remove()

    hs = out.hidden_states[LAYER + 1].detach()
    same = t.equal(captured["h"], hs)
    h = hs.float().reshape(-1, hs.shape[-1])
    loss = float(out.loss)
    stats = dict(
        label=label, hook_identical=same, loss=loss,
        finite=bool(t.isfinite(h).all()),
        norm=float(h.norm(dim=-1).mean()), var=float(h.var()),
        max_dim_var=float(h.var(0).max()), mean_dim_var=float(h.var(0).mean()),
    )
    model.to("cpu")
    return stats


try:
    toks = get_tokens()
    print(f"       {toks.shape[0]} x {toks.shape[1]} = {toks.numel()} tokens from {DATASET}")
except Exception as e:  # noqa: BLE001
    gate("corpus available", False, f"{type(e).__name__}: {e}")
    toks = None

rows = []
if toks is not None:
    if reloaded is None:
        reloaded = build_model(variant=VARIANT, seed=INIT_SEED)
    rows.append(health(reloaded, toks, f"{VARIANT}-s{INIT_SEED}"))
    del reloaded
    if not SKIP_BASE:
        rows.append(health(load_trained(), toks, "trained"))

    print(f"\n       {'arm':<18}{'|h| mean':>11}{'var':>11}{'maxdimvar/mean':>17}{'loss':>10}")
    for r in rows:
        print(f"       {r['label']:<18}{r['norm']:>11.2f}{r['var']:>11.4f}"
              f"{r['max_dim_var']/r['mean_dim_var']:>17.2f}{r['loss']:>10.3f}")

    for r in rows:
        gate(f"G6 layers.{LAYER} output == hidden_states[{LAYER+1}] ({r['label']})", r["hook_identical"])
        gate(f"G7 activations finite ({r['label']})", r["finite"])
        gate(f"G7 loss finite ({r['label']})", math.isfinite(r["loss"]), f"{r['loss']:.3f}")
        gate(f"G7 |h| non-degenerate ({r['label']})", 0.1 < r["norm"] < 1e5, f"{r['norm']:.2f}")
    print(f"\n       ln(vocab={cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}. A random net scoring "
          f"ABOVE this is CORRECT --\n       it is confidently wrong, not uniform. Record the value; "
          f"do not tune toward chance.")

# =======================================================================================
print("\n" + "=" * 78)
if FAILURES:
    print(f"FAILED {len(FAILURES)} gate(s): {FAILURES}")
    print("DO NOT TRAIN ON THIS CHECKPOINT.")
    sys.exit(1)
print(f"ALL GATES PASSED -- {VARIANT} seed {INIT_SEED} is Heap et al.'s "
      f"'re-randomized incl. embeddings' for {MODEL_NAME}.")
print(f"Record: ordered-name hash {order_hash}"
      + (f" | |h_{LAYER}| {rows[0]['norm']:.2f} | loss {rows[0]['loss']:.3f}" if rows else ""))
print("=" * 78)
