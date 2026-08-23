"""THE GATE for the skip-embed probe. Run it before training and before scoring, on both arms.

verify_randomization.py exists because a partly-randomized model trains an SAE perfectly happily
and produces a full set of plausible AUROC numbers. This file exists for the same reason one level
up: a probe that is attached but wrong -- stale tokens, the other arm's map, a hookpoint sparsify
cannot resolve -- also produces a full set of plausible numbers. Every check here is something
that would otherwise fail silently.

  G0  the ordered-parameter-name hash is UNCHANGED by attaching. This is the one that protects
      randomize_pythia.py: the random draw walks named_parameters() in order, and
      verify_randomization.py pins the hash at 245b6cc67df238e2. P is a buffer precisely so this
      holds; make it an nn.Linear and the same seed silently gives a different model.
  G1  the main forward is BIT-IDENTICAL with the probe attached. The hook returns None, so layer
      L+1 must receive exactly what it received before. Measured, not argued.
  G2  the probe's output equals h - P[tok] recomputed independently, to the last bit.
  G3  sparsify's OWN resolve_widths resolves the hookpoint and reports d_model. Checking with
      sparsify's machinery rather than our own is the direct lesson of the conversion bug, which
      passed a hand-rolled check at 1e-6 while the library computed something else entirely.
  G4  the map belongs to THIS arm (embedding fingerprint) and Var(r)/Var(h) is sane -- r must be
      neither ~0 (the map ate everything) nor ~h (the map does nothing).
  G5  it FAILS LOUDLY: with no tokens stashed, and with misaligned tokens, forward raises.
  G6  optional, needs SAE_DIR: sparsify's SparseCoder selects the SAME latents from the probe's
      output as from an independently recomputed r. This is the "verify on the SELECTED LATENTS,
      via the library's own call" check -- the exact thing convert_sae_to_sparsify.py did not do.

    ARM=trained MAP=/dev/shm/maps/P_pythia1b_L8_trained.pt python -u verify_skipembed.py
    ARM=rand RAND_MODEL=/dev/shm/pythia1b_rand_s0 \
      MAP=/dev/shm/maps/P_pythia1b_L8_rand.pt python -u verify_skipembed.py
    # after training, add the checkpoint to enable G6:
    SAE_DIR=/dev/shm/saes/<run> ARM=trained MAP=... python -u verify_skipembed.py
"""
import hashlib
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import LAYER, MODEL_NAME  # noqa: E402
from skipembed import (PROBE_NAME, attach, build_P, embed_fingerprint,  # noqa: E402
                       hookpoint, load_beta)

ARM        = os.environ.get("ARM", "trained")
RAND_MODEL = os.environ.get("RAND_MODEL", "/dev/shm/pythia1b_rand_s0")
MAP        = os.environ["MAP"]
SAE_DIR    = os.environ.get("SAE_DIR")
N_SEQ      = int(os.environ.get("N_SEQ", 4))
CTX        = int(os.environ.get("CTX", 512))
SEED       = int(os.environ.get("SEED", 0))

from transformers import AutoModel             # noqa: E402

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model_path = MODEL_NAME if ARM == "trained" else RAND_MODEL
print(f"[verify_skipembed] ARM={ARM} model={model_path} layer={LAYER} map={MAP} device={device}\n")

model = AutoModel.from_pretrained(model_path, torch_dtype=t.bfloat16).to(device).eval()
V = model.get_input_embeddings().weight.shape[0]
D = model.config.hidden_size

g = t.Generator().manual_seed(SEED)
toks = t.randint(0, V, (N_SEQ, CTX), generator=g).to(device)

fails: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


def order_hash(m) -> str:
    return hashlib.sha256("\n".join(n for n, _ in m.named_parameters()).encode()).hexdigest()[:16]


# ---- baseline, BEFORE attaching --------------------------------------------------------
with t.no_grad():
    ref_out = model(toks).last_hidden_state.detach().clone()
ref_hash = order_hash(model)
ref_nparam = sum(1 for _ in model.named_parameters())

captured = {}
model.layers[LAYER].register_forward_hook(
    lambda _m, _i, out: captured.__setitem__(
        "h", (out[0] if isinstance(out, tuple) else out).detach().clone())
)
with t.no_grad():
    model(toks)
h_ref = captured["h"]

# ---- G4a: does this map belong to this arm? --------------------------------------------
print("G4  map provenance")
try:
    beta, meta = load_beta(MAP, model)
    check("map/model fingerprint", True,
          f"arm={meta.get('arm')} layer={meta.get('layer')} tokens={meta.get('tokens'):,} "
          f"ev_centred={meta.get('explained_variance_centred'):.4f}")
except SystemExit as e:
    print(f"  [FAIL] map/model fingerprint: {e}")
    sys.exit(1)
check("map arm matches ARM", meta.get("arm") == ARM,
      f"map says {meta.get('arm')!r}, running as {ARM!r}")
check("map layer matches", meta.get("layer") == LAYER,
      f"map layer {meta.get('layer')}, run layer {LAYER}")
check("embed fingerprint", meta.get("embed_fingerprint") == embed_fingerprint(
    model.get_input_embeddings().weight), f"{meta.get('embed_fingerprint')}")

# ---- attach ----------------------------------------------------------------------------
print("\nattaching probe")
probe, _handles = attach(model, beta, LAYER)

# ---- G0: the randomization hash must not move ------------------------------------------
print("\nG0  parameter walk unchanged (protects randomize_pythia.py's seed -> model map)")
check("ordered-name hash", order_hash(model) == ref_hash,
      f"{order_hash(model)} (was {ref_hash})")
check("parameter count", sum(1 for _ in model.named_parameters()) == ref_nparam,
      f"{sum(1 for _ in model.named_parameters())} (was {ref_nparam}) -- P must be a BUFFER")
check("P is not a parameter",
      not any(PROBE_NAME in n for n, _ in model.named_parameters()),
      f"no '{PROBE_NAME}' entry in named_parameters()")
check("probe is a real submodule",
      any(n == hookpoint(LAYER) for n, _ in model.named_modules()),
      f"{hookpoint(LAYER)} resolvable via named_modules()")

# ---- G1: the model still computes what it computed -------------------------------------
print("\nG1  main forward unchanged")
with t.no_grad():
    new_out = model(toks).last_hidden_state
d = float((new_out.float() - ref_out.float()).abs().max())
check("hidden states bit-identical", d == 0.0,
      f"max|delta| = {d:.3e}" + ("" if d == 0.0 else
                                 "  <- the hook is modifying the forward, or the run is "
                                 "nondeterministic; either way do not proceed"))

# ---- G2: the probe computes what we think ----------------------------------------------
print("\nG2  probe output == h - P[tok], recomputed independently")
captured.clear()
probe_out = {}
getattr(model.layers[LAYER], PROBE_NAME).register_forward_hook(
    lambda _m, _i, out: probe_out.__setitem__("r", out.detach().clone()))
with t.no_grad():
    model(toks)
h = captured["h"]
P = build_P(model, beta)
r_manual = (h.float() - P[toks].float()).to(h.dtype)
d2 = float((probe_out["r"].float() - r_manual.float()).abs().max())
check("probe == manual", d2 == 0.0, f"max|delta| = {d2:.3e}")
check("h unchanged by the probe", float((h.float() - h_ref.float()).abs().max()) == 0.0,
      "layer L output identical to the pre-attach run")

# ---- G3: sparsify's own resolver -------------------------------------------------------
print("\nG3  sparsify resolves the hookpoint (its machinery, not ours)")
try:
    from sparsify.utils import resolve_widths
    w = resolve_widths(model, [hookpoint(LAYER)])
    check("resolve_widths", w.get(hookpoint(LAYER)) == D, f"{w} (d_model={D})")
except Exception as e:                                    # noqa: BLE001
    check("resolve_widths", False, f"{type(e).__name__}: {e}")

# ---- G4b: is the residual a sensible object? -------------------------------------------
print("\nG4  residual scale")
# Centred per dimension, over tokens -- the same convention check_saes.py reports, and the one
# that is not inflated by the mean vector's cross-dimension spread on the trained arm.
hf = h.float().reshape(-1, D)
rf = r_manual.float().reshape(-1, D)
var_h = float((hf - hf.mean(0, keepdim=True)).pow(2).mean())
var_r = float((rf - rf.mean(0, keepdim=True)).pow(2).mean())
ratio = var_r / var_h if var_h else float("nan")
check("Var(r)/Var(h) in (0.05, 0.95)", 0.05 < ratio < 0.95,
      f"{ratio:.4f}  (map explains {1 - ratio:.1%} of centred variance on random tokens)")

# ---- G5: loud failure ------------------------------------------------------------------
print("\nG5  fails loudly rather than falling back to h")
probe.tok = None
try:
    probe(h)
    check("raises with no tokens", False, "returned a value instead of raising")
except RuntimeError as e:
    check("raises with no tokens", True, str(e).split(".")[0])
probe.tok = toks[:, : CTX // 2]
try:
    probe(h)
    check("raises on misalignment", False, "returned a value instead of raising")
except RuntimeError as e:
    check("raises on misalignment", True, str(e).split(".")[0])
probe.set_tok(toks)

# ---- G6: the library's own encode, on the SELECTED latents ------------------------------
if SAE_DIR:
    print("\nG6  SparseCoder selects the same latents from the probe output as from manual r")
    from sparsify import SparseCoder
    sae_path = os.path.join(SAE_DIR, hookpoint(LAYER))
    if not os.path.isdir(sae_path):
        check("checkpoint present", False,
              f"{sae_path} missing -- was the SAE trained on {hookpoint(LAYER)}?")
    else:
        sc = SparseCoder.load_from_disk(sae_path, device=str(device))
        with t.no_grad():
            a = sc.encode(probe_out["r"].reshape(-1, D))
            b = sc.encode(r_manual.reshape(-1, D))
        ia = a.top_indices.sort(dim=-1).values
        ib = b.top_indices.sort(dim=-1).values
        agree = float((ia == ib).all(dim=-1).float().mean())
        check("selected latents identical", agree == 1.0,
              f"{agree:.6f} of positions agree on all k indices")
        l0 = float((a.top_acts.abs() > 1e-5).float().sum(-1).mean())
        print(f"       L0 = {l0:.2f} (k = {sc.cfg.k}), d_sae = {sc.num_latents}")
else:
    print("\nG6  skipped (set SAE_DIR after training to enable)")

print()
if fails:
    print("GATE FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED -- the probe is inert on the model, correct on the residual, and "
      "visible to sparsify.\nRecord Var(r)/Var(h) and the map's ev_centred alongside the run.")
