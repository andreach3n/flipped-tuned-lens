"""Skip-embed as a HOOKPOINT, so neither sparsify nor delphi has to be modified.

THE PROBLEM THIS SOLVES. A skip-embed (`resid`) SAE encodes r = h - P[tok], where P is a frozen
linear map from the input embedding to the layer-L activation. Both libraries assume the SAE's
encoder input IS the hookpoint activation: sparsify autoencodes whatever the hooked module
outputs, and delphi hooks `layers.L`, hands that tensor to SparseCoder.encode(), and caches what
fires. Feed either of them raw h and the encoder sees the wrong input -- no error, just a full set
of plausible, wrong AUROC numbers. That is the same failure shape as the conversion bug in
../DELPHI_SETUP.md, which invalidated every delphi number this project produced before 2026-08-19.

The Gemma answer was write_delphi_cache.py: bypass the library and write delphi's latent cache by
hand, then validate it to Jaccard 1.0. It worked, but the bugs that cost the project months lived
in exactly that layer -- format, indexing, sharding.

THE ANSWER HERE. Make r a hookpoint. Attach a child module to layer L whose OUTPUT is r, and
invoke it from a forward hook that returns None:

    layers.8            -> h        (unchanged, still what layer 9 receives)
    layers.8.skipembed  -> h - P[tok]

Then `--hookpoints layers.8.skipembed` trains resid in stock sparsify, and delphi pointed at the
same name feeds its encoder exactly what that encoder was trained on. Neither library is patched;
delphi still does its own caching, sharding, filter_bos and scoring. There is no hand-written
cache to get wrong.

THREE INVARIANTS, each of which verify_skipembed.py checks:

  1. THE MAIN FORWARD IS UNCHANGED. The hook returns None, so transformers keeps layer L's real
     output. Activations collected with the probe attached come from the actual model, not a
     mutilated one. This is checkable exactly (bit-identical hidden states), not by argument.

  2. P IS A BUFFER, NEVER A PARAMETER. `named_parameters()` is what randomize_pythia.py walks to
     draw the random weights, and verify_randomization.py pins its ordered-name hash at
     245b6cc67df238e2. A buffer leaves that walk untouched; an nn.Linear would insert two entries
     and silently change which model a given seed produces. If you refactor this, keep it a buffer.

  3. IT FAILS LOUDLY. If the token ids are missing or misaligned, forward RAISES. There is
     deliberately no `if tok is None: return h` fallback -- that branch would hand you a plain SAE
     labelled `resid`, which is the one outcome this whole design exists to prevent.

P is materialised as a (vocab, d_model) lookup table at attach time, from `beta` (the fitted map,
~17 MB) and the model's OWN embedding table. Deriving it from the live model rather than shipping
a precomputed table is what makes an arm mix-up structurally hard: the random arm's P is built
from the random arm's embeddings. The fingerprint check catches it anyway if you cross the wires.
Cost is ~412 MB fp32 at Pythia's 50304 x 2048 -- a gather per forward, negligible beside the
encoder's pre-activation matrix.
"""
import hashlib
import os

import torch as t
import torch.nn as nn

PROBE_NAME = os.environ.get("PROBE_NAME", "skipembed")


def embed_fingerprint(W: t.Tensor) -> str:
    """Deterministic 16-hex-char digest of an embedding table.

    Subsampled every 512th row so this stays sub-second on a 50304 x 2048 table, and taken in
    float32 so it does not move when the model is loaded in bf16 vs fp16. Its only job is to
    answer "was this map fit against THIS model" -- the arm mix-up that would otherwise produce a
    complete, plausible, wrong cell.
    """
    a = W[::512].detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


class SkipEmbed(nn.Module):
    """Outputs r = h - P[tok]. Its output is the hookpoint both libraries read.

    `tok` is stashed by a pre-forward hook on the root model rather than passed as an argument,
    because neither library gives us a way to pass extra arguments through to a hooked module.
    It is a plain attribute, NOT a buffer, so it stays out of state_dict().
    """

    def __init__(self, P: t.Tensor):
        super().__init__()
        # Buffer, not Parameter -- see invariant 2 in the module docstring.
        self.register_buffer("P", P, persistent=False)
        self.tok: t.Tensor | None = None

    def set_tok(self, tok: t.Tensor) -> None:
        self.tok = tok

    def forward(self, h: t.Tensor) -> t.Tensor:
        tok = self.tok
        if tok is None:
            raise RuntimeError(
                "SkipEmbed got no token ids. The root pre-forward hook did not fire, or the "
                "model was called with inputs_embeds instead of input_ids. Refusing to fall "
                "back to h, which would silently train/score a PLAIN SAE labelled `resid`."
            )
        if tok.shape != h.shape[:-1]:
            raise RuntimeError(
                f"token/activation misalignment: tokens {tuple(tok.shape)} vs activations "
                f"{tuple(h.shape)}. Stale stash, or the batch was reshaped between the pre-hook "
                f"and this module."
            )
        # float32 for the subtraction: Var(r) is roughly half Var(h), so this is not catastrophic
        # cancellation, but the difference is where all the signal now lives and bf16 rounding on
        # it is free to avoid. Cast back so downstream sees the same dtype the plain arm saw.
        return (h.float() - self.P[tok].float()).to(h.dtype)


def build_P(model, beta: t.Tensor) -> t.Tensor:
    """P[t] = [embed_in[t], 1] @ beta, built from THIS model's embedding table.

    beta is (d_model + 1, d_model): the augmented least-squares solution from fit_map_pythia.py,
    with the trailing row holding the bias.
    """
    W = model.get_input_embeddings().weight
    E = W.detach().float()
    ones = t.ones(E.shape[0], 1, dtype=E.dtype, device=E.device)
    return t.cat([E, ones], dim=1) @ beta.to(E.device, E.dtype)


def load_beta(path: str, model=None):
    """Load a fitted map and, if a model is given, refuse it unless it was fit for THAT model."""
    ck = t.load(path, map_location="cpu", weights_only=True)
    beta, fp = ck["beta"], ck.get("embed_fingerprint")
    if model is not None and fp is not None:
        live = embed_fingerprint(model.get_input_embeddings().weight)
        if live != fp:
            raise SystemExit(
                f"MAP/MODEL MISMATCH -- refusing to run.\n"
                f"  map {path} was fit for embed fingerprint {fp} (arm={ck.get('arm')})\n"
                f"  the loaded model's fingerprint is {live}\n"
                f"Fit the map per arm; the trained arm's map on the random model produces a "
                f"complete and entirely wrong cell."
            )
    return beta, ck


def attach(model, beta: t.Tensor, layer: int, verbose: bool = True):
    """Install the probe. Returns (probe, handles).

    Registers two hooks:
      * root pre-forward, to stash input_ids (with_kwargs, since sparsify's resolve_widths calls
        model(**dummy_inputs) with kwargs while its training loop passes them positionally);
      * layer-L forward, which invokes the probe and returns None so the real output survives.
    """
    # Resolve through base_model: an AutoModel IS the base (base_model returns self), while an
    # AutoModelForCausalLM keeps `layers` one level down under `gpt_neox`. sparsify resolves
    # hookpoints the same way (`self.model.base_model.get_submodule(name)`), so this also keeps
    # the hookpoint STRING identical in both cases -- which is what lets one --hookpoints value
    # work whichever class the library happened to load the model with.
    base = getattr(model, "base_model", model)
    layer_mod = base.layers[layer]
    if hasattr(layer_mod, PROBE_NAME):
        raise RuntimeError(f"layers.{layer}.{PROBE_NAME} already exists -- attach() called twice")

    probe = SkipEmbed(build_P(model, beta))
    setattr(layer_mod, PROBE_NAME, probe)          # registers it as a real submodule

    def _stash(_m, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if not isinstance(ids, t.Tensor) or ids.is_floating_point():
            raise RuntimeError(
                "no integer input_ids on the model call -- SkipEmbed cannot index P. delphi or "
                "sparsify passed inputs_embeds, or the call signature changed."
            )
        probe.set_tok(ids)
        return None

    def _probe(mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        getattr(mod, PROBE_NAME)(h)   # side effect only: fires the probe's own hooks
        return None                    # <- keeps layer L's real output. Invariant 1.

    # Pre-hook on the BASE model, not the outer wrapper: a causal-LM head calls its base with
    # input_ids, so this fires in both cases, and it cannot be skipped by a wrapper that
    # preprocesses its arguments.
    handles = [
        base.register_forward_pre_hook(_stash, with_kwargs=True),
        layer_mod.register_forward_hook(_probe),
    ]
    if verbose:
        print(f"[skipembed] attached layers.{layer}.{PROBE_NAME}  "
              f"P={tuple(probe.P.shape)} {probe.P.dtype}  "
              f"({probe.P.numel() * probe.P.element_size() / 1e6:.0f} MB)")
    return probe, handles


def hookpoint(layer: int) -> str:
    return f"layers.{layer}.{PROBE_NAME}"
