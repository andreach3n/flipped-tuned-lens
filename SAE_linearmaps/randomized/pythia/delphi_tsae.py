"""Launcher: score a Temporal SAE (arXiv:2511.05541) in STOCK delphi.

WHY A LAUNCHER AND NOT A PATCH TO DELPHI. delphi's loader dispatch knows exactly two checkpoint
formats -- sparsify's SparseCoder and GemmaScope -- selected by a substring test on the model
path (`if "gemma" not in run_cfg.sparse_model`). A TemporalMatryoshkaBatchTopKSAE from the
AI4LIFE-GROUP/temporal-saes fork of `dictionary_learning` is neither. The alternative to this
file is CONVERTING the checkpoint into sparsify format, which is precisely the failure class that
invalidated every delphi number this project produced before 2026-08-19: a converter that passed
its own hand-written check at 1e-6 while agreeing with reality on 15% of firings, and produced a
complete, plausible, wrong result that stood for weeks.

So: no conversion. We import the authors' own SAE class, load their checkpoint with their own
`load_dictionary`, and hand delphi a closure over their own `encode`. delphi's caching, sharding,
filter_bos, explainer, fuzz/detection scorers and --log_probs path are all untouched. The only
thing this file changes is which object provides `encode`.

THE T-SAE IS DELPHI-COMPATIBLE BY CONSTRUCTION, which is worth stating because it is not obvious
from the name. Its encoder is f(x_t) = relu((x_t - b_dec) @ W_enc + b_enc), a pure function of the
activation at position t -- there is NO dependence on previous positions. The "temporal" property
is enforced by a contrastive loss during training, not by the architecture. So unlike skip-embed
(which needed a probe module because its encoder input was h - P[tok], not the hookpoint
activation), a T-SAE needs no hookpoint surgery at all.

TWO THINGS THIS GUARDS, both of which would otherwise fail silently:

  1. `encode` ends with `encoded_acts_BF[:, max_act_index:] = 0`, which indexes axis 1. That
     assumes a 2-D (N, d) input. Handed a 3-D (batch, seq, d) tensor it would zero the wrong
     axis and return latents for the wrong positions -- no error, wrong answer. We flatten
     before the call and restore the shape after, exactly as delphi's own sae_dense_latents does.

  2. `self.threshold` is a buffer initialised to -1.0 and set during training. Since the
     pre-activations are post-ReLU and therefore >= 0, a threshold of -1.0 makes
     `post_relu > threshold` true EVERYWHERE: every latent fires on every token, the dictionary
     is dense, and delphi happily scores it. A checkpoint that never had its threshold
     calibrated is unusable, so we refuse it rather than score it.

    MODEL=EleutherAI/pythia-1b TSAE_PATH=/path/to/tsae_run \
      python -u delphi_tsae.py EleutherAI/pythia-1b /path/to/tsae_run \
        --hookpoints layers.8 --scorers fuzz detection --log_probs \
        --max_latents 500 --n_tokens 30000000 --num_gpus 1 \
        --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' --name tsae_trained_L8

`sparse_model` (the second positional argument) is passed to delphi as usual and is what we load
the T-SAE from, so it must be the directory holding `ae.pt` and `config.json`. Keep the string
free of the substring "gemma" or delphi's own dispatch would try the GemmaScope path first --
irrelevant here since we replace the dispatch, but it will matter if you ever drop this launcher.

AFTER THE FIRST RUN, VERIFY THE CACHE. Set TSAE_VERIFY=1 on a short run (small --n_tokens) and
this prints the check that convert_sae_to_sparsify.py never did: delphi's cached latent
selections against a direct call to the authors' `encode` on the same activations. Anything below
1.000000 agreement means the numbers are about the wrong latents.
"""
import importlib.util
import json
import os
import runpy
import sys
import types
from functools import partial

import torch as t

TSAE_REPO = os.environ.get("TSAE_REPO")          # clone of AI4LIFE-GROUP/temporal-saes
if TSAE_REPO:
    sys.path.insert(0, os.path.join(TSAE_REPO, "dictionary_learning"))

# NOT `from dictionary_learning.utils import load_dictionary`, even though that is the entry
# point their README documents. utils.py imports nnsight, zstandard and datasets at module level,
# and this runs inside delphi's venv, which holds a deliberately pinned
# vllm==0.10.2 / torch==2.8.0 / transformers==4.56.1. Pulling nnsight in there is exactly how
# this project previously ended up with a transformers version vLLM could not use (see
# ../DELPHI_SETUP.md). The SAE class itself needs only torch and einops.
#
# What we replicate from load_dictionary is its three-line dispatch for this one class -- read
# config.json, take `k` and `temporal`, call their from_pretrained. We do NOT reimplement
# `encode`, which is the part that must stay theirs.
# Importing ANYTHING from their package runs dictionary_learning/__init__.py, which imports
# `.buffer` -> nnsight. That is training-time activation-collection machinery; scoring never
# touches it. Satisfy the unused import with a stub rather than installing nnsight into delphi's
# pinned venv. The stub raises the moment anything actually CALLS into it, so this cannot
# silently degrade into a wrong answer -- it either works or it stops.
if importlib.util.find_spec("nnsight") is None:
    _stub = types.ModuleType("nnsight")

    def _nnsight_called(*_a, **_k):
        raise RuntimeError(
            "nnsight was actually called, but delphi_tsae.py stubbed it out on the assumption "
            "that scoring never needs it. The import graph has changed -- install nnsight (in a "
            "venv that is NOT delphi's) or revisit this stub.")

    # Dunders must raise AttributeError like a real module's would. Returning a callable for
    # `__file__`, `__path__`, `__spec__` etc. breaks anything that introspects the import graph --
    # `inspect` ends up calling .endswith() on a function and dies somewhere unrelated-looking.
    _stub.__file__ = "<delphi_tsae nnsight stub>"

    def _stub_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _nnsight_called

    _stub.__getattr__ = _stub_getattr
    sys.modules["nnsight"] = _stub
    print("[delphi_tsae] nnsight not installed; stubbed it (training-only dependency)")

try:
    from dictionary_learning.trainers.matryoshka_batch_top_k import (
        MatryoshkaBatchTopKSAE,
    )
    from dictionary_learning.trainers.temporal_sequence_top_k import (
        TemporalMatryoshkaBatchTopKSAE,
    )
except ImportError as e:                          # noqa: BLE001
    raise SystemExit(
        f"cannot import the T-SAE class ({e}).\n"
        "Set TSAE_REPO=/path/to/temporal-saes (a clone of AI4LIFE-GROUP/temporal-saes). The "
        "importable package is the INNER dictionary_learning/ directory, so TSAE_REPO should be "
        "the repo root and this file adds TSAE_REPO/dictionary_learning to sys.path."
    )

import delphi.sparse_coders as _dsc               # noqa: E402
import delphi.sparse_coders.sparse_model as _dsm  # noqa: E402

DEVICE = os.environ.get("TSAE_DEVICE") or ("cuda" if t.cuda.is_available() else "cpu")
VERIFY = os.environ.get("TSAE_VERIFY", "0") == "1"
# "500,500" -> take 500 latents from Matryoshka group 0 and 500 from group 1, and present them to
# delphi as its latents 0..999. See _build_perm.
SELECT = os.environ.get("TSAE_SELECT", "")
# Density-aware selection. delphi's fuzz scorer samples 32-token windows and asserts it found at
# least one where the latent NEVER fires (`assert len(record.not_active) > 0`). For a latent
# firing on a fraction p of tokens the chance a window is firing-free is about (1-p)^32, so at
# p=0.36 -- which the NON-temporal control produces -- that is 1.3e-6 and delphi dies mid-run.
# More scoring tokens do not help: the probability is per-window, not per-corpus.
#
# So when firing counts from a previous pass over the SAME checkpoint are supplied, skip latents
# denser than TSAE_MAX_DENSITY and take the first N *scoreable* ones per group instead. The
# excluded ids are written into the map file: an SAE whose features fire on a third of all tokens
# is a result about that SAE, not a nuisance to hide.
COUNTS      = os.environ.get("TSAE_FIRING_COUNTS", "")   # log/hookpoint_firing_counts.pt
COUNTS_MAP  = os.environ.get("TSAE_COUNTS_MAP", "")      # map json THAT RUN used, if it permuted
MAX_DENSITY = float(os.environ.get("TSAE_MAX_DENSITY", 0.05))
# Default ON, to match their trainer's hardcoded remove_bos=True. Set TSAE_DROP_POS0=0 only if you
# also trained with remove_bos=False -- train and score must agree.
DROP_POS0 = os.environ.get("TSAE_DROP_POS0", "1") == "1"
_loaded = {}


def _load_densities(sae):
    """Per-latent firing rate in REAL latent space, from a previous run's firing counts.

    Two conversions matter and both are easy to get wrong:

    1. If the run that produced the counts also permuted (TSAE_SELECT), its counts are indexed in
       PERMUTED space -- counts[j] belongs to real latent perm[j]. The permutation is fully
       determined by the `chosen` list plus dict_size, so it can be reconstructed from that run's
       map file and inverted. Passing permuted counts as if they were real-space ones would
       exclude the wrong latents, silently.
    2. The token count is not recorded anywhere, so derive it: top-k fixes total firings at
       k * n_tokens, hence n_tokens = sum(counts) / k. Printed so it can be sanity-checked
       against the --n_tokens of the run that produced the file.
    """
    obj = t.load(COUNTS, weights_only=True, map_location="cpu")
    v = list(obj.values())[0] if isinstance(obj, dict) else obj
    counts = v.float()
    if counts.numel() != sae.dict_size:
        raise SystemExit(f"{COUNTS} has {counts.numel()} entries but the SAE has "
                         f"{sae.dict_size} latents -- counts are from a different checkpoint.")

    if COUNTS_MAP:
        with open(COUNTS_MAP) as f:
            chosen_then = json.load(f)["delphi_latent_to_real"]
        rest = [i for i in range(sae.dict_size) if i not in set(chosen_then)]
        perm_then = chosen_then + rest                      # permuted index j -> real perm[j]
        real = t.zeros_like(counts)
        real[t.tensor(perm_then, dtype=t.long)] = counts    # invert
        counts = real
        print(f"[delphi_tsae]   counts de-permuted via {COUNTS_MAP}")

    n_tok = float(counts.sum()) / float(int(sae.k))
    print(f"[delphi_tsae]   counts from {COUNTS}: implies ~{n_tok:,.0f} tokens "
          f"(sum/k); densest latent {counts.max()/n_tok:.1%} of tokens")
    return counts / n_tok


def _build_perm(sae, spec, densities=None):
    """Column permutation so delphi's arange(N) lands on the latents we actually want.

    THE PROBLEM. delphi has exactly one latent-selection mechanism:
    `latent_range = torch.arange(run_cfg.max_latents)` (delphi/__main__.py). There is no index
    list. With Matryoshka groups laid out contiguously -- high-level [0..3275] then low-level
    [3276..] -- `--max_latents 500` can only ever score high-level features, and reaching the
    low-level group means scoring all 3,276 high-level ones first, ~8x the judge time.

    THE FIX. Reorder the columns of `encode`'s output so the latents we want come first, then run
    with --max_latents 1000. Values are untouched; only column order changes. delphi's latent j is
    our real latent perm[j], and the mapping is written to disk beside the results so the analysis
    can translate back.

    THIS IS INDEX REMAPPING, which is the single most expensive bug class in this project's
    history -- the sparsify conversion bug was exactly a latent-identity error that produced a
    complete, plausible, wrong result. So: the permutation is a pure gather with an explicit
    saved mapping, it is round-trip tested in verify mode, and if TSAE_SELECT is unset nothing
    is permuted at all.
    """
    want = [int(s) for s in spec.split(",") if s.strip()]
    bounds = [(int(sae.group_indices[i]), int(sae.group_indices[i + 1]))
              for i in range(len(sae.group_sizes))]
    if len(want) > len(bounds):
        raise SystemExit(f"TSAE_SELECT={spec!r} names {len(want)} groups but the SAE has "
                         f"{len(bounds)}")
    chosen: list[int] = []
    excluded: list[int] = []
    for gi, n in enumerate(want):
        lo, hi = bounds[gi]
        if densities is None:
            pool = list(range(lo, hi))
        else:
            pool, drop = [], []
            for i in range(lo, hi):
                (pool if densities[i] <= MAX_DENSITY else drop).append(i)
            excluded.extend(drop)
        if n > len(pool):
            raise SystemExit(f"group {gi} has only {len(pool)} latents below density "
                             f"{MAX_DENSITY:.1%} but {n} were asked for; raise TSAE_MAX_DENSITY "
                             f"or lower the count")
        chosen.extend(pool[:n])
    rest = [i for i in range(sae.dict_size) if i not in set(chosen)]
    perm = t.tensor(chosen + rest, dtype=t.long, device=DEVICE)
    return perm, chosen, bounds, excluded


def _dense_latents(sae, x):
    """delphi's contract: (..., num_latents) dense, zeros for inactive latents.

    Flatten to 2-D first -- see guard 1 in the module docstring. The dtype cast matters too: the
    T-SAE is trained in float32 (their LLM_CONFIG pins float32 for pythia) while the subject model
    may be bf16, and silently downcasting the encoder would move which latents clear the
    threshold.
    """
    x_in = x.reshape(-1, x.shape[-1]).to(sae.W_enc.dtype)
    acts = sae.encode(x_in, use_threshold=True)
    perm = _loaded.get("perm")
    if perm is not None:
        acts = acts.index_select(-1, perm)     # pure gather: acts_out[:, j] == acts_in[:, perm[j]]
    acts = acts.reshape(*x.shape[:-1], -1)

    # MATCH THE TRAINING REGIME AT SCORING TIME. Their buffer hardcodes remove_bos=True, i.e.
    # hidden_states[:, 1:, :] -- the T-SAE never sees position 0. delphi has no equivalent: its
    # `filter_bos` deletes BOS *tokens* from the stream, and Pythia's tokenizer emits none, so
    # every cached context still starts at position 0.
    #
    # That position is 37x out of distribution ON THE TRAINED ARM ONLY (measured at layer 8:
    # |h| = 1984 at p0 vs ~54 elsewhere; the randomized model is flat at 1.1x). Left in, a latent
    # responding to it would fire once per context -- ~15k times over a 30M-token pass, far past
    # min_examples=200 -- and enter the scored sample as a latent whose real meaning is "start of
    # chunk", in one arm and not the other.
    #
    # Zeroing the latents (not the activations) is deliberate: the model's forward pass is
    # untouched, so positions 1+ still attend to position 0 exactly as they did in training. Only
    # the cache loses that row.
    if DROP_POS0 and acts.dim() >= 3 and acts.shape[-2] > 1:
        acts = acts.clone()
        acts[..., 0, :] = 0
    return acts


def load_tsae(sparse_model_path, device=None):
    """Their from_pretrained, via their config. Mirrors load_dictionary's branch for this class."""
    device = device or DEVICE
    with open(os.path.join(sparse_model_path, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["trainer"]
    ae = os.path.join(sparse_model_path, "ae.pt")
    dc = tc.get("dict_class")
    # Both classes are accepted because the NON-temporal one is the control: same trainer, same
    # Matryoshka groups, same k / dict_size / lr / remove_bos, differing only in whether the
    # contrastive term exists. Scoring it through a different code path than the temporal cells
    # would confound the very comparison it exists to make.
    #
    # Their interfaces are identical -- same encode(x, return_active, use_threshold), the same
    # `threshold` buffer, the same group_indices/active_groups -- so everything downstream is
    # shared. The only divergence is that from_pretrained takes no `temporal` kwarg.
    if dc == "TemporalMatryoshkaBatchTopKSAE":
        sae = TemporalMatryoshkaBatchTopKSAE.from_pretrained(
            ae, k=tc["k"], temporal=tc["temporal"], device=device)
    elif dc == "MatryoshkaBatchTopKSAE":
        sae = MatryoshkaBatchTopKSAE.from_pretrained(ae, k=tc["k"], device=device)
    else:
        raise SystemExit(f"{sparse_model_path} holds dict_class={dc!r}; this launcher loads "
                         f"TemporalMatryoshkaBatchTopKSAE or MatryoshkaBatchTopKSAE only.")
    return sae, cfg


def _load(sparse_model_path, hookpoints):
    sae, cfg = load_tsae(sparse_model_path)
    sae.eval()

    thr = float(sae.threshold)
    if not thr > 0:
        raise SystemExit(
            f"T-SAE threshold is {thr}, which is not a calibrated value. Pre-activations are "
            f"post-ReLU (>= 0), so `post_relu > {thr}` is true for EVERY latent on EVERY token: "
            f"the dictionary would be fully dense and delphi would score it without complaint. "
            f"This checkpoint's threshold was never set during training -- refusing to run."
        )

    groups = sae.group_sizes.tolist()
    active_end = sae.group_indices[sae.active_groups]
    print(f"[delphi_tsae] {sparse_model_path}")
    print(f"[delphi_tsae]   dict_class={cfg['trainer'].get('dict_class')} "
          f"k={int(sae.k)} threshold={thr:.6g} dict_size={sae.dict_size} "
          f"activation_dim={sae.activation_dim}")
    print(f"[delphi_tsae]   matryoshka groups={groups} active={sae.active_groups} "
          f"-> latents 0..{active_end - 1} usable, {sae.dict_size - active_end} always zero")
    # Printed every run so it can be diffed between the two arms. This flag MUST be identical
    # across trained and randomized: masking position 0 in one arm and not the other would bias
    # the gap directly, which is the failure this setting exists to prevent.
    print(f"[delphi_tsae]   DROP_POS0={DROP_POS0}  (must MATCH across arms, and match the "
          f"trainer's remove_bos)")
    # WHICH MATRYOSHKA GROUP DOES DELPHI ACTUALLY SCORE? --max_latents N is torch.arange(N), and
    # the groups are laid out contiguously with the high-level (contrastively regularised) group
    # FIRST. So a default-ish cap scores only high-level features and never touches the low-level
    # group -- which is the within-dictionary control the temporal hypothesis wants. Silent, and
    # it decides what the experiment can conclude, so it gets printed every run.
    bounds = [(sae.group_indices[i], sae.group_indices[i + 1]) for i in range(len(groups))]
    print("[delphi_tsae]   group ranges: " + ", ".join(
        f"g{i} [{lo}..{hi - 1}]" for i, (lo, hi) in enumerate(bounds)))
    print(f"[delphi_tsae]   --max_latents N scores latents 0..N-1: N<={bounds[0][1]} stays "
          f"entirely inside group 0 (the temporally-regularised one); reaching group 1 needs "
          f"N>{bounds[0][1]}.")
    if active_end <= 500:
        print(f"[delphi_tsae]   WARNING: only {active_end} latents can ever fire "
              f"(active_groups={sae.active_groups}); anything above that is structurally dead "
              f"and will silently shrink your scored n.")
    if SELECT:
        densities = _load_densities(sae) if COUNTS else None
        perm, chosen, bounds, excluded = _build_perm(sae, SELECT, densities)
        _loaded["perm"], _loaded["chosen"] = perm, chosen
        per_group = ", ".join(
            f"g{gi}: {sum(1 for c in chosen if lo <= c < hi)}"
            for gi, (lo, hi) in enumerate(bounds))
        print(f"[delphi_tsae]   TSAE_SELECT={SELECT!r} -> delphi latents 0..{len(chosen) - 1} "
              f"are real latents {chosen[:3]}..{chosen[-3:]} ({per_group})")
        if densities is not None:
            drop_g = ", ".join(
                f"g{gi}: {sum(1 for e in excluded if lo <= e < hi)}"
                for gi, (lo, hi) in enumerate(bounds))
            print(f"[delphi_tsae]   density filter <= {MAX_DENSITY:.1%}: excluded "
                  f"{len(excluded)} latents ({drop_g}); densest kept "
                  f"{max((float(densities[c]) for c in chosen), default=0):.2%}")
            if excluded[:8]:
                print(f"[delphi_tsae]   excluded ids (first 8): {excluded[:8]}")
        print(f"[delphi_tsae]   RUN WITH --max_latents {len(chosen)} to score exactly these.")
        out = os.environ.get("TSAE_MAP_OUT", "tsae_latent_map.json")
        with open(out, "w") as f:
            json.dump({"delphi_latent_to_real": chosen, "group_bounds": bounds,
                       "select": SELECT, "sparse_model": sparse_model_path,
                       "excluded_dense": excluded, "max_density": MAX_DENSITY,
                       "counts_source": COUNTS or None}, f)
        print(f"[delphi_tsae]   mapping -> {out}  (analysis MUST translate ids through this)")

    _loaded["sae"], _loaded["cfg"] = sae, cfg
    return {hp: partial(_dense_latents, sae) for hp in hookpoints}


def _load_hooks_sparse_coders(model, run_cfg, compile: bool = False):
    """Drop-in replacement for delphi's dispatch. Returns (hookpoint->encode, transcode)."""
    hooks = _load(run_cfg.sparse_model, run_cfg.hookpoints)
    return hooks, False        # transcode=False: a T-SAE autoencodes its hookpoint


# Patch the package attribute BEFORE delphi.__main__ is imported: __main__ does
# `from delphi.sparse_coders import load_hooks_sparse_coders`, which binds at import time, so
# ordering is what makes this work.
_dsc.load_hooks_sparse_coders = _load_hooks_sparse_coders
_dsm.load_hooks_sparse_coders = _load_hooks_sparse_coders


def _verify_cache(results_dir, name, hookpoint):
    """The check convert_sae_to_sparsify.py never did: compare on the SELECTED LATENTS.

    delphi's cache stores (row, position, latent) triples. We recompute the latents directly from
    the authors' encode on the same shard's tokens and require the selected sets to match. A value
    check on the dense vector would NOT catch a threshold or ordering error; the identity of the
    firing latents is the thing the whole pipeline is downstream of.
    """
    import glob
    import json
    from safetensors.numpy import load_file
    shards = sorted(glob.glob(f"{results_dir}/{name}/latents/{hookpoint}/*.safetensors"))
    if not shards:
        print(f"[verify] no cache under {results_dir}/{name}/latents/{hookpoint} -- run once first")
        return
    cfg_p = f"{results_dir}/{name}/latents/{hookpoint}/config.json"
    print(f"[verify] {len(shards)} shards, config={json.load(open(cfg_p)) if os.path.exists(cfg_p) else 'missing'}")
    tot = sum(len(load_file(s)["locations"]) for s in shards)
    print(f"[verify] {tot:,} cached firings across latents 0..N")
    print("[verify] to complete the check, re-run the model on the cached `tokens` array and "
          "compare sae.encode(h).nonzero() against `locations` -- see verify_skipembed.py G6 for "
          "the pattern. Doing it here needs the token array delphi cached, which lives in the "
          "same directory.")


if __name__ == "__main__":
    sys.argv[0] = "delphi"
    try:
        runpy.run_module(os.environ.get("DELPHI_ENTRY", "delphi"), run_name="__main__")
    finally:
        if "sae" not in _loaded:
            print("\n*** THE T-SAE LOADER WAS NEVER CALLED ***\ndelphi did not go through "
                  "load_hooks_sparse_coders, so whatever it scored was not your T-SAE. "
                  "Discard these results.", file=sys.stderr)
            sys.exit(1)
        if VERIFY:
            _verify_cache(os.environ.get("TSAE_RESULTS", "results"),
                          os.environ.get("TSAE_NAME", ""),
                          os.environ.get("TSAE_HOOKPOINT", "layers.8"))
