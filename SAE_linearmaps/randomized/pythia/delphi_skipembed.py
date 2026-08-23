"""Launcher: run STOCK delphi against the skip-embed hookpoint.

delphi keeps doing everything it normally does -- caching, sharding, filter_bos, explanations,
fuzz/detection scoring. The ONLY change is that the model it loads has the probe attached, so
`--hookpoints layers.8.skipembed` yields r = h - P[tok] and delphi hands the encoder exactly what
that encoder was trained on.

THIS IS DELIBERATELY NOT write_delphi_cache.py. On Gemma we wrote delphi's latent cache by hand
because a broken sparsify conversion made its own path untrustworthy, and the bugs that cost this
project months lived in that hand-written layer -- format, indexing, sharding, `tokens` alignment.
Here the checkpoint is a native sparsify SparseCoder and the hookpoint is a real module, so there
is nothing to reimplement. delphi's cache is delphi's own.

delphi takes the model as a string and loads it internally, so the interception point is
transformers' loader. Both AutoModel and AutoModelForCausalLM are wrapped because which one delphi
uses is a detail of its version, and guessing wrong would silently leave the probe unattached --
so the launcher also FAILS at exit if nothing was ever attached.

    MAP=/dev/shm/maps/P_pythia1b_L8_trained.pt \
      python -u delphi_skipembed.py EleutherAI/pythia-1b /dev/shm/saes/<run> \
        --hookpoints layers.8.skipembed --max_latents 500 --n_tokens 30000000 \
        --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' \
        --scorers detection fuzz --log_probs --name <cell>

FIRST RUN ON A NEW BOX, DO THIS SMALL. delphi's cache config and firing-count log key off the
hookpoint string, and a dotted name like `layers.8.skipembed` is not a shape it has been exercised
on here. Run it once with a tiny --n_tokens and confirm `latents/layers.8.skipembed/` appears with
sane shard names before committing 30M tokens and judge time to it.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import LAYER            # noqa: E402
from skipembed import PROBE_NAME, attach, hookpoint, load_beta  # noqa: E402

MAP = os.environ.get("MAP")
if not MAP:
    raise SystemExit("MAP=<path to P_pythia1b_L*_{trained,rand}.pt> is required -- and it must be "
                     "the map for the ARM you are scoring.")
if not any(PROBE_NAME in a for a in sys.argv[1:]):
    raise SystemExit(
        f"none of the arguments name the probe. Expected --hookpoints {hookpoint(LAYER)!r}.\n"
        f"Scoring the raw hookpoint would feed the resid encoder raw h -- no error, wrong latents, "
        f"a complete and plausible wrong cell."
    )

import transformers                            # noqa: E402

_attached = 0


def _wrap(cls):
    orig = cls.from_pretrained

    def patched(*args, **kwargs):
        global _attached
        model = orig(*args, **kwargs)
        # Only the subject model has decoder layers to hook, and on a causal-LM wrapper they sit
        # under base_model. delphi may load auxiliary models, so attach only where layer L exists.
        base = getattr(model, "base_model", model)
        if not hasattr(base, "layers") or len(base.layers) <= LAYER:
            return model
        beta, meta = load_beta(MAP, model)      # fingerprint-checked against THIS model
        if meta.get("layer") != LAYER:
            raise SystemExit(f"map was fit for layer {meta.get('layer')}, scoring layer {LAYER}")
        print(f"[delphi_skipembed] map {MAP}  arm={meta.get('arm')} "
              f"ev_centred={meta.get('explained_variance_centred'):.4f}")
        attach(model, beta, LAYER)
        _attached += 1
        return model

    cls.from_pretrained = patched


for _cls in (transformers.AutoModel, transformers.AutoModelForCausalLM):
    _wrap(_cls)

if __name__ == "__main__":
    sys.argv[0] = "delphi"
    try:
        runpy.run_module(os.environ.get("DELPHI_ENTRY", "delphi"), run_name="__main__")
    finally:
        if _attached == 0:
            print("\n*** THE PROBE WAS NEVER ATTACHED ***\ndelphi did not load the subject model "
                  "through transformers' AutoModel/AutoModelForCausalLM, so these results are "
                  "from the RAW hookpoint, not skip-embed. Discard them.", file=sys.stderr)
            sys.exit(1)
