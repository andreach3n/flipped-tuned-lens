"""Launcher: run STOCK sparsify against the skip-embed hookpoint.

sparsify's CLI takes the model as a string and loads it itself, so there is no way to hand it a
model with the probe already attached. This wraps the one function that does the loading --
`sparsify.__main__.load_artifacts` -- attaches the probe to the model it returns, and then calls
sparsify's own `run()`. Everything else is stock: same argument parser, same Trainer, same loss,
same top-k, same checkpoint format. delphi will load the resulting checkpoint natively.

Arguments are passed straight through, so this is a drop-in for `python -m sparsify`:

    MAP=/dev/shm/maps/P_pythia1b_L8_trained.pt \
      python -u train_skipembed.py EleutherAI/pythia-1b Skylion007/openwebtext \
        --hookpoints layers.8.skipembed ...

THE GUARD THAT MATTERS. If --hookpoints does not name the probe, this refuses to run. Forgetting
it would train a perfectly healthy PLAIN SAE, save it under a `resid` run name, pass check_saes,
and score -- a complete wrong cell with no error anywhere. That is the failure mode this whole
design exists to prevent, so it is checked rather than trusted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from randomize_pythia import LAYER            # noqa: E402
from skipembed import PROBE_NAME, attach, hookpoint, load_beta  # noqa: E402

MAP = os.environ.get("MAP")
if not MAP:
    raise SystemExit("MAP=<path to P_pythia1b_L*_{trained,rand}.pt> is required -- run "
                     "fit_map_pythia.py for THIS arm first.")

import sparsify.__main__ as sm                # noqa: E402

_orig_load_artifacts = sm.load_artifacts
_attached = 0


def _load_artifacts(args, rank):
    global _attached
    want = hookpoint(LAYER)
    if not any(PROBE_NAME in hp for hp in args.hookpoints):
        raise SystemExit(
            f"--hookpoints {args.hookpoints} does not name the skip-embed probe.\n"
            f"Expected {want!r}. Training on the raw hookpoint here would produce a PLAIN SAE "
            f"under a resid run name -- healthy, plausible, and the wrong experiment."
        )
    model, dataset = _orig_load_artifacts(args, rank)
    beta, meta = load_beta(MAP, model)         # fingerprint-checked against THIS model
    print(f"[train_skipembed] map {MAP}  arm={meta.get('arm')} "
          f"layer={meta.get('layer')} tokens={meta.get('tokens'):,} "
          f"ev_centred={meta.get('explained_variance_centred'):.4f}")
    if meta.get("layer") != LAYER:
        raise SystemExit(f"map was fit for layer {meta.get('layer')}, this run is layer {LAYER}")
    attach(model, beta, LAYER)
    _attached += 1
    return model, dataset


sm.load_artifacts = _load_artifacts

if __name__ == "__main__":
    sm.run()
    if _attached == 0:
        raise SystemExit("BUG: sparsify never called load_artifacts -- the probe was never "
                         "attached. Do not trust this checkpoint.")
