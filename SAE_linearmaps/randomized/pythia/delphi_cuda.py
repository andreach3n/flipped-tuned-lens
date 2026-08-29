"""Launcher: run STOCK delphi on a sparsify SAE loaded from a LOCAL directory.

WHY THIS EXISTS. delphi 0.1.3's `load_sparsify_hooks` has two branches and only one of them puts
the SAE on the GPU (delphi/sparse_coders/load_sparsify.py):

    if name_path.exists():                       # a local directory
        sparse_model = SparseCoder.load_from_disk(..., device="cpu")
        ...
        sparse_model_dict[hookpoint] = sparse_model            # <- never moved
    else:                                        # an HF repo id
        sparse_models = SparseCoder.load_many(name, device="cpu")
        ...
        sparse_model_dict[hookpoint] = sparse_model.to(device) # <- moved

So an SAE passed as a path stays on CPU while the hooked activations arrive on cuda, and caching
dies immediately with

    RuntimeError: Expected all tensors to be on the same device, ... cuda:0 and cpu

That is the whole bug. It bites every run that points at a checkpoint on disk, which is every run
in this project; older delphi did not have it, which is why the earlier R=64 cells scored fine.

This wraps SparseCoder.load_from_disk to force the device when it would otherwise be cpu, then
calls delphi's own entry point. Nothing else is touched -- caching, sharding, filter_bos, the
explainer, the scorers and --log_probs are all stock. Passing the checkpoint's HF repo id instead
would also avoid the bug, by taking the other branch, but that requires the SAE to be on the Hub
and re-downloads it.

    SAE_DEVICE=cuda python delphi_cuda.py <model> <sae_dir> --hookpoints layers.8 ...

Same arguments as `python -m delphi`; they are passed through untouched.
"""
import os
import runpy
import sys

import torch as t
from sparsify import SparseCoder

SAE_DEVICE = os.environ.get("SAE_DEVICE") or ("cuda" if t.cuda.is_available() else "cpu")

_orig_load_from_disk = SparseCoder.load_from_disk
_forced = 0


def _load_from_disk(path, device="cpu", **kwargs):
    global _forced
    if str(device) == "cpu" and SAE_DEVICE != "cpu":
        _forced += 1
        print(f"[delphi_cuda] SparseCoder would have loaded on cpu -> forcing {SAE_DEVICE}")
        device = SAE_DEVICE
    return _orig_load_from_disk(path, device=device, **kwargs)


# load_many() routes through load_from_disk, so this single patch covers both entry points.
SparseCoder.load_from_disk = staticmethod(_load_from_disk)

if __name__ == "__main__":
    sys.argv[0] = "delphi"
    try:
        runpy.run_module(os.environ.get("DELPHI_ENTRY", "delphi"), run_name="__main__")
    finally:
        if _forced == 0:
            print("\n[delphi_cuda] NOTE: the device patch never fired. Either delphi passed a "
                  "real device (fixed upstream) or it never loaded a SparseCoder at all -- check "
                  "the results before trusting them.", file=sys.stderr)
