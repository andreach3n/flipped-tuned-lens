#!/usr/bin/env bash
# Fresh-pod bootstrap for the PYTHIA arm. Run from the repo root after `git clone`.
#
#   export HF_TOKEN=hf_xxx        # write scope: pushes the model, SAEs, logs and results
#   bash SAE_linearmaps/randomized/pythia/setup_pythia.sh
#
# DO NOT run the repo-root setup.sh for this work. It installs sae_lens + transformer_lens for
# the gemma pipeline; nothing in this folder imports either, and installing sae_lens is the
# documented cause of torch being bumped past the host driver. This stack is deliberately small.
#
# This covers the TRAINING side only. delphi needs its own venv with the pinned triple
# vllm==0.10.2 / transformers==4.56.1 / torch==2.8.0 -- see ../DELPHI_SETUP.md. Installing
# delphi into this environment drags torch forward and breaks it; the checkpoint on disk (or on
# the Hub) is the handoff between the two.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

# NOT -q: this can pull large wheels and take minutes. Quiet mode hides pip's progress bars,
# which makes a healthy install look like a hung pod.
python -m pip install -r "$HERE/requirements.txt"

# HF_HOME must be set before ANYTHING downloads. Unset, the container overlay (~20 GB) fills up
# and the run dies on "No space left on device" much later. /dev/shm is RAM and is wiped on a
# pod stop, which is fine here -- everything durable goes to the Hub.
export HF_HOME=${HF_HOME:-/dev/shm/hf}
mkdir -p "$HF_HOME" /dev/shm/logs /dev/shm/saes
echo "HF_HOME=$HF_HOME  (put this in ~/.bashrc -- a re-SSH that drops it reintroduces the bug)"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN is not set. Run:  export HF_TOKEN=hf_xxx   (write scope), then re-run."
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "(no WANDB_API_KEY -- training will log offline. export it for live logs.)"
fi

# ---- HARD GATES, not printouts -----------------------------------------------------------
python - <<'PY' || exit 1
import sys

import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | cuda build {torch.version.cuda} | available {ok} "
      f"| devices {torch.cuda.device_count()}")
if not ok:
    print("\n!! torch cannot see a GPU. Nothing here will work.\n"
          "   1. `nvidia-smi` -- no output => no GPU attached; redeploy.\n"
          "   2. Otherwise read its 'CUDA Version:' (the DRIVER's max) and install a torch\n"
          "      built for that or LOWER -- a cu130 wheel on a 12.8 driver fails exactly here:\n"
          "        pip install --force-reinstall --no-cache-dir torch \\\n"
          "          --index-url https://download.pytorch.org/whl/cu128   # or cu124\n",
          file=sys.stderr)
    sys.exit(1)

# THE GATE THAT MATTERS MOST, because everything above it can pass while this fails.
# transformers 5.x requires torch >= 2.5; the RunPod image ships 2.4.1+cu124. On that pairing
# transformers DISABLES ITS TORCH BACKEND at import and every AutoModel call dies with
# "requires the PyTorch library but it was not found" -- while torch itself is fine and
# cuda.is_available() is True. Checking torch alone does not catch it; ask transformers.
from transformers.utils import is_torch_available
import transformers
if not is_torch_available():
    print(f"\n!! transformers {transformers.__version__} cannot see torch {torch.__version__}.\n"
          "   transformers 5.x needs torch >= 2.5; this image has an older torch. Pin it back\n"
          "   rather than upgrading torch (which risks CUDA against the host driver):\n"
          "        pip install 'transformers==4.56.1'\n", file=sys.stderr)
    sys.exit(1)
print(f"transformers {transformers.__version__} sees torch OK")

# Confirm we got EleutherAI's sparsify and not Neural Magic's deprecated PyPI package of the
# same import name. `pip install sparsify` installs the WRONG one and fails much later with a
# confusing error, so check for the class we actually use.
try:
    import sparsify
    from sparsify import SparseCoder  # noqa: F401
    from sparsify.config import TrainConfig  # noqa: F401
except ImportError as e:
    print(f"\n!! wrong or missing sparsify ({e}).\n"
          "   `pip install sparsify` gets Neural Magic's deprecated package. Install:\n"
          "        pip install eai-sparsify\n", file=sys.stderr)
    sys.exit(1)
print(f"sparsify {getattr(sparsify, '__version__', '?')} (EleutherAI) OK")

try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes present -> 8-bit Adam (1.07 GB of optimizer state, not 4.3 GB)")
except ImportError:
    print("(!) bitsandbytes missing -> torch.optim.Adam, +3.2 GB VRAM. Fine on 48 GB+.")
PY

cat <<'NEXT'

setup done. Next, in order (see README.md for the full run order):

  1. materialize + GATE the random arm, then push it to the repo root:
       cd SAE_linearmaps/randomized/pythia
       VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/pythia1b_rand_s0 python -u randomize_pythia.py
       VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/pythia1b_rand_s0 python -u verify_randomization.py
       LOCAL=/dev/shm/pythia1b_rand_s0 PATH_IN_REPO= python -u hf_push.py

  2. train both arms, one per GPU (stagger the second until the first prints "Shuffling dataset")

  3. check_saes.py on BOTH arms before spending any judge time.

delphi is a SEPARATE venv -- do not install it here. See ../DELPHI_SETUP.md.
NEXT
