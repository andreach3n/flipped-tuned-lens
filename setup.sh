#!/usr/bin/env bash
# Fresh-pod bootstrap -- NO network volume needed. Run from the repo root after `git clone`.
#   export HF_TOKEN=hf_xxx   # write scope: pulls gated gemma AND pushes your SAEs
#   bash setup.sh
set -e

python -m pip install -q -r SAE_linearmaps/requirements.txt
python -m pip uninstall -y torchvision torchaudio   # unused here; a torch-version mismatch makes them crash transformers
python -m spacy download en_core_web_sm || echo "(spaCy model skipped -- only the probe's syntactic panel needs it)"

# keep the gemma download on the Volume Disk so a Stop/Start doesn't re-download ~5 GB
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/out "$HF_HOME"

if [ -z "$HF_TOKEN" ]; then
  echo "!! HF_TOKEN is not set. Run:  export HF_TOKEN=hf_xxx   (write scope), then re-run setup.sh"
fi

# HARD GATE, not a printout. `pip install sae_lens` can bump torch to a wheel that cannot see the
# GPU; every script then silently falls back to device="cpu" and dies much later on an obscure
# "Attempting to deserialize object on a CUDA device" from t.load. Fail here instead.
python - <<'PY' || exit 1
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | cuda build {torch.version.cuda} | available {ok}")
if not ok:
    # Seen 2026-08-11: pip pulled torch 2.13.0+cu130 onto a host with a 12.8 driver -> CUDA 13
    # needs a newer driver than RunPod provides, so torch silently has no GPU. The wheel index
    # must MATCH THE HOST DRIVER, which is why this is not pinned to one URL.
    print("\n!! torch cannot see a GPU. Nothing here will work.\n"
          "   1. `nvidia-smi` -- no output => this pod has no GPU attached; redeploy.\n"
          "   2. Otherwise read its 'CUDA Version:' (the DRIVER's max) and install a torch\n"
          "      built for that or LOWER -- a cu130 wheel on a 12.8 driver fails exactly here:\n"
          "        pip install --force-reinstall --no-cache-dir torch \\\n"
          "          --index-url https://download.pytorch.org/whl/cu128   # or cu124\n", file=sys.stderr)
    sys.exit(1)
PY

cat <<'NEXT'
setup done. Pick the workflow you are actually on:

  EVAL the 100M / R=32 fleet (k=32, d_sae=73728) -- per arm:
    cd SAE_linearmaps
    VARIANT=trained  HF_REPO=<trained-repo> K=32 SUFFIX=_d73728_100M python -u eval_fvu.py
    VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-repo> K=32 SUFFIX=_d73728_100M python -u eval_fvu.py
    (add CKPT=t20M|t40M|t60M|t80M for the milestone ladder)
    Raw FVU means nothing across configs -- re-run gauss_null.py at the SAME config.

  TRAIN a new arm (only when the arm has no artifacts yet):
    python fit_map.py            # ONCE PER ARM -- the map is model-specific.
                                 # It OVERWRITES linear_map_layer_13.pt in $HF_REPO. Do not
                                 # re-run it for an arm that already has one.
    bash randomized/launch_replication.sh <run>     # see that script for the run matrix
NEXT
