#!/usr/bin/env bash
# Fresh-pod bootstrap -- NO network volume needed. Run from the repo root after `git clone`.
#   export HF_TOKEN=hf_xxx   # write scope: pulls gated gemma AND pushes your SAEs
#   bash setup.sh
set -e

pip install -q -r SAE_linearmaps/requirements.txt

# keep the gemma download on the Volume Disk so a Stop/Start doesn't re-download ~5 GB
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/out "$HF_HOME"

if [ -z "$HF_TOKEN" ]; then
  echo "!! HF_TOKEN is not set. Run:  export HF_TOKEN=hf_xxx   (write scope), then re-run setup.sh"
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cat <<'NEXT'
setup done. From here:
  cd SAE_linearmaps
  python fit_map.py                       # fit + push linear_map_layer_13.pt (run ONCE)
  MODE=full    TRAIN_TOKENS=200000000 python train_sae_res.py
  MODE=resid   TRAIN_TOKENS=200000000 python train_sae_res.py
  MODE=hybrid  TRAIN_TOKENS=200000000 python train_sae_res.py
  MODE=outbias TRAIN_TOKENS=200000000 python train_sae_res.py
NEXT
