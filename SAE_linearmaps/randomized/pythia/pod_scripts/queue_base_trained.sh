set -u
export TSAE_REPO=/temporal-saes TSAE_SELECT=500,500 HF_HOME=/dev/shm/hf TMPDIR=/dev/shm/tmp
while pgrep -f "name base_smoke" >/dev/null; do sleep 60; done
sleep 180
cd /dev/shm/delphi_run
CUDA_VISIBLE_DEVICES=0 TSAE_MAP_OUT=/dev/shm/map_base_trained.json \
  /root/venv312/bin/python /flipped-tuned-lens/SAE_linearmaps/randomized/pythia/delphi_tsae.py \
  EleutherAI/pythia-1b /dev/shm/tsae/base_trained_L8/trainer_0 \
  --hookpoints layers.8 --scorers fuzz detection --log_probs --max_latents 1000 \
  --n_tokens 30000000 --num_gpus 1 --dataset_repo Skylion007/openwebtext \
  --dataset_split 'train[:3%]' --name pythia1b_trained_base_L8 \
  > /dev/shm/logs/delphi_base_trained.log 2>&1
echo "### trained done $(date)"
