set -u
export TSAE_REPO=/temporal-saes TSAE_SELECT=500,500 TSAE_MAX_DENSITY=0.05
export HF_HOME=/dev/shm/hf TMPDIR=/dev/shm/tmp
V=/root/venv312/bin/python
P=/flipped-tuned-lens/SAE_linearmaps/randomized/pythia
R=/dev/shm/delphi_run/results
cd /dev/shm/delphi_run

# $1=arm $2=model $3=gpu $4=pid-of-v1
arm_watch() {
  A=$1; M=$2; G=$3; PID=$4
  echo "### [$A] waiting on pid $PID  $(date)"
  while kill -0 $PID 2>/dev/null; do sleep 60; done
  echo "### [$A] v1 exited $(date)"

  CNT=$R/pythia1b_${A}_base_L8/log/hookpoint_firing_counts.pt
  if [ ! -f "$CNT" ]; then echo "### [$A] NO firing counts -- caching never finished, stopping"; return 1; fi

  N=$(ls $R/pythia1b_${A}_base_L8/scores/fuzz 2>/dev/null | wc -l)
  if ! grep -q 'No non-activating examples found' /dev/shm/logs/delphi_base_${A}.log; then
    echo "### [$A] v1 did NOT hit the density assertion ($N fuzz scores) -- no v2 needed"
    return 0
  fi
  echo "### [$A] v1 crashed on the density assertion after $N scores -- launching v2 $(date)"

  sleep 180
  CUDA_VISIBLE_DEVICES=$G     TSAE_FIRING_COUNTS=$CNT     TSAE_COUNTS_MAP=/dev/shm/map_base_${A}.json     TSAE_MAP_OUT=/dev/shm/map_base_${A}_v2.json     $V $P/delphi_tsae.py "$M" /dev/shm/tsae/base_${A}_L8/trainer_0     --hookpoints layers.8 --scorers fuzz detection --log_probs --max_latents 1000     --n_tokens 30000000 --num_gpus 1 --dataset_repo Skylion007/openwebtext     --dataset_split 'train[:3%]' --name pythia1b_${A}_base_L8_v2     > /dev/shm/logs/delphi_base_${A}_v2.log 2>&1
  echo "### [$A] v2 finished $(date); fuzz scores: $(ls $R/pythia1b_${A}_base_L8_v2/scores/fuzz 2>/dev/null | wc -l)"
}

arm_watch trained EleutherAI/pythia-1b 0 16819 &
arm_watch rand /dev/shm/pythia1b_rand_s0 1 15893 &
wait
echo "### ALL DONE $(date)"
