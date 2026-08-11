#!/usr/bin/env bash
# Possibility-2 retrain: does the Heap et al. non-replication survive proper SAE training?
# Paper recipe on gemma-2-2b: k=32, expansion factor 32 (d_sae = 32 * 2304 = 73728), 100M tokens.
# LR stays at our 4e-4 default for the long runs; the 20M LR sweep brackets it on the random arm.
#
# Run from SAE_linearmaps/ — one run per pod, or several at once on a multi-GPU pod:
#     export HF_TOKEN=hf_xxx TRAINED_REPO=... RAND_REPO=...
#     CUDA_VISIBLE_DEVICES=0 bash randomized/launch_replication.sh trained-full
#
# Artifact names are config-tagged by train_sae_res.py (e.g. sae_full_k32_d73728_100M_final.pt),
# so these runs can share the per-arm repos with the old 20M/16k artifacts without overwriting
# anything. Milestone checkpoints land every 20M tokens (…_t20M.pt etc.) — that is the
# auto-interp-vs-training-tokens curve, and the rand-full t20M file doubles as the LR=4e-4
# point of the sweep.
#
# CO-LOCATED RUNS (several GPUs, one machine). Two things that were free with one-run-per-pod:
#   * GPU: the training code asks for "cuda" = the first VISIBLE device, so WITHOUT
#     CUDA_VISIBLE_DEVICES all six runs pile onto GPU 0 (contention, then OOM). Pin each run.
#   * DISK: the config tag does not encode the model ARM, so trained-full and rand-full write the
#     SAME local filenames -- and every mode writes P.pt. A shared OUT_DIR would have them clobber
#     each other's checkpoints and push the wrong arm's weights. Hence the per-run OUT_DIR below.
#     Six runs need ~45 GB of scratch; if the disk is smaller, put it in RAM: SCRATCH=/dev/shm.
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN=hf_xxx (write scope)}"
: "${TRAINED_REPO:?export TRAINED_REPO=<trained-arm HF repo used by the 20M experiment>}"
: "${RAND_REPO:?export RAND_REPO=<random-arm HF repo, the …-rand-all-s0 one>}"

RUN="${1:-}"
export K=32 D_SAE=73728 SEED=0
export OUT_DIR="${OUT_DIR:-${SCRATCH:-/workspace}/out_$RUN}"
mkdir -p "$OUT_DIR"
echo "[launch] run=$RUN OUT_DIR=$OUT_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unpinned!>}"

case "$RUN" in
  # ---- 100M-token retrains, both arms, plain + skip-embed --------------------------------
  trained-full)   VARIANT=trained  HF_REPO=$TRAINED_REPO MODE=full  TRAIN_TOKENS=100000000 \
                  exec python -u train_sae_res.py ;;
  trained-resid)  VARIANT=trained  HF_REPO=$TRAINED_REPO MODE=resid TRAIN_TOKENS=100000000 \
                  exec python -u train_sae_res.py ;;
  rand-full)      VARIANT=rand_all INIT_SEED=0 HF_REPO=$RAND_REPO MODE=full  TRAIN_TOKENS=100000000 \
                  exec python -u train_sae_res.py ;;
  rand-resid)     VARIANT=rand_all INIT_SEED=0 HF_REPO=$RAND_REPO MODE=resid TRAIN_TOKENS=100000000 \
                  exec python -u train_sae_res.py ;;

  # ---- LR sweep: random arm, plain SAE, 20M tokens (4e-4 comes from rand-full's t20M) ----
  rand-lr-low)    VARIANT=rand_all INIT_SEED=0 HF_REPO=$RAND_REPO MODE=full LR=1e-4 \
                  exec python -u train_sae_res.py ;;
  rand-lr-high)   VARIANT=rand_all INIT_SEED=0 HF_REPO=$RAND_REPO MODE=full LR=1e-3 \
                  exec python -u train_sae_res.py ;;

  *) echo "usage: bash randomized/launch_replication.sh <run>"
     echo "runs: trained-full trained-resid rand-full rand-resid rand-lr-low rand-lr-high"
     exit 1 ;;
esac
