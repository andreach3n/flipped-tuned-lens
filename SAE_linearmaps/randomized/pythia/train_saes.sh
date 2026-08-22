#!/usr/bin/env bash
# Train the two arms' SAEs with EleutherAI sparsify -- the paper's own training code.
#
# WHY sparsify DIRECTLY, and not ../../train_sae_res.py: delphi loads sparsify checkpoints
# natively. Every delphi number this project produced before 2026-08-19 was invalid because
# of convert_sae_to_sparsify.py (b_dec applied twice + a missing decoder-norm fold; see
# ../DELPHI_SETUP.md). Training in sparsify's own format deletes that entire failure class --
# there is no conversion step left to get wrong.
#
# Paper mapping (Heap et al., arXiv:2501.17727 S3):
#   TopK, expansion factor R=64  ->  d_sae = 64 * 2048 = 131072
#   k = 32
#   100M tokens  ->  48832 sequences x 2048 ctx = 99,991,552 tokens
#   decoder normalized each step  ->  sparsify's normalize_decoder default (True)
#   LR unspecified in the paper -> sparsify's auto rule, 2e-4/sqrt(131072/2^14) = 7.07e-5
#
# TWO DOCUMENTED DEVIATIONS, both forced or justified:
#   1. CORPUS. The paper uses RedPajama; togethercomputer/RedPajama-Data-1T-Sample has been
#      REMOVED from the Hub (404 with a valid token; so has cerebras/SlimPajama-627B, and the
#      surviving parent repo is a loading script datasets>=3 will not run -- the example in
#      sparsify's own README no longer executes). We use openwebtext, which every Gemma cell
#      in this project used, so the Pythia arms are directly comparable to the existing 2x2.
#   2. WARMUP. sparsify defaults to lr_warmup_steps=1000, but 100M tokens at the default batch
#      is only ~1526 steps -- LR would still be warming up two-thirds of the way through, then
#      decay linearly to zero. The paper does not state its warmup. We use ~5% of steps (76),
#      so as not to hand the "the random arm was just undertrained" objection a free win; that
#      objection already cost this project a full re-run cycle on Gemma.
#
# MEMORY, the one thing that will bite you: sparsify's FusedEncoder is fused in the BACKWARD
# pass only -- its forward still materializes the whole (N, 131072) pre-activation matrix. At
# batch_size 32 x ctx_len 2048 that is 65536 x 131072 x 4 bytes = 34 GB in fp32, before the
# model, the ~6.5 GB of Adam state, or anything else. MICRO_ACC_STEPS chunks it without
# changing the effective batch, so the run is mathematically identical: 8 -> ~4.3 GB. Raise it
# if you OOM; it costs time, not correctness.
#
# TWO GPUS: the arms are INDEPENDENT runs, so this is one process per card pinned with
# CUDA_VISIBLE_DEVICES -- NOT torchrun/DDP. DDP would shard one run's data across both cards;
# we want two different models trained at once. Note CUDA's default FASTEST_FIRST ordering
# means these device numbers need not match nvidia-smi's, which is harmless as long as the
# pinning is consistent across launches.
#
#   CUDA_VISIBLE_DEVICES=0 ARM=trained bash train_saes.sh
#   CUDA_VISIBLE_DEVICES=1 ARM=rand RAND_MODEL=/dev/shm/pythia1b_rand_s0 bash train_saes.sh
set -euo pipefail

ARM=${ARM:-trained}
LAYER=${LAYER:-8}
K=${K:-32}
R=${R:-64}
CTX=${CTX:-2048}
BATCH=${BATCH:-32}
MICRO_ACC_STEPS=${MICRO_ACC_STEPS:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-48832}          # 48832 * 2048 = 99,991,552 tokens ~ 100M
WARMUP=${WARMUP:-76}                         # ~5% of 48832/32 = 1526 steps
AUXK=${AUXK:-0.0}                            # sparsify default. See the dead-latent note below.

# OPTIMIZER MUST BE SET EXPLICITLY. eai-sparsify 1.3.3 defaults `optimizer` to **signum**
# (sign-SGD), not adam -- verified on the installed package, and it differs from the GitHub main
# branch, which defaults to adam. Three things follow silently if you leave it unset:
#   1. you train with sign-SGD, not Adam, which is not what Gao et al. / the paper's era used;
#   2. the LR rule becomes 5e-3/sqrt(d_sae/2^14) = 1.77e-3 here, 25x adam's 2e-4/sqrt(...) rule;
#   3. the signum branch sets `lr_schedulers = []`, so --lr_warmup_steps IS SILENTLY IGNORED.
# None of that errors. Pin it.
OPTIMIZER=${OPTIMIZER:-adam}
# LR unset -> sparsify's auto rule for the chosen optimizer (adam: 2e-4/sqrt(131072/2^14)
# = 7.07e-5). Set it to sweep; the value is tagged into RUN_NAME so runs cannot overwrite.
LR=${LR:-}
DATASET=${DATASET:-Skylion007/openwebtext}
SPLIT=${SPLIT:-train[:3%]}                   # ~235M tokens; enough for 100M with margin, and
                                             # small enough that chunk_and_tokenize is minutes.
                                             # sparsify tokenizes the WHOLE split before it
                                             # applies --max_examples, so do not pass all of it.
SAVE_DIR=${SAVE_DIR:-/dev/shm/saes}
# Tokenization is CPU-bound and runs BEFORE training. sparsify defaults this to cpu_count()//2,
# which oversubscribes badly if you launch both arms at once -- two runs would each grab half
# the box. 8 leaves room for the other arm.
DATA_PROC=${DATA_PROC:-8}
RAND_MODEL=${RAND_MODEL:-/dev/shm/pythia1b_rand_s0}

if [ "$ARM" = "trained" ]; then
  MODEL="EleutherAI/pythia-1b"
elif [ "$ARM" = "rand" ]; then
  MODEL="$RAND_MODEL"
  # THE GATE. Never train on an unverified random checkpoint -- a partly-randomized model
  # trains an SAE perfectly happily and produces a full set of plausible AUROC numbers.
  if [ ! -f "$MODEL/config.json" ]; then
    echo "no checkpoint at $MODEL -- run randomize_pythia.py first" >&2; exit 1
  fi
  echo ">>> verifying the random checkpoint before training on it"
  VARIANT=rand_all INIT_SEED=${INIT_SEED:-0} OUT_DIR="$MODEL" LAYER="$LAYER" \
    python -u "$(dirname "$0")/verify_randomization.py" || {
      echo "RANDOMIZATION GATE FAILED -- refusing to train" >&2; exit 1; }
else
  echo "ARM must be 'trained' or 'rand'" >&2; exit 1
fi

# LR goes in the name so a sweep cannot overwrite itself, and so the SAE_DIR you hand to
# check_saes.py / delphi says which point of the sweep it is.
LR_TAG=""
EXTRA=()
if [ -n "$LR" ]; then
  LR_TAG="_lr${LR}"
  EXTRA+=(--lr "$LR")
fi
RUN_NAME=${RUN_NAME:-pythia1b_${ARM}_L${LAYER}_R${R}_k${K}_100M_${OPTIMIZER}${LR_TAG}}
LOG_DIR=${LOG_DIR:-/dev/shm/logs}
LOG="$LOG_DIR/train_${ARM}_L${LAYER}_${OPTIMIZER}${LR_TAG}.log"
mkdir -p "$LOG_DIR"

# WANDB. sparsify logs by default (log_to_wandb=True) and reads WANDB_ENTITY/WANDB_PROJECT from
# the environment; run name is --run_name, and it uploads the full config. With no API key it
# would block on an interactive login prompt, which on a nohup'd pod launch looks like a hang --
# so fall back to offline explicitly rather than let that happen.
export WANDB_PROJECT=${WANDB_PROJECT:-pythia1b-heap-replication}
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ] && [ ! -f "$HOME/.netrc" ]; then
  echo ">>> no WANDB_API_KEY and no ~/.netrc -- falling back to WANDB_MODE=offline."
  echo "    For live logs: export WANDB_API_KEY=... (and optionally WANDB_ENTITY=...) first."
  export WANDB_MODE=offline
fi
echo ">>> wandb project=$WANDB_PROJECT run=$RUN_NAME mode=${WANDB_MODE:-online}"

echo ">>> ARM=$ARM MODEL=$MODEL -> $SAVE_DIR/$RUN_NAME"
echo ">>> d_sae = $R * 2048 = $((R * 2048)), k=$K, $((MAX_EXAMPLES * CTX)) tokens"
echo ">>> log -> $LOG"

python -m sparsify "$MODEL" "$DATASET" \
  --split "$SPLIT" \
  --hookpoints "layers.$LAYER" \
  --expansion_factor "$R" \
  --k "$K" \
  --ctx_len "$CTX" \
  --batch_size "$BATCH" \
  --micro_acc_steps "$MICRO_ACC_STEPS" \
  --max_examples "$MAX_EXAMPLES" \
  --optimizer "$OPTIMIZER" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  --lr_warmup_steps "$WARMUP" \
  --auxk_alpha "$AUXK" \
  --save_every 500 \
  --data_preprocessing_num_proc "$DATA_PROC" \
  --run_name "$RUN_NAME" \
  --save_dir "$SAVE_DIR" 2>&1 | tee "$LOG"

# ---- push to the Hub -------------------------------------------------------------------
# /dev/shm is RAM and / is rebuilt from the image; a pod stop destroys both. Anything not on
# the Hub is gone. PATH_IN_REPO is what keeps the two arms from overwriting each other.
if [ -n "${HF_REPO:-}" ]; then
  HERE_DIR="$(dirname "$0")"
  LOCAL="$SAVE_DIR/$RUN_NAME" HF_REPO="$HF_REPO" PATH_IN_REPO="saes/$RUN_NAME" \
    python -u "$HERE_DIR/hf_push.py"
  LOCAL="$LOG_DIR" HF_REPO="$HF_REPO" PATH_IN_REPO="logs" \
    python -u "$HERE_DIR/hf_push.py"
else
  echo ">>> HF_REPO unset -- checkpoint left only at $SAVE_DIR/$RUN_NAME (LOST ON POD STOP)."
fi

echo
echo ">>> trained. NOW RUN THE SAE HEALTH GATE before spending judge time:"
echo "    SAE_DIR=$SAVE_DIR/$RUN_NAME ARM=$ARM python -u $(dirname "$0")/check_saes.py"
echo
echo ">>> DEAD LATENTS ARE THE RISK HERE. auxk_alpha defaults to 0 in sparsify, so nothing"
echo "    revives dead latents, and 131072 latents in ~1526 steps is a cold start. delphi only"
echo "    scores latents with >=200 activations, so a dead-heavy dictionary silently biases"
echo "    which latents get sampled. check_saes.py reports alive%; if it is low, rerun with"
echo "    AUXK=0.03125 (Gao et al.'s 1/32) on BOTH arms -- never on just one."
