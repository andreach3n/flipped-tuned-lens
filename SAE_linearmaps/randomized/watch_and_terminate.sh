#!/usr/bin/env bash
# Wait for the training fleet to finish, archive the logs to HF, then TERMINATE the pod.
#
# Why a watcher: a 6xL40S pod bills all six GPUs until the pod is destroyed, and the four 100M runs
# finish at slightly different times overnight. This polls for "no train_sae_res.py processes left",
# then shuts everything down.
#
# SAFETY, in the order it matters:
#   1. Logs live on /workspace and die WITH the pod -- they are tarred and pushed to HF first.
#      (Every SAE artifact is already pushed by train_sae_res.py; the logs are the only loss.)
#   2. It refuses to terminate if NOT ONE run produced a _final.pt -- that pattern means an early
#      mass crash, and you want the box alive to debug it, not destroyed at 3am.
#   3. If termination is not possible (no runpodctl, no RUNPOD_API_KEY), it archives, says so
#      loudly, and leaves the pod UP. Failing to terminate costs money; terminating wrongly costs
#      the experiment.
#
# It waits for ANY train_sae_res.py, so if you later launch seed 1/2 runs on the free GPUs it will
# wait for those too -- which is what you want.
#
# Run it detached, from SAE_linearmaps/:
#     export HF_TOKEN=hf_xxx ARCHIVE_REPO=andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0
#     nohup bash randomized/watch_and_terminate.sh > /workspace/logs/watcher.log 2>&1 &
#
# Cancel it at any time with:  pkill -f watch_and_terminate
set -uo pipefail    # deliberately NOT -e: one failed poll must never kill the watcher

POLL=${POLL:-300}                       # seconds between checks
# WHAT counts as "still working". Default is the training fleet; set it to whatever you actually
# launched, e.g. WATCH_PAT=gauss_null.py for a null sweep. Getting this wrong is dangerous in one
# direction: a pattern that matches NOTHING makes the watcher think the box is idle and terminate
# it out from under a live run.
WATCH_PAT=${WATCH_PAT:-train_sae_res.py}
LOG_DIR=${LOG_DIR:-/workspace/logs}
# UNIQUE per invocation. This was a fixed "logs_100M_runs.tar.gz", so the second pod to run this
# script silently OVERWROTE the first pod's archived logs on HF (the 100M training logs were lost
# that way on 2026-08-12 and had to be recovered from a local copy). Never reuse the name.
ARCHIVE_NAME=${ARCHIVE_NAME:-logs_$(date -u +%Y%m%dT%H%M%SZ).tar.gz}
SCRATCH=${SCRATCH:-/dev/shm}
: "${HF_TOKEN:?export HF_TOKEN=hf_xxx (write scope) -- needed to archive the logs}"
: "${ARCHIVE_REPO:?export ARCHIVE_REPO=<an HF repo you own> -- where the log tarball goes}"

echo "[watcher] started $(date -u +%FT%TZ); polling every ${POLL}s for '$WATCH_PAT'"
echo "[watcher] pod=${RUNPOD_POD_ID:-<unknown>}"

# ---- 1. wait for the fleet to drain -------------------------------------------------------
while true; do
    n=$(pgrep -f "$WATCH_PAT" | wc -l | tr -d ' ')   # `pgrep -c` is not portable
    [ "${n:-0}" -eq 0 ] && break
    echo "[watcher] $(date -u +%FT%TZ)  $n process(es) matching '$WATCH_PAT' still alive"
    sleep "$POLL"
done
echo "[watcher] $(date -u +%FT%TZ)  fleet drained"

# ---- 2. what actually finished? -----------------------------------------------------------
finals=$(find "$SCRATCH" -maxdepth 2 -name '*_final.pt' 2>/dev/null | wc -l)
echo "[watcher] _final.pt artifacts on disk: $finals"
find "$SCRATCH" -maxdepth 2 -name '*_final.pt' 2>/dev/null | sed 's/^/  /'
grep -il "traceback\|out of memory" "$LOG_DIR"/*.log 2>/dev/null | sed 's/^/  [FAILED] /'

# ---- 3. archive the logs (they die with the pod; the SAEs are already on HF) ---------------
export ARCHIVE_NAME
python - <<'PY'
import os, tarfile, sys
try:
    from huggingface_hub import HfApi
    tar = "/tmp/run_logs.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(os.environ.get("LOG_DIR", "/workspace/logs"), arcname="logs")
    HfApi().upload_file(path_or_fileobj=tar, path_in_repo=os.environ["ARCHIVE_NAME"],
                        repo_id=os.environ["ARCHIVE_REPO"], repo_type="model")
    print(f"[watcher] logs archived -> {os.environ['ARCHIVE_REPO']}/{os.environ['ARCHIVE_NAME']}")
except Exception as e:
    print(f"[watcher] LOG ARCHIVE FAILED ({type(e).__name__}: {e})", file=sys.stderr)
    sys.exit(1)
PY
archived=$?

if [ "$finals" -eq 0 ]; then
    echo "[watcher] REFUSING TO TERMINATE: no _final.pt anywhere -- looks like a mass crash."
    echo "[watcher] pod left UP for debugging. Terminate by hand when you have looked."
    exit 1
fi
if [ "$archived" -ne 0 ]; then
    echo "[watcher] REFUSING TO TERMINATE: log archive failed, and logs do not survive the pod."
    exit 1
fi

# ---- 4. terminate ---------------------------------------------------------------------------
echo "[watcher] terminating pod in 60s -- pkill -f watch_and_terminate to abort"
sleep 60

if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "[watcher] runpodctl remove pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID" && exit 0
    echo "[watcher] runpodctl failed; trying the GraphQL API"
fi

if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
         -H 'Content-Type: application/json' \
         -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}"
    echo
    exit 0
fi

echo "[watcher] NO WAY TO TERMINATE (no runpodctl, no RUNPOD_API_KEY). Logs are safe on HF."
echo "[watcher] Terminate the pod from the RunPod dashboard."
exit 1
