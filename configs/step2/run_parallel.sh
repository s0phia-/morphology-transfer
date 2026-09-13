#!/bin/bash
# The whole step_maze_2 run in one launch: stages 1-2 once, then stages 3 and 4 per
# morphology, several walkers at a time.
#
#   nohup ./configs/step2/run_parallel.sh > step2.log 2>&1 &
#
#   JOBS=5           how many walkers to run at once (default 5)
#   GPUS="2 3"       spread walkers round-robin over these GPUs (default: none, CPU)
#   WALKERS="a b"    which walkers (default: every one in the asset manifest)
#   SKIP12=1         stages 1-2 are already done; go straight to the fan-out
#
# WHY STAGES 3 AND 4 ARE ONE UNIT PER WALKER
# Stage 4 needs only ITS OWN walker's stage 3, not all of them, so pairing them lets a
# morphology finish end to end while others are still on stage 3 - and a walker that
# fails stage 3 simply does not start stage 4, instead of the whole batch waiting.
#
# WHY A CONCURRENCY LIMIT
# Each job uses ~4.5 cores of TensorFlow threading. Twenty at once wants ninety on a
# shared box, at which point they contend and the total takes longer than running fewer.
# OMP_NUM_THREADS is pinned below for the same reason; raise JOBS rather than the threads
# if the box is quiet.
#
# Per-walker logs go to logs/step2/<walker>.log, since twenty jobs interleaved on one
# stdout is unreadable and the failures are what you will want to read.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

JOBS="${JOBS:-5}"
# Round-robin over GPUS, one device per walker. Stages 3 and 4 are independent per
# morphology, so this is the level the parallelism exists at - there is nothing to split
# WITHIN a run, since SB2's off-policy loop is single-env.
#
# Empty by default. requirements.txt pins plain tensorflow==1.15, not tensorflow-gpu, so
# on a stock install these are CPU jobs and CUDA_VISIBLE_DEVICES would do nothing.
# Check before relying on it:
#   python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
GPUS="${GPUS:-}"
read -r -a GPU_LIST <<< "$GPUS"
# TF1 takes the whole device unless told otherwise, so two walkers sharing a GPU would
# have the second fail to allocate. Growth mode lets them coexist.
[ ${#GPU_LIST[@]} -gt 0 ] && export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
LOGDIR="logs/step2"; mkdir -p "$LOGDIR"

MANIFEST="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_step_maze_2}/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: no manifest at $MANIFEST - export the assets first" >&2; exit 1
fi
WALKERS="${WALKERS:-$(python -c "import json;print(' '.join(json.load(open('$MANIFEST'))['walkers']))")}"
read -r -a WALKER_LIST <<< "$WALKERS"

if [ "${SKIP12:-0}" != "1" ]; then
    echo "=== stages 1-2 (serial, nothing else can start until these finish) ==="
    # Unlimited threads here: it is one job and the fan-out has not begun.
    if ! OMP_NUM_THREADS="" MKL_NUM_THREADS="" \
         SKIP3=1 SKIP4=1 ./configs/step2/run_all.sh 2>&1 | tee "$LOGDIR/stages_1_2.log"; then
        echo "ERROR: stages 1-2 failed - see $LOGDIR/stages_1_2.log" >&2; exit 1
    fi
fi

gpu_i=0
echo "=== stages 3+4 for ${#WALKER_LIST[@]} walkers, $JOBS at a time${GPUS:+ across GPUs $GPUS} ==="
for W in "${WALKER_LIST[@]}"; do
    # Polled rather than `wait -n`, which needs bash 4.3+. Five seconds of latency is
    # nothing against a ten-hour job, and it keeps the pipe full instead of draining to
    # zero between batches the way a plain `wait` would.
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    if [ ${#GPU_LIST[@]} -gt 0 ]; then
        dev="${GPU_LIST[$(( gpu_i % ${#GPU_LIST[@]} ))]}"; gpu_i=$(( gpu_i + 1 ))
    else
        dev=""
    fi
    (
        [ -n "$dev" ] && { export CUDA_VISIBLE_DEVICES="$dev"; echo "[$W] GPU $dev"; }
        echo "[$W] stage 3 $(date +%T)"
        SKIP1=1 SKIP2=1 SKIP4=1 WALKERS="$W" ./configs/step2/run_all.sh || \
            { echo "[$W] STAGE 3 FAILED - not starting stage 4"; exit 1; }
        echo "[$W] stage 4 $(date +%T)"
        SKIP1=1 SKIP2=1 SKIP3=1 WALKERS="$W" ./configs/step2/run_all.sh || \
            { echo "[$W] STAGE 4 FAILED"; exit 1; }
        echo "[$W] done $(date +%T)"
    ) > "$LOGDIR/$W.log" 2>&1 &
    echo "launched $W -> $LOGDIR/$W.log"
done
wait

echo
echo "=== summary ==="
for W in "${WALKER_LIST[@]}"; do
    printf '  %-34s %s\n' "$W" "$(tail -1 "$LOGDIR/$W.log" 2>/dev/null || echo 'no log')"
done
