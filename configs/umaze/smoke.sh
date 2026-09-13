#!/bin/bash
# All four stages end to end on ONE walker at 1/1000 scale, in a couple of minutes.
#
#   ./configs/umaze/smoke.sh [walker]
#
# Tests the wiring, not the learning: that each stage starts, that stage 2 can load
# stage 1, that stage 3 can load stage 1 as its discriminator source, and that stage 4
# can load BOTH stage 2's high level and stage 3's low level. Those loads are where the
# chain has actually broken - a path that no longer resolves, a flag that never reached
# python, an environment without wandb - and each of them cost hours to find at full
# scale.
#
# WHY THE BUDGETS ARE WHAT THEY ARE
#   --timesteps 500     enough for several episodes at every stage's own time limit, so
#                       the Monitor CSV is non-empty and final_model.zip gets written.
#   --eval-freq 1000000 never fires. The eval is 100 episodes, which at stage 3's
#                       --time-limit 200 is 20000 steps - forty times this whole run.
#                       best_model.zip is therefore absent and load() falls back to
#                       final_model.zip, which trainer.py always writes.
#
# Runtime is dominated by TensorFlow import and graph construction, roughly 25s per
# stage, so this cannot go much below two minutes however small the budgets get.
set -uo pipefail
WALKER="${1:-floor-1409-10-3-01-15-34-29}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

echo "smoke: $WALKER"
start=$(date +%s)
WALKERS="$WALKER" ./configs/umaze/run_all.sh --timesteps 500 --eval-freq 1000000
rc=$?
elapsed=$(( $(date +%s) - start ))

echo
echo "=============== smoke summary (${elapsed}s) ==============="
fail=0
check() {  # <label> <directory name prefix>
    local label="$1" prefix="$2" dir
    dir=$(ls -dt data/*/"$prefix"* 2>/dev/null | head -1)
    if [ -z "$dir" ]; then
        printf '  %-28s NO RUN DIRECTORY\n' "$label"; fail=1; return
    fi
    # A model is the real pass condition: the CSV can be non-empty on a run that then
    # died before saving, and every downstream stage loads a model rather than a CSV.
    local eps model
    eps=$(( $(wc -l < "$dir/0.monitor.csv" 2>/dev/null || echo 1) - 2 ))
    model=$([ -f "$dir/final_model.zip" ] || [ -f "$dir/best_model.zip" ] && echo ok || { fail=1; echo "NO MODEL"; })
    printf '  %-28s %-5s episodes=%s  %s\n' "$label" "$model" "$eps" "${dir#data/}"
}
check "1 pointmass low"  "MazeEnd_PointMass_UMaze_L2Low_SAC"
check "2 pointmass high" "MazeSample_PointMass_UMaze_High_SAC"
check "3 unimal low"     "MazeEnd_Unimal_${WALKER}_L2Low_DSAC"
check "4 unimal high"    "MazeSample_Unimal_${WALKER}_High_KLSAC"
echo
[ $fail -eq 0 ] && [ $rc -eq 0 ] && echo "PASS - the chain runs end to end" \
                                 || echo "FAIL - see above (run_all exit $rc)"
exit $(( fail || rc ))
