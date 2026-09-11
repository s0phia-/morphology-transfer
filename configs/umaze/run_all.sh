#!/bin/bash
# All three umaze steps in order, resolving each one's output directory for the next.
#
#   ./configs/umaze/run_all.sh
#
# WHY A CHAIN AT ALL
# Step 2 and step 3 both need step 1's low level as a PATH, and that path is not known
# until step 1 finishes: get_paths (bot_transfer/utils/loader.py) names a run directory
# from today's date plus an index that depends on what already exists. Typing it by hand
# between steps is how a stale or missing path reaches a launch - which is exactly what
# cost the previous attempt, when step 3 died on a FileNotFoundError for a low level that
# was no longer where the command said.
#
# WHY STEP 2 CANNOT BE SKIPPED WHEN STEP 1 IS REDONE
# The High wrapper loads the low level at CONSTRUCTION time (wrappers.py) - a saved high
# level cannot even rebuild its own env without the exact low level it trained against.
# So a new step 1 obsoletes the step 2 policy that sat on top of the old one, whatever
# its checkpoint still scores.
#
# ENV OVERRIDES
#   WALKERS   space-separated walker names for step 3 (default: every walker in the
#             asset manifest, which is the set the XMLs actually exist for)
#   SKIP1/2/3 set to 1 to skip a step, e.g. SKIP1=1 SKIP2=1 to resume at step 3 using
#             whatever step 1 output is already newest
#   ASSET_DIR forwarded to each step's own script
#
# Trailing arguments are forwarded to EVERY step, so use them only for things all three
# understand. Per-step flags belong in that step's own script.
set -euo pipefail

# Captured before step 3 word-splits the walker list, and expanded with the +"" guard so
# an empty set does not trip set -u on bash before 4.4.
EXTRA=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Newest data/<date>/<prefix>* holding a params.json. Sorted by mtime rather than by the
# date in the path: a run started before midnight and finished after it lands in the
# earlier directory, and lexical ordering would then pick the wrong one.
latest_run() {
    local prefix="$1" dir
    dir=$(ls -dt data/*/"$prefix"* 2>/dev/null | head -1 || true)
    if [ -z "$dir" ]; then
        echo "ERROR: no run directory matching data/*/$prefix*" >&2
        exit 1
    fi
    if [ ! -f "$dir/params.json" ]; then
        echo "ERROR: $dir has no params.json - did that step finish?" >&2
        exit 1
    fi
    # Relative to data/, which is what the step scripts' own $PWD/data/ branch expects.
    echo "${dir#data/}"
}

banner() { echo; echo "=============== $* ==============="; echo; }

if [ "${SKIP1:-0}" != "1" ]; then
    banner "STEP 1  source low level (PointMass)"
    ./configs/umaze/train_pointmass_low.sh ${EXTRA[@]+"${EXTRA[@]}"}
fi
LOW=$(latest_run MazeEnd_PointMass_UMaze_L2Low_SAC)
echo "step 1 low level: data/$LOW"

if [ "${SKIP2:-0}" != "1" ]; then
    banner "STEP 2  source high level (SAC over the frozen low level)"
    ./configs/umaze/train_pointmass_high.sh "$LOW" ${EXTRA[@]+"${EXTRA[@]}"}
    echo "step 2 high level: data/$(latest_run MazeSample_PointMass_UMaze_High_SAC)"
fi

if [ "${SKIP3:-0}" != "1" ]; then
    MANIFEST="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_umaze}/manifest.json"
    # From the manifest rather than a walker list file: it names the walkers whose XMLs
    # were actually exported, so a missing asset fails here instead of hours in.
    WALKERS="${WALKERS:-$(python -c "import json;print(' '.join(json.load(open('$MANIFEST'))['walkers']))")}"
    # Deliberate word split: WALKERS is a space-separated list, not one name.
    read -r -a WALKER_LIST <<< "$WALKERS"
    banner "STEP 3  one imitating low level per morphology (${#WALKER_LIST[@]} walkers, SERIAL)"
    echo "This is ${#WALKER_LIST[@]} x --timesteps back to back. Submit them as separate jobs if"
    echo "the scheduler allows it; this loop exists so an unattended run finishes, not"
    echo "because serial is the right way to spend the time."
    failed=()
    for W in "${WALKER_LIST[@]}"; do
        banner "STEP 3  $W"
        # Deliberately not under set -e: one morphology failing should not discard the
        # rest of the batch, and which ones failed is reported at the end.
        if ! ./configs/umaze/train_unimal_low.sh "$W" "$LOW" ${EXTRA[@]+"${EXTRA[@]}"}; then
            failed+=("$W")
            echo "WARNING: step 3 failed for $W - continuing" >&2
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo; echo "step 3 failed for ${#failed[@]} walker(s): ${failed[*]}" >&2
        exit 1
    fi
fi

banner "done"
