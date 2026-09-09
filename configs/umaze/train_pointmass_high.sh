#!/bin/bash
# Step 2: the SOURCE high level, and the policy the method actually transfers. SAC +
# High over step 1's frozen low level. Its action is an xy delta bounded by delta_max;
# it runs the low level for --skip steps per decision and banks the maze env's own
# sparse +100. MazeSample (not MazeEnd) so the goal is redrawn every episode.
#
# --low-level is a path under data/ (or an absolute one); fill in what step 1 wrote.
LOW_LEVEL=${1:?usage: $0 <path to step 1 low level, e.g. 09_08_26/MazeEnd_PointMass_UMaze_L2Low_SAC_s1409_0> [extra train.py flags]}
# Consumed here, so the trailing "$@" forwards only what came AFTER it -
# train.py takes no positional arguments and rejects the leftover.
shift

# High (wrappers.py) and DSAC (dsac.py) both resolve a NON-ABSOLUTE --low-level against
# low_levels/, not data/ - the convention this repo's README describes for models you
# have deliberately promoted. A path straight out of data/ is what one actually has to
# hand, so accept it and make it absolute here rather than requiring a symlink dance.
if [ -d "$PWD/data/$LOW_LEVEL" ]; then
  LOW_LEVEL="$PWD/data/$LOW_LEVEL"
fi

# Which exported maze. One directory per map: u_maze has a single reset cell and a
# single goal cell, so every episode is the same (spawn, goal) pair and a sparse-reward
# high level gets no signal until it solves the hardest instance there is - measured, at
# 300k decisions: reward exactly 0, success 0, every episode hitting the 100-decision
# limit. u_maze_open has the SAME 21 wall cells, so identical physics and byte-identical
# walker XMLs, but 9 reset and 9 goal cells: 72 ordered pairs, many one cell apart.
# Override with the ASSET_DIR environment variable.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_umaze_open}"

python scripts/train.py \
    --alg SAC \
    --env MazeSample_PointMass_UMaze \
    --env-wrapper High \
    --asset-dir "$ASSET_DIR" \
    --low-level "$LOW_LEVEL" \
    --best true \
    --skip 50 \
    --time-limit 100 \
    --delta-max 2.0 2.0 \
    --epsilon 0.45 \
    --relative false \
    --goal-range-low -6.0 -8.0 \
    --goal-range-high 6.0 8.0 \
    --learning-rate 0.0003 \
    --batch-size 256 \
    --layers 256 256 \
    --timesteps 300000 \
    "$@"
