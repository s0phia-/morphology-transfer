#!/bin/bash
# Step 2: the SOURCE high level, and the policy the method actually transfers. SAC +
# High over step 1's frozen low level. Its action is an xy delta bounded by delta_max;
# it runs the low level for --skip steps per decision and banks the maze env's own
# sparse +100. MazeSample (not MazeEnd) so the goal is redrawn every episode.
#
# --low-level is a path under data/ (or an absolute one); fill in what step 1 wrote.
LOW_LEVEL=${1:?usage: $0 <path to step 1 low level, e.g. 09_07_26/MazeEnd_PointMass_UMaze_L2Low_SAC_s1409_0>}

python scripts/train.py \
    --alg SAC \
    --env MazeSample_PointMass_UMaze \
    --env-wrapper High \
    --low-level "$LOW_LEVEL" \
    --best true \
    --skip 50 \
    --time-limit 100 \
    --delta-max 4.0 4.0 \
    --epsilon 0.45 \
    --relative false \
    --goal-range-low -6.0 -8.0 \
    --goal-range-high 6.0 8.0 \
    --learning-rate 0.0003 \
    --batch-size 256 \
    --layers 256 256 \
    --timesteps 300000 \
    "$@"
