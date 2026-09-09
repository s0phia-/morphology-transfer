#!/bin/bash
# Step 2: the SOURCE high level, and the policy the method actually transfers. SAC +
# High over step 1's frozen low level. Its action is an xy delta bounded by delta_max;
# it runs the low level for --skip steps per decision and banks the maze env's own
# sparse +100. MazeSample (not MazeEnd) so the goal is redrawn every episode.
#
# --low-level is a path under data/ (or an absolute one); fill in what step 1 wrote.
#
# --ent-coef IS HELD, NOT AUTO-TUNED, and --timesteps is 2M rather than 300k. The first
# attempt on this fixed-goal task got 300k decisions at reward exactly 0 with SAC's
# auto-tuned ent_coef collapsed to 3e-9: with a sparse +100 that has never once been
# reached, auto-tuning has no reason to keep entropy up, and the resulting deterministic
# policy explores nothing - so it can never find the reward that would tell it to do
# otherwise. A fixed coefficient keeps the manager stirring until something lands.
# "auto_0.2" starts here and adapts once returns exist, which is the thing to switch to
# if entropy stops being the binding constraint.
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

# Which exported maze. One directory per map. u_maze is a FIXED (spawn, goal) pair -
# one reset cell at (-4, 6), one goal cell at (4, 6), ~40 units apart around the U - and
# that is deliberate: it is the task graph_transformer's own manager is scored on, so
# keeping it is what makes the comparison like-for-like.
#
# It is also hard for a sparse-reward high level, and the first attempt got nowhere:
# 300k decisions at reward exactly 0, success 0, every episode hitting the 100-decision
# limit, with SAC's auto-tuned ent_coef collapsed to 3e-9 - a deterministic policy with
# nothing driving exploration. Hence --ent-coef below, and a longer --timesteps.
#
# unimal_umaze_open is the same 21 walls with 9 reset and 9 goal cells (72 pairs), kept
# exported as the fallback if entropy and time are not enough. ASSET_DIR overrides.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_umaze}"

python scripts/train_wandb.py \
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
    --ent-coef 0.2 \
    --timesteps 2000000 \
    "$@"
