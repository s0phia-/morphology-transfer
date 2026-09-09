#!/bin/bash
# Step 1: the SOURCE low level. SAC + L2Low on bot_transfer's PointMass, composed
# against graph_transformer's u_maze so it shares the targets' skill space.
# Reward is the wrapper's, not the env's: L2Low discards the maze's sparse +100 and
# pays -0.1*||subgoal - xy|| with +25 inside epsilon.
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
    --env MazeEnd_PointMass_UMaze \
    --env-wrapper L2Low \
    --asset-dir "$ASSET_DIR" \
    --seed 1409 \
    --delta-max 2.0 2.0 \
    --epsilon 0.45 \
    --sparse-reward 25 \
    --reward-scale 0.1 \
    --relative false \
    --additive-goals true \
    --goal-range-low -6.0 -8.0 \
    --goal-range-high 6.0 8.0 \
    --reset-prob 0.2 \
    --reset-free-limit 10 \
    --time-limit 50 \
    --learning-rate 0.0003 \
    --batch-size 256 \
    --buffer-size 1000000 \
    --layers 256 256 \
    --timesteps 1000000 \
    "$@"
