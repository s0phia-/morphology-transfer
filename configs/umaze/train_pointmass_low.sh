#!/bin/bash
# Step 1: the SOURCE low level. SAC + L2Low on bot_transfer's PointMass, composed
# against graph_transformer's u_maze so it shares the targets' skill space.
# Reward is the wrapper's, not the env's: L2Low discards the maze's sparse +100 and
# pays -0.1*||subgoal - xy|| with +25 inside epsilon.
python scripts/train.py \
    --alg SAC \
    --env MazeEnd_PointMass_UMaze \
    --env-wrapper L2Low \
    --seed 1409 \
    --delta-max 4.0 4.0 \
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
