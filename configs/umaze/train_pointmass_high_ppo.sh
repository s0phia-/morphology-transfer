#!/bin/bash
# Step 2, PPO2 instead of SAC.
#
#   ./configs/umaze/train_pointmass_high_ppo.sh <path to step 1 low level>
#
# WHY: on u_maze's fixed (spawn, goal) pair the SAC manager scored exactly 0 over 300k
# and then 1.5M decisions. diagnose_high.py showed why: with random actions the agent
# travels ~105 units per episode but nets 1-2, spending its decisions pressed against
# the dividing wall at x=-2.5, and never gets closer than 2.49 against a 0.45 tolerance.
#
# PPO2's advantage here is not the update rule, it is --num-proc: SB2's SAC is written
# against a single env, PPO2 takes as many as the box has cores, and on a reward that
# fires once in hundreds of episodes throughput is the binding constraint. This is also
# the one respect in which graph_transformer's own manager differs most - it runs
# HRL.NUM_MANAGER_ENVS 32.
#
# ent_coef 0.05 is 5x PPO2's own default. With no reward ever seen the entropy term is
# the only thing keeping the policy from collapsing to a corner of the action space,
# which is what happened to SAC's auto-tuned coefficient (3e-9).
LOW_LEVEL=${1:?usage: $0 <path to step 1 low level> [extra train.py flags]}
shift

if [ -d "$PWD/data/$LOW_LEVEL" ]; then
  LOW_LEVEL="$PWD/data/$LOW_LEVEL"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_umaze}"

python scripts/train_wandb.py \
    --alg PPO2 \
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
    --num-proc 16 \
    --n-steps 256 \
    --nminibatches 4 \
    --noptepochs 8 \
    --ent-coef 0.05 \
    --lam 0.95 \
    --gamma 0.99 \
    --cliprange 0.2 \
    --learning-rate 0.0003 \
    --layers 256 256 \
    --timesteps 4000000 \
    "$@"
