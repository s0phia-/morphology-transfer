#!/bin/bash
# Step 3: ONE unimal's low level, trained to be indistinguishable from the source in
# skill space. DSAC = SAC on L2Low's own reward plus a discriminator bonus,
# clip(-discrim_coef * log(D + 1e-6), 0, discrim_clip) at discrim_weight.
#
# --low-level here is NOT a policy to run: trainer.py Case 3 routes it to the DSAC
# discriminator as the model being imitated. It is step 1's output.
#
#   ./configs/umaze/train_unimal_low.sh floor-1409-10-3-01-15-34-29 <step-1 path>
#
# Run once per morphology. --name is what keeps them apart - without it every walker
# writes MazeEnd_Unimal_L2Low_DSAC_<n> and the walker is nowhere in the path.
WALKER=${1:?usage: $0 <walker name> <path to step 1 low level> [extra train.py flags]}
LOW_LEVEL=${2:?usage: $0 <walker name> <path to step 1 low level> [extra train.py flags]}
# Both consumed, so the trailing "$@" forwards only what came after them.
shift 2

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
    --alg DSAC \
    --env MazeEnd_Unimal \
    --walker "$WALKER" \
    --env-wrapper L2Low \
    --asset-dir "$ASSET_DIR" \
    --low-level "$LOW_LEVEL" \
    --name "MazeEnd_Unimal_${WALKER}_L2Low_DSAC" \
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
    # 200, not the PointMass's 50. A unimal step is 0.005s x FRAME_SKIP 4 = 0.02s; the
    # PointMass's is 0.02s x 3 = 0.06s, so the same step count buys a THIRD of the
    # simulated time. On top of that these bodies are slow: measured over 100 walkers on
    # graph_transformer's maze tasks the median is ~0.5 units/s, and a subgoal drawn
    # uniformly on the +-delta_max square averages 1.53 units away - about 153 steps just
    # to arrive. At 50 the body reached the goal around step 46 of 50 and banked almost
    # none of the sparse reward, which is what a run topping out near 100 against a 1250
    # ceiling looks like. 200 leaves a median body ~47 steps of reward after arrival.
    # Slow bodies (p10, 0.38 units/s) need ~201 and still score near zero; raising this
    # further costs episodes, since the timestep budget is fixed.
    --time-limit 200 \
    --learning-rate 0.0003 \
    --batch-size 256 \
    --buffer-size 1000000 \
    --layers 256 256 \
    --timesteps 2000000 \
    --discrim-relative false \
    --discrim-include-skill true \
    --discrim-include-next-state true \
    --discrim-learning-rate 0.0005 \
    --discrim-train-freq 4 \
    --discrim-weight 0.5 \
    --discrim-decay true \
    --discrim-stop 0.6 \
    "$@"
