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

python scripts/train.py \
    --alg DSAC \
    --env MazeEnd_Unimal \
    --walker "$WALKER" \
    --env-wrapper L2Low \
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
    --time-limit 50 \
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
