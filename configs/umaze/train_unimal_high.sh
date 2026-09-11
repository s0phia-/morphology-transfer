#!/bin/bash
# Step 4: ONE morphology's high level, KL-finetuned from the source high level.
#
#   ./configs/umaze/train_unimal_high.sh <walker> <step-3 low level> <step-2 high level>
#
# trainer.py Case 5: passing --low-level AND --high-level with a kl_alg routes the source
# high level to KLSAC as `kl_model` - a reference the finetuned policy is pulled back
# toward, rather than a starting point it is free to walk away from. That is the half of
# the method the zero-shot composition does not test: step 2's manager is reused as the
# INITIALISATION and as the regulariser, and what this measures is how much adaptation a
# body needs on top of a manager that already solves the task for a point mass.
#
# WHAT IS THE AUTHORS' AND WHAT IS NOT
# Copied from their own configs/sawyer/train_high_kl.sh: --kl-coef 0.001, --kl-decay
# true, --learning-starts 100, --learning-rate 0.001, --timesteps 300000, --best true.
#
# Forced by this task rather than chosen: --env/--walker (MazeSample_Unimal is this
# project's env), the task block below, and --layers/--batch-size, which have to match
# the step 2 policy being loaded.
#
# Two deliberate departures, both recorded here so neither is mistaken for the method:
#   --alg KLSAC   their sawyer script says --alg SAC while passing --kl-coef. That
#                 cannot work: kl_coef/kl_decay exist only on KLSAC.__init__, and
#                 trainer.py routes to the KL case on `alg in kl_algs` = ["KLSAC"].
#                 With --alg SAC it would take Case 4 and then hand SAC a kl_coef it
#                 does not accept. KLSAC is what their script means.
#   --ent-coef    they pass none, leaving SAC to auto-tune. On u_maze's fixed goal pair
#                 auto-tuning collapsed to 3e-9 against a reward never once seen, which
#                 is why step 2 holds 0.2. Carried here so the finetune starts from the
#                 same exploration regime the policy was trained under.
#
# WHY THE TASK BLOCK MUST MATCH STEP 2 EXACTLY
# --skip, --time-limit, --delta-max, --epsilon and --goal-range-* define the skill space
# the source manager's weights mean something in. Changing any of them here loads a
# policy into a space it was not trained in, which does not raise - it just transfers
# nothing. They are copied verbatim from train_pointmass_high.sh.
#
# ONE KNOWN MISMATCH, AND IT IS NOT FIXED HERE
# --skip 50 is pinned to graph_transformer's HRL.MANAGER_K, but a decision is 50 * 0.02s
# = 1.0s for a unimal against 50 * 0.06s = 3.0s for the PointMass. The source manager
# learned to emit subgoals reachable within 3s of point-mass travel; a unimal gets a
# third of that time and is slower besides. Raising --skip here would change the skill
# space and break the paragraph above, so the honest options are to accept it, or to
# raise it in BOTH step 2 and here. diagnose_high.py's moved-vs-asked is what says
# whether it matters on these bodies.
WALKER=${1:?usage: $0 <walker> <step-3 low level> <step-2 high level> [extra flags]}
LOW_LEVEL=${2:?usage: $0 <walker> <step-3 low level> <step-2 high level> [extra flags]}
HIGH_LEVEL=${3:?usage: $0 <walker> <step-3 low level> <step-2 high level> [extra flags]}
shift 3

# High/DSAC resolve a non-absolute --low-level against low_levels/, and Case 4/5 resolve
# --high-level against high_levels/. A path straight out of data/ is what one actually
# has to hand, so make both absolute here rather than requiring a symlink dance.
if [ -d "$PWD/data/$LOW_LEVEL" ]; then
  LOW_LEVEL="$PWD/data/$LOW_LEVEL"
fi
if [ -d "$PWD/data/$HIGH_LEVEL" ]; then
  HIGH_LEVEL="$PWD/data/$HIGH_LEVEL"
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_umaze}"

python scripts/train_wandb.py \
    --alg KLSAC \
    --env MazeSample_Unimal \
    --walker "$WALKER" \
    --env-wrapper High \
    --asset-dir "$ASSET_DIR" \
    --low-level "$LOW_LEVEL" \
    --high-level "$HIGH_LEVEL" \
    --name "MazeSample_Unimal_${WALKER}_High_KLSAC" \
    --seed 1409 \
    --best true \
    --skip 50 \
    --time-limit 100 \
    --delta-max 2.0 2.0 \
    --epsilon 0.45 \
    --relative false \
    --goal-range-low -6.0 -8.0 \
    --goal-range-high 6.0 8.0 \
    --learning-rate 0.001 \
    --batch-size 256 \
    --layers 256 256 \
    --learning-starts 100 \
    --ent-coef 0.2 \
    --kl-coef 0.001 \
    --kl-decay true \
    --timesteps 300000 \
    "$@"
