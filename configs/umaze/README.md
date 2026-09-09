# HDIMT on graph_transformer's u_maze

Three runs, in order. Every one of them needs the exported assets first - they are not
in this repo and cannot be built here:

    # in a graph_transformer checkout, on a machine with mujoco_py
    python utils/export_maze_xml.py \
        --cfg run_configs/maze_manager_umaze100.yaml \
        --out-dir <this repo>/bot_transfer/envs/assets/unimal_umaze \
        --walkers $(grep -hv '^#' agents/mlp_baseline/configs/sample_20_u_maze.txt | grep -v '^$') \
        --pointmass-asset <this repo>/bot_transfer/envs/assets/point_mass.xml

Re-run it whenever envs/tasks/maze_maps.py or the MAZE block changes; the XMLs are a
snapshot of that geometry, and nothing detects staleness.

## Fixed goals, and what the first attempt cost

u_maze has one reset cell and one goal cell, so every episode is the same (spawn, goal)
pair: spawn (-4, 6), goal (4, 6), ~40 units apart around the U. That is deliberate - it
is the task graph_transformer's own manager is scored on, so keeping it is what makes
this a like-for-like baseline rather than a different, easier problem.

It is also hard. The first attempt at the high level got 300k decisions at reward
exactly 0, success rate 0, every episode hitting the 100-decision limit, critic losses
at 1e-11 - not slow learning, no learning signal in existence - and SAC's auto-tuned
ent_coef collapsed to 3e-9. With a sparse +100 that has never once been reached,
auto-tuning has no reason to hold entropy up, and the deterministic policy that results
explores nothing, so it can never find the reward that would tell it to do otherwise.

The response is a held --ent-coef 0.2 and a 2M-decision budget rather than 300k. If that
is still not enough, two levers remain, in the order I would try them:

* delta_max. At 2.0 the manager needs about 26 correct decisions in a row before any
  reward exists, where graph_transformer's own manager runs SUBGOAL_RADIUS 0.0 -
  maze-spanning at 31.24 - and can place a subgoal on the goal in a single decision.
  Raising it changes the METHOD, which is what a baseline is allowed to differ in, and
  costs one low-level retrain since L2Low has to be trained on the subgoal distribution
  High will emit.
* u_maze_open. Same 21 walls, so identical physics and byte-identical walker XMLs, but 9
  reset and 9 goal cells: 72 ordered pairs, many one cell apart, which is the curriculum
  a sparse reward wants. Already exported as unimal_umaze_open; ASSET_DIR points the
  scripts at it. This changes the TASK, and makes it easier than the one being compared
  against, so it is the concession of last resort rather than the first thing to reach
  for.

## 1. Source low level - `train_pointmass_low.sh`

SAC + L2Low on the PointMass, in OUR maze at OUR scale. Learns "go to this xy", which
is the skill space everything downstream is defined in. Cheap; run it first and look at
it before spending anything on step 3.

## 2. Source high level - `train_pointmass_high.sh`

SAC + High on top of (1). Emits xy subgoals and is scored on the maze's own sparse
+100. This is the policy that gets REUSED across morphologies - it is the thing the
method claims transfers.

## 3. Target low levels - `train_unimal_low.sh <walker>`

DSAC + L2Low per morphology. Same L2Low reward as (1), plus a discriminator reward for
being indistinguishable from (1) in skill space. One run per walker; 20 of them for the
sampled set. Then run (2)'s high level on top of each, unchanged.

Every script forwards extra flags to scripts/train.py, and argparse takes the LAST
occurrence of a repeated flag - so a short smoke run needs no editing:

    bash configs/umaze/train_pointmass_low.sh --timesteps 20000 --eval-freq 5000

## Numbers that are pinned to graph_transformer, not guessed

    --skip 50           HRL.MANAGER_K
    --time-limit 100    HRL.MAX_EPISODE_STEPS 5000 / MANAGER_K = 100 decisions
    --epsilon 0.45      MAZE.GOAL_TOLERANCE - subgoal precision matched to true-goal
    --goal-range-*      the u_maze bounding box at CELL_SIZE 4.0: x +/-6, y +/-8

## Numbers that are a first guess, and the first things to move

    --delta-max 2.0     Was one cell (4.0); measured, and lowered. The PointMass masses
                        ~52 (r=0.5 sphere at density 100) against a peak actuator force
                        of 75, so a ~ 1.43 u/s^2, and a 50-step episode is 3s at
                        timestep 0.02 x FRAME_SKIP 3. Accelerate-then-brake - it has to
                        STOP inside epsilon - covers about 3.2 units. Per-axis uniform
                        delta_max 4.0 has mean radial distance 4 x 0.765 = 3.06, i.e.
                        the average subgoal sat exactly at the physical limit, and
                        scripts/eval_low.py measured the predicted result: 48% arrival,
                        median arrival at step 32 of 50. At 2.0 the mean is 1.53 and the
                        max 2.83, both comfortably inside the window.

                        Still NOT matched to HRL.SUBGOAL_RADIUS, which on our side is
                        0.0 = "span the maze" (31.24 on u_maze). Matching it would make
                        the high level's action space identical at the cost of training
                        every low level on subgoals it cannot reach, which is the same
                        failure this measurement just found, only worse.

                        It must be the SAME in all three scripts: High reads delta_max
                        off the low level's params when not given, and DSAC's targets
                        have to share the source's skill space.
    --reset-prob 0.2    the low level must work ANYWHERE in the maze, so most subgoal
                        episodes continue from where the last one stopped rather than
                        teleporting back to the spawn cell.
    --timesteps         (1) and (3) count env steps; (2) counts DECISIONS, each up to
                        50 env steps, so 300k there is ~15M env steps against the
                        manager's 25M budget.
