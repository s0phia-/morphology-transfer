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

## Numbers that are pinned to graph_transformer, not guessed

    --skip 50           HRL.MANAGER_K
    --time-limit 100    HRL.MAX_EPISODE_STEPS 5000 / MANAGER_K = 100 decisions
    --epsilon 0.45      MAZE.GOAL_TOLERANCE - subgoal precision matched to true-goal
    --goal-range-*      the u_maze bounding box at CELL_SIZE 4.0: x +/-6, y +/-8

## Numbers that are a first guess, and the first things to move

    --delta-max 4.0     one cell. NOT matched to HRL.SUBGOAL_RADIUS, which on our side
                        is 0.0 = "span the maze" (31.24 on u_maze). Matching it would
                        make the high level's action space identical but train every
                        low level almost entirely on subgoals it cannot reach in 50
                        steps, so the sparse term would never fire. Raise it if the
                        high level looks starved for reach; lower it if low levels
                        plateau.
    --reset-prob 0.2    the low level must work ANYWHERE in the maze, so most subgoal
                        episodes continue from where the last one stopped rather than
                        teleporting back to the spawn cell.
    --timesteps         (1) and (3) count env steps; (2) counts DECISIONS, each up to
                        50 env steps, so 300k there is ~15M env steps against the
                        manager's 25M budget.
