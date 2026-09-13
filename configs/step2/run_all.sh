#!/bin/bash
# The umaze chain on step_maze_2 instead of u_maze.
#
#   ./configs/step2/run_all.sh [extra flags]              all four stages
#   SKIP3=1 SKIP4=1 ./configs/step2/run_all.sh            stages 1-2 only
#   SKIP1=1 SKIP2=1 SKIP4=1 WALKERS="<w>" ./configs/step2/run_all.sh
#
# Nothing about the stages changes - the envs read their geometry from the asset
# directory's manifest.json, so a different map is a different export. This wrapper
# exists because the run is launched about forty times across the three phases, and the
# two settings below have to be identical every time. Passing them by hand means one
# stage-3 job eventually gets u_maze's goal range on step_maze_2's geometry, which does
# not raise - the range is clipped to the observation space and the subgoal distribution
# quietly shifts.
#
# THE ASSETS ARE NOT IN THIS REPO. Export them from a graph_transformer checkout first:
#
#   python utils/export_maze_xml.py \
#       --cfg run_configs/maze_manager_step2_h030_nomorph.yaml \
#       --out-dir <this repo>/bot_transfer/envs/assets/unimal_step_maze_2 \
#       --walkers $(grep -hv '^#' agents/mlp_baseline/configs/sample_20_u_maze.txt | grep -v '^$') \
#       --pointmass-asset <this repo>/bot_transfer/envs/assets/point_mass.xml
#
# THE GOAL RANGE. step_maze_2's open cells span x [-6, 6] and y [-6, 6] at CELL_SIZE 4.0,
# against u_maze's y [-8, 8] - it is 5x5 where u_maze is 6x5. Computed from the map's own
# open cells plus a half cell, not guessed.
#
# ONE THING TO KNOW ABOUT THIS MAP. step_maze_2 is 2 cells over the step against 6
# around. The PointMass cannot climb, so stages 1-2 learn the detour and the transferred
# manager arrives biased toward it. Stage 4's job is then to discover that some bodies
# can take the short route. That is the experiment, but it is worth being deliberate
# about rather than surprised by.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ASSET_DIR="${ASSET_DIR:-$REPO/bot_transfer/envs/assets/unimal_step_maze_2}"

if [ ! -f "$ASSET_DIR/manifest.json" ]; then
    echo "ERROR: no manifest.json in $ASSET_DIR - export the assets first, see above" >&2
    exit 1
fi

exec "$REPO/configs/umaze/run_all.sh" \
    --goal-range-low -6.0 -6.0 --goal-range-high 6.0 6.0 "$@"
