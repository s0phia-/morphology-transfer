"""Is the hierarchy moving the agent at all?

A maximum-entropy manager - uniform random subgoals, no learning - reaches u_maze's goal
in about a third of episodes when simulated on the real geometry with a low level that
walks straight to what it is given. A trained manager scoring 0 successes in 1500
episodes is therefore not an exploration failure; something between the manager's action
and the agent's feet is not working, and this says which part.

    python scripts/diagnose_high.py -p <the low level, e.g. 09_08_26/MazeEnd_..._0>

Drives the real High wrapper with RANDOM actions and reports, per decision: how far the
agent actually moved, how far it was asked to move, and whether the low level arrived.
Nothing here is trained, so anything wrong is wiring or dynamics.

WHAT THE NUMBERS MEAN
  moved/asked ~ 1.0   the low level is doing its job; if success is still 0 the problem
                      is downstream - the success test, `done` not propagating, the goal
                      being somewhere other than where it is drawn.
  moved/asked ~ 0     the low level is not moving the agent. Most likely its observation
                      inside High differs from the one it was trained on, or the policy
                      is being loaded without the normalisation it expects.
  moved > 0 but the agent never leaves its starting cell
                      it is moving and getting stuck - walls, or subgoals that are always
                      clipped back to where it already is.
"""
import argparse
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot_transfer.envs as E  # noqa: E402
from bot_transfer.utils.loader import ModelParams  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", "-p", required=True, help="the low level to drive with")
    p.add_argument("--episodes", "-e", type=int, default=20)
    p.add_argument("--env", default="MazeSample_PointMass_UMaze")
    p.add_argument("--asset-dir", default=None)
    args = p.parse_args()

    params = ModelParams.load(args.path)
    # The saved args are L2Low's, and High takes a DIFFERENT set - sparse_reward,
    # reward_scale, reset_prob and the rest belong to the low level's own reward and
    # mean nothing here. Filter to High's real signature rather than guessing which
    # overlap, so this keeps working if either wrapper changes.
    accepted = set(inspect.signature(E.High.__init__).parameters) - {"self", "env"}
    wargs = {k: v for k, v in params["env_wrapper_args"].items() if k in accepted}
    wargs["low_level"] = (args.path if args.path.startswith("/")
                          else os.path.join(os.getcwd(), "data", args.path))
    # High reads these off the low level's params when not given; set explicitly so this
    # reports on exactly what the training run uses.
    wargs.setdefault("skip", 50)
    dropped = sorted(set(params["env_wrapper_args"]) - accepted)
    print("High args:", {k: v for k, v in wargs.items() if k != "low_level"})
    print("dropped (L2Low's own):", dropped)

    env_kwargs = {}
    if args.asset_dir:
        env_kwargs["asset_dir"] = args.asset_dir
    base = vars(E)[args.env](**env_kwargs)
    env = E.High(base, **wargs)

    goal = np.asarray(base.center_goal, dtype=np.float64)
    print("goal", np.round(goal, 2), " spawn cell", base._spawn_cell,
          " tolerance", base.goal_tolerance, "\n")

    rows = []
    for ep in range(args.episodes):
        obs = env.reset()
        start = base.torso_xy().copy()
        moved = asked = 0.0
        best = float(np.linalg.norm(start - goal))
        arrivals = 0
        for d in range(100):
            before = base.torso_xy().copy()
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            after = base.torso_xy().copy()
            moved += float(np.linalg.norm(after - before))
            asked += float(np.linalg.norm(action))
            best = min(best, float(np.linalg.norm(after - goal)))
            arrivals += int(np.linalg.norm(after - before) > 0.1)
            if done:
                break
        rows.append((moved, asked, best, np.linalg.norm(base.torso_xy() - start), done))
        print("  ep %2d  moved %7.2f  asked %7.2f  ratio %5.2f  closest to goal %6.2f  "
              "net displacement %6.2f  reached %s"
              % (ep, moved, asked, moved / max(asked, 1e-9), best,
                 np.linalg.norm(base.torso_xy() - start), done))

    moved, asked, best, net, done = map(np.array, zip(*rows))
    print("\n%d episodes: moved/asked %.2f   closest approach %.2f (tolerance %.2f)"
          "   reached %d" % (len(rows), moved.sum() / max(asked.sum(), 1e-9),
                             best.min(), base.goal_tolerance, done.sum()))
    print("simulated expectation for a random manager on this maze: ~1 episode in 3")


if __name__ == "__main__":
    main()
