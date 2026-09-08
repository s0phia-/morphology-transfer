"""How good is a low level, in the terms the hierarchy actually cares about?

eval_policy reports reward and episode length, and for an L2Low policy neither answers
the question you need before building on it. Its reward pays sparse_reward on EVERY step
spent inside epsilon (compute_reward, wrappers.py) and early_termination defaults False,
so one number conflates "arrived fast and parked" with "arrived late" with "hovered just
outside epsilon for the whole episode" - three very different policies, and only the
first is usable by a high level that hands out subgoals and expects them reached.

    python scripts/eval_low.py -p 09_08_26/MazeEnd_PointMass_UMaze_d1.0_L2Low_SAC_s1409_0
    python scripts/eval_low.py -p run_a run_b run_c        # compared side by side

Reports arrival rate, how long arrivals took, and how much of the episode was spent on
target. With more than one run it also prints a table ordered by delta_max, read from
each run's own params.json rather than from its directory name.

WHY THIS IS THE GATE BEFORE SPENDING ANYTHING ELSE
The High wrapper runs the low level for `skip` steps per decision and breaks early on
arrival; DSAC trains every target morphology to imitate this policy in skill space. A
low level that reaches its subgoal half the time makes the high level's action space
mean something different from what it says, and gives the discriminator a noisy target.
Both costs are paid 20 times over in step 3.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_transfer.utils.loader import ModelParams  # noqa: E402
from bot_transfer.utils.loader import load_from_name  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", "-p", required=True, nargs="+",
                   help="one or more low-level run directories, absolute or relative "
                        "to data/")
    p.add_argument("--episodes", "-e", type=int, default=200)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean. The High wrapper "
                        "calls model.predict() with its default (deterministic for SAC), "
                        "so the default here matches how the policy is actually used.")
    p.add_argument("--best", action="store_true", default=True)
    args = p.parse_args()

    rows = []
    for path in args.path:
        rows.append(evaluate(path, args))

    if len(rows) > 1:
        # delta_max is the high level's stride per decision, so the run to pick is not
        # the one with the best arrival rate - 1.0 wins that trivially - but the
        # largest delta_max still arriving reliably.
        print("\n" + "=" * 72)
        print("%-10s %10s %14s %12s %10s" % (
            "delta_max", "arrival", "median steps", "dwell/50", "run"))
        for r in sorted(rows, key=lambda r: (r["delta_max"] is None, r["delta_max"])):
            print("%-10s %9.1f%% %14s %12.1f %10s" % (
                "?" if r["delta_max"] is None else "%.1f" % r["delta_max"],
                100 * r["rate"],
                "-" if r["median"] is None else "%.0f" % r["median"],
                r["dwell"], r["short"]))
        print("\nPick the LARGEST delta_max still arriving reliably (>= ~0.8): it is the")
        print("high level's reach per decision, and every unit of it is capability the")
        print("hierarchy does not have to spend an extra decision to buy.")


def evaluate(path, args):
    model, env = load_from_name(path, best=args.best, load_env=True)
    # The wrapper stack is L2Low(+TimeLimit); both attributes reach through gym.Wrapper.
    epsilon = env.epsilon
    delta_max = None
    try:
        # What the run was actually trained with, not what its directory is called.
        params = ModelParams.load(path)
        dm = params["env_wrapper_args"].get("delta_max")
        delta_max = float(np.max(dm)) if dm is not None else None
    except Exception:
        pass

    arrived, steps_to_arrive, closest, dwell = [], [], [], []
    for _ in range(args.episodes):
        obs = env.reset()
        done, t, first, best = False, 0, None, np.inf
        inside = 0
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _reward, done, info = env.step(action)
            t += 1
            # L2Low publishes this per step; it is 1.0 exactly when the subgoal is
            # within epsilon, which is the event the High wrapper breaks on.
            hit = bool(info.get("success", 0.0))
            inside += hit
            if hit and first is None:
                first = t
        arrived.append(first is not None)
        if first is not None:
            steps_to_arrive.append(first)
        dwell.append(inside)

    n = len(arrived)
    rate = float(np.mean(arrived))
    print("\n{}".format(path))
    print("{} episodes, epsilon {}, delta_max {}, {} actions".format(
        n, epsilon, delta_max, "sampled" if args.stochastic else "deterministic"))
    print("  reached the subgoal      {:.1%}  ({}/{})".format(
        rate, int(np.sum(arrived)), n))
    if steps_to_arrive:
        print("  steps to first arrival   median {:.0f}   90th pct {:.0f}".format(
            np.median(steps_to_arrive), np.percentile(steps_to_arrive, 90)))
    print("  steps spent inside eps   mean {:.1f} of {} - this is what the reward pays "
          "for".format(np.mean(dwell), t))
    print("  A high level is only as good as this rate: it hands out one subgoal per "
          "\n  decision and assumes it lands. Below ~0.8 the hierarchy is being asked to "
          "\n  plan with a primitive that does not do what it is told.")
    env.close()
    return {"path": path, "short": os.path.basename(path.rstrip("/"))[-24:],
            "delta_max": delta_max, "rate": rate, "dwell": float(np.mean(dwell)),
            "median": float(np.median(steps_to_arrive)) if steps_to_arrive else None}


if __name__ == "__main__":
    main()
