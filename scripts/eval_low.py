"""How good is a low level, in the terms the hierarchy actually cares about?

eval_policy reports reward and episode length, and for an L2Low policy neither answers
the question you need before building on it. Its reward pays sparse_reward on EVERY step
spent inside epsilon (compute_reward, wrappers.py) and early_termination defaults False,
so one number conflates "arrived fast and parked" with "arrived late" with "hovered just
outside epsilon for the whole episode" - three very different policies, and only the
first is usable by a high level that hands out subgoals and expects them reached.

    python scripts/eval_low.py -p 09_07_26/MazeEnd_PointMass_UMaze_L2Low_SAC_s1409_0

Reports arrival rate, how long arrivals took, and how close the misses got.

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

from bot_transfer.utils.loader import load_from_name  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", "-p", required=True,
                   help="a low-level run directory, absolute or relative to data/")
    p.add_argument("--episodes", "-e", type=int, default=200)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean. The High wrapper "
                        "calls model.predict() with its default (deterministic for SAC), "
                        "so the default here matches how the policy is actually used.")
    p.add_argument("--best", action="store_true", default=True)
    args = p.parse_args()

    model, env = load_from_name(args.path, best=args.best, load_env=True)
    # The wrapper stack is L2Low(+TimeLimit); both attributes reach through gym.Wrapper.
    epsilon = env.epsilon

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
    print("\n{} episodes, epsilon {}, {} actions".format(
        n, epsilon, "sampled" if args.stochastic else "deterministic"))
    print("  reached the subgoal      {:.1%}  ({}/{})".format(
        rate, int(np.sum(arrived)), n))
    if steps_to_arrive:
        print("  steps to first arrival   median {:.0f}   90th pct {:.0f}".format(
            np.median(steps_to_arrive), np.percentile(steps_to_arrive, 90)))
    print("  steps spent inside eps   mean {:.1f} of {} - this is what the reward pays "
          "for".format(np.mean(dwell), t))
    print("\n  A high level is only as good as this rate: it hands out one subgoal per "
          "\n  decision and assumes it lands. Below ~0.8 the hierarchy is being asked to "
          "\n  plan with a primitive that does not do what it is told.")


if __name__ == "__main__":
    main()
