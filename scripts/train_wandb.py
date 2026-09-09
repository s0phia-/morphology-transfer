"""scripts/train.py with wandb logging around it.

train.py itself has none - bot_transfer logs to stdout and, if --tensorboard is given,
to a local TensorBoard directory. Neither lets you see whether a run is going anywhere
while it is still running, which on a job that takes hours means dud runs are only
discovered when they finish.

    python scripts/train_wandb.py --alg SAC --env MazeSample_PointMass_UMaze ...

Same arguments as train.py, which it defers to entirely. Project defaults to
"bot_transfer" and the run name to the same name the run's own data directory gets
(ModelParams.get_save_name, i.e. --name if given, else env_wrapper_alg_seed) - override
either with WANDB_PROJECT / WANDB_RUN_NAME.

WHAT GETS LOGGED
bot_transfer/utils/trainer.py's TrainCallback logs to whatever run is live, so it stays
silent under plain train.py:

  train/ep_rew_mean       mean over the last 100 finished episodes, every 1000 steps
  train/ep_len_mean       ditto. On the High wrapper an episode that ends at the
                          decision limit rather than on arrival pins this to the limit,
                          which is the clearest single sign of a run that is not working
  train/episodes          episodes finished so far
  eval/mean_reward        the 100-episode deterministic eval, at every eval_freq
  eval/best_mean_reward   its running best, i.e. what best_model.zip corresponds to

The training series come from the Monitor CSV rather than from the algorithm's own
bookkeeping, which SB2 keeps privately and differently per algorithm; the CSV is the one
place every algorithm records episodes the same way.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # noqa: E402

from bot_transfer.utils.parser import train_parser, args_to_params  # noqa: E402
from bot_transfer.utils.trainer import train  # noqa: E402


def main():
    args = train_parser().parse_args()
    params = args_to_params(args)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "bot_transfer"),
        name=os.environ.get("WANDB_RUN_NAME") or params.get_save_name(),
        # dict(params) rather than params: ModelParams is a dict subclass and wandb
        # stores the nested arg dicts fine, but the subclass itself pickles oddly.
        config=dict(params),
    )
    try:
        train(params)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
