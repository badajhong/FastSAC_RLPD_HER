"""Dedicated PPO-BC DAgger training entrypoint.

The rollout, replay, checkpoint, and W&B implementation remains in train.py.
This file owns only DAgger-specific defaults and fail-fast validation so the
two entrypoints cannot develop different training semantics.
"""

import os

import hydra
from omegaconf import DictConfig

try:
    from .train import run_training
except ImportError:
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
EXPECTED_ALGO_NAME = "ppo_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.ppo_bc_dagger.PPOBCDaggerFinetune"
)


def bc_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Return the effective single-rank rollout schedule for this config."""
    num_envs = int(cfg.task.num_envs)
    train_every = int(cfg.algo.train_every)
    total_frames = int(cfg.total_frames)
    if num_envs < 1 or train_every < 1 or total_frames < 1:
        raise ValueError(
            "task.num_envs, algo.train_every, and total_frames must be positive"
        )
    frames_per_rollout = num_envs * train_every
    total_rollouts = total_frames // frames_per_rollout
    if total_rollouts < 1:
        raise ValueError("total_frames does not contain one complete rollout")
    decay_rollouts = int(cfg.algo.dagger_beta_decay_rollouts)
    beta_zero_rollouts = (
        max(total_rollouts - decay_rollouts, 0)
        if float(cfg.algo.dagger_beta_end) == 0.0
        else 0
    )
    return {
        "frames_per_rollout": frames_per_rollout,
        "total_rollouts": total_rollouts,
        "decay_rollouts": decay_rollouts,
        "beta_zero_rollouts": beta_zero_rollouts,
    }


def validate_bc_dagger_config(cfg: DictConfig) -> None:
    algo_name = cfg.algo.get("name")
    if algo_name != EXPECTED_ALGO_NAME:
        raise ValueError(
            "scripts/bc_dagger.py only supports "
            "algo=ppo_bc_dagger_finetune; got "
            f"algo.name={algo_name!r}. Use scripts/train.py for other algorithms."
        )
    algo_target = cfg.algo.get("_target_")
    if algo_target != EXPECTED_ALGO_TARGET:
        raise ValueError(
            "scripts/bc_dagger.py requires the PPO-BC DAgger implementation; "
            f"got algo._target_={algo_target!r}"
        )
    if cfg.algo.get("phase") != "finetune":
        raise ValueError("PPO-BC DAgger must run with algo.phase=finetune")
    if cfg.algo.get("vecnorm") != "eval":
        raise ValueError("PPO-BC DAgger must run with algo.vecnorm=eval")
    if not bool(cfg.algo.get("dagger_replay_raw_observations", False)):
        raise ValueError(
            "PPO-BC DAgger requires dagger_replay_raw_observations=true"
        )
    if cfg.get("checkpoint_path") is None:
        raise ValueError(
            "scripts/bc_dagger.py requires checkpoint_path pointing to the "
            "trained PPO teacher checkpoint"
        )
    schedule = bc_dagger_rollout_schedule(cfg)
    if (
        float(cfg.algo.dagger_beta_start) > 0.0
        and float(cfg.algo.dagger_beta_end) == 0.0
        and schedule["beta_zero_rollouts"] < 1
    ):
        raise ValueError(
            "dagger_beta_decay_rollouts must be smaller than the total rollout "
            "count so training includes a beta=0 student-execution phase"
        )


@hydra.main(config_path=CONFIG_PATH, config_name="bc_dagger", version_base=None)
def main(cfg: DictConfig):
    validate_bc_dagger_config(cfg)
    schedule = bc_dagger_rollout_schedule(cfg)
    print(
        "BC DAgger schedule: "
        f"{schedule['total_rollouts']} rollouts "
        f"({schedule['frames_per_rollout']} frames each), "
        f"beta decay={schedule['decay_rollouts']}, "
        f"beta=0 student execution={schedule['beta_zero_rollouts']}"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
