"""Dedicated PPO-BC DAgger + IQL-critic training entrypoint.

The rollout, replay, checkpoint, and W&B implementation remains in train.py.
This file owns only DAgger-specific defaults and fail-fast validation so the
two entrypoints cannot develop different training semantics.
"""

import math
import os

import hydra
import torch
from omegaconf import DictConfig, open_dict

from active_adaptation.utils.wandb import parse_checkpoint_path

try:
    from .helpers import find_local_teacher_replay
    from .train import run_training
except ImportError:
    from helpers import find_local_teacher_replay
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
EXPECTED_ALGO_NAME = "ppo_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.ppo_bc_dagger.PPOBCDaggerFinetune"
)
EXPECTED_TRAINING_ALGORITHM = "vaic_ppo_bc_dagger_student_iql_v2"
REQUIRED_RESUME_OPTIMIZERS = {
    "bc_optimizer",
    "q_optimizer",
    "iql_value_optimizer",
    "adapt_optimizer",
}
REQUIRED_RESUME_STATE = {
    "bc_update_count",
    "dagger_environment_steps",
    "dagger_rng_state",
    "iql_value_update_count",
    "next_iter",
    "q_rng_state",
    "q_update_count",
}


def _resolve_bc_dagger_checkpoint(
    path, *, download_replay: bool, replay_filename: str
) -> str:
    """Resolve a local/W&B resume checkpoint and optional immutable H5."""
    value = os.path.expanduser(os.fspath(path))
    if value.startswith("run:"):
        resolved = parse_checkpoint_path(
            value,
            download_replay=download_replay,
            replay_filename=replay_filename,
        )
    else:
        resolved = os.path.realpath(hydra.utils.to_absolute_path(value))
    if resolved is None or not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"BC-DAgger resume checkpoint does not exist: {resolved}"
        )
    return os.path.realpath(resolved)


def _validate_frozen_teacher_replay(
    source_path: str, policy_state: dict
) -> None:
    """Validate replay lineage without requiring the old snapshot iteration."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to validate the BC-DAgger teacher replay copy"
        ) from exc

    expected = policy_state.get(
        "teacher_replay_state",
        policy_state.get("frozen_teacher_replay_source_state"),
    )
    if not isinstance(expected, dict):
        raise ValueError(
            "BC-DAgger checkpoint has no teacher replay lineage metadata"
        )
    required = (
        "format",
        "format_version",
        "replay_id",
        "replay_observation_semantics",
        "vecnorm_fingerprint",
    )
    with h5py.File(source_path, "r") as replay:
        for name in required:
            expected_value = expected.get(name)
            actual_value = replay.attrs.get(name)
            if name == "format_version":
                equal = int(actual_value or 0) == int(expected_value or 0)
            else:
                equal = str(actual_value or "") == str(expected_value or "")
            if not equal:
                raise ValueError(
                    f"Teacher replay {name}={actual_value!r} does not match "
                    f"checkpoint lineage {expected_value!r}"
                )
        expected_action_clip = expected.get("action_clip")
        if expected_action_clip is None:
            backend = policy_state.get("dagger_backend_config")
            if isinstance(backend, dict):
                expected_action_clip = backend.get("dagger_action_clip")
        if expected_action_clip is not None:
            expected_action_clip = float(expected_action_clip)
            if (
                not math.isfinite(expected_action_clip)
                or expected_action_clip <= 0.0
            ):
                raise ValueError(
                    "Checkpoint teacher replay action clip is invalid"
                )
            actual_action_clip = replay.attrs.get("action_clip")
            if actual_action_clip is not None:
                if not math.isclose(
                    float(actual_action_clip),
                    expected_action_clip,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Teacher replay action_clip does not match checkpoint "
                        "lineage"
                    )
            else:
                # Legacy files did not record the support explicitly. Retain
                # compatibility only after proving every stored action lies in
                # the checkpoint's configured coordinates.
                actions = replay.get("actions")
                if actions is None:
                    raise ValueError("Teacher replay has no actions dataset")
                chunk_rows = 4096
                for start in range(0, int(actions.shape[0]), chunk_rows):
                    values = torch.as_tensor(
                        actions[start : start + chunk_rows]
                    )
                    if (
                        not torch.isfinite(values).all()
                        or (values.abs() > expected_action_clip).any()
                    ):
                        raise ValueError(
                            "Legacy teacher replay actions do not fit the "
                            "checkpoint action clip"
                        )


def prepare_bc_dagger_checkpoint(cfg: DictConfig) -> dict | None:
    """Configure model/optimizer continuation with an immutable old H5.

    ``checkpoint_path`` remains the shared trainer's internal model source.
    The dedicated alias prevents that generic path from treating the adjacent
    teacher H5 as mutable same-stage state.
    """
    requested = cfg.get("bc_dagger_checkpoint", None)
    if requested is None:
        return None
    if cfg.get("teacher_replay_buffer_path", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint resumes without teacher replay; remove "
            "teacher_replay_buffer_path so the existing H5 stays immutable"
        )
    if cfg.algo.get("teacher_buffer_path", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint resumes without teacher replay; remove "
            "algo.teacher_buffer_path so the existing H5 stays immutable"
        )

    copy_replay = bool(cfg.get("bc_dagger_copy_teacher_replay", True))
    replay_filename = str(
        cfg.algo.get("teacher_buffer_filename", "teacher_replay_buffer.h5")
    )
    if (
        not replay_filename
        or replay_filename in (".", "..")
        or os.path.basename(replay_filename) != replay_filename
    ):
        raise ValueError(
            "algo.teacher_buffer_filename must be a plain file basename"
        )
    resolved = _resolve_bc_dagger_checkpoint(
        requested,
        download_replay=copy_replay,
        replay_filename=replay_filename,
    )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, dict):
        raise ValueError("BC-DAgger resume checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), dict):
        raise ValueError("BC-DAgger resume checkpoint has no VecNorm state")
    if policy_state.get("training_algorithm") != EXPECTED_TRAINING_ALGORITHM:
        raise ValueError(
            "bc_dagger_checkpoint must be an IQL-v2 PPO-BC DAgger checkpoint"
        )
    optimizer_state = policy_state.get("optimizer_resume_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("BC-DAgger resume checkpoint has no optimizer state")
    missing_optimizers = REQUIRED_RESUME_OPTIMIZERS.difference(optimizer_state)
    if missing_optimizers:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing optimizer state: "
            f"{sorted(missing_optimizers)}"
        )
    required_modules = {"actor_adapt", "qnet", "qnet_target", "iql_value"}
    missing_modules = required_modules.difference(policy_state)
    if missing_modules:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing trained modules: "
            f"{sorted(missing_modules)}"
        )
    missing_state = REQUIRED_RESUME_STATE.difference(policy_state)
    if missing_state:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing continuation state: "
            f"{sorted(missing_state)}"
        )

    replay_source = None
    if copy_replay:
        replay_source = find_local_teacher_replay(
            resolved, replay_filename
        )
        if replay_source is None:
            raise FileNotFoundError(
                "bc_dagger_copy_teacher_replay=true requires a local "
                f"{replay_filename} beside the checkpoint or at its output "
                "root. Use a local checkpoint with its H5, upload the H5 to "
                "the source W&B run, or set "
                "bc_dagger_copy_teacher_replay=false."
            )
        _validate_frozen_teacher_replay(replay_source, policy_state)

    rollout_count = int(policy_state.get("dagger_rollout_count", -1))
    if rollout_count < 0:
        raise ValueError(
            "BC-DAgger resume checkpoint has no valid dagger_rollout_count"
        )
    environment_steps = int(
        policy_state.get(
            "dagger_environment_steps",
            rollout_count * int(cfg.algo.train_every),
        )
    )
    previous_source = cfg.get("checkpoint_path", None)
    if previous_source is not None and os.fspath(previous_source) != resolved:
        print(
            "BC DAgger resume: bc_dagger_checkpoint overrides the fresh PPO "
            f"checkpoint_path={previous_source}"
        )

    with open_dict(cfg):
        cfg.bc_dagger_checkpoint = resolved
        cfg.checkpoint_path = resolved
        cfg.bc_dagger_resume_rollout_count = rollout_count
        cfg.bc_dagger_resume_environment_steps = environment_steps
        cfg._bc_dagger_model_only_resume = True
        cfg._bc_dagger_teacher_replay_copy_source = replay_source
        cfg._bc_dagger_teacher_replay_copy_path = None
        cfg.teacher_replay_buffer_path = None
        cfg.algo.teacher_buffer_path = None
        cfg.algo.save_teacher_buffer = False

    return {
        "path": resolved,
        "rollout_count": rollout_count,
        "environment_steps": environment_steps,
        "teacher_replay_source": replay_source,
    }


def bc_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Return the effective additional rollout schedule for this process."""
    num_envs = int(cfg.task.num_envs)
    train_every = int(cfg.algo.train_every)
    total_frames = int(cfg.total_frames)
    if num_envs < 1 or train_every < 1 or total_frames < 1:
        raise ValueError(
            "task.num_envs, algo.train_every, and total_frames must be positive"
        )
    frames_per_rollout = num_envs * train_every
    additional_rollouts = total_frames // frames_per_rollout
    if additional_rollouts < 1:
        raise ValueError("total_frames does not contain one complete rollout")
    start_rollout = int(cfg.get("bc_dagger_resume_rollout_count", 0))
    if start_rollout < 0:
        raise ValueError("bc_dagger_resume_rollout_count must be non-negative")
    end_rollout = start_rollout + additional_rollouts
    decay_rollouts = int(cfg.algo.dagger_beta_decay_rollouts)
    beta_zero_rollouts = (
        max(end_rollout - max(start_rollout, decay_rollouts), 0)
        if float(cfg.algo.dagger_beta_end) == 0.0
        else 0
    )
    return {
        "frames_per_rollout": frames_per_rollout,
        "total_rollouts": additional_rollouts,
        "start_rollout": start_rollout,
        "end_rollout": end_rollout,
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
    if (
        cfg.get("checkpoint_path") is None
        and cfg.get("bc_dagger_checkpoint", None) is None
    ):
        raise ValueError(
            "scripts/bc_dagger.py requires checkpoint_path pointing to the "
            "trained PPO teacher, or bc_dagger_checkpoint for same-stage resume"
        )
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        if bool(cfg.algo.get("save_teacher_buffer", True)):
            raise ValueError(
                "bc_dagger_checkpoint must disable algo.save_teacher_buffer"
            )
        if (
            cfg.get("teacher_replay_buffer_path", None) is not None
            or cfg.algo.get("teacher_buffer_path", None) is not None
        ):
            raise ValueError(
                "bc_dagger_checkpoint cannot be combined with a teacher replay path"
            )
    schedule = bc_dagger_rollout_schedule(cfg)
    if (
        float(cfg.algo.dagger_beta_start) > 0.0
        and float(cfg.algo.dagger_beta_end) == 0.0
        and schedule["beta_zero_rollouts"] < 1
    ):
        raise ValueError(
            "the cumulative end rollout must exceed "
            "dagger_beta_decay_rollouts so training includes a beta=0 "
            "student-execution phase"
        )


@hydra.main(config_path=CONFIG_PATH, config_name="bc_dagger", version_base=None)
def main(cfg: DictConfig):
    resume = prepare_bc_dagger_checkpoint(cfg)
    validate_bc_dagger_config(cfg)
    schedule = bc_dagger_rollout_schedule(cfg)
    if resume is None:
        print(
            "BC DAgger schedule: "
            f"{schedule['total_rollouts']} rollouts "
            f"({schedule['frames_per_rollout']} frames each), "
            f"beta decay={schedule['decay_rollouts']}, "
            f"beta=0 student execution={schedule['beta_zero_rollouts']}"
        )
    else:
        print(
            "BC DAgger resume schedule: "
            f"start={schedule['start_rollout']}, "
            f"additional={schedule['total_rollouts']}, "
            f"end={schedule['end_rollout']}, "
            f"beta=0 student execution={schedule['beta_zero_rollouts']}; "
            "teacher H5 collection/write/snapshot disabled; "
            f"immutable local copy={'enabled' if resume['teacher_replay_source'] else 'disabled'}"
        )
    print(
        "IQL critic warm start: "
        f"expectile={float(cfg.algo.get('iql_expectile', 0.7)):g}, "
        f"target tau={float(cfg.algo.get('q_tau', 0.005)):g}, "
        "actor Q weighting=disabled (pure DAgger BC)"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
