"""Staged finalization for a completed PPO-BC DAgger checkpoint.

This entrypoint deliberately hydrates the environment and PPO-BC DAgger
backend from the source checkpoint's saved configuration.  The checkpoint
backend contract is strict (and should remain so), while the finalization
schedule and its fixed calibration controller are runtime-only controls.

The shared trainer continues to own rollout collection, checkpointing, W&B,
and the final replay snapshot.  PPOBCDaggerFinetune dispatches its optimizer
work according to the ``dagger_finalize_*`` fields installed here.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path

try:
    from .bc_dagger import (
        EXPECTED_ALGO_NAME,
        EXPECTED_ALGO_TARGET,
        EXPECTED_CONTROL_SEMANTICS,
        EXPECTED_CRITIC_SEMANTICS,
        EXPECTED_TRAINING_ALGORITHM,
        REQUIRED_RESUME_OPTIMIZERS,
        REQUIRED_RESUME_STATE,
    )
    from .train import run_training
except ImportError:
    from bc_dagger import (
        EXPECTED_ALGO_NAME,
        EXPECTED_ALGO_TARGET,
        EXPECTED_CONTROL_SEMANTICS,
        EXPECTED_CRITIC_SEMANTICS,
        EXPECTED_TRAINING_ALGORITHM,
        REQUIRED_RESUME_OPTIMIZERS,
        REQUIRED_RESUME_STATE,
    )
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
CANONICAL_REPLAY_FILENAME = "teacher_replay_buffer.h5"
FINALIZATION_ITERATION_FIELDS = (
    "perception_consolidation_iterations",
    "actor_realignment_iterations",
    "perception_recheck_iterations",
    "replay_q_calibration_iterations",
)
CALIBRATION_CONTROL_MODES = ("beta",)


def _resolve_source_checkpoint(path) -> str:
    """Resolve a local/W&B checkpoint without downloading its stale H5."""
    if path is None:
        raise ValueError(
            "scripts/bc_dagger_finalize.py requires checkpoint_path pointing "
            "to a completed PPO-BC DAgger checkpoint"
        )
    value = os.path.expanduser(os.fspath(path))
    if value.startswith("run:"):
        resolved = parse_checkpoint_path(
            value,
            download_replay=False,
            replay_filename=CANONICAL_REPLAY_FILENAME,
        )
    else:
        resolved = hydra.utils.to_absolute_path(value)
    if resolved is None:
        raise FileNotFoundError("Unable to resolve BC-DAgger checkpoint")
    resolved = os.path.realpath(os.path.expanduser(os.fspath(resolved)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"BC-DAgger finalization checkpoint does not exist: {resolved}"
        )
    return resolved


def _as_config(value, *, name: str) -> DictConfig:
    if isinstance(value, DictConfig):
        copied = OmegaConf.create(
            OmegaConf.to_container(value, resolve=True)
        )
    elif isinstance(value, Mapping):
        copied = OmegaConf.create(dict(value))
        OmegaConf.resolve(copied)
    else:
        raise ValueError(f"BC-DAgger checkpoint has no valid {name}")
    return copied


def _validate_source_checkpoint(
    checkpoint: Mapping, requested_task_name: str | None
) -> tuple[DictConfig, Mapping]:
    """Fail before Isaac startup if the source cannot be finalized safely."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BC-DAgger checkpoint root must be a mapping")
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError("BC-DAgger checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), Mapping):
        raise ValueError("BC-DAgger checkpoint has no VecNorm state")
    source_cfg = _as_config(checkpoint.get("cfg"), name="saved config")
    if "task" not in source_cfg or "algo" not in source_cfg:
        raise ValueError(
            "BC-DAgger checkpoint config must contain task and algo blocks"
        )

    if policy_state.get("training_algorithm") != EXPECTED_TRAINING_ALGORITHM:
        raise ValueError(
            "checkpoint_path must be a SAC-critic-v6 PPO-BC DAgger checkpoint"
        )
    if policy_state.get("bc_dagger_finalization_state") is not None:
        raise ValueError(
            "checkpoint_path is already a BC-DAgger finalization checkpoint; "
            "this entrypoint currently starts only from the completed joint "
            "BC-DAgger checkpoint so it cannot accidentally mix or discard a "
            "partially collected finalization replay"
        )
    if policy_state.get("critic_learning_semantics") != (
        EXPECTED_CRITIC_SEMANTICS
    ):
        raise ValueError("BC-DAgger checkpoint critic semantics are incompatible")
    if policy_state.get("dagger_control_semantics") != (
        EXPECTED_CONTROL_SEMANTICS
    ):
        raise ValueError("BC-DAgger checkpoint control semantics are incompatible")
    if not isinstance(policy_state.get("q_backend_config"), Mapping):
        raise ValueError("BC-DAgger checkpoint lacks its Q backend contract")

    optimizer_state = policy_state.get("optimizer_resume_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("BC-DAgger checkpoint has no optimizer state")
    missing_optimizers = REQUIRED_RESUME_OPTIMIZERS.difference(optimizer_state)
    if missing_optimizers:
        raise ValueError(
            "BC-DAgger checkpoint is missing optimizer state: "
            f"{sorted(missing_optimizers)}"
        )
    missing_state = REQUIRED_RESUME_STATE.difference(policy_state)
    if missing_state:
        raise ValueError(
            "BC-DAgger checkpoint is missing continuation state: "
            f"{sorted(missing_state)}"
        )

    required_modules = {
        "actor",
        "actor_adapt",
        "encoder_priv",
        "adapt_module",
        "adapt_ema",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
    }
    if bool(source_cfg.algo.get("use_object_adapt", False)):
        required_modules.update(("object_adapt", "object_adapt_ema"))
    if bool(source_cfg.algo.get("use_depth", False)):
        required_modules.update(
            ("depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema")
        )
    missing_modules = required_modules.difference(policy_state)
    if missing_modules:
        raise ValueError(
            "BC-DAgger checkpoint is missing finalization modules: "
            f"{sorted(missing_modules)}"
        )

    if source_cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError("checkpoint saved config is not PPO-BC DAgger")
    if source_cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError("checkpoint saved config has the wrong policy target")
    if source_cfg.algo.get("phase") != "finetune":
        raise ValueError("checkpoint saved config must use phase=finetune")
    if source_cfg.algo.get("vecnorm") != "eval":
        raise ValueError("checkpoint saved config must use vecnorm=eval")

    source_task_name = source_cfg.task.get("name", None)
    if (
        requested_task_name is not None
        and source_task_name is not None
        and str(requested_task_name) != str(source_task_name)
    ):
        raise ValueError(
            "requested task does not match checkpoint task: "
            f"requested={requested_task_name!r}, saved={source_task_name!r}"
        )

    backend = policy_state.get("dagger_backend_config")
    if not isinstance(backend, Mapping):
        raise ValueError("BC-DAgger checkpoint lacks its actor backend contract")
    mismatched_backend = {
        name: (backend[name], source_cfg.algo.get(name, "<missing>"))
        for name in backend
        if source_cfg.algo.get(name, "<missing>") != backend[name]
    }
    if mismatched_backend:
        raise ValueError(
            "checkpoint saved config disagrees with dagger_backend_config: "
            f"{mismatched_backend}"
        )
    rollout_count = policy_state.get("dagger_rollout_count", -1)
    if isinstance(rollout_count, bool) or int(rollout_count) < 1:
        raise ValueError(
            "BC-DAgger finalization requires a completed joint DAgger rollout"
        )
    return source_cfg, policy_state


def _validate_iteration(name: str, value, *, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer")
    return int(value)


def validate_finalization_controls(cfg: DictConfig) -> dict[str, int | float | str]:
    values = {
        name: _validate_iteration(
            name,
            cfg.get(name, None),
            positive=(name == "replay_q_calibration_iterations"),
        )
        for name in FINALIZATION_ITERATION_FIELDS
    }
    mode = str(cfg.get("calibration_control_mode", "beta"))
    if mode not in CALIBRATION_CONTROL_MODES:
        raise ValueError("calibration_control_mode must be beta")
    probability = float(cfg.get("calibration_teacher_probability", float("nan")))
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(
            "calibration_teacher_probability must be finite and strictly "
            "between 0 and 1 so both Q replay sources are collected"
        )
    values["calibration_control_mode"] = mode
    values["calibration_teacher_probability"] = probability
    return values


def finalization_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    controls = validate_finalization_controls(cfg)
    perception_end = int(controls["perception_consolidation_iterations"])
    actor_end = perception_end + int(controls["actor_realignment_iterations"])
    recheck_end = actor_end + int(controls["perception_recheck_iterations"])
    calibration_end = recheck_end + int(
        controls["replay_q_calibration_iterations"]
    )
    num_envs = int(cfg.task.num_envs)
    train_every = int(cfg.algo.train_every)
    world_size = int(aa.get_world_size())
    if num_envs < 1 or train_every < 1 or world_size < 1:
        raise ValueError(
            "task.num_envs, algo.train_every, and distributed world size "
            "must be positive"
        )
    frames_per_rollout = num_envs * train_every
    with open_dict(cfg):
        cfg.total_frames = calibration_end * frames_per_rollout * world_size
    return {
        "perception_start": 0,
        "perception_end": perception_end,
        "actor_start": perception_end,
        "actor_end": actor_end,
        "recheck_start": actor_end,
        "recheck_end": recheck_end,
        "calibration_start": recheck_end,
        "calibration_end": calibration_end,
        "total_rollouts": calibration_end,
        "frames_per_rollout": frames_per_rollout,
        "total_frames": int(cfg.total_frames),
    }


def prepare_bc_dagger_finalization(cfg: DictConfig) -> dict:
    """Hydrate the exact source backend and install runtime phase controls."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    controls = validate_finalization_controls(cfg)
    requested_task_name = cfg.task.get("name", None)
    source_path = _resolve_source_checkpoint(cfg.get("checkpoint_path", None))
    checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    source_cfg, policy_state = _validate_source_checkpoint(
        checkpoint, requested_task_name
    )

    # Keep the user-facing process settings, but use the source checkpoint for
    # every environment/backend field that defines tensor shapes or semantics.
    source_task = OmegaConf.create(
        OmegaConf.to_container(source_cfg.task, resolve=True)
    )
    source_algo = OmegaConf.create(
        OmegaConf.to_container(source_cfg.algo, resolve=True)
    )
    with open_dict(cfg):
        cfg.task = source_task
        cfg.algo = source_algo
        cfg.checkpoint_path = source_path
        cfg.teacher_replay_buffer_path = None
        cfg.vecnorm = "eval"
        cfg._bc_dagger_finalization_source = True
        cfg._bc_dagger_finalize = True
        cfg._bc_dagger_model_only_resume = False
        cfg._bc_dagger_teacher_replay_copy_source = None
        cfg._bc_dagger_teacher_replay_copy_path = None
        cfg.bc_dagger_checkpoint = None
        cfg.bc_dagger_copy_teacher_replay = False

        cfg.algo.dagger_finalization_enabled = True
        cfg.algo.dagger_finalize_perception_iterations = int(
            controls["perception_consolidation_iterations"]
        )
        cfg.algo.dagger_finalize_actor_iterations = int(
            controls["actor_realignment_iterations"]
        )
        cfg.algo.dagger_finalize_recheck_iterations = int(
            controls["perception_recheck_iterations"]
        )
        cfg.algo.dagger_finalize_calibration_iterations = int(
            controls["replay_q_calibration_iterations"]
        )
        cfg.algo.dagger_finalize_calibration_control_mode = str(
            controls["calibration_control_mode"]
        )
        cfg.algo.dagger_finalize_calibration_teacher_probability = float(
            controls["calibration_teacher_probability"]
        )
        cfg.algo.save_teacher_buffer = True
        cfg.algo.teacher_buffer_filename = CANONICAL_REPLAY_FILENAME
        cfg.algo.teacher_buffer_path = None

    schedule = finalization_rollout_schedule(cfg)
    return {
        "path": source_path,
        "source_rollout_count": int(policy_state["dagger_rollout_count"]),
        "schedule": schedule,
    }


def validate_bc_dagger_finalize_config(cfg: DictConfig) -> None:
    if cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError("BC-DAgger finalization requires ppo_bc_dagger")
    if cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError("BC-DAgger finalization has the wrong policy target")
    if cfg.algo.get("phase") != "finetune":
        raise ValueError("BC-DAgger finalization requires phase=finetune")
    if cfg.algo.get("vecnorm") != "eval" or cfg.get("vecnorm") != "eval":
        raise ValueError("BC-DAgger finalization requires frozen VecNorm")
    if not bool(cfg.get("_bc_dagger_finalization_source", False)):
        raise ValueError("BC-DAgger finalization source guard is disabled")
    if not bool(cfg.get("_bc_dagger_finalize", False)):
        raise ValueError("BC-DAgger finalization trainer guard is disabled")
    if not bool(cfg.algo.get("dagger_finalization_enabled", False)):
        raise ValueError("BC-DAgger finalization backend is disabled")
    if not bool(cfg.algo.get("save_teacher_buffer", False)):
        raise ValueError("BC-DAgger finalization must create a fresh H5")
    if cfg.algo.get("teacher_buffer_path", None) is not None:
        raise ValueError("BC-DAgger finalization cannot restore an old H5")
    if cfg.get("teacher_replay_buffer_path", None) is not None:
        raise ValueError("BC-DAgger finalization cannot consume an old H5")
    finalization_rollout_schedule(cfg)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="bc_dagger_finalize",
    version_base=None,
)
def main(cfg: DictConfig):
    prepared = prepare_bc_dagger_finalization(cfg)
    validate_bc_dagger_finalize_config(cfg)
    schedule = prepared["schedule"]
    print(
        "BC-DAgger finalization: "
        f"source_rollouts={prepared['source_rollout_count']}, "
        f"perception=[0,{schedule['perception_end']}), "
        f"actor=[{schedule['actor_start']},{schedule['actor_end']}), "
        f"recheck=[{schedule['recheck_start']},{schedule['recheck_end']}), "
        f"calibration=[{schedule['calibration_start']},"
        f"{schedule['calibration_end']}), "
        f"control={cfg.calibration_control_mode}, "
        f"teacher_probability={float(cfg.calibration_teacher_probability):g}; "
        "old teacher H5 ignored, fresh teacher_replay_buffer.h5 enabled"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
