"""Staged PPO-BC DAgger entrypoint for a fresh PPO teacher checkpoint.

This module owns only the fail-fast source/schedule contract.  Rollout,
learning, replay, checkpoint, and W&B behavior remain in the shared trainer and
the PPO-BC DAgger policy implementation.
"""

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
        apply_bc_dagger_iteration_controls,
        validate_bc_dagger_config,
    )
    from .train import run_training
except ImportError:
    from bc_dagger import (
        EXPECTED_ALGO_NAME,
        EXPECTED_ALGO_TARGET,
        apply_bc_dagger_iteration_controls,
        validate_bc_dagger_config,
    )
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
CANONICAL_REPLAY_FILENAME = "teacher_replay_buffer.h5"
EXPECTED_PPO_ALGO_NAME = "ppo_vel"
EXPECTED_PPO_ALGO_TARGET = (
    "active_adaptation.learning.ppo.ppo_vel.PPOVEL"
)

def _validate_iteration(name: str, value, *, allow_zero: bool) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {requirement} integer")
    return int(value)


def validate_stage_controls(cfg: DictConfig) -> dict[str, int | float | str]:
    """Validate the complete phase budget and resolve its rollout total."""
    controls = {
        "joint_warmup_iterations": _validate_iteration(
            "joint_warmup_iterations",
            cfg.get("joint_warmup_iterations", None),
            allow_zero=False,
        ),
        "stage_cycles": _validate_iteration(
            "stage_cycles", cfg.get("stage_cycles", None), allow_zero=False
        ),
        "perception_iterations_per_cycle": _validate_iteration(
            "perception_iterations_per_cycle",
            cfg.get("perception_iterations_per_cycle", None),
            allow_zero=False,
        ),
        "actor_iterations_per_cycle": _validate_iteration(
            "actor_iterations_per_cycle",
            cfg.get("actor_iterations_per_cycle", None),
            allow_zero=False,
        ),
        "final_perception_iterations": _validate_iteration(
            "final_perception_iterations",
            cfg.get("final_perception_iterations", None),
            allow_zero=True,
        ),
        "final_actor_iterations": _validate_iteration(
            "final_actor_iterations",
            cfg.get("final_actor_iterations", None),
            allow_zero=True,
        ),
        "replay_q_calibration_iterations": _validate_iteration(
            "replay_q_calibration_iterations",
            cfg.get("replay_q_calibration_iterations", None),
            allow_zero=False,
        ),
    }
    mode = str(cfg.get("calibration_control_mode", "beta"))
    if mode != "beta":
        raise ValueError("calibration_control_mode must be beta")
    probability = float(
        cfg.get("calibration_teacher_probability", float("nan"))
    )
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(
            "calibration_teacher_probability must be finite and strictly "
            "between zero and one"
        )

    total_rollouts = (
        controls["joint_warmup_iterations"]
        + controls["stage_cycles"]
        * (
            controls["perception_iterations_per_cycle"]
            + controls["actor_iterations_per_cycle"]
        )
        + controls["final_perception_iterations"]
        + controls["final_actor_iterations"]
        + controls["replay_q_calibration_iterations"]
    )
    requested = cfg.get("bc_dagger_iterations", None)
    if requested is None:
        with open_dict(cfg):
            cfg.bc_dagger_iterations = total_rollouts
    else:
        requested = _validate_iteration(
            "bc_dagger_iterations", requested, allow_zero=False
        )
        if requested != total_rollouts:
            raise ValueError(
                "bc_dagger_iterations must equal the staged phase sum: "
                f"requested={requested}, phase_sum={total_rollouts}"
            )

    controls["calibration_control_mode"] = mode
    controls["calibration_teacher_probability"] = probability
    controls["total_rollouts"] = total_rollouts
    return controls


def stage_bc_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Resolve phase boundaries and the shared trainer's exact frame budget."""
    controls = validate_stage_controls(cfg)
    apply_bc_dagger_iteration_controls(cfg)

    joint_end = int(controls["joint_warmup_iterations"])
    cycle_span = int(
        controls["perception_iterations_per_cycle"]
        + controls["actor_iterations_per_cycle"]
    )
    cycles_end = joint_end + int(controls["stage_cycles"]) * cycle_span
    final_perception_end = cycles_end + int(
        controls["final_perception_iterations"]
    )
    final_actor_end = final_perception_end + int(
        controls["final_actor_iterations"]
    )
    calibration_end = final_actor_end + int(
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
    expected_frames = calibration_end * frames_per_rollout * world_size
    if int(cfg.total_frames) != expected_frames:
        raise RuntimeError(
            "staged rollout schedule disagrees with the shared frame budget"
        )
    return {
        "joint_start": 0,
        "joint_end": joint_end,
        "cycles_start": joint_end,
        "cycles_end": cycles_end,
        "cycle_span": cycle_span,
        "stage_cycles": int(controls["stage_cycles"]),
        "final_perception_start": cycles_end,
        "final_perception_end": final_perception_end,
        "final_actor_start": final_perception_end,
        "final_actor_end": final_actor_end,
        "calibration_start": final_actor_end,
        "calibration_end": calibration_end,
        "total_rollouts": calibration_end,
        "frames_per_rollout": frames_per_rollout,
        "total_frames": expected_frames,
    }


def _resolve_source_checkpoint(path) -> str:
    if path is None:
        raise ValueError(
            "scripts/stage_bc_dagger.py requires checkpoint_path pointing "
            "to a trained PPO teacher checkpoint"
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
        raise FileNotFoundError("Unable to resolve PPO teacher checkpoint")
    resolved = os.path.realpath(os.path.expanduser(os.fspath(resolved)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"PPO teacher checkpoint does not exist: {resolved}"
        )
    return resolved


def _checkpoint_config(checkpoint: Mapping) -> tuple[Mapping, Mapping, Mapping]:
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError("PPO teacher checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), Mapping):
        raise ValueError("PPO teacher checkpoint has no VecNorm state")
    source_cfg = checkpoint.get("cfg")
    if not isinstance(source_cfg, Mapping):
        raise ValueError("PPO teacher checkpoint has no saved config")
    source_algo = source_cfg.get("algo")
    source_task = source_cfg.get("task")
    if not isinstance(source_algo, Mapping) or not isinstance(
        source_task, Mapping
    ):
        raise ValueError(
            "PPO teacher checkpoint config must contain task and algo blocks"
        )
    return policy_state, source_algo, source_task


def _validate_source_checkpoint(
    checkpoint: Mapping, cfg: DictConfig
) -> Mapping:
    """Accept only a fresh train-phase PPO teacher, never staged continuation."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("PPO teacher checkpoint root must be a mapping")
    policy_state, source_algo, source_task = _checkpoint_config(checkpoint)

    if policy_state.get("training_algorithm") is not None:
        raise ValueError(
            "checkpoint_path must be a fresh PPO teacher; staged/BC-DAgger "
            "resume is intentionally unsupported"
        )
    if policy_state.get("last_phase") != "train":
        raise ValueError(
            "fresh staged BC-DAgger transfer requires PPO last_phase='train'"
        )
    if policy_state.get("bc_dagger_staging_state") is not None:
        raise ValueError(
            "staged BC-DAgger resume is intentionally unsupported"
        )
    if policy_state.get("bc_dagger_finalization_state") is not None:
        raise ValueError(
            "a BC-DAgger finalization checkpoint is not a PPO teacher"
        )

    if source_algo.get("name") != EXPECTED_PPO_ALGO_NAME:
        raise ValueError("checkpoint saved config is not a PPO velocity teacher")
    if source_algo.get("_target_") != EXPECTED_PPO_ALGO_TARGET:
        raise ValueError("checkpoint saved config has the wrong PPO target")
    if source_algo.get("phase") != "train":
        raise ValueError("checkpoint saved config must use algo.phase=train")
    if source_algo.get("enable_residual_distillation") is not True:
        raise ValueError(
            "PPO teacher checkpoint must use "
            "algo.enable_residual_distillation=true. BC-DAgger reconstructs "
            "the absolute privileged teacher action by adding ref_joint_pos "
            "to that residual head; an absolute-output PPO teacher would be "
            "added twice."
        )

    required_modules = {
        "actor",
        "actor_adapt",
        "encoder_priv",
        "adapt_module",
        "adapt_ema",
        "critic",
    }
    if bool(source_algo.get("use_object_adapt", False)):
        required_modules.update(("object_adapt", "object_adapt_ema"))
    missing = required_modules.difference(policy_state)
    if missing:
        raise ValueError(
            "PPO teacher checkpoint is missing required modules: "
            f"{sorted(missing)}"
        )

    requested_task = cfg.task.get("name", None)
    source_task_name = source_task.get("name", None)
    if (
        requested_task is not None
        and source_task_name is not None
        and str(requested_task) != str(source_task_name)
    ):
        raise ValueError(
            "requested task does not match PPO checkpoint task: "
            f"requested={requested_task!r}, saved={source_task_name!r}"
        )
    for name in ("use_object_adapt", "adapt_module", "latent_dim"):
        source_value = source_algo.get(name, None)
        runtime_value = cfg.algo.get(name, None)
        if (
            source_value is not None
            and runtime_value is not None
            and source_value != runtime_value
        ):
            raise ValueError(
                f"PPO teacher {name}={source_value!r} does not match "
                f"staged backend {runtime_value!r}"
            )
    return policy_state


def prepare_stage_bc_dagger(cfg: DictConfig) -> dict:
    """Validate the PPO source and install the exact staged runtime controls."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    controls = validate_stage_controls(cfg)

    if cfg.get("bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "stage_bc_dagger does not support bc_dagger_checkpoint resume; "
            "use a fresh PPO teacher through checkpoint_path"
        )
    if cfg.get("teacher_replay_buffer_path", None) is not None or cfg.algo.get(
        "teacher_buffer_path", None
    ) is not None:
        raise ValueError(
            "fresh staged BC-DAgger cannot restore an existing teacher replay"
        )

    source_path = _resolve_source_checkpoint(cfg.get("checkpoint_path", None))
    checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    policy_state = _validate_source_checkpoint(checkpoint, cfg)

    with open_dict(cfg):
        cfg.checkpoint_path = source_path
        cfg.bc_dagger_checkpoint = None
        cfg.bc_dagger_copy_teacher_replay = False
        cfg.teacher_replay_buffer_path = None
        cfg._bc_dagger_staging_source = True
        cfg._bc_dagger_stage = True
        cfg._bc_dagger_model_only_resume = False
        cfg._bc_dagger_finalization_source = False
        cfg._bc_dagger_finalize = False

        cfg.algo.teacher_buffer_path = None
        cfg.algo.save_teacher_buffer = True
        cfg.algo.dagger_staging_enabled = True
        cfg.algo.dagger_stage_joint_warmup_iterations = int(
            controls["joint_warmup_iterations"]
        )
        cfg.algo.dagger_stage_cycles = int(controls["stage_cycles"])
        cfg.algo.dagger_stage_perception_iterations = int(
            controls["perception_iterations_per_cycle"]
        )
        cfg.algo.dagger_stage_actor_iterations = int(
            controls["actor_iterations_per_cycle"]
        )
        cfg.algo.dagger_stage_final_perception_iterations = int(
            controls["final_perception_iterations"]
        )
        cfg.algo.dagger_stage_final_actor_iterations = int(
            controls["final_actor_iterations"]
        )
        cfg.algo.dagger_stage_calibration_iterations = int(
            controls["replay_q_calibration_iterations"]
        )
        cfg.algo.dagger_stage_calibration_control_mode = str(
            controls["calibration_control_mode"]
        )
        cfg.algo.dagger_stage_calibration_teacher_probability = float(
            controls["calibration_teacher_probability"]
        )

    schedule = stage_bc_dagger_rollout_schedule(cfg)
    return {
        "path": source_path,
        "source_last_iter": int(policy_state.get("last_iter", -1)),
        "schedule": schedule,
    }


def validate_stage_bc_dagger_config(cfg: DictConfig) -> None:
    """Enforce the staged entrypoint/backend guard after preparation."""
    validate_bc_dagger_config(cfg)
    controls = validate_stage_controls(cfg)
    if cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError("staged BC-DAgger requires ppo_bc_dagger")
    if cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError("staged BC-DAgger has the wrong policy target")
    if not bool(cfg.get("_bc_dagger_staging_source", False)):
        raise ValueError("staged BC-DAgger source guard is disabled")
    if not bool(cfg.get("_bc_dagger_stage", False)):
        raise ValueError("staged BC-DAgger trainer guard is disabled")
    if not bool(cfg.algo.get("dagger_staging_enabled", False)):
        raise ValueError("staged BC-DAgger backend is disabled")
    if bool(cfg.get("_bc_dagger_model_only_resume", False)) or cfg.get(
        "bc_dagger_checkpoint", None
    ) is not None:
        raise ValueError("staged BC-DAgger resume is unsupported")
    if str(cfg.algo.get("dagger_control_mode", "")) != "beta":
        raise ValueError("staged BC-DAgger currently requires beta control")
    if not math.isclose(
        float(cfg.algo.get("dagger_beta_start", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(cfg.algo.get("dagger_beta_end", float("nan"))),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("staged BC-DAgger requires beta_start=1 and beta_end=0")
    beta_zero = cfg.algo.get("dagger_beta_zero_iteration", None)
    if beta_zero != int(controls["joint_warmup_iterations"]):
        raise ValueError(
            "dagger_beta_zero_iteration must equal joint_warmup_iterations"
        )
    if int(cfg.algo.dagger_beta_decay_rollouts) != int(
        controls["joint_warmup_iterations"]
    ):
        raise ValueError("effective beta decay must end at the joint boundary")

    expected_algo_controls = {
        "dagger_stage_joint_warmup_iterations": controls[
            "joint_warmup_iterations"
        ],
        "dagger_stage_cycles": controls["stage_cycles"],
        "dagger_stage_perception_iterations": controls[
            "perception_iterations_per_cycle"
        ],
        "dagger_stage_actor_iterations": controls[
            "actor_iterations_per_cycle"
        ],
        "dagger_stage_final_perception_iterations": controls[
            "final_perception_iterations"
        ],
        "dagger_stage_final_actor_iterations": controls[
            "final_actor_iterations"
        ],
        "dagger_stage_calibration_iterations": controls[
            "replay_q_calibration_iterations"
        ],
        "dagger_stage_calibration_control_mode": controls[
            "calibration_control_mode"
        ],
        "dagger_stage_calibration_teacher_probability": controls[
            "calibration_teacher_probability"
        ],
    }
    mismatched = {
        name: (cfg.algo.get(name, None), expected)
        for name, expected in expected_algo_controls.items()
        if cfg.algo.get(name, None) != expected
    }
    if mismatched:
        raise ValueError(f"staged backend controls disagree: {mismatched}")
    if not bool(cfg.algo.get("save_teacher_buffer", False)):
        raise ValueError("staged BC-DAgger must create a final fresh H5")
    if cfg.algo.get("teacher_buffer_path", None) is not None or cfg.get(
        "teacher_replay_buffer_path", None
    ) is not None:
        raise ValueError("staged BC-DAgger cannot restore an old H5")
    stage_bc_dagger_rollout_schedule(cfg)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="stage_bc_dagger",
    version_base=None,
)
def main(cfg: DictConfig):
    prepared = prepare_stage_bc_dagger(cfg)
    validate_stage_bc_dagger_config(cfg)
    schedule = prepared["schedule"]
    print(
        "Staged BC-DAgger schedule: "
        f"PPO source iteration={prepared['source_last_iter']}, "
        f"joint=[{schedule['joint_start']},{schedule['joint_end']}), "
        f"cycles={schedule['stage_cycles']} x "
        f"({int(cfg.perception_iterations_per_cycle)} perception + "
        f"{int(cfg.actor_iterations_per_cycle)} actor), "
        f"cycle_region=[{schedule['cycles_start']},{schedule['cycles_end']}), "
        f"final_perception=[{schedule['final_perception_start']},"
        f"{schedule['final_perception_end']}), "
        f"final_actor=[{schedule['final_actor_start']},"
        f"{schedule['final_actor_end']}), "
        f"calibration=[{schedule['calibration_start']},"
        f"{schedule['calibration_end']}), total={schedule['total_rollouts']} "
        f"rollouts/{schedule['total_frames']} frames; beta reaches zero at "
        f"{int(cfg.algo.dagger_beta_decay_rollouts)}, final calibration "
        f"teacher_probability={float(cfg.calibration_teacher_probability):g}; "
        "staged resume disabled"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
