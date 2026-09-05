"""Train only VAIC perception from noisy privileged-Teacher rollouts.

The filename intentionally preserves the ``percetpion.py`` spelling used by
the experiment command.  Rollout/checkpoint/evaluation mechanics are delegated
to :mod:`scripts.train`; this entrypoint supplies exact iteration accounting and
strict Teacher-source validation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import active_adaptation as aa
from active_adaptation.learning.ppo.perception_only import (
    PERCEPTION_OBJECTIVE_SEMANTICS,
    ROLLOUT_SEMANTICS,
    TeacherRolloutPerceptionOnly,
)

try:
    from .train import run_training
except ImportError:
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.perception_only."
    "TeacherRolloutPerceptionOnly"
)
EXPECTED_ALGO_NAME = "teacher_rollout_perception_only"
REQUIRED_TEACHER_MODULES = (
    "actor",
    "actor_adapt",
    "encoder_priv",
    "critic",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
_ACTOR_STD_STATE_KEY = "module.0.module.2.module.actor_std"


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def apply_perception_iteration_controls(cfg: DictConfig) -> None:
    """Map user-facing rollout iterations to train.py's global-frame budget."""

    iterations = _positive_int("iteration", cfg.get("iteration"))
    num_envs = _positive_int("task.num_envs", cfg.task.num_envs)
    train_every = _positive_int("algo.train_every", cfg.algo.train_every)
    world_size = _positive_int("distributed world size", aa.get_world_size())
    if world_size != 1:
        raise ValueError(
            "Perception-only PPOVEL adaptation currently requires one process; "
            "its opt_adapt gradients are not distributed"
        )
    with open_dict(cfg):
        cfg.total_frames = iterations * num_envs * train_every * world_size
        cfg._perception_only_rollout_budget = iterations


def _as_mapping(value, name: str) -> Mapping:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"Teacher checkpoint {name} must be a mapping")
    return value


def validate_teacher_checkpoint(
    path: str | os.PathLike,
    *,
    expected_task_name: str | None = None,
) -> dict:
    """Fail before Isaac launch if Teacher action/target provenance is unclear."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Teacher checkpoint is not a file: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    checkpoint = _as_mapping(checkpoint, "top level")
    policy = _as_mapping(checkpoint.get("policy"), "policy")
    if policy.get("training_algorithm") is not None:
        raise ValueError(
            "Perception-only fresh training requires the original PPOVEL "
            "Teacher checkpoint, not a later staged algorithm checkpoint"
        )
    if policy.get("last_phase") != "train":
        raise ValueError(
            "Perception-only training requires a PPOVEL train-phase Teacher "
            "checkpoint (policy.last_phase='train')"
        )
    missing = sorted(name for name in REQUIRED_TEACHER_MODULES if name not in policy)
    if missing:
        raise ValueError(f"Teacher checkpoint is missing modules: {missing}")
    actor_state = _as_mapping(policy["actor"], "policy.actor")
    actor_std = actor_state.get(_ACTOR_STD_STATE_KEY)
    if not torch.is_tensor(actor_std) or actor_std.ndim != 1 or not actor_std.numel():
        raise ValueError("Teacher checkpoint actor has invalid actor_std state")
    if not torch.isfinite(actor_std).all() or not (actor_std > 0.0).all():
        raise ValueError("Teacher checkpoint actor_std must be finite and positive")

    source_cfg = _as_mapping(checkpoint.get("cfg"), "cfg")
    source_algo = _as_mapping(source_cfg.get("algo"), "cfg.algo")
    if source_algo.get("name") != "ppo_vel" or source_algo.get("_target_") != (
        "active_adaptation.learning.ppo.ppo_vel.PPOVEL"
    ):
        raise ValueError("Teacher source must be an original PPOVEL checkpoint")
    if source_algo.get("phase") != "train":
        raise ValueError("Teacher checkpoint cfg.algo.phase must be 'train'")
    if source_algo.get("enable_residual_distillation") is not True:
        raise ValueError(
            "Teacher source must use residual-distillation action semantics so "
            "rollout can reconstruct action = ref_joint_pos + residual sample"
        )
    if not isinstance(checkpoint.get("vecnorm"), Mapping):
        raise ValueError("Teacher checkpoint must contain VecNorm state")
    source_task = _as_mapping(source_cfg.get("task"), "cfg.task")
    source_task_name = source_task.get("name")
    if expected_task_name is not None and source_task_name != expected_task_name:
        raise ValueError(
            "Teacher checkpoint task does not match the runtime task: "
            f"source={source_task_name!r}, runtime={expected_task_name!r}"
        )

    return {
        "path": str(resolved),
        "last_iter": int(policy.get("last_iter", 0)),
        "source_noise_scale": float(actor_std.mean().item()),
        "action_dim": int(actor_std.numel()),
        "task_name": source_task_name,
    }


def validate_perception_training_config(
    cfg: DictConfig,
    *,
    expected_algo_target: str = EXPECTED_ALGO_TARGET,
    expected_algo_name: str = EXPECTED_ALGO_NAME,
    policy_cls=TeacherRolloutPerceptionOnly,
    entrypoint_name: str = "percetpion.py",
) -> dict:
    """Validate the shared Teacher-rollout stage and its exact policy type."""

    if cfg.algo.get("_target_") != expected_algo_target:
        raise ValueError(
            f"{entrypoint_name} requires algo._target_={expected_algo_target}"
        )
    if cfg.algo.get("name") != expected_algo_name:
        raise ValueError(
            f"{entrypoint_name} requires algo.name={expected_algo_name}"
        )
    # Reuse the policy's single authoritative invariant check.
    policy_cls._validate_config(cfg.algo)
    if int(cfg.algo.num_minibatches) != 8:
        raise ValueError(
            "Exact PPOVEL perception training requires algo.num_minibatches=8"
        )
    if int(cfg.algo.train_every) != 32:
        raise ValueError(
            "Exact PPOVEL perception training requires algo.train_every=32"
        )
    if not math.isclose(float(cfg.algo.lr), 3e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Exact PPOVEL perception training requires algo.lr=3e-4")
    if not math.isclose(
        float(cfg.algo.max_grad_norm), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "Exact PPOVEL perception training requires algo.max_grad_norm=1.0"
        )
    if int(cfg.algo.latent_dim) != 256 or not bool(
        cfg.algo.adapt_module_input_cmd
    ):
        raise ValueError(
            "Exact PPOVEL perception topology requires latent_dim=256 and "
            "adapt_module_input_cmd=true"
        )
    if cfg.get("vecnorm") != "eval":
        raise ValueError(
            "Teacher-rollout perception training requires top-level vecnorm=eval"
        )
    if not bool(cfg.task.get("enable_cameras", False)):
        raise ValueError(
            "Teacher-rollout perception training requires task.enable_cameras=true"
        )
    observations = cfg.task.get("observation", {})
    for key in ("depth", "object_", "object_geo_", "vel_command"):
        if key not in observations:
            raise ValueError(
                "Teacher-rollout perception task is missing observation group "
                f"{key!r}"
            )

    configured_path = cfg.get("checkpoint_path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("checkpoint_path must be an explicit local Teacher checkpoint")
    resolved = hydra.utils.to_absolute_path(configured_path)
    audit = validate_teacher_checkpoint(
        resolved,
        expected_task_name=str(cfg.task.name),
    )
    with open_dict(cfg):
        cfg.checkpoint_path = audit["path"]
        cfg._perception_teacher_checkpoint_audit = audit
    return audit


@hydra.main(config_path=CONFIG_PATH, config_name="percetpion", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    apply_perception_iteration_controls(cfg)
    audit = validate_perception_training_config(cfg)
    noise = float(cfg.algo.load_noise_scale)
    if not math.isfinite(noise):
        raise ValueError("algo.load_noise_scale must be finite")

    print(
        "Perception-only contract verified:\n"
        f"  Teacher checkpoint: {audit['path']}\n"
        f"  Teacher rollout: action = ref_joint_pos + Normal(residual, {noise:g})\n"
        "  actor_adapt influence on rollout: none\n"
        f"  perception initialization: {cfg.algo.perception_initialization}\n"
        "  optimized modules: temporal_depth_gru(depth_cnn), object_adapt, "
        "adapt_module\n"
        f"  perception objective: {PERCEPTION_OBJECTIVE_SEMANTICS}\n"
        f"  rollout semantics: {ROLLOUT_SEMANTICS}\n"
        f"  rollout iterations: {int(cfg.iteration)}"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
