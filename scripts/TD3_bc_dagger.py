"""Dedicated distributional-TD3 + Teacher-BC DAgger entrypoint.

The method is C51 distributional TD3 with joint-normalized raw-action
Teacher BC. Collection, environment stepping, checkpoint writing,
and evaluation remain owned by :mod:`scripts.train`; this module provides the
new method's Hydra surface and fail-fast fresh-source validation.  Version 3
keeps raw perception replay and introduces a fresh direct-raw-action lineage;
it intentionally does not resume an older TD3 lineage.

The Student Actor uses the original PPOVEL unbounded raw joint-command
coordinates.  Teacher BC and Critic inputs are normalized with the same
per-joint nominal center/scale, while environment execution remains raw: no
tanh/atanh transform or final action clip is part of this backend.

``PPOConfig`` contributes a few legacy exploration-objective configuration
keys which are retained only because the Actor/checkpoint topology is locked.
They are inert for this method.  New stochastic-policy configuration, density
objectives, and temperature configuration are rejected here.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

import hydra
import torch
from omegaconf import DictConfig, open_dict

import active_adaptation as aa
from active_adaptation.learning.ppo.td3_bc_dagger import (
    ACTION_CONTRACT_SEMANTICS,
    ACTOR_BACKEND,
    ACTOR_LEARNING_SEMANTICS,
    CHECKPOINT_VERSION,
    CRITIC_SEMANTICS,
    TRAINING_ALGORITHM,
)

try:
    from .train import run_training
except ImportError:
    from train import run_training

EXPECTED_ACTION_CONTRACT_SEMANTICS = ACTION_CONTRACT_SEMANTICS


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

EXPECTED_ALGO_NAME = "td3_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.td3_bc_dagger.DistributionalTD3TeacherBC"
)
EXPECTED_ACTOR_LEARNING_SEMANTICS = ACTOR_LEARNING_SEMANTICS
EXPECTED_CRITIC_SEMANTICS = CRITIC_SEMANTICS
EXPECTED_TRAINING_ALGORITHM = TRAINING_ALGORITHM
EXPECTED_CHECKPOINT_VERSION = CHECKPOINT_VERSION
EXPECTED_ACTOR_BACKEND = ACTOR_BACKEND
EXPECTED_ACTOR_IN_KEYS = (
    "command",
    "policy",
    "object_",
    "priv",
    "object_geo_",
    "vel_command",
    "depth",
)
EXPECTED_REPLAY_RAW_OBSERVATION_KEYS = (
    "vel_command",
    "policy",
    "priv",
    "command",
    "depth",
)
REQUIRED_PRETRAINED_PERCEPTION_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
PPOVEL_TRAIN_PHASE_FRESH_DEPTH_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
)
PPOVEL_TRAIN_PHASE_PARTIAL_PERCEPTION_MODULES = (
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
EXPECTED_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
)

DAGGER_BACKEND_CONFIG_FIELDS = (
    "dagger_control_mode",
    "dagger_safe_takeover_rms",
    "dagger_safe_release_rms",
    "dagger_safe_min_teacher_steps",
    "dagger_safe_zero_iteration",
    "dagger_beta_start",
    "dagger_beta_end",
    "dagger_beta_decay_rollouts",
    "dagger_seed",
    "dagger_bc_lr",
    "dagger_actor_huber_delta",
    "dagger_buffer_capacity",
    "dagger_buffer_device",
    "dagger_batch_size",
    "teacher_prefill_max_rollouts",
    "teacher_actor_replay_fraction",
    "teacher_perception_replay_fraction",
    "failure_phase_teacher_fraction",
    "failure_phase_lookback_steps",
    "failure_phase_samples_per_failure",
    "failure_phase_num_bins",
    "perception_replay_burn_in",
    "perception_encode_microbatch_size",
    "teacher_perception_batch_size",
    "teacher_perception_warmup_steps",
    "perception_depth_codec",
    "load_pretrained_perception",
    "perception_checkpoint_path",
    "train_perception",
    "eta_td3",
    "lambda_bc",
    "policy_delay",
    "target_policy_noise_std",
    "target_policy_noise_clip",
    "collector_exploration_noise_std",
    "collector_exploration_noise_clip",
    "td3_learning_starts",
    "q_hidden_dim",
    "q_num_atoms",
    "q_v_min",
    "q_v_max",
    "q_layer_norm",
    "q_action_fusion",
    "q_action_coordinates",
    "q_normalize_actions",
    "q_action_input_gain",
    "q_lr",
    "q_weight_decay",
    "q_seed",
    "q_tau",
    "q_max_grad_norm",
    "q_batch_size",
    "q_updates_per_rollout",
    "q_teacher_replay_ratio",
    "q_teacher_buffer_capacity",
    "save_teacher_buffer",
)

# These fields are inherited from PPOConfig solely to preserve construction
# and checkpoint compatibility.  DistributionalTD3TeacherBC must not read
# them.  All other objective/density keys matching the guards below fail fast.
INERT_PPO_COMPATIBILITY_FIELDS = {
    "entropy_coef_start",
    "entropy_coef_end",
    "entropy_decay_iters",
    "init_noise_scale",
    "load_noise_scale",
}


def _require_single_process_execution() -> None:
    """Reject distributed execution before constructing Isaac or the policy."""
    world_size = aa.get_world_size()
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise RuntimeError("distributed world size must be an integer")
    if bool(aa.is_distributed()) or world_size != 1:
        raise RuntimeError(
            "TD3-BC DAgger currently supports exactly one training process; "
            "distributed/multi-GPU execution is rejected because its custom "
            "Actor, Critic, and perception gradients are not synchronized"
        )


def _positive_int(name: str, value, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {requirement} integer")
    return int(value)


def _finite_nonnegative(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _finite_positive(name: str, value) -> float:
    value = _finite_nonnegative(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _finite_fraction(name: str, value) -> float:
    value = _finite_nonnegative(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _validate_perception_training_controls(cfg: DictConfig) -> str | None:
    """Validate and canonicalize a full or PPOVEL-style perception warm start."""
    load_pretrained = cfg.algo.get("load_pretrained_perception", False)
    train_perception = cfg.algo.get("train_perception", True)
    if not isinstance(load_pretrained, bool):
        raise ValueError("algo.load_pretrained_perception must be boolean")
    if not isinstance(train_perception, bool):
        raise ValueError("algo.train_perception must be boolean")

    configured_path = cfg.algo.get("perception_checkpoint_path", None)
    if not load_pretrained:
        if configured_path is not None:
            raise ValueError(
                "algo.perception_checkpoint_path must be null when "
                "algo.load_pretrained_perception=false"
            )
        if not train_perception:
            raise ValueError(
                "algo.train_perception=false requires "
                "algo.load_pretrained_perception=true; freezing a freshly "
                "initialized perception stack is unsupported"
            )
        return None

    if configured_path is None:
        raise ValueError(
            "algo.load_pretrained_perception=true requires an explicit local "
            "algo.perception_checkpoint_path"
        )
    try:
        raw_path = os.fspath(configured_path)
    except TypeError as exc:
        raise ValueError(
            "algo.perception_checkpoint_path must be a local filesystem path"
        ) from exc
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            "algo.perception_checkpoint_path must be a non-empty local filesystem path"
        )
    raw_path = os.path.expanduser(raw_path)
    if raw_path.startswith("run:") or "://" in raw_path:
        raise ValueError(
            "algo.perception_checkpoint_path must be a local filesystem path"
        )
    resolved = os.path.realpath(hydra.utils.to_absolute_path(raw_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"pretrained perception checkpoint does not exist: {resolved}"
        )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "pretrained perception checkpoint must contain a top-level mapping"
        )
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError(
            "pretrained perception checkpoint must contain a policy mapping"
        )

    invalid_modules = [
        name
        for name in REQUIRED_PRETRAINED_PERCEPTION_MODULES
        if name in policy_state and not isinstance(policy_state[name], Mapping)
    ]
    if invalid_modules:
        raise ValueError(
            "pretrained perception checkpoint contains invalid perception "
            f"module mappings: {invalid_modules}"
        )

    present_modules = {
        name for name in REQUIRED_PRETRAINED_PERCEPTION_MODULES if name in policy_state
    }
    required_modules = set(REQUIRED_PRETRAINED_PERCEPTION_MODULES)
    ppo_partial_modules = set(PPOVEL_TRAIN_PHASE_PARTIAL_PERCEPTION_MODULES)
    ppo_fresh_depth_modules = set(PPOVEL_TRAIN_PHASE_FRESH_DEPTH_MODULES)

    if present_modules == required_modules:
        pass
    elif (
        present_modules == ppo_partial_modules
        and ppo_fresh_depth_modules.isdisjoint(policy_state)
        and policy_state.get("last_phase") == "train"
    ):
        if not train_perception:
            raise ValueError(
                "PPOVEL train-phase partial perception warm start requires "
                "algo.train_perception=true because its depth perception "
                "modules are freshly initialized"
            )
    else:
        missing_modules = [
            name
            for name in REQUIRED_PRETRAINED_PERCEPTION_MODULES
            if name not in present_modules
        ]
        raise ValueError(
            "pretrained perception checkpoint must contain either the complete "
            "perception stack or a PPOVEL train-phase partial stack; missing "
            f"required perception module mappings: {missing_modules}"
        )
    with open_dict(cfg.algo):
        cfg.algo.perception_checkpoint_path = resolved
    return resolved


def _validate_failure_phase_teacher_sampling(cfg: DictConfig) -> None:
    """Validate the shared live/uniform-Teacher/focused-Teacher source mix."""
    actor_teacher_fraction = _finite_fraction(
        "algo.teacher_actor_replay_fraction",
        cfg.algo.get("teacher_actor_replay_fraction", None),
    )
    perception_teacher_fraction = _finite_fraction(
        "algo.teacher_perception_replay_fraction",
        cfg.algo.get("teacher_perception_replay_fraction", None),
    )
    critic_teacher_fraction = _finite_fraction(
        "algo.q_teacher_replay_ratio",
        cfg.algo.get("q_teacher_replay_ratio", None),
    )
    focus_fraction = _finite_fraction(
        "algo.failure_phase_teacher_fraction",
        cfg.algo.get("failure_phase_teacher_fraction", None),
    )
    lookback = _positive_int(
        "algo.failure_phase_lookback_steps",
        cfg.algo.get("failure_phase_lookback_steps", None),
    )
    samples = _positive_int(
        "algo.failure_phase_samples_per_failure",
        cfg.algo.get("failure_phase_samples_per_failure", None),
    )
    _positive_int(
        "algo.failure_phase_num_bins",
        cfg.algo.get("failure_phase_num_bins", None),
    )
    if samples > lookback + 1:
        raise ValueError(
            "algo.failure_phase_samples_per_failure cannot exceed the inclusive "
            "algo.failure_phase_lookback_steps + 1 interval"
        )

    if (
        focus_fraction > 0.0
        and max(
            actor_teacher_fraction,
            perception_teacher_fraction,
            critic_teacher_fraction,
        )
        == 0.0
    ):
        raise ValueError(
            "algo.failure_phase_teacher_fraction > 0 requires a positive "
            "Teacher source fraction"
        )


def _validate_teacher_prefill_reachability(cfg: DictConfig) -> None:
    """Reject a prefill ceiling that cannot fill the ring even without filtering."""
    max_rollouts = _positive_int(
        "algo.teacher_prefill_max_rollouts",
        cfg.algo.get("teacher_prefill_max_rollouts", None),
    )
    num_envs = _positive_int("task.num_envs", cfg.task.get("num_envs", None))
    train_every = _positive_int("algo.train_every", cfg.algo.get("train_every", None))
    capacity = _positive_int(
        "algo.q_teacher_buffer_capacity",
        cfg.algo.get("q_teacher_buffer_capacity", None),
    )
    raw_transition_upper_bound = max_rollouts * num_envs * train_every
    if raw_transition_upper_bound < capacity:
        raise ValueError(
            "algo.teacher_prefill_max_rollouts cannot possibly fill "
            "algo.q_teacher_buffer_capacity: theoretical raw-transition "
            f"upper bound {raw_transition_upper_bound} < {capacity}"
        )


def _expected_dagger_backend_config(algo: Mapping) -> dict:
    """Reconstruct the exact runtime-independent checkpoint config lock."""
    return {
        **{name: algo.get(name) for name in DAGGER_BACKEND_CONFIG_FIELDS},
        "method": EXPECTED_TRAINING_ALGORITHM,
        "actor_output": "direct_unbounded_raw_joint_command",
        "bc_loss": "joint_normalized_raw_mean_teacher_smooth_l1",
    }


def _reject_obsolete_bounded_action_controls(cfg: DictConfig) -> None:
    """Reject stale knobs that would imply a bounded/tanh action backend."""
    obsolete = sorted(
        name
        for name in ("dagger_action_clip", "dagger_teacher_action_threshold")
        if name in cfg.algo
    )
    if obsolete:
        raise ValueError(
            "the direct unbounded raw-action backend removed bounded-action "
            f"controls: {obsolete}"
        )


def _forbidden_algo_fields(algo: Mapping) -> list[str]:
    """Return stochastic-policy fields outside the inert PPO compatibility set."""
    forbidden = []
    for raw_name in algo:
        name = str(raw_name).lower().replace("-", "_")
        if name in INERT_PPO_COMPATIBILITY_FIELDS:
            continue
        compact = name.replace("_", "")
        is_forbidden = (
            name.startswith("sac_")
            or "log_std" in name
            or "log_prob" in name
            or "logprob" in compact
            or name == "alpha"
            or name.startswith("alpha_")
            or name.endswith("_alpha")
            or "target_entropy" in name
            or "entropy" in name
        )
        if is_forbidden:
            forbidden.append(str(raw_name))
    return sorted(forbidden)


def apply_td3_dagger_iteration_controls(cfg: DictConfig) -> None:
    """Map the exact main budget plus bounded dynamic prefill to total frames."""
    iterations = cfg.get("td3_dagger_iterations", None)
    if iterations is not None:
        iterations = _positive_int("td3_dagger_iterations", iterations)
        prefill_max_rollouts = _positive_int(
            "algo.teacher_prefill_max_rollouts",
            cfg.algo.teacher_prefill_max_rollouts,
        )
        num_envs = _positive_int("task.num_envs", cfg.task.num_envs)
        train_every = _positive_int("algo.train_every", cfg.algo.train_every)
        world_size = _positive_int("distributed world size", aa.get_world_size())
        with open_dict(cfg):
            # The shared runner treats this as an upper bound and stops early
            # after dynamic prefill plus the exact requested main budget.
            cfg._bc_dagger_main_rollout_budget = iterations
            cfg.total_frames = (
                (iterations + prefill_max_rollouts)
                * num_envs
                * train_every
                * world_size
            )

    beta_zero_iteration = cfg.algo.get("dagger_beta_zero_iteration", None)
    if beta_zero_iteration is not None:
        beta_zero_iteration = _positive_int(
            "algo.dagger_beta_zero_iteration", beta_zero_iteration
        )
        with open_dict(cfg.algo):
            # The existing controller reaches beta_end at this completed-
            # rollout boundary.  Beta remains categorical, never interpolation.
            cfg.algo.dagger_beta_decay_rollouts = beta_zero_iteration

    safe_zero_iteration = cfg.algo.get("dagger_safe_zero_iteration", None)
    if safe_zero_iteration is not None:
        _positive_int("algo.dagger_safe_zero_iteration", safe_zero_iteration)


def td3_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Return the exact main budget and bounded dynamic-prefill schedule."""
    apply_td3_dagger_iteration_controls(cfg)
    num_envs = _positive_int("task.num_envs", cfg.task.num_envs)
    train_every = _positive_int("algo.train_every", cfg.algo.train_every)
    world_size = _positive_int("distributed world size", aa.get_world_size())
    main_rollouts = _positive_int(
        "td3_dagger_iterations", cfg.get("td3_dagger_iterations", None)
    )
    prefill_max_rollouts = _positive_int(
        "algo.teacher_prefill_max_rollouts",
        cfg.algo.teacher_prefill_max_rollouts,
    )
    prefill_target_rows = _positive_int(
        "algo.q_teacher_buffer_capacity", cfg.algo.q_teacher_buffer_capacity
    )
    frames_per_rollout = num_envs * train_every
    max_physical_rollouts = main_rollouts + prefill_max_rollouts
    expected_frames = max_physical_rollouts * frames_per_rollout * world_size
    if int(cfg.total_frames) != expected_frames:
        raise RuntimeError("TD3 rollout/frame upper-bound schedule is inconsistent")

    start_rollout = 0
    end_rollout = start_rollout + main_rollouts
    decay_rollouts = _positive_int(
        "algo.dagger_beta_decay_rollouts",
        cfg.algo.dagger_beta_decay_rollouts,
    )
    beta_zero_rollouts = (
        max(end_rollout - max(start_rollout, decay_rollouts), 0)
        if float(cfg.algo.dagger_beta_end) == 0.0
        else 0
    )
    safe_zero_iteration = cfg.algo.get("dagger_safe_zero_iteration", None)
    safe_zero_rollouts = (
        max(end_rollout - max(start_rollout, int(safe_zero_iteration)), 0)
        if safe_zero_iteration is not None
        else 0
    )
    return {
        "frames_per_rollout": frames_per_rollout,
        "total_rollouts": main_rollouts,
        "main_rollouts": main_rollouts,
        "prefill_max_rollouts": prefill_max_rollouts,
        "prefill_target_rows": prefill_target_rows,
        "max_physical_rollouts": max_physical_rollouts,
        "start_rollout": start_rollout,
        "end_rollout": end_rollout,
        "decay_rollouts": decay_rollouts,
        "beta_zero_rollouts": beta_zero_rollouts,
        "safe_zero_rollouts": safe_zero_rollouts,
    }


def prepare_td3_bc_dagger_checkpoint(cfg: DictConfig) -> dict | None:
    """Reject same-stage continuation across the fresh-only v3 contract."""
    requested = cfg.get("td3_bc_dagger_checkpoint", None)
    if requested is None:
        return None
    raise ValueError(
        "same-stage TD3 resume is intentionally unsupported by the fresh-only "
        "direct-raw-action/raw-perception v3 contract; leave "
        "td3_bc_dagger_checkpoint=null "
        "and start from a train-phase PPO checkpoint_path"
    )


def prepare_fresh_td3_bc_dagger_source(cfg: DictConfig) -> dict | None:
    """Accept only a compatible train-phase PPO teacher for a fresh run."""
    if cfg.get("td3_bc_dagger_checkpoint", None) is not None:
        return None
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "a fresh TD3-BC DAgger run must collect a new replay lineage; "
            "remove explicit teacher replay paths"
        )
    try:
        from .stage_bc_dagger import (
            _resolve_source_checkpoint,
            _validate_source_checkpoint,
        )
    except ImportError:
        from stage_bc_dagger import (  # type: ignore
            _resolve_source_checkpoint,
            _validate_source_checkpoint,
        )

    source_path = _resolve_source_checkpoint(cfg.get("checkpoint_path", None))
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    policy_state = _validate_source_checkpoint(checkpoint, cfg)
    with open_dict(cfg):
        cfg.checkpoint_path = source_path
        # helpers.py recognizes these shared flags and therefore cannot import
        # an unrelated adjacent replay into this fresh raw-perception lineage.
        cfg._bc_dagger_fresh_source = True
        cfg._bc_dagger_model_only_resume = False
    return {
        "path": source_path,
        "source_last_iter": int(policy_state.get("last_iter", -1)),
    }


def validate_td3_bc_dagger_config(cfg: DictConfig) -> None:
    """Fail before simulator startup when the method contract is violated."""
    _require_single_process_execution()
    apply_td3_dagger_iteration_controls(cfg)
    if cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError(
            "scripts/TD3_bc_dagger.py requires algo=td3_bc_dagger_finetune; "
            f"got algo.name={cfg.algo.get('name')!r}"
        )
    if cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError(
            "scripts/TD3_bc_dagger.py requires DistributionalTD3TeacherBC; "
            f"got algo._target_={cfg.algo.get('_target_')!r}"
        )
    if cfg.algo.get("phase") != "finetune":
        raise ValueError("distributional TD3 Teacher-BC requires phase=finetune")
    if cfg.algo.get("vecnorm") != "eval":
        raise ValueError("distributional TD3 Teacher-BC requires vecnorm=eval")
    if cfg.algo.get("use_depth") is not True:
        raise ValueError("distributional TD3 requires the locked depth encoder")
    if cfg.algo.get("use_object_adapt") is not True:
        raise ValueError(
            "distributional TD3 requires the locked object adaptation path"
        )
    if cfg.algo.get("adapt_module") != "gru":
        raise ValueError("distributional TD3 requires adapt_module=gru")
    if int(cfg.algo.get("latent_dim", -1)) != 256:
        raise ValueError("distributional TD3 requires latent_dim=256")
    if tuple(cfg.algo.get("in_keys", ())) != EXPECTED_ACTOR_IN_KEYS:
        raise ValueError("distributional TD3 Actor observation keys/order are locked")
    if (
        tuple(cfg.algo.get("replay_raw_observation_keys", ()))
        != EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
    ):
        raise ValueError("distributional TD3 replay observation keys/order are locked")
    if bool(cfg.algo.get("enable_residual_distillation", False)):
        raise ValueError("distributional TD3 Teacher-BC owns the only Actor optimizer")
    if not bool(cfg.algo.get("dagger_replay_raw_observations", False)):
        raise ValueError(
            "distributional TD3 Teacher-BC requires raw replay observations"
        )
    _reject_obsolete_bounded_action_controls(cfg)
    _validate_perception_training_controls(cfg)

    forbidden = _forbidden_algo_fields(cfg.algo)
    if forbidden:
        raise ValueError(
            "distributional TD3 Teacher-BC forbids stochastic-policy fields: "
            f"{forbidden}"
        )

    for name in (
        "eta_td3",
        "lambda_bc",
        "target_policy_noise_std",
        "target_policy_noise_clip",
        "collector_exploration_noise_std",
        "collector_exploration_noise_clip",
        "q_weight_decay",
        "q_max_grad_norm",
    ):
        _finite_nonnegative(f"algo.{name}", cfg.algo.get(name, None))
    for name in (
        "dagger_safe_min_teacher_steps",
        "dagger_beta_decay_rollouts",
        "dagger_buffer_capacity",
        "dagger_batch_size",
        "policy_delay",
        "td3_learning_starts",
        "q_hidden_dim",
        "q_batch_size",
        "q_updates_per_rollout",
        "q_teacher_buffer_capacity",
        "perception_replay_burn_in",
        "perception_encode_microbatch_size",
    ):
        _positive_int(f"algo.{name}", cfg.algo.get(name, None))
    _positive_int(
        "algo.teacher_prefill_max_rollouts",
        cfg.algo.get("teacher_prefill_max_rollouts", None),
    )
    _validate_teacher_prefill_reachability(cfg)
    _validate_failure_phase_teacher_sampling(cfg)
    _positive_int(
        "algo.teacher_perception_batch_size",
        cfg.algo.get("teacher_perception_batch_size", None),
    )
    _positive_int(
        "algo.teacher_perception_warmup_steps",
        cfg.algo.get("teacher_perception_warmup_steps", None),
        allow_zero=True,
    )
    if cfg.get("td3_dagger_iterations", None) is None:
        raise ValueError(
            "dynamic Teacher prefill requires an explicit "
            "td3_dagger_iterations main-rollout budget"
        )
    if int(cfg.algo.perception_replay_burn_in) != 8:
        raise ValueError(
            "raw-perception replay v2 requires algo.perception_replay_burn_in=8"
        )
    if str(cfg.algo.get("perception_depth_codec", "")) != "uint8_div_100_v1":
        raise ValueError(
            "raw-perception replay v2 requires "
            "algo.perception_depth_codec='uint8_div_100_v1'"
        )
    if cfg.algo.get("save_teacher_buffer", None) is not False:
        raise ValueError(
            "raw-perception replay v2 requires algo.save_teacher_buffer=false; "
            "teacher_replay_buffer.h5 export is disabled"
        )
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "raw-perception replay v2 does not accept a teacher replay H5 path"
        )
    removed_h5_fields = {
        name
        for name in (
            "teacher_buffer_filename",
            "teacher_buffer_path",
            "teacher_buffer_capacity",
            "teacher_buffer_snapshot_chunk_rows",
        )
        if name in cfg.algo
    }
    if removed_h5_fields:
        raise ValueError(
            "raw-perception replay v2 removed H5-only algo fields: "
            f"{sorted(removed_h5_fields)}"
        )
    if int(cfg.algo.q_batch_size) % 2:
        raise ValueError("algo.q_batch_size must be even for exact 50/50 replay")
    if int(cfg.algo.q_teacher_buffer_capacity) < int(cfg.algo.td3_learning_starts):
        raise ValueError(
            "algo.q_teacher_buffer_capacity must cover algo.td3_learning_starts"
        )
    for name in (
        "dagger_bc_lr",
        "dagger_actor_huber_delta",
        "q_lr",
        "q_action_input_gain",
    ):
        _finite_positive(f"algo.{name}", cfg.algo.get(name, None))
    _finite_nonnegative("algo.q_tau", cfg.algo.get("q_tau", None))
    if not 0.0 < float(cfg.algo.q_tau) <= 1.0:
        raise ValueError("algo.q_tau must be in (0, 1]")
    if float(cfg.algo.eta_td3) == 0.0 and float(cfg.algo.lambda_bc) == 0.0:
        raise ValueError("eta_td3 and lambda_bc cannot both be zero")

    if int(cfg.algo.get("q_num_atoms", -1)) != 501:
        raise ValueError("distributional TD3 requires exactly 501 C51 atoms")
    if not math.isclose(float(cfg.algo.get("q_v_min", math.nan)), -20.0):
        raise ValueError("distributional TD3 requires q_v_min=-20")
    if not math.isclose(float(cfg.algo.get("q_v_max", math.nan)), 20.0):
        raise ValueError("distributional TD3 requires q_v_max=20")
    if cfg.algo.get("q_action_fusion") != "late":
        raise ValueError("distributional TD3 requires late Q-action fusion")
    if cfg.algo.get("q_layer_norm") is not True:
        raise ValueError("distributional TD3 requires LayerNorm Q Critics")
    if cfg.algo.get("q_action_coordinates") != "raw_joint_command":
        raise ValueError("distributional TD3 requires raw_joint_command Q actions")
    if cfg.algo.get("q_normalize_actions") is not True:
        raise ValueError("distributional TD3 requires normalized Q actions")
    if not math.isclose(float(cfg.algo.get("q_action_input_gain", math.nan)), 1.0):
        raise ValueError("distributional TD3 requires q_action_input_gain=1")
    if not math.isclose(
        float(cfg.algo.get("q_teacher_replay_ratio", math.nan)),
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Phase 1 requires exact 50/50 Teacher/Student Q replay")

    control_mode = str(cfg.algo.get("dagger_control_mode", "beta"))
    if control_mode not in ("beta", "safe", "hybrid"):
        raise ValueError("algo.dagger_control_mode must be beta, safe, or hybrid")
    beta_start = float(cfg.algo.get("dagger_beta_start", math.nan))
    beta_end = float(cfg.algo.get("dagger_beta_end", math.nan))
    if not (
        math.isfinite(beta_start)
        and math.isfinite(beta_end)
        and 0.0 <= beta_start <= 1.0
        and 0.0 <= beta_end <= 1.0
    ):
        raise ValueError("DAgger beta endpoints must lie in [0, 1]")
    release = float(cfg.algo.get("dagger_safe_release_rms", math.nan))
    takeover = float(cfg.algo.get("dagger_safe_takeover_rms", math.nan))
    if not (
        math.isfinite(release) and math.isfinite(takeover) and 0.0 <= release < takeover
    ):
        raise ValueError("SafeDAgger requires 0 <= release_rms < takeover_rms")
    safe_zero_iteration = cfg.algo.get("dagger_safe_zero_iteration", None)
    if safe_zero_iteration is not None and control_mode == "beta":
        raise ValueError(
            "algo.dagger_safe_zero_iteration requires safe or hybrid control"
        )

    if cfg.get("td3_bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "same-stage TD3 resume is unsupported by the direct-raw-action "
            "raw-perception v3 contract; "
            "use only a fresh train-phase PPO checkpoint_path"
        )
    obsolete_resume_values = {
        "td3_bc_dagger_copy_teacher_replay": cfg.get(
            "td3_bc_dagger_copy_teacher_replay", None
        ),
        "td3_dagger_resume_rollout_count": cfg.get(
            "td3_dagger_resume_rollout_count", None
        ),
        "td3_dagger_resume_environment_steps": cfg.get(
            "td3_dagger_resume_environment_steps", None
        ),
        "_bc_dagger_teacher_replay_copy_source": cfg.get(
            "_bc_dagger_teacher_replay_copy_source", None
        ),
        "_bc_dagger_teacher_replay_copy_path": cfg.get(
            "_bc_dagger_teacher_replay_copy_path", None
        ),
    }
    configured_obsolete = {
        name: value
        for name, value in obsolete_resume_values.items()
        if value is not None
    }
    if configured_obsolete:
        raise ValueError(
            "raw-perception replay v2 removed TD3/H5 resume controls: "
            f"{sorted(configured_obsolete)}"
        )
    if cfg.get("checkpoint_path", None) is None:
        raise ValueError(
            "scripts/TD3_bc_dagger.py requires a fresh train-phase PPO checkpoint_path"
        )
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint is not accepted by raw-perception replay v2; "
            "use a fresh train-phase PPO checkpoint_path"
        )

    schedule = td3_dagger_rollout_schedule(cfg)
    if (
        control_mode in ("beta", "hybrid")
        and cfg.algo.get("dagger_beta_zero_iteration", None) is not None
        and beta_end != 0.0
    ):
        raise ValueError("algo.dagger_beta_zero_iteration requires dagger_beta_end=0")
    if (
        control_mode in ("beta", "hybrid")
        and beta_start > 0.0
        and beta_end == 0.0
        and schedule["beta_zero_rollouts"] < 1
    ):
        raise ValueError(
            "the run must include at least one rollout after beta reaches zero"
        )
    if (
        control_mode in ("safe", "hybrid")
        and safe_zero_iteration is not None
        and schedule["safe_zero_rollouts"] < 1
    ):
        raise ValueError(
            "the run must include at least one rollout after SafeDAgger is off"
        )


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="TD3_bc_dagger",
    version_base=None,
)
def main(cfg: DictConfig):
    # This must precede checkpoint loading and, most importantly, the shared
    # runner's W&B/Isaac/policy construction.
    _require_single_process_execution()
    apply_td3_dagger_iteration_controls(cfg)
    prepare_td3_bc_dagger_checkpoint(cfg)
    prepare_fresh_td3_bc_dagger_source(cfg)
    validate_td3_bc_dagger_config(cfg)
    schedule = td3_dagger_rollout_schedule(cfg)
    print(
        "Distributional TD3 + Teacher-BC schedule: "
        f"prefill=until {schedule['prefill_target_rows']} Teacher rows "
        f"(safety ceiling {schedule['prefill_max_rollouts']} rollouts), then "
        f"start={schedule['start_rollout']}, "
        f"main={schedule['main_rollouts']}, "
        f"end={schedule['end_rollout']}, "
        f"frames/rollout={schedule['frames_per_rollout']}; "
        f"method={EXPECTED_TRAINING_ALGORITHM}"
    )
    print(
        "Distributional TD3 updates: "
        f"atoms={int(cfg.algo.q_num_atoms)}, "
        f"policy_delay={int(cfg.algo.policy_delay)}, "
        f"tau={float(cfg.algo.q_tau):g}, "
        f"eta_td3={float(cfg.algo.eta_td3):g}, "
        f"lambda_bc={float(cfg.algo.lambda_bc):g}"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
