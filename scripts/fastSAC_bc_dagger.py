"""Fresh-only distributional FastSAC + mean-action Teacher-BC entrypoint.

The shared :mod:`scripts.train` runner still owns environment construction,
rollout collection, checkpoint writing, and evaluation.  This module owns the
method-specific Hydra surface and rejects incompatible configuration before
the simulator starts.

The stochastic policy is intentional: Student collection, the soft Bellman
target, and the SAC Actor objective sample one explicitly selected
distribution. The historical normalized tanh policy remains available; an
opt-in mode reproduces PPOVEL's raw physical-action Gaussian and directly
learned joint standard deviations. Exploration therefore comes from SAC
itself; inherited TD3 noise knobs are locked to zero.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping

import hydra
import torch
from omegaconf import DictConfig, open_dict

import active_adaptation as aa
from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    ACTION_CONTRACT_SEMANTICS,
    ACTOR_BACKEND,
    CHECKPOINT_VERSION,
    NORMALIZED_TANH_ACTION_DISTRIBUTION,
    PHYSICAL_STD_BOUND_MODES,
    PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    UNIFORM_PHYSICAL_STD_BOUND_MODE,
    TRAINING_ALGORITHM,
    _validate_fastsac_entropy_target_controls,
    checkpoint_module_mismatches,
    validate_actor_adopt_checkpoint_payload,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    apply_perception_training_source,
)

try:
    from .train import run_training
except ImportError:
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

EXPECTED_ALGO_NAME = "fastsac_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.fastsac_bc_dagger.DistributionalFastSACTeacherBC"
)
EXPECTED_TRAINING_ALGORITHM = TRAINING_ALGORITHM
EXPECTED_CHECKPOINT_VERSION = CHECKPOINT_VERSION
EXPECTED_ACTOR_BACKEND = ACTOR_BACKEND
EXPECTED_ACTION_CONTRACT_SEMANTICS = ACTION_CONTRACT_SEMANTICS
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


def _require_single_process_execution() -> None:
    """Reject distributed execution before constructing Isaac or the policy."""
    world_size = aa.get_world_size()
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise RuntimeError("distributed world size must be an integer")
    if bool(aa.is_distributed()) or world_size != 1:
        raise RuntimeError(
            "FastSAC-BC DAgger currently supports exactly one training process; "
            "distributed/multi-GPU execution is rejected because its custom "
            "Actor, Critic, temperature, and perception gradients are not "
            "synchronized"
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


def _reject_obsolete_bounded_action_controls(cfg: DictConfig) -> None:
    """Reject legacy controls replaced by the single action_support_clip."""
    obsolete = sorted(
        name
        for name in ("dagger_action_clip", "dagger_teacher_action_threshold")
        if name in cfg.algo
    )
    if obsolete:
        raise ValueError(
            "legacy bounded-action controls were replaced by "
            f"algo.action_support_clip: {obsolete}"
        )


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


def _saved_checkpoint_contract(checkpoint: Mapping, *, label: str) -> dict:
    saved_cfg = checkpoint.get("cfg")
    if not isinstance(saved_cfg, Mapping):
        raise ValueError(f"{label} must contain its saved cfg mapping")
    saved_algo = saved_cfg.get("algo")
    saved_task = saved_cfg.get("task")
    if not isinstance(saved_algo, Mapping) or not isinstance(saved_task, Mapping):
        raise ValueError(f"{label} must contain cfg.algo and cfg.task mappings")
    teacher_path = saved_cfg.get("checkpoint_path")
    task_name = saved_task.get("name")
    if not isinstance(teacher_path, str) or not teacher_path.strip():
        raise ValueError(f"{label} must record cfg.checkpoint_path")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError(f"{label} must record cfg.task.name")
    return {
        "teacher_path": os.path.realpath(os.path.expanduser(teacher_path)),
        "task_name": task_name,
        "latent_dim": saved_algo.get("latent_dim"),
        "in_keys": tuple(saved_algo.get("in_keys", ())),
    }


def _validate_actor_adopt_checkpoint_controls(
    cfg: DictConfig,
    *,
    perception_path: str | None,
) -> dict | None:
    """Audit and canonicalize the optional actor_adapt-only warm start."""
    configured_path = cfg.algo.get("actor_adopt_checkpoint_path", None)
    if configured_path is None:
        return None
    if str(cfg.algo.get("student_actor_initialization", "teacher_bc")) != (
        "teacher_bc"
    ):
        raise ValueError(
            "algo.actor_adopt_checkpoint_path cannot be combined with "
            "algo.student_actor_initialization=fresh"
        )
    if cfg.algo.get("load_pretrained_perception") is not True or (
        perception_path is None
    ):
        raise ValueError(
            "algo.actor_adopt_checkpoint_path requires "
            "algo.load_pretrained_perception=true and an explicit "
            "algo.perception_checkpoint_path"
        )
    try:
        raw_path = os.fspath(configured_path)
    except TypeError as exc:
        raise ValueError(
            "algo.actor_adopt_checkpoint_path must be a local filesystem path"
        ) from exc
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            "algo.actor_adopt_checkpoint_path must be a non-empty local path"
        )
    raw_path = os.path.expanduser(raw_path)
    if raw_path.startswith("run:") or "://" in raw_path:
        raise ValueError(
            "algo.actor_adopt_checkpoint_path must be a local filesystem path"
        )
    resolved = os.path.realpath(hydra.utils.to_absolute_path(raw_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"actor_adopt checkpoint does not exist: {resolved}"
        )
    actor_checkpoint = torch.load(
        resolved, map_location="cpu", weights_only=False
    )
    _, provenance = validate_actor_adopt_checkpoint_payload(
        actor_checkpoint,
        source_path=resolved,
    )

    runtime_task = cfg.task.get("name")
    actor_contract = _saved_checkpoint_contract(
        actor_checkpoint, label="actor_adopt checkpoint"
    )
    expected_contract = {
        "task_name": runtime_task,
        "latent_dim": int(cfg.algo.get("latent_dim")),
        "in_keys": tuple(cfg.algo.get("in_keys", ())),
    }
    for name, expected in expected_contract.items():
        if actor_contract[name] != expected:
            raise ValueError(
                f"actor_adopt checkpoint {name} does not match runtime: "
                f"expected {expected!r}, got {actor_contract[name]!r}"
            )

    main_teacher_path = cfg.get("checkpoint_path")
    if not isinstance(main_teacher_path, str) or not main_teacher_path.strip():
        raise ValueError(
            "algo.actor_adopt_checkpoint_path requires an explicit root "
            "checkpoint_path for the privileged Teacher"
        )
    resolved_teacher_path = os.path.realpath(
        hydra.utils.to_absolute_path(os.path.expanduser(main_teacher_path))
    )
    if actor_contract["teacher_path"] != resolved_teacher_path:
        raise ValueError(
            "actor_adopt checkpoint was trained from a different privileged "
            "Teacher checkpoint than root checkpoint_path"
        )
    if not os.path.isfile(resolved_teacher_path):
        raise FileNotFoundError(
            f"privileged Teacher checkpoint does not exist: {resolved_teacher_path}"
        )
    teacher_checkpoint = torch.load(
        resolved_teacher_path, map_location="cpu", weights_only=False
    )
    teacher_policy = teacher_checkpoint.get("policy")
    actor_policy = actor_checkpoint["policy"]
    if not isinstance(teacher_policy, Mapping):
        raise ValueError("privileged Teacher checkpoint lacks policy mapping")
    teacher_mismatches = checkpoint_module_mismatches(
        teacher_policy,
        actor_policy,
        ("actor", "encoder_priv", "critic"),
        ignored_key_suffixes=("actor_std",),
    )
    if teacher_mismatches:
        raise ValueError(
            "actor_adopt checkpoint frozen Teacher does not match root "
            f"checkpoint_path; mismatched modules={list(teacher_mismatches)}"
        )

    perception_checkpoint = torch.load(
        perception_path, map_location="cpu", weights_only=False
    )
    if not isinstance(perception_checkpoint, Mapping) or not isinstance(
        perception_checkpoint.get("policy"), Mapping
    ):
        raise ValueError("pretrained perception checkpoint lacks policy mapping")
    perception_contract = _saved_checkpoint_contract(
        perception_checkpoint, label="pretrained perception checkpoint"
    )
    for name in ("teacher_path", "task_name", "latent_dim", "in_keys"):
        if perception_contract[name] != actor_contract[name]:
            raise ValueError(
                "actor_adopt and perception checkpoints do not share the same "
                f"Teacher/task/latent contract at {name!r}"
            )
    perception_mismatches = checkpoint_module_mismatches(
        perception_checkpoint["policy"],
        actor_policy,
        REQUIRED_PRETRAINED_PERCEPTION_MODULES,
    )
    exact_match = not perception_mismatches
    if not exact_match:
        warnings.warn(
            "actor_adapt was BC-trained against the perception tensors in its "
            "joint checkpoint, but algo.perception_checkpoint_path selects a "
            "different later/independent perception state. This pairing is "
            "allowed because both regress the same frozen Teacher latent, but "
            "it is not an exact coupled restoration. For the exact pair, set "
            "BOTH paths to the same perception_actor checkpoint. Differing "
            f"modules: {list(perception_mismatches)}",
            UserWarning,
            stacklevel=2,
        )
    with open_dict(cfg.algo):
        cfg.algo.actor_adopt_checkpoint_path = resolved
    return {
        **provenance,
        "perception_source_path": perception_path,
        "perception_exact_match": exact_match,
        "perception_mismatched_modules": perception_mismatches,
    }


def _validate_failure_phase_teacher_sampling(cfg: DictConfig) -> None:
    """Validate independently configurable Teacher source fractions."""
    _finite_fraction(
        "algo.teacher_actor_replay_fraction",
        cfg.algo.get("teacher_actor_replay_fraction", None),
    )
    _finite_fraction(
        "algo.teacher_perception_replay_fraction",
        cfg.algo.get("teacher_perception_replay_fraction", None),
    )
    _finite_fraction(
        "algo.q_teacher_replay_ratio",
        cfg.algo.get("q_teacher_replay_ratio", None),
    )
    _finite_fraction(
        "algo.q_online_dagger_replay_fraction",
        cfg.algo.get("q_online_dagger_replay_fraction", None),
    )
    _finite_fraction(
        "algo.actor_online_dagger_replay_fraction",
        cfg.algo.get("actor_online_dagger_replay_fraction", None),
    )
    _finite_fraction(
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


def _validate_perception_training_contract(cfg: DictConfig) -> None:
    """Validate FastSAC's live-rollout or explicit four-way perception mode."""
    mode = str(cfg.algo.get("perception_replay_mode", ""))
    if mode == "four_way":
        if bool(cfg.algo.get("train_dr_estimator", False)):
            raise ValueError(
                "FastSAC four_way perception replay has no DR-estimator target; "
                "set algo.train_dr_estimator=false"
            )
        return
    if mode != ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE:
        raise ValueError(
            "FastSAC perception requires algo.perception_replay_mode to be "
            "'online_student_rollout' or 'four_way'"
        )
    if float(cfg.algo.get("teacher_perception_replay_fraction", math.nan)) != 0.0:
        raise ValueError(
            "FastSAC live-rollout perception requires "
            "algo.teacher_perception_replay_fraction=0"
        )
    canonical = {
        "perception_uniform_student_fraction": 1.0,
        "perception_failure_student_fraction": 0.0,
        "perception_uniform_teacher_fraction": 0.0,
        "perception_failure_teacher_fraction": 0.0,
    }
    for name, expected in canonical.items():
        if name in cfg.algo and float(cfg.algo.get(name)) != expected:
            raise ValueError(
                "FastSAC live-rollout perception requires canonical mix "
                "US=1, FS=UT=FT=0"
            )
    if (
        str(cfg.algo.get("dagger_control_mode", "")) != "beta"
        or float(cfg.algo.get("dagger_beta_start", math.nan)) != 0.0
        or float(cfg.algo.get("dagger_beta_end", math.nan)) != 0.0
    ):
        raise ValueError(
            "FastSAC live Student perception requires beta control with "
            "algo.dagger_beta_start=algo.dagger_beta_end=0"
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


def apply_fastsac_dagger_iteration_controls(cfg: DictConfig) -> None:
    """Convert the explicit main-rollout count to the shared frame budget."""
    iterations = cfg.get("fastsac_dagger_iterations", None)
    if iterations is None:
        raise ValueError(
            "scripts/fastSAC_bc_dagger.py requires an explicit positive "
            "fastsac_dagger_iterations main-rollout budget"
        )
    iterations = _positive_int("fastsac_dagger_iterations", iterations)
    prefill_max_rollouts = _positive_int(
        "algo.teacher_prefill_max_rollouts",
        cfg.algo.get("teacher_prefill_max_rollouts", None),
    )
    num_envs = _positive_int("task.num_envs", cfg.task.num_envs)
    train_every = _positive_int("algo.train_every", cfg.algo.train_every)
    world_size = _positive_int("distributed world size", aa.get_world_size())
    with open_dict(cfg):
        cfg._bc_dagger_main_rollout_budget = iterations
        cfg.total_frames = (
            (iterations + prefill_max_rollouts) * num_envs * train_every * world_size
        )

    beta_zero_iteration = cfg.algo.get("dagger_beta_zero_iteration", None)
    if beta_zero_iteration is not None:
        beta_zero_iteration = _positive_int(
            "algo.dagger_beta_zero_iteration", beta_zero_iteration
        )
        with open_dict(cfg.algo):
            cfg.algo.dagger_beta_decay_rollouts = beta_zero_iteration

    safe_zero_iteration = cfg.algo.get("dagger_safe_zero_iteration", None)
    if safe_zero_iteration is not None:
        _positive_int("algo.dagger_safe_zero_iteration", safe_zero_iteration)


def fastsac_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Describe exact main rollouts plus the bounded dynamic prefill."""
    apply_fastsac_dagger_iteration_controls(cfg)
    num_envs = _positive_int("task.num_envs", cfg.task.num_envs)
    train_every = _positive_int("algo.train_every", cfg.algo.train_every)
    world_size = _positive_int("distributed world size", aa.get_world_size())
    main_rollouts = _positive_int(
        "fastsac_dagger_iterations", cfg.fastsac_dagger_iterations
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
        raise RuntimeError(
            "FastSAC rollout/frame upper-bound schedule is internally inconsistent"
        )

    start_rollout = 0
    end_rollout = main_rollouts
    decay_rollouts = _positive_int(
        "algo.dagger_beta_decay_rollouts",
        cfg.algo.dagger_beta_decay_rollouts,
    )
    beta_start = float(cfg.algo.dagger_beta_start)
    beta_end = float(cfg.algo.dagger_beta_end)
    if beta_start == 0.0 and beta_end == 0.0:
        beta_zero_rollouts = main_rollouts
    elif beta_end == 0.0:
        beta_zero_rollouts = max(main_rollouts - decay_rollouts, 0)
    else:
        beta_zero_rollouts = 0
    safe_zero_iteration = cfg.algo.get("dagger_safe_zero_iteration", None)
    safe_zero_rollouts = (
        max(main_rollouts - int(safe_zero_iteration), 0)
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


def prepare_fastsac_bc_dagger_checkpoint(cfg: DictConfig) -> None:
    """Reject same-stage continuation across the fresh-only v3 contract."""
    requested = cfg.get("fastsac_bc_dagger_checkpoint", None)
    if requested is None:
        return None
    raise ValueError(
        "same-stage FastSAC resume is intentionally unsupported by the "
        "fresh-only normalized-std/bounded-action v3 contract; leave "
        "fastsac_bc_dagger_checkpoint=null and use a train-phase PPO "
        "checkpoint_path"
    )


def prepare_fresh_fastsac_bc_dagger_source(cfg: DictConfig) -> dict | None:
    """Resolve and strictly validate a fresh train-phase PPO source."""
    if cfg.get("fastsac_bc_dagger_checkpoint", None) is not None:
        return None
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "a fresh FastSAC-BC DAgger run must collect a new replay lineage; "
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
        cfg._bc_dagger_fresh_source = True
        cfg._bc_dagger_model_only_resume = False
    return {
        "path": source_path,
        "source_last_iter": int(policy_state.get("last_iter", -1)),
    }


def _validate_locked_topology(cfg: DictConfig) -> None:
    if cfg.algo.get("phase") != "finetune":
        raise ValueError("distributional FastSAC Teacher-BC requires phase=finetune")
    if cfg.algo.get("vecnorm") != "eval":
        raise ValueError("distributional FastSAC Teacher-BC requires vecnorm=eval")
    if cfg.algo.get("use_depth") is not True:
        raise ValueError("distributional FastSAC requires the locked depth encoder")
    if cfg.algo.get("use_object_adapt") is not True:
        raise ValueError(
            "distributional FastSAC requires the locked object adaptation path"
        )
    if cfg.algo.get("adapt_module") != "gru":
        raise ValueError("distributional FastSAC requires adapt_module=gru")
    if int(cfg.algo.get("latent_dim", -1)) != 256:
        raise ValueError("distributional FastSAC requires latent_dim=256")
    if tuple(cfg.algo.get("in_keys", ())) != EXPECTED_ACTOR_IN_KEYS:
        raise ValueError(
            "distributional FastSAC Actor observation keys/order are locked"
        )
    if (
        tuple(cfg.algo.get("replay_raw_observation_keys", ()))
        != EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
    ):
        raise ValueError(
            "distributional FastSAC replay observation keys/order are locked"
        )
    if bool(cfg.algo.get("enable_residual_distillation", False)):
        raise ValueError(
            "distributional FastSAC Teacher-BC owns the only Actor optimizer"
        )
    if not bool(cfg.algo.get("dagger_replay_raw_observations", False)):
        raise ValueError(
            "distributional FastSAC Teacher-BC requires raw replay observations"
        )


def _validate_replay_contract(cfg: DictConfig) -> None:
    if int(cfg.algo.perception_replay_burn_in) != 8:
        raise ValueError(
            "raw-perception replay requires algo.perception_replay_burn_in=8"
        )
    if str(cfg.algo.get("perception_depth_codec", "")) != "uint8_div_100_v1":
        raise ValueError(
            "raw-perception replay requires "
            "algo.perception_depth_codec='uint8_div_100_v1'"
        )
    if cfg.algo.get("save_teacher_buffer", None) is not False:
        raise ValueError(
            "raw-perception replay requires algo.save_teacher_buffer=false; "
            "teacher_replay_buffer.h5 export is disabled"
        )
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "raw-perception replay does not accept a teacher replay H5 path"
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
            "raw-perception replay removed H5-only algo fields: "
            f"{sorted(removed_h5_fields)}"
        )


def _validate_sac_controls(cfg: DictConfig) -> None:
    student_actor_initialization = str(
        cfg.algo.get("student_actor_initialization", "teacher_bc")
    )
    if student_actor_initialization not in ("teacher_bc", "fresh"):
        raise ValueError(
            "algo.student_actor_initialization must be 'teacher_bc' or 'fresh'"
        )
    actor_observation_mode = str(
        cfg.algo.get("sac_actor_observation_mode", "student_perception")
    )
    if actor_observation_mode not in ("student_perception", "privileged_oracle"):
        raise ValueError(
            "algo.sac_actor_observation_mode must be 'student_perception' or "
            "'privileged_oracle'"
        )
    if (
        actor_observation_mode == "privileged_oracle"
        and str(cfg.algo.get("perception_replay_mode", ""))
        != ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    ):
        raise ValueError(
            "algo.sac_actor_observation_mode=privileged_oracle requires "
            "algo.perception_replay_mode=online_student_rollout"
        )
    action_distribution = str(
        cfg.algo.get(
            "sac_action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION
        )
    )
    if action_distribution not in (
        NORMALIZED_TANH_ACTION_DISTRIBUTION,
        PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    ):
        raise ValueError(
            "algo.sac_action_distribution must be 'normalized_tanh' or "
            "'ppo_physical_gaussian'"
        )
    for name in (
        "eta_sac",
        "lambda_bc",
        "q_weight_decay",
        "q_max_grad_norm",
        "sac_max_grad_norm",
    ):
        _finite_nonnegative(f"algo.{name}", cfg.algo.get(name, None))
    _finite_nonnegative(
        "algo.sac_actor_weight_decay",
        cfg.algo.get("sac_actor_weight_decay", 0.0),
    )
    for name in (
        "action_support_clip",
        "dagger_bc_lr",
        "dagger_actor_huber_delta",
        "q_lr",
        "q_action_input_gain",
        "q_update_to_data_ratio",
        "sac_actor_lr",
        "sac_alpha_init",
        "sac_alpha_lr",
    ):
        _finite_positive(f"algo.{name}", cfg.algo.get(name, None))
    if float(cfg.algo.eta_sac) == 0.0 and float(cfg.algo.lambda_bc) == 0.0:
        raise ValueError("eta_sac and lambda_bc cannot both be zero")

    if cfg.algo.get("sac_use_autotune", None) not in (True, False):
        raise ValueError("algo.sac_use_autotune must be boolean")
    if cfg.algo.get("use_q_filtered_bc", False) not in (True, False):
        raise ValueError("algo.use_q_filtered_bc must be boolean")
    if action_distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION:
        if str(
            cfg.algo.get(
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        ) != UNIFORM_PHYSICAL_STD_BOUND_MODE:
            raise ValueError(
                "algo.sac_physical_std_bound_mode=q_normalized requires "
                "algo.sac_action_distribution=ppo_physical_gaussian"
            )
        _finite_positive(
            "algo.sac_initial_action_std",
            cfg.algo.get("sac_initial_action_std", None),
        )
        _validate_fastsac_entropy_target_controls(
            cfg.algo.get("sac_log_std_min", None),
            cfg.algo.get("sac_log_std_max", None),
            cfg.algo.get("sac_target_entropy_ratio", None),
            field_prefix="algo",
        )
    else:
        _finite_positive(
            "algo.load_noise_scale", cfg.algo.get("load_noise_scale", None)
        )
        _finite_positive(
            "algo.sac_target_entropy_ratio",
            cfg.algo.get("sac_target_entropy_ratio", None),
        )
        for name in (
            "sac_physical_std_lr",
            "sac_physical_std_max_kl",
            "sac_physical_std_min",
            "sac_physical_std_max",
        ):
            _finite_positive(f"algo.{name}", cfg.algo.get(name, None))
        std_min = float(cfg.algo.sac_physical_std_min)
        std_max = float(cfg.algo.sac_physical_std_max)
        load_noise_scale = float(cfg.algo.load_noise_scale)
        if not std_min < std_max:
            raise ValueError(
                "algo.sac_physical_std_min must be smaller than "
                "algo.sac_physical_std_max"
            )
        if not std_min <= load_noise_scale <= std_max:
            raise ValueError(
                "algo.load_noise_scale must lie inside the physical std bounds"
            )
        bound_mode = str(
            cfg.algo.get(
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        )
        if bound_mode not in PHYSICAL_STD_BOUND_MODES:
            raise ValueError(
                "algo.sac_physical_std_bound_mode must be 'uniform_physical' "
                "or 'q_normalized'"
            )
        normalized_min = cfg.algo.get("sac_physical_std_normalized_min", 0.02)
        normalized_max = cfg.algo.get("sac_physical_std_normalized_max", 0.11)
        _finite_positive(
            "algo.sac_physical_std_normalized_min", normalized_min
        )
        _finite_positive(
            "algo.sac_physical_std_normalized_max", normalized_max
        )
        if not float(normalized_min) < float(normalized_max):
            raise ValueError(
                "algo.sac_physical_std_normalized_min must be smaller than "
                "algo.sac_physical_std_normalized_max"
            )
    if cfg.algo.get("sac_alpha_update_cadence", None) != "actor":
        raise ValueError(
            "algo.sac_alpha_update_cadence must be 'actor' so temperature "
            "updates match the delayed Actor cadence"
        )
    for name in (
        "sac_policy_frequency",
        "sac_learning_starts",
        "q_hidden_dim",
        "q_batch_size",
        "q_updates_per_rollout",
        "q_teacher_buffer_capacity",
    ):
        _positive_int(f"algo.{name}", cfg.algo.get(name, None))
    for name in ("q_n_step", "q_teacher_n_step"):
        if cfg.algo.get(name, None) is not None:
            _positive_int(f"algo.{name}", cfg.algo.get(name))
            if int(cfg.algo.get(name)) != 1:
                raise ValueError(
                    f"algo.{name} is locked to 1 for factual one-step "
                    "Expert/DAgger/Student SAC replay"
                )
    _finite_nonnegative("algo.sac_tau", cfg.algo.get("sac_tau", None))
    if not 0.0 < float(cfg.algo.sac_tau) <= 1.0:
        raise ValueError("algo.sac_tau must be in (0, 1]")

    # The inherited names remain in the dataclass solely for source topology
    # compatibility.  Lock aliases to the SAC controls so no stale TD3 knob can
    # silently affect cadence, warm-up, or target updates.
    if not math.isclose(float(cfg.algo.get("eta_td3", math.nan)), 0.0):
        raise ValueError("FastSAC requires inherited algo.eta_td3=0")
    locked_zero_noise = (
        "target_policy_noise_std",
        "target_policy_noise_clip",
        "collector_exploration_noise_std",
        "collector_exploration_noise_clip",
    )
    for name in locked_zero_noise:
        value = _finite_nonnegative(f"algo.{name}", cfg.algo.get(name, None))
        if value != 0.0:
            raise ValueError(
                f"FastSAC requires inherited algo.{name}=0; its learned "
                "stochastic policy is the only exploration source"
            )
    aliases = (
        ("dagger_bc_lr", "sac_actor_lr"),
        ("policy_delay", "sac_policy_frequency"),
        ("td3_learning_starts", "sac_learning_starts"),
        ("q_tau", "sac_tau"),
        ("q_max_grad_norm", "sac_max_grad_norm"),
    )
    for inherited, sac_name in aliases:
        if not math.isclose(
            float(cfg.algo.get(inherited, math.nan)),
            float(cfg.algo.get(sac_name, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"inherited algo.{inherited} must equal algo.{sac_name}")


def validate_fastsac_bc_dagger_config(cfg: DictConfig) -> None:
    """Fail before simulator startup when the method contract is violated."""
    _require_single_process_execution()
    # New commands use one unambiguous perception selector. Resolve it before
    # entrypoint validation so all existing validation and metadata continue
    # to operate on their stable internal representation.
    apply_perception_training_source(cfg.algo)
    apply_fastsac_dagger_iteration_controls(cfg)
    if cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError(
            "scripts/fastSAC_bc_dagger.py requires "
            "algo=fastsac_bc_dagger_finetune; "
            f"got algo.name={cfg.algo.get('name')!r}"
        )
    if cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError(
            "scripts/fastSAC_bc_dagger.py requires "
            "DistributionalFastSACTeacherBC; "
            f"got algo._target_={cfg.algo.get('_target_')!r}"
        )
    _validate_locked_topology(cfg)
    _reject_obsolete_bounded_action_controls(cfg)
    perception_path = _validate_perception_training_controls(cfg)
    _validate_actor_adopt_checkpoint_controls(
        cfg,
        perception_path=perception_path,
    )

    dagger_env_fraction = _finite_fraction(
        "algo.dagger_env_fraction",
        cfg.algo.get("dagger_env_fraction", None),
    )
    if not 0.0 < dagger_env_fraction < 1.0:
        raise ValueError(
            "algo.dagger_env_fraction must be strictly between 0 and 1"
        )
    num_envs = _positive_int("task.num_envs", cfg.task.get("num_envs", None))
    dagger_envs = min(
        num_envs,
        int(math.floor(num_envs * dagger_env_fraction + 0.5)),
    )
    if dagger_envs < 1 or dagger_envs >= num_envs:
        raise ValueError(
            "task.num_envs and algo.dagger_env_fraction must produce at least "
            "one DAgger and one pure-Student environment"
        )

    for name in (
        "dagger_safe_min_teacher_steps",
        "dagger_beta_decay_rollouts",
        "dagger_buffer_capacity",
        "student_buffer_capacity",
        "dagger_batch_size",
        "perception_replay_burn_in",
        "perception_encode_microbatch_size",
        "teacher_perception_batch_size",
    ):
        _positive_int(f"algo.{name}", cfg.algo.get(name, None))
    _positive_int(
        "algo.teacher_prefill_max_rollouts",
        cfg.algo.get("teacher_prefill_max_rollouts", None),
    )
    _positive_int(
        "algo.teacher_perception_warmup_steps",
        cfg.algo.get("teacher_perception_warmup_steps", None),
        allow_zero=True,
    )
    _validate_teacher_prefill_reachability(cfg)
    _validate_failure_phase_teacher_sampling(cfg)
    _validate_replay_contract(cfg)
    _validate_sac_controls(cfg)
    _validate_perception_training_contract(cfg)

    learning_starts = int(cfg.algo.sac_learning_starts)
    dagger_learning_starts = int(
        math.floor(learning_starts * dagger_env_fraction + 0.5)
    )
    student_learning_starts = learning_starts - dagger_learning_starts
    q_online_dagger_fraction = float(cfg.algo.q_online_dagger_replay_fraction)
    actor_online_dagger_fraction = float(
        cfg.algo.actor_online_dagger_replay_fraction
    )
    dagger_ring_required = (
        q_online_dagger_fraction > 0.0
        or actor_online_dagger_fraction > 0.0
    )
    student_ring_required = (
        q_online_dagger_fraction < 1.0
        or actor_online_dagger_fraction < 1.0
    )
    if (
        dagger_ring_required
        and int(cfg.algo.dagger_buffer_capacity) < dagger_learning_starts
    ):
        raise ValueError(
            "algo.dagger_buffer_capacity must cover the DAgger cohort's "
            "learning-start rows"
        )
    if (
        student_ring_required
        and int(cfg.algo.student_buffer_capacity) < student_learning_starts
    ):
        raise ValueError(
            "algo.student_buffer_capacity must cover the pure-Student "
            "cohort's learning-start rows"
        )

    q_update_to_data_ratio = cfg.algo.get("q_update_to_data_ratio", None)
    if (
        q_update_to_data_ratio is None
        or isinstance(q_update_to_data_ratio, bool)
        or not math.isfinite(float(q_update_to_data_ratio))
        or float(q_update_to_data_ratio) <= 0.0
    ):
        raise ValueError(
            "algo.q_update_to_data_ratio must be finite and positive"
        )
    if int(cfg.algo.q_teacher_buffer_capacity) < int(cfg.algo.sac_learning_starts):
        raise ValueError(
            "algo.q_teacher_buffer_capacity must cover algo.sac_learning_starts"
        )
    q_critic_type = str(cfg.algo.get("q_critic_type", "c51"))
    if q_critic_type not in ("c51", "scalar", "distributional"):
        raise ValueError(
            "algo.q_critic_type must be c51, scalar, or distributional"
        )
    q_twin_reduction = str(cfg.algo.get("q_twin_reduction", "min"))
    if q_twin_reduction not in ("min", "mean"):
        raise ValueError("algo.q_twin_reduction must be min or mean")
    if q_critic_type != "scalar":
        if int(cfg.algo.get("q_num_atoms", -1)) != 501:
            raise ValueError(
                "distributional FastSAC requires exactly 501 C51 atoms"
            )
        if not math.isclose(float(cfg.algo.get("q_v_min", math.nan)), -20.0):
            raise ValueError("distributional FastSAC requires q_v_min=-20")
        if not math.isclose(float(cfg.algo.get("q_v_max", math.nan)), 20.0):
            raise ValueError("distributional FastSAC requires q_v_max=20")
        if q_critic_type == "c51":
            if cfg.algo.get("q_action_fusion") not in ("late", "balanced"):
                raise ValueError(
                    "engineered C51 FastSAC requires late or balanced "
                    "Q-action fusion"
                )
            if cfg.algo.get("q_layer_norm") is not True:
                raise ValueError(
                    "engineered C51 FastSAC requires LayerNorm Q Critics"
                )
    if q_critic_type in ("scalar", "distributional") and cfg.algo.get(
        "q_use_residual_film", False
    ) is not False:
        raise ValueError(
            "standard split-stem Q requires algo.q_use_residual_film=false"
        )
    if cfg.algo.get("q_action_coordinates") != "raw_joint_command":
        raise ValueError("distributional FastSAC requires raw_joint_command Q actions")
    if cfg.algo.get("q_normalize_actions") is not True:
        raise ValueError("distributional FastSAC requires normalized Q actions")
    if not math.isclose(float(cfg.algo.get("q_action_input_gain", math.nan)), 1.0):
        raise ValueError("distributional FastSAC requires q_action_input_gain=1")
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

    if cfg.get("fastsac_bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "same-stage FastSAC resume is unsupported; use only a fresh "
            "train-phase PPO checkpoint_path"
        )
    if cfg.get("checkpoint_path", None) is None:
        raise ValueError(
            "scripts/fastSAC_bc_dagger.py requires a fresh train-phase PPO "
            "checkpoint_path"
        )
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint is not accepted; use a fresh train-phase "
            "PPO checkpoint_path"
        )

    schedule = fastsac_dagger_rollout_schedule(cfg)
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
        raise ValueError("the run must include a rollout after beta reaches zero")
    if (
        control_mode in ("safe", "hybrid")
        and safe_zero_iteration is not None
        and schedule["safe_zero_rollouts"] < 1
    ):
        raise ValueError("the run must include a rollout after SafeDAgger is off")


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="fastSAC_bc_dagger",
    version_base=None,
)
def main(cfg: DictConfig):
    # Fail before source-checkpoint I/O and the shared runner's W&B/Isaac/
    # policy construction.
    _require_single_process_execution()
    apply_fastsac_dagger_iteration_controls(cfg)
    prepare_fastsac_bc_dagger_checkpoint(cfg)
    prepare_fresh_fastsac_bc_dagger_source(cfg)
    validate_fastsac_bc_dagger_config(cfg)
    schedule = fastsac_dagger_rollout_schedule(cfg)
    print(
        "Distributional FastSAC + mean Teacher-BC schedule: "
        f"prefill=until {schedule['prefill_target_rows']} Teacher rows "
        f"(safety ceiling {schedule['prefill_max_rollouts']} rollouts), then "
        f"main={schedule['main_rollouts']}, "
        f"frames/rollout={schedule['frames_per_rollout']}; "
        f"method={EXPECTED_TRAINING_ALGORITHM}"
    )
    critic_type = str(cfg.algo.get("q_critic_type", "c51"))
    if critic_type in ("scalar", "distributional"):
        critic_architecture = (
            "balanced-state/action-stems->"
            f"{int(cfg.algo.q_hidden_dim)}x{int(cfg.algo.q_hidden_dim)}->"
            f"{'scalar' if critic_type == 'scalar' else 'C51'}"
        )
    else:
        critic_architecture = "engineered-action-branch->C51"
    print(
        "FastSAC updates: "
        f"critic={critic_type}, "
        f"q_twin_reduction={str(cfg.algo.get('q_twin_reduction', 'min'))}, "
        f"critic_architecture={critic_architecture}, "
        f"outputs_per_head={1 if critic_type == 'scalar' else int(cfg.algo.q_num_atoms)}, "
        f"policy_frequency={int(cfg.algo.sac_policy_frequency)}, "
        f"tau={float(cfg.algo.sac_tau):g}, "
        f"eta_sac={float(cfg.algo.eta_sac):g}, "
        f"lambda_bc={float(cfg.algo.lambda_bc):g}, "
        "actor_observation_mode="
        f"{cfg.algo.get('sac_actor_observation_mode', 'student_perception')}, "
        "student_actor_initialization="
        f"{cfg.algo.get('student_actor_initialization', 'teacher_bc')}, "
        "actor_adapt_checkpoint="
        f"{cfg.algo.get('actor_adopt_checkpoint_path', None)}, "
        "action_distribution="
        f"{cfg.algo.get('sac_action_distribution', NORMALIZED_TANH_ACTION_DISTRIBUTION)}, "
        f"load_noise_scale={cfg.algo.get('load_noise_scale', None)}, "
        f"alpha_init={float(cfg.algo.sac_alpha_init):g}, "
        f"alpha_update_cadence={cfg.algo.sac_alpha_update_cadence} "
        f"(every {int(cfg.algo.sac_policy_frequency)} Critic updates); "
        "perception=live Student rollout only (PPOVEL finetune)"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
