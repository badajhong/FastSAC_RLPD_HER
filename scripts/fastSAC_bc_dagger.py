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
    PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    TRAINING_ALGORITHM,
    _validate_fastsac_entropy_target_controls,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
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


def _validate_online_student_perception_contract(cfg: DictConfig) -> None:
    """Lock fresh FastSAC perception to PPOVEL's live-rollout optimizer."""
    mode = str(cfg.algo.get("perception_replay_mode", ""))
    if mode != ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE:
        raise ValueError(
            "FastSAC perception requires "
            "algo.perception_replay_mode='online_student_rollout'"
        )
    if float(cfg.algo.get("teacher_perception_replay_fraction", math.nan)) != 0.0:
        raise ValueError(
            "FastSAC live-rollout perception requires "
            "algo.teacher_perception_replay_fraction=0"
        )
    if int(cfg.algo.get("teacher_perception_warmup_steps", -1)) != 0:
        raise ValueError(
            "FastSAC live-rollout perception requires "
            "algo.teacher_perception_warmup_steps=0"
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
    if action_distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION:
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
    _validate_perception_training_controls(cfg)

    for name in (
        "dagger_safe_min_teacher_steps",
        "dagger_beta_decay_rollouts",
        "dagger_buffer_capacity",
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
    _validate_online_student_perception_contract(cfg)

    if not math.isclose(
        float(cfg.algo.get("q_update_to_data_ratio", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "FastSAC requires q_update_to_data_ratio=1 for row-level Q UTD=1"
        )
    if int(cfg.algo.q_teacher_buffer_capacity) < int(cfg.algo.sac_learning_starts):
        raise ValueError(
            "algo.q_teacher_buffer_capacity must cover algo.sac_learning_starts"
        )
    if int(cfg.algo.get("q_num_atoms", -1)) != 501:
        raise ValueError("distributional FastSAC requires exactly 501 C51 atoms")
    if not math.isclose(float(cfg.algo.get("q_v_min", math.nan)), -20.0):
        raise ValueError("distributional FastSAC requires q_v_min=-20")
    if not math.isclose(float(cfg.algo.get("q_v_max", math.nan)), 20.0):
        raise ValueError("distributional FastSAC requires q_v_max=20")
    if cfg.algo.get("q_action_fusion") != "late":
        raise ValueError("distributional FastSAC requires late Q-action fusion")
    if cfg.algo.get("q_layer_norm") is not True:
        raise ValueError("distributional FastSAC requires LayerNorm Q Critics")
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
    print(
        "Distributional FastSAC updates: "
        f"atoms={int(cfg.algo.q_num_atoms)}, "
        f"policy_frequency={int(cfg.algo.sac_policy_frequency)}, "
        f"tau={float(cfg.algo.sac_tau):g}, "
        f"eta_sac={float(cfg.algo.eta_sac):g}, "
        f"lambda_bc={float(cfg.algo.lambda_bc):g}, "
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
