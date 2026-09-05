"""Distributional FastSAC with exact mean-action Teacher BC.

This backend deliberately reuses the Teacher-only
prefill, DAgger source selection, timeout handling, and twin-C51 topology from
``td3_bc_dagger``.  The learning rule itself is SAC:

* Student collection and Actor updates use either a nominal-joint bounded,
  reparameterized tanh Gaussian or PPOVEL's raw physical-action Gaussian;
* Teacher BC is applied only to the distribution's noise-free mean, optionally
  with a detached SPReD-P probability from the online twin Critic;
* the soft Bellman target contains the next-policy entropy term;
* both online critics learn from either the complete lower-expected target or
  an equal mixture of the two complete targets, as explicitly configured; and
* there is a target critic but no target Actor or TD3 smoothing noise.

Student replay keeps the exact carried-hidden Actor inputs seen at collection;
successful Teacher episodes are re-encoded with the current EMA perception
modules through a non-serialized sidecar.  Replay observations therefore remain
input-authoritative for Q and Actor updates without a zero-hidden approximation.
Perception defaults to the current live Student rollout through the exact
PPOVEL finetune path.  An explicit ``four_way`` configuration may instead
sample the existing Student/Teacher replay strata.  The duplicate Teacher H5
export remains disabled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY, Actor, hard_copy_
from .fastsac_vel import (
    FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS,
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
    _fastsac_target_entropy,
    _reduce_actor_q_values,
)
from .ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_MAX_KEY,
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_SAFE_RELEASE_KEY,
    DAGGER_SAFE_TAKEOVER_KEY,
    DAGGER_SAFE_TEACHER_KEY,
    DAGGER_SAFE_UNSAFE_KEY,
    DAGGER_STUDENT_ACTION_VALID_KEY,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
)
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    VEL_CMD_KEY,
    PPOVEL,
)
from .perception_actor import (
    ACTOR_BC_PERCEPTION_SOURCE as PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE,
    ACTOR_INITIALIZATION_SEMANTICS as PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS,
    ACTOR_OBJECTIVE_SEMANTICS as PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS,
    OPTIMIZED_MODULES as PERCEPTION_ACTOR_OPTIMIZED_MODULES,
    TRAINING_ALGORITHM as PERCEPTION_ACTOR_TRAINING_ALGORITHM,
)
from .td3_bc_dagger import (
    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS,
    DAGGER_IS_DAGGER_ENV_KEY,
    FAILURE_PHASE_STUDENT_SOURCE_KEY,
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    ONLINE_STUDENT_ACTOR_WARMUP_SEMANTICS,
    PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
    SAC_ACTOR_OBSERVATION_MODES,
    STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
    OBJECT_GEO_REPLAY_SEMANTICS,
    PERCEPTION_PREFILL_WARMUP_SEMANTICS,
    PERCEPTION_REPLAY_SEMANTICS,
    PERCEPTION_PREFILL_DISABLED_SEMANTICS,
    PRETRAINED_PERCEPTION_MODULES,
    NEXT_Q_ACTUATOR_CONTEXT_KEY,
    REPLAY_SOURCE_ORDER,
    REPLAY_MOTION_ID_KEY,
    Q_ACTUATOR_CONTEXT_KEY,
    STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY,
    TEACHER_EPISODE_SIDECAR_SEMANTICS,
    TD3_BETA_KEY,
    TD3_COLLECTOR_NOISE_KEY,
    TD3_EXPLORATORY_STUDENT_ACTION_KEY,
    TD3_NOISE_FREE_STUDENT_ACTION_KEY,
    DistributionalTD3TeacherBC,
    DistributionalTD3TeacherBCConfig,
    online_rollout_perception_semantics,
    _DeterministicTD3StudentEvalPolicy,
    _exact_teacher_bc_loss,
    _joint_normalized_action_discrepancy,
    _polyak_update_,
    _project_c51_probabilities,
    _split_count,
    _valid_raw_action_rows,
)


TRAINING_ALGORITHM = "distributional_fastsac_teacher_bc_v1"
CHECKPOINT_VERSION = 7
PREVIOUS_CHECKPOINT_VERSION = 6
_PREVIOUS_PRE_PRIOR_CHECKPOINT_VERSION = 5
_SMOOTH_BOUNDED_STD_CHECKPOINT_VERSION = 4
ACTOR_BACKEND = "ppo_vel_nominal_joint_bounded_tanh_fastsac_bc_v3"
PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND = (
    "ppo_vel_raw_physical_joint_std_gaussian_fastsac_bc_v1"
)
NORMALIZED_TANH_ACTION_DISTRIBUTION = "normalized_tanh"
PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION = "ppo_physical_gaussian"
C51_Q_CRITIC_TYPE = "c51"
SCALAR_Q_CRITIC_TYPE = "scalar"
DISTRIBUTIONAL_Q_CRITIC_TYPE = "distributional"
Q_CRITIC_TYPES = frozenset(
    (
        C51_Q_CRITIC_TYPE,
        SCALAR_Q_CRITIC_TYPE,
        DISTRIBUTIONAL_Q_CRITIC_TYPE,
    )
)
Q_TWIN_REDUCTION_MIN = "min"
Q_TWIN_REDUCTION_MEAN = "mean"
Q_TWIN_REDUCTIONS = frozenset(
    (Q_TWIN_REDUCTION_MIN, Q_TWIN_REDUCTION_MEAN)
)
UNIFORM_PHYSICAL_STD_BOUND_MODE = "uniform_physical"
Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE = "q_normalized"
PHYSICAL_STD_BOUND_MODES = frozenset(
    (UNIFORM_PHYSICAL_STD_BOUND_MODE, Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE)
)
TEACHER_BC_STUDENT_ACTOR_INITIALIZATION = "teacher_bc"
FRESH_STUDENT_ACTOR_INITIALIZATION = "fresh"
STUDENT_ACTOR_INITIALIZATION_MODES = frozenset(
    (
        TEACHER_BC_STUDENT_ACTOR_INITIALIZATION,
        FRESH_STUDENT_ACTOR_INITIALIZATION,
    )
)
STUDENT_ACTOR_INITIALIZATION_SEMANTICS = (
    "ppo_bc_actor_adapt_or_pre_source_fresh_mean_v1"
)
ACTOR_ADOPT_CHECKPOINT_SEMANTICS = (
    "strict_perception_actor_checkpoint_actor_adapt_mean_body_only_overlay_v1"
)
PERCEPTION_ACTOR_ALGO_NAME = "teacher_rollout_perception_actor"
PERCEPTION_ACTOR_ALGO_TARGET = (
    "active_adaptation.learning.ppo.perception_actor."
    "TeacherRolloutPerceptionActor"
)
FASTSAC_ACTION_PROJECTION_KEY = "fastsac_action_projection"
FASTSAC_DAGGER_ENV_KEY = DAGGER_IS_DAGGER_ENV_KEY
FASTSAC_PREFILL_TEACHER_NOISE_KEY = "fastsac_prefill_teacher_ppo_noise_q"
FASTSAC_PREFILL_TEACHER_PROJECTION_KEY = (
    "fastsac_prefill_teacher_action_projection"
)
Q_EFFECTIVE_N_STEPS_KEY = "effective_n_steps"
_LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION = 3
_LEGACY_EFFECTIVE_LOG_STD_ACTOR_BACKEND = (
    "ppo_vel_normalized_std_tanh_bounded_fastsac_bc_v1"
)
ACTION_CONTRACT_SEMANTICS = (
    "finite_raw_joint_action_support_with_jointwise_normalized_q_bc_v1"
)


def _migrate_explicit_online_replay_capacities(contract: Mapping) -> dict:
    """Split the former total online capacity for old three-source configs."""
    migrated = dict(contract)
    if "student_buffer_capacity" in migrated:
        return migrated
    if "dagger_buffer_capacity" not in migrated:
        raise ValueError(
            "checkpoint lacks dagger_buffer_capacity for online "
            "replay-capacity migration"
        )
    if "dagger_env_fraction" not in migrated:
        raise ValueError(
            "checkpoint lacks dagger_env_fraction for online "
            "replay-capacity migration"
        )
    total = migrated["dagger_buffer_capacity"]
    fraction = migrated["dagger_env_fraction"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 2:
        raise ValueError("checkpoint has invalid legacy total online capacity")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 < float(fraction) < 1.0
    ):
        raise ValueError(
            "checkpoint has invalid dagger_env_fraction for online "
            "replay-capacity migration"
        )
    dagger_capacity, student_capacity = _split_count(total, float(fraction))
    if dagger_capacity < 1 or student_capacity < 1:
        raise ValueError(
            "legacy total capacity cannot populate both online replay rings"
        )
    migrated["dagger_buffer_capacity"] = dagger_capacity
    migrated["student_buffer_capacity"] = student_capacity
    return migrated


STUDENT_ACTION_CONTRACT_SEMANTICS = (
    "ppo_physical_mean_calibrated_nominal_joint_tanh_support_v1"
)
CRITIC_SEMANTICS = (
    "current_stochastic_actor_entropy_soft_target_lower_expected_complete_"
    "c51_distribution_projection_v1"
)
SCALAR_CRITIC_SEMANTICS = (
    "current_stochastic_actor_entropy_soft_target_clipped_twin_scalar_q_mse_v1"
)
ACTOR_MEAN_OPTIMIZER_SEMANTICS = (
    "adamw_explicit_actor_mean_weight_decay_with_unregularized_variance_v1"
)
SPRED_P_BC_SEMANTICS = (
    "online_twin_sample_std_gaussian_cdf_teacher_superiority_detached_v1"
)
_SPRED_P_SOURCE_METRIC_KEYS = (
    "spred_p_source_metadata_available",
    "spred_p_teacher_source_valid_fraction",
    "spred_p_student_source_valid_fraction",
    "spred_p_teacher_source_bc_weight_mean",
    "spred_p_student_source_bc_weight_mean",
    "spred_p_teacher_source_teacher_advantage_mean",
    "spred_p_student_source_teacher_advantage_mean",
    "spred_p_teacher_source_combined_q_std_mean",
    "spred_p_student_source_combined_q_std_mean",
    "q_filtered_bc_teacher_source_policy_better_fraction",
    "q_filtered_bc_student_source_policy_better_fraction",
)
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_nominal_joint_tanh_normal_alpha_logpi_minus_online_twin_min_"
    "plus_student_support_projected_teacher_mean_bc_v2"
)
PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_with_mean_only_joint_normalized_teacher_bc_separate_low_lr_std_"
    "adam_rollout_scale_kl_cap_and_hard_bounds_v3"
)
Q_NORMALIZED_PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_with_mean_only_joint_normalized_teacher_bc_separate_low_lr_std_"
    "adam_rollout_scale_kl_cap_and_jointwise_q_normalized_absolute_intersection_"
    "bounds_v1"
)
ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_delayed_actor_cadence_v3"
)
PPO_PHYSICAL_GAUSSIAN_ENTROPY_SEMANTICS = (
    "bounded_ppo_physical_joint_std_gaussian_nominal_joint_coordinate_log_"
    "probability_auto_or_fixed_temperature_actor_cadence_v3"
)
FASTSAC_TEACHER_PREFILL_SEMANTICS = (
    "ppo_gaussian_noisy_issued_action_clean_mean_bc_label_forced_valid_teacher_"
    "successful_episode_commit_until_replay_capacity_then_main_live_student_"
    "perception_v2"
)
_CRITIC_CADENCE_ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_v2"
)


def _seeded_dagger_env_mask(
    num_envs: int,
    fraction: float,
    seed: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a fixed, auditable DAgger cohort without index-order bias."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError("dagger_env_fraction must be a finite number in [0,1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("dagger_seed must be an integer")

    dagger_count = min(
        num_envs,
        int(math.floor(num_envs * float(fraction) + 0.5)),
    )
    # Draw on CPU with a private generator so the partition is independent of
    # policy RNG consumption and identical on CPU/CUDA. Randomized fixed IDs
    # avoid task, terrain, or motion ordering that may correlate with env index.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = torch.randperm(num_envs, generator=generator)[:dagger_count]
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[selected] = True
    return mask.to(device=device)


def _student_action_contract_metadata(
    joint_names,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    execution_contract: Mapping,
) -> dict:
    """Describe the Student policy support separately from Teacher safety.

    PPOVEL Teacher actions retain the broad finite execution guard owned by the
    inherited DAgger contract.  Student SAC actions instead use the nominal
    soft-joint interval that also defines the Q action coordinates.  Keeping a
    separate fingerprint prevents an older ``[-action_support_clip, +clip]``
    tanh policy from being resumed under this materially different policy.
    """
    names = list(joint_names)
    low = torch.as_tensor(action_low, dtype=torch.float32).detach().cpu()
    high = torch.as_tensor(action_high, dtype=torch.float32).detach().cpu()
    if low.shape != (len(names),) or high.shape != low.shape:
        raise ValueError("Student SAC action bounds must match the joint order")
    if (
        not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or not torch.all(high > low)
    ):
        raise ValueError("Student SAC action bounds must be finite and ordered")
    if not torch.all((low < 0.0) & (high > 0.0)):
        raise ValueError("Student SAC action zero must be inside every joint bound")
    execution_fingerprint = execution_contract.get("fingerprint")
    if not isinstance(execution_fingerprint, str) or not execution_fingerprint:
        raise ValueError("Student SAC contract requires an execution fingerprint")
    center = (low + high) * 0.5
    scale = (high - low) * 0.5
    payload = {
        "semantics": STUDENT_ACTION_CONTRACT_SEMANTICS,
        "source": "soft_joint_limits_at_default_pose",
        "joint_names": names,
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "action_center": center.tolist(),
        "action_scale": scale.tolist(),
        "ppo_mean_calibration": (
            "zero_action_exact_unit_local_physical_jacobian_affine_latent"
        ),
        "execution_safety_fingerprint": execution_fingerprint,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["fingerprint"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return payload


def _q_target_discount_factors(cfg, batch: Mapping[str, torch.Tensor]):
    """Return endpoint discount, gamma power, and factual replay horizon.

    ``q_n_step=1`` deliberately keeps the historical multiplication order.
    For an opt-in multi-step Student row, replay already owns the discounted
    raw reward and the product of environment discounts; this helper supplies
    only ``gamma**k`` for its endpoint soft bootstrap.
    """
    discounts = batch["discounts"]
    configured = int(getattr(cfg, "q_n_step", 1))
    if configured == 1:
        effective_n_steps = batch.get(
            Q_EFFECTIVE_N_STEPS_KEY, torch.ones_like(discounts)
        )
        return (
            float(cfg.gamma) * discounts,
            float(cfg.gamma),
            effective_n_steps,
        )

    if Q_EFFECTIVE_N_STEPS_KEY not in batch:
        raise KeyError(
            "multi-step Q replay is missing 'effective_n_steps'; rebuild the "
            "fresh replay rings"
        )
    effective_n_steps = batch[Q_EFFECTIVE_N_STEPS_KEY]
    if effective_n_steps.shape != discounts.shape:
        raise ValueError("effective_n_steps and discounts must have identical shape")
    if effective_n_steps.dtype == torch.bool:
        raise ValueError("effective_n_steps cannot be boolean")
    rounded_steps = effective_n_steps.round()
    invalid_steps = (
        ~torch.isfinite(effective_n_steps)
        | (effective_n_steps < 1)
        | (effective_n_steps > configured)
        | (effective_n_steps != rounded_steps)
    )
    if bool(invalid_steps.any()):
        raise ValueError(
            "effective_n_steps must contain finite integer horizons in "
            f"[1, {configured}]"
        )
    teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
    if teacher_source is not None:
        teacher_source = teacher_source.reshape(-1).bool()
        if teacher_source.shape != effective_n_steps.reshape(-1).shape:
            raise ValueError(
                "Q Teacher-source markers and effective_n_steps are misaligned"
            )
        if bool(
            (
                teacher_source
                & (effective_n_steps.reshape(-1) != 1)
            ).any()
        ):
            raise ValueError("Teacher Q replay rows must remain one-step")
    gamma_power = torch.pow(
        torch.full_like(discounts, float(cfg.gamma)),
        effective_n_steps.to(dtype=discounts.dtype),
    )
    return gamma_power * discounts, gamma_power, effective_n_steps


def _fastsac_action_distribution(cfg) -> str:
    """Return the explicit policy-distribution mode, defaulting legacy configs."""
    return str(
        getattr(cfg, "sac_action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION)
    )


def _fastsac_q_critic_type(cfg) -> str:
    """Return the critic family, defaulting historical configs to C51."""
    return str(getattr(cfg, "q_critic_type", C51_Q_CRITIC_TYPE))


def _fastsac_q_twin_reduction(cfg) -> str:
    """Return the Q1/Q2 reduction, defaulting historical configs to min."""
    return str(getattr(cfg, "q_twin_reduction", Q_TWIN_REDUCTION_MIN))


def _reduce_fastsac_twin_target(
    target_heads: torch.Tensor,
    selection_values: torch.Tensor,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce complete twin targets and return each head's contribution.

    ``min`` selects the complete target belonging to the lower scalar value on
    each row. ``mean`` forms an equal mixture of the two complete targets. For
    C51 this deliberately averages probability vectors, never logits or atoms.
    """
    if target_heads.ndim < 2 or target_heads.shape[0] != 2:
        raise ValueError("FastSAC twin target must have two leading heads")
    if selection_values.ndim != 2 or selection_values.shape[0] != 2:
        raise ValueError("FastSAC twin selection values must have shape [2, batch]")
    if target_heads.shape[1] != selection_values.shape[1]:
        raise ValueError("FastSAC twin target and selection batch sizes differ")
    if reduction == Q_TWIN_REDUCTION_MEAN:
        half = selection_values.new_tensor(0.5)
        return target_heads.mean(dim=0), half, half
    if reduction != Q_TWIN_REDUCTION_MIN:
        raise ValueError(f"unsupported Q twin reduction {reduction!r}")
    selected_head = selection_values.argmin(dim=0)
    index_shape = (1, target_heads.shape[1]) + (1,) * (
        target_heads.ndim - 2
    )
    index = selected_head.view(index_shape).expand(1, *target_heads.shape[1:])
    selected_target = target_heads.gather(0, index).squeeze(0)
    return (
        selected_target,
        (selected_head == 0).to(selection_values).mean(),
        (selected_head == 1).to(selection_values).mean(),
    )


def _fastsac_actor_backend(cfg) -> str:
    if _fastsac_action_distribution(cfg) == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION:
        return PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND
    return ACTOR_BACKEND


def _fastsac_actor_weight_decay(cfg) -> float:
    """Return the explicit Actor decay, defaulting legacy config objects to zero."""
    return float(getattr(cfg, "sac_actor_weight_decay", 0.0))


def _student_actor_initialization(cfg) -> str:
    """Return the fresh-PPO Student mean initialization contract."""
    return str(
        getattr(
            cfg,
            "student_actor_initialization",
            TEACHER_BC_STUDENT_ACTOR_INITIALIZATION,
        )
    )


def _actor_adopt_checkpoint_path(cfg) -> str | None:
    """Return the optional external spelling for the actor_adapt overlay."""
    path = getattr(cfg, "actor_adopt_checkpoint_path", None)
    return None if path is None else str(path)


def _checkpoint_config_mapping(checkpoint: Mapping) -> tuple[Mapping, Mapping]:
    """Return the saved root/algo config required by the overlay contract."""
    source_cfg = checkpoint.get("cfg")
    if not isinstance(source_cfg, Mapping):
        raise ValueError(
            "actor_adopt checkpoint must contain its saved cfg mapping"
        )
    source_algo = source_cfg.get("algo")
    if not isinstance(source_algo, Mapping):
        raise ValueError(
            "actor_adopt checkpoint must contain cfg.algo"
        )
    return source_cfg, source_algo


def _validate_actor_adapt_source_mapping(source_state: object) -> Mapping:
    """Validate the self-describing actor_adapt state before any target mutates."""
    if not isinstance(source_state, Mapping) or not source_state:
        raise ValueError(
            "actor_adopt checkpoint must contain a non-empty actor_adapt mapping"
        )
    invalid_keys = [key for key in source_state if not isinstance(key, str)]
    if invalid_keys:
        raise ValueError("actor_adopt checkpoint actor_adapt keys must be strings")
    for key, value in source_state.items():
        if not torch.is_tensor(value):
            raise ValueError(
                f"actor_adopt checkpoint actor_adapt key {key!r} is not a tensor"
            )
        if value.dtype != torch.float32:
            raise ValueError(
                f"actor_adopt checkpoint actor_adapt key {key!r} must be float32"
            )
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError(
                f"actor_adopt checkpoint actor_adapt key {key!r} is empty or non-finite"
            )

    def unique_suffix(suffix: str) -> tuple[str, torch.Tensor]:
        matches = [
            (key, value)
            for key, value in source_state.items()
            if key.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                "actor_adopt checkpoint actor_adapt must contain exactly one "
                f"{suffix!r} tensor"
            )
        return matches[0]

    _, actor_std = unique_suffix("actor_std")
    _, actor_mean_weight = unique_suffix("actor_mean.weight")
    _, actor_mean_bias = unique_suffix("actor_mean.bias")
    if actor_std.ndim != 1 or actor_mean_bias.ndim != 1:
        raise ValueError(
            "actor_adopt checkpoint actor_std and actor_mean bias must be vectors"
        )
    if actor_mean_weight.ndim != 2 or actor_mean_weight.shape[1] < 1:
        raise ValueError(
            "actor_adopt checkpoint actor_mean weight must be a non-empty matrix"
        )
    action_dim = int(actor_std.shape[0])
    if action_dim < 1 or tuple(actor_mean_bias.shape) != (action_dim,) or (
        int(actor_mean_weight.shape[0]) != action_dim
    ):
        raise ValueError(
            "actor_adopt checkpoint actor_std/actor_mean action dimensions disagree"
        )
    return source_state


def validate_actor_adopt_checkpoint_payload(
    checkpoint: object,
    *,
    source_path: str | None = None,
) -> tuple[Mapping, dict]:
    """Strictly audit a perception+Actor-BC source checkpoint.

    The intentionally misspelled public option is retained for the user's CLI;
    the only model child returned from this audit is the real ``actor_adapt``.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("actor_adopt checkpoint must be a top-level mapping")
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError("actor_adopt checkpoint must contain a policy mapping")
    exact_metadata = {
        "training_algorithm": PERCEPTION_ACTOR_TRAINING_ALGORITHM,
        "last_phase": "finetune",
        "actor_objective_semantics": PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS,
        "actor_initialization_semantics": (
            PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS
        ),
        "actor_adapt_loaded_from_teacher_checkpoint": True,
        "actor_adapt_trained": True,
        "actor_adapt_controls_rollout": False,
        "actor_bc_perception_source": PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE,
        "actor_bc_uses_online_priv_pred": False,
    }
    for name, expected in exact_metadata.items():
        if policy_state.get(name) != expected:
            raise ValueError(
                f"actor_adopt checkpoint metadata mismatch at policy.{name}: "
                f"expected {expected!r}, got {policy_state.get(name)!r}"
            )
    for name in ("last_iter", "actor_adapt_bc_update_count"):
        value = policy_state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"actor_adopt checkpoint policy.{name} must be a positive integer"
            )
    if tuple(policy_state.get("optimized_modules", ())) != tuple(
        PERCEPTION_ACTOR_OPTIMIZED_MODULES
    ):
        raise ValueError(
            "actor_adopt checkpoint optimized_modules does not match the "
            "perception+actor BC stage"
        )
    actor_state = _validate_actor_adapt_source_mapping(
        policy_state.get("actor_adapt")
    )
    missing_perception = [
        name
        for name in PRETRAINED_PERCEPTION_MODULES
        if not isinstance(policy_state.get(name), Mapping)
    ]
    if missing_perception:
        raise ValueError(
            "actor_adopt checkpoint lacks its jointly trained perception module "
            f"mappings: {missing_perception}"
        )

    source_cfg, source_algo = _checkpoint_config_mapping(checkpoint)
    if source_algo.get("name") != PERCEPTION_ACTOR_ALGO_NAME:
        raise ValueError("actor_adopt checkpoint cfg.algo.name is incompatible")
    if source_algo.get("_target_") != PERCEPTION_ACTOR_ALGO_TARGET:
        raise ValueError("actor_adopt checkpoint cfg.algo._target_ is incompatible")
    if source_algo.get("distill_with_priv_pred") is not True:
        raise ValueError(
            "actor_adopt checkpoint must have cfg.algo.distill_with_priv_pred=true"
        )
    if source_algo.get("actor_bc_perception_source") != (
        PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE
    ):
        raise ValueError(
            "actor_adopt checkpoint must use rollout EMA perception for Actor BC"
        )
    source_task = source_cfg.get("task")
    if not isinstance(source_task, Mapping) or not isinstance(
        source_task.get("name"), str
    ):
        raise ValueError("actor_adopt checkpoint must record cfg.task.name")
    teacher_path = source_cfg.get("checkpoint_path")
    if not isinstance(teacher_path, str) or not teacher_path.strip():
        raise ValueError(
            "actor_adopt checkpoint must record its Teacher cfg.checkpoint_path"
        )
    if source_algo.get("latent_dim") != 256:
        raise ValueError("actor_adopt checkpoint cfg.algo.latent_dim must be 256")

    provenance = {
        "semantics": ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
        "loaded": True,
        "source_path": source_path,
        "source_algorithm": policy_state.get("training_algorithm"),
        "source_phase": policy_state.get("last_phase"),
        "source_iter": int(policy_state["last_iter"]),
        "source_actor_bc_update_count": int(
            policy_state["actor_adapt_bc_update_count"]
        ),
        "source_actor_objective_semantics": policy_state.get(
            "actor_objective_semantics"
        ),
        "source_actor_initialization_semantics": policy_state.get(
            "actor_initialization_semantics"
        ),
        "source_actor_bc_perception_source": policy_state.get(
            "actor_bc_perception_source"
        ),
        "source_teacher_checkpoint_path": teacher_path,
        "source_task_name": source_task["name"],
        "module": "actor_adapt",
        "runtime_std_source": "load_noise_scale",
    }
    return actor_state, provenance


def checkpoint_module_mismatches(
    first_policy: Mapping,
    second_policy: Mapping,
    module_names,
    *,
    ignored_key_suffixes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return module names whose serialized child states are not bit-exact."""
    mismatches: list[str] = []
    for name in module_names:
        first = first_policy.get(name)
        second = second_policy.get(name)
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            mismatches.append(name)
            continue
        first_keys = {
            key
            for key in first
            if not any(str(key).endswith(suffix) for suffix in ignored_key_suffixes)
        }
        second_keys = {
            key
            for key in second
            if not any(str(key).endswith(suffix) for suffix in ignored_key_suffixes)
        }
        if first_keys != second_keys:
            mismatches.append(name)
            continue
        for key in first_keys:
            left = first[key]
            right = second[key]
            if torch.is_tensor(left):
                if (
                    not torch.is_tensor(right)
                    or left.shape != right.shape
                    or left.dtype != right.dtype
                    or not torch.equal(
                        left.detach().cpu(), right.detach().cpu()
                    )
                ):
                    mismatches.append(name)
                    break
            elif type(left) is not type(right) or left != right:
                mismatches.append(name)
                break
    return tuple(mismatches)


def _spred_p_teacher_probability(
    policy_q: torch.Tensor,
    teacher_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached SPReD-P Teacher weights from an online twin-Q ensemble.

    SPReD-P models the two action values as independent Gaussians fitted across
    Critic heads.  The returned probability is

    ``Phi((mean(Q_teacher) - mean(Q_policy)) / combined_std)``.

    The official implementation leaves this computation in the Actor graph.
    This backend deliberately uses the semi-gradient variant requested for
    stability: the probability, advantage, and uncertainty are all detached,
    so the Actor can reduce BC only by changing its action loss rather than by
    differentiating through the gate itself.
    """
    if policy_q.ndim != 2 or teacher_q.ndim != 2:
        raise ValueError("SPReD-P Q tensors must have shape [critic, batch]")
    if policy_q.shape != teacher_q.shape:
        raise ValueError("SPReD-P Student and Teacher Q shapes must match")
    if policy_q.shape[0] != 2:
        raise ValueError("low-memory SPReD-P requires exactly two Critic heads")
    if not policy_q.is_floating_point() or not teacher_q.is_floating_point():
        raise TypeError("SPReD-P Q tensors must be floating point")
    if policy_q.device != teacher_q.device or policy_q.dtype != teacher_q.dtype:
        raise ValueError("SPReD-P Q tensors must share device and dtype")

    policy_q = policy_q.detach()
    teacher_q = teacher_q.detach()
    q_values_are_finite = torch.isfinite(policy_q).all() & torch.isfinite(
        teacher_q
    ).all()
    if not q_values_are_finite:
        raise RuntimeError("SPReD-P online Critic values contain NaN/Inf")

    policy_mean = policy_q.mean(dim=0)
    teacher_mean = teacher_q.mean(dim=0)
    # torch.std's default Bessel correction matches the authors' official
    # ``torch.std(..., dim=0)`` implementation.  With the locked twin Critic,
    # each uncertainty estimate therefore uses the two available heads.
    policy_std = policy_q.std(dim=0)
    teacher_std = teacher_q.std(dim=0)
    variance_floor = torch.finfo(policy_q.dtype).eps
    combined_std = torch.sqrt(
        (policy_std.square() + teacher_std.square()).clamp_min(variance_floor)
    )
    teacher_advantage = teacher_mean - policy_mean
    z_score = teacher_advantage / combined_std
    teacher_probability = 0.5 * (1.0 + torch.erf(z_score / math.sqrt(2.0)))
    teacher_probability = teacher_probability.clamp(0.0, 1.0)
    return teacher_probability, teacher_advantage, combined_std


class FastSACPhysicalNormal:
    """Generator-aware diagonal Normal in PPOVEL physical-action coordinates.

    PPOVEL uses ``Independent(Normal(loc, scale), 1)`` without a tanh
    transform.  This compact wrapper preserves that distribution while also
    accepting FastSAC's private ``torch.Generator`` streams.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = torch.clamp_min(scale, 1.0e-6).expand_as(loc)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    def log_prob_for_action(self, action: torch.Tensor) -> torch.Tensor:
        standardized = (action - self.loc) / self.scale
        per_coordinate = (
            -0.5 * standardized.square()
            - self.scale.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        return per_coordinate.sum(dim=-1)

    def rsample_with_log_prob(
        self, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            self.loc.shape,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
        )
        action = self.loc + self.scale * noise
        return action, self.log_prob_for_action(action)


def _validate_fastsac_entropy_target_controls(
    sac_log_std_min,
    sac_log_std_max,
    sac_target_entropy_ratio,
    *,
    field_prefix: str = "",
) -> float:
    """Validate the normalized target against a nominal log-std envelope.

    The exact tanh-normal entropy also depends on the state-dependent mean and
    tanh Jacobian.  This strict unsquashed-Gaussian bracket is therefore a
    configuration sanity envelope, not an exact reachability guarantee.  It
    prevents autotune from requiring a log-std boundary or relying on mean
    saturation to reach its target.

    Returns the normalized target entropy per action dimension.
    """
    prefix = f"{field_prefix}." if field_prefix else ""
    values = (
        ("sac_log_std_min", sac_log_std_min),
        ("sac_log_std_max", sac_log_std_max),
        ("sac_target_entropy_ratio", sac_target_entropy_ratio),
    )
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{prefix}{name} must be a finite number")

    log_std_min = float(sac_log_std_min)
    log_std_max = float(sac_log_std_max)
    if not (
        math.isfinite(log_std_min)
        and math.isfinite(log_std_max)
        and log_std_min < log_std_max
    ):
        raise ValueError(
            f"{prefix}sac_log_std_min must be finite and below {prefix}sac_log_std_max"
        )

    target_entropy_ratio = float(sac_target_entropy_ratio)
    if not math.isfinite(target_entropy_ratio) or target_entropy_ratio <= 0.0:
        raise ValueError(
            f"{prefix}sac_target_entropy_ratio must be finite and positive"
        )

    gaussian_entropy_offset = 0.5 * math.log(2.0 * math.pi * math.e)
    min_unsquashed_entropy_per_dim = gaussian_entropy_offset + log_std_min
    max_unsquashed_entropy_per_dim = gaussian_entropy_offset + log_std_max
    target_entropy_per_dim = -target_entropy_ratio
    if not (
        min_unsquashed_entropy_per_dim
        < target_entropy_per_dim
        < max_unsquashed_entropy_per_dim
    ):
        raise ValueError(
            "FastSAC entropy target is unreachable within the nominal "
            "unsquashed-Gaussian log-std envelope: require "
            "0.5*log(2*pi*e) + sac_log_std_min "
            "< -sac_target_entropy_ratio "
            "< 0.5*log(2*pi*e) + sac_log_std_max"
        )
    return target_entropy_per_dim


@dataclass
class DistributionalFastSACTeacherBCConfig(DistributionalTD3TeacherBCConfig):
    """Hydra surface for fresh-only FastSAC + Teacher BC."""

    _target_: str = (
        "active_adaptation.learning.ppo.fastsac_bc_dagger."
        "DistributionalFastSACTeacherBC"
    )
    name: str = "fastsac_bc_dagger"

    # The frozen privileged Teacher always loads from the PPO checkpoint.
    # ``teacher_bc`` also loads that checkpoint's distilled actor_adapt mean,
    # preserving all existing runs. ``fresh`` retains the constructor-created
    # actor_adapt mean while still resetting its PPO std through
    # ``load_noise_scale`` and independently applying any perception overlay.
    student_actor_initialization: str = TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
    # Optional final actor_adapt-only overlay from percetpion_actor.py.  The
    # public ``adopt`` spelling is retained exactly for command compatibility;
    # no Teacher, Critic, Q, or perception tensor is loaded through this field.
    actor_adopt_checkpoint_path: str | None = None

    dagger_control_mode: str = "beta"
    dagger_beta_start: float = 0.0
    dagger_beta_end: float = 0.0
    # Fixed seeded cohort used only after Teacher prefill. DAgger environments
    # alternate Student/Teacher control every step and reset their phase to
    # Student on ``is_init``; the complement always executes the Student.
    dagger_env_fraction: float = 0.5
    # Independent online replay allocations. ``dagger_buffer_capacity`` owns
    # only mixed-control DAgger rows; ``student_buffer_capacity`` owns only
    # pure-Student rows. The defaults preserve the prior 131072-row total.
    dagger_buffer_capacity: int = 65_536
    student_buffer_capacity: int = 65_536

    # During successful-only Teacher prefill, reproduce PPOVEL's diagonal
    # physical-action Gaussian.  Q receives the factual noisy command while
    # Actor BC retains the clean Teacher mean. Main DAgger/eval remain mean-only.
    teacher_prefill_use_ppo_noise: bool = True

    # SAC replaces every TD3 noise mechanism.  These inherited fields remain
    # only because the raw-replay base dataclass owns them.
    eta_td3: float = 0.0
    target_policy_noise_std: float = 0.0
    target_policy_noise_clip: float = 0.0
    collector_exploration_noise_std: float = 0.0
    collector_exploration_noise_clip: float = 0.0
    policy_delay: int = 8
    td3_learning_starts: int = 8192

    # Process larger replay chunks on the target 5090 so the increased Q UTD
    # does not multiply small depth-encoder launches.
    perception_encode_microbatch_size: int = 512
    # The default matches PPOVEL finetune during main training: every main
    # iteration uses only its live recurrent Student rollout (for the
    # production setup, [512, 32]).  An optional one-shot Teacher-replay
    # post-prefill frozen-Actor Student rollout warm-up can train live
    # perception and the configured mixed-source Critic before Actor learning.
    # ``four_way`` instead opts into the raw replay sampler during main training.
    teacher_perception_replay_fraction: float = 0.0
    perception_replay_mode: str = ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    teacher_perception_warmup_steps: int = 0

    # Diagnostic observation isolation. ``student_perception`` is the
    # deployable PPOVEL path. ``privileged_oracle`` keeps the same Student SAC
    # Actor but substitutes the frozen Teacher encoder's GT latent for
    # ``priv_pred`` in rollout, replay, targets, and evaluation.
    sac_actor_observation_mode: str = STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE

    eta_sac: float = 1e-4
    lambda_bc: float = 1.0
    # Weight Teacher BC continuously with a detached SPReD-P probability from
    # the online twin Critic.  The legacy option name is retained so existing
    # Hydra commands do not need another migration.
    use_q_filtered_bc: bool = False
    # The default ``c51`` uses the engineered action branch and categorical
    # heads. ``scalar`` uses independent balanced state/action-stem MLPs with
    # scalar Bellman outputs. ``distributional`` keeps those exact split-stem
    # inputs and hidden layers but replaces only the final scalar with C51
    # logits.
    q_critic_type: str = C51_Q_CRITIC_TYPE
    # One reduction controls both the stochastic Actor objective and the
    # bootstrap target. ``min`` is clipped double-Q; ``mean`` averages scalar
    # heads or forms an equal mixture of complete C51 distributions.
    q_twin_reduction: str = Q_TWIN_REDUCTION_MIN
    # Give Q episode-constant delayed-actuator parameters. Engineered ``c51``
    # consumes them in its action branch; both split-stem modes consume them as
    # state context. Actor observations and execution remain unchanged.
    q_condition_on_actuator_state: bool = False
    # Add the exact queue/lerp counterfactual response of each candidate action.
    q_use_predicted_effect: bool = False
    # Bounded delay/alpha-conditioned residual FiLM on the Q action stem.
    q_use_residual_film: bool = False
    q_residual_film_scale: float = 0.1
    sac_actor_lr: float = 3e-4
    # Actor regularization is an independent learning-rule choice.  In
    # particular, the Critic's q_weight_decay must never leak into the
    # pretrained PPOVEL mean: at an exact BC solution such leakage would move
    # the Actor even though both configured Actor objectives have zero gradient.
    sac_actor_weight_decay: float = 0.0
    # ``normalized_tanh`` preserves the historical FastSAC policy.  The
    # opt-in ``ppo_physical_gaussian`` mode uses PPOVEL's raw physical-action
    # mean and its direct per-joint Actor.actor_std parameter. The ordinary SAC
    # Q+entropy objective chooses the std direction; dedicated controls below
    # only slow and bound its movement.
    sac_action_distribution: str = NORMALIZED_TANH_ACTION_DISTRIBUTION
    # PPOVEL finetune starts every joint at load_noise_scale=0.5 and uses a
    # small Adam LR. A rollout-old scale-only KL cap is a trust guard, never a
    # destination or checkpoint-derived prior.
    sac_physical_std_lr: float = 1e-5
    sac_physical_std_max_kl: float = 0.01
    # ``uniform_physical`` reproduces the historical scalar interval exactly.
    # ``q_normalized`` intersects that absolute interval with per-joint bounds
    # expressed in the Critic's dimensionless nominal-action coordinates.  It
    # prevents one raw std from meaning radically different exploration on
    # small- and large-range joints without replacing SAC's Q+entropy gradient.
    sac_physical_std_bound_mode: str = UNIFORM_PHYSICAL_STD_BOUND_MODE
    sac_physical_std_min: float = 0.05
    sac_physical_std_max: float = 0.5
    sac_physical_std_normalized_min: float = 0.02
    sac_physical_std_normalized_max: float = 0.11
    sac_initial_action_std: float = 0.1
    sac_log_std_min: float = -10.0
    sac_log_std_max: float = -2.0
    sac_alpha_init: float = 1e-5
    sac_alpha_lr: float = 2e-5
    sac_use_autotune: bool = True
    sac_alpha_update_cadence: str = "actor"
    sac_target_entropy_ratio: float = 1.0
    sac_policy_frequency: int = 8
    sac_learning_starts: int = 8192
    sac_tau: float = 0.005
    sac_max_grad_norm: float = 1.0

    q_updates_per_rollout: int = 32
    q_update_to_data_ratio: float | None = 1.0


ConfigStore.instance().store(
    "fastsac_bc_dagger_finetune",
    node=DistributionalFastSACTeacherBCConfig(
        in_keys=(
            CMD_KEY,
            OBS_KEY,
            OBJECT_KEY,
            OBS_PRIV_KEY,
            OBJECT_GEO_KEY,
            VEL_CMD_KEY,
            DEPTH_KEY,
        )
    ),
    group="algo",
)


class _DistributionalFastSACDaggerRolloutPolicy(_DaggerRolloutPolicy):
    """Stochastic Student collection with a fixed alternating DAgger cohort."""

    def __init__(self, owner: "DistributionalFastSACTeacherBC"):
        super().__init__(owner)
        # False means the next action is Student. This transient collector state
        # deliberately restarts at Student after a fresh env reset or process
        # resume and is never part of the model checkpoint.
        self._dagger_teacher_turn: torch.Tensor | None = None
        self._fixed_dagger_env_mask: torch.Tensor | None = None
        self._fixed_dagger_env_mask_key: tuple[int, torch.device] | None = None

    def dagger_env_mask(
        self,
        num_envs: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Expose the exact fixed cohort used by this rollout policy."""
        owner = self._owner
        resolve_count = getattr(owner, "_parallel_env_count", None)
        if callable(resolve_count):
            num_envs = resolve_count(num_envs)
        elif num_envs is None:
            env = getattr(owner, "env", None)
            num_envs = getattr(env, "num_envs", None)
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if device is None:
            device = getattr(owner, "device", torch.device("cpu"))
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        key = (int(num_envs), target_device)
        if (
            self._fixed_dagger_env_mask is not None
            and self._fixed_dagger_env_mask_key == key
        ):
            return self._fixed_dagger_env_mask

        owner_mask = getattr(owner, "dagger_env_mask", None)
        if callable(owner_mask):
            mask = owner_mask(num_envs=int(num_envs), device=target_device)
        else:
            mask = _seeded_dagger_env_mask(
                int(num_envs),
                float(getattr(owner.cfg, "dagger_env_fraction", 1.0)),
                int(getattr(owner.cfg, "dagger_seed", 0)),
                device=target_device,
            )
        mask = torch.as_tensor(mask, device=target_device)
        if mask.dtype != torch.bool or mask.shape != (int(num_envs),):
            raise RuntimeError("fixed DAgger environment mask has an invalid schema")
        if hasattr(owner.cfg, "dagger_env_fraction") and (
            not bool(mask.any()) or bool(mask.all())
        ):
            raise ValueError(
                "three-source collection requires at least one DAgger and one "
                "Student-only vector environment"
            )
        self._fixed_dagger_env_mask = mask
        self._fixed_dagger_env_mask_key = key
        return mask

    def student_only_env_mask(
        self,
        num_envs: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Expose the fixed pure-Student cohort used for masked metrics."""
        return ~self.dagger_env_mask(num_envs=num_envs, device=device)

    def _dagger_env_mask(self, reference: torch.Tensor) -> torch.Tensor:
        owner = self._owner
        fraction = getattr(owner.cfg, "dagger_env_fraction", None)
        if fraction is None:
            # Compact legacy/unit seams predate the partition and retain their
            # historical all-DAgger beta/SafeDAgger behavior.
            return torch.ones_like(reference, dtype=torch.bool)
        mask = self.dagger_env_mask(
            num_envs=reference.numel(), device=reference.device
        )
        if mask.shape != (reference.numel(),):
            raise RuntimeError("fixed DAgger environment mask has an invalid shape")
        return mask.reshape(reference.shape)

    def _current_dagger_teacher_turn(
        self,
        dagger_env: torch.Tensor,
        reset: torch.Tensor,
    ) -> torch.Tensor:
        """Return this step's source phase without advancing a failed issue."""
        state = self._dagger_teacher_turn
        if (
            state is None
            or state.shape != dagger_env.shape
            or state.device != dagger_env.device
        ):
            state = torch.zeros_like(dagger_env)
        # The first action of every reset episode is always Student, independent
        # for each environment. The phase then alternates on every control step.
        current = torch.where(reset, torch.zeros_like(state), state) & dagger_env
        return current

    @torch.no_grad()
    def forward(self, td: TensorDict):
        owner = self._owner
        teacher_prefill_active = owner._teacher_prefill_active()
        teacher_prefill_ppo_noise = teacher_prefill_active and bool(
            getattr(owner.cfg, "teacher_prefill_use_ppo_noise", False)
        )
        raw_student_mean = owner._student_raw_action_proposal(td)
        cache_enabled = getattr(
            owner, "_student_collection_actor_cache_enabled", lambda: False
        )
        if cache_enabled():
            td[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY] = (
                owner._collection_actor_observations(td)
            )
        for scratch_key in (
            "_depth_feature",
            OBJECT_PRED_KEY,
            OBJECT_PRED_TRANS_KEY,
            "_actor_inp",
            "_actor_feature",
            "_object_adapt_mlp_inp",
            "_obj_adapt_mlp",
            "_object_adapt_inp",
            "_adapt_inp",
            "loc",
            "scale",
            "sample_log_prob",
        ):
            if scratch_key in td.keys(True, True):
                td.del_(scratch_key)

        if teacher_prefill_ppo_noise:
            teacher_action, teacher_scale = owner._teacher_action_statistics(
                td.clone(False)
            )
            if teacher_scale.shape != teacher_action.shape:
                raise ValueError(
                    "PPO Teacher mean and scale must have identical shapes"
                )
        else:
            teacher_action = owner._teacher_action(td.clone(False))
            teacher_scale = None
        valid = _valid_raw_action_rows(teacher_action)
        if teacher_scale is not None:
            valid = valid & torch.isfinite(teacher_scale).all(dim=-1) & (
                teacher_scale > 0.0
            ).all(dim=-1)
        finite_teacher_action = torch.nan_to_num(
            teacher_action, nan=0.0, posinf=0.0, neginf=0.0
        )
        bounded_teacher_action = owner._project_execution_action(finite_teacher_action)
        if teacher_scale is not None:
            finite_teacher_scale = torch.nan_to_num(
                teacher_scale, nan=1.0e-6, posinf=1.0e-6, neginf=1.0e-6
            ).clamp_min(1.0e-6)
            raw_prefill_teacher_action, _ = FastSACPhysicalNormal(
                finite_teacher_action, finite_teacher_scale
            ).rsample_with_log_prob(generator=owner.teacher_prefill_action_rng)
            bounded_prefill_teacher_action = owner._project_execution_action(
                raw_prefill_teacher_action
            )
            prefill_teacher_projection = valid & (
                bounded_prefill_teacher_action != raw_prefill_teacher_action
            ).any(dim=-1)
        else:
            bounded_prefill_teacher_action = bounded_teacher_action
            prefill_teacher_projection = torch.zeros_like(valid)
        student_valid = torch.isfinite(raw_student_mean).all(dim=-1)
        finite_student_mean = torch.nan_to_num(
            raw_student_mean, nan=0.0, posinf=0.0, neginf=0.0
        )
        student_dist = owner._sac_dist_from_mean(finite_student_mean)
        mean_student_action = student_dist.mean

        # Safety and beta compare the noise-free BC quantity.  Exploration
        # itself must not trigger Teacher takeovers or bias source selection.
        discrepancy_rms, discrepancy_max = _joint_normalized_action_discrepancy(
            mean_student_action,
            bounded_teacher_action,
            owner._fastsac_q_action_scale,
        )
        discrepancy_rms = torch.where(
            valid, discrepancy_rms, torch.zeros_like(discrepancy_rms)
        )
        discrepancy_max = torch.where(
            valid, discrepancy_max, torch.zeros_like(discrepancy_max)
        )
        reset = td.get("is_init", None)
        reset = (
            torch.zeros_like(valid)
            if reset is None
            else reset.reshape(valid.shape).bool()
        )
        dagger_env = self._dagger_env_mask(valid)
        partition_enabled = hasattr(owner.cfg, "dagger_env_fraction")

        control_mode = owner._effective_control_mode()
        safe_teacher_mask = torch.zeros_like(valid)
        safe_unsafe = torch.zeros_like(valid)
        safe_takeover = torch.zeros_like(valid)
        safe_release = torch.zeros_like(valid)
        if teacher_prefill_active:
            # Prefill always selects a valid Teacher. Its optional PPOVEL draw
            # owns a separate RNG, so collection duration cannot shift either
            # Student SAC sampling stream.
            self._dagger_teacher_turn = None
            choose_teacher = valid
            beta_teacher = torch.zeros_like(valid)
        elif partition_enabled:
            teacher_turn = self._current_dagger_teacher_turn(dagger_env, reset)
            # Scheduled Teacher turns fall back to the finite Student action if
            # the Teacher label is invalid. Conversely, only DAgger-cohort
            # Student turns may use a valid Teacher as a non-finite-Student
            # safety fallback. The Student-only cohort never executes Teacher.
            choose_teacher = (teacher_turn & valid) | (
                dagger_env & ~student_valid & valid
            )
            beta_teacher = torch.zeros_like(valid)
        elif control_mode in ("safe", "hybrid"):
            if owner._safe_teacher_control_enabled():
                (
                    safe_teacher_mask,
                    safe_unsafe,
                    safe_takeover,
                    safe_release,
                ) = self._safe_selection(valid, student_valid, discrepancy_rms, reset)
            else:
                safe_unsafe = valid & (
                    ~student_valid
                    | (discrepancy_rms > float(owner.cfg.dagger_safe_takeover_rms))
                )
                self._safe_teacher_active = torch.zeros_like(valid)
                self._safe_teacher_hold = torch.zeros_like(valid, dtype=torch.long)
        elif control_mode != "beta":
            raise RuntimeError(f"Unsupported dagger_control_mode={control_mode!r}")

        scheduled_beta = owner._teacher_mixture_probability()
        if not teacher_prefill_active:
            if not partition_enabled:
                beta_teacher = torch.zeros_like(valid)
                if control_mode in ("beta", "hybrid"):
                    beta_teacher = (
                        (
                            torch.rand(
                                valid.shape,
                                device=valid.device,
                                generator=owner.dagger_rng,
                            )
                            < scheduled_beta
                        )
                        & valid
                        & ~safe_teacher_mask
                    )
                choose_teacher = safe_teacher_mask | beta_teacher
            sampled_student_action, _ = student_dist.rsample_with_log_prob(
                generator=owner.sac_rollout_rng
            )
            # A non-finite raw Student proposal remains auditable as invalid,
            # but the pure-Student cohort must still execute a finite Student
            # command rather than silently switching sources. Use the sanitized
            # distribution mean as the deterministic safety fallback.
            sampled_student_action = torch.where(
                student_valid.unsqueeze(-1),
                sampled_student_action,
                mean_student_action,
            )
        else:
            sampled_student_action = mean_student_action

        invalid_both = ~valid & ~student_valid
        if teacher_prefill_active and invalid_both.any():
            raise RuntimeError(
                "Neither Teacher nor FastSAC Student produced a finite raw action"
            )
        if not teacher_prefill_active and partition_enabled:
            if (invalid_both & dagger_env).any():
                raise RuntimeError(
                    "Neither Teacher nor FastSAC Student produced a finite raw "
                    "action in a DAgger environment"
                )
        else:
            choose_teacher = choose_teacher | (valid & ~student_valid)
        if not torch.isfinite(sampled_student_action).all():
            raise RuntimeError("FastSAC sampled a non-finite raw action")

        projected_student_action = owner._project_execution_action(
            sampled_student_action
        )
        student_projection = (~choose_teacher) & (
            projected_student_action != sampled_student_action
        ).any(dim=-1)
        issued_action = torch.where(
            choose_teacher.unsqueeze(-1),
            bounded_prefill_teacher_action,
            projected_student_action,
        )
        if not teacher_prefill_active and partition_enabled:
            # Advance only once a finite command has actually been selected.
            # Each DAgger row flips independently; reset rows selected Student
            # above and therefore correctly schedule Teacher on their next step.
            self._dagger_teacher_turn = dagger_env & ~teacher_turn
        # Both branches have already passed through the authoritative execution
        # projection. Selecting between them cannot leave that finite support,
        # so projecting the combined tensor again only repeats the same kernels.
        sample_q_deviation = owner._q_action_input(sampled_student_action) - (
            owner._q_action_input(mean_student_action)
        )
        prefill_teacher_q_noise = owner._q_action_input(
            bounded_prefill_teacher_action
        ) - owner._q_action_input(bounded_teacher_action)
        prefill_teacher_q_noise = torch.where(
            (choose_teacher & teacher_prefill_ppo_noise).unsqueeze(-1),
            prefill_teacher_q_noise,
            torch.zeros_like(prefill_teacher_q_noise),
        )

        td[ACTION_KEY] = issued_action
        td[FASTSAC_DAGGER_ENV_KEY] = dagger_env
        td[DAGGER_TEACHER_ACTION_KEY] = bounded_teacher_action
        td[DAGGER_TEACHER_ACTION_VALID_KEY] = valid
        td[DAGGER_IS_STUDENT_ACTION_KEY] = ~choose_teacher
        td[DAGGER_ACTION_DISCREPANCY_RMS_KEY] = discrepancy_rms
        td[DAGGER_ACTION_DISCREPANCY_MAX_KEY] = discrepancy_max
        td[DAGGER_SAFE_UNSAFE_KEY] = safe_unsafe
        td[DAGGER_SAFE_TEACHER_KEY] = safe_teacher_mask
        td[DAGGER_SAFE_TAKEOVER_KEY] = safe_takeover
        td[DAGGER_SAFE_RELEASE_KEY] = safe_release
        td[DAGGER_BETA_TEACHER_KEY] = beta_teacher
        td[DAGGER_STUDENT_ACTION_VALID_KEY] = student_valid

        # The inherited raw replay schema names these audit fields after TD3,
        # but their FastSAC values are respectively mean, stochastic sample,
        # and sample-minus-mean in the joint-wise Q coordinates.
        td[TD3_NOISE_FREE_STUDENT_ACTION_KEY] = mean_student_action
        td[TD3_EXPLORATORY_STUDENT_ACTION_KEY] = sampled_student_action
        td[TD3_COLLECTOR_NOISE_KEY] = sample_q_deviation
        td[FASTSAC_ACTION_PROJECTION_KEY] = student_projection
        td[FASTSAC_PREFILL_TEACHER_NOISE_KEY] = prefill_teacher_q_noise
        td[FASTSAC_PREFILL_TEACHER_PROJECTION_KEY] = (
            prefill_teacher_projection & choose_teacher
        )
        td[TD3_BETA_KEY] = torch.full_like(discrepancy_rms, float(scheduled_beta))
        motion_ids = getattr(
            getattr(getattr(owner, "env", None), "command_manager", None),
            "motion_ids",
            None,
        )
        if torch.is_tensor(motion_ids):
            motion_ids = motion_ids.detach().to(device=valid.device, dtype=torch.long)
            if motion_ids.numel() != valid.numel():
                raise RuntimeError(
                    "command motion_ids do not match rollout environments"
                )
            td[REPLAY_MOTION_ID_KEY] = motion_ids.reshape(valid.shape)
        return td


class _DeterministicFastSACStudentEvalPolicy(_DeterministicTD3StudentEvalPolicy):
    """Mode-consistent Student mean; never samples or computes log-prob."""


class DistributionalFastSACTeacherBC(DistributionalTD3TeacherBC):
    """Twin-C51 FastSAC with stochastic SAC and exact mean-only Teacher BC."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        # Transient device-local cohort cache. It is intentionally excluded
        # from checkpoints: the fixed mask is reproducible from config alone.
        self._fixed_dagger_env_mask_cache: dict[
            tuple[int, torch.device], torch.Tensor
        ] = {}
        # SAC owns no target Actor.  The TD3 base initializes this to None; keep
        # that explicit invariant for checkpoint and runtime inspection.
        self.actor_target = None
        self.actor_backend = _fastsac_actor_backend(cfg)
        self._configure_student_action_support()

        self._fastsac_entropy_reference_log_scale_sum = float(
            torch.log(self._fastsac_q_action_scale).sum().item()
        )
        physical_gaussian = self._uses_ppo_physical_gaussian()
        if physical_gaussian:
            # Registered but inert compatibility state for old checkpoint and
            # module-tree loaders. PPOVEL's actor_std owns physical variance.
            initial_log_std = self._fastsac_q_action_scale.new_zeros(
                (self.action_dim,)
            )
            initial_raw_log_std = initial_log_std.clone()
        else:
            initial_log_std = torch.log(
                self._fastsac_q_action_scale.new_full(
                    (self.action_dim,), float(cfg.sac_initial_action_std)
                )
            )
            if (initial_log_std <= float(cfg.sac_log_std_min)).any() or (
                initial_log_std >= float(cfg.sac_log_std_max)
            ).any():
                raise ValueError(
                    "sac_initial_action_std must map strictly inside normalized "
                    "log-std bounds"
                )
            initial_raw_log_std = self._inverse_smooth_log_std(
                initial_log_std,
                float(cfg.sac_log_std_min),
                float(cfg.sac_log_std_max),
            )
        self.register_buffer(
            "_fastsac_initial_log_std", initial_log_std.detach().clone()
        )
        self.register_buffer(
            "_fastsac_initial_raw_log_std", initial_raw_log_std.detach().clone()
        )
        self.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
            self.action_dim, initial_raw_log_std, self.device
        )
        if physical_gaussian:
            # PPOVEL owns the directly parameterized per-joint std.  The
            # normalized-tanh adapter remains registered only so historical
            # source/checkpoint topology stays explicit and loadable.
            self.bc_dagger_sac_adapter.requires_grad_(False)
            self._configure_training_actor_std()
        else:
            self._freeze_legacy_actor_std()

        actor_std_parameter = (
            self._ppo_actor_std_parameter() if physical_gaussian else None
        )
        actor_mean_parameters = tuple(
            parameter
            for parameter in self.actor_adapt.parameters()
            if parameter.requires_grad and parameter is not actor_std_parameter
        )
        if not actor_mean_parameters:
            raise RuntimeError("FastSAC Actor has no trainable mean parameters")
        # ``normalized_tanh`` retains the historical joint mean/std Actor
        # optimizer. In physical Gaussian mode the same SAC objective supplies
        # mean and std gradients, but the direct PPOVEL actor_std is stepped by
        # its own low-LR Adam so mean learning-rate/clipping choices cannot make
        # exploration scale jump.
        std_parameters = (
            (actor_std_parameter,)
            if physical_gaussian
            else tuple(self.bc_dagger_sac_adapter.parameters())
        )
        if not std_parameters or {
            id(parameter) for parameter in actor_mean_parameters
        }.intersection(id(parameter) for parameter in std_parameters):
            raise RuntimeError(
                "FastSAC Actor mean and variance optimizer groups must be "
                "non-empty and disjoint"
            )
        actor_optimizer_groups = [
            {
                "params": actor_mean_parameters,
                "lr": float(cfg.sac_actor_lr),
                "weight_decay": _fastsac_actor_weight_decay(cfg),
            }
        ]
        if not physical_gaussian:
            actor_optimizer_groups.append(
                {
                    "params": std_parameters,
                    "lr": float(cfg.sac_actor_lr),
                    "weight_decay": 0.0,
                }
            )
        self.actor_optimizer = torch.optim.AdamW(
            actor_optimizer_groups,
            lr=float(cfg.sac_actor_lr),
            betas=(0.9, 0.95),
        )
        self.actor_std_optimizer = (
            torch.optim.Adam(
                std_parameters,
                lr=float(cfg.sac_physical_std_lr),
                weight_decay=0.0,
            )
            if physical_gaussian
            else None
        )
        if physical_gaussian:
            self._project_physical_actor_std_()
        self.log_alpha = nn.Parameter(
            torch.tensor(
                math.log(float(cfg.sac_alpha_init)),
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=bool(cfg.sac_use_autotune),
        )
        self.alpha_optimizer = (
            torch.optim.Adam(
                (self.log_alpha,), lr=float(cfg.sac_alpha_lr), betas=(0.9, 0.95)
            )
            if bool(cfg.sac_use_autotune)
            else None
        )
        self.target_entropy = _fastsac_target_entropy(
            self._fastsac_q_action_center - self._fastsac_q_action_scale,
            self._fastsac_q_action_center + self._fastsac_q_action_scale,
            float(cfg.sac_target_entropy_ratio),
        )
        if physical_gaussian:
            # These bounds depend only on the immutable action contract and
            # configured std interval. Materialize their host scalars once;
            # rollout diagnostics and checkpoint metadata reuse the cache.
            self._fastsac_physical_entropy_bounds = (
                self._physical_normalized_entropy_bounds()
            )
            if bool(cfg.sac_use_autotune):
                self._validate_physical_entropy_target_reachable()
        generator_device = torch.device(device)
        self.sac_action_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 1
        )
        self.sac_rollout_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 2
        )
        self.teacher_prefill_action_rng = torch.Generator(
            device=generator_device
        ).manual_seed(int(cfg.q_seed) + 4)
        self.alpha_update_count = 0
        self.actor_std_update_count = 0
        self._last_fastsac_diagnostics: dict[str, float] = {}
        self._actor_initialization = {
            "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
            "mode": _student_actor_initialization(cfg),
            "teacher_actor_loaded": False,
            "actor_adapt_mean_loaded": False,
            "actor_adapt_mean_fresh": True,
            "source_phase": None,
            "source_iter": None,
        }
        self._actor_adopt_initialization = {
            "semantics": ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
            "loaded": False,
            "source_path": None,
            "source_algorithm": None,
            "source_phase": None,
            "source_iter": None,
            "source_actor_bc_update_count": None,
            "source_actor_objective_semantics": None,
            "source_actor_initialization_semantics": None,
            "source_actor_bc_perception_source": None,
            "source_teacher_checkpoint_path": None,
            "source_task_name": None,
            "module": "actor_adapt",
            "runtime_std_source": "load_noise_scale",
            "perception_source_path": getattr(
                cfg, "perception_checkpoint_path", None
            ),
            "perception_exact_match": None,
            "perception_mismatched_modules": (),
        }
        if _student_actor_initialization(cfg) == FRESH_STUDENT_ACTOR_INITIALIZATION:
            # Capture the true constructor state before any checkpoint loader
            # can mutate actor_adapt. This transient snapshot is discarded as
            # soon as a source or same-stage checkpoint has been applied.
            self._fresh_student_actor_constructor_state = copy.deepcopy(
                self.actor_adapt.state_dict()
            )
            self._fresh_student_actor_constructor_parameter_ids = tuple(
                id(parameter) for parameter in self.actor_adapt.parameters()
            )

    def _configure_student_action_support(self) -> None:
        """Use nominal joint coordinates as the one Student SAC action Box."""
        center = self._fastsac_q_action_center.detach()
        scale = self._fastsac_q_action_scale.detach()
        low = center - scale
        high = center + scale
        if (
            not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
            or not torch.all((low < 0.0) & (high > 0.0))
        ):
            raise ValueError("FastSAC Student nominal action support is invalid")
        self._fastsac_student_action_low = low
        self._fastsac_student_action_high = high
        self._fastsac_student_action_center = center
        self._fastsac_student_action_scale = scale
        self._fastsac_student_action_contract = _student_action_contract_metadata(
            self.joint_names,
            low,
            high,
            self._fastsac_action_contract,
        )

    def _q_replay_prefill_storage_fields(self) -> tuple[str, ...]:
        """Keep noisy Q actions and clean Teacher BC labels side by side."""
        fields = super()._q_replay_storage_fields()
        if (
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
            and DAGGER_REPLAY_TEACHER_ACTIONS not in fields
        ):
            fields = (*fields, DAGGER_REPLAY_TEACHER_ACTIONS)
        return fields

    def _apply_actor_optimizer_weight_decay_contract(self) -> None:
        """Install the explicit Actor decay after construction or resume.

        Adam/AdamW optimizer state dictionaries own their param-group options.
        Loading a checkpoint therefore overwrites constructor values.  Older
        FastSAC checkpoints accidentally stored ``q_weight_decay`` on the Actor
        mean group; normalize that legacy metadata to the explicit Actor
        contract while retaining its moments and step counters.
        """
        groups = self.actor_optimizer.param_groups
        if not groups:
            raise RuntimeError("FastSAC Actor optimizer has no parameter groups")
        groups[0]["weight_decay"] = _fastsac_actor_weight_decay(self.cfg)
        # The optional second group is the normalized-tanh variance adapter.
        # Physical Gaussian variance has a separate Adam optimizer.  Neither is
        # part of the Actor-mean regularization contract.
        for group in groups[1:]:
            group["weight_decay"] = 0.0

    def _validate_actor_optimizer_weight_decay_contract(self) -> None:
        groups = self.actor_optimizer.param_groups
        if not groups:
            raise RuntimeError("FastSAC Actor optimizer has no parameter groups")
        expected = (_fastsac_actor_weight_decay(self.cfg),) + (0.0,) * (
            len(groups) - 1
        )
        for index, (group, expected_decay) in enumerate(zip(groups, expected)):
            decay = group.get("weight_decay", 0.0)
            if (
                isinstance(decay, bool)
                or not isinstance(decay, (int, float))
                or not math.isfinite(float(decay))
                or not math.isclose(
                    float(decay), expected_decay, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise RuntimeError(
                    "FastSAC Actor optimizer weight-decay contract mismatch at "
                    f"param group {index}: expected {expected_decay}, got {decay!r}"
                )

    def _student_collection_actor_cache_enabled(self) -> bool:
        """Use live carried-hidden Actor inputs in the locked online mode."""
        return str(
            getattr(self.cfg, "perception_replay_mode", "legacy_online_student")
        ) == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE

    def _parallel_env_count(self, num_envs: int | None) -> int:
        """Resolve the vector width for public fixed-cohort mask consumers."""
        if num_envs is None:
            env = getattr(self, "env", None)
            num_envs = getattr(env, "num_envs", None)
            if num_envs is None:
                batch_size = getattr(env, "batch_size", None)
                if batch_size is not None and len(batch_size):
                    num_envs = int(batch_size[0])
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(
                "num_envs must be supplied when the policy environment does not "
                "expose a positive vector width"
            )
        return int(num_envs)

    def dagger_env_mask(
        self,
        num_envs: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the fixed seeded DAgger cohort for logging and collection."""
        num_envs = self._parallel_env_count(num_envs)
        if device is None:
            device = self.device
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        key = (num_envs, target_device)
        cache = getattr(self, "_fixed_dagger_env_mask_cache", None)
        if cache is None:
            cache = {}
            self._fixed_dagger_env_mask_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        mask = _seeded_dagger_env_mask(
            num_envs,
            float(self.cfg.dagger_env_fraction),
            int(self.cfg.dagger_seed),
            device=target_device,
        )
        if not bool(mask.any()) or bool(mask.all()):
            raise ValueError(
                "three-source collection requires at least one DAgger and one "
                "Student-only vector environment"
            )
        cache[key] = mask
        return mask

    def student_only_env_mask(
        self,
        num_envs: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the fixed complement used for pure Student metrics."""
        return ~self.dagger_env_mask(num_envs=num_envs, device=device)

    @staticmethod
    def _validate_td3_config(cfg) -> None:
        # Preserve every locked observation/action/replay/C51 check in the raw
        # replay base, then reject the deterministic algorithm knobs.  Present
        # eta_sac to the base's generic "at least one Actor objective" check so
        # a deliberate pure-SAC (lambda_bc=0) ablation remains expressible even
        # though the inherited eta_td3 field is correctly locked to zero.
        critic_type = _fastsac_q_critic_type(cfg)
        if critic_type not in Q_CRITIC_TYPES:
            raise ValueError(
                "q_critic_type must be 'c51', 'scalar', or 'distributional'"
            )
        q_twin_reduction = _fastsac_q_twin_reduction(cfg)
        if q_twin_reduction not in Q_TWIN_REDUCTIONS:
            raise ValueError("q_twin_reduction must be 'min' or 'mean'")
        actor_initialization = _student_actor_initialization(cfg)
        if actor_initialization not in STUDENT_ACTOR_INITIALIZATION_MODES:
            raise ValueError(
                "student_actor_initialization must be 'teacher_bc' or 'fresh'"
            )
        actor_adopt_path = getattr(cfg, "actor_adopt_checkpoint_path", None)
        if actor_adopt_path is not None:
            if not isinstance(actor_adopt_path, str) or not actor_adopt_path.strip():
                raise ValueError(
                    "actor_adopt_checkpoint_path must be null or a non-empty path"
                )
            if actor_initialization != TEACHER_BC_STUDENT_ACTOR_INITIALIZATION:
                raise ValueError(
                    "actor_adopt_checkpoint_path cannot be combined with "
                    "student_actor_initialization='fresh'; the explicit "
                    "actor_adapt overlay is the final Student Actor initialization"
                )
            if not bool(cfg.load_pretrained_perception):
                raise ValueError(
                    "actor_adopt_checkpoint_path requires "
                    "load_pretrained_perception=true because actor_adapt was "
                    "trained on a predicted privileged latent"
                )
            perception_path = getattr(cfg, "perception_checkpoint_path", None)
            if not isinstance(perception_path, str) or not perception_path.strip():
                raise ValueError(
                    "actor_adopt_checkpoint_path requires a non-empty "
                    "perception_checkpoint_path"
                )
        base_cfg = copy.copy(cfg)
        base_cfg.eta_td3 = float(cfg.eta_sac)
        DistributionalTD3TeacherBC._validate_td3_config(base_cfg)
        if not isinstance(cfg.teacher_prefill_use_ppo_noise, bool):
            raise ValueError("teacher_prefill_use_ppo_noise must be a boolean")
        dagger_env_fraction = getattr(cfg, "dagger_env_fraction", 0.5)
        if (
            isinstance(dagger_env_fraction, bool)
            or not isinstance(dagger_env_fraction, (int, float))
            or not math.isfinite(float(dagger_env_fraction))
            or not 0.0 < float(dagger_env_fraction) < 1.0
        ):
            raise ValueError(
                "dagger_env_fraction must be a finite number strictly between 0 and 1"
            )
        perception_mode = str(cfg.perception_replay_mode)
        if perception_mode not in (
            ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
            "four_way",
        ):
            raise ValueError(
                "FastSAC perception_replay_mode must be "
                "'online_student_rollout' or 'four_way'"
            )
        actor_observation_mode = str(
            getattr(
                cfg,
                "sac_actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        if actor_observation_mode not in SAC_ACTOR_OBSERVATION_MODES:
            raise ValueError(
                "sac_actor_observation_mode must be 'student_perception' or "
                "'privileged_oracle'"
            )
        if (
            actor_observation_mode == PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE
            and perception_mode != ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
        ):
            raise ValueError(
                "privileged_oracle Actor diagnostics require "
                "perception_replay_mode='online_student_rollout'"
            )
        if (
            perception_mode == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
            and float(cfg.teacher_perception_replay_fraction) != 0.0
        ):
            raise ValueError(
                "FastSAC online Student perception requires "
                "teacher_perception_replay_fraction=0"
            )
        if (
            str(cfg.dagger_control_mode) != "beta"
            or float(cfg.dagger_beta_start) != 0.0
            or float(cfg.dagger_beta_end) != 0.0
        ):
            raise ValueError(
                "FastSAC live Student perception requires beta control with "
                "dagger_beta_start=dagger_beta_end=0"
            )
        exact_zero = (
            "eta_td3",
            "target_policy_noise_std",
            "target_policy_noise_clip",
            "collector_exploration_noise_std",
            "collector_exploration_noise_clip",
        )
        for name in exact_zero:
            if float(getattr(cfg, name)) != 0.0:
                raise ValueError(f"FastSAC requires inherited TD3 field {name}=0")
        action_distribution = _fastsac_action_distribution(cfg)
        if action_distribution not in (
            NORMALIZED_TANH_ACTION_DISTRIBUTION,
            PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        ):
            raise ValueError(
                "sac_action_distribution must be 'normalized_tanh' or "
                "'ppo_physical_gaussian'"
            )
        if not isinstance(cfg.sac_use_autotune, bool):
            raise ValueError("sac_use_autotune must be a boolean")
        if action_distribution == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION:
            load_noise_scale = getattr(cfg, "load_noise_scale", None)
            if (
                isinstance(load_noise_scale, bool)
                or not isinstance(load_noise_scale, (int, float))
                or not math.isfinite(float(load_noise_scale))
                or float(load_noise_scale) <= 0.0
            ):
                raise ValueError(
                    "ppo_physical_gaussian requires a finite positive "
                    "load_noise_scale, matching PPOVEL finetune initialization"
                )
            physical_std_controls = {
                name: getattr(cfg, name, None)
                for name in (
                    "sac_physical_std_lr",
                    "sac_physical_std_max_kl",
                    "sac_physical_std_min",
                    "sac_physical_std_max",
                )
            }
            for name, value in physical_std_controls.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            std_min = float(physical_std_controls["sac_physical_std_min"])
            std_max = float(physical_std_controls["sac_physical_std_max"])
            if not std_min < std_max:
                raise ValueError(
                    "sac_physical_std_min must be smaller than "
                    "sac_physical_std_max"
                )
            if not std_min <= float(load_noise_scale) <= std_max:
                raise ValueError(
                    "load_noise_scale must lie inside the physical std bounds"
                )
            bound_mode = str(
                getattr(
                    cfg,
                    "sac_physical_std_bound_mode",
                    UNIFORM_PHYSICAL_STD_BOUND_MODE,
                )
            )
            if bound_mode not in PHYSICAL_STD_BOUND_MODES:
                raise ValueError(
                    "sac_physical_std_bound_mode must be 'uniform_physical' "
                    "or 'q_normalized'"
                )
            normalized_std_controls = {
                name: getattr(cfg, name, default)
                for name, default in (
                    ("sac_physical_std_normalized_min", 0.02),
                    ("sac_physical_std_normalized_max", 0.11),
                )
            }
            for name, value in normalized_std_controls.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            if not (
                float(normalized_std_controls["sac_physical_std_normalized_min"])
                < float(normalized_std_controls["sac_physical_std_normalized_max"])
            ):
                raise ValueError(
                    "sac_physical_std_normalized_min must be smaller than "
                    "sac_physical_std_normalized_max"
                )
        elif str(
            getattr(
                cfg,
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        ) != UNIFORM_PHYSICAL_STD_BOUND_MODE:
            raise ValueError(
                "q-normalized physical std bounds require "
                "sac_action_distribution='ppo_physical_gaussian'"
            )
        if str(cfg.sac_alpha_update_cadence) not in ("actor", "critic"):
            raise ValueError("sac_alpha_update_cadence must be 'actor' or 'critic'")
        if action_distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION:
            initial_action_std = float(cfg.sac_initial_action_std)
            if not math.isfinite(initial_action_std) or initial_action_std <= 0.0:
                raise ValueError("sac_initial_action_std must be finite and positive")
            _validate_fastsac_entropy_target_controls(
                cfg.sac_log_std_min,
                cfg.sac_log_std_max,
                cfg.sac_target_entropy_ratio,
            )
        else:
            target_ratio = cfg.sac_target_entropy_ratio
            if (
                isinstance(target_ratio, bool)
                or not isinstance(target_ratio, (int, float))
                or not math.isfinite(float(target_ratio))
                or float(target_ratio) <= 0.0
            ):
                raise ValueError(
                    "sac_target_entropy_ratio must be finite and positive"
                )
        for name in (
            "sac_actor_lr",
            "sac_alpha_init",
            "sac_alpha_lr",
            "sac_tau",
            "sac_max_grad_norm",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("eta_sac", "lambda_bc"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if getattr(cfg, "use_q_filtered_bc", False) not in (True, False):
            raise ValueError("use_q_filtered_bc must be boolean")
        if not isinstance(
            getattr(cfg, "q_condition_on_actuator_state", False), bool
        ):
            raise ValueError("q_condition_on_actuator_state must be boolean")
        if not isinstance(getattr(cfg, "q_use_predicted_effect", False), bool):
            raise ValueError("q_use_predicted_effect must be boolean")
        if bool(getattr(cfg, "q_use_predicted_effect", False)) and not bool(
            getattr(cfg, "q_condition_on_actuator_state", False)
        ):
            raise ValueError(
                "q_use_predicted_effect requires q_condition_on_actuator_state=true"
            )
        if not isinstance(getattr(cfg, "q_use_residual_film", False), bool):
            raise ValueError("q_use_residual_film must be boolean")
        if bool(getattr(cfg, "q_use_residual_film", False)):
            if not bool(getattr(cfg, "q_condition_on_actuator_state", False)):
                raise ValueError(
                    "q_use_residual_film requires "
                    "q_condition_on_actuator_state=true"
                )
            if str(cfg.q_action_fusion) not in ("late", "balanced"):
                raise ValueError(
                    "q_use_residual_film requires a late or balanced action stem"
                )
        residual_film_scale = getattr(cfg, "q_residual_film_scale", 0.1)
        if (
            isinstance(residual_film_scale, bool)
            or not isinstance(residual_film_scale, (int, float))
            or not math.isfinite(float(residual_film_scale))
            or not 0.0 < float(residual_film_scale) <= 1.0
        ):
            raise ValueError("q_residual_film_scale must be finite and in (0, 1]")
        actor_weight_decay = getattr(cfg, "sac_actor_weight_decay", 0.0)
        if (
            isinstance(actor_weight_decay, bool)
            or not isinstance(actor_weight_decay, (int, float))
            or not math.isfinite(float(actor_weight_decay))
            or float(actor_weight_decay) < 0.0
        ):
            raise ValueError(
                "sac_actor_weight_decay must be finite and non-negative"
            )
        if float(cfg.eta_sac) == 0.0 and float(cfg.lambda_bc) == 0.0:
            raise ValueError("eta_sac and lambda_bc cannot both be zero")
        for name in ("sac_policy_frequency", "sac_learning_starts"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < float(cfg.sac_tau) <= 1.0:
            raise ValueError("sac_tau must be in (0,1]")
        aliases = (
            ("dagger_bc_lr", "sac_actor_lr"),
            ("policy_delay", "sac_policy_frequency"),
            ("td3_learning_starts", "sac_learning_starts"),
            ("q_tau", "sac_tau"),
            ("q_max_grad_norm", "sac_max_grad_norm"),
        )
        for legacy, sac_name in aliases:
            left = float(getattr(cfg, legacy))
            right = float(getattr(cfg, sac_name))
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{legacy} must equal {sac_name} in this backend")

        # Each independently sampled online ring must be able to retain its
        # complete cohort share at the common learning-start boundary. Without
        # this check a too-small ring wraps before ``replay_ready`` can ever be
        # true. Keep this invariant in the backend as well as the Hydra
        # entrypoint so direct/programmatic construction is equally safe.
        learning_starts = int(cfg.sac_learning_starts)
        dagger_learning_starts, student_learning_starts = _split_count(
            learning_starts, float(dagger_env_fraction)
        )
        canonical_mix = all(
            hasattr(cfg, f"{purpose}_{source}_fraction")
            for purpose in ("q", "actor", "perception")
            for source in REPLAY_SOURCE_ORDER
        )

        def _online_fraction(purpose: str) -> float:
            if canonical_mix:
                return float(
                    getattr(cfg, f"{purpose}_uniform_student_fraction")
                ) + float(
                    getattr(cfg, f"{purpose}_failure_student_fraction")
                )
            teacher_fraction_field = {
                "q": "q_teacher_replay_ratio",
                "actor": "teacher_actor_replay_fraction",
            }[purpose]
            return 1.0 - float(getattr(cfg, teacher_fraction_field))

        q_online = _online_fraction("q") > 0.0
        actor_online = _online_fraction("actor") > 0.0
        q_dagger_fraction = float(cfg.q_online_dagger_replay_fraction)
        actor_dagger_fraction = float(cfg.actor_online_dagger_replay_fraction)
        dagger_ring_required = (
            q_online and q_dagger_fraction > 0.0
        ) or (actor_online and actor_dagger_fraction > 0.0)
        student_ring_required = (
            q_online and q_dagger_fraction < 1.0
        ) or (actor_online and actor_dagger_fraction < 1.0)
        if (
            dagger_ring_required
            and int(cfg.dagger_buffer_capacity) < dagger_learning_starts
        ):
            raise ValueError(
                "dagger_buffer_capacity must cover the DAgger cohort's "
                "learning-start rows"
            )
        if (
            student_ring_required
            and int(cfg.student_buffer_capacity) < student_learning_starts
        ):
            raise ValueError(
                "student_buffer_capacity must cover the pure-Student cohort's "
                "learning-start rows"
            )
        q_update_to_data_ratio = getattr(cfg, "q_update_to_data_ratio", None)
        if (
            q_update_to_data_ratio is None
            or isinstance(q_update_to_data_ratio, bool)
            or not math.isfinite(float(q_update_to_data_ratio))
            or float(q_update_to_data_ratio) <= 0.0
        ):
            raise ValueError(
                "q_update_to_data_ratio must be finite and positive"
            )

    def _uses_ppo_physical_gaussian(self) -> bool:
        return (
            _fastsac_action_distribution(self.cfg)
            == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
        )

    def _uses_min_q_twin_reduction(self) -> bool:
        return _fastsac_q_twin_reduction(self.cfg) == Q_TWIN_REDUCTION_MIN

    def _reduce_twin_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        return _reduce_actor_q_values(
            q_values, self._uses_min_q_twin_reduction()
        )

    def _actor_learning_semantics(self) -> str:
        if not self._uses_ppo_physical_gaussian():
            semantics = ACTOR_LEARNING_SEMANTICS
        elif (
            self._physical_std_bound_mode()
            == Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE
        ):
            semantics = Q_NORMALIZED_PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS
        else:
            semantics = PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS
        if not self._uses_min_q_twin_reduction():
            if "twin_min" not in semantics:
                raise RuntimeError(
                    "FastSAC Actor semantics lack the twin-min reduction marker"
                )
            semantics = semantics.replace("twin_min", "twin_mean", 1)
        if getattr(self.cfg, "use_q_filtered_bc", False):
            return f"{semantics}_with_{SPRED_P_BC_SEMANTICS}"
        return semantics

    def _critic_learning_semantics(self) -> str:
        scalar = _fastsac_q_critic_type(self.cfg) == SCALAR_Q_CRITIC_TYPE
        semantics = SCALAR_CRITIC_SEMANTICS if scalar else CRITIC_SEMANTICS
        if self._uses_min_q_twin_reduction():
            return semantics
        if scalar:
            return semantics.replace("clipped_twin", "mean_twin", 1)
        return semantics.replace("lower_expected_complete", "mean_complete", 1)

    def _entropy_semantics(self) -> str:
        if self._uses_ppo_physical_gaussian():
            return PPO_PHYSICAL_GAUSSIAN_ENTROPY_SEMANTICS
        return (
            ENTROPY_SEMANTICS
            if str(self.cfg.sac_alpha_update_cadence) == "actor"
            else _CRITIC_CADENCE_ENTROPY_SEMANTICS
        )

    def _ppo_actor_core(self) -> Actor:
        cores = [
            module for module in self.actor_adapt.modules() if isinstance(module, Actor)
        ]
        if len(cores) != 1:
            raise RuntimeError(
                "FastSAC Teacher-BC requires exactly one legacy Actor core; "
                f"found {len(cores)}"
            )
        core = cores[0]
        if bool(core.predict_std):
            raise RuntimeError("FastSAC mean transfer requires a separate legacy std")
        return core

    def _ppo_actor_std_parameter(self) -> nn.Parameter:
        return self._ppo_actor_core().actor_std

    def _freeze_legacy_actor_std(self) -> None:
        core = self._ppo_actor_core()
        core.actor_std.requires_grad_(False)
        core.actor_std.grad = None

    def _configure_training_actor_std(self) -> None:
        """Select PPOVEL std ownership for the configured policy distribution."""
        if self._uses_ppo_physical_gaussian():
            core = self._ppo_actor_core()
            core.actor_std.requires_grad_(True)
            core.actor_std.grad = None
            self.bc_dagger_sac_adapter.requires_grad_(False)
            for parameter in self.bc_dagger_sac_adapter.parameters():
                parameter.grad = None
        else:
            self._freeze_legacy_actor_std()
            self.bc_dagger_sac_adapter.requires_grad_(True)

    @staticmethod
    def _actor_std_from_module_state(
        module_state: Mapping, *, context: str
    ) -> torch.Tensor:
        candidates = [
            value
            for key, value in module_state.items()
            if str(key).endswith("actor_std") and torch.is_tensor(value)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{context} must contain exactly one PPOVEL actor_std tensor"
            )
        actor_std = candidates[0].detach().reshape(-1)
        if (
            actor_std.numel() == 0
            or not torch.is_floating_point(actor_std)
            or not torch.isfinite(actor_std).all()
        ):
            raise ValueError(f"{context} contains an invalid PPOVEL actor_std")
        return actor_std

    def _restore_checkpoint_physical_actor_std(
        self, module_state: Mapping, *, context: str
    ) -> None:
        """Undo PPOVEL's fresh-finetune 0.5 reset for FastSAC resume only."""
        if not self._uses_ppo_physical_gaussian():
            return
        actor_std = self._actor_std_from_module_state(module_state, context=context)
        parameter = self._ppo_actor_std_parameter()
        if actor_std.shape != parameter.shape:
            raise ValueError(
                f"{context} actor_std shape {tuple(actor_std.shape)} does not match "
                f"runtime action shape {tuple(parameter.shape)}"
            )
        parameter.data.copy_(actor_std.to(parameter))

    @staticmethod
    def _inverse_smooth_log_std(
        log_std: torch.Tensor, log_std_min: float, log_std_max: float
    ) -> torch.Tensor:
        """Inverse-map an effective log std to its unconstrained coordinate."""
        if not math.isfinite(float(log_std_min)) or not math.isfinite(
            float(log_std_max)
        ):
            raise ValueError("FastSAC log-std bounds must be finite")
        if not float(log_std_min) < float(log_std_max):
            raise ValueError("FastSAC log-std bounds must be ordered")
        if (
            not torch.isfinite(log_std).all()
            or ((log_std <= float(log_std_min)) | (log_std >= float(log_std_max))).any()
        ):
            raise ValueError(
                "FastSAC effective log std must be finite and strictly inside "
                "its bounds"
            )
        normalized = (
            2.0
            * (log_std - float(log_std_min))
            / (float(log_std_max) - float(log_std_min))
            - 1.0
        )
        return torch.atanh(normalized)

    def _bounded_log_std(self, raw_log_std: torch.Tensor | None = None) -> torch.Tensor:
        """Smoothly map the adapter parameter into configured log-std bounds."""
        if raw_log_std is None:
            raw_log_std = self.bc_dagger_sac_adapter.log_std
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )

    def _physical_std_bound_mode(self) -> str:
        return str(
            getattr(
                self.cfg,
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        )

    def _physical_std_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint-wise raw std bounds for the selected coordinate mode."""
        cached = getattr(self, "_fastsac_physical_std_bounds_cache", None)
        if cached is not None:
            return cached

        q_scale = getattr(self, "_fastsac_q_action_scale", None)
        if not torch.is_tensor(q_scale):
            q_scale = self._ppo_actor_std_parameter().detach()
        q_scale = q_scale.detach()
        if not torch.isfinite(q_scale).all() or not bool((q_scale > 0.0).all()):
            raise ValueError("physical FastSAC q-action scale is invalid")
        absolute_min = float(getattr(self.cfg, "sac_physical_std_min", 0.05))
        absolute_max = float(getattr(self.cfg, "sac_physical_std_max", 0.5))
        lower = q_scale.new_full(q_scale.shape, absolute_min)
        upper = q_scale.new_full(q_scale.shape, absolute_max)
        mode = self._physical_std_bound_mode()
        if mode == Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE:
            normalized_min = float(
                getattr(self.cfg, "sac_physical_std_normalized_min", 0.02)
            )
            normalized_max = float(
                getattr(self.cfg, "sac_physical_std_normalized_max", 0.11)
            )
            lower = torch.maximum(lower, q_scale * normalized_min)
            upper = torch.minimum(upper, q_scale * normalized_max)
        elif mode != UNIFORM_PHYSICAL_STD_BOUND_MODE:
            raise ValueError(f"unsupported physical std bound mode {mode!r}")

        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("physical FastSAC std bounds contain NaN/Inf")
        invalid = lower >= upper
        if bool(invalid.any()):
            indices = invalid.nonzero(as_tuple=False).reshape(-1).cpu().tolist()
            raise ValueError(
                "physical FastSAC std bounds are empty for joint indices "
                f"{indices}; widen the absolute or q-normalized interval"
            )
        self._fastsac_physical_std_bounds_cache = (
            lower.detach(),
            upper.detach(),
        )
        return self._fastsac_physical_std_bounds_cache

    def _clamp_physical_actor_std(self, std: torch.Tensor) -> torch.Tensor:
        lower, upper = self._physical_std_bounds()
        if std.shape != lower.shape:
            raise ValueError(
                "physical FastSAC q-action scale and actor_std shapes differ"
            )
        return torch.maximum(torch.minimum(std, upper.to(std)), lower.to(std))

    def _bounded_physical_actor_std(self, *, detach: bool = False) -> torch.Tensor:
        """Read PPOVEL's direct std through the FastSAC safety envelope."""
        std = self._clamp_physical_actor_std(self._ppo_actor_std_parameter())
        return std.detach() if detach else std

    @torch.no_grad()
    def _project_physical_actor_std_(self) -> torch.Tensor:
        """Project the direct PPOVEL parameter and report the changed fraction."""
        parameter = self._ppo_actor_std_parameter()
        if not torch.isfinite(parameter).all():
            raise RuntimeError("FastSAC physical actor_std contains NaN/Inf")
        before = parameter.detach().clone()
        parameter.copy_(self._clamp_physical_actor_std(parameter))
        return (parameter != before).float().mean()

    def _physical_normalized_entropy_bounds(self) -> tuple[float, float]:
        """Return total Normal entropy bounds in nominal joint coordinates."""
        cached = getattr(self, "_fastsac_physical_entropy_bounds", None)
        if cached is not None:
            return cached
        q_scale = self._fastsac_q_action_scale.detach().float()
        if not torch.isfinite(q_scale).all() or not bool((q_scale > 0.0).all()):
            raise ValueError("physical FastSAC entropy reference scale is invalid")
        std_min, std_max = self._physical_std_bounds()
        offset = 0.5 * math.log(2.0 * math.pi * math.e)
        minimum = (q_scale.new_tensor(offset) + std_min.log() - q_scale.log()).sum()
        maximum = (q_scale.new_tensor(offset) + std_max.log() - q_scale.log()).sum()
        bounds = (float(minimum.item()), float(maximum.item()))
        self._fastsac_physical_entropy_bounds = bounds
        return bounds

    def _validate_physical_entropy_target_reachable(self) -> None:
        """Fail if autotune would continuously press against a std boundary."""
        minimum, maximum = self._physical_normalized_entropy_bounds()
        target = float(self.target_entropy)
        if not minimum < target < maximum:
            raise ValueError(
                "physical FastSAC target entropy is unreachable within the std "
                f"bounds in nominal joint coordinates: require {minimum} < "
                f"{target} < {maximum}"
            )

    @staticmethod
    def _physical_scale_kl(
        reference_std: torch.Tensor, candidate_std: torch.Tensor
    ) -> torch.Tensor:
        """KL(N(0, reference)||N(0, candidate)) summed over joints."""
        return (
            (
                torch.log(candidate_std / reference_std)
                + reference_std.square() / (2.0 * candidate_std.square())
                - 0.5
            )
            .sum()
            .clamp_min(0.0)
        )

    @torch.no_grad()
    def _cap_physical_actor_std_rollout_kl_(
        self, fallback_reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cap cumulative std-only KL from the rollout's behavior scale."""
        parameter = self._ppo_actor_std_parameter()
        reference = getattr(
            self, "_physical_std_rollout_reference", fallback_reference
        ).to(parameter)
        if reference.shape != parameter.shape or not torch.isfinite(reference).all():
            raise RuntimeError("FastSAC physical std rollout reference is invalid")
        reference = self._clamp_physical_actor_std(reference)
        candidate = self._clamp_physical_actor_std(parameter.detach())
        maximum_kl = float(self.cfg.sac_physical_std_max_kl)
        candidate_kl = self._physical_scale_kl(reference, candidate)
        if float(candidate_kl.item()) <= maximum_kl:
            return candidate_kl, candidate_kl.new_zeros(())

        log_reference = reference.log()
        log_candidate = candidate.log()
        low = 0.0
        high = 1.0
        for _ in range(32):
            fraction = 0.5 * (low + high)
            proposal = torch.exp(
                log_reference + fraction * (log_candidate - log_reference)
            )
            proposal_kl = self._physical_scale_kl(reference, proposal)
            if float(proposal_kl.item()) <= maximum_kl:
                low = fraction
            else:
                high = fraction
        capped = torch.exp(log_reference + low * (log_candidate - log_reference))
        parameter.copy_(self._clamp_physical_actor_std(capped))
        capped_kl = self._physical_scale_kl(reference, parameter.detach())
        return capped_kl, capped_kl.new_ones(())

    def _physical_actor_std_update(
        self,
        before: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the SAC-populated direct-std gradient and hard bounds."""
        if not self._uses_ppo_physical_gaussian():
            raise RuntimeError("physical std update requested for normalized_tanh")
        optimizer = getattr(self, "actor_std_optimizer", None)
        if optimizer is None:
            raise RuntimeError("physical Gaussian Actor has no std optimizer")
        parameter = self._ppo_actor_std_parameter()
        if parameter.grad is None:
            raise RuntimeError(
                "SAC Actor objective did not produce a physical actor_std gradient"
            )
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("FastSAC physical actor_std gradient contains NaN/Inf")
        std_grad = torch.nn.utils.clip_grad_norm_(
            (parameter,), float(self.cfg.sac_max_grad_norm)
        )
        if not torch.isfinite(torch.as_tensor(std_grad)):
            raise RuntimeError("FastSAC physical actor_std gradient norm is NaN/Inf")
        optimizer.step()
        post_projection = self._project_physical_actor_std_()
        after = parameter.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        self.actor_std_update_count = (
            int(getattr(self, "actor_std_update_count", 0)) + 1
        )
        std_min, std_max = self._physical_std_bounds()
        return {
            "actor_std_sac_grad_norm": torch.as_tensor(std_grad).detach(),
            "actor_std_step_abs_mean": (after - before).abs().mean(),
            "actor_std_projection_fraction": post_projection.detach(),
            "actor_std_at_min_fraction": (after <= std_min).float().mean(),
            "actor_std_at_max_fraction": (after >= std_max).float().mean(),
            "actor_std_lr": after.new_tensor(
                float(self.cfg.sac_physical_std_lr)
            ),
        }

    def _policy_log_std(self) -> torch.Tensor:
        """Return the effective std coordinate used by the current policy."""
        if self._uses_ppo_physical_gaussian():
            return self._bounded_physical_actor_std().log()
        return self._bounded_log_std()

    def _policy_raw_log_std(self) -> torch.Tensor:
        if self._uses_ppo_physical_gaussian():
            # PPOVEL parameterizes std directly; there is no latent/raw map.
            return self._policy_log_std()
        return self.bc_dagger_sac_adapter.log_std

    def _legacy_effective_log_std_to_raw(
        self,
        legacy_log_std: torch.Tensor,
        legacy_log_std_min: float,
        legacy_log_std_max: float,
    ) -> torch.Tensor:
        """Convert a v3 hard-clamped effective log std for v4 inference.

        V3 allowed the learned tensor to leave its configured interval and
        hard-clamped it only while building a distribution. Map that effective
        value just inside the open v4 interval so inference retains the v3
        policy scale to floating-point precision without an infinite atanh.
        """
        if not torch.is_tensor(legacy_log_std) or not torch.is_floating_point(
            legacy_log_std
        ):
            raise ValueError("FastSAC v3 checkpoint has invalid adapter log_std")
        if not torch.isfinite(legacy_log_std).all():
            raise ValueError("FastSAC v3 checkpoint has non-finite adapter log_std")
        try:
            legacy_log_std_min = float(legacy_log_std_min)
            legacy_log_std_max = float(legacy_log_std_max)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "FastSAC v3 checkpoint has invalid log-std bounds"
            ) from error
        if not (
            math.isfinite(legacy_log_std_min)
            and math.isfinite(legacy_log_std_max)
            and legacy_log_std_min < legacy_log_std_max
        ):
            raise ValueError("FastSAC v3 checkpoint has invalid log-std bounds")
        # Reproduce the v3 distribution before changing coordinate systems.
        effective = legacy_log_std.clamp(legacy_log_std_min, legacy_log_std_max)
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        normalized = 2.0 * (effective - log_std_min) / (log_std_max - log_std_min) - 1.0
        inward = 4.0 * torch.finfo(normalized.dtype).eps
        return torch.atanh(normalized.clamp(-1.0 + inward, 1.0 - inward))

    def _student_action_support(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return nominal Student bounds independently of the Teacher guard."""
        q_center = self._fastsac_q_action_center.to(reference)
        q_scale = self._fastsac_q_action_scale.to(reference)
        low = getattr(
            self, "_fastsac_student_action_low", q_center - q_scale
        ).to(reference)
        high = getattr(
            self, "_fastsac_student_action_high", q_center + q_scale
        ).to(reference)
        center = getattr(
            self, "_fastsac_student_action_center", (low + high) * 0.5
        ).to(reference)
        scale = getattr(
            self, "_fastsac_student_action_scale", (high - low) * 0.5
        ).to(reference)
        return low, high, center, scale

    def _project_student_policy_action(self, action: torch.Tensor) -> torch.Tensor:
        """Put a Teacher label on the configured Student's executable support."""
        if self._uses_ppo_physical_gaussian():
            # Teacher collection already records an execution-valid physical
            # command. Keep that exact coordinate for physical-Gaussian BC and
            # Q filtering; the shared execution projection is only a defensive
            # finite-support guard for synthetic or legacy batches.
            bounded = self._project_execution_action(action)
            # Do not let the execution sanitizer hide a corrupt valid replay
            # label; the BC/Q paths retain their existing fail-closed checks.
            return torch.where(torch.isfinite(action), bounded, action)
        low, high, _, _ = self._student_action_support(action)
        return torch.maximum(torch.minimum(action, high), low)

    def _normalized_tanh_dist_from_mean(
        self,
        mean: torch.Tensor,
        *,
        normalized_std: float | None = None,
    ) -> FastSACTanhNormal:
        """Calibrate a PPO physical mean into a nominal bounded SAC policy.

        PPOVEL's final head emits an unbounded physical joint command.  A fixed
        affine calibration converts that proposal into the pre-tanh latent. At
        the neutral PPO command ``mean=0``, the transformed deterministic
        action is exactly zero and its physical Jacobian with respect to the
        PPO output is exactly one.  Unlike a clipped inverse tanh, the affine
        map remains differentiable for every finite Actor output.
        """
        if not torch.isfinite(mean).all():
            raise RuntimeError("FastSAC Actor proposal contains non-finite actions")
        low, high, center, scale = self._student_action_support(mean)
        normalized_zero = -center / scale
        if not torch.all(normalized_zero.abs() < 1.0):
            raise RuntimeError("Student SAC support does not contain action zero")
        latent_zero = torch.atanh(normalized_zero)
        inverse_slope = (
            scale * (1.0 - normalized_zero.square())
        ).reciprocal()
        latent_loc = latent_zero + inverse_slope * mean
        if normalized_std is None:
            latent_std = self._bounded_log_std().exp().to(mean)
        else:
            value = float(normalized_std)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("normalized SAC action std must be finite and positive")
            latent_std = mean.new_full((mean.shape[-1],), value)
        latent_scale = latent_std.expand_as(mean)
        return FastSACTanhNormal(
            latent_loc,
            latent_scale,
            low=low,
            high=high,
            event_dims=1,
        )

    def _sac_dist_from_mean(
        self, mean: torch.Tensor
    ) -> FastSACTanhNormal | FastSACPhysicalNormal:
        """Build the configured stochastic policy from a PPO physical mean."""
        if not torch.isfinite(mean).all():
            raise RuntimeError("FastSAC Actor proposal contains non-finite raw actions")
        if self._uses_ppo_physical_gaussian():
            # This is the same direct per-joint scale parameter and unbounded
            # physical Normal used by PPOVEL.  The shared execution projection
            # remains only as a far-tail finite-safety guard.
            action_std = self._bounded_physical_actor_std()
            return FastSACPhysicalNormal(mean, action_std.expand_as(mean))
        return self._normalized_tanh_dist_from_mean(mean)

    def _actor_mean_from_flat(self, actor_obs: torch.Tensor) -> torch.Tensor:
        # Lightweight unit policies may expose the mean module directly.  The
        # production VAIC actor instead has the TensorDict ``get_dist`` API.
        if not hasattr(self, "observation_spec"):
            return self.actor_adapt(actor_obs)
        legacy_dist = DistributionalTD3TeacherBC._actor_dist_from_flat(self, actor_obs)
        return legacy_dist.mean

    def _actor_dist_from_flat(
        self, actor_obs: torch.Tensor
    ) -> FastSACTanhNormal | FastSACPhysicalNormal:
        return self._sac_dist_from_mean(self._actor_mean_from_flat(actor_obs))

    def _normalized_action_log_prob(self, raw_log_prob: torch.Tensor):
        """Convert raw-action density to joint-normalized-coordinate density."""
        # The additive Jacobian is constant with respect to both mean and std,
        # so physical SAC keeps PPOVEL's exact raw Normal Actor gradient while
        # alpha autotune sees the same dimensionless nominal-joint coordinates
        # as Q and the configured target entropy.
        return raw_log_prob + float(self._fastsac_entropy_reference_log_scale_sum)

    @torch.no_grad()
    def _student_mean_action(self, td: TensorDict) -> torch.Tensor:
        """Use the distribution-consistent deterministic Student action."""
        raw_mean = self._student_raw_action_proposal(td)
        if not torch.isfinite(raw_mean).all():
            raise RuntimeError("FastSAC evaluation Actor produced non-finite actions")
        if self._uses_ppo_physical_gaussian():
            return self._project_execution_action(raw_mean)
        return self._normalized_tanh_dist_from_mean(raw_mean).mean

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return _DistributionalFastSACDaggerRolloutPolicy(self)
        return _DeterministicFastSACStudentEvalPolicy(self)

    def _q_output_values(self, qnet: nn.Module, outputs: torch.Tensor):
        """Convert either configured critic representation to twin scalars."""
        if _fastsac_q_critic_type(self.cfg) == SCALAR_Q_CRITIC_TYPE:
            if outputs.ndim != 2 or outputs.shape[0] != 2:
                raise ValueError("scalar twin-Q outputs must have shape [2, batch]")
            return outputs
        return (F.softmax(outputs, dim=-1) * qnet.support).sum(dim=-1)

    @torch.no_grad()
    def _distributional_fastsac_target(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        next_dist = self._actor_dist_from_flat(batch["next_observations"])
        next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        next_log_prob = self._normalized_action_log_prob(next_raw_log_prob)
        bootstrap = (batch["truncations"].bool() | ~batch["dones"].bool()).float()
        effective_discount, _, effective_n_steps = _q_target_discount_factors(
            self.cfg, batch
        )
        alpha = self.log_alpha.exp()
        entropy_tax = effective_discount * bootstrap * alpha * next_log_prob
        soft_reward = batch["rewards"] - entropy_tax

        target_logits = self._q_forward(
            self.qnet_target,
            batch["next_critic_observations"],
            next_action,
            batch.get(NEXT_Q_ACTUATOR_CONTEXT_KEY),
        )
        if _fastsac_q_critic_type(self.cfg) == SCALAR_Q_CRITIC_TYPE:
            if target_logits.ndim != 2 or target_logits.shape[0] != 2:
                raise RuntimeError(
                    "FastSAC scalar target must have shape [2, batch]"
                )
            (
                selected_next_q,
                q1_target_fraction,
                q2_target_fraction,
            ) = _reduce_fastsac_twin_target(
                target_logits,
                target_logits,
                _fastsac_q_twin_reduction(self.cfg),
            )
            scalar_target = soft_reward + (
                effective_discount * bootstrap * selected_next_q
            )
            zero = scalar_target.new_zeros(())
            reward_abs = batch["rewards"].abs().mean()
            metrics = {
                "target_expected_q1_mean": target_logits[0].mean(),
                "target_expected_q2_mean": target_logits[1].mean(),
                "projected_target_mean": scalar_target.mean(),
                "reduced_target_expected_mean": scalar_target.mean(),
                "target_q1_contribution_fraction": q1_target_fraction,
                "target_q2_contribution_fraction": q2_target_fraction,
                # Historical dashboard aliases. In mean mode these are equal
                # contribution weights; no individual head is selected.
                "selected_target_expected_mean": selected_next_q.mean(),
                "target_distribution_entropy": zero,
                "target_select_q1_fraction": q1_target_fraction,
                "target_select_q2_fraction": q2_target_fraction,
                "q1_left_support_clip_fraction": zero,
                "q1_right_support_clip_fraction": zero,
                "q2_left_support_clip_fraction": zero,
                "q2_right_support_clip_fraction": zero,
                "support_clip_fraction_mean": zero,
                "support_clip_fraction_max": zero,
                "left_support_projection_clipping_fraction": zero,
                "right_support_projection_clipping_fraction": zero,
                "target_smoothing_noise_norm": zero,
                "target_noise_free_action_abs_mean": next_dist.mean.abs().mean(),
                "target_sample_action_abs_mean": next_action.abs().mean(),
                "target_log_prob_mean": next_log_prob.mean(),
                "entropy_tax_mean": entropy_tax.mean(),
                "entropy_tax_abs_mean": entropy_tax.abs().mean(),
                "entropy_tax_reward_abs_ratio": entropy_tax.abs().mean()
                / reward_abs.clamp_min(torch.finfo(reward_abs.dtype).eps),
                "effective_n_steps_mean": effective_n_steps.float().mean(),
                "alpha": alpha.detach(),
            }
            return scalar_target.detach(), metrics, next_log_prob.detach()
        target_probabilities = F.softmax(target_logits, dim=-1)
        raw_expected_heads = (target_probabilities * self.qnet_target.support).sum(
            dim=-1
        )
        projected_heads = []
        support_clip_fractions = []
        for head_probability in target_probabilities:
            projected, left_fraction, right_fraction = _project_c51_probabilities(
                head_probability,
                soft_reward,
                bootstrap,
                effective_discount,
                self.qnet_target.support,
            )
            projected_heads.append(projected)
            support_clip_fractions.append((left_fraction, right_fraction))
        if len(support_clip_fractions) != 2:
            raise RuntimeError("FastSAC C51 target must contain exactly two Q heads")
        (
            (q1_left_support_clip_fraction, q1_right_support_clip_fraction),
            (q2_left_support_clip_fraction, q2_right_support_clip_fraction),
        ) = support_clip_fractions
        q1_support_clip_fraction = (
            q1_left_support_clip_fraction + q1_right_support_clip_fraction
        )
        q2_support_clip_fraction = (
            q2_left_support_clip_fraction + q2_right_support_clip_fraction
        )
        support_clip_fraction_mean = 0.5 * (
            q1_support_clip_fraction + q2_support_clip_fraction
        )
        support_clip_fraction_max = torch.maximum(
            q1_support_clip_fraction, q2_support_clip_fraction
        )
        left_support_projection_clipping_fraction = 0.5 * (
            q1_left_support_clip_fraction + q2_left_support_clip_fraction
        )
        right_support_projection_clipping_fraction = 0.5 * (
            q1_right_support_clip_fraction + q2_right_support_clip_fraction
        )
        projected_heads = torch.stack(projected_heads, dim=0)
        # Min mode chooses one complete distribution using its pre-projection
        # expectation. Mean mode instead forms an equal mixture of both
        # complete projected distributions. Projection is linear in the source
        # probabilities, so this equals projecting their pre-projection mix.
        (
            selected_target,
            q1_target_fraction,
            q2_target_fraction,
        ) = _reduce_fastsac_twin_target(
            projected_heads,
            raw_expected_heads,
            _fastsac_q_twin_reduction(self.cfg),
        )
        selected_expected = (
            selected_target * self.qnet_target.support
        ).sum(dim=-1)
        selected_entropy = -(
            selected_target
            * selected_target.clamp_min(torch.finfo(selected_target.dtype).tiny).log()
        ).sum(dim=-1)
        reward_abs = batch["rewards"].abs().mean()
        metrics = {
            "target_expected_q1_mean": raw_expected_heads[0].mean(),
            "target_expected_q2_mean": raw_expected_heads[1].mean(),
            "projected_target_mean": selected_expected.mean(),
            "reduced_target_expected_mean": selected_expected.mean(),
            "target_q1_contribution_fraction": q1_target_fraction,
            "target_q2_contribution_fraction": q2_target_fraction,
            # Historical dashboard aliases. In mean mode these are equal
            # contribution weights; no individual head is selected.
            "selected_target_expected_mean": selected_expected.mean(),
            "target_distribution_entropy": selected_entropy.mean(),
            "target_select_q1_fraction": q1_target_fraction,
            "target_select_q2_fraction": q2_target_fraction,
            "q1_left_support_clip_fraction": q1_left_support_clip_fraction,
            "q1_right_support_clip_fraction": q1_right_support_clip_fraction,
            "q2_left_support_clip_fraction": q2_left_support_clip_fraction,
            "q2_right_support_clip_fraction": q2_right_support_clip_fraction,
            "support_clip_fraction_mean": support_clip_fraction_mean,
            "support_clip_fraction_max": support_clip_fraction_max,
            # Keep the historical directional aliases for existing dashboards
            # and for the shared TD3 rollout aggregation surface. Unlike the
            # old loop-overwrite behavior, each alias now averages both heads.
            "left_support_projection_clipping_fraction": (
                left_support_projection_clipping_fraction
            ),
            "right_support_projection_clipping_fraction": (
                right_support_projection_clipping_fraction
            ),
            "target_smoothing_noise_norm": soft_reward.new_zeros(()),
            "target_noise_free_action_abs_mean": next_dist.mean.abs().mean(),
            "target_sample_action_abs_mean": next_action.abs().mean(),
            "target_log_prob_mean": next_log_prob.mean(),
            "entropy_tax_mean": entropy_tax.mean(),
            "entropy_tax_abs_mean": entropy_tax.abs().mean(),
            "entropy_tax_reward_abs_ratio": entropy_tax.abs().mean()
            / reward_abs.clamp_min(torch.finfo(reward_abs.dtype).eps),
            "effective_n_steps_mean": effective_n_steps.float().mean(),
            "alpha": alpha.detach(),
        }
        return selected_target.detach(), metrics, next_log_prob.detach()

    # Compatibility alias used by focused callers which name the soft target.
    _soft_c51_target = _distributional_fastsac_target

    def _alpha_update(self, log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = log_prob.new_zeros(())
        if log_prob.numel() == 0 or not bool(self.cfg.sac_use_autotune):
            return {
                "alpha_loss": zero,
                "alpha_grad_norm": zero,
                "alpha": self.log_alpha.exp().detach(),
            }
        if self.alpha_optimizer is None:
            raise RuntimeError("FastSAC alpha autotune has no optimizer")
        # Optimize the standard log-temperature surrogate. Its gradient is the
        # entropy error itself rather than alpha times that error, so a cautious
        # alpha initialization (1e-5 in the default config) can still recover.
        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + float(self.target_entropy))
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad = self.log_alpha.grad.detach().abs().clone()
        self.alpha_optimizer.step()
        self.alpha_update_count = int(getattr(self, "alpha_update_count", 0)) + 1
        self.sac_alpha_update_count = (
            int(getattr(self, "sac_alpha_update_count", 0)) + 1
        )
        return {
            "alpha_loss": alpha_loss.detach(),
            "alpha_grad_norm": alpha_grad,
            "alpha": self.log_alpha.exp().detach(),
        }

    def _critic_update(self, batch: dict[str, torch.Tensor]):
        """One configured twin-Q update and temperature-cadence update."""
        projected_target, target_metrics, target_log_prob = (
            self._distributional_fastsac_target(batch)
        )
        q_outputs = self._q_forward(
            self.qnet,
            batch["critic_observations"],
            batch["actions"],
            batch.get(Q_ACTUATOR_CONTEXT_KEY),
        )
        if _fastsac_q_critic_type(self.cfg) == SCALAR_Q_CRITIC_TYPE:
            if q_outputs.ndim != 2 or q_outputs.shape[0] != 2:
                raise RuntimeError(
                    "FastSAC scalar critic must output shape [2, batch]"
                )
            per_head = torch.stack(
                [F.mse_loss(head, projected_target) for head in q_outputs]
            )
            expected_heads = q_outputs.detach()
        else:
            log_probabilities = F.log_softmax(q_outputs, dim=-1)
            per_head = (
                -(projected_target.unsqueeze(0) * log_probabilities)
                .sum(dim=-1)
                .mean(dim=-1)
            )
            expected_heads = self._q_output_values(
                self.qnet, q_outputs.detach()
            )
        critic_loss = per_head.sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.qnet.parameters(), float(self.cfg.sac_max_grad_norm)
        )
        self.critic_optimizer.step()
        self.critic_update_count += 1

        # The sampled next action has no learning meaning after a true
        # environment terminal.  Time-limit truncations do retain their real
        # next state and therefore stay in the temperature population, exactly
        # matching the soft Bellman bootstrap truth table above.
        alpha_bootstrap = (
            batch["truncations"].bool() | ~batch["dones"].bool()
        ).reshape(-1)
        alpha_log_prob = target_log_prob.reshape(-1)
        if alpha_bootstrap.numel() != alpha_log_prob.numel():
            raise ValueError(
                "FastSAC alpha bootstrap mask and target log-probability "
                "population are misaligned"
            )
        # Baseline temperature and Actor see the same policy-update timescale.
        # Cadence is evaluated after incrementing ``critic_update_count``, so
        # an Actor-cadence update adjusts alpha immediately before the matching
        # Actor update in the inherited replay loop. TVKD checkpoints keep the
        # historical every-Critic cadence for resume compatibility.
        alpha_update_cadence = str(self.cfg.sac_alpha_update_cadence)
        critic_warmup_active = self._student_actor_warmup_active()
        alpha_update_due = not critic_warmup_active and (
            alpha_update_cadence == "critic"
            or self.critic_update_count % int(self.cfg.sac_policy_frequency) == 0
        )
        alpha_update_count_before = int(getattr(self, "alpha_update_count", 0))
        if alpha_update_due:
            alpha_metrics = self._alpha_update(alpha_log_prob[alpha_bootstrap])
        else:
            # Reuse the empty-population path to publish zero per-step loss and
            # gradient without touching the optimizer.
            alpha_metrics = self._alpha_update(alpha_log_prob[:0])
        alpha_update_performed = (
            int(getattr(self, "alpha_update_count", 0)) > alpha_update_count_before
        )
        alpha_metrics.update(
            {
                "alpha_update_due_fraction": alpha_log_prob.new_tensor(
                    float(alpha_update_due)
                ),
                "alpha_update_performed_fraction": alpha_log_prob.new_tensor(
                    float(alpha_update_performed)
                ),
            }
        )
        _polyak_update_(self.qnet_target, self.qnet, float(self.cfg.sac_tau))
        self.qnet_target.requires_grad_(False).eval()
        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_loss_1": per_head[0].detach(),
            "critic_loss_2": per_head[1].detach(),
            "critic_grad_norm": torch.as_tensor(critic_grad).detach(),
            "expected_q1_mean": expected_heads[0].mean(),
            "expected_q2_mean": expected_heads[1].mean(),
            "twin_expected_q_disagreement": (expected_heads[0] - expected_heads[1])
            .abs()
            .mean(),
            **target_metrics,
            **alpha_metrics,
        }
        if hasattr(self, "_fastsac_rollout_critic_metrics"):
            self._fastsac_rollout_critic_metrics.append(metrics)
        return metrics

    def _actor_update(self, batch: dict[str, torch.Tensor]):
        """One mean-only BC step plus the ordinary reparameterized SAC step."""
        raw_prediction = self._actor_mean_from_flat(batch["observations"])
        physical_gaussian = self._uses_ppo_physical_gaussian()
        std_before = None
        if physical_gaussian:
            self.actor_std_optimizer.zero_grad(set_to_none=True)
            std_before = self._ppo_actor_std_parameter().detach().clone()
        dist = self._sac_dist_from_mean(raw_prediction)
        prediction_action = dist.mean
        sampled_action, raw_log_prob = dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        normalized_log_prob = self._normalized_action_log_prob(raw_log_prob)
        q_state, q_action = self._q_network_inputs(
            batch["critic_observations"],
            sampled_action,
            batch.get(Q_ACTUATOR_CONTEXT_KEY),
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        original_requires_grad = [
            parameter.requires_grad for parameter in self.qnet.parameters()
        ]
        try:
            for parameter in self.qnet.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            twin_outputs = self.qnet(q_state, q_action)
            twin_expected = self._q_output_values(self.qnet, twin_outputs)
            reduced_expected = self._reduce_twin_q_values(twin_expected)
            sac_actor_loss = (
                self.log_alpha.exp().detach() * normalized_log_prob
                - reduced_expected
            ).mean()
        finally:
            for parameter, requires_grad in zip(
                self.qnet.parameters(), original_requires_grad
            ):
                parameter.requires_grad_(requires_grad)

        raw_teacher_actions = batch[DAGGER_REPLAY_TEACHER_ACTIONS]
        teacher_valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY].reshape(-1).bool()
        # Teacher replay/Q keeps the factual PPO command. A bounded tanh Actor
        # needs its closest nominal-support target; a physical Gaussian can
        # represent the same execution-valid physical command without that
        # lossy projection.
        teacher_actions = self._project_student_policy_action(raw_teacher_actions)
        teacher_bc_projection = teacher_valid & (
            teacher_actions != raw_teacher_actions
        ).any(dim=-1)
        teacher_bc_projection_fraction = (
            teacher_bc_projection.float().sum()
            / teacher_valid.float().sum().clamp_min(1.0)
        )
        q_filter_metrics = {
            # Compatibility alias: under continuous SPReD-P this is the mean
            # effective BC weight, not a fraction of binary-active rows.
            "q_filtered_bc_active_fraction": prediction_action.new_tensor(1.0),
            "q_filtered_bc_policy_better_fraction": prediction_action.new_zeros(()),
            "spred_p_bc_weight_mean": prediction_action.new_tensor(1.0),
            "spred_p_bc_weight_std": prediction_action.new_zeros(()),
            "spred_p_bc_weight_min": prediction_action.new_tensor(1.0),
            "spred_p_bc_weight_max": prediction_action.new_tensor(1.0),
            "spred_p_teacher_advantage_mean": prediction_action.new_zeros(()),
            "spred_p_combined_q_std_mean": prediction_action.new_zeros(()),
            **{
                key: prediction_action.new_zeros(())
                for key in _SPRED_P_SOURCE_METRIC_KEYS
            },
        }
        if getattr(self.cfg, "use_q_filtered_bc", False):
            if prediction_action.shape != teacher_actions.shape:
                raise ValueError("SPReD-P Student and Teacher action shapes must match")
            if prediction_action.shape[0] != teacher_valid.numel():
                raise ValueError("SPReD-P validity mask does not match batch rows")
            # Invalid replay labels may contain NaN. Substitute a finite action
            # before the online-Q forward, then exclude those rows from both
            # the probability statistics and BC normalization below.
            safe_teacher_actions = torch.where(
                teacher_valid.unsqueeze(-1),
                teacher_actions,
                prediction_action.detach(),
            )
            with torch.no_grad():
                # SPReD-P uses the current online ensemble. C51 heads collapse
                # to expectations while scalar heads pass through unchanged.
                # Sequential forwards keep peak memory below a doubled batch.
                policy_online_outputs = self._q_forward(
                    self.qnet,
                    batch["critic_observations"],
                    prediction_action.detach(),
                    batch.get(Q_ACTUATOR_CONTEXT_KEY),
                )
                policy_online_q = self._q_output_values(
                    self.qnet, policy_online_outputs
                )
                del policy_online_outputs
                teacher_online_outputs = self._q_forward(
                    self.qnet,
                    batch["critic_observations"],
                    safe_teacher_actions.detach(),
                    batch.get(Q_ACTUATOR_CONTEXT_KEY),
                )
                teacher_online_q = self._q_output_values(
                    self.qnet, teacher_online_outputs
                )
                (
                    teacher_probability,
                    teacher_advantage,
                    combined_q_std,
                ) = _spred_p_teacher_probability(
                    policy_online_q,
                    teacher_online_q,
                )

            if teacher_valid.any():
                center = self._fastsac_q_action_center.to(prediction_action)
                scale = self._fastsac_q_action_scale.to(prediction_action)
                per_element_bc = F.smooth_l1_loss(
                    (prediction_action - center) / scale,
                    (safe_teacher_actions.detach() - center) / scale,
                    beta=float(self.cfg.dagger_actor_huber_delta),
                    reduction="none",
                )
                per_row_bc = per_element_bc.flatten(start_dim=1).mean(dim=1)
                bc_weights = torch.where(
                    teacher_valid,
                    teacher_probability.to(per_row_bc),
                    torch.zeros_like(per_row_bc),
                )
                # Normalize by all valid Teacher rows exactly as an ordinary
                # per-demo weighted loss.  Its effective strength changes
                # continuously without a hand-authored lambda scheduler.
                exact_bc_loss = (
                    per_row_bc * bc_weights
                ).sum() / teacher_valid.sum()
                valid_weights = teacher_probability[teacher_valid]
                valid_advantage = teacher_advantage[teacher_valid]
                valid_combined_std = combined_q_std[teacher_valid]
                policy_better = valid_advantage < 0.0
                q_filter_metrics.update(
                    {
                        "q_filtered_bc_active_fraction": valid_weights.mean(),
                        "q_filtered_bc_policy_better_fraction": (
                            policy_better.float().mean()
                        ),
                        "spred_p_bc_weight_mean": valid_weights.mean(),
                        "spred_p_bc_weight_std": valid_weights.std(unbiased=False),
                        "spred_p_bc_weight_min": valid_weights.min(),
                        "spred_p_bc_weight_max": valid_weights.max(),
                        "spred_p_teacher_advantage_mean": valid_advantage.mean(),
                        "spred_p_combined_q_std_mean": valid_combined_std.mean(),
                    }
                )
            else:
                exact_bc_loss = prediction_action.sum() * 0.0
                q_filter_metrics = {
                    key: prediction_action.new_zeros(()) for key in q_filter_metrics
                }

            # A single aggregate can hide opposite behavior on factual
            # Teacher-replay rows and counterfactual Teacher labels attached to
            # Student-replay rows.  Preserve the replay provenance and report
            # SPReD-P diagnostics independently for both sources.  These
            # statistics are read-only: neither the masks nor the detached
            # probability feed a new gradient path into the Critic.
            teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
            if teacher_source is not None:
                if teacher_source.numel() != teacher_valid.numel():
                    raise ValueError(
                        "SPReD-P replay-source mask does not match batch rows"
                    )
                teacher_source = teacher_source.reshape(-1).bool()
                q_filter_metrics["spred_p_source_metadata_available"] = (
                    prediction_action.new_tensor(1.0)
                )
                policy_better_values = (teacher_advantage < 0.0).float()
                for source_name, source_mask in (
                    ("teacher", teacher_source),
                    ("student", ~teacher_source),
                ):
                    source_valid = teacher_valid & source_mask
                    source_valid_float = source_valid.to(teacher_probability)
                    source_valid_count = source_valid_float.sum().clamp_min(1.0)
                    q_filter_metrics.update(
                        {
                            f"spred_p_{source_name}_source_valid_fraction": (
                                source_valid_float.mean()
                            ),
                            f"spred_p_{source_name}_source_bc_weight_mean": (
                                (teacher_probability * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"spred_p_{source_name}_source_teacher_advantage_mean": (
                                (teacher_advantage * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"spred_p_{source_name}_source_combined_q_std_mean": (
                                (combined_q_std * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"q_filtered_bc_{source_name}_source_policy_better_fraction": (
                                (policy_better_values * source_valid_float).sum()
                                / source_valid_count
                            ),
                        }
                    )
        else:
            exact_bc_loss = _exact_teacher_bc_loss(
                prediction_action,
                teacher_actions,
                teacher_valid,
                self._fastsac_q_action_center,
                self._fastsac_q_action_scale,
                float(self.cfg.dagger_actor_huber_delta),
            )
        weighted_sac = float(self.cfg.eta_sac) * sac_actor_loss
        weighted_bc = float(self.cfg.lambda_bc) * exact_bc_loss
        total_actor_loss = weighted_sac + weighted_bc
        total_actor_loss.backward()
        if physical_gaussian and self._ppo_actor_std_parameter().grad is None:
            raise RuntimeError(
                "SAC Actor loss did not update physical actor_std"
            )
        actor_parameters = tuple(
            parameter
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        )
        actor_grad = torch.nn.utils.clip_grad_norm_(
            actor_parameters, float(self.cfg.sac_max_grad_norm)
        )
        if any(parameter.grad is not None for parameter in self.qnet.parameters()):
            raise RuntimeError("Critic parameters accumulated Actor-step gradients")
        self.actor_optimizer.step()
        if physical_gaussian:
            if std_before is None:
                raise RuntimeError("physical std update snapshot was not captured")
            std_metrics = self._physical_actor_std_update(std_before)
        else:
            std_metrics = {}
        self.actor_update_count = int(getattr(self, "actor_update_count", 0)) + 1
        self.sac_actor_update_count = (
            int(getattr(self, "sac_actor_update_count", 0)) + 1
        )
        teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
        actor_teacher_replay_fraction = (
            prediction_action.new_zeros(())
            if teacher_source is None
            else teacher_source.float().mean()
        )
        minimum_expected = twin_expected.min(dim=0).values
        mean_expected = twin_expected.mean(dim=0)
        metrics = {
            # Compatibility names consumed by the inherited replay loop.
            "td3_actor_loss": sac_actor_loss.detach(),
            "weighted_td3_actor_loss": weighted_sac.detach(),
            # Native FastSAC diagnostics.
            "sac_actor_loss": sac_actor_loss.detach(),
            "exact_bc_loss": exact_bc_loss.detach(),
            "weighted_sac_actor_loss": weighted_sac.detach(),
            "weighted_bc_loss": weighted_bc.detach(),
            "total_actor_loss": total_actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad).detach(),
            "actor_mean_grad_norm": torch.as_tensor(actor_grad).detach(),
            "actor_expected_q1_mean": twin_expected[0].detach().mean(),
            "actor_expected_q2_mean": twin_expected[1].detach().mean(),
            "actor_min_expected_q_mean": minimum_expected.detach().mean(),
            "actor_expected_q_min_mean": minimum_expected.detach().mean(),
            "actor_mean_expected_q_mean": mean_expected.detach().mean(),
            "actor_reduced_expected_q_mean": reduced_expected.detach().mean(),
            "actor_log_prob_mean": normalized_log_prob.detach().mean(),
            "actor_entropy": -normalized_log_prob.detach().mean(),
            "actor_sample_action_abs_mean": sampled_action.detach().abs().mean(),
            "actor_mean_action_abs_mean": dist.mean.detach().abs().mean(),
            "actor_log_std_mean": self._policy_log_std().detach().mean(),
            "actor_raw_log_std_mean": self._policy_raw_log_std().detach().mean(),
            "actor_std_min": self._policy_log_std().detach().exp().min(),
            "actor_std_max": self._policy_log_std().detach().exp().max(),
            "actor_teacher_replay_fraction": (actor_teacher_replay_fraction.detach()),
            "teacher_bc_student_support_projection_fraction": (
                teacher_bc_projection_fraction.detach()
            ),
            **{key: value.detach() for key, value in q_filter_metrics.items()},
            "actor_failure_phase_teacher_fraction": batch.get(
                FAILURE_PHASE_TEACHER_SOURCE_KEY,
                torch.zeros_like(batch[DAGGER_TEACHER_ACTION_VALID_KEY]),
            )
            .float()
            .mean()
            .detach(),
            "actor_failure_phase_student_fraction": batch.get(
                FAILURE_PHASE_STUDENT_SOURCE_KEY,
                torch.zeros_like(batch[DAGGER_TEACHER_ACTION_VALID_KEY]),
            )
            .float()
            .mean()
            .detach(),
            "alpha": self.log_alpha.exp().detach(),
            **std_metrics,
        }
        if hasattr(self, "_fastsac_rollout_actor_metrics"):
            self._fastsac_rollout_actor_metrics.append(metrics)
        return metrics

    def _maybe_delayed_actor_and_targets(self, actor_batch: dict[str, torch.Tensor]):
        if self.critic_update_count % int(self.cfg.sac_policy_frequency):
            return None
        # SAC has no target Actor.  Q target Polyak is performed on every
        # Critic update, not on this delayed policy cadence.
        return self._actor_update(actor_batch)

    def train_op(self, tensordict):
        """Reuse raw replay/prefill orchestration and publish SAC-native logs."""
        self._fastsac_rollout_critic_metrics: list[dict[str, torch.Tensor]] = []
        self._fastsac_rollout_actor_metrics: list[dict[str, torch.Tensor]] = []
        prefill_teacher_noise_q_rms = 0.0
        prefill_teacher_noise_physical_rms = 0.0
        prefill_teacher_projection_fraction = 0.0
        if (
            self._teacher_prefill_active()
            and FASTSAC_PREFILL_TEACHER_NOISE_KEY
            in tensordict.keys(True, True)
            and DAGGER_IS_STUDENT_ACTION_KEY in tensordict.keys(True, True)
        ):
            teacher_rows = ~tensordict[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
            if bool(teacher_rows.any()):
                q_noise = tensordict[FASTSAC_PREFILL_TEACHER_NOISE_KEY].reshape(
                    -1, self.action_dim
                )[teacher_rows]
                prefill_teacher_noise_q_rms = float(
                    q_noise.float().square().mean().sqrt().item()
                )
                physical_noise = (
                    q_noise
                    / float(self.cfg.q_action_input_gain)
                    * self._fastsac_q_action_scale.to(q_noise)
                )
                prefill_teacher_noise_physical_rms = float(
                    physical_noise.float().square().mean().sqrt().item()
                )
                if (
                    FASTSAC_PREFILL_TEACHER_PROJECTION_KEY
                    in tensordict.keys(True, True)
                ):
                    projection = tensordict[
                        FASTSAC_PREFILL_TEACHER_PROJECTION_KEY
                    ].reshape(-1).bool()
                    prefill_teacher_projection_fraction = float(
                        projection[teacher_rows].float().mean().item()
                    )
        if self._uses_ppo_physical_gaussian():
            self._project_physical_actor_std_()
            self._physical_std_rollout_reference = (
                self._bounded_physical_actor_std(detach=True).clone()
            )
        action_projection_fraction = 0.0
        student_nominal_bound_violation_fraction = 0.0
        student_support_saturation_fraction = 0.0
        student_q_action_abs_max = 0.0
        if (
            FASTSAC_ACTION_PROJECTION_KEY in tensordict.keys(True, True)
            and DAGGER_IS_STUDENT_ACTION_KEY in tensordict.keys(True, True)
        ):
            projected = tensordict[FASTSAC_ACTION_PROJECTION_KEY].reshape(-1).bool()
            student = tensordict[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
            if bool(student.any()):
                action_projection_fraction = float(
                    projected[student].float().mean().item()
                )
                sampled = tensordict[TD3_EXPLORATORY_STUDENT_ACTION_KEY].reshape(
                    -1, self.action_dim
                )[student]
                low, high, _, _ = self._student_action_support(sampled)
                violations = ((sampled < low) | (sampled > high)).any(dim=-1)
                student_nominal_bound_violation_fraction = float(
                    violations.float().mean().item()
                )
                endpoint_tolerance = (high - low) * 1.0e-6
                saturated = ((sampled - low) <= endpoint_tolerance) | (
                    (high - sampled) <= endpoint_tolerance
                )
                student_support_saturation_fraction = float(
                    saturated.float().mean().item()
                )
                student_q_action_abs_max = float(
                    self._q_action_input(sampled).abs().amax().item()
                )
        td3_info = DistributionalTD3TeacherBC.train_op(self, tensordict)
        physical_rollout_std_metrics = {}
        if self._uses_ppo_physical_gaussian():
            # Replay optimization has finished and no environment action can be
            # issued until this method returns. Enforce the cumulative old-scale
            # trust guard once on the policy that will collect the next rollout;
            # hard physical bounds were retained after every std Adam step.
            rollout_scale_kl, rollout_kl_capped = (
                self._cap_physical_actor_std_rollout_kl_(
                    self._physical_std_rollout_reference
                )
            )
            physical_rollout_std_metrics = {
                "actor_std_rollout_scale_kl": float(rollout_scale_kl.item()),
                "actor_std_kl_cap_fraction": float(rollout_kl_capped.item()),
            }

        replacements = {
            "method_distributional_td3_teacher_bc_v1": (
                "method_distributional_fastsac_teacher_bc_v1"
            ),
            "td3_actor_loss": "sac_actor_loss",
            "weighted_td3_actor_loss": "weighted_sac_actor_loss",
            "collector_exploration_noise_norm": ("behavior_sample_q_deviation_norm"),
        }
        info = {}
        for key, value in td3_info.items():
            if not key.startswith("td3/"):
                info[key] = value
                continue
            suffix = key.removeprefix("td3/")
            if suffix == "target_smoothing_noise_norm":
                continue
            info[f"fastsac/{replacements.get(suffix, suffix)}"] = value

        info["fastsac/prefill_teacher_ppo_noise_enabled"] = float(
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
        )
        info["fastsac/prefill_teacher_ppo_noise_q_rms"] = (
            prefill_teacher_noise_q_rms
        )
        info["fastsac/prefill_teacher_ppo_noise_physical_rms"] = (
            prefill_teacher_noise_physical_rms
        )
        info["fastsac/prefill_teacher_action_projection_fraction"] = (
            prefill_teacher_projection_fraction
        )

        critic_keys = (
            "target_sample_action_abs_mean",
            "target_log_prob_mean",
            "reduced_target_expected_mean",
            "target_q1_contribution_fraction",
            "target_q2_contribution_fraction",
            "entropy_tax_mean",
            "entropy_tax_abs_mean",
            "entropy_tax_reward_abs_ratio",
            "effective_n_steps_mean",
            "alpha_update_due_fraction",
            "alpha_update_performed_fraction",
            "q1_left_support_clip_fraction",
            "q1_right_support_clip_fraction",
            "q2_left_support_clip_fraction",
            "q2_right_support_clip_fraction",
            "support_clip_fraction_mean",
            "support_clip_fraction_max",
        )
        actor_keys = (
            "actor_mean_grad_norm",
            "actor_expected_q1_mean",
            "actor_expected_q2_mean",
            "actor_min_expected_q_mean",
            "actor_mean_expected_q_mean",
            "actor_reduced_expected_q_mean",
            "actor_log_prob_mean",
            "actor_entropy",
            "actor_sample_action_abs_mean",
            "actor_mean_action_abs_mean",
            "actor_log_std_mean",
            "actor_raw_log_std_mean",
            "actor_std_min",
            "actor_std_max",
            "teacher_bc_student_support_projection_fraction",
            "q_filtered_bc_active_fraction",
            "q_filtered_bc_policy_better_fraction",
            "spred_p_bc_weight_mean",
            "spred_p_bc_weight_std",
            "spred_p_bc_weight_min",
            "spred_p_bc_weight_max",
            "spred_p_teacher_advantage_mean",
            "spred_p_combined_q_std_mean",
            *_SPRED_P_SOURCE_METRIC_KEYS,
        )
        if self._uses_ppo_physical_gaussian():
            actor_keys += (
                "actor_std_sac_grad_norm",
                "actor_std_step_abs_mean",
                "actor_std_projection_fraction",
                "actor_std_at_min_fraction",
                "actor_std_at_max_fraction",
                "actor_std_lr",
            )
        critic = self._mean_metric_dict(
            self._fastsac_rollout_critic_metrics, critic_keys
        )
        # Skipped Critic steps intentionally carry zero temperature metrics.
        # Report loss/gradient over optimizer steps only, so a delayed cadence
        # does not dilute them in the rollout aggregate.
        if self._fastsac_rollout_critic_metrics:
            performed = torch.stack(
                [
                    metrics["alpha_update_performed_fraction"].detach().float()
                    for metrics in self._fastsac_rollout_critic_metrics
                ]
            )
            performed_count = performed.sum().clamp_min(1.0)
            for key in ("alpha_loss", "alpha_grad_norm"):
                values = torch.stack(
                    [
                        metrics[key].detach().float()
                        for metrics in self._fastsac_rollout_critic_metrics
                    ]
                )
                critic[key] = ((values * performed).sum() / performed_count).item()
        else:
            critic.update({"alpha_loss": 0.0, "alpha_grad_norm": 0.0})
        # Publish the temperature actually available to the next rollout,
        # rather than the within-rollout mean of stale and updated values.
        critic["alpha"] = self.log_alpha.detach().exp().item()
        actor = self._mean_metric_dict(self._fastsac_rollout_actor_metrics, actor_keys)
        actor.update(physical_rollout_std_metrics)
        info.update({f"fastsac/{key}": value for key, value in critic.items()})
        info.update({f"fastsac/{key}": value for key, value in actor.items()})
        info["fastsac/action_projection_fraction"] = action_projection_fraction
        info["fastsac/student_nominal_bound_violation_fraction"] = (
            student_nominal_bound_violation_fraction
        )
        info["fastsac/student_support_saturation_fraction"] = (
            student_support_saturation_fraction
        )
        info["fastsac/student_q_action_abs_max"] = student_q_action_abs_max
        info["fastsac/alpha_update_count"] = self.alpha_update_count
        info["fastsac/actor_std_update_count"] = int(
            getattr(self, "actor_std_update_count", 0)
        )
        if self._uses_ppo_physical_gaussian():
            current_std = self._bounded_physical_actor_std(detach=True)
            q_scale = self._fastsac_q_action_scale.detach().to(current_std)
            normalized_std = current_std / q_scale
            lower, upper = self._physical_std_bounds()
            current_std_values = current_std.float().cpu().tolist()
            for joint_name, std in zip(
                self.joint_names, current_std_values, strict=True
            ):
                info[f"fastsac/actor_std/{joint_name}"] = float(std)
            info["fastsac/actor_std_physical_mean"] = float(
                current_std.float().mean().item()
            )
            info["fastsac/actor_std_physical_geometric_mean"] = float(
                current_std.float().log().mean().exp().item()
            )
            info["fastsac/actor_std_q_normalized_l2"] = float(
                torch.linalg.vector_norm(normalized_std.float()).item()
            )
            info["fastsac/actor_std_q_normalized_rms"] = float(
                normalized_std.float().square().mean().sqrt().item()
            )
            info["fastsac/actor_std_q_normalized_geometric_mean"] = float(
                normalized_std.float().log().mean().exp().item()
            )
            info["fastsac/actor_std_q_normalized_min"] = float(
                normalized_std.float().min().item()
            )
            info["fastsac/actor_std_q_normalized_max"] = float(
                normalized_std.float().max().item()
            )
            info["fastsac/physical_std_lower_bound_min"] = float(
                lower.float().min().item()
            )
            info["fastsac/physical_std_lower_bound_max"] = float(
                lower.float().max().item()
            )
            info["fastsac/physical_std_upper_bound_min"] = float(
                upper.float().min().item()
            )
            info["fastsac/physical_std_upper_bound_max"] = float(
                upper.float().max().item()
            )
            entropy_min, entropy_max = self._physical_normalized_entropy_bounds()
            info["fastsac/physical_entropy_bound_min"] = entropy_min
            info["fastsac/physical_entropy_bound_max"] = entropy_max
        info["fastsac/target_entropy"] = float(self.target_entropy)
        info["fastsac/q_twin_reduction_min"] = float(
            self._uses_min_q_twin_reduction()
        )
        info["fastsac/q_twin_reduction_mean"] = float(
            not self._uses_min_q_twin_reduction()
        )
        info["fastsac/privileged_oracle_actor_observation_mode"] = float(
            self._uses_privileged_oracle_actor_observations()
        )
        self._last_fastsac_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if key.startswith("fastsac/") and isinstance(value, (int, float))
        }
        return info

    def _q_backend_metadata(self):
        metadata = DistributionalTD3TeacherBC._q_backend_metadata(self)
        alpha_update_cadence = str(self.cfg.sac_alpha_update_cadence)
        critic_type = _fastsac_q_critic_type(self.cfg)
        q_twin_reduction = _fastsac_q_twin_reduction(self.cfg)
        metadata.update(
            {
                "q_critic_type": critic_type,
                "q_twin_reduction": q_twin_reduction,
                "target_semantics": self._critic_learning_semantics(),
                "num_atoms": (
                    1
                    if critic_type == SCALAR_Q_CRITIC_TYPE
                    else int(self.cfg.q_num_atoms)
                ),
                "clipped_double_distribution": (
                    critic_type != SCALAR_Q_CRITIC_TYPE
                    and q_twin_reduction == Q_TWIN_REDUCTION_MIN
                ),
                "clipped_double_q": q_twin_reduction == Q_TWIN_REDUCTION_MIN,
                "critic_loss": (
                    "twin_scalar_bellman_mse"
                    if critic_type == SCALAR_Q_CRITIC_TYPE
                    else "twin_c51_cross_entropy"
                ),
                "categorical": critic_type != SCALAR_Q_CRITIC_TYPE,
                "v_min": (
                    None
                    if critic_type == SCALAR_Q_CRITIC_TYPE
                    else float(self.cfg.q_v_min)
                ),
                "v_max": (
                    None
                    if critic_type == SCALAR_Q_CRITIC_TYPE
                    else float(self.cfg.q_v_max)
                ),
                "target_q_reduction": (
                    "lower_complete_twin_target"
                    if q_twin_reduction == Q_TWIN_REDUCTION_MIN
                    else "mean_complete_twin_target"
                ),
                "actor_q_reduction": (
                    "minimum_online_twin_expectations"
                    if q_twin_reduction == Q_TWIN_REDUCTION_MIN
                    else "mean_online_twin_expectations"
                ),
                "actor_mean_optimizer_semantics": ACTOR_MEAN_OPTIMIZER_SEMANTICS,
                "actor_mean_weight_decay": _fastsac_actor_weight_decay(self.cfg),
                "bc_weighting_semantics": (
                    SPRED_P_BC_SEMANTICS
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "fixed_unweighted_valid_teacher_bc"
                ),
                "actor_target": False,
                "stochastic_actor": True,
                "action_distribution": _fastsac_action_distribution(self.cfg),
                "actor_observation_mode": str(
                    getattr(
                        self.cfg,
                        "sac_actor_observation_mode",
                        STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
                    )
                ),
                "student_action_contract": (
                    None
                    if self._uses_ppo_physical_gaussian()
                    else copy.deepcopy(self._fastsac_student_action_contract)
                ),
                "entropy_semantics": self._entropy_semantics(),
                "entropy_reference_log_scale_sum": (
                    self._fastsac_entropy_reference_log_scale_sum
                ),
                "target_entropy": float(self.target_entropy),
                "temperature_update_cadence": (
                    "delayed_actor_update"
                    if alpha_update_cadence == "actor"
                    else "every_critic_update"
                ),
                "temperature_update_frequency_critic_steps": (
                    int(self.cfg.sac_policy_frequency)
                    if alpha_update_cadence == "actor"
                    else 1
                ),
            }
        )
        if self._uses_ppo_physical_gaussian():
            entropy_min, entropy_max = self._physical_normalized_entropy_bounds()
            lower, upper = self._physical_std_bounds()
            metadata.update(
                {
                    "physical_std_update_semantics": (
                        "sac_q_entropy_gradient_separate_low_lr_adam_rollout_scale_"
                        "kl_cap_then_configured_jointwise_hard_projection_v3"
                    ),
                    "physical_std_lr": float(self.cfg.sac_physical_std_lr),
                    "physical_std_max_kl": float(
                        self.cfg.sac_physical_std_max_kl
                    ),
                    "physical_std_bound_mode": self._physical_std_bound_mode(),
                    "physical_std_min": float(self.cfg.sac_physical_std_min),
                    "physical_std_max": float(self.cfg.sac_physical_std_max),
                    "physical_std_normalized_min": float(
                        getattr(self.cfg, "sac_physical_std_normalized_min", 0.02)
                    ),
                    "physical_std_normalized_max": float(
                        getattr(self.cfg, "sac_physical_std_normalized_max", 0.11)
                    ),
                    "physical_std_resolved_lower": lower.detach().cpu().tolist(),
                    "physical_std_resolved_upper": upper.detach().cpu().tolist(),
                    "physical_entropy_bound_min": entropy_min,
                    "physical_entropy_bound_max": entropy_max,
                }
            )
        return metadata

    def _checkpoint_config(self):
        common = DistributionalTD3TeacherBC._checkpoint_config(self)
        for name in (
            "eta_td3",
            "target_policy_noise_std",
            "target_policy_noise_clip",
            "collector_exploration_noise_std",
            "collector_exploration_noise_clip",
        ):
            common.pop(name, None)
        common.update(
            {
                name: getattr(self.cfg, name)
                for name in (
                    "eta_sac",
                    "lambda_bc",
                    "use_q_filtered_bc",
                    "q_critic_type",
                    "q_twin_reduction",
                    "teacher_prefill_use_ppo_noise",
                    "dagger_env_fraction",
                    "student_buffer_capacity",
                    "q_condition_on_actuator_state",
                    "q_use_predicted_effect",
                    "q_use_residual_film",
                    "q_residual_film_scale",
                    "sac_actor_lr",
                    "student_actor_initialization",
                    "actor_adopt_checkpoint_path",
                    "sac_action_distribution",
                    "sac_physical_std_lr",
                    "sac_physical_std_max_kl",
                    "sac_physical_std_bound_mode",
                    "sac_physical_std_min",
                    "sac_physical_std_max",
                    "sac_physical_std_normalized_min",
                    "sac_physical_std_normalized_max",
                    "load_noise_scale",
                    "sac_initial_action_std",
                    "sac_log_std_min",
                    "sac_log_std_max",
                    "sac_alpha_init",
                    "sac_alpha_lr",
                    "sac_use_autotune",
                    "sac_alpha_update_cadence",
                    "sac_target_entropy_ratio",
                    "sac_policy_frequency",
                    "sac_learning_starts",
                    "sac_tau",
                    "sac_max_grad_norm",
                )
            }
        )
        common["sac_actor_weight_decay"] = _fastsac_actor_weight_decay(self.cfg)
        for path_name in (
            "perception_checkpoint_path",
            "actor_adopt_checkpoint_path",
        ):
            configured_path = common.get(path_name)
            if isinstance(configured_path, str) and configured_path:
                common[path_name] = str(
                    Path(configured_path).expanduser().resolve()
                )
        common["sac_actor_observation_mode"] = str(
            getattr(
                self.cfg,
                "sac_actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "actor_output": (
                    "raw_physical_joint_std_gaussian_with_bounded_std_and_finite_"
                    "action_safety_projection"
                    if self._uses_ppo_physical_gaussian()
                    else (
                        "ppo_physical_mean_calibrated_to_nominal_joint_bounded_"
                        "tanh_normal"
                    )
                ),
                "actor_observation_mode": str(
                    getattr(
                        self.cfg,
                        "sac_actor_observation_mode",
                        STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
                    )
                ),
                "bc_loss": (
                    "detached_spred_p_weighted_joint_normalized_raw_mean_teacher_"
                    "smooth_l1"
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "joint_normalized_raw_mean_teacher_smooth_l1"
                ),
                "bc_q_filter": (
                    SPRED_P_BC_SEMANTICS
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "disabled"
                ),
            }
        )
        return common

    def _fastsac_checkpoint_state(self):
        # Canonicalize even lightweight/legacy optimizer constructions before
        # serializing so the saved param-group contract has stable scalar types.
        self._apply_actor_optimizer_weight_decay_contract()
        self._validate_actor_optimizer_weight_decay_contract()
        return {
            "training_algorithm": TRAINING_ALGORITHM,
            "checkpoint_version": CHECKPOINT_VERSION,
            "actor_backend": getattr(
                self, "actor_backend", _fastsac_actor_backend(self.cfg)
            ),
            "q_critic_type": _fastsac_q_critic_type(self.cfg),
            "q_twin_reduction": _fastsac_q_twin_reduction(self.cfg),
            "critic_learning_semantics": self._critic_learning_semantics(),
            "actor_learning_semantics": self._actor_learning_semantics(),
            "actor_mean_optimizer_semantics": ACTOR_MEAN_OPTIMIZER_SEMANTICS,
            "actor_mean_weight_decay": _fastsac_actor_weight_decay(self.cfg),
            "entropy_semantics": self._entropy_semantics(),
            "action_distribution": _fastsac_action_distribution(self.cfg),
            "actor_observation_mode": str(
                getattr(
                    self.cfg,
                    "sac_actor_observation_mode",
                    STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
                )
            ),
            "actor_initialization": copy.deepcopy(
                getattr(
                    self,
                    "_actor_initialization",
                    {
                        "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
                        "mode": _student_actor_initialization(self.cfg),
                        "teacher_actor_loaded": True,
                        "actor_adapt_mean_loaded": (
                            _student_actor_initialization(self.cfg)
                            == TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
                        ),
                        "actor_adapt_mean_fresh": (
                            _student_actor_initialization(self.cfg)
                            == FRESH_STUDENT_ACTOR_INITIALIZATION
                        ),
                        "source_phase": None,
                        "source_iter": None,
                        "legacy_inferred": True,
                    },
                )
            ),
            "actor_adopt_initialization": copy.deepcopy(
                getattr(
                    self,
                    "_actor_adopt_initialization",
                    {
                        "semantics": ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
                        "loaded": False,
                        "source_path": None,
                        "module": "actor_adapt",
                        "runtime_std_source": "load_noise_scale",
                        "perception_exact_match": None,
                        "perception_mismatched_modules": (),
                        "legacy_inferred": True,
                    },
                )
            ),
            "student_action_contract": (
                None
                if self._uses_ppo_physical_gaussian()
                else copy.deepcopy(self._fastsac_student_action_contract)
            ),
            "actor_adapt": self.actor_adapt.state_dict(),
            "bc_dagger_sac_adapter": self.bc_dagger_sac_adapter.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "optimizer_resume_state": {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "actor_std_optimizer": (
                    None
                    if getattr(self, "actor_std_optimizer", None) is None
                    else self.actor_std_optimizer.state_dict()
                ),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": (
                    None
                    if self.alpha_optimizer is None
                    else self.alpha_optimizer.state_dict()
                ),
                "adapt_optimizer": (
                    None if self.opt_adapt is None else self.opt_adapt.state_dict()
                ),
            },
            "actor_update_count": int(self.actor_update_count),
            "actor_std_update_count": int(
                getattr(self, "actor_std_update_count", 0)
            ),
            "critic_update_count": int(self.critic_update_count),
            "alpha_update_count": int(self.alpha_update_count),
            "q_update_row_credit": float(getattr(self, "q_update_row_credit", 0.0)),
            "dagger_rollout_count": int(self.dagger_rollout_count),
            "dagger_environment_steps": int(self.dagger_environment_steps),
            "teacher_prefill_rollout_count": int(self.teacher_prefill_rollout_count),
            "teacher_prefill_environment_steps": int(
                self.teacher_prefill_environment_steps
            ),
            "teacher_perception_warmup_complete": bool(
                getattr(self, "_teacher_perception_warmup_complete", False)
            ),
            "teacher_perception_warmup_updates": int(
                getattr(self, "_teacher_perception_warmup_updates", 0)
            ),
            "dagger_rng_state": self.dagger_rng.get_state(),
            "q_rng_state": self.q_rng.get_state(),
            "sac_action_rng_state": self.sac_action_rng.get_state(),
            "sac_rollout_rng_state": self.sac_rollout_rng.get_state(),
            "teacher_prefill_action_rng_state": (
                self.teacher_prefill_action_rng.get_state()
            ),
            "teacher_perception_rng_state": self.teacher_perception_rng.get_state(),
            "last_fastsac_diagnostics": copy.deepcopy(self._last_fastsac_diagnostics),
        }

    def _validate_scalar_q_checkpoint_architecture(
        self,
        state: Mapping,
        *,
        context: str,
    ) -> None:
        """Reject standard split-stem checkpoints from an incompatible topology.

        ``q_critic_type=scalar`` existed briefly with the C51 state/action
        trunk and later with an early-concat MLP. Their tensors and optimizer
        moments are not compatible with the balanced split-stem scalar critic,
        so critic type alone is not a sufficient checkpoint contract. The
        split-stem categorical variant has the same input/fusion contract but
        a distinct final-layer architecture marker.
        """
        critic_type = _fastsac_q_critic_type(self.cfg)
        if critic_type not in (
            SCALAR_Q_CRITIC_TYPE,
            DISTRIBUTIONAL_Q_CRITIC_TYPE,
        ):
            return
        backend = state.get("q_backend_config")
        if not isinstance(backend, Mapping):
            raise ValueError(
                f"{context} split-stem Q checkpoint lacks architecture metadata"
            )
        architecture = backend.get("q_architecture_semantics")
        expected_architecture = (
            FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS
            if critic_type == SCALAR_Q_CRITIC_TYPE
            else FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS
        )
        if architecture != expected_architecture:
            raise ValueError(
                f"{context} split-stem Q architecture is incompatible: expected "
                f"{expected_architecture!r}, got "
                f"{architecture!r}. Legacy one-atom C51-shaped and early-concat "
                "scalar critics cannot be resumed as the balanced split-stem "
                "scalar-style critic."
            )
        state_hidden_dim = int(self.cfg.q_hidden_dim) // 2
        expected_dimensions = {
            "q_state_input_dim": int(self._q_state_input_dim),
            "q_action_input_dim": int(self._q_action_input_dim),
            "q_state_hidden_dim": state_hidden_dim,
            "q_action_hidden_dim": int(self.cfg.q_hidden_dim) - state_hidden_dim,
        }
        for name, expected in expected_dimensions.items():
            saved = backend.get(name)
            if isinstance(saved, bool) or not isinstance(saved, int):
                raise ValueError(
                    f"{context} split-stem Q checkpoint lacks integer {name}"
                )
            if int(saved) != expected:
                raise ValueError(
                    f"{context} split-stem Q {name} mismatch: "
                    f"checkpoint={int(saved)}, runtime={expected}"
                )
        expected_fusion = {
            "q_action_fusion": "balanced_split_stems",
            "q_action_fusion_semantics": (
                FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS
            ),
        }
        for name, expected in expected_fusion.items():
            saved = backend.get(name)
            if saved != expected:
                raise ValueError(
                    f"{context} split-stem Q {name} mismatch: "
                    f"checkpoint={saved!r}, runtime={expected!r}"
                )

    def _validate_q_twin_reduction_checkpoint(
        self,
        state: Mapping,
        *,
        context: str,
    ) -> None:
        """Validate the learning-rule contract, defaulting legacy state to min."""
        q_backend = state.get("q_backend_config")
        if not isinstance(q_backend, Mapping):
            q_backend = {}
        dagger_backend = state.get("dagger_backend_config")
        if not isinstance(dagger_backend, Mapping):
            dagger_backend = {}
        reductions = {
            "top-level": state.get("q_twin_reduction"),
            "Q backend": q_backend.get("q_twin_reduction"),
            "DAgger backend": dagger_backend.get("q_twin_reduction"),
        }
        present_reductions = {
            source: value
            for source, value in reductions.items()
            if value is not None
        }
        if any(
            not isinstance(value, str) or value not in Q_TWIN_REDUCTIONS
            for value in present_reductions.values()
        ):
            raise ValueError(f"{context} Q twin reduction is invalid")
        if len(set(present_reductions.values())) > 1:
            raise ValueError(
                f"{context} Q twin reduction metadata is inconsistent"
            )
        saved_reduction = next(
            iter(present_reductions.values()), Q_TWIN_REDUCTION_MIN
        )
        if saved_reduction != _fastsac_q_twin_reduction(self.cfg):
            raise ValueError(f"{context} Q twin reduction mismatch")

    def _validate_student_action_checkpoint_contract(
        self, state: Mapping, *, context: str
    ) -> None:
        saved = state.get("student_action_contract")
        if self._uses_ppo_physical_gaussian():
            if saved is not None:
                raise ValueError(
                    f"{context} physical Gaussian unexpectedly has a bounded "
                    "Student action contract"
                )
            return
        if not isinstance(saved, Mapping):
            raise ValueError(f"{context} lacks its Student action contract")
        current = self._fastsac_student_action_contract
        for key in ("semantics", "joint_names", "fingerprint"):
            if saved.get(key) != current.get(key):
                raise ValueError(
                    f"{context} Student action contract mismatch at {key!r}"
                )

    @staticmethod
    def _validate_actor_adapt_overlay_target(
        target_state: Mapping,
        source_state: Mapping,
    ) -> None:
        """Require an exact actor_adapt schema before loading any tensor."""
        target_keys = set(target_state)
        source_keys = set(source_state)
        if target_keys != source_keys:
            missing = sorted(target_keys.difference(source_keys))
            unexpected = sorted(source_keys.difference(target_keys))
            raise RuntimeError(
                "actor_adopt checkpoint actor_adapt has incompatible keys; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, target_value in target_state.items():
            source_value = source_state[key]
            if torch.is_tensor(target_value):
                if not torch.is_tensor(source_value):
                    raise RuntimeError(
                        f"actor_adopt actor_adapt key {key!r} is not a tensor"
                    )
                if target_value.shape != source_value.shape:
                    raise RuntimeError(
                        f"actor_adopt actor_adapt key {key!r} shape mismatch: "
                        f"expected {tuple(target_value.shape)}, "
                        f"got {tuple(source_value.shape)}"
                    )
                if target_value.dtype != source_value.dtype:
                    raise RuntimeError(
                        f"actor_adopt actor_adapt key {key!r} dtype mismatch: "
                        f"expected {target_value.dtype}, got {source_value.dtype}"
                    )
            elif type(target_value) is not type(source_value):
                raise RuntimeError(
                    f"actor_adopt actor_adapt key {key!r} has incompatible type"
                )

    def _load_actor_adopt_checkpoint(
        self,
        path: str,
        *,
        teacher_source_policy: Mapping,
    ) -> dict:
        """Overlay only actor_adapt from the audited perception+Actor stage."""
        resolved_path = Path(path).expanduser().resolve(strict=True)
        checkpoint = torch.load(
            resolved_path,
            map_location="cpu",
            weights_only=False,
        )
        actor_state, provenance = validate_actor_adopt_checkpoint_payload(
            checkpoint,
            source_path=str(resolved_path),
        )
        source_policy = checkpoint["policy"]
        source_cfg, source_algo = _checkpoint_config_mapping(checkpoint)
        if tuple(source_algo.get("in_keys", ())) != tuple(self.cfg.in_keys):
            raise ValueError(
                "actor_adopt checkpoint Actor observation keys/order do not "
                "match the FastSAC actor_adapt"
            )
        if int(source_algo.get("latent_dim", -1)) != int(self.cfg.latent_dim):
            raise ValueError(
                "actor_adopt checkpoint latent_dim does not match FastSAC"
            )

        # The main checkpoint remains the privileged Teacher authority.  The
        # auxiliary stage freezes these children, so exact weights (apart from
        # Actor.actor_std, which every loader intentionally resets from its
        # runtime noise scale) prove that both stages used the same Teacher.
        teacher_mismatches = checkpoint_module_mismatches(
            teacher_source_policy,
            source_policy,
            ("actor", "encoder_priv", "critic"),
            ignored_key_suffixes=("actor_std",),
        )
        if teacher_mismatches:
            raise ValueError(
                "actor_adopt checkpoint was not trained from the same privileged "
                "Teacher checkpoint; mismatched frozen modules="
                f"{list(teacher_mismatches)}"
            )

        target_state = self.actor_adapt.state_dict()
        self._validate_actor_adapt_overlay_target(target_state, actor_state)
        parameter_ids = tuple(id(parameter) for parameter in self.actor_adapt.parameters())
        actor_std_keys = [key for key in target_state if key.endswith("actor_std")]
        if len(actor_std_keys) != 1:
            raise RuntimeError(
                "FastSAC actor_adapt must expose exactly one actor_std tensor"
            )
        actor_std_key = actor_std_keys[0]
        actor_std_before = target_state[actor_std_key].detach().clone()
        actor_state_to_load = dict(actor_state)
        # The auxiliary checkpoint supplies only the learned Actor body/mean.
        # Keep runtime variance explicitly even if Actor._load_from_state_dict's
        # load_noise_scale hook changes in the future.
        actor_state_to_load[actor_std_key] = actor_std_before

        # Compare, but never load, the actor checkpoint's perception tensors.
        # Independent later perception training is permitted because both
        # checkpoints regress the same frozen Teacher latent coordinates.  The
        # exact/non-exact pairing is retained in every downstream checkpoint.
        current_perception = {
            name: getattr(self, name).state_dict()
            for name in PRETRAINED_PERCEPTION_MODULES
        }
        perception_mismatches = checkpoint_module_mismatches(
            current_perception,
            source_policy,
            PRETRAINED_PERCEPTION_MODULES,
        )

        try:
            self.actor_adapt.load_state_dict(actor_state_to_load, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "failed to load actor_adapt from actor_adopt checkpoint"
            ) from exc
        if tuple(id(parameter) for parameter in self.actor_adapt.parameters()) != (
            parameter_ids
        ):
            raise RuntimeError(
                "actor_adopt overlay replaced actor_adapt Parameter objects"
            )
        loaded_actor_std = self.actor_adapt.state_dict()[actor_std_key]
        if not torch.equal(loaded_actor_std, actor_std_before):
            raise RuntimeError(
                "actor_adopt overlay changed runtime actor_std"
            )
        if self._uses_ppo_physical_gaussian():
            expected_std = torch.full_like(
                self._ppo_actor_std_parameter().detach(),
                float(self.cfg.load_noise_scale),
            )
            if not torch.equal(
                self._ppo_actor_std_parameter().detach(), expected_std
            ):
                raise RuntimeError(
                    "actor_adopt overlay imported checkpoint actor_std instead "
                    "of resetting it from runtime load_noise_scale"
                )

        provenance.update(
            {
                "source_teacher_checkpoint_path": str(
                    Path(source_cfg["checkpoint_path"]).expanduser().resolve()
                ),
                "perception_source_path": str(
                    Path(self.cfg.perception_checkpoint_path).expanduser().resolve()
                ),
                "perception_exact_match": not perception_mismatches,
                "perception_mismatched_modules": perception_mismatches,
            }
        )
        self._actor_adopt_initialization = copy.deepcopy(provenance)
        return copy.deepcopy(provenance)

    def _restore_actor_adopt_initialization_provenance(
        self, state: Mapping
    ) -> None:
        """Restore overlay provenance without reopening the historical source."""
        backend = state.get("dagger_backend_config")
        backend = backend if isinstance(backend, Mapping) else {}
        backend_path = backend.get("actor_adopt_checkpoint_path")
        initialization = state.get("actor_adopt_initialization")
        if initialization is None:
            if backend_path is not None:
                raise ValueError(
                    "FastSAC actor_adopt checkpoint path lacks overlay provenance"
                )
            initialization = {
                "semantics": ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
                "loaded": False,
                "source_path": None,
                "source_algorithm": None,
                "source_phase": None,
                "source_iter": None,
                "source_actor_bc_update_count": None,
                "source_actor_objective_semantics": None,
                "source_actor_initialization_semantics": None,
                "source_actor_bc_perception_source": None,
                "source_teacher_checkpoint_path": None,
                "source_task_name": None,
                "module": "actor_adapt",
                "runtime_std_source": "load_noise_scale",
                "perception_source_path": None,
                "perception_exact_match": None,
                "perception_mismatched_modules": (),
                "legacy_inferred": True,
            }
        if not isinstance(initialization, Mapping):
            raise ValueError("FastSAC actor_adopt initialization is invalid")
        initialization = copy.deepcopy(dict(initialization))
        if initialization.get("semantics") != ACTOR_ADOPT_CHECKPOINT_SEMANTICS:
            raise ValueError("FastSAC actor_adopt initialization semantics mismatch")
        loaded = initialization.get("loaded")
        if not isinstance(loaded, bool):
            raise ValueError("FastSAC actor_adopt loaded flag is invalid")
        runtime_path = _actor_adopt_checkpoint_path(self.cfg)
        if loaded:
            for name, expected in (
                ("source_algorithm", PERCEPTION_ACTOR_TRAINING_ALGORITHM),
                ("source_phase", "finetune"),
                ("source_actor_objective_semantics", PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS),
                ("source_actor_initialization_semantics", PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS),
                ("source_actor_bc_perception_source", PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE),
                ("module", "actor_adapt"),
                ("runtime_std_source", "load_noise_scale"),
            ):
                if initialization.get(name) != expected:
                    raise ValueError(
                        f"FastSAC actor_adopt provenance mismatch at {name!r}"
                    )
            for name in ("source_iter", "source_actor_bc_update_count"):
                value = initialization.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"FastSAC actor_adopt provenance has invalid {name!r}"
                    )
            for name in (
                "source_teacher_checkpoint_path",
                "source_task_name",
            ):
                if not isinstance(initialization.get(name), str) or not (
                    initialization.get(name)
                ):
                    raise ValueError(
                        f"FastSAC actor_adopt provenance has invalid {name!r}"
                    )
            if not isinstance(initialization.get("perception_exact_match"), bool):
                raise ValueError(
                    "FastSAC actor_adopt provenance lacks perception exact-match flag"
                )
            mismatches = initialization.get("perception_mismatched_modules")
            if not isinstance(mismatches, (tuple, list)):
                raise ValueError(
                    "FastSAC actor_adopt perception mismatch provenance is invalid"
                )
            if (
                len(set(mismatches)) != len(mismatches)
                or not set(mismatches).issubset(PRETRAINED_PERCEPTION_MODULES)
            ):
                raise ValueError(
                    "FastSAC actor_adopt perception mismatch modules are invalid"
                )
            if bool(mismatches) == bool(
                initialization.get("perception_exact_match")
            ):
                raise ValueError(
                    "FastSAC actor_adopt perception match provenance is inconsistent"
                )
            if not isinstance(backend_path, str) or not backend_path:
                raise ValueError(
                    "FastSAC actor_adopt provenance requires backend source path"
                )
            source_path = initialization.get("source_path")
            if not isinstance(source_path, str) or not source_path or (
                Path(source_path).expanduser().resolve()
                != Path(backend_path).expanduser().resolve()
            ):
                raise ValueError(
                    "FastSAC actor_adopt provenance/backend source path mismatch"
                )
            backend_perception_path = backend.get("perception_checkpoint_path")
            if backend.get("load_pretrained_perception") is not True or not isinstance(
                backend_perception_path, str
            ) or not backend_perception_path:
                raise ValueError(
                    "FastSAC actor_adopt provenance requires pretrained perception"
                )
            provenance_perception_path = initialization.get(
                "perception_source_path"
            )
            if not isinstance(provenance_perception_path, str) or not (
                provenance_perception_path
            ) or (
                Path(provenance_perception_path).expanduser().resolve()
                != Path(backend_perception_path).expanduser().resolve()
            ):
                raise ValueError(
                    "FastSAC actor_adopt provenance/perception source path mismatch"
                )
        elif backend_path is not None or initialization.get("source_path") is not None:
            raise ValueError(
                "FastSAC disabled actor_adopt provenance unexpectedly has a source"
            )
        if (
            runtime_path is None
        ) != (backend_path is None) or (
            runtime_path is not None
            and Path(runtime_path).expanduser().resolve()
            != Path(backend_path).expanduser().resolve()
        ):
            raise ValueError(
                "FastSAC actor_adopt checkpoint path/runtime mismatch"
            )
        self._actor_adopt_initialization = initialization

    def _restore_actor_initialization_provenance(self, state: Mapping) -> None:
        """Restore saved provenance, defaulting old checkpoints to Teacher BC."""
        backend = state.get("dagger_backend_config")
        backend = backend if isinstance(backend, Mapping) else {}
        backend_mode = backend.get(
            "student_actor_initialization",
            TEACHER_BC_STUDENT_ACTOR_INITIALIZATION,
        )
        if backend_mode not in STUDENT_ACTOR_INITIALIZATION_MODES:
            raise ValueError(
                "FastSAC checkpoint backend Actor initialization mode is invalid"
            )
        initialization = state.get("actor_initialization")
        if initialization is None:
            if backend_mode != TEACHER_BC_STUDENT_ACTOR_INITIALIZATION:
                raise ValueError(
                    "FastSAC fresh-Actor checkpoint lacks initialization provenance"
                )
            initialization = {
                "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
                "mode": TEACHER_BC_STUDENT_ACTOR_INITIALIZATION,
                "teacher_actor_loaded": True,
                "actor_adapt_mean_loaded": True,
                "actor_adapt_mean_fresh": False,
                "source_phase": None,
                "source_iter": None,
                "legacy_inferred": True,
            }
        if not isinstance(initialization, Mapping):
            raise ValueError("FastSAC checkpoint Actor initialization is invalid")
        initialization = dict(initialization)
        if initialization.get("semantics") != STUDENT_ACTOR_INITIALIZATION_SEMANTICS:
            raise ValueError(
                "FastSAC checkpoint Actor initialization semantics mismatch"
            )
        if initialization.get("mode") not in STUDENT_ACTOR_INITIALIZATION_MODES:
            raise ValueError("FastSAC checkpoint Actor initialization mode is invalid")
        mode = initialization["mode"]
        if mode != backend_mode:
            raise ValueError(
                "FastSAC checkpoint Actor initialization/backend mismatch"
            )
        if mode != _student_actor_initialization(self.cfg):
            raise ValueError(
                "FastSAC checkpoint Actor initialization/runtime mismatch"
            )
        expected_flags = {
            "teacher_actor_loaded": True,
            "actor_adapt_mean_loaded": (
                mode == TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
            ),
            "actor_adapt_mean_fresh": mode == FRESH_STUDENT_ACTOR_INITIALIZATION,
        }
        for name, expected in expected_flags.items():
            if initialization.get(name) is not expected:
                raise ValueError(
                    "FastSAC checkpoint Actor initialization flags are inconsistent"
                )
        self._actor_initialization = copy.deepcopy(initialization)
        self.__dict__.pop("_fresh_student_actor_constructor_state", None)
        self.__dict__.pop("_fresh_student_actor_constructor_parameter_ids", None)

    def _load_fastsac_checkpoint_state(self, state, *, load_modules=True):
        if state.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional FastSAC Teacher-BC checkpoint")
        if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        if state.get("actor_backend") != _fastsac_actor_backend(self.cfg):
            raise ValueError("distributional FastSAC actor backend mismatch")
        if str(state.get("q_critic_type", C51_Q_CRITIC_TYPE)) != (
            _fastsac_q_critic_type(self.cfg)
        ):
            raise ValueError("FastSAC checkpoint Q critic type mismatch")
        self._validate_q_twin_reduction_checkpoint(
            state, context="FastSAC resume"
        )
        self._validate_scalar_q_checkpoint_architecture(
            state, context="FastSAC resume"
        )
        saved_distribution = state.get(
            "action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION
        )
        if saved_distribution != _fastsac_action_distribution(self.cfg):
            raise ValueError(
                "distributional FastSAC action distribution mismatch"
            )
        saved_actor_observation_mode = str(
            state.get(
                "actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        runtime_actor_observation_mode = str(
            getattr(
                self.cfg,
                "sac_actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        if saved_actor_observation_mode != runtime_actor_observation_mode:
            raise ValueError("FastSAC checkpoint Actor observation mode mismatch")
        self._validate_student_action_checkpoint_contract(
            state, context="FastSAC resume"
        )
        has_actor_decay_contract = "actor_mean_weight_decay" in state
        saved_actor_decay = state.get("actor_mean_weight_decay", 0.0)
        if (
            isinstance(saved_actor_decay, bool)
            or not isinstance(saved_actor_decay, (int, float))
            or not math.isfinite(float(saved_actor_decay))
            or float(saved_actor_decay) < 0.0
        ):
            raise ValueError("FastSAC checkpoint Actor mean weight decay is invalid")
        if not math.isclose(
            float(saved_actor_decay),
            _fastsac_actor_weight_decay(self.cfg),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("FastSAC checkpoint Actor mean weight decay mismatch")
        saved_optimizer_semantics = state.get("actor_mean_optimizer_semantics")
        if has_actor_decay_contract:
            if saved_optimizer_semantics != ACTOR_MEAN_OPTIMIZER_SEMANTICS:
                raise ValueError(
                    "FastSAC checkpoint Actor mean optimizer semantics mismatch"
                )
        elif saved_optimizer_semantics is not None:
            raise ValueError(
                "legacy FastSAC checkpoint has incomplete Actor optimizer metadata"
            )
        if load_modules:
            for name in (
                "actor_adapt",
                "bc_dagger_sac_adapter",
                "qnet",
                "qnet_target",
            ):
                getattr(self, name).load_state_dict(state[name], strict=True)
            self._restore_checkpoint_physical_actor_std(
                state["actor_adapt"], context="FastSAC checkpoint actor_adapt"
            )
            self.log_alpha.data.copy_(state["log_alpha"].to(self.log_alpha))
        optimizers = state.get("optimizer_resume_state")
        if not isinstance(optimizers, dict):
            raise ValueError("FastSAC checkpoint lacks optimizer state")
        saved_actor_optimizer = optimizers["actor_optimizer"]
        if not isinstance(saved_actor_optimizer, Mapping):
            raise ValueError("FastSAC checkpoint lacks Actor optimizer state")
        saved_actor_groups = saved_actor_optimizer.get("param_groups")
        if not isinstance(saved_actor_groups, list) or not saved_actor_groups:
            raise ValueError("FastSAC checkpoint Actor optimizer groups are invalid")
        for index, group in enumerate(saved_actor_groups):
            if not isinstance(group, Mapping):
                raise ValueError("FastSAC checkpoint Actor optimizer group is invalid")
            decay = group.get("weight_decay", 0.0)
            if (
                isinstance(decay, bool)
                or not isinstance(decay, (int, float))
                or not math.isfinite(float(decay))
                or float(decay) < 0.0
            ):
                raise ValueError(
                    "FastSAC checkpoint Actor optimizer weight decay is invalid"
                )
            if has_actor_decay_contract:
                expected_decay = float(saved_actor_decay) if index == 0 else 0.0
                if not math.isclose(
                    float(decay), expected_decay, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "FastSAC checkpoint Actor optimizer group disagrees with "
                        "its explicit weight-decay contract"
                    )
        self.actor_optimizer.load_state_dict(saved_actor_optimizer)
        self._apply_actor_optimizer_weight_decay_contract()
        self._validate_actor_optimizer_weight_decay_contract()
        actor_std_optimizer_state = optimizers.get("actor_std_optimizer")
        actor_std_optimizer = getattr(self, "actor_std_optimizer", None)
        if actor_std_optimizer is None:
            if actor_std_optimizer_state is not None:
                raise ValueError(
                    "normalized FastSAC checkpoint contains a physical std optimizer"
                )
        else:
            if not isinstance(actor_std_optimizer_state, Mapping):
                raise ValueError(
                    "physical FastSAC checkpoint lacks actor_std optimizer state"
                )
            actor_std_optimizer.load_state_dict(actor_std_optimizer_state)
            self._project_physical_actor_std_()
        self.critic_optimizer.load_state_dict(optimizers["critic_optimizer"])
        if self.alpha_optimizer is not None:
            if optimizers.get("alpha_optimizer") is None:
                raise ValueError("FastSAC checkpoint lacks alpha optimizer state")
            self.alpha_optimizer.load_state_dict(optimizers["alpha_optimizer"])
        adapt_optimizer_state = optimizers["adapt_optimizer"]
        if self.opt_adapt is None:
            if adapt_optimizer_state is not None:
                raise ValueError(
                    "frozen perception checkpoint contains an active optimizer"
                )
        else:
            if adapt_optimizer_state is None:
                raise ValueError(
                    "trainable perception checkpoint lacks optimizer state"
                )
            self.opt_adapt.load_state_dict(adapt_optimizer_state)
        for name in (
            "actor_update_count",
            "actor_std_update_count",
            "critic_update_count",
            "alpha_update_count",
            "dagger_rollout_count",
            "dagger_environment_steps",
            "teacher_prefill_rollout_count",
            "teacher_prefill_environment_steps",
        ):
            setattr(self, name, int(state[name]))
        self.q_update_row_credit = float(state.get("q_update_row_credit", 0.0))
        q_batch_size = int(getattr(self.cfg, "q_batch_size", 512))
        if not 0.0 <= self.q_update_row_credit < q_batch_size:
            raise ValueError("FastSAC checkpoint Q UTD row credit is invalid")
        self._teacher_perception_warmup_complete = bool(
            state.get("teacher_perception_warmup_complete", False)
        )
        self._teacher_perception_warmup_updates = int(
            state.get("teacher_perception_warmup_updates", 0)
        )
        self.dagger_rng.set_state(state["dagger_rng_state"])
        self.q_rng.set_state(state["q_rng_state"])
        self.sac_action_rng.set_state(state["sac_action_rng_state"])
        self.sac_rollout_rng.set_state(state["sac_rollout_rng_state"])
        teacher_prefill_rng_state = state.get("teacher_prefill_action_rng_state")
        if teacher_prefill_rng_state is None:
            self.teacher_prefill_action_rng.manual_seed(int(self.cfg.q_seed) + 4)
        else:
            self.teacher_prefill_action_rng.set_state(teacher_prefill_rng_state)
        self.teacher_perception_rng.set_state(state["teacher_perception_rng_state"])
        self._last_fastsac_diagnostics = copy.deepcopy(
            state.get("last_fastsac_diagnostics", {})
        )
        self._restore_actor_initialization_provenance(state)
        self._restore_actor_adopt_initialization_provenance(state)
        self.qnet_target.requires_grad_(False).eval()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore FastSAC model tensors without attempting replay resume."""
        if state_dict.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional FastSAC Teacher-BC checkpoint")
        checkpoint_version = int(state_dict.get("checkpoint_version", -1))
        legacy_v3 = checkpoint_version == _LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION
        if checkpoint_version not in (
            _LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION,
            _SMOOTH_BOUNDED_STD_CHECKPOINT_VERSION,
            _PREVIOUS_PRE_PRIOR_CHECKPOINT_VERSION,
            PREVIOUS_CHECKPOINT_VERSION,
            CHECKPOINT_VERSION,
        ):
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        expected_actor_backend = (
            _LEGACY_EFFECTIVE_LOG_STD_ACTOR_BACKEND
            if legacy_v3
            else _fastsac_actor_backend(self.cfg)
        )
        if state_dict.get("actor_backend") != expected_actor_backend:
            raise ValueError("distributional FastSAC actor backend mismatch")
        if str(state_dict.get("q_critic_type", C51_Q_CRITIC_TYPE)) != (
            _fastsac_q_critic_type(self.cfg)
        ):
            raise ValueError("FastSAC inference checkpoint Q critic type mismatch")
        self._validate_q_twin_reduction_checkpoint(
            state_dict, context="FastSAC inference"
        )
        self._validate_scalar_q_checkpoint_architecture(
            state_dict, context="FastSAC inference"
        )
        saved_distribution = state_dict.get(
            "action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION
        )
        if saved_distribution != _fastsac_action_distribution(self.cfg):
            raise ValueError(
                "distributional FastSAC action distribution mismatch"
            )
        saved_actor_observation_mode = str(
            state_dict.get(
                "actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        runtime_actor_observation_mode = str(
            getattr(
                self.cfg,
                "sac_actor_observation_mode",
                STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
            )
        )
        if saved_actor_observation_mode != runtime_actor_observation_mode:
            raise ValueError(
                "FastSAC inference Actor observation mode mismatch"
            )
        self._validate_student_action_checkpoint_contract(
            state_dict, context="FastSAC inference"
        )
        saved_action_contract = state_dict.get("action_contract")
        if not isinstance(saved_action_contract, Mapping):
            raise ValueError("FastSAC inference checkpoint lacks its action contract")
        if not isinstance(saved_action_contract.get("fingerprint"), str):
            raise ValueError(
                "FastSAC inference checkpoint lacks a contract fingerprint"
            )
        for key in ("joint_names", "fingerprint"):
            if saved_action_contract.get(key) != self._fastsac_action_contract.get(key):
                raise ValueError(
                    f"FastSAC inference checkpoint action contract mismatch at {key!r}"
                )

        load_state = state_dict
        if legacy_v3:
            source_adapter = state_dict.get("bc_dagger_sac_adapter")
            if not isinstance(source_adapter, Mapping):
                raise ValueError("FastSAC v3 inference checkpoint lacks its adapter")
            source_config = state_dict.get("dagger_backend_config")
            if not isinstance(source_config, Mapping):
                raise ValueError(
                    "FastSAC v3 inference checkpoint lacks its backend config"
                )
            legacy_log_std = source_adapter.get("log_std")
            converted_adapter = dict(source_adapter)
            converted_adapter["log_std"] = self._legacy_effective_log_std_to_raw(
                legacy_log_std,
                source_config.get("sac_log_std_min"),
                source_config.get("sac_log_std_max"),
            )
            load_state = dict(state_dict)
            load_state["bc_dagger_sac_adapter"] = converted_adapter

        failed = PPOVEL.load_state_dict(self, load_state, strict)
        critical = {
            "actor_adapt",
            "bc_dagger_sac_adapter",
            "qnet",
            "qnet_target",
            "temporal_depth_gru_ema",
            "object_adapt_ema",
            "adapt_ema",
        }
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                "FastSAC inference checkpoint failed to restore critical modules: "
                f"{sorted(missing)}"
            )
        for name in (
            "actor_adapt",
            "bc_dagger_sac_adapter",
            "qnet",
            "qnet_target",
        ):
            source = load_state.get(name)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"FastSAC inference checkpoint lacks module mapping {name!r}"
                )
            getattr(self, name).load_state_dict(source, strict=True)
        self._restore_checkpoint_physical_actor_std(
            load_state["actor_adapt"], context="FastSAC inference actor_adapt"
        )
        if self._uses_ppo_physical_gaussian():
            self._project_physical_actor_std_()
        log_alpha = load_state.get("log_alpha")
        if not torch.is_tensor(log_alpha) or log_alpha.numel() != 1:
            raise ValueError("FastSAC inference checkpoint lacks scalar log_alpha")
        self.log_alpha.data.copy_(log_alpha.to(self.log_alpha))

        initialization = state_dict.get("perception_initialization")
        if isinstance(initialization, Mapping):
            self._perception_initialization = copy.deepcopy(dict(initialization))
        self._restore_actor_initialization_provenance(state_dict)
        self._restore_actor_adopt_initialization_provenance(state_dict)
        self.actor_target = None
        self._teacher_prefill_complete = True
        self._teacher_perception_warmup_complete = True
        self._freeze_teacher()
        self.actor_adapt.requires_grad_(False).eval()
        self.bc_dagger_sac_adapter.requires_grad_(False).eval()
        self.qnet.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        for name in (
            "depth_cnn",
            "temporal_depth_gru",
            "temporal_depth_gru_ema",
            "object_adapt",
            "object_adapt_ema",
            "adapt_module",
            "adapt_ema",
        ):
            getattr(self, name).requires_grad_(False).eval()
        self._freeze_legacy_actor_std()
        return failed

    def state_dict(self):
        state = PPOVEL.state_dict(self)
        state.update(self._fastsac_checkpoint_state())
        state.update(
            {
                "dagger_backend_config": self._checkpoint_config(),
                "action_contract": copy.deepcopy(self._fastsac_action_contract),
                "q_backend_config": self._q_backend_metadata(),
                "replay_resume_semantics": (
                    "fresh_only_online_exact_actor_rings_and_teacher_episode_"
                    "sidecars_not_serialized_v2"
                ),
                "perception_replay_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                    if self._student_collection_actor_cache_enabled()
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "actor_replay_observation_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                ),
                "teacher_episode_sidecar_semantics": (
                    TEACHER_EPISODE_SIDECAR_SEMANTICS
                    if self._teacher_episode_cache_enabled()
                    else "disabled"
                ),
                "perception_training_semantics": (
                    online_rollout_perception_semantics(self.cfg)
                    if str(self.cfg.perception_replay_mode)
                    == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "perception_prefill_warmup_semantics": (
                    PERCEPTION_PREFILL_DISABLED_SEMANTICS
                    if int(self.cfg.teacher_perception_warmup_steps) == 0
                    else (
                        ONLINE_STUDENT_ACTOR_WARMUP_SEMANTICS
                        if str(self.cfg.perception_replay_mode)
                        == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                        else PERCEPTION_PREFILL_WARMUP_SEMANTICS
                    )
                ),
                "teacher_prefill_semantics": FASTSAC_TEACHER_PREFILL_SEMANTICS,
                "object_geo_replay_semantics": OBJECT_GEO_REPLAY_SEMANTICS,
                "perception_initialization": copy.deepcopy(
                    self._perception_initialization
                ),
                "perception_object_geo_fingerprint": (
                    self._replay_object_geo_fingerprint
                ),
                "vecnorm_fingerprint": self._replay_vecnorm_fingerprint,
                "next_iter": int(self.env.current_iter) + 1,
            }
        )
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        if algorithm == TRAINING_ALGORITHM:
            raise ValueError(
                "FastSAC Actor replay and Teacher episode sidecars are fresh-only; "
                "same-stage resume "
                "is intentionally unsupported"
            )
        if algorithm is not None:
            raise ValueError(
                f"unsupported FastSAC Teacher-BC source algorithm={algorithm!r}"
            )

        # Reuse the rigorously validated raw PPO source loading in the replay
        # base. It transiently creates a TD3 target Actor, which is discarded
        # immediately and never participates in FastSAC behavior or learning.
        # PPOVEL's compatibility loader iterates every currently registered
        # child.  The dedicated SAC variance adapter cannot exist in an older
        # PPO source, so seed only that new child with its fresh initialization
        # while leaving every source-owned module subject to strict loading.
        actor_initialization = _student_actor_initialization(self.cfg)
        constructor_actor_state = getattr(
            self, "_fresh_student_actor_constructor_state", None
        )
        constructor_actor_parameter_ids = getattr(
            self, "_fresh_student_actor_constructor_parameter_ids", None
        )
        if (
            actor_initialization == FRESH_STUDENT_ACTOR_INITIALIZATION
            and (
                not isinstance(constructor_actor_state, Mapping)
                or not isinstance(constructor_actor_parameter_ids, tuple)
            )
        ):
            raise RuntimeError(
                "fresh Student Actor initialization lacks its constructor snapshot"
            )

        padded_source = dict(state_dict)
        padded_source["bc_dagger_sac_adapter"] = self.bc_dagger_sac_adapter.state_dict()
        # A raw PPO Teacher source never owns the Student SAC critic. Preserve
        # the freshly seeded configured critic explicitly instead of asking
        # PPOVEL's compatibility loader to interpret absent/C51 Q tensors.
        padded_source["qnet"] = self.qnet.state_dict()
        padded_source["qnet_target"] = self.qnet_target.state_dict()
        failed = DistributionalTD3TeacherBC.load_state_dict(self, padded_source, strict)
        if constructor_actor_state is not None:
            # Load tensors into the existing module rather than replacing it:
            # actor/std optimizer Parameter identities therefore remain valid.
            self.actor_adapt.load_state_dict(constructor_actor_state, strict=True)
            if tuple(id(parameter) for parameter in self.actor_adapt.parameters()) != (
                constructor_actor_parameter_ids
            ):
                raise RuntimeError(
                    "fresh Student Actor initialization replaced Parameter objects"
                )
            load_noise_scale = getattr(self.cfg, "load_noise_scale", None)
            if load_noise_scale is not None:
                actual_std = self._ppo_actor_std_parameter().detach()
                expected_std = torch.full_like(actual_std, float(load_noise_scale))
                if not torch.equal(actual_std, expected_std):
                    raise RuntimeError(
                        "fresh Student actor_std was not reset from load_noise_scale"
                    )
        self._actor_initialization = {
            "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
            "mode": actor_initialization,
            "teacher_actor_loaded": True,
            "actor_adapt_mean_loaded": (
                actor_initialization == TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
            ),
            "actor_adapt_mean_fresh": (
                actor_initialization == FRESH_STUDENT_ACTOR_INITIALIZATION
            ),
            "source_phase": state_dict.get("last_phase"),
            "source_iter": state_dict.get("last_iter"),
        }
        actor_adopt_path = _actor_adopt_checkpoint_path(self.cfg)
        if actor_adopt_path is not None:
            self._load_actor_adopt_checkpoint(
                actor_adopt_path,
                teacher_source_policy=state_dict,
            )
        self.__dict__.pop("_fresh_student_actor_constructor_state", None)
        self.__dict__.pop("_fresh_student_actor_constructor_parameter_ids", None)
        self.actor_target = None
        self.bc_dagger_sac_adapter.log_std.data.copy_(self._fastsac_initial_raw_log_std)
        self.log_alpha.data.fill_(math.log(float(self.cfg.sac_alpha_init)))
        self.actor_update_count = 0
        self.actor_std_update_count = 0
        self.critic_update_count = 0
        self.alpha_update_count = 0
        self.q_update_row_credit = 0.0
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
        self.q_rng.manual_seed(int(self.cfg.q_seed))
        self.sac_action_rng.manual_seed(int(self.cfg.q_seed) + 1)
        self.sac_rollout_rng.manual_seed(int(self.cfg.q_seed) + 2)
        self.teacher_prefill_action_rng.manual_seed(int(self.cfg.q_seed) + 4)
        self.teacher_perception_rng.manual_seed(int(self.cfg.q_seed) + 3)
        # Drop deterministic-only streams after the base loader is finished.
        if hasattr(self, "collector_exploration_rng"):
            del self.collector_exploration_rng
        if hasattr(self, "target_policy_rng"):
            del self.target_policy_rng
        hard_copy_(self.qnet, self.qnet_target)
        self.qnet_target.requires_grad_(False).eval()
        self.actor_adapt.requires_grad_(True).train()
        self._configure_training_actor_std()
        if self._uses_ppo_physical_gaussian():
            expected_std = float(self.cfg.load_noise_scale)
            actual_std = self._ppo_actor_std_parameter().detach()
            if not torch.equal(
                actual_std, torch.full_like(actual_std, expected_std)
            ):
                raise RuntimeError(
                    "PPOVEL fresh finetune did not initialize every physical "
                    "actor_std from load_noise_scale"
                )
            # In q-normalized mode the scalar PPO-compatible reset is only the
            # fresh source value.  Resolve it through the joint-wise envelope
            # before the first Student rollout; same-stage resume never enters
            # this source-loading branch and therefore restores its saved std.
            self._project_physical_actor_std_()
        return failed


__all__ = [
    "ACTION_CONTRACT_SEMANTICS",
    "ACTOR_ADOPT_CHECKPOINT_SEMANTICS",
    "ACTOR_BACKEND",
    "ACTOR_MEAN_OPTIMIZER_SEMANTICS",
    "CHECKPOINT_VERSION",
    "C51_Q_CRITIC_TYPE",
    "DISTRIBUTIONAL_Q_CRITIC_TYPE",
    "FRESH_STUDENT_ACTOR_INITIALIZATION",
    "FASTSAC_ACTION_PROJECTION_KEY",
    "FASTSAC_DAGGER_ENV_KEY",
    "FASTSAC_PREFILL_TEACHER_NOISE_KEY",
    "FASTSAC_PREFILL_TEACHER_PROJECTION_KEY",
    "FastSACPhysicalNormal",
    "NORMALIZED_TANH_ACTION_DISTRIBUTION",
    "PHYSICAL_STD_BOUND_MODES",
    "PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION",
    "PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND",
    "Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE",
    "Q_CRITIC_TYPES",
    "Q_TWIN_REDUCTION_MEAN",
    "Q_TWIN_REDUCTION_MIN",
    "Q_TWIN_REDUCTIONS",
    "SCALAR_Q_CRITIC_TYPE",
    "SCALAR_CRITIC_SEMANTICS",
    "SPRED_P_BC_SEMANTICS",
    "STUDENT_ACTOR_INITIALIZATION_MODES",
    "STUDENT_ACTOR_INITIALIZATION_SEMANTICS",
    "TEACHER_BC_STUDENT_ACTOR_INITIALIZATION",
    "UNIFORM_PHYSICAL_STD_BOUND_MODE",
    "PREVIOUS_CHECKPOINT_VERSION",
    "TRAINING_ALGORITHM",
    "DistributionalFastSACTeacherBC",
    "DistributionalFastSACTeacherBCConfig",
    "_DeterministicFastSACStudentEvalPolicy",
    "_DistributionalFastSACDaggerRolloutPolicy",
    "_fastsac_q_critic_type",
    "_fastsac_q_twin_reduction",
    "_reduce_fastsac_twin_target",
    "_spred_p_teacher_probability",
    "checkpoint_module_mismatches",
    "validate_actor_adopt_checkpoint_payload",
]
