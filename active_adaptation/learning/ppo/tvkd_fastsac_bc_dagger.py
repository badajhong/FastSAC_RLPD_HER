"""TVKD value-shaped FastSAC with Teacher-value bottleneck replay.

This entrypoint is intentionally layered on top of the repository's current
``DistributionalFastSACTeacherBC`` implementation.  It therefore preserves
the existing stochastic Student rollout, frozen successful-Teacher prefill,
independently configured Student/Teacher Q and Actor mixtures, PPOVEL-style
live Student perception training, collection-exact Student Actor inputs plus
current-EMA full-episode Teacher Actor inputs,
timeout-final-observation handling, twin C51 critics, and target-update cadence.

The two additions are deliberately narrow:

* the frozen PPO Teacher critic is reused as a potential function in the
  FastSAC C51 target; and
* failed mixed-control episodes use Student-executed Teacher-TD residuals to
  register bottleneck-aligned phases in the existing successful-Teacher replay
  curriculum.

The PPO critic consumes the same observation *fields* as the SAC critic, but
its internal ``CatTensors`` module sorts keys.  Consequently this module never
feeds the flattened SAC tensor directly into the PPO MLP.  It reconstructs a
keyed ``TensorDict`` first, preserving the exact PPO observation contract.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import CMD_KEY, DONE_KEY, OBS_KEY, OBS_PRIV_KEY, REWARD_KEY, TERM_KEY
from .fastsac_bc_dagger import (
    ACTOR_BACKEND,
    ACTOR_LEARNING_SEMANTICS as BASE_NORMALIZED_ACTOR_LEARNING_SEMANTICS,
    CHECKPOINT_VERSION as BASE_FASTSAC_CHECKPOINT_VERSION,
    NORMALIZED_TANH_ACTION_DISTRIBUTION,
    PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE,
    Q_EFFECTIVE_N_STEPS_KEY,
    Q_TWIN_REDUCTION_MEAN,
    SCALAR_Q_CRITIC_TYPE,
    SPRED_P_BC_SEMANTICS,
    TRAINING_ALGORITHM as BASE_FASTSAC_TRAINING_ALGORITHM,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
    _fastsac_action_distribution,
    _fastsac_actor_backend,
    _fastsac_q_critic_type,
    _fastsac_q_twin_reduction,
    _migrate_explicit_online_replay_capacities,
    _q_target_discount_factors,
    _reduce_fastsac_twin_target,
)
from .fastsac_vel import (
    FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_INPUT_SEMANTICS,
    _Stage1NStepAccumulator,
    _vaic_truncation_mask,
)
from .ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_REPLAY_MIN_STEP_COUNT,
    DAGGER_REPLAY_TEACHER_ACTIONS,
)
from .ppo_vel import DEPTH_KEY, OBJECT_GEO_KEY, OBJECT_KEY, VEL_CMD_KEY
from .ppo_vel import PPOVEL
from .td3_bc_dagger import (
    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS,
    FAILURE_PHASE_STUDENT_SOURCE_KEY,
    OBJECT_GEO_REPLAY_SEMANTICS,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS,
    PERCEPTION_REPLAY_SEMANTICS,
    NEXT_REFERENCE_PHASE_KEY,
    NEXT_Q_ACTUATOR_CONTEXT_KEY,
    REFERENCE_PHASE_KEY,
    REPLAY_COMMAND_FINISHED_KEY,
    REPLAY_MOTION_ID_KEY,
    REPLAY_TERMINATED_KEY,
    REPLAY_TIME_LIMIT_KEY,
    STUDENT_COLLECTION_NEXT_ACTOR_OBSERVATIONS_KEY,
    STUDENT_REPLAY_EPISODE_ID_KEY,
    STUDENT_REPLAY_EPISODE_STEP_KEY,
    TEACHER_EPISODE_SIDECAR_SEMANTICS,
    online_rollout_perception_semantics,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_COMMAND_FINISHED_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _PREFILL_TERMINATED_KEY,
    allocate_source_counts,
    _project_c51_probabilities,
)

TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v9"
V8_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v8"
V7_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v7"
V6_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v6"
V5_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v5"
V4_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v4"
V3_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v3"
PREVIOUS_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v2"
LEGACY_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v1"
CHECKPOINT_VERSION = 9
V8_CHECKPOINT_VERSION = 8
V7_CHECKPOINT_VERSION = 7
V6_CHECKPOINT_VERSION = 6
V5_CHECKPOINT_VERSION = 5
V4_CHECKPOINT_VERSION = 4
V3_CHECKPOINT_VERSION = 3
PREVIOUS_CHECKPOINT_VERSION = 2
LEGACY_CHECKPOINT_VERSION = 1
EXPECTED_ALGO_NAME = "tvkd_fastsac_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger."
    "TVKDDistributionalFastSACTeacherBC"
)
CRITIC_LEARNING_SEMANTICS = (
    "frozen_raw_scale_ppo_value_potential_shaped_soft_c51_target_v3"
)
TEACHER_RESIDUAL_CRITIC_LEARNING_SEMANTICS = (
    "pbrs_consistent_teacher_potential_residual_soft_c51_target_v1"
)
PHASE_FADED_CRITIC_LEARNING_SEMANTICS = (
    "frozen_raw_scale_phase_faded_ppo_value_potential_shaped_soft_c51_target_v2"
)
V8_CRITIC_LEARNING_SEMANTICS = (
    "frozen_raw_scale_ppo_value_potential_shaped_soft_c51_target_v1"
)
TEACHER_VALUE_RETURN_SEMANTICS = (
    "source_ppo_gae_gamma_replay_discount_terminated_mask_v1"
)
TEACHER_VALUE_BOUNDARY_SEMANTICS = (
    "shared_continuation_command_terminal_timeout_final_v4"
)
V8_TEACHER_VALUE_BOUNDARY_SEMANTICS = (
    "shared_continuation_command_current_self_loop_timeout_final_v3"
)
LEGACY_TEACHER_VALUE_BOUNDARY_SEMANTICS = (
    "shared_continuation_coefficient_next_state_bootstrap_v2"
)
TEACHER_VALUE_POTENTIAL_SEMANTICS = (
    "frozen_raw_teacher_value_shared_terminal_cut_v2"
)
Q_N_STEP_SEMANTICS = (
    "factual_one_step_expert_dagger_student_reward_endpoint_soft_bootstrap_v1"
)
PHASE_FADED_TEACHER_VALUE_POTENTIAL_SEMANTICS = (
    "reference_phase_faded_teacher_value_terminal_zero_v1"
)
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_plus_fixed_joint_valid_teacher_label_mean_bc_with_separate_low_lr_"
    "std_adam_rollout_scale_kl_cap_and_hard_bounds_v4"
)
Q_NORMALIZED_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_plus_fixed_joint_valid_teacher_label_mean_bc_with_separate_low_lr_"
    "std_adam_rollout_scale_kl_cap_and_jointwise_q_normalized_absolute_"
    "intersection_bounds_v1"
)
NORMALIZED_ACTOR_LEARNING_SEMANTICS = BASE_NORMALIZED_ACTOR_LEARNING_SEMANTICS
V5_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_sac_plus_fixed_joint_valid_teacher_label_bc_v2"
)


def _tvkd_actor_learning_semantics(config) -> str:
    distribution = _fastsac_action_distribution(config)
    if distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION:
        semantics = NORMALIZED_ACTOR_LEARNING_SEMANTICS
    elif distribution == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION:
        if str(
            getattr(config, "sac_physical_std_bound_mode", "uniform_physical")
        ) == Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE:
            semantics = Q_NORMALIZED_ACTOR_LEARNING_SEMANTICS
        else:
            semantics = ACTOR_LEARNING_SEMANTICS
    else:
        raise ValueError(f"unsupported FastSAC action distribution {distribution!r}")
    if _fastsac_q_twin_reduction(config) == Q_TWIN_REDUCTION_MEAN:
        if "twin_min" not in semantics:
            raise RuntimeError("TVKD Actor semantics lack the twin-min marker")
        semantics = semantics.replace("twin_min", "twin_mean", 1)
    if getattr(config, "use_q_filtered_bc", False):
        return f"{semantics}_with_{SPRED_P_BC_SEMANTICS}"
    return semantics


def _tvkd_critic_learning_semantics(config) -> str:
    """Return the exact target contract selected by the potential ablation."""
    if bool(getattr(config, "use_teacher_residual_critic", False)):
        semantics = TEACHER_RESIDUAL_CRITIC_LEARNING_SEMANTICS
    elif bool(getattr(config, "use_phase_faded_teacher_potential", False)):
        semantics = PHASE_FADED_CRITIC_LEARNING_SEMANTICS
    else:
        semantics = CRITIC_LEARNING_SEMANTICS
    scalar = _fastsac_q_critic_type(config) == SCALAR_Q_CRITIC_TYPE
    mean_reduction = (
        _fastsac_q_twin_reduction(config) == Q_TWIN_REDUCTION_MEAN
    )
    if scalar:
        replacement = (
            "soft_mean_twin_scalar_q_mse_target"
            if mean_reduction
            else "soft_clipped_twin_scalar_q_mse_target"
        )
        semantics = semantics.replace("soft_c51_target", replacement)
    elif mean_reduction:
        semantics = semantics.replace(
            "soft_c51_target", "soft_mean_twin_c51_target"
        )
    return semantics


def _tvkd_teacher_value_potential_semantics(config) -> str:
    """Return metadata for the configured state-potential function."""
    if bool(getattr(config, "use_phase_faded_teacher_potential", False)):
        return PHASE_FADED_TEACHER_VALUE_POTENTIAL_SEMANTICS
    return TEACHER_VALUE_POTENTIAL_SEMANTICS


BOTTLENECK_SELECTION_MODES = ("first", "last", "deepest")
BOTTLENECK_LOCATION_SEMANTICS = (
    "frozen_teacher_raw_td_full_causal_mean_sustained_student_replay_valid_v2"
)
VERIFIED_HISTOGRAM_SEMANTICS = (
    "verified_teacher_value_pre_onset_student_motion_phase_v2"
)
V5_FRESH_RING_RESUME_SEMANTICS = "clear_online_rings_and_partial_row_credit_v1"
FRESH_RING_RESUME_SEMANTICS = (
    "clear_online_rings_teacher_episode_sidecars_and_partial_row_credit_v2"
)
V5_REPLAY_RESUME_SEMANTICS = (
    "model_optimizer_policy_rng_verified_bottleneck_resume_with_"
    "fresh_raw_ring_and_row_credit_rebuild_v5"
)
REPLAY_RESUME_SEMANTICS = (
    "model_optimizer_policy_rng_verified_bottleneck_resume_with_fresh_exact_"
    "actor_rings_teacher_episode_sidecars_and_row_credit_rebuild_v6"
)
UNBOUND_CONTRACT_FINGERPRINT = "unbound_direct_construction"

# Replay cache: frozen PPO Teacher values precomputed once per transition and
# stored alongside the replay row.  Q target updates read from these fields
# instead of re-running the frozen Teacher critic on every batch.
REPLAY_TEACHER_V_CURRENT_KEY = "replay_teacher_v_current"
REPLAY_TEACHER_V_NEXT_KEY = "replay_teacher_v_next"
TEACHER_VALUE_CACHE_SEMANTICS = "precomputed_frozen_ppo_float32_current_next_v1"

LEGACY_ADAPTIVE_BC_CONFIG_FIELDS = (
    "use_adaptive_student_bc",
    "student_bc_lambda_min",
    "student_bc_lambda_max",
    "student_bc_margin",
    "student_bc_temperature",
    "student_bc_scale_ema_decay",
    "student_bc_risk_ema_decay",
    "student_bc_scheduler_warmup_updates",
    "student_bc_scheduler_min_samples",
    "student_bc_scheduler_eps",
)

SOURCE_STUDENT = 0
SOURCE_UNIFORM_TEACHER = 1
SOURCE_FAILURE_TEACHER = 2
EXPECTED_ACTOR_IN_KEYS = (
    CMD_KEY,
    OBS_KEY,
    OBJECT_KEY,
    OBS_PRIV_KEY,
    OBJECT_GEO_KEY,
    VEL_CMD_KEY,
    DEPTH_KEY,
)


def _finite_scalar(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return value


def _lambda_bc_fork_override_active(
    config,
    state: Mapping,
    saved_lambda_bc,
) -> bool:
    """Validate an explicit pre-Actor fork with a different BC coefficient."""
    saved = _finite_scalar("checkpoint lambda_bc", saved_lambda_bc)
    runtime = _finite_scalar("runtime lambda_bc", config.lambda_bc)
    if saved < 0.0 or runtime < 0.0:
        raise ValueError("lambda_bc must be non-negative")
    enabled = getattr(config, "resume_allow_lambda_bc_override", False)
    if not isinstance(enabled, bool):
        raise ValueError("resume_allow_lambda_bc_override must be boolean")
    changed = not math.isclose(saved, runtime, rel_tol=0.0, abs_tol=1e-12)
    if not changed:
        return False
    if not enabled:
        raise ValueError("TVKD resume fixed BC coefficient mismatch")
    if (
        state.get("training_algorithm"),
        state.get("checkpoint_version"),
    ) != (TRAINING_ALGORITHM, CHECKPOINT_VERSION):
        raise ValueError(
            "lambda_bc fork resume requires a current TVKD checkpoint"
        )

    # Changing the Actor objective is safe only at the warmup boundary, before
    # any Actor/std/temperature update or associated Adam moment exists.
    pristine_counters = (
        "actor_update_count",
        "actor_std_update_count",
        "alpha_update_count",
        "sac_actor_update_count",
        "sac_alpha_update_count",
    )
    dirty_counters = {
        name: state.get(name)
        for name in pristine_counters
        if state.get(name) != 0
    }
    if dirty_counters:
        raise ValueError(
            "lambda_bc fork resume requires a pre-Actor checkpoint; nonzero "
            f"counters={dirty_counters}"
        )
    optimizers = state.get("optimizer_resume_state")
    if not isinstance(optimizers, Mapping):
        raise ValueError("lambda_bc fork resume lacks optimizer state")
    dirty_optimizers = []
    for name in ("actor_optimizer", "actor_std_optimizer", "alpha_optimizer"):
        optimizer = optimizers.get(name)
        if optimizer is None:
            continue
        if not isinstance(optimizer, Mapping):
            raise ValueError(
                f"lambda_bc fork resume has invalid {name} state"
            )
        optimizer_state = optimizer.get("state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError(
                f"lambda_bc fork resume has invalid {name} moments"
            )
        if optimizer_state:
            dirty_optimizers.append(name)
    if dirty_optimizers:
        raise ValueError(
            "lambda_bc fork resume requires pristine Actor/std/alpha optimizer "
            f"moments; nonempty={dirty_optimizers}"
        )
    return True


def _required_contract_fingerprint(name: str, value) -> str:
    """Reject placeholder metadata before it can enter an exact-resume seam."""
    if (
        not isinstance(value, str)
        or value == UNBOUND_CONTRACT_FINGERPRINT
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a bound lowercase SHA-256 fingerprint")
    return value


_REPLAY_MIX_SOURCES = (
    "uniform_student",
    "failure_student",
    "uniform_teacher",
    "failure_teacher",
)


def _legacy_global_replay_mix(
    teacher_fraction, teacher_focus, student_focus
) -> dict[str, float]:
    values = {
        "teacher fraction": teacher_fraction,
        "Teacher focus fraction": teacher_focus,
        "Student focus fraction": student_focus,
    }
    checked = {}
    for name, value in values.items():
        value = _finite_scalar(name, value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        checked[name] = value
    teacher = checked["teacher fraction"]
    student = 1.0 - teacher
    return {
        "uniform_student": student * (1.0 - checked["Student focus fraction"]),
        "failure_student": student * checked["Student focus fraction"],
        "uniform_teacher": teacher * (1.0 - checked["Teacher focus fraction"]),
        "failure_teacher": teacher * checked["Teacher focus fraction"],
    }


def _checkpoint_replay_mix(cfg) -> dict[str, dict[str, float]]:
    return {
        purpose: {
            source: float(getattr(cfg, f"{purpose}_{source}_fraction"))
            for source in _REPLAY_MIX_SOURCES
        }
        for purpose in ("q", "actor", "perception")
    }


def _same_verified_histogram_state(left: Mapping, right: Mapping) -> bool:
    """Compare the verified histogram and compatibility alias tensor-exactly."""
    scalar_fields = (
        "semantics",
        "episode_count",
        "anchor_count",
        "uniform_fallback_rows",
        "focused_rows",
    )
    if any(left.get(name) != right.get(name) for name in scalar_fields):
        return False
    left_histogram = left.get("histogram")
    right_histogram = right.get("histogram")
    if not (
        torch.is_tensor(left_histogram)
        and torch.is_tensor(right_histogram)
        and torch.equal(left_histogram, right_histogram)
    ):
        return False
    left_motion = left.get("motion_histograms")
    right_motion = right.get("motion_histograms")
    if not isinstance(left_motion, Mapping) or not isinstance(right_motion, Mapping):
        return False
    if set(left_motion) != set(right_motion):
        return False
    return all(
        torch.is_tensor(left_motion[motion_id])
        and torch.is_tensor(right_motion[motion_id])
        and torch.equal(left_motion[motion_id], right_motion[motion_id])
        for motion_id in left_motion
    )


def _install_v3_replay_migration(
    cfg,
    backend: Mapping,
    *,
    student_focus_default=None,
) -> None:
    """Migrate old Q/Actor mixes while adopting live Student perception."""
    teacher_focus = backend.get("failure_phase_teacher_fraction")
    student_focus = backend.get("failure_phase_student_fraction", student_focus_default)
    q_mix = _legacy_global_replay_mix(
        backend.get("q_teacher_replay_ratio"), teacher_focus, student_focus
    )
    actor_mix = _legacy_global_replay_mix(
        backend.get("teacher_actor_replay_fraction"), teacher_focus, student_focus
    )
    legacy_perception_teacher = _finite_scalar(
        "Teacher perception fraction",
        backend.get("teacher_perception_replay_fraction"),
    )
    if not 0.0 <= legacy_perception_teacher <= 1.0:
        raise ValueError("Teacher perception fraction must lie in [0, 1]")
    teacher_focus_value = _finite_scalar("Teacher focus fraction", teacher_focus)
    perception_mix = {
        "uniform_student": 1.0,
        "failure_student": 0.0,
        "uniform_teacher": 0.0,
        "failure_teacher": 0.0,
    }
    for purpose, mix in (
        ("q", q_mix),
        ("actor", actor_mix),
        ("perception", perception_mix),
    ):
        allocate_source_counts(1, mix)
        for source, value in mix.items():
            setattr(cfg, f"{purpose}_{source}_fraction", float(value))
    cadence = backend.get("sac_alpha_update_cadence", "critic")
    if cadence not in {"actor", "critic"}:
        raise ValueError("v3 checkpoint alpha cadence is invalid")
    cfg.sac_alpha_update_cadence = cadence
    cfg.q_teacher_replay_ratio = float(
        _finite_scalar(
            "checkpoint Q Teacher fraction",
            backend.get("q_teacher_replay_ratio"),
        )
    )
    cfg.teacher_actor_replay_fraction = float(
        _finite_scalar(
            "checkpoint Actor Teacher fraction",
            backend.get("teacher_actor_replay_fraction"),
        )
    )
    cfg.teacher_perception_replay_fraction = 0.0
    cfg.teacher_perception_warmup_steps = 0
    cfg.failure_phase_teacher_fraction = teacher_focus_value
    if hasattr(cfg, "failure_phase_student_fraction"):
        cfg.failure_phase_student_fraction = float(student_focus)
    cfg.perception_replay_mode = ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    cfg.bottleneck_fallback_mode = "none"
    cfg.bottleneck_include_unsuccessful_timeouts = False
    cfg.max_teacher_phase_match_distance = None


def _validate_tvkd_algorithm_config(cfg) -> None:
    """Validate controls shared by direct construction and the Hydra CLI."""
    if getattr(cfg, "sac_alpha_update_cadence", None) not in {"actor", "critic"}:
        raise ValueError("sac_alpha_update_cadence must be 'actor' or 'critic'")
    for name in ("q_n_step", "q_teacher_n_step"):
        value = getattr(cfg, name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        if int(value) != 1:
            raise ValueError(
                f"{name} is locked to 1 for factual one-step Expert/DAgger/"
                "Student SAC replay"
            )
    for name in (
        "use_tvkd_value_shaping",
        "use_teacher_residual_critic",
        "use_phase_faded_teacher_potential",
        "use_teacher_value_bottleneck_replay",
        "resume_allow_lambda_bc_override",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be boolean")
    if bool(cfg.use_teacher_residual_critic) and not bool(
        cfg.use_tvkd_value_shaping
    ):
        raise ValueError(
            "use_teacher_residual_critic requires use_tvkd_value_shaping=true"
        )
    include_timeouts = getattr(cfg, "bottleneck_include_unsuccessful_timeouts", False)
    if not isinstance(include_timeouts, bool):
        raise ValueError("bottleneck_include_unsuccessful_timeouts must be boolean")
    selection_mode = str(getattr(cfg, "bottleneck_selection_mode", "first"))
    if selection_mode not in BOTTLENECK_SELECTION_MODES:
        raise ValueError(
            "bottleneck_selection_mode must be one of "
            f"{sorted(BOTTLENECK_SELECTION_MODES)}"
        )
    fallback_mode = str(getattr(cfg, "bottleneck_fallback_mode", "none"))
    if fallback_mode not in {"none", "value_argmin"}:
        raise ValueError("bottleneck_fallback_mode must be 'none' or 'value_argmin'")

    for purpose in ("q", "actor", "perception"):
        fractions = {
            source: getattr(cfg, f"{purpose}_{source}_fraction")
            for source in (
                "uniform_student",
                "failure_student",
                "uniform_teacher",
                "failure_teacher",
            )
        }
        # The shared allocator is also the canonical strict fraction validator.
        allocate_source_counts(1, fractions)

    tvkd_lambda = _finite_scalar("tvkd_lambda", getattr(cfg, "tvkd_lambda"))
    if tvkd_lambda < 0.0:
        raise ValueError("tvkd_lambda must be non-negative")
    potential_clip = getattr(cfg, "tvkd_potential_clip")
    if potential_clip is not None:
        _finite_scalar("tvkd_potential_clip", potential_clip, positive=True)
    cache_validate_fraction = _finite_scalar(
        "teacher_value_cache_validate_fraction",
        getattr(cfg, "teacher_value_cache_validate_fraction", 0.0),
    )
    if not 0.0 <= cache_validate_fraction <= 1.0:
        raise ValueError(
            "teacher_value_cache_validate_fraction must lie in [0, 1]"
        )

    _finite_scalar(
        "bottleneck_threshold",
        getattr(cfg, "bottleneck_threshold"),
        positive=True,
    )
    for name in ("bottleneck_smoothing_window", "bottleneck_min_consecutive"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    exclusion = getattr(cfg, "bottleneck_terminal_exclusion_steps")
    if isinstance(exclusion, bool) or not isinstance(exclusion, int) or exclusion < 0:
        raise ValueError(
            "bottleneck_terminal_exclusion_steps must be a non-negative integer"
        )
    if int(cfg.failure_phase_samples_per_failure) > int(
        cfg.failure_phase_lookback_steps
    ):
        raise ValueError(
            "TVKD failure_phase_samples_per_failure cannot exceed the strictly "
            "pre-onset failure_phase_lookback_steps interval"
        )
    max_distance = getattr(cfg, "max_teacher_phase_match_distance", None)
    if max_distance is not None:
        max_distance = _finite_scalar("max_teacher_phase_match_distance", max_distance)
        if max_distance < 0.0:
            raise ValueError("max_teacher_phase_match_distance must be non-negative")
    if getattr(cfg, "teacher_value_return_semantics", None) != (
        TEACHER_VALUE_RETURN_SEMANTICS
    ):
        raise ValueError("unsupported Teacher value return semantics")
    boundary_semantics = getattr(cfg, "teacher_value_boundary_semantics", None)
    if boundary_semantics in {
        LEGACY_TEACHER_VALUE_BOUNDARY_SEMANTICS,
        V8_TEACHER_VALUE_BOUNDARY_SEMANTICS,
    }:
        # Older checkpoints embed these values in the Hydra config used to
        # construct an inference policy.  Runtime behavior is source-defined,
        # so canonicalizing the inert metadata permits replayless evaluation.
        # The untouched checkpoint state still carries its old contract and the
        # exact training-resume loader below rejects incompatible v8 state.
        cfg.teacher_value_boundary_semantics = TEACHER_VALUE_BOUNDARY_SEMANTICS
    elif boundary_semantics != TEACHER_VALUE_BOUNDARY_SEMANTICS:
        raise ValueError("unsupported Teacher value boundary semantics")
    potential_semantics = getattr(cfg, "teacher_value_potential_semantics", None)
    if potential_semantics not in {
        TEACHER_VALUE_POTENTIAL_SEMANTICS,
        PHASE_FADED_TEACHER_VALUE_POTENTIAL_SEMANTICS,
    }:
        raise ValueError("unsupported Teacher value potential semantics")
    # This is source-defined metadata, not an independent behavior knob.  Keep
    # old phase-faded Hydra/checkpoint configs evaluable while making a single
    # explicit ablation flag select both the arithmetic and its saved contract.
    cfg.teacher_value_potential_semantics = (
        _tvkd_teacher_value_potential_semantics(cfg)
    )


@dataclass
class TVKDDistributionalFastSACTeacherBCConfig(DistributionalFastSACTeacherBCConfig):
    """Hydra surface for TVKD shaping and value-bottleneck replay."""

    _target_: str = EXPECTED_ALGO_TARGET
    name: str = EXPECTED_ALGO_NAME
    # Fresh v5 matches the parent FastSAC optimizer timescale. A v3 migration
    # explicitly replaces this with the checkpoint's saved cadence.
    sac_alpha_update_cadence: str = "actor"

    # Resolved canonical mixtures. The TVKD CLI may derive Q/Actor values from
    # explicit total-Teacher and conditional failure-focus overrides before
    # policy construction; direct construction supplies these fields itself.
    q_uniform_student_fraction: float = 0.35
    q_failure_student_fraction: float = 0.15
    q_uniform_teacher_fraction: float = 0.35
    q_failure_teacher_fraction: float = 0.15
    actor_uniform_student_fraction: float = 0.35
    actor_failure_student_fraction: float = 0.15
    actor_uniform_teacher_fraction: float = 0.35
    actor_failure_teacher_fraction: float = 0.15
    # Perception is not a replay objective. These canonical values make the
    # checkpoint/config contract explicit while PPOVEL.train_adapt consumes
    # the complete live recurrent Student rollout.
    perception_uniform_student_fraction: float = 1.0
    perception_failure_student_fraction: float = 0.0
    perception_uniform_teacher_fraction: float = 0.0
    perception_failure_teacher_fraction: float = 0.0
    perception_replay_mode: str = ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE

    # Every source uses the standard one-step SAC Bellman contract.  This is
    # especially important for the online DAgger partition, where Student and
    # Teacher execute alternate actions: a multi-step target would otherwise
    # credit a Student action with a later Teacher rescue.  Delayed actuation
    # remains observable through the Critic state/action-history context and is
    # propagated by ordinary one-step bootstrapping.
    q_n_step: int = 1
    q_teacher_n_step: int = 1

    use_tvkd_value_shaping: bool = True
    # Exact PBRS reparameterization.  When enabled, C51 predicts the return
    # residual after subtracting the configured frozen Teacher potential.  The
    # represented shaped Q is recovered by adding the detached state-only
    # baseline (1 - tvkd_lambda) * Phi(s), so Actor gradients and same-state
    # action rankings are unchanged.  A matched 512-rollout, four-seed paired
    # GPU A/B promoted this parameterization for fresh TVKD training.  Legacy
    # checkpoints still migrate this field to false so their saved critic
    # coordinates are never reinterpreted during resume.
    use_teacher_residual_critic: bool = True
    # Explicit same-stage fork for a checkpoint captured before the first
    # Actor/std/alpha update. It permits changing only the fixed BC
    # coefficient while retaining trained perception and Student Critic state.
    # Ordinary continuation remains strict by default.
    resume_allow_lambda_bc_override: bool = False
    # Raw Phi=V_T is the default exact potential. Command completion already
    # cuts the absorbing successor with c_t=0, so the current decision state's
    # value must not be faded for terminal correctness. Phase fading remains an
    # explicit variance-reduction ablation.
    use_phase_faded_teacher_potential: bool = False
    tvkd_lambda: float = 0.25
    tvkd_potential_clip: float | None = None

    use_teacher_value_bottleneck_replay: bool = True
    # Raw return-unit threshold: m_t < -bottleneck_threshold. This is not
    # normalized by a running statistic.
    bottleneck_threshold: float = 0.05
    bottleneck_smoothing_window: int = 5
    bottleneck_min_consecutive: int = 3
    bottleneck_terminal_exclusion_steps: int = 5
    bottleneck_fallback_mode: str = "none"
    # Which qualifying onset becomes the prevention anchor. ``first`` keeps the
    # historical earliest-crossing rule; ``last``/``deepest`` move the window
    # toward the failure, which is what a recovery curriculum actually wants.
    bottleneck_selection_mode: str = "first"
    bottleneck_include_unsuccessful_timeouts: bool = False
    # TVKD interprets this inherited interval as strictly before t_onset.
    failure_phase_lookback_steps: int = 10
    failure_phase_samples_per_failure: int = 10
    max_teacher_phase_match_distance: float | None = None
    teacher_value_return_semantics: str = TEACHER_VALUE_RETURN_SEMANTICS
    teacher_value_boundary_semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS
    teacher_value_potential_semantics: str = TEACHER_VALUE_POTENTIAL_SEMANTICS
    teacher_value_reward_group_fingerprint: str = UNBOUND_CONTRACT_FINGERPRINT
    replay_task_fingerprint: str = UNBOUND_CONTRACT_FINGERPRINT
    # Set > 0 to recompute live Teacher values for a random fraction of each
    # Q batch and compare against the cached values.  Requires max abs error
    # <= 1e-6.  Production value is 0.0 (disabled).
    teacher_value_cache_validate_fraction: float = 0.0
    # The fresh CLI uses this conditional-within-Student fraction to derive
    # canonical Q/Actor mixes. Checkpoints still serialize the resolved mix.
    failure_phase_student_fraction: float = 0.3


ConfigStore.instance().store(
    "tvkd_fastsac_bc_dagger_finetune",
    node=TVKDDistributionalFastSACTeacherBCConfig(in_keys=EXPECTED_ACTOR_IN_KEYS),
    group="algo",
)


class FrozenTeacherValueWrapper:
    """Read the checkpoint PPO value in raw summed-reward scale.

    ``teacher_critic_obs`` is the already-VecNorm-normalized flattened SAC
    critic observation.  Its fields are split and restored by name so the PPO
    critic's own key ordering remains authoritative.

    This is deliberately a plain Python object, rather than an ``nn.Module``:
    registering a second alias to the Teacher actor/value modules would alter
    the established PPO/FastSAC checkpoint topology.
    """

    def __init__(
        self,
        teacher_actor: nn.Module,
        teacher_value: nn.Module,
        value_normalizer: nn.Module,
        critic_keys: tuple[str, ...],
        critic_widths: tuple[int, ...],
    ) -> None:
        if len(critic_keys) != len(critic_widths) or not critic_keys:
            raise ValueError("Teacher value critic keys/widths must align")
        if any(int(width) < 1 for width in critic_widths):
            raise ValueError("Teacher value critic widths must be positive")
        self.teacher_actor = teacher_actor
        self.teacher_value = teacher_value
        self.value_normalizer = value_normalizer
        self.critic_keys = tuple(critic_keys)
        self.critic_widths = tuple(int(width) for width in critic_widths)
        self.freeze()

    def freeze(self) -> None:
        for module in (
            self.teacher_actor,
            self.teacher_value,
            self.value_normalizer,
        ):
            module.requires_grad_(False)
            module.eval()
            for parameter in module.parameters():
                parameter.grad = None

    @torch.inference_mode()
    def get_frozen_teacher_value(
        self, teacher_critic_obs: torch.Tensor
    ) -> torch.Tensor:
        """Return frozen PPO value in raw summed-reward scale, shape ``[B]``."""
        if teacher_critic_obs.ndim < 2:
            raise ValueError("Teacher critic observation must have shape [..., D]")
        expected_width = sum(self.critic_widths)
        if teacher_critic_obs.shape[-1] != expected_width:
            raise ValueError(
                "Teacher critic observation width mismatch: "
                f"expected {expected_width}, got {teacher_critic_obs.shape[-1]}"
            )
        if not torch.isfinite(teacher_critic_obs).all():
            raise RuntimeError("Teacher critic observation contains NaN/Inf")

        # Teacher-value arithmetic is kept in float32 even when the surrounding
        # SAC update uses autocast/mixed precision.
        observations = teacher_critic_obs.detach().to(dtype=torch.float32)
        chunks = observations.split(self.critic_widths, dim=-1)
        value_td = TensorDict(
            dict(zip(self.critic_keys, chunks)),
            batch_size=observations.shape[:-1],
            device=observations.device,
        )
        device_type = observations.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            value_groups = self.teacher_value(value_td)["state_value"].float()
            raw_value_groups = self.value_normalizer.denormalize(value_groups).float()
            teacher_value = raw_value_groups.sum(dim=-1)
        if teacher_value.shape != observations.shape[:-1]:
            raise RuntimeError(
                "Frozen Teacher value has the wrong shape: "
                f"expected {tuple(observations.shape[:-1])}, "
                f"got {tuple(teacher_value.shape)}"
            )
        if not torch.isfinite(teacher_value).all():
            raise RuntimeError("Frozen Teacher value contains NaN/Inf")
        return teacher_value.detach()


@dataclass(frozen=True)
class TeacherValueBottleneck:
    """One raw-residual onset and its confirmation diagnostics."""

    index: int
    confirmation_index: int
    phase: float
    score: float
    raw_teacher_td_residual: float
    smoothed_teacher_td_residual: float
    student_candidate_count: int
    threshold_detected: bool
    used_fallback: bool
    selection_origin: str


class TeacherValueBottleneckDetector:
    """Find the earliest sustained raw Teacher-residual degradation.

    A moving-average endpoint is eligible only after a *full* causal window of
    consecutive Student-executed, replay-valid transitions. Teacher rows and
    reset burn-in rows break the window. Physical terminal rows and the
    configured preterminal tail are excluded before smoothing. The threshold
    is expressed directly in the frozen PPO return units; there is no adaptive
    normalization state.
    """

    def __init__(
        self,
        threshold: float,
        smoothing_window: int,
        min_consecutive: int,
        terminal_exclusion_steps: int,
        selection_mode: str = "first",
    ) -> None:
        self.threshold = _finite_scalar("threshold", threshold, positive=True)
        for name, value in (
            ("smoothing_window", smoothing_window),
            ("min_consecutive", min_consecutive),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(terminal_exclusion_steps, bool)
            or not isinstance(terminal_exclusion_steps, int)
            or terminal_exclusion_steps < 0
        ):
            raise ValueError("terminal_exclusion_steps must be non-negative")
        if selection_mode not in BOTTLENECK_SELECTION_MODES:
            raise ValueError(
                "bottleneck_selection_mode must be one of "
                f"{sorted(BOTTLENECK_SELECTION_MODES)}"
            )
        self.smoothing_window = int(smoothing_window)
        self.min_consecutive = int(min_consecutive)
        self.terminal_exclusion_steps = int(terminal_exclusion_steps)
        self.selection_mode = str(selection_mode)
        self.last_diagnostics = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics() -> dict[str, float | bool]:
        return {
            "student_transition_count": 0.0,
            "replay_valid_student_count": 0.0,
            "student_candidate_count": 0.0,
            "threshold_detected": False,
            "used_fallback": False,
            "no_candidate": True,
        }

    @torch.no_grad()
    def detect(
        self,
        teacher_td_residual: torch.Tensor,
        source_id: torch.Tensor,
        reference_phase: torch.Tensor,
        true_terminal: torch.Tensor,
        timeout: torch.Tensor,
        replay_valid: torch.Tensor,
        *,
        fallback_mode: str = "none",
    ) -> TeacherValueBottleneck | None:
        if fallback_mode not in {"none", "value_argmin"}:
            raise ValueError(
                "bottleneck fallback mode must be 'none' or 'value_argmin'"
            )
        residual = torch.as_tensor(teacher_td_residual).detach().float().reshape(-1)
        source = torch.as_tensor(source_id).detach().long().reshape(-1)
        phase = torch.as_tensor(reference_phase).detach().float().reshape(-1)
        terminal = torch.as_tensor(true_terminal).detach().bool().reshape(-1)
        timeout = torch.as_tensor(timeout).detach().bool().reshape(-1)
        replay_valid = torch.as_tensor(replay_valid).detach().bool().reshape(-1)
        size = residual.numel()
        if not all(
            value.numel() == size
            for value in (source, phase, terminal, timeout, replay_valid)
        ):
            raise ValueError("Bottleneck episode tensors must have identical length")
        if size == 0:
            self.last_diagnostics = self._empty_diagnostics()
            return None
        if not torch.isfinite(phase).all():
            raise RuntimeError("Bottleneck reference phase contains NaN/Inf")
        if bool((terminal & timeout).any()):
            raise ValueError("A transition cannot be both true-terminal and timeout")

        student = source == SOURCE_STUDENT
        student_count = int(student.sum().item())
        if student_count == 0:
            self.last_diagnostics = self._empty_diagnostics()
            return None
        if not torch.isfinite(residual[student]).all():
            raise RuntimeError("Student Teacher TD residual contains NaN/Inf")
        eligible = student & replay_valid & ~terminal & ~timeout
        boundary_indices = (terminal | timeout).nonzero(as_tuple=False).squeeze(-1)
        if boundary_indices.numel():
            final_boundary = int(boundary_indices[-1].item())
            cutoff = final_boundary - self.terminal_exclusion_steps
            eligible &= torch.arange(size, device=eligible.device) < cutoff

        # A causal average exists only after W consecutive eligible rows.
        smoothed = torch.full_like(residual, float("nan"))
        run: list[torch.Tensor] = []
        for index in range(size):
            if not bool(eligible[index]):
                run.clear()
                continue
            run.append(residual[index])
            if len(run) > self.smoothing_window:
                del run[0]
            if len(run) == self.smoothing_window:
                smoothed[index] = torch.stack(run).mean()

        candidate = eligible & torch.isfinite(smoothed)
        candidate_count = int(candidate.sum().item())

        diagnostics: dict[str, float | bool] = {
            "student_transition_count": float(student_count),
            "replay_valid_student_count": float((student & replay_valid).sum().item()),
            "student_candidate_count": float(candidate_count),
            "threshold_detected": False,
            "used_fallback": False,
            "no_candidate": candidate_count == 0,
        }

        # Every t whose K-window lies entirely below the threshold.  ``first``
        # stops at the earliest one and is the historical behavior; the other
        # modes need the complete set, so they keep scanning.
        onsets: list[int] = []
        run_length = 0
        for index in range(size):
            below = bool(candidate[index] and smoothed[index] < -self.threshold)
            if not below:
                run_length = 0
                continue
            run_length += 1
            if run_length >= self.min_consecutive:
                onsets.append(index - self.min_consecutive + 1)
                if self.selection_mode == "first":
                    break

        selected_index: int | None = None
        if onsets:
            diagnostics["threshold_detected"] = True
            if self.selection_mode == "first":
                selected_index = onsets[0]
            elif self.selection_mode == "last":
                selected_index = onsets[-1]
            else:
                # ``deepest``: the onset whose own smoothed residual is lowest.
                # Ties resolve to the latest such onset, because among equally
                # deep candidates the later one leaves less already-doomed
                # trajectory inside the prevention window.
                best = None
                for onset in onsets:
                    value = float(smoothed[onset].item())
                    if best is None or value <= best[0]:
                        best = (value, onset)
                selected_index = best[1]

        used_fallback = False
        if selected_index is None:
            if fallback_mode == "none":
                self.last_diagnostics = diagnostics
                return None
            fallback_mask = candidate
            if not bool(fallback_mask.any()):
                self.last_diagnostics = diagnostics
                return None
            fallback_indices = fallback_mask.nonzero(as_tuple=False).squeeze(-1)
            fallback_values = smoothed.index_select(0, fallback_indices)
            selected_index = int(fallback_indices[fallback_values.argmin()].item())
            used_fallback = True
            diagnostics["used_fallback"] = True

        selected_smoothed = float(smoothed[selected_index].item())
        result = TeacherValueBottleneck(
            index=selected_index,
            confirmation_index=(
                selected_index + self.min_consecutive - 1
                if not used_fallback
                else selected_index
            ),
            phase=float(phase[selected_index].item()),
            score=max(0.0, -selected_smoothed),
            raw_teacher_td_residual=float(residual[selected_index].item()),
            smoothed_teacher_td_residual=selected_smoothed,
            student_candidate_count=candidate_count,
            threshold_detected=bool(diagnostics["threshold_detected"]),
            used_fallback=used_fallback,
            selection_origin="value_argmin" if used_fallback else "threshold",
        )
        self.last_diagnostics = diagnostics
        return result

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError("Bottleneck detector state must be a mapping")
        if state_dict:
            raise ValueError("Raw bottleneck detector state must be empty")
        self.last_diagnostics = self._empty_diagnostics()


@dataclass(frozen=True)
class TeacherValueTerms:
    teacher_v: torch.Tensor
    teacher_v_next: torch.Tensor
    teacher_potential: torch.Tensor
    teacher_potential_next: torch.Tensor
    potential_delta: torch.Tensor
    shaped_reward: torch.Tensor
    teacher_td_residual: torch.Tensor


@torch.no_grad()
def continuation_bootstrap_mask(
    *,
    dones: torch.Tensor,
    truncations: torch.Tensor,
) -> torch.Tensor:
    """Return the rows whose boundary continues into ``s_{t+1}``.

    This is the single truth table TVKD is allowed to hold about boundaries:
    ordinary transitions and pure time-limit truncations continue, while
    physical termination and command completion cut.  A pure time limit uses
    its captured pre-reset final state.  Everything that needs a boundary
    decision -- the frozen-Teacher potential, the
    soft-Bellman bootstrap, and the state gathering that feeds them -- derives
    from here, so the shaping and bootstrap masks cannot drift apart.
    """
    done = torch.as_tensor(dones).detach().reshape(-1).bool()
    truncated = (
        torch.as_tensor(truncations)
        .detach()
        .to(device=done.device)
        .reshape(-1)
        .bool()
    )
    if truncated.shape != done.shape:
        raise ValueError("Continuation mask tensors must have identical shape")
    return truncated | ~done


@torch.no_grad()
def replay_truncation_mask(
    *,
    terminated: torch.Tensor,
    command_finished: torch.Tensor,
    time_limit: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the replay ``truncations`` field from explicit boundary causes.

    Mirrors ``_vaic_truncation_mask``: only a pure episode time limit truncates,
    because command completion and physical termination each end the finite
    return.  Used to cross-check the stored field, and by offline diagnostics
    that read the boundary causes rather than the field itself.
    """
    return time_limit & ~command_finished & ~terminated


@torch.no_grad()
def replay_timeout_final_mask(
    *,
    terminated: torch.Tensor,
    command_finished: torch.Tensor,
    time_limit: torch.Tensor,
) -> torch.Tensor:
    """Return rows whose bootstrap state is a captured pre-reset final state."""
    return time_limit & ~command_finished & ~terminated


@torch.no_grad()
def compute_continuation_coefficient(
    *,
    dones: torch.Tensor,
    truncations: torch.Tensor,
    discounts: torch.Tensor | float,
) -> torch.Tensor:
    """Return ``c_t``, the one continuation coefficient shared by both users.

    ``gamma * c_t`` multiplies the frozen-Teacher boundary value in the shaping
    term and the target Q value in the soft-Bellman bootstrap.  Both must read
    this same tensor: two independently derived coefficients silently disagree
    at command completion, and a shaping term whose continuation factor differs
    from the bootstrap's is no longer potential-based.
    """
    mask = continuation_bootstrap_mask(dones=dones, truncations=truncations)
    discount = torch.as_tensor(discounts, device=mask.device, dtype=torch.float32)
    if discount.ndim == 0:
        discount = discount.expand_as(mask)
    else:
        discount = discount.reshape(-1)
    if discount.shape != mask.shape:
        raise ValueError("Continuation coefficient tensors must have identical shape")
    if not bool(torch.isfinite(discount).all()):
        raise RuntimeError("Replay discount contains NaN/Inf")
    return (discount * mask.to(discount)).detach()


@torch.no_grad()
def compute_teacher_value_continuation(
    *,
    teacher_v_next: torch.Tensor,
    continuation: torch.Tensor,
    gamma: float | torch.Tensor,
    semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS,
) -> torch.Tensor:
    """Return the discounted frozen-Teacher boundary term ``gamma * c_t * B_t``.

    The boundary value ``B_t`` is the real successor value on ordinary rows and
    the captured pre-reset final value on pure time limits.  Command completion
    and physical termination cut.  Cut rows are selected away instead of
    multiplied by zero, so a stored sentinel or a non-finite next value can
    never reach the arithmetic.
    """
    if semantics != TEACHER_VALUE_BOUNDARY_SEMANTICS:
        raise ValueError(
            f"unsupported frozen Teacher value boundary semantics {semantics!r}"
        )
    coefficient = torch.as_tensor(continuation).detach().float().reshape(-1)
    gamma_value = torch.as_tensor(
        gamma, device=coefficient.device, dtype=torch.float32
    ).detach()
    if gamma_value.ndim == 0:
        gamma_scalar = _finite_scalar("teacher value gamma", gamma_value.item())
        if gamma_scalar < 0.0:
            raise ValueError("teacher value gamma must be non-negative")
        gamma_factor: float | torch.Tensor = gamma_scalar
    else:
        gamma_value = gamma_value.reshape(-1)
        if gamma_value.shape != coefficient.shape:
            raise ValueError("Teacher gamma powers and continuation must align")
        if not bool(torch.isfinite(gamma_value).all()):
            raise RuntimeError("Teacher gamma powers contain NaN/Inf")
        if bool((gamma_value < 0.0).any()):
            raise ValueError("teacher value gamma powers must be non-negative")
        gamma_factor = gamma_value
    following = (
        torch.as_tensor(teacher_v_next)
        .detach()
        .to(device=coefficient.device, dtype=torch.float32)
        .reshape(-1)
    )
    if following.shape != coefficient.shape:
        raise ValueError("Teacher continuation tensors must have identical shape")
    if not bool(torch.isfinite(coefficient).all()):
        raise RuntimeError("Teacher continuation coefficient contains NaN/Inf")
    bootstrapping = coefficient != 0.0
    following = torch.where(bootstrapping, following, torch.zeros_like(following))
    if not bool(torch.isfinite(following).all()):
        raise RuntimeError("Teacher continuation contains NaN/Inf")
    continuation_term = gamma_factor * coefficient * following
    if not torch.isfinite(continuation_term).all():
        raise RuntimeError("Discounted Teacher continuation contains NaN/Inf")
    return continuation_term.detach()


@torch.no_grad()
def _validated_reference_phase(
    name: str,
    phase: torch.Tensor | None,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return finite normalized phase metadata aligned with a value batch."""
    if phase is None:
        raise KeyError(
            f"phase-faded Teacher potential requires replay field {name!r}"
        )
    value = (
        torch.as_tensor(phase)
        .detach()
        .to(device=like.device, dtype=torch.float32)
        .reshape(-1)
    )
    if value.shape != like.shape:
        raise ValueError(f"{name} and Teacher values must have identical batch shape")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains NaN/Inf")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


@torch.no_grad()
def _teacher_value_terms_from_values(
    teacher_v: torch.Tensor,
    teacher_v_next: torch.Tensor,
    raw_reward: torch.Tensor,
    *,
    continuation: torch.Tensor,
    gamma: float | torch.Tensor,
    semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS,
    tvkd_lambda: float,
    potential_clip: float | None = None,
    reference_phase: torch.Tensor | None = None,
    next_reference_phase: torch.Tensor | None = None,
    use_phase_faded_teacher_potential: bool = False,
) -> TeacherValueTerms:
    """Build shaping terms while keeping bottleneck residuals on raw ``V_T``."""
    raw_reward = raw_reward.detach().float().reshape(-1)
    if not torch.isfinite(raw_reward).all():
        raise RuntimeError("TVKD raw reward contains NaN/Inf")
    teacher_v = torch.as_tensor(teacher_v).detach().float().reshape(-1)
    teacher_v_next = (
        torch.as_tensor(teacher_v_next)
        .detach()
        .to(device=teacher_v.device, dtype=torch.float32)
        .reshape(-1)
    )
    raw_reward = raw_reward.to(device=teacher_v.device)
    if teacher_v.shape != raw_reward.shape or teacher_v_next.shape != raw_reward.shape:
        raise ValueError("Teacher values and rewards must have identical batch shape")
    if not isinstance(use_phase_faded_teacher_potential, bool):
        raise ValueError("use_phase_faded_teacher_potential must be boolean")

    fixed_v = teacher_v
    fixed_v_next = teacher_v_next
    if potential_clip is not None:
        clip = _finite_scalar("potential_clip", potential_clip, positive=True)
        fixed_v = fixed_v.clamp(-clip, clip)
        fixed_v_next = fixed_v_next.clamp(-clip, clip)

    if use_phase_faded_teacher_potential:
        phase = _validated_reference_phase(
            REFERENCE_PHASE_KEY, reference_phase, teacher_v
        )
        next_phase = _validated_reference_phase(
            NEXT_REFERENCE_PHASE_KEY, next_reference_phase, teacher_v
        )
        teacher_potential = (1.0 - phase) * fixed_v
        teacher_potential_next = (1.0 - next_phase) * fixed_v_next
    else:
        teacher_potential = fixed_v
        teacher_potential_next = fixed_v_next

    raw_continuation = compute_teacher_value_continuation(
        teacher_v_next=teacher_v_next,
        continuation=continuation,
        gamma=gamma,
        semantics=semantics,
    )
    if not use_phase_faded_teacher_potential and potential_clip is None:
        # In the legacy ablation the shaped and raw residuals use identical
        # values, so retain the one-call fast path exactly.
        fixed_continuation = raw_continuation
    else:
        fixed_continuation = compute_teacher_value_continuation(
            teacher_v_next=teacher_potential_next,
            continuation=continuation,
            gamma=gamma,
            semantics=semantics,
        )
    potential_delta = fixed_continuation - teacher_potential
    shaping_scale = _finite_scalar("tvkd_lambda", tvkd_lambda)
    shaped_reward = raw_reward + shaping_scale * potential_delta
    # Bottleneck detection uses raw reward and the unclipped Teacher value.
    teacher_td_residual = raw_reward + raw_continuation - teacher_v
    checked_terms = (
        ("potential delta", potential_delta),
        ("shaped reward", shaped_reward),
        ("Teacher TD residual", teacher_td_residual),
    )
    all_terms_finite = torch.stack(
        tuple(torch.isfinite(value).all() for _, value in checked_terms)
    ).all()
    if not bool(all_terms_finite):
        # Keep the precise diagnostic on the exceptional slow path.
        for name, value in checked_terms:
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"TVKD {name} contains NaN/Inf")
    return TeacherValueTerms(
        teacher_v=teacher_v.detach(),
        teacher_v_next=teacher_v_next.detach(),
        teacher_potential=teacher_potential.detach(),
        teacher_potential_next=teacher_potential_next.detach(),
        potential_delta=potential_delta.detach(),
        shaped_reward=shaped_reward.detach(),
        teacher_td_residual=teacher_td_residual.detach(),
    )


@torch.no_grad()
def compute_teacher_value_terms(
    get_teacher_value: Callable[[torch.Tensor], torch.Tensor],
    teacher_critic_obs: torch.Tensor,
    next_teacher_critic_obs: torch.Tensor,
    raw_reward: torch.Tensor,
    *,
    continuation: torch.Tensor,
    gamma: float | torch.Tensor,
    semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS,
    tvkd_lambda: float,
    potential_clip: float | None = None,
    reference_phase: torch.Tensor | None = None,
    next_reference_phase: torch.Tensor | None = None,
    use_phase_faded_teacher_potential: bool = False,
) -> TeacherValueTerms:
    """Compute frozen-Teacher PBRS and the unfaded raw-reward TD residual."""
    teacher_v = get_teacher_value(teacher_critic_obs).float().reshape(-1)
    teacher_v_next = get_teacher_value(next_teacher_critic_obs).float().reshape(-1)
    return _teacher_value_terms_from_values(
        teacher_v,
        teacher_v_next,
        raw_reward,
        continuation=continuation,
        gamma=gamma,
        semantics=semantics,
        tvkd_lambda=tvkd_lambda,
        potential_clip=potential_clip,
        reference_phase=reference_phase,
        next_reference_phase=next_reference_phase,
        use_phase_faded_teacher_potential=use_phase_faded_teacher_potential,
    )


class TVKDDistributionalFastSACTeacherBC(DistributionalFastSACTeacherBC):
    """Twin-C51 FastSAC with frozen shaping and bottleneck-aligned replay."""

    @staticmethod
    def _validate_td3_config(cfg) -> None:
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)
        _validate_tvkd_algorithm_config(cfg)

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self.teacher_value_wrapper = FrozenTeacherValueWrapper(
            self.actor,
            self.critic,
            self.value_norm,
            tuple(self.q_critic_keys),
            tuple(self._q_critic_widths),
        )
        self.teacher_value_bottleneck_detector = TeacherValueBottleneckDetector(
            threshold=float(cfg.bottleneck_threshold),
            smoothing_window=int(cfg.bottleneck_smoothing_window),
            min_consecutive=int(cfg.bottleneck_min_consecutive),
            terminal_exclusion_steps=int(cfg.bottleneck_terminal_exclusion_steps),
            selection_mode=str(getattr(cfg, "bottleneck_selection_mode", "first")),
        )
        self._bottleneck_episode_histories: list[dict[str, list]] | None = None
        self._reset_student_replay_episode_tracking()
        self._reset_bottleneck_statistics()
        self._last_tvkd_diagnostics: dict[str, float] = {}
        # Per-rollout Teacher-value grids populated by
        # _student_teacher_td_residual_grid and consumed by
        # _dagger_transition_chunks to fill the replay cache fields.
        self._rollout_teacher_v_current_grid: torch.Tensor | None = None
        self._rollout_teacher_v_next_grid: torch.Tensor | None = None
        # Pending factual Student starts survive ordinary rollout boundaries.
        # Replay rings are fresh-only, so this collector-side queue is never a
        # checkpointed/resumed data source.
        self._student_q_n_step_accumulator: _Stage1NStepAccumulator | None = None

    def _needs_teacher_value_cache(self) -> bool:
        """True when at least one consumer of frozen Teacher values is active."""
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return False
        return bool(getattr(cfg, "use_tvkd_value_shaping", False)) or bool(
            getattr(cfg, "use_teacher_value_bottleneck_replay", False)
        )

    def _q_replay_storage_fields(self) -> tuple[str, ...]:
        """Extend the base schema with Teacher-value cache fields when needed."""
        base = super()._q_replay_storage_fields()
        if (
            int(getattr(self.cfg, "q_n_step", 1)) > 1
            and Q_EFFECTIVE_N_STEPS_KEY not in base
        ):
            base = (*base, Q_EFFECTIVE_N_STEPS_KEY)
        if self._needs_teacher_value_cache():
            return (*base, REPLAY_TEACHER_V_CURRENT_KEY, REPLAY_TEACHER_V_NEXT_KEY)
        return base

    def _q_replay_prefill_storage_fields(self) -> tuple[str, ...]:
        """Exclude cache fields from the prefill transition schema.

        Cache fields are added per-episode by
        ``_post_process_teacher_prefill_episode`` after the episode is
        assembled from per-rollout transition chunks, so they must not be
        requested from the raw rollout transitions dict.
        """
        base = super()._q_replay_storage_fields()
        if (
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
            and DAGGER_REPLAY_TEACHER_ACTIONS not in base
        ):
            base = (*base, DAGGER_REPLAY_TEACHER_ACTIONS)
        if (
            int(getattr(self.cfg, "q_n_step", 1)) > 1
            and Q_EFFECTIVE_N_STEPS_KEY not in base
        ):
            base = (*base, Q_EFFECTIVE_N_STEPS_KEY)
        return base  # intentionally omits cache fields

    @torch.no_grad()
    def _post_process_teacher_prefill_episode(
        self, episode: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute and attach frozen Teacher values for a committed episode.

        All rows in a successful Teacher episode are non-terminated ordinary
        transitions except the last row which has ``command_finished=True``.
        Command completion cuts the finite-horizon return, so the last row's
        cached next value is schema-only and never consumed by learning.
        """
        if not self._needs_teacher_value_cache():
            return episode
        snapshot = self._vecnorm_snapshot()
        obs_norm = self._normalize_replay_flat(
            episode["critic_observations"].to(self.device),
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        next_obs_norm = self._normalize_replay_flat(
            episode["next_critic_observations"].to(self.device),
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        teacher_v = self._batched_frozen_teacher_value(obs_norm).float()
        teacher_v_next = self._batched_frozen_teacher_value(next_obs_norm).float()
        episode = dict(episode)
        episode[REPLAY_TEACHER_V_CURRENT_KEY] = teacher_v.reshape(-1)
        episode[REPLAY_TEACHER_V_NEXT_KEY] = teacher_v_next.reshape(-1)
        return episode

    def get_frozen_teacher_value(
        self, teacher_critic_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.teacher_value_wrapper.get_frozen_teacher_value(teacher_critic_obs)

    @torch.no_grad()
    def _teacher_value_terms_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        continuation: torch.Tensor,
        *,
        gamma: float | torch.Tensor | None = None,
    ) -> "TeacherValueTerms":
        """Compute TVKD value terms, using the replay cache when available.

        In production mode (``teacher_value_cache_validate_fraction == 0``),
        reads ``replay_teacher_v_current`` and ``replay_teacher_v_next`` from
        the batch and raises a precise error if they are missing.  In debug
        validation mode, recomputes live Teacher values for a random subset of
        rows and asserts max abs error ≤ 1e-6.
        """
        validate_fraction = float(
            getattr(self.cfg, "teacher_value_cache_validate_fraction", 0.0)
        )
        teacher_gamma = float(self.cfg.gamma) if gamma is None else gamma
        cache_present = (
            REPLAY_TEACHER_V_CURRENT_KEY in batch and REPLAY_TEACHER_V_NEXT_KEY in batch
        )
        if not cache_present:
            if validate_fraction == 0.0:
                raise KeyError(
                    "TVKD value shaping requires precomputed Teacher-value cache "
                    f"fields '{REPLAY_TEACHER_V_CURRENT_KEY}' and "
                    f"'{REPLAY_TEACHER_V_NEXT_KEY}' in the replay batch. "
                    "These are missing, indicating the replay ring was built "
                    "without the cache schema. Rebuild with a fresh training run "
                    "or set teacher_value_cache_validate_fraction > 0 to enable "
                    "the debug live-inference fallback."
                )
            # Debug-only live fallback for incremental testing.
            return compute_teacher_value_terms(
                self.get_frozen_teacher_value,
                batch["critic_observations"],
                batch["next_critic_observations"],
                batch["rewards"],
                continuation=continuation,
                gamma=teacher_gamma,
                semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
                tvkd_lambda=float(self.cfg.tvkd_lambda),
                potential_clip=self.cfg.tvkd_potential_clip,
                reference_phase=batch.get(REFERENCE_PHASE_KEY),
                next_reference_phase=batch.get(NEXT_REFERENCE_PHASE_KEY),
                use_phase_faded_teacher_potential=bool(
                    self.cfg.use_phase_faded_teacher_potential
                ),
            )

        teacher_v = batch[REPLAY_TEACHER_V_CURRENT_KEY].detach().float().reshape(-1)
        teacher_v_next = batch[REPLAY_TEACHER_V_NEXT_KEY].detach().float().reshape(-1)

        if validate_fraction > 0.0:
            # Randomly sample a subset and compare live vs. cached values.
            # Learning batches have already passed through
            # _prepare_dagger_learning_batch, so both observation tensors are
            # in the frozen VecNorm coordinate system expected by the Teacher.
            batch_size = teacher_v.shape[0]
            num_validate = max(1, int(round(validate_fraction * batch_size)))
            validate_indices = torch.randperm(batch_size, device=teacher_v.device)[
                :num_validate
            ]
            obs_validate = batch["critic_observations"].index_select(
                0, validate_indices
            )
            live_v = self.get_frozen_teacher_value(obs_validate).float()
            cached_v = teacher_v.index_select(0, validate_indices)
            max_err_v = (live_v - cached_v).abs().max().item()

            # Physical termination and command completion never read V(next),
            # so Student replay may store a sentinel there. Validate only rows
            # retained by the shared continuation coefficient.
            bootstrapping = continuation != 0.0
            next_validate_indices = validate_indices[
                bootstrapping.index_select(0, validate_indices)
            ]
            if next_validate_indices.numel():
                next_obs_validate = batch["next_critic_observations"].index_select(
                    0, next_validate_indices
                )
                live_v_next = self.get_frozen_teacher_value(next_obs_validate).float()
                cached_v_next = teacher_v_next.index_select(0, next_validate_indices)
                max_err_v_next = (live_v_next - cached_v_next).abs().max().item()
            else:
                max_err_v_next = 0.0
            tolerance = 1e-6
            if max_err_v > tolerance or max_err_v_next > tolerance:
                raise RuntimeError(
                    f"Teacher-value cache validation failed: "
                    f"max_err_v={max_err_v:.3e}, "
                    f"max_err_v_next={max_err_v_next:.3e} "
                    f"(tolerance={tolerance:.0e}). "
                    "The cached values do not match live frozen-Teacher inference."
                )

        # Compute shaping from cached values without another frozen-critic call.
        return _teacher_value_terms_from_values(
            teacher_v,
            teacher_v_next,
            batch["rewards"],
            continuation=continuation,
            gamma=teacher_gamma,
            semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
            tvkd_lambda=float(self.cfg.tvkd_lambda),
            potential_clip=self.cfg.tvkd_potential_clip,
            reference_phase=batch.get(REFERENCE_PHASE_KEY),
            next_reference_phase=batch.get(NEXT_REFERENCE_PHASE_KEY),
            use_phase_faded_teacher_potential=bool(
                self.cfg.use_phase_faded_teacher_potential
            ),
        )

    def _tvkd_enabled(self) -> bool:
        if not bool(self.cfg.use_tvkd_value_shaping):
            return False
        # Residual parameterization is meaningful even at lambda=0: in that
        # gauge the represented raw Q equals Phi(s) + Z_residual(s, a).
        return bool(getattr(self.cfg, "use_teacher_residual_critic", False)) or (
            float(self.cfg.tvkd_lambda) != 0.0
        )

    @torch.no_grad()
    def _distributional_fastsac_target(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        # Preserve the baseline RNG sequence and target bit-for-bit when TVKD
        # is disabled (including the explicit lambda=0 equivalence setting).
        if not self._tvkd_enabled():
            baseline_result = (
                DistributionalFastSACTeacherBC._distributional_fastsac_target(
                    self, batch
                )
            )
            if "rewards" not in batch:
                return baseline_result
            target, metrics, log_prob = baseline_result
            metrics = dict(metrics)
            raw_reward = batch["rewards"].detach().float()
            zero = raw_reward.new_zeros(())
            metrics.update(
                {
                    "tvkd_teacher_potential_mean": zero,
                    "tvkd_teacher_potential_std": zero,
                    "tvkd_teacher_next_potential_mean": zero,
                    "tvkd_teacher_next_potential_std": zero,
                    "tvkd_potential_delta_mean": zero,
                    "tvkd_potential_delta_std": zero,
                    "tvkd_potential_delta_min": zero,
                    "tvkd_potential_delta_max": zero,
                    "tvkd_raw_reward_mean": raw_reward.mean(),
                    "tvkd_shaped_reward_mean": raw_reward.mean(),
                    "tvkd_shaped_reward_std": raw_reward.std(unbiased=False),
                    "tvkd_residual_reward_mean": raw_reward.mean(),
                    "tvkd_residual_reward_std": raw_reward.std(unbiased=False),
                    "tvkd_critic_reward_mean": raw_reward.mean(),
                    "tvkd_critic_reward_std": raw_reward.std(unbiased=False),
                    "tvkd_full_q_state_baseline_mean": zero,
                    "tvkd_full_q_state_baseline_std": zero,
                    "tvkd_teacher_residual_critic_enabled": zero,
                    "tvkd_reference_phase_mean": zero,
                    "tvkd_next_reference_phase_mean": zero,
                    "tvkd_command_terminal_potential_abs_mean": zero,
                    "tvkd_command_terminal_potential_abs_max": zero,
                    "tvkd_phase_faded_potential_enabled": zero,
                }
            )
            return target, metrics, log_prob

        next_dist = self._actor_dist_from_flat(batch["next_observations"])
        next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        next_log_prob = self._normalized_action_log_prob(next_raw_log_prob)
        required_boundary = {
            REPLAY_TERMINATED_KEY,
            REPLAY_COMMAND_FINISHED_KEY,
            REPLAY_TIME_LIMIT_KEY,
        }
        missing_boundary = required_boundary.difference(batch)
        if missing_boundary:
            raise KeyError(
                "TVKD shaping requires explicit replay boundary metadata; missing "
                f"{sorted(missing_boundary)}"
            )
        done = batch["dones"].bool()
        terminated = batch[REPLAY_TERMINATED_KEY].bool()
        command_finished = batch[REPLAY_COMMAND_FINISHED_KEY].bool()
        time_limit = batch[REPLAY_TIME_LIMIT_KEY].bool()
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError("TVKD replay contains an unknown boundary cause")
        expected_q_truncation = replay_truncation_mask(
            terminated=terminated,
            command_finished=command_finished,
            time_limit=time_limit,
        )
        if not torch.equal(batch["truncations"].bool(), expected_q_truncation):
            raise RuntimeError(
                "TVKD replay Q truncation disagrees with explicit boundary metadata"
            )
        # One continuation coefficient with two consumers: the frozen-Teacher
        # potential term inside the shaped reward, and the soft-Bellman
        # bootstrap of the C51 target below.  Both read this same tensor.
        continuation = compute_continuation_coefficient(
            dones=done,
            truncations=batch["truncations"],
            discounts=batch["discounts"],
        )
        effective_discount, gamma_power, effective_n_steps = (
            _q_target_discount_factors(self.cfg, batch)
        )
        terms = self._teacher_value_terms_from_batch(
            batch,
            continuation,
            gamma=gamma_power,
        )
        alpha = self.log_alpha.exp()
        bootstrap = continuation != 0.0
        entropy_tax = (
            effective_discount
            * bootstrap.to(effective_discount)
            * alpha
            * next_log_prob
        )
        # Let d_t = gamma**k * continuation and Phi be the configured endpoint
        # potential, where k is the factual Student horizon (Teacher rows use
        # k=1). The discounted replay reward has already accumulated k steps.
        # The ordinary shaped target represents
        #   Z^lambda = r + lambda(d_t Phi' - Phi)
        #              + d_t[Z^lambda_next - alpha log pi].
        # Reparameterize it exactly as
        #   Z^lambda(s,a) = (1-lambda) Phi(s) + Z_residual(s,a).
        # Cancelling the state-only baseline on both sides gives the
        # lambda-independent residual target
        #   r + (d_t Phi' - Phi)
        #     + d_t[Z_residual_next - alpha log pi].
        # No auxiliary loss, new reward, or Actor-gradient approximation is
        # introduced; the equality holds before C51 projection atom by atom.
        residual_reward = batch["rewards"].detach().float() + terms.potential_delta
        residual_enabled = bool(
            getattr(self.cfg, "use_teacher_residual_critic", False)
        )
        critic_reward = residual_reward if residual_enabled else terms.shaped_reward
        soft_reward = critic_reward.to(entropy_tax) - entropy_tax
        if not torch.isfinite(soft_reward).all():
            raise RuntimeError("TVKD soft critic reward contains NaN/Inf")

        target_logits = self._q_forward(
            self.qnet_target,
            batch["next_critic_observations"],
            next_action,
            batch.get(NEXT_Q_ACTUATOR_CONTEXT_KEY),
        )
        if _fastsac_q_critic_type(self.cfg) == SCALAR_Q_CRITIC_TYPE:
            if target_logits.ndim != 2 or target_logits.shape[0] != 2:
                raise RuntimeError("TVKD scalar target must have shape [2, batch]")
            raw_expected_heads = target_logits
            (
                selected_next_q,
                q1_target_fraction,
                q2_target_fraction,
            ) = _reduce_fastsac_twin_target(
                raw_expected_heads,
                raw_expected_heads,
                _fastsac_q_twin_reduction(self.cfg),
            )
            selected_target = soft_reward + effective_discount * (
                bootstrap.to(effective_discount) * selected_next_q
            )
            selected_expected = selected_target
            selected_entropy = torch.zeros_like(selected_target)
            zero = selected_target.new_zeros(())
            q1_left_support_clip_fraction = zero
            q1_right_support_clip_fraction = zero
            q2_left_support_clip_fraction = zero
            q2_right_support_clip_fraction = zero
            q1_support_clip_fraction = zero
            q2_support_clip_fraction = zero
            support_clip_fraction_mean = zero
            support_clip_fraction_max = zero
        else:
            target_probabilities = F.softmax(target_logits, dim=-1)
            raw_expected_heads = (
                target_probabilities * self.qnet_target.support
            ).sum(dim=-1)
            projection_gamma = (
                torch.full_like(continuation, gamma_power)
                if isinstance(gamma_power, float)
                else gamma_power
            )
            projected_heads = []
            support_clip_fractions = []
            for head_probability in target_probabilities:
                projected, left_fraction, right_fraction = (
                    _project_c51_probabilities(
                        head_probability,
                        soft_reward,
                        continuation,
                        projection_gamma,
                        self.qnet_target.support,
                    )
                )
                projected_heads.append(projected)
                support_clip_fractions.append((left_fraction, right_fraction))
            if len(support_clip_fractions) != 2:
                raise RuntimeError(
                    "TVKD C51 target must contain exactly two Q heads"
                )
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
            projected_heads = torch.stack(projected_heads, dim=0)
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
                * selected_target.clamp_min(
                    torch.finfo(selected_target.dtype).tiny
                ).log()
            ).sum(dim=-1)
        reward_abs = batch["rewards"].abs().mean()
        full_q_state_baseline = (
            (1.0 - float(self.cfg.tvkd_lambda)) * terms.teacher_potential
            if residual_enabled
            else torch.zeros_like(terms.teacher_potential)
        )

        def population_std(value: torch.Tensor) -> torch.Tensor:
            return value.float().std(unbiased=False)

        command_terminal = command_finished & ~terminated
        if bool(command_terminal.any()):
            command_potential_abs = terms.teacher_potential[command_terminal].abs()
            command_potential_abs_mean = command_potential_abs.mean()
            command_potential_abs_max = command_potential_abs.max()
        else:
            command_potential_abs_mean = terms.teacher_potential.new_zeros(())
            command_potential_abs_max = terms.teacher_potential.new_zeros(())
        reference_phase = batch.get(REFERENCE_PHASE_KEY)
        next_reference_phase = batch.get(NEXT_REFERENCE_PHASE_KEY)
        reference_phase_mean = (
            torch.as_tensor(reference_phase)
            .to(device=terms.teacher_v.device, dtype=torch.float32)
            .mean()
            if reference_phase is not None
            else terms.teacher_v.new_zeros(())
        )
        next_reference_phase_mean = (
            torch.as_tensor(next_reference_phase)
            .to(device=terms.teacher_v.device, dtype=torch.float32)
            .mean()
            if next_reference_phase is not None
            else terms.teacher_v.new_zeros(())
        )

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
            "left_support_projection_clipping_fraction": 0.5
            * (q1_left_support_clip_fraction + q2_left_support_clip_fraction),
            "right_support_projection_clipping_fraction": 0.5
            * (q1_right_support_clip_fraction + q2_right_support_clip_fraction),
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
            "tvkd_teacher_value_mean": terms.teacher_v.mean(),
            "tvkd_teacher_value_std": population_std(terms.teacher_v),
            "tvkd_teacher_next_value_mean": terms.teacher_v_next.mean(),
            "tvkd_teacher_next_value_std": population_std(terms.teacher_v_next),
            "tvkd_teacher_potential_mean": terms.teacher_potential.mean(),
            "tvkd_teacher_potential_std": population_std(terms.teacher_potential),
            "tvkd_teacher_next_potential_mean": terms.teacher_potential_next.mean(),
            "tvkd_teacher_next_potential_std": population_std(
                terms.teacher_potential_next
            ),
            "tvkd_potential_delta_mean": terms.potential_delta.mean(),
            "tvkd_potential_delta_std": population_std(terms.potential_delta),
            "tvkd_potential_delta_min": terms.potential_delta.min(),
            "tvkd_potential_delta_max": terms.potential_delta.max(),
            "tvkd_raw_reward_mean": batch["rewards"].float().mean(),
            "tvkd_shaped_reward_mean": terms.shaped_reward.mean(),
            "tvkd_shaped_reward_std": population_std(terms.shaped_reward),
            "tvkd_residual_reward_mean": residual_reward.mean(),
            "tvkd_residual_reward_std": population_std(residual_reward),
            "tvkd_critic_reward_mean": critic_reward.mean(),
            "tvkd_critic_reward_std": population_std(critic_reward),
            "tvkd_full_q_state_baseline_mean": full_q_state_baseline.mean(),
            "tvkd_full_q_state_baseline_std": population_std(
                full_q_state_baseline
            ),
            "tvkd_teacher_residual_critic_enabled": terms.teacher_v.new_tensor(
                float(residual_enabled)
            ),
            "tvkd_reference_phase_mean": reference_phase_mean,
            "tvkd_next_reference_phase_mean": next_reference_phase_mean,
            "tvkd_command_terminal_potential_abs_mean": (
                command_potential_abs_mean
            ),
            "tvkd_command_terminal_potential_abs_max": command_potential_abs_max,
            "tvkd_phase_faded_potential_enabled": terms.teacher_v.new_tensor(
                float(bool(self.cfg.use_phase_faded_teacher_potential))
            ),
        }
        return selected_target.detach(), metrics, next_log_prob.detach()

    # Compatibility alias used by tests and focused callers.
    _soft_c51_target = _distributional_fastsac_target

    def _reset_bottleneck_statistics(self) -> None:
        self._bottleneck_unsuccessful_episode_count = 0
        self._bottleneck_episodes_with_student_candidates = 0
        self._bottleneck_no_value_bottleneck_count = 0
        self._bottleneck_value_argmin_ablation_count = 0
        self._bottleneck_failed_student_episode_count = 0
        self._bottleneck_student_candidate_count = 0
        self._bottleneck_detected_count = 0
        self._bottleneck_fallback_count = 0
        self._bottleneck_no_candidate_count = 0
        self._bottleneck_selected_count = 0
        self._bottleneck_selected_step_sum = 0.0
        self._bottleneck_selected_phase_sum = 0.0
        self._bottleneck_score_sum = 0.0
        self._bottleneck_score_max = 0.0
        self._bottleneck_raw_td_residual_sum = 0.0
        self._bottleneck_smoothed_td_residual_sum = 0.0
        self._bottleneck_teacher_sequences_inserted = 0
        self._bottleneck_teacher_transitions_inserted = 0
        self._bottleneck_phase_match_distance_sum = 0.0
        self._bottleneck_phase_match_distance_count = 0
        # Onset quality diagnostics.  Without the distance from the onset to
        # the physical terminal there is no way to tell a genuine failure
        # precursor from an early false positive.
        self._bottleneck_onset_to_terminal_sum = 0.0
        self._bottleneck_failed_episode_length_sum = 0.0
        self._bottleneck_next_student_episode_id = 0
        self._student_focus_rows_marked = 0
        self._student_focus_rows_missing = 0
        self._failure_phase_student_focused_rows = 0
        self._failure_phase_student_uniform_fallback_rows = 0
        self._last_bottleneck_metadata: dict[str, float | int | str] = {}
        self._last_value_argmin_metadata: dict[str, float | int | str] = {}
        self._verified_failure_motion_phase_histogram: dict[int, torch.Tensor] = {}

    def _reset_student_replay_episode_tracking(self) -> None:
        """Reset ephemeral IDs whose raw replay ring is intentionally fresh."""
        self._student_replay_env_episode_ids: list[int | None] | None = None
        self._student_replay_env_episode_steps: list[int] | None = None
        self._student_replay_next_episode_id = 0
        self._student_replay_episode_id_grid: torch.Tensor | None = None
        self._student_replay_episode_step_grid: torch.Tensor | None = None
        self._pending_student_focus_events: list[tuple[int, torch.Tensor]] = []

    @staticmethod
    def _new_bottleneck_episode_history() -> dict[str, list]:
        return {
            "phase": [],
            "teacher_td_residual": [],
            "source_id": [],
            "motion_id": [],
            "true_terminal": [],
            "timeout": [],
            "replay_valid": [],
        }

    @torch.no_grad()
    def _batched_frozen_teacher_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Evaluate each cached state once while bounding inference memory."""
        if observation.ndim != 2:
            raise ValueError("Bottleneck Teacher observations must be rank two")
        if observation.shape[0] == 0:
            return observation.new_empty((0,), dtype=torch.float32)
        microbatch = max(1, int(getattr(self.cfg, "q_batch_size", 4096)))
        values = [
            self.get_frozen_teacher_value(observation[start : start + microbatch])
            for start in range(0, observation.shape[0], microbatch)
        ]
        return torch.cat(values, dim=0)

    @torch.no_grad()
    def _student_teacher_td_residual_grid(
        self, rollout: TensorDict, student: torch.Tensor
    ) -> torch.Tensor:
        """Cache raw-reward Teacher TD residuals for executed Student rows.

        Adjacent Student transitions share their state value: the union of
        required episode states is normalized and evaluated once. Terminal
        next/reset observations are not forwarded because their bootstrap mask
        is zero.
        """
        num_envs, num_steps = (int(value) for value in rollout.batch_size)
        student = student.reshape(num_envs, num_steps).bool()
        residual_grid = torch.zeros(
            (num_envs, num_steps), dtype=torch.float32, device=student.device
        )
        # Never let a short-circuit or exceptional caller reuse grids from a
        # previous rollout.
        self._rollout_teacher_v_current_grid = None
        self._rollout_teacher_v_next_grid = None
        if not bool(student.any()):
            return residual_grid
        final_batch = getattr(self, "_rollout_final_batch", None)
        if not isinstance(final_batch, Mapping):
            raise RuntimeError("Bottleneck replay requires the captured rollout final")

        reward = self._scalarize_q_reward(rollout[REWARD_KEY]).reshape(
            num_envs, num_steps
        )
        done = rollout[DONE_KEY].reshape(num_envs, num_steps).bool()
        terminated = rollout[TERM_KEY].reshape(num_envs, num_steps).bool()
        command_finished = (
            rollout["next", "stats", "command_finished"]
            .reshape(num_envs, num_steps)
            .bool()
        )
        time_limit = (
            rollout["next", "stats", "episode_time_limit"]
            .reshape(num_envs, num_steps)
            .bool()
        )
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError(
                "Bottleneck replay has an unknown or non-done boundary cause"
            )
        replay_discount = rollout["next", "discount"].reshape(num_envs, num_steps)
        truncation = _vaic_truncation_mask(rollout).reshape(num_envs, num_steps)
        timeout_final = replay_timeout_final_mask(
            terminated=terminated,
            command_finished=command_finished,
            time_limit=time_limit,
        )
        q_bootstrap = continuation_bootstrap_mask(
            dones=done, truncations=truncation
        ).reshape(num_envs, num_steps)
        state_rows = []
        state_ids = []
        # A timeout's real next observation is captured immediately before the
        # environment resets.  Give those states IDs outside the regular
        # rollout grid so they cannot alias the reset state at ``step + 1``.
        timeout_next_id_by_transition = torch.full(
            (num_envs * num_steps,),
            -1,
            dtype=torch.long,
            device=student.device,
        )
        truncation_batches = getattr(self, "_truncation_final_batches", [])
        if truncation_batches:
            captured_indices = torch.cat(
                [batch["indices"] for batch in truncation_batches], dim=0
            ).to(device=student.device, dtype=torch.long)
            captured_next_raw = torch.cat(
                [batch["next_critic_observations"] for batch in truncation_batches],
                dim=0,
            ).to(device=student.device)
            sorted_captured_indices = captured_indices.sort().values
            invalid_captured_indices = (
                (captured_indices < 0).any()
                | (captured_indices >= num_envs * num_steps).any()
                | (sorted_captured_indices[1:] == sorted_captured_indices[:-1]).any()
            )
            if bool(invalid_captured_indices):
                raise RuntimeError("Bottleneck timeout-final indices are invalid")
            if bool(
                (~timeout_final.reshape(-1).index_select(0, captured_indices)).any()
            ):
                raise RuntimeError(
                    "Bottleneck pre-reset final capture is not a pure time limit"
                )
            captured_student = student.reshape(-1).index_select(0, captured_indices)
            captured_indices = captured_indices[captured_student]
            captured_next_raw = captured_next_raw[captured_student]
            virtual_ids = num_envs * (num_steps + 1) + torch.arange(
                captured_indices.numel(),
                dtype=torch.long,
                device=student.device,
            )
            timeout_next_id_by_transition.index_copy_(0, captured_indices, virtual_ids)
            state_rows.append(captured_next_raw)
            state_ids.append(virtual_ids)
        missing_student_timeout = (
            student.reshape(-1)
            & timeout_final.reshape(-1)
            & (timeout_next_id_by_transition < 0)
        )
        if bool(missing_student_timeout.any()):
            raise RuntimeError(
                "Bottleneck replay lacks a captured timeout-final observation"
            )

        # This fixed mask is algebraically identical to the former step loop:
        # every Student row needs its current state, while only an ordinary
        # bootstrapping row needs the regular state at ``step + 1``.  A pure
        # timeout uses the virtual final state registered above. Command
        # completion is a terminal cut and therefore needs no successor state.
        regular_state_needed = torch.zeros(
            (num_envs, num_steps + 1),
            dtype=torch.bool,
            device=student.device,
        )
        regular_state_needed[:, :num_steps] |= student
        regular_state_needed[:, 1:] |= student & q_bootstrap & ~timeout_final

        rollout_state_coordinates = regular_state_needed[:, :num_steps].nonzero(
            as_tuple=False
        )
        rollout_state_envs = rollout_state_coordinates[:, 0]
        rollout_state_steps = rollout_state_coordinates[:, 1]
        # Index only the critic source leaves. Advanced-indexing the complete
        # rollout would also gather every perception/history/statistics tensor.
        replay_source_keys = (
            *self.q_critic_keys,
            *(self._raw_replay_key(key) for key in self.q_critic_keys),
        )
        rollout_sources = rollout.select(*replay_source_keys, strict=False)
        state_rows.append(
            self._cat_replay_sources(
                rollout_sources[rollout_state_envs, rollout_state_steps],
                self.q_critic_keys,
            )
        )
        state_ids.append(rollout_state_envs * (num_steps + 1) + rollout_state_steps)

        final_envs = (
            regular_state_needed[:, num_steps].nonzero(as_tuple=False).squeeze(-1)
        )
        state_rows.append(
            final_batch["next_critic_observations"]
            .reshape(num_envs, self._q_critic_dim)
            .index_select(0, final_envs)
        )
        state_ids.append(final_envs * (num_steps + 1) + num_steps)

        all_state_ids = torch.cat(state_ids, dim=0)
        sorted_state_ids, order = all_state_ids.sort()
        if bool((sorted_state_ids[1:] <= sorted_state_ids[:-1]).any()):
            raise RuntimeError("Bottleneck Teacher state cache contains duplicates")
        raw_states = torch.cat(state_rows, dim=0).index_select(0, order)
        snapshot = self._vecnorm_snapshot()
        states = self._normalize_replay_flat(
            raw_states,
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        state_values = self._batched_frozen_teacher_value(states).float()

        # ``nonzero`` on the transposed mask retains the former loop's exact
        # step-major, then environment-major transition order.
        transition_coordinates = student.transpose(0, 1).nonzero(as_tuple=False)
        transition_steps = transition_coordinates[:, 0]
        transition_envs = transition_coordinates[:, 1]
        flat_positions = transition_envs * num_steps + transition_steps
        current_ids = transition_envs * (num_steps + 1) + transition_steps
        regular_next_ids = current_ids + 1
        timeout_next_ids = timeout_next_id_by_transition.index_select(0, flat_positions)
        transition_truncation = truncation[transition_envs, transition_steps]
        transition_timeout = timeout_final[transition_envs, transition_steps]
        next_ids = torch.where(
            transition_timeout,
            timeout_next_ids,
            regular_next_ids,
        )
        current_positions = torch.searchsorted(sorted_state_ids, current_ids)
        teacher_v = state_values.index_select(0, current_positions)
        transition_q_bootstrap = q_bootstrap[transition_envs, transition_steps]
        # Current IDs are always present, so they are a safe lookup placeholder
        # for physical boundaries whose next value must remain zero.
        lookup_next_ids = torch.where(
            transition_q_bootstrap,
            next_ids,
            current_ids,
        )
        next_positions = torch.searchsorted(sorted_state_ids, lookup_next_ids)
        looked_up_next = state_values.index_select(0, next_positions)
        teacher_v_next = torch.where(
            transition_q_bootstrap,
            looked_up_next,
            torch.zeros_like(looked_up_next),
        )
        transition_continuation = compute_continuation_coefficient(
            dones=done[transition_envs, transition_steps],
            truncations=transition_truncation,
            discounts=replay_discount[transition_envs, transition_steps],
        )
        teacher_continuation = compute_teacher_value_continuation(
            teacher_v_next=teacher_v_next,
            continuation=transition_continuation,
            gamma=float(self.cfg.gamma),
            semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
        )
        transition_residual = (
            reward[transition_envs, transition_steps].float()
            + teacher_continuation
            - teacher_v
        )
        if not torch.isfinite(transition_residual).all():
            raise RuntimeError("Bottleneck Teacher TD residual contains NaN/Inf")
        residual_grid.reshape(-1).index_copy_(
            0,
            flat_positions,
            transition_residual.to(residual_grid),
        )
        # Also populate per-transition value grids for the replay cache.
        v_current_grid = torch.zeros(
            (num_envs, num_steps), dtype=torch.float32, device=student.device
        )
        v_next_grid = torch.zeros(
            (num_envs, num_steps), dtype=torch.float32, device=student.device
        )
        v_current_grid.reshape(-1).index_copy_(0, flat_positions, teacher_v.float())
        v_next_grid.reshape(-1).index_copy_(0, flat_positions, teacher_v_next.float())
        self._rollout_teacher_v_current_grid = v_current_grid
        self._rollout_teacher_v_next_grid = v_next_grid
        return residual_grid

    def _student_bottleneck_anchor_indices(
        self,
        history: Mapping[str, list],
        onset: int,
    ) -> torch.Tensor:
        """Choose only replay-valid Student rows strictly before the onset."""
        history_length = len(history["phase"])
        lookback = int(self.cfg.failure_phase_lookback_steps)
        onset = int(onset)
        if not 0 <= onset < history_length:
            raise IndexError("Bottleneck onset lies outside the episode history")
        start = max(0, onset - lookback)
        source = torch.tensor(history["source_id"], dtype=torch.long)
        if any(
            key not in history or len(history[key]) != history_length
            for key in ("motion_id", "replay_valid")
        ):
            raise RuntimeError("Bottleneck history lacks aligned motion IDs")
        motion_id = torch.tensor(history["motion_id"], dtype=torch.long)
        replay_valid = torch.tensor(history["replay_valid"], dtype=torch.bool)
        onset_motion_id = motion_id[onset]
        episode_step = torch.arange(history_length)
        candidates = (
            (
                (source == SOURCE_STUDENT)
                & replay_valid
                & (motion_id == onset_motion_id)
                & (episode_step >= start)
                & (episode_step < onset)
            )
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        if candidates.numel() == 0:
            return candidates
        requested = min(
            int(self.cfg.failure_phase_samples_per_failure),
            int(candidates.numel()),
        )
        # If a diagnostic ablation requests fewer than the configured window,
        # retain the closest causal rows. Never reach farther back or forward
        # to fill a Teacher/replay-invalid gap.
        return candidates[-requested:]

    def _queue_student_bottleneck_rows(
        self,
        replay_episode_id: int,
        precursor_indices: torch.Tensor,
    ) -> None:
        # Verification is retrospective at the physical terminal. The normal
        # replay ring may already have overwritten a very early precursor; the
        # existing student_focus_rows_missing counter makes that loss explicit
        # instead of substituting onset/post-onset rows.
        if precursor_indices.numel():
            events = getattr(self, "_pending_student_focus_events", None)
            if events is None:
                events = []
                self._pending_student_focus_events = events
            events.append(
                (int(replay_episode_id), precursor_indices.detach().long().cpu())
            )

    @torch.no_grad()
    def _rebuild_teacher_phase_match_index(self) -> None:
        """Index immutable Teacher rows by motion and phase bin.

        The canonical Teacher ring is frozen after prefill.  Phase-match
        diagnostics used to rescan every row once for every newly verified
        anchor, even though the eligible row pools never change.  Retain each
        pool in ascending replay-row order and resolve the (at most two) tied
        nearest-bin union lazily for each ``(motion, risk_bin)`` pair.
        """
        phase_bins = self._teacher_replay_phase_bins
        motion_ids = self._teacher_replay_motion_ids
        if phase_bins.device.type != "cpu" or motion_ids.device.type != "cpu":
            raise RuntimeError("Teacher phase-match index must be CPU resident")
        if phase_bins.ndim != 1 or motion_ids.shape != phase_bins.shape:
            raise RuntimeError("Teacher phase-match index tensors must align")

        bins = int(self.cfg.failure_phase_num_bins)
        unique_motions, motion_inverse = motion_ids.unique(
            sorted=True, return_inverse=True
        )
        pair_keys = motion_inverse * bins + phase_bins
        # Stable grouping keeps replay rows ascending inside every pair, which
        # is the order produced by the former ``same_motion[mask]`` expression.
        pair_order = torch.argsort(pair_keys, stable=True)
        sorted_pair_keys = pair_keys.index_select(0, pair_order)
        unique_pair_keys, pair_counts = torch.unique_consecutive(
            sorted_pair_keys, return_counts=True
        )
        pair_groups = pair_order.split(pair_counts.tolist())
        pair_rows = {
            int(key): rows
            for key, rows in zip(
                unique_pair_keys.tolist(), pair_groups, strict=True
            )
        }
        empty_rows = torch.empty(0, dtype=torch.long)
        rows_by_motion_bin: dict[int, tuple[torch.Tensor, ...]] = {}
        occupied_bins_by_motion: dict[int, torch.Tensor] = {}
        for motion_position, motion_id in enumerate(unique_motions.tolist()):
            motion_id = int(motion_id)
            bin_rows = tuple(
                pair_rows.get(motion_position * bins + bin_index, empty_rows)
                for bin_index in range(bins)
            )
            rows_by_motion_bin[motion_id] = bin_rows
            occupied_bins_by_motion[motion_id] = torch.tensor(
                [
                    bin_index
                    for bin_index, rows in enumerate(bin_rows)
                    if rows.numel()
                ],
                dtype=torch.long,
            )

        self._teacher_phase_match_rows_by_motion_bin = rows_by_motion_bin
        self._teacher_phase_match_occupied_bins_by_motion = (
            occupied_bins_by_motion
        )
        self._teacher_phase_match_nearest_pool_cache: dict[
            tuple[int, int], tuple[int, torch.Tensor]
        ] = {}
        self._teacher_phase_match_index_source = (
            phase_bins,
            motion_ids,
            bins,
            int(phase_bins.numel()),
        )

    @torch.no_grad()
    def _build_teacher_phase_index(self) -> None:
        """Build the sampler index and invalidate phase-match pool caches."""
        super()._build_teacher_phase_index()
        self._rebuild_teacher_phase_match_index()

    def _teacher_phase_match_pool(
        self,
        motion_id: int,
        risk_bin: int,
    ) -> tuple[int, torch.Tensor] | None:
        """Return the original nearest-bin row union for one anchor class."""
        motion_id = int(motion_id)
        risk_bin = int(risk_bin)
        key = (motion_id, risk_bin)
        cached = self._teacher_phase_match_nearest_pool_cache.get(key)
        if cached is not None:
            return cached

        bin_rows = self._teacher_phase_match_rows_by_motion_bin.get(motion_id)
        occupied = self._teacher_phase_match_occupied_bins_by_motion.get(motion_id)
        if bin_rows is None or occupied is None or occupied.numel() == 0:
            return None
        distances = (occupied - risk_bin).abs()
        nearest_distance = int(distances.min().item())
        nearest_bins = occupied[distances == nearest_distance]
        nearest_rows = [bin_rows[int(index)] for index in nearest_bins.tolist()]
        rows = nearest_rows[0]
        if len(nearest_rows) > 1:
            # The legacy mask selected from ``same_motion`` and therefore
            # returned the union in ascending replay-row order, not bin order.
            rows = torch.cat(nearest_rows).sort().values
        result = (nearest_distance, rows)
        self._teacher_phase_match_nearest_pool_cache[key] = result
        return result

    @torch.no_grad()
    def _record_teacher_phase_match_distances(
        self,
        anchor_phases: torch.Tensor,
        anchor_motion_ids: torch.Tensor,
    ) -> None:
        anchor_phases = anchor_phases.detach().float().cpu().reshape(-1)
        anchor_motion_ids = anchor_motion_ids.detach().long().cpu().reshape(-1)
        if anchor_phases.shape != anchor_motion_ids.shape:
            raise ValueError("Teacher phase anchors and motion IDs must align")
        replay = getattr(self, "q_teacher_replay", None)
        if replay is None or int(getattr(replay, "size", 0)) < 1:
            return
        if REFERENCE_PHASE_KEY not in getattr(replay, "data", {}):
            return
        if not bool(getattr(self, "_teacher_phase_index_ready", False)) or not all(
            hasattr(self, name)
            for name in (
                "_teacher_replay_phase_bins",
                "_teacher_replay_motion_ids",
            )
        ):
            self._build_teacher_phase_index()
        bins = int(self.cfg.failure_phase_num_bins)
        expected_source = (
            self._teacher_replay_phase_bins,
            self._teacher_replay_motion_ids,
            bins,
            int(self._teacher_replay_phase_bins.numel()),
        )
        cached_source = getattr(self, "_teacher_phase_match_index_source", None)
        if (
            cached_source is None
            or len(cached_source) != len(expected_source)
            or cached_source[0] is not expected_source[0]
            or cached_source[1] is not expected_source[1]
            or cached_source[2:] != expected_source[2:]
        ):
            self._rebuild_teacher_phase_match_index()
        teacher_phase = replay.data[REFERENCE_PHASE_KEY][: replay.size]
        teacher_phase = teacher_phase.reshape(replay.size, -1)[:, 0].float().cpu()
        max_distance = getattr(self.cfg, "max_teacher_phase_match_distance", None)
        if max_distance is not None:
            max_distance = _finite_scalar(
                "max_teacher_phase_match_distance", max_distance
            )
            if max_distance < 0.0:
                raise ValueError(
                    "max_teacher_phase_match_distance must be non-negative"
                )
        for target, motion_id in zip(anchor_phases, anchor_motion_ids, strict=True):
            risk_bin = min(max(int(float(target) * bins), 0), bins - 1)
            match = self._teacher_phase_match_pool(int(motion_id), risk_bin)
            if match is None:
                # The canonical sampler will uniformly backfill this anchor.
                # Do not report a cross-motion phase distance as a match.
                continue
            nearest_distance, rows = match
            if (
                max_distance is not None
                and nearest_distance / float(bins) > max_distance
            ):
                continue
            # This mirrors the canonical sampler: exact motion first, nearest
            # occupied phase bin second, then a uniform row inside that pool.
            distance = (teacher_phase.index_select(0, rows) - target).abs().mean()
            self._bottleneck_phase_match_distance_sum += float(distance.item())
            self._bottleneck_phase_match_distance_count += 1

    @torch.no_grad()
    def _register_bottleneck_teacher_sequence(
        self,
        history: Mapping[str, list],
        precursor_indices: torch.Tensor,
    ) -> int:
        # Failure Teacher examples must match the same *preventive* phases as
        # Failure Student examples. Onset/post-onset phases are intentionally
        # not promoted into this curriculum.
        precursor_indices = torch.as_tensor(precursor_indices).long().reshape(-1)
        if precursor_indices.numel() == 0:
            return 0
        index_list = [int(index) for index in precursor_indices.tolist()]
        phases = torch.tensor(
            [history["phase"][index] for index in index_list],
            dtype=torch.float64,
        )
        if "motion_id" not in history or len(history["motion_id"]) != len(
            history["phase"]
        ):
            raise RuntimeError("Bottleneck history lacks aligned motion IDs")
        motion_ids = torch.tensor(
            [history["motion_id"][index] for index in index_list],
            dtype=torch.long,
        )
        if bool((motion_ids < 0).any()):
            raise RuntimeError("Bottleneck history contains a negative motion ID")
        source = torch.tensor(
            [history["source_id"][index] for index in index_list], dtype=torch.long
        )
        replay_valid = torch.tensor(
            [history["replay_valid"][index] for index in index_list],
            dtype=torch.bool,
        )
        if not bool((source == SOURCE_STUDENT).all()) or not bool(replay_valid.all()):
            raise RuntimeError(
                "Teacher prevention anchors must come from replay-valid Student rows"
            )
        bins = int(self.cfg.failure_phase_num_bins)
        bin_indices = torch.floor(phases * bins).long().clamp_(0, bins - 1)
        self._failure_phase_histogram.index_add_(
            0, bin_indices, torch.ones_like(phases)
        )
        motion_histograms = getattr(
            self, "_verified_failure_motion_phase_histogram", None
        )
        if motion_histograms is None:
            motion_histograms = {}
            self._verified_failure_motion_phase_histogram = motion_histograms
        if not isinstance(motion_histograms, dict):
            raise RuntimeError("Verified motion histogram state must be a dict")
        for motion_id in motion_ids.unique(sorted=True).tolist():
            motion_id = int(motion_id)
            motion_histogram = motion_histograms.get(motion_id)
            if motion_histogram is None:
                motion_histogram = torch.zeros(bins, dtype=torch.float64, device="cpu")
                motion_histograms[motion_id] = motion_histogram
            if (
                motion_histogram.shape != (bins,)
                or motion_histogram.dtype != torch.float64
                or motion_histogram.device.type != "cpu"
            ):
                raise RuntimeError(
                    "Verified motion histogram must be CPU float64 with one value per bin"
                )
            same_motion = motion_ids == motion_id
            selected_bins = bin_indices[same_motion]
            motion_histogram.index_add_(
                0,
                selected_bins,
                torch.ones_like(selected_bins, dtype=torch.float64),
            )
        count = int(phases.numel())
        self._failure_phase_episode_count += 1
        self._failure_phase_anchor_count += count
        self._bottleneck_teacher_sequences_inserted += 1
        self._bottleneck_teacher_transitions_inserted += count
        self._record_teacher_phase_match_distances(phases, motion_ids)
        self._failure_histogram_device_cache.clear()
        return count

    def _record_bottleneck_selection(
        self,
        *,
        episode_id: int,
        step: int,
        confirmation_step: int,
        phase: float,
        score: float,
        raw_residual: float,
        smoothed_residual: float,
        precursor_indices: torch.Tensor,
        fallback: str,
    ) -> None:
        self._bottleneck_selected_count += 1
        self._bottleneck_selected_step_sum += float(step)
        self._bottleneck_selected_phase_sum += float(phase)
        self._bottleneck_score_sum += float(score)
        self._bottleneck_score_max = max(self._bottleneck_score_max, float(score))
        self._bottleneck_raw_td_residual_sum += float(raw_residual)
        self._bottleneck_smoothed_td_residual_sum += float(smoothed_residual)
        precursor_indices = torch.as_tensor(precursor_indices).long().reshape(-1)
        self._last_bottleneck_metadata = {
            "student_episode_id": int(episode_id),
            "bottleneck_step": int(step),
            "bottleneck_confirmation_step": int(confirmation_step),
            "bottleneck_phase": float(phase),
            "bottleneck_score": float(score),
            "raw_teacher_td_residual": float(raw_residual),
            "smoothed_teacher_td_residual": float(smoothed_residual),
            "precursor_start_step": (
                int(precursor_indices[0]) if precursor_indices.numel() else -1
            ),
            "precursor_end_step": (
                int(precursor_indices[-1]) if precursor_indices.numel() else -1
            ),
            "precursor_row_count": int(precursor_indices.numel()),
            "fallback": fallback,
        }

    @torch.no_grad()
    def _process_failed_student_episode(
        self, history: Mapping[str, list], *, replay_episode_id: int | None = None
    ) -> int:
        episode_id = self._bottleneck_next_student_episode_id
        self._bottleneck_next_student_episode_id += 1
        self._bottleneck_failed_student_episode_count += 1
        episode_length = len(history["phase"])
        self._bottleneck_failed_episode_length_sum += float(episode_length)
        residual = torch.tensor(history["teacher_td_residual"], dtype=torch.float32)
        source = torch.tensor(history["source_id"], dtype=torch.long)
        phase = torch.tensor(history["phase"], dtype=torch.float32)
        terminal = torch.tensor(history["true_terminal"], dtype=torch.bool)
        timeout = torch.tensor(history["timeout"], dtype=torch.bool)
        replay_valid = torch.tensor(history["replay_valid"], dtype=torch.bool)
        fallback_mode = str(getattr(self.cfg, "bottleneck_fallback_mode", "none"))
        result = self.teacher_value_bottleneck_detector.detect(
            residual,
            source,
            phase,
            terminal,
            timeout,
            replay_valid,
            fallback_mode=fallback_mode,
        )
        diagnostics = self.teacher_value_bottleneck_detector.last_diagnostics
        candidate_count = int(diagnostics["student_candidate_count"])
        self._bottleneck_student_candidate_count += candidate_count
        if candidate_count:
            self._bottleneck_episodes_with_student_candidates += 1
        if bool(diagnostics["no_candidate"]):
            self._bottleneck_no_candidate_count += 1

        if result is None:
            self._bottleneck_no_value_bottleneck_count += 1
            return 0

        if result.threshold_detected:
            self._bottleneck_detected_count += 1
            # The dispatched history ends on its own physical terminal, so the
            # last index is tau.  A small distance means the onset sits just
            # before the failure; a large one means it fired early.
            self._bottleneck_onset_to_terminal_sum += float(
                (episode_length - 1) - int(result.index)
            )
            precursor_indices = self._student_bottleneck_anchor_indices(
                history, result.index
            )
            self._record_bottleneck_selection(
                episode_id=episode_id,
                step=result.index,
                confirmation_step=result.confirmation_index,
                phase=result.phase,
                score=result.score,
                raw_residual=result.raw_teacher_td_residual,
                smoothed_residual=result.smoothed_teacher_td_residual,
                precursor_indices=precursor_indices,
                fallback="none",
            )
            if replay_episode_id is not None and precursor_indices.numel():
                self._queue_student_bottleneck_rows(
                    replay_episode_id, precursor_indices
                )
            return self._register_bottleneck_teacher_sequence(
                history, precursor_indices
            )

        if result.selection_origin != "value_argmin" or not result.used_fallback:
            raise RuntimeError(
                "Bottleneck detector returned an unknown selection origin"
            )
        # ``value_argmin`` is an explicit ablation diagnostic.  It never updates
        # the verified histogram or the verified Failure Student pool.
        self._bottleneck_fallback_count += 1
        self._bottleneck_value_argmin_ablation_count += 1
        self._bottleneck_no_value_bottleneck_count += 1
        self._last_value_argmin_metadata = {
            "student_episode_id": int(episode_id),
            "selected_step": int(result.index),
            "selected_phase": float(result.phase),
            "score": float(result.score),
            "raw_teacher_td_residual": float(result.raw_teacher_td_residual),
            "smoothed_teacher_td_residual": float(result.smoothed_teacher_td_residual),
            "selection_origin": "value_argmin",
        }
        return 0

    @torch.no_grad()
    def _update_failure_phase_histogram(self, rollout: TensorDict) -> int:
        """Register preventive phases from value-verified physical failures."""
        if not bool(self.cfg.use_teacher_value_bottleneck_replay):
            # Disabling value bottlenecks leaves both focused sources empty.  In
            # particular, never fall through to the parent's terminal-lookback
            # curriculum.
            self._bottleneck_episode_histories = None
            self._reset_student_replay_episode_tracking()
            histogram = getattr(self, "_failure_phase_histogram", None)
            motion_histograms = getattr(
                self, "_verified_failure_motion_phase_histogram", None
            )
            if (torch.is_tensor(histogram) and bool((histogram != 0).any())) or (
                isinstance(motion_histograms, Mapping) and bool(motion_histograms)
            ):
                self._reset_failure_curriculum_state()
            self._verified_failure_motion_phase_histogram = {}
            if not isinstance(
                getattr(self, "_rollout_final_batch", None), Mapping
            ):
                # No replay transitions will be emitted for this rollout. A
                # pending factual return cannot jump across that unobserved
                # collection gap.
                self._student_q_n_step_accumulator = None
            # When value shaping is active without bottleneck detection, compute
            # the value grids needed to populate the replay cache.
            if (
                self._needs_teacher_value_cache()
                and len(rollout.batch_size) == 2
                and isinstance(getattr(self, "_rollout_final_batch", None), Mapping)
                and DAGGER_IS_STUDENT_ACTION_KEY in rollout.keys(True, True)
            ):
                ne, ns = (int(v) for v in rollout.batch_size)
                student_mask = (
                    rollout[DAGGER_IS_STUDENT_ACTION_KEY]
                    .reshape(ne, ns, -1)
                    .bool()
                    .any(dim=-1)
                )
                self._student_teacher_td_residual_grid(rollout, student_mask)
            else:
                self._rollout_teacher_v_current_grid = None
                self._rollout_teacher_v_next_grid = None
            return 0
        if len(rollout.batch_size) != 2:
            raise ValueError("bottleneck tracking requires an [env,time] rollout")
        if not isinstance(getattr(self, "_rollout_final_batch", None), Mapping):
            # Staging/finalization phases can intentionally suppress raw replay
            # capture. Do not splice an unobserved gap into episode histories.
            self._bottleneck_episode_histories = None
            self._student_replay_env_episode_ids = None
            self._student_replay_env_episode_steps = None
            self._student_replay_episode_id_grid = None
            self._student_replay_episode_step_grid = None
            self._rollout_teacher_v_current_grid = None
            self._rollout_teacher_v_next_grid = None
            self._student_q_n_step_accumulator = None
            events = getattr(self, "_pending_student_focus_events", None)
            if events is not None:
                events.clear()
            return 0
        num_envs, num_steps = (int(value) for value in rollout.batch_size)

        def grid(value: torch.Tensor, *, boolean: bool = False) -> torch.Tensor:
            value = value.reshape(num_envs, num_steps, -1)
            if boolean:
                return value.bool().any(dim=-1)
            if value.shape[-1] != 1:
                raise ValueError("bottleneck phase must have one scalar per step")
            return value[..., 0]

        phase = grid(self._reference_phase(rollout))
        raw_motion_id = rollout.get(REPLAY_MOTION_ID_KEY, None)
        command_manager = getattr(getattr(self, "env", None), "command_manager", None)
        dataset = getattr(command_manager, "dataset", None)
        num_motions = getattr(dataset, "num_motions", None)
        if callable(num_motions):
            num_motions = num_motions()
        if num_motions is not None:
            if isinstance(num_motions, bool):
                raise RuntimeError("Command dataset has an invalid motion count")
            num_motions = int(num_motions)
            if num_motions < 1:
                raise RuntimeError("Command dataset must contain at least one motion")
        if raw_motion_id is None:
            if num_motions != 1:
                raise KeyError(
                    "Bottleneck tracking requires replay_motion_id for multi-motion "
                    "or unknown-motion rollouts"
                )
            motion_id = torch.zeros_like(phase, dtype=torch.long)
        else:
            raw_motion_id = raw_motion_id.reshape(num_envs, num_steps, -1)
            if raw_motion_id.shape[-1] != 1:
                raise ValueError("replay_motion_id must have one scalar per step")
            raw_motion_id = raw_motion_id[..., 0]
            if raw_motion_id.dtype == torch.bool:
                raise ValueError("replay_motion_id cannot be boolean")
            if raw_motion_id.is_floating_point():
                if not torch.isfinite(raw_motion_id).all() or not torch.equal(
                    raw_motion_id, raw_motion_id.round()
                ):
                    raise ValueError("replay_motion_id must contain finite integers")
            motion_id = raw_motion_id.long()
            if bool((motion_id < 0).any()):
                raise ValueError("replay_motion_id cannot be negative")
            if num_motions is not None and bool((motion_id >= num_motions).any()):
                raise ValueError("replay_motion_id exceeds the command dataset")
        student = grid(rollout[DAGGER_IS_STUDENT_ACTION_KEY], boolean=True)
        done = grid(rollout[DONE_KEY], boolean=True)
        terminated = grid(rollout[TERM_KEY], boolean=True)
        command_finished = grid(
            rollout["next", "stats", "command_finished"], boolean=True
        )
        time_limit = grid(rollout["next", "stats", "episode_time_limit"], boolean=True)
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError(
                "Bottleneck episode contains an unknown or non-done boundary cause"
            )
        timeout = time_limit & ~command_finished & ~terminated
        is_init_value = rollout.get("is_init", None)
        is_init = (
            torch.zeros_like(done)
            if is_init_value is None
            else grid(is_init_value, boolean=True)
        )
        if "step_count" not in rollout.keys():
            raise KeyError(
                "Bottleneck tracking requires step_count to match replay burn-in"
            )
        step_count = grid(rollout["step_count"])
        if not torch.isfinite(step_count.float()).all():
            raise RuntimeError("Bottleneck step_count contains NaN/Inf")
        replay_valid = step_count > DAGGER_REPLAY_MIN_STEP_COUNT
        residual = self._student_teacher_td_residual_grid(rollout, student)
        true_terminal = done & terminated
        packed = torch.stack(
            (
                phase.float(),
                residual.float(),
                student.float(),
                done.float(),
                true_terminal.float(),
                timeout.float(),
                command_finished.float(),
                is_init.float(),
                replay_valid.float(),
            ),
            dim=-1,
        ).detach()
        if packed.device.type != "cpu":
            packed = packed.cpu()
        motion_id = motion_id.detach()
        if motion_id.device.type != "cpu":
            motion_id = motion_id.cpu()
        # Convert once instead of extracting nine CPU Tensor scalars for every
        # env/step cell below. On the production 512x32 rollout this removes
        # more than 150k tiny Python/Tensor dispatches while retaining the
        # exact env-major history order and float32 scalar values.
        packed_rows = packed.tolist()
        motion_id_rows = motion_id.tolist()

        histories = self._bottleneck_episode_histories
        if histories is None or len(histories) != num_envs:
            histories = [
                self._new_bottleneck_episode_history() for _ in range(num_envs)
            ]
        env_episode_ids = getattr(self, "_student_replay_env_episode_ids", None)
        env_episode_steps = getattr(self, "_student_replay_env_episode_steps", None)
        if env_episode_ids is None or len(env_episode_ids) != num_envs:
            env_episode_ids = [None for _ in range(num_envs)]
            env_episode_steps = [0 for _ in range(num_envs)]
        if env_episode_steps is None or len(env_episode_steps) != num_envs:
            raise RuntimeError("Student replay episode-step tracking is misaligned")
        episode_id_rows = [[-1] * num_steps for _ in range(num_envs)]
        episode_step_rows = [[-1] * num_steps for _ in range(num_envs)]
        include_timeout = bool(
            getattr(
                self.cfg,
                "bottleneck_include_unsuccessful_timeouts",
                False,
            )
        )
        anchors_added = 0
        for env_index in range(num_envs):
            history = histories[env_index]
            for step in range(num_steps):
                row = packed_rows[env_index][step]
                if bool(row[7]):
                    history = self._new_bottleneck_episode_history()
                    env_episode_ids[env_index] = None
                    env_episode_steps[env_index] = 0
                if env_episode_ids[env_index] is None:
                    next_episode_id = int(
                        getattr(self, "_student_replay_next_episode_id", 0)
                    )
                    env_episode_ids[env_index] = next_episode_id
                    self._student_replay_next_episode_id = next_episode_id + 1
                replay_episode_id = int(env_episode_ids[env_index])
                replay_episode_step = int(env_episode_steps[env_index])
                episode_id_rows[env_index][step] = replay_episode_id
                episode_step_rows[env_index][step] = replay_episode_step
                is_student = bool(row[2])
                history["phase"].append(float(row[0]))
                history["teacher_td_residual"].append(
                    float(row[1]) if is_student else 0.0
                )
                history["source_id"].append(
                    SOURCE_STUDENT if is_student else SOURCE_UNIFORM_TEACHER
                )
                history["motion_id"].append(int(motion_id_rows[env_index][step]))
                history["true_terminal"].append(bool(row[4]))
                history["timeout"].append(bool(row[5]))
                history["replay_valid"].append(bool(row[8]))

                if bool(row[3]) and not bool(row[6]):
                    self._bottleneck_unsuccessful_episode_count += 1
                    outcome_is_eligible = not bool(row[5]) or include_timeout
                    if outcome_is_eligible and SOURCE_STUDENT in history["source_id"]:
                        anchors_added += self._process_failed_student_episode(
                            history, replay_episode_id=replay_episode_id
                        )
                env_episode_steps[env_index] = replay_episode_step + 1
                if bool(row[3]):
                    history = self._new_bottleneck_episode_history()
                    env_episode_ids[env_index] = None
                    env_episode_steps[env_index] = 0
            histories[env_index] = history
        episode_id_grid = torch.tensor(episode_id_rows, dtype=torch.long)
        episode_step_grid = torch.tensor(episode_step_rows, dtype=torch.long)
        self._bottleneck_episode_histories = histories
        self._student_replay_env_episode_ids = env_episode_ids
        self._student_replay_env_episode_steps = env_episode_steps
        self._student_replay_episode_id_grid = episode_id_grid
        self._student_replay_episode_step_grid = episode_step_grid
        return anchors_added

    @staticmethod
    def _student_focus_event_keys(
        events: list[tuple[int, torch.Tensor]] | tuple[tuple[int, torch.Tensor], ...],
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Encode episode/step pairs without retaining raw episode payloads."""
        stride = 1 << 32
        keys = []
        for episode_id, steps in events:
            steps = torch.as_tensor(steps, dtype=torch.long, device=device)
            if bool((steps < 0).any()) or bool((steps >= stride).any()):
                raise ValueError("Student replay episode step is outside uint32 range")
            keys.append(steps + int(episode_id) * stride)
        if not keys:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(keys).unique(sorted=True)

    @staticmethod
    def _student_focus_matches(
        episode_ids: torch.Tensor,
        episode_steps: torch.Tensor,
        event_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride = 1 << 32
        episode_ids = episode_ids.long()
        episode_steps = episode_steps.long()
        valid = (episode_ids >= 0) & (episode_steps >= 0)
        row_keys = episode_ids * stride + episode_steps
        matched = torch.zeros_like(valid)
        if event_keys.numel() and bool(valid.any()):
            sorted_events = event_keys.to(row_keys.device)
            valid_keys = row_keys[valid]
            positions = torch.searchsorted(sorted_events, valid_keys)
            safe_positions = positions.clamp_max(sorted_events.numel() - 1)
            valid_matches = (positions < sorted_events.numel()) & (
                sorted_events.index_select(0, safe_positions) == valid_keys
            )
            matched[valid] = valid_matches
        return matched, row_keys

    def _get_student_q_n_step_accumulator(self) -> _Stage1NStepAccumulator:
        """Lazily build the vector-environment Student return accumulator."""
        configured = int(getattr(self.cfg, "q_n_step", 1))
        accumulator = getattr(self, "_student_q_n_step_accumulator", None)
        if accumulator is None:
            endpoint_fields = (
                "next_critic_observations",
                NEXT_REFERENCE_PHASE_KEY,
                NEXT_Q_ACTUATOR_CONTEXT_KEY,
                STUDENT_COLLECTION_NEXT_ACTOR_OBSERVATIONS_KEY,
                REPLAY_TEACHER_V_NEXT_KEY,
                REPLAY_TERMINATED_KEY,
                REPLAY_COMMAND_FINISHED_KEY,
                REPLAY_TIME_LIMIT_KEY,
                _PREFILL_TERMINATED_KEY,
                _PREFILL_COMMAND_FINISHED_KEY,
            )
            accumulator = _Stage1NStepAccumulator(
                configured,
                float(self.cfg.gamma),
                next_fields=endpoint_fields,
            )
            self._student_q_n_step_accumulator = accumulator
        elif accumulator.n_steps != configured:
            raise RuntimeError("q_n_step changed while Student returns were pending")
        return accumulator

    def _aggregate_student_q_n_step_chunk(
        self,
        transitions: dict[str, torch.Tensor],
        *,
        num_envs: int,
    ) -> dict[str, torch.Tensor]:
        """Densify a filtered step and emit completed factual Student starts."""
        if int(getattr(self.cfg, "q_n_step", 1)) == 1:
            result = dict(transitions)
            result[Q_EFFECTIVE_N_STEPS_KEY] = torch.ones_like(
                transitions["rewards"]
            )
            return result

        env_indices = transitions[_PREFILL_ENV_INDEX_KEY].reshape(-1).long()
        row_count = int(transitions["rewards"].shape[0])
        if env_indices.shape != (row_count,):
            raise ValueError("n-step replay environment indices are misaligned")
        if DAGGER_IS_STUDENT_ACTION_KEY not in transitions:
            raise KeyError(
                "multi-step Student replay requires executed-action provenance"
            )
        student_rows = transitions[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
        if student_rows.shape != (row_count,):
            raise ValueError("n-step Student provenance is misaligned")
        valid = torch.zeros(
            int(num_envs),
            dtype=torch.bool,
            device=transitions["rewards"].device,
        )
        if row_count:
            valid.index_copy_(
                0,
                env_indices.to(valid.device),
                student_rows.to(valid.device),
            )

        dense: dict[str, torch.Tensor] = {}
        for name, value in transitions.items():
            if int(value.shape[0]) != row_count:
                raise ValueError(f"n-step replay field {name!r} is misaligned")
            full = value.new_empty((int(num_envs), *value.shape[1:]))
            if row_count:
                full.index_copy_(0, env_indices.to(value.device), value)
            dense[name] = full
        return self._get_student_q_n_step_accumulator().append(dense, valid)

    def _dagger_transition_chunks(self, td: TensorDict):
        """Attach stable episode IDs and exact failed-bottleneck eligibility."""
        q_n_step = int(getattr(getattr(self, "cfg", None), "q_n_step", 1))
        num_envs = int(td.batch_size[0]) if q_n_step > 1 else None
        teacher_prefill_active = (
            self._teacher_prefill_active() if q_n_step > 1 else False
        )
        pending_events = getattr(self, "_pending_student_focus_events", [])
        events = tuple(pending_events)
        event_keys_cpu = self._student_focus_event_keys(events, device="cpu")
        found_keys: list[torch.Tensor] = []
        deferred_found_keys: dict[
            torch.device, list[tuple[torch.Tensor, torch.Tensor]]
        ] = {}

        online_replays = [self.dagger_replay]
        if self._uses_separate_online_replays():
            online_replays.append(self.student_replay)
        for replay in online_replays:
            if (
                event_keys_cpu.numel()
                and STUDENT_REPLAY_EPISODE_ID_KEY in replay.data
                and STUDENT_REPLAY_EPISODE_STEP_KEY in replay.data
                and FAILURE_PHASE_STUDENT_SOURCE_KEY in replay.data
                and replay.size
            ):
                matched, row_keys = self._student_focus_matches(
                    replay.data[STUDENT_REPLAY_EPISODE_ID_KEY][: replay.size],
                    replay.data[STUDENT_REPLAY_EPISODE_STEP_KEY][: replay.size],
                    event_keys_cpu.to(replay.device),
                )
                replay.data[FAILURE_PHASE_STUDENT_SOURCE_KEY][
                    : replay.size
                ].logical_or_(matched)
                if bool(matched.any()):
                    found_keys.append(row_keys[matched].detach().cpu().unique())
                replay._valid_index_cache.pop(
                    FAILURE_PHASE_STUDENT_SOURCE_KEY, None
                )

        id_grid = self._student_replay_episode_id_grid
        step_grid = self._student_replay_episode_step_grid
        needs_teacher_value_cache = self._needs_teacher_value_cache()
        v_current_grid = getattr(self, "_rollout_teacher_v_current_grid", None)
        v_next_grid = getattr(self, "_rollout_teacher_v_next_grid", None)
        # Rollout transitions live on one device in production. Keep a small
        # per-device cache for test seams while ensuring CPU tracking grids and
        # event keys cross the device boundary at most once, never once per
        # rollout step.
        annotation_tensors: dict[
            torch.device,
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor],
        ] = {}
        value_grids: dict[
            torch.device, tuple[torch.Tensor | None, torch.Tensor | None]
        ] = {}
        try:
            for transitions in super()._dagger_transition_chunks(td):
                row_count = int(transitions["actions"].shape[0])
                transition_device = transitions["actions"].device
                annotation = annotation_tensors.get(transition_device)
                if annotation is None:
                    annotation = (
                        None if id_grid is None else id_grid.to(transition_device),
                        None if step_grid is None else step_grid.to(transition_device),
                        event_keys_cpu.to(transition_device),
                    )
                    annotation_tensors[transition_device] = annotation
                id_grid_device, step_grid_device, event_keys_device = annotation

                env_indices = transitions[_PREFILL_ENV_INDEX_KEY].long()
                rollout_steps = transitions[_PREFILL_STEP_INDEX_KEY].long()
                if env_indices.device != transition_device:
                    env_indices = env_indices.to(transition_device)
                if rollout_steps.device != transition_device:
                    rollout_steps = rollout_steps.to(transition_device)

                if id_grid_device is None or step_grid_device is None:
                    episode_ids = torch.full(
                        (row_count,),
                        -1,
                        dtype=torch.long,
                        device=transition_device,
                    )
                    episode_steps = torch.full_like(episode_ids, -1)
                else:
                    episode_ids = id_grid_device[env_indices, rollout_steps]
                    episode_steps = step_grid_device[env_indices, rollout_steps]
                matched, row_keys = self._student_focus_matches(
                    episode_ids,
                    episode_steps,
                    event_keys_device,
                )
                if event_keys_device.numel():
                    deferred_found_keys.setdefault(transition_device, []).append(
                        (row_keys.detach().clone(), matched.detach().clone())
                    )
                transitions[STUDENT_REPLAY_EPISODE_ID_KEY] = episode_ids
                transitions[STUDENT_REPLAY_EPISODE_STEP_KEY] = episode_steps
                transitions[FAILURE_PHASE_STUDENT_SOURCE_KEY] = matched
                # Attach Teacher-value cache fields for every transition so
                # both online provenance rings allocate one complete schema.
                if needs_teacher_value_cache:
                    target_device = transitions["rewards"].device
                    cached_value_grids = value_grids.get(target_device)
                    if cached_value_grids is None:
                        cached_value_grids = (
                            None
                            if v_current_grid is None
                            else v_current_grid.to(
                                device=target_device, dtype=torch.float32
                            ),
                            None
                            if v_next_grid is None
                            else v_next_grid.to(
                                device=target_device, dtype=torch.float32
                            ),
                        )
                        value_grids[target_device] = cached_value_grids
                    v_current_grid_device, v_next_grid_device = cached_value_grids
                    if (
                        v_current_grid_device is not None
                        and v_next_grid_device is not None
                    ):
                        cache_env_indices = env_indices
                        cache_steps = rollout_steps
                        if cache_env_indices.device != target_device:
                            cache_env_indices = cache_env_indices.to(target_device)
                        if cache_steps.device != target_device:
                            cache_steps = cache_steps.to(target_device)
                        v_current = v_current_grid_device[
                            cache_env_indices, cache_steps
                        ]
                        v_next = v_next_grid_device[cache_env_indices, cache_steps]
                    else:
                        zero = transitions["rewards"].new_zeros(
                            transitions["rewards"].shape[0], dtype=torch.float32
                        )
                        v_current = zero
                        v_next = zero
                    transitions[REPLAY_TEACHER_V_CURRENT_KEY] = v_current
                    transitions[REPLAY_TEACHER_V_NEXT_KEY] = v_next
                if q_n_step == 1 or teacher_prefill_active:
                    # Teacher episode staging and its recurrent next-state cache
                    # remain transition-aligned one-step contracts.
                    # Some schema-only unit seams intentionally omit rewards.
                    # One-step replay does not require this optional diagnostic
                    # field, while every production transition includes it.
                    if "rewards" in transitions:
                        transitions[Q_EFFECTIVE_N_STEPS_KEY] = torch.ones_like(
                            transitions["rewards"]
                        )
                    yield transitions
                else:
                    aggregated = self._aggregate_student_q_n_step_chunk(
                        transitions,
                        num_envs=int(num_envs),
                    )
                    # A failed-episode bottleneck can be discovered after an
                    # n-step start was buffered in the previous rollout. That
                    # start is not yet in the replay ring, and its old raw
                    # annotation predates the event. Match the completed start
                    # again here so cross-rollout pending rows retain exact
                    # retrospective focus eligibility.
                    if (
                        event_keys_device.numel()
                        and int(aggregated["actions"].shape[0])
                    ):
                        aggregate_matched, aggregate_row_keys = (
                            self._student_focus_matches(
                                aggregated[STUDENT_REPLAY_EPISODE_ID_KEY],
                                aggregated[STUDENT_REPLAY_EPISODE_STEP_KEY],
                                event_keys_device,
                            )
                        )
                        aggregated[
                            FAILURE_PHASE_STUDENT_SOURCE_KEY
                        ].logical_or_(aggregate_matched)
                        deferred_found_keys.setdefault(
                            transition_device, []
                        ).append(
                            (
                                aggregate_row_keys.detach().clone(),
                                aggregate_matched.detach().clone(),
                            )
                        )
                    yield aggregated
        finally:
            # Defer current-rollout device-to-host transfer until all chunks have
            # been annotated. This retains the exact union/count semantics while
            # replacing a possible transfer and scalar synchronization per step
            # with at most one transfer per transition device.
            for device_entries in deferred_found_keys.values():
                row_keys = torch.cat([entry[0] for entry in device_entries])
                matched = torch.cat([entry[1] for entry in device_entries])
                found_keys.append(row_keys[matched].detach().cpu())
            found = torch.cat(found_keys).unique().numel() if found_keys else 0
            requested = int(event_keys_cpu.numel())
            self._student_focus_rows_marked = int(
                getattr(self, "_student_focus_rows_marked", 0)
            ) + int(found)
            self._student_focus_rows_missing = int(
                getattr(self, "_student_focus_rows_missing", 0)
            ) + max(requested - int(found), 0)
            pending_events.clear()
            self._student_replay_episode_id_grid = None
            self._student_replay_episode_step_grid = None
            self._rollout_teacher_v_current_grid = None
            self._rollout_teacher_v_next_grid = None

    @staticmethod
    def _mean_optional_metric(
        metrics: list[dict[str, torch.Tensor]], key: str
    ) -> float:
        values = [
            torch.as_tensor(item[key]).detach().float()
            for item in metrics
            if key in item
        ]
        return 0.0 if not values else torch.stack(values).mean().item()

    def _failure_teacher_buffer_size(self) -> int:
        """Return the effective successful-Teacher pool behind focused replay."""
        if self._has_canonical_replay_mix():
            rows, _ = self._verified_teacher_focus_pool()
            return int(rows.numel())
        if not bool(getattr(self, "_teacher_phase_index_ready", False)):
            return 0
        histogram = getattr(self, "_failure_phase_histogram", None)
        if not torch.is_tensor(histogram) or not bool((histogram > 0).any()):
            return 0
        risk_bins = (histogram > 0).nonzero(as_tuple=False).squeeze(-1)
        teacher_bins = self._teacher_phase_nearest_nonempty.index_select(0, risk_bins)
        unique_bins = teacher_bins.unique()
        return sum(
            int(self._teacher_phase_bin_rows[int(index)].numel())
            for index in unique_bins
        )

    def _bottleneck_metrics(self) -> dict[str, float]:
        selected = max(self._bottleneck_selected_count, 1)
        match_count = max(self._bottleneck_phase_match_distance_count, 1)
        detected = max(self._bottleneck_detected_count, 1)
        failed_episodes = max(self._bottleneck_failed_student_episode_count, 1)
        return {
            "onset_to_terminal_distance_mean": (
                self._bottleneck_onset_to_terminal_sum / detected
            ),
            "failed_episode_length_mean": (
                self._bottleneck_failed_episode_length_sum / failed_episodes
            ),
            "unsuccessful_episode_count": float(
                self._bottleneck_unsuccessful_episode_count
            ),
            "episodes_with_student_candidates": float(
                self._bottleneck_episodes_with_student_candidates
            ),
            "value_detected_count": float(self._bottleneck_detected_count),
            "no_value_bottleneck_count": float(
                self._bottleneck_no_value_bottleneck_count
            ),
            "value_argmin_ablation_count": float(
                self._bottleneck_value_argmin_ablation_count
            ),
            "candidate_count": float(self._bottleneck_student_candidate_count),
            "failed_student_episode_count": float(
                self._bottleneck_failed_student_episode_count
            ),
            "student_candidate_count": float(self._bottleneck_student_candidate_count),
            "detected_count": float(self._bottleneck_detected_count),
            "fallback_count": float(self._bottleneck_fallback_count),
            "no_candidate_count": float(self._bottleneck_no_candidate_count),
            "selected_step_mean": self._bottleneck_selected_step_sum / selected,
            "selected_phase_mean": self._bottleneck_selected_phase_sum / selected,
            "score_mean": self._bottleneck_score_sum / selected,
            "score_max": float(self._bottleneck_score_max),
            "raw_td_residual_mean": (self._bottleneck_raw_td_residual_sum / selected),
            "raw_teacher_td_residual_mean": (
                self._bottleneck_raw_td_residual_sum / selected
            ),
            "smoothed_td_residual_mean": (
                self._bottleneck_smoothed_td_residual_sum / selected
            ),
            "smoothed_teacher_td_residual_mean": (
                self._bottleneck_smoothed_td_residual_sum / selected
            ),
            # The repository implements Failure Teacher as a virtual focused
            # source over the successful Teacher ring. These two values count
            # registered phase sequences/anchors, not copied replay tensors.
            "teacher_sequences_inserted": float(
                self._bottleneck_teacher_sequences_inserted
            ),
            "teacher_transitions_inserted": float(
                self._bottleneck_teacher_transitions_inserted
            ),
            "phase_match_distance_mean": (
                self._bottleneck_phase_match_distance_sum / match_count
            ),
            "teacher_phase_match_distance": (
                self._bottleneck_phase_match_distance_sum / match_count
            ),
            "failure_teacher_buffer_size": float(self._failure_teacher_buffer_size()),
            "student_focus_pool_size": float(
                self.dagger_replay.valid_count(FAILURE_PHASE_STUDENT_SOURCE_KEY)
                if FAILURE_PHASE_STUDENT_SOURCE_KEY in self.dagger_replay.data
                else 0
            ),
            "student_focus_rows_marked": float(self._student_focus_rows_marked),
            "student_focus_rows_missing": float(self._student_focus_rows_missing),
            "student_focus_sampled_rows": float(
                self._failure_phase_student_focused_rows
            ),
            "student_focus_uniform_fallback_rows": float(
                self._failure_phase_student_uniform_fallback_rows
            ),
            "student_focus_fraction_config": float(
                self._replay_mix_fractions("q")["failure_student"]
                / max(
                    self._replay_mix_fractions("q")["uniform_student"]
                    + self._replay_mix_fractions("q")["failure_student"],
                    torch.finfo(torch.float32).eps,
                )
            ),
            "student_focus_q_global_fraction_cap": float(
                self._replay_mix_fractions("q")["failure_student"]
            ),
            "student_focus_actor_global_fraction_cap": float(
                self._replay_mix_fractions("actor")["failure_student"]
            ),
        }

    def train_op(self, tensordict):
        info = DistributionalFastSACTeacherBC.train_op(self, tensordict)
        warmup_metric_names = {
            "student_actor_warmup_active": "active",
            "student_actor_warmup_completed_rollouts": "completed_rollouts",
            "student_actor_warmup_target_rollouts": "target_rollouts",
            "student_actor_warmup_complete": "complete",
            "prefill_perception_warmup_steps": "perception_updates",
            "prefill_perception_warmup_complete": "complete",
            "prefill_perception_warmup_priv_loss": "priv_loss",
            "prefill_perception_warmup_object_loss": "object_loss",
            "prefill_perception_warmup_grad_norm": "perception_grad_norm",
            "prefill_perception_warmup_skipped_frozen": (
                "perception_skipped_frozen"
            ),
        }
        for old_name, new_name in warmup_metric_names.items():
            old_key = f"fastsac/{old_name}"
            if old_key in info:
                info[f"warmup/{new_name}"] = info.pop(old_key)
        critic_metrics = getattr(self, "_fastsac_rollout_critic_metrics", [])
        actor_metrics = getattr(self, "_fastsac_rollout_actor_metrics", [])

        tvkd_mapping = {
            "teacher_value_mean": "tvkd_teacher_value_mean",
            "teacher_value_std": "tvkd_teacher_value_std",
            "teacher_next_value_mean": "tvkd_teacher_next_value_mean",
            "teacher_next_value_std": "tvkd_teacher_next_value_std",
            "teacher_potential_mean": "tvkd_teacher_potential_mean",
            "teacher_potential_std": "tvkd_teacher_potential_std",
            "teacher_next_potential_mean": "tvkd_teacher_next_potential_mean",
            "teacher_next_potential_std": "tvkd_teacher_next_potential_std",
            "potential_delta_mean": "tvkd_potential_delta_mean",
            "potential_delta_std": "tvkd_potential_delta_std",
            "potential_delta_min": "tvkd_potential_delta_min",
            "potential_delta_max": "tvkd_potential_delta_max",
            "raw_reward_mean": "tvkd_raw_reward_mean",
            "shaped_reward_mean": "tvkd_shaped_reward_mean",
            "shaped_reward_std": "tvkd_shaped_reward_std",
            "residual_reward_mean": "tvkd_residual_reward_mean",
            "residual_reward_std": "tvkd_residual_reward_std",
            "critic_reward_mean": "tvkd_critic_reward_mean",
            "critic_reward_std": "tvkd_critic_reward_std",
            "full_q_state_baseline_mean": "tvkd_full_q_state_baseline_mean",
            "full_q_state_baseline_std": "tvkd_full_q_state_baseline_std",
            "teacher_residual_critic_enabled": (
                "tvkd_teacher_residual_critic_enabled"
            ),
            "reference_phase_mean": "tvkd_reference_phase_mean",
            "next_reference_phase_mean": "tvkd_next_reference_phase_mean",
            "command_terminal_potential_abs_mean": (
                "tvkd_command_terminal_potential_abs_mean"
            ),
            "command_terminal_potential_abs_max": (
                "tvkd_command_terminal_potential_abs_max"
            ),
            "phase_faded_potential_enabled": (
                "tvkd_phase_faded_potential_enabled"
            ),
        }
        for output_name, metric_name in tvkd_mapping.items():
            info[f"tvkd/{output_name}"] = self._mean_optional_metric(
                critic_metrics, metric_name
            )
        for name, value in self._bottleneck_metrics().items():
            info[f"bottleneck/{name}"] = float(value)

        info["loss/critic"] = self._mean_optional_metric(critic_metrics, "critic_loss")
        info["loss/actor_total"] = self._mean_optional_metric(
            actor_metrics, "total_actor_loss"
        )
        info["loss/actor_sac"] = self._mean_optional_metric(
            actor_metrics, "sac_actor_loss"
        )
        info["loss/bc"] = self._mean_optional_metric(actor_metrics, "exact_bc_loss")
        info["loss/fixed_bc_coefficient"] = float(self.cfg.lambda_bc)
        info["loss/alpha"] = self._mean_optional_metric(critic_metrics, "alpha_loss")
        info["source/student_transition_count"] = float(
            info.get("fastsac/student_replay_rows_this_rollout", 0.0)
        )
        info["source/student_action_execution_ratio"] = float(
            info.get("fastsac/student_source_fraction", 0.0)
        )
        info["source/teacher_action_execution_ratio"] = float(
            info.get("fastsac/teacher_source_fraction", 0.0)
        )
        info["source/actor_bottleneck_student_fraction"] = self._mean_optional_metric(
            actor_metrics, "actor_failure_phase_student_fraction"
        )
        info["tvkd/student_q_n_step_config"] = float(
            getattr(self.cfg, "q_n_step", 1)
        )
        info["tvkd/teacher_q_n_step_config"] = float(
            getattr(self.cfg, "q_teacher_n_step", 1)
        )
        info["tvkd/effective_q_n_steps"] = float(
            info.get("fastsac/effective_n_steps_mean", 0.0)
        )
        info["tvkd/method_distributional_tvkd_fastsac_teacher_bc_v9"] = 1.0
        self._last_tvkd_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if (
                key.startswith("tvkd/")
                or key.startswith("bottleneck/")
                or key.startswith("loss/")
                or key.startswith("source/")
            )
            and isinstance(value, (int, float))
        }
        return info

    def _checkpoint_config(self):
        common = DistributionalFastSACTeacherBC._checkpoint_config(self)
        common.update(
            {
                name: getattr(self.cfg, name)
                for name in (
                    "use_tvkd_value_shaping",
                    "use_teacher_residual_critic",
                    "use_phase_faded_teacher_potential",
                    "tvkd_lambda",
                    "tvkd_potential_clip",
                    "use_teacher_value_bottleneck_replay",
                    "bottleneck_threshold",
                    "bottleneck_smoothing_window",
                    "bottleneck_min_consecutive",
                    "bottleneck_terminal_exclusion_steps",
                    "bottleneck_fallback_mode",
                    "bottleneck_selection_mode",
                    "bottleneck_include_unsuccessful_timeouts",
                    "max_teacher_phase_match_distance",
                    "perception_replay_mode",
                    "teacher_value_return_semantics",
                    "teacher_value_boundary_semantics",
                    "teacher_value_potential_semantics",
                    "teacher_value_reward_group_fingerprint",
                    "replay_task_fingerprint",
                    "failure_phase_student_fraction",
                    "value_norm",
                )
            }
        )
        for purpose in ("q", "actor", "perception"):
            for source in (
                "uniform_student",
                "failure_student",
                "uniform_teacher",
                "failure_teacher",
            ):
                name = f"{purpose}_{source}_fraction"
                common[name] = getattr(self.cfg, name)
        # The inherited checkpoint surfaces historically selected only a
        # subset of PPO/DAgger fields. V4 direct/programmatic resume promises
        # the same exact algorithm contract as the CLI, so retain every
        # behavior-affecting structured-config field. Identity/source-location
        # fields do not alter the already embedded policy and are excluded.
        non_behavioral = {
            "_target_",
            "name",
            "checkpoint_path",
            "resume_allow_lambda_bc_override",
        }
        for field in dataclass_fields(TVKDDistributionalFastSACTeacherBCConfig):
            if field.name in non_behavioral:
                continue
            value = copy.deepcopy(getattr(self.cfg, field.name))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = tuple(value)
            common.setdefault(field.name, value)
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "teacher_value_semantics": _tvkd_critic_learning_semantics(
                    self.cfg
                ),
                "teacher_value_potential_semantics": (
                    _tvkd_teacher_value_potential_semantics(self.cfg)
                ),
                "bc_loss": "fixed_joint_valid_teacher_label_normalized_smooth_l1",
                "teacher_value_cache_semantics": (
                    TEACHER_VALUE_CACHE_SEMANTICS
                    if self._needs_teacher_value_cache()
                    else "disabled"
                ),
            }
        )
        return common

    def _q_backend_metadata(self):
        metadata = DistributionalFastSACTeacherBC._q_backend_metadata(self)
        metadata.update(
            {
                "target_semantics": _tvkd_critic_learning_semantics(self.cfg),
                "teacher_value_source": "frozen_checkpoint_ppo_critic",
                "teacher_value_output_scale": "valuenorm_denormalized_sum_groups",
                "teacher_value_norm_enabled": bool(self.cfg.value_norm),
                "teacher_value_potential_semantics": (
                    _tvkd_teacher_value_potential_semantics(self.cfg)
                ),
                "q_n_step_semantics": Q_N_STEP_SEMANTICS,
                "student_q_n_step": int(getattr(self.cfg, "q_n_step", 1)),
                "teacher_q_n_step": int(
                    getattr(self.cfg, "q_teacher_n_step", 1)
                ),
                "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "bottleneck_selection_mode": str(
                    getattr(self.cfg, "bottleneck_selection_mode", "first")
                ),
            }
        )
        return metadata

    def _checkpoint_q_backend_metadata(self) -> dict:
        """Return the complete Q contract, including for minimal test policies."""
        if hasattr(self, "q_actor_keys"):
            return self._q_backend_metadata()
        metadata = {
            "target_semantics": _tvkd_critic_learning_semantics(self.cfg),
            "q_twin_reduction": _fastsac_q_twin_reduction(self.cfg),
            "q_action_input_dim": int(
                getattr(self, "_q_action_input_dim", getattr(self, "action_dim", 0))
            ),
            "q_actuator_context": copy.deepcopy(
                getattr(
                    self,
                    "_q_actuator_context_metadata_value",
                    {"enabled": False},
                )
            ),
            "q_action_feature_semantics": self._q_action_feature_semantics(),
            "q_predicted_effect": self._q_predicted_effect_metadata(),
            "q_residual_film": self._q_residual_film_metadata(),
            "teacher_value_potential_semantics": (
                _tvkd_teacher_value_potential_semantics(self.cfg)
            ),
            "q_n_step_semantics": Q_N_STEP_SEMANTICS,
            "student_q_n_step": int(getattr(self.cfg, "q_n_step", 1)),
            "teacher_q_n_step": int(getattr(self.cfg, "q_teacher_n_step", 1)),
            "bc_weighting_semantics": (
                SPRED_P_BC_SEMANTICS
                if getattr(self.cfg, "use_q_filtered_bc", False)
                else "fixed_unweighted_valid_teacher_bc"
            ),
            "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
            "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
            "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
            "bottleneck_selection_mode": str(
                getattr(self.cfg, "bottleneck_selection_mode", "first")
            ),
        }
        if self._uses_standard_split_stem_q():
            hidden_dim = int(getattr(self.cfg, "q_hidden_dim", 768))
            state_hidden_dim = hidden_dim // 2
            q_architecture_semantics = (
                FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS
                if self._uses_standard_scalar_q()
                else FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS
            )
            metadata.update(
                {
                    "q_state_input_dim": int(
                        getattr(
                            self,
                            "_q_state_input_dim",
                            getattr(self, "_q_critic_dim", 0)
                            + getattr(self, "_q_actuator_context_dim", 0),
                        )
                    ),
                    "q_state_context_dim": int(
                        getattr(self, "_q_actuator_context_dim", 0)
                    ),
                    "q_input_semantics": (
                        FASTSAC_STANDARD_SCALAR_Q_INPUT_SEMANTICS
                    ),
                    "q_architecture_semantics": (
                        q_architecture_semantics
                    ),
                    "hidden_dim": hidden_dim,
                    "q_action_fusion": "balanced_split_stems",
                    "q_state_hidden_dim": state_hidden_dim,
                    "q_action_hidden_dim": hidden_dim - state_hidden_dim,
                    "q_action_fusion_semantics": (
                        FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS
                    ),
                }
            )
        return metadata

    def _frozen_teacher_module_names(self) -> tuple[str, ...]:
        names = ["actor", "encoder_priv", "critic", "value_norm"]
        if hasattr(self, "height_encoder"):
            names.append("height_encoder")
        return tuple(names)

    def _failure_curriculum_checkpoint_state(self) -> dict:
        histogram = getattr(self, "_failure_phase_histogram", None)
        if not torch.is_tensor(histogram):
            histogram = torch.zeros(
                int(self.cfg.failure_phase_num_bins), dtype=torch.float64
            )
        expected_shape = (int(self.cfg.failure_phase_num_bins),)
        if (
            histogram.dtype != torch.float64
            or histogram.shape != expected_shape
            or not torch.isfinite(histogram).all()
            or bool((histogram < 0.0).any())
            or not torch.equal(histogram, histogram.round())
        ):
            raise RuntimeError("Verified global histogram state is invalid")
        motion_histograms = getattr(
            self, "_verified_failure_motion_phase_histogram", {}
        )
        if not isinstance(motion_histograms, Mapping):
            raise RuntimeError("Verified motion histogram state must be a mapping")
        histogram = histogram.detach().to(device="cpu", dtype=torch.float64).clone()
        serialized_motion_histograms = {}
        for motion_id, value in motion_histograms.items():
            if (
                isinstance(motion_id, bool)
                or not isinstance(motion_id, int)
                or motion_id < 0
                or not torch.is_tensor(value)
                or value.dtype != torch.float64
                or value.shape != histogram.shape
                or not torch.isfinite(value).all()
                or bool((value < 0.0).any())
                or not torch.equal(value, value.round())
            ):
                raise RuntimeError("Verified motion histogram state is invalid")
            serialized_motion_histograms[motion_id] = (
                value.detach().to(device="cpu").clone()
            )
        motion_total = torch.zeros_like(histogram)
        for value in serialized_motion_histograms.values():
            motion_total.add_(value)
        if not torch.equal(motion_total, histogram):
            raise RuntimeError("Verified global/motion histograms disagree")
        counters = {}
        for name in (
            "episode_count",
            "anchor_count",
            "uniform_fallback_rows",
            "focused_rows",
        ):
            value = getattr(self, f"_failure_phase_{name}", 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Verified histogram {name} is invalid")
            counters[name] = value
        if int(histogram.sum().item()) != counters["anchor_count"]:
            raise RuntimeError("Verified histogram/anchor count mismatch")
        return {
            "semantics": VERIFIED_HISTOGRAM_SEMANTICS,
            "histogram": histogram,
            **counters,
            "motion_histograms": serialized_motion_histograms,
        }

    def _load_failure_curriculum_checkpoint_state(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
        if state.get("semantics") != VERIFIED_HISTOGRAM_SEMANTICS:
            raise ValueError("TVKD verified histogram semantics mismatch")
        histogram = state.get("histogram")
        expected_bins = int(self.cfg.failure_phase_num_bins)
        if (
            not torch.is_tensor(histogram)
            or histogram.dtype != torch.float64
            or histogram.shape != (expected_bins,)
            or not torch.isfinite(histogram).all()
            or (histogram < 0.0).any()
            or not torch.equal(histogram, histogram.round())
        ):
            raise ValueError("TVKD failure curriculum histogram is invalid")
        counters = {}
        for name in (
            "episode_count",
            "anchor_count",
            "uniform_fallback_rows",
            "focused_rows",
        ):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD failure curriculum {name} is invalid")
            counters[name] = int(value)
        if int(histogram.sum().item()) != counters["anchor_count"]:
            raise ValueError("TVKD failure curriculum histogram/anchor count mismatch")
        motion_histograms = state.get("motion_histograms")
        if not isinstance(motion_histograms, Mapping):
            raise ValueError("TVKD verified histogram lacks motion partitions")
        restored_motion_histograms: dict[int, torch.Tensor] = {}
        for motion_id, value in motion_histograms.items():
            if (
                isinstance(motion_id, bool)
                or not isinstance(motion_id, int)
                or motion_id < 0
                or not torch.is_tensor(value)
                or value.dtype != torch.float64
                or value.shape != (expected_bins,)
                or not torch.isfinite(value).all()
                or bool((value < 0.0).any())
                or not torch.equal(value, value.round())
            ):
                raise ValueError("TVKD verified motion histogram is invalid")
            restored_motion_histograms[int(motion_id)] = (
                value.detach().to(device="cpu").clone()
            )
        motion_total = torch.zeros_like(histogram)
        for value in restored_motion_histograms.values():
            motion_total.add_(value)
        if not torch.equal(motion_total, histogram):
            raise ValueError("TVKD verified motion/global histogram counts disagree")
        self._failure_phase_histogram = histogram.detach().to(device="cpu").clone()
        self._failure_phase_episode_count = counters["episode_count"]
        self._failure_phase_anchor_count = counters["anchor_count"]
        self._failure_phase_uniform_fallback_rows = counters["uniform_fallback_rows"]
        self._failure_phase_focused_rows = counters["focused_rows"]
        self._verified_failure_motion_phase_histogram = restored_motion_histograms
        self._failure_histogram_device_cache.clear()

    def _reset_failure_curriculum_state(self) -> None:
        self._failure_phase_histogram = torch.zeros(
            int(self.cfg.failure_phase_num_bins), dtype=torch.float64
        )
        self._failure_phase_episode_count = 0
        self._failure_phase_anchor_count = 0
        self._failure_phase_uniform_fallback_rows = 0
        self._failure_phase_focused_rows = 0
        self._verified_failure_motion_phase_histogram = {}
        self._failure_histogram_device_cache.clear()

    def _bottleneck_replay_checkpoint_state(self) -> dict:
        return {
            "location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
            "detector": self.teacher_value_bottleneck_detector.state_dict(),
            "unsuccessful_episode_count": int(
                self._bottleneck_unsuccessful_episode_count
            ),
            "episodes_with_student_candidates": int(
                self._bottleneck_episodes_with_student_candidates
            ),
            "no_value_bottleneck_count": int(
                self._bottleneck_no_value_bottleneck_count
            ),
            "value_argmin_ablation_count": int(
                self._bottleneck_value_argmin_ablation_count
            ),
            "failed_student_episode_count": int(
                self._bottleneck_failed_student_episode_count
            ),
            "student_candidate_count": int(self._bottleneck_student_candidate_count),
            "detected_count": int(self._bottleneck_detected_count),
            "fallback_count": int(self._bottleneck_fallback_count),
            "no_candidate_count": int(self._bottleneck_no_candidate_count),
            "selected_count": int(self._bottleneck_selected_count),
            "teacher_sequences_inserted": int(
                self._bottleneck_teacher_sequences_inserted
            ),
            "teacher_transitions_inserted": int(
                self._bottleneck_teacher_transitions_inserted
            ),
            "phase_match_distance_count": int(
                self._bottleneck_phase_match_distance_count
            ),
            "next_student_episode_id": int(self._bottleneck_next_student_episode_id),
            "student_focus_rows_marked": int(self._student_focus_rows_marked),
            "student_focus_rows_missing": int(self._student_focus_rows_missing),
            "student_focus_sampled_rows": int(self._failure_phase_student_focused_rows),
            "student_focus_uniform_fallback_rows": int(
                self._failure_phase_student_uniform_fallback_rows
            ),
            "selected_step_sum": float(self._bottleneck_selected_step_sum),
            "selected_phase_sum": float(self._bottleneck_selected_phase_sum),
            "score_sum": float(self._bottleneck_score_sum),
            "score_max": float(self._bottleneck_score_max),
            "raw_td_residual_sum": float(self._bottleneck_raw_td_residual_sum),
            "smoothed_td_residual_sum": float(
                self._bottleneck_smoothed_td_residual_sum
            ),
            "phase_match_distance_sum": float(
                self._bottleneck_phase_match_distance_sum
            ),
            "onset_to_terminal_sum": float(self._bottleneck_onset_to_terminal_sum),
            "failed_episode_length_sum": float(
                self._bottleneck_failed_episode_length_sum
            ),
            "last_metadata": copy.deepcopy(self._last_bottleneck_metadata),
            "last_value_argmin_metadata": copy.deepcopy(
                self._last_value_argmin_metadata
            ),
        }

    def _load_bottleneck_replay_checkpoint_state(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("TVKD checkpoint lacks bottleneck replay state")
        if state.get("location_semantics") != BOTTLENECK_LOCATION_SEMANTICS:
            raise ValueError("TVKD checkpoint bottleneck semantics mismatch")
        detector_state = state.get("detector")
        if not isinstance(detector_state, Mapping):
            raise ValueError("TVKD checkpoint lacks bottleneck detector state")
        self.teacher_value_bottleneck_detector.load_state_dict(detector_state)
        integer_fields = (
            "unsuccessful_episode_count",
            "episodes_with_student_candidates",
            "no_value_bottleneck_count",
            "value_argmin_ablation_count",
            "failed_student_episode_count",
            "student_candidate_count",
            "detected_count",
            "fallback_count",
            "no_candidate_count",
            "selected_count",
            "teacher_sequences_inserted",
            "teacher_transitions_inserted",
            "phase_match_distance_count",
            "next_student_episode_id",
            "student_focus_rows_marked",
            "student_focus_rows_missing",
            "student_focus_sampled_rows",
            "student_focus_uniform_fallback_rows",
        )
        integers = {}
        for name in integer_fields:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD bottleneck replay {name} is invalid")
            integers[name] = int(value)
        float_fields = (
            "selected_step_sum",
            "selected_phase_sum",
            "score_sum",
            "score_max",
            "raw_td_residual_sum",
            "smoothed_td_residual_sum",
            "phase_match_distance_sum",
        )
        floats = {
            name: _finite_scalar(f"bottleneck replay {name}", state.get(name))
            for name in float_fields
        }
        for name in (
            "selected_step_sum",
            "selected_phase_sum",
            "score_sum",
            "score_max",
            "phase_match_distance_sum",
        ):
            if floats[name] < 0.0:
                raise ValueError(f"TVKD bottleneck replay {name} is negative")
        # Added after the first v6 checkpoints shipped, so a missing key is a
        # legitimate older checkpoint rather than corruption.
        optional_floats = {
            name: _finite_scalar(f"bottleneck replay {name}", state.get(name, 0.0))
            for name in ("onset_to_terminal_sum", "failed_episode_length_sum")
        }
        for name, value in optional_floats.items():
            if value < 0.0:
                raise ValueError(f"TVKD bottleneck replay {name} is negative")
        metadata = state.get("last_metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("TVKD bottleneck replay metadata is invalid")
        argmin_metadata = state.get("last_value_argmin_metadata", {})
        if not isinstance(argmin_metadata, Mapping):
            raise ValueError("TVKD value-argmin metadata is invalid")

        self._bottleneck_unsuccessful_episode_count = integers[
            "unsuccessful_episode_count"
        ]
        self._bottleneck_episodes_with_student_candidates = integers[
            "episodes_with_student_candidates"
        ]
        self._bottleneck_no_value_bottleneck_count = integers[
            "no_value_bottleneck_count"
        ]
        self._bottleneck_value_argmin_ablation_count = integers[
            "value_argmin_ablation_count"
        ]
        self._bottleneck_failed_student_episode_count = integers[
            "failed_student_episode_count"
        ]
        self._bottleneck_student_candidate_count = integers["student_candidate_count"]
        self._bottleneck_detected_count = integers["detected_count"]
        self._bottleneck_fallback_count = integers["fallback_count"]
        self._bottleneck_no_candidate_count = integers["no_candidate_count"]
        self._bottleneck_selected_count = integers["selected_count"]
        self._bottleneck_teacher_sequences_inserted = integers[
            "teacher_sequences_inserted"
        ]
        self._bottleneck_teacher_transitions_inserted = integers[
            "teacher_transitions_inserted"
        ]
        self._bottleneck_phase_match_distance_count = integers[
            "phase_match_distance_count"
        ]
        self._bottleneck_next_student_episode_id = integers["next_student_episode_id"]
        self._student_focus_rows_marked = integers["student_focus_rows_marked"]
        self._student_focus_rows_missing = integers["student_focus_rows_missing"]
        self._failure_phase_student_focused_rows = integers[
            "student_focus_sampled_rows"
        ]
        self._failure_phase_student_uniform_fallback_rows = integers[
            "student_focus_uniform_fallback_rows"
        ]
        self._bottleneck_onset_to_terminal_sum = optional_floats[
            "onset_to_terminal_sum"
        ]
        self._bottleneck_failed_episode_length_sum = optional_floats[
            "failed_episode_length_sum"
        ]
        self._bottleneck_selected_step_sum = floats["selected_step_sum"]
        self._bottleneck_selected_phase_sum = floats["selected_phase_sum"]
        self._bottleneck_score_sum = floats["score_sum"]
        self._bottleneck_score_max = floats["score_max"]
        self._bottleneck_raw_td_residual_sum = floats["raw_td_residual_sum"]
        self._bottleneck_smoothed_td_residual_sum = floats["smoothed_td_residual_sum"]
        self._bottleneck_phase_match_distance_sum = floats["phase_match_distance_sum"]
        self._last_bottleneck_metadata = copy.deepcopy(dict(metadata))
        self._last_value_argmin_metadata = copy.deepcopy(dict(argmin_metadata))

    def _fastsac_checkpoint_state(self):
        # The baseline seam intentionally omits the frozen PPO modules because
        # baseline same-stage replay resume is disabled. TVKD's value function
        # is itself part of the learning rule, so persist its exact frozen
        # checkpoint chain alongside the normal SAC state.
        state = DistributionalFastSACTeacherBC._fastsac_checkpoint_state(self)
        optimizer_state = state.get("optimizer_resume_state")
        if not isinstance(optimizer_state, dict):
            raise RuntimeError("FastSAC checkpoint did not expose optimizer state")
        if bool(self.cfg.train_dr_estimator):
            optimizer = getattr(self, "opt_dr_estimator", None)
            if optimizer is None:
                raise RuntimeError("trainable DR estimator has no optimizer")
            optimizer_state["dr_estimator_optimizer"] = optimizer.state_dict()
        else:
            optimizer_state["dr_estimator_optimizer"] = None
        replay_mix_state = {
            purpose: {
                source: float(getattr(self.cfg, f"{purpose}_{source}_fraction"))
                for source in (
                    "uniform_student",
                    "failure_student",
                    "uniform_teacher",
                    "failure_teacher",
                )
            }
            for purpose in ("q", "actor", "perception")
        }
        vecnorm_fingerprint = getattr(self, "_replay_vecnorm_fingerprint", None)
        if not isinstance(vecnorm_fingerprint, str) or not vecnorm_fingerprint:
            raise RuntimeError("TVKD checkpoint lacks its Teacher VecNorm fingerprint")
        reward_fingerprint = _required_contract_fingerprint(
            "Teacher reward-group fingerprint",
            getattr(self.cfg, "teacher_value_reward_group_fingerprint", None),
        )
        task_fingerprint = _required_contract_fingerprint(
            "replay task fingerprint",
            getattr(self.cfg, "replay_task_fingerprint", None),
        )
        verified_histogram_state = self._failure_curriculum_checkpoint_state()
        critic_semantics = _tvkd_critic_learning_semantics(self.cfg)
        potential_semantics = _tvkd_teacher_value_potential_semantics(self.cfg)
        state.update(
            {
                "training_algorithm": TRAINING_ALGORITHM,
                "checkpoint_version": CHECKPOINT_VERSION,
                "critic_learning_semantics": critic_semantics,
                "actor_learning_semantics": _tvkd_actor_learning_semantics(
                    self.cfg
                ),
                # Preserve the complete base Q topology/action contract, then
                # apply TVKD's target/value/n-step overrides in its metadata
                # method. Evaluation and resume must reconstruct the expanded
                # predicted-effect action stem before loading its weights.
                "q_backend_config": self._checkpoint_q_backend_metadata(),
                "replay_mix_state": replay_mix_state,
                "perception_replay_mode": str(self.cfg.perception_replay_mode),
                "perception_live_env_scope": str(
                    self.cfg.perception_live_env_scope
                ),
                "perception_training_semantics": (
                    online_rollout_perception_semantics(self.cfg)
                    if str(self.cfg.perception_replay_mode)
                    == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "actor_replay_observation_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                ),
                "teacher_episode_sidecar_semantics": (
                    TEACHER_EPISODE_SIDECAR_SEMANTICS
                ),
                "object_geo_replay_semantics": OBJECT_GEO_REPLAY_SEMANTICS,
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "bottleneck_selection_mode": str(
                    getattr(self.cfg, "bottleneck_selection_mode", "first")
                ),
                "teacher_value_return_semantics": str(
                    self.cfg.teacher_value_return_semantics
                ),
                "teacher_value_boundary_semantics": str(
                    self.cfg.teacher_value_boundary_semantics
                ),
                "teacher_value_potential_semantics": str(
                    potential_semantics
                ),
                "teacher_value_gamma": float(self.cfg.gamma),
                "teacher_value_reward_group_fingerprint": reward_fingerprint,
                "teacher_value_vecnorm_fingerprint": vecnorm_fingerprint,
                "replay_task_fingerprint": task_fingerprint,
                "verified_teacher_value_histogram_state": (verified_histogram_state),
                "fresh_ring_resume_semantics": FRESH_RING_RESUME_SEMANTICS,
                "teacher_value_bottleneck_replay_state": (
                    self._bottleneck_replay_checkpoint_state()
                ),
                "frozen_teacher_state": {
                    name: copy.deepcopy(getattr(self, name).state_dict())
                    for name in self._frozen_teacher_module_names()
                },
                # Compatibility alias. The semantics sentinel prevents a v3
                # mixed-origin histogram from being mistaken for v5 verified data.
                "failure_phase_curriculum_state": verified_histogram_state,
                "num_updates": int(self.num_updates),
                "sac_actor_update_count": int(
                    getattr(
                        self,
                        "sac_actor_update_count",
                        getattr(self, "actor_update_count", 0),
                    )
                ),
                "sac_alpha_update_count": int(
                    getattr(
                        self,
                        "sac_alpha_update_count",
                        getattr(self, "alpha_update_count", 0),
                    )
                ),
                "last_tvkd_diagnostics": copy.deepcopy(self._last_tvkd_diagnostics),
            }
        )
        return state

    def _load_fastsac_checkpoint_state(self, state, *, load_modules=True):
        algorithm = state.get("training_algorithm")
        version = state.get("checkpoint_version", -1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("TVKD FastSAC checkpoint version is invalid")
        legacy = (
            algorithm == LEGACY_TRAINING_ALGORITHM
            and version == LEGACY_CHECKPOINT_VERSION
        )
        previous = (
            algorithm == PREVIOUS_TRAINING_ALGORITHM
            and version == PREVIOUS_CHECKPOINT_VERSION
        )
        v3 = algorithm == V3_TRAINING_ALGORITHM and version == V3_CHECKPOINT_VERSION
        v4 = algorithm == V4_TRAINING_ALGORITHM and version == V4_CHECKPOINT_VERSION
        v5 = algorithm == V5_TRAINING_ALGORITHM and version == V5_CHECKPOINT_VERSION
        v6 = algorithm == V6_TRAINING_ALGORITHM and version == V6_CHECKPOINT_VERSION
        v7 = algorithm == V7_TRAINING_ALGORITHM and version == V7_CHECKPOINT_VERSION
        v8 = algorithm == V8_TRAINING_ALGORITHM and version == V8_CHECKPOINT_VERSION
        current = algorithm == TRAINING_ALGORITHM and version == CHECKPOINT_VERSION
        if v4:
            raise ValueError(
                "TVKD v4 resume is incompatible with the v5 raw-residual "
                "pre-onset replay contract; start v5 from the frozen PPO source"
            )
        if v5:
            raise ValueError(
                "TVKD v5 training resume is incompatible with the v9 "
                "finite-horizon boundary, replay-sidecar, and Teacher-residual "
                "critic contracts; start v9 from the frozen PPO source. The v5 "
                "checkpoint remains supported for model-only inference."
            )
        if v6:
            raise ValueError(
                "TVKD v6 training resume is incompatible with the v8 physical-std "
                "optimizer contract; start v8 from the frozen PPO source. The v6 "
                "checkpoint remains supported for model-only inference."
            )
        if v7:
            raise ValueError(
                "TVKD v7 training resume is incompatible with the v8 physical-std "
                "SAC-gradient, separate-Adam, and rollout-KL contract; start v8 "
                "from the frozen PPO source. The v7 checkpoint remains supported "
                "for model-only inference."
            )
        if v8:
            raise ValueError(
                "TVKD v8 training resume is incompatible with the v9 finite-horizon "
                "command-terminal boundary and exact replay-sidecar contract; start "
                "v9 from the frozen PPO source. The v8 checkpoint remains supported "
                "for model-only inference."
            )
        if not (legacy or previous or v3 or current):
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        saved_object_geo_semantics = state.get("object_geo_replay_semantics")
        if (
            saved_object_geo_semantics is not None
            and saved_object_geo_semantics != OBJECT_GEO_REPLAY_SEMANTICS
        ):
            raise ValueError(
                "TVKD resume metadata mismatch at 'object_geo_replay_semantics'"
            )
        backend = state.get("dagger_backend_config")
        if not isinstance(backend, Mapping):
            raise ValueError("TVKD checkpoint lacks backend config")
        if current and "student_buffer_capacity" not in backend:
            # Early v9 checkpoints stored one total online capacity. Recreate
            # their exact DAgger/pure-Student allocation before enforcing the
            # current explicit two-capacity checkpoint schema.
            backend = _migrate_explicit_online_replay_capacities(backend)
        saved_lambda_bc = _finite_scalar(
            "checkpoint lambda_bc", backend.get("lambda_bc")
        )
        lambda_bc_fork = _lambda_bc_fork_override_active(
            self.cfg,
            state,
            saved_lambda_bc,
        )
        if legacy or previous or v3:
            # Programmatic loads must perform the same explicit schema
            # migration as the Hydra entrypoint. Perception adopts the PPOVEL
            # live Student rollout while the saved Q/Actor mix and alpha cadence
            # remain explicit.
            _install_v3_replay_migration(
                self.cfg,
                backend,
                student_focus_default=(0.0 if legacy or previous else None),
            )
        if current:
            contract_label = "v9"
            expected_backend = self._checkpoint_config()
            if "sac_action_distribution" not in backend:
                if (
                    state.get("actor_backend") != ACTOR_BACKEND
                    or expected_backend.get("sac_action_distribution")
                    != NORMALIZED_TANH_ACTION_DISTRIBUTION
                ):
                    raise ValueError(
                        f"TVKD {contract_label} checkpoint lacks its physical "
                        "action-distribution contract"
                    )
                backend = dict(backend)
                backend["sac_action_distribution"] = (
                    NORMALIZED_TANH_ACTION_DISTRIBUTION
                )
            # Checkpoints written before the joint-aware envelope used the
            # scalar physical interval.  Install that exact historical contract
            # rather than inheriting runtime values: an old optimizer may resume
            # only into the equivalent uniform mode, never silently into the new
            # q-normalized feasible set.
            legacy_physical_bound_defaults = {
                "sac_physical_std_bound_mode": "uniform_physical",
                "sac_physical_std_normalized_min": 0.02,
                "sac_physical_std_normalized_max": 0.11,
            }
            missing_physical_bound_fields = set(
                legacy_physical_bound_defaults
            ).difference(backend)
            if missing_physical_bound_fields:
                backend = dict(backend)
                for name in missing_physical_bound_fields:
                    backend[name] = legacy_physical_bound_defaults[name]
            # Checkpoints written before Actor decay was disentangled from the
            # Critic did not carry this field.  Their explicit migrated contract
            # is zero Actor decay; the baseline loader below also rewrites the
            # stale optimizer param-group value after restoring its moments.
            if "sac_actor_weight_decay" not in backend:
                backend = dict(backend)
                backend["sac_actor_weight_decay"] = 0.0
            if "use_q_filtered_bc" not in backend:
                backend = dict(backend)
                backend["use_q_filtered_bc"] = False
            if "q_critic_type" not in backend:
                backend = dict(backend)
                backend["q_critic_type"] = "c51"
            if "q_twin_reduction" not in backend:
                backend = dict(backend)
                backend["q_twin_reduction"] = "min"
            if "use_teacher_residual_critic" not in backend:
                backend = dict(backend)
                backend["use_teacher_residual_critic"] = False
            if "q_n_step" not in backend or "q_teacher_n_step" not in backend:
                backend = dict(backend)
                backend.setdefault("q_n_step", 1)
                backend.setdefault("q_teacher_n_step", 1)
            if "q_use_predicted_effect" not in backend:
                backend = dict(backend)
                backend.setdefault("q_use_predicted_effect", False)
            if (
                "q_use_residual_film" not in backend
                or "q_residual_film_scale" not in backend
            ):
                backend = dict(backend)
                backend.setdefault("q_use_residual_film", False)
                backend.setdefault("q_residual_film_scale", 0.1)
            # These knobs are measurement-only and were added after the first
            # v6 checkpoints.  They do not affect policy, critic, replay, or
            # optimizer semantics, so an older checkpoint may safely inherit
            # the current config's diagnostic defaults.
            legacy_optional_diagnostic_keys = frozenset(
                {
                    "perception_staleness_probe_num_envs",
                    "perception_staleness_probe_max_episodes",
                    "perception_staleness_probe_max_generation_age",
                    "perception_staleness_probe_interval",
                }
            )
            missing_optional = set(expected_backend).difference(backend) & (
                legacy_optional_diagnostic_keys
            )
            if missing_optional:
                backend = dict(backend)
                for name in missing_optional:
                    backend[name] = expected_backend[name]
            if set(backend) != set(expected_backend):
                missing = sorted(set(expected_backend).difference(backend))
                unexpected = sorted(set(backend).difference(expected_backend))
                raise ValueError(
                    "TVKD resume algorithm config keys mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for name, expected_value in expected_backend.items():
                if lambda_bc_fork and name == "lambda_bc":
                    continue
                if backend[name] != expected_value:
                    raise ValueError(f"TVKD resume config mismatch at {name!r}")

            expected_mix = _checkpoint_replay_mix(self.cfg)
            saved_mix = state.get("replay_mix_state")
            if not isinstance(saved_mix, Mapping):
                raise ValueError(
                    f"TVKD {contract_label} checkpoint lacks replay mix state"
                )
            normalized_mix: dict[str, dict[str, float]] = {}
            for purpose, expected in expected_mix.items():
                saved_purpose = saved_mix.get(purpose)
                if not isinstance(saved_purpose, Mapping):
                    raise ValueError(
                        f"TVKD {contract_label} checkpoint lacks {purpose!r} "
                        "replay mix"
                    )
                normalized_mix[purpose] = {}
                for source, expected_value in expected.items():
                    saved_value = _finite_scalar(
                        f"checkpoint {purpose} {source} fraction",
                        saved_purpose.get(source),
                    )
                    normalized_mix[purpose][source] = saved_value
                    if not math.isclose(
                        saved_value,
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"TVKD resume replay mix mismatch at {purpose}.{source}"
                        )
                allocate_source_counts(1, normalized_mix[purpose])

            critic_semantics = _tvkd_critic_learning_semantics(self.cfg)
            potential_semantics = _tvkd_teacher_value_potential_semantics(
                self.cfg
            )
            exact_metadata = {
                "critic_learning_semantics": critic_semantics,
                "actor_learning_semantics": (
                    V5_ACTOR_LEARNING_SEMANTICS
                    if v5
                    else _tvkd_actor_learning_semantics(self.cfg)
                ),
                "perception_replay_mode": str(self.cfg.perception_replay_mode),
                "perception_live_env_scope": str(
                    self.cfg.perception_live_env_scope
                ),
                "perception_training_semantics": (
                    online_rollout_perception_semantics(self.cfg)
                    if str(self.cfg.perception_replay_mode)
                    == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "bottleneck_selection_mode": str(
                    getattr(self.cfg, "bottleneck_selection_mode", "first")
                ),
                "teacher_value_return_semantics": str(
                    self.cfg.teacher_value_return_semantics
                ),
                "teacher_value_boundary_semantics": str(
                    self.cfg.teacher_value_boundary_semantics
                ),
                "teacher_value_potential_semantics": str(
                    potential_semantics
                ),
                "teacher_value_reward_group_fingerprint": str(
                    self.cfg.teacher_value_reward_group_fingerprint
                ),
                "teacher_value_vecnorm_fingerprint": getattr(
                    self, "_replay_vecnorm_fingerprint", None
                ),
                "replay_task_fingerprint": str(self.cfg.replay_task_fingerprint),
                "fresh_ring_resume_semantics": (
                    FRESH_RING_RESUME_SEMANTICS
                    if current
                    else V5_FRESH_RING_RESUME_SEMANTICS
                ),
            }
            if current:
                exact_metadata.update(
                    {
                        "actor_replay_observation_semantics": (
                            COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                        ),
                        "teacher_episode_sidecar_semantics": (
                            TEACHER_EPISODE_SIDECAR_SEMANTICS
                        ),
                    }
                )
            if "replay_resume_semantics" in state:
                exact_metadata["replay_resume_semantics"] = (
                    REPLAY_RESUME_SEMANTICS
                    if current
                    else V5_REPLAY_RESUME_SEMANTICS
                )
            _required_contract_fingerprint(
                "runtime Teacher reward-group fingerprint",
                exact_metadata["teacher_value_reward_group_fingerprint"],
            )
            _required_contract_fingerprint(
                "runtime replay task fingerprint",
                exact_metadata["replay_task_fingerprint"],
            )
            # Checkpoints written before the selection knob carry no such key;
            # their behavior was exactly what "first" reproduces.
            q_context_metadata = copy.deepcopy(
                getattr(
                    self,
                    "_q_actuator_context_metadata_value",
                    {"enabled": False},
                )
            )
            q_action_input_dim = int(
                getattr(
                    self,
                    "_q_action_input_dim",
                    getattr(self, "action_dim", 0),
                )
            )
            metadata_defaults = {
                "bottleneck_selection_mode": "first",
                "perception_live_env_scope": "pure_student",
                "bc_weighting_semantics": "fixed_unweighted_valid_teacher_bc",
                "q_action_input_dim": q_action_input_dim,
                "q_actuator_context": {"enabled": False},
                "q_action_feature_semantics": "normalized_issued_command_only_v1",
                "q_predicted_effect": {"enabled": False},
                "q_residual_film": {"enabled": False},
                "q_twin_reduction": "min",
                "q_n_step_semantics": Q_N_STEP_SEMANTICS,
                "student_q_n_step": 1,
                "teacher_q_n_step": 1,
            }
            for name, expected in exact_metadata.items():
                if not isinstance(expected, str) or not expected:
                    raise ValueError(f"TVKD runtime lacks required metadata {name!r}")
                if state.get(name, metadata_defaults.get(name)) != expected:
                    raise ValueError(f"TVKD resume metadata mismatch at {name!r}")
            q_backend = state.get("q_backend_config")
            if not isinstance(q_backend, Mapping):
                raise ValueError(
                    f"TVKD {contract_label} checkpoint lacks Q backend metadata"
                )
            expected_q_metadata = {
                "target_semantics": critic_semantics,
                "q_twin_reduction": _fastsac_q_twin_reduction(self.cfg),
                "q_action_input_dim": q_action_input_dim,
                "q_actuator_context": q_context_metadata,
                "q_action_feature_semantics": self._q_action_feature_semantics(),
                "q_predicted_effect": self._q_predicted_effect_metadata(),
                "q_residual_film": self._q_residual_film_metadata(),
                "teacher_value_potential_semantics": (
                    potential_semantics
                ),
                "q_n_step_semantics": Q_N_STEP_SEMANTICS,
                "student_q_n_step": int(getattr(self.cfg, "q_n_step", 1)),
                "teacher_q_n_step": int(
                    getattr(self.cfg, "q_teacher_n_step", 1)
                ),
                "bc_weighting_semantics": (
                    SPRED_P_BC_SEMANTICS
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "fixed_unweighted_valid_teacher_bc"
                ),
                "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "bottleneck_selection_mode": str(
                    getattr(self.cfg, "bottleneck_selection_mode", "first")
                ),
            }
            if self._uses_standard_split_stem_q():
                runtime_q_backend = self._checkpoint_q_backend_metadata()
                for name in (
                    "q_state_input_dim",
                    "q_state_context_dim",
                    "q_input_semantics",
                    "q_architecture_semantics",
                ):
                    expected_q_metadata[name] = runtime_q_backend[name]
            for name, expected in expected_q_metadata.items():
                if q_backend.get(name, metadata_defaults.get(name)) != expected:
                    raise ValueError(
                        f"TVKD resume Q backend metadata mismatch at {name!r}"
                    )
            saved_gamma = _finite_scalar(
                "checkpoint Teacher value gamma",
                state.get("teacher_value_gamma"),
            )
            if not math.isclose(
                saved_gamma,
                float(self.cfg.gamma),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("TVKD resume Teacher value gamma mismatch")
            for name in (
                "sac_alpha_update_cadence",
                "q_condition_on_actuator_state",
                "q_use_predicted_effect",
                "q_use_residual_film",
                "q_residual_film_scale",
                "perception_replay_mode",
                "perception_live_env_scope",
                "perception_replay_batch_size",
                "use_tvkd_value_shaping",
                "use_teacher_residual_critic",
                "use_phase_faded_teacher_potential",
                "use_teacher_value_bottleneck_replay",
                "bottleneck_threshold",
                "bottleneck_smoothing_window",
                "bottleneck_min_consecutive",
                "bottleneck_terminal_exclusion_steps",
                "bottleneck_fallback_mode",
                "bottleneck_include_unsuccessful_timeouts",
                "max_teacher_phase_match_distance",
            ):
                default = "pure_student" if name == "perception_live_env_scope" else None
                if backend.get(name, default) != getattr(self.cfg, name):
                    raise ValueError(f"TVKD resume config mismatch at {name!r}")
        if legacy:
            warnings.warn(
                "Migrating a TVKD v1 adaptive-BC checkpoint to fixed BC and "
                "Teacher-value bottleneck replay. The adaptive BC scheduler "
                "and terminal-lookback failure histogram are ignored; "
                "perception switches to the PPOVEL live Student rollout.",
                UserWarning,
                stacklevel=2,
            )
        elif previous:
            warnings.warn(
                "Migrating a TVKD v2 checkpoint to v3 Student bottleneck "
                "replay. The raw Student ring is rebuilt and the new focused "
                "Student counters start from zero; perception switches to the "
                "PPOVEL live Student rollout.",
                UserWarning,
                stacklevel=2,
            )
        elif v3:
            warnings.warn(
                "Migrating a TVKD v3 checkpoint to v5: the saved alpha "
                "cadence and model/optimizer state are retained, perception "
                "switches to the PPOVEL live Student rollout, and raw rings, "
                "detector statistics, and the mixed-origin histogram are reset.",
                UserWarning,
                stacklevel=2,
            )
        bottleneck_state = state.get("teacher_value_bottleneck_replay_state")
        if (current or previous or v3) and not isinstance(
            bottleneck_state, Mapping
        ):
            raise ValueError("TVKD checkpoint lacks bottleneck replay state")
        frozen_teacher_state = state.get("frozen_teacher_state")
        if not isinstance(frozen_teacher_state, Mapping):
            raise ValueError("TVKD checkpoint lacks frozen Teacher value state")
        failure_curriculum_state = (
            state.get("verified_teacher_value_histogram_state")
            if current
            else state.get("failure_phase_curriculum_state")
        )
        if not isinstance(failure_curriculum_state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
        if current or v5:
            compatibility_histogram = state.get("failure_phase_curriculum_state")
            if not isinstance(compatibility_histogram, Mapping):
                raise ValueError(
                    "TVKD v5+ checkpoint lacks histogram compatibility state"
                )
            if not _same_verified_histogram_state(
                compatibility_histogram, failure_curriculum_state
            ):
                raise ValueError(
                    "TVKD v5+ verified histogram aliases are inconsistent"
                )
        optimizer_state = state.get("optimizer_resume_state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("TVKD checkpoint lacks optimizer state")
        if current and "actor_std_optimizer" not in optimizer_state:
            raise ValueError("TVKD checkpoint lacks actor_std optimizer entry")
        if "dr_estimator_optimizer" not in optimizer_state:
            raise ValueError("TVKD checkpoint lacks DR-estimator optimizer state")
        dr_optimizer_state = optimizer_state["dr_estimator_optimizer"]
        if bool(self.cfg.train_dr_estimator):
            if not isinstance(dr_optimizer_state, Mapping):
                raise ValueError("trainable DR estimator lacks optimizer state")
            if getattr(self, "opt_dr_estimator", None) is None:
                raise RuntimeError("trainable DR estimator has no optimizer")
        elif dr_optimizer_state is not None:
            raise ValueError("frozen DR estimator checkpoint has an optimizer state")
        resume_counters = {}
        counter_names = [
            "num_updates",
            "sac_actor_update_count",
            "sac_alpha_update_count",
        ]
        if current:
            counter_names.append("actor_std_update_count")
        for name in counter_names:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD checkpoint {name} is invalid")
            resume_counters[name] = int(value)

        # A public policy checkpoint contains every PPO/Student/perception
        # child in addition to this algorithm-specific state.  Restore that
        # complete module tree first; restoring only the baseline FastSAC seam
        # would leave perception/EMA children at constructor initialization
        # while loading optimizer moments that refer to different weights.
        if load_modules:
            checkpoint_fingerprint = state.get("vecnorm_fingerprint")
            current_fingerprint = getattr(self, "_replay_vecnorm_fingerprint", None)
            if (
                not isinstance(checkpoint_fingerprint, str)
                or not checkpoint_fingerprint
                or not isinstance(current_fingerprint, str)
                or not current_fingerprint
                or checkpoint_fingerprint != current_fingerprint
            ):
                raise ValueError("TVKD resume checkpoint VecNorm fingerprint mismatch")
            perception_initialization = state.get("perception_initialization")
            if not isinstance(perception_initialization, Mapping):
                raise ValueError(
                    "TVKD resume checkpoint lacks perception initialization state"
                )
            self._perception_initialization = copy.deepcopy(
                dict(perception_initialization)
            )
            self._set_perception_trainable(bool(self.cfg.train_perception))
            missing_children = [
                name for name, _ in self.named_children() if name not in state
            ]
            if missing_children:
                raise ValueError(
                    "TVKD resume requires a full policy state; missing child "
                    f"modules: {missing_children}"
                )
            failed = PPOVEL.load_state_dict(self, state, strict=True)
            if failed:
                raise RuntimeError(
                    "TVKD resume failed to restore policy child modules: "
                    f"{sorted(failed)}"
                )

        # Reuse the baseline's strict optimizer/RNG/counter restoration after
        # translating only its two format sentinels.
        baseline_state = dict(state)
        baseline_state["training_algorithm"] = BASE_FASTSAC_TRAINING_ALGORITHM
        baseline_state["checkpoint_version"] = BASE_FASTSAC_CHECKPOINT_VERSION
        if not current:
            baseline_state["actor_std_update_count"] = 0
            optimizer_resume_state = dict(
                baseline_state.get("optimizer_resume_state", {})
            )
            optimizer_resume_state.setdefault("actor_std_optimizer", None)
            baseline_state["optimizer_resume_state"] = optimizer_resume_state
        DistributionalFastSACTeacherBC._load_fastsac_checkpoint_state(
            self, baseline_state, load_modules=load_modules
        )
        if bool(self.cfg.train_dr_estimator):
            self.opt_dr_estimator.load_state_dict(dr_optimizer_state)
        self.num_updates = resume_counters["num_updates"]
        self.sac_actor_update_count = resume_counters["sac_actor_update_count"]
        self.sac_alpha_update_count = resume_counters["sac_alpha_update_count"]
        if load_modules:
            for name in self._frozen_teacher_module_names():
                module_state = frozen_teacher_state.get(name)
                if not isinstance(module_state, Mapping):
                    raise ValueError(
                        f"TVKD checkpoint lacks frozen Teacher module {name!r}"
                    )
                getattr(self, name).load_state_dict(module_state, strict=True)
        if legacy or previous or v3:
            # Pre-v5 detector scales and phase anchors were collected under
            # terminal/recent-control gates and could contain argmin or legacy
            # fallbacks. They are never promoted to the verified v5
            # threshold-only curriculum.
            self._reset_bottleneck_statistics()
            self._reset_failure_curriculum_state()
        else:
            self._load_bottleneck_replay_checkpoint_state(bottleneck_state)
            self._load_failure_curriculum_checkpoint_state(failure_curriculum_state)
        restored_diagnostics = state.get("last_tvkd_diagnostics", {})
        if not isinstance(restored_diagnostics, Mapping):
            restored_diagnostics = {}
        self._last_tvkd_diagnostics = {
            key: copy.deepcopy(value)
            for key, value in restored_diagnostics.items()
            if not str(key).startswith("bc_scheduler/")
        }
        self._reset_student_replay_episode_tracking()
        self._freeze_teacher()
        self.teacher_value_wrapper.freeze()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore a TVKD model for replayless deterministic evaluation."""
        algorithm = state_dict.get("training_algorithm")
        version = state_dict.get("checkpoint_version", -1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("TVKD inference checkpoint version is invalid")
        if (algorithm, version) not in {
            (TRAINING_ALGORITHM, CHECKPOINT_VERSION),
            (V8_TRAINING_ALGORITHM, V8_CHECKPOINT_VERSION),
            (V7_TRAINING_ALGORITHM, V7_CHECKPOINT_VERSION),
            (V6_TRAINING_ALGORITHM, V6_CHECKPOINT_VERSION),
            (V5_TRAINING_ALGORITHM, V5_CHECKPOINT_VERSION),
            (V4_TRAINING_ALGORITHM, V4_CHECKPOINT_VERSION),
            (V3_TRAINING_ALGORITHM, V3_CHECKPOINT_VERSION),
            (PREVIOUS_TRAINING_ALGORITHM, PREVIOUS_CHECKPOINT_VERSION),
            (LEGACY_TRAINING_ALGORITHM, LEGACY_CHECKPOINT_VERSION),
        }:
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        baseline_state = dict(state_dict)
        baseline_state["training_algorithm"] = BASE_FASTSAC_TRAINING_ALGORITHM
        baseline_state["checkpoint_version"] = BASE_FASTSAC_CHECKPOINT_VERSION
        failed = DistributionalFastSACTeacherBC.load_inference_state_dict(
            self, baseline_state, strict
        )
        frozen_teacher_state = state_dict.get("frozen_teacher_state")
        if not isinstance(frozen_teacher_state, Mapping):
            raise ValueError("TVKD inference checkpoint lacks frozen Teacher state")
        for name in self._frozen_teacher_module_names():
            module_state = frozen_teacher_state.get(name)
            if not isinstance(module_state, Mapping):
                raise ValueError(
                    f"TVKD inference checkpoint lacks Teacher module {name!r}"
                )
            getattr(self, name).load_state_dict(module_state, strict=True)
        restored_diagnostics = state_dict.get("last_tvkd_diagnostics", {})
        self._last_tvkd_diagnostics = (
            {
                key: copy.deepcopy(value)
                for key, value in restored_diagnostics.items()
                if not str(key).startswith("bc_scheduler/")
            }
            if isinstance(restored_diagnostics, Mapping)
            else {}
        )
        self.teacher_value_wrapper.freeze()
        return failed

    def state_dict(self):
        state = DistributionalFastSACTeacherBC.state_dict(self)
        state["replay_resume_semantics"] = REPLAY_RESUME_SEMANTICS
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Fresh training remains the baseline's rigorously validated PPO-source
        # transfer. Same-stage TVKD continuation restores every model,
        # optimizer, RNG, counter, failure curriculum, and bottleneck state,
        # then deliberately rebuilds both online replay rings and every
        # non-serialized Teacher episode/current-EMA cache sidecar.
        if state_dict.get("training_algorithm") not in {
            TRAINING_ALGORITHM,
            V8_TRAINING_ALGORITHM,
            V7_TRAINING_ALGORITHM,
            V6_TRAINING_ALGORITHM,
            V5_TRAINING_ALGORITHM,
            V4_TRAINING_ALGORITHM,
            V3_TRAINING_ALGORITHM,
            PREVIOUS_TRAINING_ALGORITHM,
            LEGACY_TRAINING_ALGORITHM,
        }:
            failed = DistributionalFastSACTeacherBC.load_state_dict(
                self, state_dict, strict
            )
            self._student_q_n_step_accumulator = None
            self.teacher_value_wrapper.freeze()
            return failed
        expected_actor_backend = getattr(
            self, "actor_backend", _fastsac_actor_backend(self.cfg)
        )
        if state_dict.get("actor_backend") != expected_actor_backend:
            raise ValueError("TVKD FastSAC resume actor backend mismatch")
        saved_action_contract = state_dict.get("action_contract")
        if not isinstance(saved_action_contract, Mapping):
            raise ValueError("TVKD FastSAC resume lacks its action contract")
        for key in ("joint_names", "fingerprint"):
            if saved_action_contract.get(key) != self._fastsac_action_contract.get(key):
                raise ValueError(
                    f"TVKD FastSAC resume action contract mismatch at {key!r}"
                )
        self._load_fastsac_checkpoint_state(state_dict, load_modules=True)

        self.dagger_replay.clear()
        if self._uses_separate_online_replays():
            self.student_replay.clear()
        self.q_teacher_replay.clear()
        self._student_q_n_step_accumulator = None
        # The restored partial UTD debt belongs to rows in the discarded
        # rings. It must not combine with the first rows of the rebuilt rings.
        self.q_update_row_credit = 0.0
        self._teacher_prefill_complete = False
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self._teacher_prefill_pending = None
        self._reset_teacher_episode_cache_state()
        self._teacher_prefill_successful_episodes = 0
        self._teacher_prefill_failed_episodes = 0
        self._teacher_prefill_timeout_episodes = 0
        self._teacher_prefill_incomplete_episodes = 0
        self._teacher_prefill_discarded_rows = 0
        self._rollout_final_batch = None
        self._truncation_final_batches = []
        self._last_truncation_finals_used = 0
        self._perception_replay_history = None
        self._perception_replay_history_count = 0
        self._failure_phase_history = None
        self._failure_phase_student_history = None
        self._failure_phase_takeover_history = None
        self._bottleneck_episode_histories = None
        self._reset_student_replay_episode_tracking()
        self._teacher_phase_index_ready = False
        self._teacher_phase_bin_rows = ()
        self._teacher_phase_nearest_nonempty = torch.empty(0, dtype=torch.long)
        self._teacher_phase_flat_rows = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_starts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_counts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_device_cache.clear()
        self.actor_adapt.requires_grad_(True).train()
        # ``requires_grad_(True)`` recursively touches PPOVEL's actor_std.
        # Restore the mode-specific single variance owner: direct actor_std for
        # physical Gaussian, SAC adapter for historical normalized tanh.
        self._configure_training_actor_std()
        self.bc_dagger_sac_adapter.train()
        self.qnet.requires_grad_(True).train()
        self.qnet_target.requires_grad_(False).eval()
        self._set_perception_trainable(bool(self.cfg.train_perception))
        if hasattr(self, "env"):
            self.env.set_progress(
                int(state_dict.get("next_iter", state_dict.get("last_iter", -1) + 1))
            )
        self._freeze_teacher()
        self.teacher_value_wrapper.freeze()
        return []


__all__ = [
    "ACTOR_BACKEND",
    "ACTOR_LEARNING_SEMANTICS",
    "BOTTLENECK_LOCATION_SEMANTICS",
    "BOTTLENECK_SELECTION_MODES",
    "CHECKPOINT_VERSION",
    "FRESH_RING_RESUME_SEMANTICS",
    "LEGACY_ADAPTIVE_BC_CONFIG_FIELDS",
    "LEGACY_CHECKPOINT_VERSION",
    "LEGACY_TEACHER_VALUE_BOUNDARY_SEMANTICS",
    "LEGACY_TRAINING_ALGORITHM",
    "NORMALIZED_ACTOR_LEARNING_SEMANTICS",
    "OBJECT_GEO_REPLAY_SEMANTICS",
    "PHASE_FADED_CRITIC_LEARNING_SEMANTICS",
    "PHASE_FADED_TEACHER_VALUE_POTENTIAL_SEMANTICS",
    "Q_NORMALIZED_ACTOR_LEARNING_SEMANTICS",
    "PREVIOUS_CHECKPOINT_VERSION",
    "PREVIOUS_TRAINING_ALGORITHM",
    "REPLAY_RESUME_SEMANTICS",
    "SOURCE_FAILURE_TEACHER",
    "SOURCE_STUDENT",
    "SOURCE_UNIFORM_TEACHER",
    "TEACHER_VALUE_BOUNDARY_SEMANTICS",
    "TEACHER_VALUE_POTENTIAL_SEMANTICS",
    "TEACHER_VALUE_RETURN_SEMANTICS",
    "TRAINING_ALGORITHM",
    "V8_CHECKPOINT_VERSION",
    "V8_CRITIC_LEARNING_SEMANTICS",
    "V8_TEACHER_VALUE_BOUNDARY_SEMANTICS",
    "V8_TRAINING_ALGORITHM",
    "V7_CHECKPOINT_VERSION",
    "V7_TRAINING_ALGORITHM",
    "V6_CHECKPOINT_VERSION",
    "V6_TRAINING_ALGORITHM",
    "V5_CHECKPOINT_VERSION",
    "V5_ACTOR_LEARNING_SEMANTICS",
    "V5_FRESH_RING_RESUME_SEMANTICS",
    "V5_REPLAY_RESUME_SEMANTICS",
    "V5_TRAINING_ALGORITHM",
    "V4_CHECKPOINT_VERSION",
    "V4_TRAINING_ALGORITHM",
    "V3_CHECKPOINT_VERSION",
    "V3_TRAINING_ALGORITHM",
    "VERIFIED_HISTOGRAM_SEMANTICS",
    "FrozenTeacherValueWrapper",
    "TeacherValueBottleneck",
    "TeacherValueBottleneckDetector",
    "TeacherValueTerms",
    "TVKDDistributionalFastSACTeacherBC",
    "TVKDDistributionalFastSACTeacherBCConfig",
    "_tvkd_actor_learning_semantics",
    "_tvkd_critic_learning_semantics",
    "_tvkd_teacher_value_potential_semantics",
    "compute_continuation_coefficient",
    "compute_teacher_value_continuation",
    "compute_teacher_value_terms",
    "continuation_bootstrap_mask",
    "replay_timeout_final_mask",
    "replay_truncation_mask",
    "_validate_tvkd_algorithm_config",
]
