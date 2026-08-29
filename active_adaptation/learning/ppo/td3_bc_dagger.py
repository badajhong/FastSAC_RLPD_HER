"""C51 distributional TD3 with bounded raw-action Teacher BC.

This module implements ``distributional_td3_teacher_bc_v1``.  Version 5 keeps
the locked VAIC Actor, observation, DAgger, timeout, action, and C51 interfaces.
FastSAC/TVKD Student replay stores the exact flat Actor inputs produced by live
carried-hidden collection; successful Teacher replay stores each raw episode
once and reconstructs it with the current EMA from its real reset.  Legacy TD3
keeps its finite raw-window path.  Every raw state carries its matching object
geometry, preserving PPOVEL's per-environment observation contract.  The
duplicate teacher H5 export is disabled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    hard_copy_,
    make_batch,
    soft_copy_,
)
from .fastsac_vel import (
    FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
    FASTSAC_Q_BALANCED_FUSION_SEMANTICS,
    FASTSAC_Q_DEFAULT_ACTION_FUSION,
    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_LATE_FUSION_SEMANTICS,
    FASTSAC_Q_PREDICTED_EFFECT_CONTEXT_SEMANTICS,
    FASTSAC_Q_RESIDUAL_FILM_SEMANTICS,
    FASTSAC_RAW_OBSERVATION_ROOT,
    REPLAY_OBSERVATION_SEMANTICS,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    _build_isolated_q_network,
    _filter_replay_rows,
    _measure_or_clip_grad_norm,
    _normalize_q_actuator_context_metadata,
    _q_action_hidden_dim,
    _q_state_hidden_dim,
    _project_to_execution_support,
    _sac_bootstrap_mask as _bootstrap_mask,
    _vaic_pre_reset_final_observation_mask,
    _validate_action_safety_clip,
    _vaic_truncation_mask,
    _vaic_nominal_action_coordinates,
)
from .ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_MAX_KEY,
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_CONTROL_MODES,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_MIN_STEP_COUNT,
    DAGGER_REPLAY_OBSERVATION_SEMANTICS,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_SAFE_RELEASE_KEY,
    DAGGER_SAFE_TAKEOVER_KEY,
    DAGGER_SAFE_TEACHER_KEY,
    DAGGER_SAFE_UNSAFE_KEY,
    DAGGER_STUDENT_ACTION_VALID_KEY,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_SEMANTICS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
    _DeviceReplay,
    PPOBCDaggerFinetune,
    _linear_teacher_probability,
    _resolve_replay_device,
)
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    PPOConfig,
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    VEL_CMD_KEY,
    set_recurrent_mode,
)
from .teacher_episode_replay import (
    CurrentEMATeacherActorCache,
    TeacherActorCacheLineage,
    TeacherBoundaryCause,
    TeacherEpisodeSequenceStore,
    classify_teacher_boundary,
)


TRAINING_ALGORITHM = "distributional_td3_teacher_bc_v1"
CHECKPOINT_VERSION = 5
PREVIOUS_CHECKPOINT_VERSION = 4
ACTOR_BACKEND = "ppo_vel_physical_mean_tanh_bounded_td3_bc_v1"
ACTION_CONTRACT_SEMANTICS = (
    "finite_raw_joint_action_support_with_jointwise_normalized_q_bc_v1"
)
CRITIC_SEMANTICS = (
    "deterministic_bounded_target_actor_q_coordinate_noise_and_support_clipped_"
    "lower_expected_complete_c51_distribution_projection_v1"
)
ACTOR_LEARNING_SEMANTICS = (
    "expected_online_q1_plus_joint_normalized_raw_teacher_huber_bc_v1"
)
DAGGER_CONTROL_SEMANTICS = (
    "bounded_raw_mean_joint_normalized_safe_or_beta_execution_projection_v1"
)

TD3_NOISE_FREE_STUDENT_ACTION_KEY = "td3_noise_free_student_action"
TD3_EXPLORATORY_STUDENT_ACTION_KEY = "td3_exploratory_student_action"
TD3_COLLECTOR_NOISE_KEY = "td3_collector_q_noise"
TD3_BETA_KEY = "td3_beta_probability"
TEACHER_PREFILL_SEMANTICS = (
    "forced_valid_teacher_successful_episode_commit_until_replay_capacity_"
    "then_teacher_perception_warmup_then_main_dagger_v5"
)
PERCEPTION_PREFILL_WARMUP_SEMANTICS = (
    "teacher_raw_replay_supervised_online_updates_then_hard_ema_sync_v1"
)
PERCEPTION_PREFILL_DISABLED_SEMANTICS = (
    "disabled_for_ppovel_live_student_rollout_v1"
)

# Authoritative raw perception fields.  Legacy TD3 stores ten-frame windows;
# FastSAC/TVKD store each successful Teacher episode once in a sidecar and keep
# the exact flat Actor inputs produced by the live carried-hidden Student
# collector under the two generic replay keys below.
PERCEPTION_DEPTH_U8_KEY = "perception_depth_u8"
PERCEPTION_POLICY_RAW_KEY = "perception_policy_raw"
PERCEPTION_VEL_COMMAND_RAW_KEY = "perception_vel_command_raw"
PERCEPTION_OBJECT_GEO_ID_KEY = "perception_object_geo_id"
PERCEPTION_IS_INIT_KEY = "perception_is_init"
REPLAY_ACTOR_OBSERVATIONS_KEY = "replay_actor_observations"
REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY = "replay_next_actor_observations"
# The EMA generation that produced a Student collection Actor cache.  This is
# provenance only: it must never be used to reconstruct a recurrent state.
# Teacher rows are re-encoded lazily at the current generation, so the
# staleness metrics deliberately exclude them.
REPLAY_PERCEPTION_EMA_GENERATION_KEY = "replay_perception_ema_generation"
_STUDENT_REPLAY_EMA_AGE_METRIC_KEYS = (
    "available",
    "student_rows",
    "mean",
    "p95",
    "max",
    "stale_fraction",
)
# Transitional internal aliases keep the Student collection patch isolated;
# Teacher episode replay can populate the same generic schema directly.
STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY = REPLAY_ACTOR_OBSERVATIONS_KEY
STUDENT_COLLECTION_NEXT_ACTOR_OBSERVATIONS_KEY = (
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY
)
REFERENCE_PHASE_KEY = "reference_phase"
NEXT_REFERENCE_PHASE_KEY = "next_reference_phase"
FAILURE_PHASE_TEACHER_SOURCE_KEY = "failure_phase_teacher_source"
FAILURE_PHASE_STUDENT_SOURCE_KEY = "failure_phase_student_source"
REPLAY_INTRINSIC_FOCUSED_KEY = "replay_intrinsic_focused"
REPLAY_SAMPLE_PROVENANCE_KEY = "_replay_sample_provenance"
REPLAY_SAMPLE_PHYSICAL_INDEX_KEY = "_replay_sample_physical_index"
REPLAY_SAMPLE_IS_TEACHER_KEY = "_replay_sample_is_teacher"
REPLAY_TERMINATED_KEY = "replay_terminated"
REPLAY_COMMAND_FINISHED_KEY = "replay_command_finished"
REPLAY_TIME_LIMIT_KEY = "replay_time_limit"
REPLAY_MOTION_ID_KEY = "replay_motion_id"
Q_ACTUATOR_CONTEXT_KEY = "q_actuator_context"
NEXT_Q_ACTUATOR_CONTEXT_KEY = "next_q_actuator_context"
Q_ACTUATOR_ACTION_FEATURE_SEMANTICS = (
    "normalized_issued_command_plus_detached_delay_one_hot_centered_alpha_"
    "action_branch_v1"
)
Q_PREDICTED_EFFECT_INTERVALS = 3
Q_PREDICTED_EFFECT_ACTION_FEATURE_SEMANTICS = (
    "normalized_issued_command_plus_normalized_command_delta_plus_three_"
    "control_interval_mean_counterfactual_actuator_effects_plus_detached_"
    "delay_one_hot_centered_alpha_action_branch_v1"
)
TEACHER_EPISODE_UID_KEY = "teacher_episode_uid"
TEACHER_EPISODE_STEP_KEY = "teacher_episode_step"
_TEACHER_PENDING_EPISODE_STEP_KEY = "_teacher_pending_episode_step"
TEACHER_ACTOR_CACHE_ENCODER_SEMANTICS = (
    "full_success_episode_current_ema_carried_hidden_per_state_object_geo_v2"
)
COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS = (
    "student_collection_carried_hidden_current_next_plus_teacher_success_"
    "episode_current_ema_v1"
)
TEACHER_EPISODE_SIDECAR_SEMANTICS = (
    "fresh_nonserialized_success_episode_raw_journal_and_current_ema_cache_v1"
)
OBJECT_GEO_REPLAY_SEMANTICS = (
    "exact_append_only_geometry_codebook_transition_aligned_int32_ids_v1"
)

REPLAY_SOURCE_ORDER = (
    "uniform_student",
    "failure_student",
    "uniform_teacher",
    "failure_teacher",
)
(
    REPLAY_SOURCE_UNIFORM_STUDENT,
    REPLAY_SOURCE_FAILURE_STUDENT,
    REPLAY_SOURCE_UNIFORM_TEACHER,
    REPLAY_SOURCE_FAILURE_TEACHER,
) = range(len(REPLAY_SOURCE_ORDER))
_REPLAY_MIX_DEVICE_COUNTER_KEYS = (
    "intrinsic_focused_rows",
    "failure_provenance_rows",
    "uniform_intrinsic_rows",
    "uniform_rows",
    "duplicate_rows",
    "valid_rows",
    *(f"actual_{source}_rows" for source in REPLAY_SOURCE_ORDER),
    *(f"valid_{source}_rows" for source in REPLAY_SOURCE_ORDER),
)
STUDENT_REPLAY_EPISODE_ID_KEY = "student_replay_episode_id"
STUDENT_REPLAY_EPISODE_STEP_KEY = "student_replay_episode_step"
_PREFILL_ENV_INDEX_KEY = "_teacher_prefill_env_index"
_PREFILL_STEP_INDEX_KEY = "_teacher_prefill_step_index"
_PREFILL_TERMINATED_KEY = "_teacher_prefill_terminated"
_PREFILL_COMMAND_FINISHED_KEY = "_teacher_prefill_command_finished"
_PREFILL_INTERNAL_FIELDS = (
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _PREFILL_TERMINATED_KEY,
    _PREFILL_COMMAND_FINISHED_KEY,
)
FAILURE_PHASE_REPLAY_SEMANTICS = (
    "student_physical_failure_causal_lookback_phase_histogram_"
    "teacher_internal_uniform_plus_focused_v1"
)
PERCEPTION_REPLAY_SEMANTICS = (
    "raw_input_per_state_object_geo_current_ema_reencode_zero_boundary_"
    "burn_in_8_v2"
)
ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE = "online_student_rollout"
ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS = (
    "ppovel_live_recurrent_student_rollout_2epoch_v1"
)
PERCEPTION_DEPTH_CODEC = "uint8_div_100_v1"
PERCEPTION_WARMSTART_SEMANTICS = (
    "strict_full_student_or_ppovel_train_partial_perception_overlay_v2"
)
PRETRAINED_PERCEPTION_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
PPOVEL_PARTIAL_PERCEPTION_MODULES = (
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
PPOVEL_PARTIAL_FRESH_DEPTH_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
)
PERCEPTION_WARMSTART_MODE_FULL_STUDENT = "strict_full_student"
PERCEPTION_WARMSTART_MODE_PPOVEL_PARTIAL = "ppo_vel_train_partial"
PERCEPTION_WARMSTART_MODE_FRESH = "fresh_constructor"
_ONLINE_PERCEPTION_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "object_adapt",
    "adapt_module",
)
_EMA_PERCEPTION_MODULES = (
    "temporal_depth_gru_ema",
    "object_adapt_ema",
    "adapt_ema",
)

_PERCEPTION_REPLAY_FIELDS = (
    PERCEPTION_DEPTH_U8_KEY,
    PERCEPTION_POLICY_RAW_KEY,
    PERCEPTION_VEL_COMMAND_RAW_KEY,
    PERCEPTION_OBJECT_GEO_ID_KEY,
    PERCEPTION_IS_INIT_KEY,
)

# Drift diagnostics keep the actual pre-VecNorm depth.  Reusing the replay's
# uint8 codec would measure quantization error together with EMA drift and
# make a same-generation comparison non-zero for the wrong reason.
_STUDENT_DRIFT_RAW_FIELDS = (
    DEPTH_KEY,
    PERCEPTION_POLICY_RAW_KEY,
    PERCEPTION_VEL_COMMAND_RAW_KEY,
    PERCEPTION_OBJECT_GEO_ID_KEY,
    PERCEPTION_IS_INIT_KEY,
)

_Q_REPLAY_FIELDS = (
    "critic_observations",
    "actions",
    "rewards",
    "dones",
    "truncations",
    "discounts",
    "next_critic_observations",
    REFERENCE_PHASE_KEY,
    REPLAY_PERCEPTION_EMA_GENERATION_KEY,
    *_PERCEPTION_REPLAY_FIELDS,
)

# TVKD additionally needs exact boundary causes, motion identity, and the raw
# successor phase in replay.  Keep those fields separate from the base TD3
# schema; canonical four-way policies opt into the extended storage below.
_V4_REPLAY_CONTEXT_FIELDS = (
    REPLAY_TERMINATED_KEY,
    REPLAY_COMMAND_FINISHED_KEY,
    REPLAY_TIME_LIMIT_KEY,
    REPLAY_MOTION_ID_KEY,
    # Raw metadata for a time-dependent TVKD potential.  It is never appended
    # to the Actor or Critic observation; sampled replay uses it only to form
    # Phi(s') independently from Phi(s).
    NEXT_REFERENCE_PHASE_KEY,
)


@dataclass
class _StudentPerceptionDriftEpisode:
    """Small diagnostic-only complete episode captured from the live Student."""

    raw_fields: dict[str, torch.Tensor]
    collection_actor: torch.Tensor
    collection_ema_generation: torch.Tensor
    eligible_student_rows: torch.Tensor


def _failure_lookback_offsets(
    lookback_steps: int,
    samples_per_failure: int,
    *,
    device=None,
) -> torch.Tensor:
    """Return evenly spaced chronological offsets over an inclusive lookback."""
    if isinstance(lookback_steps, bool) or int(lookback_steps) < 0:
        raise ValueError("lookback_steps must be a non-negative integer")
    if isinstance(samples_per_failure, bool) or int(samples_per_failure) < 1:
        raise ValueError("samples_per_failure must be a positive integer")
    lookback_steps = int(lookback_steps)
    count = min(int(samples_per_failure), lookback_steps + 1)
    return (
        torch.linspace(
            0,
            lookback_steps,
            count,
            device=device,
            dtype=torch.float64,
        )
        .round()
        .long()
    )


def _source_counts(
    batch_size: int,
    teacher_fraction: float = 0.5,
    failure_within_teacher_fraction: float = 0.3,
) -> tuple[int, int, int]:
    """Return Student, uniform-Teacher, focused-Teacher row counts."""
    if isinstance(batch_size, bool) or int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer")
    for name, fraction in (
        ("teacher_fraction", teacher_fraction),
        ("failure_within_teacher_fraction", failure_within_teacher_fraction),
    ):
        if (
            isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError(f"{name} must be in [0,1]")
    batch_size = int(batch_size)
    teacher_count = min(
        batch_size, int(math.floor(batch_size * float(teacher_fraction) + 0.5))
    )
    focused_count = min(
        teacher_count,
        int(math.floor(teacher_count * float(failure_within_teacher_fraction) + 0.5)),
    )
    return batch_size - teacher_count, teacher_count - focused_count, focused_count


def allocate_source_counts(
    batch_size: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    """Allocate exactly ``batch_size`` rows by deterministic largest remainder.

    Ties are resolved by :data:`REPLAY_SOURCE_ORDER`, independent of the input
    mapping's insertion order.  Fractions are validated here as well as by the
    structured algorithm validator because this helper is a public unit seam.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(fractions, Mapping):
        raise TypeError("fractions must be a mapping")
    if set(fractions) != set(REPLAY_SOURCE_ORDER):
        raise ValueError(
            "fractions must contain exactly " + ", ".join(REPLAY_SOURCE_ORDER)
        )

    values: list[float] = []
    for name in REPLAY_SOURCE_ORDER:
        value = fractions[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} fraction must be a finite number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} fraction must lie in [0,1]")
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) >= 1e-8:
        raise ValueError(f"source fractions must sum to 1; got {total!r}")

    exact = [batch_size * value for value in values]
    floors = [int(math.floor(value)) for value in exact]
    remaining = batch_size - sum(floors)
    if not 0 <= remaining <= len(REPLAY_SOURCE_ORDER):
        # With the supported batch sizes this can only be reached by fractions
        # whose small sum error is amplified by an implausibly large batch.
        raise ValueError("source fractions cannot be allocated at this batch size")
    ranking = sorted(
        range(len(REPLAY_SOURCE_ORDER)),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )
    for index in ranking[:remaining]:
        floors[index] += 1
    counts = dict(zip(REPLAY_SOURCE_ORDER, floors))
    if (
        any(value < 0 for value in counts.values())
        or sum(counts.values()) != batch_size
    ):
        raise RuntimeError("largest-remainder allocation violated its row contract")
    return counts


def _masked_feature_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean a feature loss over selected rows/timesteps without source scaling."""
    mask = mask.bool()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error)
    denominator = mask.sum()
    if not bool(denominator):
        return error.sum() * 0.0
    return error.masked_select(mask).mean()


@torch.no_grad()
def _sample_replay_by_indices(replay, indices, output_device, *, fields):
    """Indexed replay gather supporting both baseline and TD3 CPU FIFOs."""
    if hasattr(replay, "sample_by_indices"):
        return replay.sample_by_indices(indices, output_device, fields=fields)
    replay_indices = indices.to(replay.device)
    output_device = torch.device(output_device)
    result = {}
    for key in fields:
        value = replay.data[key].index_select(0, replay_indices)
        result[key] = (
            value if value.device == output_device else value.to(output_device)
        )
    return result


@torch.no_grad()
def _encode_replay_depth_u8(depth: torch.Tensor) -> torch.Tensor:
    """Losslessly encode the task's 0.01-grid pre-VecNorm depth values."""
    if not depth.is_floating_point():
        raise TypeError("raw replay depth must be floating point")
    if not torch.isfinite(depth).all():
        raise ValueError("raw replay depth contains non-finite values")
    scaled = depth * 100.0
    bins = scaled.round()
    tolerance = 4.0 * torch.finfo(depth.dtype).eps * 100.0
    if (
        (depth < -tolerance).any()
        or (depth > 1.0 + tolerance).any()
        or not torch.allclose(scaled, bins, rtol=0.0, atol=tolerance)
    ):
        raise ValueError("raw replay depth must lie in [0,1] on the task's 0.01 grid")
    return bins.clamp_(0, 100).to(torch.uint8)


@torch.no_grad()
def _decode_replay_depth_u8(depth: torch.Tensor) -> torch.Tensor:
    """Decode ``uint8_div_100_v1`` without applying VecNorm."""
    if depth.dtype != torch.uint8:
        raise TypeError("encoded replay depth must be uint8")
    if (depth > 100).any():
        raise ValueError("encoded replay depth has bins outside [0,100]")
    return depth.float().div_(100.0)


def _categorical_expected_value(
    logits: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Return the expectation of categorical logits on a fixed support."""
    if logits.ndim < 2:
        raise ValueError("C51 logits must contain batch and atom dimensions")
    if support.ndim != 1 or logits.shape[-1] != support.numel():
        raise ValueError("C51 support does not match the logit atom dimension")
    return (F.softmax(logits, dim=-1) * support).sum(dim=-1)


@torch.no_grad()
def _select_lower_expected_c51_distribution(
    target_logits: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one complete target-head distribution per row.

    Ties select Q1.  No atom-wise minimum, averaging, or scalar fitting is
    performed. The returned target values are detached by construction.
    """
    if target_logits.ndim != 3 or target_logits.shape[0] != 2:
        raise ValueError("target logits must have shape [2, batch, atoms]")
    if support.ndim != 1 or target_logits.shape[-1] != support.numel():
        raise ValueError("C51 support does not match target logits")
    probabilities = F.softmax(target_logits, dim=-1)
    expected_heads = (probabilities * support).sum(dim=-1)
    selected_head = (expected_heads[1] < expected_heads[0]).long()
    selected_probability = probabilities.gather(
        0,
        selected_head[None, :, None].expand(
            1, target_logits.shape[1], target_logits.shape[2]
        ),
    ).squeeze(0)
    return selected_probability, expected_heads, selected_head


@torch.no_grad()
def _project_c51_probabilities(
    probabilities: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    effective_discount: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a selected complete C51 distribution through one-step Bellman."""
    if probabilities.ndim != 2:
        raise ValueError("C51 probabilities must have shape [batch, atoms]")
    batch_size, atom_count = probabilities.shape
    if support.ndim != 1 or support.numel() != atom_count or atom_count < 2:
        raise ValueError("C51 support does not match probabilities")
    for name, value in (
        ("rewards", rewards),
        ("bootstrap", bootstrap),
        ("effective_discount", effective_discount),
    ):
        if value.reshape(-1).shape != (batch_size,):
            raise ValueError(f"{name} must contain one value per replay row")
    deltas = support[1:] - support[:-1]
    expected_delta = (support[-1] - support[0]) / (atom_count - 1)
    spacing_atol = (
        8.0
        * torch.finfo(support.dtype).eps
        * support.detach().abs().max().clamp_min(1.0)
    )
    if (
        not torch.isfinite(probabilities).all()
        or not torch.isfinite(rewards).all()
        or not torch.isfinite(bootstrap).all()
        or not torch.isfinite(effective_discount).all()
        or not torch.isfinite(support).all()
        or not torch.all(deltas > 0)
        # Float32 linspace accumulates a few ulps of spacing variation over a
        # wide, dense support (the locked contract is 501 atoms on [-20, 20]).
        # Compare against the analytic spacing with a dtype/scale-aware bound.
        or not torch.allclose(
            deltas,
            expected_delta.expand_as(deltas),
            rtol=1e-4,
            atol=spacing_atol,
        )
    ):
        raise ValueError("C51 projection inputs must be finite and uniform")

    rewards = rewards.reshape(batch_size)
    bootstrap = bootstrap.reshape(batch_size).to(probabilities.dtype)
    effective_discount = effective_discount.reshape(batch_size)
    transformed = rewards[:, None] + (bootstrap * effective_discount)[:, None] * support
    support_low = support[0]
    support_high = support[-1]
    left_fraction = (transformed < support_low).float().mean()
    right_fraction = (transformed > support_high).float().mean()

    delta = expected_delta
    atom_position = (
        (transformed.clamp(support_low, support_high) - support_low)
        .div(delta)
        .clamp(0.0, float(atom_count - 1))
    )
    lower = atom_position.floor().long()
    upper = atom_position.ceil().long()
    same_atom = lower == upper
    # Match the authoritative existing C51 projection exactly at grid points.
    lower = torch.where(same_atom & (lower > 0), lower - 1, lower)
    upper = torch.where(same_atom & (upper == 0), upper + 1, upper)
    offset = torch.arange(batch_size, device=probabilities.device)[:, None] * atom_count
    projected = torch.zeros_like(probabilities)
    projected.view(-1).index_add_(
        0,
        (lower + offset).reshape(-1),
        (probabilities * (upper.to(atom_position.dtype) - atom_position)).reshape(-1),
    )
    projected.view(-1).index_add_(
        0,
        (upper + offset).reshape(-1),
        (probabilities * (atom_position - lower.to(atom_position.dtype))).reshape(-1),
    )
    return projected, left_fraction, right_fraction


@torch.no_grad()
def _polyak_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-update matching module parameters without sharing storage."""
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("Polyak tau must be finite and in [0, 1]")
    target_parameters = tuple(target.parameters())
    source_parameters = tuple(source.parameters())
    if len(target_parameters) != len(source_parameters):
        raise ValueError("Polyak modules have different parameter counts")
    for target_parameter, source_parameter in zip(target_parameters, source_parameters):
        if target_parameter.shape != source_parameter.shape:
            raise ValueError("Polyak modules have different parameter shapes")
        if target_parameter.data_ptr() == source_parameter.data_ptr():
            raise ValueError("Polyak target/source parameters share storage")
        target_parameter.lerp_(source_parameter, tau)


def _bounded_action_contract_metadata(
    joint_names,
    nominal_action_low: torch.Tensor,
    nominal_action_high: torch.Tensor,
    offset_low: torch.Tensor,
    offset_high: torch.Tensor,
    execution_action_low: torch.Tensor,
    execution_action_high: torch.Tensor,
) -> dict:
    """Describe finite execution support and nominal joint Q/BC coordinates."""
    joint_names = list(joint_names)
    values = {
        "nominal_action_low": torch.as_tensor(nominal_action_low, dtype=torch.float32)
        .detach()
        .cpu(),
        "nominal_action_high": torch.as_tensor(nominal_action_high, dtype=torch.float32)
        .detach()
        .cpu(),
        "joint_offset_low": torch.as_tensor(offset_low, dtype=torch.float32)
        .detach()
        .cpu(),
        "joint_offset_high": torch.as_tensor(offset_high, dtype=torch.float32)
        .detach()
        .cpu(),
        "action_low": torch.as_tensor(execution_action_low, dtype=torch.float32)
        .detach()
        .cpu(),
        "action_high": torch.as_tensor(execution_action_high, dtype=torch.float32)
        .detach()
        .cpu(),
    }
    expected = (len(joint_names),)
    for name, value in values.items():
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite vector matching the joint order")
    low = values["nominal_action_low"]
    high = values["nominal_action_high"]
    if not torch.all(high > low):
        raise ValueError("nominal raw-action coordinates must have positive width")
    action_low = values["action_low"]
    action_high = values["action_high"]
    if not torch.all(action_high > action_low):
        raise ValueError("execution raw-action support must have positive width")
    action_center = (action_low + action_high) * 0.5
    action_scale = (action_high - action_low) * 0.5
    q_center = (low + high) * 0.5
    q_scale = (high - low) * 0.5

    def fingerprint(payload):
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    execution_payload = {
        "semantics": "bounded_raw_joint_command_projection_v1",
        "joint_names": joint_names,
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
    }
    q_payload = {
        "semantics": "execution_project_then_jointwise_nominal_center_scale_v1",
        "joint_names": joint_names,
        "action_center": q_center.tolist(),
        "action_scale": q_scale.tolist(),
        "physical_clamp_low": action_low.tolist(),
        "physical_clamp_high": action_high.tolist(),
    }
    entropy_payload = {
        "semantics": "jointwise_nominal_bounded_action_density_v1",
        "joint_names": joint_names,
        "action_scale": q_scale.tolist(),
    }
    payload = {
        "semantics": ACTION_CONTRACT_SEMANTICS,
        "action_bound_source": "scalar_finite_action_support",
        "execution_support_semantics": execution_payload["semantics"],
        "execution_support_fingerprint": fingerprint(execution_payload),
        "joint_names": joint_names,
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
        "action_center": action_center.tolist(),
        "action_scale": action_scale.tolist(),
        "q_action_coordinate_source": "soft_joint_limits_at_default_pose",
        "nominal_action_low": low.tolist(),
        "nominal_action_high": high.tolist(),
        "q_action_center": q_center.tolist(),
        "q_action_scale": q_scale.tolist(),
        "q_action_clamp": "physical_execution_support_before_affine",
        "q_action_transform_semantics": q_payload["semantics"],
        "q_action_transform_fingerprint": fingerprint(q_payload),
        "bc_action_transform_semantics": q_payload["semantics"],
        "bc_action_transform_fingerprint": fingerprint(q_payload),
        "entropy_reference_source": "nominal_joint_action_coordinates",
        "entropy_reference_scale": q_scale.tolist(),
        "entropy_reference_semantics": entropy_payload["semantics"],
        "entropy_reference_fingerprint": fingerprint(entropy_payload),
        "joint_offset_low": values["joint_offset_low"].tolist(),
        "joint_offset_high": values["joint_offset_high"].tolist(),
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _valid_raw_action_rows(actions: torch.Tensor) -> torch.Tensor:
    """Validate raw proposals before projecting them to execution support."""
    if actions.ndim < 1 or actions.shape[-1] < 1:
        raise ValueError("raw actions must contain an action dimension")
    return torch.isfinite(actions).all(dim=-1)


def _joint_normalized_action_discrepancy(
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    action_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RMS/max raw-action error in joint-wise nominal coordinates."""
    if student_action.shape != teacher_action.shape:
        raise ValueError("SafeDAgger Teacher and Student action shapes must match")
    scale = torch.as_tensor(
        action_scale, device=student_action.device, dtype=student_action.dtype
    )
    if scale.shape != student_action.shape[-1:] or not torch.all(scale > 0):
        raise ValueError("SafeDAgger action scale must match the action dimension")
    error = (student_action - teacher_action).abs() / scale
    return error.square().mean(dim=-1).sqrt(), error.amax(dim=-1)


def _exact_teacher_bc_loss(
    prediction_action: torch.Tensor,
    teacher_action: torch.Tensor,
    valid_mask: torch.Tensor,
    action_center: torch.Tensor,
    action_scale: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """SmoothL1 between raw actions after joint-wise coordinate normalization."""
    valid = valid_mask.reshape(-1).bool()
    if prediction_action.shape != teacher_action.shape:
        raise ValueError("BC prediction and Teacher action shapes must match")
    if prediction_action.shape[0] != valid.numel():
        raise ValueError("BC validity mask does not match batch rows")
    if not valid.any():
        return prediction_action.sum() * 0.0
    selected_prediction = prediction_action[valid]
    if not torch.isfinite(selected_prediction).all():
        raise RuntimeError("BC Student raw action contains non-finite values")
    selected_teacher = teacher_action[valid].detach()
    if not torch.isfinite(selected_teacher).all():
        raise RuntimeError("BC Teacher raw action contains non-finite values")
    center = action_center.to(selected_prediction)
    scale = action_scale.to(selected_prediction)
    if center.shape != selected_prediction.shape[-1:] or scale.shape != center.shape:
        raise ValueError("BC action coordinates do not match action dimension")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("BC action scale must be finite and positive")
    prediction_normalized = (selected_prediction - center) / scale
    teacher_normalized = (selected_teacher - center) / scale
    return F.smooth_l1_loss(
        prediction_normalized,
        teacher_normalized,
        beta=float(huber_delta),
    )


def _td3_actor_q1_loss(
    qnet: nn.Module,
    critic_observations: torch.Tensor,
    q_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``-E[Q1]`` while retaining dQ/da but no Q-parameter graph."""
    q1 = qnet.qnets[0] if hasattr(qnet, "qnets") else None
    frozen_module = q1 if q1 is not None else qnet
    original_requires_grad = [
        parameter.requires_grad for parameter in frozen_module.parameters()
    ]
    try:
        for parameter in frozen_module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        logits = (
            q1(critic_observations, q_action)
            if q1 is not None
            else qnet(critic_observations, q_action)[0]
        )
        expected_q1 = _categorical_expected_value(logits, qnet.support)
        return -expected_q1.mean(), expected_q1
    finally:
        for parameter, requires_grad in zip(
            frozen_module.parameters(), original_requires_grad
        ):
            parameter.requires_grad_(requires_grad)


def _apply_student_collector_noise(
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    student_selected_mask: torch.Tensor,
    noise_std: float,
    noise_clip: float,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    q_action_center: torch.Tensor,
    q_action_scale: torch.Tensor,
    q_action_gain: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add Q-coordinate noise and keep the executable action inside support."""
    if student_action.shape != teacher_action.shape:
        raise ValueError("Teacher and Student action shapes must match")
    selected = student_selected_mask.reshape(student_action.shape[:-1]).bool()
    noise_std = float(noise_std)
    noise_clip = float(noise_clip)
    gain = float(q_action_gain)
    if (
        not math.isfinite(noise_std)
        or noise_std < 0.0
        or not math.isfinite(noise_clip)
        or noise_clip < 0.0
        or not math.isfinite(gain)
        or gain <= 0.0
    ):
        raise ValueError("collector noise parameters are invalid")

    center = torch.as_tensor(
        q_action_center, device=student_action.device, dtype=student_action.dtype
    )
    scale = torch.as_tensor(
        q_action_scale, device=student_action.device, dtype=student_action.dtype
    )
    low = torch.as_tensor(
        action_low, device=student_action.device, dtype=student_action.dtype
    )
    high = torch.as_tensor(
        action_high, device=student_action.device, dtype=student_action.dtype
    )
    if center.shape != student_action.shape[-1:] or scale.shape != center.shape:
        raise ValueError("Q action transform does not match action dimension")
    if low.shape != center.shape or high.shape != center.shape:
        raise ValueError("execution support does not match action dimension")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("Q action scale must be finite and positive")
    if (
        not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or not torch.all(high > low)
    ):
        raise ValueError("execution support must be finite with positive width")
    bounded_student = torch.maximum(torch.minimum(student_action, high), low)
    bounded_teacher = torch.maximum(torch.minimum(teacher_action, high), low)
    if noise_std == 0.0 or noise_clip == 0.0 or not selected.any():
        exploratory_student = bounded_student
        issued_action = torch.where(
            selected.unsqueeze(-1), exploratory_student, bounded_teacher
        )
        return issued_action, exploratory_student, torch.zeros_like(student_action)
    q_student = ((bounded_student - center) / scale) * gain
    q_low = ((low - center) / scale) * gain
    q_high = ((high - center) / scale) * gain
    q_min = torch.minimum(q_low, q_high)
    q_max = torch.maximum(q_low, q_high)
    sampled = (
        torch.randn(
            q_student.shape,
            device=q_student.device,
            dtype=q_student.dtype,
            generator=generator,
        )
        * noise_std
    )
    sampled = sampled.clamp(-noise_clip, noise_clip)
    sampled = torch.where(selected.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    noisy_q = torch.maximum(torch.minimum(q_student + sampled, q_max), q_min)
    exploratory_student = (noisy_q / gain) * scale + center
    exploratory_student = torch.maximum(torch.minimum(exploratory_student, high), low)
    exploratory_student = torch.where(
        selected.unsqueeze(-1), exploratory_student, bounded_student
    )
    actual_noise = (((exploratory_student - center) / scale) * gain) - q_student
    actual_noise = torch.where(
        selected.unsqueeze(-1), actual_noise, torch.zeros_like(actual_noise)
    )
    issued_action = torch.where(
        selected.unsqueeze(-1), exploratory_student, bounded_teacher
    )
    if (
        not torch.isfinite(exploratory_student).all()
        or not torch.isfinite(issued_action).all()
    ):
        raise RuntimeError("TD3 collector noise produced a non-finite raw action")
    return issued_action, exploratory_student, actual_noise


class _TD3DeviceReplay(_DeviceReplay):
    """TD3-only replay sampling optimized for a CPU-resident FIFO.

    Random physical indices are supplied by the rollout-level prefetch plan,
    avoiding a CUDA-index-to-CPU synchronization for every sampled source.
    CPU fields are gathered with the same row/index semantics as
    :class:`_DeviceReplay`, then packed by dtype into reusable pinned staging
    buffers.  A blocking packed transfer is intentional: it is safe to reuse
    the staging allocation on the next call while reducing pageable H2D calls
    from one per field to one per dtype.
    """

    def __init__(self, capacity: int, device):
        super().__init__(capacity, device)
        self._pinned_sample_staging: dict[
            tuple[torch.dtype, int, int], torch.Tensor
        ] = {}

    def clear(self) -> None:
        super().clear()
        self._pinned_sample_staging.clear()

    @torch.no_grad()
    def extend_by_indices(
        self,
        transitions: Mapping[str, torch.Tensor],
        indices: torch.Tensor,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> int:
        """Append selected rows directly into this ring without a host copy.

        This is the indexed equivalent of ``extend({k: v[indices]})``.  In the
        production CUDA-to-CPU path each selected field is copied straight
        into its final FIFO slice, avoiding a pageable CPU staging tensor and
        the subsequent CPU-to-CPU ring copy.  Selection order, wraparound,
        capacity truncation, counters, and stored dtypes remain identical.
        """
        if not transitions:
            return 0
        if not isinstance(indices, torch.Tensor):
            raise TypeError("Replay indices must be a tensor")
        if indices.dtype != torch.long or indices.ndim != 1:
            raise TypeError("Replay indices must be a one-dimensional int64 tensor")
        fields = tuple(transitions) if fields is None else tuple(fields)
        unknown = set(fields).difference(transitions)
        if unknown:
            raise KeyError(f"Unknown source transition fields: {sorted(unknown)}")
        if not fields:
            return 0
        count = int(indices.numel())
        if count == 0:
            return 0
        for key in fields:
            value = transitions[key]
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(f"Replay field {key!r} must be a non-scalar tensor")
            if value.device != indices.device:
                raise RuntimeError(
                    "Replay indices and source transition fields must share a device"
                )
        minimum_source_rows = min(int(transitions[key].shape[0]) for key in fields)
        if bool((indices < 0).any()) or bool(
            (indices >= minimum_source_rows).any()
        ):
            raise IndexError("Replay index is outside a source transition field")

        if not self.data:
            self.data = {
                key: torch.empty(
                    (self.capacity, *transitions[key].shape[1:]),
                    dtype=transitions[key].dtype,
                    device=self.device,
                )
                for key in fields
            }
        elif set(fields) != set(self.data):
            raise KeyError("DAgger replay field set changed after allocation")

        self._valid_index_cache.clear()
        self.seen += count
        if count >= self.capacity:
            write_indices = indices[-self.capacity :]
            for key in fields:
                value = transitions[key].detach()
                self.data[key].copy_(value.index_select(0, write_indices))
            self.ptr = 0
            self.size = self.capacity
            return count

        first = min(count, self.capacity - self.ptr)
        second = count - first
        first_indices = indices[:first]
        second_indices = indices[first:]
        for key in fields:
            value = transitions[key].detach()
            self.data[key][self.ptr : self.ptr + first].copy_(
                value.index_select(0, first_indices)
            )
            if second:
                self.data[key][:second].copy_(
                    value.index_select(0, second_indices)
                )
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    @torch.no_grad()
    def sample_by_indices(self, indices: torch.Tensor, output_device, fields=None):
        """Gather exact physical replay rows and move them as a coalesced batch."""
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty DAgger replay")
        if not isinstance(indices, torch.Tensor):
            raise TypeError("Replay indices must be a tensor")
        if indices.dtype != torch.long or indices.ndim != 1:
            raise TypeError("Replay indices must be a one-dimensional int64 tensor")
        if indices.numel() < 1:
            raise ValueError("Replay indices cannot be empty")
        if indices.device != self.device:
            indices = indices.to(self.device)

        fields = tuple(self.data) if fields is None else tuple(fields)
        unknown = set(fields).difference(self.data)
        if unknown:
            raise KeyError(f"Unknown DAgger replay sample fields: {sorted(unknown)}")
        output_device = torch.device(output_device)
        if self.device.type != "cpu" or output_device.type != "cuda":
            sampled = {key: self.data[key].index_select(0, indices) for key in fields}
            return {
                key: (
                    value if value.device == output_device else value.to(output_device)
                )
                for key, value in sampled.items()
            }

        # Replay includes float, bool, and compact integer geometry IDs.
        # Grouping by dtype keeps every field exact while coalescing transfers.
        dtype_groups: dict[torch.dtype, list[str]] = {}
        for key in fields:
            dtype_groups.setdefault(self.data[key].dtype, []).append(key)
        transferred: dict[str, torch.Tensor] = {}
        row_count = int(indices.numel())
        for dtype, keys in dtype_groups.items():
            shapes = [(row_count, *self.data[key].shape[1:]) for key in keys]
            sizes = [math.prod(shape) for shape in shapes]
            total_size = sum(sizes)
            staging_key = (dtype, row_count, total_size)
            staging = self._pinned_sample_staging.get(staging_key)
            if staging is None:
                staging = torch.empty(
                    (total_size,),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self._pinned_sample_staging[staging_key] = staging
            offset = 0
            for key, shape, size in zip(keys, shapes, sizes):
                destination = staging[offset : offset + size].view(shape)
                torch.index_select(self.data[key], 0, indices, out=destination)
                offset += size
            # Blocking transfer makes reuse of the pinned buffer race-free.
            packed = staging.to(output_device, non_blocking=False)
            offset = 0
            for key, shape, size in zip(keys, shapes, sizes):
                transferred[key] = packed[offset : offset + size].view(shape)
                offset += size
        return {key: transferred[key] for key in fields}


@dataclass(frozen=True)
class _TD3ReplaySamplePlan:
    """Physical CPU replay rows plus the original device-side Q permutation."""

    teacher_indices: torch.Tensor
    student_indices: torch.Tensor
    permutation: torch.Tensor
    actor_indices: torch.Tensor | None
    actor_teacher_indices: torch.Tensor | None
    teacher_focused: torch.Tensor | None = None
    actor_teacher_focused: torch.Tensor | None = None
    student_focused: torch.Tensor | None = None
    actor_student_focused: torch.Tensor | None = None


@torch.no_grad()
def _prefetch_td3_replay_sample_plans(
    dagger_replay: _DeviceReplay,
    q_teacher_replay: _DeviceReplay,
    *,
    q_batch_size: int,
    actor_batch_size: int,
    update_count: int,
    policy_delay: int,
    critic_update_count: int,
    q_teacher_replay_ratio: float = 0.5,
    teacher_actor_replay_fraction: float = 0.0,
    output_device,
    generator: torch.Generator,
) -> tuple[_TD3ReplaySamplePlan, ...]:
    """Draw one rollout's CPU replay indices with a single device-to-host copy.

    The calls to ``randint``/``randperm`` remain in precisely the same order as
    sequential balanced-Q and delayed-Actor sampling.  Consequently sampled
    values and the checkpointed ``q_rng`` state are unchanged; only the point
    at which the exclusive replay RNG is consumed moves before the optimizer
    loop.
    """
    if dagger_replay.device.type != "cpu" or q_teacher_replay.device.type != "cpu":
        raise ValueError("TD3 replay index prefetch requires CPU-resident FIFOs")
    q_batch_size = int(q_batch_size)
    actor_batch_size = int(actor_batch_size)
    update_count = int(update_count)
    policy_delay = int(policy_delay)
    if q_batch_size < 1:
        raise ValueError("q_batch_size must be a positive integer")
    if actor_batch_size < 1 or update_count < 0 or policy_delay < 1:
        raise ValueError("TD3 sample-plan sizes and policy_delay are invalid")
    for name, value in (
        ("q_teacher_replay_ratio", q_teacher_replay_ratio),
        ("teacher_actor_replay_fraction", teacher_actor_replay_fraction),
    ):
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be in [0,1]")
    q_teacher_replay_ratio = float(q_teacher_replay_ratio)
    teacher_actor_replay_fraction = float(teacher_actor_replay_fraction)
    if update_count == 0:
        return ()

    output_device = torch.device(output_device)
    generator_device = torch.device(generator.device)
    if generator_device.type != output_device.type or (
        generator_device.index is not None
        and output_device.index is not None
        and generator_device.index != output_device.index
    ):
        raise ValueError("Replay generator and policy output device must match")
    student_count, teacher_count, _ = _source_counts(
        q_batch_size, q_teacher_replay_ratio, 0.0
    )
    actor_main_count, actor_teacher_count, _ = _source_counts(
        actor_batch_size, teacher_actor_replay_fraction, 0.0
    )
    actor_update_scheduled = any(
        (int(critic_update_count) + update_index + 1) % policy_delay == 0
        for update_index in range(update_count)
    )
    student_rows_required = student_count or (
        actor_update_scheduled and actor_main_count
    )
    valid_student_indices = (
        dagger_replay._valid_indices(DAGGER_IS_STUDENT_ACTION_KEY)
        if student_rows_required
        else torch.empty(0, dtype=torch.long, device=dagger_replay.device)
    )
    if (teacher_count or (actor_update_scheduled and actor_teacher_count)) and (
        q_teacher_replay.size < 1
    ):
        raise RuntimeError("Cannot sample before a required teacher transition exists")
    if student_rows_required and valid_student_indices.numel() < 1:
        raise RuntimeError("Cannot sample before a required student transition exists")
    index_draws: list[torch.Tensor] = []
    records: list[
        tuple[int | None, int | None, torch.Tensor, int | None, int | None]
    ] = []
    for update_index in range(update_count):
        teacher_draw = None
        if teacher_count:
            teacher_draw = len(index_draws)
            index_draws.append(
                torch.randint(
                    0,
                    q_teacher_replay.size,
                    (teacher_count,),
                    device=generator_device,
                    generator=generator,
                )
            )
        student_draw = None
        if student_count:
            student_draw = len(index_draws)
            index_draws.append(
                torch.randint(
                    0,
                    valid_student_indices.numel(),
                    (student_count,),
                    device=generator_device,
                    generator=generator,
                )
            )
        permutation = torch.randperm(
            q_batch_size, device=generator_device, generator=generator
        )
        actor_draw = None
        actor_teacher_draw = None
        if (int(critic_update_count) + update_index + 1) % policy_delay == 0:
            if actor_teacher_count:
                actor_teacher_draw = len(index_draws)
                index_draws.append(
                    torch.randint(
                        0,
                        q_teacher_replay.size,
                        (actor_teacher_count,),
                        device=generator_device,
                        generator=generator,
                    )
                )
            if actor_main_count:
                actor_draw = len(index_draws)
                index_draws.append(
                    torch.randint(
                        0,
                        valid_student_indices.numel(),
                        (actor_main_count,),
                        device=generator_device,
                        generator=generator,
                    )
                )
        records.append(
            (
                teacher_draw,
                student_draw,
                permutation,
                actor_draw,
                actor_teacher_draw,
            )
        )

    lengths = [int(draw.numel()) for draw in index_draws]
    packed_indices = torch.cat(index_draws)
    if packed_indices.device.type != "cpu":
        packed_indices = packed_indices.to("cpu")
    cpu_draws = packed_indices.split(lengths)
    plans = []
    for (
        teacher_draw,
        student_draw,
        permutation,
        actor_draw,
        actor_teacher_draw,
    ) in records:
        teacher_indices = (
            torch.empty(0, dtype=torch.long)
            if teacher_draw is None
            else cpu_draws[teacher_draw]
        )
        student_indices = (
            torch.empty(0, dtype=torch.long)
            if student_draw is None
            else valid_student_indices.index_select(0, cpu_draws[student_draw])
        )
        actor_indices = (
            None
            if actor_draw is None
            else valid_student_indices.index_select(0, cpu_draws[actor_draw])
        )
        actor_teacher_indices = (
            None if actor_teacher_draw is None else cpu_draws[actor_teacher_draw]
        )
        plans.append(
            _TD3ReplaySamplePlan(
                teacher_indices=teacher_indices,
                student_indices=student_indices,
                permutation=permutation,
                actor_indices=actor_indices,
                actor_teacher_indices=actor_teacher_indices,
            )
        )
    return tuple(plans)


@dataclass
class DistributionalTD3TeacherBCConfig(PPOConfig):
    _target_: str = (
        "active_adaptation.learning.ppo.td3_bc_dagger.DistributionalTD3TeacherBC"
    )
    name: str = "td3_bc_dagger"
    phase: str = "finetune"
    vecnorm: str = "eval"
    enable_residual_distillation: bool = False

    dagger_control_mode: str = "safe"
    dagger_safe_takeover_rms: float = 0.12
    dagger_safe_release_rms: float = 0.08
    dagger_safe_min_teacher_steps: int = 8
    dagger_safe_zero_iteration: int | None = None
    dagger_beta_start: float = 1.0
    dagger_beta_end: float = 0.0
    dagger_beta_decay_rollouts: int = 1800
    dagger_beta_zero_iteration: int | None = None
    dagger_seed: int = 0
    # Explicit finite support for every action consumed by the environment,
    # replay, BC, and Q. Nominal soft-limit coordinates remain the Q/BC scale.
    action_support_clip: float = 20.0
    dagger_bc_lr: float = 3e-4
    dagger_actor_huber_delta: float = 1.0
    dagger_buffer_capacity: int = 131_072
    dagger_buffer_device: str = "cpu"
    dagger_batch_size: int = 4096
    # Optional GA-DDPG-style expert-row share in every Actor update.  Teacher
    # rows come from q_teacher_replay and use their executed action as the exact
    # BC label; zero preserves the original main-DAgger-only Actor sampling.
    teacher_actor_replay_fraction: float = 0.5
    # Loss weight assigned to frozen Teacher raw replay in every existing
    # adaptation optimizer step; the remainder weights the live rollout loss.
    # The raw replay stores sensor/model inputs rather than stale latents.
    teacher_perception_replay_fraction: float = 0.5
    # Teacher rows are internally split 70/30 between the complete frozen
    # prefill ring and phases where the live Student recently failed.
    failure_phase_teacher_fraction: float = 0.3
    failure_phase_lookback_steps: int = 50
    failure_phase_samples_per_failure: int = 10
    failure_phase_num_bins: int = 1024
    # Successful-only Teacher collection continues until q_teacher_replay is
    # full. This guard prevents an indefinitely failing Teacher from hanging.
    teacher_prefill_max_rollouts: int = 1000
    dagger_replay_raw_observations: bool = True
    replay_raw_observation_keys: tuple[str, ...] = (
        VEL_CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        CMD_KEY,
        DEPTH_KEY,
    )
    perception_replay_burn_in: int = 8
    perception_encode_microbatch_size: int = 128
    teacher_perception_batch_size: int = 128
    # ``online_student_rollout`` delegates perception training exactly to
    # PPOVEL.train_adapt: two epochs over the current recurrent rollout and no
    # perception replay sampling.  The older modes remain available only for
    # explicitly configured TD3/checkpoint-compatibility paths.
    perception_replay_mode: str = "legacy_online_student"
    perception_replay_batch_size: int = 128
    # Optional reset-exact diagnostic for collection-cache representation
    # drift.  Zero disables every raw journal allocation/transfer.  Enabled
    # probes retain only a few complete Student episodes and never feed their
    # re-encoded values back into optimization.
    perception_staleness_probe_num_envs: int = 0
    perception_staleness_probe_max_episodes: int = 8
    perception_staleness_probe_max_generation_age: int = 8
    perception_staleness_probe_interval: int = 1
    # Teacher-replay-only supervised updates performed exactly once after the
    # successful-only prefill ring reaches capacity and before Student control
    # begins. Zero keeps the old no-warm-up behavior.
    teacher_perception_warmup_steps: int = 128
    perception_depth_codec: str = PERCEPTION_DEPTH_CODEC
    # Optional perception warm start. A complete Student checkpoint overlays
    # all seven online/EMA children. A phase=train PPOVEL checkpoint may instead
    # provide only object/adaptation children while depth remains freshly
    # initialized and trainable, matching PPOVEL finetune initialization.
    load_pretrained_perception: bool = False
    perception_checkpoint_path: str | None = None
    train_perception: bool = True

    eta_td3: float = 1.0
    lambda_bc: float = 1.0
    policy_delay: int = 2
    target_policy_noise_std: float = 0.2
    target_policy_noise_clip: float = 0.5
    collector_exploration_noise_std: float = 0.1
    collector_exploration_noise_clip: float = 0.5
    td3_learning_starts: int = 8192

    q_hidden_dim: int = 768
    q_num_atoms: int = 501
    q_v_min: float = -20.0
    q_v_max: float = 20.0
    q_layer_norm: bool = True
    q_action_fusion: str = FASTSAC_Q_DEFAULT_ACTION_FUSION
    q_action_coordinates: str = "raw_joint_command"
    q_normalize_actions: bool = True
    q_action_input_gain: float = 1.0
    # Q-only privileged actuator context.  The Actor observation, issued-action
    # distribution, reward, entropy, and BC coordinates remain unchanged.
    q_condition_on_actuator_state: bool = False
    # Analytically expose the candidate command's causal actuator response.
    # This reuses the Q-only delay/alpha context and previous issued command;
    # it never changes Actor observations, rewards, or environment dynamics.
    q_use_predicted_effect: bool = False
    # Delay/actuator-alpha conditioned residual FiLM on the Q action stem.
    # The zero-initialized bounded modulation is exactly identity at startup.
    q_use_residual_film: bool = False
    q_residual_film_scale: float = 0.1
    q_lr: float = 3e-5
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    q_tau: float = 0.005
    q_max_grad_norm: float = 1.0
    q_batch_size: int = 512
    q_updates_per_rollout: int = 128
    # Optional row-level scheduler.  When configured, each newly accepted
    # Student replay row earns this many total Q sample-row credits.  Dividing
    # by q_batch_size makes the applied update count invariant to num_envs and
    # rollout chunking; q_updates_per_rollout remains the legacy fallback.
    q_update_to_data_ratio: float | None = None
    # Teacher row share in each Q batch. The realized count uses nearest-integer
    # half-up rounding; the remaining rows come from Student-executed replay.
    q_teacher_replay_ratio: float = 0.5
    q_teacher_buffer_capacity: int = 131_072

    # TD3 keeps only the two online learning rings.  The duplicate
    # teacher_replay_buffer.h5/export FIFO is intentionally unsupported.
    save_teacher_buffer: bool = False


ConfigStore.instance().store(
    "td3_bc_dagger_finetune",
    node=DistributionalTD3TeacherBCConfig(
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


class _DistributionalTD3DaggerRolloutPolicy(_DaggerRolloutPolicy):
    """Locked DAgger source selection plus Student-only Q-coordinate noise."""

    @torch.no_grad()
    def forward(self, td: TensorDict):
        owner = self._owner
        teacher_prefill_active = owner._teacher_prefill_active()
        raw_student_action = owner._student_raw_action_proposal(td)
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

        teacher_action = owner._teacher_action(td.clone(False))
        valid = _valid_raw_action_rows(teacher_action)
        finite_teacher_action = torch.nan_to_num(
            teacher_action, nan=0.0, posinf=0.0, neginf=0.0
        )
        bounded_teacher_action = owner._project_execution_action(finite_teacher_action)
        student_valid = torch.isfinite(raw_student_action).all(dim=-1)
        finite_student_proposal = torch.nan_to_num(
            raw_student_action, nan=0.0, posinf=0.0, neginf=0.0
        )
        bounded_student_action = owner._bounded_actor_mean(finite_student_proposal)
        discrepancy_rms, discrepancy_max = _joint_normalized_action_discrepancy(
            bounded_student_action,
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

        control_mode = owner._effective_control_mode()
        safe_teacher_mask = torch.zeros_like(valid)
        safe_unsafe = torch.zeros_like(valid)
        safe_takeover = torch.zeros_like(valid)
        safe_release = torch.zeros_like(valid)
        if teacher_prefill_active:
            # Prefill is an algorithm-local phase, not beta=1 DAgger.  Force
            # every valid Teacher row without advancing the DAgger RNG or the
            # SafeDAgger hysteresis.  Invalid Teacher rows use the deterministic
            # Student only as a safe environment fallback and are not retained.
            choose_teacher = valid
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
        if (~valid & ~student_valid).any():
            raise RuntimeError(
                "Neither Teacher nor Student produced a finite raw action"
            )
        # A finite Teacher is the only valid fallback for a non-finite Student,
        # independent of the configured beta/safety controller.
        choose_teacher = choose_teacher | (valid & ~student_valid)
        issued_action, exploratory_student, collector_noise = (
            _apply_student_collector_noise(
                bounded_student_action,
                bounded_teacher_action,
                ~choose_teacher,
                (
                    0.0
                    if teacher_prefill_active
                    else float(owner.cfg.collector_exploration_noise_std)
                ),
                float(owner.cfg.collector_exploration_noise_clip),
                owner._fastsac_action_low,
                owner._fastsac_action_high,
                owner._fastsac_q_action_center,
                owner._fastsac_q_action_scale,
                float(owner.cfg.q_action_input_gain),
                owner.collector_exploration_rng,
            )
        )

        td[ACTION_KEY] = issued_action
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
        td[TD3_NOISE_FREE_STUDENT_ACTION_KEY] = bounded_student_action
        td[TD3_EXPLORATORY_STUDENT_ACTION_KEY] = exploratory_student
        td[TD3_COLLECTOR_NOISE_KEY] = collector_noise
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


class _DeterministicTD3StudentEvalPolicy(nn.Module):
    """Run the EMA Student mean without probabilistic Actor ``forward``."""

    def __init__(self, owner: "DistributionalTD3TeacherBC"):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        # Register the referenced modules (not the owner) so policy.eval()
        # propagates the evaluation mode through the complete Student path.
        if hasattr(owner, "temporal_depth_gru_ema"):
            self.temporal_depth_gru_ema = owner.temporal_depth_gru_ema
        if bool(owner.cfg.use_object_adapt):
            self.object_adapt_ema = owner.object_adapt_ema
            self.object_pred_transform = owner.object_pred_transform
        self.adapt_ema = owner.adapt_ema
        self.actor_adapt = owner.actor_adapt

    @torch.no_grad()
    def forward(self, td: TensorDict):
        # _student_mean_action is the authoritative EMA-perception path and calls
        # actor_adapt.get_dist(td).mean directly.  It never samples and never
        # asks TorchRL's ProbabilisticActor to compute sample_log_prob.
        action = self._owner._student_mean_action(td)
        if not torch.isfinite(action).all():
            raise RuntimeError("TD3 evaluation Actor produced non-finite raw actions")
        td[ACTION_KEY] = action
        return td


class DistributionalTD3TeacherBC(PPOBCDaggerFinetune):
    """Deterministic C51 TD3 with a single combined Teacher-BC Actor step."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_td3_config(cfg)
        # Bypass PPOBCDaggerFinetune.__init__: it constructs a stochastic
        # next-action adapter that is outside this method.  PPOVEL owns the
        # locked Actor/perception topology; inherited DAgger helpers are pure.
        PPOVEL.__init__(
            self, cfg, observation_spec, action_spec, reward_spec, device, env
        )
        nominal_low, nominal_high, offset_low, offset_high = (
            _vaic_nominal_action_coordinates(env, device)
        )
        _validate_action_safety_clip(
            nominal_low, nominal_high, float(cfg.action_support_clip)
        )
        action_low = torch.full_like(nominal_low, -float(cfg.action_support_clip))
        action_high = torch.full_like(nominal_high, float(cfg.action_support_clip))
        self._fastsac_action_low = action_low.detach()
        self._fastsac_action_high = action_high.detach()
        self._fastsac_actor_action_center = ((action_low + action_high) * 0.5).detach()
        self._fastsac_actor_action_scale = ((action_high - action_low) * 0.5).detach()
        self._fastsac_q_action_center = ((nominal_low + nominal_high) * 0.5).detach()
        self._fastsac_q_action_scale = ((nominal_high - nominal_low) * 0.5).detach()
        self._fastsac_joint_offset_low = offset_low.detach()
        self._fastsac_joint_offset_high = offset_high.detach()
        self._fastsac_action_contract = _bounded_action_contract_metadata(
            self.joint_names,
            nominal_low,
            nominal_high,
            offset_low,
            offset_high,
            action_low,
            action_high,
        )

        command_key = (
            "command_"
            if observation_spec.get("command_", None) is not None
            else CMD_KEY
        )
        self.q_actor_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        self.q_critic_keys = [OBS_PRIV_KEY, OBS_KEY, command_key]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            self.q_critic_keys.append(OBJECT_KEY)
        self._q_actor_widths = [
            int(observation_spec[VEL_CMD_KEY].shape[-1]),
            int(observation_spec[OBS_KEY].shape[-1]),
            int(cfg.latent_dim),
        ]
        self._q_critic_widths = [
            int(observation_spec[key].shape[-1]) for key in self.q_critic_keys
        ]
        self._q_actor_dim = sum(self._q_actor_widths)
        self._q_critic_dim = sum(self._q_critic_widths)
        self._q_actuator_context_metadata_value = (
            self._resolve_q_actuator_context_metadata()
        )
        self._q_actuator_context_dim = int(
            self._q_actuator_context_metadata_value.get("dimension", 0)
        )
        self._q_actuator_parameter_context_dim = (
            int(
                self._q_actuator_context_metadata_value["delay_range"][1]
                - self._q_actuator_context_metadata_value["delay_range"][0]
                + 2
            )
            if self._q_conditions_on_actuator_state()
            else 0
        )
        if self._q_uses_predicted_effect():
            self._q_action_input_dim = (
                (2 + Q_PREDICTED_EFFECT_INTERVALS) * self.action_dim
                + self._q_actuator_parameter_context_dim
            )
        else:
            self._q_action_input_dim = self.action_dim + self._q_actuator_context_dim
        self.qnet = _build_isolated_q_network(
            self._q_critic_dim,
            self._q_action_input_dim,
            cfg.q_hidden_dim,
            cfg.q_num_atoms,
            cfg.q_v_min,
            cfg.q_v_max,
            cfg.q_layer_norm,
            device,
            cfg.q_seed,
            cfg.q_action_fusion,
            residual_film_condition_dim=(
                self._q_actuator_parameter_context_dim
                if self._q_uses_residual_film()
                else 0
            ),
            residual_film_scale=float(cfg.q_residual_film_scale),
        )
        self.qnet_target = copy.deepcopy(self.qnet).requires_grad_(False)
        self.actor_target = None

        fused = str(device).startswith("cuda")
        self.critic_optimizer = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=cfg.q_lr,
            weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95),
            fused=fused,
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_adapt.parameters(),
            lr=cfg.dagger_bc_lr,
            weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95),
            fused=fused,
        )
        self.opt_policy = None
        self.opt_critic = None
        self.actor_backend = ACTOR_BACKEND
        self._freeze_teacher()

        replay_device = _resolve_replay_device(cfg.dagger_buffer_device, device)
        self.dagger_replay = _TD3DeviceReplay(cfg.dagger_buffer_capacity, replay_device)
        self.q_teacher_replay = _TD3DeviceReplay(
            cfg.q_teacher_buffer_capacity, replay_device
        )
        self.teacher_replay = None
        object.__setattr__(self, "_replay_vecnorm", None)
        self._replay_vecnorm_keys = set()
        self._replay_vecnorm_fingerprint = None
        self._rollout_final_batch = None
        self._rollout_q_actuator_contexts: list[torch.Tensor] = []
        self._truncation_final_batches = []
        self._last_truncation_finals_used = 0
        self._perception_replay_history = None
        self._perception_replay_history_count = 0
        # Values are replayed per raw state.  This fingerprint describes the
        # stable shape/dtype contract.  The ID mapping is append-only, and its
        # lifetime is reset atomically with the Teacher raw store; together the
        # raw-store identity/generation and this contract fully define cache
        # lineage without invalidating it for unrelated later appends.
        self._replay_object_geo_fingerprint = None
        self._replay_object_geo_bank = None
        self._replay_object_geo_hash_index: dict[str, list[int]] = {}
        self._replay_object_geo_bank_generation = 0
        self._replay_object_geo_device_banks: dict[
            tuple[str, torch.dtype], tuple[int, torch.Tensor]
        ] = {}
        self._failure_phase_histogram = torch.zeros(
            int(cfg.failure_phase_num_bins), dtype=torch.float64, device="cpu"
        )
        self._failure_phase_history: list[list[float]] | None = None
        self._failure_phase_student_history: list[list[bool]] | None = None
        self._failure_phase_takeover_history: list[list[bool]] | None = None
        self._teacher_phase_bin_rows: tuple[torch.Tensor, ...] = ()
        self._teacher_phase_nearest_nonempty = torch.empty(0, dtype=torch.long)
        self._teacher_phase_flat_rows = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_starts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_counts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_device_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._verified_teacher_focus_device_cache: dict[
            str,
            tuple[
                tuple,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._failure_histogram_device_cache: dict[str, tuple[int, torch.Tensor]] = {}
        self._teacher_phase_index_ready = False
        self._failure_phase_episode_count = 0
        self._failure_phase_anchor_count = 0
        self._failure_phase_uniform_fallback_rows = 0
        self._failure_phase_focused_rows = 0
        # Prefill rows remain pending by environment until their episode ends.
        # This prevents the prefix of a later-failed Teacher trajectory from
        # leaking into the frozen Teacher replay.
        self._teacher_prefill_pending: list[list[dict[str, torch.Tensor]]] | None = None
        self._teacher_prefill_raw_pending: (
            list[list[dict[str, torch.Tensor]]] | None
        ) = None
        self._teacher_episode_store = TeacherEpisodeSequenceStore(
            is_init_key=PERCEPTION_IS_INIT_KEY
        )
        self._teacher_actor_cache = CurrentEMATeacherActorCache(self._q_actor_dim)
        self._perception_ema_generation = 0
        self._teacher_ring_cache_lineage: TeacherActorCacheLineage | None = None
        self._teacher_episode_device_raw_fields = None
        self._teacher_episode_device_raw_lineage = None
        self._student_perception_drift_pending: list[dict | None] | None = None
        self._student_perception_drift_episodes: list[
            _StudentPerceptionDriftEpisode
        ] = []
        self._student_perception_drift_completed = 0
        self._student_perception_drift_discarded_incomplete = 0
        self._teacher_prefill_successful_episodes = 0
        # Per-motion commit counts.  Failure Teacher matching is same-motion,
        # so a motion whose successful prefill is starved silently weakens its
        # own curriculum; the total alone cannot show that.
        self._teacher_prefill_successful_by_motion: dict[int, int] = {}
        self._teacher_prefill_failed_episodes = 0
        self._teacher_prefill_timeout_episodes = 0
        self._teacher_prefill_incomplete_episodes = 0
        self._teacher_prefill_discarded_rows = 0

        generator_device = torch.device(device)
        self.dagger_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.dagger_seed)
        )
        self.q_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed)
        )
        self.collector_exploration_rng = torch.Generator(
            device=generator_device
        ).manual_seed(int(cfg.q_seed) + 1)
        self.target_policy_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 2
        )
        self.teacher_perception_rng = torch.Generator(
            device=generator_device
        ).manual_seed(int(cfg.q_seed) + 3)
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self._teacher_prefill_complete = False
        self._teacher_perception_warmup_complete = False
        self._teacher_perception_warmup_updates = 0
        self._last_teacher_perception_warmup_metrics: dict[str, float] = {}
        self.critic_update_count = 0
        self.actor_update_count = 0
        self.q_update_row_credit = 0.0
        self._perception_optimizer = self.opt_adapt
        self._perception_initialization = {
            "semantics": PERCEPTION_WARMSTART_SEMANTICS,
            "mode": PERCEPTION_WARMSTART_MODE_FRESH,
            "loaded": False,
            "source_path": None,
            "source_algorithm": None,
            "source_phase": None,
            "source_iter": None,
            "modules": (),
            "fresh_modules": PRETRAINED_PERCEPTION_MODULES,
            "trainable": bool(cfg.train_perception),
        }

    @staticmethod
    def _validate_td3_config(cfg) -> None:
        try:
            configured_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("WORLD_SIZE must be an integer") from exc
        process_group_world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if configured_world_size != 1 or process_group_world_size != 1:
            raise RuntimeError(
                "TD3/FastSAC BC-DAgger supports one process only; custom Actor, "
                "Critic, temperature, and perception gradients are not "
                "distributed-synchronized"
            )
        _linear_teacher_probability(
            cfg.dagger_beta_start,
            cfg.dagger_beta_end,
            cfg.dagger_beta_decay_rollouts,
            0,
        )
        if cfg.phase != "finetune" or cfg.vecnorm != "eval":
            raise ValueError("TD3 Teacher-BC requires phase=finetune and vecnorm=eval")
        if bool(cfg.enable_residual_distillation):
            raise ValueError("TD3 Teacher-BC owns the only Actor optimizer")
        if not bool(cfg.dagger_replay_raw_observations):
            raise ValueError("TD3 replay requires raw pre-VecNorm observations")
        if not bool(cfg.use_depth) or not bool(cfg.use_object_adapt):
            raise ValueError(
                "TD3 raw perception replay requires depth and object adapt"
            )
        if str(cfg.adapt_module) != "gru":
            raise ValueError("TD3 raw perception replay requires recurrent adaptation")
        if not isinstance(cfg.load_pretrained_perception, bool):
            raise ValueError("load_pretrained_perception must be boolean")
        if not isinstance(cfg.train_perception, bool):
            raise ValueError("train_perception must be boolean")
        perception_path = cfg.perception_checkpoint_path
        if bool(cfg.load_pretrained_perception):
            if not isinstance(perception_path, str) or not perception_path.strip():
                raise ValueError(
                    "load_pretrained_perception=true requires a non-empty "
                    "perception_checkpoint_path"
                )
        elif perception_path is not None:
            raise ValueError(
                "perception_checkpoint_path must be null when "
                "load_pretrained_perception=false"
            )
        if not bool(cfg.train_perception) and not bool(cfg.load_pretrained_perception):
            raise ValueError(
                "train_perception=false requires load_pretrained_perception=true"
            )
        expected_raw_keys = (
            VEL_CMD_KEY,
            OBS_KEY,
            OBS_PRIV_KEY,
            CMD_KEY,
            DEPTH_KEY,
        )
        if tuple(cfg.replay_raw_observation_keys) != expected_raw_keys:
            raise ValueError(
                "TD3 raw replay keys must include pre-VecNorm depth in locked order"
            )
        if int(cfg.perception_replay_burn_in) != 8:
            raise ValueError("perception_replay_burn_in is locked to 8")
        if str(cfg.perception_depth_codec) != PERCEPTION_DEPTH_CODEC:
            raise ValueError(
                f"perception_depth_codec must be {PERCEPTION_DEPTH_CODEC!r}"
            )
        if bool(cfg.save_teacher_buffer):
            raise ValueError(
                "save_teacher_buffer must be false; TD3 never writes a teacher H5"
            )
        if str(cfg.dagger_control_mode) not in DAGGER_CONTROL_MODES:
            raise ValueError(f"invalid dagger_control_mode={cfg.dagger_control_mode!r}")
        if (
            int(cfg.q_num_atoms) != 501
            or not math.isclose(float(cfg.q_v_min), -20.0)
            or not math.isclose(float(cfg.q_v_max), 20.0)
        ):
            raise ValueError(
                "distributional TD3 requires the locked 501-atom [-20,20] support"
            )
        if not bool(cfg.q_layer_norm) or str(cfg.q_action_fusion) not in (
            FASTSAC_Q_DEFAULT_ACTION_FUSION,
            "balanced",
        ):
            raise ValueError(
                "distributional TD3 requires late or balanced-fusion LayerNorm Q"
            )
        if str(cfg.q_action_coordinates) != "raw_joint_command" or not bool(
            cfg.q_normalize_actions
        ):
            raise ValueError(
                "distributional TD3 requires normalized raw-joint-command Q actions"
            )
        if not isinstance(
            getattr(cfg, "q_condition_on_actuator_state", False), bool
        ):
            raise ValueError("q_condition_on_actuator_state must be a boolean")
        if not isinstance(getattr(cfg, "q_use_predicted_effect", False), bool):
            raise ValueError("q_use_predicted_effect must be a boolean")
        if bool(getattr(cfg, "q_use_predicted_effect", False)) and not bool(
            getattr(cfg, "q_condition_on_actuator_state", False)
        ):
            raise ValueError(
                "q_use_predicted_effect requires q_condition_on_actuator_state=true"
            )
        if not isinstance(getattr(cfg, "q_use_residual_film", False), bool):
            raise ValueError("q_use_residual_film must be a boolean")
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
        positive_integers = (
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
            "teacher_perception_batch_size",
            "perception_replay_batch_size",
            "failure_phase_lookback_steps",
            "failure_phase_samples_per_failure",
            "failure_phase_num_bins",
        )
        for name in positive_integers:
            value = getattr(cfg, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        probe_num_envs = getattr(cfg, "perception_staleness_probe_num_envs", 0)
        if isinstance(probe_num_envs, bool) or int(probe_num_envs) < 0:
            raise ValueError(
                "perception_staleness_probe_num_envs must be a non-negative integer"
            )
        for name in (
            "perception_staleness_probe_max_episodes",
            "perception_staleness_probe_max_generation_age",
            "perception_staleness_probe_interval",
        ):
            value = getattr(cfg, name, 1)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        perception_replay_mode = str(
            getattr(cfg, "perception_replay_mode", "legacy_online_student")
        )
        if perception_replay_mode not in (
            "legacy_online_student",
            "four_way",
            ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
        ):
            raise ValueError(
                "perception_replay_mode must be 'online_student_rollout', "
                "'legacy_online_student', or 'four_way'"
            )
        if perception_replay_mode == "four_way" and bool(cfg.train_dr_estimator):
            raise ValueError(
                "four_way perception replay has no raw replay target for the "
                "DR estimator; set train_dr_estimator=false"
            )
        canonical_names = tuple(
            f"{purpose}_{source}_fraction"
            for purpose in ("q", "actor", "perception")
            for source in REPLAY_SOURCE_ORDER
        )
        present_canonical = tuple(
            name for name in canonical_names if hasattr(cfg, name)
        )
        if present_canonical and len(present_canonical) != len(canonical_names):
            missing = sorted(set(canonical_names).difference(present_canonical))
            raise ValueError(
                "canonical four-way replay mix is incomplete; missing "
                + ", ".join(missing)
            )
        if present_canonical:
            for purpose in ("q", "actor", "perception"):
                allocate_source_counts(
                    1,
                    {
                        source: getattr(cfg, f"{purpose}_{source}_fraction")
                        for source in REPLAY_SOURCE_ORDER
                    },
                )
        if perception_replay_mode == "four_way" and not present_canonical:
            raise ValueError(
                "perception_replay_mode='four_way' requires all canonical replay fractions"
            )
        if perception_replay_mode == "four_way" and bool(cfg.train_dr_estimator):
            raise ValueError(
                "four-way perception replay does not store the DR-estimator target"
            )
        max_phase_distance = getattr(cfg, "max_teacher_phase_match_distance", None)
        if max_phase_distance is not None and (
            isinstance(max_phase_distance, bool)
            or not isinstance(max_phase_distance, (int, float))
            or not math.isfinite(float(max_phase_distance))
            or float(max_phase_distance) < 0.0
        ):
            raise ValueError(
                "max_teacher_phase_match_distance must be null or finite and "
                "non-negative"
            )
        q_update_to_data_ratio = getattr(cfg, "q_update_to_data_ratio", None)
        if q_update_to_data_ratio is not None and (
            isinstance(q_update_to_data_ratio, bool)
            or not math.isfinite(float(q_update_to_data_ratio))
            or float(q_update_to_data_ratio) <= 0.0
        ):
            raise ValueError(
                "q_update_to_data_ratio must be null or finite and positive"
            )
        prefill_max_rollouts = cfg.teacher_prefill_max_rollouts
        if isinstance(prefill_max_rollouts, bool) or int(prefill_max_rollouts) < 1:
            raise ValueError("teacher_prefill_max_rollouts must be a positive integer")
        warmup_steps = cfg.teacher_perception_warmup_steps
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or warmup_steps < 0
        ):
            raise ValueError(
                "teacher_perception_warmup_steps must be a non-negative integer"
            )
        if perception_replay_mode == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE:
            if float(cfg.teacher_perception_replay_fraction) != 0.0:
                raise ValueError(
                    "online_student_rollout perception requires "
                    "teacher_perception_replay_fraction=0"
                )
            if int(warmup_steps) != 0:
                raise ValueError(
                    "online_student_rollout perception requires "
                    "teacher_perception_warmup_steps=0"
                )
            if present_canonical:
                expected_perception_mix = {
                    "uniform_student": 1.0,
                    "failure_student": 0.0,
                    "uniform_teacher": 0.0,
                    "failure_teacher": 0.0,
                }
                for source, expected in expected_perception_mix.items():
                    actual = float(
                        getattr(cfg, f"perception_{source}_fraction")
                    )
                    if actual != expected:
                        raise ValueError(
                            "online_student_rollout perception requires "
                            "perception_uniform_student_fraction=1 and every "
                            "perception replay fraction=0"
                        )
        if int(cfg.q_teacher_buffer_capacity) < int(cfg.td3_learning_starts):
            raise ValueError("q_teacher_buffer_capacity must cover td3_learning_starts")
        for name in (
            "teacher_actor_replay_fraction",
            "teacher_perception_replay_fraction",
            "q_teacher_replay_ratio",
            "failure_phase_teacher_fraction",
        ):
            fraction = getattr(cfg, name)
            if (
                isinstance(fraction, bool)
                or not math.isfinite(float(fraction))
                or not 0.0 <= float(fraction) <= 1.0
            ):
                raise ValueError(f"{name} must be in [0,1]")
        if (
            int(cfg.failure_phase_samples_per_failure)
            > int(cfg.failure_phase_lookback_steps) + 1
        ):
            raise ValueError(
                "failure_phase_samples_per_failure cannot exceed the inclusive "
                "failure_phase_lookback_steps + 1 interval"
            )
        for name in (
            "action_support_clip",
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "q_lr",
            "q_action_input_gain",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
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
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(cfg.eta_td3) == 0.0 and float(cfg.lambda_bc) == 0.0:
            raise ValueError("eta_td3 and lambda_bc cannot both be zero")
        if not 0.0 <= float(cfg.q_tau) <= 1.0:
            raise ValueError("q_tau must be in [0,1]")
        release = float(cfg.dagger_safe_release_rms)
        takeover = float(cfg.dagger_safe_takeover_rms)
        if not (
            math.isfinite(release)
            and math.isfinite(takeover)
            and 0.0 <= release < takeover
        ):
            raise ValueError("SafeDAgger release/takeover thresholds are invalid")

    @staticmethod
    def _validate_perception_module_state(
        name: str,
        target_state: Mapping,
        source_state: Mapping,
    ) -> None:
        """Validate one child completely before any perception child mutates."""
        target_keys = set(target_state)
        source_keys = set(source_state)
        if target_keys != source_keys:
            missing = sorted(target_keys.difference(source_keys))
            unexpected = sorted(source_keys.difference(target_keys))
            raise RuntimeError(
                f"pretrained perception module {name!r} has incompatible keys; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, target_value in target_state.items():
            source_value = source_state[key]
            if torch.is_tensor(target_value):
                if not torch.is_tensor(source_value):
                    raise RuntimeError(
                        f"pretrained perception module {name!r} key {key!r} "
                        "is not a tensor"
                    )
                if target_value.shape != source_value.shape:
                    raise RuntimeError(
                        f"pretrained perception module {name!r} key {key!r} "
                        f"shape mismatch: expected {tuple(target_value.shape)}, "
                        f"got {tuple(source_value.shape)}"
                    )
                if target_value.dtype != source_value.dtype:
                    raise RuntimeError(
                        f"pretrained perception module {name!r} key {key!r} "
                        f"dtype mismatch: expected {target_value.dtype}, "
                        f"got {source_value.dtype}"
                    )
            elif type(target_value) is not type(source_value):
                raise RuntimeError(
                    f"pretrained perception module {name!r} key {key!r} "
                    "has an incompatible value type"
                )

    @staticmethod
    def _validate_depth_checkpoint_aliases(policy_state: Mapping) -> None:
        """Require the standalone and temporal aliases to describe one CNN."""
        standalone = policy_state["depth_cnn"]
        temporal = policy_state["temporal_depth_gru"]
        prefix = "depth_cnn."
        nested = {
            key[len(prefix) :]: value
            for key, value in temporal.items()
            if str(key).startswith(prefix)
        }
        # Tiny unit-test modules and older generic checkpoints can expose the
        # two required children without registering the CNN as a nested alias.
        # Real VAIC TemporalDepthGRU checkpoints do expose it; validate exact
        # equality whenever that authoritative alias is present.
        if not nested:
            return
        if set(nested) != set(standalone):
            raise RuntimeError(
                "pretrained perception depth_cnn aliases have incompatible keys"
            )
        for key, standalone_value in standalone.items():
            nested_value = nested[key]
            if torch.is_tensor(standalone_value):
                if not torch.equal(standalone_value, nested_value):
                    raise RuntimeError(
                        "pretrained perception depth_cnn standalone/temporal "
                        f"alias mismatch at {key!r}"
                    )
            elif standalone_value != nested_value:
                raise RuntimeError(
                    "pretrained perception depth_cnn standalone/temporal "
                    f"alias mismatch at {key!r}"
                )

    def _load_pretrained_perception_checkpoint(self, path) -> dict:
        """Strictly overlay a full Student or partial PPOVEL perception source.

        A phase=train PPOVEL checkpoint has no depth stack: PPOVEL finetune
        creates that stack in its constructor, then trains it jointly with the
        four object/adaptation children loaded from the train checkpoint.  The
        partial mode below deliberately reproduces that behavior.  Any other
        subset is rejected instead of silently treating a damaged Student
        checkpoint as a valid partial warm start.
        """
        resolved_path = Path(path).expanduser().resolve(strict=True)
        checkpoint = torch.load(
            resolved_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                "pretrained perception checkpoint must be a top-level mapping"
            )
        policy_state = checkpoint.get("policy")
        if not isinstance(policy_state, Mapping):
            raise ValueError(
                "pretrained perception checkpoint must contain a policy mapping"
            )

        all_names = set(PRETRAINED_PERCEPTION_MODULES)
        partial_names = set(PPOVEL_PARTIAL_PERCEPTION_MODULES)
        present_names = {name for name in all_names if name in policy_state}
        if present_names == all_names:
            mode = PERCEPTION_WARMSTART_MODE_FULL_STUDENT
            selected_modules = PRETRAINED_PERCEPTION_MODULES
            fresh_modules: tuple[str, ...] = ()
        elif present_names == partial_names:
            if policy_state.get("last_phase") != "train":
                raise ValueError(
                    "partial PPOVEL perception checkpoint requires "
                    "policy.last_phase='train'"
                )
            if not bool(self.cfg.train_perception):
                raise ValueError(
                    "partial PPOVEL perception warm start requires "
                    "train_perception=true because its depth modules are "
                    "freshly initialized"
                )
            mode = PERCEPTION_WARMSTART_MODE_PPOVEL_PARTIAL
            selected_modules = PPOVEL_PARTIAL_PERCEPTION_MODULES
            fresh_modules = PPOVEL_PARTIAL_FRESH_DEPTH_MODULES
        else:
            missing = sorted(all_names.difference(present_names))
            present = sorted(present_names)
            raise ValueError(
                "pretrained perception checkpoint must contain either all "
                "seven Student perception modules or exactly the four PPOVEL "
                "train-phase object/adaptation modules; "
                f"present={present}, missing={missing}"
            )

        validated: list[tuple[str, nn.Module, Mapping]] = []
        for name in selected_modules:
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module):
                raise RuntimeError(
                    f"target policy lacks required perception module {name!r}"
                )
            source_state = policy_state.get(name)
            if not isinstance(source_state, Mapping):
                raise ValueError(
                    "pretrained perception checkpoint is missing required "
                    f"module mapping {name!r}"
                )
            self._validate_perception_module_state(
                name,
                module.state_dict(),
                source_state,
            )
            validated.append((name, module, source_state))
        if mode == PERCEPTION_WARMSTART_MODE_FULL_STUDENT:
            self._validate_depth_checkpoint_aliases(policy_state)

        # No target module is mutated until every selected mapping passes
        # strict key/shape/dtype validation. In partial mode this also leaves
        # all three constructor-created depth children untouched.
        for name, module, source_state in validated:
            try:
                module.load_state_dict(source_state, strict=True)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load pretrained perception module {name!r}"
                ) from exc

        policy_algorithm = policy_state.get("training_algorithm")
        metadata = {
            "semantics": PERCEPTION_WARMSTART_SEMANTICS,
            "mode": mode,
            "loaded": True,
            "source_path": str(resolved_path),
            "source_algorithm": policy_algorithm,
            "source_phase": policy_state.get("last_phase"),
            "source_iter": policy_state.get("last_iter"),
            "modules": selected_modules,
            "fresh_modules": fresh_modules,
            "trainable": bool(self.cfg.train_perception),
        }
        self._perception_initialization = metadata
        return copy.deepcopy(metadata)

    def _set_perception_trainable(self, trainable: bool) -> None:
        """Enable perception adaptation or freeze online and EMA stacks fully."""
        if not isinstance(trainable, bool):
            raise ValueError("perception trainable flag must be boolean")
        if not hasattr(self, "_perception_optimizer"):
            self._perception_optimizer = getattr(self, "opt_adapt", None)

        for name in _ONLINE_PERCEPTION_MODULES:
            module = getattr(self, name, None)
            if module is None:
                continue
            module.requires_grad_(trainable)
            module.train(trainable)
        for name in _EMA_PERCEPTION_MODULES:
            module = getattr(self, name, None)
            if module is None:
                continue
            module.requires_grad_(False)
            module.eval()

        if trainable:
            if getattr(self, "opt_adapt", None) is None:
                if self._perception_optimizer is None:
                    raise RuntimeError(
                        "cannot re-enable perception without its optimizer"
                    )
                self.opt_adapt = self._perception_optimizer
        else:
            if getattr(self, "opt_adapt", None) is not None:
                self._perception_optimizer = self.opt_adapt
            self.opt_adapt = None

        metadata = copy.deepcopy(
            getattr(
                self,
                "_perception_initialization",
                {
                    "semantics": PERCEPTION_WARMSTART_SEMANTICS,
                    "mode": PERCEPTION_WARMSTART_MODE_FRESH,
                    "loaded": False,
                    "source_path": None,
                    "modules": (),
                    "fresh_modules": PRETRAINED_PERCEPTION_MODULES,
                },
            )
        )
        metadata["trainable"] = trainable
        self._perception_initialization = metadata

    def _apply_perception_initialization_policy(self) -> None:
        """Apply the configured overlay once, then enforce update semantics."""
        if bool(self.cfg.load_pretrained_perception):
            configured_path = str(Path(self.cfg.perception_checkpoint_path).resolve())
            loaded = bool(self._perception_initialization.get("loaded", False))
            loaded_path = self._perception_initialization.get("source_path")
            if not loaded or loaded_path != configured_path:
                self._load_pretrained_perception_checkpoint(configured_path)
        self._set_perception_trainable(bool(self.cfg.train_perception))

    def _q_conditions_on_actuator_state(self) -> bool:
        return bool(getattr(self.cfg, "q_condition_on_actuator_state", False))

    def _q_uses_predicted_effect(self) -> bool:
        return bool(getattr(self.cfg, "q_use_predicted_effect", False))

    def _q_uses_residual_film(self) -> bool:
        return bool(getattr(self.cfg, "q_use_residual_film", False))

    def _resolve_q_actuator_context_metadata(self) -> dict:
        """Describe the episode-constant delay parameters used by Q only."""
        if not self._q_conditions_on_actuator_state():
            return {"enabled": False}
        manager = getattr(self.env, "action_manager", None)
        required = ("min_delay", "max_delay", "alpha_range", "delay", "alpha")
        missing = [name for name in required if not hasattr(manager, name)]
        if missing:
            raise ValueError(
                "q_condition_on_actuator_state requires a delayed JointPosition "
                f"action manager; missing attributes {missing}"
            )
        delay_min = manager.min_delay
        delay_max = manager.max_delay
        if (
            isinstance(delay_min, bool)
            or isinstance(delay_max, bool)
            or int(delay_min) != delay_min
            or int(delay_max) != delay_max
            or int(delay_min) > int(delay_max)
        ):
            raise ValueError(
                "q_condition_on_actuator_state requires ordered integer delay bounds"
            )
        alpha_range = manager.alpha_range
        if not isinstance(alpha_range, (list, tuple)) or len(alpha_range) != 2:
            raise ValueError(
                "q_condition_on_actuator_state requires a two-value alpha_range"
            )
        alpha_low, alpha_high = (float(value) for value in alpha_range)
        if (
            not math.isfinite(alpha_low)
            or not math.isfinite(alpha_high)
            or alpha_low > alpha_high
        ):
            raise ValueError(
                "q_condition_on_actuator_state requires finite ordered alpha bounds"
            )
        delay_min = int(delay_min)
        delay_max = int(delay_max)
        metadata = {
            "enabled": True,
            "semantics": FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
            "dimension": delay_max - delay_min + 2,
            "delay_range": [delay_min, delay_max],
            "alpha_range": [alpha_low, alpha_high],
        }
        if self._q_uses_predicted_effect():
            decimation = getattr(self.env, "decimation", None)
            if (
                isinstance(decimation, bool)
                or not isinstance(decimation, Integral)
                or int(decimation) < 1
            ):
                raise ValueError(
                    "q_use_predicted_effect requires a positive integer env decimation"
                )
            minimum_intervals = math.ceil(max(0, delay_max) / int(decimation)) + 1
            if Q_PREDICTED_EFFECT_INTERVALS < minimum_intervals:
                raise ValueError(
                    "Q predicted-effect horizon does not cover the configured delay: "
                    f"need at least {minimum_intervals} control intervals, got "
                    f"{Q_PREDICTED_EFFECT_INTERVALS}"
                )
            action_buf = getattr(manager, "action_buf", None)
            if (
                not isinstance(action_buf, torch.Tensor)
                or action_buf.ndim != 3
                or int(action_buf.shape[1]) != int(self.action_dim)
                or int(action_buf.shape[2]) < Q_PREDICTED_EFFECT_INTERVALS
            ):
                raise ValueError(
                    "q_use_predicted_effect requires an action buffer with at least "
                    f"{Q_PREDICTED_EFFECT_INTERVALS} command slots"
                )
            metadata.update(
                {
                    "semantics": FASTSAC_Q_PREDICTED_EFFECT_CONTEXT_SEMANTICS,
                    "dimension": delay_max - delay_min + 2 + int(self.action_dim),
                    "previous_action_dim": int(self.action_dim),
                    "control_decimation": int(decimation),
                    "effect_intervals": Q_PREDICTED_EFFECT_INTERVALS,
                }
            )
        return _normalize_q_actuator_context_metadata(metadata)

    @torch.no_grad()
    def _encode_q_actuator_context(
        self,
        delay: torch.Tensor,
        alpha: torch.Tensor,
        previous_action: torch.Tensor | None = None,
        *,
        validate_values: bool = True,
    ) -> torch.Tensor:
        metadata = self._q_actuator_context_metadata_value
        if not metadata["enabled"]:
            raise RuntimeError("Q actuator context is disabled")
        if delay.ndim < 1 or delay.shape[-1] != 1:
            raise ValueError(
                f"Actuator delay must end in one value, got {tuple(delay.shape)}"
            )
        if alpha.shape != delay.shape:
            raise ValueError(
                "Actuator delay and alpha shapes must match, got "
                f"{tuple(delay.shape)} and {tuple(alpha.shape)}"
            )
        if delay.device != alpha.device:
            raise ValueError("Actuator delay and alpha must share one device")
        delay_values = delay.squeeze(-1)
        delay_min, delay_max = metadata["delay_range"]
        alpha_low, alpha_high = metadata["alpha_range"]
        if validate_values:
            if delay_values.is_floating_point() and not torch.equal(
                delay_values, delay_values.round()
            ):
                raise ValueError("Actuator delay values must be integral")
            if ((delay_values < delay_min) | (delay_values > delay_max)).any():
                raise ValueError(
                    "Actuator delay is outside the checkpointed context range "
                    f"[{delay_min}, {delay_max}]"
                )
            if not torch.isfinite(alpha).all():
                raise ValueError("Actuator alpha contains non-finite values")
            if ((alpha < alpha_low) | (alpha > alpha_high)).any():
                raise ValueError(
                    "Actuator alpha is outside the checkpointed context range "
                    f"[{alpha_low}, {alpha_high}]"
                )
        delay_one_hot = F.one_hot(
            delay_values.long() - delay_min,
            num_classes=delay_max - delay_min + 1,
        ).to(dtype=torch.float32)
        if alpha_high == alpha_low:
            alpha_centered = torch.zeros_like(alpha, dtype=torch.float32)
        else:
            alpha_centered = (
                2.0 * (alpha.float() - alpha_low) / (alpha_high - alpha_low) - 1.0
            )
        context = torch.cat((delay_one_hot, alpha_centered), dim=-1)
        if self._q_uses_predicted_effect():
            expected_previous_shape = (*delay.shape[:-1], self.action_dim)
            if previous_action is None:
                raise ValueError(
                    "Q predicted effect requires the previous issued command"
                )
            if tuple(previous_action.shape) != expected_previous_shape:
                raise ValueError(
                    "Previous issued command shape does not match actuator context: "
                    f"got {tuple(previous_action.shape)}, expected "
                    f"{expected_previous_shape}"
                )
            if previous_action.device != delay.device:
                raise ValueError(
                    "Previous issued command and actuator context must share a device"
                )
            if not previous_action.is_floating_point():
                raise ValueError("Previous issued command must be floating point")
            if validate_values and not torch.isfinite(previous_action).all():
                raise ValueError("Previous issued command contains non-finite values")
            context = torch.cat((context, previous_action.float()), dim=-1)
        elif previous_action is not None:
            raise ValueError(
                "Previous issued command is only valid when predicted effect is enabled"
            )
        if context.shape[-1] != self._q_actuator_context_dim:
            raise RuntimeError("Encoded Q actuator-context dimension is inconsistent")
        return context

    @torch.no_grad()
    def capture_q_actuator_context(self) -> torch.Tensor | None:
        """Snapshot delay/alpha before environment step/reset mutates them."""
        if not self._q_conditions_on_actuator_state():
            return None
        manager = self.env.action_manager
        return self._encode_q_actuator_context(
            manager.delay,
            manager.alpha,
            (
                manager.action_buf[:, :, 0]
                if self._q_uses_predicted_effect()
                else None
            ),
            validate_values=False,
        ).detach().clone()

    def begin_transition_collection(self) -> None:
        """Start one non-interleaved rollout with an empty context journal."""
        self._rollout_q_actuator_contexts = []

    def record_rollout_q_actuator_context(
        self, context: torch.Tensor | None
    ) -> None:
        if not self._q_conditions_on_actuator_state():
            if context is not None:
                raise ValueError(
                    "Received actuator context while Q conditioning is disabled"
                )
            return
        if context is None:
            raise ValueError("Enabled Q actuator conditioning requires context")
        self._rollout_q_actuator_contexts.append(context.detach().clone())

    def _transition_q_actuator_context(
        self,
        context: torch.Tensor | None,
        row_count: int,
        device: torch.device | None,
    ) -> torch.Tensor | None:
        if not self._q_conditions_on_actuator_state():
            if context is not None:
                raise ValueError(
                    "Received transition context while Q conditioning is disabled"
                )
            return None
        if context is None:
            raise ValueError(
                "Enabled Q actuator conditioning requires transition context"
            )
        expected = (int(row_count), self._q_actuator_context_dim)
        if tuple(context.shape) != expected:
            raise ValueError(
                f"Transition actuator context has shape {tuple(context.shape)}, "
                f"expected {expected}"
            )
        if device is not None and context.device != device:
            raise ValueError(
                f"Transition actuator context is on {context.device}, expected {device}"
            )
        if not context.is_floating_point():
            raise ValueError("Transition actuator context must be floating point")
        torch._assert_async(
            torch.isfinite(context).all(),
            "Transition actuator context contains NaN/Inf",
        )
        return context.detach()

    def _consume_rollout_q_actuator_contexts(
        self, num_steps: int
    ) -> torch.Tensor | None:
        contexts = getattr(self, "_rollout_q_actuator_contexts", [])
        self._rollout_q_actuator_contexts = []
        if not self._q_conditions_on_actuator_state():
            if contexts:
                raise RuntimeError(
                    "Captured actuator contexts while conditioning is disabled"
                )
            return None
        if len(contexts) != int(num_steps):
            raise RuntimeError(
                "DAgger rollout actuator-context count does not match rollout "
                f"length: got {len(contexts)}, expected {int(num_steps)}"
            )
        return torch.stack(contexts, dim=1)

    def _q_actuator_parameter_dim(self) -> int:
        """Width of delay one-hot plus centered alpha inside stored context."""
        cached = getattr(self, "_q_actuator_parameter_context_dim", None)
        if cached is not None:
            return int(cached)
        metadata = getattr(self, "_q_actuator_context_metadata_value", None)
        if metadata is None:
            if self._q_uses_predicted_effect():
                raise RuntimeError(
                    "Q predicted effect requires actuator-context metadata"
                )
            return int(self._q_actuator_context_dim)
        if not metadata["enabled"]:
            return 0
        delay_min, delay_max = metadata["delay_range"]
        return int(delay_max - delay_min + 2)

    def _split_q_actuator_context(
        self,
        action: torch.Tensor,
        actuator_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Validate replay context and separate dynamics from prior command."""
        if actuator_context is None:
            raise ValueError(
                "q_condition_on_actuator_state=true requires context for every Q call"
            )
        expected = (*action.shape[:-1], self._q_actuator_context_dim)
        if tuple(actuator_context.shape) != expected:
            raise ValueError(
                "Q actuator-context shape does not match action: got "
                f"{tuple(actuator_context.shape)}, expected {expected}"
            )
        if actuator_context.device != action.device:
            raise ValueError("Q action and actuator context must share a device")
        torch._assert_async(
            torch.isfinite(actuator_context).all(),
            "Q actuator context contains NaN/Inf",
        )
        parameter_dim = self._q_actuator_parameter_dim()
        parameters = actuator_context[..., :parameter_dim].detach().to(
            dtype=action.dtype
        )
        previous_action = (
            actuator_context[..., parameter_dim:].detach().to(dtype=action.dtype)
            if self._q_uses_predicted_effect()
            else None
        )
        if self._q_uses_predicted_effect() and previous_action.shape[-1] != self.action_dim:
            raise RuntimeError(
                "Q predicted-effect context has an invalid previous-action width"
            )
        return parameters, previous_action

    def _append_q_actuator_context(
        self,
        q_action: torch.Tensor,
        actuator_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Append detached dynamics context to an already normalized command."""
        if not self._q_conditions_on_actuator_state():
            if actuator_context is not None:
                raise ValueError(
                    "Received Q actuator context while conditioning is disabled"
                )
            return q_action
        if self._q_uses_predicted_effect():
            raise RuntimeError(
                "Predicted-effect Q inputs must be built with _q_action_features"
            )
        parameters, previous_action = self._split_q_actuator_context(
            q_action, actuator_context
        )
        if previous_action is not None:
            raise RuntimeError("Plain Q actuator context unexpectedly has an action")
        return torch.cat((q_action, parameters), dim=-1)

    def _predicted_effect_interval_gains(
        self,
        parameter_context: torch.Tensor,
    ) -> torch.Tensor:
        """Exact mean post-lerp impulse gain for the current command.

        The two counterfactual actuator pipelines differ only at the current
        command slot: one inserts the candidate and the other holds u_(t-1).
        Their applied-action difference is therefore a scalar impulse gain
        times the per-joint command delta.  Computing the scalar response here
        is exact for JointPosition's queue plus lerp dynamics and avoids a
        learned forward-model approximation.
        """
        metadata = self._q_actuator_context_metadata_value
        delay_min, delay_max = metadata["delay_range"]
        delay_classes = int(delay_max - delay_min + 1)
        expected = (*parameter_context.shape[:-1], delay_classes + 1)
        if tuple(parameter_context.shape) != expected:
            raise ValueError(
                "Predicted-effect parameter context has shape "
                f"{tuple(parameter_context.shape)}, expected {expected}"
            )
        delay = (
            parameter_context[..., :delay_classes].argmax(dim=-1, keepdim=True)
            + int(delay_min)
        )
        alpha_centered = parameter_context[..., delay_classes:]
        alpha_low, alpha_high = metadata["alpha_range"]
        if alpha_high == alpha_low:
            alpha = torch.full_like(alpha_centered, float(alpha_low))
        else:
            alpha = float(alpha_low) + 0.5 * (alpha_centered + 1.0) * (
                float(alpha_high) - float(alpha_low)
            )
        decimation = int(metadata["control_decimation"])
        intervals = int(metadata["effect_intervals"])
        applied_gain = torch.zeros_like(alpha)
        interval_means: list[torch.Tensor] = []
        for interval in range(intervals):
            interval_sum = torch.zeros_like(alpha)
            for substep in range(decimation):
                selected_slot = (
                    delay - substep + decimation - 1
                ) // decimation
                selected_gain = (selected_slot == interval).to(dtype=alpha.dtype)
                applied_gain = (1.0 - alpha) * applied_gain + alpha * selected_gain
                interval_sum = interval_sum + applied_gain
            interval_means.append(interval_sum / float(decimation))
        return torch.cat(interval_means, dim=-1).detach()

    def _q_action_features(
        self,
        action: torch.Tensor,
        actuator_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map one issued-action candidate to the complete Q action branch."""
        return self._q_action_features_from_q_input(
            self._q_action_input(action), actuator_context
        )

    def _q_action_features_from_q_input(
        self,
        q_action: torch.Tensor,
        actuator_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build features from an already normalized executable command."""
        if not self._q_uses_predicted_effect():
            return self._append_q_actuator_context(q_action, actuator_context)
        parameters, previous_action = self._split_q_actuator_context(
            q_action, actuator_context
        )
        if previous_action is None:
            raise RuntimeError("Q predicted effect lacks its previous command")
        previous_q_action = self._q_action_input(previous_action)
        command_delta = q_action - previous_q_action
        gains = self._predicted_effect_interval_gains(parameters)
        effects = (gains.unsqueeze(-1) * command_delta.unsqueeze(-2)).flatten(-2)
        features = torch.cat(
            (q_action, command_delta, effects, parameters), dim=-1
        )
        if int(features.shape[-1]) != int(self._q_action_input_dim):
            raise RuntimeError(
                "Predicted-effect Q action feature dimension is inconsistent: "
                f"got {int(features.shape[-1])}, expected {self._q_action_input_dim}"
            )
        return features

    def _next_q_actuator_context(
        self,
        actuator_context: torch.Tensor,
        issued_action: torch.Tensor,
    ) -> torch.Tensor:
        """Advance only the previous-command component to the successor state."""
        if not self._q_uses_predicted_effect():
            return actuator_context
        _, previous_action = self._split_q_actuator_context(
            issued_action, actuator_context
        )
        if previous_action is None:
            raise RuntimeError("Q predicted effect lacks its previous command")
        next_context = actuator_context.detach().clone()
        next_context[..., self._q_actuator_parameter_dim() :] = (
            self._project_execution_action(issued_action).detach().to(next_context)
        )
        return next_context

    def _q_action_input(self, action: torch.Tensor) -> torch.Tensor:
        """Project to execution support, then normalize in nominal coordinates."""
        torch._assert_async(
            torch.isfinite(action).all(), "Q received a non-finite raw joint action"
        )
        bounded = self._project_execution_action(action)
        center = self._fastsac_q_action_center.to(bounded)
        scale = self._fastsac_q_action_scale.to(bounded)
        normalized = (bounded - center) / scale
        gain = float(self.cfg.q_action_input_gain)
        transformed = normalized if gain == 1.0 else normalized * gain
        # Avoid a host synchronization in this hot replay path while still
        # turning any finite-raw-action overflow into a device-side failure.
        torch._assert_async(
            torch.isfinite(transformed).all(),
            "raw joint action overflowed Q normalization",
        )
        return transformed

    def _q_action_to_physical(self, q_action: torch.Tensor) -> torch.Tensor:
        center = self._fastsac_q_action_center.to(q_action)
        scale = self._fastsac_q_action_scale.to(q_action)
        physical = (q_action / float(self.cfg.q_action_input_gain)) * scale + center
        return self._project_execution_action(physical)

    def _project_execution_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the one authoritative finite physical-command support."""
        return _project_to_execution_support(
            action,
            self._fastsac_action_low,
            self._fastsac_action_high,
            float(self.cfg.action_support_clip),
        )

    def _q_execution_bounds(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low = self._q_action_input(self._fastsac_action_low.to(reference))
        high = self._q_action_input(self._fastsac_action_high.to(reference))
        return torch.minimum(low, high), torch.maximum(low, high)

    def _bounded_actor_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        """Map a PPO-compatible physical proposal smoothly into finite support."""
        center = self._fastsac_actor_action_center.to(raw_mean)
        scale = self._fastsac_actor_action_scale.to(raw_mean)
        bounded = center + scale * torch.tanh((raw_mean - center) / scale)
        return self._project_execution_action(bounded)

    def _student_collection_actor_cache_enabled(self) -> bool:
        """Whether replay consumes collection-time carried-hidden Student inputs.

        Baseline TD3 keeps its historical raw-window/current-EMA contract.  The
        FastSAC backend overrides this seam for its locked
        ``online_student_rollout`` perception mode.
        """
        return False

    def _teacher_episode_cache_enabled(self) -> bool:
        """Use the full-episode Teacher cache only with collection-exact replay."""
        return self._student_collection_actor_cache_enabled()

    def _ensure_teacher_episode_cache_state(self) -> None:
        """Lazily install sidecar state for focused/unit construction seams."""
        if not hasattr(self, "_teacher_episode_store"):
            self._teacher_episode_store = TeacherEpisodeSequenceStore(
                is_init_key=PERCEPTION_IS_INIT_KEY
            )
        if not hasattr(self, "_teacher_actor_cache"):
            self._teacher_actor_cache = CurrentEMATeacherActorCache(
                self._q_actor_dim
            )
        if not hasattr(self, "_perception_ema_generation"):
            self._perception_ema_generation = 0
        if not hasattr(self, "_teacher_ring_cache_lineage"):
            self._teacher_ring_cache_lineage = None
        if not hasattr(self, "_teacher_episode_device_raw_fields"):
            self._teacher_episode_device_raw_fields = None
        if not hasattr(self, "_teacher_episode_device_raw_lineage"):
            self._teacher_episode_device_raw_lineage = None
        if not hasattr(self, "_teacher_prefill_raw_pending"):
            self._teacher_prefill_raw_pending = None

    def _reset_teacher_episode_cache_state(self) -> None:
        """Rebuild non-serialized geometry-ID sidecars from an empty lineage.

        Checkpoints deliberately contain neither replay rows nor the raw episode
        journal/current-EMA cache that interprets those rows.  A same-instance
        model restore must therefore discard every sidecar together, otherwise
        a stale episode UID, drift-probe geometry ID, or EMA lineage can alias
        rows collected after load.
        """
        self._teacher_prefill_raw_pending = None
        self._teacher_episode_store = TeacherEpisodeSequenceStore(
            is_init_key=PERCEPTION_IS_INIT_KEY
        )
        self._teacher_actor_cache = CurrentEMATeacherActorCache(self._q_actor_dim)
        self._perception_ema_generation = 0
        self._teacher_ring_cache_lineage = None
        self._teacher_episode_device_raw_fields = None
        self._teacher_episode_device_raw_lineage = None
        self._student_perception_drift_pending = None
        self._student_perception_drift_episodes = []
        self._student_perception_drift_completed = 0
        self._student_perception_drift_discarded_incomplete = 0
        self._reset_replay_object_geo_codebook()

    def _teacher_actor_cache_lineage(self) -> TeacherActorCacheLineage:
        self._ensure_teacher_episode_cache_state()
        store = self._teacher_episode_store
        if not store.frozen:
            raise RuntimeError("Teacher episode journal must be frozen before caching")
        vecnorm_fingerprint = getattr(self, "_replay_vecnorm_fingerprint", None)
        object_geo_fingerprint = getattr(
            self, "_replay_object_geo_fingerprint", None
        )
        if not isinstance(vecnorm_fingerprint, str) or not vecnorm_fingerprint:
            raise RuntimeError("Teacher Actor cache requires a VecNorm fingerprint")
        if not isinstance(object_geo_fingerprint, str) or not object_geo_fingerprint:
            raise RuntimeError("Teacher Actor cache requires object-geometry lineage")
        raw_lineage = store.lineage
        return TeacherActorCacheLineage(
            raw_store_id=raw_lineage.store_id,
            raw_generation=raw_lineage.generation,
            ema_generation=int(self._perception_ema_generation),
            vecnorm_fingerprint=vecnorm_fingerprint,
            object_geo_fingerprint=object_geo_fingerprint,
            encoder_semantics=TEACHER_ACTOR_CACHE_ENCODER_SEMANTICS,
        )

    def _mark_perception_ema_updated(self) -> None:
        """Invalidate only the recomputable Teacher cache after an EMA update."""
        if not self._teacher_episode_cache_enabled():
            return
        self._ensure_teacher_episode_cache_state()
        self._perception_ema_generation += 1
        self._teacher_actor_cache.invalidate()
        self._teacher_ring_cache_lineage = None

    def _student_replay_ema_ages(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Return Student collection-cache ages without reducing diagnostics.

        In collection-exact FastSAC replay, a Student Actor input is exact for
        the EMA generation at collection.  The generation gap to the current
        EMA bounds how far its representation *may* have drifted.  It is not a
        feature-space distance, and therefore does not justify a zero-hidden
        reconstruction.  Teacher rows are intentionally omitted because their
        Actor cache is rebuilt from a complete episode at the current EMA.

        Future-generation validation deliberately remains on every sampled
        batch and before its learning update.  The diagnostic reductions are
        deferred so they do not introduce several device synchronizations per
        Q/Actor sample.
        """
        generation = batch.get(REPLAY_PERCEPTION_EMA_GENERATION_KEY, None)
        is_teacher = batch.get(REPLAY_SAMPLE_IS_TEACHER_KEY, None)
        is_student = batch.get(DAGGER_IS_STUDENT_ACTION_KEY, None)
        source = is_teacher if is_teacher is not None else is_student
        if generation is None or source is None:
            # Compatibility for a deliberately fresh-only replay or focused
            # unit-test batches which predate the provenance field.
            return None
        if not torch.is_tensor(generation) or not torch.is_tensor(source):
            raise TypeError("replay EMA provenance fields must be tensors")
        generation = generation.reshape(-1).long()
        student = (~source if is_teacher is not None else source).reshape(-1).bool()
        if generation.shape != student.shape:
            raise ValueError("replay EMA provenance fields are misaligned")
        generation = generation[student]
        if generation.numel() == 0:
            return None
        current = int(getattr(self, "_perception_ema_generation", 0))
        age = current - generation
        if bool((age < 0).any()):
            raise RuntimeError("Student replay row has a future EMA generation")
        return age.detach().float()

    def _aggregate_student_replay_ema_ages(
        self, age_batches: list[torch.Tensor | None]
    ) -> torch.Tensor:
        """Reproduce the historical mean-of-batch age metrics on-device.

        Missing provenance and teacher-only batches remain zero-valued samples
        in the rollout average.  A padded matrix permits one quantile reduction
        for all non-empty samples, including unusually different Student row
        counts.  The caller performs the sole host extraction for both Q and
        Actor diagnostics after their respective tensors are complete.
        """
        first_age = next((age for age in age_batches if age is not None), None)
        device = (
            first_age.device
            if first_age is not None
            else getattr(self, "device", torch.device("cpu"))
        )
        result = torch.zeros(
            len(_STUDENT_REPLAY_EMA_AGE_METRIC_KEYS),
            dtype=torch.float32,
            device=device,
        )
        if not age_batches or first_age is None:
            return result

        available_ages = [age for age in age_batches if age is not None]
        if any(age.device != first_age.device for age in available_ages):
            raise ValueError("replay EMA age samples must share a device")
        row_counts = torch.as_tensor(
            [age.numel() for age in available_ages],
            dtype=torch.float32,
            device=first_age.device,
        )
        max_rows = max(age.numel() for age in available_ages)
        padded = first_age.new_full(
            (len(available_ages), max_rows), float("nan")
        )
        for row, age in enumerate(available_ages):
            padded[row, : age.numel()] = age.reshape(-1)

        denominator = float(len(age_batches))
        per_batch_mean = torch.nanmean(padded, dim=1)
        per_batch_p95 = torch.nanquantile(padded, 0.95, dim=1)
        per_batch_max = padded.nan_to_num(nan=-torch.inf).max(dim=1).values
        per_batch_stale_fraction = (padded > 0).sum(dim=1) / row_counts
        return torch.stack(
            (
                padded.new_tensor(len(available_ages) / denominator),
                row_counts.sum() / denominator,
                per_batch_mean.sum() / denominator,
                per_batch_p95.sum() / denominator,
                per_batch_max.sum() / denominator,
                per_batch_stale_fraction.sum() / denominator,
            )
        )

    def _student_replay_ema_age_metrics(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, float]:
        """Compatibility wrapper for one-batch diagnostic callers/tests."""
        metrics = self._aggregate_student_replay_ema_ages(
            [self._student_replay_ema_ages(batch)]
        )
        values = metrics.detach().cpu().tolist()
        return dict(zip(_STUDENT_REPLAY_EMA_AGE_METRIC_KEYS, values))

    @staticmethod
    def _empty_student_perception_drift_metrics() -> dict[str, float]:
        return {
            "enabled": 0.0,
            "available": 0.0,
            "episodes": 0.0,
            "rows": 0.0,
            "ema_age_mean": 0.0,
            "ema_age_p95": 0.0,
            "ema_age_max": 0.0,
            "actor_feature_mse": 0.0,
            "actor_feature_relative_rmse": 0.0,
            "actor_feature_cosine_distance": 0.0,
            "perception_latent_mse": 0.0,
            "perception_latent_relative_rmse": 0.0,
            "perception_latent_cosine_distance": 0.0,
            "action_normalized_rmse": 0.0,
        }

    def _student_perception_drift_probe_enabled(self) -> bool:
        return bool(
            self._student_collection_actor_cache_enabled()
            and int(getattr(self.cfg, "perception_staleness_probe_num_envs", 0)) > 0
        )

    @torch.no_grad()
    def _student_perception_drift_raw_values(
        self, td: TensorDict
    ) -> dict[str, torch.Tensor]:
        return {
            DEPTH_KEY: self._replay_source(td, DEPTH_KEY).detach(),
            PERCEPTION_POLICY_RAW_KEY: self._replay_source(td, OBS_KEY).detach(),
            PERCEPTION_VEL_COMMAND_RAW_KEY: self._replay_source(
                td, VEL_CMD_KEY
            ).detach(),
            PERCEPTION_OBJECT_GEO_ID_KEY: self._encode_replay_object_geo(td),
            PERCEPTION_IS_INIT_KEY: td["is_init"].detach().bool(),
        }

    @torch.no_grad()
    def _capture_student_perception_drift_rollout(self, rollout: TensorDict) -> None:
        """Journal a few complete raw episodes without affecting replay.

        The journal starts only at a real ``is_init`` row and keeps every raw
        state needed to reconstruct recurrent context.  It is intentionally
        small, CPU-resident, fresh-only, and diagnostic-only.
        """
        if not self._student_perception_drift_probe_enabled():
            return
        if len(rollout.batch_size) != 2:
            raise ValueError("Student drift probe requires an [env,time] rollout")
        num_envs, num_steps = (int(value) for value in rollout.batch_size)
        probe_envs = min(
            num_envs,
            int(self.cfg.perception_staleness_probe_num_envs),
        )
        actor = rollout.get(STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY, None)
        if actor is None or actor.shape != (num_envs, num_steps, self._q_actor_dim):
            raise RuntimeError(
                "Student drift probe requires the collection-exact Actor cache"
            )
        student = rollout.get(DAGGER_IS_STUDENT_ACTION_KEY, None)
        if student is None:
            raise RuntimeError("Student drift probe requires rollout source labels")

        raw = self._student_perception_drift_raw_values(rollout)
        raw_cpu = {
            key: value[:probe_envs].detach().to(device="cpu").contiguous()
            for key, value in raw.items()
        }
        actor_cpu = actor[:probe_envs].detach().to(device="cpu").contiguous()
        done_cpu = (
            rollout[DONE_KEY][:probe_envs]
            .reshape(probe_envs, num_steps, -1)
            .bool()
            .any(dim=-1)
            .cpu()
        )
        init_cpu = (
            rollout["is_init"][:probe_envs]
            .reshape(probe_envs, num_steps, -1)
            .bool()
            .any(dim=-1)
            .cpu()
        )
        student_cpu = (
            student[:probe_envs]
            .reshape(probe_envs, num_steps, -1)
            .bool()
            .any(dim=-1)
            .cpu()
        )

        pending = self._student_perception_drift_pending
        if pending is None or len(pending) != probe_envs:
            pending = [None for _ in range(probe_envs)]
        generation = int(getattr(self, "_perception_ema_generation", 0))

        def _new_pending() -> dict:
            return {
                "raw": {key: [] for key in _STUDENT_DRIFT_RAW_FIELDS},
                "actor": [],
                "generation": [],
                "eligible": [],
            }

        for env_index in range(probe_envs):
            for step in range(num_steps):
                if bool(init_cpu[env_index, step]):
                    if pending[env_index] is not None:
                        self._student_perception_drift_discarded_incomplete += 1
                    pending[env_index] = _new_pending()
                entry = pending[env_index]
                if entry is None:
                    continue
                for key in _STUDENT_DRIFT_RAW_FIELDS:
                    entry["raw"][key].append(raw_cpu[key][env_index, step].clone())
                entry["actor"].append(actor_cpu[env_index, step].clone())
                entry["generation"].append(generation)
                entry["eligible"].append(
                    bool(student_cpu[env_index, step])
                    and not bool(init_cpu[env_index, step])
                )
                if not bool(done_cpu[env_index, step]):
                    continue

                raw_episode = {
                    key: torch.stack(values, dim=0).contiguous()
                    for key, values in entry["raw"].items()
                }
                length = int(next(iter(raw_episode.values())).shape[0])
                starts_at_reset = bool(
                    raw_episode[PERCEPTION_IS_INIT_KEY]
                    .reshape(length, -1)
                    .bool()
                    .any(dim=-1)[0]
                )
                if not starts_at_reset:
                    raise RuntimeError("Student drift episode does not start at reset")
                self._student_perception_drift_episodes.append(
                    _StudentPerceptionDriftEpisode(
                        raw_fields=raw_episode,
                        collection_actor=torch.stack(
                            entry["actor"], dim=0
                        ).float().contiguous(),
                        collection_ema_generation=torch.tensor(
                            entry["generation"], dtype=torch.long
                        ),
                        eligible_student_rows=torch.tensor(
                            entry["eligible"], dtype=torch.bool
                        ),
                    )
                )
                maximum = int(self.cfg.perception_staleness_probe_max_episodes)
                self._student_perception_drift_episodes = (
                    self._student_perception_drift_episodes[-maximum:]
                )
                self._student_perception_drift_completed += 1
                pending[env_index] = None
        self._student_perception_drift_pending = pending

    @torch.no_grad()
    def _reencode_student_perception_drift_episode(
        self, episode: _StudentPerceptionDriftEpisode
    ) -> torch.Tensor:
        """Re-encode one complete episode from its true reset at current EMA."""
        length = int(episode.collection_actor.shape[0])
        if length < 1 or episode.collection_actor.shape != (
            length,
            self._q_actor_dim,
        ):
            raise ValueError("Student drift episode Actor cache is misaligned")
        if any(int(value.shape[0]) != length for value in episode.raw_fields.values()):
            raise ValueError("Student drift episode raw fields are misaligned")

        snapshot = self._vecnorm_snapshot()
        depth_state = torch.zeros(1, self.depth_feature_dim, device=self.device)
        adapt_state = torch.zeros(1, int(self.cfg.latent_dim), device=self.device)
        encoded: list[torch.Tensor] = []
        chunk_size = max(1, int(self.cfg.train_every))

        with set_recurrent_mode(True):
            for start in range(0, length, chunk_size):
                stop = min(start + chunk_size, length)
                sequence_length = stop - start
                depth_raw = episode.raw_fields[DEPTH_KEY][start:stop].to(
                    self.device
                ).unsqueeze(0)
                policy_raw = episode.raw_fields[PERCEPTION_POLICY_RAW_KEY][
                    start:stop
                ].to(self.device).unsqueeze(0)
                vel_raw = episode.raw_fields[PERCEPTION_VEL_COMMAND_RAW_KEY][
                    start:stop
                ].to(self.device).unsqueeze(0)
                geometry = self._decode_replay_object_geo(
                    episode.raw_fields[PERCEPTION_OBJECT_GEO_ID_KEY][start:stop],
                    device=self.device,
                    dtype=policy_raw.dtype,
                ).unsqueeze(0)
                depth = self._normalize_replay_value(
                    DEPTH_KEY, depth_raw, snapshot
                )
                policy = self._normalize_replay_value(OBS_KEY, policy_raw, snapshot)
                vel = self._normalize_replay_value(VEL_CMD_KEY, vel_raw, snapshot)
                td = TensorDict(
                    {
                        DEPTH_KEY: depth,
                        OBS_KEY: policy,
                        VEL_CMD_KEY: vel,
                        OBJECT_GEO_KEY: geometry.to(dtype=policy.dtype),
                        "is_init": episode.raw_fields[PERCEPTION_IS_INIT_KEY][
                            start:stop
                        ].to(self.device).unsqueeze(0),
                        "depth_hx": depth_state.unsqueeze(1).expand(
                            1, sequence_length, -1
                        ),
                        "adapt_hx": adapt_state.unsqueeze(1).expand(
                            1, sequence_length, -1
                        ),
                    },
                    batch_size=(1, sequence_length),
                    device=self.device,
                )
                if hasattr(self, "temporal_depth_gru_ema"):
                    self.temporal_depth_gru_ema(td)
                    depth_state = td["next", "depth_hx"][:, -1]
                else:
                    td["_depth_feature"] = torch.zeros(
                        1,
                        sequence_length,
                        self.depth_feature_dim,
                        device=self.device,
                        dtype=policy.dtype,
                    )
                if bool(self.cfg.use_object_adapt):
                    self.object_adapt_ema(td)
                    self.object_pred_transform(td)
                self.adapt_ema(td)
                adapt_state = td["next", "adapt_hx"][:, -1]
                actor = torch.cat([td[key] for key in self.q_actor_keys], dim=-1)
                if actor.shape != (1, sequence_length, self._q_actor_dim):
                    raise RuntimeError("Student drift re-encode has invalid shape")
                encoded.append(actor.squeeze(0).float())
        result = torch.cat(encoded, dim=0).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Student drift re-encode contains NaN/Inf")
        return result

    @torch.no_grad()
    def _student_perception_drift_metrics(self) -> dict[str, float]:
        """Compare collection features with reset-exact current-EMA features."""
        result = self._empty_student_perception_drift_metrics()
        if not self._student_perception_drift_probe_enabled():
            return result
        result["enabled"] = 1.0
        current = int(getattr(self, "_perception_ema_generation", 0))
        interval = int(self.cfg.perception_staleness_probe_interval)
        if current % interval:
            return result
        maximum_age = int(self.cfg.perception_staleness_probe_max_generation_age)
        retained = []
        old_values = []
        new_values = []
        ages = []
        used_episodes = 0
        for episode in self._student_perception_drift_episodes:
            row_age = current - episode.collection_ema_generation
            if bool((row_age < 0).any()):
                raise RuntimeError("Student drift journal has a future EMA generation")
            live = row_age <= maximum_age
            if not bool(live.any()):
                continue
            retained.append(episode)
            selected = live & episode.eligible_student_rows
            if not bool(selected.any()):
                continue
            current_actor = self._reencode_student_perception_drift_episode(episode)
            selected_device = selected.to(self.device)
            old_values.append(
                episode.collection_actor[selected].to(self.device)
            )
            new_values.append(current_actor[selected_device])
            ages.append(row_age[selected].to(self.device))
            used_episodes += 1
        self._student_perception_drift_episodes = retained
        if not old_values:
            return result

        old = torch.cat(old_values, dim=0).float()
        new = torch.cat(new_values, dim=0).float()
        age = torch.cat(ages, dim=0).float()
        difference = new - old
        mse = difference.square().mean()
        reference_rms = old.square().mean().sqrt().clamp_min(1e-8)
        cosine_distance = 1.0 - F.cosine_similarity(old, new, dim=-1).mean()

        latent_start = 0
        latent_slice = None
        for key, width in zip(self.q_actor_keys, self._q_actor_widths):
            if key == PRIV_PRED_KEY:
                latent_slice = slice(latent_start, latent_start + int(width))
                break
            latent_start += int(width)
        if latent_slice is None:
            raise RuntimeError("Student drift probe cannot locate perception latent")
        old_latent = old[:, latent_slice]
        new_latent = new[:, latent_slice]
        latent_difference = new_latent - old_latent
        latent_mse = latent_difference.square().mean()
        latent_reference_rms = old_latent.square().mean().sqrt().clamp_min(1e-8)
        latent_cosine_distance = 1.0 - F.cosine_similarity(
            old_latent, new_latent, dim=-1
        ).mean()

        old_action = self._actor_dist_from_flat(old).mean
        new_action = self._actor_dist_from_flat(new).mean
        action_difference = new_action - old_action
        action_scale = getattr(self, "_fastsac_q_action_scale", None)
        if torch.is_tensor(action_scale):
            action_difference = action_difference / action_scale.to(action_difference)

        result.update(
            {
                "available": 1.0,
                "episodes": float(used_episodes),
                "rows": float(age.numel()),
                "ema_age_mean": age.mean().item(),
                "ema_age_p95": torch.quantile(age, 0.95).item(),
                "ema_age_max": age.max().item(),
                "actor_feature_mse": mse.item(),
                "actor_feature_relative_rmse": (mse.sqrt() / reference_rms).item(),
                "actor_feature_cosine_distance": cosine_distance.item(),
                "perception_latent_mse": latent_mse.item(),
                "perception_latent_relative_rmse": (
                    latent_mse.sqrt() / latent_reference_rms
                ).item(),
                "perception_latent_cosine_distance": (
                    latent_cosine_distance.item()
                ),
                "action_normalized_rmse": action_difference.square()
                .mean()
                .sqrt()
                .item(),
            }
        )
        return result

    @torch.no_grad()
    def _collection_actor_observations(self, td: TensorDict) -> torch.Tensor:
        """Flatten the exact normalized Actor inputs already present in ``td``."""
        chunks: list[torch.Tensor] = []
        for key, width in zip(self.q_actor_keys, self._q_actor_widths):
            if key not in td.keys(True, True):
                raise KeyError(
                    f"Student collection Actor cache is missing input {key!r}"
                )
            value = td[key]
            if int(value.shape[-1]) != int(width):
                raise ValueError(
                    f"Student collection Actor input {key!r} has width "
                    f"{int(value.shape[-1])}; expected {int(width)}"
                )
            if tuple(value.shape[:-1]) != tuple(td.batch_size):
                raise ValueError(
                    f"Student collection Actor input {key!r} is batch-misaligned"
                )
            chunks.append(value)
        observations = torch.cat(chunks, dim=-1)
        if observations.shape != (*td.batch_size, self._q_actor_dim):
            raise RuntimeError("Student collection Actor cache has an invalid shape")
        if not torch.isfinite(observations).all():
            raise RuntimeError("Student collection Actor cache contains NaN/Inf")
        # Collection runs inside ``torch.inference_mode()``.  An inference
        # tensor cannot subsequently be saved for backward by the Actor even
        # though the cached observations themselves never require gradients.
        # Clone with inference mode explicitly disabled so replay always owns
        # a regular detached tensor.
        with torch.inference_mode(False):
            return observations.detach().clone()

    def _actor_dist_from_flat_module(self, module: nn.Module, actor_obs: torch.Tensor):
        vel_dim = int(self.observation_spec[VEL_CMD_KEY].shape[-1])
        policy_dim = int(self.observation_spec[OBS_KEY].shape[-1])
        td = TensorDict(
            {
                VEL_CMD_KEY: actor_obs[..., :vel_dim],
                OBS_KEY: actor_obs[..., vel_dim : vel_dim + policy_dim],
                PRIV_PRED_KEY: actor_obs[..., vel_dim + policy_dim :],
            },
            batch_size=actor_obs.shape[:-1],
            device=actor_obs.device,
        )
        return module.get_dist(td)

    @torch.no_grad()
    def _student_raw_action_proposal(self, td: TensorDict) -> torch.Tensor:
        """Return the unchanged PPOVEL physical-command head proposal."""
        return PPOBCDaggerFinetune._student_latent(self, td)

    @torch.no_grad()
    def _student_mean_action(self, td: TensorDict) -> torch.Tensor:
        """Return the deterministic Student action inside execution support."""
        raw_mean = self._student_raw_action_proposal(td)
        if not torch.isfinite(raw_mean).all():
            raise RuntimeError("TD3 evaluation Actor produced non-finite raw actions")
        return self._bounded_actor_mean(raw_mean)

    def _actor_dist_from_flat(self, actor_obs: torch.Tensor):
        return self._actor_dist_from_flat_module(self.actor_adapt, actor_obs)

    def _actor_target_dist_from_flat(self, actor_obs: torch.Tensor):
        if self.actor_target is None:
            raise RuntimeError("actor_target is initialized only after source loading")
        return self._actor_dist_from_flat_module(self.actor_target, actor_obs)

    def _teacher_prefill_active(self) -> bool:
        """Whether successful-only Teacher collection still owns behavior."""
        return not bool(getattr(self, "_teacher_prefill_complete", False))

    def is_teacher_prefill_active(self) -> bool:
        """Public trainer hook for phase-aware statistics and scheduling."""
        return self._teacher_prefill_active()

    def _collect_teacher_q_replay_this_rollout(self) -> bool:
        """The Teacher Q partition is collected only during dynamic prefill."""
        return self._teacher_prefill_active()

    def _teacher_q_replay_frozen(self) -> bool:
        """Whether the capacity-filled Teacher partition is now immutable."""
        return not self._teacher_prefill_active()

    def _all_ranks_teacher_replay_full(self) -> bool:
        """Synchronize the prefill→main boundary across distributed ranks."""
        local_full = self.q_teacher_replay.size >= self.q_teacher_replay.capacity
        if not (dist.is_available() and dist.is_initialized()):
            return local_full
        flag = torch.tensor(
            int(local_full),
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @staticmethod
    def _prefill_pending_row_count(
        chunks: list[dict[str, torch.Tensor]],
    ) -> int:
        if not chunks:
            return 0
        return sum(int(chunk["actions"].shape[0]) for chunk in chunks)

    def _ensure_teacher_prefill_pending(self, num_envs: int) -> None:
        num_envs = int(num_envs)
        if num_envs < 1:
            raise ValueError("Teacher prefill requires at least one environment")
        pending = self._teacher_prefill_pending
        if pending is None:
            self._teacher_prefill_pending = [[] for _ in range(num_envs)]
        elif len(pending) < num_envs:
            pending.extend([] for _ in range(num_envs - len(pending)))
        if self._teacher_episode_cache_enabled():
            self._ensure_teacher_episode_cache_state()
            raw_pending = self._teacher_prefill_raw_pending
            if raw_pending is None:
                self._teacher_prefill_raw_pending = [
                    [] for _ in range(num_envs)
                ]
            elif len(raw_pending) < num_envs:
                raw_pending.extend(
                    [] for _ in range(num_envs - len(raw_pending))
                )

    @torch.no_grad()
    def _post_process_teacher_prefill_episode(
        self, episode: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Optionally augment a committed Teacher episode before ring storage.

        Subclasses that precompute per-transition quantities (e.g. frozen
        Teacher-value cache fields) override this method to add those fields
        to ``episode`` before ``q_teacher_replay.extend`` allocates storage.
        The default is a no-op that returns ``episode`` unchanged.
        """
        return episode

    def _q_replay_prefill_storage_fields(self) -> tuple[str, ...]:
        """Fields sourced from rollout transitions during Teacher prefill.

        Subclasses that add post-computed cache fields to ``_q_replay_storage_fields``
        should exclude those fields here so ``_stage_teacher_prefill_rows`` does not
        try to read them from the in-flight rollout transitions dict.
        """
        return self._q_replay_storage_fields()

    def _stage_teacher_prefill_rows(
        self,
        transitions: dict[str, torch.Tensor],
        rollout: TensorDict | None = None,
    ) -> tuple[int, int]:
        """Commit only completed, non-failed Teacher episodes to replay.

        Rows are staged separately for each vector environment across rollout
        boundaries.  ``command_finished`` without a physical termination is a
        successful trajectory and commits every valid Teacher-executed row in
        that pending episode.  Physical termination, a pure time limit, or an
        unexpected reset discards the entire pending episode.  Invalid Teacher
        rows may execute the deterministic Student as an environment fallback,
        but are never staged as Teacher replay data.
        """
        collection_exact = self._teacher_episode_cache_enabled()
        if collection_exact and rollout is None:
            raise ValueError(
                "collection-exact Teacher prefill requires the full [env,time] rollout"
            )
        replay_fields = self._q_replay_prefill_storage_fields()
        required = set(replay_fields).union(
            _PREFILL_INTERNAL_FIELDS,
            {
                DAGGER_TEACHER_ACTION_VALID_KEY,
                DAGGER_IS_STUDENT_ACTION_KEY,
                "dones",
            },
        )
        missing = required.difference(transitions)
        if missing:
            raise KeyError(
                f"Teacher prefill transitions are missing: {sorted(missing)}"
            )
        row_count = int(transitions["actions"].shape[0])
        if row_count == 0 and rollout is None:
            return 0, 0
        if any(int(transitions[key].shape[0]) != row_count for key in required):
            raise ValueError("Teacher prefill transition fields are misaligned")

        payload_metadata = torch.stack(
            (
                transitions[_PREFILL_ENV_INDEX_KEY].reshape(row_count).long(),
                transitions[_PREFILL_STEP_INDEX_KEY].reshape(row_count).long(),
                (
                    transitions[DAGGER_TEACHER_ACTION_VALID_KEY]
                    .reshape(row_count)
                    .bool()
                    & ~transitions[DAGGER_IS_STUDENT_ACTION_KEY]
                    .reshape(row_count)
                    .bool()
                ).long(),
            ),
            dim=-1,
        ).detach()
        if payload_metadata.device.type != "cpu":
            payload_metadata = payload_metadata.to("cpu")
        payload_env = payload_metadata[:, 0]
        payload_step = payload_metadata[:, 1]
        teacher_executed = payload_metadata[:, 2].bool()

        if rollout is None:
            # Pure unit seams may supply an already unfiltered transition grid.
            # Production always passes the original [env,time] rollout so a
            # terminal/reset row hidden by replay filtering cannot be missed.
            event_env = payload_env
            event_step = payload_step
            done = transitions["dones"].reshape(row_count).detach().bool().cpu()
            terminated = (
                transitions[_PREFILL_TERMINATED_KEY]
                .reshape(row_count)
                .detach()
                .bool()
                .cpu()
            )
            command_finished = (
                transitions[_PREFILL_COMMAND_FINISHED_KEY]
                .reshape(row_count)
                .detach()
                .bool()
                .cpu()
            )
            time_limit = transitions.get(REPLAY_TIME_LIMIT_KEY, None)
            time_limit = (
                torch.zeros(row_count, dtype=torch.bool)
                if time_limit is None
                else time_limit.reshape(row_count).detach().bool().cpu()
            )
            is_init = (
                transitions[PERCEPTION_IS_INIT_KEY][:, -2]
                .reshape(row_count, -1)
                .detach()
                .bool()
                .any(dim=-1)
                .cpu()
            )
        else:
            if len(rollout.batch_size) != 2:
                raise ValueError("Teacher prefill rollout must have [env,time] shape")
            num_envs, num_steps = (int(value) for value in rollout.batch_size)

            def _event_grid(value: torch.Tensor) -> torch.Tensor:
                return value.reshape(num_envs, num_steps, -1).bool().any(dim=-1)

            event_env = (
                torch.arange(num_envs, device=rollout.device)
                .unsqueeze(1)
                .expand(num_envs, num_steps)
            )
            event_step = (
                torch.arange(num_steps, device=rollout.device)
                .unsqueeze(0)
                .expand(num_envs, num_steps)
            )
            rollout_is_init = rollout.get("is_init", None)
            is_init_grid = (
                torch.zeros(
                    num_envs,
                    num_steps,
                    dtype=torch.bool,
                    device=rollout.device,
                )
                if rollout_is_init is None
                else _event_grid(rollout_is_init)
            )
            event_metadata = torch.stack(
                (
                    event_env,
                    event_step,
                    _event_grid(rollout[DONE_KEY]).long(),
                    _event_grid(rollout[TERM_KEY]).long(),
                    _event_grid(rollout["next", "stats", "command_finished"]).long(),
                    _event_grid(
                        rollout["next", "stats", "episode_time_limit"]
                        if (
                            "next",
                            "stats",
                            "episode_time_limit",
                        )
                        in rollout.keys(True, True)
                        else torch.zeros_like(rollout[DONE_KEY])
                    ).long(),
                    is_init_grid.long(),
                ),
                dim=-1,
            ).reshape(-1, 7)
            if event_metadata.device.type != "cpu":
                event_metadata = event_metadata.to("cpu")
            event_env = event_metadata[:, 0]
            event_step = event_metadata[:, 1]
            done = event_metadata[:, 2].bool()
            terminated = event_metadata[:, 3].bool()
            command_finished = event_metadata[:, 4].bool()
            time_limit = event_metadata[:, 5].bool()
            is_init = event_metadata[:, 6].bool()

        raw_rollout = None
        if collection_exact:
            if rollout is None:  # pragma: no cover - guarded above
                raise RuntimeError("collection-exact prefill lost its rollout")
            raw_rollout = self._raw_perception_values(rollout)
            if any(
                tuple(value.shape[:2]) != (num_envs, num_steps)
                for value in raw_rollout.values()
            ):
                raise ValueError("Teacher raw episode journal is rollout-misaligned")
            # One coalesced device-to-host transfer per field.  Copying each
            # environment slice independently would issue up to 4 * num_envs
            # tiny transfers every prefill rollout.
            raw_rollout = {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in raw_rollout.items()
            }

        maximum_env = int(event_env.max().item())
        if int(event_env.min().item()) < 0:
            raise ValueError("Teacher prefill environment index is negative")
        self._ensure_teacher_prefill_pending(maximum_env + 1)
        pending = self._teacher_prefill_pending
        if pending is None:  # pragma: no cover - guarded by initializer above
            raise RuntimeError("Teacher prefill pending storage was not initialized")
        raw_pending = (
            self._teacher_prefill_raw_pending if collection_exact else None
        )
        if collection_exact and raw_pending is None:
            raise RuntimeError("Teacher raw episode pending storage was not initialized")

        replay_device = self.q_teacher_replay.device
        payload = {
            key: transitions[key].detach().to(replay_device) for key in replay_fields
        }
        committed_rows = 0
        discarded_rows = 0

        def _append_segment(env_index: int, start_step: int, stop_step: int) -> None:
            raw_offset = 0
            if collection_exact:
                if raw_pending is None or raw_rollout is None:  # pragma: no cover
                    raise RuntimeError("Teacher raw episode journal is unavailable")
                raw_offset = sum(
                    int(chunk[PERCEPTION_IS_INIT_KEY].shape[0])
                    for chunk in raw_pending[env_index]
                )
                raw_pending[env_index].append(
                    {
                        key: value[env_index, start_step:stop_step]
                        .detach()
                        .contiguous()
                        .clone()
                        for key, value in raw_rollout.items()
                    }
                )
            selected = (
                (
                    (payload_env == env_index)
                    & (payload_step >= start_step)
                    & (payload_step < stop_step)
                    & teacher_executed
                )
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            if selected.numel() == 0:
                return
            selected = selected.to(replay_device)
            chunk = {
                key: value.index_select(0, selected) for key, value in payload.items()
            }
            if collection_exact:
                selected_cpu = selected.to(device="cpu")
                chunk[_TEACHER_PENDING_EPISODE_STEP_KEY] = (
                    payload_step.index_select(0, selected_cpu)
                    - int(start_step)
                    + int(raw_offset)
                ).to(replay_device)
            pending[env_index].append(chunk)

        def _discard(env_index: int, kind: str) -> int:
            rows = self._prefill_pending_row_count(pending[env_index])
            pending[env_index].clear()
            if raw_pending is not None:
                raw_pending[env_index].clear()
            if kind == "failed":
                self._teacher_prefill_failed_episodes += 1
            elif kind == "timeout":
                self._teacher_prefill_timeout_episodes += 1
            elif kind == "incomplete":
                self._teacher_prefill_incomplete_episodes += 1
            else:  # pragma: no cover - internal invariant
                raise RuntimeError(f"unknown Teacher prefill discard kind={kind!r}")
            self._teacher_prefill_discarded_rows += rows
            return rows

        def _commit(env_index: int) -> int:
            chunks = pending[env_index]
            raw_chunks = None if raw_pending is None else raw_pending[env_index]
            if collection_exact:
                if not raw_chunks:
                    raise RuntimeError(
                        "successful Teacher boundary lacks a raw episode journal"
                    )
                starts_at_reset = bool(
                    raw_chunks[0][PERCEPTION_IS_INIT_KEY]
                    .reshape(int(raw_chunks[0][PERCEPTION_IS_INIT_KEY].shape[0]), -1)
                    .bool()
                    .any(dim=-1)[0]
                )
                if not starts_at_reset:
                    nonlocal discarded_rows
                    discarded_rows += _discard(env_index, "incomplete")
                    return 0
            if chunks:
                episode = {
                    key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                    for key in replay_fields
                }
                episode = self._post_process_teacher_prefill_episode(episode)
                if collection_exact:
                    if raw_chunks is None:  # pragma: no cover - guarded above
                        raise RuntimeError("Teacher raw episode journal vanished")
                    episode_steps = torch.cat(
                        [
                            chunk[_TEACHER_PENDING_EPISODE_STEP_KEY]
                            for chunk in chunks
                        ],
                        dim=0,
                    ).long()
                    raw_episode = {
                        key: torch.cat(
                            [chunk[key] for chunk in raw_chunks], dim=0
                        )
                        for key in _PERCEPTION_REPLAY_FIELDS
                    }
                    self._ensure_teacher_episode_cache_state()
                    episode_uid = self._teacher_episode_store.allocate_episode_uid()
                    self._teacher_episode_store.commit_successful_episode(
                        episode_uid,
                        raw_episode,
                        boundary_cause=TeacherBoundaryCause.SUCCESS_COMMAND,
                    )
                    episode[TEACHER_EPISODE_UID_KEY] = torch.full_like(
                        episode_steps, episode_uid
                    )
                    episode[TEACHER_EPISODE_STEP_KEY] = episode_steps
                    try:
                        rows = self.q_teacher_replay.extend(episode)
                    except Exception:
                        self._teacher_episode_store.rollback_episode(episode_uid)
                        raise
                else:
                    rows = self.q_teacher_replay.extend(episode)
            else:
                rows = 0
            # Capture before clear(): the emptied list would otherwise read as
            # "no episode" and silence the per-motion counter entirely.
            committed_episode = episode if chunks else None
            chunks.clear()
            if raw_chunks is not None:
                raw_chunks.clear()
            self._teacher_prefill_successful_episodes += 1
            self._count_prefill_success_motion(committed_episode)
            return rows

        for env_index in event_env.unique(sorted=True).tolist():
            event_rows = (
                (event_env == int(env_index)).nonzero(as_tuple=False).squeeze(-1)
            )
            segment_start_step = int(event_step[event_rows[0]])
            for row in event_rows.tolist():
                step = int(event_step[row])
                if bool(is_init[row]):
                    if step > segment_start_step:
                        _append_segment(int(env_index), segment_start_step, step)
                    if pending[int(env_index)] or bool(
                        raw_pending is not None and raw_pending[int(env_index)]
                    ):
                        discarded_rows += _discard(int(env_index), "incomplete")
                    segment_start_step = step
                if not bool(done[row]):
                    continue
                _append_segment(int(env_index), segment_start_step, step + 1)
                if collection_exact:
                    cause = classify_teacher_boundary(
                        done=True,
                        terminated=bool(terminated[row]),
                        command_finished=bool(command_finished[row]),
                        time_limit=bool(time_limit[row]),
                    )
                    if cause is TeacherBoundaryCause.TERMINATED:
                        discarded_rows += _discard(int(env_index), "failed")
                    elif cause is TeacherBoundaryCause.SUCCESS_COMMAND:
                        committed_rows += _commit(int(env_index))
                    elif cause is TeacherBoundaryCause.TIME_LIMIT:
                        discarded_rows += _discard(int(env_index), "timeout")
                    else:
                        _discard(int(env_index), "incomplete")
                        raise RuntimeError(
                            "Teacher prefill encountered an unclassified done boundary"
                        )
                elif bool(terminated[row]):
                    discarded_rows += _discard(int(env_index), "failed")
                elif bool(command_finished[row]):
                    committed_rows += _commit(int(env_index))
                else:
                    discarded_rows += _discard(int(env_index), "timeout")
                segment_start_step = step + 1
            stop_step = int(event_step[event_rows[-1]]) + 1
            if segment_start_step < stop_step:
                _append_segment(int(env_index), segment_start_step, stop_step)

        return committed_rows, discarded_rows

    @torch.no_grad()
    def _discard_unresolved_teacher_prefill_rows(self) -> int:
        """Discard trajectories whose success is unknown when prefill ends."""
        pending = self._teacher_prefill_pending
        if pending is None:
            return 0
        raw_pending = getattr(self, "_teacher_prefill_raw_pending", None)
        discarded = 0
        for env_index, chunks in enumerate(pending):
            rows = self._prefill_pending_row_count(chunks)
            has_raw = bool(
                raw_pending is not None
                and env_index < len(raw_pending)
                and raw_pending[env_index]
            )
            if rows or has_raw:
                discarded += rows
                self._teacher_prefill_incomplete_episodes += 1
                self._teacher_prefill_discarded_rows += rows
                chunks.clear()
                if raw_pending is not None and env_index < len(raw_pending):
                    raw_pending[env_index].clear()
        return discarded

    def _teacher_prefill_pending_rows(self) -> int:
        pending = self._teacher_prefill_pending or ()
        return sum(self._prefill_pending_row_count(chunks) for chunks in pending)

    @torch.no_grad()
    def _freeze_teacher_episode_replay(self) -> None:
        """Freeze/GC complete raw episodes referenced by the full Teacher FIFO."""
        if not self._teacher_episode_cache_enabled():
            return
        self._ensure_teacher_episode_cache_state()
        if self._teacher_episode_store.frozen:
            return
        if self.q_teacher_replay.size != self.q_teacher_replay.capacity:
            raise RuntimeError(
                "Teacher episode replay can freeze only after its FIFO is full"
            )
        for key in (TEACHER_EPISODE_UID_KEY, TEACHER_EPISODE_STEP_KEY):
            if key not in self.q_teacher_replay.data:
                raise KeyError(f"Teacher FIFO is missing episode lineage {key!r}")
        size = int(self.q_teacher_replay.size)
        self._teacher_episode_store.freeze(
            self.q_teacher_replay.data[TEACHER_EPISODE_UID_KEY][:size],
            self.q_teacher_replay.data[TEACHER_EPISODE_STEP_KEY][:size],
        )
        self._teacher_actor_cache.invalidate()
        self._teacher_ring_cache_lineage = None
        self._teacher_prefill_raw_pending = None

    def _reference_phase(self, td: TensorDict) -> torch.Tensor:
        """Return the raw normalized reference phase used by the simulator."""
        keys = td.keys(True, True)
        if REFERENCE_PHASE_KEY in keys:
            phase = td[REFERENCE_PHASE_KEY]
        elif "ref_motion_phase_" in keys:
            phase = td["ref_motion_phase_"]
        elif CMD_KEY not in keys and not hasattr(self, "observation_spec"):
            # Small unit seams created without an environment predate phase
            # metadata. Production policies always own the raw command group.
            return torch.zeros(td.batch_size, dtype=torch.float32, device=td.device)
        else:
            # ``ref_motion_phase`` is the final scalar in the raw command group.
            phase = self._replay_source(td, CMD_KEY)[..., -1:]
        if not torch.isfinite(phase).all():
            raise ValueError("reference phase contains non-finite values")
        return phase[..., -1].clamp(0.0, 1.0)

    @torch.no_grad()
    def _update_failure_phase_histogram(self, rollout: TensorDict) -> int:
        """Update risk phases from Student physical failures without replaying them.

        Each eligible episode contributes evenly spaced phase anchors from its
        causal ``[done-lookback, done]`` history.  Pure timeouts, motion
        completion, and Teacher-only failures do not contribute. Direct Student
        control on the final or preceding step is causal; an explicit SafeDAgger
        takeover marker remains causal through the later Teacher hold window.
        A physical termination wins when reset causes coincide.
        """
        if len(rollout.batch_size) != 2:
            raise ValueError("failure-phase tracking requires an [env,time] rollout")
        num_envs, num_steps = (int(value) for value in rollout.batch_size)

        def _grid(value: torch.Tensor, *, boolean: bool = False) -> torch.Tensor:
            value = value.reshape(num_envs, num_steps, -1)
            if boolean:
                return value.bool().any(dim=-1)
            if value.shape[-1] != 1:
                raise ValueError("reference phase must have one scalar per step")
            return value[..., 0]

        phase = _grid(self._reference_phase(rollout))
        student = _grid(rollout[DAGGER_IS_STUDENT_ACTION_KEY], boolean=True)
        safe_takeover_value = rollout.get(DAGGER_SAFE_TAKEOVER_KEY, None)
        safe_takeover = (
            torch.zeros_like(student)
            if safe_takeover_value is None
            else _grid(safe_takeover_value, boolean=True)
        )
        done = _grid(rollout[DONE_KEY], boolean=True)
        terminated = _grid(rollout[TERM_KEY], boolean=True)
        command_finished = _grid(
            rollout["next", "stats", "command_finished"], boolean=True
        )
        is_init_value = rollout.get("is_init", None)
        is_init = (
            torch.zeros_like(done)
            if is_init_value is None
            else _grid(is_init_value, boolean=True)
        )
        # One packed device-to-host transfer avoids synchronizing separately
        # for every reset/failure flag on each rollout.
        packed = torch.stack(
            (
                phase.float(),
                student.float(),
                done.float(),
                terminated.float(),
                command_finished.float(),
                is_init.float(),
                safe_takeover.float(),
            ),
            dim=-1,
        ).detach()
        if packed.device.type != "cpu":
            packed = packed.to("cpu")
        phase = packed[..., 0]
        student = packed[..., 1].bool()
        done = packed[..., 2].bool()
        terminated = packed[..., 3].bool()
        command_finished = packed[..., 4].bool()
        is_init = packed[..., 5].bool()
        safe_takeover = packed[..., 6].bool()

        histories = self._failure_phase_history
        if histories is None or len(histories) != num_envs:
            histories = [[] for _ in range(num_envs)]
        student_histories = getattr(self, "_failure_phase_student_history", None)
        if student_histories is None or len(student_histories) != num_envs:
            student_histories = [[] for _ in range(num_envs)]
        takeover_histories = getattr(self, "_failure_phase_takeover_history", None)
        if takeover_histories is None or len(takeover_histories) != num_envs:
            takeover_histories = [[] for _ in range(num_envs)]
        lookback = int(self.cfg.failure_phase_lookback_steps)
        maximum_history = lookback + 1
        requested = int(self.cfg.failure_phase_samples_per_failure)
        bins = int(self.cfg.failure_phase_num_bins)
        anchors_added = 0

        for env_index in range(num_envs):
            history = histories[env_index]
            student_history = student_histories[env_index]
            takeover_history = takeover_histories[env_index]
            for step in range(num_steps):
                if bool(is_init[env_index, step]):
                    history.clear()
                    student_history.clear()
                    takeover_history.clear()
                history.append(float(phase[env_index, step]))
                student_history.append(bool(student[env_index, step]))
                takeover_history.append(bool(safe_takeover[env_index, step]))
                if len(history) > maximum_history:
                    trim = len(history) - maximum_history
                    del history[:trim]
                    del student_history[:trim]
                    del takeover_history[:trim]

                # Direct Student control on the terminal or preceding step is
                # the conservative causal seam. A SafeDAgger takeover marker
                # remains attributable throughout its hysteresis/hold window,
                # without relabeling unrelated Teacher failures merely because
                # the Student acted much earlier in the episode.
                causal_student_control = (
                    bool(student_history[-1])
                    or bool(len(student_history) >= 2 and student_history[-2])
                    or any(takeover_history)
                )

                physical_student_failure = bool(
                    done[env_index, step]
                    and terminated[env_index, step]
                    and causal_student_control
                    and not command_finished[env_index, step]
                )
                if physical_student_failure:
                    count = min(requested, len(history))
                    offsets = _failure_lookback_offsets(
                        len(history) - 1, count
                    ).tolist()
                    anchor_phases = torch.tensor(
                        [history[offset] for offset in offsets], dtype=torch.float64
                    )
                    bin_indices = torch.floor(anchor_phases * bins).long()
                    bin_indices.clamp_(0, bins - 1)
                    self._failure_phase_histogram.index_add_(
                        0,
                        bin_indices,
                        torch.ones_like(anchor_phases),
                    )
                    anchors_added += count
                    self._failure_phase_episode_count += 1
                if bool(done[env_index, step]):
                    history.clear()
                    student_history.clear()
                    takeover_history.clear()
            histories[env_index] = history
            student_histories[env_index] = student_history
            takeover_histories[env_index] = takeover_history

        self._failure_phase_history = histories
        self._failure_phase_student_history = student_histories
        self._failure_phase_takeover_history = takeover_histories
        self._failure_phase_anchor_count += anchors_added
        if anchors_added:
            getattr(self, "_failure_histogram_device_cache", {}).clear()
        return anchors_added

    def _has_canonical_replay_mix(self) -> bool:
        return all(
            hasattr(self.cfg, f"{purpose}_{source}_fraction")
            for purpose in ("q", "actor", "perception")
            for source in REPLAY_SOURCE_ORDER
        )

    def _q_replay_storage_fields(self) -> tuple[str, ...]:
        """Return the replay schema selected by the algorithm contract."""
        fields = (
            (*_Q_REPLAY_FIELDS, *_V4_REPLAY_CONTEXT_FIELDS)
            if self._has_canonical_replay_mix()
            else _Q_REPLAY_FIELDS
        )
        if self._q_conditions_on_actuator_state():
            fields = (
                *fields,
                Q_ACTUATOR_CONTEXT_KEY,
                NEXT_Q_ACTUATOR_CONTEXT_KEY,
            )
        if self._teacher_episode_cache_enabled():
            return tuple(
                key for key in fields if key not in _PERCEPTION_REPLAY_FIELDS
            )
        return fields

    def _replay_mix_fractions(self, purpose: str) -> dict[str, float]:
        """Resolve one purpose's global four-way source fractions.

        Canonical v4 fields are authoritative when present.  The derived branch
        exists only to retain the exact source meaning of older TD3/FastSAC
        configs; it is not used to represent a v4 TVKD experiment.
        """
        if purpose not in ("q", "actor", "perception"):
            raise ValueError(f"unknown replay mix purpose={purpose!r}")
        if self._has_canonical_replay_mix():
            fractions = {
                source: float(getattr(self.cfg, f"{purpose}_{source}_fraction"))
                for source in REPLAY_SOURCE_ORDER
            }
            # Validate without changing the configured values.
            allocate_source_counts(1, fractions)
            return fractions

        teacher_fraction = {
            "q": float(getattr(self.cfg, "q_teacher_replay_ratio", 0.5)),
            "actor": float(getattr(self.cfg, "teacher_actor_replay_fraction", 0.0)),
            "perception": float(
                getattr(self.cfg, "teacher_perception_replay_fraction", 0.0)
            ),
        }[purpose]
        teacher_failure = float(
            getattr(self.cfg, "failure_phase_teacher_fraction", 0.0)
        )
        student_failure = float(
            getattr(self.cfg, "failure_phase_student_fraction", 0.0)
        )
        student_fraction = 1.0 - teacher_fraction
        return {
            "uniform_student": student_fraction * (1.0 - student_failure),
            "failure_student": student_fraction * student_failure,
            "uniform_teacher": teacher_fraction * (1.0 - teacher_failure),
            "failure_teacher": teacher_fraction * teacher_failure,
        }

    def _replay_source_counts(self, purpose: str, batch_size: int) -> dict[str, int]:
        if self._has_canonical_replay_mix():
            return allocate_source_counts(
                int(batch_size), self._replay_mix_fractions(purpose)
            )

        # Preserve legacy nested half-up rounding byte-for-byte.  Largest
        # remainder is mandatory only for canonical v4 source fractions.
        teacher_fraction = {
            "q": float(getattr(self.cfg, "q_teacher_replay_ratio", 0.5)),
            "actor": float(getattr(self.cfg, "teacher_actor_replay_fraction", 0.0)),
            "perception": float(
                getattr(self.cfg, "teacher_perception_replay_fraction", 0.0)
            ),
        }[purpose]
        student_count, uniform_teacher, failure_teacher = _source_counts(
            int(batch_size),
            teacher_fraction,
            float(getattr(self.cfg, "failure_phase_teacher_fraction", 0.0)),
        )
        failure_student = (
            0
            if student_count == 0
            else _source_counts(
                student_count,
                1.0,
                float(getattr(self.cfg, "failure_phase_student_fraction", 0.0)),
            )[2]
        )
        return {
            "uniform_student": student_count - failure_student,
            "failure_student": failure_student,
            "uniform_teacher": uniform_teacher,
            "failure_teacher": failure_teacher,
        }

    def _student_intrinsic_focused_mask(self, indices: torch.Tensor) -> torch.Tensor:
        replay = self.dagger_replay
        key = (
            REPLAY_INTRINSIC_FOCUSED_KEY
            if REPLAY_INTRINSIC_FOCUSED_KEY in replay.data
            else FAILURE_PHASE_STUDENT_SOURCE_KEY
        )
        if key not in replay.data:
            return torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
        replay_indices = indices.to(replay.device)
        result = replay.data[key].index_select(0, replay_indices).reshape(-1).bool()
        return result if result.device == indices.device else result.to(indices.device)

    def _count_prefill_success_motion(self, episode) -> None:
        """Record which motion a committed successful prefill episode used."""
        if episode is None or REPLAY_MOTION_ID_KEY not in episode:
            return
        motion_id = episode[REPLAY_MOTION_ID_KEY]
        if not torch.is_tensor(motion_id) or motion_id.numel() == 0:
            return
        unique_ids = motion_id.reshape(-1).long().unique()
        if unique_ids.numel() != 1:
            raise RuntimeError(
                "a successful Teacher prefill episode spans multiple motions"
            )
        key = int(unique_ids.item())
        counts = self._teacher_prefill_successful_by_motion
        counts[key] = counts.get(key, 0) + 1

    def _prefill_success_motion_metrics(self) -> dict[str, float]:
        """Expose per-motion prefill commits plus their imbalance ratio."""
        counts = getattr(self, "_teacher_prefill_successful_by_motion", None)
        if not counts:
            return {}
        metrics = {
            f"td3/prefill_successful_episodes_motion_{motion}": float(value)
            for motion, value in sorted(counts.items())
        }
        values = [float(value) for value in counts.values()]
        metrics["td3/prefill_motion_imbalance_ratio"] = max(values) / max(
            min(values), 1.0
        )
        return metrics

    def _teacher_phase_match_pool(
        self,
        motion_id: int,
        risk_bin: int,
    ) -> tuple[int, torch.Tensor] | None:
        """Return the nearest same-motion phase rows in ascending replay order."""
        teacher_motion = self._teacher_replay_motion_ids
        same_motion = (
            (teacher_motion == int(motion_id)).nonzero(as_tuple=False).squeeze(-1)
        )
        if same_motion.numel() == 0:
            return None
        motion_bins = self._teacher_replay_phase_bins.index_select(0, same_motion)
        distances = (motion_bins - int(risk_bin)).abs()
        nearest_distance = int(distances.min().item())
        return nearest_distance, same_motion[distances == nearest_distance]

    def _verified_teacher_focus_pool(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unique eligible FT rows and their verified-anchor weights.

        The pool is built on CPU and cached by the complete motion histogram,
        phase-distance contract, and immutable Teacher replay size.  Assigning
        anchor weight to rows only after same-motion nearest-phase matching
        prevents phase aliases between different motions.
        """
        mapping = getattr(self, "_verified_failure_motion_phase_histogram", None)
        if not isinstance(mapping, Mapping) or not mapping:
            empty_rows = torch.empty(0, dtype=torch.long)
            empty_weights = torch.empty(0, dtype=torch.float64)
            signature = ("empty", int(self.q_teacher_replay.size))
            cached = getattr(self, "_verified_teacher_focus_pool_cache", None)
            if (
                isinstance(cached, tuple)
                and len(cached) == 3
                and cached[0] == signature
            ):
                return cached[1], cached[2]
            self._verified_teacher_focus_pool_cache = (
                signature,
                empty_rows,
                empty_weights,
            )
            getattr(self, "_verified_teacher_focus_device_cache", {}).clear()
            return empty_rows, empty_weights
        if not bool(getattr(self, "_teacher_phase_index_ready", False)):
            self._build_teacher_phase_index()

        bin_count = int(self.cfg.failure_phase_num_bins)
        normalized: list[tuple[int, torch.Tensor]] = []
        for raw_motion, raw_histogram in sorted(
            mapping.items(), key=lambda item: int(item[0])
        ):
            if not torch.is_tensor(raw_histogram) or raw_histogram.shape != (
                bin_count,
            ):
                raise ValueError(
                    "verified failure motion histogram has an invalid shape"
                )
            histogram = raw_histogram.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(histogram).all() or bool((histogram < 0).any()):
                raise ValueError("verified failure motion histogram is invalid")
            normalized.append((int(raw_motion), histogram))

        max_distance = getattr(self.cfg, "max_teacher_phase_match_distance", None)
        if max_distance is not None:
            if (
                isinstance(max_distance, bool)
                or not isinstance(max_distance, (int, float))
                or not math.isfinite(float(max_distance))
                or float(max_distance) < 0.0
            ):
                raise ValueError(
                    "max_teacher_phase_match_distance must be null or finite and "
                    "non-negative"
                )
            max_distance = float(max_distance)
        signature = (
            int(self.q_teacher_replay.size),
            bin_count,
            max_distance,
            tuple(
                (motion, tuple(float(value) for value in histogram.tolist()))
                for motion, histogram in normalized
            ),
        )
        cached = getattr(self, "_verified_teacher_focus_pool_cache", None)
        if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == signature:
            return cached[1], cached[2]

        row_weights = torch.zeros(int(self.q_teacher_replay.size), dtype=torch.float64)
        for motion, histogram in normalized:
            for risk_bin in (
                (histogram > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            ):
                match = self._teacher_phase_match_pool(motion, int(risk_bin))
                if match is None:
                    continue
                nearest_distance, pool = match
                if (
                    max_distance is not None
                    and nearest_distance / float(bin_count) > max_distance
                ):
                    continue
                # Preserve anchor-frequency weighting without making a dense
                # phase bin more likely solely because it contains more rows.
                row_weights[pool] += float(histogram[risk_bin]) / int(pool.numel())

        rows = (row_weights > 0).nonzero(as_tuple=False).squeeze(-1)
        weights = row_weights.index_select(0, rows)
        self._verified_teacher_focus_pool_cache = (signature, rows, weights)
        getattr(self, "_verified_teacher_focus_device_cache", {}).clear()
        return rows, weights

    def _verified_teacher_focus_tensors(
        self, device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache exact focused rows, weights, and dense membership per device."""
        device = torch.device(device)
        rows, weights = self._verified_teacher_focus_pool()
        signature = self._verified_teacher_focus_pool_cache[0]
        cache = getattr(self, "_verified_teacher_focus_device_cache", None)
        if cache is None:
            cache = {}
            self._verified_teacher_focus_device_cache = cache
        key = str(device)
        entry = cache.get(key)
        if entry is None or entry[0] != signature:
            device_rows = rows.to(device)
            device_weights = weights.to(device)
            membership = torch.zeros(
                int(self.q_teacher_replay.size), dtype=torch.bool, device=device
            )
            if device_rows.numel():
                membership[device_rows] = True
            entry = (signature, device_rows, device_weights, membership)
            cache[key] = entry
        return entry[1], entry[2], entry[3]

    def _teacher_intrinsic_focused_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """Return dynamic membership in the current verified Teacher focus pool."""
        if indices.numel() == 0:
            return torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
        if self._has_canonical_replay_mix():
            rows, _, membership = self._verified_teacher_focus_tensors(indices.device)
            if rows.numel() == 0:
                return torch.zeros(
                    indices.shape, dtype=torch.bool, device=indices.device
                )
            flat_indices = indices.detach().reshape(-1).long().to(membership.device)
            return membership.index_select(0, flat_indices).reshape(indices.shape)

        histogram = getattr(self, "_failure_phase_histogram", None)
        if (
            not torch.is_tensor(histogram)
            or not bool((histogram > 0).any())
            or REFERENCE_PHASE_KEY not in self.q_teacher_replay.data
        ):
            return torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
        if not bool(getattr(self, "_teacher_phase_index_ready", False)):
            self._build_teacher_phase_index()
        risk_bins = (histogram > 0).nonzero(as_tuple=False).squeeze(-1)
        teacher_bins = self._teacher_phase_nearest_nonempty.index_select(
            0, risk_bins.cpu()
        ).unique()
        replay_indices = indices.to(self.q_teacher_replay.device)
        phase = self.q_teacher_replay.data[REFERENCE_PHASE_KEY].index_select(
            0, replay_indices
        )
        phase = phase.reshape(indices.numel(), -1)[:, 0].float().clamp(0.0, 1.0)
        bins = (
            torch.floor(phase * int(self.cfg.failure_phase_num_bins))
            .long()
            .clamp_(0, int(self.cfg.failure_phase_num_bins) - 1)
        )
        teacher_bins = teacher_bins.to(bins.device)
        result = (bins.unsqueeze(-1) == teacher_bins).any(dim=-1)
        return result if result.device == indices.device else result.to(indices.device)

    def _attach_replay_sample_metadata(
        self,
        batch: dict[str, torch.Tensor],
        *,
        teacher_indices: torch.Tensor,
        student_indices: torch.Tensor,
        teacher_failure: torch.Tensor,
        student_failure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Attach batch-only provenance without overwriting intrinsic metadata."""
        row_count = int(teacher_indices.numel() + student_indices.numel())
        if not batch:
            raise RuntimeError("cannot attach replay metadata to an empty batch")
        device = next(iter(batch.values())).device
        teacher_failure = teacher_failure.reshape(-1).bool().to(device)
        student_failure = student_failure.reshape(-1).bool().to(device)
        if teacher_failure.numel() != teacher_indices.numel():
            raise ValueError("Teacher failure provenance is misaligned")
        if student_failure.numel() != student_indices.numel():
            raise ValueError("Student failure provenance is misaligned")
        provenance = torch.cat(
            (
                torch.where(
                    teacher_failure,
                    torch.full_like(
                        teacher_failure, REPLAY_SOURCE_FAILURE_TEACHER, dtype=torch.long
                    ),
                    torch.full_like(
                        teacher_failure, REPLAY_SOURCE_UNIFORM_TEACHER, dtype=torch.long
                    ),
                ),
                torch.where(
                    student_failure,
                    torch.full_like(
                        student_failure, REPLAY_SOURCE_FAILURE_STUDENT, dtype=torch.long
                    ),
                    torch.full_like(
                        student_failure, REPLAY_SOURCE_UNIFORM_STUDENT, dtype=torch.long
                    ),
                ),
            )
        )
        intrinsic = torch.cat(
            (
                self._teacher_intrinsic_focused_mask(teacher_indices).to(device),
                self._student_intrinsic_focused_mask(student_indices).to(device),
            )
        )
        # The two replay rings may live on a different device from each other
        # in test seams, and an empty single-source placeholder is CPU by
        # default.  Normalize before concatenation so the zero-share edges are
        # just as safe as a genuinely mixed batch.
        physical_indices = torch.cat(
            (
                teacher_indices.reshape(-1).to(device=device, dtype=torch.long),
                student_indices.reshape(-1).to(device=device, dtype=torch.long),
            )
        )
        is_teacher = torch.cat(
            (
                torch.ones(teacher_indices.numel(), dtype=torch.bool, device=device),
                torch.zeros(student_indices.numel(), dtype=torch.bool, device=device),
            )
        )
        if provenance.numel() != row_count or intrinsic.numel() != row_count:
            raise RuntimeError("replay metadata row count is inconsistent")
        batch[REPLAY_SAMPLE_PROVENANCE_KEY] = provenance
        batch[REPLAY_INTRINSIC_FOCUSED_KEY] = intrinsic.bool()
        batch[REPLAY_SAMPLE_PHYSICAL_INDEX_KEY] = physical_indices
        batch[REPLAY_SAMPLE_IS_TEACHER_KEY] = is_teacher
        # Deprecated compatibility masks remain provenance-only.
        batch[DAGGER_Q_TEACHER_SOURCE_KEY] = is_teacher
        batch[FAILURE_PHASE_TEACHER_SOURCE_KEY] = (
            provenance == REPLAY_SOURCE_FAILURE_TEACHER
        )
        batch[FAILURE_PHASE_STUDENT_SOURCE_KEY] = (
            provenance == REPLAY_SOURCE_FAILURE_STUDENT
        )
        return batch

    def _reset_replay_mix_rollout_metrics(self) -> None:
        self._replay_mix_rollout_metrics: dict[
            str, dict[str, float | torch.Tensor]
        ] = {}

    def _record_replay_mix_batch(
        self,
        purpose: str,
        batch: Mapping[str, torch.Tensor],
        requested_counts: Mapping[str, int],
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        provenance = batch[REPLAY_SAMPLE_PROVENANCE_KEY].reshape(-1).long()
        intrinsic = batch[REPLAY_INTRINSIC_FOCUSED_KEY].reshape(-1).bool()
        physical = batch[REPLAY_SAMPLE_PHYSICAL_INDEX_KEY].reshape(-1).long()
        is_teacher = batch[REPLAY_SAMPLE_IS_TEACHER_KEY].reshape(-1).bool()
        rows = int(provenance.numel())
        if not (
            intrinsic.numel() == rows
            and physical.numel() == rows
            and is_teacher.numel() == rows
        ):
            raise ValueError("replay metric metadata is misaligned")
        accumulator = self._replay_mix_rollout_metrics.get(purpose)
        if accumulator is None:
            accumulator = {
                "batches": 0.0,
                "rows": 0.0,
                **{f"requested_{source}_rows": 0.0 for source in REPLAY_SOURCE_ORDER},
                **{
                    key: torch.zeros((), dtype=torch.int64, device=provenance.device)
                    for key in _REPLAY_MIX_DEVICE_COUNTER_KEYS
                },
            }
            self._replay_mix_rollout_metrics[purpose] = accumulator
        accumulator["batches"] += 1.0
        accumulator["rows"] += float(rows)
        failure = (provenance == REPLAY_SOURCE_FAILURE_STUDENT) | (
            provenance == REPLAY_SOURCE_FAILURE_TEACHER
        )
        uniform = ~failure
        composite = physical + is_teacher.long() * int(self.dagger_replay.capacity)
        ordered_composite = composite.sort().values
        duplicate_rows = (ordered_composite[1:] == ordered_composite[:-1]).sum()
        if valid_mask is None:
            valid = torch.ones(rows, dtype=torch.bool, device=provenance.device)
        else:
            valid = valid_mask.reshape(-1).bool().to(provenance.device)
            if valid.numel() != rows:
                raise ValueError("perception valid mask is misaligned")

        source_masks = torch.stack(
            [provenance == source_id for source_id in range(len(REPLAY_SOURCE_ORDER))]
        )
        batch_device_counts = (
            intrinsic.sum(),
            failure.sum(),
            (uniform & intrinsic).sum(),
            uniform.sum(),
            duplicate_rows,
            valid.sum(),
            *source_masks.sum(dim=-1).unbind(),
            *(source_masks & valid.unsqueeze(0)).sum(dim=-1).unbind(),
        )
        for key, value in zip(
            _REPLAY_MIX_DEVICE_COUNTER_KEYS, batch_device_counts, strict=True
        ):
            target = accumulator[key]
            if not torch.is_tensor(target) or target.device != provenance.device:
                raise RuntimeError("replay metric accumulator device changed")
            target.add_(value)
        for source in REPLAY_SOURCE_ORDER:
            accumulator[f"requested_{source}_rows"] += float(
                int(requested_counts[source])
            )

    def _replay_mix_metrics(self, purpose: str) -> dict[str, float]:
        fractions = self._replay_mix_fractions(purpose)
        accumulator = self._replay_mix_rollout_metrics.get(purpose, {})
        rows = float(accumulator.get("rows", 0.0))
        device_counters: dict[str, float] = {}
        if accumulator:
            packed_counters = torch.stack(
                [
                    torch.as_tensor(accumulator[key]).detach().long()
                    for key in _REPLAY_MIX_DEVICE_COUNTER_KEYS
                ]
            )
            host_counters = packed_counters.to("cpu").tolist()
            device_counters = dict(
                zip(_REPLAY_MIX_DEVICE_COUNTER_KEYS, host_counters, strict=True)
            )
        valid_rows = float(device_counters.get("valid_rows", 0.0))
        result: dict[str, float] = {}
        for source in REPLAY_SOURCE_ORDER:
            requested = float(accumulator.get(f"requested_{source}_rows", 0.0))
            actual = float(device_counters.get(f"actual_{source}_rows", 0.0))
            valid = float(device_counters.get(f"valid_{source}_rows", 0.0))
            result[f"configured_{source}_fraction"] = float(fractions[source])
            result[f"requested_{source}_rows"] = requested
            result[f"requested_{source}_fraction"] = requested / max(rows, 1.0)
            result[f"actual_{source}_rows"] = actual
            result[f"actual_{source}_fraction"] = actual / max(rows, 1.0)
            result[f"final_{source}_rows"] = actual
            result[f"valid_loss_{source}_fraction"] = valid / max(valid_rows, 1.0)
        result["failure_student_backfill_rows"] = max(
            result["requested_failure_student_rows"]
            - result["actual_failure_student_rows"],
            0.0,
        )
        result["failure_teacher_backfill_rows"] = max(
            result["requested_failure_teacher_rows"]
            - result["actual_failure_teacher_rows"],
            0.0,
        )
        result["intrinsic_focused_row_fraction"] = float(
            device_counters.get("intrinsic_focused_rows", 0.0)
        ) / max(rows, 1.0)
        result["failure_stratum_provenance_fraction"] = float(
            device_counters.get("failure_provenance_rows", 0.0)
        ) / max(rows, 1.0)
        result["duplicate_row_fraction"] = float(
            device_counters.get("duplicate_rows", 0.0)
        ) / max(rows, 1.0)
        result["uniform_draw_intrinsic_focused_fraction"] = float(
            device_counters.get("uniform_intrinsic_rows", 0.0)
        ) / max(float(device_counters.get("uniform_rows", 0.0)), 1.0)
        result["sampled_rows"] = rows
        result["valid_loss_rows"] = valid_rows
        return result

    @torch.no_grad()
    def _build_teacher_phase_index(self) -> None:
        """Index immutable Teacher rows by reference phase for focused draws."""
        if self.q_teacher_replay.size < 1:
            raise RuntimeError("cannot index an empty Teacher replay")
        if REFERENCE_PHASE_KEY not in self.q_teacher_replay.data:
            raise KeyError("Teacher replay lacks reference phase metadata")
        bins = int(self.cfg.failure_phase_num_bins)
        phase = self.q_teacher_replay.data[REFERENCE_PHASE_KEY][
            : self.q_teacher_replay.size
        ]
        phase = phase.reshape(self.q_teacher_replay.size, -1)[:, 0]
        phase = phase.detach().float().cpu().clamp(0.0, 1.0)
        phase_bins = torch.floor(phase * bins).long().clamp_(0, bins - 1)
        motion_ids = self.q_teacher_replay.data.get(REPLAY_MOTION_ID_KEY)
        if motion_ids is None:
            if self._has_canonical_replay_mix():
                raise KeyError("canonical Teacher replay lacks replay_motion_id")
            motion_ids = torch.zeros(self.q_teacher_replay.size, dtype=torch.long)
        else:
            motion_ids = (
                motion_ids[: self.q_teacher_replay.size]
                .reshape(self.q_teacher_replay.size, -1)[:, 0]
                .detach()
                .long()
                .cpu()
            )
        rows = tuple(
            (phase_bins == bin_index).nonzero(as_tuple=False).squeeze(-1)
            for bin_index in range(bins)
        )
        counts = torch.tensor([row.numel() for row in rows], dtype=torch.long)
        nonempty = counts.nonzero(as_tuple=False).squeeze(-1)
        if nonempty.numel() < 1:
            raise RuntimeError("Teacher phase index contains no rows")
        distance = (
            torch.arange(bins, dtype=torch.long).unsqueeze(1) - nonempty.unsqueeze(0)
        ).abs()
        nearest = nonempty.index_select(0, distance.argmin(dim=1))
        if bool((counts.index_select(0, nearest) < 1).any()):
            raise RuntimeError("Teacher nearest-bin index selected an empty bin")
        flat_rows = torch.cat([rows[index] for index in nonempty.tolist()])
        starts = torch.zeros(bins, dtype=torch.long)
        running = 0
        for bin_index in range(bins):
            starts[bin_index] = running
            running += int(counts[bin_index])

        self._teacher_phase_bin_rows = rows
        self._teacher_phase_nearest_nonempty = nearest
        self._teacher_phase_flat_rows = flat_rows
        self._teacher_phase_bin_starts = starts
        self._teacher_phase_bin_counts = counts
        self._teacher_replay_phase_bins = phase_bins
        self._teacher_replay_motion_ids = motion_ids
        getattr(self, "_teacher_phase_device_cache", {}).clear()
        self._verified_teacher_focus_pool_cache = None
        getattr(self, "_verified_teacher_focus_device_cache", {}).clear()
        self._teacher_phase_index_ready = True

    def _teacher_phase_tensors(self, device):
        device = torch.device(device)
        cache = getattr(self, "_teacher_phase_device_cache", None)
        if cache is None:
            cache = {}
            self._teacher_phase_device_cache = cache
        key = str(device)
        tensors = cache.get(key)
        if tensors is None:
            tensors = tuple(
                value.to(device)
                for value in (
                    self._teacher_phase_nearest_nonempty,
                    self._teacher_phase_bin_starts,
                    self._teacher_phase_bin_counts,
                    self._teacher_phase_flat_rows,
                )
            )
            cache[key] = tensors
        return tensors

    def _failure_histogram_tensor(self, device) -> torch.Tensor:
        device = torch.device(device)
        histogram = getattr(self, "_failure_phase_histogram", None)
        if histogram is None:
            histogram = torch.zeros(
                int(self.cfg.failure_phase_num_bins), dtype=torch.float64
            )
            self._failure_phase_histogram = histogram
        cache = getattr(self, "_failure_histogram_device_cache", None)
        if cache is None:
            cache = {}
            self._failure_histogram_device_cache = cache
        version = int(getattr(self, "_failure_phase_anchor_count", 0))
        key = str(device)
        entry = cache.get(key)
        if entry is None or entry[0] != version:
            entry = (version, histogram.to(device))
            cache[key] = entry
        return entry[1]

    @torch.no_grad()
    def _draw_verified_motion_teacher_indices(
        self,
        count: int,
        generator: torch.Generator,
        *,
        focused_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw unique canonical FT rows, backfilling shortage with uniform Teacher."""
        generator_device = torch.device(generator.device)

        def _uniform(draw_count: int) -> torch.Tensor:
            return torch.randint(
                0,
                self.q_teacher_replay.size,
                (draw_count,),
                device=generator_device,
                generator=generator,
            )

        if focused_count == 0:
            return (
                _uniform(count),
                torch.zeros(count, dtype=torch.bool, device=generator_device),
            )
        focused_pool, focused_weights, _ = self._verified_teacher_focus_tensors(
            generator_device
        )
        actual_focused = min(focused_count, int(focused_pool.numel()))
        if actual_focused == 0:
            self._failure_phase_uniform_fallback_rows = (
                int(getattr(self, "_failure_phase_uniform_fallback_rows", 0))
                + focused_count
            )
            return (
                _uniform(count),
                torch.zeros(count, dtype=torch.bool, device=generator_device),
            )
        focused_positions = torch.multinomial(
            focused_weights,
            actual_focused,
            replacement=False,
            generator=generator,
        )
        focused_rows = focused_pool.index_select(0, focused_positions)
        uniform_count = count - actual_focused
        indices = torch.cat((_uniform(uniform_count), focused_rows), dim=0)
        focused = torch.cat(
            (
                torch.zeros(uniform_count, dtype=torch.bool, device=generator_device),
                torch.ones(actual_focused, dtype=torch.bool, device=generator_device),
            )
        )
        self._failure_phase_focused_rows = (
            int(getattr(self, "_failure_phase_focused_rows", 0)) + actual_focused
        )
        self._failure_phase_uniform_fallback_rows = (
            int(getattr(self, "_failure_phase_uniform_fallback_rows", 0))
            + focused_count
            - actual_focused
        )
        return indices, focused

    @torch.no_grad()
    def _draw_teacher_indices(
        self,
        count: int,
        generator: torch.Generator,
        *,
        focused_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Teacher rows on the generator device without synchronizing."""
        count = int(count)
        focused_count = int(focused_count)
        if count < 1 or not 0 <= focused_count <= count:
            raise ValueError("Teacher sample counts are invalid")
        if self.q_teacher_replay.size < 1:
            raise RuntimeError("cannot sample an empty Teacher replay")
        if self._has_canonical_replay_mix():
            return self._draw_verified_motion_teacher_indices(
                count, generator, focused_count=focused_count
            )
        generator_device = torch.device(generator.device)
        uniform_count = count - focused_count

        def _uniform(draw_count: int) -> torch.Tensor:
            return torch.randint(
                0,
                self.q_teacher_replay.size,
                (draw_count,),
                device=generator_device,
                generator=generator,
            )

        if focused_count == 0:
            indices = _uniform(count)
            focused = torch.zeros(count, dtype=torch.bool, device=generator_device)
            return indices, focused

        histogram_ready = bool(
            getattr(self, "_failure_phase_anchor_count", 0)
            or getattr(self, "_failure_phase_histogram", torch.empty(0)).sum().item()
            > 0.0
        )
        if not histogram_ready:
            indices = _uniform(count)
            focused = torch.zeros(count, dtype=torch.bool, device=generator_device)
            if focused_count:
                self._failure_phase_uniform_fallback_rows = (
                    int(getattr(self, "_failure_phase_uniform_fallback_rows", 0))
                    + focused_count
                )
        else:
            if not bool(getattr(self, "_teacher_phase_index_ready", False)):
                self._build_teacher_phase_index()
            probabilities = self._failure_histogram_tensor(generator_device)
            risk_bins = torch.multinomial(
                probabilities,
                focused_count,
                replacement=True,
                generator=generator,
            )
            nearest, starts, counts, flat_rows = self._teacher_phase_tensors(
                generator_device
            )
            teacher_bins = nearest.index_select(0, risk_bins)
            selected_counts = counts.index_select(0, teacher_bins)
            offsets = torch.floor(
                torch.rand(
                    focused_count,
                    device=generator_device,
                    generator=generator,
                )
                * selected_counts
            ).long()
            flat_positions = starts.index_select(0, teacher_bins) + offsets
            focused_rows = flat_rows.index_select(0, flat_positions)
            indices = torch.cat((_uniform(uniform_count), focused_rows), dim=0)
            focused = torch.cat(
                (
                    torch.zeros(
                        uniform_count, dtype=torch.bool, device=generator_device
                    ),
                    torch.ones(
                        focused_count, dtype=torch.bool, device=generator_device
                    ),
                )
            )
            self._failure_phase_focused_rows = (
                int(getattr(self, "_failure_phase_focused_rows", 0)) + focused_count
            )

        return indices, focused

    @torch.no_grad()
    def _sample_teacher_indices(
        self,
        count: int,
        generator: torch.Generator,
        *,
        focused_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Teacher rows and move indices to the replay device."""
        indices, focused = self._draw_teacher_indices(
            count, generator, focused_count=focused_count
        )
        replay_device = self.q_teacher_replay.device
        if indices.device != replay_device:
            indices = indices.to(replay_device)
            focused = focused.to(replay_device)
        return indices, focused

    def _student_replay_source_rows(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all-Student and exact failed-bottleneck populations."""
        replay = self.dagger_replay
        all_student = replay._valid_indices(DAGGER_IS_STUDENT_ACTION_KEY)
        if all_student.numel() < 1:
            raise RuntimeError("Cannot sample before a Student row exists")
        focused_key = (
            REPLAY_INTRINSIC_FOCUSED_KEY
            if REPLAY_INTRINSIC_FOCUSED_KEY in replay.data
            else FAILURE_PHASE_STUDENT_SOURCE_KEY
        )
        if focused_key not in replay.data:
            empty = torch.empty(0, dtype=torch.long, device=replay.device)
            return all_student, empty
        focused = replay._valid_indices(focused_key)
        return all_student, focused

    @torch.no_grad()
    def _draw_student_indices(
        self,
        count: int,
        generator: torch.Generator,
        *,
        focused_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a capped focused-Student stratum plus uniform backfill.

        Focused rows are sampled without replacement within a batch.  If the
        exact failed-bottleneck population has fewer rows than the configured
        quota, its shortfall is reassigned to the uniform Student population.
        Uniform replay retains the established with-replacement behavior.
        """
        count = int(count)
        focused_count = int(focused_count)
        if count < 1 or not 0 <= focused_count <= count:
            raise ValueError("Student sample counts are invalid")
        uniform_rows, focused_rows = self._student_replay_source_rows()
        generator_device = torch.device(generator.device)
        actual_focused = min(focused_count, int(focused_rows.numel()))
        uniform_count = count - actual_focused

        uniform_positions = torch.randint(
            0,
            uniform_rows.numel(),
            (uniform_count,),
            device=generator_device,
            generator=generator,
        )
        if uniform_positions.device != uniform_rows.device:
            uniform_positions = uniform_positions.to(uniform_rows.device)
        uniform_indices = uniform_rows.index_select(0, uniform_positions)
        if uniform_indices.device != generator_device:
            uniform_indices = uniform_indices.to(generator_device)

        if actual_focused:
            focus_order = torch.randperm(
                focused_rows.numel(),
                device=generator_device,
                generator=generator,
            )[:actual_focused]
            if focus_order.device != focused_rows.device:
                focus_order = focus_order.to(focused_rows.device)
            focused_indices = focused_rows.index_select(0, focus_order)
            if focused_indices.device != generator_device:
                focused_indices = focused_indices.to(generator_device)
            indices = torch.cat((uniform_indices, focused_indices), dim=0)
        else:
            indices = uniform_indices

        focused = torch.cat(
            (
                torch.zeros(uniform_count, dtype=torch.bool, device=generator_device),
                torch.ones(actual_focused, dtype=torch.bool, device=generator_device),
            )
        )
        shortfall = focused_count - actual_focused
        self._failure_phase_student_focused_rows = (
            int(getattr(self, "_failure_phase_student_focused_rows", 0))
            + actual_focused
        )
        self._failure_phase_student_uniform_fallback_rows = (
            int(getattr(self, "_failure_phase_student_uniform_fallback_rows", 0))
            + shortfall
        )
        return indices, focused

    @torch.no_grad()
    def _sample_student_indices(
        self,
        count: int,
        generator: torch.Generator,
        *,
        focused_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, focused = self._draw_student_indices(
            count, generator, focused_count=focused_count
        )
        replay_device = self.dagger_replay.device
        if indices.device != replay_device:
            indices = indices.to(replay_device)
            focused = focused.to(replay_device)
        return indices, focused

    @torch.no_grad()
    def _prefetch_curriculum_sample_plans(
        self, update_count: int
    ) -> tuple[_TD3ReplaySamplePlan, ...]:
        """Prefetch phase-aware CPU replay rows for one optimizer burst.

        This retains the coalesced pinned replay transfer path.  It also draws
        every Student Actor row from the Student-executed subset.  Teacher
        focused rows are sampled independently for Q and Actor updates from the
        same online failure-phase distribution.
        """
        update_count = int(update_count)
        if update_count < 0:
            raise ValueError("update_count must be non-negative")
        batch_size = int(self.cfg.q_batch_size)
        q_counts = self._replay_source_counts("q", batch_size)
        student_count = q_counts["uniform_student"] + q_counts["failure_student"]
        q_focused_count = q_counts["failure_teacher"]
        teacher_count = q_counts["uniform_teacher"] + q_focused_count
        actor_batch_size = int(self.cfg.dagger_batch_size)
        actor_counts = self._replay_source_counts("actor", actor_batch_size)
        actor_main_count = (
            actor_counts["uniform_student"] + actor_counts["failure_student"]
        )
        actor_focused_count = actor_counts["failure_teacher"]
        actor_teacher_count = actor_counts["uniform_teacher"] + actor_focused_count
        q_student_focused_count = q_counts["failure_student"]
        actor_student_focused_count = actor_counts["failure_student"]
        generator_device = torch.device(self.q_rng.device)
        if update_count == 0:
            return ()
        index_draws: list[torch.Tensor] = []
        records = []
        for update_index in range(update_count):
            teacher_draw = None
            teacher_focused = torch.zeros(0, dtype=torch.bool, device=generator_device)
            if teacher_count:
                teacher_indices, teacher_focused = self._draw_teacher_indices(
                    teacher_count,
                    self.q_rng,
                    focused_count=q_focused_count,
                )
                teacher_draw = len(index_draws)
                index_draws.append(teacher_indices)
            student_draw = None
            student_focused = torch.zeros(0, dtype=torch.bool, device=generator_device)
            if student_count:
                (
                    student_indices,
                    student_focused,
                ) = self._draw_student_indices(
                    student_count,
                    self.q_rng,
                    focused_count=q_student_focused_count,
                )
                student_draw = len(index_draws)
                index_draws.append(student_indices)
            permutation = torch.randperm(
                batch_size, device=generator_device, generator=self.q_rng
            )

            actor_draw = None
            actor_teacher_draw = None
            actor_teacher_focused = None
            actor_student_focused = None
            if (int(self.critic_update_count) + update_index + 1) % int(
                self.cfg.policy_delay
            ) == 0:
                if actor_teacher_count:
                    actor_teacher_indices, actor_teacher_focused = (
                        self._draw_teacher_indices(
                            actor_teacher_count,
                            self.q_rng,
                            focused_count=actor_focused_count,
                        )
                    )
                    actor_teacher_draw = len(index_draws)
                    index_draws.append(actor_teacher_indices)
                if actor_main_count:
                    (
                        actor_student_indices,
                        actor_student_focused,
                    ) = self._draw_student_indices(
                        actor_main_count,
                        self.q_rng,
                        focused_count=actor_student_focused_count,
                    )
                    actor_draw = len(index_draws)
                    index_draws.append(actor_student_indices)
            records.append(
                (
                    teacher_draw,
                    teacher_focused,
                    student_draw,
                    permutation,
                    actor_draw,
                    actor_teacher_draw,
                    actor_teacher_focused,
                    student_focused,
                    actor_student_focused,
                )
            )

        lengths = [int(draw.numel()) for draw in index_draws]
        packed_indices = torch.cat(index_draws)
        if packed_indices.device.type != "cpu":
            packed_indices = packed_indices.to("cpu")
        cpu_draws = packed_indices.split(lengths)
        plans = []
        for (
            teacher_draw,
            teacher_focused,
            student_draw,
            permutation,
            actor_draw,
            actor_teacher_draw,
            actor_teacher_focused,
            student_focused,
            actor_student_focused,
        ) in records:
            teacher_indices = (
                torch.empty(0, dtype=torch.long)
                if teacher_draw is None
                else cpu_draws[teacher_draw]
            )
            student_indices = (
                torch.empty(0, dtype=torch.long)
                if student_draw is None
                else cpu_draws[student_draw]
            )
            actor_indices = None if actor_draw is None else cpu_draws[actor_draw]
            actor_teacher_indices = (
                None if actor_teacher_draw is None else cpu_draws[actor_teacher_draw]
            )
            plans.append(
                _TD3ReplaySamplePlan(
                    teacher_indices=teacher_indices,
                    student_indices=student_indices,
                    permutation=permutation,
                    actor_indices=actor_indices,
                    actor_teacher_indices=actor_teacher_indices,
                    teacher_focused=teacher_focused,
                    actor_teacher_focused=actor_teacher_focused,
                    student_focused=student_focused,
                    actor_student_focused=actor_student_focused,
                )
            )
        return tuple(plans)

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return _DistributionalTD3DaggerRolloutPolicy(self)
        # Evaluation is the unprojected PPOVEL-compatible raw Student mean.
        return _DeterministicTD3StudentEvalPolicy(self)

    def configure_teacher_replay(self, path, restore_path=None):
        """Disable the duplicate persistent H5/export FIFO for TD3."""
        if restore_path is not None:
            raise ValueError("TD3 raw-perception replay cannot restore a teacher H5")
        self.teacher_replay = None

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        """The two online CPU rings are intentionally not exported to H5."""
        del iteration, checkpoint_name
        return None

    def restore_q_teacher_replay(self, source_path):
        del source_path
        raise ValueError(
            "TD3 is fresh-only and cannot refill raw perception windows from H5"
        )

    def _ensure_replay_object_geo_codebook(self) -> None:
        """Install fresh-only geometry codebook state for focused test seams."""
        if not hasattr(self, "_replay_object_geo_bank"):
            self._replay_object_geo_bank = None
        if not hasattr(self, "_replay_object_geo_hash_index"):
            self._replay_object_geo_hash_index = {}
        if not hasattr(self, "_replay_object_geo_bank_generation"):
            self._replay_object_geo_bank_generation = 0
        if not hasattr(self, "_replay_object_geo_device_banks"):
            self._replay_object_geo_device_banks = {}
        if not hasattr(self, "_replay_object_geo_fingerprint"):
            self._replay_object_geo_fingerprint = None

    def _reset_replay_object_geo_codebook(self) -> None:
        """Discard geometry IDs together with the fresh-only replay rings."""
        self._replay_object_geo_bank = None
        self._replay_object_geo_hash_index = {}
        self._replay_object_geo_bank_generation = 0
        self._replay_object_geo_device_banks = {}
        self._replay_object_geo_fingerprint = None

    @torch.no_grad()
    def _encode_replay_object_geo(self, td: TensorDict) -> torch.Tensor:
        """Intern PPOVEL geometry rows and return transition-aligned int32 IDs."""
        self._ensure_replay_object_geo_codebook()
        if OBJECT_GEO_KEY not in td.keys(True, True):
            raise KeyError(f"raw perception replay is missing {OBJECT_GEO_KEY!r}")
        geometry = td[OBJECT_GEO_KEY]
        spec = self.observation_spec.get(OBJECT_GEO_KEY, None)
        if spec is None:
            raise KeyError(f"observation spec is missing {OBJECT_GEO_KEY!r}")
        width = int(spec.shape[-1])
        expected_shape = (*td.batch_size, width)
        if geometry.shape != expected_shape:
            raise ValueError(
                f"{OBJECT_GEO_KEY} has shape {tuple(geometry.shape)}; "
                f"expected {tuple(expected_shape)}"
            )
        if geometry.numel() == 0:
            raise ValueError(f"{OBJECT_GEO_KEY} replay batch cannot be empty")
        if not geometry.is_floating_point() or not torch.isfinite(geometry).all():
            raise ValueError(f"{OBJECT_GEO_KEY} must be finite floating point")

        contract_payload = json.dumps(
            {
                "semantics": "exact_append_only_geometry_codebook_int32_id_v1",
                "width": width,
                "dtype": str(geometry.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = "sha256:" + hashlib.sha256(contract_payload).hexdigest()
        current = getattr(self, "_replay_object_geo_fingerprint", None)
        if current is None:
            self._replay_object_geo_fingerprint = fingerprint
        elif current != fingerprint:
            raise RuntimeError("object_geo_ replay shape/dtype contract changed")

        flat = geometry.detach().reshape(-1, width)

        # A rollout normally repeats one geometry for many consecutive states.
        # Running ``torch.unique`` across every wide state would sort hundreds
        # of MiB for a typical many-environment rollout.  First retain only the
        # first state and exact temporal changes for each environment.  This
        # still handles arbitrary reset-time changes and degrades safely to all
        # rows if geometry genuinely changes at every step.
        temporal_change = None
        if len(td.batch_size) == 2:
            env_count, step_count = map(int, td.batch_size)
            temporal = geometry.reshape(env_count, step_count, width)
            temporal_change = torch.ones(
                (env_count, step_count), dtype=torch.bool, device=geometry.device
            )
            if step_count > 1:
                changed = torch.zeros(
                    (env_count, step_count - 1),
                    dtype=torch.bool,
                    device=geometry.device,
                )
                # Bound comparison temporaries independently of geometry width.
                for feature_start in range(0, width, 128):
                    feature_stop = min(feature_start + 128, width)
                    changed.logical_or_(
                        temporal[:, 1:, feature_start:feature_stop]
                        .ne(temporal[:, :-1, feature_start:feature_stop])
                        .any(dim=-1)
                    )
                temporal_change[:, 1:] = changed
            candidate_indices = temporal_change.reshape(-1).nonzero(
                as_tuple=False
            ).squeeze(-1)
            candidates = flat.index_select(0, candidate_indices)
        else:
            candidates = flat

        candidates_cpu = candidates.to(device="cpu").contiguous()
        unique_cpu, candidate_inverse = torch.unique(
            candidates_cpu, dim=0, return_inverse=True
        )
        bank = self._replay_object_geo_bank
        if bank is not None and (
            bank.dtype != unique_cpu.dtype or int(bank.shape[1]) != width
        ):
            raise RuntimeError("object_geo_ codebook contract changed")

        resolved_unique_ids: list[int] = []
        new_rows: list[torch.Tensor] = []
        new_digests: list[str] = []
        base_size = 0 if bank is None else int(bank.shape[0])
        for row in unique_cpu:
            digest = hashlib.sha256(
                row.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            resolved = None
            for candidate in self._replay_object_geo_hash_index.get(digest, ()):
                if bank is not None and torch.equal(bank[candidate], row):
                    resolved = int(candidate)
                    break
            if resolved is None:
                # torch.unique has already removed duplicates within this call,
                # so every unresolved row receives exactly one new ID.
                resolved = base_size + len(new_rows)
                new_rows.append(row.clone())
                new_digests.append(digest)
            resolved_unique_ids.append(resolved)

        if new_rows:
            if base_size + len(new_rows) > torch.iinfo(torch.int32).max + 1:
                raise OverflowError("object geometry replay codebook exhausted int32 IDs")
            appended = torch.stack(new_rows, dim=0).contiguous()
            with torch.inference_mode(False):
                appended = appended.detach().clone()
                self._replay_object_geo_bank = (
                    appended
                    if bank is None
                    else torch.cat((bank, appended), dim=0).contiguous()
                )
            for offset, digest in enumerate(new_digests):
                self._replay_object_geo_hash_index.setdefault(digest, []).append(
                    base_size + offset
                )
            self._replay_object_geo_bank_generation += 1
            self._replay_object_geo_device_banks.clear()

        unique_ids = torch.tensor(resolved_unique_ids, dtype=torch.int32)
        candidate_ids = unique_ids.index_select(0, candidate_inverse.reshape(-1))
        if temporal_change is None:
            encoded = candidate_ids.reshape(tuple(td.batch_size)).to(
                geometry.device
            )
        else:
            # Map every unchanged state to the most recent candidate within its
            # own environment.  Candidate ordinals increase in flattened
            # environment/time order, so cumulative max is an exact forward fill.
            change_cpu = temporal_change.to(device="cpu")
            candidate_ordinal = torch.full(
                change_cpu.shape, -1, dtype=torch.long, device="cpu"
            )
            candidate_ordinal[change_cpu] = torch.arange(
                candidate_ids.numel(), dtype=torch.long
            )
            candidate_ordinal = candidate_ordinal.cummax(dim=1).values
            encoded = candidate_ids.index_select(
                0, candidate_ordinal.reshape(-1)
            ).reshape(tuple(td.batch_size)).to(geometry.device)
        # Collection may run under inference_mode; replay fields must own
        # ordinary tensors because they are later copied into mutable rings.
        with torch.inference_mode(False):
            return encoded.detach().clone()

    @torch.no_grad()
    def _decode_replay_object_geo(
        self,
        geometry_ids: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resolve transition-aligned IDs to exact PPOVEL geometry values."""
        self._ensure_replay_object_geo_codebook()
        bank = self._replay_object_geo_bank
        if bank is None or int(bank.shape[0]) < 1:
            raise RuntimeError("raw perception replay has no object geometry codebook")
        if geometry_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("object geometry replay IDs must be int32 or int64")
        output_device = torch.device(device)
        cache_key = (str(output_device), dtype)
        cached = self._replay_object_geo_device_banks.get(cache_key)
        generation = int(self._replay_object_geo_bank_generation)
        if cached is None or cached[0] != generation:
            # Decode may first run inside rollout ``inference_mode``.  Cache an
            # ordinary tensor so a later differentiable perception update can
            # safely use geometry in operations that save it for backward.
            with torch.inference_mode(False):
                device_bank = (
                    bank.to(device=output_device, dtype=dtype)
                    .contiguous()
                    .detach()
                    .clone()
                )
            self._replay_object_geo_device_banks[cache_key] = (
                generation,
                device_bank,
            )
        else:
            device_bank = cached[1]
        flat_ids = geometry_ids.reshape(-1).to(
            device=output_device, dtype=torch.long
        )
        if bool((flat_ids < 0).any()) or bool(
            (flat_ids >= int(device_bank.shape[0])).any()
        ):
            raise IndexError("object geometry replay ID is outside the codebook")
        # Keep the public decode result ordinary as well.  This makes the API
        # safe even if a caller first decodes under rollout inference mode and
        # later uses that tensor in differentiable object-transform code.
        with torch.inference_mode(False):
            return device_bank.index_select(0, flat_ids).reshape(
                *geometry_ids.shape, int(device_bank.shape[-1])
            )

    @torch.no_grad()
    def _raw_perception_values(self, td: TensorDict) -> dict[str, torch.Tensor]:
        return {
            PERCEPTION_DEPTH_U8_KEY: _encode_replay_depth_u8(
                self._replay_source(td, DEPTH_KEY)
            ),
            PERCEPTION_POLICY_RAW_KEY: self._replay_source(td, OBS_KEY).detach(),
            PERCEPTION_VEL_COMMAND_RAW_KEY: self._replay_source(
                td, VEL_CMD_KEY
            ).detach(),
            PERCEPTION_OBJECT_GEO_ID_KEY: self._encode_replay_object_geo(td),
            PERCEPTION_IS_INIT_KEY: td["is_init"].detach().bool(),
        }

    @torch.no_grad()
    def _materialize_collection_actor_observations(
        self, td: TensorDict
    ) -> torch.Tensor:
        """Advance a cloned state through the live EMA perception stack once.

        This is used only for a rollout-final continuation or a true pre-reset
        timeout final.  Ordinary next states reuse the following rollout row's
        already-materialized cache and therefore require no extra encoder pass.
        """
        if hasattr(self, "temporal_depth_gru_ema"):
            self.temporal_depth_gru_ema(td)
        else:
            reference = td[OBS_KEY]
            td["_depth_feature"] = torch.zeros(
                *td.batch_size,
                self.depth_feature_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        if bool(self.cfg.use_object_adapt):
            self.object_adapt_ema(td)
            self.object_pred_transform(td)
        self.adapt_ema(td)
        return self._collection_actor_observations(td)

    @torch.no_grad()
    def _rebuild_teacher_actor_cache(
        self, lineage: TeacherActorCacheLineage
    ) -> None:
        """Stream complete Teacher episodes through the current EMA exactly."""
        self._ensure_teacher_episode_cache_state()
        store = self._teacher_episode_store
        if not store.frozen:
            raise RuntimeError("Teacher Actor cache requires a frozen episode store")

        # Keep the derived cache on the learning device.  Copying every
        # 525-wide state back to the CPU ring (and then copying sampled rows
        # to CUDA again) made an exact refresh substantially slower than the
        # legacy ten-frame path.  A single device cache lets the recurrent
        # stream publish without per-chunk D2H synchronization; Q/Actor later
        # gather only the Teacher rows that their unchanged replay indices
        # selected.
        actor_by_node = self._teacher_actor_cache.allocate_build_tensor(
            store, device=self.device
        )
        write_counts = torch.zeros(
            store.node_count, dtype=torch.int16, device=self.device
        )
        snapshot = self._vecnorm_snapshot()
        time_chunk_size = max(1, int(self.cfg.train_every))
        node_budget = max(
            1,
            int(self.cfg.perception_encode_microbatch_size)
            * (int(self.cfg.perception_replay_burn_in) + 2),
        )
        episode_batch_size = max(1, node_budget // time_chunk_size)
        raw_lineage = store.lineage
        device_raw_lineage = (
            raw_lineage.store_id,
            raw_lineage.generation,
            str(torch.device(self.device)),
        )
        if self._teacher_episode_device_raw_lineage != device_raw_lineage:
            # The Teacher store is immutable after prefill.  Upload its
            # compact, frame-once raw tensors once and reuse them across all
            # later EMA generations.  Only the derived Actor cache is
            # invalidated after perception learning.
            self._teacher_episode_device_raw_fields = {
                key: value.to(self.device).contiguous()
                for key, value in store.raw_fields.items()
            }
            self._teacher_episode_device_raw_lineage = device_raw_lineage
        device_raw_fields = self._teacher_episode_device_raw_fields
        if device_raw_fields is None:  # pragma: no cover - guarded above
            raise RuntimeError("Teacher raw device mirror is unavailable")
        current_group = None
        depth_state = None
        adapt_state = None

        with set_recurrent_mode(True):
            for chunk in store.iter_sequence_chunks(
                episode_batch_size=episode_batch_size,
                time_chunk_size=time_chunk_size,
                raw_fields=device_raw_fields,
            ):
                if current_group != chunk.group_id:
                    current_group = chunk.group_id
                    depth_state = torch.zeros(
                        chunk.group_size,
                        self.depth_feature_dim,
                        device=self.device,
                    )
                    adapt_state = torch.zeros(
                        chunk.group_size,
                        int(self.cfg.latent_dim),
                        device=self.device,
                    )
                if depth_state is None or adapt_state is None:  # pragma: no cover
                    raise RuntimeError("Teacher recurrent cache state is unavailable")

                positions = chunk.batch_positions.to(
                    device=self.device, dtype=torch.long
                )
                count, sequence_length = chunk.valid.shape
                depth_u8 = chunk.raw_fields[PERCEPTION_DEPTH_U8_KEY].to(
                    self.device
                )
                policy_raw = chunk.raw_fields[PERCEPTION_POLICY_RAW_KEY].to(
                    self.device
                )
                vel_raw = chunk.raw_fields[PERCEPTION_VEL_COMMAND_RAW_KEY].to(
                    self.device
                )
                geometry = self._decode_replay_object_geo(
                    chunk.raw_fields[PERCEPTION_OBJECT_GEO_ID_KEY],
                    device=self.device,
                    dtype=policy_raw.dtype,
                )
                depth = self._normalize_replay_value(
                    DEPTH_KEY, _decode_replay_depth_u8(depth_u8), snapshot
                )
                policy = self._normalize_replay_value(
                    OBS_KEY, policy_raw, snapshot
                )
                vel = self._normalize_replay_value(VEL_CMD_KEY, vel_raw, snapshot)
                depth_hx = depth_state.index_select(0, positions)
                adapt_hx = adapt_state.index_select(0, positions)
                td = TensorDict(
                    {
                        DEPTH_KEY: depth,
                        OBS_KEY: policy,
                        VEL_CMD_KEY: vel,
                        OBJECT_GEO_KEY: geometry.to(dtype=policy.dtype),
                        "is_init": chunk.raw_fields[PERCEPTION_IS_INIT_KEY].to(
                            self.device
                        ),
                        "depth_hx": depth_hx.unsqueeze(1).expand(
                            count, sequence_length, -1
                        ),
                        "adapt_hx": adapt_hx.unsqueeze(1).expand(
                            count, sequence_length, -1
                        ),
                    },
                    batch_size=(count, sequence_length),
                    device=self.device,
                )
                if hasattr(self, "temporal_depth_gru_ema"):
                    self.temporal_depth_gru_ema(td)
                    if ("next", "depth_hx") not in td.keys(True, True):
                        raise RuntimeError(
                            "Teacher cache requires the locked recurrent depth EMA"
                        )
                    depth_state.index_copy_(
                        0, positions, td["next", "depth_hx"][:, -1]
                    )
                else:
                    td["_depth_feature"] = torch.zeros(
                        count,
                        sequence_length,
                        self.depth_feature_dim,
                        device=self.device,
                        dtype=policy.dtype,
                    )
                if bool(self.cfg.use_object_adapt):
                    self.object_adapt_ema(td)
                    self.object_pred_transform(td)
                self.adapt_ema(td)
                if ("next", "adapt_hx") not in td.keys(True, True):
                    raise RuntimeError(
                        "Teacher cache requires the locked recurrent adaptation EMA"
                    )
                adapt_state.index_copy_(
                    0, positions, td["next", "adapt_hx"][:, -1]
                )
                actor_parts = []
                for key, width in zip(self.q_actor_keys, self._q_actor_widths):
                    if key not in td.keys(True, True):
                        raise KeyError(
                            f"Teacher Actor cache is missing input {key!r}"
                        )
                    value = td[key]
                    if int(value.shape[-1]) != int(width):
                        raise ValueError(
                            f"Teacher Actor cache input {key!r} has width "
                            f"{int(value.shape[-1])}; expected {int(width)}"
                        )
                    actor_parts.append(value)
                actor = torch.cat(actor_parts, dim=-1)
                if actor.shape != (*td.batch_size, self._q_actor_dim):
                    raise RuntimeError("Teacher Actor cache has an invalid shape")
                valid_device = chunk.valid.to(self.device)
                node_indices = chunk.flat_node_indices[chunk.valid].to(self.device)
                actor_by_node.index_copy_(
                    0,
                    node_indices,
                    actor[valid_device].float(),
                )
                write_counts.index_add_(
                    0,
                    node_indices,
                    torch.ones_like(node_indices, dtype=write_counts.dtype),
                )

        if not bool((write_counts == 1).all()):
            raise RuntimeError(
                "Teacher Actor cache must materialize every node exactly once"
            )
        self._teacher_actor_cache.publish(lineage, store, actor_by_node)

    @torch.no_grad()
    def _ensure_teacher_actor_cache_current(self) -> None:
        """Refresh one device-resident Teacher cache per EMA lineage."""
        if not self._teacher_episode_cache_enabled():
            return
        self._ensure_teacher_episode_cache_state()
        lineage = self._teacher_actor_cache_lineage()
        if (
            self._teacher_ring_cache_lineage == lineage
            and self._teacher_actor_cache.lineage == lineage
            and self._teacher_actor_cache.ready
        ):
            return
        if (
            self._teacher_actor_cache.lineage != lineage
            or not self._teacher_actor_cache.ready
        ):
            self._rebuild_teacher_actor_cache(lineage)

        size = int(self.q_teacher_replay.size)
        if size < 1:
            raise RuntimeError("Teacher Actor cache cannot populate an empty FIFO")
        for key in (TEACHER_EPISODE_UID_KEY, TEACHER_EPISODE_STEP_KEY):
            if key not in self.q_teacher_replay.data:
                raise KeyError(f"Teacher FIFO lacks cache lineage key {key!r}")
        self._teacher_ring_cache_lineage = lineage

    @torch.no_grad()
    def _teacher_actor_observations_for_indices(
        self,
        physical_indices: torch.Tensor,
        *,
        next_state: bool,
    ) -> torch.Tensor:
        """Gather current-lineage Teacher Actor inputs for sampled FIFO rows."""
        self._ensure_teacher_actor_cache_current()
        lineage = self._teacher_actor_cache_lineage()
        if self._teacher_ring_cache_lineage != lineage:
            raise RuntimeError("Teacher FIFO Actor cache lineage is stale")
        indices = physical_indices.reshape(-1).to(
            device=self.q_teacher_replay.device, dtype=torch.long
        )
        if bool((indices < 0).any()) or bool(
            (indices >= int(self.q_teacher_replay.size)).any()
        ):
            raise IndexError("Teacher Actor cache sample index is outside the FIFO")
        episode_uids = self.q_teacher_replay.data[
            TEACHER_EPISODE_UID_KEY
        ].index_select(0, indices)
        episode_steps = self.q_teacher_replay.data[
            TEACHER_EPISODE_STEP_KEY
        ].index_select(0, indices)
        return self._teacher_actor_cache.gather(
            self._teacher_episode_store,
            episode_uids,
            episode_steps,
            lineage=lineage,
            next_state=next_state,
            output_device=self.device,
        )

    @torch.no_grad()
    def _prepare_raw_final_state(
        self,
        td: TensorDict,
        *,
        collection_actor_observations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result = {
            **self._raw_perception_values(td),
            "next_critic_observations": self._cat_replay_sources(
                td, self.q_critic_keys
            ).clone(),
            NEXT_REFERENCE_PHASE_KEY: self._reference_phase(td).clone(),
        }
        if self._student_collection_actor_cache_enabled():
            if collection_actor_observations is None:
                collection_actor_observations = (
                    self._materialize_collection_actor_observations(td)
                )
            expected_shape = (*td.batch_size, self._q_actor_dim)
            if collection_actor_observations.shape != expected_shape:
                raise ValueError(
                    "Student final-state collection Actor cache is batch-misaligned"
                )
            if (
                not collection_actor_observations.is_floating_point()
                or not torch.isfinite(collection_actor_observations).all()
            ):
                raise ValueError(
                    "Student final-state collection Actor cache contains NaN/Inf"
                )
            with torch.inference_mode(False):
                result[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY] = (
                    collection_actor_observations.detach().clone()
                )
        return result

    @torch.no_grad()
    def capture_truncation_final_observations(self, td: TensorDict, step: int):
        """Capture the true pre-reset timeout input, never the reset carry."""
        if not self._collect_dagger_replay_this_rollout():
            self._truncation_final_batches = []
            self._last_truncation_finals_used = 0
            return
        capture_mask = _vaic_pre_reset_final_observation_mask(td).reshape(-1).bool()
        if not capture_mask.any():
            return
        indices = capture_mask.nonzero(as_tuple=False).squeeze(-1)
        # Run the perception stack once at its normal full-environment batch
        # geometry, then retain only true timeout rows.  Subset execution can
        # select a different compiled kernel (and introduce numerical jitter)
        # from the live collector path.
        full_next = td["next"].clone()
        full_collection_actor = None
        if self._student_collection_actor_cache_enabled():
            full_collection_actor = self._materialize_collection_actor_observations(
                full_next
            )
        values = self._prepare_raw_final_state(
            td["next"][indices].clone(),
            collection_actor_observations=(
                None
                if full_collection_actor is None
                else full_collection_actor.index_select(0, indices)
            ),
        )
        values["indices"] = indices * int(self.cfg.train_every) + int(step)
        self._truncation_final_batches.append(values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        if not self._collect_dagger_replay_this_rollout():
            self._rollout_final_batch = None
            self._truncation_final_batches = []
            self._last_truncation_finals_used = 0
            return
        self._rollout_final_batch = self._prepare_raw_final_state(carry.clone())

    def _dagger_transition_chunks(self, td: TensorDict):
        """Build self-contained raw-input windows for current-EMA re-encoding.

        Each row contains eight preceding inputs, the current input, and the
        true next input.  GRU hidden states are initialized to zero at the
        window boundary.  Reconstruction is exact when an episode reset lies
        in the window and is an explicit finite-burn-in approximation otherwise.
        """
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "TD3 raw replay requires capture_rollout_final_observation(carry)"
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError("rollout length does not match train_every")
        actuator_contexts = self._consume_rollout_q_actuator_contexts(int(t))
        burn_in = int(self.cfg.perception_replay_burn_in)
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        collection_cache_enabled = self._student_collection_actor_cache_enabled()
        raw_current = (
            None if collection_cache_enabled else self._raw_perception_values(td)
        )
        collection_actor = None
        if collection_cache_enabled:
            if STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY not in td.keys(True, True):
                raise KeyError(
                    "FastSAC Student rollout is missing its collection Actor cache"
                )
            if STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY not in final_batch:
                raise KeyError(
                    "FastSAC rollout-final state is missing its collection Actor cache"
                )
            collection_actor = td[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY]
            expected_shape = (int(n), int(t), self._q_actor_dim)
            if collection_actor.shape != expected_shape:
                raise ValueError(
                    "FastSAC Student rollout collection Actor cache is misaligned"
                )
            final_actor = final_batch[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY]
            if final_actor.shape != (int(n), self._q_actor_dim):
                raise ValueError(
                    "FastSAC rollout-final collection Actor cache is misaligned"
                )

        history_count = int(self._perception_replay_history_count)
        history = {}
        sequence = {}
        if not collection_cache_enabled:
            if raw_current is None:  # pragma: no cover - branch invariant
                raise RuntimeError("legacy replay lost its raw perception inputs")
            if self._perception_replay_history is None:
                history = {key: value[:, :0] for key, value in raw_current.items()}
            else:
                history = self._perception_replay_history
                if any(value.shape[0] != int(n) for value in history.values()):
                    raise RuntimeError("raw perception replay environment count changed")

            for key, current_value in raw_current.items():
                final_value = final_batch[key].reshape(
                    int(n), *current_value.shape[2:]
                )
                sequence[key] = torch.cat(
                    (history[key], current_value, final_value.unsqueeze(1)), dim=1
                )

        truncation_batches = self._truncation_final_batches
        self._truncation_final_batches = []
        truncation_finals = None
        if truncation_batches:
            truncation_finals = {
                key: torch.cat([batch[key] for batch in truncation_batches], 0)
                for key in truncation_batches[0]
            }
            flat_indices = truncation_finals["indices"].long()
            if (flat_indices < 0).any() or (flat_indices >= int(n * t)).any():
                raise IndexError("captured truncation index is outside rollout")
        expected_final_indices = (
            _vaic_pre_reset_final_observation_mask(td)
            .reshape(int(n * t))
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        actual_final_indices = (
            expected_final_indices[:0]
            if truncation_finals is None
            else truncation_finals["indices"].to(
                device=expected_final_indices.device, dtype=torch.long
            )
        )
        if (
            actual_final_indices.numel() != expected_final_indices.numel()
            or not torch.equal(
                actual_final_indices.sort().values,
                expected_final_indices.sort().values,
            )
        ):
            raise RuntimeError(
                "captured pre-reset final observations do not exactly match "
                "the rollout's pure time-limit rows"
            )
        used_truncation_finals = 0

        for step in range(int(t)):
            current = td[:, step]
            position = history_count + step
            if position < burn_in:
                continue
            window_values = (
                {}
                if collection_cache_enabled
                else {
                    key: value[:, position - burn_in : position + 2]
                    for key, value in sequence.items()
                }
            )
            if not collection_cache_enabled and any(
                value.shape[1] != burn_in + 2
                for value in window_values.values()
            ):
                raise RuntimeError("raw perception replay window has invalid length")

            current_critic = self._cat_replay_sources(
                current, self.q_critic_keys
            ).reshape(int(n), self._q_critic_dim)
            truncations = _vaic_truncation_mask(current).reshape(int(n)).bool()
            if step + 1 < int(t):
                next_critic = self._cat_replay_sources(
                    td[:, step + 1], self.q_critic_keys
                ).reshape(int(n), self._q_critic_dim)
                next_reference_phase = self._reference_phase(
                    td[:, step + 1]
                ).reshape(int(n))
            else:
                next_critic = final_batch["next_critic_observations"].reshape(
                    int(n), self._q_critic_dim
                )
                next_reference_phase = final_batch[
                    NEXT_REFERENCE_PHASE_KEY
                ].reshape(int(n))
            if collection_cache_enabled:
                current_actor = collection_actor[:, step].reshape(
                    int(n), self._q_actor_dim
                )
                next_actor = (
                    collection_actor[:, step + 1]
                    if step + 1 < int(t)
                    else final_batch[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY]
                ).reshape(int(n), self._q_actor_dim)

            if truncation_finals is not None:
                flat_indices = truncation_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    next_critic = next_critic.clone()
                    next_critic.index_copy_(
                        0,
                        env_indices,
                        truncation_finals["next_critic_observations"][selected],
                    )
                    next_reference_phase = next_reference_phase.clone()
                    next_reference_phase.index_copy_(
                        0,
                        env_indices,
                        truncation_finals[NEXT_REFERENCE_PHASE_KEY][selected]
                        .reshape(-1),
                    )
                    if not collection_cache_enabled:
                        for key in _PERCEPTION_REPLAY_FIELDS:
                            window_values[key] = window_values[key].clone()
                            window_values[key][env_indices, -1] = truncation_finals[
                                key
                            ][selected]
                    if collection_cache_enabled:
                        if (
                            STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY
                            not in truncation_finals
                        ):
                            raise KeyError(
                                "pure-timeout final is missing its carried-hidden "
                                "Student Actor cache"
                            )
                        next_actor = next_actor.clone()
                        next_actor.index_copy_(
                            0,
                            env_indices,
                            truncation_finals[
                                STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY
                            ][selected],
                        )

            motion_id = current.get(REPLAY_MOTION_ID_KEY, None)
            if motion_id is None:
                dataset = getattr(
                    getattr(getattr(self, "env", None), "command_manager", None),
                    "dataset",
                    None,
                )
                num_motions = len(getattr(dataset, "lengths", ()))
                if num_motions > 1:
                    raise RuntimeError(
                        "multi-motion replay requires rollout-time replay_motion_id"
                    )
                motion_id = torch.zeros(int(n), dtype=torch.long, device=current.device)

            transitions = {
                "critic_observations": current_critic,
                "actions": current[ACTION_KEY].reshape(int(n), self.action_dim),
                DAGGER_REPLAY_TEACHER_ACTIONS: current[
                    DAGGER_TEACHER_ACTION_KEY
                ].reshape(int(n), self.action_dim),
                DAGGER_TEACHER_ACTION_VALID_KEY: current[
                    DAGGER_TEACHER_ACTION_VALID_KEY
                ]
                .reshape(int(n))
                .bool(),
                DAGGER_IS_STUDENT_ACTION_KEY: current[DAGGER_IS_STUDENT_ACTION_KEY]
                .reshape(int(n))
                .bool(),
                "rewards": self._scalarize_q_reward(current[REWARD_KEY]).reshape(
                    int(n)
                ),
                "dones": current[DONE_KEY].reshape(int(n)).bool(),
                "truncations": truncations,
                "discounts": current["next", "discount"].reshape(int(n)),
                "next_critic_observations": next_critic,
                REFERENCE_PHASE_KEY: self._reference_phase(current).reshape(int(n)),
                NEXT_REFERENCE_PHASE_KEY: next_reference_phase,
                REPLAY_TERMINATED_KEY: current[TERM_KEY].reshape(int(n)).bool(),
                REPLAY_COMMAND_FINISHED_KEY: current[
                    "next", "stats", "command_finished"
                ]
                .reshape(int(n))
                .bool(),
                REPLAY_TIME_LIMIT_KEY: current["next", "stats", "episode_time_limit"]
                .reshape(int(n))
                .bool(),
                REPLAY_MOTION_ID_KEY: motion_id.reshape(int(n)).long(),
                REPLAY_PERCEPTION_EMA_GENERATION_KEY: torch.full(
                    (int(n),),
                    int(getattr(self, "_perception_ema_generation", 0)),
                    dtype=torch.long,
                    device=current.device,
                ),
                _PREFILL_ENV_INDEX_KEY: torch.arange(
                    int(n), device=current.device, dtype=torch.long
                ),
                _PREFILL_STEP_INDEX_KEY: torch.full(
                    (int(n),), step, device=current.device, dtype=torch.long
                ),
                _PREFILL_TERMINATED_KEY: current[TERM_KEY].reshape(int(n)).bool(),
                _PREFILL_COMMAND_FINISHED_KEY: current[
                    "next", "stats", "command_finished"
                ]
                .reshape(int(n))
                .bool(),
                **window_values,
                TD3_NOISE_FREE_STUDENT_ACTION_KEY: current[
                    TD3_NOISE_FREE_STUDENT_ACTION_KEY
                ].reshape(int(n), self.action_dim),
                TD3_EXPLORATORY_STUDENT_ACTION_KEY: current[
                    TD3_EXPLORATORY_STUDENT_ACTION_KEY
                ].reshape(int(n), self.action_dim),
                TD3_COLLECTOR_NOISE_KEY: current[TD3_COLLECTOR_NOISE_KEY].reshape(
                    int(n), self.action_dim
                ),
                TD3_BETA_KEY: current[TD3_BETA_KEY].reshape(int(n)),
            }
            if actuator_contexts is not None:
                actuator_context = self._transition_q_actuator_context(
                    actuator_contexts[:, step], int(n), current.device
                )
                transitions[Q_ACTUATOR_CONTEXT_KEY] = actuator_context
                # Delay/alpha are episode-constant. Predicted-effect context
                # additionally advances u_(t-1) to this transition's issued
                # action, including a pre-reset timeout bootstrap successor.
                transitions[NEXT_Q_ACTUATOR_CONTEXT_KEY] = (
                    self._next_q_actuator_context(
                        actuator_context, transitions["actions"]
                    )
                )
            if collection_cache_enabled:
                transitions[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY] = current_actor
                transitions[STUDENT_COLLECTION_NEXT_ACTOR_OBSERVATIONS_KEY] = (
                    next_actor
                )
            transitions, valid = _filter_replay_rows(
                current, transitions, DAGGER_REPLAY_MIN_STEP_COUNT
            )
            if truncation_finals is not None:
                flat_indices = truncation_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    used_truncation_finals += int(valid[env_indices].sum().item())
            yield transitions

        if collection_cache_enabled:
            self._perception_replay_history = None
        else:
            if raw_current is None:  # pragma: no cover - branch invariant
                raise RuntimeError("legacy replay lost its raw perception inputs")
            combined_history = {
                key: torch.cat((history[key], value), dim=1)[:, -burn_in:].detach()
                for key, value in raw_current.items()
            }
            self._perception_replay_history = combined_history
        self._perception_replay_history_count = min(burn_in, history_count + int(t))
        self._last_truncation_finals_used = used_truncation_finals

    @torch.no_grad()
    def _reencode_perception_windows(
        self,
        batch: dict[str, torch.Tensor],
        *,
        include_current: bool,
        include_next: bool,
    ) -> dict[str, torch.Tensor]:
        """Rebuild Actor observations with the current EMA perception weights."""
        if not include_current and not include_next:
            return {}
        missing = set(_PERCEPTION_REPLAY_FIELDS).difference(batch)
        if missing:
            raise KeyError(
                f"raw perception replay fields are missing: {sorted(missing)}"
            )

        depth_u8 = batch[PERCEPTION_DEPTH_U8_KEY]
        policy_raw = batch[PERCEPTION_POLICY_RAW_KEY]
        vel_raw = batch[PERCEPTION_VEL_COMMAND_RAW_KEY]
        geometry_ids = batch[PERCEPTION_OBJECT_GEO_ID_KEY]
        is_init = batch[PERCEPTION_IS_INIT_KEY]
        row_count, window_length = depth_u8.shape[:2]
        expected_length = int(self.cfg.perception_replay_burn_in) + 2
        if int(window_length) != expected_length:
            raise ValueError(
                f"perception replay window has length {window_length}; "
                f"expected {expected_length}"
            )
        if policy_raw.shape[:2] != (row_count, window_length) or vel_raw.shape[:2] != (
            row_count,
            window_length,
        ):
            raise ValueError("raw perception replay fields have inconsistent windows")
        if geometry_ids.shape != (row_count, window_length):
            raise ValueError("object geometry replay IDs have inconsistent windows")

        snapshot = self._vecnorm_snapshot()
        current_chunks: list[torch.Tensor] = []
        next_chunks: list[torch.Tensor] = []
        microbatch = int(self.cfg.perception_encode_microbatch_size)
        with set_recurrent_mode(True):
            for start in range(0, int(row_count), microbatch):
                stop = min(start + microbatch, int(row_count))
                depth = self._normalize_replay_value(
                    DEPTH_KEY,
                    _decode_replay_depth_u8(depth_u8[start:stop]),
                    snapshot,
                )
                policy = self._normalize_replay_value(
                    OBS_KEY, policy_raw[start:stop], snapshot
                )
                vel = self._normalize_replay_value(
                    VEL_CMD_KEY, vel_raw[start:stop], snapshot
                )
                geometry = self._decode_replay_object_geo(
                    geometry_ids[start:stop],
                    device=depth_u8.device,
                    dtype=policy.dtype,
                )
                count = stop - start
                td = TensorDict(
                    {
                        DEPTH_KEY: depth,
                        OBS_KEY: policy,
                        VEL_CMD_KEY: vel,
                        OBJECT_GEO_KEY: geometry,
                        "is_init": is_init[start:stop],
                        "depth_hx": torch.zeros(
                            count,
                            window_length,
                            self.depth_feature_dim,
                            device=depth.device,
                            dtype=policy.dtype,
                        ),
                        "adapt_hx": torch.zeros(
                            count,
                            window_length,
                            int(self.cfg.latent_dim),
                            device=depth.device,
                            dtype=policy.dtype,
                        ),
                    },
                    batch_size=(count, window_length),
                    device=depth.device,
                )
                self.temporal_depth_gru_ema(td)
                self.object_adapt_ema(td)
                self.object_pred_transform(td)
                self.adapt_ema(td)
                priv_pred = td[PRIV_PRED_KEY]
                if include_current:
                    current_chunks.append(
                        torch.cat((vel[:, -2], policy[:, -2], priv_pred[:, -2]), -1)
                    )
                if include_next:
                    next_chunks.append(
                        torch.cat((vel[:, -1], policy[:, -1], priv_pred[:, -1]), -1)
                    )

        result = {}
        if include_current:
            result["observations"] = torch.cat(current_chunks, 0)
        if include_next:
            result["next_observations"] = torch.cat(next_chunks, 0)
        return result

    def _prepare_dagger_learning_batch(self, batch):
        """Normalize critic inputs and materialize only needed Actor states.

        FastSAC/TVKD batches already contain source-specific flat Actor inputs:
        collection-time carried-hidden values for Student rows and reset-exact
        current-EMA values from the successful-episode cache for Teacher rows.
        Baseline TD3 never enables this cache and follows its raw-window path.
        """
        prepared = PPOBCDaggerFinetune._prepare_dagger_learning_batch(self, batch)
        include_current = DAGGER_REPLAY_TEACHER_ACTIONS in batch
        include_next = "next_critic_observations" in batch
        if not self._student_collection_actor_cache_enabled():
            encoded = self._reencode_perception_windows(
                prepared,
                include_current=include_current,
                include_next=include_next,
            )
            prepared.update(encoded)
            return prepared
        requested = tuple(
            item
            for item in (
                (
                    "observations",
                    REPLAY_ACTOR_OBSERVATIONS_KEY,
                    include_current,
                ),
                (
                    "next_observations",
                    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
                    include_next,
                ),
            )
            if item[2]
        )
        for output_key, cache_key, _ in requested:
            cache = batch.get(cache_key, None)
            if cache is None:
                raise KeyError(
                    f"collection-exact replay is missing Actor cache {cache_key!r}"
                )
            if cache.ndim != 2 or cache.shape[-1] != self._q_actor_dim:
                raise ValueError(
                    f"collection-exact Actor cache {cache_key!r} is misaligned"
                )
            if not cache.is_floating_point() or not torch.isfinite(cache).all():
                raise ValueError(
                    f"collection-exact Actor cache {cache_key!r} is invalid"
                )
            prepared[output_key] = cache
        return prepared

    @torch.no_grad()
    def _smoothed_target_q_action(
        self, next_actor_observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_next_mean = self._actor_target_dist_from_flat(next_actor_observations).mean
        if not torch.isfinite(raw_next_mean).all():
            raise RuntimeError("TD3 target Actor produced non-finite raw actions")
        next_action = self._bounded_actor_mean(raw_next_mean)
        next_q = self._q_action_input(next_action)
        std = float(self.cfg.target_policy_noise_std)
        clip = float(self.cfg.target_policy_noise_clip)
        if std == 0.0:
            sampled_noise = torch.zeros_like(next_q)
        else:
            sampled_noise = (
                torch.randn(
                    next_q.shape,
                    device=next_q.device,
                    dtype=next_q.dtype,
                    generator=self.target_policy_rng,
                )
                * std
            )
            sampled_noise.clamp_(-clip, clip)
        q_low, q_high = self._q_execution_bounds(next_q)
        smoothed_q = torch.maximum(torch.minimum(next_q + sampled_noise, q_high), q_low)
        # Round-trip through the physical support so target Q receives exactly
        # the same finite action variable used by behavior and replay.
        smoothed_q = self._q_action_input(self._q_action_to_physical(smoothed_q))
        applied_noise = smoothed_q - next_q
        return smoothed_q, applied_noise, next_action

    @torch.no_grad()
    def _distributional_td3_target(self, batch: dict[str, torch.Tensor]):
        smoothed_q_action, target_noise, next_action = self._smoothed_target_q_action(
            batch["next_observations"]
        )
        target_logits = self.qnet_target(
            batch["next_critic_observations"],
            self._q_action_features_from_q_input(
                smoothed_q_action,
                batch.get(NEXT_Q_ACTUATOR_CONTEXT_KEY),
            ),
        )
        selected_probability, target_expected_heads, selected_head = (
            _select_lower_expected_c51_distribution(
                target_logits, self.qnet_target.support
            )
        )
        bootstrap = _bootstrap_mask(batch["dones"], batch["truncations"])
        effective_discount = float(self.cfg.gamma) * batch["discounts"]
        projected, left_fraction, right_fraction = _project_c51_probabilities(
            selected_probability,
            batch["rewards"],
            bootstrap,
            effective_discount,
            self.qnet_target.support,
        )
        projected = projected.detach()
        selected_probability = selected_probability.detach()
        selected_expected = target_expected_heads.gather(
            0, selected_head.unsqueeze(0)
        ).squeeze(0)
        entropy = -(
            selected_probability
            * selected_probability.clamp_min(
                torch.finfo(selected_probability.dtype).tiny
            ).log()
        ).sum(dim=-1)
        diagnostics = {
            "target_expected_q1_mean": target_expected_heads[0].mean(),
            "target_expected_q2_mean": target_expected_heads[1].mean(),
            "projected_target_mean": (projected * self.qnet_target.support)
            .sum(dim=-1)
            .mean(),
            "selected_target_expected_mean": selected_expected.mean(),
            "target_distribution_entropy": entropy.mean(),
            "target_select_q1_fraction": (selected_head == 0).float().mean(),
            "target_select_q2_fraction": (selected_head == 1).float().mean(),
            "left_support_projection_clipping_fraction": left_fraction,
            "right_support_projection_clipping_fraction": right_fraction,
            "target_smoothing_noise_norm": target_noise.norm(dim=-1).mean(),
            "target_noise_free_action_abs_mean": next_action.abs().mean(),
        }
        return projected, diagnostics

    def _critic_update(self, batch: dict[str, torch.Tensor]):
        """One independent twin-Q C51 update against a common detached target."""
        projected_target, target_metrics = self._distributional_td3_target(batch)
        logits = self.qnet(
            batch["critic_observations"],
            self._q_action_features(
                batch["actions"], batch.get(Q_ACTUATOR_CONTEXT_KEY)
            ),
        )
        log_probabilities = F.log_softmax(logits, dim=-1)
        per_head = (
            -(projected_target.unsqueeze(0) * log_probabilities)
            .sum(dim=-1)
            .mean(dim=-1)
        )
        critic_loss = per_head.sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = _measure_or_clip_grad_norm(
            self.qnet.parameters(), float(self.cfg.q_max_grad_norm)
        )
        self.critic_optimizer.step()
        self.critic_update_count += 1
        with torch.no_grad():
            expected_heads = (log_probabilities.detach().exp() * self.qnet.support).sum(
                dim=-1
            )
        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_loss_1": per_head[0].detach(),
            "critic_loss_2": per_head[1].detach(),
            "critic_grad_norm": critic_grad.detach(),
            "expected_q1_mean": expected_heads[0].mean(),
            "expected_q2_mean": expected_heads[1].mean(),
            "twin_expected_q_disagreement": (expected_heads[0] - expected_heads[1])
            .abs()
            .mean(),
            **target_metrics,
        }
        return metrics

    def _actor_update(self, batch: dict[str, torch.Tensor]):
        """Apply one backward/step to the weighted TD3 plus exact BC loss."""
        raw_prediction = self._actor_dist_from_flat(batch["observations"]).mean
        if not torch.isfinite(raw_prediction).all():
            raise RuntimeError("TD3 Actor produced non-finite raw actions")
        prediction_action = self._bounded_actor_mean(raw_prediction)
        q_action = self._q_action_features(
            prediction_action, batch.get(Q_ACTUATOR_CONTEXT_KEY)
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        td3_actor_loss, expected_q1 = _td3_actor_q1_loss(
            self.qnet, batch["critic_observations"], q_action
        )
        exact_bc_loss = _exact_teacher_bc_loss(
            prediction_action,
            batch[DAGGER_REPLAY_TEACHER_ACTIONS],
            batch[DAGGER_TEACHER_ACTION_VALID_KEY],
            self._fastsac_q_action_center,
            self._fastsac_q_action_scale,
            float(self.cfg.dagger_actor_huber_delta),
        )
        weighted_td3 = float(self.cfg.eta_td3) * td3_actor_loss
        weighted_bc = float(self.cfg.lambda_bc) * exact_bc_loss
        total_actor_loss = weighted_td3 + weighted_bc
        total_actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(
            self.actor_adapt.parameters(), float(self.cfg.max_grad_norm)
        )
        if any(parameter.grad is not None for parameter in self.qnet.parameters()):
            raise RuntimeError("Critic parameters accumulated Actor-step gradients")
        self.actor_optimizer.step()
        self.actor_update_count += 1
        teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
        actor_teacher_replay_fraction = (
            prediction_action.new_zeros(())
            if teacher_source is None
            else teacher_source.float().mean()
        )
        return {
            "td3_actor_loss": td3_actor_loss.detach(),
            "exact_bc_loss": exact_bc_loss.detach(),
            "weighted_td3_actor_loss": weighted_td3.detach(),
            "weighted_bc_loss": weighted_bc.detach(),
            "total_actor_loss": total_actor_loss.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "actor_expected_q1_mean": expected_q1.detach().mean(),
            "actor_teacher_replay_fraction": (actor_teacher_replay_fraction.detach()),
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
        }

    def _maybe_delayed_actor_and_targets(self, actor_batch: dict[str, torch.Tensor]):
        if self.critic_update_count % int(self.cfg.policy_delay):
            return None
        if self.actor_target is None:
            raise RuntimeError("delayed update requires a loaded actor_target")
        actor_metrics = self._actor_update(actor_batch)
        _polyak_update_(self.actor_target, self.actor_adapt, float(self.cfg.q_tau))
        _polyak_update_(self.qnet_target, self.qnet, float(self.cfg.q_tau))
        self.actor_target.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        return actor_metrics

    def _q_action_feature_semantics(self) -> str:
        if self._q_uses_predicted_effect():
            return Q_PREDICTED_EFFECT_ACTION_FEATURE_SEMANTICS
        if self._q_conditions_on_actuator_state():
            return Q_ACTUATOR_ACTION_FEATURE_SEMANTICS
        return "normalized_issued_command_only_v1"

    def _q_predicted_effect_metadata(self) -> dict:
        if not self._q_uses_predicted_effect():
            return {"enabled": False}
        return {
            "enabled": True,
            "intervals": Q_PREDICTED_EFFECT_INTERVALS,
            "aggregation": "mean_post_lerp_applied_action_per_control_interval",
            "reference": "hold_previous_issued_command",
            "action_gradient": "candidate_recomputed_not_replay_detached",
        }

    def _q_residual_film_metadata(self) -> dict:
        if not self._q_uses_residual_film():
            return {"enabled": False}
        scale = float(self.cfg.q_residual_film_scale)
        return {
            "enabled": True,
            "semantics": FASTSAC_Q_RESIDUAL_FILM_SEMANTICS,
            "condition": "delay_one_hot_plus_centered_actuator_alpha_detached",
            "condition_dim": int(self._q_actuator_parameter_context_dim),
            "insertion": "post_action_stem_pre_state_action_fusion",
            "gain_scale": scale,
            "shift_scale": 0.5 * scale,
            "initialization": "exact_identity_zero_weight_and_bias",
        }

    def _q_backend_metadata(self):
        q_fractions = self._replay_mix_fractions("q")
        actor_fractions = self._replay_mix_fractions("actor")
        perception_fractions = self._replay_mix_fractions("perception")
        q_counts = self._replay_source_counts("q", int(self.cfg.q_batch_size))
        q_student_rows = q_counts["uniform_student"] + q_counts["failure_student"]
        q_teacher_rows = q_counts["uniform_teacher"] + q_counts["failure_teacher"]
        q_teacher_fraction = (
            q_fractions["uniform_teacher"] + q_fractions["failure_teacher"]
        )
        legacy_half_mix = (
            not self._has_canonical_replay_mix()
            and int(self.cfg.q_batch_size) % 2 == 0
            and math.isclose(q_teacher_fraction, 0.5, rel_tol=0.0, abs_tol=1e-12)
        )
        return {
            "actor_obs_keys": list(self.q_actor_keys),
            "critic_obs_keys": list(self.q_critic_keys),
            "actor_obs_dim": self._q_actor_dim,
            "critic_obs_dim": self._q_critic_dim,
            "action_dim": self.action_dim,
            "q_action_input_dim": self._q_action_input_dim,
            "q_actuator_context": copy.deepcopy(
                self._q_actuator_context_metadata_value
            ),
            "q_action_feature_semantics": self._q_action_feature_semantics(),
            "q_predicted_effect": self._q_predicted_effect_metadata(),
            "q_residual_film": self._q_residual_film_metadata(),
            "hidden_dim": int(self.cfg.q_hidden_dim),
            "q_action_fusion": str(self.cfg.q_action_fusion),
            "q_state_hidden_dim": _q_state_hidden_dim(
                self.cfg.q_hidden_dim, self.cfg.q_action_fusion
            ),
            "q_action_hidden_dim": _q_action_hidden_dim(
                self.cfg.q_hidden_dim, self.cfg.q_action_fusion
            ),
            "q_action_fusion_semantics": (
                FASTSAC_Q_BALANCED_FUSION_SEMANTICS
                if str(self.cfg.q_action_fusion) == "balanced"
                else FASTSAC_Q_LATE_FUSION_SEMANTICS
            ),
            "q_architecture_semantics": FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
            "num_atoms": int(self.cfg.q_num_atoms),
            "v_min": float(self.cfg.q_v_min),
            "v_max": float(self.cfg.q_v_max),
            "layer_norm": bool(self.cfg.q_layer_norm),
            "gamma": float(self.cfg.gamma),
            "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
            "reward_scalarization": "sum_existing_reward_groups_v1",
            "reward_groups": list(self.reward_groups),
            "truncation_semantics": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "q_action_coordinates": str(self.cfg.q_action_coordinates),
            "q_action_normalized": bool(self.cfg.q_normalize_actions),
            "q_action_center": self._fastsac_q_action_center.detach().cpu().tolist(),
            "q_action_scale": self._fastsac_q_action_scale.detach().cpu().tolist(),
            "q_action_joint_names": list(self.joint_names),
            "action_support_low": self._fastsac_action_low.detach().cpu().tolist(),
            "action_support_high": self._fastsac_action_high.detach().cpu().tolist(),
            "q_action_support_projection": "physical_before_affine",
            "q_action_transform_fingerprint": self._fastsac_action_contract[
                "q_action_transform_fingerprint"
            ],
            "q_action_input_gain": float(self.cfg.q_action_input_gain),
            "clipped_double_distribution": True,
            "target_semantics": CRITIC_SEMANTICS,
            "actor_q_reduction": "online_q1_expectation_only",
            "replay_mix_semantics": (
                "capacity_filled_successful_prefill_teacher_0.5_student_executed_0.5_v3"
                if legacy_half_mix
                else (
                    "capacity_filled_successful_prefill_configured_teacher_"
                    "student_executed_v4"
                )
            ),
            "q_teacher_replay_ratio": q_teacher_fraction,
            "q_teacher_rows_per_batch": q_teacher_rows,
            "q_student_rows_per_batch": q_student_rows,
            "q_realized_teacher_fraction": (
                q_teacher_rows / int(self.cfg.q_batch_size)
            ),
            "actor_teacher_replay_fraction": (
                actor_fractions["uniform_teacher"] + actor_fractions["failure_teacher"]
            ),
            "q_replay_source_fractions": q_fractions,
            "actor_replay_source_fractions": actor_fractions,
            "perception_replay_source_fractions": perception_fractions,
            "perception_replay_mode": str(
                getattr(self.cfg, "perception_replay_mode", "legacy_online_student")
            ),
            "perception_training_semantics": (
                ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS
                if str(
                    getattr(
                        self.cfg,
                        "perception_replay_mode",
                        "legacy_online_student",
                    )
                )
                == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                else PERCEPTION_REPLAY_SEMANTICS
            ),
            "actor_replay_mix_semantics": (
                "combined_rl_bc_on_prefill_teacher_and_student_executed_rows_v2"
            ),
            "failure_phase_replay_semantics": FAILURE_PHASE_REPLAY_SEMANTICS,
            "failure_phase_teacher_fraction": float(
                self.cfg.failure_phase_teacher_fraction
            ),
            "failure_phase_lookback_steps": int(self.cfg.failure_phase_lookback_steps),
            "failure_phase_samples_per_failure": int(
                self.cfg.failure_phase_samples_per_failure
            ),
            "failure_phase_num_bins": int(self.cfg.failure_phase_num_bins),
        }

    def _checkpoint_config(self):
        names = (
            "dagger_control_mode",
            "dagger_safe_takeover_rms",
            "dagger_safe_release_rms",
            "dagger_safe_min_teacher_steps",
            "dagger_safe_zero_iteration",
            "dagger_beta_start",
            "dagger_beta_end",
            "dagger_beta_decay_rollouts",
            "dagger_seed",
            "action_support_clip",
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "dagger_buffer_capacity",
            "dagger_buffer_device",
            "dagger_batch_size",
            "teacher_actor_replay_fraction",
            "teacher_perception_replay_fraction",
            "failure_phase_teacher_fraction",
            "failure_phase_lookback_steps",
            "failure_phase_samples_per_failure",
            "failure_phase_num_bins",
            "teacher_prefill_max_rollouts",
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
            "q_condition_on_actuator_state",
            "q_use_predicted_effect",
            "q_use_residual_film",
            "q_residual_film_scale",
            "q_lr",
            "q_weight_decay",
            "q_seed",
            "q_tau",
            "q_max_grad_norm",
            "q_batch_size",
            "q_updates_per_rollout",
            "q_update_to_data_ratio",
            "q_teacher_replay_ratio",
            "q_teacher_buffer_capacity",
            "perception_replay_burn_in",
            "perception_encode_microbatch_size",
            "teacher_perception_batch_size",
            "perception_replay_mode",
            "perception_replay_batch_size",
            "perception_staleness_probe_num_envs",
            "perception_staleness_probe_max_episodes",
            "perception_staleness_probe_max_generation_age",
            "perception_staleness_probe_interval",
            "teacher_perception_warmup_steps",
            "perception_depth_codec",
            "load_pretrained_perception",
            "perception_checkpoint_path",
            "train_perception",
            "save_teacher_buffer",
        )
        result = {
            **{name: getattr(self.cfg, name) for name in names},
            "method": TRAINING_ALGORITHM,
            "actor_output": "ppo_physical_proposal_tanh_bounded_raw_joint_command",
            "bc_loss": "joint_normalized_raw_mean_teacher_smooth_l1",
        }
        # Canonical v4 fields are intentionally flat so Hydra structured config,
        # checkpoint diffing, and downstream audit tooling all see the same keys.
        for purpose in ("q", "actor", "perception"):
            for source in REPLAY_SOURCE_ORDER:
                name = f"{purpose}_{source}_fraction"
                if hasattr(self.cfg, name):
                    result[name] = getattr(self.cfg, name)
        return result

    def _sample_balanced_q_batch(self, sample_plan: _TD3ReplaySamplePlan | None = None):
        batch_size = int(self.cfg.q_batch_size)
        requested_counts = self._replay_source_counts("q", batch_size)
        focused_student_count = requested_counts["failure_student"]
        focused_teacher_count = requested_counts["failure_teacher"]
        student_count = requested_counts["uniform_student"] + focused_student_count
        teacher_count = requested_counts["uniform_teacher"] + focused_teacher_count
        if (
            teacher_count
            and self._teacher_episode_cache_enabled()
            and bool(getattr(self, "_teacher_prefill_complete", False))
        ):
            self._ensure_teacher_actor_cache_current()
        q_fields = self._q_replay_storage_fields()
        generation_replays = []
        if teacher_count:
            generation_replays.append(self.q_teacher_replay)
        if student_count:
            generation_replays.append(self.dagger_replay)
        # Replays are deliberately fresh-only, but compact legacy/unit seams
        # may not carry the optional measurement provenance.  Sampling must
        # retain its old schema in that case; metrics simply report unavailable.
        if any(
            REPLAY_PERCEPTION_EMA_GENERATION_KEY not in replay.data
            for replay in generation_replays
        ):
            q_fields = tuple(
                key
                for key in q_fields
                if key != REPLAY_PERCEPTION_EMA_GENERATION_KEY
            )
        if not self._has_canonical_replay_mix():
            # Legacy unit/checkpoint seams may predate explicit boundary fields.
            if teacher_count and student_count:
                available = set(self.q_teacher_replay.data).intersection(
                    self.dagger_replay.data
                )
            elif teacher_count:
                available = set(self.q_teacher_replay.data)
            else:
                available = set(self.dagger_replay.data)
            q_fields = tuple(key for key in q_fields if key in available)
        collection_cache_enabled = self._student_collection_actor_cache_enabled()
        student_q_fields = q_fields
        if collection_cache_enabled:
            cache_key = REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY
            if student_count and cache_key not in self.dagger_replay.data:
                raise RuntimeError(
                    "Student Q replay lacks collection-exact next Actor observations"
                )
            # Teacher Actor inputs live once in the current-EMA device cache;
            # only Student rows own self-contained collection-time cache
            # fields in their CPU FIFO.
            student_q_fields = (*q_fields, cache_key)
        teacher = None
        student = None
        teacher_indices = torch.empty(0, dtype=torch.long)
        student_indices = torch.empty(0, dtype=torch.long)
        if sample_plan is None:
            focused_teacher = torch.zeros(0, dtype=torch.bool, device=self.device)
            if teacher_count:
                teacher_indices, focused_teacher = self._sample_teacher_indices(
                    teacher_count,
                    self.q_rng,
                    focused_count=focused_teacher_count,
                )
                teacher = _sample_replay_by_indices(
                    self.q_teacher_replay,
                    teacher_indices,
                    self.device,
                    fields=q_fields,
                )
                focused_teacher = focused_teacher.to(self.device)
            focused_student = torch.zeros(0, dtype=torch.bool, device=self.device)
            if student_count:
                (
                    student_indices,
                    focused_student,
                ) = self._sample_student_indices(
                    student_count,
                    self.q_rng,
                    focused_count=focused_student_count,
                )
                student = _sample_replay_by_indices(
                    self.dagger_replay,
                    student_indices,
                    self.device,
                    fields=student_q_fields,
                )
                focused_student = focused_student.to(self.device)
            permutation = torch.randperm(
                batch_size, device=self.device, generator=self.q_rng
            )
        else:
            teacher_indices = sample_plan.teacher_indices
            student_indices = sample_plan.student_indices
            if sample_plan.teacher_indices.numel() != teacher_count:
                raise ValueError("Teacher replay sample plan has the wrong row count")
            if sample_plan.student_indices.numel() != student_count:
                raise ValueError("Student replay sample plan has the wrong row count")
            if teacher_count:
                teacher = _sample_replay_by_indices(
                    self.q_teacher_replay,
                    sample_plan.teacher_indices,
                    self.device,
                    fields=q_fields,
                )
            if student_count:
                student = _sample_replay_by_indices(
                    self.dagger_replay,
                    sample_plan.student_indices,
                    self.device,
                    fields=student_q_fields,
                )
            focused_teacher = (
                torch.zeros(teacher_count, dtype=torch.bool, device=self.device)
                if sample_plan.teacher_focused is None
                else sample_plan.teacher_focused.to(self.device)
            )
            focused_student = (
                torch.zeros(student_count, dtype=torch.bool, device=self.device)
                if sample_plan.student_focused is None
                else sample_plan.student_focused.to(self.device)
            )
            permutation = sample_plan.permutation
            if permutation.shape != (batch_size,):
                raise ValueError("Q replay sample plan has the wrong permutation shape")
        if collection_cache_enabled and teacher is not None:
            teacher[REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY] = (
                self._teacher_actor_observations_for_indices(
                    teacher_indices, next_state=True
                )
            )
        if teacher is None:
            if (
                student is None
            ):  # pragma: no cover - batch_size validation prevents this
                raise RuntimeError("Q replay mix contains no source rows")
            mixed = dict(student)
        elif student is None:
            mixed = dict(teacher)
        else:
            mixed = {
                key: torch.cat((teacher[key], student[key]), dim=0) for key in teacher
            }
        self._attach_replay_sample_metadata(
            mixed,
            teacher_indices=teacher_indices,
            student_indices=student_indices,
            teacher_failure=focused_teacher,
            student_failure=focused_student,
        )
        mixed = {key: value[permutation] for key, value in mixed.items()}
        if not hasattr(self, "_replay_mix_rollout_metrics"):
            self._reset_replay_mix_rollout_metrics()
        self._record_replay_mix_batch("q", mixed, requested_counts)
        return mixed

    def _sample_actor_batch(
        self,
        indices: torch.Tensor | None = None,
        teacher_indices: torch.Tensor | None = None,
        teacher_focused: torch.Tensor | None = None,
        student_focused: torch.Tensor | None = None,
    ):
        """Sample one mixed Actor batch without changing the zero-share path."""
        collection_cache_enabled = self._student_collection_actor_cache_enabled()
        actuator_actor_fields = (
            (Q_ACTUATOR_CONTEXT_KEY,)
            if self._q_conditions_on_actuator_state()
            else ()
        )
        main_fields = (
            "critic_observations",
            DAGGER_REPLAY_TEACHER_ACTIONS,
            DAGGER_TEACHER_ACTION_VALID_KEY,
            REPLAY_PERCEPTION_EMA_GENERATION_KEY,
            *actuator_actor_fields,
            *(
                (REPLAY_ACTOR_OBSERVATIONS_KEY,)
                if collection_cache_enabled
                else _PERCEPTION_REPLAY_FIELDS
            ),
        )
        batch_size = int(self.cfg.dagger_batch_size)
        canonical_mix = self._has_canonical_replay_mix()
        curriculum_enabled = canonical_mix or hasattr(
            self.cfg, "failure_phase_teacher_fraction"
        )
        requested_counts = self._replay_source_counts("actor", batch_size)
        focused_student_count = requested_counts["failure_student"]
        focused_teacher_count = requested_counts["failure_teacher"]
        main_count = requested_counts["uniform_student"] + focused_student_count
        teacher_count = requested_counts["uniform_teacher"] + focused_teacher_count
        generation_replays = []
        if teacher_count:
            generation_replays.append(self.q_teacher_replay)
        if main_count:
            generation_replays.append(self.dagger_replay)
        if any(
            REPLAY_PERCEPTION_EMA_GENERATION_KEY not in replay.data
            for replay in generation_replays
        ):
            main_fields = tuple(
                key
                for key in main_fields
                if key != REPLAY_PERCEPTION_EMA_GENERATION_KEY
            )
        if (
            teacher_count
            and self._teacher_episode_cache_enabled()
            and bool(getattr(self, "_teacher_prefill_complete", False))
        ):
            self._ensure_teacher_actor_cache_current()
        student_actor_fields = main_fields
        if collection_cache_enabled:
            if (
                main_count
                and REPLAY_ACTOR_OBSERVATIONS_KEY not in self.dagger_replay.data
            ):
                raise RuntimeError(
                    "Student Actor replay lacks collection-exact Actor observations"
                )
        student_curriculum_enabled = (
            hasattr(self.cfg, "failure_phase_student_fraction") or canonical_mix
        )

        # Keep the old operation and RNG sequence byte-for-byte at the
        # backward-compatible default.
        if teacher_count == 0:
            if teacher_indices is not None:
                raise ValueError("Actor replay plan unexpectedly contains Teacher rows")
            if teacher_focused is not None:
                raise ValueError("Actor replay plan unexpectedly marks focused rows")
            if indices is None:
                if student_curriculum_enabled:
                    (
                        indices,
                        focused_student,
                    ) = self._sample_student_indices(
                        batch_size,
                        self.q_rng,
                        focused_count=focused_student_count,
                    )
                    batch = _sample_replay_by_indices(
                        self.dagger_replay,
                        indices,
                        self.device,
                        fields=student_actor_fields,
                    )
                    focused_student = focused_student.to(self.device)
                else:
                    batch = self.dagger_replay.sample(
                        batch_size,
                        self.device,
                        self.q_rng,
                        valid_key=(
                            DAGGER_IS_STUDENT_ACTION_KEY if curriculum_enabled else None
                        ),
                        fields=student_actor_fields,
                    )
                    focused_student = torch.zeros(
                        batch_size, dtype=torch.bool, device=self.device
                    )
            else:
                if indices.numel() != batch_size:
                    raise ValueError("Actor replay sample plan has the wrong row count")
                batch = self.dagger_replay.sample_by_indices(
                    indices, self.device, fields=student_actor_fields
                )
                focused_student = (
                    torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    if student_focused is None
                    else student_focused.to(self.device)
                )
            if canonical_mix:
                self._attach_replay_sample_metadata(
                    batch,
                    teacher_indices=torch.empty(0, dtype=torch.long),
                    student_indices=indices,
                    teacher_failure=torch.empty(0, dtype=torch.bool),
                    student_failure=focused_student,
                )
                if not hasattr(self, "_replay_mix_rollout_metrics"):
                    self._reset_replay_mix_rollout_metrics()
                self._record_replay_mix_batch("actor", batch, requested_counts)
            else:
                if curriculum_enabled:
                    batch[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.zeros(
                        batch_size, dtype=torch.bool, device=self.device
                    )
                    batch[FAILURE_PHASE_TEACHER_SOURCE_KEY] = torch.zeros(
                        batch_size, dtype=torch.bool, device=self.device
                    )
                if student_curriculum_enabled:
                    batch[FAILURE_PHASE_STUDENT_SOURCE_KEY] = focused_student
            return self._prepare_dagger_learning_batch(batch)

        clean_teacher_labels_available = (
            DAGGER_REPLAY_TEACHER_ACTIONS in self.q_teacher_replay.data
        )
        if (
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
            and not clean_teacher_labels_available
        ):
            raise RuntimeError(
                "noisy Teacher prefill replay lacks clean Teacher BC labels"
            )
        teacher_action_field = (
            DAGGER_REPLAY_TEACHER_ACTIONS
            if clean_teacher_labels_available
            else "actions"
        )
        teacher_fields = (
            "critic_observations",
            teacher_action_field,
            *actuator_actor_fields,
            *(
                (REPLAY_PERCEPTION_EMA_GENERATION_KEY,)
                if REPLAY_PERCEPTION_EMA_GENERATION_KEY in main_fields
                else ()
            ),
            *(() if collection_cache_enabled else _PERCEPTION_REPLAY_FIELDS),
        )
        if teacher_indices is None:
            if curriculum_enabled or collection_cache_enabled:
                # The device-resident Teacher cache is addressed by the exact
                # physical FIFO rows selected for this Actor update.  Draw
                # those indices explicitly even for legacy/no-curriculum
                # fixtures; this preserves the replay RNG sequence used by
                # ``_TD3DeviceReplay.sample`` while making cache addressing
                # fail-closed and auditable.
                teacher_indices, focused_teacher = self._sample_teacher_indices(
                    teacher_count,
                    self.q_rng,
                    focused_count=focused_teacher_count,
                )
                teacher = _sample_replay_by_indices(
                    self.q_teacher_replay,
                    teacher_indices,
                    self.device,
                    fields=teacher_fields,
                )
                focused_teacher = focused_teacher.to(self.device)
            else:
                teacher = self.q_teacher_replay.sample(
                    teacher_count,
                    self.device,
                    self.q_rng,
                    fields=teacher_fields,
                )
                focused_teacher = torch.zeros(
                    teacher_count, dtype=torch.bool, device=self.device
                )
        else:
            if teacher_indices.numel() != teacher_count:
                raise ValueError(
                    "Teacher Actor replay sample plan has the wrong row count"
                )
            teacher = _sample_replay_by_indices(
                self.q_teacher_replay,
                teacher_indices,
                self.device,
                fields=teacher_fields,
            )
            focused_teacher = (
                torch.zeros(teacher_count, dtype=torch.bool, device=self.device)
                if teacher_focused is None
                else teacher_focused.to(self.device)
            )
        if collection_cache_enabled:
            if teacher_indices is None:
                raise RuntimeError(
                    "collection-exact Teacher Actor sampling requires physical indices"
                )
            teacher[REPLAY_ACTOR_OBSERVATIONS_KEY] = (
                self._teacher_actor_observations_for_indices(
                    teacher_indices, next_state=False
                )
            )
        if teacher_action_field == "actions":
            # Legacy deterministic Teacher rings used their factual action as
            # the identical BC label. Noisy prefill fails closed above instead
            # of allowing its exploration draw to become an imitation target.
            teacher[DAGGER_REPLAY_TEACHER_ACTIONS] = teacher.pop("actions")
        teacher[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(
            teacher_count, dtype=torch.bool, device=self.device
        )

        main = None
        focused_student = torch.zeros(main_count, dtype=torch.bool, device=self.device)
        if main_count:
            if indices is None:
                if student_curriculum_enabled:
                    (
                        indices,
                        focused_student,
                    ) = self._sample_student_indices(
                        main_count,
                        self.q_rng,
                        focused_count=focused_student_count,
                    )
                    main = _sample_replay_by_indices(
                        self.dagger_replay,
                        indices,
                        self.device,
                        fields=student_actor_fields,
                    )
                    focused_student = focused_student.to(self.device)
                else:
                    main = self.dagger_replay.sample(
                        main_count,
                        self.device,
                        self.q_rng,
                        valid_key=(
                            DAGGER_IS_STUDENT_ACTION_KEY if curriculum_enabled else None
                        ),
                        fields=student_actor_fields,
                    )
            else:
                if indices.numel() != main_count:
                    raise ValueError("Actor replay sample plan has the wrong row count")
                main = self.dagger_replay.sample_by_indices(
                    indices, self.device, fields=student_actor_fields
                )
                focused_student = (
                    torch.zeros(main_count, dtype=torch.bool, device=self.device)
                    if student_focused is None
                    else student_focused.to(self.device)
                )
        elif indices is not None and indices.numel() != 0:
            raise ValueError("Actor replay sample plan has the wrong row count")

        if main is None:
            batch = teacher
        else:
            batch = {
                key: torch.cat((teacher[key], main[key]), dim=0) for key in main_fields
            }
        if canonical_mix:
            student_indices = (
                torch.empty(0, dtype=torch.long) if indices is None else indices
            )
            self._attach_replay_sample_metadata(
                batch,
                teacher_indices=teacher_indices,
                student_indices=student_indices,
                teacher_failure=focused_teacher,
                student_failure=focused_student,
            )
            if not hasattr(self, "_replay_mix_rollout_metrics"):
                self._reset_replay_mix_rollout_metrics()
            self._record_replay_mix_batch("actor", batch, requested_counts)
        else:
            batch[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.cat(
                (
                    torch.ones(teacher_count, dtype=torch.bool, device=self.device),
                    torch.zeros(main_count, dtype=torch.bool, device=self.device),
                )
            )
            if curriculum_enabled:
                batch[FAILURE_PHASE_TEACHER_SOURCE_KEY] = torch.cat(
                    (
                        focused_teacher,
                        torch.zeros(main_count, dtype=torch.bool, device=self.device),
                    )
                )
            if student_curriculum_enabled:
                batch[FAILURE_PHASE_STUDENT_SOURCE_KEY] = torch.cat(
                    (
                        torch.zeros(
                            teacher_count, dtype=torch.bool, device=self.device
                        ),
                        focused_student,
                    )
                )
        return self._prepare_dagger_learning_batch(batch)

    def _sample_four_way_perception_batch(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        batch_size = int(
            getattr(
                self.cfg,
                "perception_replay_batch_size",
                self.cfg.teacher_perception_batch_size,
            )
        )
        requested_counts = self._replay_source_counts("perception", batch_size)
        student_count = (
            requested_counts["uniform_student"] + requested_counts["failure_student"]
        )
        teacher_count = (
            requested_counts["uniform_teacher"] + requested_counts["failure_teacher"]
        )
        fields = ("critic_observations", *_PERCEPTION_REPLAY_FIELDS)
        teacher_indices = torch.empty(0, dtype=torch.long)
        student_indices = torch.empty(0, dtype=torch.long)
        teacher_failure = torch.empty(0, dtype=torch.bool)
        student_failure = torch.empty(0, dtype=torch.bool)
        teacher = None
        student = None
        if teacher_count:
            teacher_indices, teacher_failure = self._sample_teacher_indices(
                teacher_count,
                self.teacher_perception_rng,
                focused_count=requested_counts["failure_teacher"],
            )
            teacher = _sample_replay_by_indices(
                self.q_teacher_replay,
                teacher_indices,
                self.device,
                fields=fields,
            )
        if student_count:
            student_indices, student_failure = self._sample_student_indices(
                student_count,
                self.teacher_perception_rng,
                focused_count=requested_counts["failure_student"],
            )
            student = _sample_replay_by_indices(
                self.dagger_replay,
                student_indices,
                self.device,
                fields=fields,
            )
        if teacher is None and student is None:  # pragma: no cover - validated mix
            raise RuntimeError("four-way perception replay contains no rows")
        if teacher is None:
            batch = dict(student)
        elif student is None:
            batch = dict(teacher)
        else:
            batch = {
                key: torch.cat((teacher[key], student[key]), dim=0) for key in fields
            }
        self._attach_replay_sample_metadata(
            batch,
            teacher_indices=teacher_indices,
            student_indices=student_indices,
            teacher_failure=teacher_failure,
            student_failure=student_failure,
        )
        return batch, requested_counts

    def _teacher_perception_replay_loss(
        self, *, four_way: bool = False
    ) -> dict[str, torch.Tensor]:
        """Recompute one supervised perception loss from frozen Teacher inputs.

        The replay is authoritative only for model inputs.  Learned recurrent
        states and ``priv_pred`` are never stored: the online depth/object/adapt
        modules are rerun from a zero boundary through the raw burn-in window.
        The privileged target is reconstructed from the current replay state.
        """
        teacher_rows_required = True
        if four_way:
            replay_batch_size = int(
                getattr(
                    self.cfg,
                    "perception_replay_batch_size",
                    self.cfg.teacher_perception_batch_size,
                )
            )
            perception_counts = self._replay_source_counts(
                "perception", replay_batch_size
            )
            teacher_rows_required = (
                perception_counts["uniform_teacher"]
                + perception_counts["failure_teacher"]
                > 0
            )
        if self.q_teacher_replay.size < 1 and teacher_rows_required:
            raise RuntimeError(
                "teacher_perception_replay_fraction requires non-empty q_teacher_replay"
            )

        requested_counts = None
        if four_way:
            batch, requested_counts = self._sample_four_way_perception_batch()
            provenance = batch[REPLAY_SAMPLE_PROVENANCE_KEY]
            focused_teacher = provenance == REPLAY_SOURCE_FAILURE_TEACHER
            focused_student = provenance == REPLAY_SOURCE_FAILURE_STUDENT
            teacher_source = batch[REPLAY_SAMPLE_IS_TEACHER_KEY].bool()
        else:
            fields = ("critic_observations", *_PERCEPTION_REPLAY_FIELDS)
            batch_size = int(self.cfg.teacher_perception_batch_size)
            focused_count = _source_counts(
                batch_size,
                1.0,
                float(getattr(self.cfg, "failure_phase_teacher_fraction", 0.0)),
            )[2]
            indices, focused_teacher = self._sample_teacher_indices(
                batch_size,
                self.teacher_perception_rng,
                focused_count=focused_count,
            )
            batch = _sample_replay_by_indices(
                self.q_teacher_replay,
                indices,
                self.device,
                fields=fields,
            )
            focused_teacher = focused_teacher.to(self.device)
            focused_student = torch.zeros_like(focused_teacher)
            teacher_source = torch.ones_like(focused_teacher)
        batch = PPOBCDaggerFinetune._prepare_dagger_learning_batch(self, batch)
        critic_chunks = batch["critic_observations"].split(
            self._q_critic_widths, dim=-1
        )
        critic = dict(zip(self.q_critic_keys, critic_chunks))
        if OBS_PRIV_KEY not in critic or OBJECT_KEY not in critic:
            raise RuntimeError(
                "Teacher perception replay lacks privileged/object targets"
            )

        depth_u8 = batch[PERCEPTION_DEPTH_U8_KEY]
        policy_raw = batch[PERCEPTION_POLICY_RAW_KEY]
        vel_raw = batch[PERCEPTION_VEL_COMMAND_RAW_KEY]
        geometry_ids = batch[PERCEPTION_OBJECT_GEO_ID_KEY]
        is_init = batch[PERCEPTION_IS_INIT_KEY]
        row_count, window_length = depth_u8.shape[:2]
        expected_length = int(self.cfg.perception_replay_burn_in) + 2
        if int(window_length) != expected_length:
            raise ValueError(
                f"Teacher perception window has length {window_length}; "
                f"expected {expected_length}"
            )
        if geometry_ids.shape != (row_count, window_length):
            raise ValueError("Teacher object geometry IDs are window-misaligned")

        snapshot = self._vecnorm_snapshot()
        depth = self._normalize_replay_value(
            DEPTH_KEY, _decode_replay_depth_u8(depth_u8), snapshot
        )
        policy = self._normalize_replay_value(OBS_KEY, policy_raw, snapshot)
        vel = self._normalize_replay_value(VEL_CMD_KEY, vel_raw, snapshot)
        geometry = self._decode_replay_object_geo(
            geometry_ids,
            device=depth.device,
            dtype=policy.dtype,
        )

        target = TensorDict(
            {
                OBS_PRIV_KEY: critic[OBS_PRIV_KEY],
                OBJECT_KEY: critic[OBJECT_KEY],
                OBJECT_GEO_KEY: geometry[:, -2],
            },
            batch_size=(row_count,),
            device=depth.device,
        )
        with torch.no_grad():
            self.object_transform(target)
            self.encoder_priv(target)

        student = TensorDict(
            {
                DEPTH_KEY: depth,
                OBS_KEY: policy,
                VEL_CMD_KEY: vel,
                OBJECT_GEO_KEY: geometry,
                "is_init": is_init,
                "depth_hx": torch.zeros(
                    row_count,
                    window_length,
                    self.depth_feature_dim,
                    device=depth.device,
                    dtype=policy.dtype,
                ),
                "adapt_hx": torch.zeros(
                    row_count,
                    window_length,
                    int(self.cfg.latent_dim),
                    device=depth.device,
                    dtype=policy.dtype,
                ),
            },
            batch_size=(row_count, window_length),
            device=depth.device,
        )
        self.temporal_depth_gru(student)
        self.object_adapt(student)
        self.object_pred_transform(student)
        self.adapt_module(student)

        reset = is_init[:, -2].bool().reshape(row_count, -1).any(dim=-1)
        valid = (~reset).to(policy.dtype)
        object_error = self.adapt_loss_fn(
            student[OBJECT_PRED_KEY][:, -2], target[OBJECT_KEY]
        )
        priv_error = self.adapt_loss_fn(
            student[PRIV_PRED_KEY][:, -2], target[PRIV_FEATURE_KEY]
        )
        if four_way:
            # A reset-invalid row must not dilute the configured replay mix or
            # loss scale.  Source composition is already expressed by sampling;
            # there is no additional Student/Teacher coefficient here.
            object_loss = _masked_feature_mean(object_error, ~reset)
            priv_loss = _masked_feature_mean(priv_error, ~reset)
        else:
            # Preserve the legacy Teacher-replay loss scale for old configs.
            object_loss = (object_error * valid.unsqueeze(-1)).mean()
            priv_loss = (priv_error * valid.unsqueeze(-1)).mean()
        if four_way:
            if requested_counts is None:  # pragma: no cover - branch invariant
                raise RuntimeError("four-way perception counts are unavailable")
            if not hasattr(self, "_replay_mix_rollout_metrics"):
                self._reset_replay_mix_rollout_metrics()
            self._record_replay_mix_batch(
                "perception", batch, requested_counts, valid_mask=~reset
            )
        return {
            "priv_loss": priv_loss,
            "object_loss": object_loss,
            "priv_feature_norm": target[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean(),
            "priv_pred_norm": student[PRIV_PRED_KEY][:, -2].norm(p=2, dim=-1).mean(),
            "depth_feature_norm": student["_depth_feature"][:, -2]
            .norm(p=2, dim=-1)
            .mean(),
            "valid_fraction": valid.mean(),
            "rows": priv_loss.new_tensor(float(row_count)),
            "failure_phase_teacher_fraction": focused_teacher.float().mean(),
            "failure_phase_student_fraction": focused_student.float().mean(),
            "teacher_source_fraction": teacher_source.float().mean(),
            "student_source_fraction": (~teacher_source).float().mean(),
        }

    @set_recurrent_mode(True)
    def _run_teacher_perception_warmup(self) -> dict[str, float]:
        """Warm perception from frozen Teacher inputs before Student control.

        Dynamic prefill is intentionally optimizer-free while it is collecting
        trajectories.  Once the successful-only Teacher ring is complete, this
        one-shot phase trains only the online depth/object/adaptation stack from
        replay.  A hard online-to-EMA copy is required at the boundary: using
        the normal 0.04 rollout Polyak step would leave Student behavior almost
        entirely on the constructor-fresh depth model for its first rollout.
        Actor, Critic, alpha, replay contents, and main-rollout counters are not
        touched here.
        """
        if bool(getattr(self, "_teacher_perception_warmup_complete", False)):
            return copy.deepcopy(
                getattr(self, "_last_teacher_perception_warmup_metrics", {})
            )
        prefill_flag = getattr(self, "_teacher_prefill_complete", None)
        if prefill_flag is not None and not bool(prefill_flag):
            raise RuntimeError(
                "Teacher perception warm-up requires a completed Teacher prefill"
            )

        requested_steps = int(self.cfg.teacher_perception_warmup_steps)
        if requested_steps < 0:
            raise ValueError(
                "teacher_perception_warmup_steps must be a non-negative integer"
            )
        trainable = bool(self.cfg.train_perception)
        if requested_steps and trainable:
            if (
                prefill_flag is not None
                and self.q_teacher_replay.size < self.q_teacher_replay.capacity
            ):
                raise RuntimeError(
                    "Teacher perception warm-up requires a capacity-filled replay"
                )
            if self.opt_adapt is None:
                raise RuntimeError(
                    "Teacher perception warm-up requires the perception optimizer"
                )

        priv_losses: list[torch.Tensor] = []
        object_losses: list[torch.Tensor] = []
        grad_norms: list[torch.Tensor] = []
        remaining = max(
            requested_steps
            - int(getattr(self, "_teacher_perception_warmup_updates", 0)),
            0,
        )
        if trainable:
            for _ in range(remaining):
                self.opt_adapt.zero_grad(set_to_none=True)
                teacher = self._teacher_perception_replay_loss()
                priv_loss = teacher["priv_loss"]
                object_loss = teacher["object_loss"]
                (priv_loss + object_loss).backward()
                parameters = list(self.adapt_module.parameters())
                parameters += list(self.object_adapt.parameters())
                parameters += list(self.temporal_depth_gru.parameters())
                grad_norm = nn.utils.clip_grad_norm_(
                    parameters, float(self.cfg.max_grad_norm)
                )
                self.opt_adapt.step()
                self._teacher_perception_warmup_updates += 1
                priv_losses.append(priv_loss.detach())
                object_losses.append(object_loss.detach())
                grad_norms.append(torch.as_tensor(grad_norm).detach())

            if requested_steps:
                self.adapt_ema.load_state_dict(
                    self.adapt_module.state_dict(), strict=True
                )
                self.object_adapt_ema.load_state_dict(
                    self.object_adapt.state_dict(), strict=True
                )
                self.temporal_depth_gru_ema.load_state_dict(
                    self.temporal_depth_gru.state_dict(), strict=True
                )
                for module in (
                    self.adapt_ema,
                    self.object_adapt_ema,
                    self.temporal_depth_gru_ema,
                ):
                    module.requires_grad_(False).eval()

        def _mean(values: list[torch.Tensor]) -> float:
            if not values:
                return 0.0
            return torch.stack([value.float() for value in values]).mean().item()

        self._teacher_perception_warmup_complete = True
        metrics = {
            "prefill_perception_warmup_steps": float(
                self._teacher_perception_warmup_updates
            ),
            "prefill_perception_warmup_complete": 1.0,
            "prefill_perception_warmup_priv_loss": _mean(priv_losses),
            "prefill_perception_warmup_object_loss": _mean(object_losses),
            "prefill_perception_warmup_grad_norm": _mean(grad_norms),
            "prefill_perception_warmup_skipped_frozen": float(not trainable),
        }
        self._last_teacher_perception_warmup_metrics = copy.deepcopy(metrics)
        return metrics

    def _train_adapt_four_way(self) -> dict[str, float]:
        """Run main perception updates entirely from the four replay strata."""
        if self.opt_adapt is None:
            raise RuntimeError("four-way perception replay requires opt_adapt")
        if bool(self.cfg.train_dr_estimator):
            raise RuntimeError(
                "four-way perception replay has no raw replay target for the DR "
                "estimator; disable train_dr_estimator or use legacy mode"
            )

        infos: list[dict[str, torch.Tensor]] = []
        update_count = 2 * int(self.cfg.num_minibatches)
        for _ in range(update_count):
            self.opt_adapt.zero_grad(set_to_none=True)
            replay = self._teacher_perception_replay_loss(four_way=True)
            (replay["priv_loss"] + replay["object_loss"]).backward()

            depth_params = list(self.temporal_depth_gru.parameters())
            depth_parameter_norms = [
                parameter.grad.detach().norm(2)
                for parameter in depth_params
                if parameter.grad is not None
            ]
            depth_grad_norm = (
                torch.stack(depth_parameter_norms).norm(2)
                if depth_parameter_norms
                else torch.zeros((), device=self.device)
            )
            all_params = list(self.adapt_module.parameters())
            all_params += list(self.object_adapt.parameters())
            all_params += depth_params
            grad_norm = nn.utils.clip_grad_norm_(
                all_params, float(self.cfg.max_grad_norm)
            )
            self.opt_adapt.step()

            zero = replay["priv_loss"].new_zeros(())
            infos.append(
                {
                    "adapt/priv_loss": replay["priv_loss"].detach(),
                    "adapt/object_loss": replay["object_loss"].detach(),
                    "adapt/online_priv_loss": zero,
                    "adapt/online_object_loss": zero,
                    "adapt/teacher_replay_priv_loss": replay["priv_loss"].detach(),
                    "adapt/teacher_replay_object_loss": replay["object_loss"].detach(),
                    "adapt/teacher_replay_fraction": replay[
                        "teacher_source_fraction"
                    ].detach(),
                    "adapt/teacher_replay_valid_fraction": replay[
                        "valid_fraction"
                    ].detach(),
                    "adapt/teacher_replay_rows": replay["rows"].detach(),
                    "adapt/online_student_fraction": zero,
                    "adapt/replay_student_fraction": replay[
                        "student_source_fraction"
                    ].detach(),
                    "adapt/replay_teacher_fraction": replay[
                        "teacher_source_fraction"
                    ].detach(),
                    "adapt/failure_phase_student_fraction": replay[
                        "failure_phase_student_fraction"
                    ].detach(),
                    "adapt/failure_phase_teacher_fraction": replay[
                        "failure_phase_teacher_fraction"
                    ].detach(),
                    "adapt/grad_norm": torch.as_tensor(grad_norm).detach(),
                    "adapt/depth_grad_norm": depth_grad_norm.detach(),
                    "adapt/priv_feature_norm": replay["priv_feature_norm"].detach(),
                    "adapt/priv_pred_norm": replay["priv_pred_norm"].detach(),
                    "adapt/depth_feature_norm": replay["depth_feature_norm"].detach(),
                    "adapt/teacher_replay_priv_feature_norm": replay[
                        "priv_feature_norm"
                    ].detach(),
                    "adapt/teacher_replay_priv_pred_norm": replay[
                        "priv_pred_norm"
                    ].detach(),
                    "adapt/teacher_replay_depth_feature_norm": replay[
                        "depth_feature_norm"
                    ].detach(),
                }
            )

        # Match the legacy perception optimizer cadence: one EMA update after
        # exactly 2 * num_minibatches replay-only optimizer steps.
        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        soft_copy_(self.object_adapt, self.object_adapt_ema, 0.04)
        soft_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema, 0.04)

        result = {
            key: torch.stack([item[key].float() for item in infos]).mean().item()
            for key in sorted(infos[0])
        }
        with torch.no_grad():
            squared_error = torch.zeros((), device=self.device)
            parameter_count = 0
            for online, ema in zip(
                self.temporal_depth_gru.parameters(),
                self.temporal_depth_gru_ema.parameters(),
            ):
                squared_error += (online - ema).square().sum()
                parameter_count += online.numel()
            result["adapt/depth_ema_rms_gap"] = (
                (squared_error / max(parameter_count, 1)).sqrt().item()
            )
        result["adapt/perception_frozen"] = 0.0
        result["adapt/perception_four_way"] = 1.0
        return result

    @set_recurrent_mode(True)
    def train_adapt(self, tensordict: TensorDict):
        """Train perception under the selected live or compatibility contract."""
        if not bool(self.cfg.train_perception):
            # Frozen perception is inference-only: do not construct an online
            # graph, step its optimizer, or advance any EMA weights.
            return {
                "adapt/perception_frozen": 1.0,
                "adapt/priv_loss": 0.0,
                "adapt/object_loss": 0.0,
                "adapt/grad_norm": 0.0,
                "adapt/depth_grad_norm": 0.0,
            }
        perception_mode = str(
            getattr(self.cfg, "perception_replay_mode", "legacy_online_student")
        )
        if perception_mode == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE:
            reset = tensordict["is_init"].bool()
            student_source = tensordict.get(DAGGER_IS_STUDENT_ACTION_KEY, None)
            if student_source is not None:
                while student_source.ndim < reset.ndim:
                    student_source = student_source.unsqueeze(-1)
                while reset.ndim < student_source.ndim:
                    reset = reset.unsqueeze(-1)
                teacher_controlled = (~reset) & ~student_source.bool()
                if bool(teacher_controlled.any()):
                    raise RuntimeError(
                        "online_student_rollout perception received a "
                        "Teacher-controlled live transition"
                    )
            # This is deliberately a direct delegation, not a reimplementation:
            # PPOVEL owns the recurrent [num_envs, train_every] minibatching,
            # reset masking, two optimizer epochs, gradient clipping, and the
            # single online-to-EMA update.  No replay sampler or zero-state
            # recurrent reconstruction is reachable from this branch.
            result = PPOVEL.train_adapt(self, tensordict)
            if student_source is None:
                online_student_fraction = (~reset).float().mean().item()
            else:
                online_student_fraction = (
                    (~reset) & student_source.bool()
                ).float().mean().item()
            result.update(
                {
                    "adapt/perception_frozen": 0.0,
                    "adapt/perception_online_student_rollout": 1.0,
                    "adapt/perception_four_way": 0.0,
                    "adapt/perception_replay_rows": 0.0,
                    "adapt/perception_live_rows": float(
                        math.prod(tensordict.batch_size)
                    ),
                    "adapt/perception_optimizer_steps": float(
                        2 * int(self.cfg.num_minibatches)
                    ),
                    "adapt/perception_sequence_length": float(
                        tensordict.batch_size[-1]
                    ),
                    "adapt/online_priv_loss": result["adapt/priv_loss"],
                    "adapt/online_object_loss": result["adapt/object_loss"],
                    "adapt/online_student_fraction": online_student_fraction,
                    "adapt/teacher_replay_priv_loss": 0.0,
                    "adapt/teacher_replay_object_loss": 0.0,
                    "adapt/teacher_replay_fraction": 0.0,
                    "adapt/teacher_replay_rows": 0.0,
                    "adapt/teacher_replay_valid_fraction": 0.0,
                    "adapt/replay_student_fraction": 0.0,
                    "adapt/replay_teacher_fraction": 0.0,
                    "adapt/failure_phase_student_fraction": 0.0,
                    "adapt/failure_phase_teacher_fraction": 0.0,
                }
            )
            return result
        if (
            perception_mode == "four_way"
        ):
            # Deliberately do not inspect or transform ``tensordict`` here: in
            # v4 it is not an implicit fifth perception source.
            return self._train_adapt_four_way()
        fraction = float(self.cfg.teacher_perception_replay_fraction)
        curriculum_enabled = hasattr(self.cfg, "failure_phase_teacher_fraction")
        if fraction == 0.0 and not curriculum_enabled:
            # Preserve the original implementation, RNG stream, optimizer
            # count, and EMA update exactly at the backward-compatible default.
            return PPOVEL.train_adapt(self, tensordict)
        if fraction > 0.0 and self.q_teacher_replay.size < 1:
            raise RuntimeError(
                "teacher_perception_replay_fraction is positive but the "
                "Teacher replay is empty"
            )

        if fraction < 1.0:
            with torch.no_grad():
                self.object_transform(tensordict)
                self.encoder_priv(tensordict)

        infos: list[dict[str, torch.Tensor]] = []
        for _ in range(2):
            minibatches = (
                make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every)
                if fraction < 1.0
                else (None for _ in range(int(self.cfg.num_minibatches)))
            )
            for minibatch in minibatches:
                if minibatch is not None:
                    self.temporal_depth_gru(minibatch)
                    self.object_adapt(minibatch)
                    student_source = minibatch.get(DAGGER_IS_STUDENT_ACTION_KEY, None)
                    if student_source is None:
                        student_source = torch.ones_like(
                            minibatch["is_init"], dtype=torch.bool
                        )
                    reset_source = minibatch["is_init"].bool()
                    while student_source.ndim < reset_source.ndim:
                        student_source = student_source.unsqueeze(-1)
                    while reset_source.ndim < student_source.ndim:
                        reset_source = reset_source.unsqueeze(-1)
                    online_valid = (~reset_source) & student_source.bool()
                    online_object_loss = _masked_feature_mean(
                        self.adapt_loss_fn(
                            minibatch[OBJECT_PRED_KEY], minibatch[OBJECT_KEY]
                        ),
                        online_valid,
                    )
                    self.object_pred_transform(minibatch)
                    self.adapt_module(minibatch)
                    online_priv_loss = _masked_feature_mean(
                        self.adapt_loss_fn(
                            minibatch[PRIV_PRED_KEY], minibatch[PRIV_FEATURE_KEY]
                        ),
                        online_valid,
                    )
                    online_student_fraction = online_valid.float().mean()
                    online_priv_feature_norm = (
                        minibatch[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean()
                    )
                    online_priv_pred_norm = (
                        minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean()
                    )
                    online_depth_feature_norm = (
                        minibatch["_depth_feature"].norm(p=2, dim=-1).mean()
                    )
                else:
                    online_priv_loss = torch.zeros((), device=self.device)
                    online_object_loss = torch.zeros((), device=self.device)
                    online_priv_feature_norm = torch.zeros((), device=self.device)
                    online_priv_pred_norm = torch.zeros((), device=self.device)
                    online_depth_feature_norm = torch.zeros((), device=self.device)
                    online_student_fraction = torch.zeros((), device=self.device)

                self.opt_adapt.zero_grad()
                online_total = online_priv_loss + online_object_loss
                if fraction < 1.0:
                    ((1.0 - fraction) * online_total).backward()

                # Build the replay graph only after releasing the much larger
                # live-rollout graph.  Two backwards before one optimizer step
                # are exactly the gradient of the weighted sum while reducing
                # peak CNN/GRU activation memory.
                if fraction > 0.0:
                    teacher = self._teacher_perception_replay_loss()
                    teacher_total = teacher["priv_loss"] + teacher["object_loss"]
                    (fraction * teacher_total).backward()
                else:
                    teacher = {
                        "priv_loss": torch.zeros((), device=self.device),
                        "object_loss": torch.zeros((), device=self.device),
                        "valid_fraction": torch.zeros((), device=self.device),
                        "rows": torch.zeros((), device=self.device),
                        "failure_phase_teacher_fraction": torch.zeros(
                            (), device=self.device
                        ),
                        "priv_feature_norm": torch.zeros((), device=self.device),
                        "priv_pred_norm": torch.zeros((), device=self.device),
                        "depth_feature_norm": torch.zeros((), device=self.device),
                    }
                priv_loss = (
                    1.0 - fraction
                ) * online_priv_loss.detach() + fraction * teacher["priv_loss"].detach()
                object_loss = (
                    1.0 - fraction
                ) * online_object_loss.detach() + fraction * teacher[
                    "object_loss"
                ].detach()
                all_params = list(self.adapt_module.parameters())
                all_params += list(self.object_adapt.parameters())
                depth_params = list(self.temporal_depth_gru.parameters())
                depth_parameter_norms = [
                    parameter.grad.detach().norm(2)
                    for parameter in depth_params
                    if parameter.grad is not None
                ]
                depth_grad_norm = (
                    torch.stack(depth_parameter_norms).norm(2)
                    if depth_parameter_norms
                    else torch.zeros((), device=self.device)
                )
                all_params += depth_params
                opt_adapt_grad_norm = nn.utils.clip_grad_norm_(
                    all_params, self.cfg.max_grad_norm
                )
                self.opt_adapt.step()

                info = {
                    "adapt/priv_loss": priv_loss.detach(),
                    "adapt/object_loss": object_loss.detach(),
                    "adapt/online_priv_loss": online_priv_loss.detach(),
                    "adapt/online_object_loss": online_object_loss.detach(),
                    "adapt/teacher_replay_priv_loss": teacher["priv_loss"].detach(),
                    "adapt/teacher_replay_object_loss": teacher["object_loss"].detach(),
                    "adapt/teacher_replay_fraction": priv_loss.new_tensor(fraction),
                    "adapt/teacher_replay_valid_fraction": teacher[
                        "valid_fraction"
                    ].detach(),
                    "adapt/teacher_replay_rows": teacher["rows"].detach(),
                    "adapt/online_student_fraction": online_student_fraction.detach(),
                    "adapt/failure_phase_teacher_fraction": teacher[
                        "failure_phase_teacher_fraction"
                    ].detach(),
                    "adapt/grad_norm": torch.as_tensor(opt_adapt_grad_norm).detach(),
                    "adapt/depth_grad_norm": depth_grad_norm.detach(),
                    "adapt/priv_feature_norm": online_priv_feature_norm.detach(),
                    "adapt/priv_pred_norm": online_priv_pred_norm.detach(),
                    "adapt/depth_feature_norm": online_depth_feature_norm.detach(),
                    "adapt/teacher_replay_priv_feature_norm": teacher[
                        "priv_feature_norm"
                    ].detach(),
                    "adapt/teacher_replay_priv_pred_norm": teacher[
                        "priv_pred_norm"
                    ].detach(),
                    "adapt/teacher_replay_depth_feature_norm": teacher[
                        "depth_feature_norm"
                    ].detach(),
                }

                if self.cfg.train_dr_estimator and minibatch is not None:
                    minibatch[PRIV_PRED_KEY] = minibatch[PRIV_PRED_KEY].detach()
                    self.dr_estimator(minibatch)
                    dr_est_loss = (
                        (minibatch["dr_pred"] - minibatch["dr_"]).square().mean()
                    )
                    self.opt_dr_estimator.zero_grad()
                    dr_est_loss.backward()
                    dr_est_grad_norm = nn.utils.clip_grad_norm_(
                        self.dr_estimator.parameters(), self.cfg.max_grad_norm
                    )
                    self.opt_dr_estimator.step()
                    info["adapt/dr_est_grad_norm"] = torch.as_tensor(
                        dr_est_grad_norm
                    ).detach()
                    info["adapt/dr_est_loss"] = dr_est_loss.detach()
                infos.append(info)

        # Exactly one EMA update per main rollout, matching PPOVEL.train_adapt.
        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        soft_copy_(self.object_adapt, self.object_adapt_ema, 0.04)
        soft_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema, 0.04)

        result = {
            key: torch.stack([item[key].float() for item in infos]).mean().item()
            for key in sorted(infos[0])
        }
        with torch.no_grad():
            squared_error = torch.zeros((), device=self.device)
            parameter_count = 0
            for online, ema in zip(
                self.temporal_depth_gru.parameters(),
                self.temporal_depth_gru_ema.parameters(),
            ):
                squared_error += (online - ema).square().sum()
                parameter_count += online.numel()
            result["adapt/depth_ema_rms_gap"] = (
                (squared_error / max(parameter_count, 1)).sqrt().item()
            )
        result["adapt/perception_frozen"] = 0.0
        return result

    @staticmethod
    def _mean_metric_dict(metrics: list[dict[str, torch.Tensor]], keys):
        if not metrics:
            return {key: 0.0 for key in keys}
        return {
            key: torch.stack(
                [torch.as_tensor(item[key]).detach().float() for item in metrics]
            )
            .mean()
            .item()
            for key in keys
        }

    def _q_updates_due(self, accepted_student_rows: int) -> int:
        """Return Q updates from total sampled-row/new-Student-row credit.

        ``q_update_to_data_ratio=None`` preserves the historical fixed update
        burst.  A configured ratio makes optimizer work invariant to the
        number of environments and to how an equivalent set of replay rows is
        chunked across calls.  Credit is intentionally earned only after both
        replay sources satisfy the learning-start gate, avoiding a catch-up
        burst for rows collected while Q learning was unavailable.
        """
        if (
            isinstance(accepted_student_rows, bool)
            or not isinstance(accepted_student_rows, Integral)
            or int(accepted_student_rows) < 0
        ):
            raise ValueError("accepted_student_rows must be a non-negative integer")

        ratio = getattr(self.cfg, "q_update_to_data_ratio", None)
        if ratio is None:
            return int(self.cfg.q_updates_per_rollout)
        if (
            isinstance(ratio, bool)
            or not math.isfinite(float(ratio))
            or float(ratio) <= 0.0
        ):
            raise ValueError("q_update_to_data_ratio must be finite and positive")

        batch_size = int(self.cfg.q_batch_size)
        self.q_update_row_credit = float(
            getattr(self, "q_update_row_credit", 0.0)
        ) + int(accepted_student_rows) * float(ratio)
        updates = int(self.q_update_row_credit // batch_size)
        self.q_update_row_credit -= updates * batch_size
        if self.q_update_row_credit < 0.0:
            if self.q_update_row_credit > -1e-9:
                self.q_update_row_credit = 0.0
            else:
                raise RuntimeError("Q UTD row credit became negative")
        if not 0.0 <= self.q_update_row_credit < batch_size:
            raise RuntimeError("Q UTD row credit escaped its batch range")
        return updates

    def train_op(self, tensordict):
        """Collect locked transitions and run Phase-1 TD3/BC updates only."""
        self._reset_replay_mix_rollout_metrics()
        teacher_prefill_active = self._teacher_prefill_active()
        collect_teacher_q = self._collect_teacher_q_replay_this_rollout()
        rollout = tensordict.exclude("stats")
        failure_anchors_added = 0
        if (
            not teacher_prefill_active
            and len(rollout.batch_size) == 2
            and DAGGER_IS_STUDENT_ACTION_KEY in rollout.keys(True, True)
            and TERM_KEY in rollout.keys(True, True)
        ):
            failure_anchors_added = self._update_failure_phase_histogram(rollout)
            self._capture_student_perception_drift_rollout(rollout)
        transition_chunks = tuple(self._dagger_transition_chunks(rollout))
        appended = 0
        teacher_rows_appended = 0
        teacher_rows_discarded = 0
        teacher_unresolved_rows_discarded = 0
        teacher_selected = 0
        valid_labels = 0
        student_selected = 0
        source_rows = 0
        collector_noise_norm = 0.0
        if transition_chunks:
            transitions = {
                key: torch.cat([chunk[key] for chunk in transition_chunks], dim=0)
                for key in transition_chunks[0]
            }
            del transition_chunks
            replay_device = self.dagger_replay.device
            valid = transitions[DAGGER_TEACHER_ACTION_VALID_KEY].bool()
            is_student = transitions[DAGGER_IS_STUDENT_ACTION_KEY].bool()
            valid_labels = int(valid.sum().item())
            teacher_selected = int((~is_student).sum().item())
            student_selected = int(is_student.sum().item())
            source_rows = teacher_selected + student_selected
            collector_noise_norm = (
                transitions[TD3_COLLECTOR_NOISE_KEY].float().norm(dim=-1).mean().item()
            )

            def _stage_rows(keys, indices):
                return {
                    key: transitions[key]
                    .index_select(0, indices)
                    .detach()
                    .to(replay_device)
                    for key in keys
                }

            if teacher_prefill_active:
                committed, discarded = self._stage_teacher_prefill_rows(
                    transitions,
                    rollout if len(rollout.batch_size) == 2 else None,
                )
                teacher_rows_appended += committed
                teacher_rows_discarded += discarded
            elif collect_teacher_q:
                teacher_executed = valid & ~is_student
                if teacher_executed.any():
                    teacher_indices = teacher_executed.nonzero(as_tuple=False).squeeze(
                        -1
                    )
                    q_teacher = _stage_rows(
                        self._q_replay_storage_fields(), teacher_indices
                    )
                    teacher_rows_appended += self.q_teacher_replay.extend(q_teacher)
                    del q_teacher, teacher_indices
                del teacher_executed
            if not teacher_prefill_active:
                # The main replay source is Student-only at every configured mix.
                # Teacher-executed main rows are neither sampled by Q/Actor nor
                # used by the live perception loss, so retaining them would
                # only consume replay horizon and host-transfer bandwidth.
                student_indices = is_student.nonzero(as_tuple=False).squeeze(-1)
                if student_indices.numel():
                    student_fields = tuple(
                        key
                        for key in transitions
                        if key not in _PREFILL_INTERNAL_FIELDS
                    )
                    if isinstance(self.dagger_replay, _TD3DeviceReplay):
                        appended = self.dagger_replay.extend_by_indices(
                            transitions,
                            student_indices,
                            fields=student_fields,
                        )
                    else:
                        student_staged = _stage_rows(
                            student_fields, student_indices
                        )
                        appended = self.dagger_replay.extend(student_staged)
                        del student_staged
                del student_indices
            del valid, is_student, transitions

        if teacher_prefill_active:
            # Collection owns no optimizer and no main replay. Only the final
            # capacity-filling call runs the explicit Teacher perception
            # warm-up below, before Student behavior can begin.
            self.teacher_prefill_rollout_count += 1
            self.teacher_prefill_environment_steps += int(self.cfg.train_every)
            warmup_metrics = {
                "prefill_perception_warmup_steps": float(
                    getattr(self, "_teacher_perception_warmup_updates", 0)
                ),
                "prefill_perception_warmup_complete": float(
                    getattr(self, "_teacher_perception_warmup_complete", False)
                ),
                "prefill_perception_warmup_priv_loss": 0.0,
                "prefill_perception_warmup_object_loss": 0.0,
                "prefill_perception_warmup_grad_norm": 0.0,
                "prefill_perception_warmup_skipped_frozen": 0.0,
            }
            if self._all_ranks_teacher_replay_full():
                self._teacher_prefill_complete = True
                teacher_unresolved_rows_discarded = (
                    self._discard_unresolved_teacher_prefill_rows()
                )
                teacher_rows_discarded += teacher_unresolved_rows_discarded
                self._freeze_teacher_episode_replay()
            elif self.teacher_prefill_rollout_count >= int(
                self.cfg.teacher_prefill_max_rollouts
            ):
                raise RuntimeError(
                    "Successful-only Teacher prefill reached "
                    f"teacher_prefill_max_rollouts={int(self.cfg.teacher_prefill_max_rollouts)} "
                    "before every rank filled q_teacher_replay; "
                    f"local_size={self.q_teacher_replay.size}, "
                    f"capacity={self.q_teacher_replay.capacity}, "
                    f"successful_episodes={self._teacher_prefill_successful_episodes}, "
                    f"failed_episodes={self._teacher_prefill_failed_episodes}, "
                    f"timeout_episodes={self._teacher_prefill_timeout_episodes}, "
                    f"pending_rows={self._teacher_prefill_pending_rows()}"
                )
            if self._teacher_prefill_complete and hasattr(
                self.cfg, "failure_phase_num_bins"
            ):
                self._build_teacher_phase_index()
            if self._teacher_prefill_complete:
                warmup_metrics = self._run_teacher_perception_warmup()
                self._ensure_teacher_actor_cache_current()
            prefill_progress = self.q_teacher_replay.size / max(
                self.q_teacher_replay.capacity, 1
            )
            info = {
                "td3/method_distributional_td3_teacher_bc_v1": 1.0,
                "td3/prefill_active": 1.0,
                "td3/prefill_rollout_count": self.teacher_prefill_rollout_count,
                "td3/prefill_max_rollouts": int(self.cfg.teacher_prefill_max_rollouts),
                "td3/prefill_target_rows": self.q_teacher_replay.capacity,
                "td3/prefill_progress": prefill_progress,
                "td3/prefill_environment_steps": (
                    self.teacher_prefill_environment_steps
                ),
                "td3/prefill_rows_this_rollout": teacher_rows_appended,
                "td3/prefill_discarded_rows_this_rollout": teacher_rows_discarded,
                "td3/prefill_unresolved_rows_discarded": (
                    teacher_unresolved_rows_discarded
                ),
                "td3/prefill_pending_rows": self._teacher_prefill_pending_rows(),
                "td3/prefill_successful_episodes": (
                    self._teacher_prefill_successful_episodes
                ),
                **self._prefill_success_motion_metrics(),
                "td3/prefill_failed_episodes": self._teacher_prefill_failed_episodes,
                "td3/prefill_timeout_episodes": (
                    self._teacher_prefill_timeout_episodes
                ),
                "td3/prefill_incomplete_episodes": (
                    self._teacher_prefill_incomplete_episodes
                ),
                "td3/prefill_discarded_rows": self._teacher_prefill_discarded_rows,
                "td3/teacher_replay_rows_this_rollout": teacher_rows_appended,
                "td3/teacher_replay_frozen": float(self._teacher_q_replay_frozen()),
                "td3/prefill_forced_teacher_fraction": teacher_selected
                / max(source_rows, 1),
                "td3/teacher_replay_size": self.q_teacher_replay.size,
                "td3/replay_size": self.dagger_replay.size,
                "td3/replay_seen": self.dagger_replay.seen,
                "td3/replay_ready": 0.0,
                "td3/perception_replay_ready": 0.0,
                "td3/actor_update_count": self.actor_update_count,
                "td3/critic_update_count": self.critic_update_count,
                "td3/q_update_to_data_ratio_config": (
                    -1.0
                    if getattr(self.cfg, "q_update_to_data_ratio", None) is None
                    else float(self.cfg.q_update_to_data_ratio)
                ),
                "td3/q_update_row_credit": float(
                    getattr(self, "q_update_row_credit", 0.0)
                ),
                "td3/q_teacher_replay_fraction_config": float(
                    self._replay_mix_fractions("q")["uniform_teacher"]
                    + self._replay_mix_fractions("q")["failure_teacher"]
                ),
                "td3/actor_updates_this_rollout": 0,
                "td3/critic_updates_this_rollout": 0,
                "td3/actor_teacher_replay_fraction": 0.0,
                "td3/collector_exploration_noise_norm": collector_noise_norm,
                "td3/valid_teacher_fraction": valid_labels / max(source_rows, 1),
                "td3/beta": float(self._teacher_mixture_probability()),
                "td3/rollout_count": self.dagger_rollout_count,
                "td3/environment_steps": self.dagger_environment_steps,
                "td3/failure_phase_episodes": int(
                    getattr(self, "_failure_phase_episode_count", 0)
                ),
                "td3/failure_phase_anchors": int(
                    getattr(self, "_failure_phase_anchor_count", 0)
                ),
            }
            info.update({f"td3/{key}": value for key, value in warmup_metrics.items()})
            for purpose in ("q", "actor", "perception"):
                info.update(
                    {
                        f"replay/{purpose}/{name}": value
                        for name, value in self._replay_mix_metrics(purpose).items()
                    }
                )
            self._last_td3_diagnostics = {
                key: float(value)
                for key, value in info.items()
                if key.startswith("td3/") and isinstance(value, (int, float))
            }
            return info

        critic_metrics: list[dict[str, torch.Tensor]] = []
        actor_metrics: list[dict[str, torch.Tensor]] = []
        q_staleness_age_batches: list[torch.Tensor | None] = []
        actor_staleness_age_batches: list[torch.Tensor | None] = []
        student_q_rows = self.dagger_replay.valid_count(DAGGER_IS_STUDENT_ACTION_KEY)
        learning_starts = int(self.cfg.td3_learning_starts)
        q_counts = self._replay_source_counts("q", int(self.cfg.q_batch_size))
        q_student_count = q_counts["uniform_student"] + q_counts["failure_student"]
        q_teacher_count = q_counts["uniform_teacher"] + q_counts["failure_teacher"]
        actor_counts = self._replay_source_counts(
            "actor", int(self.cfg.dagger_batch_size)
        )
        actor_student_count = (
            actor_counts["uniform_student"] + actor_counts["failure_student"]
        )
        actor_teacher_count = (
            actor_counts["uniform_teacher"] + actor_counts["failure_teacher"]
        )
        teacher_replay_required = q_teacher_count > 0 or actor_teacher_count > 0
        student_replay_required = q_student_count > 0 or actor_student_count > 0
        replay_ready = (
            not teacher_replay_required or self.q_teacher_replay.size >= learning_starts
        ) and (not student_replay_required or student_q_rows >= learning_starts)
        if replay_ready:
            if teacher_replay_required:
                self._ensure_teacher_actor_cache_current()
            q_updates = self._q_updates_due(appended)
            sample_plans = None
            if (
                isinstance(self.dagger_replay, _TD3DeviceReplay)
                and isinstance(self.q_teacher_replay, _TD3DeviceReplay)
                and self.dagger_replay.device.type == "cpu"
                and self.q_teacher_replay.device.type == "cpu"
            ):
                sample_plans = (
                    self._prefetch_curriculum_sample_plans(q_updates)
                    if self._has_canonical_replay_mix()
                    or hasattr(self.cfg, "failure_phase_teacher_fraction")
                    else _prefetch_td3_replay_sample_plans(
                        self.dagger_replay,
                        self.q_teacher_replay,
                        q_batch_size=int(self.cfg.q_batch_size),
                        actor_batch_size=int(self.cfg.dagger_batch_size),
                        update_count=q_updates,
                        policy_delay=int(self.cfg.policy_delay),
                        critic_update_count=int(self.critic_update_count),
                        q_teacher_replay_ratio=float(self.cfg.q_teacher_replay_ratio),
                        teacher_actor_replay_fraction=float(
                            self.cfg.teacher_actor_replay_fraction
                        ),
                        output_device=self.device,
                        generator=self.q_rng,
                    )
                )
            for update_index in range(q_updates):
                sample_plan = (
                    None if sample_plans is None else sample_plans[update_index]
                )
                q_batch = self._sample_balanced_q_batch(sample_plan)
                q_staleness_age_batches.append(
                    self._student_replay_ema_ages(q_batch)
                )
                q_batch = self._prepare_dagger_learning_batch(q_batch)
                critic_metrics.append(self._critic_update(q_batch))
                if self.critic_update_count % int(self.cfg.policy_delay) == 0:
                    actor_indices = (
                        None if sample_plan is None else sample_plan.actor_indices
                    )
                    actor_teacher_indices = (
                        None
                        if sample_plan is None
                        else sample_plan.actor_teacher_indices
                    )
                    if (
                        sample_plan is not None
                        and actor_indices is None
                        and actor_teacher_indices is None
                    ):
                        raise RuntimeError("delayed Actor replay plan is missing")
                    actor_batch = self._sample_actor_batch(
                        actor_indices,
                        actor_teacher_indices,
                        (
                            None
                            if sample_plan is None
                            else sample_plan.actor_teacher_focused
                        ),
                        (
                            None
                            if sample_plan is None
                            else sample_plan.actor_student_focused
                        ),
                    )
                    actor_staleness_age_batches.append(
                        self._student_replay_ema_ages(actor_batch)
                    )
                    delayed = self._maybe_delayed_actor_and_targets(
                        actor_batch
                    )
                    if delayed is None:
                        raise RuntimeError("scheduled delayed Actor update was skipped")
                    actor_metrics.append(delayed)
            del sample_plans

        perception_four_way = (
            str(getattr(self.cfg, "perception_replay_mode", "legacy_online_student"))
            == "four_way"
        )
        perception_replay_ready = True
        if perception_four_way and bool(self.cfg.train_perception):
            perception_counts = self._replay_source_counts(
                "perception", int(self.cfg.perception_replay_batch_size)
            )
            perception_teacher_required = (
                perception_counts["uniform_teacher"]
                + perception_counts["failure_teacher"]
                > 0
            )
            perception_student_required = (
                perception_counts["uniform_student"]
                + perception_counts["failure_student"]
                > 0
            )
            perception_replay_ready = (
                not perception_teacher_required or self.q_teacher_replay.size > 0
            ) and (not perception_student_required or student_q_rows > 0)

        if perception_replay_ready:
            replay_only_keys = tuple(
                key
                for key in (
                    FASTSAC_RAW_OBSERVATION_ROOT,
                    REPLAY_ACTOR_OBSERVATIONS_KEY,
                    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
                )
                if key in rollout.keys()
            )
            # Replay staging above is the sole consumer of the authoritative
            # pre-VecNorm raw copy and flat Actor caches.  PPOVEL.train_adapt
            # reads only the authoritative top-level live observation fields;
            # excluding these replay-only roots prevents make_batch from
            # gathering the large raw-depth copy for every perception minibatch.
            perception_rollout = rollout.exclude(*replay_only_keys)
            adapt_info = self.train_adapt(perception_rollout)
            if bool(getattr(self.cfg, "train_perception", True)):
                self._mark_perception_ema_updated()
        else:
            # An entire provenance ring can be unavailable on the first main
            # rollout (for example when beta selected Teacher everywhere).
            # Focus-pool shortage itself never blocks: the samplers backfill it
            # from the same provenance's uniform ring.
            adapt_info = {
                "adapt/perception_frozen": 0.0,
                "adapt/perception_four_way": 1.0,
                "adapt/perception_replay_deferred": 1.0,
                "adapt/priv_loss": 0.0,
                "adapt/object_loss": 0.0,
                "adapt/grad_norm": 0.0,
                "adapt/depth_grad_norm": 0.0,
            }
        exact_staleness = self._student_perception_drift_metrics()
        self.num_updates += 1
        self.dagger_rollout_count += 1
        self.dagger_environment_steps += int(self.cfg.train_every)

        critic_keys = (
            "critic_loss",
            "critic_loss_1",
            "critic_loss_2",
            "critic_grad_norm",
            "expected_q1_mean",
            "expected_q2_mean",
            "twin_expected_q_disagreement",
            "target_expected_q1_mean",
            "target_expected_q2_mean",
            "projected_target_mean",
            "selected_target_expected_mean",
            "target_distribution_entropy",
            "target_select_q1_fraction",
            "target_select_q2_fraction",
            "left_support_projection_clipping_fraction",
            "right_support_projection_clipping_fraction",
            "target_smoothing_noise_norm",
            "target_noise_free_action_abs_mean",
        )
        actor_keys = (
            "td3_actor_loss",
            "exact_bc_loss",
            "weighted_td3_actor_loss",
            "weighted_bc_loss",
            "total_actor_loss",
            "actor_grad_norm",
            "actor_expected_q1_mean",
            "actor_teacher_replay_fraction",
            "actor_failure_phase_teacher_fraction",
            "actor_failure_phase_student_fraction",
        )
        critic = self._mean_metric_dict(critic_metrics, critic_keys)
        actor = self._mean_metric_dict(actor_metrics, actor_keys)
        q_staleness_tensor = self._aggregate_student_replay_ema_ages(
            q_staleness_age_batches
        )
        actor_staleness_tensor = self._aggregate_student_replay_ema_ages(
            actor_staleness_age_batches
        )
        staleness_values = (
            torch.cat((q_staleness_tensor, actor_staleness_tensor))
            .detach()
            .cpu()
            .tolist()
        )
        metric_count = len(_STUDENT_REPLAY_EMA_AGE_METRIC_KEYS)
        q_staleness = dict(
            zip(
                _STUDENT_REPLAY_EMA_AGE_METRIC_KEYS,
                staleness_values[:metric_count],
            )
        )
        actor_staleness = dict(
            zip(
                _STUDENT_REPLAY_EMA_AGE_METRIC_KEYS,
                staleness_values[metric_count:],
            )
        )

        def rollout_fraction(key):
            value = rollout.get(key, None)
            if value is None or value.numel() == 0:
                return 0.0
            return value.float().mean().item()

        beta = rollout.get(TD3_BETA_KEY, None)
        beta_value = (
            float(self._teacher_mixture_probability())
            if beta is None or beta.numel() == 0
            else beta.float().mean().item()
        )
        info = {
            "td3/method_distributional_td3_teacher_bc_v1": 1.0,
            "td3/prefill_active": 0.0,
            "td3/prefill_rollout_count": self.teacher_prefill_rollout_count,
            "td3/prefill_max_rollouts": int(self.cfg.teacher_prefill_max_rollouts),
            "td3/prefill_target_rows": self.q_teacher_replay.capacity,
            "td3/prefill_progress": self.q_teacher_replay.size
            / max(self.q_teacher_replay.capacity, 1),
            "td3/prefill_environment_steps": self.teacher_prefill_environment_steps,
            "td3/prefill_pending_rows": self._teacher_prefill_pending_rows(),
            "td3/prefill_successful_episodes": (
                self._teacher_prefill_successful_episodes
            ),
            **self._prefill_success_motion_metrics(),
            "td3/prefill_failed_episodes": self._teacher_prefill_failed_episodes,
            "td3/prefill_timeout_episodes": self._teacher_prefill_timeout_episodes,
            "td3/prefill_incomplete_episodes": (
                self._teacher_prefill_incomplete_episodes
            ),
            "td3/prefill_discarded_rows": self._teacher_prefill_discarded_rows,
            "td3/teacher_replay_frozen": float(self._teacher_q_replay_frozen()),
            "td3/teacher_replay_rows_this_rollout": teacher_rows_appended,
            "td3/critic_loss": critic["critic_loss"],
            "td3/critic_loss_1": critic["critic_loss_1"],
            "td3/critic_loss_2": critic["critic_loss_2"],
            "td3/critic_grad_norm": critic["critic_grad_norm"],
            "td3/td3_actor_loss": actor["td3_actor_loss"],
            "td3/exact_bc_loss": actor["exact_bc_loss"],
            "td3/weighted_td3_actor_loss": actor["weighted_td3_actor_loss"],
            "td3/weighted_bc_loss": actor["weighted_bc_loss"],
            "td3/total_actor_loss": actor["total_actor_loss"],
            "td3/actor_grad_norm": actor["actor_grad_norm"],
            "td3/expected_q1_mean": critic["expected_q1_mean"],
            "td3/expected_q2_mean": critic["expected_q2_mean"],
            "td3/actor_expected_q1_mean": actor["actor_expected_q1_mean"],
            "td3/actor_teacher_replay_fraction": actor["actor_teacher_replay_fraction"],
            "td3/actor_failure_phase_teacher_fraction": actor[
                "actor_failure_phase_teacher_fraction"
            ],
            "td3/actor_failure_phase_student_fraction": actor[
                "actor_failure_phase_student_fraction"
            ],
            "td3/target_expected_q1_mean": critic["target_expected_q1_mean"],
            "td3/target_expected_q2_mean": critic["target_expected_q2_mean"],
            "td3/twin_expected_q_disagreement": critic["twin_expected_q_disagreement"],
            "td3/projected_target_mean": critic["projected_target_mean"],
            "td3/selected_target_expected_mean": critic[
                "selected_target_expected_mean"
            ],
            "td3/target_distribution_entropy": critic["target_distribution_entropy"],
            "td3/target_select_q1_fraction": critic["target_select_q1_fraction"],
            "td3/target_select_q2_fraction": critic["target_select_q2_fraction"],
            "td3/left_support_projection_clipping_fraction": critic[
                "left_support_projection_clipping_fraction"
            ],
            "td3/right_support_projection_clipping_fraction": critic[
                "right_support_projection_clipping_fraction"
            ],
            "td3/target_smoothing_noise_norm": critic["target_smoothing_noise_norm"],
            "td3/collector_exploration_noise_norm": collector_noise_norm,
            "td3/actor_update_count": self.actor_update_count,
            "td3/critic_update_count": self.critic_update_count,
            "td3/q_update_to_data_ratio_config": (
                -1.0
                if getattr(self.cfg, "q_update_to_data_ratio", None) is None
                else float(self.cfg.q_update_to_data_ratio)
            ),
            "td3/q_update_row_credit": float(getattr(self, "q_update_row_credit", 0.0)),
            "td3/q_teacher_replay_fraction_config": float(
                self._replay_mix_fractions("q")["uniform_teacher"]
                + self._replay_mix_fractions("q")["failure_teacher"]
            ),
            "td3/q_sampled_rows_this_rollout": (
                len(critic_metrics) * int(self.cfg.q_batch_size)
            ),
            "td3/q_sampled_teacher_rows_this_rollout": (
                len(critic_metrics) * q_teacher_count
            ),
            "td3/q_sampled_student_rows_this_rollout": (
                len(critic_metrics) * q_student_count
            ),
            "td3/actor_updates_this_rollout": len(actor_metrics),
            "td3/critic_updates_this_rollout": len(critic_metrics),
            "td3/replay_ready": float(replay_ready),
            "td3/perception_replay_ready": float(perception_replay_ready),
            "td3/replay_size": self.dagger_replay.size,
            "td3/replay_seen": self.dagger_replay.seen,
            "td3/student_replay_rows_this_rollout": appended,
            "td3/teacher_replay_size": self.q_teacher_replay.size,
            "td3/student_replay_rows": student_q_rows,
            "td3/student_source_fraction": student_selected / max(source_rows, 1),
            "td3/teacher_source_fraction": teacher_selected / max(source_rows, 1),
            "td3/valid_teacher_fraction": valid_labels / max(source_rows, 1),
            "td3/beta": beta_value,
            "td3/rollout_count": self.dagger_rollout_count,
            "td3/environment_steps": self.dagger_environment_steps,
            "td3/truncation_finals": self._last_truncation_finals_used,
            "td3/failure_phase_anchors_this_rollout": failure_anchors_added,
            "td3/failure_phase_episodes": int(
                getattr(self, "_failure_phase_episode_count", 0)
            ),
            "td3/failure_phase_anchors": int(
                getattr(self, "_failure_phase_anchor_count", 0)
            ),
            "td3/failure_phase_histogram_bins": int(
                (getattr(self, "_failure_phase_histogram", torch.empty(0)) > 0)
                .sum()
                .item()
            ),
            "td3/failure_phase_focused_rows": int(
                getattr(self, "_failure_phase_focused_rows", 0)
            ),
            "td3/failure_phase_uniform_fallback_rows": int(
                getattr(self, "_failure_phase_uniform_fallback_rows", 0)
            ),
            "dagger/safe_unsafe_fraction": rollout_fraction(DAGGER_SAFE_UNSAFE_KEY),
            "dagger/safe_teacher_fraction": rollout_fraction(DAGGER_SAFE_TEACHER_KEY),
            "dagger/safe_takeover_fraction": rollout_fraction(DAGGER_SAFE_TAKEOVER_KEY),
            "dagger/safe_release_fraction": rollout_fraction(DAGGER_SAFE_RELEASE_KEY),
            "dagger/beta_teacher_fraction": rollout_fraction(DAGGER_BETA_TEACHER_KEY),
        }
        info.update(adapt_info)
        info.update(
            {
                f"replay/staleness/q_student_ema_generation_age_{key}": value
                for key, value in q_staleness.items()
            }
        )
        info.update(
            {
                f"replay/staleness/actor_student_ema_generation_age_{key}": value
                for key, value in actor_staleness.items()
            }
        )
        info["replay/staleness/current_ema_generation"] = float(
            getattr(self, "_perception_ema_generation", 0)
        )
        info.update(
            {
                f"replay/staleness/exact_{key}": value
                for key, value in exact_staleness.items()
            }
        )
        info["replay/staleness/exact_completed_episodes_total"] = float(
            self._student_perception_drift_completed
        )
        info["replay/staleness/exact_discarded_incomplete_total"] = float(
            self._student_perception_drift_discarded_incomplete
        )
        for purpose in ("q", "actor", "perception"):
            info.update(
                {
                    f"replay/{purpose}/{name}": value
                    for name, value in self._replay_mix_metrics(purpose).items()
                }
            )
        self._last_td3_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if key.startswith("td3/") and isinstance(value, (int, float))
        }
        return info

    def _td3_checkpoint_state(self):
        if self.actor_target is None:
            raise RuntimeError("cannot checkpoint before actor_target initialization")
        return {
            "training_algorithm": TRAINING_ALGORITHM,
            "checkpoint_version": CHECKPOINT_VERSION,
            "actor_backend": ACTOR_BACKEND,
            "critic_learning_semantics": CRITIC_SEMANTICS,
            "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
            "actor_adapt": self.actor_adapt.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "optimizer_resume_state": {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "adapt_optimizer": (
                    None if self.opt_adapt is None else self.opt_adapt.state_dict()
                ),
            },
            "actor_update_count": int(self.actor_update_count),
            "critic_update_count": int(self.critic_update_count),
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
            "collector_exploration_rng_state": (
                self.collector_exploration_rng.get_state()
            ),
            "target_policy_rng_state": self.target_policy_rng.get_state(),
            "teacher_perception_rng_state": self.teacher_perception_rng.get_state(),
            "last_td3_diagnostics": copy.deepcopy(
                getattr(self, "_last_td3_diagnostics", {})
            ),
        }

    def _load_td3_checkpoint_state(self, state, *, load_modules=True):
        if state.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional TD3 Teacher-BC checkpoint")
        if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("distributional TD3 checkpoint version mismatch")
        if self.actor_target is None:
            self.actor_target = copy.deepcopy(self.actor_adapt).requires_grad_(False)
        if load_modules:
            for name in ("actor_adapt", "actor_target", "qnet", "qnet_target"):
                getattr(self, name).load_state_dict(state[name], strict=True)
        optimizers = state.get("optimizer_resume_state")
        if not isinstance(optimizers, dict):
            raise ValueError("TD3 checkpoint lacks optimizer state")
        self.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        self.critic_optimizer.load_state_dict(optimizers["critic_optimizer"])
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
        self.actor_update_count = int(state["actor_update_count"])
        self.critic_update_count = int(state["critic_update_count"])
        self.q_update_row_credit = float(state.get("q_update_row_credit", 0.0))
        q_batch_size = int(getattr(self.cfg, "q_batch_size", 512))
        if not 0.0 <= self.q_update_row_credit < q_batch_size:
            raise ValueError("TD3 checkpoint Q UTD row credit is invalid")
        self.dagger_rollout_count = int(state["dagger_rollout_count"])
        self.dagger_environment_steps = int(state["dagger_environment_steps"])
        self.teacher_prefill_rollout_count = int(state["teacher_prefill_rollout_count"])
        self.teacher_prefill_environment_steps = int(
            state["teacher_prefill_environment_steps"]
        )
        self._teacher_perception_warmup_complete = bool(
            state.get("teacher_perception_warmup_complete", False)
        )
        self._teacher_perception_warmup_updates = int(
            state.get("teacher_perception_warmup_updates", 0)
        )
        self.dagger_rng.set_state(state["dagger_rng_state"])
        self.q_rng.set_state(state["q_rng_state"])
        self.collector_exploration_rng.set_state(
            state["collector_exploration_rng_state"]
        )
        self.target_policy_rng.set_state(state["target_policy_rng_state"])
        self.teacher_perception_rng.set_state(state["teacher_perception_rng_state"])
        self._last_td3_diagnostics = copy.deepcopy(
            state.get("last_td3_diagnostics", {})
        )
        self.actor_target.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore a self-contained TD3 checkpoint for deterministic inference.

        Online replay is intentionally absent from checkpoints, so the normal
        loader continues to reject same-stage *training resume*. Evaluation is
        different: it needs only the serialized module tree and must not load
        optimizer, replay, RNG, or training-counter state.
        """
        if state_dict.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional TD3 Teacher-BC checkpoint")
        if int(state_dict.get("checkpoint_version", -1)) not in (
            PREVIOUS_CHECKPOINT_VERSION,
            CHECKPOINT_VERSION,
        ):
            raise ValueError("distributional TD3 checkpoint version mismatch")
        if state_dict.get("actor_backend") != ACTOR_BACKEND:
            raise ValueError("distributional TD3 actor backend mismatch")
        saved_action_contract = state_dict.get("action_contract")
        if not isinstance(saved_action_contract, Mapping):
            raise ValueError("TD3 inference checkpoint lacks its action contract")
        if not isinstance(saved_action_contract.get("fingerprint"), str):
            raise ValueError("TD3 inference checkpoint lacks a contract fingerprint")
        for key in ("joint_names", "fingerprint"):
            if saved_action_contract.get(key) != self._fastsac_action_contract.get(key):
                raise ValueError(
                    f"TD3 inference checkpoint action contract mismatch at {key!r}"
                )

        if self.actor_target is None:
            self.actor_target = copy.deepcopy(self.actor_adapt).requires_grad_(False)
        failed = PPOVEL.load_state_dict(self, state_dict, strict)
        critical = {
            "actor_adapt",
            "qnet",
            "qnet_target",
            "temporal_depth_gru_ema",
            "object_adapt_ema",
            "adapt_ema",
        }
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                "TD3 inference checkpoint failed to restore critical modules: "
                f"{sorted(missing)}"
            )
        for name in ("actor_adapt", "actor_target", "qnet", "qnet_target"):
            source = state_dict.get(name)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"TD3 inference checkpoint lacks module mapping {name!r}"
                )
            getattr(self, name).load_state_dict(source, strict=True)

        initialization = state_dict.get("perception_initialization")
        if isinstance(initialization, Mapping):
            self._perception_initialization = copy.deepcopy(dict(initialization))
        self._teacher_prefill_complete = True
        self._teacher_perception_warmup_complete = True
        self._freeze_teacher()
        self.actor_adapt.requires_grad_(False).eval()
        self.actor_target.requires_grad_(False).eval()
        self.qnet.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        for name in PRETRAINED_PERCEPTION_MODULES:
            getattr(self, name).requires_grad_(False).eval()
        return failed

    def state_dict(self):
        state = PPOVEL.state_dict(self)
        state.update(self._td3_checkpoint_state())
        state.update(
            {
                "dagger_backend_config": self._checkpoint_config(),
                "action_contract": copy.deepcopy(self._fastsac_action_contract),
                "q_backend_config": self._q_backend_metadata(),
                "teacher_action_semantics": DAGGER_TEACHER_ACTION_SEMANTICS,
                "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
                "replay_observation_semantics": (DAGGER_REPLAY_OBSERVATION_SEMANTICS),
                "replay_transition_semantics": (
                    "exact_issued_action_with_separate_teacher_student_noise_metadata_v1"
                ),
                "replay_resume_semantics": (
                    "fresh_only_online_rings_and_teacher_sidecars_not_serialized_v3"
                ),
                "perception_replay_semantics": PERCEPTION_REPLAY_SEMANTICS,
                "actor_replay_observation_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                    if self._student_collection_actor_cache_enabled()
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "teacher_episode_sidecar_semantics": (
                    TEACHER_EPISODE_SIDECAR_SEMANTICS
                    if self._teacher_episode_cache_enabled()
                    else "disabled"
                ),
                "perception_training_semantics": (
                    ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS
                    if str(
                        getattr(
                            self.cfg,
                            "perception_replay_mode",
                            "legacy_online_student",
                        )
                    )
                    == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
                    else PERCEPTION_REPLAY_SEMANTICS
                ),
                "perception_prefill_warmup_semantics": (
                    PERCEPTION_PREFILL_WARMUP_SEMANTICS
                ),
                "perception_initialization": copy.deepcopy(
                    self._perception_initialization
                ),
                "teacher_prefill_semantics": TEACHER_PREFILL_SEMANTICS,
                "object_geo_replay_semantics": OBJECT_GEO_REPLAY_SEMANTICS,
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
        same_stage = algorithm == TRAINING_ALGORITHM
        if same_stage:
            raise ValueError(
                "TD3 raw-perception replay is fresh-only; same-stage TD3 "
                "checkpoint resume is intentionally unsupported"
            )
        elif algorithm is not None:
            raise ValueError(f"unsupported TD3 source training_algorithm={algorithm!r}")
        elif state_dict.get("last_phase") != "train":
            raise ValueError("fresh TD3 Teacher-BC requires a PPO train checkpoint")

        # ``load_pretrained_perception=false`` means exactly constructor-fresh
        # perception. PPOVEL's compatibility loader would otherwise import the
        # Teacher checkpoint's object/adaptation children opportunistically and
        # make the setting indistinguishable from a partial warm start while
        # reporting false provenance. Snapshot all seven children atomically
        # before the PPO load, then restore them into the same Parameter objects
        # so the already-created optimizer remains valid.
        constructor_perception = None
        if not bool(self.cfg.load_pretrained_perception) and all(
            hasattr(self, name) for name in PRETRAINED_PERCEPTION_MODULES
        ):
            constructor_perception = {
                name: copy.deepcopy(getattr(self, name).state_dict())
                for name in PRETRAINED_PERCEPTION_MODULES
            }
        failed = PPOVEL.load_state_dict(self, state_dict, strict)
        if constructor_perception is not None:
            for name in PRETRAINED_PERCEPTION_MODULES:
                getattr(self, name).load_state_dict(
                    constructor_perception[name], strict=True
                )
        if not same_stage:
            allowed_fresh = {
                "depth_cnn",
                "temporal_depth_gru",
                "temporal_depth_gru_ema",
                "qnet",
                "qnet_target",
            }
            unexpected = set(failed).difference(allowed_fresh)
            if unexpected:
                raise RuntimeError(
                    f"failed to load critical PPO source modules: {sorted(unexpected)}"
                )
            hard_copy_(self.qnet, self.qnet_target)
            self.qnet_target.requires_grad_(False).eval()
            # The PPO physical-command head stays unchanged. Finite support is
            # supplied by the differentiable runtime wrapper, not weight migration.
            self.actor_target = copy.deepcopy(self.actor_adapt).requires_grad_(False)
            self.actor_target.eval()
            self.actor_update_count = 0
            self.critic_update_count = 0
            self.q_update_row_credit = 0.0
            self.dagger_rollout_count = 0
            self.dagger_environment_steps = 0
            self.teacher_prefill_rollout_count = 0
            self.teacher_prefill_environment_steps = 0
            self._teacher_prefill_complete = False
            self._teacher_perception_warmup_complete = False
            self._teacher_perception_warmup_updates = 0
            self._last_teacher_perception_warmup_metrics = {}
            self._teacher_prefill_pending = None
            self._teacher_prefill_successful_episodes = 0
            self._teacher_prefill_successful_by_motion = {}
            self._teacher_prefill_failed_episodes = 0
            self._teacher_prefill_timeout_episodes = 0
            self._teacher_prefill_incomplete_episodes = 0
            self._teacher_prefill_discarded_rows = 0
            self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
            self.q_rng.manual_seed(int(self.cfg.q_seed))
            self.collector_exploration_rng.manual_seed(int(self.cfg.q_seed) + 1)
            self.target_policy_rng.manual_seed(int(self.cfg.q_seed) + 2)
            self.teacher_perception_rng.manual_seed(int(self.cfg.q_seed) + 3)
            if not bool(self.cfg.load_pretrained_perception):
                self._perception_initialization = {
                    "semantics": PERCEPTION_WARMSTART_SEMANTICS,
                    "mode": PERCEPTION_WARMSTART_MODE_FRESH,
                    "loaded": False,
                    "source_path": None,
                    "source_algorithm": None,
                    "source_phase": None,
                    "source_iter": None,
                    "modules": (),
                    "fresh_modules": PRETRAINED_PERCEPTION_MODULES,
                    "trainable": bool(self.cfg.train_perception),
                }
            self._apply_perception_initialization_policy()
        self._freeze_teacher()
        self.actor_adapt.requires_grad_(True).train()
        self.qnet.requires_grad_(True).train()
        self.actor_target.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        return failed
