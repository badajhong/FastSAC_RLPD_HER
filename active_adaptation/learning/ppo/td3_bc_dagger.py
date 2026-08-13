"""C51 distributional TD3 with bounded raw-action Teacher BC.

This module implements ``distributional_td3_teacher_bc_v1``.  Version 4 keeps
the locked VAIC Actor, observation, DAgger, timeout, action, and C51 interfaces,
but makes replay perception-input authoritative: it stores finite raw recurrent
windows and re-encodes them with the current EMA perception modules instead of
persisting collection-time ``priv_pred`` latents.  The duplicate teacher H5
export is deliberately disabled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
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
    FASTSAC_Q_DEFAULT_ACTION_FUSION,
    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_LATE_FUSION_SEMANTICS,
    REPLAY_OBSERVATION_SEMANTICS,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    _build_isolated_q_network,
    _filter_replay_rows,
    _measure_or_clip_grad_norm,
    _q_action_hidden_dim,
    _project_to_execution_support,
    _sac_bootstrap_mask as _bootstrap_mask,
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


TRAINING_ALGORITHM = "distributional_td3_teacher_bc_v1"
CHECKPOINT_VERSION = 4
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

# Authoritative perception replay fields.  These contain only sensor/model
# inputs; collection-time priv_pred/depth_hx/adapt_hx are deliberately absent.
PERCEPTION_DEPTH_U8_KEY = "perception_depth_u8"
PERCEPTION_POLICY_RAW_KEY = "perception_policy_raw"
PERCEPTION_VEL_COMMAND_RAW_KEY = "perception_vel_command_raw"
PERCEPTION_IS_INIT_KEY = "perception_is_init"
REFERENCE_PHASE_KEY = "reference_phase"
FAILURE_PHASE_TEACHER_SOURCE_KEY = "failure_phase_teacher_source"
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
    "raw_input_current_ema_reencode_zero_boundary_burn_in_8_v1"
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
    *_PERCEPTION_REPLAY_FIELDS,
)


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
        if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
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

        # All active replay fields are float32 or bool, but grouping by dtype
        # keeps this helper exact if an audit-only integer field is introduced.
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
    if q_batch_size < 2 or q_batch_size % 2:
        raise ValueError("q_batch_size must be a positive even integer")
    if actor_batch_size < 1 or update_count < 0 or policy_delay < 1:
        raise ValueError("TD3 sample-plan sizes and policy_delay are invalid")
    teacher_actor_replay_fraction = float(teacher_actor_replay_fraction)
    if (
        not math.isfinite(teacher_actor_replay_fraction)
        or not 0.0 <= teacher_actor_replay_fraction <= 1.0
    ):
        raise ValueError("teacher_actor_replay_fraction must be in [0,1]")
    if update_count == 0:
        return ()
    if q_teacher_replay.size < 1:
        raise RuntimeError("Cannot sample Q before a teacher transition exists")
    valid_student_indices = dagger_replay._valid_indices(DAGGER_IS_STUDENT_ACTION_KEY)
    if valid_student_indices.numel() < 1:
        raise RuntimeError("Cannot sample Q before a student transition exists")

    output_device = torch.device(output_device)
    generator_device = torch.device(generator.device)
    if generator_device.type != output_device.type or (
        generator_device.index is not None
        and output_device.index is not None
        and generator_device.index != output_device.index
    ):
        raise ValueError("Replay generator and policy output device must match")
    teacher_count = q_batch_size // 2
    student_count = q_batch_size - teacher_count
    index_draws: list[torch.Tensor] = []
    records: list[tuple[int, int, torch.Tensor, int | None, int | None]] = []
    for update_index in range(update_count):
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
            actor_teacher_count = min(
                actor_batch_size,
                int(math.floor(actor_batch_size * teacher_actor_replay_fraction + 0.5)),
            )
            actor_main_count = actor_batch_size - actor_teacher_count
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
        teacher_indices = cpu_draws[teacher_draw]
        student_indices = valid_student_indices.index_select(0, cpu_draws[student_draw])
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
    q_lr: float = 3e-5
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    q_tau: float = 0.005
    q_max_grad_norm: float = 1.0
    q_batch_size: int = 512
    q_updates_per_rollout: int = 128
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
        self.qnet = _build_isolated_q_network(
            self._q_critic_dim,
            self.action_dim,
            cfg.q_hidden_dim,
            cfg.q_num_atoms,
            cfg.q_v_min,
            cfg.q_v_max,
            cfg.q_layer_norm,
            device,
            cfg.q_seed,
            cfg.q_action_fusion,
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
        self._truncation_final_batches = []
        self._last_truncation_finals_used = 0
        self._perception_replay_history = None
        self._perception_replay_history_count = 0
        self._replay_object_geo = None
        self._replay_object_geo_fingerprint = None
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
        self._teacher_prefill_successful_episodes = 0
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
        if (
            not bool(cfg.q_layer_norm)
            or str(cfg.q_action_fusion) != FASTSAC_Q_DEFAULT_ACTION_FUSION
        ):
            raise ValueError(
                "distributional TD3 requires the locked late-fusion LayerNorm Q"
            )
        if str(cfg.q_action_coordinates) != "raw_joint_command" or not bool(
            cfg.q_normalize_actions
        ):
            raise ValueError(
                "distributional TD3 requires normalized raw-joint-command Q actions"
            )
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
            "failure_phase_lookback_steps",
            "failure_phase_samples_per_failure",
            "failure_phase_num_bins",
        )
        for name in positive_integers:
            value = getattr(cfg, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
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
        if int(cfg.q_teacher_buffer_capacity) < int(cfg.td3_learning_starts):
            raise ValueError("q_teacher_buffer_capacity must cover td3_learning_starts")
        if int(cfg.q_batch_size) % 2:
            raise ValueError("q_batch_size must be even for exact 50/50 replay")
        if not math.isclose(
            float(cfg.q_teacher_replay_ratio), 0.5, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Phase 1 keeps exact 50/50 Teacher/Student Q replay")
        for name in (
            "teacher_actor_replay_fraction",
            "teacher_perception_replay_fraction",
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

    @torch.no_grad()
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
        required = set(_Q_REPLAY_FIELDS).union(
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
                    is_init_grid.long(),
                ),
                dim=-1,
            ).reshape(-1, 6)
            if event_metadata.device.type != "cpu":
                event_metadata = event_metadata.to("cpu")
            event_env = event_metadata[:, 0]
            event_step = event_metadata[:, 1]
            done = event_metadata[:, 2].bool()
            terminated = event_metadata[:, 3].bool()
            command_finished = event_metadata[:, 4].bool()
            is_init = event_metadata[:, 5].bool()

        maximum_env = int(event_env.max().item())
        if int(event_env.min().item()) < 0:
            raise ValueError("Teacher prefill environment index is negative")
        self._ensure_teacher_prefill_pending(maximum_env + 1)
        pending = self._teacher_prefill_pending
        if pending is None:  # pragma: no cover - guarded by initializer above
            raise RuntimeError("Teacher prefill pending storage was not initialized")

        replay_device = self.q_teacher_replay.device
        payload = {
            key: transitions[key].detach().to(replay_device) for key in _Q_REPLAY_FIELDS
        }
        committed_rows = 0
        discarded_rows = 0

        def _append_segment(env_index: int, start_step: int, stop_step: int) -> None:
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
            pending[env_index].append(
                {key: value.index_select(0, selected) for key, value in payload.items()}
            )

        def _discard(env_index: int, kind: str) -> int:
            rows = self._prefill_pending_row_count(pending[env_index])
            pending[env_index].clear()
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
            if chunks:
                episode = {
                    key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                    for key in _Q_REPLAY_FIELDS
                }
                rows = self.q_teacher_replay.extend(episode)
            else:
                rows = 0
            chunks.clear()
            self._teacher_prefill_successful_episodes += 1
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
                    if pending[int(env_index)]:
                        discarded_rows += _discard(int(env_index), "incomplete")
                    segment_start_step = step
                if not bool(done[row]):
                    continue
                _append_segment(int(env_index), segment_start_step, step + 1)
                if bool(terminated[row]):
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
        discarded = 0
        for chunks in pending:
            rows = self._prefill_pending_row_count(chunks)
            if rows:
                discarded += rows
                self._teacher_prefill_incomplete_episodes += 1
                self._teacher_prefill_discarded_rows += rows
                chunks.clear()
        return discarded

    def _teacher_prefill_pending_rows(self) -> int:
        pending = self._teacher_prefill_pending or ()
        return sum(self._prefill_pending_row_count(chunks) for chunks in pending)

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
        getattr(self, "_teacher_phase_device_cache", {}).clear()
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
        teacher_count = batch_size // 2
        student_count = batch_size - teacher_count
        student_rows = self.dagger_replay._valid_indices(DAGGER_IS_STUDENT_ACTION_KEY)
        if student_rows.numel() < 1:
            raise RuntimeError("Cannot prefetch before a Student row exists")
        actor_batch_size = int(self.cfg.dagger_batch_size)
        actor_main_count, _, actor_focused_count = _source_counts(
            actor_batch_size,
            float(self.cfg.teacher_actor_replay_fraction),
            float(self.cfg.failure_phase_teacher_fraction),
        )
        actor_teacher_count = actor_batch_size - actor_main_count
        q_focused_count = _source_counts(
            teacher_count,
            1.0,
            float(self.cfg.failure_phase_teacher_fraction),
        )[2]
        generator_device = torch.device(self.q_rng.device)
        if update_count == 0:
            return ()
        index_draws: list[torch.Tensor] = []
        records = []
        for update_index in range(update_count):
            teacher_indices, teacher_focused = self._draw_teacher_indices(
                teacher_count,
                self.q_rng,
                focused_count=q_focused_count,
            )
            teacher_draw = len(index_draws)
            index_draws.append(teacher_indices)
            student_draw = len(index_draws)
            index_draws.append(
                torch.randint(
                    0,
                    student_rows.numel(),
                    (student_count,),
                    device=generator_device,
                    generator=self.q_rng,
                )
            )
            permutation = torch.randperm(
                batch_size, device=generator_device, generator=self.q_rng
            )

            actor_draw = None
            actor_teacher_draw = None
            actor_teacher_focused = None
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
                    actor_draw = len(index_draws)
                    index_draws.append(
                        torch.randint(
                            0,
                            student_rows.numel(),
                            (actor_main_count,),
                            device=generator_device,
                            generator=self.q_rng,
                        )
                    )
            records.append(
                (
                    teacher_draw,
                    teacher_focused,
                    student_draw,
                    permutation,
                    actor_draw,
                    actor_teacher_draw,
                    actor_teacher_focused,
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
        ) in records:
            student_indices = student_rows.index_select(0, cpu_draws[student_draw])
            actor_indices = (
                None
                if actor_draw is None
                else student_rows.index_select(0, cpu_draws[actor_draw])
            )
            actor_teacher_indices = (
                None if actor_teacher_draw is None else cpu_draws[actor_teacher_draw]
            )
            plans.append(
                _TD3ReplaySamplePlan(
                    teacher_indices=cpu_draws[teacher_draw],
                    student_indices=student_indices,
                    permutation=permutation,
                    actor_indices=actor_indices,
                    actor_teacher_indices=actor_teacher_indices,
                    teacher_focused=teacher_focused,
                    actor_teacher_focused=actor_teacher_focused,
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

    @torch.no_grad()
    def _register_replay_object_geo(self, td: TensorDict) -> None:
        geometry = td[OBJECT_GEO_KEY]
        width = int(self.observation_spec[OBJECT_GEO_KEY].shape[-1])
        flat = geometry.reshape(-1, width)
        reference = flat[0].detach()
        if not torch.equal(flat, reference.expand_as(flat)):
            raise RuntimeError(
                "TD3 raw replay requires object_geo_ to be constant within a run"
            )
        if self._replay_object_geo is None:
            # Rollout collection runs under torch.inference_mode().  A clone
            # made in that context remains an inference tensor and cannot be
            # saved by autograd when Teacher replay later differentiates the
            # predicted object transform.  Disable inference mode explicitly
            # so this run-constant cache is an ordinary tensor.
            with torch.inference_mode(False):
                self._replay_object_geo = reference.clone()
            payload = reference.float().contiguous().cpu().numpy().tobytes()
            self._replay_object_geo_fingerprint = hashlib.sha256(payload).hexdigest()
        elif not torch.equal(
            reference.to(self._replay_object_geo), self._replay_object_geo
        ):
            raise RuntimeError("object_geo_ changed after replay initialization")

    @torch.no_grad()
    def _raw_perception_values(self, td: TensorDict) -> dict[str, torch.Tensor]:
        self._register_replay_object_geo(td)
        return {
            PERCEPTION_DEPTH_U8_KEY: _encode_replay_depth_u8(
                self._replay_source(td, DEPTH_KEY)
            ),
            PERCEPTION_POLICY_RAW_KEY: self._replay_source(td, OBS_KEY).detach(),
            PERCEPTION_VEL_COMMAND_RAW_KEY: self._replay_source(
                td, VEL_CMD_KEY
            ).detach(),
            PERCEPTION_IS_INIT_KEY: td["is_init"].detach().bool(),
        }

    @torch.no_grad()
    def _prepare_raw_final_state(self, td: TensorDict) -> dict[str, torch.Tensor]:
        return {
            **self._raw_perception_values(td),
            "next_critic_observations": self._cat_replay_sources(
                td, self.q_critic_keys
            ).clone(),
        }

    @torch.no_grad()
    def capture_truncation_final_observations(self, td: TensorDict, step: int):
        """Capture the true pre-reset timeout input, never the reset carry."""
        if not self._collect_dagger_replay_this_rollout():
            self._truncation_final_batches = []
            self._last_truncation_finals_used = 0
            return
        truncations = _vaic_truncation_mask(td).reshape(-1).bool()
        if not truncations.any():
            return
        indices = truncations.nonzero(as_tuple=False).squeeze(-1)
        values = self._prepare_raw_final_state(td["next"][indices].clone())
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
        burn_in = int(self.cfg.perception_replay_burn_in)
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        raw_current = self._raw_perception_values(td)

        history_count = int(self._perception_replay_history_count)
        if self._perception_replay_history is None:
            history = {key: value[:, :0] for key, value in raw_current.items()}
        else:
            history = self._perception_replay_history
            if any(value.shape[0] != int(n) for value in history.values()):
                raise RuntimeError("raw perception replay environment count changed")

        sequence = {}
        for key, current_value in raw_current.items():
            final_value = final_batch[key].reshape(int(n), *current_value.shape[2:])
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
        used_truncation_finals = 0

        for step in range(int(t)):
            current = td[:, step]
            position = history_count + step
            if position < burn_in:
                continue
            window_values = {
                key: value[:, position - burn_in : position + 2]
                for key, value in sequence.items()
            }
            if any(value.shape[1] != burn_in + 2 for value in window_values.values()):
                raise RuntimeError("raw perception replay window has invalid length")

            if step + 1 < int(t):
                next_critic = self._cat_replay_sources(
                    td[:, step + 1], self.q_critic_keys
                ).reshape(int(n), self._q_critic_dim)
            else:
                next_critic = final_batch["next_critic_observations"].reshape(
                    int(n), self._q_critic_dim
                )

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
                    for key in _PERCEPTION_REPLAY_FIELDS:
                        window_values[key] = window_values[key].clone()
                        window_values[key][env_indices, -1] = truncation_finals[key][
                            selected
                        ]

            transitions = {
                "critic_observations": self._cat_replay_sources(
                    current, self.q_critic_keys
                ).reshape(int(n), self._q_critic_dim),
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
                "truncations": _vaic_truncation_mask(current).reshape(int(n)).bool(),
                "discounts": current["next", "discount"].reshape(int(n)),
                "next_critic_observations": next_critic,
                REFERENCE_PHASE_KEY: self._reference_phase(current).reshape(int(n)),
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
        if self._replay_object_geo is None:
            raise RuntimeError("raw perception replay has no object geometry contract")

        depth_u8 = batch[PERCEPTION_DEPTH_U8_KEY]
        policy_raw = batch[PERCEPTION_POLICY_RAW_KEY]
        vel_raw = batch[PERCEPTION_VEL_COMMAND_RAW_KEY]
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

        snapshot = self._vecnorm_snapshot()
        current_chunks: list[torch.Tensor] = []
        next_chunks: list[torch.Tensor] = []
        microbatch = int(self.cfg.perception_encode_microbatch_size)
        geometry = self._replay_object_geo.to(
            device=depth_u8.device, dtype=policy_raw.dtype
        )
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
                count = stop - start
                td = TensorDict(
                    {
                        DEPTH_KEY: depth,
                        OBS_KEY: policy,
                        VEL_CMD_KEY: vel,
                        OBJECT_GEO_KEY: geometry.view(1, 1, -1).expand(
                            count, window_length, -1
                        ),
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
        """Normalize critic inputs and materialize only needed Actor states."""
        prepared = PPOBCDaggerFinetune._prepare_dagger_learning_batch(self, batch)
        include_current = DAGGER_REPLAY_TEACHER_ACTIONS in batch
        include_next = "next_critic_observations" in batch
        prepared.update(
            self._reencode_perception_windows(
                prepared,
                include_current=include_current,
                include_next=include_next,
            )
        )
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
            batch["next_critic_observations"], smoothed_q_action
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
            batch["critic_observations"], self._q_action_input(batch["actions"])
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
        q_action = self._q_action_input(prediction_action)

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

    def _q_backend_metadata(self):
        return {
            "actor_obs_keys": list(self.q_actor_keys),
            "critic_obs_keys": list(self.q_critic_keys),
            "actor_obs_dim": self._q_actor_dim,
            "critic_obs_dim": self._q_critic_dim,
            "action_dim": self.action_dim,
            "hidden_dim": int(self.cfg.q_hidden_dim),
            "q_action_fusion": str(self.cfg.q_action_fusion),
            "q_action_hidden_dim": _q_action_hidden_dim(
                self.cfg.q_hidden_dim, self.cfg.q_action_fusion
            ),
            "q_action_fusion_semantics": FASTSAC_Q_LATE_FUSION_SEMANTICS,
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
            ),
            "actor_teacher_replay_fraction": float(
                self.cfg.teacher_actor_replay_fraction
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
            "q_lr",
            "q_weight_decay",
            "q_seed",
            "q_tau",
            "q_max_grad_norm",
            "q_batch_size",
            "q_updates_per_rollout",
            "q_teacher_replay_ratio",
            "q_teacher_buffer_capacity",
            "perception_replay_burn_in",
            "perception_encode_microbatch_size",
            "teacher_perception_batch_size",
            "teacher_perception_warmup_steps",
            "perception_depth_codec",
            "load_pretrained_perception",
            "perception_checkpoint_path",
            "train_perception",
            "save_teacher_buffer",
        )
        return {
            **{name: getattr(self.cfg, name) for name in names},
            "method": TRAINING_ALGORITHM,
            "actor_output": "ppo_physical_proposal_tanh_bounded_raw_joint_command",
            "bc_loss": "joint_normalized_raw_mean_teacher_smooth_l1",
        }

    def _sample_balanced_q_batch(self, sample_plan: _TD3ReplaySamplePlan | None = None):
        curriculum_enabled = hasattr(self.cfg, "failure_phase_teacher_fraction")
        if sample_plan is None and not curriculum_enabled:
            return PPOBCDaggerFinetune._sample_balanced_q_batch(self)
        batch_size = int(self.cfg.q_batch_size)
        teacher_count = batch_size // 2
        student_count = batch_size - teacher_count
        q_fields = tuple(self.q_teacher_replay.data)
        if sample_plan is None:
            failure_fraction = float(
                getattr(self.cfg, "failure_phase_teacher_fraction", 0.0)
            )
            focused_count = _source_counts(teacher_count, 1.0, failure_fraction)[2]
            teacher_indices, focused_teacher = self._sample_teacher_indices(
                teacher_count,
                self.q_rng,
                focused_count=focused_count,
            )
            teacher = _sample_replay_by_indices(
                self.q_teacher_replay,
                teacher_indices,
                self.device,
                fields=q_fields,
            )
            student = self.dagger_replay.sample(
                student_count,
                self.device,
                self.q_rng,
                valid_key=DAGGER_IS_STUDENT_ACTION_KEY,
                fields=q_fields,
            )
            focused_teacher = focused_teacher.to(self.device)
            permutation = torch.randperm(
                batch_size, device=self.device, generator=self.q_rng
            )
        else:
            if sample_plan.teacher_indices.numel() != teacher_count:
                raise ValueError("Teacher replay sample plan has the wrong row count")
            if sample_plan.student_indices.numel() != student_count:
                raise ValueError("Student replay sample plan has the wrong row count")
            teacher = _sample_replay_by_indices(
                self.q_teacher_replay,
                sample_plan.teacher_indices,
                self.device,
                fields=q_fields,
            )
            student = _sample_replay_by_indices(
                self.dagger_replay,
                sample_plan.student_indices,
                self.device,
                fields=q_fields,
            )
            focused_teacher = (
                torch.zeros(teacher_count, dtype=torch.bool, device=self.device)
                if sample_plan.teacher_focused is None
                else sample_plan.teacher_focused.to(self.device)
            )
            permutation = sample_plan.permutation
            if permutation.shape != (batch_size,):
                raise ValueError("Q replay sample plan has the wrong permutation shape")
        mixed = {key: torch.cat((teacher[key], student[key]), dim=0) for key in teacher}
        mixed[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.cat(
            (
                torch.ones(teacher_count, dtype=torch.bool, device=self.device),
                torch.zeros(student_count, dtype=torch.bool, device=self.device),
            )
        )
        if curriculum_enabled:
            mixed[FAILURE_PHASE_TEACHER_SOURCE_KEY] = torch.cat(
                (
                    focused_teacher,
                    torch.zeros(student_count, dtype=torch.bool, device=self.device),
                )
            )
        return {key: value[permutation] for key, value in mixed.items()}

    def _sample_actor_batch(
        self,
        indices: torch.Tensor | None = None,
        teacher_indices: torch.Tensor | None = None,
        teacher_focused: torch.Tensor | None = None,
    ):
        """Sample one mixed Actor batch without changing the zero-share path."""
        main_fields = (
            "critic_observations",
            DAGGER_REPLAY_TEACHER_ACTIONS,
            DAGGER_TEACHER_ACTION_VALID_KEY,
            *_PERCEPTION_REPLAY_FIELDS,
        )
        batch_size = int(self.cfg.dagger_batch_size)
        fraction = float(getattr(self.cfg, "teacher_actor_replay_fraction", 0.0))
        curriculum_enabled = hasattr(self.cfg, "failure_phase_teacher_fraction")
        failure_fraction = float(
            getattr(self.cfg, "failure_phase_teacher_fraction", 0.0)
        )
        main_count, uniform_teacher_count, focused_teacher_count = _source_counts(
            batch_size,
            fraction,
            failure_fraction if curriculum_enabled else 0.0,
        )
        teacher_count = uniform_teacher_count + focused_teacher_count

        # Keep the old operation and RNG sequence byte-for-byte at the
        # backward-compatible default.
        if teacher_count == 0:
            if teacher_indices is not None:
                raise ValueError("Actor replay plan unexpectedly contains Teacher rows")
            if teacher_focused is not None:
                raise ValueError("Actor replay plan unexpectedly marks focused rows")
            if indices is None:
                batch = self.dagger_replay.sample(
                    batch_size,
                    self.device,
                    self.q_rng,
                    valid_key=(
                        DAGGER_IS_STUDENT_ACTION_KEY if curriculum_enabled else None
                    ),
                    fields=main_fields,
                )
            else:
                if indices.numel() != batch_size:
                    raise ValueError("Actor replay sample plan has the wrong row count")
                batch = self.dagger_replay.sample_by_indices(
                    indices, self.device, fields=main_fields
                )
            if curriculum_enabled:
                batch[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.zeros(
                    batch_size, dtype=torch.bool, device=self.device
                )
                batch[FAILURE_PHASE_TEACHER_SOURCE_KEY] = torch.zeros(
                    batch_size, dtype=torch.bool, device=self.device
                )
            return self._prepare_dagger_learning_batch(batch)

        teacher_fields = (
            "critic_observations",
            "actions",
            *_PERCEPTION_REPLAY_FIELDS,
        )
        if teacher_indices is None:
            if curriculum_enabled:
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
        teacher[DAGGER_REPLAY_TEACHER_ACTIONS] = teacher.pop("actions")
        teacher[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(
            teacher_count, dtype=torch.bool, device=self.device
        )

        main = None
        if main_count:
            if indices is None:
                main = self.dagger_replay.sample(
                    main_count,
                    self.device,
                    self.q_rng,
                    valid_key=(
                        DAGGER_IS_STUDENT_ACTION_KEY if curriculum_enabled else None
                    ),
                    fields=main_fields,
                )
            else:
                if indices.numel() != main_count:
                    raise ValueError("Actor replay sample plan has the wrong row count")
                main = self.dagger_replay.sample_by_indices(
                    indices, self.device, fields=main_fields
                )
        elif indices is not None and indices.numel() != 0:
            raise ValueError("Actor replay sample plan has the wrong row count")

        if main is None:
            batch = teacher
        else:
            batch = {
                key: torch.cat((teacher[key], main[key]), dim=0) for key in main_fields
            }
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
        return self._prepare_dagger_learning_batch(batch)

    def _teacher_perception_replay_loss(self) -> dict[str, torch.Tensor]:
        """Recompute one supervised perception loss from frozen Teacher inputs.

        The replay is authoritative only for model inputs.  Learned recurrent
        states and ``priv_pred`` are never stored: the online depth/object/adapt
        modules are rerun from a zero boundary through the raw burn-in window.
        The privileged target is reconstructed from the current replay state.
        """
        if self.q_teacher_replay.size < 1:
            raise RuntimeError(
                "teacher_perception_replay_fraction requires non-empty q_teacher_replay"
            )
        if self._replay_object_geo is None:
            raise RuntimeError("Teacher perception replay has no object geometry")

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
        is_init = batch[PERCEPTION_IS_INIT_KEY]
        row_count, window_length = depth_u8.shape[:2]
        expected_length = int(self.cfg.perception_replay_burn_in) + 2
        if int(window_length) != expected_length:
            raise ValueError(
                f"Teacher perception window has length {window_length}; "
                f"expected {expected_length}"
            )

        snapshot = self._vecnorm_snapshot()
        depth = self._normalize_replay_value(
            DEPTH_KEY, _decode_replay_depth_u8(depth_u8), snapshot
        )
        policy = self._normalize_replay_value(OBS_KEY, policy_raw, snapshot)
        vel = self._normalize_replay_value(VEL_CMD_KEY, vel_raw, snapshot)
        geometry = self._replay_object_geo.to(device=depth.device, dtype=policy.dtype)

        target = TensorDict(
            {
                OBS_PRIV_KEY: critic[OBS_PRIV_KEY],
                OBJECT_KEY: critic[OBJECT_KEY],
                OBJECT_GEO_KEY: geometry.view(1, -1).expand(row_count, -1),
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
                OBJECT_GEO_KEY: geometry.view(1, 1, -1).expand(
                    row_count, window_length, -1
                ),
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
        object_loss = (object_error * valid.unsqueeze(-1)).mean()
        priv_loss = (priv_error * valid.unsqueeze(-1)).mean()
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

    @set_recurrent_mode(True)
    def train_adapt(self, tensordict: TensorDict):
        """Mix frozen-Teacher raw replay into the existing adaptation steps."""
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

    def train_op(self, tensordict):
        """Collect locked transitions and run Phase-1 TD3/BC updates only."""
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
                    q_teacher = _stage_rows(_Q_REPLAY_FIELDS, teacher_indices)
                    teacher_rows_appended += self.q_teacher_replay.extend(q_teacher)
                    del q_teacher, teacher_indices
                del teacher_executed
            if not teacher_prefill_active:
                # The main source in the 50/35/15 curriculum is Student-only.
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
                    student_staged = _stage_rows(student_fields, student_indices)
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
                "td3/actor_update_count": self.actor_update_count,
                "td3/critic_update_count": self.critic_update_count,
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
            self._last_td3_diagnostics = {
                key: float(value)
                for key, value in info.items()
                if key.startswith("td3/") and isinstance(value, (int, float))
            }
            return info

        critic_metrics: list[dict[str, torch.Tensor]] = []
        actor_metrics: list[dict[str, torch.Tensor]] = []
        student_q_rows = self.dagger_replay.valid_count(DAGGER_IS_STUDENT_ACTION_KEY)
        learning_starts = int(self.cfg.td3_learning_starts)
        replay_ready = (
            self.q_teacher_replay.size >= learning_starts
            and student_q_rows >= learning_starts
        )
        if replay_ready:
            q_updates = int(self.cfg.q_updates_per_rollout)
            sample_plans = None
            if (
                isinstance(self.dagger_replay, _TD3DeviceReplay)
                and isinstance(self.q_teacher_replay, _TD3DeviceReplay)
                and self.dagger_replay.device.type == "cpu"
                and self.q_teacher_replay.device.type == "cpu"
            ):
                sample_plans = (
                    self._prefetch_curriculum_sample_plans(q_updates)
                    if hasattr(self.cfg, "failure_phase_teacher_fraction")
                    else _prefetch_td3_replay_sample_plans(
                        self.dagger_replay,
                        self.q_teacher_replay,
                        q_batch_size=int(self.cfg.q_batch_size),
                        actor_batch_size=int(self.cfg.dagger_batch_size),
                        update_count=q_updates,
                        policy_delay=int(self.cfg.policy_delay),
                        critic_update_count=int(self.critic_update_count),
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
                    delayed = self._maybe_delayed_actor_and_targets(
                        self._sample_actor_batch(
                            actor_indices,
                            actor_teacher_indices,
                            (
                                None
                                if sample_plan is None
                                else sample_plan.actor_teacher_focused
                            ),
                        )
                    )
                    if delayed is None:
                        raise RuntimeError("scheduled delayed Actor update was skipped")
                    actor_metrics.append(delayed)
            del sample_plans

        # Retain the existing supervised student-perception update unchanged.
        adapt_info = self.train_adapt(rollout.copy())
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
        )
        critic = self._mean_metric_dict(critic_metrics, critic_keys)
        actor = self._mean_metric_dict(actor_metrics, actor_keys)

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
            "td3/actor_updates_this_rollout": len(actor_metrics),
            "td3/critic_updates_this_rollout": len(critic_metrics),
            "td3/replay_ready": float(replay_ready),
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
        if int(state_dict.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
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
                    "fresh_only_online_raw_perception_rings_not_serialized_v2"
                ),
                "perception_replay_semantics": PERCEPTION_REPLAY_SEMANTICS,
                "perception_prefill_warmup_semantics": (
                    PERCEPTION_PREFILL_WARMUP_SEMANTICS
                ),
                "perception_initialization": copy.deepcopy(
                    self._perception_initialization
                ),
                "teacher_prefill_semantics": TEACHER_PREFILL_SEMANTICS,
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
