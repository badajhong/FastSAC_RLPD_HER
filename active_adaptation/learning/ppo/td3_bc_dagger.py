"""C51 distributional TD3 with the exact existing Teacher-BC objective.

This module implements ``distributional_td3_teacher_bc_v1``.  Version 2 keeps
the locked VAIC Actor, observation, DAgger, timeout, action, and C51 interfaces,
but makes replay perception-input authoritative: it stores finite raw recurrent
windows and re-encodes them with the current EMA perception modules instead of
persisting collection-time ``priv_pred`` latents.  The duplicate teacher H5
export is deliberately disabled.
"""

from __future__ import annotations

import copy
import math
import hashlib
from dataclasses import dataclass

import torch
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
    hard_copy_,
)
from .fastsac_vel import (
    FASTSAC_Q_DEFAULT_ACTION_FUSION,
    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_LATE_FUSION_SEMANTICS,
    FASTSAC_REFERENCE_EPS,
    REPLAY_OBSERVATION_SEMANTICS,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    _build_isolated_q_network,
    _fastsac_action_center_to_latent,
    _filter_replay_rows,
    _measure_or_clip_grad_norm,
    _project_to_execution_support,
    _q_action_hidden_dim,
    _sac_bootstrap_mask as _bootstrap_mask,
    _vaic_truncation_mask,
    _vaic_action_contract_metadata,
    _vaic_nominal_action_coordinates,
    _validate_action_safety_clip,
)
from .ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_MAX_KEY,
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_CONTROL_MODES,
    DAGGER_CONTROL_SEMANTICS,
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
    PPO_BC_DAGGER_ACTOR_BACKEND,
    _DaggerRolloutPolicy,
    _DeviceReplay,
    _LatentStudentRolloutPolicy,
    PPOBCDaggerFinetune,
    _linear_teacher_probability,
    _normalized_action_discrepancy,
    _resolve_replay_device,
    _valid_teacher_action_rows,
)
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    PPOConfig,
    PPOVEL,
    PRIV_PRED_KEY,
    VEL_CMD_KEY,
    set_recurrent_mode,
)


TRAINING_ALGORITHM = "distributional_td3_teacher_bc_v1"
CHECKPOINT_VERSION = 2
ACTOR_BACKEND = PPO_BC_DAGGER_ACTOR_BACKEND
CRITIC_SEMANTICS = (
    "deterministic_target_actor_q_coordinate_clipped_smoothing_"
    "lower_expected_complete_c51_distribution_projection_v1"
)
ACTOR_LEARNING_SEMANTICS = (
    "expected_online_q1_plus_exact_inverse_tanh_teacher_huber_bc_v1"
)

TD3_NOISE_FREE_STUDENT_ACTION_KEY = "td3_noise_free_student_action"
TD3_EXPLORATORY_STUDENT_ACTION_KEY = "td3_exploratory_student_action"
TD3_COLLECTOR_NOISE_KEY = "td3_collector_q_noise"
TD3_BETA_KEY = "td3_beta_probability"
TEACHER_PREFILL_SEMANTICS = (
    "forced_valid_teacher_q_replay_only_then_frozen_before_main_dagger_v2"
)

# Authoritative perception replay fields.  These contain only sensor/model
# inputs; collection-time priv_pred/depth_hx/adapt_hx are deliberately absent.
PERCEPTION_DEPTH_U8_KEY = "perception_depth_u8"
PERCEPTION_POLICY_RAW_KEY = "perception_policy_raw"
PERCEPTION_VEL_COMMAND_RAW_KEY = "perception_vel_command_raw"
PERCEPTION_IS_INIT_KEY = "perception_is_init"
PERCEPTION_REPLAY_SEMANTICS = (
    "raw_input_current_ema_reencode_zero_boundary_burn_in_8_v1"
)
PERCEPTION_DEPTH_CODEC = "uint8_div_100_v1"

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
    *_PERCEPTION_REPLAY_FIELDS,
)


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


def _exact_teacher_bc_loss(
    prediction_latent: torch.Tensor,
    teacher_action: torch.Tensor,
    valid_mask: torch.Tensor,
    action_center: torch.Tensor,
    action_scale: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """Pure form of the authoritative Teacher-BC latent SmoothL1 loss."""
    valid = valid_mask.reshape(-1).bool()
    if prediction_latent.shape != teacher_action.shape:
        raise ValueError("BC prediction and Teacher action shapes must match")
    if prediction_latent.shape[0] != valid.numel():
        raise ValueError("BC validity mask does not match batch rows")
    if not valid.any():
        return prediction_latent.sum() * 0.0
    selected_prediction = prediction_latent[valid]
    if not torch.isfinite(selected_prediction).all():
        raise RuntimeError("BC student latent contains non-finite values")
    selected_teacher = teacher_action[valid].detach()
    target_latent = _fastsac_action_center_to_latent(
        selected_teacher,
        action_scale.to(selected_teacher),
        action_center.to(selected_teacher),
        FASTSAC_REFERENCE_EPS,
    ).detach()
    return F.smooth_l1_loss(
        selected_prediction,
        target_latent,
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
    project_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add optional noise in Q coordinates to Student-selected rows only."""
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
        raise ValueError("execution bounds do not match action dimension")
    if noise_std == 0.0 or noise_clip == 0.0 or not selected.any():
        exploratory_student = student_action.clone()
        issued_action = torch.where(
            selected.unsqueeze(-1), exploratory_student, teacher_action
        )
        return issued_action, exploratory_student, torch.zeros_like(student_action)
    q_student = ((student_action - center) / scale) * gain
    q_low = ((low - center) / scale) * gain
    q_high = ((high - center) / scale) * gain
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
    noisy_q = torch.maximum(torch.minimum(q_student + sampled, q_high), q_low)
    exploratory_student = (noisy_q / gain) * scale + center
    if project_fn is None:
        exploratory_student = _project_to_execution_support(
            exploratory_student,
            low,
            high,
            float(torch.maximum(low.abs(), high.abs()).max()),
        )
    else:
        exploratory_student = project_fn(exploratory_student)
    exploratory_student = torch.where(
        selected.unsqueeze(-1), exploratory_student, student_action
    )
    actual_noise = (((exploratory_student - center) / scale) * gain) - q_student
    actual_noise = torch.where(
        selected.unsqueeze(-1), actual_noise, torch.zeros_like(actual_noise)
    )
    issued_action = torch.where(
        selected.unsqueeze(-1), exploratory_student, teacher_action
    )
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
    records: list[tuple[int, int, torch.Tensor, int | None]] = []
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
        if (int(critic_update_count) + update_index + 1) % policy_delay == 0:
            actor_draw = len(index_draws)
            index_draws.append(
                torch.randint(
                    0,
                    dagger_replay.size,
                    (actor_batch_size,),
                    device=generator_device,
                    generator=generator,
                )
            )
        records.append((teacher_draw, student_draw, permutation, actor_draw))

    lengths = [int(draw.numel()) for draw in index_draws]
    packed_indices = torch.cat(index_draws)
    if packed_indices.device.type != "cpu":
        packed_indices = packed_indices.to("cpu")
    cpu_draws = packed_indices.split(lengths)
    plans = []
    for teacher_draw, student_draw, permutation, actor_draw in records:
        teacher_indices = cpu_draws[teacher_draw]
        student_indices = valid_student_indices.index_select(0, cpu_draws[student_draw])
        actor_indices = None if actor_draw is None else cpu_draws[actor_draw]
        plans.append(
            _TD3ReplaySamplePlan(
                teacher_indices=teacher_indices,
                student_indices=student_indices,
                permutation=permutation,
                actor_indices=actor_indices,
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
    dagger_safe_takeover_rms: float = 0.006
    dagger_safe_release_rms: float = 0.004
    dagger_safe_min_teacher_steps: int = 8
    dagger_safe_zero_iteration: int | None = None
    dagger_beta_start: float = 1.0
    dagger_beta_end: float = 0.0
    dagger_beta_decay_rollouts: int = 1800
    dagger_beta_zero_iteration: int | None = None
    dagger_seed: int = 0
    dagger_teacher_action_threshold: float = 20.0
    dagger_action_clip: float = 20.0
    dagger_bc_lr: float = 3e-4
    dagger_actor_huber_delta: float = 1.0
    dagger_buffer_capacity: int = 131_072
    dagger_buffer_device: str = "cpu"
    dagger_batch_size: int = 4096
    # Optional collection-only phase before main DAgger.  These rollouts force
    # every valid Teacher action and populate only q_teacher_replay; they never
    # enter dagger_replay or advance the main beta/training counters.
    teacher_prefill_rollouts: int = 0
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
    perception_depth_codec: str = PERCEPTION_DEPTH_CODEC

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
    q_action_coordinates: str = "absolute"
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

    # TD3 v2 keeps only the two online learning rings.  The duplicate
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
        raw_student_latent = owner._student_latent(td)
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
        valid = _valid_teacher_action_rows(
            teacher_action, owner.cfg.dagger_teacher_action_threshold
        )
        clipped_teacher_action = owner._project_execution_action(teacher_action)
        student_valid = torch.isfinite(raw_student_latent).all(dim=-1)
        bounded_student_action = owner._student_action_from_latent(raw_student_latent)
        discrepancy_rms, discrepancy_max = _normalized_action_discrepancy(
            bounded_student_action,
            clipped_teacher_action,
            float(owner.cfg.dagger_action_clip),
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
        issued_action, exploratory_student, collector_noise = (
            _apply_student_collector_noise(
                bounded_student_action,
                clipped_teacher_action,
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
        td[DAGGER_TEACHER_ACTION_KEY] = clipped_teacher_action
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
        # _student_latent is the authoritative EMA-perception path and calls
        # actor_adapt.get_dist(td).mean directly.  It never samples and never
        # asks TorchRL's ProbabilisticActor to compute sample_log_prob.
        td[ACTION_KEY] = self._owner._student_latent(td)
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
            nominal_low, nominal_high, float(cfg.dagger_action_clip)
        )
        action_low = torch.full_like(nominal_low, -float(cfg.dagger_action_clip))
        action_high = torch.full_like(nominal_high, float(cfg.dagger_action_clip))
        self._fastsac_action_low = action_low.detach()
        self._fastsac_action_high = action_high.detach()
        self._fastsac_actor_action_center = ((action_low + action_high) * 0.5).detach()
        self._fastsac_actor_action_scale = ((action_high - action_low) * 0.5).detach()
        self._fastsac_q_action_center = ((nominal_low + nominal_high) * 0.5).detach()
        self._fastsac_q_action_scale = ((nominal_high - nominal_low) * 0.5).detach()
        self._fastsac_joint_offset_low = offset_low.detach()
        self._fastsac_joint_offset_high = offset_high.detach()
        self._fastsac_action_contract = _vaic_action_contract_metadata(
            self.joint_names,
            nominal_low,
            nominal_high,
            offset_low,
            offset_high,
            execution_action_low=action_low,
            execution_action_high=action_high,
            execution_support_source="scalar_dagger_safety_envelope",
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
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self.critic_update_count = 0
        self.actor_update_count = 0

    @staticmethod
    def _validate_td3_config(cfg) -> None:
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
                "save_teacher_buffer must be false; TD3 v2 never writes a teacher H5"
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
        if str(cfg.q_action_coordinates) != "absolute" or not bool(
            cfg.q_normalize_actions
        ):
            raise ValueError(
                "distributional TD3 requires normalized absolute Q actions"
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
        )
        for name in positive_integers:
            value = getattr(cfg, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        prefill_rollouts = cfg.teacher_prefill_rollouts
        if isinstance(prefill_rollouts, bool) or int(prefill_rollouts) < 0:
            raise ValueError("teacher_prefill_rollouts must be a non-negative integer")
        if (
            str(cfg.dagger_control_mode) == "beta"
            and float(cfg.dagger_beta_start) == 0.0
            and float(cfg.dagger_beta_end) == 0.0
            and int(prefill_rollouts) == 0
        ):
            raise ValueError(
                "pure Student beta control requires teacher_prefill_rollouts > 0 "
                "so the 50/50 Q replay can become ready"
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
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "dagger_teacher_action_threshold",
            "dagger_action_clip",
            "q_lr",
            "q_action_input_gain",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(cfg.dagger_teacher_action_threshold) > float(cfg.dagger_action_clip):
            raise ValueError("Teacher validity threshold exceeds execution support")
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
        if not (0.0 <= release < takeover <= 2.0):
            raise ValueError("SafeDAgger release/takeover thresholds are invalid")

    def _q_action_input(self, action: torch.Tensor) -> torch.Tensor:
        """Map issued absolute commands into the locked nominal Q coordinates."""
        center = self._fastsac_q_action_center.to(action)
        scale = self._fastsac_q_action_scale.to(action)
        normalized = (action - center) / scale
        gain = float(self.cfg.q_action_input_gain)
        return normalized if gain == 1.0 else normalized * gain

    def _q_action_to_physical(self, q_action: torch.Tensor) -> torch.Tensor:
        center = self._fastsac_q_action_center.to(q_action)
        scale = self._fastsac_q_action_scale.to(q_action)
        physical = (q_action / float(self.cfg.q_action_input_gain)) * scale + center
        return self._project_execution_action(physical)

    def _q_execution_bounds(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low = self._q_action_input(self._fastsac_action_low.to(reference))
        high = self._q_action_input(self._fastsac_action_high.to(reference))
        return torch.minimum(low, high), torch.maximum(low, high)

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

    def _actor_dist_from_flat(self, actor_obs: torch.Tensor):
        return self._actor_dist_from_flat_module(self.actor_adapt, actor_obs)

    def _actor_target_dist_from_flat(self, actor_obs: torch.Tensor):
        if self.actor_target is None:
            raise RuntimeError("actor_target is initialized only after source loading")
        return self._actor_dist_from_flat_module(self.actor_target, actor_obs)

    def _teacher_prefill_active(self) -> bool:
        """Whether collection is still in the separate Teacher-only phase."""
        return int(self.teacher_prefill_rollout_count) < int(
            self.cfg.teacher_prefill_rollouts
        )

    def _collect_teacher_q_replay_this_rollout(self) -> bool:
        """Collect Teacher Q rows online only when no prefill was requested."""
        return int(self.cfg.teacher_prefill_rollouts) == 0 or (
            self._teacher_prefill_active()
        )

    def _teacher_q_replay_frozen(self) -> bool:
        """Whether a completed prefill now owns the immutable Teacher partition."""
        return int(self.cfg.teacher_prefill_rollouts) > 0 and not (
            self._teacher_prefill_active()
        )

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return _DistributionalTD3DaggerRolloutPolicy(self)
        # Evaluation remains deterministic, Student-only, and noise-free.
        return _LatentStudentRolloutPolicy(
            _DeterministicTD3StudentEvalPolicy(self),
            self._fastsac_action_low,
            self._fastsac_action_high,
            self.cfg.dagger_action_clip,
        )

    def configure_teacher_replay(self, path, restore_path=None):
        """Disable the duplicate persistent H5/export FIFO for TD3 v2."""
        if restore_path is not None:
            raise ValueError("TD3 v2 raw-perception replay cannot restore a teacher H5")
        self.teacher_replay = None

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        """The two online CPU rings are intentionally not exported to H5."""
        del iteration, checkpoint_name
        return None

    def restore_q_teacher_replay(self, source_path):
        del source_path
        raise ValueError(
            "TD3 v2 is fresh-only and cannot refill raw perception windows from H5"
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
        next_latent = self._actor_target_dist_from_flat(next_actor_observations).mean
        next_physical = self._student_action_from_latent(next_latent)
        next_q = self._q_action_input(next_physical)
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
        # Mapping through the physical command path enforces the authoritative
        # execution projection before the final Q transform.
        smoothed_physical = self._q_action_to_physical(smoothed_q)
        smoothed_q = self._q_action_input(smoothed_physical)
        applied_noise = smoothed_q - next_q
        return smoothed_q, applied_noise, next_physical

    @torch.no_grad()
    def _distributional_td3_target(self, batch: dict[str, torch.Tensor]):
        smoothed_q_action, target_noise, next_physical = self._smoothed_target_q_action(
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
            "target_noise_free_action_abs_mean": next_physical.abs().mean(),
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
        prediction_latent = self._actor_dist_from_flat(batch["observations"]).mean
        if not torch.isfinite(prediction_latent).all():
            raise RuntimeError("TD3 Actor latent contains non-finite values")
        physical_action = self._student_action_from_latent(prediction_latent)
        q_action = self._q_action_input(physical_action)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        td3_actor_loss, expected_q1 = _td3_actor_q1_loss(
            self.qnet, batch["critic_observations"], q_action
        )
        exact_bc_loss = _exact_teacher_bc_loss(
            prediction_latent,
            batch[DAGGER_REPLAY_TEACHER_ACTIONS],
            batch[DAGGER_TEACHER_ACTION_VALID_KEY],
            self._fastsac_actor_action_center,
            self._fastsac_actor_action_scale,
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
        return {
            "td3_actor_loss": td3_actor_loss.detach(),
            "exact_bc_loss": exact_bc_loss.detach(),
            "weighted_td3_actor_loss": weighted_td3.detach(),
            "weighted_bc_loss": weighted_bc.detach(),
            "total_actor_loss": total_actor_loss.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "actor_expected_q1_mean": expected_q1.detach().mean(),
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
            "q_action_transform_fingerprint": self._fastsac_action_contract[
                "q_action_transform_fingerprint"
            ],
            "q_action_input_gain": float(self.cfg.q_action_input_gain),
            "clipped_double_distribution": True,
            "target_semantics": CRITIC_SEMANTICS,
            "actor_q_reduction": "online_q1_expectation_only",
            "replay_mix_semantics": (
                "frozen_prefill_teacher_0.5_student_executed_0.5_v2"
                if int(self.cfg.teacher_prefill_rollouts) > 0
                else "online_teacher_executed_0.5_student_executed_0.5_v1"
            ),
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
            "dagger_teacher_action_threshold",
            "dagger_action_clip",
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "dagger_buffer_capacity",
            "dagger_buffer_device",
            "dagger_batch_size",
            "teacher_prefill_rollouts",
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
            "perception_depth_codec",
            "save_teacher_buffer",
        )
        return {
            **{name: getattr(self.cfg, name) for name in names},
            "method": TRAINING_ALGORITHM,
            "actor_output": "pre_tanh_latent_to_locked_execution_support",
            "bc_loss": "exact_inverse_tanh_mean_smooth_l1",
        }

    def _sample_balanced_q_batch(self, sample_plan: _TD3ReplaySamplePlan | None = None):
        if sample_plan is None:
            return PPOBCDaggerFinetune._sample_balanced_q_batch(self)
        batch_size = int(self.cfg.q_batch_size)
        teacher_count = batch_size // 2
        student_count = batch_size - teacher_count
        if sample_plan.teacher_indices.numel() != teacher_count:
            raise ValueError("Teacher replay sample plan has the wrong row count")
        if sample_plan.student_indices.numel() != student_count:
            raise ValueError("Student replay sample plan has the wrong row count")
        q_fields = tuple(self.q_teacher_replay.data)
        teacher = self.q_teacher_replay.sample_by_indices(
            sample_plan.teacher_indices, self.device, fields=q_fields
        )
        student = self.dagger_replay.sample_by_indices(
            sample_plan.student_indices, self.device, fields=q_fields
        )
        mixed = {key: torch.cat((teacher[key], student[key]), dim=0) for key in teacher}
        mixed[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.cat(
            (
                torch.ones(teacher_count, dtype=torch.bool, device=self.device),
                torch.zeros(student_count, dtype=torch.bool, device=self.device),
            )
        )
        if sample_plan.permutation.shape != (batch_size,):
            raise ValueError("Q replay sample plan has the wrong permutation shape")
        return {key: value[sample_plan.permutation] for key, value in mixed.items()}

    def _sample_actor_batch(self, indices: torch.Tensor | None = None):
        fields = (
            "critic_observations",
            DAGGER_REPLAY_TEACHER_ACTIONS,
            DAGGER_TEACHER_ACTION_VALID_KEY,
            *_PERCEPTION_REPLAY_FIELDS,
        )
        if indices is None:
            batch = self.dagger_replay.sample(
                int(self.cfg.dagger_batch_size),
                self.device,
                self.q_rng,
                fields=fields,
            )
        else:
            if indices.numel() != int(self.cfg.dagger_batch_size):
                raise ValueError("Actor replay sample plan has the wrong row count")
            batch = self.dagger_replay.sample_by_indices(
                indices, self.device, fields=fields
            )
        return self._prepare_dagger_learning_batch(batch)

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
        transition_chunks = tuple(self._dagger_transition_chunks(rollout))
        appended = 0
        teacher_rows_appended = 0
        teacher_selected = 0
        valid_labels = 0
        student_selected = 0
        collector_noise_norm = 0.0
        if transition_chunks:
            transitions = {
                key: torch.cat([chunk[key] for chunk in transition_chunks], dim=0)
                for key in transition_chunks[0]
            }
            del transition_chunks
            replay_device = self.dagger_replay.device
            staged = {
                key: (
                    value.detach()
                    if value.device == replay_device
                    else value.detach().to(replay_device)
                )
                for key, value in transitions.items()
            }
            valid = staged[DAGGER_TEACHER_ACTION_VALID_KEY].bool()
            is_student = staged[DAGGER_IS_STUDENT_ACTION_KEY].bool()
            valid_labels = int(valid.sum().item())
            teacher_selected = int((~is_student).sum().item())
            student_selected = int(is_student.sum().item())
            collector_noise_norm = (
                staged[TD3_COLLECTOR_NOISE_KEY].float().norm(dim=-1).mean().item()
            )
            if collect_teacher_q:
                teacher_executed = valid & ~is_student
                if teacher_executed.any():
                    teacher_indices = teacher_executed.nonzero(as_tuple=False).squeeze(
                        -1
                    )
                    q_teacher = {
                        key: staged[key].index_select(0, teacher_indices)
                        for key in _Q_REPLAY_FIELDS
                    }
                    teacher_rows_appended += self.q_teacher_replay.extend(q_teacher)
                    del q_teacher, teacher_indices
                del teacher_executed
            if not teacher_prefill_active:
                appended = self.dagger_replay.extend(staged)
            del valid, is_student, staged, transitions

        if teacher_prefill_active:
            # This phase deliberately owns no optimizer and no main replay.
            # Keeping the raw recurrent history built above makes the first
            # main-rollout windows temporally continuous with prefill.
            self.teacher_prefill_rollout_count += 1
            self.teacher_prefill_environment_steps += int(self.cfg.train_every)
            if not self._teacher_prefill_active() and self.q_teacher_replay.size < int(
                self.cfg.td3_learning_starts
            ):
                raise RuntimeError(
                    "Teacher-only prefill ended before q_teacher_replay reached "
                    "td3_learning_starts; increase teacher_prefill_rollouts"
                )
            info = {
                "td3/method_distributional_td3_teacher_bc_v1": 1.0,
                "td3/prefill_active": 1.0,
                "td3/prefill_rollout_count": self.teacher_prefill_rollout_count,
                "td3/prefill_target_rollouts": int(self.cfg.teacher_prefill_rollouts),
                "td3/prefill_environment_steps": (
                    self.teacher_prefill_environment_steps
                ),
                "td3/prefill_rows_this_rollout": teacher_rows_appended,
                "td3/teacher_replay_rows_this_rollout": teacher_rows_appended,
                "td3/teacher_replay_frozen": float(self._teacher_q_replay_frozen()),
                "td3/prefill_forced_teacher_fraction": teacher_selected
                / max(valid_labels + student_selected, 1),
                "td3/teacher_replay_size": self.q_teacher_replay.size,
                "td3/replay_size": self.dagger_replay.size,
                "td3/replay_seen": self.dagger_replay.seen,
                "td3/replay_ready": 0.0,
                "td3/actor_update_count": self.actor_update_count,
                "td3/critic_update_count": self.critic_update_count,
                "td3/actor_updates_this_rollout": 0,
                "td3/critic_updates_this_rollout": 0,
                "td3/collector_exploration_noise_norm": collector_noise_norm,
                "td3/valid_teacher_fraction": valid_labels
                / max(valid_labels + student_selected, 1),
                "td3/beta": float(self._teacher_mixture_probability()),
                "td3/rollout_count": self.dagger_rollout_count,
                "td3/environment_steps": self.dagger_environment_steps,
            }
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
                sample_plans = _prefetch_td3_replay_sample_plans(
                    self.dagger_replay,
                    self.q_teacher_replay,
                    q_batch_size=int(self.cfg.q_batch_size),
                    actor_batch_size=int(self.cfg.dagger_batch_size),
                    update_count=q_updates,
                    policy_delay=int(self.cfg.policy_delay),
                    critic_update_count=int(self.critic_update_count),
                    output_device=self.device,
                    generator=self.q_rng,
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
                    if sample_plan is not None and actor_indices is None:
                        raise RuntimeError("delayed Actor replay plan is missing")
                    delayed = self._maybe_delayed_actor_and_targets(
                        self._sample_actor_batch(actor_indices)
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
            "td3/prefill_target_rollouts": int(self.cfg.teacher_prefill_rollouts),
            "td3/prefill_environment_steps": self.teacher_prefill_environment_steps,
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
            "td3/teacher_replay_size": self.q_teacher_replay.size,
            "td3/student_replay_rows": student_q_rows,
            "td3/student_source_fraction": student_selected / max(appended, 1),
            "td3/teacher_source_fraction": teacher_selected / max(appended, 1),
            "td3/valid_teacher_fraction": valid_labels / max(appended, 1),
            "td3/beta": beta_value,
            "td3/rollout_count": self.dagger_rollout_count,
            "td3/environment_steps": self.dagger_environment_steps,
            "td3/truncation_finals": self._last_truncation_finals_used,
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
                "adapt_optimizer": self.opt_adapt.state_dict(),
            },
            "actor_update_count": int(self.actor_update_count),
            "critic_update_count": int(self.critic_update_count),
            "dagger_rollout_count": int(self.dagger_rollout_count),
            "dagger_environment_steps": int(self.dagger_environment_steps),
            "teacher_prefill_rollout_count": int(self.teacher_prefill_rollout_count),
            "teacher_prefill_environment_steps": int(
                self.teacher_prefill_environment_steps
            ),
            "dagger_rng_state": self.dagger_rng.get_state(),
            "q_rng_state": self.q_rng.get_state(),
            "collector_exploration_rng_state": (
                self.collector_exploration_rng.get_state()
            ),
            "target_policy_rng_state": self.target_policy_rng.get_state(),
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
        self.opt_adapt.load_state_dict(optimizers["adapt_optimizer"])
        self.actor_update_count = int(state["actor_update_count"])
        self.critic_update_count = int(state["critic_update_count"])
        self.dagger_rollout_count = int(state["dagger_rollout_count"])
        self.dagger_environment_steps = int(state["dagger_environment_steps"])
        self.teacher_prefill_rollout_count = int(state["teacher_prefill_rollout_count"])
        self.teacher_prefill_environment_steps = int(
            state["teacher_prefill_environment_steps"]
        )
        self.dagger_rng.set_state(state["dagger_rng_state"])
        self.q_rng.set_state(state["q_rng_state"])
        self.collector_exploration_rng.set_state(
            state["collector_exploration_rng_state"]
        )
        self.target_policy_rng.set_state(state["target_policy_rng_state"])
        self._last_td3_diagnostics = copy.deepcopy(
            state.get("last_td3_diagnostics", {})
        )
        self.actor_target.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()

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
                "TD3 v2 raw-perception replay is fresh-only; same-stage TD3 "
                "checkpoint resume is intentionally unsupported"
            )
        elif algorithm is not None:
            raise ValueError(f"unsupported TD3 source training_algorithm={algorithm!r}")
        elif state_dict.get("last_phase") != "train":
            raise ValueError("fresh TD3 Teacher-BC requires a PPO train checkpoint")

        failed = PPOVEL.load_state_dict(self, state_dict, strict)
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
            self._migrate_fresh_ppo_student_actor_to_latent()
            hard_copy_(self.qnet, self.qnet_target)
            self.qnet_target.requires_grad_(False).eval()
            # The exact copy is intentionally created only after the online
            # Actor has loaded and its physical-action head has been migrated.
            self.actor_target = copy.deepcopy(self.actor_adapt).requires_grad_(False)
            self.actor_target.eval()
            self.actor_update_count = 0
            self.critic_update_count = 0
            self.dagger_rollout_count = 0
            self.dagger_environment_steps = 0
            self.teacher_prefill_rollout_count = 0
            self.teacher_prefill_environment_steps = 0
            self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
            self.q_rng.manual_seed(int(self.cfg.q_seed))
            self.collector_exploration_rng.manual_seed(int(self.cfg.q_seed) + 1)
            self.target_policy_rng.manual_seed(int(self.cfg.q_seed) + 2)
        self._freeze_teacher()
        self.actor_adapt.requires_grad_(True).train()
        self.qnet.requires_grad_(True).train()
        self.actor_target.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        return failed
