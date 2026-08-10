"""PPO-teacher DAgger finetuning with a Stage-2-compatible SAC critic.

This stage deliberately does *not* run PPO or SAC actor optimization.  A
frozen PPO residual policy is the privileged DAgger oracle, ``actor_adapt`` is
trained only by behavior cloning, while the exact FastSAC C51 Q1/Q2 topology
is trained from a beta-independent 50/50 mixture of teacher-executed and
student-executed transitions. Q-derived actor weighting is deliberately absent.
The original VAIC observation, reward, termination, depth-supervision, and EMA
paths remain owned by :class:`PPOVEL`.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import math
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass

import numpy as np
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
    FASTSAC_BC_DAGGER_TARGET_ENTROPY_SEMANTICS,
    FASTSAC_CLIPPED_DOUBLE_Q_SEMANTICS,
    FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS,
    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_EARLY_FUSION_SEMANTICS,
    FASTSAC_REFERENCE_EPS,
    FASTSAC_RAW_OBSERVATION_ROOT,
    FastSACTanhNormal,
    REPLAY_OBSERVATION_SEMANTICS,
    SAC_REWARD_SCALARIZATION,
    TEACHER_REPLAY_FIELDS,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    TeacherReplayBuffer,
    _BCDaggerSACAdapter,
    _build_isolated_q_network,
    _filter_replay_rows,
    _measure_or_clip_grad_norm,
    _q_action_hidden_dim,
    _sac_bootstrap_mask,
    _select_c51_twin_target,
    _vaic_truncation_mask,
    _vecnorm_state_fingerprint,
)
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    PPOConfig,
    PPOVEL,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    PRIV_PRED_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    ZeroDepthInjector,
)


PPO_BC_DAGGER_TRAINING_ALGORITHM = (
    "vaic_ppo_bc_dagger_student_sac_critic_v3"
)
PPO_BC_DAGGER_IQL_TRAINING_ALGORITHM = (
    "vaic_ppo_bc_dagger_student_iql_v2"
)
PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM = (
    "vaic_ppo_bc_dagger_student_v1"
)
PPO_BC_DAGGER_ACTOR_BACKEND = "vaic_ppo_independent_normal_bc_dagger_v1"
PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS = (
    "beta_independent_half_teacher_half_student_executed_action_"
    "stochastic_student_q_only_c51_clipped_double_q_v1"
)
# Kept only as a checkpoint/version sentinel. The production path below no
# longer constructs or optimizes an IQL value network.
PPO_BC_DAGGER_IQL_CRITIC_SEMANTICS = (
    "dataset_action_target_twin_expected_c51_expectile_v_to_scalar_td_"
    "c51_projection_v1"
)
PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS = (
    "dagger_teacher_huber_bc_only_no_q_or_advantage_weighting_v1"
)
DAGGER_TEACHER_REPLAY_FORMAT = "vaic_ppo_bc_dagger_teacher_buffer"
DAGGER_TEACHER_REPLAY_FORMAT_VERSION = 2
DAGGER_LEGACY_REPLAY_OBSERVATION_SEMANTICS = (
    "normalized_frozen_vecnorm_v1"
)
DAGGER_REPLAY_OBSERVATION_SEMANTICS = REPLAY_OBSERVATION_SEMANTICS
DAGGER_INITIAL_TRANSITION_FILTER = "step_count_gt_5"
DAGGER_ACTION_PARAMETERIZATION = (
    "absolute_executable_safe_hysteresis_or_beta_teacher_or_student_v2"
)
DAGGER_TEACHER_ACTION_SEMANTICS = "ppo_reference_plus_residual_v1"
DAGGER_CONTROL_SEMANTICS = (
    "clipped_deterministic_mean_normalized_rms_safe_hysteresis_or_beta_v1"
)
DAGGER_FINALIZATION_SEMANTICS = (
    "perception_then_actor_then_perception_then_fresh_replay_q_v1"
)
DAGGER_FINALIZATION_PHASES = (
    "perception_consolidation",
    "actor_realignment",
    "perception_recheck",
    "replay_q_calibration",
)
DAGGER_STAGING_SEMANTICS = (
    "joint_then_cyclic_perception_actor_then_final_perception_actor_"
    "then_fresh_replay_q_v1"
)
DAGGER_STAGING_PHASES = (
    "joint_warmup",
    "cycle_perception",
    "cycle_actor",
    "final_perception",
    "final_actor",
    "replay_q_calibration",
)
DAGGER_TEACHER_ACTION_KEY = "teacher_action"
DAGGER_TEACHER_ACTION_VALID_KEY = "teacher_action_valid"
DAGGER_IS_STUDENT_ACTION_KEY = "is_student_action"
DAGGER_ACTION_DISCREPANCY_RMS_KEY = "dagger_action_discrepancy_rms"
DAGGER_ACTION_DISCREPANCY_MAX_KEY = "dagger_action_discrepancy_max"
DAGGER_SAFE_UNSAFE_KEY = "dagger_safe_unsafe"
DAGGER_SAFE_TEACHER_KEY = "dagger_safe_teacher"
DAGGER_SAFE_TAKEOVER_KEY = "dagger_safe_takeover"
DAGGER_SAFE_RELEASE_KEY = "dagger_safe_release"
DAGGER_BETA_TEACHER_KEY = "dagger_beta_teacher"
DAGGER_STUDENT_ACTION_VALID_KEY = "dagger_student_action_valid"
DAGGER_REPLAY_TEACHER_ACTIONS = "teacher_actions"
DAGGER_Q_TEACHER_SOURCE_KEY = "_dagger_q_teacher_source"
DAGGER_REPLAY_MIN_STEP_COUNT = 5
DAGGER_CONTROL_MODES = ("beta", "safe", "hybrid")


def _valid_teacher_action_rows(
    actions: torch.Tensor, threshold: float = 20.0
) -> torch.Tensor:
    """Return one validity bit per row for finite, bounded teacher actions."""
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("teacher action threshold must be finite and positive")
    if actions.ndim < 1:
        raise ValueError("teacher actions must contain an action dimension")
    return torch.isfinite(actions).all(dim=-1) & (actions.abs() <= threshold).all(
        dim=-1
    )


def _normalized_action_discrepancy(
    student_action: torch.Tensor,
    teacher_mean_action: torch.Tensor,
    action_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row RMS and max error in the critic's unit action box."""
    if student_action.shape != teacher_mean_action.shape:
        raise ValueError(
            "SafeDAgger student/teacher action shapes must match, got "
            f"{tuple(student_action.shape)} and {tuple(teacher_mean_action.shape)}"
        )
    if student_action.ndim < 1 or student_action.shape[-1] < 1:
        raise ValueError("SafeDAgger actions must contain an action dimension")
    action_clip = float(action_clip)
    if not math.isfinite(action_clip) or action_clip <= 0.0:
        raise ValueError("SafeDAgger action_clip must be finite and positive")
    normalized_error = (student_action - teacher_mean_action).abs() / action_clip
    return (
        normalized_error.square().mean(dim=-1).sqrt(),
        normalized_error.amax(dim=-1),
    )


def _linear_teacher_probability(
    start: float, end: float, decay_rollouts: int, rollout_count: int
) -> float:
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("DAgger beta endpoints must be finite")
    if not 0.0 <= end <= start <= 1.0:
        raise ValueError("DAgger beta must satisfy 0 <= end <= start <= 1")
    if isinstance(decay_rollouts, bool) or int(decay_rollouts) < 1:
        raise ValueError("dagger_beta_decay_rollouts must be positive")
    progress = min(max(int(rollout_count), 0) / int(decay_rollouts), 1.0)
    return float(start + (end - start) * progress)


def _iql_expectile_loss(
    difference: torch.Tensor, expectile: float, *, validate: bool = True
) -> torch.Tensor:
    """Elementwise IQL asymmetric squared loss.

    ``difference`` follows the official IQL convention ``target_q - value``.
    The function deliberately returns unreduced values so callers can log the
    positive/negative sides without changing the optimization objective.
    """
    expectile = float(expectile)
    if not math.isfinite(expectile) or not 0.0 < expectile < 1.0:
        raise ValueError("IQL expectile must be finite and strictly in (0, 1)")
    if validate and not torch.isfinite(difference).all():
        raise ValueError("IQL expectile differences must be finite")
    weight = torch.where(
        difference > 0.0,
        expectile,
        1.0 - expectile,
    )
    return weight * difference.square()


def _project_scalar_to_c51(
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Project a scalar IQL TD target onto the existing FastSAC C51 support."""
    if target.ndim != 1:
        raise ValueError("IQL scalar C51 target must have shape [batch]")
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("IQL C51 support must be a one-dimensional atom vector")
    deltas = support[1:] - support[:-1]
    if validate:
        if not torch.isfinite(target).all() or not torch.isfinite(support).all():
            raise ValueError("IQL scalar targets and C51 support must be finite")
        if not torch.all(deltas > 0.0) or not torch.allclose(
            deltas, deltas[:1].expand_as(deltas), rtol=1e-5, atol=1e-7
        ):
            raise ValueError(
                "IQL C51 support must be strictly increasing and uniform"
            )

    v_min = support[0]
    v_max = support[-1]
    delta = deltas[0]
    clipped = target.clamp(v_min, v_max)
    atom_position = ((clipped - v_min) / delta).clamp(
        0.0, float(support.numel() - 1)
    )
    lower = atom_position.floor().long().clamp(0, support.numel() - 1)
    upper = atom_position.ceil().long().clamp(0, support.numel() - 1)
    same_atom = lower == upper
    lower_weight = torch.where(
        same_atom,
        torch.ones_like(atom_position),
        upper.to(atom_position.dtype) - atom_position,
    )
    upper_weight = torch.where(
        same_atom,
        torch.zeros_like(atom_position),
        atom_position - lower.to(atom_position.dtype),
    )
    projection = torch.zeros(
        target.shape[0],
        support.numel(),
        device=target.device,
        dtype=target.dtype,
    )
    projection.scatter_add_(1, lower[:, None], lower_weight[:, None])
    projection.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return projection


class _IQLValueNetwork(nn.Module):
    """Scalar V(s) used only to form in-dataset IQL critic targets."""

    def __init__(self, obs_dim, hidden_dim, layer_norm=True):
        super().__init__()
        if int(hidden_dim) < 4:
            raise ValueError("IQL value hidden dimension must be at least four")
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        layers.extend(
            (nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4))
        )
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 4))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 4, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, observations):
        return self.net(observations).squeeze(-1)


def _build_isolated_iql_value_network(
    obs_dim, hidden_dim, layer_norm, device, seed
):
    """Initialize IQL V without advancing rollout/environment RNG streams."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(int(seed))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        value = _IQLValueNetwork(
            obs_dim, hidden_dim, layer_norm
        ).to(device)
    return value


@dataclass
class PPOBCDaggerFinetuneConfig(PPOConfig):
    _target_: str = (
        "active_adaptation.learning.ppo.ppo_bc_dagger."
        "PPOBCDaggerFinetune"
    )
    name: str = "ppo_bc_dagger"
    phase: str = "finetune"
    vecnorm: str = "eval"
    enable_residual_distillation: bool = False

    # SafeDAgger is the default controller. ``beta`` remains as a reproducible
    # ablation, while ``hybrid`` takes over unsafe rows and also applies beta on
    # otherwise safe rows. All comparisons use clipped deterministic means.
    dagger_control_mode: str = "safe"
    dagger_safe_takeover_rms: float = 0.006
    dagger_safe_release_rms: float = 0.004
    dagger_safe_min_teacher_steps: int = 8
    # Optional cumulative rollout boundary for a deliberate student-only tail.
    # Indices below K retain SafeDAgger; index K onward disables only the safe
    # controller (hybrid beta selection, if configured, remains independent).
    dagger_safe_zero_iteration: int | None = None
    # beta is stage-local and is used only by beta/hybrid control modes.
    dagger_beta_start: float = 1.0
    dagger_beta_end: float = 0.0
    dagger_beta_decay_rollouts: int = 4000
    # Dedicated scripts/bc_dagger.py CLI alias. The entrypoint resolves this
    # completed-rollout boundary into dagger_beta_decay_rollouts before
    # constructing the policy; None preserves the legacy setting above.
    dagger_beta_zero_iteration: int | None = None
    dagger_seed: int = 0
    dagger_teacher_action_threshold: float = 20.0
    dagger_action_clip: float = 20.0

    # Optional post-DAgger consolidation pipeline.  These fields are internal
    # controls populated by scripts/bc_dagger_finalize.py; keeping them out of
    # ``dagger_backend_config`` lets the source checkpoint retain its exact
    # controller/Q contract while finalization supplies a runtime-only policy.
    dagger_finalization_enabled: bool = False
    dagger_finalize_perception_iterations: int = 0
    dagger_finalize_actor_iterations: int = 0
    dagger_finalize_recheck_iterations: int = 0
    dagger_finalize_calibration_iterations: int = 0
    dagger_finalize_calibration_control_mode: str = "beta"
    dagger_finalize_calibration_teacher_probability: float = 0.5

    # Optional block-coordinate BC-DAgger schedule for a fresh PPO teacher.
    # Unlike finalization, the joint warmup may train all three optimizer
    # owners.  Cyclic/final perception and actor blocks isolate their owner,
    # and only the terminal Q block is allowed to persist teacher rows to H5.
    dagger_staging_enabled: bool = False
    dagger_stage_joint_warmup_iterations: int = 0
    dagger_stage_cycles: int = 0
    dagger_stage_perception_iterations: int = 0
    dagger_stage_actor_iterations: int = 0
    dagger_stage_final_perception_iterations: int = 0
    dagger_stage_final_actor_iterations: int = 0
    dagger_stage_calibration_iterations: int = 0
    dagger_stage_calibration_control_mode: str = "beta"
    dagger_stage_calibration_teacher_probability: float = 0.5

    dagger_bc_lr: float = 3e-4
    dagger_bc_epochs: int = 1
    dagger_actor_huber_delta: float = 1.0
    # One eighth of HOI's 524,288-row stack keeps the VAIC privileged replay
    # practical (its critic observation is much wider) while retaining 256
    # vector-control steps at the default 512 environments.
    dagger_buffer_capacity: int = 131_072
    dagger_buffer_device: str = "cpu"
    dagger_batch_size: int = 4096
    dagger_updates_per_rollout: int = 32
    # Both replay rings store pre-VecNorm environment fields. BC/Q normalize
    # sampled minibatches once with the frozen teacher-checkpoint statistics.
    dagger_replay_raw_observations: bool = True
    # Do not duplicate the large depth image in the N x T rollout. Perception
    # still consumes the normal depth field; only direct actor/Q replay inputs
    # need a pre-VecNorm alias.
    replay_raw_observation_keys: tuple[str, ...] = (
        VEL_CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        CMD_KEY,
    )

    q_hidden_dim: int = 768
    q_num_atoms: int = 501
    q_v_min: float = -20.0
    q_v_max: float = 20.0
    q_layer_norm: bool = True
    q_action_fusion: str = "early"
    q_action_coordinates: str = "absolute"
    sac_q_normalize_actions: bool = True
    sac_q_action_input_gain: float = 1.0
    sac_clipped_double_q: bool = True
    q_lr: float = 3e-5
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    # These are intentionally identical to FastSACVelFinetune defaults.
    q_tau: float = 0.001
    q_max_grad_norm: float = 1.0
    # 128 x 512 samples for each 512-env x 32-step rollout is UTD=4, matching
    # the validated Stage-2 experiment. Each draw is exactly 256/256.
    q_batch_size: int = 512
    q_updates_per_rollout: int = 128
    q_teacher_replay_ratio: float = 0.5
    q_teacher_buffer_capacity: int = 131_072
    # Ordinary replay warm-up, not a critic-confidence gate. It prevents the
    # first handful of rare student rows at beta~=1 from being oversampled by
    # the full UTD-4 update burst.
    q_learning_starts_per_source: int = 8_192

    # DAgger control remains deterministic. Only the SAC Bellman next action
    # uses this dedicated small-noise distribution; PPO actor_std is unused.
    sac_bc_initial_action_std: float = 0.01
    sac_bc_log_std_min: float = -8.0
    sac_bc_log_std_max: float = -2.0
    # Initialized here and transferred for Stage 2; BC's Q-only target keeps
    # effective alpha at zero exactly like the Stage-2 actor burn-in.
    sac_alpha_init: float = 1e-5
    sac_entropy_reference_scale: float = 1.0

    save_teacher_buffer: bool = True
    teacher_buffer_filename: str = "teacher_replay_buffer.h5"
    teacher_buffer_path: str | None = None
    # Match the HOI teacher-export horizon.  This FIFO is created only on the
    # rank that writes checkpoints; other ranks keep only their learning ring.
    teacher_buffer_capacity: int = 524_288
    teacher_buffer_snapshot_chunk_rows: int = 4096


ConfigStore.instance().store(
    "ppo_bc_dagger_finetune",
    node=PPOBCDaggerFinetuneConfig(
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


def _resolve_replay_device(configured, policy_device) -> torch.device:
    value = str(configured).strip().lower()
    if value == "policy":
        return torch.device(policy_device)
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("dagger_buffer_device supports only CPU or CUDA")
    if device.type == "cuda" and device.index is None:
        policy_device = torch.device(policy_device)
        device = torch.device(
            "cuda", policy_device.index if policy_device.type == "cuda" else 0
        )
    return device


class _DeviceReplay:
    """Circular all-transition DAgger replay with one transfer per sample."""

    def __init__(self, capacity: int, device):
        if isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError("dagger_buffer_capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.seen = 0
        # Boolean validity fields change only when the FIFO is extended.  BC and
        # balanced-Q sampling otherwise asked ``nonzero`` to rescan the full
        # 131k-row ring for every optimizer update (160 scans per rollout at the
        # dedicated defaults).  Cache the same ascending physical indices until
        # the next write; this does not change the sampling population or RNG.
        self._valid_index_cache: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        """Drop every row and allocation from this ephemeral learning FIFO."""
        self.data.clear()
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self._valid_index_cache.clear()

    @torch.no_grad()
    def extend(self, transitions: dict[str, torch.Tensor]) -> int:
        if not transitions:
            return 0
        count = int(next(iter(transitions.values())).shape[0])
        values = {}
        for key, value in transitions.items():
            value = value.detach()
            if int(value.shape[0]) != count:
                raise ValueError(f"Replay field {key!r} has misaligned rows")
            values[key] = value.to(self.device)
        if count == 0:
            return 0
        self._valid_index_cache.clear()
        if not self.data:
            self.data = {
                key: torch.empty(
                    (self.capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device=self.device,
                )
                for key, value in values.items()
            }
        elif set(values) != set(self.data):
            raise KeyError("DAgger replay field set changed after allocation")

        self.seen += count
        if count >= self.capacity:
            for key, value in values.items():
                self.data[key].copy_(value[-self.capacity :])
            self.ptr = 0
            self.size = self.capacity
            return count
        first = min(count, self.capacity - self.ptr)
        second = count - first
        for key, value in values.items():
            self.data[key][self.ptr : self.ptr + first].copy_(value[:first])
            if second:
                self.data[key][:second].copy_(value[first:])
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    def _valid_indices(self, key: str) -> torch.Tensor:
        if self.size < 1:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if key not in self.data:
            raise KeyError(f"Unknown DAgger replay validity field {key!r}")
        value = self.data[key][: self.size]
        if value.dtype is not torch.bool or value.ndim != 1:
            raise TypeError(
                f"DAgger replay validity field {key!r} must be one-dimensional bool"
            )
        indices = self._valid_index_cache.get(key)
        if indices is None:
            indices = value.nonzero(as_tuple=False).squeeze(-1)
            self._valid_index_cache[key] = indices
        return indices

    def valid_count(self, key: str) -> int:
        return int(self._valid_indices(key).numel())

    def sample(
        self,
        count: int,
        output_device,
        generator=None,
        valid_key: str | None = None,
        fields=None,
    ):
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty DAgger replay")
        count = int(count)
        generator_device = self.device
        if generator is not None:
            generator_device = torch.device(generator.device)
        if valid_key is None:
            indices = torch.randint(
                0,
                self.size,
                (count,),
                device=generator_device,
                generator=generator,
            )
        else:
            valid_indices = self._valid_indices(valid_key)
            if valid_indices.numel() == 0:
                raise RuntimeError("Cannot sample DAgger BC without valid labels")
            positions = torch.randint(
                0,
                valid_indices.numel(),
                (count,),
                device=generator_device,
                generator=generator,
            )
            if positions.device != valid_indices.device:
                positions = positions.to(valid_indices.device)
            indices = valid_indices[positions]
        if indices.device != self.device:
            indices = indices.to(self.device)
        fields = tuple(self.data) if fields is None else tuple(fields)
        unknown = set(fields).difference(self.data)
        if unknown:
            raise KeyError(
                f"Unknown DAgger replay sample fields: {sorted(unknown)}"
            )
        output_device = torch.device(output_device)
        return {
            key: (
                sampled
                if sampled.device == output_device
                else sampled.to(output_device)
            )
            for key in fields
            for sampled in (self.data[key][indices],)
        }


class _DaggerTeacherReplayBuffer(TeacherReplayBuffer):
    """Teacher-executed FIFO with a DAgger-specific, truthful H5 manifest."""

    def __init__(self, *args, vecnorm_fingerprint, action_clip, **kwargs):
        super().__init__(*args, **kwargs)
        fingerprint = str(vecnorm_fingerprint or "")
        if not fingerprint.startswith("sha256:"):
            raise ValueError(
                "Raw DAgger replay requires a checkpoint VecNorm fingerprint"
            )
        self.vecnorm_fingerprint = fingerprint
        self.action_clip = float(action_clip)
        if not math.isfinite(self.action_clip) or self.action_clip <= 0.0:
            raise ValueError("DAgger replay action_clip must be finite and positive")

    def _validated_values(self, data):
        count, values = super()._validated_values(data)
        actions = values["actions"]
        if not torch.isfinite(actions).all():
            raise ValueError("DAgger replay actions must all be finite")
        if (actions.abs() > self.action_clip).any():
            raise ValueError(
                "DAgger replay action lies outside the configured symmetric "
                f"support [-{self.action_clip}, {self.action_clip}]"
            )
        return count, values

    def checkpoint_metadata(self):
        has_snapshot = self.last_snapshot_id is not None
        return {
            "format": DAGGER_TEACHER_REPLAY_FORMAT,
            "format_version": DAGGER_TEACHER_REPLAY_FORMAT_VERSION,
            "initial_transition_filter": DAGGER_INITIAL_TRANSITION_FILTER,
            "action_parameterization": DAGGER_ACTION_PARAMETERIZATION,
            "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
            "replay_observation_semantics": DAGGER_REPLAY_OBSERVATION_SEMANTICS,
            "vecnorm_fingerprint": self.vecnorm_fingerprint,
            "reward_scalarization": SAC_REWARD_SCALARIZATION,
            "truncation_next_observation": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "replay_id": self.replay_id,
            "actor_backend": PPO_BC_DAGGER_ACTOR_BACKEND,
            "actor_obs_keys": list(self.actor_obs_keys),
            "critic_obs_keys": list(self.critic_obs_keys),
            "actor_obs_dim": self.actor_dim,
            "critic_obs_dim": self.critic_dim,
            "action_dim": self.action_dim,
            "action_clip": self.action_clip,
            "capacity": self.capacity,
            "size": self.last_snapshot_size if has_snapshot else self.size,
            "seen": self.last_snapshot_seen if has_snapshot else self.seen,
            "storage_fields": list(self.storage_fields),
            "field_shapes": {
                name: list(self.shapes[name]) for name in self.storage_fields
            },
            "snapshot_iteration": self.last_snapshot_iteration,
            "checkpoint_name": self.last_snapshot_name,
            "snapshot_id": self.last_snapshot_id,
        }

    @torch.no_grad()
    def restore(self, source_path, expected_metadata=None):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to restore DAgger replay") from exc
        if self.data or self.size or self.seen:
            raise RuntimeError("DAgger teacher replay restore needs an empty FIFO")
        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        with h5py.File(source_path, "r") as replay:
            if str(replay.attrs.get("format", "")) != DAGGER_TEACHER_REPLAY_FORMAT:
                raise ValueError(f"Not a VAIC PPO-BC DAgger replay: {source_path}")
            if int(replay.attrs.get("format_version", 0)) != (
                DAGGER_TEACHER_REPLAY_FORMAT_VERSION
            ):
                raise ValueError("Unsupported PPO-BC DAgger replay version")
            if str(replay.attrs.get("replay_observation_semantics", "")) != (
                DAGGER_REPLAY_OBSERVATION_SEMANTICS
            ):
                raise ValueError("DAgger replay observation semantics mismatch")
            if str(replay.attrs.get("vecnorm_fingerprint", "")) != (
                self.vecnorm_fingerprint
            ):
                raise ValueError(
                    "DAgger replay VecNorm fingerprint does not match checkpoint"
                )
            required_attrs = {
                "teacher_only": True,
                "storage_policy": "circular_fifo",
                "storage_order": "oldest_to_newest",
                "initial_transition_filter": DAGGER_INITIAL_TRANSITION_FILTER,
                "action_parameterization": DAGGER_ACTION_PARAMETERIZATION,
                "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
                "reward_scalarization": SAC_REWARD_SCALARIZATION,
                "truncation_next_observation": (
                    TRUNCATION_NEXT_OBSERVATION_SEMANTICS
                ),
                "actor_backend": PPO_BC_DAGGER_ACTOR_BACKEND,
            }
            for name, expected in required_attrs.items():
                actual = replay.attrs.get(name)
                if isinstance(expected, bool):
                    actual = bool(actual)
                else:
                    actual = str(actual)
                if actual != expected:
                    raise ValueError(
                        f"DAgger replay {name}={actual!r} does not match "
                        f"{expected!r}"
                    )
            replay_action_clip = replay.attrs.get("action_clip")
            if (
                replay_action_clip is not None
                and not math.isclose(
                    float(replay_action_clip),
                    self.action_clip,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("DAgger replay action clip mismatch")
            manifest = {
                "actor_obs_keys": json.loads(
                    str(replay.attrs.get("actor_obs_keys", "[]"))
                ),
                "critic_obs_keys": json.loads(
                    str(replay.attrs.get("critic_obs_keys", "[]"))
                ),
                "actor_obs_dim": int(replay.attrs.get("actor_obs_dim", -1)),
                "critic_obs_dim": int(
                    replay.attrs.get("critic_obs_dim", -1)
                ),
                "action_dim": int(replay.attrs.get("action_dim", -1)),
                "capacity": int(replay.attrs.get("buffer_capacity", -1)),
            }
            expected_manifest = {
                "actor_obs_keys": list(self.actor_obs_keys),
                "critic_obs_keys": list(self.critic_obs_keys),
                "actor_obs_dim": self.actor_dim,
                "critic_obs_dim": self.critic_dim,
                "action_dim": self.action_dim,
                "capacity": self.capacity,
            }
            if manifest != expected_manifest:
                raise ValueError(
                    "DAgger replay observation/action manifest does not match "
                    f"the resumed policy: file={manifest}, "
                    f"policy={expected_manifest}"
                )
            names = set(replay.keys())
            if names != set(self.storage_fields):
                raise ValueError("DAgger replay dataset fields do not match")
            size = int(replay.attrs.get("num_transitions", -1))
            seen = int(replay.attrs.get("num_seen_transitions", -1))
            if size < 0 or size > self.capacity or seen < size:
                raise ValueError("DAgger replay counters are invalid")
            if size != min(seen, self.capacity):
                raise ValueError("DAgger replay FIFO counters are inconsistent")
            actual_id = str(replay.attrs.get("replay_id", ""))
            if expected_metadata is not None:
                expected_id = expected_metadata.get("replay_id")
                if expected_id and actual_id != str(expected_id):
                    raise ValueError("DAgger replay id does not match checkpoint")
                expected_snapshot = expected_metadata.get("snapshot_id")
                actual_snapshot = str(replay.attrs.get("snapshot_id", ""))
                if expected_snapshot and actual_snapshot != str(expected_snapshot):
                    raise ValueError(
                        "DAgger replay snapshot does not match checkpoint"
                    )
                expected_fingerprint = expected_metadata.get(
                    "vecnorm_fingerprint"
                )
                if (
                    expected_fingerprint
                    and str(expected_fingerprint) != self.vecnorm_fingerprint
                ):
                    raise ValueError(
                        "Checkpoint teacher replay VecNorm fingerprint mismatch"
                    )
            if size:
                self._allocate()
                for name in self.storage_fields:
                    expected_shape = (size, *self.shapes[name])
                    if tuple(replay[name].shape) != expected_shape:
                        raise ValueError(
                            f"DAgger replay {name!r} shape mismatch"
                        )
                    expected_dtype = np.dtype(
                        np.bool_
                        if self.dtypes[name] == torch.bool
                        else np.float32
                    )
                    if np.dtype(replay[name].dtype) != expected_dtype:
                        raise ValueError(
                            f"DAgger replay {name!r} dtype mismatch"
                        )
                    for start in range(0, size, self.snapshot_chunk_rows):
                        end = min(start + self.snapshot_chunk_rows, size)
                        host = torch.from_numpy(np.asarray(replay[name][start:end]))
                        if name == "actions":
                            if not torch.isfinite(host).all():
                                raise ValueError(
                                    "DAgger replay actions must all be finite"
                                )
                            if (host.abs() > self.action_clip).any():
                                raise ValueError(
                                    "DAgger replay action lies outside the "
                                    "configured symmetric support"
                                )
                        self.data[name][start:end].copy_(host)
            self.size = size
            self.seen = seen
            self.ptr = size % self.capacity
            self.last_snapshot_iteration = int(
                replay.attrs.get("snapshot_iteration", -1)
            )
            self.last_snapshot_name = str(replay.attrs.get("checkpoint_name", ""))
            self.last_snapshot_id = str(replay.attrs.get("snapshot_id", "")) or None
            self.last_snapshot_size = size
            self.last_snapshot_seen = seen
        return self.size

    def snapshot(self, iteration, checkpoint_name, row_count=None, seen_count=None):
        snapshot_size = self.size if row_count is None else min(
            int(row_count), self.size
        )
        snapshot_seen = self.seen if seen_count is None else int(seen_count)
        if snapshot_size == 0:
            return None
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to snapshot DAgger replay") from exc

        # A pure-beta rollout adds no teacher-executed rows after beta reaches
        # zero.  Keep the immutable snapshot lineage already referenced by the
        # checkpoint instead of rewriting the same (often 20+ GiB) H5 at every
        # later save.  Validate the file identity so a restore from a different
        # source path, a deleted file, or a stale replacement still rewrites it.
        if (
            row_count is None
            and seen_count is None
            and self.last_snapshot_id is not None
            and self.last_snapshot_size == snapshot_size
            and self.last_snapshot_seen == snapshot_seen
            and os.path.isfile(self.path)
        ):
            try:
                with h5py.File(self.path, "r") as replay:
                    snapshot_is_current = (
                        str(replay.attrs.get("replay_id", "")) == self.replay_id
                        and str(replay.attrs.get("snapshot_id", ""))
                        == self.last_snapshot_id
                        and int(replay.attrs.get("num_transitions", -1))
                        == snapshot_size
                        and int(replay.attrs.get("num_seen_transitions", -1))
                        == snapshot_seen
                    )
            except OSError:
                snapshot_is_current = False
            if snapshot_is_current:
                return self.path

        snapshot_id = str(uuid.uuid4())
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.path)}.", suffix=".tmp", dir=directory
        )
        os.close(fd)
        try:
            with h5py.File(temporary, "w") as replay:
                replay.attrs.update(
                    {
                        "format": DAGGER_TEACHER_REPLAY_FORMAT,
                        "format_version": DAGGER_TEACHER_REPLAY_FORMAT_VERSION,
                        "source": (
                            "frozen_ppo_teacher_executed_raw_observations"
                        ),
                        "teacher_only": True,
                        "storage_policy": "circular_fifo",
                        "storage_order": "oldest_to_newest",
                        "initial_transition_filter": DAGGER_INITIAL_TRANSITION_FILTER,
                        "action_parameterization": DAGGER_ACTION_PARAMETERIZATION,
                        "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
                        "replay_observation_semantics": DAGGER_REPLAY_OBSERVATION_SEMANTICS,
                        "vecnorm_fingerprint": self.vecnorm_fingerprint,
                        "reward_scalarization": SAC_REWARD_SCALARIZATION,
                        "truncation_next_observation": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
                        "actor_backend": PPO_BC_DAGGER_ACTOR_BACKEND,
                        "replay_id": self.replay_id,
                        "actor_obs_keys": json.dumps(self.actor_obs_keys),
                        "critic_obs_keys": json.dumps(self.critic_obs_keys),
                        "storage_fields": json.dumps(list(self.storage_fields)),
                        "field_shapes": json.dumps(
                            {
                                name: list(self.shapes[name])
                                for name in self.storage_fields
                            }
                        ),
                        "actor_obs_dim": self.actor_dim,
                        "critic_obs_dim": self.critic_dim,
                        "action_dim": self.action_dim,
                        "action_clip": self.action_clip,
                        "buffer_capacity": self.capacity,
                        "num_transitions": snapshot_size,
                        "num_seen_transitions": snapshot_seen,
                        "snapshot_iteration": int(iteration),
                        "checkpoint_name": str(checkpoint_name),
                        "snapshot_id": snapshot_id,
                    }
                )
                datasets = {
                    name: replay.create_dataset(
                        name,
                        (snapshot_size, *self.shapes[name]),
                        dtype=(
                            np.bool_
                            if self.dtypes[name] == torch.bool
                            else np.float32
                        ),
                    )
                    for name in self.storage_fields
                }
                destination = 0
                for segment_start, segment_end in self._chronological_segments(
                    snapshot_size
                ):
                    for start in range(
                        segment_start, segment_end, self.snapshot_chunk_rows
                    ):
                        end = min(start + self.snapshot_chunk_rows, segment_end)
                        count = end - start
                        for name in self.storage_fields:
                            datasets[name][destination : destination + count] = (
                                self.data[name][start:end].to("cpu").numpy()
                            )
                        destination += count
                replay.flush()
            file_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.replace(temporary, self.path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        self.last_snapshot_iteration = int(iteration)
        self.last_snapshot_name = str(checkpoint_name)
        self.last_snapshot_id = snapshot_id
        self.last_snapshot_size = snapshot_size
        self.last_snapshot_seen = snapshot_seen
        return self.path


class _DaggerRolloutPolicy(nn.Module):
    """Run both deterministic policies and choose one source per environment."""

    def __init__(self, owner: "PPOBCDaggerFinetune"):
        super().__init__()
        # Avoid registering the owner as our child (which would create a module
        # cycle and duplicate every parameter in state_dict()).
        object.__setattr__(self, "_owner", owner)
        # The shared resume path starts from a fresh env.reset(), so this
        # transient per-environment latch intentionally starts empty on resume.
        self._safe_teacher_active = None
        self._safe_teacher_hold = None

    def _safe_selection(
        self,
        valid,
        student_valid,
        discrepancy_rms,
        reset,
    ):
        owner = self._owner
        if (
            self._safe_teacher_active is None
            or self._safe_teacher_active.shape != valid.shape
            or self._safe_teacher_active.device != valid.device
        ):
            active = torch.zeros_like(valid)
            hold = torch.zeros_like(valid, dtype=torch.long)
        else:
            active = self._safe_teacher_active
            hold = self._safe_teacher_hold

        active = active & valid & ~reset
        hold = torch.where(active, hold, torch.zeros_like(hold))
        unsafe = valid & (
            ~student_valid
            | (discrepancy_rms > float(owner.cfg.dagger_safe_takeover_rms))
        )
        takeover = valid & ~active & unsafe
        active = active | takeover
        hold = torch.where(
            takeover,
            torch.full_like(hold, int(owner.cfg.dagger_safe_min_teacher_steps)),
            hold,
        )
        releasable = valid & active & ~takeover & (hold <= 0)
        release = (
            releasable
            & student_valid
            & (discrepancy_rms < float(owner.cfg.dagger_safe_release_rms))
        )
        active = active & ~release
        hold = torch.where(
            active & (hold > 0), hold - 1, torch.zeros_like(hold)
        )
        self._safe_teacher_active = active
        self._safe_teacher_hold = hold
        return active, unsafe, takeover, release

    @torch.no_grad()
    def forward(self, td: TensorDict):
        owner = self._owner
        raw_student_action = owner._student_action(td)
        # Unlike PPO, DAgger does not need the old distribution parameters or
        # perception intermediates in its N x T rollout.  Keep only PRIV_PRED
        # and recurrent next-state keys needed by replay/adaptation; otherwise
        # the student depth rollout carries hundreds of needless features per
        # environment step.
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
        # The privileged pass writes generic actor scratch keys (loc/scale).
        # Query it on a shallow container clone so those values cannot replace
        # the student actor's rollout scratch state.
        teacher_action = owner._teacher_action(td.clone(False))
        valid = _valid_teacher_action_rows(
            teacher_action, owner.cfg.dagger_teacher_action_threshold
        )
        # Invalid teacher rows are retained only as masked labels; never expose
        # NaN/Inf through a rollout TensorDict or to the environment action path.
        action_clip = float(owner.cfg.dagger_action_clip)
        clipped_teacher_action = torch.nan_to_num(
            teacher_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)
        student_valid = torch.isfinite(raw_student_action).all(dim=-1) & (
            raw_student_action.abs() <= action_clip
        ).all(dim=-1)
        clipped_student_action = torch.nan_to_num(
            raw_student_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)
        discrepancy_rms, discrepancy_max = _normalized_action_discrepancy(
            clipped_student_action, clipped_teacher_action, action_clip
        )
        discrepancy_rms = torch.where(
            valid, discrepancy_rms, torch.zeros_like(discrepancy_rms)
        )
        discrepancy_max = torch.where(
            valid, discrepancy_max, torch.zeros_like(discrepancy_max)
        )
        reset = td.get("is_init", None)
        if reset is None:
            reset = torch.zeros_like(valid)
        else:
            reset = reset.reshape(valid.shape).bool()

        control_mode = owner._effective_control_mode()
        safe_teacher_mask = torch.zeros_like(valid)
        safe_unsafe = torch.zeros_like(valid)
        safe_takeover = torch.zeros_like(valid)
        safe_release = torch.zeros_like(valid)
        if control_mode in ("safe", "hybrid"):
            if owner._safe_teacher_control_enabled():
                (
                    safe_teacher_mask,
                    safe_unsafe,
                    safe_takeover,
                    safe_release,
                ) = self._safe_selection(
                    valid,
                    student_valid,
                    discrepancy_rms,
                    reset,
                )
            else:
                # A configured cutoff is an explicit safety override. Preserve
                # the counterfactual unsafe diagnostic, but clear hysteresis and
                # never execute the teacher from the safe branch at/after K.
                safe_unsafe = valid & (
                    ~student_valid
                    | (
                        discrepancy_rms
                        > float(owner.cfg.dagger_safe_takeover_rms)
                    )
                )
                self._safe_teacher_active = torch.zeros_like(valid)
                self._safe_teacher_hold = torch.zeros_like(
                    valid, dtype=torch.long
                )
        elif control_mode != "beta":
            raise RuntimeError(
                f"Unsupported dagger_control_mode={control_mode!r}"
            )

        beta_teacher = torch.zeros_like(valid)
        if control_mode in ("beta", "hybrid"):
            beta = owner._teacher_mixture_probability()
            beta_teacher = (
                torch.rand(
                    valid.shape,
                    device=valid.device,
                    generator=owner.dagger_rng,
                )
                < beta
            ) & valid & ~safe_teacher_mask
        choose_teacher = safe_teacher_mask | beta_teacher
        td[ACTION_KEY] = torch.where(
            choose_teacher.unsqueeze(-1),
            clipped_teacher_action,
            clipped_student_action,
        )
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
        return td


class _ClippedStudentRolloutPolicy(nn.Module):
    """Keep play/evaluation action semantics identical to DAgger collection."""

    def __init__(self, policy: nn.Module, action_clip: float):
        super().__init__()
        self.policy = policy
        self.action_clip = float(action_clip)

    @torch.no_grad()
    def forward(self, td: TensorDict):
        td = self.policy(td)
        td[ACTION_KEY] = torch.nan_to_num(
            td[ACTION_KEY],
            nan=0.0,
            posinf=self.action_clip,
            neginf=-self.action_clip,
        ).clamp(-self.action_clip, self.action_clip)
        return td


class PPOBCDaggerFinetune(PPOVEL):
    """Depth student with pure DAgger BC and a FastSAC-compatible critic."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_config(cfg)
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)

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
            int(observation_spec[key].shape[-1])
            for key in self.q_critic_keys
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
        action_clip = float(cfg.dagger_action_clip)
        action_low = torch.full(
            (self.action_dim,), -action_clip, device=device, dtype=torch.float32
        )
        action_high = torch.full_like(action_low, action_clip)
        self.sac_dist_cls = functools.partial(
            FastSACTanhNormal,
            low=action_low,
            high=action_high,
            event_dims=1,
        )
        initial_log_std = math.log(
            float(cfg.sac_bc_initial_action_std) / action_clip
        )
        self.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
            self.action_dim, initial_log_std, device
        )
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=cfg.q_lr,
            weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95),
            fused=str(device).startswith("cuda"),
        )
        self.bc_optimizer = torch.optim.AdamW(
            self.actor_adapt.parameters(),
            lr=cfg.dagger_bc_lr,
            weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95),
            fused=str(device).startswith("cuda"),
        )

        self.opt_policy = None
        self.opt_critic = None
        self.actor_backend = PPO_BC_DAGGER_ACTOR_BACKEND
        self._freeze_teacher()

        self.dagger_replay = _DeviceReplay(
            cfg.dagger_buffer_capacity,
            _resolve_replay_device(cfg.dagger_buffer_device, device),
        )
        # This separate FIFO preserves teacher-executed rows after beta reaches
        # zero. The normal DAgger ring remains unchanged and continues to own
        # BC sampling from the recent state distribution.
        self.q_teacher_replay = _DeviceReplay(
            cfg.q_teacher_buffer_capacity,
            _resolve_replay_device(cfg.dagger_buffer_device, device),
        )
        self.teacher_replay = None
        self.teacher_replay_id = str(uuid.uuid4())
        self._loaded_teacher_replay_metadata = None
        object.__setattr__(self, "_replay_vecnorm", None)
        self._replay_vecnorm_keys = set()
        self._replay_vecnorm_fingerprint = None
        self._rollout_final_batch = None
        self._truncation_final_batches = []
        self._last_truncation_finals_used = 0

        generator_device = torch.device(device)
        self.dagger_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.dagger_seed)
        )
        self.q_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed)
        )
        self.sac_action_rng = torch.Generator(
            device=generator_device
        ).manual_seed(int(cfg.q_seed) + 1)
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.bc_update_count = 0
        self.q_update_count = 0
        self.finalization_rollout_count = 0
        self._finalization_last_phase = None
        self._finalization_source_state = None
        self.staging_rollout_count = 0
        self._staging_last_phase = None
        self._staging_calibration_start_q_update_count = None

    @staticmethod
    def _validate_config(cfg):
        _linear_teacher_probability(
            cfg.dagger_beta_start,
            cfg.dagger_beta_end,
            cfg.dagger_beta_decay_rollouts,
            0,
        )
        if cfg.phase != "finetune" or cfg.vecnorm != "eval":
            raise ValueError("PPO-BC DAgger requires phase=finetune and vecnorm=eval")
        if not bool(cfg.dagger_replay_raw_observations):
            raise ValueError(
                "PPO-BC DAgger requires dagger_replay_raw_observations=true"
            )
        if cfg.enable_residual_distillation:
            raise ValueError(
                "PPO-BC DAgger owns actor_adapt with one BC optimizer; disable "
                "the inherited residual-distillation optimizer"
            )
        control_mode = str(cfg.dagger_control_mode)
        if control_mode not in DAGGER_CONTROL_MODES:
            raise ValueError(
                "dagger_control_mode must be one of "
                f"{DAGGER_CONTROL_MODES}, got {control_mode!r}"
            )
        safe_zero_iteration = getattr(
            cfg, "dagger_safe_zero_iteration", None
        )
        if safe_zero_iteration is not None:
            if (
                isinstance(safe_zero_iteration, bool)
                or not isinstance(safe_zero_iteration, int)
                or safe_zero_iteration < 1
            ):
                raise ValueError(
                    "dagger_safe_zero_iteration must be a positive integer"
                )
            if control_mode == "beta":
                raise ValueError(
                    "dagger_safe_zero_iteration requires safe or hybrid control"
                )
        if bool(getattr(cfg, "dagger_finalization_enabled", False)):
            stage_names = (
                "dagger_finalize_perception_iterations",
                "dagger_finalize_actor_iterations",
                "dagger_finalize_recheck_iterations",
                "dagger_finalize_calibration_iterations",
            )
            for name in stage_names:
                value = getattr(cfg, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
            if int(cfg.dagger_finalize_calibration_iterations) < 1:
                raise ValueError(
                    "BC-DAgger finalization requires at least one fresh "
                    "replay/Q calibration rollout"
                )
            calibration_mode = str(
                cfg.dagger_finalize_calibration_control_mode
            )
            if calibration_mode != "beta":
                raise ValueError(
                    "dagger_finalize_calibration_control_mode currently "
                    "requires 'beta'"
                )
            probability = float(
                cfg.dagger_finalize_calibration_teacher_probability
            )
            if not math.isfinite(probability) or not 0.0 < probability < 1.0:
                raise ValueError(
                    "dagger_finalize_calibration_teacher_probability must be "
                    "finite and strictly between zero and one"
                )
        staging_enabled = bool(
            getattr(cfg, "dagger_staging_enabled", False)
        )
        if staging_enabled:
            if bool(getattr(cfg, "dagger_finalization_enabled", False)):
                raise ValueError(
                    "BC-DAgger staging and finalization are mutually exclusive"
                )
            staging_iteration_names = (
                "dagger_stage_joint_warmup_iterations",
                "dagger_stage_cycles",
                "dagger_stage_perception_iterations",
                "dagger_stage_actor_iterations",
                "dagger_stage_final_perception_iterations",
                "dagger_stage_final_actor_iterations",
                "dagger_stage_calibration_iterations",
            )
            for name in staging_iteration_names:
                value = getattr(cfg, name, 0)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
            if int(getattr(cfg, "dagger_stage_calibration_iterations", 0)) < 1:
                raise ValueError(
                    "Staged BC-DAgger requires at least one fresh replay/Q "
                    "calibration rollout"
                )
            if str(
                getattr(cfg, "dagger_stage_calibration_control_mode", "beta")
            ) != "beta":
                raise ValueError(
                    "dagger_stage_calibration_control_mode currently requires "
                    "'beta'"
                )
            probability = float(
                getattr(
                    cfg,
                    "dagger_stage_calibration_teacher_probability",
                    0.5,
                )
            )
            if not math.isfinite(probability) or not 0.0 < probability < 1.0:
                raise ValueError(
                    "dagger_stage_calibration_teacher_probability must be "
                    "finite and strictly between zero and one"
                )
        for name in (
            "dagger_bc_epochs",
            "dagger_safe_min_teacher_steps",
            "dagger_buffer_capacity",
            "dagger_batch_size",
            "dagger_updates_per_rollout",
            "q_hidden_dim",
            "q_num_atoms",
            "q_batch_size",
            "q_updates_per_rollout",
            "q_teacher_buffer_capacity",
            "q_learning_starts_per_source",
            "teacher_buffer_capacity",
            "teacher_buffer_snapshot_chunk_rows",
        ):
            if isinstance(getattr(cfg, name), bool) or int(getattr(cfg, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "q_lr",
            "sac_bc_initial_action_std",
            "sac_alpha_init",
            "sac_entropy_reference_scale",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "dagger_teacher_action_threshold",
            "dagger_action_clip",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        release = float(cfg.dagger_safe_release_rms)
        takeover = float(cfg.dagger_safe_takeover_rms)
        if not (
            math.isfinite(release)
            and math.isfinite(takeover)
            and 0.0 <= release < takeover <= 2.0
        ):
            raise ValueError(
                "SafeDAgger thresholds require 0 <= "
                "dagger_safe_release_rms < dagger_safe_takeover_rms <= 2"
            )
        if int(cfg.q_hidden_dim) < 4:
            raise ValueError("q_hidden_dim must be at least 4")
        if cfg.q_num_atoms < 2 or not cfg.q_v_min < cfg.q_v_max:
            raise ValueError("distributional Q support is invalid")
        if int(cfg.q_num_atoms) != 501:
            raise ValueError("BC-DAgger FastSAC critic requires q_num_atoms=501")
        if not bool(cfg.q_layer_norm):
            raise ValueError("BC-DAgger FastSAC critic requires LayerNorm")
        if str(cfg.q_action_fusion) != "early":
            raise ValueError("BC-DAgger FastSAC critic requires early action fusion")
        if str(cfg.q_action_coordinates) != "absolute":
            raise ValueError(
                "BC-DAgger FastSAC critic requires absolute action coordinates"
            )
        if not bool(cfg.sac_q_normalize_actions):
            raise ValueError(
                "BC-DAgger FastSAC critic requires normalized absolute actions"
            )
        if not math.isclose(
            float(cfg.sac_q_action_input_gain),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("BC-DAgger FastSAC critic requires action gain 1")
        if not bool(cfg.sac_clipped_double_q):
            raise ValueError("BC-DAgger FastSAC critic requires clipped double Q")
        if int(cfg.q_batch_size) % 2:
            raise ValueError("q_batch_size must be even for an exact 50/50 draw")
        if not math.isclose(
            float(cfg.q_teacher_replay_ratio),
            0.5,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "q_teacher_replay_ratio is fixed at 0.5 independently of beta"
            )
        if not 0.0 <= float(cfg.q_tau) <= 1.0:
            raise ValueError("q_tau must be in [0, 1]")
        if not (
            math.isfinite(float(cfg.sac_bc_log_std_min))
            and math.isfinite(float(cfg.sac_bc_log_std_max))
            and float(cfg.sac_bc_log_std_min) < float(cfg.sac_bc_log_std_max)
        ):
            raise ValueError("SAC log-std bounds must be finite and ordered")
        initial_log_std = math.log(
            float(cfg.sac_bc_initial_action_std)
            / float(cfg.dagger_action_clip)
        )
        if not (
            float(cfg.sac_bc_log_std_min)
            <= initial_log_std
            <= float(cfg.sac_bc_log_std_max)
        ):
            raise ValueError(
                "sac_bc_initial_action_std lies outside the configured "
                "dedicated SAC log-std bounds"
            )
        if not math.isfinite(float(cfg.q_max_grad_norm)) or cfg.q_max_grad_norm < 0:
            raise ValueError("q_max_grad_norm must be finite and non-negative")
        if not math.isfinite(float(cfg.q_weight_decay)) or cfg.q_weight_decay < 0:
            raise ValueError("q_weight_decay must be finite and non-negative")
        replay_filename = os.fspath(cfg.teacher_buffer_filename)
        if (
            replay_filename in ("", ".", "..")
            or os.path.basename(replay_filename) != replay_filename
        ):
            raise ValueError("teacher_buffer_filename must be a plain filename")

    def _freeze_teacher(self):
        for name in ("actor", "encoder_priv", "critic", "height_encoder"):
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def requires_training_replay(self):
        # The all-transition learning ring is constructed in __init__ on every
        # rank.  Returning false here makes train.py configure the much larger
        # teacher-only H5 FIFO only on the checkpoint-writing rank.
        return False

    def requires_value_bootstrap(self):
        return False

    def _finalization_config(self):
        return {
            "semantics": DAGGER_FINALIZATION_SEMANTICS,
            "perception_consolidation_iterations": int(
                getattr(self.cfg, "dagger_finalize_perception_iterations", 0)
            ),
            "actor_realignment_iterations": int(
                getattr(self.cfg, "dagger_finalize_actor_iterations", 0)
            ),
            "perception_recheck_iterations": int(
                getattr(self.cfg, "dagger_finalize_recheck_iterations", 0)
            ),
            "replay_q_calibration_iterations": int(
                getattr(self.cfg, "dagger_finalize_calibration_iterations", 0)
            ),
            "calibration_control_mode": str(
                getattr(
                    self.cfg,
                    "dagger_finalize_calibration_control_mode",
                    "beta",
                )
            ),
            "calibration_teacher_probability": float(
                getattr(
                    self.cfg,
                    "dagger_finalize_calibration_teacher_probability",
                    0.5,
                )
            ),
        }

    def _finalization_enabled(self):
        return bool(getattr(self.cfg, "dagger_finalization_enabled", False))

    def _finalization_phase(self, rollout_count=None):
        if not self._finalization_enabled():
            return None
        index = (
            self.finalization_rollout_count
            if rollout_count is None
            else int(rollout_count)
        )
        config = self._finalization_config()
        boundary = 0
        lengths = (
            config["perception_consolidation_iterations"],
            config["actor_realignment_iterations"],
            config["perception_recheck_iterations"],
            config["replay_q_calibration_iterations"],
        )
        for phase, length in zip(DAGGER_FINALIZATION_PHASES, lengths):
            boundary += int(length)
            if index < boundary:
                return phase
        return "complete"

    def _staging_config(self):
        return {
            "semantics": DAGGER_STAGING_SEMANTICS,
            "joint_warmup_iterations": int(
                getattr(self.cfg, "dagger_stage_joint_warmup_iterations", 0)
            ),
            "cycles": int(getattr(self.cfg, "dagger_stage_cycles", 0)),
            "perception_iterations": int(
                getattr(self.cfg, "dagger_stage_perception_iterations", 0)
            ),
            "actor_iterations": int(
                getattr(self.cfg, "dagger_stage_actor_iterations", 0)
            ),
            "final_perception_iterations": int(
                getattr(
                    self.cfg,
                    "dagger_stage_final_perception_iterations",
                    0,
                )
            ),
            "final_actor_iterations": int(
                getattr(self.cfg, "dagger_stage_final_actor_iterations", 0)
            ),
            "replay_q_calibration_iterations": int(
                getattr(self.cfg, "dagger_stage_calibration_iterations", 0)
            ),
            "calibration_control_mode": str(
                getattr(
                    self.cfg,
                    "dagger_stage_calibration_control_mode",
                    "beta",
                )
            ),
            "calibration_teacher_probability": float(
                getattr(
                    self.cfg,
                    "dagger_stage_calibration_teacher_probability",
                    0.5,
                )
            ),
        }

    def _staging_enabled(self):
        return bool(getattr(self.cfg, "dagger_staging_enabled", False))

    def _staging_phase_details(self, rollout_count=None):
        if not self._staging_enabled():
            return None, -1
        index = (
            self.staging_rollout_count
            if rollout_count is None
            else int(rollout_count)
        )
        if index < 0:
            raise ValueError("staging rollout count must be non-negative")
        config = self._staging_config()
        boundary = config["joint_warmup_iterations"]
        if index < boundary:
            return "joint_warmup", -1

        cycle_width = (
            config["perception_iterations"] + config["actor_iterations"]
        )
        if cycle_width:
            cycle_offset = index - boundary
            cycle_span = config["cycles"] * cycle_width
            if cycle_offset < cycle_span:
                cycle_index = cycle_offset // cycle_width
                within_cycle = cycle_offset % cycle_width
                if within_cycle < config["perception_iterations"]:
                    return "cycle_perception", int(cycle_index)
                return "cycle_actor", int(cycle_index)
            boundary += cycle_span

        boundary += config["final_perception_iterations"]
        if index < boundary:
            return "final_perception", -1
        boundary += config["final_actor_iterations"]
        if index < boundary:
            return "final_actor", -1
        boundary += config["replay_q_calibration_iterations"]
        if index < boundary:
            return "replay_q_calibration", -1
        return "complete", -1

    def _staging_phase(self, rollout_count=None):
        return self._staging_phase_details(rollout_count)[0]

    def _staging_cycle_index(self, rollout_count=None):
        return self._staging_phase_details(rollout_count)[1]

    def _effective_control_mode(self):
        if self._finalization_enabled():
            if self._finalization_phase() == "replay_q_calibration":
                return self._finalization_config()["calibration_control_mode"]
            # Consolidation/realignment are pure-student rollouts. The teacher
            # is still evaluated on every row for labels and diagnostics.
            return "beta"
        if (
            self._staging_enabled()
            and self._staging_phase() == "replay_q_calibration"
        ):
            return self._staging_config()["calibration_control_mode"]
        return str(getattr(self.cfg, "dagger_control_mode", "beta"))

    def _teacher_mixture_probability(self):
        if self._finalization_enabled():
            if self._finalization_phase() == "replay_q_calibration":
                return self._finalization_config()[
                    "calibration_teacher_probability"
                ]
            return 0.0
        if (
            self._staging_enabled()
            and self._staging_phase() == "replay_q_calibration"
        ):
            return self._staging_config()[
                "calibration_teacher_probability"
            ]
        return _linear_teacher_probability(
            self.cfg.dagger_beta_start,
            self.cfg.dagger_beta_end,
            self.cfg.dagger_beta_decay_rollouts,
            self.dagger_rollout_count,
        )

    def _safe_teacher_control_enabled(self):
        """Whether SafeDAgger may execute the teacher this cumulative rollout."""
        if self._finalization_enabled():
            return False
        if (
            self._staging_enabled()
            and self._staging_phase() == "replay_q_calibration"
        ):
            return False
        cutoff = getattr(self.cfg, "dagger_safe_zero_iteration", None)
        return cutoff is None or self.dagger_rollout_count < int(cutoff)

    def _collect_dagger_replay_this_rollout(self):
        if self._finalization_enabled():
            return self._finalization_phase() in (
                "actor_realignment",
                "replay_q_calibration",
            )
        if self._staging_enabled():
            return self._staging_phase() in (
                "joint_warmup",
                "cycle_actor",
                "final_actor",
                "replay_q_calibration",
            )
        return True

    def _actor_updates_this_rollout(self):
        if self._finalization_enabled():
            enabled = self._finalization_phase() == "actor_realignment"
        elif self._staging_enabled():
            enabled = self._staging_phase() in (
                "joint_warmup",
                "cycle_actor",
                "final_actor",
            )
        else:
            enabled = True
        if enabled:
            return int(self.cfg.dagger_updates_per_rollout)
        return 0

    def _q_updates_this_rollout(self):
        if self._finalization_enabled():
            enabled = self._finalization_phase() == "replay_q_calibration"
        elif self._staging_enabled():
            enabled = self._staging_phase() in (
                "joint_warmup",
                "replay_q_calibration",
            )
        else:
            enabled = True
        if enabled:
            return int(self.cfg.q_updates_per_rollout)
        return 0

    def _collect_q_teacher_rows_this_rollout(self):
        if self._finalization_enabled():
            return self._finalization_phase() == "replay_q_calibration"
        if self._staging_enabled():
            return self._staging_phase() in (
                "joint_warmup",
                "replay_q_calibration",
            )
        return True

    def _export_teacher_rows_this_rollout(self):
        if self._finalization_enabled():
            return self._finalization_phase() == "replay_q_calibration"
        if self._staging_enabled():
            return self._staging_phase() == "replay_q_calibration"
        return True

    def _adaptation_update_this_rollout(self, rollout):
        if self._finalization_enabled():
            enabled = self._finalization_phase() in (
                "perception_consolidation",
                "perception_recheck",
            )
        elif self._staging_enabled():
            enabled = self._staging_phase() in (
                "joint_warmup",
                "cycle_perception",
                "final_perception",
            )
        else:
            enabled = True
        if enabled:
            return self.train_adapt(rollout.copy())
        return {}

    def _prepare_finalization_phase(self):
        phase = self._finalization_phase()
        if phase is None or phase == self._finalization_last_phase:
            return phase
        if phase in (
            "actor_realignment",
            "perception_recheck",
            "replay_q_calibration",
        ):
            self.dagger_replay.clear()
        if phase == "replay_q_calibration":
            self.q_teacher_replay.clear()
            if self.teacher_replay is not None and (
                self.teacher_replay.size or self.teacher_replay.seen
            ):
                raise RuntimeError(
                    "Fresh finalization H5 received rows before replay/Q "
                    "calibration"
                )
        self._apply_finalization_freeze_mask(phase)
        self._finalization_last_phase = phase
        return phase

    def _prepare_staging_phase(self):
        phase = self._staging_phase()
        if phase is None or phase == self._staging_last_phase:
            return phase
        # Actor replay embeds the EMA-produced priv_pred. Every perception/actor
        # handoff therefore starts with rows from the newly frozen producer.
        if phase in (
            "cycle_perception",
            "cycle_actor",
            "final_perception",
            "final_actor",
            "replay_q_calibration",
        ):
            self.dagger_replay.clear()
        if phase == "replay_q_calibration":
            # Joint warmup Q rows are useful only while that joint policy is
            # fixed. The terminal Q/H5 lineage must describe the final frozen
            # actor and perception representation exclusively.
            self.q_teacher_replay.clear()
            if getattr(
                self,
                "_staging_calibration_start_q_update_count",
                None,
            ) is None:
                self._staging_calibration_start_q_update_count = int(
                    self.q_update_count
                )
            if self.teacher_replay is not None and (
                self.teacher_replay.size or self.teacher_replay.seen
            ):
                raise RuntimeError(
                    "Staged BC-DAgger H5 received rows before final replay/Q "
                    "calibration"
                )
        self._apply_staging_freeze_mask(phase)
        self._staging_last_phase = phase
        return phase

    def _prepare_training_phase(self):
        if self._finalization_enabled():
            return self._prepare_finalization_phase()
        if self._staging_enabled():
            return self._prepare_staging_phase()
        return None

    def _apply_optimizer_ownership(
        self,
        *,
        perception_enabled: bool,
        actor_enabled: bool,
        q_enabled: bool,
    ):
        perception = [self.adapt_module]
        if hasattr(self, "object_adapt"):
            perception.append(self.object_adapt)
        if hasattr(self, "temporal_depth_gru"):
            perception.append(self.temporal_depth_gru)
        for module in (*perception, self.actor_adapt, self.qnet):
            module.requires_grad_(False)
            module.eval()
            for parameter in module.parameters():
                parameter.grad = None
        if perception_enabled:
            for module in perception:
                module.requires_grad_(True)
                module.train()
        if actor_enabled:
            self.actor_adapt.requires_grad_(True)
            self.actor_adapt.train()
        if q_enabled:
            self.qnet.requires_grad_(True)
            self.qnet.train()

    def _apply_finalization_freeze_mask(self, phase):
        """Make the phase's optimizer ownership explicit at module level."""
        self._apply_optimizer_ownership(
            perception_enabled=phase in (
                "perception_consolidation",
                "perception_recheck",
            ),
            actor_enabled=phase == "actor_realignment",
            q_enabled=phase == "replay_q_calibration",
        )

    def _apply_staging_freeze_mask(self, phase):
        """Expose only the optimizer owners selected by the staged phase."""
        self._apply_optimizer_ownership(
            perception_enabled=phase in (
                "joint_warmup",
                "cycle_perception",
                "final_perception",
            ),
            actor_enabled=phase in (
                "joint_warmup",
                "cycle_actor",
                "final_actor",
            ),
            q_enabled=phase in (
                "joint_warmup",
                "replay_q_calibration",
            ),
        )

    @torch.no_grad()
    def _teacher_action(self, td: TensorDict):
        """Return the checkpoint PPO teacher in executable absolute coordinates."""
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        self.encoder_priv(td)
        residual = self.actor.get_dist(td).mean
        return (td[REF_JPOS_KEY] + residual).detach()

    @torch.no_grad()
    def _student_action(self, td: TensorDict):
        if hasattr(self, "temporal_depth_gru_ema"):
            self.temporal_depth_gru_ema(td)
        else:
            ZeroDepthInjector(self.depth_feature_dim, self.device)(td)
        if self.cfg.use_object_adapt:
            self.object_adapt_ema(td)
            self.object_pred_transform(td)
        self.adapt_ema(td)
        return self.actor_adapt.get_dist(td).mean.detach()

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return _DaggerRolloutPolicy(self)
        return _ClippedStudentRolloutPolicy(
            super().get_rollout_policy(mode), self.cfg.dagger_action_clip
        )

    def configure_teacher_replay(self, path, restore_path=None):
        if not self.cfg.save_teacher_buffer:
            self.teacher_replay = None
            return
        if self._replay_vecnorm_fingerprint is None:
            raise RuntimeError(
                "Raw PPO-BC DAgger replay requires the checkpoint VecNorm "
                "before configuring its H5"
            )
        if (
            restore_path is not None
            and self._loaded_teacher_replay_metadata is not None
            and self._loaded_teacher_replay_metadata.get(
                "replay_observation_semantics"
            ) == DAGGER_LEGACY_REPLAY_OBSERVATION_SEMANTICS
        ):
            raise ValueError(
                "A legacy normalized DAgger H5 cannot be resumed into the new "
                "raw mutable FIFO because that would mix observation "
                "coordinates. It remains valid as read-only Stage-2 offline "
                "data; start a fresh BC-DAgger run to collect raw replay."
            )
        self.teacher_replay = _DaggerTeacherReplayBuffer(
            path,
            self.cfg.teacher_buffer_capacity,
            self._q_actor_dim,
            self._q_critic_dim,
            self.action_dim,
            self.cfg.dagger_seed,
            # The export FIFO is deliberately host-resident even when the user
            # puts the much smaller learning ring on the policy GPU.  At the
            # default VAIC dimensions a 524k-row export is about 11 GiB and
            # would otherwise exhaust a typical accelerator immediately.
            device=torch.device("cpu"),
            snapshot_chunk_rows=self.cfg.teacher_buffer_snapshot_chunk_rows,
            replay_id=self.teacher_replay_id,
            actor_backend=PPO_BC_DAGGER_ACTOR_BACKEND,
            actor_obs_keys=self.q_actor_keys,
            critic_obs_keys=self.q_critic_keys,
            vecnorm_fingerprint=self._replay_vecnorm_fingerprint,
            action_clip=self.cfg.dagger_action_clip,
        )
        if (
            restore_path is not None
            and self._loaded_teacher_replay_metadata is None
        ):
            # A PPO bootstrap has no paired DAgger replay.  helpers.py may find
            # another H5 beside that checkpoint; never ingest it implicitly.
            logging.warning(
                "Ignoring unpaired teacher replay %s while bootstrapping from "
                "the PPO teacher checkpoint.",
                restore_path,
            )
            restore_path = None
        if restore_path is not None:
            self.teacher_replay.restore(
                restore_path, self._loaded_teacher_replay_metadata
            )
        elif self._loaded_teacher_replay_metadata is not None:
            expected_size = int(
                self._loaded_teacher_replay_metadata.get("size", 0)
            )
            if expected_size > 0:
                raise FileNotFoundError(
                    "Same-stage PPO-BC DAgger resume requires the teacher H5 "
                    "paired with this checkpoint. Set algo.teacher_buffer_path "
                    "or keep teacher_replay_buffer.h5 beside the checkpoint."
                )

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        if self.teacher_replay is None:
            return None
        if self._staging_enabled() and self._staging_phase() not in (
            "replay_q_calibration",
            "complete",
        ):
            # Checkpoint the model/schedule normally, but do not create even an
            # empty persistent H5 for representation-changing staged phases.
            return None
        return self.teacher_replay.snapshot(iteration, checkpoint_name)

    @torch.no_grad()
    def restore_q_teacher_replay(self, source_path):
        """Refill the persistent critic teacher partition from its paired H5."""
        if self.q_teacher_replay.size or self.q_teacher_replay.seen:
            raise RuntimeError("Q teacher replay restore requires an empty FIFO")
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to restore the Q teacher replay"
            ) from exc
        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        with h5py.File(source_path, "r") as replay:
            required_attrs = {
                "format": DAGGER_TEACHER_REPLAY_FORMAT,
                "format_version": DAGGER_TEACHER_REPLAY_FORMAT_VERSION,
                "teacher_only": True,
                "action_parameterization": DAGGER_ACTION_PARAMETERIZATION,
                "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
                "replay_observation_semantics": (
                    DAGGER_REPLAY_OBSERVATION_SEMANTICS
                ),
                "reward_scalarization": SAC_REWARD_SCALARIZATION,
                "truncation_next_observation": (
                    TRUNCATION_NEXT_OBSERVATION_SEMANTICS
                ),
            }
            for name, expected in required_attrs.items():
                actual = replay.attrs.get(name)
                if isinstance(expected, bool):
                    actual = bool(actual)
                elif isinstance(expected, int):
                    actual = int(actual or 0)
                else:
                    actual = str(actual or "")
                if actual != expected:
                    raise ValueError(
                        f"Q teacher replay {name}={actual!r} does not match "
                        f"{expected!r}"
                    )
            if str(replay.attrs.get("vecnorm_fingerprint", "")) != str(
                self._replay_vecnorm_fingerprint
            ):
                raise ValueError("Q teacher replay VecNorm fingerprint mismatch")
            if not math.isclose(
                float(replay.attrs.get("action_clip", float("nan"))),
                float(self.cfg.dagger_action_clip),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Q teacher replay action clip mismatch")
            missing = set(TEACHER_REPLAY_FIELDS).difference(replay.keys())
            if missing:
                raise ValueError(
                    f"Q teacher replay is missing fields: {sorted(missing)}"
                )
            size = int(replay.attrs.get("num_transitions", -1))
            if size < 1:
                raise ValueError("Q teacher replay H5 is empty")
            expected_shapes = {
                "observations": (self._q_actor_dim,),
                "critic_observations": (self._q_critic_dim,),
                "actions": (self.action_dim,),
                "rewards": (),
                "dones": (),
                "truncations": (),
                "discounts": (),
                "next_observations": (self._q_actor_dim,),
                "next_critic_observations": (self._q_critic_dim,),
            }
            boolean_fields = {"dones", "truncations"}
            for key, trailing_shape in expected_shapes.items():
                dataset = replay[key]
                expected_shape = (size, *trailing_shape)
                if tuple(dataset.shape) != expected_shape:
                    raise ValueError(
                        f"Q teacher replay field {key!r} has shape "
                        f"{tuple(dataset.shape)}, expected {expected_shape}"
                    )
                if key in boolean_fields:
                    valid_dtype = np.issubdtype(dataset.dtype, np.bool_)
                else:
                    valid_dtype = np.issubdtype(dataset.dtype, np.floating)
                if not valid_dtype:
                    raise TypeError(
                        f"Q teacher replay field {key!r} has invalid dtype "
                        f"{dataset.dtype}"
                    )
            start = max(0, size - int(self.q_teacher_replay.capacity))
            chunk_rows = int(self.cfg.teacher_buffer_snapshot_chunk_rows)
            for chunk_start in range(start, size, chunk_rows):
                chunk_end = min(chunk_start + chunk_rows, size)
                values = {
                    key: torch.from_numpy(
                        np.asarray(replay[key][chunk_start:chunk_end])
                    )
                    for key in TEACHER_REPLAY_FIELDS
                }
                for key, value in values.items():
                    if value.is_floating_point() and not torch.isfinite(
                        value
                    ).all():
                        raise ValueError(
                            f"Q teacher replay field {key!r} contains "
                            "non-finite values"
                        )
                if (
                    values["actions"].abs()
                    > float(self.cfg.dagger_action_clip)
                ).any():
                    raise ValueError(
                        "Q teacher replay action lies outside the configured "
                        "executable support"
                    )
                self.q_teacher_replay.extend(values)
        logging.info(
            "Restored %d persistent teacher critic rows from %s.",
            self.q_teacher_replay.size,
            source_path,
        )
        return self.q_teacher_replay.size

    @staticmethod
    def _raw_replay_key(key):
        if isinstance(key, tuple):
            return (FASTSAC_RAW_OBSERVATION_ROOT, *key)
        return (FASTSAC_RAW_OBSERVATION_ROOT, key)

    def configure_replay_vecnorm(self, vecnorm):
        """Attach the frozen checkpoint VecNorm used by raw replay samples."""
        object.__setattr__(self, "_replay_vecnorm", vecnorm)
        self._replay_vecnorm_keys = set(vecnorm.in_keys)
        self._replay_vecnorm_fingerprint = _vecnorm_state_fingerprint(vecnorm)
        replay_sources = set(self.q_actor_keys).union(self.q_critic_keys)
        required_raw = replay_sources.intersection(self._replay_vecnorm_keys)
        missing = [
            key
            for key in required_raw
            if self.observation_spec.get(self._raw_replay_key(key), None) is None
        ]
        if missing:
            raise KeyError(
                "PPO-BC DAgger raw replay aliases are missing for normalized "
                f"observations: {sorted(missing)}"
            )

    def _replay_source(self, td, key):
        if key not in self._replay_vecnorm_keys:
            return td[key]
        raw_key = self._raw_replay_key(key)
        if raw_key not in td.keys(True, True):
            raise KeyError(
                "PPO-BC DAgger replay is missing raw observation alias "
                f"{raw_key!r}"
            )
        return td[raw_key]

    def _cat_replay_sources(self, td, keys):
        return torch.cat([self._replay_source(td, key) for key in keys], dim=-1)

    def _vecnorm_snapshot(self):
        vecnorm = self._replay_vecnorm
        if vecnorm is None:
            raise RuntimeError(
                "PPO-BC DAgger raw replay requires configure_replay_vecnorm()"
            )
        return vecnorm.loc, vecnorm.scale

    def _normalize_replay_value(self, key, value, snapshot):
        if key not in self._replay_vecnorm_keys:
            return value
        loc, scale = snapshot
        eps = float(self._replay_vecnorm.eps)
        return (value - loc[key]) / scale[key].clamp_min(eps)

    def _normalize_replay_flat(self, value, keys, widths, snapshot):
        chunks = []
        offset = 0
        for key, width in zip(keys, widths):
            chunk = value[..., offset : offset + width]
            chunks.append(self._normalize_replay_value(key, chunk, snapshot))
            offset += width
        if offset != value.shape[-1]:
            raise RuntimeError(
                f"DAgger replay split consumed {offset} of "
                f"{value.shape[-1]} values"
            )
        return torch.cat(chunks, dim=-1)

    def _prepare_dagger_learning_batch(self, batch):
        """Normalize present raw replay fields without mutating replay data.

        BC projects its replay sample to the three fields it actually consumes;
        Q still supplies all four observation fields.  Accepting a projected
        mapping here avoids transferring and normalizing the two 2,341-wide
        critic tensors and unused next state for every 4,096-row BC update.
        """
        snapshot = self._vecnorm_snapshot()
        prepared = dict(batch)
        for field in ("observations", "next_observations"):
            if field not in batch:
                continue
            prepared[field] = self._normalize_replay_flat(
                batch[field],
                self.q_actor_keys,
                self._q_actor_widths,
                snapshot,
            )
        for field in ("critic_observations", "next_critic_observations"):
            if field not in batch:
                continue
            prepared[field] = self._normalize_replay_flat(
                batch[field],
                self.q_critic_keys,
                self._q_critic_widths,
                snapshot,
            )
        return prepared

    @staticmethod
    def _scalarize_q_reward(reward):
        return reward.sum(dim=-1)

    @torch.no_grad()
    def _prepare_student_final_state(self, td: TensorDict):
        if hasattr(self, "temporal_depth_gru_ema"):
            self.temporal_depth_gru_ema(td)
        else:
            td["_depth_feature"] = torch.zeros(
                *td.batch_size,
                self.depth_feature_dim,
                device=td.device,
                dtype=td[OBS_KEY].dtype,
            )
        if self.cfg.use_object_adapt:
            self.object_adapt_ema(td)
            self.object_pred_transform(td)
        self.adapt_ema(td)
        return {
            "next_observations": self._cat_replay_sources(
                td, self.q_actor_keys
            ).clone(),
            "next_critic_observations": self._cat_replay_sources(
                td, self.q_critic_keys
            ).clone(),
        }

    @torch.no_grad()
    def capture_truncation_final_observations(self, td: TensorDict, step: int):
        if not self._collect_dagger_replay_this_rollout():
            self._truncation_final_batches = []
            self._last_truncation_finals_used = 0
            return
        truncations = _vaic_truncation_mask(td).reshape(-1).bool()
        if not truncations.any():
            return
        indices = truncations.nonzero(as_tuple=False).squeeze(-1)
        values = self._prepare_student_final_state(td["next"][indices].clone())
        values["indices"] = indices * int(self.cfg.train_every) + int(step)
        self._truncation_final_batches.append(values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        if not self._collect_dagger_replay_this_rollout():
            self._rollout_final_batch = None
            self._truncation_final_batches = []
            self._last_truncation_finals_used = 0
            return
        self._rollout_final_batch = self._prepare_student_final_state(carry.clone())

    def _dagger_transition_chunks(self, td: TensorDict):
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "PPO-BC DAgger requires capture_rollout_final_observation(carry)"
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError("rollout length does not match train_every")
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
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
            valid = td["step_count"].reshape(n * t) > DAGGER_REPLAY_MIN_STEP_COUNT
            self._last_truncation_finals_used = int(valid[flat_indices].sum())
        else:
            self._last_truncation_finals_used = 0

        for step in range(int(t)):
            current = td[:, step]
            if step + 1 < int(t):
                following = td[:, step + 1]
                next_values = {
                    "next_observations": self._cat_replay_sources(
                        following, self.q_actor_keys
                    ).reshape(n, self._q_actor_dim),
                    "next_critic_observations": self._cat_replay_sources(
                        following, self.q_critic_keys
                    ).reshape(n, self._q_critic_dim),
                }
            else:
                next_values = final_batch
            transitions = {
                "observations": self._cat_replay_sources(
                    current, self.q_actor_keys
                ).reshape(n, self._q_actor_dim),
                "critic_observations": self._cat_replay_sources(
                    current, self.q_critic_keys
                ).reshape(n, self._q_critic_dim),
                "actions": current[ACTION_KEY].reshape(n, self.action_dim),
                DAGGER_REPLAY_TEACHER_ACTIONS: current[
                    DAGGER_TEACHER_ACTION_KEY
                ].reshape(n, self.action_dim),
                DAGGER_TEACHER_ACTION_VALID_KEY: current[
                    DAGGER_TEACHER_ACTION_VALID_KEY
                ].reshape(n).bool(),
                DAGGER_IS_STUDENT_ACTION_KEY: current[
                    DAGGER_IS_STUDENT_ACTION_KEY
                ].reshape(n).bool(),
                "rewards": self._scalarize_q_reward(current[REWARD_KEY]).reshape(n),
                "dones": current[DONE_KEY].reshape(n).bool(),
                "truncations": _vaic_truncation_mask(current).reshape(n).bool(),
                "discounts": current["next", "discount"].reshape(n),
                **next_values,
            }
            if truncation_finals is not None:
                flat_indices = truncation_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    for key, values in truncation_finals.items():
                        if key == "indices":
                            continue
                        transitions[key] = transitions[key].clone()
                        transitions[key].index_copy_(
                            0, env_indices, values[selected]
                        )
            transitions, _ = _filter_replay_rows(
                current, transitions, DAGGER_REPLAY_MIN_STEP_COUNT
            )
            yield transitions

    def _actor_dist_from_flat(self, actor_obs):
        vel_dim = self.observation_spec[VEL_CMD_KEY].shape[-1]
        policy_dim = self.observation_spec[OBS_KEY].shape[-1]
        td = TensorDict(
            {
                VEL_CMD_KEY: actor_obs[..., :vel_dim],
                OBS_KEY: actor_obs[..., vel_dim : vel_dim + policy_dim],
                PRIV_PRED_KEY: actor_obs[..., vel_dim + policy_dim :],
            },
            batch_size=actor_obs.shape[:-1],
            device=actor_obs.device,
        )
        return self.actor_adapt.get_dist(td)

    def _sac_critic_dist_from_flat(self, actor_obs):
        """Bounded small-noise target policy centred on the DAgger BC mean."""
        mean_action = self._actor_dist_from_flat(actor_obs).mean
        action_clip = float(self.cfg.dagger_action_clip)
        mean_action = torch.nan_to_num(
            mean_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)
        normalized = (mean_action / action_clip).clamp(
            -1.0 + FASTSAC_REFERENCE_EPS,
            1.0 - FASTSAC_REFERENCE_EPS,
        )
        loc = torch.atanh(normalized)
        log_std = self.bc_dagger_sac_adapter.log_std.clamp(
            float(self.cfg.sac_bc_log_std_min),
            float(self.cfg.sac_bc_log_std_max),
        )
        scale = log_std.exp().expand_as(loc)
        return mean_action, self.sac_dist_cls(loc, scale)

    def _q_action_input(self, action):
        """Normalize absolute executable actions exactly as Stage-2 does."""
        action_clip = float(self.cfg.dagger_action_clip)
        normalized = (action / action_clip).clamp(-1.0, 1.0)
        gain = float(self.cfg.sac_q_action_input_gain)
        return normalized if gain == 1.0 else normalized * gain

    def _normalized_action_log_prob(self, physical_log_prob):
        return physical_log_prob + float(
            self.action_dim
            * math.log(float(self.cfg.sac_entropy_reference_scale))
        )

    def _q_backend_metadata(self):
        """Full Stage-2-facing critic contract saved with every checkpoint."""
        return {
            "actor_obs_keys": list(self.q_actor_keys),
            "critic_obs_keys": list(self.q_critic_keys),
            "actor_obs_dim": self._q_actor_dim,
            "critic_obs_dim": self._q_critic_dim,
            "q_input_dim": self._q_critic_dim,
            "q_actuator_context": {"enabled": False},
            "action_dim": self.action_dim,
            "hidden_dim": int(self.cfg.q_hidden_dim),
            "q_action_fusion": str(self.cfg.q_action_fusion),
            "q_action_hidden_dim": _q_action_hidden_dim(
                self.cfg.q_hidden_dim, self.cfg.q_action_fusion
            ),
            "q_action_fusion_semantics": FASTSAC_Q_EARLY_FUSION_SEMANTICS,
            "q_reference_dueling": False,
            "q_architecture_semantics": FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
            "num_atoms": int(self.cfg.q_num_atoms),
            "v_min": float(self.cfg.q_v_min),
            "v_max": float(self.cfg.q_v_max),
            "layer_norm": bool(self.cfg.q_layer_norm),
            "gamma": float(self.cfg.gamma),
            "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
            "reward_scalarization": SAC_REWARD_SCALARIZATION,
            "reward_groups": list(self.reward_groups),
            "truncation_semantics": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "target_entropy_semantics": (
                FASTSAC_BC_DAGGER_TARGET_ENTROPY_SEMANTICS
            ),
            "q_action_coordinates": str(self.cfg.q_action_coordinates),
            "q_action_normalized": bool(self.cfg.sac_q_normalize_actions),
            "q_action_input_gain": float(self.cfg.sac_q_action_input_gain),
            "q_action_semantics": FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS,
            "clipped_double_q": bool(self.cfg.sac_clipped_double_q),
            "q_backup_semantics": FASTSAC_CLIPPED_DOUBLE_Q_SEMANTICS,
            "actor_q_reduction": "minimum",
            "alpha_autotune": False,
            "pretrain_effective_alpha": 0.0,
            "stage2_alpha_init": float(self.cfg.sac_alpha_init),
            "pretrain_backup_semantics": (
                "stochastic_next_action_q_only_effective_alpha_zero_v1"
            ),
            "pretrain_target_policy": (
                "dedicated_small_noise_stochastic_current_bc_student_v1"
            ),
            "replay_mix_semantics": (
                "beta_independent_teacher_executed_0.5_student_executed_0.5_v1"
            ),
        }

    def _sample_balanced_q_batch(self):
        """Draw an exact beta-independent half-teacher/half-student batch."""
        batch_size = int(self.cfg.q_batch_size)
        teacher_count = batch_size // 2
        student_count = batch_size - teacher_count
        if self.q_teacher_replay.size < 1:
            raise RuntimeError("Cannot sample Q before a teacher transition exists")
        if self.dagger_replay.valid_count(DAGGER_IS_STUDENT_ACTION_KEY) < 1:
            raise RuntimeError("Cannot sample Q before a student transition exists")
        # The all-transition ring also owns BC-only labels.  Project both Q
        # sources to the persistent teacher partition's field set so those
        # extras never cross the replay-device boundary during Q updates.
        q_fields = tuple(self.q_teacher_replay.data)
        teacher = self.q_teacher_replay.sample(
            teacher_count, self.device, self.q_rng, fields=q_fields
        )
        student = self.dagger_replay.sample(
            student_count,
            self.device,
            self.q_rng,
            valid_key=DAGGER_IS_STUDENT_ACTION_KEY,
            fields=q_fields,
        )
        mixed = {
            key: torch.cat((teacher[key], student[key]), dim=0)
            for key in teacher
        }
        mixed[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.cat((
            torch.ones(
                teacher_count, dtype=torch.bool, device=self.device
            ),
            torch.zeros(
                student_count, dtype=torch.bool, device=self.device
            ),
        ))
        permutation = torch.randperm(
            batch_size, device=self.device, generator=self.q_rng
        )
        return {
            key: value[permutation]
            for key, value in mixed.items()
        }

    def _bc_update(self, batch):
        valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY].reshape(-1).bool()
        if not valid.any():
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero
        prediction = self._actor_dist_from_flat(batch["observations"]).mean[valid]
        target = batch[DAGGER_REPLAY_TEACHER_ACTIONS][valid].detach()
        loss = F.smooth_l1_loss(
            prediction,
            target,
            beta=float(self.cfg.dagger_actor_huber_delta),
        )
        self.bc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(
            self.actor_adapt.parameters(), self.cfg.max_grad_norm
        )
        self.bc_optimizer.step()
        self.bc_update_count += 1
        mae = (prediction.detach() - target).abs().mean()
        return loss.detach(), mae, grad.detach()

    def _q_update(self, batch):
        """Apply the Stage-2 Q-only stochastic C51 clipped-double-Q update."""
        critic_observations = batch["critic_observations"]
        with torch.no_grad():
            next_mean, next_dist = self._sac_critic_dist_from_flat(
                batch["next_observations"]
            )
            next_action, next_log_prob = next_dist.rsample_with_log_prob(
                generator=self.sac_action_rng
            )
            next_log_prob = self._normalized_action_log_prob(next_log_prob)
            bootstrap = _sac_bootstrap_mask(
                batch["dones"], batch["truncations"]
            )
            discount = float(self.cfg.gamma) * batch["discounts"]
            # Match Stage-2 before actor release exactly: keep stochastic
            # next-policy sampling, but the Q-only effective alpha is zero.
            entropy_tax = torch.zeros_like(batch["rewards"])
            soft_reward = batch["rewards"]
            unprojected_atoms = (
                soft_reward[:, None]
                + bootstrap[:, None]
                * discount[:, None]
                * self.qnet_target.support
            )
            target = self.qnet_target.projection(
                batch["next_critic_observations"],
                self._q_action_input(next_action),
                soft_reward,
                bootstrap,
                discount,
            )
            # ``projection`` already returns probabilities.  Applying
            # ``values`` here used to softmax those probabilities a second time
            # for diagnostics only.  Take their actual expectation instead.
            raw_target_q_heads = (
                target * self.qnet_target.support
            ).sum(dim=-1)
            target, selected_target_q_heads = _select_c51_twin_target(
                target,
                self.qnet_target.support,
                bool(self.cfg.sac_clipped_double_q),
            )

        logits = self.qnet(
            critic_observations, self._q_action_input(batch["actions"])
        )
        log_probs = F.log_softmax(logits, dim=-1)
        per_q = -(target * log_probs).sum(-1).mean(-1)
        loss = per_q.sum()
        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        grad = _measure_or_clip_grad_norm(
            self.qnet.parameters(), float(self.cfg.q_max_grad_norm)
        )
        self.opt_q.step()
        self.q_update_count += 1
        with torch.no_grad():
            torch._foreach_lerp_(
                tuple(self.qnet_target.parameters()),
                tuple(self.qnet.parameters()),
                float(self.cfg.q_tau),
            )
        with torch.no_grad():
            # Reuse the normalization already computed for the C51 loss instead
            # of launching a second softmax solely for telemetry.
            online_q_heads = (
                log_probs.detach().exp() * self.qnet.support
            ).sum(dim=-1)
            data_q = online_q_heads.min(dim=0).values
            teacher_source = batch[DAGGER_Q_TEACHER_SOURCE_KEY].bool()
            student_source = ~teacher_source
            support_low = self.qnet.support[0]
            support_high = self.qnet.support[-1]
            metrics = {
                "target_q_mean": selected_target_q_heads[0].mean(),
                "target_q_twin_disagreement": (
                    raw_target_q_heads[0] - raw_target_q_heads[1]
                ).abs().mean(),
                "td_target_mean": selected_target_q_heads[0].mean(),
                "td_target_support_low_fraction": (
                    unprojected_atoms < support_low
                ).float().mean(),
                "td_target_support_high_fraction": (
                    unprojected_atoms > support_high
                ).float().mean(),
                "q1_mean": online_q_heads[0].mean(),
                "q2_mean": online_q_heads[1].mean(),
                "q_twin_disagreement": (
                    online_q_heads[0] - online_q_heads[1]
                ).abs().mean(),
                "data_q_mean": data_q.mean(),
                "teacher_data_q_mean": data_q[teacher_source].mean(),
                "student_data_q_mean": data_q[student_source].mean(),
                "next_action_mean_abs_deviation": (
                    next_action - next_mean
                ).abs().mean(),
                "next_log_prob_mean": next_log_prob.mean(),
                "entropy_tax_mean": entropy_tax.mean(),
                "entropy_tax_abs_mean": entropy_tax.abs().mean(),
                "reward_abs_mean": batch["rewards"].abs().mean(),
                "entropy_tax_reward_abs_ratio": (
                    entropy_tax.abs().mean()
                    / batch["rewards"].abs().mean().clamp_min(
                        torch.finfo(batch["rewards"].dtype).eps
                    )
                ),
            }
        return loss.detach(), per_q.detach(), grad.detach(), metrics

    def train_op(self, tensordict):
        rollout = tensordict.exclude("stats")
        training_phase = self._prepare_training_phase()
        finalization_phase = (
            training_phase if self._finalization_enabled() else None
        )
        staging_phase = training_phase if self._staging_enabled() else None
        staging_cycle_index = (
            self._staging_cycle_index() if self._staging_enabled() else -1
        )
        control_mode = self._effective_control_mode()
        safe_zero_iteration = getattr(
            self.cfg, "dagger_safe_zero_iteration", None
        )
        safe_control_enabled = (
            control_mode in ("safe", "hybrid")
            and self._safe_teacher_control_enabled()
        )
        scheduled_beta = self._teacher_mixture_probability()
        rollout_beta = (
            scheduled_beta if control_mode in ("beta", "hybrid") else 0.0
        )

        valid_diagnostic = rollout.get(
            DAGGER_TEACHER_ACTION_VALID_KEY, None
        )
        if valid_diagnostic is None:
            valid_diagnostic = torch.zeros(
                rollout.batch_size,
                dtype=torch.bool,
                device=rollout.device,
            )
        else:
            valid_diagnostic = valid_diagnostic.reshape(-1).bool()
        valid_diagnostic_count = int(valid_diagnostic.sum().item())

        def diagnostic_values(key):
            value = rollout.get(key, None)
            if value is None or valid_diagnostic_count < 1:
                return None
            value = value.reshape(-1)
            return value[valid_diagnostic]

        discrepancy_rms_values = diagnostic_values(
            DAGGER_ACTION_DISCREPANCY_RMS_KEY
        )
        discrepancy_max_values = diagnostic_values(
            DAGGER_ACTION_DISCREPANCY_MAX_KEY
        )

        def diagnostic_mean(key):
            value = diagnostic_values(key)
            return 0.0 if value is None else value.float().mean().item()

        if discrepancy_rms_values is None:
            discrepancy_rms_mean = discrepancy_rms_p95 = 0.0
            discrepancy_rms_max = 0.0
        else:
            discrepancy_rms_values = discrepancy_rms_values.float()
            discrepancy_rms_mean = discrepancy_rms_values.mean().item()
            discrepancy_rms_p95 = torch.quantile(
                discrepancy_rms_values, 0.95
            ).item()
            discrepancy_rms_max = discrepancy_rms_values.max().item()
        if discrepancy_max_values is None:
            discrepancy_joint_max_mean = discrepancy_joint_max_p95 = 0.0
        else:
            discrepancy_max_values = discrepancy_max_values.float()
            discrepancy_joint_max_mean = discrepancy_max_values.mean().item()
            discrepancy_joint_max_p95 = torch.quantile(
                discrepancy_max_values, 0.95
            ).item()

        appended = 0
        teacher_exported = 0
        teacher_selected = 0
        valid_labels = 0
        transition_chunks = (
            tuple(self._dagger_transition_chunks(rollout))
            if self._collect_dagger_replay_this_rollout()
            else ()
        )
        if transition_chunks:
            # Chunks are yielded in step-major, environment-major order.  One
            # concatenation preserves that exact FIFO order while replacing 32
            # replay appends and roughly 128 device synchronizations per
            # rollout with one staged append and one pair of scalar counts.
            transitions = {
                key: torch.cat(
                    [chunk[key] for chunk in transition_chunks], dim=0
                )
                for key in transition_chunks[0]
            }
            replay_device = getattr(
                self.dagger_replay,
                "device",
                next(iter(transitions.values())).device,
            )
            staged = {
                key: (
                    value.detach()
                    if value.device == replay_device
                    else value.detach().to(replay_device)
                )
                for key, value in transitions.items()
            }
            # Do not retain the concatenated GPU staging tensors throughout all
            # BC/Q/adaptation updates.  The CPU replay path now owns its staged
            # copy; the policy-device path retains it through ``staged`` only.
            del transitions, transition_chunks
            appended = self.dagger_replay.extend(staged)
            valid = staged[DAGGER_TEACHER_ACTION_VALID_KEY]
            teacher_executed = (
                valid & ~staged[DAGGER_IS_STUDENT_ACTION_KEY]
            )
            valid_labels = int(valid.sum().item())
            teacher_selected = int(teacher_executed.sum().item())
            if teacher_selected and self._collect_q_teacher_rows_this_rollout():
                # Boolean indexing recomputes the same dynamic selection for
                # every 23kB replay field.  One ascending index vector preserves
                # the old row order and is reused by both teacher FIFOs.
                teacher_indices = teacher_executed.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                q_teacher = {
                    key: staged[key].index_select(0, teacher_indices)
                    for key in TEACHER_REPLAY_FIELDS
                }
                self.q_teacher_replay.extend(q_teacher)
                if (
                    self._export_teacher_rows_this_rollout()
                    and self.teacher_replay is not None
                ):
                    export = {
                        key: (
                            value
                            if value.device == self.teacher_replay.device
                            else value.to(self.teacher_replay.device)
                        )
                        for key, value in q_teacher.items()
                    }
                    teacher_exported = self.teacher_replay.append(export)
                    del export
                del q_teacher, teacher_indices
            del staged, valid, teacher_executed

        q_metrics = []
        bc_metrics = []
        student_q_rows = 0
        if self.dagger_replay.size:
            valid_bc_rows = self.dagger_replay.valid_count(
                DAGGER_TEACHER_ACTION_VALID_KEY
            )
            # DAgger BC remains the only actor objective and retains its normal
            # recent all-transition sampling distribution.
            for _ in range(self._actor_updates_this_rollout()):
                if valid_bc_rows:
                    for _ in range(int(self.cfg.dagger_bc_epochs)):
                        batch = self.dagger_replay.sample(
                            self.cfg.dagger_batch_size,
                            self.device,
                            self.q_rng,
                            valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
                            fields=(
                                "observations",
                                DAGGER_REPLAY_TEACHER_ACTIONS,
                                DAGGER_TEACHER_ACTION_VALID_KEY,
                            ),
                        )
                        batch = self._prepare_dagger_learning_batch(batch)
                        bc_metrics.append(self._bc_update(batch))

            # Critic sampling is deliberately independent of beta. Sampling
            # with replacement starts as soon as both behavior sources exist.
            student_q_rows = self.dagger_replay.valid_count(
                DAGGER_IS_STUDENT_ACTION_KEY
            )
            q_learning_starts = int(self.cfg.q_learning_starts_per_source)
            if (
                self.q_teacher_replay.size >= q_learning_starts
                and student_q_rows >= q_learning_starts
            ):
                for _ in range(self._q_updates_this_rollout()):
                    batch = self._sample_balanced_q_batch()
                    batch = self._prepare_dagger_learning_batch(batch)
                    q_metrics.append(self._q_update(batch))

        # Preserve the original VAIC supervised depth/object/latent updates and
        # their EMA soft copies.  actor_adapt is excluded because inherited
        # residual distillation is disabled in this config.
        adapt_info = self._adaptation_update_this_rollout(rollout)
        self.num_updates += 1
        self.dagger_rollout_count += 1
        self.dagger_environment_steps += int(self.cfg.train_every)
        if self._finalization_enabled():
            self.finalization_rollout_count += 1
        if self._staging_enabled():
            self.staging_rollout_count += 1

        if q_metrics:
            q_loss = torch.stack([item[0] for item in q_metrics]).mean().item()
            q1_loss = torch.stack([item[1][0] for item in q_metrics]).mean().item()
            q2_loss = torch.stack([item[1][1] for item in q_metrics]).mean().item()
            q_grad = torch.stack([item[2] for item in q_metrics]).mean().item()
            critic_metrics = {
                key: torch.stack(
                    [item[3][key] for item in q_metrics]
                ).mean().item()
                for key in q_metrics[0][3]
            }
        else:
            q_loss = q1_loss = q2_loss = q_grad = 0.0
            critic_metrics = {
                key: 0.0
                for key in (
                    "target_q_mean",
                    "target_q_twin_disagreement",
                    "td_target_mean",
                    "td_target_support_low_fraction",
                    "td_target_support_high_fraction",
                    "q1_mean",
                    "q2_mean",
                    "q_twin_disagreement",
                    "data_q_mean",
                    "teacher_data_q_mean",
                    "student_data_q_mean",
                    "next_action_mean_abs_deviation",
                    "next_log_prob_mean",
                    "entropy_tax_mean",
                    "entropy_tax_abs_mean",
                    "reward_abs_mean",
                    "entropy_tax_reward_abs_ratio",
                )
            }
        if bc_metrics:
            bc_loss = torch.stack([item[0] for item in bc_metrics]).mean().item()
            bc_mae = torch.stack([item[1] for item in bc_metrics]).mean().item()
            bc_grad = torch.stack([item[2] for item in bc_metrics]).mean().item()
        else:
            bc_loss = bc_mae = bc_grad = 0.0
        info = {
            # Report the probability that collected this rollout, not the next
            # rollout's value after the stage-local counter increment.
            "dagger/beta": rollout_beta,
            "dagger/beta_schedule": scheduled_beta,
            "dagger/control_mode_safe": float(control_mode == "safe"),
            "dagger/control_mode_hybrid": float(control_mode == "hybrid"),
            "dagger/control_mode_beta": float(control_mode == "beta"),
            "dagger/safe_control_enabled": float(safe_control_enabled),
            "dagger/action_discrepancy_rms_mean": discrepancy_rms_mean,
            "dagger/action_discrepancy_rms_p95": discrepancy_rms_p95,
            "dagger/action_discrepancy_rms_max": discrepancy_rms_max,
            "dagger/action_discrepancy_joint_max_mean": (
                discrepancy_joint_max_mean
            ),
            "dagger/action_discrepancy_joint_max_p95": (
                discrepancy_joint_max_p95
            ),
            "dagger/safe_unsafe_fraction": diagnostic_mean(
                DAGGER_SAFE_UNSAFE_KEY
            ),
            "dagger/safe_teacher_fraction": diagnostic_mean(
                DAGGER_SAFE_TEACHER_KEY
            ),
            "dagger/safe_takeover_fraction": diagnostic_mean(
                DAGGER_SAFE_TAKEOVER_KEY
            ),
            "dagger/safe_release_fraction": diagnostic_mean(
                DAGGER_SAFE_RELEASE_KEY
            ),
            "dagger/beta_teacher_fraction": diagnostic_mean(
                DAGGER_BETA_TEACHER_KEY
            ),
            "dagger/student_invalid_fraction": 1.0 - diagnostic_mean(
                DAGGER_STUDENT_ACTION_VALID_KEY
            ) if valid_diagnostic_count else 0.0,
            "dagger/safe_takeover_rms": float(
                self.cfg.dagger_safe_takeover_rms
            ),
            "dagger/safe_release_rms": float(
                self.cfg.dagger_safe_release_rms
            ),
            "dagger/safe_min_teacher_steps": int(
                self.cfg.dagger_safe_min_teacher_steps
            ),
            "dagger/safe_zero_iteration": (
                int(safe_zero_iteration)
                if safe_zero_iteration is not None
                else -1
            ),
            "dagger/replay_size": self.dagger_replay.size,
            "dagger/replay_seen": self.dagger_replay.seen,
            "dagger/rollout_count": self.dagger_rollout_count,
            "dagger/rollout_index": self.dagger_rollout_count - 1,
            "dagger/environment_steps": self.dagger_environment_steps,
            "dagger/beta_zero_iteration": (
                int(self.cfg.dagger_beta_decay_rollouts)
                if float(self.cfg.dagger_beta_end) == 0.0
                else -1
            ),
            "dagger/teacher_fraction": teacher_selected / max(appended, 1),
            "dagger/valid_teacher_fraction": valid_labels / max(appended, 1),
            "dagger/teacher_exported": teacher_exported,
            "dagger/bc_loss": bc_loss,
            "dagger/bc_mae": bc_mae,
            "dagger/bc_grad_norm": bc_grad,
            "dagger/bc_update_count": self.bc_update_count,
            "dagger/q_loss": q_loss,
            "dagger/q1_loss": q1_loss,
            "dagger/q2_loss": q2_loss,
            "dagger/q_grad_norm": q_grad,
            "dagger/q_update_count": self.q_update_count,
            "dagger/critic_target_q_mean": critic_metrics["target_q_mean"],
            "dagger/critic_target_q_twin_disagreement": critic_metrics[
                "target_q_twin_disagreement"
            ],
            "dagger/critic_td_target_mean": critic_metrics["td_target_mean"],
            "dagger/critic_td_target_support_low_fraction": critic_metrics[
                "td_target_support_low_fraction"
            ],
            "dagger/critic_td_target_support_high_fraction": critic_metrics[
                "td_target_support_high_fraction"
            ],
            "dagger/critic_q1_mean": critic_metrics["q1_mean"],
            "dagger/critic_q2_mean": critic_metrics["q2_mean"],
            "dagger/critic_q_twin_disagreement": critic_metrics[
                "q_twin_disagreement"
            ],
            "dagger/critic_data_q_mean": critic_metrics["data_q_mean"],
            "dagger/critic_teacher_data_q_mean": critic_metrics[
                "teacher_data_q_mean"
            ],
            "dagger/critic_student_data_q_mean": critic_metrics[
                "student_data_q_mean"
            ],
            "dagger/critic_next_action_mean_abs_deviation": critic_metrics[
                "next_action_mean_abs_deviation"
            ],
            "dagger/critic_next_log_prob_mean": critic_metrics[
                "next_log_prob_mean"
            ],
            "dagger/critic_entropy_tax_mean": critic_metrics[
                "entropy_tax_mean"
            ],
            "dagger/critic_entropy_tax_abs_mean": critic_metrics[
                "entropy_tax_abs_mean"
            ],
            "dagger/critic_reward_abs_mean": critic_metrics[
                "reward_abs_mean"
            ],
            "dagger/critic_entropy_tax_reward_abs_ratio": critic_metrics[
                "entropy_tax_reward_abs_ratio"
            ],
            "dagger/critic_effective_alpha": 0.0,
            "dagger/critic_teacher_replay_size": self.q_teacher_replay.size,
            "dagger/critic_teacher_replay_seen": self.q_teacher_replay.seen,
            "dagger/critic_student_replay_rows": student_q_rows,
            "dagger/critic_sampled_teacher_rows": (
                len(q_metrics) * int(self.cfg.q_batch_size) // 2
            ),
            "dagger/critic_sampled_student_rows": (
                len(q_metrics) * int(self.cfg.q_batch_size) // 2
            ),
            "dagger/critic_teacher_ratio": 0.5,
            "dagger/critic_replay_ready": float(
                self.q_teacher_replay.size
                >= int(self.cfg.q_learning_starts_per_source)
                and student_q_rows
                >= int(self.cfg.q_learning_starts_per_source)
            ),
            "dagger/critic_q_action_normalized": 1.0,
            "dagger/critic_clipped_double_q": 1.0,
            "dagger/actor_q_weighting_enabled": 0.0,
            "dagger/truncation_finals": self._last_truncation_finals_used,
            "dagger/replay_observation_raw_pre_vecnorm": 1.0,
            "dagger/fixed_actor_anchor_enabled": 0.0,
        }
        if self._finalization_enabled():
            phase_values = {
                name: float(finalization_phase == name)
                for name in DAGGER_FINALIZATION_PHASES
            }
            info.update(
                {
                    "dagger/finalization_enabled": 1.0,
                    "dagger/finalization_rollout_count": (
                        self.finalization_rollout_count
                    ),
                    "dagger/finalization_rollout_index": (
                        self.finalization_rollout_count - 1
                    ),
                    "dagger/finalization_complete": float(
                        self._finalization_phase() == "complete"
                    ),
                    **{
                        f"dagger/finalization_phase_{name}": value
                        for name, value in phase_values.items()
                    },
                }
            )
        if self._staging_enabled():
            phase_values = {
                name: float(staging_phase == name)
                for name in DAGGER_STAGING_PHASES
            }
            info.update(
                {
                    "dagger/staging_enabled": 1.0,
                    "dagger/staging_rollout_count": (
                        self.staging_rollout_count
                    ),
                    "dagger/staging_rollout_index": (
                        self.staging_rollout_count - 1
                    ),
                    "dagger/staging_cycle_index": staging_cycle_index,
                    "dagger/staging_complete": float(
                        self._staging_phase() == "complete"
                    ),
                    **{
                        f"dagger/staging_phase_{name}": value
                        for name, value in phase_values.items()
                    },
                }
            )
        info.update(adapt_info)
        return info

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
            "dagger_bc_epochs",
            "dagger_actor_huber_delta",
            "dagger_buffer_capacity",
            "dagger_buffer_device",
            "dagger_batch_size",
            "dagger_updates_per_rollout",
            "q_hidden_dim",
            "q_num_atoms",
            "q_v_min",
            "q_v_max",
            "q_layer_norm",
            "q_action_fusion",
            "q_action_coordinates",
            "sac_q_normalize_actions",
            "sac_q_action_input_gain",
            "sac_clipped_double_q",
            "q_lr",
            "q_weight_decay",
            "q_seed",
            "q_tau",
            "q_max_grad_norm",
            "q_batch_size",
            "q_updates_per_rollout",
            "q_teacher_replay_ratio",
            "q_teacher_buffer_capacity",
            "q_learning_starts_per_source",
            "sac_bc_initial_action_std",
            "sac_bc_log_std_min",
            "sac_bc_log_std_max",
            "sac_alpha_init",
            "sac_entropy_reference_scale",
            "teacher_buffer_capacity",
        )
        return {name: getattr(self.cfg, name) for name in names}

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "training_algorithm": PPO_BC_DAGGER_TRAINING_ALGORITHM,
                "actor_backend": PPO_BC_DAGGER_ACTOR_BACKEND,
                "critic_learning_semantics": (
                    PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS
                ),
                "actor_learning_semantics": (
                    PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS
                ),
                "dagger_backend_config": self._checkpoint_config(),
                "q_backend_config": self._q_backend_metadata(),
                "teacher_action_semantics": (
                    DAGGER_TEACHER_ACTION_SEMANTICS
                ),
                "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
                "dagger_rollout_count": self.dagger_rollout_count,
                "dagger_environment_steps": self.dagger_environment_steps,
                "bc_update_count": self.bc_update_count,
                "q_update_count": self.q_update_count,
                "dagger_rng_state": self.dagger_rng.get_state(),
                "q_rng_state": self.q_rng.get_state(),
                "sac_action_rng_state": self.sac_action_rng.get_state(),
                "optimizer_resume_state": {
                    "bc_optimizer": self.bc_optimizer.state_dict(),
                    "q_optimizer": self.opt_q.state_dict(),
                    "adapt_optimizer": self.opt_adapt.state_dict(),
                },
                "teacher_replay_id": self.teacher_replay_id,
                "replay_resume_semantics": (
                    "staged_ephemeral_rings_omitted_h5_only_after_final_q_v1"
                    if self._staging_enabled()
                    else (
                        "recent_student_ring_omitted_teacher_partition_"
                        "h5_refill_v2"
                    )
                ),
                "replay_observation_semantics": (
                    DAGGER_REPLAY_OBSERVATION_SEMANTICS
                ),
                "vecnorm_fingerprint": self._replay_vecnorm_fingerprint,
                "next_iter": int(self.env.current_iter) + 1,
            }
        )
        if self.teacher_replay is not None:
            state["teacher_replay_state"] = self.teacher_replay.checkpoint_metadata()
        elif self._loaded_teacher_replay_metadata is not None:
            # This is provenance only: model-only resume deliberately has no
            # live/paired H5 and must not claim an exact replay snapshot.
            state["frozen_teacher_replay_source_state"] = copy.deepcopy(
                self._loaded_teacher_replay_metadata
            )
        if self._finalization_enabled():
            state["bc_dagger_finalization_state"] = {
                "semantics": DAGGER_FINALIZATION_SEMANTICS,
                "config": self._finalization_config(),
                "rollout_count": int(self.finalization_rollout_count),
                "phase": self._finalization_phase(),
                "last_phase": self._finalization_last_phase,
                "complete": self._finalization_phase() == "complete",
                "source_state": copy.deepcopy(self._finalization_source_state),
                "fresh_replay_id": str(self.teacher_replay_id),
            }
        if self._staging_enabled():
            calibration_start_q_updates = getattr(
                self,
                "_staging_calibration_start_q_update_count",
                None,
            )
            state["bc_dagger_staging_state"] = {
                "semantics": DAGGER_STAGING_SEMANTICS,
                "config": self._staging_config(),
                "rollout_count": int(self.staging_rollout_count),
                "phase": self._staging_phase(),
                "last_phase": self._staging_last_phase,
                "complete": self._staging_phase() == "complete",
                "fresh_replay_id": str(self.teacher_replay_id),
                "persistent_replay_semantics": (
                    "h5_disabled_until_final_q_calibration_v1"
                ),
                "calibration_start_q_update_count": (
                    None
                    if calibration_start_q_updates is None
                    else int(calibration_start_q_updates)
                ),
                "calibration_q_updates": (
                    0
                    if calibration_start_q_updates is None
                    else int(self.q_update_count)
                    - int(calibration_start_q_updates)
                ),
            }
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        same_stage = algorithm == PPO_BC_DAGGER_TRAINING_ALGORITHM
        if same_stage:
            checkpoint_staging_state = state_dict.get(
                "bc_dagger_staging_state"
            )
            if self._staging_enabled() != (
                checkpoint_staging_state is not None
            ):
                raise ValueError(
                    "PPO-BC DAgger staged/non-staged resume mode mismatch"
                )
            if "qnet" not in state_dict:
                raise KeyError("same-stage checkpoint is missing qnet")
            if "qnet_target" not in state_dict:
                raise KeyError("same-stage checkpoint is missing qnet_target")
            if state_dict.get("critic_learning_semantics") != (
                PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS
            ):
                raise ValueError("PPO-BC DAgger SAC critic semantics mismatch")
            if state_dict.get("actor_learning_semantics") != (
                PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS
            ):
                raise ValueError("PPO-BC DAgger actor learning semantics mismatch")
            if state_dict.get("actor_backend") != PPO_BC_DAGGER_ACTOR_BACKEND:
                raise ValueError("PPO-BC DAgger actor backend mismatch")
            if state_dict.get("teacher_action_semantics") != (
                DAGGER_TEACHER_ACTION_SEMANTICS
            ):
                raise ValueError("PPO-BC DAgger teacher action semantics mismatch")
            if state_dict.get("dagger_control_semantics") != (
                DAGGER_CONTROL_SEMANTICS
            ):
                raise ValueError("PPO-BC DAgger control semantics mismatch")
            checkpoint_fingerprint = state_dict.get("vecnorm_fingerprint")
            if (
                checkpoint_fingerprint is not None
                and self._replay_vecnorm_fingerprint is not None
                and str(checkpoint_fingerprint)
                != self._replay_vecnorm_fingerprint
            ):
                raise ValueError(
                    "PPO-BC DAgger checkpoint VecNorm fingerprint mismatch"
                )
            actual_config = state_dict.get("dagger_backend_config")
            expected_config = self._checkpoint_config() if hasattr(self, "device") else None
            if actual_config is not None and expected_config is not None:
                # Checkpoints predating the optional cutoff are equivalent to
                # its disabled default, but a non-null resumed cutoff must match.
                actual_config = dict(actual_config)
                actual_config.setdefault("dagger_safe_zero_iteration", None)
                if actual_config != expected_config:
                    raise ValueError("PPO-BC DAgger checkpoint config mismatch")
            actual_q_backend = state_dict.get("q_backend_config")
            if actual_q_backend != self._q_backend_metadata():
                raise ValueError("PPO-BC DAgger Q backend contract mismatch")
        elif algorithm == PPO_BC_DAGGER_IQL_TRAINING_ALGORITHM:
            raise ValueError(
                "IQL-v2 BC-DAgger checkpoints cannot resume the SAC-critic "
                "stage; start a new scripts/bc_dagger.py run from the PPO teacher."
            )
        elif algorithm == PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM:
            raise ValueError(
                "Legacy BC-DAgger checkpoints do not contain the compatible "
                "SAC critic; start a new scripts/bc_dagger.py run."
            )
        elif algorithm is not None:
            raise ValueError(
                f"Unsupported checkpoint training_algorithm={algorithm!r}"
            )
        elif state_dict.get("last_phase") != "train":
            raise ValueError(
                "Initial PPO-BC DAgger transfer requires a PPO teacher checkpoint"
            )

        failed = PPOVEL.load_state_dict(self, state_dict, strict)
        if same_stage:
            critical = {
                "actor",
                "actor_adapt",
                "encoder_priv",
                "adapt_module",
                "adapt_ema",
                "qnet",
                "qnet_target",
                "bc_dagger_sac_adapter",
            }
            if getattr(self.cfg, "use_object_adapt", False):
                critical.update(("object_adapt", "object_adapt_ema"))
            if hasattr(self, "temporal_depth_gru"):
                critical.update(
                    ("depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema")
                )
            missing = critical.intersection(failed)
            if missing:
                raise RuntimeError(
                    f"Failed to restore DAgger modules: {sorted(missing)}"
                )
            optimizers = state_dict.get("optimizer_resume_state")
            if not isinstance(optimizers, dict):
                raise ValueError("same-stage checkpoint lacks optimizer state")
            self.bc_optimizer.load_state_dict(optimizers["bc_optimizer"])
            self.opt_q.load_state_dict(optimizers["q_optimizer"])
            self.opt_adapt.load_state_dict(optimizers["adapt_optimizer"])
            self.dagger_rollout_count = int(
                state_dict.get("dagger_rollout_count", 0)
            )
            self.dagger_environment_steps = int(
                state_dict.get("dagger_environment_steps", 0)
            )
            self.bc_update_count = int(state_dict.get("bc_update_count", 0))
            self.q_update_count = int(state_dict.get("q_update_count", 0))
            self.dagger_rng.set_state(state_dict["dagger_rng_state"])
            self.q_rng.set_state(state_dict["q_rng_state"])
            self.sac_action_rng.set_state(state_dict["sac_action_rng_state"])
            self.teacher_replay_id = str(state_dict.get("teacher_replay_id"))
            self._loaded_teacher_replay_metadata = copy.deepcopy(
                state_dict.get(
                    "teacher_replay_state",
                    state_dict.get("frozen_teacher_replay_source_state"),
                )
            )
            if hasattr(self, "env"):
                self.env.set_progress(
                    int(
                        state_dict.get(
                            "next_iter", state_dict.get("last_iter", -1) + 1
                        )
                    )
                )
            is_fresh_finalization = (
                self._finalization_enabled()
                and state_dict.get("bc_dagger_finalization_state") is None
            )
            if is_fresh_finalization:
                logging.info(
                    "Starting BC-DAgger finalization from model/optimizer "
                    "state only; the source H5 and both ephemeral learning "
                    "rings are deliberately discarded."
                )
            elif self._staging_enabled():
                warnings.warn(
                    "Staged BC-DAgger checkpoint resume omits both ephemeral "
                    "learning rings. They will refill from fresh rollouts; "
                    "pre-calibration checkpoints intentionally have no H5."
                )
            else:
                warnings.warn(
                    "The recent student DAgger ring is intentionally not in "
                    "the checkpoint. The persistent teacher critic partition "
                    "must be refilled from the immutable H5 before Q updates "
                    "resume."
                )
            if self._finalization_enabled():
                finalization_state = state_dict.get(
                    "bc_dagger_finalization_state"
                )
                if finalization_state is None:
                    # Forking from a completed joint BC-DAgger run deliberately
                    # ignores its representation-stale H5. Preserve weights,
                    # optimizers, and RNG streams, but establish a new replay
                    # lineage and a separate local phase counter.
                    self._finalization_source_state = {
                        "training_algorithm": algorithm,
                        "dagger_rollout_count": self.dagger_rollout_count,
                        "dagger_environment_steps": (
                            self.dagger_environment_steps
                        ),
                        "bc_update_count": self.bc_update_count,
                        "q_update_count": self.q_update_count,
                        "teacher_replay_id": self.teacher_replay_id,
                        "teacher_replay_state": copy.deepcopy(
                            self._loaded_teacher_replay_metadata
                        ),
                    }
                    self.finalization_rollout_count = 0
                    self._finalization_last_phase = None
                    self.teacher_replay_id = str(uuid.uuid4())
                    self._loaded_teacher_replay_metadata = None
                    self.dagger_replay.clear()
                    self.q_teacher_replay.clear()
                else:
                    if finalization_state.get("semantics") != (
                        DAGGER_FINALIZATION_SEMANTICS
                    ):
                        raise ValueError(
                            "BC-DAgger finalization semantics mismatch"
                        )
                    if finalization_state.get("config") != (
                        self._finalization_config()
                    ):
                        raise ValueError(
                            "BC-DAgger finalization schedule mismatch"
                        )
                    self.finalization_rollout_count = int(
                        finalization_state.get("rollout_count", 0)
                    )
                    self._finalization_last_phase = finalization_state.get(
                        "last_phase"
                    )
                    self._finalization_source_state = copy.deepcopy(
                        finalization_state.get("source_state")
                    )
                    expected_replay_id = str(
                        finalization_state.get("fresh_replay_id", "")
                    )
                    if expected_replay_id and (
                        expected_replay_id != self.teacher_replay_id
                    ):
                        raise ValueError(
                            "BC-DAgger finalization replay lineage mismatch"
                        )
            if self._staging_enabled():
                staging_state = state_dict.get("bc_dagger_staging_state")
                if not isinstance(staging_state, dict):
                    raise ValueError(
                        "Staged BC-DAgger checkpoint lacks staging state"
                    )
                if staging_state.get("semantics") != DAGGER_STAGING_SEMANTICS:
                    raise ValueError("BC-DAgger staging semantics mismatch")
                if staging_state.get("config") != self._staging_config():
                    raise ValueError("BC-DAgger staging schedule mismatch")
                self.staging_rollout_count = int(
                    staging_state.get("rollout_count", 0)
                )
                self._staging_last_phase = staging_state.get("last_phase")
                calibration_start_q_updates = staging_state.get(
                    "calibration_start_q_update_count"
                )
                self._staging_calibration_start_q_update_count = (
                    None
                    if calibration_start_q_updates is None
                    else int(calibration_start_q_updates)
                )
                expected_replay_id = str(
                    staging_state.get("fresh_replay_id", "")
                )
                if expected_replay_id and (
                    expected_replay_id != self.teacher_replay_id
                ):
                    raise ValueError(
                        "BC-DAgger staging replay lineage mismatch"
                    )
        else:
            # The PPO checkpoint has neither depth-camera modules nor Qs.  Those
            # are the only expected fresh modules; teacher/student core failures
            # still abort rather than silently training from the wrong policy.
            allowed_fresh = {
                "depth_cnn",
                "temporal_depth_gru",
                "temporal_depth_gru_ema",
                "qnet",
                "qnet_target",
                "bc_dagger_sac_adapter",
            }
            unexpected = set(failed).difference(allowed_fresh)
            if unexpected:
                raise RuntimeError(
                    "Failed to load critical PPO teacher/student modules: "
                    f"{sorted(unexpected)}"
                )
            hard_copy_(self.qnet, self.qnet_target)
            self.qnet_target.requires_grad_(False)
            self.q_update_count = 0
            self.bc_update_count = 0
            self.dagger_rollout_count = 0
            self.dagger_environment_steps = 0
            self.q_rng.manual_seed(int(self.cfg.q_seed))
            self.sac_action_rng.manual_seed(int(self.cfg.q_seed) + 1)
            self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
        if hasattr(self, "_freeze_teacher"):
            self._freeze_teacher()
        if self._staging_enabled():
            self._apply_staging_freeze_mask(self._staging_phase())
        return failed
