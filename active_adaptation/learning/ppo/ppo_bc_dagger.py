"""PPO-teacher DAgger finetuning with VAIC perception and an IQL critic.

This stage deliberately does *not* run PPO or SAC actor optimization.  A
frozen PPO residual policy is the privileged DAgger oracle, ``actor_adapt`` is
trained only by behavior cloning, and an IQL-style V plus C51 Q1/Q2 are learned
as future FastSAC warm-start weights from actions actually sent to the
environment. Q-derived advantage weighting is deliberately absent.
The original VAIC observation, reward, termination, depth-supervision, and EMA
paths remain owned by :class:`PPOVEL`.
"""

from __future__ import annotations

import copy
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
    FASTSAC_RAW_OBSERVATION_ROOT,
    REPLAY_OBSERVATION_SEMANTICS,
    SAC_REWARD_SCALARIZATION,
    TEACHER_REPLAY_FIELDS,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    TeacherReplayBuffer,
    _build_isolated_q_network,
    _filter_replay_rows,
    _sac_bootstrap_mask,
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


PPO_BC_DAGGER_TRAINING_ALGORITHM = "vaic_ppo_bc_dagger_student_iql_v2"
PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM = (
    "vaic_ppo_bc_dagger_student_v1"
)
PPO_BC_DAGGER_ACTOR_BACKEND = "vaic_ppo_independent_normal_bc_dagger_v1"
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
    "absolute_executable_bernoulli_teacher_or_student_v1"
)
DAGGER_TEACHER_ACTION_SEMANTICS = "ppo_reference_plus_residual_v1"
DAGGER_TEACHER_ACTION_KEY = "teacher_action"
DAGGER_TEACHER_ACTION_VALID_KEY = "teacher_action_valid"
DAGGER_IS_STUDENT_ACTION_KEY = "is_student_action"
DAGGER_REPLAY_TEACHER_ACTIONS = "teacher_actions"
DAGGER_REPLAY_MIN_STEP_COUNT = 5


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

    # beta is a per-environment Bernoulli teacher-execution probability.  It is
    # local to this new stage and never derives from the inherited PPO iteration.
    dagger_beta_start: float = 1.0
    dagger_beta_end: float = 0.0
    dagger_beta_decay_rollouts: int = 4000
    dagger_seed: int = 0
    dagger_teacher_action_threshold: float = 20.0
    dagger_action_clip: float = 20.0

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
    q_num_atoms: int = 101
    q_v_min: float = -20.0
    q_v_max: float = 20.0
    q_layer_norm: bool = True
    q_lr: float = 3e-4
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    # Official IQL uses a slowly moving target critic. This remains separate
    # from Stage-2 FastSAC's sac_tau after the Q weights are transferred.
    q_tau: float = 0.005
    q_max_grad_norm: float = 1.0
    iql_expectile: float = 0.7
    iql_value_lr: float = 3e-4
    # BC keeps its large 4,096-row batch.  At 32 updates per rollout, 1,024
    # IQL rows still gives UTD=2 with 512 envs x 32 control steps, while the
    # additional V forward/backward does not dominate DAgger wall time.
    iql_batch_size: int = 1024

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

    def valid_count(self, key: str) -> int:
        if self.size < 1:
            return 0
        if key not in self.data:
            raise KeyError(f"Unknown DAgger replay validity field {key!r}")
        value = self.data[key][: self.size]
        if value.dtype is not torch.bool or value.ndim != 1:
            raise TypeError(
                f"DAgger replay validity field {key!r} must be one-dimensional bool"
            )
        return int(value.sum().item())

    def sample(
        self,
        count: int,
        output_device,
        generator=None,
        valid_key: str | None = None,
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
            if valid_key not in self.data:
                raise KeyError(
                    f"Unknown DAgger replay validity field {valid_key!r}"
                )
            valid = self.data[valid_key][: self.size]
            if valid.dtype is not torch.bool or valid.ndim != 1:
                raise TypeError(
                    "DAgger replay validity sampling requires a one-dimensional "
                    "boolean field"
                )
            valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
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
        output_device = torch.device(output_device)
        return {
            key: value[indices].to(output_device)
            for key, value in self.data.items()
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
        beta = owner._teacher_mixture_probability()
        choose_teacher = (
            torch.rand(
                valid.shape,
                device=valid.device,
                generator=owner.dagger_rng,
            )
            < beta
        ) & valid
        # Invalid teacher rows are retained only as masked labels; never expose
        # NaN/Inf through a rollout TensorDict or to the environment action path.
        action_clip = float(owner.cfg.dagger_action_clip)
        safe_teacher = torch.nan_to_num(
            teacher_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)
        safe_student = torch.nan_to_num(
            raw_student_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)
        td[ACTION_KEY] = torch.where(
            choose_teacher.unsqueeze(-1), safe_teacher, safe_student
        )
        td[DAGGER_TEACHER_ACTION_KEY] = safe_teacher
        td[DAGGER_TEACHER_ACTION_VALID_KEY] = valid
        td[DAGGER_IS_STUDENT_ACTION_KEY] = ~choose_teacher
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
    """Depth student with pure DAgger BC and an IQL-pretrained C51 critic."""

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
        )
        self.qnet_target = copy.deepcopy(self.qnet).requires_grad_(False)
        self.iql_value = _build_isolated_iql_value_network(
            self._q_critic_dim,
            cfg.q_hidden_dim,
            cfg.q_layer_norm,
            device,
            int(cfg.q_seed) + 1,
        )
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=cfg.q_lr,
            weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95),
            fused=str(device).startswith("cuda"),
        )
        self.opt_iql_value = torch.optim.Adam(
            self.iql_value.parameters(),
            lr=cfg.iql_value_lr,
            betas=(0.9, 0.999),
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
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.bc_update_count = 0
        self.q_update_count = 0
        self.iql_value_update_count = 0

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
        for name in (
            "dagger_bc_epochs",
            "dagger_buffer_capacity",
            "dagger_batch_size",
            "dagger_updates_per_rollout",
            "q_hidden_dim",
            "q_num_atoms",
            "iql_batch_size",
            "teacher_buffer_capacity",
            "teacher_buffer_snapshot_chunk_rows",
        ):
            if isinstance(getattr(cfg, name), bool) or int(getattr(cfg, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "dagger_bc_lr",
            "dagger_actor_huber_delta",
            "q_lr",
            "iql_value_lr",
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
        if int(cfg.q_hidden_dim) < 4:
            raise ValueError("q_hidden_dim must be at least 4")
        if cfg.q_num_atoms < 2 or not cfg.q_v_min < cfg.q_v_max:
            raise ValueError("distributional Q support is invalid")
        if not 0.0 <= float(cfg.q_tau) <= 1.0:
            raise ValueError("q_tau must be in [0, 1]")
        if not 0.0 < float(cfg.iql_expectile) < 1.0:
            raise ValueError("iql_expectile must be strictly in (0, 1)")
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

    def _teacher_mixture_probability(self):
        return _linear_teacher_probability(
            self.cfg.dagger_beta_start,
            self.cfg.dagger_beta_end,
            self.cfg.dagger_beta_decay_rollouts,
            self.dagger_rollout_count,
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
        return self.teacher_replay.snapshot(iteration, checkpoint_name)

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
        """Normalize one raw DAgger minibatch without mutating replay data."""
        snapshot = self._vecnorm_snapshot()
        prepared = dict(batch)
        for field in ("observations", "next_observations"):
            prepared[field] = self._normalize_replay_flat(
                batch[field],
                self.q_actor_keys,
                self._q_actor_widths,
                snapshot,
            )
        for field in ("critic_observations", "next_critic_observations"):
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
        truncations = _vaic_truncation_mask(td).reshape(-1).bool()
        if not truncations.any():
            return
        indices = truncations.nonzero(as_tuple=False).squeeze(-1)
        values = self._prepare_student_final_state(td["next"][indices].clone())
        values["indices"] = indices * int(self.cfg.train_every) + int(step)
        self._truncation_final_batches.append(values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
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
        """Update V then twin C51 Qs with dataset-action IQL targets.

        The DAgger actor is intentionally absent from this method.  This follows
        IQL's in-sample critic update while keeping the existing distributional
        Q topology required for an exact Stage-2 FastSAC weight transfer.
        """
        critic_observations = batch["critic_observations"]
        with torch.no_grad():
            target_logits = self.qnet_target(
                critic_observations, batch["actions"]
            )
            target_q_heads = self.qnet_target.values(target_logits)
            target_q = target_q_heads.min(dim=0).values

        value = self.iql_value(critic_observations)
        advantage = target_q - value
        value_loss = _iql_expectile_loss(
            advantage, self.cfg.iql_expectile, validate=False
        ).mean()
        self.opt_iql_value.zero_grad(set_to_none=True)
        value_loss.backward()
        if float(self.cfg.q_max_grad_norm) > 0.0:
            value_grad = nn.utils.clip_grad_norm_(
                self.iql_value.parameters(),
                float(self.cfg.q_max_grad_norm),
            )
        else:
            value_grad = nn.utils.clip_grad_norm_(
                self.iql_value.parameters(), float("inf")
            )
        self.opt_iql_value.step()
        self.iql_value_update_count += 1

        with torch.no_grad():
            next_value = self.iql_value(batch["next_critic_observations"])
            bootstrap = _sac_bootstrap_mask(
                batch["dones"], batch["truncations"]
            )
            discount = float(self.cfg.gamma) * batch["discounts"]
            scalar_target = (
                batch["rewards"]
                + bootstrap * discount * next_value
            )
            target = _project_scalar_to_c51(
                scalar_target, self.qnet.support, validate=False
            )
        logits = self.qnet(critic_observations, batch["actions"])
        per_q = -(
            target.unsqueeze(0) * F.log_softmax(logits, dim=-1)
        ).sum(-1).mean(-1)
        loss = per_q.sum()
        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.cfg.q_max_grad_norm) > 0.0:
            grad = nn.utils.clip_grad_norm_(
                self.qnet.parameters(), float(self.cfg.q_max_grad_norm)
            )
        else:
            grad = nn.utils.clip_grad_norm_(self.qnet.parameters(), float("inf"))
        self.opt_q.step()
        self.q_update_count += 1
        with torch.no_grad():
            for online, target_parameter in zip(
                self.qnet.parameters(), self.qnet_target.parameters()
            ):
                target_parameter.lerp_(online, float(self.cfg.q_tau))
        with torch.no_grad():
            online_q_heads = self.qnet.values(logits)
            support_low = self.qnet.support[0]
            support_high = self.qnet.support[-1]
            metrics = {
                "value_loss": value_loss.detach(),
                "value_grad_norm": value_grad.detach(),
                "value_mean": value.detach().mean(),
                "value_std": value.detach().std(unbiased=False),
                "target_q_mean": target_q.mean(),
                "target_q_twin_disagreement": (
                    target_q_heads[0] - target_q_heads[1]
                ).abs().mean(),
                "advantage_mean": advantage.detach().mean(),
                "advantage_std": advantage.detach().std(unbiased=False),
                "advantage_positive_fraction": (
                    advantage.detach() > 0.0
                ).float().mean(),
                "td_target_mean": scalar_target.mean(),
                "td_target_std": scalar_target.std(unbiased=False),
                "td_target_support_low_fraction": (
                    scalar_target < support_low
                ).float().mean(),
                "td_target_support_high_fraction": (
                    scalar_target > support_high
                ).float().mean(),
                "q1_mean": online_q_heads[0].mean(),
                "q2_mean": online_q_heads[1].mean(),
                "q_twin_disagreement": (
                    online_q_heads[0] - online_q_heads[1]
                ).abs().mean(),
            }
        return loss.detach(), per_q.detach(), grad.detach(), metrics

    def train_op(self, tensordict):
        rollout = tensordict.exclude("stats")
        rollout_beta = self._teacher_mixture_probability()
        appended = 0
        teacher_exported = 0
        teacher_selected = 0
        valid_labels = 0
        for transitions in self._dagger_transition_chunks(rollout):
            appended += self.dagger_replay.extend(transitions)
            valid = transitions[DAGGER_TEACHER_ACTION_VALID_KEY]
            teacher_executed = valid & ~transitions[DAGGER_IS_STUDENT_ACTION_KEY]
            valid_labels += int(valid.sum().item())
            teacher_selected += int(teacher_executed.sum().item())
            if self.teacher_replay is not None and teacher_executed.any():
                export = {
                    key: transitions[key][teacher_executed].to(
                        self.teacher_replay.device
                    )
                    for key in TEACHER_REPLAY_FIELDS
                }
                teacher_exported += self.teacher_replay.append(export)

        q_metrics = []
        bc_metrics = []
        if self.dagger_replay.size:
            valid_bc_rows = self.dagger_replay.valid_count(
                DAGGER_TEACHER_ACTION_VALID_KEY
            )
            # DAgger BC remains the only actor objective. IQL then updates V/Q
            # from a separate all-transition sample and never queries that actor,
            # so the critic cannot feed back into the student action update.
            for _ in range(int(self.cfg.dagger_updates_per_rollout)):
                if valid_bc_rows:
                    for _ in range(int(self.cfg.dagger_bc_epochs)):
                        batch = self.dagger_replay.sample(
                            self.cfg.dagger_batch_size,
                            self.device,
                            self.q_rng,
                            valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
                        )
                        batch = self._prepare_dagger_learning_batch(batch)
                        bc_metrics.append(self._bc_update(batch))
                batch = self.dagger_replay.sample(
                    self.cfg.iql_batch_size, self.device, self.q_rng
                )
                batch = self._prepare_dagger_learning_batch(batch)
                q_metrics.append(self._q_update(batch))

        # Preserve the original VAIC supervised depth/object/latent updates and
        # their EMA soft copies.  actor_adapt is excluded because inherited
        # residual distillation is disabled in this config.
        adapt_info = self.train_adapt(rollout.copy())
        self.num_updates += 1
        self.dagger_rollout_count += 1
        self.dagger_environment_steps += int(self.cfg.train_every)

        if q_metrics:
            q_loss = torch.stack([item[0] for item in q_metrics]).mean().item()
            q1_loss = torch.stack([item[1][0] for item in q_metrics]).mean().item()
            q2_loss = torch.stack([item[1][1] for item in q_metrics]).mean().item()
            q_grad = torch.stack([item[2] for item in q_metrics]).mean().item()
            iql_metrics = {
                key: torch.stack(
                    [item[3][key] for item in q_metrics]
                ).mean().item()
                for key in q_metrics[0][3]
            }
        else:
            q_loss = q1_loss = q2_loss = q_grad = 0.0
            iql_metrics = {
                key: 0.0
                for key in (
                    "value_loss",
                    "value_grad_norm",
                    "value_mean",
                    "value_std",
                    "target_q_mean",
                    "target_q_twin_disagreement",
                    "advantage_mean",
                    "advantage_std",
                    "advantage_positive_fraction",
                    "td_target_mean",
                    "td_target_std",
                    "td_target_support_low_fraction",
                    "td_target_support_high_fraction",
                    "q1_mean",
                    "q2_mean",
                    "q_twin_disagreement",
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
            "dagger/replay_size": self.dagger_replay.size,
            "dagger/replay_seen": self.dagger_replay.seen,
            "dagger/rollout_count": self.dagger_rollout_count,
            "dagger/environment_steps": self.dagger_environment_steps,
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
            "dagger/iql_value_loss": iql_metrics["value_loss"],
            "dagger/iql_value_grad_norm": iql_metrics[
                "value_grad_norm"
            ],
            "dagger/iql_value_mean": iql_metrics["value_mean"],
            "dagger/iql_value_std": iql_metrics["value_std"],
            "dagger/iql_target_q_mean": iql_metrics["target_q_mean"],
            "dagger/iql_target_q_twin_disagreement": iql_metrics[
                "target_q_twin_disagreement"
            ],
            "dagger/iql_advantage_mean": iql_metrics[
                "advantage_mean"
            ],
            "dagger/iql_advantage_std": iql_metrics["advantage_std"],
            "dagger/iql_advantage_positive_fraction": iql_metrics[
                "advantage_positive_fraction"
            ],
            "dagger/iql_td_target_mean": iql_metrics["td_target_mean"],
            "dagger/iql_td_target_std": iql_metrics["td_target_std"],
            "dagger/iql_td_target_support_low_fraction": iql_metrics[
                "td_target_support_low_fraction"
            ],
            "dagger/iql_td_target_support_high_fraction": iql_metrics[
                "td_target_support_high_fraction"
            ],
            "dagger/iql_q1_mean": iql_metrics["q1_mean"],
            "dagger/iql_q2_mean": iql_metrics["q2_mean"],
            "dagger/iql_q_twin_disagreement": iql_metrics[
                "q_twin_disagreement"
            ],
            "dagger/iql_expectile": float(self.cfg.iql_expectile),
            "dagger/iql_value_update_count": self.iql_value_update_count,
            "dagger/actor_q_weighting_enabled": 0.0,
            "dagger/truncation_finals": self._last_truncation_finals_used,
            "dagger/replay_observation_raw_pre_vecnorm": 1.0,
            "dagger/fixed_actor_anchor_enabled": 0.0,
        }
        info.update(adapt_info)
        return info

    def _checkpoint_config(self):
        names = (
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
            "q_lr",
            "q_weight_decay",
            "q_seed",
            "q_tau",
            "q_max_grad_norm",
            "iql_expectile",
            "iql_value_lr",
            "iql_batch_size",
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
                    PPO_BC_DAGGER_IQL_CRITIC_SEMANTICS
                ),
                "actor_learning_semantics": (
                    PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS
                ),
                "dagger_backend_config": self._checkpoint_config(),
                "teacher_action_semantics": (
                    DAGGER_TEACHER_ACTION_SEMANTICS
                ),
                "dagger_rollout_count": self.dagger_rollout_count,
                "dagger_environment_steps": self.dagger_environment_steps,
                "bc_update_count": self.bc_update_count,
                "q_update_count": self.q_update_count,
                "iql_value_update_count": self.iql_value_update_count,
                "dagger_rng_state": self.dagger_rng.get_state(),
                "q_rng_state": self.q_rng.get_state(),
                "optimizer_resume_state": {
                    "bc_optimizer": self.bc_optimizer.state_dict(),
                    "q_optimizer": self.opt_q.state_dict(),
                    "iql_value_optimizer": (
                        self.opt_iql_value.state_dict()
                    ),
                    "adapt_optimizer": self.opt_adapt.state_dict(),
                },
                "teacher_replay_id": self.teacher_replay_id,
                "replay_resume_semantics": (
                    "all_transition_ring_omitted_non_exact_resume_v1"
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
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        same_stage = algorithm == PPO_BC_DAGGER_TRAINING_ALGORITHM
        if same_stage:
            if "qnet" not in state_dict:
                raise KeyError("same-stage checkpoint is missing qnet")
            if "qnet_target" not in state_dict:
                raise KeyError("same-stage checkpoint is missing qnet_target")
            if "iql_value" not in state_dict:
                raise KeyError("same-stage checkpoint is missing iql_value")
            if state_dict.get("critic_learning_semantics") != (
                PPO_BC_DAGGER_IQL_CRITIC_SEMANTICS
            ):
                raise ValueError("PPO-BC DAgger IQL critic semantics mismatch")
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
                if actual_config != expected_config:
                    raise ValueError("PPO-BC DAgger checkpoint config mismatch")
        elif algorithm == PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM:
            raise ValueError(
                "Legacy Bellman BC-DAgger checkpoints do not contain the IQL "
                "value network; start a new scripts/bc_dagger.py run."
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
                "iql_value",
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
            if "iql_value_optimizer" not in optimizers:
                raise ValueError(
                    "same-stage checkpoint lacks IQL value optimizer state"
                )
            self.opt_iql_value.load_state_dict(
                optimizers["iql_value_optimizer"]
            )
            self.opt_adapt.load_state_dict(optimizers["adapt_optimizer"])
            self.dagger_rollout_count = int(
                state_dict.get("dagger_rollout_count", 0)
            )
            self.dagger_environment_steps = int(
                state_dict.get("dagger_environment_steps", 0)
            )
            self.bc_update_count = int(state_dict.get("bc_update_count", 0))
            self.q_update_count = int(state_dict.get("q_update_count", 0))
            self.iql_value_update_count = int(
                state_dict.get("iql_value_update_count", 0)
            )
            self.dagger_rng.set_state(state_dict["dagger_rng_state"])
            self.q_rng.set_state(state_dict["q_rng_state"])
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
            warnings.warn(
                "The live all-transition DAgger ring is intentionally not in "
                "the checkpoint; same-stage replay resume is non-exact."
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
                "iql_value",
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
            self.iql_value_update_count = 0
            self.bc_update_count = 0
            self.dagger_rollout_count = 0
            self.dagger_environment_steps = 0
            self.q_rng.manual_seed(int(self.cfg.q_seed))
            self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
        if hasattr(self, "_freeze_teacher"):
            self._freeze_teacher()
        return failed
