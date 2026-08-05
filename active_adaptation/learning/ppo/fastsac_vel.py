"""VAIC teacher/student training with distributional FastSAC.

Both stages use FastSAC actor/Q optimization. VAIC continues to own the
observations, rewards, terminations, depth perception, and student adaptation.
The PPOVEL base is used only to construct those unchanged VAIC modules; this
module contains no PPO optimization path.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch.distributions.kl import register_kl
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal

from .common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    hard_copy_,
)
from .ppo_vel import (
    DEPTH_KEY,
    HEIGHT_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    PPOConfig,
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    VEL_CMD_KEY,
)


TEACHER_REPLAY_FORMAT_VERSION = 4
TIMEOUT_NEXT_OBSERVATION_SEMANTICS = "pre_reset_final"
FASTSAC_ACTOR_BACKEND = "hoi_fastsac_tanh_gaussian_v1"
FASTSAC_DETERMINISTIC_ACTION_KEY = "_fastsac_deterministic_action"
FASTSAC_TEACHER_TRAINING_ALGORITHM = "hoi_fastsac_teacher_v2"
FASTSAC_STUDENT_TRAINING_ALGORITHM = "hoi_fastsac_student_rlpd_v2"
TEACHER_OBJECT_GEO_FIELD = "teacher_object_geo"
TEACHER_HEIGHT_FIELD = "teacher_height"
NEXT_TEACHER_HEIGHT_FIELD = "next_teacher_height"
TEACHER_REPLAY_FIELDS = (
    "observations", "critic_observations", "actions", "rewards", "dones",
    "truncations", "discounts", "next_observations", "next_critic_observations",
)


class FastSACTanhNormal(TanhNormal):
    """FastSAC's reparameterized tanh Gaussian with TensorDict helpers."""

    dist_keys = ["loc", "scale"]

    @property
    def mean(self):
        # TanhNormal has no analytic expectation. HOI uses tanh(mu) for
        # deterministic inference, which is also the quantity distilled by VAIC.
        return self.deterministic_sample

    @property
    def mode(self):
        return self.deterministic_sample

    def entropy(self):
        # The squashed distribution has no analytic entropy. FastSAC itself
        # uses the transformed log probability directly.
        sample = self.rsample()
        return -self.log_prob(sample)


@register_kl(FastSACTanhNormal, FastSACTanhNormal)
def _kl_fastsac_tanh_normal(p, q):
    # Both distributions use the same bijective tanh/affine action transform,
    # so their KL equals the KL between the underlying diagonal Gaussians.
    if not torch.equal(p.low, q.low) or not torch.equal(p.high, q.high):
        raise ValueError("FastSACTanhNormal KL requires identical action bounds")
    return torch.distributions.kl_divergence(p.base_dist, q.base_dist)


class FastSACActor(nn.Module):
    """HOI FastSAC actor network, parameterized for VAIC observation/action sizes."""

    def __init__(
        self, input_dim, action_dim, hidden_dim, log_std_min, log_std_max,
        action_low, action_high, layer_norm=True,
    ):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("fastsac_actor_hidden_dim must be at least 4")
        if not log_std_min < log_std_max:
            raise ValueError("fastsac_log_std_min must be below fastsac_log_std_max")
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("FastSAC action bounds must match the VAIC action dimension")
        if not torch.all(action_high > action_low):
            raise ValueError("FastSAC action upper bounds must exceed lower bounds")

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        mid_dim = hidden_dim // 2
        last_dim = hidden_dim // 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(mid_dim, last_dim),
            nn.LayerNorm(last_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(last_dim, action_dim))
        self.fc_logstd = nn.Linear(last_dim, action_dim)

        # Match HOI: the two output heads start at mean=0 and log_std midpoint.
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        self.register_buffer("action_scale", (action_high - action_low) * 0.5)
        self.register_buffer("action_bias", (action_high + action_low) * 0.5)

    def forward(self, observations):
        features = self.net(observations)
        loc = self.fc_mu(features)
        raw_log_std = torch.tanh(self.fc_logstd(features))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (raw_log_std + 1.0)
        scale = log_std.exp()
        deterministic_action = torch.tanh(loc) * self.action_scale + self.action_bias
        return loc, scale, deterministic_action


def _timeout_bootstrap_mask(dones: torch.Tensor, truncations: torch.Tensor) -> torch.Tensor:
    """Bootstrap ordinary transitions and time-limit truncations, not true terminals."""
    return (truncations.bool() | ~dones.bool()).float()


class DistributionalQNetwork(nn.Module):
    """The distributional Q network used by HOI FastSAC/r1-student."""

    def __init__(self, obs_dim, action_dim, hidden_dim, num_atoms, layer_norm=True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_dim + action_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 4))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        return self.net(torch.cat((obs, action), dim=-1))


class TwinDistributionalQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_atoms, v_min, v_max, layer_norm=True):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.qnets = nn.ModuleList(
            DistributionalQNetwork(obs_dim, action_dim, hidden_dim, num_atoms, layer_norm)
            for _ in range(2)
        )
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, obs, action):
        return torch.stack([q(obs, action) for q in self.qnets], dim=0)

    def values(self, logits):
        return (F.softmax(logits, dim=-1) * self.support).sum(dim=-1)

    @torch.no_grad()
    def projection(self, obs, action, reward, bootstrap, discount):
        """C51 Bellman projection, independently for Q1 and Q2 (as in HOI)."""
        delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
        target = reward[:, None] + bootstrap[:, None] * discount[:, None] * self.support
        target = target.clamp(self.v_min, self.v_max)
        b = (target - self.v_min) / delta
        lower = b.floor().long()
        upper = b.ceil().long()
        same = lower == upper
        lower = torch.where(same & (lower > 0), lower - 1, lower)
        upper = torch.where(same & (upper == 0), upper + 1, upper)
        offset = torch.arange(reward.shape[0], device=reward.device)[:, None] * self.num_atoms

        projected = []
        for qnet in self.qnets:
            probs = F.softmax(qnet(obs, action), dim=-1)
            out = torch.zeros_like(probs)
            out.view(-1).index_add_(0, (lower + offset).reshape(-1), (probs * (upper - b)).reshape(-1))
            out.view(-1).index_add_(0, (upper + offset).reshape(-1), (probs * (b - lower)).reshape(-1))
            projected.append(out)
        return torch.stack(projected, dim=0)


def _build_isolated_q_network(
    obs_dim, action_dim, hidden_dim, num_atoms, v_min, v_max,
    layer_norm, device, seed,
):
    """Build Q1/Q2 without advancing VAIC/environment default RNG streams."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    # Module constructors use the CPU default generator today, but also fork the
    # target CUDA generator so a future device-side initializer cannot leak into
    # rollout sampling.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(int(seed))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        qnet = TwinDistributionalQ(
            obs_dim, action_dim, hidden_dim, num_atoms, v_min, v_max, layer_norm,
        ).to(device)
    return qnet


class TeacherReplayBuffer:
    """Device-resident FIFO with atomic H5 snapshots for later FastSAC use.

    Rollout appends are device-local.  H5 and host transfers occur only when a
    model checkpoint explicitly requests a snapshot.
    """

    fields = TEACHER_REPLAY_FIELDS

    def __init__(
        self, path, capacity, actor_dim, critic_dim, action_dim, seed,
        device="cpu", snapshot_chunk_rows=4096, replay_id=None,
        actor_backend=FASTSAC_ACTOR_BACKEND, actor_obs_keys=None,
        critic_obs_keys=None, extra_shapes=None,
    ):
        if int(capacity) < 1:
            raise ValueError("teacher replay capacity must be positive")
        if int(snapshot_chunk_rows) < 1:
            raise ValueError("snapshot_chunk_rows must be positive")
        self.path = os.fspath(path)
        self.capacity = int(capacity)
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.replay_id = str(replay_id or uuid.uuid4())
        self.actor_backend = str(actor_backend)
        self.actor_obs_keys = list(actor_obs_keys or [])
        self.critic_obs_keys = list(critic_obs_keys or [])
        self.device = torch.device(device)
        self.snapshot_chunk_rows = int(snapshot_chunk_rows)
        self.shapes = {
            "observations": (self.actor_dim,),
            "critic_observations": (self.critic_dim,),
            "actions": (self.action_dim,),
            "rewards": (),
            "dones": (),
            "truncations": (),
            "discounts": (),
            "next_observations": (self.actor_dim,),
            "next_critic_observations": (self.critic_dim,),
        }
        extra_shapes = {
            str(name): tuple(int(dim) for dim in shape)
            for name, shape in dict(extra_shapes or {}).items()
        }
        duplicate_fields = set(extra_shapes).intersection(self.shapes)
        if duplicate_fields:
            raise ValueError(
                f"Teacher replay extra fields duplicate base fields: {sorted(duplicate_fields)}"
            )
        self.shapes.update(extra_shapes)
        self.storage_fields = (*self.fields, *extra_shapes.keys())
        self.dtypes = {
            name: torch.bool if name in ("dones", "truncations") else torch.float32
            for name in self.storage_fields
        }
        # Allocate lazily at the first FastSAC rollout append.
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None

    @property
    def saved(self):
        return self.size

    @property
    def estimated_bytes(self):
        return self.capacity * sum(
            int(np.prod(tail, dtype=np.int64) if tail else 1)
            * torch.empty((), dtype=self.dtypes[name]).element_size()
            for name, tail in self.shapes.items()
        )

    def _allocate(self):
        if self.data:
            return
        try:
            for name in self.storage_fields:
                self.data[name] = torch.empty(
                    (self.capacity, *self.shapes[name]),
                    dtype=self.dtypes[name], device=self.device,
                )
        except torch.OutOfMemoryError as exc:
            self.data.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gib = self.estimated_bytes / (1024 ** 3)
            raise RuntimeError(
                f"Unable to allocate the {gib:.2f} GiB teacher replay FIFO on "
                f"{self.device}. Reduce teacher_buffer_capacity."
            ) from exc

    def _validated_values(self, data):
        missing = [name for name in self.storage_fields if name not in data]
        if missing:
            raise KeyError(f"Teacher replay append is missing fields: {missing}")
        count = int(data["rewards"].shape[0])
        values = {}
        for name in self.storage_fields:
            value = data[name].detach()
            expected = (count, *self.shapes[name])
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Teacher replay field {name!r} has shape {tuple(value.shape)}, "
                    f"expected {expected}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"Teacher replay field {name!r} is on {value.device}, "
                    f"expected {self.device}"
                )
            values[name] = value.to(dtype=self.dtypes[name])
        return count, values

    @torch.no_grad()
    def append(self, data):
        count, values = self._validated_values(data)
        if count == 0:
            return 0
        # The in-memory FIFO has advanced beyond the last on-disk snapshot.
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None
        self._allocate()
        self.seen += count

        if count >= self.capacity:
            for name in self.storage_fields:
                self.data[name].copy_(values[name][-self.capacity:])
            self.ptr = 0
            self.size = self.capacity
            return count

        first = min(count, self.capacity - self.ptr)
        second = count - first
        for name in self.storage_fields:
            self.data[name][self.ptr:self.ptr + first].copy_(values[name][:first])
            if second:
                self.data[name][:second].copy_(values[name][first:])
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    def clear(self):
        """Logically empty the FIFO without reallocating its device tensors."""
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None

    def sample(self, count, device=None, generator=None, fields=None):
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty teacher FastSAC replay.")
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Teacher replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(
            0, self.size, (int(count),), device=self.device, generator=generator,
        )
        fields = self.storage_fields if fields is None else tuple(fields)
        unknown = set(fields).difference(self.storage_fields)
        if unknown:
            raise KeyError(f"Unknown teacher replay sample fields: {sorted(unknown)}")
        return {name: self.data[name][indices] for name in fields}

    def _chronological_segments(self, row_count=None):
        if self.size == 0:
            return []
        if self.size < self.capacity:
            segments = [(0, self.size)]
        elif self.ptr == 0:
            segments = [(0, self.capacity)]
        else:
            segments = [(self.ptr, self.capacity), (0, self.ptr)]

        if row_count is None:
            return segments
        row_count = max(0, min(int(row_count), self.size))
        skip = self.size - row_count
        selected = []
        for start, end in segments:
            length = end - start
            if skip >= length:
                skip -= length
                continue
            selected.append((start + skip, end))
            skip = 0
        return selected

    def checkpoint_metadata(self):
        """Return the FIFO/snapshot manifest stored beside the policy weights."""
        has_snapshot = self.last_snapshot_id is not None
        return {
            "format_version": TEACHER_REPLAY_FORMAT_VERSION,
            "replay_id": self.replay_id,
            "actor_backend": self.actor_backend,
            "actor_obs_keys": list(self.actor_obs_keys),
            "critic_obs_keys": list(self.critic_obs_keys),
            "actor_obs_dim": self.actor_dim,
            "critic_obs_dim": self.critic_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "size": (
                self.last_snapshot_size if has_snapshot else self.size
            ),
            "seen": (
                self.last_snapshot_seen if has_snapshot else self.seen
            ),
            "storage_fields": list(self.storage_fields),
            "field_shapes": {
                name: list(self.shapes[name]) for name in self.storage_fields
            },
            "snapshot_iteration": self.last_snapshot_iteration,
            "checkpoint_name": self.last_snapshot_name,
            "snapshot_id": self.last_snapshot_id,
        }

    def _validate_restore_file(self, replay, expected_metadata=None):
        path = self.path
        if str(replay.attrs.get("format", "")) != "vaic_fastsac_teacher_buffer":
            raise ValueError(f"Not a VAIC FastSAC teacher buffer: {path}")
        version = int(replay.attrs.get("format_version", 0))
        if version != TEACHER_REPLAY_FORMAT_VERSION:
            raise ValueError(
                f"Teacher replay {path} uses format version {version}; expected "
                f"{TEACHER_REPLAY_FORMAT_VERSION}."
            )
        if (
            str(replay.attrs.get("timeout_next_observation", ""))
            != TIMEOUT_NEXT_OBSERVATION_SEMANTICS
        ):
            raise ValueError(
                f"Teacher replay {path} does not contain true pre-reset timeout "
                "final observations."
            )
        if str(replay.attrs.get("storage_policy", "")) != "circular_fifo":
            raise ValueError(f"Teacher replay {path} is not a circular FIFO snapshot.")
        if str(replay.attrs.get("storage_order", "")) != "oldest_to_newest":
            raise ValueError(
                f"Teacher replay {path} is not stored oldest-to-newest."
            )

        actor_keys = json.loads(str(replay.attrs.get("actor_obs_keys", "[]")))
        critic_keys = json.loads(str(replay.attrs.get("critic_obs_keys", "[]")))
        actual_storage_fields = json.loads(str(replay.attrs.get(
            "storage_fields", json.dumps(list(replay.keys()))
        )))
        actual_field_shapes = json.loads(str(replay.attrs.get(
            "field_shapes",
            json.dumps({name: list(replay[name].shape[1:]) for name in replay.keys()}),
        )))
        actual = {
            "format_version": version,
            "replay_id": str(replay.attrs.get("replay_id", "")),
            "actor_backend": str(replay.attrs.get("actor_backend", "")),
            "actor_obs_keys": actor_keys,
            "critic_obs_keys": critic_keys,
            "actor_obs_dim": int(replay.attrs.get("actor_obs_dim", -1)),
            "critic_obs_dim": int(replay.attrs.get("critic_obs_dim", -1)),
            "action_dim": int(replay.attrs.get("action_dim", -1)),
            "capacity": int(replay.attrs.get("buffer_capacity", -1)),
            "size": int(replay.attrs.get("num_transitions", -1)),
            "seen": int(replay.attrs.get("num_seen_transitions", -1)),
            "storage_fields": actual_storage_fields,
            "field_shapes": actual_field_shapes,
            "snapshot_iteration": int(replay.attrs.get("snapshot_iteration", -1)),
            "checkpoint_name": str(replay.attrs.get("checkpoint_name", "")),
            "snapshot_id": str(replay.attrs.get("snapshot_id", "")),
        }
        required = {
            "replay_id": self.replay_id,
            "actor_backend": self.actor_backend,
            "actor_obs_keys": self.actor_obs_keys,
            "critic_obs_keys": self.critic_obs_keys,
            "actor_obs_dim": self.actor_dim,
            "critic_obs_dim": self.critic_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "storage_fields": list(self.storage_fields),
            "field_shapes": {
                name: list(self.shapes[name]) for name in self.storage_fields
            },
        }
        for key, expected in required.items():
            if actual[key] != expected:
                raise ValueError(
                    f"Teacher replay {key} {actual[key]!r} does not match the "
                    f"resumed policy value {expected!r}."
                )

        if actual["size"] < 0 or actual["size"] > self.capacity:
            raise ValueError(
                f"Teacher replay size {actual['size']} is outside [0, {self.capacity}]."
            )
        if actual["seen"] < actual["size"]:
            raise ValueError(
                f"Teacher replay seen count {actual['seen']} is smaller than its "
                f"stored size {actual['size']}."
            )
        if actual["size"] != min(actual["seen"], self.capacity):
            raise ValueError(
                "Teacher replay FIFO counters are inconsistent: "
                f"size={actual['size']}, seen={actual['seen']}, "
                f"capacity={self.capacity}."
            )

        if expected_metadata is not None:
            for key, expected in dict(expected_metadata).items():
                if key in actual and expected is not None and actual[key] != expected:
                    raise ValueError(
                        f"Teacher replay snapshot {key} {actual[key]!r} does not "
                        f"match checkpoint value {expected!r}. Use the H5 saved "
                        "with that exact checkpoint."
                    )

        actual_names = set(replay.keys())
        expected_names = set(self.storage_fields)
        if actual_names != expected_names:
            raise ValueError(
                f"Teacher replay datasets {sorted(actual_names)} do not match "
                f"{sorted(expected_names)}."
            )
        for name in self.storage_fields:
            expected_shape = (actual["size"], *self.shapes[name])
            if tuple(replay[name].shape) != expected_shape:
                raise ValueError(
                    f"Teacher replay dataset {name!r} has shape "
                    f"{tuple(replay[name].shape)}, expected {expected_shape}."
                )
            expected_dtype = np.dtype(
                np.bool_ if self.dtypes[name] == torch.bool else np.float32
            )
            if np.dtype(replay[name].dtype) != expected_dtype:
                raise ValueError(
                    f"Teacher replay dataset {name!r} has dtype "
                    f"{replay[name].dtype}, expected {expected_dtype}."
                )
        return actual

    @torch.no_grad()
    def restore(self, source_path, expected_metadata=None):
        """Restore a matching H5 snapshot into the device-resident FIFO once."""
        if self.data or self.size or self.seen:
            raise RuntimeError("Teacher replay restore requires a new empty FIFO.")
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to restore teacher replay") from exc

        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Teacher replay snapshot does not exist: {source_path}")
        try:
            with h5py.File(source_path, "r") as replay:
                # Error messages should identify the source, not the new run's
                # destination path.
                destination_path = self.path
                self.path = source_path
                try:
                    actual = self._validate_restore_file(replay, expected_metadata)
                finally:
                    self.path = destination_path

                if actual["size"]:
                    self._allocate()
                    for name in self.storage_fields:
                        for start in range(
                            0, actual["size"], self.snapshot_chunk_rows
                        ):
                            end = min(start + self.snapshot_chunk_rows, actual["size"])
                            host = torch.from_numpy(np.asarray(replay[name][start:end]))
                            self.data[name][start:end].copy_(host)
        except Exception:
            self.data.clear()
            self.ptr = 0
            self.size = 0
            self.seen = 0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise

        self.size = actual["size"]
        self.seen = actual["seen"]
        self.ptr = self.size % self.capacity
        self.last_snapshot_iteration = actual["snapshot_iteration"]
        self.last_snapshot_name = actual["checkpoint_name"]
        self.last_snapshot_id = actual["snapshot_id"] or None
        self.last_snapshot_size = actual["size"]
        self.last_snapshot_seen = actual["seen"]
        return self.size

    def snapshot(
        self, iteration, checkpoint_name, row_count=None, seen_count=None
    ):
        """Write oldest-to-newest contents and atomically replace the last H5."""
        snapshot_size = (
            self.size if row_count is None else min(int(row_count), self.size)
        )
        snapshot_seen = self.seen if seen_count is None else int(seen_count)
        if snapshot_size == 0:
            return None
        if snapshot_seen < snapshot_size:
            raise ValueError(
                f"snapshot seen_count={snapshot_seen} is smaller than "
                f"row_count={snapshot_size}"
            )
        if snapshot_size != min(snapshot_seen, self.capacity):
            raise ValueError(
                "snapshot row_count must be the circular-FIFO tail implied by "
                f"seen_count: rows={snapshot_size}, seen={snapshot_seen}, "
                f"capacity={self.capacity}"
            )
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to snapshot teacher replay") from exc

        snapshot_id = str(uuid.uuid4())
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.path)}.", suffix=".tmp", dir=directory,
        )
        os.close(fd)
        try:
            with h5py.File(temp_path, "w") as replay:
                replay.attrs.update({
                    "format": "vaic_fastsac_teacher_buffer",
                    "format_version": TEACHER_REPLAY_FORMAT_VERSION,
                    "timeout_next_observation": TIMEOUT_NEXT_OBSERVATION_SEMANTICS,
                    "source": f"{self.actor_backend}_teacher",
                    "actor_backend": self.actor_backend,
                    "replay_id": self.replay_id,
                    "teacher_only": True,
                    "storage_policy": "circular_fifo",
                    "storage_order": "oldest_to_newest",
                    "actor_obs_keys": json.dumps(self.actor_obs_keys),
                    "critic_obs_keys": json.dumps(self.critic_obs_keys),
                    "storage_fields": json.dumps(list(self.storage_fields)),
                    "field_shapes": json.dumps({
                        name: list(self.shapes[name]) for name in self.storage_fields
                    }),
                    "actor_obs_dim": self.actor_dim,
                    "critic_obs_dim": self.critic_dim,
                    "action_dim": self.action_dim,
                    "buffer_capacity": self.capacity,
                    # Legacy alias retained for format-v2 consumers.
                    "reservoir_capacity": self.capacity,
                    "num_transitions": snapshot_size,
                    "num_seen_transitions": snapshot_seen,
                    "snapshot_iteration": int(iteration),
                    "checkpoint_name": str(checkpoint_name),
                    "snapshot_id": snapshot_id,
                })
                datasets = {
                    name: replay.create_dataset(
                        name, (snapshot_size, *self.shapes[name]),
                        dtype=np.bool_ if self.dtypes[name] == torch.bool else np.float32,
                    )
                    for name in self.storage_fields
                }
                destination = 0
                for segment_start, segment_end in self._chronological_segments(
                    snapshot_size
                ):
                    for chunk_start in range(
                        segment_start, segment_end, self.snapshot_chunk_rows
                    ):
                        chunk_end = min(
                            chunk_start + self.snapshot_chunk_rows, segment_end
                        )
                        count = chunk_end - chunk_start
                        for name in self.storage_fields:
                            arrays = self.data[name][chunk_start:chunk_end].to("cpu").numpy()
                            datasets[name][destination:destination + count] = arrays
                        destination += count
                replay.flush()

            file_fd = os.open(temp_path, os.O_RDONLY)
            try:
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.replace(temp_path, self.path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        self.last_snapshot_iteration = int(iteration)
        self.last_snapshot_name = str(checkpoint_name)
        self.last_snapshot_id = snapshot_id
        self.last_snapshot_size = snapshot_size
        self.last_snapshot_seen = snapshot_seen
        return self.path


class OfflineReplayH5:
    """Load a teacher H5 once, then sample entirely from the target device."""

    def __init__(
        self, path, actor_dim, critic_dim, action_dim, device="cpu",
        max_size=None, seed=0, load_chunk_rows=4096,
        expected_replay_id=None, expected_actor_backend=None,
        expected_actor_obs_keys=None, expected_critic_obs_keys=None,
        expected_snapshot_metadata=None,
    ):
        import h5py
        if max_size is not None and int(max_size) < 1:
            raise ValueError("offline replay max_size must be positive")
        if int(load_chunk_rows) < 1:
            raise ValueError("offline replay load_chunk_rows must be positive")
        self.path = path
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))
        shapes = {
            "observations": (int(actor_dim),),
            "critic_observations": (int(critic_dim),),
            "actions": (int(action_dim),),
            "rewards": (), "dones": (), "truncations": (), "discounts": (),
            "next_observations": (int(actor_dim),),
            "next_critic_observations": (int(critic_dim),),
        }
        dtypes = {
            name: torch.bool if name in ("dones", "truncations") else torch.float32
            for name in TEACHER_REPLAY_FIELDS
        }
        with h5py.File(path, "r") as f:
            if str(f.attrs.get("format", "")) != "vaic_fastsac_teacher_buffer":
                raise ValueError(f"Not a VAIC FastSAC teacher buffer: {path}")
            version = int(f.attrs.get("format_version", 0))
            if version != TEACHER_REPLAY_FORMAT_VERSION:
                raise ValueError(
                    f"Teacher replay {path} uses legacy format version {version}; version "
                    f"{TEACHER_REPLAY_FORMAT_VERSION} with true timeout final observations is required."
                )
            if str(f.attrs.get("timeout_next_observation", "")) != TIMEOUT_NEXT_OBSERVATION_SEMANTICS:
                raise ValueError(
                    f"Teacher replay {path} does not declare pre-reset timeout final observations."
                )
            actual_replay_id = str(f.attrs.get("replay_id", ""))
            actual_actor_backend = str(f.attrs.get("actor_backend", ""))
            if expected_replay_id is not None and actual_replay_id != str(expected_replay_id):
                raise ValueError(
                    f"Teacher replay id {actual_replay_id!r} does not match checkpoint "
                    f"replay id {str(expected_replay_id)!r}."
                )
            if (
                expected_actor_backend is not None
                and actual_actor_backend != str(expected_actor_backend)
            ):
                raise ValueError(
                    f"Teacher replay actor backend {actual_actor_backend!r} does not match "
                    f"checkpoint backend {str(expected_actor_backend)!r}."
                )
            actual_actor_keys = json.loads(str(f.attrs.get("actor_obs_keys", "[]")))
            actual_critic_keys = json.loads(str(f.attrs.get("critic_obs_keys", "[]")))
            if (
                expected_actor_obs_keys is not None
                and actual_actor_keys != list(expected_actor_obs_keys)
            ):
                raise ValueError(
                    f"Teacher replay actor observation keys {actual_actor_keys} do not match "
                    f"checkpoint keys {list(expected_actor_obs_keys)}."
                )
            if (
                expected_critic_obs_keys is not None
                and actual_critic_keys != list(expected_critic_obs_keys)
            ):
                raise ValueError(
                    f"Teacher replay critic observation keys {actual_critic_keys} do not match "
                    f"checkpoint keys {list(expected_critic_obs_keys)}."
                )
            expected = (actor_dim, critic_dim, action_dim)
            actual = tuple(int(f.attrs[k]) for k in ("actor_obs_dim", "critic_obs_dim", "action_dim"))
            if actual != expected:
                raise ValueError(f"Teacher replay dimensions {actual} do not match FastSAC {expected}")
            file_size = int(f.attrs["num_transitions"])
            if file_size < 1:
                raise ValueError(f"Teacher replay buffer is empty: {path}")
            snapshot_actual = {
                "snapshot_id": str(f.attrs.get("snapshot_id", "")),
                "snapshot_iteration": int(
                    f.attrs.get("snapshot_iteration", -1)
                ),
                "checkpoint_name": str(f.attrs.get("checkpoint_name", "")),
                "size": file_size,
                "seen": int(f.attrs.get("num_seen_transitions", -1)),
            }
            if expected_snapshot_metadata is not None:
                expected_snapshot_metadata = dict(expected_snapshot_metadata)
                for key, actual_value in snapshot_actual.items():
                    expected_value = expected_snapshot_metadata.get(key)
                    if (
                        expected_value is not None
                        and actual_value != expected_value
                    ):
                        raise ValueError(
                            f"Teacher replay snapshot {key} {actual_value!r} does "
                            f"not match checkpoint value {expected_value!r}. Use "
                            "the H5 saved with that exact checkpoint."
                        )
            self.snapshot_metadata = snapshot_actual
            self.size = file_size if max_size is None else min(file_size, int(max_size))
            source_start = file_size - self.size
            if self.size < file_size:
                storage_order = str(f.attrs.get("storage_order", ""))
                selection = (
                    "newest" if storage_order == "oldest_to_newest"
                    else "trailing deterministic subset"
                )
                logging.warning(
                    "Teacher replay has %d rows; loading a %s of %d rows onto %s.",
                    file_size, selection, self.size, self.device,
                )
            for name in TEACHER_REPLAY_FIELDS:
                expected_shape = (file_size, *shapes[name])
                if name not in f or tuple(f[name].shape) != expected_shape:
                    actual_shape = tuple(f[name].shape) if name in f else None
                    raise ValueError(
                        f"Teacher replay dataset {name!r} has shape {actual_shape}, "
                        f"expected {expected_shape}"
                    )
            self.data = {}
            try:
                for name in TEACHER_REPLAY_FIELDS:
                    self.data[name] = torch.empty(
                        (self.size, *shapes[name]), dtype=dtypes[name],
                        device=self.device,
                    )
                for name in TEACHER_REPLAY_FIELDS:
                    for destination in range(0, self.size, int(load_chunk_rows)):
                        count = min(int(load_chunk_rows), self.size - destination)
                        source = source_start + destination
                        host = torch.from_numpy(np.asarray(f[name][source:source + count]))
                        self.data[name][destination:destination + count].copy_(host)
            except torch.OutOfMemoryError as exc:
                self.data.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                required = self.size * sum(
                    int(np.prod(tail, dtype=np.int64) if tail else 1)
                    * torch.empty((), dtype=dtypes[name]).element_size()
                    for name, tail in shapes.items()
                )
                raise RuntimeError(
                    f"Unable to load {required / (1024 ** 3):.2f} GiB teacher replay "
                    f"onto {self.device}. Reduce teacher_buffer_capacity."
                ) from exc

    def sample(self, count, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Offline replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(
            0, self.size, (int(count),), device=self.device, generator=self.rng,
        )
        return {name: value[indices] for name, value in self.data.items()}


class OnlineReplay:
    """Device-resident circular replay collected by the FastSAC student."""

    def __init__(self, capacity, device="cpu"):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("online replay capacity must be positive")
        self.device = torch.device(device)
        self.data = {}
        self.ptr = 0
        self.size = 0

    def extend(self, transitions):
        values = {
            k: v.detach().reshape(v.shape[0], *v.shape[1:])
            for k, v in transitions.items()
        }
        wrong_devices = {
            key: str(value.device)
            for key, value in values.items()
            if value.device != self.device
        }
        if wrong_devices:
            raise ValueError(
                f"Online replay transitions must already be on {self.device}; got "
                f"{wrong_devices}"
            )
        n = next(iter(values.values())).shape[0]
        if not self.data:
            try:
                self.data = {
                    k: torch.empty(
                        (self.capacity, *v.shape[1:]),
                        dtype=v.dtype, device=self.device,
                    )
                    for k, v in values.items()
                }
            except torch.OutOfMemoryError as exc:
                self.data.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Unable to allocate FastSAC online replay on {self.device}; "
                    "reduce online_buffer_capacity."
                ) from exc
        if n >= self.capacity:
            values = {k: v[-self.capacity:] for k, v in values.items()}
            n = self.capacity
        first = min(n, self.capacity - self.ptr)
        second = n - first
        for key, value in values.items():
            self.data[key][self.ptr:self.ptr + first].copy_(value[:first])
            if second:
                self.data[key][:second].copy_(value[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, count, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Online replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(0, self.size, (count,), device=self.device)
        return {k: v[indices] for k, v in self.data.items()}


@dataclass
class FastSACVelConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.ppo.fastsac_vel.FastSACVEL"
    q_hidden_dim: int = 768
    q_num_atoms: int = 101
    q_v_min: float = -20.0
    q_v_max: float = 20.0
    q_layer_norm: bool = True
    q_lr: float = 3e-4
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    save_teacher_buffer: bool = True
    teacher_buffer_filename: str = "teacher_replay_buffer.h5"
    # For same-stage teacher resume this is the source FIFO snapshot. For
    # student finetuning it is the immutable 50% RLPD offline dataset.
    teacher_buffer_path: str | None = None
    teacher_buffer_capacity: int = 262_144
    teacher_buffer_seed: int = 0
    teacher_buffer_snapshot_chunk_rows: int = 4096
    # True teacher FastSAC uses its replay immediately for learning, but only
    # exposes rows collected at/after this iteration in checkpoint H5 files.
    teacher_buffer_start_iteration: int = 5100
    fastsac_actor_hidden_dim: int = 512
    fastsac_log_std_min: float = -5.0
    fastsac_log_std_max: float = 0.0
    fastsac_actor_layer_norm: bool = True
    # HOI FastSAC loss/hyperparameter defaults. Teacher training inserts each
    # vector-environment step and completes its 8 Q/alpha plus two delayed actor
    # updates before selecting the next action. Student finetuning intentionally
    # keeps VAIC's existing 32-step collection/adaptation schedule.
    sac_learning_starts: int = 10
    sac_batch_size: int = 8192
    sac_updates_per_env_step: int = 8
    sac_policy_frequency: int = 4
    sac_actor_lr: float = 3e-4
    sac_alpha_lr: float = 3e-4
    sac_alpha_init: float = 0.001
    sac_target_entropy_ratio: float = 0.0
    sac_tau: float = 0.125
    sac_max_grad_norm: float = 0.0


@dataclass
class FastSACVelFinetuneConfig(FastSACVelConfig):
    _target_: str = "active_adaptation.learning.ppo.fastsac_vel.FastSACVelFinetune"
    phase: str = "finetune"
    vecnorm: str = "eval"
    enable_residual_distillation: bool = False
    save_teacher_buffer: bool = False
    teacher_buffer_ratio: float = 0.5
    online_buffer_capacity: int = 262_144


_in_keys: List[str] = (
    CMD_KEY,
    OBS_KEY,
    OBJECT_KEY,
    OBS_PRIV_KEY,
    OBJECT_GEO_KEY,
    HEIGHT_KEY,
    VEL_CMD_KEY,
)
ConfigStore.instance().store(
    "fastsac_vel_train",
    node=FastSACVelConfig(
        name="fastsac_vel", phase="train", vecnorm="train",
        gamma=0.97, entropy_coef_start=0.001, entropy_coef_end=0.001,
        teacher_buffer_start_iteration=5100,
        in_keys=_in_keys,
    ),
    group="algo",
)
ConfigStore.instance().store(
    "fastsac_vel_finetune",
    node=FastSACVelFinetuneConfig(
        name="fastsac_vel", phase="finetune", vecnorm="eval",
        gamma=0.97, enable_residual_distillation=False,
        in_keys=(
            CMD_KEY,
            OBS_KEY,
            OBJECT_KEY,
            OBS_PRIV_KEY,
            OBJECT_GEO_KEY,
            VEL_CMD_KEY,
            DEPTH_KEY,
        ),
    ),
    group="algo",
)


class _FastSACVAICBase(PPOVEL):
    """Shared FastSAC plumbing around unchanged VAIC perception modules."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if cfg.q_hidden_dim < 4:
            raise ValueError("q_hidden_dim must be at least 4")
        if cfg.q_num_atoms < 2 or not cfg.q_v_min < cfg.q_v_max:
            raise ValueError("distributional Q support must contain at least two atoms")
        if cfg.teacher_buffer_capacity < 1:
            raise ValueError("teacher_buffer_capacity must be positive")
        if cfg.teacher_buffer_snapshot_chunk_rows < 1:
            raise ValueError("teacher_buffer_snapshot_chunk_rows must be positive")
        replay_filename = os.fspath(cfg.teacher_buffer_filename)
        if (
            replay_filename in ("", ".", "..")
            or os.path.basename(replay_filename) != replay_filename
        ):
            raise ValueError(
                "teacher_buffer_filename must be a plain filename without a "
                "directory; use teacher_buffer_path to select an input path"
            )
        command_key = "command_" if observation_spec.get("command_", None) is not None else CMD_KEY
        self.q_critic_keys = [OBS_PRIV_KEY, OBS_KEY, command_key]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            self.q_critic_keys.append(OBJECT_KEY)
        self.q_actor_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        critic_dim = sum(observation_spec[k].shape[-1] for k in self.q_critic_keys)
        actor_dim = sum(observation_spec[k].shape[-1] for k in (VEL_CMD_KEY, OBS_KEY)) + cfg.latent_dim
        self.qnet = _build_isolated_q_network(
            critic_dim, self.action_dim, cfg.q_hidden_dim, cfg.q_num_atoms,
            cfg.q_v_min, cfg.q_v_max, cfg.q_layer_norm, device, cfg.q_seed,
        )
        self.qnet_target = copy.deepcopy(self.qnet).requires_grad_(False)
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95), fused=str(device).startswith("cuda"),
        )
        self._q_actor_dim = actor_dim
        self._q_critic_dim = critic_dim
        self.q_rng = torch.Generator(device=device).manual_seed(int(cfg.q_seed))
        self.teacher_replay = None
        self.teacher_replay_id = str(uuid.uuid4())
        self.actor_backend = FASTSAC_ACTOR_BACKEND
        self._loaded_checkpoint_phase = None
        self._loaded_teacher_replay_metadata = None
        self._teacher_replay_extra_shapes = {}
        self._timeout_final_batches = []
        self._last_timeout_finals_used = 0
        self.q_update_count = 0
        self._configure_actor_backend()

    def _configure_actor_backend(self):
        raise NotImplementedError

    def requires_training_replay(self):
        """Whether every training rank needs a replay even without H5 export."""
        return False

    def requires_value_bootstrap(self):
        return False

    def _optimizer_manifests(self, optimizers):
        parameter_names = {id(parameter): name for name, parameter in self.named_parameters()}
        manifests = {}
        for optimizer_name, optimizer in optimizers.items():
            groups = []
            for group in optimizer.param_groups:
                names = []
                for parameter in group["params"]:
                    name = parameter_names.get(id(parameter))
                    if name is None:
                        raise RuntimeError(
                            f"Optimizer {optimizer_name!r} contains a parameter that "
                            "is not registered on the policy."
                        )
                    names.append(name)
                groups.append(names)
            manifests[optimizer_name] = groups
        return manifests

    def _optimizer_resume_signature(self, optimizer_names):
        return {
            "policy_family": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
                if self.cfg.phase == "train"
                else FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
            "phase": str(self.cfg.phase),
            "actor_backend": str(self.actor_backend),
            "optimizer_names": list(optimizer_names),
        }

    @staticmethod
    def _normalize_optimizer_resume_signature(signature):
        """Map pre-rename true-FastSAC class names to a stable family id."""
        if not isinstance(signature, dict):
            return signature
        normalized = copy.deepcopy(signature)
        legacy_class = normalized.pop("policy_class", None)
        legacy_families = {
            "active_adaptation.learning.ppo.ppo_fastsac_vel.HOIFastSACVEL": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.ppo_fastsac_vel.FastSACVelFinetune": (
                FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.fastsac_vel.FastSACVEL": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.fastsac_vel.FastSACVelFinetune": (
                FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
        }
        if legacy_class is not None:
            family = legacy_families.get(legacy_class)
            if family is None:
                # In particular, do not treat the removed PPOFastSACVEL hybrid
                # as a true FastSAC checkpoint.
                normalized["policy_class"] = legacy_class
            else:
                normalized["policy_family"] = family
        return normalized

    def _optimizer_resume_state(self):
        optimizers = self._optimizer_registry()
        return {
            "version": 1,
            "signature": self._optimizer_resume_signature(optimizers.keys()),
            "parameter_manifests": self._optimizer_manifests(optimizers),
            "optimizer_states": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "num_updates": int(self.num_updates),
        }

    def _restore_optimizer_resume_state(self, state_dict):
        checkpoint_phase = state_dict.get("last_phase")
        if checkpoint_phase != self.cfg.phase:
            logging.info(
                "Starting phase %s from phase %s weights; optimizer moments are "
                "intentionally reinitialized.",
                self.cfg.phase, checkpoint_phase,
            )
            return False

        resume_state = state_dict.get("optimizer_resume_state")
        if resume_state is None:
            logging.warning(
                "This same-phase checkpoint predates optimizer-state saving; "
                "continuing with freshly initialized Adam/AdamW moments."
            )
            return False
        if int(resume_state.get("version", 0)) != 1:
            raise ValueError(
                "Unsupported FastSAC optimizer resume-state version: "
                f"{resume_state.get('version')!r}"
            )

        optimizers = self._optimizer_registry()
        expected_signature = self._optimizer_resume_signature(optimizers.keys())
        actual_signature = self._normalize_optimizer_resume_signature(
            resume_state.get("signature")
        )
        if actual_signature != expected_signature:
            logging.warning(
                "Skipping optimizer moments because the resume signature changed: "
                "checkpoint=%s, current=%s",
                actual_signature, expected_signature,
            )
            return False

        expected_manifests = self._optimizer_manifests(optimizers)
        actual_manifests = resume_state.get("parameter_manifests")
        if actual_manifests != expected_manifests:
            raise ValueError(
                "FastSAC optimizer parameter topology does not match the checkpoint; "
                "refusing to attach Adam moments to different parameters."
            )
        optimizer_states = resume_state.get("optimizer_states", {})
        if set(optimizer_states) != set(optimizers):
            raise ValueError(
                "FastSAC checkpoint optimizer set does not match the current policy: "
                f"checkpoint={sorted(optimizer_states)}, current={sorted(optimizers)}"
            )
        for name, optimizer in optimizers.items():
            try:
                optimizer.load_state_dict(optimizer_states[name])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to restore optimizer {name!r} for same-phase resume."
                ) from exc
        self.num_updates = int(resume_state.get("num_updates", 0))
        logging.info(
            "Restored %d optimizer states and num_updates=%d for same-phase resume.",
            len(optimizers), self.num_updates,
        )
        return True

    def _teacher_replay_checkpoint_metadata(self):
        if self.teacher_replay is None:
            return None
        return self.teacher_replay.checkpoint_metadata()

    def load_state_dict(self, state_dict, strict=True):
        failed = super().load_state_dict(state_dict, strict)
        if "q_rng_state" in state_dict:
            self.q_rng.set_state(state_dict["q_rng_state"].cpu())
        self.q_update_count = int(state_dict.get("q_update_count", 0))
        if "teacher_replay_id" in state_dict:
            self.teacher_replay_id = str(state_dict["teacher_replay_id"])
        self._loaded_checkpoint_phase = state_dict.get("last_phase")
        replay_metadata = state_dict.get("teacher_replay_state")
        self._loaded_teacher_replay_metadata = copy.deepcopy(replay_metadata)
        self._restore_optimizer_resume_state(state_dict)
        return failed

    def configure_teacher_replay(self, path, restore_path=None):
        if not self.cfg.save_teacher_buffer and not self.requires_training_replay():
            return
        self.teacher_replay = TeacherReplayBuffer(
            path, self.cfg.teacher_buffer_capacity, self._q_actor_dim,
            self._q_critic_dim, self.action_dim, self.cfg.teacher_buffer_seed,
            device=self.device,
            snapshot_chunk_rows=self.cfg.teacher_buffer_snapshot_chunk_rows,
            replay_id=self.teacher_replay_id,
            actor_backend=self.actor_backend,
            actor_obs_keys=self.q_actor_keys,
            critic_obs_keys=self.q_critic_keys,
            extra_shapes=self._teacher_replay_extra_shapes,
        )
        if restore_path is not None and self._loaded_checkpoint_phase is None:
            raise ValueError(
                "teacher_buffer_path can restore a FIFO only together with a "
                "same-stage checkpoint_path."
            )
        if (
            restore_path is not None
            and self._loaded_checkpoint_phase != self.cfg.phase
        ):
            logging.warning(
                "Ignoring teacher replay from phase %s while starting phase %s.",
                self._loaded_checkpoint_phase, self.cfg.phase,
            )
            restore_path = None

        expected = self._loaded_teacher_replay_metadata
        expected_size = None if expected is None else expected.get("size")
        replay_required = (
            self._loaded_checkpoint_phase == self.cfg.phase
            and expected_size is not None
            and int(expected_size) > 0
        )
        if (
            restore_path is not None
            and self._loaded_checkpoint_phase == self.cfg.phase
            and expected is None
            and not replay_required
        ):
            # The helper can discover a run's final H5 while the user selected
            # an older checkpoint from that run. Without a checkpoint replay
            # manifest there is no way to prove that the snapshots are paired;
            # starting empty is safer than leaking future transitions.
            logging.warning(
                "Ignoring unpaired teacher replay %s because this checkpoint "
                "contains no exact replay manifest.",
                restore_path,
            )
            restore_path = None
        if restore_path is None and replay_required:
            raise FileNotFoundError(
                "Same-phase FastSAC resume requires the teacher replay H5 saved "
                "with this checkpoint. Set algo.teacher_buffer_path explicitly "
                "or keep teacher_replay_buffer.h5 beside the checkpoint."
            )
        if restore_path is not None:
            restored = self.teacher_replay.restore(restore_path, expected)
            logging.info(
                "Restored %d teacher replay rows (seen=%d) from %s onto %s.",
                restored, self.teacher_replay.seen, restore_path, self.device,
            )
        logging.info(
            "Teacher replay FIFO: capacity=%d, device=%s, estimated=%.2f GiB",
            self.teacher_replay.capacity, self.teacher_replay.device,
            self.teacher_replay.estimated_bytes / (1024 ** 3),
        )

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        if self.teacher_replay is None:
            return None
        return self.teacher_replay.snapshot(iteration, checkpoint_name)

    def get_rollout_policy(self, mode="train"):
        policy = super().get_rollout_policy(mode)
        if self.cfg.phase == "train" and mode == "train":
            # PPOVEL already computes PRIV_PRED_KEY in its rollout sequence but
            # intentionally filters it from the returned TensorDict.  FastSAC
            # replay needs that deploy-facing actor observation, so retain the
            # existing result only for this subclass.
            selected = list(policy.out_keys)
            policy.reset_out_keys()
            policy.select_out_keys(*selected, PRIV_PRED_KEY)
        return policy

    def _cat(self, td, keys):
        return torch.cat([td[k] for k in keys], dim=-1)

    def begin_transition_collection(self):
        """Clear sparse pre-reset observations at the start of every rollout."""
        self._timeout_final_batches.clear()


class FastSACVEL(_FastSACVAICBase):
    """True FastSAC teacher with VAIC observations and student distillation.

    The private base reuses VAIC module construction and checkpoint plumbing.
    Teacher optimization contains no PPO/GAE/value update.
    """

    def _vaic_action_bounds(self):
        manager = self.env.action_manager
        joint_ids = manager.joint_ids
        limits = manager.asset.data.soft_joint_pos_limits[0, joint_ids].detach()
        default = manager.default_joint_pos[0, joint_ids].detach()
        scaling = manager.action_scaling.detach()
        raw_a = (limits[:, 0] - default) / scaling
        raw_b = (limits[:, 1] - default) / scaling
        # Match HOI's symmetric FastSAC action coordinates.  VAIC raw action 0
        # maps to the robot's default joint pose, so symmetric bounds also make
        # the zero-initialized HOI heads start at that safe/default pose.
        magnitude = torch.maximum(raw_a.abs(), raw_b.abs())
        action_low = (-magnitude).to(self.device, torch.float32)
        action_high = magnitude.to(self.device, torch.float32)
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise ValueError("VAIC joint limits produced non-finite FastSAC action bounds")
        return action_low, action_high

    def _actor_backend_metadata(self):
        return {
            "teacher_input_dim": self._fastsac_teacher_actor_dim,
            "student_input_dim": self._q_actor_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.cfg.fastsac_actor_hidden_dim,
            "log_std_min": self.cfg.fastsac_log_std_min,
            "log_std_max": self.cfg.fastsac_log_std_max,
            "layer_norm": self.cfg.fastsac_actor_layer_norm,
            "action_low": self._fastsac_action_low,
            "action_high": self._fastsac_action_high,
            "joint_names": list(self.joint_names),
        }

    def _q_backend_metadata(self):
        return {
            "actor_obs_keys": list(self.q_actor_keys),
            "critic_obs_keys": list(self.q_critic_keys),
            "actor_obs_dim": self._q_actor_dim,
            "critic_obs_dim": self._q_critic_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.cfg.q_hidden_dim,
            "num_atoms": self.cfg.q_num_atoms,
            "v_min": self.cfg.q_v_min,
            "v_max": self.cfg.q_v_max,
            "layer_norm": self.cfg.q_layer_norm,
            "gamma": self.cfg.gamma,
        }

    def _build_fastsac_actor(self, in_keys, input_dim, action_low, action_high):
        core = FastSACActor(
            input_dim=input_dim,
            action_dim=self.action_dim,
            hidden_dim=self.cfg.fastsac_actor_hidden_dim,
            log_std_min=self.cfg.fastsac_log_std_min,
            log_std_max=self.cfg.fastsac_log_std_max,
            action_low=action_low,
            action_high=action_high,
            layer_norm=self.cfg.fastsac_actor_layer_norm,
        ).to(self.device)
        params = Seq(
            CatTensors(in_keys, "_fastsac_actor_input", del_keys=False, sort=False),
            Mod(
                core,
                ["_fastsac_actor_input"],
                ["loc", "scale", FASTSAC_DETERMINISTIC_ACTION_KEY],
            ),
        )
        return ProbabilisticActor(
            module=params,
            in_keys=self.dist_keys,
            out_keys=[ACTION_KEY],
            distribution_class=self.dist_cls,
            return_log_prob=True,
        ).to(self.device)

    def _configure_actor_backend(self):
        action_low, action_high = self._vaic_action_bounds()
        self.dist_cls = functools.partial(
            FastSACTanhNormal,
            low=action_low,
            high=action_high,
            event_dims=1,
        )
        self.dist_keys = list(FastSACTanhNormal.dist_keys)
        command_key = (
            "command_" if self.observation_spec.get("command_", None) is not None
            else CMD_KEY
        )
        teacher_keys = [command_key, OBS_KEY, PRIV_FEATURE_KEY]
        student_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        teacher_dim = (
            self.observation_spec[command_key].shape[-1]
            + self.observation_spec[OBS_KEY].shape[-1]
            + self.cfg.latent_dim
        )
        student_dim = self._q_actor_dim
        self.actor = self._build_fastsac_actor(
            teacher_keys, teacher_dim, action_low, action_high
        )
        self.actor_adapt = self._build_fastsac_actor(
            student_keys, student_dim, action_low, action_high
        )
        self.actor_backend = FASTSAC_ACTOR_BACKEND
        self._q_critic_widths = [
            int(self.observation_spec[key].shape[-1])
            for key in self.q_critic_keys
        ]
        self._fastsac_teacher_actor_dim = int(teacher_dim)
        self._fastsac_action_low = action_low.detach().cpu().tolist()
        self._fastsac_action_high = action_high.detach().cpu().tolist()

        # PPOVEL builds policy/value optimizers while constructing the unchanged
        # VAIC modules. They are deliberately disabled here: FastSAC owns actor
        # and Q optimization, while the VAIC adaptation optimizers remain active.
        self.opt_policy = None
        self.opt_critic = None
        # Remove the unused PPO value path from the registered model/checkpoint
        # tree as well. PPOVEL is only a constructor for VAIC perception here.
        for attribute in ("critic", "value_norm", "gae", "critic_loss_fn"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        if self.cfg.enable_residual_distillation:
            self.opt_adapt_actor = torch.optim.Adam(
                self.actor_adapt.parameters(), lr=self.cfg.lr
            )

        # HOI updates the actor from replayed raw observations.  Do not store
        # encoder_priv outputs: they become stale as the encoder learns and
        # cannot carry gradients back into the unchanged VAIC perception path.
        # Most raw inputs are already packed in critic_observations; only the
        # tensors required to rebuild object/height features are extra fields.
        composite_batch_dims = len(self.observation_spec.shape)

        def observation_tail(key):
            return tuple(
                int(dim)
                for dim in self.observation_spec[key].shape[composite_batch_dims:]
            )

        self._teacher_raw_replay_fields = []
        self._teacher_replay_extra_shapes = {}
        if OBJECT_KEY in self.q_critic_keys:
            self._teacher_raw_replay_fields.append(
                (
                    OBJECT_GEO_KEY,
                    TEACHER_OBJECT_GEO_FIELD,
                    None,
                )
            )
            geo_shape = observation_tail(OBJECT_GEO_KEY)
            # Object geometry is fixed throughout an episode. Ordinary and
            # timeout-bootstrap next states therefore share the current row's
            # geometry; true terminals do not bootstrap. Store it only once.
            self._teacher_replay_extra_shapes[
                TEACHER_OBJECT_GEO_FIELD
            ] = geo_shape
        if hasattr(self, "height_encoder"):
            self._teacher_raw_replay_fields.append(
                (HEIGHT_KEY, TEACHER_HEIGHT_FIELD, NEXT_TEACHER_HEIGHT_FIELD)
            )
            height_shape = observation_tail(HEIGHT_KEY)
            self._teacher_replay_extra_shapes.update({
                TEACHER_HEIGHT_FIELD: height_shape,
                NEXT_TEACHER_HEIGHT_FIELD: height_shape,
            })
        teacher_learning_fields = [
            "critic_observations",
            "actions",
            "rewards",
            "dones",
            "truncations",
            "discounts",
            "next_critic_observations",
        ]
        for _, current_field, next_field in self._teacher_raw_replay_fields:
            teacher_learning_fields.append(current_field)
            if next_field is not None:
                teacher_learning_fields.append(next_field)
        self._teacher_learning_fields = tuple(teacher_learning_fields)
        self._rollout_final_batch = None
        self._teacher_export_started = False
        self._teacher_export_start_seen = None
        self.sac_environment_steps = 0
        self.sac_rollout_count = 0
        self.sac_update_count = 0
        self.sac_actor_update_count = 0
        self.sac_alpha_update_count = 0

        if self.cfg.phase == "train":
            if int(self.cfg.teacher_buffer_start_iteration) < 0:
                raise ValueError("teacher_buffer_start_iteration must be non-negative")
            if int(self.cfg.sac_learning_starts) < 0:
                raise ValueError("sac_learning_starts must be non-negative")
            if int(self.cfg.sac_batch_size) < 1:
                raise ValueError("sac_batch_size must be positive")
            if self._sac_updates_per_env_step() < 1:
                raise ValueError("sac_updates_per_env_step must be positive")
            if int(self.cfg.sac_policy_frequency) < 1:
                raise ValueError("sac_policy_frequency must be positive")
            if float(self.cfg.sac_alpha_init) <= 0.0:
                raise ValueError("sac_alpha_init must be positive")
            if not 0.0 <= float(self.cfg.sac_tau) <= 1.0:
                raise ValueError("sac_tau must be in [0, 1]")
            if float(self.cfg.sac_max_grad_norm) < 0.0:
                raise ValueError("sac_max_grad_norm must be non-negative")

            teacher_parameters = list(self.actor.parameters())
            teacher_parameters.extend(self.encoder_priv.parameters())
            if hasattr(self, "height_cnn"):
                teacher_parameters.extend(self.height_cnn.parameters())
            self._teacher_actor_parameters = teacher_parameters
            self.sac_teacher_actor_optimizer = torch.optim.AdamW(
                teacher_parameters,
                lr=self.cfg.sac_actor_lr,
                weight_decay=self.cfg.q_weight_decay,
                betas=(0.9, 0.95),
                fused=str(self.device).startswith("cuda"),
            )
            self.log_alpha = nn.Parameter(torch.tensor(
                float(np.log(self.cfg.sac_alpha_init)),
                device=self.device,
                dtype=torch.float32,
            ))
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.cfg.sac_alpha_lr,
                betas=(0.9, 0.95),
                fused=str(self.device).startswith("cuda"),
            )
            self.target_entropy = (
                -self.action_dim * float(self.cfg.sac_target_entropy_ratio)
            )

    def requires_training_replay(self):
        return self.cfg.phase == "train"

    def _sac_updates_per_env_step(self):
        """Return HOI's update count for one vector-environment step."""
        return int(self.cfg.sac_updates_per_env_step)

    def _sac_actor_update_is_due(
        self, update_index: int, logical_step: int, updates_per_step: int
    ) -> bool:
        """Apply HOI's delayed actor schedule, including frequency-one configs."""
        frequency = int(self.cfg.sac_policy_frequency)
        if updates_per_step > 1:
            return frequency == 1 or update_index % frequency == 1
        return logical_step % frequency == 0

    def configure_teacher_replay(self, path, restore_path=None):
        if (
            self.cfg.phase == "train"
            and self._loaded_checkpoint_phase == "train"
            and not self._teacher_export_started
            and restore_path is not None
        ):
            # A pre-gate checkpoint has no paired H5 by definition. Helpers may
            # still discover the final H5 sitting beside an older periodic
            # checkpoint; loading it would leak future transitions.
            logging.warning(
                "Ignoring teacher replay %s because the resumed checkpoint is "
                "from before the H5 export gate; FastSAC replay starts empty.",
                restore_path,
            )
            restore_path = None
        result = super().configure_teacher_replay(path, restore_path=restore_path)
        if self.cfg.phase == "train" and self._teacher_export_started:
            # A restored post-gate H5 contains export-eligible rows only. Treat
            # every restored row, and all new rows, as eligible for later H5s.
            self._teacher_export_start_seen = 0
        return result

    def _optimizer_registry(self):
        if self.cfg.phase == "train":
            names = (
                "opt_adapt",
                "opt_adapt_actor",
                "opt_dr_estimator",
                "opt_q",
                "sac_teacher_actor_optimizer",
                "alpha_optimizer",
            )
        else:
            names = (
                "opt_adapt",
                "opt_dr_estimator",
                "opt_q",
                "sac_actor_optimizer",
                "alpha_optimizer",
            )
        return OrderedDict(
            (name, getattr(self, name))
            for name in names
            if isinstance(getattr(self, name, None), torch.optim.Optimizer)
        )

    def get_rollout_policy(self, mode="train"):
        # The base replay hook retains priv_pred for the deploy-facing H5. Raw
        # teacher inputs needed by FastSAC are already environment observations;
        # keeping encoded latents/deterministic actions would only enlarge the
        # dense VAIC rollout buffer.
        return super().get_rollout_policy(mode)

    def begin_transition_collection(self):
        super().begin_transition_collection()
        self._rollout_final_batch = None
        self._interleaved_steps_collected = 0
        self._interleaved_q_metrics = []
        self._interleaved_actor_metrics = []
        self._interleaved_replay_accepted = 0
        self._interleaved_reward_sum = torch.zeros((), device=self.device)
        self._interleaved_transition_count = 0
        self._last_timeout_finals_used = 0

    def uses_interleaved_updates(self):
        """Teacher FastSAC updates before the next environment action."""
        return self.cfg.phase == "train"

    @torch.no_grad()
    def _prepare_teacher_final_state(self, td: TensorDict):
        """Prepare deploy/Q inputs and raw teacher inputs for a true next state."""
        raw_values = {
            next_field: td[source_key].clone()
            for source_key, _, next_field in self._teacher_raw_replay_fields
            if next_field is not None
        }
        self.object_transform(td)
        td["_depth_feature"] = torch.zeros(
            *td.batch_size,
            self.depth_feature_dim,
            device=td.device,
            dtype=td[OBS_KEY].dtype,
        )
        self.adapt_module(td)
        result = {
            "next_observations": self._cat(td, self.q_actor_keys).clone(),
            "next_critic_observations": self._cat(td, self.q_critic_keys).clone(),
        }
        result.update(raw_values)
        return result

    @torch.no_grad()
    def _teacher_transition_from_step(
        self, current: TensorDict, rollout_carry: TensorDict
    ):
        """Build one vector-step transition before the environment resets are lost."""
        n = int(current.batch_size[0])
        next_values = self._prepare_teacher_final_state(rollout_carry.clone())
        timeouts = (current[DONE_KEY] & ~current[TERM_KEY]).reshape(n).bool()
        if timeouts.any():
            env_indices = timeouts.nonzero(as_tuple=False).squeeze(-1)
            timeout_values = self._prepare_teacher_final_state(
                current["next"][env_indices].clone()
            )
            for key, values in timeout_values.items():
                next_values[key].index_copy_(0, env_indices, values)
            self._last_timeout_finals_used += int(env_indices.numel())

        transitions = {
            "observations": self._cat(
                current, self.q_actor_keys
            ).reshape(n, self._q_actor_dim),
            "critic_observations": self._cat(
                current, self.q_critic_keys
            ).reshape(n, self._q_critic_dim),
            "actions": current[ACTION_KEY].reshape(n, self.action_dim),
            "rewards": (
                current[REWARD_KEY] * self.reward_scales
            ).sum(-1).reshape(n),
            "dones": current[DONE_KEY].reshape(n).bool(),
            "truncations": (
                current[DONE_KEY] & ~current[TERM_KEY]
            ).reshape(n).bool(),
            "discounts": current["next", "discount"].reshape(n),
            **next_values,
        }
        transitions.update({
            current_field: current[source_key]
            for source_key, current_field, _ in self._teacher_raw_replay_fields
        })
        return transitions

    @torch.no_grad()
    def capture_timeout_final_observations(self, td: TensorDict, step: int):
        if self.cfg.phase != "train":
            return

        timeouts = (td[DONE_KEY] & ~td[TERM_KEY]).reshape(-1).bool()
        if not timeouts.any():
            return
        env_indices = timeouts.nonzero(as_tuple=False).squeeze(-1)
        final_values = self._prepare_teacher_final_state(
            td["next"][env_indices].clone()
        )
        final_values["indices"] = (
            env_indices * int(self.cfg.train_every) + int(step)
        )
        self._timeout_final_batches.append(final_values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        """Retain s_(t+1) for the last row of a chunked VAIC rollout."""
        if self.cfg.phase != "train":
            return
        self._rollout_final_batch = self._prepare_teacher_final_state(carry.clone())

    def _teacher_transition_chunks(self, td: TensorDict):
        """Yield one N-row transition batch per VAIC vector-environment step.

        Building the old dense N*T transition dictionary temporarily consumed
        several GiB at 4096 environments.  Step-sized chunks also let the
        optimizer reproduce HOI's per-environment-step update cadence.
        """
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "Teacher FastSAC needs the final rollout observation. The train "
                "loop must call capture_rollout_final_observation(carry)."
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError(
                f"Rollout length {t} does not match train_every={self.cfg.train_every}."
            )
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        timeout_batches = self._timeout_final_batches
        self._timeout_final_batches = []
        timeout_finals = None
        if timeout_batches:
            timeout_finals = {
                key: torch.cat([batch[key] for batch in timeout_batches], dim=0)
                for key in timeout_batches[0]
            }
            indices = timeout_finals["indices"].long()
            if (indices < 0).any() or (indices >= n * t).any():
                raise IndexError("Captured timeout index is outside the rollout")
            self._last_timeout_finals_used = int(indices.numel())
        else:
            self._last_timeout_finals_used = 0

        for step in range(int(t)):
            current = td[:, step]
            if step + 1 < int(t):
                following = td[:, step + 1]
                next_values = {
                    "next_observations": self._cat(
                        following, self.q_actor_keys
                    ).reshape(n, self._q_actor_dim),
                    "next_critic_observations": self._cat(
                        following, self.q_critic_keys
                    ).reshape(n, self._q_critic_dim),
                }
                next_values.update({
                    next_field: following[source_key]
                    for source_key, _, next_field in self._teacher_raw_replay_fields
                    if next_field is not None
                })
            else:
                next_values = final_batch

            transitions = {
                "observations": self._cat(
                    current, self.q_actor_keys
                ).reshape(n, self._q_actor_dim),
                "critic_observations": self._cat(
                    current, self.q_critic_keys
                ).reshape(n, self._q_critic_dim),
                "actions": current[ACTION_KEY].reshape(n, self.action_dim),
                "rewards": (
                    current[REWARD_KEY] * self.reward_scales
                ).sum(-1).reshape(n),
                "dones": current[DONE_KEY].reshape(n).bool(),
                "truncations": (
                    current[DONE_KEY] & ~current[TERM_KEY]
                ).reshape(n).bool(),
                "discounts": current["next", "discount"].reshape(n),
                **next_values,
            }
            transitions.update({
                current_field: current[source_key]
                for source_key, current_field, _ in self._teacher_raw_replay_fields
            })

            if timeout_finals is not None:
                flat_indices = timeout_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    for key, values in timeout_finals.items():
                        if key == "indices":
                            continue
                        transitions[key] = transitions[key].clone()
                        transitions[key].index_copy_(
                            0, env_indices, values[selected]
                        )
            yield transitions

    def _teacher_transitions(self, td: TensorDict):
        """Materialize all rows for diagnostics/tests; training streams chunks."""
        chunks = list(self._teacher_transition_chunks(td))
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }

    def _teacher_state_from_replay(self, batch, next_state=False):
        critic_key = (
            "next_critic_observations" if next_state else "critic_observations"
        )
        critic_obs = batch[critic_key]
        values = {}
        offset = 0
        for key, width in zip(self.q_critic_keys, self._q_critic_widths):
            values[key] = critic_obs[..., offset:offset + width]
            offset += width
        if offset != self._q_critic_dim:
            raise RuntimeError("Teacher FastSAC critic observation split is inconsistent")
        for source_key, current_field, next_field in self._teacher_raw_replay_fields:
            values[source_key] = batch[
                next_field
                if next_state and next_field is not None
                else current_field
            ]
        td = TensorDict(
            values,
            batch_size=critic_obs.shape[:-1],
            device=critic_obs.device,
        )
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        self.encoder_priv(td)
        return td

    def _teacher_q_alpha_update(self, batch):
        with torch.no_grad():
            next_td = self._teacher_state_from_replay(batch, next_state=True)
            next_dist = self.actor.get_dist(next_td)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action)
            bootstrap = _timeout_bootstrap_mask(
                batch["dones"], batch["truncations"]
            )
            discount = self.cfg.gamma * batch["discounts"]
            soft_reward = (
                batch["rewards"]
                - discount
                * bootstrap
                * self.log_alpha.exp()
                * next_log_prob
            )
            target = self.qnet_target.projection(
                batch["next_critic_observations"],
                next_action,
                soft_reward,
                bootstrap,
                discount,
            )
            target_values = (target * self.qnet_target.support).sum(-1)

        logits = self.qnet(batch["critic_observations"], batch["actions"])
        per_q = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
        q_loss = per_q.sum()
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.cfg.sac_max_grad_norm > 0:
            q_grad = nn.utils.clip_grad_norm_(
                self.qnet.parameters(), self.cfg.sac_max_grad_norm
            )
        else:
            q_grad = torch.zeros((), device=self.device)
        self.opt_q.step()
        self.q_update_count += 1
        self.sac_update_count += 1

        alpha_loss = -(
            self.log_alpha.exp()
            * (next_log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.sac_alpha_update_count += 1

        return {
            "q_loss": q_loss.detach(),
            "q1_loss": per_q[0].detach(),
            "q2_loss": per_q[1].detach(),
            "q_grad_norm": q_grad.detach(),
            "alpha_loss": alpha_loss.detach(),
            "target_q_min": target_values.min().detach(),
            "target_q_max": target_values.max().detach(),
        }

    @torch.no_grad()
    def _soft_update_teacher_q_target(self):
        for source, target_param in zip(
            self.qnet.parameters(), self.qnet_target.parameters()
        ):
            target_param.lerp_(source, self.cfg.sac_tau)

    def _teacher_actor_update(self, batch):
        # Match HOI: update policy from the same replay sample as Q. Raw replay
        # fields rebuild the current VAIC encoder path, so encoder_priv and the
        # optional height CNN receive valid gradients instead of stale latents.
        minibatch = self._teacher_state_from_replay(batch, next_state=False)
        dist = self.actor.get_dist(minibatch)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        critic_obs = batch["critic_observations"]

        # Q contributes dQ/da but must not accumulate or step critic weights.
        self.opt_q.zero_grad(set_to_none=True)
        q_requires_grad = [parameter.requires_grad for parameter in self.qnet.parameters()]
        for parameter in self.qnet.parameters():
            parameter.requires_grad_(False)
        try:
            q_logits = self.qnet(critic_obs, action)
            q_value = self.qnet.values(q_logits).mean(0)
            actor_loss = (
                self.log_alpha.exp().detach() * log_prob - q_value
            ).mean()
            self.sac_teacher_actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.cfg.sac_max_grad_norm > 0:
                actor_grad = nn.utils.clip_grad_norm_(
                    self._teacher_actor_parameters,
                    self.cfg.sac_max_grad_norm,
                )
            else:
                actor_grad = torch.zeros((), device=self.device)
            self.sac_teacher_actor_optimizer.step()
        finally:
            for parameter, requires_grad in zip(
                self.qnet.parameters(), q_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
        self.sac_actor_update_count += 1
        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "entropy": -log_prob.mean().detach(),
            "action_std": dist.scale.mean().detach(),
        }

    @staticmethod
    def _mean_metric(metrics, key, device):
        if not metrics:
            return torch.zeros((), device=device)
        return torch.stack([metric[key] for metric in metrics]).mean()

    def _teacher_export_is_due(self):
        return self.env.current_iter >= int(
            self.cfg.teacher_buffer_start_iteration
        )

    def _teacher_export_counts(self):
        if (
            self.teacher_replay is None
            or not self._teacher_export_started
            or self._teacher_export_start_seen is None
        ):
            return 0, 0
        seen = max(
            int(self.teacher_replay.seen) - int(self._teacher_export_start_seen),
            0,
        )
        return min(int(self.teacher_replay.size), seen), seen

    def _teacher_replay_checkpoint_metadata(self):
        if self.cfg.phase == "train" and (
            not self.cfg.save_teacher_buffer
            or not self._teacher_export_started
            or self.teacher_replay is None
            or self.teacher_replay.last_snapshot_id is None
        ):
            return None
        return super()._teacher_replay_checkpoint_metadata()

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        if self.teacher_replay is None:
            return None
        if (
            self.cfg.phase == "train"
            and (
                not self.cfg.save_teacher_buffer
                or not self._teacher_export_started
            )
        ):
            return None
        rows, seen = self._teacher_export_counts()
        return self.teacher_replay.snapshot(
            iteration,
            checkpoint_name,
            row_count=rows,
            seen_count=seen,
        )

    def collect_environment_step(
        self, transition_td: TensorDict, rollout_carry: TensorDict
    ):
        """Insert one vector step and run HOI updates before the next action."""
        if self.cfg.phase != "train":
            raise RuntimeError(
                "Interleaved collection is only valid for teacher FastSAC."
            )
        if self.teacher_replay is None:
            raise RuntimeError("Teacher FastSAC replay is not configured")
        if not hasattr(self, "_interleaved_steps_collected"):
            raise RuntimeError(
                "begin_transition_collection() must run before collection"
            )
        if self._interleaved_steps_collected >= int(self.cfg.train_every):
            raise RuntimeError("Collected more FastSAC steps than train_every")

        if not self._teacher_export_started and self._teacher_export_is_due():
            # Preserve the learning FIFO. Only the later H5 snapshot is gated.
            self._teacher_export_started = True
            self._teacher_export_start_seen = int(self.teacher_replay.seen)

        transitions = self._teacher_transition_from_step(
            transition_td, rollout_carry
        )
        self._interleaved_replay_accepted += self.teacher_replay.append(
            transitions
        )
        self._interleaved_reward_sum += transitions["rewards"].sum()
        self._interleaved_transition_count += int(
            transitions["rewards"].numel()
        )

        logical_step = int(self.sac_environment_steps)
        updates_per_step = self._sac_updates_per_env_step()
        if logical_step > int(self.cfg.sac_learning_starts):
            for update_index in range(updates_per_step):
                batch = self.teacher_replay.sample(
                    self.cfg.sac_batch_size,
                    device=self.device,
                    generator=self.q_rng,
                    fields=self._teacher_learning_fields,
                )
                self._interleaved_q_metrics.append(
                    self._teacher_q_alpha_update(batch)
                )
                if self._sac_actor_update_is_due(
                    update_index, logical_step, updates_per_step
                ):
                    self._interleaved_actor_metrics.append(
                        self._teacher_actor_update(batch)
                    )
                # Match HOI ordering: Q -> alpha -> delayed actor -> target Q.
                self._soft_update_teacher_q_target()

        self.sac_environment_steps += 1
        self._interleaved_steps_collected += 1

    def train_op(self, tensordict):
        if self.cfg.phase != "train":
            raise RuntimeError(
                "FastSACVEL.train_op is the teacher FastSAC path; "
                "student training must use FastSACVelFinetune."
            )
        if self.teacher_replay is None:
            raise RuntimeError(
                "Teacher FastSAC replay is not configured. Construct training "
                "with configure_replay=True."
            )

        updates_per_step = self._sac_updates_per_env_step()
        rollout_td = tensordict.exclude("stats")
        interleaved_steps = int(
            getattr(self, "_interleaved_steps_collected", 0)
        )
        if interleaved_steps:
            if interleaved_steps != int(self.cfg.train_every):
                raise RuntimeError(
                    "Teacher FastSAC rollout ended after "
                    f"{interleaved_steps} interleaved steps; expected "
                    f"{self.cfg.train_every}."
                )
            accepted = self._interleaved_replay_accepted
            reward_sum = self._interleaved_reward_sum
            transition_count = self._interleaved_transition_count
            q_metrics = self._interleaved_q_metrics
            actor_metrics = self._interleaved_actor_metrics
        else:
            # Compatibility path for alternate collectors and unit tests. The
            # production train.py path updates immediately after every step.
            if not self._teacher_export_started and self._teacher_export_is_due():
                self._teacher_export_started = True
                self._teacher_export_start_seen = int(self.teacher_replay.seen)
            start_environment_step = self.sac_environment_steps
            accepted = 0
            reward_sum = torch.zeros((), device=self.device)
            transition_count = 0
            q_metrics = []
            actor_metrics = []
            for local_step, transitions in enumerate(
                self._teacher_transition_chunks(rollout_td)
            ):
                accepted += self.teacher_replay.append(transitions)
                reward_sum += transitions["rewards"].sum()
                transition_count += int(transitions["rewards"].numel())
                logical_step = start_environment_step + local_step
                if logical_step <= int(self.cfg.sac_learning_starts):
                    continue
                for update_index in range(updates_per_step):
                    batch = self.teacher_replay.sample(
                        self.cfg.sac_batch_size,
                        device=self.device,
                        generator=self.q_rng,
                        fields=self._teacher_learning_fields,
                    )
                    q_metrics.append(self._teacher_q_alpha_update(batch))
                    if self._sac_actor_update_is_due(
                        update_index, logical_step, updates_per_step
                    ):
                        actor_metrics.append(self._teacher_actor_update(batch))
                    self._soft_update_teacher_q_target()
            self.sac_environment_steps = (
                start_environment_step + int(self.cfg.train_every)
            )

        self.sac_rollout_count += 1

        # Preserve VAIC's teacher->student supervised adaptation/distillation.
        # This is independent of PPO and follows the just-updated SAC teacher.
        adapt_info = self._train_adapt_no_depth(rollout_td.copy())
        self.num_updates += 1

        h5_rows, _ = self._teacher_export_counts()
        info = {
            "fastsac/q_active": float(bool(q_metrics)),
            "fastsac/q_loss": self._mean_metric(q_metrics, "q_loss", self.device).item(),
            "fastsac/q1_loss": self._mean_metric(q_metrics, "q1_loss", self.device).item(),
            "fastsac/q2_loss": self._mean_metric(q_metrics, "q2_loss", self.device).item(),
            "fastsac/q_grad_norm": self._mean_metric(q_metrics, "q_grad_norm", self.device).item(),
            "fastsac/target_q_min": self._mean_metric(q_metrics, "target_q_min", self.device).item(),
            "fastsac/target_q_max": self._mean_metric(q_metrics, "target_q_max", self.device).item(),
            "fastsac/actor_loss": self._mean_metric(actor_metrics, "actor_loss", self.device).item(),
            "fastsac/actor_grad_norm": self._mean_metric(actor_metrics, "actor_grad_norm", self.device).item(),
            "fastsac/entropy": self._mean_metric(actor_metrics, "entropy", self.device).item(),
            "fastsac/action_std": self._mean_metric(actor_metrics, "action_std", self.device).item(),
            "fastsac/alpha_loss": self._mean_metric(q_metrics, "alpha_loss", self.device).item(),
            "fastsac/alpha": self.log_alpha.exp().item(),
            "fastsac/q_update_count": self.q_update_count,
            "fastsac/actor_update_count": self.sac_actor_update_count,
            "fastsac/alpha_update_count": self.sac_alpha_update_count,
            "fastsac/environment_steps": self.sac_environment_steps,
            "fastsac/effective_updates_per_env_step": (
                len(q_metrics) / float(self.cfg.train_every)
            ),
            "fastsac/updates_per_env_step_config": updates_per_step,
            "fastsac/buffer_reward": (
                reward_sum / max(transition_count, 1)
            ).item(),
            "fastsac/replay_accepted": accepted,
            "fastsac/replay_saved": self.teacher_replay.saved,
            "fastsac/replay_seen": self.teacher_replay.seen,
            "fastsac/replay_fill_ratio": (
                self.teacher_replay.saved / self.teacher_replay.capacity
            ),
            "fastsac/h5_export_active": float(self._teacher_export_started),
            "fastsac/h5_export_rows": (
                h5_rows
            ),
            "fastsac/h5_start_iteration": int(
                self.cfg.teacher_buffer_start_iteration
            ),
            "fastsac/timeout_finals": self._last_timeout_finals_used,
        }
        actor_scale = rollout_td["scale"].detach().reshape(
            -1, self.action_dim
        ).mean(0)
        for joint_name, std in zip(self.joint_names, actor_scale):
            info[f"actor_std/{joint_name}"] = std.item()
        info["actor_std/mean"] = actor_scale.mean().item()
        info.update(adapt_info)
        return info

    def state_dict(self):
        # Synchronize checkpoint EMA copies while saving actor/student exactly
        # as trained.
        if self.cfg.phase == "train":
            hard_copy_(self.adapt_module, self.adapt_ema)
            if self.cfg.use_object_adapt:
                hard_copy_(self.object_adapt, self.object_adapt_ema)
            if hasattr(self, "temporal_depth_gru"):
                hard_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema)

        state = OrderedDict(
            (name, module.state_dict()) for name, module in self.named_children()
        )
        state["last_phase"] = self.cfg.phase
        state["last_iter"] = self.env.current_iter
        # ``last_iter`` is the rollout already completed by this checkpoint.
        # Same-stage resume must continue at the following curriculum iteration.
        state["next_iter"] = int(self.env.current_iter) + 1
        state["q_rng_state"] = self.q_rng.get_state()
        state["q_update_count"] = self.q_update_count
        state["actor_backend"] = FASTSAC_ACTOR_BACKEND
        state["training_algorithm"] = (
            FASTSAC_TEACHER_TRAINING_ALGORITHM
            if self.cfg.phase == "train"
            else FASTSAC_STUDENT_TRAINING_ALGORITHM
        )
        state["actor_backend_config"] = self._actor_backend_metadata()
        state["q_backend_config"] = self._q_backend_metadata()
        state["teacher_replay_id"] = self.teacher_replay_id
        state["sac_environment_steps"] = self.sac_environment_steps
        state["sac_rollout_count"] = self.sac_rollout_count
        state["sac_update_count"] = self.sac_update_count
        state["sac_actor_update_count"] = self.sac_actor_update_count
        state["sac_alpha_update_count"] = self.sac_alpha_update_count
        state["teacher_export_started"] = self._teacher_export_started
        if hasattr(self, "log_alpha"):
            state["log_alpha"] = self.log_alpha.detach().clone()
        state["optimizer_resume_state"] = self._optimizer_resume_state()
        replay_metadata = self._teacher_replay_checkpoint_metadata()
        if replay_metadata is not None:
            state["teacher_replay_state"] = replay_metadata
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        allowed_algorithms = (
            {FASTSAC_TEACHER_TRAINING_ALGORITHM}
            if self.cfg.phase == "train"
            else {
                FASTSAC_TEACHER_TRAINING_ALGORITHM,
                FASTSAC_STUDENT_TRAINING_ALGORITHM,
            }
        )
        if algorithm not in allowed_algorithms:
            raise ValueError(
                "Checkpoint is not from the true FastSAC training path. Retrain "
                "with algo=fastsac_vel_train; old PPO-based FastSAC-hybrid "
                f"checkpoints are intentionally rejected (training_algorithm={algorithm!r})."
            )
        backend = state_dict.get("actor_backend")
        if backend != FASTSAC_ACTOR_BACKEND:
            raise ValueError(
                "This algorithm requires a checkpoint produced by "
                "algo=fastsac_vel_train (HOI FastSAC actor backend); got "
                f"actor_backend={backend!r}."
            )
        expected = self._actor_backend_metadata()
        actual = state_dict.get("actor_backend_config")
        if actual != expected:
            raise ValueError(
                f"FastSAC actor checkpoint config {actual} does not match {expected}"
            )
        expected_q = self._q_backend_metadata()
        actual_q = state_dict.get("q_backend_config")
        if actual_q != expected_q:
            raise ValueError(
                f"FastSAC Q checkpoint config {actual_q} does not match {expected_q}"
            )
        if not state_dict.get("teacher_replay_id"):
            raise ValueError("FastSAC checkpoint is missing its teacher replay id")
        failed = super().load_state_dict(state_dict, strict)
        critical = {
            "actor", "actor_adapt", "qnet", "qnet_target",
            "encoder_priv", "adapt_module", "adapt_ema",
        }
        if self.cfg.use_object_adapt:
            critical.update(("object_adapt", "object_adapt_ema"))
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                f"Failed to restore critical FastSAC modules: {sorted(missing)}"
            )
        if state_dict.get("last_phase") == self.cfg.phase:
            if "log_alpha" in state_dict and hasattr(self, "log_alpha"):
                self.log_alpha.data.copy_(
                    state_dict["log_alpha"].to(
                        device=self.log_alpha.device,
                        dtype=self.log_alpha.dtype,
                    )
                )
            self.sac_environment_steps = int(
                state_dict.get("sac_environment_steps", 0)
            )
            self.sac_rollout_count = int(state_dict.get("sac_rollout_count", 0))
            self.sac_update_count = int(state_dict.get("sac_update_count", 0))
            self.sac_actor_update_count = int(
                state_dict.get("sac_actor_update_count", 0)
            )
            self.sac_alpha_update_count = int(
                state_dict.get("sac_alpha_update_count", 0)
            )
            self._teacher_export_started = bool(
                state_dict.get("teacher_export_started", False)
            )
            self.env.set_progress(
                int(state_dict.get("next_iter", state_dict.get("last_iter", -1) + 1))
            )
        return failed


class FastSACVelFinetune(FastSACVEL):
    """VAIC student actor + HOI-style distributional FastSAC with RLPD replay."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.enable_residual_distillation:
            raise ValueError(
                "fastsac_vel_finetune requires enable_residual_distillation=false: "
                "SAC already optimizes actor_adapt, so a second distillation optimizer "
                "would apply a conflicting objective and independent Adam state."
            )
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if not 0.0 <= cfg.teacher_buffer_ratio <= 1.0:
            raise ValueError("teacher_buffer_ratio must be in [0, 1]")
        self.online_replay = OnlineReplay(cfg.online_buffer_capacity, device=self.device)
        self.offline_replay = None
        self.sac_actor_optimizer = torch.optim.AdamW(
            self.actor_adapt.parameters(), lr=cfg.sac_actor_lr,
            weight_decay=cfg.q_weight_decay, betas=(0.9, 0.95),
        )
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95), fused=str(device).startswith("cuda"),
        )
        self.log_alpha = nn.Parameter(torch.tensor(
            float(np.log(cfg.sac_alpha_init)), device=device, dtype=torch.float32,
        ))
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.sac_alpha_lr, betas=(0.9, 0.95))
        self.target_entropy = -self.action_dim * cfg.sac_target_entropy_ratio
        self.sac_update_count = 0

    def configure_offline_replay(self, path):
        if path is None:
            raise ValueError(
                "FastSAC finetune requires teacher_buffer_path, or a checkpoint run containing "
                "teacher_replay_buffer.h5"
            )
        self.offline_replay = OfflineReplayH5(
            path, self._q_actor_dim, self._q_critic_dim, self.action_dim,
            device=self.device, max_size=self.cfg.teacher_buffer_capacity,
            seed=self.cfg.teacher_buffer_seed,
            expected_replay_id=self.teacher_replay_id,
            expected_actor_backend=self.actor_backend,
            expected_actor_obs_keys=self.q_actor_keys,
            expected_critic_obs_keys=self.q_critic_keys,
            expected_snapshot_metadata=self._loaded_teacher_replay_metadata,
        )
        self.offline_replay_source_path = os.path.abspath(os.fspath(path))

    def get_offline_replay_path(self):
        return getattr(self, "offline_replay_source_path", None)

    def load_state_dict(self, state_dict, strict=True):
        if "qnet" not in state_dict:
            raise KeyError(
                "Checkpoint has no FastSAC Q1/Q2. Use a checkpoint trained with "
                "algo=fastsac_vel_train."
            )
        failed = super().load_state_dict(state_dict, strict)
        if "qnet" in failed or "qnet_target" in failed:
            raise RuntimeError("Failed to load teacher FastSAC Q1/Q2 from checkpoint")
        source_phase = state_dict.get("last_phase")
        if source_phase == "finetune":
            perception_modules = {"encoder_priv", "adapt_module", "adapt_ema"}
            if self.cfg.use_object_adapt:
                perception_modules.update(("object_adapt", "object_adapt_ema"))
            if hasattr(self, "temporal_depth_gru"):
                perception_modules.update(
                    ("depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema")
                )
            missing_perception = perception_modules.intersection(failed)
            if missing_perception:
                raise RuntimeError(
                    "Failed to restore student perception/EMA modules: "
                    f"{sorted(missing_perception)}"
                )
        if self.q_update_count < 1:
            raise RuntimeError(
                "Checkpoint Q1/Q2 have not been trained by FastSAC yet."
            )
        if source_phase == "train":
            replay_state = state_dict.get("teacher_replay_state")
            if replay_state is None or int(replay_state.get("size", 0)) < 1:
                raise RuntimeError(
                    "This teacher checkpoint has no eligible H5 replay snapshot. "
                    "Stage-2 RLPD requires save_teacher_buffer=true and a "
                    "checkpoint saved at or after "
                    "algo.teacher_buffer_start_iteration."
                )
            # Student SAC owns a new entropy process and update cadence. Transfer
            # teacher/student actor and Q weights, but start alpha/counters fresh.
            self.log_alpha.data.fill_(float(np.log(self.cfg.sac_alpha_init)))
            self.q_update_count = 0
            self.sac_update_count = 0
        elif source_phase == "finetune":
            replay_state = state_dict.get("offline_teacher_replay_state")
            if replay_state is None:
                raise RuntimeError(
                    "Student FastSAC checkpoint is missing its exact offline "
                    "teacher replay manifest."
                )
            self._loaded_teacher_replay_metadata = copy.deepcopy(replay_state)
        return failed

    def state_dict(self):
        state = super().state_dict()
        if self.offline_replay is not None:
            state["offline_teacher_replay_state"] = copy.deepcopy(
                self.offline_replay.snapshot_metadata
            )
        return state

    @torch.no_grad()
    def _prepare_student_final_state(self, td: TensorDict):
        """Run the unchanged VAIC EMA perception path on a true next state."""
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
            "next_observations": self._cat(td, self.q_actor_keys).clone(),
            "next_critic_observations": self._cat(
                td, self.q_critic_keys
            ).clone(),
        }

    @torch.no_grad()
    def capture_timeout_final_observations(self, td: TensorDict, step: int):
        timeouts = (td[DONE_KEY] & ~td[TERM_KEY]).reshape(-1).bool()
        if not timeouts.any():
            return
        env_indices = timeouts.nonzero(as_tuple=False).squeeze(-1)
        final_values = self._prepare_student_final_state(
            td["next"][env_indices].clone()
        )
        final_values["indices"] = (
            env_indices * int(self.cfg.train_every) + int(step)
        )
        self._timeout_final_batches.append(final_values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        self._rollout_final_batch = self._prepare_student_final_state(carry.clone())

    def _student_transition_chunks(self, td: TensorDict):
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "Student FastSAC needs the final rollout observation. The train "
                "loop must call capture_rollout_final_observation(carry)."
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError(
                f"Rollout length {t} does not match train_every={self.cfg.train_every}."
            )
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        timeout_batches = self._timeout_final_batches
        self._timeout_final_batches = []
        timeout_finals = None
        if timeout_batches:
            timeout_finals = {
                key: torch.cat([batch[key] for batch in timeout_batches], dim=0)
                for key in timeout_batches[0]
            }
            indices = timeout_finals["indices"].long()
            if (indices < 0).any() or (indices >= n * t).any():
                raise IndexError("Captured timeout index is outside the rollout")
            self._last_timeout_finals_used = int(indices.numel())
        else:
            self._last_timeout_finals_used = 0

        for step in range(int(t)):
            current = td[:, step]
            if step + 1 < int(t):
                following = td[:, step + 1]
                next_values = {
                    "next_observations": self._cat(
                        following, self.q_actor_keys
                    ).reshape(n, self._q_actor_dim),
                    "next_critic_observations": self._cat(
                        following, self.q_critic_keys
                    ).reshape(n, self._q_critic_dim),
                }
            else:
                next_values = final_batch
            transitions = {
                "observations": self._cat(
                    current, self.q_actor_keys
                ).reshape(n, self._q_actor_dim),
                "critic_observations": self._cat(
                    current, self.q_critic_keys
                ).reshape(n, self._q_critic_dim),
                "actions": current[ACTION_KEY].reshape(n, self.action_dim),
                "rewards": (
                    current[REWARD_KEY] * self.reward_scales
                ).sum(-1).reshape(n),
                "dones": current[DONE_KEY].reshape(n).bool(),
                "truncations": (
                    current[DONE_KEY] & ~current[TERM_KEY]
                ).reshape(n).bool(),
                "discounts": current["next", "discount"].reshape(n),
                **next_values,
            }
            if timeout_finals is not None:
                flat_indices = timeout_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    for key, values in timeout_finals.items():
                        if key == "indices":
                            continue
                        transitions[key] = transitions[key].clone()
                        transitions[key].index_copy_(
                            0, env_indices, values[selected]
                        )
            yield transitions

    def _actor_dist_from_flat(self, actor_obs):
        vel_dim = self.observation_spec[VEL_CMD_KEY].shape[-1]
        policy_dim = self.observation_spec[OBS_KEY].shape[-1]
        td = TensorDict({
            VEL_CMD_KEY: actor_obs[..., :vel_dim],
            OBS_KEY: actor_obs[..., vel_dim:vel_dim + policy_dim],
            PRIV_PRED_KEY: actor_obs[..., vel_dim + policy_dim:],
        }, batch_size=actor_obs.shape[:-1], device=actor_obs.device)
        return self.actor_adapt.get_dist(td)

    def _mix_batch(self):
        if self.online_replay.size < 1:
            raise RuntimeError("Cannot train student FastSAC from empty online replay")
        # Sampling is with replacement, matching HOI's global 8192-row batch
        # even before the online FIFO itself contains 8192 distinct rows.
        total = int(self.cfg.sac_batch_size)
        offline_count = round(total * self.cfg.teacher_buffer_ratio)
        online_count = total - offline_count
        parts = []
        if online_count:
            parts.append(self.online_replay.sample(online_count, self.device))
        if offline_count:
            if self.offline_replay is None:
                raise RuntimeError("Offline teacher replay was not configured")
            parts.append(self.offline_replay.sample(offline_count, self.device))
        keys = TEACHER_REPLAY_FIELDS
        mixed = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
        permutation = torch.randperm(total, device=self.device)
        return {key: value[permutation] for key, value in mixed.items()}

    def _sac_update(self, batch, update_actor):
        with torch.no_grad():
            next_dist = self._actor_dist_from_flat(batch["next_observations"])
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action)
            discount = self.cfg.gamma * batch["discounts"]
            # Match HOI: time-limit truncation bootstraps, true termination does
            # not, and entropy belongs to the discounted next-state target.
            bootstrap = _timeout_bootstrap_mask(batch["dones"], batch["truncations"])
            soft_reward = (
                batch["rewards"]
                - discount * bootstrap * self.log_alpha.exp() * next_log_prob
            )
            target = self.qnet_target.projection(
                batch["next_critic_observations"], next_action, soft_reward,
                bootstrap, discount,
            )
        logits = self.qnet(batch["critic_observations"], batch["actions"])
        per_q = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
        q_loss = per_q.sum()
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.cfg.sac_max_grad_norm > 0:
            q_grad = nn.utils.clip_grad_norm_(
                self.qnet.parameters(), self.cfg.sac_max_grad_norm
            )
        else:
            q_grad = torch.zeros((), device=self.device)
        self.opt_q.step()
        self.q_update_count += 1

        # HOI updates temperature on every Q update, independently of the
        # lower-frequency actor update.
        alpha_loss = -(
            self.log_alpha.exp() * (next_log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.sac_alpha_update_count += 1

        actor_loss = torch.zeros((), device=self.device)
        entropy = torch.zeros((), device=self.device)
        if update_actor:
            dist = self._actor_dist_from_flat(batch["observations"])
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            q_values = self.qnet.values(self.qnet(batch["critic_observations"], action)).mean(0)
            actor_loss = (self.log_alpha.exp().detach() * log_prob - q_values).mean()
            self.sac_actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.cfg.sac_max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.actor_adapt.parameters(), self.cfg.sac_max_grad_norm
                )
            self.sac_actor_optimizer.step()
            self.sac_actor_update_count += 1

            entropy = -log_prob.mean().detach()

        with torch.no_grad():
            for source, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                target_param.lerp_(source, self.cfg.sac_tau)
        return q_loss.detach(), per_q.detach(), q_grad.detach(), actor_loss.detach(), alpha_loss.detach(), entropy

    def train_op(self, tensordict):
        rollout_td = tensordict.exclude("stats")
        start_environment_step = self.sac_environment_steps
        updates_per_step = self._sac_updates_per_env_step()
        metrics = []
        for local_step, online in enumerate(
            self._student_transition_chunks(rollout_td)
        ):
            self.online_replay.extend(online)
            logical_step = start_environment_step + local_step
            if logical_step <= int(self.cfg.sac_learning_starts):
                continue
            for update_index in range(updates_per_step):
                batch = self._mix_batch()
                update_actor = self._sac_actor_update_is_due(
                    update_index, logical_step, updates_per_step
                )
                metrics.append(self._sac_update(batch, update_actor))
                self.sac_update_count += 1

        self.sac_environment_steps = (
            start_environment_step + int(self.cfg.train_every)
        )
        self.sac_rollout_count += 1

        # Keep the original VAIC student-perception learning path.  SAC owns
        # actor_adapt/Q/alpha, while train_adapt owns the depth GRU, object
        # estimator, and privileged-latent estimator and then soft-updates their
        # rollout EMA copies.  The FastSAC config disables actor distillation, so
        # these optimizers have no trainable parameters in common.
        adapt_info = self.train_adapt(rollout_td.copy())
        self.num_updates += 1

        if metrics:
            stacked = list(zip(*metrics))
            q_loss = torch.stack(stacked[0]).mean().item()
            q1_loss = torch.stack([x[0] for x in stacked[1]]).mean().item()
            q2_loss = torch.stack([x[1] for x in stacked[1]]).mean().item()
            q_grad_norm = torch.stack(stacked[2]).mean().item()
            actor_loss = torch.stack(stacked[3]).mean().item()
            alpha_loss = torch.stack(stacked[4]).mean().item()
            entropy = torch.stack(stacked[5]).mean().item()
        else:
            q_loss = q1_loss = q2_loss = q_grad_norm = 0.0
            actor_loss = alpha_loss = entropy = 0.0

        info = {
            "fastsac/q_active": float(bool(metrics)),
            "fastsac/q_loss": q_loss,
            "fastsac/q1_loss": q1_loss,
            "fastsac/q2_loss": q2_loss,
            "fastsac/q_grad_norm": q_grad_norm,
            "fastsac/actor_loss": actor_loss,
            "fastsac/alpha_loss": alpha_loss,
            "fastsac/entropy": entropy,
            "fastsac/alpha": self.log_alpha.exp().item(),
            "fastsac/q_update_count": self.q_update_count,
            "fastsac/actor_update_count": self.sac_actor_update_count,
            "fastsac/alpha_update_count": self.sac_alpha_update_count,
            "fastsac/online_replay_size": self.online_replay.size,
            "fastsac/offline_ratio": self.cfg.teacher_buffer_ratio,
            "fastsac/timeout_finals": self._last_timeout_finals_used,
            "fastsac/environment_steps": self.sac_environment_steps,
            "fastsac/effective_updates_per_env_step": (
                len(metrics) / float(self.cfg.train_every)
            ),
            "fastsac/updates_per_env_step_config": updates_per_step,
        }
        info.update(adapt_info)
        return info
