"""Sequence-native raw replay support for successful Teacher episodes.

The Teacher transition FIFO owns the sampling population.  This module owns a
sidecar of complete raw episodes, keyed by the ``episode_uid`` and
``episode_step`` stored in each FIFO row.  Keeping the keys in the FIFO makes a
capacity-crossing write unambiguous: after the FIFO is frozen, the sidecar
retains the *whole* episode for every UID still referenced by at least one row.

No model code lives here.  A policy streams :meth:`iter_sequence_chunks`
through its current EMA perception modules, writes one flat Actor observation
per raw state, and atomically publishes the finished tensor to
:class:`CurrentEMATeacherActorCache`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

import torch


class TeacherBoundaryCause(str, Enum):
    """Mutually exclusive end causes, in the collector's required precedence."""

    SUCCESS_COMMAND = "success_command"
    TERMINATED = "terminated"
    TIME_LIMIT = "time_limit"
    OTHER_DONE = "other_done"
    RESET_INCOMPLETE = "reset_incomplete"


def classify_teacher_boundary(
    *,
    done: bool,
    terminated: bool,
    command_finished: bool,
    time_limit: bool,
) -> TeacherBoundaryCause | None:
    """Classify one event without confusing a successful command with timeout.

    Physical termination wins over every other flag.  A command completion is
    successful even when the environment also reports its time-limit flag,
    matching the existing successful-only Teacher prefill contract.
    """

    if not bool(done):
        return None
    if bool(terminated):
        return TeacherBoundaryCause.TERMINATED
    if bool(command_finished):
        return TeacherBoundaryCause.SUCCESS_COMMAND
    if bool(time_limit):
        return TeacherBoundaryCause.TIME_LIMIT
    return TeacherBoundaryCause.OTHER_DONE


@dataclass(frozen=True)
class SequenceStoreLineage:
    store_id: int
    generation: int


@dataclass(frozen=True)
class PackedSequenceChunk:
    """One time chunk from one bounded episode bucket.

    ``batch_positions`` are stable within ``group_id`` and let the caller carry
    recurrent states across chunks even as shorter episodes leave the active
    set.  Invalid padded positions have ``flat_node_indices == -1``.
    """

    group_id: int
    group_size: int
    start_step: int
    batch_positions: torch.Tensor
    episode_uids: torch.Tensor
    valid: torch.Tensor
    flat_node_indices: torch.Tensor
    raw_fields: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TeacherActorCacheLineage:
    raw_store_id: int
    raw_generation: int
    ema_generation: int
    vecnorm_fingerprint: str
    object_geo_fingerprint: str
    encoder_semantics: str


@dataclass
class _RawEpisode:
    uid: int
    length: int
    fields: dict[str, torch.Tensor]


_STORE_IDS = itertools.count()


def _cpu_int64_1d(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.long or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional int64 tensor")
    return value.detach().to(device="cpu").contiguous()


class TeacherEpisodeSequenceStore:
    """Complete successful raw episodes referenced by a frozen Teacher FIFO."""

    def __init__(self, *, is_init_key: str):
        if not isinstance(is_init_key, str) or not is_init_key:
            raise ValueError("is_init_key must be a non-empty string")
        self.is_init_key = is_init_key
        self.store_id = next(_STORE_IDS)
        self._generation = 0
        self._next_episode_uid = 0
        self._issued_uids: set[int] = set()
        self._consumed_uids: set[int] = set()
        self._episodes: dict[int, _RawEpisode] = {}
        self._field_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] | None = None
        self._frozen = False
        self._flat_fields: dict[str, torch.Tensor] = {}
        self._flat_episode_uids = torch.empty(0, dtype=torch.long)
        self._episode_uids = torch.empty(0, dtype=torch.long)
        self._episode_offsets = torch.empty(0, dtype=torch.long)
        self._episode_lengths = torch.empty(0, dtype=torch.long)
        self._uid_to_offset = torch.empty(0, dtype=torch.long)
        self._uid_to_length = torch.empty(0, dtype=torch.long)

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def episode_count(self) -> int:
        return int(self._episode_uids.numel()) if self._frozen else len(self._episodes)

    @property
    def node_count(self) -> int:
        if self._frozen:
            return int(self._flat_episode_uids.numel())
        return sum(episode.length for episode in self._episodes.values())

    @property
    def raw_fields(self) -> Mapping[str, torch.Tensor]:
        if not self._frozen:
            raise RuntimeError("Teacher raw episodes must be frozen before flat access")
        return self._flat_fields

    @property
    def lineage(self) -> SequenceStoreLineage:
        if not self._frozen:
            raise RuntimeError("Teacher raw episodes must be frozen before caching")
        return SequenceStoreLineage(self.store_id, self._generation)

    def allocate_episode_uid(self) -> int:
        if self._frozen:
            raise RuntimeError("Cannot allocate an episode UID after freeze")
        uid = self._next_episode_uid
        self._next_episode_uid += 1
        self._issued_uids.add(uid)
        return uid

    def _prepare_raw_fields(
        self, raw_fields: Mapping[str, torch.Tensor]
    ) -> tuple[int, dict[str, torch.Tensor], dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        if not raw_fields:
            raise ValueError("A Teacher episode must contain raw fields")
        if self.is_init_key not in raw_fields:
            raise KeyError(f"Teacher episode lacks {self.is_init_key!r}")
        keys = set(raw_fields)
        if self._field_specs is not None and keys != set(self._field_specs):
            raise KeyError("Teacher raw field set changed after the first commit")

        length = int(next(iter(raw_fields.values())).shape[0])
        if length < 1:
            raise ValueError("A Teacher episode cannot be empty")
        prepared: dict[str, torch.Tensor] = {}
        specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        for key, value in raw_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Teacher raw field {key!r} must be a tensor")
            if int(value.shape[0]) != length:
                raise ValueError("Teacher raw episode fields are misaligned")
            cpu = value.detach().to(device="cpu").contiguous().clone()
            prepared[key] = cpu
            specs[key] = (cpu.dtype, tuple(cpu.shape[1:]))
        if self._field_specs is not None and specs != self._field_specs:
            raise ValueError("Teacher raw field dtype or trailing shape changed")

        is_init = prepared[self.is_init_key].reshape(length, -1).bool().any(dim=-1)
        if not bool(is_init[0]):
            raise ValueError("A committed Teacher episode must start at is_init")
        if length > 1 and bool(is_init[1:].any()):
            raise ValueError("A committed Teacher episode contains an internal reset")
        return length, prepared, specs

    def commit_successful_episode(
        self,
        episode_uid: int,
        raw_fields: Mapping[str, torch.Tensor],
        *,
        boundary_cause: TeacherBoundaryCause = TeacherBoundaryCause.SUCCESS_COMMAND,
    ) -> None:
        """Commit a complete raw episode before its keyed rows enter the FIFO.

        The caller may use :meth:`rollback_episode` if the subsequent FIFO
        extend fails.  All validation and tensor allocation happen before this
        method publishes anything, so validation failures are atomic.
        """

        if self._frozen:
            raise RuntimeError("Teacher episode store is immutable after freeze")
        if boundary_cause is not TeacherBoundaryCause.SUCCESS_COMMAND:
            raise ValueError("Only command-successful Teacher episodes may be stored")
        uid = int(episode_uid)
        if uid not in self._issued_uids:
            raise ValueError("Teacher episode UID was not allocated by this store")
        if uid in self._consumed_uids:
            raise ValueError("Teacher episode UID has already been consumed")

        length, prepared, specs = self._prepare_raw_fields(raw_fields)
        episode = _RawEpisode(uid=uid, length=length, fields=prepared)
        if self._field_specs is None:
            self._field_specs = specs
        self._episodes[uid] = episode
        self._consumed_uids.add(uid)
        self._generation += 1

    def rollback_episode(self, episode_uid: int) -> None:
        """Remove an unpublished prefill commit after a failed FIFO extend."""

        if self._frozen:
            raise RuntimeError("Cannot roll back a frozen Teacher episode store")
        uid = int(episode_uid)
        if uid not in self._episodes:
            raise KeyError(f"Unknown committed Teacher episode UID {uid}")
        del self._episodes[uid]
        # The UID deliberately remains issued and can never be reused.
        self._generation += 1

    def freeze(
        self,
        live_episode_uids: torch.Tensor,
        live_episode_steps: torch.Tensor,
    ) -> None:
        """Freeze and compact exactly the complete episodes referenced by FIFO rows."""

        if self._frozen:
            raise RuntimeError("Teacher episode store is already frozen")
        uids = _cpu_int64_1d(live_episode_uids, "live_episode_uids")
        steps = _cpu_int64_1d(live_episode_steps, "live_episode_steps")
        if uids.shape != steps.shape:
            raise ValueError("Teacher FIFO episode UID and step fields are misaligned")
        if uids.numel() < 1:
            raise ValueError("Cannot freeze a Teacher store without live FIFO rows")
        if bool((uids < 0).any()) or bool((steps < 0).any()):
            raise ValueError("Teacher FIFO episode keys must be non-negative")

        live_uids = uids.unique(sorted=True)
        for uid in live_uids.tolist():
            if int(uid) not in self._episodes:
                raise KeyError(f"Teacher FIFO references unknown episode UID {uid}")
        for uid in live_uids.tolist():
            selected = uids == int(uid)
            episode_steps = steps[selected]
            length = self._episodes[int(uid)].length
            if bool((episode_steps >= length).any()):
                raise IndexError(
                    f"Teacher FIFO step exceeds episode UID {uid} length {length}"
                )

        retained = [self._episodes[int(uid)] for uid in live_uids.tolist()]
        field_names = tuple(retained[0].fields)
        flat_fields = {
            key: torch.cat([episode.fields[key] for episode in retained], dim=0)
            for key in field_names
        }
        lengths = torch.tensor([episode.length for episode in retained], dtype=torch.long)
        offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]), dim=0
        )
        flat_uids = torch.repeat_interleave(live_uids, lengths)
        max_uid = int(live_uids[-1])
        uid_to_offset = torch.full((max_uid + 1,), -1, dtype=torch.long)
        uid_to_length = torch.full((max_uid + 1,), -1, dtype=torch.long)
        uid_to_offset[live_uids] = offsets
        uid_to_length[live_uids] = lengths

        # Publish only after every validation/allocation above succeeded.
        # The compact tensors are now authoritative.  Releasing the per-episode
        # tensors avoids retaining a second ~full-buffer copy after freeze.
        self._episodes = {}
        self._flat_fields = flat_fields
        self._flat_episode_uids = flat_uids
        self._episode_uids = live_uids
        self._episode_offsets = offsets
        self._episode_lengths = lengths
        self._uid_to_offset = uid_to_offset
        self._uid_to_length = uid_to_length
        self._frozen = True
        self._generation += 1

    def resolve_node_indices(
        self,
        episode_uids: torch.Tensor,
        episode_steps: torch.Tensor,
        *,
        next_state: bool = False,
    ) -> torch.Tensor:
        """Resolve replay row keys to compact raw-node indices.

        A successful command has no in-episode successor node.  For dense
        cache lookup only, its padded ``next`` index aliases the final node
        rather than an autoreset observation from another episode.  Learning
        must pair that placeholder with zero continuation; it is not a
        Bellman self-loop.
        """

        if not self._frozen:
            raise RuntimeError("Teacher episode store must be frozen before lookup")
        uids = _cpu_int64_1d(episode_uids, "episode_uids")
        steps = _cpu_int64_1d(episode_steps, "episode_steps")
        if uids.shape != steps.shape:
            raise ValueError("Teacher episode UID and step lookups are misaligned")
        if uids.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if bool((uids < 0).any()) or int(uids.max()) >= int(self._uid_to_offset.numel()):
            raise KeyError("Teacher cache lookup contains an unknown episode UID")
        offsets = self._uid_to_offset.index_select(0, uids)
        lengths = self._uid_to_length.index_select(0, uids)
        if bool((offsets < 0).any()):
            raise KeyError("Teacher cache lookup references an evicted episode UID")
        if bool((steps < 0).any()) or bool((steps >= lengths).any()):
            raise IndexError("Teacher cache lookup contains an invalid episode step")
        resolved_steps = torch.minimum(steps + int(bool(next_state)), lengths - 1)
        return offsets + resolved_steps

    def iter_sequence_chunks(
        self,
        *,
        episode_batch_size: int,
        time_chunk_size: int,
        raw_fields: Mapping[str, torch.Tensor] | None = None,
    ) -> Iterator[PackedSequenceChunk]:
        """Yield 2-D episode/time chunks for recurrent streaming with hx carry.

        ``raw_fields`` may be a device mirror of :attr:`raw_fields`.  The CPU
        tensors remain the authoritative episode store, while a frozen CUDA
        mirror avoids retransferring hundreds of MiB for every EMA refresh.
        """

        if not self._frozen:
            raise RuntimeError("Teacher episode store must be frozen before streaming")
        episode_batch_size = int(episode_batch_size)
        time_chunk_size = int(time_chunk_size)
        if episode_batch_size < 1 or time_chunk_size < 1:
            raise ValueError("Episode and time chunk sizes must be positive")

        source_fields = self._flat_fields if raw_fields is None else raw_fields
        if set(source_fields) != set(self._flat_fields):
            raise KeyError("Teacher stream raw-field mirror has the wrong schema")
        source_device: torch.device | None = None
        for key, source in source_fields.items():
            authority = self._flat_fields[key]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"Teacher stream raw field {key!r} must be a tensor")
            if source.dtype != authority.dtype or source.shape != authority.shape:
                raise ValueError(
                    f"Teacher stream raw field {key!r} does not match its authority"
                )
            if not source.is_contiguous():
                raise ValueError(
                    f"Teacher stream raw field {key!r} must be contiguous"
                )
            if source_device is None:
                source_device = source.device
            elif source.device != source_device:
                raise ValueError("Teacher stream raw fields span multiple devices")
        if source_device is None:  # pragma: no cover - freeze forbids no fields
            raise RuntimeError("Teacher stream has no raw fields")

        episode_count = int(self._episode_uids.numel())
        time_offsets = torch.arange(time_chunk_size, dtype=torch.long)
        for group_id, group_start in enumerate(
            range(0, episode_count, episode_batch_size)
        ):
            group_stop = min(group_start + episode_batch_size, episode_count)
            group_uids = self._episode_uids[group_start:group_stop]
            group_offsets = self._episode_offsets[group_start:group_stop]
            group_lengths = self._episode_lengths[group_start:group_stop]
            group_size = group_stop - group_start
            maximum_length = int(group_lengths.max())
            for start in range(0, maximum_length, time_chunk_size):
                active = (group_lengths > start).nonzero(as_tuple=False).squeeze(-1)
                lengths = group_lengths.index_select(0, active)
                offsets = group_offsets.index_select(0, active)
                local = start + time_offsets.unsqueeze(0).expand(active.numel(), -1)
                valid = local < lengths.unsqueeze(-1)
                safe_local = torch.minimum(local, (lengths - 1).unsqueeze(-1))
                flat = offsets.unsqueeze(-1) + safe_local
                source_flat = flat.reshape(-1).to(source_device)
                source_valid = valid.to(source_device)
                gathered = {}
                for key, value in source_fields.items():
                    chunk = value.index_select(0, source_flat).reshape(
                        active.numel(), time_chunk_size, *value.shape[1:]
                    )
                    mask = source_valid
                    while mask.ndim < chunk.ndim:
                        mask = mask.unsqueeze(-1)
                    gathered[key] = torch.where(mask, chunk, torch.zeros_like(chunk))
                yield PackedSequenceChunk(
                    group_id=group_id,
                    group_size=group_size,
                    start_step=start,
                    batch_positions=active,
                    episode_uids=group_uids.index_select(0, active),
                    valid=valid,
                    flat_node_indices=torch.where(
                        valid, flat, torch.full_like(flat, -1)
                    ),
                    raw_fields=gathered,
                )


class CurrentEMATeacherActorCache:
    """Ephemeral full Actor-input cache for one EMA/VecNorm/raw lineage."""

    def __init__(self, actor_dim: int):
        actor_dim = int(actor_dim)
        if actor_dim < 1:
            raise ValueError("actor_dim must be positive")
        self.actor_dim = actor_dim
        self._lineage: TeacherActorCacheLineage | None = None
        self._actor_by_node: torch.Tensor | None = None

    @property
    def lineage(self) -> TeacherActorCacheLineage | None:
        return self._lineage

    @property
    def ready(self) -> bool:
        return self._actor_by_node is not None

    def allocate_build_tensor(
        self,
        store: TeacherEpisodeSequenceStore,
        *,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> torch.Tensor:
        if not store.frozen:
            raise RuntimeError("Cannot build an Actor cache from a mutable raw store")
        output_device = torch.device(device)
        if bool(pin_memory) and output_device.type != "cpu":
            raise ValueError("Only a CPU Teacher Actor cache can be pinned")
        return torch.empty(
            (store.node_count, self.actor_dim),
            dtype=torch.float32,
            device=output_device,
            pin_memory=bool(pin_memory),
        )

    def publish(
        self,
        lineage: TeacherActorCacheLineage,
        store: TeacherEpisodeSequenceStore,
        actor_by_node: torch.Tensor,
    ) -> None:
        """Atomically replace the cache after a complete finite build.

        Ownership of the tensor is transferred to the cache; callers must not
        mutate it after publication.
        """

        expected = store.lineage
        if (
            lineage.raw_store_id != expected.store_id
            or lineage.raw_generation != expected.generation
        ):
            raise ValueError("Teacher Actor cache lineage does not match raw store")
        if not isinstance(actor_by_node, torch.Tensor):
            raise TypeError("actor_by_node must be a tensor")
        if actor_by_node.dtype != torch.float32:
            raise TypeError("Teacher Actor cache must be float32")
        if tuple(actor_by_node.shape) != (store.node_count, self.actor_dim):
            raise ValueError("Teacher Actor cache has the wrong shape")
        if not actor_by_node.is_contiguous():
            raise ValueError("Teacher Actor cache must be contiguous")
        if not bool(torch.isfinite(actor_by_node).all()):
            raise ValueError("Teacher Actor cache contains non-finite values")
        self._actor_by_node = actor_by_node.detach()
        self._lineage = lineage

    def invalidate(self) -> None:
        self._lineage = None
        self._actor_by_node = None

    def gather(
        self,
        store: TeacherEpisodeSequenceStore,
        episode_uids: torch.Tensor,
        episode_steps: torch.Tensor,
        *,
        lineage: TeacherActorCacheLineage,
        next_state: bool,
        output_device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if self._actor_by_node is None or self._lineage != lineage:
            raise RuntimeError("Teacher Actor cache is missing or stale")
        expected = store.lineage
        if (
            lineage.raw_store_id != expected.store_id
            or lineage.raw_generation != expected.generation
        ):
            raise RuntimeError("Teacher Actor cache raw lineage is stale")
        indices = store.resolve_node_indices(
            episode_uids, episode_steps, next_state=next_state
        ).to(device=self._actor_by_node.device)
        value = self._actor_by_node.index_select(0, indices)
        if output_device is not None:
            value = value.to(output_device)
        return value


__all__ = [
    "CurrentEMATeacherActorCache",
    "PackedSequenceChunk",
    "SequenceStoreLineage",
    "TeacherActorCacheLineage",
    "TeacherBoundaryCause",
    "TeacherEpisodeSequenceStore",
    "classify_teacher_boundary",
]
