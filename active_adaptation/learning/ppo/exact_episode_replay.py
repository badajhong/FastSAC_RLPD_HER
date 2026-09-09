"""Lossless, reset-rooted raw histories for exact recurrent replay.

Replay transitions reference ``(episode_uid, episode_step)``.  This sidecar
retains the complete preceding history while any transition or live collector
still references the episode.  Appends keep separate chunks, so collecting a
long episode does not repeatedly copy its growing prefix.  Model execution and
perception-cache invalidation belong to the caller.
"""

from __future__ import annotations

import operator
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch
import numpy as np


def _copy_cpu_tensor_(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Bit-preserving CPU copy without launching an intra-op thread team.

    Thousands of modest depth copies otherwise repeatedly enlist the global
    PyTorch CPU thread pool. NumPy views add no conversion or allocation, and
    copyto only copies bytes/strides of the already matching dtype. Keep the
    original Torch path for dtypes NumPy cannot represent (e.g. bfloat16).
    These destinations are private no-grad snapshots or owned staging memory.
    """
    try:
        destination_array = destination.numpy()
        source_array = source.detach().numpy()
    except (TypeError, RuntimeError):
        destination.copy_(source)
    else:
        np.copyto(destination_array, source_array, casting="no")


def _clone_cpu_tensor(source: torch.Tensor) -> torch.Tensor:
    source = source.detach().to(device="cpu")
    result = torch.empty(source.shape, dtype=source.dtype, device="cpu")
    _copy_cpu_tensor_(result, source)
    return result


def _zero_cpu_tensor_(value: torch.Tensor) -> torch.Tensor:
    try:
        array = value.numpy()
    except (TypeError, RuntimeError):
        value.zero_()
    else:
        array.fill(0)
    return value


@dataclass
class _Episode:
    length: int = 0
    starts: list[int] = field(default_factory=list)
    chunks: list[dict[str, torch.Tensor]] = field(default_factory=list)


class ExactEpisodePrefixStore:
    """Mutable CPU episode histories with exact overlap validation.

    All fields are time-major tensors and keep their original dtype.  Returned
    slices are independent ordinary tensors, even when accessed from inference
    mode.  An evicted episode UID cannot be reused or silently reconstructed
    from a partial replay tail.
    """

    def __init__(self, is_init_key: str):
        if not isinstance(is_init_key, str) or not is_init_key:
            raise ValueError("is_init_key must be a non-empty string")
        self.is_init_key = is_init_key
        self._next_episode_uid = 0
        self._pending_uids: set[int] = set()
        self._episodes: dict[int, _Episode] = {}
        self._field_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] | None = None
        self._node_count = 0

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def node_count(self) -> int:
        return self._node_count

    def allocate_episode_uid(self) -> int:
        uid = self._next_episode_uid
        self._next_episode_uid += 1
        self._pending_uids.add(uid)
        return uid

    def _episode(self, uid: int) -> _Episode:
        uid = operator.index(uid)
        try:
            return self._episodes[uid]
        except KeyError:
            raise RuntimeError(
                f"Exact replay episode {uid} is missing or evicted; "
                "its complete reset-rooted history is required"
            ) from None

    def length(self, uid: int) -> int:
        return self._episode(uid).length

    def append(
        self,
        uid: int,
        start_step: int,
        fields: Mapping[str, torch.Tensor],
    ) -> None:
        """Append contiguous nodes, accepting only exactly matching overlap.

        ``start_step`` is the episode-relative index of the first input node.
        It can precede the current end, in which case every overlapping field
        must match via ``torch.equal``.  Validation failures leave the store
        unchanged.  Only new suffix nodes become additional storage.
        """

        uid = operator.index(uid)
        start_step = operator.index(start_step)
        episode = self._episodes.get(uid)
        if episode is None and uid not in self._pending_uids:
            raise RuntimeError(
                f"Exact replay episode {uid} was not allocated or was evicted; "
                "episode UIDs cannot be reused"
            )
        current_length = 0 if episode is None else episode.length
        if start_step < 0:
            raise ValueError("Exact replay start_step must be non-negative")
        if start_step > current_length:
            raise RuntimeError(
                f"Exact replay episode {uid} has a history gap: "
                f"append starts at {start_step}, available prefix ends at {current_length}"
            )
        if not fields:
            raise ValueError("Exact replay raw fields cannot be empty")
        if self.is_init_key not in fields:
            raise ValueError(f"Exact replay raw fields lack {self.is_init_key!r}")

        specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        count: int | None = None
        for key, value in fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Exact replay raw field {key!r} must be a tensor")
            if value.ndim == 0:
                raise ValueError(f"Exact replay raw field {key!r} must be time-major")
            if count is None:
                count = int(value.shape[0])
            if int(value.shape[0]) != count:
                raise ValueError("Exact replay raw fields have different time lengths")
            specs[key] = (value.dtype, tuple(value.shape[1:]))
        if not count:
            raise ValueError("Exact replay append cannot be empty")
        if self._field_specs is not None and specs != self._field_specs:
            raise ValueError("Exact replay raw field names, dtype or trailing shape changed")

        # Explicitly disabling inference mode is necessary: collection often
        # calls this method from inference_mode(), but these tensors can later
        # become inputs to a gradient-enabled perception/Actor computation.
        with torch.inference_mode(False), torch.no_grad():
            prepared = {
                key: _clone_cpu_tensor(value)
                for key, value in fields.items()
            }
            reset = prepared[self.is_init_key]
            if reset.numel() == 0:
                raise ValueError("Exact replay reset field cannot have empty trailing dimensions")
            reset = reset.reshape(count, -1).bool().any(dim=-1)
            if start_step == 0:
                if not bool(reset[0]):
                    raise RuntimeError(
                        f"Exact replay episode {uid} must start at is_init=True"
                    )
                internal_reset = reset[1:]
            else:
                internal_reset = reset
            if bool(internal_reset.any()):
                raise RuntimeError(f"Exact replay episode {uid} contains an internal reset")

            overlap = min(count, current_length - start_step)
            if overlap:
                assert episode is not None
                for chunk, offset, relative_start, relative_stop in self._pieces(
                    episode, start_step, start_step + overlap
                ):
                    for key, value in prepared.items():
                        incoming = value[offset : offset + relative_stop - relative_start]
                        existing = chunk[key][relative_start:relative_stop]
                        if not torch.equal(existing, incoming):
                            raise RuntimeError(
                                f"Exact replay episode {uid} overlap mismatch in field {key!r} "
                                f"at episode steps {start_step + offset}:"
                                f"{start_step + offset + relative_stop - relative_start}"
                            )
            new_count = count - overlap
            if new_count == 0:
                return
            if overlap:
                # Clone rather than keep a view that owns duplicated overlap.
                prepared = {key: _clone_cpu_tensor(value[overlap:]) for key, value in prepared.items()}

        if episode is None:
            episode = _Episode()
            self._episodes[uid] = episode
            self._pending_uids.remove(uid)
        if self._field_specs is None:
            self._field_specs = specs
        episode.starts.append(current_length)
        episode.chunks.append(prepared)
        episode.length += new_count
        self._node_count += new_count

    @staticmethod
    def _pieces(episode: _Episode, start: int, stop: int):
        """Yield only chunks intersecting the requested half-open interval."""

        if start == stop:
            return
        index = bisect_right(episode.starts, start) - 1
        while index < len(episode.chunks):
            chunk_start = episode.starts[index]
            if chunk_start >= stop:
                break
            chunk_stop = (
                episode.starts[index + 1]
                if index + 1 < len(episode.starts)
                else episode.length
            )
            left = max(start, chunk_start)
            right = min(stop, chunk_stop)
            yield episode.chunks[index], left - start, left - chunk_start, right - chunk_start
            index += 1

    def prefix(self, uid: int, stop: int) -> dict[str, torch.Tensor]:
        """Return contiguous CPU tensors for episode nodes ``[0, stop)``."""

        return self.slice(uid, 0, stop)

    def slice(self, uid: int, start: int, stop: int) -> dict[str, torch.Tensor]:
        """Copy ``[start, stop)`` without materializing unrelated history."""

        episode = self._episode(uid)
        start, stop = operator.index(start), operator.index(stop)
        if start < 0 or stop < start:
            raise ValueError("Exact replay slice requires 0 <= start <= stop")
        if stop > episode.length:
            raise RuntimeError(
                f"Exact replay episode {uid} lacks requested history through step {stop - 1}; "
                f"available prefix length is {episode.length}"
            )
        assert self._field_specs is not None
        with torch.inference_mode(False), torch.no_grad():
            result = {
                key: torch.empty((stop - start, *shape), dtype=dtype, device="cpu")
                for key, (dtype, shape) in self._field_specs.items()
            }
            for chunk, offset, left, right in self._pieces(episode, start, stop):
                for key, value in chunk.items():
                    _copy_cpu_tensor_(result[key][offset : offset + right - left], value[left:right])
        return result

    def gather_field(self, key: str, refs: torch.Tensor) -> torch.Tensor:
        """Copy one field at ordered ``(episode_uid, step)`` references.

        Actor supervision needs only object-geometry IDs. Selecting that field
        directly avoids copying depth images or materializing episode prefixes.
        Duplicate references preserve their positions in the returned batch.
        """
        if refs.ndim != 2 or refs.shape[-1] != 2 or refs.dtype not in (
            torch.int32, torch.int64,
        ):
            raise ValueError("Exact replay field references must be an integer [N, 2] tensor")
        if self._field_specs is None or key not in self._field_specs:
            raise KeyError(f"Exact replay raw history lacks field {key!r}")
        dtype, shape = self._field_specs[key]
        groups: dict[tuple[int, int], tuple[torch.Tensor, list[int], list[int]]] = {}
        for row, (uid, step) in enumerate(refs.detach().cpu().tolist()):
            episode = self._episode(uid)
            if not 0 <= step < episode.length:
                raise IndexError("Exact replay field reference is outside its episode")
            chunk_index = bisect_right(episode.starts, step) - 1
            group = groups.setdefault(
                (uid, chunk_index), (episode.chunks[chunk_index][key], [], [])
            )
            group[1].append(row)
            group[2].append(step - episode.starts[chunk_index])
        with torch.inference_mode(False), torch.no_grad():
            result = torch.empty((refs.shape[0], *shape), dtype=dtype, device="cpu")
            for source, rows, offsets in groups.values():
                result.index_copy_(
                    0, torch.tensor(rows, dtype=torch.long),
                    source.index_select(0, torch.tensor(offsets, dtype=torch.long)),
                )
        return result

    def batch_slice(
        self,
        intervals: Iterable[tuple[int, int, int]],
        *,
        buffers: dict[str, torch.Tensor] | None = None,
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Gather complete intervals directly into one padded CPU batch.

        Padding is trailing only. The caller must use each interval's valid
        length when selecting recurrent states. Optional flat buffers bound
        staging allocations; callers must finish any asynchronous transfer
        before reusing them. No per-episode slice/stack copies are needed.
        """
        intervals = [tuple(map(operator.index, interval)) for interval in intervals]
        if not intervals:
            raise ValueError("Exact replay batch cannot be empty")
        episodes = []
        for uid, start, stop in intervals:
            episode = self._episode(uid)
            if not 0 <= start < stop <= episode.length:
                raise RuntimeError("Exact replay batch interval lacks valid complete history")
            episodes.append(episode)
        width = max(stop - start for _, start, stop in intervals)
        buffers = {} if buffers is None else buffers
        assert self._field_specs is not None
        with torch.inference_mode(False), torch.no_grad():
            result = {}
            for key, (dtype, tail) in self._field_specs.items():
                shape = (len(intervals), width, *tail)
                count = 1
                for size in shape:
                    count *= size
                buffer = buffers.get(key)
                if buffer is not None and (
                    buffer.dtype != dtype or buffer.device.type != "cpu"
                    or buffer.ndim != 1 or not buffer.is_contiguous() or buffer.requires_grad
                    or (pin_memory and not buffer.is_pinned())
                ):
                    raise ValueError(f"Exact replay staging buffer contract changed for {key!r}")
                if buffer is None or buffer.numel() < count:
                    buffer = torch.empty(count, dtype=dtype, device="cpu", pin_memory=pin_memory)
                    buffers[key] = buffer
                result[key] = _zero_cpu_tensor_(buffer[:count].view(shape))
            for row, (episode, (_, start, stop)) in enumerate(zip(episodes, intervals)):
                for chunk, offset, left, right in self._pieces(episode, start, stop):
                    for key, value in chunk.items():
                        _copy_cpu_tensor_(result[key][row, offset:offset + right - left], value[left:right])
        return result

    def retain(self, uids: Iterable[int]) -> None:
        """Keep whole episodes referenced by replay rows or live collectors.

        Pending allocated UIDs are valid live references, but unknown or
        previously evicted UIDs fail before any episode is removed.
        """

        retained = {operator.index(uid) for uid in uids}
        missing = retained.difference(self._episodes).difference(self._pending_uids)
        if missing:
            raise RuntimeError(f"Exact replay cannot retain missing or evicted episodes {sorted(missing)}")
        for uid in tuple(self._episodes):
            if uid not in retained:
                self._node_count -= self._episodes.pop(uid).length
        self._pending_uids.intersection_update(retained)
