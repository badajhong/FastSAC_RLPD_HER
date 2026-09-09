"""Exact-history pruning must preserve replay provenance and cache identity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from active_adaptation.learning.ppo.exact_online_perception import (
    EXACT_CURRENT_REF,
    EXACT_NEXT_REF,
    ExactOnlinePerceptionReplayMixin,
    _EncodedPrefix,
)


class _PruneHarness(ExactOnlinePerceptionReplayMixin):
    def __init__(self, *, first_uid=0):
        self._ensure_exact_online_state()
        self._exact_online_store._next_episode_uid = first_uid
        self.uids = []
        for _ in range(5):
            uid = self._exact_online_store.allocate_episode_uid()
            self._exact_online_store.append(uid, 0, {
                "is_init": torch.tensor([[True], [False], [False]]),
                "payload": torch.tensor([[uid], [uid], [uid]], dtype=torch.long),
            })
            self._exact_online_prefixes[uid] = _EncodedPrefix(
                length=3, actor_chunks=[torch.tensor([[uid]], dtype=torch.long)],
                depth_hx=torch.tensor([float(uid % 13)]),
                adapt_hx=torch.tensor([float(uid % 17)]),
            )
            self.uids.append(uid)
        self.dagger_replay = SimpleNamespace(size=0, data={})
        self.student_replay = SimpleNamespace(size=0, data={})
        self._exact_online_actor_bank = torch.ones(2, 3)
        self._exact_online_actor_offsets = {uid: index for index, uid in enumerate(self.uids)}
        self._exact_online_generation = (17, "vecnorm", "geometry")
        self._exact_online_live_generation = self._exact_online_generation
        self._exact_online_encoded_nodes = 15
        self._exact_online_cache_hits = 7


def _original_torch_prune(harness):
    """Frozen pre-optimization production semantics, independent of NumPy."""
    referenced = []
    for replay in (harness.dagger_replay, harness.student_replay):
        for key in (EXACT_CURRENT_REF, EXACT_NEXT_REF):
            if replay.size:
                referenced.append(replay.data[key][:replay.size, 0].cpu())
    if harness._exact_online_carry_refs is not None:
        referenced.append(harness._exact_online_carry_refs[:, 0])
    keep = set(torch.cat(referenced).unique().tolist()) if referenced else set()
    harness._exact_online_store.retain(keep)
    if set(harness._exact_online_prefixes).difference(keep):
        harness._exact_online_actor_bank = None
    harness._exact_online_prefixes = {
        uid: value for uid, value in harness._exact_online_prefixes.items() if uid in keep
    }


@pytest.mark.parametrize("first_uid", [0, 2**60 + 1, 2**63 - 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prune_matches_torch_union_without_changing_retained_objects(first_uid, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA reference-transfer check requires an available GPU")
    harness = _PruneHarness(first_uid=first_uid)
    a, b, c, d, expired = harness.uids
    harness.dagger_replay = SimpleNamespace(size=3, data={
        EXACT_CURRENT_REF: torch.tensor([[a, 0], [a, 1], [b, 0], [expired, 0]], device=device),
        EXACT_NEXT_REF: torch.tensor([[a, 1], [b, 0], [c, 0], [expired, 1]], device=device),
    })
    harness.student_replay = SimpleNamespace(size=1, data={
        EXACT_CURRENT_REF: torch.tensor([[c, 0], [expired, 0]], device=device),
        EXACT_NEXT_REF: torch.tensor([[c, 1], [expired, 1]], device=device),
    })
    harness._exact_online_carry_refs = torch.tensor([[d, 2]])
    reference_tensors = [
        value for replay in (harness.dagger_replay, harness.student_replay)
        for value in replay.data.values()
    ] + [harness._exact_online_carry_refs]
    before_refs = [value.clone() for value in reference_tensors]
    before_pointers = [value.data_ptr() for value in reference_tensors]
    episodes = dict(harness._exact_online_store._episodes)
    prefixes = dict(harness._exact_online_prefixes)
    offsets = harness._exact_online_actor_offsets
    generation = harness._exact_online_generation

    _original_torch_prune(harness)
    expected_ids = set(harness._exact_online_store._episodes)
    expected_nodes = harness._exact_online_store.node_count
    expected_cache_ids = set(harness._exact_online_prefixes)
    expected_bank = harness._exact_online_actor_bank
    harness._exact_online_store._episodes = dict(episodes)
    harness._exact_online_store._node_count = 15
    harness._exact_online_prefixes = dict(prefixes)
    harness._exact_online_actor_bank = torch.ones(2, 3)

    harness._prune_exact_online_history()

    assert set(harness._exact_online_store._episodes) == expected_ids == {a, b, c, d}
    assert harness._exact_online_store.node_count == expected_nodes == 12
    assert set(harness._exact_online_prefixes) == expected_cache_ids
    assert harness._exact_online_actor_bank is expected_bank is None
    for uid in expected_ids:
        assert harness._exact_online_store._episodes[uid] is episodes[uid]
        assert harness._exact_online_prefixes[uid] is prefixes[uid]
        assert harness._exact_online_store.length(uid) == 3
    assert all(torch.equal(actual, before) for actual, before in zip(reference_tensors, before_refs))
    assert [value.data_ptr() for value in reference_tensors] == before_pointers
    assert harness._exact_online_actor_offsets is offsets
    assert harness._exact_online_generation is generation
    assert harness._exact_online_live_generation is generation
    assert harness._exact_online_encoded_nodes == 15
    assert harness._exact_online_cache_hits == 7


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_live_only_prune_preserves_valid_bank_and_pending_episode(dtype):
    harness = _PruneHarness()
    pending = harness._exact_online_store.allocate_episode_uid()
    harness._exact_online_carry_refs = torch.tensor(
        [[uid, 0] for uid in harness.uids + [pending]], dtype=dtype
    )
    bank = harness._exact_online_actor_bank
    prefixes = dict(harness._exact_online_prefixes)

    harness._prune_exact_online_history()

    assert set(harness._exact_online_store._episodes) == set(harness.uids)
    assert harness._exact_online_store._pending_uids == {pending}
    assert harness._exact_online_store.node_count == 15
    assert harness._exact_online_actor_bank is bank
    assert all(harness._exact_online_prefixes[uid] is entry for uid, entry in prefixes.items())


def test_single_pending_reference_preserves_its_allocation_only():
    harness = _PruneHarness()
    pending = harness._exact_online_store.allocate_episode_uid()
    harness._exact_online_carry_refs = torch.tensor([[pending, 0]])

    harness._prune_exact_online_history()

    assert harness._exact_online_store.episode_count == 0
    assert harness._exact_online_store.node_count == 0
    assert harness._exact_online_store._pending_uids == {pending}
    assert harness._exact_online_prefixes == {}
    assert harness._exact_online_actor_bank is None


@pytest.mark.parametrize("carry", [None, torch.empty((0, 2), dtype=torch.long)])
def test_empty_prune_removes_all_unreferenced_raw_history_and_cache(carry):
    harness = _PruneHarness()
    harness._exact_online_store.allocate_episode_uid()
    harness._exact_online_carry_refs = carry

    harness._prune_exact_online_history()

    assert harness._exact_online_store.episode_count == 0
    assert harness._exact_online_store.node_count == 0
    assert harness._exact_online_store._pending_uids == set()
    assert harness._exact_online_prefixes == {}
    assert harness._exact_online_actor_bank is None


def test_prune_still_rejects_missing_replay_reference_fields():
    harness = _PruneHarness()
    harness.dagger_replay = SimpleNamespace(size=1, data={})

    with pytest.raises(RuntimeError, match="lacks exact episode references"):
        harness._prune_exact_online_history()

    assert harness._exact_online_store.episode_count == 5


def test_prune_still_rejects_evicted_uid_without_mutating_store_or_cache():
    harness = _PruneHarness()
    harness._exact_online_carry_refs = torch.tensor([[999, 0]])
    bank = harness._exact_online_actor_bank

    with pytest.raises(RuntimeError, match="cannot retain missing or evicted"):
        harness._prune_exact_online_history()

    assert harness._exact_online_store.episode_count == 5
    assert set(harness._exact_online_prefixes) == set(harness.uids)
    assert harness._exact_online_actor_bank is bank
