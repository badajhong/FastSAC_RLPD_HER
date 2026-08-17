from __future__ import annotations

import random

import pytest
import torch

from active_adaptation.learning.ppo.teacher_episode_replay import (
    CurrentEMATeacherActorCache,
    TeacherActorCacheLineage,
    TeacherBoundaryCause,
    TeacherEpisodeSequenceStore,
    classify_teacher_boundary,
)


IS_INIT = "perception_is_init"


def _raw(uid: int, length: int) -> dict[str, torch.Tensor]:
    value = torch.arange(length, dtype=torch.float32) + 1000.0 * uid
    return {
        "perception_depth_u8": value.to(torch.uint8).view(length, 1),
        "perception_policy_raw": value.view(length, 1),
        "perception_vel_command_raw": (value + 0.25).view(length, 1),
        IS_INIT: torch.tensor([True, *([False] * (length - 1))]).view(length, 1),
    }


def _lineage(
    store: TeacherEpisodeSequenceStore, *, ema_generation: int = 3
) -> TeacherActorCacheLineage:
    raw = store.lineage
    return TeacherActorCacheLineage(
        raw_store_id=raw.store_id,
        raw_generation=raw.generation,
        ema_generation=ema_generation,
        vecnorm_fingerprint="vec",
        object_geo_fingerprint="geo",
        encoder_semantics="test_encoder_v1",
    )


def test_teacher_boundary_precedence_is_explicit():
    assert classify_teacher_boundary(
        done=False, terminated=True, command_finished=True, time_limit=True
    ) is None
    assert classify_teacher_boundary(
        done=True, terminated=True, command_finished=True, time_limit=True
    ) is TeacherBoundaryCause.TERMINATED
    assert classify_teacher_boundary(
        done=True, terminated=False, command_finished=True, time_limit=True
    ) is TeacherBoundaryCause.SUCCESS_COMMAND
    assert classify_teacher_boundary(
        done=True, terminated=False, command_finished=False, time_limit=True
    ) is TeacherBoundaryCause.TIME_LIMIT
    assert classify_teacher_boundary(
        done=True, terminated=False, command_finished=False, time_limit=False
    ) is TeacherBoundaryCause.OTHER_DONE


def test_only_complete_reset_rooted_successful_episodes_commit():
    store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
    uid = store.allocate_episode_uid()
    with pytest.raises(ValueError, match="Only command-successful"):
        store.commit_successful_episode(
            uid, _raw(uid, 3), boundary_cause=TeacherBoundaryCause.TIME_LIMIT
        )

    bad = _raw(uid, 3)
    bad[IS_INIT][0] = False
    with pytest.raises(ValueError, match="start at is_init"):
        store.commit_successful_episode(uid, bad)

    bad = _raw(uid, 3)
    bad[IS_INIT][2] = True
    with pytest.raises(ValueError, match="internal reset"):
        store.commit_successful_episode(uid, bad)

    store.commit_successful_episode(uid, _raw(uid, 3))
    store.rollback_episode(uid)
    assert store.node_count == 0
    with pytest.raises(ValueError, match="already been consumed"):
        # Rolled-back UIDs remain consumed and are not legal for re-commit.
        store.commit_successful_episode(uid, _raw(uid, 3))


def test_freeze_keeps_full_partially_referenced_episode_and_gc_unreferenced():
    store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
    uids = [store.allocate_episode_uid() for _ in range(4)]
    for uid, length in zip(uids, (7, 4, 5)):
        store.commit_successful_episode(uid, _raw(uid, length))

    # This is the state after a capacity-crossing FIFO write: only the tail of
    # the oldest episode remains, the middle episode is gone, and the newest is
    # wholly present.  The sidecar must retain all seven raw nodes of UID 0.
    live_uid = torch.tensor([uids[0], uids[0], uids[2], uids[2], uids[2]])
    live_step = torch.tensor([5, 6, 1, 3, 4])
    store.freeze(live_uid, live_step)

    assert store.frozen
    assert store.episode_count == 2
    assert store.node_count == 12
    assert store.resolve_node_indices(torch.tensor([uids[0]]), torch.tensor([0])).item() == 0
    assert (
        store.resolve_node_indices(
            torch.tensor([uids[0]]), torch.tensor([6]), next_state=True
        ).item()
        == 6
    )
    with pytest.raises(KeyError, match="evicted|unknown"):
        store.resolve_node_indices(torch.tensor([uids[1]]), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="immutable"):
        store.commit_successful_episode(uids[3], _raw(9, 2))


def test_filtered_prefix_is_present_and_stream_chunks_cover_each_node_once():
    store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
    uids = [store.allocate_episode_uid() for _ in range(3)]
    lengths = (10, 3, 12)
    for uid, length in zip(uids, lengths):
        store.commit_successful_episode(uid, _raw(uid, length))

    # Replay rows begin at step 6, but exact recurrence still needs steps 0..5.
    store.freeze(
        torch.tensor([uids[0], uids[0], uids[1], uids[2]]),
        torch.tensor([6, 9, 2, 8]),
    )
    seen = torch.zeros(store.node_count, dtype=torch.int64)
    policy = torch.empty(store.node_count)
    last_group = -1
    last_start = -1
    for chunk in store.iter_sequence_chunks(
        episode_batch_size=2, time_chunk_size=4
    ):
        if chunk.group_id != last_group:
            last_group = chunk.group_id
            last_start = -1
        assert chunk.start_step > last_start
        last_start = chunk.start_step
        indices = chunk.flat_node_indices[chunk.valid]
        values = chunk.raw_fields["perception_policy_raw"][chunk.valid].reshape(-1)
        seen.index_add_(0, indices, torch.ones_like(indices))
        policy.index_copy_(0, indices, values)
    assert bool((seen == 1).all())
    assert torch.equal(policy, store.raw_fields["perception_policy_raw"].reshape(-1))
    # UID 0's unreferenced filtered prefix is really present in the compact store.
    assert torch.equal(policy[:6], torch.arange(6, dtype=torch.float32))


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ),
)
def test_stream_accepts_one_frozen_raw_device_mirror_without_semantic_drift(device):
    store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
    uids = [store.allocate_episode_uid() for _ in range(2)]
    for uid, length in zip(uids, (7, 3)):
        store.commit_successful_episode(uid, _raw(uid, length))
    store.freeze(torch.tensor(uids), torch.tensor([5, 2]))

    mirror = {
        key: value.to(device).contiguous()
        for key, value in store.raw_fields.items()
    }
    reference = list(
        store.iter_sequence_chunks(episode_batch_size=2, time_chunk_size=4)
    )
    actual = list(
        store.iter_sequence_chunks(
            episode_batch_size=2,
            time_chunk_size=4,
            raw_fields=mirror,
        )
    )
    assert len(actual) == len(reference)
    for expected_chunk, actual_chunk in zip(reference, actual):
        assert torch.equal(actual_chunk.valid, expected_chunk.valid)
        assert torch.equal(
            actual_chunk.flat_node_indices, expected_chunk.flat_node_indices
        )
        for key in expected_chunk.raw_fields:
            assert actual_chunk.raw_fields[key].device.type == device
            assert torch.equal(
                actual_chunk.raw_fields[key].cpu(),
                expected_chunk.raw_fields[key],
            )


def test_current_ema_cache_current_next_and_lineage_rejection():
    store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
    uid = store.allocate_episode_uid()
    store.commit_successful_episode(uid, _raw(uid, 4))
    store.freeze(torch.tensor([uid, uid]), torch.tensor([1, 3]))

    cache = CurrentEMATeacherActorCache(actor_dim=2)
    actor = cache.allocate_build_tensor(store)
    actor.copy_(torch.arange(8, dtype=torch.float32).view(4, 2))
    lineage = _lineage(store)
    cache.publish(lineage, store, actor)

    current = cache.gather(
        store,
        torch.tensor([uid, uid]),
        torch.tensor([1, 3]),
        lineage=lineage,
        next_state=False,
    )
    next_value = cache.gather(
        store,
        torch.tensor([uid, uid]),
        torch.tensor([1, 3]),
        lineage=lineage,
        next_state=True,
    )
    assert torch.equal(current, actor[[1, 3]])
    assert torch.equal(next_value, actor[[2, 3]])  # successful final aliases current

    stale = _lineage(store, ema_generation=4)
    with pytest.raises(RuntimeError, match="missing or stale"):
        cache.gather(
            store,
            torch.tensor([uid]),
            torch.tensor([1]),
            lineage=stale,
            next_state=False,
        )
    cache.invalidate()
    assert not cache.ready


@pytest.mark.parametrize("capacity", [5, 7, 11, 20])
def test_random_fifo_wrap_freeze_contract(capacity: int):
    for seed in range(8):
        rng = random.Random(10_000 * capacity + seed)
        store = TeacherEpisodeSequenceStore(is_init_key=IS_INIT)
        ring_uid = torch.full((capacity,), -1, dtype=torch.long)
        ring_step = torch.full((capacity,), -1, dtype=torch.long)
        ptr = 0
        size = 0
        raw_by_uid: dict[int, torch.Tensor] = {}

        for _ in range(40):
            uid = store.allocate_episode_uid()
            length = rng.randint(1, min(9, capacity + 3))
            raw = _raw(uid, length)
            store.commit_successful_episode(uid, raw)
            raw_by_uid[uid] = raw["perception_policy_raw"].reshape(-1)

            # A successful episode can have filtered/invalid-action gaps.
            steps = [step for step in range(length) if rng.random() < 0.65]
            if not steps:
                continue
            incoming_uid = torch.full((len(steps),), uid, dtype=torch.long)
            incoming_step = torch.tensor(steps, dtype=torch.long)
            count = len(steps)
            if count >= capacity:
                ring_uid.copy_(incoming_uid[-capacity:])
                ring_step.copy_(incoming_step[-capacity:])
                ptr = 0
                size = capacity
            else:
                first = min(count, capacity - ptr)
                second = count - first
                ring_uid[ptr : ptr + first] = incoming_uid[:first]
                ring_step[ptr : ptr + first] = incoming_step[:first]
                if second:
                    ring_uid[:second] = incoming_uid[first:]
                    ring_step[:second] = incoming_step[first:]
                ptr = (ptr + count) % capacity
                size = min(size + count, capacity)

        store.freeze(ring_uid[:size], ring_step[:size])
        live_uids = set(ring_uid[:size].tolist())
        assert store.episode_count == len(live_uids)
        for uid in live_uids:
            length = int(raw_by_uid[uid].numel())
            indices = store.resolve_node_indices(
                torch.full((length,), uid), torch.arange(length)
            )
            retained = store.raw_fields["perception_policy_raw"].reshape(-1)[indices]
            assert torch.equal(retained, raw_by_uid[uid])
