import os

import h5py
import pytest
import torch

from active_adaptation.learning.ppo.fastsac_vel import (
    NEXT_TEACHER_HEIGHT_FIELD,
    OfflineReplayH5,
    OnlineReplay,
    FastSACVelConfig,
    TEACHER_REPLAY_FIELDS,
    TEACHER_REPLAY_FORMAT_VERSION,
    TEACHER_HEIGHT_FIELD,
    TEACHER_OBJECT_GEO_FIELD,
    TeacherReplayBuffer,
)


def _batch(ids, device="cpu", actor_dim=3, critic_dim=4, action_dim=2):
    ids = torch.as_tensor(ids, dtype=torch.float32, device=device)

    def expanded(offset, width):
        return (ids + offset).unsqueeze(-1).expand(-1, width).clone()

    integer_ids = ids.long()
    return {
        "observations": expanded(0, actor_dim),
        "critic_observations": expanded(100, critic_dim),
        "actions": expanded(200, action_dim),
        "rewards": ids + 300,
        "dones": integer_ids.remainder(2).bool(),
        "truncations": integer_ids.remainder(3).eq(0),
        "discounts": torch.full_like(ids, 0.9),
        "next_observations": expanded(400, actor_dim),
        "next_critic_observations": expanded(500, critic_dim),
    }


def _buffer(path, capacity=5, device="cpu", chunk_rows=2):
    return TeacherReplayBuffer(
        path, capacity=capacity, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, device=device, snapshot_chunk_rows=chunk_rows,
    )


def _ordered(buffer, name):
    return torch.cat(
        [buffer.data[name][start:end] for start, end in buffer._chronological_segments()]
    )


def _assert_aligned(data):
    ids = data["observations"][:, 0]
    assert torch.equal(data["critic_observations"][:, 0], ids + 100)
    assert torch.equal(data["actions"][:, 0], ids + 200)
    assert torch.equal(data["rewards"], ids + 300)
    assert torch.equal(data["dones"], ids.long().remainder(2).bool())
    assert torch.equal(data["truncations"], ids.long().remainder(3).eq(0))
    assert torch.equal(data["discounts"], torch.full_like(ids, 0.9))
    assert torch.equal(data["next_observations"][:, 0], ids + 400)
    assert torch.equal(data["next_critic_observations"][:, 0], ids + 500)


def test_teacher_replay_capacity_is_262144():
    cfg = FastSACVelConfig()
    assert cfg.teacher_buffer_capacity == 262_144
    assert cfg.sac_updates_per_env_step == 8
    replay = TeacherReplayBuffer(
        "unused.h5", capacity=262_144, actor_dim=525, critic_dim=2341,
        action_dim=23, seed=0, device="cpu",
    )
    assert replay.estimated_bytes == 6_037_176_320
    assert replay.estimated_bytes / (1024 ** 3) == pytest.approx(5.62255859375)
    # Allocation is deliberately delayed until the first selected transition.
    assert replay.data == {}


def test_skateboard_teacher_raw_replay_estimate_is_about_six_gib():
    replay = TeacherReplayBuffer(
        "unused.h5", capacity=262_144, actor_dim=525, critic_dim=2341,
        action_dim=23, seed=0,
        extra_shapes={TEACHER_OBJECT_GEO_FIELD: (384,)},
    )
    assert replay.estimated_bytes == 6_439_829_504
    assert replay.estimated_bytes / (1024 ** 3) == pytest.approx(
        5.99755859375
    )
    assert replay.data == {}


def test_fifo_wrap_keeps_newest_rows_and_append_has_no_host_io(tmp_path, monkeypatch):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("append must not copy replay data to the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden_cpu)
        replay.append(_batch([0, 1, 2]))
        replay.append(_batch([3, 4, 5, 6]))

    assert not path.exists()
    assert replay.size == replay.saved == 5
    assert replay.ptr == 2
    assert replay.seen == 7
    assert torch.equal(_ordered(replay, "observations")[:, 0], torch.arange(2.0, 7.0))
    _assert_aligned({name: _ordered(replay, name) for name in TEACHER_REPLAY_FIELDS})

    replay.append(_batch([7, 8, 9, 10, 11, 12, 13]))
    assert replay.ptr == 0
    assert replay.seen == 14
    assert torch.equal(_ordered(replay, "observations")[:, 0], torch.arange(9.0, 14.0))


def test_online_replay_is_device_resident_and_fifo_aligned():
    replay = OnlineReplay(capacity=5, device="cpu")
    replay.extend(_batch([0, 1, 2]))
    replay.extend(_batch([3, 4, 5, 6]))

    assert replay.size == 5
    assert all(value.device.type == "cpu" for value in replay.data.values())
    sample = replay.sample(64, device="cpu")
    _assert_aligned(sample)


def test_snapshot_is_ordered_and_offline_sampling_never_reopens_h5(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.append(_batch([3, 4, 5, 6]))
    pointers = {name: value.data_ptr() for name, value in replay.data.items()}
    ptr, size, seen = replay.ptr, replay.size, replay.seen

    assert replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100") == str(path)
    assert replay.ptr == ptr and replay.size == size and replay.seen == seen
    assert {name: value.data_ptr() for name, value in replay.data.items()} == pointers

    with h5py.File(path, "r") as snapshot:
        assert int(snapshot.attrs["format_version"]) == TEACHER_REPLAY_FORMAT_VERSION
        assert snapshot.attrs["timeout_next_observation"] == "pre_reset_final"
        assert snapshot.attrs["storage_policy"] == "circular_fifo"
        assert snapshot.attrs["storage_order"] == "oldest_to_newest"
        assert int(snapshot.attrs["buffer_capacity"]) == 5
        assert int(snapshot.attrs["num_transitions"]) == 5
        assert int(snapshot.attrs["num_seen_transitions"]) == 7
        assert int(snapshot.attrs["snapshot_iteration"]) == 5100
        assert snapshot.attrs["checkpoint_name"] == "checkpoint_5100"
        assert snapshot.attrs["snapshot_id"] == replay.last_snapshot_id
        assert snapshot["dones"].dtype == torch.zeros((), dtype=torch.bool).numpy().dtype
        assert snapshot["observations"][:, 0].tolist() == [2, 3, 4, 5, 6]

    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        device="cpu", max_size=5, seed=7, load_chunk_rows=2,
    )
    path.unlink()
    sample = offline.sample(64)
    assert all(value.device.type == "cpu" for value in sample.values())
    _assert_aligned(sample)


def test_gated_snapshot_writes_only_eligible_fifo_tail_without_clearing(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2, 3, 4]))
    replay.append(_batch([5, 6]))

    replay.snapshot(
        iteration=5100,
        checkpoint_name="checkpoint_5100",
        row_count=2,
        seen_count=2,
    )

    # FastSAC still trains from the complete rolling FIFO.
    assert torch.equal(
        _ordered(replay, "observations")[:, 0],
        torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    # The stage-2 file contains only rows appended after the H5 gate.
    with h5py.File(path, "r") as snapshot:
        assert int(snapshot.attrs["num_transitions"]) == 2
        assert int(snapshot.attrs["num_seen_transitions"]) == 2
        assert snapshot["observations"][:, 0].tolist() == [5.0, 6.0]
    assert replay.checkpoint_metadata()["size"] == 2
    assert replay.checkpoint_metadata()["seen"] == 2


def test_teacher_internal_fields_round_trip_but_stage2_ignores_them(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    extras = {
        TEACHER_HEIGHT_FIELD: (2,),
        NEXT_TEACHER_HEIGHT_FIELD: (2,),
    }
    replay = TeacherReplayBuffer(
        path, capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, extra_shapes=extras, snapshot_chunk_rows=2,
    )
    batch = _batch([0, 1, 2, 3])
    ids = batch["observations"][:, :1]
    batch[TEACHER_HEIGHT_FIELD] = ids.expand(-1, 2) + 600
    batch[NEXT_TEACHER_HEIGHT_FIELD] = ids.expand(-1, 2) + 700
    replay.append(batch)
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    resumed = TeacherReplayBuffer(
        tmp_path / "resumed.h5", capacity=5, actor_dim=3, critic_dim=4,
        action_dim=2, seed=0, replay_id=replay.replay_id,
        extra_shapes=extras, snapshot_chunk_rows=2,
    )
    resumed.restore(path, replay.checkpoint_metadata())
    assert torch.equal(
        _ordered(resumed, TEACHER_HEIGHT_FIELD)[:, 0],
        torch.arange(600.0, 604.0),
    )
    assert torch.equal(
        _ordered(resumed, NEXT_TEACHER_HEIGHT_FIELD)[:, 0],
        torch.arange(700.0, 704.0),
    )

    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        device="cpu", max_size=5,
    )
    assert set(offline.data) == set(TEACHER_REPLAY_FIELDS)
    _assert_aligned(offline.data)

    base_bytes = _buffer(tmp_path / "base.h5").estimated_bytes
    assert replay.estimated_bytes == base_bytes + 5 * 4 * 4


def test_offline_loader_keeps_newest_fifo_tail(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path, capacity=8)
    replay.append(_batch(range(7)))
    replay.append(_batch(range(7, 11)))
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        device="cpu", max_size=3, seed=7, load_chunk_rows=2,
    )
    assert offline.size == 3
    assert torch.equal(
        offline.data["observations"][:, 0], torch.tensor([8.0, 9.0, 10.0])
    )
    _assert_aligned(offline.data)


def test_teacher_fifo_restore_then_append_matches_uninterrupted_fifo(tmp_path):
    source_path = tmp_path / "source_teacher_replay.h5"
    source = TeacherReplayBuffer(
        source_path, capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, replay_id="same-run", actor_backend="same-backend",
        actor_obs_keys=["actor"], critic_obs_keys=["critic"],
        snapshot_chunk_rows=2,
    )
    source.append(_batch([0, 1, 2]))
    source.append(_batch([3, 4, 5, 6]))
    source.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")
    checkpoint_metadata = source.checkpoint_metadata()

    resumed = TeacherReplayBuffer(
        tmp_path / "new_run_teacher_replay.h5",
        capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, replay_id="same-run", actor_backend="same-backend",
        actor_obs_keys=["actor"], critic_obs_keys=["critic"],
        snapshot_chunk_rows=2,
    )
    assert resumed.restore(source_path, checkpoint_metadata) == 5
    assert resumed.size == 5
    assert resumed.seen == 7
    assert resumed.ptr == 0
    assert resumed.last_snapshot_id == source.last_snapshot_id
    _assert_aligned({name: _ordered(resumed, name) for name in TEACHER_REPLAY_FIELDS})

    source.append(_batch([7, 8, 9]))
    resumed.append(_batch([7, 8, 9]))
    assert resumed.seen == source.seen == 10
    for name in TEACHER_REPLAY_FIELDS:
        assert torch.equal(_ordered(resumed, name), _ordered(source, name))


def test_teacher_fifo_restore_rejects_wrong_checkpoint_snapshot(tmp_path):
    source_path = tmp_path / "source_teacher_replay.h5"
    source = _buffer(source_path)
    source.append(_batch([0, 1, 2]))
    source.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")
    expected = source.checkpoint_metadata()
    expected["snapshot_id"] = "a-different-checkpoint-snapshot"

    resumed = TeacherReplayBuffer(
        tmp_path / "new_run_teacher_replay.h5",
        capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, replay_id=source.replay_id,
    )
    with pytest.raises(ValueError, match="exact checkpoint"):
        resumed.restore(source_path, expected)
    assert resumed.data == {}
    assert resumed.ptr == resumed.size == resumed.seen == 0


def test_teacher_fifo_restore_requires_exact_capacity(tmp_path):
    source_path = tmp_path / "source_teacher_replay.h5"
    source = _buffer(source_path, capacity=5)
    source.append(_batch([0, 1, 2]))
    source.snapshot(iteration=1, checkpoint_name="checkpoint_1")

    resumed = TeacherReplayBuffer(
        tmp_path / "new_run_teacher_replay.h5",
        capacity=6, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, replay_id=source.replay_id,
    )
    with pytest.raises(ValueError, match="capacity"):
        resumed.restore(source_path)
    assert resumed.data == {}


def test_offline_loader_rejects_inconsistent_row_metadata(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")
    with h5py.File(path, "r+") as snapshot:
        snapshot.attrs["num_transitions"] = 4

    with pytest.raises(ValueError, match="expected"):
        OfflineReplayH5(path, actor_dim=3, critic_dim=4, action_dim=2)


def test_offline_loader_rejects_checkpoint_replay_provenance_mismatch(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = TeacherReplayBuffer(
        path, capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, replay_id="paired-run", actor_backend="paired-backend",
        actor_obs_keys=["actor_a", "actor_b"],
        critic_obs_keys=["critic_a"],
    )
    replay.append(_batch([0, 1]))
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        expected_replay_id="paired-run",
        expected_actor_backend="paired-backend",
        expected_actor_obs_keys=["actor_a", "actor_b"],
        expected_critic_obs_keys=["critic_a"],
    )
    with pytest.raises(ValueError, match="does not match checkpoint replay id"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2,
            expected_replay_id="wrong-run",
        )
    with pytest.raises(ValueError, match="does not match checkpoint backend"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2,
            expected_actor_backend="wrong-backend",
        )
    with pytest.raises(ValueError, match="observation keys"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2,
            expected_actor_obs_keys=["wrong-key"],
        )


def test_offline_loader_requires_exact_checkpoint_snapshot(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")
    expected = replay.checkpoint_metadata()

    OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        expected_snapshot_metadata=expected,
    )
    expected["snapshot_id"] = "snapshot-from-a-different-checkpoint"
    with pytest.raises(ValueError, match="exact checkpoint"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2,
            expected_snapshot_metadata=expected,
        )


def test_snapshot_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")
    before = path.read_bytes()
    replay.append(_batch([3, 4]))

    def fail_replace(source, destination):
        assert os.path.dirname(source) == os.path.dirname(destination)
        assert os.path.exists(source)
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        replay.snapshot(iteration=2, checkpoint_name="checkpoint_2")

    assert path.read_bytes() == before
    assert list(tmp_path.glob(".teacher_replay_buffer.h5.*.tmp")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fifo_and_offline_replay_stay_on_device(tmp_path, monkeypatch):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path, capacity=5, device="cuda:0")

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("append must remain on CUDA")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden_cpu)
        replay.append(_batch([0, 1, 2], device="cuda:0"))
        replay.append(_batch([3, 4, 5, 6], device="cuda:0"))
        torch.cuda.synchronize()

    assert all(value.device.type == "cuda" for value in replay.data.values())
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")
    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        device="cuda:0", max_size=5, seed=7, load_chunk_rows=2,
    )
    path.unlink()
    sample = offline.sample(64, device="cuda:0")
    assert all(value.device.type == "cuda" for value in offline.data.values())
    assert all(value.device.type == "cuda" for value in sample.values())
    _assert_aligned(sample)
