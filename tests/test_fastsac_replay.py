import os

import h5py
import pytest
import torch

from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTION_PARAMETERIZATION,
    NEXT_TEACHER_HEIGHT_FIELD,
    NEXT_TEACHER_REF_ACTION_FIELD,
    OfflineReplayH5,
    OnlineReplay,
    FastSACVelConfig,
    TEACHER_REPLAY_FIELDS,
    TEACHER_REPLAY_FORMAT_VERSION,
    TEACHER_REPLAY_INITIAL_TRANSITION_FILTER,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    TEACHER_TRAINING_REPLAY_FIELDS,
    TEACHER_HEIGHT_FIELD,
    TEACHER_OBJECT_GEO_FIELD,
    TEACHER_REF_ACTION_FIELD,
    TeacherReplayBuffer,
    TeacherTrainingReplayBuffer,
    _resolve_teacher_training_replay_device,
    _validate_seed_replay_partition,
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


def _training_batch(ids, device="cpu", critic_dim=4, action_dim=2, geometry=None):
    ids = torch.as_tensor(ids, dtype=torch.float32, device=device)

    def expanded(offset, width):
        return (ids + offset).unsqueeze(-1).expand(-1, width).clone()

    if geometry is None:
        geometry = torch.tensor([7.0, 8.0, 9.0], device=device)
    geometry = torch.as_tensor(geometry, dtype=torch.float32, device=device)
    if geometry.ndim == 1:
        geometry = geometry.expand(ids.numel(), -1).clone()
    integer_ids = ids.long()
    return {
        "critic_observations": expanded(100, critic_dim),
        "actions": expanded(200, action_dim),
        "rewards": ids + 300,
        "dones": integer_ids.remainder(2).bool(),
        "truncations": integer_ids.remainder(3).eq(0),
        "discounts": torch.full_like(ids, 0.9),
        "next_critic_observations": expanded(500, critic_dim),
        TEACHER_REF_ACTION_FIELD: expanded(600, action_dim),
        NEXT_TEACHER_REF_ACTION_FIELD: expanded(700, action_dim),
        TEACHER_OBJECT_GEO_FIELD: geometry,
    }


def _training_buffer(
    capacity=5,
    device="cpu",
    seed_storage_ratio=0.0,
    seed_sample_ratio=0.0,
):
    return TeacherTrainingReplayBuffer(
        capacity=capacity,
        critic_dim=4,
        action_dim=2,
        device=device,
        extra_shapes={
            TEACHER_REF_ACTION_FIELD: (2,),
            NEXT_TEACHER_REF_ACTION_FIELD: (2,),
        },
        constant_shapes={TEACHER_OBJECT_GEO_FIELD: (3,)},
        seed_storage_ratio=seed_storage_ratio,
        seed_sample_ratio=seed_sample_ratio,
    )


def _ordered_training(buffer, name):
    if buffer.size < buffer.capacity:
        segments = ((0, buffer.size),)
    elif buffer.ptr == 0:
        segments = ((0, buffer.capacity),)
    else:
        segments = ((buffer.ptr, buffer.capacity), (0, buffer.ptr))
    return torch.cat([buffer.data[name][start:end] for start, end in segments])


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
    assert cfg.teacher_training_replay_device == "policy"
    assert cfg.sac_updates_per_env_step == 4
    assert cfg.sac_teacher_updates_per_env_step == 4
    assert cfg.sac_teacher_n_steps == 1
    assert cfg.sac_teacher_actor_objective == "sac"
    assert cfg.sac_teacher_awac_beta == 0.01
    assert cfg.sac_teacher_awac_weight_clip == 20.0
    assert cfg.sac_teacher_learning_starts_transitions == 98_304
    assert cfg.q_num_atoms == 501
    assert cfg.q_action_fusion == "early"
    assert cfg.q_reference_dueling is False
    assert cfg.save_teacher_buffer is False
    replay = TeacherReplayBuffer(
        "unused.h5", capacity=262_144, actor_dim=525, critic_dim=2341,
        action_dim=23, seed=0, device="cpu",
    )
    assert replay.estimated_bytes == 6_037_176_320
    assert replay.estimated_bytes / (1024 ** 3) == pytest.approx(5.62255859375)
    # Allocation is deliberately delayed until the first selected transition.
    assert replay.data == {}


def test_stage1_replay_storage_device_config_is_strict():
    assert _resolve_teacher_training_replay_device(
        "policy", "cpu"
    ) == torch.device("cpu")
    assert _resolve_teacher_training_replay_device(
        "cpu", "cpu"
    ) == torch.device("cpu")
    assert _resolve_teacher_training_replay_device(
        "cuda", "cuda:2"
    ) == torch.device("cuda:2")
    assert _resolve_teacher_training_replay_device(
        "cuda:3", "cpu"
    ) == torch.device("cuda:3")

    for invalid in (None, 0, "", " cpu", "cpu ", "cpu:0", "mps"):
        with pytest.raises(ValueError, match="teacher_training_replay_device"):
            _resolve_teacher_training_replay_device(invalid, "cpu")


def test_stage1_cpu_replay_sampling_keeps_policy_device_dtype_and_rng(monkeypatch):
    replay = _training_buffer(capacity=5, device="cpu")
    replay.append(_training_batch([0, 1, 2, 3, 4], device="cpu"))

    # Sampling onto the same policy/storage device must not stage any field
    # through another Tensor.to call. The cross-device branch uses one call per
    # selected field when the actual policy is CUDA.
    def forbidden_to(self, *args, **kwargs):
        raise AssertionError("CPU-to-CPU replay sampling must not transfer")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", forbidden_to)
        first = replay.sample(
            32,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(17),
        )

    # A fresh generator with the same state selects exactly the same rows.
    second = replay.sample(
        32,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(17),
    )
    assert first.keys() == second.keys()
    for name in replay.sample_fields:
        assert first[name].device == torch.device("cpu")
        assert first[name].dtype == replay.dtypes[name]
        assert torch.equal(first[name], second[name])


def test_skateboard_stage1_compact_replay_estimate_excludes_actor_observations():
    replay = TeacherTrainingReplayBuffer(
        capacity=262_144,
        critic_dim=2341,
        action_dim=23,
        extra_shapes={
            TEACHER_REF_ACTION_FIELD: (23,),
            NEXT_TEACHER_REF_ACTION_FIELD: (23,),
        },
        constant_shapes={TEACHER_OBJECT_GEO_FIELD: (384,)},
    )
    # 19,018 bytes per row plus one 384-float geometry tensor. In particular,
    # neither 525-wide current/next student actor observation is allocated.
    assert replay.estimated_bytes == 4_985_456_128
    assert replay.estimated_bytes / (1024 ** 3) == pytest.approx(
        4.643067836761475
    )
    assert "observations" not in replay.storage_fields
    assert "next_observations" not in replay.storage_fields
    assert tuple(replay.fields) == TEACHER_TRAINING_REPLAY_FIELDS
    assert replay.data == {}


def test_stage1_compact_fifo_keeps_rows_aligned_and_geometry_once(monkeypatch):
    replay = _training_buffer(capacity=5)

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("Stage-1 append must remain device-local")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden_cpu)
        replay.append(_training_batch([0, 1, 2]))
        replay.append(_training_batch([3, 4, 5, 6]))

    assert replay.size == replay.saved == 5
    assert replay.seen == 7
    assert replay.ptr == 2
    ids = _ordered_training(replay, "critic_observations")[:, 0] - 100
    assert torch.equal(ids, torch.arange(2.0, 7.0))
    assert torch.equal(
        _ordered_training(replay, "next_critic_observations")[:, 0],
        ids + 500,
    )
    assert torch.equal(
        _ordered_training(replay, TEACHER_REF_ACTION_FIELD)[:, 0],
        ids + 600,
    )
    assert torch.equal(
        _ordered_training(replay, "effective_n_steps"), torch.ones(5)
    )
    assert set(replay.data) == set(replay.storage_fields)
    assert TEACHER_OBJECT_GEO_FIELD not in replay.data
    assert replay.constants[TEACHER_OBJECT_GEO_FIELD].shape == (3,)

    sample = replay.sample(32, generator=torch.Generator().manual_seed(3))
    assert "observations" not in sample
    assert "next_observations" not in sample
    assert sample[TEACHER_OBJECT_GEO_FIELD].shape == (32, 3)
    assert torch.equal(
        sample[TEACHER_OBJECT_GEO_FIELD],
        torch.tensor([7.0, 8.0, 9.0]).expand(32, -1),
    )


def test_stage1_seed_partition_freezes_prefix_and_mixes_seed_online_rows():
    replay = _training_buffer(
        capacity=8,
        seed_storage_ratio=0.25,
        seed_sample_ratio=0.5,
    )
    replay.append(_training_batch(range(8)))

    assert replay.freeze_seed_partition()
    assert not replay.freeze_seed_partition()
    assert replay.seed_frozen
    assert replay.seed_size == replay.seed_capacity == 2
    assert replay.online_size == replay.online_capacity == 6
    frozen = replay.data["critic_observations"][:2].clone()

    # A complete online-partition replacement must never mutate the prefix.
    replay.append(_training_batch(range(8, 14)))
    assert torch.equal(replay.data["critic_observations"][:2], frozen)
    assert torch.equal(
        replay.data["critic_observations"][2:, 0] - 100,
        torch.arange(8.0, 14.0),
    )

    sample = replay.sample(
        100,
        generator=torch.Generator().manual_seed(11),
    )
    ids = sample["critic_observations"][:, 0] - 100
    assert int((ids < 2).sum()) == 50
    assert torch.all((ids < 2) | ((ids >= 8) & (ids < 14)))

    replay.clear()
    assert not replay.seed_frozen
    assert replay.size == replay.seed_size == replay.online_size == 0


def test_stage1_seed_partition_requires_full_replay_and_valid_paired_ratios():
    replay = _training_buffer(
        capacity=8,
        seed_storage_ratio=0.25,
        seed_sample_ratio=0.5,
    )
    replay.append(_training_batch(range(7)))
    with pytest.raises(RuntimeError, match="only after the FIFO is full"):
        replay.freeze_seed_partition()

    for storage, sample in (
        (-0.1, 0.5),
        (1.0, 0.5),
        (0.25, -0.1),
        (0.25, 1.1),
        (0.0, 0.5),
        (0.25, 0.0),
    ):
        with pytest.raises(ValueError, match="sac_teacher_seed"):
            _validate_seed_replay_partition(storage, sample, capacity=8)

    with pytest.raises(ValueError, match="at least one seed"):
        _validate_seed_replay_partition(0.01, 0.5, capacity=8)


def test_stage1_compact_replay_rejects_geometry_that_is_not_global_constant():
    replay = _training_buffer(capacity=5)
    varying = torch.tensor([[7.0, 8.0, 9.0], [7.0, 8.0, 10.0]])
    with pytest.raises(ValueError, match="varies within"):
        replay.append(_training_batch([0, 1], geometry=varying))

    replay.append(_training_batch([0, 1]))
    with pytest.raises(ValueError, match="changed after collection"):
        replay.append(_training_batch([2], geometry=[7.0, 8.0, 10.0]))


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
        assert (
            snapshot.attrs["truncation_next_observation"]
            == TRUNCATION_NEXT_OBSERVATION_SEMANTICS
        )
        assert (
            snapshot.attrs["initial_transition_filter"]
            == TEACHER_REPLAY_INITIAL_TRANSITION_FILTER
        )
        assert (
            snapshot.attrs["action_parameterization"]
            == FASTSAC_ACTION_PARAMETERIZATION
        )
        assert snapshot.attrs["q_action_fusion"] == "early"
        assert int(snapshot.attrs["q_action_hidden_dim"]) == 0
        assert bool(snapshot.attrs["q_reference_dueling"]) is False
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


def test_teacher_h5_records_and_enforces_late_q_action_fusion(tmp_path):
    path = tmp_path / "late_teacher_replay.h5"
    replay = TeacherReplayBuffer(
        path,
        capacity=4,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        seed=0,
        q_action_fusion="late",
        q_action_hidden_dim=128,
    )
    replay.append(_batch([0, 1]))
    replay.snapshot(iteration=2, checkpoint_name="checkpoint_2")

    with h5py.File(path, "r") as snapshot:
        assert snapshot.attrs["q_action_fusion"] == "late"
        assert int(snapshot.attrs["q_action_hidden_dim"]) == 128

    offline = OfflineReplayH5(
        path,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        device="cpu",
        expected_q_action_fusion="late",
        expected_q_action_hidden_dim=128,
    )
    assert offline.snapshot_metadata["q_action_fusion"] == "late"
    assert offline.snapshot_metadata["q_action_hidden_dim"] == 128

    with pytest.raises(ValueError, match="Q action fusion"):
        OfflineReplayH5(
            path,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            device="cpu",
            expected_q_action_fusion="early",
            expected_q_action_hidden_dim=0,
        )


def test_pre_fusion_h5_metadata_is_compatible_with_early_only(tmp_path):
    path = tmp_path / "legacy_early_teacher_replay.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1]))
    replay.snapshot(iteration=2, checkpoint_name="checkpoint_2")
    with h5py.File(path, "r+") as snapshot:
        del snapshot.attrs["q_action_fusion"]
        del snapshot.attrs["q_action_hidden_dim"]

    OfflineReplayH5(
        path,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        device="cpu",
        expected_q_action_fusion="early",
        expected_q_action_hidden_dim=0,
    )
    with pytest.raises(ValueError, match="Q action fusion"):
        OfflineReplayH5(
            path,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            device="cpu",
            expected_q_action_fusion="late",
            expected_q_action_hidden_dim=128,
        )


def test_offline_loader_rejects_old_action_parameterization(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    with h5py.File(path, "r+") as snapshot:
        snapshot.attrs["action_parameterization"] = "absolute_default_centered_v1"

    with pytest.raises(ValueError, match="action coordinates"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2, device="cpu"
        )


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


def test_reference_action_fields_round_trip_but_stage2_rlpd_ignores_them(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    extras = {
        TEACHER_REF_ACTION_FIELD: (2,),
        NEXT_TEACHER_REF_ACTION_FIELD: (2,),
    }
    replay = TeacherReplayBuffer(
        path, capacity=5, actor_dim=3, critic_dim=4, action_dim=2,
        seed=0, extra_shapes=extras, snapshot_chunk_rows=2,
    )
    batch = _batch([0, 1, 2, 3])
    ids = batch["observations"][:, :1]
    batch[TEACHER_REF_ACTION_FIELD] = ids.expand(-1, 2) + 600
    batch[NEXT_TEACHER_REF_ACTION_FIELD] = ids.expand(-1, 2) + 700
    replay.append(batch)
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    with h5py.File(path, "r") as snapshot:
        assert snapshot[TEACHER_REF_ACTION_FIELD][:, 0].tolist() == [
            600, 601, 602, 603
        ]
        assert snapshot[NEXT_TEACHER_REF_ACTION_FIELD][:, 0].tolist() == [
            700, 701, 702, 703
        ]

    resumed = TeacherReplayBuffer(
        tmp_path / "resumed.h5", capacity=5, actor_dim=3, critic_dim=4,
        action_dim=2, seed=0, replay_id=replay.replay_id,
        extra_shapes=extras, snapshot_chunk_rows=2,
    )
    resumed.restore(path, replay.checkpoint_metadata())
    assert torch.equal(
        _ordered(resumed, TEACHER_REF_ACTION_FIELD)[:, 0],
        torch.arange(600.0, 604.0),
    )
    assert torch.equal(
        _ordered(resumed, NEXT_TEACHER_REF_ACTION_FIELD)[:, 0],
        torch.arange(700.0, 704.0),
    )

    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
        device="cpu", max_size=5,
    )
    assert set(offline.data) == set(TEACHER_REPLAY_FIELDS)
    assert TEACHER_REF_ACTION_FIELD not in offline.sample(8)
    assert NEXT_TEACHER_REF_ACTION_FIELD not in offline.sample(8)
    _assert_aligned(offline.data)


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


def test_teacher_fifo_restore_rejects_wrong_replay_id(tmp_path):
    source_path = tmp_path / "source_teacher_replay.h5"
    source = _buffer(source_path)
    source.append(_batch([0, 1, 2]))
    source.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")

    resumed = TeacherReplayBuffer(
        tmp_path / "new_run_teacher_replay.h5",
        capacity=5,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        seed=0,
        replay_id="different-replay-id",
    )
    with pytest.raises(ValueError, match="replay_id"):
        resumed.restore(source_path, source.checkpoint_metadata())
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


def test_offline_loader_checks_semantics_not_checkpoint_identity(tmp_path):
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
        expected_actor_backend="paired-backend",
        expected_actor_obs_keys=["actor_a", "actor_b"],
        expected_critic_obs_keys=["critic_a"],
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
    with pytest.raises(ValueError, match="observation keys"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=2,
            expected_critic_obs_keys=["wrong-key"],
        )


def test_offline_loader_records_but_does_not_require_checkpoint_snapshot(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1, 2]))
    replay.snapshot(iteration=5100, checkpoint_name="checkpoint_5100")
    offline = OfflineReplayH5(
        path, actor_dim=3, critic_dim=4, action_dim=2,
    )
    expected = replay.checkpoint_metadata()
    assert offline.snapshot_metadata == {
        key: expected[key]
        for key in (
            "snapshot_id",
            "snapshot_iteration",
            "checkpoint_name",
            "size",
            "seen",
            "q_action_fusion",
            "q_action_hidden_dim",
            "q_action_coordinates",
            "q_reference_dueling",
        )
    }


def test_reference_residual_h5_requires_and_loads_current_and_next_refs(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    extras = {
        TEACHER_REF_ACTION_FIELD: (2,),
        NEXT_TEACHER_REF_ACTION_FIELD: (2,),
    }
    replay = TeacherReplayBuffer(
        path,
        capacity=5,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        seed=0,
        extra_shapes=extras,
        q_action_coordinates="reference_residual",
    )
    batch = _batch([0, 1, 2])
    ids = batch["observations"][:, :1]
    batch[TEACHER_REF_ACTION_FIELD] = ids.expand(-1, 2) + 600.0
    batch[NEXT_TEACHER_REF_ACTION_FIELD] = ids.expand(-1, 2) + 700.0
    replay.append(batch)
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")

    offline = OfflineReplayH5(
        path,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        expected_q_action_coordinates="reference_residual",
    )

    assert set(offline.data) == set(TEACHER_REPLAY_FIELDS) | set(extras)
    assert torch.equal(
        offline.data[TEACHER_REF_ACTION_FIELD][:, 0],
        torch.tensor([600.0, 601.0, 602.0]),
    )
    assert torch.equal(
        offline.data[NEXT_TEACHER_REF_ACTION_FIELD][:, 0],
        torch.tensor([700.0, 701.0, 702.0]),
    )
    with pytest.raises(ValueError, match="coordinates.*do not match"):
        OfflineReplayH5(
            path,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            expected_q_action_coordinates="absolute",
        )


def test_reference_residual_h5_rejects_missing_reference_schema(tmp_path):
    with pytest.raises(ValueError, match="requires current and next"):
        TeacherReplayBuffer(
            tmp_path / "missing_refs.h5",
            capacity=5,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            seed=0,
            q_action_coordinates="reference_residual",
        )


def test_reference_dueling_h5_requires_refs_and_rejects_architecture_mismatch(
    tmp_path,
):
    path = tmp_path / "dueling_teacher_replay.h5"
    extras = {
        TEACHER_REF_ACTION_FIELD: (2,),
        NEXT_TEACHER_REF_ACTION_FIELD: (2,),
    }
    replay = TeacherReplayBuffer(
        path,
        capacity=5,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        seed=0,
        extra_shapes=extras,
        q_action_coordinates="absolute",
        q_reference_dueling=True,
    )
    batch = _batch([0, 1, 2])
    batch[TEACHER_REF_ACTION_FIELD] = torch.full((3, 2), 0.25)
    batch[NEXT_TEACHER_REF_ACTION_FIELD] = torch.full((3, 2), 0.5)
    replay.append(batch)
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")

    offline = OfflineReplayH5(
        path,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        expected_q_action_coordinates="absolute",
        expected_q_reference_dueling=True,
    )
    assert offline.snapshot_metadata["q_reference_dueling"] is True
    assert set(offline.data) == set(TEACHER_REPLAY_FIELDS) | set(extras)

    with pytest.raises(ValueError, match="reference-dueling setting"):
        OfflineReplayH5(
            path,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            expected_q_reference_dueling=False,
        )
    with pytest.raises(ValueError, match="requires current and next"):
        TeacherReplayBuffer(
            tmp_path / "missing_dueling_refs.h5",
            capacity=5,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            seed=0,
            q_reference_dueling=True,
        )


def test_pre_dueling_h5_metadata_is_compatible_with_direct_only(tmp_path):
    path = tmp_path / "legacy_direct_teacher_replay.h5"
    replay = _buffer(path)
    replay.append(_batch([0, 1]))
    replay.snapshot(iteration=2, checkpoint_name="checkpoint_2")
    with h5py.File(path, "r+") as snapshot:
        del snapshot.attrs["q_reference_dueling"]

    OfflineReplayH5(
        path,
        actor_dim=3,
        critic_dim=4,
        action_dim=2,
        expected_q_reference_dueling=False,
    )
    with pytest.raises(ValueError, match="reference-dueling setting"):
        OfflineReplayH5(
            path,
            actor_dim=3,
            critic_dim=4,
            action_dim=2,
            expected_q_reference_dueling=True,
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
