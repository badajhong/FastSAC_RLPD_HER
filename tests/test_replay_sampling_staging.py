"""Replay gather/staging preserves physical rows, exact bits and ownership."""

from __future__ import annotations

import pytest
import torch

from active_adaptation.learning.ppo import replay_sample_staging as staging_module
from active_adaptation.learning.ppo.td3_bc_dagger import _TD3DeviceReplay
from active_adaptation.learning.ppo.replay_sample_staging import (
    _gather_packed_replay,
    _packed_replay_layout,
    _unpack_replay_sample,
)


@pytest.fixture(params=["cpu", "cuda:0"])
def output_device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA staging requires an available GPU")
    return request.param


def _rows():
    bit_patterns = torch.tensor(
        [0x00000000, 0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000, 0x3F800001],
        dtype=torch.int64,
    ).to(torch.int32)
    return {
        "vector": torch.arange(18, dtype=torch.float32).reshape(6, 3) / 7.0,
        "pair": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "matrix": torch.arange(24, dtype=torch.float64).reshape(6, 2, 2),
        "valid": torch.tensor([True, False, True, True, False, True]),
        "pixels": torch.tensor([[0, 255], [1, 254], [2, 253], [3, 252], [4, 251], [5, 250]], dtype=torch.uint8),
        "refs": torch.tensor([[2**60 + row, 2**63 - 6 + row] for row in range(6)], dtype=torch.int64),
        "compact": torch.arange(6, dtype=torch.int16),
        "bfloat": torch.arange(12, dtype=torch.bfloat16).reshape(6, 2),
        "float_bits": bit_patterns.view(torch.float32),
    }


def _bits(value):
    return value.detach().to("cpu").contiguous().view(torch.uint8)


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(_bits(actual), _bits(expected))


def _replay():
    replay = _TD3DeviceReplay(6, "cpu")
    replay.extend(_rows())
    replay.valid_count("valid")
    return replay


def test_sampling_preserves_bits_field_order_duplicate_indices_and_replay_state(output_device):
    replay = _replay()
    fields = ("refs", "float_bits", "valid", "matrix", "vector", "pixels", "bfloat", "compact")
    # Deliberately noncontiguous and unsorted, with repeated physical rows.
    indices = torch.tensor([4, 99, 1, 99, 4, 99, 2, 99, 0, 99])[::2]
    index_before = indices.clone()
    data_before = {key: value.clone() for key, value in replay.data.items()}
    counters = replay.ptr, replay.size, replay.seen
    valid_indices = replay._valid_index_cache["valid"]
    rng_before = torch.get_rng_state().clone()

    sampled = replay.sample_by_indices(indices, output_device, fields=fields)

    assert tuple(sampled) == fields
    for key in fields:
        _assert_exact(sampled[key], data_before[key].index_select(0, indices))
        assert sampled[key].device == torch.device(output_device)
        assert sampled[key].is_contiguous()
        assert not sampled[key].requires_grad
    assert torch.equal(indices, index_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert (replay.ptr, replay.size, replay.seen) == counters
    assert replay._valid_index_cache["valid"] is valid_indices
    for key, value in replay.data.items():
        _assert_exact(value, data_before[key])


def test_repeated_staging_with_same_buffer_shape_cannot_overwrite_previous_sample(output_device):
    replay = _replay()
    first_indices = torch.tensor([5, 1, 5, 0])
    first = replay.sample_by_indices(first_indices, output_device, fields=("vector", "pair", "refs", "valid"))
    first_snapshot = {key: value.clone() for key, value in first.items()}
    staging_pointers = {key: value.data_ptr() for key, value in replay._pinned_sample_staging.items()}

    # Same row count and per-dtype total size, but a different field order and
    # different rows. Reuse must repack each field into its current offset.
    second_indices = torch.tensor([2, 3, 1, 2])
    second = replay.sample_by_indices(second_indices, output_device, fields=("pair", "vector", "refs", "valid"))

    for key, value in first.items():
        _assert_exact(value, first_snapshot[key])
    for key, value in second.items():
        _assert_exact(value, replay.data[key].index_select(0, second_indices))
    if torch.device(output_device).type == "cuda":
        assert staging_pointers
        assert staging_pointers == {
            key: value.data_ptr() for key, value in replay._pinned_sample_staging.items()
        }
        assert all(value.is_pinned() for value in replay._pinned_sample_staging.values())
    second["vector"].fill_(12345)
    _assert_exact(first["vector"], first_snapshot["vector"])
    _assert_exact(replay.data["vector"], _rows()["vector"])


def test_sampling_after_ring_wrap_uses_physical_indices_and_returns_owned_values(output_device):
    replay = _replay()
    replacement = {key: value[:2].clone() for key, value in _rows().items()}
    replacement["vector"].add_(1000)
    replay.extend(replacement)
    assert replay.ptr == 2 and replay.size == 6 and replay.seen == 8
    indices = torch.tensor([0, 5, 1, 0])
    expected = {key: value.index_select(0, indices) for key, value in replay.data.items()}

    sampled = replay.sample_by_indices(indices, output_device)

    assert tuple(sampled) == tuple(replay.data)
    for key, value in sampled.items():
        _assert_exact(value, expected[key])
    replay.clear()
    assert replay._pinned_sample_staging == {}
    assert replay._valid_index_cache == {}
    assert (replay.ptr, replay.size, replay.seen) == (0, 0, 0)
    for key, value in sampled.items():
        _assert_exact(value, expected[key])


def test_inference_first_sample_staging_can_be_reused_under_no_grad(output_device):
    replay = _replay()
    first_indices = torch.tensor([5, 1, 3])
    with torch.inference_mode():
        first = replay.sample_by_indices(first_indices, output_device)
    snapshots = {key: value.clone() for key, value in first.items()}
    pointers = {key: value.data_ptr() for key, value in replay._pinned_sample_staging.items()}
    assert all(not value.is_inference() for value in replay._pinned_sample_staging.values())

    second_indices = torch.tensor([2, 0, 2])
    with torch.no_grad():
        second = replay.sample_by_indices(second_indices, output_device)

    assert pointers == {key: value.data_ptr() for key, value in replay._pinned_sample_staging.items()}
    for key in first:
        _assert_exact(first[key], snapshots[key])
        _assert_exact(second[key], replay.data[key].index_select(0, second_indices))


@pytest.mark.parametrize("indices,error", [
    ([0], TypeError),
    (torch.tensor([0], dtype=torch.int32), TypeError),
    (torch.tensor([[0]], dtype=torch.long), TypeError),
    (torch.empty(0, dtype=torch.long), ValueError),
    (torch.tensor([-1]), (IndexError, RuntimeError)),
    (torch.tensor([6]), (IndexError, RuntimeError)),
    (torch.tensor([2**63 - 1]), (IndexError, RuntimeError)),
])
def test_invalid_sampling_indices_fail_without_changing_replay(indices, error):
    replay = _replay()
    before = {key: value.clone() for key, value in replay.data.items()}
    counters = replay.ptr, replay.size, replay.seen

    with pytest.raises(error):
        replay.sample_by_indices(indices, "cpu")

    assert (replay.ptr, replay.size, replay.seen) == counters
    for key, value in replay.data.items():
        _assert_exact(value, before[key])


def test_field_selection_and_empty_replay_errors_preserve_existing_contract():
    replay = _replay()
    indices = torch.tensor([2, 0])
    assert replay.sample_by_indices(indices, "cpu", fields=()) == {}
    with pytest.raises(KeyError, match="Unknown DAgger replay sample fields"):
        replay.sample_by_indices(indices, "cpu", fields=("vector", "missing"))
    replay.clear()
    with pytest.raises(RuntimeError, match="empty DAgger replay"):
        replay.sample_by_indices(indices, "cpu")


def test_cpu_packed_gather_preserves_mixed_dtype_bits_and_aligned_field_views():
    data = _rows()
    fields = ("valid", "refs", "float_bits", "matrix", "pixels", "bfloat", "compact", "vector")
    indices = torch.tensor([4, 99, 1, 99, 4, 99, 2, 99, 0, 99])[::2]
    source_before = {key: value.clone() for key, value in data.items()}
    index_before = indices.clone()
    layout, total_bytes = _packed_replay_layout(data, fields, indices.numel())
    backing = torch.full((total_bytes + 32,), 0xAC, dtype=torch.uint8)
    staging = backing[:total_bytes]
    rng_before = torch.get_rng_state().clone()

    _gather_packed_replay(data, indices, layout, staging)
    packed = staging.clone()  # CPU stand-in for the independently owned H2D result.
    sampled = _unpack_replay_sample(packed, layout)

    assert tuple(sampled) == fields
    assert torch.equal(backing[total_bytes:], torch.full((32,), 0xAC, dtype=torch.uint8))
    for key in fields:
        _assert_exact(sampled[key], source_before[key].index_select(0, indices))
        assert sampled[key].is_contiguous()
        assert sampled[key].data_ptr() % sampled[key].element_size() == 0
        assert sampled[key].untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    staging.zero_()
    for key in fields:
        _assert_exact(sampled[key], source_before[key].index_select(0, indices))
    for key in data:
        _assert_exact(data[key], source_before[key])
    assert torch.equal(indices, index_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    # Duplicate selections are separate rows, with no source or row alias.
    duplicate_before = sampled["vector"][2].clone()
    sampled["vector"][0].fill_(999)
    _assert_exact(sampled["vector"][2], duplicate_before)
    _assert_exact(data["vector"], source_before["vector"])


def test_cpu_packed_gather_reuses_storage_with_new_rows_and_same_sized_field_order():
    data = _rows()
    first_indices = torch.tensor([5, 1, 5, 0])
    second_indices = torch.tensor([2, 3, 1, 2])
    first_fields = ("vector", "pair", "refs", "valid")
    second_fields = ("pair", "vector", "refs", "valid")
    first_layout, total_bytes = _packed_replay_layout(data, first_fields, first_indices.numel())
    second_layout, second_bytes = _packed_replay_layout(data, second_fields, second_indices.numel())
    assert total_bytes == second_bytes
    staging = torch.empty(total_bytes, dtype=torch.uint8)
    pointer = staging.data_ptr()
    _gather_packed_replay(data, first_indices, first_layout, staging)
    first = _unpack_replay_sample(staging.clone(), first_layout)
    snapshots = {key: value.clone() for key, value in first.items()}

    _gather_packed_replay(data, second_indices, second_layout, staging)
    second = _unpack_replay_sample(staging.clone(), second_layout)

    assert staging.data_ptr() == pointer
    assert tuple(second) == second_fields
    for key, value in first.items():
        _assert_exact(value, snapshots[key])
    for key, value in second.items():
        _assert_exact(value, data[key].index_select(0, second_indices))


def test_cpu_packed_gather_handles_noncontiguous_fields_and_bfloat_fallback():
    source = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    data = {
        "strided": source[:, ::2],
        "transposed": torch.arange(36, dtype=torch.float64).reshape(6, 2, 3).transpose(1, 2),
        "bfloat": torch.arange(18, dtype=torch.bfloat16).reshape(6, 3),
    }
    indices = torch.tensor([5, 0, 3, 5])
    layout, total_bytes = _packed_replay_layout(data, tuple(data), indices.numel())
    staging = torch.empty(total_bytes, dtype=torch.uint8)

    _gather_packed_replay(data, indices, layout, staging)
    sampled = _unpack_replay_sample(staging, layout)

    for key, value in sampled.items():
        _assert_exact(value, data[key].index_select(0, indices))
        assert value.is_contiguous()


@pytest.mark.parametrize("indices", [torch.tensor([5, 1, 5]), torch.empty(0, dtype=torch.long)])
def test_cpu_packed_layout_handles_empty_fields_rows_and_complex_alignment(indices):
    real = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    data = {
        "flag": torch.tensor([True, False, True, False, True, False]),
        "empty": torch.empty(6, 0, 3, dtype=torch.float32),
        "complex": torch.complex(real, -real),
        "scalar": torch.arange(6, dtype=torch.float64),
    }
    fields = ("flag", "empty", "complex", "complex", "scalar")
    layout, total_bytes = _packed_replay_layout(data, fields, indices.numel())
    staging = torch.empty(total_bytes, dtype=torch.uint8)

    _gather_packed_replay(data, indices, layout, staging)
    sampled = _unpack_replay_sample(staging, layout)

    assert tuple(sampled) == ("flag", "empty", "complex", "scalar")
    for key, value in sampled.items():
        _assert_exact(value, data[key].index_select(0, indices))
    if indices.numel():
        assert sampled["complex"].data_ptr() % sampled["complex"].element_size() == 0


@pytest.mark.parametrize("reported_threads", [1, 4, 5, 24])
def test_packed_dispatch_preserves_bits_with_small_or_large_thread_teams(monkeypatch, reported_threads):
    rows = _rows()
    data = {key: rows[key] for key in ("float_bits", "refs", "valid", "bfloat")}
    data["strided"] = torch.arange(36, dtype=torch.float32).reshape(6, 6)[:, ::2]
    indices = torch.tensor([4, 2, 1, 4, 0])
    expected = {key: value.index_select(0, indices) for key, value in data.items()}
    layout, total_bytes = _packed_replay_layout(data, tuple(data), indices.numel())
    staging = torch.empty(total_bytes, dtype=torch.uint8)
    calls = {"numpy": 0, "torch": 0}
    original_take = staging_module.np.take
    original_index_select = torch.index_select

    def numpy_take(*args, **kwargs):
        calls["numpy"] += 1
        return original_take(*args, **kwargs)

    def torch_take(*args, **kwargs):
        calls["torch"] += 1
        return original_index_select(*args, **kwargs)

    # Select both production dispatch branches without creating extra CPU
    # threads or changing the process-wide execution setting during the test.
    monkeypatch.setattr(torch, "get_num_threads", lambda: reported_threads)
    monkeypatch.setattr(staging_module.np, "take", numpy_take)
    monkeypatch.setattr(torch, "index_select", torch_take)

    _gather_packed_replay(data, indices, layout, staging)
    actual = _unpack_replay_sample(staging, layout)

    for key in data:
        _assert_exact(actual[key], expected[key])
    # Unsupported bfloat16 and strided sources retain the Torch fallback even
    # when the larger reported thread count selects NumPy for other fields.
    assert calls == ({"numpy": 3, "torch": 2} if reported_threads > 4
                     else {"numpy": 0, "torch": 5})


@pytest.mark.parametrize("indices", [torch.tensor([-1]), torch.tensor([6]), torch.tensor([2**63 - 1])])
def test_cpu_packed_gather_rejects_out_of_storage_indices_instead_of_clipping(indices):
    data = _rows()
    before = {key: value.clone() for key, value in data.items()}
    layout, total_bytes = _packed_replay_layout(data, tuple(data), indices.numel())
    staging = torch.empty(total_bytes, dtype=torch.uint8)

    with pytest.raises((IndexError, RuntimeError)):
        _gather_packed_replay(data, indices, layout, staging)

    for key in data:
        _assert_exact(data[key], before[key])
