from __future__ import annotations

import pytest
import torch

from active_adaptation.learning.ppo.replay_provenance import (
    _ReplayProvenanceBatch,
    _packed_provenance_cpu,
    prepared_with_provenance_snapshot,
    replay_provenance_cpu,
)


KEYS = ("teacher", "dagger", "physical")


def _batch():
    return {
        "observations": torch.arange(12).reshape(3, 4),
        "teacher": torch.tensor([True, False, False]),
        "dagger": torch.tensor([False, True, False]),
        "physical": torch.tensor([7, 9, 9]),
    }


def test_prepared_snapshot_preserves_keys_order_values_and_public_tensor_identity():
    source = _batch()
    prepared, first = prepared_with_provenance_snapshot(dict(source), source, KEYS)
    assert isinstance(prepared, dict)
    assert list(prepared) == list(source)
    assert all(prepared[key] is value for key, value in source.items())
    assert replay_provenance_cpu(prepared, KEYS) is first
    # Ordinary dictionary copies deliberately shed the private snapshot.
    assert type(dict(prepared)) is dict
    assert not hasattr(dict(prepared), "_provenance_snapshot")


@pytest.mark.parametrize("change", ["mutate", "replace", "resize", "dtype"])
def test_snapshot_invalidates_when_metadata_changes(change):
    source = _batch()
    prepared, first = prepared_with_provenance_snapshot(source, source, KEYS)
    if change == "mutate":
        prepared["physical"][0] = 2
    elif change == "replace":
        prepared["physical"] = prepared["physical"].clone()
    elif change == "resize":
        prepared["physical"].resize_(1)
    else:
        prepared["physical"] = prepared["physical"].float()
    second = replay_provenance_cpu(prepared, KEYS)
    assert second is not first
    assert second[2].dtype == prepared["physical"].dtype
    assert torch.equal(second[2], prepared["physical"].reshape(-1))


def test_inference_metadata_is_never_reused():
    with torch.inference_mode():
        source = _batch()
        prepared, first = prepared_with_provenance_snapshot(source, source, KEYS)
        assert prepared._provenance_snapshot is None
        prepared["physical"][0] = 2
        second = replay_provenance_cpu(prepared, KEYS)
    assert second is not first
    assert second[2].tolist() == [2, 9, 9]


def test_snapshot_invalidates_same_tensor_data_storage_replacement():
    source = _batch()
    prepared, first = prepared_with_provenance_snapshot(source, source, KEYS)
    indices = prepared["physical"]
    old_version = indices._version
    old_pointer = indices.data_ptr()
    indices.data = torch.tensor([2, 4, 6])
    assert prepared["physical"] is indices
    assert indices._version == old_version
    assert indices.data_ptr() != old_pointer
    second = replay_provenance_cpu(prepared, KEYS)
    assert second is not first
    assert first[2].tolist() == [7, 9, 9]
    assert second[2].tolist() == [2, 4, 6]


def test_snapshot_invalidates_same_storage_data_stride_replacement():
    source = _batch()
    indices = torch.tensor([7, 8, 9, 10, 11, 12])
    source["physical"] = indices[:3]
    prepared, first = prepared_with_provenance_snapshot(source, source, KEYS)
    metadata = prepared["physical"]
    old_version, old_pointer = metadata._version, metadata.data_ptr()
    metadata.data = indices[::2]
    assert metadata._version == old_version
    assert metadata.data_ptr() == old_pointer
    second = replay_provenance_cpu(prepared, KEYS)
    assert second is not first
    assert second[2].tolist() == [7, 9, 11]


def test_changed_or_projected_preparation_does_not_attach_source_snapshot():
    source = _batch()
    changed = dict(source)
    changed["physical"] = torch.tensor([1, 2, 3])
    prepared, source_snapshot = prepared_with_provenance_snapshot(changed, source, KEYS)
    assert prepared._provenance_snapshot is None
    actual = replay_provenance_cpu(prepared, KEYS)
    assert actual is not source_snapshot
    assert actual[2].tolist() == [1, 2, 3]
    projected, _ = prepared_with_provenance_snapshot({"observations": source["observations"]}, source, KEYS)
    assert projected._provenance_snapshot is None
    assert list(projected) == ["observations"]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_packed_copy_preserves_integer_precision_duplicates_and_uses_one_cpu_call(monkeypatch, dtype, empty):
    maximum = torch.iinfo(dtype).max
    values = (
        torch.tensor([True, False, True]),
        torch.tensor([False, True, False]),
        torch.tensor([maximum, maximum - 1, maximum], dtype=dtype),
    )
    if empty:
        values = tuple(value[:0] for value in values)
    calls = []
    original = torch.Tensor.cpu

    def counted_cpu(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    actual = _packed_provenance_cpu(values)
    assert len(calls) == 1
    for result, expected in zip(actual, values):
        assert result.dtype == expected.dtype
        assert torch.equal(result, expected)


def test_cpu_fallback_preserves_invalid_dtypes_and_missing_key_errors():
    source = _ReplayProvenanceBatch(_batch())
    source["teacher"] = source["teacher"].int()
    source["physical"] = source["physical"].float()
    actual = replay_provenance_cpu(source, KEYS)
    assert actual[0].dtype == torch.int32
    assert actual[2].dtype == torch.float32
    del source["physical"]
    with pytest.raises(KeyError, match="physical"):
        replay_provenance_cpu(source, KEYS)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_prepared_snapshot_copies_once_and_recopies_mutated_metadata(monkeypatch, index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA provenance transfer requires an available GPU")
    maximum = torch.iinfo(index_dtype).max
    cpu_source = _batch()
    cpu_source["physical"] = torch.tensor([maximum, maximum - 1, maximum], dtype=index_dtype)
    source = {key: value.cuda() for key, value in cpu_source.items()}
    copies = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(value, *args, **kwargs):
        if value.device.type == "cuda":
            copies.append((tuple(value.shape), value.dtype))
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)

    prepared, first = prepared_with_provenance_snapshot(dict(source), source, KEYS)

    assert copies == [((3, 3), index_dtype)]
    assert all(prepared[key] is value for key, value in source.items())
    assert replay_provenance_cpu(prepared, KEYS) is first
    assert replay_provenance_cpu(prepared, KEYS) is first
    assert len(copies) == 1
    for actual, key in zip(first, KEYS):
        assert actual.dtype == cpu_source[key].dtype
        assert actual.device.type == "cpu"
        assert torch.equal(actual, cpu_source[key])

    prepared["physical"][0] = maximum - 2
    second = replay_provenance_cpu(prepared, KEYS)

    assert second is not first
    assert copies == [((3, 3), index_dtype), ((3, 3), index_dtype)]
    assert first[2].tolist() == [maximum, maximum - 1, maximum]
    assert second[2].tolist() == [maximum - 2, maximum - 1, maximum]
    assert replay_provenance_cpu(prepared, KEYS) is second
    assert len(copies) == 2
