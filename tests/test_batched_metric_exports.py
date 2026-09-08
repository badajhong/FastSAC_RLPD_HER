"""Metric transport must not change reductions, scalar values, or types."""

import math
import struct

import pytest
import torch

from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC,
    _scalar_metrics_to_python,
)


def _assert_same_python_scalar(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        # Python's double representation catches signed zero and all mantissa
        # bits. For NaNs the relevant contract is a preserved NaN value.
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert struct.pack("!d", actual) == struct.pack("!d", expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mean_metrics_preserve_original_reduction_bits_and_one_transfer(dtype, monkeypatch):
    metrics = [
        {
            "vector": torch.tensor([0.1001, -0.5003, 0.8757], dtype=dtype),
            "scalar": torch.tensor([(-1) ** index * (index + 0.12345)], dtype=dtype),
            "integer": torch.tensor([2**24 + index], dtype=torch.int64),
            "python": index + 0.2345,
        }
        for index in range(7)
    ]
    keys = ("integer", "vector", "python", "scalar")
    expected = {
        key: torch.stack(
            [torch.as_tensor(item[key]).detach().float() for item in metrics]
        ).mean().item()
        for key in keys
    }
    transfers = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        transfers.append((tensor.device, tensor.dtype, tensor.numel()))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    actual = DistributionalTD3TeacherBC._mean_metric_dict(metrics, iter(keys))
    assert list(actual) == list(keys)
    for key in keys:
        _assert_same_python_scalar(actual[key], expected[key])
    assert transfers == [(torch.device("cpu"), torch.float32, len(keys))]


def test_scalar_metric_export_groups_without_promotion_or_type_changes(monkeypatch):
    metrics = {
        "fp64": torch.tensor(1 + 2**-40, dtype=torch.float64),
        "integer": torch.tensor(2**60 + 1, dtype=torch.int64),
        "negative_zero": torch.tensor(-0.0, dtype=torch.float32),
        "positive_zero": torch.tensor(0.0, dtype=torch.float32),
        "bool": torch.tensor(True),
        "fp16": torch.tensor(0.12345, dtype=torch.float16),
        "bf16": torch.tensor(-0.12345, dtype=torch.bfloat16),
        "nan": torch.tensor(float("nan")),
        "infinity": torch.tensor(float("inf")),
        "complex": torch.tensor(1 + 2j, dtype=torch.complex128),
        "python_int": 2**62 + 3,
        "python_float": 1 + 2**-45,
        "python_bool": False,
        "parameter": torch.nn.Parameter(torch.tensor([0.2345])),
    }
    expected = {
        key: value.item() if isinstance(value, torch.Tensor) else value
        for key, value in metrics.items()
    }
    transfers = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        assert not tensor.requires_grad
        transfers.append((tensor.device, tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)

    def forbidden_item(tensor, *args, **kwargs):
        raise AssertionError("Export must not synchronize separately for each scalar")

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    actual = _scalar_metrics_to_python(metrics)
    assert list(actual) == list(metrics)
    for key in metrics:
        _assert_same_python_scalar(actual[key], expected[key])
    expected_groups = {
        (value.device, value.dtype)
        for value in metrics.values()
        if isinstance(value, torch.Tensor)
    }
    assert len(transfers) == len(expected_groups)
    assert set(transfers) == expected_groups


def test_empty_metrics_require_no_tensor_transfers(monkeypatch):
    def forbidden_cpu(tensor, *args, **kwargs):
        raise AssertionError("An empty metric group needs no transfer")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    assert _scalar_metrics_to_python({}) == {}
    assert DistributionalTD3TeacherBC._mean_metric_dict([], ["b", "a"]) == {
        "b": 0.0,
        "a": 0.0,
    }
    assert DistributionalTD3TeacherBC._mean_metric_dict([{}], []) == {}


def test_scalar_export_rejects_non_scalar_metrics():
    with pytest.raises(RuntimeError):
        _scalar_metrics_to_python({"not_reduced": torch.zeros(2)})
