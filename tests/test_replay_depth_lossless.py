from __future__ import annotations

import pytest
import torch

from active_adaptation.learning.ppo.td3_bc_dagger import (
    _decode_replay_depth_u8,
    _encode_replay_depth_u8,
)


def test_all_task_depth_bins_round_trip_without_changing_input():
    # The sensor's output arithmetic is integer-valued float / num_bins.
    depth = (torch.arange(101, dtype=torch.float32) / 100).reshape(1, 1, 1, 101)
    original = depth.clone()

    encoded = _encode_replay_depth_u8(depth)

    assert encoded.dtype == torch.uint8
    assert torch.equal(_decode_replay_depth_u8(encoded), original)
    assert torch.equal(depth, original)


def test_sensor_floor_and_pixel_dropout_remain_lossless():
    depth = torch.linspace(0, 1, 1001, dtype=torch.float32)
    depth = (depth * 100).floor_().clamp_(0, 100) / 100
    mask = (torch.arange(depth.numel()) % 3 != 0).to(depth.dtype)
    depth = depth * mask

    assert torch.equal(_decode_replay_depth_u8(_encode_replay_depth_u8(depth)), depth)


@pytest.mark.parametrize("direction", [0.0, 1.0])
def test_nearest_float_to_grid_is_rejected_instead_of_silently_rounded(direction):
    depth = torch.tensor([0.4], dtype=torch.float32)
    depth = torch.nextafter(depth, torch.full_like(depth, direction))
    original = depth.clone()
    scaled = depth * 100
    tolerance = 4 * torch.finfo(depth.dtype).eps * 100
    # This is the previous codec's acceptance test; it allowed changed values.
    assert torch.allclose(scaled, scaled.round(), rtol=0, atol=tolerance)

    with pytest.raises(ValueError, match="cannot be encoded losslessly"):
        _encode_replay_depth_u8(depth)

    assert torch.equal(depth, original)


@pytest.mark.parametrize("depth", [torch.tensor([-1e-8]), torch.tensor([1.0 + 1e-7])])
def test_near_boundary_values_are_rejected_instead_of_clipped(depth):
    with pytest.raises(ValueError, match="cannot be encoded losslessly"):
        _encode_replay_depth_u8(depth)
