from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from active_adaptation.learning.ppo.common import FlattenBatch
from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule,
    GRU,
    GRUModule,
    TemporalDepthGRU,
    TransformObject,
    exact_recurrent_lengths,
    set_recurrent_mode,
)


class _PerceptionStack(nn.Module):
    """Real two-GRU path, including CNN, object transform and residual fusion."""

    def __init__(self, residual):
        super().__init__()
        self.depth = TemporalDepthGRU(
            FlattenBatch(
                nn.Sequential(
                    nn.Conv2d(1, 4, 3), nn.Mish(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                ),
                data_dim=3,
            ),
            hidden_dim=4,
        )
        self.object = nn.Linear(7, 12)
        self.transform = TransformObject(
            12, ["object_pred", "object_geo_"], ["object_pred_trans"]
        )
        self.adapt = DepthResidualGRUModule(8, 4) if residual else GRUModule(8)
        self.residual = residual
        if residual:
            with torch.no_grad():
                self.adapt.depth_projection.weight.normal_(0.0, 0.2)

    def forward(self, td):
        self.depth(td)
        td["object_pred"] = self.object(torch.cat([
            td["policy"], td["vel_command"], td["_depth_feature"],
        ], -1))
        self.transform(td)
        features = torch.cat([
            td["policy"], td["vel_command"], td["object_pred"], td["object_pred_trans"],
        ], -1)
        inputs = (features, td["is_init"], td["adapt_hx"])
        if self.residual:
            inputs += (td["_depth_feature"],)
        td["priv_pred"], td["next", "adapt_hx"] = self.adapt(*inputs)
        return td


def _input(batch=3, time=32):
    rng = torch.Generator().manual_seed(317)
    reset = torch.zeros(batch, time, 1, dtype=torch.bool)
    reset[:, 0] = True
    return TensorDict({
        "depth": torch.randn(batch, time, 1, 6, 6, generator=rng),
        "policy": torch.randn(batch, time, 2, generator=rng),
        "vel_command": torch.randn(batch, time, 1, generator=rng),
        "object_geo_": torch.randn(batch, time, 384, generator=rng),
        "is_init": reset,
        "depth_hx": torch.randn(batch, time, 4, generator=rng),
        "adapt_hx": torch.randn(batch, time, 8, generator=rng),
    }, [batch, time])


def _pipeline(residual):
    with torch.random.fork_rng():
        torch.manual_seed(319)
        model = _PerceptionStack(residual)
        with torch.no_grad(), set_recurrent_mode(True):
            model(_input(batch=1, time=1))
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("lengths", [(1, 31, 32), (32, 1, 31), (1, 1, 1), (32, 32, 32)])
def test_padded_perception_matches_unpadded_prefix_and_both_raw_hidden_states(residual, lengths):
    model = _pipeline(residual)
    raw = _input()
    weights = copy.deepcopy(model.state_dict())
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths(lengths, torch.tensor(lengths)):
            actual = model(raw.clone())
        for row, length in enumerate(lengths):
            expected = model(raw[row:row + 1, :length].clone())
            for key in ("_depth_feature", "object_pred", "object_pred_trans", "priv_pred"):
                torch.testing.assert_close(actual[key][row, :length], expected[key][0])
            for key in (("next", "depth_hx"), ("next", "adapt_hx")):
                torch.testing.assert_close(
                    actual[key][row], expected[key][0, -1].expand_as(actual[key][row])
                )
    assert weights.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])


@pytest.mark.parametrize("residual", [False, True])
def test_padded_true_final_states_continue_the_exact_full_history(residual):
    model = _pipeline(residual)
    lengths = [1, 31, 32]
    suffix_lengths = [4, 3, 1]
    raw = _input(time=36)
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths(lengths, torch.tensor(lengths)):
            prefix = model(raw[:, :32].clone())
        for row, (length, suffix_length) in enumerate(zip(lengths, suffix_lengths)):
            suffix = raw[row:row + 1, length:length + suffix_length].clone()
            for name in ("depth_hx", "adapt_hx"):
                suffix[name] = prefix["next", name][row, -1].expand_as(suffix[name]).clone()
            continued = model(suffix)
            expected = model(raw[row:row + 1, :length + suffix_length].clone())
            torch.testing.assert_close(continued["priv_pred"], expected["priv_pred"][:, length:])
            for name in ("depth_hx", "adapt_hx"):
                torch.testing.assert_close(
                    continued["next", name][:, -1], expected["next", name][:, -1]
                )


@pytest.mark.parametrize("residual", [False, True])
def test_context_is_bit_identical_for_full_lengths_and_restores_ordinary_behavior(residual):
    model = _pipeline(residual)
    raw = _input()
    with torch.no_grad(), set_recurrent_mode(True):
        before = model(raw.clone())
        with exact_recurrent_lengths([32] * 3, torch.tensor([32] * 3)):
            padded = model(raw.clone())
        after = model(raw.clone())
    for key in ("priv_pred", ("next", "adapt_hx"), ("next", "depth_hx")):
        assert torch.equal(before[key], padded[key])
        assert torch.equal(before[key], after[key])


def test_ordinary_training_outputs_and_gradients_match_the_original_gru():
    core = GRU(3, 5)
    reference = copy.deepcopy(core)
    inputs = _gru_inputs()
    inputs[1][0, 2] = True
    with set_recurrent_mode(True):
        actual, actual_hx = core(*inputs)
    x, reset, hx = inputs
    hx = hx[:, 0]
    sequence = []
    for time in range(x.shape[1]):
        hx = reference.gru(x[:, time], hx * (1.0 - reset[:, time]))
        if reference.burn_in and time < x.shape[1] // 4:
            hx = hx.detach()
        sequence.append(hx)
    expected = reference.ln(torch.stack(sequence, 1))
    expected_hx = hx.unsqueeze(1).expand_as(actual_hx)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_hx, expected_hx)
    (actual.square().sum() + actual_hx.square().sum()).backward()
    (expected.square().sum() + expected_hx.square().sum()).backward()
    for parameter, reference_parameter in zip(core.parameters(), reference.parameters()):
        assert torch.equal(parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize("residual", [False, True])
def test_nonfinite_trailing_padding_cannot_contaminate_real_outputs_or_final_states(residual):
    model = _pipeline(residual)
    raw = _input()
    padded = raw.clone()
    lengths = [1, 31, 32]
    for row, length in enumerate(lengths):
        for key in ("depth", "policy", "vel_command", "object_geo_"):
            padded[key][row, length:] = float("nan")
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths(lengths, torch.tensor(lengths)):
            actual = model(padded)
        for row, length in enumerate(lengths):
            expected = model(raw[row:row + 1, :length].clone())
            torch.testing.assert_close(actual["priv_pred"][row, :length], expected["priv_pred"][0])
            for name in ("depth_hx", "adapt_hx"):
                torch.testing.assert_close(
                    actual["next", name][row, -1], expected["next", name][0, -1]
                )


def _gru_inputs():
    return torch.randn(2, 4, 3), torch.zeros(2, 4, 1), torch.randn(2, 4, 5)


def test_nested_context_and_exception_restore_the_outer_lengths():
    core = GRU(3, 5)
    inputs = _gru_inputs()
    with torch.no_grad(), set_recurrent_mode(True):
        normal = core(*inputs)
        with exact_recurrent_lengths([1, 2], torch.tensor([1, 2])):
            outer = core(*inputs)
            with pytest.raises(RuntimeError, match="example failure"):
                with exact_recurrent_lengths([3, 4], torch.tensor([3, 4])):
                    inner = core(*inputs)
                    raise RuntimeError("example failure")
            restored = core(*inputs)
        after = core(*inputs)
    assert not torch.equal(outer[1], inner[1])
    assert torch.equal(outer[1], restored[1])
    assert torch.equal(normal[1], after[1])


def test_length_context_is_thread_local():
    core = GRU(3, 5).requires_grad_(False)
    inputs = _gru_inputs()

    def run_without_lengths():
        with torch.no_grad():
            return core(*inputs)[1]

    with ThreadPoolExecutor(max_workers=1) as pool:
        with torch.no_grad(), set_recurrent_mode(True):
            expected = core(*inputs)[1]
            with exact_recurrent_lengths([1, 1], torch.tensor([1, 1])):
                shortened = core(*inputs)[1]
                other_thread = pool.submit(run_without_lengths).result()
    assert not torch.equal(shortened, expected)
    assert torch.equal(other_thread, expected)


@pytest.mark.parametrize("lengths, tensor, error", [
    ([], torch.tensor([], dtype=torch.long), "positive CPU integers"),
    ([0, 2], torch.tensor([0, 2]), "positive CPU integers"),
    ([-1, 2], torch.tensor([-1, 2]), "positive CPU integers"),
    ([True, 2], torch.tensor([1, 2]), "positive CPU integers"),
    ([1.0, 2], torch.tensor([1, 2]), "positive CPU integers"),
    ([1, 2], torch.tensor([1.0, 2.0]), "1D torch.long"),
    ([1, 2], torch.tensor([[1, 2]]), "1D torch.long"),
    ([1, 2], torch.tensor([1]), "1D torch.long"),
    ([1, 2], torch.tensor([2, 1]), "must match"),
])
def test_invalid_lengths_fail_before_inference(lengths, tensor, error):
    with torch.no_grad(), set_recurrent_mode(True), pytest.raises(ValueError, match=error):
        with exact_recurrent_lengths(lengths, tensor):
            pytest.fail("invalid length context must not open")


@pytest.mark.parametrize("lengths", [[1], [1, 5]])
def test_lengths_must_match_the_actual_recurrent_batch(lengths):
    core = GRU(3, 5)
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths(lengths, torch.tensor(lengths)):
            with pytest.raises(ValueError, match=r"fit the \[N, T\] input batch"):
                core(*_gru_inputs())


def test_length_context_rejects_training_and_nonrecurrent_execution():
    with set_recurrent_mode(True), pytest.raises(RuntimeError, match="requires no_grad"):
        with exact_recurrent_lengths([1], torch.tensor([1])):
            pass
    with torch.no_grad(), set_recurrent_mode(False), pytest.raises(RuntimeError, match="recurrent mode"):
        with exact_recurrent_lengths([1], torch.tensor([1])):
            pass
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths([1, 2], torch.tensor([1, 2])):
            with torch.enable_grad(), pytest.raises(RuntimeError, match="requires no_grad"):
                GRU(3, 5)(*_gru_inputs())
            with set_recurrent_mode(False), pytest.raises(RuntimeError, match="recurrent mode"):
                GRU(3, 5)(*_gru_inputs())


def test_lengths_must_share_input_device_without_reading_device_values():
    # Metadata-only tensor models device mismatch without requiring a GPU or
    # calling .item()/.tolist() on the device tensor.
    core = GRU(3, 5)
    with torch.no_grad(), set_recurrent_mode(True):
        with exact_recurrent_lengths([1, 2], torch.empty(2, dtype=torch.long, device="meta")):
            with pytest.raises(ValueError, match="same device"):
                core(*_gru_inputs())
