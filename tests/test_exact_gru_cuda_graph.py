"""CUDA graph replay must execute the same GRU operators and current weights."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from active_adaptation.learning.ppo import exact_gru_cuda_graph as graph_module
from active_adaptation.learning.ppo.exact_gru_cuda_graph import (
    exact_gru_cuda_graphs,
    run_exact_gru_sequence,
)
from active_adaptation.learning.ppo.ppo_vel import (
    GRU,
    exact_recurrent_lengths,
    set_recurrent_mode,
)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph correctness requires an available GPU")
    return torch.device("cuda", torch.cuda.current_device())


def _core(device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(173)
        return GRU(4, 5).to(device).eval()


def _inputs(device="cpu", *, time=12, seed=179):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(3, time, 4, generator=generator).to(device)
    reset = torch.zeros(3, time, 1, dtype=torch.bool, device=device)
    reset[0, 0] = True
    reset[1, time // 2] = True
    # Match exact replay: one compact raw carry expanded across the sequence.
    hidden = torch.randn(3, 5, generator=generator).to(device)
    return x, reset, hidden.unsqueeze(1).expand(-1, time, -1)


def _reference_raw(core, x, reset, hidden):
    sequence = []
    reset = 1.0 - reset.float().reshape(x.shape[0], x.shape[1], 1)
    for step in range(x.shape[1]):
        hidden = core.gru(x[:, step], hidden * reset[:, step])
        if core.burn_in and step < x.shape[1] // 4:
            hidden = hidden.detach()
        sequence.append(hidden)
    return torch.stack(sequence, dim=1), hidden


def _reference(core, inputs, lengths=None):
    x, reset, hidden = inputs
    output, final = _reference_raw(core, x, reset, hidden[:, 0])
    if lengths is not None:
        final = output[torch.arange(x.shape[0], device=x.device),
                       torch.tensor(lengths, device=x.device) - 1]
    return core.ln(output), final.unsqueeze(1).expand(-1, x.shape[1], -1)


def _forward(core, inputs, *, enabled=True, lengths=None):
    with torch.no_grad(), set_recurrent_mode(True), exact_gru_cuda_graphs(enabled):
        context = nullcontext() if lengths is None else exact_recurrent_lengths(
            lengths, torch.tensor(lengths, device=inputs[0].device)
        )
        with context:
            return core(*inputs)


def _capture(core, inputs):
    for _ in range(3):
        actual = _forward(core, inputs)
    cache = core._exact_gru_graph_cache
    assert cache.captures == 1
    assert len(cache.entries) == 1
    return actual, cache


def test_graph_context_restores_nested_and_exception_scopes():
    assert not graph_module._ENABLED.get()
    with exact_gru_cuda_graphs():
        assert graph_module._ENABLED.get()
        with pytest.raises(RuntimeError, match="scope failure"):
            with exact_gru_cuda_graphs(False):
                assert not graph_module._ENABLED.get()
                raise RuntimeError("scope failure")
        assert graph_module._ENABLED.get()
    assert not graph_module._ENABLED.get()


def test_cpu_inference_remains_eager_and_bitwise_equal():
    core = _core()
    inputs = _inputs()
    with torch.no_grad():
        expected = _reference(core, inputs)
    for _ in range(4):
        actual = _forward(core, inputs)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert not hasattr(core, "_exact_gru_graph_cache")


def test_grad_enabled_scope_preserves_original_outputs_and_gradients():
    core = _core()
    reference = copy.deepcopy(core)
    inputs = _inputs()
    with set_recurrent_mode(True), exact_gru_cuda_graphs():
        actual = core(*inputs)
    expected = _reference(reference, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    for parameter, original in zip(core.parameters(), reference.parameters()):
        assert torch.equal(parameter.grad, original.grad)
    assert not hasattr(core, "_exact_gru_graph_cache")


@pytest.mark.parametrize("lengths", [None, (1, 7, 12), (12, 12, 12)])
def test_cuda_raw_and_normalized_outputs_match_with_padding_resets_and_ema(cuda_device, lengths):
    core = _core(cuda_device).requires_grad_(False)
    first = _inputs(cuda_device)
    _capture(core, first)
    cache = core._exact_gru_graph_cache
    previous_output = None
    for case in range(3):
        inputs = _inputs(cuda_device, seed=179 + case)
        if case == 2:
            with torch.no_grad():
                for parameter in core.parameters():
                    parameter.add_(0.03125)
        with torch.no_grad():
            expected_raw = _reference_raw(core, inputs[0], inputs[1], inputs[2][:, 0])
            expected = _reference(core, inputs, lengths)
            with exact_gru_cuda_graphs():
                actual_raw = run_exact_gru_sequence(
                    core, inputs[0], inputs[1], inputs[2][:, 0], core._sequence_raw
                )
        actual = _forward(core, inputs, lengths=lengths)
        assert all(torch.equal(a, b) for a, b in zip(actual_raw, expected_raw))
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert cache.captures == 1
        if previous_output is not None:
            assert not torch.equal(actual[0], previous_output)
        previous_output = actual[0].clone()
    assert cache.replays >= 7


def test_cuda_replay_copies_outputs_and_reads_mutated_input_storage(cuda_device):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device)
    _capture(core, inputs)
    with torch.no_grad(), exact_gru_cuda_graphs():
        previous = run_exact_gru_sequence(
            core, inputs[0], inputs[1], inputs[2][:, 0], core._sequence_raw
        )
        snapshot = tuple(value.clone() for value in previous)
        inputs[0].add_(0.5)
        inputs[1].logical_not_()
        inputs[2][:, 0].add_(0.25)
        current = run_exact_gru_sequence(
            core, inputs[0], inputs[1], inputs[2][:, 0], core._sequence_raw
        )
        expected = _reference_raw(core, inputs[0], inputs[1], inputs[2][:, 0])
    assert all(torch.equal(a, b) for a, b in zip(previous, snapshot))
    assert all(torch.equal(a, b) for a, b in zip(current, expected))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(previous, current))
    assert not torch.equal(previous[0], current[0])


@pytest.mark.parametrize("replacement", ["parameter", "storage"])
def test_cuda_parameter_storage_replacement_invalidates_capture(cuda_device, replacement):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device)
    _, cache = _capture(core, inputs)
    previous_entry = next(iter(cache.entries.values()))
    new_storage = core.gru.weight_ih.detach().clone() + 0.125
    if replacement == "storage":
        core.gru.weight_ih.data = new_storage
    else:
        core.gru.weight_ih = nn.Parameter(new_storage, requires_grad=False)

    actual = _forward(core, inputs)

    assert not cache.entries
    with torch.no_grad():
        expected = _reference(core, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    _forward(core, inputs)
    actual = _forward(core, inputs)
    assert cache.captures == 2
    assert next(iter(cache.entries.values())) is not previous_entry
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_cuda_graphs_are_absent_from_state_dict_and_discarded_by_deepcopy(cuda_device):
    core = _core(cuda_device).requires_grad_(False)
    before = {name: value.clone() for name, value in core.state_dict().items()}
    inputs = _inputs(cuda_device)
    actual, cache = _capture(core, inputs)
    assert core.state_dict().keys() == before.keys()
    assert all(torch.equal(value, before[name]) for name, value in core.state_dict().items())

    clone = copy.deepcopy(core)

    assert clone._exact_gru_graph_cache is not cache
    assert not clone._exact_gru_graph_cache.entries
    assert not clone._exact_gru_graph_cache.seen
    assert clone._exact_gru_graph_cache.captures == clone._exact_gru_graph_cache.replays == 0
    copied = _forward(clone, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, copied))
    clone.load_state_dict(before)
    assert all(torch.equal(value, before[name]) for name, value in clone.state_dict().items())


def test_cuda_graph_and_unseen_shape_bookkeeping_are_bounded(cuda_device):
    core = _core(cuda_device).requires_grad_(False)
    for length in range(8, 13):
        inputs = _inputs(cuda_device, time=length)
        for _ in range(3):
            actual = _forward(core, inputs)
        with torch.no_grad():
            expected = _reference(core, inputs)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    cache = core._exact_gru_graph_cache
    assert len(cache.entries) == cache.captures == cache.max_graphs == 4
    assert len(cache.seen) <= 32
    uncaptured = _core(cuda_device).requires_grad_(False)
    for length in range(8, 44):
        _forward(uncaptured, _inputs(cuda_device, time=length))
    assert not uncaptured._exact_gru_graph_cache.entries
    assert len(uncaptured._exact_gru_graph_cache.seen) <= 32


@pytest.mark.parametrize("hook_kind", ["forward", "pre", "global_forward", "global_pre"])
def test_cuda_hooks_added_after_capture_still_run_on_every_timestep(cuda_device, hook_kind):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device)
    _, cache = _capture(core, inputs)
    replay_count = cache.replays
    calls = []

    def forward_hook(module, args, output):
        if module is core.gru:
            calls.append(True)
            return output + 0.125

    def pre_hook(module, args):
        if module is core.gru:
            calls.append(True)
            return args[0] + 0.125, args[1]

    register, hook = {
        "forward": (core.gru.register_forward_hook, forward_hook),
        "pre": (core.gru.register_forward_pre_hook, pre_hook),
        "global_forward": (nn.modules.module.register_module_forward_hook, forward_hook),
        "global_pre": (nn.modules.module.register_module_forward_pre_hook, pre_hook),
    }[hook_kind]
    handle = register(hook)
    try:
        actual = _forward(core, inputs)
        assert len(calls) == inputs[0].shape[1]
        with torch.no_grad():
            expected = _reference(core, inputs)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    finally:
        handle.remove()
    assert cache.replays == replay_count


def test_cuda_grad_enabled_call_does_not_replay_an_existing_graph(cuda_device):
    core = _core(cuda_device)
    inputs = _inputs(cuda_device)
    _, cache = _capture(core, inputs)
    replay_count = cache.replays
    reference = copy.deepcopy(core)
    with set_recurrent_mode(True), exact_gru_cuda_graphs():
        actual = core(*inputs)
    expected = _reference(reference, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    assert cache.replays == replay_count
    for parameter, original in zip(core.parameters(), reference.parameters()):
        assert torch.equal(parameter.grad, original.grad)


@pytest.mark.parametrize("reason", ["disabled", "short", "double", "overlap", "subclass"])
def test_cuda_unsupported_paths_keep_eager_semantics(cuda_device, reason):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device, time=4 if reason == "short" else 12)
    if reason == "double":
        core.double()
        inputs = inputs[0].double(), inputs[1], inputs[2].double()
    elif reason == "overlap":
        inputs = inputs[0], inputs[1][:, :1].expand_as(inputs[1]), inputs[2]
    elif reason == "subclass":
        class CustomCell(nn.GRUCell):
            pass
        core.gru.__class__ = CustomCell
    for _ in range(4):
        actual = _forward(core, inputs, enabled=reason != "disabled")
    with torch.no_grad():
        expected = _reference(core, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert not hasattr(core, "_exact_gru_graph_cache")


def test_cuda_capture_oom_falls_back_without_retrying_each_chunk(cuda_device, monkeypatch):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device)
    attempts = []

    def fail_capture(*args, **kwargs):
        attempts.append(True)
        raise torch.cuda.OutOfMemoryError("simulated capture allocation failure")

    monkeypatch.setattr(graph_module, "_GraphEntry", fail_capture)
    for _ in range(5):
        actual = _forward(core, inputs)
    with torch.no_grad():
        expected = _reference(core, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert len(attempts) == 1
    assert core._exact_gru_graph_cache.capture_failed
    assert not core._exact_gru_graph_cache.entries


def test_cuda_graph_captured_in_inference_mode_replays_in_no_grad(cuda_device):
    core = _core(cuda_device).requires_grad_(False)
    with torch.inference_mode():
        first = _inputs(cuda_device)
        _capture(core, first)
    inputs = _inputs(cuda_device, seed=193)

    actual = _forward(core, inputs)

    with torch.no_grad():
        expected = _reference(core, inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert core._exact_gru_graph_cache.captures == 1


def test_cuda_graph_padding_nan_cannot_replace_true_final_hidden(cuda_device):
    core = _core(cuda_device).requires_grad_(False)
    inputs = _inputs(cuda_device)
    _capture(core, inputs)
    lengths = (1, 7, 12)
    with torch.no_grad():
        expected = _reference(core, inputs, lengths)
        for row, length in enumerate(lengths):
            inputs[0][row, length:] = float("nan")

    actual = _forward(core, inputs, lengths=lengths)

    for row, length in enumerate(lengths):
        assert torch.equal(actual[0][row, :length], expected[0][row, :length])
    assert torch.equal(actual[1], expected[1])
    assert torch.isfinite(actual[1]).all()
