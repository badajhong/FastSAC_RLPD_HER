"""Bounded CUDA graphs of the existing inference-only GRUCell sequence.

Only exact replay encoders opt in. Graphs cache execution, never latents or
old EMA weights: every call copies new inputs and reads the module's current
parameter storage. Padding selection and LayerNorm remain in the caller.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn


_ENABLED = ContextVar("exact_gru_cuda_graphs", default=False)


@contextmanager
def exact_gru_cuda_graphs(enabled=True):
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


class _GraphEntry:
    def __init__(self, eager, inputs, parameters):
        # Ordinary buffers remain writable even when the first caller was
        # under inference_mode. Preserve strides to keep the same GEMM path.
        with torch.inference_mode(False), torch.no_grad():
            self.inputs = tuple(
                torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                    device=value.device)
                for value in inputs
            )
            for target, source in zip(self.inputs, inputs):
                target.copy_(source)
            self.parameters = parameters  # Keep captured storage alive.
            stream = torch.cuda.Stream(device=inputs[0].device)
            current = torch.cuda.current_stream(inputs[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    eager(*self.inputs)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = eager(*self.inputs)

    def replay(self, inputs):
        with torch.inference_mode(False), torch.no_grad():
            for target, source in zip(self.inputs, inputs):
                target.copy_(source)
            self.graph.replay()
            # A later chunk may replay this graph while an earlier hidden or
            # Actor input is still retained. Never expose graph-owned outputs.
            return tuple(value.clone() for value in self.outputs)


class _GRUGraphCache:
    max_graphs = 4
    warmup_calls = 2

    def __init__(self):
        self.signature = None
        self.entries = {}
        self.seen = OrderedDict()
        self.captures = 0
        self.replays = 0
        self.capture_failed = False

    def __deepcopy__(self, memo):
        # Graphs are derived process-local execution state, not model state.
        result = type(self)()
        memo[id(self)] = result
        return result

    def run(self, module, inputs, eager):
        parameters = tuple(module.gru.parameters())
        signature = (
            tuple((id(p), p.data_ptr(), tuple(p.shape), p.stride(), p.dtype, p.device)
                  for p in parameters),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.are_deterministic_algorithms_enabled(),
        )
        if self.signature != signature:
            self.entries.clear()
            self.seen.clear()
            self.signature = signature
            self.capture_failed = False
        key = (
            tuple((tuple(value.shape), value.stride(), value.dtype, value.device)
                  for value in inputs),
            torch.cuda.current_stream(inputs[0].device).cuda_stream,
        )
        entry = self.entries.get(key)
        if entry is None:
            # Do not evict useful full chunks for a stream of unique ragged
            # tails. Both graph memory and bookkeeping remain bounded.
            if self.capture_failed or len(self.entries) >= self.max_graphs:
                return eager(*inputs)
            calls = self.seen.pop(key, 0) + 1
            self.seen[key] = calls
            if len(self.seen) > 32:
                self.seen.popitem(last=False)
            if calls <= self.warmup_calls:
                return eager(*inputs)
            free_bytes, _ = torch.cuda.mem_get_info(inputs[0].device)
            if free_bytes < 512 * 1024**2:
                return eager(*inputs)
            try:
                entry = _GraphEntry(eager, inputs, parameters)
            except torch.cuda.OutOfMemoryError:
                # Capture needs more temporary storage than eager execution.
                # Disable new captures until parameter storage changes; a
                # nearly-full device must not retry this on every chunk.
                self.capture_failed = True
            if self.capture_failed:
                return eager(*inputs)
            self.entries[key] = entry
            self.captures += 1
        self.replays += 1
        return entry.replay(inputs)


def run_exact_gru_sequence(module, x, is_init, hidden, eager):
    """Use a graph only for standard, repeated FP32 CUDA replay sequences."""
    if (
        not _ENABLED.get()
        or torch.is_grad_enabled()
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.shape[1] < 8
        or torch.is_autocast_enabled("cuda")
        or torch.compiler.is_compiling()
        or type(module.gru) is not nn.GRUCell
        or module.gru._forward_hooks
        or module.gru._forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
        or x.device.index != torch.cuda.current_device()
        or torch.cuda.is_current_stream_capturing()
    ):
        return eager(x, is_init, hidden)
    # Overlapping inputs (e.g. expanded reset masks) cannot be copied into
    # matching-stride static storage. Leave unusual layouts on the eager path.
    inputs = (x, is_init, hidden)
    if any(torch._debug_has_internal_overlap(value) != 0 for value in inputs):
        return eager(*inputs)
    cache = getattr(module, "_exact_gru_graph_cache", None)
    if cache is None:
        cache = module._exact_gru_graph_cache = _GRUGraphCache()
    return cache.run(module, inputs, eager)
