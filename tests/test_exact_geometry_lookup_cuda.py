"""Opt-in byte-equality comparison of old/new geometry placement on CUDA."""

import os
import time

import pytest
import torch

from benchmark_exact_online_encoder import BenchmarkPolicy, make_requests, require_idle_cuda


class _CPUDecodedGeometry(BenchmarkPolicy):
    def _exact_online_geometry_bank(self, geometry_ids, *, dtype):
        return None


def _assert_cache_bytes_equal(original, optimized):
    for uid, expected in original._exact_online_prefixes.items():
        actual = optimized._exact_online_prefixes[uid]
        assert expected.length == actual.length
        for left, right in (
            (torch.cat(expected.actor_chunks), torch.cat(actual.actor_chunks)),
            (expected.depth_hx, actual.depth_hx),
            (expected.adapt_hx, actual.adapt_hx),
        ):
            assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.skipif(
    os.environ.get("VAIC_EXACT_CUDA_TESTS") != "1",
    reason="Set VAIC_EXACT_CUDA_TESTS=1 for bounded idle-GPU byte-equality tests",
)
@pytest.mark.parametrize("precision", ["highest", "high"])
def test_geometry_transfer_only_change_preserves_full_actor_and_both_hidden_bytes(precision):
    allowed = [name for name in os.environ.get("VAIC_EXACT_CUDA_ALLOW_PROCESS", "").split(",") if name]
    try:
        require_idle_cuda(allowed)
    except RuntimeError as error:
        pytest.skip(str(error))
    previous = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
                torch.get_num_threads())
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = precision == "high"
        torch.set_num_threads(1)
        old, new = _CPUDecodedGeometry(), BenchmarkPolicy()
        requests = make_requests(old, range(129, 137))
        assert make_requests(new, range(129, 137)) == requests
        prefix = {uid: stop - 5 for uid, stop in requests.items()}
        times = {"cpu_decoded_geometry": [], "device_lookup": []}
        for pending in (prefix, requests, requests):
            for name, policy in (("cpu_decoded_geometry", old), ("device_lookup", new)):
                start = time.perf_counter()
                policy._encode_exact_online_prefixes(pending)
                times[name].append(time.perf_counter() - start)
            _assert_cache_bytes_equal(old, new)
            for policy in (old, new):
                # Last loop tests new-generation reset-prefix reconstruction.
                if pending is requests:
                    with torch.no_grad():
                        next(policy.adapt_ema.parameters()).add_(0.01)
                    policy._perception_ema_generation += 1
        assert old._exact_online_encoded_nodes == new._exact_online_encoded_nodes
        assert old._exact_online_encoder_batches == new._exact_online_encoder_batches
        print({"precision": precision, "byte_equal_actor_and_both_hidden": True,
               "phase_seconds_not_end_to_end_benchmark": times})
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cudnn.allow_tf32 = previous[1]
        torch.set_num_threads(previous[2])
