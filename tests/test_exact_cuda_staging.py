"""Opt-in CUDA equality check; ordinary test runs allocate no CUDA memory."""

from __future__ import annotations

import os

import pytest
import torch
from tensordict import TensorDict

from benchmark_exact_online_encoder import BenchmarkPolicy, make_requests, require_idle_cuda
from active_adaptation.learning.ppo.ppo_vel import set_recurrent_mode
from active_adaptation.learning.ppo.td3_bc_dagger import PERCEPTION_OBJECT_GEO_ID_KEY


def _assert_full_history(policy, requests):
    with torch.no_grad(), set_recurrent_mode(True):
        for uid, stop in requests.items():
            data = {
                key: value.unsqueeze(0).to(policy.device)
                for key, value in policy._exact_online_store.prefix(uid, stop).items()
            }
            data["object_geo_"] = policy._decode_replay_object_geo(
                data.pop(PERCEPTION_OBJECT_GEO_ID_KEY), device=policy.device, dtype=data["policy"].dtype,
            )
            data["depth_hx"] = torch.zeros(1, stop, policy.depth_feature_dim, device=policy.device)
            data["adapt_hx"] = torch.zeros(1, stop, policy.cfg.latent_dim, device=policy.device)
            expected = policy.run_stack(TensorDict(data, [1, stop], device=policy.device))
            entry = policy._exact_online_prefixes[uid]
            assert entry.length == stop
            actor = torch.cat([expected[key] for key in policy.q_actor_keys], -1)[0]
            torch.testing.assert_close(torch.cat(entry.actor_chunks), actor)
            for name in ("depth_hx", "adapt_hx"):
                torch.testing.assert_close(getattr(entry, name), expected["next", name][0, -1])


@pytest.mark.skipif(
    os.environ.get("VAIC_EXACT_CUDA_TESTS") != "1",
    reason="Set VAIC_EXACT_CUDA_TESTS=1 for the bounded idle-GPU staging check",
)
def test_pinned_double_buffer_reuse_mixed_lengths_suffix_and_ema_rebuild_match_full_history():
    allowed = [name for name in os.environ.get("VAIC_EXACT_CUDA_ALLOW_PROCESS", "").split(",") if name]
    try:
        require_idle_cuda(allowed)
    except RuntimeError as error:
        pytest.skip(str(error))
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_threads = torch.get_num_threads()
    try:
        # Scoped FP32 isolates exact sequence/history handling from TF32
        # kernel-shape rounding. Production precision settings are restored.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        policy = BenchmarkPolicy()
        requests = make_requests(policy, range(129, 137))
        prefix = {uid: length - 5 for uid, length in requests.items()}
        policy._encode_exact_online_prefixes(prefix)
        _assert_full_history(policy, prefix)
        # Multiple time chunks recycle each pinned slot more than once; later
        # suffix evaluation must start at the true (unpadded) raw hidden state.
        policy._encode_exact_online_prefixes(requests)
        assert policy._exact_online_encoded_nodes == sum(requests.values())
        _assert_full_history(policy, requests)
        with torch.no_grad():
            next(policy.adapt_ema.parameters()).add_(0.01)
        policy._perception_ema_generation += 1
        policy._encode_exact_online_prefixes(requests)
        assert policy._exact_online_encoded_nodes == 2 * sum(requests.values())
        _assert_full_history(policy, requests)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_num_threads(previous_threads)
