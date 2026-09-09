from __future__ import annotations

import types

import pytest
import torch

from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC as TD3,
    PERCEPTION_OBJECT_GEO_ID_KEY,
)
from teacher_cache_test_helpers import (
    TeacherCacheDiagnosticPolicy, capture_rebuild, compare, install_store,
    reference_rebuild_teacher_actor_cache,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("residual", [False, True])
def test_complete_teacher_cache_and_hidden_states_match_frozen_reference(device, residual):
    policy = TeacherCacheDiagnosticPolicy(device, microbatch=8, residual=residual)
    install_store(policy, [1, 17, 33, 65, 34])
    previous = None
    for generation in range(2):
        if generation:
            with torch.no_grad():
                for parameter in policy.adapt_ema.parameters():
                    parameter.mul_(0.9).add_(0.001)
            policy._perception_ema_generation += 1
        reference = capture_rebuild(policy, reference_rebuild_teacher_actor_cache)
        candidate = capture_rebuild(policy, TD3._rebuild_teacher_actor_cache)
        report = compare(reference, candidate)
        assert all(item["torch_equal"] for item in report.values())
        assert policy._teacher_actor_cache.lineage.ema_generation == generation
        if previous is not None:
            assert not torch.equal(previous, candidate["actor_inputs"])
        previous = candidate["actor_inputs"]


def test_custom_decoder_keeps_original_per_chunk_contract():
    policy = TeacherCacheDiagnosticPolicy("cpu", microbatch=4)
    install_store(policy, [17, 65])
    calls = []

    def decode(self, ids, *, device, dtype):
        calls.append(tuple(ids.shape))
        return TD3._decode_replay_object_geo(self, ids, device=device, dtype=dtype) + 0.25

    policy._decode_replay_object_geo = types.MethodType(decode, policy)
    reference = capture_rebuild(policy, reference_rebuild_teacher_actor_cache)
    reference_calls = list(calls)
    calls.clear()
    candidate = capture_rebuild(policy, TD3._rebuild_teacher_actor_cache)
    compare(reference, candidate)
    assert calls == reference_calls
    assert len(calls) == 4


def test_codebook_generation_change_reloads_correct_values():
    policy = TeacherCacheDiagnosticPolicy("cpu", microbatch=4)
    install_store(policy, [17, 33])
    before = capture_rebuild(policy, TD3._rebuild_teacher_actor_cache)
    policy._replay_object_geo_bank = policy._replay_object_geo_bank + 0.5
    policy._replay_object_geo_bank_generation += 1
    reference = capture_rebuild(policy, reference_rebuild_teacher_actor_cache)
    candidate = capture_rebuild(policy, TD3._rebuild_teacher_actor_cache)
    compare(reference, candidate)
    assert not torch.equal(before["actor_inputs"], candidate["actor_inputs"])


@pytest.mark.parametrize("invalid_id", [-1, 4])
def test_invalid_ids_fail_without_publishing_over_existing_cache(invalid_id):
    policy = TeacherCacheDiagnosticPolicy("cpu", microbatch=4)
    store = install_store(policy, [17])
    policy._rebuild_teacher_actor_cache(policy._teacher_actor_cache_lineage())
    previous = policy._teacher_actor_cache._actor_by_node
    previous_lineage = policy._teacher_actor_cache.lineage
    # Force a fresh mirror, as with a newly installed invalid raw store.
    store.raw_fields[PERCEPTION_OBJECT_GEO_ID_KEY][0] = invalid_id
    policy._teacher_episode_device_raw_lineage = None
    with pytest.raises(IndexError, match="outside the codebook"):
        policy._rebuild_teacher_actor_cache(policy._teacher_actor_cache_lineage())
    assert policy._teacher_actor_cache._actor_by_node is previous
    assert policy._teacher_actor_cache.lineage == previous_lineage


def test_invalid_geometry_dtype_and_empty_codebook_keep_clear_errors():
    policy = TeacherCacheDiagnosticPolicy("cpu", microbatch=4)
    store = install_store(policy, [17])
    original = store.raw_fields[PERCEPTION_OBJECT_GEO_ID_KEY]
    store._flat_fields[PERCEPTION_OBJECT_GEO_ID_KEY] = original.float()
    with pytest.raises(TypeError, match="int32 or int64"):
        policy._rebuild_teacher_actor_cache(policy._teacher_actor_cache_lineage())
    store._flat_fields[PERCEPTION_OBJECT_GEO_ID_KEY] = original
    policy._teacher_episode_device_raw_lineage = None
    policy._replay_object_geo_bank = torch.empty(0, 384)
    with pytest.raises(RuntimeError, match="no object geometry codebook"):
        policy._rebuild_teacher_actor_cache(policy._teacher_actor_cache_lineage())
