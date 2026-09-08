"""Geometry lookup placement must preserve bytes and both recurrent states."""

from types import MethodType

import pytest
import torch

from active_adaptation.learning.ppo.td3_bc_dagger import DistributionalTD3TeacherBC as TD3
from test_exact_online_perception import _Policy, _raw


def _codebook_policy(*, legacy=False):
    policy = _Policy(chunk=3)
    for name in ("_ensure_replay_object_geo_codebook", "_replay_object_geo_bank_for",
                 "_decode_replay_object_geo"):
        setattr(policy, name, MethodType(getattr(TD3, name), policy))
    policy._ensure_replay_object_geo_codebook()
    policy._replay_object_geo_bank = policy.geometry.clone()
    policy._replay_object_geo_bank_generation = 1
    if legacy:
        policy._exact_online_geometry_bank = lambda *args, **kwargs: None
    return policy


def _assert_identical(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_checked_lookup_is_byte_identical_to_original_decoder(id_dtype, dtype):
    policy = _codebook_policy()
    policy._replay_object_geo_bank[0, :4] = torch.tensor([0.0, -0.0, 1.0e-30, -1.0e-30])
    ids = torch.tensor([[2, 1, 0], [1, 0, 2]], dtype=id_dtype)[:, ::2]
    expected = policy._decode_replay_object_geo(ids, device="cpu", dtype=dtype)
    with torch.inference_mode():
        bank = policy._exact_online_geometry_bank(ids, dtype=dtype)
    assert not bank.is_inference()
    actual = bank.index_select(0, ids.reshape(-1).long()).reshape(*ids.shape, bank.shape[-1])
    _assert_identical(actual, expected)
    assert policy._exact_online_geometry_bank(ids, dtype=dtype).data_ptr() == bank.data_ptr()


@pytest.mark.parametrize("ids,exception", [
    (torch.tensor([-1]), IndexError),
    (torch.tensor([3]), IndexError),
    (torch.tensor([0.5]), TypeError),
    (torch.tensor([True]), TypeError),
])
def test_bad_geometry_ids_fail_before_device_lookup(ids, exception):
    policy = _codebook_policy()
    # A nonexistent CUDA device proves invalid metadata cannot reach a copy.
    policy.device = torch.device("cuda:99")
    with pytest.raises(exception):
        policy._exact_online_geometry_bank(ids, dtype=torch.float32)
    assert not policy._replay_object_geo_device_banks


def test_geometry_cache_tracks_codebook_generation_and_owns_ordinary_storage():
    policy = _codebook_policy()
    ids = torch.tensor([0, 2])
    with torch.inference_mode():
        first = policy._exact_online_geometry_bank(ids, dtype=torch.float32)
    assert not first.is_inference()
    original = first.clone()
    policy._replay_object_geo_bank = torch.cat([policy.geometry, policy.geometry[:1] + 0.1])
    policy._replay_object_geo_bank_generation += 1
    second = policy._exact_online_geometry_bank(torch.tensor([3]), dtype=torch.float32)
    assert first.data_ptr() != second.data_ptr()
    _assert_identical(first, original)
    _assert_identical(second, policy._replay_object_geo_bank)
    parameter = torch.ones_like(second, requires_grad=True)
    (parameter * second).sum().backward()
    _assert_identical(parameter.grad, second)


def test_custom_geometry_decoder_keeps_its_existing_contract():
    policy = _Policy()
    assert policy._exact_online_geometry_bank(torch.tensor([0]), dtype=torch.float32) is None
    uid = policy.add(_raw(7))
    actual = policy._gather_exact_online_actor(torch.tensor([[uid, 6]]))
    torch.testing.assert_close(actual[0], policy.direct(_raw(7))[0][6])


def test_id_only_batches_match_original_actor_and_raw_hidden_bits_through_ema_rebuild(monkeypatch):
    original, optimized = _codebook_policy(legacy=True), _codebook_policy()
    raw = [_raw(11, seed=880), _raw(16, seed=881)]
    requests = []
    for policy in (original, optimized):
        requests.append({policy.add(value): length for value, length in zip(raw, (5, 9))})

    # The optimized encoder must never ask the decoder to build full CPU
    # geometry. Its staging dictionaries must carry only the compact IDs.
    real_slice = optimized._exact_online_store.batch_slice
    staging = []

    def capture_staging(*args, **kwargs):
        staging.append(kwargs["buffers"])
        return real_slice(*args, **kwargs)

    monkeypatch.setattr(optimized._exact_online_store, "batch_slice", capture_staging)
    for generation in range(2):
        for extend in (False, True):
            for policy, pending in zip((original, optimized), requests):
                if extend:
                    pending.update(zip(pending, (11, 16)))
                policy._encode_exact_online_prefixes(pending)
            for old, new in zip(original._exact_online_prefixes.values(), optimized._exact_online_prefixes.values()):
                assert old.length == new.length
                _assert_identical(torch.cat(old.actor_chunks), torch.cat(new.actor_chunks))
                _assert_identical(old.depth_hx, new.depth_hx)
                _assert_identical(old.adapt_hx, new.adapt_hx)
        for policy in (original, optimized):
            with torch.no_grad():
                next(policy.adapt_ema.parameters()).add_(0.0125)
            policy._perception_ema_generation += 1
    assert staging
    assert all("object_geo_" not in slot for slot in staging)
    assert all("perception_object_geo_id" in slot for slot in staging)
    assert original._exact_online_encoded_nodes == optimized._exact_online_encoded_nodes
