"""Exact batching must reduce calls without discarding any recurrent history."""

import pytest
import torch

from active_adaptation.learning.ppo.exact_episode_replay import ExactEpisodePrefixStore
from test_exact_online_perception import _Policy, _raw


def test_direct_batch_gather_preserves_values_padding_and_reuses_storage():
    store = ExactEpisodePrefixStore("is_init")
    fields = [_raw(9), _raw(7, seed=912)]
    uids = []
    for raw in fields:
        uid = store.allocate_episode_uid()
        store.append(uid, 0, {key: value[:3] for key, value in raw.items()})
        store.append(uid, 3, {key: value[3:] for key, value in raw.items()})
        uids.append(uid)
    buffers = {}
    with torch.inference_mode():
        batch = store.batch_slice([(uids[0], 1, 8), (uids[1], 2, 5)], buffers=buffers)
    for key, value in batch.items():
        assert not value.is_inference()
        assert torch.equal(value[0], fields[0][key][1:8])
        assert torch.equal(value[1, :3], fields[1][key][2:5])
        assert torch.count_nonzero(value[1, 3:]) == 0
    pointers = {key: value.data_ptr() for key, value in buffers.items()}
    smaller = store.batch_slice([(uids[1], 1, 4)], buffers=buffers)
    assert pointers == {key: value.data_ptr() for key, value in buffers.items()}
    for key, value in smaller.items():
        assert torch.equal(value[0], fields[1][key][1:4])
    # Staging may be mutated/reused; immutable original histories cannot be.
    smaller["policy"].zero_()
    assert torch.equal(store.prefix(uids[1], 7)["policy"], fields[1]["policy"])


@pytest.mark.parametrize("interval", [(0, 0), (-1, 2), (2, 10), (5, 3)])
def test_batch_gather_rejects_missing_or_empty_history(interval):
    store = ExactEpisodePrefixStore("is_init")
    uid = store.allocate_episode_uid()
    store.append(uid, 0, _raw(9))
    with pytest.raises(RuntimeError, match="complete history"):
        store.batch_slice([(uid, *interval)])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64,
                                    torch.bfloat16, torch.int64, torch.bool])
def test_cpu_staging_copies_strided_values_without_precision_conversion(dtype):
    store = ExactEpisodePrefixStore("is_init")
    source = torch.arange(24).reshape(3, 8).to(dtype)[:, ::2]
    raw = {"value": source, "is_init": torch.tensor([[True], [False], [False]])}
    uid = store.allocate_episode_uid()
    store.append(uid, 0, raw)
    saved = store.prefix(uid, 3)["value"]
    assert saved.dtype == dtype
    assert torch.equal(saved, source)
    result = store.batch_slice([(uid, 0, 3), (uid, 1, 2)])["value"]
    assert result.dtype == dtype
    assert torch.equal(result[0], source)
    assert torch.equal(result[1, 0], source[1])
    assert torch.count_nonzero(result[1, 1:]) == 0


def test_mixed_length_encoder_uses_six_batches_not_forty_single_step_tails():
    policy = _Policy(chunk=32)
    policy.cfg.perception_encode_microbatch_size = 128  # 40 episodes per batch.
    lengths = list(range(129, 169))
    raw = [_raw(length, seed=930 + index) for index, length in enumerate(lengths)]
    uids = [policy.add(value) for value in raw]
    expected = [policy.direct(value) for value in raw]
    calls = []
    hook = policy.temporal_depth_gru_ema.register_forward_pre_hook(
        lambda _, args: calls.append(tuple(args[0].batch_size))
    )
    try:
        actual = policy._gather_exact_online_actor(torch.tensor([
            (uid, length - 1) for uid, length in zip(uids, lengths)
        ]))
    finally:
        hook.remove()
    assert len(calls) == 6  # Old shortest-remainder scheduler required 44.
    assert policy._exact_online_encoder_batches == 6
    assert policy._exact_online_encoded_nodes == sum(lengths)
    assert sum(n * t for n, t in calls) == sum(lengths) + policy._exact_online_padded_nodes
    for row, (uid, (actor, direct)) in enumerate(zip(uids, expected)):
        torch.testing.assert_close(actual[row], actor[-1])
        entry = policy._exact_online_prefixes[uid]
        assert entry.length == lengths[row]
        torch.testing.assert_close(entry.depth_hx, direct["next", "depth_hx"][0, -1])
        torch.testing.assert_close(entry.adapt_hx, direct["next", "adapt_hx"][0, -1])
        assert not entry.depth_hx.is_inference()
        assert not entry.actor_chunks[0].is_inference()


def test_padding_is_not_cached_or_included_in_finite_validation():
    policy = _Policy(chunk=32)
    short, long = policy.add(_raw(1)), policy.add(_raw(3, seed=990))

    def poison_padding(_, args, output):
        output["priv_pred"][0, 1:] = float("nan")

    hook = policy.adapt_ema.register_forward_hook(poison_padding)
    try:
        result = policy._gather_exact_online_actor(torch.tensor([(short, 0), (long, 2)]))
    finally:
        hook.remove()
    assert torch.isfinite(result).all()
    assert policy._exact_online_prefixes[short].length == 1
    assert policy._exact_online_actor_bank.shape[0] == 4


def test_nonfinite_real_state_fails_closed_without_poisoned_cache():
    policy = _Policy(chunk=3)
    raw = _raw(5)
    raw["policy"][2, 0] = float("nan")
    uid = policy.add(raw)
    with pytest.raises(RuntimeError, match="nonfinite"):
        policy._gather_exact_online_actor(torch.tensor([(uid, 4)]))
    assert policy._exact_online_prefixes == {}
    assert policy._exact_online_actor_bank is None


def test_inference_scope_cache_can_later_feed_actor_backward():
    policy = _Policy(chunk=3)
    uid = policy.add(_raw(5))
    with torch.inference_mode():
        inputs = policy._gather_exact_online_actor(torch.tensor([(uid, 2), (uid, 4)]))
    assert not inputs.is_inference()
    assert not policy._exact_online_actor_bank.is_inference()
    actor = torch.nn.Linear(policy._q_actor_dim, 2)
    actor(inputs).square().sum().backward()
    assert torch.isfinite(actor.weight.grad).all()
