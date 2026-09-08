from __future__ import annotations

import pytest
import torch

from active_adaptation.learning.ppo.exact_episode_replay import ExactEpisodePrefixStore


IS_INIT = "perception_is_init"


def _raw(length: int = 9) -> dict[str, torch.Tensor]:
    return {
        "depth": torch.arange(length * 6, dtype=torch.float64).reshape(length, 2, 3) / 997,
        "policy": torch.arange(length * 2, dtype=torch.float32).reshape(length, 2) / 1237,
        IS_INIT: torch.tensor([True] + [False] * (length - 1)).reshape(length, 1),
    }


def _slice(fields, start, stop):
    return {key: value[start:stop] for key, value in fields.items()}


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype
        assert torch.equal(actual[key], value)


def test_append_overlap_and_cross_chunk_slices_preserve_every_node_once():
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    fields = _raw()
    store.append(uid, 0, _slice(fields, 0, 3))
    store.append(uid, 2, _slice(fields, 2, 6))
    store.append(uid, 6, _slice(fields, 6, 9))
    store.append(uid, 1, _slice(fields, 1, 8))
    assert store.length(uid) == store.node_count == 9
    assert store.episode_count == 1
    _assert_equal(store.prefix(uid, 9), fields)
    _assert_equal(store.slice(uid, 2, 8), _slice(fields, 2, 8))
    _assert_equal(store.slice(uid, 9, 9), _slice(fields, 9, 9))
    _assert_equal(store.prefix(uid, 0), _slice(fields, 0, 0))


@pytest.mark.parametrize("field", ["depth", "policy"])
def test_overlap_checks_every_field_and_failed_append_is_atomic(field):
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    fields = _raw()
    store.append(uid, 0, _slice(fields, 0, 3))
    store.append(uid, 3, _slice(fields, 3, 6))
    bad = {key: value[2:9].clone() for key, value in fields.items()}
    bad[field][2].add_(1)
    with pytest.raises(RuntimeError, match=f"overlap mismatch in field '{field}'"):
        store.append(uid, 2, bad)
    assert store.length(uid) == store.node_count == 6
    _assert_equal(store.prefix(uid, 6), _slice(fields, 0, 6))


def test_missing_history_gaps_and_resets_fail_explicitly():
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    fields = _raw()
    with pytest.raises(RuntimeError, match="missing or evicted"):
        store.prefix(uid, 1)
    with pytest.raises(RuntimeError, match="history gap"):
        store.append(uid, 1, _slice(fields, 1, 3))
    bad = _raw()
    bad[IS_INIT][0] = False
    with pytest.raises(RuntimeError, match="start at is_init=True"):
        store.append(uid, 0, bad)
    bad = _raw()
    bad[IS_INIT][5] = True
    with pytest.raises(RuntimeError, match="internal reset"):
        store.append(uid, 0, bad)
    assert store.episode_count == store.node_count == 0
    store.append(uid, 0, _slice(fields, 0, 3))
    with pytest.raises(RuntimeError, match="internal reset"):
        store.append(uid, 3, _slice(fields, 0, 3))
    with pytest.raises(RuntimeError, match="history gap"):
        store.append(uid, 4, _slice(fields, 4, 5))
    with pytest.raises(RuntimeError, match="lacks requested history"):
        store.prefix(uid, 4)


def test_retention_keeps_complete_prefix_and_never_reuses_evicted_uid():
    store = ExactEpisodePrefixStore(IS_INIT)
    first, second = (store.allocate_episode_uid() for _ in range(2))
    fields = _raw()
    store.append(first, 0, fields)
    store.append(second, 0, _slice(fields, 0, 3))
    pending = store.allocate_episode_uid()
    store.retain([first, pending])
    assert store.episode_count == 1
    assert store.node_count == 9
    _assert_equal(store.prefix(first, 9), fields)
    for operation in (
        lambda: store.length(second),
        lambda: store.append(second, 0, fields),
        lambda: store.retain([second]),
    ):
        with pytest.raises(RuntimeError, match="evicted"):
            operation()
    # Failed retain must not remove the still-valid first episode.
    assert store.length(first) == 9
    store.append(pending, 0, _slice(fields, 0, 1))
    store.retain([])
    assert store.episode_count == store.node_count == 0
    assert store.allocate_episode_uid() > pending


def test_raw_precision_and_ownership_survive_collection_and_reads():
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    fields = _raw()
    original = {key: value.clone() for key, value in fields.items()}
    fields["policy"].requires_grad_(True)
    store.append(uid, 0, fields)
    with torch.no_grad():
        fields["depth"].add_(100)
        fields["policy"].add_(100)
    result = store.prefix(uid, 9)
    _assert_equal(result, original)
    assert all(not value.requires_grad and value.device.type == "cpu" for value in result.values())
    result["depth"].zero_()
    _assert_equal(store.prefix(uid, 9), original)


def test_inference_collection_yields_ordinary_tensors_usable_by_autograd():
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    with torch.inference_mode():
        fields = _raw()
        assert fields["policy"].is_inference()
        store.append(uid, 0, _slice(fields, 0, 5))
        store.append(uid, 4, _slice(fields, 4, 9))
        result = store.prefix(uid, 9)
    assert all(not value.is_inference() for value in result.values())
    weights = torch.ones((2, 3), requires_grad=True)
    (result["policy"] @ weights).sum().backward()
    assert weights.grad is not None


def test_field_schema_is_validated_without_committing_partial_data():
    store = ExactEpisodePrefixStore(IS_INIT)
    uid = store.allocate_episode_uid()
    fields = _raw()
    store.append(uid, 0, _slice(fields, 0, 3))
    bad = _slice(fields, 3, 6)
    bad["depth"] = bad["depth"].float()
    with pytest.raises(ValueError, match="dtype or trailing shape changed"):
        store.append(uid, 3, bad)
    assert store.length(uid) == 3
    bad = _slice(fields, 3, 6)
    bad["policy"] = bad["policy"][:2]
    with pytest.raises(ValueError, match="different time lengths"):
        store.append(uid, 3, bad)
    assert store.length(uid) == 3
