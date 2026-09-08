"""Exercise online history provenance independently of the perception networks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.exact_online_perception import (
    EXACT_CURRENT_REF,
    EXACT_NEXT_REF,
    ExactOnlinePerceptionReplayMixin,
)
from active_adaptation.learning.ppo.ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBS_KEY,
    VEL_CMD_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DAGGER_IS_DAGGER_ENV_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    PERCEPTION_OBJECT_GEO_ID_KEY,
    DistributionalTD3TeacherBC,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _TD3DeviceReplay,
)


class _JournalParent:
    """Supply collection seams; route rows using the production ring writer."""

    _extend_online_replays = DistributionalTD3TeacherBC._extend_online_replays

    def _uses_separate_online_replays(self):
        return True

    def _encode_replay_object_geo(self, td):
        self.geometry_encode_calls += 1
        # The fixture geometry itself is its stable registry identifier.
        return td[OBJECT_GEO_KEY].long()

    def _prepare_raw_final_state(self, td, **kwargs):
        return {"parent_final": td[OBS_KEY].clone()}

    def _reset_teacher_episode_cache_state(self):
        self.parent_resets += 1

    def _dagger_transition_chunks(self, td):
        n, t = map(int, td.batch_size)
        flat = {
            _PREFILL_ENV_INDEX_KEY: torch.arange(n).repeat_interleave(t),
            _PREFILL_STEP_INDEX_KEY: torch.arange(t).repeat(n),
            DAGGER_IS_DAGGER_ENV_KEY: td[DAGGER_IS_DAGGER_ENV_KEY].reshape(-1),
            DAGGER_IS_STUDENT_ACTION_KEY: td[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1),
            "row_identity": td[OBS_KEY][..., :1].reshape(n * t, 1),
        }
        self._rollout_final_batch = None
        self._truncation_final_batches = []
        for start in range(0, n * t, self.transition_chunk_rows):
            self.parent_chunk_count += 1
            yield {key: value[start : start + self.transition_chunk_rows] for key, value in flat.items()}


class _JournalHarness(ExactOnlinePerceptionReplayMixin, _JournalParent):
    def _exact_online_replay_enabled(self):
        return True

    def __init__(self, *, capacity=32, chunk_rows=3):
        self.cfg = SimpleNamespace()
        self.device = torch.device("cpu")
        self._rollout_final_batch = None
        self._truncation_final_batches = []
        self.geometry_encode_calls = 0
        self.parent_chunk_count = 0
        self.parent_resets = 0
        self.transition_chunk_rows = chunk_rows
        self.prefill = False
        self.dagger_replay = _TD3DeviceReplay(capacity, "cpu")
        self.student_replay = _TD3DeviceReplay(capacity, "cpu")

    def _teacher_prefill_active(self):
        return self.prefill


def _states(values, resets) -> TensorDict:
    values = torch.as_tensor(values, dtype=torch.float32)
    n, length = values.shape
    depth = values[..., None, None, None].expand(n, length, 1, 2, 2).clone() / 997 + 0.25
    # Deliberately use values that a depth float->uint8->float codec cannot
    # preserve. Exact online history must preserve every input bit.
    depth = torch.nextafter(depth, torch.full_like(depth, float("inf")))
    cohort = torch.arange(n)[:, None].eq(0).expand(n, length)
    controller = torch.arange(length)[None].remainder(2).eq(0).expand(n, length) | ~cohort
    return TensorDict(
        {
            DEPTH_KEY: depth,
            OBS_KEY: torch.stack((values, values / 17 + 0.12345), dim=-1),
            VEL_CMD_KEY: torch.stack((values / 23, values / 29, values / 31), dim=-1),
            OBJECT_GEO_KEY: values.remainder(3).long().unsqueeze(-1),
            "is_init": torch.as_tensor(resets, dtype=torch.bool).reshape(n, length, 1),
            DAGGER_IS_DAGGER_ENV_KEY: cohort.unsqueeze(-1),
            DAGGER_IS_STUDENT_ACTION_KEY: controller.unsqueeze(-1),
        },
        batch_size=(n, length),
        device="cpu",
    )


def _rollout(harness, states, start=0, stop=None):
    if stop is None:
        stop = states.batch_size[1] - 1
    harness._rollout_final_batch = harness._prepare_raw_final_state(states[:, stop])
    harness._truncation_final_batches = []
    return states[:, start:stop]


def _collect(harness, rollout):
    # Production train_op materializes and joins every chunk before the one
    # replay extension/prune call. Do not prune a partially staged rollout.
    chunks = tuple(harness._dagger_transition_chunks(rollout))
    table = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    harness._extend_online_replays(table)
    return table


def _expected_raw(states, env=0):
    return {
        DEPTH_KEY: states[DEPTH_KEY][env],
        OBS_KEY: states[OBS_KEY][env],
        VEL_CMD_KEY: states[VEL_CMD_KEY][env],
        PERCEPTION_OBJECT_GEO_ID_KEY: states[OBJECT_GEO_KEY][env],
        "is_init": states["is_init"][env],
    }


def _assert_raw_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype
        assert torch.equal(actual[key], value), key


def test_journal_spans_rollouts_deduplicates_endpoint_and_preserves_float_inputs():
    harness = _JournalHarness()
    states = _states([[0, 1, 2, 3, 4]], [[True, False, False, False, False]])
    rollout = _rollout(harness, states, stop=2)
    rng = torch.get_rng_state().clone()
    current, successor = harness._journal_exact_online_rollout(rollout)
    assert torch.equal(torch.get_rng_state(), rng)
    uid = int(current[0, 0, 0])
    assert current.tolist() == [[[uid, 0], [uid, 1]]]
    assert successor.tolist() == [[[uid, 1], [uid, 2]]]
    assert harness._exact_online_store.node_count == 3

    current, successor = harness._journal_exact_online_rollout(_rollout(harness, states, 2, 4))
    assert current.tolist() == [[[uid, 2], [uid, 3]]]
    assert successor.tolist() == [[[uid, 3], [uid, 4]]]
    assert harness._exact_online_carry_refs.tolist() == [[uid, 4]]
    assert harness._exact_online_store.episode_count == 1
    assert harness._exact_online_store.node_count == 5
    original = {key: value.clone() for key, value in _expected_raw(states).items()}
    _assert_raw_equal(harness._exact_online_store.prefix(uid, 5), original)
    assert not torch.equal(original[DEPTH_KEY], (original[DEPTH_KEY] * 100).round() / 100)
    states[DEPTH_KEY].zero_()
    _assert_raw_equal(harness._exact_online_store.prefix(uid, 5), original)
    # One stable geometry encoding per rollout plus one per final capture.
    assert harness.geometry_encode_calls == 4
    assert torch.equal(torch.get_rng_state(), rng)


def test_journal_segments_at_real_resets_and_keeps_reset_carry_in_new_episode():
    harness = _JournalHarness()
    states = _states([[0, 1, 100, 101, 200]], [[True, False, True, False, True]])
    current, successor = harness._journal_exact_online_rollout(_rollout(harness, states))
    first, second = int(current[0, 0, 0]), int(current[0, 2, 0])
    live = int(harness._exact_online_carry_refs[0, 0])
    assert len({first, second, live}) == 3
    assert current.tolist() == [[[first, 0], [first, 1], [second, 0], [second, 1]]]
    assert successor.tolist() == [[[first, 1], [second, 0], [second, 1], [live, 0]]]
    assert [harness._exact_online_store.length(uid) for uid in (first, second, live)] == [2, 2, 1]
    assert harness._exact_online_store.prefix(live, 1)[OBS_KEY][0, 0].item() == 200


def test_timeout_successor_is_true_final_old_episode_state_not_reset_observation():
    harness = _JournalHarness()
    states = _states([[0, 1, 100, 101, 200]], [[True, False, True, False, True]])
    rollout = _rollout(harness, states)
    finals = _states([[2], [102]], [[False], [False]])[:, 0]
    captured = harness._prepare_raw_final_state(finals)
    captured["indices"] = torch.tensor([1, 3])
    harness._truncation_final_batches = [captured]

    current, successor = harness._journal_exact_online_rollout(rollout)
    first, second = int(current[0, 0, 0]), int(current[0, 2, 0])
    live = int(harness._exact_online_carry_refs[0, 0])
    assert successor.tolist() == [[[first, 1], [first, 2], [second, 1], [second, 2]]]
    assert harness._exact_online_carry_refs.tolist() == [[live, 0]]
    assert harness._exact_online_store.length(live) == 1
    for row, uid in enumerate((first, second)):
        final_raw = harness._exact_online_store.slice(uid, 2, 3)
        for key in (DEPTH_KEY, OBS_KEY, VEL_CMD_KEY, "is_init"):
            assert torch.equal(final_raw[key][0], finals[key][row])
        assert final_raw[OBS_KEY][0, 0].item() == (2 if row == 0 else 102)
    assert harness._exact_online_store.prefix(live, 1)[OBS_KEY][0, 0].item() == 200


def test_controller_switches_keep_one_history_and_chunk_rows_keep_original_refs():
    harness = _JournalHarness(chunk_rows=3)
    states = _states(
        [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]],
        [[True, False, False, False, False], [True, False, False, False, False]],
    )
    rollout = _rollout(harness, states)
    rng = torch.get_rng_state().clone()
    table = _collect(harness, rollout)
    assert harness.parent_chunk_count == 3
    assert table["row_identity"].flatten().tolist() == [0, 1, 2, 3, 10, 11, 12, 13]
    uids = table[EXACT_CURRENT_REF][:, 0].unique().tolist()
    assert len(uids) == harness._exact_online_store.episode_count == 2
    assert harness._exact_online_store.node_count == 10
    for env, uid in enumerate(uids):
        rows = slice(env * 4, (env + 1) * 4)
        assert table[EXACT_CURRENT_REF][rows].tolist() == [[uid, step] for step in range(4)]
        assert table[EXACT_NEXT_REF][rows].tolist() == [[uid, step] for step in range(1, 5)]
        _assert_raw_equal(harness._exact_online_store.prefix(uid, 5), _expected_raw(states, env))
    assert harness.dagger_replay.size == harness.student_replay.size == 4
    # Teacher-controlled rows 1 and 3 still belong to the same online DAgger
    # sequence; neither history reconstruction nor routing resamples actions.
    assert harness.dagger_replay.data[DAGGER_IS_STUDENT_ACTION_KEY][:4].tolist() == [True, False, True, False]
    assert torch.equal(torch.get_rng_state(), rng)


def test_missing_initial_reset_or_final_capture_fails_instead_of_using_partial_history():
    harness = _JournalHarness()
    states = _states([[5, 6, 7]], [[False, False, False]])
    with pytest.raises(RuntimeError, match="real reset"):
        harness._journal_exact_online_rollout(_rollout(harness, states))
    assert harness._exact_online_store.node_count == 0
    harness._rollout_final_batch = None
    with pytest.raises(RuntimeError, match="rollout-final observations"):
        harness._journal_exact_online_rollout(states[:, :2])


def test_overlap_input_drift_is_rejected_even_when_difference_is_one_float_ulp():
    harness = _JournalHarness()
    states = _states([[0, 1, 2, 3, 4]], [[True, False, False, False, False]])
    harness._journal_exact_online_rollout(_rollout(harness, states, stop=2))
    states[DEPTH_KEY][:, 2] = torch.nextafter(
        states[DEPTH_KEY][:, 2], torch.full_like(states[DEPTH_KEY][:, 2], float("inf"))
    )
    with pytest.raises(RuntimeError, match="overlap mismatch.*depth"):
        harness._journal_exact_online_rollout(_rollout(harness, states, 2, 4))


def test_ring_wraparound_retains_whole_prefix_and_active_uid_until_last_reference_leaves():
    harness = _JournalHarness(capacity=2)
    first_states = _states(
        [[0, 100, 101, 102, 200], [10, 11, 12, 13, 14]],
        [[True, True, False, False, True], [True, False, False, False, False]],
    )
    table = _collect(harness, _rollout(harness, first_states))
    evicted = int(table[EXACT_CURRENT_REF][0, 0])
    tail_uid = int(table[EXACT_CURRENT_REF][1, 0])
    active = set(harness._exact_online_carry_refs[:, 0].tolist())
    with pytest.raises(RuntimeError, match="evicted"):
        harness._exact_online_store.length(evicted)
    # Only episode steps 1 and 2 remain in the DAgger ring, but step 0 is
    # necessary for exact recurrence and stays in the sidecar.
    assert harness.dagger_replay.data[EXACT_CURRENT_REF].tolist() == [[tail_uid, 1], [tail_uid, 2]]
    assert harness._exact_online_store.prefix(tail_uid, 3)[OBS_KEY][:, 0].tolist() == [100, 101, 102]
    assert all(harness._exact_online_store.length(uid) >= 1 for uid in active)

    harness._exact_online_prefixes[evicted] = object()
    harness._exact_online_prefixes[tail_uid] = object()
    second = _states([[200, 201], [14, 15]], [[True, False], [False, False]])
    _collect(harness, _rollout(harness, second))
    assert harness.dagger_replay.ptr == 1
    assert harness._exact_online_store.length(tail_uid) == 3
    assert evicted not in harness._exact_online_prefixes
    assert tail_uid in harness._exact_online_prefixes

    third = _states([[201, 202], [15, 16]], [[False, False], [False, False]])
    _collect(harness, _rollout(harness, third))
    assert harness.dagger_replay.ptr == 0
    with pytest.raises(RuntimeError, match="evicted"):
        harness._exact_online_store.length(tail_uid)
    assert tail_uid not in harness._exact_online_prefixes
    assert harness._exact_online_store.episode_count == 2
    assert set(harness._exact_online_carry_refs[:, 0].tolist()) == active


def test_sidecar_reset_removes_old_history_refs_and_derived_cache():
    harness = _JournalHarness()
    states = _states([[0, 1, 2]], [[True, False, False]])
    harness._journal_exact_online_rollout(_rollout(harness, states))
    old_store = harness._exact_online_store
    harness._exact_online_prefixes[0] = object()
    harness._exact_online_generation = (7, "vecnorm", "geometry")
    harness._exact_online_encoded_nodes = 20
    harness._reset_teacher_episode_cache_state()
    assert harness.parent_resets == 1
    for name in ("_exact_online_store", "_exact_online_carry_refs", "_exact_online_prefixes", "_exact_online_generation"):
        assert not hasattr(harness, name)
    harness._ensure_exact_online_state()
    assert harness._exact_online_store is not old_store
    assert harness._exact_online_store.episode_count == 0
    assert harness._exact_online_carry_refs is None
    assert harness._exact_online_prefixes == {}
    assert harness._exact_online_generation is None
    assert harness._exact_online_encoded_nodes == 0


@pytest.mark.parametrize("prefill, enabled", [(True, True), (False, False)])
def test_teacher_prefill_and_non_tvkd_do_not_create_online_journal(prefill, enabled):
    harness = _JournalHarness()
    harness.prefill = prefill
    harness._exact_online_replay_enabled = lambda: enabled
    states = _states([[5, 6, 7]], [[False, False, False]])
    chunks = tuple(harness._dagger_transition_chunks(_rollout(harness, states)))
    assert chunks
    assert all(EXACT_CURRENT_REF not in chunk and EXACT_NEXT_REF not in chunk for chunk in chunks)
    assert not hasattr(harness, "_exact_online_store")
