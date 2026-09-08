"""Same-EMA live outputs are an exact cache source, not a history shortcut."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.exact_online_perception import (
    EXACT_LIVE_GENERATION_KEY,
    EXACT_LIVE_HIDDEN_KEY,
    ExactOnlinePerceptionReplayMixin,
)
from active_adaptation.learning.ppo.fastsac_bc_dagger import _DistributionalFastSACDaggerRolloutPolicy
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DAGGER_IS_DAGGER_ENV_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
)
from test_exact_online_perception import _Policy, _raw
from test_fastsac_bc_dagger import _rollout_owner


def _states(policy, raw):
    return torch.cat([policy._inputs(fields) for fields in raw], dim=0)


def _collect(policy, states, start, stop, *, carry=None):
    """Run the real recurrent stack one step at a time before action choice."""
    n = states.batch_size[0]
    depth = torch.zeros(n, policy.depth_feature_dim) if carry is None else carry["depth_hx"]
    adapt = torch.zeros(n, policy.cfg.latent_dim) if carry is None else carry["adapt_hx"]
    rows = []
    with torch.inference_mode():
        for step in range(start, stop):
            td = states[:, step].clone()
            td["depth_hx"], td["adapt_hx"] = depth, adapt
            policy._run_stack(td, recurrent=False)
            td[REPLAY_ACTOR_OBSERVATIONS_KEY] = torch.cat([td[key] for key in policy.q_actor_keys], -1)
            policy._capture_exact_online_live_state(td)
            # This is the collector's actual ordering: source/action selection
            # follows Student perception and must not filter the cache source.
            td[DAGGER_IS_DAGGER_ENV_KEY] = torch.arange(n).eq(0)
            td[DAGGER_IS_STUDENT_ACTION_KEY] = ~td[DAGGER_IS_DAGGER_ENV_KEY] | (step % 2 == 0)
            depth = td["next", "depth_hx"].clone()
            adapt = td["next", "adapt_hx"].clone()
            # Mimic step_and_maybe_reset replacing nested next hidden states.
            # The flat snapshots must remain the true pre-reset values.
            if step + 1 < states.batch_size[1]:
                reset = states["is_init"][:, step + 1].expand_as(depth)
                depth = depth.masked_fill(reset, 0)
                adapt = adapt.masked_fill(reset, 0)
                td["next", "depth_hx"] = depth
                td["next", "adapt_hx"] = adapt
            rows.append(td.clone())
    rollout = torch.stack(rows, dim=1)
    final = states[:, stop].clone()
    final["depth_hx"], final["adapt_hx"] = depth.clone(), adapt.clone()
    policy._rollout_final_batch = {
        "exact_input__" + key: value
        for key, value in policy._exact_perception_inputs(final).items()
    }
    policy._truncation_final_batches = []
    return rollout, final


def test_live_cache_reuses_teacher_controlled_dagger_and_pure_student_without_forward():
    policy = _Policy(chunk=3)
    raw = [_raw(7), _raw(7, seed=912)]
    states = _states(policy, raw)
    rollout, _ = _collect(policy, states, 0, 5)

    current, successor = policy._journal_exact_online_rollout(rollout)

    assert policy._exact_online_reused_live_nodes == 10
    assert policy._exact_online_encoded_nodes == 0
    assert bool((~rollout[DAGGER_IS_STUDENT_ACTION_KEY][0]).any())
    assert bool(rollout[DAGGER_IS_STUDENT_ACTION_KEY][1].all())
    actual = policy._gather_exact_online_actor(current.reshape(-1, 2)).reshape(2, 5, -1)
    for env, fields in enumerate(raw):
        expected, _ = policy.direct(fields)
        torch.testing.assert_close(actual[env], expected[:5])
    assert policy._exact_online_encoded_nodes == 0
    # Pending observations were never consumed by the live policy: precisely
    # those two nodes still need one exact suffix forward for Q successors.
    policy._gather_exact_online_actor(successor[:, -1])
    assert policy._exact_online_encoded_nodes == 2


def test_production_collector_captures_every_env_before_teacher_action_selection():
    owner, mean, _ = _rollout_owner(prefill=False, dagger_env_fraction=0.5)
    perception = _Policy()
    states = _states(perception, [_raw(3, seed=950 + env) for env in range(len(mean))])
    captures = []

    def propose(td):
        perception._run_stack(td, recurrent=False)
        return mean.clone()

    def capture(td):
        assert REPLAY_ACTOR_OBSERVATIONS_KEY in td
        assert DAGGER_IS_STUDENT_ACTION_KEY not in td
        perception._capture_exact_online_live_state(td)
        captures.append(td[EXACT_LIVE_HIDDEN_KEY].clone())

    owner._student_raw_action_proposal = propose
    owner._student_collection_actor_cache_enabled = lambda: True
    owner._collection_actor_observations = lambda td: torch.cat([td[key] for key in perception.q_actor_keys], -1)
    owner._capture_exact_online_live_state = capture
    collector = _DistributionalFastSACDaggerRolloutPolicy(owner)
    with torch.inference_mode():
        first = collector(states[:, 0].clone())
        second = states[:, 1].clone()
        second["depth_hx"] = first["next", "depth_hx"].clone()
        second["adapt_hx"] = first["next", "adapt_hx"].clone()
        second = collector(second)
    assert len(captures) == 2
    teacher_action = ~second[DAGGER_IS_STUDENT_ACTION_KEY]
    assert int(teacher_action.sum()) == len(mean) // 2
    for cohort in (teacher_action, ~teacher_action):
        torch.testing.assert_close(second[EXACT_LIVE_HIDDEN_KEY][cohort], captures[-1][cohort])
        torch.testing.assert_close(second[EXACT_LIVE_HIDDEN_KEY][cohort, :perception.depth_feature_dim],
                                   second["next", "depth_hx"][cohort])


def test_live_cache_owns_ordinary_compact_tensors_not_mutable_rollout_storage():
    policy = _Policy()
    rollout, _ = _collect(policy, _states(policy, [_raw(5)]), 0, 4)
    current, _ = policy._journal_exact_online_rollout(rollout)
    uid = int(current[0, 0, 0])
    entry = policy._exact_online_prefixes[uid]
    before = entry.actor_chunks[0].clone()
    depth, adapt = entry.depth_hx.clone(), entry.adapt_hx.clone()
    rollout[REPLAY_ACTOR_OBSERVATIONS_KEY].zero_()
    rollout[EXACT_LIVE_HIDDEN_KEY].fill_(-100)
    assert torch.equal(entry.actor_chunks[0], before)
    assert torch.equal(entry.depth_hx, depth)
    assert torch.equal(entry.adapt_hx, adapt)
    assert not entry.actor_chunks[0].is_inference()
    assert not entry.depth_hx.is_inference()
    actor = torch.nn.Linear(policy._q_actor_dim, 2)
    actor(entry.actor_chunks[0]).sum().backward()
    assert actor.weight.grad is not None


def test_reuse_preserves_cache_that_already_consumed_pending_observation():
    policy = _Policy()
    fields = _raw(8)
    states = _states(policy, [fields])
    first, carry = _collect(policy, states, 0, 3)
    _, successor = policy._journal_exact_online_rollout(first)
    uid = int(successor[0, -1, 0])
    policy._gather_exact_online_actor(successor[:, -1])
    assert policy._exact_online_prefixes[uid].length == 4
    encoded = policy._exact_online_encoded_nodes
    reused = policy._exact_online_reused_live_nodes
    # Replaying an older capture cannot shrink a prefix already extended by Q.
    policy._reuse_exact_online_live_prefixes(
        first, torch.stack((torch.full((1, 3), uid), torch.arange(3).view(1, 3)), -1),
        generation=policy._exact_cache_generation(), live_generation=policy._exact_cache_generation(),
    )
    assert policy._exact_online_prefixes[uid].length == 4
    assert policy._exact_online_reused_live_nodes == reused

    second, _ = _collect(policy, states, 3, 6, carry=carry)
    current, _ = policy._journal_exact_online_rollout(second)

    assert policy._exact_online_prefixes[uid].length == 6
    assert policy._exact_online_reused_live_nodes - reused == 2
    actual = policy._gather_exact_online_actor(current.reshape(-1, 2))
    expected, direct = policy.direct(fields, stop=6)
    torch.testing.assert_close(actual, expected[3:6])
    torch.testing.assert_close(policy._exact_online_prefixes[uid].adapt_hx, direct["next", "adapt_hx"][0, -1])
    assert policy._exact_online_encoded_nodes == encoded


def test_new_ema_carry_refresh_then_live_suffix_reuse_is_exact():
    policy = _Policy()
    fields = _raw(8)
    states = _states(policy, [fields])
    first, carry = _collect(policy, states, 0, 3)
    policy._journal_exact_online_rollout(first)
    with torch.no_grad():
        policy.temporal_depth_gru_ema.gru.gru.bias_hh.add_(0.31)
    policy._perception_ema_generation += 1
    carry = policy.refresh_exact_perception_carry(carry)
    assert policy._exact_online_encoded_nodes == 3

    second, _ = _collect(policy, states, 3, 6, carry=carry)
    current, _ = policy._journal_exact_online_rollout(second)

    actual = policy._gather_exact_online_actor(current.reshape(-1, 2))
    expected, _ = policy.direct(fields, stop=6)
    torch.testing.assert_close(actual, expected[3:6])
    assert policy._exact_online_encoded_nodes == 3
    assert policy._exact_online_reused_live_nodes == 6


@pytest.mark.parametrize("missing", [EXACT_LIVE_HIDDEN_KEY, EXACT_LIVE_GENERATION_KEY, REPLAY_ACTOR_OBSERVATIONS_KEY])
def test_missing_optional_capture_falls_back_to_complete_exact_encoding(missing):
    policy = _Policy()
    fields = _raw(6)
    rollout, _ = _collect(policy, _states(policy, [fields]), 0, 5)
    rollout.del_(missing)
    current, _ = policy._journal_exact_online_rollout(rollout)
    assert policy._exact_online_reused_live_nodes == 0
    actual = policy._gather_exact_online_actor(current[:, -1])
    expected, _ = policy.direct(fields, stop=5)
    torch.testing.assert_close(actual[0], expected[-1])
    assert policy._exact_online_encoded_nodes == 5


def test_old_generation_outputs_are_reencoded_not_reused():
    policy = _Policy()
    fields = _raw(6)
    rollout, _ = _collect(policy, _states(policy, [fields]), 0, 5)
    with torch.no_grad():
        policy.temporal_depth_gru_ema.gru.gru.bias_hh.add_(0.31)
    policy._perception_ema_generation += 1
    current, _ = policy._journal_exact_online_rollout(rollout)
    assert policy._exact_online_reused_live_nodes == 0
    assert policy._exact_online_live_generation is None
    actual = policy._gather_exact_online_actor(current[:, -1])
    expected, _ = policy.direct(fields, stop=5)
    torch.testing.assert_close(actual[0], expected[-1])
    assert policy._exact_online_encoded_nodes == 5


def test_stale_live_carry_and_cache_gap_never_seed_a_continuing_episode():
    policy = _Policy()
    fields = _raw(8)
    states = _states(policy, [fields])
    first, carry = _collect(policy, states, 0, 3)
    policy._journal_exact_online_rollout(first)
    with torch.no_grad():
        policy.temporal_depth_gru_ema.gru.gru.bias_hh.add_(0.31)
    policy._perception_ema_generation += 1
    # Deliberately violate the normal trainer's refresh boundary. Generation
    # stamps alone are insufficient because the incoming hidden state is old.
    second, _ = _collect(policy, states, 3, 6, carry=carry)
    reused = policy._exact_online_reused_live_nodes
    current, _ = policy._journal_exact_online_rollout(second)
    assert policy._exact_online_reused_live_nodes == reused
    assert policy._exact_online_live_generation is None
    actual = policy._gather_exact_online_actor(current[:, -1])
    expected, _ = policy.direct(fields, stop=6)
    torch.testing.assert_close(actual[0], expected[-1])
    assert policy._exact_online_encoded_nodes == 6


@pytest.mark.parametrize("key", ["depth_hx", "adapt_hx"])
def test_nonzero_live_reset_input_hidden_cannot_seed_exact_reset_cache(key):
    policy = _Policy()
    fields = _raw(6)
    states = _states(policy, [fields])
    carry = states[:, 0].clone()
    carry[key].fill_(0.7)
    # The actual one-step GRU does not mask is_init, so this simulates a broken
    # environment-primer reset and really produces a non-reset live latent.
    rollout, _ = _collect(policy, states, 0, 5, carry=carry)

    current, _ = policy._journal_exact_online_rollout(rollout)

    assert policy._exact_online_reused_live_nodes == 0
    assert policy._exact_online_live_generation is None
    actual = policy._gather_exact_online_actor(current[:, -1])
    expected, _ = policy.direct(fields, stop=5)
    torch.testing.assert_close(actual[0], expected[-1])
    assert policy._exact_online_encoded_nodes == 5


def test_internal_reset_uses_pre_autoreset_hidden_and_does_not_mix_episodes():
    policy = _Policy()
    first, second = _raw(3), _raw(4, seed=914)
    fields = {key: torch.cat((first[key], second[key])) for key in first}
    rollout, _ = _collect(policy, _states(policy, [fields]), 0, 6)
    current, _ = policy._journal_exact_online_rollout(rollout)
    u, v = int(current[0, 0, 0]), int(current[0, 3, 0])
    assert u != v
    assert policy._exact_online_reused_live_nodes == 6
    _, first_direct = policy.direct(first)
    _, second_direct = policy.direct(second, stop=3)
    for uid, direct in ((u, first_direct), (v, second_direct)):
        entry = policy._exact_online_prefixes[uid]
        torch.testing.assert_close(entry.depth_hx, direct["next", "depth_hx"][0, -1])
        torch.testing.assert_close(entry.adapt_hx, direct["next", "adapt_hx"][0, -1])
    assert bool(rollout["next", "depth_hx"][0, 2].eq(0).all())
    assert not bool(policy._exact_online_prefixes[u].depth_hx.eq(0).all())


def test_timeout_successor_extends_seeded_old_episode_not_reset_episode():
    policy = _Policy()
    first, second = _raw(3), _raw(4, seed=914)
    fields = {key: torch.cat((first[key], second[key])) for key in first}
    rollout, _ = _collect(policy, _states(policy, [fields]), 0, 6)
    timeout = _raw(1, seed=956)
    timeout["is_init"].fill_(False)
    final_td = _states(policy, [timeout])[:, 0]
    policy._truncation_final_batches = [{
        **{"exact_input__" + key: value for key, value in policy._exact_perception_inputs(final_td).items()},
        "indices": torch.tensor([2]),
    }]

    current, successor = policy._journal_exact_online_rollout(rollout)

    old_uid = int(current[0, 2, 0])
    new_uid = int(current[0, 3, 0])
    assert old_uid != new_uid
    assert successor[0, 2].tolist() == [old_uid, 3]
    assert policy._exact_online_reused_live_nodes == 6
    actual = policy._gather_exact_online_actor(successor[0, 2:3])
    reference = {key: torch.cat((first[key], timeout[key])) for key in first}
    expected, direct = policy.direct(reference)
    torch.testing.assert_close(actual[0], expected[-1])
    torch.testing.assert_close(policy._exact_online_prefixes[old_uid].depth_hx, direct["next", "depth_hx"][0, -1])
    assert policy._exact_online_encoded_nodes == 1
    assert policy._exact_online_prefixes[new_uid].length == 3


def test_capture_only_fields_are_removed_before_perception_minibatching():
    class Parent:
        def _live_perception_rollout(self, rollout):
            return rollout

    class Harness(ExactOnlinePerceptionReplayMixin, Parent):
        pass

    rollout = TensorDict({"policy": torch.ones(2, 3, 4), EXACT_LIVE_HIDDEN_KEY: torch.ones(2, 3, 8),
                          EXACT_LIVE_GENERATION_KEY: torch.zeros(2, 3, 1, dtype=torch.long)}, [2, 3])
    result = Harness()._live_perception_rollout(rollout)
    assert set(result.keys()) == {"policy"}
    assert EXACT_LIVE_HIDDEN_KEY in rollout
