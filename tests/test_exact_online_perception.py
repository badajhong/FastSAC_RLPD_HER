from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.envs.transforms import CatTensors, ObservationNorm

from active_adaptation.learning.ppo.exact_online_perception import (
    EXACT_CURRENT_REF,
    EXACT_NEXT_REF,
    ExactOnlinePerceptionReplayMixin,
)
from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule,
    GRUModule,
    TemporalDepthGRU,
    TransformObject,
    set_recurrent_mode,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    PERCEPTION_OBJECT_GEO_ID_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    REPLAY_SAMPLE_IS_DAGGER_ENV_KEY,
    REPLAY_SAMPLE_IS_TEACHER_KEY,
    REPLAY_SAMPLE_PHYSICAL_INDEX_KEY,
    _TD3ReplaySamplePlan,
)


class _PreparedReplayBase(nn.Module):
    def _prepare_dagger_learning_batch(self, batch):
        return {
            output: batch[key]
            for output, key in (
                ("observations", REPLAY_ACTOR_OBSERVATIONS_KEY),
                ("next_observations", REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY),
            )
            if key in batch
        }


class _Policy(ExactOnlinePerceptionReplayMixin, _PreparedReplayBase):
    """Production recurrent modules with a small depth feature extractor."""

    def _exact_online_replay_enabled(self):
        return True

    def __init__(self, *, chunk=3, residual=True):
        super().__init__()
        self.cfg = SimpleNamespace(
            train_every=chunk,
            perception_encode_microbatch_size=64,
            latent_dim=4,
            use_object_adapt=True,
        )
        self.device = torch.device("cpu")
        self.depth_feature_dim = 4
        self.q_actor_keys = ("vel_command", "policy", "priv_pred")
        self._q_actor_dim = 7
        self._perception_ema_generation = 0
        self._replay_vecnorm_fingerprint = "frozen-input-normalizer"
        self._replay_object_geo_fingerprint = "exact-geometry-bank"
        self._teacher_prefill_active = lambda: False
        with torch.random.fork_rng():
            torch.manual_seed(401)
            self.geometry = torch.randn(3, 384)
            self.temporal_depth_gru_ema = TemporalDepthGRU(nn.Linear(3, 4), hidden_dim=4)
            self.object_adapt_ema = TensorDictSequential(
                CatTensors(["policy", "vel_command", "_depth_feature"], "object_input", sort=False, del_keys=False),
                TensorDictModule(nn.Linear(7, 12), ["object_input"], ["object_pred"]),
            )
            self.object_pred_transform = TransformObject(
                12, ["object_pred", "object_geo_"], ["object_pred_trans"]
            )
            core = DepthResidualGRUModule(4, 4) if residual else GRUModule(4)
            keys = ["adapt_input", "is_init", "adapt_hx"]
            if residual:
                keys.append("_depth_feature")
                # Exercise the active residual route, rather than its zero init.
                with torch.no_grad():
                    core.depth_projection.weight.normal_(0.0, 0.13)
            self.adapt_ema = TensorDictSequential(
                CatTensors(["policy", "object_pred", "object_pred_trans"], "adapt_input", sort=False, del_keys=False),
                TensorDictModule(core, keys, ["priv_pred", ("next", "adapt_hx")]),
            )
            self._run_stack(self._inputs(_raw(2), stop=2))
        self.requires_grad_(False)
        self._ensure_exact_online_state()

    def _decode_replay_object_geo(self, indices, *, device, dtype):
        return self.geometry[indices.long()].to(device=device, dtype=dtype)

    def _encode_replay_object_geo(self, td):
        geometry = td["object_geo_"]
        matching = (geometry.unsqueeze(-2) == self.geometry).all(-1)
        assert bool(matching.sum(-1).eq(1).all())
        return matching.long().argmax(-1)

    def _inputs(self, raw, *, stop=None):
        stop = len(raw["policy"]) if stop is None else stop
        data = {key: value[:stop].unsqueeze(0).clone() for key, value in raw.items()}
        data["object_geo_"] = self._decode_replay_object_geo(
            data.pop(PERCEPTION_OBJECT_GEO_ID_KEY), device="cpu", dtype=torch.float32
        )
        data["depth_hx"] = torch.zeros(1, stop, self.depth_feature_dim)
        data["adapt_hx"] = torch.zeros(1, stop, int(self.cfg.latent_dim))
        return TensorDict(data, [1, stop])

    def _run_stack(self, td, *, recurrent=True):
        with torch.no_grad(), set_recurrent_mode(recurrent):
            self.temporal_depth_gru_ema(td)
            self.object_adapt_ema(td)
            self.object_pred_transform(td)
            self.adapt_ema(td)
        return td

    def direct(self, raw, *, stop=None):
        td = self._run_stack(self._inputs(raw, stop=stop))
        return torch.cat([td[key] for key in self.q_actor_keys], -1)[0], td

    def add(self, raw):
        store = self._exact_online_store
        uid = store.allocate_episode_uid()
        store.append(uid, 0, raw)
        return uid


def _raw(length, *, seed=411):
    generator = torch.Generator().manual_seed(seed)
    reset = torch.zeros(length, 1, dtype=torch.bool)
    reset[0] = True
    return {
        "depth": torch.randn(length, 3, generator=generator),
        "policy": torch.randn(length, 2, generator=generator),
        "vel_command": torch.randn(length, 1, generator=generator),
        PERCEPTION_OBJECT_GEO_ID_KEY: torch.arange(length) % 3,
        "is_init": reset,
    }


@pytest.mark.parametrize("chunk", [1, 3, 32])
@pytest.mark.parametrize("residual", [False, True])
def test_chunked_cache_matches_complete_reset_prefix_for_both_recurrent_states(chunk, residual):
    policy = _Policy(chunk=chunk, residual=residual)
    raw = [_raw(11), _raw(14, seed=412)]
    uids = [policy.add(value) for value in raw]
    refs = torch.tensor([(uids[1], 12), (uids[0], 8), (uids[1], 3), (uids[0], 0)])
    expected = [policy.direct(value)[0] for value in raw]

    actual = policy._gather_exact_online_actor(refs)

    torch.testing.assert_close(actual, torch.stack([expected[1][12], expected[0][8], expected[1][3], expected[0][0]]))
    assert policy._exact_online_encoded_nodes == 13 + 9
    for uid, fields, stop in zip(uids, raw, (9, 13)):
        _, direct = policy.direct(fields, stop=stop)
        cached = policy._exact_online_prefixes[uid]
        torch.testing.assert_close(cached.depth_hx, direct["next", "depth_hx"][0, -1])
        torch.testing.assert_close(cached.adapt_hx, direct["next", "adapt_hx"][0, -1])


def test_cache_reuses_shared_prefix_then_rebuilds_under_new_ema_weights():
    policy = _Policy(chunk=3)
    raw = _raw(13)
    uid = policy.add(raw)
    calls = []
    hook = policy.temporal_depth_gru_ema.register_forward_hook(lambda _, args, output: calls.append(tuple(output.batch_size)))
    try:
        first = policy._gather_exact_online_actor(torch.tensor([(uid, 2), (uid, 5)]))
        first_calls = len(calls)
        assert policy._exact_online_encoded_nodes == 6

        repeated = policy._gather_exact_online_actor(torch.tensor([(uid, 5), (uid, 2)]))
        torch.testing.assert_close(repeated, first.flip(0), rtol=0, atol=0)
        assert len(calls) == first_calls
        policy._gather_exact_online_actor(torch.tensor([(uid, 9), (uid, 4)]))
        assert policy._exact_online_encoded_nodes == 10
        assert sum(n * t for n, t in calls) == 10

        old = policy._gather_exact_online_actor(torch.tensor([(uid, 9)]))
        with torch.no_grad():
            policy.temporal_depth_gru_ema.gru.gru.weight_hh.add_(0.3)
            next(module for module in policy.adapt_ema.modules() if isinstance(module, DepthResidualGRUModule)).depth_projection.weight.add_(0.25)
        policy._perception_ema_generation += 1
        new = policy._gather_exact_online_actor(torch.tensor([(uid, 9)]))
        assert policy._exact_online_encoded_nodes == 20
        assert not torch.allclose(old[:, -4:], new[:, -4:])
        expected, _ = policy.direct(raw, stop=10)
        torch.testing.assert_close(new[0], expected[-1])
    finally:
        hook.remove()


def _ring(current, successor):
    return SimpleNamespace(
        device=torch.device("cpu"), size=len(current),
        data={EXACT_CURRENT_REF: torch.tensor(current), EXACT_NEXT_REF: torch.tensor(successor)},
    )


def test_mixed_actor_and_q_batches_replace_only_online_rows_from_correct_rings():
    policy = _Policy()
    first, second = _raw(9), _raw(9, seed=418)
    u, v = policy.add(first), policy.add(second)
    policy.dagger_replay = _ring([(u, 1), (u, 4)], [(u, 2), (u, 5)])
    policy.student_replay = _ring([(v, 3), (v, 6)], [(v, 4), (v, 7)])
    source = torch.tensor([True, False, False, True, False, False])
    dagger = torch.tensor([True, True, False, False, False, True])
    physical = torch.tensor([100, 1, 0, 200, 1, 0])
    batch = {
        REPLAY_SAMPLE_IS_TEACHER_KEY: source,
        REPLAY_SAMPLE_IS_DAGGER_ENV_KEY: dagger,
        REPLAY_SAMPLE_PHYSICAL_INDEX_KEY: physical,
        REPLAY_ACTOR_OBSERVATIONS_KEY: torch.full((6, 7), -17.0),
        REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY: torch.full((6, 7), -23.0),
    }
    original = {key: value.clone() for key, value in batch.items()}
    plan = SimpleNamespace(student_indices=torch.tensor([0, 1]), pure_student_indices=torch.tensor([0, 1]),
                           actor_indices=torch.tensor([0, 1]), actor_pure_student_indices=torch.tensor([0, 1]))
    policy._prepare_exact_online_replay_cache([plan])
    encoded_nodes = policy._exact_online_encoded_nodes

    actual = policy._prepare_dagger_learning_batch(batch)

    assert policy._exact_online_encoded_nodes == encoded_nodes == 6 + 8
    direct_u, _ = policy.direct(first)
    direct_v, _ = policy.direct(second)
    expected_current = torch.stack([torch.full((7,), -17.0), direct_u[4], direct_v[3],
                                    torch.full((7,), -17.0), direct_v[6], direct_u[1]])
    expected_next = torch.stack([torch.full((7,), -23.0), direct_u[5], direct_v[4],
                                 torch.full((7,), -23.0), direct_v[7], direct_u[2]])
    torch.testing.assert_close(actual["observations"], expected_current)
    torch.testing.assert_close(actual["next_observations"], expected_next)
    for key, value in original.items():
        assert torch.equal(batch[key], value)
    # Actor gradients must use refreshed online latents while EMA stays frozen.
    actor = nn.Linear(7, 2)
    actor(actual["observations"]).square().mean().backward()
    observed_grad = actor.weight.grad.clone()
    actor.zero_grad()
    actor(expected_current).square().mean().backward()
    torch.testing.assert_close(observed_grad, actor.weight.grad)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_sample_plan_union_encodes_only_requested_episode_prefixes_once():
    policy = _Policy(chunk=3)
    raw = [_raw(17), _raw(15, seed=441), _raw(20, seed=442)]
    u, v, unrequested = [policy.add(fields) for fields in raw]
    policy.dagger_replay = _ring(
        [(u, 1), (u, 4), (u, 9), (unrequested, 17)],
        [(u, 2), (u, 5), (u, 10), (unrequested, 18)],
    )
    policy.student_replay = _ring(
        [(v, 2), (v, 6), (v, 11)], [(v, 3), (v, 7), (v, 12)]
    )

    def plan(q_dagger, q_student, actor_dagger, actor_student):
        indices = lambda value: None if value is None else torch.tensor(value, dtype=torch.long)
        return _TD3ReplaySamplePlan(
            # Teacher rows have their own cache and cannot index either ring.
            teacher_indices=torch.tensor([1000]), student_indices=indices(q_dagger),
            permutation=torch.arange(1 + len(q_dagger) + len(q_student or [])),
            actor_indices=indices(actor_dagger), actor_teacher_indices=None,
            pure_student_indices=indices(q_student), actor_pure_student_indices=indices(actor_student),
        )

    plans = (
        plan([0, 2, 2], [1], [1, 0], [0]),
        plan([1], [0, 1], None, None),  # Delayed Actor update.
        plan([], None, [0, 1, 1], [1, 0]),
        plan([], [], None, []),  # Entirely Teacher-sourced update.
    )
    expected = {uid: policy.direct(fields)[0] for uid, fields in zip((u, v), raw)}
    calls = []
    hook = policy.temporal_depth_gru_ema.register_forward_hook(
        lambda _, args, output: calls.append(tuple(output.batch_size))
    )
    try:
        policy._prepare_exact_online_replay_cache(None)
        policy._prepare_exact_online_replay_cache([])
        assert not calls

        policy._prepare_exact_online_replay_cache(plans)

        # Q successors need u[0..10] and v[0..7]. Actor requests are subsets.
        assert policy._exact_online_encoded_nodes == 11 + 8
        assert sum(n * t for n, t in calls) == 11 + 8 + policy._exact_online_padded_nodes
        assert set(policy._exact_online_prefixes) == {u, v}
        assert unrequested not in policy._exact_online_prefixes
        call_count = len(calls)
        for sample in plans:
            for replay, name, next_state in (
                (policy.dagger_replay, "student_indices", True),
                (policy.student_replay, "pure_student_indices", True),
                (policy.dagger_replay, "actor_indices", False),
                (policy.student_replay, "actor_pure_student_indices", False),
            ):
                sampled = getattr(sample, name)
                if sampled is None or not sampled.numel():
                    continue
                refs = policy._exact_ring_refs(replay, sampled, next_state=next_state)
                actual = policy._gather_exact_online_actor(refs)
                reference = torch.stack([expected[uid][step] for uid, step in refs.tolist()])
                torch.testing.assert_close(actual, reference)
        assert len(calls) == call_count
        assert policy._exact_online_encoded_nodes == 11 + 8
    finally:
        hook.remove()


def test_live_carry_refresh_stops_before_pending_input_and_matches_next_single_step():
    policy = _Policy(chunk=3)
    raw = _raw(8)
    uid = policy.add(raw)
    policy._exact_online_carry_refs = torch.tensor([(uid, 7)])
    # Q used this successor in the old generation before perception changed.
    policy._gather_exact_online_actor(torch.tensor([(uid, 7)]))
    with torch.no_grad():
        policy.temporal_depth_gru_ema.gru.gru.bias_hh.add_(0.4)
    policy._perception_ema_generation += 1
    carry = policy._inputs(raw)[:, -1]
    carry["depth_hx"].fill_(101.0)
    carry["adapt_hx"].fill_(-101.0)
    original_depth = carry["depth_hx"].clone()

    refreshed = policy.refresh_exact_perception_carry(carry)

    assert policy._exact_online_prefixes[uid].length == 7
    assert torch.equal(carry["depth_hx"], original_depth)
    _, preceding = policy.direct(raw, stop=7)
    torch.testing.assert_close(refreshed["depth_hx"], preceding["next", "depth_hx"][:, -1])
    torch.testing.assert_close(refreshed["adapt_hx"], preceding["next", "adapt_hx"][:, -1])
    _, complete = policy.direct(raw)
    live = policy._run_stack(refreshed, recurrent=False)
    torch.testing.assert_close(live["priv_pred"], complete["priv_pred"][:, -1])
    torch.testing.assert_close(live["next", "depth_hx"], complete["next", "depth_hx"][:, -1])
    torch.testing.assert_close(live["next", "adapt_hx"], complete["next", "adapt_hx"][:, -1])


def test_unchanged_live_generation_reuses_carry_even_when_q_cached_pending_state():
    policy = _Policy()
    raw = _raw(8)
    uid = policy.add(raw)
    policy._exact_online_carry_refs = torch.tensor([(uid, 7)])
    policy._exact_online_live_generation = policy._exact_cache_generation()
    _, preceding = policy.direct(raw, stop=7)
    carry = policy._inputs(raw)[:, -1]
    carry["depth_hx"] = preceding["next", "depth_hx"][:, -1].clone()
    carry["adapt_hx"] = preceding["next", "adapt_hx"][:, -1].clone()
    policy._gather_exact_online_actor(torch.tensor([(uid, 7)]))
    assert policy._exact_online_prefixes[uid].length == 8
    encoded_nodes = policy._exact_online_encoded_nodes

    assert policy.refresh_exact_perception_carry(carry) is carry
    assert policy._exact_online_encoded_nodes == encoded_nodes
    _, complete = policy.direct(raw)
    live = policy._run_stack(carry.clone(), recurrent=False)
    torch.testing.assert_close(live["priv_pred"], complete["priv_pred"][:, -1])


def _captured_inputs(policy, td):
    return {"exact_input__" + key: value for key, value in policy._exact_perception_inputs(td).items()}


def test_journal_keeps_timeout_successor_on_old_episode_and_overlapping_live_history():
    policy = _Policy()
    raw = _raw(6)
    raw["is_init"][3] = True
    inputs = policy._inputs(raw)
    policy._rollout_final_batch = _captured_inputs(policy, inputs[:, -1])
    timeout = policy._inputs(_raw(1, seed=931))[:, 0]
    timeout["is_init"].fill_(False)
    policy._truncation_final_batches = [{**_captured_inputs(policy, timeout), "indices": torch.tensor([2])}]

    current, successor = policy._journal_exact_online_rollout(inputs[:, :5])

    old_uid = int(current[0, 0, 0])
    new_uid = int(current[0, 3, 0])
    assert new_uid != old_uid
    assert torch.equal(successor[0, 2], torch.tensor([old_uid, 3]))
    assert torch.equal(current[0, 3], torch.tensor([new_uid, 0]))
    old_raw = {key: torch.cat((value[:3], policy._exact_perception_inputs(timeout)[key]), dim=0)
               for key, value in raw.items()}
    old_actor, _ = policy.direct(old_raw)
    timeout_actor = policy._gather_exact_online_actor(successor[0, 2:3])
    torch.testing.assert_close(timeout_actor[0], old_actor[-1])

    # Next rollout shares exactly one pending observation with the journal.
    continuation = _raw(3, seed=934)
    for key in continuation:
        continuation[key][0] = raw[key][-1]
    next_inputs = policy._inputs(continuation)
    policy._rollout_final_batch = _captured_inputs(policy, next_inputs[:, -1])
    policy._truncation_final_batches = []
    next_current, next_successor = policy._journal_exact_online_rollout(next_inputs[:, :2])
    assert torch.equal(next_current[0, 0], successor[0, -1])
    complete_raw = {key: torch.cat((value[3:], continuation[key][1:]), dim=0) for key, value in raw.items()}
    direct, _ = policy.direct(complete_raw)
    actual = policy._gather_exact_online_actor(next_successor[0, -1:])
    torch.testing.assert_close(actual[0], direct[-1])
    assert policy._exact_online_store.length(new_uid) == 5


def test_full_depth_cnn_ema_matches_frozen_normalized_reset_forward_and_live_steps():
    from test_tvkd_depth_residual import build_full_ppovel

    # Constructor fixture contains the actual 36x64 camera CNN, depth GRU,
    # object predictor and object geometry transform, and adaptation GRU.
    with torch.random.fork_rng():
        torch.manual_seed(701)
        source = build_full_ppovel(residual=True)
    policy = _Policy(chunk=3)
    policy.cfg.latent_dim = int(source.cfg.latent_dim)
    policy.depth_feature_dim = int(source.depth_feature_dim)
    policy._q_actor_dim = 5 + 10 + int(source.cfg.latent_dim)
    for name in ("temporal_depth_gru_ema", "object_adapt_ema", "object_pred_transform", "adapt_ema"):
        setattr(policy, name, getattr(source, name))
    core = next(module for module in policy.adapt_ema.modules() if isinstance(module, DepthResidualGRUModule))
    generator = torch.Generator().manual_seed(708)
    with torch.no_grad():
        core.depth_projection.weight.copy_(torch.randn(core.depth_projection.weight.shape, generator=generator) * 0.1)
    n, time = 2, 10
    td = TensorDict({
        "depth": torch.randint(0, 101, (n, time, 1, 36, 64), generator=generator).float() / 100,
        "policy": torch.randn(n, time, 10, generator=generator),
        "vel_command": torch.randn(n, time, 5, generator=generator),
        "object_geo_": policy.geometry[torch.arange(n)].unsqueeze(1).expand(n, time, -1).clone(),
        "is_init": torch.zeros(n, time, 1, dtype=torch.bool),
        "depth_hx": torch.zeros(n, time, policy.depth_feature_dim),
        "adapt_hx": torch.zeros(n, time, int(policy.cfg.latent_dim)),
    }, [n, time])
    td["is_init"][:, 0] = True
    normalizer = ObservationNorm(loc=0.15, scale=0.7, standard_normal=True,
                                 in_keys=["depth", "policy", "vel_command"])
    normalized = normalizer(td.clone())
    assert not torch.equal(normalized["depth"], td["depth"])
    lengths = (7, 10)
    fields = [policy._exact_perception_inputs(normalized[env, :length])
              for env, length in enumerate(lengths)]
    uids = [policy.add(value) for value in fields]
    for uid, raw in zip(uids, fields):
        stored = policy._exact_online_store.prefix(uid, len(raw["depth"]))
        for key, value in raw.items():
            assert torch.equal(stored[key], value)

    complete = policy._run_stack(normalized.clone())
    assert core.depth_projection(complete["_depth_feature"]).abs().max().item() > 1e-3
    expected = torch.cat([complete[key] for key in policy.q_actor_keys], -1)
    refs = torch.tensor([(uids[1], 9), (uids[0], 6), (uids[0], 1), (uids[1], 4)])

    cached = policy._gather_exact_online_actor(refs)

    torch.testing.assert_close(cached, torch.stack([expected[1, 9], expected[0, 6], expected[0, 1], expected[1, 4]]))
    assert policy._exact_online_encoded_nodes == sum(lengths)
    depth_hx = torch.zeros(n, policy.depth_feature_dim)
    adapt_hx = torch.zeros(n, int(policy.cfg.latent_dim))
    live_actor = []
    for step in range(time):
        live = normalized[:, step].clone()
        live["depth_hx"], live["adapt_hx"] = depth_hx, adapt_hx
        policy._run_stack(live, recurrent=False)
        depth_hx, adapt_hx = live["next", "depth_hx"], live["next", "adapt_hx"]
        live_actor.append(torch.cat([live[key] for key in policy.q_actor_keys], -1))
    torch.testing.assert_close(torch.stack(live_actor, dim=1), expected)
    torch.testing.assert_close(depth_hx, complete["next", "depth_hx"][:, -1])
    torch.testing.assert_close(adapt_hx, complete["next", "adapt_hx"][:, -1])
    for uid, raw, length in zip(uids, fields, lengths):
        _, direct = policy.direct(raw)
        entry = policy._exact_online_prefixes[uid]
        assert entry.length == length
        torch.testing.assert_close(entry.depth_hx, direct["next", "depth_hx"][0, -1])
        torch.testing.assert_close(entry.adapt_hx, direct["next", "adapt_hx"][0, -1])
