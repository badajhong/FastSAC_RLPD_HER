from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
)
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
    NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
    TEACHER_ACTUATOR_CONTEXT_FIELD,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    PPOBCDaggerFinetune,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY,
    Q_ACTOR_STATE_SEMANTICS,
    Q_TERMINATION_COUNTER_CONTEXT_KEY,
    Q_TERMINATION_COUNTER_CONTEXT_SEMANTICS,
    REFERENCE_PHASE_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    REPLAY_PERCEPTION_EMA_GENERATION_KEY,
    PERCEPTION_IS_INIT_KEY,
    DistributionalTD3TeacherBC,
    DistributionalTD3TeacherBCConfig,
    _TD3DeviceReplay,
    _TD3ReplaySamplePlan,
    _PREFILL_COMMAND_FINISHED_KEY,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _PREFILL_TERMINATED_KEY,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBCConfig,
)


def _load_cumulative_error_mixin_without_isaac():
    """Compile the exact mixin AST without importing the Isaac environment.

    Importing ``active_adaptation.envs`` initializes USD/PhysX modules that are
    intentionally unavailable in the ordinary CPU unit-test process. The mixin
    itself depends only on torch, so extracting its class node tests the real
    implementation while keeping this focused contract suite CPU-collectable.
    """
    source_path = (
        Path(__file__).parents[1]
        / "active_adaptation/envs/mdp/commands/hdmi/terminations.py"
    )
    parsed = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "_cum_error_mixin"
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[class_node], type_ignores=[])
    )
    namespace = {"torch": torch}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["_cum_error_mixin"]


_cum_error_mixin = _load_cumulative_error_mixin_without_isaac()


def _actuator_metadata() -> dict:
    return {
        "enabled": True,
        "semantics": FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
        "dimension": 6,
        "delay_range": [2, 6],
        "alpha_range": [0.8, 1.0],
    }


def _bare_conditioned_policy(
    *,
    actor_state: bool = True,
    actuator_state: bool = True,
    termination_counters: bool = False,
) -> DistributionalTD3TeacherBC:
    policy = DistributionalTD3TeacherBC.__new__(DistributionalTD3TeacherBC)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_condition_on_actor_state=actor_state,
        q_condition_on_actuator_state=actuator_state,
        q_condition_on_termination_counters=termination_counters,
    )
    # Actor cache layout: [vel_command(2), duplicated_policy_obs(3), belief(2)].
    policy._q_critic_dim = 3
    policy._q_actor_dim = 7
    policy._q_vel_command_dim = 2
    policy._q_belief_dim = 2
    policy._q_actor_state_dim = 4 if actor_state else 0
    policy._q_actuator_context_dim = 6 if actuator_state else 0
    policy._q_termination_counter_dim = 2 if termination_counters else 0
    policy._q_input_dim = (
        policy._q_critic_dim
        + policy._q_actor_state_dim
        + policy._q_actuator_context_dim
        + policy._q_termination_counter_dim
    )
    policy._q_actuator_context_metadata_value = (
        _actuator_metadata() if actuator_state else {"enabled": False}
    )
    return policy


def test_q_markov_state_defaults_are_fresh_tvkd_only():
    td3 = DistributionalTD3TeacherBCConfig()
    fastsac = DistributionalFastSACTeacherBCConfig()
    tvkd = TVKDDistributionalFastSACTeacherBCConfig()

    assert td3.q_condition_on_actor_state is False
    assert td3.q_condition_on_actuator_state is False
    assert td3.q_condition_on_termination_counters is False
    assert fastsac.q_condition_on_actor_state is False
    assert fastsac.q_condition_on_actuator_state is False
    assert fastsac.q_condition_on_termination_counters is False
    assert tvkd.q_condition_on_actor_state is True
    assert tvkd.q_condition_on_actuator_state is True
    assert tvkd.q_condition_on_termination_counters is True


class _CounterBase:
    def __init__(self, *, num_envs: int, device: str = "cpu") -> None:
        self._counter_num_envs = int(num_envs)
        self._counter_device = torch.device(device)

    @property
    def num_envs(self) -> int:
        return self._counter_num_envs

    @property
    def device(self) -> torch.device:
        return self._counter_device


class _TestCumulativeCounter(_cum_error_mixin, _CounterBase):
    pass


def test_cumulative_termination_public_progress_preserves_pre_reset_snapshot():
    counter = _TestCumulativeCounter(num_envs=3, min_steps=4, threshold=0.5)
    assert torch.equal(
        counter.q_cumulative_counter_progress(), torch.zeros(3, 1)
    )

    counter.error.copy_(torch.tensor([0.8, 0.1, 0.9]))
    counter.update()
    counter.update()
    expected_before_reset = torch.tensor([[0.5], [0.0], [0.5]])
    assert torch.equal(
        counter.q_cumulative_counter_progress(), expected_before_reset
    )
    assert torch.equal(
        counter.q_cumulative_counter_progress(after_last_update=True),
        expected_before_reset,
    )

    counter.reset(torch.tensor([0]))
    assert torch.equal(
        counter.q_cumulative_counter_progress(),
        torch.tensor([[0.0], [0.0], [0.5]]),
    )
    # Reset may clear the live next episode, but must not erase the transition's
    # post-action state needed by replay for a terminal or timeout row.
    preserved = counter.q_cumulative_counter_progress(after_last_update=True)
    assert torch.equal(preserved, expected_before_reset)
    assert preserved.dtype == torch.float32
    assert preserved.requires_grad is False

    preserved.fill_(1.0)
    assert torch.equal(
        counter.q_cumulative_counter_progress(after_last_update=True),
        expected_before_reset,
    )
    with pytest.raises(TypeError):
        counter.q_cumulative_counter_progress(after_last_update=1)


def test_cumulative_termination_progress_is_normalized_and_saturates():
    counter = _TestCumulativeCounter(num_envs=2, min_steps=2, threshold=0.5)
    counter.error.copy_(torch.tensor([0.8, 0.2]))
    for _ in range(5):
        counter.update()
    assert torch.equal(
        counter.q_cumulative_counter_progress(), torch.tensor([[1.0], [0.0]])
    )
    with pytest.raises(ValueError, match="min_steps"):
        _TestCumulativeCounter(num_envs=1, min_steps=0)


def test_q_input_appends_exact_vel_belief_actuator_order_and_detaches_context():
    policy = _bare_conditioned_policy()
    assert policy._q_actor_state_metadata() == {
        "enabled": True,
        "semantics": Q_ACTOR_STATE_SEMANTICS,
        "dimension": 4,
        "vel_command_dim": 2,
        "belief_dim": 2,
    }
    critic = torch.tensor(
        [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], requires_grad=True
    )
    actor = torch.tensor(
        [
            [1.0, 2.0, 101.0, 102.0, 103.0, 7.0, 8.0],
            [3.0, 4.0, 201.0, 202.0, 203.0, 9.0, 10.0],
        ],
        requires_grad=True,
    )
    actuator = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],
        ],
        requires_grad=True,
    )

    actual = policy._q_observation_input(
        critic,
        actor_observations=actor,
        actuator_context=actuator,
    )
    expected = torch.cat(
        (
            critic,
            actor[:, :2].detach(),
            actor[:, -2:].detach(),
            actuator.detach(),
        ),
        dim=-1,
    )

    assert actual.shape == (2, 13)
    assert torch.equal(actual, expected)
    # The duplicated policy-observation slice [2:5] must not enter Q twice.
    assert not torch.isin(actor[:, 2:5].detach(), actual[:, 3:]).any()

    actual.sum().backward()
    assert torch.equal(critic.grad, torch.ones_like(critic))
    assert actor.grad is None
    assert actuator.grad is None


def test_q_input_appends_termination_counters_after_actor_and_actuator_state():
    policy = _bare_conditioned_policy(termination_counters=True)
    critic = torch.tensor([[10.0, 11.0, 12.0]], requires_grad=True)
    actor = torch.tensor(
        [[1.0, 2.0, 101.0, 102.0, 103.0, 7.0, 8.0]],
        requires_grad=True,
    )
    actuator = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, -0.5]], requires_grad=True
    )
    counters = torch.tensor([[0.25, 0.75]], requires_grad=True)

    actual = policy._q_observation_input(
        critic,
        actor_observations=actor,
        actuator_context=actuator,
        termination_counter_context=counters,
    )
    expected = torch.cat(
        (
            critic,
            actor[:, :2].detach(),
            actor[:, -2:].detach(),
            actuator.detach(),
            counters.detach(),
        ),
        dim=-1,
    )

    assert actual.shape == (1, 15)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.equal(critic.grad, torch.ones_like(critic))
    assert actor.grad is None
    assert actuator.grad is None
    assert counters.grad is None

    with pytest.raises(ValueError, match="requires cumulative"):
        policy._q_observation_input(
            critic.detach(),
            actor_observations=actor.detach(),
            actuator_context=actuator.detach(),
        )
    with pytest.raises(ValueError, match="shape"):
        policy._q_observation_input(
            critic.detach(),
            actor_observations=actor.detach(),
            actuator_context=actuator.detach(),
            termination_counter_context=torch.zeros(1, 3),
        )
    with pytest.raises(ValueError, match="floating point"):
        policy._q_observation_input(
            critic.detach(),
            actor_observations=actor.detach(),
            actuator_context=actuator.detach(),
            termination_counter_context=torch.zeros(1, 2, dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("actor", "actuator"),
    (
        (None, torch.zeros(2, 6)),
        (torch.zeros(2, 7), None),
        (torch.zeros(2, 6), torch.zeros(2, 6)),
        (torch.zeros(3, 7), torch.zeros(2, 6)),
        (torch.zeros(2, 7, dtype=torch.long), torch.zeros(2, 6)),
        (torch.zeros(2, 7), torch.zeros(2, 5)),
        (torch.zeros(2, 7), torch.zeros(2, 6, dtype=torch.long)),
    ),
)
def test_q_input_fails_closed_on_missing_or_misaligned_enabled_state(
    actor: torch.Tensor | None,
    actuator: torch.Tensor | None,
):
    policy = _bare_conditioned_policy()
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        policy._q_observation_input(
            torch.zeros(2, 3),
            actor_observations=actor,
            actuator_context=actuator,
        )


def test_q_input_disabled_path_rejects_unexpected_context_without_copying_baseline():
    policy = _bare_conditioned_policy(actor_state=False, actuator_state=False)
    critic = torch.randn(2, 3)

    assert policy._q_observation_input(critic) is critic
    with pytest.raises((ValueError, RuntimeError)):
        policy._q_observation_input(
            critic, actor_observations=torch.zeros(2, 7)
        )
    with pytest.raises((ValueError, RuntimeError)):
        policy._q_observation_input(
            critic, actuator_context=torch.zeros(2, 6)
        )
    with pytest.raises((ValueError, RuntimeError)):
        policy._q_observation_input(
            critic, termination_counter_context=torch.zeros(2, 2)
        )


def test_q_batch_state_kwargs_select_exact_current_and_next_fields():
    policy = _bare_conditioned_policy(termination_counters=True)
    current_actor = torch.randn(2, 7)
    next_actor = torch.randn(2, 7)
    current_actuator = torch.randn(2, 6)
    next_actuator = torch.randn(2, 6)
    current_counters = torch.rand(2, 2)
    next_counters = torch.rand(2, 2)
    batch = {
        "observations": current_actor,
        "next_observations": next_actor,
        TEACHER_ACTUATOR_CONTEXT_FIELD: current_actuator,
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: next_actuator,
        Q_TERMINATION_COUNTER_CONTEXT_KEY: current_counters,
        NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY: next_counters,
    }

    current = policy._q_batch_state_kwargs(batch, next_state=False)
    following = policy._q_batch_state_kwargs(batch, next_state=True)

    assert current["actor_observations"] is current_actor
    assert current["actuator_context"] is current_actuator
    assert following["actor_observations"] is next_actor
    assert following["actuator_context"] is next_actuator
    assert current["termination_counter_context"] is current_counters
    assert following["termination_counter_context"] is next_counters
    with pytest.raises(KeyError):
        policy._q_batch_state_kwargs(
            {key: value for key, value in batch.items() if key != "next_observations"},
            next_state=True,
        )


@pytest.mark.parametrize(
    "invalid_key",
    (
        REPLAY_ACTOR_OBSERVATIONS_KEY,
        REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
        TEACHER_ACTUATOR_CONTEXT_FIELD,
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
        Q_TERMINATION_COUNTER_CONTEXT_KEY,
        NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY,
    ),
)
def test_replay_batch_rejects_nonfinite_state_once_before_hot_q_path(
    monkeypatch: pytest.MonkeyPatch,
    invalid_key: str,
):
    # Isolate the state-contract layer from unrelated VecNorm/replay preparation.
    monkeypatch.setattr(
        PPOBCDaggerFinetune,
        "_prepare_dagger_learning_batch",
        lambda _owner, batch: dict(batch),
    )
    policy = _bare_conditioned_policy(termination_counters=True)
    policy._student_collection_actor_cache_enabled = MethodType(
        lambda _owner: True, policy
    )
    batch = {
        "critic_observations": torch.zeros(2, 3),
        "next_critic_observations": torch.ones(2, 3),
        REPLAY_ACTOR_OBSERVATIONS_KEY: torch.zeros(2, 7),
        REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY: torch.ones(2, 7),
        TEACHER_ACTUATOR_CONTEXT_FIELD: torch.zeros(2, 6),
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: torch.ones(2, 6),
        Q_TERMINATION_COUNTER_CONTEXT_KEY: torch.zeros(2, 2),
        NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY: torch.ones(2, 2),
    }
    # Use +Inf rather than NaN so this test cannot poison PyTorch's caching
    # allocator and expose unrelated tests that compare unused `empty` tails
    # with `torch.equal` (for which NaN is unequal to itself).
    batch[invalid_key][0, 0] = float("inf")

    # The one-time replay preparation owns finite validation so repeated Q
    # forwards do not introduce CUDA synchronizations in the hot update loop.
    with pytest.raises(ValueError, match="invalid"):
        policy._prepare_dagger_learning_batch(batch)


def _termination_context_policy():
    first = _TestCumulativeCounter(num_envs=3, min_steps=4, threshold=0.5)
    second = _TestCumulativeCounter(num_envs=3, min_steps=2, threshold=0.5)
    termination_funcs = OrderedDict(
        (("body_pos", first), ("lost_contact", second))
    )
    policy = _bare_conditioned_policy(
        actor_state=False,
        actuator_state=False,
        termination_counters=True,
    )
    policy.device = torch.device("cpu")
    policy.env = SimpleNamespace(
        num_envs=3,
        termination_funcs=termination_funcs,
    )
    policy._q_termination_counter_metadata_value = (
        policy._resolve_q_termination_counter_metadata()
    )
    policy._q_termination_counter_dim = int(
        policy._q_termination_counter_metadata_value["dimension"]
    )
    policy._rollout_q_termination_counter_contexts = []
    return policy, first, second


def test_termination_counter_metadata_and_capture_preserve_configured_order():
    policy, first, second = _termination_context_policy()
    assert policy._q_termination_counter_metadata_value == {
        "enabled": True,
        "semantics": Q_TERMINATION_COUNTER_CONTEXT_SEMANTICS,
        "dimension": 2,
        "names": ["body_pos", "lost_contact"],
        "min_steps": [4, 2],
        "normalization": "clamp(cumulative_steps/min_steps,0,1)",
    }

    first.error.copy_(torch.tensor([0.8, 0.1, 0.8]))
    second.error.copy_(torch.tensor([0.8, 0.8, 0.1]))
    first.update()
    second.update()
    captured = policy.capture_q_termination_counter_context()

    assert captured is not None
    assert captured.requires_grad is False
    assert torch.equal(
        captured,
        torch.tensor([[0.25, 0.5], [0.0, 0.5], [0.25, 0.0]]),
    )
    captured.zero_()
    assert torch.equal(
        policy.capture_q_termination_counter_context(),
        torch.tensor([[0.25, 0.5], [0.0, 0.5], [0.25, 0.0]]),
    )


def test_timeout_reset_records_post_update_counter_as_true_next_state():
    policy, first, second = _termination_context_policy()
    first.error.copy_(torch.tensor([0.8, 0.1, 0.8]))
    # Row 0 stays below every min_steps boundary: its reset below represents a
    # pure time limit, not a cumulative-error physical termination.
    second.error.copy_(torch.tensor([0.1, 0.8, 0.1]))
    first.update()
    second.update()
    current = policy.capture_q_termination_counter_context()

    # Simulate the next environment step followed by an immediate timeout reset
    # of row 0. The post-update channel must remain the transition's s_(t+1).
    first.update()
    second.update()
    following = policy.capture_q_termination_counter_context(
        after_last_update=True
    )
    # Row 1 reaches lost_contact progress 1.0 (physical terminal); row 2 is an
    # ordinary nonterminal. Reset both done rows exactly as the vector env does.
    assert torch.equal(following[1], torch.tensor([0.0, 1.0]))
    assert torch.equal(following[2], torch.tensor([0.5, 0.0]))
    first.reset(torch.tensor([0, 1]))
    second.reset(torch.tensor([0, 1]))
    reset_carry = policy.capture_q_termination_counter_context()

    assert current is not None and following is not None and reset_carry is not None
    assert torch.equal(current[0], torch.tensor([0.25, 0.0]))
    assert torch.equal(following[0], torch.tensor([0.5, 0.0]))
    assert torch.equal(reset_carry[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(reset_carry[1], torch.tensor([0.0, 0.0]))
    assert torch.equal(reset_carry[2], following[2])
    assert torch.equal(
        policy.capture_q_termination_counter_context(after_last_update=True)[0],
        following[0],
    )

    policy.record_rollout_q_termination_counter_context(current, following)
    expected_current = current.clone()
    expected_following = following.clone()
    current.fill_(0.9)
    following.fill_(0.9)
    recorded_current, recorded_following = (
        policy._consume_rollout_q_termination_counter_contexts(1)
    )
    assert recorded_current.shape == (3, 1, 2)
    assert recorded_following.shape == (3, 1, 2)
    assert torch.equal(recorded_current[:, 0], expected_current)
    assert torch.equal(recorded_following[:, 0], expected_following)


def test_delay_one_hot_and_centered_alpha_encoding_are_exact_and_snapshotted():
    policy = _bare_conditioned_policy(actor_state=False, actuator_state=True)
    manager = SimpleNamespace(
        min_delay=2,
        max_delay=6,
        alpha_range=(0.8, 1.0),
        delay=torch.tensor([[2], [3], [4], [5], [6]]),
        alpha=torch.tensor([[0.8], [0.85], [0.9], [0.95], [1.0]]),
    )
    policy.env = SimpleNamespace(action_manager=manager)
    policy._q_actuator_context_metadata_value = (
        policy._resolve_q_actuator_context_metadata()
    )

    captured = policy.capture_q_actuator_context()

    assert captured is not None
    assert captured.requires_grad is False
    assert torch.equal(captured[:, :5], torch.eye(5))
    assert torch.allclose(
        captured[:, -1],
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
        atol=1e-6,
    )
    assert policy._q_actuator_context_metadata_value == _actuator_metadata()

    # The saved pre-step transition context must not alias a manager reset.
    manager.delay.fill_(6)
    manager.alpha.fill_(1.0)
    assert torch.equal(captured[:, :5], torch.eye(5))


def _prefill_counter_rows(
    action: float,
    current: tuple[float, float],
    following: tuple[float, float],
    *,
    step: int,
    success: bool,
) -> dict[str, torch.Tensor]:
    return {
        "actions": torch.tensor([[action]]),
        Q_TERMINATION_COUNTER_CONTEXT_KEY: torch.tensor([current]),
        NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY: torch.tensor([following]),
        "dones": torch.tensor([success]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True]),
        DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([False]),
        _PREFILL_ENV_INDEX_KEY: torch.tensor([0]),
        _PREFILL_STEP_INDEX_KEY: torch.tensor([step]),
        _PREFILL_TERMINATED_KEY: torch.tensor([False]),
        _PREFILL_COMMAND_FINISHED_KEY: torch.tensor([success]),
        PERCEPTION_IS_INIT_KEY: torch.zeros(1, 2, dtype=torch.bool),
    }


def test_teacher_prefill_staging_preserves_counter_current_next_across_chunks():
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(q_condition_on_termination_counters=True)
    policy.q_teacher_replay = _TD3DeviceReplay(4, "cpu")
    policy._q_termination_counter_dim = 2
    policy._teacher_prefill_pending = None
    policy._teacher_prefill_successful_episodes = 0
    policy._teacher_prefill_failed_episodes = 0
    policy._teacher_prefill_timeout_episodes = 0
    policy._teacher_prefill_incomplete_episodes = 0
    policy._teacher_prefill_discarded_rows = 0
    policy._teacher_prefill_successful_by_motion = {}
    policy._teacher_episode_cache_enabled = MethodType(
        lambda _owner: False, policy
    )
    policy._q_replay_prefill_storage_fields = MethodType(
        lambda _owner: (
            "actions",
            Q_TERMINATION_COUNTER_CONTEXT_KEY,
            NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY,
        ),
        policy,
    )

    first = _prefill_counter_rows(
        1.0, (0.0, 0.25), (0.25, 0.5), step=0, success=False
    )
    second = _prefill_counter_rows(
        2.0, (0.25, 0.5), (0.5, 0.75), step=1, success=True
    )
    assert policy._stage_teacher_prefill_rows(first) == (0, 0)
    assert policy._stage_teacher_prefill_rows(second) == (2, 0)

    assert policy.q_teacher_replay.size == 2
    assert torch.equal(
        policy.q_teacher_replay.data[Q_TERMINATION_COUNTER_CONTEXT_KEY][:2],
        torch.tensor([[0.0, 0.25], [0.25, 0.5]]),
    )
    assert torch.equal(
        policy.q_teacher_replay.data[NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY][:2],
        torch.tensor([[0.25, 0.5], [0.5, 0.75]]),
    )


def _q_rows(count: int, offset: float) -> dict[str, torch.Tensor]:
    row = torch.arange(count, dtype=torch.float32) + offset
    return {
        "critic_observations": torch.stack((row, row + 0.25), dim=-1),
        "actions": row[:, None],
        "rewards": row + 1.0,
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.ones(count),
        "next_critic_observations": torch.stack((row + 10.0, row + 10.25), dim=-1),
        REFERENCE_PHASE_KEY: torch.linspace(0.0, 1.0, count),
        REPLAY_PERCEPTION_EMA_GENERATION_KEY: torch.zeros(count, dtype=torch.long),
    }


def test_balanced_q_replay_keeps_teacher_and_student_current_next_state_aligned():
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        perception_replay_mode="online_student_rollout",
        q_condition_on_actor_state=True,
        q_condition_on_actuator_state=True,
        q_condition_on_termination_counters=True,
        q_batch_size=4,
        dagger_batch_size=4,
        q_teacher_replay_ratio=0.5,
        teacher_actor_replay_fraction=0.5,
        failure_phase_teacher_fraction=0.0,
        failure_phase_student_fraction=0.0,
    )
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(17)
    policy._q_actor_dim = 7
    policy._q_vel_command_dim = 2
    policy._q_belief_dim = 2
    policy._q_actor_state_dim = 4
    policy._q_actuator_context_dim = 6
    policy._q_termination_counter_dim = 2
    policy._q_actuator_context_metadata_value = _actuator_metadata()
    policy.q_teacher_replay = _TD3DeviceReplay(8, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(8, "cpu")
    policy._teacher_prefill_complete = False

    teacher = _q_rows(3, 100.0)
    teacher[TEACHER_ACTUATOR_CONTEXT_FIELD] = torch.stack(
        (torch.arange(6), torch.arange(6) + 10, torch.arange(6) + 20)
    ).float()
    teacher[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD] = (
        teacher[TEACHER_ACTUATOR_CONTEXT_FIELD] + 0.5
    )
    teacher[Q_TERMINATION_COUNTER_CONTEXT_KEY] = torch.tensor(
        [[0.0, 0.25], [0.5, 0.75], [1.0, 0.0]]
    )
    teacher[NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY] = torch.tensor(
        [[0.25, 0.5], [0.75, 1.0], [0.0, 0.25]]
    )
    policy.q_teacher_replay.extend(teacher)

    student = _q_rows(3, 1_000.0)
    student[REPLAY_ACTOR_OBSERVATIONS_KEY] = torch.stack(
        (torch.arange(7), torch.arange(7) + 10, torch.arange(7) + 20)
    ).float()
    student[REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY] = (
        student[REPLAY_ACTOR_OBSERVATIONS_KEY] + 0.25
    )
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = student["actions"] + 1.0
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(
        3, dtype=torch.bool
    )
    student[TEACHER_ACTUATOR_CONTEXT_FIELD] = torch.stack(
        (torch.arange(6) + 30, torch.arange(6) + 40, torch.arange(6) + 50)
    ).float()
    student[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD] = (
        student[TEACHER_ACTUATOR_CONTEXT_FIELD] + 0.75
    )
    student[Q_TERMINATION_COUNTER_CONTEXT_KEY] = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
    student[NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY] = torch.tensor(
        [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]
    )
    policy.dagger_replay.extend(student)

    def teacher_actor_cache(owner, indices, *, next_state):
        base = 200.0 if next_state else 100.0
        rows = indices.reshape(-1).float()
        return rows[:, None] + base + torch.arange(7).float()[None]

    policy._teacher_actor_observations_for_indices = MethodType(
        teacher_actor_cache, policy
    )
    plan = _TD3ReplaySamplePlan(
        teacher_indices=torch.tensor([0, 2]),
        student_indices=torch.tensor([1, 2]),
        permutation=torch.arange(4),
        actor_indices=None,
        actor_teacher_indices=None,
        teacher_focused=torch.zeros(2, dtype=torch.bool),
        student_focused=torch.zeros(2, dtype=torch.bool),
    )

    batch = policy._sample_balanced_q_batch(plan)

    expected_teacher_current = teacher_actor_cache(
        policy, plan.teacher_indices, next_state=False
    )
    expected_teacher_next = teacher_actor_cache(
        policy, plan.teacher_indices, next_state=True
    )
    assert torch.equal(
        batch[REPLAY_ACTOR_OBSERVATIONS_KEY][:2], expected_teacher_current
    )
    assert torch.equal(
        batch[REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY][:2], expected_teacher_next
    )
    assert torch.equal(
        batch[REPLAY_ACTOR_OBSERVATIONS_KEY][2:],
        student[REPLAY_ACTOR_OBSERVATIONS_KEY][plan.student_indices],
    )
    assert torch.equal(
        batch[REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY][2:],
        student[REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY][plan.student_indices],
    )
    for key in (
        TEACHER_ACTUATOR_CONTEXT_FIELD,
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
        Q_TERMINATION_COUNTER_CONTEXT_KEY,
        NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY,
    ):
        expected = torch.cat(
            (teacher[key][plan.teacher_indices], student[key][plan.student_indices])
        )
        assert torch.equal(batch[key], expected)

    # The Actor consumes only s_t. It must carry the current counter/actuator
    # and exact Actor cache, without accidentally requiring any s_(t+1) field.
    policy._prepare_dagger_learning_batch = MethodType(
        lambda _owner, raw_batch: raw_batch, policy
    )
    actor_batch = policy._sample_actor_batch(
        indices=plan.student_indices,
        teacher_indices=plan.teacher_indices,
        teacher_focused=torch.zeros(2, dtype=torch.bool),
        student_focused=torch.zeros(2, dtype=torch.bool),
    )
    assert Q_TERMINATION_COUNTER_CONTEXT_KEY in actor_batch
    assert NEXT_Q_TERMINATION_COUNTER_CONTEXT_KEY not in actor_batch
    assert TEACHER_ACTUATOR_CONTEXT_FIELD in actor_batch
    assert NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD not in actor_batch
    assert REPLAY_ACTOR_OBSERVATIONS_KEY in actor_batch
    assert REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY not in actor_batch
    expected_actor_counter = torch.cat(
        (
            teacher[Q_TERMINATION_COUNTER_CONTEXT_KEY][plan.teacher_indices],
            student[Q_TERMINATION_COUNTER_CONTEXT_KEY][plan.student_indices],
        )
    )
    assert torch.equal(
        actor_batch[Q_TERMINATION_COUNTER_CONTEXT_KEY], expected_actor_counter
    )
