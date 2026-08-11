import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTOR_BACKEND,
    FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS,
    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_REFERENCE_DUELING_ARCHITECTURE_SEMANTICS,
    FASTSAC_Q_REFERENCE_RESIDUAL_SEMANTICS,
    FASTSAC_Q_RAW_ACTION_SEMANTICS,
    FASTSAC_STAGE1_BEHAVIOR_UNCERTAINTY_GATE_SEMANTICS,
    FASTSAC_STAGE1_REFERENCE_AWAC_SEMANTICS,
    FASTSAC_STAGE1_UPDATE_MODE,
    FASTSAC_TEACHER_TRAINING_ALGORITHM,
    NEXT_TEACHER_REF_ACTION_FIELD,
    TEACHER_REF_ACTION_FIELD,
    DistributionalQNetwork,
    FastSACActor,
    FastSACTanhNormal,
    FastSACVEL,
    FastSACVelConfig,
    FastSACVelFinetune,
    TeacherTrainingReplayBuffer,
    TwinDistributionalQ,
    _FastSACVAICBase,
    _build_isolated_q_network,
    _fastsac_target_entropy,
    _measure_or_clip_grad_norm,
    _policy_replay_conservative_q_penalty,
    _reference_awac_weights,
    _reduce_actor_q_values,
    _q_action_hidden_dim,
    _sac_bootstrap_mask,
    _select_c51_twin_target,
    _Stage1NStepAccumulator,
    _validate_fastsac_teacher_config,
    _validate_pure_fastsac_checkpoint_provenance,
)
from active_adaptation.learning.ppo.ppo_vel import REF_JPOS_KEY


def _teacher_transition_policy():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(train_every=3, phase="train")
    policy.q_critic_keys = ["critic"]
    policy._teacher_actor_keys = ["teacher"]
    policy._q_critic_dim = 1
    policy._fastsac_teacher_actor_dim = 1
    policy._teacher_raw_replay_fields = []
    policy.action_dim = 1
    policy.reward_scales = torch.tensor([0.25, 0.75])
    policy._truncation_final_batches = []
    policy._last_truncation_finals_used = 0
    policy._rollout_final_batch = {
        "next_critic_observations": torch.tensor([[92.0], [192.0]]),
    }
    return policy


def _teacher_rollout():
    n, t = 2, 3
    marker = torch.tensor(
        [[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]]
    )
    done = torch.zeros(n, t, 1, dtype=torch.bool)
    done[1, 2] = True
    terminated = torch.zeros_like(done)
    episode_time_limit = torch.zeros_like(done)
    episode_time_limit[1, 2] = True
    command_finished = torch.zeros_like(done)
    reward = torch.stack((marker, marker + 4.0), dim=-1).squeeze(-2)
    return TensorDict(
        {
            "actor_a": marker,
            "actor_b": marker + 20.0,
            "critic": marker + 30.0,
            "teacher": marker + 40.0,
            ACTION_KEY: marker + 50.0,
            # Match asynchronous VAIC resets.  Teacher FastSAC must retain
            # only rows whose per-episode step_count is greater than one.
            "step_count": torch.tensor(
                [[[0], [1], [2]], [[2], [3], [4]]]
            ),
            "next": TensorDict(
                {
                    "reward": reward,
                    "done": done,
                    "terminated": terminated,
                    "discount": torch.ones(n, t, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": episode_time_limit,
                            "command_finished": command_finished,
                        },
                        batch_size=[n, t],
                    ),
                },
                batch_size=[n, t],
            ),
        },
        batch_size=[n, t],
    )


def test_teacher_transitions_keep_timeout_final_state_alignment():
    policy = _teacher_transition_policy()
    rollout = _teacher_rollout()
    policy._truncation_final_batches.append(
        {
            # Last row of env 1: env * T + step = 1 * 3 + 2.
            "indices": torch.tensor([5]),
            "next_critic_observations": torch.tensor([[903.0]]),
        }
    )

    transitions = policy._teacher_transitions(rollout)

    assert transitions["rewards"].shape == (4,)
    assert "observations" not in transitions
    assert "next_observations" not in transitions
    assert torch.equal(
        transitions["critic_observations"][:, 0],
        torch.tensor([40.0, 41.0, 32.0, 42.0]),
    )
    assert torch.equal(
        transitions["next_critic_observations"][:, 0],
        torch.tensor([41.0, 42.0, 92.0, 903.0]),
    )
    assert policy._last_truncation_finals_used == 1
    assert policy._truncation_final_batches == []
    assert policy._rollout_final_batch is None


def test_teacher_replay_reconstructs_timeout_next_reference():
    policy = _teacher_transition_policy()
    policy._teacher_raw_replay_fields = [
        (
            REF_JPOS_KEY,
            TEACHER_REF_ACTION_FIELD,
            NEXT_TEACHER_REF_ACTION_FIELD,
        )
    ]
    policy._q_critic_widths = [1]
    policy.object_transform = lambda td: td
    policy.encoder_priv = lambda td: td
    policy._rollout_final_batch.update({
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[960.0], [1960.0]])
    })

    rollout = _teacher_rollout()
    marker = rollout["actor_a"]
    rollout[REF_JPOS_KEY] = marker + 60.0
    policy._truncation_final_batches.append({
        # Last row of env 1: env * T + step = 1 * 3 + 2.
        "indices": torch.tensor([5]),
        "next_critic_observations": torch.tensor([[903.0]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[1903.0]]),
    })

    transitions = policy._teacher_transitions(rollout)

    assert torch.equal(
        transitions[TEACHER_REF_ACTION_FIELD].squeeze(-1),
        torch.tensor([70.0, 71.0, 62.0, 72.0]),
    )
    assert torch.equal(
        transitions[NEXT_TEACHER_REF_ACTION_FIELD].squeeze(-1),
        torch.tensor([71.0, 72.0, 960.0, 1903.0]),
    )

    current_state = policy._teacher_state_from_replay(
        transitions, next_state=False
    )
    next_state = policy._teacher_state_from_replay(transitions, next_state=True)
    assert torch.equal(
        current_state[REF_JPOS_KEY], transitions[TEACHER_REF_ACTION_FIELD]
    )
    assert torch.equal(
        next_state[REF_JPOS_KEY],
        transitions[NEXT_TEACHER_REF_ACTION_FIELD],
    )


def test_interleaved_teacher_transition_filters_step_count_at_insertion():
    policy = _teacher_transition_policy()
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": torch.full((td.batch_size[0], 1), 98.0),
    }
    current = TensorDict(
        {
            "critic": torch.tensor([[21.0], [22.0], [23.0]]),
            ACTION_KEY: torch.tensor([[31.0], [32.0], [33.0]]),
            "step_count": torch.tensor([[0], [1], [2]]),
            "next": TensorDict(
                {
                    "reward": torch.ones(3, 2),
                    "done": torch.zeros(3, 1, dtype=torch.bool),
                    "terminated": torch.zeros(3, 1, dtype=torch.bool),
                    "discount": torch.ones(3, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(
                                3, 1, dtype=torch.bool
                            ),
                            "command_finished": torch.zeros(
                                3, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[3],
            ),
        },
        batch_size=[3],
    )

    transitions = policy._teacher_transition_from_step(
        current, TensorDict({}, batch_size=[3])
    )

    assert transitions["rewards"].shape == (1,)
    assert "observations" not in transitions
    assert "next_observations" not in transitions
    assert torch.equal(
        transitions["critic_observations"][:, 0], torch.tensor([23.0])
    )
    assert torch.equal(transitions["actions"][:, 0], torch.tensor([33.0]))


def test_interleaved_teacher_uses_pre_reset_timeout_final_observation():
    policy = _teacher_transition_policy()
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": td["marker"].clone(),
    }
    done = torch.tensor([[False], [True], [True]])
    terminated = torch.tensor([[False], [False], [True]])
    current = TensorDict(
        {
            "critic": torch.tensor([[1.0], [2.0], [3.0]]),
            ACTION_KEY: torch.tensor([[11.0], [12.0], [13.0]]),
            "step_count": torch.full((3, 1), 2),
            "next": TensorDict(
                {
                    # This is the first step_and_maybe_reset return: the real
                    # simulator state before asynchronous reset.
                    "marker": torch.tensor([[101.0], [102.0], [103.0]]),
                    "reward": torch.ones(3, 2),
                    "done": done,
                    "terminated": terminated,
                    "discount": torch.ones(3, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [[False], [True], [True]]
                            ),
                            "command_finished": torch.zeros(
                                3, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[3],
            ),
        },
        batch_size=[3],
    )
    reset_carry = TensorDict(
        {"marker": torch.tensor([[201.0], [202.0], [203.0]])},
        batch_size=[3],
    )

    transitions = policy._teacher_transition_from_step(current, reset_carry)

    assert torch.equal(
        transitions["next_critic_observations"].squeeze(-1),
        # Ordinary row uses carry. Timeout must use 102, not reset
        # carry 202. The simultaneous true termination does not bootstrap.
        torch.tensor([201.0, 102.0, 203.0]),
    )
    assert torch.equal(
        transitions["truncations"], torch.tensor([False, True, False])
    )
    assert torch.equal(
        transitions["dones"], torch.tensor([False, True, True])
    )
    assert policy._last_truncation_finals_used == 1


def test_command_completion_ends_teacher_return_without_bootstrap():
    policy = _teacher_transition_policy()
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": td["marker"].clone(),
    }
    current = TensorDict(
        {
            "critic": torch.tensor([[2.0]]),
            ACTION_KEY: torch.tensor([[12.0]]),
            "step_count": torch.tensor([[2]]),
            "next": TensorDict(
                {
                    "marker": torch.tensor([[102.0]]),
                    "reward": torch.ones(1, 1),
                    "done": torch.tensor([[True]]),
                    "terminated": torch.tensor([[False]]),
                    "discount": torch.ones(1, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor([[False]]),
                            "command_finished": torch.tensor([[True]]),
                        },
                        batch_size=[1],
                    ),
                },
                batch_size=[1],
            ),
        },
        batch_size=[1],
    )
    reset_carry = TensorDict(
        {"marker": torch.tensor([[202.0]])}, batch_size=[1]
    )

    transition = policy._teacher_transition_from_step(current, reset_carry)

    assert torch.equal(transition["dones"], torch.tensor([True]))
    assert torch.equal(transition["truncations"], torch.tensor([False]))
    assert torch.equal(
        _sac_bootstrap_mask(
            transition["dones"], transition["truncations"]
        ),
        torch.zeros(1),
    )
    # The reset carry is harmless because the stored bootstrap mask is zero.
    assert torch.equal(
        transition["next_critic_observations"], torch.tensor([[202.0]])
    )


def test_teacher_collector_hook_captures_timeout_final_before_reset():
    policy = _teacher_transition_policy()
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": td["marker"].clone(),
    }
    td = TensorDict(
        {
            "next": TensorDict(
                {
                    "marker": torch.tensor([[101.0], [102.0], [103.0]]),
                    "done": torch.tensor([[False], [True], [True]]),
                    "terminated": torch.tensor([[False], [False], [True]]),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [[False], [True], [True]]
                            ),
                            "command_finished": torch.zeros(
                                3, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[3],
            )
        },
        batch_size=[3],
    )

    policy.capture_truncation_final_observations(td, step=2)

    assert len(policy._truncation_final_batches) == 1
    captured = policy._truncation_final_batches[0]
    assert torch.equal(captured["indices"], torch.tensor([5]))
    assert torch.equal(
        captured["next_critic_observations"], torch.tensor([[102.0]])
    )


def _n_step_batch(
    rewards,
    *,
    markers=None,
    dones=None,
    truncations=None,
    discounts=None,
):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    count = rewards.numel()
    markers = (
        torch.arange(count, dtype=torch.float32)
        if markers is None
        else torch.as_tensor(markers, dtype=torch.float32)
    )
    dones = (
        torch.zeros(count, dtype=torch.bool)
        if dones is None
        else torch.as_tensor(dones, dtype=torch.bool)
    )
    truncations = (
        torch.zeros(count, dtype=torch.bool)
        if truncations is None
        else torch.as_tensor(truncations, dtype=torch.bool)
    )
    discounts = (
        torch.ones(count)
        if discounts is None
        else torch.as_tensor(discounts, dtype=torch.float32)
    )
    return {
        "critic_observations": markers[:, None],
        "actions": (markers + 100.0)[:, None],
        "rewards": rewards,
        "dones": dones,
        "truncations": truncations,
        "discounts": discounts,
        "next_critic_observations": (markers + 0.5)[:, None],
    }


def test_stage1_four_step_return_uses_gamma_and_nonunit_env_discounts():
    accumulator = _Stage1NStepAccumulator(
        n_steps=4,
        gamma=0.5,
        next_fields=("next_critic_observations",),
    )
    emitted = []
    for reward, marker, discount in zip(
        (1.0, 2.0, 4.0, 8.0),
        (10.0, 11.0, 12.0, 13.0),
        (0.5, 0.25, 1.0, 0.4),
    ):
        emitted.append(accumulator.append(
            _n_step_batch(
                [reward], markers=[marker], discounts=[discount]
            ),
            torch.tensor([True]),
        ))

    assert [batch["rewards"].numel() for batch in emitted] == [0, 0, 0, 1]
    result = emitted[-1]
    # 1 + (.5*.5)*2 + (.5*.5*.5*.25)*4
    #   + (.5*.5*.5*.25*.5*1)*8
    assert torch.allclose(result["rewards"], torch.tensor([1.75]))
    assert torch.allclose(result["discounts"], torch.tensor([0.05]))
    assert torch.equal(result["effective_n_steps"], torch.tensor([4.0]))
    assert torch.equal(result["critic_observations"], torch.tensor([[10.0]]))
    assert torch.equal(
        result["next_critic_observations"], torch.tensor([[13.5]])
    )
    assert torch.allclose(
        (0.5 ** result["effective_n_steps"]) * result["discounts"],
        torch.tensor([0.003125]),
    )


def test_stage1_n_step_keeps_start_reference_and_endpoint_next_reference():
    accumulator = _Stage1NStepAccumulator(
        n_steps=3,
        gamma=0.99,
        next_fields=(
            "next_critic_observations",
            NEXT_TEACHER_REF_ACTION_FIELD,
        ),
    )
    result = None
    for marker in (10.0, 11.0, 12.0):
        transition = _n_step_batch([1.0], markers=[marker])
        transition[TEACHER_REF_ACTION_FIELD] = torch.tensor([[marker + 100.0]])
        transition[NEXT_TEACHER_REF_ACTION_FIELD] = torch.tensor(
            [[marker + 100.5]]
        )
        result = accumulator.append(transition, torch.tensor([True]))

    assert result is not None
    assert torch.equal(
        result[TEACHER_REF_ACTION_FIELD], torch.tensor([[110.0]])
    )
    assert torch.equal(
        result[NEXT_TEACHER_REF_ACTION_FIELD], torch.tensor([[112.5]])
    )


def test_stage1_terminal_flushes_partial_returns_without_bootstrap():
    accumulator = _Stage1NStepAccumulator(
        n_steps=4,
        gamma=0.5,
        next_fields=("next_critic_observations",),
    )
    first = accumulator.append(
        _n_step_batch([2.0], markers=[20.0], discounts=[0.25]),
        torch.tensor([True]),
    )
    result = accumulator.append(
        _n_step_batch(
            [4.0],
            markers=[21.0],
            dones=[True],
            truncations=[False],
            discounts=[0.5],
        ),
        torch.tensor([True]),
    )

    assert first["rewards"].numel() == 0
    assert torch.allclose(result["rewards"], torch.tensor([2.5, 4.0]))
    assert torch.allclose(result["discounts"], torch.tensor([0.125, 0.5]))
    assert torch.equal(result["effective_n_steps"], torch.tensor([2.0, 1.0]))
    assert torch.equal(result["dones"], torch.tensor([True, True]))
    assert torch.equal(result["truncations"], torch.tensor([False, False]))
    assert torch.equal(
        _sac_bootstrap_mask(result["dones"], result["truncations"]),
        torch.zeros(2),
    )
    assert torch.equal(
        result["next_critic_observations"], torch.tensor([[21.5], [21.5]])
    )


def test_stage1_timeout_partial_return_bootstraps_from_true_final_state():
    policy = _teacher_transition_policy()
    policy.cfg.sac_teacher_n_steps = 4
    policy.cfg.gamma = 0.5
    policy.reward_scales = torch.ones(1)
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": td["marker"].clone(),
    }

    def step(marker, reward, *, done=False, timeout=False, final=0.0, carry=0.0):
        current = TensorDict(
            {
                "critic": torch.tensor([[marker]]),
                ACTION_KEY: torch.tensor([[marker + 100.0]]),
                "step_count": torch.tensor([[2]]),
                "next": TensorDict(
                    {
                        "marker": torch.tensor([[final]]),
                        "reward": torch.tensor([[reward]]),
                        "done": torch.tensor([[done]]),
                        "terminated": torch.tensor([[False]]),
                        "discount": torch.tensor([[0.5]]),
                        "stats": TensorDict(
                            {
                                "episode_time_limit": torch.tensor([[timeout]]),
                                "command_finished": torch.tensor([[False]]),
                            },
                            batch_size=[1],
                        ),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[1],
        )
        reset_carry = TensorDict(
            {"marker": torch.tensor([[carry]])}, batch_size=[1]
        )
        return policy._teacher_transition_from_step(current, reset_carry)

    first = step(1.0, 2.0, final=2.0, carry=2.0)
    result = step(
        2.0, 4.0, done=True, timeout=True, final=102.0, carry=202.0
    )

    assert first["rewards"].numel() == 0
    assert torch.allclose(result["rewards"], torch.tensor([3.0, 4.0]))
    assert torch.equal(result["effective_n_steps"], torch.tensor([2.0, 1.0]))
    assert torch.equal(result["truncations"], torch.tensor([True, True]))
    assert torch.equal(
        _sac_bootstrap_mask(result["dones"], result["truncations"]),
        torch.ones(2),
    )
    assert torch.equal(
        result["next_critic_observations"], torch.tensor([[102.0], [102.0]])
    )
    # The two returns use d0*d1 and d1 respectively; gamma is applied later.
    assert torch.allclose(result["discounts"], torch.tensor([0.25, 0.5]))


def test_stage1_n_step_keeps_interleaved_envs_and_resets_separate():
    accumulator = _Stage1NStepAccumulator(
        n_steps=4,
        gamma=1.0,
        next_fields=("next_critic_observations",),
    )
    all_valid = torch.tensor([True, True])
    accumulator.append(
        _n_step_batch([1.0, 10.0], markers=[1.0, 10.0]), all_valid
    )
    env0_terminal = accumulator.append(
        _n_step_batch(
            [2.0, 20.0], markers=[2.0, 20.0], dones=[True, False]
        ),
        all_valid,
    )
    reset_transient = accumulator.append(
        _n_step_batch([999.0, 30.0], markers=[999.0, 30.0]),
        torch.tensor([False, True]),
    )
    env1_full = accumulator.append(
        _n_step_batch([3.0, 40.0], markers=[3.0, 40.0]), all_valid
    )

    assert torch.equal(env0_terminal["rewards"], torch.tensor([3.0, 2.0]))
    assert torch.equal(
        env0_terminal["critic_observations"], torch.tensor([[1.0], [2.0]])
    )
    assert reset_transient["rewards"].numel() == 0
    assert torch.equal(env1_full["rewards"], torch.tensor([100.0]))
    assert torch.equal(
        env1_full["critic_observations"], torch.tensor([[10.0]])
    )
    # Env 0's new episode is still only one valid transition long. The invalid
    # reward 999 and the previous episode cannot enter that pending return.
    env0_new_terminal = accumulator.append(
        _n_step_batch(
            [4.0, 0.0], markers=[4.0, 50.0], dones=[True, False]
        ),
        torch.tensor([True, False]),
    )
    assert torch.equal(env0_new_terminal["rewards"], torch.tensor([7.0, 4.0]))
    assert torch.equal(
        env0_new_terminal["critic_observations"], torch.tensor([[3.0], [4.0]])
    )


def test_stage1_default_one_step_is_exact_filtered_transition():
    accumulator = _Stage1NStepAccumulator(
        n_steps=1,
        gamma=0.99,
        next_fields=("next_critic_observations",),
    )
    source = _n_step_batch(
        [1.0, 2.0, 3.0],
        markers=[10.0, 20.0, 30.0],
        dones=[False, True, False],
        truncations=[False, True, False],
        discounts=[0.2, 0.4, 0.8],
    )
    result = accumulator.append(source, torch.tensor([False, True, True]))

    for name, value in source.items():
        assert torch.equal(result[name], value[torch.tensor([False, True, True])])
    assert torch.equal(result["effective_n_steps"], torch.ones(2))
    assert accumulator._lengths is None
    assert accumulator._buffers == {}


def test_stage1_pending_n_step_return_crosses_rollout_boundary():
    policy = _teacher_transition_policy()
    policy.cfg.sac_teacher_n_steps = 4
    policy.cfg.gamma = 1.0
    policy.device = torch.device("cpu")
    accumulator = policy._get_teacher_n_step_accumulator()
    valid = torch.tensor([True])
    accumulator.append(_n_step_batch([1.0], markers=[1.0]), valid)
    accumulator.append(_n_step_batch([2.0], markers=[2.0]), valid)

    policy.begin_transition_collection()

    assert policy._get_teacher_n_step_accumulator() is accumulator
    assert torch.equal(accumulator._lengths, torch.tensor([2]))
    accumulator.append(_n_step_batch([3.0], markers=[3.0]), valid)
    result = accumulator.append(_n_step_batch([4.0], markers=[4.0]), valid)
    assert torch.equal(result["rewards"], torch.tensor([10.0]))
    assert torch.equal(result["effective_n_steps"], torch.tensor([4.0]))


class _Replay:
    def __init__(self):
        self.events = []
        self.sample_counts = []
        self.saved = 0
        self.seen = 0
        self.capacity = 10
        self.last_sample = None

    @property
    def size(self):
        return self.saved

    def clear(self):
        self.events.append("clear")
        self.saved = self.seen = 0

    def append(self, transitions):
        self.events.append("append")
        count = int(transitions["rewards"].shape[0])
        self.saved = min(self.capacity, self.saved + count)
        self.seen += count
        return count

    def sample(self, count, device=None, generator=None, fields=None):
        self.events.append("sample")
        self.sample_counts.append(count)
        self.last_sample = {"batch": torch.zeros(count)}
        return self.last_sample


def _metric(value=1.0):
    scalar = torch.tensor(value)
    return {
        "q_loss": scalar,
        "bellman_q_loss": scalar,
        "q1_loss": scalar,
        "q2_loss": scalar,
        "conservative_q_active": scalar,
        "conservative_q_penalty": scalar,
        "conservative_q_loss": scalar,
        "conservative_policy_replay_q_gap": scalar,
        "conservative_positive_gap_fraction": scalar,
        "conservative_above_margin_fraction": scalar,
        "q_grad_norm": scalar,
        "alpha_loss": scalar,
        "target_q_min": scalar,
        "target_q_max": scalar,
    }


def test_sac_reward_is_sum_of_vaic_groups_not_ppo_group_average():
    reward = torch.tensor([[1.5, -0.25, 0.0]])
    scalar = FastSACVEL._scalarize_sac_reward(reward)
    assert torch.equal(scalar, torch.tensor([1.25]))


def test_policy_replay_conservative_q_penalty_is_per_head_and_relative():
    policy_q = torch.tensor(
        [[0.006, -0.004], [0.010, 0.002]], requires_grad=True
    )
    replay_q = torch.zeros_like(policy_q, requires_grad=True)

    per_head, gap = _policy_replay_conservative_q_penalty(
        policy_q, replay_q, margin=0.002, temperature=0.002
    )

    expected = 0.002 * torch.nn.functional.softplus(
        (policy_q - replay_q - 0.002) / 0.002
    )
    assert torch.allclose(per_head, expected.mean(dim=1))
    assert torch.equal(gap, policy_q - replay_q)
    per_head.sum().backward()
    assert torch.all(policy_q.grad > 0.0)
    assert torch.all(replay_q.grad < 0.0)

    with pytest.raises(ValueError, match="matching shapes"):
        _policy_replay_conservative_q_penalty(
            policy_q, replay_q[:, :1], margin=0.0, temperature=0.1
        )


def test_reference_awac_weights_subtract_per_head_then_take_pessimistic_min():
    replay = torch.tensor([[3.0, 5.0, 1.0], [4.0, 1.0, 2.0]])
    reference = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 0.0]])

    weights, advantages = _reference_awac_weights(
        replay, reference, beta=1.0, weight_clip=20.0
    )

    assert torch.equal(advantages, torch.tensor([2.0, -1.0, 0.0]))
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] > weights[2] > weights[1]
    assert not weights.requires_grad


@pytest.mark.parametrize(
    "beta,weight_clip,match",
    [
        (0.0, 20.0, "beta"),
        (float("nan"), 20.0, "beta"),
        (0.01, 0.5, "weight clip"),
        (0.01, float("inf"), "weight clip"),
    ],
)
def test_reference_awac_weights_validate_hyperparameters(
    beta, weight_clip, match
):
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError, match=match):
        _reference_awac_weights(
            values, values, beta=beta, weight_clip=weight_clip
        )


def test_target_entropy_is_defined_in_normalized_action_coordinates():
    low = torch.tensor([-2.0, -4.0])
    high = torch.tensor([2.0, 4.0])
    ratio = 0.5
    target = _fastsac_target_entropy(low, high, ratio)
    assert target == pytest.approx(-1.0)

    shifted_low = low * 7.0 + 3.0
    shifted_high = high * 7.0 + 3.0
    assert _fastsac_target_entropy(
        shifted_low, shifted_high, ratio
    ) == pytest.approx(target)


def test_physical_log_prob_is_converted_back_to_normalized_coordinates():
    policy = FastSACVEL.__new__(FastSACVEL)
    policy._fastsac_action_log_scale_sum = math.log(3.0) + math.log(7.0)
    normalized_log_prob = torch.tensor([1.25, -0.75])
    physical_log_prob = (
        normalized_log_prob - policy._fastsac_action_log_scale_sum
    )

    converted = policy._normalized_action_log_prob(physical_log_prob)

    assert torch.allclose(converted, normalized_log_prob)


def test_target_entropy_matches_hoi_for_unit_bounds_and_rejects_invalid_input():
    assert _fastsac_target_entropy(
        torch.full((3,), -1.0), torch.ones(3), 0.5
    ) == pytest.approx(-1.5)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _fastsac_target_entropy(torch.tensor([-1.0]), torch.tensor([1.0]), float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        _fastsac_target_entropy(torch.tensor([0.0]), torch.tensor([0.0]), 0.5)


def test_clipped_double_q_selects_lower_expected_full_c51_distribution():
    support = torch.tensor([-1.0, 0.0, 1.0])
    target = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ])

    clipped, values = _select_c51_twin_target(target, support, True)

    expected = torch.stack((target[0, 0], target[1, 1]))
    assert torch.equal(clipped[0], expected)
    assert torch.equal(clipped[1], expected)
    assert torch.equal(values, torch.tensor([[-1.0, -1.0], [-1.0, -1.0]]))

    independent, independent_values = _select_c51_twin_target(
        target, support, False
    )
    assert independent is target
    assert torch.equal(
        independent_values, torch.tensor([[-1.0, 1.0], [0.0, -1.0]])
    )


def test_actor_q_reduction_uses_minimum_or_legacy_twin_mean():
    q_values = torch.tensor([[3.0, -1.0], [1.0, 5.0]])
    assert torch.equal(
        _reduce_actor_q_values(q_values, True), torch.tensor([1.0, -1.0])
    )
    assert torch.equal(
        _reduce_actor_q_values(q_values, False), torch.tensor([2.0, 2.0])
    )


def test_q_action_input_uses_unclipped_nominal_joint_coordinates():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
    )
    policy._fastsac_q_action_center = torch.tensor([2.0, 3.0])
    policy._fastsac_q_action_scale = torch.tensor([4.0, 2.0])
    executable = torch.tensor([
        [-2.0, 1.0],
        [2.0, 3.0],
        [6.0, 5.0],
        [-3.0, 8.0],
    ])

    normalized = policy._q_action_input(executable)

    assert torch.equal(normalized[0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(normalized[1], torch.tensor([0.0, 0.0]))
    assert torch.equal(normalized[2], torch.tensor([1.0, 1.0]))
    assert torch.equal(normalized[3], torch.tensor([-1.25, 2.5]))
    # Q normalization never mutates the replay/environment tensor.
    assert torch.equal(executable[-1], torch.tensor([-3.0, 8.0]))

    policy.cfg.sac_q_normalize_actions = False
    assert policy._q_action_input(executable) is executable


def test_q_action_input_gain_applies_after_optional_normalization():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=2.5,
    )
    policy._fastsac_q_action_center = torch.tensor([2.0, 3.0])
    policy._fastsac_q_action_scale = torch.tensor([4.0, 2.0])
    executable = torch.tensor([[0.0, 3.0]])

    assert torch.equal(
        policy._q_action_input(executable),
        torch.tensor([[-1.25, 0.0]]),
    )

    policy.cfg.sac_q_normalize_actions = False
    policy.cfg.sac_q_action_input_gain = 3.0
    assert torch.equal(
        policy._q_action_input(executable),
        executable * 3.0,
    )
    assert torch.equal(executable, torch.tensor([[0.0, 3.0]]))


def test_q_reference_residual_coordinates_scale_without_clamp_and_keep_gradient():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_action_coordinates="reference_residual",
        # This absolute-coordinate option is deliberately irrelevant here.
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=2.0,
    )
    policy._fastsac_q_action_scale = torch.tensor([4.0, 2.0])
    action = torch.tensor([[10.0, -3.0]], requires_grad=True)
    reference = torch.tensor([[2.0, 1.0]])

    q_action = policy._q_action_input(action, reference)

    # (action-reference)/half-range, followed by the configured gain. The
    # first coordinate is 4: residual coordinates must not be clipped.
    assert torch.equal(q_action, torch.tensor([[4.0, -4.0]]))
    q_action.sum().backward()
    assert torch.equal(action.grad, torch.tensor([[0.5, 1.0]]))
    with pytest.raises(ValueError, match="require.*reference_action"):
        policy._q_action_input(action)
    with pytest.raises(ValueError, match="shapes must match"):
        policy._q_action_input(action, reference[:, :1])


def test_explicit_legacy_early_q_fusion_preserves_exact_module_topology():
    q = DistributionalQNetwork(
        obs_dim=7,
        action_dim=3,
        hidden_dim=12,
        num_atoms=5,
        layer_norm=True,
        action_fusion="early",
    )

    assert q.action_fusion == "early"
    assert q.action_hidden_dim == 0
    assert list(q._modules) == ["net"]
    assert q.net[0].in_features == 10 and q.net[0].out_features == 12
    assert q.net[3].in_features == 12 and q.net[3].out_features == 6
    assert q.net[6].in_features == 6 and q.net[6].out_features == 3
    assert q.net[9].in_features == 3 and q.net[9].out_features == 5
    assert list(q.state_dict()) == [
        "net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
        "net.3.weight", "net.3.bias", "net.4.weight", "net.4.bias",
        "net.6.weight", "net.6.bias", "net.7.weight", "net.7.bias",
        "net.9.weight", "net.9.bias",
    ]
    assert q(torch.randn(4, 7), torch.randn(4, 3)).shape == (4, 5)


@pytest.mark.parametrize("action_fusion", ["early", "late"])
def test_reference_dueling_q_is_anchored_at_value_and_actor_gradient_is_advantage_only(
    action_fusion,
):
    q = DistributionalQNetwork(
        obs_dim=7,
        action_dim=3,
        hidden_dim=12,
        num_atoms=5,
        layer_norm=True,
        action_fusion=action_fusion,
        reference_dueling=True,
    )
    observations = torch.randn(4, 7)
    action = torch.randn(4, 3, requires_grad=True)
    reference = torch.randn(4, 3, requires_grad=True)

    anchored = q(observations, reference, reference)
    assert torch.allclose(
        anchored, q.value_logits(observations), atol=1e-6, rtol=1e-6
    )

    logits = q(observations, action, reference)
    actual_action_grad = torch.autograd.grad(
        logits.sum(), action, retain_graph=True
    )[0]
    advantage_action_grad = torch.autograd.grad(
        q.advantage_logits(observations, action).sum(), action
    )[0]
    assert torch.allclose(
        actual_action_grad, advantage_action_grad, atol=1e-7, rtol=1e-6
    )
    assert torch.count_nonzero(actual_action_grad) > 0
    assert torch.autograd.grad(
        logits.sum(), reference, allow_unused=True
    )[0] is None


def test_reference_dueling_twin_state_is_incompatible_with_direct_twin():
    dueling = TwinDistributionalQ(
        11, 3, 24, 7, -2.0, 2.0, True, "early", True
    )
    direct = TwinDistributionalQ(
        11, 3, 24, 7, -2.0, 2.0, True, "early", False
    )

    with pytest.raises(RuntimeError):
        direct.load_state_dict(dueling.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        dueling.load_state_dict(direct.state_dict(), strict=True)


class _ReferenceDuelingQCallSpy:
    def __init__(self):
        self.forward_seen = None
        self.projection_seen = None

    def __call__(self, observations, actions, reference_actions):
        self.forward_seen = (actions.clone(), reference_actions.clone())
        return torch.zeros(2, observations.shape[0], 3)

    def projection(
        self,
        observations,
        actions,
        reward,
        bootstrap,
        discount,
        reference_actions,
    ):
        self.projection_seen = (
            actions.clone(), reference_actions.clone()
        )
        return torch.zeros(2, observations.shape[0], 3)


def test_reference_dueling_q_wrappers_use_current_and_next_reference_frames():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_action_coordinates="absolute",
        q_reference_dueling=True,
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
    )
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    q_spy = _ReferenceDuelingQCallSpy()
    observations = torch.zeros(2, 1)
    current_action = torch.tensor([[1.0], [-1.0]])
    current_reference = torch.tensor([[0.5], [-0.5]])
    next_action = torch.tensor([[0.25], [0.75]])
    next_reference = torch.tensor([[-0.5], [1.0]])

    policy._q_forward(
        q_spy, observations, current_action, current_reference
    )
    policy._q_projection(
        q_spy,
        observations,
        next_action,
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
        next_reference,
    )

    assert torch.equal(
        q_spy.forward_seen[0], torch.tensor([[0.5], [-0.5]])
    )
    assert torch.equal(
        q_spy.forward_seen[1], torch.tensor([[0.25], [-0.25]])
    )
    assert torch.equal(
        q_spy.projection_seen[0], torch.tensor([[0.125], [0.375]])
    )
    assert torch.equal(
        q_spy.projection_seen[1], torch.tensor([[-0.25], [0.5]])
    )


def test_default_late_q_fusion_has_separate_768_and_128_feature_stems():
    q = DistributionalQNetwork(
        obs_dim=2341,
        action_dim=23,
        hidden_dim=768,
        num_atoms=501,
        layer_norm=True,
    )

    assert q.action_fusion == "late"
    assert q.action_hidden_dim == 128
    assert q.obs_net[0].in_features == 2341
    assert q.obs_net[0].out_features == 768
    assert isinstance(q.obs_net[1], torch.nn.LayerNorm)
    assert q.action_net[0].in_features == 23
    assert q.action_net[0].out_features == 128
    assert isinstance(q.action_net[1], torch.nn.LayerNorm)
    assert q.net[0].in_features == 896 and q.net[0].out_features == 384
    assert q.net[3].in_features == 384 and q.net[3].out_features == 192
    assert q.net[6].in_features == 192 and q.net[6].out_features == 501

    observations = torch.randn(3, 2341)
    actions = torch.randn(3, 23, requires_grad=True)
    logits = q(observations, actions)
    assert logits.shape == (3, 501)
    logits.square().mean().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert torch.count_nonzero(actions.grad) > 0


def test_late_twin_q_build_is_seeded_without_advancing_global_rng():
    global_before = torch.random.get_rng_state().clone()
    first = _build_isolated_q_network(
        11, 3, 24, 7, -2.0, 2.0, True, "cpu", 19, "late"
    )
    assert torch.equal(torch.random.get_rng_state(), global_before)
    second = _build_isolated_q_network(
        11, 3, 24, 7, -2.0, 2.0, True, "cpu", 19, "late"
    )
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert isinstance(first, TwinDistributionalQ)
    assert list(first.state_dict()) == list(second.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert not torch.equal(
        first.qnets[0].obs_net[0].weight,
        first.qnets[1].obs_net[0].weight,
    )
    observations = torch.randn(5, 11)
    actions = torch.randn(5, 3)
    expected = first(observations, actions)
    transferred = TwinDistributionalQ(
        11, 3, 24, 7, -2.0, 2.0, True, "late"
    )
    transferred.load_state_dict(first.state_dict(), strict=True)
    assert torch.equal(transferred(observations, actions), expected)
    with pytest.raises(RuntimeError):
        TwinDistributionalQ(
            11, 3, 24, 7, -2.0, 2.0, True, "early"
        ).load_state_dict(first.state_dict(), strict=True)


def test_default_late_twin_q_build_remains_isolated_and_reproducible():
    global_before = torch.random.get_rng_state().clone()
    first = _build_isolated_q_network(
        11, 3, 24, 7, -2.0, 2.0, True, "cpu", 29
    )
    second = _build_isolated_q_network(
        11, 3, 24, 7, -2.0, 2.0, True, "cpu", 29, "late"
    )

    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert any("obs_net" in name for name in first.state_dict())
    assert any("action_net" in name for name in first.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


@pytest.mark.parametrize("fusion", ["", "Late", "middle", None])
def test_q_action_fusion_rejects_unknown_values(fusion):
    with pytest.raises(ValueError, match="q_action_fusion"):
        _q_action_hidden_dim(768, fusion)


def test_q_checkpoint_metadata_makes_action_coordinates_incompatible():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
        sac_use_autotune=False,
        sac_teacher_seed_storage_ratio=0.25,
        sac_teacher_seed_sample_ratio=0.5,
        q_hidden_dim=8,
        q_num_atoms=3,
        q_v_min=-1.0,
        q_v_max=1.0,
        q_layer_norm=True,
        q_action_fusion="early",
        gamma=0.99,
    )
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_dim = 2
    policy._q_critic_dim = 3
    policy.action_dim = 1
    policy.joint_names = ["joint"]
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy._fastsac_action_contract = {
        "fingerprint": "test-contract",
        "q_action_transform_fingerprint": "test-q-transform",
    }
    policy.reward_groups = ["task"]

    normalized = policy._q_backend_metadata()
    assert normalized["q_action_coordinates"] == "absolute"
    assert normalized["q_action_normalized"] is True
    assert normalized["q_action_input_gain"] == 1.0
    assert normalized["q_action_fusion"] == "early"
    assert normalized["q_action_hidden_dim"] == 0
    assert normalized["q_reference_dueling"] is False
    assert (
        normalized["q_architecture_semantics"]
        == FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS
    )
    assert (
        normalized["q_action_semantics"]
        == FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS
    )
    assert normalized["clipped_double_q"] is True
    assert normalized["actor_q_reduction"] == "minimum"
    assert normalized["alpha_autotune"] is False

    policy.cfg.sac_q_action_input_gain = 2.0
    gained = policy._q_backend_metadata()
    assert gained["q_action_input_gain"] == 2.0
    assert gained != normalized

    policy.cfg.sac_q_normalize_actions = False
    policy.cfg.sac_clipped_double_q = False
    raw = policy._q_backend_metadata()
    assert raw["q_action_normalized"] is False
    assert raw["q_action_semantics"] == FASTSAC_Q_RAW_ACTION_SEMANTICS
    assert raw["actor_q_reduction"] == "mean"
    assert raw != normalized

    policy.cfg.q_action_coordinates = "reference_residual"
    residual = policy._q_backend_metadata()
    assert residual["q_action_coordinates"] == "reference_residual"
    assert (
        residual["q_action_semantics"]
        == FASTSAC_Q_REFERENCE_RESIDUAL_SEMANTICS
    )
    assert residual != raw

    policy.cfg.q_action_fusion = "late"
    late = policy._q_backend_metadata()
    assert late["q_action_fusion"] == "late"
    assert late["q_action_hidden_dim"] == 2
    assert late["q_action_fusion_semantics"] != normalized[
        "q_action_fusion_semantics"
    ]
    assert late != raw

    policy.cfg.q_reference_dueling = True
    dueling = policy._q_backend_metadata()
    assert dueling["q_reference_dueling"] is True
    assert (
        dueling["q_architecture_semantics"]
        == FASTSAC_Q_REFERENCE_DUELING_ARCHITECTURE_SEMANTICS
    )
    assert dueling != late


def test_disabled_grad_clipping_measures_inf_without_mutating_gradient():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(float("inf"))

    norm = _measure_or_clip_grad_norm([parameter], 0.0)

    assert torch.isinf(norm)
    assert torch.isinf(parameter.grad)


def test_teacher_uncertainty_gate_config_requires_boolean():
    _validate_fastsac_teacher_config(SimpleNamespace(
        sac_teacher_actor_uncertainty_gate=False
    ))
    with pytest.raises(
        ValueError, match="sac_teacher_actor_uncertainty_gate"
    ):
        _validate_fastsac_teacher_config(SimpleNamespace(
            sac_teacher_actor_uncertainty_gate="true"
        ))


def test_teacher_uncertainty_gate_metadata_is_behavior_anchored_when_enabled():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_teacher_actor_uncertainty_gate=False)

    assert policy._stage1_actor_uncertainty_gate_config() == {
        "enabled": False
    }

    policy.cfg.sac_teacher_actor_uncertainty_gate = True
    assert policy._stage1_actor_uncertainty_gate_config() == {
        "enabled": True,
        "anchor": "same_replay_row_recorded_behavior_action",
        "criterion": (
            "mean_twin_improvement_gt_absolute_head_disagreement"
        ),
        "q_component": "gated_full_batch_denominator",
        "entropy_component": "ungated_all_rows",
        "semantics": FASTSAC_STAGE1_BEHAVIOR_UNCERTAINTY_GATE_SEMANTICS,
    }


def test_teacher_reference_awac_config_requires_fixed_entropy_and_no_sac_gate():
    valid = SimpleNamespace(
        sac_teacher_actor_uncertainty_gate=False,
        sac_teacher_actor_objective="reference_awac",
        sac_teacher_awac_beta=0.01,
        sac_teacher_awac_weight_clip=20.0,
        sac_use_autotune=False,
    )
    _validate_fastsac_teacher_config(valid)

    with pytest.raises(ValueError, match="actor_objective"):
        _validate_fastsac_teacher_config(SimpleNamespace(
            sac_teacher_actor_uncertainty_gate=False,
            sac_teacher_actor_objective="unknown",
        ))
    with pytest.raises(ValueError, match="uncertainty_gate"):
        _validate_fastsac_teacher_config(SimpleNamespace(
            **{**vars(valid), "sac_teacher_actor_uncertainty_gate": True}
        ))
    with pytest.raises(ValueError, match="sac_use_autotune"):
        _validate_fastsac_teacher_config(SimpleNamespace(
            **{**vars(valid), "sac_use_autotune": True}
        ))


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("sac_teacher_awac_beta", 0.0, "beta"),
        ("sac_teacher_awac_beta", float("inf"), "beta"),
        ("sac_teacher_awac_weight_clip", 0.5, "weight_clip"),
        ("sac_teacher_awac_weight_clip", float("nan"), "weight_clip"),
    ],
)
def test_teacher_reference_awac_hyperparameter_validation(name, value, match):
    config = SimpleNamespace(
        sac_teacher_actor_uncertainty_gate=False,
        sac_teacher_actor_objective="reference_awac",
        sac_teacher_awac_beta=0.01,
        sac_teacher_awac_weight_clip=20.0,
        sac_use_autotune=False,
    )
    setattr(config, name, value)
    with pytest.raises(ValueError, match=match):
        _validate_fastsac_teacher_config(config)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("sac_teacher_conservative_q_coef", -0.1, "coef"),
        ("sac_teacher_conservative_q_coef", float("nan"), "coef"),
        ("sac_teacher_conservative_q_margin", -0.1, "margin"),
        ("sac_teacher_conservative_q_temperature", 0.0, "temperature"),
        ("sac_teacher_conservative_q_temperature", float("inf"), "temperature"),
        ("sac_teacher_conservative_q_starts_q_updates", -1, "starts_q_updates"),
        ("sac_teacher_conservative_q_starts_q_updates", True, "starts_q_updates"),
    ],
)
def test_teacher_conservative_q_config_validation(name, value, match):
    config = SimpleNamespace(sac_teacher_actor_uncertainty_gate=False)
    setattr(config, name, value)
    with pytest.raises(ValueError, match=match):
        _validate_fastsac_teacher_config(config)


def test_teacher_conservative_q_default_start_tracks_actor_gate():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_teacher_conservative_q_coef=1.0,
        sac_teacher_conservative_q_margin=0.002,
        sac_teacher_conservative_q_temperature=0.002,
        sac_teacher_conservative_q_starts_q_updates=None,
        sac_teacher_actor_learning_starts_q_updates=12_345,
    )
    policy.sac_update_count = 12_343

    config = policy._stage1_conservative_q_config()
    assert config["starts_q_updates"] == 12_345
    assert policy._teacher_conservative_q_is_active() is False
    policy.sac_update_count = 12_344
    assert policy._teacher_conservative_q_is_active() is True


def test_stage1_actor_objective_metadata_ignores_inactive_awac_knobs():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_objective="reference_awac",
        sac_teacher_awac_beta=0.01,
        sac_teacher_awac_weight_clip=20.0,
    )

    assert policy._stage1_actor_objective_config() == {
        "objective": "reference_awac",
        "beta": 0.01,
        "weight_clip": 20.0,
        "semantics": FASTSAC_STAGE1_REFERENCE_AWAC_SEMANTICS,
    }

    policy.cfg.sac_teacher_actor_objective = "sac"
    policy.cfg.sac_teacher_awac_beta = 99.0
    policy.cfg.sac_teacher_awac_weight_clip = 999.0
    assert policy._stage1_actor_objective_config() == {"objective": "sac"}


@pytest.mark.parametrize("n_steps", [0, -1, 1.5, True])
def test_teacher_n_step_config_requires_positive_integer(n_steps):
    with pytest.raises(ValueError, match="sac_teacher_n_steps"):
        _validate_fastsac_teacher_config(SimpleNamespace(
            sac_teacher_n_steps=n_steps,
            sac_teacher_actor_uncertainty_gate=False,
        ))


def _teacher_std_schedule_config(**overrides):
    values = {
        "sac_teacher_actor_uncertainty_gate": False,
        "fastsac_log_std_min": -5.0,
        "fastsac_log_std_max": 0.0,
        "sac_teacher_initial_log_std": -1.5,
        "sac_teacher_actor_reset_log_std": -2.5,
        "sac_teacher_actor_std_reset_q_updates": 8_000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_teacher_std_schedule_validation_requires_interior_paired_targets():
    _validate_fastsac_teacher_config(_teacher_std_schedule_config())

    with pytest.raises(ValueError, match="strictly inside"):
        _validate_fastsac_teacher_config(_teacher_std_schedule_config(
            sac_teacher_initial_log_std=-5.0,
        ))
    with pytest.raises(ValueError, match="configured together"):
        _validate_fastsac_teacher_config(_teacher_std_schedule_config(
            sac_teacher_actor_std_reset_q_updates=None,
        ))
    with pytest.raises(ValueError, match="configured together"):
        _validate_fastsac_teacher_config(_teacher_std_schedule_config(
            sac_teacher_actor_reset_log_std=None,
        ))
    with pytest.raises(ValueError, match="positive integer"):
        _validate_fastsac_teacher_config(_teacher_std_schedule_config(
            sac_teacher_actor_std_reset_q_updates=0,
        ))


def _teacher_std_schedule_policy(phase="train"):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = _teacher_std_schedule_config()
    policy.cfg.phase = phase
    core = FastSACActor(
        input_dim=4,
        action_dim=2,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=-torch.ones(2),
        action_high=torch.ones(2),
        layer_norm=True,
    )
    policy.actor = torch.nn.Sequential(core)
    policy.sac_teacher_actor_optimizer = torch.optim.AdamW(
        core.parameters(), lr=1e-3
    )
    policy.sac_update_count = 0
    policy._teacher_actor_std_reset_applied = False
    policy._teacher_actor_std_reset_applied_q_updates = None
    policy._teacher_actor_std_reset_event_q_updates = None
    return policy, core


def test_teacher_std_reset_is_once_before_exact_q_update_and_clears_head_state():
    policy, core = _teacher_std_schedule_policy()
    policy.cfg.sac_teacher_actor_std_reset_q_updates = 3
    core.reset_log_std_head(-1.5)
    with torch.no_grad():
        core.fc_logstd.weight.add_(0.25)
    core.fc_logstd.weight.grad = torch.ones_like(core.fc_logstd.weight)
    core.fc_logstd.bias.grad = torch.ones_like(core.fc_logstd.bias)
    optimizer = policy.sac_teacher_actor_optimizer
    optimizer.state[core.fc_logstd.weight]["marker"] = torch.tensor(1.0)
    optimizer.state[core.fc_logstd.bias]["marker"] = torch.tensor(1.0)
    preserved_mu = {
        name: parameter.detach().clone()
        for name, parameter in core.named_parameters()
        if not name.startswith("fc_logstd.")
    }

    policy.sac_update_count = 1
    assert policy._maybe_reset_teacher_actor_std_before_q_update() is False
    policy.sac_update_count = 2
    assert policy._maybe_reset_teacher_actor_std_before_q_update() is True

    assert torch.count_nonzero(core.fc_logstd.weight) == 0
    assert torch.count_nonzero(core.fc_logstd.bias) == 0
    assert core.fc_logstd.weight.grad is None
    assert core.fc_logstd.bias.grad is None
    assert core.fc_logstd.weight not in optimizer.state
    assert core.fc_logstd.bias not in optimizer.state
    assert policy._teacher_actor_std_reset_applied is True
    assert policy._teacher_actor_std_reset_applied_q_updates == 3
    assert policy._teacher_actor_std_reset_event_q_updates == 3
    for name, parameter in core.named_parameters():
        if name in preserved_mu:
            assert torch.equal(parameter, preserved_mu[name])
    assert policy._maybe_reset_teacher_actor_std_before_q_update() is False


def _schedule_seed_replay(count):
    replay = TeacherTrainingReplayBuffer(
        capacity=4,
        critic_dim=1,
        action_dim=1,
        seed_storage_ratio=0.5,
        seed_sample_ratio=0.5,
    )
    ids = torch.arange(float(count))
    replay.append({
        "critic_observations": ids[:, None],
        "actions": ids[:, None],
        "rewards": ids,
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.ones(count),
        "next_critic_observations": (ids + 1)[:, None],
    })
    return replay


def test_teacher_std_reset_freezes_and_retains_broad_seed_first():
    policy, _ = _teacher_std_schedule_policy()
    policy.cfg.sac_teacher_actor_std_reset_q_updates = 1
    policy.cfg.sac_teacher_seed_storage_ratio = 0.5
    policy.teacher_replay = _schedule_seed_replay(4)
    frozen = policy.teacher_replay.data["critic_observations"][:2].clone()
    events = []
    original_set_log_std = policy._set_teacher_actor_log_std

    def set_log_std(target, *, clear_optimizer_state):
        assert policy.teacher_replay.seed_frozen
        events.append("reset")
        return original_set_log_std(
            target, clear_optimizer_state=clear_optimizer_state
        )

    policy._set_teacher_actor_log_std = set_log_std

    assert policy._maybe_reset_teacher_actor_std_before_q_update() is True

    assert events == ["reset"]
    assert policy.teacher_replay.seed_frozen
    policy.teacher_replay.append({
        "critic_observations": torch.tensor([[10.0], [11.0]]),
        "actions": torch.tensor([[10.0], [11.0]]),
        "rewards": torch.tensor([10.0, 11.0]),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        "next_critic_observations": torch.tensor([[11.0], [12.0]]),
    })
    assert torch.equal(
        policy.teacher_replay.data["critic_observations"][:2], frozen
    )


def test_teacher_std_reset_rejects_incomplete_seed_replay():
    policy, core = _teacher_std_schedule_policy()
    policy.cfg.sac_teacher_actor_std_reset_q_updates = 1
    policy.cfg.sac_teacher_seed_storage_ratio = 0.5
    policy.teacher_replay = _schedule_seed_replay(3)
    before = core.fc_logstd.bias.detach().clone()

    with pytest.raises(RuntimeError, match="before the replay is full"):
        policy._maybe_reset_teacher_actor_std_before_q_update()

    assert not policy.teacher_replay.seed_frozen
    assert policy._teacher_actor_std_reset_applied is False
    assert torch.equal(core.fc_logstd.bias, before)


def test_teacher_std_schedule_checkpoint_restores_one_shot_state():
    source, _ = _teacher_std_schedule_policy()
    source._teacher_actor_std_reset_applied = True
    source._teacher_actor_std_reset_applied_q_updates = 8_000
    schedule_state = source._teacher_actor_std_schedule_checkpoint_state()

    resumed, _ = _teacher_std_schedule_policy()
    resumed.sac_update_count = 8_500
    validated = resumed._validate_teacher_actor_std_schedule_checkpoint({
        "stage1_actor_std_schedule": schedule_state,
    })
    resumed._restore_teacher_actor_std_schedule_checkpoint(validated)

    assert resumed._teacher_actor_std_reset_applied is True
    assert resumed._teacher_actor_std_reset_applied_q_updates == 8_000
    assert resumed._teacher_actor_std_reset_event_q_updates is None
    assert resumed._maybe_reset_teacher_actor_std_before_q_update() is False

    missing, _ = _teacher_std_schedule_policy()
    with pytest.raises(ValueError, match="predates"):
        missing._validate_teacher_actor_std_schedule_checkpoint({})


def test_teacher_std_schedule_never_mutates_stage2_actor():
    policy, core = _teacher_std_schedule_policy(phase="finetune")
    core.reset_log_std_head(-1.5)
    before = {
        name: parameter.detach().clone()
        for name, parameter in core.named_parameters()
    }
    policy.sac_update_count = 7_999

    assert policy._maybe_reset_teacher_actor_std_before_q_update() is False
    for name, parameter in core.named_parameters():
        assert torch.equal(parameter, before[name])


@pytest.mark.parametrize("accepted_rows", [0, 1, 256, 1024, 8192])
def test_teacher_update_cadence_matches_fixed_wbt_updates(accepted_rows):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_batch_size=8192,
        sac_teacher_updates_per_env_step=4,
        sac_teacher_update_interval_env_steps=1,
        sac_teacher_learning_starts_transitions=0,
        sac_teacher_actor_learning_starts_q_updates=2,
        sac_teacher_policy_frequency=2,
    )
    policy.teacher_replay = _Replay()
    policy.teacher_replay.saved = 1
    policy.teacher_replay.seen = 1
    policy.sac_environment_steps = 1

    assert policy._teacher_updates_due(accepted_rows) == 4
    policy.sac_update_count = 1
    assert policy._teacher_actor_update_is_due() is False
    policy.sac_update_count = 2
    assert policy._teacher_actor_update_is_due() is True
    policy.sac_update_count = 4
    policy.cfg.sac_teacher_actor_learning_starts_q_updates = 5
    assert policy._teacher_actor_update_is_due() is False


def test_teacher_interval_eight_waits_for_warmup_without_backlog():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_teacher_updates_per_env_step=4,
        sac_teacher_update_interval_env_steps=8,
        sac_teacher_learning_starts_transitions=3,
    )
    policy.teacher_replay = _Replay()
    policy.teacher_replay.saved = 2
    policy.teacher_replay.seen = 2

    # Step 8 was a scheduled boundary, but warmup suppresses its burst.
    policy.sac_environment_steps = 8
    assert policy._teacher_updates_due(10_000) == 0
    # Reaching warmup at step 9 does not create catch-up credit. The next burst
    # is due only on the next interval boundary at step 16.
    policy.teacher_replay.seen = 3
    for environment_step in range(9, 16):
        policy.sac_environment_steps = environment_step
        assert policy._teacher_updates_due(0) == 0
    policy.sac_environment_steps = 16
    assert policy._teacher_updates_due(0) == 4
    policy.sac_environment_steps = 17
    assert policy._teacher_updates_due(0) == 0


def test_teacher_actor_batch_zero_reuses_prepared_q_batch_without_resampling():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_teacher_actor_batch_size=0)
    policy.teacher_replay = _Replay()
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(0)
    policy._teacher_learning_fields = ("batch",)
    q_batch = {"batch": torch.ones(3), "prepared": True}
    policy._prepare_teacher_learning_batch = lambda batch: pytest.fail(
        "the already prepared Q minibatch must not be prepared again"
    )

    actor_batch = policy._teacher_actor_learning_batch(q_batch)

    assert actor_batch is q_batch
    assert policy.teacher_replay.events == []
    assert policy.teacher_replay.sample_counts == []


def test_teacher_actor_batch_rejects_negative_size():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_teacher_actor_batch_size=-1)

    with pytest.raises(ValueError, match="must be non-negative"):
        policy._teacher_actor_learning_batch({})


def test_positive_teacher_actor_batch_samples_fresh_only_when_actor_is_due():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        train_every=1,
        sac_batch_size=3,
        sac_teacher_actor_batch_size=7,
        sac_teacher_learning_starts_transitions=0,
        sac_teacher_updates_per_env_step=1,
    )
    policy.teacher_replay = _Replay()
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(0)
    policy._teacher_learning_fields = ("batch",)
    policy.sac_environment_steps = 0
    policy._teacher_export_started = True
    policy._interleaved_steps_collected = 0
    policy._interleaved_q_metrics = []
    policy._interleaved_actor_metrics = []
    policy._interleaved_replay_accepted = 0
    policy._interleaved_reward_sum = torch.zeros(())
    policy._interleaved_transition_count = 0
    policy._teacher_transition_from_step = lambda current, carry: {
        "rewards": torch.ones(1)
    }
    prepared_raw_batches = []
    q_batches = []
    actor_batches = []

    def prepare(raw_batch):
        prepared_raw_batches.append(raw_batch)
        return {
            "batch": raw_batch["batch"].clone(),
            "prepare_index": len(prepared_raw_batches),
        }

    policy._prepare_teacher_learning_batch = prepare
    policy._teacher_q_alpha_update = (
        lambda batch: q_batches.append(batch) or _metric()
    )
    policy._teacher_actor_update_is_due = lambda: True
    policy._teacher_actor_update = (
        lambda batch: actor_batches.append(batch) or {}
    )
    policy._soft_update_teacher_q_target = lambda: None

    empty = TensorDict({}, batch_size=[1])
    policy.collect_environment_step(empty, empty)

    assert policy.teacher_replay.events == ["append", "sample", "sample"]
    assert policy.teacher_replay.sample_counts == [3, 7]
    assert len(prepared_raw_batches) == 2
    assert q_batches[0]["prepare_index"] == 1
    assert q_batches[0]["batch"].shape == (3,)
    assert actor_batches[0]["prepare_index"] == 2
    assert actor_batches[0]["batch"].shape == (7,)
    assert actor_batches[0] is not q_batches[0]


def test_seed_partition_freezes_before_sampling_first_actor_update_batch():
    replay = TeacherTrainingReplayBuffer(
        capacity=4,
        critic_dim=1,
        action_dim=1,
        seed_storage_ratio=0.5,
        seed_sample_ratio=0.5,
    )
    ids = torch.arange(4.0)
    replay.append({
        "critic_observations": ids[:, None],
        "actions": ids[:, None],
        "rewards": ids,
        "dones": torch.zeros(4, dtype=torch.bool),
        "truncations": torch.zeros(4, dtype=torch.bool),
        "discounts": torch.ones(4),
        "next_critic_observations": (ids + 1)[:, None],
    })
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_teacher_seed_storage_ratio=0.5,
        sac_teacher_actor_learning_starts_q_updates=2,
        sac_teacher_policy_frequency=2,
    )
    policy.teacher_replay = replay
    policy.sac_update_count = 1

    assert policy._maybe_freeze_teacher_seed_replay_before_q_update()
    assert replay.seed_frozen
    batch = replay.sample(20, generator=torch.Generator().manual_seed(4))
    assert int((batch["critic_observations"][:, 0] < 2).sum()) == 10

    policy.sac_update_count = 2
    assert policy._teacher_actor_update_is_due()


def test_interleaved_resume_waits_for_first_accepted_replay_row():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        train_every=2,
        sac_learning_starts=0,
        sac_policy_frequency=100,
        sac_batch_size=2,
        sac_teacher_learning_starts_transitions=0,
        sac_teacher_updates_per_env_step=1,
    )
    policy.device = torch.device("cpu")
    policy.teacher_replay = _Replay()
    policy._teacher_learning_fields = ()
    policy.q_rng = torch.Generator().manual_seed(0)
    policy.sac_environment_steps = 100
    policy._teacher_export_started = True
    policy._interleaved_steps_collected = 0
    policy._interleaved_q_metrics = []
    policy._interleaved_actor_metrics = []
    policy._interleaved_replay_accepted = 0
    policy._interleaved_reward_sum = torch.zeros(())
    policy._interleaved_transition_count = 0
    policy._last_truncation_finals_used = 0
    events = []
    batches = iter((
        {"rewards": torch.empty(0)},
        {"rewards": torch.ones(1)},
    ))
    policy._teacher_transition_from_step = lambda current, carry: next(batches)
    policy._teacher_q_alpha_update = lambda batch: events.append("q") or _metric()
    policy._prepare_teacher_learning_batch = lambda batch: batch
    policy._teacher_actor_update_is_due = lambda: False
    policy._teacher_actor_update = lambda batch: pytest.fail(
        "actor update is not due in this test"
    )
    policy._soft_update_teacher_q_target = lambda: events.append("target")
    empty = TensorDict({}, batch_size=[1])

    policy.collect_environment_step(empty, empty)
    assert policy.sac_environment_steps == 101
    assert policy.teacher_replay.size == 0
    assert "sample" not in policy.teacher_replay.events
    assert events == []

    policy.collect_environment_step(empty, empty)
    assert policy.sac_environment_steps == 102
    assert policy.teacher_replay.size == 1
    assert policy.teacher_replay.events[-1] == "sample"
    assert events == ["q", "target"]


def test_interleaved_interval_eight_collects_every_step_and_runs_one_burst():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        train_every=8,
        sac_batch_size=2,
        sac_teacher_actor_batch_size=7,
        sac_teacher_learning_starts_transitions=0,
        sac_teacher_updates_per_env_step=4,
        sac_teacher_update_interval_env_steps=8,
    )
    policy.device = torch.device("cpu")
    policy.teacher_replay = _Replay()
    policy._teacher_learning_fields = ()
    policy.q_rng = torch.Generator().manual_seed(0)
    policy.sac_environment_steps = 0
    policy.sac_update_count = 0
    policy._teacher_export_started = True
    policy._interleaved_steps_collected = 0
    policy._interleaved_q_metrics = []
    policy._interleaved_actor_metrics = []
    policy._interleaved_replay_accepted = 0
    policy._interleaved_reward_sum = torch.zeros(())
    policy._interleaved_transition_count = 0
    policy._last_truncation_finals_used = 0
    policy._teacher_transition_from_step = lambda current, carry: {
        "rewards": torch.ones(1)
    }
    events = []

    def q_update(batch):
        policy.sac_update_count += 1
        events.append("q")
        return _metric()

    policy._teacher_q_alpha_update = q_update
    policy._prepare_teacher_learning_batch = lambda batch: batch
    policy._teacher_actor_update_is_due = lambda: False
    policy._soft_update_teacher_q_target = lambda: events.append("target")
    empty = TensorDict({}, batch_size=[1])

    for expected_step in range(1, 8):
        policy.collect_environment_step(empty, empty)
        assert policy.sac_environment_steps == expected_step
        assert events == []
    policy.collect_environment_step(empty, empty)

    assert policy.sac_environment_steps == 8
    assert policy.teacher_replay.seen == 8
    assert events == ["q", "target"] * 4
    assert policy.teacher_replay.events.count("sample") == 4
    assert policy.teacher_replay.sample_counts == [2] * 4


def test_stage1_exposes_no_optional_ppo_training_surface():
    config = FastSACVelConfig()
    removed_fields = (
        "sac_teacher_ppo_warmup_rollouts",
        "sac_teacher_ppo_behavior_distill_rollouts",
        "sac_teacher_ppo_behavior_distill_epochs",
        "sac_teacher_ppo_behavior_distill_lr",
        "sac_teacher_distilled_ppo_warmup_rollouts",
    )
    assert all(not hasattr(config, name) for name in removed_fields)
    removed_method_tokens = (
        "ppo_warmup",
        "ppo_behavior",
        "distilled_ppo",
        "_train_ppo",
        "_update_ppo",
    )
    assert not any(
        token in name.lower()
        for name in FastSACVEL.__dict__
        for token in removed_method_tokens
    )

    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    parameter = torch.nn.Parameter(torch.zeros(()))
    policy.register_parameter("optimizer_test_parameter", parameter)
    policy.opt_q = torch.optim.SGD([parameter], lr=0.0)
    policy.opt_policy = torch.optim.SGD([parameter], lr=0.0)
    policy.opt_critic = torch.optim.SGD([parameter], lr=0.0)
    policy.ppo_behavior_actor_optimizer = torch.optim.SGD(
        [parameter], lr=0.0
    )

    assert policy.requires_value_bootstrap() is False
    assert tuple(policy._optimizer_registry()) == ("opt_q",)


@pytest.mark.parametrize("train_student_models", [True, False])
def test_true_teacher_train_op_never_calls_ppo_or_enables_h5_export(
    train_student_models,
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        train_every=2,
        teacher_buffer_start_iteration=5100,
        sac_learning_starts=0,
        sac_policy_frequency=2,
        sac_batch_size=3,
        sac_teacher_learning_starts_transitions=0,
        sac_teacher_updates_per_env_step=1,
        sac_teacher_actor_learning_starts_q_updates=0,
        sac_teacher_policy_frequency=2,
        train_student_models=train_student_models,
    )
    policy.env = SimpleNamespace(current_iter=5099)
    policy.device = torch.device("cpu")
    policy.action_dim = 1
    policy.joint_names = ["joint"]
    policy.teacher_replay = _Replay()
    policy._teacher_learning_fields = ()
    policy.q_rng = torch.Generator().manual_seed(0)
    policy.log_alpha = torch.nn.Parameter(torch.tensor(-1.0))
    policy.target_entropy = 0.0
    policy._fastsac_action_log_scale_sum = 0.0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.sac_update_count = 0
    policy.sac_environment_steps = 0
    policy.sac_rollout_count = 0
    policy.num_updates = 0
    policy._teacher_export_started = False
    policy._teacher_export_start_seen = None
    policy._last_truncation_finals_used = 0
    events = []
    policy._teacher_transition_chunks = lambda td: iter((
        {"rewards": torch.zeros(1)},
        {"rewards": torch.zeros(1)},
    ))
    def q_update(batch):
        policy.sac_update_count += 1
        events.append("q")
        return _metric()

    policy._teacher_q_alpha_update = q_update
    policy._prepare_teacher_learning_batch = lambda batch: batch

    def actor_update(batch):
        assert batch is policy.teacher_replay.last_sample
        events.append("actor")
        return {
            "actor_loss": torch.tensor(1.0),
            "actor_sac_loss": torch.tensor(1.0),
            "actor_grad_norm": torch.tensor(1.0),
            "actor_uncertainty_gate_acceptance_fraction": torch.tensor(0.0),
            "actor_uncertainty_gate_accepted_confidence_margin": torch.tensor(0.0),
            "actor_uncertainty_gate_mean_confidence_margin": torch.tensor(0.0),
            "actor_uncertainty_gate_policy_replay_improvement": torch.tensor(0.0),
            "actor_uncertainty_gate_policy_replay_disagreement": torch.tensor(0.0),
            "entropy": torch.tensor(1.0),
            "action_std": torch.tensor(1.0),
            "reference_mean_action_error": torch.tensor(0.0),
            "policy_q_mean": torch.tensor(0.0),
            "deterministic_policy_q_mean": torch.tensor(0.0),
            "reference_q_mean": torch.tensor(0.0),
            "replay_q_mean": torch.tensor(0.0),
            "policy_replay_q_gap": torch.tensor(0.0),
            "deterministic_policy_reference_advantage": torch.tensor(0.0),
            "deterministic_reference_q1_advantage": torch.tensor(0.0),
            "deterministic_reference_q2_advantage": torch.tensor(0.0),
            "deterministic_reference_pessimistic_advantage": torch.tensor(0.0),
            "deterministic_reference_advantage_disagreement": torch.tensor(0.0),
            "twin_q_disagreement": torch.tensor(0.0),
            "deterministic_policy_twin_q_disagreement": torch.tensor(0.0),
            "reference_twin_q_disagreement": torch.tensor(0.0),
            "replay_twin_q_disagreement": torch.tensor(0.0),
            "normalized_action_mean_deviation": torch.tensor(0.0),
            "action_saturation_fraction": torch.tensor(0.0),
            "awac_active": torch.tensor(0.0),
            "awac_loss": torch.tensor(0.0),
            "awac_replay_log_prob": torch.tensor(0.0),
            "awac_advantage_mean": torch.tensor(0.0),
            "awac_advantage_std": torch.tensor(0.0),
            "awac_advantage_min": torch.tensor(0.0),
            "awac_advantage_max": torch.tensor(0.0),
            "awac_positive_advantage_fraction": torch.tensor(0.0),
            "awac_weight_mean": torch.tensor(0.0),
            "awac_weight_max": torch.tensor(0.0),
            "awac_weight_ess_fraction": torch.tensor(0.0),
            "awac_weighted_replay_reference_deviation": torch.tensor(0.0),
        }

    policy._teacher_actor_update = actor_update
    policy._soft_update_teacher_q_target = lambda: events.append("target")
    if train_student_models:
        policy._train_adapt_no_depth = lambda td: events.append("adapt") or {}
    else:
        policy._train_adapt_no_depth = lambda td: pytest.fail(
            "student adaptation/distillation must remain disabled"
        )
    policy.train_policy = lambda td: pytest.fail("PPO train_policy must not run")
    policy._update_ppo = lambda td: pytest.fail("PPO update must not run")
    policy.critic = lambda td: pytest.fail("PPO critic must not run")
    policy.gae = lambda *args, **kwargs: pytest.fail("GAE must not run")
    assert "train_op" not in _FastSACVAICBase.__dict__
    assert "_train_ppo_path" not in _FastSACVAICBase.__dict__
    rollout = TensorDict(
        {"x": torch.zeros(1, 2, 1), "scale": torch.ones(1, 2, 1)},
        batch_size=[1, 2],
    )

    before_gate = policy.train_op(rollout)
    assert policy.sac_environment_steps == 2
    assert policy.teacher_replay.events == ["append", "sample", "append", "sample"]
    assert policy.teacher_replay.sample_counts == [3, 3]
    expected_events = ["q", "target", "q", "actor", "target"]
    if train_student_models:
        expected_events.append("adapt")
    assert events == expected_events
    assert before_gate["fastsac/h5_export_active"] == 0.0
    assert before_gate["fastsac/student_training_active"] == float(
        train_student_models
    )
    assert before_gate["fastsac/effective_actor_batch_size"] == 3
    assert not any(
        key.startswith("fastsac/ppo_")
        or "ppo_warmup" in key
        or "ppo_behavior" in key
        or "distilled_ppo" in key
        for key in before_gate
    )

    policy.env.current_iter = 5100
    events.clear()
    policy.teacher_replay.events.clear()
    at_gate = policy.train_op(rollout)
    assert policy.sac_environment_steps == 4
    assert policy.teacher_replay.events[:2] == ["append", "sample"]
    assert "clear" not in policy.teacher_replay.events
    assert at_gate["fastsac/h5_export_active"] == 0.0
    assert at_gate["fastsac/h5_export_rows"] == 0
    assert not any(
        key.startswith("fastsac/ppo_")
        or "ppo_warmup" in key
        or "ppo_behavior" in key
        or "distilled_ppo" in key
        for key in at_gate
    )
    assert policy.snapshot_teacher_replay(5100, "checkpoint_5100") is None

    policy.env.current_iter = 5101
    policy.teacher_replay.events.clear()
    policy.train_op(rollout)
    assert policy.sac_environment_steps == 6
    assert "clear" not in policy.teacher_replay.events


class _FixedDist:
    def rsample_with_log_prob(self, generator=None):
        return torch.tensor([[0.25], [0.25]]), torch.tensor([-2.0, -2.0])


class _TinyQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 3))
        self.seen = None

    def forward(self, obs, action):
        self.seen = action.detach().clone()
        return self.logits[:, None].expand(-1, obs.shape[0], -1)


class _ActionSensitiveC51Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))
        self.seen_actions = []

    def forward(self, obs, action):
        self.seen_actions.append(action.detach().clone())
        value = action[:, 0] * self.slope
        q1 = torch.stack((-value, torch.zeros_like(value), value), dim=-1)
        q2 = torch.stack((-2.0 * value, torch.zeros_like(value), 2.0 * value), dim=-1)
        return torch.stack((q1, q2), dim=0)

    def values(self, logits):
        return (torch.softmax(logits, dim=-1) * self.support).sum(dim=-1)


class _ProjectionSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))
        self.seen = None

    def projection(self, obs, action, reward, bootstrap, discount):
        self.seen = (action.clone(), reward.clone(), bootstrap.clone(), discount.clone())
        target = torch.zeros(2, obs.shape[0], 3)
        target[..., 1] = 1.0
        return target


class _ActorLossDist:
    def __init__(self, parameter, batch_size):
        self.parameter = parameter
        self.batch_size = batch_size
        self.scale = torch.ones(batch_size, 1)
        self.low = torch.full((1,), -2.0)
        self.high = torch.full((1,), 2.0)

    @property
    def mean(self):
        return self.parameter.expand(self.batch_size, 1)

    def rsample_with_log_prob(self, generator=None):
        action = self.parameter.expand(self.batch_size, 1)
        log_prob = self.parameter.expand(self.batch_size) * 0.0 - 2.0
        return action, log_prob


class _ReferenceAWACActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = torch.nn.Parameter(torch.tensor(0.0))
        self.log_scale = torch.nn.Parameter(torch.log(torch.tensor(0.2)))

    def get_dist(self, td):
        batch_size = td.batch_size[0]
        return FastSACTanhNormal(
            self.loc.expand(batch_size, 1),
            self.log_scale.exp().expand(batch_size, 1),
            low=torch.tensor([-2.0]),
            high=torch.tensor([2.0]),
            event_dims=1,
        )


class _ReferenceAWACTargetQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, observations, actions):
        self.seen.append((
            observations.detach().clone(), actions.detach().clone()
        ))
        first = actions[:, 0] + self.dummy * 0.0
        second = 2.0 * actions[:, 0] + self.dummy * 0.0
        return torch.stack((first, second), dim=0).unsqueeze(-1)

    @staticmethod
    def values(logits):
        return logits.squeeze(-1)


class _ZeroActorQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.seen_actions = []

    def forward(self, observations, actions):
        self.seen_actions.append(actions.detach().clone())
        zeros = actions[..., :1] * 0.0 + self.dummy * 0.0
        return zeros.unsqueeze(0).expand(2, -1, -1)

    @staticmethod
    def values(logits):
        return logits.squeeze(-1)


class _AsymmetricActorQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, observations, actions):
        first = actions[:, 0] + self.dummy * 0.0
        second = 3.0 * actions[:, 0] + self.dummy * 0.0
        return torch.stack((first, second), dim=0).unsqueeze(-1)

    @staticmethod
    def values(logits):
        return logits.squeeze(-1)


class _EntropyGradientDist(_ActorLossDist):
    def rsample_with_log_prob(self, generator=None):
        action = self.parameter.expand(self.batch_size, 1)
        # Deliberately retain a direct entropy gradient so the test can prove
        # that rejecting every Q row does not freeze the SAC entropy term.
        log_prob = self.parameter.expand(self.batch_size)
        return action, log_prob


class _RelativeUncertaintyActorQ(torch.nn.Module):
    """Twin linear Q whose per-row action slopes come from critic observations."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.seen_actions = []

    def forward(self, observations, actions):
        self.seen_actions.append(actions.detach().clone())
        first = observations[:, 0] * actions[:, 0] + self.dummy * 0.0
        second = observations[:, 1] * actions[:, 0] + self.dummy * 0.0
        return torch.stack((first, second), dim=0).unsqueeze(-1)

    @staticmethod
    def values(logits):
        return logits.squeeze(-1)


@pytest.mark.parametrize(
    "actor_start, autotune, expected_alpha_updates",
    [(0, True, 1), (2, True, 0), (0, False, 0)],
)
def test_teacher_q_target_uses_stochastic_action_and_truncation_bootstrap(
    actor_start, autotune, expected_alpha_updates
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.97,
        sac_teacher_q_max_grad_norm=0.0,
        sac_teacher_actor_learning_starts_q_updates=actor_start,
        sac_q_normalize_actions=True,
        sac_clipped_double_q=True,
        sac_use_autotune=autotune,
        sac_teacher_n_steps=4,
    )
    policy._fastsac_q_action_center = torch.tensor([2.0])
    policy._fastsac_q_action_scale = torch.tensor([4.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.qnet = _TinyQ()
    policy.qnet_target = _ProjectionSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.01)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_update_count = 0
    policy.sac_alpha_update_count = 0
    state_calls = []

    def teacher_state(batch, next_state=False):
        state_calls.append(next_state)
        return TensorDict({}, batch_size=[2])

    policy._teacher_state_from_replay = teacher_state
    policy.actor = SimpleNamespace(get_dist=lambda td: _FixedDist())
    batch = {
        "next_critic_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[-2.0], [6.0]]),
        "rewards": torch.ones(2),
        "dones": torch.tensor([True, True]),
        "truncations": torch.tensor([True, False]),
        "discounts": torch.tensor([0.5, 1.0]),
        "effective_n_steps": torch.tensor([4.0, 2.0]),
    }

    initial_log_alpha = policy.log_alpha.detach().clone()
    metrics = policy._teacher_q_alpha_update(batch)
    action, soft_reward, bootstrap, discount = policy.qnet_target.seen
    assert torch.equal(action, torch.full((2, 1), -0.4375))
    assert torch.equal(policy.qnet.seen, torch.tensor([[-1.0], [1.0]]))
    assert torch.equal(bootstrap, torch.tensor([1.0, 0.0]))
    expected_discount = torch.tensor([0.5 * 0.97 ** 4, 0.97 ** 2])
    assert torch.allclose(discount, expected_discount)
    assert torch.allclose(
        soft_reward, torch.tensor([1.0 + expected_discount[0], 1.0])
    )
    assert policy.q_update_count == 1
    assert policy.sac_update_count == 1
    assert policy.sac_alpha_update_count == expected_alpha_updates
    assert state_calls == [True]
    assert metrics["conservative_q_active"].item() == 0.0
    assert metrics["conservative_q_loss"].item() == 0.0
    assert torch.equal(metrics["q_loss"], metrics["bellman_q_loss"])
    assert torch.equal(policy.log_alpha.detach(), initial_log_alpha) is (
        expected_alpha_updates == 0
    )


def test_teacher_conservative_q_uses_detached_policy_and_current_reference():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.97,
        q_action_coordinates="reference_residual",
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
        sac_use_autotune=False,
        sac_teacher_n_steps=1,
        sac_teacher_q_max_grad_norm=0.0,
        sac_teacher_actor_learning_starts_q_updates=0,
        sac_teacher_conservative_q_coef=1.0,
        sac_teacher_conservative_q_margin=0.002,
        sac_teacher_conservative_q_temperature=0.002,
        sac_teacher_conservative_q_starts_q_updates=None,
    )
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.qnet = _ActionSensitiveC51Q()
    policy.qnet_target = _ProjectionSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.1)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.01)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_update_count = 0
    policy.sac_alpha_update_count = 0

    actor_action = torch.nn.Parameter(torch.tensor(0.75))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(actor_action, td.batch_size[0])
    )
    state_calls = []

    def teacher_state(batch, next_state=False):
        state_calls.append(next_state)
        reference_key = (
            NEXT_TEACHER_REF_ACTION_FIELD
            if next_state
            else TEACHER_REF_ACTION_FIELD
        )
        return TensorDict(
            {REF_JPOS_KEY: batch[reference_key]}, batch_size=[2]
        )

    policy._teacher_state_from_replay = teacher_state
    batch = {
        "next_critic_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[0.25], [0.0]]),
        TEACHER_REF_ACTION_FIELD: torch.tensor([[0.25], [0.5]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[0.5], [1.0]]),
        "rewards": torch.ones(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        "effective_n_steps": torch.ones(2),
    }

    initial_actor_action = actor_action.detach().clone()
    metrics = policy._teacher_q_alpha_update(batch)

    # Target, replay, and policy actions each use the reference from their own
    # frame before entering Q's residual coordinate system.
    assert torch.allclose(
        policy.qnet_target.seen[0], torch.tensor([[0.125], [-0.125]])
    )
    assert torch.allclose(
        policy.qnet.seen_actions[0], torch.tensor([[0.0], [-0.25]])
    )
    assert torch.allclose(
        policy.qnet.seen_actions[1], torch.tensor([[0.25], [0.125]])
    )
    assert state_calls == [True, False]
    assert actor_action.grad is None
    assert torch.equal(actor_action.detach(), initial_actor_action)
    assert metrics["conservative_q_active"].item() == 1.0
    assert metrics["conservative_q_penalty"].item() > 0.0
    assert metrics["conservative_q_loss"].item() > 0.0
    assert metrics["conservative_policy_replay_q_gap"].item() > 0.0
    assert metrics["conservative_positive_gap_fraction"].item() == 1.0
    assert metrics["q_loss"].item() > metrics["bellman_q_loss"].item()


def test_student_fixed_alpha_skips_autotune_update_but_keeps_entropy_target():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.97,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        sac_q_normalize_actions=False,
        sac_clipped_double_q=True,
        sac_use_autotune=False,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.qnet = _TinyQ()
    policy.qnet_target = _ProjectionSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.25)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.01)
    policy.target_entropy = -1.0
    policy.q_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    policy._actor_dist_from_flat = lambda observations: _FixedDist()
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.ones(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    initial_log_alpha = policy.log_alpha.detach().clone()
    result = policy._sac_update(batch, update_actor=False)

    assert result[4].item() == 0.0
    assert policy.sac_alpha_update_count == 0
    assert torch.equal(policy.log_alpha.detach(), initial_log_alpha)


def test_teacher_actor_loss_is_pure_sac_without_reference_kl():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(sac_teacher_actor_max_grad_norm=0.0)
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(
            actor_parameter, td.batch_size[0]
        )
    )
    policy.qnet = _ZeroActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict(
            {REF_JPOS_KEY: torch.zeros(2, 1)},
            batch_size=[2],
        )
    )
    batch = {
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
    }

    metrics = policy._teacher_actor_update(batch)

    assert not any("kl" in key for key in metrics)
    # alpha * mean(log_prob) - Q = 0.5 * -2 - 0
    assert metrics["actor_loss"].item() == pytest.approx(-1.0)
    assert metrics["actor_uncertainty_gate_acceptance_fraction"].item() == 0.0
    assert (
        metrics["actor_uncertainty_gate_accepted_confidence_margin"].item()
        == 0.0
    )
    assert metrics["actor_uncertainty_gate_mean_confidence_margin"].item() == 0.0
    assert metrics[
        "actor_uncertainty_gate_policy_replay_improvement"
    ].item() == 0.0
    assert metrics[
        "actor_uncertainty_gate_policy_replay_disagreement"
    ].item() == 0.0
    assert policy.sac_actor_update_count == 1


def test_teacher_reference_awac_uses_target_replay_reference_q_and_freezes_scale():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.action_dim = 1
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_objective="reference_awac",
        sac_teacher_awac_beta=1.0,
        sac_teacher_awac_weight_clip=20.0,
        sac_teacher_actor_max_grad_norm=0.0,
        q_condition_on_actuator_state=True,
        q_action_coordinates="absolute",
        sac_q_normalize_actions=False,
        sac_q_action_input_gain=1.0,
        q_reference_dueling=False,
    )
    policy._q_actuator_context_dim = 1
    policy.sac_action_rng = torch.Generator().manual_seed(7)
    policy.actor = _ReferenceAWACActor()
    policy.qnet = _ZeroActorQ()
    policy.qnet_target = _ReferenceAWACTargetQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy._teacher_actor_parameters = list(policy.actor.parameters())
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        policy.actor.parameters(), lr=0.05
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict(
            {REF_JPOS_KEY: torch.zeros(2, 1)},
            batch_size=[2],
        )
    )
    actuator_context = torch.tensor([[0.25], [0.75]])
    batch = {
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[1.0], [-1.0]]),
        "teacher_actuator_context": actuator_context,
    }
    initial_loc = policy.actor.loc.detach().clone()
    initial_log_scale = policy.actor.log_scale.detach().clone()

    metrics = policy._teacher_actor_update(batch)

    assert policy.actor.loc.item() > initial_loc.item()
    assert torch.equal(policy.actor.log_scale.detach(), initial_log_scale)
    assert policy.actor.log_scale.grad is None
    assert policy.qnet_target.dummy.grad is None
    assert len(policy.qnet_target.seen) == 3
    for observations, _ in policy.qnet_target.seen:
        assert torch.equal(observations[:, -1:], actuator_context)
    assert torch.equal(
        policy.qnet_target.seen[0][1], torch.tensor([[1.0], [-1.0]])
    )
    assert torch.equal(
        policy.qnet_target.seen[1][1], torch.zeros(2, 1)
    )
    assert metrics["awac_active"].item() == 1.0
    assert metrics["awac_advantage_mean"].item() == pytest.approx(-0.5)
    assert metrics["awac_positive_advantage_fraction"].item() == 0.5
    assert metrics["awac_weight_mean"].item() == pytest.approx(1.0)
    assert 0.0 < metrics["awac_weight_ess_fraction"].item() <= 1.0
    assert metrics["actor_sac_loss"].item() == 0.0
    assert policy.sac_actor_update_count == 1


def test_teacher_uncertainty_gate_rejects_q_but_keeps_entropy_gradient():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_max_grad_norm=0.0,
        sac_teacher_actor_uncertainty_gate=True,
        sac_clipped_double_q=True,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _EntropyGradientDist(
            actor_parameter, td.batch_size[0]
        )
    )
    policy.qnet = _RelativeUncertaintyActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict({REF_JPOS_KEY: torch.zeros(2, 1)}, batch_size=[2])
    )
    batch = {
        # d1=+1 and d2=-1 on every row: mean improvement is zero while
        # relative twin uncertainty is two, so every Q contribution is gated.
        "critic_observations": torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
        "actions": torch.zeros(2, 1),
    }

    metrics = policy._teacher_actor_update(batch)

    assert metrics["actor_uncertainty_gate_acceptance_fraction"].item() == 0.0
    assert (
        metrics["actor_uncertainty_gate_accepted_confidence_margin"].item()
        == 0.0
    )
    assert metrics["actor_uncertainty_gate_mean_confidence_margin"].item() == -2.0
    assert metrics[
        "actor_uncertainty_gate_policy_replay_improvement"
    ].item() == 0.0
    assert metrics[
        "actor_uncertainty_gate_policy_replay_disagreement"
    ].item() == 2.0
    # The rejected Q term would contribute gradient +1 through min(Q1,Q2).
    # Only alpha*d(log_prob)/d(parameter)=0.5 remains in the actual actor loss.
    assert metrics["actor_sac_loss"].item() == pytest.approx(1.5)
    assert metrics["actor_loss"].item() == pytest.approx(0.5)
    assert metrics["actor_grad_norm"].item() == pytest.approx(0.5)
    assert policy.qnet.dummy.grad is None
    assert policy.qnet.dummy.requires_grad


def test_teacher_uncertainty_gate_anchors_to_replay_not_raw_reference():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_max_grad_norm=0.0,
        sac_teacher_actor_uncertainty_gate=True,
        sac_clipped_double_q=True,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(
            actor_parameter, td.batch_size[0]
        )
    )
    policy.qnet = _RelativeUncertaintyActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict({REF_JPOS_KEY: torch.zeros(2, 1)}, batch_size=[2])
    )
    batch = {
        "critic_observations": torch.tensor(
            [[3.0, 2.0], [3.0, 2.0]]
        ),
        # pi=1 would beat the raw reference action 0 under both heads, but it
        # is worse than the behavior action 2 that actually generated the row.
        "actions": torch.full((2, 1), 2.0),
    }

    metrics = policy._teacher_actor_update(batch)

    assert torch.equal(policy.qnet.seen_actions[0], torch.ones(2, 1))
    assert torch.equal(policy.qnet.seen_actions[1], batch["actions"])
    assert metrics[
        "actor_uncertainty_gate_acceptance_fraction"
    ].item() == 0.0
    assert metrics[
        "actor_uncertainty_gate_policy_replay_improvement"
    ].item() == pytest.approx(-2.5)
    assert metrics[
        "actor_uncertainty_gate_policy_replay_disagreement"
    ].item() == pytest.approx(1.0)
    assert metrics[
        "actor_uncertainty_gate_mean_confidence_margin"
    ].item() == pytest.approx(-3.5)


def test_teacher_uncertainty_gate_uses_full_batch_denominator():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_max_grad_norm=0.0,
        sac_teacher_actor_uncertainty_gate=True,
        sac_clipped_double_q=True,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(actor_parameter, td.batch_size[0])
    )
    policy.qnet = _RelativeUncertaintyActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict({REF_JPOS_KEY: torch.zeros(2, 1)}, batch_size=[2])
    )
    batch = {
        # Row 0: d=(3,2), margin=2.5-1=1.5, accepted.
        # Row 1: d=(1,-1), margin=0-2=-2, rejected.
        "critic_observations": torch.tensor([[3.0, 2.0], [1.0, -1.0]]),
        "actions": torch.zeros(2, 1),
    }

    metrics = policy._teacher_actor_update(batch)

    assert metrics["actor_uncertainty_gate_acceptance_fraction"].item() == 0.5
    assert (
        metrics["actor_uncertainty_gate_accepted_confidence_margin"].item()
        == pytest.approx(1.5)
    )
    assert metrics["actor_uncertainty_gate_mean_confidence_margin"].item() == (
        pytest.approx(-0.25)
    )
    assert metrics[
        "actor_uncertainty_gate_policy_replay_improvement"
    ].item() == pytest.approx(1.25)
    assert metrics[
        "actor_uncertainty_gate_policy_replay_disagreement"
    ].item() == pytest.approx(1.5)
    # Only min(Q1,Q2)=2 from row 0 contributes, divided by the full batch of 2.
    # An accepted-row denominator would incorrectly produce a gradient norm of 2.
    assert metrics["actor_grad_norm"].item() == pytest.approx(1.0)
    assert policy.qnet.dummy.grad is None


@pytest.mark.parametrize(
    "clipped_double_q, expected_grad_norm", [(True, 1.0), (False, 2.0)]
)
def test_teacher_actor_uses_pessimistic_or_legacy_q_reduction(
    clipped_double_q, expected_grad_norm
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_max_grad_norm=0.0,
        sac_clipped_double_q=clipped_double_q,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(actor_parameter, td.batch_size[0])
    )
    policy.qnet = _AsymmetricActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: (
        TensorDict({REF_JPOS_KEY: torch.zeros(2, 1)}, batch_size=[2])
    )
    batch = {
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
    }

    metrics = policy._teacher_actor_update(batch)

    assert metrics["actor_grad_norm"].item() == pytest.approx(
        expected_grad_norm
    )
    assert metrics["deterministic_reference_q1_advantage"].item() == 1.0
    assert metrics["deterministic_reference_q2_advantage"].item() == 3.0
    assert (
        metrics["deterministic_reference_pessimistic_advantage"].item()
        == 1.0
    )
    assert (
        metrics["deterministic_reference_advantage_disagreement"].item()
        == 2.0
    )


def test_teacher_actor_and_diagnostics_send_only_normalized_actions_to_q():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        sac_teacher_actor_max_grad_norm=0.0,
        sac_q_normalize_actions=True,
    )
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    policy.actor = SimpleNamespace(
        get_dist=lambda td: _ActorLossDist(actor_parameter, td.batch_size[0])
    )
    policy.qnet = _ZeroActorQ()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy._teacher_actor_parameters = [actor_parameter]
    policy.sac_teacher_actor_optimizer = torch.optim.SGD(
        [actor_parameter], lr=0.0
    )
    policy.sac_actor_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: TensorDict(
        {REF_JPOS_KEY: torch.zeros(2, 1)}, batch_size=[2]
    )
    batch = {
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[-2.0], [2.0]]),
    }

    metrics = policy._teacher_actor_update(batch)

    # Sampled policy, deterministic policy, reference, and replay diagnostics.
    assert len(policy.qnet.seen_actions) == 4
    assert torch.equal(policy.qnet.seen_actions[0], torch.full((2, 1), 0.5))
    assert torch.equal(policy.qnet.seen_actions[1], torch.full((2, 1), 0.5))
    assert torch.equal(policy.qnet.seen_actions[2], torch.zeros(2, 1))
    assert torch.equal(
        policy.qnet.seen_actions[3], torch.tensor([[-1.0], [1.0]])
    )
    assert metrics["normalized_action_mean_deviation"].item() == pytest.approx(0.5)
    assert metrics["deterministic_policy_reference_advantage"].item() == 0.0
    for key in (
        "deterministic_policy_q_mean",
        "reference_q_mean",
        "replay_q_mean",
        "deterministic_policy_twin_q_disagreement",
        "reference_twin_q_disagreement",
        "replay_twin_q_disagreement",
    ):
        assert key in metrics


def test_true_teacher_rejects_markerless_old_ppo_hybrid_checkpoint():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    with pytest.raises(ValueError, match="old PPO-based"):
        policy.load_state_dict({"actor_backend": "hoi_fastsac_tanh_gaussian_v1"})


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"stage1_update_mode": dict(FASTSAC_STAGE1_UPDATE_MODE)},
        {
            "stage1_ppo_warmup_config": {"rollouts": 0},
            "stage1_ppo_behavior_distill_config": {"rollouts": 0},
            "stage1_distilled_ppo_warmup_config": {"end_rollout": 0},
        },
    ],
)
def test_pure_fastsac_checkpoint_provenance_accepts_pure_history(state):
    _validate_pure_fastsac_checkpoint_provenance(state)


@pytest.mark.parametrize(
    "state",
    [
        {"stage1_update_mode": {"version": 1, "mode": "ppo_assisted"}},
        {"stage1_ppo_warmup_config": {"rollouts": 1}},
        {"stage1_ppo_behavior_distill_config": {"rollouts": 1}},
        {"stage1_distilled_ppo_warmup_config": {"end_rollout": 1}},
        {
            "stage1_update_mode": dict(FASTSAC_STAGE1_UPDATE_MODE),
            "stage1_ppo_warmup_config": {"rollouts": 1},
        },
    ],
)
def test_pure_fastsac_checkpoint_provenance_rejects_ppo_assistance(state):
    with pytest.raises(ValueError, match="pure FastSAC|PPO-assisted"):
        _validate_pure_fastsac_checkpoint_provenance(state)


def test_reference_residual_semantics_reject_pre_v13_checkpoint():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    with pytest.raises(ValueError, match="current VAIC FastSAC training path"):
        policy.load_state_dict({
            "training_algorithm": "vaic_fastsac_teacher_v12",
            "actor_backend": FASTSAC_ACTOR_BACKEND,
        })


def test_stage1_resume_ignores_h5_and_starts_compact_live_replay(monkeypatch):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        save_teacher_buffer=False,
        teacher_buffer_capacity=5,
        teacher_training_replay_device="cpu",
    )
    policy.device = torch.device("cpu")
    policy._q_critic_dim = 4
    policy.action_dim = 2
    policy._teacher_replay_extra_shapes = {
        TEACHER_REF_ACTION_FIELD: (2,),
        NEXT_TEACHER_REF_ACTION_FIELD: (2,),
    }
    policy._teacher_replay_constant_shapes = {}
    policy._loaded_checkpoint_phase = "train"
    policy._teacher_export_started = False
    policy._loaded_teacher_replay_metadata = {"snapshot_id": "old-snapshot"}
    object.__setattr__(policy, "_replay_vecnorm", SimpleNamespace())

    def configure_base(self, path, restore_path=None):
        pytest.fail("Stage-1 must not configure the dense H5 replay")

    monkeypatch.setattr(_FastSACVAICBase, "configure_teacher_replay", configure_base)
    policy.configure_teacher_replay("new.h5", restore_path="future-final.h5")
    assert isinstance(policy.teacher_replay, TeacherTrainingReplayBuffer)
    assert policy.teacher_replay.size == policy.teacher_replay.seen == 0
    assert "observations" not in policy.teacher_replay.storage_fields
    assert "next_observations" not in policy.teacher_replay.storage_fields
    assert policy._loaded_teacher_replay_metadata is None
    assert policy._teacher_export_started is False
    assert policy._teacher_export_start_seen is None


def test_same_stage_resume_continues_after_completed_iteration(monkeypatch):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train", use_object_adapt=False)
    progress = {}
    policy.env = SimpleNamespace(
        set_progress=lambda iteration: progress.setdefault("iteration", iteration)
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}

    monkeypatch.setattr(
        _FastSACVAICBase,
        "load_state_dict",
        lambda self, state_dict, strict=True: [],
    )
    policy.load_state_dict({
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
        "last_iter": 5100,
        "next_iter": 5101,
        "stage1_n_step_config": policy._stage1_n_step_config(),
    })

    assert progress["iteration"] == 5101


def test_same_stage_resume_rejects_n_step_semantics_mismatch():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train", use_object_adapt=False, sac_teacher_n_steps=4
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}

    with pytest.raises(ValueError, match="checkpoint n-step config"):
        policy.load_state_dict({
            "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
            "actor_backend": FASTSAC_ACTOR_BACKEND,
            "actor_backend_config": {"actor": "same"},
            "q_backend_config": {"q": "same"},
            "teacher_replay_id": "replay",
            "last_phase": "train",
            "stage1_n_step_config": {
                "n_steps": 1,
                "semantics": policy._stage1_n_step_config()["semantics"],
            },
        })


@pytest.mark.parametrize(
    "checkpoint_config",
    [
        None,
        {
            "coefficient": 0.25,
            "margin": 0.002,
            "temperature": 0.002,
            "starts_q_updates": 8_000,
            "semantics": "deterministic_policy_vs_replay_expected_c51_softplus_margin_v1",
        },
    ],
)
def test_same_stage_resume_rejects_missing_or_changed_conservative_q_config(
    checkpoint_config,
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        sac_teacher_conservative_q_coef=1.0,
        sac_teacher_conservative_q_margin=0.002,
        sac_teacher_conservative_q_temperature=0.002,
        sac_teacher_conservative_q_starts_q_updates=None,
        sac_teacher_actor_learning_starts_q_updates=8_000,
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}
    state = {
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
        "stage1_n_step_config": policy._stage1_n_step_config(),
    }
    if checkpoint_config is not None:
        state["stage1_conservative_q_config"] = checkpoint_config

    with pytest.raises(ValueError, match="conservative Q"):
        policy.load_state_dict(state)


@pytest.mark.parametrize(
    "checkpoint_config",
    [
        None,
        {"objective": "sac"},
        {
            "objective": "reference_awac",
            "beta": 0.02,
            "weight_clip": 20.0,
            "semantics": FASTSAC_STAGE1_REFERENCE_AWAC_SEMANTICS,
        },
    ],
)
def test_same_stage_resume_rejects_missing_or_changed_actor_objective(
    checkpoint_config,
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        sac_teacher_actor_objective="reference_awac",
        sac_teacher_awac_beta=0.01,
        sac_teacher_awac_weight_clip=20.0,
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}
    state = {
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
        "stage1_n_step_config": policy._stage1_n_step_config(),
    }
    if checkpoint_config is not None:
        state["stage1_actor_objective_config"] = checkpoint_config

    with pytest.raises(ValueError, match="actor objective"):
        policy.load_state_dict(state)


@pytest.mark.parametrize(
    "checkpoint_config",
    [
        None,
        {
            "enabled": True,
            "anchor": "framewise_raw_reference_action",
            "criterion": (
                "mean_twin_improvement_gt_absolute_head_disagreement"
            ),
            "q_component": "gated_full_batch_denominator",
            "entropy_component": "ungated_all_rows",
            "semantics": "legacy_raw_reference_gate",
        },
    ],
)
def test_same_stage_resume_rejects_missing_or_legacy_enabled_actor_gate(
    checkpoint_config,
):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        use_object_adapt=False,
        sac_teacher_n_steps=1,
        sac_teacher_actor_uncertainty_gate=True,
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}
    state = {
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
        "stage1_n_step_config": policy._stage1_n_step_config(),
    }
    if checkpoint_config is not None:
        state["stage1_actor_uncertainty_gate_config"] = checkpoint_config

    with pytest.raises(
        ValueError, match="behavior-anchored|actor uncertainty gate config"
    ):
        policy.load_state_dict(state)


def test_checkpoint_rejects_q_action_input_gain_mismatch():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q_action_input_gain": 2.0}

    with pytest.raises(ValueError, match="FastSAC Q checkpoint config"):
        policy.load_state_dict({
            "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
            "actor_backend": FASTSAC_ACTOR_BACKEND,
            "actor_backend_config": {"actor": "same"},
            "q_backend_config": {"q_action_input_gain": 1.0},
        })


def test_checkpoint_treats_pre_fusion_metadata_as_early_only(monkeypatch):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="finetune", use_object_adapt=False)
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    expected_q = {
        "q": "same",
        "q_action_fusion": "early",
        "q_action_hidden_dim": 0,
        "q_action_fusion_semantics": "input_concat_then_shared_trunk_v1",
        "q_reference_dueling": False,
        "q_architecture_semantics": (
            FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS
        ),
    }
    policy._q_backend_metadata = lambda: expected_q
    monkeypatch.setattr(
        _FastSACVAICBase,
        "load_state_dict",
        lambda self, state_dict, strict=True: [],
    )
    legacy_state = {
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
    }

    policy.load_state_dict(legacy_state)

    policy._q_backend_metadata = lambda: {
        **expected_q,
        "q_action_fusion": "late",
        "q_action_hidden_dim": 128,
        "q_action_fusion_semantics": (
            "separate_obs_and_action_stems_then_shared_trunk_v1"
        ),
    }
    with pytest.raises(ValueError, match="FastSAC Q checkpoint config"):
        policy.load_state_dict(legacy_state)
