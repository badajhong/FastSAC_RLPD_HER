from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTOR_BACKEND,
    FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
    FASTSAC_Q_EARLY_FUSION_SEMANTICS,
    FASTSAC_STUDENT_TRAINING_ALGORITHM,
    NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
    TEACHER_ACTUATOR_CONTEXT_FIELD,
    TEACHER_REPLAY_FIELDS,
    FastSACActor,
    FastSACVEL,
    FastSACVelConfig,
    FastSACVelFinetune,
    FastSACVelFinetuneConfig,
    OfflineReplayH5,
    TeacherReplayBuffer,
    _Stage1NStepAccumulator,
    _build_isolated_q_network,
)


def _context_metadata(delay_min=2, delay_max=6, alpha_low=0.8, alpha_high=1.0):
    return {
        "enabled": True,
        "semantics": FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
        "dimension": delay_max - delay_min + 2,
        "delay_range": [delay_min, delay_max],
        "alpha_range": [alpha_low, alpha_high],
    }


def _context_policy(policy_cls=FastSACVEL):
    policy = policy_cls.__new__(policy_cls)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_condition_on_actuator_state=True,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        sac_q_normalize_actions=False,
        sac_q_action_input_gain=1.0,
    )
    policy._q_actuator_context_metadata_value = _context_metadata()
    policy._q_actuator_context_dim = 6
    policy._q_critic_dim = 1
    policy._q_input_dim = 7
    policy._rollout_q_actuator_contexts = []
    manager = SimpleNamespace(
        min_delay=2,
        max_delay=6,
        alpha_range=(0.8, 1.0),
        delay=torch.tensor([[2], [3], [4], [5], [6]]),
        alpha=torch.tensor([[0.8], [0.85], [0.9], [0.95], [1.0]]),
    )
    policy.env = SimpleNamespace(action_manager=manager)
    return policy


def test_actuator_context_is_default_off_for_both_stages():
    assert FastSACVelConfig().q_condition_on_actuator_state is False
    assert FastSACVelFinetuneConfig().q_condition_on_actuator_state is False


def test_actuator_context_encoding_and_pre_step_snapshot_are_exact():
    policy = _context_policy()
    captured = policy.capture_q_actuator_context()

    assert captured.shape == (5, 6)
    assert torch.equal(captured[:, :5], torch.eye(5))
    assert torch.allclose(
        captured[:, -1],
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
        atol=1e-6,
    )

    # A reset may mutate the manager immediately after the snapshot. The replay
    # owner must still hold the old episode's independent tensor.
    policy.env.action_manager.delay.fill_(6)
    policy.env.action_manager.alpha.fill_(1.0)
    assert torch.equal(captured[:, :5], torch.eye(5))

    with pytest.raises(ValueError, match="outside the checkpointed context range"):
        policy._encode_q_actuator_context(
            torch.tensor([[7]]), torch.tensor([[0.9]])
        )


def test_q_wrappers_append_context_only_to_current_and_next_q_inputs():
    policy = _context_policy()

    class QSpy:
        def __init__(self):
            self.forward_observations = None
            self.projected_observations = None

        def __call__(self, observations, actions):
            self.forward_observations = observations.clone()
            return observations[..., :1]

        def projection(self, observations, actions, reward, bootstrap, discount):
            self.projected_observations = observations.clone()
            return observations[..., :1]

    spy = QSpy()
    current_observations = torch.tensor([[10.0], [20.0]])
    next_observations = torch.tensor([[11.0], [21.0]])
    current_context = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -0.5],
    ])
    next_context = torch.tensor([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
    ])
    actions = torch.zeros(2, 1)

    policy._q_forward(
        spy, current_observations, actions, actuator_context=current_context
    )
    policy._q_projection(
        spy,
        next_observations,
        actions,
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
        actuator_context=next_context,
    )

    assert torch.equal(
        spy.forward_observations,
        torch.cat((current_observations, current_context), dim=-1),
    )
    assert torch.equal(
        spy.projected_observations,
        torch.cat((next_observations, next_context), dim=-1),
    )
    assert torch.equal(current_observations, torch.tensor([[10.0], [20.0]]))
    assert torch.equal(next_observations, torch.tensor([[11.0], [21.0]]))


def test_q_context_dimension_does_not_change_actor_initialization_or_output():
    def actor_built_after_q(q_observation_dim):
        torch.manual_seed(1234)
        _build_isolated_q_network(
            q_observation_dim,
            action_dim=2,
            hidden_dim=8,
            num_atoms=5,
            v_min=-2.0,
            v_max=2.0,
            layer_norm=False,
            device="cpu",
            seed=77,
        )
        return FastSACActor(
            input_dim=3,
            action_dim=2,
            hidden_dim=8,
            log_std_min=-5.0,
            log_std_max=0.0,
            action_low=torch.tensor([-1.0, -2.0]),
            action_high=torch.tensor([1.0, 2.0]),
            layer_norm=False,
        )

    actor_without_context = actor_built_after_q(4)
    actor_with_context = actor_built_after_q(10)
    for name, value in actor_without_context.state_dict().items():
        assert torch.equal(value, actor_with_context.state_dict()[name])
    observation = torch.tensor([[0.2, -0.3, 0.4]])
    for left, right in zip(
        actor_without_context(observation), actor_with_context(observation)
    ):
        assert torch.equal(left, right)


def test_interleaved_timeout_keeps_pre_reset_actuator_context():
    policy = _context_policy()
    policy.q_critic_keys = ["critic"]
    policy.action_dim = 1
    policy._teacher_raw_replay_fields = []
    policy._teacher_n_step_accumulator = _Stage1NStepAccumulator(
        1,
        0.99,
        next_fields=(
            "next_critic_observations",
            NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
        ),
    )
    policy._last_truncation_finals_used = 0
    policy._prepare_teacher_final_state = lambda td: {
        "next_critic_observations": td["critic"].clone()
    }

    current = TensorDict(
        {
            "critic": torch.tensor([[1.0], [2.0]]),
            ACTION_KEY: torch.tensor([[0.1], [0.2]]),
            "step_count": torch.tensor([[2], [2]]),
            "next": TensorDict(
                {
                    # Row 1 is the true pre-reset timeout final state.
                    "critic": torch.tensor([[11.0], [102.0]]),
                    "reward": torch.tensor([[1.0], [2.0]]),
                    "done": torch.tensor([[False], [True]]),
                    "terminated": torch.tensor([[False], [False]]),
                    "discount": torch.ones(2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor([[False], [True]]),
                            "command_finished": torch.tensor([[False], [False]]),
                        },
                        batch_size=[2],
                        device="cpu",
                    ),
                },
                batch_size=[2],
                device="cpu",
            ),
        },
        batch_size=[2],
        device="cpu",
    )
    reset_carry = TensorDict(
        {"critic": torch.tensor([[11.0], [202.0]])},
        batch_size=[2],
        device="cpu",
    )
    context = policy._encode_q_actuator_context(
        torch.tensor([[2], [4]]), torch.tensor([[0.8], [0.9]])
    )
    transitions = policy._teacher_transition_from_step(
        current, reset_carry, context
    )

    assert transitions["next_critic_observations"][:, 0].tolist() == [11.0, 102.0]
    assert torch.equal(transitions[TEACHER_ACTUATOR_CONTEXT_FIELD], context)
    assert torch.equal(
        transitions[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD], context
    )


def test_n_step_keeps_start_context_and_endpoint_next_context():
    accumulator = _Stage1NStepAccumulator(
        3,
        0.99,
        next_fields=(NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,),
    )
    result = None
    for marker in (0.0, 1.0, 2.0):
        transition = {
            "critic_observations": torch.tensor([[marker]]),
            "actions": torch.tensor([[marker]]),
            "rewards": torch.ones(1),
            "dones": torch.zeros(1, dtype=torch.bool),
            "truncations": torch.zeros(1, dtype=torch.bool),
            "discounts": torch.ones(1),
            TEACHER_ACTUATOR_CONTEXT_FIELD: torch.tensor([[marker, 10.0]]),
            NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: torch.tensor(
                [[marker + 0.5, 20.0]]
            ),
        }
        result = accumulator.append(transition, torch.tensor([True]))

    assert result is not None
    assert torch.equal(
        result[TEACHER_ACTUATOR_CONTEXT_FIELD], torch.tensor([[0.0, 10.0]])
    )
    assert torch.equal(
        result[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD],
        torch.tensor([[2.5, 20.0]]),
    )


def test_student_online_replay_uses_each_pre_step_context_for_both_fields():
    policy = _context_policy(FastSACVelFinetune)
    policy.cfg.train_every = 2
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_dim = 1
    policy.action_dim = 1
    policy._last_truncation_finals_used = 0
    policy._truncation_final_batches = []
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[90.0]]),
        "next_critic_observations": torch.tensor([[91.0]]),
    }
    first = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    second = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, -0.5]])
    policy.record_rollout_q_actuator_context(first)
    policy.record_rollout_q_actuator_context(second)
    marker = torch.tensor([[[0.0], [1.0]]])
    done = torch.zeros(1, 2, 1, dtype=torch.bool)
    rollout = TensorDict(
        {
            "actor": marker,
            "critic": marker + 10.0,
            ACTION_KEY: marker + 20.0,
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": marker,
                    "done": done,
                    "terminated": done.clone(),
                    "discount": torch.ones_like(marker),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": done.clone(),
                            "command_finished": done.clone(),
                        },
                        batch_size=[1, 2],
                        device="cpu",
                    ),
                },
                batch_size=[1, 2],
                device="cpu",
            ),
        },
        batch_size=[1, 2],
        device="cpu",
    )

    chunks = list(policy._student_transition_chunks(rollout))
    assert torch.equal(chunks[0][TEACHER_ACTUATOR_CONTEXT_FIELD], first)
    assert torch.equal(chunks[0][NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD], first)
    assert torch.equal(chunks[1][TEACHER_ACTUATOR_CONTEXT_FIELD], second)
    assert torch.equal(chunks[1][NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD], second)


def _h5_batch(context):
    count = context.shape[0]
    batch = {
        "observations": torch.zeros(count, 2),
        "critic_observations": torch.zeros(count, 3),
        "actions": torch.zeros(count, 1),
        "rewards": torch.arange(count, dtype=torch.float32),
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.ones(count),
        "next_observations": torch.ones(count, 2),
        "next_critic_observations": torch.ones(count, 3),
        TEACHER_ACTUATOR_CONTEXT_FIELD: context,
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: context + 0.25,
    }
    assert set(TEACHER_REPLAY_FIELDS).issubset(batch)
    return batch


def test_actuator_context_h5_roundtrip_and_enabled_disabled_guard(tmp_path):
    path = tmp_path / "teacher_context.h5"
    metadata = _context_metadata()
    extras = {
        TEACHER_ACTUATOR_CONTEXT_FIELD: (6,),
        NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: (6,),
    }
    replay = TeacherReplayBuffer(
        path,
        capacity=4,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        seed=0,
        extra_shapes=extras,
        q_actuator_context=metadata,
    )
    context = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -0.5],
    ])
    replay.append(_h5_batch(context))
    replay.snapshot(iteration=3, checkpoint_name="checkpoint_3")

    offline = OfflineReplayH5(
        path,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        expected_q_actuator_context=metadata,
    )
    assert offline.snapshot_metadata["q_actuator_context"] == metadata
    assert torch.equal(
        offline.data[TEACHER_ACTUATOR_CONTEXT_FIELD], context
    )
    assert torch.equal(
        offline.data[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD], context + 0.25
    )

    with pytest.raises(ValueError, match="does not match policy metadata"):
        OfflineReplayH5(
            path,
            actor_dim=2,
            critic_dim=3,
            action_dim=1,
            expected_q_actuator_context={"enabled": False},
        )


def test_actuator_conditioned_checkpoint_rejects_legacy_disabled_q_metadata():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="finetune", use_object_adapt=False)
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    expected_q = {
        "q_action_fusion": "early",
        "q_action_hidden_dim": 0,
        "q_action_fusion_semantics": FASTSAC_Q_EARLY_FUSION_SEMANTICS,
        "q_input_dim": 7,
        "q_actuator_context": _context_metadata(),
    }
    policy._q_backend_metadata = lambda: expected_q
    checkpoint = {
        "training_algorithm": FASTSAC_STUDENT_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        # Missing context metadata is interpreted only as legacy disabled.
        "q_backend_config": {
            "q_action_fusion": "early",
            "q_action_hidden_dim": 0,
            "q_action_fusion_semantics": FASTSAC_Q_EARLY_FUSION_SEMANTICS,
            "q_input_dim": 7,
        },
    }

    with pytest.raises(ValueError, match="Q checkpoint config"):
        policy.load_state_dict(checkpoint)
