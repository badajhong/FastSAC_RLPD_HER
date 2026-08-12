from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_REFERENCE_EPS,
    TwinDistributionalQ,
    _fastsac_action_center_to_latent,
    _sac_bootstrap_mask,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
    _DeviceReplay,
    PPOBCDaggerFinetune,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from active_adaptation.learning.ppo.td3_bc_dagger import (
    ACTOR_BACKEND,
    ACTOR_LEARNING_SEMANTICS,
    CHECKPOINT_VERSION,
    CRITIC_SEMANTICS,
    TRAINING_ALGORITHM,
    DistributionalTD3TeacherBC,
    DistributionalTD3TeacherBCConfig,
    _DistributionalTD3DaggerRolloutPolicy,
    _TD3DeviceReplay,
    _apply_student_collector_noise,
    _categorical_expected_value,
    _exact_teacher_bc_loss,
    _polyak_update_,
    _prefetch_td3_replay_sample_plans,
    _project_c51_probabilities,
    _select_lower_expected_c51_distribution,
    _td3_actor_q1_loss,
)


def _parameter_storage(module: nn.Module) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in module.parameters()}


def _bare_policy(**cfg) -> DistributionalTD3TeacherBC:
    policy = DistributionalTD3TeacherBC.__new__(DistributionalTD3TeacherBC)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    policy.device = torch.device("cpu")
    return policy


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _take_optimizer_step(module: nn.Module, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in module.parameters())
    loss.backward()
    optimizer.step()


def _replay_rows(count: int, offset: int) -> dict[str, torch.Tensor]:
    row = torch.arange(count, dtype=torch.float32) + float(offset)
    return {
        "observations": torch.stack((row, row + 0.25), dim=-1),
        "critic_observations": torch.stack((row, row + 0.5, row + 0.75), dim=-1),
        "actions": row[:, None] + 1.0,
        "rewards": row + 2.0,
        "dones": torch.arange(count) % 3 == 0,
        "truncations": torch.arange(count) % 4 == 0,
        "discounts": torch.full((count,), 0.95),
        "next_observations": torch.stack((row + 3.0, row + 3.25), dim=-1),
        "next_critic_observations": torch.stack(
            (row + 4.0, row + 4.5, row + 4.75), dim=-1
        ),
    }


def _sampling_policy(replay_type, seed: int, device):
    policy = _bare_policy(q_batch_size=6, dagger_batch_size=5, policy_delay=2)
    policy.device = torch.device(device)
    policy.q_rng = torch.Generator(device=device).manual_seed(seed)
    policy.critic_update_count = 3
    policy.q_teacher_replay = replay_type(16, "cpu")
    policy.dagger_replay = replay_type(16, "cpu")
    teacher = _replay_rows(7, 100)
    student = _replay_rows(11, 1_000)
    student[DAGGER_IS_STUDENT_ACTION_KEY] = torch.tensor(
        [False, True, True, False, True, False, True, True, False, True, False]
    )
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(11, dtype=torch.float32)[:, None] + 2_000.0
    )
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.arange(11) % 3 != 0
    policy.q_teacher_replay.extend(teacher)
    policy.dagger_replay.extend(student)
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )
    return policy


def test_td3_indexed_cpu_replay_preserves_requested_fields_values_and_state():
    replay = _TD3DeviceReplay(8, "cpu")
    rows = {
        "vector": torch.arange(18, dtype=torch.float32).reshape(6, 3),
        "valid": torch.tensor([True, False, True, True, False, True]),
        "integer": torch.arange(6, dtype=torch.int64),
    }
    replay.extend(rows)
    before = (replay.ptr, replay.size, replay.seen)
    indices = torch.tensor([4, 1, 4, 0], dtype=torch.long)

    sampled = replay.sample_by_indices(
        indices, "cpu", fields=("valid", "vector", "integer")
    )

    assert tuple(sampled) == ("valid", "vector", "integer")
    for key in sampled:
        assert torch.equal(sampled[key], rows[key].index_select(0, indices))
        assert sampled[key].dtype == rows[key].dtype
        assert sampled[key].shape == (indices.numel(), *rows[key].shape[1:])
        assert sampled[key].device == torch.device("cpu")
    assert (replay.ptr, replay.size, replay.seen) == before
    assert replay._pinned_sample_staging == {}


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ),
)
def test_prefetched_td3_cpu_sampling_matches_sequential_rng_and_batches_exactly(
    device,
):
    seed = 731
    update_count = 5
    sequential = _sampling_policy(_DeviceReplay, seed, device)
    prefetched = _sampling_policy(_TD3DeviceReplay, seed, device)
    actor_fields = (
        "observations",
        "critic_observations",
        DAGGER_REPLAY_TEACHER_ACTIONS,
        DAGGER_TEACHER_ACTION_VALID_KEY,
    )

    expected = []
    for update_index in range(update_count):
        q_batch = PPOBCDaggerFinetune._sample_balanced_q_batch(sequential)
        actor_batch = None
        if (sequential.critic_update_count + update_index + 1) % 2 == 0:
            actor_batch = sequential.dagger_replay.sample(
                5,
                sequential.device,
                sequential.q_rng,
                fields=actor_fields,
            )
        expected.append((q_batch, actor_batch))

    plans = _prefetch_td3_replay_sample_plans(
        prefetched.dagger_replay,
        prefetched.q_teacher_replay,
        q_batch_size=6,
        actor_batch_size=5,
        update_count=update_count,
        policy_delay=2,
        critic_update_count=prefetched.critic_update_count,
        output_device=prefetched.device,
        generator=prefetched.q_rng,
    )
    actual = []
    for plan in plans:
        q_batch = prefetched._sample_balanced_q_batch(plan)
        actor_batch = (
            None
            if plan.actor_indices is None
            else prefetched._sample_actor_batch(plan.actor_indices)
        )
        actual.append((q_batch, actor_batch))

    assert torch.equal(sequential.q_rng.get_state(), prefetched.q_rng.get_state())
    for (expected_q, expected_actor), (actual_q, actual_actor) in zip(expected, actual):
        _assert_nested_equal(expected_q, actual_q)
        if expected_actor is None:
            assert actual_actor is None
        else:
            _assert_nested_equal(expected_actor, actual_actor)
        source = actual_q[DAGGER_Q_TEACHER_SOURCE_KEY]
        assert source.dtype is torch.bool
        assert source.sum().item() == 3
        assert source.numel() - source.sum().item() == 3
        assert all(value.device == torch.device(device) for value in actual_q.values())
        # Row encodings distinguish the teacher and student source rings.
        assert torch.all(actual_q["actions"][source] < 1_000.0)
        assert torch.all(actual_q["actions"][~source] >= 1_000.0)
        student_rows = (actual_q["actions"][~source, 0] - 1_001.0).long()
        assert torch.all(
            prefetched.dagger_replay.data[DAGGER_IS_STUDENT_ACTION_KEY][
                student_rows.cpu()
            ]
        )
        if torch.device(device).type == "cuda":
            assert all(value.is_contiguous() for value in actual_q.values())

    if torch.device(device).type == "cuda":
        staging = (
            *prefetched.dagger_replay._pinned_sample_staging.values(),
            *prefetched.q_teacher_replay._pinned_sample_staging.values(),
        )
        assert staging
        assert all(value.is_pinned() for value in staging)
    restored_rng = torch.Generator(device=device)
    restored_rng.set_state(prefetched.q_rng.get_state())
    assert torch.equal(
        torch.rand(16, device=device, generator=prefetched.q_rng),
        torch.rand(16, device=device, generator=restored_rng),
    )


def test_td3_config_rejects_export_fifo_smaller_than_q_teacher_fifo():
    cfg = DistributionalTD3TeacherBCConfig()
    cfg.save_teacher_buffer = True
    cfg.q_teacher_buffer_capacity = 128
    cfg.teacher_buffer_capacity = 127

    with pytest.raises(ValueError, match="teacher_buffer_capacity"):
        DistributionalTD3TeacherBC._validate_td3_config(cfg)

    cfg.save_teacher_buffer = False
    DistributionalTD3TeacherBC._validate_td3_config(cfg)


def test_lower_expected_head_selects_one_complete_c51_distribution_per_row():
    support = torch.tensor([-1.0, 0.0, 1.0])
    probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.05, 0.15, 0.80], [0.20, 0.60, 0.20]],
            [[0.20, 0.60, 0.20], [0.65, 0.25, 0.10], [0.10, 0.80, 0.10]],
        ]
    )
    logits = probabilities.log().requires_grad_()

    selected, expected_heads, selected_head = _select_lower_expected_c51_distribution(
        logits, support
    )

    expected = torch.stack(
        (probabilities[0, 0], probabilities[1, 1], probabilities[0, 2])
    )
    assert torch.allclose(selected, expected)
    assert torch.equal(selected_head, torch.tensor([0, 1, 0]))
    assert torch.allclose(
        expected_heads,
        torch.tensor([[-0.75, 0.75, 0.00], [-0.00, -0.55, 0.00]]),
        atol=1e-7,
    )
    assert torch.allclose(selected.sum(-1), torch.ones(3))

    # An atom-wise minimum is not a probability distribution here and differs
    # from every required selected row.
    atomwise_minimum = probabilities.min(dim=0).values
    assert not torch.allclose(selected, atomwise_minimum)
    assert not torch.allclose(atomwise_minimum.sum(-1), torch.ones(3))


def test_categorical_expected_value_uses_softmax_probabilities():
    support = torch.tensor([-2.0, 0.0, 2.0])
    logits = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0]])

    expected = _categorical_expected_value(logits, support)

    manual = (logits.softmax(-1) * support).sum(-1)
    assert torch.allclose(expected, manual)
    assert expected[0].item() == pytest.approx(0.0)
    assert expected[1].item() > 0.0


def test_c51_projection_obeys_ordinary_timeout_and_terminal_truth_table():
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    source = torch.zeros(5, support.numel())
    source[:, -1] = 1.0

    # ordinary, pure timeout, true termination, command completion, and a
    # timeout coincident with true termination after the authoritative VAIC
    # cause-resolution logic. Only the pure timeout remains a truncation.
    dones = torch.tensor([False, True, True, True, True])
    truncations = torch.tensor([False, True, False, False, False])
    bootstrap = _sac_bootstrap_mask(dones, truncations)
    assert torch.equal(bootstrap, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.zeros(5),
        bootstrap=bootstrap,
        effective_discount=torch.full((5,), 0.5),
        support=support,
    )

    expected = torch.zeros_like(source)
    expected[:2, 3] = 1.0  # 0 + 0.5 * z_max = 1
    expected[2:, 2] = 1.0  # terminals collapse onto immediate reward 0
    assert torch.equal(projected, expected)
    assert torch.equal(projected.sum(-1), torch.ones(5))
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(0.0)


def test_c51_projection_interpolates_and_reports_preprojection_clipping():
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    source = torch.zeros(2, support.numel())
    source[0, 2] = 1.0
    source[1, -1] = 1.0

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.tensor([0.25, 3.0]),
        bootstrap=torch.ones(2),
        effective_discount=torch.ones(2),
        support=support,
    )

    assert torch.equal(projected[0], torch.tensor([0.0, 0.0, 0.75, 0.25, 0.0]))
    assert torch.equal(projected[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(projected.sum(-1), torch.ones(2))
    # Row 0 has one transformed atom above the support; row 1 has three.
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(4.0 / 10.0)


def test_c51_projection_accepts_locked_float32_501_atom_support():
    support = torch.linspace(-20.0, 20.0, 501, dtype=torch.float32)
    source = torch.zeros(1, support.numel())
    source[0, support.numel() // 2] = 1.0

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.zeros(1),
        bootstrap=torch.ones(1),
        effective_discount=torch.ones(1),
        support=support,
    )

    assert torch.allclose(projected, source)
    assert projected.sum().item() == pytest.approx(1.0)
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(0.0)


def test_both_online_critics_train_against_the_same_detached_projection():
    support = torch.tensor([-1.0, 0.0, 1.0])
    target_probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.10, 0.20, 0.70]],
            [[0.20, 0.60, 0.20], [0.70, 0.20, 0.10]],
        ],
        requires_grad=True,
    )
    with torch.no_grad():
        selected, _, _ = _select_lower_expected_c51_distribution(
            target_probabilities.log(), support
        )
        projected, _, _ = _project_c51_probabilities(
            selected,
            rewards=torch.zeros(2),
            bootstrap=torch.ones(2),
            effective_discount=torch.ones(2),
            support=support,
        )
    online_logits = nn.Parameter(
        torch.tensor(
            [
                [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]],
                [[-0.3, 0.4, 0.2], [0.6, -0.2, 0.0]],
            ]
        )
    )

    per_head = (
        -(projected.unsqueeze(0) * F.log_softmax(online_logits, dim=-1))
        .sum(-1)
        .mean(-1)
    )
    per_head.sum().backward()

    expected_gradient = (
        online_logits.detach().softmax(-1) - projected.unsqueeze(0)
    ) / projected.shape[0]
    assert torch.allclose(online_logits.grad, expected_gradient)
    assert target_probabilities.grad is None
    assert projected.grad_fn is None


def test_online_twins_and_all_targets_have_independent_parameter_storage():
    online_q = TwinDistributionalQ(
        obs_dim=3,
        action_dim=2,
        hidden_dim=8,
        num_atoms=5,
        v_min=-2.0,
        v_max=2.0,
        layer_norm=False,
    )
    target_q = copy.deepcopy(online_q).requires_grad_(False)
    online_actor = nn.Linear(3, 2)
    target_actor = copy.deepcopy(online_actor).requires_grad_(False)

    q1_storage = _parameter_storage(online_q.qnets[0])
    q2_storage = _parameter_storage(online_q.qnets[1])
    assert q1_storage.isdisjoint(q2_storage)
    assert _parameter_storage(online_q).isdisjoint(_parameter_storage(target_q))
    assert _parameter_storage(online_actor).isdisjoint(_parameter_storage(target_actor))
    assert all(not parameter.requires_grad for parameter in target_q.parameters())
    assert all(not parameter.requires_grad for parameter in target_actor.parameters())

    target_q_before = [
        parameter.detach().clone() for parameter in target_q.parameters()
    ]
    q2_before = [
        parameter.detach().clone() for parameter in online_q.qnets[1].parameters()
    ]
    with torch.no_grad():
        next(online_q.qnets[0].parameters()).add_(1.0)
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(target_q.parameters(), target_q_before)
    )
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(online_q.qnets[1].parameters(), q2_before)
    )


class _ActionSensitiveC51Head(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, observations, actions):
        del observations
        action = actions[:, 0]
        return torch.stack(
            (-self.scale * action, torch.zeros_like(action), self.scale * action),
            dim=-1,
        )


class _ActionSensitiveTwinC51(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnets = nn.ModuleList(
            (_ActionSensitiveC51Head(1.0), _ActionSensitiveC51Head(-7.0))
        )
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(qnet(observations, actions) for qnet in self.qnets), dim=0
        )


class _TableC51Head(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, observations, actions):
        del actions
        return self.logits[: observations.shape[0]]


class _TableTwinC51(nn.Module):
    def __init__(self, head_1: torch.Tensor, head_2: torch.Tensor):
        super().__init__()
        self.qnets = nn.ModuleList((_TableC51Head(head_1), _TableC51Head(head_2)))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(qnet(observations, actions) for qnet in self.qnets), dim=0
        )


def test_actor_loss_uses_expected_online_q1_and_freezes_critic_gradients():
    actor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(0.25)
    critic = _ActionSensitiveTwinC51()
    observations = torch.ones(4, 1)
    action = actor(observations)

    loss, expected_q1 = _td3_actor_q1_loss(critic, observations, action)
    direct_logits = critic(observations, action.detach())[0]
    direct_expected_q1 = _categorical_expected_value(direct_logits, critic.support)

    assert torch.allclose(expected_q1.detach(), direct_expected_q1)
    assert torch.allclose(loss.detach(), -direct_expected_q1.mean())
    assert all(parameter.requires_grad for parameter in critic.parameters())

    loss.backward()
    assert actor.weight.grad is not None
    assert actor.weight.grad.abs().item() > 0.0
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_integrated_critic_uses_common_detached_target_and_no_sac_path():
    policy = _bare_policy(
        target_policy_noise_std=0.3,
        target_policy_noise_clip=0.2,
        q_action_input_gain=1.0,
        dagger_action_clip=1.0,
        gamma=1.0,
        q_max_grad_norm=0.0,
    )
    _install_unit_action_contract(policy)
    online_logits_1 = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    online_logits_2 = torch.tensor([[-0.3, 0.4, 0.2], [0.6, -0.2, 0.0]])
    target_probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.05, 0.15, 0.80]],
            [[0.20, 0.60, 0.20], [0.65, 0.25, 0.10]],
        ]
    )
    policy.qnet = _TableTwinC51(online_logits_1, online_logits_2)
    policy.qnet_target = _TableTwinC51(
        target_probabilities[0].log(), target_probabilities[1].log()
    ).requires_grad_(False)
    policy.actor_target = nn.Linear(1, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        policy.actor_target.weight.zero_()
    policy._actor_target_dist_from_flat = MethodType(
        lambda owner, observations: SimpleNamespace(
            mean=owner.actor_target(observations)
        ),
        policy,
    )
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.critic_update_count = 0
    policy.target_policy_rng = torch.Generator().manual_seed(31)
    policy.collector_exploration_rng = torch.Generator().manual_seed(32)
    target_rng_before = policy.target_policy_rng.get_state().clone()
    collector_rng_before = policy.collector_exploration_rng.get_state().clone()

    def forbidden_stochastic_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SAC/log-probability path was invoked")

    policy._sac_critic_dist_from_flat = forbidden_stochastic_path
    policy._normalized_action_log_prob = forbidden_stochastic_path
    policy.sac_dist_cls = forbidden_stochastic_path
    batch = {
        "observations": torch.ones(2, 1),
        "critic_observations": torch.ones(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        "next_observations": torch.ones(2, 1),
        "next_critic_observations": torch.ones(2, 1),
    }

    projected, _ = policy._distributional_td3_target(batch)
    expected_target = torch.stack(
        (target_probabilities[0, 0], target_probabilities[1, 1])
    )
    assert torch.allclose(projected, expected_target)
    assert projected.grad_fn is None
    assert projected.requires_grad is False
    assert not torch.equal(policy.target_policy_rng.get_state(), target_rng_before)
    assert torch.equal(
        policy.collector_exploration_rng.get_state(), collector_rng_before
    )

    metrics = policy._critic_update(batch)

    expected_gradient_1 = (
        online_logits_1.softmax(-1) - expected_target
    ) / expected_target.shape[0]
    expected_gradient_2 = (
        online_logits_2.softmax(-1) - expected_target
    ) / expected_target.shape[0]
    assert torch.allclose(policy.qnet.qnets[0].logits.grad, expected_gradient_1)
    assert torch.allclose(policy.qnet.qnets[1].logits.grad, expected_gradient_2)
    assert policy.critic_optimizer.step_calls == 1
    assert policy.critic_update_count == 1
    assert metrics["critic_loss"].item() == pytest.approx(
        metrics["critic_loss_1"].item() + metrics["critic_loss_2"].item()
    )
    assert all(parameter.grad is None for parameter in policy.qnet_target.parameters())
    assert all(parameter.grad is None for parameter in policy.actor_target.parameters())


def _install_unit_action_contract(policy: DistributionalTD3TeacherBC) -> None:
    policy._fastsac_action_low = torch.tensor([-1.0])
    policy._fastsac_action_high = torch.tensor([1.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([1.0])
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([1.0])


def test_actor_update_combines_td3_and_bc_before_one_backward_and_step():
    policy = _bare_policy(
        eta_td3=0.7,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        dagger_action_clip=1.0,
        q_action_input_gain=1.0,
        max_grad_norm=1.0e6,
    )
    _install_unit_action_contract(policy)
    policy.actor_adapt = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.actor_adapt.weight.fill_(0.25)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.actor_optimizer = _CountingSGD(policy.actor_adapt.parameters(), lr=0.1)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.1)
    policy.actor_update_count = 0
    policy._actor_dist_from_flat = MethodType(
        lambda owner, observations: SimpleNamespace(
            mean=owner.actor_adapt(observations)
        ),
        policy,
    )
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [-1.0]]),
        "critic_observations": torch.ones(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.5], [-0.2], [0.1]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
    }

    expected_actor = copy.deepcopy(policy.actor_adapt)
    expected_critic = copy.deepcopy(policy.qnet)
    expected_latent = expected_actor(batch["observations"])
    expected_q_action = expected_latent.tanh()
    expected_td3, _ = _td3_actor_q1_loss(
        expected_critic, batch["critic_observations"], expected_q_action
    )
    expected_bc = _exact_teacher_bc_loss(
        expected_latent,
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        batch[DAGGER_TEACHER_ACTION_VALID_KEY],
        policy._fastsac_actor_action_center,
        policy._fastsac_actor_action_scale,
        policy.cfg.dagger_actor_huber_delta,
    )
    expected_total = (
        policy.cfg.eta_td3 * expected_td3 + policy.cfg.lambda_bc * expected_bc
    )
    expected_gradient = torch.autograd.grad(expected_total, expected_actor.weight)[0]
    expected_weight = policy.actor_adapt.weight.detach() - 0.1 * expected_gradient
    backward_gradients = []
    policy.actor_adapt.weight.register_hook(
        lambda gradient: backward_gradients.append(gradient.detach().clone())
    )

    metrics = policy._actor_update(batch)

    assert policy.actor_optimizer.step_calls == 1
    assert policy.actor_optimizer.zero_grad_calls == 1
    assert len(backward_gradients) == 1
    assert torch.allclose(backward_gradients[0], expected_gradient)
    assert torch.allclose(policy.actor_adapt.weight, expected_weight)
    assert metrics["weighted_td3_actor_loss"].item() == pytest.approx(
        policy.cfg.eta_td3 * metrics["td3_actor_loss"].item()
    )
    assert metrics["weighted_bc_loss"].item() == pytest.approx(
        policy.cfg.lambda_bc * metrics["exact_bc_loss"].item()
    )
    assert metrics["total_actor_loss"].item() == pytest.approx(
        metrics["weighted_td3_actor_loss"].item() + metrics["weighted_bc_loss"].item()
    )
    assert policy.actor_update_count == 1
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())


def test_exact_teacher_bc_matches_authoritative_latent_huber_computation():
    prediction_latent = nn.Parameter(
        torch.tensor([[0.20, -0.40], [99.0, 99.0], [-0.80, 0.60], [99.0, 99.0]])
    )
    teacher_action = torch.tensor(
        [[1.40, -2.60], [float("nan"), float("nan")], [0.20, 0.40], [9.0, 9.0]]
    )
    valid = torch.tensor([True, False, True, False])
    center = torch.tensor([1.0, -1.0])
    scale = torch.tensor([2.0, 4.0])
    huber_delta = 0.35

    actual = _exact_teacher_bc_loss(
        prediction_latent,
        teacher_action,
        valid,
        center,
        scale,
        huber_delta,
    )
    target_latent = _fastsac_action_center_to_latent(
        teacher_action[valid],
        scale,
        center,
        FASTSAC_REFERENCE_EPS,
    ).detach()
    expected = F.smooth_l1_loss(
        prediction_latent[valid], target_latent, beta=huber_delta
    )

    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.equal(
        prediction_latent.grad[~valid],
        torch.zeros_like(prediction_latent.grad[~valid]),
    )


def test_polyak_update_is_in_place_and_numerically_exact():
    source = nn.Linear(2, 1)
    target = copy.deepcopy(source).requires_grad_(False)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(4.0)
        target.weight.fill_(-2.0)
        target.bias.fill_(-4.0)
    source_before = copy.deepcopy(source.state_dict())

    _polyak_update_(target, source, tau=0.25)

    assert torch.equal(target.weight, torch.full_like(target.weight, -1.0))
    assert torch.equal(target.bias, torch.full_like(target.bias, -2.0))
    assert all(
        torch.equal(source.state_dict()[key], value)
        for key, value in source_before.items()
    )
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_actor_and_both_targets_update_only_on_policy_delay():
    policy = _bare_policy(policy_delay=2, q_tau=0.25)
    policy.actor_adapt = nn.Linear(1, 1)
    policy.actor_target = copy.deepcopy(policy.actor_adapt).requires_grad_(False)
    policy.qnet = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    with torch.no_grad():
        for parameter in policy.actor_adapt.parameters():
            parameter.fill_(2.0)
        for parameter in policy.actor_target.parameters():
            parameter.fill_(-2.0)
        for parameter in policy.qnet.parameters():
            parameter.fill_(4.0)
        for parameter in policy.qnet_target.parameters():
            parameter.zero_()
    policy.actor_update_count = 0

    def actor_update(owner, batch):
        del batch
        owner.actor_update_count += 1
        return {"updated": torch.tensor(1.0)}

    policy._actor_update = MethodType(actor_update, policy)
    actor_target_before = copy.deepcopy(policy.actor_target.state_dict())
    q_target_before = copy.deepcopy(policy.qnet_target.state_dict())
    policy.critic_update_count = 1

    assert policy._maybe_delayed_actor_and_targets({}) is None
    assert policy.actor_update_count == 0
    assert all(
        torch.equal(policy.actor_target.state_dict()[key], value)
        for key, value in actor_target_before.items()
    )
    assert all(
        torch.equal(policy.qnet_target.state_dict()[key], value)
        for key, value in q_target_before.items()
    )

    policy.critic_update_count = 2
    metrics = policy._maybe_delayed_actor_and_targets({})

    assert metrics["updated"].item() == pytest.approx(1.0)
    assert policy.actor_update_count == 1
    assert all(
        torch.equal(parameter, torch.full_like(parameter, -1.0))
        for parameter in policy.actor_target.parameters()
    )
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 1.0))
        for parameter in policy.qnet_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in policy.actor_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in policy.qnet_target.parameters()
    )


def test_collector_noise_is_q_coordinate_noise_on_student_rows_only():
    student = torch.tensor([[1.0, -1.0], [3.0, 3.0], [-1.0, -5.0]])
    teacher = torch.tensor([[8.0, 7.0], [6.0, 5.0], [4.0, 3.0]])
    teacher_before = teacher.clone()
    student_selected = torch.tensor([False, True, False])
    center = torch.tensor([1.0, -1.0])
    scale = torch.tensor([2.0, 4.0])
    gain = 0.5
    collector_rng = torch.Generator().manual_seed(123)
    target_rng = torch.Generator().manual_seed(456)
    target_rng_before = target_rng.get_state().clone()

    issued, exploratory_student, applied_q_noise = _apply_student_collector_noise(
        student_action=student,
        teacher_action=teacher,
        student_selected_mask=student_selected,
        noise_std=0.4,
        noise_clip=0.2,
        q_action_center=center,
        q_action_scale=scale,
        q_action_gain=gain,
        action_low=torch.tensor([-10.0, -10.0]),
        action_high=torch.tensor([10.0, 10.0]),
        generator=collector_rng,
        project_fn=lambda value: value,
    )

    assert torch.equal(teacher, teacher_before)
    assert torch.equal(issued[~student_selected], teacher[~student_selected])
    assert torch.equal(
        exploratory_student[~student_selected], student[~student_selected]
    )
    assert torch.equal(
        applied_q_noise[~student_selected],
        torch.zeros_like(applied_q_noise[~student_selected]),
    )
    assert applied_q_noise[student_selected].abs().max().item() <= 0.2

    nominal_q = (student - center) / scale * gain
    exploratory_q = (exploratory_student - center) / scale * gain
    assert torch.allclose(
        exploratory_q[student_selected] - nominal_q[student_selected],
        applied_q_noise[student_selected],
    )
    assert torch.equal(issued[student_selected], exploratory_student[student_selected])
    assert torch.equal(target_rng.get_state(), target_rng_before)


def test_disabled_collector_noise_is_bitwise_noop_and_does_not_advance_rng():
    student = torch.tensor([[1.0, -1.0], [3.0, 3.0]])
    teacher = torch.tensor([[8.0, 7.0], [6.0, 5.0]])
    selected = torch.tensor([True, False])
    generator = torch.Generator().manual_seed(123)
    generator_before = generator.get_state().clone()

    issued, exploratory_student, applied_q_noise = _apply_student_collector_noise(
        student_action=student,
        teacher_action=teacher,
        student_selected_mask=selected,
        noise_std=0.0,
        noise_clip=0.2,
        q_action_center=torch.tensor([1.0, -1.0]),
        q_action_scale=torch.tensor([2.0, 4.0]),
        q_action_gain=0.5,
        action_low=torch.tensor([-10.0, -10.0]),
        action_high=torch.tensor([10.0, 10.0]),
        generator=generator,
    )

    assert torch.equal(exploratory_student, student)
    assert torch.equal(issued[selected], student[selected])
    assert torch.equal(issued[~selected], teacher[~selected])
    assert torch.equal(applied_q_noise, torch.zeros_like(student))
    assert torch.equal(generator.get_state(), generator_before)


def test_eta_zero_and_disabled_noise_preserve_seeded_baseline_dagger_actions():
    latent = torch.tensor([[0.25, -0.5], [0.75, 0.1], [-0.2, 0.4], [0.0, -0.8]])
    teacher = torch.tensor([[2.0, -3.0], [4.0, 5.0], [-6.0, 7.0], [8.0, -9.0]])
    cfg = SimpleNamespace(
        eta_td3=0.0,
        lambda_bc=1.0,
        target_policy_noise_std=0.0,
        target_policy_noise_clip=0.0,
        collector_exploration_noise_std=0.0,
        collector_exploration_noise_clip=0.0,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
        q_action_input_gain=1.0,
    )

    def owner(seed: int):
        value = SimpleNamespace(cfg=cfg)
        value.dagger_rng = torch.Generator().manual_seed(seed)
        value.collector_exploration_rng = torch.Generator().manual_seed(seed + 1)
        value._student_latent = lambda td: latent.clone()
        value._teacher_action = lambda td: teacher.clone()
        value._project_execution_action = lambda action: action.clamp(-20.0, 20.0)
        value._student_action_from_latent = lambda value: value.tanh() * 20.0
        value._effective_control_mode = lambda: "beta"
        value._teacher_mixture_probability = lambda: 0.5
        value._fastsac_action_low = torch.full((2,), -20.0)
        value._fastsac_action_high = torch.full((2,), 20.0)
        value._fastsac_q_action_center = torch.zeros(2)
        value._fastsac_q_action_scale = torch.ones(2)
        return value

    baseline_owner = owner(123)
    td3_owner = owner(123)
    collector_state = td3_owner.collector_exploration_rng.get_state().clone()
    baseline = _DaggerRolloutPolicy(baseline_owner)
    td3 = _DistributionalTD3DaggerRolloutPolicy(td3_owner)
    rollout = TensorDict({"is_init": torch.zeros(4, dtype=torch.bool)}, batch_size=[4])

    baseline_out = baseline(rollout.clone())
    td3_out = td3(rollout.clone())

    for key in (
        ACTION_KEY,
        DAGGER_TEACHER_ACTION_KEY,
        DAGGER_TEACHER_ACTION_VALID_KEY,
        DAGGER_IS_STUDENT_ACTION_KEY,
        DAGGER_BETA_TEACHER_KEY,
    ):
        assert torch.equal(td3_out[key], baseline_out[key])
    assert torch.equal(
        td3_owner.dagger_rng.get_state(), baseline_owner.dagger_rng.get_state()
    )
    assert torch.equal(td3_owner.collector_exploration_rng.get_state(), collector_state)


class _MeanOnlyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_dist_calls = 0

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluation must not invoke probabilistic forward")

    def get_dist(self, td: TensorDict):
        self.get_dist_calls += 1

        class _Distribution:
            mean = torch.full((*td.batch_size, 1), 0.5)

            def log_prob(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("evaluation must not compute log probability")

        return _Distribution()


def test_evaluation_is_student_only_deterministic_and_noise_free(monkeypatch):
    policy = _bare_policy(dagger_action_clip=1.0, use_object_adapt=False)
    _install_unit_action_contract(policy)
    policy.depth_feature_dim = 1
    policy.adapt_ema = nn.Identity()
    policy.actor_adapt = _MeanOnlyActor()
    policy.collector_exploration_rng = torch.Generator().manual_seed(91)
    collector_state = policy.collector_exploration_rng.get_state().clone()

    def forbidden_teacher(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluation must never evaluate the Teacher")

    policy._teacher_action = forbidden_teacher
    monkeypatch.setattr(
        PPOVEL,
        "get_rollout_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation must bypass probabilistic rollout")
        ),
    )
    evaluation_policy = policy.get_rollout_policy("eval")
    evaluation_policy.eval()

    first = evaluation_policy(TensorDict({}, batch_size=[3]))[ACTION_KEY]
    second = evaluation_policy(TensorDict({}, batch_size=[3]))[ACTION_KEY]

    assert torch.equal(first, second)
    assert torch.allclose(first, torch.full((3, 1), torch.tanh(torch.tensor(0.5))))
    assert torch.equal(policy.collector_exploration_rng.get_state(), collector_state)
    assert policy.actor_adapt.get_dist_calls == 2
    assert policy.actor_adapt.training is False


def _checkpoint_test_policy(seed: int, *, with_actor_target: bool):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = _bare_policy()
        policy.actor_adapt = nn.Linear(2, 1)
        policy.actor_target = (
            copy.deepcopy(policy.actor_adapt).requires_grad_(False)
            if with_actor_target
            else None
        )
        policy.qnet = _TableTwinC51(torch.randn(2, 3), torch.randn(2, 3))
        policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
        policy.adapt_probe = nn.Linear(2, 2)
    policy.actor_optimizer = torch.optim.Adam(
        policy.actor_adapt.parameters(), lr=3.0e-3
    )
    policy.critic_optimizer = torch.optim.Adam(policy.qnet.parameters(), lr=4.0e-3)
    policy.opt_adapt = torch.optim.Adam(policy.adapt_probe.parameters(), lr=5.0e-3)
    policy.actor_update_count = 0
    policy.critic_update_count = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.dagger_rng = torch.Generator().manual_seed(seed + 1)
    policy.q_rng = torch.Generator().manual_seed(seed + 2)
    policy.collector_exploration_rng = torch.Generator().manual_seed(seed + 3)
    policy.target_policy_rng = torch.Generator().manual_seed(seed + 4)
    policy._last_td3_diagnostics = {}
    return policy


def test_td3_checkpoint_seam_round_trips_all_owned_training_state():
    source = _checkpoint_test_policy(100, with_actor_target=True)
    _take_optimizer_step(source.actor_adapt, source.actor_optimizer)
    _take_optimizer_step(source.qnet, source.critic_optimizer)
    _take_optimizer_step(source.adapt_probe, source.opt_adapt)
    source.actor_update_count = 17
    source.critic_update_count = 39
    source.dagger_rollout_count = 11
    source.dagger_environment_steps = 12_345
    source._last_td3_diagnostics = {
        "td3/left_support_projection_clipping_fraction": 0.125,
        "td3/right_support_projection_clipping_fraction": 0.25,
    }
    generators = (
        source.dagger_rng,
        source.q_rng,
        source.collector_exploration_rng,
        source.target_policy_rng,
    )
    for draw_count, generator in enumerate(generators, start=1):
        torch.rand(draw_count, generator=generator)
    evaluation_observation = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
    source_evaluation = torch.tanh(source.actor_adapt(evaluation_observation))
    state = copy.deepcopy(source._td3_checkpoint_state())

    restored = _checkpoint_test_policy(900, with_actor_target=False)
    restored._load_td3_checkpoint_state(state)

    for module_name in ("actor_adapt", "actor_target", "qnet", "qnet_target"):
        _assert_nested_equal(
            getattr(source, module_name).state_dict(),
            getattr(restored, module_name).state_dict(),
        )
    _assert_nested_equal(
        source.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict()
    )
    _assert_nested_equal(
        source.critic_optimizer.state_dict(),
        restored.critic_optimizer.state_dict(),
    )
    _assert_nested_equal(source.opt_adapt.state_dict(), restored.opt_adapt.state_dict())
    assert restored.actor_update_count == 17
    assert restored.critic_update_count == 39
    assert restored.dagger_rollout_count == 11
    assert restored.dagger_environment_steps == 12_345
    assert restored._last_td3_diagnostics == source._last_td3_diagnostics

    for name in (
        "dagger_rng",
        "q_rng",
        "collector_exploration_rng",
        "target_policy_rng",
    ):
        source_generator = getattr(source, name)
        restored_generator = getattr(restored, name)
        assert torch.equal(source_generator.get_state(), restored_generator.get_state())
        assert torch.equal(
            torch.rand(8, generator=source_generator),
            torch.rand(8, generator=restored_generator),
        )

    restored_evaluation = torch.tanh(restored.actor_adapt(evaluation_observation))
    assert torch.equal(restored_evaluation, source_evaluation)
    assert _parameter_storage(restored.actor_adapt).isdisjoint(
        _parameter_storage(restored.actor_target)
    )
    assert _parameter_storage(restored.qnet).isdisjoint(
        _parameter_storage(restored.qnet_target)
    )
    assert all(
        not parameter.requires_grad for parameter in restored.actor_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in restored.qnet_target.parameters()
    )


def test_same_stage_load_rejects_live_vecnorm_fingerprint_mismatch(monkeypatch):
    policy = _bare_policy()
    policy.actor_adapt = nn.Linear(2, 1)
    policy._fastsac_action_contract = {"fingerprint": "sha256:action"}
    policy._replay_vecnorm_fingerprint = "sha256:" + "1" * 64
    policy._checkpoint_config = lambda: {"method": "td3"}
    policy._q_backend_metadata = lambda: {"critic": "c51"}
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": CHECKPOINT_VERSION,
        "actor_backend": ACTOR_BACKEND,
        "critic_learning_semantics": CRITIC_SEMANTICS,
        "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
        "dagger_backend_config": policy._checkpoint_config(),
        "q_backend_config": policy._q_backend_metadata(),
        "action_contract": policy._fastsac_action_contract,
        "vecnorm_fingerprint": "sha256:" + "2" * 64,
    }
    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fingerprint mismatch must fail before module loading")
        ),
    )

    with pytest.raises(ValueError, match="VecNorm fingerprint mismatch"):
        policy.load_state_dict(state)
