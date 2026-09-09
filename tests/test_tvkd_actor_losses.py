from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    DistributionalFastSACTeacherBC,
    _spred_p_teacher_probability,
)
from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC,
)
from test_fastsac_bc_dagger import (
    _ActionSensitiveTwinC51,
    _CountingSGD,
    _install_tiny_stochastic_actor,
    _install_unit_action_contract,
    _tiny_physical_batch,
    _tiny_physical_policy,
)


def _policy(*, policy_type=TVKDDistributionalFastSACTeacherBC, **overrides):
    policy = policy_type.__new__(policy_type)
    nn.Module.__init__(policy)
    cfg = dict(
        eta_sac=0.7,
        lambda_bc=1.3,
        actor_bc_loss_type="mse",
        actor_consistency_coef=0.6,
        actor_gt_bc_coef=1.4,
        dagger_actor_huber_delta=0.4,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
    )
    cfg.update(overrides)
    policy.cfg = SimpleNamespace(**cfg)
    policy.device = torch.device("cpu")
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = _CountingSGD(policy._fastsac_actor_parameters, lr=0.05)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    return policy


def _batch():
    return {
        "observations": torch.tensor([[1.0], [2.0], [-1.0]]),
        "actor_gt_observations": torch.tensor(
            [[3.0], [-2.0], [1.0]], requires_grad=True
        ),
        "critic_observations": torch.ones(3, 1),
        # The third replay row is unlabeled. The first checks tanh-support
        # projection of the frozen Teacher's physical action.
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[2.0], [-0.2], [float("nan")]], requires_grad=True
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
    }


@pytest.mark.parametrize("consistency_coef,gt_bc_coef", [(0.6, 1.4), (0.6, 0), (0, 1.4)])
def test_auxiliary_losses_match_analytic_shared_actor_gradient(
    consistency_coef, gt_bc_coef
):
    policy = _policy(
        actor_consistency_coef=consistency_coef, actor_gt_bc_coef=gt_bc_coef
    )
    # Each row has the same state in both inputs; only its second (latent)
    # feature changes. Two actions with unequal scales check joint averaging.
    policy.actor_adapt = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        policy.actor_adapt.weight.copy_(torch.tensor([[0.2, 0.3], [-0.1, 0.4]]))
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=2, initial_log_std=torch.zeros(2), device="cpu"
    )
    policy._fastsac_q_action_center = torch.tensor([0.2, -0.3])
    policy._fastsac_q_action_scale = torch.tensor([2.0, 0.5])
    predicted_input = torch.tensor([[1.0, 0.5], [2.0, -0.3], [-1.0, 0.7]])
    gt_input = torch.tensor(
        [[1.0, -0.2], [2.0, 0.8], [-1.0, 1.2]], requires_grad=True
    )
    teacher = torch.tensor(
        [[2.0, -0.4], [-0.2, -2.0], [float("nan"), float("nan")]],
        requires_grad=True,
    )
    batch = {
        "actor_gt_observations": gt_input,
        DAGGER_REPLAY_TEACHER_ACTIONS: teacher,
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
    }
    prediction = policy._actor_dist_from_flat(predicted_input).mean
    with torch.no_grad():
        p = torch.tanh(predicted_input @ policy.actor_adapt.weight.T)
        g = torch.tanh(gt_input @ policy.actor_adapt.weight.T)
        target = teacher[:2].clamp(-1.0, 1.0)
        scale = policy._fastsac_q_action_scale
        consistency = ((p - g) / scale).square().mean()
        gt_bc = ((g[:2] - target) / scale).square().mean()
        # Only dp/dW belongs to consistency. GT-BC supplies dg/dW separately.
        expected_gradient = (
            2 * consistency_coef / p.numel()
            * ((p - g) / scale.square() * (1 - p.square())).T
            @ predicted_input
        ) + (
            2 * gt_bc_coef / g[:2].numel()
            * ((g[:2] - target) / scale.square() * (1 - g[:2].square())).T
            @ gt_input[:2]
        )
    rng_before = torch.get_rng_state().clone()
    sac_rng_before = policy.sac_action_rng.get_state().clone()

    loss, metrics = policy._actor_auxiliary_loss(batch, prediction)
    loss.backward()

    assert loss.item() == pytest.approx(
        consistency_coef * consistency.item() + gt_bc_coef * gt_bc.item()
    )
    assert torch.allclose(policy.actor_adapt.weight.grad, expected_gradient)
    assert teacher.grad is None
    assert gt_input.grad is None
    assert policy.bc_dagger_sac_adapter.log_std.grad is None
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(policy.sac_action_rng.get_state(), sac_rng_before)
    for name, expected in {
        "actor_consistency_loss": consistency.item() if consistency_coef else 0.0,
        "weighted_actor_consistency_loss": consistency_coef * consistency.item(),
        "actor_gt_bc_loss": gt_bc.item() if gt_bc_coef else 0.0,
        "weighted_actor_gt_bc_loss": gt_bc_coef * gt_bc.item(),
    }.items():
        assert metrics[name].item() == pytest.approx(expected)
        assert not metrics[name].requires_grad


def test_unlabeled_rows_keep_consistency_and_have_zero_gt_bc():
    policy = _policy()
    batch = _batch()
    batch[DAGGER_REPLAY_TEACHER_ACTIONS] = torch.full(
        (3, 1), float("nan"), requires_grad=True
    )
    batch[DAGGER_TEACHER_ACTION_VALID_KEY].zero_()
    p = policy._actor_dist_from_flat(batch["observations"]).mean
    with torch.no_grad():
        g = torch.tanh(batch["actor_gt_observations"] * 0.25)
        expected = policy.cfg.actor_consistency_coef * (p - g).square().mean()

    loss, metrics = policy._actor_auxiliary_loss(batch, p)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["actor_consistency_loss"].item() > 0.0
    assert metrics["actor_gt_bc_loss"].item() == 0.0
    assert metrics["weighted_actor_gt_bc_loss"].item() == 0.0
    assert torch.isfinite(policy.actor_adapt.weight.grad).all()
    assert policy.actor_adapt.weight.grad.abs().sum().item() > 0.0
    assert batch["actor_gt_observations"].grad is None
    assert batch[DAGGER_REPLAY_TEACHER_ACTIONS].grad is None


def test_actor_update_combines_sac_and_all_three_mean_losses_in_one_step():
    policy = _policy()
    batch = _batch()
    actor = copy.deepcopy(policy.actor_adapt)
    adapter = copy.deepcopy(policy.bc_dagger_sac_adapter)
    critic = copy.deepcopy(policy.qnet)
    expected_rng = torch.Generator().set_state(policy.sac_action_rng.get_state())
    mean = actor(batch["observations"])
    dist = FastSACTanhNormal(
        mean,
        policy._bounded_log_std(adapter.log_std).exp().expand_as(mean),
        low=torch.tensor([-1.0]),
        high=torch.tensor([1.0]),
        event_dims=1,
    )
    action, log_prob = dist.rsample_with_log_prob(generator=expected_rng)
    expected_sac = (
        policy.log_alpha.detach().exp() * log_prob
        - critic.values(critic(batch["critic_observations"], action)).min(dim=0).values
    ).mean()
    p = dist.mean
    g = torch.tanh(actor(batch["actor_gt_observations"].detach()))
    teacher = batch[DAGGER_REPLAY_TEACHER_ACTIONS][:2].detach().clamp(-1.0, 1.0)
    expected_bc = (p[:2] - teacher).square().mean()
    expected_consistency = (p - g.detach()).square().mean()
    expected_gt_bc = (g[:2] - teacher).square().mean()
    expected_total = (
        policy.cfg.eta_sac * expected_sac
        + policy.cfg.lambda_bc * expected_bc
        + policy.cfg.actor_consistency_coef * expected_consistency
        + policy.cfg.actor_gt_bc_coef * expected_gt_bc
    )
    expected_gradients = torch.autograd.grad(
        expected_total, (actor.weight, adapter.log_std)
    )
    weight_before = policy.actor_adapt.weight.detach().clone()
    std_before = policy.bc_dagger_sac_adapter.log_std.detach().clone()

    metrics = policy._actor_update(batch)

    assert policy.actor_optimizer.step_calls == 1
    assert policy.critic_optimizer.step_calls == 0
    assert policy.actor_update_count == 1
    assert torch.allclose(
        policy.actor_adapt.weight, weight_before - 0.05 * expected_gradients[0]
    )
    assert torch.allclose(
        policy.bc_dagger_sac_adapter.log_std, std_before - 0.05 * expected_gradients[1]
    )
    for name, expected in {
        "sac_actor_loss": expected_sac,
        "exact_bc_loss": expected_bc,
        "actor_consistency_loss": expected_consistency,
        "actor_gt_bc_loss": expected_gt_bc,
        "total_actor_loss": expected_total,
    }.items():
        assert metrics[name].item() == pytest.approx(expected.item())
        assert not metrics[name].requires_grad
    assert batch["actor_gt_observations"].grad is None
    assert batch[DAGGER_REPLAY_TEACHER_ACTIONS].grad is None
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())
    assert all(parameter.requires_grad for parameter in policy.qnet.parameters())
    assert policy.log_alpha.grad is None
    assert torch.equal(policy.sac_action_rng.get_state(), expected_rng.get_state())


def test_disabled_auxiliary_losses_preserve_baseline_step_without_gt_input():
    policy = _policy(actor_consistency_coef=0.0, actor_gt_bc_coef=0.0)
    baseline = _policy(policy_type=DistributionalFastSACTeacherBC)
    del policy.cfg.actor_bc_loss_type
    del baseline.cfg.actor_bc_loss_type
    batch = _batch()
    del batch["actor_gt_observations"]

    def unexpected_gt_forward(*args, **kwargs):
        raise AssertionError("Disabled auxiliary losses must skip the GT forward")

    policy._actor_dist_from_flat = unexpected_gt_forward
    assert policy._actor_auxiliary_loss(batch, torch.zeros(3, 1)) == (None, {})

    actual_metrics = policy._actor_update(batch)
    expected_metrics = baseline._actor_update(batch)

    assert actual_metrics.keys() == expected_metrics.keys()
    for name in actual_metrics:
        assert torch.equal(actual_metrics[name], expected_metrics[name]), name
    assert torch.equal(policy.actor_adapt.weight, baseline.actor_adapt.weight)
    assert torch.equal(
        policy.bc_dagger_sac_adapter.log_std, baseline.bc_dagger_sac_adapter.log_std
    )
    assert torch.equal(policy.sac_action_rng.get_state(), baseline.sac_action_rng.get_state())


def test_physical_actor_auxiliary_losses_change_mean_but_preserve_sac_std_update():
    baseline = _tiny_physical_policy(action_dim=2)
    baseline.cfg.lambda_bc = 0.0
    baseline.cfg.actor_bc_loss_type = "mse"
    baseline.cfg.sac_max_grad_norm = 1.0e6
    baseline._fastsac_q_action_scale = torch.tensor([2.0, 0.5])
    with torch.no_grad():
        baseline._ppo_actor_std_parameter().fill_(0.3)
    policy = copy.deepcopy(baseline)
    policy.__class__ = TVKDDistributionalFastSACTeacherBC
    # Optimizer deepcopy preserves parameter ownership but omits the test
    # subclasses' counters, which are outside PyTorch's optimizer state.
    for optimizer in (policy.actor_optimizer, policy.actor_std_optimizer, policy.critic_optimizer):
        optimizer.step_calls = 0
    policy.cfg.lambda_bc = 1.3
    policy.cfg.actor_consistency_coef = 0.6
    policy.cfg.actor_gt_bc_coef = 1.4
    batch = _tiny_physical_batch(action_dim=2)
    batch["actor_gt_observations"] = torch.tensor(
        [[3.0], [-2.0], [1.0]], requires_grad=True
    )
    # Physical Gaussian means retain Teacher commands beyond the nominal Q
    # scale; they must not inherit the tanh Actor's target projection.
    batch[DAGGER_REPLAY_TEACHER_ACTIONS] = torch.tensor(
        [[2.5, -0.4], [-0.2, -2.5], [float("nan"), float("nan")]],
        requires_grad=True,
    )
    p = policy.actor_adapt(batch["observations"])
    g = policy.actor_adapt(batch["actor_gt_observations"].detach())
    target = batch[DAGGER_REPLAY_TEACHER_ACTIONS][:2].detach()
    scale = policy._fastsac_q_action_scale
    bc = ((p[:2] - target) / scale).square().mean()
    consistency = ((p - g.detach()) / scale).square().mean()
    gt_bc = ((g[:2] - target) / scale).square().mean()
    expected_mean_loss = (
        policy.cfg.lambda_bc * bc
        + policy.cfg.actor_consistency_coef * consistency
        + policy.cfg.actor_gt_bc_coef * gt_bc
    )
    expected_mean_gradient = torch.autograd.grad(
        expected_mean_loss, policy.actor_adapt.mean_weight
    )[0]
    std_before = policy._ppo_actor_std_parameter().detach().clone()

    baseline_metrics = baseline._actor_update(batch)
    metrics = policy._actor_update(batch)

    assert policy.actor_optimizer.step_calls == 1
    assert policy.actor_std_optimizer.step_calls == 1
    assert policy.actor_update_count == policy.actor_std_update_count == 1
    assert torch.allclose(
        policy.actor_adapt.mean_weight,
        baseline.actor_adapt.mean_weight - 0.05 * expected_mean_gradient,
    )
    assert not torch.equal(policy.actor_adapt.mean_weight, baseline.actor_adapt.mean_weight)
    assert torch.equal(policy._ppo_actor_std_parameter(), baseline._ppo_actor_std_parameter())
    assert not torch.equal(policy._ppo_actor_std_parameter(), std_before)
    assert torch.equal(
        metrics["actor_std_sac_grad_norm"], baseline_metrics["actor_std_sac_grad_norm"]
    )
    assert torch.equal(metrics["sac_actor_loss"], baseline_metrics["sac_actor_loss"])
    assert torch.equal(policy.sac_action_rng.get_state(), baseline.sac_action_rng.get_state())
    for name, expected in {
        "exact_bc_loss": bc,
        "weighted_bc_loss": policy.cfg.lambda_bc * bc,
        "actor_consistency_loss": consistency,
        "weighted_actor_consistency_loss": policy.cfg.actor_consistency_coef * consistency,
        "actor_gt_bc_loss": gt_bc,
        "weighted_actor_gt_bc_loss": policy.cfg.actor_gt_bc_coef * gt_bc,
        "total_actor_loss": (
            policy.cfg.eta_sac * baseline_metrics["sac_actor_loss"] + expected_mean_loss
        ),
    }.items():
        assert metrics[name].item() == pytest.approx(expected.item())
        assert not metrics[name].requires_grad
    assert batch["actor_gt_observations"].grad is None
    assert batch[DAGGER_REPLAY_TEACHER_ACTIONS].grad is None
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())


def test_q_filtered_bc_applies_mse_and_detached_teacher_probability():
    policy = _policy(
        use_q_filtered_bc=True, eta_sac=0.0, lambda_bc=1.0,
        actor_consistency_coef=0.0, actor_gt_bc_coef=0.0,
    )
    batch = _batch()
    p = policy._actor_dist_from_flat(batch["observations"]).mean
    valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY]
    with torch.no_grad():
        target = batch[DAGGER_REPLAY_TEACHER_ACTIONS].clamp(-1.0, 1.0)
        target[~valid] = p[~valid]
        policy_q = policy.qnet.values(policy.qnet(batch["critic_observations"], p))
        teacher_q = policy.qnet.values(policy.qnet(batch["critic_observations"], target))
        probability, _, _ = _spred_p_teacher_probability(policy_q, teacher_q)
    expected_bc = (
        (p[valid] - target[valid]).square().mean(dim=1) * probability[valid]
    ).mean()
    expected_gradient = torch.autograd.grad(expected_bc, policy.actor_adapt.weight)[0]
    weight_before = policy.actor_adapt.weight.detach().clone()

    metrics = policy._actor_update(batch)

    assert metrics["exact_bc_loss"].item() == pytest.approx(expected_bc.item())
    assert torch.allclose(policy.actor_adapt.weight, weight_before - 0.05 * expected_gradient)
    assert batch[DAGGER_REPLAY_TEACHER_ACTIONS].grad is None
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())
