import torch
import torch.nn as nn
import pytest

from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACActor,
    FastSACTanhNormal,
)


def test_fastsac_actor_matches_fastsac_architecture_and_initialization():
    actor = FastSACActor(
        input_dim=525,
        action_dim=23,
        hidden_dim=512,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=-torch.ones(23),
        action_high=torch.ones(23),
        layer_norm=True,
    )

    assert isinstance(actor.net[0], nn.Linear)
    assert actor.net[0].in_features == 525 and actor.net[0].out_features == 512
    assert isinstance(actor.net[1], nn.LayerNorm)
    assert isinstance(actor.net[2], nn.SiLU)
    assert actor.net[3].in_features == 512 and actor.net[3].out_features == 256
    assert actor.net[6].in_features == 256 and actor.net[6].out_features == 128
    assert actor.fc_mu[0].in_features == 128 and actor.fc_mu[0].out_features == 23
    assert actor.fc_logstd.in_features == 128 and actor.fc_logstd.out_features == 23
    assert torch.count_nonzero(actor.fc_mu[0].weight) == 0
    assert torch.count_nonzero(actor.fc_mu[0].bias) == 0
    assert torch.count_nonzero(actor.fc_logstd.weight) == 0
    assert torch.count_nonzero(actor.fc_logstd.bias) == 0

    loc, scale, deterministic = actor(torch.randn(4, 525))
    assert loc.shape == scale.shape == deterministic.shape == (4, 23)
    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.allclose(scale, torch.full_like(scale, torch.exp(torch.tensor(-2.5))))
    assert torch.equal(deterministic, torch.zeros_like(deterministic))


def test_fastsac_tanh_normal_is_bounded_reparameterized_and_has_finite_log_prob():
    loc = torch.randn(32, 3, requires_grad=True)
    scale = torch.rand(32, 3).add(0.05)
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([3.0, 1.5, 2.0])
    dist = FastSACTanhNormal(loc, scale, low=low, high=high, event_dims=1)

    action = dist.rsample()
    log_prob = dist.log_prob(action)
    assert action.shape == (32, 3)
    assert log_prob.shape == (32,)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action > low)
    assert torch.all(action < high)
    assert torch.allclose(dist.mean, torch.tanh(loc) * ((high - low) / 2) + ((high + low) / 2))
    assert torch.equal(dist.mode, dist.mean)

    (-log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()


def test_fastsac_actor_state_roundtrip_preserves_student_mu_and_logstd_heads():
    kwargs = dict(
        input_dim=7,
        action_dim=2,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=torch.tensor([-1.0, -2.0]),
        action_high=torch.tensor([1.0, 2.0]),
        layer_norm=True,
    )
    source = FastSACActor(**kwargs)
    with torch.no_grad():
        source.fc_mu[0].weight.fill_(0.125)
        source.fc_logstd.weight.fill_(-0.25)
        source.fc_logstd.bias.copy_(torch.tensor([0.3, -0.4]))

    target = FastSACActor(**kwargs)
    target.load_state_dict(source.state_dict(), strict=True)
    observations = torch.randn(11, 7)
    source_outputs = source(observations)
    target_outputs = target(observations)
    for expected, actual in zip(source_outputs, target_outputs):
        assert torch.equal(expected, actual)


def test_fastsac_tanh_normal_kl_requires_the_same_action_coordinates():
    loc_a = torch.tensor([[0.1, -0.2]])
    loc_b = torch.tensor([[-0.3, 0.4]])
    scale_a = torch.tensor([[0.5, 0.7]])
    scale_b = torch.tensor([[0.8, 0.6]])
    low = torch.tensor([-2.0, -1.0])
    high = torch.tensor([2.0, 1.0])
    p = FastSACTanhNormal(loc_a, scale_a, low=low, high=high, event_dims=1)
    q = FastSACTanhNormal(loc_b, scale_b, low=low, high=high, event_dims=1)

    assert torch.allclose(
        torch.distributions.kl_divergence(p, q),
        torch.distributions.kl_divergence(p.base_dist, q.base_dist),
    )

    incompatible = FastSACTanhNormal(
        loc_b, scale_b, low=low - 1.0, high=high + 1.0, event_dims=1
    )
    with pytest.raises(ValueError, match="identical action bounds"):
        torch.distributions.kl_divergence(p, incompatible)
