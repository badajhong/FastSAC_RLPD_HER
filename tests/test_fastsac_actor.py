import functools
from types import SimpleNamespace

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from active_adaptation.learning.ppo.common import ACTION_KEY, OBS_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACActor,
    FastSACTanhNormal,
    FastSACVEL,
)
from active_adaptation.learning.ppo.ppo_vel import PRIV_FEATURE_KEY, PRIV_PRED_KEY


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


def test_fastsac_actor_log_std_reset_inverse_maps_exact_targets_only():
    actor = FastSACActor(
        input_dim=7,
        action_dim=3,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=-torch.ones(3),
        action_high=torch.ones(3),
        layer_norm=True,
    )
    preserved = {
        name: parameter.detach().clone()
        for name, parameter in actor.named_parameters()
        if not name.startswith("fc_logstd.")
    }
    with torch.no_grad():
        actor.fc_logstd.weight.fill_(0.75)
        actor.fc_logstd.bias.fill_(-0.25)

    raw_bias = actor.reset_log_std_head(-1.5)

    assert abs(raw_bias - 0.42364893019360184) < 1e-12
    assert torch.count_nonzero(actor.fc_logstd.weight) == 0
    assert torch.allclose(
        actor.fc_logstd.bias,
        torch.full_like(actor.fc_logstd.bias, 0.42364893),
    )
    _, scale, _ = actor(torch.randn(5, 7))
    assert torch.allclose(scale, torch.full_like(scale, torch.exp(torch.tensor(-1.5))))
    for name, parameter in actor.named_parameters():
        if name in preserved:
            assert torch.equal(parameter, preserved[name])

    raw_bias = actor.reset_log_std_head(-2.5)

    assert raw_bias == 0.0
    assert torch.count_nonzero(actor.fc_logstd.weight) == 0
    assert torch.count_nonzero(actor.fc_logstd.bias) == 0
    _, scale, _ = actor(torch.randn(5, 7))
    assert torch.allclose(scale, torch.full_like(scale, torch.exp(torch.tensor(-2.5))))


def test_fastsac_tanh_normal_is_bounded_reparameterized_and_has_finite_log_prob():
    loc = torch.randn(32, 3, requires_grad=True)
    scale = torch.rand(32, 3).add(0.05)
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([3.0, 1.5, 2.0])
    dist = FastSACTanhNormal(loc, scale, low=low, high=high, event_dims=1)

    action, log_prob = dist.rsample_with_log_prob()
    assert action.shape == (32, 3)
    assert log_prob.shape == (32,)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action > low)
    assert torch.all(action < high)
    assert torch.allclose(dist.mean, torch.tanh(loc) * ((high - low) / 2) + ((high + low) / 2))
    assert torch.equal(dist.mode, dist.mean)

    (-log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()


def test_fastsac_training_generator_does_not_advance_global_rollout_rng():
    dist = FastSACTanhNormal(
        torch.zeros(4, 2),
        torch.full((4, 2), 0.2),
        low=-torch.ones(2),
        high=torch.ones(2),
        event_dims=1,
    )
    sac_rng = torch.Generator().manual_seed(11)
    global_before = torch.random.get_rng_state().clone()

    first_action, first_log_prob = dist.rsample_with_log_prob(
        generator=sac_rng
    )

    assert torch.equal(torch.random.get_rng_state(), global_before)
    replayed_rng = torch.Generator().manual_seed(11)
    second_action, second_log_prob = dist.rsample_with_log_prob(
        generator=replayed_rng
    )
    assert torch.equal(first_action, second_action)
    assert torch.equal(first_log_prob, second_log_prob)


def test_fastsac_pre_tanh_log_prob_stays_finite_with_recovery_gradient_at_saturation():
    loc = torch.tensor([[20.0, -20.0, 100.0]], requires_grad=True)
    scale = torch.full_like(loc, 0.05)
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([3.0, 1.5, 2.0])
    dist = FastSACTanhNormal(loc, scale, low=low, high=high, event_dims=1)

    torch.manual_seed(17)
    action, log_prob = dist.rsample_with_log_prob()

    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.all(action >= low) and torch.all(action <= high)
    (-log_prob.mean()).backward()
    assert torch.isfinite(loc.grad).all()
    assert torch.count_nonzero(loc.grad) == loc.numel()


def test_fastsac_replay_log_prob_matches_transformed_distribution_interior():
    loc = torch.tensor([[0.2, -0.4], [0.7, 0.1]], requires_grad=True)
    scale = torch.full_like(loc, 0.3)
    low = torch.tensor([-2.0, -1.0])
    high = torch.tensor([3.0, 2.0])
    dist = FastSACTanhNormal(
        loc, scale, low=low, high=high, event_dims=1
    )
    action = torch.tensor([[0.5, 0.25], [-0.75, 1.25]])

    stable = dist.log_prob_for_action(action)
    generic = dist.log_prob(action)

    assert stable.shape == (2,)
    assert torch.allclose(stable, generic, atol=2e-6, rtol=2e-6)


def test_fastsac_replay_log_prob_is_finite_at_bounds_and_can_freeze_scale():
    loc = torch.zeros(2, 2, requires_grad=True)
    scale = torch.full((2, 2), 0.2, requires_grad=True)
    low = torch.tensor([-2.0, -1.0])
    high = torch.tensor([3.0, 2.0])
    dist = FastSACTanhNormal(
        loc, scale, low=low, high=high, event_dims=1
    )
    action = torch.stack((low, high))

    log_prob = dist.log_prob_for_action(action, detach_scale=True)
    (-log_prob.mean()).backward()

    assert torch.isfinite(log_prob).all()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is None


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


def test_reference_centered_teacher_zero_head_tracks_interior_reference_exactly():
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([1.0, 3.0, 2.5])
    actor = FastSACActor(
        input_dim=5,
        action_dim=3,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=low,
        action_high=high,
        layer_norm=True,
        reference_centered=True,
    )
    observations = torch.randn(2, 5)
    reference_action = torch.tensor([
        [-1.25, 2.00, 0.25],
        [0.50, -0.25, 2.00],
    ])

    loc, scale, deterministic = actor(
        observations, reference_action=reference_action
    )

    assert torch.isfinite(loc).all()
    assert torch.isfinite(scale).all()
    # The zero-initialized FastSAC mean head represents zero residual in the
    # teacher.  Its absolute action must therefore be the VAIC reference action,
    # even though the executable action interval is asymmetric.
    assert torch.allclose(deterministic, reference_action, atol=1e-6, rtol=1e-6)


def test_fastsac_actor_builder_wires_reference_only_into_teacher():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.action_dim = 3
    policy.cfg = SimpleNamespace(
        fastsac_actor_hidden_dim=16,
        fastsac_log_std_min=-5.0,
        fastsac_log_std_max=0.0,
        fastsac_actor_layer_norm=True,
    )
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([1.0, 3.0, 2.5])
    policy.dist_cls = functools.partial(
        FastSACTanhNormal, low=low, high=high, event_dims=1
    )
    policy.dist_keys = list(FastSACTanhNormal.dist_keys)
    teacher = policy._build_fastsac_actor(
        ["teacher_obs_a", "teacher_obs_b"], 5, low, high,
        reference_key="reference",
    )
    student = policy._build_fastsac_actor(
        ["student_obs_a", "student_obs_b"], 5, low, high
    )
    reference = torch.tensor([[-1.0, 2.0, 0.0], [0.5, -0.5, 2.0]])
    td = TensorDict(
        {
            "teacher_obs_a": torch.randn(2, 3),
            "teacher_obs_b": torch.randn(2, 2),
            "student_obs_a": torch.randn(2, 3),
            "student_obs_b": torch.randn(2, 2),
            "reference": reference,
        },
        batch_size=[2],
    )

    assert torch.allclose(teacher.get_dist(td).mean, reference, atol=1e-6)
    assert torch.allclose(student.get_dist(td).mean, torch.zeros_like(reference))


def test_fastsac_rollout_does_not_select_unavailable_log_probability():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.action_dim = 2
    policy.depth_feature_dim = 2
    policy.cfg = SimpleNamespace(
        phase="train",
        adapt_module="mlp",
        train_dr_estimator=False,
        fastsac_actor_hidden_dim=16,
        fastsac_log_std_min=-5.0,
        fastsac_log_std_max=0.0,
        fastsac_actor_layer_norm=True,
    )
    low = -torch.ones(2)
    high = torch.ones(2)
    policy.dist_cls = functools.partial(
        FastSACTanhNormal, low=low, high=high, event_dims=1
    )
    policy.dist_keys = list(FastSACTanhNormal.dist_keys)
    policy.object_transform = TensorDictModule(
        nn.Identity(), [OBS_KEY], ["object_passthrough"]
    )
    policy.encoder_priv = TensorDictModule(
        nn.Linear(3, 2), [OBS_KEY], [PRIV_FEATURE_KEY]
    )
    policy.actor = policy._build_fastsac_actor(
        [OBS_KEY, PRIV_FEATURE_KEY], 5, low, high
    )
    policy.adapt_module = TensorDictModule(
        nn.Linear(3, 2), [OBS_KEY], [PRIV_PRED_KEY]
    )

    rollout = policy.get_rollout_policy("train")
    result = rollout(
        TensorDict({OBS_KEY: torch.randn(4, 3)}, batch_size=[4])
    )

    assert "sample_log_prob" not in rollout.out_keys
    assert PRIV_PRED_KEY not in rollout.out_keys
    assert PRIV_PRED_KEY not in result.keys(include_nested=True)
    assert ACTION_KEY in result


def test_reference_centered_teacher_is_finite_at_asymmetric_action_boundaries():
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([1.0, 3.0, 2.5])
    actor = FastSACActor(
        input_dim=4,
        action_dim=3,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=low,
        action_high=high,
        layer_norm=True,
        reference_centered=True,
    )
    reference_action = torch.stack((low, high))

    loc, scale, deterministic = actor(
        torch.randn(2, 4), reference_action=reference_action
    )

    assert torch.isfinite(loc).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(deterministic).all()
    assert torch.all(deterministic >= low)
    assert torch.all(deterministic <= high)
    assert torch.allclose(deterministic, reference_action, atol=1e-5, rtol=0.0)


def test_absolute_student_zero_head_uses_default_pose_not_reference():
    low = torch.tensor([-2.0, -1.0, -0.5])
    high = torch.tensor([1.0, 3.0, 2.5])
    actor = FastSACActor(
        input_dim=5,
        action_dim=3,
        hidden_dim=16,
        log_std_min=-5.0,
        log_std_max=0.0,
        action_low=low,
        action_high=high,
        layer_norm=True,
        reference_centered=False,
    )
    observations = torch.randn(2, 5)
    references = torch.tensor([
        [-1.5, 2.5, 0.0],
        [0.5, -0.5, 2.0],
    ])

    without_reference = actor(observations)
    with_reference = actor(observations, reference_action=references)
    expected = torch.zeros_like(without_reference[2])

    # The deploy-facing VAIC student has no reference observation.  It remains
    # an absolute-action actor, but zero raw action is VAIC's safe/default joint
    # pose and must remain the zero-head initialization even for asymmetric
    # bounds.  Supplying an unrelated reference must not change it.
    assert torch.allclose(without_reference[2], expected)
    for actual, ignored_reference_result in zip(without_reference, with_reference):
        assert torch.equal(actual, ignored_reference_result)


def test_vaic_action_bounds_are_asymmetric_executable_raw_joint_limits():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.joint_names = ["joint_0", "joint_1", "joint_2"]

    soft_limits = torch.tensor([[[-1.0, 2.0], [-3.0, 1.0], [0.0, 3.0]]])
    default_joint_pos = torch.tensor([[0.0, -1.0, 0.5]])
    action_scaling = torch.tensor([0.5, 0.25, 2.0])
    manager = SimpleNamespace(
        joint_ids=torch.tensor([0, 1, 2]),
        asset=SimpleNamespace(
            data=SimpleNamespace(soft_joint_pos_limits=soft_limits)
        ),
        default_joint_pos=default_joint_pos,
        action_scaling=action_scaling,
    )
    policy.env = SimpleNamespace(action_manager=manager, randomizations={})

    action_low, action_high = policy._vaic_action_bounds()

    expected_low = torch.tensor([-2.0, -8.0, -0.25])
    expected_high = torch.tensor([4.0, 8.0, 1.25])
    assert torch.equal(action_low, expected_low)
    assert torch.equal(action_high, expected_high)
    assert torch.all(action_low < action_high)
    assert not torch.equal(action_low, -action_high)

    # Mapping either raw boundary through VAIC's unchanged controller reaches
    # the corresponding physical limit exactly, so no hidden clamp aliases two
    # different replay actions to the same joint target.
    default = default_joint_pos[0, manager.joint_ids]
    assert torch.equal(default + action_low * action_scaling, soft_limits[0, :, 0])
    assert torch.equal(default + action_high * action_scaling, soft_limits[0, :, 1])


def test_vaic_nominal_action_coordinates_ignore_episode_random_offset():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.joint_names = ["joint_0", "joint_1", "joint_2"]

    soft_limits = torch.tensor([[[-1.0, 2.0], [-3.0, 1.0], [0.0, 3.0]]])
    default_joint_pos = torch.tensor([[0.0, -1.0, 0.5]])
    action_scaling = torch.tensor([0.5, 0.25, 2.0])
    manager = SimpleNamespace(
        joint_ids=torch.tensor([0, 1, 2]),
        asset=SimpleNamespace(
            data=SimpleNamespace(soft_joint_pos_limits=soft_limits)
        ),
        default_joint_pos=default_joint_pos,
        action_scaling=action_scaling,
    )
    # Match random_joint_offset's real representation: asset joint ids plus a
    # [num_envs, randomized_joints, low/high] configured range. Joint 1 is not
    # randomized and therefore retains a zero offset interval.
    offset_randomization = SimpleNamespace(
        joint_ids=torch.tensor([0, 2]),
        offset_range=torch.tensor([
            [[-0.10, 0.20], [-0.40, 0.30]],
            [[-0.10, 0.20], [-0.40, 0.30]],
        ]),
    )
    policy.env = SimpleNamespace(
        action_manager=manager,
        randomizations={"random_joint_offset": offset_randomization},
    )

    action_low, action_high = policy._vaic_action_bounds()

    expected_low = torch.tensor([-2.0, -8.0, -0.25])
    expected_high = torch.tensor([4.0, 8.0, 1.25])
    assert torch.allclose(action_low, expected_low)
    assert torch.allclose(action_high, expected_high)

    default = default_joint_pos[0, manager.joint_ids]
    offset_low = torch.tensor([-0.10, 0.0, -0.40])
    offset_high = torch.tensor([0.20, 0.0, 0.30])
    lower_targets = default + action_low * action_scaling
    upper_targets = default + action_high * action_scaling
    assert torch.allclose(lower_targets, soft_limits[0, :, 0])
    assert torch.allclose(upper_targets, soft_limits[0, :, 1])

    # Episode offsets are provenance/dynamics, not a moving Q coordinate.
    assert torch.allclose(
        torch.tensor(policy._fastsac_action_contract["joint_offset_low"]),
        offset_low,
    )
    assert torch.allclose(
        torch.tensor(policy._fastsac_action_contract["joint_offset_high"]),
        offset_high,
    )
