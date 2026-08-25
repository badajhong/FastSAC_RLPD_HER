from __future__ import annotations

import copy
import json
import math
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY, Actor
from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
    _fastsac_target_entropy,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from active_adaptation.learning.ppo.td3_bc_dagger import (
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    PRETRAINED_PERCEPTION_MODULES,
    TD3_COLLECTOR_NOISE_KEY,
    TD3_EXPLORATORY_STUDENT_ACTION_KEY,
    TD3_NOISE_FREE_STUDENT_ACTION_KEY,
    DistributionalTD3TeacherBC,
    _failure_lookback_offsets,
    _source_counts,
    _categorical_expected_value,
    _exact_teacher_bc_loss,
    _project_c51_probabilities,
)
from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    ACTOR_BACKEND,
    CHECKPOINT_VERSION,
    FastSACPhysicalNormal,
    NORMALIZED_TANH_ACTION_DISTRIBUTION,
    PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND,
    PREVIOUS_CHECKPOINT_VERSION,
    TRAINING_ALGORITHM,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
    _DeterministicFastSACStudentEvalPolicy,
    _DistributionalFastSACDaggerRolloutPolicy,
)
from active_adaptation.learning.ppo.fastsac_gradient_probe import (
    GRADIENT_PROBE_SCHEMA,
    diagnose_fastsac_actor_gradients,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
)


def _bare_policy(**cfg) -> DistributionalFastSACTeacherBC:
    policy = DistributionalFastSACTeacherBC.__new__(DistributionalFastSACTeacherBC)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    policy.device = torch.device("cpu")
    return policy


class _LatentActor(nn.Module):
    """Tiny BC Actor exposing the same ``get_dist(...).mean`` contract."""

    def __init__(self, weight: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[weight]], dtype=torch.float32))

    def get_dist(self, td: TensorDict):
        return SimpleNamespace(mean=td["actor_input"] @ self.weight)


class _ActionSensitiveC51Head(nn.Module):
    def __init__(self, slope: float):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope)))

    def forward(self, observations, actions):
        del observations
        score = self.slope * actions[:, 0]
        return torch.stack((-score, torch.zeros_like(score), score), dim=-1)


class _ActionSensitiveTwinC51(nn.Module):
    def __init__(self, first_slope: float = 1.0, second_slope: float = -2.0):
        super().__init__()
        self.qnets = nn.ModuleList(
            (
                _ActionSensitiveC51Head(first_slope),
                _ActionSensitiveC51Head(second_slope),
            )
        )
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(head(observations, actions) for head in self.qnets), dim=0
        )

    def values(self, logits):
        return _categorical_expected_value(logits, self.support)


class _TableC51Head(nn.Module):
    def __init__(self, probabilities: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(probabilities.log())

    def forward(self, observations, actions):
        del actions
        return self.logits[: observations.shape[0]]


class _TableTwinC51(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.qnets = nn.ModuleList((_TableC51Head(first), _TableC51Head(second)))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(head(observations, actions) for head in self.qnets), dim=0
        )


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class _CountingAdam(torch.optim.Adam):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _install_unit_action_contract(policy) -> None:
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([1.0])
    policy._fastsac_action_low = torch.tensor([-20.0])
    policy._fastsac_action_high = torch.tensor([20.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([20.0])
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.cfg.action_support_clip = 20.0


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


def _optimizer_step(parameters, optimizer) -> None:
    parameters = tuple(parameters)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()


def test_config_identifies_fastsac_and_locks_all_inherited_td3_noise_off():
    cfg = DistributionalFastSACTeacherBCConfig()

    assert TRAINING_ALGORITHM == "distributional_fastsac_teacher_bc_v1"
    assert cfg._target_.endswith(".DistributionalFastSACTeacherBC")
    assert cfg.name == "fastsac_bc_dagger"
    assert cfg.target_policy_noise_std == 0.0
    assert cfg.target_policy_noise_clip == 0.0
    assert cfg.collector_exploration_noise_std == 0.0
    assert cfg.collector_exploration_noise_clip == 0.0
    assert cfg.eta_td3 == 0.0
    assert cfg.policy_delay == cfg.sac_policy_frequency == 8
    assert cfg.sac_alpha_update_cadence == "actor"
    assert cfg.sac_action_distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION
    assert cfg.sac_physical_std_lr == pytest.approx(1.0e-5)
    assert cfg.sac_physical_std_max_kl == pytest.approx(0.01)
    assert cfg.sac_physical_std_min == pytest.approx(0.05)
    assert cfg.sac_physical_std_max == pytest.approx(0.5)
    assert cfg.sac_target_entropy_ratio == pytest.approx(1.0)
    assert cfg.q_update_to_data_ratio == pytest.approx(1.0)
    assert cfg.perception_encode_microbatch_size == 512
    assert cfg.perception_replay_mode == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    assert cfg.teacher_perception_replay_fraction == 0.0
    assert cfg.teacher_perception_warmup_steps == 0


def test_physical_gaussian_config_allows_autotune_and_requires_ppo_load_scale():
    valid = DistributionalFastSACTeacherBCConfig(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        sac_use_autotune=True,
        load_noise_scale=0.5,
    )
    DistributionalFastSACTeacherBC._validate_td3_config(valid)

    with pytest.raises(ValueError, match="load_noise_scale"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
                sac_use_autotune=False,
                load_noise_scale=None,
            )
        )

    with pytest.raises(ValueError, match="sac_physical_std_max_kl"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
                sac_use_autotune=True,
                load_noise_scale=0.5,
                sac_physical_std_max_kl=0.0,
            )
        )
    with pytest.raises(ValueError, match="smaller"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
                sac_use_autotune=True,
                load_noise_scale=0.5,
                sac_physical_std_min=0.5,
                sac_physical_std_max=0.5,
            )
        )


def test_ppo_finetune_fresh_load_resets_saved_joint_std_to_load_noise_scale():
    actor = Actor(2, init_noise_scale=1.0, load_noise_scale=0.5)
    actor(torch.zeros(1, 3))
    saved = copy.deepcopy(actor.state_dict())
    saved["actor_std"] = torch.tensor([0.17, 0.29])

    actor.load_state_dict(saved, strict=True)

    assert torch.equal(actor.actor_std, torch.tensor([0.5, 0.5]))


def test_config_allows_explicit_pure_sac_ablation_without_inherited_td3_eta():
    cfg = DistributionalFastSACTeacherBCConfig(lambda_bc=0.0, eta_sac=1.0)

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_config_rejects_entropy_target_above_maximum_gaussian_entropy():
    cfg = DistributionalFastSACTeacherBCConfig(sac_log_std_max=-3.0)

    with pytest.raises(ValueError, match="entropy target is unreachable"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_config_rejects_entropy_target_below_minimum_gaussian_entropy():
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_target_entropy_ratio=9.0
    )

    with pytest.raises(ValueError, match="entropy target is unreachable"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    ("ratio", "expected_target"), ((2.0, -46.0), (2.5, -57.5))
)
def test_config_allows_lower_entropy_targets_above_unit_ratio(
    ratio, expected_target
):
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_target_entropy_ratio=ratio
    )

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)

    action_low = torch.full((23,), -1.0)
    action_high = torch.full((23,), 1.0)
    assert _fastsac_target_entropy(action_low, action_high, ratio) == pytest.approx(
        expected_target
    )


@pytest.mark.parametrize("ratio", (0.0, -1.0, math.inf, math.nan))
def test_config_rejects_nonpositive_or_nonfinite_entropy_ratio(ratio):
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_target_entropy_ratio=ratio
    )

    with pytest.raises(ValueError, match="finite and positive"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_config_rejects_unknown_temperature_update_cadence():
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_alpha_update_cadence="rollout"
    )

    with pytest.raises(ValueError, match="sac_alpha_update_cadence.*critic"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    "field",
    (
        "teacher_actor_replay_fraction",
        "q_teacher_replay_ratio",
    ),
)
@pytest.mark.parametrize("fraction", (0.0, 0.1, 0.5, 1.0))
def test_backend_accepts_configurable_teacher_source_fractions(field, fraction):
    cfg = DistributionalFastSACTeacherBCConfig()
    setattr(cfg, field, fraction)

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize("fraction", (0.1, 0.5, 1.0))
def test_backend_rejects_teacher_perception_replay(fraction):
    cfg = DistributionalFastSACTeacherBCConfig(
        teacher_perception_replay_fraction=fraction
    )

    with pytest.raises(ValueError, match="teacher_perception_replay_fraction=0"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize("mode", ("legacy_online_student", "four_way"))
def test_backend_locks_perception_to_live_student_rollout(mode):
    cfg = DistributionalFastSACTeacherBCConfig(perception_replay_mode=mode)

    with pytest.raises(ValueError, match="online_student_rollout|four_way"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dagger_control_mode", "safe"),
        ("dagger_control_mode", "hybrid"),
        ("dagger_beta_start", 0.1),
        ("dagger_beta_end", 0.1),
    ),
)
def test_backend_locks_live_perception_to_pure_student_control(field, value):
    cfg = DistributionalFastSACTeacherBCConfig()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match="beta"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_backend_locks_row_level_q_utd_to_one():
    cfg = DistributionalFastSACTeacherBCConfig(q_update_to_data_ratio=0.5)

    with pytest.raises(ValueError, match="row-level Q UTD=1"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_tanh_normal_is_bounded_reparameterized_and_has_exact_log_prob():
    loc = torch.tensor([[0.0, 1.25], [-1.55, 0.05]], requires_grad=True)
    scale = torch.tensor([[0.01, 0.02], [0.01, 0.02]], requires_grad=True)
    dist = FastSACTanhNormal(
        loc,
        scale,
        low=torch.tensor([-20.0, -20.0]),
        high=torch.tensor([20.0, 20.0]),
        event_dims=1,
    )

    first, first_log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(71)
    )
    second, second_log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(72)
    )
    assert not torch.equal(first, second)
    assert torch.isfinite(first_log_prob).all()
    assert torch.isfinite(second_log_prob).all()
    assert torch.all(first >= -20.0) and torch.all(first <= 20.0)
    assert torch.allclose(
        first_log_prob, dist.log_prob_for_action(first), rtol=2e-5, atol=2e-5
    )
    (first.sum() - first_log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_backend_distribution_maps_normalized_std_to_joint_scaled_bounded_policy():
    policy = _bare_policy(
        sac_log_std_min=-8.0, sac_log_std_max=-1.0, action_support_clip=20.0
    )
    expected_log_std = torch.log(torch.tensor([0.1, 0.2]))
    initial_raw_log_std = policy._inverse_smooth_log_std(
        expected_log_std,
        policy.cfg.sac_log_std_min,
        policy.cfg.sac_log_std_max,
    )
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=2,
        initial_log_std=initial_raw_log_std,
        device="cpu",
    )
    policy._fastsac_q_action_scale = torch.tensor([2.0, 4.0])
    policy._fastsac_action_low = torch.tensor([-20.0, -20.0])
    policy._fastsac_action_high = torch.tensor([20.0, 20.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0, 0.0])
    policy._fastsac_actor_action_scale = torch.tensor([20.0, 20.0])
    raw_mean = torch.tensor([[0.0, 25.0], [-31.0, 1.0]])

    dist = policy._sac_dist_from_mean(raw_mean)

    assert torch.allclose(dist.loc, raw_mean / 20.0)
    assert torch.allclose(
        dist.scale,
        (
            expected_log_std.exp() * torch.tensor([2.0, 4.0]) / 20.0
        ).expand_as(raw_mean),
    )
    assert torch.allclose(dist.mean, 20.0 * torch.tanh(raw_mean / 20.0))
    assert torch.all(dist.mean >= -20.0) and torch.all(dist.mean <= 20.0)


def _install_ppo_physical_std(policy, values=(0.5, 0.5)):
    policy.actor_adapt = nn.Sequential(
        Actor(
            len(values),
            init_noise_scale=1.0,
            load_noise_scale=0.5,
        )
    )
    with torch.no_grad():
        policy._ppo_actor_std_parameter().copy_(torch.tensor(values))
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        len(values), torch.zeros(len(values)), "cpu"
    )


def test_ppo_physical_gaussian_matches_independent_normal_without_tanh_or_q_offset():
    policy = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    )
    _install_ppo_physical_std(policy, values=(0.2, 0.7))
    policy._fastsac_entropy_reference_log_scale_sum = math.log(37.0)
    raw_mean = torch.tensor([[2.0, -3.0], [4.0, 1.0]], requires_grad=True)

    dist = policy._sac_dist_from_mean(raw_mean)
    assert isinstance(dist, FastSACPhysicalNormal)
    assert torch.equal(dist.mean, raw_mean)
    # The physical Normal remains unsquashed, but its direct PPOVEL std is now
    # defensively bounded before sampling.
    assert torch.allclose(dist.scale, torch.tensor([[0.2, 0.5], [0.2, 0.5]]))

    sample_generator = torch.Generator().manual_seed(91)
    expected_generator = torch.Generator().manual_seed(91)
    sample, log_prob = dist.rsample_with_log_prob(generator=sample_generator)
    expected_noise = torch.randn(raw_mean.shape, generator=expected_generator)
    expected_sample = raw_mean + dist.scale * expected_noise
    expected_dist = torch.distributions.Independent(
        torch.distributions.Normal(raw_mean, dist.scale), 1
    )
    assert torch.allclose(sample, expected_sample)
    assert torch.allclose(log_prob, expected_dist.log_prob(sample))
    assert torch.allclose(
        policy._normalized_action_log_prob(log_prob), log_prob + math.log(37.0)
    )

    (sample.sum() - log_prob.mean()).backward()
    assert raw_mean.grad is not None and torch.isfinite(raw_mean.grad).all()
    assert policy._ppo_actor_std_parameter().grad is not None


def test_physical_mode_owns_trainable_ppo_std_and_freezes_tanh_adapter():
    physical = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    )
    _install_ppo_physical_std(physical)
    physical._configure_training_actor_std()
    assert physical._ppo_actor_std_parameter().requires_grad
    assert not any(
        parameter.requires_grad
        for parameter in physical.bc_dagger_sac_adapter.parameters()
    )

    normalized = _bare_policy(
        sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION
    )
    _install_ppo_physical_std(normalized)
    normalized._configure_training_actor_std()
    assert not normalized._ppo_actor_std_parameter().requires_grad
    assert all(
        parameter.requires_grad
        for parameter in normalized.bc_dagger_sac_adapter.parameters()
    )


def test_physical_checkpoint_restore_preserves_learned_joint_std_after_ppo_reset():
    policy = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    )
    _install_ppo_physical_std(policy, values=(0.5, 0.5))
    saved = {"module.actor_std": torch.tensor([0.17, 0.29])}

    policy._restore_checkpoint_physical_actor_std(
        saved, context="unit checkpoint actor_adapt"
    )

    assert torch.equal(
        policy._ppo_actor_std_parameter(), torch.tensor([0.17, 0.29])
    )


def test_physical_deterministic_eval_uses_raw_mean_before_only_safety_projection():
    policy = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    )
    raw_mean = torch.tensor([[2.0, 25.0]])
    policy._student_raw_action_proposal = lambda td: raw_mean.clone()
    policy._project_execution_action = lambda action: action.clamp(-20.0, 20.0)

    action = policy._student_mean_action(TensorDict({}, batch_size=[1]))

    assert torch.equal(action, torch.tensor([[2.0, 20.0]]))
    assert action[0, 0].item() != pytest.approx(
        20.0 * math.tanh(raw_mean[0, 0].item() / 20.0)
    )


def test_smooth_log_std_stays_bounded_and_keeps_gradient_past_old_hard_cap():
    policy = _bare_policy(
        sac_log_std_min=-8.0, sac_log_std_max=-1.0, action_support_clip=20.0
    )
    # A value this far above the old direct-log-std cap produced exactly zero
    # gradient through Tensor.clamp. It is now an unconstrained coordinate.
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.tensor(3.0),
        device="cpu",
    )
    policy._fastsac_q_action_scale = torch.tensor([1.0])
    policy._fastsac_action_low = torch.tensor([-20.0])
    policy._fastsac_action_high = torch.tensor([20.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([20.0])

    effective_log_std = policy._bounded_log_std()
    dist = policy._sac_dist_from_mean(torch.zeros(2, 1))
    dist.scale.sum().backward()

    assert torch.all(effective_log_std > policy.cfg.sac_log_std_min)
    assert torch.all(effective_log_std < policy.cfg.sac_log_std_max)
    assert policy.bc_dagger_sac_adapter.log_std.grad is not None
    assert policy.bc_dagger_sac_adapter.log_std.grad.abs().item() > 0.0


def test_bounded_log_prob_converts_exactly_to_nonunit_q_coordinates():
    latent_mean = torch.tensor([[0.05, -0.10], [0.025, 0.15]])
    latent_scale = torch.tensor([[0.02, 0.04], [0.02, 0.04]])
    raw_action = torch.tensor([[1.3, -1.1], [-0.2, 4.5]])
    q_center = torch.tensor([3.0, -1.0])
    q_scale = torch.tensor([2.0, 4.0])
    action_low = torch.tensor([-20.0, -20.0])
    action_high = torch.tensor([20.0, 20.0])
    physical_dist = FastSACTanhNormal(
        latent_mean, latent_scale, low=action_low, high=action_high, event_dims=1
    )
    normalized_dist = FastSACTanhNormal(
        latent_mean,
        latent_scale,
        low=(action_low - q_center) / q_scale,
        high=(action_high - q_center) / q_scale,
        event_dims=1,
    )
    raw_log_prob = physical_dist.log_prob_for_action(raw_action)
    normalized_log_prob = normalized_dist.log_prob_for_action(
        (raw_action - q_center) / q_scale
    )
    policy = _bare_policy()
    policy._fastsac_entropy_reference_log_scale_sum = float(q_scale.log().sum())

    assert torch.allclose(
        policy._normalized_action_log_prob(raw_log_prob), normalized_log_prob
    )
    assert torch.allclose(normalized_log_prob, raw_log_prob + q_scale.log().sum())


def test_physical_autotune_ratio_one_target_is_reachable_for_g1_action_scales():
    # Locked nominal half-ranges for the 23 controlled G1 skateboard joints.
    q_scale = torch.tensor(
        [
            4.4267721,
            4.4267721,
            4.2840000,
            4.4880424,
            4.4880424,
            1.0636362,
            4.5124359,
            4.5124359,
            1.0636362,
            3.8148003,
            3.8148003,
            5.8904991,
            5.8904991,
            1.4280033,
            1.4280033,
            3.9269657,
            3.9269657,
            0.5354999,
            0.5354999,
            5.3550000,
            5.3550000,
            3.2129998,
            3.2129998,
        ]
    )
    policy = _bare_policy(sac_physical_std_min=0.05, sac_physical_std_max=0.5)
    policy._fastsac_q_action_scale = q_scale
    policy.target_entropy = _fastsac_target_entropy(-q_scale, q_scale, 1.0)

    entropy_bounds = policy._physical_normalized_entropy_bounds()
    entropy_min, entropy_max = entropy_bounds
    policy._validate_physical_entropy_target_reachable()

    assert policy.target_entropy == pytest.approx(-23.0)
    assert entropy_min < policy.target_entropy < entropy_max
    assert policy._fastsac_physical_entropy_bounds is entropy_bounds
    assert policy._physical_normalized_entropy_bounds() is entropy_bounds
    # The equivalent uniform physical std lies comfortably within the guard.
    target_uniform_std = math.exp(
        (
            policy.target_entropy
            + q_scale.double().log().sum().item()
        )
        / q_scale.numel()
        - 0.5 * math.log(2.0 * math.pi * math.e)
    )
    assert 0.05 < target_uniform_std < 0.5
    assert target_uniform_std == pytest.approx(0.26035, rel=1.0e-4)


def test_exact_bc_on_deterministic_raw_mean_has_no_log_std_gradient():
    raw_mean = nn.Parameter(torch.tensor([[0.2], [-0.4], [0.8]]))
    adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.tensor(-4.0),
        device="cpu",
    )
    teacher = torch.tensor([[0.5], [-0.25], [9.0]])
    valid = torch.tensor([True, True, False])

    policy = _bare_policy(
        sac_log_std_min=-8.0, sac_log_std_max=-2.0, action_support_clip=20.0
    )
    _install_unit_action_contract(policy)
    policy.bc_dagger_sac_adapter = adapter
    bounded_mean = policy._sac_dist_from_mean(raw_mean).mean
    # BC owns only the bounded noise-free mean, never policy variance.
    loss = _exact_teacher_bc_loss(
        bounded_mean,
        teacher,
        valid,
        torch.tensor([0.0]),
        torch.tensor([20.0]),
        huber_delta=1.0,
    )
    loss.backward()

    assert raw_mean.grad is not None
    assert raw_mean.grad.abs().sum().item() > 0.0
    assert adapter.log_std.grad is None
    assert torch.equal(raw_mean.grad[~valid], torch.zeros_like(raw_mean.grad[~valid]))


def test_raw_replay_and_teacher_prefill_implementation_is_inherited_unchanged():
    assert issubclass(DistributionalFastSACTeacherBC, DistributionalTD3TeacherBC)
    inherited_methods = (
        "_teacher_prefill_active",
        "_collect_teacher_q_replay_this_rollout",
        "_teacher_q_replay_frozen",
        "_stage_teacher_prefill_rows",
        "_discard_unresolved_teacher_prefill_rows",
        "configure_teacher_replay",
        "snapshot_teacher_replay",
        "_raw_perception_values",
        "_prepare_raw_final_state",
        "capture_truncation_final_observations",
        "capture_rollout_final_observation",
        "_dagger_transition_chunks",
        "_reencode_perception_windows",
        "_prepare_dagger_learning_batch",
        "_sample_balanced_q_batch",
        "_sample_actor_batch",
        "_update_failure_phase_histogram",
        "_sample_teacher_indices",
        "_build_teacher_phase_index",
        "_prefetch_curriculum_sample_plans",
        "_load_pretrained_perception_checkpoint",
        "_set_perception_trainable",
        "_run_teacher_perception_warmup",
        "train_adapt",
        "_student_latent",
    )
    for method in inherited_methods:
        assert getattr(DistributionalFastSACTeacherBC, method) is getattr(
            DistributionalTD3TeacherBC, method
        )


def test_fastsac_uses_the_shared_td3_failure_phase_curriculum_contract():
    assert _failure_lookback_offsets(50, 10).tolist() == [
        0,
        6,
        11,
        17,
        22,
        28,
        33,
        39,
        44,
        50,
    ]
    assert _source_counts(4096, 0.5, 0.3) == (2048, 1434, 614)


def test_class_does_not_own_a_target_actor_or_any_td3_noise_rng_contract():
    # SAC evaluates its current stochastic Actor in the soft Bellman target.
    # There must be no FastSAC override that routes back through TD3 smoothing.
    assert "_smoothed_target_q_action" not in DistributionalFastSACTeacherBC.__dict__
    assert "_actor_target_dist_from_flat" not in DistributionalFastSACTeacherBC.__dict__
    checkpoint_names = set(
        DistributionalFastSACTeacherBC._fastsac_checkpoint_state.__code__.co_names
    )
    assert "actor_target" not in checkpoint_names
    assert "target_policy_rng" not in checkpoint_names
    assert "collector_exploration_rng" not in checkpoint_names


def _install_tiny_stochastic_actor(policy, *, actor_weight=0.25, log_std=-1.0):
    policy.actor_adapt = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.actor_adapt.weight.fill_(actor_weight)
    raw_log_std = policy._inverse_smooth_log_std(
        torch.tensor(log_std),
        policy.cfg.sac_log_std_min,
        policy.cfg.sac_log_std_max,
    )
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=raw_log_std,
        device="cpu",
    )

    def actor_mean(owner, observations):
        return owner.actor_adapt(observations)

    policy._actor_mean_from_flat = MethodType(actor_mean, policy)
    policy._fastsac_actor_parameters = tuple(policy.actor_adapt.parameters()) + tuple(
        policy.bc_dagger_sac_adapter.parameters()
    )
    return policy


class _TinyPhysicalActor(nn.Module):
    """Small mean module that still owns the production PPOVEL Actor core."""

    def __init__(self, action_dim: int, *, mean_weight: float, std: float):
        super().__init__()
        self.mean_weight = nn.Parameter(
            torch.full((1, action_dim), float(mean_weight))
        )
        self.actor_core = Actor(
            action_dim,
            init_noise_scale=float(std),
            load_noise_scale=0.5,
        )
        # Materialize the compatibility LazyLinear parameters. The test mean
        # intentionally bypasses them, but production parameter ownership does
        # not special-case this harness.
        self.actor_core(torch.zeros(1, 1))

    def forward(self, observations):
        return observations @ self.mean_weight


def _tiny_physical_policy(*, q_slope: float = 1.0, action_dim: int = 1):
    policy = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        sac_physical_std_lr=1.0e-5,
        sac_physical_std_max_kl=0.01,
        sac_physical_std_min=0.05,
        sac_physical_std_max=0.5,
        eta_sac=0.7,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        sac_max_grad_norm=1.0,
        q_action_input_gain=1.0,
    )
    policy.actor_adapt = _TinyPhysicalActor(
        action_dim, mean_weight=0.25, std=0.5
    )
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim, torch.zeros(action_dim), "cpu"
    ).requires_grad_(False)
    policy._fastsac_q_action_center = torch.zeros(action_dim)
    policy._fastsac_q_action_scale = torch.ones(action_dim)
    policy._fastsac_action_low = torch.full((action_dim,), -20.0)
    policy._fastsac_action_high = torch.full((action_dim,), 20.0)
    policy._fastsac_actor_action_center = torch.zeros(action_dim)
    policy._fastsac_actor_action_scale = torch.full((action_dim,), 20.0)
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.cfg.action_support_clip = 20.0
    actor_std = policy._ppo_actor_std_parameter()
    mean_parameters = tuple(
        parameter
        for parameter in policy.actor_adapt.parameters()
        if parameter is not actor_std
    )
    policy.actor_optimizer = _CountingSGD(mean_parameters, lr=0.05)
    policy.actor_std_optimizer = _CountingAdam(
        (actor_std,), lr=policy.cfg.sac_physical_std_lr
    )
    policy.qnet = _ActionSensitiveTwinC51(
        first_slope=float(q_slope), second_slope=-2.0 * float(q_slope)
    )
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    policy.actor_std_update_count = 0
    return policy


def _apply_physical_std_gradient(policy, value: float):
    parameter = policy._ppo_actor_std_parameter()
    policy.actor_std_optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, float(value))
    return policy._physical_actor_std_update(before)


def _tiny_physical_batch(action_dim: int = 1):
    return {
        "observations": torch.tensor([[1.0], [2.0], [-1.0]]),
        "critic_observations": torch.ones(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[0.5], [-0.2], [0.1]]
        ).expand(-1, action_dim),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([True, False, True]),
    }


def test_physical_actor_mean_and_std_optimizers_are_disjoint_with_separate_lrs():
    policy = _tiny_physical_policy()
    mean_parameters = {
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    std_parameters = {
        id(parameter)
        for group in policy.actor_std_optimizer.param_groups
        for parameter in group["params"]
    }

    assert not mean_parameters.intersection(std_parameters)
    assert std_parameters == {id(policy._ppo_actor_std_parameter())}
    assert policy.actor_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert policy.actor_std_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert isinstance(policy.actor_std_optimizer, torch.optim.Adam)


def test_physical_scale_only_kl_cap_uses_rollout_reference_and_limit_point_zero_one():
    policy = _tiny_physical_policy(action_dim=2)
    reference = torch.full((2,), 0.5)
    policy._physical_std_rollout_reference = reference.clone()
    with torch.no_grad():
        policy._ppo_actor_std_parameter().fill_(0.1)

    uncapped_kl = policy._physical_scale_kl(
        reference, policy._ppo_actor_std_parameter().detach()
    )
    capped_kl, capped = policy._cap_physical_actor_std_rollout_kl_(reference)

    assert uncapped_kl.item() > 0.01
    assert capped.item() == 1.0
    assert capped_kl.item() <= 0.01 + 1.0e-7
    assert capped_kl.item() == pytest.approx(0.01, abs=1.0e-6)
    assert torch.all(policy._ppo_actor_std_parameter() < reference)
    assert torch.all(policy._ppo_actor_std_parameter() > 0.1)


def test_physical_gradient_probe_reports_sac_q_and_entropy_but_no_bc_std():
    policy = _tiny_physical_policy()

    result = diagnose_fastsac_actor_gradients(
        policy,
        _tiny_physical_batch(),
        sample_seed=313,
        source_gradients=False,
    )

    std_group = result["gradients"]["all"]["groups"]["global_log_std"]
    assert std_group["unweighted_norms"]["bc"] == 0.0
    assert std_group["unweighted_norms"]["q"] > 0.0
    assert std_group["unweighted_norms"]["entropy"] > 0.0
    assert std_group["unweighted_norms"]["sac"] > 0.0


def test_physical_actor_update_applies_ordinary_sac_gradient_to_direct_std():
    policy = _tiny_physical_policy(q_slope=1.0)
    with torch.no_grad():
        policy._ppo_actor_std_parameter().fill_(0.3)
    std_before = policy._ppo_actor_std_parameter().detach().clone()

    metrics = policy._actor_update(_tiny_physical_batch())

    assert policy.actor_std_optimizer.step_calls == 1
    assert policy.actor_std_update_count == 1
    assert not torch.equal(policy._ppo_actor_std_parameter(), std_before)
    assert policy._ppo_actor_std_parameter().grad is None
    assert metrics["actor_std_sac_grad_norm"].item() > 0.0
    assert metrics["actor_std_step_abs_mean"].item() > 0.0
    assert "actor_std_rollout_scale_kl" not in metrics
    assert "actor_std_kl_cap_fraction" not in metrics
    assert torch.all(policy._ppo_actor_std_parameter() >= 0.05)
    assert torch.all(policy._ppo_actor_std_parameter() <= 0.5)


def test_physical_rollout_applies_cumulative_kl_once_after_all_replay_updates(
    monkeypatch,
):
    policy = _tiny_physical_policy(action_dim=2)
    policy.joint_names = ("joint_0", "joint_1")
    policy.target_entropy = -2.0
    policy.alpha_update_count = 0
    cap_calls = 0
    original_cap = policy._cap_physical_actor_std_rollout_kl_

    def counted_cap(owner, fallback_reference):
        nonlocal cap_calls
        cap_calls += 1
        return original_cap(fallback_reference)

    policy._cap_physical_actor_std_rollout_kl_ = MethodType(counted_cap, policy)

    def replay_updates(owner, tensordict):
        del tensordict
        # Model multiple replay steps by presenting their final in-bound std.
        # The rollout-old trust guard must run only after this method returns.
        with torch.no_grad():
            owner._ppo_actor_std_parameter().fill_(0.1)
        assert cap_calls == 0
        return {}

    monkeypatch.setattr(DistributionalTD3TeacherBC, "train_op", replay_updates)

    info = DistributionalFastSACTeacherBC.train_op(
        policy, TensorDict({}, batch_size=[])
    )

    assert cap_calls == 1
    assert info["fastsac/actor_std_kl_cap_fraction"] == pytest.approx(1.0)
    assert info["fastsac/actor_std_rollout_scale_kl"] == pytest.approx(
        0.01, abs=1.0e-6
    )
    final_std = policy._ppo_actor_std_parameter().detach()
    assert torch.all(final_std > 0.1)
    assert torch.all(final_std < 0.5)
    assert info["fastsac/actor_std/joint_0"] == pytest.approx(final_std[0].item())
    assert info["fastsac/actor_std/joint_1"] == pytest.approx(final_std[1].item())


def test_physical_std_projection_and_nonfinite_failure_are_explicit():
    policy = _tiny_physical_policy(action_dim=2)
    with torch.no_grad():
        policy._ppo_actor_std_parameter().copy_(torch.tensor([0.9, 0.01]))

    projection_fraction = policy._project_physical_actor_std_()

    std = policy._ppo_actor_std_parameter().detach()
    assert torch.all((std >= 0.05) & (std <= 0.5))
    assert projection_fraction.item() == pytest.approx(1.0)
    with torch.no_grad():
        policy._ppo_actor_std_parameter()[0] = torch.nan
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        policy._project_physical_actor_std_()


def test_reparameterized_actor_step_combines_entropy_min_twin_q_and_exact_bc():
    policy = _bare_policy(
        eta_sac=0.7,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = _CountingSGD(policy._fastsac_actor_parameters, lr=0.05)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [-1.0]]),
        "critic_observations": torch.ones(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.5], [-0.2], [0.1]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([True, False, True]),
    }

    expected_actor = copy.deepcopy(policy.actor_adapt)
    expected_adapter = copy.deepcopy(policy.bc_dagger_sac_adapter)
    expected_q = copy.deepcopy(policy.qnet)
    expected_rng = torch.Generator().set_state(policy.sac_action_rng.get_state())
    raw_mean = expected_actor(batch["observations"])
    expected_dist = FastSACTanhNormal(
        raw_mean / 20.0,
        (
            policy._bounded_log_std(expected_adapter.log_std).exp() / 20.0
        ).expand_as(raw_mean),
        low=torch.tensor([-20.0]),
        high=torch.tensor([20.0]),
        event_dims=1,
    )
    sampled_action, physical_log_prob = expected_dist.rsample_with_log_prob(
        generator=expected_rng
    )
    log_prob = physical_log_prob
    expected_heads = expected_q.values(
        expected_q(batch["critic_observations"], sampled_action)
    )
    expected_pessimistic_q = expected_heads.min(dim=0).values
    expected_sac = (
        policy.log_alpha.detach().exp() * log_prob - expected_pessimistic_q
    ).mean()
    expected_bc = _exact_teacher_bc_loss(
        expected_dist.mean,
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        batch[DAGGER_TEACHER_ACTION_VALID_KEY],
        policy._fastsac_q_action_center,
        policy._fastsac_q_action_scale,
        policy.cfg.dagger_actor_huber_delta,
    )
    expected_total = (
        policy.cfg.eta_sac * expected_sac + policy.cfg.lambda_bc * expected_bc
    )
    expected_gradients = torch.autograd.grad(
        expected_total,
        (expected_actor.weight, expected_adapter.log_std),
    )
    actor_weight_before = policy.actor_adapt.weight.detach().clone()
    log_std_before = policy.bc_dagger_sac_adapter.log_std.detach().clone()

    metrics = policy._actor_update(batch)

    assert policy.actor_optimizer.step_calls == 1
    assert policy.actor_update_count == 1
    assert torch.allclose(
        policy.actor_adapt.weight,
        actor_weight_before - 0.05 * expected_gradients[0],
    )
    assert torch.allclose(
        policy.bc_dagger_sac_adapter.log_std,
        log_std_before - 0.05 * expected_gradients[1],
    )
    assert metrics["sac_actor_loss"].item() == pytest.approx(expected_sac.item())
    assert metrics["exact_bc_loss"].item() == pytest.approx(expected_bc.item())
    assert metrics["weighted_sac_actor_loss"].item() == pytest.approx(
        policy.cfg.eta_sac * expected_sac.item()
    )
    assert metrics["weighted_bc_loss"].item() == pytest.approx(
        policy.cfg.lambda_bc * expected_bc.item()
    )
    assert metrics["actor_min_expected_q_mean"].item() == pytest.approx(
        expected_pessimistic_q.mean().item()
    )
    assert metrics["actor_teacher_replay_fraction"].item() == pytest.approx(2 / 3)
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())
    assert all(parameter.requires_grad for parameter in policy.qnet.parameters())


def test_bc_only_actor_step_does_not_update_global_log_std():
    policy = _bare_policy(
        eta_sac=0.0,
        lambda_bc=1.0,
        dagger_actor_huber_delta=1.0,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = _CountingSGD(policy._fastsac_actor_parameters, lr=0.05)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    log_std_before = policy.bc_dagger_sac_adapter.log_std.detach().clone()
    batch = {
        "observations": torch.tensor([[1.0], [2.0]]),
        "critic_observations": torch.ones(2, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.5], [-0.2]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.ones(2, dtype=torch.bool),
    }

    policy._actor_update(batch)

    assert torch.equal(policy.bc_dagger_sac_adapter.log_std, log_std_before)


def test_gradient_probe_separates_components_and_is_strictly_read_only():
    policy = _bare_policy(
        eta_sac=0.2,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy, actor_weight=0.25, log_std=-1.0)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = torch.optim.AdamW(
        (
            {"params": tuple(policy.actor_adapt.parameters())},
            {"params": tuple(policy.bc_dagger_sac_adapter.parameters())},
        ),
        lr=0.01,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(71)
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [-1.0], [0.5]]),
        "critic_observations": torch.ones(4, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[0.5], [-0.2], [0.1], [0.3]]
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor(
            [True, True, False, True]
        ),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor(
            [False, False, True, True]
        ),
        FAILURE_PHASE_TEACHER_SOURCE_KEY: torch.tensor(
            [False, False, False, True]
        ),
    }

    actor_parameters = tuple(policy.actor_adapt.parameters()) + tuple(
        policy.bc_dagger_sac_adapter.parameters()
    )
    for index, parameter in enumerate(actor_parameters):
        parameter.grad = torch.full_like(parameter, float(index + 1))
        parameter.requires_grad_(False)
    policy.qnet.requires_grad_(False)
    policy.log_alpha.requires_grad_(False)
    model_before = copy.deepcopy(nn.Module.state_dict(policy))
    optimizer_before = copy.deepcopy(policy.actor_optimizer.state_dict())
    rng_before = policy.sac_action_rng.get_state().clone()
    gradients_before = tuple(parameter.grad.clone() for parameter in actor_parameters)
    requires_grad_before = tuple(
        parameter.requires_grad for parameter in actor_parameters
    )

    result = diagnose_fastsac_actor_gradients(
        policy,
        batch,
        sample_seed=313,
        source_gradients=True,
    )

    assert result["schema"] == GRADIENT_PROBE_SCHEMA
    assert result["batch_rows"] == 4
    assert result["strata"]["student"]["rows"] == 2
    assert result["strata"]["teacher"]["rows"] == 2
    assert result["strata"]["failure_teacher"]["rows"] == 1
    assert result["strata"]["all"]["dqda_policy_sample_norm_mean"] > 0.0
    assert (
        result["strata"]["all"][
            "cos_dqda_mean_with_teacher_minus_mean"
        ]
        is not None
    )
    assert (
        result["strata"]["all"][
            "teacher_direction_fd_sign_agreement_fraction"
        ]
        == pytest.approx(1.0)
    )
    assert (
        result["strata"]["all"][
            "teacher_direction_fd_relative_error_mean"
        ]
        < 0.02
    )
    all_groups = result["gradients"]["all"]["groups"]
    assert all_groups["actor_mean"]["unweighted_norms"]["bc"] > 0.0
    assert all_groups["global_log_std"]["unweighted_norms"]["bc"] == 0.0
    assert all_groups["global_log_std"]["unweighted_norms"]["q"] > 0.0
    assert all_groups["global_log_std"]["unweighted_norms"]["entropy"] > 0.0
    assert all_groups["global_log_std"]["cosines"]["bc_q"] is None
    json.dumps(result, allow_nan=False)

    _assert_nested_equal(nn.Module.state_dict(policy), model_before)
    _assert_nested_equal(policy.actor_optimizer.state_dict(), optimizer_before)
    assert torch.equal(policy.sac_action_rng.get_state(), rng_before)
    assert tuple(parameter.requires_grad for parameter in actor_parameters) == (
        requires_grad_before
    )
    for parameter, gradient in zip(actor_parameters, gradients_before):
        assert torch.equal(parameter.grad, gradient)


def _rollout_owner(*, prefill: bool, beta: float = 0.5):
    action_dim = 2
    batch_size = 64
    raw_mean = torch.zeros(batch_size, action_dim)
    teacher = torch.zeros_like(raw_mean)
    cfg = SimpleNamespace(
        teacher_prefill_max_rollouts=1,
        dagger_control_mode="beta",
        dagger_safe_takeover_rms=0.006,
        dagger_safe_release_rms=0.004,
        dagger_safe_min_teacher_steps=8,
        q_action_input_gain=1.0,
        collector_exploration_noise_std=0.0,
        collector_exploration_noise_clip=0.0,
        sac_log_std_min=-10.0,
        sac_log_std_max=-2.0,
        action_support_clip=20.0,
    )
    owner = SimpleNamespace(cfg=cfg)
    owner.teacher_prefill_rollout_count = 0 if prefill else 1
    owner.dagger_rollout_count = 0
    owner.dagger_rng = torch.Generator().manual_seed(17)
    owner.sac_rollout_rng = torch.Generator().manual_seed(18)
    owner.sac_action_rng = torch.Generator().manual_seed(19)
    owner._student_raw_action_proposal = lambda td: raw_mean.clone()
    owner._student_mean_action = lambda td: 20.0 * torch.tanh(raw_mean / 20.0)
    owner._teacher_action = lambda td: teacher.clone()
    owner._teacher_prefill_active = lambda: prefill
    owner._effective_control_mode = lambda: "beta"
    owner._teacher_mixture_probability = lambda: beta
    owner._safe_teacher_control_enabled = lambda: False
    owner._fastsac_q_action_center = torch.zeros(action_dim)
    owner._fastsac_q_action_scale = torch.ones(action_dim)
    owner._fastsac_action_low = torch.full((action_dim,), -20.0)
    owner._fastsac_action_high = torch.full((action_dim,), 20.0)
    owner._project_execution_action = lambda action: action.clamp(-20.0, 20.0)

    def stochastic_dist(value):
        return FastSACTanhNormal(
            value / 20.0,
            torch.full_like(value, 0.15 / 20.0),
            low=owner._fastsac_action_low,
            high=owner._fastsac_action_high,
            event_dims=1,
        )

    owner._sac_dist_from_mean = stochastic_dist
    owner._q_action_input = lambda action: action.clamp(-20.0, 20.0)
    return owner, raw_mean, teacher


def test_main_rollout_keeps_teacher_rows_exact_and_samples_only_student_behavior():
    owner, raw_mean, teacher = _rollout_owner(prefill=False, beta=0.5)
    projection_calls = 0
    project_execution_action = owner._project_execution_action

    def counted_projection(action):
        nonlocal projection_calls
        projection_calls += 1
        return project_execution_action(action)

    owner._project_execution_action = counted_projection
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(raw_mean.shape[0], dtype=torch.bool)},
        batch_size=[raw_mean.shape[0]],
    )
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()
    learning_rng_before = owner.sac_action_rng.get_state().clone()

    result = policy(td)

    student = result[DAGGER_IS_STUDENT_ACTION_KEY]
    assert student.any() and (~student).any()
    assert torch.equal(result[DAGGER_TEACHER_ACTION_KEY], teacher)
    assert result[DAGGER_TEACHER_ACTION_VALID_KEY].all()
    # Teacher and Student are each projected once. Their elementwise selection
    # is already inside the same support and must not be projected a third time.
    assert projection_calls == 2
    assert torch.equal(result[ACTION_KEY][~student], teacher[~student])
    assert torch.equal(result[TD3_NOISE_FREE_STUDENT_ACTION_KEY], raw_mean)
    assert not torch.equal(
        result[ACTION_KEY][student],
        result[TD3_NOISE_FREE_STUDENT_ACTION_KEY][student],
    )
    # Safety compares Teacher with the noise-free mean, never a lucky/unlucky
    # exploration draw. Here those means are identical, so discrepancy is zero.
    assert torch.equal(
        result[DAGGER_ACTION_DISCREPANCY_RMS_KEY],
        torch.zeros(raw_mean.shape[0]),
    )
    assert not torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)
    assert torch.equal(owner.sac_action_rng.get_state(), learning_rng_before)


def test_teacher_only_prefill_is_bitwise_exact_and_does_not_draw_sac_noise():
    owner, raw_mean, teacher = _rollout_owner(prefill=True, beta=0.0)
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(raw_mean.shape[0], dtype=torch.bool)},
        batch_size=[raw_mean.shape[0]],
    )
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()

    result = policy(td)

    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], teacher)
    assert torch.equal(result[TD3_EXPLORATORY_STUDENT_ACTION_KEY], teacher)
    assert torch.equal(result[TD3_COLLECTOR_NOISE_KEY], torch.zeros_like(teacher))
    assert torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)


def test_deterministic_eval_uses_mean_without_advancing_any_sac_rng():
    owner, raw_mean, _ = _rollout_owner(prefill=False, beta=0.0)
    raw_mean = torch.tensor([[25.0, -31.0]]).expand_as(raw_mean).clone()
    owner.cfg.use_object_adapt = False
    owner.depth_feature_dim = 1
    owner.adapt_ema = nn.Identity()
    owner.actor_adapt = nn.Identity()
    owner._student_raw_action_proposal = lambda td: raw_mean.clone()
    owner._student_mean_action = lambda td: 20.0 * torch.tanh(raw_mean / 20.0)
    eval_policy = _DeterministicFastSACStudentEvalPolicy(owner)
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()
    learning_rng_before = owner.sac_action_rng.get_state().clone()
    td = TensorDict({}, batch_size=[raw_mean.shape[0]])

    first = eval_policy(td.clone())[ACTION_KEY]
    second = eval_policy(td.clone())[ACTION_KEY]

    assert torch.equal(first, second)
    assert torch.allclose(first, 20.0 * torch.tanh(raw_mean / 20.0))
    assert torch.all(first >= -20.0) and torch.all(first <= 20.0)
    assert torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)
    assert torch.equal(owner.sac_action_rng.get_state(), learning_rng_before)


class _FixedStochasticDist:
    def __init__(self, batch_size: int, *, action: float, log_prob: float):
        self.loc = torch.zeros(batch_size, 1)
        self.mean = torch.zeros(batch_size, 1)
        self._action = float(action)
        self._log_prob = float(log_prob)

    def rsample_with_log_prob(self, generator=None):
        if generator is not None:
            # Deliberately consume this stream so independence is observable.
            torch.rand((), generator=generator)
        action = self.loc.new_full(self.loc.shape, self._action)
        log_prob = self.loc.new_full((self.loc.shape[0],), self._log_prob)
        return action, log_prob


def test_soft_c51_target_uses_entropy_and_one_complete_lower_twin_head_detached():
    policy = _bare_policy(
        gamma=1.0,
        q_action_input_gain=1.0,
        sac_use_autotune=True,
    )
    _install_unit_action_contract(policy)
    first = torch.tensor([[0.05, 0.15, 0.80], [0.05, 0.15, 0.80]])
    second = torch.tensor([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05]])
    policy.qnet_target = _TableTwinC51(first, second).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2)))
    policy.sac_action_rng = torch.Generator().manual_seed(91)
    policy._actor_dist_from_flat = lambda observations: _FixedStochasticDist(
        observations.shape[0], action=0.25, log_prob=-0.5
    )
    policy._actor_target_dist_from_flat = lambda observations: (_ for _ in ()).throw(
        AssertionError("SAC target must use the online stochastic Actor")
    )
    rng_before = policy.sac_action_rng.get_state().clone()
    batch = {
        "next_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    result = policy._distributional_fastsac_target(batch)
    projected, diagnostics = result[:2]

    # r - gamma * alpha * log(pi) = 0 - 1 * .2 * -.5 = +.1.
    expected, _, _ = _project_c51_probabilities(
        second,
        rewards=torch.full((2,), 0.1),
        bootstrap=torch.ones(2),
        effective_discount=torch.ones(2),
        support=policy.qnet_target.support,
    )
    assert torch.allclose(projected, expected)
    assert projected.grad_fn is None
    assert projected.requires_grad is False
    assert diagnostics["target_select_q2_fraction"].item() == pytest.approx(1.0)
    assert diagnostics["target_log_prob_mean"].item() == pytest.approx(-0.5)
    assert diagnostics["entropy_tax_mean"].item() == pytest.approx(-0.1)
    assert not torch.equal(policy.sac_action_rng.get_state(), rng_before)
    assert all(parameter.grad is None for parameter in policy.qnet_target.parameters())


@pytest.mark.parametrize(
    ("log_prob", "direction"),
    ((2.0, "increase"), (-2.0, "decrease")),
)
def test_temperature_update_has_the_standard_entropy_dual_sign(log_prob, direction):
    policy = _bare_policy(sac_use_autotune=True)
    policy.log_alpha = nn.Parameter(torch.tensor(0.0))
    policy.alpha_optimizer = _CountingSGD([policy.log_alpha], lr=0.1)
    policy.target_entropy = -1.0
    policy.alpha_update_count = 0
    before = policy.log_alpha.detach().clone()

    metrics = policy._alpha_update(torch.full((4,), log_prob))

    assert policy.alpha_optimizer.step_calls == 1
    assert policy.alpha_update_count == 1
    if direction == "increase":
        assert policy.log_alpha.item() > before.item()
    else:
        assert policy.log_alpha.item() < before.item()
    assert metrics["alpha"].item() == pytest.approx(policy.log_alpha.exp().item())


def test_temperature_log_alpha_gradient_is_not_attenuated_by_small_alpha():
    gradients = []
    deltas = []
    for initial_alpha in (1.0, 1.0e-5):
        policy = _bare_policy(sac_use_autotune=True)
        policy.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha)))
        policy.alpha_optimizer = _CountingSGD([policy.log_alpha], lr=0.1)
        policy.target_entropy = -1.0
        policy.alpha_update_count = 0
        before = policy.log_alpha.detach().clone()

        metrics = policy._alpha_update(torch.full((4,), 2.0))

        gradients.append(metrics["alpha_grad_norm"].item())
        deltas.append((policy.log_alpha.detach() - before).item())

    assert gradients == pytest.approx([1.0, 1.0])
    assert deltas == pytest.approx([0.1, 0.1], abs=1.0e-6)


def test_critic_temperature_population_excludes_true_terminals_but_keeps_timeouts():
    """Actor-cadence alpha uses only next states which can bootstrap."""
    policy = _bare_policy(
        sac_use_autotune=True,
        sac_alpha_update_cadence="actor",
        sac_policy_frequency=2,
        sac_max_grad_norm=1.0e6,
        sac_tau=0.0,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = nn.Parameter(torch.tensor(0.0))
    policy.alpha_optimizer = _CountingSGD([policy.log_alpha], lr=0.1)
    policy.target_entropy = -1.0
    policy.critic_update_count = 0
    policy.alpha_update_count = 0
    policy.sac_alpha_update_count = 0

    target = torch.full((3, 3), 1.0 / 3.0)
    # The physical terminal has an extreme opposite-sign log probability.  It
    # would reverse the update if it leaked into the alpha population.
    target_log_prob = torch.tensor([2.0, -100.0, 2.0])
    policy._distributional_fastsac_target = MethodType(
        lambda owner, batch: (target, {}, target_log_prob), policy
    )
    before = policy.log_alpha.detach().clone()
    batch = {
        "critic_observations": torch.zeros(3, 1),
        "actions": torch.zeros(3, 1),
        "dones": torch.tensor([False, True, True]),
        # Row 2 is a time-limit truncation and therefore has a valid next state.
        "truncations": torch.tensor([False, False, True]),
    }

    skipped = policy._critic_update(batch)

    assert torch.equal(policy.log_alpha, before)
    assert policy.alpha_optimizer.step_calls == 0
    assert policy.alpha_update_count == 0
    assert skipped["alpha_update_due_fraction"].item() == 0.0
    assert skipped["alpha_update_performed_fraction"].item() == 0.0
    assert skipped["alpha_loss"].item() == 0.0
    assert skipped["alpha_grad_norm"].item() == 0.0

    performed = policy._critic_update(batch)

    assert policy.log_alpha.item() > before.item()
    assert policy.alpha_optimizer.step_calls == 1
    assert policy.alpha_update_count == 1
    assert performed["alpha_update_due_fraction"].item() == 1.0
    assert performed["alpha_update_performed_fraction"].item() == 1.0

    alpha_seen_by_actor = []
    policy._actor_update = MethodType(
        lambda owner, actor_batch: alpha_seen_by_actor.append(
            owner.log_alpha.detach().clone()
        )
        or {"batch": actor_batch},
        policy,
    )
    policy._maybe_delayed_actor_and_targets({"sentinel": True})
    assert len(alpha_seen_by_actor) == 1
    assert torch.equal(alpha_seen_by_actor[0], policy.log_alpha.detach())

    after_valid_population = policy.log_alpha.detach().clone()
    batch["dones"] = torch.ones(3, dtype=torch.bool)
    batch["truncations"] = torch.zeros(3, dtype=torch.bool)
    # Critic update 3 is off cadence; Critic update 4 is due but has no valid
    # temperature population.  Neither may invoke the optimizer.
    policy._critic_update(batch)
    terminal_due = policy._critic_update(batch)
    assert torch.equal(policy.log_alpha, after_valid_population)
    assert policy.alpha_optimizer.step_calls == 1
    assert policy.alpha_update_count == 1
    assert terminal_due["alpha_update_due_fraction"].item() == 1.0
    assert terminal_due["alpha_update_performed_fraction"].item() == 0.0


def test_fixed_temperature_skips_optimizer_but_remains_in_soft_target():
    policy = _bare_policy(sac_use_autotune=False)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.25)))
    policy.alpha_optimizer = _CountingSGD([policy.log_alpha], lr=0.1)
    policy.target_entropy = -1.0
    policy.alpha_update_count = 0
    before = policy.log_alpha.detach().clone()

    metrics = policy._alpha_update(torch.full((4,), 2.0))

    assert policy.alpha_optimizer.step_calls == 0
    assert policy.alpha_update_count == 0
    assert torch.equal(policy.log_alpha, before)
    assert metrics["alpha"].item() == pytest.approx(0.25)


def _checkpoint_policy(seed: int):
    policy = _bare_policy(
        sac_use_autotune=True,
        sac_alpha_update_cadence="actor",
        sac_log_std_min=-10.0,
        sac_log_std_max=-2.0,
        q_batch_size=512,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy.actor_adapt = nn.Linear(2, 1)
        policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(1, torch.tensor(-2.0), "cpu")
        policy.qnet = _ActionSensitiveTwinC51()
        policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
        policy.adapt_probe = nn.Linear(2, 2)
    actor_parameters = tuple(policy.actor_adapt.parameters()) + tuple(
        policy.bc_dagger_sac_adapter.parameters()
    )
    policy.actor_optimizer = torch.optim.Adam(actor_parameters, lr=3e-3)
    policy.critic_optimizer = torch.optim.Adam(policy.qnet.parameters(), lr=4e-3)
    policy.log_alpha = nn.Parameter(torch.tensor(-1.0))
    policy.alpha_optimizer = torch.optim.Adam([policy.log_alpha], lr=5e-3)
    policy.opt_adapt = torch.optim.Adam(policy.adapt_probe.parameters(), lr=6e-3)
    policy.actor_update_count = 0
    policy.critic_update_count = 0
    policy.alpha_update_count = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.teacher_prefill_rollout_count = 0
    policy.teacher_prefill_environment_steps = 0
    policy.dagger_rng = torch.Generator().manual_seed(seed + 1)
    policy.q_rng = torch.Generator().manual_seed(seed + 2)
    policy.sac_action_rng = torch.Generator().manual_seed(seed + 3)
    policy.sac_rollout_rng = torch.Generator().manual_seed(seed + 4)
    policy.teacher_perception_rng = torch.Generator().manual_seed(seed + 5)
    policy.actor_target = None
    policy._last_fastsac_diagnostics = {}
    policy._fastsac_action_contract = {
        "joint_names": ["unit_joint"],
        "fingerprint": "sha256:unit-test",
    }
    return policy


def _physical_checkpoint_policy(seed: int):
    policy = _checkpoint_policy(seed)
    policy.cfg.sac_action_distribution = PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    policy.cfg.sac_physical_std_lr = 1.0e-5
    policy.cfg.sac_physical_std_max_kl = 0.01
    policy.cfg.sac_physical_std_min = 0.05
    policy.cfg.sac_physical_std_max = 0.5
    policy.cfg.sac_max_grad_norm = 1.0
    policy.actor_backend = PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND
    policy.actor_adapt = _TinyPhysicalActor(1, mean_weight=0.25, std=0.5)
    actor_std = policy._ppo_actor_std_parameter()
    mean_parameters = tuple(
        parameter
        for parameter in policy.actor_adapt.parameters()
        if parameter is not actor_std
    )
    policy.actor_optimizer = torch.optim.Adam(mean_parameters, lr=3.0e-3)
    policy.actor_std_optimizer = torch.optim.Adam((actor_std,), lr=1.0e-5)
    policy._fastsac_q_action_scale = torch.ones(1)
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.target_entropy = -1.0
    policy.actor_std_update_count = 0
    return policy


def test_physical_checkpoint_metadata_names_its_distribution_and_backend():
    policy = _physical_checkpoint_policy(90)

    state = policy._fastsac_checkpoint_state()

    assert state["action_distribution"] == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    assert state["actor_backend"] == PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND
    assert "physical_joint_std_gaussian" in state["actor_learning_semantics"]
    assert "physical_joint_std_gaussian" in state["entropy_semantics"]


def test_physical_checkpoint_round_trips_separate_std_optimizer_and_next_step():
    source = _physical_checkpoint_policy(95)
    mean_parameters = tuple(
        parameter
        for group in source.actor_optimizer.param_groups
        for parameter in group["params"]
    )
    _optimizer_step(mean_parameters, source.actor_optimizer)
    _apply_physical_std_gradient(source, 0.2)
    _apply_physical_std_gradient(source, -0.1)
    state = copy.deepcopy(source._fastsac_checkpoint_state())

    assert isinstance(
        state["optimizer_resume_state"]["actor_std_optimizer"], dict
    )
    assert state["actor_std_update_count"] == 2

    restored = _physical_checkpoint_policy(195)
    restored._load_fastsac_checkpoint_state(state)

    assert torch.equal(
        restored._ppo_actor_std_parameter(), source._ppo_actor_std_parameter()
    )
    assert restored.actor_std_update_count == 2
    _assert_nested_equal(
        restored.actor_optimizer.state_dict(), source.actor_optimizer.state_dict()
    )
    _assert_nested_equal(
        restored.actor_std_optimizer.state_dict(),
        source.actor_std_optimizer.state_dict(),
    )
    _apply_physical_std_gradient(source, 0.15)
    _apply_physical_std_gradient(restored, 0.15)
    assert torch.equal(
        restored._ppo_actor_std_parameter(), source._ppo_actor_std_parameter()
    )
    assert restored.actor_std_update_count == source.actor_std_update_count == 3


def test_checkpoint_seam_round_trips_sac_state_and_both_independent_rngs():
    source = _checkpoint_policy(100)
    _optimizer_step(
        tuple(source.actor_adapt.parameters())
        + tuple(source.bc_dagger_sac_adapter.parameters()),
        source.actor_optimizer,
    )
    _optimizer_step(source.qnet.parameters(), source.critic_optimizer)
    _optimizer_step((source.log_alpha,), source.alpha_optimizer)
    _optimizer_step(source.adapt_probe.parameters(), source.opt_adapt)
    source.actor_update_count = 7
    source.critic_update_count = 11
    source.alpha_update_count = 13
    source.q_update_row_credit = 123.0
    source.dagger_rollout_count = 17
    source.dagger_environment_steps = 19
    source.teacher_prefill_rollout_count = 3
    source.teacher_prefill_environment_steps = 23
    source._last_fastsac_diagnostics = {"fastsac/alpha": 0.25}
    for draw_count, generator in enumerate(
        (
            source.dagger_rng,
            source.q_rng,
            source.sac_action_rng,
            source.sac_rollout_rng,
            source.teacher_perception_rng,
        ),
        start=1,
    ):
        torch.rand(draw_count, generator=generator)

    state = copy.deepcopy(source._fastsac_checkpoint_state())
    assert "actor_target" not in state
    assert "target_policy_rng_state" not in state
    assert "collector_exploration_rng_state" not in state
    assert "sac_action_rng_state" in state
    assert "sac_rollout_rng_state" in state
    assert "teacher_perception_rng_state" in state

    restored = _checkpoint_policy(900)
    restored._load_fastsac_checkpoint_state(state)

    for module_name in (
        "actor_adapt",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
    ):
        _assert_nested_equal(
            getattr(source, module_name).state_dict(),
            getattr(restored, module_name).state_dict(),
        )
    for optimizer_name in (
        "actor_optimizer",
        "critic_optimizer",
        "alpha_optimizer",
        "opt_adapt",
    ):
        _assert_nested_equal(
            getattr(source, optimizer_name).state_dict(),
            getattr(restored, optimizer_name).state_dict(),
        )
    assert torch.equal(source.log_alpha, restored.log_alpha)
    assert restored.actor_target is None
    assert restored.actor_update_count == 7
    assert restored.critic_update_count == 11
    assert restored.alpha_update_count == 13
    assert restored.q_update_row_credit == pytest.approx(123.0)
    assert restored.dagger_rollout_count == 17
    assert restored.dagger_environment_steps == 19
    assert restored.teacher_prefill_rollout_count == 3
    assert restored.teacher_prefill_environment_steps == 23
    assert restored._last_fastsac_diagnostics == source._last_fastsac_diagnostics

    for name in (
        "dagger_rng",
        "q_rng",
        "sac_action_rng",
        "sac_rollout_rng",
        "teacher_perception_rng",
    ):
        source_generator = getattr(source, name)
        restored_generator = getattr(restored, name)
        assert torch.equal(source_generator.get_state(), restored_generator.get_state())
        assert torch.equal(
            torch.rand(8, generator=source_generator),
            torch.rand(8, generator=restored_generator),
        )


def _install_tiny_fastsac_inference_perception_stack(policy, seed: int) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for name in PRETRAINED_PERCEPTION_MODULES:
            setattr(policy, name, nn.Linear(2, 2))


@pytest.mark.parametrize(
    ("legacy_v3", "checkpoint_version"),
    (
        (False, CHECKPOINT_VERSION),
        (False, PREVIOUS_CHECKPOINT_VERSION),
        (False, 5),
        (False, 4),
        (True, 3),
    ),
)
def test_fastsac_inference_loader_restores_models_without_training_state(
    legacy_v3, checkpoint_version
):
    source = _checkpoint_policy(1300)
    _install_tiny_fastsac_inference_perception_stack(source, 1301)
    _optimizer_step(
        tuple(source.actor_adapt.parameters())
        + tuple(source.bc_dagger_sac_adapter.parameters()),
        source.actor_optimizer,
    )
    _optimizer_step(source.qnet.parameters(), source.critic_optimizer)
    _optimizer_step((source.log_alpha,), source.alpha_optimizer)
    _optimizer_step(source.adapt_probe.parameters(), source.opt_adapt)
    source.actor_update_count = 17
    source.critic_update_count = 19
    source.alpha_update_count = 23
    source.dagger_rollout_count = 29
    source.dagger_environment_steps = 31
    source.teacher_prefill_rollout_count = 37
    source.teacher_prefill_environment_steps = 41
    for generator in (
        source.dagger_rng,
        source.q_rng,
        source.sac_action_rng,
        source.sac_rollout_rng,
        source.teacher_perception_rng,
    ):
        torch.rand(7, generator=generator)

    state = {
        name: copy.deepcopy(module.state_dict())
        for name, module in source.named_children()
    }
    state.update(copy.deepcopy(source._fastsac_checkpoint_state()))
    state.update(
        {
            "last_phase": "finetune",
            "last_iter": 79,
            "action_contract": copy.deepcopy(source._fastsac_action_contract),
            "perception_initialization": {"loaded": True, "mode": "test"},
        }
    )
    state["checkpoint_version"] = checkpoint_version
    if legacy_v3:
        # V3 stored the effective log std directly and allowed the learned
        # tensor to move beyond the hard-clamp ceiling.
        state["checkpoint_version"] = 3
        state["actor_backend"] = (
            "ppo_vel_normalized_std_tanh_bounded_fastsac_bc_v1"
        )
        state["dagger_backend_config"] = {
            "sac_log_std_min": -10.0,
            "sac_log_std_max": -3.0,
        }
        state["bc_dagger_sac_adapter"] = {"log_std": torch.tensor([5.0])}

    restored = _checkpoint_policy(2300)
    _install_tiny_fastsac_inference_perception_stack(restored, 2301)
    restored._freeze_legacy_actor_std = lambda: None
    progress = []
    restored.env = SimpleNamespace(set_progress=progress.append)
    restored.actor_update_count = 101
    restored.critic_update_count = 103
    restored.alpha_update_count = 107
    restored.dagger_rollout_count = 109
    restored.dagger_environment_steps = 113
    restored.teacher_prefill_rollout_count = 127
    restored.teacher_prefill_environment_steps = 131
    optimizer_before = {
        name: copy.deepcopy(getattr(restored, name).state_dict())
        for name in (
            "actor_optimizer",
            "critic_optimizer",
            "alpha_optimizer",
            "opt_adapt",
        )
    }
    rng_before = {
        name: getattr(restored, name).get_state().clone()
        for name in (
            "dagger_rng",
            "q_rng",
            "sac_action_rng",
            "sac_rollout_rng",
            "teacher_perception_rng",
        )
    }

    failed = restored.load_inference_state_dict(state)

    assert failed == []
    for name in (
        "actor_adapt",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
        *PRETRAINED_PERCEPTION_MODULES,
    ):
        if name == "bc_dagger_sac_adapter" and legacy_v3:
            assert restored._bounded_log_std().item() == pytest.approx(
                -3.0, abs=1.0e-5
            )
            assert torch.isfinite(restored.bc_dagger_sac_adapter.log_std).all()
        else:
            _assert_nested_equal(
                getattr(restored, name).state_dict(),
                getattr(source, name).state_dict(),
            )
        assert not any(
            parameter.requires_grad
            for parameter in getattr(restored, name).parameters()
        )
    assert torch.equal(restored.log_alpha, source.log_alpha)
    assert progress == [79]
    assert (
        restored.actor_update_count,
        restored.critic_update_count,
        restored.alpha_update_count,
        restored.dagger_rollout_count,
        restored.dagger_environment_steps,
        restored.teacher_prefill_rollout_count,
        restored.teacher_prefill_environment_steps,
    ) == (101, 103, 107, 109, 113, 127, 131)
    for name, expected in optimizer_before.items():
        _assert_nested_equal(getattr(restored, name).state_dict(), expected)
    for name, expected in rng_before.items():
        assert torch.equal(getattr(restored, name).get_state(), expected)
    assert restored.actor_target is None
    assert restored._teacher_prefill_complete is True
    assert restored._teacher_perception_warmup_complete is True

    with pytest.raises(ValueError, match="same-stage resume"):
        restored.load_state_dict(state)


@pytest.mark.parametrize(
    ("version", "backend", "message"),
    (
        (2, ACTOR_BACKEND, "version mismatch"),
        (
            CHECKPOINT_VERSION,
            "ppo_vel_direct_raw_gaussian_fastsac_bc_v1",
            "backend mismatch",
        ),
    ),
)
def test_fastsac_inference_rejects_raw_std_checkpoint_contract_before_loading(
    monkeypatch, version, backend, message
):
    policy = _bare_policy()
    policy._fastsac_action_contract = {
        "joint_names": ["unit_joint"],
        "fingerprint": "sha256:raw-unit-test",
    }
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": version,
        "actor_backend": backend,
        "action_contract": copy.deepcopy(policy._fastsac_action_contract),
    }
    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda *args, **kwargs: pytest.fail(
            "incompatible checkpoint must be rejected before module loading"
        ),
    )

    with pytest.raises(ValueError, match=message):
        policy.load_inference_state_dict(state)


@pytest.mark.parametrize("saved_fingerprint", (None, "sha256:different"))
def test_fastsac_inference_requires_exact_bounded_action_contract_fingerprint(
    monkeypatch, saved_fingerprint
):
    policy = _bare_policy()
    policy._fastsac_action_contract = {
        "joint_names": ["unit_joint"],
        "fingerprint": "sha256:raw-unit-test",
    }
    saved_contract = {"joint_names": ["unit_joint"]}
    if saved_fingerprint is not None:
        saved_contract["fingerprint"] = saved_fingerprint
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": CHECKPOINT_VERSION,
        "actor_backend": ACTOR_BACKEND,
        "action_contract": saved_contract,
    }
    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda *args, **kwargs: pytest.fail(
            "invalid contract must be rejected before module loading"
        ),
    )

    message = (
        "lacks a contract fingerprint" if saved_fingerprint is None else "mismatch"
    )
    with pytest.raises(ValueError, match=message):
        policy.load_inference_state_dict(state)
