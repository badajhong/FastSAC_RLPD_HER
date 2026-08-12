from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
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
    TRAINING_ALGORITHM,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
    _DeterministicFastSACStudentEvalPolicy,
    _DistributionalFastSACDaggerRolloutPolicy,
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


def _install_unit_action_contract(policy) -> None:
    policy._fastsac_action_low = torch.tensor([-1.0])
    policy._fastsac_action_high = torch.tensor([1.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([1.0])
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([1.0])
    policy._fastsac_action_log_scale_sum = 0.0
    policy._fastsac_entropy_reference_log_scale_sum = 0.0


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
    assert cfg.policy_delay == cfg.sac_policy_frequency


def test_config_allows_explicit_pure_sac_ablation_without_inherited_td3_eta():
    cfg = DistributionalFastSACTeacherBCConfig(lambda_bc=0.0, eta_sac=1.0)

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_global_log_std_builds_direct_execution_distribution_on_plus_minus_20():
    adapter = _BCDaggerSACAdapter(
        action_dim=2,
        initial_log_std=torch.log(torch.tensor([0.01, 0.02])),
        device="cpu",
    )
    latent_mean = torch.tensor([[0.0, 0.5], [-0.5, 1.0]])
    scale = adapter.log_std.exp().expand_as(latent_mean)
    dist = FastSACTanhNormal(
        latent_mean,
        scale,
        low=torch.full((2,), -20.0),
        high=torch.full((2,), 20.0),
        event_dims=1,
    )

    assert tuple(adapter.parameters()) == (adapter.log_std,)
    assert adapter.log_std.shape == (2,)
    assert torch.equal(dist.low, torch.full((2,), -20.0))
    assert torch.equal(dist.high, torch.full((2,), 20.0))
    assert torch.allclose(dist.mean, latent_mean.tanh() * 20.0)
    first, first_log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(71)
    )
    second, second_log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(72)
    )
    assert not torch.equal(first, second)
    assert torch.isfinite(first_log_prob).all()
    assert torch.isfinite(second_log_prob).all()
    assert (first >= -20.0).all() and (first <= 20.0).all()


def test_backend_distribution_seam_uses_global_std_and_direct_execution_support():
    policy = _bare_policy(sac_log_std_min=-8.0, sac_log_std_max=-2.0)
    policy._fastsac_action_low = torch.full((2,), -20.0)
    policy._fastsac_action_high = torch.full((2,), 20.0)
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=2,
        initial_log_std=torch.tensor([-4.0, -3.0]),
        device="cpu",
    )
    latent = torch.tensor([[0.0, 0.5], [-0.5, 1.0]])

    dist = policy._sac_dist_from_latent(latent)

    assert torch.equal(dist.loc, latent)
    assert torch.allclose(
        dist.scale,
        policy.bc_dagger_sac_adapter.log_std.exp().expand_as(latent),
    )
    assert torch.equal(dist.low, torch.full((2,), -20.0))
    assert torch.equal(dist.high, torch.full((2,), 20.0))
    assert torch.allclose(dist.mean, latent.tanh() * 20.0)


def test_exact_bc_on_deterministic_latent_mean_has_no_log_std_gradient():
    latent_mean = nn.Parameter(torch.tensor([[0.2], [-0.4], [0.8]]))
    adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.tensor(-4.0),
        device="cpu",
    )
    teacher = torch.tensor([[0.5], [-0.25], [9.0]])
    valid = torch.tensor([True, True, False])

    # Creating the executable stochastic policy must not make its variance part
    # of the supervised objective. BC owns only the noise-free latent location.
    FastSACTanhNormal(
        latent_mean,
        adapter.log_std.exp().expand_as(latent_mean),
        low=-20.0,
        high=20.0,
        event_dims=1,
    )
    loss = _exact_teacher_bc_loss(
        latent_mean,
        teacher,
        valid,
        torch.tensor([0.0]),
        torch.tensor([20.0]),
        huber_delta=1.0,
    )
    loss.backward()

    assert latent_mean.grad is not None
    assert latent_mean.grad.abs().sum().item() > 0.0
    assert adapter.log_std.grad is None
    assert torch.equal(
        latent_mean.grad[~valid], torch.zeros_like(latent_mean.grad[~valid])
    )


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
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.tensor(log_std),
        device="cpu",
    )

    def actor_latent(owner, observations):
        return owner.actor_adapt(observations)

    def actor_dist(owner, observations):
        loc = owner._actor_latent_from_flat(observations)
        scale = (
            owner.bc_dagger_sac_adapter.log_std.clamp(
                float(owner.cfg.sac_log_std_min),
                float(owner.cfg.sac_log_std_max),
            )
            .exp()
            .expand_as(loc)
        )
        return FastSACTanhNormal(
            loc,
            scale,
            low=owner._fastsac_action_low,
            high=owner._fastsac_action_high,
            event_dims=1,
        )

    policy._actor_latent_from_flat = MethodType(actor_latent, policy)
    policy._actor_dist_from_flat = MethodType(actor_dist, policy)
    policy._fastsac_actor_parameters = tuple(policy.actor_adapt.parameters()) + tuple(
        policy.bc_dagger_sac_adapter.parameters()
    )
    return policy


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
    loc = expected_actor(batch["observations"])
    expected_dist = FastSACTanhNormal(
        loc,
        expected_adapter.log_std.exp().expand_as(loc),
        low=-1.0,
        high=1.0,
        event_dims=1,
    )
    sampled_action, log_prob = expected_dist.rsample_with_log_prob(
        generator=expected_rng
    )
    expected_heads = expected_q.values(
        expected_q(batch["critic_observations"], sampled_action)
    )
    expected_pessimistic_q = expected_heads.min(dim=0).values
    expected_sac = (
        policy.log_alpha.detach().exp() * log_prob - expected_pessimistic_q
    ).mean()
    expected_bc = _exact_teacher_bc_loss(
        loc,
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        batch[DAGGER_TEACHER_ACTION_VALID_KEY],
        policy._fastsac_actor_action_center,
        policy._fastsac_actor_action_scale,
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


def _rollout_owner(*, prefill: bool, beta: float = 0.5):
    action_dim = 2
    batch_size = 64
    latent = torch.zeros(batch_size, action_dim)
    teacher = torch.zeros_like(latent)
    cfg = SimpleNamespace(
        teacher_prefill_max_rollouts=1,
        dagger_control_mode="beta",
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
        dagger_safe_takeover_rms=0.006,
        dagger_safe_release_rms=0.004,
        dagger_safe_min_teacher_steps=8,
        q_action_input_gain=1.0,
        collector_exploration_noise_std=0.0,
        collector_exploration_noise_clip=0.0,
        sac_log_std_min=-10.0,
        sac_log_std_max=-2.0,
    )
    owner = SimpleNamespace(cfg=cfg)
    owner.teacher_prefill_rollout_count = 0 if prefill else 1
    owner.dagger_rollout_count = 0
    owner.dagger_rng = torch.Generator().manual_seed(17)
    owner.sac_rollout_rng = torch.Generator().manual_seed(18)
    owner.sac_action_rng = torch.Generator().manual_seed(19)
    owner._student_latent = lambda td: latent.clone()
    owner._teacher_action = lambda td: teacher.clone()
    owner._project_execution_action = lambda action: action.clamp(-20.0, 20.0)
    owner._student_action_from_latent = lambda value: value.tanh() * 20.0
    owner._teacher_prefill_active = lambda: prefill
    owner._effective_control_mode = lambda: "beta"
    owner._teacher_mixture_probability = lambda: beta
    owner._safe_teacher_control_enabled = lambda: False
    owner._fastsac_action_low = torch.full((action_dim,), -20.0)
    owner._fastsac_action_high = torch.full((action_dim,), 20.0)
    owner._fastsac_q_action_center = torch.zeros(action_dim)
    owner._fastsac_q_action_scale = torch.ones(action_dim)

    def stochastic_dist(value):
        return FastSACTanhNormal(
            value,
            torch.full_like(value, 0.15),
            low=owner._fastsac_action_low,
            high=owner._fastsac_action_high,
            event_dims=1,
        )

    # Keep the fixture tolerant to a private naming change while the public
    # contract remains one direct stochastic distribution from Student latent.
    owner._student_sac_dist_from_latent = stochastic_dist
    owner._sac_dist_from_latent = stochastic_dist
    owner._student_distribution_from_latent = stochastic_dist
    owner._q_action_input = lambda action: action
    return owner, latent, teacher


def test_main_rollout_keeps_teacher_rows_exact_and_samples_only_student_behavior():
    owner, latent, teacher = _rollout_owner(prefill=False, beta=0.5)
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(latent.shape[0], dtype=torch.bool)},
        batch_size=[latent.shape[0]],
    )
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()
    learning_rng_before = owner.sac_action_rng.get_state().clone()

    result = policy(td)

    student = result[DAGGER_IS_STUDENT_ACTION_KEY]
    assert student.any() and (~student).any()
    assert torch.equal(result[DAGGER_TEACHER_ACTION_KEY], teacher)
    assert result[DAGGER_TEACHER_ACTION_VALID_KEY].all()
    assert torch.equal(result[ACTION_KEY][~student], teacher[~student])
    assert torch.equal(result[TD3_NOISE_FREE_STUDENT_ACTION_KEY], latent.tanh() * 20.0)
    assert not torch.equal(
        result[ACTION_KEY][student],
        result[TD3_NOISE_FREE_STUDENT_ACTION_KEY][student],
    )
    # Safety compares Teacher with the noise-free mean, never a lucky/unlucky
    # exploration draw. Here those means are identical, so discrepancy is zero.
    assert torch.equal(
        result[DAGGER_ACTION_DISCREPANCY_RMS_KEY],
        torch.zeros(latent.shape[0]),
    )
    assert not torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)
    assert torch.equal(owner.sac_action_rng.get_state(), learning_rng_before)


def test_teacher_only_prefill_is_bitwise_exact_and_does_not_draw_sac_noise():
    owner, latent, teacher = _rollout_owner(prefill=True, beta=0.0)
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(latent.shape[0], dtype=torch.bool)},
        batch_size=[latent.shape[0]],
    )
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()

    result = policy(td)

    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], teacher)
    assert torch.equal(result[TD3_EXPLORATORY_STUDENT_ACTION_KEY], teacher)
    assert torch.equal(result[TD3_COLLECTOR_NOISE_KEY], torch.zeros_like(teacher))
    assert torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)


def test_deterministic_eval_uses_mean_without_advancing_any_sac_rng():
    owner, latent, _ = _rollout_owner(prefill=False, beta=0.0)
    owner.cfg.use_object_adapt = False
    owner.depth_feature_dim = 1
    owner.adapt_ema = nn.Identity()
    owner.actor_adapt = nn.Identity()
    eval_policy = _DeterministicFastSACStudentEvalPolicy(owner)
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()
    learning_rng_before = owner.sac_action_rng.get_state().clone()
    td = TensorDict({}, batch_size=[latent.shape[0]])

    first = eval_policy(td.clone())[ACTION_KEY]
    second = eval_policy(td.clone())[ACTION_KEY]

    assert torch.equal(first, second)
    assert torch.equal(first, latent.tanh() * 20.0)
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
    policy = _bare_policy(sac_use_autotune=True)
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
    return policy


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
