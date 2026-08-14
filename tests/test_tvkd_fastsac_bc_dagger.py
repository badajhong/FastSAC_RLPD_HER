from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    DistributionalFastSACTeacherBC,
)
from active_adaptation.learning.ppo.common import Actor
from active_adaptation.learning.ppo.fastsac_vel import _BCDaggerSACAdapter
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from active_adaptation.learning.ppo.td3_bc_dagger import (
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    _project_c51_probabilities,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    ACTOR_BACKEND as TVKD_ACTOR_BACKEND,
    CHECKPOINT_VERSION as TVKD_CHECKPOINT_VERSION,
    SOURCE_FAILURE_TEACHER,
    SOURCE_STUDENT,
    SOURCE_UNIFORM_TEACHER,
    FrozenTeacherValueWrapper,
    TeacherValueBCScheduler,
    TRAINING_ALGORITHM as TVKD_TRAINING_ALGORITHM,
    TVKDDistributionalFastSACTeacherBC,
    _source_ids_from_batch,
    compute_source_separated_bc_losses,
    compute_teacher_value_terms,
)
from scripts.TVKD_fasSAC_bc_dagger import (
    _prepare_tvkd_checkpoint,
    validate_tvkd_fastsac_bc_dagger_config,
)
from scripts.helpers import _fill_replayless_inference_algo_defaults


class _KeyedTeacherValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, td: TensorDict):
        # The input flat layout in the test is [z, a].  This formula detects a
        # wrapper which accidentally forwards that flat tensor without keys.
        scalar = self.gain * (10.0 * td["a"][:, 0] + td["z"][:, 0])
        td["state_value"] = torch.stack((scalar, scalar + 1.0), dim=-1)
        return td


class _AffineValueNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor([2.0, 3.0]))

    def denormalize(self, value):
        return value + self.offset


class _FixedDist:
    def __init__(self, batch_size: int, action: float = 0.0, log_prob: float = 0.0):
        self.loc = torch.zeros(batch_size, 1)
        self.mean = torch.zeros(batch_size, 1)
        self.action = float(action)
        self.log_prob = float(log_prob)

    def rsample_with_log_prob(self, generator=None):
        if generator is not None:
            torch.rand((), generator=generator)
        return (
            self.loc.new_full(self.loc.shape, self.action),
            self.loc.new_full((self.loc.shape[0],), self.log_prob),
        )


class _TableTwin(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.register_buffer("first", first.log())
        self.register_buffer("second", second.log())
        self.register_buffer("support", torch.tensor([-10.0, 0.0, 10.0]))

    def forward(self, observations, actions):
        del actions
        count = observations.shape[0]
        return torch.stack((self.first[:count], self.second[:count]), dim=0)


class _ActionSensitiveTwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor([1.0, -2.0]))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        del observations
        score = self.slope[:, None] * actions[:, 0][None, :]
        zeros = torch.zeros_like(score)
        return torch.stack((-score, zeros, score), dim=-1)


def _scheduler(**overrides):
    kwargs = {
        "lambda_min": 0.05,
        "lambda_max": 1.0,
        "margin": 0.0,
        "temperature": 1.0,
        "scale_ema_decay": 0.0,
        "risk_ema_decay": 0.0,
        "warmup_updates": 0,
        "min_student_samples": 1,
        "eps": 1e-6,
    }
    kwargs.update(overrides)
    return TeacherValueBCScheduler(**kwargs)


def test_scheduler_orders_bad_neutral_and_good_student_transitions():
    bad = _scheduler()
    neutral = _scheduler()
    good = _scheduler()

    bad_lambda = bad.update(torch.full((32,), -10.0))
    neutral_lambda = neutral.update(torch.zeros(32))
    good_lambda = good.update(torch.full((32,), 10.0))

    assert bad.risk_ema > neutral.risk_ema > good.risk_ema
    assert bad_lambda > neutral_lambda > good_lambda
    assert bad.num_updates == neutral.num_updates == good.num_updates == 1


def test_scheduler_warmup_minimum_population_and_state_round_trip():
    scheduler = _scheduler(
        scale_ema_decay=0.9,
        risk_ema_decay=0.8,
        warmup_updates=2,
        min_student_samples=4,
    )
    before = scheduler.state_dict()
    assert scheduler.update(torch.tensor([-3.0, -2.0])) == pytest.approx(1.0)
    assert scheduler.num_updates == 0
    assert scheduler.residual_scale_ema == before["residual_scale_ema"]
    assert scheduler.risk_ema == before["risk_ema"]

    assert scheduler.update(torch.full((4,), -3.0)) == pytest.approx(1.0)
    assert scheduler.update(torch.full((4,), 3.0)) == pytest.approx(1.0)
    adaptive = scheduler.update(torch.full((4,), 3.0))
    assert adaptive < 1.0

    restored = _scheduler(
        scale_ema_decay=0.9,
        risk_ema_decay=0.8,
        warmup_updates=2,
        min_student_samples=4,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()


def test_frozen_teacher_value_restores_keys_denormalizes_and_sums_groups():
    actor = nn.Linear(1, 1)
    value = _KeyedTeacherValue()
    normalizer = _AffineValueNorm()
    wrapper = FrozenTeacherValueWrapper(
        actor,
        value,
        normalizer,
        critic_keys=("z", "a"),
        critic_widths=(1, 1),
    )

    # Flat rows are [z, a]. Values by group before denormalization are
    # [10*a+z, 10*a+z+1], then offsets [2,3] are added and groups are summed.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = wrapper.get_frozen_teacher_value(
            torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        )
    assert torch.equal(result, torch.tensor([70.0, 156.0]))
    assert result.shape == (2,)
    assert result.dtype == torch.float32
    assert result.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in actor.parameters())
    assert all(parameter.requires_grad is False for parameter in value.parameters())
    assert actor.training is False
    assert value.training is False
    assert normalizer.training is False

    student = nn.Parameter(torch.tensor(2.0))
    (result.sum() + student.square()).backward()
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in value.parameters())


def test_teacher_value_terms_clip_potential_before_difference_and_keep_raw_td():
    def value(obs):
        return obs[:, 0]

    current = torch.tensor([[20.0], [-4.0], [2.0]])
    next_observation = torch.tensor([[-20.0], [8.0], [9.0]])
    reward = torch.tensor([1.0, 2.0, 3.0])
    bootstrap = torch.tensor([1.0, 0.0, 1.0])
    discount = torch.tensor([0.9, 0.9, 0.5])
    terms = compute_teacher_value_terms(
        value,
        current,
        next_observation,
        reward,
        bootstrap,
        discount,
        tvkd_lambda=0.25,
        potential_clip=5.0,
    )

    expected_potential = torch.tensor([0.9 * -5.0 - 5.0, 4.0, 0.5 * 5.0 - 2.0])
    expected_td = reward + discount * bootstrap * next_observation[:, 0] - current[:, 0]
    assert torch.allclose(terms.potential_delta, expected_potential)
    assert torch.allclose(terms.shaped_reward, reward + 0.25 * expected_potential)
    # TD residual uses raw reward and the *unclipped* frozen value.
    assert torch.allclose(terms.teacher_td_residual, expected_td)


def test_tvkd_lambda_zero_keeps_raw_reward_exactly():
    terms = compute_teacher_value_terms(
        lambda observation: observation[:, 0],
        torch.tensor([[2.0], [3.0]]),
        torch.tensor([[8.0], [9.0]]),
        torch.tensor([1.5, -2.0]),
        torch.tensor([1.0, 0.0]),
        0.99,
        tvkd_lambda=0.0,
    )
    assert torch.equal(terms.shaped_reward, torch.tensor([1.5, -2.0]))


def test_source_separated_bc_is_empty_safe_pretanh_and_does_not_touch_log_std():
    latent = nn.Parameter(torch.tensor([[0.0], [0.2], [-0.3], [0.4]]))
    log_std = nn.Parameter(torch.tensor(-2.0))
    teacher_action = torch.tensor([[0.0], [0.5], [-0.5], [1.0]])
    valid = torch.tensor([True, True, True, False])
    source = torch.tensor(
        [
            SOURCE_STUDENT,
            SOURCE_UNIFORM_TEACHER,
            SOURCE_FAILURE_TEACHER,
            SOURCE_STUDENT,
        ]
    )
    teacher_loss, student_loss = compute_source_separated_bc_losses(
        latent,
        teacher_action,
        valid,
        source,
        action_center=torch.tensor([0.0]),
        action_half_range=torch.tensor([1.0]),
        huber_delta=1.0,
    )
    expected_student = F.smooth_l1_loss(latent[:1], torch.zeros(1, 1))
    teacher_target = torch.atanh(torch.tensor([[0.5], [-0.5]]))
    expected_teacher = F.smooth_l1_loss(latent[1:3], teacher_target)
    assert torch.allclose(student_loss, expected_student)
    assert torch.allclose(teacher_loss, expected_teacher)

    (teacher_loss + student_loss).backward()
    assert latent.grad is not None
    assert log_std.grad is None

    empty_teacher, all_student = compute_source_separated_bc_losses(
        latent.detach().requires_grad_(),
        teacher_action,
        valid,
        torch.full((4,), SOURCE_STUDENT),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        1.0,
    )
    assert torch.isfinite(empty_teacher)
    assert empty_teacher.item() == 0.0
    assert torch.isfinite(all_student)


def test_sample_time_source_ids_preserve_uniform_fallback_semantics():
    valid = torch.ones(5, dtype=torch.bool)
    source = _source_ids_from_batch(
        {
            DAGGER_TEACHER_ACTION_VALID_KEY: valid,
            DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([False, True, True, False, True]),
            FAILURE_PHASE_TEACHER_SOURCE_KEY: torch.tensor(
                [False, False, True, False, False]
            ),
        }
    )
    assert source.tolist() == [
        SOURCE_STUDENT,
        SOURCE_UNIFORM_TEACHER,
        SOURCE_FAILURE_TEACHER,
        SOURCE_STUDENT,
        SOURCE_UNIFORM_TEACHER,
    ]


def test_disabled_tvkd_target_delegates_to_exact_baseline_path(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=False,
        tvkd_lambda=0.25,
    )
    sentinel = (
        torch.tensor([[1.0]]),
        {"baseline": torch.tensor(2.0)},
        torch.tensor([3.0]),
    )

    def baseline(self, batch):
        assert batch == {"unchanged": True}
        return sentinel

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_distributional_fastsac_target",
        baseline,
    )
    result = policy._distributional_fastsac_target({"unchanged": True})
    assert result is sentinel


@pytest.mark.parametrize(
    ("use_adaptive", "minimum", "maximum"),
    [(False, 0.05, 1.0), (True, 0.7, 0.7)],
)
def test_equivalent_actor_modes_delegate_before_new_source_processing(
    monkeypatch, use_adaptive, minimum, maximum
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_adaptive_student_bc=use_adaptive,
        lambda_bc=0.7,
        student_bc_lambda_min=minimum,
        student_bc_lambda_max=maximum,
    )
    sentinel = {"baseline_actor_loss": torch.tensor(4.0)}

    def baseline(self, batch):
        assert batch == {"no_tvkd_source_fields": True}
        return sentinel

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_actor_update",
        baseline,
    )
    result = policy._actor_update({"no_tvkd_source_fields": True})
    assert result is sentinel


def test_adaptive_bc_actor_step_updates_mean_but_not_production_log_std():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_adaptive_student_bc=True,
        student_bc_lambda_min=0.05,
        student_bc_lambda_max=1.0,
        lambda_bc=1.0,
        eta_sac=0.0,
        dagger_actor_huber_delta=1.0,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
        action_support_clip=20.0,
    )
    policy.actor_adapt = nn.Linear(1, 1, bias=False)
    policy.actor_adapt.weight.data.fill_(0.25)
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.tensor(-1.0),
        device="cpu",
    )

    def actor_mean(owner, observations):
        return owner.actor_adapt(observations)

    policy._actor_mean_from_flat = MethodType(actor_mean, policy)
    policy.qnet = _ActionSensitiveTwin()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy._fastsac_action_low = torch.tensor([-20.0])
    policy._fastsac_action_high = torch.tensor([20.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([20.0])
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([1.0])
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.actor_optimizer = torch.optim.SGD(
        tuple(policy.actor_adapt.parameters())
        + tuple(policy.bc_dagger_sac_adapter.parameters()),
        lr=0.05,
    )
    policy.critic_optimizer = torch.optim.SGD(policy.qnet.parameters(), lr=0.05)
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    policy.sac_actor_update_count = 0
    policy.student_bc_scheduler = SimpleNamespace(current_lambda_bc_student=0.8)
    log_std_before = policy.bc_dagger_sac_adapter.log_std.detach().clone()
    actor_before = policy.actor_adapt.weight.detach().clone()
    batch = {
        "observations": torch.tensor([[1.0], [2.0]]),
        "critic_observations": torch.ones(2, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[5.0], [-4.0]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.ones(2, dtype=torch.bool),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.zeros(2, dtype=torch.bool),
        FAILURE_PHASE_TEACHER_SOURCE_KEY: torch.zeros(2, dtype=torch.bool),
    }

    metrics = policy._actor_update(batch)

    assert metrics["bc_teacher_loss"].item() == 0.0
    assert metrics["bc_student_loss"].item() > 0.0
    assert not torch.equal(policy.actor_adapt.weight, actor_before)
    assert torch.equal(policy.bc_dagger_sac_adapter.log_std, log_std_before)
    assert torch.equal(
        policy.bc_dagger_sac_adapter.log_std.grad,
        torch.zeros_like(policy.bc_dagger_sac_adapter.log_std),
    )


def test_tvkd_c51_target_inserts_shaped_reward_once_and_stays_detached():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=True,
        tvkd_lambda=0.25,
        tvkd_potential_clip=None,
        gamma=1.0,
    )
    first = torch.tensor([[0.05, 0.15, 0.80], [0.05, 0.15, 0.80], [0.05, 0.15, 0.80]])
    second = torch.tensor([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05], [0.80, 0.15, 0.05]])
    policy.qnet_target = _TableTwin(first, second).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2)))
    policy.sac_action_rng = torch.Generator().manual_seed(19)
    policy._actor_dist_from_flat = lambda observations: _FixedDist(
        observations.shape[0]
    )
    policy._normalized_action_log_prob = lambda value: value
    policy._q_action_input = lambda action: action
    policy.get_frozen_teacher_value = lambda observation: observation[:, 0]
    batch = {
        "observations": torch.zeros(3, 1),
        "next_observations": torch.zeros(3, 1),
        "critic_observations": torch.tensor([[2.0], [3.0], [6.0]]),
        "next_critic_observations": torch.tensor([[4.0], [5.0], [8.0]]),
        "rewards": torch.ones(3),
        "dones": torch.tensor([False, True, True]),
        "truncations": torch.tensor([False, False, True]),
        "discounts": torch.ones(3),
    }
    raw_reward_before = batch["rewards"].clone()

    projected, metrics, _ = policy._distributional_fastsac_target(batch)
    # Ordinary and timeout rows bootstrap; the true terminal does not.
    # potential=[4-2, -3, 8-6], shaped reward=[1.5, .25, 1.5].
    expected_shaped = torch.tensor([1.5, 0.25, 1.5])
    expected, _, _ = _project_c51_probabilities(
        second,
        expected_shaped,
        torch.tensor([1.0, 0.0, 1.0]),
        torch.ones(3),
        policy.qnet_target.support,
    )
    assert torch.allclose(projected, expected)
    assert metrics["tvkd_shaped_reward_mean"].item() == pytest.approx(13.0 / 12.0)
    assert metrics["tvkd_potential_delta_mean"].item() == pytest.approx(1.0 / 3.0)
    assert torch.equal(batch["rewards"], raw_reward_before)
    assert projected.requires_grad is False
    assert projected.grad_fn is None


@pytest.mark.parametrize("prefill", [False, True])
def test_recent_scheduler_hook_selects_only_main_rollout_student_rows(
    monkeypatch, prefill
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy._teacher_prefill_active = lambda: prefill
    received = []
    policy._update_scheduler_from_recent_student = lambda chunks: received.append(
        chunks
    )
    transitions = {
        DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([True, False, True]),
        "critic_observations": torch.tensor([[1.0], [2.0], [3.0]]),
        "next_critic_observations": torch.tensor([[11.0], [12.0], [13.0]]),
        "rewards": torch.tensor([21.0, 22.0, 23.0]),
        "dones": torch.tensor([False, False, True]),
        "truncations": torch.tensor([False, False, True]),
    }

    def baseline_chunks(self, td):
        del self, td
        yield transitions

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_dagger_transition_chunks",
        baseline_chunks,
    )
    yielded = list(policy._dagger_transition_chunks(TensorDict({}, batch_size=[])))
    assert len(yielded) == 1
    assert yielded[0] is transitions
    assert len(received) == 1
    if prefill:
        assert received[0] == []
    else:
        assert len(received[0]) == 1
        recent = received[0][0]
        assert recent["rewards"].tolist() == [21.0, 23.0]
        assert recent["critic_observations"].squeeze(-1).tolist() == [1.0, 3.0]


def test_recent_student_scheduler_normalizes_once_and_uses_terminal_mask():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_adaptive_student_bc=True,
        gamma=0.9,
        student_bc_scheduler_eps=1e-6,
        student_bc_margin=0.0,
    )
    policy.q_critic_keys = ("critic",)
    policy._q_critic_widths = (1,)
    policy._vecnorm_snapshot = lambda: {"frozen": True}
    normalization_calls = []

    def normalize(value, keys, widths, snapshot):
        normalization_calls.append((value.clone(), keys, widths, snapshot))
        return value + 10.0

    policy._normalize_replay_flat = normalize
    policy.get_frozen_teacher_value = lambda observation: observation[:, 0]
    residuals = []
    policy.student_bc_scheduler = SimpleNamespace(
        update=lambda residual: residuals.append(residual.clone()) or 0.4,
        residual_scale_ema=2.0,
        last_risk_batch_mean=0.3,
        risk_ema=0.25,
    )
    chunks = [
        {
            "critic_observations": torch.tensor([[1.0], [2.0], [3.0]]),
            "next_critic_observations": torch.tensor([[4.0], [5.0], [6.0]]),
            "rewards": torch.zeros(3),
            "dones": torch.tensor([False, True, True]),
            "truncations": torch.tensor([False, False, True]),
        }
    ]

    policy._update_scheduler_from_recent_student(chunks)
    assert len(normalization_calls) == 2
    assert all(
        call[1:] == (("critic",), (1,), {"frozen": True})
        for call in normalization_calls
    )
    # normalized V=[11,12,13], V_next=[14,15,16]; bootstrap=[1,0,1]
    assert torch.allclose(residuals[0], torch.tensor([1.6, -12.0, 1.4]))
    assert policy._last_bc_scheduler_metrics["recent_student_sample_count"] == 3.0
    assert policy._last_bc_scheduler_metrics["student_bc_lambda"] == 0.4


def test_tvkd_inference_forces_checkpoint_value_norm_before_construction():
    cfg = OmegaConf.create({"algo": {"value_norm": False}})
    result = _fill_replayless_inference_algo_defaults(
        cfg,
        {
            "training_algorithm": "distributional_tvkd_fastsac_teacher_bc_v1",
            "dagger_backend_config": {"value_norm": True},
        },
        inference_only=True,
    )
    assert cfg.algo.value_norm is True
    assert "value_norm" in result["checkpoint"]


def test_tvkd_logging_surface_is_finite_without_any_optimizer_update(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(lambda_bc=1.0, use_adaptive_student_bc=True)
    policy._last_bc_scheduler_metrics = policy._empty_scheduler_metrics()
    policy._fastsac_rollout_critic_metrics = []
    policy._fastsac_rollout_actor_metrics = []
    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "train_op",
        lambda self, tensordict: {
            "fastsac/student_replay_rows_this_rollout": 0.0,
            "fastsac/student_source_fraction": 0.0,
            "fastsac/teacher_source_fraction": 1.0,
        },
    )

    info = policy.train_op(TensorDict({}, batch_size=[]))
    required = {
        "tvkd/teacher_value_mean",
        "tvkd/teacher_next_value_mean",
        "tvkd/potential_delta_mean",
        "tvkd/raw_reward_mean",
        "tvkd/shaped_reward_mean",
        "bc_scheduler/teacher_td_residual_mean",
        "bc_scheduler/residual_scale_ema",
        "bc_scheduler/risk_ema",
        "bc_scheduler/student_bc_lambda",
        "loss/critic",
        "loss/actor_total",
        "loss/actor_sac",
        "loss/bc_teacher",
        "loss/bc_student",
        "loss/alpha",
        "source/student_transition_count",
    }
    assert required.issubset(info)
    assert all(torch.isfinite(torch.as_tensor(info[key])) for key in required)


def test_tvkd_checkpoint_seam_saves_teacher_chain_and_restores_scheduler(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    policy.height_encoder = nn.Linear(1, 1)
    policy.cfg = SimpleNamespace(
        failure_phase_num_bins=4,
        train_dr_estimator=True,
    )
    policy.dr_estimator = nn.Linear(1, 1)
    policy.opt_dr_estimator = torch.optim.Adam(
        policy.dr_estimator.parameters(), lr=1e-3
    )
    policy.opt_dr_estimator.zero_grad(set_to_none=True)
    policy.dr_estimator(torch.ones(1, 1)).square().sum().backward()
    policy.opt_dr_estimator.step()
    policy.num_updates = 17
    policy.actor_update_count = 19
    policy.sac_actor_update_count = 19
    policy.alpha_update_count = 23
    policy.sac_alpha_update_count = 23
    policy._failure_phase_histogram = torch.tensor(
        [0.0, 1.0, 2.0, 0.0], dtype=torch.float64
    )
    policy._failure_phase_episode_count = 2
    policy._failure_phase_anchor_count = 3
    policy._failure_phase_uniform_fallback_rows = 4
    policy._failure_phase_focused_rows = 5
    policy._failure_histogram_device_cache = {}
    policy.student_bc_scheduler = _scheduler(
        scale_ema_decay=0.8,
        risk_ema_decay=0.7,
    )
    policy.student_bc_scheduler.update(torch.full((8,), -2.0))
    expected_scheduler = policy.student_bc_scheduler.state_dict()
    policy._last_tvkd_diagnostics = {"tvkd/shaped_reward_mean": 1.25}
    freeze_calls = []
    policy.teacher_value_wrapper = SimpleNamespace(
        freeze=lambda: freeze_calls.append(True)
    )

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_fastsac_checkpoint_state",
        lambda self: {
            "baseline_state": torch.tensor(1.0),
            "optimizer_resume_state": {},
        },
    )
    translated = []

    def baseline_load(self, state, *, load_modules=True):
        translated.append((state, load_modules))

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        baseline_load,
    )

    state = policy._fastsac_checkpoint_state()
    assert state["teacher_value_bc_scheduler"] == expected_scheduler
    assert set(state["frozen_teacher_state"]) == {
        "actor",
        "encoder_priv",
        "critic",
        "value_norm",
        "height_encoder",
    }
    assert torch.equal(
        state["failure_phase_curriculum_state"]["histogram"],
        policy._failure_phase_histogram,
    )
    expected_dr_optimizer = state["optimizer_resume_state"]["dr_estimator_optimizer"]
    assert expected_dr_optimizer is not None
    assert state["num_updates"] == 17
    assert state["sac_actor_update_count"] == 19
    assert state["sac_alpha_update_count"] == 23

    policy.student_bc_scheduler.update(torch.full((8,), 4.0))
    policy.opt_dr_estimator = torch.optim.Adam(
        policy.dr_estimator.parameters(), lr=0.25
    )
    assert policy.student_bc_scheduler.state_dict() != expected_scheduler
    policy._load_fastsac_checkpoint_state(state, load_modules=False)
    assert policy.student_bc_scheduler.state_dict() == expected_scheduler
    assert policy.num_updates == 17
    assert policy.sac_actor_update_count == 19
    assert policy.sac_alpha_update_count == 23
    restored_dr_optimizer = policy.opt_dr_estimator.state_dict()
    assert (
        restored_dr_optimizer["param_groups"] == expected_dr_optimizer["param_groups"]
    )
    for parameter_id, parameter_state in expected_dr_optimizer["state"].items():
        for name, expected in parameter_state.items():
            actual = restored_dr_optimizer["state"][parameter_id][name]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert freeze_calls == [True]
    assert translated[0][0]["training_algorithm"].startswith(
        "distributional_fastsac_teacher_bc"
    )
    assert translated[0][1] is False


def test_tvkd_resume_entrypoint_accepts_checkpoint_and_uses_additional_budget(
    tmp_path,
):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "fastsac_dagger_iterations=100",
                "algo.dagger_beta_start=1.0",
                "algo.dagger_beta_end=0.0",
                "algo.dagger_beta_decay_rollouts=500",
            ],
        )
    saved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    # Legacy TVKD v1 configs predate the explicit cadence provenance field but
    # already used one alpha update per Critic step.
    del saved_cfg.algo.sac_alpha_update_cadence
    checkpoint_path = tmp_path / "checkpoint_12.pt"
    rng_state = torch.Generator().manual_seed(11).get_state()
    module_names = (
        "actor",
        "actor_adapt",
        "encoder_priv",
        "critic",
        "value_norm",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
        "adapt_module",
        "adapt_ema",
        "object_adapt",
        "object_adapt_ema",
        "depth_cnn",
        "temporal_depth_gru",
        "temporal_depth_gru_ema",
    )
    policy_state = {name: {} for name in module_names}
    policy_state.update(
        {
            "training_algorithm": TVKD_TRAINING_ALGORITHM,
            "checkpoint_version": TVKD_CHECKPOINT_VERSION,
            "actor_backend": TVKD_ACTOR_BACKEND,
            "teacher_value_bc_scheduler": {
                "residual_scale_ema": 1.0,
                "risk_ema": 0.5,
                "num_updates": 12,
                "current_lambda_bc_student": 0.5,
            },
            "frozen_teacher_state": {
                name: {} for name in ("actor", "encoder_priv", "critic", "value_norm")
            },
            "failure_phase_curriculum_state": {
                "histogram": torch.zeros(
                    int(cfg.algo.failure_phase_num_bins), dtype=torch.float64
                ),
                "episode_count": 0,
                "anchor_count": 0,
                "uniform_fallback_rows": 0,
                "focused_rows": 0,
            },
            "optimizer_resume_state": {
                "actor_optimizer": {},
                "critic_optimizer": {},
                "alpha_optimizer": {},
                "adapt_optimizer": {},
                "dr_estimator_optimizer": None,
            },
            "action_contract": {
                "joint_names": ("joint",),
                "fingerprint": "action-fingerprint",
            },
            "perception_initialization": {},
            "dagger_backend_config": {"value_norm": False},
            "q_backend_config": {},
            "vecnorm_fingerprint": "vecnorm",
            "log_alpha": torch.tensor(-2.0),
            "actor_update_count": 1,
            "critic_update_count": 2,
            "alpha_update_count": 3,
            "dagger_rollout_count": 600,
            "dagger_environment_steps": 600 * int(cfg.algo.train_every),
            "teacher_prefill_rollout_count": 4,
            "teacher_prefill_environment_steps": 4 * int(cfg.algo.train_every),
            "num_updates": 600,
            "sac_actor_update_count": 1,
            "sac_alpha_update_count": 3,
            "dagger_rng_state": rng_state.clone(),
            "q_rng_state": rng_state.clone(),
            "sac_action_rng_state": rng_state.clone(),
            "sac_rollout_rng_state": rng_state.clone(),
            "teacher_perception_rng_state": rng_state.clone(),
            "last_phase": "finetune",
            "last_iter": 40,
            "next_iter": 41,
        }
    )
    torch.save(
        {
            "policy": policy_state,
            "vecnorm": {},
            "cfg": saved_cfg,
        },
        checkpoint_path,
    )
    cfg.fastsac_bc_dagger_checkpoint = str(checkpoint_path)

    result = _prepare_tvkd_checkpoint(cfg)
    validate_tvkd_fastsac_bc_dagger_config(cfg)

    assert result == {"path": str(checkpoint_path), "rollout_count": 600}
    assert cfg.checkpoint_path == str(checkpoint_path)
    assert cfg._tvkd_model_only_resume is True
    assert cfg._bc_dagger_fresh_source is True
    assert cfg.algo.value_norm is False

    drifted = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    drifted.fastsac_bc_dagger_checkpoint = str(checkpoint_path)
    drifted.algo.train_every = int(drifted.algo.train_every) + 1
    with pytest.raises(ValueError, match="algorithm config"):
        _prepare_tvkd_checkpoint(drifted)


def test_public_tvkd_resume_calls_full_seam_then_rebuilds_online_rings(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy._fastsac_action_contract = {
        "joint_names": ("joint",),
        "fingerprint": "same",
    }
    policy.cfg = SimpleNamespace(train_perception=True)
    restored = []

    def restore(self, state, *, load_modules=True):
        restored.append((state, load_modules))

    monkeypatch.setattr(
        TVKDDistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        restore,
    )

    class _Ring:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    policy.dagger_replay = _Ring()
    policy.q_teacher_replay = _Ring()
    policy.actor_adapt = Actor(1)
    policy.actor_adapt(torch.zeros(1, 1))
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1, initial_log_std=torch.tensor(-1.0), device="cpu"
    )
    policy.qnet = nn.Linear(1, 1)
    policy.qnet_target = nn.Linear(1, 1)
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(1))
    policy._teacher_phase_device_cache = {}
    policy._set_perception_trainable = lambda trainable: None
    progress = []
    policy.env = SimpleNamespace(set_progress=lambda value: progress.append(value))

    result = policy.load_state_dict(
        {
            "training_algorithm": TVKD_TRAINING_ALGORITHM,
            "actor_backend": TVKD_ACTOR_BACKEND,
            "action_contract": {
                "joint_names": ("joint",),
                "fingerprint": "same",
            },
            "last_iter": 40,
            "next_iter": 41,
        }
    )

    assert result == []
    assert restored and restored[0][1] is True
    assert policy.dagger_replay.cleared == 1
    assert policy.q_teacher_replay.cleared == 1
    assert policy._teacher_prefill_complete is False
    assert policy.teacher_prefill_rollout_count == 0
    assert policy.actor_adapt.training is True
    assert policy.actor_adapt.actor_std.requires_grad is False
    assert policy.actor_adapt.actor_std.grad is None
    assert policy.qnet.training is True
    assert policy.qnet_target.training is False
    assert progress == [41]
    assert freezes == [1]


def test_private_resume_configures_frozen_perception_before_optimizer_load(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_perception=False,
        train_dr_estimator=False,
        failure_phase_num_bins=2,
    )
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    perception_parameter = nn.Parameter(torch.tensor(1.0))
    original_optimizer = torch.optim.Adam([perception_parameter], lr=1e-3)
    policy.opt_adapt = original_optimizer
    policy._perception_optimizer = original_optimizer
    policy._perception_initialization = {}
    policy._replay_vecnorm_fingerprint = "same-vecnorm"
    policy._failure_histogram_device_cache = {}
    policy.student_bc_scheduler = _scheduler()
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(1))
    state = {
        "training_algorithm": TVKD_TRAINING_ALGORITHM,
        "checkpoint_version": TVKD_CHECKPOINT_VERSION,
        "teacher_value_bc_scheduler": policy.student_bc_scheduler.state_dict(),
        "frozen_teacher_state": {
            name: getattr(policy, name).state_dict()
            for name in ("actor", "encoder_priv", "critic", "value_norm")
        },
        "failure_phase_curriculum_state": {
            "histogram": torch.zeros(2, dtype=torch.float64),
            "episode_count": 0,
            "anchor_count": 0,
            "uniform_fallback_rows": 0,
            "focused_rows": 0,
        },
        "vecnorm_fingerprint": "same-vecnorm",
        "perception_initialization": {"trainable": False},
        "optimizer_resume_state": {"dr_estimator_optimizer": None},
        "num_updates": 0,
        "sac_actor_update_count": 0,
        "sac_alpha_update_count": 0,
        "actor": policy.actor.state_dict(),
        "encoder_priv": policy.encoder_priv.state_dict(),
        "critic": policy.critic.state_dict(),
        "value_norm": policy.value_norm.state_dict(),
    }
    monkeypatch.setattr(PPOVEL, "load_state_dict", lambda self, state, strict: [])
    observed = []

    def baseline_load(self, state, *, load_modules=True):
        observed.append((self.opt_adapt, load_modules))

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        baseline_load,
    )

    policy._load_fastsac_checkpoint_state(state, load_modules=True)

    assert observed == [(None, True)]
    assert policy.opt_adapt is None
    assert policy._perception_optimizer is original_optimizer
    assert policy._perception_initialization["trainable"] is False
    assert freezes == [1]


def test_tvkd_hydra_config_inherits_locked_source_mix_and_new_defaults():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
            ],
        )

    assert cfg.algo.name == "tvkd_fastsac_bc_dagger"
    assert cfg.algo.sac_alpha_update_cadence == "critic"
    assert cfg.algo.use_tvkd_value_shaping is True
    assert cfg.algo.tvkd_lambda == pytest.approx(0.25)
    assert cfg.algo.use_adaptive_student_bc is True
    assert cfg.algo.student_bc_lambda_min == pytest.approx(0.05)
    assert cfg.algo.student_bc_lambda_max == pytest.approx(cfg.algo.lambda_bc)
    assert cfg.algo.teacher_actor_replay_fraction == pytest.approx(0.5)
    assert cfg.algo.failure_phase_teacher_fraction == pytest.approx(0.3)
    assert cfg.algo.q_teacher_replay_ratio == pytest.approx(0.5)
    assert cfg.algo.q_updates_per_rollout == 32

    cfg.algo.sac_alpha_update_cadence = "actor"
    with pytest.raises(ValueError, match="TVKD v1 requires.*critic"):
        validate_tvkd_fastsac_bc_dagger_config(cfg)
