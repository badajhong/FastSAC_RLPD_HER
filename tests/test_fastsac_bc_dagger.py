from __future__ import annotations

import copy
import json
import math
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY, OBS_KEY, Actor
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS,
    FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS,
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
    _build_isolated_distributional_scalar_q_network,
    _build_isolated_scalar_q_network,
    _fastsac_target_entropy,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_STUDENT_ACTION_VALID_KEY,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.ppo_vel import (
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    NEXT_Q_ACTUATOR_CONTEXT_KEY,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    PRETRAINED_PERCEPTION_MODULES,
    PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
    Q_ACTUATOR_CONTEXT_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
    TD3_COLLECTOR_NOISE_KEY,
    TD3_EXPLORATORY_STUDENT_ACTION_KEY,
    TD3_NOISE_FREE_STUDENT_ACTION_KEY,
    DistributionalTD3TeacherBC,
    apply_perception_training_source,
    _failure_lookback_offsets,
    _source_counts,
    _categorical_expected_value,
    _exact_teacher_bc_loss,
    _project_c51_probabilities,
)
from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
    ACTOR_BACKEND,
    CHECKPOINT_VERSION,
    FRESH_STUDENT_ACTOR_INITIALIZATION,
    FASTSAC_DAGGER_ENV_KEY,
    FASTSAC_PREFILL_TEACHER_NOISE_KEY,
    FASTSAC_PREFILL_TEACHER_PROJECTION_KEY,
    FastSACPhysicalNormal,
    NORMALIZED_TANH_ACTION_DISTRIBUTION,
    Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE,
    PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
    PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND,
    PREVIOUS_CHECKPOINT_VERSION,
    Q_TWIN_REDUCTION_MEAN,
    Q_TWIN_REDUCTION_MIN,
    STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
    TEACHER_BC_STUDENT_ACTOR_INITIALIZATION,
    TRAINING_ALGORITHM,
    UNIFORM_PHYSICAL_STD_BOUND_MODE,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
    _DeterministicFastSACStudentEvalPolicy,
    _DistributionalFastSACDaggerRolloutPolicy,
    _seeded_dagger_env_mask,
    _spred_p_teacher_probability,
    checkpoint_module_mismatches,
    validate_actor_adopt_checkpoint_payload,
)
from active_adaptation.learning.ppo.perception_actor import (
    ACTOR_BC_PERCEPTION_SOURCE as PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE,
    ACTOR_INITIALIZATION_SEMANTICS as PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS,
    ACTOR_OBJECTIVE_SEMANTICS as PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS,
    OPTIMIZED_MODULES as PERCEPTION_ACTOR_OPTIMIZED_MODULES,
    TRAINING_ALGORITHM as PERCEPTION_ACTOR_TRAINING_ALGORITHM,
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


class _TensorDictWrite(nn.Module):
    """Write a fixed value while preserving the TensorDict module contract."""

    def __init__(self, key, value: torch.Tensor, *, next_key=None):
        super().__init__()
        self.key = key
        self.next_key = next_key
        self.register_buffer("value", value)
        self.calls = 0

    def forward(self, td: TensorDict):
        self.calls += 1
        value = self.value.to(td[OBS_KEY]).expand(*td.batch_size, -1).clone()
        td[self.key] = value
        if self.next_key is not None:
            td[self.next_key] = value.clone()
        return td


class _PrivilegedLatentActor(nn.Module):
    def get_dist(self, td: TensorDict):
        return SimpleNamespace(mean=td[PRIV_PRED_KEY])


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


class _FixedTwinScalarQ(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_values", values)

    def forward(self, observations, actions):
        del actions
        if observations.shape[0] != self.fixed_values.shape[1]:
            raise ValueError("fixed scalar-Q batch size mismatch")
        return self.fixed_values

    @staticmethod
    def values(outputs):
        return outputs


class _RecordingActionSensitiveTwinC51(_ActionSensitiveTwinC51):
    def __init__(self, first_slope: float = 1.0, second_slope: float = -2.0):
        super().__init__(first_slope, second_slope)
        self.action_inputs: list[torch.Tensor] = []

    def forward(self, observations, actions):
        self.action_inputs.append(actions.detach().clone())
        logits = super().forward(observations, actions)
        # Keep a zero-valued autograd edge to every appended context feature.
        # If the production action-feature helper stops detaching the context,
        # the Actor/SPReD test below observes a zero Tensor gradient instead of
        # ``None`` and fails without changing the fixture's Q values.
        context_edge = actions[:, 1:].sum(dim=-1) * 0.0
        return logits + context_edge.unsqueeze(0).unsqueeze(-1)


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


class _RecordingTableTwinC51(_TableTwinC51):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__(first, second)
        self.action_inputs: list[torch.Tensor] = []

    def forward(self, observations, actions):
        self.action_inputs.append(actions.detach().clone())
        return super().forward(observations, actions)


class _UnexpectedCriticCall(nn.Module):
    def forward(self, observations, actions):
        del observations, actions
        raise AssertionError("SPReD-P must not query the target Critic")


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
    policy._fastsac_student_action_low = torch.tensor([-1.0])
    policy._fastsac_student_action_high = torch.tensor([1.0])
    policy._fastsac_student_action_center = torch.tensor([0.0])
    policy._fastsac_student_action_scale = torch.tensor([1.0])
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
    assert (
        cfg.student_actor_initialization
        == TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
    )
    assert cfg.actor_adopt_checkpoint_path is None
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
    assert cfg.sac_physical_std_bound_mode == UNIFORM_PHYSICAL_STD_BOUND_MODE
    assert cfg.sac_physical_std_min == pytest.approx(0.05)
    assert cfg.sac_physical_std_max == pytest.approx(0.5)
    assert cfg.sac_physical_std_normalized_min == pytest.approx(0.02)
    assert cfg.sac_physical_std_normalized_max == pytest.approx(0.11)
    assert cfg.sac_target_entropy_ratio == pytest.approx(1.0)
    assert cfg.q_twin_reduction == Q_TWIN_REDUCTION_MIN
    assert cfg.q_update_to_data_ratio == pytest.approx(1.0)
    assert cfg.perception_encode_microbatch_size == 512
    assert cfg.perception_replay_mode == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    assert cfg.teacher_perception_replay_fraction == 0.0
    assert cfg.teacher_perception_warmup_steps == 0
    assert cfg.teacher_prefill_use_ppo_noise is True
    assert cfg.dagger_env_fraction == pytest.approx(0.5)
    assert (
        cfg.sac_actor_observation_mode
        == STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE
    )


@pytest.mark.parametrize(
    "mode",
    (TEACHER_BC_STUDENT_ACTOR_INITIALIZATION, FRESH_STUDENT_ACTOR_INITIALIZATION),
)
def test_student_actor_initialization_accepts_only_explicit_modes(mode):
    cfg = DistributionalFastSACTeacherBCConfig(
        student_actor_initialization=mode
    )
    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_student_actor_initialization_rejects_unknown_mode():
    cfg = DistributionalFastSACTeacherBCConfig(
        student_actor_initialization="teacher"
    )
    with pytest.raises(ValueError, match="student_actor_initialization"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_actor_adopt_overlay_rejects_fresh_or_missing_perception_contract():
    fresh = DistributionalFastSACTeacherBCConfig(
        student_actor_initialization="fresh",
        actor_adopt_checkpoint_path="/tmp/actor.pt",
        load_pretrained_perception=True,
        perception_checkpoint_path="/tmp/perception.pt",
    )
    with pytest.raises(ValueError, match="cannot be combined.*fresh"):
        DistributionalFastSACTeacherBC._validate_td3_config(fresh)

    missing_perception = DistributionalFastSACTeacherBCConfig(
        actor_adopt_checkpoint_path="/tmp/actor.pt",
    )
    with pytest.raises(ValueError, match="load_pretrained_perception=true"):
        DistributionalFastSACTeacherBC._validate_td3_config(missing_perception)


@pytest.mark.parametrize("reduction", (Q_TWIN_REDUCTION_MIN, Q_TWIN_REDUCTION_MEAN))
def test_q_twin_reduction_accepts_min_and_mean(reduction):
    cfg = DistributionalFastSACTeacherBCConfig(q_twin_reduction=reduction)

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize("reduction", ("max", "average", "", True, None))
def test_q_twin_reduction_rejects_unknown_modes(reduction):
    cfg = DistributionalFastSACTeacherBCConfig()
    cfg.q_twin_reduction = reduction

    with pytest.raises(ValueError, match="q_twin_reduction"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_privileged_oracle_actor_observation_mode_is_validated_and_live_only():
    valid = DistributionalFastSACTeacherBCConfig(
        sac_actor_observation_mode=PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
    )
    DistributionalFastSACTeacherBC._validate_td3_config(valid)

    invalid = DistributionalFastSACTeacherBCConfig()
    invalid.sac_actor_observation_mode = "teacher_action"
    with pytest.raises(ValueError, match="sac_actor_observation_mode"):
        DistributionalFastSACTeacherBC._validate_td3_config(invalid)

    four_way = DistributionalFastSACTeacherBCConfig(
        sac_actor_observation_mode=PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
        perception_replay_mode="four_way",
    )
    for purpose in ("q", "actor", "perception"):
        for source in (
            "uniform_student",
            "failure_student",
            "uniform_teacher",
            "failure_teacher",
        ):
            setattr(four_way, f"{purpose}_{source}_fraction", 0.25)
    with pytest.raises(ValueError, match="online_student_rollout"):
        DistributionalFastSACTeacherBC._validate_td3_config(four_way)


def test_privileged_oracle_replaces_only_actor_latent_and_advances_perception():
    policy = _bare_policy(
        sac_actor_observation_mode=PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
        use_object_adapt=True,
    )
    policy.depth_feature_dim = 3
    predicted = torch.tensor([[-7.0, -8.0, -9.0]])
    oracle = torch.tensor([[1.0, 2.0, 3.0]])
    policy.temporal_depth_gru_ema = _TensorDictWrite(
        "_depth_feature", torch.zeros(1, 3), next_key=("next", "depth_hx")
    )
    policy.object_adapt_ema = _TensorDictWrite(
        "object_pred", torch.zeros(1, 2)
    )
    policy.object_pred_transform = _TensorDictWrite(
        "object_pred_trans", torch.zeros(1, 2)
    )
    policy.adapt_ema = _TensorDictWrite(
        PRIV_PRED_KEY, predicted, next_key=("next", "adapt_hx")
    )
    policy.object_transform = _TensorDictWrite(
        "object_trans", torch.zeros(1, 2)
    )
    policy.encoder_priv = _TensorDictWrite(PRIV_FEATURE_KEY, oracle)
    policy.actor_adapt = _PrivilegedLatentActor()
    td = TensorDict(
        {OBS_KEY: torch.zeros(2, 4)}, batch_size=(2,), device="cpu"
    )

    action = policy._student_raw_action_proposal(td)

    assert torch.equal(action, oracle.expand(2, -1))
    assert torch.equal(td[PRIV_PRED_KEY], oracle.expand(2, -1))
    assert policy.temporal_depth_gru_ema.calls == 1
    assert policy.object_adapt_ema.calls == 1
    assert policy.adapt_ema.calls == 1
    assert ("next", "depth_hx") in td.keys(True, True)
    assert ("next", "adapt_hx") in td.keys(True, True)


def test_privileged_oracle_teacher_fifo_stores_current_and_next_actor_inputs():
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_actor_observation_mode=PRIVILEGED_ORACLE_ACTOR_OBSERVATION_MODE,
    )
    policy = _bare_policy(**vars(cfg))

    fields = policy._q_replay_storage_fields()

    assert REPLAY_ACTOR_OBSERVATIONS_KEY in fields
    assert REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY in fields

    default_cfg = DistributionalFastSACTeacherBCConfig()
    default_policy = _bare_policy(**vars(default_cfg))
    default_fields = default_policy._q_replay_storage_fields()
    assert REPLAY_ACTOR_OBSERVATIONS_KEY not in default_fields
    assert REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY not in default_fields


def test_teacher_prefill_ppo_noise_flag_must_be_boolean():
    cfg = DistributionalFastSACTeacherBCConfig()
    cfg.teacher_prefill_use_ppo_noise = 1

    with pytest.raises(ValueError, match="teacher_prefill_use_ppo_noise"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    "fraction", (0.0, 1.0, -0.1, 1.1, math.inf, math.nan, True)
)
def test_three_source_config_requires_strict_dagger_env_fraction(fraction):
    cfg = DistributionalFastSACTeacherBCConfig(dagger_env_fraction=fraction)

    with pytest.raises(ValueError, match="dagger_env_fraction"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    ("field", "required"),
    (
        ("dagger_buffer_capacity", 4096),
        ("student_buffer_capacity", 4096),
    ),
)
def test_backend_online_ring_capacity_covers_cohort_learning_start(field, required):
    cfg = DistributionalFastSACTeacherBCConfig()
    setattr(cfg, field, required - 1)

    with pytest.raises(ValueError, match=field):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_backend_allows_small_online_ring_when_q_and_actor_never_sample_it():
    dagger_unused = DistributionalFastSACTeacherBCConfig(
        q_online_dagger_replay_fraction=0.0,
        actor_online_dagger_replay_fraction=0.0,
        dagger_buffer_capacity=1,
    )
    DistributionalFastSACTeacherBC._validate_td3_config(dagger_unused)

    student_unused = DistributionalFastSACTeacherBCConfig(
        q_online_dagger_replay_fraction=1.0,
        actor_online_dagger_replay_fraction=1.0,
        student_buffer_capacity=1,
    )
    DistributionalFastSACTeacherBC._validate_td3_config(student_unused)


def test_fixed_dagger_partition_is_seeded_exact_rounded_and_cached():
    policy = _bare_policy(dagger_env_fraction=0.5, dagger_seed=17)

    first = policy.dagger_env_mask(num_envs=5, device="cpu")
    second = policy.dagger_env_mask(num_envs=5, device=torch.device("cpu"))
    expected = torch.zeros(5, dtype=torch.bool)
    expected[torch.tensor([4, 2, 0])] = True

    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(first, expected)
    assert first.sum().item() == 3
    assert torch.equal(policy.student_only_env_mask(5, "cpu"), ~first)
    assert not torch.equal(first, torch.tensor([True, True, True, False, False]))
    with pytest.raises(ValueError, match="at least one DAgger"):
        policy.dagger_env_mask(num_envs=1, device="cpu")

    # The low-level deterministic constructor remains useful for audit tooling
    # at the mathematical boundary even though runtime three-source configs are
    # deliberately strict and nonempty.
    assert not _seeded_dagger_env_mask(4, 0.0, 17, device="cpu").any()
    assert _seeded_dagger_env_mask(4, 1.0, 17, device="cpu").all()


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


def test_q_normalized_physical_std_config_is_opt_in_and_strictly_validated():
    valid = DistributionalFastSACTeacherBCConfig(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        sac_physical_std_bound_mode=Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE,
        sac_physical_std_normalized_min=0.02,
        sac_physical_std_normalized_max=0.11,
        load_noise_scale=0.15,
        sac_physical_std_min=0.05,
        sac_physical_std_max=0.2,
    )

    DistributionalFastSACTeacherBC._validate_td3_config(valid)

    with pytest.raises(ValueError, match="sac_physical_std_bound_mode"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
                sac_physical_std_bound_mode="teacher_vector",
                load_noise_scale=0.15,
            )
        )
    with pytest.raises(ValueError, match="normalized_min must be smaller"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
                sac_physical_std_bound_mode=Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE,
                sac_physical_std_normalized_min=0.11,
                sac_physical_std_normalized_max=0.11,
                load_noise_scale=0.15,
            )
        )
    with pytest.raises(ValueError, match="require.*ppo_physical_gaussian"):
        DistributionalFastSACTeacherBC._validate_td3_config(
            DistributionalFastSACTeacherBCConfig(
                sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION,
                sac_physical_std_bound_mode=Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE,
            )
        )


def test_ppo_finetune_fresh_load_resets_saved_joint_std_to_load_noise_scale():
    actor = Actor(2, init_noise_scale=1.0, load_noise_scale=0.5)
    actor(torch.zeros(1, 3))
    saved = copy.deepcopy(actor.state_dict())
    saved["actor_std"] = torch.tensor([0.17, 0.29])

    actor.load_state_dict(saved, strict=True)

    assert torch.equal(actor.actor_std, torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize(
    "action_distribution",
    (NORMALIZED_TANH_ACTION_DISTRIBUTION, PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION),
)
def test_config_allows_explicit_pure_sac_ablation_without_inherited_td3_eta(
    action_distribution,
):
    cfg = DistributionalFastSACTeacherBCConfig(
        lambda_bc=0.0,
        eta_sac=1.0,
        sac_action_distribution=action_distribution,
    )

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize("actor_weight_decay", (0.0, 1.0e-4))
def test_actor_weight_decay_is_an_explicit_independent_nonnegative_control(
    actor_weight_decay,
):
    cfg = DistributionalFastSACTeacherBCConfig(
        q_weight_decay=0.37,
        sac_actor_weight_decay=actor_weight_decay,
    )

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)

    assert cfg.q_weight_decay == pytest.approx(0.37)
    assert cfg.sac_actor_weight_decay == pytest.approx(actor_weight_decay)


@pytest.mark.parametrize("actor_weight_decay", (-1.0, math.inf, math.nan, True))
def test_actor_weight_decay_rejects_negative_nonfinite_and_boolean_values(
    actor_weight_decay,
):
    cfg = DistributionalFastSACTeacherBCConfig(
        sac_actor_weight_decay=actor_weight_decay,
    )

    with pytest.raises(ValueError, match="sac_actor_weight_decay.*non-negative"):
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
def test_backend_rejects_teacher_perception_replay_in_live_rollout_mode(fraction):
    cfg = DistributionalFastSACTeacherBCConfig(
        teacher_perception_replay_fraction=fraction
    )

    with pytest.raises(ValueError, match="teacher_perception_replay_fraction=0"):
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_backend_accepts_teacher_only_prefill_warmup_in_live_rollout_mode():
    cfg = DistributionalFastSACTeacherBCConfig(
        perception_replay_mode=ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
        teacher_perception_replay_fraction=0.0,
        teacher_perception_warmup_steps=5000,
    )

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    ("source", "expected_mode", "expected_scope"),
    (
        ("pure_student", ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE, "pure_student"),
        ("all", ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE, "all"),
        ("four_way", "four_way", "pure_student"),
    ),
)
def test_perception_training_source_resolves_the_legacy_pair(
    source, expected_mode, expected_scope
):
    cfg = DistributionalFastSACTeacherBCConfig(perception_training_source=source)

    assert apply_perception_training_source(cfg) == source
    assert cfg.perception_replay_mode == expected_mode
    assert cfg.perception_live_env_scope == expected_scope


def test_perception_training_source_rejects_unknown_value():
    cfg = DistributionalFastSACTeacherBCConfig(perception_training_source="student")

    with pytest.raises(ValueError, match="perception_training_source"):
        apply_perception_training_source(cfg)


def test_backend_rejects_unsupported_perception_mode():
    mode = "legacy_online_student"
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


def test_backend_accepts_positive_row_level_q_utd():
    cfg = DistributionalFastSACTeacherBCConfig(q_update_to_data_ratio=4.0)

    DistributionalFastSACTeacherBC._validate_td3_config(cfg)


def test_tanh_normal_is_bounded_reparameterized_and_has_exact_log_prob():
    loc = torch.tensor([[0.0, 1.25], [-1.55, 0.05]], requires_grad=True)
    scale = torch.tensor([[0.01, 0.02], [0.01, 0.02]], requires_grad=True)
    action_low = torch.tensor([-2.0, -6.0])
    action_high = torch.tensor([4.0, 2.0])
    dist = FastSACTanhNormal(
        loc,
        scale,
        low=action_low,
        high=action_high,
        event_dims=1,
    )

    sample_generator = torch.Generator().manual_seed(71)
    oracle_generator = torch.Generator().manual_seed(71)
    first, first_log_prob = dist.rsample_with_log_prob(
        generator=sample_generator
    )
    oracle_noise = torch.randn(loc.shape, generator=oracle_generator)
    oracle_latent = loc + scale * oracle_noise
    action_scale = (action_high - action_low) * 0.5
    action_center = (action_high + action_low) * 0.5
    expected_action = (
        torch.tanh(oracle_latent) * action_scale + action_center
    )
    log_tanh_jacobian = 2.0 * (
        math.log(2.0)
        - oracle_latent
        - torch.nn.functional.softplus(-2.0 * oracle_latent)
    )
    expected_log_prob = (
        torch.distributions.Normal(loc, scale).log_prob(oracle_latent)
        - log_tanh_jacobian
        - torch.log(action_scale)
    ).sum(dim=-1)

    second, second_log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(72)
    )
    assert not torch.equal(first, second)
    assert torch.allclose(first, expected_action, rtol=0.0, atol=1.0e-7)
    assert torch.allclose(
        first_log_prob, expected_log_prob, rtol=1.0e-6, atol=1.0e-6
    )
    assert torch.isfinite(first_log_prob).all()
    assert torch.isfinite(second_log_prob).all()
    assert torch.all(first >= action_low) and torch.all(first <= action_high)
    assert torch.allclose(
        first_log_prob, dist.log_prob_for_action(first), rtol=2e-5, atol=2e-5
    )
    (first.sum() - first_log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_backend_distribution_uses_calibrated_nominal_student_support():
    policy = _bare_policy(
        sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION,
        sac_log_std_min=-8.0,
        sac_log_std_max=-1.0,
        action_support_clip=20.0,
        q_action_input_gain=1.0,
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
    # Teacher/replay execution retains the legacy finite safety envelope. The
    # Student policy itself uses the asymmetric nominal joint coordinates.
    policy._fastsac_action_low = torch.tensor([-20.0, -20.0])
    policy._fastsac_action_high = torch.tensor([20.0, 20.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0, 0.0])
    policy._fastsac_actor_action_scale = torch.tensor([20.0, 20.0])
    policy._fastsac_student_action_low = torch.tensor([-1.0, -6.0])
    policy._fastsac_student_action_high = torch.tensor([3.0, 2.0])
    policy._fastsac_student_action_center = torch.tensor([1.0, -2.0])
    policy._fastsac_student_action_scale = torch.tensor([2.0, 4.0])
    policy._fastsac_q_action_center = policy._fastsac_student_action_center.clone()
    policy._fastsac_q_action_scale = policy._fastsac_student_action_scale.clone()

    zero_mean = torch.zeros(3, 2, requires_grad=True)
    zero_dist = policy._normalized_tanh_dist_from_mean(zero_mean)
    assert torch.equal(zero_dist.mean, torch.zeros_like(zero_mean))
    zero_slope = torch.autograd.grad(zero_dist.mean.sum(), zero_mean)[0]
    assert torch.allclose(zero_slope, torch.ones_like(zero_slope), atol=1.0e-6)

    raw_mean = torch.tensor(
        [[0.0, 0.0], [0.4, -0.3], [100.0, -100.0]],
        requires_grad=True,
    )
    dist = policy._normalized_tanh_dist_from_mean(raw_mean)
    normalized_zero = (
        -policy._fastsac_student_action_center
        / policy._fastsac_student_action_scale
    )
    latent_zero = torch.atanh(normalized_zero)
    inverse_local_slope = 1.0 / (
        policy._fastsac_student_action_scale
        * (1.0 - normalized_zero.square())
    )
    expected_loc = latent_zero + raw_mean * inverse_local_slope

    assert torch.allclose(dist.loc, expected_loc)
    assert torch.allclose(dist.scale, expected_log_std.exp().expand_as(raw_mean))
    assert torch.equal(dist.low, policy._fastsac_student_action_low)
    assert torch.equal(dist.high, policy._fastsac_student_action_high)
    assert torch.all(dist.mean >= policy._fastsac_student_action_low)
    assert torch.all(dist.mean <= policy._fastsac_student_action_high)

    sampled_action, _ = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(91)
    )
    assert torch.all(sampled_action >= policy._fastsac_student_action_low)
    assert torch.all(sampled_action <= policy._fastsac_student_action_high)
    # Student samples are already executable. The +/-20 projection is only a
    # final finite-safety guard and must not alter the Actor/Q action.
    assert torch.equal(
        policy._project_execution_action(sampled_action), sampled_action
    )
    normalized_q_action = policy._q_action_input(sampled_action)
    assert torch.all(normalized_q_action >= -1.0)
    assert torch.all(normalized_q_action <= 1.0)

    delegated = policy._sac_dist_from_mean(raw_mean)
    assert torch.equal(delegated.loc, dist.loc)
    assert torch.equal(delegated.scale, dist.scale)
    assert torch.equal(delegated.mean, dist.mean)


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


def test_teacher_bc_target_uses_distribution_specific_student_support():
    teacher_action = torch.tensor([[2.0, 25.0]])

    physical = _bare_policy(
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
    )
    physical._fastsac_action_low = torch.tensor([-20.0, -20.0])
    physical._fastsac_action_high = torch.tensor([20.0, 20.0])
    physical.cfg.action_support_clip = 20.0
    assert torch.equal(
        physical._project_student_policy_action(teacher_action),
        torch.tensor([[2.0, 20.0]]),
    )

    bounded = _bare_policy(
        sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION
    )
    bounded._fastsac_q_action_center = torch.zeros(2)
    bounded._fastsac_q_action_scale = torch.ones(2)
    bounded._fastsac_student_action_low = torch.full((2,), -1.0)
    bounded._fastsac_student_action_high = torch.ones(2)
    assert torch.equal(
        bounded._project_student_policy_action(teacher_action),
        torch.tensor([[1.0, 1.0]]),
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
    _install_unit_action_contract(policy)

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


def test_noisy_teacher_prefill_schema_retains_clean_bc_label_separately():
    policy = _bare_policy(teacher_prefill_use_ppo_noise=True)
    policy._has_canonical_replay_mix = lambda: False
    policy._q_conditions_on_actuator_state = lambda: False
    policy._teacher_episode_cache_enabled = lambda: False

    fields = policy._q_replay_prefill_storage_fields()

    assert "actions" in fields
    assert DAGGER_REPLAY_TEACHER_ACTIONS in fields
    assert fields.count(DAGGER_REPLAY_TEACHER_ACTIONS) == 1


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


@pytest.mark.parametrize(
    ("mode", "expected_mean"),
    (
        (TEACHER_BC_STUDENT_ACTOR_INITIALIZATION, 7.0),
        (FRESH_STUDENT_ACTOR_INITIALIZATION, 0.25),
    ),
)
def test_fresh_ppo_source_selects_student_mean_without_touching_teacher_or_perception(
    monkeypatch, mode, expected_mean
):
    policy = _bare_policy(
        student_actor_initialization=mode,
        load_noise_scale=0.5,
        sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION,
        sac_alpha_init=1.0e-5,
        dagger_seed=11,
        q_seed=13,
    )
    policy.actor = nn.Linear(1, 1, bias=False)
    policy.actor_adapt = _TinyPhysicalActor(
        1, mean_weight=0.25, std=0.17
    )
    policy.perception_probe = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.actor.weight.fill_(0.1)
        policy.perception_probe.weight.fill_(0.2)
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        1, torch.tensor([-2.0]), "cpu"
    )
    policy._fastsac_initial_raw_log_std = torch.tensor([-2.0])
    policy.qnet = nn.Linear(1, 1)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.tensor(-1.0))
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.sac_action_rng = torch.Generator().manual_seed(3)
    policy.sac_rollout_rng = torch.Generator().manual_seed(4)
    policy.teacher_prefill_action_rng = torch.Generator().manual_seed(5)
    policy.teacher_perception_rng = torch.Generator().manual_seed(6)
    policy.actor_optimizer = torch.optim.Adam(
        policy.actor_adapt.parameters(), lr=1.0e-3
    )
    optimizer_parameter_ids = tuple(
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    )
    if mode == FRESH_STUDENT_ACTOR_INITIALIZATION:
        policy._fresh_student_actor_constructor_state = copy.deepcopy(
            policy.actor_adapt.state_dict()
        )
        policy._fresh_student_actor_constructor_parameter_ids = tuple(
            id(parameter) for parameter in policy.actor_adapt.parameters()
        )

    def fake_ppo_source_load(owner, state, strict=True):
        del state, strict
        with torch.no_grad():
            owner.actor.weight.fill_(5.0)
            owner.actor_adapt.mean_weight.fill_(7.0)
            owner.actor_adapt.actor_core.actor_mean.weight.fill_(8.0)
            owner.actor_adapt.actor_core.actor_mean.bias.fill_(9.0)
            # Match Actor._load_from_state_dict: source means load while std
            # is reset to the runtime load_noise_scale.
            owner.actor_adapt.actor_core.actor_std.fill_(0.5)
            owner.perception_probe.weight.fill_(6.0)
        return []

    monkeypatch.setattr(
        DistributionalTD3TeacherBC,
        "load_state_dict",
        fake_ppo_source_load,
    )
    failed = policy.load_state_dict(
        {"last_phase": "train", "last_iter": 123}, strict=True
    )

    assert failed == []
    assert policy.actor.weight.item() == pytest.approx(5.0)
    assert policy.perception_probe.weight.item() == pytest.approx(6.0)
    assert policy.actor_adapt.mean_weight.item() == pytest.approx(expected_mean)
    assert torch.equal(
        policy.actor_adapt.actor_core.actor_std,
        torch.full_like(policy.actor_adapt.actor_core.actor_std, 0.5),
    )
    assert optimizer_parameter_ids == tuple(
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    )
    assert policy._actor_initialization == {
        "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
        "mode": mode,
        "teacher_actor_loaded": True,
        "actor_adapt_mean_loaded": (
            mode == TEACHER_BC_STUDENT_ACTOR_INITIALIZATION
        ),
        "actor_adapt_mean_fresh": mode == FRESH_STUDENT_ACTOR_INITIALIZATION,
        "source_phase": "train",
        "source_iter": 123,
    }
    assert not hasattr(policy, "_fresh_student_actor_constructor_state")


def _actor_adopt_checkpoint_payload(
    actor_state,
    perception_state,
    *,
    teacher_path="/tmp/teacher.pt",
):
    policy = {
        "training_algorithm": PERCEPTION_ACTOR_TRAINING_ALGORITHM,
        "last_phase": "finetune",
        "last_iter": 1200,
        "actor_objective_semantics": PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS,
        "actor_initialization_semantics": (
            PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS
        ),
        "actor_adapt_loaded_from_teacher_checkpoint": True,
        "actor_adapt_trained": True,
        "actor_adapt_controls_rollout": False,
        "actor_bc_perception_source": PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE,
        "actor_bc_uses_online_priv_pred": False,
        "actor_adapt_bc_update_count": 1201,
        "optimized_modules": PERCEPTION_ACTOR_OPTIMIZED_MODULES,
        "actor_adapt": copy.deepcopy(actor_state),
        "actor": {},
        "encoder_priv": {},
        "critic": {},
    }
    policy.update(copy.deepcopy(perception_state))
    return {
        "policy": policy,
        "cfg": {
            "checkpoint_path": teacher_path,
            "task": {"name": "G1SkateboardGeneralTracking"},
            "algo": {
                "name": "teacher_rollout_perception_actor",
                "_target_": (
                    "active_adaptation.learning.ppo.perception_actor."
                    "TeacherRolloutPerceptionActor"
                ),
                "distill_with_priv_pred": True,
                "actor_bc_perception_source": (
                    PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE
                ),
                "latent_dim": 256,
                "in_keys": [
                    "command",
                    "policy",
                    "object_",
                    "priv",
                    "object_geo_",
                    "vel_command",
                    "depth",
                ],
            },
        },
    }


def test_actor_adopt_payload_requires_exact_stage_metadata_and_actor_shape():
    actor = _TinyPhysicalActor(2, mean_weight=0.25, std=0.17)
    perception = {
        name: nn.Linear(2, 2).state_dict()
        for name in PRETRAINED_PERCEPTION_MODULES
    }
    checkpoint = _actor_adopt_checkpoint_payload(
        actor.state_dict(), perception
    )

    actor_state, provenance = validate_actor_adopt_checkpoint_payload(
        checkpoint,
        source_path="/tmp/actor.pt",
    )

    _assert_nested_equal(dict(actor_state), dict(actor.state_dict()))
    assert provenance["semantics"] == ACTOR_ADOPT_CHECKPOINT_SEMANTICS
    assert provenance["source_actor_bc_update_count"] == 1201
    assert provenance["source_actor_bc_perception_source"] == (
        PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE
    )
    assert provenance["runtime_std_source"] == "load_noise_scale"

    invalid = copy.deepcopy(checkpoint)
    invalid["policy"]["actor_adapt_trained"] = False
    with pytest.raises(ValueError, match="actor_adapt_trained"):
        validate_actor_adopt_checkpoint_payload(invalid)

    invalid = copy.deepcopy(checkpoint)
    invalid["policy"]["actor_bc_uses_online_priv_pred"] = True
    with pytest.raises(ValueError, match="actor_bc_uses_online_priv_pred"):
        validate_actor_adopt_checkpoint_payload(invalid)

    invalid = copy.deepcopy(checkpoint)
    invalid["cfg"]["algo"]["actor_bc_perception_source"] = "online"
    with pytest.raises(ValueError, match="rollout EMA perception"):
        validate_actor_adopt_checkpoint_payload(invalid)

    invalid = copy.deepcopy(checkpoint)
    mean_key = next(
        key
        for key in invalid["policy"]["actor_adapt"]
        if key.endswith("actor_mean.weight")
    )
    invalid["policy"]["actor_adapt"][mean_key] = torch.zeros(3, 1)
    with pytest.raises(ValueError, match="action dimensions disagree"):
        validate_actor_adopt_checkpoint_payload(invalid)


def test_actor_adopt_loader_overlays_only_mean_body_and_keeps_runtime_std(
    tmp_path,
):
    actor_path = tmp_path / "actor_adopt.pt"
    perception_path = tmp_path / "perception.pt"
    teacher_path = tmp_path / "teacher.pt"
    policy = _bare_policy(
        actor_adopt_checkpoint_path=str(actor_path),
        perception_checkpoint_path=str(perception_path),
        student_actor_initialization="teacher_bc",
        in_keys=(
            "command",
            "policy",
            "object_",
            "priv",
            "object_geo_",
            "vel_command",
            "depth",
        ),
        latent_dim=256,
        sac_action_distribution=PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        load_noise_scale=0.05,
    )
    policy.actor_adapt = _TinyPhysicalActor(2, mean_weight=0.25, std=0.05)
    policy.actor_adapt.actor_core.load_noise_scale = 0.05
    for index, name in enumerate(PRETRAINED_PERCEPTION_MODULES):
        with torch.random.fork_rng():
            torch.manual_seed(100 + index)
            setattr(policy, name, nn.Linear(2, 2))
    perception = {
        name: copy.deepcopy(getattr(policy, name).state_dict())
        for name in PRETRAINED_PERCEPTION_MODULES
    }
    source_actor_state = copy.deepcopy(policy.actor_adapt.state_dict())
    for key, value in source_actor_state.items():
        if key.endswith("actor_std"):
            value.fill_(0.19)
        else:
            value.fill_(7.0)
    torch.save(
        _actor_adopt_checkpoint_payload(
            source_actor_state,
            perception,
            teacher_path=str(teacher_path),
        ),
        actor_path,
    )
    torch.save({"policy": perception}, perception_path)
    torch.save({"policy": {}}, teacher_path)
    parameter_ids = tuple(id(item) for item in policy.actor_adapt.parameters())
    optimizer = torch.optim.Adam(policy.actor_adapt.parameters(), lr=1.0e-3)
    optimizer_ids = tuple(
        id(item)
        for group in optimizer.param_groups
        for item in group["params"]
    )

    provenance = policy._load_actor_adopt_checkpoint(
        str(actor_path),
        teacher_source_policy={"actor": {}, "encoder_priv": {}, "critic": {}},
    )

    assert policy.actor_adapt.mean_weight.eq(7.0).all()
    assert policy.actor_adapt.actor_core.actor_mean.weight.eq(7.0).all()
    assert policy.actor_adapt.actor_core.actor_std.eq(0.05).all()
    assert tuple(id(item) for item in policy.actor_adapt.parameters()) == parameter_ids
    assert optimizer_ids == tuple(
        id(item)
        for group in optimizer.param_groups
        for item in group["params"]
    )
    assert provenance["perception_exact_match"] is True
    assert provenance["perception_mismatched_modules"] == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_module_comparison_is_device_neutral():
    cpu_state = {"module": {"weight": torch.tensor([1.0])}}
    cuda_state = {"module": {"weight": torch.tensor([1.0], device="cuda")}}

    assert checkpoint_module_mismatches(
        cpu_state, cuda_state, ("module",)
    ) == ()


def test_actor_initialization_provenance_must_match_backend_runtime_and_flags():
    policy = _bare_policy(student_actor_initialization="fresh")
    valid = {
        "dagger_backend_config": {"student_actor_initialization": "fresh"},
        "actor_initialization": {
            "semantics": STUDENT_ACTOR_INITIALIZATION_SEMANTICS,
            "mode": "fresh",
            "teacher_actor_loaded": True,
            "actor_adapt_mean_loaded": False,
            "actor_adapt_mean_fresh": True,
        },
    }
    policy._restore_actor_initialization_provenance(valid)
    assert policy._actor_initialization["mode"] == "fresh"

    backend_mismatch = copy.deepcopy(valid)
    backend_mismatch["dagger_backend_config"][
        "student_actor_initialization"
    ] = "teacher_bc"
    with pytest.raises(ValueError, match="initialization/backend mismatch"):
        policy._restore_actor_initialization_provenance(backend_mismatch)

    runtime_policy = _bare_policy(student_actor_initialization="teacher_bc")
    with pytest.raises(ValueError, match="initialization/runtime mismatch"):
        runtime_policy._restore_actor_initialization_provenance(valid)

    invalid_flags = copy.deepcopy(valid)
    invalid_flags["actor_initialization"]["actor_adapt_mean_loaded"] = True
    with pytest.raises(ValueError, match="flags are inconsistent"):
        policy._restore_actor_initialization_provenance(invalid_flags)


def test_legacy_actor_initialization_provenance_infers_only_teacher_bc():
    legacy = _bare_policy(student_actor_initialization="teacher_bc")
    legacy._restore_actor_initialization_provenance(
        {"dagger_backend_config": {}}
    )
    assert legacy._actor_initialization["mode"] == "teacher_bc"
    assert legacy._actor_initialization["legacy_inferred"] is True

    fresh = _bare_policy(student_actor_initialization="fresh")
    with pytest.raises(ValueError, match="initialization/runtime mismatch"):
        fresh._restore_actor_initialization_provenance(
            {"dagger_backend_config": {}}
        )


def test_actor_adopt_resume_provenance_is_strict_and_source_files_are_not_opened():
    actor_path = "/historical/actor.pt"
    perception_path = "/historical/perception.pt"
    policy = _bare_policy(
        actor_adopt_checkpoint_path=actor_path,
        perception_checkpoint_path=perception_path,
    )
    provenance = {
        "semantics": ACTOR_ADOPT_CHECKPOINT_SEMANTICS,
        "loaded": True,
        "source_path": actor_path,
        "source_algorithm": PERCEPTION_ACTOR_TRAINING_ALGORITHM,
        "source_phase": "finetune",
        "source_iter": 1200,
        "source_actor_bc_update_count": 1201,
        "source_actor_objective_semantics": PERCEPTION_ACTOR_OBJECTIVE_SEMANTICS,
        "source_actor_initialization_semantics": (
            PERCEPTION_ACTOR_INITIALIZATION_SEMANTICS
        ),
        "source_actor_bc_perception_source": (
            PERCEPTION_ACTOR_BC_PERCEPTION_SOURCE
        ),
        "source_teacher_checkpoint_path": "/historical/teacher.pt",
        "source_task_name": "G1SkateboardGeneralTracking",
        "module": "actor_adapt",
        "runtime_std_source": "load_noise_scale",
        "perception_source_path": perception_path,
        "perception_exact_match": False,
        "perception_mismatched_modules": ("adapt_module",),
    }
    state = {
        "dagger_backend_config": {
            "actor_adopt_checkpoint_path": actor_path,
            "load_pretrained_perception": True,
            "perception_checkpoint_path": perception_path,
        },
        "actor_adopt_initialization": provenance,
    }

    policy._restore_actor_adopt_initialization_provenance(state)
    assert policy._actor_adopt_initialization == provenance

    tampered = copy.deepcopy(state)
    tampered["actor_adopt_initialization"][
        "perception_mismatched_modules"
    ] = ("actor",)
    with pytest.raises(ValueError, match="mismatch modules are invalid"):
        policy._restore_actor_adopt_initialization_provenance(tampered)

    tampered = copy.deepcopy(state)
    tampered["dagger_backend_config"]["perception_checkpoint_path"] = (
        "/historical/other.pt"
    )
    with pytest.raises(ValueError, match="perception source path mismatch"):
        policy._restore_actor_adopt_initialization_provenance(tampered)


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


def test_q_normalized_physical_std_bounds_intersect_absolute_envelope_per_joint():
    policy = _tiny_physical_policy(action_dim=3)
    policy.cfg.sac_physical_std_bound_mode = Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE
    policy.cfg.sac_physical_std_min = 0.05
    policy.cfg.sac_physical_std_max = 0.2
    policy.cfg.sac_physical_std_normalized_min = 0.02
    policy.cfg.sac_physical_std_normalized_max = 0.11
    policy._fastsac_q_action_scale = torch.tensor([0.5355, 1.063636, 5.890499])

    lower, upper = policy._physical_std_bounds()

    assert torch.allclose(lower, torch.tensor([0.05, 0.05, 0.11780998]))
    assert torch.allclose(upper, torch.tensor([0.058905, 0.11699996, 0.2]))
    with torch.no_grad():
        policy._ppo_actor_std_parameter().copy_(torch.tensor([0.01, 0.15, 0.9]))
    projected = policy._project_physical_actor_std_()
    assert projected.item() == pytest.approx(1.0)
    assert torch.allclose(
        policy._ppo_actor_std_parameter(),
        torch.tensor([0.05, 0.11699996, 0.2]),
    )


def test_q_normalized_physical_std_rejects_empty_joint_intersection():
    policy = _tiny_physical_policy()
    policy.cfg.sac_physical_std_bound_mode = Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE
    policy.cfg.sac_physical_std_min = 0.05
    policy.cfg.sac_physical_std_max = 0.2
    policy.cfg.sac_physical_std_normalized_min = 0.02
    policy.cfg.sac_physical_std_normalized_max = 0.11
    policy._fastsac_q_action_scale = torch.tensor([0.1])

    with pytest.raises(ValueError, match="empty for joint indices"):
        policy._physical_std_bounds()


def test_g1_q_normalized_envelope_has_intended_exploration_and_entropy_margin():
    q_scale = torch.tensor(
        [
            4.426772,
            4.426772,
            4.284,
            4.488042,
            4.488042,
            1.063636,
            4.512436,
            4.512436,
            1.063636,
            3.8148,
            3.8148,
            5.890499,
            5.890499,
            1.428003,
            1.428003,
            3.926966,
            3.926966,
            0.5355,
            0.5355,
            5.355,
            5.355,
            3.213,
            3.213,
        ]
    )
    policy = _tiny_physical_policy(action_dim=q_scale.numel())
    policy.cfg.sac_physical_std_bound_mode = Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE
    policy.cfg.sac_physical_std_min = 0.05
    policy.cfg.sac_physical_std_max = 0.2
    policy.cfg.sac_physical_std_normalized_min = 0.02
    policy.cfg.sac_physical_std_normalized_max = 0.11
    policy._fastsac_q_action_scale = q_scale
    policy.target_entropy = -1.6 * q_scale.numel()
    with torch.no_grad():
        policy._ppo_actor_std_parameter().fill_(0.15)

    policy._project_physical_actor_std_()
    projected_std = policy._bounded_physical_actor_std(detach=True)
    normalized_l2 = torch.linalg.vector_norm(projected_std / q_scale)
    entropy_min, entropy_max = policy._physical_normalized_entropy_bounds()

    assert normalized_l2.item() == pytest.approx(0.30277, rel=2.0e-5)
    assert torch.linalg.vector_norm(policy._physical_std_bounds()[1] / q_scale).item() == (
        pytest.approx(0.33210, rel=2.0e-5)
    )
    assert entropy_min < policy.target_entropy < entropy_max
    policy._validate_physical_entropy_target_reachable()


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


def test_physical_actor_bc_keeps_teacher_action_outside_nominal_q_box():
    policy = _tiny_physical_policy(q_slope=1.0)
    batch = _tiny_physical_batch()
    batch[DAGGER_REPLAY_TEACHER_ACTIONS] = torch.tensor(
        [[2.5], [-2.5], [0.1]]
    )
    with torch.no_grad():
        prediction = policy._actor_mean_from_flat(batch["observations"])
        expected_bc = _exact_teacher_bc_loss(
            prediction,
            batch[DAGGER_REPLAY_TEACHER_ACTIONS],
            batch[DAGGER_TEACHER_ACTION_VALID_KEY],
            policy._fastsac_q_action_center,
            policy._fastsac_q_action_scale,
            policy.cfg.dagger_actor_huber_delta,
        )

    metrics = policy._actor_update(batch)

    assert metrics["exact_bc_loss"].item() == pytest.approx(expected_bc.item())
    assert metrics[
        "teacher_bc_student_support_projection_fraction"
    ].item() == pytest.approx(0.0)


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
    assert info["fastsac/actor_std_physical_geometric_mean"] == pytest.approx(
        final_std.log().mean().exp().item()
    )
    assert info["fastsac/actor_std_q_normalized_l2"] == pytest.approx(
        torch.linalg.vector_norm(final_std).item()
    )


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


@pytest.mark.parametrize(
    "q_twin_reduction", (Q_TWIN_REDUCTION_MIN, Q_TWIN_REDUCTION_MEAN)
)
def test_reparameterized_actor_step_uses_configured_twin_q_and_exact_bc(
    q_twin_reduction,
):
    policy = _bare_policy(
        eta_sac=0.7,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
        q_twin_reduction=q_twin_reduction,
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
        raw_mean,
        policy._bounded_log_std(expected_adapter.log_std).exp().expand_as(raw_mean),
        low=torch.tensor([-1.0]),
        high=torch.tensor([1.0]),
        event_dims=1,
    )
    sampled_action, physical_log_prob = expected_dist.rsample_with_log_prob(
        generator=expected_rng
    )
    log_prob = physical_log_prob
    expected_heads = expected_q.values(
        expected_q(batch["critic_observations"], sampled_action)
    )
    expected_min_q = expected_heads.min(dim=0).values
    expected_reduced_q = (
        expected_min_q
        if q_twin_reduction == Q_TWIN_REDUCTION_MIN
        else expected_heads.mean(dim=0)
    )
    expected_sac = (
        policy.log_alpha.detach().exp() * log_prob - expected_reduced_q
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
        expected_min_q.mean().item()
    )
    assert metrics["actor_reduced_expected_q_mean"].item() == pytest.approx(
        expected_reduced_q.mean().item()
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


def test_spred_p_teacher_probability_matches_official_cdf_and_is_detached():
    policy_q = torch.tensor([[0.0, 2.0], [2.0, 4.0]], requires_grad=True)
    teacher_q = torch.tensor([[2.0, 0.0], [4.0, 2.0]], requires_grad=True)

    weights, advantages, combined_std = _spred_p_teacher_probability(
        policy_q, teacher_q
    )

    expected_weights = 0.5 * (
        1.0 + torch.erf(torch.tensor([1.0, -1.0]) / math.sqrt(2.0))
    )
    assert torch.allclose(weights, expected_weights)
    assert torch.equal(advantages, torch.tensor([2.0, -2.0]))
    assert torch.allclose(combined_std, torch.tensor([2.0, 2.0]))
    assert not weights.requires_grad
    assert not advantages.requires_grad
    assert not combined_std.requires_grad

    # Identical, zero-disagreement twins are numerically defined: equal action
    # values mean an exactly neutral Teacher probability rather than NaN.
    neutral, _, uncertainty = _spred_p_teacher_probability(
        torch.zeros(2, 1), torch.zeros(2, 1)
    )
    assert neutral.item() == pytest.approx(0.5)
    assert torch.isfinite(uncertainty).all()


def test_q_filtered_bc_uses_online_twin_detached_spred_p_and_valid_denominator():
    policy = _bare_policy(
        eta_sac=0.0,
        lambda_bc=1.0,
        use_q_filtered_bc=True,
        dagger_actor_huber_delta=1.0,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        sac_max_grad_norm=1.0e6,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy, actor_weight=0.25)
    policy.qnet = _ActionSensitiveTwinC51(first_slope=1.0, second_slope=2.0)
    policy.qnet_target = _UnexpectedCriticCall()
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = _CountingSGD(policy._fastsac_actor_parameters, lr=0.05)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.sac_action_rng = torch.Generator().manual_seed(313)
    policy.actor_update_count = 0
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [3.0]]),
        "critic_observations": torch.ones(3, 1),
        # Row 0 favors Student and receives a soft weight below 0.5. Row 1
        # favors Teacher and receives a soft weight above 0.5. Row 2 is invalid.
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.0], [1.0], [float("nan")]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        # Row 0 came from Student replay; rows 1 and 2 came from Teacher replay.
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([False, True, True]),
    }

    raw_prediction = policy._actor_mean_from_flat(batch["observations"])
    prediction_action = policy._sac_dist_from_mean(raw_prediction).mean
    safe_teacher = torch.where(
        batch[DAGGER_TEACHER_ACTION_VALID_KEY].unsqueeze(-1),
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        prediction_action.detach(),
    )
    with torch.no_grad():
        policy_q = policy.qnet.values(
            policy.qnet(
                batch["critic_observations"],
                policy._q_action_input(prediction_action.detach()),
            )
        )
        teacher_q = policy.qnet.values(
            policy.qnet(
                batch["critic_observations"],
                policy._q_action_input(safe_teacher),
            )
        )
        weights, advantages, combined_std = _spred_p_teacher_probability(
            policy_q, teacher_q
        )
    per_row_bc = torch.nn.functional.smooth_l1_loss(
        prediction_action,
        safe_teacher,
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    expected_bc = (per_row_bc[:2] * weights[:2]).sum() / 2.0
    expected_actor_gradient = torch.autograd.grad(
        expected_bc, policy.actor_adapt.weight
    )[0]
    actor_weight_before = policy.actor_adapt.weight.detach().clone()

    metrics = policy._actor_update(batch)

    assert 0.0 < weights[0].item() < 0.5
    assert 0.5 < weights[1].item() < 1.0
    assert metrics["exact_bc_loss"].item() == pytest.approx(
        expected_bc.item(), abs=5.0e-6
    )
    assert torch.allclose(
        policy.actor_adapt.weight,
        actor_weight_before - 0.05 * expected_actor_gradient,
        atol=5.0e-6,
    )
    assert metrics["q_filtered_bc_active_fraction"].item() == pytest.approx(
        weights[:2].mean().item()
    )
    assert metrics["q_filtered_bc_policy_better_fraction"].item() == pytest.approx(
        0.5
    )
    assert metrics["spred_p_bc_weight_mean"].item() == pytest.approx(
        weights[:2].mean().item()
    )
    assert metrics["spred_p_bc_weight_std"].item() == pytest.approx(
        weights[:2].std(unbiased=False).item()
    )
    assert metrics["spred_p_teacher_advantage_mean"].item() == pytest.approx(
        advantages[:2].mean().item()
    )
    assert metrics["spred_p_combined_q_std_mean"].item() == pytest.approx(
        combined_std[:2].mean().item()
    )
    assert metrics["spred_p_source_metadata_available"].item() == pytest.approx(1.0)
    assert metrics["spred_p_teacher_source_valid_fraction"].item() == pytest.approx(
        1.0 / 3.0
    )
    assert metrics["spred_p_student_source_valid_fraction"].item() == pytest.approx(
        1.0 / 3.0
    )
    assert metrics["spred_p_teacher_source_bc_weight_mean"].item() == pytest.approx(
        weights[1].item()
    )
    assert metrics["spred_p_student_source_bc_weight_mean"].item() == pytest.approx(
        weights[0].item()
    )
    assert metrics[
        "spred_p_teacher_source_teacher_advantage_mean"
    ].item() == pytest.approx(advantages[1].item())
    assert metrics[
        "spred_p_student_source_teacher_advantage_mean"
    ].item() == pytest.approx(advantages[0].item())
    assert metrics[
        "spred_p_teacher_source_combined_q_std_mean"
    ].item() == pytest.approx(combined_std[1].item())
    assert metrics[
        "spred_p_student_source_combined_q_std_mean"
    ].item() == pytest.approx(combined_std[0].item())
    assert metrics[
        "q_filtered_bc_teacher_source_policy_better_fraction"
    ].item() == pytest.approx(0.0)
    assert metrics[
        "q_filtered_bc_student_source_policy_better_fraction"
    ].item() == pytest.approx(1.0)
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())
    assert all(parameter.grad is None for parameter in policy.qnet_target.parameters())


def test_gradient_probe_matches_production_q_filtered_bc_and_supports_override():
    policy = _bare_policy(
        eta_sac=0.2,
        lambda_bc=1.0,
        use_q_filtered_bc=True,
        dagger_actor_huber_delta=1.0,
        sac_log_std_min=-5.0,
        sac_log_std_max=1.0,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy, actor_weight=0.25)
    policy.qnet = _ActionSensitiveTwinC51(first_slope=1.0, second_slope=2.0)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = torch.optim.AdamW(
        (
            {"params": tuple(policy.actor_adapt.parameters())},
            {"params": tuple(policy.bc_dagger_sac_adapter.parameters())},
        ),
        lr=0.01,
    )
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [3.0]]),
        "critic_observations": torch.ones(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[0.0], [1.0], [float("nan")]]
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([False, True, True]),
    }

    prediction = policy._sac_dist_from_mean(
        policy._actor_mean_from_flat(batch["observations"])
    ).mean
    safe_teacher = torch.where(
        batch[DAGGER_TEACHER_ACTION_VALID_KEY].unsqueeze(-1),
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        prediction.detach(),
    )
    with torch.no_grad():
        policy_q = policy.qnet.values(
            policy.qnet(
                batch["critic_observations"],
                policy._q_action_input(prediction.detach()),
            )
        )
        teacher_q = policy.qnet.values(
            policy.qnet(
                batch["critic_observations"],
                policy._q_action_input(safe_teacher),
            )
        )
        weights, _, _ = _spred_p_teacher_probability(policy_q, teacher_q)
    per_row_bc = torch.nn.functional.smooth_l1_loss(
        prediction,
        safe_teacher,
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    expected_filtered_bc = (per_row_bc[:2] * weights[:2]).sum() / 2.0
    expected_filtered_gradient = torch.autograd.grad(
        expected_filtered_bc, policy.actor_adapt.weight, retain_graph=True
    )[0]

    filtered = diagnose_fastsac_actor_gradients(
        policy,
        batch,
        sample_seed=313,
        source_gradients=False,
    )
    ordinary = diagnose_fastsac_actor_gradients(
        policy,
        batch,
        sample_seed=313,
        source_gradients=False,
        use_q_filtered_bc=False,
    )
    policy.cfg.use_q_filtered_bc = False
    counterfactual_filtered = diagnose_fastsac_actor_gradients(
        policy,
        batch,
        sample_seed=313,
        source_gradients=False,
        use_q_filtered_bc=True,
    )

    assert filtered["bc_filter"] == {
        "configured": True,
        "enabled_for_probe": True,
        "overridden": False,
        "weight_semantics": (
            "detached_online_twin_spred_p_teacher_probability"
        ),
    }
    assert filtered["gradients"]["all"]["losses"]["bc"] == pytest.approx(
        expected_filtered_bc.item(), abs=5.0e-6
    )
    assert filtered["gradients"]["all"]["groups"]["actor_mean"][
        "unweighted_norms"
    ]["bc"] == pytest.approx(expected_filtered_gradient.norm().item(), abs=5.0e-6)
    assert filtered["strata"]["all"][
        "spred_p_teacher_probability_mean"
    ] == pytest.approx(weights[:2].mean().item())
    assert ordinary["bc_filter"]["enabled_for_probe"] is False
    assert ordinary["bc_filter"]["overridden"] is True
    assert ordinary["strata"]["all"]["bc_effective_weight_mean"] == 1.0
    assert ordinary["gradients"]["all"]["losses"]["bc"] == pytest.approx(
        per_row_bc[:2].mean().item(), abs=5.0e-6
    )
    assert filtered["gradients"]["all"]["losses"]["bc"] != pytest.approx(
        ordinary["gradients"]["all"]["losses"]["bc"]
    )
    assert counterfactual_filtered["bc_filter"]["configured"] is False
    assert counterfactual_filtered["bc_filter"]["enabled_for_probe"] is True
    assert counterfactual_filtered["bc_filter"]["overridden"] is True
    assert counterfactual_filtered["gradients"]["all"]["losses"][
        "bc"
    ] == pytest.approx(filtered["gradients"]["all"]["losses"]["bc"])
    assert counterfactual_filtered["gradients"]["all"]["groups"][
        "actor_mean"
    ]["unweighted_norms"]["bc"] == pytest.approx(
        filtered["gradients"]["all"]["groups"]["actor_mean"][
            "unweighted_norms"
        ]["bc"]
    )


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


def test_zero_gradient_actor_update_is_noop_despite_nonzero_q_weight_decay():
    policy = _bare_policy(
        eta_sac=0.0,
        lambda_bc=1.0,
        q_weight_decay=0.37,
        sac_actor_weight_decay=0.0,
        dagger_actor_huber_delta=1.0,
        sac_log_std_min=-10.0,
        sac_log_std_max=-2.0,
        sac_max_grad_norm=1.0,
        q_action_input_gain=1.0,
    )
    _install_unit_action_contract(policy)
    _install_tiny_stochastic_actor(policy, actor_weight=0.25, log_std=-3.0)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.critic_optimizer = torch.optim.AdamW(
        policy.qnet.parameters(), lr=0.01, weight_decay=policy.cfg.q_weight_decay
    )
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.4)))
    policy.actor_optimizer = torch.optim.AdamW(
        (
            {
                "params": tuple(policy.actor_adapt.parameters()),
                "weight_decay": policy.cfg.q_weight_decay,
            },
            {
                "params": tuple(policy.bc_dagger_sac_adapter.parameters()),
                "weight_decay": policy.cfg.q_weight_decay,
            },
        ),
        lr=0.01,
    )
    policy._apply_actor_optimizer_weight_decay_contract()
    policy.sac_action_rng = torch.Generator().manual_seed(919)
    policy.actor_update_count = 0
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if name.startswith(("actor_adapt.", "bc_dagger_sac_adapter."))
    }
    observations = torch.tensor([[1.0], [-2.0]])
    with torch.no_grad():
        exact_teacher_actions = policy._sac_dist_from_mean(
            policy._actor_mean_from_flat(observations)
        ).mean.clone()
    batch = {
        "observations": observations,
        "critic_observations": torch.ones(2, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: exact_teacher_actions,
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.ones(2, dtype=torch.bool),
    }

    metrics = policy._actor_update(batch)

    assert [
        group["weight_decay"] for group in policy.actor_optimizer.param_groups
    ] == [0.0, 0.0]
    assert policy.critic_optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        0.37
    )
    assert metrics["exact_bc_loss"].item() == pytest.approx(0.0)
    assert metrics["weighted_sac_actor_loss"].item() == pytest.approx(0.0)
    assert metrics["total_actor_loss"].item() == pytest.approx(0.0)
    for name, expected in before.items():
        assert torch.equal(dict(policy.named_parameters())[name], expected)


def _rollout_owner(
    *,
    prefill: bool,
    beta: float = 0.5,
    teacher_prefill_ppo_noise: bool = False,
    dagger_env_fraction: float | None = None,
):
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
        teacher_prefill_use_ppo_noise=teacher_prefill_ppo_noise,
        dagger_seed=17,
    )
    if dagger_env_fraction is not None:
        cfg.dagger_env_fraction = dagger_env_fraction
    owner = SimpleNamespace(cfg=cfg)
    owner.teacher_prefill_rollout_count = 0 if prefill else 1
    owner.dagger_rollout_count = 0
    owner.dagger_rng = torch.Generator().manual_seed(17)
    owner.sac_rollout_rng = torch.Generator().manual_seed(18)
    owner.sac_action_rng = torch.Generator().manual_seed(19)
    owner.teacher_prefill_action_rng = torch.Generator().manual_seed(20)
    owner._student_raw_action_proposal = lambda td: raw_mean.clone()
    owner._student_mean_action = lambda td: 20.0 * torch.tanh(raw_mean / 20.0)
    owner._teacher_action = lambda td: teacher.clone()
    owner._teacher_action_statistics = lambda td: (
        teacher.clone(),
        torch.full_like(teacher, 0.1),
    )
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


def _rollout_step(
    policy: _DistributionalFastSACDaggerRolloutPolicy,
    batch_size: int,
    *,
    reset: torch.Tensor | None = None,
) -> TensorDict:
    if reset is None:
        reset = torch.zeros(batch_size, dtype=torch.bool)
    return policy(
        TensorDict({"is_init": reset}, batch_size=[batch_size])
    )


def test_partition_alternates_each_step_and_resets_each_env_to_student():
    owner, raw_mean, _ = _rollout_owner(
        prefill=False, beta=1.0, dagger_env_fraction=0.5
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    batch_size = raw_mean.shape[0]

    first = _rollout_step(
        policy, batch_size, reset=torch.ones(batch_size, dtype=torch.bool)
    )
    dagger_env = first[FASTSAC_DAGGER_ENV_KEY]
    student_only = ~dagger_env
    assert dagger_env.sum().item() == batch_size // 2
    assert first[DAGGER_IS_STUDENT_ACTION_KEY].all()

    second = _rollout_step(policy, batch_size)
    assert torch.equal(second[FASTSAC_DAGGER_ENV_KEY], dagger_env)
    assert not second[DAGGER_IS_STUDENT_ACTION_KEY][dagger_env].any()
    assert second[DAGGER_IS_STUDENT_ACTION_KEY][student_only].all()

    third = _rollout_step(policy, batch_size)
    assert third[DAGGER_IS_STUDENT_ACTION_KEY].all()

    reset = torch.zeros(batch_size, dtype=torch.bool)
    reset_index = dagger_env.nonzero(as_tuple=False)[0, 0]
    reset[reset_index] = True
    fourth = _rollout_step(policy, batch_size, reset=reset)
    expected_student = student_only.clone()
    expected_student[reset_index] = True
    assert torch.equal(fourth[DAGGER_IS_STUDENT_ACTION_KEY], expected_student)
    assert fourth[DAGGER_IS_STUDENT_ACTION_KEY][reset_index]

    # Logging receives the collector-owned cached tensor, not a newly sampled
    # or reconstructed partition.
    public_mask = policy.dagger_env_mask(batch_size, "cpu")
    cached_mask = policy.dagger_env_mask(batch_size, "cpu")
    assert public_mask.data_ptr() == cached_mask.data_ptr()
    assert torch.equal(public_mask, dagger_env)
    assert torch.equal(policy.student_only_env_mask(batch_size, "cpu"), student_only)


def test_partition_prefill_remains_all_teacher_control():
    owner, raw_mean, teacher = _rollout_owner(
        prefill=True, beta=0.0, dagger_env_fraction=0.5
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)

    result = _rollout_step(
        policy,
        raw_mean.shape[0],
        reset=torch.ones(raw_mean.shape[0], dtype=torch.bool),
    )

    assert result[FASTSAC_DAGGER_ENV_KEY].sum().item() == raw_mean.shape[0] // 2
    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], teacher)


def test_invalid_student_never_switches_student_only_env_to_teacher():
    owner, raw_mean, teacher = _rollout_owner(
        prefill=False, beta=1.0, dagger_env_fraction=0.5
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    dagger_env = policy.dagger_env_mask(raw_mean.shape[0], "cpu")
    dagger_index = dagger_env.nonzero(as_tuple=False)[0, 0]
    student_only_index = (~dagger_env).nonzero(as_tuple=False)[0, 0]
    raw_mean[dagger_index] = torch.nan
    raw_mean[student_only_index] = torch.nan

    result = _rollout_step(
        policy,
        raw_mean.shape[0],
        reset=torch.ones(raw_mean.shape[0], dtype=torch.bool),
    )

    assert not result[DAGGER_STUDENT_ACTION_VALID_KEY][dagger_index]
    assert not result[DAGGER_STUDENT_ACTION_VALID_KEY][student_only_index]
    # A DAgger Student turn may take the valid Teacher safety fallback.
    assert not result[DAGGER_IS_STUDENT_ACTION_KEY][dagger_index]
    assert torch.equal(result[ACTION_KEY][dagger_index], teacher[dagger_index])
    # The pure cohort keeps honest Student provenance and executes the finite,
    # sanitized Student-mean fallback instead of contaminating its metrics.
    assert result[DAGGER_IS_STUDENT_ACTION_KEY][student_only_index]
    assert torch.isfinite(result[ACTION_KEY][student_only_index]).all()
    assert torch.equal(
        result[ACTION_KEY][student_only_index],
        torch.zeros_like(result[ACTION_KEY][student_only_index]),
    )


def test_failed_dagger_issue_does_not_advance_alternating_phase():
    owner, raw_mean, teacher = _rollout_owner(
        prefill=False, beta=1.0, dagger_env_fraction=0.5
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    dagger_env = policy.dagger_env_mask(raw_mean.shape[0], "cpu")
    failed_index = dagger_env.nonzero(as_tuple=False)[0, 0]
    raw_mean[failed_index] = torch.nan
    teacher[failed_index] = torch.nan

    with pytest.raises(RuntimeError, match="in a DAgger environment"):
        _rollout_step(policy, raw_mean.shape[0])

    raw_mean[failed_index] = 0.0
    teacher[failed_index] = 0.0
    recovered = _rollout_step(policy, raw_mean.shape[0])
    assert recovered[DAGGER_IS_STUDENT_ACTION_KEY].all()


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
    teacher_rng_before = owner.teacher_prefill_action_rng.get_state().clone()

    result = policy(td)

    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], teacher)
    assert torch.equal(result[TD3_EXPLORATORY_STUDENT_ACTION_KEY], teacher)
    assert torch.equal(result[TD3_COLLECTOR_NOISE_KEY], torch.zeros_like(teacher))
    assert torch.equal(
        result[FASTSAC_PREFILL_TEACHER_NOISE_KEY], torch.zeros_like(teacher)
    )
    assert torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)
    assert torch.equal(owner.teacher_prefill_action_rng.get_state(), teacher_rng_before)


def test_teacher_prefill_uses_exact_ppo_gaussian_but_keeps_clean_bc_label():
    owner, _, teacher = _rollout_owner(
        prefill=True, beta=0.0, teacher_prefill_ppo_noise=True
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(teacher.shape[0], dtype=torch.bool)},
        batch_size=[teacher.shape[0]],
    )
    expected_rng = torch.Generator().set_state(
        owner.teacher_prefill_action_rng.get_state()
    )
    expected_action, _ = FastSACPhysicalNormal(
        teacher, torch.full_like(teacher, 0.1)
    ).rsample_with_log_prob(generator=expected_rng)
    teacher_rng_before = owner.teacher_prefill_action_rng.get_state().clone()
    dagger_rng_before = owner.dagger_rng.get_state().clone()
    rollout_rng_before = owner.sac_rollout_rng.get_state().clone()
    learning_rng_before = owner.sac_action_rng.get_state().clone()

    result = policy(td)

    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], expected_action)
    assert torch.equal(result[DAGGER_TEACHER_ACTION_KEY], teacher)
    assert torch.equal(
        result[FASTSAC_PREFILL_TEACHER_NOISE_KEY], expected_action - teacher
    )
    assert not result[FASTSAC_PREFILL_TEACHER_PROJECTION_KEY].any()
    assert not torch.equal(
        owner.teacher_prefill_action_rng.get_state(), teacher_rng_before
    )
    assert torch.equal(owner.dagger_rng.get_state(), dagger_rng_before)
    assert torch.equal(owner.sac_rollout_rng.get_state(), rollout_rng_before)
    assert torch.equal(owner.sac_action_rng.get_state(), learning_rng_before)


def test_prefill_ppo_noise_never_changes_main_teacher_takeover_action():
    owner, _, teacher = _rollout_owner(
        prefill=False, beta=1.0, teacher_prefill_ppo_noise=True
    )
    policy = _DistributionalFastSACDaggerRolloutPolicy(owner)
    td = TensorDict(
        {"is_init": torch.zeros(teacher.shape[0], dtype=torch.bool)},
        batch_size=[teacher.shape[0]],
    )
    teacher_rng_before = owner.teacher_prefill_action_rng.get_state().clone()

    result = policy(td)

    assert not result[DAGGER_IS_STUDENT_ACTION_KEY].any()
    assert torch.equal(result[ACTION_KEY], teacher)
    assert torch.equal(result[DAGGER_TEACHER_ACTION_KEY], teacher)
    assert torch.equal(
        result[FASTSAC_PREFILL_TEACHER_NOISE_KEY], torch.zeros_like(teacher)
    )
    assert torch.equal(owner.teacher_prefill_action_rng.get_state(), teacher_rng_before)


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


def test_twin_scalar_q_has_two_scalar_heads_and_actor_action_gradient():
    rng_before = torch.random.get_rng_state().clone()
    qnet = _build_isolated_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=32,
        device="cpu",
        seed=17,
    )
    rng_after = torch.random.get_rng_state().clone()
    same_seed = _build_isolated_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=32,
        device="cpu",
        seed=17,
    )
    observations = torch.randn(5, 7)
    actions = torch.randn(5, 3, requires_grad=True)

    values = qnet(observations, actions)
    loss = -values.min(dim=0).values.mean()
    loss.backward()

    assert values.shape == (2, 5)
    assert torch.equal(qnet.values(values), values)
    assert not hasattr(qnet, "support")
    assert len(qnet.qnets) == 2
    for head in qnet.qnets:
        assert head.state_stem[0].in_features == 7
        assert head.state_stem[0].out_features == 16
        assert head.action_stem[0].in_features == 3
        assert head.action_stem[0].out_features == 16
        assert head.trunk[0].in_features == 32
        assert head.trunk[0].out_features == 32
        assert head.trunk[2].in_features == 32
        assert head.trunk[2].out_features == 32
        assert head.trunk[4].out_features == 1
    assert not any(isinstance(module, nn.LayerNorm) for module in qnet.modules())
    assert not set(qnet.qnets[0].parameters()).intersection(
        set(qnet.qnets[1].parameters())
    )
    assert torch.equal(rng_before, rng_after)
    for left, right in zip(qnet.parameters(), same_seed.parameters(), strict=True):
        assert torch.equal(left, right)
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum().item() > 0.0


def test_twin_distributional_q_keeps_scalar_topology_and_emits_c51_logits():
    scalar = _build_isolated_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=32,
        device="cpu",
        seed=17,
    )
    distributional = _build_isolated_distributional_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=32,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        device="cpu",
        seed=17,
    )
    observations = torch.randn(5, 7)
    actions = torch.randn(5, 3, requires_grad=True)

    logits = distributional(observations, actions)
    values = distributional.values(logits)
    (-values.min(dim=0).values.mean()).backward()

    assert logits.shape == (2, 5, 51)
    assert values.shape == (2, 5)
    assert torch.equal(
        distributional.support, torch.linspace(-10.0, 10.0, 51)
    )
    assert distributional.state_hidden_dim == scalar.state_hidden_dim == 16
    assert distributional.action_hidden_dim == scalar.action_hidden_dim == 16
    for scalar_head, distributional_head in zip(
        scalar.qnets, distributional.qnets, strict=True
    ):
        assert scalar_head.state_stem[0].in_features == (
            distributional_head.state_stem[0].in_features
        )
        assert scalar_head.state_stem[0].out_features == (
            distributional_head.state_stem[0].out_features
        )
        assert scalar_head.action_stem[0].in_features == (
            distributional_head.action_stem[0].in_features
        )
        assert scalar_head.action_stem[0].out_features == (
            distributional_head.action_stem[0].out_features
        )
        assert scalar_head.trunk[0].in_features == (
            distributional_head.trunk[0].in_features
        )
        assert scalar_head.trunk[2].out_features == (
            distributional_head.trunk[2].out_features
        )
        assert scalar_head.trunk[4].out_features == 1
        assert distributional_head.trunk[4].out_features == 51
    assert not any(
        isinstance(module, nn.LayerNorm) for module in distributional.modules()
    )
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum().item() > 0.0


def test_twin_scalar_q_skateboard_widths_match_balanced_split_design():
    qnet = _build_isolated_scalar_q_network(
        obs_dim=2370,
        action_dim=23,
        hidden_dim=768,
        device="cpu",
        seed=19,
    )

    assert qnet.state_hidden_dim == 384
    assert qnet.action_hidden_dim == 384
    for head in qnet.qnets:
        assert (head.state_stem[0].in_features, head.state_stem[0].out_features) == (
            2370,
            384,
        )
        assert (
            head.action_stem[0].in_features,
            head.action_stem[0].out_features,
        ) == (23, 384)
        assert (head.trunk[0].in_features, head.trunk[0].out_features) == (
            768,
            768,
        )
        assert (head.trunk[2].in_features, head.trunk[2].out_features) == (
            768,
            768,
        )
    assert sum(parameter.numel() for parameter in qnet.parameters()) == 4_203_266


def test_twin_scalar_q_module_and_optimizer_round_trip():
    source = _build_isolated_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=16,
        device="cpu",
        seed=23,
    )
    source_optimizer = torch.optim.Adam(source.parameters(), lr=3.0e-4)
    observations = torch.randn(4, 7)
    actions = torch.randn(4, 3)
    source(observations, actions).square().mean().backward()
    source_optimizer.step()

    restored = _build_isolated_scalar_q_network(
        obs_dim=7,
        action_dim=3,
        hidden_dim=16,
        device="cpu",
        seed=29,
    )
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3.0e-4)
    restored.load_state_dict(source.state_dict())
    restored_optimizer.load_state_dict(source_optimizer.state_dict())

    assert torch.equal(
        source(observations, actions), restored(observations, actions)
    )
    assert len(restored_optimizer.state) == len(source_optimizer.state)


@pytest.mark.parametrize("critic_type", ("scalar", "distributional"))
def test_standard_split_stem_q_routes_actuator_memory_as_state_and_action_once(
    critic_type,
):
    policy = _bare_policy(
        q_critic_type=critic_type,
        q_condition_on_actuator_state=True,
        q_use_predicted_effect=True,
        q_action_input_gain=1.0,
        action_support_clip=20.0,
    )
    policy.action_dim = 2
    policy._q_critic_dim = 3
    policy._q_actuator_parameter_context_dim = 6
    policy._q_actuator_context_dim = 8
    policy._q_state_input_dim = 11
    policy._q_action_input_dim = 2
    policy._fastsac_q_action_center = torch.tensor([1.0, -1.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0, 4.0])
    policy._fastsac_action_low = torch.tensor([-20.0, -20.0])
    policy._fastsac_action_high = torch.tensor([20.0, 20.0])
    critic_observations = torch.tensor([[0.1, -0.2, 0.3]])
    candidate = torch.tensor([[3.0, 3.0]], requires_grad=True)
    context = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1.0, 7.0]],
        requires_grad=True,
    )

    q_state, q_action = policy._q_network_inputs(
        critic_observations, candidate, context
    )

    expected_previous = torch.tensor([[-1.0, 2.0]])
    expected_state = torch.cat(
        (critic_observations, context[:, :6].detach(), expected_previous), dim=-1
    )
    assert torch.equal(q_state, expected_state)
    assert torch.equal(q_action, torch.tensor([[1.0, 1.0]]))
    assert q_state.shape[-1] == 3 + 6 + 2
    assert q_action.shape[-1] == 2
    changed_state, changed_action = policy._q_network_inputs(
        critic_observations,
        torch.tensor([[-1.0, -5.0]]),
        context,
    )
    assert torch.equal(changed_state, q_state)
    assert not torch.equal(changed_action, q_action.detach())

    (q_state.sum() + q_action.sum()).backward()
    assert torch.allclose(candidate.grad, torch.tensor([[0.5, 0.25]]))
    assert context.grad is None


@pytest.mark.parametrize(
    "legacy_architecture",
    [
        "monolithic_action_conditioned_c51_logits_v1",
        "twin_independent_normalized_state_action_concat_scalar_mlp_v1",
    ],
)
def test_scalar_checkpoint_rejects_legacy_topology(legacy_architecture):
    policy = _bare_policy(q_critic_type="scalar", q_hidden_dim=768)
    policy._q_state_input_dim = 2370
    policy._q_action_input_dim = 23
    legacy = {
        "q_backend_config": {
            "q_architecture_semantics": legacy_architecture,
            "q_state_input_dim": 2341,
            "q_action_input_dim": 121,
        }
    }

    with pytest.raises(ValueError, match="balanced split-stem scalar"):
        policy._validate_scalar_q_checkpoint_architecture(
            legacy, context="unit test"
        )

    current = {
        "q_backend_config": {
            "q_architecture_semantics": (
                FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS
            ),
            "q_state_input_dim": 2370,
            "q_action_input_dim": 23,
            "q_state_hidden_dim": 384,
            "q_action_hidden_dim": 384,
            "q_action_fusion": "balanced_split_stems",
            "q_action_fusion_semantics": (
                FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS
            ),
        }
    }
    policy._validate_scalar_q_checkpoint_architecture(
        current, context="unit test"
    )


def test_scalar_checkpoint_architecture_validation_does_not_affect_c51():
    policy = _bare_policy(q_critic_type="c51")
    policy._validate_scalar_q_checkpoint_architecture({}, context="unit test")


def test_distributional_checkpoint_requires_split_stem_c51_architecture():
    policy = _bare_policy(q_critic_type="distributional", q_hidden_dim=32)
    policy._q_state_input_dim = 11
    policy._q_action_input_dim = 3
    current = {
        "q_backend_config": {
            "q_architecture_semantics": (
                FASTSAC_STANDARD_DISTRIBUTIONAL_Q_ARCHITECTURE_SEMANTICS
            ),
            "q_state_input_dim": 11,
            "q_action_input_dim": 3,
            "q_state_hidden_dim": 16,
            "q_action_hidden_dim": 16,
            "q_action_fusion": "balanced_split_stems",
            "q_action_fusion_semantics": (
                FASTSAC_STANDARD_SCALAR_Q_FUSION_SEMANTICS
            ),
        }
    }

    policy._validate_scalar_q_checkpoint_architecture(
        current, context="unit test"
    )
    current["q_backend_config"]["q_architecture_semantics"] = (
        FASTSAC_STANDARD_SCALAR_Q_ARCHITECTURE_SEMANTICS
    )
    with pytest.raises(ValueError, match="split-stem Q architecture"):
        policy._validate_scalar_q_checkpoint_architecture(
            current, context="unit test"
        )


def test_q_twin_reduction_checkpoint_contract_defaults_legacy_to_min():
    minimum = _bare_policy(q_twin_reduction=Q_TWIN_REDUCTION_MIN)
    minimum._validate_q_twin_reduction_checkpoint({}, context="unit test")

    mean = _bare_policy(q_twin_reduction=Q_TWIN_REDUCTION_MEAN)
    state = {
        "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
        "q_backend_config": {
            "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
        },
        "dagger_backend_config": {
            "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
        },
    }
    mean._validate_q_twin_reduction_checkpoint(state, context="unit test")
    with pytest.raises(ValueError, match="Q twin reduction mismatch"):
        mean._validate_q_twin_reduction_checkpoint({}, context="unit test")
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        mean._validate_q_twin_reduction_checkpoint(
            {
                "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
                "q_backend_config": {
                    "q_twin_reduction": Q_TWIN_REDUCTION_MIN,
                },
            },
            context="unit test",
        )
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        mean._validate_q_twin_reduction_checkpoint(
            {
                "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
                "q_backend_config": {
                    "q_twin_reduction": Q_TWIN_REDUCTION_MEAN,
                },
                "dagger_backend_config": {
                    "q_twin_reduction": Q_TWIN_REDUCTION_MIN,
                },
            },
            context="unit test",
        )


def test_mean_q_twin_reduction_has_distinct_learning_semantics(monkeypatch):
    policy = _bare_policy(
        q_critic_type="distributional",
        q_twin_reduction=Q_TWIN_REDUCTION_MEAN,
        q_num_atoms=501,
        q_v_min=-20.0,
        q_v_max=20.0,
        sac_action_distribution=NORMALIZED_TANH_ACTION_DISTRIBUTION,
        sac_alpha_update_cadence="actor",
        sac_policy_frequency=1,
        sac_actor_observation_mode=STUDENT_PERCEPTION_ACTOR_OBSERVATION_MODE,
        use_q_filtered_bc=False,
    )
    policy._fastsac_student_action_contract = {}
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.target_entropy = -1.0
    monkeypatch.setattr(
        DistributionalTD3TeacherBC,
        "_q_backend_metadata",
        lambda owner: {},
    )

    assert "mean_complete" in policy._critic_learning_semantics()
    assert "twin_mean" in policy._actor_learning_semantics()
    metadata = policy._q_backend_metadata()
    assert metadata["q_twin_reduction"] == Q_TWIN_REDUCTION_MEAN
    assert metadata["clipped_double_q"] is False
    assert metadata["clipped_double_distribution"] is False
    assert metadata["target_q_reduction"] == "mean_complete_twin_target"
    assert metadata["actor_q_reduction"] == "mean_online_twin_expectations"

@pytest.mark.parametrize(
    ("q_twin_reduction", "expected_continuing_target"),
    ((Q_TWIN_REDUCTION_MIN, 2.89), (Q_TWIN_REDUCTION_MEAN, 3.34)),
)
def test_scalar_soft_target_uses_configured_twin_q_and_terminal_cut(
    q_twin_reduction,
    expected_continuing_target,
):
    policy = _bare_policy(
        gamma=0.9,
        q_action_input_gain=1.0,
        q_critic_type="scalar",
        q_n_step=1,
        q_twin_reduction=q_twin_reduction,
        sac_use_autotune=True,
    )
    _install_unit_action_contract(policy)
    policy.qnet_target = _FixedTwinScalarQ(
        torch.tensor([[3.0, 4.0], [2.0, 5.0]])
    )
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2)))
    policy.sac_action_rng = torch.Generator().manual_seed(91)
    policy._actor_dist_from_flat = lambda observations: _FixedStochasticDist(
        observations.shape[0], action=0.25, log_prob=-0.5
    )
    batch = {
        "next_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([False, True]),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    target, diagnostics, _ = policy._distributional_fastsac_target(batch)

    # Row 0 is 2.89 for min(3,2) and 3.34 for mean(3,2).
    # Row 1 is a true terminal and therefore receives neither entropy nor Q'.
    assert torch.allclose(
        target, torch.tensor([expected_continuing_target, 2.0])
    )
    assert target.grad_fn is None
    assert diagnostics["target_select_q1_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["target_select_q2_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["target_q1_contribution_fraction"].item() == pytest.approx(
        0.5
    )
    assert diagnostics["target_q2_contribution_fraction"].item() == pytest.approx(
        0.5
    )
    assert diagnostics["reduced_target_expected_mean"].item() == pytest.approx(
        (expected_continuing_target + 2.0) / 2.0
    )
    assert diagnostics["support_clip_fraction_mean"].item() == 0.0
    assert diagnostics["target_distribution_entropy"].item() == 0.0


def test_scalar_critic_update_uses_twin_bellman_mse():
    policy = _bare_policy(
        q_critic_type="scalar",
        q_condition_on_actuator_state=False,
        q_action_input_gain=1.0,
        action_support_clip=20.0,
        sac_max_grad_norm=1.0e6,
        sac_alpha_update_cadence="actor",
        sac_policy_frequency=2,
        sac_use_autotune=False,
        sac_tau=0.005,
    )
    _install_unit_action_contract(policy)
    policy._q_critic_dim = 1
    policy._q_state_input_dim = 1
    policy._q_action_input_dim = 1
    policy.qnet = _build_isolated_scalar_q_network(
        obs_dim=1,
        action_dim=1,
        hidden_dim=8,
        device="cpu",
        seed=23,
    )
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = nn.Parameter(torch.tensor(-20.0))
    policy.critic_update_count = 0
    policy.alpha_update_count = 0
    policy._student_actor_warmup_active = lambda: False
    scalar_target = torch.tensor([1.5, -0.5])
    policy._distributional_fastsac_target = MethodType(
        lambda owner, batch: (scalar_target, {}, torch.zeros(2)), policy
    )
    batch = {
        "critic_observations": torch.tensor([[0.2], [-0.4]]),
        "actions": torch.tensor([[0.3], [0.1]]),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
    }
    with torch.no_grad():
        before = policy._q_forward(
            policy.qnet,
            batch["critic_observations"],
            batch["actions"],
            None,
        )
        expected_per_head = torch.stack(
            [torch.nn.functional.mse_loss(head, scalar_target) for head in before]
        )

    metrics = policy._critic_update(batch)

    assert policy.critic_optimizer.step_calls == 1
    assert policy.critic_update_count == 1
    assert torch.allclose(
        torch.stack((metrics["critic_loss_1"], metrics["critic_loss_2"])),
        expected_per_head,
    )
    assert metrics["critic_loss"].item() == pytest.approx(
        expected_per_head.sum().item()
    )


def test_distributional_split_stem_critic_update_uses_c51_cross_entropy():
    policy = _bare_policy(
        q_critic_type="distributional",
        q_condition_on_actuator_state=False,
        q_action_input_gain=1.0,
        action_support_clip=20.0,
        sac_max_grad_norm=1.0e6,
        sac_alpha_update_cadence="actor",
        sac_policy_frequency=2,
        sac_use_autotune=False,
        sac_tau=0.005,
    )
    _install_unit_action_contract(policy)
    policy._q_critic_dim = 1
    policy._q_state_input_dim = 1
    policy._q_action_input_dim = 1
    policy.qnet = _build_isolated_distributional_scalar_q_network(
        obs_dim=1,
        action_dim=1,
        hidden_dim=8,
        num_atoms=3,
        v_min=-1.0,
        v_max=1.0,
        device="cpu",
        seed=23,
    )
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = nn.Parameter(torch.tensor(-20.0))
    policy.critic_update_count = 0
    policy.alpha_update_count = 0
    policy._student_actor_warmup_active = lambda: False
    categorical_target = torch.tensor(
        [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]
    )
    policy._distributional_fastsac_target = MethodType(
        lambda owner, batch: (categorical_target, {}, torch.zeros(2)), policy
    )
    batch = {
        "critic_observations": torch.tensor([[0.2], [-0.4]]),
        "actions": torch.tensor([[0.3], [0.1]]),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
    }
    with torch.no_grad():
        before = policy._q_forward(
            policy.qnet,
            batch["critic_observations"],
            batch["actions"],
            None,
        )
        expected_per_head = -(
            categorical_target.unsqueeze(0)
            * torch.nn.functional.log_softmax(before, dim=-1)
        ).sum(dim=-1).mean(dim=-1)

    metrics = policy._critic_update(batch)

    assert policy.critic_optimizer.step_calls == 1
    assert policy.critic_update_count == 1
    assert torch.allclose(
        torch.stack((metrics["critic_loss_1"], metrics["critic_loss_2"])),
        expected_per_head,
    )
    assert metrics["critic_loss"].item() == pytest.approx(
        expected_per_head.sum().item()
    )


@pytest.mark.parametrize("critic_type", ("c51", "distributional"))
@pytest.mark.parametrize(
    "q_twin_reduction", (Q_TWIN_REDUCTION_MIN, Q_TWIN_REDUCTION_MEAN)
)
def test_soft_c51_target_uses_configured_complete_twin_distribution_detached(
    critic_type, q_twin_reduction,
):
    policy = _bare_policy(
        gamma=1.0,
        q_action_input_gain=1.0,
        sac_use_autotune=True,
        q_critic_type=critic_type,
        q_twin_reduction=q_twin_reduction,
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
    expected_source = (
        second
        if q_twin_reduction == Q_TWIN_REDUCTION_MIN
        else 0.5 * (first + second)
    )
    expected, _, _ = _project_c51_probabilities(
        expected_source,
        rewards=torch.full((2,), 0.1),
        bootstrap=torch.ones(2),
        effective_discount=torch.ones(2),
        support=policy.qnet_target.support,
    )
    assert torch.allclose(projected, expected)
    assert projected.grad_fn is None
    assert projected.requires_grad is False
    expected_q2_fraction = (
        1.0 if q_twin_reduction == Q_TWIN_REDUCTION_MIN else 0.5
    )
    assert diagnostics["target_select_q2_fraction"].item() == pytest.approx(
        expected_q2_fraction
    )
    assert diagnostics["target_q1_contribution_fraction"].item() == pytest.approx(
        1.0 - expected_q2_fraction
    )
    assert diagnostics["target_q2_contribution_fraction"].item() == pytest.approx(
        expected_q2_fraction
    )
    assert diagnostics["reduced_target_expected_mean"].item() == pytest.approx(
        (projected * policy.qnet_target.support).sum(dim=-1).mean().item()
    )
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

    policy.critic_update_count = 1
    policy.cfg.perception_replay_mode = ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    policy.cfg.teacher_perception_warmup_steps = 2
    policy._teacher_prefill_complete = True
    policy.dagger_rollout_count = 0
    policy._fastsac_rollout_critic_metrics = []
    warmup_alpha_before = policy.log_alpha.detach().clone()
    warmup = policy._critic_update(batch)

    assert torch.equal(policy.log_alpha, warmup_alpha_before)
    assert policy.alpha_update_count == 1
    assert warmup["alpha_update_due_fraction"].item() == 0.0
    assert warmup["alpha_update_performed_fraction"].item() == 0.0
    assert policy._fastsac_rollout_critic_metrics == [warmup]
    policy.dagger_rollout_count = 2

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
        q_seed=seed,
        teacher_prefill_use_ppo_noise=True,
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
    policy.teacher_prefill_action_rng = torch.Generator().manual_seed(seed + 6)
    policy.actor_target = None
    policy._last_fastsac_diagnostics = {}
    policy.joint_names = ["unit_joint"]
    _install_unit_action_contract(policy)
    policy._fastsac_action_contract = {
        "joint_names": ["unit_joint"],
        "fingerprint": "sha256:unit-test",
    }
    policy._configure_student_action_support()
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
            source.teacher_prefill_action_rng,
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
    assert "teacher_prefill_action_rng_state" in state
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
        "teacher_prefill_action_rng",
        "teacher_perception_rng",
    ):
        source_generator = getattr(source, name)
        restored_generator = getattr(restored, name)
        assert torch.equal(source_generator.get_state(), restored_generator.get_state())
        assert torch.equal(
            torch.rand(8, generator=source_generator),
            torch.rand(8, generator=restored_generator),
        )


def test_legacy_checkpoint_actor_q_decay_is_sanitized_without_touching_critic():
    source = _checkpoint_policy(1200)
    source.critic_optimizer.param_groups[0]["weight_decay"] = 0.019
    state = copy.deepcopy(source._fastsac_checkpoint_state())
    state.pop("actor_mean_optimizer_semantics")
    state.pop("actor_mean_weight_decay")
    state["optimizer_resume_state"]["actor_optimizer"]["param_groups"][0][
        "weight_decay"
    ] = 0.019

    restored = _checkpoint_policy(2200)
    restored._load_fastsac_checkpoint_state(state)

    assert restored.actor_optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        0.0
    )
    assert restored.critic_optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        0.019
    )
    assert (
        restored.actor_optimizer.state_dict()["state"]
        == source.actor_optimizer.state_dict()["state"]
    )


def test_checkpoint_rejects_actor_optimizer_group_that_conflicts_with_contract():
    source = _checkpoint_policy(1250)
    state = copy.deepcopy(source._fastsac_checkpoint_state())
    state["optimizer_resume_state"]["actor_optimizer"]["param_groups"][0][
        "weight_decay"
    ] = 0.019

    restored = _checkpoint_policy(2250)
    with pytest.raises(ValueError, match="disagrees.*weight-decay contract"):
        restored._load_fastsac_checkpoint_state(state)


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
        source.teacher_prefill_action_rng,
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
            "teacher_prefill_action_rng",
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
    policy.joint_names = ["unit_joint"]
    _install_unit_action_contract(policy)
    policy._fastsac_action_contract = {
        "joint_names": ["unit_joint"],
        "fingerprint": "sha256:raw-unit-test",
    }
    policy._configure_student_action_support()
    saved_contract = {"joint_names": ["unit_joint"]}
    if saved_fingerprint is not None:
        saved_contract["fingerprint"] = saved_fingerprint
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": CHECKPOINT_VERSION,
        "actor_backend": ACTOR_BACKEND,
        "action_contract": saved_contract,
        "student_action_contract": copy.deepcopy(
            policy._fastsac_student_action_contract
        ),
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


def test_fastsac_target_routes_only_next_actuator_context_to_q_action_branch():
    policy = _bare_policy(
        gamma=1.0,
        q_action_input_gain=1.0,
        q_condition_on_actuator_state=True,
    )
    _install_unit_action_contract(policy)
    policy._q_actuator_context_dim = 6
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    policy.qnet_target = _RecordingTableTwinC51(
        probabilities, probabilities
    ).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.tensor(-20.0))
    policy.sac_action_rng = torch.Generator().manual_seed(91)
    policy._actor_dist_from_flat = lambda observations: _FixedStochasticDist(
        observations.shape[0], action=0.25, log_prob=0.0
    )
    next_context = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.25],
        ]
    )
    batch = {
        "next_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        NEXT_Q_ACTUATOR_CONTEXT_KEY: next_context,
    }

    policy._distributional_fastsac_target(batch)

    assert len(policy.qnet_target.action_inputs) == 1
    expected = torch.cat((torch.full((2, 1), 0.25), next_context), dim=-1)
    assert torch.equal(policy.qnet_target.action_inputs[0], expected)


def test_fastsac_critic_routes_current_actuator_context_to_factual_action():
    policy = _bare_policy(
        q_action_input_gain=1.0,
        q_condition_on_actuator_state=True,
        sac_max_grad_norm=1.0,
        sac_alpha_update_cadence="actor",
        sac_policy_frequency=2,
        sac_use_autotune=False,
        sac_tau=0.005,
    )
    _install_unit_action_contract(policy)
    policy._q_actuator_context_dim = 6
    policy.qnet = _RecordingActionSensitiveTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = nn.Parameter(torch.tensor(-20.0))
    policy.critic_update_count = 0
    policy.alpha_update_count = 0
    projected = torch.full((3, 3), 1.0 / 3.0)
    policy._distributional_fastsac_target = MethodType(
        lambda owner, batch: (projected, {}, torch.zeros(3)), policy
    )
    current_context = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    actions = torch.tensor([[0.1], [0.2], [0.3]])
    batch = {
        "critic_observations": torch.ones(3, 1),
        "actions": actions,
        "dones": torch.zeros(3, dtype=torch.bool),
        "truncations": torch.zeros(3, dtype=torch.bool),
        Q_ACTUATOR_CONTEXT_KEY: current_context,
    }

    policy._critic_update(batch)

    assert len(policy.qnet.action_inputs) == 1
    assert torch.equal(
        policy.qnet.action_inputs[0], torch.cat((actions, current_context), dim=-1)
    )


def test_fastsac_actor_and_spred_share_detached_current_actuator_context():
    policy = _tiny_physical_policy()
    policy.cfg.q_condition_on_actuator_state = True
    policy.cfg.use_q_filtered_bc = True
    policy._q_actuator_context_dim = 6
    policy.qnet = _RecordingActionSensitiveTwinC51(
        first_slope=1.0, second_slope=2.0
    )
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    context = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    batch = _tiny_physical_batch()
    batch[Q_ACTUATOR_CONTEXT_KEY] = context

    policy._actor_update(batch)

    # One SAC sampled-action query, followed by SPReD-P Student-mean and
    # Teacher-label queries. All compare actions under the identical dynamics.
    assert len(policy.qnet.action_inputs) == 3
    for action_input in policy.qnet.action_inputs:
        assert action_input.shape == (3, 7)
        assert torch.equal(action_input[:, 1:], context.detach())
    assert context.grad is None
