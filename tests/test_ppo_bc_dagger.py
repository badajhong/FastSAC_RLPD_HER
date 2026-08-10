from __future__ import annotations

import copy
import importlib
from types import MethodType, SimpleNamespace

import h5py
import pytest
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

bc_dagger_module = importlib.import_module(
    "active_adaptation.learning.ppo.ppo_bc_dagger"
)
from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_vel import (
    CMD_KEY,
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    PPOVEL,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_FINALIZATION_PHASES,
    DAGGER_FINALIZATION_SEMANTICS,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_ACTION_DISCREPANCY_MAX_KEY,
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_CONTROL_SEMANTICS,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    DAGGER_SAFE_RELEASE_KEY,
    DAGGER_SAFE_TAKEOVER_KEY,
    DAGGER_SAFE_TEACHER_KEY,
    DAGGER_SAFE_UNSAFE_KEY,
    DAGGER_STUDENT_ACTION_VALID_KEY,
    PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
    PPO_BC_DAGGER_IQL_CRITIC_SEMANTICS,
    PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM,
    PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS,
    PPO_BC_DAGGER_TRAINING_ALGORITHM,
    _DaggerRolloutPolicy,
    _DaggerTeacherReplayBuffer,
    _DeviceReplay,
    PPOBCDaggerFinetune,
    PPOBCDaggerFinetuneConfig,
    _iql_expectile_loss,
    _normalized_action_discrepancy,
    _project_scalar_to_c51,
    _valid_teacher_action_rows,
)
from active_adaptation.learning.ppo.fastsac_vel import (
    BCDaggerOfflineReplayH5,
    TEACHER_REPLAY_FIELDS,
)


EXPECTED_DAGGER_ALGORITHM = PPO_BC_DAGGER_TRAINING_ALGORITHM
TEACHER_ACTION_KEY = "teacher_action"
TEACHER_ACTION_VALID_KEY = "teacher_action_valid"
IS_STUDENT_ACTION_KEY = "is_student_action"


def _bare_policy(**cfg):
    policy = PPOBCDaggerFinetune.__new__(PPOBCDaggerFinetune)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    return policy


def test_config_store_registers_exact_student_observation_surface():
    store = ConfigStore.instance()
    dagger = store.load("algo/ppo_bc_dagger_finetune.yaml").node
    ppo_student = store.load("algo/ppo_vel_finetune.yaml").node

    assert dagger._target_.endswith(
        "ppo_bc_dagger.PPOBCDaggerFinetune"
    )
    assert dagger.name == "ppo_bc_dagger"
    assert dagger.phase == "finetune"
    assert dagger.vecnorm == "eval"
    assert dagger.use_depth is True
    assert dagger.enable_residual_distillation is False
    assert list(dagger.in_keys) == list(ppo_student.in_keys)
    assert list(dagger.in_keys) == [
        CMD_KEY,
        OBS_KEY,
        OBJECT_KEY,
        OBS_PRIV_KEY,
        OBJECT_GEO_KEY,
        VEL_CMD_KEY,
        DEPTH_KEY,
    ]

    assert dagger.dagger_beta_start == pytest.approx(1.0)
    assert dagger.dagger_beta_end == pytest.approx(0.0)
    assert dagger.dagger_beta_decay_rollouts == 4000
    assert dagger.dagger_beta_zero_iteration is None
    assert dagger.dagger_control_mode == "safe"
    assert dagger.dagger_safe_takeover_rms == pytest.approx(0.006)
    assert dagger.dagger_safe_release_rms == pytest.approx(0.004)
    assert dagger.dagger_safe_min_teacher_steps == 8
    assert dagger.dagger_safe_zero_iteration is None
    assert dagger.dagger_bc_lr > 0.0
    assert dagger.dagger_bc_epochs > 0
    assert dagger.dagger_actor_huber_delta == pytest.approx(1.0)
    assert dagger.dagger_action_clip == pytest.approx(20.0)
    assert dagger.dagger_batch_size == 4096
    assert dagger.dagger_replay_raw_observations is True
    assert set(dagger.replay_raw_observation_keys) == {
        VEL_CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        CMD_KEY,
    }
    assert DEPTH_KEY not in dagger.replay_raw_observation_keys
    assert dagger.save_teacher_buffer is True
    # Stage 1 builds the exact critic topology and action coordinates consumed
    # by the default Stage-2 FastSAC backend; transfer must not require a hidden
    # 101->501 atom conversion or raw->normalized action reinterpretation.
    assert dagger.q_num_atoms == 501
    assert dagger.q_v_min < dagger.q_v_max
    assert dagger.q_layer_norm is True
    assert dagger.q_action_fusion == "early"
    assert dagger.q_action_coordinates == "absolute"
    assert dagger.sac_q_normalize_actions is True
    assert dagger.sac_q_action_input_gain == pytest.approx(1.0)
    assert dagger.sac_clipped_double_q is True
    assert dagger.q_batch_size == 512
    assert dagger.q_updates_per_rollout == 128
    assert dagger.q_teacher_replay_ratio == pytest.approx(0.5)
    assert dagger.q_learning_starts_per_source == 8_192
    assert dagger.q_teacher_buffer_capacity > 0


def test_config_dataclass_has_stage_local_beta_and_separate_q_targets():
    cfg = PPOBCDaggerFinetuneConfig()

    for name in (
        "dagger_control_mode",
        "dagger_safe_takeover_rms",
        "dagger_safe_release_rms",
        "dagger_safe_min_teacher_steps",
        "dagger_safe_zero_iteration",
        "dagger_beta_start",
        "dagger_beta_end",
        "dagger_beta_decay_rollouts",
        "dagger_beta_zero_iteration",
        "dagger_seed",
        "dagger_bc_lr",
        "dagger_bc_epochs",
        "dagger_actor_huber_delta",
        "dagger_buffer_capacity",
        "dagger_buffer_device",
        "dagger_batch_size",
        "dagger_updates_per_rollout",
        "q_hidden_dim",
        "q_num_atoms",
        "q_v_min",
        "q_v_max",
        "q_layer_norm",
        "q_action_fusion",
        "q_action_coordinates",
        "sac_q_normalize_actions",
        "sac_q_action_input_gain",
        "sac_clipped_double_q",
        "q_lr",
        "q_weight_decay",
        "q_seed",
        "q_tau",
        "q_max_grad_norm",
        "q_batch_size",
        "q_updates_per_rollout",
        "q_teacher_replay_ratio",
        "q_learning_starts_per_source",
        "q_teacher_buffer_capacity",
        "sac_bc_initial_action_std",
        "sac_bc_log_std_min",
        "sac_bc_log_std_max",
        "sac_alpha_init",
        "sac_entropy_reference_scale",
    ):
        assert hasattr(cfg, name), name


def test_bc_critic_rejects_layernorm_override_at_startup():
    cfg = PPOBCDaggerFinetuneConfig(q_layer_norm=False)

    with pytest.raises(ValueError, match="LayerNorm"):
        PPOBCDaggerFinetune._validate_config(cfg)


@pytest.mark.parametrize("value", [0, True, 1.5, "10"])
def test_safe_zero_iteration_rejects_non_positive_integer_values(value):
    cfg = PPOBCDaggerFinetuneConfig(dagger_safe_zero_iteration=value)

    with pytest.raises(ValueError, match="dagger_safe_zero_iteration"):
        PPOBCDaggerFinetune._validate_config(cfg)


def test_safe_zero_iteration_rejects_beta_only_control():
    cfg = PPOBCDaggerFinetuneConfig(
        dagger_control_mode="beta", dagger_safe_zero_iteration=10
    )

    with pytest.raises(ValueError, match="requires safe or hybrid"):
        PPOBCDaggerFinetune._validate_config(cfg)


def test_bc_critic_checkpoint_metadata_matches_fastsac_q_contract():
    cfg = PPOBCDaggerFinetuneConfig()
    policy = _bare_policy()
    policy.cfg = cfg
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 3
    policy._q_critic_dim = 5
    policy._q_input_dim = 5
    policy.action_dim = 2
    policy.reward_groups = ["task"]

    dagger_backend = policy._checkpoint_config()
    assert "dagger_beta_zero_iteration" not in dagger_backend
    assert dagger_backend["dagger_safe_zero_iteration"] is None
    q_backend = policy._q_backend_metadata()

    assert dagger_backend["q_num_atoms"] == 501
    assert dagger_backend["q_layer_norm"] is True
    assert dagger_backend["q_action_fusion"] == "early"
    assert dagger_backend["q_action_coordinates"] == "absolute"
    assert dagger_backend["sac_q_normalize_actions"] is True
    assert dagger_backend["sac_q_action_input_gain"] == pytest.approx(1.0)
    assert dagger_backend["sac_clipped_double_q"] is True
    assert q_backend["num_atoms"] == 501
    assert q_backend["layer_norm"] is True
    assert q_backend["q_action_fusion"] == "early"
    assert q_backend["q_action_coordinates"] == "absolute"
    assert q_backend["q_action_normalized"] is True
    assert q_backend["q_action_input_gain"] == pytest.approx(1.0)
    assert q_backend["clipped_double_q"] is True


def test_bc_critic_normalizes_only_q_input_and_keeps_executed_action_physical():
    policy = _bare_policy(
        q_action_coordinates="absolute",
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
        dagger_action_clip=20.0,
    )
    policy._fastsac_q_action_low = torch.tensor([-20.0, -20.0])
    policy._fastsac_q_action_scale = torch.tensor([20.0, 20.0])
    executed = torch.tensor(
        [[-20.0, 0.0], [10.0, 20.0]], dtype=torch.float32
    )
    physical_copy = executed.clone()

    q_action = policy._q_action_input(executed)

    assert torch.equal(
        q_action, torch.tensor([[-1.0, 0.0], [0.5, 1.0]])
    )
    # Normalization is a critic-input transform, never a replay mutation.
    assert torch.equal(executed, physical_copy)


@pytest.mark.parametrize(
    ("actions", "threshold", "expected"),
    (
        (
            [[0.0, 1.0], [20.0, -20.0], [20.01, 0.0]],
            20.0,
            [True, True, False],
        ),
        (
            [[float("nan"), 0.0], [float("inf"), 0.0], [1.0, 2.0]],
            20.0,
            [False, False, True],
        ),
    ),
)
def test_teacher_action_validity_rejects_nonfinite_and_outlier_rows(
    actions, threshold, expected
):
    actual = _valid_teacher_action_rows(
        torch.tensor(actions), threshold=threshold
    )

    assert actual.dtype is torch.bool
    assert actual.shape == (len(actions),)
    assert torch.equal(actual, torch.tensor(expected))


def test_safe_dagger_discrepancy_is_normalized_rms_and_scale_invariant():
    student_10 = torch.tensor([[2.0, -2.0]])
    teacher_10 = torch.zeros_like(student_10)
    student_20 = student_10 * 2.0
    teacher_20 = teacher_10.clone()
    originals = tuple(
        value.clone()
        for value in (student_10, teacher_10, student_20, teacher_20)
    )

    rms_10, max_10 = _normalized_action_discrepancy(
        student_10, teacher_10, 10.0
    )
    rms_20, max_20 = _normalized_action_discrepancy(
        student_20, teacher_20, 20.0
    )

    assert rms_10.item() == pytest.approx(0.2)
    assert max_10.item() == pytest.approx(0.2)
    assert torch.equal(rms_10, rms_20)
    assert torch.equal(max_10, max_20)
    for actual, expected in zip(
        (student_10, teacher_10, student_20, teacher_20), originals
    ):
        assert torch.equal(actual, expected)


def _safe_rollout_fixture(*, envs=1, hold=3):
    policy = _bare_policy(
        dagger_control_mode="safe",
        dagger_safe_takeover_rms=0.2,
        dagger_safe_release_rms=0.1,
        dagger_safe_min_teacher_steps=hold,
        dagger_beta_start=1.0,
        dagger_beta_end=1.0,
        dagger_beta_decay_rollouts=1,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=10.0,
    )
    policy.dagger_rollout_count = 0
    policy.dagger_rng = torch.Generator().manual_seed(123)
    policy.test_student_action = torch.zeros(envs, 1)
    policy.test_teacher_action = torch.zeros(envs, 1)
    policy._student_action = lambda td: policy.test_student_action.clone()
    policy._teacher_action = lambda td: policy.test_teacher_action.clone()
    return policy, _DaggerRolloutPolicy(policy)


def _safe_step(wrapper, is_init=None):
    envs = wrapper._owner.test_student_action.shape[0]
    if is_init is None:
        is_init = torch.zeros(envs, dtype=torch.bool)
    td = TensorDict({"is_init": is_init}, batch_size=[envs])
    return wrapper(td)


def test_safe_dagger_hysteresis_and_minimum_hold_are_per_environment():
    policy, wrapper = _safe_rollout_fixture(envs=2, hold=3)
    # Enter is strict: row 0 exceeds 0.2, row 1 equals it and remains student.
    policy.test_student_action = torch.tensor([[2.1], [2.0]])
    first = _safe_step(wrapper)
    assert torch.equal(
        first[DAGGER_IS_STUDENT_ACTION_KEY], torch.tensor([False, True])
    )
    assert torch.equal(
        first[DAGGER_SAFE_TAKEOVER_KEY], torch.tensor([True, False])
    )

    # The trigger step counts as hold step one. Even a zero-error student is
    # held for exactly two more calls, then released on the fourth call.
    policy.test_student_action.zero_()
    second = _safe_step(wrapper)
    third = _safe_step(wrapper)
    fourth = _safe_step(wrapper)
    assert second[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [False, True]
    assert third[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [False, True]
    assert fourth[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [True, True]
    assert fourth[DAGGER_SAFE_RELEASE_KEY].tolist() == [True, False]


def test_safe_dagger_release_boundary_and_episode_reset_clear_latch():
    policy, wrapper = _safe_rollout_fixture(envs=2, hold=1)
    policy.test_student_action = torch.tensor([[3.0], [3.0]])
    _safe_step(wrapper)

    # Equality with the release threshold retains the current controller.
    policy.test_student_action = torch.tensor([[1.0], [1.0]])
    boundary = _safe_step(wrapper)
    assert boundary[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [False, False]

    # Reset clears only row 0. Row 1 remains latched in the hysteresis band.
    reset = _safe_step(wrapper, torch.tensor([True, False]))
    assert reset[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [True, False]

    policy.test_student_action = torch.tensor([[0.9], [0.9]])
    released = _safe_step(wrapper)
    assert released[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [True, True]


def test_safe_mode_ignores_beta_rng_and_invalid_student_forces_teacher():
    policy, wrapper = _safe_rollout_fixture(envs=2, hold=1)
    before = policy.dagger_rng.get_state().clone()
    policy.test_student_action = torch.tensor([[float("nan")], [0.5]])

    td = _safe_step(wrapper)

    assert torch.equal(policy.dagger_rng.get_state(), before)
    assert td[DAGGER_STUDENT_ACTION_VALID_KEY].tolist() == [False, True]
    assert td[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [False, True]
    assert td[ACTION_KEY].isfinite().all()
    assert td[ACTION_KEY][0].item() == pytest.approx(0.0)


def test_safe_dagger_never_releases_to_a_persistently_invalid_student():
    policy, wrapper = _safe_rollout_fixture(envs=1, hold=1)
    policy.test_student_action = torch.tensor([[float("nan")]])

    first = _safe_step(wrapper)
    second = _safe_step(wrapper)
    third = _safe_step(wrapper)

    for td in (first, second, third):
        assert td[DAGGER_IS_STUDENT_ACTION_KEY].tolist() == [False]
        assert td[DAGGER_SAFE_RELEASE_KEY].tolist() == [False]
        assert td[ACTION_KEY].isfinite().all()


def test_safe_dagger_invalid_teacher_breaks_hold_and_uses_student():
    policy, wrapper = _safe_rollout_fixture(envs=1, hold=8)
    policy.test_student_action.fill_(3.0)
    _safe_step(wrapper)
    policy.test_student_action.fill_(0.5)
    policy.test_teacher_action.fill_(float("nan"))

    td = _safe_step(wrapper)

    assert td[DAGGER_TEACHER_ACTION_VALID_KEY].item() is False
    assert td[DAGGER_IS_STUDENT_ACTION_KEY].item() is True
    assert td[ACTION_KEY].item() == pytest.approx(0.5)
    assert td[DAGGER_TEACHER_ACTION_KEY].isfinite().all()
    assert wrapper._safe_teacher_active.item() is False
    assert wrapper._safe_teacher_hold.item() == 0


def test_safe_dagger_zero_iteration_is_exact_and_clears_the_latch():
    policy, wrapper = _safe_rollout_fixture(envs=1, hold=8)
    policy.cfg.dagger_safe_zero_iteration = 2
    policy.dagger_rollout_count = 1
    policy.test_student_action.fill_(3.0)

    before_cutoff = _safe_step(wrapper)

    assert before_cutoff[DAGGER_SAFE_TEACHER_KEY].item() is True
    assert before_cutoff[DAGGER_IS_STUDENT_ACTION_KEY].item() is False
    assert wrapper._safe_teacher_active.item() is True

    policy.dagger_rollout_count = 2
    at_cutoff = _safe_step(wrapper)

    assert at_cutoff[DAGGER_SAFE_UNSAFE_KEY].item() is True
    assert at_cutoff[DAGGER_SAFE_TEACHER_KEY].item() is False
    assert at_cutoff[DAGGER_SAFE_TAKEOVER_KEY].item() is False
    assert at_cutoff[DAGGER_SAFE_RELEASE_KEY].item() is False
    assert at_cutoff[DAGGER_IS_STUDENT_ACTION_KEY].item() is True
    assert at_cutoff[ACTION_KEY].item() == pytest.approx(3.0)
    assert at_cutoff[DAGGER_TEACHER_ACTION_VALID_KEY].item() is True
    assert wrapper._safe_teacher_active.item() is False
    assert wrapper._safe_teacher_hold.item() == 0

    # The explicit cutoff also overrides the usual invalid-student takeover.
    # The existing action sanitization still prevents NaN from reaching the env.
    policy.test_student_action.fill_(float("nan"))
    invalid_student = _safe_step(wrapper)
    assert invalid_student[DAGGER_SAFE_UNSAFE_KEY].item() is True
    assert invalid_student[DAGGER_SAFE_TEACHER_KEY].item() is False
    assert invalid_student[DAGGER_IS_STUDENT_ACTION_KEY].item() is True
    assert invalid_student[ACTION_KEY].isfinite().all()


def test_hybrid_safe_cutoff_leaves_beta_teacher_selection_independent():
    policy, wrapper = _safe_rollout_fixture(envs=1, hold=1)
    policy.cfg.dagger_control_mode = "hybrid"
    policy.cfg.dagger_safe_zero_iteration = 2
    policy.dagger_rollout_count = 2
    policy.test_student_action.fill_(3.0)

    td = _safe_step(wrapper)

    assert td[DAGGER_SAFE_TEACHER_KEY].item() is False
    assert td[DAGGER_BETA_TEACHER_KEY].item() is True
    assert td[DAGGER_IS_STUDENT_ACTION_KEY].item() is False


def test_beta_schedule_uses_stage_local_rollouts_and_has_exact_endpoints():
    policy = _bare_policy(
        dagger_beta_start=1.0,
        dagger_beta_end=0.1,
        dagger_beta_decay_rollouts=10,
    )
    # The inherited environment progress can be 6102 when bootstrapping from
    # the supplied PPO teacher. It must not advance the new DAgger schedule.
    policy.env = SimpleNamespace(current_iter=6102)

    policy.dagger_rollout_count = 0
    assert policy._teacher_mixture_probability() == pytest.approx(1.0)

    policy.dagger_rollout_count = 10
    assert policy._teacher_mixture_probability() == pytest.approx(0.1)

    policy.dagger_rollout_count = 10_000
    assert policy._teacher_mixture_probability() == pytest.approx(0.1)


def test_iql_expectile_loss_weights_positive_and_negative_advantages():
    difference = torch.tensor([2.0, -2.0, 0.0])

    actual = _iql_expectile_loss(difference, expectile=0.8)

    # IQL defines the residual as target_Q - V.  A high expectile must weigh
    # under-estimated V (positive residual) more heavily than over-estimated V.
    assert torch.allclose(actual, torch.tensor([3.2, 0.8, 0.0]))
    assert actual[0] == pytest.approx(4.0 * actual[1])


@pytest.mark.parametrize("expectile", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_iql_expectile_loss_rejects_invalid_expectiles(expectile):
    with pytest.raises(ValueError, match="expectile"):
        _iql_expectile_loss(torch.tensor([1.0]), expectile)


def test_iql_expectile_loss_rejects_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        _iql_expectile_loss(torch.tensor([float("inf")]), 0.7)


def test_scalar_iql_target_projects_to_two_c51_atoms_with_clipping():
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    scalar = torch.tensor([-3.0, -1.5, 0.0, 0.25, 2.0, 3.0])

    projection = _project_scalar_to_c51(scalar, support)

    assert projection.shape == (scalar.numel(), support.numel())
    assert torch.allclose(projection.sum(-1), torch.ones_like(scalar))
    assert (projection > 0.0).sum(-1).le(2).all()
    expected_value = scalar.clamp(support[0], support[-1])
    assert torch.allclose(projection @ support, expected_value)
    assert torch.equal(projection[1], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(projection[2], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(projection[3], torch.tensor([0.0, 0.0, 0.75, 0.25, 0.0]))
    assert torch.equal(projection[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.equal(projection[-1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    ("target", "support", "message"),
    (
        (torch.zeros(1, 1), torch.tensor([-1.0, 0.0, 1.0]), "shape"),
        (torch.zeros(1), torch.tensor([0.0]), "atom"),
        (torch.zeros(1), torch.tensor([-1.0, 0.0, 2.0]), "uniform"),
        (torch.tensor([float("nan")]), torch.tensor([-1.0, 0.0]), "finite"),
    ),
)
def test_scalar_iql_target_projection_validates_inputs(target, support, message):
    with pytest.raises(ValueError, match=message):
        _project_scalar_to_c51(target, support)


class _NoOpTensorDictModule(nn.Module):
    def forward(self, td):
        return td


class _ResidualTeacher(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = nn.Parameter(torch.as_tensor(residual).float())

    def get_dist(self, td):
        mean = self.residual.expand(*td.batch_size, -1)
        return SimpleNamespace(mean=mean)


def test_teacher_oracle_restores_absolute_ppo_action_and_is_detached():
    policy = _bare_policy()
    policy.object_transform = _NoOpTensorDictModule()
    policy.encoder_priv = _NoOpTensorDictModule()
    policy.actor = _ResidualTeacher([0.2, -0.3])
    td = TensorDict(
        {REF_JPOS_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        batch_size=[2],
    )

    teacher_action = policy._teacher_action(td)

    assert torch.allclose(
        teacher_action,
        torch.tensor([[1.2, 1.7], [3.2, 3.7]]),
    )
    assert teacher_action.requires_grad is False
    assert policy.actor.residual.grad is None


def test_rollout_uses_exact_source_choice_clips_actions_and_falls_back():
    policy = _bare_policy(
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=4000,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
    )
    policy.dagger_rollout_count = 0
    policy.dagger_rng = torch.Generator().manual_seed(0)
    policy._student_action = lambda td: torch.tensor(
        [[30.0, float("nan")], [-3.0, 4.0]]
    )
    # Row zero is valid and must be executed exactly. Row one is an outlier,
    # so beta=1 still falls back to the bounded student action.
    policy._teacher_action = lambda td: torch.tensor(
        [[1.5, -2.5], [20.01, 0.0]]
    )
    td = TensorDict({}, batch_size=[2])

    _DaggerRolloutPolicy(policy)(td)

    assert torch.equal(
        td[ACTION_KEY], torch.tensor([[1.5, -2.5], [-3.0, 4.0]])
    )
    assert torch.equal(
        td[DAGGER_TEACHER_ACTION_VALID_KEY], torch.tensor([True, False])
    )
    assert torch.equal(
        td[DAGGER_IS_STUDENT_ACTION_KEY], torch.tensor([False, True])
    )
    # The invalid label is retained only as a clipped, masked diagnostic.
    assert torch.equal(
        td[DAGGER_TEACHER_ACTION_KEY],
        torch.tensor([[1.5, -2.5], [20.0, 0.0]]),
    )


def test_bc_replay_sampler_draws_only_valid_teacher_labels():
    replay = _DeviceReplay(capacity=8, device="cpu")
    replay.extend(
        {
            "row": torch.arange(6),
            DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor(
                [False, True, False, True, False, False]
            ),
        }
    )

    sampled = replay.sample(
        128,
        "cpu",
        generator=torch.Generator().manual_seed(3),
        valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
    )

    assert replay.valid_count(DAGGER_TEACHER_ACTION_VALID_KEY) == 2
    assert sampled[DAGGER_TEACHER_ACTION_VALID_KEY].all()
    assert set(sampled["row"].tolist()) == {1, 3}


def test_replay_projected_sample_preserves_rows_rng_and_invalidates_cache():
    replay = _DeviceReplay(capacity=5, device="cpu")
    replay.extend(
        {
            "row": torch.arange(4),
            "unused": torch.arange(40, 44),
            DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor(
                [False, True, False, True]
            ),
        }
    )
    # Populate the cache, then wrap the ring and change the valid population.
    assert replay.valid_count(DAGGER_TEACHER_ACTION_VALID_KEY) == 2
    replay.extend(
        {
            "row": torch.tensor([4, 5]),
            "unused": torch.tensor([44, 45]),
            DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, False]),
        }
    )
    assert replay.valid_count(DAGGER_TEACHER_ACTION_VALID_KEY) == 3

    full_rng = torch.Generator().manual_seed(17)
    projected_rng = torch.Generator().manual_seed(17)
    full = replay.sample(
        128,
        "cpu",
        generator=full_rng,
        valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
    )
    projected = replay.sample(
        128,
        "cpu",
        generator=projected_rng,
        valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
        fields=("row", DAGGER_TEACHER_ACTION_VALID_KEY),
    )

    assert set(projected) == {"row", DAGGER_TEACHER_ACTION_VALID_KEY}
    assert torch.equal(projected["row"], full["row"])
    assert torch.equal(
        projected[DAGGER_TEACHER_ACTION_VALID_KEY],
        full[DAGGER_TEACHER_ACTION_VALID_KEY],
    )
    assert torch.equal(full_rng.get_state(), projected_rng.get_state())
    assert set(projected["row"].tolist()) == {1, 3, 4}


def _q_source_rows(row_ids, *, student):
    """Build tiny replay rows whose action and BC-label identities differ."""
    row_ids = torch.as_tensor(row_ids, dtype=torch.float32)
    count = row_ids.numel()
    action_offset = 200.0 if student else 100.0
    return {
        "row": row_ids.clone(),
        "observations": row_ids.unsqueeze(-1),
        "actions": (row_ids + action_offset).unsqueeze(-1),
        # This must never replace the action actually sent to the environment.
        DAGGER_REPLAY_TEACHER_ACTIONS: (row_ids + 1_000.0).unsqueeze(-1),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.ones(count, dtype=torch.bool),
        DAGGER_IS_STUDENT_ACTION_KEY: torch.full(
            (count,), bool(student), dtype=torch.bool
        ),
    }


def _q_teacher_rows(row_ids):
    rows = _q_source_rows(row_ids, student=False)
    # The persistent teacher partition intentionally shares the exported H5
    # schema and therefore carries no DAgger source/BC-label-only fields.
    return {key: rows[key] for key in ("row", "actions")}


@pytest.mark.parametrize("dagger_rollout_count", [0, 4_000, 40_000])
def test_critic_q_batch_is_fixed_half_teacher_half_student_independent_of_beta(
    dagger_rollout_count,
):
    policy = _bare_policy(
        q_batch_size=128,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=4_000,
    )
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(23)
    policy.dagger_rollout_count = dagger_rollout_count
    # The all-transition ring is deliberately very imbalanced. Q sampling must
    # not inherit either this 7:1 imbalance or the rollout beta that created it.
    policy.dagger_replay = _DeviceReplay(capacity=32, device="cpu")
    policy.dagger_replay.extend(_q_source_rows(range(2), student=False))
    policy.dagger_replay.extend(_q_source_rows(range(2, 16), student=True))
    # Teacher rows have their own persistent source so beta=0 cannot make the
    # critic silently become student-only after the all-transition FIFO turns.
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay.extend(_q_teacher_rows(range(20, 24)))

    batch = policy._sample_balanced_q_batch()

    assert batch is not None
    teacher_source = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
    assert teacher_source.dtype is torch.bool
    assert int(teacher_source.sum()) == 64
    assert int((~teacher_source).sum()) == 64
    # Both partitions retain the deterministic action actually executed. In
    # particular, student rows must not be rewritten to their teacher BC label.
    expected_offset = torch.where(teacher_source, 100.0, 200.0)
    assert torch.equal(
        batch["actions"].squeeze(-1), batch["row"] + expected_offset
    )


@pytest.mark.parametrize("missing_source", ["teacher", "student"])
def test_critic_q_batch_waits_until_both_execution_sources_exist(missing_source):
    policy = _bare_policy(q_batch_size=16)
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(11)
    policy.dagger_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    if missing_source != "student":
        policy.dagger_replay.extend(_q_source_rows(range(4), student=True))
    else:
        # Teacher rows in the all-transition ring are not a substitute for the
        # required student half of the critic batch.
        policy.dagger_replay.extend(_q_source_rows(range(4), student=False))
    if missing_source != "teacher":
        policy.q_teacher_replay.extend(
            _q_teacher_rows(range(10, 14))
        )

    with pytest.raises(RuntimeError, match=missing_source):
        policy._sample_balanced_q_batch()


def test_resume_refills_persistent_teacher_critic_partition_from_h5(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    row_count = 6
    fields = {
        "observations": torch.arange(row_count * 2, dtype=torch.float32).view(
            row_count, 2
        ),
        "critic_observations": torch.arange(
            row_count * 3, dtype=torch.float32
        ).view(row_count, 3),
        "actions": torch.arange(row_count, dtype=torch.float32).view(
            row_count, 1
        ),
        "rewards": torch.arange(row_count, dtype=torch.float32),
        "dones": torch.zeros(row_count, dtype=torch.bool),
        "truncations": torch.zeros(row_count, dtype=torch.bool),
        "discounts": torch.ones(row_count),
        "next_observations": torch.arange(
            row_count * 2, dtype=torch.float32
        ).view(row_count, 2) + 100.0,
        "next_critic_observations": torch.arange(
            row_count * 3, dtype=torch.float32
        ).view(row_count, 3) + 100.0,
    }
    fingerprint = "sha256:" + "b" * 64
    with h5py.File(path, "w") as replay:
        replay.attrs.update({
            "format": bc_dagger_module.DAGGER_TEACHER_REPLAY_FORMAT,
            "format_version": (
                bc_dagger_module.DAGGER_TEACHER_REPLAY_FORMAT_VERSION
            ),
            "teacher_only": True,
            "action_parameterization": (
                bc_dagger_module.DAGGER_ACTION_PARAMETERIZATION
            ),
            "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
            "replay_observation_semantics": (
                bc_dagger_module.DAGGER_REPLAY_OBSERVATION_SEMANTICS
            ),
            "reward_scalarization": bc_dagger_module.SAC_REWARD_SCALARIZATION,
            "truncation_next_observation": (
                bc_dagger_module.TRUNCATION_NEXT_OBSERVATION_SEMANTICS
            ),
            "vecnorm_fingerprint": fingerprint,
            "action_clip": 20.0,
            "num_transitions": row_count,
        })
        for key in TEACHER_REPLAY_FIELDS:
            replay.create_dataset(key, data=fields[key].numpy())

    policy = _bare_policy(
        dagger_action_clip=20.0,
        teacher_buffer_snapshot_chunk_rows=2,
    )
    policy._q_actor_dim = 2
    policy._q_critic_dim = 3
    policy.action_dim = 1
    policy._replay_vecnorm_fingerprint = fingerprint
    policy.q_teacher_replay = _DeviceReplay(capacity=3, device="cpu")

    restored = policy.restore_q_teacher_replay(path)

    assert restored == 3
    assert policy.q_teacher_replay.size == 3
    assert policy.q_teacher_replay.seen == 3
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:3],
        fields["actions"][-3:],
    )


@pytest.mark.parametrize("dagger_rollout_count", [0, 4_000])
def test_train_op_sends_exact_balanced_executed_actions_to_critic(
    dagger_rollout_count,
):
    policy = _bare_policy(
        train_every=32,
        dagger_updates_per_rollout=1,
        dagger_bc_epochs=1,
        dagger_batch_size=4,
        q_batch_size=32,
        q_updates_per_rollout=1,
        q_learning_starts_per_source=1,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=4_000,
    )
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(31)
    policy.dagger_rollout_count = dagger_rollout_count
    policy.dagger_replay = _DeviceReplay(capacity=32, device="cpu")
    policy.dagger_replay.extend(_q_source_rows(range(2), student=False))
    policy.dagger_replay.extend(_q_source_rows(range(2, 16), student=True))
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay.extend(_q_teacher_rows(range(20, 24)))
    policy.teacher_replay = None
    policy._dagger_transition_chunks = lambda rollout: iter(())
    policy._prepare_dagger_learning_batch = lambda batch: batch
    zero = torch.tensor(0.0)
    policy._bc_update = lambda batch: (zero, zero, zero)

    class CapturedCriticBatch(Exception):
        pass

    captured = []

    def capture_q_update(batch):
        captured.append(batch)
        raise CapturedCriticBatch

    policy._q_update = capture_q_update

    with pytest.raises(CapturedCriticBatch):
        policy.train_op(TensorDict({}, batch_size=[]))

    assert len(captured) == 1
    batch = captured[0]
    teacher_source = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
    assert int(teacher_source.sum()) == 16
    assert int((~teacher_source).sum()) == 16
    expected_offset = torch.where(teacher_source, 100.0, 200.0)
    assert torch.equal(
        batch["actions"].squeeze(-1), batch["row"] + expected_offset
    )


def _cat_fields(td, keys):
    return torch.cat([td[key] for key in keys], dim=-1)


def test_replay_keeps_executed_action_separate_from_teacher_bc_label():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._cat_replay_sources = _cat_fields
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[99.0]]),
        "next_critic_observations": torch.tensor([[199.0]]),
    }
    policy._truncation_final_batches = []

    rollout = TensorDict(
        {
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[101.0], [102.0]]]),
            ACTION_KEY: torch.tensor([[[3.0], [4.0]]]),
            TEACHER_ACTION_KEY: torch.tensor([[[30.0], [40.0]]]),
            TEACHER_ACTION_VALID_KEY: torch.tensor([[True, True]]),
            IS_STUDENT_ACTION_KEY: torch.tensor([[False, True]]),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.tensor([[[0.25, 0.75], [1.0, 2.0]]]),
                    "done": torch.zeros(1, 2, 1, dtype=torch.bool),
                    "terminated": torch.zeros(1, 2, 1, dtype=torch.bool),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(
                                1, 2, 1, dtype=torch.bool
                            ),
                            "command_finished": torch.zeros(
                                1, 2, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))
    transitions = {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in chunks[0]
    }

    assert torch.equal(transitions["actions"], torch.tensor([[3.0], [4.0]]))
    assert torch.equal(
        transitions["teacher_actions"], torch.tensor([[30.0], [40.0]])
    )
    assert torch.equal(
        transitions["teacher_action_valid"], torch.tensor([True, True])
    )
    assert torch.equal(
        transitions["is_student_action"], torch.tensor([False, True])
    )
    assert torch.equal(transitions["rewards"], torch.tensor([1.0, 3.0]))
    # This is a Bernoulli source choice, not numeric teacher/student blending:
    # every recorded source row is exactly teacher or exactly student.
    assert transitions["is_student_action"].dtype is torch.bool


def test_timeout_uses_true_final_state_while_command_completion_is_terminal():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._cat_replay_sources = _cat_fields
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[99.0]]),
        "next_critic_observations": torch.tensor([[199.0]]),
    }
    policy._truncation_final_batches = [
        {
            "indices": torch.tensor([0]),
            "next_observations": torch.tensor([[77.0]]),
            "next_critic_observations": torch.tensor([[177.0]]),
        }
    ]
    done = torch.ones(1, 2, 1, dtype=torch.bool)
    rollout = TensorDict(
        {
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[101.0], [102.0]]]),
            ACTION_KEY: torch.tensor([[[3.0], [4.0]]]),
            TEACHER_ACTION_KEY: torch.tensor([[[30.0], [40.0]]]),
            TEACHER_ACTION_VALID_KEY: torch.ones(1, 2, dtype=torch.bool),
            IS_STUDENT_ACTION_KEY: torch.zeros(1, 2, dtype=torch.bool),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.ones(1, 2, 1),
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [[[True], [False]]]
                            ),
                            "command_finished": torch.tensor(
                                [[[False], [True]]]
                            ),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))

    assert torch.equal(chunks[0]["next_observations"], torch.tensor([[77.0]]))
    assert torch.equal(chunks[1]["next_observations"], torch.tensor([[99.0]]))
    assert chunks[0]["truncations"].item() is True
    assert chunks[1]["truncations"].item() is False
    assert policy._last_truncation_finals_used == 1


def test_dagger_transition_ring_uses_raw_current_next_and_timeout_final():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._replay_vecnorm_keys = {"actor_obs", "critic_obs"}
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[999.0]]),
        "next_critic_observations": torch.tensor([[1999.0]]),
    }
    policy._truncation_final_batches = [
        {
            "indices": torch.tensor([0]),
            "next_observations": torch.tensor([[777.0]]),
            "next_critic_observations": torch.tensor([[1777.0]]),
        }
    ]
    done = torch.tensor([[[True], [False]]])
    rollout = TensorDict(
        {
            # Main fields are post-VecNorm and deliberately very different.
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[11.0], [12.0]]]),
            "_fastsac_raw": TensorDict(
                {
                    "actor_obs": torch.tensor([[[101.0], [102.0]]]),
                    "critic_obs": torch.tensor([[[111.0], [112.0]]]),
                },
                batch_size=[1, 2],
            ),
            ACTION_KEY: torch.zeros(1, 2, 1),
            TEACHER_ACTION_KEY: torch.zeros(1, 2, 1),
            TEACHER_ACTION_VALID_KEY: torch.ones(1, 2, dtype=torch.bool),
            IS_STUDENT_ACTION_KEY: torch.zeros(1, 2, dtype=torch.bool),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.zeros(1, 2, 1),
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": done,
                            "command_finished": torch.zeros_like(done),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))

    assert torch.equal(chunks[0]["observations"], torch.tensor([[101.0]]))
    assert torch.equal(chunks[1]["observations"], torch.tensor([[102.0]]))
    # Timeout uses the captured pre-reset final state, not step 1/reset state.
    assert torch.equal(chunks[0]["next_observations"], torch.tensor([[777.0]]))
    # Last non-timeout row uses the raw rollout carry captured after the loop.
    assert torch.equal(chunks[1]["next_observations"], torch.tensor([[999.0]]))


def test_dagger_sample_time_normalization_is_once_and_non_mutating():
    policy = _bare_policy()
    policy.q_actor_keys = ["normalized", "latent"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_widths = [2, 1]
    policy._q_critic_widths = [2]
    policy._replay_vecnorm_keys = {"normalized", "critic"}
    object.__setattr__(
        policy,
        "_replay_vecnorm",
        SimpleNamespace(
            eps=1e-4,
            loc={
                "normalized": torch.tensor([10.0, 20.0]),
                "critic": torch.tensor([100.0, 200.0]),
            },
            scale={
                "normalized": torch.tensor([2.0, 4.0]),
                "critic": torch.tensor([10.0, 20.0]),
            },
        ),
    )
    batch = {
        "observations": torch.tensor([[12.0, 24.0, 7.0]]),
        "next_observations": torch.tensor([[8.0, 12.0, 9.0]]),
        "critic_observations": torch.tensor([[110.0, 220.0]]),
        "next_critic_observations": torch.tensor([[90.0, 160.0]]),
        "actions": torch.ones(1, 1),
    }
    original = {key: value.clone() for key, value in batch.items()}

    prepared = policy._prepare_dagger_learning_batch(batch)

    assert torch.equal(
        prepared["observations"], torch.tensor([[1.0, 1.0, 7.0]])
    )
    assert torch.equal(
        prepared["next_observations"], torch.tensor([[-1.0, -2.0, 9.0]])
    )
    assert torch.equal(
        prepared["critic_observations"], torch.tensor([[1.0, 1.0]])
    )
    assert torch.equal(
        prepared["next_critic_observations"], torch.tensor([[-1.0, -2.0]])
    )
    assert torch.equal(prepared["actions"], batch["actions"])
    for key, value in original.items():
        assert torch.equal(batch[key], value)

    projected = policy._prepare_dagger_learning_batch(
        {
            "observations": batch["observations"],
            "actions": batch["actions"],
        }
    )
    assert set(projected) == {"observations", "actions"}
    assert torch.equal(
        projected["observations"], torch.tensor([[1.0, 1.0, 7.0]])
    )


class _NoTrainingReplay:
    def __init__(self):
        self.size = 0
        self.seen = 0
        self.rows = None

    def extend(self, rows):
        self.rows = rows
        count = rows["rewards"].shape[0]
        self.seen += count
        return count


class _TeacherExportRecorder:
    device = torch.device("cpu")

    def __init__(self):
        self.rows = None

    def append(self, rows):
        self.rows = rows
        return rows["rewards"].shape[0]


def test_train_op_never_calls_ppo_and_exports_only_teacher_executed_rows():
    policy = _bare_policy(**vars(PPOBCDaggerFinetuneConfig()))
    policy.cfg.train_every = 2
    policy.cfg.dagger_updates_per_rollout = 1
    policy.cfg.dagger_bc_epochs = 1
    policy.cfg.dagger_batch_size = 1
    policy.cfg.q_updates_per_rollout = 1
    policy.cfg.dagger_control_mode = "beta"
    transitions = {
        "observations": torch.zeros(3, 1),
        "critic_observations": torch.zeros(3, 1),
        "actions": torch.tensor([[10.0], [20.0], [30.0]]),
        "rewards": torch.zeros(3),
        "dones": torch.zeros(3, dtype=torch.bool),
        "truncations": torch.zeros(3, dtype=torch.bool),
        "discounts": torch.ones(3),
        "next_observations": torch.zeros(3, 1),
        "next_critic_observations": torch.zeros(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[10.0], [21.0], [31.0]]
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([False, True, False]),
    }
    policy._dagger_transition_chunks = lambda td: iter(
        [
            {key: value[:2] for key, value in transitions.items()},
            {key: value[2:] for key, value in transitions.items()},
        ]
    )
    policy._teacher_mixture_probability = lambda: 0.75
    policy.dagger_replay = _NoTrainingReplay()
    policy.q_teacher_replay = _NoTrainingReplay()
    policy.teacher_replay = _TeacherExportRecorder()
    policy.train_adapt = lambda td: {"adapt/called": 1.0}
    policy.train_policy = lambda td: pytest.fail("PPO train_policy was called")
    policy._compute_advantage = lambda *args, **kwargs: pytest.fail(
        "GAE was called"
    )
    policy._update_ppo = lambda td: pytest.fail("PPO update was called")
    policy.num_updates = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.bc_update_count = 0
    policy.q_update_count = 0
    policy._last_truncation_finals_used = 0

    info = policy.train_op(TensorDict({}, batch_size=[1, 2]))

    assert info["adapt/called"] == pytest.approx(1.0)
    assert info["dagger/beta"] == pytest.approx(0.75)
    assert info["dagger/rollout_count"] == 1
    assert info["dagger/rollout_index"] == 0
    assert info["dagger/beta_zero_iteration"] == 4000
    assert info["dagger/safe_zero_iteration"] == -1
    assert info["dagger/safe_control_enabled"] == pytest.approx(0.0)
    assert info["dagger/teacher_exported"] == 1
    assert torch.equal(
        policy.teacher_replay.rows["actions"], torch.tensor([[10.0]])
    )
    assert set(policy.teacher_replay.rows) == set(TEACHER_REPLAY_FIELDS)
    # The persistent critic source must contain the same teacher-executed row
    # as the exported H5, while the ordinary DAgger ring keeps all three rows.
    assert policy.q_teacher_replay.seen == 1
    assert torch.equal(
        policy.q_teacher_replay.rows["actions"], torch.tensor([[10.0]])
    )
    assert policy.dagger_replay.seen == 3
    assert torch.equal(
        policy.dagger_replay.rows["actions"],
        torch.tensor([[10.0], [20.0], [30.0]]),
    )


def test_teacher_h5_roundtrip_has_truthful_dagger_manifest(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    fingerprint = "sha256:" + "a" * 64
    replay = _DaggerTeacherReplayBuffer(
        path,
        capacity=4,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="paired-replay",
        actor_obs_keys=["a", "b"],
        critic_obs_keys=["c"],
        vecnorm_fingerprint=fingerprint,
        action_clip=20.0,
    )
    rows = {
        "observations": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "critic_observations": torch.arange(
            12, dtype=torch.float32
        ).reshape(4, 3),
        "actions": torch.arange(4, dtype=torch.float32).reshape(4, 1),
        "rewards": torch.arange(4, dtype=torch.float32),
        "dones": torch.tensor([False, False, True, False]),
        "truncations": torch.tensor([False, False, True, False]),
        "discounts": torch.ones(4),
        "next_observations": torch.ones(4, 2),
        "next_critic_observations": torch.ones(4, 3),
    }
    replay.append(rows)
    replay.snapshot(iteration=7, checkpoint_name="checkpoint_7")
    metadata = replay.checkpoint_metadata()

    restored = _DaggerTeacherReplayBuffer(
        tmp_path / "new.h5",
        capacity=4,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="paired-replay",
        actor_obs_keys=["a", "b"],
        critic_obs_keys=["c"],
        vecnorm_fingerprint=fingerprint,
        action_clip=20.0,
    )
    restored.restore(path, expected_metadata=metadata)

    assert restored.size == 4
    assert restored.seen == 4
    assert torch.equal(restored.data["actions"], rows["actions"])
    assert metadata["actor_backend"].startswith("vaic_ppo_")
    assert metadata["replay_observation_semantics"] == (
        "raw_pre_vecnorm_sample_current_v1"
    )
    assert metadata["format_version"] == 2
    assert metadata["vecnorm_fingerprint"] == fingerprint
    assert metadata["action_clip"] == pytest.approx(20.0)

    offline = BCDaggerOfflineReplayH5(
        path,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        expected_actor_obs_keys=["a", "b"],
        expected_critic_obs_keys=["c"],
        expected_vecnorm_fingerprint=fingerprint,
        expected_action_clip=20.0,
        expected_replay_metadata=metadata,
    )
    assert offline.size == 4
    assert offline.observations_pre_normalized is False
    assert torch.equal(offline.data["actions"], rows["actions"])
    wrong_snapshot = copy.deepcopy(metadata)
    wrong_snapshot["snapshot_id"] = "stale-compatible-snapshot"
    with pytest.raises(ValueError, match="completed staged checkpoint snapshot"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=2,
            critic_dim=3,
            action_dim=1,
            expected_actor_obs_keys=["a", "b"],
            expected_critic_obs_keys=["c"],
            expected_vecnorm_fingerprint=fingerprint,
            expected_action_clip=20.0,
            expected_replay_metadata=wrong_snapshot,
        )
    with pytest.raises(ValueError, match="fingerprint"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=2,
            critic_dim=3,
            action_dim=1,
            expected_vecnorm_fingerprint="sha256:" + "f" * 64,
        )

    import h5py

    with h5py.File(path, "r+") as h5:
        h5["actions"][0, 0] = 20.01
    with pytest.raises(ValueError, match="outside.*support"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=2,
            critic_dim=3,
            action_dim=1,
            expected_actor_obs_keys=["a", "b"],
            expected_critic_obs_keys=["c"],
            expected_vecnorm_fingerprint=fingerprint,
            expected_action_clip=20.0,
        )


def test_teacher_h5_reuses_unchanged_snapshot_and_rewrites_after_append(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = _DaggerTeacherReplayBuffer(
        path,
        capacity=2,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="stable-replay",
        actor_obs_keys=["actor"],
        critic_obs_keys=["critic"],
        vecnorm_fingerprint="sha256:" + "c" * 64,
        action_clip=20.0,
    )

    def row(value):
        scalar = torch.tensor([float(value)])
        column = scalar[:, None]
        return {
            "observations": column,
            "critic_observations": column,
            "actions": column,
            "rewards": scalar,
            "dones": torch.zeros(1, dtype=torch.bool),
            "truncations": torch.zeros(1, dtype=torch.bool),
            "discounts": torch.ones(1),
            "next_observations": column + 1.0,
            "next_critic_observations": column + 1.0,
        }

    replay.append(row(1.0))
    assert replay.snapshot(1, "checkpoint_1") == str(path)
    first_metadata = replay.checkpoint_metadata()
    first_stat = path.stat()

    assert replay.snapshot(2, "checkpoint_2") == str(path)
    second_metadata = replay.checkpoint_metadata()
    second_stat = path.stat()
    assert second_metadata == first_metadata
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns

    replay.append(row(2.0))
    assert replay.snapshot(3, "checkpoint_3") == str(path)
    third_metadata = replay.checkpoint_metadata()
    assert third_metadata["snapshot_id"] != first_metadata["snapshot_id"]
    assert third_metadata["snapshot_iteration"] == 3
    assert third_metadata["checkpoint_name"] == "checkpoint_3"
    assert third_metadata["size"] == 2


def test_teacher_h5_rejects_action_outside_manifest_support(tmp_path):
    replay = _DaggerTeacherReplayBuffer(
        tmp_path / "teacher_replay_buffer.h5",
        capacity=1,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        seed=0,
        device="cpu",
        actor_obs_keys=["actor"],
        critic_obs_keys=["critic"],
        vecnorm_fingerprint="sha256:" + "c" * 64,
        action_clip=20.0,
    )
    rows = {
        "observations": torch.zeros(1, 1),
        "critic_observations": torch.zeros(1, 1),
        "actions": torch.tensor([[20.01]]),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1, dtype=torch.bool),
        "truncations": torch.zeros(1, dtype=torch.bool),
        "discounts": torch.ones(1),
        "next_observations": torch.zeros(1, 1),
        "next_critic_observations": torch.zeros(1, 1),
    }

    with pytest.raises(ValueError, match="action"):
        replay.append(rows)

    assert replay.size == 0
    assert replay.seen == 0
    assert replay.data == {}


def test_stage2_accepts_legacy_normalized_dagger_h5_but_rejects_mixed_schema(
    tmp_path,
):
    import h5py

    path = tmp_path / "teacher_replay_buffer.h5"
    fingerprint = "sha256:" + "b" * 64
    replay = _DaggerTeacherReplayBuffer(
        path,
        capacity=2,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        seed=0,
        device="cpu",
        actor_obs_keys=["actor"],
        critic_obs_keys=["critic"],
        vecnorm_fingerprint=fingerprint,
        action_clip=20.0,
    )
    replay.append(
        {
            "observations": torch.tensor([[1.0], [2.0]]),
            "critic_observations": torch.tensor([[3.0], [4.0]]),
            "actions": torch.zeros(2, 1),
            "rewards": torch.zeros(2),
            "dones": torch.zeros(2, dtype=torch.bool),
            "truncations": torch.zeros(2, dtype=torch.bool),
            "discounts": torch.ones(2),
            "next_observations": torch.tensor([[5.0], [6.0]]),
            "next_critic_observations": torch.tensor([[7.0], [8.0]]),
        }
    )
    replay.snapshot(1, "checkpoint_1")
    with h5py.File(path, "r+") as h5:
        h5.attrs["format_version"] = 1
        h5.attrs["replay_observation_semantics"] = (
            "normalized_frozen_vecnorm_v1"
        )
        del h5.attrs["vecnorm_fingerprint"]

    legacy = BCDaggerOfflineReplayH5(
        path,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        expected_actor_obs_keys=["actor"],
        expected_critic_obs_keys=["critic"],
    )
    assert legacy.observations_pre_normalized is True

    with h5py.File(path, "r+") as h5:
        h5.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="Unsupported.*schema"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=1,
            critic_dim=1,
            action_dim=1,
        )


class _TinyTwinC51(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))
        self.last_actions = None
        self.last_projection = None

    def forward(self, obs, actions):
        self.last_actions = actions.detach().clone()
        return self.logits[:, None, :].expand(2, obs.shape[0], 3)

    def values(self, logits):
        return (logits.softmax(-1) * self.support).sum(-1)

    @torch.no_grad()
    def projection(self, obs, actions, reward, bootstrap, discount):
        self.last_projection = {
            "obs": obs.detach().clone(),
            "actions": actions.detach().clone(),
            "reward": reward.detach().clone(),
            "bootstrap": bootstrap.detach().clone(),
            "discount": discount.detach().clone(),
        }
        # Deliberately different head targets. Collapsing to a clipped/minimum
        # target would make the online twins receive the same gradient.
        batch = obs.shape[0]
        target = torch.zeros(2, batch, 3, device=obs.device)
        target[0, :, 0] = 1.0
        target[1, :, 2] = 1.0
        return target


class _StudentMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def get_dist_from_flat(self, obs):
        return SimpleNamespace(mean=self.linear(obs))


def test_sac_critic_update_matches_q_only_target_and_timeout_bootstrap():
    policy = _bare_policy(
        gamma=0.5,
        q_tau=0.5,
        q_max_grad_norm=0.0,
        dagger_action_clip=20.0,
        sac_q_action_input_gain=1.0,
        sac_alpha_init=0.1,
        sac_clipped_double_q=True,
    )
    policy.device = torch.device("cpu")
    policy.action_dim = 1
    policy.qnet = _TinyTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.2)
    policy.q_update_count = 0
    policy.sac_action_rng = torch.Generator().manual_seed(5)

    next_action = torch.tensor([[20.0], [-20.0], [10.0]])
    next_log_prob = torch.tensor([-1.0, -2.0, -3.0])

    class _FixedTargetDist:
        def rsample_with_log_prob(self, generator=None):
            assert generator is policy.sac_action_rng
            return next_action, next_log_prob

    policy._sac_critic_dist_from_flat = lambda obs: (
        torch.zeros_like(next_action),
        _FixedTargetDist(),
    )
    policy._normalized_action_log_prob = lambda value: value

    batch = {
        "observations": torch.zeros(3, 2),
        "critic_observations": torch.tensor(
            [[-0.3, 1.0], [-0.3, 2.0], [-0.3, 3.0]]
        ),
        "actions": torch.tensor([[20.0], [-10.0], [5.0]]),
        "rewards": torch.tensor([0.1, 0.2, 0.3]),
        # row 0: ordinary; row 1: true terminal; row 2: time-limit timeout.
        "dones": torch.tensor([False, True, True]),
        "truncations": torch.tensor([False, False, True]),
        "discounts": torch.tensor([1.0, 0.8, 0.5]),
        "next_observations": torch.ones(3, 2),
        "next_critic_observations": torch.tensor(
            [[0.4, 4.0], [0.6, 5.0], [0.8, 6.0]]
        ),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([True, False, True]),
    }
    _, _, _, metrics = policy._q_update(batch)

    assert torch.equal(
        policy.qnet.last_actions,
        torch.tensor([[1.0], [-0.5], [0.25]]),
    )
    projection = policy.qnet_target.last_projection
    assert projection is not None
    assert torch.equal(
        projection["actions"], torch.tensor([[1.0], [-1.0], [0.5]])
    )
    # A true terminal suppresses Q bootstrap; a timeout uses the captured
    # final state exactly like an ordinary transition. The stochastic action
    # remains sampled, while Q-only effective alpha is exactly zero.
    assert torch.equal(
        projection["bootstrap"], torch.tensor([1.0, 0.0, 1.0])
    )
    assert torch.allclose(
        projection["discount"], torch.tensor([0.5, 0.4, 0.25])
    )
    assert torch.allclose(
        projection["reward"], torch.tensor([0.1, 0.2, 0.3])
    )
    assert metrics["entropy_tax_abs_mean"] == pytest.approx(0.0)
    assert metrics["target_q_mean"] == pytest.approx(-1.0)
    assert policy.q_update_count == 1


def test_legacy_ppo_bootstrap_accepts_fresh_q_but_same_stage_requires_both_qs(
    monkeypatch,
):
    policy = _bare_policy(
        q_seed=7,
        dagger_seed=11,
        q_tau=0.05,
        use_object_adapt=False,
    )
    policy.qnet = nn.Linear(2, 1, bias=False)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.q_update_count = 123
    policy.dagger_rollout_count = 456
    policy.q_rng = torch.Generator().manual_seed(99)
    policy.sac_action_rng = torch.Generator().manual_seed(98)
    policy.dagger_rng = torch.Generator().manual_seed(100)

    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda self, state_dict, strict=True: [
            "depth_cnn",
            "temporal_depth_gru",
            "temporal_depth_gru_ema",
            "qnet",
            "qnet_target",
        ],
    )

    failed = policy.load_state_dict({"last_phase": "train"})

    assert set(failed).issuperset({"qnet", "qnet_target"})
    assert policy.q_update_count == 0
    assert policy.dagger_rollout_count == 0
    assert torch.equal(
        policy.qnet.weight, policy.qnet_target.weight
    )

    same_stage_without_target = {
        "training_algorithm": EXPECTED_DAGGER_ALGORITHM,
        "last_phase": "finetune",
        "qnet": policy.qnet.state_dict(),
    }
    with pytest.raises((KeyError, ValueError, RuntimeError), match="qnet_target|target"):
        policy.load_state_dict(same_stage_without_target)

    with pytest.raises(ValueError, match="Legacy"):
        policy.load_state_dict(
            {
                "training_algorithm": PPO_BC_DAGGER_LEGACY_TRAINING_ALGORITHM,
                "last_phase": "finetune",
            }
        )


def test_same_stage_resume_restores_learning_state_without_teacher_h5(
    monkeypatch,
):
    policy = _bare_policy(**vars(PPOBCDaggerFinetuneConfig()))
    policy.device = torch.device("cpu")
    policy.cfg.use_object_adapt = False
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 2
    policy._q_critic_dim = 2
    policy.action_dim = 1
    policy.reward_groups = ["task"]
    policy.qnet = nn.Linear(2, 1, bias=False)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.bc_module = nn.Linear(2, 1, bias=False)
    policy.adapt_module_for_test = nn.Linear(2, 1, bias=False)
    policy.bc_optimizer = torch.optim.AdamW(
        policy.bc_module.parameters(), lr=1e-3
    )
    policy.opt_q = torch.optim.AdamW(policy.qnet.parameters(), lr=2e-3)
    policy.opt_adapt = torch.optim.AdamW(
        policy.adapt_module_for_test.parameters(), lr=4e-3
    )
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.sac_action_rng = torch.Generator().manual_seed(3)
    progress = []
    policy.env = SimpleNamespace(set_progress=progress.append)

    source_modules = [
        nn.Linear(2, 1, bias=False),
        nn.Linear(2, 1, bias=False),
        nn.Linear(2, 1, bias=False),
    ]
    source_optimizers = [
        torch.optim.AdamW(source_modules[0].parameters(), lr=1e-3),
        torch.optim.AdamW(source_modules[1].parameters(), lr=2e-3),
        torch.optim.AdamW(source_modules[2].parameters(), lr=4e-3),
    ]
    for index, (module, optimizer) in enumerate(
        zip(source_modules, source_optimizers), start=1
    ):
        optimizer.zero_grad(set_to_none=True)
        (module.weight.square().sum() * index).backward()
        optimizer.step()

    resumed_dagger_rng = torch.Generator().manual_seed(101)
    resumed_q_rng = torch.Generator().manual_seed(202)
    resumed_sac_action_rng = torch.Generator().manual_seed(303)
    torch.rand(7, generator=resumed_dagger_rng)
    torch.rand(9, generator=resumed_q_rng)
    torch.rand(11, generator=resumed_sac_action_rng)
    replay_metadata = {
        "snapshot_id": "frozen-checkpoint-800",
        "size": 1_048_576,
    }
    checkpoint = {
        "training_algorithm": PPO_BC_DAGGER_TRAINING_ALGORITHM,
        "last_phase": "finetune",
        "qnet": policy.qnet.state_dict(),
        "qnet_target": policy.qnet_target.state_dict(),
        "critic_learning_semantics": PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS,
        "actor_learning_semantics": PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "actor_backend": bc_dagger_module.PPO_BC_DAGGER_ACTOR_BACKEND,
        "teacher_action_semantics": (
            bc_dagger_module.DAGGER_TEACHER_ACTION_SEMANTICS
        ),
        "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
        "optimizer_resume_state": {
            "bc_optimizer": source_optimizers[0].state_dict(),
            "q_optimizer": source_optimizers[1].state_dict(),
            "adapt_optimizer": source_optimizers[2].state_dict(),
        },
        "dagger_rollout_count": 801,
        "dagger_environment_steps": 25_632,
        "bc_update_count": 25_632,
        "q_update_count": 25_632,
        "dagger_rng_state": resumed_dagger_rng.get_state(),
        "q_rng_state": resumed_q_rng.get_state(),
        "sac_action_rng_state": resumed_sac_action_rng.get_state(),
        "teacher_replay_id": "original-frozen-replay",
        "teacher_replay_state": replay_metadata,
        "next_iter": 6_801,
    }
    checkpoint["dagger_backend_config"] = policy._checkpoint_config()
    checkpoint["dagger_backend_config"].pop("dagger_safe_zero_iteration")
    checkpoint["q_backend_config"] = policy._q_backend_metadata()

    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda self, state_dict, strict=True: [],
    )

    policy.load_state_dict(checkpoint)

    def assert_optimizer_state_equal(actual, expected):
        actual_state = actual.state_dict()
        assert actual_state["param_groups"] == expected["param_groups"]
        assert actual_state["state"].keys() == expected["state"].keys()
        for parameter_id in expected["state"]:
            for key, expected_value in expected["state"][parameter_id].items():
                actual_value = actual_state["state"][parameter_id][key]
                if torch.is_tensor(expected_value):
                    assert torch.equal(actual_value, expected_value)
                else:
                    assert actual_value == expected_value

    for actual, source in zip(
        (
            policy.bc_optimizer,
            policy.opt_q,
            policy.opt_adapt,
        ),
        source_optimizers,
    ):
        assert_optimizer_state_equal(actual, source.state_dict())
    assert policy.dagger_rollout_count == 801
    assert policy.dagger_environment_steps == 25_632
    assert policy.bc_update_count == 25_632
    assert policy.q_update_count == 25_632
    assert torch.equal(policy.dagger_rng.get_state(), checkpoint["dagger_rng_state"])
    assert torch.equal(policy.q_rng.get_state(), checkpoint["q_rng_state"])
    assert torch.equal(
        policy.sac_action_rng.get_state(), checkpoint["sac_action_rng_state"]
    )
    assert policy.teacher_replay_id == "original-frozen-replay"
    assert policy._loaded_teacher_replay_metadata == replay_metadata
    assert progress == [6_801]


def test_state_dict_names_online_and_target_q_separately():
    # This is deliberately a topology-level assertion. A single aliased module
    # cannot satisfy the user's requirement to save Q1/Q2 and target Q1/Q2.
    policy = _bare_policy()
    policy.qnet = _TinyTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    children = dict(policy.named_children())

    assert children["qnet"] is not children["qnet_target"]
    assert set(children["qnet"].state_dict()) == set(
        children["qnet_target"].state_dict()
    )
    assert children["qnet"].logits.data_ptr() != (
        children["qnet_target"].logits.data_ptr()
    )
    assert "iql_value" not in children


def test_sac_critic_checkpoint_markers_keep_actor_bc_only():
    policy = _bare_policy(**vars(PPOBCDaggerFinetuneConfig()))
    checkpoint_config = policy._checkpoint_config()

    assert EXPECTED_DAGGER_ALGORITHM.endswith("_sac_critic_v3")
    assert "half_teacher_half_student" in (
        PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS
    )
    assert "bc_only" in PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS
    assert checkpoint_config["q_teacher_replay_ratio"] == pytest.approx(0.5)
    assert checkpoint_config["q_batch_size"] == 512
    assert checkpoint_config["q_learning_starts_per_source"] == 8_192


def _finalization_policy(
    *,
    perception=2,
    actor=1,
    recheck=1,
    calibration=3,
    teacher_probability=0.5,
):
    cfg = PPOBCDaggerFinetuneConfig()
    cfg.dagger_finalization_enabled = True
    cfg.dagger_finalize_perception_iterations = perception
    cfg.dagger_finalize_actor_iterations = actor
    cfg.dagger_finalize_recheck_iterations = recheck
    cfg.dagger_finalize_calibration_iterations = calibration
    cfg.dagger_finalize_calibration_control_mode = "beta"
    cfg.dagger_finalize_calibration_teacher_probability = teacher_probability
    policy = _bare_policy()
    policy.cfg = cfg
    policy.finalization_rollout_count = 0
    policy._finalization_last_phase = None
    return policy


def test_finalization_phase_boundaries_skip_zero_length_stages():
    policy = _finalization_policy(
        perception=25,
        actor=0,
        recheck=0,
        calibration=128,
    )

    assert policy._finalization_config() == {
        "semantics": DAGGER_FINALIZATION_SEMANTICS,
        "perception_consolidation_iterations": 25,
        "actor_realignment_iterations": 0,
        "perception_recheck_iterations": 0,
        "replay_q_calibration_iterations": 128,
        "calibration_control_mode": "beta",
        "calibration_teacher_probability": pytest.approx(0.5),
    }
    assert policy._finalization_phase(0) == "perception_consolidation"
    assert policy._finalization_phase(24) == "perception_consolidation"
    # Both zero-length middle stages are skipped at the same boundary.
    assert policy._finalization_phase(25) == "replay_q_calibration"
    assert policy._finalization_phase(152) == "replay_q_calibration"
    assert policy._finalization_phase(153) == "complete"


def test_finalization_phase_gates_isolate_actor_q_and_perception_updates():
    policy = _finalization_policy(
        perception=1,
        actor=1,
        recheck=1,
        calibration=1,
    )
    policy.cfg.dagger_updates_per_rollout = 7
    policy.cfg.q_updates_per_rollout = 11
    adaptation_calls = []
    policy.train_adapt = (
        lambda rollout: adaptation_calls.append(rollout) or {"adapt/called": 1.0}
    )
    rollout = TensorDict({"marker": torch.ones(1, 1)}, batch_size=[1])

    expected = {
        "perception_consolidation": (False, 0, 0, True),
        "actor_realignment": (True, 7, 0, False),
        "perception_recheck": (False, 0, 0, True),
        "replay_q_calibration": (True, 0, 11, False),
    }
    for count, phase in enumerate(DAGGER_FINALIZATION_PHASES):
        policy.finalization_rollout_count = count
        collect, actor_updates, q_updates, adapt = expected[phase]
        assert policy._finalization_phase() == phase
        assert policy._collect_dagger_replay_this_rollout() is collect
        assert policy._actor_updates_this_rollout() == actor_updates
        assert policy._q_updates_this_rollout() == q_updates
        result = policy._adaptation_update_this_rollout(rollout)
        assert bool(result) is adapt

    assert len(adaptation_calls) == 2
    # train_adapt receives a container copy, never the rollout object that will
    # subsequently be reused by the collector.
    assert all(actual is not rollout for actual in adaptation_calls)


def test_finalization_freeze_mask_exposes_only_the_phase_owned_parameters():
    policy = _finalization_policy()
    policy.adapt_module = nn.Linear(1, 1)
    policy.object_adapt = nn.Linear(1, 1)
    policy.temporal_depth_gru = nn.Linear(1, 1)
    policy.actor_adapt = nn.Linear(1, 1)
    policy.qnet = nn.Linear(1, 1)
    policy.qnet_target = nn.Linear(1, 1).requires_grad_(False)
    policy.encoder_priv = nn.Linear(1, 1).requires_grad_(False)
    policy.actor = nn.Linear(1, 1).requires_grad_(False)
    policy.adapt_ema = nn.Linear(1, 1).requires_grad_(False)
    policy.object_adapt_ema = nn.Linear(1, 1).requires_grad_(False)
    policy.temporal_depth_gru_ema = nn.Linear(1, 1).requires_grad_(False)

    owned = {
        "perception_consolidation": {
            "adapt_module",
            "object_adapt",
            "temporal_depth_gru",
        },
        "actor_realignment": {"actor_adapt"},
        "perception_recheck": {
            "adapt_module",
            "object_adapt",
            "temporal_depth_gru",
        },
        "replay_q_calibration": {"qnet"},
    }
    online = {
        name: getattr(policy, name)
        for name in (
            "adapt_module",
            "object_adapt",
            "temporal_depth_gru",
            "actor_adapt",
            "qnet",
        )
    }
    always_frozen = (
        policy.qnet_target,
        policy.encoder_priv,
        policy.actor,
        policy.adapt_ema,
        policy.object_adapt_ema,
        policy.temporal_depth_gru_ema,
    )

    for phase, expected_trainable in owned.items():
        policy._apply_finalization_freeze_mask(phase)
        for name, module in online.items():
            expected = name in expected_trainable
            assert all(
                parameter.requires_grad is expected
                for parameter in module.parameters()
            )
            assert module.training is expected
        assert all(
            not parameter.requires_grad
            for module in always_frozen
            for parameter in module.parameters()
        )


def test_finalization_runtime_beta_half_does_not_mutate_source_controller():
    envs = 64
    policy = _finalization_policy(
        perception=1,
        actor=0,
        recheck=0,
        calibration=1,
        teacher_probability=0.5,
    )
    policy.finalization_rollout_count = 1
    policy.dagger_rollout_count = 1_234
    policy.dagger_rng = torch.Generator().manual_seed(87)
    policy.test_student_action = torch.ones(envs, 1)
    policy.test_teacher_action = torch.zeros(envs, 1)
    policy._student_action = lambda td: policy.test_student_action.clone()
    policy._teacher_action = lambda td: policy.test_teacher_action.clone()
    source_backend = policy._checkpoint_config().copy()

    expected_rng = torch.Generator()
    expected_rng.set_state(policy.dagger_rng.get_state())
    expected_teacher = torch.rand(envs, generator=expected_rng) < 0.5
    rollout_policy = _DaggerRolloutPolicy(policy)
    td = TensorDict(
        {"is_init": torch.zeros(envs, dtype=torch.bool)},
        batch_size=[envs],
    )

    actual = rollout_policy(td)

    assert policy._effective_control_mode() == "beta"
    assert policy._teacher_mixture_probability() == pytest.approx(0.5)
    assert torch.equal(actual[DAGGER_BETA_TEACHER_KEY], expected_teacher)
    assert torch.equal(
        actual[DAGGER_IS_STUDENT_ACTION_KEY], ~expected_teacher
    )
    assert not actual[DAGGER_SAFE_TEACHER_KEY].any()
    assert torch.equal(policy.dagger_rng.get_state(), expected_rng.get_state())
    # Runtime finalization controls must not rewrite checkpoint compatibility
    # fields inherited from the completed BC-DAgger source.
    assert policy._checkpoint_config() == source_backend
    assert policy.cfg.dagger_control_mode == "safe"
    assert policy.cfg.dagger_beta_start == pytest.approx(1.0)
    assert policy.cfg.dagger_beta_end == pytest.approx(0.0)


def test_device_replay_clear_drops_rows_allocations_and_cached_indices():
    replay = _DeviceReplay(capacity=8, device="cpu")
    replay.extend(
        {
            "row": torch.arange(4),
            "valid": torch.tensor([True, False, True, False]),
        }
    )
    assert replay.valid_count("valid") == 2
    assert replay._valid_index_cache

    replay.clear()

    assert replay.data == {}
    assert replay.ptr == 0
    assert replay.size == 0
    assert replay.seen == 0
    assert replay._valid_index_cache == {}


class _FinalizationTeacherExportRecorder:
    device = torch.device("cpu")

    def __init__(self):
        self.rows = []
        self.size = 0
        self.seen = 0

    def append(self, rows):
        copied = {key: value.detach().clone() for key, value in rows.items()}
        self.rows.append(copied)
        count = int(copied["rewards"].shape[0])
        self.size += count
        self.seen += count
        return count


def test_finalization_h5_and_q_teacher_partition_start_at_calibration_only():
    policy = _finalization_policy(
        perception=0,
        actor=1,
        recheck=0,
        calibration=1,
    )
    policy.cfg.train_every = 2
    # This test isolates replay routing; optimizer dispatch is covered above.
    policy.cfg.dagger_updates_per_rollout = 0
    policy.cfg.q_updates_per_rollout = 0
    policy.cfg.q_learning_starts_per_source = 8_192
    policy.device = torch.device("cpu")
    policy.adapt_module = nn.Linear(1, 1)
    policy.object_adapt = nn.Linear(1, 1)
    policy.temporal_depth_gru = nn.Linear(1, 1)
    policy.actor_adapt = nn.Linear(1, 1)
    policy.qnet = nn.Linear(1, 1)
    policy.dagger_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.teacher_replay = _FinalizationTeacherExportRecorder()
    policy.dagger_rollout_count = 400
    policy.dagger_environment_steps = 12_800
    policy.bc_update_count = 9
    policy.q_update_count = 10
    policy.num_updates = 0
    policy._last_truncation_finals_used = 0
    policy.train_adapt = lambda td: pytest.fail(
        "actor/Q-only finalization phase called train_adapt"
    )

    transitions = {
        "observations": torch.zeros(3, 1),
        "critic_observations": torch.zeros(3, 1),
        "actions": torch.tensor([[10.0], [20.0], [30.0]]),
        "rewards": torch.zeros(3),
        "dones": torch.zeros(3, dtype=torch.bool),
        "truncations": torch.zeros(3, dtype=torch.bool),
        "discounts": torch.ones(3),
        "next_observations": torch.zeros(3, 1),
        "next_critic_observations": torch.zeros(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[10.0], [21.0], [31.0]]
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([False, True, False]),
    }
    policy._dagger_transition_chunks = lambda td: iter([transitions])
    rollout = TensorDict({}, batch_size=[1, 2])

    actor_info = policy.train_op(rollout)

    assert actor_info["dagger/finalization_phase_actor_realignment"] == 1.0
    assert policy.dagger_replay.size == 3
    # Actor realignment may use the all-transition ring for BC, but it must not
    # contaminate either Phase-4 teacher source with stale priv_pred rows.
    assert policy.q_teacher_replay.size == 0
    assert policy.teacher_replay.rows == []

    calibration_info = policy.train_op(rollout)

    assert calibration_info[
        "dagger/finalization_phase_replay_q_calibration"
    ] == 1.0
    # Entry to calibration clears the actor-realignment ring before inserting
    # the fresh rollout. Only its one valid, teacher-executed row is exported.
    assert policy.dagger_replay.size == 3
    assert policy.dagger_replay.seen == 3
    assert policy.q_teacher_replay.size == 1
    assert policy.q_teacher_replay.seen == 1
    assert len(policy.teacher_replay.rows) == 1
    exported = policy.teacher_replay.rows[0]
    assert set(exported) == set(TEACHER_REPLAY_FIELDS)
    assert torch.equal(exported["actions"], torch.tensor([[10.0]]))


def test_finalization_checkpoint_metadata_is_separate_from_v3_backend(
    monkeypatch,
):
    policy = _finalization_policy(
        perception=25,
        actor=0,
        recheck=0,
        calibration=128,
    )
    policy.finalization_rollout_count = 25
    policy._finalization_last_phase = "perception_consolidation"
    policy._finalization_source_state = {
        "dagger_rollout_count": 2_000,
        "q_update_count": 99,
        "teacher_replay_id": "stale-source-replay",
    }
    policy.dagger_rollout_count = 2_025
    policy.dagger_environment_steps = 64_800
    policy.bc_update_count = 100
    policy.q_update_count = 101
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.sac_action_rng = torch.Generator().manual_seed(3)
    policy.bc_optimizer = SimpleNamespace(state_dict=lambda: {"name": "bc"})
    policy.opt_q = SimpleNamespace(state_dict=lambda: {"name": "q"})
    policy.opt_adapt = SimpleNamespace(
        state_dict=lambda: {"name": "adapt"}
    )
    policy.teacher_replay_id = "fresh-calibration-replay"
    policy.teacher_replay = None
    policy._loaded_teacher_replay_metadata = None
    policy._replay_vecnorm_fingerprint = "sha256:" + "d" * 64
    policy.env = SimpleNamespace(current_iter=77)
    policy._q_backend_metadata = lambda: {"contract": "unchanged-v3"}
    monkeypatch.setattr(PPOVEL, "state_dict", lambda self: {})

    state = policy.state_dict()

    assert state["training_algorithm"] == PPO_BC_DAGGER_TRAINING_ALGORITHM
    assert state["dagger_backend_config"] == policy._checkpoint_config()
    assert not any(
        key.startswith("dagger_finalize")
        for key in state["dagger_backend_config"]
    )
    finalization = state["bc_dagger_finalization_state"]
    assert finalization == {
        "semantics": DAGGER_FINALIZATION_SEMANTICS,
        "config": policy._finalization_config(),
        "rollout_count": 25,
        "phase": "replay_q_calibration",
        "last_phase": "perception_consolidation",
        "complete": False,
        "source_state": policy._finalization_source_state,
        "fresh_replay_id": "fresh-calibration-replay",
    }


def test_finalization_resume_restores_local_stage_counter_and_source_state(
    monkeypatch,
):
    policy = _finalization_policy(
        perception=25,
        actor=0,
        recheck=0,
        calibration=128,
    )
    policy.device = torch.device("cpu")
    policy._replay_vecnorm_fingerprint = None
    policy.dagger_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.qnet = nn.Linear(1, 1, bias=False)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    bc_parameters = nn.Linear(1, 1, bias=False)
    adapt_parameters = nn.Linear(1, 1, bias=False)
    policy.bc_optimizer = torch.optim.AdamW(bc_parameters.parameters())
    policy.opt_q = torch.optim.AdamW(policy.qnet.parameters())
    policy.opt_adapt = torch.optim.Adam(adapt_parameters.parameters())
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.sac_action_rng = torch.Generator().manual_seed(3)
    policy.teacher_replay_id = "fresh-calibration-replay"
    policy._loaded_teacher_replay_metadata = None
    progress = []
    policy.env = SimpleNamespace(set_progress=progress.append)
    policy._q_backend_metadata = lambda: {"contract": "v3"}

    source_state = {
        "training_algorithm": PPO_BC_DAGGER_TRAINING_ALGORITHM,
        "dagger_rollout_count": 2_000,
        "dagger_environment_steps": 64_000,
        "bc_update_count": 90,
        "q_update_count": 100,
        "teacher_replay_id": "pre-finalization-replay",
    }
    checkpoint = {
        "training_algorithm": PPO_BC_DAGGER_TRAINING_ALGORITHM,
        "last_phase": "finetune",
        "actor_backend": bc_dagger_module.PPO_BC_DAGGER_ACTOR_BACKEND,
        "critic_learning_semantics": PPO_BC_DAGGER_SAC_CRITIC_SEMANTICS,
        "actor_learning_semantics": PPO_BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "teacher_action_semantics": (
            bc_dagger_module.DAGGER_TEACHER_ACTION_SEMANTICS
        ),
        "dagger_control_semantics": DAGGER_CONTROL_SEMANTICS,
        "dagger_backend_config": policy._checkpoint_config(),
        "q_backend_config": {"contract": "v3"},
        "actor": {},
        "actor_adapt": {},
        "encoder_priv": {},
        "adapt_module": {},
        "adapt_ema": {},
        "object_adapt": {},
        "object_adapt_ema": {},
        "qnet": policy.qnet.state_dict(),
        "qnet_target": policy.qnet_target.state_dict(),
        "bc_dagger_sac_adapter": {},
        "optimizer_resume_state": {
            "bc_optimizer": policy.bc_optimizer.state_dict(),
            "q_optimizer": policy.opt_q.state_dict(),
            "adapt_optimizer": policy.opt_adapt.state_dict(),
        },
        "dagger_rollout_count": 2_025,
        "dagger_environment_steps": 64_800,
        "bc_update_count": 90,
        "q_update_count": 356,
        "dagger_rng_state": torch.Generator().manual_seed(11).get_state(),
        "q_rng_state": torch.Generator().manual_seed(12).get_state(),
        "sac_action_rng_state": torch.Generator().manual_seed(13).get_state(),
        "teacher_replay_id": "fresh-calibration-replay",
        "next_iter": 9_001,
        "bc_dagger_finalization_state": {
            "semantics": DAGGER_FINALIZATION_SEMANTICS,
            "config": policy._finalization_config(),
            "rollout_count": 25,
            "phase": "replay_q_calibration",
            "last_phase": "perception_consolidation",
            "complete": False,
            "source_state": source_state,
            "fresh_replay_id": "fresh-calibration-replay",
        },
    }
    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda self, state_dict, strict=True: [],
    )

    policy.load_state_dict(checkpoint)

    assert policy.finalization_rollout_count == 25
    assert policy._finalization_phase() == "replay_q_calibration"
    assert policy._finalization_last_phase == "perception_consolidation"
    assert policy._finalization_source_state == source_state
    assert policy.dagger_rollout_count == 2_025
    assert policy.q_update_count == 356
    assert policy.teacher_replay_id == "fresh-calibration-replay"
    assert progress == [9_001]
