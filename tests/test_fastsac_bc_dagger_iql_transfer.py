import copy
import functools
import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from active_adaptation.learning.ppo.fastsac_vel import (
    BC_DAGGER_ACTOR_BACKEND,
    BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
    BC_DAGGER_FRESH_ACTOR_INITIALIZATION_SEMANTICS,
    BC_DAGGER_STAGING_FINAL_ONLY_REPLAY_SEMANTICS,
    BC_DAGGER_SAC_CRITIC_SEMANTICS,
    BC_DAGGER_LEGACY_TRAINING_ALGORITHM,
    BC_DAGGER_TRAINING_ALGORITHM,
    FASTSAC_BC_DAGGER_ACTOR_BACKEND,
    STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY,
    STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY,
    FastSACTanhNormal,
    FastSACVEL,
    FastSACVelFinetune,
    FastSACVelFinetuneConfig,
    OnlineReplay,
    _BCDaggerSACAdapter,
    _BCDaggerSACRolloutActor,
    _validate_complete_bc_dagger_staging_source,
    _validate_fastsac_finetune_config,
    _vaic_action_contract_metadata,
)
from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    _LatentStudentRolloutPolicy,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from scripts.helpers import _apply_direct_sac_dagger_q_transfer


_DELETE = object()


def _complete_staging_source_state():
    replay_id = "fresh-staged-replay"
    semantics = (
        "joint_then_cyclic_perception_actor_then_final_perception_actor_"
        "then_fresh_replay_q_v1"
    )
    return {
        "teacher_replay_id": replay_id,
        "dagger_rollout_count": 3000,
        "q_update_count": 101,
        "teacher_replay_state": {
            "replay_id": replay_id,
            "snapshot_id": "staged-final-snapshot",
            "snapshot_iteration": 8999,
            "checkpoint_name": "checkpoint_final",
            "size": 1024,
            "seen": 1024,
        },
        "bc_dagger_staging_state": {
            "semantics": semantics,
            "config": {
                "semantics": semantics,
                "joint_warmup_iterations": 500,
                "cycles": 7,
                "perception_iterations": 100,
                "actor_iterations": 200,
                "final_perception_iterations": 100,
                "final_actor_iterations": 172,
                "replay_q_calibration_iterations": 128,
                "calibration_control_mode": "beta",
                "calibration_teacher_probability": 0.5,
                "h5_final_only": False,
            },
            "rollout_count": 3000,
            "phase": "complete",
            "last_phase": "replay_q_calibration",
            "complete": True,
            "fresh_replay_id": replay_id,
            "persistent_replay_semantics": (
                "h5_disabled_until_final_q_calibration_v1"
            ),
            "calibration_start_q_update_count": 100,
            "calibration_q_updates": 1,
        },
    }


def test_stage2_accepts_only_complete_staged_bc_dagger_source():
    state = _complete_staging_source_state()

    _validate_complete_bc_dagger_staging_source(state)

    state["bc_dagger_staging_state"]["complete"] = False
    state["bc_dagger_staging_state"]["phase"] = "final_actor"
    state["bc_dagger_staging_state"]["rollout_count"] = 2872
    with pytest.raises(ValueError, match="completed staged rollout budget"):
        _validate_complete_bc_dagger_staging_source(state)


def _inline_complete_staging_source_state():
    state = _complete_staging_source_state()
    config = state["bc_dagger_staging_state"]["config"]
    config.update(
        {
            "joint_warmup_iterations": 2400,
            "cycles": 0,
            "perception_iterations": 0,
            "actor_iterations": 0,
            "final_perception_iterations": 0,
            "final_actor_iterations": 50,
            "replay_q_calibration_iterations": 128,
            "h5_final_only": True,
        }
    )
    completed = 2400 + 50 + 128
    state["dagger_rollout_count"] = completed
    state["bc_dagger_staging_state"]["rollout_count"] = completed
    state["bc_dagger_staging_state"]["persistent_replay_semantics"] = (
        BC_DAGGER_STAGING_FINAL_ONLY_REPLAY_SEMANTICS
    )
    return state


def test_stage2_accepts_complete_inline_joint_actor_q_source():
    state = _inline_complete_staging_source_state()

    _validate_complete_bc_dagger_staging_source(state)


@pytest.mark.parametrize(
    "field",
    ("perception_iterations", "actor_iterations"),
)
def test_stage2_rejects_unused_nonzero_cycle_lengths_when_cycles_are_zero(
    field,
):
    state = _inline_complete_staging_source_state()
    state["bc_dagger_staging_state"]["config"][field] = 1

    with pytest.raises(ValueError, match="cycles|zero|unused"):
        _validate_complete_bc_dagger_staging_source(state)


def test_stage2_rejects_inline_source_without_final_checkpoint_snapshot():
    state = _inline_complete_staging_source_state()
    state["teacher_replay_state"]["checkpoint_name"] = "checkpoint_2552"

    with pytest.raises(ValueError, match="checkpoint_final|final replay"):
        _validate_complete_bc_dagger_staging_source(state)


def test_stage2_rejects_non_boolean_staged_h5_publication_contract():
    state = _inline_complete_staging_source_state()
    state["bc_dagger_staging_state"]["config"]["h5_final_only"] = 1

    with pytest.raises(ValueError, match="h5_final_only|boolean"):
        _validate_complete_bc_dagger_staging_source(state)


def test_stage2_rejects_staged_source_with_wrong_replay_lineage():
    state = _complete_staging_source_state()
    state["teacher_replay_id"] = "wrong-replay"

    with pytest.raises(ValueError, match="replay lineage"):
        _validate_complete_bc_dagger_staging_source(state)


def test_stage2_rejects_staged_source_without_terminal_q_updates():
    state = _complete_staging_source_state()
    state["bc_dagger_staging_state"]["calibration_q_updates"] = 0
    state["q_update_count"] = 100

    with pytest.raises(ValueError, match="calibration_q_updates"):
        _validate_complete_bc_dagger_staging_source(state)


def _stage2_policy(*, load_pretrained_q=True):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        load_pretrained_q=load_pretrained_q,
        q_hidden_dim=8,
        q_num_atoms=501,
        q_v_min=-2.0,
        q_v_max=2.0,
        q_layer_norm=True,
        q_action_coordinates="absolute",
        q_action_fusion="late",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_q_normalize_actions=True,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
        use_object_adapt=False,
        sac_bc_action_clip=20.0,
        sac_bc_initial_action_std=0.05,
        sac_stage2_initial_action_std=None,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
        latent_dim=1,
        q_seed=17,
        q_lr=3e-5,
        q_weight_decay=1e-3,
        sac_tau=0.001,
        sac_max_grad_norm=1.0,
        sac_alpha_init=0.01,
    )
    policy.actor_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.bc_dagger_actor_anchor = copy.deepcopy(
        policy.actor_adapt
    ).requires_grad_(False)
    action_low = torch.tensor([-2.0])
    action_high = torch.tensor([4.0])
    action_center = (action_low + action_high) * 0.5
    action_scale = (action_high - action_low) * 0.5
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=torch.log(
            torch.full_like(action_scale, policy.cfg.sac_bc_initial_action_std)
            / action_scale
        ),
        device="cpu",
    )
    policy.qnet = torch.nn.Linear(1, 1, bias=False)
    policy.qnet_target = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.qnet.weight.zero_()
        policy.qnet_target.weight.zero_()
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.q_rng = torch.Generator().manual_seed(1)
    policy.sac_action_rng = torch.Generator().manual_seed(2)
    policy.sac_rollout_rng = torch.Generator().manual_seed(3)
    policy.teacher_replay_id = "stage2-replay"
    policy._replay_vecnorm_fingerprint = "frozen-vecnorm"
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy._q_input_dim = 1
    policy.action_dim = 1
    policy.joint_names = ["joint"]
    policy._fastsac_action_low = action_low.tolist()
    policy._fastsac_action_high = action_high.tolist()
    policy._fastsac_actor_action_center = action_center
    policy._fastsac_actor_action_scale = action_scale
    policy._fastsac_q_action_center = action_center
    policy._fastsac_q_action_scale = action_scale
    policy._fastsac_action_log_scale_sum = float(torch.log(action_scale).sum())
    policy._fastsac_entropy_reference_log_scale_sum = float(
        torch.log(action_scale).sum()
    )
    policy._fastsac_action_contract = _vaic_action_contract_metadata(
        policy.joint_names,
        action_low,
        action_high,
        torch.zeros_like(action_low),
        torch.zeros_like(action_high),
    )
    policy._vaic_action_bounds = lambda: (action_low, action_high)
    policy.sac_update_row_credit = 0.0
    policy.reward_groups = ["task"]
    policy.cfg.gamma = 0.99
    policy.observation_spec = {
        "actor_obs": torch.zeros(1),
        "critic_obs": torch.zeros(1),
        "vel_command": torch.zeros(1),
        "policy": torch.zeros(1),
    }
    return policy


def test_stage2_checkpoint_before_q_updates_needs_no_teacher_export_state(
    monkeypatch,
):
    policy = _stage2_policy()
    policy.cfg = FastSACVelFinetuneConfig(
        finetune_checkpoint_source="bc_dagger"
    )
    policy.env = SimpleNamespace(current_iter=0)
    policy.actor_backend = FASTSAC_BC_DAGGER_ACTOR_BACKEND
    policy.teacher_replay = None
    policy.offline_replay = None
    policy.q_update_count = 0
    policy.sac_environment_steps = 0
    policy.sac_rollout_count = 0
    policy.sac_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.num_updates = 0
    monkeypatch.setattr(policy, "_actor_backend_metadata", lambda: {})
    monkeypatch.setattr(policy, "_q_backend_metadata", lambda: {})

    assert not hasattr(policy, "_teacher_export_started")
    assert not hasattr(policy, "_teacher_export_start_seen")

    checkpoint = policy.state_dict()

    assert checkpoint["last_phase"] == "finetune"
    assert checkpoint["q_update_count"] == 0
    assert checkpoint["teacher_export_started"] is False


def test_stage2_loads_pretrained_q_by_default_and_validates_boolean_option():
    cfg = FastSACVelFinetuneConfig()
    assert cfg.load_pretrained_q is True

    cfg.load_pretrained_q = "false"
    with pytest.raises(ValueError, match="load_pretrained_q"):
        _validate_fastsac_finetune_config(cfg)


def _helper_direct_sac_q_backend(*, fusion="late", gain=1.0):
    backend = {
        "q_action_coordinates": "absolute",
        "q_action_normalized": True,
        "q_action_input_gain": gain,
        "clipped_double_q": True,
    }
    if fusion is not None:
        backend["q_action_fusion"] = fusion
    return backend


def test_helper_materializes_late_unit_gain_bc_dagger_q_transfer():
    algo = OmegaConf.create({
        # Source metadata must win over destination defaults/overrides whenever
        # pretrained tensors are requested.
        "q_action_coordinates": "reference_residual",
        "q_action_fusion": "early",
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
        "sac_q_normalize_actions": False,
        "sac_q_action_input_gain": 7.0,
        "sac_clipped_double_q": False,
    })

    _apply_direct_sac_dagger_q_transfer(
        algo, _helper_direct_sac_q_backend()
    )

    assert algo.q_action_coordinates == "absolute"
    assert algo.q_action_fusion == "late"
    assert algo.sac_q_normalize_actions is True
    assert algo.sac_q_action_input_gain == pytest.approx(1.0)
    assert algo.sac_clipped_double_q is True


@pytest.mark.parametrize("fusion", ("early", None))
def test_helper_requires_fresh_q_for_legacy_early_bc_dagger_source(fusion):
    algo = OmegaConf.create({
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
    })

    with pytest.raises(
        ValueError,
        match=(
            r"late-fusion source.*load_pretrained_q=false.*fresh Stage-2 Q"
        ),
    ):
        _apply_direct_sac_dagger_q_transfer(
            algo, _helper_direct_sac_q_backend(fusion=fusion)
        )


def test_helper_preserves_positive_non_unit_gain_late_bc_dagger_q_source():
    algo = OmegaConf.create({
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
    })

    _apply_direct_sac_dagger_q_transfer(
        algo, _helper_direct_sac_q_backend(gain=2.0)
    )

    assert algo.sac_q_action_input_gain == pytest.approx(2.0)


@pytest.mark.parametrize("gain", (0.0, -1.0, float("inf"), float("nan")))
def test_helper_rejects_non_positive_or_non_finite_late_q_gain(gain):
    algo = OmegaConf.create({
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
    })

    with pytest.raises(ValueError, match="finite-positive-gain"):
        _apply_direct_sac_dagger_q_transfer(
            algo, _helper_direct_sac_q_backend(gain=gain)
        )


def test_stochastic_stage2_resume_requires_checkpointed_rollout_rng():
    policy = _stage2_policy()
    policy.cfg.sac_deterministic_rollout = False

    with pytest.raises(ValueError, match="rollout RNG state"):
        policy.load_state_dict({
            "last_phase": "finetune",
            "qnet": {},
        })


@pytest.mark.parametrize(
    ("override", "load_pretrained_q", "message"),
    [
        (
            {"q_action_coordinates": "reference_residual"},
            False,
            "fields",
        ),
        ({"q_reference_dueling": True}, False, "fields"),
        ({"q_condition_on_actuator_state": True}, False, "fields"),
        ({"sac_q_normalize_actions": False}, True, "normalize_actions"),
    ],
)
def test_bc_dagger_q_coordinates_must_match_checkpoint_and_replay_schema(
    override, load_pretrained_q, message
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    values = {
        "q_action_coordinates": "absolute",
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
        "q_action_fusion": "late",
        "sac_q_action_input_gain": 1.0,
        "sac_q_normalize_actions": True,
        "load_pretrained_q": load_pretrained_q,
    }
    values.update(override)
    policy.cfg = SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        policy._configure_bc_dagger_actor_backend()


def test_bc_pretrained_q_backend_accepts_normalized_late_unit_gain(
    monkeypatch,
):
    policy = _stage2_policy()
    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25])),
        load_noise_scale=123.0,
    )
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )

    policy._configure_bc_dagger_actor_backend()

    assert policy.actor_backend == FASTSAC_BC_DAGGER_ACTOR_BACKEND
    assert policy.cfg.q_num_atoms == 501
    assert policy.cfg.q_layer_norm is True
    assert policy.cfg.q_action_fusion == "late"
    assert policy.cfg.sac_q_normalize_actions is True
    assert policy.cfg.sac_q_action_input_gain == pytest.approx(1.0)
    assert policy.cfg.sac_clipped_double_q is True
    assert policy._fastsac_action_low == [-20.0]
    assert policy._fastsac_action_high == [20.0]
    assert torch.equal(
        policy._fastsac_actor_action_scale, torch.tensor([20.0])
    )
    assert torch.equal(policy._fastsac_q_action_center, torch.tensor([1.0]))
    assert torch.equal(policy._fastsac_q_action_scale, torch.tensor([3.0]))
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(
        math.log(policy.cfg.sac_bc_initial_action_std / 3.0)
    )
    assert policy._fastsac_entropy_reference_log_scale_sum == pytest.approx(
        math.log(3.0)
    )
    assert policy._fastsac_action_contract["q_action_clamp"] is None
    assert policy._fastsac_action_contract[
        "q_action_transform_fingerprint"
    ].startswith("sha256:")
    assert actor_core.load_noise_scale is None


def _v3_bc_critic_checkpoint(policy):
    q_backend = policy._q_backend_metadata()
    q_backend.update({
        "alpha_autotune": False,
        "pretrain_effective_alpha": 0.0,
        "stage2_alpha_init": float(policy.cfg.sac_alpha_init),
        "pretrain_backup_semantics": (
            "stochastic_next_action_q_only_effective_alpha_zero_v1"
        ),
        "pretrain_target_policy": (
            "dedicated_small_noise_bc_centered_nominal_bounded_residual_v3"
        ),
        "replay_mix_semantics": (
            "beta_independent_teacher_executed_0.5_"
            "student_executed_0.5_v1"
        ),
    })
    return {
        "training_algorithm": BC_DAGGER_TRAINING_ALGORITHM,
        "actor_backend": BC_DAGGER_ACTOR_BACKEND,
        "critic_learning_semantics": BC_DAGGER_SAC_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "vecnorm_fingerprint": "frozen-vecnorm",
        "action_contract": copy.deepcopy(policy._fastsac_action_contract),
        "dagger_backend_config": {
            "dagger_action_clip": 20.0,
            "fresh_ppo_actor_initialization_semantics": (
                BC_DAGGER_FRESH_ACTOR_INITIALIZATION_SEMANTICS
            ),
            "q_hidden_dim": 8,
            "q_num_atoms": 501,
            "q_v_min": -2.0,
            "q_v_max": 2.0,
            "q_layer_norm": True,
            "q_action_fusion": "late",
            "q_action_coordinates": "absolute",
            "sac_q_normalize_actions": True,
            "sac_q_action_input_gain": 1.0,
            "sac_clipped_double_q": True,
            "q_lr": 3e-5,
            "q_weight_decay": 1e-3,
            "q_tau": 0.001,
            "q_max_grad_norm": 1.0,
            "sac_bc_initial_action_std": float(
                policy.cfg.sac_bc_initial_action_std
            ),
            "sac_bc_log_std_min": float(policy.cfg.sac_bc_log_std_min),
            "sac_bc_log_std_max": float(policy.cfg.sac_bc_log_std_max),
        },
        # BC pretraining must expose the same complete Q contract consumed by
        # FastSAC checkpoints, rather than relying on a few coincidentally equal
        # tensor shapes in dagger_backend_config.
        "q_backend_config": q_backend,
        "qnet": {"weight": torch.tensor([[3.0]])},
        "qnet_target": {"weight": torch.tensor([[-4.0]])},
        "bc_dagger_sac_adapter": {"log_std": torch.tensor([-6.0])},
        "q_update_count": 123,
        "teacher_replay_id": "dagger-replay",
        "teacher_replay_state": {"snapshot_id": "dagger-snapshot"},
    }


def test_stage2_accepts_full_fastsac_501_normalized_late_ln_bc_q_metadata(
    monkeypatch,
):
    policy = _stage2_policy()
    copied = []
    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )

    def copy_modules(self, state_dict, strict=True):
        copied.append(strict)
        return []

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )
    state = _v3_bc_critic_checkpoint(policy)

    assert "iql_value" not in state
    assert state["q_backend_config"]["num_atoms"] == 501
    assert state["q_backend_config"]["q_action_normalized"] is True
    assert state["q_backend_config"]["q_action_fusion"] == "late"
    assert state["q_backend_config"]["layer_norm"] is True
    assert state["q_backend_config"]["q_action_input_gain"] == pytest.approx(1.0)
    assert state["q_backend_config"]["clipped_double_q"] is True

    failed = policy.load_state_dict(state)

    assert failed == []
    assert copied == [True]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("num_atoms", 101),
        ("q_action_normalized", False),
        ("q_action_fusion", "early"),
        ("layer_norm", False),
        ("q_action_input_gain", 2.0),
        ("clipped_double_q", False),
    ),
)
def test_stage2_rejects_bc_q_backend_metadata_mismatch_before_copy(
    monkeypatch, field, bad_value,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Q metadata mismatch reached module-copy path")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _v3_bc_critic_checkpoint(policy)
    state["q_backend_config"][field] = bad_value

    with pytest.raises(ValueError, match="Q.*(config|metadata)|backend"):
        policy.load_state_dict(state)


def test_stage2_rejects_legacy_bc_dagger_checkpoint_before_module_copy(
    monkeypatch,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("legacy checkpoint reached module-copy path")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _v3_bc_critic_checkpoint(policy)
    state["training_algorithm"] = BC_DAGGER_LEGACY_TRAINING_ALGORITHM

    with pytest.raises(ValueError, match="Legacy|predate"):
        policy.load_state_dict(state)


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        (
            "critic_learning_semantics",
            "plain-bellman-q",
            ValueError,
            "critic semantics mismatch",
        ),
        (
            "actor_learning_semantics",
            "q-weighted-actor",
            ValueError,
            "not trained by pure DAgger BC",
        ),
        (
            "q_update_count",
            0,
            RuntimeError,
            "Q1/Q2 were never updated",
        ),
    ],
)
def test_stage2_requires_complete_sac_critic_v3_training_provenance(
    monkeypatch, field, value, exception, message,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("invalid provenance reached module-copy path")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _v3_bc_critic_checkpoint(policy)
    if value is _DELETE:
        del state[field]
    else:
        state[field] = value

    with pytest.raises(exception, match=message):
        policy.load_state_dict(state)


def test_stage2_transfers_sac_critic_q_twins_targets_and_dedicated_std(
    monkeypatch,
):
    policy = _stage2_policy()
    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )
    ppo_std_before = actor_core.actor_std.detach().clone()
    initial_sac_log_std = policy.bc_dagger_sac_adapter.log_std.detach().clone()
    copied = []

    def copy_ppo_modules(self, state_dict, strict=True):
        copied.append(strict)
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        self.bc_dagger_sac_adapter.load_state_dict(
            state_dict["bc_dagger_sac_adapter"]
        )
        return []

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_ppo_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )

    checkpoint = _v3_bc_critic_checkpoint(policy)
    assert policy.cfg.sac_stage2_initial_action_std is None
    failed = policy.load_state_dict(checkpoint)

    assert failed == []
    assert copied == [True]
    assert policy.qnet.weight.item() == pytest.approx(3.0)
    assert policy.qnet_target.weight.item() == pytest.approx(-4.0)
    assert not policy.qnet_target.weight.requires_grad
    assert torch.equal(actor_core.actor_std, ppo_std_before)
    assert not actor_core.actor_std.requires_grad
    assert not torch.equal(
        policy.bc_dagger_sac_adapter.log_std,
        initial_sac_log_std,
    )
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(-6.0)
    assert torch.equal(
        policy.bc_dagger_actor_anchor.weight,
        policy.actor_adapt.weight,
    )
    assert not policy.bc_dagger_actor_anchor.weight.requires_grad
    assert "iql_value" not in checkpoint
    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0
    assert policy.teacher_replay_id == "dagger-replay"
    assert policy._loaded_teacher_replay_metadata == {
        "snapshot_id": "dagger-snapshot"
    }
    assert policy._bc_dagger_critic_source == {
        "critic_learning_semantics": BC_DAGGER_SAC_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "source_q_updates": 123,
        "q_backend_config": checkpoint["q_backend_config"],
        "q_weights_stage2_usage": "transferred_exact_sac_contract",
        "q_optimizer_stage2_usage": (
            "fresh_adamw_moments_same_lr_weight_decay"
        ),
        "q_update_counter_stage2_usage": "reset_for_stage2_schedule",
    }


def test_stage2_direct_v3_transfer_can_reset_dedicated_action_std(
    monkeypatch,
):
    policy = _stage2_policy()
    policy.cfg.sac_stage2_initial_action_std = 0.2
    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )
    adapter_values_during_copy = []

    def copy_ppo_modules(self, state_dict, strict=True):
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        self.bc_dagger_sac_adapter.load_state_dict(
            state_dict["bc_dagger_sac_adapter"]
        )
        adapter_values_during_copy.append(
            self.bc_dagger_sac_adapter.log_std.item()
        )
        return []

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_ppo_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )
    checkpoint = _v3_bc_critic_checkpoint(policy)

    failed = policy.load_state_dict(checkpoint)

    assert failed == []
    # Loading first proves this is an explicit transfer reset, not an ignored
    # checkpoint tensor or a constructor-only initial value.
    assert adapter_values_during_copy == [pytest.approx(-6.0)]
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(
        math.log(0.2 / policy._fastsac_actor_action_scale.item())
    )
    assert checkpoint["bc_dagger_sac_adapter"]["log_std"].item() == (
        pytest.approx(-6.0)
    )


def test_stage2_resume_restores_learned_adapter_without_reapplying_std_reset(
    monkeypatch,
):
    policy = _stage2_policy()
    policy.cfg = FastSACVelFinetuneConfig(
        finetune_checkpoint_source="bc_dagger",
        sac_stage2_initial_action_std=0.2,
    )
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))

    learned_log_std = torch.tensor([-5.5])

    def load_student(self, state_dict, strict=True):
        self.bc_dagger_sac_adapter.load_state_dict(
            state_dict["bc_dagger_sac_adapter"]
        )
        self.q_update_count = 77
        self.sac_actor_update_count = 0
        return []

    monkeypatch.setattr(FastSACVEL, "load_state_dict", load_student)
    checkpoint = {
        "qnet": {},
        "last_phase": "finetune",
        "stage2_schedule_config": policy._stage2_schedule_config(),
        "stage2_actor_release_q_update": None,
        "sac_update_row_credit": 0.0,
        "sac_rollout_rng_state": torch.Generator().manual_seed(9).get_state(),
        "bc_dagger_sac_adapter": {"log_std": learned_log_std.clone()},
    }

    failed = policy.load_state_dict(checkpoint)

    assert failed == []
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(-5.5)
    assert policy.bc_dagger_sac_adapter.log_std.item() != pytest.approx(
        math.log(0.2 / policy.cfg.sac_bc_action_clip)
    )


def test_stage2_can_transfer_bc_student_but_start_q_and_targets_fresh(
    monkeypatch,
):
    policy = _stage2_policy(load_pretrained_q=False)
    policy.actor_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.encoder_priv = torch.nn.Linear(1, 1, bias=False)
    policy.adapt_module = torch.nn.Linear(1, 1, bias=False)
    policy.adapt_ema = torch.nn.Linear(1, 1, bias=False)
    policy.object_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.object_adapt_ema = torch.nn.Linear(1, 1, bias=False)
    policy.depth_cnn = torch.nn.Linear(1, 1, bias=False)
    policy.temporal_depth_gru = torch.nn.Linear(1, 1, bias=False)
    policy.temporal_depth_gru_ema = torch.nn.Linear(1, 1, bias=False)
    policy.cfg.use_object_adapt = True

    with torch.no_grad():
        for module in (
            policy.actor_adapt,
            policy.encoder_priv,
            policy.adapt_module,
            policy.adapt_ema,
            policy.object_adapt,
            policy.object_adapt_ema,
            policy.depth_cnn,
            policy.temporal_depth_gru,
            policy.temporal_depth_gru_ema,
        ):
            module.weight.zero_()
        policy.qnet.weight.fill_(0.125)
        policy.qnet_target.weight.fill_(-0.75)
    policy.opt_q = torch.optim.AdamW(policy.qnet.parameters(), lr=3e-4)

    # A fresh Stage-2 process must never inherit progress from Stage 1 even if
    # stale values happen to exist before loading.
    policy.q_update_count = 101
    policy.sac_update_count = 102
    policy.sac_actor_update_count = 103
    policy.sac_alpha_update_count = 104
    policy.sac_environment_steps = 105
    policy.sac_rollout_count = 106

    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )
    ppo_std_before = actor_core.actor_std.detach().clone()
    initial_sac_log_std = policy.bc_dagger_sac_adapter.log_std.detach().clone()

    def copy_available_ppo_modules(self, state_dict, strict=True):
        failed = []
        for name, module in self.named_children():
            if name not in state_dict:
                failed.append(name)
                continue
            module.load_state_dict(state_dict[name], strict=strict)
        return failed

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_available_ppo_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )

    state = _v3_bc_critic_checkpoint(policy)
    # Opting out of Q transfer makes critic tensors/update provenance
    # irrelevant while preserving the pure-BC actor and dedicated SAC std.
    for field in (
        "qnet",
        "qnet_target",
        "q_update_count",
    ):
        state.pop(field)
    state["optimizers"] = {
        # Deliberately invalid as an Adam state: the Stage-1 optimizer belongs
        # to a different objective and must never be restored in Stage 2.
        "q_optimizer": {"stage1_only": True},
    }
    transferred_modules = (
        "actor_adapt",
        "encoder_priv",
        "adapt_module",
        "adapt_ema",
        "object_adapt",
        "object_adapt_ema",
        "depth_cnn",
        "temporal_depth_gru",
        "temporal_depth_gru_ema",
    )
    for index, name in enumerate(transferred_modules, start=1):
        state[name] = {"weight": torch.tensor([[float(index)]])}

    failed = policy.load_state_dict(state)

    assert "qnet" not in failed
    assert "qnet_target" not in failed
    for index, name in enumerate(transferred_modules, start=1):
        assert getattr(policy, name).weight.item() == pytest.approx(float(index))

    # The online Q remains at its seeded Stage-2 initialization and the target
    # becomes an exact, frozen copy of that fresh online network.
    assert policy.qnet.weight.item() == pytest.approx(0.125)
    assert policy.qnet_target.weight.item() == pytest.approx(0.125)
    assert not policy.qnet_target.weight.requires_grad
    assert torch.equal(actor_core.actor_std, ppo_std_before)
    assert not actor_core.actor_std.requires_grad
    assert not torch.equal(
        policy.bc_dagger_sac_adapter.log_std,
        initial_sac_log_std,
    )
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(-6.0)
    assert torch.equal(
        policy.bc_dagger_actor_anchor.weight,
        policy.actor_adapt.weight,
    )
    assert not policy.bc_dagger_actor_anchor.weight.requires_grad
    assert policy.opt_q.state == {}
    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0
    assert policy.sac_actor_update_count == 0
    assert policy.sac_alpha_update_count == 0
    assert policy.sac_environment_steps == 0
    assert policy.sac_rollout_count == 0
    assert policy._bc_dagger_critic_source == {
        "critic_learning_semantics": BC_DAGGER_SAC_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "source_q_updates": 0,
        "q_weights_stage2_usage": "discarded_fresh_q_seed",
        "q_seed": 17,
    }


def test_fresh_q_option_still_requires_pure_bc_actor_provenance(monkeypatch):
    policy = _stage2_policy(load_pretrained_q=False)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("invalid actor provenance reached module copy")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _v3_bc_critic_checkpoint(policy)
    state["actor_learning_semantics"] = "q-weighted-actor"
    for field in (
        "qnet",
        "qnet_target",
        "q_update_count",
    ):
        state.pop(field)

    with pytest.raises(ValueError, match="not trained by pure DAgger BC"):
        policy.load_state_dict(state)


def test_stage2_rejects_safety_clip_that_does_not_contain_action_contract(
    monkeypatch,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("action-support mismatch reached module copy")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _v3_bc_critic_checkpoint(policy)
    state["dagger_backend_config"]["dagger_action_clip"] = 2.0

    with pytest.raises(
        ValueError,
        match="safety clip|safety envelope|contain every executable",
    ):
        policy.load_state_dict(state)


def test_stage2_deterministic_rollout_exactly_matches_dagger_latent_tanh():
    raw_action = torch.tensor(
        [[-25.0, -1.5, 0.25, 30.0, float("nan"), float("inf"), -float("inf")]]
    )
    action_clip = 20.0

    class _RawActionPolicy(torch.nn.Module):
        def forward(self, td):
            td[ACTION_KEY] = raw_action.expand(*td.batch_size, -1)
            return td

    class _MeanDistActor(torch.nn.Module):
        def get_dist(self, td):
            return SimpleNamespace(mean=raw_action.expand(*td.batch_size, -1))

    source_td = TensorDict({}, batch_size=[1])
    low = torch.tensor([-2.0, -2.0, -1.0, -4.0, -5.0, -6.0, -7.0])
    high = torch.tensor([3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0])
    _LatentStudentRolloutPolicy(
        _RawActionPolicy(), low, high, action_clip=action_clip
    )(source_td)

    stage2 = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(stage2)
    stage2.cfg = SimpleNamespace(
        sac_bc_action_clip=action_clip,
        sac_bc_initial_action_std=0.05,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
    )
    stage2.actor_adapt = _MeanDistActor()
    stage2.bc_dagger_actor_anchor = copy.deepcopy(
        stage2.actor_adapt
    ).requires_grad_(False)
    stage2.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=raw_action.shape[-1],
        initial_log_std=torch.log(
            torch.full_like(low, stage2.cfg.sac_bc_initial_action_std)
            / ((high - low) * 0.5)
        ),
        device="cpu",
    )
    stage2._fastsac_action_low = low.tolist()
    stage2._fastsac_action_high = high.tolist()
    stage2._fastsac_q_action_center = (low + high) * 0.5
    stage2._fastsac_q_action_scale = (high - low) * 0.5
    stage2.dist_cls = functools.partial(
        FastSACTanhNormal,
        low=low,
        high=high,
        event_dims=1,
    )

    stage2_td = TensorDict({}, batch_size=[1])
    _BCDaggerSACRolloutActor(stage2)(stage2_td)

    # A BC action rounded onto an execution endpoint is moved inward by the
    # residual distribution's documented bijectivity epsilon. Everywhere else
    # the fresh zero-residual Stage-2 policy is the transferred DAgger policy.
    parity_error = (stage2_td[ACTION_KEY] - source_td[ACTION_KEY]).abs()
    assert torch.all(
        parity_error
        <= (high - low) * (1e-6 + torch.finfo(torch.float32).eps)
    )
    assert ((stage2_td[ACTION_KEY] >= low) & (
        stage2_td[ACTION_KEY] <= high
    )).all()
    assert torch.isfinite(stage2_td["loc"]).all()
    assert torch.isfinite(stage2_td["scale"]).all()
    assert torch.allclose(
        stage2_td["scale"],
        torch.exp(stage2.bc_dagger_sac_adapter.log_std).expand_as(
            stage2_td["scale"]
        ),
    )


def test_stage2_stochastic_rollout_sample_is_exact_online_replay_action():
    actor_latent = torch.tensor([[0.2, -0.3]])
    action_clip = 1.0

    class _MeanDistActor(torch.nn.Module):
        def get_dist(self, td):
            return SimpleNamespace(mean=actor_latent.expand(*td.batch_size, -1))

    stage2 = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(stage2)
    stage2.cfg = SimpleNamespace(
        train_every=1,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_bc_action_clip=action_clip,
        sac_bc_initial_action_std=0.1,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
    )
    stage2.actor_adapt = _MeanDistActor()
    stage2.bc_dagger_actor_anchor = copy.deepcopy(
        stage2.actor_adapt
    ).requires_grad_(False)
    stage2.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=2,
        initial_log_std=math.log(stage2.cfg.sac_bc_initial_action_std),
        device="cpu",
    )
    low = torch.full((2,), -action_clip)
    high = torch.full_like(low, action_clip)
    stage2._fastsac_action_low = low.tolist()
    stage2._fastsac_action_high = high.tolist()
    stage2._fastsac_q_action_center = (low + high) * 0.5
    stage2._fastsac_q_action_scale = (high - low) * 0.5
    stage2.dist_cls = functools.partial(
        FastSACTanhNormal,
        low=low,
        high=high,
        event_dims=1,
    )
    stage2.sac_action_rng = torch.Generator().manual_seed(41)
    stage2.sac_rollout_rng = torch.Generator().manual_seed(42)

    actor_td = TensorDict({}, batch_size=[1])
    expected_mean, expected_dist = stage2._bc_dagger_behavior_action_and_dist(
        actor_td
    )
    expected_action, _ = expected_dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(42)
    )
    global_rng_before = torch.random.get_rng_state().clone()
    learning_rng_before = stage2.sac_action_rng.get_state().clone()

    _BCDaggerSACRolloutActor(stage2, deterministic=False)(actor_td)

    assert torch.equal(actor_td[ACTION_KEY], expected_action)
    assert not torch.equal(actor_td[ACTION_KEY], expected_mean)
    assert ((actor_td[ACTION_KEY] > low) & (actor_td[ACTION_KEY] < high)).all()
    expected_deviation = (expected_action - expected_mean).abs()
    assert torch.equal(
        actor_td[STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY],
        expected_deviation.mean(dim=-1),
    )
    assert torch.equal(
        actor_td[STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY],
        expected_deviation.amax(dim=-1),
    )
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert torch.equal(stage2.sac_action_rng.get_state(), learning_rng_before)

    rollout_rng_before_eval = stage2.sac_rollout_rng.get_state().clone()
    eval_td = TensorDict({}, batch_size=[1])
    _BCDaggerSACRolloutActor(stage2, deterministic=True)(eval_td)
    assert torch.equal(eval_td[ACTION_KEY], expected_mean)
    assert torch.equal(
        stage2.sac_rollout_rng.get_state(), rollout_rng_before_eval
    )

    stage2.q_actor_keys = ["actor"]
    stage2.q_critic_keys = ["critic"]
    stage2._q_actor_dim = 1
    stage2._q_critic_dim = 1
    stage2.action_dim = 2
    stage2._rollout_final_batch = {
        "next_observations": torch.tensor([[1.0]]),
        "next_critic_observations": torch.tensor([[2.0]]),
    }
    stage2._truncation_final_batches = []
    stage2._rollout_q_actuator_contexts = []
    rollout = TensorDict(
        {
            "actor": torch.zeros(1, 1, 1),
            "critic": torch.zeros(1, 1, 1),
            ACTION_KEY: actor_td[ACTION_KEY].unsqueeze(1),
            "step_count": torch.full((1, 1, 1), 6),
            "next": TensorDict(
                {
                    "reward": torch.zeros(1, 1, 1),
                    "done": torch.zeros(1, 1, 1, dtype=torch.bool),
                    "terminated": torch.zeros(1, 1, 1, dtype=torch.bool),
                    "discount": torch.ones(1, 1, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(
                                1, 1, 1, dtype=torch.bool
                            ),
                            "command_finished": torch.zeros(
                                1, 1, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[1, 1],
                    ),
                },
                batch_size=[1, 1],
            ),
        },
        batch_size=[1, 1],
    )

    transition = next(stage2._student_transition_chunks(rollout))
    replay = OnlineReplay(capacity=4, device="cpu")
    replay.extend(transition)

    assert torch.equal(transition["actions"], expected_action)
    assert torch.equal(replay.data["actions"][:1], expected_action)
