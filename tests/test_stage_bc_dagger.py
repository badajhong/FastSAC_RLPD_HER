from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from scripts import stage_bc_dagger as stage


bc_dagger_module = importlib.import_module(
    "active_adaptation.learning.ppo.ppo_bc_dagger"
)
from active_adaptation.learning.ppo.fastsac_vel import (  # noqa: E402
    TEACHER_REPLAY_FIELDS,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (  # noqa: E402
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    PPOBCDaggerFinetune,
    PPOBCDaggerFinetuneConfig,
    _DeviceReplay,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL  # noqa: E402


def _bare_policy(**cfg):
    policy = PPOBCDaggerFinetune.__new__(PPOBCDaggerFinetune)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    return policy


def _staging_policy(
    *,
    joint=500,
    cycles=7,
    perception=100,
    actor=200,
    final_perception=100,
    final_actor=172,
    calibration=128,
    teacher_probability=0.5,
):
    cfg = PPOBCDaggerFinetuneConfig()
    cfg.dagger_staging_enabled = True
    cfg.dagger_stage_joint_warmup_iterations = joint
    cfg.dagger_stage_cycles = cycles
    cfg.dagger_stage_perception_iterations = perception
    cfg.dagger_stage_actor_iterations = actor
    cfg.dagger_stage_final_perception_iterations = final_perception
    cfg.dagger_stage_final_actor_iterations = final_actor
    cfg.dagger_stage_calibration_iterations = calibration
    cfg.dagger_stage_calibration_control_mode = "beta"
    cfg.dagger_stage_calibration_teacher_probability = teacher_probability
    policy = _bare_policy()
    policy.cfg = cfg
    policy.staging_rollout_count = 0
    policy._staging_last_phase = None
    return policy


def _runtime_cfg(**overrides):
    values = {
        "task": {"name": "G1Skateboard", "num_envs": 512},
        "algo": {
            "name": "ppo_bc_dagger",
            "_target_": stage.EXPECTED_ALGO_TARGET,
            "phase": "finetune",
            "vecnorm": "eval",
            "train_every": 32,
            "dagger_control_mode": "beta",
            "dagger_beta_start": 1.0,
            "dagger_beta_end": 0.0,
            "dagger_beta_decay_rollouts": 500,
            "dagger_beta_zero_iteration": 500,
            "dagger_safe_takeover_rms": 0.006,
            "dagger_safe_release_rms": 0.004,
            "dagger_safe_min_teacher_steps": 8,
            "dagger_safe_zero_iteration": None,
            "dagger_replay_raw_observations": True,
            "use_object_adapt": False,
            "save_teacher_buffer": True,
            "teacher_buffer_path": None,
        },
        "checkpoint_path": "/tmp/fresh_ppo_teacher.pt",
        "bc_dagger_checkpoint": None,
        "teacher_replay_buffer_path": None,
        "bc_dagger_iterations": None,
        "joint_warmup_iterations": 500,
        "stage_cycles": 7,
        "perception_iterations_per_cycle": 100,
        "actor_iterations_per_cycle": 200,
        "final_perception_iterations": 100,
        "final_actor_iterations": 172,
        "replay_q_calibration_iterations": 128,
        "calibration_control_mode": "beta",
        "calibration_teacher_probability": 0.5,
        "_bc_dagger_staging_source": True,
        "_bc_dagger_stage": True,
        "vecnorm": "eval",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _write_fresh_ppo_teacher(path: Path):
    source_cfg = OmegaConf.create(
        {
            "task": {"name": "G1Skateboard", "num_envs": 512},
            "algo": {
                "name": stage.EXPECTED_PPO_ALGO_NAME,
                "_target_": stage.EXPECTED_PPO_ALGO_TARGET,
                "phase": "train",
                "use_object_adapt": False,
            },
        }
    )
    policy = {
        "last_phase": "train",
        "last_iter": 6000,
        "actor": {},
        "actor_adapt": {},
        "encoder_priv": {},
        "adapt_module": {},
        "adapt_ema": {},
        "critic": {},
    }
    torch.save(
        {"policy": policy, "vecnorm": {}, "cfg": source_cfg}, path
    )
    return path


def test_stage_hydra_surface_and_default_schedule_are_exactly_3000():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="stage_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/checkpoint_6000.pt",
                "bc_dagger_iterations=3000",
                "joint_warmup_iterations=500",
                "stage_cycles=7",
                "perception_iterations_per_cycle=100",
                "actor_iterations_per_cycle=200",
                "final_perception_iterations=100",
                "final_actor_iterations=172",
                "replay_q_calibration_iterations=128",
                "calibration_control_mode=beta",
                "calibration_teacher_probability=0.5",
            ],
        )

    controls = stage.validate_stage_controls(cfg)
    schedule = stage.stage_bc_dagger_rollout_schedule(cfg)

    assert controls["joint_warmup_iterations"] == 500
    assert controls["stage_cycles"] == 7
    assert controls["perception_iterations_per_cycle"] == 100
    assert controls["actor_iterations_per_cycle"] == 200
    assert controls["final_perception_iterations"] == 100
    assert controls["final_actor_iterations"] == 172
    assert controls["replay_q_calibration_iterations"] == 128
    assert controls["calibration_control_mode"] == "beta"
    assert controls["calibration_teacher_probability"] == pytest.approx(0.5)
    assert schedule["total_rollouts"] == 3000
    assert schedule["frames_per_rollout"] == 16_384
    assert schedule["total_frames"] == 49_152_000
    assert cfg.total_frames == 49_152_000


def test_stage_schedule_derives_budget_and_rejects_an_explicit_mismatch():
    cfg = _runtime_cfg()

    schedule = stage.stage_bc_dagger_rollout_schedule(cfg)

    assert schedule["total_rollouts"] == 3000
    assert cfg.bc_dagger_iterations == 3000
    assert cfg.total_frames == 49_152_000

    mismatched = _runtime_cfg(bc_dagger_iterations=2999)
    with pytest.raises(ValueError, match="3000|schedule|sum|match"):
        stage.stage_bc_dagger_rollout_schedule(mismatched)


def test_prepare_stage_accepts_only_fresh_ppo_and_installs_backend_controls(
    tmp_path,
):
    checkpoint = _write_fresh_ppo_teacher(tmp_path / "checkpoint_6000.pt")
    cfg = _runtime_cfg(checkpoint_path=str(checkpoint))

    prepared = stage.prepare_stage_bc_dagger(cfg)
    stage.validate_stage_bc_dagger_config(cfg)

    assert prepared == {
        "path": str(checkpoint.resolve()),
        "source_last_iter": 6000,
        "schedule": stage.stage_bc_dagger_rollout_schedule(cfg),
    }
    assert prepared["schedule"]["joint_end"] == 500
    assert prepared["schedule"]["cycles_end"] == 2600
    assert prepared["schedule"]["final_perception_end"] == 2700
    assert prepared["schedule"]["final_actor_end"] == 2872
    assert prepared["schedule"]["calibration_end"] == 3000
    assert cfg.checkpoint_path == str(checkpoint.resolve())
    assert cfg._bc_dagger_staging_source is True
    assert cfg._bc_dagger_stage is True
    assert cfg._bc_dagger_model_only_resume is False
    assert cfg.algo.dagger_staging_enabled is True
    assert cfg.algo.dagger_stage_joint_warmup_iterations == 500
    assert cfg.algo.dagger_stage_cycles == 7
    assert cfg.algo.dagger_stage_perception_iterations == 100
    assert cfg.algo.dagger_stage_actor_iterations == 200
    assert cfg.algo.dagger_stage_final_perception_iterations == 100
    assert cfg.algo.dagger_stage_final_actor_iterations == 172
    assert cfg.algo.dagger_stage_calibration_iterations == 128
    assert cfg.algo.dagger_stage_calibration_teacher_probability == (
        pytest.approx(0.5)
    )
    assert cfg.algo.save_teacher_buffer is True
    assert cfg.algo.teacher_buffer_path is None
    assert cfg.teacher_replay_buffer_path is None


@pytest.mark.parametrize("complete", (False, True))
def test_prepare_stage_rejects_incomplete_and_complete_staged_sources(
    tmp_path, complete
):
    checkpoint = _write_fresh_ppo_teacher(tmp_path / "staged.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["policy"]["training_algorithm"] = (
        bc_dagger_module.PPO_BC_DAGGER_TRAINING_ALGORITHM
    )
    state["policy"]["bc_dagger_staging_state"] = {
        "semantics": bc_dagger_module.DAGGER_STAGING_SEMANTICS,
        "complete": complete,
    }
    torch.save(state, checkpoint)

    cfg = _runtime_cfg(checkpoint_path=str(checkpoint))
    with pytest.raises(ValueError, match="fresh PPO|resume"):
        stage.prepare_stage_bc_dagger(cfg)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"joint_warmup_iterations": 0}, "positive"),
        ({"stage_cycles": 0}, "positive"),
        ({"perception_iterations_per_cycle": 0}, "positive"),
        ({"actor_iterations_per_cycle": True}, "positive"),
        ({"replay_q_calibration_iterations": 0}, "positive"),
        ({"calibration_control_mode": "safe"}, "beta"),
        ({"calibration_teacher_probability": 0.0}, "strictly between"),
        ({"calibration_teacher_probability": 1.0}, "strictly between"),
    ),
)
def test_stage_controls_reject_invalid_values_before_training(
    override, message
):
    cfg = _runtime_cfg(**override)
    with pytest.raises(ValueError, match=message):
        stage.validate_stage_controls(cfg)


def test_stage_entrypoint_rejects_same_stage_resume_alias():
    cfg = _runtime_cfg(bc_dagger_checkpoint="/tmp/staged_checkpoint.pt")

    with pytest.raises(ValueError, match="bc_dagger_checkpoint|fresh PPO"):
        stage.prepare_stage_bc_dagger(cfg)


def test_staging_config_dataclass_exposes_every_runtime_control():
    cfg = PPOBCDaggerFinetuneConfig()

    for name in (
        "dagger_staging_enabled",
        "dagger_stage_joint_warmup_iterations",
        "dagger_stage_cycles",
        "dagger_stage_perception_iterations",
        "dagger_stage_actor_iterations",
        "dagger_stage_final_perception_iterations",
        "dagger_stage_final_actor_iterations",
        "dagger_stage_calibration_iterations",
        "dagger_stage_calibration_control_mode",
        "dagger_stage_calibration_teacher_probability",
    ):
        assert hasattr(cfg, name), name


@pytest.mark.parametrize(
    ("rollout", "phase", "cycle"),
    (
        (0, "joint_warmup", -1),
        (499, "joint_warmup", -1),
        (500, "cycle_perception", 0),
        (599, "cycle_perception", 0),
        (600, "cycle_actor", 0),
        (799, "cycle_actor", 0),
        (800, "cycle_perception", 1),
        (2299, "cycle_actor", 5),
        (2300, "cycle_perception", 6),
        (2599, "cycle_actor", 6),
        (2600, "final_perception", -1),
        (2699, "final_perception", -1),
        (2700, "final_actor", -1),
        (2871, "final_actor", -1),
        (2872, "replay_q_calibration", -1),
        (2999, "replay_q_calibration", -1),
        (3000, "complete", -1),
    ),
)
def test_staging_phase_and_cycle_boundaries_are_exact(rollout, phase, cycle):
    policy = _staging_policy()

    assert policy._staging_phase(rollout) == phase
    assert policy._staging_cycle_index(rollout) == cycle


def test_staging_phase_dispatch_isolates_actor_q_and_perception_updates():
    policy = _staging_policy(
        joint=1,
        cycles=1,
        perception=1,
        actor=1,
        final_perception=1,
        final_actor=1,
        calibration=1,
    )
    policy.cfg.dagger_updates_per_rollout = 7
    policy.cfg.q_updates_per_rollout = 11
    adaptation_calls = []
    policy.train_adapt = (
        lambda rollout: adaptation_calls.append(rollout)
        or {"adapt/called": 1.0}
    )
    rollout = TensorDict({"marker": torch.ones(1, 1)}, batch_size=[1])

    expected = (
        ("joint_warmup", True, True, 7, 11, True),
        ("cycle_perception", False, False, 0, 0, True),
        ("cycle_actor", True, False, 7, 0, False),
        ("final_perception", False, False, 0, 0, True),
        ("final_actor", True, False, 7, 0, False),
        ("replay_q_calibration", True, True, 0, 11, False),
    )
    for count, (
        phase,
        collect_actor,
        collect_q_teacher,
        actor_updates,
        q_updates,
        adapt,
    ) in enumerate(expected):
        policy.staging_rollout_count = count
        assert policy._staging_phase() == phase
        assert policy._collect_dagger_replay_this_rollout() is collect_actor
        assert (
            policy._collect_q_teacher_rows_this_rollout()
            is collect_q_teacher
        )
        assert policy._actor_updates_this_rollout() == actor_updates
        assert policy._q_updates_this_rollout() == q_updates
        result = policy._adaptation_update_this_rollout(rollout)
        assert bool(result) is adapt

    assert len(adaptation_calls) == 3
    assert all(actual is not rollout for actual in adaptation_calls)


def test_staging_freeze_mask_exposes_only_phase_owned_parameters():
    policy = _staging_policy()
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

    perception = {
        "adapt_module",
        "object_adapt",
        "temporal_depth_gru",
    }
    owned = {
        "joint_warmup": perception | {"actor_adapt", "qnet"},
        "cycle_perception": perception,
        "cycle_actor": {"actor_adapt"},
        "final_perception": perception,
        "final_actor": {"actor_adapt"},
        "replay_q_calibration": {"qnet"},
        "complete": set(),
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
        policy._apply_staging_freeze_mask(phase)
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


class _TeacherExportRecorder:
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


class _TeacherSnapshotRecorder:
    def __init__(self):
        self.calls = []

    def snapshot(self, iteration, checkpoint_name):
        self.calls.append((iteration, checkpoint_name))
        return "snapshot-result"


def test_staging_teacher_h5_snapshot_is_disabled_until_final_calibration():
    policy = _staging_policy()
    policy.teacher_replay = _TeacherSnapshotRecorder()

    for count in (0, 499, 500, 2599, 2600, 2871):
        policy.staging_rollout_count = count
        assert policy.snapshot_teacher_replay(count, f"checkpoint_{count}") is None

    assert policy.teacher_replay.calls == []

    policy.staging_rollout_count = 2872
    assert policy.snapshot_teacher_replay(2872, "checkpoint_2872") == (
        "snapshot-result"
    )
    policy.staging_rollout_count = 3000
    assert policy.snapshot_teacher_replay(3000, "checkpoint_final") == (
        "snapshot-result"
    )
    assert policy.teacher_replay.calls == [
        (2872, "checkpoint_2872"),
        (3000, "checkpoint_final"),
    ]


def test_staging_h5_is_final_only_and_q_teacher_rows_skip_actor_phases():
    policy = _staging_policy(
        joint=1,
        cycles=1,
        perception=1,
        actor=1,
        final_perception=1,
        final_actor=1,
        calibration=1,
    )
    policy.cfg.train_every = 2
    policy.cfg.dagger_updates_per_rollout = 0
    policy.cfg.q_updates_per_rollout = 0
    policy.cfg.q_learning_starts_per_source = 8_192
    policy.device = torch.device("cpu")
    policy.adapt_module = nn.Linear(1, 1)
    policy.object_adapt = nn.Linear(1, 1)
    policy.temporal_depth_gru = nn.Linear(1, 1)
    policy.actor_adapt = nn.Linear(1, 1)
    policy.qnet = nn.Linear(1, 1)
    policy.qnet_target = nn.Linear(1, 1).requires_grad_(False)
    policy.dagger_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.q_teacher_replay = _DeviceReplay(capacity=8, device="cpu")
    policy.teacher_replay = _TeacherExportRecorder()
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.bc_update_count = 0
    policy.q_update_count = 0
    policy.num_updates = 0
    policy._last_truncation_finals_used = 0
    adapt_phases = []
    policy.train_adapt = (
        lambda td: adapt_phases.append(policy._staging_phase()) or {}
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

    # Joint warmup may build the in-memory Q teacher partition, but must never
    # write the final Stage-2 H5.
    policy.train_op(rollout)
    assert policy.q_teacher_replay.size == 1
    assert policy.teacher_replay.rows == []

    # Perception, actor, final-perception, and final-actor rollouts cannot add
    # rows to either final replay source.
    for _ in range(4):
        policy.train_op(rollout)
        assert policy.q_teacher_replay.size == 1
        assert policy.teacher_replay.rows == []

    policy.train_op(rollout)

    # Entering final calibration clears both learning rings. Its one valid
    # teacher-executed row is the first and only exported H5 row.
    assert policy.dagger_replay.seen == 3
    assert policy.q_teacher_replay.seen == 1
    assert len(policy.teacher_replay.rows) == 1
    assert set(policy.teacher_replay.rows[0]) == set(TEACHER_REPLAY_FIELDS)
    assert torch.equal(
        policy.teacher_replay.rows[0]["actions"], torch.tensor([[10.0]])
    )
    assert adapt_phases == [
        "joint_warmup",
        "cycle_perception",
        "final_perception",
    ]


@pytest.mark.parametrize(
    ("count", "phase", "complete"),
    (
        (2872, "replay_q_calibration", False),
        (3000, "complete", True),
    ),
)
def test_staging_checkpoint_metadata_marks_incomplete_and_complete(
    monkeypatch, count, phase, complete
):
    policy = _staging_policy()
    policy.staging_rollout_count = count
    policy._staging_last_phase = (
        "final_actor" if count == 2872 else "replay_q_calibration"
    )
    policy.dagger_rollout_count = count
    policy.dagger_environment_steps = count * 32
    policy.bc_update_count = 100
    policy.q_update_count = 101
    policy._staging_calibration_start_q_update_count = (
        100 if complete else None
    )
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.sac_action_rng = torch.Generator().manual_seed(3)
    policy.bc_optimizer = SimpleNamespace(state_dict=lambda: {"name": "bc"})
    policy.opt_q = SimpleNamespace(state_dict=lambda: {"name": "q"})
    policy.opt_adapt = SimpleNamespace(
        state_dict=lambda: {"name": "adapt"}
    )
    policy.teacher_replay_id = "fresh-staged-replay"
    policy.teacher_replay = None
    policy._loaded_teacher_replay_metadata = None
    policy._replay_vecnorm_fingerprint = "sha256:" + "d" * 64
    policy.env = SimpleNamespace(current_iter=77)
    policy._q_backend_metadata = lambda: {"contract": "unchanged-v3"}
    monkeypatch.setattr(PPOVEL, "state_dict", lambda self: {})

    state = policy.state_dict()

    assert state["training_algorithm"] == (
        bc_dagger_module.PPO_BC_DAGGER_TRAINING_ALGORITHM
    )
    staging = state["bc_dagger_staging_state"]
    assert staging["semantics"] == bc_dagger_module.DAGGER_STAGING_SEMANTICS
    assert staging["config"] == policy._staging_config()
    assert staging["rollout_count"] == count
    assert staging["phase"] == phase
    assert staging["last_phase"] == policy._staging_last_phase
    assert staging["complete"] is complete
    assert staging["fresh_replay_id"] == "fresh-staged-replay"
    assert staging["calibration_start_q_update_count"] == (
        100 if complete else None
    )
    assert staging["calibration_q_updates"] == (1 if complete else 0)
