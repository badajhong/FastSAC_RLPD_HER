import hashlib
import json
from pathlib import Path

import h5py
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts import bc_dagger
from scripts.train import run_training as shared_run_training


_EXECUTION_PAYLOAD = {
    "semantics": "bounded_policy_and_replay_command_support_v1",
    "source": "scalar_dagger_safety_envelope",
    "joint_names": ["joint_a", "joint_b"],
    "action_low": [-20.0, -20.0],
    "action_high": [20.0, 20.0],
}
_Q_TRANSFORM_PAYLOAD = {
    "semantics": "affine_nominal_joint_coordinates_unclipped_then_fixed_gain_v3",
    "source": "soft_joint_limits_at_default_pose",
    "joint_names": ["joint_a", "joint_b"],
    "action_center": [0.0, 1.0],
    "action_scale": [2.0, 2.0],
    "clamp": None,
}
_ENTROPY_PAYLOAD = {
    "semantics": "nominal_joint_action_density_coordinates_v1",
    "source": "nominal_joint_action_coordinates",
    "joint_names": ["joint_a", "joint_b"],
    "action_scale": [2.0, 2.0],
}


def _sha256(payload):
    return "sha256:" + hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


_ACTION_CONTRACT_PAYLOAD = {
    "semantics": bc_dagger.EXPECTED_ACTION_CONTRACT_SEMANTICS,
    "action_bound_source": "scalar_dagger_safety_envelope",
    "execution_support_semantics": _EXECUTION_PAYLOAD["semantics"],
    "execution_support_fingerprint": _sha256(_EXECUTION_PAYLOAD),
    "joint_names": ["joint_a", "joint_b"],
    "action_low": [-20.0, -20.0],
    "action_high": [20.0, 20.0],
    "action_center": [0.0, 0.0],
    "action_scale": [20.0, 20.0],
    "q_action_coordinate_source": "soft_joint_limits_at_default_pose",
    "nominal_action_low": [-2.0, -1.0],
    "nominal_action_high": [2.0, 3.0],
    "q_action_center": [0.0, 1.0],
    "q_action_scale": [2.0, 2.0],
    "q_action_clamp": None,
    "q_action_transform_semantics": _Q_TRANSFORM_PAYLOAD["semantics"],
    "q_action_transform_fingerprint": _sha256(_Q_TRANSFORM_PAYLOAD),
    "entropy_reference_source": "nominal_joint_action_coordinates",
    "entropy_reference_scale": [2.0, 2.0],
    "entropy_reference_semantics": _ENTROPY_PAYLOAD["semantics"],
    "entropy_reference_fingerprint": _sha256(_ENTROPY_PAYLOAD),
    "joint_offset_low": [0.0, 0.0],
    "joint_offset_high": [0.0, 0.0],
}
ACTION_CONTRACT = {
    **_ACTION_CONTRACT_PAYLOAD,
    "fingerprint": _sha256(_ACTION_CONTRACT_PAYLOAD),
}

REPLAY_LINEAGE = {
    "format": bc_dagger.EXPECTED_REPLAY_FORMAT,
    "format_version": bc_dagger.EXPECTED_REPLAY_FORMAT_VERSION,
    "replay_id": "frozen-replay-id",
    "dagger_control_semantics": bc_dagger.EXPECTED_CONTROL_SEMANTICS,
    "replay_observation_semantics": "raw_pre_vecnorm_sample_current_v1",
    "vecnorm_fingerprint": "sha256:" + "a" * 64,
    "actor_backend": bc_dagger.EXPECTED_ACTOR_BACKEND,
    "action_parameterization": bc_dagger.EXPECTED_ACTION_PARAMETERIZATION,
    "action_clip": 20.0,
    "action_contract": ACTION_CONTRACT,
}


def _cfg(
    *,
    name="ppo_bc_dagger",
    target=(
        "active_adaptation.learning.ppo.ppo_bc_dagger."
        "PPOBCDaggerFinetune"
    ),
    phase="finetune",
    vecnorm="eval",
    checkpoint="run:x/y/z",
    total_frames=39_321_600,
    iterations=None,
    num_envs=512,
    train_every=32,
    beta_start=1.0,
    beta_end=0.0,
    beta_decay=1800,
    beta_zero_iteration=None,
    control_mode="safe",
    safe_takeover=0.05,
    safe_release=0.03,
    safe_hold=8,
    safe_zero_iteration=None,
    raw_replay=True,
    bc_dagger_checkpoint=None,
    copy_teacher_replay=False,
    teacher_replay_buffer_path=None,
):
    return OmegaConf.create({
        "algo": {
            "name": name,
            "_target_": target,
            "phase": phase,
            "vecnorm": vecnorm,
            "train_every": train_every,
            "dagger_beta_start": beta_start,
            "dagger_beta_end": beta_end,
            "dagger_beta_decay_rollouts": beta_decay,
            "dagger_beta_zero_iteration": beta_zero_iteration,
            "dagger_control_mode": control_mode,
            "dagger_safe_takeover_rms": safe_takeover,
            "dagger_safe_release_rms": safe_release,
            "dagger_safe_min_teacher_steps": safe_hold,
            "dagger_safe_zero_iteration": safe_zero_iteration,
            "dagger_replay_raw_observations": raw_replay,
            "save_teacher_buffer": True,
            "teacher_buffer_path": None,
        },
        "task": {"num_envs": num_envs},
        "total_frames": total_frames,
        "bc_dagger_iterations": iterations,
        "checkpoint_path": checkpoint,
        "bc_dagger_checkpoint": bc_dagger_checkpoint,
        "bc_dagger_copy_teacher_replay": copy_teacher_replay,
        "teacher_replay_buffer_path": teacher_replay_buffer_path,
        # Legacy-focused fixtures opt out explicitly. Hydra's real dedicated
        # config defaults this to true and is covered separately below.
        "bc_dagger_inline_finalization": False,
        "perception_consolidation_iterations": 0,
        "actor_realignment_iterations": 0,
        "replay_q_calibration_iterations": 1,
        "calibration_teacher_probability": 0.5,
    })


def _inline_cfg(
    *,
    joint_iterations=10,
    perception_iterations=2,
    actor_iterations=3,
    calibration_iterations=4,
    teacher_probability=0.5,
    enabled=True,
    **overrides,
):
    """Small inline-tail schedule without changing legacy resume fixtures."""
    cfg = _cfg(
        total_frames=1,
        iterations=joint_iterations,
        control_mode="beta",
        beta_decay=joint_iterations,
        beta_zero_iteration=joint_iterations,
        **overrides,
    )
    with open_dict(cfg):
        cfg.bc_dagger_inline_finalization = enabled
        cfg.perception_consolidation_iterations = perception_iterations
        cfg.actor_realignment_iterations = actor_iterations
        cfg.replay_q_calibration_iterations = calibration_iterations
        cfg.calibration_teacher_probability = teacher_probability
    return cfg


def _write_resume_checkpoint(
    path: Path,
    *,
    rollout_count: int = 801,
    environment_steps: int = 25_632,
    algorithm: str = bc_dagger.EXPECTED_TRAINING_ALGORITHM,
    missing_optimizer: str | None = None,
    missing_module: str | None = None,
    missing_state: str | None = None,
    include_vecnorm: bool = True,
):
    optimizer_state = {
        name: {"state": {}, "param_groups": []}
        for name in bc_dagger.REQUIRED_RESUME_OPTIMIZERS
        if name != missing_optimizer
    }
    policy = {
        "training_algorithm": algorithm,
        "actor_backend": bc_dagger.EXPECTED_ACTOR_BACKEND,
        "critic_learning_semantics": bc_dagger.EXPECTED_CRITIC_SEMANTICS,
        "dagger_control_semantics": bc_dagger.EXPECTED_CONTROL_SEMANTICS,
        "action_contract": ACTION_CONTRACT,
        "dagger_backend_config": {
            "dagger_action_clip": 20.0,
            "fresh_ppo_actor_initialization_semantics": (
                bc_dagger.EXPECTED_FRESH_ACTOR_INITIALIZATION_SEMANTICS
            ),
        },
        "q_backend_config": {
            "q_action_transform_fingerprint": ACTION_CONTRACT[
                "q_action_transform_fingerprint"
            ]
        },
        "actor_adapt": {},
        "bc_dagger_sac_adapter": {},
        "qnet": {},
        "qnet_target": {},
        "optimizer_resume_state": optimizer_state,
        "dagger_rollout_count": rollout_count,
        "dagger_environment_steps": environment_steps,
        "bc_update_count": 25_632,
        "q_update_count": 25_632,
        "dagger_rng_state": torch.Generator().get_state(),
        "q_rng_state": torch.Generator().get_state(),
        "sac_action_rng_state": torch.Generator().get_state(),
        "next_iter": 6_801,
        "teacher_replay_state": dict(REPLAY_LINEAGE),
    }
    if missing_module is not None:
        policy.pop(missing_module)
    if missing_state is not None:
        policy.pop(missing_state)
    checkpoint = {"policy": policy}
    if include_vecnorm:
        checkpoint["vecnorm"] = {}
    torch.save(checkpoint, path)
    with h5py.File(path.with_name("teacher_replay_buffer.h5"), "w") as replay:
        for key, value in REPLAY_LINEAGE.items():
            replay.attrs[key] = (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if key == "action_contract"
                else value
            )
    return path


def _write_fresh_ppo_checkpoint(path: Path, *, task_name="G1Skateboard"):
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
    source_cfg = OmegaConf.create(
        {
            "task": {"name": task_name},
            "algo": {
                "name": "ppo_vel",
                "_target_": "active_adaptation.learning.ppo.ppo_vel.PPOVEL",
                "phase": "train",
                "enable_residual_distillation": True,
                "use_object_adapt": False,
            },
        }
    )
    torch.save(
        {"policy": policy, "vecnorm": {}, "cfg": source_cfg}, path
    )
    return path


def test_bc_dagger_config_inherits_train_and_selects_dedicated_defaults():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="bc_dagger",
            overrides=["task=G1/vaic/skateboard_stu"],
        )

    assert cfg.algo.name == "ppo_bc_dagger"
    assert cfg.algo.phase == "finetune"
    assert cfg.algo.vecnorm == "eval"
    assert cfg.wandb.project == "vaic_dagger"
    assert cfg.task.enable_cameras is True
    assert cfg.total_frames == 39_321_600
    assert cfg.bc_dagger_iterations is None
    assert cfg.save_interval == 100
    assert cfg.bc_dagger_checkpoint is None
    assert cfg.bc_dagger_copy_teacher_replay is True
    assert cfg.algo.dagger_beta_start == pytest.approx(1.0)
    assert cfg.algo.dagger_beta_end == pytest.approx(0.0)
    assert cfg.algo.dagger_beta_decay_rollouts == 1800
    assert cfg.algo.dagger_beta_zero_iteration is None
    assert cfg.algo.dagger_control_mode == "safe"
    assert cfg.algo.dagger_safe_takeover_rms == pytest.approx(0.006)
    assert cfg.algo.dagger_safe_release_rms == pytest.approx(0.004)
    assert cfg.algo.dagger_action_clip == pytest.approx(20.0)
    assert cfg.algo.dagger_safe_min_teacher_steps == 8
    assert cfg.algo.dagger_safe_zero_iteration is None
    assert cfg.algo.dagger_replay_raw_observations is True
    assert cfg.algo.teacher_buffer_capacity == 1_048_576

    # The normal entrypoint owns its terminal cleanup now; the iteration
    # values remain user-facing so a short ablation can disable an optional
    # phase without switching to another script.
    assert cfg.bc_dagger_inline_finalization is True
    assert cfg.perception_consolidation_iterations == 0
    assert cfg.actor_realignment_iterations == 50
    assert cfg.replay_q_calibration_iterations == 128
    assert 0.0 < cfg.calibration_teacher_probability < 1.0

    bc_dagger.apply_inline_finalization_controls(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    tail_rollouts = (
        int(cfg.perception_consolidation_iterations)
        + int(cfg.actor_realignment_iterations)
        + int(cfg.replay_q_calibration_iterations)
    )
    assert schedule["frames_per_rollout"] == 16_384
    assert schedule["joint_rollouts"] == 2400
    assert schedule["tail_rollouts"] == tail_rollouts
    assert schedule["total_rollouts"] == 2400 + tail_rollouts
    assert schedule["start_rollout"] == 0
    assert schedule["end_rollout"] == 2400 + tail_rollouts
    assert schedule["decay_rollouts"] == 1800
    assert schedule["beta_zero_rollouts"] >= 600
    assert schedule["safe_zero_rollouts"] == 0
    assert cfg.total_frames == (2400 + tail_rollouts) * 16_384


def test_inline_finalization_maps_simple_surface_to_existing_staging_backend():
    cfg = _inline_cfg()

    controls = bc_dagger.apply_inline_finalization_controls(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)

    assert controls == {
        "joint_iterations": 10,
        "perception_consolidation_iterations": 2,
        "actor_realignment_iterations": 3,
        "replay_q_calibration_iterations": 4,
        "calibration_teacher_probability": pytest.approx(0.5),
    }
    assert cfg.algo.dagger_staging_enabled is True
    assert cfg.algo.dagger_stage_joint_warmup_iterations == 10
    assert cfg.algo.dagger_stage_cycles == 0
    assert cfg.algo.dagger_stage_perception_iterations == 0
    assert cfg.algo.dagger_stage_actor_iterations == 0
    assert cfg.algo.dagger_stage_final_perception_iterations == 2
    assert cfg.algo.dagger_stage_final_actor_iterations == 3
    assert cfg.algo.dagger_stage_calibration_iterations == 4
    assert cfg.algo.dagger_stage_calibration_control_mode == "beta"
    assert cfg.algo.dagger_stage_calibration_teacher_probability == (
        pytest.approx(0.5)
    )
    assert cfg.algo.dagger_stage_h5_final_only is True
    assert schedule["joint_rollouts"] == 10
    assert schedule["tail_rollouts"] == 9
    assert schedule["total_rollouts"] == 19
    assert schedule["start_rollout"] == 0
    assert schedule["end_rollout"] == 19
    assert cfg.total_frames == 19 * 512 * 32


def test_disabling_inline_finalization_preserves_joint_only_legacy_schedule():
    cfg = _inline_cfg(enabled=False)

    controls = bc_dagger.apply_inline_finalization_controls(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)

    assert controls is None
    assert schedule["total_rollouts"] == 10
    assert schedule.get("joint_rollouts", 10) == 10
    assert schedule.get("tail_rollouts", 0) == 0
    assert cfg.total_frames == 10 * 512 * 32
    assert not bool(cfg.algo.get("dagger_staging_enabled", False))


@pytest.mark.parametrize(
    ("perception_iterations", "actor_iterations", "expected_total"),
    ((0, 0, 14), (2, 0, 16), (0, 3, 17)),
)
def test_inline_optional_supervised_tail_phases_may_be_skipped(
    perception_iterations, actor_iterations, expected_total
):
    cfg = _inline_cfg(
        perception_iterations=perception_iterations,
        actor_iterations=actor_iterations,
    )

    bc_dagger.apply_inline_finalization_controls(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)

    assert cfg.algo.dagger_stage_final_perception_iterations == (
        perception_iterations
    )
    assert cfg.algo.dagger_stage_final_actor_iterations == actor_iterations
    assert schedule["total_rollouts"] == expected_total


@pytest.mark.parametrize("explicit_joint_iterations", (False, True))
def test_inline_control_application_and_schedule_are_idempotent(
    explicit_joint_iterations,
):
    cfg = _inline_cfg(joint_iterations=10)
    if not explicit_joint_iterations:
        with open_dict(cfg):
            cfg.bc_dagger_iterations = None
            cfg.total_frames = 10 * 512 * 32

    first = bc_dagger.apply_inline_finalization_controls(cfg)
    first_total = int(cfg.total_frames)
    second = bc_dagger.apply_inline_finalization_controls(cfg)
    bc_dagger.apply_bc_dagger_iteration_controls(cfg)
    schedule_a = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)
    schedule_b = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)

    assert first == second
    assert first_total == 19 * 512 * 32
    assert cfg.total_frames == first_total
    assert schedule_a == schedule_b
    assert schedule_a["joint_rollouts"] == 10
    assert schedule_a["tail_rollouts"] == 9
    assert schedule_a["total_rollouts"] == 19


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("perception_consolidation_iterations", -1, "non-negative"),
        ("perception_consolidation_iterations", True, "non-negative"),
        ("actor_realignment_iterations", -1, "non-negative"),
        ("replay_q_calibration_iterations", 0, "positive"),
        ("replay_q_calibration_iterations", True, "positive"),
        ("calibration_teacher_probability", 0.0, "strictly between"),
        ("calibration_teacher_probability", 1.0, "strictly between"),
    ),
)
def test_inline_finalization_rejects_invalid_controls_before_training(
    field, value, message
):
    cfg = _inline_cfg()
    with open_dict(cfg):
        cfg[field] = value

    with pytest.raises(ValueError, match=message):
        bc_dagger.apply_inline_finalization_controls(cfg)


def test_inline_finalization_rejects_same_stage_resume_before_h5_lookup():
    cfg = _inline_cfg(
        checkpoint=None,
        bc_dagger_checkpoint="/does/not/need/to/exist.pt",
    )

    with pytest.raises(ValueError, match="inline finalization|resume"):
        bc_dagger.apply_inline_finalization_controls(cfg)


def test_inline_source_accepts_fresh_ppo_and_canonicalizes_runtime_flags(
    tmp_path,
):
    checkpoint = _write_fresh_ppo_checkpoint(
        tmp_path / "checkpoint_6000.pt"
    )
    cfg = _inline_cfg(checkpoint=str(checkpoint))

    prepared = bc_dagger.prepare_inline_bc_dagger_source(cfg)

    assert prepared == {
        "path": str(checkpoint.resolve()),
        "source_last_iter": 6000,
    }
    assert cfg.checkpoint_path == str(checkpoint.resolve())
    assert cfg._bc_dagger_fresh_source is True
    assert cfg._bc_dagger_staging_source is True
    assert cfg._bc_dagger_model_only_resume is False


def test_inline_source_rejects_bc_checkpoint_through_generic_path(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_final.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh = _write_fresh_ppo_checkpoint(tmp_path / "fresh.pt")
    fresh_saved = torch.load(fresh, map_location="cpu", weights_only=False)
    saved["cfg"] = fresh_saved["cfg"]
    torch.save(saved, checkpoint)
    cfg = _inline_cfg(checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="fresh PPO teacher|resume"):
        bc_dagger.prepare_inline_bc_dagger_source(cfg)


def test_noninline_fresh_source_is_validated_and_marked(tmp_path):
    checkpoint = _write_fresh_ppo_checkpoint(tmp_path / "fresh_noninline.pt")
    cfg = _cfg(checkpoint=str(checkpoint))

    prepared = bc_dagger.prepare_fresh_bc_dagger_source(cfg)

    assert prepared["path"] == str(checkpoint.resolve())
    assert cfg._bc_dagger_fresh_source is True
    assert not bool(cfg.get("_bc_dagger_staging_source", False))


def test_noninline_fresh_source_rejects_explicit_old_replay(tmp_path):
    checkpoint = _write_fresh_ppo_checkpoint(tmp_path / "fresh_with_h5.pt")
    cfg = _cfg(
        checkpoint=str(checkpoint),
        teacher_replay_buffer_path="/tmp/old_teacher_replay.h5",
    )

    with pytest.raises(ValueError, match="fresh.*replay|remove"):
        bc_dagger.prepare_fresh_bc_dagger_source(cfg)


def test_hydra_accepts_inline_tail_overrides_without_plus_prefix():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "bc_dagger_iterations=10",
                "perception_consolidation_iterations=2",
                "actor_realignment_iterations=3",
                "replay_q_calibration_iterations=4",
                "calibration_teacher_probability=0.25",
            ],
        )

    controls = bc_dagger.apply_inline_finalization_controls(cfg)

    assert controls["joint_iterations"] == 10
    assert controls["perception_consolidation_iterations"] == 2
    assert controls["actor_realignment_iterations"] == 3
    assert controls["replay_q_calibration_iterations"] == 4
    assert controls["calibration_teacher_probability"] == pytest.approx(0.25)
    assert cfg.total_frames == 19 * 512 * 32


def test_hydra_accepts_iteration_and_teacher_zero_overrides_without_plus_prefix():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "bc_dagger_inline_finalization=false",
                "bc_dagger_iterations=1200",
                "algo.dagger_control_mode=hybrid",
                "algo.dagger_beta_zero_iteration=900",
                "algo.dagger_safe_zero_iteration=1000",
            ],
        )

    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)

    assert cfg.total_frames == 19_660_800
    assert cfg.algo.dagger_beta_decay_rollouts == 900
    assert schedule["total_rollouts"] == 1200
    assert schedule["beta_zero_rollouts"] == 300
    assert schedule["safe_zero_rollouts"] == 200


def test_resume_checkpoint_iterations_are_additional(
    tmp_path, capsys
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    frozen_h5 = tmp_path / "teacher_replay_buffer.h5"
    before = frozen_h5.read_bytes()
    cfg = _cfg(
        checkpoint="/old/ppo/checkpoint_6000.pt",
        bc_dagger_checkpoint=str(checkpoint),
        total_frames=1,
        iterations=399,
        beta_zero_iteration=1000,
        safe_zero_iteration=1000,
    )

    resume = bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)

    assert resume == {
        "path": str(checkpoint.resolve()),
        "rollout_count": 801,
        "environment_steps": 25_632,
        "teacher_replay_source": str(frozen_h5.resolve()),
    }
    assert cfg.checkpoint_path == str(checkpoint.resolve())
    assert cfg.bc_dagger_checkpoint == str(checkpoint.resolve())
    assert cfg.bc_dagger_resume_rollout_count == 801
    assert cfg.bc_dagger_resume_environment_steps == 25_632
    assert cfg._bc_dagger_model_only_resume is True
    assert cfg._bc_dagger_teacher_replay_copy_source == str(
        frozen_h5.resolve()
    )
    assert cfg._bc_dagger_teacher_replay_copy_path is None
    assert cfg.algo.save_teacher_buffer is False
    assert cfg.algo.teacher_buffer_path is None
    assert cfg.teacher_replay_buffer_path is None
    assert cfg.total_frames == 399 * 16_384
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 399,
        "start_rollout": 801,
        "end_rollout": 1200,
        "decay_rollouts": 1000,
        "beta_zero_rollouts": 200,
        "safe_zero_rollouts": 200,
    }
    # The checkpoint alias supersedes the original PPO bootstrap source, but
    # never discovers, opens, truncates, or snapshots the adjacent H5.
    assert "overrides the fresh PPO checkpoint_path" in capsys.readouterr().out
    assert frozen_h5.read_bytes() == before


def test_resume_run_alias_opt_out_resolves_model_without_requesting_replay(
    monkeypatch, tmp_path
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_final.pt")
    calls = []

    def resolve(path, **kwargs):
        calls.append((path, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(bc_dagger, "parse_checkpoint_path", resolve)
    cfg = _cfg(
        checkpoint=None,
        bc_dagger_checkpoint="run:entity/project/run-id",
    )

    bc_dagger.prepare_bc_dagger_checkpoint(cfg)

    assert calls == [
        (
            "run:entity/project/run-id",
            {
                "download_replay": False,
                "replay_filename": "teacher_replay_buffer.h5",
            },
        )
    ]
    assert cfg.algo.save_teacher_buffer is False


@pytest.mark.parametrize(
    "replay_location",
    ("teacher_replay_buffer_path", "algo.teacher_buffer_path"),
)
def test_resume_rejects_any_mutable_teacher_replay_path(
    tmp_path, replay_location
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    cfg = _cfg(checkpoint=None, bc_dagger_checkpoint=str(checkpoint))
    if replay_location == "teacher_replay_buffer_path":
        cfg.teacher_replay_buffer_path = str(tmp_path / "teacher.h5")
    else:
        cfg.algo.teacher_buffer_path = str(tmp_path / "teacher.h5")

    with pytest.raises(ValueError, match="paired immutable teacher replay"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    (
        ({"algorithm": "old-dagger"}, "SAC-critic-v6"),
        ({"include_vecnorm": False}, "VecNorm"),
        ({"missing_optimizer": "q_optimizer"}, "optimizer state"),
        ({"missing_module": "bc_dagger_sac_adapter"}, "trained modules"),
        ({"missing_state": "q_rng_state"}, "continuation state"),
    ),
)
def test_resume_rejects_incomplete_learning_state(
    tmp_path, checkpoint_kwargs, message
):
    checkpoint = _write_resume_checkpoint(
        tmp_path / "invalid.pt", **checkpoint_kwargs
    )
    cfg = _cfg(checkpoint=None, bc_dagger_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match=message):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


def test_bc_dagger_entrypoint_reuses_shared_training_engine(monkeypatch):
    assert bc_dagger.run_training is shared_run_training
    cfg = _cfg()
    received = []
    monkeypatch.setattr(
        bc_dagger,
        "run_training",
        lambda actual: received.append(actual) or "shared-result",
    )
    monkeypatch.setattr(
        bc_dagger,
        "prepare_fresh_bc_dagger_source",
        lambda actual: {"path": "/fresh/ppo.pt"},
    )

    result = bc_dagger.main.__wrapped__(cfg)

    assert result == "shared-result"
    assert received == [cfg]


def test_bc_dagger_main_installs_inline_tail_before_shared_training(monkeypatch):
    cfg = _inline_cfg()
    received = []
    sources = []
    monkeypatch.setattr(
        bc_dagger,
        "prepare_inline_bc_dagger_source",
        lambda actual: sources.append(actual) or {"path": "/fresh/ppo.pt"},
    )
    monkeypatch.setattr(
        bc_dagger,
        "run_training",
        lambda actual: received.append(actual) or "inline-result",
    )

    result = bc_dagger.main.__wrapped__(cfg)

    assert result == "inline-result"
    assert sources == [cfg]
    assert received == [cfg]
    assert cfg.algo.dagger_staging_enabled is True
    assert cfg.algo.dagger_stage_joint_warmup_iterations == 10
    assert cfg.algo.dagger_stage_final_perception_iterations == 2
    assert cfg.algo.dagger_stage_final_actor_iterations == 3
    assert cfg.algo.dagger_stage_calibration_iterations == 4
    assert cfg.algo.dagger_stage_h5_final_only is True
    assert cfg.total_frames == 19 * 512 * 32


def test_iteration_controls_override_frames_and_name_beta_zero_index():
    cfg = _cfg(
        total_frames=1,
        iterations=1200,
        control_mode="hybrid",
        beta_decay=1800,
        beta_zero_iteration=900,
        safe_zero_iteration=1000,
    )

    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)

    assert cfg.total_frames == 1200 * 512 * 32
    assert cfg.algo.dagger_beta_decay_rollouts == 900
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 1200,
        "start_rollout": 0,
        "end_rollout": 1200,
        "decay_rollouts": 900,
        "beta_zero_rollouts": 300,
        "safe_zero_rollouts": 200,
    }


def test_iteration_budget_matches_shared_trainer_on_multiple_ranks(monkeypatch):
    monkeypatch.setattr(bc_dagger.aa, "get_world_size", lambda: 2)
    cfg = _cfg(
        total_frames=1,
        iterations=10,
        num_envs=4,
        train_every=3,
    )

    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)

    assert cfg.total_frames == 10 * 4 * 3 * 2
    assert schedule["frames_per_rollout"] == 12
    assert schedule["total_rollouts"] == 10


@pytest.mark.parametrize(
    ("cfg", "message"),
    (
        (_cfg(name="ppo_vel"), "only supports"),
        (_cfg(target="wrong.Policy"), "requires the PPO-BC DAgger implementation"),
        (_cfg(phase="train"), "phase=finetune"),
        (_cfg(vecnorm="train"), "vecnorm=eval"),
        (
            _cfg(raw_replay=False),
            "dagger_replay_raw_observations=true",
        ),
        (_cfg(checkpoint=None), "requires checkpoint_path"),
        (
            _cfg(control_mode="beta", total_frames=16_384 * 1800),
            "cumulative end rollout",
        ),
        (_cfg(control_mode="unknown"), "control_mode"),
        (_cfg(safe_release=0.06), "release_rms"),
        (_cfg(safe_hold=0), "positive integer"),
        (_cfg(iterations=0), "bc_dagger_iterations"),
        (_cfg(iterations=True), "bc_dagger_iterations"),
        (_cfg(iterations=1.5), "bc_dagger_iterations"),
        (_cfg(iterations="12"), "bc_dagger_iterations"),
        (_cfg(beta_zero_iteration=0), "dagger_beta_zero_iteration"),
        (_cfg(beta_zero_iteration=True), "dagger_beta_zero_iteration"),
        (_cfg(beta_zero_iteration=1.5), "dagger_beta_zero_iteration"),
        (_cfg(beta_zero_iteration="12"), "dagger_beta_zero_iteration"),
        (_cfg(safe_zero_iteration=0), "dagger_safe_zero_iteration"),
        (_cfg(safe_zero_iteration=True), "dagger_safe_zero_iteration"),
        (_cfg(safe_zero_iteration=1.5), "dagger_safe_zero_iteration"),
        (_cfg(safe_zero_iteration="12"), "dagger_safe_zero_iteration"),
        (
            _cfg(control_mode="beta", safe_zero_iteration=900),
            "requires safe or hybrid",
        ),
        (
            _cfg(iterations=900, safe_zero_iteration=900),
            "cumulative end rollout",
        ),
        (
            _cfg(
                control_mode="hybrid",
                beta_end=0.1,
                beta_zero_iteration=900,
            ),
            "requires dagger_beta_end=0",
        ),
    ),
)
def test_bc_dagger_entrypoint_rejects_invalid_stage_before_training(cfg, message):
    with pytest.raises(ValueError, match=message):
        bc_dagger.validate_bc_dagger_config(cfg)


def test_safe_mode_does_not_require_a_beta_zero_training_phase():
    cfg = _cfg(control_mode="safe", total_frames=16_384 * 1800)

    bc_dagger.validate_bc_dagger_config(cfg)
