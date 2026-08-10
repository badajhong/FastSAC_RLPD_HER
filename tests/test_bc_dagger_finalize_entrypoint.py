from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import bc_dagger_finalize as finalize


def _runtime_cfg(checkpoint_path, **overrides):
    values = {
        "task": {"name": "G1Skateboard", "num_envs": 512},
        "algo": {
            "name": "ppo_bc_dagger",
            "_target_": finalize.EXPECTED_ALGO_TARGET,
            "phase": "finetune",
            "vecnorm": "eval",
            "train_every": 32,
        },
        "checkpoint_path": str(checkpoint_path),
        "teacher_replay_buffer_path": None,
        "vecnorm": "eval",
        "perception_consolidation_iterations": 25,
        "actor_realignment_iterations": 0,
        "perception_recheck_iterations": 0,
        "replay_q_calibration_iterations": 128,
        "calibration_control_mode": "beta",
        "calibration_teacher_probability": 0.5,
        "_bc_dagger_finalization_source": True,
        "_bc_dagger_finalize": True,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _write_source_checkpoint(path: Path):
    backend = {
        "dagger_control_mode": "beta",
        "dagger_beta_decay_rollouts": 500,
    }
    source_cfg = OmegaConf.create({
        "task": {
            "name": "G1Skateboard",
            "num_envs": 512,
        },
        "algo": {
            "name": "ppo_bc_dagger",
            "_target_": finalize.EXPECTED_ALGO_TARGET,
            "phase": "finetune",
            "vecnorm": "eval",
            "train_every": 32,
            "use_object_adapt": False,
            "use_depth": False,
            "dagger_control_mode": "beta",
            "dagger_beta_decay_rollouts": 500,
            "save_teacher_buffer": True,
            "teacher_buffer_filename": "teacher_replay_buffer.h5",
            "teacher_buffer_path": None,
        },
        "vecnorm": "eval",
    })
    optimizer_state = {
        name: {"state": {}, "param_groups": []}
        for name in finalize.REQUIRED_RESUME_OPTIMIZERS
    }
    policy = {
        "training_algorithm": finalize.EXPECTED_TRAINING_ALGORITHM,
        "critic_learning_semantics": finalize.EXPECTED_CRITIC_SEMANTICS,
        "dagger_control_semantics": finalize.EXPECTED_CONTROL_SEMANTICS,
        "dagger_backend_config": backend,
        "q_backend_config": {},
        "optimizer_resume_state": optimizer_state,
        "actor": {},
        "actor_adapt": {},
        "encoder_priv": {},
        "adapt_module": {},
        "adapt_ema": {},
        "bc_dagger_sac_adapter": {},
        "qnet": {},
        "qnet_target": {},
        "dagger_rollout_count": 3000,
        "dagger_environment_steps": 96_000,
        "bc_update_count": 1,
        "q_update_count": 1,
        "dagger_rng_state": torch.Generator().get_state(),
        "q_rng_state": torch.Generator().get_state(),
        "sac_action_rng_state": torch.Generator().get_state(),
        "next_iter": 9000,
    }
    torch.save(
        {"policy": policy, "vecnorm": {}, "cfg": source_cfg}, path
    )
    return path


def test_finalize_config_accepts_exact_cli_without_plus_prefix():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="bc_dagger_finalize",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/checkpoint_final.pt",
                "perception_consolidation_iterations=25",
                "actor_realignment_iterations=0",
                "perception_recheck_iterations=0",
                "replay_q_calibration_iterations=128",
                "calibration_control_mode=beta",
                "calibration_teacher_probability=0.5",
            ],
        )

    assert cfg.perception_consolidation_iterations == 25
    assert cfg.actor_realignment_iterations == 0
    assert cfg.perception_recheck_iterations == 0
    assert cfg.replay_q_calibration_iterations == 128
    assert cfg.calibration_control_mode == "beta"
    assert cfg.calibration_teacher_probability == pytest.approx(0.5)
    assert cfg._bc_dagger_finalization_source is True
    assert cfg._bc_dagger_finalize is True


def test_prepare_hydrates_source_backend_and_computes_exact_schedule(tmp_path):
    checkpoint = _write_source_checkpoint(tmp_path / "checkpoint_final.pt")
    cfg = _runtime_cfg(checkpoint)
    # Current/default runtime composition may disagree with the original run.
    cfg.algo.dagger_control_mode = "safe"
    cfg.algo.dagger_beta_decay_rollouts = 1800

    prepared = finalize.prepare_bc_dagger_finalization(cfg)
    finalize.validate_bc_dagger_finalize_config(cfg)

    assert prepared["path"] == str(checkpoint.resolve())
    assert prepared["source_rollout_count"] == 3000
    assert prepared["schedule"] == {
        "perception_start": 0,
        "perception_end": 25,
        "actor_start": 25,
        "actor_end": 25,
        "recheck_start": 25,
        "recheck_end": 25,
        "calibration_start": 25,
        "calibration_end": 153,
        "total_rollouts": 153,
        "frames_per_rollout": 16_384,
        "total_frames": 2_506_752,
    }
    assert cfg.total_frames == 2_506_752
    assert cfg.algo.dagger_control_mode == "beta"
    assert cfg.algo.dagger_beta_decay_rollouts == 500
    assert cfg.algo.dagger_finalization_enabled is True
    assert cfg.algo.dagger_finalize_perception_iterations == 25
    assert cfg.algo.dagger_finalize_actor_iterations == 0
    assert cfg.algo.dagger_finalize_recheck_iterations == 0
    assert cfg.algo.dagger_finalize_calibration_iterations == 128
    assert cfg.algo.dagger_finalize_calibration_control_mode == "beta"
    assert (
        cfg.algo.dagger_finalize_calibration_teacher_probability
        == pytest.approx(0.5)
    )
    assert cfg.algo.teacher_buffer_path is None
    assert cfg.teacher_replay_buffer_path is None
    # A fresh finalization source is model-only; its old paired H5 is neither
    # required nor created by entrypoint preparation.
    assert not (tmp_path / "teacher_replay_buffer.h5").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"perception_consolidation_iterations": -1}, "non-negative"),
        ({"actor_realignment_iterations": True}, "non-negative"),
        ({"replay_q_calibration_iterations": 0}, "positive"),
        ({"calibration_control_mode": "safe"}, "must be beta"),
        ({"calibration_teacher_probability": 0.0}, "strictly between"),
        ({"calibration_teacher_probability": 1.0}, "strictly between"),
    ),
)
def test_finalize_controls_fail_before_checkpoint_loading(override, message):
    cfg = _runtime_cfg("/does/not/matter.pt", **override)
    with pytest.raises(ValueError, match=message):
        finalize.validate_finalization_controls(cfg)


def test_finalize_main_reuses_shared_training_engine(tmp_path, monkeypatch):
    checkpoint = _write_source_checkpoint(tmp_path / "checkpoint_final.pt")
    cfg = _runtime_cfg(checkpoint)
    received = []
    monkeypatch.setattr(
        finalize,
        "run_training",
        lambda actual: received.append(actual) or "shared-result",
    )

    result = finalize.main.__wrapped__(cfg)

    assert result == "shared-result"
    assert received == [cfg]


def test_finalize_rejects_a_partial_finalization_checkpoint(tmp_path):
    checkpoint = _write_source_checkpoint(tmp_path / "checkpoint_final.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["policy"]["bc_dagger_finalization_state"] = {
        "rollout_count": 1,
    }
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="already a BC-DAgger finalization"):
        finalize.prepare_bc_dagger_finalization(_runtime_cfg(checkpoint))
