from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import bc_dagger
from scripts.train import run_training as shared_run_training


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
    num_envs=512,
    train_every=32,
    beta_start=1.0,
    beta_end=0.0,
    beta_decay=1800,
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
            "dagger_replay_raw_observations": raw_replay,
            "save_teacher_buffer": True,
            "teacher_buffer_path": None,
        },
        "task": {"num_envs": num_envs},
        "total_frames": total_frames,
        "checkpoint_path": checkpoint,
        "bc_dagger_checkpoint": bc_dagger_checkpoint,
        "bc_dagger_copy_teacher_replay": copy_teacher_replay,
        "teacher_replay_buffer_path": teacher_replay_buffer_path,
    })


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
        "actor_adapt": {},
        "qnet": {},
        "qnet_target": {},
        "iql_value": {},
        "optimizer_resume_state": optimizer_state,
        "dagger_rollout_count": rollout_count,
        "dagger_environment_steps": environment_steps,
        "bc_update_count": 25_632,
        "q_update_count": 25_632,
        "iql_value_update_count": 25_632,
        "dagger_rng_state": torch.Generator().get_state(),
        "q_rng_state": torch.Generator().get_state(),
        "next_iter": 6_801,
    }
    if missing_module is not None:
        policy.pop(missing_module)
    if missing_state is not None:
        policy.pop(missing_state)
    checkpoint = {"policy": policy}
    if include_vecnorm:
        checkpoint["vecnorm"] = {}
    torch.save(checkpoint, path)
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
    assert cfg.save_interval == 100
    assert cfg.bc_dagger_checkpoint is None
    assert cfg.bc_dagger_copy_teacher_replay is True
    assert cfg.algo.dagger_beta_start == pytest.approx(1.0)
    assert cfg.algo.dagger_beta_end == pytest.approx(0.0)
    assert cfg.algo.dagger_beta_decay_rollouts == 1800
    assert cfg.algo.dagger_replay_raw_observations is True
    assert cfg.algo.teacher_buffer_capacity == 1_048_576

    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 2400,
        "start_rollout": 0,
        "end_rollout": 2400,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 600,
    }


def test_resume_checkpoint_is_model_only_and_total_frames_are_additional(
    tmp_path, capsys
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    frozen_h5 = tmp_path / "teacher_replay_buffer.h5"
    frozen_h5.write_bytes(b"must remain untouched")
    before = frozen_h5.read_bytes()
    cfg = _cfg(
        checkpoint="/old/ppo/checkpoint_6000.pt",
        bc_dagger_checkpoint=str(checkpoint),
        total_frames=399 * 16_384,
        beta_decay=1000,
    )

    resume = bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    bc_dagger.validate_bc_dagger_config(cfg)

    assert resume == {
        "path": str(checkpoint.resolve()),
        "rollout_count": 801,
        "environment_steps": 25_632,
        "teacher_replay_source": None,
    }
    assert cfg.checkpoint_path == str(checkpoint.resolve())
    assert cfg.bc_dagger_checkpoint == str(checkpoint.resolve())
    assert cfg.bc_dagger_resume_rollout_count == 801
    assert cfg.bc_dagger_resume_environment_steps == 25_632
    assert cfg._bc_dagger_model_only_resume is True
    assert cfg._bc_dagger_teacher_replay_copy_source is None
    assert cfg._bc_dagger_teacher_replay_copy_path is None
    assert cfg.algo.save_teacher_buffer is False
    assert cfg.algo.teacher_buffer_path is None
    assert cfg.teacher_replay_buffer_path is None
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 399,
        "start_rollout": 801,
        "end_rollout": 1200,
        "decay_rollouts": 1000,
        "beta_zero_rollouts": 200,
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

    with pytest.raises(ValueError, match="without teacher replay"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    (
        ({"algorithm": "old-dagger"}, "IQL-v2"),
        ({"include_vecnorm": False}, "VecNorm"),
        ({"missing_optimizer": "q_optimizer"}, "optimizer state"),
        ({"missing_module": "iql_value"}, "trained modules"),
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

    result = bc_dagger.main.__wrapped__(cfg)

    assert result == "shared-result"
    assert received == [cfg]


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
            _cfg(total_frames=16_384 * 1800),
            "cumulative end rollout",
        ),
    ),
)
def test_bc_dagger_entrypoint_rejects_invalid_stage_before_training(cfg, message):
    with pytest.raises(ValueError, match=message):
        bc_dagger.validate_bc_dagger_config(cfg)
