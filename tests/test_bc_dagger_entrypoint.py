from pathlib import Path

import pytest
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
        },
        "task": {"num_envs": num_envs},
        "total_frames": total_frames,
        "checkpoint_path": checkpoint,
    })


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
    assert cfg.algo.dagger_beta_start == pytest.approx(1.0)
    assert cfg.algo.dagger_beta_end == pytest.approx(0.0)
    assert cfg.algo.dagger_beta_decay_rollouts == 1800
    assert cfg.algo.dagger_replay_raw_observations is True
    assert cfg.algo.teacher_buffer_capacity == 1_048_576

    schedule = bc_dagger.bc_dagger_rollout_schedule(cfg)
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 2400,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 600,
    }


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
            "must be smaller than the total rollout count",
        ),
    ),
)
def test_bc_dagger_entrypoint_rejects_invalid_stage_before_training(cfg, message):
    with pytest.raises(ValueError, match=message):
        bc_dagger.validate_bc_dagger_config(cfg)
