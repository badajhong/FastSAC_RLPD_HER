import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from active_adaptation.utils import wandb as wandb_utils
from scripts.helpers import (
    apply_fastsac_buffer_steps,
    apply_teacher_replay_buffer_path_alias,
)


class _RemoteFile:
    def __init__(self, name, downloads):
        self.name = name
        self._downloads = downloads

    def download(self, root, replace):
        self._downloads.append((self.name, root, replace))


def test_replay_name_containing_checkpoint_is_not_model_checkpoint(
    monkeypatch, tmp_path
):
    downloads = []
    remote_files = [
        _RemoteFile("files/checkpoint_12.pt", downloads),
        _RemoteFile("files/checkpoint_final.pt", downloads),
        _RemoteFile("files/my_checkpoint_replay.h5", downloads),
    ]
    run = SimpleNamespace(id="stable-run-id", files=lambda: remote_files)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "Api",
        lambda: SimpleNamespace(run=lambda _: run),
    )
    monkeypatch.setattr(wandb_utils, "__file__", str(tmp_path / "wandb.py"))

    path = wandb_utils.parse_checkpoint_path(
        "run:entity/project/run-id",
        download_replay=True,
        replay_filename="my_checkpoint_replay.h5",
    )

    assert path == os.path.join(
        str(tmp_path), "wandb", "stable-run-id", "files", "checkpoint_final.pt"
    )
    assert [name for name, _, _ in downloads] == [
        "files/checkpoint_final.pt",
        "files/my_checkpoint_replay.h5",
    ]


def _training_cfg(replay_path, checkpoint_path="run:entity/project/run-id"):
    return OmegaConf.create({
        "checkpoint_path": checkpoint_path,
        "teacher_replay_buffer_path": replay_path,
        "algo": {"teacher_buffer_path": None},
    })


def test_explicit_teacher_replay_alias_sets_internal_fastsac_path(tmp_path):
    replay = tmp_path / "teacher_replay_buffer.h5"
    replay.touch()
    cfg = _training_cfg(str(replay))

    resolved = apply_teacher_replay_buffer_path_alias(cfg)

    assert resolved == str(replay.resolve())
    assert cfg.teacher_replay_buffer_path == resolved
    assert cfg.algo.teacher_buffer_path == resolved


def test_stage1_rejects_offline_teacher_replay_path(tmp_path):
    replay = tmp_path / "teacher_replay_buffer.h5"
    replay.touch()
    cfg = _training_cfg(str(replay))
    cfg.algo.phase = "train"

    with pytest.raises(ValueError, match="ephemeral compact learning FIFO"):
        apply_teacher_replay_buffer_path_alias(cfg)


def test_train_config_declares_exact_teacher_replay_cli_alias():
    train_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "cfg" / "train.yaml"
    )
    assert "teacher_replay_buffer_path" in train_cfg
    assert train_cfg.teacher_replay_buffer_path is None


def _buffer_steps_cfg(num_envs=2048, buffer_steps=1024, phase="train"):
    return OmegaConf.create({
        "task": {"num_envs": num_envs, "buffer_steps": buffer_steps},
        "algo": {
            "phase": phase,
            "teacher_buffer_capacity": 262_144,
        },
    })


def test_fastsac_buffer_steps_derives_flat_capacity_from_local_env_count():
    cfg = _buffer_steps_cfg()

    capacity = apply_fastsac_buffer_steps(cfg)

    assert capacity == 2_097_152
    assert cfg.algo.teacher_buffer_capacity == capacity


def test_null_buffer_steps_preserves_explicit_capacity():
    cfg = _buffer_steps_cfg(buffer_steps=None)

    assert apply_fastsac_buffer_steps(cfg) is None
    assert cfg.algo.teacher_buffer_capacity == 262_144


@pytest.mark.parametrize(
    ("num_envs", "buffer_steps"),
    [(0, 64), (2048, 0), (2048, 1.5), (True, 64)],
)
def test_fastsac_buffer_steps_requires_positive_integers(
    num_envs, buffer_steps,
):
    cfg = _buffer_steps_cfg(num_envs=num_envs, buffer_steps=buffer_steps)

    with pytest.raises(ValueError, match="must be a positive integer"):
        apply_fastsac_buffer_steps(cfg)


def test_fastsac_buffer_steps_is_teacher_stage_only():
    cfg = _buffer_steps_cfg(phase="finetune")

    with pytest.raises(ValueError, match="only supported by Stage-1"):
        apply_fastsac_buffer_steps(cfg)


def test_legacy_nested_teacher_replay_path_is_also_normalized(tmp_path):
    replay = tmp_path / "teacher_replay_buffer.h5"
    replay.touch()
    cfg = _training_cfg(None)
    cfg.algo.teacher_buffer_path = str(replay)

    resolved = apply_teacher_replay_buffer_path_alias(cfg)

    assert resolved == str(replay.resolve())
    assert cfg.teacher_replay_buffer_path is None
    assert cfg.algo.teacher_buffer_path == resolved


def test_explicit_teacher_replay_requires_checkpoint_and_existing_file(tmp_path):
    replay = tmp_path / "teacher_replay_buffer.h5"
    replay.touch()
    with pytest.raises(ValueError, match="used with checkpoint_path"):
        apply_teacher_replay_buffer_path_alias(
            _training_cfg(str(replay), checkpoint_path=None)
        )

    missing = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        apply_teacher_replay_buffer_path_alias(_training_cfg(str(missing)))


def test_explicit_teacher_replay_rejects_conflicting_internal_path(tmp_path):
    explicit = tmp_path / "explicit.h5"
    explicit.touch()
    cfg = _training_cfg(str(explicit))
    cfg.algo.teacher_buffer_path = str(tmp_path / "different.h5")

    with pytest.raises(ValueError, match="Conflicting teacher replay paths"):
        apply_teacher_replay_buffer_path_alias(cfg)
