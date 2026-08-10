import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from active_adaptation.utils import wandb as wandb_utils
from scripts import helpers as helpers_module
from scripts.helpers import (
    apply_fastsac_buffer_steps,
    apply_teacher_replay_buffer_path_alias,
    find_local_teacher_replay,
    teacher_replay_storage_dir,
)
from scripts.train import make_wandb_settings, maybe_upload_teacher_replay


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


def test_teacher_replay_wandb_upload_is_explicitly_opt_in():
    train_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "cfg" / "train.yaml"
    )

    assert "upload_teacher_replay" in train_cfg.wandb
    assert train_cfg.wandb.upload_teacher_replay is False


class _RecordingRun:
    def __init__(self):
        self.saved = []

    def save(self, path, **kwargs):
        self.saved.append((os.fspath(path), kwargs))


class _OfflineReplayPolicy:
    def __init__(self, path):
        self.path = path
        self.calls = 0

    def get_offline_replay_path(self):
        self.calls += 1
        return self.path


@pytest.mark.parametrize("source", ["snapshot", "offline"])
def test_final_teacher_replay_upload_is_disabled_for_both_sources_by_default(
    source,
    tmp_path,
):
    replay = tmp_path / f"{source}.h5"
    replay.touch()
    run = _RecordingRun()
    policy = _OfflineReplayPolicy(replay if source == "offline" else None)
    snapshot = replay if source == "snapshot" else None
    cfg = OmegaConf.create({"wandb": {"upload_teacher_replay": False}})

    result = maybe_upload_teacher_replay(
        run,
        cfg,
        policy,
        snapshot,
        artifact=True,
    )

    assert result is None
    assert run.saved == []
    assert policy.calls == (0 if source == "snapshot" else 1)


@pytest.mark.parametrize("source", ["snapshot", "offline"])
def test_final_teacher_replay_upload_can_be_explicitly_enabled_for_both_sources(
    source,
    tmp_path,
):
    replay = tmp_path / f"{source}.h5"
    replay.touch()
    run = _RecordingRun()
    policy = _OfflineReplayPolicy(replay if source == "offline" else None)
    snapshot = replay if source == "snapshot" else None
    cfg = OmegaConf.create({"wandb": {"upload_teacher_replay": True}})

    result = maybe_upload_teacher_replay(
        run,
        cfg,
        policy,
        snapshot,
        artifact=True,
    )

    assert result == replay
    assert run.saved == [
        (
            os.fspath(replay),
            {"policy": "now", "base_path": os.fspath(tmp_path)},
        )
    ]
    assert policy.calls == (0 if source == "snapshot" else 1)


def test_periodic_checkpoint_never_uploads_teacher_replay_even_when_enabled(
    tmp_path,
):
    replay = tmp_path / "teacher_replay_buffer.h5"
    replay.touch()
    run = _RecordingRun()
    policy = _OfflineReplayPolicy(replay)
    cfg = OmegaConf.create({"wandb": {"upload_teacher_replay": True}})

    result = maybe_upload_teacher_replay(
        run,
        cfg,
        policy,
        replay,
        artifact=False,
    )

    assert result is None
    assert run.saved == []
    assert policy.calls == 0


def _wandb_files_layout(tmp_path):
    output_root = tmp_path / "outputs" / "run"
    files_dir = output_root / "wandb" / "run-20260806-id" / "files"
    files_dir.mkdir(parents=True)
    checkpoint = files_dir / "checkpoint_final.pt"
    checkpoint.touch()
    return output_root, files_dir, checkpoint


def test_teacher_replay_storage_is_hydra_root_outside_wandb_files(tmp_path):
    output_root, files_dir, _ = _wandb_files_layout(tmp_path)

    storage_dir = Path(teacher_replay_storage_dir(files_dir))

    assert storage_dir == output_root.resolve()
    assert storage_dir != files_dir.resolve()
    assert files_dir.resolve() not in storage_dir.parents
    assert storage_dir / "teacher_replay_buffer.h5" != (
        files_dir / "teacher_replay_buffer.h5"
    )


def test_local_checkpoint_finds_output_root_teacher_replay(tmp_path):
    output_root, _, checkpoint = _wandb_files_layout(tmp_path)
    replay = output_root / "teacher_replay_buffer.h5"
    replay.touch()

    resolved = find_local_teacher_replay(
        checkpoint,
        "teacher_replay_buffer.h5",
    )

    assert resolved == str(replay.resolve())


def test_local_checkpoint_still_finds_legacy_adjacent_teacher_replay(tmp_path):
    _, files_dir, checkpoint = _wandb_files_layout(tmp_path)
    replay = files_dir / "teacher_replay_buffer.h5"
    replay.touch()

    resolved = find_local_teacher_replay(
        checkpoint,
        "teacher_replay_buffer.h5",
    )

    assert resolved == str(replay.resolve())


def test_local_checkpoint_rejects_ambiguous_root_and_legacy_replays(tmp_path):
    output_root, files_dir, checkpoint = _wandb_files_layout(tmp_path)
    root_replay = output_root / "teacher_replay_buffer.h5"
    legacy_replay = files_dir / "teacher_replay_buffer.h5"
    root_replay.touch()
    legacy_replay.touch()

    with pytest.raises(RuntimeError, match="Multiple teacher replay buffers"):
        find_local_teacher_replay(
            checkpoint,
            "teacher_replay_buffer.h5",
        )


@pytest.mark.parametrize(
    ("upload_enabled", "expected_globs"),
    [(False, ("custom_teacher_replay.h5",)), (True, ())],
)
def test_wandb_settings_ignore_local_teacher_replay_unless_upload_opted_in(
    upload_enabled,
    expected_globs,
):
    cfg = OmegaConf.create({
        "wandb": {"upload_teacher_replay": upload_enabled},
        "algo": {"teacher_buffer_filename": "custom_teacher_replay.h5"},
    })

    settings = make_wandb_settings(cfg)

    assert settings.ignore_globs == expected_globs


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


def test_bc_dagger_resume_skips_shared_helper_h5_auto_download(
    monkeypatch,
):
    class StopAfterCheckpointResolution(Exception):
        pass

    calls = []

    def resolve(path, **kwargs):
        calls.append((path, kwargs))
        raise StopAfterCheckpointResolution

    # Stop immediately after the decision under test; no simulator or policy
    # needs to be constructed for this replay-resolution regression.
    monkeypatch.setitem(
        sys.modules,
        "active_adaptation.envs",
        SimpleNamespace(SimpleEnv=lambda task: SimpleNamespace()),
    )
    monkeypatch.setattr(helpers_module, "parse_checkpoint_path", resolve)
    cfg = OmegaConf.create({
        "seed": 0,
        "checkpoint_path": "/models/checkpoint_800.pt",
        "bc_dagger_checkpoint": "/models/checkpoint_800.pt",
        "_bc_dagger_model_only_resume": True,
        "teacher_replay_buffer_path": None,
        "task": {"observation": {}},
        "algo": {
            "in_keys": [],
            "phase": "finetune",
            "save_teacher_buffer": False,
            "teacher_buffer_path": None,
            "teacher_buffer_filename": "teacher_replay_buffer.h5",
        },
    })

    with pytest.raises(StopAfterCheckpointResolution):
        helpers_module.make_env_policy(cfg, configure_replay=True)

    assert calls == [
        (
            "/models/checkpoint_800.pt",
            {
                "download_replay": False,
                "replay_filename": "teacher_replay_buffer.h5",
            },
        )
    ]


def test_staged_bc_dagger_source_skips_shared_helper_h5_auto_download(
    monkeypatch,
):
    class StopAfterCheckpointResolution(Exception):
        pass

    calls = []

    def resolve(path, **kwargs):
        calls.append((path, kwargs))
        raise StopAfterCheckpointResolution

    # The fresh staged source intentionally ignores any H5 adjacent to the PPO
    # teacher. Stop before simulator/policy construction to isolate that guard.
    monkeypatch.setitem(
        sys.modules,
        "active_adaptation.envs",
        SimpleNamespace(SimpleEnv=lambda task: SimpleNamespace()),
    )
    monkeypatch.setattr(helpers_module, "parse_checkpoint_path", resolve)
    cfg = OmegaConf.create({
        "seed": 0,
        "checkpoint_path": "/models/checkpoint_6000.pt",
        "_bc_dagger_staging_source": True,
        "teacher_replay_buffer_path": None,
        "task": {"observation": {}},
        "algo": {
            "in_keys": [],
            "phase": "finetune",
            "save_teacher_buffer": True,
            "teacher_buffer_path": None,
            "teacher_buffer_filename": "teacher_replay_buffer.h5",
        },
    })

    with pytest.raises(StopAfterCheckpointResolution):
        helpers_module.make_env_policy(cfg, configure_replay=True)

    assert calls == [
        (
            "/models/checkpoint_6000.pt",
            {
                "download_replay": False,
                "replay_filename": "teacher_replay_buffer.h5",
            },
        )
    ]


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
