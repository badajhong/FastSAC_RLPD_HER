import hashlib
import json
import os
import stat
from pathlib import Path

import h5py
import pytest
import torch
from omegaconf import OmegaConf

from scripts import bc_dagger
from scripts import helpers as helpers_module


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


def _write_resume_checkpoint(path: Path) -> Path:
    optimizer_state = {
        name: {"state": {}, "param_groups": []}
        for name in bc_dagger.REQUIRED_RESUME_OPTIMIZERS
    }
    policy = {
        "training_algorithm": bc_dagger.EXPECTED_TRAINING_ALGORITHM,
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
        "dagger_rollout_count": 801,
        "dagger_environment_steps": 25_632,
        "bc_update_count": 25_632,
        "q_update_count": 25_632,
        "dagger_rng_state": torch.Generator().get_state(),
        "q_rng_state": torch.Generator().get_state(),
        "sac_action_rng_state": torch.Generator().get_state(),
        "next_iter": 6_801,
        "teacher_replay_state": dict(REPLAY_LINEAGE),
    }
    torch.save({"policy": policy, "vecnorm": {}}, path)
    return path


def _write_teacher_replay(path: Path, payload=b"offline-transitions") -> Path:
    with h5py.File(path, "w") as replay:
        for key, value in REPLAY_LINEAGE.items():
            replay.attrs[key] = (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if key == "action_contract"
                else value
            )
        replay.create_dataset("payload", data=list(payload), dtype="u1")
    return path


def _resume_cfg(
    checkpoint,
    *,
    copy_teacher_replay: bool = True,
):
    return OmegaConf.create({
        "checkpoint_path": None,
        "bc_dagger_checkpoint": os.fspath(checkpoint),
        "bc_dagger_copy_teacher_replay": copy_teacher_replay,
        "teacher_replay_buffer_path": None,
        "algo": {
            "train_every": 32,
            "save_teacher_buffer": True,
            "teacher_buffer_path": None,
            "teacher_buffer_filename": "teacher_replay_buffer.h5",
        },
    })


def _source_identity(path: Path):
    info = path.stat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "links": info.st_nlink,
        "mode": stat.S_IMODE(info.st_mode),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_resume_prepares_adjacent_immutable_replay_copy_source(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    source = _write_teacher_replay(tmp_path / "teacher_replay_buffer.h5")
    cfg = _resume_cfg(checkpoint)

    resume = bc_dagger.prepare_bc_dagger_checkpoint(cfg)

    assert resume["teacher_replay_source"] == str(source.resolve())
    assert cfg._bc_dagger_teacher_replay_copy_source == str(source.resolve())
    # The source is only an export source. It must never become the policy's
    # mutable live H5 or its Stage-2 replay input during this resumed stage.
    assert cfg.algo.save_teacher_buffer is False
    assert cfg.algo.teacher_buffer_path is None
    assert cfg.teacher_replay_buffer_path is None


def test_resume_copy_is_physical_read_only_and_does_not_mutate_source(
    tmp_path,
):
    source_dir = tmp_path / "old-run"
    source_dir.mkdir()
    checkpoint = _write_resume_checkpoint(source_dir / "checkpoint_800.pt")
    source = _write_teacher_replay(
        source_dir / "teacher_replay_buffer.h5",
        b"raw observation replay\0" * 128,
    )
    source.chmod(0o640)
    cfg = _resume_cfg(checkpoint)
    bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    source_before = _source_identity(source)
    output_dir = tmp_path / "new-run" / "files"
    output_dir.mkdir(parents=True)

    copied = helpers_module.copy_frozen_teacher_replay(
        cfg._bc_dagger_teacher_replay_copy_source,
        output_dir,
        cfg.algo.teacher_buffer_filename,
    )

    destination = output_dir / "teacher_replay_buffer.h5"
    assert copied == str(destination.resolve())
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert _source_identity(source) == source_before
    assert list(output_dir.glob(".teacher_replay_buffer.h5.*.copying")) == []


def test_resume_copy_refuses_to_replace_existing_destination(tmp_path):
    source_dir = tmp_path / "old-run"
    source_dir.mkdir()
    checkpoint = _write_resume_checkpoint(source_dir / "checkpoint_800.pt")
    source = _write_teacher_replay(
        source_dir / "teacher_replay_buffer.h5", b"complete-source-replay"
    )
    cfg = _resume_cfg(checkpoint)
    bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    output_dir = tmp_path / "new-run"
    output_dir.mkdir()
    destination = output_dir / "teacher_replay_buffer.h5"
    destination.write_bytes(b"stale-partial-copy")
    source_before = _source_identity(source)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        helpers_module.copy_frozen_teacher_replay(
            cfg._bc_dagger_teacher_replay_copy_source,
            output_dir,
            cfg.algo.teacher_buffer_filename,
        )

    assert destination.read_bytes() == b"stale-partial-copy"
    assert _source_identity(source) == source_before
    assert list(output_dir.glob(".teacher_replay_buffer.h5.*.copying")) == []


def test_resume_copy_rejects_non_basename_destination(tmp_path):
    source = _write_teacher_replay(tmp_path / "source.h5")

    with pytest.raises(ValueError, match="basename"):
        helpers_module.copy_frozen_teacher_replay(
            source, tmp_path / "new-run", "../teacher_replay_buffer.h5"
        )


def test_resume_copy_publication_failure_leaves_no_partial_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    source_dir = tmp_path / "old-run"
    source_dir.mkdir()
    checkpoint = _write_resume_checkpoint(source_dir / "checkpoint_800.pt")
    source = _write_teacher_replay(
        source_dir / "teacher_replay_buffer.h5", b"new-source-replay"
    )
    cfg = _resume_cfg(checkpoint)
    bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    output_dir = tmp_path / "new-run"
    output_dir.mkdir()
    destination = output_dir / "teacher_replay_buffer.h5"
    source_before = _source_identity(source)

    def fail_replace(source_path, destination_path):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(helpers_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="atomic replace failure"):
        helpers_module.copy_frozen_teacher_replay(
            cfg._bc_dagger_teacher_replay_copy_source,
            output_dir,
            cfg.algo.teacher_buffer_filename,
        )

    assert not destination.exists()
    assert _source_identity(source) == source_before
    assert list(output_dir.glob(".teacher_replay_buffer.h5.*.copying")) == []


def test_enabled_resume_copy_fails_fast_when_adjacent_source_is_missing(
    tmp_path,
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    cfg = _resume_cfg(checkpoint)

    with pytest.raises(FileNotFoundError, match="teacher_replay_buffer.h5"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)

    assert not hasattr(cfg, "_bc_dagger_teacher_replay_copy_source")


def test_enabled_resume_copy_rejects_replay_from_different_lineage(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    source = _write_teacher_replay(tmp_path / "teacher_replay_buffer.h5")
    with h5py.File(source, "r+") as replay:
        replay.attrs["replay_id"] = "different-run"
    cfg = _resume_cfg(checkpoint)

    with pytest.raises(ValueError, match="replay_id.*does not match"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


def test_enabled_resume_copy_rejects_replay_with_different_action_clip(
    tmp_path,
):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    source = _write_teacher_replay(tmp_path / "teacher_replay_buffer.h5")
    with h5py.File(source, "r+") as replay:
        replay.attrs["action_clip"] = 10.0
    cfg = _resume_cfg(checkpoint)

    with pytest.raises(ValueError, match="action_clip.*does not match"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


def test_enabled_resume_copy_requires_executable_action_contract(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    source = _write_teacher_replay(tmp_path / "teacher_replay_buffer.h5")
    with h5py.File(source, "r+") as replay:
        del replay.attrs["action_contract"]
    cfg = _resume_cfg(checkpoint)

    with pytest.raises(ValueError, match="missing.*action_contract"):
        bc_dagger.prepare_bc_dagger_checkpoint(cfg)


def test_disabled_resume_copy_still_requires_local_source_and_copy_is_noop(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path / "checkpoint_800.pt")
    source = _write_teacher_replay(tmp_path / "teacher_replay_buffer.h5")
    cfg = _resume_cfg(checkpoint, copy_teacher_replay=False)

    resume = bc_dagger.prepare_bc_dagger_checkpoint(cfg)
    output_dir = tmp_path / "new-run"
    output_dir.mkdir()

    assert resume["teacher_replay_source"] == str(source.resolve())
    assert cfg._bc_dagger_teacher_replay_copy_source == str(source.resolve())
    assert not (output_dir / "teacher_replay_buffer.h5").exists()


@pytest.mark.parametrize(
    ("copy_teacher_replay", "expected_download"),
    ((True, True), (False, False)),
)
def test_run_resume_downloads_replay_only_when_copy_requested(
    monkeypatch,
    tmp_path,
    copy_teacher_replay,
    expected_download,
):
    cache_dir = tmp_path / "wandb-cache" / "files"
    cache_dir.mkdir(parents=True)
    checkpoint = _write_resume_checkpoint(cache_dir / "checkpoint_final.pt")
    # Opting out suppresses the remote download request, but a cached local
    # replay is still mandatory to refill the persistent 50% teacher partition.
    _write_teacher_replay(cache_dir / "teacher_replay_buffer.h5")
    calls = []

    def resolve(path, **kwargs):
        calls.append((path, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(bc_dagger, "parse_checkpoint_path", resolve)
    cfg = _resume_cfg(
        "run:entity/project/run-id",
        copy_teacher_replay=copy_teacher_replay,
    )

    bc_dagger.prepare_bc_dagger_checkpoint(cfg)

    assert calls == [
        (
            "run:entity/project/run-id",
            {
                "download_replay": expected_download,
                "replay_filename": "teacher_replay_buffer.h5",
            },
        )
    ]
