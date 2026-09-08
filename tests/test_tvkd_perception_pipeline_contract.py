from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from active_adaptation.learning.ppo.perception_pipeline_contract import (
    CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    DEPTH_INIT_ONLY_CONTRACT,
    DEPTH_NO_VECNORM_ONLY_CONTRACT,
    LEGACY_PERCEPTION_PIPELINE_CONTRACT,
    PERCEPTION_PIPELINE_CONTRACTS,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    CHECKPOINT_VERSION,
    PERCEPTION_PIPELINE_CONTRACT_KEY,
    TRAINING_ALGORITHM,
    TVKDDistributionalFastSACTeacherBC as TVKD,
    TVKDDistributionalFastSACTeacherBCConfig as Config,
    _require_same_stage_perception_pipeline_contract,
    _runtime_perception_pipeline_contract,
    _saved_perception_pipeline_contract,
    _validate_tvkd_algorithm_config,
)


_FULL_PERCEPTION_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
_PARTIAL_PERCEPTION_MODULES = (
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)


def _perception_warmstart_owner(contract: str) -> TVKD:
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config(
        perception_pipeline_contract=contract,
        perception_depth_residual=False,
        train_perception=True,
    )
    for name in _FULL_PERCEPTION_MODULES:
        setattr(policy, name, nn.Linear(2, 2))
    return policy


def _perception_module_state(policy: TVKD, names) -> dict:
    return {
        name: getattr(policy, name).state_dict()
        for name in names
    }


@pytest.mark.parametrize("contract", sorted(PERCEPTION_PIPELINE_CONTRACTS))
def test_runtime_contract_accepts_only_versioned_values(contract):
    assert (
        _runtime_perception_pipeline_contract(
            SimpleNamespace(perception_pipeline_contract=contract)
        )
        == contract
    )
    cfg = Config(perception_pipeline_contract=contract)
    _validate_tvkd_algorithm_config(cfg)


def test_missing_runtime_and_checkpoint_metadata_resolve_to_legacy():
    assert _runtime_perception_pipeline_contract(SimpleNamespace()) == (
        LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )
    assert _saved_perception_pipeline_contract({}, {}) == (
        LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )


@pytest.mark.parametrize("value", [None, True, "legacy", "corrected_v2"])
def test_invalid_runtime_contract_fails_closed(value):
    with pytest.raises(ValueError, match="perception_pipeline_contract"):
        _runtime_perception_pipeline_contract(
            SimpleNamespace(perception_pipeline_contract=value)
        )


@pytest.mark.parametrize(
    ("top_level", "backend"),
    [
        (CORRECTED_PERCEPTION_PIPELINE_CONTRACT, None),
        (None, CORRECTED_PERCEPTION_PIPELINE_CONTRACT),
        (DEPTH_INIT_ONLY_CONTRACT, DEPTH_NO_VECNORM_ONLY_CONTRACT),
    ],
)
def test_checkpoint_metadata_locations_must_agree(top_level, backend):
    state = {} if top_level is None else {
        PERCEPTION_PIPELINE_CONTRACT_KEY: top_level
    }
    backend_state = {} if backend is None else {
        PERCEPTION_PIPELINE_CONTRACT_KEY: backend
    }
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        _saved_perception_pipeline_contract(state, backend_state)


def test_same_stage_contract_mismatch_hard_fails():
    state = {
        PERCEPTION_PIPELINE_CONTRACT_KEY: (
            CORRECTED_PERCEPTION_PIPELINE_CONTRACT
        )
    }
    backend = dict(state)

    with pytest.raises(
        ValueError,
        match="TVKD resume perception_pipeline_contract mismatch",
    ):
        _require_same_stage_perception_pipeline_contract(
            Config(
                perception_pipeline_contract=(
                    LEGACY_PERCEPTION_PIPELINE_CONTRACT
                )
            ),
            state,
            backend,
            context="TVKD resume",
        )


@pytest.mark.parametrize(
    ("loader_name", "context"),
    [
        ("_load_fastsac_checkpoint_state", "TVKD resume"),
        ("load_inference_state_dict", "TVKD inference"),
    ],
)
def test_public_same_stage_loaders_reject_pipeline_mismatch(
    loader_name,
    context,
):
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config(
        perception_pipeline_contract=LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": CHECKPOINT_VERSION,
        PERCEPTION_PIPELINE_CONTRACT_KEY: (
            CORRECTED_PERCEPTION_PIPELINE_CONTRACT
        ),
        "dagger_backend_config": {
            PERCEPTION_PIPELINE_CONTRACT_KEY: (
                CORRECTED_PERCEPTION_PIPELINE_CONTRACT
            )
        },
    }

    with pytest.raises(
        ValueError,
        match=f"{context} perception_pipeline_contract mismatch",
    ):
        getattr(policy, loader_name)(state)


def test_full_student_perception_path_rejects_depth_pipeline_mismatch(
    tmp_path,
):
    policy = _perception_warmstart_owner(
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )
    source = _perception_module_state(policy, _FULL_PERCEPTION_MODULES)
    source["last_phase"] = "finetune"
    # Missing historical metadata is explicitly legacy_v1.
    path = tmp_path / "legacy_full_student.pt"
    torch.save({"policy": source}, path)

    with pytest.raises(
        ValueError,
        match=(
            "perception_checkpoint_path perception_pipeline_contract "
            "mismatch"
        ),
    ):
        policy._load_pretrained_perception_checkpoint(path)


def test_partial_train_teacher_path_allows_contract_mismatch_without_depth(
    tmp_path,
):
    policy = _perception_warmstart_owner(
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )
    source = _perception_module_state(policy, _PARTIAL_PERCEPTION_MODULES)
    source["last_phase"] = "train"
    path = tmp_path / "legacy_partial_teacher.pt"
    torch.save({"policy": source}, path)

    metadata = policy._load_pretrained_perception_checkpoint(path)

    assert metadata["mode"] == "ppo_vel_train_partial"
    assert metadata["pipeline_contract_match_required"] is False
    assert metadata["source_perception_pipeline_contract"] == (
        LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )
    assert metadata["runtime_perception_pipeline_contract"] == (
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )
    assert metadata["source_depth_modules"] == ()


def test_full_plain_ppovel_perception_path_accepts_matching_policy_metadata(
    tmp_path,
):
    policy = _perception_warmstart_owner(
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )
    source = _perception_module_state(policy, _FULL_PERCEPTION_MODULES)
    source.update(
        {
            "last_phase": "finetune",
            PERCEPTION_PIPELINE_CONTRACT_KEY: (
                CORRECTED_PERCEPTION_PIPELINE_CONTRACT
            ),
        }
    )
    path = tmp_path / "corrected_full_student.pt"
    torch.save({"policy": source}, path)

    metadata = policy._load_pretrained_perception_checkpoint(path)

    assert metadata["mode"] == "strict_full_student"
    assert metadata["pipeline_contract_match_required"] is True
    assert metadata["source_perception_pipeline_contract"] == (
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {PERCEPTION_PIPELINE_CONTRACT_KEY: "corrected_v2"},
        {
            PERCEPTION_PIPELINE_CONTRACT_KEY: (
                CORRECTED_PERCEPTION_PIPELINE_CONTRACT
            ),
            "dagger_backend_config": {},
        },
        {
            PERCEPTION_PIPELINE_CONTRACT_KEY: (
                CORRECTED_PERCEPTION_PIPELINE_CONTRACT
            ),
            "dagger_backend_config": None,
        },
    ],
)
def test_perception_path_rejects_malformed_or_inconsistent_metadata(
    tmp_path,
    metadata,
):
    policy = _perception_warmstart_owner(
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )
    source = _perception_module_state(policy, _FULL_PERCEPTION_MODULES)
    source.update(metadata)
    source["last_phase"] = "finetune"
    path = tmp_path / "malformed_perception.pt"
    torch.save({"policy": source}, path)

    with pytest.raises(ValueError, match="perception.*metadata|unsupported"):
        policy._load_pretrained_perception_checkpoint(path)


def test_checkpoint_backend_config_saves_explicit_contract():
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config(
        perception_pipeline_contract=CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )

    backend = policy._checkpoint_config()

    assert backend[PERCEPTION_PIPELINE_CONTRACT_KEY] == (
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )


def test_checkpoint_rejects_contract_mutated_after_policy_construction():
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.perception_pipeline_contract = LEGACY_PERCEPTION_PIPELINE_CONTRACT
    policy.cfg = Config(
        perception_pipeline_contract=CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    )

    with pytest.raises(RuntimeError, match="changed after policy construction"):
        policy._checkpoint_config()


def test_existing_tvkd_hydra_config_defaults_to_legacy_contract():
    assert Config().perception_pipeline_contract == (
        LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=["task=G1/vaic/skateboard_stu"],
        )

    assert cfg.algo.perception_pipeline_contract == (
        LEGACY_PERCEPTION_PIPELINE_CONTRACT
    )
    assert "perception_pipeline" not in cfg
    assert PERCEPTION_PIPELINE_CONTRACT_KEY not in cfg


@pytest.mark.parametrize(
    "contract",
    [
        LEGACY_PERCEPTION_PIPELINE_CONTRACT,
        DEPTH_NO_VECNORM_ONLY_CONTRACT,
        DEPTH_INIT_ONLY_CONTRACT,
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    ],
)
def test_hydra_perception_pipeline_presets_select_exact_contract(contract):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                f"perception_pipeline={contract}",
            ],
        )

    assert cfg.algo.perception_pipeline_contract == contract
    assert "perception_pipeline" not in cfg
    assert PERCEPTION_PIPELINE_CONTRACT_KEY not in cfg
