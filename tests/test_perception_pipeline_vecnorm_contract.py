import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, RenameTransform, VecNorm

from scripts.helpers import (
    _fill_replayless_inference_algo_defaults,
    _resolve_perception_pipeline_contract,
    _vecnorm_observation_keys,
)


_TVKD_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v9"


def _tvkd_inference_state(*, top_level=..., backend=...):
    state = {
        "training_algorithm": _TVKD_TRAINING_ALGORITHM,
        "dagger_backend_config": {"value_norm": False},
    }
    if top_level is not ...:
        state["perception_pipeline_contract"] = top_level
    if backend is not ...:
        state["dagger_backend_config"]["perception_pipeline_contract"] = backend
    return state


@pytest.mark.parametrize(
    ("contract", "expected"),
    [
        ("legacy_v1", ["policy", "depth", "vel_command"]),
        ("depth_no_vecnorm_only_v1", ["policy", "vel_command"]),
        ("depth_init_only_v1", ["policy", "depth", "vel_command"]),
        ("corrected_v1", ["policy", "vel_command"]),
    ],
)
def test_vecnorm_keys_follow_perception_pipeline_contract(contract, expected):
    all_observation_keys = ["policy", "depth", "vel_command"]

    selected = _vecnorm_observation_keys(all_observation_keys, contract)

    assert selected == expected
    # Selection must not mutate the complete key set later used by raw replay.
    assert all_observation_keys == ["policy", "depth", "vel_command"]


def test_missing_pipeline_contract_preserves_legacy_behavior():
    contract = _resolve_perception_pipeline_contract({})

    assert contract == "legacy_v1"
    assert _vecnorm_observation_keys(["policy", "depth"], contract) == [
        "policy",
        "depth",
    ]


def test_legacy_fake_depth_statistics_reproduce_hundred_x_input():
    """Lock the historical zero-fake-stat behavior as an explicit baseline."""

    vecnorm = VecNorm(["depth"], decay=0.9999)
    vecnorm(TensorDict({"depth": torch.zeros(512, 1)}, batch_size=[512]))
    frozen = vecnorm.to_observation_norm()
    raw = torch.tensor([[0.0], [0.1], [0.5], [1.0]])

    transformed = frozen(
        TensorDict({"depth": raw.clone()}, batch_size=[raw.shape[0]])
    )

    torch.testing.assert_close(
        transformed["depth"].flatten(),
        torch.tensor([0.0, 10.0, 50.0, 100.0]),
    )


def test_loading_legacy_vecnorm_state_cannot_reenable_corrected_depth():
    """The constructor-owned key topology remains authoritative on load."""

    legacy = VecNorm(["policy", "depth"], decay=0.9999)
    legacy(
        TensorDict(
            {
                "policy": torch.zeros(8, 2),
                "depth": torch.zeros(8, 1),
            },
            batch_size=[8],
        )
    )
    corrected = VecNorm(["policy"], decay=0.9999)
    corrected(TensorDict({"policy": torch.zeros(8, 2)}, batch_size=[8]))

    corrected.load_state_dict(legacy.state_dict())
    assert corrected.in_keys == ["policy"]

    policy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    depth = torch.tensor([[0.1], [1.0]])
    transformed = corrected.to_observation_norm()(
        TensorDict(
            {"policy": policy.clone(), "depth": depth.clone()},
            batch_size=[2],
        )
    )
    assert torch.equal(transformed["depth"], depth)


@pytest.mark.parametrize("contract", [None, True, "corrected", "legacy_v2"])
def test_invalid_pipeline_contract_fails_closed(contract):
    with pytest.raises(ValueError, match="perception_pipeline_contract"):
        _resolve_perception_pipeline_contract(
            {"perception_pipeline_contract": contract}
        )


def test_corrected_inference_hydrates_contract_before_env_construction():
    cfg = OmegaConf.create(
        {"algo": {"perception_pipeline_contract": "legacy_v1"}}
    )
    state = _tvkd_inference_state(
        top_level="corrected_v1",
        backend="corrected_v1",
    )

    filled = _fill_replayless_inference_algo_defaults(
        cfg,
        state,
        inference_only=True,
    )

    assert cfg.algo.perception_pipeline_contract == "corrected_v1"
    assert "perception_pipeline_contract" in filled["checkpoint"]
    assert _vecnorm_observation_keys(
        ["policy", "depth"],
        cfg.algo.perception_pipeline_contract,
    ) == ["policy"]


def test_plain_ppovel_inference_hydrates_corrected_contract():
    cfg = OmegaConf.create(
        {"algo": {"perception_pipeline_contract": "legacy_v1"}}
    )
    state = {
        "perception_pipeline_contract": "corrected_v1",
        "depth_cnn": {},
    }

    filled = _fill_replayless_inference_algo_defaults(
        cfg,
        state,
        inference_only=True,
    )

    assert cfg.algo.perception_pipeline_contract == "corrected_v1"
    assert filled == {
        "checkpoint": ("perception_pipeline_contract",),
        "defaults": (),
    }
    assert _vecnorm_observation_keys(
        ["policy", "depth"],
        cfg.algo.perception_pipeline_contract,
    ) == ["policy"]


def test_historical_plain_student_inference_hydrates_legacy_contract():
    cfg = OmegaConf.create(
        {"algo": {"perception_pipeline_contract": "corrected_v1"}}
    )

    _fill_replayless_inference_algo_defaults(
        cfg,
        {"depth_cnn": {}},
        inference_only=True,
    )

    assert cfg.algo.perception_pipeline_contract == "legacy_v1"


def test_plain_inference_rejects_invalid_explicit_contract():
    cfg = OmegaConf.create({"algo": {}})

    with pytest.raises(ValueError, match="perception_pipeline_contract"):
        _fill_replayless_inference_algo_defaults(
            cfg,
            {"perception_pipeline_contract": "unknown_v9"},
            inference_only=True,
        )


def test_legacy_inference_metadata_is_explicitly_hydrated_when_missing():
    cfg = OmegaConf.create({"algo": {}})

    filled = _fill_replayless_inference_algo_defaults(
        cfg,
        _tvkd_inference_state(),
        inference_only=True,
    )

    assert cfg.algo.perception_pipeline_contract == "legacy_v1"
    assert "perception_pipeline_contract" in filled["checkpoint"]


@pytest.mark.parametrize(
    ("top_level", "backend"),
    [
        ("corrected_v1", ...),
        (..., "corrected_v1"),
        ("unknown_v9", "unknown_v9"),
    ],
)
def test_inference_rejects_incomplete_or_invalid_checkpoint_contract(
    top_level,
    backend,
):
    cfg = OmegaConf.create({"algo": {}})

    with pytest.raises(
        ValueError,
        match="perception_pipeline_contract",
    ):
        _fill_replayless_inference_algo_defaults(
            cfg,
            _tvkd_inference_state(
                top_level=top_level,
                backend=backend,
            ),
            inference_only=True,
        )


def test_training_path_does_not_hydrate_pipeline_contract_from_checkpoint():
    cfg = OmegaConf.create(
        {"algo": {"perception_pipeline_contract": "legacy_v1"}}
    )

    filled = _fill_replayless_inference_algo_defaults(
        cfg,
        _tvkd_inference_state(
            top_level="corrected_v1",
            backend="corrected_v1",
        ),
        inference_only=False,
    )

    assert cfg.algo.perception_pipeline_contract == "legacy_v1"
    assert filled == {"checkpoint": (), "defaults": ()}


@pytest.mark.parametrize(
    ("contract", "depth_is_normalized"),
    [
        ("legacy_v1", True),
        ("depth_init_only_v1", True),
        ("depth_no_vecnorm_only_v1", False),
        ("corrected_v1", False),
    ],
)
def test_raw_depth_copy_is_preserved_independently_of_vecnorm(
    contract,
    depth_is_normalized,
):
    all_observation_keys = ["policy", "depth"]
    vecnorm_keys = _vecnorm_observation_keys(all_observation_keys, contract)
    raw_keys = [("_fastsac_raw", key) for key in all_observation_keys]
    transform = Compose(
        RenameTransform(all_observation_keys, raw_keys, create_copy=True),
        VecNorm(vecnorm_keys, decay=0.9999),
    )
    policy = torch.tensor([[1.0, 3.0], [4.0, 8.0]])
    depth = torch.tensor([[0.1, 0.5], [0.2, 1.0]])
    td = TensorDict(
        {"policy": policy.clone(), "depth": depth.clone()},
        batch_size=[2],
    )

    transformed = transform(td)

    assert torch.equal(transformed["_fastsac_raw", "policy"], policy)
    assert torch.equal(transformed["_fastsac_raw", "depth"], depth)
    assert not torch.equal(transformed["policy"], policy)
    if depth_is_normalized:
        assert not torch.equal(transformed["depth"], depth)
    else:
        assert torch.equal(transformed["depth"], depth)
