from __future__ import annotations

from pathlib import Path
from dataclasses import fields
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from active_adaptation.learning.ppo.exact_online_perception import (
    EXACT_ONLINE_ACTOR_REPLAY_SEMANTICS,
    ExactOnlinePerceptionReplayMixin,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TRAINING_ALGORITHM,
    TVKDDistributionalFastSACTeacherBC,
    TVKDDistributionalFastSACTeacherBCConfig,
    _saved_online_replay_latent_mode,
    _tvkd_actor_replay_observation_semantics,
    _validate_tvkd_algorithm_config,
)
from scripts.helpers import _fill_replayless_inference_algo_defaults


def test_tvkd_exact_online_latents_are_built_in_not_a_config_option():
    config = TVKDDistributionalFastSACTeacherBCConfig()
    assert "online_replay_latent_mode" not in {field.name for field in fields(config)}
    config_dir = str(Path(__file__).resolve().parents[1] / "cfg")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=["task=G1/vaic/skateboard_general_tracking_stu"],
        )
    assert "online_replay_latent_mode" not in cfg.algo
    assert TVKDDistributionalFastSACTeacherBC._exact_online_replay_enabled(None) is True
    assert ExactOnlinePerceptionReplayMixin._exact_online_replay_enabled(None) is False
    assert _tvkd_actor_replay_observation_semantics(config) == EXACT_ONLINE_ACTOR_REPLAY_SEMANTICS


@pytest.mark.parametrize("mode", ["collection", "exact_current_ema", "burn_in"])
def test_removed_cli_selector_cannot_change_tvkd_replay(mode):
    config_dir = str(Path(__file__).resolve().parents[1] / "cfg")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        with pytest.raises(ConfigCompositionException, match="online_replay_latent_mode"):
            compose(config_name="TVKD_fasSAC_bc_dagger", overrides=["task=G1/vaic/skateboard_stu", f"algo.online_replay_latent_mode={mode}"])
        forced = compose(config_name="TVKD_fasSAC_bc_dagger", overrides=["task=G1/vaic/skateboard_stu", f"++algo.online_replay_latent_mode={mode}"])
    with pytest.raises(ValueError, match="not configurable"):
        _validate_tvkd_algorithm_config(forced.algo)
    # The class invariant also cannot be disabled by attaching a stale field.
    stale = SimpleNamespace(cfg=SimpleNamespace(online_replay_latent_mode=mode))
    assert TVKDDistributionalFastSACTeacherBC._exact_online_replay_enabled(stale)


@pytest.mark.parametrize(
    "field,value",
    [
        ("online_replay_latent_mode", "burn_in"),
        ("online_replay_latent_mode", []),
        ("perception_replay_mode", "four_way"),
        ("sac_actor_observation_mode", "privileged_oracle"),
        ("q_n_step", 2),
    ],
)
def test_exact_mode_rejects_incompatible_contract(field, value):
    cfg = TVKDDistributionalFastSACTeacherBCConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="online_replay_latent_mode|exact_current_ema|q_n_step is locked"):
        _validate_tvkd_algorithm_config(cfg)


def test_missing_checkpoint_mode_describes_history_not_runtime_behavior():
    assert _saved_online_replay_latent_mode({}, {}) == "collection"
    assert _tvkd_actor_replay_observation_semantics(TVKDDistributionalFastSACTeacherBCConfig()) == EXACT_ONLINE_ACTOR_REPLAY_SEMANTICS


def test_new_checkpoint_records_exact_semantics_without_a_backend_selector():
    assert _saved_online_replay_latent_mode({"online_replay_latent_mode": "exact_current_ema"}, {}) == "exact_current_ema"


@pytest.mark.parametrize("mode", ["collection", "exact_current_ema"])
def test_checkpoint_mode_is_validated_but_not_restored_as_an_inference_option(mode):
    backend = {"online_replay_latent_mode": mode, "value_norm": False}
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "dagger_backend_config": backend,
        "online_replay_latent_mode": mode,
    }
    assert _saved_online_replay_latent_mode(state, backend) == mode
    cfg = OmegaConf.create({"algo": {"online_replay_latent_mode": "collection"}})
    _fill_replayless_inference_algo_defaults(cfg, state, inference_only=True)
    assert "online_replay_latent_mode" not in cfg.algo
    state["online_replay_latent_mode"] = "collection" if mode == "exact_current_ema" else "exact_current_ema"
    with pytest.raises(ValueError, match="online_replay_latent_mode"):
        _saved_online_replay_latent_mode(state, backend)
    with pytest.raises(ValueError, match="online_replay_latent_mode"):
        _fill_replayless_inference_algo_defaults(cfg, state, inference_only=True)


def test_legacy_inference_strips_obsolete_selector():
    cfg = OmegaConf.create({"algo": {"online_replay_latent_mode": "exact_current_ema"}})
    state = {"training_algorithm": TRAINING_ALGORITHM, "dagger_backend_config": {"value_norm": False}}
    _fill_replayless_inference_algo_defaults(cfg, state, inference_only=True)
    assert "online_replay_latent_mode" not in cfg.algo
