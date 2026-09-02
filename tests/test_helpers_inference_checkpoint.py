from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts import helpers


class _InferencePolicy:
    def __init__(self):
        self.calls = []

    def load_inference_state_dict(self, state, strict=True):
        self.calls.append(("inference", state, strict))
        return ["model-only"]

    def load_state_dict(self, state):
        self.calls.append(("ordinary", state))
        return ["ordinary"]


class _OrdinaryPolicy:
    def __init__(self):
        self.calls = []

    def load_state_dict(self, state):
        self.calls.append(state)
        return ["ordinary"]


def test_algorithm_checkpoint_uses_model_only_loader_during_inference():
    policy = _InferencePolicy()
    state = {"training_algorithm": "distributional_td3_teacher_bc_v1"}

    result = helpers._load_policy_checkpoint(
        policy,
        state,
        inference_only=True,
    )

    assert result == ["model-only"]
    assert policy.calls == [("inference", state, True)]


def test_training_path_keeps_using_guarded_ordinary_loader():
    policy = _InferencePolicy()
    state = {"training_algorithm": "distributional_td3_teacher_bc_v1"}

    result = helpers._load_policy_checkpoint(
        policy,
        state,
        inference_only=False,
    )

    assert result == ["ordinary"]
    assert policy.calls == [("ordinary", state)]


def test_ppo_checkpoint_keeps_using_ordinary_loader_during_inference():
    policy = _OrdinaryPolicy()
    state = {"last_phase": "train"}

    result = helpers._load_policy_checkpoint(
        policy,
        state,
        inference_only=True,
    )

    assert result == ["ordinary"]
    assert policy.calls == [state]


def test_other_algorithm_checkpoint_keeps_its_existing_loader_contract():
    policy = _OrdinaryPolicy()
    state = {"training_algorithm": "vaic_ppo_bc_dagger_student_v1"}

    result = helpers._load_policy_checkpoint(
        policy,
        state,
        inference_only=True,
    )

    assert result == ["ordinary"]
    assert policy.calls == [state]


def test_algorithm_checkpoint_requires_matching_inference_capable_policy():
    policy = _OrdinaryPolicy()
    state = {"training_algorithm": "distributional_fastsac_teacher_bc_v1"}

    with pytest.raises(ValueError, match="matching algo config"):
        helpers._load_policy_checkpoint(
            policy,
            state,
            inference_only=True,
        )

    assert policy.calls == []


def test_eval_entrypoint_explicitly_requests_inference_only_loading():
    source = (Path(__file__).parents[1] / "scripts" / "eval.py").read_text()

    assert "make_env_policy(cfg, inference_only=True)" in source


def test_legacy_fastsac_inference_config_uses_checkpoint_then_defaults():
    cfg = OmegaConf.create(
        {
            "algo": {
                "_target_": "legacy.FastSAC",
                "name": "legacy-name",
                "q_updates_per_rollout": 777,
                "teacher_prefill_rollouts": 10,
            }
        }
    )
    policy_state = {
        "training_algorithm": "distributional_fastsac_teacher_bc_v1",
        "dagger_backend_config": {
            "q_updates_per_rollout": 4,
            "sac_policy_frequency": 1,
            "sac_initial_action_std": 0.05,
            # Metadata is deliberately not a current dataclass field.
            "actor_output": "legacy-output-contract",
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg,
        policy_state,
        inference_only=True,
    )

    # Existing Hydra values always win, even over checkpoint metadata.
    assert cfg.algo._target_ == "legacy.FastSAC"
    assert cfg.algo.name == "legacy-name"
    assert cfg.algo.q_updates_per_rollout == 777
    assert cfg.algo.teacher_prefill_rollouts == 10
    # Missing historical values use the checkpoint contract before defaults.
    assert cfg.algo.sac_policy_frequency == 1
    assert cfg.algo.sac_initial_action_std == 0.05
    assert "sac_policy_frequency" in filled["checkpoint"]
    assert "sac_initial_action_std" in filled["checkpoint"]
    # Fields introduced after that checkpoint use current construction defaults.
    assert cfg.algo.teacher_perception_warmup_steps == 0
    assert cfg.algo.teacher_perception_replay_fraction == 0.0
    assert cfg.algo.perception_replay_mode == "online_student_rollout"
    assert cfg.algo.teacher_actor_replay_fraction == 0.5
    assert cfg.algo.teacher_perception_batch_size == 128
    assert cfg.algo.load_pretrained_perception is False
    assert cfg.algo.perception_checkpoint_path is None
    assert cfg.algo.train_perception is True
    assert "teacher_perception_warmup_steps" in filled["defaults"]
    assert "actor_output" not in cfg.algo


def test_physical_fastsac_checkpoint_overrides_structured_tanh_inference_default():
    cfg = OmegaConf.create(
        {"algo": {"sac_action_distribution": "normalized_tanh"}}
    )
    policy_state = {
        "training_algorithm": "distributional_fastsac_teacher_bc_v1",
        "dagger_backend_config": {
            "sac_action_distribution": "ppo_physical_gaussian",
            "sac_use_autotune": True,
            "load_noise_scale": 0.5,
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.sac_action_distribution == "ppo_physical_gaussian"
    assert cfg.algo.sac_use_autotune is True
    assert cfg.algo.load_noise_scale == pytest.approx(0.5)
    assert "sac_action_distribution" in filled["checkpoint"]


def test_predicted_effect_checkpoint_overrides_structured_q_inference_defaults():
    cfg = OmegaConf.create(
        {
            "algo": {
                "q_condition_on_actuator_state": False,
                "q_use_predicted_effect": False,
            }
        }
    )
    policy_state = {
        "training_algorithm": "distributional_tvkd_fastsac_teacher_bc_v9",
        "dagger_backend_config": {
            "q_condition_on_actuator_state": True,
            "q_use_predicted_effect": True,
            "value_norm": False,
        },
        "q_backend_config": {
            "q_actuator_context": {"enabled": True},
            "q_predicted_effect": {"enabled": True},
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.q_condition_on_actuator_state is True
    assert cfg.algo.q_use_predicted_effect is True
    assert "q_condition_on_actuator_state" in filled["checkpoint"]
    assert "q_use_predicted_effect" in filled["checkpoint"]


@pytest.mark.parametrize("critic_type", ("scalar", "distributional"))
def test_split_stem_checkpoint_restores_type_and_previous_action_state_context(
    critic_type,
):
    cfg = OmegaConf.create(
        {
            "algo": {
                "q_critic_type": "c51",
                "q_condition_on_actuator_state": False,
                "q_use_predicted_effect": False,
            }
        }
    )
    policy_state = {
        "training_algorithm": "distributional_tvkd_fastsac_teacher_bc_v9",
        "q_critic_type": critic_type,
        "dagger_backend_config": {
            "q_critic_type": critic_type,
            "q_condition_on_actuator_state": True,
            "q_use_predicted_effect": True,
            "value_norm": False,
        },
        "q_backend_config": {
            "q_actuator_context": {"enabled": True},
            "q_predicted_effect": {
                "enabled": False,
                "configured_context_collection": True,
                "previous_action_usage": (
                    "normalized_detached_scalar_state_context"
                ),
                "candidate_effect_features": "disabled",
            },
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.q_critic_type == critic_type
    assert cfg.algo.q_condition_on_actuator_state is True
    assert cfg.algo.q_use_predicted_effect is True
    assert "q_critic_type" in filled["checkpoint"]
    assert "q_use_predicted_effect" in filled["checkpoint"]


@pytest.mark.parametrize(
    "training_algorithm",
    (
        "distributional_fastsac_teacher_bc_v1",
        "distributional_tvkd_fastsac_teacher_bc_v9",
    ),
)
def test_fastsac_inference_restores_mean_q_twin_reduction_over_default_min(
    training_algorithm,
):
    cfg = OmegaConf.create(
        {
            "algo": {
                "q_twin_reduction": "min",
                "q_condition_on_actuator_state": False,
                "q_use_predicted_effect": False,
                "q_use_residual_film": False,
                "q_residual_film_scale": 0.1,
            }
        }
    )
    policy_state = {
        "training_algorithm": training_algorithm,
        "q_twin_reduction": "mean",
        "dagger_backend_config": {
            "q_twin_reduction": "mean",
            "value_norm": False,
        },
        "q_backend_config": {
            "q_twin_reduction": "mean",
            "q_actuator_context": {"enabled": False},
            "q_predicted_effect": {"enabled": False},
            "q_residual_film": {"enabled": False},
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.q_twin_reduction == "mean"
    assert "q_twin_reduction" in filled["checkpoint"]


def test_fastsac_inference_rejects_inconsistent_q_twin_reduction_metadata():
    cfg = OmegaConf.create({"algo": {"q_twin_reduction": "min"}})
    policy_state = {
        "training_algorithm": "distributional_fastsac_teacher_bc_v1",
        "q_twin_reduction": "mean",
        "q_backend_config": {"q_twin_reduction": "mean"},
        "dagger_backend_config": {"q_twin_reduction": "min"},
    }

    with pytest.raises(ValueError, match="inconsistent q_twin_reduction"):
        helpers._fill_replayless_inference_algo_defaults(
            cfg, policy_state, inference_only=True
        )


def test_physical_fastsac_inference_restores_sac_std_guard_contract():
    cfg = OmegaConf.create(
        {
            "algo": {
                "sac_action_distribution": "normalized_tanh",
                "sac_physical_std_lr": 1.0e-5,
                "sac_physical_std_max_kl": 0.01,
                "sac_physical_std_min": 0.05,
                "sac_physical_std_max": 0.5,
            }
        }
    )
    policy_state = {
        "training_algorithm": "distributional_fastsac_teacher_bc_v1",
        "dagger_backend_config": {
            "sac_action_distribution": "ppo_physical_gaussian",
            "sac_use_autotune": True,
            "load_noise_scale": 0.5,
            "sac_physical_std_lr": 2.0e-5,
            "sac_physical_std_max_kl": 0.02,
            "sac_physical_std_min": 0.04,
            "sac_physical_std_max": 0.55,
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.sac_physical_std_lr == pytest.approx(2.0e-5)
    assert cfg.algo.sac_physical_std_max_kl == pytest.approx(0.02)
    assert cfg.algo.sac_physical_std_min == pytest.approx(0.04)
    assert cfg.algo.sac_physical_std_max == pytest.approx(0.55)
    assert "sac_physical_std_lr" in filled["checkpoint"]
    assert "sac_physical_std_max_kl" in filled["checkpoint"]


def test_old_fastsac_checkpoint_without_distribution_defaults_to_normalized_tanh():
    cfg = OmegaConf.create(
        {"algo": {"sac_action_distribution": "ppo_physical_gaussian"}}
    )
    policy_state = {
        "training_algorithm": "distributional_fastsac_teacher_bc_v1",
        "dagger_backend_config": {},
    }

    helpers._fill_replayless_inference_algo_defaults(
        cfg, policy_state, inference_only=True
    )

    assert cfg.algo.sac_action_distribution == "normalized_tanh"


def test_legacy_td3_inference_config_preserves_none_and_false_values():
    cfg = OmegaConf.create(
        {
            "algo": {
                "dagger_beta_zero_iteration": None,
                "load_pretrained_perception": False,
                "q_action_input_gain": 3.0,
            }
        }
    )
    policy_state = {
        "training_algorithm": "distributional_td3_teacher_bc_v1",
        "dagger_backend_config": {
            "dagger_beta_zero_iteration": 500,
            "load_pretrained_perception": True,
            "q_action_input_gain": 1.5,
            "q_updates_per_rollout": 16,
        },
    }

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg,
        policy_state,
        inference_only=True,
    )

    assert cfg.algo.dagger_beta_zero_iteration is None
    assert cfg.algo.load_pretrained_perception is False
    assert cfg.algo.q_action_input_gain == 3.0
    assert cfg.algo.q_updates_per_rollout == 16
    assert "q_updates_per_rollout" in filled["checkpoint"]
    assert cfg.algo.teacher_perception_warmup_steps == 128


@pytest.mark.parametrize(
    ("inference_only", "algorithm"),
    (
        (False, "distributional_fastsac_teacher_bc_v1"),
        (True, "some_other_algorithm"),
    ),
)
def test_algo_default_fill_never_mutates_training_or_other_algorithms(
    inference_only,
    algorithm,
):
    cfg = OmegaConf.create({"algo": {"sentinel": 9}})
    before = OmegaConf.to_container(cfg, resolve=False)

    filled = helpers._fill_replayless_inference_algo_defaults(
        cfg,
        {"training_algorithm": algorithm, "dagger_backend_config": {}},
        inference_only=inference_only,
    )

    assert OmegaConf.to_container(cfg, resolve=False) == before
    assert filled == {"checkpoint": (), "defaults": ()}
