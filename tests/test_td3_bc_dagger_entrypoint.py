import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts import TD3_bc_dagger as td3_entry
from scripts.train import run_training as shared_run_training


def _cfg(
    *,
    checkpoint="/tmp/fresh_ppo.pt",
    resume=None,
    iterations=None,
    total_frames=39_321_600,
    control_mode="safe",
    beta_zero_iteration=None,
    safe_zero_iteration=None,
):
    return OmegaConf.create(
        {
            "algo": {
                "name": td3_entry.EXPECTED_ALGO_NAME,
                "_target_": td3_entry.EXPECTED_ALGO_TARGET,
                "phase": "finetune",
                "vecnorm": "eval",
                "gamma": 0.99,
                "train_every": 32,
                "enable_residual_distillation": False,
                "use_object_adapt": True,
                "use_depth": True,
                "adapt_module": "gru",
                "latent_dim": 256,
                "in_keys": list(td3_entry.EXPECTED_ACTOR_IN_KEYS),
                "dagger_control_mode": control_mode,
                "dagger_safe_takeover_rms": 0.006,
                "dagger_safe_release_rms": 0.004,
                "dagger_safe_min_teacher_steps": 8,
                "dagger_safe_zero_iteration": safe_zero_iteration,
                "dagger_beta_start": 1.0,
                "dagger_beta_end": 0.0,
                "dagger_beta_decay_rollouts": 1800,
                "dagger_beta_zero_iteration": beta_zero_iteration,
                "dagger_seed": 0,
                "dagger_teacher_action_threshold": 20.0,
                "dagger_action_clip": 20.0,
                "dagger_bc_lr": 3.0e-4,
                "dagger_actor_huber_delta": 1.0,
                "dagger_buffer_capacity": 131_072,
                "dagger_buffer_device": "cpu",
                "dagger_batch_size": 4096,
                "dagger_replay_raw_observations": True,
                "replay_raw_observation_keys": list(
                    td3_entry.EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
                ),
                "eta_td3": 1.0,
                "lambda_bc": 1.0,
                "policy_delay": 2,
                "target_policy_noise_std": 0.2,
                "target_policy_noise_clip": 0.5,
                "collector_exploration_noise_std": 0.1,
                "collector_exploration_noise_clip": 0.5,
                "td3_learning_starts": 8192,
                "q_hidden_dim": 768,
                "q_num_atoms": 501,
                "q_v_min": -20.0,
                "q_v_max": 20.0,
                "q_layer_norm": True,
                "q_action_fusion": "late",
                "q_action_coordinates": "absolute",
                "q_normalize_actions": True,
                "q_action_input_gain": 1.0,
                "q_lr": 3.0e-5,
                "q_weight_decay": 1.0e-3,
                "q_seed": 0,
                "q_tau": 0.005,
                "q_max_grad_norm": 1.0,
                "q_batch_size": 512,
                "q_updates_per_rollout": 128,
                "q_teacher_replay_ratio": 0.5,
                "q_teacher_buffer_capacity": 131_072,
                "teacher_buffer_filename": "teacher_replay_buffer.h5",
                "teacher_buffer_path": None,
                "save_teacher_buffer": True,
                "teacher_buffer_capacity": 131_072,
                "teacher_buffer_snapshot_chunk_rows": 4096,
                # Inert fields inherited solely from PPOConfig.
                "entropy_coef_start": 0.001,
                "entropy_coef_end": 0.001,
                "entropy_decay_iters": 1000,
                "init_noise_scale": 1.0,
                "load_noise_scale": 0.5,
            },
            "task": {"name": "G1Skateboard", "num_envs": 512},
            "total_frames": total_frames,
            "td3_dagger_iterations": iterations,
            "td3_dagger_resume_rollout_count": 0,
            "td3_dagger_resume_environment_steps": 0,
            "checkpoint_path": checkpoint,
            "td3_bc_dagger_checkpoint": resume,
            "td3_bc_dagger_copy_teacher_replay": True,
            "teacher_replay_buffer_path": None,
            "_td3_bc_dagger_same_stage_resume": False,
            "_bc_dagger_fresh_source": False,
            "_bc_dagger_model_only_resume": False,
        }
    )


def _write_fresh_ppo_checkpoint(path: Path) -> Path:
    policy = {
        "last_phase": "train",
        "last_iter": 6000,
        "actor": {},
        "actor_adapt": {},
        "encoder_priv": {},
        "adapt_module": {},
        "adapt_ema": {},
        "critic": {},
        "object_adapt": {},
        "object_adapt_ema": {},
    }
    source_cfg = OmegaConf.create(
        {
            "task": {"name": "G1Skateboard"},
            "algo": {
                "name": "ppo_vel",
                "_target_": "active_adaptation.learning.ppo.ppo_vel.PPOVEL",
                "phase": "train",
                "enable_residual_distillation": True,
                "use_object_adapt": True,
                "adapt_module": "gru",
                "latent_dim": 256,
            },
        }
    )
    torch.save({"policy": policy, "vecnorm": {}, "cfg": source_cfg}, path)
    return path


def _test_action_contract() -> dict:
    action_dim = len(td3_entry.EXPECTED_JOINT_NAMES)
    payload = {
        "semantics": td3_entry.EXPECTED_ACTION_CONTRACT_SEMANTICS,
        "joint_names": list(td3_entry.EXPECTED_JOINT_NAMES),
        "action_low": [-20.0] * action_dim,
        "action_high": [20.0] * action_dim,
        "q_action_center": [0.0] * action_dim,
        "q_action_scale": [1.0] * action_dim,
        "q_action_clamp": None,
        "execution_support_fingerprint": "sha256:test-execution",
        "q_action_transform_fingerprint": "sha256:test-q-transform",
        "entropy_reference_fingerprint": "sha256:test-entropy",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **payload,
        "fingerprint": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }


def _write_td3_checkpoint(
    path: Path,
    *,
    algorithm=td3_entry.EXPECTED_TRAINING_ALGORITHM,
    version=td3_entry.EXPECTED_CHECKPOINT_VERSION,
    replay_size=0,
    remove=None,
    algo=None,
) -> Path:
    generator_state = torch.Generator().get_state()
    algo = _cfg().algo if algo is None else algo
    action_contract = _test_action_contract()
    policy = {
        "training_algorithm": algorithm,
        "checkpoint_version": version,
        "actor_backend": td3_entry.EXPECTED_ACTOR_BACKEND,
        "critic_learning_semantics": td3_entry.EXPECTED_CRITIC_SEMANTICS,
        "actor_learning_semantics": (td3_entry.EXPECTED_ACTOR_LEARNING_SEMANTICS),
        "dagger_backend_config": td3_entry._expected_dagger_backend_config(algo),
        "q_backend_config": {
            "actor_obs_keys": ["vel_command", "policy", "priv_pred"],
            "critic_obs_keys": ["priv", "policy", "command", "object_"],
            "actor_obs_dim": 525,
            "critic_obs_dim": 2341,
            "action_dim": len(td3_entry.EXPECTED_JOINT_NAMES),
            "hidden_dim": int(algo.q_hidden_dim),
            "q_action_fusion": str(algo.q_action_fusion),
            "num_atoms": int(algo.q_num_atoms),
            "v_min": float(algo.q_v_min),
            "v_max": float(algo.q_v_max),
            "layer_norm": bool(algo.q_layer_norm),
            "gamma": float(algo.gamma),
            "q_action_coordinates": str(algo.q_action_coordinates),
            "q_action_normalized": bool(algo.q_normalize_actions),
            "q_action_input_gain": float(algo.q_action_input_gain),
            "target_semantics": td3_entry.EXPECTED_CRITIC_SEMANTICS,
            "actor_q_reduction": "online_q1_expectation_only",
            "replay_mix_semantics": (
                "beta_independent_teacher_executed_0.5_student_executed_0.5_v1"
            ),
            "q_action_transform_fingerprint": action_contract[
                "q_action_transform_fingerprint"
            ],
        },
        "action_contract": action_contract,
        "vecnorm_fingerprint": "sha256:" + "0" * 64,
        "actor": {},
        "actor_adapt": {},
        "actor_target": {},
        "adapt_ema": {},
        "adapt_module": {},
        "depth_cnn": {},
        "encoder_priv": {},
        "object_adapt": {},
        "object_adapt_ema": {},
        "qnet": {},
        "qnet_target": {},
        "temporal_depth_gru": {},
        "temporal_depth_gru_ema": {},
        "optimizer_resume_state": {
            "actor_optimizer": {},
            "critic_optimizer": {},
            "adapt_optimizer": {},
        },
        "actor_update_count": 10,
        "critic_update_count": 20,
        "dagger_rng_state": generator_state,
        "q_rng_state": generator_state,
        "collector_exploration_rng_state": generator_state,
        "target_policy_rng_state": generator_state,
        "dagger_rollout_count": 12,
        "dagger_environment_steps": 384,
        "teacher_replay_state": {"size": replay_size},
    }
    if remove in policy:
        policy.pop(remove)
    elif remove in policy["optimizer_resume_state"]:
        policy["optimizer_resume_state"].pop(remove)
    torch.save({"policy": policy, "vecnorm": {}}, path)
    return path


def test_td3_config_composes_without_simulator_startup():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TD3_bc_dagger",
            overrides=["task=G1/vaic/skateboard_stu"],
        )

    assert cfg.algo.name == "td3_bc_dagger"
    assert cfg.algo._target_ == td3_entry.EXPECTED_ALGO_TARGET
    assert cfg.algo.phase == "finetune"
    assert cfg.algo.vecnorm == "eval"
    assert list(cfg.algo.in_keys) == [
        "command",
        "policy",
        "object_",
        "priv",
        "object_geo_",
        "vel_command",
        "depth",
    ]
    assert cfg.algo.q_num_atoms == 501
    assert cfg.algo.q_v_min == pytest.approx(-20.0)
    assert cfg.algo.q_v_max == pytest.approx(20.0)
    assert cfg.algo.q_action_coordinates == "absolute"
    assert cfg.algo.q_normalize_actions is True
    assert cfg.algo.q_action_input_gain == pytest.approx(1.0)
    assert cfg.algo.eta_td3 == pytest.approx(1.0)
    assert cfg.algo.lambda_bc == pytest.approx(1.0)
    assert cfg.algo.policy_delay == 2
    assert cfg.algo.target_policy_noise_std == pytest.approx(0.2)
    assert cfg.algo.target_policy_noise_clip == pytest.approx(0.5)
    assert cfg.algo.collector_exploration_noise_std == pytest.approx(0.1)
    assert cfg.algo.collector_exploration_noise_clip == pytest.approx(0.5)
    assert cfg.algo.td3_learning_starts == 8192
    assert cfg.algo.q_tau == pytest.approx(0.005)
    assert cfg.algo.dagger_buffer_capacity == 131_072
    assert cfg.algo.q_teacher_buffer_capacity == 131_072
    assert cfg.algo.teacher_buffer_capacity == 131_072
    assert cfg.algo.q_batch_size == 512
    assert cfg.algo.q_updates_per_rollout == 128
    assert cfg.wandb.project == "vaic_dagger"

    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 2400,
        "start_rollout": 0,
        "end_rollout": 2400,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 600,
        "safe_zero_rollouts": 0,
    }


def test_recommended_3000_rollout_command_composes_safe_export_capacity():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TD3_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "td3_dagger_iterations=3000",
                "algo.dagger_control_mode=beta",
                "algo.dagger_beta_start=1",
                "algo.dagger_beta_end=0",
                "algo.collector_exploration_noise_std=0.1",
                "algo.teacher_buffer_capacity=131072",
                "save_interval=100",
                "wandb.mode=online",
            ],
        )

    td3_entry.validate_td3_bc_dagger_config(cfg)
    assert cfg.total_frames == 49_152_000
    assert cfg.algo.teacher_buffer_capacity == 131_072
    assert cfg.algo.dagger_buffer_capacity == 131_072
    assert cfg.algo.q_teacher_buffer_capacity == 131_072
    assert cfg.algo.q_batch_size == 512
    assert cfg.algo.q_updates_per_rollout == 128
    assert cfg.algo.policy_delay == 2
    assert td3_entry.td3_dagger_rollout_schedule(cfg) == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 3000,
        "start_rollout": 0,
        "end_rollout": 3000,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 1200,
        "safe_zero_rollouts": 0,
    }


def test_yaml_adds_no_forbidden_stochastic_policy_fields():
    config_path = Path(__file__).resolve().parents[1] / "cfg" / "TD3_bc_dagger.yaml"
    yaml_cfg = OmegaConf.load(config_path)
    assert td3_entry._forbidden_algo_fields(yaml_cfg.algo) == []

    composed = _cfg()
    assert td3_entry._forbidden_algo_fields(composed.algo) == []
    assert td3_entry.INERT_PPO_COMPATIBILITY_FIELDS.issubset(composed.algo)
    assert not any(str(key).startswith("sac_") for key in composed.algo)


def test_valid_config_allows_only_inert_inherited_ppo_fields():
    cfg = _cfg()
    td3_entry.validate_td3_bc_dagger_config(cfg)


def test_persistent_teacher_export_must_cover_online_teacher_ring():
    cfg = _cfg()
    cfg.algo.teacher_buffer_capacity = cfg.algo.q_teacher_buffer_capacity - 1
    with pytest.raises(ValueError, match="refill the complete online Teacher"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


@pytest.mark.parametrize(
    "field",
    (
        "sac_alpha_init",
        "target_entropy",
        "policy_log_std",
        "sample_log_prob",
        "entropy_bonus",
    ),
)
def test_config_rejects_forbidden_stochastic_policy_fields(field):
    cfg = _cfg()
    with open_dict(cfg.algo):
        cfg.algo[field] = 0.0
    with pytest.raises(ValueError, match="stochastic-policy fields"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("eta_td3", -1.0, "non-negative"),
        ("use_depth", False, "depth encoder"),
        ("use_object_adapt", False, "object adaptation"),
        ("adapt_module", "mlp", "adapt_module=gru"),
        ("latent_dim", 128, "latent_dim=256"),
        ("in_keys", ["policy", "command"], "keys/order"),
        (
            "replay_raw_observation_keys",
            ["policy", "vel_command", "priv", "command"],
            "keys/order",
        ),
        ("lambda_bc", float("nan"), "non-negative"),
        ("policy_delay", 0, "positive"),
        ("target_policy_noise_std", -0.1, "non-negative"),
        ("collector_exploration_noise_clip", -0.1, "non-negative"),
        ("td3_learning_starts", 0, "positive"),
        ("q_tau", 0.0, "q_tau"),
        ("q_num_atoms", 51, "501"),
        ("q_v_min", -10.0, "q_v_min"),
        ("q_v_max", 10.0, "q_v_max"),
        ("q_action_fusion", "early", "late"),
        ("q_layer_norm", False, "LayerNorm"),
        ("q_action_coordinates", "unit", "absolute"),
        ("q_normalize_actions", False, "normalized"),
        ("q_action_input_gain", 2.0, "input_gain"),
        ("q_batch_size", 3, "even"),
        ("q_updates_per_rollout", 0, "positive"),
        ("q_teacher_replay_ratio", 0.4, "50/50"),
        ("dagger_bc_lr", 0.0, "positive"),
        ("q_weight_decay", -1.0, "non-negative"),
        ("teacher_buffer_filename", "../replay.h5", "basename"),
    ),
)
def test_invalid_phase1_hyperparameters_fail_before_training(field, value, message):
    cfg = _cfg()
    cfg.algo[field] = value
    with pytest.raises(ValueError, match=message):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_both_actor_objective_weights_cannot_be_zero():
    cfg = _cfg()
    cfg.algo.eta_td3 = 0.0
    cfg.algo.lambda_bc = 0.0
    with pytest.raises(ValueError, match="cannot both be zero"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_residual_distillation_fails_before_training():
    cfg = _cfg()
    cfg.algo.enable_residual_distillation = True
    with pytest.raises(ValueError, match="only Actor optimizer"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_teacher_validity_threshold_cannot_exceed_execution_clip():
    cfg = _cfg()
    cfg.algo.dagger_teacher_action_threshold = 20.1
    with pytest.raises(ValueError, match="cannot exceed"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_iteration_alias_preserves_existing_categorical_beta_schedule():
    cfg = _cfg(
        iterations=1200,
        total_frames=1,
        control_mode="hybrid",
        beta_zero_iteration=900,
        safe_zero_iteration=1000,
    )
    td3_entry.validate_td3_bc_dagger_config(cfg)
    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)

    assert cfg.total_frames == 1200 * 512 * 32
    assert cfg.algo.dagger_beta_decay_rollouts == 900
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 1200,
        "start_rollout": 0,
        "end_rollout": 1200,
        "decay_rollouts": 900,
        "beta_zero_rollouts": 300,
        "safe_zero_rollouts": 200,
    }


def test_fresh_source_accepts_only_compatible_ppo_teacher(tmp_path):
    source = _write_fresh_ppo_checkpoint(tmp_path / "checkpoint_6000.pt")
    cfg = _cfg(checkpoint=str(source))

    prepared = td3_entry.prepare_fresh_td3_bc_dagger_source(cfg)

    assert prepared == {
        "path": str(source.resolve()),
        "source_last_iter": 6000,
    }
    assert cfg.checkpoint_path == str(source.resolve())
    assert cfg._bc_dagger_fresh_source is True
    assert cfg._bc_dagger_model_only_resume is False
    assert cfg._td3_bc_dagger_same_stage_resume is False


def test_fresh_source_rejects_a_staged_checkpoint(tmp_path):
    source = _write_fresh_ppo_checkpoint(tmp_path / "wrong_source.pt")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["policy"]["training_algorithm"] = td3_entry.EXPECTED_TRAINING_ALGORITHM
    torch.save(checkpoint, source)
    cfg = _cfg(checkpoint=str(source))

    with pytest.raises(ValueError, match="fresh PPO teacher"):
        td3_entry.prepare_fresh_td3_bc_dagger_source(cfg)


def test_same_stage_resume_requires_td3_marker_and_full_learning_state(tmp_path):
    checkpoint = _write_td3_checkpoint(tmp_path / "checkpoint_12.pt")
    cfg = _cfg(
        checkpoint="/old/fresh_ppo.pt",
        resume=str(checkpoint),
        iterations=8,
        total_frames=1,
    )

    prepared = td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)
    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)

    assert prepared == {
        "path": str(checkpoint.resolve()),
        "rollout_count": 12,
        "environment_steps": 384,
        "teacher_replay_source": None,
    }
    assert cfg.checkpoint_path == str(checkpoint.resolve())
    assert cfg.td3_dagger_resume_rollout_count == 12
    assert cfg.td3_dagger_resume_environment_steps == 384
    assert cfg._td3_bc_dagger_same_stage_resume is True
    assert cfg._bc_dagger_model_only_resume is False
    assert schedule["start_rollout"] == 12
    assert schedule["total_rollouts"] == 8
    assert schedule["end_rollout"] == 20


def test_resume_schedule_validates_against_cumulative_rollout_count(tmp_path):
    cfg = _cfg(
        checkpoint=None,
        resume=None,
        iterations=8,
        total_frames=1,
        control_mode="hybrid",
        beta_zero_iteration=15,
    )
    td3_entry.apply_td3_dagger_iteration_controls(cfg)
    checkpoint = _write_td3_checkpoint(tmp_path / "checkpoint_12.pt", algo=cfg.algo)
    cfg.td3_bc_dagger_checkpoint = str(checkpoint)

    td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)
    td3_entry.validate_td3_bc_dagger_config(cfg)

    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)
    assert schedule["start_rollout"] == 12
    assert schedule["end_rollout"] == 20
    assert schedule["beta_zero_rollouts"] == 5


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    (
        (
            {"algorithm": "vaic_ppo_bc_dagger_student_sac_critic_v6"},
            "training_algorithm",
        ),
        ({"version": 2}, "checkpoint_version"),
        ({"remove": "actor_target"}, "trained modules"),
        ({"remove": "critic_optimizer"}, "optimizer state"),
        ({"remove": "target_policy_rng_state"}, "continuation state"),
        ({"remove": "dagger_rollout_count"}, "continuation state"),
        ({"remove": "dagger_environment_steps"}, "continuation state"),
        ({"remove": "critic_learning_semantics"}, "Critic semantics"),
        ({"remove": "dagger_backend_config"}, "backend config"),
        ({"remove": "q_backend_config"}, "Q backend config"),
        ({"remove": "action_contract"}, "action contract"),
        ({"remove": "vecnorm_fingerprint"}, "VecNorm fingerprint"),
    ),
)
def test_same_stage_resume_rejects_incompatible_checkpoint(
    tmp_path, checkpoint_kwargs, message
):
    checkpoint = _write_td3_checkpoint(tmp_path / "invalid.pt", **checkpoint_kwargs)
    cfg = _cfg(checkpoint=None, resume=str(checkpoint))
    with pytest.raises(ValueError, match=message):
        td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)


def test_resume_rejects_legacy_checkpoint_version_alias(tmp_path):
    checkpoint = _write_td3_checkpoint(tmp_path / "invalid_alias.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["policy"]["td3_checkpoint_version"] = state["policy"].pop(
        "checkpoint_version"
    )
    torch.save(state, checkpoint)
    cfg = _cfg(checkpoint=None, resume=str(checkpoint))

    with pytest.raises(ValueError, match="checkpoint_version"):
        td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)


def test_resume_with_teacher_rows_requires_paired_h5(tmp_path):
    checkpoint = _write_td3_checkpoint(
        tmp_path / "checkpoint_with_replay.pt", replay_size=10
    )
    cfg = _cfg(checkpoint=None, resume=str(checkpoint))
    with pytest.raises(FileNotFoundError, match="paired immutable teacher replay"):
        td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)


def test_entrypoint_reuses_shared_training_engine(monkeypatch):
    assert td3_entry.run_training is shared_run_training
    cfg = _cfg()
    received = []
    sources = []
    monkeypatch.setattr(
        td3_entry,
        "prepare_fresh_td3_bc_dagger_source",
        lambda actual: sources.append(actual) or {"path": "/fresh/ppo.pt"},
    )
    monkeypatch.setattr(
        td3_entry,
        "run_training",
        lambda actual: received.append(actual) or "shared-result",
    )

    result = td3_entry.main.__wrapped__(cfg)

    assert result == "shared-result"
    assert sources == [cfg]
    assert received == [cfg]
