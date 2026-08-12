from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts import fastSAC_bc_dagger as sac_entry
from scripts.train import run_training as shared_run_training


def _cfg(*, checkpoint="/tmp/fresh_ppo.pt", iterations=3000):
    return OmegaConf.create(
        {
            "algo": {
                "name": sac_entry.EXPECTED_ALGO_NAME,
                "_target_": sac_entry.EXPECTED_ALGO_TARGET,
                "phase": "finetune",
                "vecnorm": "eval",
                "gamma": 0.99,
                "train_every": 32,
                "enable_residual_distillation": False,
                "use_object_adapt": True,
                "use_depth": True,
                "adapt_module": "gru",
                "latent_dim": 256,
                "in_keys": list(sac_entry.EXPECTED_ACTOR_IN_KEYS),
                "dagger_control_mode": "beta",
                "dagger_safe_takeover_rms": 0.006,
                "dagger_safe_release_rms": 0.004,
                "dagger_safe_min_teacher_steps": 8,
                "dagger_safe_zero_iteration": None,
                "dagger_beta_start": 0.0,
                "dagger_beta_end": 0.0,
                "dagger_beta_decay_rollouts": 1800,
                "dagger_beta_zero_iteration": None,
                "dagger_seed": 0,
                "dagger_teacher_action_threshold": 20.0,
                "dagger_action_clip": 20.0,
                "dagger_bc_lr": 3.0e-4,
                "dagger_actor_huber_delta": 1.0,
                "dagger_buffer_capacity": 131_072,
                "dagger_buffer_device": "cpu",
                "dagger_batch_size": 4096,
                "teacher_prefill_rollouts": 10,
                "dagger_replay_raw_observations": True,
                "replay_raw_observation_keys": list(
                    sac_entry.EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
                ),
                "perception_replay_burn_in": 8,
                "perception_encode_microbatch_size": 128,
                "perception_depth_codec": "uint8_div_100_v1",
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
                "q_updates_per_rollout": 32,
                "q_teacher_replay_ratio": 0.5,
                "q_teacher_buffer_capacity": 131_072,
                "eta_td3": 0.0,
                "policy_delay": 2,
                "target_policy_noise_std": 0.0,
                "target_policy_noise_clip": 0.0,
                "collector_exploration_noise_std": 0.0,
                "collector_exploration_noise_clip": 0.0,
                "td3_learning_starts": 8192,
                "eta_sac": 1.0e-4,
                "lambda_bc": 1.0,
                "sac_actor_lr": 3.0e-4,
                "sac_initial_action_std": 0.1,
                "sac_log_std_min": -10.0,
                "sac_log_std_max": -2.0,
                "sac_alpha_init": 1.0e-5,
                "sac_alpha_lr": 2.0e-5,
                "sac_use_autotune": True,
                "sac_target_entropy_ratio": 1.0,
                "sac_policy_frequency": 2,
                "sac_learning_starts": 8192,
                "sac_tau": 0.005,
                "sac_max_grad_norm": 1.0,
                "save_teacher_buffer": False,
            },
            "task": {"name": "G1Skateboard", "num_envs": 512},
            "total_frames": 1,
            "fastsac_dagger_iterations": iterations,
            "checkpoint_path": checkpoint,
            "fastsac_bc_dagger_checkpoint": None,
            "bc_dagger_checkpoint": None,
            "teacher_replay_buffer_path": None,
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


def test_fastsac_config_composes_with_stochastic_mean_bc_contract():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="fastSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=3000",
            ],
        )

    assert cfg.algo.name == "fastsac_bc_dagger"
    assert cfg.algo._target_ == sac_entry.EXPECTED_ALGO_TARGET
    assert cfg.task.num_envs == 512
    assert cfg.algo.teacher_prefill_rollouts == 10
    assert cfg.algo.dagger_beta_start == pytest.approx(0.0)
    assert cfg.algo.dagger_beta_end == pytest.approx(0.0)
    assert cfg.algo.eta_sac == pytest.approx(1.0e-4)
    assert cfg.algo.lambda_bc == pytest.approx(1.0)
    assert cfg.algo.sac_initial_action_std == pytest.approx(0.1)
    assert cfg.algo.sac_use_autotune is True
    assert cfg.algo.q_batch_size == 512
    assert cfg.algo.q_updates_per_rollout == 32
    assert cfg.algo.save_teacher_buffer is False
    assert cfg.algo.target_policy_noise_std == 0.0
    assert cfg.algo.collector_exploration_noise_std == 0.0

    sac_entry.validate_fastsac_bc_dagger_config(cfg)
    assert cfg.total_frames == 3010 * 512 * 32
    assert sac_entry.fastsac_dagger_rollout_schedule(cfg) == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 3000,
        "main_rollouts": 3000,
        "prefill_rollouts": 10,
        "physical_rollouts": 3010,
        "start_rollout": 0,
        "end_rollout": 3000,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 3000,
        "safe_zero_rollouts": 0,
    }


def test_fastsac_yaml_environment_count_can_be_overridden():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="fastSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=3000",
                "task.num_envs=256",
            ],
        )

    sac_entry.validate_fastsac_bc_dagger_config(cfg)
    assert cfg.task.num_envs == 256
    assert cfg.total_frames == 3010 * 256 * 32


def test_explicit_main_rollout_budget_is_required():
    cfg = _cfg(iterations=None)
    with pytest.raises(ValueError, match="explicit positive fastsac_dagger_iterations"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_pure_student_main_rollout_requires_teacher_prefill():
    cfg = _cfg()
    cfg.algo.teacher_prefill_rollouts = 0
    with pytest.raises(ValueError, match="teacher_prefill_rollouts > 0"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


@pytest.mark.parametrize(
    "field",
    (
        "target_policy_noise_std",
        "target_policy_noise_clip",
        "collector_exploration_noise_std",
        "collector_exploration_noise_clip",
    ),
)
def test_inherited_td3_noise_must_remain_zero(field):
    cfg = _cfg()
    cfg.algo[field] = 0.1
    with pytest.raises(ValueError, match=f"{field}=0"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("eta_sac", -1.0, "non-negative"),
        ("sac_actor_lr", 0.0, "positive"),
        ("sac_initial_action_std", 0.0, "positive"),
        ("sac_log_std_min", -1.0, "below"),
        ("sac_alpha_init", 0.0, "positive"),
        ("sac_target_entropy_ratio", 1.1, r"\(0, 1\]"),
        ("sac_policy_frequency", 0, "positive"),
        ("sac_learning_starts", 0, "positive"),
        ("sac_tau", 0.0, "sac_tau"),
        ("q_num_atoms", 51, "501"),
        ("q_batch_size", 3, "even"),
        ("q_updates_per_rollout", 0, "positive"),
        ("q_teacher_replay_ratio", 0.4, "50/50"),
    ),
)
def test_invalid_fastsac_controls_fail_before_training(field, value, message):
    cfg = _cfg()
    cfg.algo[field] = value
    with pytest.raises(ValueError, match=message):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_sac_and_bc_weights_cannot_both_be_zero():
    cfg = _cfg()
    cfg.algo.eta_sac = 0.0
    cfg.algo.lambda_bc = 0.0
    with pytest.raises(ValueError, match="cannot both be zero"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_inherited_aliases_cannot_disagree_with_sac_controls():
    cfg = _cfg()
    cfg.algo.policy_delay = 3
    with pytest.raises(ValueError, match="policy_delay.*sac_policy_frequency"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_h5_replay_export_and_input_are_rejected():
    cfg = _cfg()
    cfg.algo.save_teacher_buffer = True
    with pytest.raises(ValueError, match="teacher_replay_buffer.h5 export is disabled"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)

    cfg = _cfg()
    with open_dict(cfg.algo):
        cfg.algo.teacher_buffer_path = "/tmp/teacher_replay_buffer.h5"
    with pytest.raises(ValueError, match="does not accept a teacher replay H5 path"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_fresh_source_accepts_only_compatible_ppo_teacher(tmp_path):
    source = _write_fresh_ppo_checkpoint(tmp_path / "checkpoint_6000.pt")
    cfg = _cfg(checkpoint=str(source))

    prepared = sac_entry.prepare_fresh_fastsac_bc_dagger_source(cfg)

    assert prepared == {
        "path": str(source.resolve()),
        "source_last_iter": 6000,
    }
    assert cfg.checkpoint_path == str(source.resolve())
    assert cfg._bc_dagger_fresh_source is True
    assert cfg._bc_dagger_model_only_resume is False


def test_fresh_source_rejects_staged_checkpoint(tmp_path):
    source = _write_fresh_ppo_checkpoint(tmp_path / "wrong_source.pt")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["policy"]["training_algorithm"] = sac_entry.EXPECTED_TRAINING_ALGORITHM
    torch.save(checkpoint, source)
    cfg = _cfg(checkpoint=str(source))

    with pytest.raises(ValueError, match="fresh PPO teacher"):
        sac_entry.prepare_fresh_fastsac_bc_dagger_source(cfg)


def test_same_stage_resume_is_rejected():
    cfg = _cfg()
    cfg.fastsac_bc_dagger_checkpoint = "/tmp/fastsac.pt"
    with pytest.raises(ValueError, match="same-stage FastSAC resume"):
        sac_entry.prepare_fastsac_bc_dagger_checkpoint(cfg)
    with pytest.raises(ValueError, match="same-stage FastSAC resume"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_entrypoint_reuses_shared_training_engine(monkeypatch):
    assert sac_entry.run_training is shared_run_training
    cfg = _cfg()
    received = []
    sources = []
    monkeypatch.setattr(
        sac_entry,
        "prepare_fresh_fastsac_bc_dagger_source",
        lambda actual: sources.append(actual) or {"path": "/fresh/ppo.pt"},
    )
    monkeypatch.setattr(
        sac_entry,
        "run_training",
        lambda actual: received.append(actual) or "shared-result",
    )

    result = sac_entry.main.__wrapped__(cfg)

    assert result == "shared-result"
    assert sources == [cfg]
    assert received == [cfg]
