from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts import TD3_bc_dagger as td3_entry
from scripts.train import (
    _bc_dagger_checkpoint_index,
    _bc_dagger_main_budget_complete,
    run_training as shared_run_training,
)


def _cfg(
    *,
    checkpoint="/tmp/fresh_ppo.pt",
    resume=None,
    iterations=2400,
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
                "teacher_prefill_max_rollouts": 10,
                "teacher_actor_replay_fraction": 0.0,
                "teacher_perception_replay_fraction": 0.0,
                "failure_phase_teacher_fraction": 0.0,
                "failure_phase_lookback_steps": 50,
                "failure_phase_samples_per_failure": 10,
                "failure_phase_num_bins": 1024,
                "dagger_replay_raw_observations": True,
                "replay_raw_observation_keys": list(
                    td3_entry.EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
                ),
                "perception_replay_burn_in": 8,
                "perception_encode_microbatch_size": 128,
                "teacher_perception_batch_size": 128,
                "perception_depth_codec": "uint8_div_100_v1",
                "load_pretrained_perception": False,
                "perception_checkpoint_path": None,
                "train_perception": True,
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
                "save_teacher_buffer": False,
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
            "checkpoint_path": checkpoint,
            "td3_bc_dagger_checkpoint": resume,
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


def _write_perception_checkpoint(path: Path, *, omit: str | None = None) -> Path:
    policy = {name: {} for name in td3_entry.REQUIRED_PRETRAINED_PERCEPTION_MODULES}
    if omit is not None:
        policy.pop(omit)
    torch.save({"policy": policy}, path)
    return path


def test_td3_config_composes_without_simulator_startup():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TD3_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "td3_dagger_iterations=2400",
            ],
        )

    assert cfg.algo.name == "td3_bc_dagger"
    assert cfg.algo._target_ == td3_entry.EXPECTED_ALGO_TARGET
    assert cfg.task.num_envs == 256
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
    assert cfg.algo.teacher_prefill_max_rollouts == 1000
    assert "teacher_prefill_rollouts" not in cfg.algo
    assert cfg.algo.teacher_actor_replay_fraction == pytest.approx(0.5)
    assert cfg.algo.teacher_perception_replay_fraction == pytest.approx(0.5)
    assert cfg.algo.failure_phase_teacher_fraction == pytest.approx(0.3)
    assert cfg.algo.failure_phase_lookback_steps == 50
    assert cfg.algo.failure_phase_samples_per_failure == 10
    assert cfg.algo.failure_phase_num_bins == 1024
    assert (
        cfg.algo.teacher_actor_replay_fraction * cfg.algo.failure_phase_teacher_fraction
        == pytest.approx(0.15)
    )
    assert cfg.algo.teacher_actor_replay_fraction * (
        1.0 - cfg.algo.failure_phase_teacher_fraction
    ) == pytest.approx(0.35)
    assert list(cfg.algo.replay_raw_observation_keys) == [
        "vel_command",
        "policy",
        "priv",
        "command",
        "depth",
    ]
    assert cfg.algo.perception_replay_burn_in == 8
    assert cfg.algo.perception_encode_microbatch_size == 128
    assert cfg.algo.teacher_perception_batch_size == 128
    assert cfg.algo.perception_depth_codec == "uint8_div_100_v1"
    assert cfg.algo.load_pretrained_perception is False
    assert cfg.algo.perception_checkpoint_path is None
    assert cfg.algo.train_perception is True
    assert cfg.algo.save_teacher_buffer is False
    assert "teacher_buffer_capacity" not in cfg.algo
    assert "teacher_buffer_filename" not in cfg.algo
    assert "teacher_buffer_path" not in cfg.algo
    assert cfg.algo.q_batch_size == 512
    assert cfg.algo.q_updates_per_rollout == 128
    assert cfg.wandb.project == "vaic_dagger"

    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)
    assert cfg._bc_dagger_main_rollout_budget == 2400
    assert schedule == {
        "frames_per_rollout": 8_192,
        "total_rollouts": 2400,
        "main_rollouts": 2400,
        "prefill_max_rollouts": 1000,
        "prefill_target_rows": 131_072,
        "max_physical_rollouts": 3400,
        "start_rollout": 0,
        "end_rollout": 2400,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 600,
        "safe_zero_rollouts": 0,
    }


def test_recommended_3000_rollout_command_composes_raw_perception_replay():
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
                "save_interval=100",
                "wandb.mode=online",
            ],
        )

    td3_entry.validate_td3_bc_dagger_config(cfg)
    assert cfg.total_frames == 4000 * 256 * 32
    assert cfg._bc_dagger_main_rollout_budget == 3000
    assert cfg.algo.dagger_buffer_capacity == 131_072
    assert cfg.algo.q_teacher_buffer_capacity == 131_072
    assert cfg.algo.perception_replay_burn_in == 8
    assert cfg.algo.perception_encode_microbatch_size == 128
    assert cfg.algo.teacher_perception_batch_size == 128
    assert cfg.algo.perception_depth_codec == "uint8_div_100_v1"
    assert cfg.algo.save_teacher_buffer is False
    assert cfg.algo.q_batch_size == 512
    assert cfg.algo.q_updates_per_rollout == 128
    assert cfg.algo.policy_delay == 2
    assert td3_entry.td3_dagger_rollout_schedule(cfg) == {
        "frames_per_rollout": 8_192,
        "total_rollouts": 3000,
        "main_rollouts": 3000,
        "prefill_max_rollouts": 1000,
        "prefill_target_rows": 131_072,
        "max_physical_rollouts": 4000,
        "start_rollout": 0,
        "end_rollout": 3000,
        "decay_rollouts": 1800,
        "beta_zero_rollouts": 1200,
        "safe_zero_rollouts": 0,
    }


def test_td3_yaml_environment_count_can_be_overridden():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TD3_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "task.num_envs=256",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "td3_dagger_iterations=3000",
            ],
        )

    td3_entry.validate_td3_bc_dagger_config(cfg)
    assert cfg.task.num_envs == 256
    assert cfg.total_frames == 4000 * 256 * 32


def test_teacher_prefill_upper_bound_does_not_shorten_main_beta_schedule():
    cfg = _cfg(iterations=3000, total_frames=1, control_mode="beta")
    cfg.algo.teacher_prefill_max_rollouts = 10

    td3_entry.validate_td3_bc_dagger_config(cfg)
    schedule = td3_entry.td3_dagger_rollout_schedule(cfg)

    frames_per_rollout = 512 * 32
    assert cfg.total_frames == 3010 * frames_per_rollout
    assert cfg._bc_dagger_main_rollout_budget == 3000
    assert schedule["prefill_max_rollouts"] == 10
    assert schedule["prefill_target_rows"] == 131_072
    assert schedule["main_rollouts"] == 3000
    assert schedule["max_physical_rollouts"] == 3010
    assert schedule["start_rollout"] == 0
    assert schedule["end_rollout"] == 3000
    assert schedule["decay_rollouts"] == 1800
    assert schedule["beta_zero_rollouts"] == 1200


def test_teacher_prefill_requires_an_explicit_main_rollout_budget():
    cfg = _cfg(iterations=None, control_mode="beta")
    cfg.algo.teacher_prefill_max_rollouts = 10

    with pytest.raises(ValueError, match="explicit td3_dagger_iterations"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_pure_student_beta_uses_dynamic_teacher_prefill():
    cfg = _cfg(iterations=3000, control_mode="beta")
    cfg.algo.dagger_beta_start = 0.0
    cfg.algo.dagger_beta_end = 0.0

    td3_entry.validate_td3_bc_dagger_config(cfg)
    assert cfg.algo.teacher_prefill_max_rollouts == 10


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

    backend = td3_entry._expected_dagger_backend_config(cfg.algo)
    assert backend["load_pretrained_perception"] is False
    assert backend["perception_checkpoint_path"] is None
    assert backend["train_perception"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("load_pretrained_perception", 1),
        ("load_pretrained_perception", "true"),
        ("train_perception", 0),
        ("train_perception", "false"),
    ),
)
def test_perception_switches_must_be_boolean(field, value):
    cfg = _cfg()
    cfg.algo[field] = value

    with pytest.raises(ValueError, match=f"algo.{field} must be boolean"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_pretrained_perception_requires_an_explicit_existing_local_path(tmp_path):
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True

    with pytest.raises(ValueError, match="requires an explicit local"):
        td3_entry.validate_td3_bc_dagger_config(cfg)

    cfg.algo.perception_checkpoint_path = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_pretrained_perception_path_is_canonicalized_without_replacing_teacher_source(
    tmp_path,
):
    perception = _write_perception_checkpoint(tmp_path / "student_perception.pt")
    cfg = _cfg(checkpoint="/tmp/train_ppo_teacher.pt")
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)

    td3_entry.validate_td3_bc_dagger_config(cfg)

    assert cfg.algo.perception_checkpoint_path == str(perception.resolve())
    assert cfg.checkpoint_path == "/tmp/train_ppo_teacher.pt"


def test_disabled_pretrained_perception_rejects_path_and_freeze_mode(tmp_path):
    cfg = _cfg()
    cfg.algo.perception_checkpoint_path = str(tmp_path / "unused.pt")
    with pytest.raises(ValueError, match="must be null"):
        td3_entry.validate_td3_bc_dagger_config(cfg)

    cfg.algo.perception_checkpoint_path = None
    cfg.algo.train_perception = False
    with pytest.raises(ValueError, match="requires.*load_pretrained_perception=true"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


@pytest.mark.parametrize("train_perception", (True, False))
def test_pretrained_perception_supports_update_and_freeze_modes(
    tmp_path, train_perception
):
    perception = _write_perception_checkpoint(tmp_path / "student_perception.pt")
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)
    cfg.algo.train_perception = train_perception

    td3_entry.validate_td3_bc_dagger_config(cfg)


def test_pretrained_perception_requires_all_module_mappings(tmp_path):
    missing = td3_entry.REQUIRED_PRETRAINED_PERCEPTION_MODULES[0]
    perception = _write_perception_checkpoint(
        tmp_path / "incomplete_perception.pt", omit=missing
    )
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)

    with pytest.raises(ValueError, match=f"module mappings.*{missing}"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


@pytest.mark.parametrize(
    "field",
    (
        "teacher_actor_replay_fraction",
        "teacher_perception_replay_fraction",
        "failure_phase_teacher_fraction",
    ),
)
@pytest.mark.parametrize("value", (-0.1, 1.1, float("nan"), True))
def test_teacher_replay_fractions_must_be_finite_unit_interval(field, value):
    cfg = _cfg()
    cfg.algo[field] = value
    with pytest.raises(ValueError, match=r"non-negative|\[0, 1\]"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_dynamic_teacher_prefill_requires_positive_safety_ceiling():
    cfg = _cfg()
    cfg.algo.teacher_prefill_max_rollouts = 0
    cfg.algo.teacher_actor_replay_fraction = 0.25

    with pytest.raises(ValueError, match="teacher_prefill_max_rollouts.*positive"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_teacher_prefill_ceiling_must_theoretically_reach_ring_capacity():
    cfg = _cfg()
    cfg.algo.teacher_prefill_max_rollouts = 7

    with pytest.raises(ValueError, match=r"cannot possibly fill.*upper bound"):
        td3_entry.validate_td3_bc_dagger_config(cfg)

    # Equality is a valid static upper bound. Runtime still requires that the
    # rows belong to successful complete Teacher episodes.
    cfg.algo.teacher_prefill_max_rollouts = 8
    td3_entry.validate_td3_bc_dagger_config(cfg)


def test_failure_phase_focus_requires_an_enabled_teacher_source():
    cfg = _cfg(iterations=1)
    cfg.algo.teacher_prefill_max_rollouts = 8
    cfg.algo.failure_phase_teacher_fraction = 0.3
    cfg.algo.q_teacher_replay_ratio = 0.0

    with pytest.raises(ValueError, match="positive Teacher source fraction"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_failure_phase_samples_fit_in_inclusive_lookback_interval():
    cfg = _cfg()
    cfg.algo.failure_phase_lookback_steps = 8
    cfg.algo.failure_phase_samples_per_failure = 10

    with pytest.raises(ValueError, match=r"cannot exceed.*lookback_steps \+ 1"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_persistent_teacher_h5_export_is_rejected():
    cfg = _cfg()
    cfg.algo.save_teacher_buffer = True
    with pytest.raises(ValueError, match="teacher_replay_buffer.h5 export is disabled"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


def test_teacher_h5_input_path_is_rejected():
    cfg = _cfg()
    with open_dict(cfg.algo):
        cfg.algo.teacher_buffer_path = "/tmp/teacher_replay_buffer.h5"
    with pytest.raises(ValueError, match="does not accept a teacher replay H5 path"):
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
        ("perception_replay_burn_in", 7, "burn_in=8"),
        ("perception_encode_microbatch_size", 0, "positive"),
        ("teacher_perception_batch_size", 0, "positive"),
        ("perception_depth_codec", "float16", "uint8_div_100_v1"),
        ("teacher_prefill_max_rollouts", -1, "positive"),
        ("failure_phase_lookback_steps", 0, "positive"),
        ("failure_phase_samples_per_failure", 0, "positive"),
        ("failure_phase_num_bins", 0, "positive"),
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


def test_teacher_q_ring_must_cover_learning_start_threshold():
    cfg = _cfg()
    cfg.algo.q_teacher_buffer_capacity = cfg.algo.td3_learning_starts - 1

    with pytest.raises(ValueError, match="must cover"):
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

    assert cfg.total_frames == 1210 * 512 * 32
    assert cfg.algo.dagger_beta_decay_rollouts == 900
    assert schedule == {
        "frames_per_rollout": 16_384,
        "total_rollouts": 1200,
        "main_rollouts": 1200,
        "prefill_max_rollouts": 10,
        "prefill_target_rows": 131_072,
        "max_physical_rollouts": 1210,
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


def test_fresh_source_rejects_a_staged_checkpoint(tmp_path):
    source = _write_fresh_ppo_checkpoint(tmp_path / "wrong_source.pt")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["policy"]["training_algorithm"] = td3_entry.EXPECTED_TRAINING_ALGORITHM
    torch.save(checkpoint, source)
    cfg = _cfg(checkpoint=str(source))

    with pytest.raises(ValueError, match="fresh PPO teacher"):
        td3_entry.prepare_fresh_td3_bc_dagger_source(cfg)


@pytest.mark.parametrize("version", (1, td3_entry.EXPECTED_CHECKPOINT_VERSION))
def test_same_stage_td3_resume_is_rejected_for_legacy_and_current_versions(
    tmp_path, version
):
    checkpoint = tmp_path / f"td3_v{version}.pt"
    torch.save(
        {
            "policy": {
                "training_algorithm": td3_entry.EXPECTED_TRAINING_ALGORITHM,
                "checkpoint_version": version,
            }
        },
        checkpoint,
    )
    cfg = _cfg(checkpoint="/fresh/ppo.pt", resume=str(checkpoint))

    with pytest.raises(ValueError, match="fresh-only raw-perception replay v2"):
        td3_entry.prepare_td3_bc_dagger_checkpoint(cfg)
    with pytest.raises(ValueError, match="same-stage TD3 resume is unsupported"):
        td3_entry.validate_td3_bc_dagger_config(cfg)

    assert cfg.checkpoint_path == "/fresh/ppo.pt"
    assert td3_entry.td3_dagger_rollout_schedule(cfg)["start_rollout"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("td3_bc_dagger_copy_teacher_replay", True),
        ("td3_dagger_resume_rollout_count", 12),
        ("td3_dagger_resume_environment_steps", 384),
        ("_bc_dagger_teacher_replay_copy_source", "/tmp/teacher.h5"),
        ("_bc_dagger_teacher_replay_copy_path", "/tmp/teacher.h5"),
    ),
)
def test_removed_td3_and_h5_resume_controls_are_rejected(field, value):
    cfg = _cfg()
    with open_dict(cfg):
        cfg[field] = value

    with pytest.raises(ValueError, match="removed TD3/H5 resume controls"):
        td3_entry.validate_td3_bc_dagger_config(cfg)


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


def test_dynamic_prefill_checkpoint_index_uses_only_completed_main_rollouts():
    policy = type("Policy", (), {"dagger_rollout_count": 0})()

    # Any number of physical Teacher-prefill rollouts remains at main index 0.
    assert _bc_dagger_checkpoint_index(policy, 57, 3000, model_only_resume=False) == 0

    policy.dagger_rollout_count = 100
    assert (
        _bc_dagger_checkpoint_index(policy, 157, 3000, model_only_resume=False) == 100
    )


def test_checkpoint_index_preserves_legacy_and_model_only_resume_paths():
    plain = object()
    assert _bc_dagger_checkpoint_index(plain, 37, None, model_only_resume=False) == 37

    resumed = type("Policy", (), {"dagger_rollout_count": 101})()
    assert _bc_dagger_checkpoint_index(resumed, 37, None, model_only_resume=True) == 100


def test_dynamic_main_budget_stops_exactly_and_rejects_overshoot():
    policy = type("Policy", (), {"dagger_rollout_count": 2999})()
    assert _bc_dagger_main_budget_complete(policy, 3000) is False

    policy.dagger_rollout_count = 3000
    assert _bc_dagger_main_budget_complete(policy, 3000) is True

    policy.dagger_rollout_count = 3001
    with pytest.raises(RuntimeError, match=r"exceeded.*3001 > 3000"):
        _bc_dagger_main_budget_complete(policy, 3000)

    assert _bc_dagger_main_budget_complete(policy, None) is False
