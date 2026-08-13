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
                "teacher_prefill_max_rollouts": 10,
                "teacher_actor_replay_fraction": 0.5,
                "teacher_perception_replay_fraction": 0.5,
                "failure_phase_teacher_fraction": 0.3,
                "failure_phase_lookback_steps": 50,
                "failure_phase_samples_per_failure": 10,
                "failure_phase_num_bins": 1024,
                "dagger_replay_raw_observations": True,
                "replay_raw_observation_keys": list(
                    sac_entry.EXPECTED_REPLAY_RAW_OBSERVATION_KEYS
                ),
                "perception_replay_burn_in": 8,
                "perception_encode_microbatch_size": 128,
                "teacher_perception_batch_size": 128,
                "teacher_perception_warmup_steps": 128,
                "perception_depth_codec": "uint8_div_100_v1",
                "load_pretrained_perception": False,
                "perception_checkpoint_path": None,
                "train_perception": True,
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


def _write_perception_checkpoint(path: Path, *, omit: str | None = None) -> Path:
    policy = {name: {} for name in sac_entry.REQUIRED_PRETRAINED_PERCEPTION_MODULES}
    if omit is not None:
        policy.pop(omit)
    torch.save({"policy": policy}, path)
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
    assert cfg.task.num_envs == 256
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
    assert cfg.algo.teacher_perception_batch_size == 128
    assert cfg.algo.teacher_perception_warmup_steps == 128
    assert cfg.algo.load_pretrained_perception is False
    assert cfg.algo.perception_checkpoint_path is None
    assert cfg.algo.train_perception is True
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
    assert cfg.total_frames == 4000 * 256 * 32
    assert cfg._bc_dagger_main_rollout_budget == 3000
    assert sac_entry.fastsac_dagger_rollout_schedule(cfg) == {
        "frames_per_rollout": 8_192,
        "total_rollouts": 3000,
        "main_rollouts": 3000,
        "prefill_max_rollouts": 1000,
        "prefill_target_rows": 131_072,
        "max_physical_rollouts": 4000,
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
    assert cfg.total_frames == 4000 * 256 * 32


def test_explicit_main_rollout_budget_is_required():
    cfg = _cfg(iterations=None)
    with pytest.raises(ValueError, match="explicit positive fastsac_dagger_iterations"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_pure_student_main_rollout_uses_dynamic_teacher_prefill():
    cfg = _cfg()
    sac_entry.validate_fastsac_bc_dagger_config(cfg)
    assert cfg.algo.teacher_prefill_max_rollouts == 10


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
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_pretrained_perception_requires_an_explicit_existing_local_path(tmp_path):
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True

    with pytest.raises(ValueError, match="requires an explicit local"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)

    cfg.algo.perception_checkpoint_path = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_pretrained_perception_path_is_canonicalized_without_replacing_teacher_source(
    tmp_path,
):
    perception = _write_perception_checkpoint(tmp_path / "student_perception.pt")
    cfg = _cfg(checkpoint="/tmp/train_ppo_teacher.pt")
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)

    sac_entry.validate_fastsac_bc_dagger_config(cfg)

    assert cfg.algo.perception_checkpoint_path == str(perception.resolve())
    assert cfg.checkpoint_path == "/tmp/train_ppo_teacher.pt"


def test_ppo_vel_train_checkpoint_can_be_the_same_partial_perception_source(tmp_path):
    teacher = _write_fresh_ppo_checkpoint(tmp_path / "ppo_vel_train.pt")
    cfg = _cfg(checkpoint=str(teacher))
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(teacher)
    cfg.algo.train_perception = True

    sac_entry.validate_fastsac_bc_dagger_config(cfg)

    assert cfg.checkpoint_path == str(teacher)
    assert cfg.algo.perception_checkpoint_path == str(teacher.resolve())


def test_ppo_vel_train_partial_perception_source_cannot_freeze_fresh_depth(tmp_path):
    teacher = _write_fresh_ppo_checkpoint(tmp_path / "ppo_vel_train.pt")
    cfg = _cfg(checkpoint=str(teacher))
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(teacher)
    cfg.algo.train_perception = False

    with pytest.raises(ValueError, match="partial.*train_perception=true"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_ppo_vel_train_partial_perception_requires_all_four_adapt_mappings(tmp_path):
    teacher = _write_fresh_ppo_checkpoint(tmp_path / "ppo_vel_train.pt")
    checkpoint = torch.load(teacher, map_location="cpu", weights_only=False)
    checkpoint["policy"].pop("adapt_ema")
    torch.save(checkpoint, teacher)
    cfg = _cfg(checkpoint=str(teacher))
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(teacher)

    with pytest.raises(ValueError, match="adapt_ema"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_disabled_pretrained_perception_rejects_path_and_freeze_mode(tmp_path):
    cfg = _cfg()
    cfg.algo.perception_checkpoint_path = str(tmp_path / "unused.pt")
    with pytest.raises(ValueError, match="must be null"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)

    cfg.algo.perception_checkpoint_path = None
    cfg.algo.train_perception = False
    with pytest.raises(ValueError, match="requires.*load_pretrained_perception=true"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


@pytest.mark.parametrize("train_perception", (True, False))
def test_pretrained_perception_supports_update_and_freeze_modes(
    tmp_path, train_perception
):
    perception = _write_perception_checkpoint(tmp_path / "student_perception.pt")
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)
    cfg.algo.train_perception = train_perception

    sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_pretrained_perception_requires_all_module_mappings(tmp_path):
    missing = sac_entry.REQUIRED_PRETRAINED_PERCEPTION_MODULES[0]
    perception = _write_perception_checkpoint(
        tmp_path / "incomplete_perception.pt", omit=missing
    )
    cfg = _cfg()
    cfg.algo.load_pretrained_perception = True
    cfg.algo.perception_checkpoint_path = str(perception)

    with pytest.raises(ValueError, match=f"module mappings.*{missing}"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


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
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_dynamic_teacher_prefill_requires_positive_safety_ceiling():
    cfg = _cfg()
    cfg.algo.teacher_prefill_max_rollouts = 0
    cfg.algo.teacher_perception_replay_fraction = 0.5

    with pytest.raises(ValueError, match="teacher_prefill_max_rollouts.*positive"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_teacher_prefill_ceiling_must_theoretically_reach_ring_capacity():
    cfg = _cfg()
    cfg.algo.teacher_prefill_max_rollouts = 7

    with pytest.raises(ValueError, match=r"cannot possibly fill.*upper bound"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)

    # Equality is a valid static upper bound. Runtime still requires that the
    # rows belong to successful complete Teacher episodes.
    cfg.algo.teacher_prefill_max_rollouts = 8
    sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_failure_phase_focus_requires_an_enabled_teacher_source():
    cfg = _cfg()
    cfg.algo.teacher_actor_replay_fraction = 0.0
    cfg.algo.teacher_perception_replay_fraction = 0.0
    cfg.algo.q_teacher_replay_ratio = 0.0

    with pytest.raises(ValueError, match="positive Teacher source fraction"):
        sac_entry.validate_fastsac_bc_dagger_config(cfg)


def test_failure_phase_samples_fit_in_inclusive_lookback_interval():
    cfg = _cfg()
    cfg.algo.failure_phase_lookback_steps = 8
    cfg.algo.failure_phase_samples_per_failure = 10

    with pytest.raises(ValueError, match=r"cannot exceed.*lookback_steps \+ 1"):
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
        ("teacher_perception_warmup_steps", -1, "non-negative"),
        ("failure_phase_lookback_steps", 0, "positive"),
        ("failure_phase_samples_per_failure", 0, "positive"),
        ("failure_phase_num_bins", 0, "positive"),
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


def test_teacher_perception_warmup_can_be_disabled():
    cfg = _cfg()
    cfg.algo.teacher_perception_warmup_steps = 0

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


@pytest.mark.parametrize(
    ("distributed", "world_size"),
    ((True, 1), (False, 2)),
)
def test_fastsac_entrypoint_rejects_distributed_before_source_or_training(
    monkeypatch, distributed, world_size
):
    cfg = _cfg()
    source_calls = []
    training_calls = []
    monkeypatch.setattr(sac_entry.aa, "is_distributed", lambda: distributed)
    monkeypatch.setattr(sac_entry.aa, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        sac_entry,
        "prepare_fresh_fastsac_bc_dagger_source",
        lambda actual: source_calls.append(actual),
    )
    monkeypatch.setattr(
        sac_entry,
        "run_training",
        lambda actual: training_calls.append(actual),
    )

    with pytest.raises(RuntimeError, match="exactly one training process"):
        sac_entry.main.__wrapped__(cfg)

    assert source_calls == []
    assert training_calls == []
