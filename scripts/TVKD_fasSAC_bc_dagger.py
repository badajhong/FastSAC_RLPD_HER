"""CLI for frozen-Teacher-value TVKD FastSAC + bottleneck replay.

The algorithm lives in the installed ``active_adaptation.learning.ppo``
package so Hydra can import it both from this direct script and from tests.
This requested filename owns only source-checkpoint preparation, validation,
the Hydra surface, and delegation to the shared training runner.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import warnings
from collections.abc import Mapping

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    ACTOR_BACKEND,
    ACTOR_LEARNING_SEMANTICS,
    BOTTLENECK_LOCATION_SEMANTICS,
    CHECKPOINT_VERSION,
    CRITIC_LEARNING_SEMANTICS,
    EXPECTED_ACTOR_IN_KEYS,
    EXPECTED_ALGO_NAME,
    EXPECTED_ALGO_TARGET,
    LEGACY_ADAPTIVE_BC_CONFIG_FIELDS,
    LEGACY_CHECKPOINT_VERSION,
    LEGACY_TRAINING_ALGORITHM,
    PREVIOUS_CHECKPOINT_VERSION,
    PREVIOUS_TRAINING_ALGORITHM,
    FRESH_RING_RESUME_SEMANTICS,
    REPLAY_TEACHER_V_CURRENT_KEY,
    REPLAY_TEACHER_V_NEXT_KEY,
    TEACHER_VALUE_CACHE_SEMANTICS,
    V3_CHECKPOINT_VERSION,
    V3_TRAINING_ALGORITHM,
    SOURCE_FAILURE_TEACHER,
    SOURCE_STUDENT,
    SOURCE_UNIFORM_TEACHER,
    TRAINING_ALGORITHM,
    VERIFIED_HISTOGRAM_SEMANTICS,
    FrozenTeacherValueWrapper,
    TeacherValueBottleneckDetector,
    TVKDDistributionalFastSACTeacherBC,
    TVKDDistributionalFastSACTeacherBCConfig,
    _same_verified_histogram_state,
    _validate_tvkd_algorithm_config,
    compute_teacher_value_terms,
)
from active_adaptation.utils.wandb import parse_checkpoint_path

try:
    from .fastSAC_bc_dagger import (
        EXPECTED_ALGO_NAME as BASE_EXPECTED_ALGO_NAME,
        EXPECTED_ALGO_TARGET as BASE_EXPECTED_ALGO_TARGET,
        _require_single_process_execution,
        apply_fastsac_dagger_iteration_controls,
        fastsac_dagger_rollout_schedule,
        prepare_fresh_fastsac_bc_dagger_source,
        validate_fastsac_bc_dagger_config,
    )
    from .train import run_training
except ImportError:
    from fastSAC_bc_dagger import (  # type: ignore
        EXPECTED_ALGO_NAME as BASE_EXPECTED_ALGO_NAME,
        EXPECTED_ALGO_TARGET as BASE_EXPECTED_ALGO_TARGET,
        _require_single_process_execution,
        apply_fastsac_dagger_iteration_controls,
        fastsac_dagger_rollout_schedule,
        prepare_fresh_fastsac_bc_dagger_source,
        validate_fastsac_bc_dagger_config,
    )
    from train import run_training  # type: ignore


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def _resolved_fingerprint(value) -> str:
    """Return a stable SHA-256 fingerprint for a resolved Hydra contract."""
    node = value if OmegaConf.is_config(value) else OmegaConf.create(value)
    payload = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_task_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint the task/MDP contract while ignoring execution-only size/UI."""
    task = OmegaConf.to_container(cfg.task, resolve=True, enum_to_str=True)
    if not isinstance(task, dict):
        raise ValueError("TVKD runtime task must resolve to a mapping")
    for name in ("num_envs", "viewer"):
        task.pop(name, None)
    return _resolved_fingerprint(task)


def _install_teacher_contract_fingerprints(cfg: DictConfig) -> None:
    reward = cfg.task.get("reward")
    if reward is None:
        raise ValueError("TVKD runtime task lacks its reward contract")
    with open_dict(cfg.algo):
        cfg.algo.teacher_value_reward_group_fingerprint = _resolved_fingerprint(reward)
        cfg.algo.replay_task_fingerprint = _runtime_task_fingerprint(cfg)


def _validate_teacher_return_contract(source_cfg: Mapping, cfg: DictConfig) -> bool:
    """Validate that frozen PPO values and SAC raw rewards share one scale."""
    source_algo = source_cfg.get("algo") if isinstance(source_cfg, Mapping) else None
    if not isinstance(source_algo, Mapping):
        raise ValueError("PPO Teacher checkpoint lacks saved algo config")
    if "value_norm" not in source_algo:
        raise ValueError("PPO Teacher checkpoint lacks value_norm config")
    source_value_norm = source_algo.get("value_norm")
    if not isinstance(source_value_norm, bool):
        raise ValueError("PPO Teacher value_norm config must be boolean")
    if "clip_neg_reward" not in source_algo:
        raise ValueError("PPO Teacher checkpoint lacks clip_neg_reward config")
    source_clip_neg_reward = source_algo.get("clip_neg_reward")
    if not isinstance(source_clip_neg_reward, bool):
        raise ValueError("PPO Teacher clip_neg_reward config must be boolean")
    if source_clip_neg_reward:
        raise ValueError("TVKD requires a PPO Teacher trained on unclipped raw rewards")
    source_gamma = source_algo.get("gamma")
    runtime_gamma = cfg.algo.get("gamma")
    for label, value in (("source", source_gamma), ("runtime", runtime_gamma)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"TVKD {label} gamma must be a finite number")
    if not math.isclose(
        float(source_gamma),
        float(runtime_gamma),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "TVKD requires the PPO Teacher and FastSAC runtime to use the same gamma"
        )

    source_task = source_cfg.get("task")
    source_reward = (
        source_task.get("reward") if isinstance(source_task, Mapping) else None
    )
    runtime_reward = cfg.task.get("reward")
    if source_reward is None or runtime_reward is None:
        raise ValueError("TVKD requires reward config in source and runtime tasks")
    if tuple(source_reward.keys()) != tuple(runtime_reward.keys()):
        raise ValueError(
            "TVKD requires the PPO Teacher and FastSAC runtime to use the "
            "same ordered reward groups"
        )
    for group_name in source_reward.keys():
        source_group = source_reward[group_name]
        runtime_group = runtime_reward[group_name]
        if not isinstance(source_group, Mapping) or not isinstance(
            runtime_group, Mapping
        ):
            raise ValueError("TVKD reward groups must be mappings")
        if tuple(source_group.keys()) != tuple(runtime_group.keys()):
            raise ValueError(
                f"TVKD requires matching ordered reward terms in group {group_name!r}"
            )
    source_reward_contract = OmegaConf.to_container(
        source_reward
        if OmegaConf.is_config(source_reward)
        else OmegaConf.create(source_reward),
        resolve=True,
        enum_to_str=True,
    )
    runtime_reward_contract = OmegaConf.to_container(
        runtime_reward, resolve=True, enum_to_str=True
    )
    if source_reward_contract != runtime_reward_contract:
        raise ValueError(
            "TVKD requires the PPO Teacher and FastSAC runtime to use the "
            "same raw reward-group contract"
        )
    source_sim = source_task.get("sim")
    runtime_sim = cfg.task.get("sim")
    source_step_dt = (
        source_sim.get("step_dt") if isinstance(source_sim, Mapping) else None
    )
    runtime_step_dt = (
        runtime_sim.get("step_dt") if isinstance(runtime_sim, Mapping) else None
    )
    for label, value in (
        ("source", source_step_dt),
        ("runtime", runtime_step_dt),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"TVKD {label} reward step_dt must be positive")
    if not math.isclose(
        float(source_step_dt),
        float(runtime_step_dt),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "TVKD requires the PPO Teacher and FastSAC runtime to use the "
            "same reward integration step_dt"
        )
    return source_value_norm


def _validate_same_stage_task_contract(source_cfg: Mapping, cfg: DictConfig) -> None:
    """Reject runtime task drift that would change resumed state semantics."""
    source_task = source_cfg.get("task")
    if not isinstance(source_task, Mapping):
        raise ValueError("TVKD resume checkpoint lacks saved task config")
    # Only execution-scale/UI controls may differ. Everything that changes the
    # MDP, observation layout, termination, reward, or randomization is part of
    # the checkpoint contract.
    runtime_only = {"num_envs", "viewer"}

    def contract(task):
        node = task if OmegaConf.is_config(task) else OmegaConf.create(task)
        value = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
        if not isinstance(value, dict):
            raise ValueError("TVKD resume task config must resolve to a mapping")
        return {key: item for key, item in value.items() if key not in runtime_only}

    if contract(source_task) != contract(cfg.task):
        raise ValueError("TVKD resume task/MDP contract mismatch")


def _validate_tvkd_resume_policy_state(
    policy_state: Mapping,
    source_algo: Mapping,
    *,
    legacy: bool = False,
    require_student_focus_counters: bool = False,
) -> None:
    """Fail before W&B/Isaac startup when continuation state is incomplete."""
    current_v4 = (
        policy_state.get("training_algorithm") == TRAINING_ALGORITHM
        and policy_state.get("checkpoint_version") == CHECKPOINT_VERSION
    )

    required_modules = {
        "actor",
        "actor_adapt",
        "encoder_priv",
        "critic",
        "value_norm",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
        "adapt_module",
        "adapt_ema",
        "object_transform",
        "object_pred_transform",
    }
    if source_algo.get("use_object_adapt") is True:
        required_modules.update(("object_adapt", "object_adapt_ema"))
    if source_algo.get("use_depth") is True:
        required_modules.update(
            ("depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema")
        )
    if source_algo.get("train_dr_estimator") is True:
        required_modules.add("dr_estimator")
    missing_modules = sorted(
        name
        for name in required_modules
        if not isinstance(policy_state.get(name), Mapping)
    )
    if missing_modules:
        raise ValueError(
            f"TVKD resume checkpoint lacks policy module mappings: {missing_modules}"
        )

    frozen = policy_state.get("frozen_teacher_state")
    frozen_names = ["actor", "encoder_priv", "critic", "value_norm"]
    if isinstance(policy_state.get("height_encoder"), Mapping):
        frozen_names.append("height_encoder")
    missing_frozen = sorted(
        name for name in frozen_names if not isinstance(frozen.get(name), Mapping)
    )
    if missing_frozen:
        raise ValueError(
            f"TVKD resume checkpoint lacks frozen Teacher mappings: {missing_frozen}"
        )
    log_alpha = policy_state.get("log_alpha")
    if (
        not torch.is_tensor(log_alpha)
        or log_alpha.numel() != 1
        or not torch.isfinite(log_alpha).all()
    ):
        raise ValueError("TVKD resume checkpoint lacks finite scalar log_alpha")

    optimizers = policy_state.get("optimizer_resume_state")
    for name in ("actor_optimizer", "critic_optimizer"):
        if not isinstance(optimizers.get(name), Mapping):
            raise ValueError(f"TVKD resume checkpoint lacks {name}")
    conditional_optimizers = (
        ("alpha_optimizer", bool(source_algo.get("sac_use_autotune"))),
        ("adapt_optimizer", bool(source_algo.get("train_perception"))),
        ("dr_estimator_optimizer", bool(source_algo.get("train_dr_estimator"))),
    )
    for name, enabled in conditional_optimizers:
        value = optimizers.get(name)
        if enabled and not isinstance(value, Mapping):
            raise ValueError(f"TVKD resume checkpoint lacks active {name}")
        if not enabled and value is not None:
            raise ValueError(f"TVKD resume checkpoint has unexpected {name}")

    counters = (
        "actor_update_count",
        "critic_update_count",
        "alpha_update_count",
        "dagger_rollout_count",
        "dagger_environment_steps",
        "teacher_prefill_rollout_count",
        "teacher_prefill_environment_steps",
        "num_updates",
        "sac_actor_update_count",
        "sac_alpha_update_count",
    )
    for name in counters:
        value = policy_state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"TVKD resume checkpoint has invalid {name}")
    row_credit = policy_state.get("q_update_row_credit")
    q_batch_size = int(source_algo.get("q_batch_size"))
    if (
        isinstance(row_credit, bool)
        or not isinstance(row_credit, (int, float))
        or not math.isfinite(float(row_credit))
        or not 0.0 <= float(row_credit) < q_batch_size
    ):
        raise ValueError("TVKD resume checkpoint has invalid Q row credit")
    train_every = int(source_algo.get("train_every"))
    if policy_state["dagger_environment_steps"] != (
        policy_state["dagger_rollout_count"] * train_every
    ):
        raise ValueError("TVKD resume DAgger rollout/environment counters disagree")
    if policy_state["teacher_prefill_environment_steps"] != (
        policy_state["teacher_prefill_rollout_count"] * train_every
    ):
        raise ValueError("TVKD resume Teacher-prefill counters disagree")

    for name in (
        "dagger_rng_state",
        "q_rng_state",
        "sac_action_rng_state",
        "sac_rollout_rng_state",
        "teacher_perception_rng_state",
    ):
        value = policy_state.get(name)
        if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"TVKD resume checkpoint has invalid {name}")
    last_iter = policy_state.get("last_iter")
    next_iter = policy_state.get("next_iter")
    if (
        isinstance(last_iter, bool)
        or not isinstance(last_iter, int)
        or last_iter < 0
        or isinstance(next_iter, bool)
        or not isinstance(next_iter, int)
        or next_iter != last_iter + 1
    ):
        raise ValueError("TVKD resume checkpoint has invalid iteration progress")
    if policy_state.get("last_phase") != "finetune":
        raise ValueError("TVKD resume checkpoint is not a finetune-stage policy")

    if not legacy:
        bottleneck = policy_state.get("teacher_value_bottleneck_replay_state")
        detector = (
            bottleneck.get("detector") if isinstance(bottleneck, Mapping) else None
        )
        if not isinstance(detector, Mapping):
            raise ValueError("TVKD resume checkpoint lacks bottleneck detector state")
        if current_v4 and bottleneck.get("location_semantics") != (
            BOTTLENECK_LOCATION_SEMANTICS
        ):
            raise ValueError("TVKD v4 bottleneck location semantics mismatch")
        scale = detector.get("bottleneck_residual_scale_ema")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise ValueError("TVKD resume bottleneck residual scale is invalid")
        updates = detector.get("num_scale_updates", 0)
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("TVKD resume bottleneck scale update count is invalid")
        bottleneck_counter_names = [
            "failed_student_episode_count",
            "student_candidate_count",
            "detected_count",
            "fallback_count",
            "no_candidate_count",
            "selected_count",
            "teacher_sequences_inserted",
            "teacher_transitions_inserted",
            "phase_match_distance_count",
            "next_student_episode_id",
        ]
        if require_student_focus_counters:
            bottleneck_counter_names.extend(
                (
                    "student_focus_rows_marked",
                    "student_focus_rows_missing",
                    "student_focus_sampled_rows",
                    "student_focus_uniform_fallback_rows",
                )
            )
        if current_v4:
            bottleneck_counter_names.extend(
                (
                    "unsuccessful_episode_count",
                    "episodes_with_student_candidates",
                    "no_value_bottleneck_count",
                    "value_argmin_ablation_count",
                )
            )
        for name in bottleneck_counter_names:
            value = bottleneck.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD resume bottleneck state {name} is invalid")
        nonnegative_float_fields = {
            "selected_step_sum",
            "selected_phase_sum",
            "score_sum",
            "score_max",
            "phase_match_distance_sum",
        }
        for name in (
            *nonnegative_float_fields,
            "raw_td_residual_sum",
            "normalized_td_residual_sum",
        ):
            value = bottleneck.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (name in nonnegative_float_fields and float(value) < 0.0)
            ):
                raise ValueError(f"TVKD resume bottleneck state {name} is invalid")
        if not isinstance(bottleneck.get("last_metadata", {}), Mapping):
            raise ValueError("TVKD resume bottleneck metadata is invalid")
        if current_v4 and not isinstance(
            bottleneck.get("last_value_argmin_metadata", {}), Mapping
        ):
            raise ValueError("TVKD resume value-argmin metadata is invalid")

    failure = policy_state.get(
        "verified_teacher_value_histogram_state"
        if current_v4
        else "failure_phase_curriculum_state"
    )
    if not isinstance(failure, Mapping):
        raise ValueError("TVKD resume failure histogram state is missing")
    if current_v4 and failure.get("semantics") != VERIFIED_HISTOGRAM_SEMANTICS:
        raise ValueError("TVKD v4 verified histogram semantics mismatch")
    histogram = failure.get("histogram")
    if (
        not torch.is_tensor(histogram)
        or histogram.dtype != torch.float64
        or histogram.ndim != 1
        or histogram.numel() != int(source_algo.get("failure_phase_num_bins"))
        or not torch.isfinite(histogram).all()
        or (histogram < 0.0).any()
        or not torch.equal(histogram, histogram.round())
    ):
        raise ValueError("TVKD resume failure histogram is invalid")
    failure_counters = {}
    for name in (
        "episode_count",
        "anchor_count",
        "uniform_fallback_rows",
        "focused_rows",
    ):
        value = failure.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"TVKD resume failure state {name} is invalid")
        failure_counters[name] = value
    if int(histogram.sum().item()) != failure_counters["anchor_count"]:
        raise ValueError("TVKD resume failure histogram/anchor count mismatch")
    if current_v4:
        motion_histograms = failure.get("motion_histograms")
        if not isinstance(motion_histograms, Mapping):
            raise ValueError("TVKD v4 histogram lacks motion partitions")
        motion_total = torch.zeros_like(histogram)
        for motion_id, motion_histogram in motion_histograms.items():
            if (
                isinstance(motion_id, bool)
                or not isinstance(motion_id, int)
                or motion_id < 0
                or not torch.is_tensor(motion_histogram)
                or motion_histogram.dtype != torch.float64
                or motion_histogram.shape != histogram.shape
                or not torch.isfinite(motion_histogram).all()
                or bool((motion_histogram < 0.0).any())
                or not torch.equal(motion_histogram, motion_histogram.round())
            ):
                raise ValueError("TVKD v4 motion histogram is invalid")
            motion_total.add_(motion_histogram)
        if not torch.equal(motion_total, histogram):
            raise ValueError("TVKD v4 motion/global histogram counts disagree")

    action_contract = policy_state.get("action_contract")
    if not isinstance(
        action_contract.get("joint_names"), (list, tuple)
    ) or not isinstance(action_contract.get("fingerprint"), str):
        raise ValueError("TVKD resume action contract is incomplete")
    if not isinstance(policy_state.get("q_backend_config"), Mapping):
        raise ValueError("TVKD resume checkpoint lacks Q backend config")


def _prepare_tvkd_fresh_source(cfg: DictConfig) -> dict | None:
    """Load the baseline PPO source and mirror its ValueNorm construction."""
    _install_teacher_contract_fingerprints(cfg)
    prepared = prepare_fresh_fastsac_bc_dagger_source(cfg)
    source_path = cfg.get("checkpoint_path", None)
    if source_path is None:
        return prepared
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_cfg = checkpoint.get("cfg")
    if not isinstance(source_cfg, Mapping):
        raise ValueError("PPO Teacher checkpoint lacks saved config")
    source_value_norm = _validate_teacher_return_contract(source_cfg, cfg)
    with open_dict(cfg.algo):
        cfg.algo.value_norm = source_value_norm
    return prepared


def _resolve_tvkd_checkpoint(path) -> str:
    """Resolve a local or W&B TVKD checkpoint from Hydra's launch cwd."""
    value = os.path.expanduser(os.fspath(path))
    if value.startswith("run:"):
        resolved = parse_checkpoint_path(value, download_replay=False)
    else:
        resolved = hydra.utils.to_absolute_path(value)
    if resolved is None:
        raise FileNotFoundError("Unable to resolve TVKD resume checkpoint")
    resolved = os.path.realpath(os.path.expanduser(os.fspath(resolved)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"TVKD resume checkpoint does not exist: {resolved}")
    return resolved


_FOUR_WAY_SUFFIXES = (
    "uniform_student",
    "failure_student",
    "uniform_teacher",
    "failure_teacher",
)


def _validate_v4_policy_contract(policy_state: Mapping, cfg: DictConfig) -> None:
    """Validate the scientific v4 metadata before simulator construction."""
    replay_mix = policy_state.get("replay_mix_state")
    if not isinstance(replay_mix, Mapping):
        raise ValueError("TVKD v4 checkpoint lacks replay mix state")
    for purpose in ("q", "actor", "perception"):
        saved = replay_mix.get(purpose)
        if not isinstance(saved, Mapping):
            raise ValueError(f"TVKD v4 checkpoint lacks {purpose!r} replay mix")
        total = 0.0
        for source in _FOUR_WAY_SUFFIXES:
            value = saved.get(source)
            expected = cfg.algo.get(f"{purpose}_{source}_fraction")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), float(expected), rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError(f"TVKD v4 replay mix mismatch at {purpose}.{source}")
            total += float(value)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"TVKD v4 {purpose} replay mix does not sum to one")

    vecnorm_fingerprint = policy_state.get("vecnorm_fingerprint")
    exact = {
        "critic_learning_semantics": CRITIC_LEARNING_SEMANTICS,
        "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
        "perception_replay_mode": str(cfg.algo.perception_replay_mode),
        "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
        "bottleneck_fallback_mode": str(cfg.algo.bottleneck_fallback_mode),
        "teacher_value_return_semantics": str(cfg.algo.teacher_value_return_semantics),
        "teacher_value_boundary_semantics": str(
            cfg.algo.teacher_value_boundary_semantics
        ),
        "teacher_value_reward_group_fingerprint": str(
            cfg.algo.teacher_value_reward_group_fingerprint
        ),
        "teacher_value_vecnorm_fingerprint": vecnorm_fingerprint,
        "replay_task_fingerprint": str(cfg.algo.replay_task_fingerprint),
        "fresh_ring_resume_semantics": FRESH_RING_RESUME_SEMANTICS,
    }
    for name, expected in exact.items():
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"TVKD runtime lacks required metadata {name!r}")
        if policy_state.get(name) != expected:
            raise ValueError(f"TVKD v4 metadata mismatch at {name!r}")
    q_backend = policy_state.get("q_backend_config")
    if not isinstance(q_backend, Mapping):
        raise ValueError("TVKD v4 checkpoint lacks Q backend metadata")
    expected_q_metadata = {
        "target_semantics": CRITIC_LEARNING_SEMANTICS,
        "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
        "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
        "bottleneck_fallback_mode": str(cfg.algo.bottleneck_fallback_mode),
    }
    for name, expected in expected_q_metadata.items():
        if q_backend.get(name) != expected:
            raise ValueError(f"TVKD v4 Q backend metadata mismatch at {name!r}")
    gamma = policy_state.get("teacher_value_gamma")
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(float(gamma))
        or not math.isclose(
            float(gamma), float(cfg.algo.gamma), rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("TVKD v4 Teacher value gamma mismatch")
    verified = policy_state.get("verified_teacher_value_histogram_state")
    if (
        not isinstance(verified, Mapping)
        or verified.get("semantics") != VERIFIED_HISTOGRAM_SEMANTICS
    ):
        raise ValueError("TVKD v4 checkpoint lacks verified histogram semantics")
    compatibility = policy_state.get("failure_phase_curriculum_state")
    if not isinstance(compatibility, Mapping) or not _same_verified_histogram_state(
        verified, compatibility
    ):
        raise ValueError("TVKD v4 verified histogram aliases are inconsistent")


def _legacy_fraction(name: str, value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"TVKD legacy checkpoint has invalid {name}")
    return float(value)


def _legacy_four_way_mix(
    *, teacher_fraction: float, teacher_focus: float, student_focus: float
) -> dict[str, float]:
    teacher_fraction = _legacy_fraction("Teacher fraction", teacher_fraction)
    teacher_focus = _legacy_fraction("Teacher focus fraction", teacher_focus)
    student_focus = _legacy_fraction("Student focus fraction", student_focus)
    student_fraction = 1.0 - teacher_fraction
    return {
        "uniform_student": student_fraction * (1.0 - student_focus),
        "failure_student": student_fraction * student_focus,
        "uniform_teacher": teacher_fraction * (1.0 - teacher_focus),
        "failure_teacher": teacher_fraction * teacher_focus,
    }


def _install_legacy_v4_replay_contract(
    cfg: DictConfig, backend: Mapping, *, student_focus_default: float
) -> dict[str, object]:
    """Derive an explicit v4 contract from a v1-v3 nested replay mixture."""
    teacher_focus = _legacy_fraction(
        "failure_phase_teacher_fraction",
        backend.get("failure_phase_teacher_fraction"),
    )
    student_focus = _legacy_fraction(
        "failure_phase_student_fraction",
        backend.get("failure_phase_student_fraction", student_focus_default),
    )
    q_mix = _legacy_four_way_mix(
        teacher_fraction=backend.get("q_teacher_replay_ratio"),
        teacher_focus=teacher_focus,
        student_focus=student_focus,
    )
    actor_mix = _legacy_four_way_mix(
        teacher_fraction=backend.get("teacher_actor_replay_fraction"),
        teacher_focus=teacher_focus,
        student_focus=student_focus,
    )
    perception_teacher = _legacy_fraction(
        "teacher_perception_replay_fraction",
        backend.get("teacher_perception_replay_fraction"),
    )
    perception_mix = {
        "uniform_student": 1.0 - perception_teacher,
        "failure_student": 0.0,
        "uniform_teacher": perception_teacher * (1.0 - teacher_focus),
        "failure_teacher": perception_teacher * teacher_focus,
    }
    cadence = backend.get("sac_alpha_update_cadence", "critic")
    if cadence not in {"actor", "critic"}:
        raise ValueError("TVKD legacy checkpoint has invalid alpha cadence")
    with open_dict(cfg.algo):
        cfg.algo.q_teacher_replay_ratio = _legacy_fraction(
            "q_teacher_replay_ratio", backend.get("q_teacher_replay_ratio")
        )
        cfg.algo.teacher_actor_replay_fraction = _legacy_fraction(
            "teacher_actor_replay_fraction",
            backend.get("teacher_actor_replay_fraction"),
        )
        cfg.algo.teacher_perception_replay_fraction = perception_teacher
        cfg.algo.failure_phase_teacher_fraction = teacher_focus
        for purpose, mix in (
            ("q", q_mix),
            ("actor", actor_mix),
            ("perception", perception_mix),
        ):
            for source, fraction in mix.items():
                cfg.algo[f"{purpose}_{source}_fraction"] = fraction
        cfg.algo.perception_replay_mode = "legacy_online_student"
        cfg.algo.bottleneck_fallback_mode = "none"
        cfg.algo.bottleneck_include_unsuccessful_timeouts = True
        cfg.algo.max_teacher_phase_match_distance = None
        cfg.algo.sac_alpha_update_cadence = cadence
        cfg.algo.failure_phase_student_fraction = student_focus
    return {
        **{
            f"{purpose}_{source}_fraction": fraction
            for purpose, mix in (
                ("q", q_mix),
                ("actor", actor_mix),
                ("perception", perception_mix),
            )
            for source, fraction in mix.items()
        },
        "perception_replay_mode": "legacy_online_student",
        "bottleneck_fallback_mode": "none",
        "bottleneck_include_unsuccessful_timeouts": True,
        "max_teacher_phase_match_distance": None,
        "sac_alpha_update_cadence": cadence,
    }


def _prepare_tvkd_checkpoint(cfg: DictConfig) -> dict | None:
    """Validate a model-only TVKD continuation and bottleneck state."""
    requested = cfg.get("fastsac_bc_dagger_checkpoint", None)
    if requested is None:
        return None
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "TVKD model-state continuation rebuilds its online rings; remove "
            "explicit teacher replay paths"
        )
    resolved = _resolve_tvkd_checkpoint(requested)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy")
    source_cfg = checkpoint.get("cfg")
    if not isinstance(policy_state, Mapping):
        raise ValueError("TVKD resume checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), Mapping):
        raise ValueError("TVKD resume checkpoint has no VecNorm state")
    if not isinstance(source_cfg, Mapping):
        raise ValueError("TVKD resume checkpoint has no saved config")
    _install_teacher_contract_fingerprints(cfg)
    algorithm = policy_state.get("training_algorithm")
    version = policy_state.get("checkpoint_version", -1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("TVKD resume checkpoint version is invalid")
    legacy = (
        algorithm == LEGACY_TRAINING_ALGORITHM and version == LEGACY_CHECKPOINT_VERSION
    )
    previous = (
        algorithm == PREVIOUS_TRAINING_ALGORITHM
        and version == PREVIOUS_CHECKPOINT_VERSION
    )
    v3 = algorithm == V3_TRAINING_ALGORITHM and version == V3_CHECKPOINT_VERSION
    current = algorithm == TRAINING_ALGORITHM and version == CHECKPOINT_VERSION
    if not (legacy or previous or v3 or current):
        raise ValueError("fastsac_bc_dagger_checkpoint is not a TVKD checkpoint")
    if legacy:
        warnings.warn(
            "TVKD v1 adaptive BC state will be ignored; resume uses the fixed "
            "baseline BC coefficient and a fresh bottleneck residual scale.",
            UserWarning,
            stacklevel=2,
        )
    elif previous:
        warnings.warn(
            "TVKD v2 checkpoints predate focused Student replay; continuation "
            "preserves their replay behavior with "
            "failure_phase_student_fraction=0.0.",
            UserWarning,
            stacklevel=2,
        )
    elif v3:
        warnings.warn(
            "Migrating a TVKD v3 checkpoint to the v4 replay/value contract: "
            "model and optimizer state are retained, perception uses its explicit "
            "legacy-online mode, and detector/verified-histogram state is reset.",
            UserWarning,
            stacklevel=2,
        )
    if policy_state.get("actor_backend") != ACTOR_BACKEND:
        raise ValueError("TVKD resume checkpoint actor backend mismatch")
    required_mappings = [
        "frozen_teacher_state",
        "failure_phase_curriculum_state",
        "optimizer_resume_state",
        "action_contract",
        "perception_initialization",
    ]
    if previous or v3 or current:
        required_mappings.append("teacher_value_bottleneck_replay_state")
    for name in required_mappings:
        if not isinstance(policy_state.get(name), Mapping):
            raise ValueError(f"TVKD resume checkpoint lacks {name!r}")
    vecnorm_fingerprint = policy_state.get("vecnorm_fingerprint")
    if not isinstance(vecnorm_fingerprint, str) or not vecnorm_fingerprint:
        raise ValueError("TVKD resume checkpoint lacks VecNorm fingerprint")
    backend = policy_state.get("dagger_backend_config")
    if not isinstance(backend, Mapping):
        raise ValueError("TVKD resume checkpoint lacks backend config")
    saved_lambda_bc = backend.get("lambda_bc")
    if (
        isinstance(saved_lambda_bc, bool)
        or not isinstance(saved_lambda_bc, (int, float))
        or not math.isfinite(float(saved_lambda_bc))
        or float(saved_lambda_bc) < 0.0
        or not math.isclose(
            float(saved_lambda_bc),
            float(cfg.algo.lambda_bc),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("TVKD resume fixed BC coefficient mismatch")
    migration_fields: dict[str, object] = {}
    if legacy or previous or v3:
        migration_fields = _install_legacy_v4_replay_contract(
            cfg,
            backend,
            student_focus_default=(
                0.0
                if legacy or previous
                else backend.get("failure_phase_student_fraction")
            ),
        )
    source_value_norm = _validate_teacher_return_contract(source_cfg, cfg)
    _validate_same_stage_task_contract(source_cfg, cfg)
    # ValueNorm changes the module type, so mirror the saved construction
    # choice before comparing the complete same-stage algorithm contract.
    with open_dict(cfg.algo):
        cfg.algo.value_norm = source_value_norm
        if previous:
            # v2 had no focused Student partition. Same-stage continuation is
            # intentionally behavior-preserving even though fresh v3 runs use
            # the new 0.3 default.
            cfg.algo.failure_phase_student_fraction = 0.0
    source_algo = source_cfg.get("algo")
    if not isinstance(source_algo, Mapping):
        raise ValueError("TVKD resume checkpoint lacks saved algo config")
    if source_algo.get("load_pretrained_perception") is True:
        saved_perception_path = source_algo.get("perception_checkpoint_path")
        if not isinstance(saved_perception_path, str) or not saved_perception_path:
            raise ValueError(
                "TVKD resume checkpoint lacks saved perception provenance path"
            )
        # Resume restores the embedded online/EMA perception children and
        # optimizer; the historical warm-start file is provenance only and is
        # deliberately not reopened. Mirror its saved string so the full algo
        # contract stays exact without requiring the old file to still exist.
        with open_dict(cfg.algo):
            cfg.algo.perception_checkpoint_path = saved_perception_path
    # Full same-stage semantics are fixed by the saved algo config.  The new
    # run may change only outer execution controls (additional rollout count,
    # logging, W&B, and checkpoint path), none of which live in cfg.algo.
    source_algo_contract = OmegaConf.to_container(
        source_algo
        if OmegaConf.is_config(source_algo)
        else OmegaConf.create(source_algo),
        resolve=True,
        enum_to_str=True,
    )
    # Legacy v1 checkpoints predate this explicit provenance field but already
    # used one alpha update per Critic. v2+ checkpoints must contain it.
    if legacy:
        source_algo_contract.setdefault("sac_alpha_update_cadence", "critic")
    runtime_algo_contract = OmegaConf.to_container(
        cfg.algo, resolve=True, enum_to_str=True
    )
    if legacy:
        for name in LEGACY_ADAPTIVE_BC_CONFIG_FIELDS:
            source_algo_contract.pop(name, None)
        for name in (
            "use_teacher_value_bottleneck_replay",
            "bottleneck_threshold",
            "bottleneck_smoothing_window",
            "bottleneck_min_consecutive",
            "bottleneck_terminal_exclusion_steps",
            "bottleneck_residual_scale_ema_decay",
            "bottleneck_eps",
            "failure_phase_student_fraction",
        ):
            source_algo_contract[name] = runtime_algo_contract[name]
    elif previous:
        # Structured v2 configs did not contain this v3 replay control. Its
        # behavior was equivalent to a zero focused share.
        source_algo_contract.setdefault("failure_phase_student_fraction", 0.0)
    if legacy or previous or v3:
        # These controls did not exist in the saved schema.  They are explicit
        # migration semantics, not claims that the old run used v4 behavior.
        source_algo_contract.update(migration_fields)
        for name in (
            "perception_replay_batch_size",
            "teacher_value_return_semantics",
            "teacher_value_boundary_semantics",
            "teacher_value_reward_group_fingerprint",
            "replay_task_fingerprint",
        ):
            source_algo_contract[name] = runtime_algo_contract[name]
    if source_algo_contract != runtime_algo_contract:
        raise ValueError("TVKD resume algorithm config does not match checkpoint")
    if current:
        _validate_v4_policy_contract(policy_state, cfg)
    _validate_tvkd_resume_policy_state(
        policy_state,
        source_algo,
        legacy=legacy,
        require_student_focus_counters=current or v3,
    )
    if backend.get("value_norm") is not source_value_norm:
        raise ValueError("TVKD resume checkpoint ValueNorm metadata is inconsistent")
    metadata_only = {
        "method",
        "actor_output",
        "bc_loss",
        "teacher_value_semantics",
    }
    for name, saved_value in backend.items():
        if name in metadata_only or name not in cfg.algo:
            continue
        current_value = cfg.algo.get(name)

        def plain(value):
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True, enum_to_str=True)
            if isinstance(value, (Mapping, list, tuple)):
                return OmegaConf.to_container(
                    OmegaConf.create(value), resolve=True, enum_to_str=True
                )
            return value

        if plain(saved_value) != plain(current_value):
            raise ValueError(
                f"TVKD resume config mismatch at algo.{name}: "
                f"checkpoint={saved_value!r}, runtime={current_value!r}"
            )
    rollout_count = policy_state.get("dagger_rollout_count")
    if isinstance(rollout_count, bool) or not isinstance(rollout_count, int):
        raise ValueError("TVKD resume checkpoint lacks a valid rollout count")
    if rollout_count < 0:
        raise ValueError("TVKD resume rollout count must be non-negative")

    with open_dict(cfg):
        cfg.fastsac_bc_dagger_checkpoint = resolved
        cfg.checkpoint_path = resolved
        cfg._tvkd_model_only_resume = True
        # Suppress generic H5 discovery: this continuation deliberately
        # rebuilds the two raw online rings with a fresh Teacher prefill.
        cfg._bc_dagger_fresh_source = True
        cfg.tvkd_resume_rollout_count = int(rollout_count)
    print(
        "TVKD model-state continuation: restored checkpoint at main rollout "
        f"{rollout_count}, last_iter={policy_state['last_iter']}, "
        f"next_iter={policy_state['next_iter']}, "
        f"phase={policy_state['last_phase']!r}; the raw Teacher/Student replay "
        "rings will be rebuilt and saved Q row credit "
        f"{float(policy_state['q_update_row_credit']):g} will reset to 0."
    )
    return {
        "path": resolved,
        "rollout_count": int(rollout_count),
    }


def validate_tvkd_fastsac_bc_dagger_config(cfg: DictConfig) -> None:
    """Reuse every baseline lock, then validate only the added controls."""
    apply_fastsac_dagger_iteration_controls(cfg)
    if cfg.algo.get("name") != EXPECTED_ALGO_NAME:
        raise ValueError(f"TVKD entrypoint requires algo.name={EXPECTED_ALGO_NAME!r}")
    if cfg.algo.get("_target_") != EXPECTED_ALGO_TARGET:
        raise ValueError(
            f"TVKD entrypoint requires algo._target_={EXPECTED_ALGO_TARGET!r}"
        )
    _validate_tvkd_algorithm_config(cfg.algo)
    # Keep one source of truth for topology, replay ratios, UTD, target cadence,
    # perception/randomization controls, and fresh PPO checkpoint rules.
    baseline_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
    )
    with open_dict(baseline_cfg.algo):
        baseline_cfg.algo.name = BASE_EXPECTED_ALGO_NAME
        baseline_cfg.algo._target_ = BASE_EXPECTED_ALGO_TARGET
        # Baseline validation owns the fresh production Actor cadence. A
        # migrated v3 policy may retain its explicit saved Critic cadence, so
        # translate only the validation clone here.
        baseline_cfg.algo.sac_alpha_update_cadence = "actor"
    # The baseline intentionally rejects continuation because it has no model
    # for TVKD's bottleneck-aware, fresh-ring resume contract. Every topology
    # and source-ratio lock still applies to the translated validation config.
    with open_dict(baseline_cfg):
        baseline_cfg.fastsac_bc_dagger_checkpoint = None
        resume_rollouts = int(cfg.get("tvkd_resume_rollout_count", 0))
        if resume_rollouts:
            # Baseline cutoff checks use absolute DAgger rollout coordinates.
            # Keep the real frame budget additional-only, but validate the
            # resumed schedule against its cumulative end coordinate.
            baseline_cfg.fastsac_dagger_iterations = resume_rollouts + int(
                cfg.fastsac_dagger_iterations
            )
    model_only_resume = bool(cfg.get("_tvkd_model_only_resume", False))
    if model_only_resume and bool(baseline_cfg.algo.load_pretrained_perception):
        # A TVKD checkpoint already owns every online/EMA perception child.
        # Do not require the historical warm-start file to remain available;
        # resume restores the saved children and optimizer state directly.
        with open_dict(baseline_cfg.algo):
            baseline_cfg.algo.load_pretrained_perception = False
            baseline_cfg.algo.perception_checkpoint_path = None
            baseline_cfg.algo.train_perception = True
    validate_fastsac_bc_dagger_config(baseline_cfg)
    if not model_only_resume and bool(baseline_cfg.algo.load_pretrained_perception):
        # Baseline validation resolves local paths against Hydra's original
        # launch cwd.  Propagate that canonical path to the real TVKD config;
        # otherwise policy construction would resolve the untouched relative
        # string against Hydra's per-run output directory.
        with open_dict(cfg.algo):
            cfg.algo.perception_checkpoint_path = (
                baseline_cfg.algo.perception_checkpoint_path
            )


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="TVKD_fasSAC_bc_dagger",
    version_base=None,
)
def main(cfg: DictConfig):
    _require_single_process_execution()
    apply_fastsac_dagger_iteration_controls(cfg)
    resume = _prepare_tvkd_checkpoint(cfg)
    if resume is None:
        _prepare_tvkd_fresh_source(cfg)
    validate_tvkd_fastsac_bc_dagger_config(cfg)
    schedule = fastsac_dagger_rollout_schedule(cfg)
    start_rollout = 0 if resume is None else int(resume["rollout_count"])
    if resume is not None:
        with open_dict(cfg):
            cfg._bc_dagger_main_rollout_budget = start_rollout + int(
                cfg.fastsac_dagger_iterations
            )

    def replay_mix_summary(purpose: str) -> str:
        fractions = {
            source: float(cfg.algo[f"{purpose}_{source}_fraction"])
            for source in _FOUR_WAY_SUFFIXES
        }
        student_fraction = fractions["uniform_student"] + fractions["failure_student"]
        teacher_fraction = fractions["uniform_teacher"] + fractions["failure_teacher"]
        return (
            f"{purpose}=US {100.0 * fractions['uniform_student']:g}% / "
            f"FS {100.0 * fractions['failure_student']:g}% / "
            f"UT {100.0 * fractions['uniform_teacher']:g}% / "
            f"FT {100.0 * fractions['failure_teacher']:g}% "
            f"(Student {100.0 * student_fraction:g}%, "
            f"Teacher {100.0 * teacher_fraction:g}%)"
        )

    print(
        "TVKD Distributional FastSAC + fixed BC + value-bottleneck replay: "
        f"prefill=until {schedule['prefill_target_rows']} Teacher rows, "
        f"main_additional={schedule['main_rollouts']}, "
        f"main_range=[{start_rollout}, "
        f"{start_rollout + schedule['main_rollouts']}), "
        f"frames/rollout={schedule['frames_per_rollout']}; "
        f"tvkd_lambda={float(cfg.algo.tvkd_lambda):g}, "
        f"bottleneck={bool(cfg.algo.use_teacher_value_bottleneck_replay)}, "
        f"alpha_update_cadence={cfg.algo.sac_alpha_update_cadence}, "
        f"{replay_mix_summary('q')}; "
        f"{replay_mix_summary('actor')}; "
        f"{replay_mix_summary('perception')}; "
        f"perception_mode={cfg.algo.perception_replay_mode}"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_ACTOR_IN_KEYS",
    "EXPECTED_ALGO_NAME",
    "EXPECTED_ALGO_TARGET",
    "SOURCE_FAILURE_TEACHER",
    "SOURCE_STUDENT",
    "SOURCE_UNIFORM_TEACHER",
    "TRAINING_ALGORITHM",
    "FrozenTeacherValueWrapper",
    "TeacherValueBottleneckDetector",
    "TVKDDistributionalFastSACTeacherBC",
    "TVKDDistributionalFastSACTeacherBCConfig",
    "_prepare_tvkd_checkpoint",
    "_prepare_tvkd_fresh_source",
    "compute_teacher_value_terms",
    "main",
    "validate_tvkd_fastsac_bc_dagger_config",
]
