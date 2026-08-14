"""CLI for frozen-Teacher-value TVKD FastSAC + adaptive Student BC.

The algorithm lives in the installed ``active_adaptation.learning.ppo``
package so Hydra can import it both from this direct script and from tests.
This requested filename owns only source-checkpoint preparation, validation,
the Hydra surface, and delegation to the shared training runner.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    ACTOR_BACKEND,
    CHECKPOINT_VERSION,
    EXPECTED_ACTOR_IN_KEYS,
    EXPECTED_ALGO_NAME,
    EXPECTED_ALGO_TARGET,
    SOURCE_FAILURE_TEACHER,
    SOURCE_STUDENT,
    SOURCE_UNIFORM_TEACHER,
    TRAINING_ALGORITHM,
    FrozenTeacherValueWrapper,
    TeacherValueBCScheduler,
    TVKDDistributionalFastSACTeacherBC,
    TVKDDistributionalFastSACTeacherBCConfig,
    _validate_tvkd_algorithm_config,
    compute_source_separated_bc_losses,
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
    policy_state: Mapping, source_algo: Mapping
) -> None:
    """Fail before W&B/Isaac startup when continuation state is incomplete."""

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

    scheduler = policy_state.get("teacher_value_bc_scheduler")
    for name in (
        "residual_scale_ema",
        "risk_ema",
        "num_updates",
        "current_lambda_bc_student",
    ):
        if name not in scheduler:
            raise ValueError(f"TVKD resume scheduler lacks {name}")
    if (
        not isinstance(scheduler["num_updates"], int)
        or isinstance(scheduler["num_updates"], bool)
        or scheduler["num_updates"] < 0
    ):
        raise ValueError("TVKD resume scheduler update count is invalid")
    for name in ("residual_scale_ema", "risk_ema", "current_lambda_bc_student"):
        value = scheduler[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"TVKD resume scheduler {name} is invalid")
    if (
        float(scheduler["residual_scale_ema"]) <= 0.0
        or not 0.0 <= float(scheduler["risk_ema"]) <= 1.0
    ):
        raise ValueError("TVKD resume scheduler scale/risk is invalid")
    if (
        not float(source_algo.get("student_bc_lambda_min"))
        <= float(scheduler["current_lambda_bc_student"])
        <= float(source_algo.get("student_bc_lambda_max"))
    ):
        raise ValueError("TVKD resume scheduler coefficient is out of bounds")

    failure = policy_state.get("failure_phase_curriculum_state")
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

    action_contract = policy_state.get("action_contract")
    if not isinstance(
        action_contract.get("joint_names"), (list, tuple)
    ) or not isinstance(action_contract.get("fingerprint"), str):
        raise ValueError("TVKD resume action contract is incomplete")
    if not isinstance(policy_state.get("q_backend_config"), Mapping):
        raise ValueError("TVKD resume checkpoint lacks Q backend config")


def _prepare_tvkd_fresh_source(cfg: DictConfig) -> dict | None:
    """Load the baseline PPO source and mirror its ValueNorm construction."""
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


def _prepare_tvkd_checkpoint(cfg: DictConfig) -> dict | None:
    """Validate a model-only TVKD continuation and preserve its scheduler."""
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
    resolved = parse_checkpoint_path(os.fspath(requested), download_replay=False)
    if resolved is None or not os.path.isfile(resolved):
        raise FileNotFoundError(f"TVKD resume checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy")
    source_cfg = checkpoint.get("cfg")
    if not isinstance(policy_state, Mapping):
        raise ValueError("TVKD resume checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), Mapping):
        raise ValueError("TVKD resume checkpoint has no VecNorm state")
    if not isinstance(source_cfg, Mapping):
        raise ValueError("TVKD resume checkpoint has no saved config")
    if policy_state.get("training_algorithm") != TRAINING_ALGORITHM:
        raise ValueError("fastsac_bc_dagger_checkpoint is not a TVKD checkpoint")
    if int(policy_state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("TVKD resume checkpoint version mismatch")
    if policy_state.get("actor_backend") != ACTOR_BACKEND:
        raise ValueError("TVKD resume checkpoint actor backend mismatch")
    for name in (
        "teacher_value_bc_scheduler",
        "frozen_teacher_state",
        "failure_phase_curriculum_state",
        "optimizer_resume_state",
        "action_contract",
        "perception_initialization",
    ):
        if not isinstance(policy_state.get(name), Mapping):
            raise ValueError(f"TVKD resume checkpoint lacks {name!r}")
    vecnorm_fingerprint = policy_state.get("vecnorm_fingerprint")
    if not isinstance(vecnorm_fingerprint, str) or not vecnorm_fingerprint:
        raise ValueError("TVKD resume checkpoint lacks VecNorm fingerprint")
    backend = policy_state.get("dagger_backend_config")
    if not isinstance(backend, Mapping):
        raise ValueError("TVKD resume checkpoint lacks backend config")
    source_value_norm = _validate_teacher_return_contract(source_cfg, cfg)
    _validate_same_stage_task_contract(source_cfg, cfg)
    # ValueNorm changes the module type, so mirror the saved construction
    # choice before comparing the complete same-stage algorithm contract.
    with open_dict(cfg.algo):
        cfg.algo.value_norm = source_value_norm
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
    # Checkpoints written before this explicit provenance field used the TVKD
    # v1 every-Critic cadence.  Normalize that known legacy omission without
    # relaxing any other same-stage algorithm comparison.
    source_algo_contract.setdefault("sac_alpha_update_cadence", "critic")
    runtime_algo_contract = OmegaConf.to_container(
        cfg.algo, resolve=True, enum_to_str=True
    )
    if source_algo_contract != runtime_algo_contract:
        raise ValueError("TVKD resume algorithm config does not match checkpoint")
    _validate_tvkd_resume_policy_state(policy_state, source_algo)
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
        cfg.checkpoint_path = resolved
        cfg._tvkd_model_only_resume = True
        # Suppress generic H5 discovery: this continuation deliberately
        # rebuilds the two raw online rings with a fresh Teacher prefill.
        cfg._bc_dagger_fresh_source = True
        cfg.tvkd_resume_rollout_count = int(rollout_count)
    print(
        "TVKD model-state continuation: restored checkpoint at main rollout "
        f"{rollout_count}; the raw Teacher/Student replay rings will be rebuilt."
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
        # Baseline validation locks its new production entrypoint to Actor
        # cadence. TVKD itself is locked above to its legacy Critic cadence.
        baseline_cfg.algo.sac_alpha_update_cadence = "actor"
    # The baseline intentionally rejects continuation because it has no model
    # for TVKD's scheduler-aware, fresh-ring resume contract.  Every topology
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
    if resume_rollouts and bool(baseline_cfg.algo.load_pretrained_perception):
        # A TVKD checkpoint already owns every online/EMA perception child.
        # Do not require the historical warm-start file to remain available;
        # resume restores the saved children and optimizer state directly.
        with open_dict(baseline_cfg.algo):
            baseline_cfg.algo.load_pretrained_perception = False
            baseline_cfg.algo.perception_checkpoint_path = None
            baseline_cfg.algo.train_perception = True
    validate_fastsac_bc_dagger_config(baseline_cfg)


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
    print(
        "TVKD Distributional FastSAC + adaptive Student-BC schedule: "
        f"prefill=until {schedule['prefill_target_rows']} Teacher rows, "
        f"main_additional={schedule['main_rollouts']}, "
        f"main_range=[{start_rollout}, "
        f"{start_rollout + schedule['main_rollouts']}), "
        f"frames/rollout={schedule['frames_per_rollout']}; "
        f"tvkd_lambda={float(cfg.algo.tvkd_lambda):g}, "
        f"alpha_update_cadence={cfg.algo.sac_alpha_update_cadence}, "
        "sources=Student 50% / uniform Teacher 35% / failure Teacher 15%"
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
    "TeacherValueBCScheduler",
    "TVKDDistributionalFastSACTeacherBC",
    "TVKDDistributionalFastSACTeacherBCConfig",
    "_prepare_tvkd_checkpoint",
    "_prepare_tvkd_fresh_source",
    "compute_source_separated_bc_losses",
    "compute_teacher_value_terms",
    "main",
    "validate_tvkd_fastsac_bc_dagger_config",
]
