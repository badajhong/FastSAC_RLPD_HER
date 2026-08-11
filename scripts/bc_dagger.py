"""Dedicated PPO-BC DAgger + SAC-compatible critic entrypoint.

The rollout, replay, checkpoint, and W&B implementation remains in train.py.
This file owns only DAgger-specific defaults and fail-fast validation so the
two entrypoints cannot develop different training semantics.
"""

import hashlib
import json
import math
import os

import hydra
import torch
from omegaconf import DictConfig, open_dict

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path

try:
    from .helpers import find_local_teacher_replay
    from .train import run_training
except ImportError:
    from helpers import find_local_teacher_replay
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
EXPECTED_ALGO_NAME = "ppo_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.ppo_bc_dagger.PPOBCDaggerFinetune"
)
EXPECTED_TRAINING_ALGORITHM = (
    "vaic_ppo_bc_dagger_student_sac_critic_v6"
)
EXPECTED_ACTOR_BACKEND = "vaic_ppo_latent_tanh_bc_dagger_v4"
EXPECTED_CRITIC_SEMANTICS = (
    "beta_independent_half_teacher_half_student_safety_envelope_action_"
    "unclipped_nominal_joint_q_coordinates_bc_centered_bounded_residual_"
    "stochastic_student_q_only_c51_clipped_double_q_v5"
)
EXPECTED_CONTROL_SEMANTICS = (
    "safety_envelope_latent_tanh_mean_envelope_normalized_safe_or_beta_v5"
)
EXPECTED_REPLAY_FORMAT = "vaic_ppo_bc_dagger_teacher_buffer"
EXPECTED_REPLAY_FORMAT_VERSION = 5
EXPECTED_REPLAY_OBSERVATION_SEMANTICS = "raw_pre_vecnorm_sample_current_v1"
EXPECTED_ACTION_PARAMETERIZATION = (
    "absolute_safety_envelope_teacher_or_latent_tanh_student_v5"
)
EXPECTED_FRESH_ACTOR_INITIALIZATION_SEMANTICS = (
    "physical_absolute_head_safety_tanh_linearized_at_zero_v2"
)
EXPECTED_ACTION_CONTRACT_SEMANTICS = (
    "separate_execution_support_q_and_entropy_coordinates_v2"
)
REQUIRED_RESUME_OPTIMIZERS = {
    "bc_optimizer",
    "q_optimizer",
    "adapt_optimizer",
}
REQUIRED_RESUME_STATE = {
    "bc_update_count",
    "dagger_environment_steps",
    "dagger_rng_state",
    "next_iter",
    "q_rng_state",
    "q_update_count",
    "sac_action_rng_state",
}


def _validate_action_contract(contract, *, context: str) -> dict:
    """Validate separated execution, Q, and entropy action coordinates."""
    if not isinstance(contract, dict):
        raise ValueError(f"{context} has no executable action contract")
    if contract.get("semantics") != EXPECTED_ACTION_CONTRACT_SEMANTICS:
        raise ValueError(f"{context} has incompatible action-contract semantics")
    fingerprint = str(contract.get("fingerprint", ""))
    payload = dict(contract)
    payload.pop("fingerprint", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    expected_fingerprint = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError(f"{context} action-contract fingerprint is invalid")

    joint_names = contract.get("joint_names")
    action_low = contract.get("action_low")
    action_high = contract.get("action_high")
    q_action_center = contract.get("q_action_center")
    q_action_scale = contract.get("q_action_scale")
    if (
        not isinstance(joint_names, list)
        or not joint_names
        or not isinstance(action_low, list)
        or not isinstance(action_high, list)
        or len(action_low) != len(joint_names)
        or len(action_high) != len(joint_names)
        or not isinstance(q_action_center, list)
        or not isinstance(q_action_scale, list)
        or len(q_action_center) != len(joint_names)
        or len(q_action_scale) != len(joint_names)
    ):
        raise ValueError(f"{context} action contract has invalid joint bounds")
    for low, high in zip(action_low, action_high):
        low = float(low)
        high = float(high)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(
                f"{context} action contract has invalid executable bounds"
            )
    if any(
        not math.isfinite(float(center))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        for center, scale in zip(q_action_center, q_action_scale)
    ):
        raise ValueError(f"{context} action contract has invalid Q coordinates")
    if contract.get("q_action_clamp", "missing") is not None:
        raise ValueError(f"{context} Q action transform must be unclipped")
    if not str(contract.get("execution_support_fingerprint", "")).startswith(
        "sha256:"
    ):
        raise ValueError(f"{context} lacks an execution-support fingerprint")
    if not str(contract.get("q_action_transform_fingerprint", "")).startswith(
        "sha256:"
    ):
        raise ValueError(f"{context} lacks a Q-transform fingerprint")
    if not str(contract.get("entropy_reference_fingerprint", "")).startswith(
        "sha256:"
    ):
        raise ValueError(f"{context} lacks an entropy-reference fingerprint")
    return contract


def _decode_h5_action_contract(value, *, context: str) -> dict:
    if value is None:
        raise ValueError(f"{context} is missing its action_contract attribute")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        contract = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} action_contract is not valid JSON") from exc
    return _validate_action_contract(contract, context=context)


def _resolve_bc_dagger_checkpoint(
    path, *, download_replay: bool, replay_filename: str
) -> str:
    """Resolve a local/W&B resume checkpoint and optional immutable H5."""
    value = os.path.expanduser(os.fspath(path))
    if value.startswith("run:"):
        resolved = parse_checkpoint_path(
            value,
            download_replay=download_replay,
            replay_filename=replay_filename,
        )
    else:
        resolved = os.path.realpath(hydra.utils.to_absolute_path(value))
    if resolved is None or not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"BC-DAgger resume checkpoint does not exist: {resolved}"
        )
    return os.path.realpath(resolved)


def _validate_frozen_teacher_replay(
    source_path: str, policy_state: dict
) -> None:
    """Validate replay lineage without requiring the old snapshot iteration."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to validate the BC-DAgger teacher replay copy"
        ) from exc

    expected = policy_state.get(
        "teacher_replay_state",
        policy_state.get("frozen_teacher_replay_source_state"),
    )
    if not isinstance(expected, dict):
        raise ValueError(
            "BC-DAgger checkpoint has no teacher replay lineage metadata"
        )
    checkpoint_action_contract = _validate_action_contract(
        policy_state.get("action_contract"), context="BC-DAgger checkpoint"
    )
    if expected.get("action_contract") != checkpoint_action_contract:
        raise ValueError(
            "Checkpoint teacher replay lineage action contract does not match "
            "the policy action contract"
        )
    expected_generation = {
        "format": EXPECTED_REPLAY_FORMAT,
        "format_version": EXPECTED_REPLAY_FORMAT_VERSION,
        "actor_backend": EXPECTED_ACTOR_BACKEND,
        "dagger_control_semantics": EXPECTED_CONTROL_SEMANTICS,
        "replay_observation_semantics": (
            EXPECTED_REPLAY_OBSERVATION_SEMANTICS
        ),
        "action_parameterization": EXPECTED_ACTION_PARAMETERIZATION,
    }
    generation_mismatches = {
        name: (expected.get(name), value)
        for name, value in expected_generation.items()
        if expected.get(name) != value
    }
    if generation_mismatches:
        raise ValueError(
            "BC-DAgger checkpoint teacher replay lineage uses an unsupported "
            f"format generation: {generation_mismatches}"
        )
    required = (
        "format",
        "format_version",
        "replay_id",
        "dagger_control_semantics",
        "replay_observation_semantics",
        "vecnorm_fingerprint",
        "actor_backend",
        "action_parameterization",
    )
    with h5py.File(source_path, "r") as replay:
        for name in required:
            expected_value = expected.get(name)
            actual_value = replay.attrs.get(name)
            if name == "format_version":
                equal = int(actual_value or 0) == int(expected_value or 0)
            else:
                equal = str(actual_value or "") == str(expected_value or "")
            if not equal:
                raise ValueError(
                    f"Teacher replay {name}={actual_value!r} does not match "
                    f"checkpoint lineage {expected_value!r}"
                )
        replay_action_contract = _decode_h5_action_contract(
            replay.attrs.get("action_contract"), context="Teacher replay"
        )
        if replay_action_contract != checkpoint_action_contract:
            raise ValueError(
                "Teacher replay executable action contract does not match "
                "checkpoint lineage"
            )
        expected_action_clip = expected.get("action_clip")
        if expected_action_clip is None:
            backend = policy_state.get("dagger_backend_config")
            if isinstance(backend, dict):
                expected_action_clip = backend.get("dagger_action_clip")
        if expected_action_clip is None:
            raise ValueError(
                "Checkpoint teacher replay lineage lacks its final action "
                "safety clip"
            )
        expected_action_clip = float(expected_action_clip)
        if not math.isfinite(expected_action_clip) or expected_action_clip <= 0.0:
            raise ValueError(
                "Checkpoint teacher replay action safety clip is invalid"
            )
        support_max = max(
            abs(float(value))
            for key in ("action_low", "action_high")
            for value in checkpoint_action_contract[key]
        )
        if not math.isclose(
            expected_action_clip, support_max, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Checkpoint action safety clip does not exactly match the "
                "serialized actor execution support"
            )
        actual_action_clip = replay.attrs.get("action_clip")
        if actual_action_clip is None or not math.isclose(
            float(actual_action_clip),
            expected_action_clip,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Teacher replay action_clip safety guard does not match "
                "checkpoint lineage"
            )


def prepare_bc_dagger_checkpoint(cfg: DictConfig) -> dict | None:
    """Configure model/optimizer continuation with an immutable old H5.

    ``checkpoint_path`` remains the shared trainer's internal model source.
    The dedicated alias prevents that generic path from treating the adjacent
    teacher H5 as mutable same-stage state. The H5 is still read to refill the
    in-memory fixed teacher critic partition; it is never appended on resume.
    """
    requested = cfg.get("bc_dagger_checkpoint", None)
    if requested is None:
        return None
    if cfg.get("teacher_replay_buffer_path", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint finds its paired immutable teacher replay "
            "automatically; remove teacher_replay_buffer_path"
        )
    if cfg.algo.get("teacher_buffer_path", None) is not None:
        raise ValueError(
            "bc_dagger_checkpoint finds its paired immutable teacher replay "
            "automatically; remove algo.teacher_buffer_path"
        )

    copy_replay = bool(cfg.get("bc_dagger_copy_teacher_replay", True))
    replay_filename = str(
        cfg.algo.get("teacher_buffer_filename", "teacher_replay_buffer.h5")
    )
    if (
        not replay_filename
        or replay_filename in (".", "..")
        or os.path.basename(replay_filename) != replay_filename
    ):
        raise ValueError(
            "algo.teacher_buffer_filename must be a plain file basename"
        )
    resolved = _resolve_bc_dagger_checkpoint(
        requested,
        download_replay=copy_replay,
        replay_filename=replay_filename,
    )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, dict):
        raise ValueError("BC-DAgger resume checkpoint has no policy state")
    if not isinstance(checkpoint.get("vecnorm"), dict):
        raise ValueError("BC-DAgger resume checkpoint has no VecNorm state")
    if policy_state.get("training_algorithm") != EXPECTED_TRAINING_ALGORITHM:
        raise ValueError(
            "bc_dagger_checkpoint must be a SAC-critic-v6 PPO-BC DAgger "
            "checkpoint"
        )
    if policy_state.get("actor_backend") != EXPECTED_ACTOR_BACKEND:
        raise ValueError(
            "BC-DAgger resume checkpoint has an incompatible actor backend"
        )
    if policy_state.get("critic_learning_semantics") != (
        EXPECTED_CRITIC_SEMANTICS
    ):
        raise ValueError(
            "BC-DAgger resume checkpoint has incompatible critic semantics"
        )
    if policy_state.get("dagger_control_semantics") != (
        EXPECTED_CONTROL_SEMANTICS
    ):
        raise ValueError(
            "BC-DAgger resume checkpoint has incompatible control semantics"
        )
    backend_config = policy_state.get("dagger_backend_config")
    if not isinstance(backend_config, dict) or backend_config.get(
        "fresh_ppo_actor_initialization_semantics"
    ) != EXPECTED_FRESH_ACTOR_INITIALIZATION_SEMANTICS:
        raise ValueError(
            "BC-DAgger resume checkpoint lacks the fresh PPO physical-to-latent "
            "actor migration contract"
        )
    if not isinstance(policy_state.get("q_backend_config"), dict):
        raise ValueError(
            "BC-DAgger resume checkpoint lacks its Q backend contract"
        )
    action_contract = _validate_action_contract(
        policy_state.get("action_contract"), context="BC-DAgger checkpoint"
    )
    if policy_state["q_backend_config"].get(
        "q_action_transform_fingerprint"
    ) != action_contract["q_action_transform_fingerprint"]:
        raise ValueError(
            "BC-DAgger resume checkpoint Q-transform fingerprints do not "
            "match"
        )
    optimizer_state = policy_state.get("optimizer_resume_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("BC-DAgger resume checkpoint has no optimizer state")
    missing_optimizers = REQUIRED_RESUME_OPTIMIZERS.difference(optimizer_state)
    if missing_optimizers:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing optimizer state: "
            f"{sorted(missing_optimizers)}"
        )
    required_modules = {
        "actor_adapt",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
    }
    missing_modules = required_modules.difference(policy_state)
    if missing_modules:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing trained modules: "
            f"{sorted(missing_modules)}"
        )
    missing_state = REQUIRED_RESUME_STATE.difference(policy_state)
    if missing_state:
        raise ValueError(
            "BC-DAgger resume checkpoint is missing continuation state: "
            f"{sorted(missing_state)}"
        )

    replay_source = find_local_teacher_replay(resolved, replay_filename)
    if replay_source is None:
        raise FileNotFoundError(
            "SAC-critic BC-DAgger resume requires the paired immutable "
            f"{replay_filename} beside the checkpoint or at its output root "
            "to refill the persistent 50% teacher critic partition."
        )
    _validate_frozen_teacher_replay(replay_source, policy_state)

    rollout_count = int(policy_state.get("dagger_rollout_count", -1))
    if rollout_count < 0:
        raise ValueError(
            "BC-DAgger resume checkpoint has no valid dagger_rollout_count"
        )
    environment_steps = int(
        policy_state.get(
            "dagger_environment_steps",
            rollout_count * int(cfg.algo.train_every),
        )
    )
    previous_source = cfg.get("checkpoint_path", None)
    if previous_source is not None and os.fspath(previous_source) != resolved:
        print(
            "BC DAgger resume: bc_dagger_checkpoint overrides the fresh PPO "
            f"checkpoint_path={previous_source}"
        )

    with open_dict(cfg):
        cfg.bc_dagger_checkpoint = resolved
        cfg.checkpoint_path = resolved
        cfg.bc_dagger_resume_rollout_count = rollout_count
        cfg.bc_dagger_resume_environment_steps = environment_steps
        cfg._bc_dagger_model_only_resume = True
        cfg._bc_dagger_teacher_replay_copy_source = replay_source
        cfg._bc_dagger_teacher_replay_copy_path = None
        cfg.teacher_replay_buffer_path = None
        cfg.algo.teacher_buffer_path = None
        cfg.algo.save_teacher_buffer = False

    return {
        "path": resolved,
        "rollout_count": rollout_count,
        "environment_steps": environment_steps,
        "teacher_replay_source": replay_source,
    }


def _inline_iteration(name: str, value, *, positive: bool) -> int:
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer")
    return int(value)


def apply_inline_finalization_controls(cfg: DictConfig) -> dict | None:
    """Map the compact joint/perception/actor/Q tail onto staged ownership.

    The hidden joint count makes this function idempotent: once ``total_frames``
    includes the appended tail, later validation/schedule calls must not infer a
    progressively larger joint budget from that rewritten frame count.
    """
    enabled = cfg.get("bc_dagger_inline_finalization", False)
    if not isinstance(enabled, bool):
        raise ValueError("bc_dagger_inline_finalization must be boolean")
    if not enabled:
        return None
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        raise ValueError(
            "BC-DAgger inline finalization does not support same-stage resume: "
            "pre-final checkpoints deliberately have no paired H5. Start from "
            "the fresh PPO checkpoint, or set "
            "bc_dagger_inline_finalization=false for the legacy resume path."
        )
    if cfg.get("teacher_replay_buffer_path", None) is not None or cfg.algo.get(
        "teacher_buffer_path", None
    ) is not None:
        raise ValueError(
            "BC-DAgger inline finalization requires a fresh final replay and "
            "cannot restore an existing teacher H5"
        )

    stored_joint = cfg.get("_bc_dagger_inline_joint_iterations", None)
    requested_joint = cfg.get("bc_dagger_iterations", None)
    if stored_joint is not None:
        joint_iterations = _inline_iteration(
            "_bc_dagger_inline_joint_iterations",
            stored_joint,
            positive=True,
        )
        if requested_joint is not None and _inline_iteration(
            "bc_dagger_iterations", requested_joint, positive=True
        ) != joint_iterations:
            raise ValueError(
                "bc_dagger_iterations changed after inline finalization was "
                "resolved"
            )
    elif requested_joint is not None:
        joint_iterations = _inline_iteration(
            "bc_dagger_iterations", requested_joint, positive=True
        )
    else:
        num_envs = int(cfg.task.num_envs)
        train_every = int(cfg.algo.train_every)
        world_size = int(aa.get_world_size())
        total_frames = int(cfg.total_frames)
        if min(num_envs, train_every, world_size, total_frames) < 1:
            raise ValueError(
                "task.num_envs, algo.train_every, distributed world size, "
                "and total_frames must be positive"
            )
        joint_iterations = total_frames // (
            num_envs * train_every * world_size
        )
        if joint_iterations < 1:
            raise ValueError(
                "total_frames does not contain one complete joint rollout"
            )

    perception_iterations = _inline_iteration(
        "perception_consolidation_iterations",
        cfg.get("perception_consolidation_iterations", None),
        positive=False,
    )
    actor_iterations = _inline_iteration(
        "actor_realignment_iterations",
        cfg.get("actor_realignment_iterations", None),
        positive=False,
    )
    calibration_iterations = _inline_iteration(
        "replay_q_calibration_iterations",
        cfg.get("replay_q_calibration_iterations", None),
        positive=True,
    )
    probability = cfg.get("calibration_teacher_probability", None)
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 < float(probability) < 1.0
    ):
        raise ValueError(
            "calibration_teacher_probability must be finite and strictly "
            "between zero and one"
        )

    controls = {
        "joint_iterations": joint_iterations,
        "perception_consolidation_iterations": perception_iterations,
        "actor_realignment_iterations": actor_iterations,
        "replay_q_calibration_iterations": calibration_iterations,
        "calibration_teacher_probability": float(probability),
    }
    total_iterations = sum(
        controls[name]
        for name in (
            "joint_iterations",
            "perception_consolidation_iterations",
            "actor_realignment_iterations",
            "replay_q_calibration_iterations",
        )
    )
    num_envs = int(cfg.task.num_envs)
    train_every = int(cfg.algo.train_every)
    world_size = int(aa.get_world_size())
    if min(num_envs, train_every, world_size) < 1:
        raise ValueError(
            "task.num_envs, algo.train_every, and distributed world size "
            "must be positive"
        )

    with open_dict(cfg):
        cfg._bc_dagger_inline_joint_iterations = joint_iterations
        cfg._bc_dagger_inline_finalization_applied = True
        # Prevent helpers.py from auto-pairing an unrelated H5 found beside
        # the fresh PPO source checkpoint.
        cfg._bc_dagger_staging_source = True
        cfg.total_frames = (
            total_iterations * num_envs * train_every * world_size
        )
        cfg.teacher_replay_buffer_path = None
    with open_dict(cfg.algo):
        cfg.algo.teacher_buffer_path = None
        cfg.algo.save_teacher_buffer = True
        cfg.algo.dagger_staging_enabled = True
        cfg.algo.dagger_stage_joint_warmup_iterations = joint_iterations
        cfg.algo.dagger_stage_cycles = 0
        cfg.algo.dagger_stage_perception_iterations = 0
        cfg.algo.dagger_stage_actor_iterations = 0
        cfg.algo.dagger_stage_final_perception_iterations = (
            perception_iterations
        )
        cfg.algo.dagger_stage_final_actor_iterations = actor_iterations
        cfg.algo.dagger_stage_calibration_iterations = calibration_iterations
        cfg.algo.dagger_stage_calibration_control_mode = "beta"
        cfg.algo.dagger_stage_calibration_teacher_probability = float(
            probability
        )
        cfg.algo.dagger_stage_h5_final_only = True
    return controls


def prepare_fresh_bc_dagger_source(cfg: DictConfig) -> dict | None:
    """Validate and mark any new BC-DAgger run's fresh PPO source."""
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        return None
    if (
        cfg.get("teacher_replay_buffer_path", None) is not None
        or cfg.algo.get("teacher_buffer_path", None) is not None
    ):
        raise ValueError(
            "A fresh BC-DAgger run must collect a new replay lineage; remove "
            "teacher_replay_buffer_path and algo.teacher_buffer_path"
        )
    try:
        from .stage_bc_dagger import (
            _resolve_source_checkpoint,
            _validate_source_checkpoint,
        )
    except ImportError:
        from stage_bc_dagger import (  # type: ignore
            _resolve_source_checkpoint,
            _validate_source_checkpoint,
        )

    source_path = _resolve_source_checkpoint(cfg.get("checkpoint_path", None))
    checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    policy_state = _validate_source_checkpoint(checkpoint, cfg)
    with open_dict(cfg):
        cfg.checkpoint_path = source_path
        cfg._bc_dagger_fresh_source = True
        cfg._bc_dagger_model_only_resume = False
        if bool(cfg.get("bc_dagger_inline_finalization", False)):
            cfg._bc_dagger_staging_source = True
            cfg._bc_dagger_finalization_source = False
            cfg._bc_dagger_finalize = False
    return {
        "path": source_path,
        "source_last_iter": int(policy_state.get("last_iter", -1)),
    }


def prepare_inline_bc_dagger_source(cfg: DictConfig) -> dict | None:
    """Require a fresh PPO teacher for the non-resumable inline schedule."""
    if not bool(cfg.get("bc_dagger_inline_finalization", False)):
        return None
    return prepare_fresh_bc_dagger_source(cfg)


def apply_bc_dagger_iteration_controls(cfg: DictConfig) -> None:
    """Resolve iteration-facing CLI controls into the shared frame trainer."""
    inline = apply_inline_finalization_controls(cfg)
    iterations = (
        inline["joint_iterations"]
        if inline is not None
        else cfg.get("bc_dagger_iterations", None)
    )
    if iterations is not None:
        iterations = _inline_iteration(
            "bc_dagger_iterations", iterations, positive=True
        )
        num_envs = int(cfg.task.num_envs)
        train_every = int(cfg.algo.train_every)
        if num_envs < 1 or train_every < 1:
            raise ValueError("task.num_envs and algo.train_every must be positive")
        world_size = int(aa.get_world_size())
        if world_size < 1:
            raise ValueError("distributed world size must be positive")
        with open_dict(cfg):
            # train.py interprets total_frames as the all-rank budget and divides
            # it by world_size before constructing each rank's iteration range.
            total_iterations = iterations
            if inline is not None:
                total_iterations += (
                    inline["perception_consolidation_iterations"]
                    + inline["actor_realignment_iterations"]
                    + inline["replay_q_calibration_iterations"]
                )
            cfg.total_frames = (
                total_iterations * num_envs * train_every * world_size
            )

    beta_zero_iteration = cfg.algo.get(
        "dagger_beta_zero_iteration", None
    )
    if beta_zero_iteration is not None:
        if (
            isinstance(beta_zero_iteration, bool)
            or not isinstance(beta_zero_iteration, int)
            or beta_zero_iteration < 1
        ):
            raise ValueError(
                "algo.dagger_beta_zero_iteration must be a positive integer"
            )
        with open_dict(cfg.algo):
            # _linear_teacher_probability reaches dagger_beta_end exactly when
            # dagger_rollout_count equals this completed-rollout boundary.
            cfg.algo.dagger_beta_decay_rollouts = beta_zero_iteration

    safe_zero_iteration = cfg.algo.get(
        "dagger_safe_zero_iteration", None
    )
    if safe_zero_iteration is not None and (
        isinstance(safe_zero_iteration, bool)
        or not isinstance(safe_zero_iteration, int)
        or safe_zero_iteration < 1
    ):
        raise ValueError(
            "algo.dagger_safe_zero_iteration must be a positive integer"
        )


def bc_dagger_rollout_schedule(cfg: DictConfig) -> dict[str, int]:
    """Return the effective additional rollout schedule for this process."""
    apply_bc_dagger_iteration_controls(cfg)
    inline = apply_inline_finalization_controls(cfg)
    num_envs = int(cfg.task.num_envs)
    train_every = int(cfg.algo.train_every)
    total_frames = int(cfg.total_frames)
    if num_envs < 1 or train_every < 1 or total_frames < 1:
        raise ValueError(
            "task.num_envs, algo.train_every, and total_frames must be positive"
        )
    frames_per_rollout = num_envs * train_every
    world_size = int(aa.get_world_size())
    if world_size < 1:
        raise ValueError("distributed world size must be positive")
    per_rank_frames = total_frames // world_size
    additional_rollouts = per_rank_frames // frames_per_rollout
    if additional_rollouts < 1:
        raise ValueError("total_frames does not contain one complete rollout")
    start_rollout = int(cfg.get("bc_dagger_resume_rollout_count", 0))
    if start_rollout < 0:
        raise ValueError("bc_dagger_resume_rollout_count must be non-negative")
    end_rollout = start_rollout + additional_rollouts
    controller_end_rollout = end_rollout
    pure_student_tail = 0
    if inline is not None:
        expected_rollouts = sum(
            inline[name]
            for name in (
                "joint_iterations",
                "perception_consolidation_iterations",
                "actor_realignment_iterations",
                "replay_q_calibration_iterations",
            )
        )
        if start_rollout != 0 or additional_rollouts != expected_rollouts:
            raise ValueError(
                "inline BC-DAgger frame budget disagrees with its phase sum"
            )
        controller_end_rollout = inline["joint_iterations"]
        pure_student_tail = (
            inline["perception_consolidation_iterations"]
            + inline["actor_realignment_iterations"]
        )
    decay_rollouts = int(cfg.algo.dagger_beta_decay_rollouts)
    beta_zero_rollouts = (
        max(
            controller_end_rollout - max(start_rollout, decay_rollouts),
            0,
        )
        + pure_student_tail
        if float(cfg.algo.dagger_beta_end) == 0.0
        else 0
    )
    safe_zero_iteration = cfg.algo.get(
        "dagger_safe_zero_iteration", None
    )
    safe_zero_rollouts = (
        max(
            controller_end_rollout
            - max(start_rollout, safe_zero_iteration),
            0,
        )
        + pure_student_tail
        if safe_zero_iteration is not None
        else 0
    )
    schedule = {
        "frames_per_rollout": frames_per_rollout,
        "total_rollouts": additional_rollouts,
        "start_rollout": start_rollout,
        "end_rollout": end_rollout,
        "decay_rollouts": decay_rollouts,
        "beta_zero_rollouts": beta_zero_rollouts,
        "safe_zero_rollouts": safe_zero_rollouts,
    }
    if inline is not None:
        schedule.update(
            {
                "joint_rollouts": inline["joint_iterations"],
                "perception_consolidation_rollouts": inline[
                    "perception_consolidation_iterations"
                ],
                "actor_realignment_rollouts": inline[
                    "actor_realignment_iterations"
                ],
                "replay_q_calibration_rollouts": inline[
                    "replay_q_calibration_iterations"
                ],
                "tail_rollouts": additional_rollouts
                - inline["joint_iterations"],
            }
        )
    return schedule


def validate_bc_dagger_config(cfg: DictConfig) -> None:
    apply_bc_dagger_iteration_controls(cfg)
    inline = apply_inline_finalization_controls(cfg)
    algo_name = cfg.algo.get("name")
    if algo_name != EXPECTED_ALGO_NAME:
        raise ValueError(
            "scripts/bc_dagger.py only supports "
            "algo=ppo_bc_dagger_finetune; got "
            f"algo.name={algo_name!r}. Use scripts/train.py for other algorithms."
        )
    algo_target = cfg.algo.get("_target_")
    if algo_target != EXPECTED_ALGO_TARGET:
        raise ValueError(
            "scripts/bc_dagger.py requires the PPO-BC DAgger implementation; "
            f"got algo._target_={algo_target!r}"
        )
    if cfg.algo.get("phase") != "finetune":
        raise ValueError("PPO-BC DAgger must run with algo.phase=finetune")
    if cfg.algo.get("vecnorm") != "eval":
        raise ValueError("PPO-BC DAgger must run with algo.vecnorm=eval")
    if not bool(cfg.algo.get("dagger_replay_raw_observations", False)):
        raise ValueError(
            "PPO-BC DAgger requires dagger_replay_raw_observations=true"
        )
    control_mode = str(cfg.algo.get("dagger_control_mode", "beta"))
    if control_mode not in ("beta", "safe", "hybrid"):
        raise ValueError(
            "algo.dagger_control_mode must be beta, safe, or hybrid"
        )
    safe_zero_iteration = cfg.algo.get(
        "dagger_safe_zero_iteration", None
    )
    if safe_zero_iteration is not None and control_mode == "beta":
        raise ValueError(
            "algo.dagger_safe_zero_iteration requires safe or hybrid control"
        )
    release = float(cfg.algo.get("dagger_safe_release_rms", float("nan")))
    takeover = float(cfg.algo.get("dagger_safe_takeover_rms", float("nan")))
    hold = cfg.algo.get("dagger_safe_min_teacher_steps", None)
    if not (
        math.isfinite(release)
        and math.isfinite(takeover)
        and 0.0 <= release < takeover <= 2.0
    ):
        raise ValueError(
            "SafeDAgger requires 0 <= release_rms < takeover_rms <= 2"
        )
    if isinstance(hold, bool) or not isinstance(hold, int) or hold < 1:
        raise ValueError(
            "algo.dagger_safe_min_teacher_steps must be a positive integer"
        )
    if (
        cfg.get("checkpoint_path") is None
        and cfg.get("bc_dagger_checkpoint", None) is None
    ):
        raise ValueError(
            "scripts/bc_dagger.py requires checkpoint_path pointing to the "
            "trained PPO teacher, or bc_dagger_checkpoint for same-stage resume"
        )
    if cfg.get("bc_dagger_checkpoint", None) is not None:
        if bool(cfg.algo.get("save_teacher_buffer", True)):
            raise ValueError(
                "bc_dagger_checkpoint must disable algo.save_teacher_buffer"
            )
        if (
            cfg.get("teacher_replay_buffer_path", None) is not None
            or cfg.algo.get("teacher_buffer_path", None) is not None
        ):
            raise ValueError(
                "bc_dagger_checkpoint cannot be combined with a teacher replay path"
            )
    if inline is not None:
        if not bool(cfg.algo.get("dagger_staging_enabled", False)):
            raise ValueError("inline BC-DAgger staging backend is disabled")
        if not bool(cfg.algo.get("dagger_stage_h5_final_only", False)):
            raise ValueError("inline BC-DAgger requires final-only H5 output")
        if not bool(cfg.algo.get("save_teacher_buffer", False)):
            raise ValueError("inline BC-DAgger must create a final fresh H5")
    schedule = bc_dagger_rollout_schedule(cfg)
    if (
        control_mode in ("beta", "hybrid")
        and cfg.algo.get("dagger_beta_zero_iteration", None) is not None
        and float(cfg.algo.dagger_beta_end) != 0.0
    ):
        raise ValueError(
            "algo.dagger_beta_zero_iteration requires dagger_beta_end=0"
        )
    if (
        control_mode in ("beta", "hybrid")
        and float(cfg.algo.dagger_beta_start) > 0.0
        and float(cfg.algo.dagger_beta_end) == 0.0
        and schedule["beta_zero_rollouts"] < 1
    ):
        raise ValueError(
            "the cumulative end rollout must exceed "
            "dagger_beta_zero_iteration/decay_rollouts so training includes "
            "a phase with "
            "no random beta teacher selections"
        )
    if (
        control_mode in ("safe", "hybrid")
        and safe_zero_iteration is not None
        and schedule["safe_zero_rollouts"] < 1
    ):
        raise ValueError(
            "the cumulative end rollout must exceed "
            "dagger_safe_zero_iteration so training includes a phase with "
            "SafeDAgger teacher control forced to zero"
        )


@hydra.main(config_path=CONFIG_PATH, config_name="bc_dagger", version_base=None)
def main(cfg: DictConfig):
    inline = apply_inline_finalization_controls(cfg)
    apply_bc_dagger_iteration_controls(cfg)
    resume = prepare_bc_dagger_checkpoint(cfg)
    if resume is None:
        if inline is not None:
            prepare_inline_bc_dagger_source(cfg)
        else:
            prepare_fresh_bc_dagger_source(cfg)
    validate_bc_dagger_config(cfg)
    schedule = bc_dagger_rollout_schedule(cfg)
    control_mode = str(cfg.algo.get("dagger_control_mode", "beta"))
    safe_zero_iteration = cfg.algo.get(
        "dagger_safe_zero_iteration", None
    )
    if control_mode == "safe":
        if safe_zero_iteration is None:
            control_schedule = "SafeDAgger state-wise control; beta unused"
        else:
            control_schedule = (
                "SafeDAgger teacher control is forced to zero from cumulative "
                f"rollout index {safe_zero_iteration}; student-only="
                f"{schedule['safe_zero_rollouts']} rollouts; beta unused"
            )
    else:
        control_schedule = (
            f"beta is zero from cumulative rollout index "
            f"{schedule['decay_rollouts']}, beta-random-free="
            f"{schedule['beta_zero_rollouts']} rollouts"
        )
        if control_mode == "hybrid" and safe_zero_iteration is not None:
            control_schedule += (
                "; SafeDAgger teacher control is forced to zero from "
                f"cumulative rollout index {safe_zero_iteration}, "
                f"safe-teacher-free={schedule['safe_zero_rollouts']} rollouts"
            )
    if resume is None:
        print(
            "BC DAgger schedule: "
            f"{schedule['total_rollouts']} rollouts "
            f"({schedule['frames_per_rollout']} frames each), "
            f"{control_schedule}"
        )
        if inline is not None:
            print(
                "Inline finalization: "
                f"joint={schedule['joint_rollouts']}, "
                "pure-student perception="
                f"{schedule['perception_consolidation_rollouts']}, "
                "pure-student frozen-perception actor BC="
                f"{schedule['actor_realignment_rollouts']}, "
                f"Q calibration={schedule['replay_q_calibration_rollouts']} "
                f"at teacher probability "
                f"{inline['calibration_teacher_probability']:.3f}; "
                "H5 is materialized only with checkpoint_final"
            )
    else:
        print(
            "BC DAgger resume schedule: "
            f"start={schedule['start_rollout']}, "
            f"additional={schedule['total_rollouts']}, "
            f"end={schedule['end_rollout']}, "
            f"{control_schedule}; "
            "teacher H5 collection/write/snapshot disabled; "
            "immutable local copy="
            f"{'enabled' if bool(cfg.get('bc_dagger_copy_teacher_replay', True)) else 'disabled'}; "
            "teacher critic refill=enabled"
        )
    print(
        "SAC-compatible critic warm start: "
        f"atoms={int(cfg.algo.get('q_num_atoms', 501))}, "
        f"batch={int(cfg.algo.get('q_batch_size', 512))}, "
        f"teacher/student=50/50, target tau="
        f"{float(cfg.algo.get('q_tau', 0.001)):g}, "
        "actor Q weighting=disabled (pure DAgger BC)"
    )
    if control_mode in ("safe", "hybrid"):
        print(
            "SafeDAgger control: "
            f"mode={control_mode}, safety-envelope-normalized RMS "
            "takeover>"
            f"{float(cfg.algo.dagger_safe_takeover_rms):g}, release<"
            f"{float(cfg.algo.dagger_safe_release_rms):g}, minimum teacher "
            f"hold={int(cfg.algo.dagger_safe_min_teacher_steps)} steps, "
            + (
                "no forced cutoff"
                if safe_zero_iteration is None
                else (
                    "forced off from cumulative rollout index "
                    f"{safe_zero_iteration}"
                )
            )
        )
    else:
        print("SafeDAgger control: disabled (legacy beta ablation)")
    return run_training(cfg)


if __name__ == "__main__":
    main()
