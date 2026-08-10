"""Dedicated PPO-BC DAgger + SAC-compatible critic entrypoint.

The rollout, replay, checkpoint, and W&B implementation remains in train.py.
This file owns only DAgger-specific defaults and fail-fast validation so the
two entrypoints cannot develop different training semantics.
"""

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
    "vaic_ppo_bc_dagger_student_sac_critic_v3"
)
EXPECTED_CRITIC_SEMANTICS = (
    "beta_independent_half_teacher_half_student_executed_action_"
    "stochastic_student_q_only_c51_clipped_double_q_v1"
)
EXPECTED_CONTROL_SEMANTICS = (
    "clipped_deterministic_mean_normalized_rms_safe_hysteresis_or_beta_v1"
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
    required = (
        "format",
        "format_version",
        "replay_id",
        "dagger_control_semantics",
        "replay_observation_semantics",
        "vecnorm_fingerprint",
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
        expected_action_clip = expected.get("action_clip")
        if expected_action_clip is None:
            backend = policy_state.get("dagger_backend_config")
            if isinstance(backend, dict):
                expected_action_clip = backend.get("dagger_action_clip")
        if expected_action_clip is not None:
            expected_action_clip = float(expected_action_clip)
            if (
                not math.isfinite(expected_action_clip)
                or expected_action_clip <= 0.0
            ):
                raise ValueError(
                    "Checkpoint teacher replay action clip is invalid"
                )
            actual_action_clip = replay.attrs.get("action_clip")
            if actual_action_clip is not None:
                if not math.isclose(
                    float(actual_action_clip),
                    expected_action_clip,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Teacher replay action_clip does not match checkpoint "
                        "lineage"
                    )
            else:
                # Legacy files did not record the support explicitly. Retain
                # compatibility only after proving every stored action lies in
                # the checkpoint's configured coordinates.
                actions = replay.get("actions")
                if actions is None:
                    raise ValueError("Teacher replay has no actions dataset")
                chunk_rows = 4096
                for start in range(0, int(actions.shape[0]), chunk_rows):
                    values = torch.as_tensor(
                        actions[start : start + chunk_rows]
                    )
                    if (
                        not torch.isfinite(values).all()
                        or (values.abs() > expected_action_clip).any()
                    ):
                        raise ValueError(
                            "Legacy teacher replay actions do not fit the "
                            "checkpoint action clip"
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
            "bc_dagger_checkpoint must be a SAC-critic-v3 PPO-BC DAgger "
            "checkpoint"
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
    if not isinstance(policy_state.get("q_backend_config"), dict):
        raise ValueError(
            "BC-DAgger resume checkpoint lacks its Q backend contract"
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


def apply_bc_dagger_iteration_controls(cfg: DictConfig) -> None:
    """Resolve iteration-facing CLI controls into the shared frame trainer."""
    iterations = cfg.get("bc_dagger_iterations", None)
    if iterations is not None:
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise ValueError("bc_dagger_iterations must be a positive integer")
        if iterations < 1:
            raise ValueError("bc_dagger_iterations must be a positive integer")
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
            cfg.total_frames = iterations * num_envs * train_every * world_size

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
    decay_rollouts = int(cfg.algo.dagger_beta_decay_rollouts)
    beta_zero_rollouts = (
        max(end_rollout - max(start_rollout, decay_rollouts), 0)
        if float(cfg.algo.dagger_beta_end) == 0.0
        else 0
    )
    safe_zero_iteration = cfg.algo.get(
        "dagger_safe_zero_iteration", None
    )
    safe_zero_rollouts = (
        max(end_rollout - max(start_rollout, safe_zero_iteration), 0)
        if safe_zero_iteration is not None
        else 0
    )
    return {
        "frames_per_rollout": frames_per_rollout,
        "total_rollouts": additional_rollouts,
        "start_rollout": start_rollout,
        "end_rollout": end_rollout,
        "decay_rollouts": decay_rollouts,
        "beta_zero_rollouts": beta_zero_rollouts,
        "safe_zero_rollouts": safe_zero_rollouts,
    }


def validate_bc_dagger_config(cfg: DictConfig) -> None:
    apply_bc_dagger_iteration_controls(cfg)
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
    apply_bc_dagger_iteration_controls(cfg)
    resume = prepare_bc_dagger_checkpoint(cfg)
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
            f"mode={control_mode}, normalized RMS takeover>"
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
