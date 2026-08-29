"""Collect a fresh fixed batch and inspect FastSAC Actor gradients read-only.

FastSAC-BC-DAgger checkpoints deliberately omit both replay rings, so an exact
historical minibatch cannot be reconstructed.  This runner instead reloads the
model for inference, collects fresh raw recurrent windows using the same
training rollout policy, and creates a deterministic Student/Teacher source
batch before calling the optimizer-free gradient probe.
"""

from __future__ import annotations

import json
import os
import copy
from contextlib import contextmanager

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

try:
    from ._isaaclab_bootstrap import AppLauncher
    from .helpers import make_env_policy
except ImportError:
    from _isaaclab_bootstrap import AppLauncher
    from helpers import make_env_policy

from active_adaptation.learning.ppo.fastsac_gradient_probe import (
    diagnose_fastsac_actor_gradients,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    Q_ACTUATOR_CONTEXT_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    _PERCEPTION_REPLAY_FIELDS,
)


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg")


def _checkpoint_runtime_config(cfg: DictConfig) -> DictConfig:
    """Use the checkpoint's exact task/algo config plus probe-only overrides."""
    checkpoint_path = os.path.realpath(
        hydra.utils.to_absolute_path(os.fspath(cfg.checkpoint_path))
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("cfg")
    if saved_cfg is None:
        raise ValueError("FastSAC checkpoint is missing its saved runtime cfg")
    probe_cfg = copy.deepcopy(OmegaConf.to_container(cfg.gradient_probe))
    requested_envs = int(cfg.task.num_envs)
    requested_headless = bool(cfg.headless)

    runtime = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
    OmegaConf.set_struct(runtime, False)
    runtime.checkpoint_path = checkpoint_path
    runtime.task.num_envs = requested_envs
    runtime.headless = requested_headless
    runtime.eval_render = False
    runtime.vecnorm = "eval"
    runtime.app.headless = requested_headless
    runtime.app.enable_cameras = True
    runtime.gradient_probe = probe_cfg
    OmegaConf.resolve(runtime)
    return runtime


def _reset_collection_windows(policy) -> None:
    policy._perception_replay_history = None
    policy._perception_replay_history_count = 0
    policy._rollout_final_batch = None
    policy._truncation_final_batches = []
    policy._last_truncation_finals_used = 0


@contextmanager
def _forced_beta_source(policy, teacher_probability: float, seed: int):
    fields = (
        "dagger_control_mode",
        "dagger_beta_start",
        "dagger_beta_end",
        "dagger_beta_decay_rollouts",
    )
    cfg_before = {name: getattr(policy.cfg, name) for name in fields}
    rollout_count_before = int(policy.dagger_rollout_count)
    prefill_before = bool(policy._teacher_prefill_complete)
    dagger_rng_before = policy.dagger_rng.get_state().clone()
    rollout_rng_before = policy.sac_rollout_rng.get_state().clone()
    try:
        policy.cfg.dagger_control_mode = "beta"
        policy.cfg.dagger_beta_start = float(teacher_probability)
        policy.cfg.dagger_beta_end = float(teacher_probability)
        policy.cfg.dagger_beta_decay_rollouts = 1
        policy.dagger_rollout_count = 0
        policy._teacher_prefill_complete = True
        policy.dagger_rng.manual_seed(int(seed))
        policy.sac_rollout_rng.manual_seed(int(seed) + 1)
        yield
    finally:
        for name, value in cfg_before.items():
            setattr(policy.cfg, name, value)
        policy.dagger_rollout_count = rollout_count_before
        policy._teacher_prefill_complete = prefill_before
        policy.dagger_rng.set_state(dagger_rng_before)
        policy.sac_rollout_rng.set_state(rollout_rng_before)


def _trim_next(td: TensorDict) -> TensorDict:
    td["next"] = td["next"].select(
        "done",
        "terminated",
        "discount",
        "reward",
        "stats",
        "is_init",
        "adapt_hx",
        strict=False,
    )
    return td


@torch.inference_mode()
def _collect_one_rollout(env, policy, *, teacher_probability: float, seed: int):
    _reset_collection_windows(policy)
    env.set_seed(int(seed))
    carry = env.reset()
    rollout_policy = policy.get_rollout_policy("train")
    rollout_policy.eval()

    with _forced_beta_source(policy, teacher_probability, seed), set_exploration_type(
        ExplorationType.RANDOM
    ):
        torch.compiler.cudagraph_mark_step_begin()
        warm = rollout_policy(carry.clone(False))
        warm_td, carry = env.step_and_maybe_reset(warm.clone(False))
        warm_td = _trim_next(warm_td)

        env_count = int(env.num_envs)
        step_count = int(policy.cfg.train_every)
        policy.begin_transition_collection()
        data = TensorDict(
            {}, batch_size=(env_count, step_count), device=env.device
        )
        for key, value in warm_td.items(include_nested=True, leaves_only=True):
            data.set(
                key,
                torch.empty(
                    (env_count, step_count, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                ),
            )

        for step in range(step_count):
            carry = rollout_policy(carry)
            actuator_context = (
                policy.capture_q_actuator_context()
                if hasattr(policy, "capture_q_actuator_context")
                else None
            )
            if hasattr(policy, "record_rollout_q_actuator_context"):
                policy.record_rollout_q_actuator_context(actuator_context)
            td, carry = env.step_and_maybe_reset(carry)
            policy.capture_truncation_final_observations(td, step)
            data[:, step] = _trim_next(td)
        policy.capture_rollout_final_observation(carry)

        chunks = tuple(policy._dagger_transition_chunks(data.exclude("stats")))
        if not chunks:
            raise RuntimeError(
                "No diagnostic replay rows survived burn-in/min-step filtering"
            )
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }


def _actor_probe_fields(policy) -> tuple[str, ...]:
    """Mirror the production Actor sampler's source-specific field set.

    ``online_student_rollout`` carries collection-time Actor observations and
    never materializes the raw recurrent perception windows, so the probe must
    request the same fields ``_sample_dagger_actor_batch`` does or the batch
    cannot be prepared.
    """
    return (
        "critic_observations",
        DAGGER_REPLAY_TEACHER_ACTIONS,
        DAGGER_TEACHER_ACTION_VALID_KEY,
        *(
            (Q_ACTUATOR_CONTEXT_KEY,)
            if policy._q_conditions_on_actuator_state()
            else ()
        ),
        *(
            (REPLAY_ACTOR_OBSERVATIONS_KEY,)
            if policy._student_collection_actor_cache_enabled()
            else _PERCEPTION_REPLAY_FIELDS
        ),
    )


def _select_rows(
    transitions: dict[str, torch.Tensor],
    count: int,
    *,
    require_student: bool,
    seed: int,
    actor_fields: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    valid = transitions[DAGGER_TEACHER_ACTION_VALID_KEY].reshape(-1).bool()
    student = transitions[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
    eligible = valid & (student if require_student else ~student)
    indices = eligible.nonzero(as_tuple=False).squeeze(-1)
    if indices.numel() < int(count):
        source = "Student" if require_student else "Teacher"
        raise RuntimeError(
            f"Fresh {source} collection yielded {indices.numel()} valid rows, "
            f"fewer than requested {int(count)}"
        )
    generator = torch.Generator(device=indices.device).manual_seed(int(seed))
    selected = indices.index_select(
        0,
        torch.randperm(indices.numel(), device=indices.device, generator=generator)[
            : int(count)
        ],
    )
    return {
        key: transitions[key].index_select(0, selected)
        for key in actor_fields
    }


def _merge_source_batch(
    student: dict[str, torch.Tensor] | None,
    teacher: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    sources = tuple(value for value in (teacher, student) if value is not None)
    if not sources:
        raise ValueError("At least one diagnostic source is required")
    batch = {
        key: torch.cat([source[key] for source in sources], dim=0)
        for key in sources[0]
    }
    teacher_rows = 0 if teacher is None else int(next(iter(teacher.values())).shape[0])
    student_rows = 0 if student is None else int(next(iter(student.values())).shape[0])
    device = next(iter(batch.values())).device
    batch[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.cat(
        (
            torch.ones(teacher_rows, dtype=torch.bool, device=device),
            torch.zeros(student_rows, dtype=torch.bool, device=device),
        )
    )
    # Do not synthesize a false all-zero failure-phase label.  Field absence is
    # how the diagnostic distinguishes unavailable historical sampler labels
    # from an observed batch that genuinely contains no focused Teacher rows.
    return batch


def _collect_probe_batch(env, policy, cfg: DictConfig):
    mode = str(cfg.source_mode)
    if mode not in ("balanced", "student", "teacher"):
        raise ValueError("gradient_probe.source_mode must be balanced/student/teacher")
    batch_size = int(cfg.batch_size)
    if batch_size < 2:
        raise ValueError("gradient_probe.batch_size must be at least two")
    if mode == "balanced" and batch_size % 2:
        raise ValueError("balanced gradient probe batch_size must be even")
    rollouts = int(cfg.rollouts_per_source)
    if rollouts < 1:
        raise ValueError("gradient_probe.rollouts_per_source must be positive")

    seed = int(cfg.collection_seed)
    collected = {}
    for name, teacher_probability in (
        ("student", 0.0),
        ("teacher", 1.0),
    ):
        if mode not in ("balanced", name):
            continue
        chunks = [
            _collect_one_rollout(
                env,
                policy,
                teacher_probability=teacher_probability,
                seed=seed + source_offset + rollout,
            )
            for rollout in range(rollouts)
            for source_offset in ((0 if name == "student" else 100_000),)
        ]
        collected[name] = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }

    source_count = batch_size // 2 if mode == "balanced" else batch_size
    actor_fields = _actor_probe_fields(policy)
    student = (
        _select_rows(
            collected["student"],
            source_count,
            require_student=True,
            seed=seed + 200_001,
            actor_fields=actor_fields,
        )
        if "student" in collected
        else None
    )
    teacher = (
        _select_rows(
            collected["teacher"],
            source_count,
            require_student=False,
            seed=seed + 200_002,
            actor_fields=actor_fields,
        )
        if "teacher" in collected
        else None
    )
    raw_batch = _merge_source_batch(student, teacher)
    return policy._prepare_dagger_learning_batch(raw_batch)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="fastsac_gradient_probe",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    cfg = _checkpoint_runtime_config(cfg)

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app
    env = None
    try:
        # configure_replay installs the pre-VecNorm raw aliases needed to
        # reconstruct recurrent Actor inputs; inference_only prevents any
        # optimizer/replay resume attempt.
        env, policy, _ = make_env_policy(
            cfg,
            configure_replay=True,
            inference_only=True,
        )
        prepared = _collect_probe_batch(env, policy, cfg.gradient_probe)
        result = diagnose_fastsac_actor_gradients(
            policy,
            prepared,
            sample_seed=int(cfg.gradient_probe.sample_seed),
            source_gradients=bool(cfg.gradient_probe.source_gradients),
            use_q_filtered_bc=(
                None
                if cfg.gradient_probe.use_q_filtered_bc_override is None
                else bool(cfg.gradient_probe.use_q_filtered_bc_override)
            ),
        )
        result["checkpoint_path"] = os.path.realpath(
            os.path.abspath(os.fspath(cfg.checkpoint_path))
        )
        result["collection"] = {
            "source_mode": str(cfg.gradient_probe.source_mode),
            "rollouts_per_source": int(cfg.gradient_probe.rollouts_per_source),
            "collection_seed": int(cfg.gradient_probe.collection_seed),
            "historical_replay_available": False,
            "failure_phase_labels_available": False,
            "student_source_semantics": (
                "fresh_stochastic_student_executed_states"
            ),
            "teacher_source_semantics": (
                "fresh_forced_teacher_executed_states_not_success_filtered"
            ),
        }
        output_path = os.path.realpath(
            os.path.abspath(os.fspath(cfg.gradient_probe.output_path))
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        print(f"Saved FastSAC gradient probe: {output_path}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
