"""Read-only full-episode FastSAC critic causality diagnostic.

Unlike ``fastsac_gradient_probe.py``, this runner resets once per replay source
and follows the first episode through the complete motion.  It keeps the exact
executed action transition, constructs a phase-balanced held-out batch, and
asks whether the configured critic fits its production Bellman target better
for the correct state/action pair than for matched shuffled decoys.

No optimizer is constructed or stepped, no replay is mutated, and no W&B run
is created.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

try:
    from ._isaaclab_bootstrap import AppLauncher
    from .fastsac_gradient_probe import (
        _forced_beta_source,
        _reset_collection_windows,
        _trim_next,
    )
    from .helpers import make_env_policy
except ImportError:
    from _isaaclab_bootstrap import AppLauncher
    from fastsac_gradient_probe import (
        _forced_beta_source,
        _reset_collection_windows,
        _trim_next,
    )
    from helpers import make_env_policy

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_critic_probe import (
    CRITIC_PROBE_SCHEMA,
    matched_decoy_indices,
    phase_balanced_sample_indices,
    phase_bin_indices,
    summarize_distributional_critic_conditions,
    summarize_scalar_critic_conditions,
)
from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    _migrate_explicit_online_replay_capacities,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    NEXT_REFERENCE_PHASE_KEY,
    NEXT_Q_ACTUATOR_CONTEXT_KEY,
    Q_ACTUATOR_CONTEXT_KEY,
    REFERENCE_PHASE_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    REPLAY_COMMAND_FINISHED_KEY,
    REPLAY_MOTION_ID_KEY,
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    REPLAY_TERMINATED_KEY,
    REPLAY_TIME_LIMIT_KEY,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    REPLAY_TEACHER_V_CURRENT_KEY,
    REPLAY_TEACHER_V_NEXT_KEY,
    compute_continuation_coefficient,
)


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfg")
_ENV_INDEX_KEY = "_teacher_prefill_env_index"
_STEP_INDEX_KEY = "_teacher_prefill_step_index"


def _critic_checkpoint_runtime_config(cfg: DictConfig) -> DictConfig:
    """Load the checkpoint's exact task/algo config plus critic-probe controls."""
    checkpoint_path = os.path.realpath(
        hydra.utils.to_absolute_path(os.fspath(cfg.checkpoint_path))
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("cfg")
    if saved_cfg is None:
        raise ValueError("FastSAC checkpoint is missing its saved runtime cfg")
    probe_cfg = copy.deepcopy(OmegaConf.to_container(cfg.critic_probe))
    requested_envs = int(cfg.task.num_envs)
    requested_headless = bool(cfg.headless)

    runtime = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
    OmegaConf.set_struct(runtime, False)
    if (
        "dagger_env_fraction" in runtime.algo
        and "student_buffer_capacity" not in runtime.algo
    ):
        capacities = _migrate_explicit_online_replay_capacities(
            {
                "dagger_buffer_capacity": runtime.algo.dagger_buffer_capacity,
                "dagger_env_fraction": runtime.algo.dagger_env_fraction,
            }
        )
        runtime.algo.dagger_buffer_capacity = capacities[
            "dagger_buffer_capacity"
        ]
        runtime.algo.student_buffer_capacity = capacities[
            "student_buffer_capacity"
        ]
    # ``use_teacher_residual_critic`` was added after the first TVKD FastSAC
    # checkpoints.  The normal resume loader explicitly migrates an absent
    # backend field to ``False``; do the same before model construction here.
    # Otherwise inference-only config completion would inject today's
    # dataclass default (currently ``True``) and silently evaluate an old raw-Q
    # checkpoint with the residual Bellman target.
    policy_state = checkpoint.get("policy", {})
    backend = (
        policy_state.get("dagger_backend_config", {})
        if isinstance(policy_state, dict)
        else {}
    )
    if "use_teacher_residual_critic" not in runtime.algo:
        runtime.algo.use_teacher_residual_critic = bool(
            backend.get("use_teacher_residual_critic", False)
        )
    if "q_twin_reduction" not in runtime.algo:
        q_backend = (
            policy_state.get("q_backend_config", {})
            if isinstance(policy_state, dict)
            else {}
        )
        saved_reduction = (
            policy_state.get(
                "q_twin_reduction",
                q_backend.get(
                    "q_twin_reduction",
                    backend.get("q_twin_reduction", "min"),
                ),
            )
            if isinstance(policy_state, dict)
            else "min"
        )
        if saved_reduction not in ("min", "mean"):
            raise ValueError("FastSAC probe checkpoint has invalid Q reduction")
        runtime.algo.q_twin_reduction = saved_reduction
    runtime.checkpoint_path = checkpoint_path
    runtime.task.num_envs = requested_envs
    runtime.headless = requested_headless
    runtime.eval_render = False
    runtime.vecnorm = "eval"
    runtime.app.headless = requested_headless
    runtime.app.enable_cameras = True
    runtime.critic_probe = probe_cfg
    OmegaConf.resolve(runtime)
    return runtime


def _expected_twin_q(
    policy,
    observations,
    actions,
    actuator_context: torch.Tensor | None,
):
    outputs = policy._q_forward(
        policy.qnet,
        observations,
        actions,
        actuator_context,
    )
    expected = policy._q_output_values(policy.qnet, outputs)
    return outputs, expected, policy._reduce_twin_q_values(expected)


def _required_transition_fields(policy) -> tuple[str, ...]:
    fields = (
        "critic_observations",
        "next_critic_observations",
        "actions",
        "rewards",
        "dones",
        "truncations",
        "discounts",
        REFERENCE_PHASE_KEY,
        NEXT_REFERENCE_PHASE_KEY,
        REPLAY_TERMINATED_KEY,
        REPLAY_COMMAND_FINISHED_KEY,
        REPLAY_TIME_LIMIT_KEY,
        REPLAY_MOTION_ID_KEY,
        DAGGER_REPLAY_TEACHER_ACTIONS,
        DAGGER_TEACHER_ACTION_VALID_KEY,
        DAGGER_IS_STUDENT_ACTION_KEY,
        REPLAY_ACTOR_OBSERVATIONS_KEY,
        REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    )
    if policy._q_conditions_on_actuator_state():
        fields = (
            *fields,
            Q_ACTUATOR_CONTEXT_KEY,
            NEXT_Q_ACTUATOR_CONTEXT_KEY,
        )
    return fields


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        maximum = (
            float((actual.float() - expected.float()).abs().max().item())
            if actual.shape == expected.shape and actual.numel()
            else None
        )
        raise RuntimeError(
            f"critic probe {name} alignment failed: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}, max_abs_error={maximum}"
        )


@torch.inference_mode()
def _collect_contiguous_first_episodes(
    env,
    policy,
    *,
    teacher_probability: float,
    seed: int,
    max_chunks: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Collect complete first episodes without resetting every 32 steps."""
    _reset_collection_windows(policy)
    env.set_seed(int(seed))
    carry = env.reset()
    rollout_policy = policy.get_rollout_policy("train")
    rollout_policy.eval()
    env_count = int(env.num_envs)
    step_count = int(policy.cfg.train_every)
    context_enabled = policy._q_conditions_on_actuator_state()
    required_fields = _required_transition_fields(policy)
    alive = torch.ones(env_count, dtype=torch.bool, device=env.device)
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in required_fields}
    source_is_teacher = bool(float(teacher_probability) == 1.0)
    total_eligible = 0
    total_active = 0
    chunks_collected = 0

    with _forced_beta_source(policy, teacher_probability, seed), set_exploration_type(
        ExplorationType.RANDOM
    ):
        # Preserve the existing probe's one-step CUDA/rollout warm-up, but do
        # not repeatedly reset to this early-motion region.
        torch.compiler.cudagraph_mark_step_begin()
        warm = rollout_policy(carry.clone(False))
        warm_td, carry = env.step_and_maybe_reset(warm.clone(False))
        alive &= ~warm_td["next", "done"].reshape(-1).bool()

        for chunk_index in range(int(max_chunks)):
            policy.begin_transition_collection()
            active_grid = torch.zeros(
                (env_count, step_count), dtype=torch.bool, device=env.device
            )
            sent_actions = torch.empty(
                (env_count, step_count, policy.action_dim),
                dtype=warm_td[ACTION_KEY].dtype,
                device=env.device,
            )
            sent_critic = torch.empty(
                (env_count, step_count, policy._q_critic_dim),
                dtype=warm_td[policy.q_critic_keys[0]].dtype,
                device=env.device,
            )
            sent_context = (
                torch.empty(
                    (env_count, step_count, policy._q_actuator_context_dim),
                    dtype=torch.float32,
                    device=env.device,
                )
                if context_enabled
                else None
            )

            warm_trimmed = _trim_next(warm_td)
            data = TensorDict(
                {}, batch_size=(env_count, step_count), device=env.device
            )
            for key, value in warm_trimmed.items(
                include_nested=True, leaves_only=True
            ):
                data.set(
                    key,
                    torch.empty(
                        (env_count, step_count, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    ),
                )

            for local_step in range(step_count):
                active_grid[:, local_step] = alive
                carry = rollout_policy(carry)
                action_before_step = carry[ACTION_KEY].reshape(
                    env_count, policy.action_dim
                ).detach()
                critic_before_step = policy._cat_replay_sources(
                    carry, policy.q_critic_keys
                ).reshape(env_count, policy._q_critic_dim).detach()
                sent_actions[:, local_step].copy_(action_before_step)
                sent_critic[:, local_step].copy_(critic_before_step)

                actuator_context = (
                    policy.capture_q_actuator_context()
                    if hasattr(policy, "capture_q_actuator_context")
                    else None
                )
                if sent_context is not None:
                    if actuator_context is None:
                        raise RuntimeError(
                            "enabled critic probe did not capture actuator context"
                        )
                    sent_context[:, local_step].copy_(actuator_context)
                if hasattr(policy, "record_rollout_q_actuator_context"):
                    policy.record_rollout_q_actuator_context(actuator_context)
                transition, carry = env.step_and_maybe_reset(carry)
                _assert_exact(
                    "environment executed action",
                    transition[ACTION_KEY].reshape(env_count, policy.action_dim),
                    action_before_step,
                )
                _assert_exact(
                    "environment current critic state",
                    policy._cat_replay_sources(
                        transition, policy.q_critic_keys
                    ).reshape(env_count, policy._q_critic_dim),
                    critic_before_step,
                )
                policy.capture_truncation_final_observations(
                    transition, local_step
                )
                data[:, local_step] = _trim_next(transition)
                done = transition["next", "done"].reshape(-1).bool()
                alive &= ~done

            final_critic = policy._cat_replay_sources(
                carry, policy.q_critic_keys
            ).reshape(env_count, policy._q_critic_dim).detach()
            policy.capture_rollout_final_observation(carry)
            chunks = tuple(policy._dagger_transition_chunks(data.exclude("stats")))
            for transitions in chunks:
                missing = set(required_fields).difference(transitions)
                if missing:
                    raise KeyError(
                        "critic probe transition is missing fields: "
                        f"{sorted(missing)}"
                    )
                env_indices = transitions[_ENV_INDEX_KEY].long()
                local_steps = transitions[_STEP_INDEX_KEY].long()
                active = active_grid[env_indices, local_steps]
                valid = transitions[DAGGER_TEACHER_ACTION_VALID_KEY].bool()
                executed_by_student = transitions[
                    DAGGER_IS_STUDENT_ACTION_KEY
                ].bool()
                expected_source = (
                    ~executed_by_student
                    if source_is_teacher
                    else executed_by_student
                )
                eligible = active & valid & expected_source
                total_active += int(active.sum().item())
                total_eligible += int(eligible.sum().item())
                if not bool(eligible.any()):
                    continue

                expected_action = sent_actions[env_indices, local_steps]
                expected_current = sent_critic[env_indices, local_steps]
                _assert_exact(
                    "replay executed action",
                    transitions["actions"],
                    expected_action,
                )
                _assert_exact(
                    "replay current critic state",
                    transitions["critic_observations"],
                    expected_current,
                )
                if sent_context is not None:
                    expected_context = sent_context[env_indices, local_steps]
                    _assert_exact(
                        "replay current actuator context",
                        transitions[Q_ACTUATOR_CONTEXT_KEY],
                        expected_context,
                    )
                    _assert_exact(
                        "replay next actuator context",
                        transitions[NEXT_Q_ACTUATOR_CONTEXT_KEY],
                        expected_context,
                    )

                # Ordinary successors must be the next collected state.  True
                # boundaries are checked by the production timeout/terminal
                # tests and are intentionally excluded from this equality.
                ordinary = ~transitions["dones"].bool()
                if bool(ordinary.any()):
                    following_steps = local_steps + 1
                    expected_next = torch.empty_like(
                        transitions["next_critic_observations"]
                    )
                    inside = following_steps < step_count
                    if bool(inside.any()):
                        expected_next[inside] = sent_critic[
                            env_indices[inside], following_steps[inside]
                        ]
                    if bool((~inside).any()):
                        expected_next[~inside] = final_critic[
                            env_indices[~inside]
                        ]
                    _assert_exact(
                        "replay ordinary next critic state",
                        transitions["next_critic_observations"][ordinary],
                        expected_next[ordinary],
                    )

                for key in pieces:
                    pieces[key].append(transitions[key][eligible].detach().cpu())
            chunks_collected += 1
            warm_td = data[:, -1]
            if not bool(alive.any()):
                break

    if not pieces["actions"]:
        raise RuntimeError("critic probe collected no eligible replay rows")
    merged = {key: torch.cat(values, dim=0) for key, values in pieces.items()}
    row_count = int(merged["actions"].shape[0])
    merged[DAGGER_Q_TEACHER_SOURCE_KEY] = torch.full(
        (row_count,), source_is_teacher, dtype=torch.bool
    )
    return merged, {
        "source": "teacher" if source_is_teacher else "student",
        "chunks_collected": chunks_collected,
        "simulator_steps_after_warmup": chunks_collected * step_count,
        "first_episode_completed_fraction": float((~alive).float().mean().item()),
        "active_replay_rows": total_active,
        "eligible_source_rows": total_eligible,
        "retained_rows": row_count,
        "alignment_action_max_abs_error": 0.0,
        "alignment_current_state_max_abs_error": 0.0,
        "alignment_ordinary_next_state_max_abs_error": 0.0,
        "alignment_actuator_context_max_abs_error": (
            0.0 if context_enabled else None
        ),
    }


def _merge_sources(*sources: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = set(sources[0])
    if any(set(source) != keys for source in sources[1:]):
        raise ValueError("critic probe source schemas do not match")
    return {key: torch.cat([source[key] for source in sources]) for key in keys}


def _index_batch(
    batch: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, indices).to(device)
        for key, value in batch.items()
    }


def _install_live_teacher_value_cache(policy, batch) -> torch.Tensor:
    continuation = compute_continuation_coefficient(
        dones=batch["dones"],
        truncations=batch["truncations"],
        discounts=batch["discounts"],
    )
    with torch.no_grad():
        current = policy.get_frozen_teacher_value(
            batch["critic_observations"]
        ).float().reshape(-1)
        following = policy.get_frozen_teacher_value(
            batch["next_critic_observations"]
        ).float().reshape(-1)
        following = torch.where(
            continuation != 0.0, following, torch.zeros_like(following)
        )
    batch[REPLAY_TEACHER_V_CURRENT_KEY] = current
    batch[REPLAY_TEACHER_V_NEXT_KEY] = following
    return continuation


def _stratum_masks(source, phase_bins, num_phase_bins):
    masks = {
        "all": torch.ones_like(source, dtype=torch.bool),
        "student": ~source,
        "teacher": source,
    }
    for phase_bin in range(int(num_phase_bins)):
        masks[f"phase_{phase_bin}"] = phase_bins == phase_bin
        masks[f"student_phase_{phase_bin}"] = (~source) & (
            phase_bins == phase_bin
        )
        masks[f"teacher_phase_{phase_bin}"] = source & (
            phase_bins == phase_bin
        )
    return masks


def _ranking_report(policy, batch, masks):
    actuator_context = batch.get(Q_ACTUATOR_CONTEXT_KEY)
    with torch.no_grad():
        raw_mean = policy._actor_mean_from_flat(batch["observations"])
        policy_action = policy._sac_dist_from_mean(raw_mean).mean
        teacher_action = batch[DAGGER_REPLAY_TEACHER_ACTIONS]
        _, _, policy_q = _expected_twin_q(
            policy,
            batch["critic_observations"],
            policy_action,
            actuator_context,
        )
        _, _, teacher_q = _expected_twin_q(
            policy,
            batch["critic_observations"],
            teacher_action,
            actuator_context,
        )
    difference = policy_q - teacher_q
    action_scale = policy._fastsac_q_action_scale.to(policy_action)
    report = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            continue
        report[name] = {
            "rows": int(mask.sum().item()),
            "q_policy_mean": float(policy_q[mask].mean().item()),
            "q_teacher_mean": float(teacher_q[mask].mean().item()),
            "q_policy_minus_teacher_mean": float(difference[mask].mean().item()),
            "policy_q_above_teacher_fraction": float(
                (difference[mask] > 0.0).float().mean().item()
            ),
            "policy_teacher_action_deviation_rms": float(
                (
                    (
                        (policy_action[mask] - teacher_action[mask])
                        / action_scale
                    )
                    .square()
                    .mean()
                    .sqrt()
                ).item()
            ),
        }
    return report, policy_action.detach()


def _local_action_gradient_report(policy, batch, policy_action, epsilon):
    actuator_context = batch.get(Q_ACTUATOR_CONTEXT_KEY)
    # Keep the derivative in the historical normalized issued-command
    # coordinates.  The context is a fixed replay condition, not a variable
    # that the Actor can choose, so only the physical-action-width prefix is a
    # gradient leaf.
    q_action = policy._q_action_input(policy_action).detach().requires_grad_(True)
    if policy._uses_standard_split_stem_q():
        q_state = policy._standard_scalar_q_state_input(
            batch["critic_observations"], policy_action, actuator_context
        ).detach()

        def values_from_q_action(candidate):
            outputs = policy.qnet(q_state, candidate)
            return policy._q_output_values(policy.qnet, outputs)

    else:

        def values_from_q_action(candidate):
            outputs = policy.qnet(
                batch["critic_observations"],
                policy._q_action_features_from_q_input(
                    candidate, actuator_context
                ),
            )
            return policy._q_output_values(policy.qnet, outputs)

    expected = values_from_q_action(q_action)
    reduced = policy._reduce_twin_q_values(expected)
    gradient = torch.autograd.grad(reduced.sum(), q_action)[0].detach().float()
    norm = gradient.norm(dim=-1)
    direction = gradient / norm.clamp_min(torch.finfo(gradient.dtype).eps).unsqueeze(-1)
    with torch.no_grad():
        plus_expected = values_from_q_action(
            q_action.detach() + float(epsilon) * direction
        )
        minus_expected = values_from_q_action(
            q_action.detach() - float(epsilon) * direction
        )
        plus = policy._reduce_twin_q_values(plus_expected)
        minus = policy._reduce_twin_q_values(minus_expected)
        finite_difference = (plus - minus) / (2.0 * float(epsilon))
        relative_error = (finite_difference - norm).abs() / (
            finite_difference.abs() + norm.abs()
        ).clamp_min(1.0e-6)
        teacher_direction = policy._q_action_input(
            batch[DAGGER_REPLAY_TEACHER_ACTIONS]
        ) - q_action.detach()
        teacher_direction_norm = teacher_direction.norm(dim=-1)
        cosine = (gradient * teacher_direction).sum(dim=-1) / (
            norm * teacher_direction_norm
        ).clamp_min(1.0e-6)
    return {
        "epsilon_q_coordinates": float(epsilon),
        "dqda_norm_mean": float(norm.mean().item()),
        "dqda_norm_median": float(norm.median().item()),
        "gradient_ascent_fd_mean": float(finite_difference.mean().item()),
        "gradient_ascent_fd_positive_fraction": float(
            (finite_difference > 0.0).float().mean().item()
        ),
        "gradient_ascent_fd_relative_error_mean": float(
            relative_error.mean().item()
        ),
        "teacher_direction_gradient_cosine_mean": float(cosine.mean().item()),
    }


@torch.no_grad()
def _residual_identity_report(policy, batch, continuation, rng_state):
    enabled = bool(getattr(policy.cfg, "use_teacher_residual_critic", False))
    result: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        return result
    policy.sac_action_rng.set_state(rng_state)
    next_dist = policy._actor_dist_from_flat(batch["next_observations"])
    next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
        generator=policy.sac_action_rng
    )
    next_log_prob = policy._normalized_action_log_prob(next_raw_log_prob)
    policy.sac_action_rng.set_state(rng_state)
    next_outputs = policy._q_forward(
        policy.qnet_target,
        batch["next_critic_observations"],
        next_action,
        batch.get(NEXT_Q_ACTUATOR_CONTEXT_KEY),
    )
    next_expected = policy._q_output_values(policy.qnet_target, next_outputs)
    next_q = policy._reduce_twin_q_values(next_expected)
    terms = policy._teacher_value_terms_from_batch(batch, continuation)
    gamma_continuation = float(policy.cfg.gamma) * continuation
    entropy_adjusted_next = next_q - policy.log_alpha.exp() * next_log_prob
    residual_target = (
        batch["rewards"].float()
        + terms.potential_delta
        + gamma_continuation * entropy_adjusted_next
    )
    lam = float(policy.cfg.tvkd_lambda)
    current_baseline = (1.0 - lam) * terms.teacher_potential
    next_full_q = (1.0 - lam) * terms.teacher_potential_next + next_q
    full_target = terms.shaped_reward + gamma_continuation * (
        next_full_q - policy.log_alpha.exp() * next_log_prob
    )
    identity_error = full_target - (current_baseline + residual_target)
    return {
        "enabled": True,
        "identity": "y_full=(1-lambda)*Phi(s)+y_residual",
        "max_abs_error": float(identity_error.abs().max().item()),
        "mean_abs_error": float(identity_error.abs().mean().item()),
        "actor_gradient_identity": "dQ_full/da=dQ_residual/da",
    }


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="fastsac_critic_probe",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    cfg = _critic_checkpoint_runtime_config(cfg)
    probe_cfg = cfg.critic_probe
    if not bool(getattr(cfg.algo, "use_tvkd_value_shaping", False)):
        raise ValueError("critic causality probe currently requires TVKD shaping")
    if int(probe_cfg.max_chunks) < 1:
        raise ValueError("critic_probe.max_chunks must be positive")
    if not math.isfinite(float(probe_cfg.finite_difference_epsilon)) or float(
        probe_cfg.finite_difference_epsilon
    ) <= 0.0:
        raise ValueError("finite_difference_epsilon must be finite and positive")

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app
    env = None
    try:
        env, policy, _ = make_env_policy(
            cfg, configure_replay=True, inference_only=True
        )
        if not policy._student_collection_actor_cache_enabled():
            raise RuntimeError(
                "critic probe requires online_student_rollout collection Actor cache"
            )
        collection_seed = int(probe_cfg.collection_seed)
        student, student_info = _collect_contiguous_first_episodes(
            env,
            policy,
            teacher_probability=0.0,
            seed=collection_seed,
            max_chunks=int(probe_cfg.max_chunks),
        )
        teacher, teacher_info = _collect_contiguous_first_episodes(
            env,
            policy,
            teacher_probability=1.0,
            seed=collection_seed,
            max_chunks=int(probe_cfg.max_chunks),
        )
        raw = _merge_sources(student, teacher)

        cpu_generator = torch.Generator().manual_seed(int(probe_cfg.sample_seed))
        selected_indices = phase_balanced_sample_indices(
            raw[DAGGER_Q_TEACHER_SOURCE_KEY],
            raw[REFERENCE_PHASE_KEY],
            rows_per_source=int(probe_cfg.rows_per_source),
            num_phase_bins=int(probe_cfg.num_phase_bins),
            generator=cpu_generator,
        )
        selected = _index_batch(raw, selected_indices, policy.device)
        prepared = policy._prepare_dagger_learning_batch(selected)
        expected_normalized = policy._normalize_replay_flat(
            selected["critic_observations"],
            policy.q_critic_keys,
            policy._q_critic_widths,
            policy._vecnorm_snapshot(),
        )
        normalization_error = (
            prepared["critic_observations"] - expected_normalized
        ).abs().max()
        if float(normalization_error.item()) > 1.0e-6:
            raise RuntimeError("critic replay VecNorm alignment failed")
        continuation = _install_live_teacher_value_cache(policy, prepared)

        source = prepared[DAGGER_Q_TEACHER_SOURCE_KEY].reshape(-1).bool()
        phase_bins = phase_bin_indices(
            prepared[REFERENCE_PHASE_KEY], int(probe_cfg.num_phase_bins)
        ).to(policy.device)
        decoy_cpu, decoy_quality = matched_decoy_indices(
            source.cpu(),
            phase_bins.cpu(),
            prepared[REPLAY_MOTION_ID_KEY].cpu(),
            generator=cpu_generator,
        )
        decoy = decoy_cpu.to(policy.device)
        masks = _stratum_masks(source, phase_bins, int(probe_cfg.num_phase_bins))
        actuator_context = prepared.get(Q_ACTUATOR_CONTEXT_KEY)
        with torch.no_grad():
            correct_q_action = policy._q_action_input(prepared["actions"])
            decoy_q_action = correct_q_action.index_select(0, decoy)
            decoy_action_rms = (
                (correct_q_action - decoy_q_action)
                .square()
                .mean(dim=-1)
                .sqrt()
            )
            decoy_state_rms = (
                (
                    prepared["critic_observations"]
                    - prepared["critic_observations"].index_select(0, decoy)
                )
                .square()
                .mean(dim=-1)
                .sqrt()
            )
            decoy_context_rms = (
                None
                if actuator_context is None
                else (
                    actuator_context
                    - actuator_context.index_select(0, decoy)
                )
                .square()
                .mean(dim=-1)
                .sqrt()
            )

        rng_state = policy.sac_action_rng.get_state().clone()
        with torch.no_grad():
            target, target_metrics, _ = policy._distributional_fastsac_target(
                prepared
            )
            policy.sac_action_rng.set_state(rng_state)
            correct_outputs = policy._q_forward(
                policy.qnet,
                prepared["critic_observations"],
                prepared["actions"],
                actuator_context,
            )
            shuffled_action_outputs = policy._q_forward(
                policy.qnet,
                prepared["critic_observations"],
                prepared["actions"].index_select(0, decoy),
                actuator_context,
            )
            shuffled_state_outputs = policy._q_forward(
                policy.qnet,
                prepared["critic_observations"].index_select(0, decoy),
                prepared["actions"],
                (
                    None
                    if actuator_context is None
                    else actuator_context.index_select(0, decoy)
                ),
            )

        if policy._uses_standard_scalar_q():
            fit_report = summarize_scalar_critic_conditions(
                correct_values=correct_outputs,
                shuffled_action_values=shuffled_action_outputs,
                shuffled_state_values=shuffled_state_outputs,
                target=target,
                masks=masks,
            )
            action_fit_metric = "MSE"
        else:
            fit_report = summarize_distributional_critic_conditions(
                correct_logits=correct_outputs,
                shuffled_action_logits=shuffled_action_outputs,
                shuffled_state_logits=shuffled_state_outputs,
                target=target,
                support=policy.qnet.support,
                masks=masks,
            )
            action_fit_metric = "KL"
        ranking_report, policy_action = _ranking_report(policy, prepared, masks)
        gradient_report = _local_action_gradient_report(
            policy,
            prepared,
            policy_action,
            float(probe_cfg.finite_difference_epsilon),
        )
        residual_report = _residual_identity_report(
            policy, prepared, continuation, rng_state
        )
        policy.sac_action_rng.set_state(rng_state)

        result = {
            "schema": CRITIC_PROBE_SCHEMA,
            "checkpoint_path": os.path.realpath(
                os.path.abspath(os.fspath(cfg.checkpoint_path))
            ),
            "read_only": True,
            "optimizer_steps": 0,
            "collection": {
                "reset_semantics": "one_reset_then_contiguous_first_episode",
                "collection_seed": collection_seed,
                "num_envs": int(env.num_envs),
                "train_every": int(policy.cfg.train_every),
                "student": student_info,
                "teacher": teacher_info,
            },
            "heldout_rows": int(prepared["actions"].shape[0]),
            "rows_per_source_requested": int(probe_cfg.rows_per_source),
            "num_phase_bins": int(probe_cfg.num_phase_bins),
            "alignment": {
                "executed_action_chain_exact": True,
                "current_critic_state_chain_exact": True,
                "ordinary_next_critic_state_chain_exact": True,
                "actuator_context_chain_exact": (
                    True if actuator_context is not None else None
                ),
                "actuator_context_dim": (
                    0
                    if actuator_context is None
                    else int(actuator_context.shape[-1])
                ),
                "prepared_vecnorm_max_abs_error": float(
                    normalization_error.item()
                ),
            },
            "decoy_matching": {
                **decoy_quality,
                "q_coordinate_action_rms_mean": float(
                    decoy_action_rms.mean().item()
                ),
                "q_coordinate_action_rms_median": float(
                    decoy_action_rms.median().item()
                ),
                "normalized_critic_state_rms_mean": float(
                    decoy_state_rms.mean().item()
                ),
                "normalized_critic_state_rms_median": float(
                    decoy_state_rms.median().item()
                ),
                "normalized_actuator_context_rms_mean": (
                    None
                    if decoy_context_rms is None
                    else float(decoy_context_rms.mean().item())
                ),
                "normalized_actuator_context_rms_median": (
                    None
                    if decoy_context_rms is None
                    else float(decoy_context_rms.median().item())
                ),
            },
            "critic_target_fit": fit_report,
            "q_critic_type": str(getattr(policy.cfg, "q_critic_type", "c51")),
            "q_twin_reduction": str(
                getattr(policy.cfg, "q_twin_reduction", "min")
            ),
            "teacher_policy_ranking": ranking_report,
            "local_action_gradient": gradient_report,
            "residual_parameterization": residual_report,
            "target_metrics": {
                key: float(torch.as_tensor(value).detach().item())
                for key, value in target_metrics.items()
                if torch.as_tensor(value).numel() == 1
            },
            "interpretation": {
                "action_identifiability": (
                    "shuffled_action_minus_correct "
                    f"{action_fit_metric} should be positive; near zero "
                    "with positive shuffled-state delta indicates a state-dominated "
                    "critic that cannot supply a reliable SAC action signal"
                ),
                "shuffle_contract": (
                    "action shuffle changes only issued action; state shuffle moves "
                    "normalized critic state and actuator context together"
                ),
                "residual_on_off": (
                    "do not toggle use_teacher_residual_critic on this learned "
                    "checkpoint; a causal ON/OFF comparison needs separately trained "
                    "matched critics"
                ),
            },
        }
        output_path = os.path.realpath(
            os.path.abspath(os.fspath(probe_cfg.output_path))
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        print(f"Saved FastSAC critic causality probe: {output_path}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
