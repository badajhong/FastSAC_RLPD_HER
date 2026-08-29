"""Distributional FastSAC with exact mean-action Teacher BC.

This backend deliberately reuses the Teacher-only
prefill, DAgger source selection, timeout handling, and twin-C51 topology from
``td3_bc_dagger``.  The learning rule itself is SAC:

* Student collection and Actor updates use either the historical bounded
  Gaussian in nominal joint coordinates or an opt-in PPOVEL-compatible raw
  physical-action Gaussian;
* Teacher BC is applied only to the distribution's noise-free mean, optionally
  with a detached SPReD-P probability from the online twin Critic;
* the soft Bellman target contains the next-policy entropy term;
* both online critics learn from the complete target-C51 head with the lower
  expected return; and
* there is a target critic but no target Actor or TD3 smoothing noise.

Student replay keeps the exact carried-hidden Actor inputs seen at collection;
successful Teacher episodes are re-encoded with the current EMA perception
modules through a non-serialized sidecar.  Replay observations therefore remain
input-authoritative for Q and Actor updates without a zero-hidden approximation.
Perception itself is trained only from the current live Student rollout through
the exact PPOVEL finetune path; it never samples the replay rings.  The
duplicate Teacher H5 export remains disabled.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY, Actor, hard_copy_
from .fastsac_vel import (
    FastSACTanhNormal,
    _BCDaggerSACAdapter,
    _fastsac_target_entropy,
    _reduce_actor_q_values,
)
from .ppo_bc_dagger import (
    DAGGER_ACTION_DISCREPANCY_MAX_KEY,
    DAGGER_ACTION_DISCREPANCY_RMS_KEY,
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_SAFE_RELEASE_KEY,
    DAGGER_SAFE_TAKEOVER_KEY,
    DAGGER_SAFE_TEACHER_KEY,
    DAGGER_SAFE_UNSAFE_KEY,
    DAGGER_STUDENT_ACTION_VALID_KEY,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
)
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    VEL_CMD_KEY,
    PPOVEL,
)
from .td3_bc_dagger import (
    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS,
    FAILURE_PHASE_STUDENT_SOURCE_KEY,
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE,
    ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS,
    OBJECT_GEO_REPLAY_SEMANTICS,
    PERCEPTION_PREFILL_DISABLED_SEMANTICS,
    NEXT_Q_ACTUATOR_CONTEXT_KEY,
    REPLAY_MOTION_ID_KEY,
    Q_ACTUATOR_CONTEXT_KEY,
    STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY,
    TEACHER_EPISODE_SIDECAR_SEMANTICS,
    TD3_BETA_KEY,
    TD3_COLLECTOR_NOISE_KEY,
    TD3_EXPLORATORY_STUDENT_ACTION_KEY,
    TD3_NOISE_FREE_STUDENT_ACTION_KEY,
    DistributionalTD3TeacherBC,
    DistributionalTD3TeacherBCConfig,
    _DeterministicTD3StudentEvalPolicy,
    _exact_teacher_bc_loss,
    _joint_normalized_action_discrepancy,
    _polyak_update_,
    _project_c51_probabilities,
    _valid_raw_action_rows,
)


TRAINING_ALGORITHM = "distributional_fastsac_teacher_bc_v1"
CHECKPOINT_VERSION = 7
PREVIOUS_CHECKPOINT_VERSION = 6
_PREVIOUS_PRE_PRIOR_CHECKPOINT_VERSION = 5
_SMOOTH_BOUNDED_STD_CHECKPOINT_VERSION = 4
ACTOR_BACKEND = "ppo_vel_smooth_bounded_normalized_std_tanh_fastsac_bc_v2"
PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND = (
    "ppo_vel_raw_physical_joint_std_gaussian_fastsac_bc_v1"
)
NORMALIZED_TANH_ACTION_DISTRIBUTION = "normalized_tanh"
PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION = "ppo_physical_gaussian"
UNIFORM_PHYSICAL_STD_BOUND_MODE = "uniform_physical"
Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE = "q_normalized"
PHYSICAL_STD_BOUND_MODES = frozenset(
    (UNIFORM_PHYSICAL_STD_BOUND_MODE, Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE)
)
FASTSAC_ACTION_PROJECTION_KEY = "fastsac_action_projection"
FASTSAC_PREFILL_TEACHER_NOISE_KEY = "fastsac_prefill_teacher_ppo_noise_q"
FASTSAC_PREFILL_TEACHER_PROJECTION_KEY = (
    "fastsac_prefill_teacher_action_projection"
)
Q_EFFECTIVE_N_STEPS_KEY = "effective_n_steps"
_LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION = 3
_LEGACY_EFFECTIVE_LOG_STD_ACTOR_BACKEND = (
    "ppo_vel_normalized_std_tanh_bounded_fastsac_bc_v1"
)
ACTION_CONTRACT_SEMANTICS = (
    "finite_raw_joint_action_support_with_jointwise_normalized_q_bc_v1"
)
CRITIC_SEMANTICS = (
    "current_stochastic_actor_entropy_soft_target_lower_expected_complete_"
    "c51_distribution_projection_v1"
)
ACTOR_MEAN_OPTIMIZER_SEMANTICS = (
    "adamw_explicit_actor_mean_weight_decay_with_unregularized_variance_v1"
)
SPRED_P_BC_SEMANTICS = (
    "online_twin_sample_std_gaussian_cdf_teacher_superiority_detached_v1"
)
_SPRED_P_SOURCE_METRIC_KEYS = (
    "spred_p_source_metadata_available",
    "spred_p_teacher_source_valid_fraction",
    "spred_p_student_source_valid_fraction",
    "spred_p_teacher_source_bc_weight_mean",
    "spred_p_student_source_bc_weight_mean",
    "spred_p_teacher_source_teacher_advantage_mean",
    "spred_p_student_source_teacher_advantage_mean",
    "spred_p_teacher_source_combined_q_std_mean",
    "spred_p_student_source_combined_q_std_mean",
    "q_filtered_bc_teacher_source_policy_better_fraction",
    "q_filtered_bc_student_source_policy_better_fraction",
)
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_normalized_std_bounded_alpha_logpi_minus_online_twin_min_plus_"
    "joint_normalized_raw_teacher_mean_bc_v1"
)
PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_with_mean_only_joint_normalized_teacher_bc_separate_low_lr_std_"
    "adam_rollout_scale_kl_cap_and_hard_bounds_v3"
)
Q_NORMALIZED_PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_ppo_physical_joint_std_gaussian_alpha_logpi_minus_online_"
    "twin_min_with_mean_only_joint_normalized_teacher_bc_separate_low_lr_std_"
    "adam_rollout_scale_kl_cap_and_jointwise_q_normalized_absolute_intersection_"
    "bounds_v1"
)
ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_delayed_actor_cadence_v3"
)
PPO_PHYSICAL_GAUSSIAN_ENTROPY_SEMANTICS = (
    "bounded_ppo_physical_joint_std_gaussian_nominal_joint_coordinate_log_"
    "probability_auto_or_fixed_temperature_actor_cadence_v3"
)
FASTSAC_TEACHER_PREFILL_SEMANTICS = (
    "ppo_gaussian_noisy_issued_action_clean_mean_bc_label_forced_valid_teacher_"
    "successful_episode_commit_until_replay_capacity_then_main_live_student_"
    "perception_v2"
)
_CRITIC_CADENCE_ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_v2"
)


def _q_target_discount_factors(cfg, batch: Mapping[str, torch.Tensor]):
    """Return endpoint discount, gamma power, and factual replay horizon.

    ``q_n_step=1`` deliberately keeps the historical multiplication order.
    For an opt-in multi-step Student row, replay already owns the discounted
    raw reward and the product of environment discounts; this helper supplies
    only ``gamma**k`` for its endpoint soft bootstrap.
    """
    discounts = batch["discounts"]
    configured = int(getattr(cfg, "q_n_step", 1))
    if configured == 1:
        effective_n_steps = batch.get(
            Q_EFFECTIVE_N_STEPS_KEY, torch.ones_like(discounts)
        )
        return (
            float(cfg.gamma) * discounts,
            float(cfg.gamma),
            effective_n_steps,
        )

    if Q_EFFECTIVE_N_STEPS_KEY not in batch:
        raise KeyError(
            "multi-step Q replay is missing 'effective_n_steps'; rebuild the "
            "fresh replay rings"
        )
    effective_n_steps = batch[Q_EFFECTIVE_N_STEPS_KEY]
    if effective_n_steps.shape != discounts.shape:
        raise ValueError("effective_n_steps and discounts must have identical shape")
    if effective_n_steps.dtype == torch.bool:
        raise ValueError("effective_n_steps cannot be boolean")
    rounded_steps = effective_n_steps.round()
    invalid_steps = (
        ~torch.isfinite(effective_n_steps)
        | (effective_n_steps < 1)
        | (effective_n_steps > configured)
        | (effective_n_steps != rounded_steps)
    )
    if bool(invalid_steps.any()):
        raise ValueError(
            "effective_n_steps must contain finite integer horizons in "
            f"[1, {configured}]"
        )
    teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
    if teacher_source is not None:
        teacher_source = teacher_source.reshape(-1).bool()
        if teacher_source.shape != effective_n_steps.reshape(-1).shape:
            raise ValueError(
                "Q Teacher-source markers and effective_n_steps are misaligned"
            )
        if bool(
            (
                teacher_source
                & (effective_n_steps.reshape(-1) != 1)
            ).any()
        ):
            raise ValueError("Teacher Q replay rows must remain one-step")
    gamma_power = torch.pow(
        torch.full_like(discounts, float(cfg.gamma)),
        effective_n_steps.to(dtype=discounts.dtype),
    )
    return gamma_power * discounts, gamma_power, effective_n_steps


def _fastsac_action_distribution(cfg) -> str:
    """Return the explicit policy-distribution mode, defaulting legacy configs."""
    return str(
        getattr(cfg, "sac_action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION)
    )


def _fastsac_actor_backend(cfg) -> str:
    if _fastsac_action_distribution(cfg) == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION:
        return PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND
    return ACTOR_BACKEND


def _fastsac_actor_weight_decay(cfg) -> float:
    """Return the explicit Actor decay, defaulting legacy config objects to zero."""
    return float(getattr(cfg, "sac_actor_weight_decay", 0.0))


def _spred_p_teacher_probability(
    policy_q: torch.Tensor,
    teacher_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached SPReD-P Teacher weights from an online twin-Q ensemble.

    SPReD-P models the two action values as independent Gaussians fitted across
    Critic heads.  The returned probability is

    ``Phi((mean(Q_teacher) - mean(Q_policy)) / combined_std)``.

    The official implementation leaves this computation in the Actor graph.
    This backend deliberately uses the semi-gradient variant requested for
    stability: the probability, advantage, and uncertainty are all detached,
    so the Actor can reduce BC only by changing its action loss rather than by
    differentiating through the gate itself.
    """
    if policy_q.ndim != 2 or teacher_q.ndim != 2:
        raise ValueError("SPReD-P Q tensors must have shape [critic, batch]")
    if policy_q.shape != teacher_q.shape:
        raise ValueError("SPReD-P Student and Teacher Q shapes must match")
    if policy_q.shape[0] != 2:
        raise ValueError("low-memory SPReD-P requires exactly two Critic heads")
    if not policy_q.is_floating_point() or not teacher_q.is_floating_point():
        raise TypeError("SPReD-P Q tensors must be floating point")
    if policy_q.device != teacher_q.device or policy_q.dtype != teacher_q.dtype:
        raise ValueError("SPReD-P Q tensors must share device and dtype")

    policy_q = policy_q.detach()
    teacher_q = teacher_q.detach()
    q_values_are_finite = torch.isfinite(policy_q).all() & torch.isfinite(
        teacher_q
    ).all()
    if not q_values_are_finite:
        raise RuntimeError("SPReD-P online Critic values contain NaN/Inf")

    policy_mean = policy_q.mean(dim=0)
    teacher_mean = teacher_q.mean(dim=0)
    # torch.std's default Bessel correction matches the authors' official
    # ``torch.std(..., dim=0)`` implementation.  With the locked twin Critic,
    # each uncertainty estimate therefore uses the two available heads.
    policy_std = policy_q.std(dim=0)
    teacher_std = teacher_q.std(dim=0)
    variance_floor = torch.finfo(policy_q.dtype).eps
    combined_std = torch.sqrt(
        (policy_std.square() + teacher_std.square()).clamp_min(variance_floor)
    )
    teacher_advantage = teacher_mean - policy_mean
    z_score = teacher_advantage / combined_std
    teacher_probability = 0.5 * (1.0 + torch.erf(z_score / math.sqrt(2.0)))
    teacher_probability = teacher_probability.clamp(0.0, 1.0)
    return teacher_probability, teacher_advantage, combined_std


class FastSACPhysicalNormal:
    """Generator-aware diagonal Normal in PPOVEL physical-action coordinates.

    PPOVEL uses ``Independent(Normal(loc, scale), 1)`` without a tanh
    transform.  This compact wrapper preserves that distribution while also
    accepting FastSAC's private ``torch.Generator`` streams.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = torch.clamp_min(scale, 1.0e-6).expand_as(loc)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    def log_prob_for_action(self, action: torch.Tensor) -> torch.Tensor:
        standardized = (action - self.loc) / self.scale
        per_coordinate = (
            -0.5 * standardized.square()
            - self.scale.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        return per_coordinate.sum(dim=-1)

    def rsample_with_log_prob(
        self, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            self.loc.shape,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
        )
        action = self.loc + self.scale * noise
        return action, self.log_prob_for_action(action)


def _validate_fastsac_entropy_target_controls(
    sac_log_std_min,
    sac_log_std_max,
    sac_target_entropy_ratio,
    *,
    field_prefix: str = "",
) -> float:
    """Validate the normalized target against a nominal log-std envelope.

    The exact tanh-normal entropy also depends on the state-dependent mean and
    tanh Jacobian.  This strict unsquashed-Gaussian bracket is therefore a
    configuration sanity envelope, not an exact reachability guarantee.  It
    prevents autotune from requiring a log-std boundary or relying on mean
    saturation to reach its target.

    Returns the normalized target entropy per action dimension.
    """
    prefix = f"{field_prefix}." if field_prefix else ""
    values = (
        ("sac_log_std_min", sac_log_std_min),
        ("sac_log_std_max", sac_log_std_max),
        ("sac_target_entropy_ratio", sac_target_entropy_ratio),
    )
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{prefix}{name} must be a finite number")

    log_std_min = float(sac_log_std_min)
    log_std_max = float(sac_log_std_max)
    if not (
        math.isfinite(log_std_min)
        and math.isfinite(log_std_max)
        and log_std_min < log_std_max
    ):
        raise ValueError(
            f"{prefix}sac_log_std_min must be finite and below {prefix}sac_log_std_max"
        )

    target_entropy_ratio = float(sac_target_entropy_ratio)
    if not math.isfinite(target_entropy_ratio) or target_entropy_ratio <= 0.0:
        raise ValueError(
            f"{prefix}sac_target_entropy_ratio must be finite and positive"
        )

    gaussian_entropy_offset = 0.5 * math.log(2.0 * math.pi * math.e)
    min_unsquashed_entropy_per_dim = gaussian_entropy_offset + log_std_min
    max_unsquashed_entropy_per_dim = gaussian_entropy_offset + log_std_max
    target_entropy_per_dim = -target_entropy_ratio
    if not (
        min_unsquashed_entropy_per_dim
        < target_entropy_per_dim
        < max_unsquashed_entropy_per_dim
    ):
        raise ValueError(
            "FastSAC entropy target is unreachable within the nominal "
            "unsquashed-Gaussian log-std envelope: require "
            "0.5*log(2*pi*e) + sac_log_std_min "
            "< -sac_target_entropy_ratio "
            "< 0.5*log(2*pi*e) + sac_log_std_max"
        )
    return target_entropy_per_dim


@dataclass
class DistributionalFastSACTeacherBCConfig(DistributionalTD3TeacherBCConfig):
    """Hydra surface for fresh-only FastSAC + Teacher BC."""

    _target_: str = (
        "active_adaptation.learning.ppo.fastsac_bc_dagger."
        "DistributionalFastSACTeacherBC"
    )
    name: str = "fastsac_bc_dagger"

    dagger_control_mode: str = "beta"
    dagger_beta_start: float = 0.0
    dagger_beta_end: float = 0.0

    # During successful-only Teacher prefill, reproduce PPOVEL's diagonal
    # physical-action Gaussian.  Q receives the factual noisy command while
    # Actor BC retains the clean Teacher mean. Main DAgger/eval remain mean-only.
    teacher_prefill_use_ppo_noise: bool = True

    # SAC replaces every TD3 noise mechanism.  These inherited fields remain
    # only because the raw-replay base dataclass owns them.
    eta_td3: float = 0.0
    target_policy_noise_std: float = 0.0
    target_policy_noise_clip: float = 0.0
    collector_exploration_noise_std: float = 0.0
    collector_exploration_noise_clip: float = 0.0
    policy_delay: int = 8
    td3_learning_starts: int = 8192

    # Process larger replay chunks on the target 5090 so the increased Q UTD
    # does not multiply small depth-encoder launches.
    perception_encode_microbatch_size: int = 512
    # Match PPOVEL finetune exactly for perception: every main iteration uses
    # only its live recurrent Student rollout (for the production setup,
    # [512, 32]), with no Teacher/Student perception replay or prefill warm-up.
    teacher_perception_replay_fraction: float = 0.0
    perception_replay_mode: str = ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
    teacher_perception_warmup_steps: int = 0

    eta_sac: float = 1e-4
    lambda_bc: float = 1.0
    # Weight Teacher BC continuously with a detached SPReD-P probability from
    # the online twin Critic.  The legacy option name is retained so existing
    # Hydra commands do not need another migration.
    use_q_filtered_bc: bool = False
    # Give Q the episode-constant delayed-actuator parameters as action-side
    # execution metadata. Actor and Critic observations remain unchanged.
    q_condition_on_actuator_state: bool = False
    # Add the exact queue/lerp counterfactual response of each candidate action.
    q_use_predicted_effect: bool = False
    # Bounded delay/alpha-conditioned residual FiLM on the Q action stem.
    q_use_residual_film: bool = False
    q_residual_film_scale: float = 0.1
    sac_actor_lr: float = 3e-4
    # Actor regularization is an independent learning-rule choice.  In
    # particular, the Critic's q_weight_decay must never leak into the
    # pretrained PPOVEL mean: at an exact BC solution such leakage would move
    # the Actor even though both configured Actor objectives have zero gradient.
    sac_actor_weight_decay: float = 0.0
    # ``normalized_tanh`` preserves the historical FastSAC policy.  The
    # opt-in ``ppo_physical_gaussian`` mode uses PPOVEL's raw physical-action
    # mean and its direct per-joint Actor.actor_std parameter. The ordinary SAC
    # Q+entropy objective chooses the std direction; dedicated controls below
    # only slow and bound its movement.
    sac_action_distribution: str = NORMALIZED_TANH_ACTION_DISTRIBUTION
    # PPOVEL finetune starts every joint at load_noise_scale=0.5 and uses a
    # small Adam LR. A rollout-old scale-only KL cap is a trust guard, never a
    # destination or checkpoint-derived prior.
    sac_physical_std_lr: float = 1e-5
    sac_physical_std_max_kl: float = 0.01
    # ``uniform_physical`` reproduces the historical scalar interval exactly.
    # ``q_normalized`` intersects that absolute interval with per-joint bounds
    # expressed in the Critic's dimensionless nominal-action coordinates.  It
    # prevents one raw std from meaning radically different exploration on
    # small- and large-range joints without replacing SAC's Q+entropy gradient.
    sac_physical_std_bound_mode: str = UNIFORM_PHYSICAL_STD_BOUND_MODE
    sac_physical_std_min: float = 0.05
    sac_physical_std_max: float = 0.5
    sac_physical_std_normalized_min: float = 0.02
    sac_physical_std_normalized_max: float = 0.11
    sac_initial_action_std: float = 0.1
    sac_log_std_min: float = -10.0
    sac_log_std_max: float = -2.0
    sac_alpha_init: float = 1e-5
    sac_alpha_lr: float = 2e-5
    sac_use_autotune: bool = True
    sac_alpha_update_cadence: str = "actor"
    sac_target_entropy_ratio: float = 1.0
    sac_policy_frequency: int = 8
    sac_learning_starts: int = 8192
    sac_tau: float = 0.005
    sac_max_grad_norm: float = 1.0

    q_updates_per_rollout: int = 32
    q_update_to_data_ratio: float | None = 1.0


ConfigStore.instance().store(
    "fastsac_bc_dagger_finetune",
    node=DistributionalFastSACTeacherBCConfig(
        in_keys=(
            CMD_KEY,
            OBS_KEY,
            OBJECT_KEY,
            OBS_PRIV_KEY,
            OBJECT_GEO_KEY,
            VEL_CMD_KEY,
            DEPTH_KEY,
        )
    ),
    group="algo",
)


class _DistributionalFastSACDaggerRolloutPolicy(_DaggerRolloutPolicy):
    """Locked DAgger selection with stochastic Student-only execution."""

    @torch.no_grad()
    def forward(self, td: TensorDict):
        owner = self._owner
        teacher_prefill_active = owner._teacher_prefill_active()
        teacher_prefill_ppo_noise = teacher_prefill_active and bool(
            getattr(owner.cfg, "teacher_prefill_use_ppo_noise", False)
        )
        raw_student_mean = owner._student_raw_action_proposal(td)
        cache_enabled = getattr(
            owner, "_student_collection_actor_cache_enabled", lambda: False
        )
        if cache_enabled():
            td[STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY] = (
                owner._collection_actor_observations(td)
            )
        for scratch_key in (
            "_depth_feature",
            OBJECT_PRED_KEY,
            OBJECT_PRED_TRANS_KEY,
            "_actor_inp",
            "_actor_feature",
            "_object_adapt_mlp_inp",
            "_obj_adapt_mlp",
            "_object_adapt_inp",
            "_adapt_inp",
            "loc",
            "scale",
            "sample_log_prob",
        ):
            if scratch_key in td.keys(True, True):
                td.del_(scratch_key)

        if teacher_prefill_ppo_noise:
            teacher_action, teacher_scale = owner._teacher_action_statistics(
                td.clone(False)
            )
            if teacher_scale.shape != teacher_action.shape:
                raise ValueError(
                    "PPO Teacher mean and scale must have identical shapes"
                )
        else:
            teacher_action = owner._teacher_action(td.clone(False))
            teacher_scale = None
        valid = _valid_raw_action_rows(teacher_action)
        if teacher_scale is not None:
            valid = valid & torch.isfinite(teacher_scale).all(dim=-1) & (
                teacher_scale > 0.0
            ).all(dim=-1)
        finite_teacher_action = torch.nan_to_num(
            teacher_action, nan=0.0, posinf=0.0, neginf=0.0
        )
        bounded_teacher_action = owner._project_execution_action(finite_teacher_action)
        if teacher_scale is not None:
            finite_teacher_scale = torch.nan_to_num(
                teacher_scale, nan=1.0e-6, posinf=1.0e-6, neginf=1.0e-6
            ).clamp_min(1.0e-6)
            raw_prefill_teacher_action, _ = FastSACPhysicalNormal(
                finite_teacher_action, finite_teacher_scale
            ).rsample_with_log_prob(generator=owner.teacher_prefill_action_rng)
            bounded_prefill_teacher_action = owner._project_execution_action(
                raw_prefill_teacher_action
            )
            prefill_teacher_projection = valid & (
                bounded_prefill_teacher_action != raw_prefill_teacher_action
            ).any(dim=-1)
        else:
            bounded_prefill_teacher_action = bounded_teacher_action
            prefill_teacher_projection = torch.zeros_like(valid)
        student_valid = torch.isfinite(raw_student_mean).all(dim=-1)
        finite_student_mean = torch.nan_to_num(
            raw_student_mean, nan=0.0, posinf=0.0, neginf=0.0
        )
        student_dist = owner._sac_dist_from_mean(finite_student_mean)
        mean_student_action = student_dist.mean

        # Safety and beta compare the noise-free BC quantity.  Exploration
        # itself must not trigger Teacher takeovers or bias source selection.
        discrepancy_rms, discrepancy_max = _joint_normalized_action_discrepancy(
            mean_student_action,
            bounded_teacher_action,
            owner._fastsac_q_action_scale,
        )
        discrepancy_rms = torch.where(
            valid, discrepancy_rms, torch.zeros_like(discrepancy_rms)
        )
        discrepancy_max = torch.where(
            valid, discrepancy_max, torch.zeros_like(discrepancy_max)
        )
        reset = td.get("is_init", None)
        reset = (
            torch.zeros_like(valid)
            if reset is None
            else reset.reshape(valid.shape).bool()
        )

        control_mode = owner._effective_control_mode()
        safe_teacher_mask = torch.zeros_like(valid)
        safe_unsafe = torch.zeros_like(valid)
        safe_takeover = torch.zeros_like(valid)
        safe_release = torch.zeros_like(valid)
        if teacher_prefill_active:
            # Prefill always selects a valid Teacher. Its optional PPOVEL draw
            # owns a separate RNG, so collection duration cannot shift either
            # Student SAC sampling stream.
            choose_teacher = valid
            beta_teacher = torch.zeros_like(valid)
        elif control_mode in ("safe", "hybrid"):
            if owner._safe_teacher_control_enabled():
                (
                    safe_teacher_mask,
                    safe_unsafe,
                    safe_takeover,
                    safe_release,
                ) = self._safe_selection(valid, student_valid, discrepancy_rms, reset)
            else:
                safe_unsafe = valid & (
                    ~student_valid
                    | (discrepancy_rms > float(owner.cfg.dagger_safe_takeover_rms))
                )
                self._safe_teacher_active = torch.zeros_like(valid)
                self._safe_teacher_hold = torch.zeros_like(valid, dtype=torch.long)
        elif control_mode != "beta":
            raise RuntimeError(f"Unsupported dagger_control_mode={control_mode!r}")

        scheduled_beta = owner._teacher_mixture_probability()
        if not teacher_prefill_active:
            beta_teacher = torch.zeros_like(valid)
            if control_mode in ("beta", "hybrid"):
                beta_teacher = (
                    (
                        torch.rand(
                            valid.shape,
                            device=valid.device,
                            generator=owner.dagger_rng,
                        )
                        < scheduled_beta
                    )
                    & valid
                    & ~safe_teacher_mask
                )
            choose_teacher = safe_teacher_mask | beta_teacher
            sampled_student_action, _ = student_dist.rsample_with_log_prob(
                generator=owner.sac_rollout_rng
            )
        else:
            sampled_student_action = mean_student_action

        if (~valid & ~student_valid).any():
            raise RuntimeError(
                "Neither Teacher nor FastSAC Student produced a finite raw action"
            )
        choose_teacher = choose_teacher | (valid & ~student_valid)
        if not torch.isfinite(sampled_student_action).all():
            raise RuntimeError("FastSAC sampled a non-finite raw action")

        projected_student_action = owner._project_execution_action(
            sampled_student_action
        )
        student_projection = (~choose_teacher) & (
            projected_student_action != sampled_student_action
        ).any(dim=-1)
        issued_action = torch.where(
            choose_teacher.unsqueeze(-1),
            bounded_prefill_teacher_action,
            projected_student_action,
        )
        # Both branches have already passed through the authoritative execution
        # projection. Selecting between them cannot leave that finite support,
        # so projecting the combined tensor again only repeats the same kernels.
        sample_q_deviation = owner._q_action_input(sampled_student_action) - (
            owner._q_action_input(mean_student_action)
        )
        prefill_teacher_q_noise = owner._q_action_input(
            bounded_prefill_teacher_action
        ) - owner._q_action_input(bounded_teacher_action)
        prefill_teacher_q_noise = torch.where(
            (choose_teacher & teacher_prefill_ppo_noise).unsqueeze(-1),
            prefill_teacher_q_noise,
            torch.zeros_like(prefill_teacher_q_noise),
        )

        td[ACTION_KEY] = issued_action
        td[DAGGER_TEACHER_ACTION_KEY] = bounded_teacher_action
        td[DAGGER_TEACHER_ACTION_VALID_KEY] = valid
        td[DAGGER_IS_STUDENT_ACTION_KEY] = ~choose_teacher
        td[DAGGER_ACTION_DISCREPANCY_RMS_KEY] = discrepancy_rms
        td[DAGGER_ACTION_DISCREPANCY_MAX_KEY] = discrepancy_max
        td[DAGGER_SAFE_UNSAFE_KEY] = safe_unsafe
        td[DAGGER_SAFE_TEACHER_KEY] = safe_teacher_mask
        td[DAGGER_SAFE_TAKEOVER_KEY] = safe_takeover
        td[DAGGER_SAFE_RELEASE_KEY] = safe_release
        td[DAGGER_BETA_TEACHER_KEY] = beta_teacher
        td[DAGGER_STUDENT_ACTION_VALID_KEY] = student_valid

        # The inherited raw replay schema names these audit fields after TD3,
        # but their FastSAC values are respectively mean, stochastic sample,
        # and sample-minus-mean in the joint-wise Q coordinates.
        td[TD3_NOISE_FREE_STUDENT_ACTION_KEY] = mean_student_action
        td[TD3_EXPLORATORY_STUDENT_ACTION_KEY] = sampled_student_action
        td[TD3_COLLECTOR_NOISE_KEY] = sample_q_deviation
        td[FASTSAC_ACTION_PROJECTION_KEY] = student_projection
        td[FASTSAC_PREFILL_TEACHER_NOISE_KEY] = prefill_teacher_q_noise
        td[FASTSAC_PREFILL_TEACHER_PROJECTION_KEY] = (
            prefill_teacher_projection & choose_teacher
        )
        td[TD3_BETA_KEY] = torch.full_like(discrepancy_rms, float(scheduled_beta))
        motion_ids = getattr(
            getattr(getattr(owner, "env", None), "command_manager", None),
            "motion_ids",
            None,
        )
        if torch.is_tensor(motion_ids):
            motion_ids = motion_ids.detach().to(device=valid.device, dtype=torch.long)
            if motion_ids.numel() != valid.numel():
                raise RuntimeError(
                    "command motion_ids do not match rollout environments"
                )
            td[REPLAY_MOTION_ID_KEY] = motion_ids.reshape(valid.shape)
        return td


class _DeterministicFastSACStudentEvalPolicy(_DeterministicTD3StudentEvalPolicy):
    """Mode-consistent Student mean; never samples or computes log-prob."""


class DistributionalFastSACTeacherBC(DistributionalTD3TeacherBC):
    """Twin-C51 FastSAC with stochastic SAC and exact mean-only Teacher BC."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        # SAC owns no target Actor.  The TD3 base initializes this to None; keep
        # that explicit invariant for checkpoint and runtime inspection.
        self.actor_target = None
        self.actor_backend = _fastsac_actor_backend(cfg)

        self._fastsac_entropy_reference_log_scale_sum = float(
            torch.log(self._fastsac_q_action_scale).sum().item()
        )
        physical_gaussian = self._uses_ppo_physical_gaussian()
        if physical_gaussian:
            # Registered but inert compatibility state for old checkpoint and
            # module-tree loaders. PPOVEL's actor_std owns physical variance.
            initial_log_std = self._fastsac_q_action_scale.new_zeros(
                (self.action_dim,)
            )
            initial_raw_log_std = initial_log_std.clone()
        else:
            initial_log_std = torch.log(
                self._fastsac_q_action_scale.new_full(
                    (self.action_dim,), float(cfg.sac_initial_action_std)
                )
            )
            if (initial_log_std <= float(cfg.sac_log_std_min)).any() or (
                initial_log_std >= float(cfg.sac_log_std_max)
            ).any():
                raise ValueError(
                    "sac_initial_action_std must map strictly inside normalized "
                    "log-std bounds"
                )
            initial_raw_log_std = self._inverse_smooth_log_std(
                initial_log_std,
                float(cfg.sac_log_std_min),
                float(cfg.sac_log_std_max),
            )
        self.register_buffer(
            "_fastsac_initial_log_std", initial_log_std.detach().clone()
        )
        self.register_buffer(
            "_fastsac_initial_raw_log_std", initial_raw_log_std.detach().clone()
        )
        self.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
            self.action_dim, initial_raw_log_std, self.device
        )
        if physical_gaussian:
            # PPOVEL owns the directly parameterized per-joint std.  The
            # normalized-tanh adapter remains registered only so historical
            # source/checkpoint topology stays explicit and loadable.
            self.bc_dagger_sac_adapter.requires_grad_(False)
            self._configure_training_actor_std()
        else:
            self._freeze_legacy_actor_std()

        actor_std_parameter = (
            self._ppo_actor_std_parameter() if physical_gaussian else None
        )
        actor_mean_parameters = tuple(
            parameter
            for parameter in self.actor_adapt.parameters()
            if parameter.requires_grad and parameter is not actor_std_parameter
        )
        if not actor_mean_parameters:
            raise RuntimeError("FastSAC Actor has no trainable mean parameters")
        # ``normalized_tanh`` retains the historical joint mean/std Actor
        # optimizer. In physical Gaussian mode the same SAC objective supplies
        # mean and std gradients, but the direct PPOVEL actor_std is stepped by
        # its own low-LR Adam so mean learning-rate/clipping choices cannot make
        # exploration scale jump.
        std_parameters = (
            (actor_std_parameter,)
            if physical_gaussian
            else tuple(self.bc_dagger_sac_adapter.parameters())
        )
        if not std_parameters or {
            id(parameter) for parameter in actor_mean_parameters
        }.intersection(id(parameter) for parameter in std_parameters):
            raise RuntimeError(
                "FastSAC Actor mean and variance optimizer groups must be "
                "non-empty and disjoint"
            )
        actor_optimizer_groups = [
            {
                "params": actor_mean_parameters,
                "lr": float(cfg.sac_actor_lr),
                "weight_decay": _fastsac_actor_weight_decay(cfg),
            }
        ]
        if not physical_gaussian:
            actor_optimizer_groups.append(
                {
                    "params": std_parameters,
                    "lr": float(cfg.sac_actor_lr),
                    "weight_decay": 0.0,
                }
            )
        self.actor_optimizer = torch.optim.AdamW(
            actor_optimizer_groups,
            lr=float(cfg.sac_actor_lr),
            betas=(0.9, 0.95),
        )
        self.actor_std_optimizer = (
            torch.optim.Adam(
                std_parameters,
                lr=float(cfg.sac_physical_std_lr),
                weight_decay=0.0,
            )
            if physical_gaussian
            else None
        )
        if physical_gaussian:
            self._project_physical_actor_std_()
        self.log_alpha = nn.Parameter(
            torch.tensor(
                math.log(float(cfg.sac_alpha_init)),
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=bool(cfg.sac_use_autotune),
        )
        self.alpha_optimizer = (
            torch.optim.Adam(
                (self.log_alpha,), lr=float(cfg.sac_alpha_lr), betas=(0.9, 0.95)
            )
            if bool(cfg.sac_use_autotune)
            else None
        )
        self.target_entropy = _fastsac_target_entropy(
            self._fastsac_q_action_center - self._fastsac_q_action_scale,
            self._fastsac_q_action_center + self._fastsac_q_action_scale,
            float(cfg.sac_target_entropy_ratio),
        )
        if physical_gaussian:
            # These bounds depend only on the immutable action contract and
            # configured std interval. Materialize their host scalars once;
            # rollout diagnostics and checkpoint metadata reuse the cache.
            self._fastsac_physical_entropy_bounds = (
                self._physical_normalized_entropy_bounds()
            )
            if bool(cfg.sac_use_autotune):
                self._validate_physical_entropy_target_reachable()
        generator_device = torch.device(device)
        self.sac_action_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 1
        )
        self.sac_rollout_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 2
        )
        self.teacher_prefill_action_rng = torch.Generator(
            device=generator_device
        ).manual_seed(int(cfg.q_seed) + 4)
        self.alpha_update_count = 0
        self.actor_std_update_count = 0
        self._last_fastsac_diagnostics: dict[str, float] = {}

    def _q_replay_prefill_storage_fields(self) -> tuple[str, ...]:
        """Keep noisy Q actions and clean Teacher BC labels side by side."""
        fields = super()._q_replay_storage_fields()
        if (
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
            and DAGGER_REPLAY_TEACHER_ACTIONS not in fields
        ):
            fields = (*fields, DAGGER_REPLAY_TEACHER_ACTIONS)
        return fields

    def _apply_actor_optimizer_weight_decay_contract(self) -> None:
        """Install the explicit Actor decay after construction or resume.

        Adam/AdamW optimizer state dictionaries own their param-group options.
        Loading a checkpoint therefore overwrites constructor values.  Older
        FastSAC checkpoints accidentally stored ``q_weight_decay`` on the Actor
        mean group; normalize that legacy metadata to the explicit Actor
        contract while retaining its moments and step counters.
        """
        groups = self.actor_optimizer.param_groups
        if not groups:
            raise RuntimeError("FastSAC Actor optimizer has no parameter groups")
        groups[0]["weight_decay"] = _fastsac_actor_weight_decay(self.cfg)
        # The optional second group is the normalized-tanh variance adapter.
        # Physical Gaussian variance has a separate Adam optimizer.  Neither is
        # part of the Actor-mean regularization contract.
        for group in groups[1:]:
            group["weight_decay"] = 0.0

    def _validate_actor_optimizer_weight_decay_contract(self) -> None:
        groups = self.actor_optimizer.param_groups
        if not groups:
            raise RuntimeError("FastSAC Actor optimizer has no parameter groups")
        expected = (_fastsac_actor_weight_decay(self.cfg),) + (0.0,) * (
            len(groups) - 1
        )
        for index, (group, expected_decay) in enumerate(zip(groups, expected)):
            decay = group.get("weight_decay", 0.0)
            if (
                isinstance(decay, bool)
                or not isinstance(decay, (int, float))
                or not math.isfinite(float(decay))
                or not math.isclose(
                    float(decay), expected_decay, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise RuntimeError(
                    "FastSAC Actor optimizer weight-decay contract mismatch at "
                    f"param group {index}: expected {expected_decay}, got {decay!r}"
                )

    def _student_collection_actor_cache_enabled(self) -> bool:
        """Use live carried-hidden Actor inputs in the locked online mode."""
        return str(
            getattr(self.cfg, "perception_replay_mode", "legacy_online_student")
        ) == ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE

    @staticmethod
    def _validate_td3_config(cfg) -> None:
        # Preserve every locked observation/action/replay/C51 check in the raw
        # replay base, then reject the deterministic algorithm knobs.  Present
        # eta_sac to the base's generic "at least one Actor objective" check so
        # a deliberate pure-SAC (lambda_bc=0) ablation remains expressible even
        # though the inherited eta_td3 field is correctly locked to zero.
        base_cfg = copy.copy(cfg)
        base_cfg.eta_td3 = float(cfg.eta_sac)
        DistributionalTD3TeacherBC._validate_td3_config(base_cfg)
        if not isinstance(cfg.teacher_prefill_use_ppo_noise, bool):
            raise ValueError("teacher_prefill_use_ppo_noise must be a boolean")
        if str(cfg.perception_replay_mode) != (
            ONLINE_STUDENT_ROLLOUT_PERCEPTION_MODE
        ):
            raise ValueError(
                "FastSAC perception is locked to online_student_rollout"
            )
        if float(cfg.teacher_perception_replay_fraction) != 0.0:
            raise ValueError(
                "FastSAC online Student perception requires "
                "teacher_perception_replay_fraction=0"
            )
        if int(cfg.teacher_perception_warmup_steps) != 0:
            raise ValueError(
                "FastSAC online Student perception requires "
                "teacher_perception_warmup_steps=0"
            )
        if (
            str(cfg.dagger_control_mode) != "beta"
            or float(cfg.dagger_beta_start) != 0.0
            or float(cfg.dagger_beta_end) != 0.0
        ):
            raise ValueError(
                "FastSAC live Student perception requires beta control with "
                "dagger_beta_start=dagger_beta_end=0"
            )
        exact_zero = (
            "eta_td3",
            "target_policy_noise_std",
            "target_policy_noise_clip",
            "collector_exploration_noise_std",
            "collector_exploration_noise_clip",
        )
        for name in exact_zero:
            if float(getattr(cfg, name)) != 0.0:
                raise ValueError(f"FastSAC requires inherited TD3 field {name}=0")
        action_distribution = _fastsac_action_distribution(cfg)
        if action_distribution not in (
            NORMALIZED_TANH_ACTION_DISTRIBUTION,
            PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION,
        ):
            raise ValueError(
                "sac_action_distribution must be 'normalized_tanh' or "
                "'ppo_physical_gaussian'"
            )
        if not isinstance(cfg.sac_use_autotune, bool):
            raise ValueError("sac_use_autotune must be a boolean")
        if action_distribution == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION:
            load_noise_scale = getattr(cfg, "load_noise_scale", None)
            if (
                isinstance(load_noise_scale, bool)
                or not isinstance(load_noise_scale, (int, float))
                or not math.isfinite(float(load_noise_scale))
                or float(load_noise_scale) <= 0.0
            ):
                raise ValueError(
                    "ppo_physical_gaussian requires a finite positive "
                    "load_noise_scale, matching PPOVEL finetune initialization"
                )
            physical_std_controls = {
                name: getattr(cfg, name, None)
                for name in (
                    "sac_physical_std_lr",
                    "sac_physical_std_max_kl",
                    "sac_physical_std_min",
                    "sac_physical_std_max",
                )
            }
            for name, value in physical_std_controls.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            std_min = float(physical_std_controls["sac_physical_std_min"])
            std_max = float(physical_std_controls["sac_physical_std_max"])
            if not std_min < std_max:
                raise ValueError(
                    "sac_physical_std_min must be smaller than "
                    "sac_physical_std_max"
                )
            if not std_min <= float(load_noise_scale) <= std_max:
                raise ValueError(
                    "load_noise_scale must lie inside the physical std bounds"
                )
            bound_mode = str(
                getattr(
                    cfg,
                    "sac_physical_std_bound_mode",
                    UNIFORM_PHYSICAL_STD_BOUND_MODE,
                )
            )
            if bound_mode not in PHYSICAL_STD_BOUND_MODES:
                raise ValueError(
                    "sac_physical_std_bound_mode must be 'uniform_physical' "
                    "or 'q_normalized'"
                )
            normalized_std_controls = {
                name: getattr(cfg, name, default)
                for name, default in (
                    ("sac_physical_std_normalized_min", 0.02),
                    ("sac_physical_std_normalized_max", 0.11),
                )
            }
            for name, value in normalized_std_controls.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            if not (
                float(normalized_std_controls["sac_physical_std_normalized_min"])
                < float(normalized_std_controls["sac_physical_std_normalized_max"])
            ):
                raise ValueError(
                    "sac_physical_std_normalized_min must be smaller than "
                    "sac_physical_std_normalized_max"
                )
        elif str(
            getattr(
                cfg,
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        ) != UNIFORM_PHYSICAL_STD_BOUND_MODE:
            raise ValueError(
                "q-normalized physical std bounds require "
                "sac_action_distribution='ppo_physical_gaussian'"
            )
        if str(cfg.sac_alpha_update_cadence) not in ("actor", "critic"):
            raise ValueError("sac_alpha_update_cadence must be 'actor' or 'critic'")
        if action_distribution == NORMALIZED_TANH_ACTION_DISTRIBUTION:
            initial_action_std = float(cfg.sac_initial_action_std)
            if not math.isfinite(initial_action_std) or initial_action_std <= 0.0:
                raise ValueError("sac_initial_action_std must be finite and positive")
            _validate_fastsac_entropy_target_controls(
                cfg.sac_log_std_min,
                cfg.sac_log_std_max,
                cfg.sac_target_entropy_ratio,
            )
        else:
            target_ratio = cfg.sac_target_entropy_ratio
            if (
                isinstance(target_ratio, bool)
                or not isinstance(target_ratio, (int, float))
                or not math.isfinite(float(target_ratio))
                or float(target_ratio) <= 0.0
            ):
                raise ValueError(
                    "sac_target_entropy_ratio must be finite and positive"
                )
        for name in (
            "sac_actor_lr",
            "sac_alpha_init",
            "sac_alpha_lr",
            "sac_tau",
            "sac_max_grad_norm",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("eta_sac", "lambda_bc"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if getattr(cfg, "use_q_filtered_bc", False) not in (True, False):
            raise ValueError("use_q_filtered_bc must be boolean")
        if not isinstance(
            getattr(cfg, "q_condition_on_actuator_state", False), bool
        ):
            raise ValueError("q_condition_on_actuator_state must be boolean")
        if not isinstance(getattr(cfg, "q_use_predicted_effect", False), bool):
            raise ValueError("q_use_predicted_effect must be boolean")
        if bool(getattr(cfg, "q_use_predicted_effect", False)) and not bool(
            getattr(cfg, "q_condition_on_actuator_state", False)
        ):
            raise ValueError(
                "q_use_predicted_effect requires q_condition_on_actuator_state=true"
            )
        if not isinstance(getattr(cfg, "q_use_residual_film", False), bool):
            raise ValueError("q_use_residual_film must be boolean")
        if bool(getattr(cfg, "q_use_residual_film", False)):
            if not bool(getattr(cfg, "q_condition_on_actuator_state", False)):
                raise ValueError(
                    "q_use_residual_film requires "
                    "q_condition_on_actuator_state=true"
                )
            if str(cfg.q_action_fusion) not in ("late", "balanced"):
                raise ValueError(
                    "q_use_residual_film requires a late or balanced action stem"
                )
        residual_film_scale = getattr(cfg, "q_residual_film_scale", 0.1)
        if (
            isinstance(residual_film_scale, bool)
            or not isinstance(residual_film_scale, (int, float))
            or not math.isfinite(float(residual_film_scale))
            or not 0.0 < float(residual_film_scale) <= 1.0
        ):
            raise ValueError("q_residual_film_scale must be finite and in (0, 1]")
        actor_weight_decay = getattr(cfg, "sac_actor_weight_decay", 0.0)
        if (
            isinstance(actor_weight_decay, bool)
            or not isinstance(actor_weight_decay, (int, float))
            or not math.isfinite(float(actor_weight_decay))
            or float(actor_weight_decay) < 0.0
        ):
            raise ValueError(
                "sac_actor_weight_decay must be finite and non-negative"
            )
        if float(cfg.eta_sac) == 0.0 and float(cfg.lambda_bc) == 0.0:
            raise ValueError("eta_sac and lambda_bc cannot both be zero")
        for name in ("sac_policy_frequency", "sac_learning_starts"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < float(cfg.sac_tau) <= 1.0:
            raise ValueError("sac_tau must be in (0,1]")
        aliases = (
            ("dagger_bc_lr", "sac_actor_lr"),
            ("policy_delay", "sac_policy_frequency"),
            ("td3_learning_starts", "sac_learning_starts"),
            ("q_tau", "sac_tau"),
            ("q_max_grad_norm", "sac_max_grad_norm"),
        )
        for legacy, sac_name in aliases:
            left = float(getattr(cfg, legacy))
            right = float(getattr(cfg, sac_name))
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{legacy} must equal {sac_name} in this backend")
        q_update_to_data_ratio = getattr(cfg, "q_update_to_data_ratio", None)
        if (
            q_update_to_data_ratio is None
            or isinstance(q_update_to_data_ratio, bool)
            or not math.isclose(
                float(q_update_to_data_ratio),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "FastSAC requires q_update_to_data_ratio=1 for row-level Q UTD=1"
            )

    def _uses_ppo_physical_gaussian(self) -> bool:
        return (
            _fastsac_action_distribution(self.cfg)
            == PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION
        )

    def _actor_learning_semantics(self) -> str:
        if not self._uses_ppo_physical_gaussian():
            semantics = ACTOR_LEARNING_SEMANTICS
        elif (
            self._physical_std_bound_mode()
            == Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE
        ):
            semantics = Q_NORMALIZED_PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS
        else:
            semantics = PPO_PHYSICAL_GAUSSIAN_ACTOR_LEARNING_SEMANTICS
        if getattr(self.cfg, "use_q_filtered_bc", False):
            return f"{semantics}_with_{SPRED_P_BC_SEMANTICS}"
        return semantics

    def _entropy_semantics(self) -> str:
        if self._uses_ppo_physical_gaussian():
            return PPO_PHYSICAL_GAUSSIAN_ENTROPY_SEMANTICS
        return (
            ENTROPY_SEMANTICS
            if str(self.cfg.sac_alpha_update_cadence) == "actor"
            else _CRITIC_CADENCE_ENTROPY_SEMANTICS
        )

    def _ppo_actor_core(self) -> Actor:
        cores = [
            module for module in self.actor_adapt.modules() if isinstance(module, Actor)
        ]
        if len(cores) != 1:
            raise RuntimeError(
                "FastSAC Teacher-BC requires exactly one legacy Actor core; "
                f"found {len(cores)}"
            )
        core = cores[0]
        if bool(core.predict_std):
            raise RuntimeError("FastSAC mean transfer requires a separate legacy std")
        return core

    def _ppo_actor_std_parameter(self) -> nn.Parameter:
        return self._ppo_actor_core().actor_std

    def _freeze_legacy_actor_std(self) -> None:
        core = self._ppo_actor_core()
        core.actor_std.requires_grad_(False)
        core.actor_std.grad = None

    def _configure_training_actor_std(self) -> None:
        """Select PPOVEL std ownership for the configured policy distribution."""
        if self._uses_ppo_physical_gaussian():
            core = self._ppo_actor_core()
            core.actor_std.requires_grad_(True)
            core.actor_std.grad = None
            self.bc_dagger_sac_adapter.requires_grad_(False)
            for parameter in self.bc_dagger_sac_adapter.parameters():
                parameter.grad = None
        else:
            self._freeze_legacy_actor_std()
            self.bc_dagger_sac_adapter.requires_grad_(True)

    @staticmethod
    def _actor_std_from_module_state(
        module_state: Mapping, *, context: str
    ) -> torch.Tensor:
        candidates = [
            value
            for key, value in module_state.items()
            if str(key).endswith("actor_std") and torch.is_tensor(value)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{context} must contain exactly one PPOVEL actor_std tensor"
            )
        actor_std = candidates[0].detach().reshape(-1)
        if (
            actor_std.numel() == 0
            or not torch.is_floating_point(actor_std)
            or not torch.isfinite(actor_std).all()
        ):
            raise ValueError(f"{context} contains an invalid PPOVEL actor_std")
        return actor_std

    def _restore_checkpoint_physical_actor_std(
        self, module_state: Mapping, *, context: str
    ) -> None:
        """Undo PPOVEL's fresh-finetune 0.5 reset for FastSAC resume only."""
        if not self._uses_ppo_physical_gaussian():
            return
        actor_std = self._actor_std_from_module_state(module_state, context=context)
        parameter = self._ppo_actor_std_parameter()
        if actor_std.shape != parameter.shape:
            raise ValueError(
                f"{context} actor_std shape {tuple(actor_std.shape)} does not match "
                f"runtime action shape {tuple(parameter.shape)}"
            )
        parameter.data.copy_(actor_std.to(parameter))

    @staticmethod
    def _inverse_smooth_log_std(
        log_std: torch.Tensor, log_std_min: float, log_std_max: float
    ) -> torch.Tensor:
        """Inverse-map an effective log std to its unconstrained coordinate."""
        if not math.isfinite(float(log_std_min)) or not math.isfinite(
            float(log_std_max)
        ):
            raise ValueError("FastSAC log-std bounds must be finite")
        if not float(log_std_min) < float(log_std_max):
            raise ValueError("FastSAC log-std bounds must be ordered")
        if (
            not torch.isfinite(log_std).all()
            or ((log_std <= float(log_std_min)) | (log_std >= float(log_std_max))).any()
        ):
            raise ValueError(
                "FastSAC effective log std must be finite and strictly inside "
                "its bounds"
            )
        normalized = (
            2.0
            * (log_std - float(log_std_min))
            / (float(log_std_max) - float(log_std_min))
            - 1.0
        )
        return torch.atanh(normalized)

    def _bounded_log_std(self, raw_log_std: torch.Tensor | None = None) -> torch.Tensor:
        """Smoothly map the adapter parameter into configured log-std bounds."""
        if raw_log_std is None:
            raw_log_std = self.bc_dagger_sac_adapter.log_std
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )

    def _physical_std_bound_mode(self) -> str:
        return str(
            getattr(
                self.cfg,
                "sac_physical_std_bound_mode",
                UNIFORM_PHYSICAL_STD_BOUND_MODE,
            )
        )

    def _physical_std_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint-wise raw std bounds for the selected coordinate mode."""
        cached = getattr(self, "_fastsac_physical_std_bounds_cache", None)
        if cached is not None:
            return cached

        q_scale = getattr(self, "_fastsac_q_action_scale", None)
        if not torch.is_tensor(q_scale):
            q_scale = self._ppo_actor_std_parameter().detach()
        q_scale = q_scale.detach()
        if not torch.isfinite(q_scale).all() or not bool((q_scale > 0.0).all()):
            raise ValueError("physical FastSAC q-action scale is invalid")
        absolute_min = float(getattr(self.cfg, "sac_physical_std_min", 0.05))
        absolute_max = float(getattr(self.cfg, "sac_physical_std_max", 0.5))
        lower = q_scale.new_full(q_scale.shape, absolute_min)
        upper = q_scale.new_full(q_scale.shape, absolute_max)
        mode = self._physical_std_bound_mode()
        if mode == Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE:
            normalized_min = float(
                getattr(self.cfg, "sac_physical_std_normalized_min", 0.02)
            )
            normalized_max = float(
                getattr(self.cfg, "sac_physical_std_normalized_max", 0.11)
            )
            lower = torch.maximum(lower, q_scale * normalized_min)
            upper = torch.minimum(upper, q_scale * normalized_max)
        elif mode != UNIFORM_PHYSICAL_STD_BOUND_MODE:
            raise ValueError(f"unsupported physical std bound mode {mode!r}")

        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("physical FastSAC std bounds contain NaN/Inf")
        invalid = lower >= upper
        if bool(invalid.any()):
            indices = invalid.nonzero(as_tuple=False).reshape(-1).cpu().tolist()
            raise ValueError(
                "physical FastSAC std bounds are empty for joint indices "
                f"{indices}; widen the absolute or q-normalized interval"
            )
        self._fastsac_physical_std_bounds_cache = (
            lower.detach(),
            upper.detach(),
        )
        return self._fastsac_physical_std_bounds_cache

    def _clamp_physical_actor_std(self, std: torch.Tensor) -> torch.Tensor:
        lower, upper = self._physical_std_bounds()
        if std.shape != lower.shape:
            raise ValueError(
                "physical FastSAC q-action scale and actor_std shapes differ"
            )
        return torch.maximum(torch.minimum(std, upper.to(std)), lower.to(std))

    def _bounded_physical_actor_std(self, *, detach: bool = False) -> torch.Tensor:
        """Read PPOVEL's direct std through the FastSAC safety envelope."""
        std = self._clamp_physical_actor_std(self._ppo_actor_std_parameter())
        return std.detach() if detach else std

    @torch.no_grad()
    def _project_physical_actor_std_(self) -> torch.Tensor:
        """Project the direct PPOVEL parameter and report the changed fraction."""
        parameter = self._ppo_actor_std_parameter()
        if not torch.isfinite(parameter).all():
            raise RuntimeError("FastSAC physical actor_std contains NaN/Inf")
        before = parameter.detach().clone()
        parameter.copy_(self._clamp_physical_actor_std(parameter))
        return (parameter != before).float().mean()

    def _physical_normalized_entropy_bounds(self) -> tuple[float, float]:
        """Return total Normal entropy bounds in nominal joint coordinates."""
        cached = getattr(self, "_fastsac_physical_entropy_bounds", None)
        if cached is not None:
            return cached
        q_scale = self._fastsac_q_action_scale.detach().float()
        if not torch.isfinite(q_scale).all() or not bool((q_scale > 0.0).all()):
            raise ValueError("physical FastSAC entropy reference scale is invalid")
        std_min, std_max = self._physical_std_bounds()
        offset = 0.5 * math.log(2.0 * math.pi * math.e)
        minimum = (q_scale.new_tensor(offset) + std_min.log() - q_scale.log()).sum()
        maximum = (q_scale.new_tensor(offset) + std_max.log() - q_scale.log()).sum()
        bounds = (float(minimum.item()), float(maximum.item()))
        self._fastsac_physical_entropy_bounds = bounds
        return bounds

    def _validate_physical_entropy_target_reachable(self) -> None:
        """Fail if autotune would continuously press against a std boundary."""
        minimum, maximum = self._physical_normalized_entropy_bounds()
        target = float(self.target_entropy)
        if not minimum < target < maximum:
            raise ValueError(
                "physical FastSAC target entropy is unreachable within the std "
                f"bounds in nominal joint coordinates: require {minimum} < "
                f"{target} < {maximum}"
            )

    @staticmethod
    def _physical_scale_kl(
        reference_std: torch.Tensor, candidate_std: torch.Tensor
    ) -> torch.Tensor:
        """KL(N(0, reference)||N(0, candidate)) summed over joints."""
        return (
            (
                torch.log(candidate_std / reference_std)
                + reference_std.square() / (2.0 * candidate_std.square())
                - 0.5
            )
            .sum()
            .clamp_min(0.0)
        )

    @torch.no_grad()
    def _cap_physical_actor_std_rollout_kl_(
        self, fallback_reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cap cumulative std-only KL from the rollout's behavior scale."""
        parameter = self._ppo_actor_std_parameter()
        reference = getattr(
            self, "_physical_std_rollout_reference", fallback_reference
        ).to(parameter)
        if reference.shape != parameter.shape or not torch.isfinite(reference).all():
            raise RuntimeError("FastSAC physical std rollout reference is invalid")
        reference = self._clamp_physical_actor_std(reference)
        candidate = self._clamp_physical_actor_std(parameter.detach())
        maximum_kl = float(self.cfg.sac_physical_std_max_kl)
        candidate_kl = self._physical_scale_kl(reference, candidate)
        if float(candidate_kl.item()) <= maximum_kl:
            return candidate_kl, candidate_kl.new_zeros(())

        log_reference = reference.log()
        log_candidate = candidate.log()
        low = 0.0
        high = 1.0
        for _ in range(32):
            fraction = 0.5 * (low + high)
            proposal = torch.exp(
                log_reference + fraction * (log_candidate - log_reference)
            )
            proposal_kl = self._physical_scale_kl(reference, proposal)
            if float(proposal_kl.item()) <= maximum_kl:
                low = fraction
            else:
                high = fraction
        capped = torch.exp(log_reference + low * (log_candidate - log_reference))
        parameter.copy_(self._clamp_physical_actor_std(capped))
        capped_kl = self._physical_scale_kl(reference, parameter.detach())
        return capped_kl, capped_kl.new_ones(())

    def _physical_actor_std_update(
        self,
        before: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the SAC-populated direct-std gradient and hard bounds."""
        if not self._uses_ppo_physical_gaussian():
            raise RuntimeError("physical std update requested for normalized_tanh")
        optimizer = getattr(self, "actor_std_optimizer", None)
        if optimizer is None:
            raise RuntimeError("physical Gaussian Actor has no std optimizer")
        parameter = self._ppo_actor_std_parameter()
        if parameter.grad is None:
            raise RuntimeError(
                "SAC Actor objective did not produce a physical actor_std gradient"
            )
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("FastSAC physical actor_std gradient contains NaN/Inf")
        std_grad = torch.nn.utils.clip_grad_norm_(
            (parameter,), float(self.cfg.sac_max_grad_norm)
        )
        if not torch.isfinite(torch.as_tensor(std_grad)):
            raise RuntimeError("FastSAC physical actor_std gradient norm is NaN/Inf")
        optimizer.step()
        post_projection = self._project_physical_actor_std_()
        after = parameter.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        self.actor_std_update_count = (
            int(getattr(self, "actor_std_update_count", 0)) + 1
        )
        std_min, std_max = self._physical_std_bounds()
        return {
            "actor_std_sac_grad_norm": torch.as_tensor(std_grad).detach(),
            "actor_std_step_abs_mean": (after - before).abs().mean(),
            "actor_std_projection_fraction": post_projection.detach(),
            "actor_std_at_min_fraction": (after <= std_min).float().mean(),
            "actor_std_at_max_fraction": (after >= std_max).float().mean(),
            "actor_std_lr": after.new_tensor(
                float(self.cfg.sac_physical_std_lr)
            ),
        }

    def _policy_log_std(self) -> torch.Tensor:
        """Return the effective std coordinate used by the current policy."""
        if self._uses_ppo_physical_gaussian():
            return self._bounded_physical_actor_std().log()
        return self._bounded_log_std()

    def _policy_raw_log_std(self) -> torch.Tensor:
        if self._uses_ppo_physical_gaussian():
            # PPOVEL parameterizes std directly; there is no latent/raw map.
            return self._policy_log_std()
        return self.bc_dagger_sac_adapter.log_std

    def _legacy_effective_log_std_to_raw(
        self,
        legacy_log_std: torch.Tensor,
        legacy_log_std_min: float,
        legacy_log_std_max: float,
    ) -> torch.Tensor:
        """Convert a v3 hard-clamped effective log std for v4 inference.

        V3 allowed the learned tensor to leave its configured interval and
        hard-clamped it only while building a distribution. Map that effective
        value just inside the open v4 interval so inference retains the v3
        policy scale to floating-point precision without an infinite atanh.
        """
        if not torch.is_tensor(legacy_log_std) or not torch.is_floating_point(
            legacy_log_std
        ):
            raise ValueError("FastSAC v3 checkpoint has invalid adapter log_std")
        if not torch.isfinite(legacy_log_std).all():
            raise ValueError("FastSAC v3 checkpoint has non-finite adapter log_std")
        try:
            legacy_log_std_min = float(legacy_log_std_min)
            legacy_log_std_max = float(legacy_log_std_max)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "FastSAC v3 checkpoint has invalid log-std bounds"
            ) from error
        if not (
            math.isfinite(legacy_log_std_min)
            and math.isfinite(legacy_log_std_max)
            and legacy_log_std_min < legacy_log_std_max
        ):
            raise ValueError("FastSAC v3 checkpoint has invalid log-std bounds")
        # Reproduce the v3 distribution before changing coordinate systems.
        effective = legacy_log_std.clamp(legacy_log_std_min, legacy_log_std_max)
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        normalized = 2.0 * (effective - log_std_min) / (log_std_max - log_std_min) - 1.0
        inward = 4.0 * torch.finfo(normalized.dtype).eps
        return torch.atanh(normalized.clamp(-1.0 + inward, 1.0 - inward))

    def _sac_dist_from_mean(
        self, mean: torch.Tensor
    ) -> FastSACTanhNormal | FastSACPhysicalNormal:
        """Build the configured stochastic policy from a physical-action mean."""
        if not torch.isfinite(mean).all():
            raise RuntimeError("FastSAC Actor proposal contains non-finite raw actions")
        if self._uses_ppo_physical_gaussian():
            # This is the same direct per-joint scale parameter and unbounded
            # physical Normal used by PPOVEL.  The shared execution projection
            # remains only as a far-tail finite-safety guard.
            action_std = self._bounded_physical_actor_std()
            return FastSACPhysicalNormal(mean, action_std.expand_as(mean))
        log_std = self._bounded_log_std()
        actor_center = self._fastsac_actor_action_center.to(mean)
        actor_scale = self._fastsac_actor_action_scale.to(mean)
        q_scale = self._fastsac_q_action_scale.to(mean)
        latent_loc = (mean - actor_center) / actor_scale
        # log_std is dimensionless in nominal Q coordinates. Converting it to
        # pre-tanh coordinates makes the unsquashed physical proposal noise
        # exactly q_scale * exp(log_std) for every joint.
        latent_scale = (log_std.exp() * q_scale / actor_scale).expand_as(mean)
        return FastSACTanhNormal(
            latent_loc,
            latent_scale,
            low=self._fastsac_action_low.to(mean),
            high=self._fastsac_action_high.to(mean),
            event_dims=1,
        )

    def _actor_mean_from_flat(self, actor_obs: torch.Tensor) -> torch.Tensor:
        # Lightweight unit policies may expose the mean module directly.  The
        # production VAIC actor instead has the TensorDict ``get_dist`` API.
        if not hasattr(self, "observation_spec"):
            return self.actor_adapt(actor_obs)
        legacy_dist = DistributionalTD3TeacherBC._actor_dist_from_flat(self, actor_obs)
        return legacy_dist.mean

    def _actor_dist_from_flat(
        self, actor_obs: torch.Tensor
    ) -> FastSACTanhNormal | FastSACPhysicalNormal:
        return self._sac_dist_from_mean(self._actor_mean_from_flat(actor_obs))

    def _normalized_action_log_prob(self, raw_log_prob: torch.Tensor):
        """Convert raw-action density to joint-normalized-coordinate density."""
        # The additive Jacobian is constant with respect to both mean and std,
        # so physical SAC keeps PPOVEL's exact raw Normal Actor gradient while
        # alpha autotune sees the same dimensionless nominal-joint coordinates
        # as Q and the configured target entropy.
        return raw_log_prob + float(self._fastsac_entropy_reference_log_scale_sum)

    @torch.no_grad()
    def _student_mean_action(self, td: TensorDict) -> torch.Tensor:
        """Use the distribution-consistent deterministic Student action."""
        if not self._uses_ppo_physical_gaussian():
            return DistributionalTD3TeacherBC._student_mean_action(self, td)
        raw_mean = self._student_raw_action_proposal(td)
        if not torch.isfinite(raw_mean).all():
            raise RuntimeError("FastSAC evaluation Actor produced non-finite actions")
        return self._project_execution_action(raw_mean)

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return _DistributionalFastSACDaggerRolloutPolicy(self)
        return _DeterministicFastSACStudentEvalPolicy(self)

    @torch.no_grad()
    def _distributional_fastsac_target(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        next_dist = self._actor_dist_from_flat(batch["next_observations"])
        next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        next_log_prob = self._normalized_action_log_prob(next_raw_log_prob)
        bootstrap = (batch["truncations"].bool() | ~batch["dones"].bool()).float()
        effective_discount, _, effective_n_steps = _q_target_discount_factors(
            self.cfg, batch
        )
        alpha = self.log_alpha.exp()
        entropy_tax = effective_discount * bootstrap * alpha * next_log_prob
        soft_reward = batch["rewards"] - entropy_tax

        target_logits = self.qnet_target(
            batch["next_critic_observations"],
            self._q_action_features(
                next_action,
                batch.get(NEXT_Q_ACTUATOR_CONTEXT_KEY),
            ),
        )
        target_probabilities = F.softmax(target_logits, dim=-1)
        raw_expected_heads = (target_probabilities * self.qnet_target.support).sum(
            dim=-1
        )
        projected_heads = []
        support_clip_fractions = []
        for head_probability in target_probabilities:
            projected, left_fraction, right_fraction = _project_c51_probabilities(
                head_probability,
                soft_reward,
                bootstrap,
                effective_discount,
                self.qnet_target.support,
            )
            projected_heads.append(projected)
            support_clip_fractions.append((left_fraction, right_fraction))
        if len(support_clip_fractions) != 2:
            raise RuntimeError("FastSAC C51 target must contain exactly two Q heads")
        (
            (q1_left_support_clip_fraction, q1_right_support_clip_fraction),
            (q2_left_support_clip_fraction, q2_right_support_clip_fraction),
        ) = support_clip_fractions
        q1_support_clip_fraction = (
            q1_left_support_clip_fraction + q1_right_support_clip_fraction
        )
        q2_support_clip_fraction = (
            q2_left_support_clip_fraction + q2_right_support_clip_fraction
        )
        support_clip_fraction_mean = 0.5 * (
            q1_support_clip_fraction + q2_support_clip_fraction
        )
        support_clip_fraction_max = torch.maximum(
            q1_support_clip_fraction, q2_support_clip_fraction
        )
        left_support_projection_clipping_fraction = 0.5 * (
            q1_left_support_clip_fraction + q2_left_support_clip_fraction
        )
        right_support_projection_clipping_fraction = 0.5 * (
            q1_right_support_clip_fraction + q2_right_support_clip_fraction
        )
        projected_heads = torch.stack(projected_heads, dim=0)
        projected_expected_heads = (projected_heads * self.qnet_target.support).sum(
            dim=-1
        )
        # Clipped double-Q chooses the lower next-state distribution before the
        # reward/discount C51 projection.  Projection is nonlinear at support
        # boundaries, so selecting after projection can reverse the intended
        # head exactly on the rows where clipping is most consequential.
        selected_head = raw_expected_heads.argmin(dim=0)
        selected_target = projected_heads.gather(
            0,
            selected_head[None, :, None].expand(
                1, projected_heads.shape[1], projected_heads.shape[2]
            ),
        ).squeeze(0)
        selected_expected = projected_expected_heads.gather(
            0, selected_head.unsqueeze(0)
        ).squeeze(0)
        selected_entropy = -(
            selected_target
            * selected_target.clamp_min(torch.finfo(selected_target.dtype).tiny).log()
        ).sum(dim=-1)
        reward_abs = batch["rewards"].abs().mean()
        metrics = {
            "target_expected_q1_mean": raw_expected_heads[0].mean(),
            "target_expected_q2_mean": raw_expected_heads[1].mean(),
            "projected_target_mean": selected_expected.mean(),
            "selected_target_expected_mean": selected_expected.mean(),
            "target_distribution_entropy": selected_entropy.mean(),
            "target_select_q1_fraction": (selected_head == 0).float().mean(),
            "target_select_q2_fraction": (selected_head == 1).float().mean(),
            "q1_left_support_clip_fraction": q1_left_support_clip_fraction,
            "q1_right_support_clip_fraction": q1_right_support_clip_fraction,
            "q2_left_support_clip_fraction": q2_left_support_clip_fraction,
            "q2_right_support_clip_fraction": q2_right_support_clip_fraction,
            "support_clip_fraction_mean": support_clip_fraction_mean,
            "support_clip_fraction_max": support_clip_fraction_max,
            # Keep the historical directional aliases for existing dashboards
            # and for the shared TD3 rollout aggregation surface. Unlike the
            # old loop-overwrite behavior, each alias now averages both heads.
            "left_support_projection_clipping_fraction": (
                left_support_projection_clipping_fraction
            ),
            "right_support_projection_clipping_fraction": (
                right_support_projection_clipping_fraction
            ),
            "target_smoothing_noise_norm": soft_reward.new_zeros(()),
            "target_noise_free_action_abs_mean": next_dist.mean.abs().mean(),
            "target_sample_action_abs_mean": next_action.abs().mean(),
            "target_log_prob_mean": next_log_prob.mean(),
            "entropy_tax_mean": entropy_tax.mean(),
            "entropy_tax_abs_mean": entropy_tax.abs().mean(),
            "entropy_tax_reward_abs_ratio": entropy_tax.abs().mean()
            / reward_abs.clamp_min(torch.finfo(reward_abs.dtype).eps),
            "effective_n_steps_mean": effective_n_steps.float().mean(),
            "alpha": alpha.detach(),
        }
        return selected_target.detach(), metrics, next_log_prob.detach()

    # Compatibility alias used by focused callers which name the soft target.
    _soft_c51_target = _distributional_fastsac_target

    def _alpha_update(self, log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = log_prob.new_zeros(())
        if log_prob.numel() == 0 or not bool(self.cfg.sac_use_autotune):
            return {
                "alpha_loss": zero,
                "alpha_grad_norm": zero,
                "alpha": self.log_alpha.exp().detach(),
            }
        if self.alpha_optimizer is None:
            raise RuntimeError("FastSAC alpha autotune has no optimizer")
        # Optimize the standard log-temperature surrogate. Its gradient is the
        # entropy error itself rather than alpha times that error, so a cautious
        # alpha initialization (1e-5 in the default config) can still recover.
        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + float(self.target_entropy))
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad = self.log_alpha.grad.detach().abs().clone()
        self.alpha_optimizer.step()
        self.alpha_update_count = int(getattr(self, "alpha_update_count", 0)) + 1
        self.sac_alpha_update_count = (
            int(getattr(self, "sac_alpha_update_count", 0)) + 1
        )
        return {
            "alpha_loss": alpha_loss.detach(),
            "alpha_grad_norm": alpha_grad,
            "alpha": self.log_alpha.exp().detach(),
        }

    def _critic_update(self, batch: dict[str, torch.Tensor]):
        """One twin-C51 update and a configured-cadence temperature update."""
        projected_target, target_metrics, target_log_prob = (
            self._distributional_fastsac_target(batch)
        )
        logits = self.qnet(
            batch["critic_observations"],
            self._q_action_features(
                batch["actions"],
                batch.get(Q_ACTUATOR_CONTEXT_KEY),
            ),
        )
        log_probabilities = F.log_softmax(logits, dim=-1)
        per_head = (
            -(projected_target.unsqueeze(0) * log_probabilities)
            .sum(dim=-1)
            .mean(dim=-1)
        )
        critic_loss = per_head.sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.qnet.parameters(), float(self.cfg.sac_max_grad_norm)
        )
        self.critic_optimizer.step()
        self.critic_update_count += 1

        # The sampled next action has no learning meaning after a true
        # environment terminal.  Time-limit truncations do retain their real
        # next state and therefore stay in the temperature population, exactly
        # matching the soft Bellman bootstrap truth table above.
        alpha_bootstrap = (
            batch["truncations"].bool() | ~batch["dones"].bool()
        ).reshape(-1)
        alpha_log_prob = target_log_prob.reshape(-1)
        if alpha_bootstrap.numel() != alpha_log_prob.numel():
            raise ValueError(
                "FastSAC alpha bootstrap mask and target log-probability "
                "population are misaligned"
            )
        # Baseline temperature and Actor see the same policy-update timescale.
        # Cadence is evaluated after incrementing ``critic_update_count``, so
        # an Actor-cadence update adjusts alpha immediately before the matching
        # Actor update in the inherited replay loop. TVKD checkpoints keep the
        # historical every-Critic cadence for resume compatibility.
        alpha_update_cadence = str(self.cfg.sac_alpha_update_cadence)
        alpha_update_due = alpha_update_cadence == "critic" or (
            self.critic_update_count % int(self.cfg.sac_policy_frequency) == 0
        )
        alpha_update_count_before = int(getattr(self, "alpha_update_count", 0))
        if alpha_update_due:
            alpha_metrics = self._alpha_update(alpha_log_prob[alpha_bootstrap])
        else:
            # Reuse the empty-population path to publish zero per-step loss and
            # gradient without touching the optimizer.
            alpha_metrics = self._alpha_update(alpha_log_prob[:0])
        alpha_update_performed = (
            int(getattr(self, "alpha_update_count", 0)) > alpha_update_count_before
        )
        alpha_metrics.update(
            {
                "alpha_update_due_fraction": alpha_log_prob.new_tensor(
                    float(alpha_update_due)
                ),
                "alpha_update_performed_fraction": alpha_log_prob.new_tensor(
                    float(alpha_update_performed)
                ),
            }
        )
        _polyak_update_(self.qnet_target, self.qnet, float(self.cfg.sac_tau))
        self.qnet_target.requires_grad_(False).eval()
        with torch.no_grad():
            expected_heads = (log_probabilities.detach().exp() * self.qnet.support).sum(
                dim=-1
            )
        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_loss_1": per_head[0].detach(),
            "critic_loss_2": per_head[1].detach(),
            "critic_grad_norm": torch.as_tensor(critic_grad).detach(),
            "expected_q1_mean": expected_heads[0].mean(),
            "expected_q2_mean": expected_heads[1].mean(),
            "twin_expected_q_disagreement": (expected_heads[0] - expected_heads[1])
            .abs()
            .mean(),
            **target_metrics,
            **alpha_metrics,
        }
        if hasattr(self, "_fastsac_rollout_critic_metrics"):
            self._fastsac_rollout_critic_metrics.append(metrics)
        return metrics

    def _actor_update(self, batch: dict[str, torch.Tensor]):
        """One mean-only BC step plus the ordinary reparameterized SAC step."""
        raw_prediction = self._actor_mean_from_flat(batch["observations"])
        physical_gaussian = self._uses_ppo_physical_gaussian()
        std_before = None
        if physical_gaussian:
            self.actor_std_optimizer.zero_grad(set_to_none=True)
            std_before = self._ppo_actor_std_parameter().detach().clone()
        dist = self._sac_dist_from_mean(raw_prediction)
        prediction_action = dist.mean
        sampled_action, raw_log_prob = dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        normalized_log_prob = self._normalized_action_log_prob(raw_log_prob)
        q_action = self._q_action_features(
            sampled_action,
            batch.get(Q_ACTUATOR_CONTEXT_KEY),
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        original_requires_grad = [
            parameter.requires_grad for parameter in self.qnet.parameters()
        ]
        try:
            for parameter in self.qnet.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            twin_logits = self.qnet(batch["critic_observations"], q_action)
            twin_expected = (F.softmax(twin_logits, dim=-1) * self.qnet.support).sum(
                dim=-1
            )
            minimum_expected = _reduce_actor_q_values(twin_expected, True)
            sac_actor_loss = (
                self.log_alpha.exp().detach() * normalized_log_prob - minimum_expected
            ).mean()
        finally:
            for parameter, requires_grad in zip(
                self.qnet.parameters(), original_requires_grad
            ):
                parameter.requires_grad_(requires_grad)

        teacher_actions = batch[DAGGER_REPLAY_TEACHER_ACTIONS]
        teacher_valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY].reshape(-1).bool()
        q_filter_metrics = {
            # Compatibility alias: under continuous SPReD-P this is the mean
            # effective BC weight, not a fraction of binary-active rows.
            "q_filtered_bc_active_fraction": prediction_action.new_tensor(1.0),
            "q_filtered_bc_policy_better_fraction": prediction_action.new_zeros(()),
            "spred_p_bc_weight_mean": prediction_action.new_tensor(1.0),
            "spred_p_bc_weight_std": prediction_action.new_zeros(()),
            "spred_p_bc_weight_min": prediction_action.new_tensor(1.0),
            "spred_p_bc_weight_max": prediction_action.new_tensor(1.0),
            "spred_p_teacher_advantage_mean": prediction_action.new_zeros(()),
            "spred_p_combined_q_std_mean": prediction_action.new_zeros(()),
            **{
                key: prediction_action.new_zeros(())
                for key in _SPRED_P_SOURCE_METRIC_KEYS
            },
        }
        if getattr(self.cfg, "use_q_filtered_bc", False):
            if prediction_action.shape != teacher_actions.shape:
                raise ValueError("SPReD-P Student and Teacher action shapes must match")
            if prediction_action.shape[0] != teacher_valid.numel():
                raise ValueError("SPReD-P validity mask does not match batch rows")
            # Invalid replay labels may contain NaN. Substitute a finite action
            # before the online-Q forward, then exclude those rows from both
            # the probability statistics and BC normalization below.
            safe_teacher_actions = torch.where(
                teacher_valid.unsqueeze(-1),
                teacher_actions,
                prediction_action.detach(),
            )
            with torch.no_grad():
                # SPReD-P uses the current online ensemble.  Collapse each C51
                # head to its expected return before fitting the two Gaussian
                # action-value distributions.  Sequential forwards keep peak
                # memory lower than concatenating a doubled Actor batch.
                policy_online_logits = self.qnet(
                    batch["critic_observations"],
                    self._q_action_features(
                        prediction_action.detach(),
                        batch.get(Q_ACTUATOR_CONTEXT_KEY),
                    ),
                )
                policy_online_q = (
                    F.softmax(policy_online_logits, dim=-1) * self.qnet.support
                ).sum(dim=-1)
                del policy_online_logits
                teacher_online_logits = self.qnet(
                    batch["critic_observations"],
                    self._q_action_features(
                        safe_teacher_actions.detach(),
                        batch.get(Q_ACTUATOR_CONTEXT_KEY),
                    ),
                )
                teacher_online_q = (
                    F.softmax(teacher_online_logits, dim=-1) * self.qnet.support
                ).sum(dim=-1)
                (
                    teacher_probability,
                    teacher_advantage,
                    combined_q_std,
                ) = _spred_p_teacher_probability(
                    policy_online_q,
                    teacher_online_q,
                )

            if teacher_valid.any():
                center = self._fastsac_q_action_center.to(prediction_action)
                scale = self._fastsac_q_action_scale.to(prediction_action)
                per_element_bc = F.smooth_l1_loss(
                    (prediction_action - center) / scale,
                    (safe_teacher_actions.detach() - center) / scale,
                    beta=float(self.cfg.dagger_actor_huber_delta),
                    reduction="none",
                )
                per_row_bc = per_element_bc.flatten(start_dim=1).mean(dim=1)
                bc_weights = torch.where(
                    teacher_valid,
                    teacher_probability.to(per_row_bc),
                    torch.zeros_like(per_row_bc),
                )
                # Normalize by all valid Teacher rows exactly as an ordinary
                # per-demo weighted loss.  Its effective strength changes
                # continuously without a hand-authored lambda scheduler.
                exact_bc_loss = (
                    per_row_bc * bc_weights
                ).sum() / teacher_valid.sum()
                valid_weights = teacher_probability[teacher_valid]
                valid_advantage = teacher_advantage[teacher_valid]
                valid_combined_std = combined_q_std[teacher_valid]
                policy_better = valid_advantage < 0.0
                q_filter_metrics.update(
                    {
                        "q_filtered_bc_active_fraction": valid_weights.mean(),
                        "q_filtered_bc_policy_better_fraction": (
                            policy_better.float().mean()
                        ),
                        "spred_p_bc_weight_mean": valid_weights.mean(),
                        "spred_p_bc_weight_std": valid_weights.std(unbiased=False),
                        "spred_p_bc_weight_min": valid_weights.min(),
                        "spred_p_bc_weight_max": valid_weights.max(),
                        "spred_p_teacher_advantage_mean": valid_advantage.mean(),
                        "spred_p_combined_q_std_mean": valid_combined_std.mean(),
                    }
                )
            else:
                exact_bc_loss = prediction_action.sum() * 0.0
                q_filter_metrics = {
                    key: prediction_action.new_zeros(()) for key in q_filter_metrics
                }

            # A single aggregate can hide opposite behavior on factual
            # Teacher-replay rows and counterfactual Teacher labels attached to
            # Student-replay rows.  Preserve the replay provenance and report
            # SPReD-P diagnostics independently for both sources.  These
            # statistics are read-only: neither the masks nor the detached
            # probability feed a new gradient path into the Critic.
            teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
            if teacher_source is not None:
                if teacher_source.numel() != teacher_valid.numel():
                    raise ValueError(
                        "SPReD-P replay-source mask does not match batch rows"
                    )
                teacher_source = teacher_source.reshape(-1).bool()
                q_filter_metrics["spred_p_source_metadata_available"] = (
                    prediction_action.new_tensor(1.0)
                )
                policy_better_values = (teacher_advantage < 0.0).float()
                for source_name, source_mask in (
                    ("teacher", teacher_source),
                    ("student", ~teacher_source),
                ):
                    source_valid = teacher_valid & source_mask
                    source_valid_float = source_valid.to(teacher_probability)
                    source_valid_count = source_valid_float.sum().clamp_min(1.0)
                    q_filter_metrics.update(
                        {
                            f"spred_p_{source_name}_source_valid_fraction": (
                                source_valid_float.mean()
                            ),
                            f"spred_p_{source_name}_source_bc_weight_mean": (
                                (teacher_probability * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"spred_p_{source_name}_source_teacher_advantage_mean": (
                                (teacher_advantage * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"spred_p_{source_name}_source_combined_q_std_mean": (
                                (combined_q_std * source_valid_float).sum()
                                / source_valid_count
                            ),
                            f"q_filtered_bc_{source_name}_source_policy_better_fraction": (
                                (policy_better_values * source_valid_float).sum()
                                / source_valid_count
                            ),
                        }
                    )
        else:
            exact_bc_loss = _exact_teacher_bc_loss(
                prediction_action,
                teacher_actions,
                teacher_valid,
                self._fastsac_q_action_center,
                self._fastsac_q_action_scale,
                float(self.cfg.dagger_actor_huber_delta),
            )
        weighted_sac = float(self.cfg.eta_sac) * sac_actor_loss
        weighted_bc = float(self.cfg.lambda_bc) * exact_bc_loss
        total_actor_loss = weighted_sac + weighted_bc
        total_actor_loss.backward()
        if physical_gaussian and self._ppo_actor_std_parameter().grad is None:
            raise RuntimeError(
                "SAC Actor loss did not update physical actor_std"
            )
        actor_parameters = tuple(
            parameter
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        )
        actor_grad = torch.nn.utils.clip_grad_norm_(
            actor_parameters, float(self.cfg.sac_max_grad_norm)
        )
        if any(parameter.grad is not None for parameter in self.qnet.parameters()):
            raise RuntimeError("Critic parameters accumulated Actor-step gradients")
        self.actor_optimizer.step()
        if physical_gaussian:
            if std_before is None:
                raise RuntimeError("physical std update snapshot was not captured")
            std_metrics = self._physical_actor_std_update(std_before)
        else:
            std_metrics = {}
        self.actor_update_count = int(getattr(self, "actor_update_count", 0)) + 1
        self.sac_actor_update_count = (
            int(getattr(self, "sac_actor_update_count", 0)) + 1
        )
        teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
        actor_teacher_replay_fraction = (
            prediction_action.new_zeros(())
            if teacher_source is None
            else teacher_source.float().mean()
        )
        metrics = {
            # Compatibility names consumed by the inherited replay loop.
            "td3_actor_loss": sac_actor_loss.detach(),
            "weighted_td3_actor_loss": weighted_sac.detach(),
            # Native FastSAC diagnostics.
            "sac_actor_loss": sac_actor_loss.detach(),
            "exact_bc_loss": exact_bc_loss.detach(),
            "weighted_sac_actor_loss": weighted_sac.detach(),
            "weighted_bc_loss": weighted_bc.detach(),
            "total_actor_loss": total_actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad).detach(),
            "actor_mean_grad_norm": torch.as_tensor(actor_grad).detach(),
            "actor_expected_q1_mean": twin_expected[0].detach().mean(),
            "actor_expected_q2_mean": twin_expected[1].detach().mean(),
            "actor_min_expected_q_mean": minimum_expected.detach().mean(),
            "actor_expected_q_min_mean": minimum_expected.detach().mean(),
            "actor_log_prob_mean": normalized_log_prob.detach().mean(),
            "actor_entropy": -normalized_log_prob.detach().mean(),
            "actor_sample_action_abs_mean": sampled_action.detach().abs().mean(),
            "actor_mean_action_abs_mean": dist.mean.detach().abs().mean(),
            "actor_log_std_mean": self._policy_log_std().detach().mean(),
            "actor_raw_log_std_mean": self._policy_raw_log_std().detach().mean(),
            "actor_std_min": self._policy_log_std().detach().exp().min(),
            "actor_std_max": self._policy_log_std().detach().exp().max(),
            "actor_teacher_replay_fraction": (actor_teacher_replay_fraction.detach()),
            **{key: value.detach() for key, value in q_filter_metrics.items()},
            "actor_failure_phase_teacher_fraction": batch.get(
                FAILURE_PHASE_TEACHER_SOURCE_KEY,
                torch.zeros_like(batch[DAGGER_TEACHER_ACTION_VALID_KEY]),
            )
            .float()
            .mean()
            .detach(),
            "actor_failure_phase_student_fraction": batch.get(
                FAILURE_PHASE_STUDENT_SOURCE_KEY,
                torch.zeros_like(batch[DAGGER_TEACHER_ACTION_VALID_KEY]),
            )
            .float()
            .mean()
            .detach(),
            "alpha": self.log_alpha.exp().detach(),
            **std_metrics,
        }
        if hasattr(self, "_fastsac_rollout_actor_metrics"):
            self._fastsac_rollout_actor_metrics.append(metrics)
        return metrics

    def _maybe_delayed_actor_and_targets(self, actor_batch: dict[str, torch.Tensor]):
        if self.critic_update_count % int(self.cfg.sac_policy_frequency):
            return None
        # SAC has no target Actor.  Q target Polyak is performed on every
        # Critic update, not on this delayed policy cadence.
        return self._actor_update(actor_batch)

    def train_op(self, tensordict):
        """Reuse raw replay/prefill orchestration and publish SAC-native logs."""
        self._fastsac_rollout_critic_metrics: list[dict[str, torch.Tensor]] = []
        self._fastsac_rollout_actor_metrics: list[dict[str, torch.Tensor]] = []
        prefill_teacher_noise_q_rms = 0.0
        prefill_teacher_noise_physical_rms = 0.0
        prefill_teacher_projection_fraction = 0.0
        if (
            self._teacher_prefill_active()
            and FASTSAC_PREFILL_TEACHER_NOISE_KEY
            in tensordict.keys(True, True)
            and DAGGER_IS_STUDENT_ACTION_KEY in tensordict.keys(True, True)
        ):
            teacher_rows = ~tensordict[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
            if bool(teacher_rows.any()):
                q_noise = tensordict[FASTSAC_PREFILL_TEACHER_NOISE_KEY].reshape(
                    -1, self.action_dim
                )[teacher_rows]
                prefill_teacher_noise_q_rms = float(
                    q_noise.float().square().mean().sqrt().item()
                )
                physical_noise = (
                    q_noise
                    / float(self.cfg.q_action_input_gain)
                    * self._fastsac_q_action_scale.to(q_noise)
                )
                prefill_teacher_noise_physical_rms = float(
                    physical_noise.float().square().mean().sqrt().item()
                )
                if (
                    FASTSAC_PREFILL_TEACHER_PROJECTION_KEY
                    in tensordict.keys(True, True)
                ):
                    projection = tensordict[
                        FASTSAC_PREFILL_TEACHER_PROJECTION_KEY
                    ].reshape(-1).bool()
                    prefill_teacher_projection_fraction = float(
                        projection[teacher_rows].float().mean().item()
                    )
        if self._uses_ppo_physical_gaussian():
            self._project_physical_actor_std_()
            self._physical_std_rollout_reference = (
                self._bounded_physical_actor_std(detach=True).clone()
            )
        action_projection_fraction = 0.0
        if (
            FASTSAC_ACTION_PROJECTION_KEY in tensordict.keys(True, True)
            and DAGGER_IS_STUDENT_ACTION_KEY in tensordict.keys(True, True)
        ):
            projected = tensordict[FASTSAC_ACTION_PROJECTION_KEY].reshape(-1).bool()
            student = tensordict[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
            if bool(student.any()):
                action_projection_fraction = float(
                    projected[student].float().mean().item()
                )
        td3_info = DistributionalTD3TeacherBC.train_op(self, tensordict)
        physical_rollout_std_metrics = {}
        if self._uses_ppo_physical_gaussian():
            # Replay optimization has finished and no environment action can be
            # issued until this method returns. Enforce the cumulative old-scale
            # trust guard once on the policy that will collect the next rollout;
            # hard physical bounds were retained after every std Adam step.
            rollout_scale_kl, rollout_kl_capped = (
                self._cap_physical_actor_std_rollout_kl_(
                    self._physical_std_rollout_reference
                )
            )
            physical_rollout_std_metrics = {
                "actor_std_rollout_scale_kl": float(rollout_scale_kl.item()),
                "actor_std_kl_cap_fraction": float(rollout_kl_capped.item()),
            }

        replacements = {
            "method_distributional_td3_teacher_bc_v1": (
                "method_distributional_fastsac_teacher_bc_v1"
            ),
            "td3_actor_loss": "sac_actor_loss",
            "weighted_td3_actor_loss": "weighted_sac_actor_loss",
            "collector_exploration_noise_norm": ("behavior_sample_q_deviation_norm"),
        }
        info = {}
        for key, value in td3_info.items():
            if not key.startswith("td3/"):
                info[key] = value
                continue
            suffix = key.removeprefix("td3/")
            if suffix == "target_smoothing_noise_norm":
                continue
            info[f"fastsac/{replacements.get(suffix, suffix)}"] = value

        info["fastsac/prefill_teacher_ppo_noise_enabled"] = float(
            bool(getattr(self.cfg, "teacher_prefill_use_ppo_noise", False))
        )
        info["fastsac/prefill_teacher_ppo_noise_q_rms"] = (
            prefill_teacher_noise_q_rms
        )
        info["fastsac/prefill_teacher_ppo_noise_physical_rms"] = (
            prefill_teacher_noise_physical_rms
        )
        info["fastsac/prefill_teacher_action_projection_fraction"] = (
            prefill_teacher_projection_fraction
        )

        critic_keys = (
            "target_sample_action_abs_mean",
            "target_log_prob_mean",
            "entropy_tax_mean",
            "entropy_tax_abs_mean",
            "entropy_tax_reward_abs_ratio",
            "effective_n_steps_mean",
            "alpha_update_due_fraction",
            "alpha_update_performed_fraction",
            "q1_left_support_clip_fraction",
            "q1_right_support_clip_fraction",
            "q2_left_support_clip_fraction",
            "q2_right_support_clip_fraction",
            "support_clip_fraction_mean",
            "support_clip_fraction_max",
        )
        actor_keys = (
            "actor_mean_grad_norm",
            "actor_expected_q1_mean",
            "actor_expected_q2_mean",
            "actor_min_expected_q_mean",
            "actor_log_prob_mean",
            "actor_entropy",
            "actor_sample_action_abs_mean",
            "actor_mean_action_abs_mean",
            "actor_log_std_mean",
            "actor_raw_log_std_mean",
            "actor_std_min",
            "actor_std_max",
            "q_filtered_bc_active_fraction",
            "q_filtered_bc_policy_better_fraction",
            "spred_p_bc_weight_mean",
            "spred_p_bc_weight_std",
            "spred_p_bc_weight_min",
            "spred_p_bc_weight_max",
            "spred_p_teacher_advantage_mean",
            "spred_p_combined_q_std_mean",
            *_SPRED_P_SOURCE_METRIC_KEYS,
        )
        if self._uses_ppo_physical_gaussian():
            actor_keys += (
                "actor_std_sac_grad_norm",
                "actor_std_step_abs_mean",
                "actor_std_projection_fraction",
                "actor_std_at_min_fraction",
                "actor_std_at_max_fraction",
                "actor_std_lr",
            )
        critic = self._mean_metric_dict(
            self._fastsac_rollout_critic_metrics, critic_keys
        )
        # Skipped Critic steps intentionally carry zero temperature metrics.
        # Report loss/gradient over optimizer steps only, so a delayed cadence
        # does not dilute them in the rollout aggregate.
        if self._fastsac_rollout_critic_metrics:
            performed = torch.stack(
                [
                    metrics["alpha_update_performed_fraction"].detach().float()
                    for metrics in self._fastsac_rollout_critic_metrics
                ]
            )
            performed_count = performed.sum().clamp_min(1.0)
            for key in ("alpha_loss", "alpha_grad_norm"):
                values = torch.stack(
                    [
                        metrics[key].detach().float()
                        for metrics in self._fastsac_rollout_critic_metrics
                    ]
                )
                critic[key] = ((values * performed).sum() / performed_count).item()
        else:
            critic.update({"alpha_loss": 0.0, "alpha_grad_norm": 0.0})
        # Publish the temperature actually available to the next rollout,
        # rather than the within-rollout mean of stale and updated values.
        critic["alpha"] = self.log_alpha.detach().exp().item()
        actor = self._mean_metric_dict(self._fastsac_rollout_actor_metrics, actor_keys)
        actor.update(physical_rollout_std_metrics)
        info.update({f"fastsac/{key}": value for key, value in critic.items()})
        info.update({f"fastsac/{key}": value for key, value in actor.items()})
        info["fastsac/action_projection_fraction"] = action_projection_fraction
        info["fastsac/alpha_update_count"] = self.alpha_update_count
        info["fastsac/actor_std_update_count"] = int(
            getattr(self, "actor_std_update_count", 0)
        )
        if self._uses_ppo_physical_gaussian():
            current_std = self._bounded_physical_actor_std(detach=True)
            q_scale = self._fastsac_q_action_scale.detach().to(current_std)
            normalized_std = current_std / q_scale
            lower, upper = self._physical_std_bounds()
            current_std_values = current_std.float().cpu().tolist()
            for joint_name, std in zip(
                self.joint_names, current_std_values, strict=True
            ):
                info[f"fastsac/actor_std/{joint_name}"] = float(std)
            info["fastsac/actor_std_physical_mean"] = float(
                current_std.float().mean().item()
            )
            info["fastsac/actor_std_physical_geometric_mean"] = float(
                current_std.float().log().mean().exp().item()
            )
            info["fastsac/actor_std_q_normalized_l2"] = float(
                torch.linalg.vector_norm(normalized_std.float()).item()
            )
            info["fastsac/actor_std_q_normalized_rms"] = float(
                normalized_std.float().square().mean().sqrt().item()
            )
            info["fastsac/actor_std_q_normalized_geometric_mean"] = float(
                normalized_std.float().log().mean().exp().item()
            )
            info["fastsac/actor_std_q_normalized_min"] = float(
                normalized_std.float().min().item()
            )
            info["fastsac/actor_std_q_normalized_max"] = float(
                normalized_std.float().max().item()
            )
            info["fastsac/physical_std_lower_bound_min"] = float(
                lower.float().min().item()
            )
            info["fastsac/physical_std_lower_bound_max"] = float(
                lower.float().max().item()
            )
            info["fastsac/physical_std_upper_bound_min"] = float(
                upper.float().min().item()
            )
            info["fastsac/physical_std_upper_bound_max"] = float(
                upper.float().max().item()
            )
            entropy_min, entropy_max = self._physical_normalized_entropy_bounds()
            info["fastsac/physical_entropy_bound_min"] = entropy_min
            info["fastsac/physical_entropy_bound_max"] = entropy_max
        info["fastsac/target_entropy"] = float(self.target_entropy)
        self._last_fastsac_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if key.startswith("fastsac/") and isinstance(value, (int, float))
        }
        return info

    def _q_backend_metadata(self):
        metadata = DistributionalTD3TeacherBC._q_backend_metadata(self)
        alpha_update_cadence = str(self.cfg.sac_alpha_update_cadence)
        metadata.update(
            {
                "target_semantics": CRITIC_SEMANTICS,
                "actor_q_reduction": "minimum_online_twin_expectations",
                "actor_mean_optimizer_semantics": ACTOR_MEAN_OPTIMIZER_SEMANTICS,
                "actor_mean_weight_decay": _fastsac_actor_weight_decay(self.cfg),
                "bc_weighting_semantics": (
                    SPRED_P_BC_SEMANTICS
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "fixed_unweighted_valid_teacher_bc"
                ),
                "actor_target": False,
                "stochastic_actor": True,
                "action_distribution": _fastsac_action_distribution(self.cfg),
                "entropy_semantics": self._entropy_semantics(),
                "entropy_reference_log_scale_sum": (
                    self._fastsac_entropy_reference_log_scale_sum
                ),
                "target_entropy": float(self.target_entropy),
                "temperature_update_cadence": (
                    "delayed_actor_update"
                    if alpha_update_cadence == "actor"
                    else "every_critic_update"
                ),
                "temperature_update_frequency_critic_steps": (
                    int(self.cfg.sac_policy_frequency)
                    if alpha_update_cadence == "actor"
                    else 1
                ),
            }
        )
        if self._uses_ppo_physical_gaussian():
            entropy_min, entropy_max = self._physical_normalized_entropy_bounds()
            lower, upper = self._physical_std_bounds()
            metadata.update(
                {
                    "physical_std_update_semantics": (
                        "sac_q_entropy_gradient_separate_low_lr_adam_rollout_scale_"
                        "kl_cap_then_configured_jointwise_hard_projection_v3"
                    ),
                    "physical_std_lr": float(self.cfg.sac_physical_std_lr),
                    "physical_std_max_kl": float(
                        self.cfg.sac_physical_std_max_kl
                    ),
                    "physical_std_bound_mode": self._physical_std_bound_mode(),
                    "physical_std_min": float(self.cfg.sac_physical_std_min),
                    "physical_std_max": float(self.cfg.sac_physical_std_max),
                    "physical_std_normalized_min": float(
                        getattr(self.cfg, "sac_physical_std_normalized_min", 0.02)
                    ),
                    "physical_std_normalized_max": float(
                        getattr(self.cfg, "sac_physical_std_normalized_max", 0.11)
                    ),
                    "physical_std_resolved_lower": lower.detach().cpu().tolist(),
                    "physical_std_resolved_upper": upper.detach().cpu().tolist(),
                    "physical_entropy_bound_min": entropy_min,
                    "physical_entropy_bound_max": entropy_max,
                }
            )
        return metadata

    def _checkpoint_config(self):
        common = DistributionalTD3TeacherBC._checkpoint_config(self)
        for name in (
            "eta_td3",
            "target_policy_noise_std",
            "target_policy_noise_clip",
            "collector_exploration_noise_std",
            "collector_exploration_noise_clip",
        ):
            common.pop(name, None)
        common.update(
            {
                name: getattr(self.cfg, name)
                for name in (
                    "eta_sac",
                    "lambda_bc",
                    "use_q_filtered_bc",
                    "teacher_prefill_use_ppo_noise",
                    "q_condition_on_actuator_state",
                    "q_use_predicted_effect",
                    "q_use_residual_film",
                    "q_residual_film_scale",
                    "sac_actor_lr",
                    "sac_action_distribution",
                    "sac_physical_std_lr",
                    "sac_physical_std_max_kl",
                    "sac_physical_std_bound_mode",
                    "sac_physical_std_min",
                    "sac_physical_std_max",
                    "sac_physical_std_normalized_min",
                    "sac_physical_std_normalized_max",
                    "load_noise_scale",
                    "sac_initial_action_std",
                    "sac_log_std_min",
                    "sac_log_std_max",
                    "sac_alpha_init",
                    "sac_alpha_lr",
                    "sac_use_autotune",
                    "sac_alpha_update_cadence",
                    "sac_target_entropy_ratio",
                    "sac_policy_frequency",
                    "sac_learning_starts",
                    "sac_tau",
                    "sac_max_grad_norm",
                )
            }
        )
        common["sac_actor_weight_decay"] = _fastsac_actor_weight_decay(self.cfg)
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "actor_output": (
                    "raw_physical_joint_std_gaussian_with_bounded_std_and_finite_"
                    "action_safety_projection"
                    if self._uses_ppo_physical_gaussian()
                    else "smooth_tanh_bounded_normalized_std_raw_action_normal"
                ),
                "bc_loss": (
                    "detached_spred_p_weighted_joint_normalized_raw_mean_teacher_"
                    "smooth_l1"
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "joint_normalized_raw_mean_teacher_smooth_l1"
                ),
                "bc_q_filter": (
                    SPRED_P_BC_SEMANTICS
                    if getattr(self.cfg, "use_q_filtered_bc", False)
                    else "disabled"
                ),
            }
        )
        return common

    def _fastsac_checkpoint_state(self):
        # Canonicalize even lightweight/legacy optimizer constructions before
        # serializing so the saved param-group contract has stable scalar types.
        self._apply_actor_optimizer_weight_decay_contract()
        self._validate_actor_optimizer_weight_decay_contract()
        return {
            "training_algorithm": TRAINING_ALGORITHM,
            "checkpoint_version": CHECKPOINT_VERSION,
            "actor_backend": getattr(
                self, "actor_backend", _fastsac_actor_backend(self.cfg)
            ),
            "critic_learning_semantics": CRITIC_SEMANTICS,
            "actor_learning_semantics": self._actor_learning_semantics(),
            "actor_mean_optimizer_semantics": ACTOR_MEAN_OPTIMIZER_SEMANTICS,
            "actor_mean_weight_decay": _fastsac_actor_weight_decay(self.cfg),
            "entropy_semantics": self._entropy_semantics(),
            "action_distribution": _fastsac_action_distribution(self.cfg),
            "actor_adapt": self.actor_adapt.state_dict(),
            "bc_dagger_sac_adapter": self.bc_dagger_sac_adapter.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "optimizer_resume_state": {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "actor_std_optimizer": (
                    None
                    if getattr(self, "actor_std_optimizer", None) is None
                    else self.actor_std_optimizer.state_dict()
                ),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": (
                    None
                    if self.alpha_optimizer is None
                    else self.alpha_optimizer.state_dict()
                ),
                "adapt_optimizer": (
                    None if self.opt_adapt is None else self.opt_adapt.state_dict()
                ),
            },
            "actor_update_count": int(self.actor_update_count),
            "actor_std_update_count": int(
                getattr(self, "actor_std_update_count", 0)
            ),
            "critic_update_count": int(self.critic_update_count),
            "alpha_update_count": int(self.alpha_update_count),
            "q_update_row_credit": float(getattr(self, "q_update_row_credit", 0.0)),
            "dagger_rollout_count": int(self.dagger_rollout_count),
            "dagger_environment_steps": int(self.dagger_environment_steps),
            "teacher_prefill_rollout_count": int(self.teacher_prefill_rollout_count),
            "teacher_prefill_environment_steps": int(
                self.teacher_prefill_environment_steps
            ),
            "teacher_perception_warmup_complete": bool(
                getattr(self, "_teacher_perception_warmup_complete", False)
            ),
            "teacher_perception_warmup_updates": int(
                getattr(self, "_teacher_perception_warmup_updates", 0)
            ),
            "dagger_rng_state": self.dagger_rng.get_state(),
            "q_rng_state": self.q_rng.get_state(),
            "sac_action_rng_state": self.sac_action_rng.get_state(),
            "sac_rollout_rng_state": self.sac_rollout_rng.get_state(),
            "teacher_prefill_action_rng_state": (
                self.teacher_prefill_action_rng.get_state()
            ),
            "teacher_perception_rng_state": self.teacher_perception_rng.get_state(),
            "last_fastsac_diagnostics": copy.deepcopy(self._last_fastsac_diagnostics),
        }

    def _load_fastsac_checkpoint_state(self, state, *, load_modules=True):
        if state.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional FastSAC Teacher-BC checkpoint")
        if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        if state.get("actor_backend") != _fastsac_actor_backend(self.cfg):
            raise ValueError("distributional FastSAC actor backend mismatch")
        saved_distribution = state.get(
            "action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION
        )
        if saved_distribution != _fastsac_action_distribution(self.cfg):
            raise ValueError(
                "distributional FastSAC action distribution mismatch"
            )
        has_actor_decay_contract = "actor_mean_weight_decay" in state
        saved_actor_decay = state.get("actor_mean_weight_decay", 0.0)
        if (
            isinstance(saved_actor_decay, bool)
            or not isinstance(saved_actor_decay, (int, float))
            or not math.isfinite(float(saved_actor_decay))
            or float(saved_actor_decay) < 0.0
        ):
            raise ValueError("FastSAC checkpoint Actor mean weight decay is invalid")
        if not math.isclose(
            float(saved_actor_decay),
            _fastsac_actor_weight_decay(self.cfg),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("FastSAC checkpoint Actor mean weight decay mismatch")
        saved_optimizer_semantics = state.get("actor_mean_optimizer_semantics")
        if has_actor_decay_contract:
            if saved_optimizer_semantics != ACTOR_MEAN_OPTIMIZER_SEMANTICS:
                raise ValueError(
                    "FastSAC checkpoint Actor mean optimizer semantics mismatch"
                )
        elif saved_optimizer_semantics is not None:
            raise ValueError(
                "legacy FastSAC checkpoint has incomplete Actor optimizer metadata"
            )
        if load_modules:
            for name in (
                "actor_adapt",
                "bc_dagger_sac_adapter",
                "qnet",
                "qnet_target",
            ):
                getattr(self, name).load_state_dict(state[name], strict=True)
            self._restore_checkpoint_physical_actor_std(
                state["actor_adapt"], context="FastSAC checkpoint actor_adapt"
            )
            self.log_alpha.data.copy_(state["log_alpha"].to(self.log_alpha))
        optimizers = state.get("optimizer_resume_state")
        if not isinstance(optimizers, dict):
            raise ValueError("FastSAC checkpoint lacks optimizer state")
        saved_actor_optimizer = optimizers["actor_optimizer"]
        if not isinstance(saved_actor_optimizer, Mapping):
            raise ValueError("FastSAC checkpoint lacks Actor optimizer state")
        saved_actor_groups = saved_actor_optimizer.get("param_groups")
        if not isinstance(saved_actor_groups, list) or not saved_actor_groups:
            raise ValueError("FastSAC checkpoint Actor optimizer groups are invalid")
        for index, group in enumerate(saved_actor_groups):
            if not isinstance(group, Mapping):
                raise ValueError("FastSAC checkpoint Actor optimizer group is invalid")
            decay = group.get("weight_decay", 0.0)
            if (
                isinstance(decay, bool)
                or not isinstance(decay, (int, float))
                or not math.isfinite(float(decay))
                or float(decay) < 0.0
            ):
                raise ValueError(
                    "FastSAC checkpoint Actor optimizer weight decay is invalid"
                )
            if has_actor_decay_contract:
                expected_decay = float(saved_actor_decay) if index == 0 else 0.0
                if not math.isclose(
                    float(decay), expected_decay, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "FastSAC checkpoint Actor optimizer group disagrees with "
                        "its explicit weight-decay contract"
                    )
        self.actor_optimizer.load_state_dict(saved_actor_optimizer)
        self._apply_actor_optimizer_weight_decay_contract()
        self._validate_actor_optimizer_weight_decay_contract()
        actor_std_optimizer_state = optimizers.get("actor_std_optimizer")
        actor_std_optimizer = getattr(self, "actor_std_optimizer", None)
        if actor_std_optimizer is None:
            if actor_std_optimizer_state is not None:
                raise ValueError(
                    "normalized FastSAC checkpoint contains a physical std optimizer"
                )
        else:
            if not isinstance(actor_std_optimizer_state, Mapping):
                raise ValueError(
                    "physical FastSAC checkpoint lacks actor_std optimizer state"
                )
            actor_std_optimizer.load_state_dict(actor_std_optimizer_state)
            self._project_physical_actor_std_()
        self.critic_optimizer.load_state_dict(optimizers["critic_optimizer"])
        if self.alpha_optimizer is not None:
            if optimizers.get("alpha_optimizer") is None:
                raise ValueError("FastSAC checkpoint lacks alpha optimizer state")
            self.alpha_optimizer.load_state_dict(optimizers["alpha_optimizer"])
        adapt_optimizer_state = optimizers["adapt_optimizer"]
        if self.opt_adapt is None:
            if adapt_optimizer_state is not None:
                raise ValueError(
                    "frozen perception checkpoint contains an active optimizer"
                )
        else:
            if adapt_optimizer_state is None:
                raise ValueError(
                    "trainable perception checkpoint lacks optimizer state"
                )
            self.opt_adapt.load_state_dict(adapt_optimizer_state)
        for name in (
            "actor_update_count",
            "actor_std_update_count",
            "critic_update_count",
            "alpha_update_count",
            "dagger_rollout_count",
            "dagger_environment_steps",
            "teacher_prefill_rollout_count",
            "teacher_prefill_environment_steps",
        ):
            setattr(self, name, int(state[name]))
        self.q_update_row_credit = float(state.get("q_update_row_credit", 0.0))
        q_batch_size = int(getattr(self.cfg, "q_batch_size", 512))
        if not 0.0 <= self.q_update_row_credit < q_batch_size:
            raise ValueError("FastSAC checkpoint Q UTD row credit is invalid")
        self._teacher_perception_warmup_complete = bool(
            state.get("teacher_perception_warmup_complete", False)
        )
        self._teacher_perception_warmup_updates = int(
            state.get("teacher_perception_warmup_updates", 0)
        )
        self.dagger_rng.set_state(state["dagger_rng_state"])
        self.q_rng.set_state(state["q_rng_state"])
        self.sac_action_rng.set_state(state["sac_action_rng_state"])
        self.sac_rollout_rng.set_state(state["sac_rollout_rng_state"])
        teacher_prefill_rng_state = state.get("teacher_prefill_action_rng_state")
        if teacher_prefill_rng_state is None:
            self.teacher_prefill_action_rng.manual_seed(int(self.cfg.q_seed) + 4)
        else:
            self.teacher_prefill_action_rng.set_state(teacher_prefill_rng_state)
        self.teacher_perception_rng.set_state(state["teacher_perception_rng_state"])
        self._last_fastsac_diagnostics = copy.deepcopy(
            state.get("last_fastsac_diagnostics", {})
        )
        self.qnet_target.requires_grad_(False).eval()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore FastSAC model tensors without attempting replay resume."""
        if state_dict.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional FastSAC Teacher-BC checkpoint")
        checkpoint_version = int(state_dict.get("checkpoint_version", -1))
        legacy_v3 = checkpoint_version == _LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION
        if checkpoint_version not in (
            _LEGACY_EFFECTIVE_LOG_STD_CHECKPOINT_VERSION,
            _SMOOTH_BOUNDED_STD_CHECKPOINT_VERSION,
            _PREVIOUS_PRE_PRIOR_CHECKPOINT_VERSION,
            PREVIOUS_CHECKPOINT_VERSION,
            CHECKPOINT_VERSION,
        ):
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        expected_actor_backend = (
            _LEGACY_EFFECTIVE_LOG_STD_ACTOR_BACKEND
            if legacy_v3
            else _fastsac_actor_backend(self.cfg)
        )
        if state_dict.get("actor_backend") != expected_actor_backend:
            raise ValueError("distributional FastSAC actor backend mismatch")
        saved_distribution = state_dict.get(
            "action_distribution", NORMALIZED_TANH_ACTION_DISTRIBUTION
        )
        if saved_distribution != _fastsac_action_distribution(self.cfg):
            raise ValueError(
                "distributional FastSAC action distribution mismatch"
            )
        saved_action_contract = state_dict.get("action_contract")
        if not isinstance(saved_action_contract, Mapping):
            raise ValueError("FastSAC inference checkpoint lacks its action contract")
        if not isinstance(saved_action_contract.get("fingerprint"), str):
            raise ValueError(
                "FastSAC inference checkpoint lacks a contract fingerprint"
            )
        for key in ("joint_names", "fingerprint"):
            if saved_action_contract.get(key) != self._fastsac_action_contract.get(key):
                raise ValueError(
                    f"FastSAC inference checkpoint action contract mismatch at {key!r}"
                )

        load_state = state_dict
        if legacy_v3:
            source_adapter = state_dict.get("bc_dagger_sac_adapter")
            if not isinstance(source_adapter, Mapping):
                raise ValueError("FastSAC v3 inference checkpoint lacks its adapter")
            source_config = state_dict.get("dagger_backend_config")
            if not isinstance(source_config, Mapping):
                raise ValueError(
                    "FastSAC v3 inference checkpoint lacks its backend config"
                )
            legacy_log_std = source_adapter.get("log_std")
            converted_adapter = dict(source_adapter)
            converted_adapter["log_std"] = self._legacy_effective_log_std_to_raw(
                legacy_log_std,
                source_config.get("sac_log_std_min"),
                source_config.get("sac_log_std_max"),
            )
            load_state = dict(state_dict)
            load_state["bc_dagger_sac_adapter"] = converted_adapter

        failed = PPOVEL.load_state_dict(self, load_state, strict)
        critical = {
            "actor_adapt",
            "bc_dagger_sac_adapter",
            "qnet",
            "qnet_target",
            "temporal_depth_gru_ema",
            "object_adapt_ema",
            "adapt_ema",
        }
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                "FastSAC inference checkpoint failed to restore critical modules: "
                f"{sorted(missing)}"
            )
        for name in (
            "actor_adapt",
            "bc_dagger_sac_adapter",
            "qnet",
            "qnet_target",
        ):
            source = load_state.get(name)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"FastSAC inference checkpoint lacks module mapping {name!r}"
                )
            getattr(self, name).load_state_dict(source, strict=True)
        self._restore_checkpoint_physical_actor_std(
            load_state["actor_adapt"], context="FastSAC inference actor_adapt"
        )
        if self._uses_ppo_physical_gaussian():
            self._project_physical_actor_std_()
        log_alpha = load_state.get("log_alpha")
        if not torch.is_tensor(log_alpha) or log_alpha.numel() != 1:
            raise ValueError("FastSAC inference checkpoint lacks scalar log_alpha")
        self.log_alpha.data.copy_(log_alpha.to(self.log_alpha))

        initialization = state_dict.get("perception_initialization")
        if isinstance(initialization, Mapping):
            self._perception_initialization = copy.deepcopy(dict(initialization))
        self.actor_target = None
        self._teacher_prefill_complete = True
        self._teacher_perception_warmup_complete = True
        self._freeze_teacher()
        self.actor_adapt.requires_grad_(False).eval()
        self.bc_dagger_sac_adapter.requires_grad_(False).eval()
        self.qnet.requires_grad_(False).eval()
        self.qnet_target.requires_grad_(False).eval()
        for name in (
            "depth_cnn",
            "temporal_depth_gru",
            "temporal_depth_gru_ema",
            "object_adapt",
            "object_adapt_ema",
            "adapt_module",
            "adapt_ema",
        ):
            getattr(self, name).requires_grad_(False).eval()
        self._freeze_legacy_actor_std()
        return failed

    def state_dict(self):
        state = PPOVEL.state_dict(self)
        state.update(self._fastsac_checkpoint_state())
        state.update(
            {
                "dagger_backend_config": self._checkpoint_config(),
                "action_contract": copy.deepcopy(self._fastsac_action_contract),
                "q_backend_config": self._q_backend_metadata(),
                "replay_resume_semantics": (
                    "fresh_only_online_exact_actor_rings_and_teacher_episode_"
                    "sidecars_not_serialized_v2"
                ),
                "perception_replay_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                ),
                "actor_replay_observation_semantics": (
                    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
                ),
                "teacher_episode_sidecar_semantics": (
                    TEACHER_EPISODE_SIDECAR_SEMANTICS
                ),
                "perception_training_semantics": (
                    ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS
                ),
                "perception_prefill_warmup_semantics": (
                    PERCEPTION_PREFILL_DISABLED_SEMANTICS
                ),
                "teacher_prefill_semantics": FASTSAC_TEACHER_PREFILL_SEMANTICS,
                "object_geo_replay_semantics": OBJECT_GEO_REPLAY_SEMANTICS,
                "perception_initialization": copy.deepcopy(
                    self._perception_initialization
                ),
                "perception_object_geo_fingerprint": (
                    self._replay_object_geo_fingerprint
                ),
                "vecnorm_fingerprint": self._replay_vecnorm_fingerprint,
                "next_iter": int(self.env.current_iter) + 1,
            }
        )
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        if algorithm == TRAINING_ALGORITHM:
            raise ValueError(
                "FastSAC Actor replay and Teacher episode sidecars are fresh-only; "
                "same-stage resume "
                "is intentionally unsupported"
            )
        if algorithm is not None:
            raise ValueError(
                f"unsupported FastSAC Teacher-BC source algorithm={algorithm!r}"
            )

        # Reuse the rigorously validated raw PPO source loading in the replay
        # base. It transiently creates a TD3 target Actor, which is discarded
        # immediately and never participates in FastSAC behavior or learning.
        # PPOVEL's compatibility loader iterates every currently registered
        # child.  The dedicated SAC variance adapter cannot exist in an older
        # PPO source, so seed only that new child with its fresh initialization
        # while leaving every source-owned module subject to strict loading.
        padded_source = dict(state_dict)
        padded_source["bc_dagger_sac_adapter"] = self.bc_dagger_sac_adapter.state_dict()
        failed = DistributionalTD3TeacherBC.load_state_dict(self, padded_source, strict)
        self.actor_target = None
        self.bc_dagger_sac_adapter.log_std.data.copy_(self._fastsac_initial_raw_log_std)
        self.log_alpha.data.fill_(math.log(float(self.cfg.sac_alpha_init)))
        self.actor_update_count = 0
        self.actor_std_update_count = 0
        self.critic_update_count = 0
        self.alpha_update_count = 0
        self.q_update_row_credit = 0.0
        self.dagger_rollout_count = 0
        self.dagger_environment_steps = 0
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self.dagger_rng.manual_seed(int(self.cfg.dagger_seed))
        self.q_rng.manual_seed(int(self.cfg.q_seed))
        self.sac_action_rng.manual_seed(int(self.cfg.q_seed) + 1)
        self.sac_rollout_rng.manual_seed(int(self.cfg.q_seed) + 2)
        self.teacher_prefill_action_rng.manual_seed(int(self.cfg.q_seed) + 4)
        self.teacher_perception_rng.manual_seed(int(self.cfg.q_seed) + 3)
        # Drop deterministic-only streams after the base loader is finished.
        if hasattr(self, "collector_exploration_rng"):
            del self.collector_exploration_rng
        if hasattr(self, "target_policy_rng"):
            del self.target_policy_rng
        hard_copy_(self.qnet, self.qnet_target)
        self.qnet_target.requires_grad_(False).eval()
        self.actor_adapt.requires_grad_(True).train()
        self._configure_training_actor_std()
        if self._uses_ppo_physical_gaussian():
            expected_std = float(self.cfg.load_noise_scale)
            actual_std = self._ppo_actor_std_parameter().detach()
            if not torch.equal(
                actual_std, torch.full_like(actual_std, expected_std)
            ):
                raise RuntimeError(
                    "PPOVEL fresh finetune did not initialize every physical "
                    "actor_std from load_noise_scale"
                )
            # In q-normalized mode the scalar PPO-compatible reset is only the
            # fresh source value.  Resolve it through the joint-wise envelope
            # before the first Student rollout; same-stage resume never enters
            # this source-loading branch and therefore restores its saved std.
            self._project_physical_actor_std_()
        return failed


__all__ = [
    "ACTION_CONTRACT_SEMANTICS",
    "ACTOR_BACKEND",
    "ACTOR_MEAN_OPTIMIZER_SEMANTICS",
    "CHECKPOINT_VERSION",
    "FASTSAC_ACTION_PROJECTION_KEY",
    "FASTSAC_PREFILL_TEACHER_NOISE_KEY",
    "FASTSAC_PREFILL_TEACHER_PROJECTION_KEY",
    "FastSACPhysicalNormal",
    "NORMALIZED_TANH_ACTION_DISTRIBUTION",
    "PHYSICAL_STD_BOUND_MODES",
    "PPO_PHYSICAL_GAUSSIAN_ACTION_DISTRIBUTION",
    "PPO_PHYSICAL_GAUSSIAN_ACTOR_BACKEND",
    "Q_NORMALIZED_PHYSICAL_STD_BOUND_MODE",
    "SPRED_P_BC_SEMANTICS",
    "UNIFORM_PHYSICAL_STD_BOUND_MODE",
    "PREVIOUS_CHECKPOINT_VERSION",
    "TRAINING_ALGORITHM",
    "DistributionalFastSACTeacherBC",
    "DistributionalFastSACTeacherBCConfig",
    "_DeterministicFastSACStudentEvalPolicy",
    "_DistributionalFastSACDaggerRolloutPolicy",
    "_spred_p_teacher_probability",
]
