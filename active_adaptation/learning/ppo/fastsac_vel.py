"""VAIC teacher/student training with distributional FastSAC.

Both stages use FastSAC actor/Q optimization. VAIC continues to own the
observations, rewards, terminations, depth perception, and student adaptation.
PPOVEL is used only to construct those unchanged VAIC modules; this module has
no PPO optimization or PPO rollout-policy path.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import logging
import math
import os
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal

from .common import (
    ACTION_KEY,
    Actor,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    hard_copy_,
)
from .ppo_vel import (
    DEPTH_KEY,
    HEIGHT_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    PPOConfig,
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    ZeroDepthInjector,
)


TEACHER_REPLAY_FORMAT_VERSION = 11
TRUNCATION_NEXT_OBSERVATION_SEMANTICS = (
    "episode_time_limit_pre_reset_final_command_finished_terminal_v3"
)
REPLAY_OBSERVATION_SEMANTICS = "raw_pre_vecnorm_sample_current_v1"
SAC_REWARD_SCALARIZATION = "sum_reward_groups_v1"
TEACHER_REPLAY_INITIAL_TRANSITION_FILTER = "step_count_gt_1"
TEACHER_REPLAY_MIN_STEP_COUNT = 1
STUDENT_REPLAY_MIN_STEP_COUNT = 5
FASTSAC_ACTOR_BACKEND = "vaic_fastsac_tanh_gaussian_v2"
BC_DAGGER_TRAINING_ALGORITHM = "vaic_ppo_bc_dagger_student_iql_v2"
BC_DAGGER_LEGACY_TRAINING_ALGORITHM = "vaic_ppo_bc_dagger_student_v1"
BC_DAGGER_ACTOR_BACKEND = "vaic_ppo_independent_normal_bc_dagger_v1"
BC_DAGGER_IQL_CRITIC_SEMANTICS = (
    "dataset_action_target_twin_expected_c51_expectile_v_to_scalar_td_"
    "c51_projection_v1"
)
BC_DAGGER_ACTOR_LEARNING_SEMANTICS = (
    "dagger_teacher_huber_bc_only_no_q_or_advantage_weighting_v1"
)
BC_DAGGER_REPLAY_FORMAT = "vaic_ppo_bc_dagger_teacher_buffer"
BC_DAGGER_REPLAY_FORMAT_VERSION = 2
BC_DAGGER_LEGACY_REPLAY_FORMAT_VERSION = 1
BC_DAGGER_LEGACY_REPLAY_OBSERVATION_SEMANTICS = (
    "normalized_frozen_vecnorm_v1"
)
BC_DAGGER_REPLAY_OBSERVATION_SEMANTICS = REPLAY_OBSERVATION_SEMANTICS
FASTSAC_BC_DAGGER_ACTOR_BACKEND = "vaic_fastsac_bc_dagger_adapter_v2"
FASTSAC_BC_DAGGER_LEGACY_ACTOR_BACKEND = (
    "vaic_fastsac_bc_dagger_adapter_v1"
)
PRE_NORMALIZED_REPLAY_KEY = "_fastsac_pre_normalized_replay"
# Ephemeral minibatch provenance only. This marker is constructed after replay
# sampling, never stored in either replay, and is intentionally independent of
# whether a legacy H5 row was already VecNorm-normalized.
STAGE2_OFFLINE_SOURCE_KEY = "_fastsac_stage2_offline_source"
STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY = (
    "_fastsac_stage2_behavior_mean_abs_deviation"
)
STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY = (
    "_fastsac_stage2_behavior_max_abs_deviation"
)
FASTSAC_ACTION_PARAMETERIZATION = (
    "teacher_reference_centered_student_absolute_asymmetric_v2"
)
FASTSAC_TARGET_ENTROPY_SEMANTICS = (
    "normalized_tanh_action_log_prob_target_v3"
)
FASTSAC_BC_DAGGER_TARGET_ENTROPY_SEMANTICS = (
    "fixed_raw_action_reference_scale_log_prob_target_v1"
)
FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS = (
    "affine_executable_to_unit_box_then_fixed_gain_v2"
)
FASTSAC_Q_RAW_ACTION_SEMANTICS = (
    "executable_action_coordinates_then_fixed_gain_v2"
)
FASTSAC_Q_REFERENCE_RESIDUAL_SEMANTICS = (
    "executable_action_minus_frame_reference_divide_half_range_no_clamp_then_gain_v1"
)
FASTSAC_Q_ACTION_COORDINATES = ("absolute", "reference_residual")
FASTSAC_Q_ACTION_FUSIONS = ("early", "late")
FASTSAC_Q_EARLY_FUSION_SEMANTICS = "input_concat_then_shared_trunk_v1"
FASTSAC_Q_LATE_FUSION_SEMANTICS = (
    "separate_obs_and_action_stems_then_shared_trunk_v1"
)
FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS = (
    "monolithic_action_conditioned_c51_logits_v1"
)
FASTSAC_Q_REFERENCE_DUELING_ARCHITECTURE_SEMANTICS = (
    "state_value_plus_action_advantage_minus_fixed_reference_advantage_c51_logits_v1"
)
FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS = (
    "delay_one_hot_min_to_max_plus_fixed_range_centered_alpha_q_only_v1"
)
FASTSAC_DETERMINISTIC_ACTION_KEY = "_fastsac_deterministic_action"
FASTSAC_REFERENCE_EPS = 1e-6
FASTSAC_TEACHER_TRAINING_ALGORITHM = "vaic_fastsac_teacher_v13"
FASTSAC_STUDENT_TRAINING_ALGORITHM = "vaic_fastsac_student_rlpd_v13"
FASTSAC_STAGE1_UPDATE_MODE = {
    "version": 1,
    "mode": "pure_fastsac",
}
FASTSAC_STAGE1_N_STEP_RETURN_SEMANTICS = (
    "collector_cumulative_gamma_env_discount_done_partial_flush_v1"
)
FASTSAC_STAGE1_CONSERVATIVE_Q_SEMANTICS = (
    "deterministic_policy_vs_replay_expected_c51_softplus_margin_v1"
)
FASTSAC_STAGE1_ACTOR_OBJECTIVES = ("sac", "reference_awac")
FASTSAC_STAGE1_REFERENCE_AWAC_SEMANTICS = (
    "target_twin_replay_minus_reference_pessimistic_normalized_exp_"
    "detached_scale_v1"
)
FASTSAC_STAGE1_BEHAVIOR_UNCERTAINTY_GATE_SEMANTICS = (
    "online_twin_sampled_policy_minus_same_row_replay_mean_improvement_"
    "greater_than_head_disagreement_q_only_entropy_ungated_v1"
)
FASTSAC_STAGE2_ACTOR_CONFIDENCE_GATE_SEMANTICS = (
    "sampled_current_policy_pessimistic_q_minus_frozen_bc_pessimistic_q_"
    "gain_gt_max_raw_actor_or_bc_twin_disagreement_row_mask_"
    "minimum_batch_coverage_nonsticky_full_batch_denominator_v2"
)
FASTSAC_CLIPPED_DOUBLE_Q_SEMANTICS = (
    "lower_expected_projected_c51_head_common_twin_target_v1"
)
FASTSAC_INDEPENDENT_DOUBLE_Q_SEMANTICS = (
    "independent_projected_c51_head_targets_v1"
)
FASTSAC_RAW_OBSERVATION_ROOT = "_fastsac_raw"


def _vecnorm_state_fingerprint(vecnorm) -> str:
    """Hash the exact fixed VecNorm coordinates used by replay consumers.

    Raw replay is independent of normalization while stored, but transferred
    actor/Q weights are not.  Pairing the H5 with this digest prevents a raw
    dataset from being silently interpreted in a different observation frame.
    """
    digest = hashlib.sha256()
    digest.update(b"vaic_vecnorm_loc_scale_v1\0")
    in_keys = list(vecnorm.in_keys)
    out_keys = list(getattr(vecnorm, "out_keys", in_keys))
    if len(in_keys) != len(out_keys):
        raise ValueError("VecNorm input/output key counts do not match")
    digest.update(repr(float(vecnorm.eps)).encode("ascii"))
    digest.update(b"\0")
    for in_key, out_key in zip(in_keys, out_keys):
        digest.update(
            json.dumps(
                {"in": in_key, "out": out_key},
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
        for label, value in (
            ("loc", vecnorm.loc[in_key]),
            ("scale", vecnorm.scale[out_key]),
        ):
            tensor = value.detach().to("cpu").contiguous()
            digest.update(label.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(
                np.asarray(tuple(tensor.shape), dtype=np.int64).tobytes()
            )
            digest.update(tensor.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"
TEACHER_REF_ACTION_FIELD = "teacher_ref_action"
NEXT_TEACHER_REF_ACTION_FIELD = "next_teacher_ref_action"
TEACHER_OBJECT_GEO_FIELD = "teacher_object_geo"
TEACHER_HEIGHT_FIELD = "teacher_height"
NEXT_TEACHER_HEIGHT_FIELD = "next_teacher_height"
TEACHER_ACTUATOR_CONTEXT_FIELD = "teacher_actuator_context"
NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD = "next_teacher_actuator_context"
TEACHER_REPLAY_FIELDS = (
    "observations", "critic_observations", "actions", "rewards", "dones",
    "truncations", "discounts", "next_observations", "next_critic_observations",
)
TEACHER_TRAINING_REPLAY_FIELDS = (
    "critic_observations",
    "actions",
    "rewards",
    "dones",
    "truncations",
    "discounts",
    "effective_n_steps",
    "next_critic_observations",
)


def _normalize_q_actuator_context_metadata(metadata=None) -> dict:
    """Return a strict, JSON-safe Q-only actuator-context descriptor."""
    if metadata is None:
        return {"enabled": False}
    if not isinstance(metadata, dict):
        raise ValueError("Q actuator-context metadata must be a dictionary")
    enabled = metadata.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Q actuator-context metadata enabled must be boolean")
    if not enabled:
        if set(metadata).difference({"enabled"}):
            raise ValueError(
                "Disabled Q actuator-context metadata must contain only enabled=false"
            )
        return {"enabled": False}

    required = {
        "enabled",
        "semantics",
        "dimension",
        "delay_range",
        "alpha_range",
    }
    if set(metadata) != required:
        raise ValueError(
            "Enabled Q actuator-context metadata fields do not match the current "
            f"schema: got {sorted(metadata)}, expected {sorted(required)}"
        )
    semantics = str(metadata["semantics"])
    if semantics != FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS:
        raise ValueError(
            f"Unsupported Q actuator-context semantics {semantics!r}"
        )
    delay_range = metadata["delay_range"]
    if not isinstance(delay_range, (list, tuple)) or len(delay_range) != 2:
        raise ValueError("Q actuator-context delay_range must contain two integers")
    if any(isinstance(value, bool) for value in delay_range):
        raise ValueError("Q actuator-context delay bounds must be integers")
    delay_min, delay_max = (int(value) for value in delay_range)
    if any(int(value) != value for value in delay_range) or delay_min > delay_max:
        raise ValueError("Q actuator-context delay_range is invalid")

    alpha_range = metadata["alpha_range"]
    if not isinstance(alpha_range, (list, tuple)) or len(alpha_range) != 2:
        raise ValueError("Q actuator-context alpha_range must contain two values")
    alpha_low, alpha_high = (float(value) for value in alpha_range)
    if (
        not math.isfinite(alpha_low)
        or not math.isfinite(alpha_high)
        or alpha_low > alpha_high
    ):
        raise ValueError("Q actuator-context alpha_range is invalid")

    dimension = metadata["dimension"]
    if isinstance(dimension, bool) or int(dimension) != dimension:
        raise ValueError("Q actuator-context dimension must be an integer")
    dimension = int(dimension)
    expected_dimension = delay_max - delay_min + 2
    if dimension != expected_dimension:
        raise ValueError(
            "Q actuator-context dimension does not match delay one-hot plus alpha: "
            f"got {dimension}, expected {expected_dimension}"
        )
    return {
        "enabled": True,
        "semantics": semantics,
        "dimension": dimension,
        "delay_range": [delay_min, delay_max],
        "alpha_range": [alpha_low, alpha_high],
    }


def _q_action_hidden_dim(hidden_dim: int, action_fusion: str) -> int:
    """Return the late action-stem width, or zero for exact early fusion.

    The late stem uses one sixth of the observation-stem width.  This yields
    128 action features for the configured 768-wide Skateboard critic while
    keeping the ratio stable for smaller test/ablation networks.  The floor of
    two avoids a one-feature LayerNorm, which would erase the action signal.
    """
    if action_fusion not in FASTSAC_Q_ACTION_FUSIONS:
        raise ValueError(
            f"q_action_fusion must be one of {FASTSAC_Q_ACTION_FUSIONS}, "
            f"got {action_fusion!r}"
        )
    if action_fusion == "early":
        return 0
    return max(2, int(hidden_dim) // 6)


def _resolve_teacher_training_replay_device(configured, policy_device):
    """Resolve the ephemeral Stage-1 replay storage device.

    ``policy`` preserves the historical device-local behavior.  Explicit CPU
    storage is useful for long replay horizons, while explicit CUDA devices
    remain available for multi-GPU setups.  Only CPU/CUDA storage is supported
    because every minibatch must ultimately be consumed by the policy device.
    """
    policy_device = torch.device(policy_device)
    if policy_device.type not in ("cpu", "cuda"):
        raise ValueError(
            "FastSAC policy device must be a CPU or CUDA device, got "
            f"{policy_device}."
        )
    if not isinstance(configured, str):
        raise ValueError(
            "teacher_training_replay_device must be 'policy', 'cpu', "
            "'cuda', or 'cuda:<index>'"
        )
    if configured != configured.strip() or not configured:
        raise ValueError(
            "teacher_training_replay_device must not be empty or contain "
            "surrounding whitespace"
        )
    if configured == "policy":
        return policy_device
    try:
        replay_device = torch.device(configured)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "teacher_training_replay_device must be 'policy', 'cpu', "
            "'cuda', or 'cuda:<index>'"
        ) from exc
    if replay_device.type not in ("cpu", "cuda"):
        raise ValueError(
            "teacher_training_replay_device supports only CPU or CUDA "
            f"storage, got {configured!r}"
        )
    if replay_device.type == "cpu" and replay_device.index is not None:
        raise ValueError(
            "teacher_training_replay_device='cpu' must not include an index"
        )
    # An unindexed CUDA setting should follow the already selected policy GPU
    # instead of relying on a possibly different process-global current device.
    if replay_device.type == "cuda" and replay_device.index is None:
        if policy_device.type == "cuda":
            return policy_device
    return replay_device


def _validate_seed_replay_partition(
    storage_ratio: float,
    sample_ratio: float,
    capacity: int | None = None,
) -> tuple[float, float, int | None]:
    """Validate the optional frozen Stage-1 reference-data partition."""
    storage_ratio = float(storage_ratio)
    sample_ratio = float(sample_ratio)
    if not math.isfinite(storage_ratio) or not 0.0 <= storage_ratio < 1.0:
        raise ValueError(
            "sac_teacher_seed_storage_ratio must be finite and in [0, 1)"
        )
    if not math.isfinite(sample_ratio) or not 0.0 <= sample_ratio <= 1.0:
        raise ValueError(
            "sac_teacher_seed_sample_ratio must be finite and in [0, 1]"
        )
    if (storage_ratio == 0.0) != (sample_ratio == 0.0):
        raise ValueError(
            "sac_teacher_seed_storage_ratio and "
            "sac_teacher_seed_sample_ratio must either both be zero or both "
            "be positive"
        )
    seed_capacity = None
    if capacity is not None:
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("teacher training replay capacity must be positive")
        seed_capacity = round(capacity * storage_ratio)
        if storage_ratio > 0.0 and not 0 < seed_capacity < capacity:
            raise ValueError(
                "sac_teacher_seed_storage_ratio must reserve at least one seed "
                "and one online replay row"
            )
    return storage_ratio, sample_ratio, seed_capacity


def _validate_pure_fastsac_checkpoint_provenance(state_dict) -> None:
    """Reject checkpoints that used the removed Stage-1 PPO training paths."""
    update_mode = state_dict.get("stage1_update_mode")
    if update_mode is not None:
        if update_mode != FASTSAC_STAGE1_UPDATE_MODE:
            raise ValueError(
                "Checkpoint Stage-1 update mode is not pure FastSAC: "
                f"{update_mode!r}"
            )

    # Older pure-FastSAC checkpoints either predate these optional paths or
    # record their exact disabled sentinels. Positive or malformed metadata
    # means PPO contributed to the saved weights and must never be silently
    # resumed or transferred as pure FastSAC.
    legacy_disabled = {
        "stage1_ppo_warmup_config": {"rollouts": 0},
        "stage1_ppo_behavior_distill_config": {"rollouts": 0},
        "stage1_distilled_ppo_warmup_config": {"end_rollout": 0},
    }
    for key, disabled in legacy_disabled.items():
        actual = state_dict.get(key)
        if actual is not None and actual != disabled:
            raise ValueError(
                "PPO-assisted Stage-1 checkpoint cannot be loaded by pure "
                f"FastSAC ({key}={actual!r})."
            )


def _select_c51_twin_target(
    target: torch.Tensor,
    support: torch.Tensor,
    clipped_double_q: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the configured twin C51 target and its expected values.

    Distributional clipped double-Q cannot take an elementwise minimum of atom
    probabilities.  Instead, choose the target head with the lower expected
    return for each sample, retain that head's complete probability
    distribution, and train both online heads against the common distribution.
    """
    if target.ndim != 3 or target.shape[0] != 2:
        raise ValueError("FastSAC C51 target must have shape [2, batch, atoms]")
    if support.ndim != 1 or support.shape[0] != target.shape[-1]:
        raise ValueError("FastSAC C51 support does not match target atoms")
    values = (target * support).sum(dim=-1)
    if not clipped_double_q:
        return target, values
    lower_head = values.argmin(dim=0)
    selected = target.gather(
        0,
        lower_head[None, :, None].expand(1, target.shape[1], target.shape[2]),
    ).squeeze(0)
    common_target = selected.unsqueeze(0).expand_as(target)
    selected_values = values.gather(0, lower_head.unsqueeze(0))
    return common_target, selected_values.expand_as(values)


def _reduce_actor_q_values(
    q_values: torch.Tensor, clipped_double_q: bool
) -> torch.Tensor:
    """Use pessimistic clipped double-Q or the legacy HOI twin mean."""
    if q_values.ndim < 1 or q_values.shape[0] != 2:
        raise ValueError("FastSAC actor Q values must have two leading heads")
    if clipped_double_q:
        return q_values.min(dim=0).values
    return q_values.mean(dim=0)


def _policy_replay_conservative_q_penalty(
    policy_q_values: torch.Tensor,
    replay_q_values: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth one-sided policy-vs-replay penalty per Q head.

    Both inputs are expected C51 values, not logits.  The replay values remain
    attached so the regularizer learns their *relative* ranking, while the
    Bellman cross-entropy continues to anchor replay actions to observed data.
    Actor actions are detached by the caller; this helper therefore changes Q
    learning only and leaves the SAC actor/entropy objective untouched.
    """
    if policy_q_values.shape != replay_q_values.shape:
        raise ValueError(
            "FastSAC conservative Q policy/replay values must have matching "
            f"shapes, got {tuple(policy_q_values.shape)} and "
            f"{tuple(replay_q_values.shape)}"
        )
    if policy_q_values.ndim < 2:
        raise ValueError(
            "FastSAC conservative Q values must contain head and batch dimensions"
        )
    margin = float(margin)
    temperature = float(temperature)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("FastSAC conservative Q margin must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("FastSAC conservative Q temperature must be finite and positive")

    gap = policy_q_values - replay_q_values
    penalty = temperature * F.softplus((gap - margin) / temperature)
    reduce_dims = tuple(range(1, penalty.ndim))
    return penalty.mean(dim=reduce_dims), gap


def _reference_awac_weights(
    replay_q_values: torch.Tensor,
    reference_q_values: torch.Tensor,
    beta: float,
    weight_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized in-support AWAC weights and pessimistic advantages.

    Subtract within each target-Q head before taking the pessimistic minimum.
    This cancels head-specific state-value calibration without allowing a head
    switch between the replay and reference actions to manufacture an
    advantage.  The caller evaluates both actions without gradients.
    """
    if replay_q_values.shape != reference_q_values.shape:
        raise ValueError(
            "Reference AWAC replay/reference Q values must have matching "
            f"shapes, got {tuple(replay_q_values.shape)} and "
            f"{tuple(reference_q_values.shape)}"
        )
    if replay_q_values.ndim < 2 or replay_q_values.shape[0] != 2:
        raise ValueError(
            "Reference AWAC Q values must have two leading target-Q heads"
        )
    beta = float(beta)
    weight_clip = float(weight_clip)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Reference AWAC beta must be finite and positive")
    if not math.isfinite(weight_clip) or weight_clip < 1.0:
        raise ValueError(
            "Reference AWAC weight clip must be finite and at least one"
        )

    relative_advantages = replay_q_values - reference_q_values
    advantages = relative_advantages.min(dim=0).values
    log_weights = (advantages / beta).clamp(
        min=-5.0,
        max=math.log(weight_clip),
    )
    unnormalized = log_weights.exp()
    denominator = unnormalized.mean().clamp_min(
        torch.finfo(unnormalized.dtype).tiny
    )
    return unnormalized / denominator, advantages


class FastSACTanhNormal(TanhNormal):
    """FastSAC's reparameterized tanh Gaussian with TensorDict helpers."""

    dist_keys = ["loc", "scale"]

    @property
    def mean(self):
        # TanhNormal has no analytic expectation. HOI uses tanh(mu) for
        # deterministic inference, which is also the quantity distilled by VAIC.
        return self.deterministic_sample

    @property
    def mode(self):
        return self.deterministic_sample

    def entropy(self):
        # The squashed distribution has no analytic entropy. FastSAC itself
        # uses the transformed log probability directly.
        _, log_prob = self.rsample_with_log_prob()
        return -log_prob

    def rsample_with_log_prob(
        self, sample_shape=torch.Size(), generator: torch.Generator | None = None
    ):
        """Sample and score from the retained pre-tanh variable.

        ``TransformedDistribution.log_prob(action)`` reconstructs the latent
        with inverse tanh.  Once a float32 action saturates at a bound that
        inverse is numerically catastrophic.  FastSAC already samples the
        latent, so use the analytic tanh Jacobian directly instead.
        """
        normal = torch.distributions.Normal(self.loc, self.scale)
        if generator is None:
            raw_action = normal.rsample(sample_shape)
        else:
            # SAC learning noise must not advance the global generator used by
            # rollout actions and environment randomization.  Keep the sample
            # reparameterized so actor gradients are unchanged.
            noise = torch.randn(
                (*sample_shape, *self.loc.shape),
                dtype=self.loc.dtype,
                device=self.loc.device,
                generator=generator,
            )
            raw_action = self.loc + self.scale * noise
        tanh_action = torch.tanh(raw_action)
        action_scale = (self.high - self.low) * 0.5
        action_bias = (self.high + self.low) * 0.5
        action = tanh_action * action_scale + action_bias

        # Stable equivalent of log(1 - tanh(raw_action) ** 2).  Unlike the
        # literal expression, this keeps a useful gradient at saturated logits.
        log_tanh_jacobian = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )
        per_dim_log_prob = (
            normal.log_prob(raw_action)
            - log_tanh_jacobian
            - torch.log(action_scale)
        )
        if self._event_dims:
            event_axes = tuple(range(
                per_dim_log_prob.ndim - self._event_dims,
                per_dim_log_prob.ndim,
            ))
            log_prob = per_dim_log_prob.sum(dim=event_axes)
        else:
            log_prob = per_dim_log_prob
        return action, log_prob

    def log_prob_for_action(
        self,
        action: torch.Tensor,
        *,
        detach_scale: bool = False,
    ) -> torch.Tensor:
        """Score a bounded action through a stable inverse-tanh transform.

        Replay can contain an action rounded exactly onto an executable bound.
        The generic transformed-distribution inverse then produces an infinite
        latent.  Clamp only the normalized inverse coordinate and reuse the
        analytic tanh Jacobian from :meth:`rsample_with_log_prob`.

        ``detach_scale`` implements the fixed-scale Stage-1 AWAC regression:
        the likelihood still has its configured Gaussian weighting, while no
        gradient can update the log-standard-deviation head or reach shared
        features through that head.
        """
        action_scale = (self.high - self.low) * 0.5
        action_bias = (self.high + self.low) * 0.5
        normalized_action = ((action - action_bias) / action_scale).clamp(
            min=-1.0 + FASTSAC_REFERENCE_EPS,
            max=1.0 - FASTSAC_REFERENCE_EPS,
        )
        raw_action = torch.atanh(normalized_action)
        normal_scale = self.scale.detach() if detach_scale else self.scale
        normal = torch.distributions.Normal(self.loc, normal_scale)
        log_tanh_jacobian = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )
        per_dim_log_prob = (
            normal.log_prob(raw_action)
            - log_tanh_jacobian
            - torch.log(action_scale)
        )
        if self._event_dims:
            event_axes = tuple(range(
                per_dim_log_prob.ndim - self._event_dims,
                per_dim_log_prob.ndim,
            ))
            return per_dim_log_prob.sum(dim=event_axes)
        return per_dim_log_prob


def _fastsac_action_center_to_latent(
    center_action, action_scale, action_bias, reference_eps
):
    """Invert the shared asymmetric tanh/affine action transform."""
    center_normalized = (center_action - action_bias) / action_scale
    center_normalized = center_normalized.clamp(
        min=-1.0 + reference_eps,
        max=1.0 - reference_eps,
    )
    return torch.atanh(center_normalized)


def _fastsac_target_entropy(
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    target_entropy_ratio: float,
) -> float:
    """Return a coordinate-invariant normalized-action entropy target.

    SAC objectives convert the distribution's physical-action log probability
    back to the normalized tanh coordinate by adding the constant affine log
    determinant. The corresponding target is therefore ``-dim * ratio`` and
    is invariant to joint units or the chosen executable action support.
    """
    action_low = torch.as_tensor(action_low, dtype=torch.float64)
    action_high = torch.as_tensor(action_high, dtype=torch.float64)
    if action_low.ndim != 1 or action_high.shape != action_low.shape:
        raise ValueError("FastSAC entropy bounds must be matching vectors")
    action_scale = (action_high - action_low) * 0.5
    if not torch.isfinite(action_scale).all() or not torch.all(action_scale > 0):
        raise ValueError("FastSAC entropy action scales must be finite and positive")
    target_entropy_ratio = float(target_entropy_ratio)
    if not math.isfinite(target_entropy_ratio) or target_entropy_ratio < 0.0:
        raise ValueError(
            "FastSAC target_entropy_ratio must be finite and non-negative"
        )
    return float(-action_low.numel() * target_entropy_ratio)


def _measure_or_clip_grad_norm(parameters, max_norm: float) -> torch.Tensor:
    """Measure gradients without mutating them when clipping is disabled."""
    parameters = tuple(parameters)
    with_grad = tuple(
        parameter for parameter in parameters if parameter.grad is not None
    )
    if not with_grad:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros((), device=device)
    if float(max_norm) > 0.0:
        return nn.utils.clip_grad_norm_(with_grad, float(max_norm))
    per_parameter = torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in with_grad
    ])
    return torch.linalg.vector_norm(per_parameter)


class FastSACActor(nn.Module):
    """FastSAC actor network in VAIC's final absolute action coordinates.

    The deployed student predicts a correction around VAIC raw action zero (the
    default pose).  The motion teacher predicts a correction around the current
    reference action.  Both centers are applied *before* the bijective tanh/affine
    transform so sampled actions, deterministic actions, and log probabilities
    all describe the same bounded final action variable (unlike adding/clamping a
    residual after sampling).
    """

    def __init__(
        self, input_dim, action_dim, hidden_dim, log_std_min, log_std_max,
        action_low, action_high, layer_norm=True, reference_centered=False,
        reference_eps=FASTSAC_REFERENCE_EPS,
    ):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("fastsac_actor_hidden_dim must be at least 4")
        if not log_std_min < log_std_max:
            raise ValueError("fastsac_log_std_min must be below fastsac_log_std_max")
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("FastSAC action bounds must match the VAIC action dimension")
        if not torch.all(action_high > action_low):
            raise ValueError("FastSAC action upper bounds must exceed lower bounds")
        if not 0.0 < float(reference_eps) < 1.0:
            raise ValueError("FastSAC reference_eps must be in (0, 1)")

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.reference_centered = bool(reference_centered)
        self.reference_eps = float(reference_eps)
        mid_dim = hidden_dim // 2
        last_dim = hidden_dim // 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(mid_dim, last_dim),
            nn.LayerNorm(last_dim) if layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(last_dim, action_dim))
        self.fc_logstd = nn.Linear(last_dim, action_dim)

        # Match HOI: the two output heads start at mean=0 and log_std midpoint.
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        self.register_buffer("action_scale", (action_high - action_low) * 0.5)
        self.register_buffer("action_bias", (action_high + action_low) * 0.5)

    @torch.no_grad()
    def reset_log_std_head(self, target_log_std: float) -> float:
        """Make every state emit one exact log standard deviation.

        ``fc_logstd`` is passed through ``tanh`` before the configured log-std
        affine transform, so the requested value must be inverse-mapped rather
        than copied directly into the bias.
        """
        target_log_std = float(target_log_std)
        if (
            not math.isfinite(target_log_std)
            or not self.log_std_min < target_log_std < self.log_std_max
        ):
            raise ValueError(
                "target_log_std must be finite and strictly inside the "
                "FastSAC log-std bounds"
            )
        raw_target = (
            2.0
            * (target_log_std - self.log_std_min)
            / (self.log_std_max - self.log_std_min)
            - 1.0
        )
        raw_bias = math.atanh(raw_target)
        self.fc_logstd.weight.zero_()
        self.fc_logstd.bias.fill_(raw_bias)
        return raw_bias

    def forward(self, observations, reference_action=None):
        features = self.net(observations)
        residual_loc = self.fc_mu(features)
        if self.reference_centered:
            if reference_action is None:
                raise ValueError(
                    "Reference-centered FastSAC teacher requires reference_action"
                )
            if reference_action.shape[-1] != self.action_scale.shape[-1]:
                raise ValueError(
                    "FastSAC reference action dimension does not match actor output"
                )
            center_action = reference_action
        else:
            # Asymmetric bounds have a nonzero interval midpoint.  Explicitly
            # center the student at raw action zero so its zero-initialized head
            # retains original VAIC's safe default-pose behavior.
            center_action = torch.zeros_like(residual_loc)
        # Invert the exact asymmetric tanh/affine transform, then add the learned
        # correction in latent space.  For the teacher, a zero-initialized mean
        # head therefore reproduces the executable reference pose.
        loc = _fastsac_action_center_to_latent(
            center_action,
            self.action_scale,
            self.action_bias,
            self.reference_eps,
        )
        loc = loc + residual_loc
        raw_log_std = torch.tanh(self.fc_logstd(features))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (raw_log_std + 1.0)
        scale = log_std.exp()
        deterministic_action = torch.tanh(loc) * self.action_scale + self.action_bias
        return loc, scale, deterministic_action


def _sac_bootstrap_mask(dones: torch.Tensor, truncations: torch.Tensor) -> torch.Tensor:
    """Bootstrap ordinary transitions and VAIC truncations, not terminals."""
    return (truncations.bool() | ~dones.bool()).float()


def _vaic_truncation_mask(td: TensorDict) -> torch.Tensor:
    """Return only SAC's bootstrapping timeout rows.

    The environment flags remain unchanged.  FastSAC interprets an episode
    time limit as a truncation and bootstraps from its real pre-reset final
    observation.  Command/motion completion is instead a task terminal, as in
    HOI WBT, so it ends an n-step return without bootstrap.  Command completion
    and physical termination both win if causes happen simultaneously.
    """
    required = (
        ("next", "stats", "episode_time_limit"),
        ("next", "stats", "command_finished"),
    )
    missing = [key for key in required if key not in td.keys(True, True)]
    if missing:
        raise KeyError(
            "FastSAC requires explicit VAIC termination-cause stats; missing "
            f"{missing}."
        )
    episode_time_limit = td[required[0]].bool()
    command_finished = td[required[1]].bool()
    terminated = td[TERM_KEY].bool()
    reset_cause = episode_time_limit | command_finished
    if (reset_cause & ~td[DONE_KEY].bool()).any():
        raise RuntimeError("A VAIC FastSAC reset-cause row is not marked done")
    return episode_time_limit & ~command_finished & ~terminated


def _replay_valid_mask(current: TensorDict, min_step_count: int):
    """Return VAIC's PPO-equivalent post-reset learning-row mask."""
    count = int(current.batch_size[0])
    if "step_count" not in current.keys():
        raise KeyError(
            "FastSAC replay collection requires the VAIC 'step_count' observation"
        )
    step_count = current["step_count"]
    if step_count.numel() != count:
        raise ValueError(
            f"step_count has {step_count.numel()} values for {count} replay rows"
        )
    return step_count.reshape(count) > int(min_step_count)


def _filter_replay_rows(
    current: TensorDict, transitions: dict[str, torch.Tensor], min_step_count: int
):
    """Drop reset-transient rows while preserving transition-field alignment.

    VAIC's reset controller/reference caches need a few control steps to settle.
    PPO already excludes these rows from its losses.  Apply the same rule before
    FastSAC inserts data into either replay so an invalid row can never be
    sampled or exported to H5.
    """
    count = int(current.batch_size[0])
    valid = _replay_valid_mask(current, min_step_count)
    filtered = {}
    for key, value in transitions.items():
        if value.shape[0] != count:
            raise ValueError(
                f"Replay field {key!r} has {value.shape[0]} rows; expected {count}"
            )
        filtered[key] = value[valid]
    return filtered, valid


class _Stage1NStepAccumulator:
    """Aggregate interleaved vector transitions before the flat Stage-1 FIFO.

    Pending rows stay separated by environment.  A full horizon emits its
    oldest start; any environment ``done`` emits every remaining partial start
    and clears that environment.  This mirrors HOI's sampled n-step return while
    avoiding circular-buffer episode crossings and retaining VAIC's true
    timeout final observation.
    """

    def __init__(self, n_steps: int, gamma: float, next_fields=()):
        if (
            isinstance(n_steps, bool)
            or not isinstance(n_steps, (int, np.integer))
            or int(n_steps) < 1
        ):
            raise ValueError("Stage-1 FastSAC n_steps must be a positive integer")
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("Stage-1 FastSAC gamma must be finite and non-negative")
        self.n_steps = int(n_steps)
        self.gamma = gamma
        self.next_fields = frozenset(next_fields)
        self._buffers: dict[str, torch.Tensor] = {}
        self._lengths: torch.Tensor | None = None

    def clear(self):
        self._buffers.clear()
        self._lengths = None

    def _initialize(self, transitions: dict[str, torch.Tensor]):
        count = int(transitions["rewards"].shape[0])
        device = transitions["rewards"].device
        self._lengths = torch.zeros(count, dtype=torch.long, device=device)
        self._buffers = {
            name: torch.empty(
                (count, self.n_steps, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            for name, value in transitions.items()
        }

    def _validate(self, transitions: dict[str, torch.Tensor], valid: torch.Tensor):
        required = {"rewards", "dones", "truncations", "discounts"}
        missing = required.difference(transitions)
        if missing:
            raise KeyError(f"Stage-1 n-step input is missing fields: {sorted(missing)}")
        count = int(transitions["rewards"].shape[0])
        valid = valid.reshape(-1).bool()
        if valid.shape != (count,):
            raise ValueError(
                f"Stage-1 n-step valid mask has shape {tuple(valid.shape)}, "
                f"expected {(count,)}"
            )
        for name, value in transitions.items():
            if value.shape[0] != count:
                raise ValueError(
                    f"Stage-1 n-step field {name!r} has {value.shape[0]} rows; "
                    f"expected {count}"
                )
            if value.device != transitions["rewards"].device:
                raise ValueError(
                    "Stage-1 n-step fields must share one collection device"
                )
        if valid.device != transitions["rewards"].device:
            valid = valid.to(transitions["rewards"].device)
        if self.n_steps == 1:
            return valid
        if self._lengths is None:
            self._initialize(transitions)
        elif (
            self._lengths.shape != (count,)
            or set(self._buffers) != set(transitions)
        ):
            raise ValueError(
                "Stage-1 n-step vector count or transition schema changed "
                "while returns were pending"
            )
        for name, value in transitions.items():
            buffer = self._buffers[name]
            expected = (count, self.n_steps, *value.shape[1:])
            if (
                tuple(buffer.shape) != expected
                or buffer.dtype != value.dtype
                or buffer.device != value.device
            ):
                raise ValueError(
                    f"Stage-1 n-step field {name!r} changed shape, dtype, or device"
                )
        return valid

    @staticmethod
    def _one_step(
        transitions: dict[str, torch.Tensor], valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out = {name: value[valid] for name, value in transitions.items()}
        out["effective_n_steps"] = torch.ones_like(out["rewards"])
        return out

    def _aggregate(
        self,
        transitions: dict[str, torch.Tensor],
        env_indices: torch.Tensor,
        starts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert self._lengths is not None
        ends = self._lengths[env_indices] - 1
        effective = ends - starts + 1
        reward = torch.zeros_like(self._buffers["rewards"][env_indices, starts])
        reward_weight = torch.ones_like(reward)
        env_discount = torch.ones_like(reward)

        for offset in range(self.n_steps):
            active = offset < effective
            positions = (starts + offset).clamp_max(self.n_steps - 1)
            step_reward = self._buffers["rewards"][env_indices, positions]
            step_discount = self._buffers["discounts"][env_indices, positions]
            reward = reward + torch.where(
                active, reward_weight * step_reward, torch.zeros_like(reward)
            )
            reward_weight = torch.where(
                active,
                reward_weight * self.gamma * step_discount,
                reward_weight,
            )
            env_discount = torch.where(
                active, env_discount * step_discount, env_discount
            )

        out = {}
        for name in transitions:
            buffer = self._buffers[name]
            if name == "rewards":
                out[name] = reward
            elif name == "discounts":
                out[name] = env_discount
            elif name in ("dones", "truncations") or name in self.next_fields:
                out[name] = buffer[env_indices, ends]
            else:
                out[name] = buffer[env_indices, starts]
        out["effective_n_steps"] = effective.to(dtype=reward.dtype)
        return out

    @torch.no_grad()
    def append(
        self,
        transitions: dict[str, torch.Tensor],
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Append one vector step and return all newly complete starts."""
        valid = self._validate(transitions, valid)
        if self.n_steps == 1:
            return self._one_step(transitions, valid)

        assert self._lengths is not None
        # An invalid reset-transient row cannot be a continuation of a valid
        # return. Ordinarily its predecessor was already flushed by ``done``;
        # clearing here also prevents silent cross-episode aggregation if the
        # collector ever presents an incomplete reset boundary.
        self._lengths[~valid] = 0
        valid_envs = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_envs.numel():
            positions = self._lengths[valid_envs]
            if (positions >= self.n_steps).any():
                raise RuntimeError("Stage-1 n-step pending queue overflow")
            for name, value in transitions.items():
                self._buffers[name][valid_envs, positions] = value[valid_envs]
            self._lengths[valid_envs] += 1

        done_now = valid & transitions["dones"].reshape(-1).bool()
        full = valid & ~done_now & (self._lengths == self.n_steps)
        offsets = torch.arange(self.n_steps, device=self._lengths.device)
        emit = done_now[:, None] & (offsets[None, :] < self._lengths[:, None])
        emit[:, 0] |= full
        pairs = emit.nonzero(as_tuple=False)
        if pairs.numel():
            out = self._aggregate(transitions, pairs[:, 0], pairs[:, 1])
        else:
            out = {name: value[:0] for name, value in transitions.items()}
            out["effective_n_steps"] = torch.empty_like(
                transitions["rewards"][:0]
            )

        self._lengths[done_now] = 0
        if full.any():
            for buffer in self._buffers.values():
                buffer[full, :-1] = buffer[full, 1:].clone()
            self._lengths[full] = self.n_steps - 1
        return out


class DistributionalQNetwork(nn.Module):
    """The distributional Q network used by HOI FastSAC/r1-student."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        num_atoms,
        layer_norm=True,
        action_fusion="early",
        reference_dueling=False,
    ):
        super().__init__()
        if not isinstance(reference_dueling, bool):
            raise ValueError("reference_dueling must be a boolean")
        self.reference_dueling = reference_dueling
        self.action_fusion = str(action_fusion)
        self.action_hidden_dim = _q_action_hidden_dim(
            hidden_dim, self.action_fusion
        )
        if self.reference_dueling:
            value_layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim)]
            if layer_norm:
                value_layers.append(nn.LayerNorm(hidden_dim))
            value_layers.extend(
                (nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2))
            )
            if layer_norm:
                value_layers.append(nn.LayerNorm(hidden_dim // 2))
            value_layers.extend(
                (nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4))
            )
            if layer_norm:
                value_layers.append(nn.LayerNorm(hidden_dim // 4))
            value_layers.extend(
                (nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms))
            )
            self.value_net = nn.Sequential(*value_layers)

            if self.action_fusion == "early":
                advantage_layers: list[nn.Module] = [
                    nn.Linear(obs_dim + action_dim, hidden_dim)
                ]
                if layer_norm:
                    advantage_layers.append(nn.LayerNorm(hidden_dim))
                advantage_layers.extend(
                    (nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2))
                )
                if layer_norm:
                    advantage_layers.append(nn.LayerNorm(hidden_dim // 2))
                advantage_layers.extend(
                    (nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4))
                )
                if layer_norm:
                    advantage_layers.append(nn.LayerNorm(hidden_dim // 4))
                advantage_layers.extend(
                    (nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms))
                )
                self.advantage_net = nn.Sequential(*advantage_layers)
                return

            advantage_obs_layers: list[nn.Module] = [
                nn.Linear(obs_dim, hidden_dim)
            ]
            if layer_norm:
                advantage_obs_layers.append(nn.LayerNorm(hidden_dim))
            advantage_obs_layers.append(nn.SiLU())
            self.advantage_obs_net = nn.Sequential(*advantage_obs_layers)

            advantage_action_layers: list[nn.Module] = [
                nn.Linear(action_dim, self.action_hidden_dim)
            ]
            if layer_norm:
                advantage_action_layers.append(
                    nn.LayerNorm(self.action_hidden_dim)
                )
            advantage_action_layers.append(nn.SiLU())
            self.advantage_action_net = nn.Sequential(
                *advantage_action_layers
            )

            advantage_layers = [
                nn.Linear(
                    hidden_dim + self.action_hidden_dim, hidden_dim // 2
                )
            ]
            if layer_norm:
                advantage_layers.append(nn.LayerNorm(hidden_dim // 2))
            advantage_layers.extend(
                (nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4))
            )
            if layer_norm:
                advantage_layers.append(nn.LayerNorm(hidden_dim // 4))
            advantage_layers.extend(
                (nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms))
            )
            self.advantage_net = nn.Sequential(*advantage_layers)
            return

        if self.action_fusion == "early":
            # Keep the historical module names, construction order, parameter
            # shapes, and RNG consumption exactly unchanged.
            layers: list[nn.Module] = [
                nn.Linear(obs_dim + action_dim, hidden_dim)
            ]
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend((nn.SiLU(), nn.Linear(hidden_dim, hidden_dim // 2)))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim // 2))
            layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4)))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim // 4))
            layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms)))
            self.net = nn.Sequential(*layers)
            return

        obs_layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim)]
        if layer_norm:
            obs_layers.append(nn.LayerNorm(hidden_dim))
        obs_layers.append(nn.SiLU())
        self.obs_net = nn.Sequential(*obs_layers)

        action_layers: list[nn.Module] = [
            nn.Linear(action_dim, self.action_hidden_dim)
        ]
        if layer_norm:
            action_layers.append(nn.LayerNorm(self.action_hidden_dim))
        action_layers.append(nn.SiLU())
        self.action_net = nn.Sequential(*action_layers)

        layers = [
            nn.Linear(
                hidden_dim + self.action_hidden_dim, hidden_dim // 2
            )
        ]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 4)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 4))
        layers.extend((nn.SiLU(), nn.Linear(hidden_dim // 4, num_atoms)))
        self.net = nn.Sequential(*layers)

    def value_logits(self, obs):
        if not self.reference_dueling:
            raise RuntimeError(
                "value_logits is available only for reference-dueling Q"
            )
        return self.value_net(obs)

    def advantage_logits(self, obs, action):
        if not self.reference_dueling:
            raise RuntimeError(
                "advantage_logits is available only for reference-dueling Q"
            )
        if self.action_fusion == "early":
            return self.advantage_net(torch.cat((obs, action), dim=-1))
        obs_features = self.advantage_obs_net(obs)
        action_features = self.advantage_action_net(action)
        return self.advantage_net(
            torch.cat((obs_features, action_features), dim=-1)
        )

    def forward(self, obs, action, reference_action=None):
        if self.reference_dueling:
            if reference_action is None:
                raise ValueError(
                    "reference-dueling Q requires reference_action"
                )
            if reference_action.shape != action.shape:
                raise ValueError(
                    "Reference-dueling Q action/reference shapes must match, "
                    f"got {tuple(action.shape)} and "
                    f"{tuple(reference_action.shape)}"
                )
            # The reference is an exogenous state-conditioned anchor, never an
            # actor output. Detaching its input preserves critic-parameter
            # gradients through A(s, a_ref) while making the actor derivative
            # come only from A(s, a).
            return (
                self.value_logits(obs)
                + self.advantage_logits(obs, action)
                - self.advantage_logits(obs, reference_action.detach())
            )
        if self.action_fusion == "early":
            return self.net(torch.cat((obs, action), dim=-1))
        obs_features = self.obs_net(obs)
        action_features = self.action_net(action)
        return self.net(torch.cat((obs_features, action_features), dim=-1))


class TwinDistributionalQ(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        num_atoms,
        v_min,
        v_max,
        layer_norm=True,
        action_fusion="early",
        reference_dueling=False,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        if not isinstance(reference_dueling, bool):
            raise ValueError("reference_dueling must be a boolean")
        self.reference_dueling = reference_dueling
        self.qnets = nn.ModuleList(
            DistributionalQNetwork(
                obs_dim,
                action_dim,
                hidden_dim,
                num_atoms,
                layer_norm,
                action_fusion,
                reference_dueling,
            )
            for _ in range(2)
        )
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, obs, action, reference_action=None):
        if self.reference_dueling:
            return torch.stack(
                [q(obs, action, reference_action) for q in self.qnets],
                dim=0,
            )
        return torch.stack([q(obs, action) for q in self.qnets], dim=0)

    def values(self, logits):
        return (F.softmax(logits, dim=-1) * self.support).sum(dim=-1)

    @torch.no_grad()
    def projection(
        self, obs, action, reward, bootstrap, discount,
        reference_action=None,
    ):
        """C51 Bellman projection, independently for Q1 and Q2 (as in HOI)."""
        delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
        target = reward[:, None] + bootstrap[:, None] * discount[:, None] * self.support
        target = target.clamp(self.v_min, self.v_max)
        b = (target - self.v_min) / delta
        lower = b.floor().long()
        upper = b.ceil().long()
        same = lower == upper
        lower = torch.where(same & (lower > 0), lower - 1, lower)
        upper = torch.where(same & (upper == 0), upper + 1, upper)
        offset = torch.arange(reward.shape[0], device=reward.device)[:, None] * self.num_atoms

        projected = []
        for qnet in self.qnets:
            if self.reference_dueling:
                logits = qnet(obs, action, reference_action)
            else:
                logits = qnet(obs, action)
            probs = F.softmax(logits, dim=-1)
            out = torch.zeros_like(probs)
            out.view(-1).index_add_(0, (lower + offset).reshape(-1), (probs * (upper - b)).reshape(-1))
            out.view(-1).index_add_(0, (upper + offset).reshape(-1), (probs * (b - lower)).reshape(-1))
            projected.append(out)
        return torch.stack(projected, dim=0)


def _build_isolated_q_network(
    obs_dim, action_dim, hidden_dim, num_atoms, v_min, v_max,
    layer_norm, device, seed, action_fusion="early",
    reference_dueling=False,
):
    """Build Q1/Q2 without advancing VAIC/environment default RNG streams."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    # Module constructors use the CPU default generator today, but also fork the
    # target CUDA generator so a future device-side initializer cannot leak into
    # rollout sampling.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(int(seed))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        qnet = TwinDistributionalQ(
            obs_dim, action_dim, hidden_dim, num_atoms, v_min, v_max, layer_norm,
            action_fusion, reference_dueling,
        ).to(device)
    return qnet


class TeacherTrainingReplayBuffer:
    """Minimal CPU/CUDA FIFO used only by Stage-1 teacher FastSAC.

    Stage 1 no longer doubles as the producer of the Stage-2 RLPD dataset.  It
    therefore stores only inputs consumed by the teacher Q/actor updates.
    Caller-declared task-wide constants (the Skateboard object geometry here)
    are retained once and expanded when sampled instead of copied into every row.
    """

    fields = TEACHER_TRAINING_REPLAY_FIELDS

    def __init__(
        self,
        capacity,
        critic_dim,
        action_dim,
        device="cpu",
        extra_shapes=None,
        constant_shapes=None,
        seed_storage_ratio=0.0,
        seed_sample_ratio=0.0,
    ):
        if int(capacity) < 1:
            raise ValueError("teacher training replay capacity must be positive")
        self.capacity = int(capacity)
        (
            self.seed_storage_ratio,
            self.seed_sample_ratio,
            seed_capacity,
        ) = _validate_seed_replay_partition(
            seed_storage_ratio, seed_sample_ratio, self.capacity
        )
        self.seed_capacity = int(seed_capacity or 0)
        self.online_capacity = self.capacity - self.seed_capacity
        self.seed_frozen = False
        self.critic_dim = int(critic_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.shapes = {
            "critic_observations": (self.critic_dim,),
            "actions": (self.action_dim,),
            "rewards": (),
            "dones": (),
            "truncations": (),
            "discounts": (),
            "effective_n_steps": (),
            "next_critic_observations": (self.critic_dim,),
        }
        extra_shapes = {
            str(name): tuple(int(dim) for dim in shape)
            for name, shape in dict(extra_shapes or {}).items()
        }
        constant_shapes = {
            str(name): tuple(int(dim) for dim in shape)
            for name, shape in dict(constant_shapes or {}).items()
        }
        duplicate_fields = (
            set(extra_shapes).intersection(self.shapes)
            | set(constant_shapes).intersection(self.shapes)
            | set(extra_shapes).intersection(constant_shapes)
        )
        if duplicate_fields:
            raise ValueError(
                "Teacher training replay fields overlap: "
                f"{sorted(duplicate_fields)}"
            )
        self.shapes.update(extra_shapes)
        self.constant_shapes = constant_shapes
        self.storage_fields = (*self.fields, *extra_shapes.keys())
        self.sample_fields = (*self.storage_fields, *constant_shapes.keys())
        self.dtypes = {
            name: torch.bool if name in ("dones", "truncations") else torch.float32
            for name in self.sample_fields
        }
        self.data: dict[str, torch.Tensor] = {}
        self.constants: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.seen = 0

    @property
    def seed_size(self):
        return self.seed_capacity if self.seed_frozen else 0

    @property
    def online_size(self):
        return self.size - self.seed_size

    def freeze_seed_partition(self):
        """Reserve the physical prefix once it contains reference-policy data.

        Freezing is deliberately zero-copy: before the first actor update every
        replay row was collected by the reference-centred policy, so the
        already populated physical prefix is a valid immutable seed set.  The
        suffix becomes an independent online FIFO from this point onward.
        """
        if self.seed_capacity == 0 or self.seed_frozen:
            return False
        if self.size != self.capacity:
            raise RuntimeError(
                "Stage-1 seed replay can freeze only after the FIFO is full"
            )
        self.seed_frozen = True
        # Retain every existing suffix row as initial online data, then replace
        # it sequentially without ever touching the frozen prefix.
        self.ptr = self.seed_capacity
        return True

    @property
    def saved(self):
        return self.size

    @property
    def estimated_bytes(self):
        row_bytes = sum(
            int(np.prod(tail, dtype=np.int64) if tail else 1)
            * torch.empty((), dtype=self.dtypes[name]).element_size()
            for name, tail in self.shapes.items()
        )
        constant_bytes = sum(
            int(np.prod(tail, dtype=np.int64) if tail else 1)
            * torch.empty((), dtype=self.dtypes[name]).element_size()
            for name, tail in self.constant_shapes.items()
        )
        return self.capacity * row_bytes + constant_bytes

    def _allocate(self):
        if self.data:
            return
        try:
            for name in self.storage_fields:
                self.data[name] = torch.empty(
                    (self.capacity, *self.shapes[name]),
                    dtype=self.dtypes[name],
                    device=self.device,
                )
        except torch.OutOfMemoryError as exc:
            self.data.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gib = self.estimated_bytes / (1024 ** 3)
            raise RuntimeError(
                f"Unable to allocate the {gib:.2f} GiB Stage-1 FastSAC replay "
                f"on {self.device}. Reduce teacher_buffer_capacity."
            ) from exc

    def _validated_values(self, data):
        # Older one-step test/utility callers may omit this derivable field.
        # Treat those rows exactly as one-step transitions while keeping the
        # live Stage-1 collector explicit.
        missing = [
            name for name in self.sample_fields
            if name not in data and name != "effective_n_steps"
        ]
        if missing:
            raise KeyError(
                f"Teacher training replay append is missing fields: {missing}"
            )
        count = int(data["rewards"].shape[0])
        data = dict(data)
        if "effective_n_steps" not in data:
            data["effective_n_steps"] = torch.ones_like(data["rewards"])
        values = {}
        for name in self.storage_fields:
            value = data[name].detach()
            expected = (count, *self.shapes[name])
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Teacher training replay field {name!r} has shape "
                    f"{tuple(value.shape)}, expected {expected}"
                )
            # Keep the rollout tensor on its source device here. ``copy_`` in
            # append performs the one source-to-storage transfer and casts to
            # the replay schema dtype without a second staging allocation.
            values[name] = value

        if count:
            for name, tail in self.constant_shapes.items():
                value = data[name].detach()
                expected = (count, *tail)
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"Teacher training replay constant {name!r} has shape "
                        f"{tuple(value.shape)}, expected {expected}"
                    )
                source_candidate = value[0]
                if not torch.equal(
                    value, source_candidate.expand_as(value)
                ):
                    raise ValueError(
                        f"Teacher training replay constant {name!r} varies "
                        "within one vector-environment step."
                    )
                candidate = source_candidate.to(
                    device=self.device, dtype=self.dtypes[name]
                )
                if name in self.constants:
                    if not torch.equal(candidate, self.constants[name]):
                        raise ValueError(
                            f"Teacher training replay constant {name!r} changed "
                            "after collection started."
                        )
                else:
                    self.constants[name] = candidate.clone()
        return count, values

    @torch.no_grad()
    def append(self, data):
        count, values = self._validated_values(data)
        if count == 0:
            return 0
        self._allocate()
        self.seen += count

        if self.seed_frozen:
            if count >= self.online_capacity:
                for name in self.storage_fields:
                    self.data[name][self.seed_capacity:].copy_(
                        values[name][-self.online_capacity:]
                    )
                self.ptr = self.seed_capacity
                self.size = self.capacity
                return count

            first = min(count, self.capacity - self.ptr)
            second = count - first
            for name in self.storage_fields:
                self.data[name][self.ptr:self.ptr + first].copy_(
                    values[name][:first]
                )
                if second:
                    self.data[name][
                        self.seed_capacity:self.seed_capacity + second
                    ].copy_(values[name][first:])
            self.ptr = self.seed_capacity + (
                (self.ptr - self.seed_capacity + count) % self.online_capacity
            )
            self.size = self.capacity
            return count

        if count >= self.capacity:
            for name in self.storage_fields:
                self.data[name].copy_(values[name][-self.capacity:])
            self.ptr = 0
            self.size = self.capacity
            return count

        first = min(count, self.capacity - self.ptr)
        second = count - first
        for name in self.storage_fields:
            self.data[name][self.ptr:self.ptr + first].copy_(values[name][:first])
            if second:
                self.data[name][:second].copy_(values[name][first:])
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    def sample(self, count, device=None, generator=None, fields=None):
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty teacher training replay.")
        target_device = self.device if device is None else torch.device(device)
        # Keep the existing policy-device RNG stream even when storage moves to
        # CPU. This makes sampled row indices identical to the former GPU FIFO
        # for a given checkpointed q_rng state. Only the tiny index tensor is
        # then moved to the storage device.
        storage_device = next(iter(self.data.values())).device
        index_device = (
            storage_device
            if generator is None
            else torch.device(generator.device)
        )
        count = int(count)
        if self.seed_frozen:
            seed_count = round(count * self.seed_sample_ratio)
            online_count = count - seed_count
            index_parts = []
            if seed_count:
                index_parts.append(torch.randint(
                    0,
                    self.seed_capacity,
                    (seed_count,),
                    device=index_device,
                    generator=generator,
                ))
            if online_count:
                index_parts.append(torch.randint(
                    self.seed_capacity,
                    self.seed_capacity + self.online_size,
                    (online_count,),
                    device=index_device,
                    generator=generator,
                ))
            indices = torch.cat(index_parts)
            if len(index_parts) > 1:
                permutation = torch.randperm(
                    count, device=index_device, generator=generator
                )
                indices = indices[permutation]
        else:
            indices = torch.randint(
                0,
                self.size,
                (count,),
                device=index_device,
                generator=generator,
            )
        if indices.device != storage_device:
            indices = indices.to(device=storage_device)
        fields = self.sample_fields if fields is None else tuple(fields)
        unknown = set(fields).difference(self.sample_fields)
        if unknown:
            raise KeyError(
                f"Unknown teacher training replay sample fields: "
                f"{sorted(unknown)}"
            )
        missing_constants = set(fields).intersection(
            self.constant_shapes
        ).difference(self.constants)
        if missing_constants:
            raise RuntimeError(
                "Teacher training replay constants were not initialized: "
                f"{sorted(missing_constants)}"
            )
        sampled = {
            name: self.data[name][indices]
            for name in fields
            if name in self.shapes
        }
        sampled.update({
            name: self.constants[name].expand(count, *self.constant_shapes[name])
            for name in fields
            if name in self.constant_shapes
        })
        # Each selected field crosses devices exactly once here. VecNorm and Q
        # therefore see a complete policy-device batch and never perform an
        # additional replay-storage transfer.
        return {
            name: (
                value
                if value.device == target_device
                else value.to(device=target_device)
            )
            for name, value in sampled.items()
        }

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self.seed_frozen = False
        self.constants.clear()


class TeacherReplayBuffer:
    """Device-resident FIFO with atomic H5 snapshots for later FastSAC use.

    Rollout appends are device-local.  H5 and host transfers occur only when a
    model checkpoint explicitly requests a snapshot.
    """

    fields = TEACHER_REPLAY_FIELDS

    def __init__(
        self, path, capacity, actor_dim, critic_dim, action_dim, seed,
        device="cpu", snapshot_chunk_rows=4096, replay_id=None,
        actor_backend=FASTSAC_ACTOR_BACKEND, actor_obs_keys=None,
        critic_obs_keys=None, extra_shapes=None, q_action_fusion="early",
        q_action_hidden_dim=None, q_action_coordinates="absolute",
        q_reference_dueling=False, q_actuator_context=None,
    ):
        if int(capacity) < 1:
            raise ValueError("teacher replay capacity must be positive")
        if int(snapshot_chunk_rows) < 1:
            raise ValueError("snapshot_chunk_rows must be positive")
        self.path = os.fspath(path)
        self.capacity = int(capacity)
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.replay_id = str(replay_id or uuid.uuid4())
        self.actor_backend = str(actor_backend)
        self.actor_obs_keys = list(actor_obs_keys or [])
        self.critic_obs_keys = list(critic_obs_keys or [])
        self.q_action_fusion = str(q_action_fusion)
        if self.q_action_fusion not in FASTSAC_Q_ACTION_FUSIONS:
            raise ValueError(
                f"q_action_fusion must be one of {FASTSAC_Q_ACTION_FUSIONS}"
            )
        default_action_hidden = 0 if self.q_action_fusion == "early" else None
        if q_action_hidden_dim is None:
            q_action_hidden_dim = default_action_hidden
        if (
            q_action_hidden_dim is None
            or isinstance(q_action_hidden_dim, bool)
            or int(q_action_hidden_dim) < 0
            or (
                self.q_action_fusion == "late"
                and int(q_action_hidden_dim) < 1
            )
            or (
                self.q_action_fusion == "early"
                and int(q_action_hidden_dim) != 0
            )
        ):
            raise ValueError(
                "q_action_hidden_dim must be zero for early fusion and "
                "positive for late fusion"
            )
        self.q_action_hidden_dim = int(q_action_hidden_dim)
        self.q_action_coordinates = str(q_action_coordinates)
        if self.q_action_coordinates not in FASTSAC_Q_ACTION_COORDINATES:
            raise ValueError(
                "q_action_coordinates must be one of "
                f"{FASTSAC_Q_ACTION_COORDINATES}"
            )
        if not isinstance(q_reference_dueling, bool):
            raise ValueError("q_reference_dueling must be a boolean")
        self.q_reference_dueling = q_reference_dueling
        self.q_actuator_context = _normalize_q_actuator_context_metadata(
            q_actuator_context
        )
        self.device = torch.device(device)
        self.snapshot_chunk_rows = int(snapshot_chunk_rows)
        self.shapes = {
            "observations": (self.actor_dim,),
            "critic_observations": (self.critic_dim,),
            "actions": (self.action_dim,),
            "rewards": (),
            "dones": (),
            "truncations": (),
            "discounts": (),
            "next_observations": (self.actor_dim,),
            "next_critic_observations": (self.critic_dim,),
        }
        extra_shapes = {
            str(name): tuple(int(dim) for dim in shape)
            for name, shape in dict(extra_shapes or {}).items()
        }
        if (
            self.q_action_coordinates == "reference_residual"
            or self.q_reference_dueling
        ):
            required_reference_shapes = {
                TEACHER_REF_ACTION_FIELD: (self.action_dim,),
                NEXT_TEACHER_REF_ACTION_FIELD: (self.action_dim,),
            }
            missing_or_mismatched = {
                name: extra_shapes.get(name)
                for name, shape in required_reference_shapes.items()
                if extra_shapes.get(name) != shape
            }
            if missing_or_mismatched:
                raise ValueError(
                    "reference-dependent teacher replay requires current and "
                    "next reference-action fields with the action shape; got "
                    f"{missing_or_mismatched}"
                )
        actuator_fields = {
            TEACHER_ACTUATOR_CONTEXT_FIELD,
            NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
        }
        if self.q_actuator_context["enabled"]:
            context_shape = (self.q_actuator_context["dimension"],)
            mismatched_context = {
                name: extra_shapes.get(name)
                for name in actuator_fields
                if extra_shapes.get(name) != context_shape
            }
            if mismatched_context:
                raise ValueError(
                    "Actuator-conditioned teacher replay requires matching "
                    "current and next context fields; got "
                    f"{mismatched_context}"
                )
        elif actuator_fields.intersection(extra_shapes):
            raise ValueError(
                "Disabled Q actuator conditioning cannot store actuator-context fields"
            )
        duplicate_fields = set(extra_shapes).intersection(self.shapes)
        if duplicate_fields:
            raise ValueError(
                f"Teacher replay extra fields duplicate base fields: {sorted(duplicate_fields)}"
            )
        self.shapes.update(extra_shapes)
        self.storage_fields = (*self.fields, *extra_shapes.keys())
        self.dtypes = {
            name: torch.bool if name in ("dones", "truncations") else torch.float32
            for name in self.storage_fields
        }
        # Allocate lazily at the first FastSAC rollout append.
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None

    @property
    def saved(self):
        return self.size

    @property
    def estimated_bytes(self):
        return self.capacity * sum(
            int(np.prod(tail, dtype=np.int64) if tail else 1)
            * torch.empty((), dtype=self.dtypes[name]).element_size()
            for name, tail in self.shapes.items()
        )

    def _allocate(self):
        if self.data:
            return
        try:
            for name in self.storage_fields:
                self.data[name] = torch.empty(
                    (self.capacity, *self.shapes[name]),
                    dtype=self.dtypes[name], device=self.device,
                )
        except torch.OutOfMemoryError as exc:
            self.data.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gib = self.estimated_bytes / (1024 ** 3)
            raise RuntimeError(
                f"Unable to allocate the {gib:.2f} GiB teacher replay FIFO on "
                f"{self.device}. Reduce teacher_buffer_capacity."
            ) from exc

    def _validated_values(self, data):
        missing = [name for name in self.storage_fields if name not in data]
        if missing:
            raise KeyError(f"Teacher replay append is missing fields: {missing}")
        count = int(data["rewards"].shape[0])
        values = {}
        for name in self.storage_fields:
            value = data[name].detach()
            expected = (count, *self.shapes[name])
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Teacher replay field {name!r} has shape {tuple(value.shape)}, "
                    f"expected {expected}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"Teacher replay field {name!r} is on {value.device}, "
                    f"expected {self.device}"
                )
            values[name] = value.to(dtype=self.dtypes[name])
        return count, values

    @torch.no_grad()
    def append(self, data):
        count, values = self._validated_values(data)
        if count == 0:
            return 0
        # The in-memory FIFO has advanced beyond the last on-disk snapshot.
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None
        self._allocate()
        self.seen += count

        if count >= self.capacity:
            for name in self.storage_fields:
                self.data[name].copy_(values[name][-self.capacity:])
            self.ptr = 0
            self.size = self.capacity
            return count

        first = min(count, self.capacity - self.ptr)
        second = count - first
        for name in self.storage_fields:
            self.data[name][self.ptr:self.ptr + first].copy_(values[name][:first])
            if second:
                self.data[name][:second].copy_(values[name][first:])
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    def clear(self):
        """Logically empty the FIFO without reallocating its device tensors."""
        self.ptr = 0
        self.size = 0
        self.seen = 0
        self.last_snapshot_iteration = None
        self.last_snapshot_name = None
        self.last_snapshot_id = None
        self.last_snapshot_size = None
        self.last_snapshot_seen = None

    def sample(self, count, device=None, generator=None, fields=None):
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty teacher FastSAC replay.")
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Teacher replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(
            0, self.size, (int(count),), device=self.device, generator=generator,
        )
        fields = self.storage_fields if fields is None else tuple(fields)
        unknown = set(fields).difference(self.storage_fields)
        if unknown:
            raise KeyError(f"Unknown teacher replay sample fields: {sorted(unknown)}")
        return {name: self.data[name][indices] for name in fields}

    def _chronological_segments(self, row_count=None):
        if self.size == 0:
            return []
        if self.size < self.capacity:
            segments = [(0, self.size)]
        elif self.ptr == 0:
            segments = [(0, self.capacity)]
        else:
            segments = [(self.ptr, self.capacity), (0, self.ptr)]

        if row_count is None:
            return segments
        row_count = max(0, min(int(row_count), self.size))
        skip = self.size - row_count
        selected = []
        for start, end in segments:
            length = end - start
            if skip >= length:
                skip -= length
                continue
            selected.append((start + skip, end))
            skip = 0
        return selected

    def checkpoint_metadata(self):
        """Return the FIFO/snapshot manifest stored beside the policy weights."""
        has_snapshot = self.last_snapshot_id is not None
        return {
            "format_version": TEACHER_REPLAY_FORMAT_VERSION,
            "initial_transition_filter": TEACHER_REPLAY_INITIAL_TRANSITION_FILTER,
            "action_parameterization": FASTSAC_ACTION_PARAMETERIZATION,
            "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
            "reward_scalarization": SAC_REWARD_SCALARIZATION,
            "truncation_next_observation": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "replay_id": self.replay_id,
            "actor_backend": self.actor_backend,
            "q_action_fusion": self.q_action_fusion,
            "q_action_hidden_dim": self.q_action_hidden_dim,
            "q_action_coordinates": self.q_action_coordinates,
            "q_reference_dueling": self.q_reference_dueling,
            "q_actuator_context": copy.deepcopy(self.q_actuator_context),
            "actor_obs_keys": list(self.actor_obs_keys),
            "critic_obs_keys": list(self.critic_obs_keys),
            "actor_obs_dim": self.actor_dim,
            "critic_obs_dim": self.critic_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "size": (
                self.last_snapshot_size if has_snapshot else self.size
            ),
            "seen": (
                self.last_snapshot_seen if has_snapshot else self.seen
            ),
            "storage_fields": list(self.storage_fields),
            "field_shapes": {
                name: list(self.shapes[name]) for name in self.storage_fields
            },
            "snapshot_iteration": self.last_snapshot_iteration,
            "checkpoint_name": self.last_snapshot_name,
            "snapshot_id": self.last_snapshot_id,
        }

    def _validate_restore_file(self, replay, expected_metadata=None):
        path = self.path
        if str(replay.attrs.get("format", "")) != "vaic_fastsac_teacher_buffer":
            raise ValueError(f"Not a VAIC FastSAC teacher buffer: {path}")
        version = int(replay.attrs.get("format_version", 0))
        if version != TEACHER_REPLAY_FORMAT_VERSION:
            raise ValueError(
                f"Teacher replay {path} uses format version {version}; expected "
                f"{TEACHER_REPLAY_FORMAT_VERSION}."
            )
        if (
            str(replay.attrs.get("truncation_next_observation", ""))
            != TRUNCATION_NEXT_OBSERVATION_SEMANTICS
        ):
            raise ValueError(
                f"Teacher replay {path} does not contain true pre-reset VAIC "
                "truncation final observations."
            )
        if (
            str(replay.attrs.get("replay_observation_semantics", ""))
            != REPLAY_OBSERVATION_SEMANTICS
        ):
            raise ValueError(
                f"Teacher replay {path} does not store raw pre-VecNorm "
                "observations."
            )
        if (
            str(replay.attrs.get("reward_scalarization", ""))
            != SAC_REWARD_SCALARIZATION
        ):
            raise ValueError(
                f"Teacher replay {path} does not use summed reward groups."
            )
        if (
            str(replay.attrs.get("initial_transition_filter", ""))
            != TEACHER_REPLAY_INITIAL_TRANSITION_FILTER
        ):
            raise ValueError(
                f"Teacher replay {path} does not exclude reset-transient "
                "step_count <= 1 rows."
            )
        if (
            str(replay.attrs.get("action_parameterization", ""))
            != FASTSAC_ACTION_PARAMETERIZATION
        ):
            raise ValueError(
                f"Teacher replay {path} does not use the current VAIC FastSAC "
                "reference-centered teacher / absolute student action coordinates."
            )
        if str(replay.attrs.get("storage_policy", "")) != "circular_fifo":
            raise ValueError(f"Teacher replay {path} is not a circular FIFO snapshot.")
        if str(replay.attrs.get("storage_order", "")) != "oldest_to_newest":
            raise ValueError(
                f"Teacher replay {path} is not stored oldest-to-newest."
            )

        actor_keys = json.loads(str(replay.attrs.get("actor_obs_keys", "[]")))
        critic_keys = json.loads(str(replay.attrs.get("critic_obs_keys", "[]")))
        actual_q_actuator_context = _normalize_q_actuator_context_metadata(
            json.loads(str(replay.attrs.get(
                "q_actuator_context", '{"enabled": false}'
            )))
        )
        actual_storage_fields = json.loads(str(replay.attrs.get(
            "storage_fields", json.dumps(list(replay.keys()))
        )))
        actual_field_shapes = json.loads(str(replay.attrs.get(
            "field_shapes",
            json.dumps({name: list(replay[name].shape[1:]) for name in replay.keys()}),
        )))
        actual = {
            "format_version": version,
            "initial_transition_filter": str(
                replay.attrs.get("initial_transition_filter", "")
            ),
            "action_parameterization": str(
                replay.attrs.get("action_parameterization", "")
            ),
            "replay_observation_semantics": str(
                replay.attrs.get("replay_observation_semantics", "")
            ),
            "reward_scalarization": str(
                replay.attrs.get("reward_scalarization", "")
            ),
            "truncation_next_observation": str(
                replay.attrs.get("truncation_next_observation", "")
            ),
            "replay_id": str(replay.attrs.get("replay_id", "")),
            "actor_backend": str(replay.attrs.get("actor_backend", "")),
            "q_action_fusion": str(
                replay.attrs.get("q_action_fusion", "early")
            ),
            "q_action_hidden_dim": int(
                replay.attrs.get("q_action_hidden_dim", 0)
            ),
            "q_action_coordinates": str(
                replay.attrs.get("q_action_coordinates", "")
            ),
            "q_reference_dueling": bool(
                replay.attrs.get("q_reference_dueling", False)
            ),
            "q_actuator_context": actual_q_actuator_context,
            "actor_obs_keys": actor_keys,
            "critic_obs_keys": critic_keys,
            "actor_obs_dim": int(replay.attrs.get("actor_obs_dim", -1)),
            "critic_obs_dim": int(replay.attrs.get("critic_obs_dim", -1)),
            "action_dim": int(replay.attrs.get("action_dim", -1)),
            "capacity": int(replay.attrs.get("buffer_capacity", -1)),
            "size": int(replay.attrs.get("num_transitions", -1)),
            "seen": int(replay.attrs.get("num_seen_transitions", -1)),
            "storage_fields": actual_storage_fields,
            "field_shapes": actual_field_shapes,
            "snapshot_iteration": int(replay.attrs.get("snapshot_iteration", -1)),
            "checkpoint_name": str(replay.attrs.get("checkpoint_name", "")),
            "snapshot_id": str(replay.attrs.get("snapshot_id", "")),
        }
        required = {
            "initial_transition_filter": TEACHER_REPLAY_INITIAL_TRANSITION_FILTER,
            "action_parameterization": FASTSAC_ACTION_PARAMETERIZATION,
            "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
            "reward_scalarization": SAC_REWARD_SCALARIZATION,
            "truncation_next_observation": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "replay_id": self.replay_id,
            "actor_backend": self.actor_backend,
            "q_action_fusion": self.q_action_fusion,
            "q_action_hidden_dim": self.q_action_hidden_dim,
            "q_action_coordinates": self.q_action_coordinates,
            "q_reference_dueling": self.q_reference_dueling,
            "q_actuator_context": self.q_actuator_context,
            "actor_obs_keys": self.actor_obs_keys,
            "critic_obs_keys": self.critic_obs_keys,
            "actor_obs_dim": self.actor_dim,
            "critic_obs_dim": self.critic_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "storage_fields": list(self.storage_fields),
            "field_shapes": {
                name: list(self.shapes[name]) for name in self.storage_fields
            },
        }
        for key, expected in required.items():
            if actual[key] != expected:
                raise ValueError(
                    f"Teacher replay {key} {actual[key]!r} does not match the "
                    f"resumed policy value {expected!r}."
                )

        if actual["size"] < 0 or actual["size"] > self.capacity:
            raise ValueError(
                f"Teacher replay size {actual['size']} is outside [0, {self.capacity}]."
            )
        if actual["seen"] < actual["size"]:
            raise ValueError(
                f"Teacher replay seen count {actual['seen']} is smaller than its "
                f"stored size {actual['size']}."
            )
        if actual["size"] != min(actual["seen"], self.capacity):
            raise ValueError(
                "Teacher replay FIFO counters are inconsistent: "
                f"size={actual['size']}, seen={actual['seen']}, "
                f"capacity={self.capacity}."
            )

        if expected_metadata is not None:
            for key, expected in dict(expected_metadata).items():
                if key in actual and expected is not None and actual[key] != expected:
                    raise ValueError(
                        f"Teacher replay snapshot {key} {actual[key]!r} does not "
                        f"match checkpoint value {expected!r}. Use the H5 saved "
                        "with that exact checkpoint."
                    )

        actual_names = set(replay.keys())
        expected_names = set(self.storage_fields)
        if actual_names != expected_names:
            raise ValueError(
                f"Teacher replay datasets {sorted(actual_names)} do not match "
                f"{sorted(expected_names)}."
            )
        for name in self.storage_fields:
            expected_shape = (actual["size"], *self.shapes[name])
            if tuple(replay[name].shape) != expected_shape:
                raise ValueError(
                    f"Teacher replay dataset {name!r} has shape "
                    f"{tuple(replay[name].shape)}, expected {expected_shape}."
                )
            expected_dtype = np.dtype(
                np.bool_ if self.dtypes[name] == torch.bool else np.float32
            )
            if np.dtype(replay[name].dtype) != expected_dtype:
                raise ValueError(
                    f"Teacher replay dataset {name!r} has dtype "
                    f"{replay[name].dtype}, expected {expected_dtype}."
                )
        return actual

    @torch.no_grad()
    def restore(self, source_path, expected_metadata=None):
        """Restore a matching H5 snapshot into the device-resident FIFO once."""
        if self.data or self.size or self.seen:
            raise RuntimeError("Teacher replay restore requires a new empty FIFO.")
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to restore teacher replay") from exc

        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Teacher replay snapshot does not exist: {source_path}")
        try:
            with h5py.File(source_path, "r") as replay:
                # Error messages should identify the source, not the new run's
                # destination path.
                destination_path = self.path
                self.path = source_path
                try:
                    actual = self._validate_restore_file(replay, expected_metadata)
                finally:
                    self.path = destination_path

                if actual["size"]:
                    self._allocate()
                    for name in self.storage_fields:
                        for start in range(
                            0, actual["size"], self.snapshot_chunk_rows
                        ):
                            end = min(start + self.snapshot_chunk_rows, actual["size"])
                            host = torch.from_numpy(np.asarray(replay[name][start:end]))
                            self.data[name][start:end].copy_(host)
        except Exception:
            self.data.clear()
            self.ptr = 0
            self.size = 0
            self.seen = 0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise

        self.size = actual["size"]
        self.seen = actual["seen"]
        self.ptr = self.size % self.capacity
        self.last_snapshot_iteration = actual["snapshot_iteration"]
        self.last_snapshot_name = actual["checkpoint_name"]
        self.last_snapshot_id = actual["snapshot_id"] or None
        self.last_snapshot_size = actual["size"]
        self.last_snapshot_seen = actual["seen"]
        return self.size

    def snapshot(
        self, iteration, checkpoint_name, row_count=None, seen_count=None
    ):
        """Write oldest-to-newest contents and atomically replace the last H5."""
        snapshot_size = (
            self.size if row_count is None else min(int(row_count), self.size)
        )
        snapshot_seen = self.seen if seen_count is None else int(seen_count)
        if snapshot_size == 0:
            return None
        if snapshot_seen < snapshot_size:
            raise ValueError(
                f"snapshot seen_count={snapshot_seen} is smaller than "
                f"row_count={snapshot_size}"
            )
        if snapshot_size != min(snapshot_seen, self.capacity):
            raise ValueError(
                "snapshot row_count must be the circular-FIFO tail implied by "
                f"seen_count: rows={snapshot_size}, seen={snapshot_seen}, "
                f"capacity={self.capacity}"
            )
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to snapshot teacher replay") from exc

        snapshot_id = str(uuid.uuid4())
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.path)}.", suffix=".tmp", dir=directory,
        )
        os.close(fd)
        try:
            with h5py.File(temp_path, "w") as replay:
                replay.attrs.update({
                    "format": "vaic_fastsac_teacher_buffer",
                    "format_version": TEACHER_REPLAY_FORMAT_VERSION,
                    "truncation_next_observation": (
                        TRUNCATION_NEXT_OBSERVATION_SEMANTICS
                    ),
                    "initial_transition_filter": (
                        TEACHER_REPLAY_INITIAL_TRANSITION_FILTER
                    ),
                    "action_parameterization": FASTSAC_ACTION_PARAMETERIZATION,
                    "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
                    "reward_scalarization": SAC_REWARD_SCALARIZATION,
                    "source": f"{self.actor_backend}_teacher",
                    "actor_backend": self.actor_backend,
                    "q_action_fusion": self.q_action_fusion,
                    "q_action_hidden_dim": self.q_action_hidden_dim,
                    "q_action_coordinates": self.q_action_coordinates,
                    "q_reference_dueling": self.q_reference_dueling,
                    "q_actuator_context": json.dumps(
                        self.q_actuator_context, sort_keys=True
                    ),
                    "replay_id": self.replay_id,
                    "teacher_only": True,
                    "storage_policy": "circular_fifo",
                    "storage_order": "oldest_to_newest",
                    "actor_obs_keys": json.dumps(self.actor_obs_keys),
                    "critic_obs_keys": json.dumps(self.critic_obs_keys),
                    "storage_fields": json.dumps(list(self.storage_fields)),
                    "field_shapes": json.dumps({
                        name: list(self.shapes[name]) for name in self.storage_fields
                    }),
                    "actor_obs_dim": self.actor_dim,
                    "critic_obs_dim": self.critic_dim,
                    "action_dim": self.action_dim,
                    "buffer_capacity": self.capacity,
                    # Legacy alias retained for format-v2 consumers.
                    "reservoir_capacity": self.capacity,
                    "num_transitions": snapshot_size,
                    "num_seen_transitions": snapshot_seen,
                    "snapshot_iteration": int(iteration),
                    "checkpoint_name": str(checkpoint_name),
                    "snapshot_id": snapshot_id,
                })
                datasets = {
                    name: replay.create_dataset(
                        name, (snapshot_size, *self.shapes[name]),
                        dtype=np.bool_ if self.dtypes[name] == torch.bool else np.float32,
                    )
                    for name in self.storage_fields
                }
                destination = 0
                for segment_start, segment_end in self._chronological_segments(
                    snapshot_size
                ):
                    for chunk_start in range(
                        segment_start, segment_end, self.snapshot_chunk_rows
                    ):
                        chunk_end = min(
                            chunk_start + self.snapshot_chunk_rows, segment_end
                        )
                        count = chunk_end - chunk_start
                        for name in self.storage_fields:
                            arrays = self.data[name][chunk_start:chunk_end].to("cpu").numpy()
                            datasets[name][destination:destination + count] = arrays
                        destination += count
                replay.flush()

            file_fd = os.open(temp_path, os.O_RDONLY)
            try:
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.replace(temp_path, self.path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        self.last_snapshot_iteration = int(iteration)
        self.last_snapshot_name = str(checkpoint_name)
        self.last_snapshot_id = snapshot_id
        self.last_snapshot_size = snapshot_size
        self.last_snapshot_seen = snapshot_seen
        return self.path


class OfflineReplayH5:
    """Load a teacher H5 once, then sample entirely from the target device."""

    def __init__(
        self, path, actor_dim, critic_dim, action_dim, device="cpu",
        max_size=None, seed=0, load_chunk_rows=4096,
        expected_actor_backend=None,
        expected_actor_obs_keys=None, expected_critic_obs_keys=None,
        expected_q_action_fusion=None, expected_q_action_hidden_dim=None,
        expected_q_action_coordinates=None,
        expected_q_reference_dueling=None,
        expected_q_actuator_context=None,
    ):
        import h5py
        if max_size is not None and int(max_size) < 1:
            raise ValueError("offline replay max_size must be positive")
        if int(load_chunk_rows) < 1:
            raise ValueError("offline replay load_chunk_rows must be positive")
        if (
            expected_q_reference_dueling is not None
            and not isinstance(expected_q_reference_dueling, bool)
        ):
            raise ValueError(
                "expected_q_reference_dueling must be a boolean or None"
            )
        normalized_expected_q_actuator_context = (
            None
            if expected_q_actuator_context is None
            else _normalize_q_actuator_context_metadata(
                expected_q_actuator_context
            )
        )
        self.path = path
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))
        shapes = {
            "observations": (int(actor_dim),),
            "critic_observations": (int(critic_dim),),
            "actions": (int(action_dim),),
            "rewards": (), "dones": (), "truncations": (), "discounts": (),
            "next_observations": (int(actor_dim),),
            "next_critic_observations": (int(critic_dim),),
            TEACHER_REF_ACTION_FIELD: (int(action_dim),),
            NEXT_TEACHER_REF_ACTION_FIELD: (int(action_dim),),
        }
        dtypes = {
            name: torch.bool if name in ("dones", "truncations") else torch.float32
            for name in shapes
        }
        with h5py.File(path, "r") as f:
            if str(f.attrs.get("format", "")) != "vaic_fastsac_teacher_buffer":
                raise ValueError(f"Not a VAIC FastSAC teacher buffer: {path}")
            version = int(f.attrs.get("format_version", 0))
            if version != TEACHER_REPLAY_FORMAT_VERSION:
                raise ValueError(
                    f"Teacher replay {path} uses legacy format version {version}; version "
                    f"{TEACHER_REPLAY_FORMAT_VERSION} with true VAIC truncation "
                    "final observations and reset-transient filtering is required."
                )
            if (
                str(f.attrs.get("truncation_next_observation", ""))
                != TRUNCATION_NEXT_OBSERVATION_SEMANTICS
            ):
                raise ValueError(
                    f"Teacher replay {path} does not declare pre-reset VAIC "
                    "truncation final observations."
                )
            if (
                str(f.attrs.get("replay_observation_semantics", ""))
                != REPLAY_OBSERVATION_SEMANTICS
            ):
                raise ValueError(
                    f"Teacher replay {path} does not contain raw pre-VecNorm "
                    "observations."
                )
            if (
                str(f.attrs.get("reward_scalarization", ""))
                != SAC_REWARD_SCALARIZATION
            ):
                raise ValueError(
                    f"Teacher replay {path} does not use summed reward groups."
                )
            if (
                str(f.attrs.get("initial_transition_filter", ""))
                != TEACHER_REPLAY_INITIAL_TRANSITION_FILTER
            ):
                raise ValueError(
                    f"Teacher replay {path} does not exclude reset-transient "
                    "step_count <= 1 rows."
                )
            if (
                str(f.attrs.get("action_parameterization", ""))
                != FASTSAC_ACTION_PARAMETERIZATION
            ):
                raise ValueError(
                    f"Teacher replay {path} does not use the current VAIC FastSAC "
                    "reference-centered teacher / absolute student action coordinates."
                )
            actual_actor_backend = str(f.attrs.get("actor_backend", ""))
            if (
                expected_actor_backend is not None
                and actual_actor_backend != str(expected_actor_backend)
            ):
                raise ValueError(
                    f"Teacher replay actor backend {actual_actor_backend!r} does not match "
                    f"checkpoint backend {str(expected_actor_backend)!r}."
                )
            actual_q_action_fusion = str(
                f.attrs.get("q_action_fusion", "early")
            )
            actual_q_action_hidden_dim = int(
                f.attrs.get("q_action_hidden_dim", 0)
            )
            actual_q_action_coordinates = str(
                f.attrs.get("q_action_coordinates", "")
            )
            actual_q_reference_dueling = bool(
                f.attrs.get("q_reference_dueling", False)
            )
            actual_q_actuator_context = _normalize_q_actuator_context_metadata(
                json.loads(str(f.attrs.get(
                    "q_actuator_context", '{"enabled": false}'
                )))
            )
            if (
                normalized_expected_q_actuator_context is not None
                and actual_q_actuator_context
                != normalized_expected_q_actuator_context
            ):
                raise ValueError(
                    "Teacher replay Q actuator-context metadata "
                    f"{actual_q_actuator_context!r} does not match policy "
                    f"metadata {normalized_expected_q_actuator_context!r}."
                )
            actuator_fields = {
                TEACHER_ACTUATOR_CONTEXT_FIELD,
                NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
            }
            present_actuator_fields = actuator_fields.intersection(f.keys())
            if actual_q_actuator_context["enabled"]:
                context_shape = (actual_q_actuator_context["dimension"],)
                shapes.update({
                    TEACHER_ACTUATOR_CONTEXT_FIELD: context_shape,
                    NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: context_shape,
                })
                dtypes.update({
                    TEACHER_ACTUATOR_CONTEXT_FIELD: torch.float32,
                    NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: torch.float32,
                })
                if present_actuator_fields != actuator_fields:
                    raise ValueError(
                        "Actuator-conditioned teacher replay is missing current "
                        "or next actuator-context datasets"
                    )
            elif present_actuator_fields:
                raise ValueError(
                    "Teacher replay contains actuator-context datasets but declares "
                    "Q actuator conditioning disabled"
                )
            if actual_q_action_coordinates not in FASTSAC_Q_ACTION_COORDINATES:
                raise ValueError(
                    "Teacher replay has invalid or missing Q action "
                    f"coordinates {actual_q_action_coordinates!r}."
                )
            if (
                expected_q_action_fusion is not None
                and actual_q_action_fusion != str(expected_q_action_fusion)
            ):
                raise ValueError(
                    "Teacher replay Q action fusion "
                    f"{actual_q_action_fusion!r} does not match policy fusion "
                    f"{str(expected_q_action_fusion)!r}."
                )
            if (
                expected_q_action_hidden_dim is not None
                and actual_q_action_hidden_dim
                != int(expected_q_action_hidden_dim)
            ):
                raise ValueError(
                    "Teacher replay Q action hidden dim "
                    f"{actual_q_action_hidden_dim} does not match policy value "
                    f"{int(expected_q_action_hidden_dim)}."
                )
            if (
                expected_q_action_coordinates is not None
                and actual_q_action_coordinates
                != str(expected_q_action_coordinates)
            ):
                raise ValueError(
                    "Teacher replay Q action coordinates "
                    f"{actual_q_action_coordinates!r} do not match policy "
                    f"coordinates {str(expected_q_action_coordinates)!r}."
                )
            if (
                expected_q_reference_dueling is not None
                and actual_q_reference_dueling
                != expected_q_reference_dueling
            ):
                raise ValueError(
                    "Teacher replay Q reference-dueling setting "
                    f"{actual_q_reference_dueling!r} does not match policy "
                    f"setting {bool(expected_q_reference_dueling)!r}."
                )
            actual_actor_keys = json.loads(str(f.attrs.get("actor_obs_keys", "[]")))
            actual_critic_keys = json.loads(str(f.attrs.get("critic_obs_keys", "[]")))
            if (
                expected_actor_obs_keys is not None
                and actual_actor_keys != list(expected_actor_obs_keys)
            ):
                raise ValueError(
                    f"Teacher replay actor observation keys {actual_actor_keys} do not match "
                    f"checkpoint keys {list(expected_actor_obs_keys)}."
                )
            if (
                expected_critic_obs_keys is not None
                and actual_critic_keys != list(expected_critic_obs_keys)
            ):
                raise ValueError(
                    f"Teacher replay critic observation keys {actual_critic_keys} do not match "
                    f"checkpoint keys {list(expected_critic_obs_keys)}."
                )
            expected = (actor_dim, critic_dim, action_dim)
            actual = tuple(int(f.attrs[k]) for k in ("actor_obs_dim", "critic_obs_dim", "action_dim"))
            if actual != expected:
                raise ValueError(f"Teacher replay dimensions {actual} do not match FastSAC {expected}")
            file_size = int(f.attrs["num_transitions"])
            if file_size < 1:
                raise ValueError(f"Teacher replay buffer is empty: {path}")
            snapshot_actual = {
                "snapshot_id": str(f.attrs.get("snapshot_id", "")),
                "snapshot_iteration": int(
                    f.attrs.get("snapshot_iteration", -1)
                ),
                "checkpoint_name": str(f.attrs.get("checkpoint_name", "")),
                "size": file_size,
                "seen": int(f.attrs.get("num_seen_transitions", -1)),
                "q_action_fusion": actual_q_action_fusion,
                "q_action_hidden_dim": actual_q_action_hidden_dim,
                "q_action_coordinates": actual_q_action_coordinates,
                "q_reference_dueling": actual_q_reference_dueling,
                "q_actuator_context": actual_q_actuator_context,
            }
            if not actual_q_actuator_context["enabled"]:
                # Preserve the exact legacy Stage-2 provenance dictionary for
                # default-off datasets; enabled datasets carry the new schema.
                snapshot_actual.pop("q_actuator_context")
            self.snapshot_metadata = snapshot_actual
            self.size = file_size if max_size is None else min(file_size, int(max_size))
            source_start = file_size - self.size
            if self.size < file_size:
                storage_order = str(f.attrs.get("storage_order", ""))
                selection = (
                    "newest" if storage_order == "oldest_to_newest"
                    else "trailing deterministic subset"
                )
                logging.warning(
                    "Teacher replay has %d rows; loading a %s of %d rows onto %s.",
                    file_size, selection, self.size, self.device,
                )
            load_fields = list(TEACHER_REPLAY_FIELDS)
            if (
                actual_q_action_coordinates == "reference_residual"
                or actual_q_reference_dueling
            ):
                load_fields.extend((
                    TEACHER_REF_ACTION_FIELD,
                    NEXT_TEACHER_REF_ACTION_FIELD,
                ))
            if actual_q_actuator_context["enabled"]:
                load_fields.extend((
                    TEACHER_ACTUATOR_CONTEXT_FIELD,
                    NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
                ))
            self.fields = tuple(load_fields)
            for name in self.fields:
                expected_shape = (file_size, *shapes[name])
                if name not in f or tuple(f[name].shape) != expected_shape:
                    actual_shape = tuple(f[name].shape) if name in f else None
                    raise ValueError(
                        f"Teacher replay dataset {name!r} has shape {actual_shape}, "
                        f"expected {expected_shape}"
                    )
            self.data = {}
            try:
                for name in self.fields:
                    self.data[name] = torch.empty(
                        (self.size, *shapes[name]), dtype=dtypes[name],
                        device=self.device,
                    )
                for name in self.fields:
                    for destination in range(0, self.size, int(load_chunk_rows)):
                        count = min(int(load_chunk_rows), self.size - destination)
                        source = source_start + destination
                        host = torch.from_numpy(np.asarray(f[name][source:source + count]))
                        self.data[name][destination:destination + count].copy_(host)
            except torch.OutOfMemoryError as exc:
                self.data.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                required = self.size * sum(
                    int(np.prod(tail, dtype=np.int64) if tail else 1)
                    * torch.empty((), dtype=dtypes[name]).element_size()
                    for name in self.fields
                    for tail in (shapes[name],)
                )
                raise RuntimeError(
                    f"Unable to load {required / (1024 ** 3):.2f} GiB teacher replay "
                    f"onto {self.device}. Reduce teacher_buffer_capacity."
                ) from exc

    def sample(self, count, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Offline replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(
            0, self.size, (int(count),), device=self.device, generator=self.rng,
        )
        return {name: value[indices] for name, value in self.data.items()}


class BCDaggerOfflineReplayH5:
    """Load raw-v2 or legacy normalized-v1 BC-DAgger data for Stage 2."""

    def __init__(
        self,
        path,
        actor_dim,
        critic_dim,
        action_dim,
        device="cpu",
        max_size=None,
        seed=0,
        load_chunk_rows=4096,
        expected_actor_obs_keys=None,
        expected_critic_obs_keys=None,
        expected_vecnorm_fingerprint=None,
        expected_action_clip=None,
    ):
        import h5py

        if max_size is not None and int(max_size) < 1:
            raise ValueError("offline replay max_size must be positive")
        if int(load_chunk_rows) < 1:
            raise ValueError("offline replay load_chunk_rows must be positive")
        self.path = os.path.abspath(os.fspath(path))
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))
        if expected_action_clip is not None:
            expected_action_clip = float(expected_action_clip)
            if (
                not math.isfinite(expected_action_clip)
                or expected_action_clip <= 0.0
            ):
                raise ValueError(
                    "expected_action_clip must be finite and positive"
                )
        shapes = {
            "observations": (int(actor_dim),),
            "critic_observations": (int(critic_dim),),
            "actions": (int(action_dim),),
            "rewards": (),
            "dones": (),
            "truncations": (),
            "discounts": (),
            "next_observations": (int(actor_dim),),
            "next_critic_observations": (int(critic_dim),),
        }
        dtypes = {
            name: torch.bool if name in ("dones", "truncations")
            else torch.float32
            for name in shapes
        }
        with h5py.File(self.path, "r") as replay:
            if str(replay.attrs.get("format", "")) != BC_DAGGER_REPLAY_FORMAT:
                raise ValueError(f"Not a VAIC PPO-BC DAgger replay: {self.path}")
            version = int(replay.attrs.get("format_version", 0))
            observation_semantics = str(
                replay.attrs.get("replay_observation_semantics", "")
            )
            schema = (version, observation_semantics)
            raw_schema = (
                BC_DAGGER_REPLAY_FORMAT_VERSION,
                BC_DAGGER_REPLAY_OBSERVATION_SEMANTICS,
            )
            legacy_schema = (
                BC_DAGGER_LEGACY_REPLAY_FORMAT_VERSION,
                BC_DAGGER_LEGACY_REPLAY_OBSERVATION_SEMANTICS,
            )
            if schema not in (raw_schema, legacy_schema):
                raise ValueError(
                    "Unsupported PPO-BC DAgger replay schema: "
                    f"version={version}, observations={observation_semantics!r}"
                )
            self.observations_pre_normalized = schema == legacy_schema
            actual_vecnorm_fingerprint = str(
                replay.attrs.get("vecnorm_fingerprint", "")
            )
            if schema == raw_schema:
                expected_fingerprint = str(
                    expected_vecnorm_fingerprint or ""
                )
                if not expected_fingerprint.startswith("sha256:"):
                    raise ValueError(
                        "Raw BC-DAgger replay requires the Stage-2 checkpoint "
                        "VecNorm fingerprint"
                    )
                if actual_vecnorm_fingerprint != expected_fingerprint:
                    raise ValueError(
                        "Raw BC-DAgger replay VecNorm fingerprint does not "
                        "match the Stage-2 checkpoint"
                    )
            if str(replay.attrs.get("reward_scalarization", "")) != (
                SAC_REWARD_SCALARIZATION
            ):
                raise ValueError("BC-DAgger replay reward scalarization mismatch")
            if str(replay.attrs.get("actor_backend", "")) != (
                BC_DAGGER_ACTOR_BACKEND
            ):
                raise ValueError("BC-DAgger replay actor backend mismatch")
            replay_action_clip = replay.attrs.get("action_clip")
            if (
                expected_action_clip is not None
                and replay_action_clip is not None
                and not math.isclose(
                    float(replay_action_clip),
                    expected_action_clip,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "BC-DAgger replay action clip does not match Stage 2"
                )
            actor_keys = json.loads(str(
                replay.attrs.get("actor_obs_keys", "[]")
            ))
            critic_keys = json.loads(str(
                replay.attrs.get("critic_obs_keys", "[]")
            ))
            if (
                expected_actor_obs_keys is not None
                and actor_keys != list(expected_actor_obs_keys)
            ):
                raise ValueError(
                    f"BC-DAgger actor observation keys {actor_keys} do not "
                    f"match Stage 2 {list(expected_actor_obs_keys)}"
                )
            if (
                expected_critic_obs_keys is not None
                and critic_keys != list(expected_critic_obs_keys)
            ):
                raise ValueError(
                    f"BC-DAgger critic observation keys {critic_keys} do not "
                    f"match Stage 2 {list(expected_critic_obs_keys)}"
                )
            expected_dims = (int(actor_dim), int(critic_dim), int(action_dim))
            actual_dims = tuple(int(replay.attrs.get(name, -1)) for name in (
                "actor_obs_dim", "critic_obs_dim", "action_dim"
            ))
            if actual_dims != expected_dims:
                raise ValueError(
                    f"BC-DAgger replay dimensions {actual_dims} do not match "
                    f"Stage 2 {expected_dims}"
                )
            file_size = int(replay.attrs.get("num_transitions", -1))
            if file_size < 1:
                raise ValueError(f"BC-DAgger replay is empty: {self.path}")
            self.size = (
                file_size if max_size is None else min(file_size, int(max_size))
            )
            source_start = file_size - self.size
            self.fields = tuple(TEACHER_REPLAY_FIELDS)
            self.snapshot_metadata = {
                "format": BC_DAGGER_REPLAY_FORMAT,
                "format_version": version,
                "replay_observation_semantics": observation_semantics,
                "vecnorm_fingerprint": actual_vecnorm_fingerprint,
                "snapshot_id": str(replay.attrs.get("snapshot_id", "")),
                "snapshot_iteration": int(
                    replay.attrs.get("snapshot_iteration", -1)
                ),
                "checkpoint_name": str(
                    replay.attrs.get("checkpoint_name", "")
                ),
                "size": self.size,
                "seen": int(replay.attrs.get("num_seen_transitions", -1)),
                "observations_pre_normalized": (
                    self.observations_pre_normalized
                ),
                "action_clip": (
                    None if replay_action_clip is None
                    else float(replay_action_clip)
                ),
            }
            self.data = {}
            try:
                for name in self.fields:
                    expected_shape = (file_size, *shapes[name])
                    actual_shape = (
                        tuple(replay[name].shape) if name in replay else None
                    )
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"BC-DAgger replay field {name!r} has shape "
                            f"{actual_shape}, expected {expected_shape}"
                        )
                    self.data[name] = torch.empty(
                        (self.size, *shapes[name]),
                        dtype=dtypes[name],
                        device=self.device,
                    )
                for name in self.fields:
                    for destination in range(
                        0, self.size, int(load_chunk_rows)
                    ):
                        count = min(
                            int(load_chunk_rows), self.size - destination
                        )
                        source = source_start + destination
                        host = torch.from_numpy(np.asarray(
                            replay[name][source : source + count]
                        ))
                        if name == "actions" and expected_action_clip is not None:
                            if (
                                not torch.isfinite(host).all()
                                or (host.abs() > expected_action_clip).any()
                            ):
                                raise ValueError(
                                    "BC-DAgger replay contains an action outside "
                                    "the Stage-2 actor/Q support"
                                )
                        self.data[name][
                            destination : destination + count
                        ].copy_(host)
            except torch.OutOfMemoryError as exc:
                self.data.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "Unable to load the BC-DAgger offline replay onto "
                    f"{self.device}; reduce algo.teacher_buffer_capacity"
                ) from exc

    def sample(self, count, device=None):
        output_device = self.device if device is None else torch.device(device)
        indices = torch.randint(
            0,
            self.size,
            (int(count),),
            device=self.device,
            generator=self.rng,
        )
        return {
            name: value[indices].to(output_device)
            for name, value in self.data.items()
        }


class OnlineReplay:
    """Device-resident circular replay collected by the FastSAC student."""

    def __init__(self, capacity, device="cpu"):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("online replay capacity must be positive")
        self.device = torch.device(device)
        self.data = {}
        self.ptr = 0
        self.size = 0
        # Total accepted rows, including rows later overwritten by the FIFO.
        # Stage-2 warm-up is defined in transitions rather than ambiguous
        # vector-environment control steps.
        self.seen = 0

    def extend(self, transitions):
        values = {
            k: v.detach().reshape(v.shape[0], *v.shape[1:])
            for k, v in transitions.items()
        }
        wrong_devices = {
            key: str(value.device)
            for key, value in values.items()
            if value.device != self.device
        }
        if wrong_devices:
            raise ValueError(
                f"Online replay transitions must already be on {self.device}; got "
                f"{wrong_devices}"
            )
        n = next(iter(values.values())).shape[0]
        self.seen += int(n)
        if not self.data:
            try:
                self.data = {
                    k: torch.empty(
                        (self.capacity, *v.shape[1:]),
                        dtype=v.dtype, device=self.device,
                    )
                    for k, v in values.items()
                }
            except torch.OutOfMemoryError as exc:
                self.data.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Unable to allocate FastSAC online replay on {self.device}; "
                    "reduce online_buffer_capacity."
                ) from exc
        if n >= self.capacity:
            values = {k: v[-self.capacity:] for k, v in values.items()}
            n = self.capacity
        first = min(n, self.capacity - self.ptr)
        second = n - first
        for key, value in values.items():
            self.data[key][self.ptr:self.ptr + first].copy_(value[:first])
            if second:
                self.data[key][:second].copy_(value[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, count, device=None, generator=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Online replay resides on {self.device}, requested sample on {device}"
            )
        indices = torch.randint(
            0,
            self.size,
            (count,),
            device=self.device,
            generator=generator,
        )
        return {k: v[indices] for k, v in self.data.items()}


@dataclass
class FastSACVelConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.ppo.fastsac_vel.FastSACVEL"
    q_hidden_dim: int = 768
    q_num_atoms: int = 501
    q_v_min: float = -20.0
    q_v_max: float = 20.0
    q_layer_norm: bool = True
    # ``early`` is the exact HOI/current concat-before-first-linear critic.
    # ``late`` gives observations and executable actions independent stems
    # before the 384/192 C51 trunk (768 hidden -> 128 action features).
    q_action_fusion: str = "early"
    # Optional reference-anchored dueling C51 critic:
    # logits(s,a)=V(s)+A(s,a)-A(s,a_ref). The default retains the exact
    # historical monolithic action-conditioned Q topology and execution path.
    q_reference_dueling: bool = False
    # Optional asymmetric critic-only actuator state. The actor and configured
    # VAIC observations remain unchanged. Q receives a delay one-hot plus the
    # fixed-range-centered low-pass alpha captured before each environment step.
    # False preserves the historical network, replay, and checkpoint topology.
    q_condition_on_actuator_state: bool = False
    q_lr: float = 3e-4
    q_weight_decay: float = 1e-3
    q_seed: int = 0
    # Q-only action coordinates. ``absolute`` retains the historical FastSAC
    # input. ``reference_residual`` subtracts the frame's executable VAIC
    # motion reference and divides by the fixed executable half-range, without
    # clamping. The actor, environment, and replay continue to use absolute
    # executable actions in either mode.
    q_action_coordinates: str = "absolute"
    # The actor, environment, and replay retain VAIC's executable joint-action
    # coordinates.  Only Q1/Q2 inputs use the dimensionless unit box, avoiding
    # badly conditioned action columns whose physical ranges differ by joint.
    sac_q_normalize_actions: bool = True
    # Fixed Q-only gain applied after the optional executable-action affine
    # transform. Actor, environment, and replay action coordinates are unchanged.
    sac_q_action_input_gain: float = 1.0
    # Standard clipped double-Q adapted to C51: both critics learn the full
    # distribution from the lower-expectation target head and actor updates use
    # min(Q1, Q2). Disable only to reproduce HOI's independent-target/twin-mean
    # implementation exactly.
    sac_clipped_double_q: bool = True
    # When disabled, sac_alpha_init is the fixed entropy coefficient. Keeping
    # the parameter and optimizer registered makes checkpoints topology-stable.
    sac_use_autotune: bool = True
    # Stage-1 only: train the VAIC adaptation/object-adaptation modules and
    # distill the FastSAC teacher into actor_adapt.  Disabling this leaves the
    # teacher actor/Q/alpha path unchanged, but does not produce a pretrained
    # student warm-start for stage 2.
    train_student_models: bool = True
    # Stage 1 owns only the live Q-training FIFO. A separate collector must
    # produce the full actor+critic H5 used by Stage-2 RLPD.
    save_teacher_buffer: bool = False
    teacher_buffer_filename: str = "teacher_replay_buffer.h5"
    # Stage-2 only: immutable offline data mixed into the RLPD minibatch.
    # Stage 1 never restores or exports its ephemeral learning FIFO.
    teacher_buffer_path: str | None = None
    teacher_buffer_capacity: int = 262_144
    # ``policy`` preserves the historical GPU-local FIFO. Set this to ``cpu``
    # when a long Stage-1 horizon should consume host RAM instead; sampled
    # minibatches are transferred once to the policy device before VecNorm/Q.
    teacher_training_replay_device: str = "policy"
    # Optional zero-extra-storage RLPD-style anchor for Stage 1. Once replay is
    # full at the actor gate, the prefix is frozen as reference-policy data and
    # only the suffix remains an online FIFO. Both ratios must be zero (off) or
    # positive. Sampling may deliberately oversample the smaller seed prefix.
    sac_teacher_seed_storage_ratio: float = 0.0
    sac_teacher_seed_sample_ratio: float = 0.0
    # A 1024-step WBT horizon can be requested with task.buffer_steps=1024.
    # It is not forced here because the compact skateboard replay alone is
    # about 18.6 GiB at 1024 environments (about 74 GiB at 4096 environments).
    teacher_buffer_seed: int = 0
    teacher_buffer_snapshot_chunk_rows: int = 4096
    # Retained for old configs; Stage-1 H5 export is disabled. A future
    # dedicated collector owns the Stage-2 dataset collection start.
    teacher_buffer_start_iteration: int = 7000
    fastsac_actor_hidden_dim: int = 512
    fastsac_log_std_min: float = -5.0
    fastsac_log_std_max: float = 0.0
    # Stage-1-only two-phase exploration schedule. The initial target is
    # applied only to the reference-centred teacher's log-std head. When both
    # reset fields are configured, that head is reset once immediately before
    # the numbered Q update; the actor-learning gate remains independent.
    # None preserves the historical midpoint initialization with no reset.
    sac_teacher_initial_log_std: float | None = None
    sac_teacher_actor_reset_log_std: float | None = None
    sac_teacher_actor_std_reset_q_updates: int | None = None
    fastsac_actor_layer_norm: bool = True
    # Preserve each VecNorm input before normalization. Replay stores this raw
    # value and applies the current VecNorm snapshot only after sampling.
    sac_replay_raw_observations: bool = True
    # The Q-update burst size is independent of num_envs and minibatch size.
    # By default it runs after every vector control step. Increasing the
    # interval preserves collection on every step but runs one burst only every
    # K steps, with no catch-up backlog.
    sac_teacher_updates_per_env_step: int = 4
    sac_teacher_update_interval_env_steps: int = 1
    # Collector-side n-step return horizon. One preserves the historical SAC
    # target exactly; four better spans the task's delayed-action dynamics.
    sac_teacher_n_steps: int = 1
    # Stage 1 can use a larger, independently sampled actor minibatch without
    # changing the Q/alpha minibatch. Zero preserves the historical behavior:
    # the delayed actor update reuses the already prepared Q minibatch exactly.
    sac_teacher_actor_batch_size: int = 0
    # Stage-1 policy improvement. ``sac`` preserves the original reparameterized
    # soft-Q objective exactly. ``reference_awac`` instead regresses replayed
    # actions with detached target-twin advantages relative to VAIC's
    # framewise reference action, preventing actor gradients through an
    # out-of-support Q(s, pi(s)) query. Its Gaussian scale is detached so the
    # configured exploration schedule remains fixed during AWAC updates.
    sac_teacher_actor_objective: str = "sac"
    sac_teacher_awac_beta: float = 0.01
    sac_teacher_awac_weight_clip: float = 20.0
    # Stage-1-only optional behavior-support safeguard. When enabled, a sampled
    # policy action contributes a Q gradient only where its twin-Q improvement
    # over the *same replay row's recorded behavior action* is larger than the
    # twins' disagreement about that improvement. This compares against the
    # action that generated the transition rather than VAIC's raw kinematic
    # reference. The entropy term remains the ordinary SAC objective on every
    # row. False preserves the exact historical actor update and does not affect
    # Stage-2 RLPD.
    sac_teacher_actor_uncertainty_gate: bool = False
    # Optional Stage-1-only conservative Q regularizer. Once active, each Q
    # head is penalized when the deterministic current-policy action is ranked
    # above that replay row's recorded action by more than the configured
    # margin. A zero coefficient is an exact no-op. None starts on the same
    # numbered Q update as the actor-learning gate, including CLI overrides.
    sac_teacher_conservative_q_coef: float = 0.0
    sac_teacher_conservative_q_margin: float = 0.002
    sac_teacher_conservative_q_temperature: float = 0.002
    sac_teacher_conservative_q_starts_q_updates: int | None = None
    sac_teacher_learning_starts_transitions: int = 98_304
    # The reference-centered policy initially collects useful motion-tracking
    # data, while a freshly initialized C51 critic has arbitrary action
    # gradients.  Train Q first so the actor cannot immediately exploit those
    # random, out-of-support gradients. Replay collection, Q, and target-Q
    # remain active; actor and alpha start together after the burn-in gate.
    sac_teacher_actor_learning_starts_q_updates: int = 8_000
    sac_teacher_policy_frequency: int = 32
    sac_teacher_actor_lr: float = 3e-6
    # With policy_frequency=32, alpha receives 32 optimizer steps per actor
    # step.  2e-5 gives nearly the same cumulative alpha step per actor update
    # as HOI's 2:1 update ratio at 3e-4.
    sac_teacher_alpha_lr: float = 2e-5
    # Keep the Q optimizer faithful to HOI's unclipped FastSAC update.  The
    # separately configured actor guard is only a safety bound; the validated
    # skateboard run stayed far below it.
    sac_teacher_q_max_grad_norm: float = 0.0
    sac_teacher_actor_max_grad_norm: float = 1.0
    # Stage-2 retains the original per-control-step RLPD update schedule.
    sac_learning_starts: int = 10
    sac_batch_size: int = 8192
    sac_updates_per_env_step: int = 4
    sac_policy_frequency: int = 2
    sac_actor_lr: float = 3e-4
    sac_alpha_lr: float = 3e-4
    sac_alpha_init: float = 0.001
    sac_target_entropy_ratio: float = 0.5
    sac_tau: float = 0.05
    sac_max_grad_norm: float = 0.0


@dataclass
class FastSACVelFinetuneConfig(FastSACVelConfig):
    _target_: str = "active_adaptation.learning.ppo.fastsac_vel.FastSACVelFinetune"
    phase: str = "finetune"
    vecnorm: str = "eval"
    enable_residual_distillation: bool = False
    save_teacher_buffer: bool = False
    # ``auto`` inspects checkpoint provenance in scripts/helpers.py.  A
    # BC-DAgger source retains its PPO actor backbone behind a bounded SAC
    # distribution adapter and retains the pretrained Q action coordinates.
    finetune_checkpoint_source: str = "auto"
    # Keep the historical warm-start by default.  Set this to false to load
    # only the actor/perception/EMA student state and initialize Stage-2 Q1/Q2
    # (and both targets) from the current FastSAC seed instead of Stage-1 IQL.
    load_pretrained_q: bool = True
    # Stage-2 counts accepted online transitions, not vector control steps,
    # before beginning Q updates.  The actor and alpha have a second, longer
    # Q-update gate below.
    sac_learning_starts: int = 98_304
    # Match one 512-row vector step with one replay minibatch while adapting the
    # transferred IQL critic conservatively rather than repeatedly oversampling
    # each newly accepted transition.
    sac_batch_size: int = 512
    sac_updates_per_env_step: int = 1
    q_lr: float = 3e-5
    sac_policy_frequency: int = 128
    sac_actor_lr: float = 3e-7
    sac_alpha_lr: float = 2e-5
    # The BC/IQL critic was trained with a hard Bellman backup. Stage 2 starts
    # from that exact objective and only introduces SAC temperature after the
    # actor is actually released. The raw alpha is deliberately conservative;
    # its effective value is linearly ramped over the configured Q updates.
    sac_alpha_init: float = 1e-5
    sac_alpha_ramp_q_updates: int = 20_000
    # In the raw-action unit reference, the standard SAC target is -action_dim.
    # The old ratio 4 compensated for the now-removed +log(20) clip offset.
    sac_target_entropy_ratio: float = 1.0
    sac_tau: float = 0.001
    sac_max_grad_norm: float = 1.0
    sac_actor_learning_starts_q_updates: int = 8_000
    # The frozen BC policy is a detached critic-confidence reference, not a
    # behavior-cloning loss. Every scheduled actor tick is independently
    # accepted only when enough sampled actions improve over that reference in
    # both Q heads by more than their disagreement about the action effect.
    sac_actor_confidence_gate: bool = True
    sac_actor_gate_disagreement_multiplier: float = 1.0
    sac_actor_gate_min_accept_fraction: float = 0.10
    sac_actor_gate_absolute_margin: float = 0.0
    # A BC-DAgger source clips every executed teacher/student action to this
    # symmetric raw-action support. scripts/helpers.py replaces the default
    # with the exact checkpoint value before policy construction.
    sac_bc_action_clip: float = 20.0
    # Entropy density coordinates are independent of the DAgger safety clip.
    # A value of one means log pi is measured per raw-action unit rather than
    # in the artificial [-sac_bc_action_clip, sac_bc_action_clip] unit box.
    sac_entropy_reference_scale: float = 1.0
    # Dedicated Stage-2 exploration state. This is intentionally independent
    # of PPO Actor.actor_std, which pure DAgger BC never optimized or sampled.
    sac_bc_initial_action_std: float = 0.01
    sac_bc_log_std_min: float = -8.0
    sac_bc_log_std_max: float = -2.0
    # Training collection samples the same bounded distribution used by SAC so
    # online replay supports the actions queried by the critic and actor. Eval
    # always executes the clipped BC/SAC mean. True is retained only as an
    # explicit deterministic behavior ablation.
    sac_deterministic_rollout: bool = False
    # Confidence gating, not an auxiliary BC regression loss, guards actor
    # release. Keep the legacy schedule fields as explicit zero-valued resume
    # metadata so the default actor objective remains no-anchor SAC.
    sac_bc_anchor_coef_start: float = 0.0
    sac_bc_anchor_coef_end: float = 0.0
    sac_bc_anchor_decay_q_updates: int = 100_000
    sac_bc_anchor_huber_delta: float = 0.1
    # Replay stores priv_pred rather than raw recurrent perception history, so
    # Stage 2 cannot recompute historical latents after an encoder update. Keep
    # the loaded perception fixed and prevent any additional coordinate drift.
    sac_freeze_perception: bool = True
    teacher_buffer_ratio: float = 0.5
    online_buffer_capacity: int = 262_144
    # Accept the doubled BC-DAgger offline FIFO without silently truncating it.
    teacher_buffer_capacity: int = 1_048_576


def _validate_fastsac_teacher_config(cfg) -> None:
    """Validate options that are meaningful only for Stage-1 FastSAC."""
    if not isinstance(
        getattr(cfg, "q_condition_on_actuator_state", False), bool
    ):
        raise ValueError("q_condition_on_actuator_state must be a boolean")
    n_steps = getattr(cfg, "sac_teacher_n_steps", 1)
    if (
        isinstance(n_steps, bool)
        or not isinstance(n_steps, (int, np.integer))
        or int(n_steps) < 1
    ):
        raise ValueError("sac_teacher_n_steps must be a positive integer")
    if not isinstance(cfg.sac_teacher_actor_uncertainty_gate, bool):
        raise ValueError(
            "sac_teacher_actor_uncertainty_gate must be a boolean"
        )
    actor_objective = str(getattr(
        cfg, "sac_teacher_actor_objective", "sac"
    ))
    if actor_objective not in FASTSAC_STAGE1_ACTOR_OBJECTIVES:
        raise ValueError(
            "sac_teacher_actor_objective must be one of "
            f"{FASTSAC_STAGE1_ACTOR_OBJECTIVES}, got {actor_objective!r}"
        )
    awac_beta = float(getattr(cfg, "sac_teacher_awac_beta", 0.01))
    if not math.isfinite(awac_beta) or awac_beta <= 0.0:
        raise ValueError(
            "sac_teacher_awac_beta must be finite and positive"
        )
    awac_weight_clip = float(getattr(
        cfg, "sac_teacher_awac_weight_clip", 20.0
    ))
    if not math.isfinite(awac_weight_clip) or awac_weight_clip < 1.0:
        raise ValueError(
            "sac_teacher_awac_weight_clip must be finite and at least one"
        )
    if (
        actor_objective == "reference_awac"
        and cfg.sac_teacher_actor_uncertainty_gate
    ):
        raise ValueError(
            "sac_teacher_actor_uncertainty_gate applies only to the sac actor "
            "objective; reference_awac already uses in-support target-Q weights"
        )
    if (
        actor_objective == "reference_awac"
        and bool(getattr(cfg, "sac_use_autotune", True))
    ):
        raise ValueError(
            "reference_awac fixes the actor scale, so sac_use_autotune must be "
            "false; otherwise the entropy dual has no trainable scale response"
        )
    conservative_coef = float(getattr(
        cfg, "sac_teacher_conservative_q_coef", 0.0
    ))
    if not math.isfinite(conservative_coef) or conservative_coef < 0.0:
        raise ValueError(
            "sac_teacher_conservative_q_coef must be finite and non-negative"
        )
    conservative_margin = float(getattr(
        cfg, "sac_teacher_conservative_q_margin", 0.002
    ))
    if not math.isfinite(conservative_margin) or conservative_margin < 0.0:
        raise ValueError(
            "sac_teacher_conservative_q_margin must be finite and non-negative"
        )
    conservative_temperature = float(getattr(
        cfg, "sac_teacher_conservative_q_temperature", 0.002
    ))
    if (
        not math.isfinite(conservative_temperature)
        or conservative_temperature <= 0.0
    ):
        raise ValueError(
            "sac_teacher_conservative_q_temperature must be finite and positive"
        )
    conservative_starts = getattr(
        cfg, "sac_teacher_conservative_q_starts_q_updates", None
    )
    if conservative_starts is not None and (
        isinstance(conservative_starts, bool)
        or not isinstance(conservative_starts, (int, np.integer))
        or int(conservative_starts) < 0
    ):
        raise ValueError(
            "sac_teacher_conservative_q_starts_q_updates must be a "
            "non-negative integer or None"
        )
    log_std_min = float(getattr(cfg, "fastsac_log_std_min", -5.0))
    log_std_max = float(getattr(cfg, "fastsac_log_std_max", 0.0))
    if (
        not math.isfinite(log_std_min)
        or not math.isfinite(log_std_max)
        or not log_std_min < log_std_max
    ):
        raise ValueError(
            "fastsac_log_std_min and fastsac_log_std_max must be finite and ordered"
        )
    for name in (
        "sac_teacher_initial_log_std",
        "sac_teacher_actor_reset_log_std",
    ):
        target = getattr(cfg, name, None)
        if target is None:
            continue
        target = float(target)
        if (
            not math.isfinite(target)
            or not log_std_min < target < log_std_max
        ):
            raise ValueError(
                f"{name} must be finite and strictly inside the configured "
                "FastSAC log-std bounds"
            )
    reset_target = getattr(cfg, "sac_teacher_actor_reset_log_std", None)
    reset_q_updates = getattr(
        cfg, "sac_teacher_actor_std_reset_q_updates", None
    )
    if (reset_target is None) != (reset_q_updates is None):
        raise ValueError(
            "sac_teacher_actor_reset_log_std and "
            "sac_teacher_actor_std_reset_q_updates must be configured together"
        )
    if reset_q_updates is not None and (
        isinstance(reset_q_updates, bool)
        or not isinstance(reset_q_updates, (int, np.integer))
        or int(reset_q_updates) < 1
    ):
        raise ValueError(
            "sac_teacher_actor_std_reset_q_updates must be a positive integer"
        )
def _validate_fastsac_finetune_config(cfg) -> None:
    """Reject Stage-2 settings that would silently disable or corrupt SAC."""
    if cfg.phase != "finetune" or cfg.vecnorm != "eval":
        raise ValueError(
            "FastSAC Stage 2 requires phase=finetune and vecnorm=eval so "
            "offline and online raw replay share the checkpoint normalizer"
        )
    if not isinstance(getattr(cfg, "load_pretrained_q", True), bool):
        raise ValueError("load_pretrained_q must be a boolean")
    for name in (
        "sac_learning_starts",
        "sac_actor_learning_starts_q_updates",
    ):
        value = getattr(cfg, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    alpha_ramp_q_updates = getattr(cfg, "sac_alpha_ramp_q_updates")
    if (
        isinstance(alpha_ramp_q_updates, bool)
        or not isinstance(alpha_ramp_q_updates, (int, np.integer))
        or int(alpha_ramp_q_updates) < 1
    ):
        raise ValueError("sac_alpha_ramp_q_updates must be a positive integer")
    anchor_decay = getattr(cfg, "sac_bc_anchor_decay_q_updates")
    if (
        isinstance(anchor_decay, bool)
        or not isinstance(anchor_decay, (int, np.integer))
        or int(anchor_decay) < 1
    ):
        raise ValueError(
            "sac_bc_anchor_decay_q_updates must be a positive integer"
        )
    for name in (
        "sac_deterministic_rollout",
        "sac_freeze_perception",
        "sac_actor_confidence_gate",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be a boolean")
    if not cfg.sac_freeze_perception:
        raise ValueError(
            "Stage-2 replay stores priv_pred, so sac_freeze_perception must "
            "remain true until replay stores raw recurrent perception inputs"
        )
    gate_disagreement_multiplier = float(
        cfg.sac_actor_gate_disagreement_multiplier
    )
    gate_min_accept_fraction = float(
        cfg.sac_actor_gate_min_accept_fraction
    )
    gate_absolute_margin = float(cfg.sac_actor_gate_absolute_margin)
    if (
        not math.isfinite(gate_disagreement_multiplier)
        or gate_disagreement_multiplier < 0.0
    ):
        raise ValueError(
            "sac_actor_gate_disagreement_multiplier must be finite and "
            "non-negative"
        )
    if (
        not math.isfinite(gate_min_accept_fraction)
        or not 0.0 < gate_min_accept_fraction <= 1.0
    ):
        raise ValueError(
            "sac_actor_gate_min_accept_fraction must be finite and in (0, 1]"
        )
    if not math.isfinite(gate_absolute_margin) or gate_absolute_margin < 0.0:
        raise ValueError(
            "sac_actor_gate_absolute_margin must be finite and non-negative"
        )
    action_clip = float(cfg.sac_bc_action_clip)
    entropy_reference_scale = float(cfg.sac_entropy_reference_scale)
    initial_action_std = float(cfg.sac_bc_initial_action_std)
    bc_log_std_min = float(cfg.sac_bc_log_std_min)
    bc_log_std_max = float(cfg.sac_bc_log_std_max)
    if not math.isfinite(action_clip) or action_clip <= 0.0:
        raise ValueError("sac_bc_action_clip must be finite and positive")
    if (
        not math.isfinite(entropy_reference_scale)
        or entropy_reference_scale <= 0.0
    ):
        raise ValueError(
            "sac_entropy_reference_scale must be finite and positive"
        )
    if not math.isfinite(initial_action_std) or initial_action_std <= 0.0:
        raise ValueError(
            "sac_bc_initial_action_std must be finite and positive"
        )
    if (
        not math.isfinite(bc_log_std_min)
        or not math.isfinite(bc_log_std_max)
        or not bc_log_std_min < bc_log_std_max
    ):
        raise ValueError(
            "sac_bc_log_std_min and sac_bc_log_std_max must be finite and ordered"
        )
    initial_log_std = math.log(initial_action_std / action_clip)
    if not bc_log_std_min <= initial_log_std <= bc_log_std_max:
        raise ValueError(
            "sac_bc_initial_action_std maps outside the dedicated BC-adapter "
            "log-std bounds"
        )
    anchor_start = float(cfg.sac_bc_anchor_coef_start)
    anchor_end = float(cfg.sac_bc_anchor_coef_end)
    anchor_huber_delta = float(cfg.sac_bc_anchor_huber_delta)
    if (
        not math.isfinite(anchor_start)
        or not math.isfinite(anchor_end)
        or anchor_start < 0.0
        or anchor_end < 0.0
        or anchor_end > anchor_start
        or not math.isfinite(anchor_huber_delta)
        or anchor_huber_delta <= 0.0
    ):
        raise ValueError(
            "BC anchor coefficients must be finite, non-negative, and decay "
            "from start to end; Huber delta must be finite and positive"
        )
    if not isinstance(
        getattr(cfg, "q_condition_on_actuator_state", False), bool
    ):
        raise ValueError("q_condition_on_actuator_state must be a boolean")
    positive_ints = ("sac_updates_per_env_step", "sac_policy_frequency", "sac_batch_size")
    for name in positive_ints:
        if int(getattr(cfg, name)) < 1:
            raise ValueError(f"{name} must be positive")
    positive_floats = (
        "q_lr",
        "sac_actor_lr",
        "sac_alpha_lr",
        "sac_alpha_init",
    )
    for name in positive_floats:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    tau = float(cfg.sac_tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("sac_tau must be finite and in [0, 1]")
    max_grad_norm = float(cfg.sac_max_grad_norm)
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0.0:
        raise ValueError("sac_max_grad_norm must be finite and non-negative")
    ratio = float(cfg.teacher_buffer_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("teacher_buffer_ratio must be finite and in [0, 1]")
    entropy_ratio = float(cfg.sac_target_entropy_ratio)
    if not math.isfinite(entropy_ratio) or entropy_ratio < 0.0:
        raise ValueError(
            "sac_target_entropy_ratio must be finite and non-negative"
        )
    if not isinstance(cfg.sac_q_normalize_actions, bool):
        raise ValueError("sac_q_normalize_actions must be a boolean")
    if not isinstance(getattr(cfg, "q_reference_dueling", False), bool):
        raise ValueError("q_reference_dueling must be a boolean")
    q_action_coordinates = str(getattr(
        cfg, "q_action_coordinates", "absolute"
    ))
    if q_action_coordinates not in FASTSAC_Q_ACTION_COORDINATES:
        raise ValueError(
            "q_action_coordinates must be one of "
            f"{FASTSAC_Q_ACTION_COORDINATES}, got {q_action_coordinates!r}"
        )
    q_action_input_gain = float(cfg.sac_q_action_input_gain)
    if not math.isfinite(q_action_input_gain) or q_action_input_gain <= 0.0:
        raise ValueError("sac_q_action_input_gain must be finite and positive")
    if not isinstance(cfg.sac_clipped_double_q, bool):
        raise ValueError("sac_clipped_double_q must be a boolean")
    if not isinstance(cfg.sac_use_autotune, bool):
        raise ValueError("sac_use_autotune must be a boolean")


_in_keys: List[str] = (
    CMD_KEY,
    OBS_KEY,
    OBJECT_KEY,
    OBS_PRIV_KEY,
    OBJECT_GEO_KEY,
    HEIGHT_KEY,
    VEL_CMD_KEY,
)
ConfigStore.instance().store(
    "fastsac_vel_train",
    node=FastSACVelConfig(
        name="fastsac_vel", phase="train", vecnorm="train",
        gamma=0.99, entropy_coef_start=0.001, entropy_coef_end=0.001,
        teacher_buffer_start_iteration=7000,
        in_keys=_in_keys,
    ),
    group="algo",
)
ConfigStore.instance().store(
    "fastsac_vel_finetune",
    node=FastSACVelFinetuneConfig(
        name="fastsac_vel", phase="finetune", vecnorm="eval",
        gamma=0.99, enable_residual_distillation=False,
        in_keys=(
            CMD_KEY,
            OBS_KEY,
            OBJECT_KEY,
            OBS_PRIV_KEY,
            OBJECT_GEO_KEY,
            VEL_CMD_KEY,
            DEPTH_KEY,
        ),
    ),
    group="algo",
)


class _FastSACVAICBase(PPOVEL):
    """Shared FastSAC plumbing around unchanged VAIC perception modules."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        # Bound by scripts/helpers.py after the copy-before-VecNorm transform is
        # constructed. Keep this as a non-registered reference: VecNorm already
        # belongs to the environment/checkpoint and must not be duplicated here.
        object.__setattr__(self, "_replay_vecnorm", None)
        self._replay_vecnorm_keys = set()
        self._replay_vecnorm_fingerprint = None
        if cfg.q_hidden_dim < 4:
            raise ValueError("q_hidden_dim must be at least 4")
        _q_action_hidden_dim(
            cfg.q_hidden_dim, getattr(cfg, "q_action_fusion", "early")
        )
        q_action_coordinates = str(getattr(
            cfg, "q_action_coordinates", "absolute"
        ))
        if q_action_coordinates not in FASTSAC_Q_ACTION_COORDINATES:
            raise ValueError(
                "q_action_coordinates must be one of "
                f"{FASTSAC_Q_ACTION_COORDINATES}, got "
                f"{q_action_coordinates!r}"
            )
        if not isinstance(getattr(cfg, "q_reference_dueling", False), bool):
            raise ValueError("q_reference_dueling must be a boolean")
        if not isinstance(
            getattr(cfg, "q_condition_on_actuator_state", False), bool
        ):
            raise ValueError("q_condition_on_actuator_state must be a boolean")
        if cfg.q_num_atoms < 2 or not cfg.q_v_min < cfg.q_v_max:
            raise ValueError("distributional Q support must contain at least two atoms")
        if not isinstance(cfg.sac_q_normalize_actions, bool):
            raise ValueError("sac_q_normalize_actions must be a boolean")
        q_action_input_gain = float(
            getattr(cfg, "sac_q_action_input_gain", 1.0)
        )
        if not math.isfinite(q_action_input_gain) or q_action_input_gain <= 0.0:
            raise ValueError(
                "sac_q_action_input_gain must be finite and positive"
            )
        if not isinstance(cfg.sac_clipped_double_q, bool):
            raise ValueError("sac_clipped_double_q must be a boolean")
        if not isinstance(cfg.sac_use_autotune, bool):
            raise ValueError("sac_use_autotune must be a boolean")
        if cfg.teacher_buffer_capacity < 1:
            raise ValueError("teacher_buffer_capacity must be positive")
        _validate_seed_replay_partition(
            cfg.sac_teacher_seed_storage_ratio,
            cfg.sac_teacher_seed_sample_ratio,
            cfg.teacher_buffer_capacity,
        )
        self._teacher_training_replay_device = (
            _resolve_teacher_training_replay_device(
                cfg.teacher_training_replay_device, device
            )
        )
        if cfg.teacher_buffer_snapshot_chunk_rows < 1:
            raise ValueError("teacher_buffer_snapshot_chunk_rows must be positive")
        if cfg.phase in ("train", "finetune") and not bool(
            cfg.sac_replay_raw_observations
        ):
            raise ValueError(
                "Current FastSAC requires sac_replay_raw_observations=true so "
                "replay coordinates remain valid while VecNorm changes."
            )
        replay_filename = os.fspath(cfg.teacher_buffer_filename)
        if (
            replay_filename in ("", ".", "..")
            or os.path.basename(replay_filename) != replay_filename
        ):
            raise ValueError(
                "teacher_buffer_filename must be a plain filename without a "
                "directory; use teacher_buffer_path to select an input path"
            )
        command_key = "command_" if observation_spec.get("command_", None) is not None else CMD_KEY
        self.q_critic_keys = [OBS_PRIV_KEY, OBS_KEY, command_key]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            self.q_critic_keys.append(OBJECT_KEY)
        self.q_actor_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        critic_dim = sum(observation_spec[k].shape[-1] for k in self.q_critic_keys)
        actor_dim = sum(observation_spec[k].shape[-1] for k in (VEL_CMD_KEY, OBS_KEY)) + cfg.latent_dim
        self._q_actuator_context_metadata_value = (
            self._resolve_q_actuator_context_metadata()
        )
        self._q_actuator_context_dim = int(
            self._q_actuator_context_metadata_value.get("dimension", 0)
        )
        q_input_dim = critic_dim + self._q_actuator_context_dim
        self.qnet = _build_isolated_q_network(
            q_input_dim, self.action_dim, cfg.q_hidden_dim, cfg.q_num_atoms,
            cfg.q_v_min, cfg.q_v_max, cfg.q_layer_norm, device, cfg.q_seed,
            getattr(cfg, "q_action_fusion", "early"),
            getattr(cfg, "q_reference_dueling", False),
        )
        self.qnet_target = copy.deepcopy(self.qnet).requires_grad_(False)
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95), fused=str(device).startswith("cuda"),
        )
        self._q_actor_dim = actor_dim
        self._q_critic_dim = critic_dim
        self._q_input_dim = q_input_dim
        self.q_rng = torch.Generator(device=device).manual_seed(int(cfg.q_seed))
        self.sac_action_rng = torch.Generator(device=device).manual_seed(
            int(cfg.q_seed) + 1
        )
        # Environment behavior has its own stream: changing the number of SAC
        # gradient samples must not change the action sequence collected into
        # online replay, and collection must not advance environment/global RNG.
        self.sac_rollout_rng = torch.Generator(device=device).manual_seed(
            int(cfg.q_seed) + 2
        )
        self.teacher_replay = None
        self.teacher_replay_id = str(uuid.uuid4())
        self.actor_backend = FASTSAC_ACTOR_BACKEND
        self._loaded_checkpoint_phase = None
        self._loaded_teacher_replay_metadata = None
        self._teacher_replay_extra_shapes = {}
        self._truncation_final_batches = []
        self._rollout_q_actuator_contexts = []
        self._last_truncation_finals_used = 0
        self.q_update_count = 0
        # Shared lifecycle state must exist for every actor backend.  The
        # BC-DAgger adapter returns early from _configure_actor_backend and
        # therefore must not depend on the native Stage-1 setup below to make
        # checkpoint serialization well-defined.
        self._teacher_n_step_accumulator = None
        self._rollout_final_batch = None
        self._teacher_export_started = False
        self._teacher_export_start_seen = None
        self.sac_environment_steps = 0
        self.sac_rollout_count = 0
        self.sac_update_count = 0
        self.sac_actor_update_count = 0
        self.sac_alpha_update_count = 0
        self._configure_actor_backend()

    def _q_conditions_on_actuator_state(self) -> bool:
        return bool(getattr(self.cfg, "q_condition_on_actuator_state", False))

    def _resolve_q_actuator_context_metadata(self) -> dict:
        if not self._q_conditions_on_actuator_state():
            return {"enabled": False}
        manager = getattr(self.env, "action_manager", None)
        required = ("min_delay", "max_delay", "alpha_range", "delay", "alpha")
        missing = [name for name in required if not hasattr(manager, name)]
        if missing:
            raise ValueError(
                "q_condition_on_actuator_state requires a delayed JointPosition "
                f"action manager; missing attributes {missing}"
            )
        delay_min = manager.min_delay
        delay_max = manager.max_delay
        if (
            isinstance(delay_min, bool)
            or isinstance(delay_max, bool)
            or int(delay_min) != delay_min
            or int(delay_max) != delay_max
            or int(delay_min) > int(delay_max)
        ):
            raise ValueError(
                "q_condition_on_actuator_state requires ordered integer delay bounds"
            )
        alpha_range = manager.alpha_range
        if not isinstance(alpha_range, (list, tuple)) or len(alpha_range) != 2:
            raise ValueError(
                "q_condition_on_actuator_state requires a two-value alpha_range"
            )
        alpha_low, alpha_high = (float(value) for value in alpha_range)
        if (
            not math.isfinite(alpha_low)
            or not math.isfinite(alpha_high)
            or alpha_low > alpha_high
        ):
            raise ValueError(
                "q_condition_on_actuator_state requires finite ordered alpha bounds"
            )
        delay_min = int(delay_min)
        delay_max = int(delay_max)
        return _normalize_q_actuator_context_metadata({
            "enabled": True,
            "semantics": FASTSAC_Q_ACTUATOR_CONTEXT_SEMANTICS,
            "dimension": delay_max - delay_min + 2,
            "delay_range": [delay_min, delay_max],
            "alpha_range": [alpha_low, alpha_high],
        })

    @torch.no_grad()
    def _encode_q_actuator_context(
        self,
        delay: torch.Tensor,
        alpha: torch.Tensor,
        *,
        validate_values: bool = True,
    ) -> torch.Tensor:
        metadata = self._q_actuator_context_metadata_value
        if not metadata["enabled"]:
            raise RuntimeError("Q actuator context is disabled")
        if delay.ndim < 1 or delay.shape[-1] != 1:
            raise ValueError(
                f"Actuator delay must end in one value, got {tuple(delay.shape)}"
            )
        if alpha.shape != delay.shape:
            raise ValueError(
                "Actuator delay and alpha shapes must match, got "
                f"{tuple(delay.shape)} and {tuple(alpha.shape)}"
            )
        if delay.device != alpha.device:
            raise ValueError("Actuator delay and alpha must share one device")
        delay_values = delay.squeeze(-1)
        delay_min, delay_max = metadata["delay_range"]
        alpha_low, alpha_high = metadata["alpha_range"]
        if validate_values:
            # Validation is useful for tests, custom collectors, and imported
            # data, but each tensor predicate synchronizes CUDA. The production
            # capture path trusts the already validated ActionManager sampler.
            if delay_values.is_floating_point() and not torch.equal(
                delay_values, delay_values.round()
            ):
                raise ValueError("Actuator delay values must be integral")
            if ((delay_values < delay_min) | (delay_values > delay_max)).any():
                raise ValueError(
                    "Actuator delay is outside the checkpointed context range "
                    f"[{delay_min}, {delay_max}]"
                )
            if not torch.isfinite(alpha).all():
                raise ValueError("Actuator alpha contains non-finite values")
            if ((alpha < alpha_low) | (alpha > alpha_high)).any():
                raise ValueError(
                    "Actuator alpha is outside the checkpointed context range "
                    f"[{alpha_low}, {alpha_high}]"
                )
        delay_values = delay_values.long()
        delay_one_hot = F.one_hot(
            delay_values - delay_min,
            num_classes=delay_max - delay_min + 1,
        ).to(dtype=torch.float32)
        if alpha_high == alpha_low:
            alpha_centered = torch.zeros_like(alpha, dtype=torch.float32)
        else:
            alpha_centered = (
                2.0 * (alpha.float() - alpha_low) / (alpha_high - alpha_low) - 1.0
            )
        context = torch.cat((delay_one_hot, alpha_centered), dim=-1)
        if context.shape[-1] != self._q_actuator_context_dim:
            raise RuntimeError("Encoded Q actuator-context dimension is inconsistent")
        return context

    @torch.no_grad()
    def capture_q_actuator_context(self) -> torch.Tensor | None:
        """Snapshot Q-only actuator state before step/reset mutates the manager."""
        if not self._q_conditions_on_actuator_state():
            return None
        manager = self.env.action_manager
        return self._encode_q_actuator_context(
            manager.delay, manager.alpha, validate_values=False
        ).detach().clone()

    def record_rollout_q_actuator_context(
        self, context: torch.Tensor | None
    ) -> None:
        """Retain a pre-step context for non-interleaved Stage-2 replay."""
        if not self._q_conditions_on_actuator_state():
            if context is not None:
                raise ValueError("Received actuator context while Q conditioning is disabled")
            return
        if context is None:
            raise ValueError("Enabled Q actuator conditioning requires pre-step context")
        self._rollout_q_actuator_contexts.append(context.detach().clone())

    def _transition_q_actuator_context(
        self,
        context: torch.Tensor | None,
        row_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self._q_conditions_on_actuator_state():
            if context is not None:
                raise ValueError(
                    "Received transition actuator context while conditioning is disabled"
                )
            return None
        if context is None:
            raise ValueError(
                "Enabled Q actuator conditioning requires a pre-step transition context"
            )
        expected = (int(row_count), self._q_actuator_context_dim)
        if tuple(context.shape) != expected:
            raise ValueError(
                f"Transition actuator context has shape {tuple(context.shape)}, "
                f"expected {expected}"
            )
        if context.device != device:
            raise ValueError(
                f"Transition actuator context is on {context.device}, expected {device}"
            )
        return context.detach()

    def _consume_rollout_q_actuator_contexts(
        self, num_steps: int
    ) -> torch.Tensor | None:
        contexts = getattr(self, "_rollout_q_actuator_contexts", [])
        self._rollout_q_actuator_contexts = []
        if not self._q_conditions_on_actuator_state():
            if contexts:
                raise RuntimeError("Captured actuator contexts while conditioning is disabled")
            return None
        if len(contexts) != int(num_steps):
            raise RuntimeError(
                "FastSAC rollout actuator-context count does not match rollout length: "
                f"got {len(contexts)}, expected {int(num_steps)}"
            )
        return torch.stack(contexts, dim=1)

    @staticmethod
    def _raw_replay_key(key):
        if isinstance(key, tuple):
            return (FASTSAC_RAW_OBSERVATION_ROOT, *key)
        return (FASTSAC_RAW_OBSERVATION_ROOT, key)

    def _normalize_executable_action(self, action: torch.Tensor) -> torch.Tensor:
        """Affinely map executable VAIC actions into the closed unit box."""
        action_low = self._fastsac_q_action_low.to(
            device=action.device, dtype=action.dtype
        )
        action_scale = self._fastsac_q_action_scale.to(
            device=action.device, dtype=action.dtype
        )
        normalized = (action - action_low) / action_scale - 1.0
        return normalized.clamp(-1.0, 1.0)

    def _q_uses_reference_residual(self) -> bool:
        return str(getattr(
            self.cfg, "q_action_coordinates", "absolute"
        )) == "reference_residual"

    def _q_uses_reference_dueling(self) -> bool:
        return bool(getattr(self.cfg, "q_reference_dueling", False))

    def _q_requires_reference_actions(self) -> bool:
        return (
            self._q_uses_reference_residual()
            or self._q_uses_reference_dueling()
        )

    def _q_action_input(
        self,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the action coordinates consumed by Q1/Q2 only.

        Replay and environment actions intentionally remain in executable VAIC
        coordinates. Absolute normalization clamps only as a numerical guard:
        valid actor/replay actions already lie in the configured executable
        interval. Reference-residual coordinates deliberately do not clamp;
        their state-dependent valid interval may extend to roughly [-2, 2].
        """
        coordinates = str(getattr(
            self.cfg, "q_action_coordinates", "absolute"
        ))
        if coordinates == "reference_residual":
            if reference_action is None:
                raise ValueError(
                    "reference_residual Q coordinates require the frame's "
                    "reference_action"
                )
            if reference_action.shape != action.shape:
                raise ValueError(
                    "FastSAC Q action/reference shapes must match, got "
                    f"{tuple(action.shape)} and "
                    f"{tuple(reference_action.shape)}"
                )
            action_scale = self._fastsac_q_action_scale.to(
                device=action.device, dtype=action.dtype
            )
            q_action = (action - reference_action) / action_scale
        elif coordinates == "absolute":
            if bool(getattr(self.cfg, "sac_q_normalize_actions", False)):
                q_action = self._normalize_executable_action(action)
            else:
                q_action = action
        else:
            raise ValueError(
                "q_action_coordinates must be one of "
                f"{FASTSAC_Q_ACTION_COORDINATES}, got {coordinates!r}"
            )
        gain = float(getattr(self.cfg, "sac_q_action_input_gain", 1.0))
        # Keep the default path bit-for-bit identical, including returning the
        # original raw-action tensor when normalization is disabled.
        if gain == 1.0:
            return q_action
        return q_action * gain

    def _q_forward(
        self,
        qnet,
        observations,
        actions,
        reference_actions=None,
        actuator_context=None,
    ):
        observations = self._q_observation_input(
            observations, actuator_context
        )
        q_actions = self._q_action_input(actions, reference_actions)
        if not self._q_uses_reference_dueling():
            return qnet(observations, q_actions)
        if reference_actions is None:
            raise ValueError(
                "reference-dueling Q requires the frame's reference action"
            )
        q_reference_actions = self._q_action_input(
            reference_actions, reference_actions
        )
        return qnet(observations, q_actions, q_reference_actions)

    def _q_projection(
        self,
        qnet,
        observations,
        actions,
        reward,
        bootstrap,
        discount,
        reference_actions=None,
        actuator_context=None,
    ):
        observations = self._q_observation_input(
            observations, actuator_context
        )
        q_actions = self._q_action_input(actions, reference_actions)
        if not self._q_uses_reference_dueling():
            return qnet.projection(
                observations, q_actions, reward, bootstrap, discount
            )
        if reference_actions is None:
            raise ValueError(
                "reference-dueling Q requires the frame's reference action"
            )
        q_reference_actions = self._q_action_input(
            reference_actions, reference_actions
        )
        return qnet.projection(
            observations,
            q_actions,
            reward,
            bootstrap,
            discount,
            q_reference_actions,
        )

    def _q_observation_input(
        self,
        observations: torch.Tensor,
        actuator_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Append fixed-coordinate actuator state only at the Q boundary."""
        if not self._q_conditions_on_actuator_state():
            if actuator_context is not None:
                raise ValueError(
                    "Received Q actuator context while conditioning is disabled"
                )
            return observations
        if actuator_context is None:
            raise ValueError(
                "q_condition_on_actuator_state=true requires actuator context "
                "for every Q call"
            )
        expected = (*observations.shape[:-1], self._q_actuator_context_dim)
        if tuple(actuator_context.shape) != expected:
            raise ValueError(
                "Q actuator-context shape does not match observations: got "
                f"{tuple(actuator_context.shape)}, expected {expected}"
            )
        if actuator_context.device != observations.device:
            raise ValueError("Q observations and actuator context must share a device")
        return torch.cat(
            (observations, actuator_context.to(dtype=observations.dtype)), dim=-1
        )

    def configure_replay_vecnorm(self, vecnorm):
        """Attach VecNorm and validate every normalized replay source alias."""
        object.__setattr__(self, "_replay_vecnorm", vecnorm)
        self._replay_vecnorm_keys = set(vecnorm.in_keys)
        self._replay_vecnorm_fingerprint = _vecnorm_state_fingerprint(vecnorm)
        replay_sources = set(self.q_critic_keys)
        replay_sources.update((VEL_CMD_KEY, OBS_KEY))
        if hasattr(self, "height_encoder"):
            replay_sources.add(HEIGHT_KEY)
        required_raw = replay_sources.intersection(self._replay_vecnorm_keys)
        missing = [
            key for key in required_raw
            if self.observation_spec.get(self._raw_replay_key(key), None) is None
        ]
        if missing:
            raise KeyError(
                "FastSAC raw replay aliases are missing for normalized "
                f"observations: {sorted(missing)}"
            )

    def _replay_source(self, td, key):
        """Read raw pre-VecNorm data for normalized keys at collection time."""
        if key not in getattr(self, "_replay_vecnorm_keys", set()):
            return td[key]
        raw_key = self._raw_replay_key(key)
        if raw_key not in td.keys(True, True):
            raise KeyError(
                f"FastSAC replay is missing raw observation alias {raw_key!r}"
            )
        return td[raw_key]

    def _cat_replay_sources(self, td, keys):
        return torch.cat([self._replay_source(td, key) for key in keys], dim=-1)

    def _vecnorm_snapshot(self):
        vecnorm = getattr(self, "_replay_vecnorm", None)
        if vecnorm is None:
            return None
        # One immutable-by-convention snapshot is shared by all fields in one
        # replay minibatch. Reading these tensors does not update VecNorm.
        return vecnorm.loc, vecnorm.scale

    def _normalize_replay_value(self, key, value, snapshot):
        if snapshot is None or key not in self._replay_vecnorm_keys:
            return value
        loc, scale = snapshot
        eps = float(self._replay_vecnorm.eps)
        return (value - loc[key]) / scale[key].clamp_min(eps)

    def _normalize_replay_flat(self, value, keys, widths, snapshot):
        if snapshot is None:
            return value
        chunks = []
        offset = 0
        for key, width in zip(keys, widths):
            chunk = value[..., offset:offset + width]
            chunks.append(self._normalize_replay_value(key, chunk, snapshot))
            offset += width
        if offset != value.shape[-1]:
            raise RuntimeError(
                f"FastSAC replay split consumed {offset} of {value.shape[-1]} values"
            )
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def _scalarize_sac_reward(reward):
        # Preserve all environment term/group computations and relative weights;
        # only remove PPO's equal-group averaging from the scalar SAC target.
        return reward.sum(dim=-1)

    def _normalized_action_log_prob(self, physical_log_prob):
        """Express log pi in the configured entropy reference coordinates.

        Actor distributions always score executable physical actions.  The
        entropy objective may use a different fixed unit, and therefore adds
        only that reference coordinate's log determinant.  In particular, the
        BC-DAgger safety clip is not implicitly an entropy normalization scale.
        """
        return physical_log_prob + float(getattr(
            self,
            "_fastsac_entropy_reference_log_scale_sum",
            getattr(self, "_fastsac_action_log_scale_sum", 0.0),
        ))

    def _configure_actor_backend(self):
        raise NotImplementedError

    def requires_training_replay(self):
        """Whether every training rank needs a replay even without H5 export."""
        return False

    def requires_value_bootstrap(self):
        return False

    def _optimizer_manifests(self, optimizers):
        parameter_names = {id(parameter): name for name, parameter in self.named_parameters()}
        manifests = {}
        for optimizer_name, optimizer in optimizers.items():
            groups = []
            for group in optimizer.param_groups:
                names = []
                for parameter in group["params"]:
                    name = parameter_names.get(id(parameter))
                    if name is None:
                        raise RuntimeError(
                            f"Optimizer {optimizer_name!r} contains a parameter that "
                            "is not registered on the policy."
                        )
                    names.append(name)
                groups.append(names)
            manifests[optimizer_name] = groups
        return manifests

    def _optimizer_resume_signature(self, optimizer_names):
        return {
            "policy_family": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
                if self.cfg.phase == "train"
                else FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
            "phase": str(self.cfg.phase),
            "actor_backend": str(self.actor_backend),
            "optimizer_names": list(optimizer_names),
        }

    @staticmethod
    def _normalize_optimizer_resume_signature(signature):
        """Map pre-rename true-FastSAC class names to a stable family id."""
        if not isinstance(signature, dict):
            return signature
        normalized = copy.deepcopy(signature)
        legacy_class = normalized.pop("policy_class", None)
        legacy_families = {
            "active_adaptation.learning.ppo.ppo_fastsac_vel.HOIFastSACVEL": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.ppo_fastsac_vel.FastSACVelFinetune": (
                FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.fastsac_vel.FastSACVEL": (
                FASTSAC_TEACHER_TRAINING_ALGORITHM
            ),
            "active_adaptation.learning.ppo.fastsac_vel.FastSACVelFinetune": (
                FASTSAC_STUDENT_TRAINING_ALGORITHM
            ),
        }
        if legacy_class is not None:
            family = legacy_families.get(legacy_class)
            if family is None:
                # In particular, do not treat the removed PPOFastSACVEL hybrid
                # as a true FastSAC checkpoint.
                normalized["policy_class"] = legacy_class
            else:
                normalized["policy_family"] = family
        return normalized

    def _optimizer_resume_state(self):
        optimizers = self._optimizer_registry()
        return {
            "version": 1,
            "signature": self._optimizer_resume_signature(optimizers.keys()),
            "parameter_manifests": self._optimizer_manifests(optimizers),
            "optimizer_states": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "num_updates": int(self.num_updates),
        }

    def _restore_optimizer_resume_state(self, state_dict):
        checkpoint_phase = state_dict.get("last_phase")
        if checkpoint_phase != self.cfg.phase:
            logging.info(
                "Starting phase %s from phase %s weights; optimizer moments are "
                "intentionally reinitialized.",
                self.cfg.phase, checkpoint_phase,
            )
            return False

        resume_state = state_dict.get("optimizer_resume_state")
        if resume_state is None:
            logging.warning(
                "This same-phase checkpoint predates optimizer-state saving; "
                "continuing with freshly initialized Adam/AdamW moments."
            )
            return False
        if int(resume_state.get("version", 0)) != 1:
            raise ValueError(
                "Unsupported FastSAC optimizer resume-state version: "
                f"{resume_state.get('version')!r}"
            )

        optimizers = self._optimizer_registry()
        expected_signature = self._optimizer_resume_signature(optimizers.keys())
        actual_signature = self._normalize_optimizer_resume_signature(
            resume_state.get("signature")
        )
        if actual_signature != expected_signature:
            logging.warning(
                "Skipping optimizer moments because the resume signature changed: "
                "checkpoint=%s, current=%s",
                actual_signature, expected_signature,
            )
            return False

        expected_manifests = self._optimizer_manifests(optimizers)
        actual_manifests = resume_state.get("parameter_manifests")
        if actual_manifests != expected_manifests:
            raise ValueError(
                "FastSAC optimizer parameter topology does not match the checkpoint; "
                "refusing to attach Adam moments to different parameters."
            )
        optimizer_states = resume_state.get("optimizer_states", {})
        if set(optimizer_states) != set(optimizers):
            raise ValueError(
                "FastSAC checkpoint optimizer set does not match the current policy: "
                f"checkpoint={sorted(optimizer_states)}, current={sorted(optimizers)}"
            )
        for name, optimizer in optimizers.items():
            try:
                optimizer.load_state_dict(optimizer_states[name])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to restore optimizer {name!r} for same-phase resume."
                ) from exc
        self.num_updates = int(resume_state.get("num_updates", 0))
        logging.info(
            "Restored %d optimizer states and num_updates=%d for same-phase resume.",
            len(optimizers), self.num_updates,
        )
        return True

    def _teacher_replay_checkpoint_metadata(self):
        if self.teacher_replay is None:
            return None
        return self.teacher_replay.checkpoint_metadata()

    def load_state_dict(self, state_dict, strict=True):
        failed = super().load_state_dict(state_dict, strict)
        if "q_rng_state" in state_dict:
            self.q_rng.set_state(state_dict["q_rng_state"].cpu())
        if "sac_action_rng_state" in state_dict:
            self.sac_action_rng.set_state(
                state_dict["sac_action_rng_state"].cpu()
            )
        if "sac_rollout_rng_state" in state_dict:
            self.sac_rollout_rng.set_state(
                state_dict["sac_rollout_rng_state"].cpu()
            )
        self.q_update_count = int(state_dict.get("q_update_count", 0))
        if "teacher_replay_id" in state_dict:
            self.teacher_replay_id = str(state_dict["teacher_replay_id"])
        self._loaded_checkpoint_phase = state_dict.get("last_phase")
        replay_metadata = state_dict.get("teacher_replay_state")
        self._loaded_teacher_replay_metadata = copy.deepcopy(replay_metadata)
        self._restore_optimizer_resume_state(state_dict)
        return failed

    def configure_teacher_replay(self, path, restore_path=None):
        if not self.cfg.save_teacher_buffer and not self.requires_training_replay():
            return
        self.teacher_replay = TeacherReplayBuffer(
            path, self.cfg.teacher_buffer_capacity, self._q_actor_dim,
            self._q_critic_dim, self.action_dim, self.cfg.teacher_buffer_seed,
            device=self.device,
            snapshot_chunk_rows=self.cfg.teacher_buffer_snapshot_chunk_rows,
            replay_id=self.teacher_replay_id,
            actor_backend=self.actor_backend,
            actor_obs_keys=self.q_actor_keys,
            critic_obs_keys=self.q_critic_keys,
            extra_shapes=self._teacher_replay_extra_shapes,
            q_action_fusion=getattr(self.cfg, "q_action_fusion", "early"),
            q_action_hidden_dim=_q_action_hidden_dim(
                self.cfg.q_hidden_dim,
                getattr(self.cfg, "q_action_fusion", "early"),
            ),
            q_action_coordinates=getattr(
                self.cfg, "q_action_coordinates", "absolute"
            ),
            q_reference_dueling=getattr(
                self.cfg, "q_reference_dueling", False
            ),
            q_actuator_context=getattr(
                self, "_q_actuator_context_metadata_value", {"enabled": False}
            ),
        )
        if restore_path is not None and self._loaded_checkpoint_phase is None:
            raise ValueError(
                "teacher_buffer_path can restore a FIFO only together with a "
                "same-stage checkpoint_path."
            )
        if (
            restore_path is not None
            and self._loaded_checkpoint_phase != self.cfg.phase
        ):
            logging.warning(
                "Ignoring teacher replay from phase %s while starting phase %s.",
                self._loaded_checkpoint_phase, self.cfg.phase,
            )
            restore_path = None

        expected = self._loaded_teacher_replay_metadata
        expected_size = None if expected is None else expected.get("size")
        replay_required = (
            self._loaded_checkpoint_phase == self.cfg.phase
            and expected_size is not None
            and int(expected_size) > 0
        )
        if (
            restore_path is not None
            and self._loaded_checkpoint_phase == self.cfg.phase
            and expected is None
            and not replay_required
        ):
            # The helper can discover a run's final H5 while the user selected
            # an older checkpoint from that run. Without a checkpoint replay
            # manifest there is no way to prove that the snapshots are paired;
            # starting empty is safer than leaking future transitions.
            logging.warning(
                "Ignoring unpaired teacher replay %s because this checkpoint "
                "contains no exact replay manifest.",
                restore_path,
            )
            restore_path = None
        if restore_path is None and replay_required:
            raise FileNotFoundError(
                "Same-phase FastSAC resume requires the teacher replay H5 saved "
                "with this checkpoint. Set algo.teacher_buffer_path explicitly "
                "or keep teacher_replay_buffer.h5 beside the checkpoint."
            )
        if restore_path is not None:
            restored = self.teacher_replay.restore(restore_path, expected)
            logging.info(
                "Restored %d teacher replay rows (seen=%d) from %s onto %s.",
                restored, self.teacher_replay.seen, restore_path, self.device,
            )
        logging.info(
            "Teacher replay FIFO: capacity=%d, device=%s, estimated=%.2f GiB",
            self.teacher_replay.capacity, self.teacher_replay.device,
            self.teacher_replay.estimated_bytes / (1024 ** 3),
        )

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        if self.teacher_replay is None:
            return None
        return self.teacher_replay.snapshot(iteration, checkpoint_name)

    def _cat(self, td, keys):
        return torch.cat([td[k] for k in keys], dim=-1)

    def begin_transition_collection(self):
        """Clear sparse pre-reset observations at the start of every rollout."""
        self._truncation_final_batches.clear()
        self._rollout_q_actuator_contexts = []


class FastSACVEL(_FastSACVAICBase):
    """True FastSAC teacher with VAIC observations and student distillation.

    The private base reuses VAIC module construction and checkpoint plumbing.
    Teacher optimization contains no PPO/GAE/value update.
    """

    def _vaic_action_bounds(self):
        manager = self.env.action_manager
        joint_ids = torch.as_tensor(
            manager.joint_ids, device=self.device, dtype=torch.long
        )
        limits = manager.asset.data.soft_joint_pos_limits[
            0, joint_ids
        ].detach().to(self.device, torch.float32)
        default = manager.default_joint_pos[
            0, joint_ids
        ].detach().to(self.device, torch.float32)
        scaling = manager.action_scaling.detach().to(self.device, torch.float32)
        if scaling.shape != default.shape or not torch.all(scaling > 0):
            raise ValueError(
                "FastSAC requires one strictly positive action scale per VAIC joint"
            )

        # JointPosition executes default + randomized_offset + scale * action.
        # Use the intersection of the allowed intervals over the configured
        # random_joint_offset range.  This makes every actor sample executable in
        # every environment rather than relying on the later physical clamp.
        offset_low = torch.zeros_like(default)
        offset_high = torch.zeros_like(default)
        randomizations = getattr(self.env, "randomizations", {})
        offset_randomizer = (
            randomizations.get("random_joint_offset")
            if hasattr(randomizations, "get")
            else None
        )
        if offset_randomizer is not None:
            random_joint_ids = torch.as_tensor(
                offset_randomizer.joint_ids, dtype=torch.long
            ).detach().cpu().tolist()
            random_ranges = torch.as_tensor(
                offset_randomizer.offset_range,
                device=self.device,
                dtype=torch.float32,
            )
            if random_ranges.ndim == 3:
                configured_low = random_ranges.amin(dim=(0, 2))
                configured_high = random_ranges.amax(dim=(0, 2))
            elif random_ranges.ndim == 2:
                configured_low = random_ranges.amin(dim=1)
                configured_high = random_ranges.amax(dim=1)
            else:
                raise ValueError(
                    "random_joint_offset range must have shape [J, 2] or [N, J, 2]"
                )
            if configured_low.numel() != len(random_joint_ids):
                raise ValueError(
                    "random_joint_offset joint ids and configured ranges disagree"
                )
            range_by_joint = {
                int(asset_joint_id): (configured_low[index], configured_high[index])
                for index, asset_joint_id in enumerate(random_joint_ids)
            }
            for index, asset_joint_id in enumerate(joint_ids.detach().cpu().tolist()):
                if asset_joint_id in range_by_joint:
                    low, high = range_by_joint[asset_joint_id]
                    offset_low[index] = low
                    offset_high[index] = high

        action_low = (limits[:, 0] - default - offset_low) / scaling
        action_high = (limits[:, 1] - default - offset_high) / scaling
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise ValueError("VAIC joint limits produced non-finite FastSAC action bounds")
        if not torch.all(action_high > action_low):
            raise ValueError(
                "VAIC joint limits and joint-offset randomization produced an empty "
                "FastSAC action interval"
            )
        if not torch.all((action_low < 0.0) & (action_high > 0.0)):
            raise ValueError(
                "VAIC raw action zero must be strictly inside every executable "
                "joint interval across the configured joint-offset range"
            )
        self._fastsac_joint_offset_low = offset_low.detach().cpu().tolist()
        self._fastsac_joint_offset_high = offset_high.detach().cpu().tolist()
        return action_low, action_high

    def _actor_backend_metadata(self):
        if self.actor_backend == FASTSAC_BC_DAGGER_ACTOR_BACKEND:
            deterministic_rollout = bool(self.cfg.sac_deterministic_rollout)
            metadata = {
                "action_parameterization": (
                    "bc_dagger_clipped_mean_deterministic_rollout_"
                    "dedicated_tanh_gaussian_learning_v2"
                    if deterministic_rollout
                    else
                    "bc_dagger_dedicated_tanh_gaussian_train_behavior_"
                    "deterministic_clipped_mean_eval_v3"
                ),
                "source_actor_backend": BC_DAGGER_ACTOR_BACKEND,
                "student_input_dim": self._q_actor_dim,
                "action_dim": self.action_dim,
                "log_std_parameter": "bc_dagger_sac_adapter.log_std",
                "initial_action_std": float(
                    self.cfg.sac_bc_initial_action_std
                ),
                "initial_log_std": math.log(
                    float(self.cfg.sac_bc_initial_action_std)
                    / float(self.cfg.sac_bc_action_clip)
                ),
                "log_std_min": self.cfg.sac_bc_log_std_min,
                "log_std_max": self.cfg.sac_bc_log_std_max,
                "action_bound_source": "bc_dagger_action_clip",
                "action_clip": float(self.cfg.sac_bc_action_clip),
                "rollout_behavior": (
                    "deterministic_clipped_bc_mean"
                    if deterministic_rollout
                    else
                    "train_stochastic_dedicated_tanh_gaussian_"
                    "eval_deterministic_clipped_bc_mean"
                ),
                "perception_frozen": bool(self.cfg.sac_freeze_perception),
                "action_low": self._fastsac_action_low,
                "action_high": self._fastsac_action_high,
                "action_log_scale_sum": self._fastsac_action_log_scale_sum,
                "entropy_reference_coordinates": (
                    "raw_action_divide_fixed_reference_scale_v1"
                ),
                "entropy_reference_scale": float(
                    self.cfg.sac_entropy_reference_scale
                ),
                "entropy_reference_log_scale_sum": float(
                    self._fastsac_entropy_reference_log_scale_sum
                ),
                "q_only_target_policy": (
                    "deterministic_behavior_mean_hard_target_v2"
                    if deterministic_rollout
                    else "frozen_stochastic_behavior_hard_target_v2"
                ),
                "q_only_alpha_semantics": (
                    "effective_zero_until_actual_actor_release_then_linear_"
                    "ramp_shared_by_q_actor_and_dual_v2"
                ),
                "alpha_ramp_q_updates": int(
                    self.cfg.sac_alpha_ramp_q_updates
                ),
                "joint_names": list(self.joint_names),
            }
            if not deterministic_rollout:
                metadata["rollout_rng"] = (
                    "dedicated_checkpointed_torch_generator_q_seed_plus_2_v1"
                )
            return metadata
        return {
            "action_parameterization": FASTSAC_ACTION_PARAMETERIZATION,
            "teacher_input_dim": self._fastsac_teacher_actor_dim,
            "student_input_dim": self._q_actor_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.cfg.fastsac_actor_hidden_dim,
            "log_std_min": self.cfg.fastsac_log_std_min,
            "log_std_max": self.cfg.fastsac_log_std_max,
            "layer_norm": self.cfg.fastsac_actor_layer_norm,
            "teacher_action_center": "reference_pre_tanh",
            "student_action_center": "zero_pre_tanh",
            "action_bound_source": (
                "soft_joint_limits_intersect_random_joint_offset_range"
            ),
            "action_low": self._fastsac_action_low,
            "action_high": self._fastsac_action_high,
            "action_log_scale_sum": self._fastsac_action_log_scale_sum,
            "joint_offset_low": self._fastsac_joint_offset_low,
            "joint_offset_high": self._fastsac_joint_offset_high,
            "joint_names": list(self.joint_names),
        }

    def _q_backend_metadata(self):
        q_action_normalized = bool(self.cfg.sac_q_normalize_actions)
        q_reference_dueling = bool(getattr(
            self.cfg, "q_reference_dueling", False
        ))
        q_action_coordinates = str(getattr(
            self.cfg, "q_action_coordinates", "absolute"
        ))
        q_action_fusion = str(getattr(
            self.cfg, "q_action_fusion", "early"
        ))
        q_action_hidden_dim = _q_action_hidden_dim(
            self.cfg.q_hidden_dim, q_action_fusion
        )
        clipped_double_q = bool(
            getattr(self.cfg, "sac_clipped_double_q", True)
        )
        return {
            "actor_obs_keys": list(self.q_actor_keys),
            "critic_obs_keys": list(self.q_critic_keys),
            "actor_obs_dim": self._q_actor_dim,
            "critic_obs_dim": self._q_critic_dim,
            "q_input_dim": getattr(
                self,
                "_q_input_dim",
                self._q_critic_dim + int(getattr(
                    self, "_q_actuator_context_dim", 0
                )),
            ),
            "q_actuator_context": copy.deepcopy(
                getattr(
                    self,
                    "_q_actuator_context_metadata_value",
                    {"enabled": False},
                )
            ),
            "action_dim": self.action_dim,
            "hidden_dim": self.cfg.q_hidden_dim,
            "q_action_fusion": q_action_fusion,
            "q_action_hidden_dim": q_action_hidden_dim,
            "q_action_fusion_semantics": (
                FASTSAC_Q_EARLY_FUSION_SEMANTICS
                if q_action_fusion == "early"
                else FASTSAC_Q_LATE_FUSION_SEMANTICS
            ),
            "q_reference_dueling": q_reference_dueling,
            "q_architecture_semantics": (
                FASTSAC_Q_REFERENCE_DUELING_ARCHITECTURE_SEMANTICS
                if q_reference_dueling
                else FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS
            ),
            "num_atoms": self.cfg.q_num_atoms,
            "v_min": self.cfg.q_v_min,
            "v_max": self.cfg.q_v_max,
            "layer_norm": self.cfg.q_layer_norm,
            "gamma": self.cfg.gamma,
            "replay_observation_semantics": REPLAY_OBSERVATION_SEMANTICS,
            "reward_scalarization": SAC_REWARD_SCALARIZATION,
            "reward_groups": list(self.reward_groups),
            "truncation_semantics": TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
            "target_entropy_semantics": (
                FASTSAC_BC_DAGGER_TARGET_ENTROPY_SEMANTICS
                if self._uses_bc_dagger_finetune_source()
                else FASTSAC_TARGET_ENTROPY_SEMANTICS
            ),
            "q_action_coordinates": q_action_coordinates,
            "q_action_normalized": q_action_normalized,
            "q_action_input_gain": float(
                getattr(self.cfg, "sac_q_action_input_gain", 1.0)
            ),
            "q_action_semantics": (
                FASTSAC_Q_REFERENCE_RESIDUAL_SEMANTICS
                if q_action_coordinates == "reference_residual"
                else (
                    FASTSAC_Q_ACTION_NORMALIZATION_SEMANTICS
                    if q_action_normalized
                    else FASTSAC_Q_RAW_ACTION_SEMANTICS
                )
            ),
            "clipped_double_q": clipped_double_q,
            "q_backup_semantics": (
                FASTSAC_CLIPPED_DOUBLE_Q_SEMANTICS
                if clipped_double_q
                else FASTSAC_INDEPENDENT_DOUBLE_Q_SEMANTICS
            ),
            "actor_q_reduction": "minimum" if clipped_double_q else "mean",
            "alpha_autotune": bool(
                getattr(self.cfg, "sac_use_autotune", True)
            ),
        }

    def _build_fastsac_actor(
        self, in_keys, input_dim, action_low, action_high, reference_key=None
    ):
        core = FastSACActor(
            input_dim=input_dim,
            action_dim=self.action_dim,
            hidden_dim=self.cfg.fastsac_actor_hidden_dim,
            log_std_min=self.cfg.fastsac_log_std_min,
            log_std_max=self.cfg.fastsac_log_std_max,
            action_low=action_low,
            action_high=action_high,
            layer_norm=self.cfg.fastsac_actor_layer_norm,
            reference_centered=reference_key is not None,
        ).to(self.device)
        core_in_keys = ["_fastsac_actor_input"]
        if reference_key is not None:
            core_in_keys.append(reference_key)
        params = Seq(
            CatTensors(in_keys, "_fastsac_actor_input", del_keys=False, sort=False),
            Mod(
                core,
                core_in_keys,
                ["loc", "scale", FASTSAC_DETERMINISTIC_ACTION_KEY],
            ),
        )
        return ProbabilisticActor(
            module=params,
            in_keys=self.dist_keys,
            out_keys=[ACTION_KEY],
            distribution_class=self.dist_cls,
            # FastSAC scores retained pre-tanh samples in its update path.
            # Rollout log-probability is unused and inverse-tanh scoring can be
            # numerically unstable at an action bound.
            return_log_prob=False,
        ).to(self.device)

    def _teacher_actor_core(self) -> FastSACActor:
        cores = [
            module for module in self.actor.modules()
            if isinstance(module, FastSACActor)
        ]
        if len(cores) != 1:
            raise RuntimeError(
                "Expected exactly one FastSACActor inside the teacher actor, "
                f"found {len(cores)}"
            )
        return cores[0]

    def _teacher_actor_std_schedule_config(self) -> dict[str, float | int | None]:
        def optional_float(name):
            value = getattr(self.cfg, name, None)
            return None if value is None else float(value)

        reset_q_updates = getattr(
            self.cfg, "sac_teacher_actor_std_reset_q_updates", None
        )
        return {
            "initial_log_std": optional_float("sac_teacher_initial_log_std"),
            "actor_reset_log_std": optional_float(
                "sac_teacher_actor_reset_log_std"
            ),
            "reset_q_updates": (
                None if reset_q_updates is None else int(reset_q_updates)
            ),
        }

    def _teacher_actor_std_schedule_checkpoint_state(self) -> dict:
        return {
            "version": 1,
            "config": self._teacher_actor_std_schedule_config(),
            "reset_applied": bool(self._teacher_actor_std_reset_applied),
            "reset_applied_q_updates": (
                None
                if self._teacher_actor_std_reset_applied_q_updates is None
                else int(self._teacher_actor_std_reset_applied_q_updates)
            ),
        }

    def _validate_teacher_actor_std_schedule_checkpoint(self, state_dict):
        expected_config = self._teacher_actor_std_schedule_config()
        schedule_state = state_dict.get("stage1_actor_std_schedule")
        if schedule_state is None:
            if any(value is not None for value in expected_config.values()):
                raise ValueError(
                    "Stage-1 checkpoint predates the configured teacher actor "
                    "std schedule; restart the run instead of risking a repeated "
                    "or skipped reset."
                )
            return None
        if int(schedule_state.get("version", 0)) != 1:
            raise ValueError(
                "Unsupported Stage-1 teacher actor std schedule checkpoint "
                f"version: {schedule_state.get('version')!r}"
            )
        actual_config = schedule_state.get("config")
        if actual_config != expected_config:
            raise ValueError(
                "Stage-1 teacher actor std schedule config does not match the "
                f"checkpoint: checkpoint={actual_config}, current={expected_config}"
            )
        applied = schedule_state.get("reset_applied", False)
        if not isinstance(applied, bool):
            raise ValueError(
                "Stage-1 teacher actor std schedule reset_applied must be boolean"
            )
        applied_q_updates = schedule_state.get("reset_applied_q_updates")
        configured_reset = expected_config["reset_q_updates"]
        if applied:
            if (
                configured_reset is None
                or applied_q_updates is None
                or int(applied_q_updates) != configured_reset
            ):
                raise ValueError(
                    "Stage-1 teacher actor std checkpoint has inconsistent reset "
                    "application metadata"
                )
        elif applied_q_updates is not None:
            raise ValueError(
                "Stage-1 teacher actor std checkpoint records a reset count "
                "without an applied reset"
            )
        return schedule_state

    def _restore_teacher_actor_std_schedule_checkpoint(
        self, schedule_state
    ) -> None:
        if schedule_state is None:
            self._teacher_actor_std_reset_applied = False
            self._teacher_actor_std_reset_applied_q_updates = None
            self._teacher_actor_std_reset_event_q_updates = None
            return
        applied = bool(schedule_state["reset_applied"])
        applied_q_updates = schedule_state.get("reset_applied_q_updates")
        configured_reset = self._teacher_actor_std_schedule_config()[
            "reset_q_updates"
        ]
        if applied and self.sac_update_count < int(applied_q_updates):
            raise ValueError(
                "Stage-1 teacher actor std reset is marked applied after more Q "
                "updates than the checkpoint contains"
            )
        if (
            not applied
            and configured_reset is not None
            and self.sac_update_count >= configured_reset
        ):
            raise ValueError(
                "Stage-1 checkpoint crossed the teacher actor std reset threshold "
                "without recording the reset"
            )
        self._teacher_actor_std_reset_applied = applied
        self._teacher_actor_std_reset_applied_q_updates = (
            None if applied_q_updates is None else int(applied_q_updates)
        )
        # A checkpoint resume reports persistent applied/current state, not a
        # duplicate one-shot event from a prior process.
        self._teacher_actor_std_reset_event_q_updates = None

    def _set_teacher_actor_log_std(
        self, target_log_std: float, *, clear_optimizer_state: bool
    ) -> float:
        core = self._teacher_actor_core()
        raw_bias = core.reset_log_std_head(target_log_std)
        if clear_optimizer_state:
            optimizer = getattr(self, "sac_teacher_actor_optimizer", None)
            if optimizer is not None:
                # A reset configured after actor learning begins must not be
                # immediately undone by stale Adam moments for the two reset
                # parameters. Other actor/trunk optimizer state is preserved.
                optimizer.state.pop(core.fc_logstd.weight, None)
                optimizer.state.pop(core.fc_logstd.bias, None)
        core.fc_logstd.weight.grad = None
        core.fc_logstd.bias.grad = None
        return raw_bias

    def _maybe_reset_teacher_actor_std_before_q_update(self) -> bool:
        """Apply the Stage-1 std reset once, before its numbered Q update."""
        reset_q_updates = getattr(
            self.cfg, "sac_teacher_actor_std_reset_q_updates", None
        )
        if (
            self.cfg.phase != "train"
            or reset_q_updates is None
            or bool(getattr(
                self, "_teacher_actor_std_reset_applied", False
            ))
        ):
            return False
        reset_q_updates = int(reset_q_updates)
        next_q_update = self.sac_update_count + 1
        if next_q_update < reset_q_updates:
            return False
        if next_q_update > reset_q_updates:
            raise RuntimeError(
                "Stage-1 actor std reset threshold was crossed without applying "
                f"the reset: current_q_updates={self.sac_update_count}, "
                f"configured_reset={reset_q_updates}"
            )
        seed_enabled = float(getattr(
            self.cfg, "sac_teacher_seed_storage_ratio", 0.0
        )) > 0.0
        seed_frozen = bool(getattr(
            getattr(self, "teacher_replay", None), "seed_frozen", False
        ))
        if seed_enabled and not seed_frozen:
            replay = getattr(self, "teacher_replay", None)
            if replay is None or replay.size < replay.capacity:
                raise RuntimeError(
                    "Stage-1 teacher actor std reset cannot preserve broad seed "
                    "data before the replay is full. Increase the reset Q-update "
                    "threshold or disable the seed replay partition."
                )
            if not self._maybe_freeze_teacher_seed_replay():
                raise RuntimeError(
                    "Stage-1 teacher actor std reset failed to freeze the broad "
                    "seed replay partition"
                )
        target_log_std = float(self.cfg.sac_teacher_actor_reset_log_std)
        raw_bias = self._set_teacher_actor_log_std(
            target_log_std, clear_optimizer_state=True
        )
        self._teacher_actor_std_reset_applied = True
        self._teacher_actor_std_reset_applied_q_updates = reset_q_updates
        self._teacher_actor_std_reset_event_q_updates = reset_q_updates
        logging.info(
            "Reset Stage-1 teacher actor log std to %.6f (raw head bias %.6f) "
            "before Q update %d.",
            target_log_std,
            raw_bias,
            reset_q_updates,
        )
        return True

    def _uses_bc_dagger_finetune_source(self) -> bool:
        return (
            getattr(self.cfg, "phase", "finetune") == "finetune"
            and str(getattr(
                self.cfg, "finetune_checkpoint_source", "fastsac"
            )) == "bc_dagger"
        )

    @staticmethod
    def _ppo_actor_core(actor) -> Actor:
        cores = [module for module in actor.modules() if isinstance(module, Actor)]
        if len(cores) != 1:
            raise RuntimeError(
                "Expected exactly one VAIC PPO Actor core, found "
                f"{len(cores)}"
            )
        return cores[0]

    def _configure_bc_dagger_actor_backend(self):
        """Retain exact BC behavior and add separate SAC stochastic state."""
        replay_incompatible = {
            "q_action_coordinates": str(getattr(
                self.cfg, "q_action_coordinates", "absolute"
            )) != "absolute",
            "q_reference_dueling": bool(getattr(
                self.cfg, "q_reference_dueling", False
            )),
            "q_condition_on_actuator_state": bool(getattr(
                self.cfg, "q_condition_on_actuator_state", False
            )),
        }
        unsupported = [
            name for name, enabled in replay_incompatible.items() if enabled
        ]
        if unsupported:
            raise ValueError(
                "BC-DAgger offline replay stores absolute actions without "
                "reference-action or actuator-context fields; incompatible "
                f"Stage-2 Q settings: {unsupported}"
            )
        if (
            bool(getattr(self.cfg, "load_pretrained_q", True))
            and bool(getattr(self.cfg, "sac_q_normalize_actions", True))
        ):
            raise ValueError(
                "BC-DAgger Q transfer requires sac_q_normalize_actions=false"
            )
        if bool(getattr(self.cfg, "load_pretrained_q", True)):
            pretrained_q_incompatible = {
                "q_action_fusion": str(getattr(
                    self.cfg, "q_action_fusion", "early"
                )) != "early",
                "sac_q_action_input_gain": not math.isclose(
                    float(getattr(
                        self.cfg, "sac_q_action_input_gain", 1.0
                    )),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            }
            incompatible = [
                name
                for name, enabled in pretrained_q_incompatible.items()
                if enabled
            ]
            if incompatible:
                raise ValueError(
                    "Pretrained BC-DAgger Q requires its original early-fusion "
                    "raw-action input with unit gain; incompatible settings: "
                    f"{incompatible}"
                )
        action_clip = float(self.cfg.sac_bc_action_clip)
        action_low = torch.full(
            (self.action_dim,), -action_clip, device=self.device,
            dtype=torch.float32,
        )
        action_high = torch.full_like(action_low, action_clip)
        self.dist_cls = functools.partial(
            FastSACTanhNormal,
            low=action_low,
            high=action_high,
            event_dims=1,
        )
        self.dist_keys = list(FastSACTanhNormal.dist_keys)
        self.actor_backend = FASTSAC_BC_DAGGER_ACTOR_BACKEND
        self._q_critic_widths = [
            int(self.observation_spec[key].shape[-1])
            for key in self.q_critic_keys
        ]
        self._q_actor_widths = [
            int(self.observation_spec[VEL_CMD_KEY].shape[-1]),
            int(self.observation_spec[OBS_KEY].shape[-1]),
            int(self.cfg.latent_dim),
        ]
        self._fastsac_action_low = action_low.detach().cpu().tolist()
        self._fastsac_action_high = action_high.detach().cpu().tolist()
        self._fastsac_q_action_low = action_low.detach()
        self._fastsac_q_action_scale = (
            (action_high - action_low) * 0.5
        ).detach()
        self._fastsac_action_log_scale_sum = float(
            torch.log((action_high - action_low) * 0.5).sum().item()
        )
        # The executable support is a DAgger safety guard, not an entropy
        # coordinate choice. Score entropy in a fixed raw-action unit so a
        # different safety clip cannot silently add action_dim * log(clip) to
        # every SAC target and actor loss.
        self._fastsac_entropy_reference_scale = float(
            self.cfg.sac_entropy_reference_scale
        )
        self._fastsac_entropy_reference_log_scale_sum = float(
            self.action_dim
            * math.log(self._fastsac_entropy_reference_scale)
        )
        initial_log_std = math.log(
            float(self.cfg.sac_bc_initial_action_std) / action_clip
        )
        self.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
            self.action_dim, initial_log_std, self.device
        )

        # Pure DAgger optimized and executed only the mean. Keep PPO's unused
        # positive standard-deviation tensor intact for checkpoint fidelity,
        # but make it impossible for Stage-2 SAC to optimize or reinterpret it.
        bc_actor_core = self._ppo_actor_core(self.actor_adapt)
        bc_actor_core.actor_std.requires_grad_(False)
        # Actor._load_from_state_dict optionally replaces checkpoint std with
        # cfg.load_noise_scale. DAgger never used this tensor, but preserving
        # it bit-for-bit keeps checkpoint provenance truthful and prevents any
        # future code from confusing a constructor override with learned state.
        bc_actor_core.load_noise_scale = None

        # Stage 2 uses neither PPO's value network nor its optimizers. Keep the
        # loaded BC actor modules themselves, because they are the requested
        # warm start for the bounded SAC adapter below.
        self.opt_policy = None
        self.opt_critic = None
        for attribute in ("critic", "value_norm", "gae", "critic_loss_fn"):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _configure_actor_backend(self):
        if self._uses_bc_dagger_finetune_source():
            self._configure_bc_dagger_actor_backend()
            return
        self._teacher_actor_std_reset_applied = False
        self._teacher_actor_std_reset_applied_q_updates = None
        self._teacher_actor_std_reset_event_q_updates = None
        if self.cfg.phase == "train":
            _validate_fastsac_teacher_config(self.cfg)
        action_low, action_high = self._vaic_action_bounds()
        self.dist_cls = functools.partial(
            FastSACTanhNormal,
            low=action_low,
            high=action_high,
            event_dims=1,
        )
        self.dist_keys = list(FastSACTanhNormal.dist_keys)
        command_key = (
            "command_" if self.observation_spec.get("command_", None) is not None
            else CMD_KEY
        )
        teacher_keys = [command_key, OBS_KEY, PRIV_FEATURE_KEY]
        student_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        if self.observation_spec.get(REF_JPOS_KEY, None) is None:
            raise KeyError(
                f"FastSAC teacher requires VAIC observation {REF_JPOS_KEY!r}"
            )
        if self.observation_spec[REF_JPOS_KEY].shape[-1] != self.action_dim:
            raise ValueError(
                "VAIC reference action dimension does not match FastSAC action dimension"
            )
        teacher_dim = (
            self.observation_spec[command_key].shape[-1]
            + self.observation_spec[OBS_KEY].shape[-1]
            + self.cfg.latent_dim
        )
        student_dim = self._q_actor_dim
        self.actor = self._build_fastsac_actor(
            teacher_keys, teacher_dim, action_low, action_high,
            reference_key=REF_JPOS_KEY,
        )
        self.actor_adapt = self._build_fastsac_actor(
            student_keys, student_dim, action_low, action_high
        )
        initial_log_std = getattr(
            self.cfg, "sac_teacher_initial_log_std", None
        )
        if self.cfg.phase == "train" and initial_log_std is not None:
            raw_bias = self._set_teacher_actor_log_std(
                float(initial_log_std), clear_optimizer_state=False
            )
            logging.info(
                "Initialized Stage-1 teacher actor log std to %.6f "
                "(raw head bias %.6f).",
                float(initial_log_std),
                raw_bias,
            )
        self.actor_backend = FASTSAC_ACTOR_BACKEND
        self._q_critic_widths = [
            int(self.observation_spec[key].shape[-1])
            for key in self.q_critic_keys
        ]
        self._q_actor_widths = [
            int(self.observation_spec[VEL_CMD_KEY].shape[-1]),
            int(self.observation_spec[OBS_KEY].shape[-1]),
            int(self.cfg.latent_dim),
        ]
        self._fastsac_teacher_actor_dim = int(teacher_dim)
        self._fastsac_action_low = action_low.detach().cpu().tolist()
        self._fastsac_action_high = action_high.detach().cpu().tolist()
        # These tensors are deliberately Q-input transforms, not changes to the
        # actor distribution or replay schema.  They are derivable from the
        # checkpoint-validated actor bounds and therefore need no state entry.
        self._fastsac_q_action_low = action_low.detach()
        self._fastsac_q_action_scale = (
            (action_high - action_low) * 0.5
        ).detach()
        self._fastsac_action_log_scale_sum = float(
            torch.log((action_high - action_low) * 0.5).sum().item()
        )

        # PPOVEL constructs the shared VAIC modules before this backend replaces
        # its actor. FastSAC never retains or steps the PPO actor/value path.
        self.opt_policy = None
        self.opt_critic = None
        for attribute in ("critic", "value_norm", "gae", "critic_loss_fn"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        if self.cfg.enable_residual_distillation:
            self.opt_adapt_actor = torch.optim.Adam(
                self.actor_adapt.parameters(), lr=self.cfg.lr
            )

        # HOI updates the actor from replayed raw observations.  Do not store
        # encoder_priv outputs: they become stale as the encoder learns and
        # cannot carry gradients back into the unchanged VAIC perception path.
        # Most raw inputs are already packed in critic_observations; only the
        # tensors required to rebuild object/height features are extra fields.
        composite_batch_dims = len(self.observation_spec.shape)

        def observation_tail(key):
            return tuple(
                int(dim)
                for dim in self.observation_spec[key].shape[composite_batch_dims:]
            )

        ref_shape = observation_tail(REF_JPOS_KEY)
        if ref_shape != (self.action_dim,):
            raise ValueError(
                f"VAIC reference action shape {ref_shape} does not match "
                f"FastSAC action shape {(self.action_dim,)}"
            )
        self._teacher_raw_replay_fields = [
            (
                REF_JPOS_KEY,
                TEACHER_REF_ACTION_FIELD,
                NEXT_TEACHER_REF_ACTION_FIELD,
            )
        ]
        self._teacher_replay_extra_shapes = {
            TEACHER_REF_ACTION_FIELD: ref_shape,
            NEXT_TEACHER_REF_ACTION_FIELD: ref_shape,
        }
        if self._q_conditions_on_actuator_state():
            context_shape = (self._q_actuator_context_dim,)
            self._teacher_replay_extra_shapes.update({
                TEACHER_ACTUATOR_CONTEXT_FIELD: context_shape,
                NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD: context_shape,
            })
        self._teacher_replay_constant_shapes = {}
        if OBJECT_KEY in self.q_critic_keys:
            self._teacher_raw_replay_fields.append(
                (
                    OBJECT_GEO_KEY,
                    TEACHER_OBJECT_GEO_FIELD,
                    None,
                )
            )
            geo_shape = observation_tail(OBJECT_GEO_KEY)
            # Object geometry is fixed throughout an episode. Ordinary and
            # truncation-bootstrap next states therefore share the current row's
            # geometry; true terminals do not bootstrap. Store it only once.
            self._teacher_replay_constant_shapes[
                TEACHER_OBJECT_GEO_FIELD
            ] = geo_shape
        if hasattr(self, "height_encoder"):
            self._teacher_raw_replay_fields.append(
                (HEIGHT_KEY, TEACHER_HEIGHT_FIELD, NEXT_TEACHER_HEIGHT_FIELD)
            )
            height_shape = observation_tail(HEIGHT_KEY)
            self._teacher_replay_extra_shapes.update({
                TEACHER_HEIGHT_FIELD: height_shape,
                NEXT_TEACHER_HEIGHT_FIELD: height_shape,
            })
        teacher_learning_fields = [
            "critic_observations",
            "actions",
            "rewards",
            "dones",
            "truncations",
            "discounts",
            "effective_n_steps",
            "next_critic_observations",
        ]
        for _, current_field, next_field in self._teacher_raw_replay_fields:
            teacher_learning_fields.append(current_field)
            if next_field is not None:
                teacher_learning_fields.append(next_field)
        if self._q_conditions_on_actuator_state():
            teacher_learning_fields.extend((
                TEACHER_ACTUATOR_CONTEXT_FIELD,
                NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
            ))
        self._teacher_learning_fields = tuple(teacher_learning_fields)
        self._teacher_n_step_accumulator = None
        self._rollout_final_batch = None
        self._teacher_export_started = False
        self._teacher_export_start_seen = None
        self.sac_environment_steps = 0
        self.sac_rollout_count = 0
        self.sac_update_count = 0
        self.sac_actor_update_count = 0
        self.sac_alpha_update_count = 0

        if self.cfg.phase == "train":
            if bool(self.cfg.save_teacher_buffer):
                raise ValueError(
                    "fastsac_vel_train no longer exports the Stage-2 teacher H5. "
                    "Keep algo.save_teacher_buffer=false and use the dedicated "
                    "offline-data collector for Stage 2."
                )
            if int(self.cfg.sac_batch_size) < 1:
                raise ValueError("sac_batch_size must be positive")
            if int(self.cfg.sac_teacher_learning_starts_transitions) < 0:
                raise ValueError(
                    "sac_teacher_learning_starts_transitions must be non-negative"
                )
            if int(self.cfg.sac_teacher_actor_learning_starts_q_updates) < 0:
                raise ValueError(
                    "sac_teacher_actor_learning_starts_q_updates must be "
                    "non-negative"
                )
            if int(self.cfg.sac_teacher_updates_per_env_step) < 1:
                raise ValueError(
                    "sac_teacher_updates_per_env_step must be positive"
                )
            if int(self.cfg.sac_teacher_update_interval_env_steps) < 1:
                raise ValueError(
                    "sac_teacher_update_interval_env_steps must be positive"
                )
            if int(self.cfg.sac_teacher_actor_batch_size) < 0:
                raise ValueError(
                    "sac_teacher_actor_batch_size must be non-negative"
                )
            if int(self.cfg.sac_teacher_policy_frequency) < 1:
                raise ValueError("sac_teacher_policy_frequency must be positive")
            if (
                not math.isfinite(float(self.cfg.sac_teacher_actor_lr))
                or float(self.cfg.sac_teacher_actor_lr) <= 0.0
            ):
                raise ValueError("sac_teacher_actor_lr must be finite and positive")
            if (
                not math.isfinite(float(self.cfg.sac_teacher_alpha_lr))
                or float(self.cfg.sac_teacher_alpha_lr) <= 0.0
            ):
                raise ValueError("sac_teacher_alpha_lr must be finite and positive")
            if float(self.cfg.sac_alpha_init) <= 0.0:
                raise ValueError("sac_alpha_init must be positive")
            if not 0.0 <= float(self.cfg.sac_tau) <= 1.0:
                raise ValueError("sac_tau must be in [0, 1]")
            if (
                not math.isfinite(float(self.cfg.sac_teacher_q_max_grad_norm))
                or float(self.cfg.sac_teacher_q_max_grad_norm) < 0.0
            ):
                raise ValueError(
                    "sac_teacher_q_max_grad_norm must be finite and non-negative"
                )
            if (
                not math.isfinite(float(self.cfg.sac_teacher_actor_max_grad_norm))
                or float(self.cfg.sac_teacher_actor_max_grad_norm) < 0.0
            ):
                raise ValueError(
                    "sac_teacher_actor_max_grad_norm must be finite and non-negative"
                )
            if not bool(self.cfg.train_student_models):
                logging.warning(
                    "Stage-1 student adaptation/distillation is disabled. "
                    "Teacher FastSAC actor/Q/alpha training remains active, but "
                    "the resulting checkpoint does not provide a pretrained "
                    "student warm-start for fastsac_vel_finetune."
                )

            teacher_parameters = list(self.actor.parameters())
            teacher_parameters.extend(self.encoder_priv.parameters())
            if hasattr(self, "height_cnn"):
                teacher_parameters.extend(self.height_cnn.parameters())
            self._teacher_actor_parameters = teacher_parameters
            self.sac_teacher_actor_optimizer = torch.optim.AdamW(
                teacher_parameters,
                lr=self.cfg.sac_teacher_actor_lr,
                weight_decay=self.cfg.q_weight_decay,
                betas=(0.9, 0.95),
                fused=str(self.device).startswith("cuda"),
            )
            self.log_alpha = nn.Parameter(torch.tensor(
                float(np.log(self.cfg.sac_alpha_init)),
                device=self.device,
                dtype=torch.float32,
            ))
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.cfg.sac_teacher_alpha_lr,
                betas=(0.9, 0.95),
                fused=str(self.device).startswith("cuda"),
            )
            self.target_entropy = _fastsac_target_entropy(
                self._fastsac_action_low,
                self._fastsac_action_high,
                self.cfg.sac_target_entropy_ratio,
            )

    def requires_training_replay(self):
        return self.cfg.phase == "train"

    def requires_value_bootstrap(self):
        return False

    def _sac_updates_per_env_step(self):
        """Return the Stage-2 RLPD update count per vector-environment step."""
        return int(self.cfg.sac_updates_per_env_step)

    def _teacher_updates_due(self, accepted_rows: int) -> int:
        """Return one configured Q-update burst on scheduled control steps."""
        if self.teacher_replay is None:
            raise RuntimeError("Teacher replay is not configured")
        warmup = int(self.cfg.sac_teacher_learning_starts_transitions)
        # Even when warmup is configured as zero, sampling an empty replay is
        # invalid.  There is deliberately no catch-up/backlog after warmup.
        if self.teacher_replay.size < 1 or self.teacher_replay.seen < warmup:
            return 0
        interval = int(getattr(
            self.cfg, "sac_teacher_update_interval_env_steps", 1
        ))
        # The collector increments this counter exactly once before asking for
        # the burst. Missing a scheduled step during warmup never accumulates
        # update credit; learning waits for the next interval boundary.
        if (
            self.sac_environment_steps < 1
            or self.sac_environment_steps % interval != 0
        ):
            return 0
        return int(self.cfg.sac_teacher_updates_per_env_step)

    def _teacher_actor_update_is_due(self) -> bool:
        """Start from a trained critic, then apply delayed SAC actor updates."""
        actor_start = int(
            self.cfg.sac_teacher_actor_learning_starts_q_updates
        )
        due = (
            self.sac_update_count >= actor_start
            and self.sac_update_count
            % int(self.cfg.sac_teacher_policy_frequency) == 0
        )
        if not due:
            return False
        seed_enabled = float(getattr(
            self.cfg, "sac_teacher_seed_storage_ratio", 0.0
        )) > 0.0
        if (
            seed_enabled
            and not bool(getattr(self.teacher_replay, "seed_frozen", False))
            and self.teacher_replay.size < self.teacher_replay.capacity
        ):
            # Never let the actor modify collection before the seed partition
            # contains only reference-policy data.
            return False
        return True

    def _teacher_alpha_update_is_due(self) -> bool:
        """Use the SAC entropy dual after the critic-only actor burn-in."""
        return (
            bool(getattr(self.cfg, "sac_use_autotune", True))
            and self.sac_update_count >= int(
                self.cfg.sac_teacher_actor_learning_starts_q_updates
            )
        )

    def _maybe_freeze_teacher_seed_replay(self):
        if float(getattr(
            self.cfg, "sac_teacher_seed_storage_ratio", 0.0
        )) == 0.0:
            return False
        return self.teacher_replay.freeze_seed_partition()

    def _maybe_freeze_teacher_seed_replay_before_q_update(self):
        """Freeze before sampling the batch that performs the first actor step."""
        if float(getattr(
            self.cfg, "sac_teacher_seed_storage_ratio", 0.0
        )) == 0.0 or bool(getattr(
            self.teacher_replay, "seed_frozen", False
        )):
            return False
        next_q_update = self.sac_update_count + 1
        actor_start = int(
            self.cfg.sac_teacher_actor_learning_starts_q_updates
        )
        actor_due = (
            next_q_update >= actor_start
            and next_q_update
            % int(self.cfg.sac_teacher_policy_frequency) == 0
        )
        if not actor_due or self.teacher_replay.size < self.teacher_replay.capacity:
            return False
        return self._maybe_freeze_teacher_seed_replay()

    def _sac_actor_update_is_due(
        self, update_index: int, logical_step: int, updates_per_step: int
    ) -> bool:
        """Apply a true global every-K-Q-update Stage-2 actor schedule."""
        frequency = int(self.cfg.sac_policy_frequency)
        return (self.sac_update_count + 1) % frequency == 0

    def configure_teacher_replay(self, path, restore_path=None):
        if self.cfg.phase != "train":
            return super().configure_teacher_replay(path, restore_path=restore_path)
        if getattr(self, "_replay_vecnorm", None) is None:
            raise RuntimeError(
                "Stage-1 FastSAC replay requires configure_replay_vecnorm() "
                "before replay allocation."
            )
        if bool(getattr(self.cfg, "save_teacher_buffer", False)):
            raise ValueError(
                "Stage-1 FastSAC H5 export is disabled; use the dedicated "
                "Stage-2 offline-data collector."
            )
        if (
            self._loaded_checkpoint_phase == "train"
            or restore_path is not None
            or self._loaded_teacher_replay_metadata is not None
        ):
            logging.warning(
                "Stage-1 FastSAC resumes model/optimizer/counters but intentionally "
                "starts its compact live replay empty; an old replay H5 is ignored."
            )
        self._loaded_teacher_replay_metadata = None
        self._teacher_export_started = False
        self._teacher_export_start_seen = None
        # The live FIFO is intentionally rebuilt empty; pending n-step starts
        # must follow the same lifecycle and never survive a resume/configure.
        self._teacher_n_step_accumulator = None
        replay_device = getattr(self, "_teacher_training_replay_device", None)
        if replay_device is None:
            # Compatibility for lightweight test policies constructed with
            # ``__new__``; ordinary policies validate and cache this in init.
            replay_device = _resolve_teacher_training_replay_device(
                getattr(self.cfg, "teacher_training_replay_device", "policy"),
                self.device,
            )
        self.teacher_replay = TeacherTrainingReplayBuffer(
            capacity=self.cfg.teacher_buffer_capacity,
            critic_dim=self._q_critic_dim,
            action_dim=self.action_dim,
            device=replay_device,
            extra_shapes=self._teacher_replay_extra_shapes,
            constant_shapes=self._teacher_replay_constant_shapes,
            seed_storage_ratio=getattr(
                self.cfg, "sac_teacher_seed_storage_ratio", 0.0
            ),
            seed_sample_ratio=getattr(
                self.cfg, "sac_teacher_seed_sample_ratio", 0.0
            ),
        )
        logging.info(
            "Stage-1 FastSAC live replay: capacity=%d, storage_device=%s, "
            "sample_device=%s, n_steps=%d, "
            "estimated=%.2f GiB (no actor observations, no H5 export).",
            self.teacher_replay.capacity,
            self.teacher_replay.device,
            self.device,
            int(getattr(self.cfg, "sac_teacher_n_steps", 1)),
            self.teacher_replay.estimated_bytes / (1024 ** 3),
        )

    def _optimizer_registry(self):
        if self.cfg.phase == "train":
            names = (
                "opt_adapt",
                "opt_adapt_actor",
                "opt_dr_estimator",
                "opt_q",
                "sac_teacher_actor_optimizer",
                "alpha_optimizer",
            )
        else:
            names = (
                "opt_adapt",
                "opt_dr_estimator",
                "opt_q",
                "sac_actor_optimizer",
                "alpha_optimizer",
            )
        optimizers = OrderedDict()
        registered_ids = set()
        for name in names:
            optimizer = getattr(self, name, None)
            if (
                not isinstance(optimizer, torch.optim.Optimizer)
                or id(optimizer) in registered_ids
            ):
                continue
            optimizers[name] = optimizer
            registered_ids.add(id(optimizer))
        return optimizers

    def get_rollout_policy(self, mode="train"):
        policy = super().get_rollout_policy(mode)
        selected = [key for key in policy.out_keys if key != "sample_log_prob"]
        if self.cfg.phase == "train" and mode == "train":
            # Stage 1 no longer produces the Stage-2 offline dataset. The
            # adaptation module may still advance its GRU carry, but its
            # per-step priv_pred output is recomputed by _train_adapt_no_depth
            # and must not be retained in the rollout/replay buffers.
            selected = [key for key in selected if key != PRIV_PRED_KEY]
        policy.reset_out_keys()
        policy.select_out_keys(*selected)
        return policy

    def begin_transition_collection(self):
        super().begin_transition_collection()
        self._rollout_final_batch = None
        self._interleaved_steps_collected = 0
        self._interleaved_q_metrics = []
        self._interleaved_actor_metrics = []
        self._interleaved_replay_accepted = 0
        self._interleaved_reward_sum = torch.zeros((), device=self.device)
        self._interleaved_transition_count = 0
        self._interleaved_effective_n_steps_sum = torch.zeros(
            (), device=self.device
        )
        self._last_truncation_finals_used = 0

    def _stage1_n_step_config(self):
        return {
            "n_steps": int(getattr(self.cfg, "sac_teacher_n_steps", 1)),
            "semantics": FASTSAC_STAGE1_N_STEP_RETURN_SEMANTICS,
        }

    def _stage1_actor_objective_config(self):
        objective = str(getattr(
            self.cfg, "sac_teacher_actor_objective", "sac"
        ))
        if objective == "sac":
            # AWAC-only knobs cannot make an otherwise identical SAC resume
            # incompatible because they do not participate in its execution.
            return {"objective": "sac"}
        if objective != "reference_awac":
            raise ValueError(
                "sac_teacher_actor_objective must be one of "
                f"{FASTSAC_STAGE1_ACTOR_OBJECTIVES}, got {objective!r}"
            )
        return {
            "objective": objective,
            "beta": float(getattr(
                self.cfg, "sac_teacher_awac_beta", 0.01
            )),
            "weight_clip": float(getattr(
                self.cfg, "sac_teacher_awac_weight_clip", 20.0
            )),
            "semantics": FASTSAC_STAGE1_REFERENCE_AWAC_SEMANTICS,
        }

    def _stage1_actor_uncertainty_gate_config(self):
        """Describe the optional behavior-anchored Stage-1 SAC actor gate."""
        enabled = bool(getattr(
            self.cfg, "sac_teacher_actor_uncertainty_gate", False
        ))
        if not enabled:
            # Keep disabled checkpoints independent of the enabled-only
            # semantics so the default remains an exact execution no-op.
            return {"enabled": False}
        return {
            "enabled": True,
            "anchor": "same_replay_row_recorded_behavior_action",
            "criterion": "mean_twin_improvement_gt_absolute_head_disagreement",
            "q_component": "gated_full_batch_denominator",
            "entropy_component": "ungated_all_rows",
            "semantics": FASTSAC_STAGE1_BEHAVIOR_UNCERTAINTY_GATE_SEMANTICS,
        }

    def _stage1_conservative_q_config(self):
        configured_start = getattr(
            self.cfg,
            "sac_teacher_conservative_q_starts_q_updates",
            None,
        )
        resolved_start = (
            int(getattr(
                self.cfg,
                "sac_teacher_actor_learning_starts_q_updates",
                8_000,
            ))
            if configured_start is None
            else int(configured_start)
        )
        return {
            "coefficient": float(getattr(
                self.cfg, "sac_teacher_conservative_q_coef", 0.0
            )),
            "margin": float(getattr(
                self.cfg, "sac_teacher_conservative_q_margin", 0.002
            )),
            "temperature": float(getattr(
                self.cfg, "sac_teacher_conservative_q_temperature", 0.002
            )),
            "starts_q_updates": resolved_start,
            "semantics": FASTSAC_STAGE1_CONSERVATIVE_Q_SEMANTICS,
        }

    def _teacher_conservative_q_is_active(self):
        config = self._stage1_conservative_q_config()
        return (
            config["coefficient"] > 0.0
            and self.sac_update_count + 1 >= config["starts_q_updates"]
        )

    def _get_teacher_n_step_accumulator(self):
        accumulator = getattr(self, "_teacher_n_step_accumulator", None)
        if accumulator is None:
            next_fields = ["next_critic_observations"]
            next_fields.extend(
                next_field
                for _, _, next_field in self._teacher_raw_replay_fields
                if next_field is not None
            )
            if self._q_conditions_on_actuator_state():
                next_fields.append(NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD)
            accumulator = _Stage1NStepAccumulator(
                int(getattr(self.cfg, "sac_teacher_n_steps", 1)),
                float(getattr(self.cfg, "gamma", 0.99)),
                next_fields=next_fields,
            )
            self._teacher_n_step_accumulator = accumulator
        return accumulator

    def _aggregate_teacher_n_step(
        self, current: TensorDict, transitions: dict[str, torch.Tensor]
    ):
        valid = _replay_valid_mask(current, TEACHER_REPLAY_MIN_STEP_COUNT)
        return self._get_teacher_n_step_accumulator().append(transitions, valid)

    def uses_interleaved_updates(self):
        """Teacher FastSAC updates before the next environment action."""
        return self.cfg.phase == "train"

    @torch.no_grad()
    def _prepare_teacher_final_state(self, td: TensorDict):
        """Prepare only teacher-learning inputs for a true next state."""
        raw_values = {
            next_field: self._replay_source(td, source_key).clone()
            for source_key, _, next_field in self._teacher_raw_replay_fields
            if next_field is not None
        }
        result = {
            "next_critic_observations": self._cat_replay_sources(
                td, self.q_critic_keys
            ).clone(),
        }
        result.update(raw_values)
        return result

    @torch.no_grad()
    def _teacher_transition_from_step(
        self,
        current: TensorDict,
        rollout_carry: TensorDict,
        actuator_context: torch.Tensor | None = None,
    ):
        """Build one vector-step transition before the environment resets are lost."""
        n = int(current.batch_size[0])
        actuator_context = self._transition_q_actuator_context(
            actuator_context, n, current.device
        )
        next_values = self._prepare_teacher_final_state(rollout_carry.clone())
        truncations = _vaic_truncation_mask(current).reshape(n).bool()
        if truncations.any():
            env_indices = truncations.nonzero(as_tuple=False).squeeze(-1)
            truncation_values = self._prepare_teacher_final_state(
                current["next"][env_indices].clone()
            )
            for key, values in truncation_values.items():
                next_values[key].index_copy_(0, env_indices, values)

        transitions = {
            "critic_observations": self._cat_replay_sources(
                current, self.q_critic_keys
            ).reshape(n, self._q_critic_dim),
            "actions": current[ACTION_KEY].reshape(n, self.action_dim),
            "rewards": self._scalarize_sac_reward(
                current[REWARD_KEY]
            ).reshape(n),
            "dones": current[DONE_KEY].reshape(n).bool(),
            "truncations": truncations,
            "discounts": current["next", "discount"].reshape(n),
            **next_values,
        }
        if actuator_context is not None:
            transitions[TEACHER_ACTUATOR_CONTEXT_FIELD] = actuator_context
            # Delay and alpha are episode-constant. In particular, timeout rows
            # must retain this pre-reset value instead of the new episode's one.
            transitions[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD] = actuator_context
        transitions.update({
            current_field: self._replay_source(current, source_key)
            for source_key, current_field, _ in self._teacher_raw_replay_fields
        })
        valid = _replay_valid_mask(current, TEACHER_REPLAY_MIN_STEP_COUNT)
        self._last_truncation_finals_used += int(
            (truncations & valid).sum().item()
        )
        return self._get_teacher_n_step_accumulator().append(
            transitions, valid
        )

    @torch.no_grad()
    def capture_truncation_final_observations(self, td: TensorDict, step: int):
        if self.cfg.phase != "train":
            return

        truncations = _vaic_truncation_mask(td).reshape(-1).bool()
        if not truncations.any():
            return
        env_indices = truncations.nonzero(as_tuple=False).squeeze(-1)
        final_values = self._prepare_teacher_final_state(
            td["next"][env_indices].clone()
        )
        final_values["indices"] = (
            env_indices * int(self.cfg.train_every) + int(step)
        )
        self._truncation_final_batches.append(final_values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        """Retain s_(t+1) for the last row of a chunked VAIC rollout."""
        if self.cfg.phase != "train":
            return
        self._rollout_final_batch = self._prepare_teacher_final_state(carry.clone())

    def _teacher_transition_chunks(self, td: TensorDict):
        """Yield one N-row transition batch per VAIC vector-environment step.

        Building the old dense N*T transition dictionary temporarily consumed
        several GiB at 4096 environments. Step-sized chunks also let the
        optimizer spend WBT sample credit as new rows arrive.
        """
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "Teacher FastSAC needs the final rollout observation. The train "
                "loop must call capture_rollout_final_observation(carry)."
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError(
                f"Rollout length {t} does not match train_every={self.cfg.train_every}."
            )
        actuator_contexts = self._consume_rollout_q_actuator_contexts(int(t))
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        truncation_batches = self._truncation_final_batches
        self._truncation_final_batches = []
        truncation_finals = None
        if truncation_batches:
            truncation_finals = {
                key: torch.cat(
                    [batch[key] for batch in truncation_batches], dim=0
                )
                for key in truncation_batches[0]
            }
            indices = truncation_finals["indices"].long()
            if (indices < 0).any() or (indices >= n * t).any():
                raise IndexError("Captured truncation index is outside the rollout")
            valid_flat = (
                td["step_count"].reshape(n * t)
                > TEACHER_REPLAY_MIN_STEP_COUNT
            )
            self._last_truncation_finals_used = int(
                valid_flat[indices].sum().item()
            )
        else:
            self._last_truncation_finals_used = 0

        for step in range(int(t)):
            current = td[:, step]
            if step + 1 < int(t):
                following = td[:, step + 1]
                next_values = {
                    "next_critic_observations": self._cat_replay_sources(
                        following, self.q_critic_keys
                    ).reshape(n, self._q_critic_dim),
                }
                next_values.update({
                    next_field: self._replay_source(following, source_key)
                    for source_key, _, next_field in self._teacher_raw_replay_fields
                    if next_field is not None
                })
            else:
                next_values = final_batch

            transitions = {
                "critic_observations": self._cat_replay_sources(
                    current, self.q_critic_keys
                ).reshape(n, self._q_critic_dim),
                "actions": current[ACTION_KEY].reshape(n, self.action_dim),
                "rewards": self._scalarize_sac_reward(
                    current[REWARD_KEY]
                ).reshape(n),
                "dones": current[DONE_KEY].reshape(n).bool(),
                "truncations": _vaic_truncation_mask(current).reshape(n).bool(),
                "discounts": current["next", "discount"].reshape(n),
                **next_values,
            }
            if actuator_contexts is not None:
                context = self._transition_q_actuator_context(
                    actuator_contexts[:, step], n, current.device
                )
                transitions[TEACHER_ACTUATOR_CONTEXT_FIELD] = context
                transitions[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD] = context
            transitions.update({
                current_field: self._replay_source(current, source_key)
                for source_key, current_field, _ in self._teacher_raw_replay_fields
            })

            if truncation_finals is not None:
                flat_indices = truncation_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    for key, values in truncation_finals.items():
                        if key == "indices":
                            continue
                        transitions[key] = transitions[key].clone()
                        transitions[key].index_copy_(
                            0, env_indices, values[selected]
                        )
            yield self._aggregate_teacher_n_step(current, transitions)

    def _teacher_transitions(self, td: TensorDict):
        """Materialize all rows for diagnostics/tests; training streams chunks."""
        chunks = list(self._teacher_transition_chunks(td))
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }

    def _prepare_teacher_learning_batch(self, batch):
        """Normalize raw replay fields with one current VecNorm snapshot."""
        snapshot = self._vecnorm_snapshot()
        prepared = dict(batch)
        for field in ("critic_observations", "next_critic_observations"):
            prepared[field] = self._normalize_replay_flat(
                batch[field],
                self.q_critic_keys,
                self._q_critic_widths,
                snapshot,
            )
        for source_key, current_field, next_field in self._teacher_raw_replay_fields:
            prepared[current_field] = self._normalize_replay_value(
                source_key, batch[current_field], snapshot
            )
            if next_field is not None:
                prepared[next_field] = self._normalize_replay_value(
                    source_key, batch[next_field], snapshot
                )
        return prepared

    def _teacher_actor_learning_batch(self, q_batch):
        """Return the Q batch or a separately sampled Stage-1 actor batch."""
        actor_batch_size = int(getattr(
            self.cfg, "sac_teacher_actor_batch_size", 0
        ))
        if actor_batch_size < 0:
            raise ValueError(
                "sac_teacher_actor_batch_size must be non-negative"
            )
        if actor_batch_size == 0:
            # This identity return is intentional: the default consumes no
            # additional replay RNG and exactly preserves prior update data.
            return q_batch
        actor_batch = self.teacher_replay.sample(
            actor_batch_size,
            device=self.device,
            generator=self.q_rng,
            fields=self._teacher_learning_fields,
        )
        return self._prepare_teacher_learning_batch(actor_batch)

    def _teacher_state_from_replay(self, batch, next_state=False):
        critic_key = (
            "next_critic_observations" if next_state else "critic_observations"
        )
        critic_obs = batch[critic_key]
        values = {}
        offset = 0
        for key, width in zip(self.q_critic_keys, self._q_critic_widths):
            values[key] = critic_obs[..., offset:offset + width]
            offset += width
        if offset != self._q_critic_dim:
            raise RuntimeError("Teacher FastSAC critic observation split is inconsistent")
        for source_key, current_field, next_field in self._teacher_raw_replay_fields:
            values[source_key] = batch[
                next_field
                if next_state and next_field is not None
                else current_field
            ]
        td = TensorDict(
            values,
            batch_size=critic_obs.shape[:-1],
            device=critic_obs.device,
        )
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        self.encoder_priv(td)
        return td

    def _teacher_q_alpha_update(self, batch):
        with torch.no_grad():
            next_td = self._teacher_state_from_replay(batch, next_state=True)
            next_dist = self.actor.get_dist(next_td)
            next_action, next_log_prob = next_dist.rsample_with_log_prob(
                generator=self.sac_action_rng
            )
            next_log_prob = self._normalized_action_log_prob(next_log_prob)
            bootstrap = _sac_bootstrap_mask(
                batch["dones"], batch["truncations"]
            )
            effective_n_steps = batch.get(
                "effective_n_steps", torch.ones_like(batch["discounts"])
            )
            if int(getattr(self.cfg, "sac_teacher_n_steps", 1)) == 1:
                # Keep the default path numerically identical to the original
                # one-step target, including its operation ordering.
                discount = self.cfg.gamma * batch["discounts"]
            else:
                discount = (
                    float(self.cfg.gamma) ** effective_n_steps
                ) * batch["discounts"]
            soft_reward = (
                batch["rewards"]
                - discount
                * bootstrap
                * self.log_alpha.exp()
                * next_log_prob
            )
            target = self._q_projection(
                self.qnet_target,
                batch["next_critic_observations"],
                next_action,
                soft_reward,
                bootstrap,
                discount,
                batch.get(NEXT_TEACHER_REF_ACTION_FIELD),
                batch.get(NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD),
            )
            target, target_values = _select_c51_twin_target(
                target,
                self.qnet_target.support,
                bool(getattr(self.cfg, "sac_clipped_double_q", True)),
            )

        logits = self._q_forward(
            self.qnet,
            batch["critic_observations"],
            batch["actions"],
            batch.get(TEACHER_REF_ACTION_FIELD),
            batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
        )
        per_q = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
        bellman_q_loss = per_q.sum()
        conservative_config = self._stage1_conservative_q_config()
        conservative_active = self._teacher_conservative_q_is_active()
        conservative_penalty = per_q.new_zeros(per_q.shape)
        conservative_q_loss = per_q.new_zeros(())
        conservative_gap = per_q.new_zeros(())
        conservative_positive_gap_fraction = per_q.new_zeros(())
        conservative_above_margin_fraction = per_q.new_zeros(())
        if conservative_active:
            # The policy action is deliberately deterministic and detached: it
            # is the action used by play/evaluation and this regularizer must
            # update Q only, never the actor or its privileged encoder.
            with torch.no_grad():
                current_td = self._teacher_state_from_replay(
                    batch, next_state=False
                )
                policy_action = self.actor.get_dist(current_td).mean.detach()
            policy_logits = self._q_forward(
                self.qnet,
                batch["critic_observations"],
                policy_action,
                batch.get(TEACHER_REF_ACTION_FIELD),
                batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
            )
            replay_q_values = self.qnet.values(logits)
            policy_q_values = self.qnet.values(policy_logits)
            conservative_penalty, policy_replay_gap = (
                _policy_replay_conservative_q_penalty(
                    policy_q_values,
                    replay_q_values,
                    conservative_config["margin"],
                    conservative_config["temperature"],
                )
            )
            conservative_q_loss = (
                conservative_config["coefficient"]
                * conservative_penalty.sum()
            )
            with torch.no_grad():
                conservative_gap = policy_replay_gap.mean()
                conservative_positive_gap_fraction = (
                    policy_replay_gap > 0.0
                ).float().mean()
                conservative_above_margin_fraction = (
                    policy_replay_gap > conservative_config["margin"]
                ).float().mean()
        q_loss = (
            bellman_q_loss + conservative_q_loss
            if conservative_active
            else bellman_q_loss
        )
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        # ``inf`` measures the real pre-clip norm without changing gradients.
        # Previously a disabled clip logged a hard-coded zero, hiding the
        # actor/Q instability that these diagnostics are intended to expose.
        q_grad = _measure_or_clip_grad_norm(
            self.qnet.parameters(),
            self.cfg.sac_teacher_q_max_grad_norm,
        )
        self.opt_q.step()
        self.q_update_count += 1
        self.sac_update_count += 1

        # The dual variable cannot reduce its entropy error while the primal
        # actor is intentionally frozen for critic burn-in.  Keep alpha at its
        # initial value until the same Q-update gate that enables the actor.
        alpha_loss = torch.zeros((), device=self.device)
        if self._teacher_alpha_update_is_due():
            alpha_loss = -(
                self.log_alpha.exp()
                * (next_log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.sac_alpha_update_count += 1

        return {
            "q_loss": q_loss.detach(),
            "bellman_q_loss": bellman_q_loss.detach(),
            "q1_loss": per_q[0].detach(),
            "q2_loss": per_q[1].detach(),
            "conservative_q_active": per_q.new_tensor(float(
                conservative_active
            )),
            "conservative_q_penalty": conservative_penalty.sum().detach(),
            "conservative_q_loss": conservative_q_loss.detach(),
            "conservative_policy_replay_q_gap": conservative_gap.detach(),
            "conservative_positive_gap_fraction": (
                conservative_positive_gap_fraction.detach()
            ),
            "conservative_above_margin_fraction": (
                conservative_above_margin_fraction.detach()
            ),
            "q_grad_norm": q_grad.detach(),
            "alpha_loss": alpha_loss.detach(),
            "target_q_min": target_values.min().detach(),
            "target_q_max": target_values.max().detach(),
        }

    @torch.no_grad()
    def _soft_update_teacher_q_target(self):
        for source, target_param in zip(
            self.qnet.parameters(), self.qnet_target.parameters()
        ):
            target_param.lerp_(source, self.cfg.sac_tau)

    def _teacher_reference_awac_actor_update(self, batch):
        """Apply an in-support, reference-relative AWAC policy improvement.

        Q is queried only under ``no_grad`` at the replayed action and VAIC's
        framewise reference action when constructing the actor loss.  Both are
        covered by the frozen reference-policy replay partition.  Subtracting
        their values per target head removes state calibration, while the
        pessimistic relative advantage requires both heads to support the
        replayed correction.
        """
        minibatch = self._teacher_state_from_replay(batch, next_state=False)
        dist = self.actor.get_dist(minibatch)
        critic_obs = batch["critic_observations"]
        replay_action = batch["actions"].detach()
        reference_action = minibatch[REF_JPOS_KEY].detach()
        actuator_context = batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD)
        objective_config = self._stage1_actor_objective_config()

        with torch.no_grad():
            replay_q_values = self.qnet_target.values(self._q_forward(
                self.qnet_target,
                critic_obs,
                replay_action,
                reference_action,
                actuator_context,
            ))
            reference_q_values = self.qnet_target.values(self._q_forward(
                self.qnet_target,
                critic_obs,
                reference_action,
                reference_action,
                actuator_context,
            ))
            awac_weights, awac_advantages = _reference_awac_weights(
                replay_q_values,
                reference_q_values,
                objective_config["beta"],
                objective_config["weight_clip"],
            )
            # Retain the independent SAC RNG stream and the existing entropy /
            # saturation diagnostics without letting this sample enter Q.
            sampled_action, sampled_log_prob = dist.rsample_with_log_prob(
                generator=self.sac_action_rng
            )
            sampled_log_prob = self._normalized_action_log_prob(
                sampled_log_prob
            )

        if not hasattr(dist, "log_prob_for_action"):
            raise TypeError(
                "reference_awac requires FastSACTanhNormal.log_prob_for_action"
            )
        replay_log_prob = dist.log_prob_for_action(
            replay_action, detach_scale=True
        )
        replay_log_prob = self._normalized_action_log_prob(replay_log_prob)
        actor_loss = -(
            awac_weights.detach() * replay_log_prob / float(self.action_dim)
        ).mean()

        # No target/online Q parameter participates in AWAC autograd. Clear the
        # preceding Bellman gradients so diagnostics and tests can verify this
        # boundary directly.
        self.opt_q.zero_grad(set_to_none=True)
        self.sac_teacher_actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _measure_or_clip_grad_norm(
            self._teacher_actor_parameters,
            self.cfg.sac_teacher_actor_max_grad_norm,
        )
        self.sac_teacher_actor_optimizer.step()
        self.sac_actor_update_count += 1

        with torch.no_grad():
            deterministic_action = dist.mean
            deterministic_q_values = self.qnet_target.values(self._q_forward(
                self.qnet_target,
                critic_obs,
                deterministic_action,
                reference_action,
                actuator_context,
            ))
            deterministic_relative_advantages = (
                deterministic_q_values - reference_q_values
            )
            deterministic_policy_q_mean = deterministic_q_values.mean()
            reference_q_mean = reference_q_values.mean()
            replay_q_mean = replay_q_values.mean()
            normalized_sampled_action = 2.0 * (
                (sampled_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_deterministic_action = 2.0 * (
                (deterministic_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_reference_action = 2.0 * (
                (reference_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_replay_action = 2.0 * (
                (replay_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_deterministic_action = (
                normalized_deterministic_action.clamp(-1.0, 1.0)
            )
            normalized_reference_action = normalized_reference_action.clamp(
                -1.0, 1.0
            )
            replay_reference_deviation = (
                normalized_replay_action - normalized_reference_action
            ).abs().mean(dim=-1)
            flat_weights = awac_weights.reshape(-1)
            awac_ess_fraction = (
                flat_weights.sum().square()
                / (
                    float(flat_weights.numel())
                    * flat_weights.square().sum().clamp_min(
                        torch.finfo(flat_weights.dtype).tiny
                    )
                )
            )
            zero = actor_loss.new_zeros(())

        return {
            "actor_loss": actor_loss.detach(),
            # This sentinel distinguishes AWAC from the unmodified SAC loss;
            # the AWAC-specific likelihood is exposed separately below.
            "actor_sac_loss": zero,
            "actor_grad_norm": actor_grad.detach(),
            "actor_uncertainty_gate_acceptance_fraction": zero,
            "actor_uncertainty_gate_accepted_confidence_margin": zero,
            "actor_uncertainty_gate_mean_confidence_margin": zero,
            "actor_uncertainty_gate_policy_replay_improvement": zero,
            "actor_uncertainty_gate_policy_replay_disagreement": zero,
            "entropy": -sampled_log_prob.mean().detach(),
            "action_std": dist.scale.mean().detach(),
            "reference_mean_action_error": (
                deterministic_action - reference_action
            ).abs().mean().detach(),
            # AWAC does not score a sampled policy action with Q. Use the
            # deterministic target-Q diagnostic consistently for these legacy
            # dashboard fields.
            "policy_q_mean": deterministic_policy_q_mean.detach(),
            "deterministic_policy_q_mean": (
                deterministic_policy_q_mean.detach()
            ),
            "reference_q_mean": reference_q_mean.detach(),
            "replay_q_mean": replay_q_mean.detach(),
            "policy_replay_q_gap": (
                deterministic_policy_q_mean - replay_q_mean
            ).detach(),
            "deterministic_policy_reference_advantage": (
                deterministic_policy_q_mean - reference_q_mean
            ).detach(),
            "deterministic_reference_q1_advantage": (
                deterministic_relative_advantages[0].mean().detach()
            ),
            "deterministic_reference_q2_advantage": (
                deterministic_relative_advantages[1].mean().detach()
            ),
            "deterministic_reference_pessimistic_advantage": (
                deterministic_relative_advantages.min(dim=0).values.mean().detach()
            ),
            "deterministic_reference_advantage_disagreement": (
                deterministic_relative_advantages[0]
                - deterministic_relative_advantages[1]
            ).abs().mean().detach(),
            "twin_q_disagreement": (
                deterministic_q_values[0] - deterministic_q_values[1]
            ).abs().mean().detach(),
            "deterministic_policy_twin_q_disagreement": (
                deterministic_q_values[0] - deterministic_q_values[1]
            ).abs().mean().detach(),
            "reference_twin_q_disagreement": (
                reference_q_values[0] - reference_q_values[1]
            ).abs().mean().detach(),
            "replay_twin_q_disagreement": (
                replay_q_values[0] - replay_q_values[1]
            ).abs().mean().detach(),
            "normalized_action_mean_deviation": (
                normalized_deterministic_action - normalized_reference_action
            ).abs().mean().detach(),
            "action_saturation_fraction": (
                normalized_sampled_action.abs() > 0.99
            ).float().mean().detach(),
            "awac_active": actor_loss.new_ones(()),
            "awac_loss": actor_loss.detach(),
            "awac_replay_log_prob": replay_log_prob.mean().detach(),
            "awac_advantage_mean": awac_advantages.mean().detach(),
            "awac_advantage_std": awac_advantages.std(
                unbiased=False
            ).detach(),
            "awac_advantage_min": awac_advantages.min().detach(),
            "awac_advantage_max": awac_advantages.max().detach(),
            "awac_positive_advantage_fraction": (
                awac_advantages > 0.0
            ).float().mean().detach(),
            "awac_weight_mean": awac_weights.mean().detach(),
            "awac_weight_max": awac_weights.max().detach(),
            "awac_weight_ess_fraction": awac_ess_fraction.detach(),
            "awac_weighted_replay_reference_deviation": (
                awac_weights * replay_reference_deviation
            ).mean().detach(),
        }

    def _teacher_actor_update(self, batch):
        actor_objective = str(getattr(
            self.cfg, "sac_teacher_actor_objective", "sac"
        ))
        if actor_objective == "reference_awac":
            return self._teacher_reference_awac_actor_update(batch)
        if actor_objective != "sac":
            raise ValueError(
                "sac_teacher_actor_objective must be one of "
                f"{FASTSAC_STAGE1_ACTOR_OBJECTIVES}, got {actor_objective!r}"
            )
        # Match HOI: update policy from the same replay sample as Q. Raw replay
        # fields rebuild the current VAIC encoder path, so encoder_priv and the
        # optional height CNN receive valid gradients instead of stale latents.
        minibatch = self._teacher_state_from_replay(batch, next_state=False)
        dist = self.actor.get_dist(minibatch)
        action, log_prob = dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        log_prob = self._normalized_action_log_prob(log_prob)
        critic_obs = batch["critic_observations"]
        reference_action = minibatch[REF_JPOS_KEY].detach()
        uncertainty_gate_enabled = bool(getattr(
            self.cfg, "sac_teacher_actor_uncertainty_gate", False
        ))
        zero_gate_metric = log_prob.detach().new_zeros(())
        gate_acceptance_fraction = zero_gate_metric
        gate_accepted_confidence_margin = zero_gate_metric
        gate_mean_confidence_margin = zero_gate_metric
        gate_policy_replay_improvement = zero_gate_metric
        gate_policy_replay_disagreement = zero_gate_metric

        # Q contributes dQ/da but must not accumulate or step critic weights.
        self.opt_q.zero_grad(set_to_none=True)
        q_requires_grad = [parameter.requires_grad for parameter in self.qnet.parameters()]
        for parameter in self.qnet.parameters():
            parameter.requires_grad_(False)
        try:
            q_logits = self._q_forward(
                self.qnet,
                critic_obs,
                action,
                reference_action,
                batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
            )
            policy_q_values = self.qnet.values(q_logits)
            q_value = _reduce_actor_q_values(
                policy_q_values,
                bool(getattr(self.cfg, "sac_clipped_double_q", True)),
            )
            actor_sac_loss = (
                self.log_alpha.exp().detach() * log_prob - q_value
            ).mean()
            if uncertainty_gate_enabled:
                # Compare the sampled policy action with the recorded behavior
                # action that generated this replay row. The replay action is an
                # in-support behavior anchor; VAIC's framewise kinematic
                # reference is not necessarily a dynamically competent policy.
                # Subtract per head before measuring disagreement so harmless
                # head-specific state-value calibration cancels out.
                with torch.no_grad():
                    replay_q_values_for_gate = self.qnet.values(
                        self._q_forward(
                            self.qnet,
                            critic_obs,
                            batch["actions"].detach(),
                            reference_action,
                            batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                        )
                    )
                    relative_advantages = (
                        policy_q_values.detach()
                        - replay_q_values_for_gate
                    )
                    mean_delta = relative_advantages.mean(dim=0)
                    relative_uncertainty = (
                        relative_advantages[0] - relative_advantages[1]
                    ).abs()
                    confidence_margin = mean_delta - relative_uncertainty
                    q_gate = (
                        (mean_delta > relative_uncertainty)
                        & (mean_delta > 0.0)
                    ).to(q_value.dtype).detach()
                    accepted_count = q_gate.sum()
                    gate_acceptance_fraction = q_gate.mean()
                    gate_accepted_confidence_margin = (
                        (confidence_margin * q_gate).sum()
                        / accepted_count.clamp_min(1.0)
                    )
                    gate_mean_confidence_margin = confidence_margin.mean()
                    gate_policy_replay_improvement = mean_delta.mean()
                    gate_policy_replay_disagreement = (
                        relative_uncertainty.mean()
                    )

                # Gate only the Q contribution. SAC entropy remains active on
                # every sample, and the full batch denominator automatically
                # reduces the actor step when few rows have earned Q trust.
                actor_loss = (
                    self.log_alpha.exp().detach() * log_prob
                    - q_gate * q_value
                ).mean()
            else:
                actor_loss = actor_sac_loss
            reference_mean_action_error = (
                dist.mean - reference_action
            ).abs().mean()
            self.sac_teacher_actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = _measure_or_clip_grad_norm(
                self._teacher_actor_parameters,
                self.cfg.sac_teacher_actor_max_grad_norm,
            )
            self.sac_teacher_actor_optimizer.step()
        finally:
            for parameter, requires_grad in zip(
                self.qnet.parameters(), q_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
        self.sac_actor_update_count += 1
        with torch.no_grad():
            deterministic_action = dist.mean
            deterministic_q_values = self.qnet.values(
                self._q_forward(
                    self.qnet,
                    critic_obs,
                    deterministic_action,
                    reference_action,
                    batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                )
            )
            reference_q_values = self.qnet.values(
                self._q_forward(
                    self.qnet,
                    critic_obs,
                    reference_action,
                    reference_action,
                    batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                )
            )
            replay_q_values = self.qnet.values(
                self._q_forward(
                    self.qnet,
                    critic_obs,
                    batch["actions"],
                    reference_action,
                    batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                )
            )
            deterministic_relative_advantages = (
                deterministic_q_values - reference_q_values
            )
            policy_q_mean = policy_q_values.mean()
            deterministic_policy_q_mean = deterministic_q_values.mean()
            reference_q_mean = reference_q_values.mean()
            replay_q_mean = replay_q_values.mean()
            normalized_action = 2.0 * (
                (action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_deterministic_action = 2.0 * (
                (deterministic_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_deterministic_action = normalized_deterministic_action.clamp(
                -1.0, 1.0
            )
            normalized_reference_action = 2.0 * (
                (reference_action - dist.low) / (dist.high - dist.low)
            ) - 1.0
            normalized_reference_action = normalized_reference_action.clamp(
                -1.0, 1.0
            )
        return {
            "actor_loss": actor_loss.detach(),
            "actor_sac_loss": actor_sac_loss.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "actor_uncertainty_gate_acceptance_fraction": (
                gate_acceptance_fraction.detach()
            ),
            "actor_uncertainty_gate_accepted_confidence_margin": (
                gate_accepted_confidence_margin.detach()
            ),
            "actor_uncertainty_gate_mean_confidence_margin": (
                gate_mean_confidence_margin.detach()
            ),
            "actor_uncertainty_gate_policy_replay_improvement": (
                gate_policy_replay_improvement.detach()
            ),
            "actor_uncertainty_gate_policy_replay_disagreement": (
                gate_policy_replay_disagreement.detach()
            ),
            "entropy": -log_prob.mean().detach(),
            "action_std": dist.scale.mean().detach(),
            "reference_mean_action_error": reference_mean_action_error.detach(),
            "policy_q_mean": policy_q_mean.detach(),
            "deterministic_policy_q_mean": deterministic_policy_q_mean.detach(),
            "reference_q_mean": reference_q_mean.detach(),
            "replay_q_mean": replay_q_mean.detach(),
            "policy_replay_q_gap": (policy_q_mean - replay_q_mean).detach(),
            "deterministic_policy_reference_advantage": (
                deterministic_policy_q_mean - reference_q_mean
            ).detach(),
            "deterministic_reference_q1_advantage": (
                deterministic_relative_advantages[0].mean()
            ).detach(),
            "deterministic_reference_q2_advantage": (
                deterministic_relative_advantages[1].mean()
            ).detach(),
            "deterministic_reference_pessimistic_advantage": (
                deterministic_relative_advantages.min(dim=0).values.mean()
            ).detach(),
            "deterministic_reference_advantage_disagreement": (
                deterministic_relative_advantages[0]
                - deterministic_relative_advantages[1]
            ).abs().mean().detach(),
            "twin_q_disagreement": (
                policy_q_values[0] - policy_q_values[1]
            ).abs().mean().detach(),
            "deterministic_policy_twin_q_disagreement": (
                deterministic_q_values[0] - deterministic_q_values[1]
            ).abs().mean().detach(),
            "reference_twin_q_disagreement": (
                reference_q_values[0] - reference_q_values[1]
            ).abs().mean().detach(),
            "replay_twin_q_disagreement": (
                replay_q_values[0] - replay_q_values[1]
            ).abs().mean().detach(),
            "normalized_action_mean_deviation": (
                normalized_deterministic_action - normalized_reference_action
            ).abs().mean().detach(),
            "action_saturation_fraction": (
                normalized_action.abs() > 0.99
            ).float().mean().detach(),
            "awac_active": actor_loss.new_zeros(()),
            "awac_loss": actor_loss.new_zeros(()),
            "awac_replay_log_prob": actor_loss.new_zeros(()),
            "awac_advantage_mean": actor_loss.new_zeros(()),
            "awac_advantage_std": actor_loss.new_zeros(()),
            "awac_advantage_min": actor_loss.new_zeros(()),
            "awac_advantage_max": actor_loss.new_zeros(()),
            "awac_positive_advantage_fraction": actor_loss.new_zeros(()),
            "awac_weight_mean": actor_loss.new_zeros(()),
            "awac_weight_max": actor_loss.new_zeros(()),
            "awac_weight_ess_fraction": actor_loss.new_zeros(()),
            "awac_weighted_replay_reference_deviation": (
                actor_loss.new_zeros(())
            ),
        }

    @staticmethod
    def _mean_metric(metrics, key, device):
        if not metrics:
            return torch.zeros((), device=device)
        return torch.stack([metric[key] for metric in metrics]).mean()

    def _teacher_export_is_due(self):
        # Stage 1 now owns only a compact live learning FIFO. The full
        # actor+critic offline dataset is produced by a separate collector.
        return False

    def _teacher_export_counts(self):
        if (
            self.teacher_replay is None
            or not self._teacher_export_started
            or self._teacher_export_start_seen is None
        ):
            return 0, 0
        seen = max(
            int(self.teacher_replay.seen) - int(self._teacher_export_start_seen),
            0,
        )
        return min(int(self.teacher_replay.size), seen), seen

    def _teacher_replay_checkpoint_metadata(self):
        if self.cfg.phase == "train":
            # The Stage-1 FIFO (including its CPU/CUDA placement) is ephemeral
            # and deliberately rebuilt empty on resume, so it does not belong
            # in checkpoint/replay semantics metadata.
            return None
        return super()._teacher_replay_checkpoint_metadata()

    def snapshot_teacher_replay(self, iteration, checkpoint_name):
        if self.teacher_replay is None:
            return None
        if self.cfg.phase == "train":
            return None
        rows, seen = self._teacher_export_counts()
        return self.teacher_replay.snapshot(
            iteration,
            checkpoint_name,
            row_count=rows,
            seen_count=seen,
        )

    def collect_environment_step(
        self,
        transition_td: TensorDict,
        rollout_carry: TensorDict,
        actuator_context: torch.Tensor | None = None,
    ):
        """Insert one vector step and run the fixed WBT updates immediately."""
        if self.cfg.phase != "train":
            raise RuntimeError(
                "Interleaved collection is only valid for teacher FastSAC."
            )
        if self.teacher_replay is None:
            raise RuntimeError("Teacher FastSAC replay is not configured")
        if not hasattr(self, "_interleaved_steps_collected"):
            raise RuntimeError(
                "begin_transition_collection() must run before collection"
            )
        if self._interleaved_steps_collected >= int(self.cfg.train_every):
            raise RuntimeError("Collected more FastSAC steps than train_every")

        if not self._teacher_export_started and self._teacher_export_is_due():
            # Preserve the learning FIFO. Only the later H5 snapshot is gated.
            self._teacher_export_started = True
            self._teacher_export_start_seen = int(self.teacher_replay.seen)

        if actuator_context is None:
            # Preserve the historical two-argument hook for disabled policies
            # and lightweight alternate collectors.
            transitions = self._teacher_transition_from_step(
                transition_td, rollout_carry
            )
        else:
            transitions = self._teacher_transition_from_step(
                transition_td, rollout_carry, actuator_context
            )
        accepted_now = self.teacher_replay.append(transitions)
        self._interleaved_replay_accepted += accepted_now
        self._interleaved_reward_sum += transitions["rewards"].sum()
        self._interleaved_transition_count += int(
            transitions["rewards"].numel()
        )
        if not hasattr(self, "_interleaved_effective_n_steps_sum"):
            self._interleaved_effective_n_steps_sum = torch.zeros(
                (), device=self.device
            )
        self._interleaved_effective_n_steps_sum += transitions.get(
            "effective_n_steps", torch.ones_like(transitions["rewards"])
        ).sum()

        self.sac_environment_steps += 1
        updates_due = self._teacher_updates_due(accepted_now)
        for _ in range(updates_due):
            self._maybe_reset_teacher_actor_std_before_q_update()
            self._maybe_freeze_teacher_seed_replay_before_q_update()
            batch = self.teacher_replay.sample(
                self.cfg.sac_batch_size,
                device=self.device,
                generator=self.q_rng,
                fields=self._teacher_learning_fields,
            )
            batch = self._prepare_teacher_learning_batch(batch)
            self._interleaved_q_metrics.append(
                self._teacher_q_alpha_update(batch)
            )
            if self._teacher_actor_update_is_due():
                self._maybe_freeze_teacher_seed_replay()
                self._interleaved_actor_metrics.append(
                    self._teacher_actor_update(
                        self._teacher_actor_learning_batch(batch)
                    )
                )
            # Match WBT ordering: Q -> alpha -> delayed actor -> target Q.
            self._soft_update_teacher_q_target()

        self._interleaved_steps_collected += 1

    def train_op(self, tensordict):
        if self.cfg.phase != "train":
            raise RuntimeError(
                "FastSACVEL.train_op is the teacher FastSAC path; "
                "student training must use FastSACVelFinetune."
            )
        if self.teacher_replay is None:
            raise RuntimeError(
                "Teacher FastSAC replay is not configured. Construct training "
                "with configure_replay=True."
            )
        rollout_td = tensordict.exclude("stats")
        interleaved_steps = int(
            getattr(self, "_interleaved_steps_collected", 0)
        )
        if interleaved_steps:
            if interleaved_steps != int(self.cfg.train_every):
                raise RuntimeError(
                    "Teacher FastSAC rollout ended after "
                    f"{interleaved_steps} interleaved steps; expected "
                    f"{self.cfg.train_every}."
                )
            accepted = self._interleaved_replay_accepted
            reward_sum = self._interleaved_reward_sum
            transition_count = self._interleaved_transition_count
            effective_n_steps_sum = self._interleaved_effective_n_steps_sum
            q_metrics = self._interleaved_q_metrics
            actor_metrics = self._interleaved_actor_metrics
        else:
            # Compatibility path for alternate collectors and unit tests. The
            # production train.py path updates immediately after every step.
            if not self._teacher_export_started and self._teacher_export_is_due():
                self._teacher_export_started = True
                self._teacher_export_start_seen = int(self.teacher_replay.seen)
            accepted = 0
            reward_sum = torch.zeros((), device=self.device)
            transition_count = 0
            effective_n_steps_sum = torch.zeros((), device=self.device)
            q_metrics = []
            actor_metrics = []
            for transitions in (
                self._teacher_transition_chunks(rollout_td)
            ):
                accepted_now = self.teacher_replay.append(transitions)
                accepted += accepted_now
                reward_sum += transitions["rewards"].sum()
                transition_count += int(transitions["rewards"].numel())
                effective_n_steps_sum += transitions.get(
                    "effective_n_steps",
                    torch.ones_like(transitions["rewards"]),
                ).sum()
                self.sac_environment_steps += 1
                for _ in range(self._teacher_updates_due(accepted_now)):
                    self._maybe_reset_teacher_actor_std_before_q_update()
                    self._maybe_freeze_teacher_seed_replay_before_q_update()
                    batch = self.teacher_replay.sample(
                        self.cfg.sac_batch_size,
                        device=self.device,
                        generator=self.q_rng,
                        fields=self._teacher_learning_fields,
                    )
                    batch = self._prepare_teacher_learning_batch(batch)
                    q_metrics.append(self._teacher_q_alpha_update(batch))
                    if self._teacher_actor_update_is_due():
                        self._maybe_freeze_teacher_seed_replay()
                        actor_metrics.append(
                            self._teacher_actor_update(
                                self._teacher_actor_learning_batch(batch)
                            )
                        )
                    self._soft_update_teacher_q_target()
        self.sac_rollout_count += 1

        # Optionally preserve VAIC's teacher->student supervised adaptation and
        # distillation after the just-updated FastSAC teacher.
        student_training_active = bool(self.cfg.train_student_models)
        adapt_info = (
            self._train_adapt_no_depth(rollout_td.copy())
            if student_training_active
            else {}
        )
        self.num_updates += 1

        h5_rows, _ = self._teacher_export_counts()
        std_schedule = self._teacher_actor_std_schedule_config()
        std_reset_applied = bool(getattr(
            self, "_teacher_actor_std_reset_applied", False
        ))
        scheduled_log_std = (
            std_schedule["actor_reset_log_std"]
            if std_reset_applied
            else std_schedule["initial_log_std"]
        )
        if scheduled_log_std is None:
            scheduled_log_std = 0.5 * (
                float(getattr(self.cfg, "fastsac_log_std_min", -5.0))
                + float(getattr(self.cfg, "fastsac_log_std_max", 0.0))
            )
        reset_event_q_updates = getattr(
            self, "_teacher_actor_std_reset_event_q_updates", None
        )
        reset_applied_q_updates = getattr(
            self, "_teacher_actor_std_reset_applied_q_updates", None
        )
        actor_objective_config = self._stage1_actor_objective_config()
        info = {
            "fastsac/q_active": float(bool(q_metrics)),
            "fastsac/actor_objective_reference_awac": float(
                actor_objective_config["objective"] == "reference_awac"
            ),
            "fastsac/awac_beta": float(
                actor_objective_config.get("beta", 0.0)
            ),
            "fastsac/awac_weight_clip": float(
                actor_objective_config.get("weight_clip", 0.0)
            ),
            "fastsac/q_loss": self._mean_metric(q_metrics, "q_loss", self.device).item(),
            "fastsac/bellman_q_loss": self._mean_metric(
                q_metrics, "bellman_q_loss", self.device
            ).item(),
            "fastsac/q1_loss": self._mean_metric(q_metrics, "q1_loss", self.device).item(),
            "fastsac/q2_loss": self._mean_metric(q_metrics, "q2_loss", self.device).item(),
            "fastsac/conservative_q_active": self._mean_metric(
                q_metrics, "conservative_q_active", self.device
            ).item(),
            "fastsac/conservative_q_penalty": self._mean_metric(
                q_metrics, "conservative_q_penalty", self.device
            ).item(),
            "fastsac/conservative_q_loss": self._mean_metric(
                q_metrics, "conservative_q_loss", self.device
            ).item(),
            "fastsac/conservative_policy_replay_q_gap": self._mean_metric(
                q_metrics,
                "conservative_policy_replay_q_gap",
                self.device,
            ).item(),
            "fastsac/conservative_positive_gap_fraction": self._mean_metric(
                q_metrics,
                "conservative_positive_gap_fraction",
                self.device,
            ).item(),
            "fastsac/conservative_above_margin_fraction": self._mean_metric(
                q_metrics,
                "conservative_above_margin_fraction",
                self.device,
            ).item(),
            "fastsac/q_grad_norm": self._mean_metric(q_metrics, "q_grad_norm", self.device).item(),
            "fastsac/target_q_min": self._mean_metric(q_metrics, "target_q_min", self.device).item(),
            "fastsac/target_q_max": self._mean_metric(q_metrics, "target_q_max", self.device).item(),
            "fastsac/actor_loss": self._mean_metric(actor_metrics, "actor_loss", self.device).item(),
            "fastsac/actor_sac_loss": self._mean_metric(
                actor_metrics, "actor_sac_loss", self.device
            ).item(),
            "fastsac/awac_active": self._mean_metric(
                actor_metrics, "awac_active", self.device
            ).item(),
            "fastsac/awac_loss": self._mean_metric(
                actor_metrics, "awac_loss", self.device
            ).item(),
            "fastsac/awac_replay_log_prob": self._mean_metric(
                actor_metrics, "awac_replay_log_prob", self.device
            ).item(),
            "fastsac/awac_advantage_mean": self._mean_metric(
                actor_metrics, "awac_advantage_mean", self.device
            ).item(),
            "fastsac/awac_advantage_std": self._mean_metric(
                actor_metrics, "awac_advantage_std", self.device
            ).item(),
            "fastsac/awac_advantage_min": self._mean_metric(
                actor_metrics, "awac_advantage_min", self.device
            ).item(),
            "fastsac/awac_advantage_max": self._mean_metric(
                actor_metrics, "awac_advantage_max", self.device
            ).item(),
            "fastsac/awac_positive_advantage_fraction": self._mean_metric(
                actor_metrics,
                "awac_positive_advantage_fraction",
                self.device,
            ).item(),
            "fastsac/awac_weight_mean": self._mean_metric(
                actor_metrics, "awac_weight_mean", self.device
            ).item(),
            "fastsac/awac_weight_max": self._mean_metric(
                actor_metrics, "awac_weight_max", self.device
            ).item(),
            "fastsac/awac_weight_ess_fraction": self._mean_metric(
                actor_metrics, "awac_weight_ess_fraction", self.device
            ).item(),
            "fastsac/awac_weighted_replay_reference_deviation": (
                self._mean_metric(
                    actor_metrics,
                    "awac_weighted_replay_reference_deviation",
                    self.device,
                ).item()
            ),
            "fastsac/actor_grad_norm": self._mean_metric(actor_metrics, "actor_grad_norm", self.device).item(),
            "fastsac/actor_uncertainty_gate_acceptance_fraction": (
                self._mean_metric(
                    actor_metrics,
                    "actor_uncertainty_gate_acceptance_fraction",
                    self.device,
                ).item()
            ),
            "fastsac/actor_uncertainty_gate_accepted_confidence_margin": (
                self._mean_metric(
                    actor_metrics,
                    "actor_uncertainty_gate_accepted_confidence_margin",
                    self.device,
                ).item()
            ),
            "fastsac/actor_uncertainty_gate_mean_confidence_margin": (
                self._mean_metric(
                    actor_metrics,
                    "actor_uncertainty_gate_mean_confidence_margin",
                    self.device,
                ).item()
            ),
            "fastsac/actor_uncertainty_gate_policy_replay_improvement": (
                self._mean_metric(
                    actor_metrics,
                    "actor_uncertainty_gate_policy_replay_improvement",
                    self.device,
                ).item()
            ),
            "fastsac/actor_uncertainty_gate_policy_replay_disagreement": (
                self._mean_metric(
                    actor_metrics,
                    "actor_uncertainty_gate_policy_replay_disagreement",
                    self.device,
                ).item()
            ),
            "fastsac/entropy": self._mean_metric(actor_metrics, "entropy", self.device).item(),
            "fastsac/normalized_action_entropy": (
                self._mean_metric(actor_metrics, "entropy", self.device).item()
                if actor_metrics else 0.0
            ),
            "fastsac/physical_action_entropy": (
                self._mean_metric(actor_metrics, "entropy", self.device).item()
                + self._fastsac_action_log_scale_sum
                if actor_metrics else 0.0
            ),
            "fastsac/action_std": self._mean_metric(actor_metrics, "action_std", self.device).item(),
            "fastsac/reference_mean_action_error": self._mean_metric(
                actor_metrics, "reference_mean_action_error", self.device
            ).item(),
            "fastsac/policy_q_mean": self._mean_metric(
                actor_metrics, "policy_q_mean", self.device
            ).item(),
            "fastsac/deterministic_policy_q_mean": self._mean_metric(
                actor_metrics, "deterministic_policy_q_mean", self.device
            ).item(),
            "fastsac/reference_q_mean": self._mean_metric(
                actor_metrics, "reference_q_mean", self.device
            ).item(),
            "fastsac/replay_q_mean": self._mean_metric(
                actor_metrics, "replay_q_mean", self.device
            ).item(),
            "fastsac/policy_replay_q_gap": self._mean_metric(
                actor_metrics, "policy_replay_q_gap", self.device
            ).item(),
            "fastsac/deterministic_policy_reference_advantage": self._mean_metric(
                actor_metrics,
                "deterministic_policy_reference_advantage",
                self.device,
            ).item(),
            "fastsac/deterministic_reference_q1_advantage": self._mean_metric(
                actor_metrics,
                "deterministic_reference_q1_advantage",
                self.device,
            ).item(),
            "fastsac/deterministic_reference_q2_advantage": self._mean_metric(
                actor_metrics,
                "deterministic_reference_q2_advantage",
                self.device,
            ).item(),
            "fastsac/deterministic_reference_pessimistic_advantage": (
                self._mean_metric(
                    actor_metrics,
                    "deterministic_reference_pessimistic_advantage",
                    self.device,
                ).item()
            ),
            "fastsac/deterministic_reference_advantage_disagreement": (
                self._mean_metric(
                    actor_metrics,
                    "deterministic_reference_advantage_disagreement",
                    self.device,
                ).item()
            ),
            "fastsac/twin_q_disagreement": self._mean_metric(
                actor_metrics, "twin_q_disagreement", self.device
            ).item(),
            "fastsac/deterministic_policy_twin_q_disagreement": self._mean_metric(
                actor_metrics,
                "deterministic_policy_twin_q_disagreement",
                self.device,
            ).item(),
            "fastsac/reference_twin_q_disagreement": self._mean_metric(
                actor_metrics, "reference_twin_q_disagreement", self.device
            ).item(),
            "fastsac/replay_twin_q_disagreement": self._mean_metric(
                actor_metrics, "replay_twin_q_disagreement", self.device
            ).item(),
            "fastsac/normalized_action_mean_deviation": self._mean_metric(
                actor_metrics, "normalized_action_mean_deviation", self.device
            ).item(),
            "fastsac/action_saturation_fraction": self._mean_metric(
                actor_metrics, "action_saturation_fraction", self.device
            ).item(),
            "fastsac/alpha_loss": self._mean_metric(q_metrics, "alpha_loss", self.device).item(),
            "fastsac/alpha": self.log_alpha.exp().item(),
            "fastsac/target_entropy": self.target_entropy,
            "fastsac/action_log_scale_sum": self._fastsac_action_log_scale_sum,
            "fastsac/q_action_normalized": float(
                bool(getattr(self.cfg, "sac_q_normalize_actions", False))
            ),
            "fastsac/q_action_reference_residual": float(
                self._q_uses_reference_residual()
            ),
            "fastsac/q_reference_dueling": float(
                self._q_uses_reference_dueling()
            ),
            "fastsac/q_action_input_gain": float(
                getattr(self.cfg, "sac_q_action_input_gain", 1.0)
            ),
            "fastsac/clipped_double_q": float(bool(
                getattr(self.cfg, "sac_clipped_double_q", True)
            )),
            "fastsac/actor_uncertainty_gate_enabled": float(bool(getattr(
                self.cfg, "sac_teacher_actor_uncertainty_gate", False
            ))),
            "fastsac/actor_uncertainty_gate_anchor_is_replay_action": float(
                bool(getattr(
                    self.cfg, "sac_teacher_actor_uncertainty_gate", False
                ))
            ),
            "fastsac/conservative_q_coefficient": float(getattr(
                self.cfg, "sac_teacher_conservative_q_coef", 0.0
            )),
            "fastsac/conservative_q_margin": float(getattr(
                self.cfg, "sac_teacher_conservative_q_margin", 0.002
            )),
            "fastsac/conservative_q_temperature": float(getattr(
                self.cfg, "sac_teacher_conservative_q_temperature", 0.002
            )),
            "fastsac/conservative_q_starts_q_updates": int(
                self._stage1_conservative_q_config()["starts_q_updates"]
            ),
            "fastsac/q_update_count": self.q_update_count,
            "fastsac/actor_update_count": self.sac_actor_update_count,
            "fastsac/actor_active": float(
                self.sac_update_count >= int(
                    self.cfg.sac_teacher_actor_learning_starts_q_updates
                )
            ),
            "fastsac/alpha_active": float(
                bool(getattr(self.cfg, "sac_use_autotune", True))
                and self.sac_update_count >= int(
                    self.cfg.sac_teacher_actor_learning_starts_q_updates
                )
            ),
            "fastsac/alpha_autotune": float(bool(
                getattr(self.cfg, "sac_use_autotune", True)
            )),
            "fastsac/actor_learning_starts_q_updates": int(
                self.cfg.sac_teacher_actor_learning_starts_q_updates
            ),
            "fastsac/teacher_actor_std_schedule_log_std": float(
                scheduled_log_std
            ),
            "fastsac/teacher_actor_std_reset_event": float(
                reset_event_q_updates is not None
            ),
            "fastsac/teacher_actor_std_reset_event_q_updates": (
                -1
                if reset_event_q_updates is None
                else int(reset_event_q_updates)
            ),
            "fastsac/teacher_actor_std_reset_applied": float(
                std_reset_applied
            ),
            "fastsac/teacher_actor_std_reset_q_updates": (
                -1
                if std_schedule["reset_q_updates"] is None
                else int(std_schedule["reset_q_updates"])
            ),
            "fastsac/teacher_actor_std_reset_applied_q_updates": (
                -1
                if reset_applied_q_updates is None
                else int(reset_applied_q_updates)
            ),
            "fastsac/alpha_update_count": self.sac_alpha_update_count,
            "fastsac/environment_steps": self.sac_environment_steps,
            "fastsac/effective_updates_per_env_step": (
                len(q_metrics) / float(self.cfg.train_every)
            ),
            "fastsac/configured_updates_per_env_step": int(
                self.cfg.sac_teacher_updates_per_env_step
            ),
            "fastsac/update_interval_env_steps": int(
                getattr(
                    self.cfg, "sac_teacher_update_interval_env_steps", 1
                )
            ),
            "fastsac/configured_average_updates_per_env_step": (
                float(self.cfg.sac_teacher_updates_per_env_step)
                / float(getattr(
                    self.cfg, "sac_teacher_update_interval_env_steps", 1
                ))
            ),
            "fastsac/effective_actor_batch_size": (
                int(getattr(
                    self.cfg, "sac_teacher_actor_batch_size", 0
                ))
                or int(self.cfg.sac_batch_size)
            ),
            "fastsac/effective_sample_reuse": (
                len(q_metrics) * int(self.cfg.sac_batch_size) / max(accepted, 1)
            ),
            "fastsac/student_training_active": float(student_training_active),
            "fastsac/buffer_reward": (
                reward_sum / max(transition_count, 1)
            ).item(),
            "fastsac/configured_n_steps": int(getattr(
                self.cfg, "sac_teacher_n_steps", 1
            )),
            "fastsac/effective_n_steps": (
                effective_n_steps_sum / max(transition_count, 1)
            ).item(),
            "fastsac/replay_accepted": accepted,
            "fastsac/replay_saved": self.teacher_replay.saved,
            "fastsac/replay_seen": self.teacher_replay.seen,
            "fastsac/replay_fill_ratio": (
                self.teacher_replay.saved / self.teacher_replay.capacity
            ),
            "fastsac/seed_replay_frozen": float(bool(getattr(
                self.teacher_replay, "seed_frozen", False
            ))),
            "fastsac/seed_replay_size": int(getattr(
                self.teacher_replay, "seed_size", 0
            )),
            "fastsac/seed_replay_capacity": int(getattr(
                self.teacher_replay, "seed_capacity", 0
            )),
            "fastsac/online_replay_partition_size": int(getattr(
                self.teacher_replay,
                "online_size",
                self.teacher_replay.size,
            )),
            "fastsac/seed_replay_storage_ratio": float(getattr(
                self.cfg, "sac_teacher_seed_storage_ratio", 0.0
            )),
            "fastsac/seed_replay_sample_ratio": float(getattr(
                self.cfg, "sac_teacher_seed_sample_ratio", 0.0
            )),
            "fastsac/h5_export_active": float(self._teacher_export_started),
            "fastsac/h5_export_rows": (
                h5_rows
            ),
            # Retain the old dashboard keys as explicit disabled sentinels so
            # an existing W&B panel cannot mistake the legacy config value for
            # an active Stage-1 H5 collection gate.
            "fastsac/h5_start_iteration": -1,
            "fastsac/stage1_h5_enabled": 0.0,
            "fastsac/truncation_finals": self._last_truncation_finals_used,
        }
        actor_scale = rollout_td["scale"].detach().reshape(
            -1, self.action_dim
        ).mean(0)
        for joint_name, std in zip(self.joint_names, actor_scale):
            info[f"actor_std/{joint_name}"] = std.item()
        info["actor_std/mean"] = actor_scale.mean().item()
        info.update(adapt_info)
        self._teacher_actor_std_reset_event_q_updates = None
        return info

    def state_dict(self):
        # Synchronize checkpoint EMA copies while saving actor/student exactly
        # as trained.
        if self.cfg.phase == "train":
            hard_copy_(self.adapt_module, self.adapt_ema)
            if self.cfg.use_object_adapt:
                hard_copy_(self.object_adapt, self.object_adapt_ema)
            if hasattr(self, "temporal_depth_gru"):
                hard_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema)

        state = OrderedDict(
            (name, module.state_dict()) for name, module in self.named_children()
        )
        state["last_phase"] = self.cfg.phase
        state["last_iter"] = self.env.current_iter
        # ``last_iter`` is the rollout already completed by this checkpoint.
        # Same-stage resume must continue at the following curriculum iteration.
        state["next_iter"] = int(self.env.current_iter) + 1
        state["q_rng_state"] = self.q_rng.get_state()
        state["sac_action_rng_state"] = self.sac_action_rng.get_state()
        state["sac_rollout_rng_state"] = self.sac_rollout_rng.get_state()
        state["q_update_count"] = self.q_update_count
        state["actor_backend"] = self.actor_backend
        state["training_algorithm"] = (
            FASTSAC_TEACHER_TRAINING_ALGORITHM
            if self.cfg.phase == "train"
            else FASTSAC_STUDENT_TRAINING_ALGORITHM
        )
        state["actor_backend_config"] = self._actor_backend_metadata()
        state["q_backend_config"] = self._q_backend_metadata()
        state["teacher_replay_id"] = self.teacher_replay_id
        state["sac_environment_steps"] = self.sac_environment_steps
        state["sac_rollout_count"] = self.sac_rollout_count
        state["sac_update_count"] = self.sac_update_count
        state["sac_actor_update_count"] = self.sac_actor_update_count
        state["sac_alpha_update_count"] = self.sac_alpha_update_count
        state["teacher_export_started"] = bool(getattr(
            self, "_teacher_export_started", False
        ))
        if self.cfg.phase == "train":
            state["stage1_update_mode"] = dict(FASTSAC_STAGE1_UPDATE_MODE)
            state["stage1_seed_replay_config"] = {
                "storage_ratio": float(getattr(
                    self.cfg, "sac_teacher_seed_storage_ratio", 0.0
                )),
                "sample_ratio": float(getattr(
                    self.cfg, "sac_teacher_seed_sample_ratio", 0.0
                )),
                # The ephemeral replay itself is intentionally not saved.
                "frozen_at_checkpoint": bool(getattr(
                    getattr(self, "teacher_replay", None),
                    "seed_frozen",
                    False,
                )),
            }
            state["stage1_n_step_config"] = self._stage1_n_step_config()
            state["stage1_actor_objective_config"] = (
                self._stage1_actor_objective_config()
            )
            state["stage1_actor_uncertainty_gate_config"] = (
                self._stage1_actor_uncertainty_gate_config()
            )
            state["stage1_conservative_q_config"] = (
                self._stage1_conservative_q_config()
            )
            state["stage1_actor_std_schedule"] = (
                self._teacher_actor_std_schedule_checkpoint_state()
            )
        if hasattr(self, "log_alpha"):
            state["log_alpha"] = self.log_alpha.detach().clone()
        state["optimizer_resume_state"] = self._optimizer_resume_state()
        replay_metadata = self._teacher_replay_checkpoint_metadata()
        if replay_metadata is not None:
            state["teacher_replay_state"] = replay_metadata
        return state

    def load_state_dict(self, state_dict, strict=True):
        algorithm = state_dict.get("training_algorithm")
        allowed_algorithms = (
            {FASTSAC_TEACHER_TRAINING_ALGORITHM}
            if self.cfg.phase == "train"
            else {
                FASTSAC_TEACHER_TRAINING_ALGORITHM,
                FASTSAC_STUDENT_TRAINING_ALGORITHM,
            }
        )
        if algorithm not in allowed_algorithms:
            raise ValueError(
                "Checkpoint is not from the current VAIC FastSAC training path. "
                "Retrain with algo=fastsac_vel_train; old PPO-based FastSAC-hybrid "
                "checkpoints and checkpoints predating the reference-centered/"
                "asymmetric action correction are intentionally "
                f"rejected (training_algorithm={algorithm!r})."
            )
        _validate_pure_fastsac_checkpoint_provenance(state_dict)
        backend = state_dict.get("actor_backend")
        if backend != self.actor_backend:
            raise ValueError(
                "FastSAC actor checkpoint backend does not match the configured "
                f"Stage-2 source: checkpoint={backend!r}, "
                f"expected={self.actor_backend!r}."
            )
        expected = self._actor_backend_metadata()
        actual = state_dict.get("actor_backend_config")
        if actual != expected:
            raise ValueError(
                f"FastSAC actor checkpoint config {actual} does not match {expected}"
            )
        expected_q = self._q_backend_metadata()
        actual_q = state_dict.get("q_backend_config")
        if (
            isinstance(actual_q, dict)
            and "q_action_fusion" in expected_q
        ):
            # Checkpoints written before the opt-in fusion switch necessarily
            # used the exact early-concat backend. Preserve that compatibility
            # while still rejecting them for a late-fusion policy.
            actual_q = dict(actual_q)
            actual_q.setdefault("q_action_fusion", "early")
            actual_q.setdefault("q_action_hidden_dim", 0)
            if "q_input_dim" in expected_q:
                actual_q.setdefault(
                    "q_input_dim", actual_q.get("critic_obs_dim")
                )
            if "q_actuator_context" in expected_q:
                actual_q.setdefault(
                    "q_actuator_context", {"enabled": False}
                )
            actual_q.setdefault(
                "q_action_fusion_semantics",
                FASTSAC_Q_EARLY_FUSION_SEMANTICS,
            )
            if "q_reference_dueling" in expected_q:
                actual_q.setdefault("q_reference_dueling", False)
                actual_q.setdefault(
                    "q_architecture_semantics",
                    FASTSAC_Q_DIRECT_ARCHITECTURE_SEMANTICS,
                )
        if actual_q != expected_q:
            raise ValueError(
                f"FastSAC Q checkpoint config {actual_q} does not match {expected_q}"
            )
        if not state_dict.get("teacher_replay_id"):
            raise ValueError("FastSAC checkpoint is missing its teacher replay id")
        schedule_state = None
        if (
            self.cfg.phase == "train"
            and state_dict.get("last_phase") == self.cfg.phase
        ):
            expected_n_step = self._stage1_n_step_config()
            actual_n_step = state_dict.get("stage1_n_step_config")
            if actual_n_step != expected_n_step:
                raise ValueError(
                    "Stage-1 checkpoint n-step config "
                    f"{actual_n_step!r} does not match {expected_n_step!r}"
                )
            expected_actor_objective = self._stage1_actor_objective_config()
            actual_actor_objective = state_dict.get(
                "stage1_actor_objective_config"
            )
            if actual_actor_objective is None:
                # Before this option existed, Stage 1 necessarily used the
                # original SAC actor objective. Preserve that exact resume only.
                if expected_actor_objective["objective"] != "sac":
                    raise ValueError(
                        "Stage-1 checkpoint predates the configured actor "
                        "objective; restart the reference_awac run"
                    )
            elif actual_actor_objective != expected_actor_objective:
                raise ValueError(
                    "Stage-1 checkpoint actor objective config "
                    f"{actual_actor_objective!r} does not match "
                    f"{expected_actor_objective!r}"
                )
            expected_actor_gate = (
                self._stage1_actor_uncertainty_gate_config()
            )
            actual_actor_gate = state_dict.get(
                "stage1_actor_uncertainty_gate_config"
            )
            if actual_actor_gate is None:
                # Older checkpoints either had no gate or used the unsafe raw
                # kinematic-reference anchor. They are compatible only with
                # the disabled default; enabled behavior must start fresh.
                if expected_actor_gate["enabled"]:
                    raise ValueError(
                        "Stage-1 checkpoint predates the behavior-anchored "
                        "actor uncertainty gate; restart the gated run"
                    )
            elif actual_actor_gate != expected_actor_gate:
                raise ValueError(
                    "Stage-1 checkpoint actor uncertainty gate config "
                    f"{actual_actor_gate!r} does not match "
                    f"{expected_actor_gate!r}"
                )
            expected_conservative_q = self._stage1_conservative_q_config()
            actual_conservative_q = state_dict.get(
                "stage1_conservative_q_config"
            )
            if actual_conservative_q is None:
                # Pre-feature checkpoints had exactly the disabled behavior.
                # They remain resumable only when the new regularizer is also
                # disabled; enabling it mid-run would silently change Q loss.
                if expected_conservative_q["coefficient"] > 0.0:
                    raise ValueError(
                        "Stage-1 checkpoint predates the configured "
                        "conservative Q regularizer; restart the run"
                    )
            elif actual_conservative_q != expected_conservative_q:
                raise ValueError(
                    "Stage-1 checkpoint conservative Q config "
                    f"{actual_conservative_q!r} does not match "
                    f"{expected_conservative_q!r}"
                )
            schedule_state = self._validate_teacher_actor_std_schedule_checkpoint(
                state_dict
            )
        failed = super().load_state_dict(state_dict, strict)
        critical = {
            "actor", "actor_adapt", "qnet", "qnet_target",
            "encoder_priv", "adapt_module", "adapt_ema",
        }
        if self.cfg.use_object_adapt:
            critical.update(("object_adapt", "object_adapt_ema"))
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                f"Failed to restore critical FastSAC modules: {sorted(missing)}"
            )
        if state_dict.get("last_phase") == self.cfg.phase:
            if "log_alpha" in state_dict and hasattr(self, "log_alpha"):
                self.log_alpha.data.copy_(
                    state_dict["log_alpha"].to(
                        device=self.log_alpha.device,
                        dtype=self.log_alpha.dtype,
                    )
                )
            self.sac_environment_steps = int(
                state_dict.get("sac_environment_steps", 0)
            )
            self.sac_rollout_count = int(state_dict.get("sac_rollout_count", 0))
            self.sac_update_count = int(state_dict.get("sac_update_count", 0))
            self.sac_actor_update_count = int(
                state_dict.get("sac_actor_update_count", 0)
            )
            self.sac_alpha_update_count = int(
                state_dict.get("sac_alpha_update_count", 0)
            )
            if self.cfg.phase == "train":
                self._restore_teacher_actor_std_schedule_checkpoint(
                    schedule_state
                )
            self._teacher_export_started = bool(
                state_dict.get("teacher_export_started", False)
            )
            self.env.set_progress(
                int(state_dict.get("next_iter", state_dict.get("last_iter", -1) + 1))
            )
        return failed


class _BCDaggerSACAdapter(nn.Module):
    """Dedicated Stage-2 stochastic state for a deterministic BC actor."""

    def __init__(self, action_dim: int, initial_log_std: float, device):
        super().__init__()
        self.log_std = nn.Parameter(torch.full(
            (int(action_dim),),
            float(initial_log_std),
            device=device,
            dtype=torch.float32,
        ))


class _BCDaggerSACRolloutActor(nn.Module):
    """Execute the configured Stage-2 behavior and expose support diagnostics."""

    in_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
    out_keys = [
        ACTION_KEY,
        "loc",
        "scale",
        STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY,
        STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY,
    ]

    def __init__(
        self, owner: "FastSACVelFinetune", deterministic: bool = True
    ):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.deterministic = bool(deterministic)

    @torch.no_grad()
    def forward(self, td: TensorDict):
        mean_action, dist = self._owner._bc_dagger_behavior_action_and_dist(td)
        if self.deterministic:
            action = mean_action
        else:
            action, _ = dist.rsample_with_log_prob(
                generator=self._owner.sac_rollout_rng
            )
        action_clip = float(self._owner.cfg.sac_bc_action_clip)
        action_in_support = (
            torch.isfinite(action) & (action.abs() <= action_clip)
        ).all()
        if not action_in_support:
            raise RuntimeError(
                "BC-DAgger Stage-2 behavior sampled an action outside its "
                "finite actor/replay/Q support before environment execution"
            )
        absolute_deviation = (action - mean_action).abs()
        td[ACTION_KEY] = action
        td["loc"] = dist.loc
        td["scale"] = dist.scale
        td[STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY] = (
            absolute_deviation.mean(dim=-1)
        )
        td[STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY] = (
            absolute_deviation.amax(dim=-1)
        )
        return td


class FastSACVelFinetune(FastSACVEL):
    """VAIC student actor + HOI-style distributional FastSAC with RLPD replay."""

    def _freeze_stage2_perception(self):
        """Keep latent replay coordinates fixed for the full Stage-2 run."""
        module_names = (
            "encoder_priv",
            "adapt_module",
            "adapt_ema",
            "object_adapt",
            "object_adapt_ema",
            "depth_cnn",
            "temporal_depth_gru",
            "temporal_depth_gru_ema",
            "dr_estimator",
        )
        for name in module_names:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(False)
        # These optimizers belong to the mutable PPO perception path. Keeping
        # them out of the Stage-2 registry also makes resume manifests honest.
        self.opt_adapt = None
        if hasattr(self, "opt_dr_estimator"):
            self.opt_dr_estimator = None

    def _stage2_actor_parameters(self):
        if not self._uses_bc_dagger_finetune_source():
            return tuple(self.actor_adapt.parameters())
        unused_ppo_std = self._ppo_actor_core(self.actor_adapt).actor_std
        parameters = [
            parameter
            for parameter in self.actor_adapt.parameters()
            if parameter is not unused_ppo_std
        ]
        parameters.extend(self.bc_dagger_sac_adapter.parameters())
        return tuple(parameters)

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.enable_residual_distillation:
            raise ValueError(
                "fastsac_vel_finetune requires enable_residual_distillation=false: "
                "SAC already optimizes actor_adapt, so a second distillation optimizer "
                "would apply a conflicting objective and independent Adam state."
            )
        _validate_fastsac_finetune_config(cfg)
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if bool(cfg.sac_freeze_perception):
            self._freeze_stage2_perception()
        if self._uses_bc_dagger_finetune_source():
            self.bc_dagger_actor_anchor = copy.deepcopy(
                self.actor_adapt
            ).requires_grad_(False)
        self._stage2_trainable_actor_parameters = (
            self._stage2_actor_parameters()
        )
        self.online_replay = OnlineReplay(cfg.online_buffer_capacity, device=self.device)
        self.offline_replay = None
        self.sac_actor_optimizer = torch.optim.AdamW(
            self._stage2_trainable_actor_parameters, lr=cfg.sac_actor_lr,
            weight_decay=cfg.q_weight_decay, betas=(0.9, 0.95),
        )
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay,
            betas=(0.9, 0.95), fused=str(device).startswith("cuda"),
        )
        self.log_alpha = nn.Parameter(torch.tensor(
            float(np.log(cfg.sac_alpha_init)), device=device, dtype=torch.float32,
        ))
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.sac_alpha_lr, betas=(0.9, 0.95))
        self.target_entropy = _fastsac_target_entropy(
            self._fastsac_action_low,
            self._fastsac_action_high,
            cfg.sac_target_entropy_ratio,
        )
        self.sac_update_count = 0
        # None means no actor optimizer step has yet passed every release gate.
        # The first applied actor update records its numbered Q update and is
        # the zero point of the effective-alpha ramp.
        self._stage2_actor_release_q_update = None

    def configure_offline_replay(self, path):
        if getattr(self, "_replay_vecnorm", None) is None:
            raise RuntimeError(
                "Student FastSAC replay requires configure_replay_vecnorm() "
                "before loading offline data."
            )
        if path is None:
            raise ValueError(
                "FastSAC finetune requires teacher_replay_buffer_path "
                "(or algo.teacher_buffer_path), or a checkpoint run containing "
                "teacher_replay_buffer.h5."
            )
        # Stage-2 RLPD treats the H5 as an offline dataset, not as an exact
        # continuation of the checkpoint's live FIFO.  A compatible dataset
        # from another FastSAC run/iteration is therefore valid.  Keep all
        # semantic/schema checks below, but intentionally do not pass the
        # checkpoint replay id or snapshot manifest to OfflineReplayH5.
        if self._uses_bc_dagger_finetune_source():
            self.offline_replay = BCDaggerOfflineReplayH5(
                path,
                self._q_actor_dim,
                self._q_critic_dim,
                self.action_dim,
                # The doubled Skateboard FIFO is about 24 GiB and cannot share
                # a 32-GiB GPU with Isaac, cameras and the networks. Keep the
                # immutable dataset on host and transfer sampled rows only.
                device="cpu",
                max_size=self.cfg.teacher_buffer_capacity,
                seed=self.cfg.teacher_buffer_seed,
                expected_actor_obs_keys=self.q_actor_keys,
                expected_critic_obs_keys=self.q_critic_keys,
                expected_vecnorm_fingerprint=(
                    self._replay_vecnorm_fingerprint
                ),
                expected_action_clip=float(self.cfg.sac_bc_action_clip),
            )
        else:
            self.offline_replay = OfflineReplayH5(
                path, self._q_actor_dim, self._q_critic_dim, self.action_dim,
                device=self.device, max_size=self.cfg.teacher_buffer_capacity,
                seed=self.cfg.teacher_buffer_seed,
                expected_actor_backend=self.actor_backend,
                expected_actor_obs_keys=self.q_actor_keys,
                expected_critic_obs_keys=self.q_critic_keys,
                expected_q_action_fusion=getattr(
                    self.cfg, "q_action_fusion", "early"
                ),
                expected_q_action_hidden_dim=_q_action_hidden_dim(
                    getattr(self.cfg, "q_hidden_dim", 768),
                    getattr(self.cfg, "q_action_fusion", "early"),
                ),
                expected_q_action_coordinates=getattr(
                    self.cfg, "q_action_coordinates", "absolute"
                ),
                expected_q_reference_dueling=getattr(
                    self.cfg, "q_reference_dueling", False
                ),
                expected_q_actuator_context=(
                    getattr(
                        self,
                        "_q_actuator_context_metadata_value",
                        {"enabled": False},
                    )
                ),
            )
        self.offline_replay_source_path = os.path.abspath(os.fspath(path))

    def get_offline_replay_path(self):
        return getattr(self, "offline_replay_source_path", None)

    def _uses_pretrained_q(self) -> bool:
        return bool(getattr(self.cfg, "load_pretrained_q", True))

    def _stage2_schedule_config(self):
        names = (
            "load_pretrained_q",
            "sac_learning_starts",
            "sac_batch_size",
            "sac_updates_per_env_step",
            "sac_policy_frequency",
            "q_lr",
            "sac_actor_lr",
            "sac_alpha_lr",
            "sac_alpha_init",
            "sac_alpha_ramp_q_updates",
            "sac_target_entropy_ratio",
            "sac_tau",
            "sac_max_grad_norm",
            "sac_actor_learning_starts_q_updates",
            "sac_actor_confidence_gate",
            "sac_actor_gate_disagreement_multiplier",
            "sac_actor_gate_min_accept_fraction",
            "sac_actor_gate_absolute_margin",
            "sac_bc_action_clip",
            "sac_entropy_reference_scale",
            "sac_bc_initial_action_std",
            "sac_bc_log_std_min",
            "sac_bc_log_std_max",
            "sac_deterministic_rollout",
            "sac_bc_anchor_coef_start",
            "sac_bc_anchor_coef_end",
            "sac_bc_anchor_decay_q_updates",
            "sac_bc_anchor_huber_delta",
            "sac_freeze_perception",
            "teacher_buffer_ratio",
        )
        return {
            "version": 4,
            "entropy_semantics": (
                "fixed_reference_action_density_hard_q_bridge_then_"
                "post_release_linear_effective_alpha_v1"
            ),
            "actor_confidence_gate_semantics": (
                FASTSAC_STAGE2_ACTOR_CONFIDENCE_GATE_SEMANTICS
            ),
            **{name: getattr(self.cfg, name) for name in names},
        }

    def _checkpoint_for_q_transfer(self, state_dict, *, load_q=None):
        """Return checkpoint modules with Q entries selected by configuration.

        PPOVEL's compatibility loader visits every registered child.  Supplying
        the constructor-created Q states here avoids both loading Stage-1 Q
        tensors and treating intentionally fresh Q modules as missing.
        """
        if load_q is None:
            load_q = self._uses_pretrained_q()
        if load_q:
            return state_dict
        load_state = dict(state_dict)
        load_state["qnet"] = self.qnet.state_dict()
        load_state["qnet_target"] = self.qnet_target.state_dict()
        load_state["q_update_count"] = 0
        return load_state

    def _finalize_fresh_q_initialization(self):
        """Make fresh target critics and optimizer state match fresh online Q."""
        hard_copy_(self.qnet, self.qnet_target)
        self.qnet_target.requires_grad_(False)
        # A same-process test or a future loader refactor may have populated
        # Adam moments before reaching this point.  Never pair them with a
        # newly initialized critic.
        if hasattr(self, "opt_q"):
            self.opt_q.state.clear()
        self.q_update_count = 0

    def load_state_dict(self, state_dict, strict=True):
        load_pretrained_q = self._uses_pretrained_q()
        same_stage_resume = state_dict.get("last_phase") == "finetune"
        load_q = load_pretrained_q or same_stage_resume
        if load_q and "qnet" not in state_dict:
            raise KeyError(
                "Checkpoint has no transferable Q1/Q2. Use a checkpoint trained "
                "with scripts/bc_dagger.py or algo=fastsac_vel_train."
            )
        if state_dict.get("training_algorithm") == BC_DAGGER_TRAINING_ALGORITHM:
            return self._load_bc_dagger_state_dict(state_dict, strict)
        if state_dict.get("training_algorithm") == (
            BC_DAGGER_LEGACY_TRAINING_ALGORITHM
        ):
            raise ValueError(
                "Legacy BC-DAgger Q checkpoints predate IQL critic training. "
                "Run scripts/bc_dagger.py again before Stage-2 FastSAC."
            )
        if (
            same_stage_resume
            and self._uses_bc_dagger_finetune_source()
            and not bool(self.cfg.sac_deterministic_rollout)
            and "sac_rollout_rng_state" not in state_dict
        ):
            raise ValueError(
                "Stochastic BC-DAgger Stage-2 resume requires the checkpointed "
                "rollout RNG state"
            )
        if same_stage_resume:
            actual_schedule = state_dict.get("stage2_schedule_config")
            expected_schedule = self._stage2_schedule_config()
            if actual_schedule != expected_schedule:
                raise ValueError(
                    "Stage-2 checkpoint schedule/anchor/perception config does "
                    f"not match: checkpoint={actual_schedule!r}, "
                    f"current={expected_schedule!r}"
                )
            if "stage2_actor_release_q_update" not in state_dict:
                raise ValueError(
                    "Stage-2 checkpoint is missing the actor-release alpha-ramp "
                    "origin"
                )
        load_state = self._checkpoint_for_q_transfer(state_dict, load_q=load_q)
        failed = super().load_state_dict(load_state, strict)
        if load_q and (
            "qnet" in failed or "qnet_target" in failed
        ):
            raise RuntimeError("Failed to load teacher FastSAC Q1/Q2 from checkpoint")
        if not load_q:
            self._finalize_fresh_q_initialization()
            failed = [
                name for name in failed
                if name not in {"qnet", "qnet_target"}
            ]
        source_phase = state_dict.get("last_phase")
        if source_phase == "finetune":
            perception_modules = {"encoder_priv", "adapt_module", "adapt_ema"}
            if self.cfg.use_object_adapt:
                perception_modules.update(("object_adapt", "object_adapt_ema"))
            if hasattr(self, "temporal_depth_gru"):
                perception_modules.update(
                    ("depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema")
                )
            missing_perception = perception_modules.intersection(failed)
            if missing_perception:
                raise RuntimeError(
                    "Failed to restore student perception/EMA modules: "
                    f"{sorted(missing_perception)}"
                )
        if (
            load_pretrained_q
            and not same_stage_resume
            and self.q_update_count < 1
        ):
            raise RuntimeError(
                "Checkpoint Q1/Q2 have not been trained by FastSAC yet."
            )
        if source_phase == "train":
            # Student SAC owns a new entropy process and update cadence. Transfer
            # teacher/student actor and Q weights, but start alpha/counters fresh.
            self.log_alpha.data.fill_(float(np.log(self.cfg.sac_alpha_init)))
            self.q_update_count = 0
            self.sac_update_count = 0
            # Stage 2 is a new learning process, just like its fresh alpha and
            # optimizer state.  Do not inherit the teacher's advanced sampling
            # streams or silently ignore the Stage-2 q_seed.
            self.q_rng.manual_seed(int(self.cfg.q_seed))
            self.sac_action_rng.manual_seed(int(self.cfg.q_seed) + 1)
            self.sac_rollout_rng.manual_seed(int(self.cfg.q_seed) + 2)
            self._stage2_actor_release_q_update = None
        elif source_phase == "finetune":
            release_q_update = state_dict["stage2_actor_release_q_update"]
            if release_q_update is not None:
                if (
                    isinstance(release_q_update, bool)
                    or not isinstance(release_q_update, (int, np.integer))
                    or int(release_q_update) < 1
                    or int(release_q_update) > int(self.q_update_count)
                ):
                    raise ValueError(
                        "Stage-2 checkpoint actor-release alpha-ramp origin is "
                        "invalid"
                    )
                release_q_update = int(release_q_update)
            if (
                (release_q_update is None)
                != (int(getattr(self, "sac_actor_update_count", 0)) == 0)
            ):
                raise ValueError(
                    "Stage-2 checkpoint actor-update count and alpha-ramp "
                    "release origin are inconsistent"
                )
            self._stage2_actor_release_q_update = release_q_update
            if self._uses_bc_dagger_finetune_source():
                missing_bridge = {
                    "bc_dagger_sac_adapter",
                    "bc_dagger_actor_anchor",
                }.intersection(failed)
                if missing_bridge:
                    raise RuntimeError(
                        "Failed to restore Stage-2 BC bridge modules: "
                        f"{sorted(missing_bridge)}"
                    )
            replay_state = state_dict.get("offline_teacher_replay_state")
            if replay_state is not None:
                # Retain provenance for checkpoint logging, but do not require
                # the next stage-2 invocation to select the identical dataset.
                self._loaded_teacher_replay_metadata = copy.deepcopy(replay_state)
            iql_source = state_dict.get("bc_dagger_iql_source")
            if isinstance(iql_source, dict):
                self._bc_dagger_iql_source = copy.deepcopy(iql_source)
        return failed

    def _load_bc_dagger_state_dict(self, state_dict, strict=True):
        load_pretrained_q = self._uses_pretrained_q()
        if not self._uses_bc_dagger_finetune_source():
            raise ValueError(
                "A BC-DAgger checkpoint requires "
                "algo.finetune_checkpoint_source=bc_dagger (or auto detection)"
            )
        if state_dict.get("actor_backend") != BC_DAGGER_ACTOR_BACKEND:
            raise ValueError("BC-DAgger checkpoint actor backend mismatch")
        if load_pretrained_q and state_dict.get(
            "critic_learning_semantics"
        ) != BC_DAGGER_IQL_CRITIC_SEMANTICS:
            raise ValueError("BC-DAgger checkpoint IQL critic semantics mismatch")
        if state_dict.get("actor_learning_semantics") != (
            BC_DAGGER_ACTOR_LEARNING_SEMANTICS
        ):
            raise ValueError(
                "BC-DAgger checkpoint actor was not trained by pure DAgger BC"
            )
        if load_pretrained_q and "iql_value" not in state_dict:
            raise KeyError("BC-DAgger checkpoint is missing its IQL value network")
        checkpoint_fingerprint = state_dict.get("vecnorm_fingerprint")
        if (
            checkpoint_fingerprint is not None
            and str(checkpoint_fingerprint)
            != self._replay_vecnorm_fingerprint
        ):
            raise ValueError(
                "BC-DAgger checkpoint VecNorm fingerprint does not match "
                "the Stage-2 checkpoint normalizer"
            )
        backend_cfg = state_dict.get("dagger_backend_config")
        if not isinstance(backend_cfg, dict):
            raise ValueError("BC-DAgger checkpoint lacks backend configuration")
        source_action_clip = backend_cfg.get("dagger_action_clip")
        if (
            source_action_clip is None
            or not math.isclose(
                float(source_action_clip),
                float(self.cfg.sac_bc_action_clip),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "BC-DAgger checkpoint action clip does not match the Stage-2 "
                "actor/replay/Q support"
            )
        if load_pretrained_q and "iql_expectile" not in backend_cfg:
            raise ValueError(
                "BC-DAgger checkpoint lacks its IQL expectile configuration"
            )
        expected_q = {
            "q_hidden_dim": int(self.cfg.q_hidden_dim),
            "q_num_atoms": int(self.cfg.q_num_atoms),
            "q_v_min": float(self.cfg.q_v_min),
            "q_v_max": float(self.cfg.q_v_max),
            "q_layer_norm": bool(self.cfg.q_layer_norm),
        }
        actual_q = {name: backend_cfg.get(name) for name in expected_q}
        if load_pretrained_q and actual_q != expected_q:
            raise ValueError(
                f"BC-DAgger Q configuration {actual_q} does not match "
                f"Stage 2 {expected_q}"
            )
        source_q_updates = int(state_dict.get("q_update_count", 0))
        if load_pretrained_q and source_q_updates < 1:
            raise RuntimeError("BC-DAgger checkpoint Q1/Q2 were never updated")
        source_value_updates = int(
            state_dict.get("iql_value_update_count", 0)
        )
        if load_pretrained_q and source_value_updates < 1:
            raise RuntimeError("BC-DAgger checkpoint IQL V was never updated")

        # The compatibility backend deliberately retained PPOVEL's actor tree,
        # so the BC actor and depth/EMA modules load without shape translation.
        # Q entries below are either the exact checkpoint weights or the
        # constructor-created states selected by load_pretrained_q.
        load_state = self._checkpoint_for_q_transfer(state_dict)
        load_state = dict(load_state)
        # A BC-Dagger checkpoint predates the learning-only SAC adapter and
        # frozen anchor. Seed those registered Stage-2 children locally, then
        # capture the transferred actor into the anchor immediately after load.
        load_state["bc_dagger_sac_adapter"] = (
            self.bc_dagger_sac_adapter.state_dict()
        )
        load_state["bc_dagger_actor_anchor"] = (
            self.bc_dagger_actor_anchor.state_dict()
        )
        failed = PPOVEL.load_state_dict(self, load_state, strict)
        critical = {
            "actor_adapt",
            "qnet",
            "qnet_target",
            "encoder_priv",
            "adapt_module",
            "adapt_ema",
        }
        if self.cfg.use_object_adapt:
            critical.update(("object_adapt", "object_adapt_ema"))
        if hasattr(self, "temporal_depth_gru"):
            critical.update((
                "depth_cnn",
                "temporal_depth_gru",
                "temporal_depth_gru_ema",
            ))
        missing = critical.intersection(failed)
        if missing:
            raise RuntimeError(
                "Failed to transfer critical BC-DAgger modules: "
                f"{sorted(missing)}"
            )
        if not load_pretrained_q:
            self._finalize_fresh_q_initialization()
            failed = [
                name for name in failed
                if name not in {"qnet", "qnet_target"}
            ]

        # Preserve the complete BC actor bit-for-bit, including its unused
        # positive PPO std, and snapshot the transferred mean policy as the
        # fixed behavior anchor for gradual Stage-2 release.
        hard_copy_(self.actor_adapt, self.bc_dagger_actor_anchor)
        self.bc_dagger_actor_anchor.requires_grad_(False)
        self._ppo_actor_core(self.actor_adapt).actor_std.requires_grad_(False)
        failed = [
            name for name in failed
            if name not in {
                "bc_dagger_sac_adapter",
                "bc_dagger_actor_anchor",
            }
        ]
        self.qnet_target.requires_grad_(False)
        self.q_update_count = 0
        self.sac_update_count = 0
        self.sac_actor_update_count = 0
        self.sac_alpha_update_count = 0
        self.sac_environment_steps = 0
        self.sac_rollout_count = 0
        self.q_rng.manual_seed(int(self.cfg.q_seed))
        self.sac_action_rng.manual_seed(int(self.cfg.q_seed) + 1)
        self.sac_rollout_rng.manual_seed(int(self.cfg.q_seed) + 2)
        self.log_alpha.data.fill_(float(np.log(self.cfg.sac_alpha_init)))
        self._stage2_actor_release_q_update = None
        self.teacher_replay_id = str(
            state_dict.get("teacher_replay_id", self.teacher_replay_id)
        )
        self._loaded_checkpoint_phase = "bc_dagger"
        self._loaded_teacher_replay_metadata = copy.deepcopy(
            state_dict.get("teacher_replay_state")
        )
        if load_pretrained_q:
            self._bc_dagger_iql_source = {
                "critic_learning_semantics": BC_DAGGER_IQL_CRITIC_SEMANTICS,
                "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
                "source_q_updates": source_q_updates,
                "source_value_updates": source_value_updates,
                "expectile": float(backend_cfg["iql_expectile"]),
                "value_network_stage2_usage": "discarded_stage1_bootstrap_only",
            }
            logging.info(
                "Transferred pure-BC DAgger actor/perception and IQL-pretrained "
                "Q1/Q2 + targets into a fresh Stage-2 SAC optimizer/entropy "
                "process; the Stage-1-only V network was intentionally discarded."
            )
        else:
            self._bc_dagger_iql_source = {
                "critic_learning_semantics": state_dict.get(
                    "critic_learning_semantics"
                ),
                "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
                "source_q_updates": source_q_updates,
                "source_value_updates": source_value_updates,
                "expectile": backend_cfg.get("iql_expectile"),
                "value_network_stage2_usage": "not_loaded",
                "q_weights_stage2_usage": "discarded_fresh_q_seed",
                "q_seed": int(self.cfg.q_seed),
            }
            logging.info(
                "Transferred the pure-BC DAgger actor/perception modules but "
                "intentionally skipped Stage-1 IQL Q/V weights; Stage-2 Q1/Q2 "
                "are fresh q_seed=%d initializations and both target critics "
                "are frozen exact copies.",
                int(self.cfg.q_seed),
            )
        return failed

    def state_dict(self):
        state = super().state_dict()
        state["stage2_schedule_config"] = self._stage2_schedule_config()
        state["stage2_actor_release_q_update"] = getattr(
            self, "_stage2_actor_release_q_update", None
        )
        if self.offline_replay is not None:
            state["offline_teacher_replay_state"] = copy.deepcopy(
                self.offline_replay.snapshot_metadata
            )
        if hasattr(self, "_bc_dagger_iql_source"):
            state["bc_dagger_iql_source"] = copy.deepcopy(
                self._bc_dagger_iql_source
            )
        return state

    @torch.no_grad()
    def _prepare_student_final_state(self, td: TensorDict):
        """Run the unchanged VAIC EMA perception path on a true next state."""
        if hasattr(self, "temporal_depth_gru_ema"):
            self.temporal_depth_gru_ema(td)
        else:
            td["_depth_feature"] = torch.zeros(
                *td.batch_size,
                self.depth_feature_dim,
                device=td.device,
                dtype=td[OBS_KEY].dtype,
            )
        if self.cfg.use_object_adapt:
            self.object_adapt_ema(td)
            self.object_pred_transform(td)
        self.adapt_ema(td)
        result = {
            "next_observations": self._cat_replay_sources(
                td, self.q_actor_keys
            ).clone(),
            "next_critic_observations": self._cat_replay_sources(
                td, self.q_critic_keys
            ).clone(),
        }
        if self._q_requires_reference_actions():
            result[NEXT_TEACHER_REF_ACTION_FIELD] = self._replay_source(
                td, REF_JPOS_KEY
            ).clone()
        return result

    @torch.no_grad()
    def capture_truncation_final_observations(self, td: TensorDict, step: int):
        truncations = _vaic_truncation_mask(td).reshape(-1).bool()
        if not truncations.any():
            return
        env_indices = truncations.nonzero(as_tuple=False).squeeze(-1)
        final_values = self._prepare_student_final_state(
            td["next"][env_indices].clone()
        )
        final_values["indices"] = (
            env_indices * int(self.cfg.train_every) + int(step)
        )
        self._truncation_final_batches.append(final_values)

    @torch.no_grad()
    def capture_rollout_final_observation(self, carry: TensorDict):
        self._rollout_final_batch = self._prepare_student_final_state(carry.clone())

    def _student_transition_chunks(self, td: TensorDict):
        if self._rollout_final_batch is None:
            raise RuntimeError(
                "Student FastSAC needs the final rollout observation. The train "
                "loop must call capture_rollout_final_observation(carry)."
            )
        n, t = td.batch_size
        if int(t) != int(self.cfg.train_every):
            raise ValueError(
                f"Rollout length {t} does not match train_every={self.cfg.train_every}."
            )
        actuator_contexts = self._consume_rollout_q_actuator_contexts(int(t))
        final_batch = self._rollout_final_batch
        self._rollout_final_batch = None
        truncation_batches = self._truncation_final_batches
        self._truncation_final_batches = []
        truncation_finals = None
        if truncation_batches:
            truncation_finals = {
                key: torch.cat(
                    [batch[key] for batch in truncation_batches], dim=0
                )
                for key in truncation_batches[0]
            }
            indices = truncation_finals["indices"].long()
            if (indices < 0).any() or (indices >= n * t).any():
                raise IndexError("Captured truncation index is outside the rollout")
            valid_flat = (
                td["step_count"].reshape(n * t)
                > STUDENT_REPLAY_MIN_STEP_COUNT
            )
            self._last_truncation_finals_used = int(
                valid_flat[indices].sum().item()
            )
        else:
            self._last_truncation_finals_used = 0

        for step in range(int(t)):
            current = td[:, step]
            if step + 1 < int(t):
                following = td[:, step + 1]
                next_values = {
                    "next_observations": self._cat_replay_sources(
                        following, self.q_actor_keys
                    ).reshape(n, self._q_actor_dim),
                    "next_critic_observations": self._cat_replay_sources(
                        following, self.q_critic_keys
                    ).reshape(n, self._q_critic_dim),
                }
                if self._q_requires_reference_actions():
                    next_values[NEXT_TEACHER_REF_ACTION_FIELD] = (
                        self._replay_source(
                            following, REF_JPOS_KEY
                        ).reshape(n, self.action_dim)
                    )
            else:
                next_values = final_batch
            transitions = {
                "observations": self._cat_replay_sources(
                    current, self.q_actor_keys
                ).reshape(n, self._q_actor_dim),
                "critic_observations": self._cat_replay_sources(
                    current, self.q_critic_keys
                ).reshape(n, self._q_critic_dim),
                "actions": current[ACTION_KEY].reshape(n, self.action_dim),
                "rewards": self._scalarize_sac_reward(
                    current[REWARD_KEY]
                ).reshape(n),
                "dones": current[DONE_KEY].reshape(n).bool(),
                "truncations": _vaic_truncation_mask(current).reshape(n).bool(),
                "discounts": current["next", "discount"].reshape(n),
                **next_values,
            }
            if self._q_requires_reference_actions():
                transitions[TEACHER_REF_ACTION_FIELD] = self._replay_source(
                    current, REF_JPOS_KEY
                ).reshape(n, self.action_dim)
            if actuator_contexts is not None:
                context = self._transition_q_actuator_context(
                    actuator_contexts[:, step], n, current.device
                )
                transitions[TEACHER_ACTUATOR_CONTEXT_FIELD] = context
                transitions[NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD] = context
            if truncation_finals is not None:
                flat_indices = truncation_finals["indices"].long()
                selected = flat_indices.remainder(int(t)) == step
                if selected.any():
                    env_indices = flat_indices[selected].div(
                        int(t), rounding_mode="floor"
                    )
                    for key, values in truncation_finals.items():
                        if key == "indices":
                            continue
                        transitions[key] = transitions[key].clone()
                        transitions[key].index_copy_(
                            0, env_indices, values[selected]
                        )
            transitions, _ = _filter_replay_rows(
                current, transitions, STUDENT_REPLAY_MIN_STEP_COUNT
            )
            yield transitions

    def _actor_td_from_flat(self, actor_obs):
        vel_dim = self.observation_spec[VEL_CMD_KEY].shape[-1]
        policy_dim = self.observation_spec[OBS_KEY].shape[-1]
        return TensorDict({
            VEL_CMD_KEY: actor_obs[..., :vel_dim],
            OBS_KEY: actor_obs[..., vel_dim:vel_dim + policy_dim],
            PRIV_PRED_KEY: actor_obs[..., vel_dim + policy_dim:],
        }, batch_size=actor_obs.shape[:-1], device=actor_obs.device)

    def _actor_dist_from_flat(self, actor_obs):
        td = self._actor_td_from_flat(actor_obs)
        if self._uses_bc_dagger_finetune_source():
            return self._bc_dagger_actor_dist_from_td(td)
        return self.actor_adapt.get_dist(td)

    def _bc_dagger_behavior_action(self, td: TensorDict, actor=None):
        """Apply the exact finite-value guard and clamp used by DAgger."""
        actor = self.actor_adapt if actor is None else actor
        mean_action = actor.get_dist(td).mean
        action_clip = float(self.cfg.sac_bc_action_clip)
        return torch.nan_to_num(
            mean_action,
            nan=0.0,
            posinf=action_clip,
            neginf=-action_clip,
        ).clamp(-action_clip, action_clip)

    def _bc_dagger_behavior_action_and_dist(self, td: TensorDict):
        # Center the dedicated Stage-2 distribution on the exact clipped DAgger
        # behavior. The same object is used by train collection and SAC learning;
        # PPO actor_std remains untouched and unused.
        mean_action = self._bc_dagger_behavior_action(td)
        action_low = torch.as_tensor(
            self._fastsac_action_low,
            device=mean_action.device,
            dtype=mean_action.dtype,
        )
        action_high = torch.as_tensor(
            self._fastsac_action_high,
            device=mean_action.device,
            dtype=mean_action.dtype,
        )
        action_scale = (action_high - action_low) * 0.5
        action_bias = (action_high + action_low) * 0.5
        normalized = ((mean_action - action_bias) / action_scale).clamp(
            -1.0 + FASTSAC_REFERENCE_EPS,
            1.0 - FASTSAC_REFERENCE_EPS,
        )
        loc = torch.atanh(normalized)
        log_std = self.bc_dagger_sac_adapter.log_std.clamp(
            float(self.cfg.sac_bc_log_std_min),
            float(self.cfg.sac_bc_log_std_max),
        )
        scale = log_std.exp().expand_as(loc)
        return mean_action, self.dist_cls(loc, scale)

    def _bc_dagger_actor_dist_from_td(self, td: TensorDict):
        return self._bc_dagger_behavior_action_and_dist(td)[1]

    def get_rollout_policy(self, mode="train"):
        if not self._uses_bc_dagger_finetune_source():
            return super().get_rollout_policy(mode)
        modules = []
        has_depth = hasattr(self, "temporal_depth_gru")
        if has_depth:
            modules.append(self.temporal_depth_gru_ema)
        else:
            modules.append(ZeroDepthInjector(self.depth_feature_dim, self.device))
        if self.cfg.use_object_adapt:
            modules.append(self.object_adapt_ema)
            modules.append(self.object_pred_transform)
        modules.append(self.adapt_ema)
        deterministic = (
            mode != "train" or bool(self.cfg.sac_deterministic_rollout)
        )
        modules.append(_BCDaggerSACRolloutActor(
            self, deterministic=deterministic
        ))
        out_keys = [
            ACTION_KEY,
            PRIV_PRED_KEY,
            STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY,
            STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY,
        ]
        if self.cfg.adapt_module == "gru":
            out_keys.append(("next", "adapt_hx"))
        if has_depth:
            out_keys.append(("next", "depth_hx"))
        return Seq(*modules, selected_out_keys=out_keys)

    def _prepare_student_learning_batch(self, batch):
        """Normalize raw online/offline RLPD observations at sample time."""
        snapshot = self._vecnorm_snapshot()
        prepared = dict(batch)
        pre_normalized = prepared.pop(PRE_NORMALIZED_REPLAY_KEY, None)
        for field in ("observations", "next_observations"):
            normalized = self._normalize_replay_flat(
                batch[field],
                self.q_actor_keys,
                self._q_actor_widths,
                snapshot,
            )
            prepared[field] = (
                normalized
                if pre_normalized is None
                else torch.where(
                    pre_normalized.reshape(-1, 1), batch[field], normalized
                )
            )
        for field in ("critic_observations", "next_critic_observations"):
            normalized = self._normalize_replay_flat(
                batch[field],
                self.q_critic_keys,
                self._q_critic_widths,
                snapshot,
            )
            prepared[field] = (
                normalized
                if pre_normalized is None
                else torch.where(
                    pre_normalized.reshape(-1, 1), batch[field], normalized
                )
            )
        return prepared

    def _student_replay_fields(self):
        fields = list(TEACHER_REPLAY_FIELDS)
        if self._q_requires_reference_actions():
            fields.extend((
                TEACHER_REF_ACTION_FIELD,
                NEXT_TEACHER_REF_ACTION_FIELD,
            ))
        if self._q_conditions_on_actuator_state():
            fields.extend((
                TEACHER_ACTUATOR_CONTEXT_FIELD,
                NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD,
            ))
        return tuple(fields)

    def _stage2_replay_mix_counts(self):
        """Return the exact total/online/offline row counts for one Q draw."""
        total = int(self.cfg.sac_batch_size)
        offline = round(total * float(self.cfg.teacher_buffer_ratio))
        return {
            "total": total,
            "online": total - offline,
            "offline": offline,
        }

    @staticmethod
    def _stage2_replay_source_masks(batch):
        """Return offline/online masks and whether batch provenance is known."""
        rewards = batch["rewards"]
        row_count = int(rewards.shape[0])
        marker = batch.get(STAGE2_OFFLINE_SOURCE_KEY)
        if marker is None:
            empty = torch.zeros(
                row_count, dtype=torch.bool, device=rewards.device
            )
            return empty, empty, False
        if (
            not isinstance(marker, torch.Tensor)
            or marker.dtype != torch.bool
            or marker.ndim != 1
            or marker.shape[0] != row_count
            or marker.device != rewards.device
        ):
            raise ValueError(
                "Stage-2 replay source marker must be a same-device boolean "
                "vector with one entry per reward row"
            )
        return marker, ~marker, True

    def _mix_batch(self):
        # Sampling is with replacement, matching HOI's global 8192-row batch
        # even before the online FIFO itself contains 8192 distinct rows.
        mix_counts = self._stage2_replay_mix_counts()
        total = mix_counts["total"]
        online_count = mix_counts["online"]
        offline_count = mix_counts["offline"]
        if online_count and self.online_replay.size < 1:
            raise RuntimeError("Cannot train student FastSAC from empty online replay")
        parts = []
        pre_normalized_parts = []
        offline_source_parts = []
        if online_count:
            online = self.online_replay.sample(
                online_count, self.device, generator=self.q_rng
            )
            parts.append(online)
            pre_normalized_parts.append(torch.zeros(
                online_count, dtype=torch.bool, device=self.device
            ))
            offline_source_parts.append(torch.zeros(
                online_count, dtype=torch.bool, device=self.device
            ))
        if offline_count:
            if self.offline_replay is None:
                raise RuntimeError("Offline teacher replay was not configured")
            offline = self.offline_replay.sample(offline_count, self.device)
            parts.append(offline)
            pre_normalized_parts.append(torch.full(
                (offline_count,),
                bool(getattr(
                    self.offline_replay,
                    "observations_pre_normalized",
                    False,
                )),
                dtype=torch.bool,
                device=self.device,
            ))
            offline_source_parts.append(torch.ones(
                offline_count, dtype=torch.bool, device=self.device
            ))
        keys = self._student_replay_fields()
        mixed = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
        pre_normalized = torch.cat(pre_normalized_parts, dim=0)
        if pre_normalized.any():
            mixed[PRE_NORMALIZED_REPLAY_KEY] = pre_normalized
        mixed[STAGE2_OFFLINE_SOURCE_KEY] = torch.cat(offline_source_parts)
        permutation = torch.randperm(
            total, device=self.device, generator=self.q_rng
        )
        return {key: value[permutation] for key, value in mixed.items()}

    def _stage2_actor_is_active(self, q_updates=None) -> bool:
        if q_updates is None:
            q_updates = self.q_update_count
        return int(q_updates) > int(
            getattr(self.cfg, "sac_actor_learning_starts_q_updates", 0)
        )

    def _mark_stage2_actor_released(self, q_update: int) -> int:
        """Persist the first numbered Q update that applied an actor step."""
        if (
            isinstance(q_update, bool)
            or not isinstance(q_update, (int, np.integer))
            or int(q_update) < 1
        ):
            raise ValueError("Stage-2 actor release Q update must be positive")
        q_update = int(q_update)
        current = getattr(self, "_stage2_actor_release_q_update", None)
        if current is None:
            self._stage2_actor_release_q_update = q_update
            return q_update
        current = int(current)
        if q_update < current:
            raise RuntimeError(
                "Stage-2 actor release marker cannot move backward: "
                f"current={current}, requested={q_update}"
            )
        return current

    def _stage2_alpha_ramp_progress(self, q_updates=None) -> float:
        """Return the post-release effective-alpha multiplier in [0, 1]."""
        release = getattr(self, "_stage2_actor_release_q_update", None)
        if release is None:
            return 0.0
        if q_updates is None:
            q_updates = self.q_update_count
        elapsed = max(0, int(q_updates) - int(release))
        ramp_updates = int(getattr(
            self.cfg, "sac_alpha_ramp_q_updates", 20_000
        ))
        if ramp_updates < 1:
            raise ValueError("sac_alpha_ramp_q_updates must be positive")
        return min(elapsed / float(ramp_updates), 1.0)

    def _stage2_effective_alpha(self, q_updates=None) -> torch.Tensor:
        """Use one ramped temperature for Q backup, actor, and alpha dual."""
        return self.log_alpha.exp() * self._stage2_alpha_ramp_progress(q_updates)

    def _stage2_q_target_uses_stochastic_policy(self, q_updates=None) -> bool:
        """Match Q policy evaluation to the configured behavior support."""
        if q_updates is None:
            q_updates = self.q_update_count
        return (
            self._stage2_actor_is_active(q_updates)
            or (
                self._uses_bc_dagger_finetune_source()
                and not bool(self.cfg.sac_deterministic_rollout)
            )
        )

    def _sac_actor_update_is_due(
        self, update_index: int, logical_step: int, updates_per_step: int
    ) -> bool:
        """Release actor and alpha only after the Stage-2 Q-only bridge."""
        next_q_update = self.q_update_count + 1
        return (
            self._stage2_actor_is_active(next_q_update)
            and next_q_update % int(self.cfg.sac_policy_frequency) == 0
        )

    def _stage2_bc_anchor_coefficient(self, q_updates=None) -> float:
        if not self._uses_bc_dagger_finetune_source():
            return 0.0
        if q_updates is None:
            q_updates = self.q_update_count
        actor_start = int(self.cfg.sac_actor_learning_starts_q_updates)
        elapsed = max(0, int(q_updates) - actor_start)
        progress = min(
            elapsed / float(self.cfg.sac_bc_anchor_decay_q_updates), 1.0
        )
        start = float(self.cfg.sac_bc_anchor_coef_start)
        end = float(self.cfg.sac_bc_anchor_coef_end)
        return start + progress * (end - start)

    def _stage2_bc_anchor_loss(self, actor_obs):
        zero = actor_obs.new_zeros(())
        if not self._uses_bc_dagger_finetune_source():
            return zero, zero
        td = self._actor_td_from_flat(actor_obs)
        current_action = self._bc_dagger_behavior_action(td)
        with torch.no_grad():
            anchor_action = self._bc_dagger_behavior_action(
                td, actor=self.bc_dagger_actor_anchor
            )
        loss = F.smooth_l1_loss(
            current_action,
            anchor_action,
            beta=float(self.cfg.sac_bc_anchor_huber_delta),
        )
        deviation = (current_action.detach() - anchor_action).abs().mean()
        return loss, deviation

    def _stage2_deterministic_action(self, actor_obs, dist=None):
        if self._uses_bc_dagger_finetune_source():
            return self._bc_dagger_behavior_action(
                self._actor_td_from_flat(actor_obs)
            )
        if dist is None:
            dist = self._actor_dist_from_flat(actor_obs)
        return dist.mean

    def _stage2_actor_confidence_gate_enabled(self) -> bool:
        """Use the frozen BC mean only as a detached confidence reference."""
        return (
            self._uses_bc_dagger_finetune_source()
            and bool(getattr(self.cfg, "sac_actor_confidence_gate", True))
        )

    def _stage2_actor_confidence_gate(
        self,
        batch,
        sampled_action,
        sampled_twin_q_values,
    ):
        """Return a non-sticky row mask and batch decision for one actor tick.

        Gain is pessimistic Q(actor)-Q(BC). Confidence uses the larger raw
        Q1/Q2 gap at the actor or BC action, so a shared action ranking cannot
        hide critics whose absolute values still disagree. The sampled action
        is the exact candidate later used by the actor loss; this method never
        previews and then resamples an action.
        """
        batch_size = int(sampled_action.shape[0])
        if tuple(sampled_twin_q_values.shape) != (2, batch_size):
            raise ValueError(
                "Stage-2 actor confidence gate requires two scalar-Q heads "
                f"with shape (2, {batch_size}), got "
                f"{tuple(sampled_twin_q_values.shape)}"
            )
        metric_zero = sampled_twin_q_values.detach().new_zeros(())
        if not self._stage2_actor_confidence_gate_enabled():
            return (
                True,
                torch.ones(
                    batch_size,
                    dtype=sampled_twin_q_values.dtype,
                    device=sampled_twin_q_values.device,
                ),
                {
                    "enabled": metric_zero,
                    "attempted": metric_zero,
                    "passed": metric_zero,
                    "skipped": metric_zero,
                    "acceptance_fraction": metric_zero,
                    "clipped_q_gain": metric_zero,
                    "twin_disagreement": metric_zero,
                    "sampled_policy_twin_disagreement": metric_zero,
                    "frozen_bc_twin_disagreement": metric_zero,
                    "confidence_margin": metric_zero,
                    "accepted_confidence_margin": metric_zero,
                    "sampled_policy_q": metric_zero,
                    "frozen_bc_q": metric_zero,
                    "row_count": metric_zero,
                },
            )

        with torch.no_grad():
            actor_td = self._actor_td_from_flat(batch["observations"])
            frozen_bc_action = self._bc_dagger_behavior_action(
                actor_td, actor=self.bc_dagger_actor_anchor
            )
            frozen_bc_twin_q_values = self.qnet.values(
                self._q_forward(
                    self.qnet,
                    batch["critic_observations"],
                    frozen_bc_action,
                    batch.get(TEACHER_REF_ACTION_FIELD),
                    batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                )
            )
            if tuple(frozen_bc_twin_q_values.shape) != (2, batch_size):
                raise ValueError(
                    "Stage-2 frozen-BC confidence reference requires two "
                    f"scalar-Q heads with shape (2, {batch_size}), got "
                    f"{tuple(frozen_bc_twin_q_values.shape)}"
                )

            sampled_pessimistic_q = sampled_twin_q_values.detach().min(
                dim=0
            ).values
            frozen_bc_pessimistic_q = frozen_bc_twin_q_values.min(
                dim=0
            ).values
            clipped_q_gain = (
                sampled_pessimistic_q - frozen_bc_pessimistic_q
            )
            sampled_policy_twin_disagreement = (
                sampled_twin_q_values.detach()[0]
                - sampled_twin_q_values.detach()[1]
            ).abs()
            frozen_bc_twin_disagreement = (
                frozen_bc_twin_q_values[0] - frozen_bc_twin_q_values[1]
            ).abs()
            twin_disagreement = torch.maximum(
                sampled_policy_twin_disagreement,
                frozen_bc_twin_disagreement,
            )
            confidence_margin = (
                clipped_q_gain
                - float(getattr(
                    self.cfg,
                    "sac_actor_gate_disagreement_multiplier",
                    1.0,
                ))
                * twin_disagreement
                - float(getattr(
                    self.cfg, "sac_actor_gate_absolute_margin", 0.0
                ))
            )
            row_mask = (confidence_margin > 0.0).to(
                sampled_twin_q_values.dtype
            )
            acceptance_fraction = row_mask.mean()
            accepted_count = row_mask.sum()
            batch_passed = bool(
                accepted_count.item() > 0.0
                and acceptance_fraction.item()
                >= float(getattr(
                    self.cfg,
                    "sac_actor_gate_min_accept_fraction",
                    0.10,
                ))
            )
            accepted_confidence_margin = (
                (confidence_margin * row_mask).sum()
                / accepted_count.clamp_min(1.0)
            )
            sampled_policy_q = sampled_pessimistic_q.mean()
            frozen_bc_q = frozen_bc_pessimistic_q.mean()
            diagnostics = {
                "enabled": metric_zero.new_ones(()),
                "attempted": metric_zero.new_ones(()),
                "passed": metric_zero.new_tensor(float(batch_passed)),
                "skipped": metric_zero.new_tensor(float(not batch_passed)),
                "acceptance_fraction": acceptance_fraction,
                "clipped_q_gain": clipped_q_gain.mean(),
                "twin_disagreement": twin_disagreement.mean(),
                "sampled_policy_twin_disagreement": (
                    sampled_policy_twin_disagreement.mean()
                ),
                "frozen_bc_twin_disagreement": (
                    frozen_bc_twin_disagreement.mean()
                ),
                "confidence_margin": confidence_margin.mean(),
                "accepted_confidence_margin": accepted_confidence_margin,
                "sampled_policy_q": sampled_policy_q,
                "frozen_bc_q": frozen_bc_q,
                "row_count": metric_zero.new_tensor(float(batch_size)),
            }
        return batch_passed, row_mask.detach(), diagnostics

    def _sac_update(self, batch, update_actor):
        batch = self._prepare_student_learning_batch(batch)
        offline_source_mask, online_source_mask, source_marker_valid = (
            self._stage2_replay_source_masks(batch)
        )
        q_update_index = self.q_update_count + 1
        actor_active_for_update = self._stage2_actor_is_active(
            q_update_index
        )
        if update_actor and not actor_active_for_update:
            raise RuntimeError(
                "Stage-2 actor update was requested during Q-only warm-up"
            )
        alpha_ramp_progress = self._stage2_alpha_ramp_progress(q_update_index)
        effective_alpha = self._stage2_effective_alpha(q_update_index)
        stochastic_q_target = self._stage2_q_target_uses_stochastic_policy(
            q_update_index
        )
        with torch.no_grad():
            next_dist = self._actor_dist_from_flat(batch["next_observations"])
            if stochastic_q_target:
                next_action, next_log_prob = next_dist.rsample_with_log_prob(
                    generator=self.sac_action_rng
                )
                next_log_prob = self._normalized_action_log_prob(
                    next_log_prob
                )
            else:
                # The explicit deterministic-rollout ablation evaluates its
                # exact mean and omits entropy until the SAC actor is released.
                next_action = self._stage2_deterministic_action(
                    batch["next_observations"], next_dist
                )
                next_log_prob = batch["rewards"].new_zeros(
                    batch["rewards"].shape
                )
            discount = self.cfg.gamma * batch["discounts"]
            # Match VAIC PPO reset semantics: episode-limit and command-finished
            # truncations bootstrap; true termination does not. Entropy belongs
            # to the discounted next-state target.
            bootstrap = _sac_bootstrap_mask(
                batch["dones"], batch["truncations"]
            )
            # Keep the stochastic behavior sample during the IQL-to-SAC bridge,
            # but preserve IQL's hard Bellman target until a real actor update
            # starts the effective-alpha ramp.
            entropy_tax = (
                discount
                * bootstrap
                * effective_alpha.detach()
                * next_log_prob
            )
            soft_reward = batch["rewards"] - entropy_tax
            target = self._q_projection(
                self.qnet_target,
                batch["next_critic_observations"], next_action, soft_reward,
                bootstrap, discount,
                batch.get(NEXT_TEACHER_REF_ACTION_FIELD),
                batch.get(NEXT_TEACHER_ACTUATOR_CONTEXT_FIELD),
            )
            target, _ = _select_c51_twin_target(
                target,
                self.qnet_target.support,
                bool(getattr(self.cfg, "sac_clipped_double_q", True)),
            )
        logits = self._q_forward(
            self.qnet,
            batch["critic_observations"],
            batch["actions"],
            batch.get(TEACHER_REF_ACTION_FIELD),
            batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
        )
        with torch.no_grad():
            if hasattr(self.qnet, "values"):
                data_twin_q_values = self.qnet.values(logits.detach())
            else:
                # Lightweight tests and compatible C51 wrappers may expose
                # only logits; the target critic carries the identical support.
                support = getattr(
                    self.qnet, "support", self.qnet_target.support
                )
                data_twin_q_values = (
                    F.softmax(logits.detach(), dim=-1) * support
                ).sum(dim=-1)
            data_q_per_row = _reduce_actor_q_values(
                data_twin_q_values,
                bool(getattr(self.cfg, "sac_clipped_double_q", True)),
            )
            data_metric_zero = data_q_per_row.new_zeros(())

            def data_q_source_metrics(mask, valid):
                count = int(mask.sum().item()) if valid else 0
                return (
                    data_q_per_row[mask].mean()
                    if count
                    else data_metric_zero,
                    data_metric_zero.new_tensor(float(count)),
                    data_metric_zero.new_tensor(float(count > 0)),
                )

            offline_data_q, offline_data_rows, offline_data_valid = (
                data_q_source_metrics(
                    offline_source_mask, source_marker_valid
                )
            )
            online_data_q, online_data_rows, online_data_valid = (
                data_q_source_metrics(
                    online_source_mask, source_marker_valid
                )
            )
            q_diagnostics = {
                "data_q": data_q_per_row.mean().detach(),
                "data_rows": data_metric_zero.new_tensor(
                    float(data_q_per_row.numel())
                ),
                "data_valid": data_metric_zero.new_ones(()),
                "source_marker_valid": data_metric_zero.new_tensor(
                    float(source_marker_valid)
                ),
                "offline_data_q": offline_data_q.detach(),
                "offline_data_rows": offline_data_rows,
                "offline_data_valid": offline_data_valid,
                "online_data_q": online_data_q.detach(),
                "online_data_rows": online_data_rows,
                "online_data_valid": online_data_valid,
            }
        per_q = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
        q_loss = per_q.sum()
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad = _measure_or_clip_grad_norm(
            self.qnet.parameters(), self.cfg.sac_max_grad_norm
        )
        self.opt_q.step()
        self.q_update_count += 1

        alpha_loss = torch.zeros((), device=self.device)
        actor_loss = torch.zeros((), device=self.device)
        entropy = torch.zeros((), device=self.device)
        anchor_loss = torch.zeros((), device=self.device)
        anchor_deviation = torch.zeros((), device=self.device)
        anchor_coefficient = 0.0
        actor_grad = torch.zeros((), device=self.device)
        metric_zero = batch["rewards"].new_zeros(())
        actor_update_applied = False
        actor_gate_diagnostics = {
            "enabled": metric_zero,
            "attempted": metric_zero,
            "passed": metric_zero,
            "skipped": metric_zero,
            "acceptance_fraction": metric_zero,
            "clipped_q_gain": metric_zero,
            "twin_disagreement": metric_zero,
            "sampled_policy_twin_disagreement": metric_zero,
            "frozen_bc_twin_disagreement": metric_zero,
            "confidence_margin": metric_zero,
            "accepted_confidence_margin": metric_zero,
            "sampled_policy_q": metric_zero,
            "frozen_bc_q": metric_zero,
            "row_count": metric_zero,
        }
        if update_actor:
            dist = self._actor_dist_from_flat(batch["observations"])
            action, log_prob = dist.rsample_with_log_prob(
                generator=self.sac_action_rng
            )
            log_prob = self._normalized_action_log_prob(log_prob)
            q_requires_grad = [
                parameter.requires_grad for parameter in self.qnet.parameters()
            ]
            for parameter in self.qnet.parameters():
                parameter.requires_grad_(False)
            try:
                twin_q_values = self.qnet.values(
                    self._q_forward(
                        self.qnet,
                        batch["critic_observations"],
                        action,
                        batch.get(TEACHER_REF_ACTION_FIELD),
                        batch.get(TEACHER_ACTUATOR_CONTEXT_FIELD),
                    )
                )
                q_values = _reduce_actor_q_values(
                    twin_q_values,
                    bool(getattr(self.cfg, "sac_clipped_double_q", True)),
                )
                (
                    actor_gate_passed,
                    trusted_q_rows,
                    actor_gate_diagnostics,
                ) = self._stage2_actor_confidence_gate(
                    batch, action, twin_q_values
                )
                if actor_gate_passed:
                    # The mask was computed from this exact sampled action.
                    # Its detached frozen-BC reference selects trustworthy Q
                    # gradients but contributes no BC regression gradient.
                    trusted_q_values = trusted_q_rows * q_values
                    sac_actor_loss = (
                        effective_alpha.detach() * log_prob
                        - trusted_q_values
                    ).mean()
                    anchor_coefficient = self._stage2_bc_anchor_coefficient()
                    if anchor_coefficient > 0.0:
                        anchor_loss, anchor_deviation = (
                            self._stage2_bc_anchor_loss(batch["observations"])
                        )
                    actor_loss = (
                        sac_actor_loss + anchor_coefficient * anchor_loss
                    )
                    self.sac_actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    trainable_actor_parameters = getattr(
                        self,
                        "_stage2_trainable_actor_parameters",
                        tuple(self.actor_adapt.parameters()),
                    )
                    actor_grad = _measure_or_clip_grad_norm(
                        trainable_actor_parameters,
                        float(self.cfg.sac_max_grad_norm),
                    )
                    self.sac_actor_optimizer.step()
                    actor_update_applied = True
            finally:
                for parameter, requires_grad in zip(
                    self.qnet.parameters(), q_requires_grad
                ):
                    parameter.requires_grad_(requires_grad)
            if actor_update_applied:
                self.sac_actor_update_count += 1
                # Set the alpha-ramp origin only after the actor optimizer step
                # has actually succeeded. Later gate failures never erase it.
                self._mark_stage2_actor_released(q_update_index)
                entropy = -log_prob.mean().detach()

        # Couple the entropy dual to the confidence-approved actor cadence.
        # A scheduled-but-rejected tick changes Q only, leaving both actor and
        # alpha parameters/counters untouched.
        if (
            actor_update_applied
            and alpha_ramp_progress > 0.0
            and bool(getattr(self.cfg, "sac_use_autotune", True))
        ):
            alpha_loss = -(
                effective_alpha
                * (next_log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.sac_alpha_update_count += 1
        actor_gate_diagnostics["actor_update_applied"] = (
            metric_zero.new_tensor(float(actor_update_applied))
        )

        with torch.no_grad():
            reward_mean = batch["rewards"].mean()
            reward_abs_mean = batch["rewards"].abs().mean()
            entropy_tax_abs_mean = entropy_tax.abs().mean()
            entropy_metric_zero = reward_mean.new_zeros(())

            def entropy_source_metrics(mask, valid):
                count = int(mask.sum().item()) if valid else 0
                if not count:
                    return {
                        "rows": entropy_metric_zero,
                        "valid": entropy_metric_zero,
                        "ratio_valid": entropy_metric_zero,
                        "reward_mean": entropy_metric_zero,
                        "reward_abs_mean": entropy_metric_zero,
                        "entropy_tax_abs_mean": entropy_metric_zero,
                        "entropy_tax_reward_abs_ratio": entropy_metric_zero,
                    }
                source_reward = batch["rewards"][mask]
                source_tax = entropy_tax[mask]
                source_reward_abs_mean = source_reward.abs().mean()
                source_tax_abs_mean = source_tax.abs().mean()
                ratio_valid = bool(
                    source_reward_abs_mean.item()
                    > torch.finfo(source_reward.dtype).eps
                )
                ratio = (
                    source_tax_abs_mean / source_reward_abs_mean
                    if ratio_valid
                    else entropy_metric_zero
                )
                return {
                    "rows": entropy_metric_zero.new_tensor(float(count)),
                    "valid": entropy_metric_zero.new_ones(()),
                    "ratio_valid": entropy_metric_zero.new_tensor(
                        float(ratio_valid)
                    ),
                    "reward_mean": source_reward.mean(),
                    "reward_abs_mean": source_reward_abs_mean,
                    "entropy_tax_abs_mean": source_tax_abs_mean,
                    "entropy_tax_reward_abs_ratio": ratio,
                }

            offline_entropy = entropy_source_metrics(
                offline_source_mask, source_marker_valid
            )
            online_entropy = entropy_source_metrics(
                online_source_mask, source_marker_valid
            )
            entropy_diagnostics = {
                "effective_alpha": effective_alpha.detach(),
                "alpha_ramp_progress": batch["rewards"].new_tensor(
                    alpha_ramp_progress
                ),
                "q_target_log_pi_mean": next_log_prob.mean().detach(),
                "entropy_tax_mean": entropy_tax.mean().detach(),
                "entropy_tax_abs_mean": entropy_tax_abs_mean.detach(),
                "reward_mean": reward_mean.detach(),
                "reward_abs_mean": reward_abs_mean.detach(),
                "entropy_tax_reward_abs_ratio": (
                    entropy_tax_abs_mean
                    / reward_abs_mean.clamp_min(
                        torch.finfo(batch["rewards"].dtype).eps
                    )
                ).detach(),
                "source_marker_valid": reward_mean.new_tensor(
                    float(source_marker_valid)
                ),
            }
            for source_name, source_metrics in (
                ("offline", offline_entropy),
                ("online", online_entropy),
            ):
                entropy_diagnostics.update({
                    f"{source_name}_{name}": value.detach()
                    for name, value in source_metrics.items()
                })

        with torch.no_grad():
            for source, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                target_param.lerp_(source, self.cfg.sac_tau)
        return (
            q_loss.detach(),
            per_q.detach(),
            q_grad.detach(),
            actor_loss.detach(),
            alpha_loss.detach(),
            entropy,
            anchor_loss.detach(),
            float(anchor_coefficient),
            anchor_deviation.detach(),
            actor_grad.detach(),
            entropy_diagnostics,
            actor_gate_diagnostics,
            q_diagnostics,
        )

    def train_op(self, tensordict):
        rollout_td = tensordict.exclude("stats")
        start_actor_update_count = self.sac_actor_update_count
        start_alpha_update_count = self.sac_alpha_update_count
        rollout_action_mean_abs_deviation = 0.0
        rollout_action_max_abs_deviation = 0.0
        rollout_behavior_support_rows = 0
        if self._uses_bc_dagger_finetune_source():
            required_diagnostics = (
                STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY,
                STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY,
            )
            missing = [
                key for key in required_diagnostics
                if key not in rollout_td.keys(True, True)
            ]
            if missing:
                raise RuntimeError(
                    "BC-DAgger Stage-2 rollout is missing behavior-support "
                    f"diagnostics: {missing}"
                )
            valid_behavior_rows = (
                rollout_td["step_count"].reshape(-1)
                > STUDENT_REPLAY_MIN_STEP_COUNT
            )
            rollout_behavior_support_rows = int(
                valid_behavior_rows.sum().item()
            )
            if rollout_behavior_support_rows:
                mean_deviation = rollout_td[
                    STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY
                ].detach().reshape(-1)[valid_behavior_rows]
                max_deviation = rollout_td[
                    STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY
                ].detach().reshape(-1)[valid_behavior_rows]
                rollout_action_mean_abs_deviation = (
                    mean_deviation.float().mean().item()
                )
                rollout_action_max_abs_deviation = (
                    max_deviation.float().amax().item()
                )
        start_environment_step = self.sac_environment_steps
        updates_per_step = self._sac_updates_per_env_step()
        replay_mix_counts = self._stage2_replay_mix_counts()
        online_batch_count = replay_mix_counts["online"]
        new_online_rows = 0
        sampled_total_draws = 0
        sampled_online_draws = 0
        sampled_offline_draws = 0
        metrics = []
        actor_metrics = []
        actor_gate_metrics = []
        actor_update_scheduled_count = 0
        for local_step, online in enumerate(
            self._student_transition_chunks(rollout_td)
        ):
            new_online_rows += int(online["rewards"].shape[0])
            if self._uses_bc_dagger_finetune_source():
                online_actions = online["actions"]
                action_clip = float(self.cfg.sac_bc_action_clip)
                if (
                    not torch.isfinite(online_actions).all()
                    or (online_actions.abs() > action_clip).any()
                ):
                    raise RuntimeError(
                        "BC-DAgger Stage-2 rollout produced an action outside "
                        "the actor/replay/Q support"
                    )
            self.online_replay.extend(online)
            logical_step = start_environment_step + local_step
            if (
                int(getattr(
                    self.online_replay,
                    "seen",
                    self.online_replay.size,
                )) < int(self.cfg.sac_learning_starts)
                or (online_batch_count and self.online_replay.size < 1)
            ):
                continue
            for update_index in range(updates_per_step):
                batch = self._mix_batch()
                sampled_total_draws += replay_mix_counts["total"]
                sampled_online_draws += replay_mix_counts["online"]
                sampled_offline_draws += replay_mix_counts["offline"]
                update_actor = self._sac_actor_update_is_due(
                    update_index, logical_step, updates_per_step
                )
                actor_update_scheduled_count += int(update_actor)
                update_metrics = self._sac_update(batch, update_actor)
                metrics.append(update_metrics)
                gate_diagnostics = (
                    update_metrics[11]
                    if len(update_metrics) > 11
                    else None
                )
                if (
                    gate_diagnostics is not None
                    and gate_diagnostics["attempted"].item() > 0.0
                ):
                    actor_gate_metrics.append(gate_diagnostics)
                actor_applied = (
                    bool(gate_diagnostics["actor_update_applied"].item())
                    if gate_diagnostics is not None
                    else bool(update_actor)
                )
                if actor_applied:
                    actor_metrics.append(update_metrics)
                self.sac_update_count += 1

        self.sac_environment_steps = (
            start_environment_step + int(self.cfg.train_every)
        )
        self.sac_rollout_count += 1

        # Actor replay contains priv_pred rather than reconstructable raw
        # recurrent perception inputs. Keep its coordinate system immutable.
        adapt_info = {"fastsac/perception_frozen": 1.0}
        self.num_updates += 1

        if metrics:
            stacked = list(zip(*metrics))
            q_loss = torch.stack(stacked[0]).mean().item()
            q1_loss = torch.stack([x[0] for x in stacked[1]]).mean().item()
            q2_loss = torch.stack([x[1] for x in stacked[1]]).mean().item()
            q_grad_norm = torch.stack(stacked[2]).mean().item()
            alpha_loss = torch.stack(stacked[4]).mean().item()
            entropy_diagnostics = [
                metric[10] for metric in metrics if len(metric) > 10
            ]
            if entropy_diagnostics:
                def mean_entropy_diagnostic(name):
                    return torch.stack([
                        diagnostic[name]
                        for diagnostic in entropy_diagnostics
                    ]).mean().item()

                effective_alpha = mean_entropy_diagnostic("effective_alpha")
                alpha_ramp_progress = mean_entropy_diagnostic(
                    "alpha_ramp_progress"
                )
                q_target_log_pi_mean = mean_entropy_diagnostic(
                    "q_target_log_pi_mean"
                )
                entropy_tax_mean = mean_entropy_diagnostic(
                    "entropy_tax_mean"
                )
                entropy_tax_abs_mean = mean_entropy_diagnostic(
                    "entropy_tax_abs_mean"
                )
                reward_mean = mean_entropy_diagnostic("reward_mean")
                reward_abs_mean = mean_entropy_diagnostic("reward_abs_mean")
                entropy_tax_reward_abs_ratio = mean_entropy_diagnostic(
                    "entropy_tax_reward_abs_ratio"
                )
                entropy_source_marker_valid = float(all(
                    diagnostic["source_marker_valid"].item() > 0.0
                    for diagnostic in entropy_diagnostics
                ))

                def aggregate_entropy_source(source_name):
                    rows = sum(
                        diagnostic[f"{source_name}_rows"].item()
                        for diagnostic in entropy_diagnostics
                    )
                    if rows <= 0.0:
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    def weighted(field):
                        return sum(
                            diagnostic[f"{source_name}_{field}"].item()
                            * diagnostic[f"{source_name}_rows"].item()
                            for diagnostic in entropy_diagnostics
                        ) / rows

                    source_reward_mean = weighted("reward_mean")
                    source_reward_abs_mean = weighted("reward_abs_mean")
                    source_tax_abs_mean = weighted("entropy_tax_abs_mean")
                    ratio_valid = float(
                        entropy_source_marker_valid > 0.0
                        and source_reward_abs_mean
                        > torch.finfo(torch.float32).eps
                    )
                    ratio = (
                        source_tax_abs_mean / source_reward_abs_mean
                        if ratio_valid
                        else 0.0
                    )
                    return (
                        rows,
                        float(entropy_source_marker_valid > 0.0),
                        ratio_valid,
                        source_reward_mean,
                        source_reward_abs_mean,
                        source_tax_abs_mean,
                        ratio,
                    )

                (
                    offline_entropy_rows,
                    offline_entropy_valid,
                    offline_entropy_ratio_valid,
                    offline_reward_mean,
                    offline_reward_abs_mean,
                    offline_entropy_tax_abs_mean,
                    offline_entropy_tax_reward_abs_ratio,
                ) = aggregate_entropy_source("offline")
                (
                    online_entropy_rows,
                    online_entropy_valid,
                    online_entropy_ratio_valid,
                    online_reward_mean,
                    online_reward_abs_mean,
                    online_entropy_tax_abs_mean,
                    online_entropy_tax_reward_abs_ratio,
                ) = aggregate_entropy_source("online")
            else:
                effective_alpha = self._stage2_effective_alpha().item()
                alpha_ramp_progress = self._stage2_alpha_ramp_progress()
                q_target_log_pi_mean = 0.0
                entropy_tax_mean = entropy_tax_abs_mean = 0.0
                reward_mean = reward_abs_mean = 0.0
                entropy_tax_reward_abs_ratio = 0.0
                entropy_source_marker_valid = 0.0
                offline_entropy_rows = online_entropy_rows = 0.0
                offline_entropy_valid = online_entropy_valid = 0.0
                offline_entropy_ratio_valid = online_entropy_ratio_valid = 0.0
                offline_reward_mean = online_reward_mean = 0.0
                offline_reward_abs_mean = online_reward_abs_mean = 0.0
                offline_entropy_tax_abs_mean = online_entropy_tax_abs_mean = 0.0
                offline_entropy_tax_reward_abs_ratio = 0.0
                online_entropy_tax_reward_abs_ratio = 0.0
            if actor_metrics:
                actor_loss = torch.stack([
                    metric[3] for metric in actor_metrics
                ]).mean().item()
                entropy = torch.stack([
                    metric[5] for metric in actor_metrics
                ]).mean().item()
                anchor_loss = torch.stack([
                    metric[6] for metric in actor_metrics
                ]).mean().item()
                anchor_coefficient = sum(
                    metric[7] for metric in actor_metrics
                ) / len(actor_metrics)
                anchor_deviation = torch.stack([
                    metric[8] for metric in actor_metrics
                ]).mean().item()
                actor_grad_norm = torch.stack([
                    metric[9] for metric in actor_metrics
                ]).mean().item()
            else:
                actor_loss = entropy = 0.0
                anchor_loss = anchor_coefficient = 0.0
                anchor_deviation = actor_grad_norm = 0.0
        else:
            q_loss = q1_loss = q2_loss = q_grad_norm = 0.0
            actor_loss = alpha_loss = entropy = 0.0
            anchor_loss = anchor_coefficient = 0.0
            anchor_deviation = actor_grad_norm = 0.0
            effective_alpha = self._stage2_effective_alpha().item()
            alpha_ramp_progress = self._stage2_alpha_ramp_progress()
            q_target_log_pi_mean = 0.0
            entropy_tax_mean = entropy_tax_abs_mean = 0.0
            reward_mean = reward_abs_mean = 0.0
            entropy_tax_reward_abs_ratio = 0.0
            entropy_source_marker_valid = 0.0
            offline_entropy_rows = online_entropy_rows = 0.0
            offline_entropy_valid = online_entropy_valid = 0.0
            offline_entropy_ratio_valid = online_entropy_ratio_valid = 0.0
            offline_reward_mean = online_reward_mean = 0.0
            offline_reward_abs_mean = online_reward_abs_mean = 0.0
            offline_entropy_tax_abs_mean = online_entropy_tax_abs_mean = 0.0
            offline_entropy_tax_reward_abs_ratio = 0.0
            online_entropy_tax_reward_abs_ratio = 0.0

        q_diagnostics = [
            metric[12] for metric in metrics if len(metric) > 12
        ]
        if q_diagnostics:
            q_data_rows = sum(
                diagnostic["data_rows"].item()
                for diagnostic in q_diagnostics
            )
            q_data = sum(
                diagnostic["data_q"].item()
                * diagnostic["data_rows"].item()
                for diagnostic in q_diagnostics
            ) / q_data_rows
            q_data_valid = float(q_data_rows > 0.0)
            q_source_marker_valid = float(all(
                diagnostic["source_marker_valid"].item() > 0.0
                for diagnostic in q_diagnostics
            ))

            def aggregate_source_data_q(source_name):
                rows = sum(
                    diagnostic[f"{source_name}_data_rows"].item()
                    for diagnostic in q_diagnostics
                )
                if rows <= 0.0:
                    return 0.0, 0.0, 0.0
                value = sum(
                    diagnostic[f"{source_name}_data_q"].item()
                    * diagnostic[f"{source_name}_data_rows"].item()
                    for diagnostic in q_diagnostics
                ) / rows
                return value, rows, float(q_source_marker_valid > 0.0)

            (
                q_data_offline,
                q_data_offline_rows,
                q_data_offline_valid,
            ) = aggregate_source_data_q("offline")
            (
                q_data_online,
                q_data_online_rows,
                q_data_online_valid,
            ) = aggregate_source_data_q("online")
            q_data_update_count = len(q_diagnostics)
        else:
            q_data = q_data_rows = q_data_valid = 0.0
            q_source_marker_valid = 0.0
            q_data_offline = q_data_online = 0.0
            q_data_offline_rows = q_data_online_rows = 0.0
            q_data_offline_valid = q_data_online_valid = 0.0
            q_data_update_count = 0

        if actor_gate_metrics:
            def mean_actor_gate_diagnostic(name):
                return torch.stack([
                    diagnostic[name]
                    for diagnostic in actor_gate_metrics
                ]).mean().item()

            actor_gate_attempts = len(actor_gate_metrics)
            actor_gate_passes = int(round(sum(
                diagnostic["passed"].item()
                for diagnostic in actor_gate_metrics
            )))
            actor_gate_skips = int(round(sum(
                diagnostic["skipped"].item()
                for diagnostic in actor_gate_metrics
            )))
            actor_gate_acceptance_fraction = mean_actor_gate_diagnostic(
                "acceptance_fraction"
            )
            actor_gate_clipped_q_gain = mean_actor_gate_diagnostic(
                "clipped_q_gain"
            )
            actor_gate_twin_disagreement = mean_actor_gate_diagnostic(
                "twin_disagreement"
            )
            actor_gate_sampled_policy_twin_disagreement = (
                mean_actor_gate_diagnostic(
                    "sampled_policy_twin_disagreement"
                )
            )
            actor_gate_frozen_bc_twin_disagreement = (
                mean_actor_gate_diagnostic(
                    "frozen_bc_twin_disagreement"
                )
            )
            actor_gate_confidence_margin = mean_actor_gate_diagnostic(
                "confidence_margin"
            )
            actor_gate_accepted_confidence_margin = (
                mean_actor_gate_diagnostic("accepted_confidence_margin")
            )
            actor_gate_sampled_policy_q = mean_actor_gate_diagnostic(
                "sampled_policy_q"
            )
            actor_gate_frozen_bc_q = mean_actor_gate_diagnostic(
                "frozen_bc_q"
            )
            actor_gate_q_rows = int(round(sum(
                diagnostic["row_count"].item()
                for diagnostic in actor_gate_metrics
            )))
        else:
            actor_gate_attempts = actor_gate_passes = actor_gate_skips = 0
            actor_gate_acceptance_fraction = 0.0
            actor_gate_clipped_q_gain = 0.0
            actor_gate_twin_disagreement = 0.0
            actor_gate_sampled_policy_twin_disagreement = 0.0
            actor_gate_frozen_bc_twin_disagreement = 0.0
            actor_gate_confidence_margin = 0.0
            actor_gate_accepted_confidence_margin = 0.0
            actor_gate_sampled_policy_q = actor_gate_frozen_bc_q = 0.0
            actor_gate_q_rows = 0

        actor_updates_applied = (
            self.sac_actor_update_count - start_actor_update_count
        )
        alpha_updates_applied = (
            self.sac_alpha_update_count - start_alpha_update_count
        )
        actor_ever_released = (
            getattr(self, "_stage2_actor_release_q_update", None) is not None
        )
        sampled_draws_per_new_row_valid = new_online_rows > 0
        if sampled_draws_per_new_row_valid:
            sampled_total_draws_per_new_row = (
                sampled_total_draws / float(new_online_rows)
            )
            sampled_online_draws_per_new_row = (
                sampled_online_draws / float(new_online_rows)
            )
            sampled_offline_draws_per_new_row = (
                sampled_offline_draws / float(new_online_rows)
            )
        else:
            sampled_total_draws_per_new_row = 0.0
            sampled_online_draws_per_new_row = 0.0
            sampled_offline_draws_per_new_row = 0.0

        info = {
            "fastsac/q_active": float(bool(metrics)),
            "fastsac/q_loss": q_loss,
            "fastsac/q1_loss": q1_loss,
            "fastsac/q2_loss": q2_loss,
            "fastsac/q_grad_norm": q_grad_norm,
            "fastsac/q_data": q_data,
            "fastsac/q_data_valid": q_data_valid,
            "fastsac/q_data_rows": q_data_rows,
            "fastsac/q_data_update_count": q_data_update_count,
            "fastsac/q_data_source_marker_valid": q_source_marker_valid,
            "fastsac/q_data_online": q_data_online,
            "fastsac/q_data_online_valid": q_data_online_valid,
            "fastsac/q_data_online_rows": q_data_online_rows,
            "fastsac/q_data_offline": q_data_offline,
            "fastsac/q_data_offline_valid": q_data_offline_valid,
            "fastsac/q_data_offline_rows": q_data_offline_rows,
            "fastsac/new_online_rows": new_online_rows,
            "fastsac/sampled_total_draws": sampled_total_draws,
            "fastsac/sampled_online_draws": sampled_online_draws,
            "fastsac/sampled_offline_draws": sampled_offline_draws,
            "fastsac/sampled_draws_per_new_row_valid": float(
                sampled_draws_per_new_row_valid
            ),
            "fastsac/sampled_total_draws_per_new_row": (
                sampled_total_draws_per_new_row
            ),
            "fastsac/sampled_online_draws_per_new_row": (
                sampled_online_draws_per_new_row
            ),
            "fastsac/sampled_offline_draws_per_new_row": (
                sampled_offline_draws_per_new_row
            ),
            "fastsac/actor_loss": actor_loss,
            "fastsac/actor_grad_norm": actor_grad_norm,
            "fastsac/actor_active": float(actor_ever_released),
            "fastsac/q_only_warmup": float(not actor_ever_released),
            "fastsac/actor_schedule_eligible": float(
                self._stage2_actor_is_active()
            ),
            "fastsac/actor_update_scheduled_count": (
                actor_update_scheduled_count
            ),
            "fastsac/actor_update_applied_count": actor_updates_applied,
            "fastsac/actor_confidence_gate_enabled": float(
                self._stage2_actor_confidence_gate_enabled()
            ),
            "fastsac/actor_confidence_gate_attempts": actor_gate_attempts,
            "fastsac/actor_confidence_gate_passes": actor_gate_passes,
            "fastsac/actor_confidence_gate_skips": actor_gate_skips,
            "fastsac/actor_confidence_gate_acceptance_fraction": (
                actor_gate_acceptance_fraction
            ),
            "fastsac/actor_confidence_gate_clipped_q_gain": (
                actor_gate_clipped_q_gain
            ),
            "fastsac/actor_confidence_gate_twin_disagreement": (
                actor_gate_twin_disagreement
            ),
            "fastsac/actor_confidence_gate_sampled_policy_twin_disagreement": (
                actor_gate_sampled_policy_twin_disagreement
            ),
            "fastsac/actor_confidence_gate_frozen_bc_twin_disagreement": (
                actor_gate_frozen_bc_twin_disagreement
            ),
            "fastsac/actor_confidence_gate_confidence_margin": (
                actor_gate_confidence_margin
            ),
            "fastsac/actor_confidence_gate_accepted_confidence_margin": (
                actor_gate_accepted_confidence_margin
            ),
            "fastsac/actor_confidence_gate_sampled_policy_q": (
                actor_gate_sampled_policy_q
            ),
            "fastsac/actor_confidence_gate_frozen_bc_q": (
                actor_gate_frozen_bc_q
            ),
            "fastsac/q_actor": actor_gate_sampled_policy_q,
            "fastsac/q_actor_valid": float(actor_gate_attempts > 0),
            "fastsac/q_actor_rows": actor_gate_q_rows,
            "fastsac/q_frozen_bc": actor_gate_frozen_bc_q,
            "fastsac/q_frozen_bc_valid": float(actor_gate_attempts > 0),
            "fastsac/q_frozen_bc_rows": actor_gate_q_rows,
            "fastsac/q_target_stochastic_policy": float(
                self._stage2_q_target_uses_stochastic_policy(
                    self.q_update_count + 1
                )
            ),
            "fastsac/bc_anchor_loss": anchor_loss,
            "fastsac/bc_anchor_coefficient": anchor_coefficient,
            "fastsac/bc_anchor_mean_action_deviation": anchor_deviation,
            "fastsac/alpha_loss": alpha_loss,
            "fastsac/entropy": entropy,
            "fastsac/entropy_reference_action_entropy": entropy,
            # Retained for dashboard compatibility; the explicit coordinate
            # flags below identify that BC Stage 2 now uses raw-action units.
            "fastsac/normalized_action_entropy": entropy,
            "fastsac/physical_action_entropy": (
                entropy + float(getattr(
                    self,
                    "_fastsac_entropy_reference_log_scale_sum",
                    self._fastsac_action_log_scale_sum,
                ))
                if actor_metrics else 0.0
            ),
            "fastsac/alpha": self.log_alpha.exp().item(),
            "fastsac/effective_alpha": effective_alpha,
            "fastsac/alpha_ramp_progress": alpha_ramp_progress,
            "fastsac/actor_release_q_update": int(
                getattr(self, "_stage2_actor_release_q_update", None) or -1
            ),
            "fastsac/q_target_log_pi_mean": q_target_log_pi_mean,
            "fastsac/entropy_tax_mean": entropy_tax_mean,
            "fastsac/entropy_tax_abs_mean": entropy_tax_abs_mean,
            "fastsac/reward_mean": reward_mean,
            "fastsac/reward_abs_mean": reward_abs_mean,
            "fastsac/entropy_tax_reward_abs_ratio": (
                entropy_tax_reward_abs_ratio
            ),
            "fastsac/entropy_source_marker_valid": (
                entropy_source_marker_valid
            ),
            "fastsac/offline_entropy_rows": offline_entropy_rows,
            "fastsac/offline_entropy_valid": offline_entropy_valid,
            "fastsac/offline_entropy_ratio_valid": (
                offline_entropy_ratio_valid
            ),
            "fastsac/offline_reward_mean": offline_reward_mean,
            "fastsac/offline_reward_abs_mean": offline_reward_abs_mean,
            "fastsac/offline_entropy_tax_abs_mean": (
                offline_entropy_tax_abs_mean
            ),
            "fastsac/offline_entropy_tax_reward_abs_ratio": (
                offline_entropy_tax_reward_abs_ratio
            ),
            "fastsac/online_entropy_rows": online_entropy_rows,
            "fastsac/online_entropy_valid": online_entropy_valid,
            "fastsac/online_entropy_ratio_valid": (
                online_entropy_ratio_valid
            ),
            "fastsac/online_reward_mean": online_reward_mean,
            "fastsac/online_reward_abs_mean": online_reward_abs_mean,
            "fastsac/online_entropy_tax_abs_mean": (
                online_entropy_tax_abs_mean
            ),
            "fastsac/online_entropy_tax_reward_abs_ratio": (
                online_entropy_tax_reward_abs_ratio
            ),
            "fastsac/alpha_active": float(
                bool(getattr(self.cfg, "sac_use_autotune", True))
                and alpha_updates_applied > 0
            ),
            "fastsac/alpha_autotune": float(bool(
                getattr(self.cfg, "sac_use_autotune", True)
            )),
            "fastsac/target_entropy": self.target_entropy,
            "fastsac/target_entropy_normalized_coordinates": float(
                not self._uses_bc_dagger_finetune_source()
            ),
            "fastsac/target_entropy_reference_coordinates": 1.0,
            "fastsac/target_entropy_raw_action_unit_coordinates": float(
                self._uses_bc_dagger_finetune_source()
            ),
            "fastsac/action_log_scale_sum": self._fastsac_action_log_scale_sum,
            "fastsac/entropy_reference_scale": float(getattr(
                self.cfg, "sac_entropy_reference_scale", 1.0
            )),
            "fastsac/entropy_reference_log_scale_sum": float(getattr(
                self,
                "_fastsac_entropy_reference_log_scale_sum",
                self._fastsac_action_log_scale_sum,
            )),
            "fastsac/q_update_count": self.q_update_count,
            "fastsac/actor_update_count": self.sac_actor_update_count,
            "fastsac/alpha_update_count": self.sac_alpha_update_count,
            "fastsac/online_replay_size": self.online_replay.size,
            "fastsac/online_replay_seen": int(getattr(
                self.online_replay, "seen", self.online_replay.size
            )),
            "fastsac/replay_warmup_ready": float(
                int(getattr(
                    self.online_replay, "seen", self.online_replay.size
                )) >= int(self.cfg.sac_learning_starts)
            ),
            "fastsac/offline_ratio": self.cfg.teacher_buffer_ratio,
            "fastsac/truncation_finals": self._last_truncation_finals_used,
            "fastsac/environment_steps": self.sac_environment_steps,
            "fastsac/effective_updates_per_env_step": (
                len(metrics) / float(self.cfg.train_every)
            ),
            "fastsac/updates_per_env_step_config": updates_per_step,
            "fastsac/q_action_normalized": float(
                bool(getattr(self.cfg, "sac_q_normalize_actions", False))
            ),
            "fastsac/q_action_input_gain": float(
                getattr(self.cfg, "sac_q_action_input_gain", 1.0)
            ),
            "fastsac/clipped_double_q": float(bool(
                getattr(self.cfg, "sac_clipped_double_q", True)
            )),
        }
        if self._uses_bc_dagger_finetune_source():
            clipped_log_std = self.bc_dagger_sac_adapter.log_std.detach().clamp(
                float(self.cfg.sac_bc_log_std_min),
                float(self.cfg.sac_bc_log_std_max),
            )
            info.update({
                "fastsac/bc_sac_log_std_mean": clipped_log_std.mean().item(),
                "fastsac/bc_sac_center_action_std_mean": (
                    clipped_log_std.exp() * float(self.cfg.sac_bc_action_clip)
                ).mean().item(),
                "fastsac/rollout_deterministic_bc_mean": float(
                    bool(self.cfg.sac_deterministic_rollout)
                ),
                "fastsac/rollout_stochastic_exact_learning_distribution": float(
                    not bool(self.cfg.sac_deterministic_rollout)
                ),
                "fastsac/rollout_executed_action_mean_abs_deviation": (
                    rollout_action_mean_abs_deviation
                ),
                "fastsac/rollout_executed_action_max_abs_deviation": (
                    rollout_action_max_abs_deviation
                ),
                "fastsac/rollout_behavior_support_rows": (
                    rollout_behavior_support_rows
                ),
            })
        info.update(adapt_info)
        return info
