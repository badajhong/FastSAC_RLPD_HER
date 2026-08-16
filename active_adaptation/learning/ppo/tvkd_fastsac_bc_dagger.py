"""TVKD value-shaped FastSAC with Teacher-value bottleneck replay.

This entrypoint is intentionally layered on top of the repository's current
``DistributionalFastSACTeacherBC`` implementation.  It therefore preserves
the existing stochastic Student rollout, frozen successful-Teacher prefill,
independently configured Student/Teacher Q, Actor, and perception mixtures,
raw recurrent replay, timeout-final-observation handling, twin C51 critics,
and target-update cadence.

The two additions are deliberately narrow:

* the frozen PPO Teacher critic is reused as a potential function in the
  FastSAC C51 target; and
* failed mixed-control episodes use Student-executed Teacher-TD residuals to
  register bottleneck-aligned phases in the existing successful-Teacher replay
  curriculum.

The PPO critic consumes the same observation *fields* as the SAC critic, but
its internal ``CatTensors`` module sorts keys.  Consequently this module never
feeds the flattened SAC tensor directly into the PPO MLP.  It reconstructs a
keyed ``TensorDict`` first, preserving the exact PPO observation contract.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import CMD_KEY, DONE_KEY, OBS_KEY, OBS_PRIV_KEY, REWARD_KEY, TERM_KEY
from .fastsac_bc_dagger import (
    ACTOR_BACKEND,
    CHECKPOINT_VERSION as BASE_FASTSAC_CHECKPOINT_VERSION,
    TRAINING_ALGORITHM as BASE_FASTSAC_TRAINING_ALGORITHM,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
)
from .fastsac_vel import _vaic_truncation_mask
from .ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
)
from .ppo_vel import DEPTH_KEY, OBJECT_GEO_KEY, OBJECT_KEY, VEL_CMD_KEY
from .ppo_vel import PPOVEL
from .td3_bc_dagger import (
    FAILURE_PHASE_STUDENT_SOURCE_KEY,
    REFERENCE_PHASE_KEY,
    REPLAY_COMMAND_FINISHED_KEY,
    REPLAY_MOTION_ID_KEY,
    REPLAY_TERMINATED_KEY,
    REPLAY_TIME_LIMIT_KEY,
    STUDENT_REPLAY_EPISODE_ID_KEY,
    STUDENT_REPLAY_EPISODE_STEP_KEY,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    allocate_source_counts,
    _project_c51_probabilities,
)

TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v4"
V3_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v3"
PREVIOUS_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v2"
LEGACY_TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v1"
CHECKPOINT_VERSION = 4
V3_CHECKPOINT_VERSION = 3
PREVIOUS_CHECKPOINT_VERSION = 2
LEGACY_CHECKPOINT_VERSION = 1
EXPECTED_ALGO_NAME = "tvkd_fastsac_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger."
    "TVKDDistributionalFastSACTeacherBC"
)
CRITIC_LEARNING_SEMANTICS = (
    "frozen_raw_scale_ppo_value_potential_shaped_soft_c51_target_v1"
)
TEACHER_VALUE_RETURN_SEMANTICS = (
    "source_ppo_gae_gamma_replay_discount_terminated_mask_v1"
)
TEACHER_VALUE_BOUNDARY_SEMANTICS = (
    "source_ppo_physical_zero_nonphysical_done_self_bootstrap_v1"
)
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_sac_plus_fixed_joint_valid_teacher_label_bc_v2"
)
BOTTLENECK_LOCATION_SEMANTICS = (
    "frozen_teacher_raw_td_earliest_sustained_negative_student_v1"
)
VERIFIED_HISTOGRAM_SEMANTICS = "verified_teacher_value_threshold_motion_phase_v1"
FRESH_RING_RESUME_SEMANTICS = "clear_online_rings_and_partial_row_credit_v1"
UNBOUND_CONTRACT_FINGERPRINT = "unbound_direct_construction"

LEGACY_ADAPTIVE_BC_CONFIG_FIELDS = (
    "use_adaptive_student_bc",
    "student_bc_lambda_min",
    "student_bc_lambda_max",
    "student_bc_margin",
    "student_bc_temperature",
    "student_bc_scale_ema_decay",
    "student_bc_risk_ema_decay",
    "student_bc_scheduler_warmup_updates",
    "student_bc_scheduler_min_samples",
    "student_bc_scheduler_eps",
)

SOURCE_STUDENT = 0
SOURCE_UNIFORM_TEACHER = 1
SOURCE_FAILURE_TEACHER = 2
EXPECTED_ACTOR_IN_KEYS = (
    CMD_KEY,
    OBS_KEY,
    OBJECT_KEY,
    OBS_PRIV_KEY,
    OBJECT_GEO_KEY,
    VEL_CMD_KEY,
    DEPTH_KEY,
)


def _finite_scalar(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return value


def _required_contract_fingerprint(name: str, value) -> str:
    """Reject placeholder metadata before it can enter an exact-resume seam."""
    if (
        not isinstance(value, str)
        or value == UNBOUND_CONTRACT_FINGERPRINT
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a bound lowercase SHA-256 fingerprint")
    return value


_REPLAY_MIX_SOURCES = (
    "uniform_student",
    "failure_student",
    "uniform_teacher",
    "failure_teacher",
)


def _legacy_global_replay_mix(
    teacher_fraction, teacher_focus, student_focus
) -> dict[str, float]:
    values = {
        "teacher fraction": teacher_fraction,
        "Teacher focus fraction": teacher_focus,
        "Student focus fraction": student_focus,
    }
    checked = {}
    for name, value in values.items():
        value = _finite_scalar(name, value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        checked[name] = value
    teacher = checked["teacher fraction"]
    student = 1.0 - teacher
    return {
        "uniform_student": student * (1.0 - checked["Student focus fraction"]),
        "failure_student": student * checked["Student focus fraction"],
        "uniform_teacher": teacher * (1.0 - checked["Teacher focus fraction"]),
        "failure_teacher": teacher * checked["Teacher focus fraction"],
    }


def _checkpoint_replay_mix(cfg) -> dict[str, dict[str, float]]:
    return {
        purpose: {
            source: float(getattr(cfg, f"{purpose}_{source}_fraction"))
            for source in _REPLAY_MIX_SOURCES
        }
        for purpose in ("q", "actor", "perception")
    }


def _same_verified_histogram_state(left: Mapping, right: Mapping) -> bool:
    """Compare the v4 histogram and its compatibility alias tensor-exactly."""
    scalar_fields = (
        "semantics",
        "episode_count",
        "anchor_count",
        "uniform_fallback_rows",
        "focused_rows",
    )
    if any(left.get(name) != right.get(name) for name in scalar_fields):
        return False
    left_histogram = left.get("histogram")
    right_histogram = right.get("histogram")
    if not (
        torch.is_tensor(left_histogram)
        and torch.is_tensor(right_histogram)
        and torch.equal(left_histogram, right_histogram)
    ):
        return False
    left_motion = left.get("motion_histograms")
    right_motion = right.get("motion_histograms")
    if not isinstance(left_motion, Mapping) or not isinstance(right_motion, Mapping):
        return False
    if set(left_motion) != set(right_motion):
        return False
    return all(
        torch.is_tensor(left_motion[motion_id])
        and torch.is_tensor(right_motion[motion_id])
        and torch.equal(left_motion[motion_id], right_motion[motion_id])
        for motion_id in left_motion
    )


def _install_v3_replay_migration(
    cfg,
    backend: Mapping,
    *,
    student_focus_default=None,
) -> None:
    """Install an old global mix and explicit legacy perception semantics."""
    teacher_focus = backend.get("failure_phase_teacher_fraction")
    student_focus = backend.get("failure_phase_student_fraction", student_focus_default)
    q_mix = _legacy_global_replay_mix(
        backend.get("q_teacher_replay_ratio"), teacher_focus, student_focus
    )
    actor_mix = _legacy_global_replay_mix(
        backend.get("teacher_actor_replay_fraction"), teacher_focus, student_focus
    )
    perception_teacher = _finite_scalar(
        "Teacher perception fraction",
        backend.get("teacher_perception_replay_fraction"),
    )
    if not 0.0 <= perception_teacher <= 1.0:
        raise ValueError("Teacher perception fraction must lie in [0, 1]")
    teacher_focus_value = _finite_scalar("Teacher focus fraction", teacher_focus)
    perception_mix = {
        "uniform_student": 1.0 - perception_teacher,
        "failure_student": 0.0,
        "uniform_teacher": perception_teacher * (1.0 - teacher_focus_value),
        "failure_teacher": perception_teacher * teacher_focus_value,
    }
    for purpose, mix in (
        ("q", q_mix),
        ("actor", actor_mix),
        ("perception", perception_mix),
    ):
        allocate_source_counts(1, mix)
        for source, value in mix.items():
            setattr(cfg, f"{purpose}_{source}_fraction", float(value))
    cadence = backend.get("sac_alpha_update_cadence", "critic")
    if cadence not in {"actor", "critic"}:
        raise ValueError("v3 checkpoint alpha cadence is invalid")
    cfg.sac_alpha_update_cadence = cadence
    cfg.q_teacher_replay_ratio = float(
        _finite_scalar(
            "checkpoint Q Teacher fraction",
            backend.get("q_teacher_replay_ratio"),
        )
    )
    cfg.teacher_actor_replay_fraction = float(
        _finite_scalar(
            "checkpoint Actor Teacher fraction",
            backend.get("teacher_actor_replay_fraction"),
        )
    )
    cfg.teacher_perception_replay_fraction = perception_teacher
    cfg.failure_phase_teacher_fraction = teacher_focus_value
    if hasattr(cfg, "failure_phase_student_fraction"):
        cfg.failure_phase_student_fraction = float(student_focus)
    cfg.perception_replay_mode = "legacy_online_student"
    cfg.bottleneck_fallback_mode = "none"
    cfg.bottleneck_include_unsuccessful_timeouts = True
    cfg.max_teacher_phase_match_distance = None


def _validate_tvkd_algorithm_config(cfg) -> None:
    """Validate controls shared by direct construction and the Hydra CLI."""
    if getattr(cfg, "sac_alpha_update_cadence", None) not in {"actor", "critic"}:
        raise ValueError("sac_alpha_update_cadence must be 'actor' or 'critic'")
    for name in (
        "use_tvkd_value_shaping",
        "use_teacher_value_bottleneck_replay",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be boolean")
    include_timeouts = getattr(cfg, "bottleneck_include_unsuccessful_timeouts", True)
    if not isinstance(include_timeouts, bool):
        raise ValueError("bottleneck_include_unsuccessful_timeouts must be boolean")
    fallback_mode = str(getattr(cfg, "bottleneck_fallback_mode", "none"))
    if fallback_mode not in {"none", "value_argmin"}:
        raise ValueError("bottleneck_fallback_mode must be 'none' or 'value_argmin'")

    for purpose in ("q", "actor", "perception"):
        fractions = {
            source: getattr(cfg, f"{purpose}_{source}_fraction")
            for source in (
                "uniform_student",
                "failure_student",
                "uniform_teacher",
                "failure_teacher",
            )
        }
        # The shared allocator is also the canonical strict fraction validator.
        allocate_source_counts(1, fractions)

    tvkd_lambda = _finite_scalar("tvkd_lambda", getattr(cfg, "tvkd_lambda"))
    if tvkd_lambda < 0.0:
        raise ValueError("tvkd_lambda must be non-negative")
    potential_clip = getattr(cfg, "tvkd_potential_clip")
    if potential_clip is not None:
        _finite_scalar("tvkd_potential_clip", potential_clip, positive=True)

    _finite_scalar(
        "bottleneck_threshold",
        getattr(cfg, "bottleneck_threshold"),
        positive=True,
    )
    for name in ("bottleneck_smoothing_window", "bottleneck_min_consecutive"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    exclusion = getattr(cfg, "bottleneck_terminal_exclusion_steps")
    if isinstance(exclusion, bool) or not isinstance(exclusion, int) or exclusion < 0:
        raise ValueError(
            "bottleneck_terminal_exclusion_steps must be a non-negative integer"
        )
    decay = _finite_scalar(
        "bottleneck_residual_scale_ema_decay",
        getattr(cfg, "bottleneck_residual_scale_ema_decay"),
    )
    if not 0.0 <= decay < 1.0:
        raise ValueError("bottleneck_residual_scale_ema_decay must lie in [0, 1)")
    _finite_scalar("bottleneck_eps", getattr(cfg, "bottleneck_eps"), positive=True)
    max_distance = getattr(cfg, "max_teacher_phase_match_distance", None)
    if max_distance is not None:
        max_distance = _finite_scalar("max_teacher_phase_match_distance", max_distance)
        if max_distance < 0.0:
            raise ValueError("max_teacher_phase_match_distance must be non-negative")
    if getattr(cfg, "teacher_value_return_semantics", None) != (
        TEACHER_VALUE_RETURN_SEMANTICS
    ):
        raise ValueError("unsupported Teacher value return semantics")
    if getattr(cfg, "teacher_value_boundary_semantics", None) != (
        TEACHER_VALUE_BOUNDARY_SEMANTICS
    ):
        raise ValueError("unsupported Teacher value boundary semantics")


@dataclass
class TVKDDistributionalFastSACTeacherBCConfig(DistributionalFastSACTeacherBCConfig):
    """Hydra surface for TVKD shaping and value-bottleneck replay."""

    _target_: str = EXPECTED_ALGO_TARGET
    name: str = EXPECTED_ALGO_NAME
    # Fresh v4 matches the parent FastSAC optimizer timescale. A v3 migration
    # explicitly replaces this with the checkpoint's saved cadence.
    sac_alpha_update_cadence: str = "actor"

    q_uniform_student_fraction: float = 0.35
    q_failure_student_fraction: float = 0.15
    q_uniform_teacher_fraction: float = 0.35
    q_failure_teacher_fraction: float = 0.15
    actor_uniform_student_fraction: float = 0.35
    actor_failure_student_fraction: float = 0.15
    actor_uniform_teacher_fraction: float = 0.35
    actor_failure_teacher_fraction: float = 0.15
    perception_uniform_student_fraction: float = 0.35
    perception_failure_student_fraction: float = 0.15
    perception_uniform_teacher_fraction: float = 0.35
    perception_failure_teacher_fraction: float = 0.15
    perception_replay_mode: str = "four_way"

    use_tvkd_value_shaping: bool = True
    tvkd_lambda: float = 0.25
    tvkd_potential_clip: float | None = None

    use_teacher_value_bottleneck_replay: bool = True
    bottleneck_threshold: float = 1.0
    bottleneck_smoothing_window: int = 5
    bottleneck_min_consecutive: int = 3
    bottleneck_terminal_exclusion_steps: int = 5
    bottleneck_residual_scale_ema_decay: float = 0.99
    bottleneck_eps: float = 1e-6
    bottleneck_fallback_mode: str = "none"
    bottleneck_include_unsuccessful_timeouts: bool = True
    max_teacher_phase_match_distance: float | None = None
    teacher_value_return_semantics: str = TEACHER_VALUE_RETURN_SEMANTICS
    teacher_value_boundary_semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS
    teacher_value_reward_group_fingerprint: str = UNBOUND_CONTRACT_FINGERPRINT
    replay_task_fingerprint: str = UNBOUND_CONTRACT_FINGERPRINT
    # Deprecated v3 migration input only. Canonical v4 samplers never consult it.
    failure_phase_student_fraction: float = 0.3


ConfigStore.instance().store(
    "tvkd_fastsac_bc_dagger_finetune",
    node=TVKDDistributionalFastSACTeacherBCConfig(in_keys=EXPECTED_ACTOR_IN_KEYS),
    group="algo",
)


class FrozenTeacherValueWrapper:
    """Read the checkpoint PPO value in raw summed-reward scale.

    ``teacher_critic_obs`` is the already-VecNorm-normalized flattened SAC
    critic observation.  Its fields are split and restored by name so the PPO
    critic's own key ordering remains authoritative.

    This is deliberately a plain Python object, rather than an ``nn.Module``:
    registering a second alias to the Teacher actor/value modules would alter
    the established PPO/FastSAC checkpoint topology.
    """

    def __init__(
        self,
        teacher_actor: nn.Module,
        teacher_value: nn.Module,
        value_normalizer: nn.Module,
        critic_keys: tuple[str, ...],
        critic_widths: tuple[int, ...],
    ) -> None:
        if len(critic_keys) != len(critic_widths) or not critic_keys:
            raise ValueError("Teacher value critic keys/widths must align")
        if any(int(width) < 1 for width in critic_widths):
            raise ValueError("Teacher value critic widths must be positive")
        self.teacher_actor = teacher_actor
        self.teacher_value = teacher_value
        self.value_normalizer = value_normalizer
        self.critic_keys = tuple(critic_keys)
        self.critic_widths = tuple(int(width) for width in critic_widths)
        self.freeze()

    def freeze(self) -> None:
        for module in (
            self.teacher_actor,
            self.teacher_value,
            self.value_normalizer,
        ):
            module.requires_grad_(False)
            module.eval()
            for parameter in module.parameters():
                parameter.grad = None

    @torch.inference_mode()
    def get_frozen_teacher_value(
        self, teacher_critic_obs: torch.Tensor
    ) -> torch.Tensor:
        """Return frozen PPO value in raw summed-reward scale, shape ``[B]``."""
        if teacher_critic_obs.ndim < 2:
            raise ValueError("Teacher critic observation must have shape [..., D]")
        expected_width = sum(self.critic_widths)
        if teacher_critic_obs.shape[-1] != expected_width:
            raise ValueError(
                "Teacher critic observation width mismatch: "
                f"expected {expected_width}, got {teacher_critic_obs.shape[-1]}"
            )
        if not torch.isfinite(teacher_critic_obs).all():
            raise RuntimeError("Teacher critic observation contains NaN/Inf")

        # Teacher-value arithmetic is kept in float32 even when the surrounding
        # SAC update uses autocast/mixed precision.
        observations = teacher_critic_obs.detach().to(dtype=torch.float32)
        chunks = observations.split(self.critic_widths, dim=-1)
        value_td = TensorDict(
            dict(zip(self.critic_keys, chunks)),
            batch_size=observations.shape[:-1],
            device=observations.device,
        )
        device_type = observations.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            value_groups = self.teacher_value(value_td)["state_value"].float()
            raw_value_groups = self.value_normalizer.denormalize(value_groups).float()
            teacher_value = raw_value_groups.sum(dim=-1)
        if teacher_value.shape != observations.shape[:-1]:
            raise RuntimeError(
                "Frozen Teacher value has the wrong shape: "
                f"expected {tuple(observations.shape[:-1])}, "
                f"got {tuple(teacher_value.shape)}"
            )
        if not torch.isfinite(teacher_value).all():
            raise RuntimeError("Frozen Teacher value contains NaN/Inf")
        return teacher_value.detach()


@dataclass(frozen=True)
class TeacherValueBottleneck:
    """One selected Student transition and its scale-stable diagnostics."""

    index: int
    phase: float
    score: float
    raw_teacher_td_residual: float
    normalized_teacher_td_residual: float
    smoothed_normalized_teacher_td_residual: float
    student_candidate_count: int
    threshold_detected: bool
    used_fallback: bool
    selection_origin: str


class TeacherValueBottleneckDetector:
    """Find the earliest sustained Student-only Teacher-value degradation.

    Teacher-executed rows break both the moving-average window and the
    consecutive-threshold run.  This prevents separated Student actions in a
    mixed DAgger episode from being treated as one continuous failure onset.
    The only persistent adaptive value is a residual-scale EMA used for
    threshold units; it never influences the Actor or its fixed BC coefficient.
    """

    def __init__(
        self,
        threshold: float,
        smoothing_window: int,
        min_consecutive: int,
        terminal_exclusion_steps: int,
        residual_scale_ema_decay: float,
        eps: float = 1e-6,
    ) -> None:
        self.threshold = _finite_scalar("threshold", threshold, positive=True)
        for name, value in (
            ("smoothing_window", smoothing_window),
            ("min_consecutive", min_consecutive),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(terminal_exclusion_steps, bool)
            or not isinstance(terminal_exclusion_steps, int)
            or terminal_exclusion_steps < 0
        ):
            raise ValueError("terminal_exclusion_steps must be non-negative")
        decay = _finite_scalar("residual_scale_ema_decay", residual_scale_ema_decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("residual_scale_ema_decay must lie in [0, 1)")
        self.smoothing_window = int(smoothing_window)
        self.min_consecutive = int(min_consecutive)
        self.terminal_exclusion_steps = int(terminal_exclusion_steps)
        self.residual_scale_ema_decay = decay
        self.eps = _finite_scalar("eps", eps, positive=True)
        self.bottleneck_residual_scale_ema = 1.0
        self.num_scale_updates = 0
        self.last_diagnostics = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics() -> dict[str, float | bool]:
        return {
            "student_transition_count": 0.0,
            "student_candidate_count": 0.0,
            "threshold_detected": False,
            "used_fallback": False,
            "no_candidate": True,
        }

    @torch.no_grad()
    def update_residual_scale(self, student_td_residual: torch.Tensor) -> None:
        residual = torch.as_tensor(student_td_residual).detach().float().reshape(-1)
        if residual.numel() == 0:
            return
        if not torch.isfinite(residual).all():
            raise RuntimeError("Student Teacher TD residual contains NaN/Inf")
        absolute_mean = residual.abs().mean().item()
        self.bottleneck_residual_scale_ema = max(
            self.residual_scale_ema_decay * self.bottleneck_residual_scale_ema
            + (1.0 - self.residual_scale_ema_decay) * absolute_mean,
            self.eps,
        )
        self.num_scale_updates += 1

    @torch.no_grad()
    def detect(
        self,
        teacher_td_residual: torch.Tensor,
        source_id: torch.Tensor,
        reference_phase: torch.Tensor,
        true_terminal: torch.Tensor,
        timeout: torch.Tensor,
        *,
        fallback_mode: str = "none",
    ) -> TeacherValueBottleneck | None:
        if fallback_mode not in {"none", "value_argmin"}:
            raise ValueError(
                "bottleneck fallback mode must be 'none' or 'value_argmin'"
            )
        residual = torch.as_tensor(teacher_td_residual).detach().float().reshape(-1)
        source = torch.as_tensor(source_id).detach().long().reshape(-1)
        phase = torch.as_tensor(reference_phase).detach().float().reshape(-1)
        terminal = torch.as_tensor(true_terminal).detach().bool().reshape(-1)
        timeout = torch.as_tensor(timeout).detach().bool().reshape(-1)
        size = residual.numel()
        if not all(
            value.numel() == size for value in (source, phase, terminal, timeout)
        ):
            raise ValueError("Bottleneck episode tensors must have identical length")
        if size == 0:
            self.last_diagnostics = self._empty_diagnostics()
            return None
        if not torch.isfinite(phase).all():
            raise RuntimeError("Bottleneck reference phase contains NaN/Inf")
        if bool((terminal & timeout).any()):
            raise ValueError("A transition cannot be both true-terminal and timeout")

        student = source == SOURCE_STUDENT
        student_count = int(student.sum().item())
        if student_count == 0:
            self.last_diagnostics = self._empty_diagnostics()
            return None
        if not torch.isfinite(residual[student]).all():
            raise RuntimeError("Student Teacher TD residual contains NaN/Inf")
        candidate = student & ~terminal
        terminal_indices = terminal.nonzero(as_tuple=False).squeeze(-1)
        if terminal_indices.numel():
            final_terminal = int(terminal_indices[-1].item())
            cutoff = final_terminal - self.terminal_exclusion_steps
            candidate &= torch.arange(size, device=candidate.device) < cutoff
        candidate_count = int(candidate.sum().item())
        scale_population = candidate
        if not bool(scale_population.any()):
            scale_population = student & ~terminal
        self.update_residual_scale(residual[scale_population])
        normalized = residual / (self.bottleneck_residual_scale_ema + self.eps)

        # A causal moving average is restarted by every Teacher-executed row.
        smoothed = torch.full_like(normalized, float("nan"))
        run: list[torch.Tensor] = []
        for index in range(size):
            if not bool(student[index]):
                run.clear()
                continue
            run.append(normalized[index])
            if len(run) > self.smoothing_window:
                del run[0]
            smoothed[index] = torch.stack(run).mean()

        diagnostics: dict[str, float | bool] = {
            "student_transition_count": float(student_count),
            "student_candidate_count": float(candidate_count),
            "threshold_detected": False,
            "used_fallback": False,
            "no_candidate": candidate_count == 0,
        }

        selected_index: int | None = None
        run_start = -1
        run_length = 0
        for index in range(size):
            below = bool(
                candidate[index]
                and torch.isfinite(smoothed[index])
                and smoothed[index] < -self.threshold
            )
            if below:
                if run_length == 0:
                    run_start = index
                run_length += 1
                if run_length >= self.min_consecutive:
                    selected_index = run_start
                    diagnostics["threshold_detected"] = True
                    break
            else:
                run_start = -1
                run_length = 0

        used_fallback = False
        if selected_index is None:
            if fallback_mode == "none":
                self.last_diagnostics = diagnostics
                return None
            fallback_mask = candidate
            if not bool(fallback_mask.any()):
                self.last_diagnostics = diagnostics
                return None
            fallback_indices = fallback_mask.nonzero(as_tuple=False).squeeze(-1)
            fallback_values = smoothed.index_select(0, fallback_indices)
            selected_index = int(fallback_indices[fallback_values.argmin()].item())
            used_fallback = True
            diagnostics["used_fallback"] = True

        selected_smoothed = float(smoothed[selected_index].item())
        result = TeacherValueBottleneck(
            index=selected_index,
            phase=float(phase[selected_index].item()),
            score=max(0.0, -selected_smoothed),
            raw_teacher_td_residual=float(residual[selected_index].item()),
            normalized_teacher_td_residual=float(normalized[selected_index].item()),
            smoothed_normalized_teacher_td_residual=selected_smoothed,
            student_candidate_count=candidate_count,
            threshold_detected=bool(diagnostics["threshold_detected"]),
            used_fallback=used_fallback,
            selection_origin="value_argmin" if used_fallback else "threshold",
        )
        self.last_diagnostics = diagnostics
        return result

    def state_dict(self) -> dict:
        return {
            "bottleneck_residual_scale_ema": float(self.bottleneck_residual_scale_ema),
            "num_scale_updates": int(self.num_scale_updates),
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError("Bottleneck detector state must be a mapping")
        if "bottleneck_residual_scale_ema" not in state_dict:
            raise ValueError("Bottleneck detector state lacks residual scale EMA")
        scale = _finite_scalar(
            "bottleneck_residual_scale_ema",
            state_dict["bottleneck_residual_scale_ema"],
            positive=True,
        )
        updates = state_dict.get("num_scale_updates", 0)
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("Bottleneck detector scale update count is invalid")
        self.bottleneck_residual_scale_ema = scale
        self.num_scale_updates = int(updates)
        self.last_diagnostics = self._empty_diagnostics()


@dataclass(frozen=True)
class TeacherValueTerms:
    teacher_v: torch.Tensor
    teacher_v_next: torch.Tensor
    potential_delta: torch.Tensor
    shaped_reward: torch.Tensor
    teacher_td_residual: torch.Tensor


@torch.no_grad()
def compute_teacher_value_continuation(
    *,
    teacher_v: torch.Tensor,
    teacher_v_next: torch.Tensor,
    terminated: torch.Tensor,
    command_finished: torch.Tensor,
    time_limit: torch.Tensor,
    replay_discount: torch.Tensor | float,
    gamma: float,
    semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS,
) -> torch.Tensor:
    """Return source-PPO-compatible discounted frozen-value continuation.

    The frozen PPO Teacher was trained with the current value copied into every
    done row and with only ``terminated`` masking its one-step value term.  Its
    exact boundary contract is therefore different from SAC's: physical
    termination contributes zero, while a non-physical command completion or
    time-limit boundary self-bootstraps from ``V_T(s_t)``.  Ordinary rows use
    ``V_T(s_{t+1})``.  Physical termination wins when reset causes coincide.
    """
    if semantics != TEACHER_VALUE_BOUNDARY_SEMANTICS:
        raise ValueError(
            f"unsupported frozen Teacher value boundary semantics {semantics!r}"
        )
    gamma = _finite_scalar("teacher value gamma", gamma)
    if gamma < 0.0:
        raise ValueError("teacher value gamma must be non-negative")

    current = torch.as_tensor(teacher_v).detach().float().reshape(-1)
    following = (
        torch.as_tensor(teacher_v_next)
        .detach()
        .to(device=current.device, dtype=torch.float32)
        .reshape(-1)
    )
    physical = (
        torch.as_tensor(terminated)
        .detach()
        .to(device=current.device, dtype=torch.bool)
        .reshape(-1)
    )
    command = (
        torch.as_tensor(command_finished)
        .detach()
        .to(device=current.device, dtype=torch.bool)
        .reshape(-1)
    )
    timeout = (
        torch.as_tensor(time_limit)
        .detach()
        .to(device=current.device, dtype=torch.bool)
        .reshape(-1)
    )
    discount = torch.as_tensor(
        replay_discount, device=current.device, dtype=torch.float32
    )
    if discount.ndim == 0:
        discount = discount.expand_as(current)
    else:
        discount = discount.reshape(-1)
    values = (current, following, physical, command, timeout, discount)
    if any(value.shape != current.shape for value in values[1:]):
        raise ValueError("Teacher continuation tensors must have identical shape")
    if not (
        torch.isfinite(current).all()
        and torch.isfinite(following).all()
        and torch.isfinite(discount).all()
    ):
        raise RuntimeError("Teacher continuation contains NaN/Inf")

    nonphysical_self_bootstrap = ~physical & (command | timeout)
    continuation_value = torch.where(
        physical,
        torch.zeros_like(current),
        torch.where(nonphysical_self_bootstrap, current, following),
    )
    continuation = gamma * discount * continuation_value
    if not torch.isfinite(continuation).all():
        raise RuntimeError("Discounted Teacher continuation contains NaN/Inf")
    return continuation.detach()


@torch.no_grad()
def compute_teacher_value_terms(
    get_teacher_value: Callable[[torch.Tensor], torch.Tensor],
    teacher_critic_obs: torch.Tensor,
    next_teacher_critic_obs: torch.Tensor,
    raw_reward: torch.Tensor,
    *,
    terminated: torch.Tensor,
    command_finished: torch.Tensor,
    time_limit: torch.Tensor,
    replay_discount: torch.Tensor | float,
    gamma: float,
    semantics: str = TEACHER_VALUE_BOUNDARY_SEMANTICS,
    tvkd_lambda: float,
    potential_clip: float | None = None,
) -> TeacherValueTerms:
    """Compute source-PPO-compatible shaping and raw-reward TD residual."""
    raw_reward = raw_reward.detach().float().reshape(-1)
    if not torch.isfinite(raw_reward).all():
        raise RuntimeError("TVKD raw reward contains NaN/Inf")

    teacher_v = get_teacher_value(teacher_critic_obs).float().reshape(-1)
    teacher_v_next = get_teacher_value(next_teacher_critic_obs).float().reshape(-1)
    if teacher_v.shape != raw_reward.shape or teacher_v_next.shape != raw_reward.shape:
        raise ValueError("Teacher values and rewards must have identical batch shape")
    fixed_v = teacher_v
    fixed_v_next = teacher_v_next
    if potential_clip is not None:
        clip = _finite_scalar("potential_clip", potential_clip, positive=True)
        fixed_v = fixed_v.clamp(-clip, clip)
        fixed_v_next = fixed_v_next.clamp(-clip, clip)

    fixed_continuation = compute_teacher_value_continuation(
        teacher_v=fixed_v,
        teacher_v_next=fixed_v_next,
        terminated=terminated,
        command_finished=command_finished,
        time_limit=time_limit,
        replay_discount=replay_discount,
        gamma=gamma,
        semantics=semantics,
    )
    raw_continuation = compute_teacher_value_continuation(
        teacher_v=teacher_v,
        teacher_v_next=teacher_v_next,
        terminated=terminated,
        command_finished=command_finished,
        time_limit=time_limit,
        replay_discount=replay_discount,
        gamma=gamma,
        semantics=semantics,
    )
    potential_delta = fixed_continuation - fixed_v
    shaped_reward = raw_reward + float(tvkd_lambda) * potential_delta
    # Bottleneck detection uses raw reward and the unclipped Teacher value.
    teacher_td_residual = raw_reward + raw_continuation - teacher_v
    for name, value in (
        ("potential delta", potential_delta),
        ("shaped reward", shaped_reward),
        ("Teacher TD residual", teacher_td_residual),
    ):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"TVKD {name} contains NaN/Inf")
    return TeacherValueTerms(
        teacher_v=teacher_v.detach(),
        teacher_v_next=teacher_v_next.detach(),
        potential_delta=potential_delta.detach(),
        shaped_reward=shaped_reward.detach(),
        teacher_td_residual=teacher_td_residual.detach(),
    )


class TVKDDistributionalFastSACTeacherBC(DistributionalFastSACTeacherBC):
    """Twin-C51 FastSAC with frozen shaping and bottleneck-aligned replay."""

    @staticmethod
    def _validate_td3_config(cfg) -> None:
        DistributionalFastSACTeacherBC._validate_td3_config(cfg)
        _validate_tvkd_algorithm_config(cfg)

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self.teacher_value_wrapper = FrozenTeacherValueWrapper(
            self.actor,
            self.critic,
            self.value_norm,
            tuple(self.q_critic_keys),
            tuple(self._q_critic_widths),
        )
        self.teacher_value_bottleneck_detector = TeacherValueBottleneckDetector(
            threshold=float(cfg.bottleneck_threshold),
            smoothing_window=int(cfg.bottleneck_smoothing_window),
            min_consecutive=int(cfg.bottleneck_min_consecutive),
            terminal_exclusion_steps=int(cfg.bottleneck_terminal_exclusion_steps),
            residual_scale_ema_decay=float(cfg.bottleneck_residual_scale_ema_decay),
            eps=float(cfg.bottleneck_eps),
        )
        self._bottleneck_episode_histories: list[dict[str, list]] | None = None
        self._reset_student_replay_episode_tracking()
        self._reset_bottleneck_statistics()
        self._last_tvkd_diagnostics: dict[str, float] = {}

    def get_frozen_teacher_value(
        self, teacher_critic_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.teacher_value_wrapper.get_frozen_teacher_value(teacher_critic_obs)

    def _tvkd_enabled(self) -> bool:
        return (
            bool(self.cfg.use_tvkd_value_shaping) and float(self.cfg.tvkd_lambda) != 0.0
        )

    @torch.no_grad()
    def _distributional_fastsac_target(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        # Preserve the baseline RNG sequence and target bit-for-bit when TVKD
        # is disabled (including the explicit lambda=0 equivalence setting).
        if not self._tvkd_enabled():
            baseline_result = (
                DistributionalFastSACTeacherBC._distributional_fastsac_target(
                    self, batch
                )
            )
            if "rewards" not in batch:
                return baseline_result
            target, metrics, log_prob = baseline_result
            metrics = dict(metrics)
            raw_reward = batch["rewards"].detach().float()
            zero = raw_reward.new_zeros(())
            metrics.update(
                {
                    "tvkd_potential_delta_mean": zero,
                    "tvkd_potential_delta_std": zero,
                    "tvkd_potential_delta_min": zero,
                    "tvkd_potential_delta_max": zero,
                    "tvkd_raw_reward_mean": raw_reward.mean(),
                    "tvkd_shaped_reward_mean": raw_reward.mean(),
                    "tvkd_shaped_reward_std": raw_reward.std(unbiased=False),
                }
            )
            return target, metrics, log_prob

        next_dist = self._actor_dist_from_flat(batch["next_observations"])
        next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        next_log_prob = self._normalized_action_log_prob(next_raw_log_prob)
        required_boundary = {
            REPLAY_TERMINATED_KEY,
            REPLAY_COMMAND_FINISHED_KEY,
            REPLAY_TIME_LIMIT_KEY,
        }
        missing_boundary = required_boundary.difference(batch)
        if missing_boundary:
            raise KeyError(
                "TVKD shaping requires explicit replay boundary metadata; missing "
                f"{sorted(missing_boundary)}"
            )
        done = batch["dones"].bool()
        terminated = batch[REPLAY_TERMINATED_KEY].bool()
        command_finished = batch[REPLAY_COMMAND_FINISHED_KEY].bool()
        time_limit = batch[REPLAY_TIME_LIMIT_KEY].bool()
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError("TVKD replay contains an unknown boundary cause")
        expected_q_truncation = time_limit & ~command_finished & ~terminated
        if not torch.equal(batch["truncations"].bool(), expected_q_truncation):
            raise RuntimeError(
                "TVKD replay Q truncation disagrees with explicit boundary metadata"
            )
        bootstrap = (batch["truncations"].bool() | ~done).float()
        effective_discount = float(self.cfg.gamma) * batch["discounts"]
        terms = compute_teacher_value_terms(
            self.get_frozen_teacher_value,
            batch["critic_observations"],
            batch["next_critic_observations"],
            batch["rewards"],
            terminated=terminated,
            command_finished=command_finished,
            time_limit=time_limit,
            replay_discount=batch["discounts"],
            gamma=float(self.cfg.gamma),
            semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
            tvkd_lambda=float(self.cfg.tvkd_lambda),
            potential_clip=self.cfg.tvkd_potential_clip,
        )
        alpha = self.log_alpha.exp()
        entropy_tax = effective_discount * bootstrap * alpha * next_log_prob
        soft_reward = terms.shaped_reward.to(entropy_tax) - entropy_tax
        if not torch.isfinite(soft_reward).all():
            raise RuntimeError("TVKD soft C51 reward contains NaN/Inf")

        target_logits = self.qnet_target(
            batch["next_critic_observations"], self._q_action_input(next_action)
        )
        target_probabilities = F.softmax(target_logits, dim=-1)
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
            raise RuntimeError("TVKD C51 target must contain exactly two Q heads")
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
        projected_heads = torch.stack(projected_heads, dim=0)
        projected_expected_heads = (projected_heads * self.qnet_target.support).sum(
            dim=-1
        )
        selected_head = projected_expected_heads.argmin(dim=0)
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
        raw_expected_heads = (target_probabilities * self.qnet_target.support).sum(
            dim=-1
        )
        reward_abs = batch["rewards"].abs().mean()

        def population_std(value: torch.Tensor) -> torch.Tensor:
            return value.float().std(unbiased=False)

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
            "left_support_projection_clipping_fraction": 0.5
            * (q1_left_support_clip_fraction + q2_left_support_clip_fraction),
            "right_support_projection_clipping_fraction": 0.5
            * (q1_right_support_clip_fraction + q2_right_support_clip_fraction),
            "target_smoothing_noise_norm": soft_reward.new_zeros(()),
            "target_noise_free_action_abs_mean": next_dist.mean.abs().mean(),
            "target_sample_action_abs_mean": next_action.abs().mean(),
            "target_log_prob_mean": next_log_prob.mean(),
            "entropy_tax_mean": entropy_tax.mean(),
            "entropy_tax_abs_mean": entropy_tax.abs().mean(),
            "entropy_tax_reward_abs_ratio": entropy_tax.abs().mean()
            / reward_abs.clamp_min(torch.finfo(reward_abs.dtype).eps),
            "alpha": alpha.detach(),
            "tvkd_teacher_value_mean": terms.teacher_v.mean(),
            "tvkd_teacher_value_std": population_std(terms.teacher_v),
            "tvkd_teacher_next_value_mean": terms.teacher_v_next.mean(),
            "tvkd_teacher_next_value_std": population_std(terms.teacher_v_next),
            "tvkd_potential_delta_mean": terms.potential_delta.mean(),
            "tvkd_potential_delta_std": population_std(terms.potential_delta),
            "tvkd_potential_delta_min": terms.potential_delta.min(),
            "tvkd_potential_delta_max": terms.potential_delta.max(),
            "tvkd_raw_reward_mean": batch["rewards"].float().mean(),
            "tvkd_shaped_reward_mean": terms.shaped_reward.mean(),
            "tvkd_shaped_reward_std": population_std(terms.shaped_reward),
        }
        return selected_target.detach(), metrics, next_log_prob.detach()

    # Compatibility alias used by tests and focused callers.
    _soft_c51_target = _distributional_fastsac_target

    def _reset_bottleneck_statistics(self) -> None:
        self._bottleneck_unsuccessful_episode_count = 0
        self._bottleneck_episodes_with_student_candidates = 0
        self._bottleneck_no_value_bottleneck_count = 0
        self._bottleneck_value_argmin_ablation_count = 0
        self._bottleneck_failed_student_episode_count = 0
        self._bottleneck_student_candidate_count = 0
        self._bottleneck_detected_count = 0
        self._bottleneck_fallback_count = 0
        self._bottleneck_no_candidate_count = 0
        self._bottleneck_selected_count = 0
        self._bottleneck_selected_step_sum = 0.0
        self._bottleneck_selected_phase_sum = 0.0
        self._bottleneck_score_sum = 0.0
        self._bottleneck_score_max = 0.0
        self._bottleneck_raw_td_residual_sum = 0.0
        self._bottleneck_normalized_td_residual_sum = 0.0
        self._bottleneck_teacher_sequences_inserted = 0
        self._bottleneck_teacher_transitions_inserted = 0
        self._bottleneck_phase_match_distance_sum = 0.0
        self._bottleneck_phase_match_distance_count = 0
        self._bottleneck_next_student_episode_id = 0
        self._student_focus_rows_marked = 0
        self._student_focus_rows_missing = 0
        self._failure_phase_student_focused_rows = 0
        self._failure_phase_student_uniform_fallback_rows = 0
        self._last_bottleneck_metadata: dict[str, float | int | str] = {}
        self._last_value_argmin_metadata: dict[str, float | int | str] = {}
        self._verified_failure_motion_phase_histogram: dict[int, torch.Tensor] = {}

    def _reset_student_replay_episode_tracking(self) -> None:
        """Reset ephemeral IDs whose raw replay ring is intentionally fresh."""
        self._student_replay_env_episode_ids: list[int | None] | None = None
        self._student_replay_env_episode_steps: list[int] | None = None
        self._student_replay_next_episode_id = 0
        self._student_replay_episode_id_grid: torch.Tensor | None = None
        self._student_replay_episode_step_grid: torch.Tensor | None = None
        self._pending_student_focus_events: list[tuple[int, torch.Tensor]] = []

    @staticmethod
    def _new_bottleneck_episode_history() -> dict[str, list]:
        return {
            "phase": [],
            "teacher_td_residual": [],
            "source_id": [],
            "motion_id": [],
            "true_terminal": [],
            "timeout": [],
        }

    @torch.no_grad()
    def _batched_frozen_teacher_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Evaluate each cached state once while bounding inference memory."""
        if observation.ndim != 2:
            raise ValueError("Bottleneck Teacher observations must be rank two")
        if observation.shape[0] == 0:
            return observation.new_empty((0,), dtype=torch.float32)
        microbatch = max(1, int(getattr(self.cfg, "q_batch_size", 4096)))
        values = [
            self.get_frozen_teacher_value(observation[start : start + microbatch])
            for start in range(0, observation.shape[0], microbatch)
        ]
        return torch.cat(values, dim=0)

    @torch.no_grad()
    def _student_teacher_td_residual_grid(
        self, rollout: TensorDict, student: torch.Tensor
    ) -> torch.Tensor:
        """Cache raw-reward Teacher TD residuals for executed Student rows.

        Adjacent Student transitions share their state value: the union of
        required episode states is normalized and evaluated once. Terminal
        next/reset observations are not forwarded because their bootstrap mask
        is zero.
        """
        num_envs, num_steps = (int(value) for value in rollout.batch_size)
        student = student.reshape(num_envs, num_steps).bool()
        residual_grid = torch.zeros(
            (num_envs, num_steps), dtype=torch.float32, device=student.device
        )
        if not bool(student.any()):
            return residual_grid
        final_batch = getattr(self, "_rollout_final_batch", None)
        if not isinstance(final_batch, Mapping):
            raise RuntimeError("Bottleneck replay requires the captured rollout final")

        reward = self._scalarize_q_reward(rollout[REWARD_KEY]).reshape(
            num_envs, num_steps
        )
        done = rollout[DONE_KEY].reshape(num_envs, num_steps).bool()
        terminated = rollout[TERM_KEY].reshape(num_envs, num_steps).bool()
        command_finished = (
            rollout["next", "stats", "command_finished"]
            .reshape(num_envs, num_steps)
            .bool()
        )
        time_limit = (
            rollout["next", "stats", "episode_time_limit"]
            .reshape(num_envs, num_steps)
            .bool()
        )
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError(
                "Bottleneck replay has an unknown or non-done boundary cause"
            )
        replay_discount = rollout["next", "discount"].reshape(num_envs, num_steps)
        truncation = _vaic_truncation_mask(rollout).reshape(num_envs, num_steps)
        q_bootstrap = (truncation | ~done).float()
        state_rows = []
        state_ids = []
        current_state_ids = []
        next_state_ids = []
        reward_rows = []
        q_bootstrap_rows = []
        terminated_rows = []
        command_finished_rows = []
        time_limit_rows = []
        replay_discount_rows = []
        flat_positions = []
        env_state_base = torch.arange(
            num_envs, device=student.device, dtype=torch.long
        ) * (num_steps + 1)
        # A timeout's real next observation is captured immediately before the
        # environment resets.  Give those states IDs outside the regular
        # rollout grid so they cannot alias the reset state at ``step + 1``.
        timeout_next_id_by_transition = torch.full(
            (num_envs * num_steps,),
            -1,
            dtype=torch.long,
            device=student.device,
        )
        truncation_batches = getattr(self, "_truncation_final_batches", [])
        if truncation_batches:
            captured_indices = torch.cat(
                [batch["indices"] for batch in truncation_batches], dim=0
            ).to(device=student.device, dtype=torch.long)
            captured_next_raw = torch.cat(
                [batch["next_critic_observations"] for batch in truncation_batches],
                dim=0,
            ).to(device=student.device)
            if (
                (captured_indices < 0).any()
                or (captured_indices >= num_envs * num_steps).any()
                or captured_indices.unique().numel() != captured_indices.numel()
            ):
                raise RuntimeError("Bottleneck timeout-final indices are invalid")
            captured_student = student.reshape(-1).index_select(0, captured_indices)
            captured_indices = captured_indices[captured_student]
            captured_next_raw = captured_next_raw[captured_student]
            virtual_ids = num_envs * (num_steps + 1) + torch.arange(
                captured_indices.numel(),
                dtype=torch.long,
                device=student.device,
            )
            timeout_next_id_by_transition.index_copy_(0, captured_indices, virtual_ids)
            state_rows.append(captured_next_raw)
            state_ids.append(virtual_ids)
        student_timeout = (student & truncation.bool()).reshape(-1)
        if bool(student_timeout.any()) and bool(
            (timeout_next_id_by_transition[student_timeout] < 0).any()
        ):
            raise RuntimeError(
                "Bottleneck replay lacks a captured timeout-final observation"
            )
        previous_bootstrap_student = torch.zeros(
            num_envs, dtype=torch.bool, device=student.device
        )
        current_raw = self._cat_replay_sources(rollout[:, 0], self.q_critic_keys)
        for step in range(num_steps):
            current_needed = student[:, step] | previous_bootstrap_student
            needed_envs = current_needed.nonzero(as_tuple=False).squeeze(-1)
            state_rows.append(current_raw[current_needed])
            state_ids.append(env_state_base[needed_envs] + step)
            next_raw = (
                self._cat_replay_sources(rollout[:, step + 1], self.q_critic_keys)
                if step + 1 < num_steps
                else final_batch["next_critic_observations"].reshape(
                    num_envs, self._q_critic_dim
                )
            )
            mask = student[:, step]
            reward_rows.append(reward[:, step][mask])
            q_bootstrap_rows.append(q_bootstrap[:, step][mask])
            terminated_rows.append(terminated[:, step][mask])
            command_finished_rows.append(command_finished[:, step][mask])
            time_limit_rows.append(time_limit[:, step][mask])
            replay_discount_rows.append(replay_discount[:, step][mask])
            env_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            current_state_ids.append(env_state_base[env_indices] + step)
            flat_position = env_indices * num_steps + step
            regular_next_ids = env_state_base[env_indices] + step + 1
            timeout_next_ids = timeout_next_id_by_transition.index_select(
                0, flat_position
            )
            next_state_ids.append(
                torch.where(
                    truncation[:, step][mask],
                    timeout_next_ids,
                    regular_next_ids,
                )
            )
            flat_positions.append(flat_position)
            previous_bootstrap_student = (
                mask & q_bootstrap[:, step].bool() & ~truncation[:, step]
            )
            current_raw = next_raw

        final_envs = previous_bootstrap_student.nonzero(as_tuple=False).squeeze(-1)
        state_rows.append(current_raw[previous_bootstrap_student])
        state_ids.append(env_state_base[final_envs] + num_steps)
        all_state_ids = torch.cat(state_ids, dim=0)
        if all_state_ids.unique().numel() != all_state_ids.numel():
            raise RuntimeError("Bottleneck Teacher state cache contains duplicates")
        sorted_state_ids, order = all_state_ids.sort()
        raw_states = torch.cat(state_rows, dim=0).index_select(0, order)
        snapshot = self._vecnorm_snapshot()
        states = self._normalize_replay_flat(
            raw_states,
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        state_values = self._batched_frozen_teacher_value(states).float()
        current_ids = torch.cat(current_state_ids, dim=0)
        next_ids = torch.cat(next_state_ids, dim=0)
        current_positions = torch.searchsorted(sorted_state_ids, current_ids)
        teacher_v = state_values.index_select(0, current_positions)
        transition_q_bootstrap = torch.cat(q_bootstrap_rows, dim=0).float()
        teacher_v_next = torch.zeros_like(teacher_v)
        q_bootstrap_rows_mask = transition_q_bootstrap.bool()
        if bool(q_bootstrap_rows_mask.any()):
            next_positions = torch.searchsorted(
                sorted_state_ids, next_ids[q_bootstrap_rows_mask]
            )
            teacher_v_next[q_bootstrap_rows_mask] = state_values.index_select(
                0, next_positions
            )
        teacher_continuation = compute_teacher_value_continuation(
            teacher_v=teacher_v,
            teacher_v_next=teacher_v_next,
            terminated=torch.cat(terminated_rows, dim=0),
            command_finished=torch.cat(command_finished_rows, dim=0),
            time_limit=torch.cat(time_limit_rows, dim=0),
            replay_discount=torch.cat(replay_discount_rows, dim=0),
            gamma=float(self.cfg.gamma),
            semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
        )
        transition_residual = (
            torch.cat(reward_rows, dim=0).float() + teacher_continuation - teacher_v
        )
        if not torch.isfinite(transition_residual).all():
            raise RuntimeError("Bottleneck Teacher TD residual contains NaN/Inf")
        residual_grid.reshape(-1).index_copy_(
            0,
            torch.cat(flat_positions, dim=0),
            transition_residual.to(residual_grid),
        )
        return residual_grid

    def _student_bottleneck_anchor_indices(
        self,
        history: Mapping[str, list],
        center: int,
    ) -> torch.Tensor:
        """Choose actual Student rows in the same bottleneck neighborhood."""
        history_length = len(history["phase"])
        lookback = int(self.cfg.failure_phase_lookback_steps)
        pre_steps = lookback // 2
        post_steps = lookback - pre_steps
        start = max(0, int(center) - pre_steps)
        end = min(history_length - 1, int(center) + post_steps)
        source = torch.tensor(history["source_id"], dtype=torch.long)
        if "motion_id" not in history or len(history["motion_id"]) != history_length:
            raise RuntimeError("Bottleneck history lacks aligned motion IDs")
        motion_id = torch.tensor(history["motion_id"], dtype=torch.long)
        center_motion_id = motion_id[int(center)]
        candidates = (
            (
                (source == SOURCE_STUDENT)
                & (motion_id == center_motion_id)
                & (torch.arange(history_length) >= start)
                & (torch.arange(history_length) <= end)
            )
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        if candidates.numel() == 0:
            return candidates
        requested = min(
            int(self.cfg.failure_phase_samples_per_failure),
            int(candidates.numel()),
        )
        positions = torch.linspace(0, candidates.numel() - 1, requested).round().long()
        selected = candidates.index_select(0, positions).unique(sorted=True)
        if source[int(center)] == SOURCE_STUDENT and not bool(
            (selected == int(center)).any()
        ):
            nearest = int((selected - int(center)).abs().argmin().item())
            selected[nearest] = int(center)
            selected = selected.sort().values.unique_consecutive()
        return selected

    def _queue_student_bottleneck_rows(
        self,
        history: Mapping[str, list],
        center: int,
        replay_episode_id: int,
    ) -> None:
        indices = self._student_bottleneck_anchor_indices(history, center)
        if indices.numel():
            events = getattr(self, "_pending_student_focus_events", None)
            if events is None:
                events = []
                self._pending_student_focus_events = events
            events.append((int(replay_episode_id), indices.detach().cpu()))

    @torch.no_grad()
    def _record_teacher_phase_match_distances(
        self,
        anchor_phases: torch.Tensor,
        anchor_motion_ids: torch.Tensor,
    ) -> None:
        anchor_phases = anchor_phases.detach().float().cpu().reshape(-1)
        anchor_motion_ids = anchor_motion_ids.detach().long().cpu().reshape(-1)
        if anchor_phases.shape != anchor_motion_ids.shape:
            raise ValueError("Teacher phase anchors and motion IDs must align")
        replay = getattr(self, "q_teacher_replay", None)
        if replay is None or int(getattr(replay, "size", 0)) < 1:
            return
        if REFERENCE_PHASE_KEY not in getattr(replay, "data", {}):
            return
        if not bool(getattr(self, "_teacher_phase_index_ready", False)) or not all(
            hasattr(self, name)
            for name in (
                "_teacher_replay_phase_bins",
                "_teacher_replay_motion_ids",
            )
        ):
            self._build_teacher_phase_index()
        teacher_phase = replay.data[REFERENCE_PHASE_KEY][: replay.size]
        teacher_phase = teacher_phase.reshape(replay.size, -1)[:, 0].float().cpu()
        teacher_bins = self._teacher_replay_phase_bins
        teacher_motion_ids = self._teacher_replay_motion_ids
        bins = int(self.cfg.failure_phase_num_bins)
        max_distance = getattr(self.cfg, "max_teacher_phase_match_distance", None)
        if max_distance is not None:
            max_distance = _finite_scalar(
                "max_teacher_phase_match_distance", max_distance
            )
            if max_distance < 0.0:
                raise ValueError(
                    "max_teacher_phase_match_distance must be non-negative"
                )
        for target, motion_id in zip(anchor_phases, anchor_motion_ids, strict=True):
            same_motion = (
                (teacher_motion_ids == int(motion_id))
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            if same_motion.numel() == 0:
                # The canonical sampler will uniformly backfill this anchor.
                # Do not report a cross-motion phase distance as a match.
                continue
            risk_bin = min(max(int(float(target) * bins), 0), bins - 1)
            bin_distance = (teacher_bins.index_select(0, same_motion) - risk_bin).abs()
            nearest_distance = int(bin_distance.min().item())
            if (
                max_distance is not None
                and nearest_distance / float(bins) > max_distance
            ):
                continue
            rows = same_motion[bin_distance == nearest_distance]
            # This mirrors the canonical sampler: exact motion first, nearest
            # occupied phase bin second, then a uniform row inside that pool.
            distance = (teacher_phase.index_select(0, rows) - target).abs().mean()
            self._bottleneck_phase_match_distance_sum += float(distance.item())
            self._bottleneck_phase_match_distance_count += 1

    @torch.no_grad()
    def _register_bottleneck_teacher_sequence(
        self,
        history: Mapping[str, list],
        center: int,
    ) -> int:
        # The verified histogram contains only the exact onset selected by the
        # sustained Teacher-value detector.  Neighborhood Student rows are still
        # marked separately, but their phases are not promoted to verified
        # Teacher anchors.
        center = int(center)
        phases = torch.tensor([history["phase"][center]], dtype=torch.float64)
        if "motion_id" not in history or len(history["motion_id"]) != len(
            history["phase"]
        ):
            raise RuntimeError("Bottleneck history lacks aligned motion IDs")
        motion_id = int(history["motion_id"][center])
        if motion_id < 0:
            raise RuntimeError("Bottleneck history contains a negative motion ID")
        motion_ids = torch.tensor([motion_id], dtype=torch.long)
        bins = int(self.cfg.failure_phase_num_bins)
        bin_indices = torch.floor(phases * bins).long().clamp_(0, bins - 1)
        self._failure_phase_histogram.index_add_(
            0, bin_indices, torch.ones_like(phases)
        )
        motion_histograms = getattr(
            self, "_verified_failure_motion_phase_histogram", None
        )
        if motion_histograms is None:
            motion_histograms = {}
            self._verified_failure_motion_phase_histogram = motion_histograms
        if not isinstance(motion_histograms, dict):
            raise RuntimeError("Verified motion histogram state must be a dict")
        motion_histogram = motion_histograms.get(motion_id)
        if motion_histogram is None:
            motion_histogram = torch.zeros(bins, dtype=torch.float64, device="cpu")
            motion_histograms[motion_id] = motion_histogram
        if (
            motion_histogram.shape != (bins,)
            or motion_histogram.dtype != torch.float64
            or motion_histogram.device.type != "cpu"
        ):
            raise RuntimeError(
                "Verified motion histogram must be CPU float64 with one value per bin"
            )
        motion_histogram.index_add_(
            0, bin_indices, torch.ones_like(phases, dtype=torch.float64)
        )
        count = int(phases.numel())
        self._failure_phase_episode_count += 1
        self._failure_phase_anchor_count += count
        self._bottleneck_teacher_sequences_inserted += 1
        self._bottleneck_teacher_transitions_inserted += count
        self._record_teacher_phase_match_distances(phases, motion_ids)
        self._failure_histogram_device_cache.clear()
        return count

    def _record_bottleneck_selection(
        self,
        *,
        episode_id: int,
        step: int,
        phase: float,
        score: float,
        raw_residual: float,
        normalized_residual: float,
        fallback: str,
    ) -> None:
        self._bottleneck_selected_count += 1
        self._bottleneck_selected_step_sum += float(step)
        self._bottleneck_selected_phase_sum += float(phase)
        self._bottleneck_score_sum += float(score)
        self._bottleneck_score_max = max(self._bottleneck_score_max, float(score))
        self._bottleneck_raw_td_residual_sum += float(raw_residual)
        self._bottleneck_normalized_td_residual_sum += float(normalized_residual)
        self._last_bottleneck_metadata = {
            "student_episode_id": int(episode_id),
            "bottleneck_step": int(step),
            "bottleneck_phase": float(phase),
            "bottleneck_score": float(score),
            "raw_teacher_td_residual": float(raw_residual),
            "normalized_teacher_td_residual": float(normalized_residual),
            "fallback": fallback,
        }

    @torch.no_grad()
    def _process_failed_student_episode(
        self, history: Mapping[str, list], *, replay_episode_id: int | None = None
    ) -> int:
        episode_id = self._bottleneck_next_student_episode_id
        self._bottleneck_next_student_episode_id += 1
        self._bottleneck_failed_student_episode_count += 1
        residual = torch.tensor(history["teacher_td_residual"], dtype=torch.float32)
        source = torch.tensor(history["source_id"], dtype=torch.long)
        phase = torch.tensor(history["phase"], dtype=torch.float32)
        terminal = torch.tensor(history["true_terminal"], dtype=torch.bool)
        timeout = torch.tensor(history["timeout"], dtype=torch.bool)
        fallback_mode = str(getattr(self.cfg, "bottleneck_fallback_mode", "none"))
        result = self.teacher_value_bottleneck_detector.detect(
            residual,
            source,
            phase,
            terminal,
            timeout,
            fallback_mode=fallback_mode,
        )
        diagnostics = self.teacher_value_bottleneck_detector.last_diagnostics
        candidate_count = int(diagnostics["student_candidate_count"])
        self._bottleneck_student_candidate_count += candidate_count
        if candidate_count:
            self._bottleneck_episodes_with_student_candidates += 1
        if bool(diagnostics["no_candidate"]):
            self._bottleneck_no_candidate_count += 1

        if result is None:
            self._bottleneck_no_value_bottleneck_count += 1
            return 0

        if result.threshold_detected:
            self._bottleneck_detected_count += 1
            self._record_bottleneck_selection(
                episode_id=episode_id,
                step=result.index,
                phase=result.phase,
                score=result.score,
                raw_residual=result.raw_teacher_td_residual,
                normalized_residual=result.normalized_teacher_td_residual,
                fallback="none",
            )
            if replay_episode_id is not None:
                self._queue_student_bottleneck_rows(
                    history, result.index, replay_episode_id
                )
            return self._register_bottleneck_teacher_sequence(history, result.index)

        if result.selection_origin != "value_argmin" or not result.used_fallback:
            raise RuntimeError(
                "Bottleneck detector returned an unknown selection origin"
            )
        # ``value_argmin`` is an explicit ablation diagnostic.  It never updates
        # the verified histogram or the verified Failure Student pool.
        self._bottleneck_fallback_count += 1
        self._bottleneck_value_argmin_ablation_count += 1
        self._bottleneck_no_value_bottleneck_count += 1
        self._last_value_argmin_metadata = {
            "student_episode_id": int(episode_id),
            "selected_step": int(result.index),
            "selected_phase": float(result.phase),
            "score": float(result.score),
            "raw_teacher_td_residual": float(result.raw_teacher_td_residual),
            "normalized_teacher_td_residual": float(
                result.normalized_teacher_td_residual
            ),
            "selection_origin": "value_argmin",
        }
        return 0

    @torch.no_grad()
    def _update_failure_phase_histogram(self, rollout: TensorDict) -> int:
        """Register only value-verified onsets from unsuccessful episodes."""
        if not bool(self.cfg.use_teacher_value_bottleneck_replay):
            # Disabling value bottlenecks leaves both focused sources empty.  In
            # particular, never fall through to the parent's terminal-lookback
            # curriculum.
            self._bottleneck_episode_histories = None
            self._reset_student_replay_episode_tracking()
            histogram = getattr(self, "_failure_phase_histogram", None)
            motion_histograms = getattr(
                self, "_verified_failure_motion_phase_histogram", None
            )
            if (torch.is_tensor(histogram) and bool((histogram != 0).any())) or (
                isinstance(motion_histograms, Mapping) and bool(motion_histograms)
            ):
                self._reset_failure_curriculum_state()
            self._verified_failure_motion_phase_histogram = {}
            return 0
        if len(rollout.batch_size) != 2:
            raise ValueError("bottleneck tracking requires an [env,time] rollout")
        if not isinstance(getattr(self, "_rollout_final_batch", None), Mapping):
            # Staging/finalization phases can intentionally suppress raw replay
            # capture. Do not splice an unobserved gap into episode histories.
            self._bottleneck_episode_histories = None
            self._student_replay_env_episode_ids = None
            self._student_replay_env_episode_steps = None
            self._student_replay_episode_id_grid = None
            self._student_replay_episode_step_grid = None
            events = getattr(self, "_pending_student_focus_events", None)
            if events is not None:
                events.clear()
            return 0
        num_envs, num_steps = (int(value) for value in rollout.batch_size)

        def grid(value: torch.Tensor, *, boolean: bool = False) -> torch.Tensor:
            value = value.reshape(num_envs, num_steps, -1)
            if boolean:
                return value.bool().any(dim=-1)
            if value.shape[-1] != 1:
                raise ValueError("bottleneck phase must have one scalar per step")
            return value[..., 0]

        phase = grid(self._reference_phase(rollout))
        raw_motion_id = rollout.get(REPLAY_MOTION_ID_KEY, None)
        command_manager = getattr(getattr(self, "env", None), "command_manager", None)
        dataset = getattr(command_manager, "dataset", None)
        num_motions = getattr(dataset, "num_motions", None)
        if callable(num_motions):
            num_motions = num_motions()
        if num_motions is not None:
            if isinstance(num_motions, bool):
                raise RuntimeError("Command dataset has an invalid motion count")
            num_motions = int(num_motions)
            if num_motions < 1:
                raise RuntimeError("Command dataset must contain at least one motion")
        if raw_motion_id is None:
            if num_motions != 1:
                raise KeyError(
                    "Bottleneck tracking requires replay_motion_id for multi-motion "
                    "or unknown-motion rollouts"
                )
            motion_id = torch.zeros_like(phase, dtype=torch.long)
        else:
            raw_motion_id = raw_motion_id.reshape(num_envs, num_steps, -1)
            if raw_motion_id.shape[-1] != 1:
                raise ValueError("replay_motion_id must have one scalar per step")
            raw_motion_id = raw_motion_id[..., 0]
            if raw_motion_id.dtype == torch.bool:
                raise ValueError("replay_motion_id cannot be boolean")
            if raw_motion_id.is_floating_point():
                if not torch.isfinite(raw_motion_id).all() or not torch.equal(
                    raw_motion_id, raw_motion_id.round()
                ):
                    raise ValueError("replay_motion_id must contain finite integers")
            motion_id = raw_motion_id.long()
            if bool((motion_id < 0).any()):
                raise ValueError("replay_motion_id cannot be negative")
            if num_motions is not None and bool((motion_id >= num_motions).any()):
                raise ValueError("replay_motion_id exceeds the command dataset")
        student = grid(rollout[DAGGER_IS_STUDENT_ACTION_KEY], boolean=True)
        done = grid(rollout[DONE_KEY], boolean=True)
        terminated = grid(rollout[TERM_KEY], boolean=True)
        command_finished = grid(
            rollout["next", "stats", "command_finished"], boolean=True
        )
        time_limit = grid(rollout["next", "stats", "episode_time_limit"], boolean=True)
        known_boundary = terminated | command_finished | time_limit
        if bool((done ^ known_boundary).any()):
            raise RuntimeError(
                "Bottleneck episode contains an unknown or non-done boundary cause"
            )
        timeout = time_limit & ~command_finished & ~terminated
        is_init_value = rollout.get("is_init", None)
        is_init = (
            torch.zeros_like(done)
            if is_init_value is None
            else grid(is_init_value, boolean=True)
        )
        residual = self._student_teacher_td_residual_grid(rollout, student)
        true_terminal = done & terminated
        packed = torch.stack(
            (
                phase.float(),
                residual.float(),
                student.float(),
                done.float(),
                true_terminal.float(),
                timeout.float(),
                command_finished.float(),
                is_init.float(),
            ),
            dim=-1,
        ).detach()
        if packed.device.type != "cpu":
            packed = packed.cpu()
        motion_id = motion_id.detach()
        if motion_id.device.type != "cpu":
            motion_id = motion_id.cpu()

        histories = self._bottleneck_episode_histories
        if histories is None or len(histories) != num_envs:
            histories = [
                self._new_bottleneck_episode_history() for _ in range(num_envs)
            ]
        env_episode_ids = getattr(self, "_student_replay_env_episode_ids", None)
        env_episode_steps = getattr(self, "_student_replay_env_episode_steps", None)
        if env_episode_ids is None or len(env_episode_ids) != num_envs:
            env_episode_ids = [None for _ in range(num_envs)]
            env_episode_steps = [0 for _ in range(num_envs)]
        if env_episode_steps is None or len(env_episode_steps) != num_envs:
            raise RuntimeError("Student replay episode-step tracking is misaligned")
        episode_id_grid = torch.full((num_envs, num_steps), -1, dtype=torch.long)
        episode_step_grid = torch.full((num_envs, num_steps), -1, dtype=torch.long)
        anchors_added = 0
        for env_index in range(num_envs):
            history = histories[env_index]
            for step in range(num_steps):
                row = packed[env_index, step]
                if bool(row[7]):
                    history = self._new_bottleneck_episode_history()
                    env_episode_ids[env_index] = None
                    env_episode_steps[env_index] = 0
                if env_episode_ids[env_index] is None:
                    next_episode_id = int(
                        getattr(self, "_student_replay_next_episode_id", 0)
                    )
                    env_episode_ids[env_index] = next_episode_id
                    self._student_replay_next_episode_id = next_episode_id + 1
                replay_episode_id = int(env_episode_ids[env_index])
                replay_episode_step = int(env_episode_steps[env_index])
                episode_id_grid[env_index, step] = replay_episode_id
                episode_step_grid[env_index, step] = replay_episode_step
                is_student = bool(row[2])
                history["phase"].append(float(row[0]))
                history["teacher_td_residual"].append(
                    float(row[1]) if is_student else 0.0
                )
                history["source_id"].append(
                    SOURCE_STUDENT if is_student else SOURCE_UNIFORM_TEACHER
                )
                history["motion_id"].append(int(motion_id[env_index, step].item()))
                history["true_terminal"].append(bool(row[4]))
                history["timeout"].append(bool(row[5]))

                if bool(row[3]) and not bool(row[6]):
                    self._bottleneck_unsuccessful_episode_count += 1
                    include_timeout = bool(
                        getattr(
                            self.cfg,
                            "bottleneck_include_unsuccessful_timeouts",
                            True,
                        )
                    )
                    outcome_is_eligible = not bool(row[5]) or include_timeout
                    if outcome_is_eligible and SOURCE_STUDENT in history["source_id"]:
                        anchors_added += self._process_failed_student_episode(
                            history, replay_episode_id=replay_episode_id
                        )
                env_episode_steps[env_index] = replay_episode_step + 1
                if bool(row[3]):
                    history = self._new_bottleneck_episode_history()
                    env_episode_ids[env_index] = None
                    env_episode_steps[env_index] = 0
            histories[env_index] = history
        self._bottleneck_episode_histories = histories
        self._student_replay_env_episode_ids = env_episode_ids
        self._student_replay_env_episode_steps = env_episode_steps
        self._student_replay_episode_id_grid = episode_id_grid
        self._student_replay_episode_step_grid = episode_step_grid
        return anchors_added

    @staticmethod
    def _student_focus_event_keys(
        events: list[tuple[int, torch.Tensor]] | tuple[tuple[int, torch.Tensor], ...],
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Encode episode/step pairs without retaining raw episode payloads."""
        stride = 1 << 32
        keys = []
        for episode_id, steps in events:
            steps = torch.as_tensor(steps, dtype=torch.long, device=device)
            if bool((steps < 0).any()) or bool((steps >= stride).any()):
                raise ValueError("Student replay episode step is outside uint32 range")
            keys.append(steps + int(episode_id) * stride)
        if not keys:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(keys).unique(sorted=True)

    @staticmethod
    def _student_focus_matches(
        episode_ids: torch.Tensor,
        episode_steps: torch.Tensor,
        event_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride = 1 << 32
        episode_ids = episode_ids.long()
        episode_steps = episode_steps.long()
        valid = (episode_ids >= 0) & (episode_steps >= 0)
        row_keys = episode_ids * stride + episode_steps
        matched = torch.zeros_like(valid)
        if event_keys.numel() and bool(valid.any()):
            sorted_events = event_keys.to(row_keys.device)
            valid_keys = row_keys[valid]
            positions = torch.searchsorted(sorted_events, valid_keys)
            safe_positions = positions.clamp_max(sorted_events.numel() - 1)
            valid_matches = (positions < sorted_events.numel()) & (
                sorted_events.index_select(0, safe_positions) == valid_keys
            )
            matched[valid] = valid_matches
        return matched, row_keys

    def _dagger_transition_chunks(self, td: TensorDict):
        """Attach stable episode IDs and exact failed-bottleneck eligibility."""
        pending_events = getattr(self, "_pending_student_focus_events", [])
        events = tuple(pending_events)
        event_keys_cpu = self._student_focus_event_keys(events, device="cpu")
        found_keys: list[torch.Tensor] = []

        replay = self.dagger_replay
        if (
            event_keys_cpu.numel()
            and STUDENT_REPLAY_EPISODE_ID_KEY in replay.data
            and STUDENT_REPLAY_EPISODE_STEP_KEY in replay.data
            and FAILURE_PHASE_STUDENT_SOURCE_KEY in replay.data
            and replay.size
        ):
            matched, row_keys = self._student_focus_matches(
                replay.data[STUDENT_REPLAY_EPISODE_ID_KEY][: replay.size],
                replay.data[STUDENT_REPLAY_EPISODE_STEP_KEY][: replay.size],
                event_keys_cpu.to(replay.device),
            )
            replay.data[FAILURE_PHASE_STUDENT_SOURCE_KEY][: replay.size].logical_or_(
                matched
            )
            if bool(matched.any()):
                found_keys.append(row_keys[matched].detach().cpu().unique())
            replay._valid_index_cache.pop(FAILURE_PHASE_STUDENT_SOURCE_KEY, None)

        id_grid = self._student_replay_episode_id_grid
        step_grid = self._student_replay_episode_step_grid
        try:
            for transitions in super()._dagger_transition_chunks(td):
                row_count = int(transitions["actions"].shape[0])
                if id_grid is None or step_grid is None:
                    episode_ids = torch.full(
                        (row_count,),
                        -1,
                        dtype=torch.long,
                        device=transitions["actions"].device,
                    )
                    episode_steps = torch.full_like(episode_ids, -1)
                else:
                    env_indices = transitions[_PREFILL_ENV_INDEX_KEY].long().cpu()
                    rollout_steps = transitions[_PREFILL_STEP_INDEX_KEY].long().cpu()
                    episode_ids = id_grid[env_indices, rollout_steps].to(
                        transitions["actions"].device
                    )
                    episode_steps = step_grid[env_indices, rollout_steps].to(
                        transitions["actions"].device
                    )
                matched, row_keys = self._student_focus_matches(
                    episode_ids,
                    episode_steps,
                    event_keys_cpu.to(episode_ids.device),
                )
                if bool(matched.any()):
                    found_keys.append(row_keys[matched].detach().cpu().unique())
                transitions[STUDENT_REPLAY_EPISODE_ID_KEY] = episode_ids
                transitions[STUDENT_REPLAY_EPISODE_STEP_KEY] = episode_steps
                transitions[FAILURE_PHASE_STUDENT_SOURCE_KEY] = matched
                yield transitions
        finally:
            found = torch.cat(found_keys).unique().numel() if found_keys else 0
            requested = int(event_keys_cpu.numel())
            self._student_focus_rows_marked = int(
                getattr(self, "_student_focus_rows_marked", 0)
            ) + int(found)
            self._student_focus_rows_missing = int(
                getattr(self, "_student_focus_rows_missing", 0)
            ) + max(requested - int(found), 0)
            pending_events.clear()
            self._student_replay_episode_id_grid = None
            self._student_replay_episode_step_grid = None

    @staticmethod
    def _mean_optional_metric(
        metrics: list[dict[str, torch.Tensor]], key: str
    ) -> float:
        values = [
            torch.as_tensor(item[key]).detach().float()
            for item in metrics
            if key in item
        ]
        return 0.0 if not values else torch.stack(values).mean().item()

    def _failure_teacher_buffer_size(self) -> int:
        """Return the effective successful-Teacher pool behind focused replay."""
        if self._has_canonical_replay_mix():
            rows, _ = self._verified_teacher_focus_pool()
            return int(rows.numel())
        if not bool(getattr(self, "_teacher_phase_index_ready", False)):
            return 0
        histogram = getattr(self, "_failure_phase_histogram", None)
        if not torch.is_tensor(histogram) or not bool((histogram > 0).any()):
            return 0
        risk_bins = (histogram > 0).nonzero(as_tuple=False).squeeze(-1)
        teacher_bins = self._teacher_phase_nearest_nonempty.index_select(0, risk_bins)
        unique_bins = teacher_bins.unique()
        return sum(
            int(self._teacher_phase_bin_rows[int(index)].numel())
            for index in unique_bins
        )

    def _bottleneck_metrics(self) -> dict[str, float]:
        selected = max(self._bottleneck_selected_count, 1)
        match_count = max(self._bottleneck_phase_match_distance_count, 1)
        detector = self.teacher_value_bottleneck_detector
        return {
            "unsuccessful_episode_count": float(
                self._bottleneck_unsuccessful_episode_count
            ),
            "episodes_with_student_candidates": float(
                self._bottleneck_episodes_with_student_candidates
            ),
            "value_detected_count": float(self._bottleneck_detected_count),
            "no_value_bottleneck_count": float(
                self._bottleneck_no_value_bottleneck_count
            ),
            "value_argmin_ablation_count": float(
                self._bottleneck_value_argmin_ablation_count
            ),
            "candidate_count": float(self._bottleneck_student_candidate_count),
            "failed_student_episode_count": float(
                self._bottleneck_failed_student_episode_count
            ),
            "student_candidate_count": float(self._bottleneck_student_candidate_count),
            "detected_count": float(self._bottleneck_detected_count),
            "fallback_count": float(self._bottleneck_fallback_count),
            "no_candidate_count": float(self._bottleneck_no_candidate_count),
            "selected_step_mean": self._bottleneck_selected_step_sum / selected,
            "selected_phase_mean": self._bottleneck_selected_phase_sum / selected,
            "score_mean": self._bottleneck_score_sum / selected,
            "score_max": float(self._bottleneck_score_max),
            "raw_td_residual_mean": (self._bottleneck_raw_td_residual_sum / selected),
            "raw_teacher_td_residual_mean": (
                self._bottleneck_raw_td_residual_sum / selected
            ),
            "normalized_td_residual_mean": (
                self._bottleneck_normalized_td_residual_sum / selected
            ),
            "normalized_teacher_td_residual_mean": (
                self._bottleneck_normalized_td_residual_sum / selected
            ),
            "residual_scale_ema": float(detector.bottleneck_residual_scale_ema),
            # The repository implements Failure Teacher as a virtual focused
            # source over the successful Teacher ring. These two values count
            # registered phase sequences/anchors, not copied replay tensors.
            "teacher_sequences_inserted": float(
                self._bottleneck_teacher_sequences_inserted
            ),
            "teacher_transitions_inserted": float(
                self._bottleneck_teacher_transitions_inserted
            ),
            "phase_match_distance_mean": (
                self._bottleneck_phase_match_distance_sum / match_count
            ),
            "teacher_phase_match_distance": (
                self._bottleneck_phase_match_distance_sum / match_count
            ),
            "failure_teacher_buffer_size": float(self._failure_teacher_buffer_size()),
            "student_focus_pool_size": float(
                self.dagger_replay.valid_count(FAILURE_PHASE_STUDENT_SOURCE_KEY)
                if FAILURE_PHASE_STUDENT_SOURCE_KEY in self.dagger_replay.data
                else 0
            ),
            "student_focus_rows_marked": float(self._student_focus_rows_marked),
            "student_focus_rows_missing": float(self._student_focus_rows_missing),
            "student_focus_sampled_rows": float(
                self._failure_phase_student_focused_rows
            ),
            "student_focus_uniform_fallback_rows": float(
                self._failure_phase_student_uniform_fallback_rows
            ),
            "student_focus_fraction_config": float(
                self._replay_mix_fractions("q")["failure_student"]
                / max(
                    self._replay_mix_fractions("q")["uniform_student"]
                    + self._replay_mix_fractions("q")["failure_student"],
                    torch.finfo(torch.float32).eps,
                )
            ),
            "student_focus_q_global_fraction_cap": float(
                self._replay_mix_fractions("q")["failure_student"]
            ),
            "student_focus_actor_global_fraction_cap": float(
                self._replay_mix_fractions("actor")["failure_student"]
            ),
        }

    def train_op(self, tensordict):
        info = DistributionalFastSACTeacherBC.train_op(self, tensordict)
        critic_metrics = getattr(self, "_fastsac_rollout_critic_metrics", [])
        actor_metrics = getattr(self, "_fastsac_rollout_actor_metrics", [])

        tvkd_mapping = {
            "teacher_value_mean": "tvkd_teacher_value_mean",
            "teacher_value_std": "tvkd_teacher_value_std",
            "teacher_next_value_mean": "tvkd_teacher_next_value_mean",
            "teacher_next_value_std": "tvkd_teacher_next_value_std",
            "potential_delta_mean": "tvkd_potential_delta_mean",
            "potential_delta_std": "tvkd_potential_delta_std",
            "potential_delta_min": "tvkd_potential_delta_min",
            "potential_delta_max": "tvkd_potential_delta_max",
            "raw_reward_mean": "tvkd_raw_reward_mean",
            "shaped_reward_mean": "tvkd_shaped_reward_mean",
            "shaped_reward_std": "tvkd_shaped_reward_std",
        }
        for output_name, metric_name in tvkd_mapping.items():
            info[f"tvkd/{output_name}"] = self._mean_optional_metric(
                critic_metrics, metric_name
            )
        for name, value in self._bottleneck_metrics().items():
            info[f"bottleneck/{name}"] = float(value)

        info["loss/critic"] = self._mean_optional_metric(critic_metrics, "critic_loss")
        info["loss/actor_total"] = self._mean_optional_metric(
            actor_metrics, "total_actor_loss"
        )
        info["loss/actor_sac"] = self._mean_optional_metric(
            actor_metrics, "sac_actor_loss"
        )
        info["loss/bc"] = self._mean_optional_metric(actor_metrics, "exact_bc_loss")
        info["loss/fixed_bc_coefficient"] = float(self.cfg.lambda_bc)
        info["loss/alpha"] = self._mean_optional_metric(critic_metrics, "alpha_loss")
        info["source/student_transition_count"] = float(
            info.get("fastsac/student_replay_rows_this_rollout", 0.0)
        )
        info["source/student_action_execution_ratio"] = float(
            info.get("fastsac/student_source_fraction", 0.0)
        )
        info["source/teacher_action_execution_ratio"] = float(
            info.get("fastsac/teacher_source_fraction", 0.0)
        )
        info["source/actor_bottleneck_student_fraction"] = self._mean_optional_metric(
            actor_metrics, "actor_failure_phase_student_fraction"
        )
        info["tvkd/method_distributional_tvkd_fastsac_teacher_bc_v4"] = 1.0
        self._last_tvkd_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if (
                key.startswith("tvkd/")
                or key.startswith("bottleneck/")
                or key.startswith("loss/")
                or key.startswith("source/")
            )
            and isinstance(value, (int, float))
        }
        return info

    def _checkpoint_config(self):
        common = DistributionalFastSACTeacherBC._checkpoint_config(self)
        common.update(
            {
                name: getattr(self.cfg, name)
                for name in (
                    "use_tvkd_value_shaping",
                    "tvkd_lambda",
                    "tvkd_potential_clip",
                    "use_teacher_value_bottleneck_replay",
                    "bottleneck_threshold",
                    "bottleneck_smoothing_window",
                    "bottleneck_min_consecutive",
                    "bottleneck_terminal_exclusion_steps",
                    "bottleneck_residual_scale_ema_decay",
                    "bottleneck_eps",
                    "bottleneck_fallback_mode",
                    "bottleneck_include_unsuccessful_timeouts",
                    "max_teacher_phase_match_distance",
                    "perception_replay_mode",
                    "teacher_value_return_semantics",
                    "teacher_value_boundary_semantics",
                    "teacher_value_reward_group_fingerprint",
                    "replay_task_fingerprint",
                    "failure_phase_student_fraction",
                    "value_norm",
                )
            }
        )
        for purpose in ("q", "actor", "perception"):
            for source in (
                "uniform_student",
                "failure_student",
                "uniform_teacher",
                "failure_teacher",
            ):
                name = f"{purpose}_{source}_fraction"
                common[name] = getattr(self.cfg, name)
        # The inherited checkpoint surfaces historically selected only a
        # subset of PPO/DAgger fields. V4 direct/programmatic resume promises
        # the same exact algorithm contract as the CLI, so retain every
        # behavior-affecting structured-config field. Identity/source-location
        # fields do not alter the already embedded policy and are excluded.
        non_behavioral = {"_target_", "name", "checkpoint_path"}
        for field in dataclass_fields(TVKDDistributionalFastSACTeacherBCConfig):
            if field.name in non_behavioral:
                continue
            value = copy.deepcopy(getattr(self.cfg, field.name))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = tuple(value)
            common.setdefault(field.name, value)
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "teacher_value_semantics": CRITIC_LEARNING_SEMANTICS,
                "bc_loss": "fixed_joint_valid_teacher_label_normalized_smooth_l1",
            }
        )
        return common

    def _q_backend_metadata(self):
        metadata = DistributionalFastSACTeacherBC._q_backend_metadata(self)
        metadata.update(
            {
                "target_semantics": CRITIC_LEARNING_SEMANTICS,
                "teacher_value_source": "frozen_checkpoint_ppo_critic",
                "teacher_value_output_scale": "valuenorm_denormalized_sum_groups",
                "teacher_value_norm_enabled": bool(self.cfg.value_norm),
                "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
            }
        )
        return metadata

    def _frozen_teacher_module_names(self) -> tuple[str, ...]:
        names = ["actor", "encoder_priv", "critic", "value_norm"]
        if hasattr(self, "height_encoder"):
            names.append("height_encoder")
        return tuple(names)

    def _failure_curriculum_checkpoint_state(self) -> dict:
        histogram = getattr(self, "_failure_phase_histogram", None)
        if not torch.is_tensor(histogram):
            histogram = torch.zeros(
                int(self.cfg.failure_phase_num_bins), dtype=torch.float64
            )
        expected_shape = (int(self.cfg.failure_phase_num_bins),)
        if (
            histogram.dtype != torch.float64
            or histogram.shape != expected_shape
            or not torch.isfinite(histogram).all()
            or bool((histogram < 0.0).any())
            or not torch.equal(histogram, histogram.round())
        ):
            raise RuntimeError("Verified global histogram state is invalid")
        motion_histograms = getattr(
            self, "_verified_failure_motion_phase_histogram", {}
        )
        if not isinstance(motion_histograms, Mapping):
            raise RuntimeError("Verified motion histogram state must be a mapping")
        histogram = histogram.detach().to(device="cpu", dtype=torch.float64).clone()
        serialized_motion_histograms = {}
        for motion_id, value in motion_histograms.items():
            if (
                isinstance(motion_id, bool)
                or not isinstance(motion_id, int)
                or motion_id < 0
                or not torch.is_tensor(value)
                or value.dtype != torch.float64
                or value.shape != histogram.shape
                or not torch.isfinite(value).all()
                or bool((value < 0.0).any())
                or not torch.equal(value, value.round())
            ):
                raise RuntimeError("Verified motion histogram state is invalid")
            serialized_motion_histograms[motion_id] = (
                value.detach().to(device="cpu").clone()
            )
        motion_total = torch.zeros_like(histogram)
        for value in serialized_motion_histograms.values():
            motion_total.add_(value)
        if not torch.equal(motion_total, histogram):
            raise RuntimeError("Verified global/motion histograms disagree")
        counters = {}
        for name in (
            "episode_count",
            "anchor_count",
            "uniform_fallback_rows",
            "focused_rows",
        ):
            value = getattr(self, f"_failure_phase_{name}", 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Verified histogram {name} is invalid")
            counters[name] = value
        if int(histogram.sum().item()) != counters["anchor_count"]:
            raise RuntimeError("Verified histogram/anchor count mismatch")
        return {
            "semantics": VERIFIED_HISTOGRAM_SEMANTICS,
            "histogram": histogram,
            **counters,
            "motion_histograms": serialized_motion_histograms,
        }

    def _load_failure_curriculum_checkpoint_state(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
        if state.get("semantics") != VERIFIED_HISTOGRAM_SEMANTICS:
            raise ValueError("TVKD verified histogram semantics mismatch")
        histogram = state.get("histogram")
        expected_bins = int(self.cfg.failure_phase_num_bins)
        if (
            not torch.is_tensor(histogram)
            or histogram.dtype != torch.float64
            or histogram.shape != (expected_bins,)
            or not torch.isfinite(histogram).all()
            or (histogram < 0.0).any()
            or not torch.equal(histogram, histogram.round())
        ):
            raise ValueError("TVKD failure curriculum histogram is invalid")
        counters = {}
        for name in (
            "episode_count",
            "anchor_count",
            "uniform_fallback_rows",
            "focused_rows",
        ):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD failure curriculum {name} is invalid")
            counters[name] = int(value)
        if int(histogram.sum().item()) != counters["anchor_count"]:
            raise ValueError("TVKD failure curriculum histogram/anchor count mismatch")
        motion_histograms = state.get("motion_histograms")
        if not isinstance(motion_histograms, Mapping):
            raise ValueError("TVKD verified histogram lacks motion partitions")
        restored_motion_histograms: dict[int, torch.Tensor] = {}
        for motion_id, value in motion_histograms.items():
            if (
                isinstance(motion_id, bool)
                or not isinstance(motion_id, int)
                or motion_id < 0
                or not torch.is_tensor(value)
                or value.dtype != torch.float64
                or value.shape != (expected_bins,)
                or not torch.isfinite(value).all()
                or bool((value < 0.0).any())
                or not torch.equal(value, value.round())
            ):
                raise ValueError("TVKD verified motion histogram is invalid")
            restored_motion_histograms[int(motion_id)] = (
                value.detach().to(device="cpu").clone()
            )
        motion_total = torch.zeros_like(histogram)
        for value in restored_motion_histograms.values():
            motion_total.add_(value)
        if not torch.equal(motion_total, histogram):
            raise ValueError("TVKD verified motion/global histogram counts disagree")
        self._failure_phase_histogram = histogram.detach().to(device="cpu").clone()
        self._failure_phase_episode_count = counters["episode_count"]
        self._failure_phase_anchor_count = counters["anchor_count"]
        self._failure_phase_uniform_fallback_rows = counters["uniform_fallback_rows"]
        self._failure_phase_focused_rows = counters["focused_rows"]
        self._verified_failure_motion_phase_histogram = restored_motion_histograms
        self._failure_histogram_device_cache.clear()

    def _reset_failure_curriculum_state(self) -> None:
        self._failure_phase_histogram = torch.zeros(
            int(self.cfg.failure_phase_num_bins), dtype=torch.float64
        )
        self._failure_phase_episode_count = 0
        self._failure_phase_anchor_count = 0
        self._failure_phase_uniform_fallback_rows = 0
        self._failure_phase_focused_rows = 0
        self._verified_failure_motion_phase_histogram = {}
        self._failure_histogram_device_cache.clear()

    def _bottleneck_replay_checkpoint_state(self) -> dict:
        return {
            "location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
            "detector": self.teacher_value_bottleneck_detector.state_dict(),
            "unsuccessful_episode_count": int(
                self._bottleneck_unsuccessful_episode_count
            ),
            "episodes_with_student_candidates": int(
                self._bottleneck_episodes_with_student_candidates
            ),
            "no_value_bottleneck_count": int(
                self._bottleneck_no_value_bottleneck_count
            ),
            "value_argmin_ablation_count": int(
                self._bottleneck_value_argmin_ablation_count
            ),
            "failed_student_episode_count": int(
                self._bottleneck_failed_student_episode_count
            ),
            "student_candidate_count": int(self._bottleneck_student_candidate_count),
            "detected_count": int(self._bottleneck_detected_count),
            "fallback_count": int(self._bottleneck_fallback_count),
            "no_candidate_count": int(self._bottleneck_no_candidate_count),
            "selected_count": int(self._bottleneck_selected_count),
            "teacher_sequences_inserted": int(
                self._bottleneck_teacher_sequences_inserted
            ),
            "teacher_transitions_inserted": int(
                self._bottleneck_teacher_transitions_inserted
            ),
            "phase_match_distance_count": int(
                self._bottleneck_phase_match_distance_count
            ),
            "next_student_episode_id": int(self._bottleneck_next_student_episode_id),
            "student_focus_rows_marked": int(self._student_focus_rows_marked),
            "student_focus_rows_missing": int(self._student_focus_rows_missing),
            "student_focus_sampled_rows": int(self._failure_phase_student_focused_rows),
            "student_focus_uniform_fallback_rows": int(
                self._failure_phase_student_uniform_fallback_rows
            ),
            "selected_step_sum": float(self._bottleneck_selected_step_sum),
            "selected_phase_sum": float(self._bottleneck_selected_phase_sum),
            "score_sum": float(self._bottleneck_score_sum),
            "score_max": float(self._bottleneck_score_max),
            "raw_td_residual_sum": float(self._bottleneck_raw_td_residual_sum),
            "normalized_td_residual_sum": float(
                self._bottleneck_normalized_td_residual_sum
            ),
            "phase_match_distance_sum": float(
                self._bottleneck_phase_match_distance_sum
            ),
            "last_metadata": copy.deepcopy(self._last_bottleneck_metadata),
            "last_value_argmin_metadata": copy.deepcopy(
                self._last_value_argmin_metadata
            ),
        }

    def _load_bottleneck_replay_checkpoint_state(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("TVKD checkpoint lacks bottleneck replay state")
        if state.get("location_semantics") != BOTTLENECK_LOCATION_SEMANTICS:
            raise ValueError("TVKD checkpoint bottleneck semantics mismatch")
        detector_state = state.get("detector")
        if not isinstance(detector_state, Mapping):
            raise ValueError("TVKD checkpoint lacks bottleneck detector state")
        self.teacher_value_bottleneck_detector.load_state_dict(detector_state)
        integer_fields = (
            "unsuccessful_episode_count",
            "episodes_with_student_candidates",
            "no_value_bottleneck_count",
            "value_argmin_ablation_count",
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
            "student_focus_rows_marked",
            "student_focus_rows_missing",
            "student_focus_sampled_rows",
            "student_focus_uniform_fallback_rows",
        )
        integers = {}
        for name in integer_fields:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD bottleneck replay {name} is invalid")
            integers[name] = int(value)
        float_fields = (
            "selected_step_sum",
            "selected_phase_sum",
            "score_sum",
            "score_max",
            "raw_td_residual_sum",
            "normalized_td_residual_sum",
            "phase_match_distance_sum",
        )
        floats = {
            name: _finite_scalar(f"bottleneck replay {name}", state.get(name))
            for name in float_fields
        }
        for name in (
            "selected_step_sum",
            "selected_phase_sum",
            "score_sum",
            "score_max",
            "phase_match_distance_sum",
        ):
            if floats[name] < 0.0:
                raise ValueError(f"TVKD bottleneck replay {name} is negative")
        metadata = state.get("last_metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("TVKD bottleneck replay metadata is invalid")
        argmin_metadata = state.get("last_value_argmin_metadata", {})
        if not isinstance(argmin_metadata, Mapping):
            raise ValueError("TVKD value-argmin metadata is invalid")

        self._bottleneck_unsuccessful_episode_count = integers[
            "unsuccessful_episode_count"
        ]
        self._bottleneck_episodes_with_student_candidates = integers[
            "episodes_with_student_candidates"
        ]
        self._bottleneck_no_value_bottleneck_count = integers[
            "no_value_bottleneck_count"
        ]
        self._bottleneck_value_argmin_ablation_count = integers[
            "value_argmin_ablation_count"
        ]
        self._bottleneck_failed_student_episode_count = integers[
            "failed_student_episode_count"
        ]
        self._bottleneck_student_candidate_count = integers["student_candidate_count"]
        self._bottleneck_detected_count = integers["detected_count"]
        self._bottleneck_fallback_count = integers["fallback_count"]
        self._bottleneck_no_candidate_count = integers["no_candidate_count"]
        self._bottleneck_selected_count = integers["selected_count"]
        self._bottleneck_teacher_sequences_inserted = integers[
            "teacher_sequences_inserted"
        ]
        self._bottleneck_teacher_transitions_inserted = integers[
            "teacher_transitions_inserted"
        ]
        self._bottleneck_phase_match_distance_count = integers[
            "phase_match_distance_count"
        ]
        self._bottleneck_next_student_episode_id = integers["next_student_episode_id"]
        self._student_focus_rows_marked = integers["student_focus_rows_marked"]
        self._student_focus_rows_missing = integers["student_focus_rows_missing"]
        self._failure_phase_student_focused_rows = integers[
            "student_focus_sampled_rows"
        ]
        self._failure_phase_student_uniform_fallback_rows = integers[
            "student_focus_uniform_fallback_rows"
        ]
        self._bottleneck_selected_step_sum = floats["selected_step_sum"]
        self._bottleneck_selected_phase_sum = floats["selected_phase_sum"]
        self._bottleneck_score_sum = floats["score_sum"]
        self._bottleneck_score_max = floats["score_max"]
        self._bottleneck_raw_td_residual_sum = floats["raw_td_residual_sum"]
        self._bottleneck_normalized_td_residual_sum = floats[
            "normalized_td_residual_sum"
        ]
        self._bottleneck_phase_match_distance_sum = floats["phase_match_distance_sum"]
        self._last_bottleneck_metadata = copy.deepcopy(dict(metadata))
        self._last_value_argmin_metadata = copy.deepcopy(dict(argmin_metadata))

    def _fastsac_checkpoint_state(self):
        # The baseline seam intentionally omits the frozen PPO modules because
        # baseline same-stage replay resume is disabled. TVKD's value function
        # is itself part of the learning rule, so persist its exact frozen
        # checkpoint chain alongside the normal SAC state.
        state = DistributionalFastSACTeacherBC._fastsac_checkpoint_state(self)
        optimizer_state = state.get("optimizer_resume_state")
        if not isinstance(optimizer_state, dict):
            raise RuntimeError("FastSAC checkpoint did not expose optimizer state")
        if bool(self.cfg.train_dr_estimator):
            optimizer = getattr(self, "opt_dr_estimator", None)
            if optimizer is None:
                raise RuntimeError("trainable DR estimator has no optimizer")
            optimizer_state["dr_estimator_optimizer"] = optimizer.state_dict()
        else:
            optimizer_state["dr_estimator_optimizer"] = None
        replay_mix_state = {
            purpose: {
                source: float(getattr(self.cfg, f"{purpose}_{source}_fraction"))
                for source in (
                    "uniform_student",
                    "failure_student",
                    "uniform_teacher",
                    "failure_teacher",
                )
            }
            for purpose in ("q", "actor", "perception")
        }
        vecnorm_fingerprint = getattr(self, "_replay_vecnorm_fingerprint", None)
        if not isinstance(vecnorm_fingerprint, str) or not vecnorm_fingerprint:
            raise RuntimeError("TVKD checkpoint lacks its Teacher VecNorm fingerprint")
        reward_fingerprint = _required_contract_fingerprint(
            "Teacher reward-group fingerprint",
            getattr(self.cfg, "teacher_value_reward_group_fingerprint", None),
        )
        task_fingerprint = _required_contract_fingerprint(
            "replay task fingerprint",
            getattr(self.cfg, "replay_task_fingerprint", None),
        )
        verified_histogram_state = self._failure_curriculum_checkpoint_state()
        state.update(
            {
                "training_algorithm": TRAINING_ALGORITHM,
                "checkpoint_version": CHECKPOINT_VERSION,
                "critic_learning_semantics": CRITIC_LEARNING_SEMANTICS,
                "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
                "q_backend_config": {
                    "target_semantics": CRITIC_LEARNING_SEMANTICS,
                    "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
                    "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                    "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                },
                "replay_mix_state": replay_mix_state,
                "perception_replay_mode": str(self.cfg.perception_replay_mode),
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "teacher_value_return_semantics": str(
                    self.cfg.teacher_value_return_semantics
                ),
                "teacher_value_boundary_semantics": str(
                    self.cfg.teacher_value_boundary_semantics
                ),
                "teacher_value_gamma": float(self.cfg.gamma),
                "teacher_value_reward_group_fingerprint": reward_fingerprint,
                "teacher_value_vecnorm_fingerprint": vecnorm_fingerprint,
                "replay_task_fingerprint": task_fingerprint,
                "verified_teacher_value_histogram_state": (verified_histogram_state),
                "fresh_ring_resume_semantics": FRESH_RING_RESUME_SEMANTICS,
                "teacher_value_bottleneck_replay_state": (
                    self._bottleneck_replay_checkpoint_state()
                ),
                "frozen_teacher_state": {
                    name: copy.deepcopy(getattr(self, name).state_dict())
                    for name in self._frozen_teacher_module_names()
                },
                # Compatibility alias. The semantics sentinel prevents a v3
                # mixed-origin histogram from being mistaken for v4 verified data.
                "failure_phase_curriculum_state": verified_histogram_state,
                "num_updates": int(self.num_updates),
                "sac_actor_update_count": int(
                    getattr(
                        self,
                        "sac_actor_update_count",
                        getattr(self, "actor_update_count", 0),
                    )
                ),
                "sac_alpha_update_count": int(
                    getattr(
                        self,
                        "sac_alpha_update_count",
                        getattr(self, "alpha_update_count", 0),
                    )
                ),
                "last_tvkd_diagnostics": copy.deepcopy(self._last_tvkd_diagnostics),
            }
        )
        return state

    def _load_fastsac_checkpoint_state(self, state, *, load_modules=True):
        algorithm = state.get("training_algorithm")
        version = state.get("checkpoint_version", -1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("TVKD FastSAC checkpoint version is invalid")
        legacy = (
            algorithm == LEGACY_TRAINING_ALGORITHM
            and version == LEGACY_CHECKPOINT_VERSION
        )
        previous = (
            algorithm == PREVIOUS_TRAINING_ALGORITHM
            and version == PREVIOUS_CHECKPOINT_VERSION
        )
        v3 = algorithm == V3_TRAINING_ALGORITHM and version == V3_CHECKPOINT_VERSION
        current = algorithm == TRAINING_ALGORITHM and version == CHECKPOINT_VERSION
        if not (legacy or previous or v3 or current):
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        backend = state.get("dagger_backend_config")
        if not isinstance(backend, Mapping):
            raise ValueError("TVKD checkpoint lacks backend config")
        saved_lambda_bc = _finite_scalar(
            "checkpoint lambda_bc", backend.get("lambda_bc")
        )
        if not math.isclose(
            saved_lambda_bc,
            float(self.cfg.lambda_bc),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("TVKD resume fixed BC coefficient mismatch")
        if legacy or previous or v3:
            # Programmatic loads must perform the same explicit schema
            # migration as the Hydra entrypoint. In particular, v3 perception
            # remains the historical online-Student mode and the saved alpha
            # cadence is preserved rather than inheriting the fresh-v4 actor
            # default.
            _install_v3_replay_migration(
                self.cfg,
                backend,
                student_focus_default=(0.0 if legacy or previous else None),
            )
        if current:
            expected_backend = self._checkpoint_config()
            if set(backend) != set(expected_backend):
                missing = sorted(set(expected_backend).difference(backend))
                unexpected = sorted(set(backend).difference(expected_backend))
                raise ValueError(
                    "TVKD resume algorithm config keys mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for name, expected_value in expected_backend.items():
                if backend[name] != expected_value:
                    raise ValueError(f"TVKD resume config mismatch at {name!r}")

            expected_mix = _checkpoint_replay_mix(self.cfg)
            saved_mix = state.get("replay_mix_state")
            if not isinstance(saved_mix, Mapping):
                raise ValueError("TVKD v4 checkpoint lacks replay mix state")
            normalized_mix: dict[str, dict[str, float]] = {}
            for purpose, expected in expected_mix.items():
                saved_purpose = saved_mix.get(purpose)
                if not isinstance(saved_purpose, Mapping):
                    raise ValueError(f"TVKD v4 checkpoint lacks {purpose!r} replay mix")
                normalized_mix[purpose] = {}
                for source, expected_value in expected.items():
                    saved_value = _finite_scalar(
                        f"checkpoint {purpose} {source} fraction",
                        saved_purpose.get(source),
                    )
                    normalized_mix[purpose][source] = saved_value
                    if not math.isclose(
                        saved_value,
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"TVKD resume replay mix mismatch at {purpose}.{source}"
                        )
                allocate_source_counts(1, normalized_mix[purpose])

            exact_metadata = {
                "critic_learning_semantics": CRITIC_LEARNING_SEMANTICS,
                "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
                "perception_replay_mode": str(self.cfg.perception_replay_mode),
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
                "teacher_value_return_semantics": str(
                    self.cfg.teacher_value_return_semantics
                ),
                "teacher_value_boundary_semantics": str(
                    self.cfg.teacher_value_boundary_semantics
                ),
                "teacher_value_reward_group_fingerprint": str(
                    self.cfg.teacher_value_reward_group_fingerprint
                ),
                "teacher_value_vecnorm_fingerprint": getattr(
                    self, "_replay_vecnorm_fingerprint", None
                ),
                "replay_task_fingerprint": str(self.cfg.replay_task_fingerprint),
                "fresh_ring_resume_semantics": FRESH_RING_RESUME_SEMANTICS,
            }
            _required_contract_fingerprint(
                "runtime Teacher reward-group fingerprint",
                exact_metadata["teacher_value_reward_group_fingerprint"],
            )
            _required_contract_fingerprint(
                "runtime replay task fingerprint",
                exact_metadata["replay_task_fingerprint"],
            )
            for name, expected in exact_metadata.items():
                if not isinstance(expected, str) or not expected:
                    raise ValueError(f"TVKD runtime lacks required metadata {name!r}")
                if state.get(name) != expected:
                    raise ValueError(f"TVKD resume metadata mismatch at {name!r}")
            q_backend = state.get("q_backend_config")
            if not isinstance(q_backend, Mapping):
                raise ValueError("TVKD v4 checkpoint lacks Q backend metadata")
            expected_q_metadata = {
                "target_semantics": CRITIC_LEARNING_SEMANTICS,
                "failure_phase_replay_semantics": VERIFIED_HISTOGRAM_SEMANTICS,
                "bottleneck_location_semantics": BOTTLENECK_LOCATION_SEMANTICS,
                "bottleneck_fallback_mode": str(self.cfg.bottleneck_fallback_mode),
            }
            for name, expected in expected_q_metadata.items():
                if q_backend.get(name) != expected:
                    raise ValueError(
                        f"TVKD resume Q backend metadata mismatch at {name!r}"
                    )
            saved_gamma = _finite_scalar(
                "checkpoint Teacher value gamma",
                state.get("teacher_value_gamma"),
            )
            if not math.isclose(
                saved_gamma,
                float(self.cfg.gamma),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("TVKD resume Teacher value gamma mismatch")
            for name in (
                "sac_alpha_update_cadence",
                "perception_replay_mode",
                "perception_replay_batch_size",
                "use_tvkd_value_shaping",
                "use_teacher_value_bottleneck_replay",
                "bottleneck_threshold",
                "bottleneck_smoothing_window",
                "bottleneck_min_consecutive",
                "bottleneck_terminal_exclusion_steps",
                "bottleneck_residual_scale_ema_decay",
                "bottleneck_eps",
                "bottleneck_fallback_mode",
                "bottleneck_include_unsuccessful_timeouts",
                "max_teacher_phase_match_distance",
            ):
                if backend.get(name) != getattr(self.cfg, name):
                    raise ValueError(f"TVKD resume config mismatch at {name!r}")
        if legacy:
            warnings.warn(
                "Migrating a TVKD v1 adaptive-BC checkpoint to fixed BC and "
                "Teacher-value bottleneck replay. The adaptive BC scheduler "
                "and terminal-lookback failure histogram are ignored.",
                UserWarning,
                stacklevel=2,
            )
        elif previous:
            warnings.warn(
                "Migrating a TVKD v2 checkpoint to v3 Student bottleneck "
                "replay. The raw Student ring is rebuilt and the new focused "
                "Student counters start from zero.",
                UserWarning,
                stacklevel=2,
            )
        elif v3:
            warnings.warn(
                "Migrating a TVKD v3 checkpoint to v4: the saved alpha "
                "cadence and model/optimizer state are retained, while raw "
                "rings, detector statistics, and the mixed-origin histogram "
                "are reset.",
                UserWarning,
                stacklevel=2,
            )
        bottleneck_state = state.get("teacher_value_bottleneck_replay_state")
        if (current or previous or v3) and not isinstance(bottleneck_state, Mapping):
            raise ValueError("TVKD checkpoint lacks bottleneck replay state")
        frozen_teacher_state = state.get("frozen_teacher_state")
        if not isinstance(frozen_teacher_state, Mapping):
            raise ValueError("TVKD checkpoint lacks frozen Teacher value state")
        failure_curriculum_state = (
            state.get("verified_teacher_value_histogram_state")
            if current
            else state.get("failure_phase_curriculum_state")
        )
        if not isinstance(failure_curriculum_state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
        if current:
            compatibility_histogram = state.get("failure_phase_curriculum_state")
            if not isinstance(compatibility_histogram, Mapping):
                raise ValueError(
                    "TVKD v4 checkpoint lacks histogram compatibility state"
                )
            if not _same_verified_histogram_state(
                compatibility_histogram, failure_curriculum_state
            ):
                raise ValueError("TVKD v4 verified histogram aliases are inconsistent")
        optimizer_state = state.get("optimizer_resume_state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("TVKD checkpoint lacks optimizer state")
        if "dr_estimator_optimizer" not in optimizer_state:
            raise ValueError("TVKD checkpoint lacks DR-estimator optimizer state")
        dr_optimizer_state = optimizer_state["dr_estimator_optimizer"]
        if bool(self.cfg.train_dr_estimator):
            if not isinstance(dr_optimizer_state, Mapping):
                raise ValueError("trainable DR estimator lacks optimizer state")
            if getattr(self, "opt_dr_estimator", None) is None:
                raise RuntimeError("trainable DR estimator has no optimizer")
        elif dr_optimizer_state is not None:
            raise ValueError("frozen DR estimator checkpoint has an optimizer state")
        resume_counters = {}
        for name in (
            "num_updates",
            "sac_actor_update_count",
            "sac_alpha_update_count",
        ):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TVKD checkpoint {name} is invalid")
            resume_counters[name] = int(value)

        # A public policy checkpoint contains every PPO/Student/perception
        # child in addition to this algorithm-specific state.  Restore that
        # complete module tree first; restoring only the baseline FastSAC seam
        # would leave perception/EMA children at constructor initialization
        # while loading optimizer moments that refer to different weights.
        if load_modules:
            checkpoint_fingerprint = state.get("vecnorm_fingerprint")
            current_fingerprint = getattr(self, "_replay_vecnorm_fingerprint", None)
            if (
                not isinstance(checkpoint_fingerprint, str)
                or not checkpoint_fingerprint
                or not isinstance(current_fingerprint, str)
                or not current_fingerprint
                or checkpoint_fingerprint != current_fingerprint
            ):
                raise ValueError("TVKD resume checkpoint VecNorm fingerprint mismatch")
            perception_initialization = state.get("perception_initialization")
            if not isinstance(perception_initialization, Mapping):
                raise ValueError(
                    "TVKD resume checkpoint lacks perception initialization state"
                )
            self._perception_initialization = copy.deepcopy(
                dict(perception_initialization)
            )
            self._set_perception_trainable(bool(self.cfg.train_perception))
            missing_children = [
                name for name, _ in self.named_children() if name not in state
            ]
            if missing_children:
                raise ValueError(
                    "TVKD resume requires a full policy state; missing child "
                    f"modules: {missing_children}"
                )
            failed = PPOVEL.load_state_dict(self, state, strict=True)
            if failed:
                raise RuntimeError(
                    "TVKD resume failed to restore policy child modules: "
                    f"{sorted(failed)}"
                )

        # Reuse the baseline's strict optimizer/RNG/counter restoration after
        # translating only its two format sentinels.
        baseline_state = dict(state)
        baseline_state["training_algorithm"] = BASE_FASTSAC_TRAINING_ALGORITHM
        baseline_state["checkpoint_version"] = BASE_FASTSAC_CHECKPOINT_VERSION
        DistributionalFastSACTeacherBC._load_fastsac_checkpoint_state(
            self, baseline_state, load_modules=load_modules
        )
        if bool(self.cfg.train_dr_estimator):
            self.opt_dr_estimator.load_state_dict(dr_optimizer_state)
        self.num_updates = resume_counters["num_updates"]
        self.sac_actor_update_count = resume_counters["sac_actor_update_count"]
        self.sac_alpha_update_count = resume_counters["sac_alpha_update_count"]
        if load_modules:
            for name in self._frozen_teacher_module_names():
                module_state = frozen_teacher_state.get(name)
                if not isinstance(module_state, Mapping):
                    raise ValueError(
                        f"TVKD checkpoint lacks frozen Teacher module {name!r}"
                    )
                getattr(self, name).load_state_dict(module_state, strict=True)
        if legacy or previous or v3:
            # Pre-v4 detector scales and phase anchors were collected under
            # terminal/recent-control gates and could contain argmin or legacy
            # fallbacks. They are never promoted to the verified v4
            # threshold-only curriculum.
            self.teacher_value_bottleneck_detector.bottleneck_residual_scale_ema = 1.0
            self.teacher_value_bottleneck_detector.num_scale_updates = 0
            self._reset_bottleneck_statistics()
            self._reset_failure_curriculum_state()
        else:
            self._load_bottleneck_replay_checkpoint_state(bottleneck_state)
            self._load_failure_curriculum_checkpoint_state(failure_curriculum_state)
        restored_diagnostics = state.get("last_tvkd_diagnostics", {})
        if not isinstance(restored_diagnostics, Mapping):
            restored_diagnostics = {}
        self._last_tvkd_diagnostics = {
            key: copy.deepcopy(value)
            for key, value in restored_diagnostics.items()
            if not str(key).startswith("bc_scheduler/")
        }
        self._reset_student_replay_episode_tracking()
        self._freeze_teacher()
        self.teacher_value_wrapper.freeze()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore a TVKD model for replayless deterministic evaluation."""
        algorithm = state_dict.get("training_algorithm")
        version = state_dict.get("checkpoint_version", -1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("TVKD inference checkpoint version is invalid")
        if (algorithm, version) not in {
            (TRAINING_ALGORITHM, CHECKPOINT_VERSION),
            (V3_TRAINING_ALGORITHM, V3_CHECKPOINT_VERSION),
            (PREVIOUS_TRAINING_ALGORITHM, PREVIOUS_CHECKPOINT_VERSION),
            (LEGACY_TRAINING_ALGORITHM, LEGACY_CHECKPOINT_VERSION),
        }:
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        baseline_state = dict(state_dict)
        baseline_state["training_algorithm"] = BASE_FASTSAC_TRAINING_ALGORITHM
        baseline_state["checkpoint_version"] = BASE_FASTSAC_CHECKPOINT_VERSION
        failed = DistributionalFastSACTeacherBC.load_inference_state_dict(
            self, baseline_state, strict
        )
        frozen_teacher_state = state_dict.get("frozen_teacher_state")
        if not isinstance(frozen_teacher_state, Mapping):
            raise ValueError("TVKD inference checkpoint lacks frozen Teacher state")
        for name in self._frozen_teacher_module_names():
            module_state = frozen_teacher_state.get(name)
            if not isinstance(module_state, Mapping):
                raise ValueError(
                    f"TVKD inference checkpoint lacks Teacher module {name!r}"
                )
            getattr(self, name).load_state_dict(module_state, strict=True)
        restored_diagnostics = state_dict.get("last_tvkd_diagnostics", {})
        self._last_tvkd_diagnostics = (
            {
                key: copy.deepcopy(value)
                for key, value in restored_diagnostics.items()
                if not str(key).startswith("bc_scheduler/")
            }
            if isinstance(restored_diagnostics, Mapping)
            else {}
        )
        self.teacher_value_wrapper.freeze()
        return failed

    def state_dict(self):
        state = DistributionalFastSACTeacherBC.state_dict(self)
        state["replay_resume_semantics"] = (
            "model_optimizer_policy_rng_verified_bottleneck_resume_with_"
            "fresh_raw_ring_and_row_credit_rebuild_v4"
        )
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Fresh training remains the baseline's rigorously validated PPO-source
        # transfer. Same-stage TVKD continuation restores every model,
        # optimizer, RNG, counter, failure curriculum, and bottleneck state,
        # then deliberately rebuilds both raw online replay rings.
        if state_dict.get("training_algorithm") not in {
            TRAINING_ALGORITHM,
            V3_TRAINING_ALGORITHM,
            PREVIOUS_TRAINING_ALGORITHM,
            LEGACY_TRAINING_ALGORITHM,
        }:
            failed = DistributionalFastSACTeacherBC.load_state_dict(
                self, state_dict, strict
            )
            self.teacher_value_wrapper.freeze()
            return failed
        if state_dict.get("actor_backend") != ACTOR_BACKEND:
            raise ValueError("TVKD FastSAC resume actor backend mismatch")
        saved_action_contract = state_dict.get("action_contract")
        if not isinstance(saved_action_contract, Mapping):
            raise ValueError("TVKD FastSAC resume lacks its action contract")
        for key in ("joint_names", "fingerprint"):
            if saved_action_contract.get(key) != self._fastsac_action_contract.get(key):
                raise ValueError(
                    f"TVKD FastSAC resume action contract mismatch at {key!r}"
                )
        self._load_fastsac_checkpoint_state(state_dict, load_modules=True)

        self.dagger_replay.clear()
        self.q_teacher_replay.clear()
        # The restored partial UTD debt belongs to rows in the discarded
        # rings. It must not combine with the first rows of the rebuilt rings.
        self.q_update_row_credit = 0.0
        self._teacher_prefill_complete = False
        self.teacher_prefill_rollout_count = 0
        self.teacher_prefill_environment_steps = 0
        self._teacher_prefill_pending = None
        self._teacher_prefill_successful_episodes = 0
        self._teacher_prefill_failed_episodes = 0
        self._teacher_prefill_timeout_episodes = 0
        self._teacher_prefill_incomplete_episodes = 0
        self._teacher_prefill_discarded_rows = 0
        self._rollout_final_batch = None
        self._truncation_final_batches = []
        self._last_truncation_finals_used = 0
        self._perception_replay_history = None
        self._perception_replay_history_count = 0
        self._failure_phase_history = None
        self._failure_phase_student_history = None
        self._failure_phase_takeover_history = None
        self._bottleneck_episode_histories = None
        self._reset_student_replay_episode_tracking()
        self._teacher_phase_index_ready = False
        self._teacher_phase_bin_rows = ()
        self._teacher_phase_nearest_nonempty = torch.empty(0, dtype=torch.long)
        self._teacher_phase_flat_rows = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_starts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_bin_counts = torch.empty(0, dtype=torch.long)
        self._teacher_phase_device_cache.clear()
        self.actor_adapt.requires_grad_(True).train()
        # ``requires_grad_(True)`` above recursively touches the legacy PPO
        # variance parameter. FastSAC owns variance exclusively through its
        # separate SAC adapter, so immediately restore the inherited invariant.
        self._freeze_legacy_actor_std()
        self.bc_dagger_sac_adapter.requires_grad_(True).train()
        self.qnet.requires_grad_(True).train()
        self.qnet_target.requires_grad_(False).eval()
        self._set_perception_trainable(bool(self.cfg.train_perception))
        if hasattr(self, "env"):
            self.env.set_progress(
                int(state_dict.get("next_iter", state_dict.get("last_iter", -1) + 1))
            )
        self._freeze_teacher()
        self.teacher_value_wrapper.freeze()
        return []


__all__ = [
    "ACTOR_BACKEND",
    "BOTTLENECK_LOCATION_SEMANTICS",
    "CHECKPOINT_VERSION",
    "FRESH_RING_RESUME_SEMANTICS",
    "LEGACY_ADAPTIVE_BC_CONFIG_FIELDS",
    "LEGACY_CHECKPOINT_VERSION",
    "LEGACY_TRAINING_ALGORITHM",
    "PREVIOUS_CHECKPOINT_VERSION",
    "PREVIOUS_TRAINING_ALGORITHM",
    "SOURCE_FAILURE_TEACHER",
    "SOURCE_STUDENT",
    "SOURCE_UNIFORM_TEACHER",
    "TEACHER_VALUE_BOUNDARY_SEMANTICS",
    "TEACHER_VALUE_RETURN_SEMANTICS",
    "TRAINING_ALGORITHM",
    "V3_CHECKPOINT_VERSION",
    "V3_TRAINING_ALGORITHM",
    "VERIFIED_HISTOGRAM_SEMANTICS",
    "FrozenTeacherValueWrapper",
    "TeacherValueBottleneck",
    "TeacherValueBottleneckDetector",
    "TeacherValueTerms",
    "TVKDDistributionalFastSACTeacherBC",
    "TVKDDistributionalFastSACTeacherBCConfig",
    "compute_teacher_value_continuation",
    "compute_teacher_value_terms",
    "_validate_tvkd_algorithm_config",
]
