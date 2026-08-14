"""TVKD value-shaped FastSAC with source-separated adaptive Teacher BC.

This entrypoint is intentionally layered on top of the repository's current
``DistributionalFastSACTeacherBC`` implementation.  It therefore preserves
the existing stochastic Student rollout, frozen successful-Teacher prefill,
50/35/15 Student/uniform-Teacher/failure-phase-Teacher sampling curriculum,
raw recurrent replay, timeout-final-observation handling, twin C51 critics,
and target-update cadence.

The two additions are deliberately narrow:

* the frozen PPO Teacher critic is reused as a potential function in the
  FastSAC C51 target; and
* only freshly collected Student-executed transitions update a global
  Teacher-TD-residual BC scheduler.

The PPO critic consumes the same observation *fields* as the SAC critic, but
its internal ``CatTensors`` module sorts keys.  Consequently this module never
feeds the flattened SAC tensor directly into the PPO MLP.  It reconstructs a
keyed ``TensorDict`` first, preserving the exact PPO observation contract.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import CMD_KEY, OBS_KEY, OBS_PRIV_KEY
from .fastsac_bc_dagger import (
    ACTOR_BACKEND,
    CHECKPOINT_VERSION as BASE_FASTSAC_CHECKPOINT_VERSION,
    TRAINING_ALGORITHM as BASE_FASTSAC_TRAINING_ALGORITHM,
    DistributionalFastSACTeacherBC,
    DistributionalFastSACTeacherBCConfig,
)
from .fastsac_vel import (
    FASTSAC_REFERENCE_EPS,
    _reduce_actor_q_values,
)
from .ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from .ppo_vel import DEPTH_KEY, OBJECT_GEO_KEY, OBJECT_KEY, VEL_CMD_KEY
from .ppo_vel import PPOVEL
from .td3_bc_dagger import (
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    _project_c51_probabilities,
)

TRAINING_ALGORITHM = "distributional_tvkd_fastsac_teacher_bc_v1"
CHECKPOINT_VERSION = 1
EXPECTED_ALGO_NAME = "tvkd_fastsac_bc_dagger"
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger."
    "TVKDDistributionalFastSACTeacherBC"
)
CRITIC_LEARNING_SEMANTICS = (
    "frozen_raw_scale_ppo_value_potential_shaped_soft_c51_target_v1"
)
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_sac_plus_teacher_source_bc_plus_teacher_td_adaptive_"
    "student_source_pretanh_bc_v1"
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


def _validate_tvkd_algorithm_config(cfg) -> None:
    """Validate controls shared by direct construction and the Hydra CLI."""
    if getattr(cfg, "sac_alpha_update_cadence", None) != "critic":
        raise ValueError(
            "TVKD v1 requires sac_alpha_update_cadence='critic' for "
            "resume-compatible temperature updates"
        )
    for name in ("use_tvkd_value_shaping", "use_adaptive_student_bc"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be boolean")

    tvkd_lambda = _finite_scalar("tvkd_lambda", getattr(cfg, "tvkd_lambda"))
    if tvkd_lambda < 0.0:
        raise ValueError("tvkd_lambda must be non-negative")
    potential_clip = getattr(cfg, "tvkd_potential_clip")
    if potential_clip is not None:
        _finite_scalar("tvkd_potential_clip", potential_clip, positive=True)

    lambda_min = _finite_scalar(
        "student_bc_lambda_min", getattr(cfg, "student_bc_lambda_min")
    )
    lambda_max = _finite_scalar(
        "student_bc_lambda_max", getattr(cfg, "student_bc_lambda_max")
    )
    if lambda_min < 0.0 or lambda_max < lambda_min:
        raise ValueError(
            "student BC coefficients must satisfy 0 <= lambda_min <= lambda_max"
        )
    _finite_scalar("student_bc_margin", getattr(cfg, "student_bc_margin"))
    _finite_scalar(
        "student_bc_temperature",
        getattr(cfg, "student_bc_temperature"),
        positive=True,
    )
    for name in (
        "student_bc_scale_ema_decay",
        "student_bc_risk_ema_decay",
    ):
        decay = _finite_scalar(name, getattr(cfg, name))
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    warmup = getattr(cfg, "student_bc_scheduler_warmup_updates")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            "student_bc_scheduler_warmup_updates must be a non-negative integer"
        )
    minimum = getattr(cfg, "student_bc_scheduler_min_samples")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("student_bc_scheduler_min_samples must be a positive integer")
    _finite_scalar(
        "student_bc_scheduler_eps",
        getattr(cfg, "student_bc_scheduler_eps"),
        positive=True,
    )


@dataclass
class TVKDDistributionalFastSACTeacherBCConfig(DistributionalFastSACTeacherBCConfig):
    """Hydra surface for frozen-value TVKD shaping and adaptive Student BC."""

    _target_: str = EXPECTED_ALGO_TARGET
    name: str = EXPECTED_ALGO_NAME
    # Preserve the v1 TVKD optimizer timescale for exact same-stage resume.
    sac_alpha_update_cadence: str = "critic"

    use_tvkd_value_shaping: bool = True
    tvkd_lambda: float = 0.25
    tvkd_potential_clip: float | None = None

    use_adaptive_student_bc: bool = True
    student_bc_lambda_min: float = 0.05
    student_bc_lambda_max: float = 1.0
    student_bc_margin: float = 0.0
    student_bc_temperature: float = 1.0
    student_bc_scale_ema_decay: float = 0.99
    student_bc_risk_ema_decay: float = 0.99
    student_bc_scheduler_warmup_updates: int = 1000
    student_bc_scheduler_min_samples: int = 32
    student_bc_scheduler_eps: float = 1e-6


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


class TeacherValueBCScheduler:
    """EMA-normalized Teacher-TD risk scheduler for global Student BC."""

    def __init__(
        self,
        lambda_min: float,
        lambda_max: float,
        margin: float,
        temperature: float,
        scale_ema_decay: float,
        risk_ema_decay: float,
        warmup_updates: int,
        min_student_samples: int,
        eps: float = 1e-6,
    ) -> None:
        self.lambda_min = _finite_scalar("lambda_min", lambda_min)
        self.lambda_max = _finite_scalar("lambda_max", lambda_max)
        if self.lambda_min < 0.0 or self.lambda_max < self.lambda_min:
            raise ValueError("BC scheduler requires 0 <= lambda_min <= lambda_max")
        self.margin = _finite_scalar("margin", margin)
        self.temperature = _finite_scalar("temperature", temperature, positive=True)
        self.scale_ema_decay = _finite_scalar("scale_ema_decay", scale_ema_decay)
        self.risk_ema_decay = _finite_scalar("risk_ema_decay", risk_ema_decay)
        if not 0.0 <= self.scale_ema_decay < 1.0:
            raise ValueError("scale_ema_decay must lie in [0, 1)")
        if not 0.0 <= self.risk_ema_decay < 1.0:
            raise ValueError("risk_ema_decay must lie in [0, 1)")
        if (
            isinstance(warmup_updates, bool)
            or not isinstance(warmup_updates, int)
            or warmup_updates < 0
        ):
            raise ValueError("warmup_updates must be a non-negative integer")
        if (
            isinstance(min_student_samples, bool)
            or not isinstance(min_student_samples, int)
            or min_student_samples < 1
        ):
            raise ValueError("min_student_samples must be a positive integer")
        self.warmup_updates = int(warmup_updates)
        self.min_student_samples = int(min_student_samples)
        self.eps = _finite_scalar("eps", eps, positive=True)

        self.residual_scale_ema = 1.0
        self.risk_ema = 0.5
        self.num_updates = 0
        self.current_lambda_bc_student = self.lambda_max
        self.last_risk_batch_mean = 0.0
        self.last_normalized_td_residual_mean = 0.0
        self.last_student_sample_count = 0

    def _lambda_from_risk(self) -> float:
        return self.lambda_min + (self.lambda_max - self.lambda_min) * self.risk_ema

    @torch.no_grad()
    def update(self, teacher_td_residual: torch.Tensor) -> float:
        """Update state from one newly collected Student-only population."""
        residual = torch.as_tensor(teacher_td_residual).detach().float().reshape(-1)
        self.last_student_sample_count = int(residual.numel())
        if residual.numel() == 0:
            return float(self.current_lambda_bc_student)
        if not torch.isfinite(residual).all():
            raise RuntimeError("Teacher TD residual contains NaN/Inf")

        # Diagnostics are still current when a very small rollout is skipped,
        # but none of the persistent scheduler state is advanced.
        if residual.numel() < self.min_student_samples:
            normalized = residual / (self.residual_scale_ema + self.eps)
            risk = torch.sigmoid((-normalized - self.margin) / self.temperature)
            self.last_normalized_td_residual_mean = normalized.mean().item()
            self.last_risk_batch_mean = risk.mean().item()
            return float(self.current_lambda_bc_student)

        absolute_mean = residual.abs().mean().item()
        self.residual_scale_ema = (
            self.scale_ema_decay * self.residual_scale_ema
            + (1.0 - self.scale_ema_decay) * absolute_mean
        )
        self.residual_scale_ema = max(self.residual_scale_ema, self.eps)
        normalized = residual / (self.residual_scale_ema + self.eps)
        risk = torch.sigmoid((-normalized - self.margin) / self.temperature)
        risk_batch_mean = risk.mean().item()
        self.risk_ema = (
            self.risk_ema_decay * self.risk_ema
            + (1.0 - self.risk_ema_decay) * risk_batch_mean
        )
        self.num_updates += 1
        self.last_normalized_td_residual_mean = normalized.mean().item()
        self.last_risk_batch_mean = risk_batch_mean
        if self.num_updates <= self.warmup_updates:
            self.current_lambda_bc_student = self.lambda_max
        else:
            self.current_lambda_bc_student = self._lambda_from_risk()
        if not math.isfinite(self.current_lambda_bc_student):
            raise RuntimeError("Adaptive Student BC coefficient became non-finite")
        return float(self.current_lambda_bc_student)

    def state_dict(self) -> dict:
        return {
            "residual_scale_ema": float(self.residual_scale_ema),
            "risk_ema": float(self.risk_ema),
            "num_updates": int(self.num_updates),
            "current_lambda_bc_student": float(self.current_lambda_bc_student),
            "last_risk_batch_mean": float(self.last_risk_batch_mean),
            "last_normalized_td_residual_mean": float(
                self.last_normalized_td_residual_mean
            ),
            "last_student_sample_count": int(self.last_student_sample_count),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError("BC scheduler checkpoint state must be a mapping")
        required = {
            "residual_scale_ema",
            "risk_ema",
            "num_updates",
            "current_lambda_bc_student",
        }
        missing = required.difference(state_dict)
        if missing:
            raise ValueError(
                f"BC scheduler checkpoint is missing fields: {sorted(missing)}"
            )
        scale = _finite_scalar(
            "residual_scale_ema", state_dict["residual_scale_ema"], positive=True
        )
        risk = _finite_scalar("risk_ema", state_dict["risk_ema"])
        current = _finite_scalar(
            "current_lambda_bc_student",
            state_dict["current_lambda_bc_student"],
        )
        updates = state_dict["num_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("BC scheduler num_updates must be non-negative")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("BC scheduler risk_ema must lie in [0, 1]")
        if not self.lambda_min <= current <= self.lambda_max:
            raise ValueError("Restored Student BC coefficient is outside its bounds")
        self.residual_scale_ema = scale
        self.risk_ema = risk
        self.num_updates = int(updates)
        self.current_lambda_bc_student = current
        self.last_risk_batch_mean = _finite_scalar(
            "last_risk_batch_mean", state_dict.get("last_risk_batch_mean", 0.0)
        )
        self.last_normalized_td_residual_mean = _finite_scalar(
            "last_normalized_td_residual_mean",
            state_dict.get("last_normalized_td_residual_mean", 0.0),
        )
        count = state_dict.get("last_student_sample_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("last_student_sample_count must be non-negative")
        self.last_student_sample_count = int(count)


@dataclass(frozen=True)
class TeacherValueTerms:
    teacher_v: torch.Tensor
    teacher_v_next: torch.Tensor
    potential_delta: torch.Tensor
    shaped_reward: torch.Tensor
    teacher_td_residual: torch.Tensor


@torch.no_grad()
def compute_teacher_value_terms(
    get_teacher_value: Callable[[torch.Tensor], torch.Tensor],
    teacher_critic_obs: torch.Tensor,
    next_teacher_critic_obs: torch.Tensor,
    raw_reward: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    effective_discount: torch.Tensor | float,
    *,
    tvkd_lambda: float,
    potential_clip: float | None = None,
) -> TeacherValueTerms:
    """Compute raw-scale frozen-value shaping and raw-reward TD residual."""
    raw_reward = raw_reward.detach().float().reshape(-1)
    bootstrap = bootstrap_mask.detach().float().reshape(-1)
    discount = torch.as_tensor(
        effective_discount, device=raw_reward.device, dtype=torch.float32
    )
    if discount.ndim == 0:
        discount = discount.expand_as(raw_reward)
    else:
        discount = discount.reshape(-1)
    if not (raw_reward.shape == bootstrap.shape == discount.shape):
        raise ValueError("Reward, bootstrap mask, and discount shapes must match")
    if not (
        torch.isfinite(raw_reward).all()
        and torch.isfinite(bootstrap).all()
        and torch.isfinite(discount).all()
    ):
        raise RuntimeError("TVKD reward/mask/discount contains NaN/Inf")

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

    potential_delta = discount * bootstrap * fixed_v_next - fixed_v
    shaped_reward = raw_reward + float(tvkd_lambda) * potential_delta
    # Scheduler residual intentionally uses raw reward and unclipped Teacher V.
    teacher_td_residual = raw_reward + discount * bootstrap * teacher_v_next - teacher_v
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


def _masked_pretanh_bc_loss(
    actor_mean_latent: torch.Tensor,
    teacher_action: torch.Tensor,
    mask: torch.Tensor,
    action_center: torch.Tensor,
    action_half_range: torch.Tensor,
    huber_delta: float,
    eps: float = FASTSAC_REFERENCE_EPS,
) -> torch.Tensor:
    selected = mask.reshape(-1).bool()
    if actor_mean_latent.shape != teacher_action.shape:
        raise ValueError("BC latent and Teacher action shapes must match")
    if actor_mean_latent.ndim != 2 or actor_mean_latent.shape[-1] < 1:
        raise ValueError("BC tensors must have shape [batch, action_dim]")
    if selected.numel() != actor_mean_latent.shape[0]:
        raise ValueError("BC source mask does not match batch rows")
    if not selected.any():
        return actor_mean_latent.sum() * 0.0
    prediction = actor_mean_latent[selected]
    target_action = teacher_action[selected].detach()
    center = action_center.to(prediction)
    half_range = action_half_range.to(prediction)
    if center.shape != prediction.shape[-1:] or half_range.shape != center.shape:
        raise ValueError("BC action coordinates must match the action dimension")
    if not (
        torch.isfinite(prediction).all()
        and torch.isfinite(target_action).all()
        and torch.isfinite(center).all()
        and torch.isfinite(half_range).all()
        and torch.all(half_range > 0.0)
    ):
        raise RuntimeError("Pre-tanh BC inputs contain invalid values")
    normalized_target = ((target_action - center) / half_range).clamp(
        min=-1.0 + float(eps), max=1.0 - float(eps)
    )
    target_latent = torch.atanh(normalized_target)
    return F.smooth_l1_loss(
        prediction,
        target_latent,
        beta=float(huber_delta),
    )


def compute_source_separated_bc_losses(
    actor_mean_latent: torch.Tensor,
    teacher_action: torch.Tensor,
    teacher_action_valid: torch.Tensor,
    source_id: torch.Tensor,
    action_center: torch.Tensor,
    action_half_range: torch.Tensor,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empty-safe conditional Teacher-source and Student-source BC."""
    source = source_id.reshape(-1).long()
    if source.numel() != actor_mean_latent.shape[0]:
        raise ValueError("BC source_id does not match batch rows")
    allowed = (
        (source == SOURCE_STUDENT)
        | (source == SOURCE_UNIFORM_TEACHER)
        | (source == SOURCE_FAILURE_TEACHER)
    )
    if not allowed.all():
        raise ValueError("BC source_id contains an unknown source")
    valid = teacher_action_valid.reshape(-1).bool()
    if valid.numel() != actor_mean_latent.shape[0]:
        raise ValueError("BC Teacher validity mask does not match batch rows")
    teacher_mask = valid & (source != SOURCE_STUDENT)
    student_mask = valid & (source == SOURCE_STUDENT)
    teacher_loss = _masked_pretanh_bc_loss(
        actor_mean_latent,
        teacher_action,
        teacher_mask,
        action_center,
        action_half_range,
        huber_delta,
    )
    student_loss = _masked_pretanh_bc_loss(
        actor_mean_latent,
        teacher_action,
        student_mask,
        action_center,
        action_half_range,
        huber_delta,
    )
    return teacher_loss, student_loss


def _source_ids_from_batch(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Materialize the baseline's sample-time 0/1/2 source classification."""
    valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY]
    teacher = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
    if teacher is None:
        teacher = torch.zeros_like(valid, dtype=torch.bool)
    teacher = teacher.reshape(-1).bool()
    failure = batch.get(FAILURE_PHASE_TEACHER_SOURCE_KEY)
    if failure is None:
        failure = torch.zeros_like(teacher)
    failure = failure.reshape(-1).bool()
    if teacher.shape != failure.shape or teacher.numel() != valid.numel():
        raise ValueError("Teacher/failure source masks must align")
    if (failure & ~teacher).any():
        raise ValueError("Failure-Teacher rows must also be Teacher-source rows")
    source = torch.full_like(teacher, SOURCE_STUDENT, dtype=torch.long)
    source[teacher] = SOURCE_UNIFORM_TEACHER
    source[teacher & failure] = SOURCE_FAILURE_TEACHER
    return source


class TVKDDistributionalFastSACTeacherBC(DistributionalFastSACTeacherBC):
    """Twin-C51 FastSAC with frozen PPO-value shaping and adaptive Student BC."""

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
        self.student_bc_scheduler = TeacherValueBCScheduler(
            lambda_min=float(cfg.student_bc_lambda_min),
            lambda_max=float(cfg.student_bc_lambda_max),
            margin=float(cfg.student_bc_margin),
            temperature=float(cfg.student_bc_temperature),
            scale_ema_decay=float(cfg.student_bc_scale_ema_decay),
            risk_ema_decay=float(cfg.student_bc_risk_ema_decay),
            warmup_updates=int(cfg.student_bc_scheduler_warmup_updates),
            min_student_samples=int(cfg.student_bc_scheduler_min_samples),
            eps=float(cfg.student_bc_scheduler_eps),
        )
        self._last_bc_scheduler_metrics = self._empty_scheduler_metrics()
        self._last_tvkd_diagnostics: dict[str, float] = {}

    def get_frozen_teacher_value(
        self, teacher_critic_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.teacher_value_wrapper.get_frozen_teacher_value(teacher_critic_obs)

    def _tvkd_enabled(self) -> bool:
        return (
            bool(self.cfg.use_tvkd_value_shaping) and float(self.cfg.tvkd_lambda) != 0.0
        )

    def _adaptive_bc_changes_actor_loss(self) -> bool:
        if not bool(self.cfg.use_adaptive_student_bc):
            return False
        # A fixed coefficient equal to the baseline coefficient is an explicit
        # numerical-equivalence mode requested by the algorithm contract.
        baseline = float(self.cfg.lambda_bc)
        return not (
            math.isclose(
                float(self.cfg.student_bc_lambda_min),
                baseline,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(self.cfg.student_bc_lambda_max),
                baseline,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    @torch.no_grad()
    def _distributional_fastsac_target(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        # Preserve the baseline RNG sequence and target bit-for-bit when TVKD
        # is disabled (including the explicit lambda=0 equivalence setting).
        if not self._tvkd_enabled():
            return DistributionalFastSACTeacherBC._distributional_fastsac_target(
                self, batch
            )

        next_dist = self._actor_dist_from_flat(batch["next_observations"])
        next_action, next_raw_log_prob = next_dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        next_log_prob = self._normalized_action_log_prob(next_raw_log_prob)
        bootstrap = (batch["truncations"].bool() | ~batch["dones"].bool()).float()
        effective_discount = float(self.cfg.gamma) * batch["discounts"]
        teacher_discount = float(self.cfg.gamma)
        terms = compute_teacher_value_terms(
            self.get_frozen_teacher_value,
            batch["critic_observations"],
            batch["next_critic_observations"],
            batch["rewards"],
            bootstrap,
            teacher_discount,
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
        left_fraction = soft_reward.new_zeros(())
        right_fraction = soft_reward.new_zeros(())
        for head_probability in target_probabilities:
            projected, left_fraction, right_fraction = _project_c51_probabilities(
                head_probability,
                soft_reward,
                bootstrap,
                effective_discount,
                self.qnet_target.support,
            )
            projected_heads.append(projected)
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
            "left_support_projection_clipping_fraction": left_fraction,
            "right_support_projection_clipping_fraction": right_fraction,
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

    def _actor_update(self, batch: dict[str, torch.Tensor]):
        """SAC plus fixed Teacher-source BC and adaptive Student-source BC."""
        # Do not even materialize the new source masks or pre-tanh targets in
        # either explicit equivalence mode.  Delegating at the method boundary
        # preserves the baseline loss, parameter update, validation surface,
        # and RNG sequence exactly.
        if not self._adaptive_bc_changes_actor_loss():
            return DistributionalFastSACTeacherBC._actor_update(self, batch)

        raw_prediction = self._actor_mean_from_flat(batch["observations"])
        if not torch.isfinite(raw_prediction).all():
            raise RuntimeError("TVKD FastSAC Actor mean contains NaN/Inf")
        dist = self._sac_dist_from_mean(raw_prediction)
        prediction_action = dist.mean
        sampled_action, raw_log_prob = dist.rsample_with_log_prob(
            generator=self.sac_action_rng
        )
        normalized_log_prob = self._normalized_action_log_prob(raw_log_prob)
        q_action = self._q_action_input(sampled_action)

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

        source_id = _source_ids_from_batch(batch)
        teacher_bc_loss, student_bc_loss = compute_source_separated_bc_losses(
            dist.loc,
            batch[DAGGER_REPLAY_TEACHER_ACTIONS],
            batch[DAGGER_TEACHER_ACTION_VALID_KEY],
            source_id,
            self._fastsac_actor_action_center,
            self._fastsac_actor_action_scale,
            float(self.cfg.dagger_actor_huber_delta),
        )
        weighted_sac = float(self.cfg.eta_sac) * sac_actor_loss
        student_lambda = (
            float(self.student_bc_scheduler.current_lambda_bc_student)
            if bool(self.cfg.use_adaptive_student_bc)
            else float(self.cfg.lambda_bc)
        )
        weighted_teacher_bc = float(self.cfg.lambda_bc) * teacher_bc_loss
        weighted_student_bc = student_lambda * student_bc_loss
        exact_bc_loss = teacher_bc_loss + student_bc_loss
        weighted_bc = weighted_teacher_bc + weighted_student_bc
        total_actor_loss = weighted_sac + weighted_bc
        if not torch.isfinite(total_actor_loss):
            raise RuntimeError("TVKD FastSAC Actor loss contains NaN/Inf")
        total_actor_loss.backward()
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
        self.actor_update_count = int(getattr(self, "actor_update_count", 0)) + 1
        self.sac_actor_update_count = (
            int(getattr(self, "sac_actor_update_count", 0)) + 1
        )

        teacher_source = source_id != SOURCE_STUDENT
        failure_source = source_id == SOURCE_FAILURE_TEACHER
        uniform_source = source_id == SOURCE_UNIFORM_TEACHER
        metrics = {
            "td3_actor_loss": sac_actor_loss.detach(),
            "weighted_td3_actor_loss": weighted_sac.detach(),
            "sac_actor_loss": sac_actor_loss.detach(),
            "exact_bc_loss": exact_bc_loss.detach(),
            "bc_teacher_loss": teacher_bc_loss.detach(),
            "bc_student_loss": student_bc_loss.detach(),
            "weighted_teacher_bc_loss": weighted_teacher_bc.detach(),
            "weighted_student_bc_loss": weighted_student_bc.detach(),
            "student_bc_lambda": prediction_action.new_tensor(student_lambda),
            "weighted_sac_actor_loss": weighted_sac.detach(),
            "weighted_bc_loss": weighted_bc.detach(),
            "total_actor_loss": total_actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad).detach(),
            "actor_expected_q1_mean": twin_expected[0].detach().mean(),
            "actor_expected_q2_mean": twin_expected[1].detach().mean(),
            "actor_min_expected_q_mean": minimum_expected.detach().mean(),
            "actor_expected_q_min_mean": minimum_expected.detach().mean(),
            "actor_log_prob_mean": normalized_log_prob.detach().mean(),
            "actor_entropy": -normalized_log_prob.detach().mean(),
            "actor_sample_action_abs_mean": sampled_action.detach().abs().mean(),
            "actor_mean_action_abs_mean": dist.mean.detach().abs().mean(),
            "actor_log_std_mean": self.bc_dagger_sac_adapter.log_std.detach().mean(),
            "actor_teacher_replay_fraction": teacher_source.float().mean(),
            "actor_failure_phase_teacher_fraction": failure_source.float().mean(),
            "actor_student_source_count": (~teacher_source).sum().float(),
            "actor_uniform_teacher_source_count": uniform_source.sum().float(),
            "actor_failure_teacher_source_count": failure_source.sum().float(),
            "alpha": self.log_alpha.exp().detach(),
        }
        if hasattr(self, "_fastsac_rollout_actor_metrics"):
            self._fastsac_rollout_actor_metrics.append(metrics)
        return metrics

    def _empty_scheduler_metrics(self, student_count: int = 0) -> dict[str, float]:
        scheduler = getattr(self, "student_bc_scheduler", None)
        current_lambda = (
            float(self.cfg.lambda_bc)
            if scheduler is None or not bool(self.cfg.use_adaptive_student_bc)
            else float(scheduler.current_lambda_bc_student)
        )
        return {
            "teacher_td_residual_mean": 0.0,
            "teacher_td_residual_std": 0.0,
            "teacher_td_residual_min": 0.0,
            "teacher_td_residual_max": 0.0,
            "normalized_td_residual_mean": 0.0,
            "recent_student_sample_count": float(student_count),
            "residual_scale_ema": (
                1.0 if scheduler is None else float(scheduler.residual_scale_ema)
            ),
            "risk_batch_mean": 0.0,
            "risk_ema": 0.5 if scheduler is None else float(scheduler.risk_ema),
            "student_bc_lambda": current_lambda,
            "fraction_td_residual_negative": 0.0,
            "fraction_normalized_below_negative_margin": 0.0,
        }

    @torch.no_grad()
    def _update_scheduler_from_recent_student(
        self, chunks: list[dict[str, torch.Tensor]]
    ) -> None:
        if not chunks:
            self._last_bc_scheduler_metrics = self._empty_scheduler_metrics()
            return
        raw_current = torch.cat(
            [chunk["critic_observations"] for chunk in chunks], dim=0
        )
        raw_next = torch.cat(
            [chunk["next_critic_observations"] for chunk in chunks], dim=0
        )
        reward = torch.cat([chunk["rewards"] for chunk in chunks], dim=0).float()
        done = torch.cat([chunk["dones"] for chunk in chunks], dim=0).bool()
        truncation = torch.cat([chunk["truncations"] for chunk in chunks], dim=0).bool()
        student_count = int(reward.numel())
        if not bool(self.cfg.use_adaptive_student_bc):
            self._last_bc_scheduler_metrics = self._empty_scheduler_metrics(
                student_count
            )
            return

        snapshot = self._vecnorm_snapshot()
        current = self._normalize_replay_flat(
            raw_current,
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        next_observation = self._normalize_replay_flat(
            raw_next,
            self.q_critic_keys,
            self._q_critic_widths,
            snapshot,
        )
        bootstrap = (truncation | ~done).float()
        # The TVKD/Teacher-residual contract uses gamma*m exactly. The
        # environment's auxiliary soft-contact discount remains part of the
        # unchanged SAC continuation target, but is not folded into Teacher
        # potential shaping or scheduler risk.
        teacher_discount = float(self.cfg.gamma)
        teacher_v = self.get_frozen_teacher_value(current)
        teacher_v_next = self.get_frozen_teacher_value(next_observation)
        residual = (
            reward + teacher_discount * bootstrap * teacher_v_next - teacher_v
        ).float()
        if not torch.isfinite(residual).all():
            raise RuntimeError("Recent Student Teacher TD residual contains NaN/Inf")
        student_lambda = self.student_bc_scheduler.update(residual)
        normalized = residual / (
            self.student_bc_scheduler.residual_scale_ema
            + float(self.cfg.student_bc_scheduler_eps)
        )
        margin = float(self.cfg.student_bc_margin)
        self._last_bc_scheduler_metrics = {
            "teacher_td_residual_mean": residual.mean().item(),
            "teacher_td_residual_std": residual.std(unbiased=False).item(),
            "teacher_td_residual_min": residual.min().item(),
            "teacher_td_residual_max": residual.max().item(),
            "normalized_td_residual_mean": normalized.mean().item(),
            "recent_student_sample_count": float(student_count),
            "residual_scale_ema": float(self.student_bc_scheduler.residual_scale_ema),
            "risk_batch_mean": float(self.student_bc_scheduler.last_risk_batch_mean),
            "risk_ema": float(self.student_bc_scheduler.risk_ema),
            "student_bc_lambda": float(student_lambda),
            "fraction_td_residual_negative": (residual < 0.0).float().mean().item(),
            "fraction_normalized_below_negative_margin": (normalized < -margin)
            .float()
            .mean()
            .item(),
        }

    def _dagger_transition_chunks(self, td: TensorDict):
        """Capture only this rollout's Student rows after timeout correction."""
        # Invalid-label fallback rows during the deterministic Teacher prefill
        # are not generated by the current Student policy and must never
        # advance the scheduler warm-up/EMA state.
        collect_recent_student = not self._teacher_prefill_active()
        recent_student: list[dict[str, torch.Tensor]] = []
        for transitions in super()._dagger_transition_chunks(td):
            is_student = transitions[DAGGER_IS_STUDENT_ACTION_KEY].reshape(-1).bool()
            if collect_recent_student and is_student.any():
                indices = is_student.nonzero(as_tuple=False).squeeze(-1)
                recent_student.append(
                    {
                        name: transitions[name].index_select(0, indices).detach()
                        for name in (
                            "critic_observations",
                            "next_critic_observations",
                            "rewards",
                            "dones",
                            "truncations",
                        )
                    }
                )
            yield transitions
        self._update_scheduler_from_recent_student(recent_student)

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
        for name, value in self._last_bc_scheduler_metrics.items():
            info[f"bc_scheduler/{name}"] = float(value)

        info["loss/critic"] = self._mean_optional_metric(critic_metrics, "critic_loss")
        info["loss/actor_total"] = self._mean_optional_metric(
            actor_metrics, "total_actor_loss"
        )
        info["loss/actor_sac"] = self._mean_optional_metric(
            actor_metrics, "sac_actor_loss"
        )
        info["loss/bc_teacher"] = self._mean_optional_metric(
            actor_metrics, "bc_teacher_loss"
        )
        info["loss/bc_student"] = self._mean_optional_metric(
            actor_metrics, "bc_student_loss"
        )
        info["loss/alpha"] = self._mean_optional_metric(critic_metrics, "alpha_loss")
        # These are sample-time Actor curriculum populations averaged over
        # Actor updates, not rollout transition counts.  Keep the names
        # explicit because failure-focused Teacher is a sampling category,
        # not persistent replay provenance.
        info["source/actor_batch_student_sample_count_mean"] = (
            self._mean_optional_metric(actor_metrics, "actor_student_source_count")
        )
        info["source/actor_batch_uniform_teacher_sample_count_mean"] = (
            self._mean_optional_metric(
                actor_metrics, "actor_uniform_teacher_source_count"
            )
        )
        info["source/actor_batch_failure_teacher_sample_count_mean"] = (
            self._mean_optional_metric(
                actor_metrics, "actor_failure_teacher_source_count"
            )
        )
        info["source/student_transition_count"] = float(
            info.get("fastsac/student_replay_rows_this_rollout", 0.0)
        )
        info["source/student_action_execution_ratio"] = float(
            info.get("fastsac/student_source_fraction", 0.0)
        )
        info["source/teacher_action_execution_ratio"] = float(
            info.get("fastsac/teacher_source_fraction", 0.0)
        )
        info["tvkd/method_distributional_tvkd_fastsac_teacher_bc_v1"] = 1.0
        self._last_tvkd_diagnostics = {
            key: float(value)
            for key, value in info.items()
            if (
                key.startswith("tvkd/")
                or key.startswith("bc_scheduler/")
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
                    "value_norm",
                )
            }
        )
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "teacher_value_semantics": CRITIC_LEARNING_SEMANTICS,
                "bc_loss": ("teacher_and_student_source_conditional_pretanh_smooth_l1"),
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
        return {
            "histogram": histogram.detach()
            .to(device="cpu", dtype=torch.float64)
            .clone(),
            "episode_count": int(getattr(self, "_failure_phase_episode_count", 0)),
            "anchor_count": int(getattr(self, "_failure_phase_anchor_count", 0)),
            "uniform_fallback_rows": int(
                getattr(self, "_failure_phase_uniform_fallback_rows", 0)
            ),
            "focused_rows": int(getattr(self, "_failure_phase_focused_rows", 0)),
        }

    def _load_failure_curriculum_checkpoint_state(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
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
        self._failure_phase_histogram = histogram.detach().to(device="cpu").clone()
        self._failure_phase_episode_count = counters["episode_count"]
        self._failure_phase_anchor_count = counters["anchor_count"]
        self._failure_phase_uniform_fallback_rows = counters["uniform_fallback_rows"]
        self._failure_phase_focused_rows = counters["focused_rows"]
        self._failure_histogram_device_cache.clear()

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
        state.update(
            {
                "training_algorithm": TRAINING_ALGORITHM,
                "checkpoint_version": CHECKPOINT_VERSION,
                "critic_learning_semantics": CRITIC_LEARNING_SEMANTICS,
                "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
                "teacher_value_bc_scheduler": (self.student_bc_scheduler.state_dict()),
                "frozen_teacher_state": {
                    name: copy.deepcopy(getattr(self, name).state_dict())
                    for name in self._frozen_teacher_module_names()
                },
                "failure_phase_curriculum_state": (
                    self._failure_curriculum_checkpoint_state()
                ),
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
        if state.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("TVKD FastSAC checkpoint version mismatch")
        scheduler_state = state.get("teacher_value_bc_scheduler")
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("TVKD checkpoint lacks BC scheduler state")
        frozen_teacher_state = state.get("frozen_teacher_state")
        if not isinstance(frozen_teacher_state, Mapping):
            raise ValueError("TVKD checkpoint lacks frozen Teacher value state")
        failure_curriculum_state = state.get("failure_phase_curriculum_state")
        if not isinstance(failure_curriculum_state, Mapping):
            raise ValueError("TVKD checkpoint lacks failure curriculum state")
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
        self.student_bc_scheduler.load_state_dict(dict(scheduler_state))
        self._load_failure_curriculum_checkpoint_state(failure_curriculum_state)
        self._last_tvkd_diagnostics = copy.deepcopy(
            state.get("last_tvkd_diagnostics", {})
        )
        self._freeze_teacher()
        self.teacher_value_wrapper.freeze()

    def load_inference_state_dict(self, state_dict, strict=True):
        """Restore a TVKD model for replayless deterministic evaluation."""
        if state_dict.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a TVKD FastSAC Teacher-BC checkpoint")
        if int(state_dict.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("TVKD FastSAC checkpoint version mismatch")
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
        scheduler_state = state_dict.get("teacher_value_bc_scheduler")
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("TVKD inference checkpoint lacks scheduler state")
        self.student_bc_scheduler.load_state_dict(dict(scheduler_state))
        self._last_tvkd_diagnostics = copy.deepcopy(
            state_dict.get("last_tvkd_diagnostics", {})
        )
        self.teacher_value_wrapper.freeze()
        return failed

    def state_dict(self):
        state = DistributionalFastSACTeacherBC.state_dict(self)
        state["replay_resume_semantics"] = (
            "model_optimizer_policy_rng_scheduler_resume_with_fresh_raw_ring_rebuild_v1"
        )
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Fresh training remains the baseline's rigorously validated PPO-source
        # transfer. Same-stage TVKD continuation restores every model,
        # optimizer, RNG, counter, failure curriculum, and scheduler state,
        # then deliberately rebuilds both raw online replay rings.
        if state_dict.get("training_algorithm") != TRAINING_ALGORITHM:
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
    "CHECKPOINT_VERSION",
    "SOURCE_FAILURE_TEACHER",
    "SOURCE_STUDENT",
    "SOURCE_UNIFORM_TEACHER",
    "TRAINING_ALGORITHM",
    "FrozenTeacherValueWrapper",
    "TeacherValueBCScheduler",
    "TeacherValueTerms",
    "TVKDDistributionalFastSACTeacherBC",
    "TVKDDistributionalFastSACTeacherBCConfig",
    "compute_source_separated_bc_losses",
    "compute_teacher_value_terms",
    "_validate_tvkd_algorithm_config",
]
