"""Distributional FastSAC with exact mean-action Teacher BC.

This backend deliberately reuses the raw recurrent replay, Teacher-only
prefill, DAgger source selection, timeout handling, and twin-C51 topology from
``td3_bc_dagger``.  The learning rule itself is SAC:

* Student collection and Actor updates use a reparameterized bounded Gaussian
  whose standard deviation is expressed in nominal joint coordinates;
* exact Teacher BC is applied only to the bounded noise-free mean;
* the soft Bellman target contains the next-policy entropy term;
* both online critics learn from the complete target-C51 head with the lower
  expected return; and
* there is a target critic but no target Actor or TD3 smoothing noise.

Replay perception remains input-authoritative.  Collection-time ``priv_pred``
and recurrent hidden states are never treated as durable replay features, and
the duplicate Teacher H5 export remains disabled.
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
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    PERCEPTION_PREFILL_WARMUP_SEMANTICS,
    PERCEPTION_REPLAY_SEMANTICS,
    TEACHER_PREFILL_SEMANTICS,
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
CHECKPOINT_VERSION = 4
ACTOR_BACKEND = "ppo_vel_smooth_bounded_normalized_std_tanh_fastsac_bc_v2"
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
ACTOR_LEARNING_SEMANTICS = (
    "reparameterized_normalized_std_bounded_alpha_logpi_minus_online_twin_min_plus_"
    "joint_normalized_raw_teacher_mean_bc_v1"
)
ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_delayed_actor_cadence_v3"
)
_CRITIC_CADENCE_ENTROPY_SEMANTICS = (
    "smooth_bounded_log_std_tanh_normal_nominal_joint_coordinate_log_probability_"
    "auto_temperature_v2"
)


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

    eta_sac: float = 1e-4
    lambda_bc: float = 1.0
    sac_actor_lr: float = 3e-4
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
        raw_student_mean = owner._student_raw_action_proposal(td)
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

        teacher_action = owner._teacher_action(td.clone(False))
        valid = _valid_raw_action_rows(teacher_action)
        finite_teacher_action = torch.nan_to_num(
            teacher_action, nan=0.0, posinf=0.0, neginf=0.0
        )
        bounded_teacher_action = owner._project_execution_action(finite_teacher_action)
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
            # Prefill is a deterministic Teacher data phase.  Do not advance
            # either SAC sampling stream on its invalid-label fallback rows.
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

        issued_action = torch.where(
            choose_teacher.unsqueeze(-1),
            bounded_teacher_action,
            sampled_student_action,
        )
        issued_action = owner._project_execution_action(issued_action)
        sample_q_deviation = owner._q_action_input(sampled_student_action) - (
            owner._q_action_input(mean_student_action)
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
        td[TD3_BETA_KEY] = torch.full_like(discrepancy_rms, float(scheduled_beta))
        return td


class _DeterministicFastSACStudentEvalPolicy(_DeterministicTD3StudentEvalPolicy):
    """Bounded deterministic Student mean; never samples or computes log-prob."""


class DistributionalFastSACTeacherBC(DistributionalTD3TeacherBC):
    """Twin-C51 FastSAC with stochastic SAC and exact mean-only Teacher BC."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        # SAC owns no target Actor.  The TD3 base initializes this to None; keep
        # that explicit invariant for checkpoint and runtime inspection.
        self.actor_target = None
        self.actor_backend = ACTOR_BACKEND

        self._fastsac_entropy_reference_log_scale_sum = float(
            torch.log(self._fastsac_q_action_scale).sum().item()
        )
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
        self._freeze_legacy_actor_std()

        actor_mean_parameters = tuple(
            parameter
            for parameter in self.actor_adapt.parameters()
            if parameter.requires_grad
        )
        if not actor_mean_parameters:
            raise RuntimeError("FastSAC Actor has no trainable mean parameters")
        # The adapter stores the unconstrained pre-tanh log-std coordinate.
        # Keep that single parameter unregularized so weight decay cannot pull
        # the effective standard deviation toward the interval midpoint.
        self.actor_optimizer = torch.optim.AdamW(
            (
                {
                    "params": actor_mean_parameters,
                    "lr": float(cfg.sac_actor_lr),
                    "weight_decay": float(cfg.q_weight_decay),
                },
                {
                    "params": tuple(self.bc_dagger_sac_adapter.parameters()),
                    "lr": float(cfg.sac_actor_lr),
                    "weight_decay": 0.0,
                },
            ),
            lr=float(cfg.sac_actor_lr),
            betas=(0.9, 0.95),
        )
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
        generator_device = torch.device(device)
        self.sac_action_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 1
        )
        self.sac_rollout_rng = torch.Generator(device=generator_device).manual_seed(
            int(cfg.q_seed) + 2
        )
        self.alpha_update_count = 0
        self._last_fastsac_diagnostics: dict[str, float] = {}

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
        if not isinstance(cfg.sac_use_autotune, bool):
            raise ValueError("sac_use_autotune must be a boolean")
        if str(cfg.sac_alpha_update_cadence) not in ("actor", "critic"):
            raise ValueError(
                "sac_alpha_update_cadence must be 'actor' or 'critic'"
            )
        if not (
            math.isfinite(float(cfg.sac_log_std_min))
            and math.isfinite(float(cfg.sac_log_std_max))
            and float(cfg.sac_log_std_min) < float(cfg.sac_log_std_max)
        ):
            raise ValueError("FastSAC log-std bounds must be finite and ordered")
        for name in (
            "sac_actor_lr",
            "sac_initial_action_std",
            "sac_alpha_init",
            "sac_alpha_lr",
            "sac_tau",
            "sac_max_grad_norm",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("eta_sac", "lambda_bc", "sac_target_entropy_ratio"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < float(cfg.sac_target_entropy_ratio) <= 1.0:
            raise ValueError("sac_target_entropy_ratio must lie in (0,1]")
        max_unsquashed_entropy_per_dim = (
            0.5 * math.log(2.0 * math.pi * math.e)
            + float(cfg.sac_log_std_max)
        )
        target_entropy_per_dim = -float(cfg.sac_target_entropy_ratio)
        if max_unsquashed_entropy_per_dim <= target_entropy_per_dim:
            raise ValueError(
                "FastSAC entropy target is unreachable: sac_log_std_max must be "
                "greater than -sac_target_entropy_ratio - 0.5*log(2*pi*e)"
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
        for name in (
            "teacher_actor_replay_fraction",
            "teacher_perception_replay_fraction",
            "q_teacher_replay_ratio",
        ):
            if not math.isclose(
                float(getattr(cfg, name)), 0.5, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "FastSAC requires exact 50/50 frozen-Teacher/online-Student "
                    f"training sources; {name} must equal 0.5"
                )
        if int(cfg.dagger_batch_size) % 2:
            raise ValueError(
                "dagger_batch_size must be even for exact 50/50 Actor replay"
            )
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

    def _freeze_legacy_actor_std(self) -> None:
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
        core.actor_std.requires_grad_(False)
        core.actor_std.grad = None

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
        if not torch.isfinite(log_std).all() or (
            (log_std <= float(log_std_min))
            | (log_std >= float(log_std_max))
        ).any():
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

    def _bounded_log_std(
        self, raw_log_std: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Smoothly map the adapter parameter into configured log-std bounds."""
        if raw_log_std is None:
            raw_log_std = self.bc_dagger_sac_adapter.log_std
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )

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
        effective = legacy_log_std.clamp(
            legacy_log_std_min, legacy_log_std_max
        )
        log_std_min = float(self.cfg.sac_log_std_min)
        log_std_max = float(self.cfg.sac_log_std_max)
        normalized = (
            2.0
            * (effective - log_std_min)
            / (log_std_max - log_std_min)
            - 1.0
        )
        inward = 4.0 * torch.finfo(normalized.dtype).eps
        return torch.atanh(normalized.clamp(-1.0 + inward, 1.0 - inward))

    def _sac_dist_from_mean(self, mean: torch.Tensor) -> FastSACTanhNormal:
        """Build a bounded policy with std in nominal Q-action coordinates."""
        if not torch.isfinite(mean).all():
            raise RuntimeError("FastSAC Actor proposal contains non-finite raw actions")
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

    def _actor_dist_from_flat(self, actor_obs: torch.Tensor) -> FastSACTanhNormal:
        return self._sac_dist_from_mean(self._actor_mean_from_flat(actor_obs))

    def _normalized_action_log_prob(self, raw_log_prob: torch.Tensor):
        """Convert raw-action density to joint-normalized-coordinate density."""
        return raw_log_prob + float(self._fastsac_entropy_reference_log_scale_sum)

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
        effective_discount = float(self.cfg.gamma) * batch["discounts"]
        alpha = self.log_alpha.exp()
        entropy_tax = effective_discount * bootstrap * alpha * next_log_prob
        soft_reward = batch["rewards"] - entropy_tax

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
            batch["critic_observations"], self._q_action_input(batch["actions"])
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
        # Actor update in the inherited replay loop.  TVKD v1 explicitly keeps
        # its legacy every-Critic cadence for resume compatibility.
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
            int(getattr(self, "alpha_update_count", 0))
            > alpha_update_count_before
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
        """One combined reparameterized SAC plus exact mean-BC Actor step."""
        raw_prediction = self._actor_mean_from_flat(batch["observations"])
        if not torch.isfinite(raw_prediction).all():
            raise RuntimeError("FastSAC Actor mean contains non-finite raw actions")
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

        exact_bc_loss = _exact_teacher_bc_loss(
            prediction_action,
            batch[DAGGER_REPLAY_TEACHER_ACTIONS],
            batch[DAGGER_TEACHER_ACTION_VALID_KEY],
            self._fastsac_q_action_center,
            self._fastsac_q_action_scale,
            float(self.cfg.dagger_actor_huber_delta),
        )
        weighted_sac = float(self.cfg.eta_sac) * sac_actor_loss
        weighted_bc = float(self.cfg.lambda_bc) * exact_bc_loss
        total_actor_loss = weighted_sac + weighted_bc
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
            "actor_expected_q1_mean": twin_expected[0].detach().mean(),
            "actor_expected_q2_mean": twin_expected[1].detach().mean(),
            "actor_min_expected_q_mean": minimum_expected.detach().mean(),
            "actor_expected_q_min_mean": minimum_expected.detach().mean(),
            "actor_log_prob_mean": normalized_log_prob.detach().mean(),
            "actor_entropy": -normalized_log_prob.detach().mean(),
            "actor_sample_action_abs_mean": sampled_action.detach().abs().mean(),
            "actor_mean_action_abs_mean": dist.mean.detach().abs().mean(),
            "actor_log_std_mean": self._bounded_log_std().detach().mean(),
            "actor_raw_log_std_mean": (
                self.bc_dagger_sac_adapter.log_std.detach().mean()
            ),
            "actor_teacher_replay_fraction": (actor_teacher_replay_fraction.detach()),
            "actor_failure_phase_teacher_fraction": batch.get(
                FAILURE_PHASE_TEACHER_SOURCE_KEY,
                torch.zeros_like(batch[DAGGER_TEACHER_ACTION_VALID_KEY]),
            )
            .float()
            .mean()
            .detach(),
            "alpha": self.log_alpha.exp().detach(),
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
        td3_info = DistributionalTD3TeacherBC.train_op(self, tensordict)

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

        critic_keys = (
            "target_sample_action_abs_mean",
            "target_log_prob_mean",
            "entropy_tax_mean",
            "entropy_tax_abs_mean",
            "entropy_tax_reward_abs_ratio",
            "alpha_update_due_fraction",
            "alpha_update_performed_fraction",
        )
        actor_keys = (
            "actor_expected_q1_mean",
            "actor_expected_q2_mean",
            "actor_min_expected_q_mean",
            "actor_log_prob_mean",
            "actor_entropy",
            "actor_sample_action_abs_mean",
            "actor_mean_action_abs_mean",
            "actor_log_std_mean",
            "actor_raw_log_std_mean",
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
        info.update({f"fastsac/{key}": value for key, value in critic.items()})
        info.update({f"fastsac/{key}": value for key, value in actor.items()})
        info["fastsac/alpha_update_count"] = self.alpha_update_count
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
                "actor_target": False,
                "stochastic_actor": True,
                "entropy_semantics": (
                    ENTROPY_SEMANTICS
                    if alpha_update_cadence == "actor"
                    else _CRITIC_CADENCE_ENTROPY_SEMANTICS
                ),
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
                    "sac_actor_lr",
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
        common.update(
            {
                "method": TRAINING_ALGORITHM,
                "actor_output": (
                    "smooth_tanh_bounded_normalized_std_raw_action_normal"
                ),
                "bc_loss": "joint_normalized_raw_mean_teacher_smooth_l1",
            }
        )
        return common

    def _fastsac_checkpoint_state(self):
        alpha_update_cadence = str(self.cfg.sac_alpha_update_cadence)
        return {
            "training_algorithm": TRAINING_ALGORITHM,
            "checkpoint_version": CHECKPOINT_VERSION,
            "actor_backend": ACTOR_BACKEND,
            "critic_learning_semantics": CRITIC_SEMANTICS,
            "actor_learning_semantics": ACTOR_LEARNING_SEMANTICS,
            "entropy_semantics": (
                ENTROPY_SEMANTICS
                if alpha_update_cadence == "actor"
                else _CRITIC_CADENCE_ENTROPY_SEMANTICS
            ),
            "actor_adapt": self.actor_adapt.state_dict(),
            "bc_dagger_sac_adapter": self.bc_dagger_sac_adapter.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "optimizer_resume_state": {
                "actor_optimizer": self.actor_optimizer.state_dict(),
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
            "critic_update_count": int(self.critic_update_count),
            "alpha_update_count": int(self.alpha_update_count),
            "q_update_row_credit": float(
                getattr(self, "q_update_row_credit", 0.0)
            ),
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
            "teacher_perception_rng_state": self.teacher_perception_rng.get_state(),
            "last_fastsac_diagnostics": copy.deepcopy(self._last_fastsac_diagnostics),
        }

    def _load_fastsac_checkpoint_state(self, state, *, load_modules=True):
        if state.get("training_algorithm") != TRAINING_ALGORITHM:
            raise ValueError("not a distributional FastSAC Teacher-BC checkpoint")
        if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        if load_modules:
            for name in (
                "actor_adapt",
                "bc_dagger_sac_adapter",
                "qnet",
                "qnet_target",
            ):
                getattr(self, name).load_state_dict(state[name], strict=True)
            self.log_alpha.data.copy_(state["log_alpha"].to(self.log_alpha))
        optimizers = state.get("optimizer_resume_state")
        if not isinstance(optimizers, dict):
            raise ValueError("FastSAC checkpoint lacks optimizer state")
        self.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
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
            CHECKPOINT_VERSION,
        ):
            raise ValueError("distributional FastSAC checkpoint version mismatch")
        expected_actor_backend = (
            _LEGACY_EFFECTIVE_LOG_STD_ACTOR_BACKEND if legacy_v3 else ACTOR_BACKEND
        )
        if state_dict.get("actor_backend") != expected_actor_backend:
            raise ValueError("distributional FastSAC actor backend mismatch")
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
                    "fresh_only_online_raw_perception_rings_not_serialized_v1"
                ),
                "perception_replay_semantics": PERCEPTION_REPLAY_SEMANTICS,
                "perception_prefill_warmup_semantics": (
                    PERCEPTION_PREFILL_WARMUP_SEMANTICS
                ),
                "teacher_prefill_semantics": TEACHER_PREFILL_SEMANTICS,
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
                "FastSAC raw-perception replay is fresh-only; same-stage resume "
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
        self._freeze_legacy_actor_std()
        self.bc_dagger_sac_adapter.log_std.data.copy_(
            self._fastsac_initial_raw_log_std
        )
        self.log_alpha.data.fill_(math.log(float(self.cfg.sac_alpha_init)))
        self.actor_update_count = 0
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
        self.teacher_perception_rng.manual_seed(int(self.cfg.q_seed) + 3)
        # Drop deterministic-only streams after the base loader is finished.
        if hasattr(self, "collector_exploration_rng"):
            del self.collector_exploration_rng
        if hasattr(self, "target_policy_rng"):
            del self.target_policy_rng
        hard_copy_(self.qnet, self.qnet_target)
        self.qnet_target.requires_grad_(False).eval()
        self.actor_adapt.requires_grad_(True).train()
        self._freeze_legacy_actor_std()
        return failed


__all__ = [
    "ACTION_CONTRACT_SEMANTICS",
    "ACTOR_BACKEND",
    "CHECKPOINT_VERSION",
    "TRAINING_ALGORITHM",
    "DistributionalFastSACTeacherBC",
    "DistributionalFastSACTeacherBCConfig",
    "_DeterministicFastSACStudentEvalPolicy",
    "_DistributionalFastSACDaggerRolloutPolicy",
]
