"""Perception-only VAIC finetuning from privileged Teacher rollouts.

The policy keeps the original PPOVEL perception objective unchanged.  During
collection, however, the environment action is produced exclusively by the
privileged PPO Teacher.  The EMA Student perception stack is still evaluated
to advance the recurrent hidden states that ``PPOVEL.train_adapt`` expects,
but neither its latent nor ``actor_adapt`` can affect the executed action.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY
from .ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    PPOConfig,
    PPOVEL,
    ZeroDepthInjector,
)


TRAINING_ALGORITHM = "vaic_teacher_rollout_perception_only_v2"
ROLLOUT_SEMANTICS = (
    "privileged_ppo_teacher_residual_distribution_plus_reference_joint_position_v1"
)
PERCEPTION_OBJECTIVE_SEMANTICS = "ppo_vel_train_adapt_exact_two_epoch_online_rollout_v1"

ONLINE_PERCEPTION_MODULES = (
    "temporal_depth_gru",  # owns the depth CNN as a registered submodule
    "object_adapt",
    "adapt_module",
)
EMA_PERCEPTION_MODULES = (
    "temporal_depth_gru_ema",
    "object_adapt_ema",
    "adapt_ema",
)
FULL_PERCEPTION_CHECKPOINT_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)
FRESH_DEPTH_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
)
PERCEPTION_INITIALIZATION_TEACHER_WARMSTART = "teacher_warmstart"
PERCEPTION_INITIALIZATION_FRESH = "fresh"
PERCEPTION_INITIALIZATIONS = frozenset(
    {
        PERCEPTION_INITIALIZATION_TEACHER_WARMSTART,
        PERCEPTION_INITIALIZATION_FRESH,
    }
)
_ACTOR_STD_STATE_KEY = "module.0.module.2.module.actor_std"


def validate_teacher_rollout_perception_config(
    cfg,
    *,
    actor_distillation: bool,
) -> None:
    """Validate the shared frozen-Teacher perception-stage contract.

    The perception-only and perception+actor stages deliberately differ in
    exactly one optimization choice: whether ``actor_adapt`` is distilled.
    Keeping the remaining invariants here prevents the two entrypoints from
    silently drifting apart.
    """

    if cfg.phase != "finetune" or cfg.vecnorm != "eval":
        raise ValueError(
            "Teacher-rollout perception training requires phase=finetune "
            "and vecnorm=eval"
        )
    if bool(cfg.enable_residual_distillation) is not actor_distillation:
        expected = "enable" if actor_distillation else "disable"
        raise ValueError(
            "Teacher-rollout perception training must "
            f"{expected} actor distillation"
        )
    if cfg.train_dr_estimator:
        raise ValueError(
            "Teacher-rollout perception training does not optimize dr_estimator"
        )
    if not cfg.use_depth or not cfg.use_object_adapt or cfg.adapt_module != "gru":
        raise ValueError(
            "Teacher-rollout perception training requires depth, object_adapt, "
            "and the GRU adapt_module"
        )
    initialization = cfg.perception_initialization
    if initialization not in PERCEPTION_INITIALIZATIONS:
        choices = ", ".join(sorted(PERCEPTION_INITIALIZATIONS))
        raise ValueError(
            "perception_initialization must be one of "
            f"{{{choices}}}, got {initialization!r}"
        )
    noise = cfg.load_noise_scale
    if (
        isinstance(noise, bool)
        or not isinstance(noise, (int, float))
        or not torch.isfinite(torch.tensor(float(noise))).item()
        or float(noise) <= 0.0
    ):
        raise ValueError("load_noise_scale must be finite and positive")


@dataclass
class TeacherRolloutPerceptionConfig(PPOConfig):
    """Structured Hydra surface for the dedicated perception stage."""

    _target_: str = (
        "active_adaptation.learning.ppo.perception_only."
        "TeacherRolloutPerceptionOnly"
    )
    name: str = "teacher_rollout_perception_only"
    phase: str = "finetune"
    vecnorm: str = "eval"
    enable_residual_distillation: bool = False
    train_dr_estimator: bool = False
    use_object_adapt: bool = True
    use_depth: bool = True
    adapt_module: str = "gru"
    load_noise_scale: float | None = 0.2
    # ``teacher_warmstart`` reproduces PPOVEL finetune: load the Teacher's
    # zero-depth object/adaptation modules and initialize only depth freshly.
    # ``fresh`` keeps all seven constructor-created online/EMA perception
    # children fresh, preventing perception-weight transfer from the Teacher.
    perception_initialization: str = PERCEPTION_INITIALIZATION_TEACHER_WARMSTART
    in_keys: List[str] = (
        CMD_KEY,
        OBS_KEY,
        OBJECT_KEY,
        OBS_PRIV_KEY,
        OBJECT_GEO_KEY,
        VEL_CMD_KEY,
        DEPTH_KEY,
    )


ConfigStore.instance().store(
    "teacher_rollout_perception_only",
    node=TeacherRolloutPerceptionConfig(),
    group="algo",
)


class _PrivilegedTeacherPerceptionRollout(nn.Module):
    """Advance Student EMA memories but execute only the privileged Teacher."""

    _STUDENT_SCRATCH_KEYS = (
        "_depth_feature",
        OBJECT_PRED_KEY,
        OBJECT_PRED_TRANS_KEY,
        "_object_adapt_mlp_inp",
        "_obj_adapt_mlp",
        "_object_adapt_inp",
        "_adapt_inp",
    )

    def __init__(self, owner: "TeacherRolloutPerceptionOnly"):
        super().__init__()
        # Registering the owner here would create a module cycle and duplicate
        # the complete model in checkpoint state_dicts.
        object.__setattr__(self, "_owner", owner)

    @torch.no_grad()
    def forward(self, td: TensorDict) -> TensorDict:
        owner = self._owner

        # PPOVEL finetune carries EMA recurrent states from one control step to
        # the next.  Preserve that exact history contract for train_adapt, while
        # deliberately discarding its action path.
        if hasattr(owner, "temporal_depth_gru_ema"):
            owner.temporal_depth_gru_ema(td)
        else:
            ZeroDepthInjector(owner.depth_feature_dim, owner.device)(td)
        if owner.cfg.use_object_adapt:
            owner.object_adapt_ema(td)
            owner.object_pred_transform(td)
        owner.adapt_ema(td)

        # The source train-phase PPO Actor represents a residual Normal followed
        # by ref_joint_pos addition.  This target policy is constructed in
        # finetune phase (so depth modules exist), where PPOVEL's actor wrapper
        # intentionally leaves that residual untouched.  Query it on a shallow
        # container clone, then restore the original Teacher action coordinates
        # explicitly.  Under train.py's ExplorationType.RANDOM context this
        # samples the residual Normal with cfg.load_noise_scale.
        teacher_td = td.clone(False)
        owner.object_transform(teacher_td)
        if hasattr(owner, "height_encoder"):
            owner.height_encoder(teacher_td)
        owner.encoder_priv(teacher_td)
        owner.actor(teacher_td)
        td[ACTION_KEY] = teacher_td[ACTION_KEY] + td[REF_JPOS_KEY]

        # Keep priv_pred for diagnostics and, in the joint Actor stage, exact
        # rollout-EMA BC input.  Keep recurrent next-state keys for history, but
        # do not retain large intermediate tensors in the N x T buffer.
        for key in self._STUDENT_SCRATCH_KEYS:
            if key in td.keys(True, True):
                td.del_(key)
        return td


class TeacherRolloutPerceptionOnly(PPOVEL):
    """PPOVEL perception learner with a frozen privileged-Teacher collector."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_config(cfg)
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self._enforce_perception_only_ownership()

    @staticmethod
    def _validate_config(cfg) -> None:
        validate_teacher_rollout_perception_config(
            cfg,
            actor_distillation=False,
        )

    def _enforce_perception_only_ownership(self) -> None:
        online_modules = [getattr(self, name) for name in ONLINE_PERCEPTION_MODULES]
        trainable_parameter_ids = {
            id(parameter)
            for module in online_modules
            for parameter in module.parameters()
        }

        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in trainable_parameter_ids)
        for module in self.children():
            module.eval()
        for module in online_modules:
            module.train()
        for name in EMA_PERCEPTION_MODULES:
            getattr(self, name).requires_grad_(False).eval()

        # These inherited optimizers must not remain callable by accident.  The
        # sole optimizer is the unchanged PPOVEL opt_adapt.
        self.opt_policy = None
        self.opt_critic = None
        if hasattr(self, "opt_adapt_actor"):
            self.opt_adapt_actor = None
        if hasattr(self, "opt_dr_estimator"):
            self.opt_dr_estimator = None

        optimizer_parameter_ids = {
            id(parameter)
            for group in self.opt_adapt.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameter_ids != trainable_parameter_ids:
            raise RuntimeError(
                "opt_adapt parameter ownership does not exactly match the three "
                "online perception modules"
            )

    def requires_value_bootstrap(self) -> bool:
        return False

    def get_rollout_policy(self, mode: str = "train"):
        if mode == "train":
            return _PrivilegedTeacherPerceptionRollout(self)
        # Final evaluation remains the deployable EMA perception + actor_adapt
        # policy, exactly as in PPOVEL finetune.
        return super().get_rollout_policy(mode)

    def train_op(self, tensordict: TensorDict):
        # This is intentionally the inherited implementation, not a rewritten
        # approximation: same targets, two epochs, minibatches, losses, clipping,
        # optimizer, and one tau=0.04 EMA update per rollout.
        perception_batch = tensordict.exclude("stats").copy()
        info = self.train_adapt(perception_batch)
        self.num_updates += 1
        info["perception_only/teacher_control"] = 1.0
        info["perception_only/teacher_noise_scale"] = float(
            self.cfg.load_noise_scale
        )
        info["perception_only/update_count"] = int(self.num_updates)
        return info

    def _restore_frozen_student_std(self, source_state: Mapping) -> None:
        """Undo PPOVEL's shared load-noise hook for the unused Student actor."""

        actor_adapt_state = source_state.get("actor_adapt")
        if not isinstance(actor_adapt_state, Mapping):
            raise ValueError("Teacher checkpoint is missing actor_adapt state")
        source_std = actor_adapt_state.get(_ACTOR_STD_STATE_KEY)
        if not torch.is_tensor(source_std):
            raise ValueError("Teacher checkpoint actor_adapt is missing actor_std")
        target_std = self.actor_adapt.module[0][2].module.actor_std
        if target_std.shape != source_std.shape:
            raise RuntimeError("Teacher checkpoint actor_adapt std shape is incompatible")
        target_std.data.copy_(source_std.to(device=target_std.device, dtype=target_std.dtype))

    def _verify_teacher_noise_scale(self) -> None:
        teacher_std = self.actor.module[0][2].module.actor_std.detach()
        expected = torch.full_like(teacher_std, float(self.cfg.load_noise_scale))
        if not torch.equal(teacher_std, expected):
            raise RuntimeError(
                "Teacher actor_std was not reset exactly to algo.load_noise_scale"
            )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a train-phase Teacher under an explicit perception-init contract.

        ``teacher_warmstart`` reproduces PPOVEL finetune by loading the source
        object/adaptation children while leaving its absent depth children fresh.
        ``fresh`` skips every online and EMA perception child, so the source can
        provide rollout/target modules without transferring perception weights.
        All non-perception children remain strict in either mode.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError("Teacher policy checkpoint state must be a mapping")
        if state_dict.get("last_phase") != "train":
            raise ValueError(
                "Perception-only fresh training requires policy.last_phase='train'"
            )

        initialization = self.cfg.perception_initialization
        if initialization == PERCEPTION_INITIALIZATION_FRESH:
            fresh_module_names = frozenset(FULL_PERCEPTION_CHECKPOINT_MODULES)
        else:
            fresh_module_names = frozenset(FRESH_DEPTH_MODULES)

        loaded = []
        fresh = []
        for name, module in self.named_children():
            # Skip before reading the source mapping.  This preserves the exact
            # constructor values and Parameter identities already owned by
            # opt_adapt, with no transient or partial Teacher-state mutation.
            if name in fresh_module_names:
                fresh.append(name)
                continue
            if name not in state_dict:
                raise ValueError(f"Teacher checkpoint is missing required module {name!r}")
            module_state = state_dict[name]
            if not isinstance(module_state, Mapping):
                raise ValueError(
                    f"Teacher checkpoint module {name!r} is not a state mapping"
                )
            try:
                module.load_state_dict(module_state, strict=strict)
            except Exception as exc:
                raise RuntimeError(
                    f"Teacher checkpoint module {name!r} is incompatible"
                ) from exc
            loaded.append(name)

        required = {
            "actor",
            "actor_adapt",
            "encoder_priv",
        }
        if initialization == PERCEPTION_INITIALIZATION_TEACHER_WARMSTART:
            required.update(
                {
                    "object_adapt",
                    "object_adapt_ema",
                    "adapt_module",
                    "adapt_ema",
                }
            )
        missing_required = sorted(required.difference(loaded))
        if missing_required:
            raise RuntimeError(
                "Teacher checkpoint did not load required modules: "
                f"{missing_required}"
            )

        self._restore_frozen_student_std(state_dict)
        self._verify_teacher_noise_scale()
        self.env.set_progress(int(state_dict.get("last_iter", 0)))
        saved_lr = state_dict.get("lr_policy")
        if saved_lr is not None:
            self.lr_policy = float(saved_lr)
        self._enforce_perception_only_ownership()

        print(f"Successfully loaded frozen Teacher modules: {loaded}.")
        if fresh:
            print(
                "Intentionally kept perception modules fresh under "
                f"perception_initialization={initialization!r}: {fresh}. "
                "Online modules are trainable; EMA modules are frozen targets."
            )
        return fresh

    def state_dict(self):
        state = OrderedDict(super().state_dict())
        state["training_algorithm"] = TRAINING_ALGORITHM
        state["rollout_semantics"] = ROLLOUT_SEMANTICS
        state["perception_objective_semantics"] = PERCEPTION_OBJECTIVE_SEMANTICS
        state["perception_online_modules"] = ONLINE_PERCEPTION_MODULES
        state["perception_checkpoint_modules"] = FULL_PERCEPTION_CHECKPOINT_MODULES
        state["perception_initialization"] = str(self.cfg.perception_initialization)
        state["perception_fresh_modules"] = (
            FULL_PERCEPTION_CHECKPOINT_MODULES
            if self.cfg.perception_initialization == PERCEPTION_INITIALIZATION_FRESH
            else FRESH_DEPTH_MODULES
        )
        state["teacher_action_noise_scale"] = float(self.cfg.load_noise_scale)
        state["perception_update_count"] = int(self.num_updates)
        return state


__all__ = [
    "EMA_PERCEPTION_MODULES",
    "FULL_PERCEPTION_CHECKPOINT_MODULES",
    "ONLINE_PERCEPTION_MODULES",
    "PERCEPTION_OBJECTIVE_SEMANTICS",
    "PERCEPTION_INITIALIZATIONS",
    "PERCEPTION_INITIALIZATION_FRESH",
    "PERCEPTION_INITIALIZATION_TEACHER_WARMSTART",
    "ROLLOUT_SEMANTICS",
    "TRAINING_ALGORITHM",
    "TeacherRolloutPerceptionConfig",
    "TeacherRolloutPerceptionOnly",
    "validate_teacher_rollout_perception_config",
]
