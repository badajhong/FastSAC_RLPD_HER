"""Joint VAIC perception and deployment-aligned Student-actor BC.

The environment is always controlled by the frozen privileged Teacher.  The
three online perception modules learn the unchanged PPOVEL supervised losses,
while ``actor_adapt`` learns to match the Teacher's clean absolute action mean
conditioned only on the detached EMA privileged latent saved during rollout.
The Actor never consumes an online perception prediction during BC, so Actor
gradients cannot enter perception and the rollout policy remains Teacher-only.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from .perception_only import (
    EMA_PERCEPTION_MODULES,
    ONLINE_PERCEPTION_MODULES,
    TeacherRolloutPerceptionConfig,
    TeacherRolloutPerceptionOnly,
    validate_teacher_rollout_perception_config,
)
from .ppo_vel import PRIV_PRED_KEY, REF_JPOS_KEY


TRAINING_ALGORITHM = "vaic_teacher_rollout_perception_actor_ema_bc_v2"
ACTOR_OBJECTIVE_SEMANTICS = (
    "rollout_cached_ema_priv_latent_detached_to_clean_absolute_teacher_mean_bc_v2"
)
ACTOR_INITIALIZATION_SEMANTICS = (
    "ppo_vel_teacher_checkpoint_actor_adapt_then_rollout_cached_ema_priv_latent_bc_v2"
)
ACTOR_BC_PERCEPTION_SOURCE = "ema_rollout"
ROLLOUT_EMA_PRIV_PRED_KEY = "_actor_bc_rollout_ema_priv_pred"
OPTIMIZED_MODULES = (*ONLINE_PERCEPTION_MODULES, "actor_adapt")


@dataclass
class TeacherRolloutPerceptionActorConfig(TeacherRolloutPerceptionConfig):
    """Structured Hydra surface for perception plus Student-actor BC."""

    _target_: str = (
        "active_adaptation.learning.ppo.perception_actor."
        "TeacherRolloutPerceptionActor"
    )
    name: str = "teacher_rollout_perception_actor"
    enable_residual_distillation: bool = True
    # This distinguishes predicted-latent BC from PPOVEL train-phase GT-latent
    # BC.  ``actor_bc_perception_source`` fixes which prediction is permitted.
    distill_with_priv_pred: bool = True
    actor_bc_perception_source: str = ACTOR_BC_PERCEPTION_SOURCE


ConfigStore.instance().store(
    "teacher_rollout_perception_actor",
    node=TeacherRolloutPerceptionActorConfig(),
    group="algo",
)


class TeacherRolloutPerceptionActor(TeacherRolloutPerceptionOnly):
    """Train perception and actor_adapt while a frozen Teacher controls rollout."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._actor_adapt_loaded_from_teacher_checkpoint = False
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)

    @staticmethod
    def _validate_config(cfg) -> None:
        validate_teacher_rollout_perception_config(
            cfg,
            actor_distillation=True,
        )
        if not bool(cfg.distill_with_priv_pred):
            raise ValueError(
                "Perception+actor training requires distill_with_priv_pred=true: "
                "actor_adapt must consume a detached predicted priv_pred"
            )
        if str(cfg.actor_bc_perception_source) != ACTOR_BC_PERCEPTION_SOURCE:
            raise ValueError(
                "Perception+actor training requires "
                f"actor_bc_perception_source={ACTOR_BC_PERCEPTION_SOURCE!r}; "
                "online Actor-BC input is forbidden"
            )

    def _enforce_perception_only_ownership(self) -> None:
        """Own exactly perception-online parameters and actor_adapt parameters.

        The method name intentionally overrides the base-stage ownership hook.
        Separate optimizers are retained so the actor loss cannot update the
        perception stack.
        """

        perception_modules = [
            getattr(self, name) for name in ONLINE_PERCEPTION_MODULES
        ]
        perception_parameter_ids = {
            id(parameter)
            for module in perception_modules
            for parameter in module.parameters()
        }
        actor_parameter_ids = {
            id(parameter) for parameter in self.actor_adapt.parameters()
        }
        if perception_parameter_ids.intersection(actor_parameter_ids):
            raise RuntimeError("Perception and actor_adapt parameters must be disjoint")
        trainable_parameter_ids = perception_parameter_ids | actor_parameter_ids

        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in trainable_parameter_ids)
        for module in self.children():
            module.eval()
        for module in perception_modules:
            module.train()
        self.actor_adapt.train()
        for name in EMA_PERCEPTION_MODULES:
            getattr(self, name).requires_grad_(False).eval()

        # PPO policy/value learning is forbidden in this stage.  Perception and
        # actor BC retain their two independent inherited optimizers.
        self.opt_policy = None
        self.opt_critic = None
        if hasattr(self, "opt_dr_estimator"):
            self.opt_dr_estimator = None
        if not isinstance(self.opt_adapt, torch.optim.Optimizer):
            raise RuntimeError("Perception optimizer opt_adapt is unavailable")
        if not isinstance(self.opt_adapt_actor, torch.optim.Optimizer):
            raise RuntimeError("Actor BC optimizer opt_adapt_actor is unavailable")

        perception_optimizer_parameter_ids = {
            id(parameter)
            for group in self.opt_adapt.param_groups
            for parameter in group["params"]
        }
        if perception_optimizer_parameter_ids != perception_parameter_ids:
            raise RuntimeError(
                "opt_adapt parameter ownership does not exactly match the three "
                "online perception modules"
            )
        actor_optimizer_parameter_ids = {
            id(parameter)
            for group in self.opt_adapt_actor.param_groups
            for parameter in group["params"]
        }
        if actor_optimizer_parameter_ids != actor_parameter_ids:
            raise RuntimeError(
                "opt_adapt_actor parameter ownership does not exactly match "
                "actor_adapt"
            )

    def _actor_distillation_target_mean(
        self,
        tensordict: TensorDict,
        teacher_dist,
    ) -> torch.Tensor:
        """Convert the finetune-topology Teacher residual mean to absolute action."""

        if REF_JPOS_KEY not in tensordict.keys(True, True):
            raise KeyError(
                f"Actor BC requires {REF_JPOS_KEY!r} to reconstruct Teacher action"
            )
        reference = tensordict[REF_JPOS_KEY]
        if reference.shape != teacher_dist.mean.shape:
            raise RuntimeError(
                "Teacher residual mean and reference joint position shapes differ: "
                f"mean={tuple(teacher_dist.mean.shape)}, "
                f"reference={tuple(reference.shape)}"
            )
        return teacher_dist.mean + reference

    def _actor_distillation_priv_pred(
        self,
        tensordict: TensorDict,
    ) -> torch.Tensor:
        """Return only the EMA latent captured along the real rollout history."""

        if ROLLOUT_EMA_PRIV_PRED_KEY not in tensordict.keys(True, True):
            raise RuntimeError(
                "Actor EMA-BC batch is missing the rollout-cached EMA priv_pred; "
                "online priv_pred fallback is forbidden"
            )
        return tensordict[ROLLOUT_EMA_PRIV_PRED_KEY]

    def load_state_dict(self, state_dict, strict: bool = True):
        fresh = super().load_state_dict(state_dict, strict=strict)
        self._actor_adapt_loaded_from_teacher_checkpoint = True
        return fresh

    def train_op(self, tensordict: TensorDict):
        perception_actor_batch = tensordict.exclude("stats").copy()
        if PRIV_PRED_KEY not in perception_actor_batch.keys(True, True):
            raise RuntimeError(
                "Teacher rollout did not provide EMA priv_pred for Actor BC"
            )
        if ROLLOUT_EMA_PRIV_PRED_KEY in perception_actor_batch.keys(True, True):
            raise RuntimeError("Actor EMA-BC cache key already exists in rollout")

        rollout_ema_priv_pred = perception_actor_batch[PRIV_PRED_KEY]
        if rollout_ema_priv_pred.shape[-1] != int(self.cfg.latent_dim):
            raise RuntimeError(
                "Teacher rollout EMA priv_pred has an incompatible latent width: "
                f"expected={int(self.cfg.latent_dim)}, "
                f"actual={int(rollout_ema_priv_pred.shape[-1])}"
            )
        if not torch.isfinite(rollout_ema_priv_pred).all():
            raise RuntimeError("Teacher rollout EMA priv_pred contains NaN/Inf")

        # Move rather than clone the rollout output: this keeps one latent-sized
        # buffer while protecting the exact EMA recurrent trajectory from the
        # online adapt_module, which will recreate PRIV_PRED_KEY in train_adapt.
        perception_actor_batch.rename_key_(
            PRIV_PRED_KEY,
            ROLLOUT_EMA_PRIV_PRED_KEY,
        )
        info = self.train_adapt(perception_actor_batch)
        self.num_updates += 1
        info["perception_actor/teacher_control"] = 1.0
        info["perception_actor/actor_adapt_bc"] = 1.0
        info["perception_actor/rollout_ema_priv_pred_input"] = 1.0
        info["perception_actor/teacher_noise_scale"] = float(
            self.cfg.load_noise_scale
        )
        info["perception_actor/update_count"] = int(self.num_updates)
        return info

    def state_dict(self):
        state = OrderedDict(super().state_dict())
        state["training_algorithm"] = TRAINING_ALGORITHM
        state["actor_objective_semantics"] = ACTOR_OBJECTIVE_SEMANTICS
        state["actor_initialization_semantics"] = ACTOR_INITIALIZATION_SEMANTICS
        state["actor_adapt_loaded_from_teacher_checkpoint"] = bool(
            self._actor_adapt_loaded_from_teacher_checkpoint
        )
        state["actor_adapt_trained"] = True
        state["actor_adapt_controls_rollout"] = False
        state["actor_bc_perception_source"] = ACTOR_BC_PERCEPTION_SOURCE
        state["actor_bc_uses_online_priv_pred"] = False
        state["actor_adapt_bc_update_count"] = int(self.num_updates)
        state["optimized_modules"] = OPTIMIZED_MODULES
        return state


__all__ = [
    "ACTOR_BC_PERCEPTION_SOURCE",
    "ACTOR_INITIALIZATION_SEMANTICS",
    "ACTOR_OBJECTIVE_SEMANTICS",
    "OPTIMIZED_MODULES",
    "ROLLOUT_EMA_PRIV_PRED_KEY",
    "TRAINING_ALGORITHM",
    "TeacherRolloutPerceptionActor",
    "TeacherRolloutPerceptionActorConfig",
]
