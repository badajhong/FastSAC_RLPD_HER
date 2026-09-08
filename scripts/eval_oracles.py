"""Evaluation-only replacements for the Student's perception predictions."""

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential

from active_adaptation.learning.ppo.ppo_vel import (
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    ZeroDepthInjector,
)


def load_oracle_teacher(owner, policy_state):
    """Require restored Teacher weights even with PPOVEL's tolerant loader."""
    if not policy_state:
        raise ValueError("oracle_priv_pred requires a checkpoint with Teacher weights")
    names = ["encoder_priv"]
    if hasattr(owner, "height_encoder"):
        names.append("height_encoder")
    for name in names:
        if name not in policy_state:
            raise ValueError(f"oracle_priv_pred checkpoint is missing {name}")
        # PPOVEL.load_state_dict normally warns and continues on incompatible
        # children. A randomly initialized encoder cannot be an oracle target.
        getattr(owner, name).load_state_dict(policy_state[name], strict=True)


class _CopyPrediction(nn.Module):
    def forward(self, target):
        return target.detach().clone()


class _OracleStudentEvalPolicy(nn.Module):
    def __init__(self, owner, perception):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.perception = perception
        self.actor_adapt = owner.actor_adapt
        if owner.cfg.train_dr_estimator:
            self.dr_estimator = owner.dr_estimator

    @torch.no_grad()
    def forward(self, td):
        self.perception(td)
        # TD3 / FastSAC evaluation uses a deterministic physical proposal and
        # its backend-specific mapping into executable actions. Reuse exactly
        # the same conversion as ordinary Student evaluation.
        convert_mean = getattr(self._owner, "_student_action_from_raw_mean", None)
        if callable(convert_mean):
            raw_mean = self.actor_adapt.get_dist(td).mean
            td["action"] = convert_mean(raw_mean)
        else:
            self.actor_adapt(td)
        if hasattr(self, "dr_estimator"):
            self.dr_estimator(td)
        return td


def make_eval_policy(owner, *, oracle_object_pose=False, oracle_priv_pred=False):
    """Build an oracle policy from the same post-normalization inputs as training.

    ``object_`` is the full supervised target of ``object_pred``, including the
    body-frame position and 3x3 rotation used by TransformObject. The environment
    excludes ``object_`` and ``object_geo_`` from VecNorm (trailing underscore).
    Copy the target without any extra scaling, then apply the usual prediction
    transform. ``priv`` has already been normalized by the environment using
    checkpoint statistics; the Teacher's resulting latent needs no extra norm.
    """
    for name, value in (
        ("oracle_object_pose", oracle_object_pose),
        ("oracle_priv_pred", oracle_priv_pred),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if not (oracle_object_pose or oracle_priv_pred):
        return owner.get_rollout_policy("eval")
    if not isinstance(owner, PPOVEL) or owner.cfg.phase != "finetune":
        raise ValueError("Student perception oracles require a PPOVEL-family finetune policy")
    # Reject other derived action backends rather than silently evaluating
    # their actor with PPO's action semantics.
    if (
        type(owner).get_rollout_policy is not PPOVEL.get_rollout_policy
        and not callable(getattr(owner, "_student_action_from_raw_mean", None))
    ):
        raise ValueError("Student perception oracles do not support this evaluation action backend")

    if oracle_priv_pred:
        modules = [owner.object_transform]
        if hasattr(owner, "height_encoder"):
            modules.append(owner.height_encoder)
        modules.extend([
            owner.encoder_priv,
            TensorDictModule(_CopyPrediction(), [PRIV_FEATURE_KEY], [PRIV_PRED_KEY]),
        ])
        # Neither recurrent Student stack is consumed in this mode, so their
        # hidden states need not be advanced, including when both flags are set.
    else:
        if not owner.cfg.use_object_adapt:
            raise ValueError("oracle_object_pose requires algo.use_object_adapt=true")
        depth = getattr(owner, "temporal_depth_gru_ema", None)
        if depth is None:
            depth = ZeroDepthInjector(owner.depth_feature_dim, owner.device)
        modules = [
            depth,
            TensorDictModule(_CopyPrediction(), [OBJECT_KEY], [OBJECT_PRED_KEY]),
            owner.object_pred_transform,
            owner.adapt_ema,
        ]
    return _OracleStudentEvalPolicy(owner, TensorDictSequential(*modules))
