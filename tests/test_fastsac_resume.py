from types import SimpleNamespace
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTOR_BACKEND,
    FASTSAC_STUDENT_TRAINING_ALGORITHM,
    _FastSACVAICBase,
)


class _TinyFastSAC(_FastSACVAICBase):
    def __init__(self, phase="train", reverse_actor_params=False):
        nn.Module.__init__(self)
        self.cfg = SimpleNamespace(phase=phase)
        self.actor_backend = FASTSAC_ACTOR_BACKEND
        self.actor = nn.Linear(3, 2)
        self.qnet = nn.Linear(3, 1)
        actor_params = list(self.actor.parameters())
        if reverse_actor_params:
            actor_params.reverse()
        self.sac_teacher_actor_optimizer = torch.optim.AdamW(
            actor_params, lr=3e-4, betas=(0.9, 0.95)
        )
        self.opt_q = torch.optim.AdamW(
            self.qnet.parameters(), lr=3e-4, betas=(0.9, 0.95)
        )
        self.num_updates = 0

    def _optimizer_registry(self):
        return OrderedDict((
            ("opt_q", self.opt_q),
            ("sac_teacher_actor_optimizer", self.sac_teacher_actor_optimizer),
        ))


def _tiny_policy(phase="train", reverse_actor_params=False):
    return _TinyFastSAC(phase, reverse_actor_params)


def _take_optimizer_step(policy):
    policy.sac_teacher_actor_optimizer.zero_grad()
    policy.opt_q.zero_grad()
    observation = torch.tensor([[0.2, -0.4, 0.8]])
    loss = policy.actor(observation).square().sum()
    loss = loss + policy.qnet(observation).square().sum()
    loss.backward()
    policy.sac_teacher_actor_optimizer.step()
    policy.opt_q.step()


def _assert_optimizer_state_equal(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id in left_state["state"]:
        for key, left_value in left_state["state"][parameter_id].items():
            right_value = right_state["state"][parameter_id][key]
            if torch.is_tensor(left_value):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_same_phase_restores_adam_moments_and_update_counter():
    torch.manual_seed(0)
    source = _tiny_policy()
    _take_optimizer_step(source)
    source.num_updates = 37
    checkpoint = {
        "last_phase": "train",
        "optimizer_resume_state": source._optimizer_resume_state(),
    }
    signature = checkpoint["optimizer_resume_state"]["signature"]
    signature.pop("policy_family")
    signature["policy_class"] = (
        "active_adaptation.learning.ppo.ppo_fastsac_vel.HOIFastSACVEL"
    )

    torch.manual_seed(1)
    resumed = _tiny_policy()
    assert resumed._restore_optimizer_resume_state(checkpoint)
    assert resumed.num_updates == 37
    _assert_optimizer_state_equal(
        source.sac_teacher_actor_optimizer,
        resumed.sac_teacher_actor_optimizer,
    )
    _assert_optimizer_state_equal(source.opt_q, resumed.opt_q)


def test_cross_phase_does_not_restore_optimizer_moments():
    source = _tiny_policy(phase="train")
    _take_optimizer_step(source)
    checkpoint = {
        "last_phase": "train",
        "optimizer_resume_state": source._optimizer_resume_state(),
    }
    student = _tiny_policy(phase="finetune")

    assert not student._restore_optimizer_resume_state(checkpoint)
    assert student.num_updates == 0
    assert student.sac_teacher_actor_optimizer.state == {}
    assert student.opt_q.state == {}


def test_same_signature_rejects_changed_optimizer_parameter_order():
    source = _tiny_policy()
    checkpoint = {
        "last_phase": "train",
        "optimizer_resume_state": source._optimizer_resume_state(),
    }
    changed = _tiny_policy(reverse_actor_params=True)

    with pytest.raises(ValueError, match="parameter topology"):
        changed._restore_optimizer_resume_state(checkpoint)


def test_pre_rename_student_optimizer_signature_migrates_to_stable_family():
    signature = {
        "policy_class": (
            "active_adaptation.learning.ppo.ppo_fastsac_vel."
            "FastSACVelFinetune"
        ),
        "phase": "finetune",
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "optimizer_names": [
            "alpha_optimizer",
            "opt_adapt",
            "opt_q",
            "sac_actor_optimizer",
        ],
    }

    normalized = _FastSACVAICBase._normalize_optimizer_resume_signature(signature)

    assert normalized["policy_family"] == FASTSAC_STUDENT_TRAINING_ALGORITHM
    assert "policy_class" not in normalized


def test_removed_ppo_hybrid_signature_is_not_migrated_as_fastsac():
    signature = {
        "policy_class": (
            "active_adaptation.learning.ppo.ppo_fastsac_vel.PPOFastSACVEL"
        ),
        "phase": "train",
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "optimizer_names": ["opt_q"],
    }

    normalized = _FastSACVAICBase._normalize_optimizer_resume_signature(signature)

    assert normalized["policy_class"] == signature["policy_class"]
    assert "policy_family" not in normalized
