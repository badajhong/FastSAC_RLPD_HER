from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_Q_DELAY_AWARE_CONTEXT_SEMANTICS,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC,
)


_ACTION_DIM = 2
_QUEUE_DEPTH = 3
_DECIMATION = 4
_DELAY_MIN = 2
_DELAY_MAX = 6
_PARAMETER_DIM = _DELAY_MAX - _DELAY_MIN + 2
_CONTEXT_DIM = _PARAMETER_DIM + _ACTION_DIM * (_QUEUE_DEPTH + 1)


def _bare_delay_aware_policy(
    *, causal_hold: bool = False
) -> DistributionalTD3TeacherBC:
    policy = DistributionalTD3TeacherBC.__new__(DistributionalTD3TeacherBC)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_condition_on_actuator_state=True,
        q_use_delay_aware_mdp=True,
        q_use_predicted_effect=False,
        q_use_causal_hold_advantage=causal_hold,
        q_use_residual_film=False,
        q_action_input_gain=1.0,
        action_support_clip=20.0,
    )
    policy.device = torch.device("cpu")
    policy.action_dim = _ACTION_DIM
    policy._q_critic_dim = 3
    policy._q_actuator_parameter_context_dim = _PARAMETER_DIM
    policy._q_actuator_context_dim = _CONTEXT_DIM
    policy._q_network_observation_dim = policy._q_critic_dim + (
        _CONTEXT_DIM if causal_hold else 0
    )
    policy._q_action_input_dim = (
        _ACTION_DIM if causal_hold else _ACTION_DIM + _CONTEXT_DIM
    )
    policy._fastsac_action_low = torch.full((_ACTION_DIM,), -20.0)
    policy._fastsac_action_high = torch.full((_ACTION_DIM,), 20.0)
    policy._fastsac_q_action_center = torch.tensor([1.0, -2.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0, 4.0])
    policy._q_actuator_context_metadata_value = {
        "enabled": True,
        "semantics": FASTSAC_Q_DELAY_AWARE_CONTEXT_SEMANTICS,
        "dimension": _CONTEXT_DIM,
        "delay_range": [_DELAY_MIN, _DELAY_MAX],
        "alpha_range": [0.8, 1.0],
        "action_dim": _ACTION_DIM,
        "queue_depth": _QUEUE_DEPTH,
        "control_decimation": _DECIMATION,
    }
    return policy


def _expected_parameter_context(delay: int, alpha: float) -> torch.Tensor:
    one_hot = F.one_hot(
        torch.tensor(delay - _DELAY_MIN),
        num_classes=_DELAY_MAX - _DELAY_MIN + 1,
    ).float()
    centered_alpha = torch.tensor(
        [2.0 * (alpha - 0.8) / (1.0 - 0.8) - 1.0]
    )
    return torch.cat((one_hot, centered_alpha))


def _slot_major(queue: torch.Tensor) -> torch.Tensor:
    """Flatten ``[..., action, slot]`` with every slot kept contiguous."""
    return queue.transpose(-2, -1).flatten(-2)


def _simulator_successor(
    *,
    delay: int,
    alpha: float,
    applied_action: torch.Tensor,
    action_queue: torch.Tensor,
    issued_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent scalar-loop oracle for ``JointPosition.__call__``.

    The simulator shifts once at substep zero, then selects a queue slot and
    performs one lerp at each of the four physics substeps.
    """
    next_queue = torch.empty_like(action_queue)
    next_queue[:, 0] = issued_action
    for slot in range(1, action_queue.shape[-1]):
        next_queue[:, slot] = action_queue[:, slot - 1]

    next_applied = applied_action.clone()
    for substep in range(_DECIMATION):
        selected_slot = math.ceil((delay - substep) / _DECIMATION)
        selected_command = next_queue[:, selected_slot]
        next_applied = (1.0 - alpha) * next_applied + alpha * selected_command
    return next_applied, next_queue


@pytest.mark.parametrize("causal_hold", (False, True))
def test_delay_aware_capture_is_full_pre_shift_markov_actuator_state(
    causal_hold: bool,
):
    policy = _bare_delay_aware_policy(causal_hold=causal_hold)
    applied_action = torch.tensor([[0.25, -0.5], [-0.75, 0.125]])
    action_queue = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[-1.0, -2.0, -3.0], [-10.0, -20.0, -30.0]],
        ]
    )
    manager = SimpleNamespace(
        min_delay=_DELAY_MIN,
        max_delay=_DELAY_MAX,
        alpha_range=(0.8, 1.0),
        delay=torch.tensor([[_DELAY_MIN], [_DELAY_MAX]]),
        alpha=torch.tensor([[0.8], [1.0]]),
        applied_action=applied_action,
        action_buf=action_queue,
    )
    policy.env = SimpleNamespace(
        action_manager=manager,
        decimation=_DECIMATION,
    )
    policy._q_actuator_context_metadata_value = (
        policy._resolve_q_actuator_context_metadata()
    )
    policy._q_actuator_context_dim = int(
        policy._q_actuator_context_metadata_value["dimension"]
    )

    captured = policy.capture_q_actuator_context()

    assert policy._q_uses_delay_aware_mdp()
    assert policy._q_uses_causal_hold_advantage() is causal_hold
    assert not policy._q_requires_previous_action_context()
    assert policy._q_actuator_context_metadata_value == {
        "enabled": True,
        "semantics": FASTSAC_Q_DELAY_AWARE_CONTEXT_SEMANTICS,
        "dimension": _CONTEXT_DIM,
        "delay_range": [_DELAY_MIN, _DELAY_MAX],
        "alpha_range": [0.8, 1.0],
        "action_dim": _ACTION_DIM,
        "queue_depth": _QUEUE_DEPTH,
        "control_decimation": _DECIMATION,
    }
    assert captured is not None
    assert captured.shape == (2, _CONTEXT_DIM)
    assert torch.allclose(
        captured[:, :_PARAMETER_DIM],
        torch.stack(
            (
                _expected_parameter_context(_DELAY_MIN, 0.8),
                _expected_parameter_context(_DELAY_MAX, 1.0),
            )
        ),
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(
        captured[:, _PARAMETER_DIM : _PARAMETER_DIM + _ACTION_DIM],
        applied_action,
    )
    assert torch.equal(
        captured[:, _PARAMETER_DIM + _ACTION_DIM :],
        _slot_major(action_queue),
    )

    manager.delay.fill_(_DELAY_MAX)
    manager.alpha.fill_(1.0)
    manager.applied_action.fill_(99.0)
    manager.action_buf.fill_(99.0)
    assert torch.equal(
        captured[:, _PARAMETER_DIM : _PARAMETER_DIM + _ACTION_DIM],
        torch.tensor([[0.25, -0.5], [-0.75, 0.125]]),
    )
    assert torch.equal(
        captured[:, _PARAMETER_DIM + _ACTION_DIM :],
        torch.tensor(
            [
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                [-1.0, -10.0, -2.0, -20.0, -3.0, -30.0],
            ]
        ),
    )


@pytest.mark.parametrize("delay", range(_DELAY_MIN, _DELAY_MAX + 1))
@pytest.mark.parametrize("alpha", (0.8, 1.0))
def test_delay_aware_successor_matches_joint_position_substeps(
    delay: int, alpha: float
):
    policy = _bare_delay_aware_policy()
    applied_action = torch.tensor([[0.35, -0.45]])
    action_queue = torch.tensor(
        [[[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]]]
    )
    issued_action = torch.tensor([[0.7, -0.8]], requires_grad=True)
    context = policy._encode_q_actuator_context(
        torch.tensor([[delay]]),
        torch.tensor([[alpha]]),
        applied_action=applied_action,
        action_queue=action_queue,
    ).requires_grad_()

    successor = policy._next_q_actuator_context(context, issued_action)
    expected_applied, expected_queue = _simulator_successor(
        delay=delay,
        alpha=alpha,
        applied_action=applied_action[0],
        action_queue=action_queue[0],
        issued_action=issued_action.detach()[0],
    )
    expected = torch.cat(
        (
            _expected_parameter_context(delay, alpha),
            expected_applied,
            _slot_major(expected_queue),
        )
    ).unsqueeze(0)

    assert torch.allclose(successor, expected, rtol=0.0, atol=1e-6)
    assert torch.equal(successor[:, :_PARAMETER_DIM], context[:, :_PARAMETER_DIM])
    assert successor.untyped_storage().data_ptr() != (
        context.untyped_storage().data_ptr()
    )
    assert not successor.requires_grad
    assert issued_action.grad is None
    assert context.grad is None


class _RecordingOrdinaryQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.observations: torch.Tensor | None = None
        self.action_features: torch.Tensor | None = None

    def forward(
        self, observations: torch.Tensor, action_features: torch.Tensor
    ) -> torch.Tensor:
        self.observations = observations
        self.action_features = action_features
        return action_features.sum(dim=-1, keepdim=True)


class _RecordingCausalQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.observations: torch.Tensor | None = None
        self.candidate: torch.Tensor | None = None
        self.hold: torch.Tensor | None = None

    def forward(
        self,
        observations: torch.Tensor,
        candidate: torch.Tensor,
        hold: torch.Tensor,
    ) -> torch.Tensor:
        self.observations = observations
        self.candidate = candidate
        self.hold = hold
        return (
            observations.sum(dim=-1, keepdim=True)
            + candidate.sum(dim=-1, keepdim=True)
            + hold.sum(dim=-1, keepdim=True)
        )


def test_delay_aware_q_joint_branch_keeps_candidate_gradient_and_detaches_state():
    policy = _bare_delay_aware_policy()
    delay = torch.tensor([[4]])
    alpha = torch.tensor([[0.9]])
    applied_action = torch.tensor([[3.0, 2.0]])
    action_queue = torch.tensor(
        [[[5.0, 7.0, 9.0], [6.0, 10.0, 14.0]]]
    )
    context = policy._encode_q_actuator_context(
        delay,
        alpha,
        applied_action=applied_action,
        action_queue=action_queue,
    ).requires_grad_()
    critic_observations = torch.tensor([[0.2, -0.1, 0.3]])
    candidate = torch.tensor([[5.0, 6.0]], requires_grad=True)
    qnet = _RecordingOrdinaryQ()

    logits = policy._q_forward(
        qnet,
        critic_observations,
        candidate,
        context,
    )

    assert qnet.observations is critic_observations
    assert qnet.action_features is not None
    # Candidate, delay one-hot/centered alpha, then applied action and the
    # pre-shift queue. Physical actions use Q's nominal coordinates while the
    # categorical/centered actuator parameters are already normalized.
    expected_candidate = torch.tensor([[2.0, 2.0]])
    expected_parameters = _expected_parameter_context(4, 0.9).unsqueeze(0)
    expected_applied = torch.tensor([[1.0, 1.0]])
    expected_queue = torch.tensor(
        [[2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]
    )
    expected_features = torch.cat(
        (
            expected_candidate,
            expected_parameters,
            expected_applied,
            expected_queue,
        ),
        dim=-1,
    )
    assert qnet.action_features.shape == (1, _ACTION_DIM + _CONTEXT_DIM)
    assert torch.allclose(
        qnet.action_features, expected_features, rtol=0.0, atol=1e-6
    )

    logits.sum().backward()
    assert torch.allclose(
        candidate.grad,
        torch.tensor([[0.5, 0.25]]),
        rtol=0.0,
        atol=1e-7,
    )
    assert context.grad is None


def test_delay_aware_causal_q_separates_markov_state_candidate_and_hold():
    policy = _bare_delay_aware_policy(causal_hold=True)
    context = policy._encode_q_actuator_context(
        torch.tensor([[4]]),
        torch.tensor([[0.9]]),
        applied_action=torch.tensor([[3.0, 2.0]]),
        action_queue=torch.tensor(
            [[[5.0, 7.0, 9.0], [6.0, 10.0, 14.0]]]
        ),
    ).requires_grad_()
    critic_observations = torch.tensor([[0.2, -0.1, 0.3]])
    candidate = torch.tensor([[7.0, 10.0]], requires_grad=True)
    qnet = _RecordingCausalQ()

    logits = policy._q_forward(
        qnet,
        critic_observations,
        candidate,
        context,
    )

    expected_parameters = _expected_parameter_context(4, 0.9).unsqueeze(0)
    expected_actuator_state = torch.cat(
        (
            expected_parameters,
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]),
        ),
        dim=-1,
    )
    assert qnet.observations is not None
    assert qnet.candidate is not None
    assert qnet.hold is not None
    assert qnet.observations.shape == (1, 3 + _CONTEXT_DIM)
    assert torch.allclose(
        qnet.observations,
        torch.cat((critic_observations, expected_actuator_state), dim=-1),
        rtol=0.0,
        atol=1e-6,
    )
    # The action branch is the hold-relative innovation: normalized candidate
    # [3, 3] minus normalized pre-shift queue slot zero [2, 2].
    assert torch.equal(qnet.candidate, torch.tensor([[1.0, 1.0]]))
    # Reissuing queue slot zero is exactly zero innovation. The smoothed
    # applied command ([1, 1] here) is deliberately not the hold anchor.
    assert torch.equal(qnet.hold, torch.zeros(1, _ACTION_DIM))

    logits.sum().backward()
    assert torch.allclose(
        candidate.grad,
        torch.tensor([[0.5, 0.25]]),
        rtol=0.0,
        atol=1e-7,
    )
    assert context.grad is None
