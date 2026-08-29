import torch
from torch import nn

from active_adaptation.learning.ppo.fastsac_critic_probe import (
    matched_decoy_indices,
    phase_balanced_sample_indices,
    phase_bin_indices,
    summarize_distributional_critic_conditions,
)
from active_adaptation.learning.ppo.fastsac_gradient_probe import (
    _action_q_gradients,
)


class _ContextSensitiveTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, action_features):
        del observations
        score = action_features[:, 0] + 3.0 * action_features[:, 1]
        logits = torch.stack((-score, torch.zeros_like(score), score), dim=-1)
        return torch.stack((logits, logits), dim=0)


class _ContextProbePolicy:
    def __init__(self):
        self.qnet = _ContextSensitiveTwinQ()

    @staticmethod
    def _q_action_features(action, actuator_context):
        assert actuator_context is not None
        return torch.cat((action, actuator_context.detach()), dim=-1)


def test_phase_balanced_sample_covers_sources_and_bins_without_replacement():
    source = torch.tensor([False] * 12 + [True] * 12)
    phase = torch.tensor(
        [0.05, 0.20, 0.38, 0.55, 0.72, 0.90] * 4
    )
    generator = torch.Generator().manual_seed(7)

    indices = phase_balanced_sample_indices(
        source,
        phase,
        rows_per_source=6,
        num_phase_bins=6,
        generator=generator,
    )

    assert indices.numel() == 12
    assert indices.unique().numel() == 12
    assert (~source[indices]).sum().item() == 6
    assert source[indices].sum().item() == 6
    assert phase_bin_indices(phase[indices][~source[indices]], 6).unique().numel() == 6
    assert phase_bin_indices(phase[indices][source[indices]], 6).unique().numel() == 6


def test_matched_decoys_never_cross_source_or_select_self():
    source = torch.tensor([False, False, False, True, True, True])
    phase = torch.tensor([0, 0, 1, 0, 0, 1])
    motion = torch.tensor([3, 3, 4, 8, 8, 9])

    decoy, quality = matched_decoy_indices(
        source,
        phase,
        motion,
        generator=torch.Generator().manual_seed(11),
    )

    assert not torch.equal(decoy, torch.arange(decoy.numel()))
    assert not (decoy == torch.arange(decoy.numel())).any()
    assert torch.equal(source[decoy], source)
    assert quality["exact_source_phase_motion_fraction"] == 4 / 6


def test_distributional_summary_detects_action_and_state_decoy_degradation():
    support = torch.tensor([-1.0, 0.0, 1.0])
    target = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]
    )
    correct = target.clamp_min(1.0e-6).log().unsqueeze(0).repeat(2, 1, 1)
    bad = torch.zeros_like(correct)

    report = summarize_distributional_critic_conditions(
        correct_logits=correct,
        shuffled_action_logits=bad,
        shuffled_state_logits=bad,
        target=target,
        support=support,
        masks={"all": torch.ones(4, dtype=torch.bool)},
    )

    assert report["all"]["correct"]["kl_twin_mean"] < 1.0e-6
    assert (
        report["all"]["shuffled_action_minus_correct"]["kl_delta_twin_mean"]
        > 0.0
    )
    assert (
        report["all"]["shuffled_state_minus_correct"]["positive_fraction"]
        == 1.0
    )


def test_action_gradient_treats_actuator_context_as_fixed_condition():
    context = torch.tensor([[0.2], [-0.4]], requires_grad=True)
    gradient = _action_q_gradients(
        _ContextProbePolicy(),
        torch.zeros(2, 1),
        torch.tensor([[0.1], [0.3]]),
        context,
    )

    assert gradient.shape == (2, 1)
    assert torch.all(gradient > 0.0)
    assert context.grad is None
