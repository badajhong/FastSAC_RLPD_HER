import math

import pytest
import torch

from scripts.fastsac_eval_probe_utils import (
    binary_roc_auc,
    fixed_std_latent_parameters,
    paired_against_deterministic,
    summarize_condition,
    terminal_stats_to_records,
    validate_fixed_normalized_stds,
)


def test_fixed_std_parameters_preserve_nominal_unsquashed_noise_scale():
    raw_mean = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    actor_center = torch.tensor([0.0, 1.0])
    actor_scale = torch.tensor([2.0, 4.0])
    q_scale = torch.tensor([0.5, 3.0])
    latent_loc, latent_scale = fixed_std_latent_parameters(
        raw_mean, actor_center, actor_scale, q_scale, 0.05
    )
    torch.testing.assert_close(latent_loc, (raw_mean - actor_center) / actor_scale)
    torch.testing.assert_close(
        latent_scale * actor_scale,
        (q_scale * 0.05).expand_as(raw_mean),
    )


def test_fixed_std_validation_rejects_zero_and_deduplicates():
    assert validate_fixed_normalized_stds([0.02, 0.05, 0.02]) == (0.02, 0.05)
    with pytest.raises(ValueError, match="positive"):
        validate_fixed_normalized_stds([0.0])


def _terminal_fixture():
    stats = {
        "episode_len": torch.tensor([[100.0], [50.0]]),
        "success": torch.tensor([[1.0], [0.0]]),
        "command_finished": torch.tensor([[1.0], [0.0]]),
        "episode_time_limit": torch.zeros(2, 1),
        ("tracking", "return"): torch.tensor([[20.0], [5.0]]),
        ("tracking", "tracking_body_angvel"): torch.tensor([[40.0], [10.0]]),
        ("termination", "cum_body_pos_error"): torch.tensor([[0.0], [1.0]]),
    }
    return terminal_stats_to_records(
        stats,
        has_done=torch.tensor([True, True]),
        done_step=torch.tensor([99, 49]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([True, False]),
        metadata={
            "checkpoint": "checkpoint_1",
            "evaluation_seed": 0,
            "mode": "deterministic",
        },
        step_dt=0.02,
        per_env_values={
            "probe/mean_normalized_action_deviation_rms": torch.tensor([0.0, 0.1]),
            "probe/undiscounted_transition_dense_return": torch.tensor([20.0, 5.0]),
            "probe/discounted_dense_return": torch.tensor([12.0, 4.0]),
            "probe/q_effective_discounted_dense_return": torch.tensor([11.0, 3.0]),
        },
    )


def test_terminal_records_keep_raw_and_per_step_reward_and_causes():
    records = _terminal_fixture()
    assert records[0]["terminal_class"] == "command_finished"
    assert records[1]["terminal_class"] == "cum_body_pos_error"
    assert records[1]["termination/cum_body_pos_error"] is True
    assert records[0]["reward_cumulative/tracking/tracking_body_angvel"] == 40.0
    assert records[0]["reward_per_step/tracking/tracking_body_angvel"] == 0.4
    assert records[1]["total_dense_return"] == 5.0
    assert records[1]["episode_duration_seconds"] == 1.0
    assert records[0]["probe/reward_stats_consistency_error"] == 0.0


def test_summary_reports_success_uncertainty_and_dense_return_auc():
    records = _terminal_fixture()
    summary = summarize_condition(records)
    assert summary["success_rate"] == 0.5
    assert summary["dense_return_success_auc"] == 1.0
    assert summary["discounted_dense_return_success_auc"] == 1.0
    assert summary["discounted_dense_return_given_success"]["mean"] == 12.0
    assert summary["discounted_dense_return_given_failure"]["mean"] == 4.0
    assert summary["q_effective_discounted_dense_return_success_auc"] == 1.0
    assert summary["q_effective_discounted_dense_return_given_success"]["mean"] == 11.0
    assert summary["terminal_class_counts"] == {
        "command_finished": 1,
        "cum_body_pos_error": 1,
    }
    assert math.isclose(
        summary["metrics"]["reward_per_step/tracking/tracking_body_angvel"]["mean"],
        0.3,
    )
    assert binary_roc_auc([1.0, 1.0], [0.0, 1.0]) == 0.5


def test_paired_summary_uses_matching_environment_indices():
    baseline = _terminal_fixture()
    noisy = [dict(record) for record in baseline]
    for record in noisy:
        record["mode"] = "fixed_std_0.05"
    noisy[0]["success"] = 0.0
    noisy[0]["reward_per_step/tracking/tracking_body_angvel"] = 0.2
    result = paired_against_deterministic([*baseline, *noisy])
    paired = result["checkpoint_1|seed=0|fixed_std_0.05"]
    assert paired["num_pairs"] == 2
    assert paired["success_rate_delta"] == -0.5
    assert paired["worsened_success_to_failure"] == 1
    assert paired["improved_failure_to_success"] == 0
    assert math.isclose(
        paired["mean_paired_metric_delta"][
            "reward_per_step/tracking/tracking_body_angvel"
        ],
        -0.1,
    )
