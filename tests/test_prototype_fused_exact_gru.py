"""CPU correctness audit for an unenabled experimental inference kernel."""

from __future__ import annotations

import pytest
import torch

from active_adaptation.learning.ppo.ppo_vel import GRU, set_recurrent_mode
from prototype_fused_exact_gru import (
    ExperimentalFusedGRU,
    build_policy_pair,
    compare_full_pipeline,
    comparison_precision,
)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("strict_fp32", [False, True])
def test_full_cnn_pipeline_prefix_and_suffix_compare_at_unchanged_default_tolerances(residual, strict_fp32):
    with comparison_precision(strict_fp32=strict_fp32):
        policy, candidate = build_policy_pair(residual=residual)
        original_keys = set(policy.state_dict())
        report = compare_full_pipeline(policy, candidate)
        for case, fields in report.items():
            for field, result in fields.items():
                assert result["default_assert_close_passed"], (case, field, result)
                assert isinstance(result["torch_equal"], bool)
        assert set(policy.state_dict()) == original_keys
        assert not any(isinstance(module, ExperimentalFusedGRU) for module in policy.modules())
        assert sum(isinstance(module, ExperimentalFusedGRU) for module in candidate.modules()) == 2


def test_fused_candidate_uses_identical_weight_objects_without_mutating_original_module():
    source = GRU(3, 5)
    keys = set(source.state_dict())
    candidate = ExperimentalFusedGRU(source)
    for old, new in (
        ("weight_ih", "weight_ih_l0"), ("weight_hh", "weight_hh_l0"),
        ("bias_ih", "bias_ih_l0"), ("bias_hh", "bias_hh_l0"),
    ):
        assert getattr(candidate.fused, new) is getattr(source.gru, old)
    assert candidate.ln is source.ln
    assert set(source.state_dict()) == keys


def test_candidate_rejects_training_single_step_and_interior_resets():
    core = ExperimentalFusedGRU(GRU(3, 5))
    x = torch.randn(2, 4, 3)
    reset = torch.zeros(2, 4, 1, dtype=torch.bool)
    hx = torch.zeros(2, 4, 5)
    with set_recurrent_mode(True), pytest.raises(RuntimeError, match="no-grad recurrent"):
        core(x, reset, hx)
    with torch.no_grad(), set_recurrent_mode(False), pytest.raises(RuntimeError, match="no-grad recurrent"):
        core(x, reset, hx)
    reset[1, 2] = True
    with torch.no_grad(), set_recurrent_mode(True), pytest.raises(RuntimeError, match="interior resets"):
        core(x, reset, hx)
    with torch.no_grad(), set_recurrent_mode(True), pytest.raises(TypeError, match="boolean reset"):
        core(x, reset.float(), hx)


def test_precision_and_thread_settings_are_restored_even_after_an_exception():
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cudnn.allow_tf32
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="example failure"):
        with comparison_precision(strict_fp32=True):
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.cudnn.allow_tf32 is False
            raise RuntimeError("example failure")
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cudnn.allow_tf32 == tf32
    assert torch.get_num_threads() == threads
