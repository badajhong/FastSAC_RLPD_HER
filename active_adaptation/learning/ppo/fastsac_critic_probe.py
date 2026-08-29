"""Pure helpers for the read-only FastSAC critic causality probe.

The simulator-facing runner lives in :mod:`scripts.fastsac_critic_probe`.  This
module deliberately contains only tensor operations so the important
phase/source matching and distributional diagnostics can be regression tested
without Isaac Sim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


CRITIC_PROBE_SCHEMA = "fastsac_critic_causality_probe_v1"


def _finite_float(value: torch.Tensor | float) -> float:
    result = float(torch.as_tensor(value).detach().item())
    if not math.isfinite(result):
        raise RuntimeError("critic probe produced a non-finite scalar")
    return result


def phase_bin_indices(phase: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Map a finite normalized reference phase to ``[0, num_bins)``."""
    if isinstance(num_bins, bool) or int(num_bins) < 1:
        raise ValueError("num_bins must be a positive integer")
    phase = torch.as_tensor(phase).detach().float().reshape(-1)
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("reference phase contains NaN/Inf")
    tolerance = 16.0 * torch.finfo(phase.dtype).eps
    if bool(((phase < -tolerance) | (phase > 1.0 + tolerance)).any()):
        raise ValueError("reference phase must lie in [0, 1]")
    return (phase.clamp(0.0, 1.0) * int(num_bins)).long().clamp_max(
        int(num_bins) - 1
    )


def phase_balanced_sample_indices(
    source_is_teacher: torch.Tensor,
    phase: torch.Tensor,
    *,
    rows_per_source: int,
    num_phase_bins: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample without replacement while covering every available phase bin."""
    source = torch.as_tensor(source_is_teacher).detach().reshape(-1).bool().cpu()
    bins = phase_bin_indices(phase, num_phase_bins).cpu()
    if source.shape != bins.shape:
        raise ValueError("source and phase tensors must have identical shape")
    if isinstance(rows_per_source, bool) or int(rows_per_source) < 1:
        raise ValueError("rows_per_source must be a positive integer")
    if generator.device.type != "cpu":
        raise ValueError("phase-balanced sampling requires a CPU generator")

    selected: list[torch.Tensor] = []
    for teacher_value in (False, True):
        source_rows = (source == teacher_value).nonzero(as_tuple=False).squeeze(-1)
        requested = min(int(rows_per_source), int(source_rows.numel()))
        if requested < 1:
            label = "Teacher" if teacher_value else "Student"
            raise ValueError(f"critic probe has no {label} source rows")

        quota, remainder = divmod(requested, int(num_phase_bins))
        source_selected: list[torch.Tensor] = []
        for phase_bin in range(int(num_phase_bins)):
            candidates = source_rows[bins.index_select(0, source_rows) == phase_bin]
            count = min(
                int(candidates.numel()),
                quota + int(phase_bin < remainder),
            )
            if count:
                order = torch.randperm(
                    candidates.numel(), generator=generator
                )[:count]
                source_selected.append(candidates.index_select(0, order))

        chosen = (
            torch.cat(source_selected)
            if source_selected
            else torch.empty(0, dtype=torch.long)
        )
        shortfall = requested - int(chosen.numel())
        if shortfall:
            available = torch.ones(source.shape[0], dtype=torch.bool)
            available[source != teacher_value] = False
            available[chosen] = False
            candidates = available.nonzero(as_tuple=False).squeeze(-1)
            order = torch.randperm(candidates.numel(), generator=generator)
            chosen = torch.cat(
                (chosen, candidates.index_select(0, order[:shortfall]))
            )
        selected.append(chosen)

    result = torch.cat(selected)
    return result.index_select(
        0, torch.randperm(result.numel(), generator=generator)
    )


def matched_decoy_indices(
    source_is_teacher: torch.Tensor,
    phase_bins: torch.Tensor,
    motion_ids: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Choose a non-self decoy, preferring source+phase+motion matches.

    This is intentionally a row-wise map rather than a global permutation.
    Sparse motion/phase buckets may reuse a decoy, but no row is paired across
    Teacher/Student provenance and no row is ever paired with itself.
    """
    source = torch.as_tensor(source_is_teacher).detach().reshape(-1).bool().cpu()
    phase = torch.as_tensor(phase_bins).detach().reshape(-1).long().cpu()
    motion = torch.as_tensor(motion_ids).detach().reshape(-1).long().cpu()
    if source.shape != phase.shape or source.shape != motion.shape:
        raise ValueError("decoy matching fields must have identical shape")
    if generator.device.type != "cpu":
        raise ValueError("decoy matching requires a CPU generator")

    result = torch.full((source.numel(),), -1, dtype=torch.long)
    exact = 0
    phase_only = 0
    source_only = 0
    for index in range(source.numel()):
        same_source = source == source[index]
        not_self = torch.arange(source.numel()) != index
        exact_candidates = (
            same_source
            & (phase == phase[index])
            & (motion == motion[index])
            & not_self
        ).nonzero(as_tuple=False).squeeze(-1)
        phase_candidates = (
            same_source & (phase == phase[index]) & not_self
        ).nonzero(as_tuple=False).squeeze(-1)
        source_candidates = (same_source & not_self).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if exact_candidates.numel():
            candidates = exact_candidates
            exact += 1
        elif phase_candidates.numel():
            candidates = phase_candidates
            phase_only += 1
        elif source_candidates.numel():
            candidates = source_candidates
            source_only += 1
        else:
            raise ValueError("each replay source needs at least two probe rows")
        draw = torch.randint(
            candidates.numel(), (1,), generator=generator
        ).item()
        result[index] = candidates[int(draw)]

    if bool((result == torch.arange(result.numel())).any()):
        raise RuntimeError("critic decoy map contains a fixed point")
    if not torch.equal(source.index_select(0, result), source):
        raise RuntimeError("critic decoy map crossed replay sources")
    denominator = float(max(1, source.numel()))
    return result, {
        "exact_source_phase_motion_fraction": exact / denominator,
        "source_phase_fallback_fraction": phase_only / denominator,
        "source_only_fallback_fraction": source_only / denominator,
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) <= torch.finfo(left.dtype).eps:
        return None
    return _finite_float((left * right).sum() / denominator)


def _condition_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, Any]:
    if logits.ndim != 3 or logits.shape[0] != 2:
        raise ValueError("critic logits must have shape [2, rows, atoms]")
    if target.shape != logits.shape[1:]:
        raise ValueError("critic target does not match logits")
    log_prob = F.log_softmax(logits.float(), dim=-1)
    target = target.detach().float()
    target_entropy = -(
        target * target.clamp_min(torch.finfo(target.dtype).tiny).log()
    ).sum(dim=-1)
    cross_entropy = -(target.unsqueeze(0) * log_prob).sum(dim=-1)
    kl = cross_entropy - target_entropy.unsqueeze(0)
    q = (log_prob.exp() * support.float()).sum(dim=-1)
    target_q = (target * support.float()).sum(dim=-1)
    error = q - target_q.unsqueeze(0)
    return {
        "cross_entropy_per_head": cross_entropy,
        "kl_per_head": kl,
        "q_per_head": q,
        "target_q": target_q,
        "error_per_head": error,
    }


def summarize_distributional_critic_conditions(
    *,
    correct_logits: torch.Tensor,
    shuffled_action_logits: torch.Tensor,
    shuffled_state_logits: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize held-out target fit and paired decoy degradation."""
    conditions = {
        "correct": _condition_metrics(correct_logits, target, support),
        "shuffled_action": _condition_metrics(
            shuffled_action_logits, target, support
        ),
        "shuffled_state": _condition_metrics(
            shuffled_state_logits, target, support
        ),
    }
    report: dict[str, Any] = {}
    row_count = int(target.shape[0])
    for name, raw_mask in masks.items():
        mask = torch.as_tensor(raw_mask, device=target.device).reshape(-1).bool()
        if mask.shape != (row_count,):
            raise ValueError(f"critic stratum {name!r} has the wrong shape")
        if not bool(mask.any()):
            continue
        stratum: dict[str, Any] = {"rows": int(mask.sum().item())}
        for condition_name, values in conditions.items():
            kl = values["kl_per_head"][:, mask]
            error = values["error_per_head"][:, mask]
            q = values["q_per_head"][:, mask]
            target_q = values["target_q"][mask]
            stratum[condition_name] = {
                "kl_head1_mean": _finite_float(kl[0].mean()),
                "kl_head2_mean": _finite_float(kl[1].mean()),
                "kl_twin_mean": _finite_float(kl.mean()),
                "expected_q_head1_mean": _finite_float(q[0].mean()),
                "expected_q_head2_mean": _finite_float(q[1].mean()),
                "expected_target_mean": _finite_float(target_q.mean()),
                "td_bias_head1": _finite_float(error[0].mean()),
                "td_bias_head2": _finite_float(error[1].mean()),
                "td_mae_twin_mean": _finite_float(error.abs().mean()),
                "td_rmse_twin_mean": _finite_float(error.square().mean().sqrt()),
                "expected_q_target_pearson_head1": _pearson(q[0], target_q),
                "expected_q_target_pearson_head2": _pearson(q[1], target_q),
            }
        correct_kl = conditions["correct"]["kl_per_head"][:, mask]
        for decoy_name in ("shuffled_action", "shuffled_state"):
            decoy_kl = conditions[decoy_name]["kl_per_head"][:, mask]
            delta = decoy_kl - correct_kl
            stratum[f"{decoy_name}_minus_correct"] = {
                "kl_delta_twin_mean": _finite_float(delta.mean()),
                "kl_delta_twin_median": _finite_float(delta.median()),
                "positive_fraction": _finite_float((delta > 0.0).float().mean()),
            }
        report[name] = stratum
    return report
