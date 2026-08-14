"""Pure data helpers for the paired FastSAC checkpoint evaluation probe.

This module intentionally has no Isaac Lab dependency.  Keeping record
materialization and summary statistics here makes the diagnostic output easy
to unit-test without launching a simulator.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_EPISODE_META_KEYS = {
    ("episode_len",),
    ("success",),
    ("episode_time_limit",),
    ("command_finished",),
}


def normalize_leaf_key(key: str | tuple[str, ...]) -> tuple[str, ...]:
    """Return a TensorDict leaf key as a non-empty string tuple."""
    normalized = (key,) if isinstance(key, str) else tuple(key)
    if not normalized or not all(isinstance(part, str) and part for part in normalized):
        raise ValueError(f"invalid statistics leaf key: {key!r}")
    return normalized


def validate_fixed_normalized_stds(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and deduplicate a fixed nominal-Q-coordinate std sweep."""
    result: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError("fixed normalized action stds must be numbers")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fixed normalized action stds must be finite and positive")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("the fixed normalized action std sweep must not be empty")
    return tuple(result)


def fixed_std_latent_parameters(
    raw_mean: torch.Tensor,
    actor_center: torch.Tensor,
    actor_scale: torch.Tensor,
    q_scale: torch.Tensor,
    normalized_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a physical Actor proposal and nominal action std to tanh latents.

    ``normalized_std`` is dimensionless in the Critic/BC nominal joint
    coordinates.  Before the tanh transform, the physical proposal noise is
    therefore exactly ``q_scale * normalized_std`` joint by joint, matching
    FastSAC training semantics.
    """
    normalized_std = float(normalized_std)
    if not math.isfinite(normalized_std) or normalized_std <= 0.0:
        raise ValueError("normalized_std must be finite and positive")
    actor_center = torch.as_tensor(actor_center, device=raw_mean.device, dtype=raw_mean.dtype)
    actor_scale = torch.as_tensor(actor_scale, device=raw_mean.device, dtype=raw_mean.dtype)
    q_scale = torch.as_tensor(q_scale, device=raw_mean.device, dtype=raw_mean.dtype)
    if actor_center.ndim != 1 or actor_scale.shape != actor_center.shape:
        raise ValueError("Actor center and scale must be matching vectors")
    if q_scale.shape != actor_scale.shape or raw_mean.shape[-1] != actor_scale.numel():
        raise ValueError("action-coordinate dimensions do not match")
    if not torch.all(actor_scale > 0.0) or not torch.all(q_scale > 0.0):
        raise ValueError("Actor and nominal Q scales must be positive")
    latent_loc = (raw_mean - actor_center) / actor_scale
    latent_scale = (
        q_scale * raw_mean.new_tensor(normalized_std) / actor_scale
    ).expand_as(raw_mean)
    return latent_loc, latent_scale


def _as_python(value: torch.Tensor | Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            scalar = value.reshape(()).item()
            return scalar
        return value.reshape(-1).tolist()
    return value


def _scalar_at(value: torch.Tensor, index: int, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    selected = value[index].detach().reshape(-1)
    if selected.numel() != 1:
        raise ValueError("expected a scalar per-environment diagnostic value")
    return float(selected.item())


def terminal_stats_to_records(
    stat_values: Mapping[str | tuple[str, ...], torch.Tensor],
    *,
    has_done: torch.Tensor,
    done_step: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    metadata: Mapping[str, Any],
    step_dt: float,
    per_env_values: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Materialize one flat, JSON-friendly record per simulator environment."""
    normalized_stats = {
        normalize_leaf_key(key): value.detach().cpu()
        for key, value in stat_values.items()
    }
    has_done = has_done.detach().cpu().bool().reshape(-1)
    done_step = done_step.detach().cpu().long().reshape(-1)
    terminated = terminated.detach().cpu().bool().reshape(-1)
    truncated = truncated.detach().cpu().bool().reshape(-1)
    num_envs = int(has_done.numel())
    if not (
        done_step.numel() == terminated.numel() == truncated.numel() == num_envs
    ):
        raise ValueError("terminal tensors must have the same environment dimension")
    for key, value in normalized_stats.items():
        if value.shape[0] != num_envs:
            raise ValueError(f"statistics leaf {key!r} has the wrong environment count")
    prepared_per_env = {
        str(key): value.detach().cpu()
        for key, value in (per_env_values or {}).items()
    }
    for key, value in prepared_per_env.items():
        if value.shape[0] != num_envs:
            raise ValueError(f"per-environment value {key!r} has the wrong row count")

    episode_len_values = normalized_stats.get(("episode_len",))
    success_values = normalized_stats.get(("success",))
    time_limit_values = normalized_stats.get(("episode_time_limit",))
    command_finished_values = normalized_stats.get(("command_finished",))
    records: list[dict[str, Any]] = []

    for env_index in range(num_envs):
        fallback_len = int(done_step[env_index].item()) + 1
        episode_len = _scalar_at(episode_len_values, env_index, fallback_len)
        divisor = max(episode_len, 1.0)
        success = _scalar_at(success_values, env_index, 0.0)
        episode_time_limit = _scalar_at(time_limit_values, env_index, 0.0) > 0.5
        command_finished = _scalar_at(command_finished_values, env_index, 0.0) > 0.5
        record: dict[str, Any] = dict(metadata)
        record.update(
            {
                "env_index": env_index,
                "completed": bool(has_done[env_index].item()),
                "done_step_index": int(done_step[env_index].item()),
                "episode_len_steps": episode_len,
                "episode_duration_seconds": episode_len * float(step_dt),
                "success": success,
                "terminated": bool(terminated[env_index].item()),
                "truncated": bool(truncated[env_index].item()),
                "episode_time_limit": episode_time_limit,
                "command_finished": command_finished,
            }
        )
        for key, value in prepared_per_env.items():
            record[key] = _as_python(value[env_index])

        termination_causes: list[str] = []
        total_dense_return = 0.0
        for path, values in normalized_stats.items():
            if path in _EPISODE_META_KEYS:
                continue
            selected = values[env_index]
            path_text = "/".join(path)
            if path[0] == "termination":
                cause_value = float(selected.reshape(-1)[0].item())
                active = cause_value > 0.5
                record[f"termination/{'/'.join(path[1:])}"] = active
                if active:
                    termination_causes.append("/".join(path[1:]))
                continue
            cumulative = _as_python(selected)
            if isinstance(cumulative, list):
                record[f"stats/{path_text}"] = cumulative
                continue
            cumulative = float(cumulative)
            record[f"reward_cumulative/{path_text}"] = cumulative
            record[f"reward_per_step/{path_text}"] = cumulative / divisor
            if len(path) >= 2 and path[-1] == "return":
                total_dense_return += cumulative

        if bool(terminated[env_index].item()) and not termination_causes:
            termination_causes.append("unknown")
        record["termination_causes"] = termination_causes
        if termination_causes:
            record["terminal_class"] = "+".join(termination_causes)
        elif command_finished:
            record["terminal_class"] = "command_finished"
        elif episode_time_limit:
            record["terminal_class"] = "episode_time_limit"
        elif bool(has_done[env_index].item()):
            record["terminal_class"] = "other_truncation"
        else:
            record["terminal_class"] = "incomplete"
        record["total_dense_return"] = total_dense_return
        record["total_dense_return_per_step"] = total_dense_return / divisor
        transition_return = record.get("probe/undiscounted_transition_dense_return")
        if isinstance(transition_return, (int, float)) and not isinstance(
            transition_return, bool
        ):
            record["probe/reward_stats_consistency_error"] = (
                float(transition_return) - total_dense_return
            )
        records.append(record)
    return records


def binary_roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Compute tie-aware binary ROC AUC as a Mann-Whitney probability."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = torch.tensor(
        [float(score) for score, label in zip(scores, labels) if float(label) > 0.5],
        dtype=torch.float64,
    )
    negatives = torch.tensor(
        [float(score) for score, label in zip(scores, labels) if float(label) <= 0.5],
        dtype=torch.float64,
    )
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(((comparisons > 0).double() + 0.5 * (comparisons == 0).double()).mean())


def _mean_std(values: Sequence[float]) -> dict[str, float | None]:
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    if tensor.numel() == 0:
        return {"mean": None, "std": None}
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
    }


def summarize_condition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one checkpoint/seed/action-noise evaluation condition."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation condition")
    success = [float(record["success"]) for record in records]
    successes = sum(value > 0.5 for value in success)
    n = len(records)
    p = successes / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    result: dict[str, Any] = {
        "num_envs": n,
        "completed": sum(bool(record["completed"]) for record in records),
        "success_count": successes,
        "success_rate": p,
        "success_wilson_95_low": max(0.0, center - radius),
        "success_wilson_95_high": min(1.0, center + radius),
        "episode_len_steps": _mean_std(
            [float(record["episode_len_steps"]) for record in records]
        ),
        "total_dense_return": _mean_std(
            [float(record["total_dense_return"]) for record in records]
        ),
        "total_dense_return_per_step": _mean_std(
            [float(record["total_dense_return_per_step"]) for record in records]
        ),
        "dense_return_success_auc": binary_roc_auc(
            [float(record["total_dense_return"]) for record in records], success
        ),
        "terminal_class_counts": dict(
            sorted(Counter(str(record["terminal_class"]) for record in records).items())
        ),
    }
    successful_returns = [
        float(record["total_dense_return"])
        for record in records
        if float(record["success"]) > 0.5
    ]
    failed_returns = [
        float(record["total_dense_return"])
        for record in records
        if float(record["success"]) <= 0.5
    ]
    result["dense_return_given_success"] = _mean_std(successful_returns)
    result["dense_return_given_failure"] = _mean_std(failed_returns)

    discounted_return_keys = {
        "discounted_dense_return": "probe/discounted_dense_return",
        "q_effective_discounted_dense_return": (
            "probe/q_effective_discounted_dense_return"
        ),
    }
    for summary_prefix, record_key in discounted_return_keys.items():
        if not all(record_key in record for record in records):
            continue
        discounted_returns = [float(record[record_key]) for record in records]
        result[summary_prefix] = _mean_std(discounted_returns)
        result[f"{summary_prefix}_success_auc"] = binary_roc_auc(
            discounted_returns, success
        )
        result[f"{summary_prefix}_given_success"] = _mean_std(
            [
                value
                for value, label in zip(discounted_returns, success)
                if label > 0.5
            ]
        )
        result[f"{summary_prefix}_given_failure"] = _mean_std(
            [
                value
                for value, label in zip(discounted_returns, success)
                if label <= 0.5
            ]
        )

    scalar_metric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if (
                key.startswith("reward_per_step/")
                or key.startswith("probe/")
            )
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    result["metrics"] = {
        key: _mean_std([float(record[key]) for record in records if key in record])
        for key in scalar_metric_keys
    }
    return result


def paired_against_deterministic(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every noisy condition to matching deterministic environment rows."""
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["checkpoint"]),
            int(record["evaluation_seed"]),
            str(record["mode"]),
        )
        grouped[key].append(record)

    output: dict[str, Any] = {}
    for (checkpoint, seed, mode), candidate_records in sorted(grouped.items()):
        if mode == "deterministic":
            continue
        baseline_records = grouped.get((checkpoint, seed, "deterministic"), [])
        baseline = {int(record["env_index"]): record for record in baseline_records}
        candidate = {int(record["env_index"]): record for record in candidate_records}
        common = sorted(set(baseline).intersection(candidate))
        if not common:
            continue
        baseline_success = [float(baseline[index]["success"]) > 0.5 for index in common]
        candidate_success = [float(candidate[index]["success"]) > 0.5 for index in common]
        metric_keys = sorted(
            {
                key
                for index in common
                for key, value in candidate[index].items()
                if (
                    key.startswith("reward_per_step/")
                    or key.startswith("probe/")
                    or key in {"total_dense_return", "total_dense_return_per_step"}
                )
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key in baseline[index]
            }
        )
        label = f"{checkpoint}|seed={seed}|{mode}"
        output[label] = {
            "num_pairs": len(common),
            "success_rate_delta": (
                sum(candidate_success) - sum(baseline_success)
            ) / len(common),
            "improved_failure_to_success": sum(
                (not before) and after
                for before, after in zip(baseline_success, candidate_success)
            ),
            "worsened_success_to_failure": sum(
                before and (not after)
                for before, after in zip(baseline_success, candidate_success)
            ),
            "mean_paired_metric_delta": {
                key: sum(
                    float(candidate[index][key]) - float(baseline[index][key])
                    for index in common
                )
                / len(common)
                for key in metric_keys
            },
        }
    return output
