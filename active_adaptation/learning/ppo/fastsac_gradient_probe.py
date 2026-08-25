"""Read-only Actor-gradient diagnostics for FastSAC Teacher-BC.

The training checkpoints intentionally omit their replay rings.  This module
therefore operates on one *prepared* Actor batch, whether that batch came from
live diagnostic collection or a focused unit seam.  It mirrors the production
Actor objective while keeping its three causes separate:

* exact mean-action Teacher BC;
* the negative pessimistic twin-C51 expectation; and
* the temperature-weighted policy log probability.

Only :func:`torch.autograd.grad` is used.  No optimizer is zeroed or stepped,
``Parameter.grad`` is never populated, and the policy sampling generators are
not consumed.  This makes the probe safe to run against an inference-loaded
checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from .fastsac_vel import _reduce_actor_q_values
from .ppo_bc_dagger import (
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from .td3_bc_dagger import (
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    _exact_teacher_bc_loss,
)


GRADIENT_PROBE_SCHEMA = "fastsac_actor_gradient_probe_v1"


def _finite_float(value: torch.Tensor | float) -> float:
    result = float(torch.as_tensor(value).detach().item())
    if not math.isfinite(result):
        raise RuntimeError("FastSAC gradient probe produced a non-finite scalar")
    return result


def _parameter_groups(policy) -> tuple[tuple[torch.nn.Parameter, ...], ...]:
    """Return the production mean and policy-scale optimizer groups."""
    uses_physical = bool(
        getattr(policy, "_uses_ppo_physical_gaussian", lambda: False)()
    )
    optimizer = getattr(policy, "actor_optimizer", None)
    std_optimizer = getattr(policy, "actor_std_optimizer", None)
    if uses_physical and optimizer is not None and std_optimizer is not None:
        mean_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        scale_parameters = tuple(
            parameter
            for group in std_optimizer.param_groups
            for parameter in group["params"]
        )
    elif optimizer is not None and len(optimizer.param_groups) >= 2:
        mean_parameters = tuple(optimizer.param_groups[0]["params"])
        scale_parameters = tuple(optimizer.param_groups[1]["params"])
    else:
        scale_parameters = (
            (policy._ppo_actor_std_parameter(),)
            if uses_physical
            else tuple(policy.bc_dagger_sac_adapter.parameters())
        )
        scale_ids = {id(parameter) for parameter in scale_parameters}
        mean_parameters = tuple(
            parameter
            for parameter in policy.actor_adapt.parameters()
            if id(parameter) not in scale_ids
        )
    if not mean_parameters:
        raise RuntimeError("FastSAC gradient probe found no Actor mean parameters")
    if not scale_parameters:
        raise RuntimeError("FastSAC gradient probe found no policy-scale parameter")

    all_parameters = mean_parameters + scale_parameters
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise RuntimeError("FastSAC Actor diagnostic parameter groups overlap")
    return mean_parameters, scale_parameters, all_parameters


@contextmanager
def _temporarily_enable_grad(parameters: Sequence[torch.nn.Parameter]):
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def _gradient_tuple(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients)
    )


def _add_gradients(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    left_scale: float = 1.0,
    right_scale: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        left_scale * left_value + right_scale * right_value
        for left_value, right_value in zip(left, right)
    )


def _gradient_norm(gradients: Sequence[torch.Tensor], indices: range) -> float:
    squared = None
    for index in indices:
        value = gradients[index]
        contribution = value.float().square().sum()
        squared = contribution if squared is None else squared + contribution
    if squared is None:
        return 0.0
    return _finite_float(squared.sqrt())


def _gradient_cosine(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    indices: range,
) -> float | None:
    dot = None
    left_squared = None
    right_squared = None
    for index in indices:
        left_value = left[index].float()
        right_value = right[index].float()
        dot_value = (left_value * right_value).sum()
        left_value_squared = left_value.square().sum()
        right_value_squared = right_value.square().sum()
        dot = dot_value if dot is None else dot + dot_value
        left_squared = (
            left_value_squared
            if left_squared is None
            else left_squared + left_value_squared
        )
        right_squared = (
            right_value_squared
            if right_squared is None
            else right_squared + right_value_squared
        )
    if dot is None or left_squared is None or right_squared is None:
        return None
    denominator = (left_squared * right_squared).sqrt()
    if _finite_float(denominator) <= torch.finfo(torch.float32).eps:
        return None
    return _finite_float((dot / denominator).clamp(-1.0, 1.0))


def _expected_twin_q(policy, critic_observations, actions):
    logits = policy.qnet(
        critic_observations,
        policy._q_action_input(actions),
    )
    expected = (F.softmax(logits, dim=-1) * policy.qnet.support).sum(dim=-1)
    minimum = _reduce_actor_q_values(expected, True)
    return expected, minimum


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value.reshape(-1)[mask]
    if not selected.numel():
        raise ValueError("cannot average an empty diagnostic stratum")
    return selected.mean()


def _scalar_stratum_metrics(
    mask: torch.Tensor,
    *,
    exact_bc_loss: torch.Tensor,
    minimum_mean_q: torch.Tensor,
    minimum_sample_q: torch.Tensor,
    minimum_teacher_q: torch.Tensor,
    normalized_log_prob: torch.Tensor,
    teacher_valid: torch.Tensor,
    mean_action: torch.Tensor,
    sampled_action: torch.Tensor,
    teacher_action: torch.Tensor,
    action_scale: torch.Tensor,
) -> dict[str, float | int | None]:
    valid = mask & teacher_valid
    metrics: dict[str, float | int | None] = {
        "rows": int(mask.sum().item()),
        "valid_teacher_rows": int(valid.sum().item()),
        "bc_loss": _finite_float(exact_bc_loss),
        "q_policy_mean": _finite_float(_masked_mean(minimum_mean_q, mask)),
        "q_policy_sample": _finite_float(_masked_mean(minimum_sample_q, mask)),
        "normalized_log_prob": _finite_float(
            _masked_mean(normalized_log_prob, mask)
        ),
        "sample_mean_q_action_deviation_rms": _finite_float(
            (((sampled_action[mask] - mean_action[mask]) / action_scale).square())
            .mean()
            .sqrt()
        ),
    }
    if bool(valid.any()):
        q_mean_minus_teacher = minimum_mean_q[valid] - minimum_teacher_q[valid]
        q_sample_minus_teacher = (
            minimum_sample_q[valid] - minimum_teacher_q[valid]
        )
        metrics.update(
            {
                "q_teacher": _finite_float(minimum_teacher_q[valid].mean()),
                "q_policy_mean_minus_teacher": _finite_float(
                    q_mean_minus_teacher.mean()
                ),
                "q_policy_sample_minus_teacher": _finite_float(
                    q_sample_minus_teacher.mean()
                ),
                "policy_mean_q_above_teacher_fraction": _finite_float(
                    (q_mean_minus_teacher > 0).float().mean()
                ),
                "mean_teacher_q_action_deviation_rms": _finite_float(
                    (((mean_action[valid] - teacher_action[valid]) / action_scale)
                    .square())
                    .mean()
                    .sqrt()
                ),
            }
        )
    else:
        metrics.update(
            {
                "q_teacher": None,
                "q_policy_mean_minus_teacher": None,
                "q_policy_sample_minus_teacher": None,
                "policy_mean_q_above_teacher_fraction": None,
                "mean_teacher_q_action_deviation_rms": None,
            }
        )
    return metrics


def _action_q_gradients(policy, critic_observations, action):
    leaf = action.detach().clone().requires_grad_(True)
    _, minimum = _expected_twin_q(policy, critic_observations, leaf)
    # Rows are independent, so differentiating the sum returns one unscaled
    # dQ/da vector for every row.
    gradient = torch.autograd.grad(minimum.sum(), leaf, create_graph=False)[0]
    return gradient.detach().float()


def diagnose_fastsac_actor_gradients(
    policy,
    batch: Mapping[str, torch.Tensor],
    *,
    sample_seed: int = 0,
    source_gradients: bool = True,
    finite_difference_epsilon: float = 1.0e-3,
) -> dict[str, Any]:
    """Measure the production Actor objective without changing the policy.

    Args:
        policy: A constructed ``DistributionalFastSACTeacherBC``-compatible
            policy.  It may be inference-frozen.
        batch: One batch already passed through
            ``policy._prepare_dagger_learning_batch``.  Required fields are
            ``observations``, ``critic_observations``, Teacher actions, and
            their validity mask.
        sample_seed: Local diagnostic reparameterization seed.  The policy's
            SAC sampling generators are deliberately not touched.
        source_gradients: Also run separate backward probes for Student,
            Teacher, and failure-focused source masks when present.
        finite_difference_epsilon: Central-difference step in unit-normalized
            Q-action coordinates for the Teacher direction check.

    Returns:
        A JSON-serializable nested diagnostic mapping.
    """
    required = {
        "observations",
        "critic_observations",
        DAGGER_REPLAY_TEACHER_ACTIONS,
        DAGGER_TEACHER_ACTION_VALID_KEY,
    }
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"FastSAC gradient probe batch is missing {sorted(missing)}")
    if (
        not math.isfinite(float(finite_difference_epsilon))
        or float(finite_difference_epsilon) <= 0.0
    ):
        raise ValueError("finite_difference_epsilon must be finite and positive")

    observations = batch["observations"]
    critic_observations = batch["critic_observations"]
    teacher_action = batch[DAGGER_REPLAY_TEACHER_ACTIONS]
    teacher_valid = batch[DAGGER_TEACHER_ACTION_VALID_KEY].reshape(-1).bool()
    row_count = int(observations.shape[0])
    if row_count < 1:
        raise ValueError("FastSAC gradient probe requires a non-empty batch")
    if (
        critic_observations.shape[0] != row_count
        or teacher_action.shape[0] != row_count
        or teacher_valid.shape != (row_count,)
    ):
        raise ValueError("FastSAC gradient probe batch fields are misaligned")
    if not torch.isfinite(observations).all() or not torch.isfinite(
        critic_observations
    ).all():
        raise ValueError("FastSAC gradient probe observations are non-finite")

    mean_parameters, log_std_parameters, all_parameters = _parameter_groups(policy)
    mean_count = len(mean_parameters)
    group_indices = {
        "actor_mean": range(0, mean_count),
        "global_log_std": range(mean_count, len(all_parameters)),
        "all_actor": range(0, len(all_parameters)),
    }
    device = observations.device
    generator = torch.Generator(device=device).manual_seed(int(sample_seed))
    lambda_bc = float(policy.cfg.lambda_bc)
    eta_sac = float(policy.cfg.eta_sac)
    alpha = policy.log_alpha.detach().exp()

    with torch.enable_grad(), _temporarily_enable_grad(all_parameters):
        raw_prediction = policy._actor_mean_from_flat(observations)
        if not torch.isfinite(raw_prediction).all():
            raise RuntimeError("FastSAC gradient probe Actor mean is non-finite")
        distribution = policy._sac_dist_from_mean(raw_prediction)
        mean_action = distribution.mean
        sampled_action, raw_log_prob = distribution.rsample_with_log_prob(
            generator=generator
        )
        normalized_log_prob = policy._normalized_action_log_prob(raw_log_prob)
        twin_sample_q, minimum_sample_q = _expected_twin_q(
            policy, critic_observations, sampled_action
        )
        with torch.no_grad():
            twin_mean_q, minimum_mean_q = _expected_twin_q(
                policy, critic_observations, mean_action.detach()
            )
            finite_teacher_action = torch.nan_to_num(
                teacher_action, nan=0.0, posinf=0.0, neginf=0.0
            )
            twin_teacher_q, minimum_teacher_q = _expected_twin_q(
                policy, critic_observations, finite_teacher_action
            )

        teacher_source = batch.get(DAGGER_Q_TEACHER_SOURCE_KEY)
        if teacher_source is None:
            teacher_source = torch.zeros(
                row_count, dtype=torch.bool, device=device
            )
            source_available = False
        else:
            teacher_source = teacher_source.reshape(-1).bool()
            source_available = True
        failure_source = batch.get(FAILURE_PHASE_TEACHER_SOURCE_KEY)
        if failure_source is None:
            failure_source = torch.zeros_like(teacher_source)
            failure_available = False
        else:
            failure_source = failure_source.reshape(-1).bool()
            failure_available = True
        if teacher_source.shape != (row_count,) or failure_source.shape != (
            row_count,
        ):
            raise ValueError("FastSAC diagnostic source masks are misaligned")
        if bool((failure_source & ~teacher_source).any()):
            raise ValueError("failure-focused rows must be a subset of Teacher rows")

        all_mask = torch.ones(row_count, dtype=torch.bool, device=device)
        masks: dict[str, torch.Tensor] = {"all": all_mask}
        if source_available:
            masks.update(
                {
                    "student": ~teacher_source,
                    "teacher": teacher_source,
                }
            )
        if failure_available:
            masks.update(
                {
                    "uniform_teacher": teacher_source & ~failure_source,
                    "failure_teacher": failure_source,
                }
            )
        masks = {name: mask for name, mask in masks.items() if bool(mask.any())}

        gradient_masks = masks if source_gradients else {"all": all_mask}
        gradient_reports = {}
        scalar_reports = {}
        saw_all_gradients = False
        for name, mask in masks.items():
            bc_loss = _exact_teacher_bc_loss(
                mean_action[mask],
                teacher_action[mask],
                teacher_valid[mask],
                policy._fastsac_q_action_center,
                policy._fastsac_q_action_scale,
                float(policy.cfg.dagger_actor_huber_delta),
            )
            scalar_reports[name] = _scalar_stratum_metrics(
                mask,
                exact_bc_loss=bc_loss,
                minimum_mean_q=minimum_mean_q,
                minimum_sample_q=minimum_sample_q.detach(),
                minimum_teacher_q=minimum_teacher_q,
                normalized_log_prob=normalized_log_prob.detach(),
                teacher_valid=teacher_valid,
                mean_action=mean_action.detach(),
                sampled_action=sampled_action.detach(),
                teacher_action=finite_teacher_action,
                action_scale=policy._fastsac_q_action_scale,
            )
            if name not in gradient_masks:
                continue
            q_loss = -_masked_mean(minimum_sample_q, mask)
            entropy_loss = alpha * _masked_mean(normalized_log_prob, mask)
            gradients = {
                "bc": _gradient_tuple(bc_loss, all_parameters),
                "q": _gradient_tuple(q_loss, all_parameters),
                "entropy": _gradient_tuple(entropy_loss, all_parameters),
            }
            gradients["sac"] = _add_gradients(
                gradients["q"], gradients["entropy"]
            )
            gradients["total"] = _add_gradients(
                gradients["bc"],
                gradients["sac"],
                left_scale=lambda_bc,
                right_scale=eta_sac,
            )
            if name == "all":
                saw_all_gradients = True
            groups = {}
            for group_name, indices in group_indices.items():
                raw_norms = {
                    component: _gradient_norm(gradients[component], indices)
                    for component in ("bc", "q", "entropy", "sac")
                }
                groups[group_name] = {
                    "unweighted_norms": raw_norms,
                    "effective_norms": {
                        "bc": abs(lambda_bc) * raw_norms["bc"],
                        "q": abs(eta_sac) * raw_norms["q"],
                        "entropy": abs(eta_sac) * raw_norms["entropy"],
                        "sac": abs(eta_sac) * raw_norms["sac"],
                        "total": _gradient_norm(gradients["total"], indices),
                    },
                    "cosines": {
                        "bc_q": _gradient_cosine(
                            gradients["bc"], gradients["q"], indices
                        ),
                        "bc_entropy": _gradient_cosine(
                            gradients["bc"], gradients["entropy"], indices
                        ),
                        "bc_sac": _gradient_cosine(
                            gradients["bc"], gradients["sac"], indices
                        ),
                        "q_entropy": _gradient_cosine(
                            gradients["q"], gradients["entropy"], indices
                        ),
                    },
                }
            gradient_reports[name] = {
                "losses": {
                    "bc": _finite_float(bc_loss),
                    "negative_q": _finite_float(q_loss),
                    "alpha_log_prob": _finite_float(entropy_loss),
                    "sac": _finite_float(q_loss + entropy_loss),
                    "weighted_bc": _finite_float(lambda_bc * bc_loss),
                    "weighted_sac": _finite_float(
                        eta_sac * (q_loss + entropy_loss)
                    ),
                    "total": _finite_float(
                        lambda_bc * bc_loss + eta_sac * (q_loss + entropy_loss)
                    ),
                },
                "groups": groups,
            }

        if not saw_all_gradients:  # pragma: no cover - all is non-empty
            raise RuntimeError("FastSAC gradient probe lost its all-row gradients")

        # dQ/da is intentionally computed from detached physical actions.  It
        # therefore tests the Critic's local action sensitivity independently
        # of both the Actor Jacobian and the BC objective.
        sample_dqda = _action_q_gradients(
            policy, critic_observations, sampled_action
        )
        mean_dqda = _action_q_gradients(policy, critic_observations, mean_action)
        teacher_dqda = _action_q_gradients(
            policy, critic_observations, finite_teacher_action
        )

        # Check whether the local Critic direction agrees with Teacher BC in
        # physical action space.  Separately verify the autograd action path by
        # a central difference along the unit Teacher direction in the
        # Critic's joint-normalized coordinates.
        teacher_delta = finite_teacher_action - mean_action.detach()
        gradient_norm = mean_dqda.norm(dim=-1)
        teacher_delta_norm = teacher_delta.float().norm(dim=-1)
        direction_denominator = gradient_norm * teacher_delta_norm
        directional_cosine = (
            (mean_dqda * teacher_delta.float()).sum(dim=-1)
            / direction_denominator.clamp_min(torch.finfo(torch.float32).eps)
        ).clamp(-1.0, 1.0)
        directional_valid = teacher_valid & (
            direction_denominator > torch.finfo(torch.float32).eps
        )

        q_scale = policy._fastsac_q_action_scale.to(mean_action).reshape(1, -1)
        normalized_teacher_delta = teacher_delta / q_scale
        normalized_direction_norm = normalized_teacher_delta.float().norm(
            dim=-1, keepdim=True
        )
        normalized_direction_valid = (
            normalized_direction_norm.reshape(-1) > torch.finfo(torch.float32).eps
        )
        normalized_direction = normalized_teacher_delta / (
            normalized_direction_norm.to(normalized_teacher_delta).clamp_min(
                torch.finfo(normalized_teacher_delta.dtype).eps
            )
        )
        physical_step_direction = normalized_direction * q_scale
        autograd_directional_derivative = (
            mean_dqda * physical_step_direction.float()
        ).sum(dim=-1)
        epsilon = float(finite_difference_epsilon)
        with torch.no_grad():
            _, q_plus = _expected_twin_q(
                policy,
                critic_observations,
                mean_action.detach() + epsilon * physical_step_direction,
            )
            _, q_minus = _expected_twin_q(
                policy,
                critic_observations,
                mean_action.detach() - epsilon * physical_step_direction,
            )
            finite_difference_derivative = (q_plus - q_minus) / (2.0 * epsilon)
        derivative_scale = torch.maximum(
            autograd_directional_derivative.abs(),
            finite_difference_derivative.abs(),
        ).clamp_min(1.0e-6)
        finite_difference_relative_error = (
            autograd_directional_derivative - finite_difference_derivative
        ).abs() / derivative_scale
        finite_difference_valid = teacher_valid & normalized_direction_valid

        for name, mask in masks.items():
            valid = mask & teacher_valid
            scalar_reports[name]["dqda_policy_sample_norm_mean"] = _finite_float(
                sample_dqda[mask].norm(dim=-1).mean()
            )
            scalar_reports[name]["dqda_policy_mean_norm_mean"] = _finite_float(
                mean_dqda[mask].norm(dim=-1).mean()
            )
            scalar_reports[name]["dqda_teacher_norm_mean"] = (
                _finite_float(teacher_dqda[valid].norm(dim=-1).mean())
                if bool(valid.any())
                else None
            )
            cosine_rows = mask & directional_valid
            scalar_reports[name][
                "cos_dqda_mean_with_teacher_minus_mean"
            ] = (
                _finite_float(directional_cosine[cosine_rows].mean())
                if bool(cosine_rows.any())
                else None
            )
            fd_rows = mask & finite_difference_valid
            if bool(fd_rows.any()):
                autograd_derivative = autograd_directional_derivative[fd_rows]
                finite_derivative = finite_difference_derivative[fd_rows]
                scalar_reports[name].update(
                    {
                        "teacher_direction_autograd_derivative_mean": _finite_float(
                            autograd_derivative.mean()
                        ),
                        "teacher_direction_finite_difference_derivative_mean": (
                            _finite_float(finite_derivative.mean())
                        ),
                        "teacher_direction_fd_relative_error_mean": _finite_float(
                            finite_difference_relative_error[fd_rows].mean()
                        ),
                        "teacher_direction_fd_sign_agreement_fraction": _finite_float(
                            (
                                torch.sign(autograd_derivative)
                                == torch.sign(finite_derivative)
                            )
                            .float()
                            .mean()
                        ),
                    }
                )
            else:
                scalar_reports[name].update(
                    {
                        "teacher_direction_autograd_derivative_mean": None,
                        "teacher_direction_finite_difference_derivative_mean": None,
                        "teacher_direction_fd_relative_error_mean": None,
                        "teacher_direction_fd_sign_agreement_fraction": None,
                    }
                )

    return {
        "schema": GRADIENT_PROBE_SCHEMA,
        "sample_seed": int(sample_seed),
        "finite_difference_epsilon_q_coordinates": float(
            finite_difference_epsilon
        ),
        "batch_rows": row_count,
        "source_masks_present": source_available,
        "failure_source_mask_present": failure_available,
        "source_gradients": bool(source_gradients),
        "coefficients": {
            "lambda_bc": lambda_bc,
            "eta_sac": eta_sac,
            "alpha": _finite_float(alpha),
        },
        "actor_parameter_counts": {
            "mean_tensors": len(mean_parameters),
            "mean_elements": sum(parameter.numel() for parameter in mean_parameters),
            "log_std_tensors": len(log_std_parameters),
            "log_std_elements": sum(
                parameter.numel() for parameter in log_std_parameters
            ),
        },
        "strata": scalar_reports,
        "gradients": gradient_reports,
        "twin_q": {
            "sample_q1_mean": _finite_float(twin_sample_q[0].detach().mean()),
            "sample_q2_mean": _finite_float(twin_sample_q[1].detach().mean()),
            "mean_q1_mean": _finite_float(twin_mean_q[0].mean()),
            "mean_q2_mean": _finite_float(twin_mean_q[1].mean()),
            "teacher_q1_mean": _finite_float(twin_teacher_q[0].mean()),
            "teacher_q2_mean": _finite_float(twin_teacher_q[1].mean()),
        },
    }


__all__ = [
    "GRADIENT_PROBE_SCHEMA",
    "diagnose_fastsac_actor_gradients",
]
