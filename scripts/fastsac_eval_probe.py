"""Paired deterministic/fixed-std evaluation for FastSAC-BC DAgger.

The probe reuses one simulator and evaluates every checkpoint, evaluation
seed, and fixed normalized action standard deviation on matching environment
indices.  It writes one per-environment row so success/failure tails and
termination causes can be inspected instead of relying only on 512-row means.
"""

from __future__ import annotations

import csv
import faulthandler
import gc
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from torchrl.envs.transforms import Compose, ObservationNorm
from tqdm import tqdm

try:
    from ._isaaclab_bootstrap import AppLauncher
    from .fastsac_eval_probe_utils import (
        fixed_std_latent_parameters,
        paired_against_deterministic,
        summarize_condition,
        terminal_stats_to_records,
        validate_fixed_normalized_stds,
    )
    from .helpers import _load_policy_checkpoint, make_env_policy
except ImportError:
    from _isaaclab_bootstrap import AppLauncher
    from fastsac_eval_probe_utils import (
        fixed_std_latent_parameters,
        paired_against_deterministic,
        summarize_condition,
        terminal_stats_to_records,
        validate_fixed_normalized_stds,
    )
    from helpers import _load_policy_checkpoint, make_env_policy

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import FastSACTanhNormal


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
PROBE_PREFIX = "_fastsac_eval_probe"

# Long Isaac/PhysX calls are difficult to inspect from an attached debugger in
# restricted environments.  ``kill -USR1 <pid>`` prints every Python stack
# without mutating the running diagnostic.
faulthandler.register(signal.SIGUSR1)


class FixedStdFastSACStudentPolicy(nn.Module):
    """Run the EMA Student mean with deterministic or controlled SAC noise."""

    def __init__(
        self,
        owner: nn.Module,
        *,
        normalized_std: float | None,
        use_checkpoint_std: bool = False,
        action_seed: int,
    ):
        super().__init__()
        required = (
            "_student_raw_action_proposal",
            "_bounded_actor_mean",
            "_sac_dist_from_mean",
            "_project_execution_action",
            "_q_action_input",
            "_fastsac_actor_action_center",
            "_fastsac_actor_action_scale",
            "_fastsac_q_action_scale",
            "_fastsac_action_low",
            "_fastsac_action_high",
        )
        missing = [name for name in required if not hasattr(owner, name)]
        if missing:
            raise TypeError(
                "fixed-std evaluation requires the FastSAC-BC DAgger backend; "
                f"missing={missing}"
            )
        object.__setattr__(self, "_owner", owner)
        # Register exactly the modules selected by the authoritative evaluation
        # policy so .eval() propagates through the complete EMA Student path.
        self.student_eval_modules = owner.get_rollout_policy("eval")
        self.normalized_std = (
            None if normalized_std is None else float(normalized_std)
        )
        self.use_checkpoint_std = bool(use_checkpoint_std)
        if self.use_checkpoint_std and self.normalized_std is not None:
            raise ValueError(
                "checkpoint std and a fixed normalized std are mutually exclusive"
            )
        if self.normalized_std is not None:
            if not math.isfinite(self.normalized_std) or self.normalized_std <= 0.0:
                raise ValueError("normalized_std must be finite and positive")
            minimum = math.exp(float(owner.cfg.sac_log_std_min))
            maximum = math.exp(float(owner.cfg.sac_log_std_max))
            if not minimum < self.normalized_std < maximum:
                raise ValueError(
                    "fixed normalized std must lie strictly inside the checkpoint "
                    f"FastSAC bounds ({minimum:g}, {maximum:g})"
                )
        self.action_generator = None
        if self.use_checkpoint_std or self.normalized_std is not None:
            self.action_generator = torch.Generator(device=owner.device)
            self.action_generator.manual_seed(int(action_seed))

    @torch.no_grad()
    def forward(self, td: TensorDict):
        owner = self._owner
        raw_mean = owner._student_raw_action_proposal(td)
        if not torch.isfinite(raw_mean).all():
            raise RuntimeError("FastSAC evaluation Actor produced a non-finite mean")
        mean_action = owner._bounded_actor_mean(raw_mean)
        if self.normalized_std is None and not self.use_checkpoint_std:
            action = mean_action
        else:
            if self.use_checkpoint_std:
                distribution = owner._sac_dist_from_mean(raw_mean)
            else:
                latent_loc, latent_scale = fixed_std_latent_parameters(
                    raw_mean,
                    owner._fastsac_actor_action_center,
                    owner._fastsac_actor_action_scale,
                    owner._fastsac_q_action_scale,
                    self.normalized_std,
                )
                distribution = FastSACTanhNormal(
                    latent_loc,
                    latent_scale,
                    low=owner._fastsac_action_low.to(raw_mean),
                    high=owner._fastsac_action_high.to(raw_mean),
                    event_dims=1,
                )
            action, _ = distribution.rsample_with_log_prob(
                generator=self.action_generator
            )
            action = owner._project_execution_action(action)
        if not torch.isfinite(action).all():
            raise RuntimeError("FastSAC evaluation sampled a non-finite action")

        q_delta = owner._q_action_input(action) - owner._q_action_input(mean_action)
        low = owner._fastsac_action_low.to(action)
        high = owner._fastsac_action_high.to(action)
        endpoint_tolerance = (high - low) * 1.0e-6
        saturated = ((action - low) <= endpoint_tolerance) | (
            (high - action) <= endpoint_tolerance
        )
        td[ACTION_KEY] = action
        td[(PROBE_PREFIX, "normalized_action_deviation_rms")] = (
            q_delta.square().mean(dim=-1, keepdim=True).sqrt()
        )
        td[(PROBE_PREFIX, "support_saturation_fraction")] = (
            saturated.float().mean(dim=-1, keepdim=True)
        )
        return td


def _flatten_stats(stats) -> dict[tuple[str, ...], torch.Tensor]:
    flattened: dict[tuple[str, ...], torch.Tensor] = {}
    for key, value in stats.items(True, True):
        normalized = (key,) if isinstance(key, str) else tuple(key)
        flattened[normalized] = value
    return flattened


def _optional_env_scalar(td, key: str, num_envs: int) -> torch.Tensor | None:
    value = td.get(key, None)
    if value is None:
        return None
    value = value.detach()
    if value.shape[0] != num_envs:
        return None
    value = value.reshape(num_envs, -1)
    return value[:, :1].clone()


@torch.inference_mode()
def collect_first_episodes(
    env,
    policy: FixedStdFastSACStudentPolicy,
    *,
    evaluation_seed: int,
    gamma: float,
) -> tuple[dict[tuple[str, ...], torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    """Collect the first terminal snapshot from every vectorized environment."""
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("evaluation return gamma must lie in [0, 1]")
    env.base_env.eval()
    env.eval()
    policy.eval()
    env.set_seed(int(evaluation_seed))
    td = env.reset()
    num_envs = int(env.num_envs)
    device = env.device
    captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
    has_done = torch.zeros_like(captured)
    done_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    terminal_terminated = torch.zeros_like(captured)
    terminal_truncated = torch.zeros_like(captured)
    stat_buffers: dict[tuple[str, ...], torch.Tensor] = {}
    last_stats: dict[tuple[str, ...], torch.Tensor] = {}

    action_deviation_sum = torch.zeros(num_envs, device=device)
    action_saturation_sum = torch.zeros(num_envs, device=device)
    action_step_count = torch.zeros(num_envs, device=device)
    undiscounted_dense_return = torch.zeros(num_envs, device=device)
    discounted_dense_return = torch.zeros(num_envs, device=device)
    discount_power = torch.ones(num_envs, device=device)
    q_effective_discounted_return = torch.zeros(num_envs, device=device)
    q_effective_discount_power = torch.ones(num_envs, device=device)
    start_phase = _optional_env_scalar(td, "ref_motion_phase_", num_envs)
    terminal_phase = torch.zeros((num_envs, 1), device=device)
    terminal_step_count = torch.zeros(
        (num_envs, 1), dtype=torch.long, device=device
    )
    inference_seconds: list[float] = []

    progress = tqdm(range(int(env.max_episode_length)), desc="first episodes", miniters=10)
    for step in progress:
        start = time.perf_counter()
        td = policy(td)
        inference_seconds.append(time.perf_counter() - start)
        active = ~captured
        deviation = td[(PROBE_PREFIX, "normalized_action_deviation_rms")].reshape(-1)
        saturation = td[(PROBE_PREFIX, "support_saturation_fraction")].reshape(-1)
        action_deviation_sum[active] += deviation[active]
        action_saturation_sum[active] += saturation[active]
        action_step_count[active] += 1.0

        transition, td = env.step_and_maybe_reset(td)
        next_td = transition.get("next")
        # This is the exact scalarization stored in Q replay:
        # PPOBCDaggerFinetune._scalarize_q_reward sums existing reward groups.
        scalar_reward = next_td.get("reward").reshape(num_envs, -1).sum(dim=-1)
        undiscounted_dense_return[active] += scalar_reward[active]
        discounted_dense_return[active] += (
            discount_power[active] * scalar_reward[active]
        )
        q_effective_discounted_return[active] += (
            q_effective_discount_power[active] * scalar_reward[active]
        )
        discount_power[active] *= gamma
        environment_discount = next_td.get("discount").reshape(num_envs, -1)[:, 0]
        q_effective_discount_power[active] *= (
            gamma * environment_discount[active]
        )
        done = next_td.get("done").reshape(-1).bool()
        capture = done & ~captured
        next_stats = _flatten_stats(next_td.get("stats"))
        # Keep only references to this step's freshly produced terminal-stat
        # snapshot.  Cloning every leaf on every simulator step would add
        # needless GPU allocation pressure; incomplete rows are copied only
        # once after the rollout.
        last_stats = {key: value.detach() for key, value in next_stats.items()}
        if not stat_buffers:
            stat_buffers = {
                key: torch.zeros_like(value) for key, value in next_stats.items()
            }

        if bool(capture.any().item()):
            for key, value in next_stats.items():
                stat_buffers[key][capture] = value[capture]
            terminal_terminated[capture] = next_td.get("terminated").reshape(-1).bool()[capture]
            terminal_truncated[capture] = next_td.get("truncated").reshape(-1).bool()[capture]
            phase = _optional_env_scalar(next_td, "ref_motion_phase_", num_envs)
            if phase is not None:
                terminal_phase[capture] = phase[capture].to(terminal_phase.dtype)
            step_count = _optional_env_scalar(transition, "step_count", num_envs)
            if step_count is not None:
                terminal_step_count[capture] = step_count[capture].to(
                    terminal_step_count.dtype
                )
            done_step[capture] = int(step)
            has_done[capture] = True
            captured[capture] = True
            progress.set_postfix(captured=int(captured.sum().item()), refresh=False)
            if bool(captured.all().item()):
                break

    incomplete = ~captured
    if bool(incomplete.any().item()):
        if not last_stats:
            raise RuntimeError("evaluation produced no environment statistics")
        for key, value in last_stats.items():
            stat_buffers[key][incomplete] = value[incomplete]
        done_step[incomplete] = int(env.max_episode_length) - 1

    action_denominator = action_step_count.clamp_min(1.0)
    per_env = {
        "probe/mean_normalized_action_deviation_rms": (
            action_deviation_sum / action_denominator
        ),
        "probe/mean_support_saturation_fraction": (
            action_saturation_sum / action_denominator
        ),
        "probe/initial_reference_phase": (
            torch.zeros((num_envs, 1), device=device)
            if start_phase is None
            else start_phase
        ),
        "probe/terminal_reference_phase": terminal_phase,
        "probe/terminal_step_count": terminal_step_count,
        "probe/undiscounted_transition_dense_return": undiscounted_dense_return,
        "probe/discounted_dense_return": discounted_dense_return,
        "probe/terminal_discount_power": discount_power,
        "probe/q_effective_discounted_dense_return": q_effective_discounted_return,
        "probe/q_effective_terminal_discount_power": q_effective_discount_power,
        "_has_done": has_done,
        "_done_step": done_step,
        "_terminated": terminal_terminated,
        "_truncated": terminal_truncated,
    }
    timing = {
        "mean_policy_inference_seconds": float(
            sum(inference_seconds[5:]) / max(1, len(inference_seconds[5:]))
        ),
        "rollout_steps": len(inference_seconds),
    }
    return stat_buffers, per_env, timing


def _install_checkpoint_vecnorm(env, vecnorm, state: dict[str, Any]) -> None:
    if "vecnorm" not in state:
        raise ValueError("evaluation checkpoint lacks VecNorm state")
    vecnorm.load_state_dict(state["vecnorm"])
    observation_norms = vecnorm.to_observation_norm().transforms
    retained = [
        transform.clone()
        for transform in env.transform
        if not isinstance(transform, ObservationNorm)
    ]
    retained.extend(observation_norms)
    env.transform = Compose(*retained)


def _checkpoint_metadata(agent, policy_state: dict[str, Any]) -> dict[str, Any]:
    learned_std = agent._bounded_log_std().detach().exp().cpu()
    log_alpha = policy_state.get("log_alpha")
    alpha = None
    if torch.is_tensor(log_alpha) and log_alpha.numel() == 1:
        alpha = float(log_alpha.detach().float().exp().item())
    return {
        "learned_normalized_action_std_mean": float(learned_std.mean()),
        "learned_normalized_action_std_min": float(learned_std.min()),
        "learned_normalized_action_std_max": float(learned_std.max()),
        "alpha": alpha,
        "actor_update_count": int(policy_state.get("actor_update_count", -1)),
        "critic_update_count": int(policy_state.get("critic_update_count", -1)),
        "alpha_update_count": int(policy_state.get("alpha_update_count", -1)),
        "dagger_rollout_count": int(policy_state.get("dagger_rollout_count", -1)),
    }


def _safe_checkpoint_label(path: Path) -> str:
    label = path.stem
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in label)


def _write_records(records: list[dict[str, Any]], output_dir: Path) -> None:
    jsonl_path = output_dir / "per_environment.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    fieldnames = sorted({key for record in records for key in record})
    csv_path = output_dir / "per_environment.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def _probe_settings(cfg: DictConfig) -> dict[str, Any]:
    probe = cfg.get("eval_probe", {})
    configured_paths = probe.get("checkpoint_paths", None)
    if configured_paths is None:
        configured_paths = [cfg.get("checkpoint_path", None)]
    checkpoint_paths = []
    for raw_path in configured_paths:
        if raw_path is None:
            raise ValueError("eval_probe requires at least one checkpoint path")
        path = Path(hydra.utils.to_absolute_path(os.fspath(raw_path))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluation checkpoint does not exist: {path}")
        checkpoint_paths.append(path)
    seeds = tuple(int(seed) for seed in probe.get("seeds", [int(cfg.get("seed", 0))]))
    if not seeds:
        raise ValueError("eval_probe.seeds must not be empty")
    raw_fixed_stds = probe.get("fixed_normalized_action_stds", [0.02, 0.05, 0.1])
    fixed_stds = (
        validate_fixed_normalized_stds(raw_fixed_stds)
        if len(raw_fixed_stds)
        else ()
    )
    include_deterministic = bool(probe.get("include_deterministic", True))
    include_checkpoint_std = bool(probe.get("include_checkpoint_std", True))
    if not include_deterministic and not include_checkpoint_std and not fixed_stds:
        raise ValueError("eval_probe must enable at least one action mode")
    output_dir = Path(
        hydra.utils.to_absolute_path(
            os.fspath(probe.get("output_dir", "diagnostics/fastsac_eval_probe"))
        )
    ).resolve()
    return {
        "checkpoint_paths": checkpoint_paths,
        "seeds": seeds,
        "fixed_stds": fixed_stds,
        "include_deterministic": include_deterministic,
        "include_checkpoint_std": include_checkpoint_std,
        "action_seed_base": int(probe.get("action_seed_base", 9_170_003)),
        "output_dir": output_dir,
    }


@hydra.main(config_path=CONFIG_PATH, config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    settings = _probe_settings(cfg)
    first_checkpoint = settings["checkpoint_paths"][0]
    with open_dict(cfg):
        cfg.checkpoint_path = str(first_checkpoint)
        cfg.vecnorm = "eval"
        cfg.eval_render = False
        cfg.headless = True

    output_dir: Path = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app
    env = None
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    try:
        env, agent, vecnorm = make_env_policy(cfg, inference_only=True)
        for checkpoint_path in settings["checkpoint_paths"]:
            state = torch.load(
                checkpoint_path,
                map_location=env.device,
                weights_only=False,
            )
            if not isinstance(state, dict) or not isinstance(state.get("policy"), dict):
                raise ValueError(f"invalid policy checkpoint: {checkpoint_path}")
            _load_policy_checkpoint(agent, state["policy"], inference_only=True)
            _install_checkpoint_vecnorm(env, vecnorm, state)
            checkpoint_label = _safe_checkpoint_label(checkpoint_path)
            checkpoint_info = _checkpoint_metadata(agent, state["policy"])

            conditions: list[tuple[str, float | None, bool]] = []
            if settings["include_deterministic"]:
                conditions.append(("deterministic", None, False))
            if settings["include_checkpoint_std"]:
                conditions.append(("checkpoint_std", None, True))
            conditions.extend(
                (f"fixed_std_{normalized_std:g}", normalized_std, False)
                for normalized_std in settings["fixed_stds"]
            )
            for evaluation_seed in settings["seeds"]:
                for mode, normalized_std, use_checkpoint_std in conditions:
                    # Reusing this exact seed for every std provides common
                    # Gaussian draws joint-by-joint and step-by-step.
                    condition_key = (
                        f"{checkpoint_label}|seed={evaluation_seed}|{mode}"
                    )
                    action_seed = settings["action_seed_base"] + int(evaluation_seed)
                    policy = FixedStdFastSACStudentPolicy(
                        agent,
                        normalized_std=normalized_std,
                        use_checkpoint_std=use_checkpoint_std,
                        action_seed=action_seed,
                    )
                    stat_values, per_env, timing = collect_first_episodes(
                        env,
                        policy,
                        evaluation_seed=evaluation_seed,
                        gamma=float(agent.cfg.gamma),
                    )
                    print(f"[{condition_key}] rollout collection complete", flush=True)
                    metadata = {
                        "checkpoint": checkpoint_label,
                        "checkpoint_path": str(checkpoint_path),
                        "evaluation_seed": int(evaluation_seed),
                        "mode": mode,
                        "fixed_normalized_action_std": normalized_std,
                        "uses_checkpoint_std": use_checkpoint_std,
                        "action_seed": (
                            action_seed
                            if use_checkpoint_std or normalized_std is not None
                            else None
                        ),
                    }
                    records = terminal_stats_to_records(
                        stat_values,
                        has_done=per_env.pop("_has_done"),
                        done_step=per_env.pop("_done_step"),
                        terminated=per_env.pop("_terminated"),
                        truncated=per_env.pop("_truncated"),
                        metadata=metadata,
                        step_dt=float(env.step_dt),
                        per_env_values=per_env,
                    )
                    print(f"[{condition_key}] per-environment records complete", flush=True)
                    all_records.extend(records)
                    summaries[condition_key] = {
                        "checkpoint_metadata": checkpoint_info,
                        "timing": timing,
                        **summarize_condition(records),
                    }
                    print(f"[{condition_key}] summary complete", flush=True)
                    print(
                        f"[{condition_key}] success="
                        f"{summaries[condition_key]['success_rate']:.4f}, "
                        f"dense_return_auc="
                        f"{summaries[condition_key]['dense_return_success_auc']}"
                    )
                    del policy, stat_values, per_env
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del state

        _write_records(all_records, output_dir)
        report = {
            "schema_version": 1,
            "argv": sys.argv,
            "pairing": {
                "environment_pair": "checkpoint + evaluation_seed + env_index",
                "action_noise_pair": (
                    "same action_seed and Gaussian draw stream across fixed stds"
                ),
                "fixed_std_semantics": (
                    "dimensionless nominal Q/BC joint coordinates before tanh"
                ),
                "discounted_return_semantics": (
                    "sum_t gamma^t * sum(next/reward groups), stopped at first done; "
                    f"gamma={float(agent.cfg.gamma):g}"
                ),
                "q_effective_discounted_return_semantics": (
                    "sum_t [product before t of gamma * next/discount] * "
                    "sum(next/reward groups), stopped at first done; excludes the "
                    "stochastic SAC entropy tax"
                ),
            },
            "conditions": summaries,
            "paired_against_deterministic": paired_against_deterministic(all_records),
        }
        OmegaConf.save(OmegaConf.create(report), output_dir / "summary.yaml")
        print(f"FastSAC evaluation probe saved to: {output_dir}")
    except BaseException:
        # Isaac Sim shutdown can take long enough to hide the exception that
        # initiated cleanup.  Emit it before closing the app so smoke-test
        # failures remain diagnosable.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        # Rendering-enabled Isaac Sim can hang while waiting for Replicator
        # teardown after camera evaluation.  The probe has already flushed all
        # output files at this point, so use the supported non-blocking cleanup
        # path used for standalone diagnostics.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
