"""Plot and film frozen-Teacher success/failure trajectories.

This is a read-only diagnostic for the fresh TVKD prefill path.  It builds the
same TVKD policy from a train-phase PPO checkpoint and executes the same
forced-Teacher rollout policy, but it does not create replay, call ``train_op``,
initialize W&B, or update any parameter.  By default, the command starts from
motion frame zero while the environment remains in training mode, so reset and
physics randomization are preserved.  The skateboard motion has 622 states at
50 Hz and therefore a successful full episode has 621 value transitions.

The probe intentionally retains every step of each complete episode, including
the trainer's one-step shape warm-up and the replay burn-in/filter prefix,
because those omitted steps are still part of the physical trajectory the user
wants to inspect.  ``--random-start`` restores the exact random-phase reset
used during actual prefill.

Recorded videos restore the exact robot/object pose and the corresponding HDMI
reference keypoints.  The reference targets use the same red sphere marker as
``scripts/play.py``; green measured-body markers are intentionally hidden.

The PNG contains the five quantities requested for visual inspection:

    r_tilde = r_t + lambda_V * (gamma * c_t * B_t - V_tau(s_t))
    delta_t^T = r_t + gamma * c_t * B_t - V_tau(s_t)
    r_t
    V_tau(s_{t+1})
    V_tau(s_t)

Here ``c_t`` is the replay/environment discount and ``B_t`` is the
boundary-adjusted Teacher bootstrap value: ``V_tau(s_{t+1})`` on an ordinary
transition, zero on physical termination, and ``V_tau(s_t)`` on command
completion or time-limit timeout.  ``gamma`` is the temporal discount;
``lambda_V`` is the Teacher-value shaping weight, not a discount.

On physically failed trajectories, the PNG also marks a diagnostic failure
onset with a square.  Let ``tau`` be the physical-terminal transition and

    m_t = mean(delta_{t-4}^T, ..., delta_t^T).

After excluding replay burn-in and the terminal plus its five preceding rows,
``t_onset`` is the first transition for which ``m_t < -0.05`` holds at three
consecutive transition endpoints.  The prevention-sampling hypothesis is the
ten valid transitions strictly before ``t_onset``; the diagnostic does not
change the production Student replay sampler.

The CSV also saves the production TVKD residual and the literal value
difference for auditability.  Production applies gamma, the environment
discount, and the same boundary-specific continuation rules:

    delta_runtime = r_t + C_T(t) - V_T(s_t)

where physical termination uses C_T=0, command completion/time limit uses
self-bootstrap, and an ordinary transition uses V_T(s_{t+1}).  The critic's
shaped reward is also saved because it uses the potential difference rather
than ``delta_runtime`` directly.

Example (run from the repository root in the ``vaic`` environment)::

    python tests/tvkd_teacher_delta_probe.py \
      --checkpoint outputs/15-13-46-G1Skateboard-ppo_vel/wandb/\
run-20260805_151350-7tsje71w/files/checkpoint_final.pt \
      --task G1/vaic/skateboard_stu \
      --num-envs 128 \
      --record-videos \
      --output-dir diagnostics/tvkd_teacher_delta
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UNKNOWN = "unknown"
OUTCOMES = (
    OUTCOME_SUCCESS,
    OUTCOME_FAILURE,
    OUTCOME_TIMEOUT,
    OUTCOME_UNKNOWN,
)

# The numeric onset contract now matches the v5 production sampler. The probe
# remains source-agnostic so its pure-Teacher plots are an onset proxy; v5
# production applies the same raw rule only to Student-executed rows.
ONSET_SMOOTHING_WINDOW = 5
ONSET_RAW_THRESHOLD = -0.05
ONSET_MIN_CONSECUTIVE = 3
ONSET_MIN_REPLAY_STEP = 6  # Production replay retains step_count > 5.
ONSET_TERMINAL_EXCLUSION_STEPS = 5
ONSET_PRECURSOR_STEPS = 10


def classify_episode_boundary(
    *,
    done: bool,
    terminated: bool,
    command_finished: bool,
    time_limit: bool,
) -> str | None:
    """Classify a boundary with the same precedence as Teacher prefill."""
    if not done:
        return None
    if terminated:
        return OUTCOME_FAILURE
    if command_finished:
        return OUTCOME_SUCCESS
    if time_limit:
        return OUTCOME_TIMEOUT
    return OUTCOME_UNKNOWN


def compute_probe_terms(
    *,
    teacher_v,
    teacher_v_next,
    raw_reward,
    terminated,
    command_finished,
    time_limit,
    replay_discount,
    gamma: float,
    tvkd_lambda: float,
    potential_clip: float | None,
) -> dict[str, Any]:
    """Return literal, failure-detector, and critic-shaping quantities."""
    import torch

    from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
        TEACHER_VALUE_BOUNDARY_SEMANTICS,
        compute_teacher_value_continuation,
    )

    teacher_v = torch.as_tensor(teacher_v).detach().float().reshape(-1)
    teacher_v_next = (
        torch.as_tensor(teacher_v_next)
        .detach()
        .to(device=teacher_v.device, dtype=torch.float32)
        .reshape(-1)
    )
    raw_reward = (
        torch.as_tensor(raw_reward)
        .detach()
        .to(device=teacher_v.device, dtype=torch.float32)
        .reshape(-1)
    )
    terminated = (
        torch.as_tensor(terminated)
        .detach()
        .to(device=teacher_v.device, dtype=torch.bool)
        .reshape(-1)
    )
    command_finished = (
        torch.as_tensor(command_finished)
        .detach()
        .to(device=teacher_v.device, dtype=torch.bool)
        .reshape(-1)
    )
    time_limit = (
        torch.as_tensor(time_limit)
        .detach()
        .to(device=teacher_v.device, dtype=torch.bool)
        .reshape(-1)
    )
    replay_discount = (
        torch.as_tensor(replay_discount)
        .detach()
        .to(device=teacher_v.device, dtype=torch.float32)
    )
    if replay_discount.ndim == 0:
        replay_discount = replay_discount.expand_as(teacher_v)
    else:
        replay_discount = replay_discount.reshape(-1)

    shapes = (
        teacher_v_next,
        raw_reward,
        terminated,
        command_finished,
        time_limit,
        replay_discount,
    )
    if any(value.shape != teacher_v.shape for value in shapes):
        raise ValueError("probe tensors must have one aligned value per transition")
    if not math.isfinite(float(tvkd_lambda)):
        raise ValueError("tvkd_lambda must be finite")
    for name, value in (
        ("teacher_v", teacher_v),
        ("teacher_v_next", teacher_v_next),
        ("raw_reward", raw_reward),
        ("replay_discount", replay_discount),
    ):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN/Inf")

    teacher_continuation = compute_teacher_value_continuation(
        teacher_v=teacher_v,
        teacher_v_next=teacher_v_next,
        terminated=terminated,
        command_finished=command_finished,
        time_limit=time_limit,
        replay_discount=replay_discount,
        gamma=float(gamma),
        semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
    )
    boundary_teacher_v = torch.where(
        terminated,
        torch.zeros_like(teacher_v),
        torch.where(
            command_finished | time_limit,
            teacher_v,
            teacher_v_next,
        ),
    )

    fixed_v = teacher_v
    fixed_v_next = teacher_v_next
    if potential_clip is not None:
        clip = float(potential_clip)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("potential_clip must be finite and positive")
        fixed_v = fixed_v.clamp(-clip, clip)
        fixed_v_next = fixed_v_next.clamp(-clip, clip)
    fixed_continuation = compute_teacher_value_continuation(
        teacher_v=fixed_v,
        teacher_v_next=fixed_v_next,
        terminated=terminated,
        command_finished=command_finished,
        time_limit=time_limit,
        replay_discount=replay_discount,
        gamma=float(gamma),
        semantics=TEACHER_VALUE_BOUNDARY_SEMANTICS,
    )

    literal_delta = raw_reward + teacher_v_next - teacher_v
    production_delta = raw_reward + teacher_continuation - teacher_v
    potential_delta = fixed_continuation - fixed_v
    shaped_reward = raw_reward + float(tvkd_lambda) * potential_delta
    result = {
        "teacher_v": teacher_v,
        "teacher_v_next": teacher_v_next,
        "value_change": teacher_v_next - teacher_v,
        "boundary_teacher_v": boundary_teacher_v,
        "teacher_continuation": teacher_continuation,
        "fixed_teacher_continuation": fixed_continuation,
        "literal_delta": literal_delta,
        "production_delta": production_delta,
        "potential_delta": potential_delta,
        "shaped_reward": shaped_reward,
    }
    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
    return result


def prefill_progress_iteration(
    *, start_iteration: int, environment_step: int, train_every: int
) -> int:
    """Map a diagnostic step to the production physical-rollout coordinate."""
    if isinstance(start_iteration, bool) or int(start_iteration) < 0:
        raise ValueError("start_iteration must be a non-negative integer")
    if isinstance(environment_step, bool) or int(environment_step) < 0:
        raise ValueError("environment_step must be a non-negative integer")
    if isinstance(train_every, bool) or int(train_every) < 1:
        raise ValueError("train_every must be a positive integer")
    return int(start_iteration) + int(environment_step) // int(train_every)


def minimum_prefill_rollouts(*, capacity: int, num_envs: int, train_every: int) -> int:
    """Return the reachability-only ceiling needed by training validation."""
    for name, value in (
        ("capacity", capacity),
        ("num_envs", num_envs),
        ("train_every", train_every),
    ):
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    rows_per_rollout = int(num_envs) * int(train_every)
    return (int(capacity) + rows_per_rollout - 1) // rows_per_rollout


@dataclass(frozen=True)
class CapturedTrajectory:
    outcome: str
    sample_index: int
    env_index: int
    episode_index: int
    pure_teacher: bool
    rows: tuple[dict[str, Any], ...]
    scene_states: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DiagnosticFailureOnset:
    """Plot-only onset and its strictly pre-onset prevention window."""

    row_index: int
    episode_step: int
    confirmation_step: int
    raw_production_delta: float
    smoothed_production_delta: float
    precursor_start_step: int
    precursor_end_step: int
    precursor_transition_count: int


@dataclass(frozen=True)
class VideoArtifact:
    outcome: str
    sample_index: int
    env_index: int
    episode_index: int
    path: Path
    frames: int
    fps: float
    width: int
    height: int
    terminal_frame_pre_reset: bool = True


class EpisodeReservoir:
    """Maintain asynchronous per-environment histories and select outcomes."""

    def __init__(
        self,
        num_envs: int,
        *,
        num_success: int,
        num_failure: int,
        require_pure_teacher: bool,
    ) -> None:
        for name, value in (
            ("num_envs", num_envs),
            ("num_success", num_success),
            ("num_failure", num_failure),
        ):
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.num_envs = int(num_envs)
        self.quotas = {
            OUTCOME_SUCCESS: int(num_success),
            OUTCOME_FAILURE: int(num_failure),
        }
        self.require_pure_teacher = bool(require_pure_teacher)
        self.histories: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self.scene_state_histories: list[list[Any]] = [[] for _ in range(self.num_envs)]
        self.episode_indices = [0 for _ in range(self.num_envs)]
        self.selected: dict[str, list[CapturedTrajectory]] = {
            OUTCOME_SUCCESS: [],
            OUTCOME_FAILURE: [],
        }
        self.outcome_counts = {outcome: 0 for outcome in OUTCOMES}
        self.rejected_impure_counts = {
            OUTCOME_SUCCESS: 0,
            OUTCOME_FAILURE: 0,
        }
        self.incomplete_episode_count = 0

    @property
    def complete(self) -> bool:
        return all(
            len(self.selected[outcome]) >= quota
            for outcome, quota in self.quotas.items()
        )

    def add_step(
        self,
        env_index: int,
        row: Mapping[str, Any],
        *,
        scene_state_before: Any = None,
        scene_state_after: Any = None,
    ) -> CapturedTrajectory | None:
        env_index = int(env_index)
        if not 0 <= env_index < self.num_envs:
            raise IndexError("environment index is outside the reservoir")
        history = self.histories[env_index]
        state_history = self.scene_state_histories[env_index]
        if bool(row.get("is_init", False)) and history:
            history.clear()
            state_history.clear()
            self.incomplete_episode_count += 1
            self.episode_indices[env_index] += 1

        if scene_state_before is not None or scene_state_after is not None:
            if scene_state_before is None or scene_state_after is None:
                raise ValueError("both scene-state endpoints must be provided")
            if not state_history:
                state_history.append(scene_state_before)
            state_history.append(scene_state_after)
        elif state_history:
            raise ValueError("scene-state capture cannot stop inside an episode")

        record = dict(row)
        record["episode_step"] = len(history)
        history.append(record)
        outcome = classify_episode_boundary(
            done=bool(record["done"]),
            terminated=bool(record["terminated"]),
            command_finished=bool(record["command_finished"]),
            time_limit=bool(record["time_limit"]),
        )
        if outcome is None:
            return None

        self.outcome_counts[outcome] += 1
        pure_teacher = all(
            bool(item["teacher_action_valid"]) and not bool(item["student_fallback"])
            for item in history
        )
        selected = None
        if (
            outcome in self.selected
            and len(self.selected[outcome]) < self.quotas[outcome]
        ):
            if pure_teacher or not self.require_pure_teacher:
                selected = CapturedTrajectory(
                    outcome=outcome,
                    sample_index=len(self.selected[outcome]) + 1,
                    env_index=env_index,
                    episode_index=self.episode_indices[env_index],
                    pure_teacher=pure_teacher,
                    rows=tuple(dict(item) for item in history),
                    scene_states=tuple(state_history),
                )
                self.selected[outcome].append(selected)
            else:
                self.rejected_impure_counts[outcome] += 1

        history.clear()
        state_history.clear()
        self.episode_indices[env_index] += 1
        return selected


CSV_FIELDS = (
    "outcome",
    "sample_index",
    "env_index",
    "episode_index",
    "pure_teacher",
    "episode_step",
    "environment_step",
    "step_count",
    "reference_phase",
    "motion_id",
    "raw_reward",
    "teacher_v",
    "teacher_v_next",
    "value_change",
    "boundary_teacher_v",
    "teacher_continuation",
    "fixed_teacher_continuation",
    "literal_delta",
    "production_delta",
    "potential_delta",
    "shaped_reward",
    "replay_discount",
    "done",
    "terminated",
    "command_finished",
    "time_limit",
    "environment_success_stat",
    "is_init",
    "teacher_action_valid",
    "student_fallback",
    "termination_causes",
)


def _all_selected(
    selected: Mapping[str, Sequence[CapturedTrajectory]],
) -> list[CapturedTrajectory]:
    return [
        *selected.get(OUTCOME_SUCCESS, ()),
        *selected.get(OUTCOME_FAILURE, ()),
    ]


def select_video_targets(
    selected: Mapping[str, Sequence[CapturedTrajectory]],
    *,
    per_outcome: int,
) -> list[CapturedTrajectory]:
    """Choose graph trajectories whose exact episodes will be filmed."""
    if isinstance(per_outcome, bool) or int(per_outcome) < 1:
        raise ValueError("per_outcome must be a positive integer")
    count = int(per_outcome)
    return [
        *list(selected.get(OUTCOME_SUCCESS, ()))[:count],
        *list(selected.get(OUTCOME_FAILURE, ()))[:count],
    ]


def video_artifact_summary(artifact: VideoArtifact) -> dict[str, Any]:
    return {
        "outcome": artifact.outcome,
        "sample_index": artifact.sample_index,
        "env_index": artifact.env_index,
        "episode_index": artifact.episode_index,
        "path": str(artifact.path),
        "frames": artifact.frames,
        "fps": artifact.fps,
        "resolution": [artifact.width, artifact.height],
        "terminal_frame_pre_reset": artifact.terminal_frame_pre_reset,
    }


def write_csv(
    selected: Mapping[str, Sequence[CapturedTrajectory]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trajectory in _all_selected(selected):
            metadata = {
                "outcome": trajectory.outcome,
                "sample_index": trajectory.sample_index,
                "env_index": trajectory.env_index,
                "episode_index": trajectory.episode_index,
                "pure_teacher": trajectory.pure_teacher,
            }
            for row in trajectory.rows:
                writer.writerow({**metadata, **row})


def detect_diagnostic_failure_onset(
    rows: Sequence[Mapping[str, Any]],
) -> DiagnosticFailureOnset | None:
    """Find a sustained raw-residual onset on one physical-failure episode.

    The returned ``episode_step`` is the first endpoint in the confirmed
    three-endpoint run.  ``confirmation_step`` is two transitions later.  All
    rolling windows contain only replay-valid rows, and neither terminal
    boundary leakage nor its five preceding transitions can participate.
    """
    if not rows:
        return None

    steps = [int(row["episode_step"]) for row in rows]
    expected_steps = list(range(steps[0], steps[0] + len(steps)))
    if steps != expected_steps:
        raise ValueError("trajectory episode_step values must be contiguous")
    residual = [float(row["production_delta"]) for row in rows]
    if not all(math.isfinite(value) for value in residual):
        raise RuntimeError("trajectory production_delta contains NaN/Inf")

    terminal_indices = [
        index for index, row in enumerate(rows) if bool(row["terminated"])
    ]
    if not terminal_indices:
        return None
    terminal_index = terminal_indices[-1]
    terminal_step = steps[terminal_index]
    candidate_step_limit = terminal_step - ONSET_TERMINAL_EXCLUSION_STEPS

    rolling_mean: dict[int, float] = {}
    window = ONSET_SMOOTHING_WINDOW
    for end_index in range(window - 1, terminal_index):
        start_index = end_index - window + 1
        window_steps = steps[start_index : end_index + 1]
        if window_steps[0] < ONSET_MIN_REPLAY_STEP:
            continue
        if steps[end_index] >= candidate_step_limit:
            continue
        rolling_mean[end_index] = float(
            sum(residual[start_index : end_index + 1]) / window
        )

    run_start_index: int | None = None
    run_length = 0
    previous_index: int | None = None
    for end_index, mean_value in rolling_mean.items():
        consecutive = previous_index is not None and end_index == previous_index + 1
        if mean_value < ONSET_RAW_THRESHOLD:
            if run_length == 0 or not consecutive:
                run_start_index = end_index
                run_length = 1
            else:
                run_length += 1
            if run_length >= ONSET_MIN_CONSECUTIVE:
                assert run_start_index is not None
                onset_index = run_start_index
                onset_step = steps[onset_index]
                precursor_start_step = max(
                    ONSET_MIN_REPLAY_STEP,
                    onset_step - ONSET_PRECURSOR_STEPS,
                )
                precursor_end_step = onset_step - 1
                precursor_count = max(0, precursor_end_step - precursor_start_step + 1)
                return DiagnosticFailureOnset(
                    row_index=onset_index,
                    episode_step=onset_step,
                    confirmation_step=steps[end_index],
                    raw_production_delta=residual[onset_index],
                    smoothed_production_delta=rolling_mean[onset_index],
                    precursor_start_step=precursor_start_step,
                    precursor_end_step=precursor_end_step,
                    precursor_transition_count=precursor_count,
                )
        else:
            run_start_index = None
            run_length = 0
        previous_index = end_index
    return None


def _onset_summary(
    onset: DiagnosticFailureOnset | None,
) -> dict[str, Any] | None:
    if onset is None:
        return None
    return {
        "episode_step": onset.episode_step,
        "confirmation_step": onset.confirmation_step,
        "raw_production_delta": onset.raw_production_delta,
        "smoothed_production_delta": onset.smoothed_production_delta,
        "precursor_start_step": onset.precursor_start_step,
        "precursor_end_step": onset.precursor_end_step,
        "precursor_transition_count": onset.precursor_transition_count,
    }


def _trajectory_summary(trajectory: CapturedTrajectory) -> dict[str, Any]:
    rows = trajectory.rows
    deltas = [float(row["production_delta"]) for row in rows]
    literal = [float(row["literal_delta"]) for row in rows]
    onset = (
        detect_diagnostic_failure_onset(rows)
        if trajectory.outcome == OUTCOME_FAILURE
        else None
    )
    return {
        "outcome": trajectory.outcome,
        "sample_index": trajectory.sample_index,
        "env_index": trajectory.env_index,
        "episode_index": trajectory.episode_index,
        "pure_teacher": trajectory.pure_teacher,
        "length": len(rows),
        "motion_id": int(rows[0]["motion_id"]),
        "raw_return": float(sum(float(row["raw_reward"]) for row in rows)),
        "production_delta_mean": float(sum(deltas) / len(deltas)),
        "production_delta_min": float(min(deltas)),
        "production_delta_max": float(max(deltas)),
        "literal_delta_mean": float(sum(literal) / len(literal)),
        "diagnostic_failure_onset": _onset_summary(onset),
        "terminal": {
            name: bool(rows[-1][name])
            for name in ("done", "terminated", "command_finished", "time_limit")
        },
        "termination_causes": str(rows[-1]["termination_causes"]),
        "student_fallback_steps": int(
            sum(bool(row["student_fallback"]) for row in rows)
        ),
    }


def write_summary(
    selected: Mapping[str, Sequence[CapturedTrajectory]],
    output_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    payload = dict(metadata)
    payload["trajectories"] = [
        _trajectory_summary(trajectory) for trajectory in _all_selected(selected)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plot_trajectories(
    selected: Mapping[str, Sequence[CapturedTrajectory]],
    output_path: Path,
    *,
    gamma: float,
    tvkd_lambda: float,
    potential_clip: float | None = None,
) -> None:
    """Plot critic-shaped reward, environment reward, and raw Teacher values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(tempfile.gettempdir()) / "vaic-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    success = list(selected.get(OUTCOME_SUCCESS, ()))
    failure = list(selected.get(OUTCOME_FAILURE, ()))
    columns = max(1, len(success), len(failure))
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(8.0 * columns, 10.0),
        squeeze=False,
        constrained_layout=False,
    )
    outcome_rows = (
        (OUTCOME_SUCCESS, success, "#19733a"),
        (OUTCOME_FAILURE, failure, "#b42318"),
    )
    legend_handles = []
    legend_labels = []

    for row_index, (outcome, trajectories, title_color) in enumerate(outcome_rows):
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            if column_index >= len(trajectories):
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    f"{outcome.title()} sample not captured",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue

            trajectory = trajectories[column_index]
            rows = trajectory.rows
            steps = [int(item["episode_step"]) for item in rows]
            shaped_reward = [float(item["shaped_reward"]) for item in rows]
            production_delta = [float(item["production_delta"]) for item in rows]
            raw_reward = [float(item["raw_reward"]) for item in rows]
            teacher_v = [float(item["teacher_v"]) for item in rows]
            teacher_v_next = [float(item["teacher_v_next"]) for item in rows]
            diagnostic_onset = (
                detect_diagnostic_failure_onset(rows)
                if outcome == OUTCOME_FAILURE
                else None
            )

            line_specs = (
                (
                    shaped_reward,
                    "#0072B2",
                    "-",
                    2.0,
                    r"$\widetilde{r}_t=r_t+\lambda_V[\gamma c_tB_t-V_\tau(s_t)]$",
                ),
                (
                    raw_reward,
                    "#D55E00",
                    "-",
                    1.5,
                    r"environment reward $r_t$",
                ),
                (
                    production_delta,
                    "#FF0000",
                    "-",
                    1.75,
                    r"$\delta_t^T=r_t+\gamma c_tB_t-V_\tau(s_t)$",
                ),
            )
            for values, color, style, width, label in line_specs:
                (handle,) = axis.plot(
                    steps,
                    values,
                    color=color,
                    linestyle=style,
                    linewidth=width,
                    label=label,
                )
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
            if diagnostic_onset is not None:
                if diagnostic_onset.precursor_transition_count:
                    axis.axvspan(
                        diagnostic_onset.precursor_start_step - 0.5,
                        diagnostic_onset.episode_step - 0.5,
                        color="#0072B2",
                        alpha=0.07,
                        linewidth=0.0,
                    )
                onset_label = r"diagnostic $t_{\mathrm{onset}}$"
                onset_handle = axis.scatter(
                    [diagnostic_onset.episode_step],
                    [diagnostic_onset.raw_production_delta],
                    marker="s",
                    s=90,
                    facecolor="#FDE047",
                    edgecolor="#111827",
                    linewidth=1.25,
                    zorder=8,
                    label=onset_label,
                )
                axis.annotate(
                    (
                        rf"$t_{{\mathrm{{onset}}}}={diagnostic_onset.episode_step}$"
                        "\n"
                        rf"MA5 $\delta^T={diagnostic_onset.smoothed_production_delta:.3f}$"
                    ),
                    xy=(
                        diagnostic_onset.episode_step,
                        diagnostic_onset.raw_production_delta,
                    ),
                    xytext=(8, 10),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="#111827",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "#9CA3AF",
                        "alpha": 0.88,
                    },
                )
                if onset_label not in legend_labels:
                    legend_handles.append(onset_handle)
                    legend_labels.append(onset_label)
            axis.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
            axis.set_xlabel("episode control step")
            axis.set_ylabel(
                r"environment $r_t$ / critic-shaped $\widetilde{r}_t$ / "
                r"detector $\delta_t^T$"
            )
            axis.grid(True, alpha=0.2)

            value_axis = axis.twinx()
            (value_next_handle,) = value_axis.plot(
                steps,
                teacher_v_next,
                color="#009E73",
                linestyle=":",
                linewidth=1.15,
                alpha=0.9,
                label=r"$V_\tau(s_{t+1})$",
            )
            (value_current_handle,) = value_axis.plot(
                steps,
                teacher_v,
                color="#7C3AED",
                linestyle=":",
                linewidth=1.15,
                alpha=0.9,
                label=r"$V_\tau(s_t)$",
            )
            value_axis.set_ylabel(r"Frozen-Teacher value $V_\tau$")
            for handle, label in (
                (value_next_handle, r"$V_\tau(s_{t+1})$"),
                (value_current_handle, r"$V_\tau(s_t)$"),
            ):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

            raw_return = sum(float(item["raw_reward"]) for item in rows)
            terminal = rows[-1]
            terminal_flags = ", ".join(
                name
                for name in ("terminated", "command_finished", "time_limit")
                if bool(terminal[name])
            )
            termination_causes = str(terminal["termination_causes"])
            cause_line = f"\ncause={termination_causes}" if termination_causes else ""
            onset_line = ""
            if outcome == OUTCOME_FAILURE:
                onset_line = (
                    "\ndiagnostic onset=not detected"
                    if diagnostic_onset is None
                    else (
                        f"\ndiagnostic onset={diagnostic_onset.episode_step}, "
                        f"prevention rows={diagnostic_onset.precursor_start_step}"
                        f"..{diagnostic_onset.precursor_end_step}"
                    )
                )
            axis.set_title(
                f"{outcome.title()} {trajectory.sample_index} | "
                f"env {trajectory.env_index}, motion {int(rows[0]['motion_id'])}, "
                f"length {len(rows)}, return {raw_return:.3f}\n"
                f"terminal={terminal_flags or 'unknown'}, "
                f"pure_teacher={trajectory.pure_teacher}{cause_line}{onset_line}",
                color=title_color,
                fontweight="bold",
            )

    clip_note = (
        ""
        if potential_clip is None
        else f"; potential clipped to +/-{float(potential_clip):g}"
    )
    figure.suptitle(
        "Frozen-Teacher critic shaping and detector-residual diagnostic\n"
        r"$\delta_t^T=r_t+\gamma c_tB_t-V_\tau(s_t)$; "
        r"$\widetilde{r}_t=r_t+\lambda_V[\gamma c_tB_t-V_\tau(s_t)]$"
        f"  ($\\gamma={float(gamma):g}$ temporal discount, "
        f"$\\lambda_V={float(tvkd_lambda):g}$ shaping weight{clip_note})\n"
        r"$B_t=V_\tau(s_{t+1})$ ordinary; $0$ physical terminal; "
        r"$V_\tau(s_t)$ command completion/time-limit"
        "\nSquare: plot-only diagnostic onset where MA5 "
        r"$\delta_t^T<-0.05$ for 3 consecutive endpoints; shaded: "
        "up to 10 strictly pre-onset prevention rows",
        fontsize=13,
        y=0.99,
    )
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.87))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _even_positive_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("video dimensions must be even for H.264")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture two successful and two physically failed forced-Teacher "
            "trajectories and plot critic-shaped rewards and frozen-Teacher values."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fresh train-phase PPO checkpoint used by TVKD prefill.",
    )
    parser.add_argument(
        "--task",
        default="G1/vaic/skateboard_stu",
        help="Hydra task selection (default: G1/vaic/skateboard_stu).",
    )
    parser.add_argument("--num-envs", type=_positive_int, default=128)
    parser.add_argument("--num-success", type=_positive_int, default=2)
    parser.add_argument("--num-failure", type=_positive_int, default=2)
    parser.add_argument(
        "--random-start",
        action="store_true",
        help=(
            "Use production prefill's random motion-phase resets. By default "
            "the diagnostic forces command frame zero while retaining training "
            "randomization."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=5000,
        help="Maximum vectorized control steps before writing a partial result.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cuda:0",),
        default="cuda:0",
        help="Single-process TVKD is locked to the local cuda:0 device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/tvkd_teacher_delta"),
    )
    parser.add_argument(
        "--perception-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional Student perception warm start. This does not change the "
            "frozen Teacher action/value modules."
        ),
    )
    parser.add_argument(
        "--allow-student-fallback",
        action="store_true",
        help="Allow episodes containing an invalid-Teacher Student fallback.",
    )
    parser.add_argument(
        "--record-videos",
        "--record-video",
        dest="record_videos",
        action="store_true",
        help=(
            "Capture exact robot/object kinematic poses during collection and render "
            "selected episodes with red reference-keypoint markers, including their "
            "pre-reset terminal poses, to MP4."
        ),
    )
    parser.add_argument(
        "--videos-per-outcome",
        type=_positive_int,
        default=2,
        help="Number of selected success and failure episodes to film (default: 2).",
    )
    parser.add_argument("--video-width", type=_even_positive_int, default=960)
    parser.add_argument("--video-height", type=_even_positive_int, default=540)
    parser.add_argument(
        "--video-stride",
        type=_positive_int,
        default=1,
        help="Write every Nth simulator control frame (default: 1).",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional TVKD Hydra override; may be repeated.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace):
    """Compose and validate the same fresh TVKD configuration as training."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from scripts import TVKD_fasSAC_bc_dagger as tvkd_entry

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO Teacher checkpoint does not exist: {checkpoint}")
    overrides = [
        f"task={args.task}",
        "fastsac_dagger_iterations=1",
        f"task.num_envs={int(args.num_envs)}",
        "headless=true",
        "eval_render=false",
        "wandb.mode=disabled",
        *args.hydra_override,
    ]
    config_dir = REPO_ROOT / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=overrides,
        )
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.checkpoint_path = str(checkpoint)
        cfg.seed = int(args.seed)
        if not bool(args.random_start):
            # High is exclusive, so randint(0, 1) always selects motion frame
            # zero.  Unlike env.eval(), this retains training reset noise and
            # domain randomization.
            cfg.task.command.reset_range = [0, 1]
        if bool(args.record_videos):
            cfg.app.enable_cameras = True
            cfg.task.viewer.resolution = [
                int(args.video_width),
                int(args.video_height),
            ]
        if args.perception_checkpoint is not None:
            perception = args.perception_checkpoint.expanduser().resolve()
            if not perception.is_file():
                raise FileNotFoundError(
                    f"perception checkpoint does not exist: {perception}"
                )
            cfg.algo.load_pretrained_perception = True
            cfg.algo.perception_checkpoint_path = str(perception)
            cfg.algo.train_perception = True

        # Full training validation proves the replay ring can be filled before
        # its safety ceiling.  This read-only probe never fills replay, but it
        # still reuses that validator; raise only the inert ceiling so small
        # --num-envs smoke runs are not rejected for an irrelevant reason.
        required_prefill_rollouts = minimum_prefill_rollouts(
            capacity=int(cfg.algo.q_teacher_buffer_capacity),
            num_envs=int(cfg.task.num_envs),
            train_every=int(cfg.algo.train_every),
        )
        cfg.algo.teacher_prefill_max_rollouts = max(
            int(cfg.algo.teacher_prefill_max_rollouts),
            required_prefill_rollouts,
        )

    tvkd_entry._require_single_process_execution()
    tvkd_entry.apply_fastsac_dagger_iteration_controls(cfg)
    tvkd_entry._prepare_tvkd_fresh_source(cfg)
    tvkd_entry.validate_tvkd_fastsac_bc_dagger_config(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def _normalized_critic_observation(policy, td):
    """Concatenate transformed leaves; never use raw replay aliases here."""
    import torch

    chunks = []
    for key, width in zip(policy.q_critic_keys, policy._q_critic_widths):
        value = td.get(key, None)
        if value is None:
            raise KeyError(f"Teacher-value observation is missing key {key!r}")
        if int(value.shape[-1]) != int(width):
            raise ValueError(
                f"Teacher-value key {key!r} has width {value.shape[-1]}, "
                f"expected {width}"
            )
        chunks.append(value)
    result = torch.cat(chunks, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("Teacher-value observation contains NaN/Inf")
    return result


def _environment_motion_ids(policy, num_envs: int, device):
    import torch

    motion_ids = getattr(
        getattr(getattr(policy, "env", None), "command_manager", None),
        "motion_ids",
        None,
    )
    if not torch.is_tensor(motion_ids):
        return torch.zeros(num_envs, dtype=torch.long, device=device)
    motion_ids = (
        motion_ids.detach().to(device=device, dtype=torch.long).reshape(-1).clone()
    )
    if motion_ids.numel() != num_envs:
        raise RuntimeError("command motion IDs do not match vector environments")
    return motion_ids


def _raw_reference_phase(policy, td, num_envs: int):
    """Read the unnormalized HDMI phase that generated the current command."""
    import torch

    for key in ("reference_phase", "ref_motion_phase_"):
        value = td.get(key, None)
        if value is not None:
            return _scalar_per_env(value, num_envs).float().clamp(0.0, 1.0).clone()
    manager = getattr(getattr(policy, "env", None), "command_manager", None)
    command_t = getattr(manager, "t", None)
    motion_len = getattr(manager, "motion_len", None)
    if torch.is_tensor(command_t) and torch.is_tensor(motion_len):
        phase = command_t.detach().float() / motion_len.detach().float().clamp_min(1)
        phase = phase.reshape(-1)
        if phase.numel() != num_envs:
            raise RuntimeError("command reference phase does not match environments")
        return phase.clamp(0.0, 1.0).clone()
    raise KeyError(
        "raw reference phase is unavailable; refusing to report a VecNorm-"
        "normalized command component as physical phase"
    )


def _termination_cause_strings(next_td, num_envs: int) -> list[str]:
    """Return active physical termination leaves for every environment."""
    import torch

    stats = next_td.get("stats", None)
    if stats is None:
        return ["" for _ in range(num_envs)]
    labels = []
    flags = []
    for key, value in stats.items(True, True):
        key_tuple = (key,) if isinstance(key, str) else tuple(key)
        if len(key_tuple) >= 2 and key_tuple[0] == "termination":
            labels.append("/".join(str(part) for part in key_tuple[1:]))
            flags.append(_scalar_per_env(value, num_envs, boolean=True))
    if not flags:
        return ["" for _ in range(num_envs)]
    packed = torch.stack(flags, dim=-1).detach().cpu().tolist()
    return [
        "|".join(label for label, active in zip(labels, row) if bool(active))
        for row in packed
    ]


def _scalar_per_env(value, num_envs: int, *, boolean: bool = False):
    value = value.reshape(num_envs, -1)
    if boolean:
        return value.bool().any(dim=-1)
    if value.shape[-1] != 1:
        raise ValueError("expected exactly one scalar per environment")
    return value[:, 0]


def _validate_phase_zero_reset(policy, *, random_start: bool) -> None:
    """Prove that the diagnostic-only frame-zero reset was actually applied."""
    if random_start:
        return
    import torch

    manager = getattr(getattr(policy, "env", None), "command_manager", None)
    if manager is None:
        raise RuntimeError("phase-zero collection requires a command manager")
    if bool(getattr(manager, "replay_motion", False)):
        raise RuntimeError("replay_motion overrides the requested phase-zero reset")
    command_t = getattr(manager, "t", None)
    if not torch.is_tensor(command_t) or not bool((command_t == 1).all()):
        observed = None if command_t is None else command_t.detach().cpu().tolist()
        raise RuntimeError(
            "phase-zero reset did not produce the expected first reference "
            f"frame (command_manager.t must equal 1, observed {observed})"
        )


def _motion_lengths(policy) -> list[int]:
    """Return the resolved source-motion lengths without importing the dataset."""
    import torch

    manager = getattr(getattr(policy, "env", None), "command_manager", None)
    lengths = getattr(getattr(manager, "dataset", None), "lengths", None)
    if not torch.is_tensor(lengths):
        return []
    return [int(value) for value in lengths.detach().cpu().reshape(-1).tolist()]


@dataclass(frozen=True)
class _SceneAssetSlot:
    name: str
    asset: Any
    joint_count: int
    offset: int


class _SceneStateCodec:
    """Pack exact poses and reference keypoints for retrospective MP4 output."""

    def __init__(self, policy) -> None:
        base = getattr(policy, "env", None)
        manager = getattr(base, "command_manager", None)
        if base is None or manager is None:
            raise RuntimeError("video state capture requires the base environment")
        candidates = [("robot", getattr(base, "robot", None))]
        candidates.extend(
            (
                name,
                getattr(manager, name, None),
            )
            for name in ("object", "object2")
        )
        candidates.extend(
            (f"extra_object_{index}", asset)
            for index, asset in enumerate(getattr(manager, "extra_objects", ()))
        )
        self.base = base
        self.slots: list[_SceneAssetSlot] = []
        seen = set()
        offset = 0
        for name, asset in candidates:
            if asset is None or id(asset) in seen:
                continue
            seen.add(id(asset))
            root_pose = getattr(asset.data, "root_link_pose_w", None)
            if root_pose is None or int(root_pose.shape[-1]) != 7:
                raise RuntimeError(f"scene asset {name!r} has no root-link pose")
            joint_pos = getattr(asset.data, "joint_pos", None)
            joint_count = 0 if joint_pos is None else int(joint_pos.shape[-1])
            self.slots.append(
                _SceneAssetSlot(
                    name=name,
                    asset=asset,
                    joint_count=joint_count,
                    offset=offset,
                )
            )
            offset += 7 + joint_count
        if not self.slots:
            raise RuntimeError("no renderable robot/object assets were found")
        reference_positions = getattr(manager, "ref_body_pos_w", None)
        marker_visualizer = getattr(manager, "vis_markers", None)
        marker_names = list(getattr(manager, "tracking_keypoint_names", ()))
        if (
            reference_positions is None
            or marker_visualizer is None
            or reference_positions.ndim != 3
            or int(reference_positions.shape[0]) != int(base.num_envs)
            or int(reference_positions.shape[-1]) != 3
        ):
            raise RuntimeError(
                "video reference markers require command_manager.ref_body_pos_w "
                "and its play.py VisualizationMarkers"
            )
        self.manager = manager
        self.reference_marker_visualizer = marker_visualizer
        self.reference_marker_offset = offset
        self.reference_marker_count = int(reference_positions.shape[1])
        self.reference_marker_names = marker_names
        if self.reference_marker_count < 1:
            raise RuntimeError("video reference marker set is empty")
        if marker_names and len(marker_names) != self.reference_marker_count:
            raise RuntimeError("reference marker names do not match marker positions")
        offset += self.reference_marker_count * 3
        self.width = offset

    @property
    def layout(self) -> list[dict[str, Any]]:
        asset_layout = [
            {
                "name": slot.name,
                "root_pose_width": 7,
                "joint_position_width": slot.joint_count,
                "offset": slot.offset,
            }
            for slot in self.slots
        ]
        return [
            *asset_layout,
            {
                "name": "reference_keypoints",
                "position_width": self.reference_marker_count * 3,
                "keypoint_count": self.reference_marker_count,
                "keypoint_names": self.reference_marker_names,
                "offset": self.reference_marker_offset,
                "marker_color_rgb": [1.0, 0.0, 0.0],
                "marker_radius": 0.04,
            },
        ]

    def snapshot(self):
        import torch

        chunks = []
        for slot in self.slots:
            chunks.append(slot.asset.data.root_link_pose_w.detach())
            if slot.joint_count:
                chunks.append(slot.asset.data.joint_pos.detach())
        reference_positions = self.manager.ref_body_pos_w.detach()
        chunks.append(reference_positions.reshape(int(self.base.num_envs), -1))
        packed = torch.cat(chunks, dim=-1)
        if int(packed.shape[-1]) != self.width or not torch.isfinite(packed).all():
            raise RuntimeError("scene-state snapshot is malformed or non-finite")
        return packed.to(device="cpu", dtype=torch.float32).clone()

    def restore(self, state, *, env_index: int) -> None:
        import torch

        if tuple(state.shape) != (self.width,):
            raise ValueError(
                f"scene state has shape {tuple(state.shape)}, expected ({self.width},)"
            )
        device = self.base.device
        env_ids = torch.tensor([int(env_index)], dtype=torch.long, device=device)
        for slot in self.slots:
            cursor = slot.offset
            root_pose = state[cursor : cursor + 7].to(device).reshape(1, 7)
            slot.asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
            slot.asset.write_root_com_velocity_to_sim(
                torch.zeros((1, 6), device=device), env_ids=env_ids
            )
            cursor += 7
            if slot.joint_count:
                joint_pos = (
                    state[cursor : cursor + slot.joint_count]
                    .to(device)
                    .reshape(1, slot.joint_count)
                )
                slot.asset.write_joint_state_to_sim(
                    joint_pos,
                    torch.zeros_like(joint_pos),
                    env_ids=env_ids,
                )
        self.base.sim.forward()
        self.base.scene.update(0.0)
        reference_positions = (
            state[
                self.reference_marker_offset : self.reference_marker_offset
                + self.reference_marker_count * 3
            ]
            .to(device)
            .reshape(self.reference_marker_count, 3)
        )
        # Marker index 1 is the red reference sphere configured by the HDMI
        # command manager.  Supplying only these positions hides the green
        # robot markers used by interactive play.py.
        self.reference_marker_visualizer.visualize(
            translations=reference_positions,
            marker_indices=[1] * self.reference_marker_count,
        )

    def park_except(self, env_index: int) -> None:
        """Move other vectorized robots/objects below the ground for a clean view."""
        import torch

        all_ids = torch.arange(self.base.num_envs, device=self.base.device)
        env_ids = all_ids[all_ids != int(env_index)]
        if not len(env_ids):
            return
        for slot in self.slots:
            root_pose = slot.asset.data.root_link_pose_w[env_ids].clone()
            root_pose[:, 2] = -1000.0
            slot.asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
            slot.asset.write_root_com_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.base.device),
                env_ids=env_ids,
            )
        self.base.sim.forward()
        self.base.scene.update(0.0)


def _capture_rgb_for_env(env, policy, args: argparse.Namespace, env_index: int):
    """Render one vector environment through the shared Isaac viewport."""
    import numpy as np
    import torch

    base = getattr(policy, "env", None)
    if base is None or not hasattr(base, "sim") or not hasattr(base, "scene"):
        raise RuntimeError("policy does not expose the Isaac environment for video")
    origins = getattr(base.scene, "env_origins", None)
    if not torch.is_tensor(origins) or not 0 <= int(env_index) < len(origins):
        raise RuntimeError(f"cannot resolve camera origin for environment {env_index}")
    root_position = base.robot.data.root_link_pos_w[int(env_index)].detach().cpu()
    configured_eye = torch.as_tensor(base.cfg.viewer.eye).cpu().float()
    configured_target = torch.as_tensor(base.cfg.viewer.lookat).cpu().float()
    view_vector = configured_eye - configured_target
    target = root_position.float()
    target[2] += float(configured_target[2])
    eye = target + view_vector
    base.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
    frame = np.asarray(env.render(mode="rgb_array"))
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise RuntimeError(
            "Isaac RGB annotator returned an invalid frame: "
            f"shape={frame.shape}, dtype={frame.dtype}"
        )
    expected = (int(args.video_height), int(args.video_width))
    if tuple(frame.shape[:2]) != expected:
        raise RuntimeError(
            f"Isaac RGB frame is {frame.shape[1]}x{frame.shape[0]}, expected "
            f"{expected[1]}x{expected[0]}"
        )
    return np.ascontiguousarray(frame)


def _video_pose_indices(num_transitions: int, stride: int) -> list[int]:
    """Return captured pose indices, always retaining the terminal pose."""
    if isinstance(num_transitions, bool) or int(num_transitions) < 1:
        raise ValueError("num_transitions must be a positive integer")
    if isinstance(stride, bool) or int(stride) < 1:
        raise ValueError("stride must be a positive integer")
    terminal_index = int(num_transitions)
    indices = list(range(0, terminal_index + 1, int(stride)))
    if indices[-1] != terminal_index:
        indices.append(terminal_index)
    return indices


def _video_timestep_text(
    *,
    pose_index: int,
    num_transitions: int,
    step_dt: float,
    outcome: str,
) -> str:
    """Describe an exact captured state and its corresponding graph row."""
    if isinstance(num_transitions, bool) or int(num_transitions) < 1:
        raise ValueError("num_transitions must be a positive integer")
    if isinstance(pose_index, bool) or not 0 <= int(pose_index) <= int(num_transitions):
        raise ValueError("pose_index must be between 0 and num_transitions")
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    pose_index = int(pose_index)
    num_transitions = int(num_transitions)
    source_time = pose_index * float(step_dt)
    first_line = (
        f"{str(outcome).upper()} | state step {pose_index}/{num_transitions} | "
        f"time {source_time:.2f} s"
    )
    if pose_index == 0:
        second_line = "initial pose | graph transition not started"
    elif pose_index == num_transitions:
        second_line = (
            f"terminal pose (pre-reset) | after graph transition {pose_index - 1}"
        )
    else:
        second_line = f"after graph transition {pose_index - 1}"
    return f"{first_line}\n{second_line}"


@lru_cache(maxsize=8)
def _video_overlay_font(frame_height: int):
    from PIL import ImageFont

    font_size = max(18, int(frame_height) // 22)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _overlay_video_timestep(
    frame,
    *,
    pose_index: int,
    num_transitions: int,
    step_dt: float,
    outcome: str,
):
    """Draw source-state time and graph-transition alignment onto an RGB frame."""
    import numpy as np
    from PIL import Image, ImageDraw

    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("video overlay expects an HxWx3 uint8 RGB frame")
    image = Image.fromarray(np.ascontiguousarray(source)).copy()
    draw = ImageDraw.Draw(image)
    font = _video_overlay_font(int(source.shape[0]))
    text = _video_timestep_text(
        pose_index=pose_index,
        num_transitions=num_transitions,
        step_dt=step_dt,
        outcome=outcome,
    )
    spacing = max(3, int(source.shape[0]) // 180)
    left = max(10, int(source.shape[1]) // 80)
    top = max(10, int(source.shape[0]) // 45)
    padding = max(8, int(source.shape[0]) // 54)
    bounds = draw.multiline_textbbox(
        (left, top),
        text,
        font=font,
        spacing=spacing,
        stroke_width=1,
    )
    draw.rounded_rectangle(
        (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding,
        ),
        radius=max(5, padding // 2),
        fill=(8, 15, 27),
    )
    draw.multiline_text(
        (left, top),
        text,
        font=font,
        fill=(255, 255, 255),
        spacing=spacing,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return np.ascontiguousarray(np.asarray(image))


class _VideoStream:
    """Stream one candidate episode to a temporary MP4 until replay validates."""

    def __init__(
        self,
        trajectory: CapturedTrajectory,
        *,
        video_dir: Path,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        import imageio.v2 as imageio

        self.trajectory = trajectory
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frames = 0
        self.closed = False
        video_dir.mkdir(parents=True, exist_ok=True)
        suffix = (
            f"_{trajectory.outcome}_{trajectory.sample_index}_"
            f"env{trajectory.env_index}_episode{trajectory.episode_index}.mp4"
        )
        handle = tempfile.NamedTemporaryFile(
            prefix=".teacher_delta_replay_",
            suffix=suffix,
            dir=video_dir,
            delete=False,
        )
        handle.close()
        self.temp_path = Path(handle.name)
        self.final_path = video_dir / (
            f"{trajectory.outcome}_{trajectory.sample_index}_"
            f"env{trajectory.env_index}_episode{trajectory.episode_index}.mp4"
        )
        self.writer = imageio.get_writer(
            str(self.temp_path),
            fps=self.fps,
            codec="libx264",
            macro_block_size=None,
        )

    def append(self, frame) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed video stream")
        self.writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True

    def commit(self) -> VideoArtifact:
        self.close()
        os.replace(self.temp_path, self.final_path)
        return VideoArtifact(
            outcome=self.trajectory.outcome,
            sample_index=self.trajectory.sample_index,
            env_index=self.trajectory.env_index,
            episode_index=self.trajectory.episode_index,
            path=self.final_path,
            frames=self.frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    def abort(self) -> None:
        try:
            self.close()
        finally:
            self.temp_path.unlink(missing_ok=True)


def collect_trajectories(
    env,
    policy,
    cfg,
    args: argparse.Namespace,
    *,
    scene_codec: _SceneStateCodec | None = None,
):
    """Collect complete live prefill episodes without replay or training."""
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from tqdm import tqdm

    from active_adaptation.learning.ppo.ppo_bc_dagger import (
        DAGGER_IS_STUDENT_ACTION_KEY,
        DAGGER_TEACHER_ACTION_VALID_KEY,
    )

    if not (
        hasattr(policy, "is_teacher_prefill_active")
        and policy.is_teacher_prefill_active()
    ):
        raise RuntimeError("fresh TVKD policy did not enter Teacher prefill mode")
    if not hasattr(policy, "get_frozen_teacher_value"):
        raise TypeError("selected policy has no frozen Teacher value interface")

    policy.teacher_value_wrapper.freeze()
    env.train()
    env.set_seed(int(args.seed))
    rollout_policy = policy.get_rollout_policy("train")
    carry = env.reset()
    _validate_phase_zero_reset(policy, random_start=bool(args.random_start))
    scene_state_before = None if scene_codec is None else scene_codec.snapshot()
    num_envs = int(env.num_envs)
    start_iteration = int(env.current_iter)
    reservoir = EpisodeReservoir(
        num_envs,
        num_success=int(args.num_success),
        num_failure=int(args.num_failure),
        require_pure_teacher=not bool(args.allow_student_fallback),
    )
    gamma = float(cfg.algo.gamma)
    tvkd_lambda = float(cfg.algo.tvkd_lambda)
    potential_clip = cfg.algo.tvkd_potential_clip
    train_every = int(cfg.algo.train_every)

    progress = tqdm(
        range(int(args.max_steps)),
        desc="Teacher delta trajectories",
        unit="step",
        miniters=5,
    )
    completed_steps = 0
    with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
        for environment_step in progress:
            if environment_step % train_every == 0:
                env.set_progress(
                    prefill_progress_iteration(
                        start_iteration=start_iteration,
                        environment_step=environment_step,
                        train_every=train_every,
                    )
                )
            current_observation = _normalized_critic_observation(policy, carry)
            phase = _raw_reference_phase(policy, carry, num_envs)
            motion_ids = _environment_motion_ids(policy, num_envs, carry.device)
            is_init = _scalar_per_env(
                carry.get(
                    "is_init",
                    torch.zeros(num_envs, 1, dtype=torch.bool, device=carry.device),
                ),
                num_envs,
                boolean=True,
            )
            step_count_value = carry.get("step_count", None)
            step_count = (
                torch.full(
                    (num_envs,),
                    -1,
                    dtype=torch.long,
                    device=carry.device,
                )
                if step_count_value is None
                else _scalar_per_env(step_count_value, num_envs).long()
            )

            acted = rollout_policy(carry)
            teacher_valid = _scalar_per_env(
                acted[DAGGER_TEACHER_ACTION_VALID_KEY],
                num_envs,
                boolean=True,
            )
            student_fallback = _scalar_per_env(
                acted[DAGGER_IS_STUDENT_ACTION_KEY],
                num_envs,
                boolean=True,
            )
            if scene_codec is None:
                transition, carry = env.step_and_maybe_reset(acted)
                scene_state_after = None
                next_scene_state_before = None
            else:
                # Keep the physical terminal pose before TorchRL autoresets it.
                transition = env.step(acted)
                scene_state_after = scene_codec.snapshot()
                next_carry = env.step_mdp(transition)
                carry = env.maybe_reset(next_carry)
                next_scene_state_before = scene_codec.snapshot()
            next_td = transition.get("next")
            next_observation = _normalized_critic_observation(policy, next_td)
            joint_observation = torch.cat(
                (current_observation, next_observation), dim=0
            )
            joint_value = policy.get_frozen_teacher_value(joint_observation)
            teacher_v, teacher_v_next = joint_value.split(num_envs, dim=0)

            raw_reward = policy._scalarize_q_reward(next_td["reward"]).reshape(num_envs)
            done = _scalar_per_env(next_td["done"], num_envs, boolean=True)
            terminated = _scalar_per_env(next_td["terminated"], num_envs, boolean=True)
            command_finished = _scalar_per_env(
                next_td["stats", "command_finished"],
                num_envs,
                boolean=True,
            )
            time_limit = _scalar_per_env(
                next_td["stats", "episode_time_limit"],
                num_envs,
                boolean=True,
            )
            known_boundary = terminated | command_finished | time_limit
            if bool((done ^ known_boundary).any()):
                raise RuntimeError(
                    "environment emitted a done row without a known boundary "
                    "cause (or a cause without done)"
                )
            replay_discount = _scalar_per_env(next_td["discount"], num_envs).float()
            success_stat = _scalar_per_env(
                next_td["stats", "success"], num_envs
            ).float()
            termination_causes = _termination_cause_strings(next_td, num_envs)
            terms = compute_probe_terms(
                teacher_v=teacher_v,
                teacher_v_next=teacher_v_next,
                raw_reward=raw_reward,
                terminated=terminated,
                command_finished=command_finished,
                time_limit=time_limit,
                replay_discount=replay_discount,
                gamma=gamma,
                tvkd_lambda=tvkd_lambda,
                potential_clip=(
                    None if potential_clip is None else float(potential_clip)
                ),
            )

            metric_names = (
                "raw_reward",
                "teacher_v",
                "teacher_v_next",
                "value_change",
                "boundary_teacher_v",
                "teacher_continuation",
                "fixed_teacher_continuation",
                "literal_delta",
                "production_delta",
                "potential_delta",
                "shaped_reward",
                "replay_discount",
                "reference_phase",
                "step_count",
                "motion_id",
                "done",
                "terminated",
                "command_finished",
                "time_limit",
                "environment_success_stat",
                "is_init",
                "teacher_action_valid",
                "student_fallback",
            )
            metric_tensors = (
                raw_reward,
                terms["teacher_v"],
                terms["teacher_v_next"],
                terms["value_change"],
                terms["boundary_teacher_v"],
                terms["teacher_continuation"],
                terms["fixed_teacher_continuation"],
                terms["literal_delta"],
                terms["production_delta"],
                terms["potential_delta"],
                terms["shaped_reward"],
                replay_discount,
                phase,
                step_count,
                motion_ids,
                done,
                terminated,
                command_finished,
                time_limit,
                success_stat,
                is_init,
                teacher_valid,
                student_fallback,
            )
            packed = torch.stack(
                [value.reshape(num_envs).float() for value in metric_tensors],
                dim=-1,
            ).detach()
            packed_rows = packed.cpu().tolist()
            boolean_names = {
                "done",
                "terminated",
                "command_finished",
                "time_limit",
                "is_init",
                "teacher_action_valid",
                "student_fallback",
            }
            integer_names = {"step_count", "motion_id"}
            for env_index, values in enumerate(packed_rows):
                row = {
                    name: (
                        bool(value)
                        if name in boolean_names
                        else int(value)
                        if name in integer_names
                        else float(value)
                    )
                    for name, value in zip(metric_names, values)
                }
                row["environment_step"] = int(environment_step)
                row["termination_causes"] = termination_causes[env_index]
                reservoir.add_step(
                    env_index,
                    row,
                    scene_state_before=(
                        None
                        if scene_state_before is None
                        else scene_state_before[env_index].clone()
                    ),
                    scene_state_after=(
                        None
                        if scene_state_after is None
                        else scene_state_after[env_index].clone()
                    ),
                )

            scene_state_before = next_scene_state_before

            completed_steps = environment_step + 1
            progress.set_postfix(
                success=(
                    f"{len(reservoir.selected[OUTCOME_SUCCESS])}/"
                    f"{reservoir.quotas[OUTCOME_SUCCESS]}"
                ),
                failure=(
                    f"{len(reservoir.selected[OUTCOME_FAILURE])}/"
                    f"{reservoir.quotas[OUTCOME_FAILURE]}"
                ),
                timeout=reservoir.outcome_counts[OUTCOME_TIMEOUT],
                refresh=False,
            )
            if reservoir.complete:
                break
    progress.close()
    return reservoir, completed_steps, start_iteration


def record_selected_videos(
    env,
    policy,
    args: argparse.Namespace,
    selected: Mapping[str, Sequence[CapturedTrajectory]],
    *,
    scene_codec: _SceneStateCodec,
    output_dir: Path,
) -> list[VideoArtifact]:
    """Render exact captured kinematic poses without rerunning the dynamics."""
    import torch
    from tqdm import tqdm

    targets = select_video_targets(
        selected,
        per_outcome=int(args.videos_per_outcome),
    )
    if not targets:
        return []
    for target in targets:
        if not target.pure_teacher and not bool(args.allow_student_fallback):
            raise RuntimeError("refusing to film an impure Teacher trajectory")
        if len(target.scene_states) != len(target.rows) + 1:
            raise RuntimeError(
                f"{target.outcome} {target.sample_index} has "
                f"{len(target.scene_states)} scene states for "
                f"{len(target.rows)} transitions"
            )

    step_dt = float(env.step_dt)
    stride = int(args.video_stride)
    fps = 1.0 / (step_dt * stride)
    streams: list[_VideoStream] = []
    try:
        with torch.inference_mode():
            # Prime Replicator once before sending frames to ffmpeg.
            scene_codec.park_except(targets[0].env_index)
            scene_codec.restore(
                targets[0].scene_states[0], env_index=targets[0].env_index
            )
            _capture_rgb_for_env(env, policy, args, targets[0].env_index)
            _capture_rgb_for_env(env, policy, args, targets[0].env_index)

            for target in targets:
                scene_codec.park_except(target.env_index)
                stream = _VideoStream(
                    target,
                    video_dir=output_dir / "videos",
                    fps=fps,
                    width=int(args.video_width),
                    height=int(args.video_height),
                )
                streams.append(stream)
                frame_indices = _video_pose_indices(len(target.rows), stride)
                for frame_index in tqdm(
                    frame_indices,
                    desc=f"{target.outcome.title()} {target.sample_index} video",
                    unit="frame",
                    miniters=5,
                ):
                    scene_codec.restore(
                        target.scene_states[frame_index],
                        env_index=target.env_index,
                    )
                    frame = _capture_rgb_for_env(env, policy, args, target.env_index)
                    stream.append(
                        _overlay_video_timestep(
                            frame,
                            pose_index=frame_index,
                            num_transitions=len(target.rows),
                            step_dt=step_dt,
                            outcome=target.outcome,
                        )
                    )
                stream.close()

        artifacts = [stream.commit() for stream in streams]
        return sorted(
            artifacts,
            key=lambda artifact: (
                0 if artifact.outcome == OUTCOME_SUCCESS else 1,
                artifact.sample_index,
            ),
        )
    except BaseException:
        for stream in streams:
            stream.abort()
        raise


def _close_simulation_app(simulation_app) -> None:
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    except TypeError:
        simulation_app.close()


def _preflight_matplotlib() -> None:
    """Fail before Isaac startup if the graph dependency is unavailable."""
    matplotlib_cache = Path(tempfile.gettempdir()) / "vaic-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the Teacher-delta probe requires matplotlib to write its PNG"
        ) from exc
    matplotlib.use("Agg", force=True)


def _preflight_video() -> None:
    """Fail before Isaac startup if MP4 encoding support is unavailable."""
    try:
        import imageio.v2  # noqa: F401
        import imageio_ffmpeg
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--record-videos requires imageio, imageio-ffmpeg, and Pillow"
        ) from exc
    try:
        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("imageio could not locate an ffmpeg executable") from exc


def run(args: argparse.Namespace) -> Path:
    from omegaconf import OmegaConf

    from scripts._isaaclab_bootstrap import AppLauncher
    from scripts.helpers import make_env_policy

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _preflight_matplotlib()
    if bool(args.record_videos):
        _preflight_video()
    cfg = build_config(args)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=False,
        device=str(args.device),
    )
    simulation_app = app_launcher.app
    env = None
    try:
        # False is intentional: raw replay aliases are not needed, and the
        # frozen PPO critic must receive the transformed/normalized leaves.
        env, policy, _vecnorm = make_env_policy(
            cfg,
            configure_replay=False,
            inference_only=False,
        )
        # Match training's second config save: make_env_policy resolves and
        # mutates runtime-only fields while constructing transforms/policy.
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
        scene_codec = _SceneStateCodec(policy) if bool(args.record_videos) else None
        reservoir, completed_steps, start_iteration = collect_trajectories(
            env,
            policy,
            cfg,
            args,
            scene_codec=scene_codec,
        )
        video_artifacts = []
        if bool(args.record_videos) and reservoir.complete:
            video_artifacts = record_selected_videos(
                env,
                policy,
                args,
                reservoir.selected,
                scene_codec=scene_codec,
                output_dir=output_dir,
            )
        image_path = output_dir / "teacher_delta_trajectories.png"
        csv_path = output_dir / "teacher_delta_trajectories.csv"
        summary_path = output_dir / "summary.json"
        write_csv(reservoir.selected, csv_path)
        plot_trajectories(
            reservoir.selected,
            image_path,
            gamma=float(cfg.algo.gamma),
            tvkd_lambda=float(cfg.algo.tvkd_lambda),
            potential_clip=cfg.algo.tvkd_potential_clip,
        )
        metadata = {
            "schema_version": 1,
            "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
            "task": str(args.task),
            "resolved_task_name": str(cfg.task.name),
            "seed": int(args.seed),
            "device": str(env.device),
            "num_envs": int(env.num_envs),
            "requested_num_envs": int(args.num_envs),
            "requested_counts": {
                OUTCOME_SUCCESS: int(args.num_success),
                OUTCOME_FAILURE: int(args.num_failure),
            },
            "hydra_overrides": list(args.hydra_override),
            "motion_start_mode": (
                "random_prefill" if bool(args.random_start) else "phase_zero"
            ),
            "source_motion_lengths": _motion_lengths(policy),
            "source_start_iteration": int(start_iteration),
            "completed_vector_steps": int(completed_steps),
            "max_vector_steps": int(args.max_steps),
            "require_pure_teacher": not bool(args.allow_student_fallback),
            "videos": [
                video_artifact_summary(artifact) for artifact in video_artifacts
            ],
            "video_capture": (
                {
                    "method": "captured_kinematic_pose_playback",
                    "state_layout": scene_codec.layout,
                    "exact_source_episode_poses": True,
                    "dynamics_rerun": False,
                    "video_stride": int(args.video_stride),
                    "timestep_overlay": True,
                    "reference_keypoint_markers": {
                        "enabled": True,
                        "source": "command_manager.ref_body_pos_w",
                        "count": int(scene_codec.reference_marker_count),
                        "names": list(scene_codec.reference_marker_names),
                        "color_rgb": [1.0, 0.0, 0.0],
                        "radius": 0.04,
                        "measured_green_markers_visible": False,
                    },
                    "timestep_overlay_semantics": (
                        "overlay state step k is captured pose s_k at source time "
                        "k*step_dt; k=0 is the initial pose, k>0 follows graph "
                        "transition k-1, and k=L is the pre-reset terminal pose"
                    ),
                    "frame_mapping": (
                        "CSV transition row t maps captured pose s_t to s_(t+1); "
                        "for stride=1 these are video frames t and t+1"
                    ),
                }
                if scene_codec is not None
                else None
            ),
            "complete": bool(reservoir.complete),
            "outcome_counts": reservoir.outcome_counts,
            "rejected_impure_counts": reservoir.rejected_impure_counts,
            "incomplete_episode_count": reservoir.incomplete_episode_count,
            "open_episode_count": int(
                sum(bool(history) for history in reservoir.histories)
            ),
            "open_transition_count": int(
                sum(len(history) for history in reservoir.histories)
            ),
            "selected_counts": {
                outcome: len(trajectories)
                for outcome, trajectories in reservoir.selected.items()
            },
            "gamma": float(cfg.algo.gamma),
            "tvkd_lambda": float(cfg.algo.tvkd_lambda),
            "tvkd_potential_clip": cfg.algo.tvkd_potential_clip,
            "teacher_value_critic_keys": list(policy.q_critic_keys),
            "teacher_value_critic_widths": [
                int(width) for width in policy._q_critic_widths
            ],
            "equations": {
                "literal_requested": "r_t + V_T(s_{t+1}) - V_T(s_t)",
                "production_failure_residual": "r_t + C_T(t) - V_T(s_t)",
                "production_continuation": (
                    "gamma * replay_discount * boundary_value; boundary_value="
                    "0 on physical termination, V_T(s_t) on command completion/"
                    "time limit, otherwise V_T(s_{t+1})"
                ),
                "critic_shaped_reward": (
                    "r_t + tvkd_lambda * (fixed_C_T(t) - fixed_V_T(s_t))"
                ),
                "diagnostic_onset_rolling_mean": (
                    "m_t = mean(delta^T_(t-4), ..., delta^T_t)"
                ),
                "diagnostic_onset": (
                    "first t where m_t, m_(t+1), and m_(t+2) are each < -0.05"
                ),
            },
            "diagnostic_failure_onset": {
                "plot_only": True,
                "production_student_sampler_contract": (
                    "v5 uses the same raw MA5/-0.05/K3/pre-onset-10 rule, "
                    "additionally gated to Student-executed replay-valid rows"
                ),
                "residual": "production_delta in raw reward/return units",
                "smoothing_window": ONSET_SMOOTHING_WINDOW,
                "raw_threshold": ONSET_RAW_THRESHOLD,
                "minimum_consecutive": ONSET_MIN_CONSECUTIVE,
                "minimum_replay_step": ONSET_MIN_REPLAY_STEP,
                "require_full_smoothing_window_after_burn_in": True,
                "terminal_exclusion_steps_before_terminal": (
                    ONSET_TERMINAL_EXCLUSION_STEPS
                ),
                "terminal_row_excluded": True,
                "square_marker_y": "raw production_delta at onset",
                "square_annotation_value": "5-step rolling production_delta",
                "prevention_sampling_hypothesis": (
                    "replay-valid Student rows within the 10 chronological "
                    "transition steps strictly before onset"
                ),
                "prevention_steps": ONSET_PRECURSOR_STEPS,
                "onset_included_in_prevention_window": False,
                "pure_teacher_probe_note": (
                    "source-agnostic diagnostic on physical-failure probe rows; "
                    "not the production Student-only detector"
                ),
            },
            "collection_semantics": (
                "complete episodes under the production forced-Teacher prefill "
                "rollout policy; command reset starts from frame zero while "
                "training-mode reset/domain randomization remains enabled"
                if not bool(args.random_start)
                else "complete episodes from production random-phase resets "
                "under the forced-Teacher prefill rollout policy"
            ),
            "files": {
                "image": str(image_path),
                "csv": str(csv_path),
                "resolved_config": str(output_dir / "resolved_config.yaml"),
            },
        }
        write_summary(reservoir.selected, summary_path, metadata)
        print(f"Teacher-delta graph: {image_path}", flush=True)
        print(f"Per-step values: {csv_path}", flush=True)
        print(f"Summary: {summary_path}", flush=True)
        for artifact in video_artifacts:
            print(
                f"{artifact.outcome.title()} video {artifact.sample_index}: "
                f"{artifact.path}",
                flush=True,
            )
        if not reservoir.complete:
            raise RuntimeError(
                "trajectory quota was not reached before --max-steps: "
                f"selected={metadata['selected_counts']}, "
                f"outcomes={reservoir.outcome_counts}, "
                f"impure_rejected={reservoir.rejected_impure_counts}. "
                "Increase --num-envs/--max-steps, or inspect whether "
                "--allow-student-fallback is scientifically appropriate."
            )
        return image_path
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            _close_simulation_app(simulation_app)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
