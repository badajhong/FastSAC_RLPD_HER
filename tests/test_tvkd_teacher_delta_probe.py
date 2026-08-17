from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROBE_PATH = Path(__file__).with_name("tvkd_teacher_delta_probe.py")
SPEC = importlib.util.spec_from_file_location("tvkd_teacher_delta_probe", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _row(**overrides):
    row = {
        "environment_step": 0,
        "step_count": 0,
        "reference_phase": 0.0,
        "motion_id": 0,
        "raw_reward": 1.0,
        "teacher_v": 1.0,
        "teacher_v_next": 1.1,
        "value_change": 0.1,
        "boundary_teacher_v": 1.1,
        "teacher_continuation": 1.089,
        "fixed_teacher_continuation": 1.089,
        "literal_delta": 1.1,
        "production_delta": 1.089,
        "potential_delta": 0.089,
        "shaped_reward": 1.02225,
        "replay_discount": 1.0,
        "done": False,
        "terminated": False,
        "command_finished": False,
        "time_limit": False,
        "environment_success_stat": 0.0,
        "is_init": False,
        "teacher_action_valid": True,
        "student_fallback": False,
        "termination_causes": "",
    }
    row.update(overrides)
    return row


def test_boundary_classification_matches_prefill_precedence():
    assert probe.classify_episode_boundary(
        done=False,
        terminated=False,
        command_finished=False,
        time_limit=False,
    ) is None
    assert probe.classify_episode_boundary(
        done=True,
        terminated=False,
        command_finished=True,
        time_limit=False,
    ) == probe.OUTCOME_SUCCESS
    assert probe.classify_episode_boundary(
        done=True,
        terminated=True,
        command_finished=True,
        time_limit=False,
    ) == probe.OUTCOME_FAILURE
    assert probe.classify_episode_boundary(
        done=True,
        terminated=False,
        command_finished=False,
        time_limit=True,
    ) == probe.OUTCOME_TIMEOUT


def test_probe_terms_expose_literal_and_runtime_boundary_difference():
    terms = probe.compute_probe_terms(
        teacher_v=torch.tensor([10.0, 10.0, 10.0, 10.0]),
        teacher_v_next=torch.tensor([12.0, 12.0, 12.0, 12.0]),
        raw_reward=torch.ones(4),
        terminated=torch.tensor([False, True, False, False]),
        command_finished=torch.tensor([False, False, True, False]),
        time_limit=torch.tensor([False, False, False, True]),
        replay_discount=torch.ones(4),
        gamma=0.9,
        tvkd_lambda=0.25,
        potential_clip=None,
    )

    assert terms["literal_delta"].tolist() == pytest.approx([3.0] * 4)
    assert terms["teacher_continuation"].tolist() == pytest.approx(
        [10.8, 0.0, 9.0, 9.0]
    )
    assert terms["production_delta"].tolist() == pytest.approx(
        [1.8, -9.0, 0.0, 0.0]
    )
    assert terms["shaped_reward"].tolist() == pytest.approx(
        [1.2, -1.5, 0.75, 0.75]
    )


def test_prefill_progress_and_reachability_helpers():
    assert probe.prefill_progress_iteration(
        start_iteration=6102,
        environment_step=0,
        train_every=32,
    ) == 6102
    assert probe.prefill_progress_iteration(
        start_iteration=6102,
        environment_step=64,
        train_every=32,
    ) == 6104
    assert probe.minimum_prefill_rollouts(
        capacity=131_072,
        num_envs=2,
        train_every=32,
    ) == 2048


def test_probe_terms_reject_nonfinite_reward():
    with pytest.raises(RuntimeError, match="raw_reward contains NaN/Inf"):
        probe.compute_probe_terms(
            teacher_v=torch.tensor([1.0]),
            teacher_v_next=torch.tensor([1.0]),
            raw_reward=torch.tensor([float("nan")]),
            terminated=torch.tensor([False]),
            command_finished=torch.tensor([False]),
            time_limit=torch.tensor([False]),
            replay_discount=torch.tensor([1.0]),
            gamma=0.99,
            tvkd_lambda=0.25,
            potential_clip=None,
        )


def test_episode_reservoir_spans_steps_and_rejects_impure_teacher():
    reservoir = probe.EpisodeReservoir(
        2,
        num_success=1,
        num_failure=1,
        require_pure_teacher=True,
    )
    reservoir.add_step(0, _row(is_init=True))
    reservoir.add_step(1, _row(is_init=True, student_fallback=True))
    selected_success = reservoir.add_step(
        0,
        _row(done=True, command_finished=True, environment_success_stat=1.0),
    )
    rejected_failure = reservoir.add_step(
        1,
        _row(done=True, terminated=True, student_fallback=True),
    )

    assert selected_success is not None
    assert selected_success.outcome == probe.OUTCOME_SUCCESS
    assert len(selected_success.rows) == 2
    assert rejected_failure is None
    assert reservoir.rejected_impure_counts[probe.OUTCOME_FAILURE] == 1

    reservoir.add_step(1, _row(is_init=True))
    selected_failure = reservoir.add_step(1, _row(done=True, terminated=True))
    assert selected_failure is not None
    assert selected_failure.outcome == probe.OUTCOME_FAILURE
    assert reservoir.complete


def test_episode_reservoir_keeps_scene_endpoints_for_selected_video():
    reservoir = probe.EpisodeReservoir(
        1,
        num_success=1,
        num_failure=1,
        require_pure_teacher=True,
    )
    state_0 = torch.tensor([0.0, 1.0])
    state_1 = torch.tensor([1.0, 2.0])
    state_2 = torch.tensor([2.0, 3.0])
    reservoir.add_step(
        0,
        _row(is_init=True),
        scene_state_before=state_0,
        scene_state_after=state_1,
    )
    trajectory = reservoir.add_step(
        0,
        _row(done=True, command_finished=True),
        scene_state_before=state_1,
        scene_state_after=state_2,
    )

    assert trajectory is not None
    assert len(trajectory.rows) == 2
    assert len(trajectory.scene_states) == 3
    assert torch.equal(trajectory.scene_states[0], state_0)
    assert torch.equal(trajectory.scene_states[-1], state_2)


def test_scene_codec_captures_and_restores_red_reference_keypoints():
    class FakeAsset:
        def __init__(self):
            self.data = SimpleNamespace(
                root_link_pose_w=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            )
            self.root_writes = []
            self.joint_writes = []

        def write_root_link_pose_to_sim(self, value, *, env_ids):
            self.root_writes.append((value.clone(), env_ids.clone()))

        def write_root_com_velocity_to_sim(self, value, *, env_ids):
            pass

        def write_joint_state_to_sim(self, position, velocity, *, env_ids):
            self.joint_writes.append(
                (position.clone(), velocity.clone(), env_ids.clone())
            )

    class FakeMarkers:
        def __init__(self):
            self.calls = []

        def visualize(self, *, translations, marker_indices):
            self.calls.append((translations.clone(), list(marker_indices)))

    robot = FakeAsset()
    markers = FakeMarkers()
    reference_positions = torch.tensor(
        [
            [[0.0, 0.0, 1.2], [0.1, 0.0, 1.0]],
            [[2.0, 0.0, 1.2], [2.1, 0.0, 1.0]],
        ]
    )
    manager = SimpleNamespace(
        ref_body_pos_w=reference_positions,
        vis_markers=markers,
        tracking_keypoint_names=["pelvis", "left_knee_link"],
        object=None,
        object2=None,
        extra_objects=(),
    )
    base = SimpleNamespace(
        command_manager=manager,
        robot=robot,
        num_envs=2,
        device=torch.device("cpu"),
        sim=SimpleNamespace(forward=lambda: None),
        scene=SimpleNamespace(update=lambda dt: None),
    )
    codec = probe._SceneStateCodec(SimpleNamespace(env=base))
    snapshot = codec.snapshot()

    assert snapshot.shape == (2, 15)
    assert torch.equal(snapshot[:, -6:], reference_positions.reshape(2, -1))
    assert codec.layout[-1]["name"] == "reference_keypoints"
    assert codec.layout[-1]["marker_color_rgb"] == [1.0, 0.0, 0.0]

    codec.restore(snapshot[1], env_index=1)
    translations, marker_indices = markers.calls[-1]
    assert torch.equal(translations, reference_positions[1])
    assert marker_indices == [1, 1]


def test_phase_zero_defaults_and_reset_validation():
    args = probe.parse_args(["--checkpoint", "teacher.pt"])
    assert args.random_start is False
    assert args.record_videos is False

    policy = SimpleNamespace(
        env=SimpleNamespace(
            command_manager=SimpleNamespace(
                t=torch.ones(3, dtype=torch.long),
                replay_motion=False,
            )
        )
    )
    probe._validate_phase_zero_reset(policy, random_start=False)
    policy.env.command_manager.t[0] = 2
    with pytest.raises(RuntimeError, match="command_manager.t must equal 1"):
        probe._validate_phase_zero_reset(policy, random_start=False)


def test_synthetic_trajectories_write_only_requested_series(
    tmp_path, monkeypatch
):
    from matplotlib.axes import Axes

    plotted_series = []
    plotted_values = {}
    original_plot = Axes.plot

    def recording_plot(self, *args, **kwargs):
        if kwargs.get("label") is not None:
            label = kwargs["label"]
            plotted_series.append(
                (
                    label,
                    kwargs.get("color"),
                    kwargs.get("linestyle", "-"),
                )
            )
            plotted_values[label] = tuple(float(value) for value in args[1])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", recording_plot)
    success = probe.CapturedTrajectory(
        outcome=probe.OUTCOME_SUCCESS,
        sample_index=1,
        env_index=0,
        episode_index=0,
        pure_teacher=True,
        rows=(
            _row(is_init=True),
            _row(
                environment_step=1,
                done=True,
                command_finished=True,
                environment_success_stat=1.0,
            ),
        ),
    )
    failure = probe.CapturedTrajectory(
        outcome=probe.OUTCOME_FAILURE,
        sample_index=1,
        env_index=1,
        episode_index=0,
        pure_teacher=True,
        rows=(
            _row(is_init=True),
            _row(environment_step=1, done=True, terminated=True),
        ),
    )
    # EpisodeReservoir normally installs this field before capture.
    selected = {
        probe.OUTCOME_SUCCESS: [
            probe.CapturedTrajectory(
                **{
                    **success.__dict__,
                    "rows": tuple(
                        {**row, "episode_step": index}
                        for index, row in enumerate(success.rows)
                    ),
                }
            )
        ],
        probe.OUTCOME_FAILURE: [
            probe.CapturedTrajectory(
                **{
                    **failure.__dict__,
                    "rows": tuple(
                        {**row, "episode_step": index}
                        for index, row in enumerate(failure.rows)
                    ),
                }
            )
        ],
    }
    image_path = tmp_path / "plot.png"
    csv_path = tmp_path / "values.csv"
    probe.plot_trajectories(
        selected,
        image_path,
        gamma=0.99,
        tvkd_lambda=0.25,
    )
    probe.write_csv(selected, csv_path)

    assert set(plotted_series) == {
        (
            r"$\widetilde{r}_t=r_t+\lambda_V[\gamma c_tB_t-V_\tau(s_t)]$",
            "#0072B2",
            "-",
        ),
        (r"environment reward $r_t$", "#D55E00", "-"),
        (r"$\delta_t^T=r_t+\gamma c_tB_t-V_\tau(s_t)$", "#FF0000", "-"),
        (r"$V_\tau(s_{t+1})$", "#009E73", ":"),
        (r"$V_\tau(s_t)$", "#7C3AED", ":"),
    }
    assert plotted_values[
        r"$\widetilde{r}_t=r_t+\lambda_V[\gamma c_tB_t-V_\tau(s_t)]$"
    ] == pytest.approx([1.02225, 1.02225])
    assert plotted_values[
        r"$\delta_t^T=r_t+\gamma c_tB_t-V_\tau(s_t)$"
    ] == pytest.approx([1.089, 1.089])
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image_path.stat().st_size > 10_000
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 5


def _failure_rows_with_diagnostic_onset():
    rows = []
    for step in range(40):
        production_delta = -0.2 if 18 <= step <= 24 else 0.1
        rows.append(
            _row(
                episode_step=step,
                environment_step=step,
                step_count=step,
                production_delta=production_delta,
                done=step == 39,
                terminated=step == 39,
                termination_causes=("test_failure" if step == 39 else ""),
            )
        )
    return tuple(rows)


def test_diagnostic_failure_onset_uses_full_window_and_pre_onset_rows():
    rows = _failure_rows_with_diagnostic_onset()
    onset = probe.detect_diagnostic_failure_onset(rows)

    assert onset is not None
    assert onset.episode_step == 20
    assert onset.confirmation_step == 22
    assert onset.raw_production_delta == pytest.approx(-0.2)
    assert onset.smoothed_production_delta == pytest.approx(-0.08)
    assert onset.precursor_start_step == 10
    assert onset.precursor_end_step == 19
    assert onset.precursor_transition_count == 10

    summary = probe._trajectory_summary(
        probe.CapturedTrajectory(
            outcome=probe.OUTCOME_FAILURE,
            sample_index=1,
            env_index=0,
            episode_index=0,
            pure_teacher=True,
            rows=rows,
        )
    )
    assert summary["diagnostic_failure_onset"]["episode_step"] == 20
    assert summary["diagnostic_failure_onset"]["confirmation_step"] == 22


def test_diagnostic_failure_onset_ignores_burn_in_and_terminal_zone():
    rows = []
    for step in range(40):
        in_excluded_region = step <= 5 or step >= 34
        rows.append(
            _row(
                episode_step=step,
                environment_step=step,
                step_count=step,
                production_delta=(-1.0 if in_excluded_region else 0.1),
                done=step == 39,
                terminated=step == 39,
            )
        )

    assert probe.detect_diagnostic_failure_onset(rows) is None


def test_failure_plot_marks_diagnostic_onset_with_square(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    square_calls = []
    original_scatter = Axes.scatter

    def recording_scatter(self, *args, **kwargs):
        if kwargs.get("label") == r"diagnostic $t_{\mathrm{onset}}$":
            square_calls.append((tuple(args[0]), tuple(args[1]), dict(kwargs)))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", recording_scatter)
    failure = probe.CapturedTrajectory(
        outcome=probe.OUTCOME_FAILURE,
        sample_index=1,
        env_index=0,
        episode_index=0,
        pure_teacher=True,
        rows=_failure_rows_with_diagnostic_onset(),
    )
    image_path = tmp_path / "onset.png"
    probe.plot_trajectories(
        {probe.OUTCOME_SUCCESS: [], probe.OUTCOME_FAILURE: [failure]},
        image_path,
        gamma=0.99,
        tvkd_lambda=0.25,
    )

    assert len(square_calls) == 1
    x_values, y_values, kwargs = square_calls[0]
    assert x_values == (20,)
    assert y_values == pytest.approx((-0.2,))
    assert kwargs["marker"] == "s"
    assert kwargs["facecolor"] == "#FDE047"
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_video_pose_indices_and_timestep_text_preserve_state_mapping():
    assert probe._video_pose_indices(2, 1) == [0, 1, 2]
    assert probe._video_pose_indices(5, 2) == [0, 2, 4, 5]
    assert probe._video_pose_indices(6, 2) == [0, 2, 4, 6]

    initial = probe._video_timestep_text(
        pose_index=0,
        num_transitions=5,
        step_dt=0.02,
        outcome="failure",
    )
    intermediate = probe._video_timestep_text(
        pose_index=2,
        num_transitions=5,
        step_dt=0.02,
        outcome="failure",
    )
    terminal = probe._video_timestep_text(
        pose_index=5,
        num_transitions=5,
        step_dt=0.02,
        outcome="failure",
    )
    assert "state step 0/5" in initial
    assert "time 0.00 s" in initial
    assert "graph transition not started" in initial
    assert "state step 2/5" in intermediate
    assert "time 0.04 s" in intermediate
    assert "after graph transition 1" in intermediate
    assert "state step 5/5" in terminal
    assert "time 0.10 s" in terminal
    assert "terminal pose (pre-reset)" in terminal
    assert "after graph transition 4" in terminal

    with pytest.raises(ValueError, match="pose_index"):
        probe._video_timestep_text(
            pose_index=6,
            num_transitions=5,
            step_dt=0.02,
            outcome="failure",
        )


def test_video_timestep_overlay_preserves_rgb_contract_and_input():
    frame = np.zeros((120, 640, 3), dtype=np.uint8)
    before = frame.copy()
    overlaid = probe._overlay_video_timestep(
        frame,
        pose_index=3,
        num_transitions=5,
        step_dt=0.02,
        outcome="success",
    )

    assert np.array_equal(frame, before)
    assert overlaid.shape == frame.shape
    assert overlaid.dtype == np.uint8
    assert overlaid.flags.c_contiguous
    assert np.any(overlaid != frame)
