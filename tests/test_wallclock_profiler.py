import json

import pytest
import torch

from active_adaptation.utils.wallclock_profiler import (
    WallClockProfileConfig,
    WallClockProfiler,
    instrument_training_policy,
)


class _Replay:
    def __init__(self):
        self.rows = []

    def extend(self, batch):
        self.rows.append(batch)
        return next(iter(batch.values())).shape[0]


class _Policy:
    def __init__(self):
        self.dagger_replay = _Replay()
        self.q_teacher_replay = _Replay()

    def _student_raw_action_proposal(self, observations):
        return observations

    def _teacher_action(self, observations):
        return observations

    def get_frozen_teacher_value(self, observations):
        return observations[:, 0]

    def _batched_frozen_teacher_value(self, observations):
        return self.get_frozen_teacher_value(observations)

    def _critic_update(self, batch):
        return self.get_frozen_teacher_value(batch["critic_observations"])

    def _actor_update(self, batch):
        return batch

    def _record_teacher_phase_match_distances(self, phases, motion_ids):
        return phases, motion_ids

    def _update_failure_phase_histogram(self, batch):
        rows = next(iter(batch.values())).shape[0]
        phases = torch.zeros(rows)
        self._record_teacher_phase_match_distances(
            phases, torch.zeros(rows, dtype=torch.long)
        )
        return rows

    def _prepare_dagger_learning_batch(self, batch):
        return batch

    def train_op(self, batch):
        self._update_failure_phase_histogram(batch)
        return self._prepare_dagger_learning_batch(batch)


def test_phase_window_profiles_methods_and_restores_them(tmp_path):
    config = WallClockProfileConfig(
        enabled=True,
        label="D",
        phase="main_dagger",
        start_rollout=1,
        num_rollouts=2,
        cuda_events=False,
    )
    profiler = WallClockProfiler(
        config, device="cpu", output_dir=tmp_path, metadata={"seed": 7}
    )
    policy = _Policy()
    original_student = policy._student_raw_action_proposal
    instrument_training_policy(policy, profiler)

    assert not profiler.begin_rollout(
        4, phase="teacher_prefill", phase_rollout=4
    )
    assert not profiler.begin_rollout(5, phase="main_dagger", phase_rollout=0)
    assert profiler.begin_rollout(6, phase="main_dagger", phase_rollout=1)
    assert policy._student_raw_action_proposal != original_student

    observations = torch.ones(5, 3)
    policy._student_raw_action_proposal(observations)
    policy._teacher_action(observations)
    policy._batched_frozen_teacher_value(observations)
    policy._critic_update({"critic_observations": observations})
    policy.train_op({"critic_observations": observations})
    policy.dagger_replay.extend({"x": torch.ones(5, 2)})
    profiler.increment("environment_states", 10)
    profiler.end_rollout(6, phase_rollout=1)

    assert profiler.begin_rollout(7, phase="main_dagger", phase_rollout=2)
    policy._critic_update({"critic_observations": observations[:2]})
    profiler.increment("environment_states", 10)
    profiler.end_rollout(7, phase_rollout=2)

    assert profiler.finished
    assert policy._student_raw_action_proposal == original_student
    summary = json.loads((tmp_path / "wallclock_profile.json").read_text())
    assert summary["label"] == "D"
    assert summary["window"]["phase"] == "main_dagger"
    assert summary["window"]["completed"] is True
    assert summary["counters"]["profiled_rollouts"] == 2
    assert summary["counters"]["teacher_value_target_calls"] == 2
    assert summary["counters"]["teacher_value_target_states"] == 7
    assert summary["counters"]["frozen_teacher_rollout_grid_states"] == 5
    assert summary["counters"]["replay_insert_rows"] == 5
    assert summary["counters"]["training_operation_calls"] == 1
    assert summary["counters"]["failure_phase_bookkeeping_calls"] == 1
    assert summary["counters"]["teacher_phase_match_diagnostic_calls"] == 1
    assert summary["counters"]["replay_batch_preparation_rows"] == 5
    assert summary["blocks"]["c51_q_forward_backward"]["calls"] == 2
    assert summary["blocks"]["training_operation"]["calls"] == 1
    assert summary["blocks"]["failure_phase_bookkeeping"]["calls"] == 1
    assert summary["blocks"]["teacher_phase_match_diagnostics"]["calls"] == 1
    # Grid inference must not be misclassified as target-time Teacher value.
    assert summary["blocks"]["teacher_value_inside_tvkd_target"]["calls"] == 2

    with profiler.external_cpu_block("checkpoint_load_final"):
        pass
    refreshed = json.loads((tmp_path / "wallclock_profile.json").read_text())
    assert refreshed["blocks"]["checkpoint_load_final"]["calls"] == 1


def test_disabled_profiler_does_not_install_wrappers_or_write(tmp_path):
    profiler = WallClockProfiler(
        WallClockProfileConfig(enabled=False),
        device="cpu",
        output_dir=tmp_path,
    )
    policy = _Policy()
    original = policy._teacher_action
    instrument_training_policy(policy, profiler)
    assert not profiler.begin_rollout(0)
    assert policy._teacher_action == original
    policy._teacher_action(torch.ones(1, 2))
    assert profiler.finish()["blocks"] == {}
    assert not (tmp_path / "wallclock_profile.json").exists()


def test_optional_torch_profiler_counts_scalar_extractions(tmp_path):
    profiler = WallClockProfiler(
        WallClockProfileConfig(
            enabled=True,
            num_rollouts=1,
            cuda_events=False,
            torch_profiler_sync_audit=True,
        ),
        device="cpu",
        output_dir=tmp_path,
    )
    assert profiler.begin_rollout(0)
    with profiler.block("scalar_probe"):
        torch.ones(()).item()
    profiler.end_rollout(0)
    summary = profiler.summary
    assert summary is not None
    audit = summary["synchronization_audit"]
    assert audit["enabled"] is True
    assert audit["implicit_host_scalar_extractions"] >= 1
    assert summary["window"]["start_utc"] is not None
    assert summary["window"]["end_utc"] is not None


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"phase": "unknown"}, "phase"),
        ({"start_rollout": -1}, "start_rollout"),
        ({"num_rollouts": 0}, "num_rollouts"),
        ({"output_filename": "nested/profile.json"}, "output_filename"),
    ],
)
def test_profile_config_rejects_invalid_windows(mapping, message):
    with pytest.raises(ValueError, match=message):
        WallClockProfileConfig.from_mapping(mapping)
