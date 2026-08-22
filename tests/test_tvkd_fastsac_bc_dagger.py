from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from active_adaptation.learning.ppo.fastsac_bc_dagger import (
    DistributionalFastSACTeacherBC,
)
from active_adaptation.learning.ppo.common import Actor
from active_adaptation.learning.ppo.fastsac_vel import _BCDaggerSACAdapter
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_VALID_KEY,
)
from active_adaptation.learning.ppo.ppo_vel import DEPTH_KEY, PPOVEL, PRIV_PRED_KEY
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC,
    FAILURE_PHASE_STUDENT_SOURCE_KEY,
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    PERCEPTION_DEPTH_U8_KEY,
    PERCEPTION_IS_INIT_KEY,
    PERCEPTION_POLICY_RAW_KEY,
    PERCEPTION_VEL_COMMAND_RAW_KEY,
    REFERENCE_PHASE_KEY,
    REPLAY_COMMAND_FINISHED_KEY,
    REPLAY_PERCEPTION_EMA_GENERATION_KEY,
    REPLAY_MOTION_ID_KEY,
    REPLAY_SAMPLE_PROVENANCE_KEY,
    REPLAY_SOURCE_ORDER,
    REPLAY_TERMINATED_KEY,
    REPLAY_TIME_LIMIT_KEY,
    STUDENT_REPLAY_EPISODE_ID_KEY,
    STUDENT_REPLAY_EPISODE_STEP_KEY,
    STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY,
    _StudentPerceptionDriftEpisode,
    _TD3DeviceReplay,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _project_c51_probabilities,
    allocate_source_counts,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    ACTOR_BACKEND as TVKD_ACTOR_BACKEND,
    CHECKPOINT_VERSION as TVKD_CHECKPOINT_VERSION,
    LEGACY_ADAPTIVE_BC_CONFIG_FIELDS,
    LEGACY_CHECKPOINT_VERSION as TVKD_LEGACY_CHECKPOINT_VERSION,
    LEGACY_TRAINING_ALGORITHM as TVKD_LEGACY_TRAINING_ALGORITHM,
    PREVIOUS_CHECKPOINT_VERSION as TVKD_PREVIOUS_CHECKPOINT_VERSION,
    PREVIOUS_TRAINING_ALGORITHM as TVKD_PREVIOUS_TRAINING_ALGORITHM,
    REPLAY_TEACHER_V_CURRENT_KEY,
    REPLAY_TEACHER_V_NEXT_KEY,
    V3_CHECKPOINT_VERSION as TVKD_V3_CHECKPOINT_VERSION,
    V3_TRAINING_ALGORITHM as TVKD_V3_TRAINING_ALGORITHM,
    V4_CHECKPOINT_VERSION as TVKD_V4_CHECKPOINT_VERSION,
    V4_TRAINING_ALGORITHM as TVKD_V4_TRAINING_ALGORITHM,
    V5_CHECKPOINT_VERSION as TVKD_V5_CHECKPOINT_VERSION,
    V5_TRAINING_ALGORITHM as TVKD_V5_TRAINING_ALGORITHM,
    SOURCE_FAILURE_TEACHER,
    SOURCE_STUDENT,
    SOURCE_UNIFORM_TEACHER,
    FrozenTeacherValueWrapper,
    TeacherValueBottleneckDetector,
    TRAINING_ALGORITHM as TVKD_TRAINING_ALGORITHM,
    TVKDDistributionalFastSACTeacherBC,
    TVKDDistributionalFastSACTeacherBCConfig,
    compute_continuation_coefficient,
    compute_teacher_value_continuation,
    compute_teacher_value_terms,
    continuation_bootstrap_mask,
    replay_truncation_mask,
)
from scripts.TVKD_fasSAC_bc_dagger import (
    _apply_tvkd_cli_replay_mix_overrides,
    _prepare_tvkd_checkpoint,
    validate_tvkd_fastsac_bc_dagger_config,
)
from scripts.fastSAC_bc_dagger import (
    PPOVEL_TRAIN_PHASE_PARTIAL_PERCEPTION_MODULES,
)
from scripts.helpers import (
    _fill_replayless_inference_algo_defaults,
    _load_policy_checkpoint,
)

tvkd_module = importlib.import_module(
    "active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger"
)
fastsac_module = importlib.import_module(
    "active_adaptation.learning.ppo.fastsac_bc_dagger"
)
tvkd_entry = importlib.import_module("scripts.TVKD_fasSAC_bc_dagger")


class _DeviceMoveCounter:
    """Tensor-like test seam recording whole-grid device materialization."""

    def __init__(self, value: torch.Tensor):
        self.value = value
        self.calls: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append((args, kwargs))
        return self.value.to(*args, **kwargs)


class _KeyedTeacherValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, td: TensorDict):
        # The input flat layout in the test is [z, a].  This formula detects a
        # wrapper which accidentally forwards that flat tensor without keys.
        scalar = self.gain * (10.0 * td["a"][:, 0] + td["z"][:, 0])
        td["state_value"] = torch.stack((scalar, scalar + 1.0), dim=-1)
        return td


class _AffineValueNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor([2.0, 3.0]))

    def denormalize(self, value):
        return value + self.offset


class _FixedDist:
    def __init__(self, batch_size: int, action: float = 0.0, log_prob: float = 0.0):
        self.loc = torch.zeros(batch_size, 1)
        self.mean = torch.zeros(batch_size, 1)
        self.action = float(action)
        self.log_prob = float(log_prob)

    def rsample_with_log_prob(self, generator=None):
        if generator is not None:
            torch.rand((), generator=generator)
        return (
            self.loc.new_full(self.loc.shape, self.action),
            self.loc.new_full((self.loc.shape[0],), self.log_prob),
        )


class _TableTwin(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.register_buffer("first", first.log())
        self.register_buffer("second", second.log())
        self.register_buffer("support", torch.tensor([-10.0, 0.0, 10.0]))

    def forward(self, observations, actions):
        del actions
        count = observations.shape[0]
        return torch.stack((self.first[:count], self.second[:count]), dim=0)


def _detector(**overrides):
    kwargs = {
        "threshold": 0.05,
        "smoothing_window": 5,
        "min_consecutive": 3,
        "terminal_exclusion_steps": 5,
    }
    kwargs.update(overrides)
    return TeacherValueBottleneckDetector(**kwargs)


def _empty_bottleneck_checkpoint_state(
    detector_state: dict | None = None,
) -> dict:
    return {
        "location_semantics": tvkd_module.BOTTLENECK_LOCATION_SEMANTICS,
        "detector": {} if detector_state is None else detector_state,
        "unsuccessful_episode_count": 0,
        "episodes_with_student_candidates": 0,
        "no_value_bottleneck_count": 0,
        "value_argmin_ablation_count": 0,
        "failed_student_episode_count": 0,
        "student_candidate_count": 0,
        "detected_count": 0,
        "fallback_count": 0,
        "no_candidate_count": 0,
        "selected_count": 0,
        "teacher_sequences_inserted": 0,
        "teacher_transitions_inserted": 0,
        "phase_match_distance_count": 0,
        "next_student_episode_id": 0,
        "student_focus_rows_marked": 0,
        "student_focus_rows_missing": 0,
        "student_focus_sampled_rows": 0,
        "student_focus_uniform_fallback_rows": 0,
        "selected_step_sum": 0.0,
        "selected_phase_sum": 0.0,
        "score_sum": 0.0,
        "score_max": 0.0,
        "raw_td_residual_sum": 0.0,
        "smoothed_td_residual_sum": 0.0,
        "phase_match_distance_sum": 0.0,
        "last_metadata": {},
        "last_value_argmin_metadata": {},
    }


def _legacy_bottleneck_checkpoint_state(
    detector_state: dict | None = None,
) -> dict:
    state = _empty_bottleneck_checkpoint_state(detector_state)
    state["normalized_td_residual_sum"] = state.pop("smoothed_td_residual_sum")
    return state


def _checkpoint_replay_mix(cfg) -> dict[str, dict[str, float]]:
    return {
        purpose: {
            source: float(getattr(cfg, f"{purpose}_{source}_fraction"))
            for source in REPLAY_SOURCE_ORDER
        }
        for purpose in ("q", "actor", "perception")
    }


def _verified_histogram_state(
    num_bins: int,
    *,
    histogram: torch.Tensor | None = None,
    motion_id: int = 0,
) -> dict:
    if histogram is None:
        histogram = torch.zeros(num_bins, dtype=torch.float64)
    histogram = histogram.detach().to(dtype=torch.float64, device="cpu").clone()
    anchor_count = int(histogram.sum().item())
    return {
        "semantics": tvkd_module.VERIFIED_HISTOGRAM_SEMANTICS,
        "histogram": histogram,
        "episode_count": 0 if anchor_count == 0 else 1,
        "anchor_count": anchor_count,
        "uniform_fallback_rows": 0,
        "focused_rows": 0,
        "motion_histograms": (
            {} if anchor_count == 0 else {motion_id: histogram.clone()}
        ),
    }


def _strict_v5_backend_config(cfg) -> dict:
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = cfg
    return copy.deepcopy(policy._checkpoint_config())


def _strict_v5_policy_metadata(
    cfg,
    *,
    vecnorm_fingerprint: str,
    histogram_state: dict,
) -> dict:
    return {
        "critic_learning_semantics": tvkd_module.CRITIC_LEARNING_SEMANTICS,
        "actor_learning_semantics": tvkd_module.ACTOR_LEARNING_SEMANTICS,
        "replay_mix_state": _checkpoint_replay_mix(cfg),
        "perception_replay_mode": str(cfg.perception_replay_mode),
        "perception_training_semantics": (
            tvkd_module.ONLINE_STUDENT_ROLLOUT_PERCEPTION_SEMANTICS
        ),
        "actor_replay_observation_semantics": (
            tvkd_module.COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
        ),
        "teacher_episode_sidecar_semantics": (
            tvkd_module.TEACHER_EPISODE_SIDECAR_SEMANTICS
        ),
        "bottleneck_location_semantics": (tvkd_module.BOTTLENECK_LOCATION_SEMANTICS),
        "bottleneck_fallback_mode": str(cfg.bottleneck_fallback_mode),
        "teacher_value_return_semantics": str(cfg.teacher_value_return_semantics),
        "teacher_value_boundary_semantics": str(cfg.teacher_value_boundary_semantics),
        "teacher_value_gamma": float(cfg.gamma),
        "teacher_value_reward_group_fingerprint": str(
            cfg.teacher_value_reward_group_fingerprint
        ),
        "teacher_value_vecnorm_fingerprint": vecnorm_fingerprint,
        "replay_task_fingerprint": str(cfg.replay_task_fingerprint),
        "verified_teacher_value_histogram_state": copy.deepcopy(histogram_state),
        "failure_phase_curriculum_state": copy.deepcopy(histogram_state),
        "fresh_ring_resume_semantics": tvkd_module.FRESH_RING_RESUME_SEMANTICS,
        "replay_resume_semantics": tvkd_module.REPLAY_RESUME_SEMANTICS,
        "q_backend_config": {
            "target_semantics": tvkd_module.CRITIC_LEARNING_SEMANTICS,
            "failure_phase_replay_semantics": (
                tvkd_module.VERIFIED_HISTOGRAM_SEMANTICS
            ),
            "bottleneck_location_semantics": (
                tvkd_module.BOTTLENECK_LOCATION_SEMANTICS
            ),
            "bottleneck_fallback_mode": str(cfg.bottleneck_fallback_mode),
        },
    }


def _legacy_replay_backend(
    lambda_bc: float,
    *,
    student_focus: float | None,
    alpha_cadence: str = "critic",
) -> dict:
    backend = {
        "value_norm": False,
        "lambda_bc": float(lambda_bc),
        "q_teacher_replay_ratio": 0.5,
        "teacher_actor_replay_fraction": 0.5,
        "teacher_perception_replay_fraction": 0.5,
        "failure_phase_teacher_fraction": 0.3,
        "sac_alpha_update_cadence": alpha_cadence,
    }
    if student_focus is not None:
        backend["failure_phase_student_fraction"] = float(student_focus)
    return backend


def _legacy_saved_config(saved_cfg, *, include_student_focus: bool):
    legacy_cfg = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
    v4_only_fields = {
        f"{purpose}_{source}_fraction"
        for purpose in ("q", "actor", "perception")
        for source in REPLAY_SOURCE_ORDER
    }
    v4_only_fields.update(
        {
            "perception_replay_mode",
            "bottleneck_fallback_mode",
            "bottleneck_include_unsuccessful_timeouts",
            "max_teacher_phase_match_distance",
            "perception_replay_batch_size",
            "teacher_value_return_semantics",
            "teacher_value_boundary_semantics",
            "teacher_value_reward_group_fingerprint",
            "replay_task_fingerprint",
        }
    )
    for name in v4_only_fields:
        if name in legacy_cfg.algo:
            del legacy_cfg.algo[name]
    if not include_student_focus:
        del legacy_cfg.algo.failure_phase_student_fraction
    legacy_cfg.algo.sac_alpha_update_cadence = "critic"
    return legacy_cfg


def _tvkd_replay_rows(count: int, offset: int) -> dict[str, torch.Tensor]:
    row = torch.arange(count, dtype=torch.float32) + float(offset)
    time = torch.arange(2, dtype=torch.float32)
    return {
        "critic_observations": torch.stack((row, row + 0.5), dim=-1),
        "actions": row[:, None] + 1.0,
        "rewards": row + 2.0,
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.full((count,), 0.95),
        "next_critic_observations": torch.stack((row + 4.0, row + 4.5), dim=-1),
        REFERENCE_PHASE_KEY: torch.linspace(0.0, 1.0, count),
        PERCEPTION_DEPTH_U8_KEY: torch.zeros(count, 2, 1, 1, 1, dtype=torch.uint8),
        PERCEPTION_POLICY_RAW_KEY: row[:, None, None] + time[None, :, None],
        PERCEPTION_VEL_COMMAND_RAW_KEY: (
            row[:, None, None] + time[None, :, None] + 0.25
        ),
        PERCEPTION_IS_INIT_KEY: torch.zeros(count, 2, dtype=torch.bool),
    }


def _student_split_sampling_policy(
    *,
    focused_rows: int,
    seed: int = 941,
) -> tuple[TVKDDistributionalFastSACTeacherBC, set[float], set[float]]:
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        q_batch_size=20,
        q_teacher_replay_ratio=0.5,
        dagger_batch_size=20,
        teacher_actor_replay_fraction=0.5,
        failure_phase_teacher_fraction=0.3,
        failure_phase_student_fraction=0.3,
        failure_phase_num_bins=16,
        policy_delay=1,
    )
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(seed)
    policy.critic_update_count = 0
    policy.q_teacher_replay = _TD3DeviceReplay(64, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(64, "cpu")

    teacher = _tvkd_replay_rows(30, 100)
    teacher[REFERENCE_PHASE_KEY] = torch.linspace(0.0, 1.0, 30)
    policy.q_teacher_replay.extend(teacher)

    student_count = 40
    if not 0 <= focused_rows <= student_count:
        raise ValueError("focused_rows is invalid")
    student = _tvkd_replay_rows(student_count, 1_000)
    student[DAGGER_IS_STUDENT_ACTION_KEY] = torch.ones(student_count, dtype=torch.bool)
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(student_count, dtype=torch.float32)[:, None] + 2_000.0
    )
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(
        student_count, dtype=torch.bool
    )
    focused = torch.zeros(student_count, dtype=torch.bool)
    focus_start = student_count - focused_rows
    focused[focus_start:] = True
    student[FAILURE_PHASE_STUDENT_SOURCE_KEY] = focused
    student[STUDENT_REPLAY_EPISODE_ID_KEY] = torch.arange(
        student_count, dtype=torch.long
    )
    student[STUDENT_REPLAY_EPISODE_STEP_KEY] = torch.arange(
        student_count, dtype=torch.long
    )
    policy.dagger_replay.extend(student)

    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )
    policy._failure_phase_histogram = torch.zeros(16, dtype=torch.float64)
    policy._failure_phase_histogram[12] = 1.0
    policy._failure_phase_anchor_count = 1
    policy._failure_histogram_device_cache = {}
    policy._teacher_phase_device_cache = {}
    policy._teacher_phase_index_ready = False
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._build_teacher_phase_index()

    uniform_values = {float(1_001 + index) for index in range(student_count)}
    focused_values = {
        float(1_001 + index) for index in range(focus_start, student_count)
    }
    return policy, uniform_values, focused_values


def test_detector_selects_synthetic_bottleneck_onset_not_terminal_crash():
    detector = _detector()
    residual = torch.full((40,), 0.1)
    residual[18:25] = -0.2
    result = detector.detect(
        residual,
        torch.full((40,), SOURCE_STUDENT),
        torch.arange(40, dtype=torch.float32) / 40.0,
        torch.tensor([False] * 39 + [True]),
        torch.zeros(40, dtype=torch.bool),
        torch.tensor([False] * 6 + [True] * 34),
    )

    assert result is not None
    assert result.index == 20
    assert result.confirmation_index == 22
    assert result.index != 39
    assert result.threshold_detected is True
    assert result.used_fallback is False
    assert result.phase == pytest.approx(0.5)
    assert result.raw_teacher_td_residual == pytest.approx(-0.2)
    assert result.smoothed_teacher_td_residual == pytest.approx(-0.08)
    assert result.score == pytest.approx(0.08)
    assert detector.last_diagnostics["student_candidate_count"] == 24.0


def test_detector_has_no_teacher_only_candidate_and_is_stateless():
    detector = _detector()
    before = detector.state_dict()
    result = detector.detect(
        torch.tensor([float("nan"), -100.0, 4.0]),
        torch.tensor(
            [SOURCE_UNIFORM_TEACHER, SOURCE_FAILURE_TEACHER, SOURCE_UNIFORM_TEACHER]
        ),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([False, False, True]),
        torch.zeros(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
    )

    assert result is None
    assert detector.state_dict() == before
    assert detector.last_diagnostics["student_transition_count"] == 0.0
    assert detector.last_diagnostics["no_candidate"] is True


def test_detector_teacher_gap_breaks_smoothing_and_consecutive_run():
    detector = _detector(
        smoothing_window=2, min_consecutive=2, terminal_exclusion_steps=0
    )
    result = detector.detect(
        torch.full((15,), -0.2),
        torch.tensor(
            [
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_UNIFORM_TEACHER,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
                SOURCE_STUDENT,
            ]
        ),
        torch.arange(15, dtype=torch.float32) / 15.0,
        torch.tensor([False] * 14 + [True]),
        torch.zeros(15, dtype=torch.bool),
        torch.tensor([False] * 6 + [True] * 9),
    )

    assert result is not None
    # Replay-invalid rows 0..5 and the Teacher row 8 both reset the full MA.
    assert result.index == 10
    assert result.threshold_detected is True


def test_detector_argmin_fallback_excludes_timeout_and_round_trips_empty_state():
    detector = _detector(
        threshold=10.0,
        smoothing_window=1,
        min_consecutive=2,
        terminal_exclusion_steps=0,
    )
    result = detector.detect(
        torch.tensor([0.1, -0.2, -2.0]),
        torch.full((3,), SOURCE_STUDENT),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor([False, False, True]),
        torch.ones(3, dtype=torch.bool),
        fallback_mode="value_argmin",
    )

    assert result is not None
    assert result.index == 1
    assert result.used_fallback is True
    assert result.threshold_detected is False
    assert detector.last_diagnostics["student_candidate_count"] == 2.0

    restored = _detector(
        threshold=10.0,
        smoothing_window=1,
        min_consecutive=2,
        terminal_exclusion_steps=0,
    )
    restored.load_state_dict(detector.state_dict())
    assert restored.state_dict() == detector.state_dict()

    only_terminal = detector.detect(
        torch.tensor([-20.0]),
        torch.tensor([SOURCE_STUDENT]),
        torch.tensor([0.9]),
        torch.tensor([True]),
        torch.tensor([False]),
        torch.tensor([True]),
        fallback_mode="value_argmin",
    )
    assert only_terminal is None
    assert detector.last_diagnostics["no_candidate"] is True
    assert detector.last_diagnostics["used_fallback"] is False


def test_detector_uses_strict_raw_threshold_and_excludes_terminal_tail():
    detector = _detector(
        smoothing_window=1, min_consecutive=1, terminal_exclusion_steps=5
    )
    residual = torch.full((15,), 0.1)
    residual[6] = -0.05  # Equality must not trigger the strict inequality.
    residual[7] = -0.051
    result = detector.detect(
        residual,
        torch.full((15,), SOURCE_STUDENT),
        torch.arange(15, dtype=torch.float32) / 15.0,
        torch.tensor([False] * 14 + [True]),
        torch.zeros(15, dtype=torch.bool),
        torch.tensor([False] * 6 + [True] * 9),
    )
    assert result is not None
    assert result.index == 7

    # tau=14 and E=5 means only t<9 is eligible; tau-5 through tau are out.
    residual[7] = 0.1
    residual[9] = -100.0
    assert (
        detector.detect(
            residual,
            torch.full((15,), SOURCE_STUDENT),
            torch.arange(15, dtype=torch.float32) / 15.0,
            torch.tensor([False] * 14 + [True]),
            torch.zeros(15, dtype=torch.bool),
            torch.tensor([False] * 6 + [True] * 9),
        )
        is None
    )


def test_detector_rejects_k_run_whose_third_ma5_endpoint_is_in_terminal_tail():
    detector = _detector()
    residual = torch.full((30,), 0.1)
    residual[22:25] = -0.7
    result = detector.detect(
        residual,
        torch.full((30,), SOURCE_STUDENT),
        torch.arange(30, dtype=torch.float32) / 30.0,
        torch.tensor([False] * 29 + [True]),
        torch.zeros(30, dtype=torch.bool),
        torch.tensor([False] * 6 + [True] * 24),
    )
    # tau=29, E=5 => endpoints must satisfy t<24. Only m22 and m23
    # qualify; the required third endpoint m24 is deliberately excluded.
    assert result is None


def test_interior_replay_invalid_row_restarts_full_window_and_k_run():
    detector = _detector(
        smoothing_window=3, min_consecutive=3, terminal_exclusion_steps=0
    )
    replay_valid = torch.tensor([False] * 6 + [True] * 14)
    replay_valid[10] = False
    result = detector.detect(
        torch.tensor([0.1] * 6 + [-0.2] * 11 + [0.1, 0.1, 0.1]),
        torch.full((20,), SOURCE_STUDENT),
        torch.arange(20, dtype=torch.float32) / 20.0,
        torch.tensor([False] * 19 + [True]),
        torch.zeros(20, dtype=torch.bool),
        replay_valid,
    )
    assert result is not None
    assert result.index == 13
    assert result.confirmation_index == 15


def test_frozen_teacher_value_restores_keys_denormalizes_and_sums_groups():
    actor = nn.Linear(1, 1)
    value = _KeyedTeacherValue()
    normalizer = _AffineValueNorm()
    wrapper = FrozenTeacherValueWrapper(
        actor,
        value,
        normalizer,
        critic_keys=("z", "a"),
        critic_widths=(1, 1),
    )

    # Flat rows are [z, a]. Values by group before denormalization are
    # [10*a+z, 10*a+z+1], then offsets [2,3] are added and groups are summed.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = wrapper.get_frozen_teacher_value(
            torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        )
    assert torch.equal(result, torch.tensor([70.0, 156.0]))
    assert result.shape == (2,)
    assert result.dtype == torch.float32
    assert result.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in actor.parameters())
    assert all(parameter.requires_grad is False for parameter in value.parameters())
    assert actor.training is False
    assert value.training is False
    assert normalizer.training is False

    student = nn.Parameter(torch.tensor(2.0))
    (result.sum() + student.square()).backward()
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in value.parameters())


def test_teacher_value_terms_clip_potential_before_difference_and_keep_raw_td():
    def value(obs):
        return obs[:, 0]

    current = torch.tensor([[20.0], [-4.0], [2.0]])
    next_observation = torch.tensor([[-20.0], [8.0], [9.0]])
    reward = torch.tensor([1.0, 2.0, 3.0])
    discount = torch.tensor([0.9, 0.9, 0.5])
    terms = compute_teacher_value_terms(
        value,
        current,
        next_observation,
        reward,
        continuation=discount * torch.tensor([1.0, 0.0, 1.0]),
        gamma=1.0,
        tvkd_lambda=0.25,
        potential_clip=5.0,
    )

    expected_potential = torch.tensor([0.9 * -5.0 - 5.0, 4.0, 0.5 * 5.0 - 2.0])
    expected_td = (
        reward
        + discount * torch.tensor([1.0, 0.0, 1.0]) * next_observation[:, 0]
        - current[:, 0]
    )
    assert torch.allclose(terms.potential_delta, expected_potential)
    assert torch.allclose(terms.shaped_reward, reward + 0.25 * expected_potential)
    # TD residual uses raw reward and the *unclipped* frozen value.
    assert torch.allclose(terms.teacher_td_residual, expected_td)


def test_unclipped_teacher_terms_reuse_one_identical_continuation(monkeypatch):
    original = tvkd_module.compute_teacher_value_continuation
    calls = []

    def counted(**kwargs):
        result = original(**kwargs)
        calls.append(result.clone())
        return result

    monkeypatch.setattr(tvkd_module, "compute_teacher_value_continuation", counted)
    common = {
        "get_teacher_value": lambda observation: observation[:, 0],
        "teacher_critic_obs": torch.tensor([[2.0], [3.0]]),
        "next_teacher_critic_obs": torch.tensor([[4.0], [5.0]]),
        "raw_reward": torch.tensor([1.0, 2.0]),
        "continuation": torch.tensor([1.0, 0.0]),
        "gamma": 0.99,
        "tvkd_lambda": 0.25,
    }

    unclipped = compute_teacher_value_terms(**common, potential_clip=None)
    assert len(calls) == 1
    assert torch.equal(unclipped.teacher_td_residual, torch.tensor([2.96, -1.0]))

    compute_teacher_value_terms(**common, potential_clip=3.0)
    # Clipped shaping and raw bottleneck residuals intentionally differ.
    assert len(calls) == 3


def test_tvkd_lambda_zero_keeps_raw_reward_exactly():
    terms = compute_teacher_value_terms(
        lambda observation: observation[:, 0],
        torch.tensor([[2.0], [3.0]]),
        torch.tensor([[8.0], [9.0]]),
        torch.tensor([1.5, -2.0]),
        continuation=torch.tensor([1.0, 0.0]),
        gamma=0.99,
        tvkd_lambda=0.0,
    )
    assert torch.equal(terms.shaped_reward, torch.tensor([1.5, -2.0]))


def test_bottleneck_selection_mode_picks_first_last_or_deepest_onset():
    """A recovery curriculum wants the latest/deepest onset, not the earliest.

    The residual dips three separate times.  ``first`` keeps the historical
    earliest-crossing rule, ``last`` takes the dip closest to the failure, and
    ``deepest`` takes the lowest one wherever it sits.
    """
    # Windows are 1-wide so the smoothed series equals the residual series.
    residual = torch.tensor(
        [
            -0.30, -0.30, -0.30,   # onset 0 (early, shallow)
            0.50,
            -0.90, -0.90, -0.90,   # onset 4 (deepest)
            0.50,
            -0.40, -0.40, -0.40,   # onset 8 (latest)
            0.50, 0.50, 0.50, 0.50, 0.50,
        ]
    )
    size = residual.numel()
    common = dict(
        source_id=torch.full((size,), SOURCE_STUDENT, dtype=torch.long),
        reference_phase=torch.linspace(0.0, 0.9, size),
        true_terminal=torch.zeros(size, dtype=torch.bool),
        timeout=torch.zeros(size, dtype=torch.bool),
        replay_valid=torch.ones(size, dtype=torch.bool),
    )
    common["true_terminal"][-1] = True

    expected = {"first": 0, "last": 8, "deepest": 4}
    for mode, index in expected.items():
        detector = _detector(
            threshold=0.05,
            smoothing_window=1,
            min_consecutive=3,
            terminal_exclusion_steps=1,
            selection_mode=mode,
        )
        result = detector.detect(residual, **common)
        assert result is not None, mode
        assert result.threshold_detected is True
        assert result.index == index, mode
        assert result.confirmation_index == index + 2

    with pytest.raises(ValueError, match="bottleneck_selection_mode"):
        _detector(selection_mode="earliest")


def test_bottleneck_selection_mode_deepest_prefers_the_later_tie():
    """Equally deep onsets resolve toward the failure, not away from it."""
    residual = torch.tensor(
        [-0.60, -0.60, -0.60, 0.50, -0.60, -0.60, -0.60, 0.50, 0.50, 0.50]
    )
    size = residual.numel()
    true_terminal = torch.zeros(size, dtype=torch.bool)
    true_terminal[-1] = True
    detector = _detector(
        threshold=0.05,
        smoothing_window=1,
        min_consecutive=3,
        terminal_exclusion_steps=1,
        selection_mode="deepest",
    )
    result = detector.detect(
        residual,
        source_id=torch.full((size,), SOURCE_STUDENT, dtype=torch.long),
        reference_phase=torch.linspace(0.0, 0.9, size),
        true_terminal=true_terminal,
        timeout=torch.zeros(size, dtype=torch.bool),
        replay_valid=torch.ones(size, dtype=torch.bool),
    )
    assert result is not None
    assert result.index == 4


def test_student_collection_cache_ema_age_metrics_exclude_teacher_rows():
    """Only collection-time Student Actor features have representation age."""
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy._perception_ema_generation = 11

    metrics = policy._student_replay_ema_age_metrics(
        {
            REPLAY_PERCEPTION_EMA_GENERATION_KEY: torch.tensor([8, 2, 10, 11]),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([True, False, True, True]),
        }
    )
    # Teacher row generation=2 is intentionally ignored: its Actor cache is
    # rebuilt at the current EMA rather than read from collection-time replay.
    assert metrics["available"] == 1.0
    assert metrics["student_rows"] == 3.0
    assert metrics["mean"] == pytest.approx(4.0 / 3.0)
    assert metrics["max"] == 3.0
    assert metrics["stale_fraction"] == pytest.approx(2.0 / 3.0)

    assert policy._student_replay_ema_age_metrics({}) == {
        "available": 0.0,
        "student_rows": 0.0,
        "mean": 0.0,
        "p95": 0.0,
        "max": 0.0,
        "stale_fraction": 0.0,
    }
    with pytest.raises(RuntimeError, match="future EMA generation"):
        policy._student_replay_ema_age_metrics(
            {
                REPLAY_PERCEPTION_EMA_GENERATION_KEY: torch.tensor([12]),
                DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([True]),
            }
        )


def test_student_drift_probe_commits_only_reset_complete_episode():
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        perception_replay_mode="online_student_rollout",
        perception_staleness_probe_num_envs=2,
        perception_staleness_probe_max_episodes=4,
    )
    policy._q_actor_dim = 3
    policy._perception_ema_generation = 7
    policy._student_perception_drift_pending = None
    policy._student_perception_drift_episodes = []
    policy._student_perception_drift_completed = 0
    policy._student_perception_drift_discarded_incomplete = 0

    def raw_values(_self, td):
        return {
            DEPTH_KEY: td["raw_depth"],
            PERCEPTION_POLICY_RAW_KEY: td["raw_policy"],
            PERCEPTION_VEL_COMMAND_RAW_KEY: td["raw_vel"],
            PERCEPTION_IS_INIT_KEY: td["is_init"],
        }

    policy._student_perception_drift_raw_values = MethodType(raw_values, policy)
    rollout = TensorDict(
        {
            "raw_depth": torch.arange(8).reshape(2, 4, 1).to(torch.uint8),
            "raw_policy": torch.arange(8).reshape(2, 4, 1).float(),
            "raw_vel": torch.ones(2, 4, 1),
            "is_init": torch.tensor(
                [[[True], [False], [False], [False]], [[False]] * 4]
            ),
            "next": TensorDict(
                {
                    "done": torch.tensor(
                        [[[False], [False], [False], [True]], [[False]] * 4]
                    )
                },
                batch_size=(2, 4),
            ),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.ones(2, 4, 1, dtype=torch.bool),
            STUDENT_COLLECTION_ACTOR_OBSERVATIONS_KEY: torch.arange(24)
            .reshape(2, 4, 3)
            .float(),
        },
        batch_size=(2, 4),
    )
    policy._capture_student_perception_drift_rollout(rollout)

    assert policy._student_perception_drift_completed == 1
    assert len(policy._student_perception_drift_episodes) == 1
    episode = policy._student_perception_drift_episodes[0]
    assert episode.collection_actor.shape == (4, 3)
    assert episode.collection_ema_generation.tolist() == [7, 7, 7, 7]
    assert episode.eligible_student_rows.tolist() == [False, True, True, True]
    assert policy._student_perception_drift_pending[1] is None


def test_reset_exact_student_drift_metrics_measure_features_and_action_effect():
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        perception_replay_mode="online_student_rollout",
        perception_staleness_probe_num_envs=1,
        perception_staleness_probe_max_generation_age=8,
        perception_staleness_probe_interval=1,
    )
    policy._q_actor_dim = 2
    policy.device = torch.device("cpu")
    policy.q_actor_keys = (PRIV_PRED_KEY,)
    policy._q_actor_widths = (2,)
    policy._perception_ema_generation = 3
    policy._fastsac_q_action_scale = torch.ones(2)
    old = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    episode = _StudentPerceptionDriftEpisode(
        raw_fields={},
        collection_actor=old,
        collection_ema_generation=torch.tensor([1, 2, 3]),
        eligible_student_rows=torch.tensor([True, True, False]),
    )
    policy._student_perception_drift_episodes = [episode]
    policy._reencode_student_perception_drift_episode = MethodType(
        lambda _self, _episode: old + 1.0,
        policy,
    )
    policy._actor_dist_from_flat = MethodType(
        lambda _self, value: SimpleNamespace(mean=value),
        policy,
    )

    metrics = policy._student_perception_drift_metrics()
    assert metrics["enabled"] == 1.0
    assert metrics["available"] == 1.0
    assert metrics["episodes"] == 1.0
    assert metrics["rows"] == 2.0
    assert metrics["ema_age_mean"] == pytest.approx(1.5)
    assert metrics["actor_feature_mse"] == pytest.approx(1.0)
    assert metrics["perception_latent_mse"] == pytest.approx(1.0)
    assert metrics["action_normalized_rmse"] == pytest.approx(1.0)


def test_onset_to_terminal_diagnostics_round_trip_and_tolerate_legacy_state():
    """The onset-quality sums must persist, and older checkpoints must load.

    Without the onset-to-terminal distance there is no way to tell a genuine
    failure precursor from an early false positive, so the sums are part of the
    resumable state.  They shipped after the first v6 checkpoints, so a state
    dict lacking them is an older checkpoint rather than corruption.
    """

    def fresh():
        policy = TVKDDistributionalFastSACTeacherBC.__new__(
            TVKDDistributionalFastSACTeacherBC
        )
        nn.Module.__init__(policy)
        policy.teacher_value_bottleneck_detector = _detector()
        policy._reset_bottleneck_statistics()
        return policy

    saved = fresh()
    assert saved._bottleneck_onset_to_terminal_sum == 0.0
    assert saved._bottleneck_failed_episode_length_sum == 0.0
    saved._bottleneck_detected_count = 4
    saved._bottleneck_failed_student_episode_count = 5
    saved._bottleneck_onset_to_terminal_sum = 34.0
    saved._bottleneck_failed_episode_length_sum = 210.0
    state = saved._bottleneck_replay_checkpoint_state()
    assert state["onset_to_terminal_sum"] == pytest.approx(34.0)
    assert state["failed_episode_length_sum"] == pytest.approx(210.0)

    restored = fresh()
    restored._load_bottleneck_replay_checkpoint_state(state)
    assert restored._bottleneck_onset_to_terminal_sum == pytest.approx(34.0)
    assert restored._bottleneck_failed_episode_length_sum == pytest.approx(210.0)

    legacy_state = dict(state)
    legacy_state.pop("onset_to_terminal_sum")
    legacy_state.pop("failed_episode_length_sum")
    legacy = fresh()
    legacy._load_bottleneck_replay_checkpoint_state(legacy_state)
    assert legacy._bottleneck_onset_to_terminal_sum == 0.0
    assert legacy._bottleneck_failed_episode_length_sum == 0.0

    negative_state = dict(state)
    negative_state["onset_to_terminal_sum"] = -1.0
    with pytest.raises(ValueError, match="onset_to_terminal_sum"):
        fresh()._load_bottleneck_replay_checkpoint_state(negative_state)


def test_prefill_success_motion_counter_tracks_motions_and_imbalance():
    """Failure Teacher matching is same-motion, so per-motion commits matter."""
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    policy._teacher_prefill_successful_by_motion = {}

    # A cleared chunk list must not silence the counter: the caller captures
    # the episode before clear(), so None only means "nothing was committed".
    policy._count_prefill_success_motion(None)
    assert policy._prefill_success_motion_metrics() == {}

    def episode(motion_id, rows=3):
        return {
            REPLAY_MOTION_ID_KEY: torch.full((rows,), motion_id, dtype=torch.long)
        }

    for motion_id in (0, 0, 0, 1, 2, 2):
        policy._count_prefill_success_motion(episode(motion_id))
    assert policy._teacher_prefill_successful_by_motion == {0: 3, 1: 1, 2: 2}

    metrics = policy._prefill_success_motion_metrics()
    assert metrics["td3/prefill_successful_episodes_motion_0"] == 3.0
    assert metrics["td3/prefill_successful_episodes_motion_1"] == 1.0
    assert metrics["td3/prefill_successful_episodes_motion_2"] == 2.0
    assert metrics["td3/prefill_motion_imbalance_ratio"] == pytest.approx(3.0)

    # An episode is one motion by construction; a mixed one is a real defect.
    mixed = {REPLAY_MOTION_ID_KEY: torch.tensor([0, 1], dtype=torch.long)}
    with pytest.raises(RuntimeError, match="spans multiple motions"):
        policy._count_prefill_success_motion(mixed)


def test_teacher_value_boundary_contract_bootstraps_next_state_under_shared_c_t():
    # Rows: command completion, pure time limit, physical terminal, ordinary.
    terminated = torch.tensor([False, False, True, False])
    # Physical termination wins even when a boundary cause coincides.
    command_finished = torch.tensor([True, False, True, False])
    time_limit = torch.tensor([False, True, False, False])
    done = terminated | command_finished | time_limit
    truncations = replay_truncation_mask(
        terminated=terminated,
        command_finished=command_finished,
        time_limit=time_limit,
    )
    assert torch.equal(truncations, torch.tensor([False, True, False, False]))
    assert torch.equal(
        continuation_bootstrap_mask(dones=done, truncations=truncations),
        torch.tensor([False, True, False, True]),
    )

    discount = torch.tensor([1.0, 0.75, 0.4, 0.5])
    c_t = compute_continuation_coefficient(
        dones=done, truncations=truncations, discounts=discount
    )
    assert torch.equal(c_t, torch.tensor([0.0, 0.75, 0.0, 0.5]))

    # A cut row's next value is never read, so a sentinel or NaN there is inert.
    following = torch.tensor([float("nan"), 20.0, 999.0, 10.0])
    continuation = compute_teacher_value_continuation(
        teacher_v_next=following,
        continuation=c_t,
        gamma=0.99,
    )
    assert torch.allclose(
        continuation,
        torch.tensor([0.0, 0.99 * 0.75 * 20.0, 0.0, 0.99 * 0.5 * 10.0]),
    )

    # Command completion cuts: the residual reduces to r_t - V_T(s_t).
    terms = compute_teacher_value_terms(
        lambda observation: observation[:, 0],
        torch.tensor([[13.0]]),
        torch.tensor([[999.0]]),
        torch.tensor([0.16]),
        continuation=torch.zeros(1),
        gamma=0.99,
        tvkd_lambda=0.25,
    )
    assert terms.teacher_td_residual.item() == pytest.approx(-12.84, abs=1e-4)
    assert terms.shaped_reward.item() == pytest.approx(0.16 - 0.25 * 13.0, abs=1e-4)


def test_cache_validation_consumes_prepared_normalized_observations_once():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        teacher_value_cache_validate_fraction=1.0,
        gamma=0.99,
        tvkd_lambda=0.25,
        tvkd_potential_clip=None,
    )
    policy.q_critic_keys = ("critic",)
    policy._q_critic_widths = (1,)
    policy._vecnorm_snapshot = lambda: {"fixed": True}
    normalization_calls = []

    def normalize(value, keys, widths, snapshot):
        normalization_calls.append((value.clone(), keys, widths, snapshot))
        return value + 10.0

    policy._normalize_replay_flat = normalize
    policy._reencode_perception_windows = lambda *args, **kwargs: {}
    policy.get_frozen_teacher_value = lambda value: value[:, 0]
    raw = {
        "critic_observations": torch.tensor([[1.0], [2.0]]),
        "next_critic_observations": torch.tensor([[3.0], [4.0]]),
        "rewards": torch.tensor([0.5, 0.75]),
        "discounts": torch.ones(2),
        REPLAY_TEACHER_V_CURRENT_KEY: torch.tensor([11.0, 12.0]),
        REPLAY_TEACHER_V_NEXT_KEY: torch.tensor([13.0, 14.0]),
    }

    prepared = policy._prepare_dagger_learning_batch(raw)
    terms = policy._teacher_value_terms_from_batch(prepared, torch.ones(2))

    assert torch.equal(prepared["critic_observations"], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(
        prepared["next_critic_observations"], torch.tensor([[13.0], [14.0]])
    )
    assert torch.equal(terms.teacher_v, torch.tensor([11.0, 12.0]))
    assert torch.equal(terms.teacher_v_next, torch.tensor([13.0, 14.0]))
    # Exactly the two preparation calls: validation must not normalize again.
    assert len(normalization_calls) == 2


def test_cache_validation_checks_only_semantically_consumed_next_values():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        teacher_value_cache_validate_fraction=1.0,
        gamma=0.99,
        tvkd_lambda=0.25,
        tvkd_potential_clip=None,
    )
    policy.get_frozen_teacher_value = lambda value: value[:, 0]
    # Rows: ordinary, physical terminal, command completion, pure time limit.
    terminated = torch.tensor([False, True, False, False])
    command_finished = torch.tensor([False, False, True, False])
    time_limit = torch.tensor([False, False, False, True])
    continuation = compute_continuation_coefficient(
        dones=terminated | command_finished | time_limit,
        truncations=replay_truncation_mask(
            terminated=terminated,
            command_finished=command_finished,
            time_limit=time_limit,
        ),
        discounts=torch.ones(4),
    )
    assert torch.equal(continuation, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    batch = {
        "critic_observations": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "next_critic_observations": torch.tensor(
            [[11.0], [12.0], [13.0], [14.0]]
        ),
        "rewards": torch.zeros(4),
        "discounts": torch.ones(4),
        REPLAY_TEACHER_V_CURRENT_KEY: torch.tensor([1.0, 2.0, 3.0, 4.0]),
        # Only the cut rows carry the production zero sentinel; a pure time
        # limit bootstraps, so it stores its real pre-reset next value.
        REPLAY_TEACHER_V_NEXT_KEY: torch.tensor([11.0, 0.0, 0.0, 14.0]),
    }

    terms = policy._teacher_value_terms_from_batch(batch, continuation)
    assert torch.equal(terms.teacher_v_next, torch.tensor([11.0, 0.0, 0.0, 14.0]))

    corrupt_next = dict(batch)
    corrupt_next[REPLAY_TEACHER_V_NEXT_KEY] = torch.tensor([10.0, 0.0, 0.0, 14.0])
    with pytest.raises(RuntimeError, match="max_err_v_next"):
        policy._teacher_value_terms_from_batch(corrupt_next, continuation)

    # The truncation row is consumed now, so its cache is validated too.
    corrupt_timeout = dict(batch)
    corrupt_timeout[REPLAY_TEACHER_V_NEXT_KEY] = torch.tensor([11.0, 0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="max_err_v_next"):
        policy._teacher_value_terms_from_batch(corrupt_timeout, continuation)

    # Cut rows still skip validation: their sentinel is never consumed.
    sentinel_only = dict(batch)
    sentinel_only[REPLAY_TEACHER_V_NEXT_KEY] = torch.tensor([11.0, 99.0, 99.0, 14.0])
    policy._teacher_value_terms_from_batch(sentinel_only, continuation)

    corrupt_current = dict(batch)
    corrupt_current[REPLAY_TEACHER_V_CURRENT_KEY] = torch.tensor(
        [1.0, 20.0, 3.0, 4.0]
    )
    with pytest.raises(RuntimeError, match="max_err_v"):
        policy._teacher_value_terms_from_batch(corrupt_current, continuation)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_cache_validation_fraction_rejects_values_outside_unit_interval(fraction):
    cfg = TVKDDistributionalFastSACTeacherBCConfig(
        teacher_value_cache_validate_fraction=fraction
    )
    with pytest.raises(ValueError, match="teacher_value_cache_validate_fraction"):
        tvkd_module._validate_tvkd_algorithm_config(cfg)


def test_rollout_bottleneck_residual_caches_unique_student_states_and_raw_reward():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(gamma=0.9, q_batch_size=128)
    policy.q_critic_keys = ("critic",)
    policy._q_critic_widths = (1,)
    policy._q_critic_dim = 1
    policy._cat_replay_sources = lambda td, keys: td["critic"]
    policy._vecnorm_snapshot = lambda: {"frozen": True}
    normalization_calls = []

    def normalize(value, keys, widths, snapshot):
        normalization_calls.append((value.clone(), keys, widths, snapshot))
        return value

    policy._normalize_replay_flat = normalize
    teacher_value_calls = []

    def teacher_value(observation):
        teacher_value_calls.append(observation.clone())
        return observation[:, 0]

    policy.get_frozen_teacher_value = teacher_value
    policy._rollout_final_batch = {
        # Both rows are collector carry states; env 1 has already reset after
        # its timeout and must not use this value for bootstrapping.
        "next_critic_observations": torch.tensor([[4.0], [999.0]])
    }
    policy._truncation_final_batches = [
        {
            "indices": torch.tensor([5]),
            "next_critic_observations": torch.tensor([[40.0]]),
        }
    ]
    rollout = TensorDict(
        {
            "critic": torch.tensor(
                [
                    [[1.0], [2.0], [3.0]],
                    [[10.0], [20.0], [30.0]],
                ]
            ),
            "next": TensorDict(
                {
                    "reward": torch.tensor(
                        [
                            [[1.0], [2.0], [3.0]],
                            [[4.0], [5.0], [6.0]],
                        ]
                    ),
                    "done": torch.tensor(
                        [
                            [[False], [False], [True]],
                            [[False], [False], [True]],
                        ]
                    ),
                    "discount": torch.ones(2, 3, 1),
                    "terminated": torch.tensor(
                        [
                            [[False], [False], [True]],
                            [[False], [False], [False]],
                        ]
                    ),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [
                                    [[False], [False], [False]],
                                    [[False], [False], [True]],
                                ]
                            ),
                            "command_finished": torch.zeros(2, 3, 1, dtype=torch.bool),
                        },
                        batch_size=[2, 3],
                    ),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )
    student = torch.tensor([[True, True, True], [False, True, True]], dtype=torch.bool)

    residual = policy._student_teacher_td_residual_grid(rollout, student)

    # env 1 step 2 is a pure timeout: it bootstraps from the captured
    # pre-reset final state (40.0), not from its own value.
    assert torch.allclose(
        residual,
        torch.tensor([[1.8, 2.7, 0.0], [0.0, 12.0, 12.0]]),
    )
    assert len(normalization_calls) == 1
    assert normalization_calls[0][1:] == (
        ("critic",),
        (1,),
        {"frozen": True},
    )
    assert len(teacher_value_calls) == 1
    # Adjacent Student transitions share state values. The physical-terminal
    # reset and timeout reset carry are both excluded. The raw replay cache
    # still contains the captured timeout-final state, while source PPO's
    # boundary helper deliberately self-bootstraps from the current value.
    assert teacher_value_calls[0].squeeze(-1).tolist() == [1, 2, 3, 20, 30, 40]


def _old_loop_teacher_residual_reference(
    rollout: TensorDict,
    student: torch.Tensor,
    *,
    final_batch: dict[str, torch.Tensor],
    truncation_batches: list[dict[str, torch.Tensor]],
    gamma: float,
    teacher_value,
) -> dict[str, torch.Tensor]:
    """Test-only copy of the pre-vectorization state-selection loop."""
    num_envs, num_steps = (int(value) for value in rollout.batch_size)
    student = student.reshape(num_envs, num_steps).bool()
    residual_grid = torch.zeros((num_envs, num_steps), dtype=torch.float32)
    if not bool(student.any()):
        return {
            "residual": residual_grid,
            "raw_states": torch.empty(0, rollout["critic"].shape[-1]),
            "sorted_state_ids": torch.empty(0, dtype=torch.long),
            "virtual_ids": torch.empty(0, dtype=torch.long),
            "captured_student_raw": torch.empty(0, rollout["critic"].shape[-1]),
        }

    reward = rollout["next", "reward"].sum(dim=-1)
    done = rollout["next", "done"].reshape(num_envs, num_steps).bool()
    terminated = rollout["next", "terminated"].reshape(num_envs, num_steps).bool()
    command_finished = (
        rollout["next", "stats", "command_finished"].reshape(num_envs, num_steps).bool()
    )
    time_limit = (
        rollout["next", "stats", "episode_time_limit"]
        .reshape(num_envs, num_steps)
        .bool()
    )
    replay_discount = rollout["next", "discount"].reshape(num_envs, num_steps)
    truncation = time_limit & ~command_finished & ~terminated
    q_bootstrap = truncation | ~done
    continuation_grid = replay_discount * q_bootstrap.float()

    state_rows = []
    state_ids = []
    current_state_ids = []
    next_state_ids = []
    reward_rows = []
    q_bootstrap_rows = []
    continuation_rows = []
    flat_positions = []
    env_state_base = torch.arange(num_envs) * (num_steps + 1)
    timeout_next_id_by_transition = torch.full(
        (num_envs * num_steps,), -1, dtype=torch.long
    )
    virtual_ids = torch.empty(0, dtype=torch.long)
    captured_student_raw = torch.empty(0, rollout["critic"].shape[-1])
    if truncation_batches:
        captured_indices = torch.cat(
            [batch["indices"] for batch in truncation_batches], dim=0
        ).long()
        captured_next_raw = torch.cat(
            [batch["next_critic_observations"] for batch in truncation_batches],
            dim=0,
        )
        captured_student = student.reshape(-1).index_select(0, captured_indices)
        captured_indices = captured_indices[captured_student]
        captured_student_raw = captured_next_raw[captured_student]
        virtual_ids = num_envs * (num_steps + 1) + torch.arange(
            captured_indices.numel(), dtype=torch.long
        )
        timeout_next_id_by_transition.index_copy_(0, captured_indices, virtual_ids)
        state_rows.append(captured_student_raw)
        state_ids.append(virtual_ids)

    previous_bootstrap_student = torch.zeros(num_envs, dtype=torch.bool)
    current_raw = rollout["critic"][:, 0]
    for step in range(num_steps):
        current_needed = student[:, step] | previous_bootstrap_student
        needed_envs = current_needed.nonzero(as_tuple=False).squeeze(-1)
        state_rows.append(current_raw[current_needed])
        state_ids.append(env_state_base[needed_envs] + step)
        next_raw = (
            rollout["critic"][:, step + 1]
            if step + 1 < num_steps
            else final_batch["next_critic_observations"]
        )
        mask = student[:, step]
        reward_rows.append(reward[:, step][mask])
        q_bootstrap_rows.append(q_bootstrap[:, step][mask])
        continuation_rows.append(continuation_grid[:, step][mask])
        env_indices = mask.nonzero(as_tuple=False).squeeze(-1)
        current_state_ids.append(env_state_base[env_indices] + step)
        flat_position = env_indices * num_steps + step
        regular_next_ids = env_state_base[env_indices] + step + 1
        timeout_next_ids = timeout_next_id_by_transition.index_select(0, flat_position)
        next_state_ids.append(
            torch.where(
                truncation[:, step][mask],
                timeout_next_ids,
                regular_next_ids,
            )
        )
        flat_positions.append(flat_position)
        previous_bootstrap_student = mask & q_bootstrap[:, step] & ~truncation[:, step]
        current_raw = next_raw

    final_envs = previous_bootstrap_student.nonzero(as_tuple=False).squeeze(-1)
    state_rows.append(current_raw[previous_bootstrap_student])
    state_ids.append(env_state_base[final_envs] + num_steps)
    all_state_ids = torch.cat(state_ids)
    sorted_state_ids, order = all_state_ids.sort()
    raw_states = torch.cat(state_rows).index_select(0, order)
    state_values = teacher_value(raw_states).float()
    current_ids = torch.cat(current_state_ids)
    next_ids = torch.cat(next_state_ids)
    teacher_v = state_values.index_select(
        0, torch.searchsorted(sorted_state_ids, current_ids)
    )
    transition_q_bootstrap = torch.cat(q_bootstrap_rows).bool()
    teacher_v_next = torch.zeros_like(teacher_v)
    if bool(transition_q_bootstrap.any()):
        next_positions = torch.searchsorted(
            sorted_state_ids, next_ids[transition_q_bootstrap]
        )
        teacher_v_next[transition_q_bootstrap] = state_values.index_select(
            0, next_positions
        )
    continuation = compute_teacher_value_continuation(
        teacher_v_next=teacher_v_next,
        continuation=torch.cat(continuation_rows),
        gamma=gamma,
    )
    transition_residual = torch.cat(reward_rows).float() + continuation - teacher_v
    residual_grid.reshape(-1).index_copy_(
        0, torch.cat(flat_positions), transition_residual
    )
    return {
        "residual": residual_grid,
        "raw_states": raw_states,
        "sorted_state_ids": sorted_state_ids,
        "virtual_ids": virtual_ids,
        "captured_student_raw": captured_student_raw,
    }


def test_vectorized_rollout_teacher_residual_is_exact_old_loop_equivalent():
    generator = torch.Generator().manual_seed(9_381)
    gamma = 0.875
    total_evaluated_states = 0
    last_case = None

    def raw_state(identifier: torch.Tensor) -> torch.Tensor:
        identifier = identifier.float()
        return torch.stack(
            (
                identifier + 0.125,
                identifier * 0.5 + 0.25,
                identifier * -0.25 + 0.5,
            ),
            dim=-1,
        )

    teacher_weights = torch.tensor([0.5, -0.25, 0.125])

    def teacher_value(observation: torch.Tensor) -> torch.Tensor:
        return (observation * teacher_weights).sum(dim=-1)

    for case_index in range(32):
        num_envs = int(torch.randint(2, 6, (), generator=generator))
        num_steps = int(torch.randint(2, 8, (), generator=generator))
        row_count = num_envs * num_steps
        cause = torch.randint(0, 8, (row_count,), generator=generator)
        boundary_coverage = torch.arange(min(row_count, 8), dtype=torch.long)
        cause[: boundary_coverage.numel()] = boundary_coverage
        cause = cause.reshape(num_envs, num_steps)
        terminated = cause.bitwise_and(1).bool()
        command_finished = cause.bitwise_and(2).bool()
        time_limit = cause.bitwise_and(4).bool()
        done = terminated | command_finished | time_limit
        truncation = time_limit & ~command_finished & ~terminated

        student = torch.rand((num_envs, num_steps), generator=generator) < (
            0.35 + 0.01 * case_index
        )
        # Make every boundary-bit combination an executed Student row whenever
        # the sampled grid has room for the complete coverage prefix.
        student.reshape(-1)[: min(row_count, 8)] = True
        pure_timeout_rows = truncation.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if pure_timeout_rows.numel():
            student.reshape(-1)[pure_timeout_rows[0]] = True

        regular_ids = torch.arange(num_envs).unsqueeze(1) * (
            num_steps + 1
        ) + torch.arange(num_steps).unsqueeze(0)
        rollout_raw = raw_state(regular_ids)
        final_ids = torch.arange(num_envs) * (num_steps + 1) + num_steps
        final_batch = {"next_critic_observations": raw_state(final_ids)}
        timeout_raw_by_flat = raw_state(
            torch.arange(row_count) + 10_000 + case_index * row_count
        )
        truncation_batches = []
        for step in range(num_steps):
            timeout_envs = truncation[:, step].nonzero(as_tuple=False).squeeze(-1)
            if timeout_envs.numel():
                flat_indices = timeout_envs * num_steps + step
                truncation_batches.append(
                    {
                        "indices": flat_indices,
                        "next_critic_observations": timeout_raw_by_flat.index_select(
                            0, flat_indices
                        ),
                    }
                )

        reward = (
            torch.randint(
                -16, 17, (num_envs, num_steps, 2), generator=generator
            ).float()
            / 8.0
        )
        discount = (
            torch.randint(1, 5, (num_envs, num_steps, 1), generator=generator).float()
            / 4.0
        )
        rollout = TensorDict(
            {
                "critic": rollout_raw,
                "next": TensorDict(
                    {
                        "reward": reward,
                        "done": done.unsqueeze(-1),
                        "discount": discount,
                        "terminated": terminated.unsqueeze(-1),
                        "stats": TensorDict(
                            {
                                "episode_time_limit": time_limit.unsqueeze(-1),
                                "command_finished": command_finished.unsqueeze(-1),
                            },
                            batch_size=[num_envs, num_steps],
                        ),
                    },
                    batch_size=[num_envs, num_steps],
                ),
            },
            batch_size=[num_envs, num_steps],
        )
        expected = _old_loop_teacher_residual_reference(
            rollout,
            student,
            final_batch=final_batch,
            truncation_batches=truncation_batches,
            gamma=gamma,
            teacher_value=teacher_value,
        )

        policy = TVKDDistributionalFastSACTeacherBC.__new__(
            TVKDDistributionalFastSACTeacherBC
        )
        nn.Module.__init__(policy)
        policy.cfg = SimpleNamespace(gamma=gamma, q_batch_size=100_000)
        policy.q_critic_keys = ("critic",)
        policy._q_critic_widths = (3,)
        policy._q_critic_dim = 3
        policy._cat_replay_sources = lambda td, keys: td["critic"]
        policy._vecnorm_snapshot = lambda: None
        policy._normalize_replay_flat = lambda value, keys, widths, snapshot: value
        policy._rollout_final_batch = final_batch
        policy._truncation_final_batches = truncation_batches
        teacher_calls = []

        def recorded_teacher_value(observation):
            teacher_calls.append(observation.clone())
            return teacher_value(observation)

        policy.get_frozen_teacher_value = recorded_teacher_value
        actual = policy._student_teacher_td_residual_grid(rollout, student)

        assert torch.equal(actual, expected["residual"])
        assert len(teacher_calls) == 1
        assert torch.equal(teacher_calls[0], expected["raw_states"])
        assert teacher_calls[0].shape[0] == expected["sorted_state_ids"].numel()
        total_evaluated_states += int(teacher_calls[0].shape[0])
        virtual_ids = expected["virtual_ids"]
        if virtual_ids.numel():
            assert torch.equal(
                virtual_ids,
                num_envs * (num_steps + 1) + torch.arange(virtual_ids.numel()),
            )
            assert torch.equal(
                expected["sorted_state_ids"][-virtual_ids.numel() :],
                virtual_ids,
            )
            assert torch.equal(
                teacher_calls[0][-virtual_ids.numel() :],
                expected["captured_student_raw"],
            )
        last_case = (policy, rollout, teacher_calls)

    assert total_evaluated_states > 0
    policy, rollout, teacher_calls = last_case
    teacher_calls.clear()
    zero_student = torch.zeros(tuple(rollout.batch_size), dtype=torch.bool)
    zero_residual = policy._student_teacher_td_residual_grid(rollout, zero_student)
    assert torch.equal(zero_residual, torch.zeros_like(zero_residual))
    assert teacher_calls == []


def test_failed_rollout_registers_detected_bottleneck_in_existing_phase_source(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_teacher_value_bottleneck_replay=True,
        failure_phase_lookback_steps=10,
        failure_phase_samples_per_failure=10,
        failure_phase_num_bins=10,
        bottleneck_fallback_mode="none",
        bottleneck_include_unsuccessful_timeouts=False,
    )
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy._bottleneck_episode_histories = None
    policy._failure_phase_histogram = torch.zeros(10, dtype=torch.float64)
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0
    policy._failure_histogram_device_cache = {}
    policy._rollout_final_batch = {}
    residual = torch.full((1, 40), 0.1)
    residual[:, 18:25] = -0.2
    policy._student_teacher_td_residual_grid = lambda rollout, student: residual.clone()
    policy._reference_phase = lambda rollout: rollout["reference_phase"]
    policy._record_teacher_phase_match_distances = lambda phases, motion_ids: None

    def legacy_parent(owner, candidate_rollout):
        del owner, candidate_rollout
        raise AssertionError("TVKD v5 must never call the parent failure path")

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_update_failure_phase_histogram",
        legacy_parent,
    )

    shape = (1, 40, 1)
    final_true = torch.zeros(shape, dtype=torch.bool)
    final_true[:, -1] = True
    rollout = TensorDict(
        {
            "reference_phase": (
                torch.arange(40, dtype=torch.float32).reshape(1, 40, 1) / 40.0
            ),
            "step_count": torch.arange(40).reshape(shape),
            REPLAY_MOTION_ID_KEY: torch.zeros(shape, dtype=torch.long),
            "is_student_action": torch.ones(shape, dtype=torch.bool),
            "dagger_safe_takeover": torch.zeros(shape, dtype=torch.bool),
            "is_init": torch.zeros(shape, dtype=torch.bool),
            "next": TensorDict(
                {
                    "done": final_true.clone(),
                    "terminated": final_true.clone(),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(shape, dtype=torch.bool),
                            "command_finished": torch.zeros(shape, dtype=torch.bool),
                        },
                        batch_size=[1, 40],
                    ),
                },
                batch_size=[1, 40],
            ),
        },
        batch_size=[1, 40],
    )

    assert policy._update_failure_phase_histogram(rollout) == 10
    assert policy._last_bottleneck_metadata["bottleneck_step"] == 20
    assert policy._last_bottleneck_metadata["bottleneck_confirmation_step"] == 22
    assert policy._last_bottleneck_metadata["bottleneck_phase"] == pytest.approx(0.5)
    assert policy._last_bottleneck_metadata["precursor_start_step"] == 10
    assert policy._last_bottleneck_metadata["precursor_end_step"] == 19
    assert policy._last_bottleneck_metadata["precursor_row_count"] == 10
    assert policy._last_bottleneck_metadata["fallback"] == "none"
    assert policy._failure_phase_histogram.sum().item() == 10.0
    assert policy._failure_phase_histogram[5].item() == 0.0
    assert policy._bottleneck_detected_count == 1
    assert policy._bottleneck_teacher_sequences_inserted == 1
    assert len(policy._pending_student_focus_events) == 1
    replay_episode_id, replay_steps = policy._pending_student_focus_events[0]
    assert replay_episode_id == 0
    assert torch.equal(replay_steps, torch.arange(10, 20))
    assert torch.equal(
        policy._student_replay_episode_id_grid,
        torch.zeros(1, 40, dtype=torch.long),
    )
    assert torch.equal(
        policy._student_replay_episode_step_grid,
        torch.arange(40, dtype=torch.long).reshape(1, 40),
    )

    # A single non-negative early Student action and an all-Teacher failure
    # contain no sustained negative Student-value interval. Neither creates a
    # verified anchor or any terminal fallback.
    early_student = rollout.clone()
    early_student["is_student_action"].zero_()
    early_student["is_student_action"][:, 6] = True
    assert policy._update_failure_phase_histogram(early_student) == 0
    teacher_only = rollout.clone()
    teacher_only["is_student_action"].zero_()
    assert policy._update_failure_phase_histogram(teacher_only) == 0
    assert policy._bottleneck_failed_student_episode_count == 2
    assert policy._bottleneck_teacher_sequences_inserted == 1


def test_prevention_rows_are_strictly_pre_onset_and_never_backfilled():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        failure_phase_lookback_steps=10,
        failure_phase_samples_per_failure=10,
    )
    history = {
        "phase": [index / 25.0 for index in range(25)],
        "teacher_td_residual": [0.0] * 25,
        "source_id": [SOURCE_STUDENT] * 25,
        "motion_id": [0] * 25,
        "true_terminal": [False] * 24 + [True],
        "timeout": [False] * 25,
        "replay_valid": [False] * 6 + [True] * 19,
    }
    history["source_id"][12] = SOURCE_UNIFORM_TEACHER
    history["replay_valid"][15] = False
    history["motion_id"][17] = 1

    selected = policy._student_bottleneck_anchor_indices(history, 20)
    assert torch.equal(selected, torch.tensor([10, 11, 13, 14, 16, 18, 19]))
    assert not bool((selected >= 20).any())

    # The interval remains episode-time based: shortages are not filled from
    # before onset-10 or from onset/post-onset rows.
    assert torch.equal(
        policy._student_bottleneck_anchor_indices(history, 10),
        torch.tensor([6, 7, 8, 9]),
    )
    assert policy._student_bottleneck_anchor_indices(history, 6).numel() == 0


def test_detected_onset_without_same_motion_precursor_creates_no_focus():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        failure_phase_lookback_steps=10,
        failure_phase_samples_per_failure=10,
        failure_phase_num_bins=10,
        bottleneck_fallback_mode="none",
    )
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy._failure_phase_histogram = torch.zeros(10, dtype=torch.float64)
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0
    policy._failure_histogram_device_cache = {}
    policy._record_teacher_phase_match_distances = lambda phases, motion_ids: None
    residual = [0.1] * 40
    residual[18:25] = [-0.2] * 7
    motion_id = [0] * 40
    motion_id[20] = 1
    history = {
        "phase": [index / 40.0 for index in range(40)],
        "teacher_td_residual": residual,
        "source_id": [SOURCE_STUDENT] * 40,
        "motion_id": motion_id,
        "true_terminal": [False] * 39 + [True],
        "timeout": [False] * 40,
        "replay_valid": [False] * 6 + [True] * 34,
    }

    assert policy._process_failed_student_episode(history, replay_episode_id=3) == 0
    assert policy._bottleneck_detected_count == 1
    assert policy._bottleneck_selected_count == 1
    assert policy._last_bottleneck_metadata["precursor_row_count"] == 0
    assert policy._pending_student_focus_events == []
    assert policy._failure_phase_histogram.sum().item() == 0.0
    assert policy._failure_phase_episode_count == 0


def test_unsuccessful_timeout_does_not_enter_physical_failure_curriculum():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_teacher_value_bottleneck_replay=True,
        bottleneck_include_unsuccessful_timeouts=False,
    )
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy._bottleneck_episode_histories = None
    policy._rollout_final_batch = {}
    policy._reference_phase = lambda rollout: rollout["reference_phase"]
    policy._student_teacher_td_residual_grid = lambda rollout, student: torch.full(
        tuple(rollout.batch_size), -1.0
    )
    processed = []
    policy._process_failed_student_episode = lambda *args, **kwargs: processed.append(
        (args, kwargs)
    )

    shape = (1, 12, 1)
    done = torch.zeros(shape, dtype=torch.bool)
    done[:, -1] = True
    rollout = TensorDict(
        {
            "reference_phase": torch.arange(12, dtype=torch.float32).reshape(shape)
            / 12.0,
            "step_count": torch.arange(12).reshape(shape),
            REPLAY_MOTION_ID_KEY: torch.zeros(shape, dtype=torch.long),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.ones(shape, dtype=torch.bool),
            "is_init": torch.zeros(shape, dtype=torch.bool),
            "next": TensorDict(
                {
                    "done": done,
                    "terminated": torch.zeros(shape, dtype=torch.bool),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": done.clone(),
                            "command_finished": torch.zeros(shape, dtype=torch.bool),
                        },
                        batch_size=[1, 12],
                    ),
                },
                batch_size=[1, 12],
            ),
        },
        batch_size=[1, 12],
    )

    assert policy._update_failure_phase_histogram(rollout) == 0
    assert processed == []
    assert policy._bottleneck_unsuccessful_episode_count == 1


def test_student_focus_events_mark_existing_ring_and_current_transition_chunk(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy.dagger_replay = _TD3DeviceReplay(16, "cpu")
    policy.dagger_replay.extend(
        {
            "actions": torch.zeros(3, 1),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.ones(3, dtype=torch.bool),
            STUDENT_REPLAY_EPISODE_ID_KEY: torch.tensor([4, 4, 7]),
            STUDENT_REPLAY_EPISODE_STEP_KEY: torch.tensor([0, 2, 0]),
            FAILURE_PHASE_STUDENT_SOURCE_KEY: torch.zeros(3, dtype=torch.bool),
        }
    )
    policy._pending_student_focus_events = [
        (4, torch.tensor([2])),
        (9, torch.tensor([1])),
        (11, torch.tensor([5])),
    ]
    id_grid = _DeviceMoveCounter(torch.tensor([[4, 9]]))
    step_grid = _DeviceMoveCounter(torch.tensor([[1, 1]]))
    policy._student_replay_episode_id_grid = id_grid
    policy._student_replay_episode_step_grid = step_grid
    chunks = (
        {
            "actions": torch.ones(1, 1),
            _PREFILL_ENV_INDEX_KEY: torch.tensor([0]),
            _PREFILL_STEP_INDEX_KEY: torch.tensor([0]),
        },
        {
            "actions": torch.ones(1, 1),
            _PREFILL_ENV_INDEX_KEY: torch.tensor([0]),
            _PREFILL_STEP_INDEX_KEY: torch.tensor([1]),
        },
    )
    sentinel = object()

    def baseline(self, td):
        assert td is sentinel
        yield from (dict(chunk) for chunk in chunks)

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_dagger_transition_chunks",
        baseline,
    )

    annotated = tuple(policy._dagger_transition_chunks(sentinel))

    assert torch.equal(
        policy.dagger_replay.data[FAILURE_PHASE_STUDENT_SOURCE_KEY][:3],
        torch.tensor([False, True, False]),
    )
    assert torch.equal(
        torch.cat([chunk[STUDENT_REPLAY_EPISODE_ID_KEY] for chunk in annotated]),
        torch.tensor([4, 9]),
    )
    assert torch.equal(
        torch.cat([chunk[STUDENT_REPLAY_EPISODE_STEP_KEY] for chunk in annotated]),
        torch.tensor([1, 1]),
    )
    assert torch.equal(
        torch.cat([chunk[FAILURE_PHASE_STUDENT_SOURCE_KEY] for chunk in annotated]),
        torch.tensor([False, True]),
    )
    assert len(id_grid.calls) == 1
    assert len(step_grid.calls) == 1
    # Two requested rows were present across the old ring/current chunk; the
    # third event was already evicted or otherwise unavailable.
    assert policy._student_focus_rows_marked == 2
    assert policy._student_focus_rows_missing == 1
    assert policy._pending_student_focus_events == []
    assert policy._student_replay_episode_id_grid is None
    assert policy._student_replay_episode_step_grid is None


def test_shaping_without_bottleneck_attaches_nonzero_teacher_value_cache(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=True,
        tvkd_lambda=0.25,
        use_teacher_value_bottleneck_replay=False,
    )
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy.dagger_replay = _TD3DeviceReplay(8, "cpu")
    current_grid = _DeviceMoveCounter(
        torch.tensor([[11.0, 12.0], [13.0, 14.0]])
    )
    next_grid = _DeviceMoveCounter(
        torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    )
    policy._rollout_teacher_v_current_grid = current_grid
    policy._rollout_teacher_v_next_grid = next_grid
    chunks = (
        {
            "actions": torch.ones(1, 1),
            "rewards": torch.ones(1),
            _PREFILL_ENV_INDEX_KEY: torch.tensor([1]),
            _PREFILL_STEP_INDEX_KEY: torch.tensor([0]),
        },
        {
            "actions": torch.ones(1, 1),
            "rewards": torch.ones(1),
            _PREFILL_ENV_INDEX_KEY: torch.tensor([0]),
            _PREFILL_STEP_INDEX_KEY: torch.tensor([1]),
        },
    )
    sentinel = object()

    def baseline(self, td):
        del self
        assert td is sentinel
        yield from (dict(chunk) for chunk in chunks)

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_dagger_transition_chunks",
        baseline,
    )

    annotated = tuple(policy._dagger_transition_chunks(sentinel))

    assert torch.equal(
        torch.cat([chunk[REPLAY_TEACHER_V_CURRENT_KEY] for chunk in annotated]),
        torch.tensor([13.0, 12.0]),
    )
    assert torch.equal(
        torch.cat([chunk[REPLAY_TEACHER_V_NEXT_KEY] for chunk in annotated]),
        torch.tensor([23.0, 22.0]),
    )
    assert torch.equal(
        torch.cat([chunk[STUDENT_REPLAY_EPISODE_ID_KEY] for chunk in annotated]),
        torch.tensor([-1, -1]),
    )
    assert torch.equal(
        torch.cat([chunk[STUDENT_REPLAY_EPISODE_STEP_KEY] for chunk in annotated]),
        torch.tensor([-1, -1]),
    )
    assert len(current_grid.calls) == 1
    assert len(next_grid.calls) == 1
    assert policy._rollout_teacher_v_current_grid is None
    assert policy._rollout_teacher_v_next_grid is None


def test_tvkd_actor_is_exact_fixed_baseline_implementation_without_scheduler():
    assert (
        TVKDDistributionalFastSACTeacherBC._actor_update
        is DistributionalFastSACTeacherBC._actor_update
    )
    # Replay insertion is the one intentional rollout seam: it annotates
    # exact failed-bottleneck Student rows without changing the Actor loss.
    assert (
        TVKDDistributionalFastSACTeacherBC._dagger_transition_chunks
        is not DistributionalFastSACTeacherBC._dagger_transition_chunks
    )
    assert not hasattr(tvkd_module, "TeacherValueBCScheduler")
    for removed_name in (
        "_adaptive_bc_changes_actor_loss",
        "_empty_scheduler_metrics",
        "_update_scheduler_from_recent_student",
    ):
        assert removed_name not in TVKDDistributionalFastSACTeacherBC.__dict__

    cfg = TVKDDistributionalFastSACTeacherBCConfig(lambda_bc=0.73)
    detector = _detector()
    initial_bc = cfg.lambda_bc
    assert detector.state_dict() == {}
    assert cfg.lambda_bc == pytest.approx(initial_bc)


def test_feature_off_tvkd_matches_baseline_alpha_and_policy_cadence_counters():
    class _TrainableTwinC51(nn.Module):
        def __init__(self):
            super().__init__()
            probabilities = torch.tensor(
                [
                    [[0.2, 0.3, 0.5]] * 4,
                    [[0.5, 0.3, 0.2]] * 4,
                ],
                dtype=torch.float32,
            )
            self.logits = nn.Parameter(probabilities.log())
            self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

        def forward(self, observations, actions):
            del actions
            return self.logits[:, : observations.shape[0]]

    def build(policy_type):
        policy = policy_type.__new__(policy_type)
        nn.Module.__init__(policy)
        policy.cfg = SimpleNamespace(
            gamma=0.99,
            sac_use_autotune=True,
            sac_alpha_update_cadence="actor",
            sac_policy_frequency=8,
            sac_log_std_min=-5.0,
            sac_log_std_max=1.0,
            sac_max_grad_norm=1.0e6,
            sac_tau=0.05,
            q_action_input_gain=1.0,
            action_support_clip=20.0,
            eta_sac=0.7,
            lambda_bc=1.3,
            dagger_actor_huber_delta=0.4,
            use_tvkd_value_shaping=False,
            tvkd_lambda=0.25,
            use_teacher_value_bottleneck_replay=False,
            perception_replay_mode="legacy_online_student",
        )
        policy._fastsac_q_action_center = torch.zeros(1)
        policy._fastsac_q_action_scale = torch.ones(1)
        policy._fastsac_action_low = torch.full((1,), -20.0)
        policy._fastsac_action_high = torch.full((1,), 20.0)
        policy._fastsac_actor_action_center = torch.zeros(1)
        policy._fastsac_actor_action_scale = torch.full((1,), 20.0)
        policy._fastsac_entropy_reference_log_scale_sum = 0.0
        policy.actor_adapt = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.actor_adapt.weight.fill_(0.25)
        raw_log_std = policy._inverse_smooth_log_std(
            torch.tensor(-1.0),
            policy.cfg.sac_log_std_min,
            policy.cfg.sac_log_std_max,
        )
        policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
            action_dim=1,
            initial_log_std=raw_log_std,
            device="cpu",
        )
        policy.qnet = _TrainableTwinC51()
        policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
        policy.critic_optimizer = torch.optim.Adam(policy.qnet.parameters(), lr=1e-3)
        policy.actor_optimizer = torch.optim.Adam(
            (
                *policy.actor_adapt.parameters(),
                *policy.bc_dagger_sac_adapter.parameters(),
            ),
            lr=3e-4,
        )
        policy.log_alpha = nn.Parameter(torch.tensor(-2.0))
        policy.alpha_optimizer = torch.optim.Adam((policy.log_alpha,), lr=4e-3)
        policy.target_entropy = -1.0
        policy.critic_update_count = 0
        policy.actor_update_count = 0
        policy.alpha_update_count = 0
        policy.sac_actor_update_count = 0
        policy.sac_alpha_update_count = 0
        policy.sac_action_rng = torch.Generator().manual_seed(314159)
        return policy

    baseline = build(DistributionalFastSACTeacherBC)
    tvkd = build(TVKDDistributionalFastSACTeacherBC)
    assert not tvkd._tvkd_enabled()
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [-1.0], [0.5]]),
        "next_observations": torch.tensor([[0.2], [-0.7], [1.3], [0.9]]),
        "critic_observations": torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        "next_critic_observations": torch.tensor([[0.5], [0.6], [0.7], [0.8]]),
        "actions": torch.tensor([[0.3], [-0.4], [0.1], [0.2]]),
        "rewards": torch.tensor([0.2, -0.1, 0.3, 0.0]),
        "dones": torch.tensor([False, False, True, True]),
        "truncations": torch.tensor([False, False, False, True]),
        "discounts": torch.tensor([1.0, 0.95, 1.0, 0.75]),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.5], [-0.2], [0.1], [0.0]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False, True]),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([True, False, True, False]),
    }
    for _ in range(16):
        for policy in (baseline, tvkd):
            policy._critic_update(batch)
            policy._maybe_delayed_actor_and_targets(batch)

    expected_counters = (16, 2, 2, 2, 2)
    for policy in (baseline, tvkd):
        assert (
            policy.critic_update_count,
            policy.actor_update_count,
            policy.alpha_update_count,
            policy.sac_actor_update_count,
            policy.sac_alpha_update_count,
        ) == expected_counters
    assert torch.equal(baseline.log_alpha, tvkd.log_alpha)
    for baseline_module, tvkd_module_ in (
        (baseline.actor_adapt, tvkd.actor_adapt),
        (baseline.bc_dagger_sac_adapter, tvkd.bc_dagger_sac_adapter),
        (baseline.qnet, tvkd.qnet),
        (baseline.qnet_target, tvkd.qnet_target),
    ):
        for baseline_parameter, tvkd_parameter in zip(
            baseline_module.parameters(), tvkd_module_.parameters()
        ):
            assert torch.equal(baseline_parameter, tvkd_parameter)
    assert torch.equal(
        baseline.sac_action_rng.get_state(), tvkd.sac_action_rng.get_state()
    )


def test_tvkd_default_q_and_actor_mix_is_four_way_35_15_35_15():
    cfg = TVKDDistributionalFastSACTeacherBCConfig()
    total_student = 1.0 - float(cfg.q_teacher_replay_ratio)
    focused_student = total_student * float(cfg.failure_phase_student_fraction)
    uniform_student = total_student - focused_student
    focused_teacher = float(cfg.q_teacher_replay_ratio) * float(
        cfg.failure_phase_teacher_fraction
    )
    uniform_teacher = float(cfg.q_teacher_replay_ratio) - focused_teacher

    assert uniform_student == pytest.approx(0.35)
    assert focused_student == pytest.approx(0.15)
    assert uniform_teacher == pytest.approx(0.35)
    assert focused_teacher == pytest.approx(0.15)
    for method in ("_sample_balanced_q_batch", "_sample_actor_batch"):
        assert getattr(TVKDDistributionalFastSACTeacherBC, method) is getattr(
            DistributionalFastSACTeacherBC, method
        )


def test_q_and_actor_batches_split_student_half_and_preserve_source_masks():
    policy, uniform_values, focused_values = _student_split_sampling_policy(
        focused_rows=6
    )

    critic = policy._sample_balanced_q_batch()
    actor = policy._sample_actor_batch()

    for batch in (critic, actor):
        teacher = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
        focused_teacher = batch[FAILURE_PHASE_TEACHER_SOURCE_KEY]
        focused_student = batch[FAILURE_PHASE_STUDENT_SOURCE_KEY]
        uniform_student = ~teacher & ~focused_student
        assert teacher.sum().item() == 10
        assert (teacher & ~focused_teacher).sum().item() == 7
        assert focused_teacher.sum().item() == 3
        assert uniform_student.sum().item() == 7
        assert focused_student.sum().item() == 3
        assert not (focused_teacher & ~teacher).any()
        assert not (focused_student & teacher).any()
        assert "successful_student_source" not in batch

    critic_actions = critic["actions"][:, 0]
    critic_focused = critic_actions[critic[FAILURE_PHASE_STUDENT_SOURCE_KEY]]
    critic_uniform = critic_actions[
        ~critic[DAGGER_Q_TEACHER_SOURCE_KEY] & ~critic[FAILURE_PHASE_STUDENT_SOURCE_KEY]
    ]
    assert set(critic_focused.tolist()) <= focused_values
    assert set(critic_uniform.tolist()) <= uniform_values
    assert critic_focused.unique().numel() == 3

    actor_rows = actor["critic_observations"][:, 0]
    actor_focused = actor_rows[actor[FAILURE_PHASE_STUDENT_SOURCE_KEY]] + 1.0
    actor_uniform = (
        actor_rows[
            ~actor[DAGGER_Q_TEACHER_SOURCE_KEY]
            & ~actor[FAILURE_PHASE_STUDENT_SOURCE_KEY]
        ]
        + 1.0
    )
    assert set(actor_focused.tolist()) <= focused_values
    assert set(actor_uniform.tolist()) <= uniform_values
    assert actor_focused.unique().numel() == 3


def test_prefetched_plan_preserves_student_focus_masks_and_rng_batches():
    direct, _, _ = _student_split_sampling_policy(focused_rows=6, seed=977)
    prefetched, _, _ = _student_split_sampling_policy(focused_rows=6, seed=977)

    expected_q = direct._sample_balanced_q_batch()
    expected_actor = direct._sample_actor_batch()
    (plan,) = prefetched._prefetch_curriculum_sample_plans(1)
    actual_q = prefetched._sample_balanced_q_batch(plan)
    actual_actor = prefetched._sample_actor_batch(
        plan.actor_indices,
        plan.actor_teacher_indices,
        plan.actor_teacher_focused,
        plan.actor_student_focused,
    )

    assert torch.equal(direct.q_rng.get_state(), prefetched.q_rng.get_state())
    assert plan.student_focused is not None
    assert plan.student_focused.sum().item() == 3
    assert plan.actor_student_focused is not None
    assert plan.actor_student_focused.sum().item() == 3
    assert plan.student_indices.device.type == "cpu"
    assert plan.actor_indices is not None
    assert plan.actor_indices.device.type == "cpu"
    for expected, actual in (
        (expected_q, actual_q),
        (expected_actor, actual_actor),
    ):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize(("available", "actual"), ((0, 0), (2, 2)))
def test_student_focus_quota_is_a_cap_and_shortfall_backfills_uniform(
    available, actual
):
    policy, uniform_values, focused_values = _student_split_sampling_policy(
        focused_rows=available
    )

    for batch, row_values in (
        (policy._sample_balanced_q_batch(), lambda value: value["actions"][:, 0]),
        (
            policy._sample_actor_batch(),
            lambda value: value["critic_observations"][:, 0] + 1.0,
        ),
    ):
        focused = batch[FAILURE_PHASE_STUDENT_SOURCE_KEY]
        teacher = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
        uniform = ~teacher & ~focused
        assert focused.sum().item() == actual
        assert uniform.sum().item() == 10 - actual
        assert (~teacher).sum().item() == 10
        assert set(row_values(batch)[focused].tolist()) <= focused_values
        assert set(row_values(batch)[uniform].tolist()) <= uniform_values
        assert row_values(batch)[focused].unique().numel() == actual


def test_empty_student_focus_pool_backfills_uniform_and_never_blocks():
    policy, uniform_values, _ = _student_split_sampling_policy(focused_rows=0)

    critic = policy._sample_balanced_q_batch()
    actor = policy._sample_actor_batch()

    for batch in (critic, actor):
        teacher = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
        assert (~teacher).sum().item() == 10
        assert not batch[FAILURE_PHASE_STUDENT_SOURCE_KEY].any()
        if "actions" in batch:
            values = batch["actions"][:, 0]
        else:
            values = batch["critic_observations"][:, 0] + 1.0
        assert set(values[~teacher].tolist()) <= uniform_values
        assert "successful_student_source" not in batch
    # The Q and Actor calls each backfilled the requested three focused rows.
    assert policy._failure_phase_student_uniform_fallback_rows == 6


def test_disabled_tvkd_target_delegates_to_exact_baseline_path(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=False,
        tvkd_lambda=0.25,
    )
    baseline_target = torch.tensor([[1.0]])
    baseline_metric = torch.tensor(2.0)
    baseline_log_prob = torch.tensor([3.0])
    sentinel = (
        baseline_target,
        {"baseline": baseline_metric},
        baseline_log_prob,
    )
    batch = {"unchanged": True, "rewards": torch.tensor([1.0, -3.0])}

    def baseline(self, batch):
        assert batch["unchanged"] is True
        assert torch.equal(batch["rewards"], torch.tensor([1.0, -3.0]))
        return sentinel

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_distributional_fastsac_target",
        baseline,
    )
    target, metrics, log_prob = policy._distributional_fastsac_target(batch)

    assert target is baseline_target
    assert log_prob is baseline_log_prob
    assert metrics["baseline"] is baseline_metric
    assert metrics["tvkd_potential_delta_mean"].item() == 0.0
    assert metrics["tvkd_potential_delta_std"].item() == 0.0
    assert metrics["tvkd_potential_delta_min"].item() == 0.0
    assert metrics["tvkd_potential_delta_max"].item() == 0.0
    assert metrics["tvkd_raw_reward_mean"].item() == pytest.approx(-1.0)
    assert metrics["tvkd_shaped_reward_mean"].item() == pytest.approx(-1.0)
    assert metrics["tvkd_shaped_reward_std"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("variant", ("baseline", "tvkd"))
def test_c51_support_metrics_preserve_both_q_heads_and_aggregate_head_totals(
    monkeypatch,
    variant,
):
    policy_type = (
        DistributionalFastSACTeacherBC
        if variant == "baseline"
        else TVKDDistributionalFastSACTeacherBC
    )
    policy = policy_type.__new__(policy_type)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        gamma=0.99,
        use_tvkd_value_shaping=variant == "tvkd",
        tvkd_lambda=0.25,
        tvkd_potential_clip=None,
    )
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype=torch.float32
    )
    policy.qnet_target = _TableTwin(probabilities, probabilities).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.tensor(-20.0))
    policy._fastsac_entropy_reference_log_scale_sum = 0.0
    policy.sac_action_rng = torch.Generator().manual_seed(17)
    policy._actor_dist_from_flat = lambda observations: _FixedDist(
        observations.shape[0]
    )
    policy._q_action_input = lambda actions: actions
    if variant == "tvkd":
        policy.get_frozen_teacher_value = lambda observations: torch.zeros(
            observations.shape[0], dtype=torch.float32
        )

    per_head = iter(((0.1, 0.2), (0.3, 0.4)))

    def distinct_projection(
        head_probability,
        rewards,
        bootstrap,
        effective_discount,
        support,
    ):
        del rewards, bootstrap, effective_discount, support
        left, right = next(per_head)
        return (
            head_probability,
            head_probability.new_tensor(left),
            head_probability.new_tensor(right),
        )

    target_module = fastsac_module if variant == "baseline" else tvkd_module
    monkeypatch.setattr(
        target_module, "_project_c51_probabilities", distinct_projection
    )
    batch = {
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        REPLAY_TERMINATED_KEY: torch.zeros(2, dtype=torch.bool),
        REPLAY_COMMAND_FINISHED_KEY: torch.zeros(2, dtype=torch.bool),
        REPLAY_TIME_LIMIT_KEY: torch.zeros(2, dtype=torch.bool),
        REPLAY_TEACHER_V_CURRENT_KEY: torch.zeros(2),
        REPLAY_TEACHER_V_NEXT_KEY: torch.zeros(2),
    }

    _, metrics, _ = policy._distributional_fastsac_target(batch)

    expected = {
        "q1_left_support_clip_fraction": 0.1,
        "q1_right_support_clip_fraction": 0.2,
        "q2_left_support_clip_fraction": 0.3,
        "q2_right_support_clip_fraction": 0.4,
        # Aggregate left+right within each head first: Q1=.3, Q2=.7.
        "support_clip_fraction_mean": 0.5,
        "support_clip_fraction_max": 0.7,
        # Historical aliases now mean the cross-head directional averages.
        "left_support_projection_clipping_fraction": 0.2,
        "right_support_projection_clipping_fraction": 0.3,
    }
    for name, value in expected.items():
        assert metrics[name].item() == pytest.approx(value)


def test_disabled_bottleneck_replay_never_calls_legacy_failure_phase_path(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(use_teacher_value_bottleneck_replay=False)

    def baseline(self, rollout):
        del self, rollout
        raise AssertionError("TVKD v5 must never call the parent failure path")

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_update_failure_phase_histogram",
        baseline,
    )
    assert policy._update_failure_phase_histogram(object()) == 0
    assert policy._verified_failure_motion_phase_histogram == {}


def test_tvkd_c51_target_inserts_shaped_reward_once_and_stays_detached():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=True,
        tvkd_lambda=0.25,
        tvkd_potential_clip=None,
        gamma=1.0,
    )
    first = torch.tensor([[0.05, 0.15, 0.80], [0.05, 0.15, 0.80], [0.05, 0.15, 0.80]])
    second = torch.tensor([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05], [0.80, 0.15, 0.05]])
    policy.qnet_target = _TableTwin(first, second).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2)))
    policy.sac_action_rng = torch.Generator().manual_seed(19)
    policy._actor_dist_from_flat = lambda observations: _FixedDist(
        observations.shape[0]
    )
    policy._normalized_action_log_prob = lambda value: value
    policy._q_action_input = lambda action: action
    policy.get_frozen_teacher_value = lambda observation: observation[:, 0]
    batch = {
        "observations": torch.zeros(3, 1),
        "next_observations": torch.zeros(3, 1),
        "critic_observations": torch.tensor([[2.0], [3.0], [6.0]]),
        "next_critic_observations": torch.tensor([[4.0], [5.0], [8.0]]),
        "rewards": torch.ones(3),
        "dones": torch.tensor([False, True, True]),
        "truncations": torch.tensor([False, False, True]),
        "discounts": torch.ones(3),
        REPLAY_TERMINATED_KEY: torch.tensor([False, True, False]),
        REPLAY_COMMAND_FINISHED_KEY: torch.zeros(3, dtype=torch.bool),
        REPLAY_TIME_LIMIT_KEY: torch.tensor([False, False, True]),
        # Cached Teacher values: get_frozen_teacher_value = lambda obs: obs[:, 0]
        REPLAY_TEACHER_V_CURRENT_KEY: torch.tensor([2.0, 3.0, 6.0]),
        REPLAY_TEACHER_V_NEXT_KEY: torch.tensor([4.0, 5.0, 8.0]),
    }
    raw_reward_before = batch["rewards"].clone()

    projected, metrics, _ = policy._distributional_fastsac_target(batch)
    # Shaping and the Q bootstrap share one c_t = [1, 0, 1], so the timeout row
    # bootstraps V_T(s_next) in both. Potential=[4-2, 0-3, 8-6].
    expected_shaped = torch.tensor([1.5, 0.25, 1.5])
    expected, _, _ = _project_c51_probabilities(
        second,
        expected_shaped,
        torch.tensor([1.0, 0.0, 1.0]),
        torch.ones(3),
        policy.qnet_target.support,
    )
    assert torch.allclose(projected, expected)
    assert metrics["tvkd_shaped_reward_mean"].item() == pytest.approx(13.0 / 12.0)
    assert metrics["tvkd_potential_delta_mean"].item() == pytest.approx(1.0 / 3.0)
    assert torch.equal(batch["rewards"], raw_reward_before)
    assert projected.requires_grad is False
    assert projected.grad_fn is None


def test_shaping_and_q_bootstrap_read_one_continuation_coefficient():
    """The shaped reward and the C51 bootstrap must scale by the same c_t.

    The two used to be derived independently and disagreed at command
    completion.  A soft ``discount`` below one exercises the full coefficient,
    not just its boundary mask.
    """
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    gamma = 0.9
    policy.cfg = SimpleNamespace(
        use_tvkd_value_shaping=True,
        tvkd_lambda=0.5,
        tvkd_potential_clip=None,
        gamma=gamma,
    )
    first = torch.tensor([[0.05, 0.15, 0.80]] * 4)
    second = torch.tensor([[0.80, 0.15, 0.05]] * 4)
    policy.qnet_target = _TableTwin(first, second).requires_grad_(False)
    policy.log_alpha = nn.Parameter(torch.log(torch.tensor(0.2)))
    policy.sac_action_rng = torch.Generator().manual_seed(19)
    policy._actor_dist_from_flat = lambda observations: _FixedDist(
        observations.shape[0]
    )
    policy._normalized_action_log_prob = lambda value: value
    policy._q_action_input = lambda action: action
    policy.get_frozen_teacher_value = lambda observation: observation[:, 0]

    # Rows: ordinary (soft discount), physical terminal, command completion,
    # pure time limit (soft discount).
    terminated = torch.tensor([False, True, False, False])
    command_finished = torch.tensor([False, False, True, False])
    time_limit = torch.tensor([False, False, False, True])
    discounts = torch.tensor([0.4, 1.0, 1.0, 0.5])
    teacher_v = torch.tensor([2.0, 3.0, 6.0, 7.0])
    # Cut rows carry the production zero sentinel for the unused next value.
    teacher_v_next = torch.tensor([4.0, 0.0, 0.0, 9.0])
    batch = {
        "observations": torch.zeros(4, 1),
        "next_observations": torch.zeros(4, 1),
        "critic_observations": teacher_v.unsqueeze(-1),
        "next_critic_observations": teacher_v_next.unsqueeze(-1),
        "rewards": torch.ones(4),
        "dones": terminated | command_finished | time_limit,
        "truncations": replay_truncation_mask(
            terminated=terminated,
            command_finished=command_finished,
            time_limit=time_limit,
        ),
        "discounts": discounts,
        REPLAY_TERMINATED_KEY: terminated,
        REPLAY_COMMAND_FINISHED_KEY: command_finished,
        REPLAY_TIME_LIMIT_KEY: time_limit,
        REPLAY_TEACHER_V_CURRENT_KEY: teacher_v,
        REPLAY_TEACHER_V_NEXT_KEY: teacher_v_next,
    }

    c_t = compute_continuation_coefficient(
        dones=batch["dones"],
        truncations=batch["truncations"],
        discounts=discounts,
    )
    assert torch.equal(c_t, torch.tensor([0.4, 0.0, 0.0, 0.5]))

    projected, metrics, _ = policy._distributional_fastsac_target(batch)
    expected_potential = gamma * c_t * teacher_v_next - teacher_v
    expected_shaped = torch.ones(4) + 0.5 * expected_potential
    # The very same c_t drives the bootstrap of the target distribution.
    expected, _, _ = _project_c51_probabilities(
        second,
        expected_shaped,
        c_t,
        torch.full((4,), gamma),
        policy.qnet_target.support,
    )
    assert torch.allclose(projected, expected)
    assert metrics["tvkd_potential_delta_mean"].item() == pytest.approx(
        expected_potential.mean().item(), abs=1e-6
    )


def test_short_terminal_only_episode_creates_no_failure_curriculum():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        failure_phase_lookback_steps=5,
        failure_phase_samples_per_failure=3,
        failure_phase_num_bins=10,
        bottleneck_fallback_mode="none",
    )
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._failure_phase_histogram = torch.zeros(10, dtype=torch.float64)
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0
    policy._failure_histogram_device_cache = {}
    history = {
        "phase": [0.8],
        "teacher_td_residual": [-20.0],
        "source_id": [SOURCE_STUDENT],
        "motion_id": [0],
        "true_terminal": [True],
        "timeout": [False],
        "replay_valid": [False],
    }

    assert policy._process_failed_student_episode(history) == 0
    assert policy._bottleneck_fallback_count == 0
    assert policy._bottleneck_no_candidate_count == 1
    assert policy._bottleneck_no_value_bottleneck_count == 1
    assert policy._last_bottleneck_metadata == {}
    assert policy._failure_phase_histogram.sum().item() == 0.0


def test_unsuccessful_episode_without_threshold_crossing_creates_no_focus():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        failure_phase_lookback_steps=5,
        failure_phase_samples_per_failure=3,
        failure_phase_num_bins=10,
        bottleneck_fallback_mode="none",
    )
    policy.teacher_value_bottleneck_detector = _detector(threshold=5.0)
    policy._reset_bottleneck_statistics()
    policy._reset_student_replay_episode_tracking()
    policy._failure_phase_histogram = torch.zeros(10, dtype=torch.float64)
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0
    policy._failure_histogram_device_cache = {}
    history = {
        "phase": [0.1, 0.2, 0.3, 0.4],
        "teacher_td_residual": [-0.1, -0.2, -0.1, -10.0],
        "source_id": [SOURCE_STUDENT] * 4,
        "motion_id": [0] * 4,
        "true_terminal": [False, False, False, True],
        "timeout": [False] * 4,
        "replay_valid": [True] * 4,
    }

    assert policy._process_failed_student_episode(history, replay_episode_id=7) == 0
    assert policy._bottleneck_no_value_bottleneck_count == 1
    assert policy._bottleneck_detected_count == 0
    assert policy._failure_phase_histogram.sum().item() == 0.0
    assert policy._verified_failure_motion_phase_histogram == {}
    assert policy._pending_student_focus_events == []


@pytest.mark.parametrize(
    "algorithm",
    (
        TVKD_LEGACY_TRAINING_ALGORITHM,
        TVKD_PREVIOUS_TRAINING_ALGORITHM,
        TVKD_V3_TRAINING_ALGORITHM,
        TVKD_V4_TRAINING_ALGORITHM,
        TVKD_V5_TRAINING_ALGORITHM,
        TVKD_TRAINING_ALGORITHM,
    ),
)
def test_tvkd_inference_forces_checkpoint_value_norm_before_construction(algorithm):
    cfg = OmegaConf.create({"algo": {"value_norm": False}})
    result = _fill_replayless_inference_algo_defaults(
        cfg,
        {
            "training_algorithm": algorithm,
            "dagger_backend_config": {"value_norm": True},
        },
        inference_only=True,
    )
    assert cfg.algo.value_norm is True
    assert "value_norm" in result["checkpoint"]


def test_tvkd_v6_version_identity_preserves_v5_for_safe_migration():
    assert TVKD_TRAINING_ALGORITHM == "distributional_tvkd_fastsac_teacher_bc_v6"
    assert TVKD_CHECKPOINT_VERSION == 6
    assert TVKD_V5_TRAINING_ALGORITHM == (
        "distributional_tvkd_fastsac_teacher_bc_v5"
    )
    assert TVKD_V5_CHECKPOINT_VERSION == 5
    assert TVKD_V4_TRAINING_ALGORITHM == ("distributional_tvkd_fastsac_teacher_bc_v4")
    assert TVKD_V4_CHECKPOINT_VERSION == 4
    assert TVKD_V3_TRAINING_ALGORITHM == ("distributional_tvkd_fastsac_teacher_bc_v3")
    assert TVKD_V3_CHECKPOINT_VERSION == 3


def test_tvkd_v4_training_resume_is_rejected_under_v5_sampler():
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    with pytest.raises(ValueError, match="v4 resume is incompatible"):
        policy._load_fastsac_checkpoint_state(
            {
                "training_algorithm": TVKD_V4_TRAINING_ALGORITHM,
                "checkpoint_version": TVKD_V4_CHECKPOINT_VERSION,
            },
            load_modules=False,
        )


@pytest.mark.parametrize(
    "algorithm", (TVKD_V5_TRAINING_ALGORITHM, TVKD_TRAINING_ALGORITHM)
)
def test_tvkd_v5_v6_inference_uses_the_replayless_model_only_loader(algorithm):
    calls = []

    class _InferencePolicy:
        def load_inference_state_dict(self, state, strict=True):
            calls.append((state["training_algorithm"], strict))
            return ["model-only"]

        def load_state_dict(self, state):
            del state
            raise AssertionError("TVKD inference must not enter the training loader")

    result = _load_policy_checkpoint(
        _InferencePolicy(),
        {"training_algorithm": algorithm},
        inference_only=True,
    )

    assert result == ["model-only"]
    assert calls == [(algorithm, True)]


def test_tvkd_logging_surface_is_finite_without_any_optimizer_update(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        lambda_bc=1.0,
        q_teacher_replay_ratio=0.1,
        teacher_actor_replay_fraction=0.6,
    )
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy.cfg.failure_phase_student_fraction = 0.3
    policy.dagger_replay = _TD3DeviceReplay(1, "cpu")
    policy._teacher_phase_index_ready = False
    policy._fastsac_rollout_critic_metrics = []
    policy._fastsac_rollout_actor_metrics = []
    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "train_op",
        lambda self, tensordict: {
            "fastsac/student_replay_rows_this_rollout": 0.0,
            "fastsac/student_source_fraction": 0.0,
            "fastsac/teacher_source_fraction": 1.0,
        },
    )

    info = policy.train_op(TensorDict({}, batch_size=[]))
    required = {
        "tvkd/teacher_value_mean",
        "tvkd/teacher_next_value_mean",
        "tvkd/potential_delta_mean",
        "tvkd/raw_reward_mean",
        "tvkd/shaped_reward_mean",
        "bottleneck/failed_student_episode_count",
        "bottleneck/student_candidate_count",
        "bottleneck/detected_count",
        "bottleneck/fallback_count",
        "bottleneck/no_candidate_count",
        "bottleneck/selected_step_mean",
        "bottleneck/selected_phase_mean",
        "bottleneck/score_mean",
        "bottleneck/score_max",
        "bottleneck/raw_td_residual_mean",
        "bottleneck/raw_teacher_td_residual_mean",
        "bottleneck/smoothed_td_residual_mean",
        "bottleneck/smoothed_teacher_td_residual_mean",
        "bottleneck/teacher_sequences_inserted",
        "bottleneck/teacher_transitions_inserted",
        "bottleneck/phase_match_distance_mean",
        "bottleneck/teacher_phase_match_distance",
        "bottleneck/failure_teacher_buffer_size",
        "bottleneck/student_focus_rows_marked",
        "bottleneck/student_focus_rows_missing",
        "bottleneck/student_focus_sampled_rows",
        "bottleneck/student_focus_uniform_fallback_rows",
        "bottleneck/student_focus_pool_size",
        "bottleneck/student_focus_fraction_config",
        "bottleneck/student_focus_q_global_fraction_cap",
        "bottleneck/student_focus_actor_global_fraction_cap",
        "loss/critic",
        "loss/actor_total",
        "loss/actor_sac",
        "loss/bc",
        "loss/fixed_bc_coefficient",
        "loss/alpha",
        "source/student_transition_count",
    }
    assert required.issubset(info)
    assert all(torch.isfinite(torch.as_tensor(info[key])) for key in required)
    assert info["bottleneck/student_focus_q_global_fraction_cap"] == pytest.approx(0.27)
    assert info["bottleneck/student_focus_actor_global_fraction_cap"] == pytest.approx(
        0.12
    )
    assert not any(key.startswith("bc_scheduler/") for key in info)


def test_tvkd_v6_checkpoint_saves_state_and_accepts_safe_v5_migration(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    policy.height_encoder = nn.Linear(1, 1)
    policy.cfg = TVKDDistributionalFastSACTeacherBCConfig()
    policy.cfg.teacher_value_reward_group_fingerprint = "a" * 64
    policy.cfg.replay_task_fingerprint = "b" * 64
    policy.cfg.failure_phase_num_bins = 4
    policy.cfg.train_dr_estimator = True
    policy.cfg.lambda_bc = 0.73
    policy.cfg.failure_phase_student_fraction = 0.3
    policy._replay_vecnorm_fingerprint = "checkpoint-vecnorm"
    policy.dr_estimator = nn.Linear(1, 1)
    policy.opt_dr_estimator = torch.optim.Adam(
        policy.dr_estimator.parameters(), lr=1e-3
    )
    policy.opt_dr_estimator.zero_grad(set_to_none=True)
    policy.dr_estimator(torch.ones(1, 1)).square().sum().backward()
    policy.opt_dr_estimator.step()
    policy.num_updates = 17
    policy.actor_update_count = 19
    policy.sac_actor_update_count = 19
    policy.alpha_update_count = 23
    policy.sac_alpha_update_count = 23
    policy._failure_phase_histogram = torch.tensor(
        [0.0, 1.0, 2.0, 0.0], dtype=torch.float64
    )
    policy._failure_phase_episode_count = 2
    policy._failure_phase_anchor_count = 3
    policy._failure_phase_uniform_fallback_rows = 4
    policy._failure_phase_focused_rows = 5
    policy._failure_histogram_device_cache = {}
    policy.teacher_value_bottleneck_detector = _detector()
    expected_detector = policy.teacher_value_bottleneck_detector.state_dict()
    policy._reset_bottleneck_statistics()
    policy._verified_failure_motion_phase_histogram = {
        7: policy._failure_phase_histogram.clone()
    }
    policy._bottleneck_detected_count = 2
    policy._bottleneck_selected_count = 2
    policy._bottleneck_score_sum = 3.0
    policy._bottleneck_score_max = 2.0
    policy._student_focus_rows_marked = 6
    policy._student_focus_rows_missing = 1
    policy._failure_phase_student_focused_rows = 9
    policy._failure_phase_student_uniform_fallback_rows = 2
    policy._last_tvkd_diagnostics = {"tvkd/shaped_reward_mean": 1.25}
    freeze_calls = []
    policy.teacher_value_wrapper = SimpleNamespace(
        freeze=lambda: freeze_calls.append(True)
    )

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_fastsac_checkpoint_state",
        lambda self: {
            "baseline_state": torch.tensor(1.0),
            "optimizer_resume_state": {},
            "dagger_backend_config": _strict_v5_backend_config(self.cfg),
        },
    )
    translated = []

    def baseline_load(self, state, *, load_modules=True):
        translated.append((state, load_modules))

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        baseline_load,
    )

    state = policy._fastsac_checkpoint_state()
    assert state["training_algorithm"] == TVKD_TRAINING_ALGORITHM
    assert state["checkpoint_version"] == TVKD_CHECKPOINT_VERSION
    assert state["actor_replay_observation_semantics"] == (
        tvkd_module.COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
    )
    assert state["teacher_episode_sidecar_semantics"] == (
        tvkd_module.TEACHER_EPISODE_SIDECAR_SEMANTICS
    )
    assert state["dagger_backend_config"]["lambda_bc"] == pytest.approx(0.73)
    assert state["dagger_backend_config"][
        "failure_phase_student_fraction"
    ] == pytest.approx(0.3)
    assert (
        state["teacher_value_bottleneck_replay_state"]["detector"] == expected_detector
    )
    focus_state = state["teacher_value_bottleneck_replay_state"]
    assert focus_state["student_focus_rows_marked"] == 6
    assert focus_state["student_focus_rows_missing"] == 1
    assert focus_state["student_focus_sampled_rows"] == 9
    assert focus_state["student_focus_uniform_fallback_rows"] == 2
    assert "teacher_value_bc_scheduler" not in state
    assert "student_bc_scheduler" not in state
    assert set(state["frozen_teacher_state"]) == {
        "actor",
        "encoder_priv",
        "critic",
        "value_norm",
        "height_encoder",
    }
    assert torch.equal(
        state["failure_phase_curriculum_state"]["histogram"],
        policy._failure_phase_histogram,
    )
    expected_dr_optimizer = state["optimizer_resume_state"]["dr_estimator_optimizer"]
    assert expected_dr_optimizer is not None
    assert state["num_updates"] == 17
    assert state["sac_actor_update_count"] == 19
    assert state["sac_alpha_update_count"] == 23

    policy.cfg.replay_task_fingerprint = tvkd_module.UNBOUND_CONTRACT_FINGERPRINT
    with pytest.raises(ValueError, match="bound lowercase SHA-256"):
        policy._fastsac_checkpoint_state()
    policy.cfg.replay_task_fingerprint = "b" * 64

    policy.teacher_value_bottleneck_detector.last_diagnostics["no_candidate"] = False
    policy._bottleneck_detected_count = 99
    policy._student_focus_rows_marked = 99
    policy._failure_phase_student_focused_rows = 99
    policy.opt_dr_estimator = torch.optim.Adam(
        policy.dr_estimator.parameters(), lr=0.25
    )
    policy._load_fastsac_checkpoint_state(state, load_modules=False)
    assert policy.teacher_value_bottleneck_detector.state_dict() == expected_detector
    assert (
        policy.teacher_value_bottleneck_detector.last_diagnostics["no_candidate"]
        is True
    )
    assert policy._bottleneck_detected_count == 2
    assert policy._student_focus_rows_marked == 6
    assert policy._student_focus_rows_missing == 1
    assert policy._failure_phase_student_focused_rows == 9
    assert policy._failure_phase_student_uniform_fallback_rows == 2
    assert policy.cfg.lambda_bc == pytest.approx(0.73)
    assert not hasattr(policy, "student_bc_scheduler")
    assert policy.num_updates == 17
    assert policy.sac_actor_update_count == 19
    assert policy.sac_alpha_update_count == 23
    restored_dr_optimizer = policy.opt_dr_estimator.state_dict()
    assert (
        restored_dr_optimizer["param_groups"] == expected_dr_optimizer["param_groups"]
    )
    for parameter_id, parameter_state in expected_dr_optimizer["state"].items():
        for name, expected in parameter_state.items():
            actual = restored_dr_optimizer["state"][parameter_id][name]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert freeze_calls == [True]
    assert translated[0][0]["training_algorithm"].startswith(
        "distributional_fastsac_teacher_bc"
    )
    assert translated[0][1] is False

    v5_state = copy.deepcopy(state)
    v5_state["training_algorithm"] = TVKD_V5_TRAINING_ALGORITHM
    v5_state["checkpoint_version"] = TVKD_V5_CHECKPOINT_VERSION
    v5_state["fresh_ring_resume_semantics"] = (
        tvkd_module.V5_FRESH_RING_RESUME_SEMANTICS
    )
    v5_state["replay_resume_semantics"] = tvkd_module.V5_REPLAY_RESUME_SEMANTICS
    v5_state["dagger_backend_config"]["method"] = TVKD_V5_TRAINING_ALGORITHM
    v5_state.pop("actor_replay_observation_semantics")
    v5_state.pop("teacher_episode_sidecar_semantics")
    with pytest.warns(UserWarning, match="TVKD v5 checkpoint to v6"):
        policy._load_fastsac_checkpoint_state(v5_state, load_modules=False)
    assert len(translated) == 2
    assert translated[-1][0]["checkpoint_version"] == (
        tvkd_module.BASE_FASTSAC_CHECKPOINT_VERSION
    )


def test_v5_checkpoint_rejects_binwise_motion_histogram_mismatch(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        failure_phase_num_bins=4,
        value_norm=False,
        bottleneck_fallback_mode="none",
    )
    policy._failure_histogram_device_cache = {}
    mismatched = _verified_histogram_state(
        4, histogram=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    mismatched["motion_histograms"] = {
        0: torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    }

    with pytest.raises(ValueError, match="motion/global"):
        policy._load_failure_curriculum_checkpoint_state(mismatched)

    policy._failure_phase_histogram = torch.zeros(4, dtype=torch.float64)
    policy._verified_failure_motion_phase_histogram = {}
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 7
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    with pytest.raises(RuntimeError, match="anchor count"):
        policy._failure_curriculum_checkpoint_state()

    policy._failure_phase_anchor_count = 0
    policy._verified_failure_motion_phase_histogram = {
        True: torch.zeros(4, dtype=torch.float64)
    }
    with pytest.raises(RuntimeError, match="motion histogram"):
        policy._failure_curriculum_checkpoint_state()

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_q_backend_metadata",
        lambda self: {"failure_phase_replay_semantics": "legacy_terminal_lookback"},
    )
    metadata = policy._q_backend_metadata()
    assert (
        metadata["failure_phase_replay_semantics"]
        == tvkd_module.VERIFIED_HISTOGRAM_SEMANTICS
    )
    assert (
        metadata["bottleneck_location_semantics"]
        == tvkd_module.BOTTLENECK_LOCATION_SEMANTICS
    )


def test_legacy_tvkd_resume_rejects_runtime_replay_mix_overrides(tmp_path):
    checkpoint_path = tmp_path / "legacy_tvkd.pt"
    torch.save(
        {
            "policy": {
                "training_algorithm": TVKD_V3_TRAINING_ALGORITHM,
                "checkpoint_version": TVKD_V3_CHECKPOINT_VERSION,
            },
            "vecnorm": {},
            "cfg": {},
        },
        checkpoint_path,
    )
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                f"fastsac_bc_dagger_checkpoint={checkpoint_path}",
                "fastsac_dagger_iterations=10",
            ],
        )

    with pytest.raises(ValueError, match="owned by the checkpoint migration"):
        _prepare_tvkd_checkpoint(
            cfg,
            explicit_replay_mix_fields={"q_teacher_replay_ratio"},
        )


def test_tvkd_resume_entrypoint_accepts_checkpoint_and_uses_additional_budget(
    tmp_path,
    monkeypatch,
):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "fastsac_dagger_iterations=100",
            ],
        )
    tvkd_entry._install_teacher_contract_fingerprints(cfg)
    saved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    launch_dir = tmp_path / "launch"
    checkpoint_path = launch_dir / "checkpoints" / "checkpoint_12.pt"
    checkpoint_path.parent.mkdir(parents=True)
    relative_checkpoint_path = Path("checkpoints") / checkpoint_path.name

    def from_hydra_launch_dir(path):
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = launch_dir / candidate
        return str(candidate)

    monkeypatch.setattr(
        tvkd_entry.hydra.utils,
        "to_absolute_path",
        from_hydra_launch_dir,
    )
    rng_state = torch.Generator().manual_seed(11).get_state()
    module_names = (
        "actor",
        "actor_adapt",
        "encoder_priv",
        "critic",
        "value_norm",
        "bc_dagger_sac_adapter",
        "qnet",
        "qnet_target",
        "adapt_module",
        "adapt_ema",
        "object_transform",
        "object_pred_transform",
        "object_adapt",
        "object_adapt_ema",
        "depth_cnn",
        "temporal_depth_gru",
        "temporal_depth_gru_ema",
    )
    histogram_state = _verified_histogram_state(int(cfg.algo.failure_phase_num_bins))
    policy_state = {name: {} for name in module_names}
    policy_state.update(
        {
            "training_algorithm": TVKD_TRAINING_ALGORITHM,
            "checkpoint_version": TVKD_CHECKPOINT_VERSION,
            "actor_backend": TVKD_ACTOR_BACKEND,
            "teacher_value_bottleneck_replay_state": (
                _empty_bottleneck_checkpoint_state()
            ),
            **_strict_v5_policy_metadata(
                cfg.algo,
                vecnorm_fingerprint="vecnorm",
                histogram_state=histogram_state,
            ),
            "frozen_teacher_state": {
                name: {} for name in ("actor", "encoder_priv", "critic", "value_norm")
            },
            "optimizer_resume_state": {
                "actor_optimizer": {},
                "critic_optimizer": {},
                "alpha_optimizer": {},
                "adapt_optimizer": {},
                "dr_estimator_optimizer": None,
            },
            "action_contract": {
                "joint_names": ("joint",),
                "fingerprint": "action-fingerprint",
            },
            "perception_initialization": {},
            "dagger_backend_config": _strict_v5_backend_config(cfg.algo),
            "vecnorm_fingerprint": "vecnorm",
            "log_alpha": torch.tensor(-2.0),
            "actor_update_count": 1,
            "critic_update_count": 2,
            "alpha_update_count": 3,
            "q_update_row_credit": 383.0,
            "dagger_rollout_count": 600,
            "dagger_environment_steps": 600 * int(cfg.algo.train_every),
            "teacher_prefill_rollout_count": 4,
            "teacher_prefill_environment_steps": 4 * int(cfg.algo.train_every),
            "num_updates": 600,
            "sac_actor_update_count": 1,
            "sac_alpha_update_count": 3,
            "dagger_rng_state": rng_state.clone(),
            "q_rng_state": rng_state.clone(),
            "sac_action_rng_state": rng_state.clone(),
            "sac_rollout_rng_state": rng_state.clone(),
            "teacher_perception_rng_state": rng_state.clone(),
            "last_phase": "finetune",
            "last_iter": 40,
            "next_iter": 41,
        }
    )
    torch.save(
        {
            "policy": policy_state,
            "vecnorm": {},
            "cfg": saved_cfg,
        },
        checkpoint_path,
    )
    cfg.fastsac_bc_dagger_checkpoint = str(relative_checkpoint_path)

    assert "teacher_value_bc_scheduler" not in policy_state
    result = _prepare_tvkd_checkpoint(cfg)
    validate_tvkd_fastsac_bc_dagger_config(cfg)

    canonical_checkpoint_path = str(checkpoint_path.resolve())
    assert result == {"path": canonical_checkpoint_path, "rollout_count": 600}
    assert cfg.fastsac_bc_dagger_checkpoint == canonical_checkpoint_path
    assert cfg.checkpoint_path == canonical_checkpoint_path
    assert cfg._tvkd_model_only_resume is True
    assert cfg._bc_dagger_fresh_source is True
    assert cfg.algo.value_norm is False

    v5_policy_state = copy.deepcopy(policy_state)
    v5_policy_state["training_algorithm"] = TVKD_V5_TRAINING_ALGORITHM
    v5_policy_state["checkpoint_version"] = TVKD_V5_CHECKPOINT_VERSION
    v5_policy_state["fresh_ring_resume_semantics"] = (
        tvkd_module.V5_FRESH_RING_RESUME_SEMANTICS
    )
    v5_policy_state["replay_resume_semantics"] = (
        tvkd_module.V5_REPLAY_RESUME_SEMANTICS
    )
    v5_policy_state["dagger_backend_config"]["method"] = (
        TVKD_V5_TRAINING_ALGORITHM
    )
    v5_policy_state.pop("actor_replay_observation_semantics")
    v5_policy_state.pop("teacher_episode_sidecar_semantics")
    v5_checkpoint_path = checkpoint_path.with_name("checkpoint_v5.pt")
    torch.save(
        {
            "policy": v5_policy_state,
            "vecnorm": {},
            "cfg": saved_cfg,
        },
        v5_checkpoint_path,
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        v5_cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "fastsac_dagger_iterations=100",
                f"fastsac_bc_dagger_checkpoint={v5_checkpoint_path}",
            ],
        )
    with pytest.warns(UserWarning, match="TVKD v5 checkpoint to v6"):
        v5_result = _prepare_tvkd_checkpoint(v5_cfg)
    assert v5_result == {
        "path": str(v5_checkpoint_path.resolve()),
        "rollout_count": 600,
    }

    mix_overrides = [
        "algo.q_teacher_replay_ratio=0.5",
        "algo.teacher_actor_replay_fraction=0.0",
        "algo.failure_phase_teacher_fraction=0.5",
        "algo.failure_phase_student_fraction=0.5",
    ]

    def save_mix_checkpoint(runtime_cfg, path):
        tvkd_entry._install_teacher_contract_fingerprints(runtime_cfg)
        state = copy.deepcopy(policy_state)
        state.update(
            _strict_v5_policy_metadata(
                runtime_cfg.algo,
                vecnorm_fingerprint="vecnorm",
                histogram_state=histogram_state,
            )
        )
        state["dagger_backend_config"] = _strict_v5_backend_config(runtime_cfg.algo)
        torch.save(
            {
                "policy": state,
                "vecnorm": {},
                "cfg": OmegaConf.create(
                    OmegaConf.to_container(runtime_cfg, resolve=False)
                ),
            },
            path,
        )

    def compose_mix_resume(path, replay_overrides=mix_overrides):
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            return compose(
                config_name="TVKD_fasSAC_bc_dagger",
                overrides=[
                    "task=G1/vaic/skateboard_stu",
                    f"fastsac_bc_dagger_checkpoint={path}",
                    "fastsac_dagger_iterations=100",
                    *replay_overrides,
                ],
                return_hydra_config=True,
            )

    derived_path = tmp_path / "checkpoint_tvkd_v5_derived_alias_mix.pt"
    derived_cfg = compose_mix_resume(derived_path)
    assert _apply_tvkd_cli_replay_mix_overrides(
        derived_cfg, derived_cfg.hydra.overrides.task
    )
    save_mix_checkpoint(derived_cfg, derived_path)
    derived_runtime = compose_mix_resume(derived_path)
    derived_fields = tvkd_entry._explicit_algo_override_fields(
        derived_runtime.hydra.overrides.task
    )
    derived_result = _prepare_tvkd_checkpoint(
        derived_runtime,
        explicit_replay_mix_fields=derived_fields,
        task_overrides=derived_runtime.hydra.overrides.task,
    )
    assert derived_result["path"] == str(derived_path.resolve())
    assert tuple(
        float(derived_runtime.algo[f"actor_{source}_fraction"])
        for source in REPLAY_SOURCE_ORDER
    ) == pytest.approx((0.5, 0.5, 0.0, 0.0))

    mixed_interfaces = (
        (
            "q_alias_actor_canonical",
            [
                "algo.q_teacher_replay_ratio=0.2",
                "algo.actor_uniform_student_fraction=0.2",
                "algo.actor_failure_student_fraction=0.3",
                "algo.actor_uniform_teacher_fraction=0.4",
                "algo.actor_failure_teacher_fraction=0.1",
            ],
            (0.56, 0.24, 0.14, 0.06),
            (0.2, 0.3, 0.4, 0.1),
        ),
        (
            "q_canonical_actor_alias",
            [
                "algo.q_uniform_student_fraction=0.4",
                "algo.q_failure_student_fraction=0.1",
                "algo.q_uniform_teacher_fraction=0.3",
                "algo.q_failure_teacher_fraction=0.2",
                "algo.teacher_actor_replay_fraction=0.2",
            ],
            (0.4, 0.1, 0.3, 0.2),
            (0.56, 0.24, 0.14, 0.06),
        ),
    )
    for name, overrides, expected_q, expected_actor in mixed_interfaces:
        mixed_path = tmp_path / f"checkpoint_tvkd_v5_{name}.pt"
        mixed_cfg = compose_mix_resume(mixed_path, overrides)
        assert _apply_tvkd_cli_replay_mix_overrides(
            mixed_cfg, mixed_cfg.hydra.overrides.task
        )
        save_mix_checkpoint(mixed_cfg, mixed_path)
        mixed_runtime = compose_mix_resume(mixed_path, overrides)
        mixed_fields = tvkd_entry._explicit_algo_override_fields(
            mixed_runtime.hydra.overrides.task
        )
        mixed_result = _prepare_tvkd_checkpoint(
            mixed_runtime,
            explicit_replay_mix_fields=mixed_fields,
            task_overrides=mixed_runtime.hydra.overrides.task,
        )
        assert mixed_result["path"] == str(mixed_path.resolve())
        assert tuple(
            float(mixed_runtime.algo[f"q_{source}_fraction"])
            for source in REPLAY_SOURCE_ORDER
        ) == pytest.approx(expected_q)
        assert tuple(
            float(mixed_runtime.algo[f"actor_{source}_fraction"])
            for source in REPLAY_SOURCE_ORDER
        ) == pytest.approx(expected_actor)

    pre_resolution_path = tmp_path / "checkpoint_tvkd_v5_old_alias_metadata.pt"
    pre_resolution_cfg = compose_mix_resume(pre_resolution_path)
    save_mix_checkpoint(pre_resolution_cfg, pre_resolution_path)
    pre_resolution_runtime = compose_mix_resume(pre_resolution_path)
    pre_resolution_fields = tvkd_entry._explicit_algo_override_fields(
        pre_resolution_runtime.hydra.overrides.task
    )
    with pytest.warns(UserWarning, match="predates CLI alias resolution"):
        pre_resolution_result = _prepare_tvkd_checkpoint(
            pre_resolution_runtime,
            explicit_replay_mix_fields=pre_resolution_fields,
            task_overrides=pre_resolution_runtime.hydra.overrides.task,
        )
    assert pre_resolution_result["path"] == str(pre_resolution_path.resolve())
    assert tuple(
        float(pre_resolution_runtime.algo[f"actor_{source}_fraction"])
        for source in REPLAY_SOURCE_ORDER
    ) == pytest.approx((0.35, 0.15, 0.35, 0.15))

    mismatched_path = tmp_path / "checkpoint_tvkd_v4_binwise_mismatch.pt"
    mismatched_policy = copy.deepcopy(policy_state)
    mismatched_histogram = _verified_histogram_state(
        int(cfg.algo.failure_phase_num_bins),
        histogram=torch.nn.functional.pad(
            torch.ones(1, dtype=torch.float64),
            (0, int(cfg.algo.failure_phase_num_bins) - 1),
        ),
    )
    mismatched_histogram["motion_histograms"] = {
        0: torch.nn.functional.pad(
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            (0, int(cfg.algo.failure_phase_num_bins) - 2),
        )
    }
    mismatched_policy["verified_teacher_value_histogram_state"] = copy.deepcopy(
        mismatched_histogram
    )
    mismatched_policy["failure_phase_curriculum_state"] = copy.deepcopy(
        mismatched_histogram
    )
    torch.save(
        {"policy": mismatched_policy, "vecnorm": {}, "cfg": saved_cfg},
        mismatched_path,
    )
    mismatched_runtime = OmegaConf.create(
        OmegaConf.to_container(saved_cfg, resolve=False)
    )
    mismatched_runtime.fastsac_bc_dagger_checkpoint = str(mismatched_path)
    with pytest.raises(ValueError, match="motion/global"):
        _prepare_tvkd_checkpoint(mismatched_runtime)

    alias_path = tmp_path / "checkpoint_tvkd_v4_alias_mismatch.pt"
    alias_policy = copy.deepcopy(policy_state)
    alias_policy["failure_phase_curriculum_state"]["episode_count"] = 1
    torch.save(
        {"policy": alias_policy, "vecnorm": {}, "cfg": saved_cfg},
        alias_path,
    )
    alias_runtime = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
    alias_runtime.fastsac_bc_dagger_checkpoint = str(alias_path)
    with pytest.raises(ValueError, match="histogram aliases"):
        _prepare_tvkd_checkpoint(alias_runtime)

    drifted = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    drifted.fastsac_bc_dagger_checkpoint = str(checkpoint_path)
    drifted.algo.train_every = int(drifted.algo.train_every) + 1
    with pytest.raises(ValueError, match="algorithm config"):
        _prepare_tvkd_checkpoint(drifted)

    malformed_v3_path = tmp_path / "checkpoint_tvkd_v3_missing_student_fraction.pt"
    malformed_v3_policy = copy.deepcopy(policy_state)
    malformed_v3_policy["training_algorithm"] = TVKD_V3_TRAINING_ALGORITHM
    malformed_v3_policy["checkpoint_version"] = TVKD_V3_CHECKPOINT_VERSION
    malformed_v3_policy["teacher_value_bottleneck_replay_state"] = (
        _legacy_bottleneck_checkpoint_state(
            {
                "bottleneck_residual_scale_ema": 1.0,
                "num_scale_updates": 0,
            }
        )
    )
    malformed_v3_policy["dagger_backend_config"] = _legacy_replay_backend(
        float(cfg.algo.lambda_bc),
        student_focus=None,
    )
    malformed_v3_cfg = _legacy_saved_config(saved_cfg, include_student_focus=True)
    torch.save(
        {
            "policy": malformed_v3_policy,
            "vecnorm": {},
            "cfg": malformed_v3_cfg,
        },
        malformed_v3_path,
    )
    malformed_v3_runtime = OmegaConf.create(
        OmegaConf.to_container(saved_cfg, resolve=False)
    )
    malformed_v3_runtime.fastsac_bc_dagger_checkpoint = str(malformed_v3_path)

    with pytest.raises(ValueError, match="failure_phase_student_fraction"):
        _prepare_tvkd_checkpoint(malformed_v3_runtime)

    previous_path = tmp_path / "checkpoint_tvkd_v2.pt"
    previous_cfg = _legacy_saved_config(saved_cfg, include_student_focus=False)
    previous_policy = copy.deepcopy(policy_state)
    previous_policy["training_algorithm"] = TVKD_PREVIOUS_TRAINING_ALGORITHM
    previous_policy["checkpoint_version"] = TVKD_PREVIOUS_CHECKPOINT_VERSION
    previous_policy["teacher_value_bottleneck_replay_state"] = (
        _legacy_bottleneck_checkpoint_state(
            {
                "bottleneck_residual_scale_ema": 1.0,
                "num_scale_updates": 0,
            }
        )
    )
    previous_policy["dagger_backend_config"] = _legacy_replay_backend(
        float(cfg.algo.lambda_bc),
        student_focus=None,
    )
    previous_focus = previous_policy["teacher_value_bottleneck_replay_state"]
    for name in (
        "student_focus_rows_marked",
        "student_focus_rows_missing",
        "student_focus_sampled_rows",
        "student_focus_uniform_fallback_rows",
    ):
        previous_focus.pop(name)
    torch.save(
        {"policy": previous_policy, "vecnorm": {}, "cfg": previous_cfg},
        previous_path,
    )
    previous_runtime = OmegaConf.create(
        OmegaConf.to_container(saved_cfg, resolve=False)
    )
    previous_runtime.fastsac_bc_dagger_checkpoint = str(previous_path)

    with pytest.warns(UserWarning, match="v2 checkpoints predate focused Student"):
        previous_result = _prepare_tvkd_checkpoint(previous_runtime)

    assert previous_result == {
        "path": str(previous_path),
        "rollout_count": 600,
    }
    assert previous_runtime.algo.failure_phase_student_fraction == pytest.approx(0.0)


def test_direct_v2_resume_forces_uniform_student_replay_and_zeros_new_counters(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_dr_estimator=False,
        failure_phase_num_bins=2,
        lambda_bc=0.61,
        failure_phase_student_fraction=0.3,
    )
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._failure_phase_histogram = torch.zeros(2, dtype=torch.float64)
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._failure_histogram_device_cache = {}
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(True))

    bottleneck_state = _empty_bottleneck_checkpoint_state()
    for name in (
        "student_focus_rows_marked",
        "student_focus_rows_missing",
        "student_focus_sampled_rows",
        "student_focus_uniform_fallback_rows",
    ):
        bottleneck_state.pop(name)
    state = {
        "training_algorithm": TVKD_PREVIOUS_TRAINING_ALGORITHM,
        "checkpoint_version": TVKD_PREVIOUS_CHECKPOINT_VERSION,
        "dagger_backend_config": _legacy_replay_backend(
            0.61,
            student_focus=None,
        ),
        "teacher_value_bottleneck_replay_state": bottleneck_state,
        "frozen_teacher_state": {
            name: getattr(policy, name).state_dict()
            for name in ("actor", "encoder_priv", "critic", "value_norm")
        },
        "failure_phase_curriculum_state": {
            "histogram": torch.zeros(2, dtype=torch.float64),
            "episode_count": 0,
            "anchor_count": 0,
            "uniform_fallback_rows": 0,
            "focused_rows": 0,
        },
        "optimizer_resume_state": {"dr_estimator_optimizer": None},
        "num_updates": 7,
        "sac_actor_update_count": 8,
        "sac_alpha_update_count": 9,
    }
    translated = []
    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        lambda self, state, *, load_modules=True: translated.append(
            (state, load_modules)
        ),
    )

    with pytest.warns(UserWarning, match="Migrating a TVKD v2"):
        policy._load_fastsac_checkpoint_state(state, load_modules=False)

    assert policy.cfg.failure_phase_student_fraction == pytest.approx(0.0)
    assert policy._student_focus_rows_marked == 0
    assert policy._student_focus_rows_missing == 0
    assert policy._failure_phase_student_focused_rows == 0
    assert policy._failure_phase_student_uniform_fallback_rows == 0
    assert translated and translated[0][1] is False
    assert freezes == [True]


def test_direct_v3_resume_preserves_cadence_and_exact_legacy_mix_but_resets_stats(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_dr_estimator=False,
        failure_phase_num_bins=4,
        lambda_bc=0.61,
        failure_phase_student_fraction=0.3,
    )
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._bottleneck_detected_count = 9
    policy._failure_phase_histogram = torch.tensor(
        [0.0, 2.0, 1.0, 0.0], dtype=torch.float64
    )
    policy._failure_phase_episode_count = 2
    policy._failure_phase_anchor_count = 3
    policy._failure_phase_uniform_fallback_rows = 4
    policy._failure_phase_focused_rows = 5
    policy._verified_failure_motion_phase_histogram = {
        3: policy._failure_phase_histogram.clone()
    }
    policy._failure_histogram_device_cache = {}
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(True))
    state = {
        "training_algorithm": TVKD_V3_TRAINING_ALGORITHM,
        "checkpoint_version": TVKD_V3_CHECKPOINT_VERSION,
        "dagger_backend_config": _legacy_replay_backend(
            0.61,
            student_focus=0.3,
            alpha_cadence="critic",
        ),
        "teacher_value_bottleneck_replay_state": (
            _legacy_bottleneck_checkpoint_state(
                {
                    "bottleneck_residual_scale_ema": 6.0,
                    "num_scale_updates": 41,
                }
            )
        ),
        "frozen_teacher_state": {
            name: getattr(policy, name).state_dict()
            for name in ("actor", "encoder_priv", "critic", "value_norm")
        },
        "failure_phase_curriculum_state": {
            "histogram": torch.tensor([0.0, 2.0, 1.0, 0.0], dtype=torch.float64),
            "episode_count": 2,
            "anchor_count": 3,
            "uniform_fallback_rows": 4,
            "focused_rows": 5,
        },
        "optimizer_resume_state": {"dr_estimator_optimizer": None},
        "num_updates": 7,
        "sac_actor_update_count": 8,
        "sac_alpha_update_count": 9,
    }
    translated = []
    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        lambda self, state, *, load_modules=True: translated.append(
            (state, load_modules)
        ),
    )

    with pytest.warns(UserWarning, match="Migrating a TVKD v3"):
        policy._load_fastsac_checkpoint_state(state, load_modules=False)

    expected_global = {
        "uniform_student": 0.35,
        "failure_student": 0.15,
        "uniform_teacher": 0.35,
        "failure_teacher": 0.15,
    }
    for purpose in ("q", "actor"):
        for source, expected in expected_global.items():
            assert getattr(policy.cfg, f"{purpose}_{source}_fraction") == pytest.approx(
                expected
            )
    expected_perception = {
        "uniform_student": 1.0,
        "failure_student": 0.0,
        "uniform_teacher": 0.0,
        "failure_teacher": 0.0,
    }
    for source, expected in expected_perception.items():
        assert getattr(policy.cfg, f"perception_{source}_fraction") == pytest.approx(
            expected
        )
    assert policy.cfg.perception_replay_mode == "online_student_rollout"
    assert policy.cfg.teacher_perception_replay_fraction == 0.0
    assert policy.cfg.teacher_perception_warmup_steps == 0
    assert policy.cfg.sac_alpha_update_cadence == "critic"
    assert policy.cfg.bottleneck_fallback_mode == "none"
    assert policy.cfg.bottleneck_include_unsuccessful_timeouts is False
    assert policy.cfg.max_teacher_phase_match_distance is None
    assert policy.teacher_value_bottleneck_detector.state_dict() == {}
    assert torch.equal(
        policy._failure_phase_histogram,
        torch.zeros(4, dtype=torch.float64),
    )
    assert policy._verified_failure_motion_phase_histogram == {}
    assert policy._bottleneck_detected_count == 0
    assert (policy.num_updates, policy.sac_actor_update_count) == (7, 8)
    assert policy.sac_alpha_update_count == 9
    assert translated and translated[0][1] is False
    assert freezes == [True]


@pytest.mark.parametrize(
    ("algorithm", "version"),
    (
        (TVKD_TRAINING_ALGORITHM, TVKD_CHECKPOINT_VERSION),
        (TVKD_V5_TRAINING_ALGORITHM, TVKD_V5_CHECKPOINT_VERSION),
    ),
)
def test_public_tvkd_resume_rebuilds_rings_and_teacher_sidecars(
    monkeypatch, algorithm, version
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy._fastsac_action_contract = {
        "joint_names": ("joint",),
        "fingerprint": "same",
    }
    policy.cfg = SimpleNamespace(
        train_perception=True,
        q_update_to_data_ratio=1.0,
        q_batch_size=512,
    )
    policy._q_actor_dim = 1
    stale_episode_store = object()
    stale_actor_cache = object()
    policy._teacher_episode_store = stale_episode_store
    policy._teacher_actor_cache = stale_actor_cache
    policy._teacher_prefill_raw_pending = [[{"stale": torch.tensor(1.0)}]]
    policy._perception_ema_generation = 17
    policy._teacher_ring_cache_lineage = object()
    policy._teacher_episode_device_raw_fields = {
        "stale": torch.tensor([1.0])
    }
    policy._teacher_episode_device_raw_lineage = object()
    restored = []

    def restore(self, state, *, load_modules=True):
        restored.append((state, load_modules))
        self.q_update_row_credit = float(state["q_update_row_credit"])

    monkeypatch.setattr(
        TVKDDistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        restore,
    )

    class _Ring:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    policy.dagger_replay = _Ring()
    policy.q_teacher_replay = _Ring()
    policy.actor_adapt = Actor(1)
    policy.actor_adapt(torch.zeros(1, 1))
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1, initial_log_std=torch.tensor(-1.0), device="cpu"
    )
    policy.qnet = nn.Linear(1, 1)
    policy.qnet_target = nn.Linear(1, 1)
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(1))
    policy._teacher_phase_device_cache = {}
    policy._set_perception_trainable = lambda trainable: None
    progress = []
    policy.env = SimpleNamespace(set_progress=lambda value: progress.append(value))

    result = policy.load_state_dict(
        {
            "training_algorithm": algorithm,
            "checkpoint_version": version,
            "actor_backend": TVKD_ACTOR_BACKEND,
            "action_contract": {
                "joint_names": ("joint",),
                "fingerprint": "same",
            },
            "last_iter": 40,
            "next_iter": 41,
            "q_update_row_credit": 383.0,
        }
    )

    assert result == []
    assert restored and restored[0][1] is True
    assert policy.dagger_replay.cleared == 1
    assert policy.q_teacher_replay.cleared == 1
    assert policy._teacher_episode_store is not stale_episode_store
    assert policy._teacher_actor_cache is not stale_actor_cache
    assert policy._teacher_episode_store.episode_count == 0
    assert policy._teacher_actor_cache.ready is False
    assert policy._teacher_actor_cache.actor_dim == 1
    assert policy._teacher_prefill_raw_pending is None
    assert policy._perception_ema_generation == 0
    assert policy._teacher_ring_cache_lineage is None
    assert policy._teacher_episode_device_raw_fields is None
    assert policy._teacher_episode_device_raw_lineage is None
    assert policy._teacher_prefill_complete is False
    assert policy.teacher_prefill_rollout_count == 0
    assert policy.actor_adapt.training is True
    assert policy.actor_adapt.actor_std.requires_grad is False
    assert policy.actor_adapt.actor_std.grad is None
    assert policy.qnet.training is True
    assert policy.qnet_target.training is False
    assert policy.q_update_row_credit == 0.0
    assert policy._q_updates_due(129) == 0
    assert policy.q_update_row_credit == pytest.approx(129.0)
    assert progress == [41]
    assert freezes == [1]


def test_private_resume_configures_frozen_perception_before_optimizer_load(
    monkeypatch,
):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = TVKDDistributionalFastSACTeacherBCConfig()
    policy.cfg.teacher_value_reward_group_fingerprint = "a" * 64
    policy.cfg.replay_task_fingerprint = "b" * 64
    policy.cfg.train_perception = False
    policy.cfg.train_dr_estimator = False
    policy.cfg.failure_phase_num_bins = 2
    policy.cfg.lambda_bc = 0.61
    policy.cfg.failure_phase_student_fraction = 0.3
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    perception_parameter = nn.Parameter(torch.tensor(1.0))
    original_optimizer = torch.optim.Adam([perception_parameter], lr=1e-3)
    policy.opt_adapt = original_optimizer
    policy._perception_optimizer = original_optimizer
    policy._perception_initialization = {}
    policy._replay_vecnorm_fingerprint = "same-vecnorm"
    policy._failure_histogram_device_cache = {}
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(1))
    histogram_state = _verified_histogram_state(2)
    state = {
        "training_algorithm": TVKD_TRAINING_ALGORITHM,
        "checkpoint_version": TVKD_CHECKPOINT_VERSION,
        "dagger_backend_config": _strict_v5_backend_config(policy.cfg),
        "teacher_value_bottleneck_replay_state": (
            policy._bottleneck_replay_checkpoint_state()
        ),
        **_strict_v5_policy_metadata(
            policy.cfg,
            vecnorm_fingerprint="same-vecnorm",
            histogram_state=histogram_state,
        ),
        "frozen_teacher_state": {
            name: getattr(policy, name).state_dict()
            for name in ("actor", "encoder_priv", "critic", "value_norm")
        },
        "vecnorm_fingerprint": "same-vecnorm",
        "perception_initialization": {"trainable": False},
        "optimizer_resume_state": {"dr_estimator_optimizer": None},
        "num_updates": 0,
        "sac_actor_update_count": 0,
        "sac_alpha_update_count": 0,
        "actor": policy.actor.state_dict(),
        "encoder_priv": policy.encoder_priv.state_dict(),
        "critic": policy.critic.state_dict(),
        "value_norm": policy.value_norm.state_dict(),
    }
    monkeypatch.setattr(PPOVEL, "load_state_dict", lambda self, state, strict: [])
    observed = []

    def baseline_load(self, state, *, load_modules=True):
        observed.append((self.opt_adapt, load_modules))

    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        baseline_load,
    )

    drifted = copy.deepcopy(state)
    drifted["dagger_backend_config"]["tvkd_lambda"] = (
        float(policy.cfg.tvkd_lambda) + 0.1
    )
    with pytest.raises(ValueError, match="tvkd_lambda"):
        policy._load_fastsac_checkpoint_state(drifted, load_modules=False)

    drifted_cadence = copy.deepcopy(state)
    drifted_cadence["dagger_backend_config"]["train_every"] = (
        int(policy.cfg.train_every) + 1
    )
    with pytest.raises(ValueError, match="train_every"):
        policy._load_fastsac_checkpoint_state(drifted_cadence, load_modules=False)

    drifted_semantics = copy.deepcopy(state)
    drifted_semantics["actor_learning_semantics"] = "wrong"
    with pytest.raises(ValueError, match="actor_learning_semantics"):
        policy._load_fastsac_checkpoint_state(drifted_semantics, load_modules=False)

    policy._load_fastsac_checkpoint_state(state, load_modules=True)

    assert observed == [(None, True)]
    assert policy.opt_adapt is None
    assert policy._perception_optimizer is original_optimizer
    assert policy._perception_initialization["trainable"] is False
    assert not hasattr(policy, "student_bc_scheduler")
    assert freezes == [1]


def test_v1_resume_discards_scheduler_state_and_starts_fresh_detector(monkeypatch):
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_dr_estimator=False,
        failure_phase_num_bins=2,
        lambda_bc=0.44,
    )
    policy.actor = nn.Linear(1, 1)
    policy.encoder_priv = nn.Linear(1, 1)
    policy.critic = nn.Linear(1, 2)
    policy.value_norm = _AffineValueNorm()
    policy.teacher_value_bottleneck_detector = _detector()
    policy._reset_bottleneck_statistics()
    policy._failure_phase_histogram = torch.tensor([2.0, 1.0], dtype=torch.float64)
    policy._failure_phase_episode_count = 2
    policy._failure_phase_anchor_count = 3
    policy._failure_phase_uniform_fallback_rows = 4
    policy._failure_phase_focused_rows = 5
    policy._failure_histogram_device_cache = {}
    freezes = []
    policy.teacher_value_wrapper = SimpleNamespace(freeze=lambda: freezes.append(True))
    state = {
        "training_algorithm": TVKD_LEGACY_TRAINING_ALGORITHM,
        "checkpoint_version": TVKD_LEGACY_CHECKPOINT_VERSION,
        "dagger_backend_config": _legacy_replay_backend(
            0.44,
            student_focus=None,
        ),
        "teacher_value_bc_scheduler": {
            "residual_scale_ema": 99.0,
            "risk_ema": 1.0,
            "num_updates": 500,
            "current_lambda_bc_student": 0.05,
        },
        "frozen_teacher_state": {
            name: getattr(policy, name).state_dict()
            for name in ("actor", "encoder_priv", "critic", "value_norm")
        },
        "failure_phase_curriculum_state": {
            "histogram": torch.tensor([2.0, 1.0], dtype=torch.float64),
            "episode_count": 2,
            "anchor_count": 3,
            "uniform_fallback_rows": 4,
            "focused_rows": 5,
        },
        "optimizer_resume_state": {"dr_estimator_optimizer": None},
        "num_updates": 7,
        "sac_actor_update_count": 8,
        "sac_alpha_update_count": 9,
        "last_tvkd_diagnostics": {
            "tvkd/raw_reward_mean": 1.0,
            "bc_scheduler/risk_ema": 0.9,
        },
    }
    translated = []
    monkeypatch.setattr(
        DistributionalFastSACTeacherBC,
        "_load_fastsac_checkpoint_state",
        lambda self, state, *, load_modules=True: translated.append(
            (state, load_modules)
        ),
    )

    with pytest.warns(UserWarning, match="Migrating a TVKD v1"):
        policy._load_fastsac_checkpoint_state(state, load_modules=False)

    assert translated and translated[0][1] is False
    assert policy.teacher_value_bottleneck_detector.state_dict() == {}
    assert torch.equal(
        policy._failure_phase_histogram,
        torch.zeros(2, dtype=torch.float64),
    )
    assert policy._failure_phase_anchor_count == 0
    assert policy.cfg.lambda_bc == pytest.approx(0.44)
    assert policy._last_tvkd_diagnostics == {"tvkd/raw_reward_mean": 1.0}
    assert not hasattr(policy, "student_bc_scheduler")
    assert freezes == [True]


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    (
        (1, (1, 0, 0, 0)),
        (3, (1, 1, 1, 0)),
        (7, (3, 1, 2, 1)),
        (128, (45, 19, 45, 19)),
        (512, (179, 77, 179, 77)),
        (4096, (1434, 614, 1434, 614)),
    ),
)
def test_four_way_allocator_is_exact_and_deterministic_at_required_sizes(
    batch_size,
    expected,
):
    fractions = {
        "uniform_student": 0.35,
        "failure_student": 0.15,
        "uniform_teacher": 0.35,
        "failure_teacher": 0.15,
    }
    counts = allocate_source_counts(batch_size, fractions)
    reverse_counts = allocate_source_counts(
        batch_size, dict(reversed(tuple(fractions.items())))
    )

    assert tuple(counts[source] for source in REPLAY_SOURCE_ORDER) == expected
    assert counts == reverse_counts
    assert sum(counts.values()) == batch_size
    assert all(count >= 0 for count in counts.values())


@pytest.mark.parametrize("batch_size", (0, -1, 1.5, True))
def test_four_way_allocator_rejects_invalid_batch_sizes(batch_size):
    fractions = dict(zip(REPLAY_SOURCE_ORDER, (0.35, 0.15, 0.35, 0.15)))
    with pytest.raises(ValueError, match="batch_size"):
        allocate_source_counts(batch_size, fractions)


@pytest.mark.parametrize(
    "fractions",
    (
        {
            "uniform_student": -0.01,
            "failure_student": 0.16,
            "uniform_teacher": 0.35,
            "failure_teacher": 0.5,
        },
        {
            "uniform_student": float("nan"),
            "failure_student": 0.15,
            "uniform_teacher": 0.35,
            "failure_teacher": 0.5,
        },
        {
            "uniform_student": float("inf"),
            "failure_student": 0.15,
            "uniform_teacher": 0.35,
            "failure_teacher": 0.5,
        },
        {
            "uniform_student": True,
            "failure_student": 0.15,
            "uniform_teacher": 0.35,
            "failure_teacher": 0.5,
        },
        {
            "uniform_student": 0.3,
            "failure_student": 0.15,
            "uniform_teacher": 0.35,
            "failure_teacher": 0.15,
        },
        {
            "uniform_student": 0.5,
            "failure_student": 0.5,
            "uniform_teacher": 0.0,
        },
        {
            "uniform_student": 0.5,
            "failure_student": 0.5,
            "uniform_teacher": 0.0,
            "failure_teacher": 0.0,
            "unexpected": 0.0,
        },
    ),
)
def test_four_way_allocator_rejects_invalid_fraction_contracts(fractions):
    with pytest.raises(ValueError):
        allocate_source_counts(128, fractions)


def test_four_way_allocator_requires_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        allocate_source_counts(128, (0.35, 0.15, 0.35, 0.15))


def test_tvkd_hydra_config_inherits_source_mix_defaults_and_new_controls():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
            ],
            return_hydra_config=True,
        )

    assert not _apply_tvkd_cli_replay_mix_overrides(
        cfg, cfg.hydra.overrides.task
    )

    assert cfg.algo.name == "tvkd_fastsac_bc_dagger"
    assert cfg.algo.sac_alpha_update_cadence == "actor"
    assert cfg.algo.use_tvkd_value_shaping is True
    assert cfg.algo.tvkd_lambda == pytest.approx(0.25)
    assert cfg.algo.use_teacher_value_bottleneck_replay is True
    assert cfg.algo.bottleneck_threshold == pytest.approx(0.05)
    assert cfg.algo.bottleneck_smoothing_window == 5
    assert cfg.algo.bottleneck_min_consecutive == 3
    assert cfg.algo.bottleneck_terminal_exclusion_steps == 5
    assert cfg.algo.failure_phase_lookback_steps == 10
    assert cfg.algo.failure_phase_samples_per_failure == 10
    assert "bottleneck_residual_scale_ema_decay" not in cfg.algo
    assert "bottleneck_eps" not in cfg.algo
    assert set(LEGACY_ADAPTIVE_BC_CONFIG_FIELDS).isdisjoint(cfg.algo.keys())
    assert cfg.algo.teacher_actor_replay_fraction == pytest.approx(0.5)
    assert cfg.algo.failure_phase_teacher_fraction == pytest.approx(0.3)
    assert cfg.algo.failure_phase_student_fraction == pytest.approx(0.3)
    assert cfg.algo.q_teacher_replay_ratio == pytest.approx(0.5)
    assert cfg.algo.teacher_perception_replay_fraction == pytest.approx(0.0)
    assert cfg.algo.teacher_perception_warmup_steps == 0
    assert cfg.algo.q_updates_per_rollout == 32
    expected_mix = {
        "uniform_student": 0.35,
        "failure_student": 0.15,
        "uniform_teacher": 0.35,
        "failure_teacher": 0.15,
    }
    for purpose in ("q", "actor"):
        for source, expected in expected_mix.items():
            assert cfg.algo[f"{purpose}_{source}_fraction"] == pytest.approx(expected)
    perception_mix = {
        "uniform_student": 1.0,
        "failure_student": 0.0,
        "uniform_teacher": 0.0,
        "failure_teacher": 0.0,
    }
    for source, expected in perception_mix.items():
        assert cfg.algo[f"perception_{source}_fraction"] == pytest.approx(expected)
    assert cfg.algo.perception_replay_mode == "online_student_rollout"
    assert cfg.algo.bottleneck_fallback_mode == "none"
    assert cfg.algo.bottleneck_include_unsuccessful_timeouts is False
    assert cfg.algo.max_teacher_phase_match_distance is None

    cfg.algo.q_failure_student_fraction = 1.01
    with pytest.raises(ValueError, match="failure_student fraction"):
        validate_tvkd_fastsac_bc_dagger_config(cfg)
    cfg.algo.q_failure_student_fraction = 0.15
    cfg.algo.sac_alpha_update_cadence = "rollout"
    with pytest.raises(ValueError, match="sac_alpha_update_cadence"):
        validate_tvkd_fastsac_bc_dagger_config(cfg)
    cfg.algo.sac_alpha_update_cadence = "actor"
    cfg.algo.failure_phase_samples_per_failure = 11
    with pytest.raises(ValueError, match="strictly pre-onset"):
        validate_tvkd_fastsac_bc_dagger_config(cfg)


def test_tvkd_hydra_resolves_independent_q_and_actor_four_way_mixes():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        "task=G1/vaic/skateboard_stu",
        "checkpoint_path=/tmp/fresh_ppo.pt",
        "fastsac_dagger_iterations=10",
        "algo.q_uniform_student_fraction=0.40",
        "algo.q_failure_student_fraction=0.10",
        "algo.q_uniform_teacher_fraction=0.30",
        "algo.q_failure_teacher_fraction=0.20",
        "algo.actor_uniform_student_fraction=0.20",
        "algo.actor_failure_student_fraction=0.30",
        "algo.actor_uniform_teacher_fraction=0.40",
        "algo.actor_failure_teacher_fraction=0.10",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=overrides,
            return_hydra_config=True,
        )

    assert not _apply_tvkd_cli_replay_mix_overrides(
        cfg, cfg.hydra.overrides.task
    )

    expected = {
        "q": (0.40, 0.10, 0.30, 0.20),
        "actor": (0.20, 0.30, 0.40, 0.10),
    }
    for purpose, values in expected.items():
        resolved = tuple(
            float(cfg.algo[f"{purpose}_{source}_fraction"])
            for source in REPLAY_SOURCE_ORDER
        )
        assert resolved == pytest.approx(values)
        assert sum(resolved) == pytest.approx(1.0)
    tvkd_module._validate_tvkd_algorithm_config(cfg.algo)
    assert tuple(
        float(cfg.algo[f"perception_{source}_fraction"])
        for source in REPLAY_SOURCE_ORDER
    ) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_tvkd_hydra_legacy_cli_mix_translates_current_command():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        "task=G1/vaic/skateboard_stu",
        "checkpoint_path=/tmp/fresh_ppo.pt",
        "fastsac_dagger_iterations=14000",
        "algo.q_teacher_replay_ratio=0.5",
        "algo.teacher_actor_replay_fraction=0.0",
        "algo.teacher_perception_replay_fraction=0.0",
        "algo.failure_phase_teacher_fraction=0.5",
        "algo.failure_phase_student_fraction=0.5",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=overrides,
            return_hydra_config=True,
        )

    assert _apply_tvkd_cli_replay_mix_overrides(
        cfg, cfg.hydra.overrides.task
    )
    expected = {
        "q": (0.25, 0.25, 0.25, 0.25),
        "actor": (0.5, 0.5, 0.0, 0.0),
        "perception": (1.0, 0.0, 0.0, 0.0),
    }
    for purpose, values in expected.items():
        assert tuple(
            float(cfg.algo[f"{purpose}_{source}_fraction"])
            for source in REPLAY_SOURCE_ORDER
        ) == pytest.approx(values)

    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = cfg.algo
    assert policy._replay_source_counts("q", 512) == {
        "uniform_student": 128,
        "failure_student": 128,
        "uniform_teacher": 128,
        "failure_teacher": 128,
    }
    assert policy._replay_source_counts("actor", 4096) == {
        "uniform_student": 2048,
        "failure_student": 2048,
        "uniform_teacher": 0,
        "failure_teacher": 0,
    }


def test_tvkd_main_resolves_cli_mix_before_training(monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
                "algo.q_teacher_replay_ratio=0.5",
                "algo.teacher_actor_replay_fraction=0.0",
                "algo.failure_phase_teacher_fraction=0.5",
                "algo.failure_phase_student_fraction=0.5",
            ],
            return_hydra_config=True,
        )

    monkeypatch.setattr(
        tvkd_entry,
        "HydraConfig",
        SimpleNamespace(initialized=lambda: True, get=lambda: cfg.hydra),
    )
    monkeypatch.setattr(tvkd_entry, "_require_single_process_execution", lambda: None)
    monkeypatch.setattr(
        tvkd_entry, "apply_fastsac_dagger_iteration_controls", lambda runtime: None
    )
    monkeypatch.setattr(tvkd_entry, "_prepare_tvkd_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(tvkd_entry, "_prepare_tvkd_fresh_source", lambda runtime: None)
    monkeypatch.setattr(
        tvkd_entry, "validate_tvkd_fastsac_bc_dagger_config", lambda runtime: None
    )
    monkeypatch.setattr(
        tvkd_entry,
        "fastsac_dagger_rollout_schedule",
        lambda runtime: {
            "prefill_target_rows": 1,
            "main_rollouts": 10,
            "frames_per_rollout": 1,
        },
    )
    captured = {}

    def capture(runtime):
        captured.update(
            {
                purpose: tuple(
                    float(runtime.algo[f"{purpose}_{source}_fraction"])
                    for source in REPLAY_SOURCE_ORDER
                )
                for purpose in ("q", "actor", "perception")
            }
        )
        return "trained"

    monkeypatch.setattr(tvkd_entry, "run_training", capture)

    assert tvkd_entry.main.__wrapped__(cfg) == "trained"
    assert captured["q"] == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert captured["actor"] == pytest.approx((0.5, 0.5, 0.0, 0.0))
    assert captured["perception"] == pytest.approx((1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("override", "derived_purpose", "derived", "unchanged_purpose"),
    (
        (
            "algo.q_teacher_replay_ratio=0.2",
            "q",
            (0.56, 0.24, 0.14, 0.06),
            "actor",
        ),
        (
            "algo.teacher_actor_replay_fraction=0.0",
            "actor",
            (0.7, 0.3, 0.0, 0.0),
            "q",
        ),
    ),
)
def test_tvkd_hydra_total_teacher_aliases_control_q_and_actor_independently(
    override,
    derived_purpose,
    derived,
    unchanged_purpose,
):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
                override,
            ],
            return_hydra_config=True,
        )

    assert _apply_tvkd_cli_replay_mix_overrides(
        cfg, cfg.hydra.overrides.task
    )
    assert tuple(
        float(cfg.algo[f"{derived_purpose}_{source}_fraction"])
        for source in REPLAY_SOURCE_ORDER
    ) == pytest.approx(derived)
    assert tuple(
        float(cfg.algo[f"{unchanged_purpose}_{source}_fraction"])
        for source in REPLAY_SOURCE_ORDER
    ) == pytest.approx((0.35, 0.15, 0.35, 0.15))


def test_tvkd_current_resume_reapplies_the_same_alias_resolution():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    mix_overrides = [
        "algo.q_teacher_replay_ratio=0.5",
        "algo.teacher_actor_replay_fraction=0.0",
        "algo.failure_phase_teacher_fraction=0.5",
        "algo.failure_phase_student_fraction=0.5",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        fresh = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
                *mix_overrides,
            ],
            return_hydra_config=True,
        )
        resumed = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "fastsac_bc_dagger_checkpoint=/tmp/tvkd.pt",
                "fastsac_dagger_iterations=10",
                *mix_overrides,
            ],
            return_hydra_config=True,
        )

    for cfg in (fresh, resumed):
        assert _apply_tvkd_cli_replay_mix_overrides(
            cfg, cfg.hydra.overrides.task
        )
    assert OmegaConf.to_container(fresh.algo, resolve=True) == OmegaConf.to_container(
        resumed.algo, resolve=True
    )


def test_tvkd_fresh_alias_validation_does_not_report_a_legacy_checkpoint():
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
                "algo.q_teacher_replay_ratio=-0.1",
            ],
            return_hydra_config=True,
        )

    with pytest.raises(ValueError, match=r"CLI replay mix.*q_teacher.*\[0, 1\]"):
        _apply_tvkd_cli_replay_mix_overrides(
            cfg, cfg.hydra.overrides.task
        )


@pytest.mark.parametrize(
    ("legacy_override", "canonical_override"),
    (
        (
            "algo.q_teacher_replay_ratio=0.5",
            "algo.q_uniform_student_fraction=0.35",
        ),
        (
            "algo.teacher_actor_replay_fraction=0.0",
            "algo.actor_failure_student_fraction=0.15",
        ),
        (
            "algo.failure_phase_student_fraction=0.5",
            "algo.actor_uniform_teacher_fraction=0.35",
        ),
    ),
)
def test_tvkd_hydra_rejects_mixed_alias_and_canonical_mix_overrides(
    legacy_override,
    canonical_override,
):
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                "checkpoint_path=/tmp/fresh_ppo.pt",
                "fastsac_dagger_iterations=10",
                legacy_override,
                canonical_override,
            ],
            return_hydra_config=True,
        )

    with pytest.raises(ValueError, match="cannot combine.*canonical"):
        _apply_tvkd_cli_replay_mix_overrides(
            cfg, cfg.hydra.overrides.task
        )


@pytest.mark.parametrize("fraction", (0.0, 0.1, 0.5, 1.0))
def test_tvkd_fresh_configurable_teacher_sources_survive_hydra_run_chdir(
    tmp_path,
    monkeypatch,
    fraction,
):
    launch_dir = tmp_path / "launch"
    run_dir = tmp_path / "outputs" / "hydra-run"
    checkpoint_path = launch_dir / "checkpoints" / "ppo_train.pt"
    checkpoint_path.parent.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "policy": {
                "last_phase": "train",
                **{name: {} for name in PPOVEL_TRAIN_PHASE_PARTIAL_PERCEPTION_MODULES},
            }
        },
        checkpoint_path,
    )
    relative_checkpoint_path = Path("checkpoints") / checkpoint_path.name
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="TVKD_fasSAC_bc_dagger",
            overrides=[
                "task=G1/vaic/skateboard_stu",
                f"checkpoint_path={relative_checkpoint_path}",
                "algo.load_pretrained_perception=true",
                f"algo.perception_checkpoint_path={relative_checkpoint_path}",
                "algo.train_perception=true",
                f"algo.teacher_actor_replay_fraction={fraction}",
                f"algo.q_teacher_replay_ratio={fraction}",
                "fastsac_dagger_iterations=14000",
            ],
            return_hydra_config=True,
        )

    monkeypatch.chdir(launch_dir)
    assert _apply_tvkd_cli_replay_mix_overrides(
        cfg, cfg.hydra.overrides.task
    )
    validate_tvkd_fastsac_bc_dagger_config(cfg)
    assert cfg.algo.teacher_actor_replay_fraction == pytest.approx(fraction)
    assert cfg.algo.teacher_perception_replay_fraction == pytest.approx(0.0)
    assert cfg.algo.q_teacher_replay_ratio == pytest.approx(fraction)
    expected = (
        0.7 * (1.0 - fraction),
        0.3 * (1.0 - fraction),
        0.7 * fraction,
        0.3 * fraction,
    )
    for purpose in ("q", "actor"):
        assert tuple(
            float(cfg.algo[f"{purpose}_{source}_fraction"])
            for source in REPLAY_SOURCE_ORDER
        ) == pytest.approx(expected)

    canonical_checkpoint_path = checkpoint_path.resolve()
    assert Path(cfg.algo.perception_checkpoint_path) == canonical_checkpoint_path

    # Policy construction happens after Hydra moves into its per-run output
    # directory. The propagated path must remain independent of that cwd.
    monkeypatch.chdir(run_dir)
    assert (
        Path(cfg.algo.perception_checkpoint_path).resolve(strict=True)
        == canonical_checkpoint_path
    )


def _v4_replay_rows(
    count: int,
    offset: int,
    *,
    phases: torch.Tensor | None = None,
    motion_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Small canonical replay payload with a complete recurrent raw window."""
    row = torch.arange(count, dtype=torch.float32) + float(offset)
    window = torch.arange(10, dtype=torch.float32)
    if phases is None:
        phases = (torch.arange(count, dtype=torch.float32) + 0.5) / count
    if motion_ids is None:
        motion_ids = torch.zeros(count, dtype=torch.long)
    return {
        "critic_observations": torch.stack((row, row + 0.5), dim=-1),
        "actions": row[:, None] + 1.0,
        "rewards": row + 2.0,
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.ones(count),
        "next_critic_observations": torch.stack((row + 4.0, row + 4.5), dim=-1),
        REFERENCE_PHASE_KEY: phases.clone(),
        REPLAY_TERMINATED_KEY: torch.zeros(count, dtype=torch.bool),
        REPLAY_COMMAND_FINISHED_KEY: torch.zeros(count, dtype=torch.bool),
        REPLAY_TIME_LIMIT_KEY: torch.zeros(count, dtype=torch.bool),
        REPLAY_MOTION_ID_KEY: motion_ids.clone().long(),
        PERCEPTION_DEPTH_U8_KEY: torch.zeros(count, 10, 1, 1, 1, dtype=torch.uint8),
        PERCEPTION_POLICY_RAW_KEY: row[:, None, None] + window[None, :, None],
        PERCEPTION_VEL_COMMAND_RAW_KEY: (
            row[:, None, None] + window[None, :, None] + 0.25
        ),
        PERCEPTION_IS_INIT_KEY: torch.zeros(count, 10, dtype=torch.bool),
    }


def _v4_replay_mix_config() -> SimpleNamespace:
    fractions = {
        "q": (0.40, 0.10, 0.30, 0.20),
        "actor": (0.20, 0.30, 0.40, 0.10),
        "perception": (0.25, 0.25, 0.25, 0.25),
    }
    fields = {
        f"{purpose}_{source}_fraction": value
        for purpose, values in fractions.items()
        for source, value in zip(REPLAY_SOURCE_ORDER, values)
    }
    return SimpleNamespace(
        **fields,
        q_batch_size=20,
        dagger_batch_size=20,
        perception_replay_batch_size=20,
        teacher_perception_batch_size=20,
        perception_replay_mode="four_way",
        teacher_perception_replay_fraction=0.0,
        failure_phase_num_bins=10,
        max_teacher_phase_match_distance=None,
        policy_delay=1,
    )


def _v4_four_way_sampling_policy(
    *,
    student_focus_rows: int = 12,
    teacher_phases: torch.Tensor | None = None,
    teacher_motion_ids: torch.Tensor | None = None,
    verified_motion_bin: tuple[int, int] = (0, 5),
    seed: int = 1701,
) -> TVKDDistributionalFastSACTeacherBC:
    policy = TVKDDistributionalFastSACTeacherBC.__new__(
        TVKDDistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = _v4_replay_mix_config()
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(seed)
    policy.teacher_perception_rng = torch.Generator().manual_seed(seed + 1)
    policy.critic_update_count = 0
    policy.q_teacher_replay = _TD3DeviceReplay(128, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(128, "cpu")

    teacher_count = 40 if teacher_phases is None else int(teacher_phases.numel())
    teacher = _v4_replay_rows(
        teacher_count,
        100,
        phases=teacher_phases,
        motion_ids=teacher_motion_ids,
    )
    policy.q_teacher_replay.extend(teacher)

    student_count = max(40, student_focus_rows)
    student = _v4_replay_rows(student_count, 1_000)
    student[DAGGER_IS_STUDENT_ACTION_KEY] = torch.ones(student_count, dtype=torch.bool)
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(student_count, dtype=torch.float32)[:, None] + 2_000.0
    )
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(
        student_count, dtype=torch.bool
    )
    focused = torch.zeros(student_count, dtype=torch.bool)
    if student_focus_rows:
        focused[-student_focus_rows:] = True
    student[FAILURE_PHASE_STUDENT_SOURCE_KEY] = focused
    policy.dagger_replay.extend(student)

    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )
    motion, phase_bin = verified_motion_bin
    histogram = torch.zeros(10, dtype=torch.float64)
    histogram[phase_bin] = 1.0
    policy._verified_failure_motion_phase_histogram = {motion: histogram}
    policy._teacher_phase_device_cache = {}
    policy._teacher_phase_index_ready = False
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._failure_phase_student_uniform_fallback_rows = 0
    policy._failure_phase_student_focused_rows = 0
    policy._build_teacher_phase_index()
    return policy


def _v4_provenance_counts(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    provenance = batch[REPLAY_SAMPLE_PROVENANCE_KEY]
    return {
        source: int((provenance == source_id).sum().item())
        for source_id, source in enumerate(REPLAY_SOURCE_ORDER)
    }


def _legacy_teacher_phase_match_result(
    policy: TVKDDistributionalFastSACTeacherBC,
    anchor_phases: torch.Tensor,
    anchor_motion_ids: torch.Tensor,
) -> tuple[float, int]:
    """Reference the former full-ring scan exactly, including accumulation."""
    replay = policy.q_teacher_replay
    teacher_phase = replay.data[REFERENCE_PHASE_KEY][: replay.size]
    teacher_phase = teacher_phase.reshape(replay.size, -1)[:, 0].float().cpu()
    teacher_bins = policy._teacher_replay_phase_bins
    teacher_motion_ids = policy._teacher_replay_motion_ids
    bins = int(policy.cfg.failure_phase_num_bins)
    max_distance = policy.cfg.max_teacher_phase_match_distance
    total = 0.0
    count = 0
    for target, motion_id in zip(
        anchor_phases.detach().float().cpu().reshape(-1),
        anchor_motion_ids.detach().long().cpu().reshape(-1),
        strict=True,
    ):
        same_motion = (
            (teacher_motion_ids == int(motion_id))
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        if same_motion.numel() == 0:
            continue
        risk_bin = min(max(int(float(target) * bins), 0), bins - 1)
        bin_distance = (teacher_bins.index_select(0, same_motion) - risk_bin).abs()
        nearest_distance = int(bin_distance.min().item())
        if (
            max_distance is not None
            and nearest_distance / float(bins) > float(max_distance)
        ):
            continue
        rows = same_motion[bin_distance == nearest_distance]
        distance = (teacher_phase.index_select(0, rows) - target).abs().mean()
        total += float(distance.item())
        count += 1
    return total, count


def _focus_pool_match_policy(seed: int) -> TVKDDistributionalFastSACTeacherBC:
    teacher_phases = torch.tensor(
        [
            0.11,
            0.51,
            0.31,
            0.19,
            0.59,
            0.39,
            0.05,
            0.95,
            0.71,
            0.21,
            0.72,
            0.42,
        ]
    )
    teacher_motion_ids = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1, 2, 2, 0, 2])
    policy = _v4_four_way_sampling_policy(
        student_focus_rows=12,
        teacher_phases=teacher_phases,
        teacher_motion_ids=teacher_motion_ids,
        seed=seed,
    )
    histograms = {
        7: torch.tensor([0, 0, 9, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64),
        2: torch.tensor([0, 0, 0, 0, 0, 5, 0, 0, 0, 7], dtype=torch.float64),
        1: torch.tensor([0, 0, 0, 0, 4, 0, 0, 6, 0, 0], dtype=torch.float64),
        0: torch.tensor([0, 0, 2, 0, 3, 0, 0, 0, 0, 5], dtype=torch.float64),
    }
    policy._verified_failure_motion_phase_histogram = histograms
    policy.cfg.max_teacher_phase_match_distance = 0.15
    policy._verified_teacher_focus_pool_cache = None
    policy._verified_teacher_focus_device_cache = {}
    return policy


def _legacy_verified_teacher_focus_pool(
    policy: TVKDDistributionalFastSACTeacherBC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the former per-motion, per-bin full-ring scans exactly."""
    bin_count = int(policy.cfg.failure_phase_num_bins)
    max_distance = policy.cfg.max_teacher_phase_match_distance
    row_weights = torch.zeros(policy.q_teacher_replay.size, dtype=torch.float64)
    teacher_motion = policy._teacher_replay_motion_ids
    teacher_bins = policy._teacher_replay_phase_bins
    for raw_motion, raw_histogram in sorted(
        policy._verified_failure_motion_phase_histogram.items(),
        key=lambda item: int(item[0]),
    ):
        motion = int(raw_motion)
        histogram = raw_histogram.detach().to(device="cpu", dtype=torch.float64)
        same_motion = (
            (teacher_motion == motion).nonzero(as_tuple=False).squeeze(-1)
        )
        if same_motion.numel() == 0:
            continue
        motion_bins = teacher_bins.index_select(0, same_motion)
        for risk_bin in (
            (histogram > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        ):
            distances = (motion_bins - int(risk_bin)).abs()
            nearest_distance = int(distances.min().item())
            if (
                max_distance is not None
                and nearest_distance / float(bin_count) > float(max_distance)
            ):
                continue
            pool = same_motion[distances == nearest_distance]
            row_weights[pool] += float(histogram[risk_bin]) / int(pool.numel())
    rows = (row_weights > 0).nonzero(as_tuple=False).squeeze(-1)
    return rows, row_weights.index_select(0, rows)


def test_verified_teacher_focus_pool_cached_matcher_is_exact_to_legacy_scan():
    policy = _focus_pool_match_policy(seed=1901)

    expected_rows, expected_weights = _legacy_verified_teacher_focus_pool(policy)
    actual_rows, actual_weights = policy._verified_teacher_focus_pool()

    assert torch.equal(actual_rows, expected_rows)
    assert torch.equal(actual_weights, expected_weights)
    # Risk-bin 2 ties motion-0's occupied bins 1 and 3. The cached union must
    # retain the old mask's ascending replay-row order.
    nearest_distance, tied_rows = policy._teacher_phase_match_nearest_pool_cache[
        (0, 2)
    ]
    assert nearest_distance == 1
    assert torch.equal(tied_rows, torch.tensor([0, 2, 3, 5]))


def test_cached_focus_pool_preserves_prefetched_plans_counters_and_rng_exactly():
    legacy_scan = _focus_pool_match_policy(seed=1907)
    cached_matcher = _focus_pool_match_policy(seed=1907)
    legacy_scan._teacher_phase_match_pool = MethodType(
        DistributionalTD3TeacherBC._teacher_phase_match_pool,
        legacy_scan,
    )

    expected_plans = legacy_scan._prefetch_curriculum_sample_plans(4)
    actual_plans = cached_matcher._prefetch_curriculum_sample_plans(4)

    tensor_fields = (
        "teacher_indices",
        "student_indices",
        "permutation",
        "actor_indices",
        "actor_teacher_indices",
        "teacher_focused",
        "actor_teacher_focused",
        "student_focused",
        "actor_student_focused",
    )
    assert len(actual_plans) == len(expected_plans)
    for expected, actual in zip(expected_plans, actual_plans, strict=True):
        for field in tensor_fields:
            expected_value = getattr(expected, field)
            actual_value = getattr(actual, field)
            assert (expected_value is None) == (actual_value is None), field
            if expected_value is not None:
                assert torch.equal(expected_value, actual_value), field
    assert torch.equal(legacy_scan.q_rng.get_state(), cached_matcher.q_rng.get_state())
    for counter in (
        "_failure_phase_focused_rows",
        "_failure_phase_uniform_fallback_rows",
        "_failure_phase_student_focused_rows",
        "_failure_phase_student_uniform_fallback_rows",
    ):
        assert getattr(legacy_scan, counter) == getattr(cached_matcher, counter)


def test_teacher_phase_match_index_is_exact_for_motions_ties_missing_and_gate():
    # Motion-0's nearest bins around risk bin 2 are tied. Their replay rows are
    # deliberately interleaved so concatenating in phase-bin order would be
    # observably different from the legacy ascending-row mask.
    teacher_phases = torch.tensor(
        [0.11, 0.51, 0.31, 0.19, 0.59, 0.39, 0.05, 0.95]
    )
    teacher_motion_ids = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1])
    policy = _v4_four_way_sampling_policy(
        student_focus_rows=0,
        teacher_phases=teacher_phases,
        teacher_motion_ids=teacher_motion_ids,
    )
    anchors = torch.tensor([0.25, 0.52, 0.88, 0.20])
    motions = torch.tensor([0, 1, 1, 7])

    expected = _legacy_teacher_phase_match_result(policy, anchors, motions)
    policy._bottleneck_phase_match_distance_sum = 0.0
    policy._bottleneck_phase_match_distance_count = 0
    policy._record_teacher_phase_match_distances(anchors, motions)

    assert policy._bottleneck_phase_match_distance_sum == expected[0]
    assert policy._bottleneck_phase_match_distance_count == expected[1]
    nearest_distance, tied_rows = policy._teacher_phase_match_nearest_pool_cache[
        (0, 2)
    ]
    assert nearest_distance == 1
    assert torch.equal(tied_rows, torch.tensor([0, 2, 3, 5]))
    assert (7, 2) not in policy._teacher_phase_match_nearest_pool_cache

    # Keep exact-bin motion-1 anchor 0.52, while both one-bin-away anchors are
    # rejected by the strict normalized-distance gate.
    policy.cfg.max_teacher_phase_match_distance = 0.05
    expected = _legacy_teacher_phase_match_result(policy, anchors, motions)
    policy._bottleneck_phase_match_distance_sum = 0.0
    policy._bottleneck_phase_match_distance_count = 0
    policy._record_teacher_phase_match_distances(anchors, motions)
    assert policy._bottleneck_phase_match_distance_sum == expected[0]
    assert policy._bottleneck_phase_match_distance_count == expected[1] == 1


def test_teacher_phase_match_index_rebuild_invalidates_lazy_pools():
    teacher_phases = torch.tensor([0.11, 0.31, 0.51, 0.71])
    teacher_motion_ids = torch.tensor([0, 0, 1, 1])
    policy = _v4_four_way_sampling_policy(
        student_focus_rows=0,
        teacher_phases=teacher_phases,
        teacher_motion_ids=teacher_motion_ids,
    )
    anchor = torch.tensor([0.25])
    motion = torch.tensor([0])
    policy._bottleneck_phase_match_distance_sum = 0.0
    policy._bottleneck_phase_match_distance_count = 0
    policy._record_teacher_phase_match_distances(anchor, motion)
    old_source = policy._teacher_phase_match_index_source
    assert (0, 2) in policy._teacher_phase_match_nearest_pool_cache

    # Rebuild the immutable Teacher index from changed replay metadata, as a
    # restore/refill boundary would. The lazy nearest-pool cache must not retain
    # rows from the prior source tensors.
    policy.q_teacher_replay.data[REFERENCE_PHASE_KEY][:4].copy_(
        torch.tensor([0.81, 0.21, 0.61, 0.41])
    )
    policy.q_teacher_replay.data[REPLAY_MOTION_ID_KEY][:4].copy_(
        torch.tensor([1, 0, 1, 1])
    )
    policy._build_teacher_phase_index()
    assert policy._teacher_phase_match_index_source[0] is not old_source[0]
    assert policy._teacher_phase_match_index_source[1] is not old_source[1]
    assert policy._teacher_phase_match_nearest_pool_cache == {}

    expected = _legacy_teacher_phase_match_result(policy, anchor, motion)
    policy._bottleneck_phase_match_distance_sum = 0.0
    policy._bottleneck_phase_match_distance_count = 0
    policy._record_teacher_phase_match_distances(anchor, motion)
    assert policy._bottleneck_phase_match_distance_sum == expected[0]
    assert policy._bottleneck_phase_match_distance_count == expected[1] == 1
    _, rebuilt_rows = policy._teacher_phase_match_nearest_pool_cache[(0, 2)]
    assert torch.equal(rebuilt_rows, torch.tensor([1]))


def test_v4_replay_q_uses_exact_independent_40_10_30_20_provenance():
    policy = _v4_four_way_sampling_policy()

    batch = policy._sample_balanced_q_batch()

    assert _v4_provenance_counts(batch) == {
        "uniform_student": 8,
        "failure_student": 2,
        "uniform_teacher": 6,
        "failure_teacher": 4,
    }
    assert int(batch[DAGGER_Q_TEACHER_SOURCE_KEY].sum()) == 10


def test_v4_replay_actor_uses_20_30_40_10_without_q_mix_leakage():
    policy = _v4_four_way_sampling_policy()

    q_batch = policy._sample_balanced_q_batch()
    actor_batch = policy._sample_actor_batch()

    assert _v4_provenance_counts(q_batch) == {
        "uniform_student": 8,
        "failure_student": 2,
        "uniform_teacher": 6,
        "failure_teacher": 4,
    }
    assert _v4_provenance_counts(actor_batch) == {
        "uniform_student": 4,
        "failure_student": 6,
        "uniform_teacher": 8,
        "failure_teacher": 2,
    }


def test_v4_replay_shortage_backfills_only_within_student_and_teacher():
    teacher_phases = torch.full((40,), 0.95)
    teacher_phases[0] = 0.05
    policy = _v4_four_way_sampling_policy(
        student_focus_rows=1,
        teacher_phases=teacher_phases,
        verified_motion_bin=(0, 0),
    )

    batch = policy._sample_balanced_q_batch()

    assert _v4_provenance_counts(batch) == {
        "uniform_student": 9,
        "failure_student": 1,
        "uniform_teacher": 9,
        "failure_teacher": 1,
    }
    assert int(batch[DAGGER_Q_TEACHER_SOURCE_KEY].sum()) == 10
    metrics = policy._replay_mix_metrics("q")
    assert metrics["failure_student_backfill_rows"] == 1.0
    assert metrics["failure_teacher_backfill_rows"] == 3.0


def test_v4_replay_failure_teacher_never_crosses_motion_and_honors_distance():
    phases = torch.tensor([0.45, 0.45, 0.45, 0.45, 0.05, 0.05])
    motions = torch.tensor([0, 0, 1, 1, 1, 1])
    policy = _v4_four_way_sampling_policy(
        teacher_phases=phases,
        teacher_motion_ids=motions,
        verified_motion_bin=(0, 4),
    )

    indices, focused = policy._draw_teacher_indices(4, policy.q_rng, focused_count=2)
    sampled_motion = policy.q_teacher_replay.data[REPLAY_MOTION_ID_KEY][indices]
    assert focused.sum().item() == 2
    assert torch.equal(sampled_motion[focused], torch.zeros(2, dtype=torch.long))

    # Motion 1 has an exact bin-0 row, but the verified anchor belongs to
    # motion 0, whose closest row is four bins away. The distance gate must
    # backfill both requested FT rows as uniform Teacher rather than crossing.
    policy.cfg.max_teacher_phase_match_distance = 0.1
    histogram = torch.zeros(10, dtype=torch.float64)
    histogram[0] = 1.0
    policy._verified_failure_motion_phase_histogram = {0: histogram}
    _, focused = policy._draw_teacher_indices(4, policy.q_rng, focused_count=2)
    assert not focused.any()


class _V4ToyTemporalDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.seen_policy_windows: list[torch.Tensor] = []

    def forward(self, td: TensorDict):
        policy = td["policy"]
        self.seen_policy_windows.append(policy.detach().clone())
        td["_depth_feature"] = policy[..., :1] * self.scale
        return td


class _V4ToyPerceptionHead(nn.Module):
    def __init__(self, output_key: str):
        super().__init__()
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, td: TensorDict):
        td[self.output_key] = td["_depth_feature"] * self.scale
        return td


class _V4ToyPrivilegedEncoder(nn.Module):
    def forward(self, td: TensorDict):
        td["priv_feature"] = td["priv"]
        return td


def test_v4_replay_perception_uses_failure_student_raw_windows_not_live_rollout(
    monkeypatch,
):
    policy = _v4_four_way_sampling_policy(student_focus_rows=2)
    for source in REPLAY_SOURCE_ORDER:
        setattr(policy.cfg, f"perception_{source}_fraction", 0.0)
    policy.cfg.perception_failure_student_fraction = 1.0
    policy.cfg.perception_replay_batch_size = 2
    policy.cfg.teacher_perception_batch_size = 2
    policy.cfg.perception_replay_burn_in = 8
    policy.cfg.latent_dim = 1
    policy.cfg.num_minibatches = 1
    policy.cfg.max_grad_norm = 10.0
    policy.cfg.train_perception = True
    policy.cfg.train_dr_estimator = False

    raw_windows = torch.stack(
        (
            torch.linspace(0.1, 1.0, 10),
            torch.linspace(1.1, 2.0, 10),
        )
    ).unsqueeze(-1)
    focused_slice = slice(policy.dagger_replay.size - 2, policy.dagger_replay.size)
    policy.dagger_replay.data[PERCEPTION_POLICY_RAW_KEY][focused_slice] = raw_windows
    policy.dagger_replay.data["critic_observations"][focused_slice].zero_()

    policy.q_critic_keys = ("priv", "object_")
    policy._q_critic_widths = (1, 1)
    policy._replay_object_geo = torch.zeros(1)
    policy.depth_feature_dim = 1
    policy._vecnorm_snapshot = lambda: {}
    policy._normalize_replay_flat = lambda value, keys, widths, snapshot: value
    policy._normalize_replay_value = lambda key, value, snapshot: value
    policy.object_transform = nn.Identity()
    policy.encoder_priv = _V4ToyPrivilegedEncoder()
    policy.temporal_depth_gru = _V4ToyTemporalDepth()
    policy.object_adapt = _V4ToyPerceptionHead("object_pred")
    policy.object_pred_transform = nn.Identity()
    policy.adapt_module = _V4ToyPerceptionHead("priv_pred")
    policy.temporal_depth_gru_ema = copy.deepcopy(policy.temporal_depth_gru)
    policy.object_adapt_ema = copy.deepcopy(policy.object_adapt)
    policy.adapt_ema = copy.deepcopy(policy.adapt_module)
    policy.adapt_loss_fn = nn.MSELoss(reduction="none")
    parameters = list(policy.temporal_depth_gru.parameters())
    parameters += list(policy.object_adapt.parameters())
    parameters += list(policy.adapt_module.parameters())
    policy.opt_adapt = torch.optim.SGD(parameters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in parameters]

    # The four-way branch must not inspect this live-rollout sentinel.
    result = policy.train_adapt(object())

    assert result["adapt/perception_four_way"] == 1.0
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters))
    assert len(policy.temporal_depth_gru.seen_policy_windows) == 2
    for seen in policy.temporal_depth_gru.seen_policy_windows:
        assert seen.shape == (2, 10, 1)
        assert torch.allclose(
            seen[:, 0, 0].sort().values,
            torch.tensor([0.1, 1.1]),
        )
    metrics = policy._replay_mix_metrics("perception")
    assert metrics["actual_failure_student_rows"] == 4.0
    assert metrics["valid_loss_failure_student_fraction"] == 1.0

    legacy_rollout = object()
    legacy_calls = []
    monkeypatch.setattr(
        PPOVEL,
        "train_adapt",
        lambda owner, rollout: legacy_calls.append(rollout) or {"legacy": 1.0},
    )
    policy.cfg.perception_replay_mode = "legacy_online_student"
    policy.cfg.teacher_perception_replay_fraction = 0.0
    legacy_result = policy.train_adapt(legacy_rollout)
    assert legacy_result == {"legacy": 1.0}
    assert legacy_calls == [legacy_rollout]


@pytest.mark.parametrize(
    "policy_type",
    (DistributionalFastSACTeacherBC, TVKDDistributionalFastSACTeacherBC),
)
def test_live_student_perception_uses_full_512x32_ppovel_rollout_without_replay(
    policy_type,
):
    class TrackingDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.2))
            self.seen: list[tuple[torch.Size, torch.Tensor]] = []

        def forward(self, td):
            self.seen.append(
                (td.batch_size, td["row_id"].detach().reshape(-1).clone())
            )
            td["_depth_feature"] = td["depth"] * self.scale
            return td

    class PredictionHead(nn.Module):
        def __init__(self, key, initial):
            super().__init__()
            self.key = key
            self.scale = nn.Parameter(torch.tensor(initial))

        def forward(self, td):
            td[self.key] = td["_depth_feature"] * self.scale
            return td

    class PrivilegedTarget(nn.Module):
        def forward(self, td):
            td["priv_feature"] = td["priv"]
            return td

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=1.0e-3)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure)

    policy = policy_type.__new__(policy_type)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_perception=True,
        perception_replay_mode="online_student_rollout",
        num_minibatches=8,
        train_every=32,
        use_object_adapt=True,
        max_grad_norm=1.0,
        enable_residual_distillation=False,
        train_dr_estimator=False,
    )
    policy.device = torch.device("cpu")
    policy.object_transform = nn.Identity()
    policy.encoder_priv = PrivilegedTarget()
    policy.temporal_depth_gru = TrackingDepth()
    policy.object_adapt = PredictionHead("object_pred", 0.3)
    policy.object_pred_transform = nn.Identity()
    policy.adapt_module = PredictionHead("priv_pred", 0.4)
    policy.temporal_depth_gru_ema = copy.deepcopy(policy.temporal_depth_gru)
    policy.object_adapt_ema = copy.deepcopy(policy.object_adapt)
    policy.adapt_ema = copy.deepcopy(policy.adapt_module)
    policy.adapt_loss_fn = nn.MSELoss(reduction="none")
    parameters = list(policy.temporal_depth_gru.parameters())
    parameters += list(policy.object_adapt.parameters())
    parameters += list(policy.adapt_module.parameters())
    policy.opt_adapt = CountingSGD(parameters)
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    initial_ema = {
        name: [parameter.detach().clone() for parameter in module.parameters()]
        for name, module in (
            ("depth", policy.temporal_depth_gru_ema),
            ("object", policy.object_adapt_ema),
            ("adapt", policy.adapt_ema),
        )
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("live perception must not access replay")

    for name in (
        "_teacher_perception_replay_loss",
        "_train_adapt_four_way",
        "_sample_four_way_perception_batch",
        "_sample_teacher_indices",
        "_sample_student_indices",
        "_reencode_perception_windows",
    ):
        setattr(policy, name, forbidden)
    policy.q_rng = torch.Generator().manual_seed(123)
    policy.teacher_perception_rng = torch.Generator().manual_seed(456)
    q_rng_before = policy.q_rng.get_state().clone()
    perception_rng_before = policy.teacher_perception_rng.get_state().clone()

    row_id = torch.arange(512 * 32).reshape(512, 32, 1)
    depth = torch.sin(row_id.float() * 0.01)
    rollout = TensorDict(
        {
            "row_id": row_id,
            "depth": depth,
            "object_": 0.7 * depth,
            "priv": 1.3 * depth,
            "is_init": torch.zeros(512, 32, 1, dtype=torch.bool),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.ones(
                512, 32, 1, dtype=torch.bool
            ),
        },
        batch_size=[512, 32],
    )

    result = policy.train_adapt(rollout)

    assert len(policy.temporal_depth_gru.seen) == 16
    assert all(shape == torch.Size([64, 32]) for shape, _ in policy.temporal_depth_gru.seen)
    visits = torch.cat([rows for _, rows in policy.temporal_depth_gru.seen])
    assert torch.equal(
        visits.bincount(minlength=512 * 32),
        torch.full((512 * 32,), 2, dtype=torch.long),
    )
    assert policy.opt_adapt.step_count == 16
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, parameters)
    )
    for name, online, ema in (
        ("depth", policy.temporal_depth_gru, policy.temporal_depth_gru_ema),
        ("object", policy.object_adapt, policy.object_adapt_ema),
        ("adapt", policy.adapt_module, policy.adapt_ema),
    ):
        for before, online_parameter, ema_parameter in zip(
            initial_ema[name], online.parameters(), ema.parameters()
        ):
            assert torch.allclose(
                ema_parameter,
                before.lerp(online_parameter.detach(), 0.04),
            )
    assert torch.equal(policy.q_rng.get_state(), q_rng_before)
    assert torch.equal(policy.teacher_perception_rng.get_state(), perception_rng_before)
    assert result["adapt/perception_online_student_rollout"] == 1.0
    assert result["adapt/perception_live_rows"] == 512 * 32
    assert result["adapt/perception_optimizer_steps"] == 16
    assert result["adapt/perception_sequence_length"] == 32
    assert result["adapt/perception_replay_rows"] == 0.0
    assert result["adapt/teacher_replay_rows"] == 0.0
    assert result["adapt/replay_student_fraction"] == 0.0
    assert result["adapt/replay_teacher_fraction"] == 0.0
    for key in (
        "adapt/priv_loss",
        "adapt/object_loss",
        "adapt/grad_norm",
        "adapt/depth_grad_norm",
        "adapt/depth_ema_rms_gap",
    ):
        assert torch.isfinite(torch.tensor(result[key]))


def test_live_student_perception_rejects_teacher_controlled_rows():
    policy = DistributionalFastSACTeacherBC.__new__(
        DistributionalFastSACTeacherBC
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_perception=True,
        perception_replay_mode="online_student_rollout",
    )
    rollout = TensorDict(
        {
            "is_init": torch.zeros(1, 2, 1, dtype=torch.bool),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([[[True], [False]]]),
        },
        batch_size=[1, 2],
    )

    with pytest.raises(RuntimeError, match="Teacher-controlled"):
        policy.train_adapt(rollout)
