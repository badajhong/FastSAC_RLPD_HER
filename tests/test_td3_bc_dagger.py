from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY, OBS_KEY, OBS_PRIV_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_REFERENCE_EPS,
    TwinDistributionalQ,
    _fastsac_action_center_to_latent,
    _sac_bootstrap_mask,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_BETA_TEACHER_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_Q_TEACHER_SOURCE_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
)
from active_adaptation.learning.ppo.ppo_vel import (
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    PPOVEL,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    VEL_CMD_KEY,
    set_recurrent_mode,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    CHECKPOINT_VERSION,
    FAILURE_PHASE_TEACHER_SOURCE_KEY,
    PERCEPTION_DEPTH_U8_KEY,
    PERCEPTION_IS_INIT_KEY,
    PERCEPTION_POLICY_RAW_KEY,
    PERCEPTION_REPLAY_SEMANTICS,
    PERCEPTION_VEL_COMMAND_RAW_KEY,
    REFERENCE_PHASE_KEY,
    TD3_BETA_KEY,
    TD3_COLLECTOR_NOISE_KEY,
    TD3_EXPLORATORY_STUDENT_ACTION_KEY,
    TD3_NOISE_FREE_STUDENT_ACTION_KEY,
    TRAINING_ALGORITHM,
    DistributionalTD3TeacherBC,
    DistributionalTD3TeacherBCConfig,
    _DistributionalTD3DaggerRolloutPolicy,
    _PREFILL_COMMAND_FINISHED_KEY,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
    _PREFILL_TERMINATED_KEY,
    _Q_REPLAY_FIELDS,
    _TD3DeviceReplay,
    _apply_student_collector_noise,
    _categorical_expected_value,
    _exact_teacher_bc_loss,
    _decode_replay_depth_u8,
    _encode_replay_depth_u8,
    _failure_lookback_offsets,
    _polyak_update_,
    _prefetch_td3_replay_sample_plans,
    _project_c51_probabilities,
    _select_lower_expected_c51_distribution,
    _source_counts,
    _td3_actor_q1_loss,
)


def _parameter_storage(module: nn.Module) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in module.parameters()}


def _bare_policy(**cfg) -> DistributionalTD3TeacherBC:
    policy = DistributionalTD3TeacherBC.__new__(DistributionalTD3TeacherBC)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    policy.device = torch.device("cpu")
    return policy


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _take_optimizer_step(module: nn.Module, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in module.parameters())
    loss.backward()
    optimizer.step()


def _replay_rows(count: int, offset: int) -> dict[str, torch.Tensor]:
    row = torch.arange(count, dtype=torch.float32) + float(offset)
    time = torch.arange(10, dtype=torch.float32)
    return {
        "critic_observations": torch.stack((row, row + 0.5, row + 0.75), dim=-1),
        "actions": row[:, None] + 1.0,
        "rewards": row + 2.0,
        "dones": torch.arange(count) % 3 == 0,
        "truncations": torch.arange(count) % 4 == 0,
        "discounts": torch.full((count,), 0.95),
        "next_critic_observations": torch.stack(
            (row + 4.0, row + 4.5, row + 4.75), dim=-1
        ),
        REFERENCE_PHASE_KEY: row.div(max(float(offset + count), 1.0)).clamp(0.0, 1.0),
        PERCEPTION_DEPTH_U8_KEY: (
            torch.arange(count * 10).reshape(count, 10, 1, 1, 1) % 101
        ).to(torch.uint8),
        PERCEPTION_POLICY_RAW_KEY: row[:, None, None] + time[None, :, None],
        PERCEPTION_VEL_COMMAND_RAW_KEY: (
            row[:, None, None] + time[None, :, None] + 0.25
        ),
        PERCEPTION_IS_INIT_KEY: torch.zeros(count, 10, dtype=torch.bool),
    }


def test_failure_lookback_offsets_span_the_inclusive_fifty_step_window():
    offsets = _failure_lookback_offsets(50, 10)

    assert offsets.dtype is torch.long
    assert torch.equal(
        offsets,
        torch.tensor([0, 6, 11, 17, 22, 28, 33, 39, 44, 50]),
    )
    assert offsets.unique().numel() == 10
    assert offsets.tolist() == sorted(offsets.tolist())


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    (
        (4096, (2048, 1434, 614)),
        (512, (256, 179, 77)),
    ),
)
def test_shared_source_counts_are_exact_student_uniform_and_focused_shares(
    batch_size, expected
):
    assert _source_counts(batch_size, 0.5, 0.3) == expected


def test_shared_replay_schema_carries_reference_phase_but_no_failure_copy():
    assert REFERENCE_PHASE_KEY == "reference_phase"
    assert FAILURE_PHASE_TEACHER_SOURCE_KEY == "failure_phase_teacher_source"
    assert REFERENCE_PHASE_KEY in _Q_REPLAY_FIELDS
    assert all("pre_failure" not in field for field in _Q_REPLAY_FIELDS)


def _failure_rollout() -> TensorDict:
    """Four episodes: Student failure, Teacher failure, timeout, success."""
    num_envs, steps = 4, 60
    phase = torch.stack(
        (
            torch.cat((torch.linspace(0.05, 0.15, 20), torch.linspace(0.60, 0.90, 40))),
            torch.linspace(0.10, 0.80, steps),
            torch.linspace(0.20, 0.85, steps),
            torch.linspace(0.30, 1.00, steps),
        )
    )
    student = torch.ones(num_envs, steps, dtype=torch.bool)
    student[1, -1] = False
    done = torch.zeros(num_envs, steps, dtype=torch.bool)
    terminated = torch.zeros_like(done)
    timeout = torch.zeros_like(done)
    command_finished = torch.zeros_like(done)

    # Reset env 0 before its later physical failure. Its old low phases must
    # never leak through the episode boundary into the failure lookback.
    done[0, 19] = True
    command_finished[0, 19] = True
    done[:, -1] = True
    terminated[0, -1] = True  # eligible Student physical failure
    terminated[1, -1] = True  # excluded: Teacher executed the terminal action
    # Pure timeout has no physical termination and is excluded.
    terminated[3, -1] = True  # command completion still counts as success
    timeout[2, -1] = True  # excluded: simulator time limit
    command_finished[3, -1] = True  # excluded: successful motion completion

    return TensorDict(
        {
            "ref_motion_phase_": phase.unsqueeze(-1),
            REFERENCE_PHASE_KEY: phase.unsqueeze(-1),
            DAGGER_IS_STUDENT_ACTION_KEY: student.unsqueeze(-1),
            "next": TensorDict(
                {
                    "done": done.unsqueeze(-1),
                    "terminated": terminated.unsqueeze(-1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": timeout.unsqueeze(-1),
                            "command_finished": command_finished.unsqueeze(-1),
                        },
                        batch_size=(num_envs, steps),
                    ),
                },
                batch_size=(num_envs, steps),
            ),
        },
        batch_size=(num_envs, steps),
    )


def test_failure_histogram_counts_only_student_physical_failure_and_resets_history():
    policy = _bare_policy(
        failure_phase_lookback_steps=50,
        failure_phase_samples_per_failure=10,
        failure_phase_num_bins=101,
    )
    policy._failure_phase_histogram = torch.zeros(101, dtype=torch.float64)
    policy._failure_phase_history = None
    policy._failure_phase_history_lengths = None
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0

    policy._update_failure_phase_histogram(_failure_rollout())

    assert policy._failure_phase_histogram.sum().item() == pytest.approx(10.0)
    occupied = policy._failure_phase_histogram.nonzero(as_tuple=False).squeeze(-1)
    # Env 0 reset at phase 0.15 and restarted at phase 0.60. No phase from the
    # previous episode may survive in the focused-Teacher risk distribution.
    assert occupied.min().item() >= 60


def test_physical_termination_wins_over_time_limit_but_pure_timeout_is_excluded():
    num_envs, steps = 2, 60
    phase = torch.linspace(0.2, 0.9, steps).view(1, steps, 1).expand(num_envs, -1, -1)
    done = torch.zeros(num_envs, steps, 1, dtype=torch.bool)
    terminated = torch.zeros_like(done)
    timeout = torch.zeros_like(done)
    command_finished = torch.zeros_like(done)
    done[:, -1] = True
    timeout[:, -1] = True
    terminated[0, -1] = True
    rollout = TensorDict(
        {
            REFERENCE_PHASE_KEY: phase,
            DAGGER_IS_STUDENT_ACTION_KEY: torch.ones_like(done),
            "next": TensorDict(
                {
                    "done": done,
                    "terminated": terminated,
                    "stats": TensorDict(
                        {
                            "episode_time_limit": timeout,
                            "command_finished": command_finished,
                        },
                        batch_size=(num_envs, steps),
                    ),
                },
                batch_size=(num_envs, steps),
            ),
        },
        batch_size=(num_envs, steps),
    )
    policy = _bare_policy(
        failure_phase_lookback_steps=50,
        failure_phase_samples_per_failure=10,
        failure_phase_num_bins=101,
    )
    policy._failure_phase_histogram = torch.zeros(101, dtype=torch.float64)
    policy._failure_phase_history = None
    policy._failure_phase_episode_count = 0
    policy._failure_phase_anchor_count = 0

    anchors = policy._update_failure_phase_histogram(rollout)

    assert anchors == 10
    assert policy._failure_phase_episode_count == 1
    assert policy._failure_phase_histogram.sum().item() == pytest.approx(10.0)


def _focused_teacher_policy() -> DistributionalTD3TeacherBC:
    policy = _bare_policy(failure_phase_num_bins=16)
    policy.q_teacher_replay = _TD3DeviceReplay(16, "cpu")
    rows = _replay_rows(3, 100)
    rows[REFERENCE_PHASE_KEY] = torch.tensor([0.10, 0.35, 0.90])
    policy.q_teacher_replay.extend(rows)
    policy._failure_phase_histogram = torch.zeros(16, dtype=torch.float64)
    policy._teacher_phase_bin_rows = ()
    policy._teacher_phase_index_ready = False
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._build_teacher_phase_index()
    return policy


def test_focused_teacher_sampling_falls_back_to_uniform_without_failures():
    policy = _focused_teacher_policy()

    indices, focused = policy._sample_teacher_indices(
        12,
        torch.Generator().manual_seed(811),
        focused_count=5,
    )

    assert indices.shape == focused.shape == (12,)
    assert indices.dtype is torch.long
    assert focused.dtype is torch.bool
    assert ((0 <= indices) & (indices < 3)).all()
    assert not focused.any()


def test_focused_teacher_sampling_uses_nearest_occupied_phase_bin():
    policy = _focused_teacher_policy()
    # Risk bin 11 is nearer Teacher phase 0.90 than 0.35 or 0.10.
    policy._failure_phase_histogram[11] = 1.0

    indices, focused = policy._sample_teacher_indices(
        10,
        torch.Generator().manual_seed(812),
        focused_count=4,
    )

    assert torch.equal(focused, torch.tensor([False] * 6 + [True] * 4))
    assert torch.equal(indices[focused], torch.full((4,), 2, dtype=torch.long))


def test_depth_uint8_codec_is_lossless_for_every_simulator_bin():
    depth = (torch.arange(101, dtype=torch.float32) / 100.0).reshape(1, 101, 1, 1)

    encoded = _encode_replay_depth_u8(depth)
    decoded = _decode_replay_depth_u8(encoded)

    assert encoded.dtype is torch.uint8
    assert encoded.shape == depth.shape
    assert torch.equal(encoded.reshape(-1), torch.arange(101, dtype=torch.uint8))
    assert decoded.dtype is torch.float32
    assert decoded.shape == depth.shape
    assert torch.equal(decoded, depth)


@pytest.mark.parametrize(
    "invalid",
    (
        torch.tensor([0.005]),
        torch.tensor([-0.01]),
        torch.tensor([1.01]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ),
)
def test_depth_uint8_codec_rejects_noncanonical_values(invalid):
    with pytest.raises(ValueError):
        _encode_replay_depth_u8(invalid)


def test_raw_perception_replay_public_field_contract_has_no_priv_pred_latent():
    assert CHECKPOINT_VERSION == 2
    assert PERCEPTION_REPLAY_SEMANTICS
    assert (
        PERCEPTION_DEPTH_U8_KEY,
        PERCEPTION_POLICY_RAW_KEY,
        PERCEPTION_VEL_COMMAND_RAW_KEY,
        PERCEPTION_IS_INIT_KEY,
    ) == (
        "perception_depth_u8",
        "perception_policy_raw",
        "perception_vel_command_raw",
        "perception_is_init",
    )


def test_teacher_prefill_and_shared_teacher_mix_defaults_and_validation():
    cfg = DistributionalTD3TeacherBCConfig()
    assert cfg.teacher_prefill_max_rollouts == 1000
    assert cfg.teacher_actor_replay_fraction == 0.5
    assert cfg.teacher_perception_replay_fraction == 0.5
    assert cfg.failure_phase_teacher_fraction == 0.3

    for invalid in (0, -1, True):
        cfg.teacher_prefill_max_rollouts = invalid
        with pytest.raises(ValueError, match="teacher_prefill_max_rollouts.*positive"):
            DistributionalTD3TeacherBC._validate_td3_config(cfg)
    cfg.teacher_prefill_max_rollouts = 10
    DistributionalTD3TeacherBC._validate_td3_config(cfg)


@pytest.mark.parametrize(
    "field",
    ("teacher_actor_replay_fraction", "teacher_perception_replay_fraction"),
)
@pytest.mark.parametrize("invalid", (-0.01, 1.01, float("nan"), True))
def test_teacher_replay_fraction_is_a_unit_interval(field, invalid):
    cfg = DistributionalTD3TeacherBCConfig()
    setattr(cfg, field, invalid)

    with pytest.raises(ValueError, match=field):
        DistributionalTD3TeacherBC._validate_td3_config(cfg)


def test_actor_batch_optionally_mixes_exact_teacher_replay_labels_and_raw_inputs():
    policy = _bare_policy(
        dagger_batch_size=4,
        teacher_actor_replay_fraction=0.5,
    )
    policy.q_rng = torch.Generator().manual_seed(19)
    policy.q_teacher_replay = _TD3DeviceReplay(16, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(16, "cpu")
    teacher = _replay_rows(4, 100)
    main = _replay_rows(5, 1_000)
    main[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(5, dtype=torch.float32)[:, None] + 2_000.0
    )
    main[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.tensor(
        [True, False, True, True, False]
    )
    policy.q_teacher_replay.extend(teacher)
    policy.dagger_replay.extend(main)
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )

    batch = policy._sample_actor_batch(
        torch.tensor([1, 3]),
        torch.tensor([0, 2]),
    )

    source = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
    assert torch.equal(source, torch.tensor([True, True, False, False]))
    assert torch.equal(
        batch[DAGGER_REPLAY_TEACHER_ACTIONS][source],
        teacher["actions"][[0, 2]],
    )
    assert batch[DAGGER_TEACHER_ACTION_VALID_KEY][source].all()
    assert torch.equal(
        batch[DAGGER_REPLAY_TEACHER_ACTIONS][~source],
        main[DAGGER_REPLAY_TEACHER_ACTIONS][[1, 3]],
    )
    assert torch.equal(
        batch[PERCEPTION_POLICY_RAW_KEY][source],
        teacher[PERCEPTION_POLICY_RAW_KEY][[0, 2]],
    )


def test_zero_teacher_actor_fraction_preserves_main_only_sampling_and_rng():
    policy = _bare_policy(
        dagger_batch_size=4,
        teacher_actor_replay_fraction=0.0,
    )
    policy.q_rng = torch.Generator().manual_seed(37)
    expected_rng = torch.Generator().set_state(policy.q_rng.get_state())
    policy.dagger_replay = _TD3DeviceReplay(16, "cpu")
    main = _replay_rows(7, 1_000)
    main[DAGGER_REPLAY_TEACHER_ACTIONS] = main["actions"] + 5_000.0
    main[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(7, dtype=torch.bool)
    policy.dagger_replay.extend(main)
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )

    batch = policy._sample_actor_batch()
    expected = policy.dagger_replay.sample(
        4,
        "cpu",
        expected_rng,
        fields=(
            "critic_observations",
            DAGGER_REPLAY_TEACHER_ACTIONS,
            DAGGER_TEACHER_ACTION_VALID_KEY,
            PERCEPTION_DEPTH_U8_KEY,
            PERCEPTION_POLICY_RAW_KEY,
            PERCEPTION_VEL_COMMAND_RAW_KEY,
            PERCEPTION_IS_INIT_KEY,
        ),
    )

    _assert_nested_equal(batch, expected)
    assert DAGGER_Q_TEACHER_SOURCE_KEY not in batch
    assert torch.equal(policy.q_rng.get_state(), expected_rng.get_state())


def _raw_transition_policy(*, train_every: int = 10):
    policy = _bare_policy(
        train_every=train_every,
        perception_replay_burn_in=8,
    )
    policy.action_dim = 1
    policy.q_critic_keys = ("critic_raw",)
    policy._q_critic_dim = 1
    policy.observation_spec = {
        "object_geo_": SimpleNamespace(shape=(2,)),
    }
    policy._replay_vecnorm_keys = set()
    policy._replay_object_geo = None
    policy._replay_object_geo_fingerprint = None
    policy._perception_replay_history = None
    policy._perception_replay_history_count = 0
    policy._rollout_final_batch = None
    policy._truncation_final_batches = []
    policy._last_truncation_finals_used = 0
    policy._collect_dagger_replay_this_rollout = lambda: True
    policy._scalarize_q_reward = lambda reward: reward
    return policy


def _raw_transition_td(
    values: torch.Tensor,
    *,
    is_init: torch.Tensor | None = None,
    truncation_step: int | None = None,
) -> TensorDict:
    values = values.float()
    n, t = values.shape
    if is_init is None:
        is_init = torch.zeros(n, t, dtype=torch.bool)
    done = torch.zeros(n, t, dtype=torch.bool)
    timeout = torch.zeros(n, t, dtype=torch.bool)
    if truncation_step is not None:
        done[:, truncation_step] = True
        timeout[:, truncation_step] = True
    zeros_bool = torch.zeros(n, t, dtype=torch.bool)
    action = values.unsqueeze(-1)
    td = TensorDict(
        {
            "depth": (values.remainder(101) / 100.0).reshape(n, t, 1, 1, 1),
            "policy": action + 100.0,
            "vel_command": action + 200.0,
            REFERENCE_PHASE_KEY: values.div(max(float(t - 1), 1.0)).clamp(0.0, 1.0),
            "object_geo_": torch.tensor([1.0, 2.0]).expand(n, t, 2).clone(),
            "critic_raw": action + 300.0,
            "is_init": is_init,
            "step_count": torch.full((n, t), 100),
            ACTION_KEY: action,
            DAGGER_TEACHER_ACTION_KEY: action + 10.0,
            DAGGER_TEACHER_ACTION_VALID_KEY: torch.ones(n, t, dtype=torch.bool),
            DAGGER_IS_STUDENT_ACTION_KEY: zeros_bool,
            TD3_NOISE_FREE_STUDENT_ACTION_KEY: action + 20.0,
            TD3_EXPLORATORY_STUDENT_ACTION_KEY: action + 30.0,
            TD3_COLLECTOR_NOISE_KEY: torch.zeros_like(action),
            TD3_BETA_KEY: torch.ones(n, t),
            ("next", "reward"): values,
            ("next", "done"): done,
            ("next", "terminated"): zeros_bool,
            ("next", "discount"): torch.ones(n, t),
            ("next", "stats", "episode_time_limit"): timeout,
            ("next", "stats", "command_finished"): zeros_bool,
        },
        batch_size=(n, t),
    )
    return td


def _raw_final_td(value: float, *, is_init: bool = False) -> TensorDict:
    scalar = torch.tensor([[value]], dtype=torch.float32)
    return TensorDict(
        {
            "depth": torch.tensor([value % 101 / 100.0]).reshape(1, 1, 1, 1),
            "policy": scalar + 100.0,
            "vel_command": scalar + 200.0,
            "object_geo_": torch.tensor([[1.0, 2.0]]),
            "critic_raw": scalar + 300.0,
            "is_init": torch.tensor([is_init]),
        },
        batch_size=(1,),
    )


def test_replay_object_geometry_cached_during_inference_is_autograd_safe():
    policy = _raw_transition_policy()
    geometry = TensorDict(
        {
            OBJECT_GEO_KEY: torch.tensor([[[1.0, 2.0]]]),
        },
        batch_size=(1, 1),
    )

    with torch.inference_mode():
        policy._register_replay_object_geo(geometry)

    cached = policy._replay_object_geo
    assert cached is not None
    assert not cached.is_inference()

    # Regression for the exact Teacher-perception failure: TransformObject's
    # trainable orientation path must be able to save the cached points for
    # the backward of torch.matmul.
    orientation = nn.Parameter(torch.eye(cached.shape[-1]))
    torch.matmul(cached, orientation.transpose(-1, -2)).sum().backward()
    assert orientation.grad is not None
    assert torch.isfinite(orientation.grad).all()


def test_raw_replay_window_alignment_reset_markers_and_no_priv_pred_storage():
    policy = _raw_transition_policy()
    values = torch.arange(10).reshape(1, 10)
    resets = torch.zeros(1, 10, dtype=torch.bool)
    resets[0, 4] = True
    rollout = _raw_transition_td(values, is_init=resets)
    policy.capture_rollout_final_observation(_raw_final_td(10.0))

    chunks = list(policy._dagger_transition_chunks(rollout))

    assert len(chunks) == 2
    first, second = chunks
    assert first[PERCEPTION_POLICY_RAW_KEY].shape == (1, 10, 1)
    assert torch.equal(
        first[PERCEPTION_POLICY_RAW_KEY][0, :, 0], torch.arange(10) + 100.0
    )
    assert torch.equal(
        second[PERCEPTION_POLICY_RAW_KEY][0, :, 0], torch.arange(1, 11) + 100.0
    )
    assert first[PERCEPTION_POLICY_RAW_KEY][0, -2, 0].item() == 108.0
    assert first[PERCEPTION_POLICY_RAW_KEY][0, -1, 0].item() == 109.0
    assert torch.equal(
        first[PERCEPTION_IS_INIT_KEY][0],
        torch.tensor(
            [False, False, False, False, True, False, False, False, False, False]
        ),
    )
    assert not ({"observations", "next_observations", "priv_pred"} & set(first))
    assert first[PERCEPTION_DEPTH_U8_KEY].dtype is torch.uint8


def test_timeout_replay_next_slot_uses_captured_true_final_raw_state():
    policy = _raw_transition_policy()
    rollout = _raw_transition_td(torch.arange(10).reshape(1, 10), truncation_step=8)
    true_timeout_final = _raw_final_td(77.0)
    timeout_capture = rollout[:, 8].clone()
    for key in (
        "depth",
        "policy",
        "vel_command",
        "object_geo_",
        "critic_raw",
        "is_init",
    ):
        timeout_capture["next", key] = true_timeout_final[key]
    policy.capture_truncation_final_observations(timeout_capture, step=8)
    policy.capture_rollout_final_observation(_raw_final_td(10.0))

    first = next(policy._dagger_transition_chunks(rollout))

    assert first["truncations"].item() is True
    assert first[PERCEPTION_POLICY_RAW_KEY][0, -2, 0].item() == 108.0
    assert first[PERCEPTION_POLICY_RAW_KEY][0, -1, 0].item() == 177.0
    assert first[PERCEPTION_VEL_COMMAND_RAW_KEY][0, -1, 0].item() == 277.0
    assert first["next_critic_observations"][0, 0].item() == 377.0


class _NoOpPerceptionModule(nn.Module):
    def forward(self, td):
        return td


class _ResetAwareEMAProjection(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, td):
        policy = td["policy"][..., 0]
        resets = td["is_init"].bool()
        state = torch.zeros(policy.shape[0], device=policy.device, dtype=policy.dtype)
        outputs = []
        for step in range(policy.shape[1]):
            state = torch.where(resets[:, step], torch.zeros_like(state), state)
            state = state + policy[:, step] * self.scale
            outputs.append(state)
        td["priv_pred"] = torch.stack(outputs, dim=1).unsqueeze(-1)
        return td


def test_current_ema_reencoding_changes_without_mutating_stored_raw_inputs():
    policy = _bare_policy(
        perception_replay_burn_in=8,
        perception_encode_microbatch_size=1,
        latent_dim=1,
    )
    policy.depth_feature_dim = 1
    policy._replay_object_geo = torch.tensor([1.0, 2.0])
    policy._vecnorm_snapshot = lambda: None
    policy._normalize_replay_value = lambda key, value, snapshot: value
    policy.temporal_depth_gru_ema = _NoOpPerceptionModule()
    policy.object_adapt_ema = _NoOpPerceptionModule()
    policy.object_pred_transform = _NoOpPerceptionModule()
    policy.adapt_ema = _ResetAwareEMAProjection(1.0)
    raw_policy = torch.arange(1, 11, dtype=torch.float32).reshape(1, 10, 1)
    raw_vel = raw_policy + 100.0
    resets = torch.zeros(1, 10, dtype=torch.bool)
    resets[:, 4] = True
    batch = {
        PERCEPTION_DEPTH_U8_KEY: torch.zeros(1, 10, 1, 1, 1, dtype=torch.uint8),
        PERCEPTION_POLICY_RAW_KEY: raw_policy,
        PERCEPTION_VEL_COMMAND_RAW_KEY: raw_vel,
        PERCEPTION_IS_INIT_KEY: resets,
    }
    stored_before = {key: value.clone() for key, value in batch.items()}

    first = policy._reencode_perception_windows(
        batch, include_current=True, include_next=True
    )
    with torch.no_grad():
        policy.adapt_ema.scale.fill_(2.0)
    second = policy._reencode_perception_windows(
        batch, include_current=True, include_next=True
    )

    assert first["observations"].shape == (1, 3)
    assert first["next_observations"].shape == (1, 3)
    # Window slot -2 is current. Reset at slot 4 makes recurrent state exact:
    # sum(policy[4:9]) = 5 + 6 + 7 + 8 + 9 = 35.
    assert first["observations"][0, -1].item() == pytest.approx(35.0)
    assert first["next_observations"][0, -1].item() == pytest.approx(45.0)
    assert second["observations"][0, -1].item() == pytest.approx(70.0)
    assert second["next_observations"][0, -1].item() == pytest.approx(90.0)
    assert not torch.equal(first["observations"], second["observations"])
    assert not first["observations"].requires_grad
    for key, value in batch.items():
        assert torch.equal(value, stored_before[key])
    assert all(
        "priv_pred" not in field
        for field in (
            PERCEPTION_DEPTH_U8_KEY,
            PERCEPTION_POLICY_RAW_KEY,
            PERCEPTION_VEL_COMMAND_RAW_KEY,
            PERCEPTION_IS_INIT_KEY,
        )
    )
    assert set(_Q_REPLAY_FIELDS) == {
        "critic_observations",
        "actions",
        "rewards",
        "dones",
        "truncations",
        "discounts",
        "next_critic_observations",
        REFERENCE_PHASE_KEY,
        PERCEPTION_DEPTH_U8_KEY,
        PERCEPTION_POLICY_RAW_KEY,
        PERCEPTION_VEL_COMMAND_RAW_KEY,
        PERCEPTION_IS_INIT_KEY,
    }
    assert {"observations", "next_observations", "priv_pred"}.isdisjoint(
        _Q_REPLAY_FIELDS
    )


class _ReplayDepthStudent(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, td):
        td["_depth_feature"] = (
            td[DEPTH_KEY].flatten(-3).mean(-1, keepdim=True) * self.scale
        )
        return td


class _ReplayObjectStudent(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, td):
        td[OBJECT_PRED_KEY] = (td["_depth_feature"] + td[OBS_KEY][..., :1]) * self.scale
        return td


class _ReplayAdaptStudent(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, td):
        td[PRIV_PRED_KEY] = (
            td[OBJECT_PRED_KEY] + td[VEL_CMD_KEY][..., :1]
        ) * self.scale
        return td


class _ReplayPrivTarget(nn.Module):
    def forward(self, td):
        td[PRIV_FEATURE_KEY] = td[OBS_PRIV_KEY][..., :1] + td[OBJECT_KEY][..., :1]
        return td


def _teacher_perception_policy(
    fraction: float = 0.5,
    *,
    batch_size: int = 2,
    row_count: int = 4,
):
    policy = _bare_policy(
        teacher_perception_replay_fraction=fraction,
        perception_encode_microbatch_size=2,
        teacher_perception_batch_size=batch_size,
        failure_phase_teacher_fraction=0.3,
        failure_phase_num_bins=16,
        perception_replay_burn_in=8,
        latent_dim=1,
        num_minibatches=1,
        train_every=2,
        max_grad_norm=10.0,
        train_dr_estimator=False,
        load_pretrained_perception=False,
        perception_checkpoint_path=None,
        train_perception=True,
        lr=0.01,
    )
    policy.depth_feature_dim = 1
    policy.q_critic_keys = [OBS_PRIV_KEY, OBJECT_KEY]
    policy._q_critic_widths = [1, 1]
    policy._replay_object_geo = torch.tensor([1.0])
    policy._vecnorm_snapshot = lambda: None
    policy._normalize_replay_value = lambda key, value, snapshot: value
    policy.temporal_depth_gru = _ReplayDepthStudent(0.5)
    policy.depth_cnn = policy.temporal_depth_gru
    policy.object_adapt = _ReplayObjectStudent(0.75)
    policy.object_pred_transform = _NoOpPerceptionModule()
    policy.adapt_module = _ReplayAdaptStudent(1.25)
    policy.temporal_depth_gru_ema = copy.deepcopy(policy.temporal_depth_gru)
    policy.object_adapt_ema = copy.deepcopy(policy.object_adapt)
    policy.adapt_ema = copy.deepcopy(policy.adapt_module)
    policy.object_transform = _NoOpPerceptionModule()
    policy.encoder_priv = _ReplayPrivTarget()
    policy.adapt_loss_fn = nn.MSELoss(reduction="none")
    parameters = (
        list(policy.temporal_depth_gru.parameters())
        + list(policy.object_adapt.parameters())
        + list(policy.adapt_module.parameters())
    )
    policy.opt_adapt = torch.optim.SGD(parameters, lr=0.01)
    policy.q_rng = torch.Generator().manual_seed(73)
    policy.teacher_perception_rng = torch.Generator().manual_seed(74)
    policy.q_teacher_replay = _TD3DeviceReplay(max(8, row_count), "cpu")
    rows = _replay_rows(row_count, 0)
    row = torch.arange(1, row_count + 1, dtype=torch.float32)
    rows["critic_observations"] = torch.stack((row, row + 1.0), dim=-1)
    rows[REFERENCE_PHASE_KEY] = torch.linspace(0.0, 1.0, row_count)
    rows[PERCEPTION_DEPTH_U8_KEY] = torch.full(
        (row_count, 10, 1, 1, 1), 50, dtype=torch.uint8
    )
    rows[PERCEPTION_POLICY_RAW_KEY] = torch.ones(row_count, 10, 1)
    rows[PERCEPTION_VEL_COMMAND_RAW_KEY] = torch.full((row_count, 10, 1), 0.25)
    rows[PERCEPTION_IS_INIT_KEY] = torch.zeros(row_count, 10, dtype=torch.bool)
    policy.q_teacher_replay.extend(rows)
    return policy


_PERCEPTION_WARMSTART_MODULES = (
    "depth_cnn",
    "temporal_depth_gru",
    "temporal_depth_gru_ema",
    "object_adapt",
    "object_adapt_ema",
    "adapt_module",
    "adapt_ema",
)


class _WarmstartTemporal(nn.Module):
    def __init__(self, depth_cnn: nn.Module, recurrent_input: int = 2):
        super().__init__()
        self.depth_cnn = depth_cnn
        self.recurrent = nn.Linear(recurrent_input, 2)


def _perception_warmstart_policy(seed: int = 1):
    policy = _bare_policy(
        load_pretrained_perception=True,
        perception_checkpoint_path="/unused/student.pt",
        train_perception=True,
        lr=3e-4,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        depth_cnn = nn.Linear(2, 2)
        policy.depth_cnn = depth_cnn
        policy.temporal_depth_gru = _WarmstartTemporal(depth_cnn)
        policy.temporal_depth_gru_ema = _WarmstartTemporal(nn.Linear(2, 2))
        policy.object_adapt = nn.Linear(2, 2)
        policy.object_adapt_ema = nn.Linear(2, 2)
        policy.adapt_module = nn.Linear(2, 2)
        policy.adapt_ema = nn.Linear(2, 2)
        policy.actor_adapt = nn.Linear(2, 2)
        policy.encoder_priv = nn.Linear(2, 2)
        policy.qnet = nn.Linear(2, 2)
    online_parameters = []
    for name in ("temporal_depth_gru", "object_adapt", "adapt_module"):
        online_parameters.extend(getattr(policy, name).parameters())
    policy.opt_adapt = torch.optim.Adam(online_parameters, lr=3e-4)
    return policy


def _write_perception_warmstart_checkpoint(
    path,
    *,
    seed: int = 100,
    missing: str | None = None,
    incompatible: str | None = None,
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        depth_cnn = nn.Linear(2, 2)
        modules = {
            "depth_cnn": depth_cnn,
            "temporal_depth_gru": _WarmstartTemporal(
                depth_cnn,
                recurrent_input=3 if incompatible == "temporal_depth_gru" else 2,
            ),
            "temporal_depth_gru_ema": _WarmstartTemporal(nn.Linear(2, 2)),
            "object_adapt": nn.Linear(2, 2),
            "object_adapt_ema": nn.Linear(2, 2),
            "adapt_module": nn.Linear(2, 2),
            "adapt_ema": nn.Linear(2, 2),
        }
        if incompatible is not None and incompatible != "temporal_depth_gru":
            modules[incompatible] = nn.Linear(3, 2)
        actor = nn.Linear(2, 2)
        encoder = nn.Linear(2, 2)
        qnet = nn.Linear(2, 2)
    policy = {name: module.state_dict() for name, module in modules.items()}
    if missing is not None:
        policy.pop(missing)
    # Deliberately include unrelated state. The selective loader must ignore it.
    policy.update(
        {
            "actor_adapt": actor.state_dict(),
            "encoder_priv": encoder.state_dict(),
            "qnet": qnet.state_dict(),
            "last_phase": "finetune",
        }
    )
    torch.save({"policy": policy}, path)
    return modules


def test_pretrained_perception_loads_all_online_and_ema_modules_strictly(tmp_path):
    policy = _perception_warmstart_policy()
    checkpoint_path = tmp_path / "student_perception.pt"
    source = _write_perception_warmstart_checkpoint(checkpoint_path)

    policy._load_pretrained_perception_checkpoint(checkpoint_path)

    for name in _PERCEPTION_WARMSTART_MODULES:
        _assert_nested_equal(
            getattr(policy, name).state_dict(), source[name].state_dict()
        )


def test_pretrained_perception_loader_reads_nested_policy_and_not_actor_q(tmp_path):
    policy = _perception_warmstart_policy()
    checkpoint_path = tmp_path / "student_perception.pt"
    _write_perception_warmstart_checkpoint(checkpoint_path)
    untouched = {
        name: copy.deepcopy(getattr(policy, name).state_dict())
        for name in ("actor_adapt", "encoder_priv", "qnet")
    }

    policy._load_pretrained_perception_checkpoint(checkpoint_path)

    for name, expected in untouched.items():
        _assert_nested_equal(getattr(policy, name).state_dict(), expected)


def test_pretrained_perception_loader_rejects_missing_module(tmp_path):
    policy = _perception_warmstart_policy()
    checkpoint_path = tmp_path / "missing_perception.pt"
    _write_perception_warmstart_checkpoint(checkpoint_path, missing="object_adapt_ema")

    with pytest.raises((KeyError, ValueError, RuntimeError), match="object_adapt_ema"):
        policy._load_pretrained_perception_checkpoint(checkpoint_path)


def test_pretrained_perception_loader_rejects_incompatible_shape(tmp_path):
    policy = _perception_warmstart_policy()
    checkpoint_path = tmp_path / "incompatible_perception.pt"
    _write_perception_warmstart_checkpoint(
        checkpoint_path, incompatible="temporal_depth_gru"
    )

    with pytest.raises(RuntimeError, match="temporal_depth_gru"):
        policy._load_pretrained_perception_checkpoint(checkpoint_path)


def test_pretrained_perception_loader_rejects_inconsistent_depth_aliases(tmp_path):
    policy = _perception_warmstart_policy()
    checkpoint_path = tmp_path / "inconsistent_depth_aliases.pt"
    _write_perception_warmstart_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["policy"]["depth_cnn"]["weight"] = (
        checkpoint["policy"]["depth_cnn"]["weight"].clone() + 1.0
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="depth_cnn.*alias mismatch"):
        policy._load_pretrained_perception_checkpoint(checkpoint_path)


def test_frozen_pretrained_perception_disables_optimizer_and_adaptation_updates():
    policy = _teacher_perception_policy(fraction=0.5)
    policy.cfg.train_perception = False
    policy._set_perception_trainable(False)
    before = {
        name: copy.deepcopy(getattr(policy, name).state_dict())
        for name in _PERCEPTION_WARMSTART_MODULES
        if hasattr(policy, name)
    }

    result = policy.train_adapt(TensorDict({}, batch_size=[]))

    assert isinstance(result, dict)
    assert getattr(policy, "opt_adapt", None) is None
    for name, expected in before.items():
        module = getattr(policy, name)
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
        _assert_nested_equal(module.state_dict(), expected)


def test_trainable_pretrained_perception_restores_online_optimizer_and_ema_contract():
    policy = _teacher_perception_policy(fraction=0.5)
    policy.cfg.train_perception = True
    policy._set_perception_trainable(True)

    assert isinstance(policy.opt_adapt, torch.optim.Optimizer)
    for name in ("temporal_depth_gru", "object_adapt", "adapt_module"):
        module = getattr(policy, name)
        assert module.training
        assert all(parameter.requires_grad for parameter in module.parameters())
    for name in ("temporal_depth_gru_ema", "object_adapt_ema", "adapt_ema"):
        module = getattr(policy, name)
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())


class _RolloutPerceptionMarker(nn.Module):
    def __init__(self, name: str, calls: list[str]):
        super().__init__()
        self.name = name
        self.calls = calls

    def forward(self, td):
        self.calls.append(self.name)
        return td


class _FixedMeanActor:
    def get_dist(self, td):
        del td
        return SimpleNamespace(mean=torch.tensor([[0.25, -0.5]]))


def test_frozen_perception_rollout_still_uses_loaded_ema_stack_only():
    policy = _bare_policy(use_object_adapt=True)
    calls = []
    policy.temporal_depth_gru = _RolloutPerceptionMarker("online_depth", calls)
    policy.object_adapt = _RolloutPerceptionMarker("online_object", calls)
    policy.adapt_module = _RolloutPerceptionMarker("online_adapt", calls)
    policy.temporal_depth_gru_ema = _RolloutPerceptionMarker("ema_depth", calls)
    policy.object_adapt_ema = _RolloutPerceptionMarker("ema_object", calls)
    policy.adapt_ema = _RolloutPerceptionMarker("ema_adapt", calls)
    policy.object_pred_transform = _NoOpPerceptionModule()
    policy.actor_adapt = _FixedMeanActor()
    policy._set_perception_trainable(False)

    action = policy._student_latent(TensorDict({}, batch_size=[1]))

    assert calls == ["ema_depth", "ema_object", "ema_adapt"]
    assert torch.equal(action, torch.tensor([[0.25, -0.5]]))


@pytest.mark.parametrize(
    ("load_pretrained", "train_perception", "expected"),
    (
        (False, True, ["main_ppo", "actor_migration", "set_trainable_True"]),
        (
            True,
            False,
            [
                "main_ppo",
                "actor_migration",
                "perception_overlay_/student.pt",
                "set_trainable_False",
            ],
        ),
    ),
)
def test_fresh_source_public_loader_orders_main_ppo_then_optional_perception_overlay(
    monkeypatch, load_pretrained, train_perception, expected
):
    policy = _bare_policy(
        dagger_seed=3,
        q_seed=5,
        load_pretrained_perception=load_pretrained,
        perception_checkpoint_path="/student.pt" if load_pretrained else None,
        train_perception=train_perception,
    )
    policy.actor_adapt = nn.Linear(2, 2)
    policy.qnet = nn.Linear(2, 2)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.actor_target = None
    policy.dagger_rng = torch.Generator().manual_seed(1)
    policy.q_rng = torch.Generator().manual_seed(2)
    policy.collector_exploration_rng = torch.Generator().manual_seed(3)
    policy.target_policy_rng = torch.Generator().manual_seed(4)
    policy.teacher_perception_rng = torch.Generator().manual_seed(5)
    policy._perception_initialization = {
        "loaded": False,
        "source_path": None,
    }
    events = []

    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda owner, state, strict: events.append("main_ppo") or [],
    )
    policy._migrate_fresh_ppo_student_actor_to_latent = MethodType(
        lambda owner: events.append("actor_migration"), policy
    )

    def overlay(owner, path):
        events.append(f"perception_overlay_{path}")
        owner._perception_initialization = {
            "loaded": True,
            "source_path": str(path),
        }

    policy._load_pretrained_perception_checkpoint = MethodType(overlay, policy)
    policy._set_perception_trainable = MethodType(
        lambda owner, value: events.append(f"set_trainable_{value}"), policy
    )

    policy.load_state_dict({"last_phase": "train"})

    assert events == expected


def test_teacher_perception_loss_reencodes_raw_window_with_trainable_current_model():
    policy = _teacher_perception_policy()
    stored_before = {
        key: value.clone() for key, value in policy.q_teacher_replay.data.items()
    }

    with set_recurrent_mode(True):
        losses = policy._teacher_perception_replay_loss()
    (losses["priv_loss"] + losses["object_loss"]).backward()

    assert losses["rows"].item() == 2
    assert losses["valid_fraction"].item() == pytest.approx(1.0)
    assert policy.temporal_depth_gru.scale.grad.abs().item() > 0
    assert policy.object_adapt.scale.grad.abs().item() > 0
    assert policy.adapt_module.scale.grad.abs().item() > 0
    assert PRIV_PRED_KEY not in policy.q_teacher_replay.data
    for key, value in stored_before.items():
        assert torch.equal(policy.q_teacher_replay.data[key], value)


def test_perception_teacher_half_uses_seventy_thirty_uniform_focused_sampling():
    policy = _teacher_perception_policy(batch_size=10, row_count=10)
    calls = []

    def sample_teacher_indices(owner, count, generator, *, focused_count):
        del owner, generator
        calls.append((count, focused_count))
        indices = torch.arange(count, dtype=torch.long)
        focused = torch.zeros(count, dtype=torch.bool)
        focused[-focused_count:] = True
        return indices, focused

    policy._sample_teacher_indices = MethodType(sample_teacher_indices, policy)

    with set_recurrent_mode(True):
        losses = policy._teacher_perception_replay_loss()

    # The overall objective is live .50 + Teacher .50; splitting the Teacher
    # half 70/30 therefore yields the requested effective .50/.35/.15 weights.
    assert calls == [(10, 3)]
    assert policy.cfg.teacher_perception_replay_fraction == pytest.approx(0.5)
    assert losses["rows"].item() == 10


def test_teacher_perception_fraction_mixes_losses_in_existing_steps_and_updates_ema_once():
    policy = _teacher_perception_policy(fraction=0.5)
    old_ema = policy.temporal_depth_gru_ema.scale.detach().clone()
    online = TensorDict(
        {
            DEPTH_KEY: torch.full((1, 2, 1, 1, 1), 0.25),
            OBS_KEY: torch.full((1, 2, 1), 0.5),
            VEL_CMD_KEY: torch.full((1, 2, 1), 0.1),
            OBS_PRIV_KEY: torch.full((1, 2, 1), 1.0),
            OBJECT_KEY: torch.full((1, 2, 1), 2.0),
            OBJECT_GEO_KEY: torch.ones(1, 2, 1),
            "is_init": torch.zeros(1, 2, 1, dtype=torch.bool),
            "depth_hx": torch.zeros(1, 2, 1),
            "adapt_hx": torch.zeros(1, 2, 1),
        },
        batch_size=(1, 2),
    )

    info = policy.train_adapt(online)

    assert info["adapt/teacher_replay_fraction"] == pytest.approx(0.5)
    assert info["adapt/teacher_replay_rows"] == pytest.approx(2.0)
    assert "adapt/online_priv_loss" in info
    assert "adapt/teacher_replay_priv_loss" in info
    expected_ema = old_ema.lerp(policy.temporal_depth_gru.scale.detach(), 0.04)
    assert torch.allclose(policy.temporal_depth_gru_ema.scale, expected_ema)


def _initialize_prefill_episode_state(policy):
    policy._teacher_prefill_complete = False
    policy._teacher_prefill_pending = None
    policy._teacher_prefill_successful_episodes = 0
    policy._teacher_prefill_failed_episodes = 0
    policy._teacher_prefill_timeout_episodes = 0
    policy._teacher_prefill_incomplete_episodes = 0
    policy._teacher_prefill_discarded_rows = 0
    return policy


def _prefill_episode_policy(capacity: int = 64):
    policy = _bare_policy()
    policy.q_teacher_replay = _TD3DeviceReplay(capacity, "cpu")
    _initialize_prefill_episode_state(policy)
    return policy


def _mark_prefill_stub_as_success(chunk):
    count = int(chunk["actions"].shape[0])
    chunk[_PREFILL_ENV_INDEX_KEY] = torch.zeros(count, dtype=torch.long)
    chunk[_PREFILL_STEP_INDEX_KEY] = torch.arange(count, dtype=torch.long)
    chunk[_PREFILL_TERMINATED_KEY] = torch.zeros(count, dtype=torch.bool)
    chunk[_PREFILL_COMMAND_FINISHED_KEY] = torch.zeros(count, dtype=torch.bool)
    chunk[_PREFILL_COMMAND_FINISHED_KEY][-1] = True
    chunk["dones"] = torch.zeros(count, dtype=torch.bool)
    chunk["dones"][-1] = True
    return chunk


def _prefill_train_op_policy(
    *, rollouts: int, learning_starts: int, capacity: int = 64
):
    policy = _bare_policy(
        teacher_prefill_max_rollouts=rollouts,
        train_every=10,
        td3_learning_starts=learning_starts,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=100,
    )
    policy.teacher_prefill_rollout_count = 0
    policy.teacher_prefill_environment_steps = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.actor_update_count = 0
    policy.critic_update_count = 0
    policy.num_updates = 0
    policy._last_td3_diagnostics = {}
    policy.q_teacher_replay = _TD3DeviceReplay(capacity, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(64, "cpu")
    _initialize_prefill_episode_state(policy)
    policy.train_adapt = lambda rollout: (_ for _ in ()).throw(
        AssertionError("prefill must not optimize perception")
    )
    return policy


def _prefill_episode_rows(
    row_ids,
    env_indices,
    *,
    done=None,
    terminated=None,
    command_finished=None,
    valid=None,
    is_student=None,
    is_init=None,
    step_indices=None,
):
    row_ids = torch.as_tensor(row_ids, dtype=torch.float32)
    count = int(row_ids.numel())
    rows = _replay_rows(count, 0)
    rows["actions"] = row_ids[:, None]

    def _bool(value, default=False):
        if value is None:
            return torch.full((count,), default, dtype=torch.bool)
        return torch.as_tensor(value, dtype=torch.bool)

    rows["dones"] = _bool(done)
    rows["truncations"] = rows["dones"] & ~_bool(terminated)
    rows[DAGGER_TEACHER_ACTION_VALID_KEY] = _bool(valid, default=True)
    rows[DAGGER_IS_STUDENT_ACTION_KEY] = _bool(is_student)
    rows[TD3_COLLECTOR_NOISE_KEY] = torch.zeros(count, 1)
    rows[_PREFILL_ENV_INDEX_KEY] = torch.as_tensor(env_indices, dtype=torch.long)
    rows[_PREFILL_STEP_INDEX_KEY] = (
        torch.arange(count, dtype=torch.long)
        if step_indices is None
        else torch.as_tensor(step_indices, dtype=torch.long)
    )
    rows[_PREFILL_TERMINATED_KEY] = _bool(terminated)
    rows[_PREFILL_COMMAND_FINISHED_KEY] = _bool(command_finished)
    if is_init is not None:
        rows[PERCEPTION_IS_INIT_KEY][:, -2] = torch.as_tensor(is_init, dtype=torch.bool)
    return rows


def _prefill_event_rollout(
    steps: int,
    *,
    done_steps=(),
    terminated_steps=(),
    command_finished_steps=(),
    init_steps=(),
):
    done = torch.zeros(1, steps, 1, dtype=torch.bool)
    terminated = torch.zeros_like(done)
    command_finished = torch.zeros_like(done)
    is_init = torch.zeros_like(done)
    done[:, list(done_steps)] = True
    terminated[:, list(terminated_steps)] = True
    command_finished[:, list(command_finished_steps)] = True
    is_init[:, list(init_steps)] = True
    return TensorDict(
        {
            "is_init": is_init,
            "next": TensorDict(
                {
                    "done": done,
                    "terminated": terminated,
                    "stats": TensorDict(
                        {"command_finished": command_finished},
                        batch_size=(1, steps),
                    ),
                },
                batch_size=(1, steps),
            ),
        },
        batch_size=(1, steps),
    )


def test_teacher_prefill_success_commits_episode_across_multiple_train_op_chunks():
    policy = _prefill_episode_policy()

    first_committed, first_discarded = policy._stage_teacher_prefill_rows(
        _prefill_episode_rows([101, 102], [0, 0])
    )
    second_committed, second_discarded = policy._stage_teacher_prefill_rows(
        _prefill_episode_rows(
            [103, 104],
            [0, 0],
            done=[False, True],
            command_finished=[False, True],
        )
    )

    assert (first_committed, first_discarded) == (0, 0)
    assert (second_committed, second_discarded) == (4, 0)
    assert policy._teacher_prefill_pending_rows() == 0
    assert policy.q_teacher_replay.size == 4
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:4, 0],
        torch.tensor([101.0, 102.0, 103.0, 104.0]),
    )
    assert policy._teacher_prefill_successful_episodes == 1


def test_teacher_prefill_train_op_keeps_episode_pending_across_rollout_calls():
    policy = _prefill_train_op_policy(rollouts=3, learning_starts=1, capacity=4)
    chunks = [
        _prefill_episode_rows([111, 112], [0, 0]),
        _prefill_episode_rows(
            [113, 114],
            [0, 0],
            done=[False, True],
            command_finished=[False, True],
        ),
    ]
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((chunks.pop(0),)), policy
    )

    first = policy.train_op(TensorDict({}, batch_size=[]))
    assert first["td3/prefill_rows_this_rollout"] == 0
    assert policy._teacher_prefill_pending_rows() == 2
    assert policy.q_teacher_replay.size == 0

    second = policy.train_op(TensorDict({}, batch_size=[]))
    assert second["td3/prefill_rows_this_rollout"] == 4
    assert policy._teacher_prefill_pending_rows() == 0
    assert policy.q_teacher_replay.size == 4
    assert policy._teacher_prefill_complete is True
    assert second["td3/prefill_progress"] == pytest.approx(1.0)
    assert second["td3/teacher_replay_frozen"] == pytest.approx(1.0)
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:4, 0],
        torch.tensor([111.0, 112.0, 113.0, 114.0]),
    )


def test_failed_prefill_episode_does_not_fill_ring_and_collection_continues():
    policy = _prefill_train_op_policy(rollouts=2, learning_starts=1, capacity=2)
    chunks = [
        _prefill_episode_rows(
            [201, 202],
            [0, 0],
            done=[False, True],
            terminated=[False, True],
        ),
        _prefill_episode_rows(
            [301, 302],
            [0, 0],
            done=[False, True],
            command_finished=[False, True],
        ),
    ]
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((chunks.pop(0),)), policy
    )

    failed = policy.train_op(TensorDict({}, batch_size=[]))
    assert failed["td3/prefill_progress"] == pytest.approx(0.0)
    assert policy.q_teacher_replay.size == 0
    assert policy._teacher_prefill_failed_episodes == 1
    assert policy._teacher_prefill_active() is True

    succeeded = policy.train_op(TensorDict({}, batch_size=[]))
    assert succeeded["td3/prefill_progress"] == pytest.approx(1.0)
    assert policy.q_teacher_replay.size == 2
    assert policy._teacher_prefill_complete is True
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:2, 0],
        torch.tensor([301.0, 302.0]),
    )


def test_capacity_completion_discards_other_environments_unresolved_rows():
    policy = _prefill_train_op_policy(rollouts=2, learning_starts=1, capacity=2)
    chunk = _prefill_episode_rows(
        [401, 402, 501],
        [0, 0, 1],
        done=[False, True, False],
        command_finished=[False, True, False],
    )
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((chunk,)), policy
    )

    info = policy.train_op(TensorDict({}, batch_size=[]))

    assert policy.q_teacher_replay.size == 2
    assert policy._teacher_prefill_complete is True
    assert policy._teacher_prefill_pending_rows() == 0
    assert policy._teacher_prefill_incomplete_episodes == 1
    assert policy._teacher_prefill_discarded_rows == 1
    assert info["td3/prefill_unresolved_rows_discarded"] == 1
    assert info["td3/prefill_discarded_rows_this_rollout"] == 1


def test_teacher_prefill_physical_failure_discards_prior_rollout_prefix():
    policy = _prefill_episode_policy()
    policy._stage_teacher_prefill_rows(_prefill_episode_rows([201, 202], [0, 0]))

    committed, discarded = policy._stage_teacher_prefill_rows(
        _prefill_episode_rows(
            [203],
            [0],
            done=[True],
            terminated=[True],
            command_finished=[True],
        )
    )

    assert (committed, discarded) == (0, 3)
    assert policy.q_teacher_replay.size == 0
    assert policy._teacher_prefill_pending_rows() == 0
    assert policy._teacher_prefill_failed_episodes == 1
    assert policy._teacher_prefill_successful_episodes == 0
    assert policy._teacher_prefill_discarded_rows == 3


def test_teacher_prefill_interleaved_env_success_and_pure_timeout_are_isolated():
    policy = _prefill_episode_policy()

    committed, discarded = policy._stage_teacher_prefill_rows(
        _prefill_episode_rows(
            [301, 401, 302, 402],
            [0, 1, 0, 1],
            done=[False, False, True, True],
            command_finished=[False, False, True, False],
        )
    )

    assert (committed, discarded) == (2, 2)
    assert policy.q_teacher_replay.size == 2
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:2, 0],
        torch.tensor([301.0, 302.0]),
    )
    assert policy._teacher_prefill_successful_episodes == 1
    assert policy._teacher_prefill_timeout_episodes == 1
    assert policy._teacher_prefill_pending_rows() == 0


def test_teacher_prefill_invalid_fallback_rows_never_enter_successful_episode():
    policy = _prefill_episode_policy()

    committed, discarded = policy._stage_teacher_prefill_rows(
        _prefill_episode_rows(
            [501, 502, 503],
            [0, 0, 0],
            done=[False, False, True],
            command_finished=[False, False, True],
            valid=[True, False, True],
            is_student=[False, True, False],
        )
    )

    assert (committed, discarded) == (2, 0)
    assert policy.q_teacher_replay.size == 2
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:2, 0],
        torch.tensor([501.0, 503.0]),
    )


def test_teacher_prefill_end_discards_every_unresolved_env_once():
    policy = _prefill_episode_policy()
    policy._stage_teacher_prefill_rows(
        _prefill_episode_rows([601, 701, 602], [0, 1, 0])
    )

    first = policy._discard_unresolved_teacher_prefill_rows()
    second = policy._discard_unresolved_teacher_prefill_rows()

    assert (first, second) == (3, 0)
    assert policy.q_teacher_replay.size == 0
    assert policy._teacher_prefill_pending_rows() == 0
    assert policy._teacher_prefill_incomplete_episodes == 2
    assert policy._teacher_prefill_discarded_rows == 3


def test_teacher_prefill_max_rollout_guard_reports_unfilled_success_ring():
    policy = _prefill_train_op_policy(rollouts=1, learning_starts=1)
    pending = _prefill_episode_rows([611, 612], [0, 0])
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((pending,)), policy
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "teacher_prefill_max_rollouts=1.*local_size=0.*capacity=64.*pending_rows=2"
        ),
    ):
        policy.train_op(TensorDict({}, batch_size=[]))

    assert policy._teacher_prefill_pending_rows() == 2
    assert policy.q_teacher_replay.size == 0
    assert policy._teacher_prefill_incomplete_episodes == 0
    assert policy._teacher_prefill_discarded_rows == 0


def test_filtered_terminal_and_reset_cannot_leak_failed_prefix_into_later_success():
    policy = _prefill_episode_policy()
    first = _prefill_episode_rows([801], [0], step_indices=[0])
    policy._stage_teacher_prefill_rows(first, _prefill_event_rollout(steps=2))
    assert policy._teacher_prefill_pending_rows() == 1

    # The physical terminal at step 0 and reset at step 1 have no corresponding
    # replay payload rows (e.g. replay filtering removed them). A new valid row
    # at step 3 belongs to a later successful episode ending at step 5.
    second = _prefill_episode_rows([901], [0], step_indices=[3])
    committed, discarded = policy._stage_teacher_prefill_rows(
        second,
        _prefill_event_rollout(
            steps=6,
            done_steps=(0, 5),
            terminated_steps=(0,),
            command_finished_steps=(5,),
            init_steps=(1,),
        ),
    )

    assert (committed, discarded) == (1, 1)
    assert policy.q_teacher_replay.size == 1
    assert policy.q_teacher_replay.data["actions"][0, 0].item() == 901.0
    assert policy._teacher_prefill_failed_episodes == 1
    assert policy._teacher_prefill_successful_episodes == 1
    assert policy._teacher_prefill_pending_rows() == 0


def test_teacher_prefill_train_op_populates_only_q_teacher_and_touches_no_optimizer():
    policy = _bare_policy(
        teacher_prefill_max_rollouts=1,
        train_every=10,
        td3_learning_starts=3,
        dagger_beta_start=0.8,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=100,
    )
    policy.teacher_prefill_rollout_count = 0
    policy.teacher_prefill_environment_steps = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.actor_update_count = 4
    policy.critic_update_count = 6
    policy.num_updates = 9
    policy.q_teacher_replay = _TD3DeviceReplay(3, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(16, "cpu")
    _initialize_prefill_episode_state(policy)

    chunk = _mark_prefill_stub_as_success(_replay_rows(4, 100))
    chunk[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.tensor([True, False, True, True])
    chunk[DAGGER_IS_STUDENT_ACTION_KEY] = torch.tensor([False, True, False, False])
    chunk[TD3_COLLECTOR_NOISE_KEY] = torch.zeros(4, 1)
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((chunk,)), policy
    )

    actor_parameter = nn.Parameter(torch.tensor([1.0]))
    critic_parameter = nn.Parameter(torch.tensor([2.0]))
    adapt_parameter = nn.Parameter(torch.tensor([3.0]))
    policy.actor_optimizer = _CountingSGD([actor_parameter], lr=0.1)
    policy.critic_optimizer = _CountingSGD([critic_parameter], lr=0.1)
    policy.opt_adapt = _CountingSGD([adapt_parameter], lr=0.1)
    parameter_values_before = (
        actor_parameter.detach().clone(),
        critic_parameter.detach().clone(),
        adapt_parameter.detach().clone(),
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Teacher prefill must not run a training update")

    policy.train_adapt = forbidden
    policy._critic_update = forbidden
    policy._actor_update = forbidden

    info = policy.train_op(TensorDict({}, batch_size=[]))

    assert policy.teacher_prefill_rollout_count == 1
    assert policy.teacher_prefill_environment_steps == 10
    assert policy.dagger_rollout_count == 0
    assert policy.dagger_environment_steps == 0
    assert policy.num_updates == 9
    assert policy.actor_update_count == 4
    assert policy.critic_update_count == 6
    assert policy.dagger_replay.size == 0
    assert policy.dagger_replay.seen == 0
    assert policy.q_teacher_replay.size == 3
    assert policy.q_teacher_replay.seen == 3
    assert set(policy.q_teacher_replay.data) == set(_Q_REPLAY_FIELDS)
    assert torch.equal(
        policy.q_teacher_replay.data["actions"][:3],
        chunk["actions"][[0, 2, 3]],
    )
    for optimizer in (
        policy.actor_optimizer,
        policy.critic_optimizer,
        policy.opt_adapt,
    ):
        assert optimizer.step_calls == 0
        assert optimizer.zero_grad_calls == 0
    for parameter, before in zip(
        (actor_parameter, critic_parameter, adapt_parameter),
        parameter_values_before,
    ):
        assert torch.equal(parameter, before)
        assert parameter.grad is None

    assert info["td3/prefill_active"] == pytest.approx(1.0)
    assert info["td3/prefill_rollout_count"] == 1
    assert info["td3/prefill_max_rollouts"] == 1
    assert info["td3/prefill_target_rows"] == 3
    assert info["td3/prefill_progress"] == pytest.approx(1.0)
    assert info["td3/prefill_environment_steps"] == 10
    assert info["td3/prefill_rows_this_rollout"] == 3
    assert info["td3/prefill_forced_teacher_fraction"] == pytest.approx(0.75)
    assert info["td3/replay_size"] == 0
    assert info["td3/teacher_replay_size"] == 3
    assert info["td3/actor_updates_this_rollout"] == 0
    assert info["td3/critic_updates_this_rollout"] == 0
    assert info["td3/rollout_count"] == 0
    assert info["td3/beta"] == pytest.approx(policy.cfg.dagger_beta_start)
    assert policy._teacher_prefill_active() is False
    assert policy._teacher_mixture_probability() == pytest.approx(
        policy.cfg.dagger_beta_start
    )


def test_main_rollout_freezes_prefill_teacher_q_and_trains_from_both_replays():
    policy = _bare_policy(
        teacher_prefill_max_rollouts=1,
        train_every=10,
        td3_learning_starts=2,
        q_updates_per_rollout=1,
        q_batch_size=4,
        dagger_batch_size=4,
        policy_delay=1,
        dagger_beta_start=0.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=1,
        teacher_actor_replay_fraction=0.0,
        failure_phase_teacher_fraction=0.0,
        failure_phase_num_bins=16,
    )
    policy.teacher_prefill_rollout_count = 0
    policy.teacher_prefill_environment_steps = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.actor_update_count = 0
    policy.critic_update_count = 0
    policy.num_updates = 0
    policy._last_truncation_finals_used = 0
    policy._last_td3_diagnostics = {}
    policy.q_teacher_replay = _TD3DeviceReplay(2, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(16, "cpu")
    _initialize_prefill_episode_state(policy)
    policy.q_rng = torch.Generator().manual_seed(619)

    prefill = _mark_prefill_stub_as_success(_replay_rows(3, 100))
    prefill[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.tensor([True, False, True])
    prefill[DAGGER_IS_STUDENT_ACTION_KEY] = torch.tensor([False, True, False])
    prefill[DAGGER_REPLAY_TEACHER_ACTIONS] = torch.tensor([[801.0], [802.0], [803.0]])
    prefill[TD3_COLLECTOR_NOISE_KEY] = torch.zeros(3, 1)
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((prefill,)), policy
    )
    policy.train_adapt = lambda rollout: (_ for _ in ()).throw(
        AssertionError("prefill must not train perception")
    )

    prefill_info = policy.train_op(TensorDict({}, batch_size=[]))

    assert prefill_info["td3/prefill_active"] == pytest.approx(1.0)
    assert policy.q_teacher_replay.size == 2
    assert policy.dagger_replay.size == 0
    frozen_teacher_state = (
        policy.q_teacher_replay.ptr,
        policy.q_teacher_replay.size,
        policy.q_teacher_replay.seen,
        {
            key: value.detach().clone()
            for key, value in policy.q_teacher_replay.data.items()
        },
    )

    main = _replay_rows(4, 1_000)
    # Include teacher-executed rows deliberately: they remain useful DAgger BC
    # examples, but must no longer mutate the frozen prefill Q partition.
    main[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(4, dtype=torch.bool)
    main[DAGGER_IS_STUDENT_ACTION_KEY] = torch.tensor([True, False, True, False])
    main_teacher_labels = torch.tensor([[901.0], [902.0], [903.0], [904.0]])
    main[DAGGER_REPLAY_TEACHER_ACTIONS] = main_teacher_labels
    main[TD3_COLLECTOR_NOISE_KEY] = torch.zeros(4, 1)
    policy._dagger_transition_chunks = MethodType(
        lambda owner, rollout: iter((main,)), policy
    )

    critic_batches = []
    actor_batches = []
    perception_updates = []
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )

    def critic_update(batch):
        critic_batches.append(
            {key: value.detach().clone() for key, value in batch.items()}
        )
        policy.critic_update_count += 1
        return {}

    def actor_and_targets(batch):
        actor_batches.append(
            {key: value.detach().clone() for key, value in batch.items()}
        )
        policy.actor_update_count += 1
        return {}

    policy._critic_update = critic_update
    policy._maybe_delayed_actor_and_targets = actor_and_targets
    policy._mean_metric_dict = lambda metrics, keys: {key: 0.0 for key in keys}
    policy.train_adapt = lambda rollout: perception_updates.append(rollout) or {}

    main_info = policy.train_op(TensorDict({}, batch_size=[]))

    frozen_ptr, frozen_size, frozen_seen, frozen_data = frozen_teacher_state
    assert (
        policy.q_teacher_replay.ptr,
        policy.q_teacher_replay.size,
        policy.q_teacher_replay.seen,
    ) == (frozen_ptr, frozen_size, frozen_seen)
    assert policy.q_teacher_replay.data.keys() == frozen_data.keys()
    for key, frozen_value in frozen_data.items():
        assert torch.equal(
            policy.q_teacher_replay.data[key][:frozen_size],
            frozen_value[:frozen_size],
        )

    # Main replay stores only Student-executed transitions. Those rows retain
    # exact Teacher BC labels and raw perception windows; Teacher-executed main
    # rows would be dead under the shared 50/35/15 sampling contract.
    student_indices = torch.tensor([0, 2])
    assert policy.dagger_replay.size == 2
    assert policy.dagger_replay.seen == 2
    assert torch.equal(
        policy.dagger_replay.data["actions"][:2],
        main["actions"].index_select(0, student_indices),
    )
    assert torch.equal(
        policy.dagger_replay.data[DAGGER_IS_STUDENT_ACTION_KEY][:2],
        torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(
        policy.dagger_replay.data[DAGGER_REPLAY_TEACHER_ACTIONS][:2],
        main_teacher_labels.index_select(0, student_indices),
    )
    assert torch.equal(
        policy.dagger_replay.data[DAGGER_TEACHER_ACTION_VALID_KEY][:2],
        torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(
        policy.dagger_replay.data[PERCEPTION_POLICY_RAW_KEY][:2],
        main[PERCEPTION_POLICY_RAW_KEY].index_select(0, student_indices),
    )

    assert main_info["td3/prefill_active"] == pytest.approx(0.0)
    assert main_info["td3/teacher_replay_frozen"] == pytest.approx(1.0)
    assert main_info["td3/teacher_replay_rows_this_rollout"] == 0
    assert main_info["td3/replay_ready"] == pytest.approx(1.0)
    assert main_info["td3/teacher_replay_size"] == frozen_size
    assert main_info["td3/student_replay_rows"] == 2
    assert main_info["td3/critic_updates_this_rollout"] == 1
    assert main_info["td3/actor_updates_this_rollout"] == 1
    assert len(critic_batches) == 1
    assert len(actor_batches) == 1
    assert len(perception_updates) == 1

    # The critic receives an exact split: immutable prefill teacher actions and
    # only student-executed main actions. The Actor batch comes entirely from
    # main DAgger replay and therefore carries labels usable by exact BC.
    critic_batch = critic_batches[0]
    teacher_source = critic_batch[DAGGER_Q_TEACHER_SOURCE_KEY]
    assert teacher_source.sum().item() == 2
    assert (~teacher_source).sum().item() == 2
    assert set(critic_batch["actions"][teacher_source, 0].tolist()) <= {101.0, 103.0}
    assert set(critic_batch["actions"][~teacher_source, 0].tolist()) <= {
        1_001.0,
        1_003.0,
    }
    assert set(actor_batches[0][DAGGER_REPLAY_TEACHER_ACTIONS][:, 0].tolist()) <= {
        901.0,
        902.0,
        903.0,
        904.0,
    }
    assert actor_batches[0][DAGGER_TEACHER_ACTION_VALID_KEY].all()


def _sampling_policy(replay_type, seed: int, device):
    policy = _bare_policy(
        q_batch_size=6,
        q_teacher_replay_ratio=0.5,
        dagger_batch_size=5,
        policy_delay=2,
        teacher_actor_replay_fraction=0.0,
        failure_phase_teacher_fraction=0.0,
        failure_phase_num_bins=16,
    )
    policy.device = torch.device(device)
    policy.q_rng = torch.Generator(device=device).manual_seed(seed)
    policy.critic_update_count = 3
    policy.q_teacher_replay = replay_type(16, "cpu")
    policy.dagger_replay = replay_type(16, "cpu")
    teacher = _replay_rows(7, 100)
    student = _replay_rows(11, 1_000)
    student[DAGGER_IS_STUDENT_ACTION_KEY] = torch.tensor(
        [False, True, True, False, True, False, True, True, False, True, False]
    )
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(11, dtype=torch.float32)[:, None] + 2_000.0
    )
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.arange(11) % 3 != 0
    policy.q_teacher_replay.extend(teacher)
    policy.dagger_replay.extend(student)
    policy._failure_phase_histogram = torch.zeros(16, dtype=torch.float64)
    policy._teacher_phase_bin_rows = ()
    policy._teacher_phase_index_ready = False
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )
    return policy


def _curriculum_sampling_policy(seed: int = 853, device="cpu"):
    policy = _bare_policy(
        q_batch_size=20,
        q_teacher_replay_ratio=0.5,
        dagger_batch_size=20,
        teacher_actor_replay_fraction=0.5,
        failure_phase_teacher_fraction=0.3,
        failure_phase_num_bins=16,
        policy_delay=1,
    )
    policy.device = torch.device(device)
    policy.q_rng = torch.Generator(device=device).manual_seed(seed)
    policy.critic_update_count = 0
    policy.q_teacher_replay = _TD3DeviceReplay(64, "cpu")
    policy.dagger_replay = _TD3DeviceReplay(64, "cpu")

    teacher = _replay_rows(30, 100)
    teacher[REFERENCE_PHASE_KEY] = torch.linspace(0.0, 1.0, 30)
    student = _replay_rows(40, 1_000)
    student_mask = torch.arange(40) % 2 == 1
    student[DAGGER_IS_STUDENT_ACTION_KEY] = student_mask
    student[DAGGER_REPLAY_TEACHER_ACTIONS] = (
        torch.arange(40, dtype=torch.float32)[:, None] + 2_000.0
    )
    student[DAGGER_TEACHER_ACTION_VALID_KEY] = torch.ones(40, dtype=torch.bool)
    policy.q_teacher_replay.extend(teacher)
    policy.dagger_replay.extend(student)
    policy._prepare_dagger_learning_batch = MethodType(
        lambda owner, batch: batch, policy
    )
    policy._failure_phase_histogram = torch.zeros(16, dtype=torch.float64)
    policy._failure_phase_histogram[12] = 1.0
    policy._teacher_phase_bin_rows = ()
    policy._teacher_phase_index_ready = False
    policy._failure_phase_uniform_fallback_rows = 0
    policy._failure_phase_focused_rows = 0
    policy._build_teacher_phase_index()
    return policy, student_mask


def test_q_and_actor_batches_use_exact_shared_sources_and_student_only_main_rows():
    policy, student_mask = _curriculum_sampling_policy()

    critic = policy._sample_balanced_q_batch()
    actor = policy._sample_actor_batch()

    for batch in (critic, actor):
        teacher = batch[DAGGER_Q_TEACHER_SOURCE_KEY]
        focused = batch[FAILURE_PHASE_TEACHER_SOURCE_KEY]
        assert teacher.sum().item() == 10
        assert (teacher & ~focused).sum().item() == 7
        assert focused.sum().item() == 3
        assert not (focused & ~teacher).any()

    critic_teacher = critic[DAGGER_Q_TEACHER_SOURCE_KEY]
    critic_student_rows = (critic["actions"][~critic_teacher, 0] - 1_001.0).long()
    assert student_mask.index_select(0, critic_student_rows).all()

    actor_teacher = actor[DAGGER_Q_TEACHER_SOURCE_KEY]
    actor_student_rows = (
        actor["critic_observations"][~actor_teacher, 0] - 1_000.0
    ).long()
    assert student_mask.index_select(0, actor_student_rows).all()


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ),
)
def test_curriculum_prefetch_matches_direct_rng_batches_sources_and_devices(device):
    update_count = 3
    direct, _ = _curriculum_sampling_policy(887, device)
    prefetched, _ = _curriculum_sampling_policy(887, device)

    expected = []
    for _ in range(update_count):
        expected.append(
            (direct._sample_balanced_q_batch(), direct._sample_actor_batch())
        )

    plans = prefetched._prefetch_curriculum_sample_plans(update_count)
    actual = [
        (
            prefetched._sample_balanced_q_batch(plan),
            prefetched._sample_actor_batch(
                plan.actor_indices,
                plan.actor_teacher_indices,
                plan.actor_teacher_focused,
            ),
        )
        for plan in plans
    ]

    assert torch.equal(direct.q_rng.get_state(), prefetched.q_rng.get_state())
    assert len(plans) == update_count
    for plan in plans:
        assert plan.teacher_indices.device.type == "cpu"
        assert plan.student_indices.device.type == "cpu"
        assert plan.actor_indices is not None
        assert plan.actor_indices.device.type == "cpu"
        assert plan.actor_teacher_indices is not None
        assert plan.actor_teacher_indices.device.type == "cpu"
        assert plan.permutation.device == torch.device(device)
        assert plan.teacher_focused is not None
        assert plan.teacher_focused.device == torch.device(device)
        assert plan.actor_teacher_focused is not None
        assert plan.actor_teacher_focused.device == torch.device(device)

    for expected_pair, actual_pair in zip(expected, actual):
        for expected_batch, actual_batch in zip(expected_pair, actual_pair):
            _assert_nested_equal(expected_batch, actual_batch)
            teacher = actual_batch[DAGGER_Q_TEACHER_SOURCE_KEY]
            focused = actual_batch[FAILURE_PHASE_TEACHER_SOURCE_KEY]
            assert (~teacher).sum().item() == 10
            assert (teacher & ~focused).sum().item() == 7
            assert focused.sum().item() == 3
            assert all(
                value.device == torch.device(device) for value in actual_batch.values()
            )

    assert direct.q_teacher_replay.device.type == "cpu"
    assert direct.dagger_replay.device.type == "cpu"
    assert prefetched.q_teacher_replay.device.type == "cpu"
    assert prefetched.dagger_replay.device.type == "cpu"


def test_td3_indexed_cpu_replay_preserves_requested_fields_values_and_state():
    replay = _TD3DeviceReplay(8, "cpu")
    rows = {
        "vector": torch.arange(18, dtype=torch.float32).reshape(6, 3),
        "valid": torch.tensor([True, False, True, True, False, True]),
        "integer": torch.arange(6, dtype=torch.int64),
    }
    replay.extend(rows)
    before = (replay.ptr, replay.size, replay.seen)
    indices = torch.tensor([4, 1, 4, 0], dtype=torch.long)

    sampled = replay.sample_by_indices(
        indices, "cpu", fields=("valid", "vector", "integer")
    )

    assert tuple(sampled) == ("valid", "vector", "integer")
    for key in sampled:
        assert torch.equal(sampled[key], rows[key].index_select(0, indices))
        assert sampled[key].dtype == rows[key].dtype
        assert sampled[key].shape == (indices.numel(), *rows[key].shape[1:])
        assert sampled[key].device == torch.device("cpu")
    assert (replay.ptr, replay.size, replay.seen) == before
    assert replay._pinned_sample_staging == {}


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ),
)
def test_prefetched_td3_cpu_sampling_matches_sequential_rng_and_batches_exactly(
    device,
):
    seed = 731
    update_count = 5
    sequential = _sampling_policy(_TD3DeviceReplay, seed, device)
    prefetched = _sampling_policy(_TD3DeviceReplay, seed, device)
    expected = []
    for update_index in range(update_count):
        q_batch = sequential._sample_balanced_q_batch()
        actor_batch = None
        if (sequential.critic_update_count + update_index + 1) % 2 == 0:
            actor_batch = sequential._sample_actor_batch()
        expected.append((q_batch, actor_batch))

    plans = _prefetch_td3_replay_sample_plans(
        prefetched.dagger_replay,
        prefetched.q_teacher_replay,
        q_batch_size=6,
        actor_batch_size=5,
        update_count=update_count,
        policy_delay=2,
        critic_update_count=prefetched.critic_update_count,
        output_device=prefetched.device,
        generator=prefetched.q_rng,
    )
    actual = []
    for plan in plans:
        q_batch = prefetched._sample_balanced_q_batch(plan)
        actor_batch = (
            None
            if plan.actor_indices is None
            else prefetched._sample_actor_batch(plan.actor_indices)
        )
        actual.append((q_batch, actor_batch))

    assert torch.equal(sequential.q_rng.get_state(), prefetched.q_rng.get_state())
    for (expected_q, expected_actor), (actual_q, actual_actor) in zip(expected, actual):
        _assert_nested_equal(expected_q, actual_q)
        if expected_actor is None:
            assert actual_actor is None
        else:
            _assert_nested_equal(expected_actor, actual_actor)
        source = actual_q[DAGGER_Q_TEACHER_SOURCE_KEY]
        assert source.dtype is torch.bool
        assert source.sum().item() == 3
        assert source.numel() - source.sum().item() == 3
        assert all(value.device == torch.device(device) for value in actual_q.values())
        # Row encodings distinguish the teacher and student source rings.
        assert torch.all(actual_q["actions"][source] < 1_000.0)
        assert torch.all(actual_q["actions"][~source] >= 1_000.0)
        student_rows = (actual_q["actions"][~source, 0] - 1_001.0).long()
        assert torch.all(
            prefetched.dagger_replay.data[DAGGER_IS_STUDENT_ACTION_KEY][
                student_rows.cpu()
            ]
        )
        if torch.device(device).type == "cuda":
            assert all(value.is_contiguous() for value in actual_q.values())

    if torch.device(device).type == "cuda":
        staging = (
            *prefetched.dagger_replay._pinned_sample_staging.values(),
            *prefetched.q_teacher_replay._pinned_sample_staging.values(),
        )
        assert staging
        assert all(value.is_pinned() for value in staging)
    restored_rng = torch.Generator(device=device)
    restored_rng.set_state(prefetched.q_rng.get_state())
    assert torch.equal(
        torch.rand(16, device=device, generator=prefetched.q_rng),
        torch.rand(16, device=device, generator=restored_rng),
    )


def test_prefetched_actor_plan_preserves_mixed_teacher_main_rng_and_labels():
    seed = 947
    sequential = _sampling_policy(_TD3DeviceReplay, seed, "cpu")
    prefetched = _sampling_policy(_TD3DeviceReplay, seed, "cpu")
    sequential.cfg.teacher_actor_replay_fraction = 0.4
    prefetched.cfg.teacher_actor_replay_fraction = 0.4

    expected_q = sequential._sample_balanced_q_batch()
    expected_actor = sequential._sample_actor_batch()
    plan = _prefetch_td3_replay_sample_plans(
        prefetched.dagger_replay,
        prefetched.q_teacher_replay,
        q_batch_size=6,
        actor_batch_size=5,
        update_count=1,
        policy_delay=2,
        critic_update_count=3,
        teacher_actor_replay_fraction=0.4,
        output_device="cpu",
        generator=prefetched.q_rng,
    )[0]
    actual_q = prefetched._sample_balanced_q_batch(plan)
    actual_actor = prefetched._sample_actor_batch(
        plan.actor_indices,
        plan.actor_teacher_indices,
    )

    _assert_nested_equal(actual_q, expected_q)
    _assert_nested_equal(actual_actor, expected_actor)
    assert torch.equal(sequential.q_rng.get_state(), prefetched.q_rng.get_state())
    source = actual_actor[DAGGER_Q_TEACHER_SOURCE_KEY]
    assert source.sum().item() == 2
    assert (~source).sum().item() == 3
    # q_teacher_replay retained only executed Teacher actions, so synthesizing
    # Actor labels from its action field is exact and every such label is valid.
    assert actual_actor[DAGGER_TEACHER_ACTION_VALID_KEY][source].all()
    assert torch.all(actual_actor[DAGGER_REPLAY_TEACHER_ACTIONS][source] < 1_000.0)
    assert torch.all(actual_actor[DAGGER_REPLAY_TEACHER_ACTIONS][~source] >= 2_000.0)


def test_td3_config_rejects_persistent_teacher_h5_export():
    cfg = DistributionalTD3TeacherBCConfig()
    cfg.save_teacher_buffer = True

    with pytest.raises(ValueError, match="save_teacher_buffer"):
        DistributionalTD3TeacherBC._validate_td3_config(cfg)

    cfg.save_teacher_buffer = False
    cfg.teacher_prefill_max_rollouts = 10
    DistributionalTD3TeacherBC._validate_td3_config(cfg)


def test_disabled_teacher_h5_never_materializes_during_configure_or_snapshot(
    tmp_path,
):
    policy = _bare_policy(save_teacher_buffer=False)
    policy.teacher_replay = object()
    replay_path = tmp_path / "teacher_replay_buffer.h5"

    policy.configure_teacher_replay(replay_path, restore_path=None)
    snapshot = policy.snapshot_teacher_replay(100, "checkpoint_100")

    assert snapshot is None
    assert policy.teacher_replay is None
    assert not replay_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_lower_expected_head_selects_one_complete_c51_distribution_per_row():
    support = torch.tensor([-1.0, 0.0, 1.0])
    probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.05, 0.15, 0.80], [0.20, 0.60, 0.20]],
            [[0.20, 0.60, 0.20], [0.65, 0.25, 0.10], [0.10, 0.80, 0.10]],
        ]
    )
    logits = probabilities.log().requires_grad_()

    selected, expected_heads, selected_head = _select_lower_expected_c51_distribution(
        logits, support
    )

    expected = torch.stack(
        (probabilities[0, 0], probabilities[1, 1], probabilities[0, 2])
    )
    assert torch.allclose(selected, expected)
    assert torch.equal(selected_head, torch.tensor([0, 1, 0]))
    assert torch.allclose(
        expected_heads,
        torch.tensor([[-0.75, 0.75, 0.00], [-0.00, -0.55, 0.00]]),
        atol=1e-7,
    )
    assert torch.allclose(selected.sum(-1), torch.ones(3))

    # An atom-wise minimum is not a probability distribution here and differs
    # from every required selected row.
    atomwise_minimum = probabilities.min(dim=0).values
    assert not torch.allclose(selected, atomwise_minimum)
    assert not torch.allclose(atomwise_minimum.sum(-1), torch.ones(3))


def test_categorical_expected_value_uses_softmax_probabilities():
    support = torch.tensor([-2.0, 0.0, 2.0])
    logits = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0]])

    expected = _categorical_expected_value(logits, support)

    manual = (logits.softmax(-1) * support).sum(-1)
    assert torch.allclose(expected, manual)
    assert expected[0].item() == pytest.approx(0.0)
    assert expected[1].item() > 0.0


def test_c51_projection_obeys_ordinary_timeout_and_terminal_truth_table():
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    source = torch.zeros(5, support.numel())
    source[:, -1] = 1.0

    # ordinary, pure timeout, true termination, command completion, and a
    # timeout coincident with true termination after the authoritative VAIC
    # cause-resolution logic. Only the pure timeout remains a truncation.
    dones = torch.tensor([False, True, True, True, True])
    truncations = torch.tensor([False, True, False, False, False])
    bootstrap = _sac_bootstrap_mask(dones, truncations)
    assert torch.equal(bootstrap, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.zeros(5),
        bootstrap=bootstrap,
        effective_discount=torch.full((5,), 0.5),
        support=support,
    )

    expected = torch.zeros_like(source)
    expected[:2, 3] = 1.0  # 0 + 0.5 * z_max = 1
    expected[2:, 2] = 1.0  # terminals collapse onto immediate reward 0
    assert torch.equal(projected, expected)
    assert torch.equal(projected.sum(-1), torch.ones(5))
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(0.0)


def test_c51_projection_interpolates_and_reports_preprojection_clipping():
    support = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    source = torch.zeros(2, support.numel())
    source[0, 2] = 1.0
    source[1, -1] = 1.0

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.tensor([0.25, 3.0]),
        bootstrap=torch.ones(2),
        effective_discount=torch.ones(2),
        support=support,
    )

    assert torch.equal(projected[0], torch.tensor([0.0, 0.0, 0.75, 0.25, 0.0]))
    assert torch.equal(projected[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(projected.sum(-1), torch.ones(2))
    # Row 0 has one transformed atom above the support; row 1 has three.
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(4.0 / 10.0)


def test_c51_projection_accepts_locked_float32_501_atom_support():
    support = torch.linspace(-20.0, 20.0, 501, dtype=torch.float32)
    source = torch.zeros(1, support.numel())
    source[0, support.numel() // 2] = 1.0

    projected, left_clip_fraction, right_clip_fraction = _project_c51_probabilities(
        source,
        rewards=torch.zeros(1),
        bootstrap=torch.ones(1),
        effective_discount=torch.ones(1),
        support=support,
    )

    assert torch.allclose(projected, source)
    assert projected.sum().item() == pytest.approx(1.0)
    assert left_clip_fraction.item() == pytest.approx(0.0)
    assert right_clip_fraction.item() == pytest.approx(0.0)


def test_both_online_critics_train_against_the_same_detached_projection():
    support = torch.tensor([-1.0, 0.0, 1.0])
    target_probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.10, 0.20, 0.70]],
            [[0.20, 0.60, 0.20], [0.70, 0.20, 0.10]],
        ],
        requires_grad=True,
    )
    with torch.no_grad():
        selected, _, _ = _select_lower_expected_c51_distribution(
            target_probabilities.log(), support
        )
        projected, _, _ = _project_c51_probabilities(
            selected,
            rewards=torch.zeros(2),
            bootstrap=torch.ones(2),
            effective_discount=torch.ones(2),
            support=support,
        )
    online_logits = nn.Parameter(
        torch.tensor(
            [
                [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]],
                [[-0.3, 0.4, 0.2], [0.6, -0.2, 0.0]],
            ]
        )
    )

    per_head = (
        -(projected.unsqueeze(0) * F.log_softmax(online_logits, dim=-1))
        .sum(-1)
        .mean(-1)
    )
    per_head.sum().backward()

    expected_gradient = (
        online_logits.detach().softmax(-1) - projected.unsqueeze(0)
    ) / projected.shape[0]
    assert torch.allclose(online_logits.grad, expected_gradient)
    assert target_probabilities.grad is None
    assert projected.grad_fn is None


def test_online_twins_and_all_targets_have_independent_parameter_storage():
    online_q = TwinDistributionalQ(
        obs_dim=3,
        action_dim=2,
        hidden_dim=8,
        num_atoms=5,
        v_min=-2.0,
        v_max=2.0,
        layer_norm=False,
    )
    target_q = copy.deepcopy(online_q).requires_grad_(False)
    online_actor = nn.Linear(3, 2)
    target_actor = copy.deepcopy(online_actor).requires_grad_(False)

    q1_storage = _parameter_storage(online_q.qnets[0])
    q2_storage = _parameter_storage(online_q.qnets[1])
    assert q1_storage.isdisjoint(q2_storage)
    assert _parameter_storage(online_q).isdisjoint(_parameter_storage(target_q))
    assert _parameter_storage(online_actor).isdisjoint(_parameter_storage(target_actor))
    assert all(not parameter.requires_grad for parameter in target_q.parameters())
    assert all(not parameter.requires_grad for parameter in target_actor.parameters())

    target_q_before = [
        parameter.detach().clone() for parameter in target_q.parameters()
    ]
    q2_before = [
        parameter.detach().clone() for parameter in online_q.qnets[1].parameters()
    ]
    with torch.no_grad():
        next(online_q.qnets[0].parameters()).add_(1.0)
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(target_q.parameters(), target_q_before)
    )
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(online_q.qnets[1].parameters(), q2_before)
    )


class _ActionSensitiveC51Head(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, observations, actions):
        del observations
        action = actions[:, 0]
        return torch.stack(
            (-self.scale * action, torch.zeros_like(action), self.scale * action),
            dim=-1,
        )


class _ActionSensitiveTwinC51(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnets = nn.ModuleList(
            (_ActionSensitiveC51Head(1.0), _ActionSensitiveC51Head(-7.0))
        )
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(qnet(observations, actions) for qnet in self.qnets), dim=0
        )


class _TableC51Head(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, observations, actions):
        del actions
        return self.logits[: observations.shape[0]]


class _TableTwinC51(nn.Module):
    def __init__(self, head_1: torch.Tensor, head_2: torch.Tensor):
        super().__init__()
        self.qnets = nn.ModuleList((_TableC51Head(head_1), _TableC51Head(head_2)))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, observations, actions):
        return torch.stack(
            tuple(qnet(observations, actions) for qnet in self.qnets), dim=0
        )


def test_actor_loss_uses_expected_online_q1_and_freezes_critic_gradients():
    actor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(0.25)
    critic = _ActionSensitiveTwinC51()
    observations = torch.ones(4, 1)
    action = actor(observations)

    loss, expected_q1 = _td3_actor_q1_loss(critic, observations, action)
    direct_logits = critic(observations, action.detach())[0]
    direct_expected_q1 = _categorical_expected_value(direct_logits, critic.support)

    assert torch.allclose(expected_q1.detach(), direct_expected_q1)
    assert torch.allclose(loss.detach(), -direct_expected_q1.mean())
    assert all(parameter.requires_grad for parameter in critic.parameters())

    loss.backward()
    assert actor.weight.grad is not None
    assert actor.weight.grad.abs().item() > 0.0
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_integrated_critic_uses_common_detached_target_and_no_sac_path():
    policy = _bare_policy(
        target_policy_noise_std=0.3,
        target_policy_noise_clip=0.2,
        q_action_input_gain=1.0,
        dagger_action_clip=1.0,
        gamma=1.0,
        q_max_grad_norm=0.0,
    )
    _install_unit_action_contract(policy)
    online_logits_1 = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    online_logits_2 = torch.tensor([[-0.3, 0.4, 0.2], [0.6, -0.2, 0.0]])
    target_probabilities = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.05, 0.15, 0.80]],
            [[0.20, 0.60, 0.20], [0.65, 0.25, 0.10]],
        ]
    )
    policy.qnet = _TableTwinC51(online_logits_1, online_logits_2)
    policy.qnet_target = _TableTwinC51(
        target_probabilities[0].log(), target_probabilities[1].log()
    ).requires_grad_(False)
    policy.actor_target = nn.Linear(1, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        policy.actor_target.weight.zero_()
    policy._actor_target_dist_from_flat = MethodType(
        lambda owner, observations: SimpleNamespace(
            mean=owner.actor_target(observations)
        ),
        policy,
    )
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.05)
    policy.critic_update_count = 0
    policy.target_policy_rng = torch.Generator().manual_seed(31)
    policy.collector_exploration_rng = torch.Generator().manual_seed(32)
    target_rng_before = policy.target_policy_rng.get_state().clone()
    collector_rng_before = policy.collector_exploration_rng.get_state().clone()

    def forbidden_stochastic_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SAC/log-probability path was invoked")

    policy._sac_critic_dist_from_flat = forbidden_stochastic_path
    policy._normalized_action_log_prob = forbidden_stochastic_path
    policy.sac_dist_cls = forbidden_stochastic_path
    batch = {
        "observations": torch.ones(2, 1),
        "critic_observations": torch.ones(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        "next_observations": torch.ones(2, 1),
        "next_critic_observations": torch.ones(2, 1),
    }

    projected, _ = policy._distributional_td3_target(batch)
    expected_target = torch.stack(
        (target_probabilities[0, 0], target_probabilities[1, 1])
    )
    assert torch.allclose(projected, expected_target)
    assert projected.grad_fn is None
    assert projected.requires_grad is False
    assert not torch.equal(policy.target_policy_rng.get_state(), target_rng_before)
    assert torch.equal(
        policy.collector_exploration_rng.get_state(), collector_rng_before
    )

    metrics = policy._critic_update(batch)

    expected_gradient_1 = (
        online_logits_1.softmax(-1) - expected_target
    ) / expected_target.shape[0]
    expected_gradient_2 = (
        online_logits_2.softmax(-1) - expected_target
    ) / expected_target.shape[0]
    assert torch.allclose(policy.qnet.qnets[0].logits.grad, expected_gradient_1)
    assert torch.allclose(policy.qnet.qnets[1].logits.grad, expected_gradient_2)
    assert policy.critic_optimizer.step_calls == 1
    assert policy.critic_update_count == 1
    assert metrics["critic_loss"].item() == pytest.approx(
        metrics["critic_loss_1"].item() + metrics["critic_loss_2"].item()
    )
    assert all(parameter.grad is None for parameter in policy.qnet_target.parameters())
    assert all(parameter.grad is None for parameter in policy.actor_target.parameters())


def _install_unit_action_contract(policy: DistributionalTD3TeacherBC) -> None:
    policy._fastsac_action_low = torch.tensor([-1.0])
    policy._fastsac_action_high = torch.tensor([1.0])
    policy._fastsac_actor_action_center = torch.tensor([0.0])
    policy._fastsac_actor_action_scale = torch.tensor([1.0])
    policy._fastsac_q_action_center = torch.tensor([0.0])
    policy._fastsac_q_action_scale = torch.tensor([1.0])


def test_actor_update_combines_td3_and_bc_before_one_backward_and_step():
    policy = _bare_policy(
        eta_td3=0.7,
        lambda_bc=1.3,
        dagger_actor_huber_delta=0.4,
        dagger_action_clip=1.0,
        q_action_input_gain=1.0,
        max_grad_norm=1.0e6,
    )
    _install_unit_action_contract(policy)
    policy.actor_adapt = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.actor_adapt.weight.fill_(0.25)
    policy.qnet = _ActionSensitiveTwinC51()
    policy.actor_optimizer = _CountingSGD(policy.actor_adapt.parameters(), lr=0.1)
    policy.critic_optimizer = _CountingSGD(policy.qnet.parameters(), lr=0.1)
    policy.actor_update_count = 0
    policy._actor_dist_from_flat = MethodType(
        lambda owner, observations: SimpleNamespace(
            mean=owner.actor_adapt(observations)
        ),
        policy,
    )
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [-1.0]]),
        "critic_observations": torch.ones(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor([[0.5], [-0.2], [0.1]]),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_Q_TEACHER_SOURCE_KEY: torch.tensor([True, False, True]),
    }

    expected_actor = copy.deepcopy(policy.actor_adapt)
    expected_critic = copy.deepcopy(policy.qnet)
    expected_latent = expected_actor(batch["observations"])
    expected_q_action = expected_latent.tanh()
    expected_td3, _ = _td3_actor_q1_loss(
        expected_critic, batch["critic_observations"], expected_q_action
    )
    expected_bc = _exact_teacher_bc_loss(
        expected_latent,
        batch[DAGGER_REPLAY_TEACHER_ACTIONS],
        batch[DAGGER_TEACHER_ACTION_VALID_KEY],
        policy._fastsac_actor_action_center,
        policy._fastsac_actor_action_scale,
        policy.cfg.dagger_actor_huber_delta,
    )
    expected_total = (
        policy.cfg.eta_td3 * expected_td3 + policy.cfg.lambda_bc * expected_bc
    )
    expected_gradient = torch.autograd.grad(expected_total, expected_actor.weight)[0]
    expected_weight = policy.actor_adapt.weight.detach() - 0.1 * expected_gradient
    backward_gradients = []
    policy.actor_adapt.weight.register_hook(
        lambda gradient: backward_gradients.append(gradient.detach().clone())
    )

    metrics = policy._actor_update(batch)

    assert policy.actor_optimizer.step_calls == 1
    assert policy.actor_optimizer.zero_grad_calls == 1
    assert len(backward_gradients) == 1
    assert torch.allclose(backward_gradients[0], expected_gradient)
    assert torch.allclose(policy.actor_adapt.weight, expected_weight)
    assert metrics["weighted_td3_actor_loss"].item() == pytest.approx(
        policy.cfg.eta_td3 * metrics["td3_actor_loss"].item()
    )
    assert metrics["weighted_bc_loss"].item() == pytest.approx(
        policy.cfg.lambda_bc * metrics["exact_bc_loss"].item()
    )
    assert metrics["total_actor_loss"].item() == pytest.approx(
        metrics["weighted_td3_actor_loss"].item() + metrics["weighted_bc_loss"].item()
    )
    assert policy.actor_update_count == 1
    assert metrics["actor_teacher_replay_fraction"].item() == pytest.approx(2 / 3)
    assert all(parameter.grad is None for parameter in policy.qnet.parameters())


def test_exact_teacher_bc_matches_authoritative_latent_huber_computation():
    prediction_latent = nn.Parameter(
        torch.tensor([[0.20, -0.40], [99.0, 99.0], [-0.80, 0.60], [99.0, 99.0]])
    )
    teacher_action = torch.tensor(
        [[1.40, -2.60], [float("nan"), float("nan")], [0.20, 0.40], [9.0, 9.0]]
    )
    valid = torch.tensor([True, False, True, False])
    center = torch.tensor([1.0, -1.0])
    scale = torch.tensor([2.0, 4.0])
    huber_delta = 0.35

    actual = _exact_teacher_bc_loss(
        prediction_latent,
        teacher_action,
        valid,
        center,
        scale,
        huber_delta,
    )
    target_latent = _fastsac_action_center_to_latent(
        teacher_action[valid],
        scale,
        center,
        FASTSAC_REFERENCE_EPS,
    ).detach()
    expected = F.smooth_l1_loss(
        prediction_latent[valid], target_latent, beta=huber_delta
    )

    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.equal(
        prediction_latent.grad[~valid],
        torch.zeros_like(prediction_latent.grad[~valid]),
    )


def test_polyak_update_is_in_place_and_numerically_exact():
    source = nn.Linear(2, 1)
    target = copy.deepcopy(source).requires_grad_(False)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(4.0)
        target.weight.fill_(-2.0)
        target.bias.fill_(-4.0)
    source_before = copy.deepcopy(source.state_dict())

    _polyak_update_(target, source, tau=0.25)

    assert torch.equal(target.weight, torch.full_like(target.weight, -1.0))
    assert torch.equal(target.bias, torch.full_like(target.bias, -2.0))
    assert all(
        torch.equal(source.state_dict()[key], value)
        for key, value in source_before.items()
    )
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_actor_and_both_targets_update_only_on_policy_delay():
    policy = _bare_policy(policy_delay=2, q_tau=0.25)
    policy.actor_adapt = nn.Linear(1, 1)
    policy.actor_target = copy.deepcopy(policy.actor_adapt).requires_grad_(False)
    policy.qnet = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    with torch.no_grad():
        for parameter in policy.actor_adapt.parameters():
            parameter.fill_(2.0)
        for parameter in policy.actor_target.parameters():
            parameter.fill_(-2.0)
        for parameter in policy.qnet.parameters():
            parameter.fill_(4.0)
        for parameter in policy.qnet_target.parameters():
            parameter.zero_()
    policy.actor_update_count = 0

    def actor_update(owner, batch):
        del batch
        owner.actor_update_count += 1
        return {"updated": torch.tensor(1.0)}

    policy._actor_update = MethodType(actor_update, policy)
    actor_target_before = copy.deepcopy(policy.actor_target.state_dict())
    q_target_before = copy.deepcopy(policy.qnet_target.state_dict())
    policy.critic_update_count = 1

    assert policy._maybe_delayed_actor_and_targets({}) is None
    assert policy.actor_update_count == 0
    assert all(
        torch.equal(policy.actor_target.state_dict()[key], value)
        for key, value in actor_target_before.items()
    )
    assert all(
        torch.equal(policy.qnet_target.state_dict()[key], value)
        for key, value in q_target_before.items()
    )

    policy.critic_update_count = 2
    metrics = policy._maybe_delayed_actor_and_targets({})

    assert metrics["updated"].item() == pytest.approx(1.0)
    assert policy.actor_update_count == 1
    assert all(
        torch.equal(parameter, torch.full_like(parameter, -1.0))
        for parameter in policy.actor_target.parameters()
    )
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 1.0))
        for parameter in policy.qnet_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in policy.actor_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in policy.qnet_target.parameters()
    )


def test_collector_noise_is_q_coordinate_noise_on_student_rows_only():
    student = torch.tensor([[1.0, -1.0], [3.0, 3.0], [-1.0, -5.0]])
    teacher = torch.tensor([[8.0, 7.0], [6.0, 5.0], [4.0, 3.0]])
    teacher_before = teacher.clone()
    student_selected = torch.tensor([False, True, False])
    center = torch.tensor([1.0, -1.0])
    scale = torch.tensor([2.0, 4.0])
    gain = 0.5
    collector_rng = torch.Generator().manual_seed(123)
    target_rng = torch.Generator().manual_seed(456)
    target_rng_before = target_rng.get_state().clone()

    issued, exploratory_student, applied_q_noise = _apply_student_collector_noise(
        student_action=student,
        teacher_action=teacher,
        student_selected_mask=student_selected,
        noise_std=0.4,
        noise_clip=0.2,
        q_action_center=center,
        q_action_scale=scale,
        q_action_gain=gain,
        action_low=torch.tensor([-10.0, -10.0]),
        action_high=torch.tensor([10.0, 10.0]),
        generator=collector_rng,
        project_fn=lambda value: value,
    )

    assert torch.equal(teacher, teacher_before)
    assert torch.equal(issued[~student_selected], teacher[~student_selected])
    assert torch.equal(
        exploratory_student[~student_selected], student[~student_selected]
    )
    assert torch.equal(
        applied_q_noise[~student_selected],
        torch.zeros_like(applied_q_noise[~student_selected]),
    )
    assert applied_q_noise[student_selected].abs().max().item() <= 0.2

    nominal_q = (student - center) / scale * gain
    exploratory_q = (exploratory_student - center) / scale * gain
    assert torch.allclose(
        exploratory_q[student_selected] - nominal_q[student_selected],
        applied_q_noise[student_selected],
    )
    assert torch.equal(issued[student_selected], exploratory_student[student_selected])
    assert torch.equal(target_rng.get_state(), target_rng_before)


def test_disabled_collector_noise_is_bitwise_noop_and_does_not_advance_rng():
    student = torch.tensor([[1.0, -1.0], [3.0, 3.0]])
    teacher = torch.tensor([[8.0, 7.0], [6.0, 5.0]])
    selected = torch.tensor([True, False])
    generator = torch.Generator().manual_seed(123)
    generator_before = generator.get_state().clone()

    issued, exploratory_student, applied_q_noise = _apply_student_collector_noise(
        student_action=student,
        teacher_action=teacher,
        student_selected_mask=selected,
        noise_std=0.0,
        noise_clip=0.2,
        q_action_center=torch.tensor([1.0, -1.0]),
        q_action_scale=torch.tensor([2.0, 4.0]),
        q_action_gain=0.5,
        action_low=torch.tensor([-10.0, -10.0]),
        action_high=torch.tensor([10.0, 10.0]),
        generator=generator,
    )

    assert torch.equal(exploratory_student, student)
    assert torch.equal(issued[selected], student[selected])
    assert torch.equal(issued[~selected], teacher[~selected])
    assert torch.equal(applied_q_noise, torch.zeros_like(student))
    assert torch.equal(generator.get_state(), generator_before)


def test_teacher_prefill_forces_only_valid_teacher_without_advancing_replay_rngs():
    latent = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    teacher = torch.tensor([[2.0, -3.0], [25.0, 0.0], [-6.0, 7.0]])
    policy = _bare_policy(
        teacher_prefill_max_rollouts=1,
        dagger_control_mode="beta",
        dagger_beta_start=0.25,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=100,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
        collector_exploration_noise_std=0.9,
        collector_exploration_noise_clip=0.5,
        q_action_input_gain=1.0,
    )
    policy.teacher_prefill_rollout_count = 0
    policy.dagger_rollout_count = 0
    policy.dagger_rng = torch.Generator().manual_seed(71)
    policy.collector_exploration_rng = torch.Generator().manual_seed(72)
    policy._student_latent = lambda td: latent.clone()
    policy._teacher_action = lambda td: teacher.clone()
    policy._project_execution_action = lambda action: action.clamp(-20.0, 20.0)
    policy._student_action_from_latent = lambda value: value.tanh() * 20.0
    policy._fastsac_action_low = torch.full((2,), -20.0)
    policy._fastsac_action_high = torch.full((2,), 20.0)
    policy._fastsac_q_action_center = torch.zeros(2)
    policy._fastsac_q_action_scale = torch.ones(2)
    rollout_policy = _DistributionalTD3DaggerRolloutPolicy(policy)
    rollout = TensorDict({"is_init": torch.zeros(3, dtype=torch.bool)}, batch_size=[3])
    dagger_rng_before = policy.dagger_rng.get_state().clone()
    collector_rng_before = policy.collector_exploration_rng.get_state().clone()

    prefill = rollout_policy(rollout.clone())

    valid = torch.tensor([True, False, True])
    student_action = latent.tanh() * 20.0
    assert torch.equal(prefill[DAGGER_TEACHER_ACTION_VALID_KEY], valid)
    assert torch.equal(prefill[DAGGER_IS_STUDENT_ACTION_KEY], ~valid)
    assert torch.equal(prefill[ACTION_KEY][valid], teacher[valid])
    assert torch.equal(prefill[ACTION_KEY][~valid], student_action[~valid])
    assert torch.equal(prefill[TD3_NOISE_FREE_STUDENT_ACTION_KEY], student_action)
    assert torch.equal(prefill[TD3_EXPLORATORY_STUDENT_ACTION_KEY], student_action)
    assert torch.equal(
        prefill[TD3_COLLECTOR_NOISE_KEY], torch.zeros_like(student_action)
    )
    assert not prefill[DAGGER_BETA_TEACHER_KEY].any()
    assert torch.equal(policy.dagger_rng.get_state(), dagger_rng_before)
    assert torch.equal(
        policy.collector_exploration_rng.get_state(), collector_rng_before
    )

    # Completing prefill exposes the untouched main DAgger schedule. Its first
    # rollout reports beta_start and consumes the DAgger RNG normally.
    policy._teacher_prefill_complete = True
    main = rollout_policy(rollout.clone())
    assert torch.all(main[TD3_BETA_KEY] == policy.cfg.dagger_beta_start)
    assert not torch.equal(policy.dagger_rng.get_state(), dagger_rng_before)


def test_eta_zero_and_disabled_noise_preserve_seeded_baseline_dagger_actions():
    latent = torch.tensor([[0.25, -0.5], [0.75, 0.1], [-0.2, 0.4], [0.0, -0.8]])
    teacher = torch.tensor([[2.0, -3.0], [4.0, 5.0], [-6.0, 7.0], [8.0, -9.0]])
    cfg = SimpleNamespace(
        eta_td3=0.0,
        lambda_bc=1.0,
        target_policy_noise_std=0.0,
        target_policy_noise_clip=0.0,
        collector_exploration_noise_std=0.0,
        collector_exploration_noise_clip=0.0,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
        q_action_input_gain=1.0,
    )

    def owner(seed: int):
        value = SimpleNamespace(cfg=cfg)
        value.dagger_rng = torch.Generator().manual_seed(seed)
        value.collector_exploration_rng = torch.Generator().manual_seed(seed + 1)
        value._student_latent = lambda td: latent.clone()
        value._teacher_action = lambda td: teacher.clone()
        value._project_execution_action = lambda action: action.clamp(-20.0, 20.0)
        value._student_action_from_latent = lambda value: value.tanh() * 20.0
        value._teacher_prefill_active = lambda: False
        value._effective_control_mode = lambda: "beta"
        value._teacher_mixture_probability = lambda: 0.5
        value._fastsac_action_low = torch.full((2,), -20.0)
        value._fastsac_action_high = torch.full((2,), 20.0)
        value._fastsac_q_action_center = torch.zeros(2)
        value._fastsac_q_action_scale = torch.ones(2)
        return value

    baseline_owner = owner(123)
    td3_owner = owner(123)
    collector_state = td3_owner.collector_exploration_rng.get_state().clone()
    baseline = _DaggerRolloutPolicy(baseline_owner)
    td3 = _DistributionalTD3DaggerRolloutPolicy(td3_owner)
    rollout = TensorDict({"is_init": torch.zeros(4, dtype=torch.bool)}, batch_size=[4])

    baseline_out = baseline(rollout.clone())
    td3_out = td3(rollout.clone())

    for key in (
        ACTION_KEY,
        DAGGER_TEACHER_ACTION_KEY,
        DAGGER_TEACHER_ACTION_VALID_KEY,
        DAGGER_IS_STUDENT_ACTION_KEY,
        DAGGER_BETA_TEACHER_KEY,
    ):
        assert torch.equal(td3_out[key], baseline_out[key])
    assert torch.equal(
        td3_owner.dagger_rng.get_state(), baseline_owner.dagger_rng.get_state()
    )
    assert torch.equal(td3_owner.collector_exploration_rng.get_state(), collector_state)


class _MeanOnlyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_dist_calls = 0

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluation must not invoke probabilistic forward")

    def get_dist(self, td: TensorDict):
        self.get_dist_calls += 1

        class _Distribution:
            mean = torch.full((*td.batch_size, 1), 0.5)

            def log_prob(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("evaluation must not compute log probability")

        return _Distribution()


def test_evaluation_is_student_only_deterministic_and_noise_free(monkeypatch):
    policy = _bare_policy(dagger_action_clip=1.0, use_object_adapt=False)
    _install_unit_action_contract(policy)
    policy.depth_feature_dim = 1
    policy.adapt_ema = nn.Identity()
    policy.actor_adapt = _MeanOnlyActor()
    policy.collector_exploration_rng = torch.Generator().manual_seed(91)
    collector_state = policy.collector_exploration_rng.get_state().clone()

    def forbidden_teacher(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluation must never evaluate the Teacher")

    policy._teacher_action = forbidden_teacher
    monkeypatch.setattr(
        PPOVEL,
        "get_rollout_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation must bypass probabilistic rollout")
        ),
    )
    evaluation_policy = policy.get_rollout_policy("eval")
    evaluation_policy.eval()

    first = evaluation_policy(TensorDict({}, batch_size=[3]))[ACTION_KEY]
    second = evaluation_policy(TensorDict({}, batch_size=[3]))[ACTION_KEY]

    assert torch.equal(first, second)
    assert torch.allclose(first, torch.full((3, 1), torch.tanh(torch.tensor(0.5))))
    assert torch.equal(policy.collector_exploration_rng.get_state(), collector_state)
    assert policy.actor_adapt.get_dist_calls == 2
    assert policy.actor_adapt.training is False


def _checkpoint_test_policy(seed: int, *, with_actor_target: bool):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = _bare_policy()
        policy.actor_adapt = nn.Linear(2, 1)
        policy.actor_target = (
            copy.deepcopy(policy.actor_adapt).requires_grad_(False)
            if with_actor_target
            else None
        )
        policy.qnet = _TableTwinC51(torch.randn(2, 3), torch.randn(2, 3))
        policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
        policy.adapt_probe = nn.Linear(2, 2)
    policy.actor_optimizer = torch.optim.Adam(
        policy.actor_adapt.parameters(), lr=3.0e-3
    )
    policy.critic_optimizer = torch.optim.Adam(policy.qnet.parameters(), lr=4.0e-3)
    policy.opt_adapt = torch.optim.Adam(policy.adapt_probe.parameters(), lr=5.0e-3)
    policy.actor_update_count = 0
    policy.critic_update_count = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.teacher_prefill_rollout_count = 0
    policy.teacher_prefill_environment_steps = 0
    policy.dagger_rng = torch.Generator().manual_seed(seed + 1)
    policy.q_rng = torch.Generator().manual_seed(seed + 2)
    policy.collector_exploration_rng = torch.Generator().manual_seed(seed + 3)
    policy.target_policy_rng = torch.Generator().manual_seed(seed + 4)
    policy.teacher_perception_rng = torch.Generator().manual_seed(seed + 5)
    policy._last_td3_diagnostics = {}
    return policy


def test_td3_checkpoint_seam_round_trips_all_owned_training_state():
    source = _checkpoint_test_policy(100, with_actor_target=True)
    _take_optimizer_step(source.actor_adapt, source.actor_optimizer)
    _take_optimizer_step(source.qnet, source.critic_optimizer)
    _take_optimizer_step(source.adapt_probe, source.opt_adapt)
    source.actor_update_count = 17
    source.critic_update_count = 39
    source.dagger_rollout_count = 11
    source.dagger_environment_steps = 12_345
    source.teacher_prefill_rollout_count = 7
    source.teacher_prefill_environment_steps = 224
    source._last_td3_diagnostics = {
        "td3/left_support_projection_clipping_fraction": 0.125,
        "td3/right_support_projection_clipping_fraction": 0.25,
    }
    generators = (
        source.dagger_rng,
        source.q_rng,
        source.collector_exploration_rng,
        source.target_policy_rng,
        source.teacher_perception_rng,
    )
    for draw_count, generator in enumerate(generators, start=1):
        torch.rand(draw_count, generator=generator)
    evaluation_observation = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
    source_evaluation = torch.tanh(source.actor_adapt(evaluation_observation))
    state = copy.deepcopy(source._td3_checkpoint_state())

    restored = _checkpoint_test_policy(900, with_actor_target=False)
    restored._load_td3_checkpoint_state(state)

    for module_name in ("actor_adapt", "actor_target", "qnet", "qnet_target"):
        _assert_nested_equal(
            getattr(source, module_name).state_dict(),
            getattr(restored, module_name).state_dict(),
        )
    _assert_nested_equal(
        source.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict()
    )
    _assert_nested_equal(
        source.critic_optimizer.state_dict(),
        restored.critic_optimizer.state_dict(),
    )
    _assert_nested_equal(source.opt_adapt.state_dict(), restored.opt_adapt.state_dict())
    assert restored.actor_update_count == 17
    assert restored.critic_update_count == 39
    assert restored.dagger_rollout_count == 11
    assert restored.dagger_environment_steps == 12_345
    assert restored.teacher_prefill_rollout_count == 7
    assert restored.teacher_prefill_environment_steps == 224
    assert restored._last_td3_diagnostics == source._last_td3_diagnostics

    for name in (
        "dagger_rng",
        "q_rng",
        "collector_exploration_rng",
        "target_policy_rng",
        "teacher_perception_rng",
    ):
        source_generator = getattr(source, name)
        restored_generator = getattr(restored, name)
        assert torch.equal(source_generator.get_state(), restored_generator.get_state())
        assert torch.equal(
            torch.rand(8, generator=source_generator),
            torch.rand(8, generator=restored_generator),
        )

    restored_evaluation = torch.tanh(restored.actor_adapt(evaluation_observation))
    assert torch.equal(restored_evaluation, source_evaluation)
    assert _parameter_storage(restored.actor_adapt).isdisjoint(
        _parameter_storage(restored.actor_target)
    )
    assert _parameter_storage(restored.qnet).isdisjoint(
        _parameter_storage(restored.qnet_target)
    )
    assert all(
        not parameter.requires_grad for parameter in restored.actor_target.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in restored.qnet_target.parameters()
    )


@pytest.mark.parametrize("version", (1, CHECKPOINT_VERSION))
def test_public_load_rejects_old_and_current_same_stage_td3_resume(
    monkeypatch, version
):
    policy = _bare_policy()
    state = {
        "training_algorithm": TRAINING_ALGORITHM,
        "checkpoint_version": version,
    }
    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("same-stage rejection must happen before module loading")
        ),
    )

    with pytest.raises(ValueError, match="fresh-only.*same-stage TD3"):
        policy.load_state_dict(state)
