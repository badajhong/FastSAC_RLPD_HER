from types import SimpleNamespace

import h5py
import pytest
import torch
from torchrl.data import Composite, Unbounded
from tensordict import TensorDict

from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTION_PARAMETERIZATION,
    TEACHER_REPLAY_INITIAL_TRANSITION_FILTER,
    TEACHER_REPLAY_FORMAT_VERSION,
    TRUNCATION_NEXT_OBSERVATION_SEMANTICS,
    OfflineReplayH5,
    TeacherReplayBuffer,
    _sac_bootstrap_mask,
    _vaic_pre_reset_final_observation_mask,
    _vaic_truncation_mask,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from active_adaptation.utils.tensordict import zero_tensordict_rows_


def test_episode_stats_reset_preserves_bool_and_float_leaf_dtypes():
    stats = TensorDict(
        {
            "episode_len": torch.tensor([[1.0], [2.0], [3.0]]),
            "episode_time_limit": torch.tensor([[True], [True], [True]]),
            "nested": TensorDict(
                {"return": torch.tensor([[4.0], [5.0], [6.0]])},
                batch_size=[3],
            ),
        },
        batch_size=[3],
    )
    zero_tensordict_rows_(stats, torch.tensor([0, 2]))

    assert stats["episode_len"].dtype == torch.float32
    assert stats["episode_time_limit"].dtype == torch.bool
    assert torch.equal(
        stats["episode_len"].squeeze(-1), torch.tensor([0.0, 2.0, 0.0])
    )
    assert torch.equal(
        stats["episode_time_limit"].squeeze(-1),
        torch.tensor([False, True, False]),
    )
    assert torch.equal(
        stats["nested", "return"].squeeze(-1), torch.tensor([0.0, 5.0, 0.0])
    )


def test_recurrent_primer_does_not_duplicate_the_environment_batch_dimension():
    policy = PPOVEL.__new__(PPOVEL)
    torch.nn.Module.__init__(policy)
    policy.observation_spec = SimpleNamespace(shape=torch.Size([4096]))
    policy.cfg = SimpleNamespace(latent_dim=256, adapt_module="gru")
    policy.device = torch.device("cpu")
    policy.depth_feature_dim = 64

    primer = policy.make_tensordict_primer()
    observation_spec = Composite(
        marker=Unbounded((4096, 1)), shape=(4096,), device="cpu"
    )
    transformed = primer.transform_observation_spec(observation_spec)

    assert primer.expand_specs is False
    assert transformed["adapt_hx"].shape == torch.Size([4096, 256])


def test_sac_bootstrap_truth_table():
    dones = torch.tensor([False, True, True])
    truncations = torch.tensor([False, True, False])
    assert torch.equal(
        _sac_bootstrap_mask(dones, truncations),
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_only_pure_timeout_bootstraps_and_requires_final_state_capture():
    # ordinary, command completion, episode limit, term+limit, command+limit,
    # term+command. Command completion is a finite-task terminal, while only a
    # pure artificial episode limit bootstraps from its pre-reset final state.
    done = torch.tensor([[False], [True], [True], [True], [True], [True]])
    terminated = torch.tensor(
        [[False], [False], [False], [True], [False], [True]]
    )
    episode_time_limit = torch.tensor(
        [[False], [False], [True], [True], [True], [False]]
    )
    command_finished = torch.tensor(
        [[False], [True], [False], [False], [True], [True]]
    )
    td = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": done,
                    "terminated": terminated,
                    "stats": TensorDict(
                        {
                            "episode_time_limit": episode_time_limit,
                            "command_finished": command_finished,
                        },
                        batch_size=[6],
                    ),
                },
                batch_size=[6],
            )
        },
        batch_size=[6],
    )

    assert torch.equal(
        _vaic_truncation_mask(td).squeeze(-1),
        torch.tensor([False, False, True, False, False, False]),
    )
    assert torch.equal(
        _vaic_pre_reset_final_observation_mask(td).squeeze(-1),
        torch.tensor([False, False, True, False, False, False]),
    )
    truncations = _vaic_truncation_mask(td).squeeze(-1)
    assert torch.equal(
        _sac_bootstrap_mask(done.squeeze(-1), truncations),
        torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )


def test_vaic_truncation_cause_must_be_marked_done():
    td = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.tensor([[False]]),
                    "terminated": torch.tensor([[False]]),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor([[False]]),
                            "command_finished": torch.tensor([[True]]),
                        },
                        batch_size=[1],
                    ),
                },
                batch_size=[1],
            )
        },
        batch_size=[1],
    )

    with pytest.raises(RuntimeError, match="reset-cause row"):
        _vaic_truncation_mask(td)


def test_teacher_replay_records_current_truncation_semantics(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = TeacherReplayBuffer(
        path,
        capacity=8,
        actor_dim=3,
        critic_dim=4,
        action_dim=1,
        seed=0,
        device="cpu",
    )
    replay.append({
        "observations": torch.zeros(1, 3),
        "critic_observations": torch.zeros(1, 4),
        "actions": torch.zeros(1, 1),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1, dtype=torch.bool),
        "truncations": torch.zeros(1, dtype=torch.bool),
        "discounts": torch.ones(1),
        "next_observations": torch.zeros(1, 3),
        "next_critic_observations": torch.zeros(1, 4),
    })
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")
    with h5py.File(path, "r") as snapshot:
        assert int(snapshot.attrs["format_version"]) == TEACHER_REPLAY_FORMAT_VERSION
        assert (
            snapshot.attrs["truncation_next_observation"]
            == TRUNCATION_NEXT_OBSERVATION_SEMANTICS
        )
        assert (
            snapshot.attrs["initial_transition_filter"]
            == TEACHER_REPLAY_INITIAL_TRANSITION_FILTER
        )
        assert (
            snapshot.attrs["action_parameterization"]
            == FASTSAC_ACTION_PARAMETERIZATION
        )

    with h5py.File(path, "r+") as snapshot:
        snapshot.attrs["format_version"] = 1

    with pytest.raises(ValueError, match="legacy format version 1"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=1, device="cpu"
        )


def test_both_replay_loaders_reject_missing_reset_transition_filter(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = TeacherReplayBuffer(
        path,
        capacity=8,
        actor_dim=3,
        critic_dim=4,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="filtered-run",
    )
    replay.append({
        "observations": torch.zeros(1, 3),
        "critic_observations": torch.zeros(1, 4),
        "actions": torch.zeros(1, 1),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1, dtype=torch.bool),
        "truncations": torch.zeros(1, dtype=torch.bool),
        "discounts": torch.ones(1),
        "next_observations": torch.zeros(1, 3),
        "next_critic_observations": torch.zeros(1, 4),
    })
    replay.snapshot(iteration=1, checkpoint_name="checkpoint_1")
    with h5py.File(path, "r+") as snapshot:
        del snapshot.attrs["initial_transition_filter"]

    with pytest.raises(ValueError, match="step_count <= 1"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=1, device="cpu"
        )

    restored = TeacherReplayBuffer(
        tmp_path / "restored.h5",
        capacity=8,
        actor_dim=3,
        critic_dim=4,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="filtered-run",
    )
    with pytest.raises(ValueError, match="step_count <= 1"):
        restored.restore(path)
