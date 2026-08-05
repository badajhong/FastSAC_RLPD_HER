from types import SimpleNamespace

import h5py
import pytest
import torch
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.ppo.fastsac_vel import (
    TEACHER_REPLAY_FORMAT_VERSION,
    OfflineReplayH5,
    TeacherReplayBuffer,
    _timeout_bootstrap_mask,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL


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


def test_timeout_bootstrap_truth_table():
    dones = torch.tensor([False, True, True])
    truncations = torch.tensor([False, True, False])
    assert torch.equal(
        _timeout_bootstrap_mask(dones, truncations),
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_teacher_replay_rejects_legacy_timeout_semantics(tmp_path):
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
        assert snapshot.attrs["timeout_next_observation"] == "pre_reset_final"

    with h5py.File(path, "r+") as snapshot:
        snapshot.attrs["format_version"] = 1

    with pytest.raises(ValueError, match="legacy format version 1"):
        OfflineReplayH5(
            path, actor_dim=3, critic_dim=4, action_dim=1, device="cpu"
        )
