from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.envs.transforms import Compose, RenameTransform, VecNorm

from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACVEL,
    _vecnorm_state_fingerprint,
)


def test_copy_before_vecnorm_preserves_exact_raw_observation():
    values = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    td = TensorDict({"policy": values.clone()}, batch_size=[2])
    vecnorm = VecNorm(["policy"], decay=0.9999)
    transform = Compose(
        RenameTransform(
            ["policy"], [("_fastsac_raw", "policy")], create_copy=True
        ),
        vecnorm,
    )

    transformed = transform(td)

    assert torch.equal(transformed["_fastsac_raw", "policy"], values)
    assert not torch.equal(transformed["policy"], values)


class _FixedVecNorm:
    in_keys = ["normalized"]
    eps = 1e-4

    def __init__(self):
        self.loc = TensorDict(
            {"normalized": torch.tensor([10.0, 20.0])}, batch_size=[]
        )
        self.scale = TensorDict(
            {"normalized": torch.tensor([2.0, 4.0])}, batch_size=[]
        )


def test_vecnorm_fingerprint_tracks_exact_fixed_coordinates():
    first = _FixedVecNorm()
    same = _FixedVecNorm()
    changed = _FixedVecNorm()
    changed.scale["normalized"][0] = 3.0

    fingerprint = _vecnorm_state_fingerprint(first)

    assert fingerprint.startswith("sha256:")
    assert fingerprint == _vecnorm_state_fingerprint(same)
    assert fingerprint != _vecnorm_state_fingerprint(changed)


def test_teacher_replay_batch_uses_one_current_vecnorm_snapshot_without_update():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.q_critic_keys = ["normalized", "raw_"]
    policy._q_critic_widths = [2, 1]
    policy._teacher_raw_replay_fields = []
    policy.observation_spec = {
        ("_fastsac_raw", "normalized"): SimpleNamespace()
    }
    vecnorm = _FixedVecNorm()
    policy.configure_replay_vecnorm(vecnorm)
    before_loc = vecnorm.loc.clone()
    before_scale = vecnorm.scale.clone()
    batch = {
        "critic_observations": torch.tensor([[12.0, 24.0, 7.0]]),
        "next_critic_observations": torch.tensor([[8.0, 12.0, 9.0]]),
    }

    prepared = policy._prepare_teacher_learning_batch(batch)

    assert torch.equal(
        prepared["critic_observations"], torch.tensor([[1.0, 1.0, 7.0]])
    )
    assert torch.equal(
        prepared["next_critic_observations"],
        torch.tensor([[-1.0, -2.0, 9.0]]),
    )
    assert torch.equal(vecnorm.loc["normalized"], before_loc["normalized"])
    assert torch.equal(vecnorm.scale["normalized"], before_scale["normalized"])
    # Raw replay remains untouched; preparation returns normalized views/copies.
    assert torch.equal(
        batch["critic_observations"], torch.tensor([[12.0, 24.0, 7.0]])
    )


def test_missing_raw_alias_for_vecnorm_key_fails_collection():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy._replay_vecnorm_keys = {"policy"}
    td = TensorDict({"policy": torch.ones(2, 3)}, batch_size=[2])

    try:
        policy._replay_source(td, "policy")
    except KeyError as exc:
        assert "raw observation alias" in str(exc)
    else:
        raise AssertionError("missing raw replay alias was silently accepted")
