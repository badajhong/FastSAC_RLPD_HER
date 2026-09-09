from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from active_adaptation.learning.ppo.exact_episode_replay import ExactEpisodePrefixStore
from active_adaptation.learning.ppo.exact_online_perception import EXACT_CURRENT_REF
from active_adaptation.learning.ppo.fastsac_bc_dagger import DistributionalFastSACTeacherBC
from active_adaptation.learning.ppo.ppo_vel import (
    OBJECT_GEO_KEY, OBJECT_KEY, OBS_KEY, OBS_PRIV_KEY, PRIV_FEATURE_KEY,
    PRIV_PRED_KEY, VEL_CMD_KEY,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    PERCEPTION_OBJECT_GEO_ID_KEY, REPLAY_SAMPLE_IS_DAGGER_ENV_KEY,
    REPLAY_SAMPLE_IS_TEACHER_KEY, REPLAY_SAMPLE_PHYSICAL_INDEX_KEY,
    TEACHER_EPISODE_STEP_KEY, TEACHER_EPISODE_UID_KEY,
)
from active_adaptation.learning.ppo.teacher_episode_replay import TeacherEpisodeSequenceStore
from active_adaptation.learning.ppo.replay_provenance import prepared_with_provenance_snapshot
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC as TVKD,
)


def _raw(ids):
    return {
        PERCEPTION_OBJECT_GEO_ID_KEY: torch.tensor(ids),
        "is_init": torch.tensor([True] + [False] * (len(ids) - 1)),
        "depth": torch.zeros(len(ids), 3, 4),
    }


def test_exact_store_gathers_one_field_with_duplicates_and_chunk_boundaries():
    store = ExactEpisodePrefixStore(is_init_key="is_init")
    uid = store.allocate_episode_uid()
    values = _raw([1, 2, 3, 4])
    store.append(uid, 0, {key: value[:2] for key, value in values.items()})
    store.append(uid, 2, {key: value[2:] for key, value in values.items()})
    # Unrelated large fields must never be accessed while gathering IDs.
    for chunk in store._episodes[uid].chunks:
        del chunk["depth"]
    with torch.inference_mode():
        result = store.gather_field(
            PERCEPTION_OBJECT_GEO_ID_KEY,
            torch.tensor([[uid, 2], [uid, 0], [uid, 2], [uid, 3]]),
        )
    assert result.tolist() == [3, 1, 3, 4]
    assert not torch.is_inference(result)
    with pytest.raises(IndexError, match="outside its episode"):
        store.gather_field(PERCEPTION_OBJECT_GEO_ID_KEY, torch.tensor([[uid, 4]]))
    with pytest.raises(KeyError, match="lacks field"):
        store.gather_field("missing", torch.tensor([[uid, 0]]))


class _ObjectTransform(nn.Module):
    def forward(self, td):
        td["transformed_object"] = td[OBJECT_KEY] * td[OBJECT_GEO_KEY]
        return td


class _PrivilegedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, td):
        td[PRIV_FEATURE_KEY] = self.weight * td[OBS_PRIV_KEY] + td["transformed_object"]
        return td


def _policy_and_batch():
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(actor_consistency_coef=1.0, actor_gt_bc_coef=0.0)
    policy.q_actor_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
    policy._q_actor_widths = [1, 2, 1]
    policy.q_critic_keys = [OBS_PRIV_KEY, OBS_KEY, "command_", OBJECT_KEY]
    policy._q_critic_widths = [1, 2, 1, 1]
    policy.object_transform = _ObjectTransform()
    policy.encoder_priv = _PrivilegedEncoder()
    policy._decode_replay_object_geo = lambda ids, *, device, dtype: ids.to(device=device, dtype=dtype).unsqueeze(-1)

    teacher_store = TeacherEpisodeSequenceStore(is_init_key="is_init")
    first_teacher = teacher_store.allocate_episode_uid()
    second_teacher = teacher_store.allocate_episode_uid()
    teacher_store.commit_successful_episode(first_teacher, _raw([10, 11]))
    teacher_store.commit_successful_episode(second_teacher, _raw([20, 21, 22]))
    teacher_uids = torch.tensor([second_teacher, first_teacher])
    teacher_steps = torch.tensor([1, 0])
    teacher_store.freeze(teacher_uids, teacher_steps)
    policy._teacher_episode_store = teacher_store
    policy.q_teacher_replay = SimpleNamespace(size=2, device=policy.device, data={
        TEACHER_EPISODE_UID_KEY: teacher_uids,
        TEACHER_EPISODE_STEP_KEY: teacher_steps,
    })
    online_store = ExactEpisodePrefixStore(is_init_key="is_init")
    first_online = online_store.allocate_episode_uid()
    second_online = online_store.allocate_episode_uid()
    online_store.append(first_online, 0, _raw([30, 31, 32]))
    online_store.append(second_online, 0, _raw([40, 41]))
    policy._exact_online_store = online_store
    policy.dagger_replay = SimpleNamespace(size=2, device=policy.device, data={
        EXACT_CURRENT_REF: torch.tensor([[first_online, 2], [second_online, 0]]),
    })
    policy.student_replay = SimpleNamespace(size=2, device=policy.device, data={
        EXACT_CURRENT_REF: torch.tensor([[second_online, 1], [first_online, 1]]),
    })
    batch = {
        "observations": torch.randn(7, 4, requires_grad=True),
        "critic_observations": torch.randn(7, 5, requires_grad=True),
        REPLAY_SAMPLE_IS_TEACHER_KEY: torch.tensor([False, True, False, False, True, False, True]),
        REPLAY_SAMPLE_IS_DAGGER_ENV_KEY: torch.tensor([False, False, True, False, False, True, False]),
        REPLAY_SAMPLE_PHYSICAL_INDEX_KEY: torch.tensor([0, 1, 1, 1, 0, 0, 1]),
    }
    return policy, batch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_oracle_inputs_preserve_same_state_and_align_all_three_replay_sources(device):
    policy, batch = _policy_and_batch()
    policy.device = torch.device(device)
    policy.encoder_priv.to(device)
    batch = {
        key: value.detach().to(device).requires_grad_(value.requires_grad)
        for key, value in batch.items()
    }
    original = {key: value.detach().clone() for key, value in batch.items()}
    with torch.inference_mode():
        oracle = policy._actor_oracle_observations(batch)
    geometry = torch.tensor([41, 10, 40, 31, 21, 32, 10], dtype=torch.float32, device=device)
    critic = batch["critic_observations"]
    expected_latent = 2 * critic[:, 0] + critic[:, -1] * geometry
    torch.testing.assert_close(oracle[:, -1], expected_latent)
    torch.testing.assert_close(oracle[:, :-1], batch["observations"][:, :-1], rtol=0, atol=0)
    assert not oracle.requires_grad and not torch.is_inference(oracle)
    assert policy.encoder_priv.weight.grad is None
    for key in batch:
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)
    # An actor can save the reconstructed tensor for its own backward.
    actor = nn.Linear(4, 1).to(device)
    actor(oracle).square().mean().backward()
    assert actor.weight.grad is not None
    assert batch["observations"].grad is None
    assert batch["critic_observations"].grad is None
    assert policy.encoder_priv.weight.grad is None


def test_oracle_reconstruction_rejects_unavailable_or_misaligned_state():
    policy, batch = _policy_and_batch()
    policy.height_encoder = nn.Identity()
    with pytest.raises(ValueError, match="does not store height maps"):
        policy._actor_oracle_observations(batch)
    del policy.height_encoder
    missing = dict(batch)
    del missing[REPLAY_SAMPLE_PHYSICAL_INDEX_KEY]
    with pytest.raises(KeyError, match="sample provenance"):
        policy._actor_oracle_observations(missing)
    batch[REPLAY_SAMPLE_PHYSICAL_INDEX_KEY][0] = 2
    with pytest.raises(IndexError, match="outside its replay ring"):
        policy._actor_oracle_observations(batch)


def test_actor_oracle_reuses_prepared_provenance_and_revalidates_mutation(monkeypatch):
    policy, batch = _policy_and_batch()
    expected = policy._actor_oracle_observations(batch)
    keys = (REPLAY_SAMPLE_IS_TEACHER_KEY, REPLAY_SAMPLE_IS_DAGGER_ENV_KEY,
            REPLAY_SAMPLE_PHYSICAL_INDEX_KEY)
    prepared, _ = prepared_with_provenance_snapshot(dict(batch), batch, keys)
    original_keys = list(batch)
    # Cached metadata must not be copied again by Actor GT construction.
    metadata_storage = {batch[key].untyped_storage().data_ptr() for key in keys}
    original_cpu = torch.Tensor.cpu

    def no_second_metadata_copy(value, *args, **kwargs):
        assert value.untyped_storage().data_ptr() not in metadata_storage
        return original_cpu(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", no_second_metadata_copy)
        actual = policy._actor_oracle_observations(prepared)
    assert list(prepared) == original_keys
    assert torch.equal(actual, expected)
    assert all(prepared[key] is batch[key] for key in original_keys)
    prepared[REPLAY_SAMPLE_PHYSICAL_INDEX_KEY][0] = 2
    with pytest.raises(IndexError, match="outside its replay ring"):
        policy._actor_oracle_observations(prepared)


@pytest.mark.parametrize("consistency,gt_bc", [(1, 0), (0, 1)])
def test_height_input_requirement_is_checked_before_rollout(monkeypatch, consistency, gt_bc):
    policy, _ = _policy_and_batch()
    policy.height_encoder = nn.Identity()
    policy.cfg.actor_consistency_coef = consistency
    policy.cfg.actor_gt_bc_coef = gt_bc
    monkeypatch.setattr(DistributionalFastSACTeacherBC, "__init__", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="does not store height maps"):
        TVKD.__init__(policy, policy.cfg, None, None, None, policy.device, None)


@pytest.mark.parametrize("consistency,gt_bc", [(0, 0), (1, 0), (0, 1)])
def test_actor_sampling_materializes_gt_only_when_enabled(monkeypatch, consistency, gt_bc):
    policy, batch = _policy_and_batch()
    policy.cfg.actor_consistency_coef = consistency
    policy.cfg.actor_gt_bc_coef = gt_bc
    monkeypatch.setattr(DistributionalFastSACTeacherBC, "_sample_actor_batch", lambda self: dict(batch))
    prepared = policy._sample_actor_batch()
    assert ("actor_gt_observations" in prepared) == bool(consistency or gt_bc)
    assert "actor_gt_observations" not in batch
