from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_vel import (
    CMD_KEY,
    DEPTH_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    PPOVEL,
)
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    DAGGER_IS_STUDENT_ACTION_KEY,
    DAGGER_REPLAY_TEACHER_ACTIONS,
    DAGGER_TEACHER_ACTION_KEY,
    DAGGER_TEACHER_ACTION_VALID_KEY,
    _DaggerRolloutPolicy,
    _DaggerTeacherReplayBuffer,
    _DeviceReplay,
    PPOBCDaggerFinetune,
    PPOBCDaggerFinetuneConfig,
    _valid_teacher_action_rows,
)
from active_adaptation.learning.ppo.fastsac_vel import (
    BCDaggerOfflineReplayH5,
    TEACHER_REPLAY_FIELDS,
)


EXPECTED_DAGGER_ALGORITHM = "vaic_ppo_bc_dagger_student_v1"
TEACHER_ACTION_KEY = "teacher_action"
TEACHER_ACTION_VALID_KEY = "teacher_action_valid"
IS_STUDENT_ACTION_KEY = "is_student_action"


def _bare_policy(**cfg):
    policy = PPOBCDaggerFinetune.__new__(PPOBCDaggerFinetune)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(**cfg)
    return policy


def test_config_store_registers_exact_student_observation_surface():
    store = ConfigStore.instance()
    dagger = store.load("algo/ppo_bc_dagger_finetune.yaml").node
    ppo_student = store.load("algo/ppo_vel_finetune.yaml").node

    assert dagger._target_.endswith(
        "ppo_bc_dagger.PPOBCDaggerFinetune"
    )
    assert dagger.name == "ppo_bc_dagger"
    assert dagger.phase == "finetune"
    assert dagger.vecnorm == "eval"
    assert dagger.use_depth is True
    assert dagger.enable_residual_distillation is False
    assert list(dagger.in_keys) == list(ppo_student.in_keys)
    assert list(dagger.in_keys) == [
        CMD_KEY,
        OBS_KEY,
        OBJECT_KEY,
        OBS_PRIV_KEY,
        OBJECT_GEO_KEY,
        VEL_CMD_KEY,
        DEPTH_KEY,
    ]

    assert dagger.dagger_beta_start == pytest.approx(1.0)
    assert dagger.dagger_beta_end == pytest.approx(0.0)
    assert dagger.dagger_beta_decay_rollouts == 4000
    assert dagger.dagger_bc_lr > 0.0
    assert dagger.dagger_bc_epochs > 0
    assert dagger.dagger_actor_huber_delta == pytest.approx(1.0)
    assert dagger.dagger_action_clip == pytest.approx(20.0)
    assert dagger.dagger_batch_size == 4096
    assert dagger.dagger_replay_raw_observations is True
    assert set(dagger.replay_raw_observation_keys) == {
        VEL_CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        CMD_KEY,
    }
    assert DEPTH_KEY not in dagger.replay_raw_observation_keys
    assert dagger.save_teacher_buffer is True
    assert dagger.q_num_atoms > 1
    assert dagger.q_v_min < dagger.q_v_max


def test_config_dataclass_has_stage_local_beta_and_separate_q_targets():
    cfg = PPOBCDaggerFinetuneConfig()

    for name in (
        "dagger_beta_start",
        "dagger_beta_end",
        "dagger_beta_decay_rollouts",
        "dagger_seed",
        "dagger_bc_lr",
        "dagger_bc_epochs",
        "dagger_actor_huber_delta",
        "dagger_buffer_capacity",
        "dagger_buffer_device",
        "dagger_batch_size",
        "dagger_updates_per_rollout",
        "q_hidden_dim",
        "q_num_atoms",
        "q_v_min",
        "q_v_max",
        "q_layer_norm",
        "q_lr",
        "q_weight_decay",
        "q_seed",
        "q_tau",
        "q_max_grad_norm",
    ):
        assert hasattr(cfg, name), name


@pytest.mark.parametrize(
    ("actions", "threshold", "expected"),
    (
        (
            [[0.0, 1.0], [20.0, -20.0], [20.01, 0.0]],
            20.0,
            [True, True, False],
        ),
        (
            [[float("nan"), 0.0], [float("inf"), 0.0], [1.0, 2.0]],
            20.0,
            [False, False, True],
        ),
    ),
)
def test_teacher_action_validity_rejects_nonfinite_and_outlier_rows(
    actions, threshold, expected
):
    actual = _valid_teacher_action_rows(
        torch.tensor(actions), threshold=threshold
    )

    assert actual.dtype is torch.bool
    assert actual.shape == (len(actions),)
    assert torch.equal(actual, torch.tensor(expected))


def test_beta_schedule_uses_stage_local_rollouts_and_has_exact_endpoints():
    policy = _bare_policy(
        dagger_beta_start=1.0,
        dagger_beta_end=0.1,
        dagger_beta_decay_rollouts=10,
    )
    # The inherited environment progress can be 6102 when bootstrapping from
    # the supplied PPO teacher. It must not advance the new DAgger schedule.
    policy.env = SimpleNamespace(current_iter=6102)

    policy.dagger_rollout_count = 0
    assert policy._teacher_mixture_probability() == pytest.approx(1.0)

    policy.dagger_rollout_count = 10
    assert policy._teacher_mixture_probability() == pytest.approx(0.1)

    policy.dagger_rollout_count = 10_000
    assert policy._teacher_mixture_probability() == pytest.approx(0.1)


class _NoOpTensorDictModule(nn.Module):
    def forward(self, td):
        return td


class _ResidualTeacher(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = nn.Parameter(torch.as_tensor(residual).float())

    def get_dist(self, td):
        mean = self.residual.expand(*td.batch_size, -1)
        return SimpleNamespace(mean=mean)


def test_teacher_oracle_restores_absolute_ppo_action_and_is_detached():
    policy = _bare_policy()
    policy.object_transform = _NoOpTensorDictModule()
    policy.encoder_priv = _NoOpTensorDictModule()
    policy.actor = _ResidualTeacher([0.2, -0.3])
    td = TensorDict(
        {REF_JPOS_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        batch_size=[2],
    )

    teacher_action = policy._teacher_action(td)

    assert torch.allclose(
        teacher_action,
        torch.tensor([[1.2, 1.7], [3.2, 3.7]]),
    )
    assert teacher_action.requires_grad is False
    assert policy.actor.residual.grad is None


def test_rollout_uses_exact_source_choice_clips_actions_and_falls_back():
    policy = _bare_policy(
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_beta_decay_rollouts=4000,
        dagger_teacher_action_threshold=20.0,
        dagger_action_clip=20.0,
    )
    policy.dagger_rollout_count = 0
    policy.dagger_rng = torch.Generator().manual_seed(0)
    policy._student_action = lambda td: torch.tensor(
        [[30.0, float("nan")], [-3.0, 4.0]]
    )
    # Row zero is valid and must be executed exactly. Row one is an outlier,
    # so beta=1 still falls back to the bounded student action.
    policy._teacher_action = lambda td: torch.tensor(
        [[1.5, -2.5], [20.01, 0.0]]
    )
    td = TensorDict({}, batch_size=[2])

    _DaggerRolloutPolicy(policy)(td)

    assert torch.equal(
        td[ACTION_KEY], torch.tensor([[1.5, -2.5], [-3.0, 4.0]])
    )
    assert torch.equal(
        td[DAGGER_TEACHER_ACTION_VALID_KEY], torch.tensor([True, False])
    )
    assert torch.equal(
        td[DAGGER_IS_STUDENT_ACTION_KEY], torch.tensor([False, True])
    )
    # The invalid label is retained only as a clipped, masked diagnostic.
    assert torch.equal(
        td[DAGGER_TEACHER_ACTION_KEY],
        torch.tensor([[1.5, -2.5], [20.0, 0.0]]),
    )


def test_bc_replay_sampler_draws_only_valid_teacher_labels():
    replay = _DeviceReplay(capacity=8, device="cpu")
    replay.extend(
        {
            "row": torch.arange(6),
            DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor(
                [False, True, False, True, False, False]
            ),
        }
    )

    sampled = replay.sample(
        128,
        "cpu",
        generator=torch.Generator().manual_seed(3),
        valid_key=DAGGER_TEACHER_ACTION_VALID_KEY,
    )

    assert replay.valid_count(DAGGER_TEACHER_ACTION_VALID_KEY) == 2
    assert sampled[DAGGER_TEACHER_ACTION_VALID_KEY].all()
    assert set(sampled["row"].tolist()) == {1, 3}


def _cat_fields(td, keys):
    return torch.cat([td[key] for key in keys], dim=-1)


def test_replay_keeps_executed_action_separate_from_teacher_bc_label():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._cat_replay_sources = _cat_fields
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[99.0]]),
        "next_critic_observations": torch.tensor([[199.0]]),
    }
    policy._truncation_final_batches = []

    rollout = TensorDict(
        {
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[101.0], [102.0]]]),
            ACTION_KEY: torch.tensor([[[3.0], [4.0]]]),
            TEACHER_ACTION_KEY: torch.tensor([[[30.0], [40.0]]]),
            TEACHER_ACTION_VALID_KEY: torch.tensor([[True, True]]),
            IS_STUDENT_ACTION_KEY: torch.tensor([[False, True]]),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.tensor([[[0.25, 0.75], [1.0, 2.0]]]),
                    "done": torch.zeros(1, 2, 1, dtype=torch.bool),
                    "terminated": torch.zeros(1, 2, 1, dtype=torch.bool),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(
                                1, 2, 1, dtype=torch.bool
                            ),
                            "command_finished": torch.zeros(
                                1, 2, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))
    transitions = {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in chunks[0]
    }

    assert torch.equal(transitions["actions"], torch.tensor([[3.0], [4.0]]))
    assert torch.equal(
        transitions["teacher_actions"], torch.tensor([[30.0], [40.0]])
    )
    assert torch.equal(
        transitions["teacher_action_valid"], torch.tensor([True, True])
    )
    assert torch.equal(
        transitions["is_student_action"], torch.tensor([False, True])
    )
    assert torch.equal(transitions["rewards"], torch.tensor([1.0, 3.0]))
    # This is a Bernoulli source choice, not numeric teacher/student blending:
    # every recorded source row is exactly teacher or exactly student.
    assert transitions["is_student_action"].dtype is torch.bool


def test_timeout_uses_true_final_state_while_command_completion_is_terminal():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._cat_replay_sources = _cat_fields
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[99.0]]),
        "next_critic_observations": torch.tensor([[199.0]]),
    }
    policy._truncation_final_batches = [
        {
            "indices": torch.tensor([0]),
            "next_observations": torch.tensor([[77.0]]),
            "next_critic_observations": torch.tensor([[177.0]]),
        }
    ]
    done = torch.ones(1, 2, 1, dtype=torch.bool)
    rollout = TensorDict(
        {
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[101.0], [102.0]]]),
            ACTION_KEY: torch.tensor([[[3.0], [4.0]]]),
            TEACHER_ACTION_KEY: torch.tensor([[[30.0], [40.0]]]),
            TEACHER_ACTION_VALID_KEY: torch.ones(1, 2, dtype=torch.bool),
            IS_STUDENT_ACTION_KEY: torch.zeros(1, 2, dtype=torch.bool),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.ones(1, 2, 1),
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [[[True], [False]]]
                            ),
                            "command_finished": torch.tensor(
                                [[[False], [True]]]
                            ),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))

    assert torch.equal(chunks[0]["next_observations"], torch.tensor([[77.0]]))
    assert torch.equal(chunks[1]["next_observations"], torch.tensor([[99.0]]))
    assert chunks[0]["truncations"].item() is True
    assert chunks[1]["truncations"].item() is False
    assert policy._last_truncation_finals_used == 1


def test_dagger_transition_ring_uses_raw_current_next_and_timeout_final():
    policy = _bare_policy(train_every=2)
    policy.q_actor_keys = ["actor_obs"]
    policy.q_critic_keys = ["critic_obs"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy._replay_vecnorm_keys = {"actor_obs", "critic_obs"}
    policy._scalarize_q_reward = lambda reward: reward.sum(dim=-1)
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[999.0]]),
        "next_critic_observations": torch.tensor([[1999.0]]),
    }
    policy._truncation_final_batches = [
        {
            "indices": torch.tensor([0]),
            "next_observations": torch.tensor([[777.0]]),
            "next_critic_observations": torch.tensor([[1777.0]]),
        }
    ]
    done = torch.tensor([[[True], [False]]])
    rollout = TensorDict(
        {
            # Main fields are post-VecNorm and deliberately very different.
            "actor_obs": torch.tensor([[[1.0], [2.0]]]),
            "critic_obs": torch.tensor([[[11.0], [12.0]]]),
            "_fastsac_raw": TensorDict(
                {
                    "actor_obs": torch.tensor([[[101.0], [102.0]]]),
                    "critic_obs": torch.tensor([[[111.0], [112.0]]]),
                },
                batch_size=[1, 2],
            ),
            ACTION_KEY: torch.zeros(1, 2, 1),
            TEACHER_ACTION_KEY: torch.zeros(1, 2, 1),
            TEACHER_ACTION_VALID_KEY: torch.ones(1, 2, dtype=torch.bool),
            IS_STUDENT_ACTION_KEY: torch.zeros(1, 2, dtype=torch.bool),
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": torch.zeros(1, 2, 1),
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.ones(1, 2, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": done,
                            "command_finished": torch.zeros_like(done),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            ),
        },
        batch_size=[1, 2],
    )

    chunks = list(policy._dagger_transition_chunks(rollout))

    assert torch.equal(chunks[0]["observations"], torch.tensor([[101.0]]))
    assert torch.equal(chunks[1]["observations"], torch.tensor([[102.0]]))
    # Timeout uses the captured pre-reset final state, not step 1/reset state.
    assert torch.equal(chunks[0]["next_observations"], torch.tensor([[777.0]]))
    # Last non-timeout row uses the raw rollout carry captured after the loop.
    assert torch.equal(chunks[1]["next_observations"], torch.tensor([[999.0]]))


def test_dagger_sample_time_normalization_is_once_and_non_mutating():
    policy = _bare_policy()
    policy.q_actor_keys = ["normalized", "latent"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_widths = [2, 1]
    policy._q_critic_widths = [2]
    policy._replay_vecnorm_keys = {"normalized", "critic"}
    object.__setattr__(
        policy,
        "_replay_vecnorm",
        SimpleNamespace(
            eps=1e-4,
            loc={
                "normalized": torch.tensor([10.0, 20.0]),
                "critic": torch.tensor([100.0, 200.0]),
            },
            scale={
                "normalized": torch.tensor([2.0, 4.0]),
                "critic": torch.tensor([10.0, 20.0]),
            },
        ),
    )
    batch = {
        "observations": torch.tensor([[12.0, 24.0, 7.0]]),
        "next_observations": torch.tensor([[8.0, 12.0, 9.0]]),
        "critic_observations": torch.tensor([[110.0, 220.0]]),
        "next_critic_observations": torch.tensor([[90.0, 160.0]]),
        "actions": torch.ones(1, 1),
    }
    original = {key: value.clone() for key, value in batch.items()}

    prepared = policy._prepare_dagger_learning_batch(batch)

    assert torch.equal(
        prepared["observations"], torch.tensor([[1.0, 1.0, 7.0]])
    )
    assert torch.equal(
        prepared["next_observations"], torch.tensor([[-1.0, -2.0, 9.0]])
    )
    assert torch.equal(
        prepared["critic_observations"], torch.tensor([[1.0, 1.0]])
    )
    assert torch.equal(
        prepared["next_critic_observations"], torch.tensor([[-1.0, -2.0]])
    )
    assert torch.equal(prepared["actions"], batch["actions"])
    for key, value in original.items():
        assert torch.equal(batch[key], value)


class _NoTrainingReplay:
    def __init__(self):
        self.size = 0
        self.seen = 0
        self.rows = None

    def extend(self, rows):
        self.rows = rows
        count = rows["rewards"].shape[0]
        self.seen += count
        return count


class _TeacherExportRecorder:
    device = torch.device("cpu")

    def __init__(self):
        self.rows = None

    def append(self, rows):
        self.rows = rows
        return rows["rewards"].shape[0]


def test_train_op_never_calls_ppo_and_exports_only_teacher_executed_rows():
    policy = _bare_policy(
        train_every=2,
        dagger_updates_per_rollout=1,
        dagger_bc_epochs=1,
        dagger_batch_size=1,
    )
    transitions = {
        "observations": torch.zeros(3, 1),
        "critic_observations": torch.zeros(3, 1),
        "actions": torch.tensor([[10.0], [20.0], [30.0]]),
        "rewards": torch.zeros(3),
        "dones": torch.zeros(3, dtype=torch.bool),
        "truncations": torch.zeros(3, dtype=torch.bool),
        "discounts": torch.ones(3),
        "next_observations": torch.zeros(3, 1),
        "next_critic_observations": torch.zeros(3, 1),
        DAGGER_REPLAY_TEACHER_ACTIONS: torch.tensor(
            [[10.0], [21.0], [31.0]]
        ),
        DAGGER_TEACHER_ACTION_VALID_KEY: torch.tensor([True, True, False]),
        DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor([False, True, False]),
    }
    policy._dagger_transition_chunks = lambda td: iter([transitions])
    policy._teacher_mixture_probability = lambda: 0.75
    policy.dagger_replay = _NoTrainingReplay()
    policy.teacher_replay = _TeacherExportRecorder()
    policy.train_adapt = lambda td: {"adapt/called": 1.0}
    policy.train_policy = lambda td: pytest.fail("PPO train_policy was called")
    policy._compute_advantage = lambda *args, **kwargs: pytest.fail(
        "GAE was called"
    )
    policy._update_ppo = lambda td: pytest.fail("PPO update was called")
    policy.num_updates = 0
    policy.dagger_rollout_count = 0
    policy.dagger_environment_steps = 0
    policy.bc_update_count = 0
    policy.q_update_count = 0
    policy._last_truncation_finals_used = 0

    info = policy.train_op(TensorDict({}, batch_size=[1, 2]))

    assert info["adapt/called"] == pytest.approx(1.0)
    assert info["dagger/beta"] == pytest.approx(0.75)
    assert info["dagger/teacher_exported"] == 1
    assert torch.equal(
        policy.teacher_replay.rows["actions"], torch.tensor([[10.0]])
    )
    assert set(policy.teacher_replay.rows) == set(TEACHER_REPLAY_FIELDS)


def test_teacher_h5_roundtrip_has_truthful_dagger_manifest(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    fingerprint = "sha256:" + "a" * 64
    replay = _DaggerTeacherReplayBuffer(
        path,
        capacity=4,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="paired-replay",
        actor_obs_keys=["a", "b"],
        critic_obs_keys=["c"],
        vecnorm_fingerprint=fingerprint,
    )
    rows = {
        "observations": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "critic_observations": torch.arange(
            12, dtype=torch.float32
        ).reshape(4, 3),
        "actions": torch.arange(4, dtype=torch.float32).reshape(4, 1),
        "rewards": torch.arange(4, dtype=torch.float32),
        "dones": torch.tensor([False, False, True, False]),
        "truncations": torch.tensor([False, False, True, False]),
        "discounts": torch.ones(4),
        "next_observations": torch.ones(4, 2),
        "next_critic_observations": torch.ones(4, 3),
    }
    replay.append(rows)
    replay.snapshot(iteration=7, checkpoint_name="checkpoint_7")
    metadata = replay.checkpoint_metadata()

    restored = _DaggerTeacherReplayBuffer(
        tmp_path / "new.h5",
        capacity=4,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        seed=0,
        device="cpu",
        replay_id="paired-replay",
        actor_obs_keys=["a", "b"],
        critic_obs_keys=["c"],
        vecnorm_fingerprint=fingerprint,
    )
    restored.restore(path, expected_metadata=metadata)

    assert restored.size == 4
    assert restored.seen == 4
    assert torch.equal(restored.data["actions"], rows["actions"])
    assert metadata["actor_backend"].startswith("vaic_ppo_")
    assert metadata["replay_observation_semantics"] == (
        "raw_pre_vecnorm_sample_current_v1"
    )
    assert metadata["format_version"] == 2
    assert metadata["vecnorm_fingerprint"] == fingerprint

    offline = BCDaggerOfflineReplayH5(
        path,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        expected_actor_obs_keys=["a", "b"],
        expected_critic_obs_keys=["c"],
        expected_vecnorm_fingerprint=fingerprint,
    )
    assert offline.size == 4
    assert offline.observations_pre_normalized is False
    assert torch.equal(offline.data["actions"], rows["actions"])
    with pytest.raises(ValueError, match="fingerprint"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=2,
            critic_dim=3,
            action_dim=1,
            expected_vecnorm_fingerprint="sha256:" + "f" * 64,
        )


def test_stage2_accepts_legacy_normalized_dagger_h5_but_rejects_mixed_schema(
    tmp_path,
):
    import h5py

    path = tmp_path / "teacher_replay_buffer.h5"
    fingerprint = "sha256:" + "b" * 64
    replay = _DaggerTeacherReplayBuffer(
        path,
        capacity=2,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        seed=0,
        device="cpu",
        actor_obs_keys=["actor"],
        critic_obs_keys=["critic"],
        vecnorm_fingerprint=fingerprint,
    )
    replay.append(
        {
            "observations": torch.tensor([[1.0], [2.0]]),
            "critic_observations": torch.tensor([[3.0], [4.0]]),
            "actions": torch.zeros(2, 1),
            "rewards": torch.zeros(2),
            "dones": torch.zeros(2, dtype=torch.bool),
            "truncations": torch.zeros(2, dtype=torch.bool),
            "discounts": torch.ones(2),
            "next_observations": torch.tensor([[5.0], [6.0]]),
            "next_critic_observations": torch.tensor([[7.0], [8.0]]),
        }
    )
    replay.snapshot(1, "checkpoint_1")
    with h5py.File(path, "r+") as h5:
        h5.attrs["format_version"] = 1
        h5.attrs["replay_observation_semantics"] = (
            "normalized_frozen_vecnorm_v1"
        )
        del h5.attrs["vecnorm_fingerprint"]

    legacy = BCDaggerOfflineReplayH5(
        path,
        actor_dim=1,
        critic_dim=1,
        action_dim=1,
        expected_actor_obs_keys=["actor"],
        expected_critic_obs_keys=["critic"],
    )
    assert legacy.observations_pre_normalized is True

    with h5py.File(path, "r+") as h5:
        h5.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="Unsupported.*schema"):
        BCDaggerOfflineReplayH5(
            path,
            actor_dim=1,
            critic_dim=1,
            action_dim=1,
        )


class _TinyTwinC51(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))
        self.last_actions = None

    def forward(self, obs, actions):
        self.last_actions = actions.detach().clone()
        return self.logits[:, None, :].expand(2, obs.shape[0], 3)

    def values(self, logits):
        return (logits.softmax(-1) * self.support).sum(-1)

    @torch.no_grad()
    def projection(self, obs, actions, reward, bootstrap, discount):
        # Deliberately different head targets. Collapsing to a clipped/minimum
        # target would make the online twins receive the same gradient.
        batch = obs.shape[0]
        target = torch.zeros(2, batch, 3, device=obs.device)
        target[0, :, 0] = 1.0
        target[1, :, 2] = 1.0
        return target


class _StudentMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def get_dist_from_flat(self, obs):
        return SimpleNamespace(mean=self.linear(obs))


def test_q_update_uses_actual_actions_independent_twin_targets_and_no_actor_grad():
    policy = _bare_policy(gamma=0.99, q_tau=0.5, q_max_grad_norm=0.0)
    policy.device = torch.device("cpu")
    policy.qnet = _TinyTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.2)
    policy.actor_adapt = _StudentMean()
    policy._actor_dist_from_flat = (
        lambda obs: policy.actor_adapt.get_dist_from_flat(obs)
    )
    policy.q_update_count = 0

    batch = {
        "observations": torch.zeros(2, 2),
        "critic_observations": torch.zeros(2, 2),
        "actions": torch.tensor([[0.25], [-0.75]]),
        "rewards": torch.tensor([0.1, 0.2]),
        "dones": torch.tensor([False, False]),
        "truncations": torch.tensor([False, False]),
        "discounts": torch.ones(2),
        "next_observations": torch.ones(2, 2),
        "next_critic_observations": torch.ones(2, 2),
    }
    actor_before = policy.actor_adapt.linear.weight.detach().clone()

    policy._q_update(batch)

    assert torch.equal(policy.qnet.last_actions, batch["actions"])
    assert not torch.equal(policy.qnet.logits[0], policy.qnet.logits[1])
    assert not torch.equal(
        policy.qnet_target.logits[0], policy.qnet_target.logits[1]
    )
    assert torch.equal(policy.actor_adapt.linear.weight, actor_before)
    assert policy.actor_adapt.linear.weight.grad is None
    assert policy.q_update_count == 1


def test_legacy_ppo_bootstrap_accepts_fresh_q_but_same_stage_requires_both_qs(
    monkeypatch,
):
    policy = _bare_policy(
        q_seed=7,
        dagger_seed=11,
        q_tau=0.05,
        use_object_adapt=False,
    )
    policy.qnet = nn.Linear(2, 1, bias=False)
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    policy.q_update_count = 123
    policy.dagger_rollout_count = 456
    policy.q_rng = torch.Generator().manual_seed(99)
    policy.dagger_rng = torch.Generator().manual_seed(100)

    monkeypatch.setattr(
        PPOVEL,
        "load_state_dict",
        lambda self, state_dict, strict=True: [
            "depth_cnn",
            "temporal_depth_gru",
            "temporal_depth_gru_ema",
            "qnet",
            "qnet_target",
        ],
    )

    failed = policy.load_state_dict({"last_phase": "train"})

    assert set(failed).issuperset({"qnet", "qnet_target"})
    assert policy.q_update_count == 0
    assert policy.dagger_rollout_count == 0
    assert torch.equal(
        policy.qnet.weight, policy.qnet_target.weight
    )

    same_stage_without_target = {
        "training_algorithm": EXPECTED_DAGGER_ALGORITHM,
        "last_phase": "finetune",
        "qnet": policy.qnet.state_dict(),
    }
    with pytest.raises((KeyError, ValueError, RuntimeError), match="qnet_target|target"):
        policy.load_state_dict(same_stage_without_target)


def test_state_dict_names_online_and_target_q_separately():
    # This is deliberately a topology-level assertion. A single aliased module
    # cannot satisfy the user's requirement to save Q1/Q2 and target Q1/Q2.
    policy = _bare_policy()
    policy.qnet = _TinyTwinC51()
    policy.qnet_target = copy.deepcopy(policy.qnet).requires_grad_(False)
    children = dict(policy.named_children())

    assert children["qnet"] is not children["qnet_target"]
    assert set(children["qnet"].state_dict()) == set(
        children["qnet_target"].state_dict()
    )
    assert children["qnet"].logits.data_ptr() != (
        children["qnet_target"].logits.data_ptr()
    )
