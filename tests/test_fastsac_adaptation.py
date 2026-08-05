from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from hydra.core.config_store import ConfigStore
from hydra.plugins.config_source import ConfigLoadError

from active_adaptation.learning.ppo.fastsac_vel import (
    FastSACVelFinetuneConfig,
    FastSACVelFinetune,
    FastSACVEL,
    TEACHER_REPLAY_FIELDS,
    TeacherReplayBuffer,
)
from active_adaptation.learning.ppo.common import ACTION_KEY


def test_student_keeps_ppo_vel_finetune_vaic_inputs_and_removes_legacy_config():
    store = ConfigStore.instance()
    fastsac = store.load("algo/fastsac_vel_finetune.yaml").node
    ppo = store.load("algo/ppo_vel_finetune.yaml").node

    assert list(fastsac.in_keys) == list(ppo.in_keys)
    assert fastsac.vecnorm == ppo.vecnorm == "eval"
    assert fastsac.use_depth == ppo.use_depth
    assert fastsac.use_object_adapt == ppo.use_object_adapt
    assert fastsac.adapt_module == ppo.adapt_module
    assert fastsac._target_.endswith("fastsac_vel.FastSACVelFinetune")
    with pytest.raises(ConfigLoadError):
        store.load("algo/ppo_fastsac_vel_train.yaml")


class _ReplayRecorder:
    def __init__(self):
        self.extended = None
        self.size = 0

    def extend(self, transitions):
        self.extended = transitions
        self.size += next(iter(transitions.values())).shape[0]


def _fake_transitions(count=4):
    return {
        "observations": torch.zeros(count, 3),
        "critic_observations": torch.zeros(count, 4),
        "actions": torch.zeros(count, 1),
        "rewards": torch.zeros(count),
        "dones": torch.zeros(count, dtype=torch.bool),
        "truncations": torch.zeros(count, dtype=torch.bool),
        "discounts": torch.ones(count),
        "next_observations": torch.zeros(count, 3),
        "next_critic_observations": torch.zeros(count, 4),
        "next_actions": torch.zeros(count, 1),
    }


def test_student_transition_chunks_keep_all_rows_and_final_timeout_state():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(train_every=3)
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy.reward_scales = torch.tensor([1.0])
    policy._last_timeout_finals_used = 0
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[90.0], [190.0]]),
        "next_critic_observations": torch.tensor([[91.0], [191.0]]),
    }
    policy._timeout_final_batches = [{
        "indices": torch.tensor([5]),
        "next_observations": torch.tensor([[900.0]]),
        "next_critic_observations": torch.tensor([[901.0]]),
    }]

    marker = torch.tensor(
        [[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]]
    )
    done = torch.zeros(2, 3, 1, dtype=torch.bool)
    done[1, 2] = True
    rollout = TensorDict(
        {
            "actor": marker,
            "critic": marker + 20.0,
            ACTION_KEY: marker + 30.0,
            "next": TensorDict(
                {
                    "reward": marker,
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.full_like(marker, 0.75),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    chunks = list(policy._student_transition_chunks(rollout))
    transitions = {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in chunks[0]
    }
    assert transitions["rewards"].shape == (6,)
    assert torch.equal(
        transitions["observations"][:, 0],
        torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0, 12.0]),
    )
    assert torch.equal(
        transitions["next_observations"][:, 0],
        torch.tensor([1.0, 11.0, 2.0, 12.0, 90.0, 900.0]),
    )
    assert torch.equal(transitions["discounts"], torch.full((6,), 0.75))
    assert policy._last_timeout_finals_used == 1


def test_fastsac_train_op_keeps_original_adaptation_and_ema_path():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_updates_per_env_step=2,
        sac_policy_frequency=2,
        teacher_buffer_ratio=0.5,
        sac_learning_starts=1,
        train_every=3,
        sac_max_grad_norm=0.0,
    )
    policy.sac_update_count = 0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.sac_environment_steps = 0
    policy.sac_rollout_count = 0
    policy.num_updates = 0
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy._last_timeout_finals_used = 3
    policy.online_replay = _ReplayRecorder()
    events = []

    transitions = {
        key: value
        for key, value in _fake_transitions(count=2).items()
        if key in TEACHER_REPLAY_FIELDS
    }

    def transition_chunks(td):
        events.append("transitions")
        yield transitions
        yield transitions
        yield transitions

    policy._student_transition_chunks = transition_chunks
    policy._mix_batch = lambda: {"batch": torch.tensor(1.0)}

    def sac_update(batch, update_actor):
        events.append(f"sac:{update_actor}")
        return (
            torch.tensor(1.0),
            torch.tensor([2.0, 3.0]),
            torch.tensor(4.0),
            torch.tensor(5.0),
            torch.tensor(6.0),
            torch.tensor(7.0),
        )

    policy._sac_update = sac_update

    def train_adapt(td):
        events.append("adapt")
        assert "stats" not in td.keys()
        td["adapt_only"] = torch.ones(*td.batch_size, 1)
        return {"adapt/priv_loss": 0.25, "adapt/grad_norm": 0.5}

    policy.train_adapt = train_adapt
    rollout = TensorDict(
        {
            "marker": torch.zeros(2, 3, 1),
            "stats": torch.zeros(2, 3, 1),
        },
        batch_size=[2, 3],
    )

    info = policy.train_op(rollout)

    assert events == ["transitions", "sac:False", "sac:True", "adapt"]
    assert policy.sac_update_count == 2
    assert policy.num_updates == 1
    assert info["adapt/priv_loss"] == 0.25
    assert info["adapt/grad_norm"] == 0.5
    assert info["fastsac/timeout_finals"] == 3
    assert "adapt_only" not in rollout.keys()
    assert set(policy.online_replay.extended) == set(TeacherReplayBuffer.fields)


def test_fastsac_rejects_second_actor_optimizer_from_distillation():
    assert FastSACVelFinetuneConfig().enable_residual_distillation is False
    cfg = SimpleNamespace(enable_residual_distillation=True)
    with pytest.raises(ValueError, match="requires enable_residual_distillation=false"):
        FastSACVelFinetune(cfg, None, None, None, "cpu", None)


def test_teacher_to_student_resets_stage_local_q_counter(monkeypatch):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_alpha_init=0.001)
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.sac_update_count = 123

    def load_teacher(self, state_dict, strict=True):
        self.q_update_count = 77
        return []

    monkeypatch.setattr(FastSACVEL, "load_state_dict", load_teacher)
    policy.load_state_dict({
        "qnet": {},
        "last_phase": "train",
        "teacher_replay_state": {"size": 1},
    })

    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0


def test_student_checkpoint_hook_does_not_require_teacher_training_fifo():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.teacher_replay = None

    assert policy.snapshot_teacher_replay(1, "checkpoint_1") is None


def test_rlpd_batch_is_exactly_half_teacher_half_online():
    class SourceReplay:
        def __init__(self, marker, size=20_000):
            self.marker = marker
            self.size = size
            self.calls = []

        def sample(self, count, device):
            self.calls.append(count)
            widths = {
                "observations": 3,
                "critic_observations": 4,
                "actions": 2,
                "next_observations": 3,
                "next_critic_observations": 4,
            }
            data = {}
            for key in TEACHER_REPLAY_FIELDS:
                if key in ("dones", "truncations"):
                    data[key] = torch.zeros(count, dtype=torch.bool, device=device)
                elif key in ("rewards", "discounts"):
                    data[key] = torch.full((count,), self.marker, device=device)
                else:
                    data[key] = torch.full(
                        (count, widths[key]), self.marker, device=device
                    )
            return data

    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_batch_size=8192, teacher_buffer_ratio=0.5)
    policy.device = "cpu"
    policy.online_replay = SourceReplay(marker=1.0)
    policy.offline_replay = SourceReplay(marker=0.0)

    mixed = policy._mix_batch()
    assert policy.online_replay.calls == [4096]
    assert policy.offline_replay.calls == [4096]
    assert mixed["observations"].shape[0] == 8192
    assert mixed["observations"][:, 0].eq(0.0).sum().item() == 4096
    assert mixed["observations"][:, 0].eq(1.0).sum().item() == 4096
    for key in TEACHER_REPLAY_FIELDS:
        assert mixed[key].shape[0] == 8192
