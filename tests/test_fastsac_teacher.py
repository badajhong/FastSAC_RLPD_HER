from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.fastsac_vel import (
    FASTSAC_ACTOR_BACKEND,
    FASTSAC_TEACHER_TRAINING_ALGORITHM,
    FastSACVEL,
    _FastSACVAICBase,
)


def _teacher_transition_policy():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(train_every=3, phase="train")
    policy.q_actor_keys = ["actor_a", "actor_b"]
    policy.q_critic_keys = ["critic"]
    policy._teacher_actor_keys = ["teacher"]
    policy._q_actor_dim = 2
    policy._q_critic_dim = 1
    policy._fastsac_teacher_actor_dim = 1
    policy._teacher_raw_replay_fields = []
    policy.action_dim = 1
    policy.reward_scales = torch.tensor([0.25, 0.75])
    policy._timeout_final_batches = []
    policy._last_timeout_finals_used = 0
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[90.0, 91.0], [190.0, 191.0]]),
        "next_critic_observations": torch.tensor([[92.0], [192.0]]),
    }
    return policy


def _teacher_rollout():
    n, t = 2, 3
    marker = torch.tensor(
        [[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]]
    )
    done = torch.zeros(n, t, 1, dtype=torch.bool)
    terminated = torch.zeros_like(done)
    reward = torch.stack((marker, marker + 4.0), dim=-1).squeeze(-2)
    return TensorDict(
        {
            "actor_a": marker,
            "actor_b": marker + 20.0,
            "critic": marker + 30.0,
            "teacher": marker + 40.0,
            ACTION_KEY: marker + 50.0,
            "next": TensorDict(
                {
                    "reward": reward,
                    "done": done,
                    "terminated": terminated,
                    "discount": torch.ones(n, t, 1),
                },
                batch_size=[n, t],
            ),
        },
        batch_size=[n, t],
    )


def test_teacher_transitions_keep_all_nt_rows_and_last_timeout_final():
    policy = _teacher_transition_policy()
    rollout = _teacher_rollout()
    policy._timeout_final_batches.append(
        {
            # Last row of env 1: env * T + step = 1 * 3 + 2.
            "indices": torch.tensor([5]),
            "next_observations": torch.tensor([[901.0, 902.0]]),
            "next_critic_observations": torch.tensor([[903.0]]),
        }
    )

    transitions = policy._teacher_transitions(rollout)

    assert transitions["rewards"].shape == (6,)
    assert torch.equal(
        transitions["observations"][:, 0],
        torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0, 12.0]),
    )
    assert torch.equal(
        transitions["next_observations"][:, 0],
        torch.tensor([1.0, 11.0, 2.0, 12.0, 90.0, 901.0]),
    )
    assert policy._last_timeout_finals_used == 1
    assert policy._timeout_final_batches == []
    assert policy._rollout_final_batch is None


class _Replay:
    def __init__(self):
        self.events = []
        self.saved = 0
        self.seen = 0
        self.capacity = 10
        self.last_sample = None

    @property
    def size(self):
        return self.saved

    def clear(self):
        self.events.append("clear")
        self.saved = self.seen = 0

    def append(self, transitions):
        self.events.append("append")
        count = int(transitions["rewards"].shape[0])
        self.saved = min(self.capacity, self.saved + count)
        self.seen += count
        return count

    def sample(self, count, device=None, generator=None, fields=None):
        self.events.append("sample")
        self.last_sample = {"batch": torch.zeros(count)}
        return self.last_sample


def _metric(value=1.0):
    scalar = torch.tensor(value)
    return {
        "q_loss": scalar,
        "q1_loss": scalar,
        "q2_loss": scalar,
        "q_grad_norm": scalar,
        "alpha_loss": scalar,
        "target_q_min": scalar,
        "target_q_max": scalar,
    }


def test_true_teacher_train_op_never_calls_ppo_and_h5_gate_keeps_learning_fifo():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="train",
        train_every=2,
        teacher_buffer_start_iteration=5100,
        sac_learning_starts=0,
        sac_updates_per_env_step=2,
        sac_policy_frequency=2,
        sac_batch_size=3,
    )
    policy.env = SimpleNamespace(current_iter=5099)
    policy.device = torch.device("cpu")
    policy.action_dim = 1
    policy.joint_names = ["joint"]
    policy.teacher_replay = _Replay()
    policy._teacher_learning_fields = ()
    policy.q_rng = torch.Generator().manual_seed(0)
    policy.log_alpha = torch.nn.Parameter(torch.tensor(-1.0))
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.sac_environment_steps = 0
    policy.sac_rollout_count = 0
    policy.num_updates = 0
    policy._teacher_export_started = False
    policy._teacher_export_start_seen = None
    policy._last_timeout_finals_used = 0
    events = []
    policy._teacher_transition_chunks = lambda td: iter((
        {"rewards": torch.zeros(1)},
        {"rewards": torch.zeros(1)},
    ))
    policy._teacher_q_alpha_update = lambda batch: events.append("q") or _metric()
    def actor_update(batch):
        assert batch is policy.teacher_replay.last_sample
        events.append("actor")
        return {
            "actor_loss": torch.tensor(1.0),
            "actor_grad_norm": torch.tensor(1.0),
            "entropy": torch.tensor(1.0),
            "action_std": torch.tensor(1.0),
        }

    policy._teacher_actor_update = actor_update
    policy._soft_update_teacher_q_target = lambda: events.append("target")
    policy._train_adapt_no_depth = lambda td: events.append("adapt") or {}
    policy.train_policy = lambda td: pytest.fail("PPO train_policy must not run")
    assert "train_op" not in _FastSACVAICBase.__dict__
    assert "_train_ppo_path" not in _FastSACVAICBase.__dict__
    rollout = TensorDict(
        {"x": torch.zeros(1, 2, 1), "scale": torch.ones(1, 2, 1)},
        batch_size=[1, 2],
    )

    before_gate = policy.train_op(rollout)
    assert policy.teacher_replay.events == ["append", "append", "sample", "sample"]
    assert events == ["q", "target", "q", "actor", "target", "adapt"]
    assert before_gate["fastsac/h5_export_active"] == 0.0

    policy.env.current_iter = 5100
    events.clear()
    policy.teacher_replay.events.clear()
    at_gate = policy.train_op(rollout)
    assert policy.teacher_replay.events[:2] == ["append", "sample"]
    assert "clear" not in policy.teacher_replay.events
    assert at_gate["fastsac/h5_export_active"] == 1.0
    assert at_gate["fastsac/h5_export_rows"] == 2

    policy.env.current_iter = 5101
    policy.teacher_replay.events.clear()
    policy.train_op(rollout)
    assert "clear" not in policy.teacher_replay.events


class _FixedDist:
    def rsample(self):
        return torch.tensor([[0.25], [0.25]])

    def log_prob(self, action):
        assert torch.equal(action, self.rsample())
        return torch.tensor([-2.0, -2.0])


class _TinyQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 3))

    def forward(self, obs, action):
        return self.logits[:, None].expand(-1, obs.shape[0], -1)


class _ProjectionSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("support", torch.tensor([-1.0, 0.0, 1.0]))
        self.seen = None

    def projection(self, obs, action, reward, bootstrap, discount):
        self.seen = (action.clone(), reward.clone(), bootstrap.clone(), discount.clone())
        target = torch.zeros(2, obs.shape[0], 3)
        target[..., 1] = 1.0
        return target


def test_teacher_q_target_uses_stochastic_action_entropy_and_timeout_bootstrap():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(gamma=0.97, sac_max_grad_norm=0.0)
    policy.qnet = _TinyQ()
    policy.qnet_target = _ProjectionSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.01)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.01)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._teacher_state_from_replay = lambda batch, next_state=False: TensorDict(
        {}, batch_size=[2]
    )
    policy.actor = SimpleNamespace(get_dist=lambda td: _FixedDist())
    batch = {
        "next_critic_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.ones(2),
        "dones": torch.tensor([True, True]),
        "truncations": torch.tensor([True, False]),
        "discounts": torch.tensor([0.5, 1.0]),
    }

    policy._teacher_q_alpha_update(batch)
    action, soft_reward, bootstrap, discount = policy.qnet_target.seen
    assert torch.equal(action, torch.full((2, 1), 0.25))
    assert torch.equal(bootstrap, torch.tensor([1.0, 0.0]))
    assert torch.allclose(discount, torch.tensor([0.485, 0.97]))
    assert torch.allclose(soft_reward, torch.tensor([1.485, 1.0]))
    assert policy.q_update_count == 1
    assert policy.sac_update_count == 1
    assert policy.sac_alpha_update_count == 1


def test_true_teacher_rejects_markerless_old_ppo_hybrid_checkpoint():
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    with pytest.raises(ValueError, match="old PPO-based"):
        policy.load_state_dict({"actor_backend": "hoi_fastsac_tanh_gaussian_v1"})


def test_pre_gate_resume_ignores_auto_discovered_future_h5(monkeypatch):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train")
    policy._loaded_checkpoint_phase = "train"
    policy._teacher_export_started = False
    seen = {}

    def configure_base(self, path, restore_path=None):
        seen["restore_path"] = restore_path

    monkeypatch.setattr(_FastSACVAICBase, "configure_teacher_replay", configure_base)
    policy.configure_teacher_replay("new.h5", restore_path="future-final.h5")
    assert seen["restore_path"] is None


def test_same_stage_resume_continues_after_completed_iteration(monkeypatch):
    policy = FastSACVEL.__new__(FastSACVEL)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(phase="train", use_object_adapt=False)
    progress = {}
    policy.env = SimpleNamespace(
        set_progress=lambda iteration: progress.setdefault("iteration", iteration)
    )
    policy.actor_backend = FASTSAC_ACTOR_BACKEND
    policy._actor_backend_metadata = lambda: {"actor": "same"}
    policy._q_backend_metadata = lambda: {"q": "same"}

    monkeypatch.setattr(
        _FastSACVAICBase,
        "load_state_dict",
        lambda self, state_dict, strict=True: [],
    )
    policy.load_state_dict({
        "training_algorithm": FASTSAC_TEACHER_TRAINING_ALGORITHM,
        "actor_backend": FASTSAC_ACTOR_BACKEND,
        "actor_backend_config": {"actor": "same"},
        "q_backend_config": {"q": "same"},
        "teacher_replay_id": "replay",
        "last_phase": "train",
        "last_iter": 5100,
        "next_iter": 5101,
    })

    assert progress["iteration"] == 5101
