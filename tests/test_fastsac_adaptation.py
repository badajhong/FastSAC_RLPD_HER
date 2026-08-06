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
    NEXT_TEACHER_REF_ACTION_FIELD,
    TEACHER_REPLAY_FIELDS,
    TEACHER_REF_ACTION_FIELD,
    TeacherReplayBuffer,
    _validate_fastsac_finetune_config,
)
from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_vel import REF_JPOS_KEY


def test_student_keeps_ppo_vel_finetune_vaic_inputs_and_removes_legacy_config():
    store = ConfigStore.instance()
    fastsac = store.load("algo/fastsac_vel_finetune.yaml").node
    fastsac_teacher = store.load("algo/fastsac_vel_train.yaml").node
    ppo = store.load("algo/ppo_vel_finetune.yaml").node

    assert list(fastsac.in_keys) == list(ppo.in_keys)
    assert fastsac.vecnorm == ppo.vecnorm == "eval"
    assert fastsac.use_depth == ppo.use_depth
    assert fastsac.use_object_adapt == ppo.use_object_adapt
    assert fastsac.adapt_module == ppo.adapt_module
    assert fastsac._target_.endswith("fastsac_vel.FastSACVelFinetune")
    assert fastsac_teacher.train_student_models is True
    for config in (fastsac_teacher, fastsac):
        assert config.gamma == 0.99
        assert config.sac_updates_per_env_step == 4
        assert config.sac_policy_frequency == 2
        assert config.sac_target_entropy_ratio == 0.5
        assert config.sac_tau == 0.05
        assert config.q_num_atoms == 501
        assert config.sac_learning_starts == 10
        assert config.sac_replay_raw_observations is True
        assert config.sac_q_normalize_actions is True
        assert config.sac_q_action_input_gain == 1.0
        assert config.q_action_coordinates == "absolute"
        assert config.q_reference_dueling is False
        assert config.sac_clipped_double_q is True
        assert config.sac_use_autotune is True
    assert fastsac_teacher.sac_teacher_updates_per_env_step == 4
    assert fastsac_teacher.sac_teacher_update_interval_env_steps == 1
    assert fastsac_teacher.sac_teacher_actor_batch_size == 0
    assert fastsac_teacher.sac_teacher_actor_objective == "sac"
    assert fastsac_teacher.sac_teacher_awac_beta == 0.01
    assert fastsac_teacher.sac_teacher_awac_weight_clip == 20.0
    assert fastsac_teacher.sac_teacher_actor_uncertainty_gate is False
    assert fastsac_teacher.sac_teacher_conservative_q_coef == 0.0
    assert fastsac_teacher.sac_teacher_conservative_q_margin == 0.002
    assert fastsac_teacher.sac_teacher_conservative_q_temperature == 0.002
    assert fastsac_teacher.sac_teacher_conservative_q_starts_q_updates is None
    assert fastsac_teacher.sac_teacher_learning_starts_transitions == 98_304
    assert fastsac_teacher.sac_teacher_actor_learning_starts_q_updates == 8_000
    assert fastsac_teacher.sac_teacher_policy_frequency == 32
    assert fastsac_teacher.sac_teacher_actor_lr == 3e-6
    assert fastsac_teacher.sac_teacher_alpha_lr == 2e-5
    assert fastsac_teacher.sac_teacher_q_max_grad_norm == 0.0
    assert fastsac_teacher.sac_teacher_actor_max_grad_norm == 1.0
    # Stage-2 RLPD retains the original HOI update settings.
    assert fastsac.sac_policy_frequency == 2
    assert fastsac.sac_actor_lr == 3e-4
    assert fastsac.sac_max_grad_norm == 0.0
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


def test_student_transition_chunks_keep_timeout_final_state():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(train_every=3)
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy.reward_scales = torch.tensor([1.0])
    policy._last_truncation_finals_used = 0
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[90.0], [190.0]]),
        "next_critic_observations": torch.tensor([[91.0], [191.0]]),
    }
    policy._truncation_final_batches = [{
        "indices": torch.tensor([5]),
        "next_observations": torch.tensor([[900.0]]),
        "next_critic_observations": torch.tensor([[901.0]]),
    }]

    marker = torch.tensor(
        [[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]]
    )
    done = torch.zeros(2, 3, 1, dtype=torch.bool)
    done[1, 2] = True
    episode_time_limit = done.clone()
    command_finished = torch.zeros_like(done)
    rollout = TensorDict(
        {
            "actor": marker,
            "critic": marker + 20.0,
            ACTION_KEY: marker + 30.0,
            # Student FastSAC keeps PPO finetune's longer perception warm-up:
            # only step_count > 5 enters online replay.
            "step_count": torch.tensor(
                [[[0], [5], [6]], [[6], [7], [8]]]
            ),
            "next": TensorDict(
                {
                    "reward": marker,
                    "done": done,
                        "terminated": torch.zeros_like(done),
                        "discount": torch.full_like(marker, 0.75),
                        "stats": TensorDict(
                            {
                                "episode_time_limit": episode_time_limit,
                                "command_finished": command_finished,
                            },
                            batch_size=[2, 3],
                        ),
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
    assert transitions["rewards"].shape == (4,)
    assert torch.equal(
        transitions["observations"][:, 0],
        torch.tensor([10.0, 11.0, 2.0, 12.0]),
    )
    assert torch.equal(
        transitions["next_observations"][:, 0],
        torch.tensor([11.0, 12.0, 90.0, 900.0]),
    )
    assert torch.equal(transitions["discounts"], torch.full((4,), 0.75))
    assert policy._last_truncation_finals_used == 1


@pytest.mark.parametrize(
    ("q_action_coordinates", "q_reference_dueling"),
    (("reference_residual", False), ("absolute", True)),
)
def test_student_reference_dependent_replay_keeps_current_and_timeout_next_reference(
    q_action_coordinates, q_reference_dueling
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        train_every=2,
        q_action_coordinates=q_action_coordinates,
        q_reference_dueling=q_reference_dueling,
    )
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_dim = 1
    policy._q_critic_dim = 1
    policy.action_dim = 1
    policy.reward_scales = torch.ones(1)
    policy._last_truncation_finals_used = 0
    policy._rollout_final_batch = {
        "next_observations": torch.tensor([[90.0]]),
        "next_critic_observations": torch.tensor([[91.0]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[92.0]]),
    }
    policy._truncation_final_batches = [{
        "indices": torch.tensor([1]),
        "next_observations": torch.tensor([[900.0]]),
        "next_critic_observations": torch.tensor([[901.0]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[902.0]]),
    }]
    marker = torch.tensor([[[0.0], [1.0]]])
    done = torch.tensor([[[False], [True]]])
    rollout = TensorDict(
        {
            "actor": marker,
            "critic": marker + 20.0,
            REF_JPOS_KEY: marker + 60.0,
            ACTION_KEY: marker + 30.0,
            "step_count": torch.tensor([[[6], [7]]]),
            "next": TensorDict(
                {
                    "reward": marker,
                    "done": done,
                    "terminated": torch.zeros_like(done),
                    "discount": torch.ones_like(marker),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": done.clone(),
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

    chunks = list(policy._student_transition_chunks(rollout))
    transitions = {
        key: torch.cat([chunk[key] for chunk in chunks])
        for key in (
            TEACHER_REF_ACTION_FIELD,
            NEXT_TEACHER_REF_ACTION_FIELD,
        )
    }

    assert torch.equal(
        transitions[TEACHER_REF_ACTION_FIELD].squeeze(-1),
        torch.tensor([60.0, 61.0]),
    )
    assert torch.equal(
        transitions[NEXT_TEACHER_REF_ACTION_FIELD].squeeze(-1),
        torch.tensor([61.0, 902.0]),
    )


def test_student_collector_hook_captures_timeout_final_before_reset():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(train_every=3)
    policy._truncation_final_batches = []
    policy._prepare_student_final_state = lambda td: {
        "next_observations": td["actor_marker"].clone(),
        "next_critic_observations": td["critic_marker"].clone(),
    }
    td = TensorDict(
        {
            "next": TensorDict(
                {
                    "actor_marker": torch.tensor([[101.0], [102.0], [103.0]]),
                    "critic_marker": torch.tensor([[111.0], [112.0], [113.0]]),
                    "done": torch.tensor([[False], [True], [True]]),
                    "terminated": torch.tensor([[False], [False], [True]]),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.tensor(
                                [[False], [True], [True]]
                            ),
                            "command_finished": torch.zeros(
                                3, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[3],
                    ),
                },
                batch_size=[3],
            )
        },
        batch_size=[3],
    )

    policy.capture_truncation_final_observations(td, step=1)

    assert len(policy._truncation_final_batches) == 1
    captured = policy._truncation_final_batches[0]
    assert torch.equal(captured["indices"], torch.tensor([4]))
    assert torch.equal(captured["next_observations"], torch.tensor([[102.0]]))
    assert torch.equal(
        captured["next_critic_observations"], torch.tensor([[112.0]])
    )


def test_fastsac_train_op_keeps_original_adaptation_and_ema_path():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_updates_per_env_step=2,
        sac_policy_frequency=2,
        teacher_buffer_ratio=0.5,
        sac_learning_starts=1,
        sac_batch_size=2,
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
    policy.target_entropy = 0.0
    policy._fastsac_action_log_scale_sum = 0.0
    policy._last_truncation_finals_used = 3
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
    assert info["fastsac/truncation_finals"] == 3
    assert info["fastsac/actor_loss"] == 5.0
    assert info["fastsac/entropy"] == 7.0
    assert info["fastsac/normalized_action_entropy"] == 7.0
    assert "adapt_only" not in rollout.keys()
    assert set(policy.online_replay.extended) == set(TeacherReplayBuffer.fields)


def test_student_resume_waits_until_online_replay_has_a_valid_row():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_updates_per_env_step=1,
        sac_policy_frequency=2,
        teacher_buffer_ratio=0.5,
        sac_learning_starts=0,
        sac_batch_size=2,
        train_every=2,
    )
    policy.sac_environment_steps = 100
    policy.sac_rollout_count = 0
    policy.sac_update_count = 0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.num_updates = 0
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.target_entropy = 0.0
    policy._fastsac_action_log_scale_sum = 0.0
    policy._last_truncation_finals_used = 0
    policy.online_replay = _ReplayRecorder()
    events = []
    empty = {
        key: value[:0]
        for key, value in _fake_transitions(count=1).items()
        if key in TEACHER_REPLAY_FIELDS
    }
    valid = {
        key: value
        for key, value in _fake_transitions(count=1).items()
        if key in TEACHER_REPLAY_FIELDS
    }
    policy._student_transition_chunks = lambda td: iter((empty, valid))
    policy._mix_batch = lambda: events.append("mix") or {"batch": torch.ones(1)}
    policy._sac_update = lambda batch, update_actor: (
        torch.tensor(1.0),
        torch.tensor([2.0, 3.0]),
        torch.tensor(4.0),
        torch.tensor(5.0),
        torch.tensor(6.0),
        torch.tensor(7.0),
    )
    policy.train_adapt = lambda td: {}
    rollout = TensorDict(
        {
            "marker": torch.zeros(1, 2, 1),
            "stats": torch.zeros(1, 2, 1),
        },
        batch_size=[1, 2],
    )

    policy.train_op(rollout)

    assert events == ["mix"]
    assert policy.online_replay.size == 1
    assert policy.sac_update_count == 1


def test_rlpd_offline_only_mix_does_not_require_online_rows():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_batch_size=4, teacher_buffer_ratio=1.0)
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(0)
    policy.online_replay = SimpleNamespace(size=0)

    class _Offline:
        def sample(self, count, device=None):
            return {
                key: value
                for key, value in _fake_transitions(count=count).items()
                if key in TEACHER_REPLAY_FIELDS
            }

    policy.offline_replay = _Offline()
    mixed = policy._mix_batch()

    assert set(mixed) == set(TEACHER_REPLAY_FIELDS)
    assert all(value.shape[0] == 4 for value in mixed.values())


def test_fastsac_rejects_second_actor_optimizer_from_distillation():
    assert FastSACVelFinetuneConfig().enable_residual_distillation is False
    cfg = SimpleNamespace(enable_residual_distillation=True)
    with pytest.raises(ValueError, match="requires enable_residual_distillation=false"):
        FastSACVelFinetune(cfg, None, None, None, "cpu", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sac_updates_per_env_step", 0),
        ("sac_policy_frequency", 0),
        ("sac_tau", 1.1),
        ("sac_max_grad_norm", float("nan")),
        ("sac_alpha_init", 0.0),
        ("sac_target_entropy_ratio", -0.1),
        ("teacher_buffer_ratio", 1.1),
        ("sac_q_normalize_actions", "true"),
        ("q_action_coordinates", "residual"),
        ("q_reference_dueling", "true"),
        ("sac_q_action_input_gain", 0.0),
        ("sac_q_action_input_gain", -1.0),
        ("sac_q_action_input_gain", float("nan")),
        ("sac_q_action_input_gain", float("inf")),
        ("sac_clipped_double_q", "true"),
        ("sac_use_autotune", "true"),
    ],
)
def test_stage2_rejects_invalid_sac_config(field, value):
    cfg = FastSACVelFinetuneConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        _validate_fastsac_finetune_config(cfg)


class _Stage2QSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("support", torch.tensor([-1.0, 1.0]))
        self.forward_actions = []
        self.forward_references = []
        self.projection_action = None
        self.projection_reference = None

    def forward(self, observations, actions, reference_actions=None):
        self.forward_actions.append(actions.detach().clone())
        self.forward_references.append(
            None
            if reference_actions is None
            else reference_actions.detach().clone()
        )
        base = actions[..., :1] * 0.0 + self.dummy * 0.0
        logits = torch.cat((base, base), dim=-1)
        return logits.unsqueeze(0).expand(2, -1, -1)

    def values(self, logits):
        return (torch.softmax(logits, dim=-1) * self.support).sum(-1)

    @torch.no_grad()
    def projection(
        self,
        observations,
        actions,
        reward,
        bootstrap,
        discount,
        reference_actions=None,
    ):
        self.projection_action = actions.clone()
        self.projection_reference = (
            None if reference_actions is None else reference_actions.clone()
        )
        return torch.full(
            (2, observations.shape[0], 2),
            0.5,
            device=observations.device,
        )


class _Stage2Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor(1.0))

    def get_dist(self, td):
        count = td.batch_size[0]
        action = self.action

        class _Dist:
            def rsample_with_log_prob(self, generator=None):
                sampled = action.expand(count, 1)
                return sampled, sampled[:, 0] * 0.0 - 2.0

        return _Dist()


def test_stage2_target_current_and_actor_q_paths_normalize_actions():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        sac_q_normalize_actions=True,
        # This option is Stage-1-only and must be ignored by the RLPD update.
        sac_teacher_actor_uncertainty_gate=True,
    )
    policy._fastsac_q_action_low = torch.tensor([-2.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.actor_adapt = _Stage2Actor()
    policy._actor_dist_from_flat = lambda obs: policy.actor_adapt.get_dist(
        TensorDict({}, batch_size=obs.shape[:-1])
    )
    policy.qnet = _Stage2QSpy()
    policy.qnet_target = _Stage2QSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.sac_actor_optimizer = torch.optim.SGD(
        policy.actor_adapt.parameters(), lr=0.0
    )
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.0)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[-2.0], [2.0]]),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    policy._sac_update(batch, update_actor=True)

    assert torch.equal(
        policy.qnet_target.projection_action, torch.full((2, 1), 0.5)
    )
    assert torch.equal(
        policy.qnet.forward_actions[0], torch.tensor([[-1.0], [1.0]])
    )
    assert torch.equal(
        policy.qnet.forward_actions[1], torch.full((2, 1), 0.5)
    )
    assert len(policy.qnet.forward_actions) == 2


def test_stage2_q_uses_current_and_next_reference_residual_coordinates():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        q_action_coordinates="reference_residual",
        sac_q_normalize_actions=True,
    )
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.actor_adapt = _Stage2Actor()
    policy._actor_dist_from_flat = lambda obs: policy.actor_adapt.get_dist(
        TensorDict({}, batch_size=obs.shape[:-1])
    )
    policy.qnet = _Stage2QSpy()
    policy.qnet_target = _Stage2QSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.sac_actor_optimizer = torch.optim.SGD(
        policy.actor_adapt.parameters(), lr=0.0
    )
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.0)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[-2.0], [2.0]]),
        TEACHER_REF_ACTION_FIELD: torch.tensor([[-1.0], [1.0]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[0.5], [0.5]]),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    policy._sac_update(batch, update_actor=True)

    # Target actor action is 1; it must use next reference 0.5.
    assert torch.equal(
        policy.qnet_target.projection_action, torch.full((2, 1), 0.25)
    )
    # Replay Q uses current references; actor Q uses those same current refs.
    assert torch.equal(
        policy.qnet.forward_actions[0], torch.tensor([[-0.5], [0.5]])
    )
    assert torch.equal(
        policy.qnet.forward_actions[1], torch.tensor([[1.0], [0.0]])
    )


def test_stage2_reference_dueling_q_receives_current_and_next_references():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        q_action_coordinates="absolute",
        q_reference_dueling=True,
        sac_q_normalize_actions=True,
    )
    policy._fastsac_q_action_low = torch.tensor([-2.0])
    policy._fastsac_q_action_scale = torch.tensor([2.0])
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.actor_adapt = _Stage2Actor()
    policy._actor_dist_from_flat = lambda obs: policy.actor_adapt.get_dist(
        TensorDict({}, batch_size=obs.shape[:-1])
    )
    policy.qnet = _Stage2QSpy()
    policy.qnet_target = _Stage2QSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.sac_actor_optimizer = torch.optim.SGD(
        policy.actor_adapt.parameters(), lr=0.0
    )
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=0.0)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.tensor([[-2.0], [2.0]]),
        TEACHER_REF_ACTION_FIELD: torch.tensor([[-1.0], [1.0]]),
        NEXT_TEACHER_REF_ACTION_FIELD: torch.tensor([[0.5], [0.5]]),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    policy._sac_update(batch, update_actor=True)

    assert torch.equal(
        policy.qnet_target.projection_action, torch.full((2, 1), 0.5)
    )
    assert torch.equal(
        policy.qnet_target.projection_reference,
        torch.full((2, 1), 0.25),
    )
    assert torch.equal(
        policy.qnet.forward_actions[0], torch.tensor([[-1.0], [1.0]])
    )
    assert torch.equal(
        policy.qnet.forward_references[0],
        torch.tensor([[-0.5], [0.5]]),
    )
    assert torch.equal(
        policy.qnet.forward_actions[1], torch.full((2, 1), 0.5)
    )
    assert torch.equal(
        policy.qnet.forward_references[1],
        torch.tensor([[-0.5], [0.5]]),
    )


def test_teacher_to_student_without_replay_manifest_resets_stage_local_q_counter(
    monkeypatch,
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_alpha_init=0.001, q_seed=7)
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.sac_update_count = 123
    policy.q_rng = torch.Generator().manual_seed(99)
    policy.sac_action_rng = torch.Generator().manual_seed(100)
    torch.randint(10, (3,), generator=policy.q_rng)
    torch.randn(3, generator=policy.sac_action_rng)

    def load_teacher(self, state_dict, strict=True):
        self.q_update_count = 77
        return []

    monkeypatch.setattr(FastSACVEL, "load_state_dict", load_teacher)
    policy.load_state_dict({
        "qnet": {},
        "last_phase": "train",
    })

    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0
    assert torch.equal(
        policy.q_rng.get_state(), torch.Generator().manual_seed(7).get_state()
    )
    assert torch.equal(
        policy.sac_action_rng.get_state(),
        torch.Generator().manual_seed(8).get_state(),
    )


def test_student_resume_does_not_require_previous_offline_replay_manifest(
    monkeypatch,
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(use_object_adapt=False, sac_alpha_init=0.001)
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.sac_update_count = 19

    def load_student(self, state_dict, strict=True):
        self.q_update_count = 77
        return []

    monkeypatch.setattr(FastSACVEL, "load_state_dict", load_student)
    policy.load_state_dict({"qnet": {}, "last_phase": "finetune"})

    assert policy.q_update_count == 77
    assert policy.sac_update_count == 19


def test_stage2_accepts_compatible_h5_from_different_replay_and_snapshot(tmp_path):
    path = tmp_path / "teacher_replay_buffer.h5"
    replay = TeacherReplayBuffer(
        path,
        capacity=8,
        actor_dim=3,
        critic_dim=4,
        action_dim=1,
        seed=0,
        replay_id="different-teacher-run",
        actor_backend="compatible-backend",
        actor_obs_keys=["actor"],
        critic_obs_keys=["critic"],
    )
    replay.append({
        key: value
        for key, value in _fake_transitions(count=3).items()
        if key in TEACHER_REPLAY_FIELDS
    })
    replay.snapshot(iteration=7000, checkpoint_name="different_checkpoint")

    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy._q_actor_dim = 3
    policy._q_critic_dim = 4
    policy.action_dim = 1
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(teacher_buffer_capacity=8, teacher_buffer_seed=0)
    policy.teacher_replay_id = "checkpoint-replay-id"
    policy.actor_backend = "compatible-backend"
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._loaded_teacher_replay_metadata = {
        "snapshot_id": "checkpoint-snapshot-id",
        "checkpoint_name": "checkpoint_from_another_run",
    }
    object.__setattr__(policy, "_replay_vecnorm", SimpleNamespace())

    policy.configure_offline_replay(path)

    assert policy.offline_replay.size == 3
    assert policy.offline_replay.snapshot_metadata["snapshot_id"] != (
        policy._loaded_teacher_replay_metadata["snapshot_id"]
    )


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

        def sample(self, count, device, generator=None):
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
    policy.q_rng = torch.Generator().manual_seed(0)
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
