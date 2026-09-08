"""Mixed DAgger control must not filter perception histories or supervision."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.transforms import CatTensors

from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule,
    TemporalDepthGRU,
    set_recurrent_mode,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DAGGER_IS_DAGGER_ENV_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    apply_perception_training_source,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC as TVKD,
    TVKDDistributionalFastSACTeacherBCConfig as Config,
)


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(7, 2)

    def get_dist(self, td):
        inputs = torch.cat([td[key] for key in ("vel_command", "policy", "priv_pred")], -1)
        return SimpleNamespace(mean=self.head(inputs))


class _Target(nn.Module):
    def forward(self, td):
        td["priv_feature"] = td["priv"]
        return td


class _SmallTVKD(TVKD):
    """Use real TVKD training, replay, live inference, and both recurrent cores."""

    def __init__(self):
        nn.Module.__init__(self)
        self.cfg = Config(perception_training_source="all")
        apply_perception_training_source(self.cfg)
        self.cfg.train_every = 6
        self.cfg.num_minibatches = 1
        self.cfg.latent_dim = 4
        self.cfg.perception_depth_residual = True
        self.cfg.enable_residual_distillation = False
        self.cfg.train_dr_estimator = False
        self.cfg.perception_encode_microbatch_size = 4
        self.device = torch.device("cpu")
        self.depth_feature_dim = 4
        self.q_actor_keys = ("vel_command", "policy", "priv_pred")
        self._q_actor_dim = 7
        self._perception_ema_generation = 0
        self.object_transform = nn.Identity()
        self.encoder_priv = _Target()
        self.temporal_depth_gru = TemporalDepthGRU(nn.Linear(3, 4), hidden_dim=4)
        self.object_adapt = TensorDictModule(nn.Linear(4, 2), ["_depth_feature"], ["object_pred"])
        self.object_pred_transform = nn.Identity()
        self.adapt_module = TensorDictSequential(
            CatTensors(["policy", "object_pred"], "_adapt_inp", sort=False, del_keys=False),
            TensorDictModule(
                DepthResidualGRUModule(4, 4),
                ["_adapt_inp", "is_init", "adapt_hx", "_depth_feature"],
                ["priv_pred", ("next", "adapt_hx")],
            ),
        )
        self.actor_adapt = _Actor()
        self.online(_rollout())  # Materialize lazy recurrent parameters.
        self.perception_pairs = (
            ("temporal_depth_gru", "temporal_depth_gru_ema"),
            ("object_adapt", "object_adapt_ema"),
            ("adapt_module", "adapt_ema"),
        )
        for online, ema in self.perception_pairs:
            setattr(self, ema, copy.deepcopy(getattr(self, online)).requires_grad_(False))
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.opt_adapt = torch.optim.SGD(
            [parameter for online, _ in self.perception_pairs for parameter in getattr(self, online).parameters()],
            lr=0.03,
        )
        self._ensure_exact_online_state()

    def _encode_replay_object_geo(self, td):
        return td["object_geo_"].squeeze(-1).long()

    def _decode_replay_object_geo(self, indices, *, device, dtype):
        return indices.unsqueeze(-1).to(device=device, dtype=dtype)

    @torch.no_grad()
    def online(self, td):
        with set_recurrent_mode(True):
            self.temporal_depth_gru(td)
            self.object_adapt(td)
            self.adapt_module(td)
        return td

    @torch.no_grad()
    def student(self, td):
        with set_recurrent_mode(True):
            action = self._student_raw_action_proposal(td)
        return action, td["priv_pred"].clone()


def _rollout():
    generator = torch.Generator().manual_seed(799)
    reset = torch.zeros(2, 6, 1, dtype=torch.bool)
    reset[:, 0] = True
    return TensorDict(
        {
            "depth": torch.randn(2, 6, 3, generator=generator),
            "policy": torch.randn(2, 6, 2, generator=generator),
            "vel_command": torch.randn(2, 6, 1, generator=generator),
            "object_geo_": torch.zeros(2, 6, 1),
            "priv": torch.randn(2, 6, 4, generator=generator),
            "object_": torch.randn(2, 6, 2, generator=generator),
            "depth_hx": torch.zeros(2, 6, 4),
            "adapt_hx": torch.zeros(2, 6, 4),
            "is_init": reset,
            "row_id": torch.arange(12).reshape(2, 6),
            DAGGER_IS_DAGGER_ENV_KEY: torch.tensor([[True] * 6, [False] * 6]),
            DAGGER_IS_STUDENT_ACTION_KEY: torch.tensor(
                [[True, False, True, False, True, False], [True] * 6]
            ),
        },
        [2, 6],
    )


def test_all_source_perception_keeps_teacher_turns_in_full_sequences_and_loss():
    torch.manual_seed(809)
    policy = _SmallTVKD()
    reference = copy.deepcopy(policy)
    rollout = _rollout()
    all_student_labels = rollout.clone()
    all_student_labels[DAGGER_IS_STUDENT_ACTION_KEY].fill_(True)
    assert policy._live_perception_rollout(rollout) is rollout
    seen = []
    hook = policy.temporal_depth_gru.register_forward_pre_hook(
        lambda module, args: seen.append(args[0]["row_id"].clone())
    )
    try:
        torch.manual_seed(821)
        metrics = policy.train_adapt(rollout)
        torch.manual_seed(821)
        reference.train_adapt(all_student_labels)
    finally:
        hook.remove()

    # Every complete history, including Teacher turns, is processed each epoch.
    assert len(seen) == 2
    assert all(value.shape == (2, 6) for value in seen)
    assert torch.equal(torch.cat(seen).flatten().bincount(), torch.full((12,), 2))
    for pair in policy.perception_pairs:
        for name in pair:
            for actual, expected in zip(getattr(policy, name).parameters(), getattr(reference, name).parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metrics["adapt/perception_live_all_envs"] == 1.0
    assert metrics["adapt/perception_live_rows"] == 12.0


def test_teacher_turn_supervision_updates_shared_ema_for_pure_student_prediction():
    torch.manual_seed(829)
    policy = _SmallTVKD()
    rollout = _rollout()
    baseline = policy.online(rollout.clone())
    # Initially only Teacher-controlled rows have any reconstruction error.
    rollout["priv"] = baseline["priv_pred"].clone()
    rollout["object_"] = baseline["object_pred"].clone()
    teacher_turn = ~rollout[DAGGER_IS_STUDENT_ACTION_KEY]
    rollout["priv"][teacher_turn] += 1.0
    assert teacher_turn[0].any() and not teacher_turn[1].any()
    student_action_before, student_latent_before = policy.student(_rollout()[1:2])
    actor_before = copy.deepcopy(policy.actor_adapt.state_dict())

    metrics = policy.train_adapt(rollout)

    student_action_after, student_latent_after = policy.student(_rollout()[1:2])
    assert metrics["adapt/priv_loss"] > 0.0
    assert not torch.allclose(student_latent_before, student_latent_after, rtol=0, atol=1e-7)
    assert not torch.allclose(student_action_before, student_action_after, rtol=0, atol=1e-7)
    # The common perception weights changed; the actor itself did not train here.
    for key, value in policy.actor_adapt.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)


def test_exact_online_latent_includes_teacher_turn_history_without_cross_env_mixing():
    torch.manual_seed(839)
    policy = _SmallTVKD()
    states = _rollout()
    policy._rollout_final_batch = {
        "exact_input__" + key: value
        for key, value in policy._exact_perception_inputs(states[:, -1]).items()
    }
    policy._truncation_final_batches = []
    current, successor = policy._journal_exact_online_rollout(states[:, :-1])
    # DAgger's student turn 4 follows Teacher turns 1 and 3 in the same episode.
    assert states[DAGGER_IS_STUDENT_ACTION_KEY][0, :5].tolist() == [True, False, True, False, True]
    assert current[0, :, 0].unique().numel() == current[1, :, 0].unique().numel() == 1
    assert current[0, 0, 0] != current[1, 0, 0]
    actual = policy._gather_exact_online_actor(current[:, 4])
    next_actual = policy._gather_exact_online_actor(successor[:, 4])
    _, direct = policy.student(states.clone())
    torch.testing.assert_close(actual[:, -4:], direct[:, 4])
    torch.testing.assert_close(next_actual[:, -4:], direct[:, 5])

    # Dropping Teacher turns is not equivalent: their factual observations are
    # part of the recurrent history even when the next action is Student-owned.
    _, wrongly_filtered = policy.student(states[0:1, [0, 2, 4]].clone())
    assert not torch.allclose(actual[0, -4:], wrongly_filtered[0, -1], rtol=0, atol=1e-6)

    changed = states.clone()
    changed["depth"][0, [1, 3]] += 3.0
    _, changed_prediction = policy.student(changed)
    assert not torch.allclose(direct[0, 4], changed_prediction[0, 4], rtol=0, atol=1e-6)
    torch.testing.assert_close(direct[1], changed_prediction[1], rtol=0, atol=0)
