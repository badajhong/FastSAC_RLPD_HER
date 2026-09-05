from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, exploration_type, set_exploration_type

from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.perception_only import (
    EMA_PERCEPTION_MODULES,
    FULL_PERCEPTION_CHECKPOINT_MODULES,
    ONLINE_PERCEPTION_MODULES,
    TeacherRolloutPerceptionOnly,
    _PrivilegedTeacherPerceptionRollout,
)
from active_adaptation.learning.ppo.ppo_vel import (
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    PRIV_PRED_KEY,
    REF_JPOS_KEY,
)
from scripts import percetpion


CONFIG_DIR = str((Path(__file__).parents[1] / "cfg").resolve())
ACTOR_STD_KEY = "module.0.module.2.module.actor_std"


def _compose_config(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="percetpion", overrides=list(overrides))


def test_exact_user_hydra_surface_and_iteration_budget(monkeypatch):
    cfg = _compose_config(
        "task=G1/vaic/skateboard_general_tracking_stu",
        "iteration=3000",
        "task.num_envs=512",
        "algo.load_noise_scale=0.2",
    )
    monkeypatch.setattr(percetpion.aa, "get_world_size", lambda: 1)

    percetpion.apply_perception_iteration_controls(cfg)

    assert cfg.algo._target_.endswith(".TeacherRolloutPerceptionOnly")
    assert cfg.algo.name == "teacher_rollout_perception_only"
    assert cfg.algo.phase == "finetune"
    assert cfg.algo.vecnorm == "eval"
    assert cfg.vecnorm == "eval"
    assert cfg.algo.enable_residual_distillation is False
    assert cfg.algo.train_dr_estimator is False
    assert cfg.algo.use_depth is True
    assert cfg.algo.use_object_adapt is True
    assert cfg.algo.adapt_module == "gru"
    assert cfg.algo.num_minibatches == 8
    assert cfg.algo.lr == pytest.approx(3e-4)
    assert cfg.algo.max_grad_norm == pytest.approx(1.0)
    assert cfg.algo.latent_dim == 256
    assert cfg.algo.adapt_module_input_cmd is True
    assert cfg.algo.perception_initialization == "teacher_warmstart"
    assert cfg.total_frames == 3000 * 512 * 32


def test_fresh_perception_initialization_is_an_explicit_valid_mode():
    cfg = _compose_config("algo.perception_initialization=fresh")

    TeacherRolloutPerceptionOnly._validate_config(cfg.algo)

    assert cfg.algo.perception_initialization == "fresh"


def test_perception_initialization_rejects_unknown_mode():
    cfg = _compose_config("algo.perception_initialization=hybrid")

    with pytest.raises(ValueError, match="perception_initialization"):
        TeacherRolloutPerceptionOnly._validate_config(cfg.algo)


@pytest.mark.parametrize("value", (True, 0, -1, 1.5))
def test_iteration_budget_rejects_non_positive_integer(monkeypatch, value):
    cfg = _compose_config()
    cfg.iteration = value
    monkeypatch.setattr(percetpion.aa, "get_world_size", lambda: 1)

    with pytest.raises(ValueError, match="iteration.*positive integer"):
        percetpion.apply_perception_iteration_controls(cfg)


def test_iteration_budget_rejects_unsynchronized_distributed_run(monkeypatch):
    cfg = _compose_config()
    monkeypatch.setattr(percetpion.aa, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="one process"):
        percetpion.apply_perception_iteration_controls(cfg)


def _teacher_checkpoint(path, *, residual_distillation=True, phase="train"):
    policy = {
        name: {}
        for name in percetpion.REQUIRED_TEACHER_MODULES
    }
    policy["actor"] = {ACTOR_STD_KEY: torch.full((3,), 0.35)}
    policy["actor_adapt"] = {ACTOR_STD_KEY: torch.full((3,), 0.45)}
    policy["last_phase"] = phase
    policy["last_iter"] = 123
    torch.save(
        {
            "policy": policy,
            "cfg": {
                "algo": {
                    "phase": phase,
                    "enable_residual_distillation": residual_distillation,
                    "name": "ppo_vel",
                    "_target_": "active_adaptation.learning.ppo.ppo_vel.PPOVEL",
                },
                "task": {"name": "G1SkateboardGeneralTracking"},
            },
            "vecnorm": {},
        },
        path,
    )


def test_teacher_checkpoint_audit_requires_residual_train_source(tmp_path):
    valid = tmp_path / "teacher.pt"
    _teacher_checkpoint(valid)

    audit = percetpion.validate_teacher_checkpoint(valid)

    assert audit == {
        "path": str(valid.resolve()),
        "last_iter": 123,
        "source_noise_scale": pytest.approx(0.35),
        "action_dim": 3,
        "task_name": "G1SkateboardGeneralTracking",
    }

    invalid = tmp_path / "absolute_actor.pt"
    _teacher_checkpoint(invalid, residual_distillation=False)
    with pytest.raises(ValueError, match="residual-distillation"):
        percetpion.validate_teacher_checkpoint(invalid)


class _StudentPerceptionStage(nn.Module):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def forward(self, td):
        self.calls.append(self.name)
        if self.name == "ema_depth":
            td["_depth_feature"] = torch.ones(*td.batch_size, 2)
            td["next", "depth_hx"] = torch.full((*td.batch_size, 2), 2.0)
        elif self.name == "ema_object":
            td[OBJECT_PRED_KEY] = torch.ones(*td.batch_size, 2)
        elif self.name == "object_pred_transform":
            td[OBJECT_PRED_TRANS_KEY] = torch.ones(*td.batch_size, 2)
        elif self.name == "ema_adapt":
            td[PRIV_PRED_KEY] = torch.ones(*td.batch_size, 2)
            td["_adapt_inp"] = torch.ones(*td.batch_size, 2)
            td["next", "adapt_hx"] = torch.full((*td.batch_size, 2), 3.0)
        return td


class _TeacherStage(nn.Module):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def forward(self, td):
        self.calls.append(self.name)
        if self.name == "teacher_object":
            td["object_trans"] = torch.ones(*td.batch_size, 1)
        elif self.name == "teacher_encoder":
            assert "object_trans" in td
            td["priv_feature"] = torch.ones(*td.batch_size, 1)
        elif self.name == "teacher_actor":
            assert exploration_type() is ExplorationType.RANDOM
            assert "priv_feature" in td
            td[ACTION_KEY] = torch.tensor([[0.25, -0.50]]).expand(
                *td.batch_size, 2
            )
        return td


class _ForbiddenStudentActor(nn.Module):
    def forward(self, td):
        raise AssertionError("actor_adapt must not influence Teacher rollout")


def test_rollout_advances_ema_memories_but_executes_only_teacher_action():
    calls = []
    owner = SimpleNamespace(
        cfg=SimpleNamespace(use_object_adapt=True),
        device=torch.device("cpu"),
        depth_feature_dim=2,
        temporal_depth_gru_ema=_StudentPerceptionStage("ema_depth", calls),
        object_adapt_ema=_StudentPerceptionStage("ema_object", calls),
        object_pred_transform=_StudentPerceptionStage(
            "object_pred_transform", calls
        ),
        adapt_ema=_StudentPerceptionStage("ema_adapt", calls),
        object_transform=_TeacherStage("teacher_object", calls),
        encoder_priv=_TeacherStage("teacher_encoder", calls),
        actor=_TeacherStage("teacher_actor", calls),
        actor_adapt=_ForbiddenStudentActor(),
    )
    rollout = _PrivilegedTeacherPerceptionRollout(owner)
    td = TensorDict(
        {REF_JPOS_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        batch_size=[2],
    )

    with set_exploration_type(ExplorationType.RANDOM):
        result = rollout(td)

    assert calls == [
        "ema_depth",
        "ema_object",
        "object_pred_transform",
        "ema_adapt",
        "teacher_object",
        "teacher_encoder",
        "teacher_actor",
    ]
    assert torch.equal(
        result[ACTION_KEY],
        torch.tensor([[1.25, 1.50], [3.25, 3.50]]),
    )
    assert torch.equal(result["next", "depth_hx"], torch.full((2, 2), 2.0))
    assert torch.equal(result["next", "adapt_hx"], torch.full((2, 2), 3.0))
    assert PRIV_PRED_KEY in result
    assert "_depth_feature" not in result
    assert OBJECT_PRED_KEY not in result
    assert OBJECT_PRED_TRANS_KEY not in result


def _bare_perception_policy():
    policy = TeacherRolloutPerceptionOnly.__new__(TeacherRolloutPerceptionOnly)
    nn.Module.__init__(policy)
    policy.temporal_depth_gru = nn.Linear(2, 2)
    policy.object_adapt = nn.Linear(2, 2)
    policy.adapt_module = nn.Linear(2, 2)
    policy.temporal_depth_gru_ema = nn.Linear(2, 2)
    policy.object_adapt_ema = nn.Linear(2, 2)
    policy.adapt_ema = nn.Linear(2, 2)
    policy.actor = nn.Linear(2, 2)
    policy.actor_adapt = nn.Linear(2, 2)
    policy.critic = nn.Linear(2, 2)
    policy.encoder_priv = nn.Linear(2, 2)
    parameters = [
        parameter
        for name in ONLINE_PERCEPTION_MODULES
        for parameter in getattr(policy, name).parameters()
    ]
    policy.opt_adapt = torch.optim.Adam(parameters, lr=3e-4)
    policy.opt_policy = torch.optim.Adam(policy.actor_adapt.parameters(), lr=3e-4)
    policy.opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
    return policy


class _LoaderTemporal(nn.Module):
    def __init__(self, depth_cnn: nn.Module):
        super().__init__()
        self.depth_cnn = depth_cnn
        self.recurrent = nn.Linear(2, 2)


def _bare_loader_policy(*, initialization: str, seed: int = 11):
    policy = TeacherRolloutPerceptionOnly.__new__(TeacherRolloutPerceptionOnly)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        perception_initialization=initialization,
        use_object_adapt=True,
        load_noise_scale=0.2,
    )
    policy.device = torch.device("cpu")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        depth_cnn = nn.Linear(2, 2)
        policy.depth_cnn = depth_cnn
        policy.temporal_depth_gru = _LoaderTemporal(depth_cnn)
        policy.temporal_depth_gru_ema = copy.deepcopy(
            policy.temporal_depth_gru
        ).requires_grad_(False)
        policy.object_adapt = nn.Linear(2, 2)
        policy.object_adapt_ema = copy.deepcopy(policy.object_adapt).requires_grad_(
            False
        )
        policy.adapt_module = nn.Linear(2, 2)
        policy.adapt_ema = copy.deepcopy(policy.adapt_module).requires_grad_(False)
        policy.actor = nn.Linear(2, 2)
        policy.actor_adapt = nn.Linear(2, 2)
        policy.encoder_priv = nn.Linear(2, 2)
        policy.critic = nn.Linear(2, 2)
    policy.opt_adapt = torch.optim.Adam(
        [
            parameter
            for name in ONLINE_PERCEPTION_MODULES
            for parameter in getattr(policy, name).parameters()
        ],
        lr=3e-4,
    )
    policy.opt_policy = torch.optim.Adam(policy.actor_adapt.parameters(), lr=3e-4)
    policy.opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
    policy.lr_policy = 3e-4
    progress = []
    policy.env = SimpleNamespace(set_progress=progress.append)

    # The production actor has a nested actor_std parameter.  These two hooks
    # are orthogonal to perception initialization and are covered separately;
    # bypass them so this loader fixture can use small deterministic modules.
    policy._restore_frozen_student_std = MethodType(
        lambda owner, source: None, policy
    )
    policy._verify_teacher_noise_scale = MethodType(lambda owner: None, policy)
    return policy, progress


def _filled_teacher_state(policy):
    state = {}
    for index, (name, module) in enumerate(policy.named_children(), start=1):
        module_state = copy.deepcopy(module.state_dict())
        for value in module_state.values():
            value.fill_(10.0 + index)
        state[name] = module_state
    state.update({"last_phase": "train", "last_iter": 37, "lr_policy": 1e-4})
    return state


def _assert_state_exact(actual, expected):
    assert tuple(actual) == tuple(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def _assert_modules_exact(left: nn.Module, right: nn.Module):
    _assert_state_exact(left.state_dict(), right.state_dict())


def test_fresh_initialization_skips_all_teacher_perception_and_syncs_ema():
    policy, progress = _bare_loader_policy(initialization="fresh")
    source = _filled_teacher_state(policy)
    constructor_perception = {
        name: copy.deepcopy(getattr(policy, name).state_dict())
        for name in FULL_PERCEPTION_CHECKPOINT_MODULES
    }

    # This deliberately supplies all seven perception mappings, including a
    # camera stack a normal train-phase Teacher does not have.  Fresh mode must
    # ignore them by ownership, not merely rely on the mappings being absent.
    fresh = policy.load_state_dict(source)

    assert tuple(fresh) == FULL_PERCEPTION_CHECKPOINT_MODULES
    for name, expected in constructor_perception.items():
        _assert_state_exact(getattr(policy, name).state_dict(), expected)
        assert any(
            not torch.equal(value, source[name][key])
            for key, value in expected.items()
        ), f"fixture must distinguish fresh {name} from Teacher state"

    for online_name, ema_name in (
        ("temporal_depth_gru", "temporal_depth_gru_ema"),
        ("object_adapt", "object_adapt_ema"),
        ("adapt_module", "adapt_ema"),
    ):
        online = getattr(policy, online_name)
        ema = getattr(policy, ema_name)
        _assert_modules_exact(online, ema)
        assert all(
            online_parameter.data_ptr() != ema_parameter.data_ptr()
            for online_parameter, ema_parameter in zip(
                online.parameters(), ema.parameters(), strict=True
            )
        )

    for name in ("actor", "actor_adapt", "encoder_priv", "critic"):
        _assert_state_exact(getattr(policy, name).state_dict(), source[name])
    assert progress == [37]
    assert policy.lr_policy == pytest.approx(1e-4)


def test_fresh_initialization_still_requires_every_non_perception_teacher_child():
    policy, _ = _bare_loader_policy(initialization="fresh")
    source = _filled_teacher_state(policy)
    source.pop("actor")

    with pytest.raises(ValueError, match="actor"):
        policy.load_state_dict(source)


def test_parameter_ownership_is_exactly_three_online_perception_modules():
    policy = _bare_perception_policy()

    policy._enforce_perception_only_ownership()

    for name in ONLINE_PERCEPTION_MODULES:
        module = getattr(policy, name)
        assert module.training
        assert all(parameter.requires_grad for parameter in module.parameters())
    for name in (*EMA_PERCEPTION_MODULES, "actor", "actor_adapt", "critic", "encoder_priv"):
        module = getattr(policy, name)
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert policy.opt_policy is None
    assert policy.opt_critic is None


def test_train_op_delegates_only_to_train_adapt_and_skips_value_bootstrap():
    policy = TeacherRolloutPerceptionOnly.__new__(TeacherRolloutPerceptionOnly)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(load_noise_scale=0.2)
    policy.num_updates = 0
    received = []

    def train_adapt(owner, batch):
        received.append(batch)
        assert "stats" not in batch
        return {"adapt/priv_loss": 1.5}

    policy.train_adapt = MethodType(train_adapt, policy)
    rollout = TensorDict(
        {
            "value": torch.ones(2, 3),
            "stats": TensorDict({"x": torch.ones(2, 3)}, batch_size=[2, 3]),
        },
        batch_size=[2, 3],
    )

    info = policy.train_op(rollout)

    assert len(received) == 1
    assert received[0] is not rollout
    assert "stats" in rollout
    assert info["adapt/priv_loss"] == pytest.approx(1.5)
    assert info["perception_only/teacher_control"] == pytest.approx(1.0)
    assert info["perception_only/update_count"] == 1
    assert policy.requires_value_bootstrap() is False
