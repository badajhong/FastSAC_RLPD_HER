from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from active_adaptation.learning.ppo.perception_actor import (
    ACTOR_BC_PERCEPTION_SOURCE,
    ACTOR_OBJECTIVE_SEMANTICS,
    OPTIMIZED_MODULES,
    ROLLOUT_EMA_PRIV_PRED_KEY,
    TRAINING_ALGORITHM,
    TeacherRolloutPerceptionActor,
)
from active_adaptation.learning.ppo.perception_only import (
    EMA_PERCEPTION_MODULES,
    FULL_PERCEPTION_CHECKPOINT_MODULES,
    ONLINE_PERCEPTION_MODULES,
)
from active_adaptation.learning.ppo.ppo_vel import (
    PPOVEL,
    PRIV_PRED_KEY,
    REF_JPOS_KEY,
)
from scripts import percetpion, percetpion_actor


CONFIG_DIR = str((Path(__file__).parents[1] / "cfg").resolve())
ACTOR_STD_KEY = "module.0.module.2.module.actor_std"


def _compose_config(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="percetpion_actor", overrides=list(overrides))


def test_exact_user_hydra_surface_and_iteration_budget(monkeypatch):
    cfg = _compose_config(
        "task=G1/vaic/skateboard_general_tracking_stu",
        "iteration=3000",
        "task.num_envs=512",
        "algo.load_noise_scale=0.2",
        "algo.num_minibatches=8",
        "algo.perception_initialization=fresh",
    )
    monkeypatch.setattr(percetpion.aa, "get_world_size", lambda: 1)

    percetpion.apply_perception_iteration_controls(cfg)
    TeacherRolloutPerceptionActor._validate_config(cfg.algo)

    assert cfg.algo._target_.endswith(".TeacherRolloutPerceptionActor")
    assert cfg.algo.name == "teacher_rollout_perception_actor"
    assert cfg.algo.phase == "finetune"
    assert cfg.algo.enable_residual_distillation is True
    assert cfg.algo.distill_with_priv_pred is True
    assert cfg.algo.actor_bc_perception_source == ACTOR_BC_PERCEPTION_SOURCE
    assert cfg.algo.perception_initialization == "fresh"
    assert cfg.total_frames == 3000 * 512 * 32


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ("algo.enable_residual_distillation=false", "enable actor distillation"),
        ("algo.distill_with_priv_pred=false", "distill_with_priv_pred"),
        ("algo.actor_bc_perception_source=online", "online Actor-BC input"),
    ),
)
def test_config_rejects_non_predicted_latent_actor_bc(override, message):
    cfg = _compose_config(override)

    with pytest.raises(ValueError, match=message):
        TeacherRolloutPerceptionActor._validate_config(cfg.algo)


def test_actor_bc_target_restores_absolute_action_coordinates():
    policy = TeacherRolloutPerceptionActor.__new__(
        TeacherRolloutPerceptionActor
    )
    residual_mean = torch.tensor(
        [[[0.1, -0.2], [0.3, 0.4]], [[-0.5, 0.6], [0.7, -0.8]]]
    )
    reference = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    batch = TensorDict({REF_JPOS_KEY: reference}, batch_size=[2, 2])
    teacher_dist = SimpleNamespace(mean=residual_mean)

    target = policy._actor_distillation_target_mean(batch, teacher_dist)

    torch.testing.assert_close(
        target,
        residual_mean + reference,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(KeyError, match=REF_JPOS_KEY):
        policy._actor_distillation_target_mean(
            TensorDict({}, batch_size=[2, 2]),
            teacher_dist,
        )
    with pytest.raises(RuntimeError, match="shapes differ"):
        policy._actor_distillation_target_mean(
            TensorDict(
                {REF_JPOS_KEY: torch.zeros(2, 2, 3)},
                batch_size=[2, 2],
            ),
            teacher_dist,
        )


def test_default_ppovel_distillation_hook_preserves_existing_coordinates():
    policy = PPOVEL.__new__(PPOVEL)
    mean = torch.randn(2, 3)
    teacher_dist = SimpleNamespace(mean=mean)

    result = policy._actor_distillation_target_mean(
        TensorDict({}, batch_size=[2]),
        teacher_dist,
    )

    assert result is mean


def test_default_ppovel_actor_distillation_still_selects_online_priv_pred():
    policy = PPOVEL.__new__(PPOVEL)
    online = torch.randn(2, 3)
    batch = TensorDict({PRIV_PRED_KEY: online}, batch_size=[2])

    assert policy._actor_distillation_priv_pred(batch) is online


def test_joint_stage_inherits_exact_ppovel_perception_training_method():
    assert TeacherRolloutPerceptionActor.train_adapt is PPOVEL.train_adapt


class _LatentSpyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def get_dist(self, tensordict):
        self.seen = tensordict[PRIV_PRED_KEY]
        return SimpleNamespace(mean=self.seen, scale=torch.ones_like(self.seen))


def test_perception_actor_distillation_uses_only_detached_rollout_ema_latent():
    policy = TeacherRolloutPerceptionActor.__new__(
        TeacherRolloutPerceptionActor
    )
    nn.Module.__init__(policy)
    policy.actor_adapt = _LatentSpyActor()
    online = torch.full((2, 3), 13.0, requires_grad=True)
    rollout_ema = torch.full((2, 3), -7.0, requires_grad=True)
    batch = TensorDict(
        {
            PRIV_PRED_KEY: online,
            ROLLOUT_EMA_PRIV_PRED_KEY: rollout_ema,
        },
        batch_size=[2],
    )

    policy._actor_distillation_student_dist(batch)

    torch.testing.assert_close(policy.actor_adapt.seen, rollout_ema.detach())
    assert not policy.actor_adapt.seen.requires_grad
    assert batch[PRIV_PRED_KEY] is online

    with pytest.raises(RuntimeError, match="online priv_pred fallback is forbidden"):
        policy._actor_distillation_priv_pred(
            TensorDict({PRIV_PRED_KEY: online}, batch_size=[2])
        )


def _bare_policy(*, initialization="fresh"):
    policy = TeacherRolloutPerceptionActor.__new__(
        TeacherRolloutPerceptionActor
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="finetune",
        perception_initialization=initialization,
        use_object_adapt=True,
        load_noise_scale=0.2,
    )
    policy.device = torch.device("cpu")
    policy.depth_cnn = nn.Linear(2, 2)
    policy.temporal_depth_gru = nn.Linear(2, 2)
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

    perception_parameters = [
        parameter
        for name in ONLINE_PERCEPTION_MODULES
        for parameter in getattr(policy, name).parameters()
    ]
    policy.opt_adapt = torch.optim.Adam(perception_parameters, lr=3e-4)
    policy.opt_adapt_actor = torch.optim.Adam(
        policy.actor_adapt.parameters(),
        lr=3e-4,
    )
    policy.opt_policy = torch.optim.Adam(policy.actor_adapt.parameters(), lr=3e-4)
    policy.opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
    policy.lr_policy = 3e-4
    policy.num_updates = 4
    progress = []
    policy.env = SimpleNamespace(
        set_progress=progress.append,
        current_iter=19,
    )
    policy._actor_adapt_loaded_from_teacher_checkpoint = False
    policy._restore_frozen_student_std = MethodType(
        lambda owner, source: None,
        policy,
    )
    policy._verify_teacher_noise_scale = MethodType(lambda owner: None, policy)
    return policy, progress


def _optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_parameter_ownership_is_perception_plus_actor_adapt_only():
    policy, _ = _bare_policy()

    policy._enforce_perception_only_ownership()

    perception_ids = {
        id(parameter)
        for name in ONLINE_PERCEPTION_MODULES
        for parameter in getattr(policy, name).parameters()
    }
    actor_ids = {id(parameter) for parameter in policy.actor_adapt.parameters()}
    assert _optimizer_parameter_ids(policy.opt_adapt) == perception_ids
    assert _optimizer_parameter_ids(policy.opt_adapt_actor) == actor_ids
    assert perception_ids.isdisjoint(actor_ids)

    for name in (*ONLINE_PERCEPTION_MODULES, "actor_adapt"):
        module = getattr(policy, name)
        assert module.training
        assert all(parameter.requires_grad for parameter in module.parameters())
    for name in (
        *EMA_PERCEPTION_MODULES,
        "actor",
        "encoder_priv",
        "critic",
    ):
        module = getattr(policy, name)
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert policy.opt_policy is None
    assert policy.opt_critic is None


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


def test_fresh_load_preserves_perception_but_warmstarts_actor_adapt():
    policy, progress = _bare_policy()
    source = _filled_teacher_state(policy)
    fresh_states = {
        name: copy.deepcopy(getattr(policy, name).state_dict())
        for name in FULL_PERCEPTION_CHECKPOINT_MODULES
    }
    actor_parameter_ids = {
        id(parameter) for parameter in policy.actor_adapt.parameters()
    }
    actor_optimizer_ids = _optimizer_parameter_ids(policy.opt_adapt_actor)

    fresh = policy.load_state_dict(source)

    assert tuple(fresh) == FULL_PERCEPTION_CHECKPOINT_MODULES
    for name, expected in fresh_states.items():
        _assert_state_exact(getattr(policy, name).state_dict(), expected)
    _assert_state_exact(policy.actor.state_dict(), source["actor"])
    _assert_state_exact(policy.actor_adapt.state_dict(), source["actor_adapt"])
    assert {
        id(parameter) for parameter in policy.actor_adapt.parameters()
    } == actor_parameter_ids
    assert _optimizer_parameter_ids(policy.opt_adapt_actor) == actor_optimizer_ids
    assert policy._actor_adapt_loaded_from_teacher_checkpoint is True
    assert all(parameter.requires_grad for parameter in policy.actor_adapt.parameters())
    assert all(not parameter.requires_grad for parameter in policy.actor.parameters())
    assert progress == [37]


def test_checkpoint_metadata_proves_actor_warmstart_and_training_contract():
    policy, _ = _bare_policy()
    policy._actor_adapt_loaded_from_teacher_checkpoint = True
    policy._enforce_perception_only_ownership()

    state = policy.state_dict()

    assert state["training_algorithm"] == TRAINING_ALGORITHM
    assert state["actor_objective_semantics"] == ACTOR_OBJECTIVE_SEMANTICS
    assert state["actor_adapt_loaded_from_teacher_checkpoint"] is True
    assert state["actor_adapt_trained"] is True
    assert state["actor_adapt_controls_rollout"] is False
    assert state["actor_bc_perception_source"] == ACTOR_BC_PERCEPTION_SOURCE
    assert state["actor_bc_uses_online_priv_pred"] is False
    assert state["actor_adapt_bc_update_count"] == 4
    assert tuple(state["optimized_modules"]) == OPTIMIZED_MODULES


def _write_teacher_checkpoint(path: Path, task_name: str):
    policy = {name: {} for name in percetpion.REQUIRED_TEACHER_MODULES}
    policy["actor"] = {ACTOR_STD_KEY: torch.full((3,), 0.35)}
    policy["actor_adapt"] = {ACTOR_STD_KEY: torch.full((3,), 0.45)}
    policy.update({"last_phase": "train", "last_iter": 123})
    torch.save(
        {
            "policy": policy,
            "cfg": {
                "algo": {
                    "phase": "train",
                    "enable_residual_distillation": True,
                    "name": "ppo_vel",
                    "_target_": (
                        "active_adaptation.learning.ppo.ppo_vel.PPOVEL"
                    ),
                },
                "task": {"name": task_name},
            },
            "vecnorm": {},
        },
        path,
    )


def test_entry_validation_reuses_strict_teacher_checkpoint_audit(tmp_path):
    cfg = _compose_config("algo.perception_initialization=fresh")
    checkpoint = tmp_path / "teacher.pt"
    _write_teacher_checkpoint(checkpoint, str(cfg.task.name))
    cfg.checkpoint_path = str(checkpoint)

    audit = percetpion_actor.validate_perception_actor_training_config(cfg)

    assert audit["path"] == str(checkpoint.resolve())
    assert cfg.checkpoint_path == str(checkpoint.resolve())


def test_train_op_routes_only_joint_adaptation_and_uses_canonical_metrics():
    policy = TeacherRolloutPerceptionActor.__new__(
        TeacherRolloutPerceptionActor
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(load_noise_scale=0.2, latent_dim=4)
    policy.num_updates = 0
    received = []

    def train_adapt(owner, batch):
        received.append(batch)
        assert "stats" not in batch
        assert PRIV_PRED_KEY not in batch.keys(True, True)
        torch.testing.assert_close(
            batch[ROLLOUT_EMA_PRIV_PRED_KEY],
            torch.full((2, 3, 4), -7.0),
        )
        return {"adapt/priv_loss": 1.5, "adapt/adapt_loss": 0.25}

    policy.train_adapt = MethodType(train_adapt, policy)
    rollout = TensorDict(
        {
            "value": torch.ones(2, 3),
            PRIV_PRED_KEY: torch.full((2, 3, 4), -7.0),
            "stats": TensorDict({"x": torch.ones(2, 3)}, batch_size=[2, 3]),
        },
        batch_size=[2, 3],
    )

    info = policy.train_op(rollout)

    assert len(received) == 1
    assert info["adapt/adapt_loss"] == pytest.approx(0.25)
    assert info["perception_actor/update_count"] == 1
    assert info["perception_actor/teacher_control"] == pytest.approx(1.0)
    assert info["perception_actor/rollout_ema_priv_pred_input"] == pytest.approx(1.0)
    assert PRIV_PRED_KEY in rollout.keys(True, True)
    assert ROLLOUT_EMA_PRIV_PRED_KEY not in rollout.keys(True, True)
    assert not any(key.startswith("perception_only/") for key in info)


def test_train_op_rejects_missing_rollout_ema_latent_without_online_fallback():
    policy = TeacherRolloutPerceptionActor.__new__(
        TeacherRolloutPerceptionActor
    )
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(load_noise_scale=0.2, latent_dim=4)
    policy.num_updates = 0

    with pytest.raises(RuntimeError, match="did not provide EMA priv_pred"):
        policy.train_op(TensorDict({"value": torch.ones(2, 3)}, [2, 3]))
