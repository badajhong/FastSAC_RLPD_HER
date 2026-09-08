from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType, set_interaction_type
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.modules.rnn import set_recurrent_mode
from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule, PPOConfig, PPOVEL, ZeroDepthInjector,
)
from active_adaptation.learning.ppo.td3_bc_dagger import DistributionalTD3TeacherBC
from active_adaptation.learning.ppo.fastsac_bc_dagger import DistributionalFastSACTeacherBC
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC,
)
from scripts.eval_oracles import load_oracle_teacher, make_eval_policy


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def ppo_factory():
    policies = {}

    def build(*, depth=True, height=False, objects=True):
        key = depth, height, objects
        if key in policies:
            return policies[key]
        cfg = PPOConfig(
            phase="finetune", latent_dim=8, enable_residual_distillation=False,
            use_depth=depth, use_object_adapt=objects,
        )
        cfg.perception_depth_residual = depth
        dimensions = {
            "policy": (10,), "priv": (16,), "command": (8,),
            "vel_command": (5,), "object_": (22,), "object_geo_": (384,),
        }
        if depth:
            dimensions["depth"] = (1, 36, 64)
        if height:
            dimensions["height"] = (1, 36, 64)
        spec = Composite(
            {name: Unbounded((2, *shape)) for name, shape in dimensions.items()},
            shape=(2,),
        )
        env = SimpleNamespace(
            cfg=SimpleNamespace(reward={"tracking": {}}),
            action_manager=SimpleNamespace(joint_names=["left", "right"]),
        )
        with torch.random.fork_rng():
            torch.manual_seed(61)
            owner = PPOVEL(cfg, spec, Unbounded((2, 2)), Unbounded((2, 1)), "cpu", env)
        if depth:
            # A learned residual makes preservation of the depth route observable.
            core = next(m for m in owner.adapt_ema.modules()
                        if isinstance(m, DepthResidualGRUModule))
            with torch.no_grad():
                core.depth_projection.weight.copy_(
                    torch.linspace(-0.2, 0.3, core.depth_projection.weight.numel())
                    .reshape_as(core.depth_projection.weight)
                )
        policies[key] = owner
        return owner

    return build


def observation(owner):
    with torch.random.fork_rng():
        torch.manual_seed(62)
        td = owner.observation_spec.rand()
        # These are already in the environment's observation coordinates.
        td["priv"] = (torch.arange(32).reshape(2, 16).float() - 12) / 3
        td["object_"] = torch.linspace(-1.7, 2.3, 44).reshape(2, 22)
        td["is_init"] = torch.zeros(2, 1, dtype=torch.bool)
        td["adapt_hx"] = torch.randn(2, owner.cfg.latent_dim)
        td["depth_hx"] = torch.randn(2, owner.depth_feature_dim)
    return td


def forbid_forward(monkeypatch, module):
    def fail(*args, **kwargs):
        raise AssertionError("An oracle bypass called an unused model")
    monkeypatch.setattr(module, "forward", fail)


def evaluate(policy, td):
    with torch.no_grad(), set_recurrent_mode(False), set_interaction_type(InteractionType.MODE):
        return policy(td)


def test_disabled_flags_return_original_policy_without_rebuilding():
    sentinel = object()
    owner = SimpleNamespace(get_rollout_policy=Mock(return_value=sentinel))
    assert make_eval_policy(owner) is sentinel
    owner.get_rollout_policy.assert_called_once_with("eval")


@pytest.mark.parametrize("flag", ["oracle_object_pose", "oracle_priv_pred"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_oracle_flags_require_boolean_values(flag, value):
    with pytest.raises(ValueError, match=flag):
        make_eval_policy(SimpleNamespace(), **{flag: value})


@pytest.mark.parametrize("depth", [False, True])
def test_object_oracle_preserves_full_target_geometry_and_recurrent_state(
    ppo_factory, monkeypatch, depth,
):
    owner = ppo_factory(depth=depth)
    td = observation(owner)
    expected = td.clone()
    with torch.no_grad(), set_recurrent_mode(False):
        if depth:
            owner.temporal_depth_gru_ema(expected)
        else:
            ZeroDepthInjector(owner.depth_feature_dim, owner.device)(expected)
        expected["object_pred"] = expected["object_"].clone()
        owner.object_pred_transform(expected)
        owner.adapt_ema(expected)
        expected_action = owner.actor_adapt.get_dist(expected).mean
    forbid_forward(monkeypatch, owner.object_adapt_ema)
    forbid_forward(monkeypatch, owner.encoder_priv)
    actual = evaluate(make_eval_policy(owner, oracle_object_pose=True), td.clone())

    assert torch.equal(actual["object_pred"], td["object_"])
    assert actual["object_pred"].shape[-1] == 22
    assert actual["object_pred"].data_ptr() != actual["object_"].data_ptr()
    points = td["object_geo_"].reshape(2, 128, 3)
    pose = td["object_"][..., -12:]
    transformed = points @ pose[..., 3:].reshape(2, 3, 3).transpose(-1, -2)
    transformed += pose[..., :3].unsqueeze(-2)
    torch.testing.assert_close(actual["object_pred_trans"], transformed.flatten(-2))
    for key in ("priv_pred", "_depth_feature", ("next", "adapt_hx")):
        torch.testing.assert_close(actual[key], expected[key])
    torch.testing.assert_close(actual["action"], expected_action)
    if depth:
        torch.testing.assert_close(actual["next", "depth_hx"], expected["next", "depth_hx"])

    # Carried hidden states must influence the next live adaptation step.
    carried = td.clone()
    carried["adapt_hx"] = actual["next", "adapt_hx"].clone()
    if depth:
        carried["depth_hx"] = actual["next", "depth_hx"].clone()
    next_actual = evaluate(make_eval_policy(owner, oracle_object_pose=True), carried)
    assert not torch.equal(next_actual["next", "adapt_hx"], actual["next", "adapt_hx"])
    assert torch.equal(next_actual["object_pred"], td["object_"])


@pytest.mark.parametrize("height", [False, True])
@pytest.mark.parametrize("both", [False, True])
def test_priv_oracle_uses_teacher_latent_and_bypasses_all_student_perception(
    ppo_factory, monkeypatch, height, both,
):
    owner = ppo_factory(height=height)
    td = observation(owner).exclude("depth", "depth_hx", "adapt_hx", "is_init")
    expected = td.clone()
    with torch.no_grad():
        owner.object_transform(expected)
        if height:
            owner.height_encoder(expected)
        owner.encoder_priv(expected)
        expected["priv_pred"] = expected["priv_feature"].clone()
        expected_action = owner.actor_adapt.get_dist(expected).mean
    for name in ("temporal_depth_gru_ema", "object_adapt_ema", "object_pred_transform", "adapt_ema", "actor"):
        forbid_forward(monkeypatch, getattr(owner, name))
    actual = evaluate(
        make_eval_policy(owner, oracle_object_pose=both, oracle_priv_pred=True), td.clone(),
    )
    assert torch.equal(actual["priv"], td["priv"])
    assert torch.equal(actual["priv_pred"], expected["priv_feature"])
    assert actual["priv_pred"].data_ptr() != actual["priv_feature"].data_ptr()
    torch.testing.assert_close(actual["action"], expected_action)
    assert "object_pred" not in actual


def test_priv_oracle_precedence_allows_no_object_adaptation(ppo_factory):
    owner = ppo_factory(objects=False)
    with pytest.raises(ValueError, match="use_object_adapt"):
        make_eval_policy(owner, oracle_object_pose=True)
    actual = evaluate(
        make_eval_policy(owner, oracle_object_pose=True, oracle_priv_pred=True), observation(owner),
    )
    assert torch.equal(actual["priv_pred"], actual["priv_feature"])


def test_teacher_training_phase_rejects_student_oracles(ppo_factory, monkeypatch):
    owner = ppo_factory()
    monkeypatch.setattr(owner.cfg, "phase", "train")
    with pytest.raises(ValueError, match="finetune"):
        make_eval_policy(owner, oracle_priv_pred=True)


class LatentActor(nn.Module):
    def get_dist(self, td):
        return SimpleNamespace(mean=td["priv_pred"])


@pytest.mark.parametrize("backend,distribution", [
    (DistributionalTD3TeacherBC, "normalized_tanh"),
    (DistributionalFastSACTeacherBC, "normalized_tanh"),
    (DistributionalFastSACTeacherBC, "ppo_physical_gaussian"),
    (TVKDDistributionalFastSACTeacherBC, "ppo_physical_gaussian"),
])
def test_priv_oracle_preserves_backend_action_mapping(backend, distribution):
    owner = backend.__new__(backend)
    nn.Module.__init__(owner)
    owner.cfg = SimpleNamespace(
        phase="finetune", use_object_adapt=True, train_dr_estimator=False,
        sac_action_distribution=distribution, sac_log_std_min=-5., sac_log_std_max=0.,
        action_support_clip=100.,
    )
    owner.actor_adapt = LatentActor()
    owner.object_transform = TensorDictModule(nn.Identity(), ["object_"], ["object_trans"])
    owner.encoder_priv = TensorDictModule(nn.Identity(), ["priv"], ["priv_feature"])
    owner._fastsac_action_low = torch.tensor([-10., -12.])
    owner._fastsac_action_high = torch.tensor([10., 12.])
    owner._fastsac_actor_action_center = torch.tensor([0.5, -0.25])
    owner._fastsac_actor_action_scale = torch.tensor([1.5, 2.25])
    owner._fastsac_q_action_center = torch.tensor([1., -0.5])
    owner._fastsac_q_action_scale = torch.tensor([2., 2.5])
    owner.bc_dagger_sac_adapter = SimpleNamespace(log_std=torch.zeros(2))
    raw_mean = torch.tensor([[2., -3.], [20., -30.]])
    td = TensorDict({"priv": raw_mean, "object_": torch.zeros(2, 12)}, [2])

    actual = evaluate(make_eval_policy(owner, oracle_priv_pred=True), td)
    if backend is DistributionalTD3TeacherBC:
        center, scale = owner._fastsac_actor_action_center, owner._fastsac_actor_action_scale
        expected = center + scale * torch.tanh((raw_mean - center) / scale)
    elif distribution == "ppo_physical_gaussian":
        expected = raw_mean.clamp(owner._fastsac_action_low, owner._fastsac_action_high)
    else:
        center, scale = owner._fastsac_q_action_center, owner._fastsac_q_action_scale
        zero = -center / scale
        expected = center + scale * torch.tanh(torch.atanh(zero) + raw_mean / (scale * (1 - zero.square())))
    torch.testing.assert_close(actual["action"], expected)
    # The refactored ordinary evaluation path retains the same conversion.
    owner._student_raw_action_proposal = lambda td: raw_mean
    torch.testing.assert_close(owner._student_mean_action(td), expected)


@pytest.mark.parametrize("height", [False, True])
def test_load_oracle_teacher_restores_required_modules_strictly(height):
    owner = SimpleNamespace(encoder_priv=nn.Linear(3, 2))
    if height:
        owner.height_encoder = nn.Linear(4, 3)
    names = ["encoder_priv"] + (["height_encoder"] if height else [])
    checkpoint = {
        name: {key: torch.full_like(value, 0.7) for key, value in getattr(owner, name).state_dict().items()}
        for name in names
    }
    load_oracle_teacher(owner, checkpoint)
    for name in names:
        for value in getattr(owner, name).state_dict().values():
            assert torch.equal(value, torch.full_like(value, 0.7))
    for name in names:
        missing_module = dict(checkpoint)
        del missing_module[name]
        with pytest.raises(ValueError):
            load_oracle_teacher(owner, missing_module)
        damaged = copy.deepcopy(checkpoint)
        del damaged[name]["weight"]
        with pytest.raises(RuntimeError, match="Missing key"):
            load_oracle_teacher(owner, damaged)
        damaged = copy.deepcopy(checkpoint)
        damaged[name]["weight"] = torch.zeros(1, 1)
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_oracle_teacher(owner, damaged)


def test_priv_oracle_requires_teacher_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        load_oracle_teacher(SimpleNamespace(encoder_priv=nn.Linear(3, 2)), None)
