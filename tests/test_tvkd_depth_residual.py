from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import CatTensors

from active_adaptation.learning.modules.rnn import set_recurrent_mode
from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule, GRUModule, PPOVEL,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC as TVKD,
    TVKDDistributionalFastSACTeacherBCConfig as Config,
    _validate_tvkd_algorithm_config,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    PRETRAINED_PERCEPTION_MODULES, PPOVEL_PARTIAL_PERCEPTION_MODULES,
)
from scripts.helpers import _fill_replayless_inference_algo_defaults
from omegaconf import OmegaConf


def adaptation(residual):
    core = DepthResidualGRUModule(8, 4) if residual else GRUModule(8)
    keys = ['_adapt_inp', 'is_init', 'adapt_hx']
    if residual:
        keys.append('_depth_feature')
    module = TensorDictSequential(
        CatTensors(['policy', 'object_pred'], '_adapt_inp', del_keys=False, sort=False),
        TensorDictModule(core, keys, ['priv_pred', ('next', 'adapt_hx')]),
    )
    module(TensorDict({'policy': torch.zeros(2, 3), 'object_pred': torch.zeros(2, 2),
                      'is_init': torch.zeros(2, 1, dtype=torch.bool),
                      'adapt_hx': torch.zeros(2, 8), '_depth_feature': torch.zeros(2, 4)}, [2]))
    return module


def warmstart_owner():
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config(perception_depth_residual=True)
    policy.adapt_module = adaptation(True)
    policy.adapt_ema = copy.deepcopy(policy.adapt_module).requires_grad_(False)
    return policy


@pytest.mark.parametrize('recurrent', [False, True])
def test_zero_branch_preserves_outputs_and_hidden_states_exactly(recurrent):
    old = adaptation(False)
    policy = warmstart_owner()
    source = {'adapt_module': old.state_dict(), 'adapt_ema': old.state_dict()}
    prepared = policy._prepare_perception_warmstart(source)
    for name in source:
        getattr(policy, name).load_state_dict(prepared[name], strict=True)
        assert not any('depth_projection' in key for key in source[name])
    shape = (3, 7) if recurrent else (3,)
    td = TensorDict({'policy': torch.randn(*shape, 3),
                     'object_pred': torch.randn(*shape, 2),
                     'is_init': torch.rand(*shape, 1) < 0.3,
                     'adapt_hx': torch.randn(*shape, 8),
                     '_depth_feature': torch.randn(*shape, 4)}, shape)
    with set_recurrent_mode(recurrent):
        baseline = old(td.clone())
        online = policy.adapt_module(td.clone())
        ema = policy.adapt_ema(td.clone())
    for key in ('priv_pred', ('next', 'adapt_hx')):
        assert torch.equal(baseline[key], online[key])
        assert torch.equal(baseline[key], ema[key])


def test_projection_learns_immediately_then_opens_direct_depth_gradient():
    core = DepthResidualGRUModule(8, 4)
    x = torch.randn(3, 5)
    reset = torch.zeros(3, 1, dtype=torch.bool)
    hx = torch.zeros(3, 8)
    depth = torch.randn(3, 4, requires_grad=True)
    output, _ = core(x, reset, hx, depth)
    optimizer = torch.optim.SGD(core.parameters(), lr=0.1)
    output.square().mean().backward()
    assert core.depth_projection.weight.grad.abs().sum() > 0
    # Zero projection means the new route initially has zero input derivative,
    # while its own weights already receive a useful gradient (no zero gate).
    assert torch.count_nonzero(depth.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    depth.grad = None
    output, _ = core(x, reset, hx, depth)
    output.square().mean().backward()
    assert depth.grad.abs().sum() > 0


def test_legacy_migration_never_hides_missing_base_or_partial_new_weights():
    policy = warmstart_owner()
    old = adaptation(False).state_dict()
    source = {'adapt_module': dict(old), 'adapt_ema': dict(old)}
    source['adapt_module'].pop(next(iter(old)))
    prepared = policy._prepare_perception_warmstart(source)
    with pytest.raises(RuntimeError, match='Missing key'):
        policy.adapt_module.load_state_dict(prepared['adapt_module'], strict=True)

    new_source = {'perception_depth_residual': True,
                  'adapt_module': policy.adapt_module.state_dict(), 'adapt_ema': dict(old)}
    prepared = policy._prepare_perception_warmstart(new_source)
    with pytest.raises(RuntimeError, match='depth_projection'):
        policy.adapt_ema.load_state_dict(prepared['adapt_ema'], strict=True)
    new_source['dagger_backend_config'] = {'perception_depth_residual': False}
    with pytest.raises(ValueError, match='perception_depth_residual'):
        policy._prepare_perception_warmstart(new_source)

    # A new model with both projections missing is damaged, even if only the
    # backend metadata survives. It must not be treated as an old model.
    damaged = {'adapt_module': dict(old), 'adapt_ema': dict(old),
               'dagger_backend_config': {'perception_depth_residual': True}}
    prepared = policy._prepare_perception_warmstart(damaged)
    with pytest.raises(RuntimeError, match='depth_projection'):
        policy.adapt_module.load_state_dict(prepared['adapt_module'], strict=True)
    del new_source['perception_depth_residual']
    del new_source['dagger_backend_config']
    prepared = policy._prepare_perception_warmstart(new_source)
    with pytest.raises(RuntimeError, match='depth_projection'):
        policy.adapt_ema.load_state_dict(prepared['adapt_ema'], strict=True)


def test_new_checkpoint_keeps_learned_projection_and_parameter_identity():
    policy = warmstart_owner()
    source = {'perception_depth_residual': True}
    for name in ('adapt_module', 'adapt_ema'):
        state = copy.deepcopy(getattr(policy, name).state_dict())
        for key in state:
            if 'depth_projection' in key:
                state[key].fill_(0.3)
        source[name] = state
    prepared = policy._prepare_perception_warmstart(source)
    assert prepared is source
    for name in ('adapt_module', 'adapt_ema'):
        module = getattr(policy, name)
        identities = [id(p) for p in module.parameters()]
        module.load_state_dict(prepared[name], strict=True)
        assert identities == [id(p) for p in module.parameters()]
        for key, value in module.state_dict().items():
            assert torch.equal(value, source[name][key])


def build_full_ppovel(residual):
    cfg = Config(perception_depth_residual=residual)
    cfg.latent_dim = 8
    cfg.enable_residual_distillation = False
    env = SimpleNamespace(cfg=SimpleNamespace(reward={'tracking': {}}),
                          action_manager=SimpleNamespace(joint_names=['left', 'right']))
    dimensions = {'policy': (10,), 'priv': (16,), 'command': (8,),
                  'vel_command': (5,), 'object_': (12,), 'object_geo_': (384,),
                  'depth': (1, 36, 64)}
    spec = Composite({key: Unbounded((2, *shape)) for key, shape in dimensions.items()}, shape=(2,))
    return PPOVEL(cfg, spec, Unbounded((2, 2)), Unbounded((2, 1)), 'cpu', env)


def test_full_constructor_preserves_rng_old_weights_and_optimizer_ownership():
    torch.manual_seed(51)
    old = build_full_ppovel(False)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(51)
    new = build_full_ppovel(True)
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, module in old.named_children():
        target_state = getattr(new, name).state_dict()
        for key, value in module.state_dict().items():
            assert torch.equal(value, target_state[key]), (name, key)
    optimizer_ids = [id(p) for group in new.opt_adapt.param_groups for p in group['params']]
    online_core = next(m for m in new.adapt_module.modules() if isinstance(m, DepthResidualGRUModule))
    ema_core = next(m for m in new.adapt_ema.modules() if isinstance(m, DepthResidualGRUModule))
    assert optimizer_ids.count(id(online_core.depth_projection.weight)) == 1
    assert id(ema_core.depth_projection.weight) not in optimizer_ids
    assert not ema_core.depth_projection.weight.requires_grad
    assert torch.count_nonzero(online_core.depth_projection.weight) == 0
    assert online_core.depth_projection.weight.shape == (8, 64)
    assert sum(p.numel() for p in new.actor_adapt.parameters()) == sum(p.numel() for p in old.actor_adapt.parameters())


@pytest.mark.parametrize('partial', [False, True])
def test_explicit_perception_overlay_loads_old_weights_and_zero_branch(tmp_path, partial):
    old = build_full_ppovel(False)
    new = build_full_ppovel(True)
    owner = TVKD.__new__(TVKD)
    nn.Module.__init__(owner)
    owner.cfg = new.cfg
    for name in PRETRAINED_PERCEPTION_MODULES:
        setattr(owner, name, getattr(new, name))
    selected = PPOVEL_PARTIAL_PERCEPTION_MODULES if partial else PRETRAINED_PERCEPTION_MODULES
    state = {name: getattr(old, name).state_dict() for name in selected}
    state['last_phase'] = 'train' if partial else 'finetune'
    path = tmp_path / 'old_perception.pt'
    torch.save({'policy': state}, path)
    metadata = owner._load_pretrained_perception_checkpoint(path)
    assert metadata['depth_residual_zero_initialized'] is True
    assert metadata['mode'] == ('ppo_vel_train_partial' if partial else 'strict_full_student')
    for name in selected:
        actual = getattr(owner, name).state_dict()
        for key, value in state[name].items():
            assert torch.equal(value, actual[key]), (name, key)
        for key in set(actual).difference(state[name]):
            assert key.endswith('depth_projection.weight')
            assert torch.count_nonzero(actual[key]) == 0


@pytest.mark.parametrize('field,value', [('adapt_module', 'mlp'), ('use_depth', False), ('phase', 'train')])
def test_invalid_architecture_is_rejected(field, value):
    cfg = Config(perception_depth_residual=True)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match='perception_depth_residual'):
        _validate_tvkd_algorithm_config(cfg)


@pytest.mark.parametrize('saved', [False, True])
def test_inference_uses_checkpoint_architecture(saved):
    cfg = OmegaConf.create({'algo': {'perception_depth_residual': not saved}})
    state = {'training_algorithm': 'distributional_tvkd_fastsac_teacher_bc_v9',
             'dagger_backend_config': {'perception_depth_residual': saved, 'value_norm': False},
             'perception_depth_residual': saved}
    _fill_replayless_inference_algo_defaults(cfg, state, inference_only=True)
    assert cfg.algo.perception_depth_residual == saved
    state['perception_depth_residual'] = not saved
    with pytest.raises(ValueError, match='perception_depth_residual'):
        _fill_replayless_inference_algo_defaults(cfg, state, inference_only=True)
