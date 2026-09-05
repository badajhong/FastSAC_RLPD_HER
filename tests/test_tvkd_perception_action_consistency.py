from __future__ import annotations

import copy
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.transforms import CatTensors

from active_adaptation.learning.ppo.common import Actor
from active_adaptation.learning.ppo.ppo_vel import (
    DepthResidualGRUModule, GRUModule, PPOVEL, TemporalDepthGRU,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC as TVKD,
    TVKDDistributionalFastSACTeacherBCConfig as Config,
    _validate_tvkd_algorithm_config,
)
from scripts.TVKD_fasSAC_bc_dagger import validate_tvkd_fastsac_bc_dagger_config


class SmallActor(nn.Module):
    """Exercise the real FastSAC physical distribution and flat-input adapter."""

    def __init__(self):
        super().__init__()
        self.core = Actor(2, init_noise_scale=0.1)
        self.core(torch.zeros(1, 7))

    def get_dist(self, td):
        features = torch.cat((td['vel_command'], td['policy'], td['priv_pred']), -1)
        mean, scale = self.core(features)
        return torch.distributions.Normal(mean, scale)


def bare_policy(coefficient=1.0):
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config(perception_action_consistency_coef=coefficient)
    policy.cfg.sac_action_distribution = 'ppo_physical_gaussian'
    policy.actor_adapt = SmallActor()
    policy.observation_spec = {
        'vel_command': SimpleNamespace(shape=(1,)),
        'policy': SimpleNamespace(shape=(2,)),
    }
    policy._fastsac_q_action_scale = torch.tensor([0.5, 2.0])
    return policy


def batch():
    return TensorDict({
        'vel_command': torch.randn(2, 4, 1, requires_grad=True),
        'policy': torch.randn(2, 4, 2, requires_grad=True),
        'priv_pred': torch.randn(2, 4, 4, requires_grad=True),
        'priv_feature': torch.randn(2, 4, 4, requires_grad=True),
        'is_init': torch.tensor([[True, False, False, False],
                                 [False, False, True, False]]).unsqueeze(-1),
    }, batch_size=[2, 4])


def test_action_loss_matches_normalized_mean_actions_and_only_trains_latent():
    policy = bare_policy(0.7)
    td = batch()
    policy.actor_adapt.core.actor_mean.bias.requires_grad_(False)
    parameters = tuple(policy.actor_adapt.parameters())
    flags = tuple(p.requires_grad for p in parameters)
    values = [p.detach().clone() for p in parameters]
    rng_before = torch.get_rng_state().clone()
    loss, metrics = policy._perception_auxiliary_loss(td)
    assert tuple(p.requires_grad for p in parameters) == flags
    assert torch.equal(rng_before, torch.get_rng_state())
    with torch.no_grad():
        predicted = policy.actor_adapt.get_dist(td).mean
        oracle_td = td.clone(False)
        oracle_td['priv_pred'] = td['priv_feature']
        oracle = policy.actor_adapt.get_dist(oracle_td).mean
        valid = ~td['is_init'].squeeze(-1)
        expected = (((predicted - oracle) / policy._fastsac_q_action_scale)
                    .square().mean(-1)[valid]).mean()
    torch.testing.assert_close(loss, expected * 0.7)
    torch.testing.assert_close(metrics['adapt/action_consistency_loss'], expected)
    assert all(not value.requires_grad for value in metrics.values())
    loss.backward()
    assert td['priv_pred'].grad.abs().sum() > 0
    assert torch.count_nonzero(td['priv_pred'].grad[~valid]) == 0
    for key in ('priv_feature', 'policy', 'vel_command'):
        assert td[key].grad is None
    assert all(p.grad is None for p in parameters)
    assert all(torch.equal(p, before) for p, before in zip(parameters, values))

    # The same actor remains trainable on the next normal actor update.
    optimizer = torch.optim.SGD([p for p in parameters if p.requires_grad], lr=0.1)
    policy.actor_adapt.get_dist(td.detach()).mean.square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(p, before) for p, before in zip(parameters, values))


@pytest.mark.parametrize('all_reset', [False, True])
def test_identical_latents_or_all_reset_rows_have_zero_loss(all_reset):
    policy = bare_policy()
    td = batch()
    if all_reset:
        td['is_init'].fill_(True)
    else:
        td['priv_feature'] = td['priv_pred'].detach().clone()
    loss, _ = policy._perception_auxiliary_loss(td)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(td['priv_pred'].grad) == 0


def test_disabled_hook_never_reads_actor_or_batch():
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = Config()
    assert policy._perception_auxiliary_loss(TensorDict({}, [])) == (None, {})
    assert PPOVEL._perception_auxiliary_loss(policy, TensorDict({}, [])) == (None, {})


def test_actor_flags_are_restored_when_prediction_fails(monkeypatch):
    policy = bare_policy()
    flags = [p.requires_grad for p in policy.actor_adapt.parameters()]
    def fail(_):
        raise RuntimeError('forward failed')
    monkeypatch.setattr(policy, '_actor_dist_from_flat', fail)
    with pytest.raises(RuntimeError, match='forward failed'):
        policy._perception_auxiliary_loss(batch())
    assert [p.requires_grad for p in policy.actor_adapt.parameters()] == flags


class PrivilegedTarget(nn.Module):
    def forward(self, td):
        td['priv_feature'] = td['priv']
        return td


class ZeroReconstructionLoss(nn.Module):
    def forward(self, prediction, target):
        # Isolate action supervision: no reconstruction gradient can explain
        # any change in the perception modules in the integration test.
        return prediction.square() * 0.0


@pytest.mark.parametrize('depth_residual', [False, True])
def test_live_recurrent_training_updates_entire_perception_and_ema_only(depth_residual):
    torch.manual_seed(29)
    policy = bare_policy()
    policy.device = torch.device('cpu')
    policy.cfg.num_minibatches = 2
    policy.cfg.train_every = 4
    policy.cfg.perception_live_env_scope = 'all'
    policy.cfg.enable_residual_distillation = False
    policy.cfg.train_dr_estimator = False
    policy.cfg.perception_depth_residual = depth_residual
    policy.object_transform = nn.Identity()
    policy.object_pred_transform = nn.Identity()
    policy.encoder_priv = PrivilegedTarget()
    policy.temporal_depth_gru = TemporalDepthGRU(nn.Linear(3, 4), hidden_dim=4)
    policy.object_adapt = TensorDictModule(nn.Linear(4, 2), ['_depth_feature'], ['object_pred'])
    adaptation_core = DepthResidualGRUModule(4, 4) if depth_residual else GRUModule(4)
    adaptation_keys = ['_adapt_inp', 'is_init', 'adapt_hx']
    if depth_residual:
        adaptation_keys.append('_depth_feature')
    policy.adapt_module = TensorDictSequential(
        CatTensors(['policy', 'object_pred'], '_adapt_inp', sort=False, del_keys=False),
        TensorDictModule(adaptation_core, adaptation_keys,
                         ['priv_pred', ('next', 'adapt_hx')]),
    )
    rollout = TensorDict({
        'depth': torch.randn(4, 4, 3),
        'policy': torch.randn(4, 4, 2),
        'vel_command': torch.randn(4, 4, 1),
        'priv': torch.randn(4, 4, 4),
        'object_': torch.randn(4, 4, 2),
        'is_init': torch.zeros(4, 4, 1, dtype=torch.bool),
        'depth_hx': torch.zeros(4, 4, 4),
        'adapt_hx': torch.zeros(4, 4, 4),
    }, [4, 4])
    # Materialize lazy adaptation weights using the production module.
    initialization = rollout[:, 0].clone()
    policy.temporal_depth_gru(initialization)
    policy.object_adapt(initialization)
    policy.adapt_module(initialization)
    pairs = [('temporal_depth_gru', 'temporal_depth_gru_ema'),
             ('object_adapt', 'object_adapt_ema'), ('adapt_module', 'adapt_ema')]
    for online, ema in pairs:
        setattr(policy, ema, copy.deepcopy(getattr(policy, online)).requires_grad_(False))
    policy.adapt_loss_fn = ZeroReconstructionLoss()
    policy.opt_adapt = torch.optim.SGD(
        [p for online, _ in pairs for p in getattr(policy, online).parameters()], lr=0.01)
    before = {name: copy.deepcopy(getattr(policy, name).state_dict())
              for pair in pairs for name in pair}
    actor_before = copy.deepcopy(policy.actor_adapt.state_dict())
    result = policy.train_adapt(rollout)
    assert result['adapt/action_consistency_loss'] > 0
    assert result['adapt/priv_loss'] == result['adapt/object_loss'] == 0
    for online, ema in pairs:
        online_state = getattr(policy, online).state_dict()
        assert any(not torch.equal(value, before[online][key])
                   for key, value in online_state.items())
        for key, value in getattr(policy, ema).state_dict().items():
            torch.testing.assert_close(value, before[ema][key] * 0.96 + online_state[key] * 0.04)
    for key, value in policy.actor_adapt.state_dict().items():
        assert torch.equal(value, actor_before[key])
    assert all(p.grad is None for p in policy.actor_adapt.parameters())
    if depth_residual:
        assert adaptation_core.depth_projection.weight.abs().sum() > 0
        assert result['adapt/depth_residual_weight_norm'] > 0


@pytest.mark.parametrize('coefficient', [-1, float('nan'), float('inf'), True, '1'])
def test_rejects_invalid_coefficient(coefficient):
    cfg = Config(perception_action_consistency_coef=coefficient)
    with pytest.raises(ValueError, match='perception_action_consistency_coef'):
        _validate_tvkd_algorithm_config(cfg)


@pytest.mark.parametrize('field,value,message', [
    ('train_perception', False, 'train_perception'),
    ('perception_replay_mode', 'four_way', 'online_student_rollout'),
])
def test_rejects_inactive_perception_paths(field, value, message):
    cfg = Config(perception_action_consistency_coef=1.0)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=message):
        _validate_tvkd_algorithm_config(cfg)


def test_user_command_composes_and_validates_without_launching_training():
    root = Path(__file__).resolve().parents[1]
    commands = (root / 'train_command.txt').read_text().split('nohup ')
    command = next(c for c in commands if 'algo.perception_action_consistency_coef=1.0' in c)
    words = shlex.split(command.replace('\\\n', ' ').split(' > ')[0])
    start = words.index('scripts/TVKD_fasSAC_bc_dagger.py') + 1
    overrides = [word for word in words[start:] if '=' in word]
    with initialize_config_dir(config_dir=str(root / 'cfg'), version_base=None):
        cfg = compose(config_name='TVKD_fasSAC_bc_dagger', overrides=overrides)
    validate_tvkd_fastsac_bc_dagger_config(cfg)
    assert cfg.algo.perception_action_consistency_coef == 1.0
    assert cfg.algo.perception_depth_residual is True
    assert cfg.algo.train_perception is True
    assert cfg.algo.perception_live_env_scope == 'all'
    assert cfg.algo.sac_action_distribution == 'ppo_physical_gaussian'
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    policy.cfg = cfg.algo
    assert policy._checkpoint_config()['perception_action_consistency_coef'] == 1.0
