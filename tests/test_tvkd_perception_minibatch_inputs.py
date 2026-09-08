"""Input pruning and metric transfers must not change perception training."""

from __future__ import annotations

import copy
import importlib
import os
from types import MethodType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.ppo.ppo_vel import DepthResidualGRUModule, HEIGHT_KEY, PPOVEL
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DAGGER_IS_DAGGER_ENV_KEY,
    DAGGER_IS_STUDENT_ACTION_KEY,
    apply_perception_training_source,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
    TVKDDistributionalFastSACTeacherBC as TVKD,
    TVKDDistributionalFastSACTeacherBCConfig as Config,
)

ppovel = importlib.import_module("active_adaptation.learning.ppo.ppo_vel")


def _policy(*, residual=True, coefficient=1.0, device="cpu", latent_dim=8, train_every=4):
    """Real height/depth CNNs, object transform, GRUs and physical Actor."""
    cfg = Config(
        latent_dim=latent_dim, num_minibatches=8, train_every=train_every,
        perception_depth_residual=residual,
        perception_action_consistency_coef=coefficient,
        perception_training_source="all",
        sac_action_distribution="ppo_physical_gaussian",
    )
    apply_perception_training_source(cfg)
    env = SimpleNamespace(
        cfg=SimpleNamespace(reward={"tracking": {}}),
        action_manager=SimpleNamespace(joint_names=["left", "right"]),
    )
    dimensions = {
        "policy": (10,), "priv": (16,), "command": (8,),
        "vel_command": (5,), "object_": (12,), "object_geo_": (384,),
        "depth": (1, 36, 64), HEIGHT_KEY: (1, 36, 64),
    }
    spec = Composite(
        {key: Unbounded((2, *shape)) for key, shape in dimensions.items()}, shape=(2,),
    )
    source = PPOVEL(cfg, spec, Unbounded((2, 2)), Unbounded((2, 1)), "cpu", env)
    policy = TVKD.__new__(TVKD)
    nn.Module.__init__(policy)
    for name, module in source.named_children():
        setattr(policy, name, module)
    policy.cfg, policy.device = cfg, torch.device("cpu")
    policy.depth_feature_dim = source.depth_feature_dim
    policy.observation_spec = source.observation_spec
    policy.opt_adapt = source.opt_adapt
    policy._fastsac_q_action_scale = torch.tensor([0.5, 2.0])
    # The direct residual path must be active, not merely its zero initializer.
    if residual:
        with torch.no_grad():
            for module in policy.adapt_module.modules():
                if isinstance(module, DepthResidualGRUModule):
                    module.depth_projection.weight.fill_(0.03)
    policy.to(device)
    policy.device = torch.device(device)
    policy._fastsac_q_action_scale = policy._fastsac_q_action_scale.to(device)
    optimizer_ids = {id(parameter) for group in policy.opt_adapt.param_groups for parameter in group["params"]}
    assert optimizer_ids.issubset({id(parameter) for parameter in policy.parameters()})
    return policy


def _rollout(policy):
    generator = torch.Generator().manual_seed(1513)
    n, t = 16, int(policy.cfg.train_every)
    data = {
        key: torch.randn(n, t, *spec.shape[1:], generator=generator)
        for key, spec in policy.observation_spec.items()
    }
    data["policy"][..., 0] = torch.arange(n * t).reshape(n, t)
    data["is_init"] = torch.zeros(n, t, 1, dtype=torch.bool)
    data["is_init"][::2, 0] = True
    data["is_init"][3, 2] = True
    data["depth_hx"] = torch.randn(n, t, policy.depth_feature_dim, generator=generator)
    data["adapt_hx"] = torch.randn(n, t, policy.cfg.latent_dim, generator=generator)
    for key in ("depth_hx", "adapt_hx"):
        data[key].masked_fill_(data["is_init"], 0)
    data[DAGGER_IS_DAGGER_ENV_KEY] = (torch.arange(n) < 4).unsqueeze(1).expand(n, t).clone()
    data[DAGGER_IS_STUDENT_ACTION_KEY] = torch.ones(n, t, dtype=torch.bool)
    data[DAGGER_IS_STUDENT_ACTION_KEY][:4, 1::2] = False
    data["collector_diagnostics"] = torch.randn(n, t, 2048, generator=generator)
    data["priv_pred"] = torch.randn(n, t, policy.cfg.latent_dim, generator=generator)
    data["next", "reward"] = torch.randn(n, t, 1, generator=generator)
    return TensorDict(data, [n, t]).to(policy.device)


def _with_targets(policy, rollout):
    result = rollout.clone()
    with torch.no_grad():
        policy.object_transform(result)
        policy.height_encoder(result)
        policy.encoder_priv(result)
    return result


def _assert_tree_equal(left, right, path="root"):
    if torch.is_tensor(left):
        assert torch.equal(left, right), f"Tensor mismatch at {path}, shape={tuple(left.shape)}"
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_tree_equal(first, second, f"{path}[{index}]")
    else:
        assert left == right


def test_selection_is_shallow_and_drops_only_unconsumed_fields():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1501)
        policy = _policy()
    full = _with_targets(policy, _rollout(policy))
    selected = policy._perception_minibatch_inputs(full)
    expected = {
        "is_init", "priv_feature", "object_", "depth", "depth_hx",
        "policy", "vel_command", "object_geo_", "adapt_hx", "_height_feature",
    }
    assert set(selected.keys(True, True)) == expected
    assert selected.batch_size == full.batch_size
    for key in expected:
        assert selected[key] is full[key]
    for key in (HEIGHT_KEY, "priv", "command", "collector_diagnostics", "priv_pred"):
        assert key in full and key not in selected
    assert selected.bytes() < full.bytes()


def _compare_training(monkeypatch, *, residual, coefficient, device="cpu", latent_dim=8, train_every=4):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1521)
        optimized = _policy(residual=residual, coefficient=coefficient, device=device,
                            latent_dim=latent_dim, train_every=train_every)
    original = copy.deepcopy(optimized)
    original._perception_minibatch_inputs = MethodType(PPOVEL._perception_minibatch_inputs, original)
    rollout = _rollout(optimized)
    required = tuple(optimized._perception_minibatch_inputs(_with_targets(optimized, rollout)).keys(True, True))
    make_batch = ppovel.make_batch
    traces, states, rng_states, metrics = [], [], [], []
    transferred_metrics = ppovel._scalar_metrics_to_python

    for index, policy in enumerate((original, optimized)):
        batches, gradients, steps, losses, payload_sizes = [], [], [], [], []

        def record_batches(inputs, *args):
            payload_sizes.append(inputs.bytes())
            for batch in make_batch(inputs, *args):
                batches.append({key: batch[key].detach().clone() for key in required})
                yield batch

        optimizer_step = policy.opt_adapt.step
        parameters = [parameter for group in policy.opt_adapt.param_groups for parameter in group["params"]]

        def record_step(*args, **kwargs):
            gradients.append([None if parameter.grad is None else parameter.grad.detach().clone()
                              for parameter in parameters])
            result = optimizer_step(*args, **kwargs)
            steps.append([parameter.detach().clone() for parameter in parameters])
            return result

        loss_hook = policy.adapt_loss_fn.register_forward_hook(
            lambda module, args, result: losses.append(result.detach().clone())
        )
        with monkeypatch.context() as patch:
            patch.setattr(ppovel, "make_batch", record_batches)
            patch.setattr(policy.opt_adapt, "step", record_step)
            # Reference keeps the previous per-key scalar extraction too.
            patch.setattr(ppovel, "_scalar_metrics_to_python",
                          (lambda values: {key: value.item() for key, value in values.items()})
                          if index == 0 else transferred_metrics)
            torch.manual_seed(1523)
            metrics.append(policy.train_adapt(rollout.clone()))
            rng_states.append((torch.get_rng_state().clone(),
                               torch.cuda.get_rng_state(device).clone() if optimized.device.type == "cuda" else None))
        loss_hook.remove()
        assert len(steps) == len(batches) == 16  # 2 epochs * original 8 minibatches.
        assert len(losses) == 32
        traces.append((batches, gradients, steps, losses))
        states.append({"parameters": dict((name, value.detach().clone()) for name, value in policy.named_parameters()),
                       "optimizer": copy.deepcopy(policy.opt_adapt.state_dict()), "payload": payload_sizes})

    _assert_tree_equal(traces[0], traces[1])
    _assert_tree_equal(states[0]["parameters"], states[1]["parameters"])
    _assert_tree_equal(states[0]["optimizer"], states[1]["optimizer"])
    assert metrics[0] == metrics[1]
    _assert_tree_equal(rng_states[0], rng_states[1])
    assert all(new < old for old, new in zip(states[0]["payload"], states[1]["payload"]))
    # Every factual row appears twice, including Teacher-action DAgger rows.
    rows = torch.cat([batch["policy"][..., 0].flatten() for batch in traces[1][0]]).long().cpu()
    assert torch.equal(rows.bincount(), torch.full((rollout.numel(),), 2))
    teacher_rows = (~rollout[DAGGER_IS_STUDENT_ACTION_KEY]).flatten().cpu()
    assert teacher_rows.any() and (~teacher_rows).any()
    assert torch.equal(rows.bincount()[teacher_rows], torch.full((int(teacher_rows.sum()),), 2))
    assert all(parameter.grad is None for parameter in optimized.actor_adapt.parameters())


@pytest.mark.parametrize("residual,coefficient", [(False, 0.0), (False, 1.0), (True, 1.0)])
def test_pruned_and_original_full_training_are_bitwise_identical(monkeypatch, residual, coefficient):
    _compare_training(monkeypatch, residual=residual, coefficient=coefficient)


@pytest.mark.skipif(
    os.environ.get("VAIC_EXACT_CUDA_TESTS") != "1",
    reason="Set VAIC_EXACT_CUDA_TESTS=1 for the bounded idle-GPU training equality check",
)
@pytest.mark.parametrize("precision,train_every,latent_dim", [
    ("high", 4, 8), ("highest", 4, 8), ("high", 32, 256), ("highest", 32, 256),
])
def test_cuda_deterministic_pruned_training_matches_all_original_steps_bitwise(monkeypatch, precision, train_every, latent_dim):
    from benchmark_exact_online_encoder import require_idle_cuda

    allowed = [name for name in os.environ.get("VAIC_EXACT_CUDA_ALLOW_PROCESS", "").split(",") if name]
    try:
        require_idle_cuda(allowed)
    except RuntimeError as error:
        pytest.skip(str(error))
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_threads = torch.get_num_threads()
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = precision == "high"
        # Default cuDNN convolution backward is not bit-reproducible even in
        # old-vs-old controls. Scope deterministic kernels to this equality
        # test only; neither production determinism nor precision is changed.
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[0]):
            _compare_training(monkeypatch, residual=True, coefficient=1.0, device="cuda:0",
                              latent_dim=latent_dim, train_every=train_every)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("custom", ["unknown_module", "auxiliary", "actor_distillation", "dr_estimator"])
def test_custom_training_contracts_preserve_the_full_input(custom):
    policy = _policy()
    full = _with_targets(policy, _rollout(policy))
    if custom == "unknown_module":
        policy.object_pred_transform = nn.Identity()
    elif custom == "auxiliary":
        policy._perception_auxiliary_loss = MethodType(lambda self, td: (None, {}), policy)
    elif custom == "actor_distillation":
        policy.cfg.enable_residual_distillation = True
    else:
        policy.cfg.train_dr_estimator = True
    assert policy._perception_minibatch_inputs(full) is full
    assert PPOVEL._perception_minibatch_inputs(policy, full) is full


def test_metric_transfer_preserves_mixed_dtypes_values_order_and_groups(monkeypatch):
    values = {
        "float64": torch.tensor(1.0 + 2.0 ** -40, dtype=torch.float64),
        "float32": torch.tensor(-0.0),
        "half": torch.tensor(0.3333, dtype=torch.float16),
        "second32": torch.tensor(float("inf")),
        "bfloat": torch.tensor(0.3333, dtype=torch.bfloat16),
    }
    calls = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        calls.append((tensor.device, tensor.dtype, tensor.numel()))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    actual = ppovel._scalar_metrics_to_python(values)
    assert list(actual) == list(values)
    assert actual == {key: value.item() for key, value in values.items()}
    assert len(calls) == 4  # Two float32 scalars share one transfer, no casting.
    assert any(dtype == torch.float32 and count == 2 for _, dtype, count in calls)
    assert actual["float64"] != float(torch.tensor(actual["float64"], dtype=torch.float32))
