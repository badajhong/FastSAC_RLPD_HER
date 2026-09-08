"""Focused contract tests for the corrected depth-CNN initialization."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.ppo.perception_pipeline_contract import (
    CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    DEPTH_INIT_ONLY_CONTRACT,
    DEPTH_NO_VECNORM_ONLY_CONTRACT,
    LEGACY_PERCEPTION_PIPELINE_CONTRACT,
)
from active_adaptation.learning.ppo.ppo_vel import (
    DEPTH_KEY,
    DepthResidualGRUModule,
    PPOConfig,
    PPOVEL,
    TemporalDepthGRU,
)


class _Env:
    def __init__(self):
        self.cfg = SimpleNamespace(reward={"tracking": {}})
        self.action_manager = SimpleNamespace(joint_names=["left", "right"])
        self.current_iter = 7

    def set_progress(self, iteration):
        self.current_iter = int(iteration)


def _build_policy(
    contract=LEGACY_PERCEPTION_PIPELINE_CONTRACT,
    *,
    omit_contract=False,
):
    cfg = PPOConfig(
        phase="finetune",
        enable_residual_distillation=False,
        latent_dim=8,
        perception_pipeline_contract=contract,
    )
    if omit_contract:
        del cfg.perception_pipeline_contract
    # This is a TVKD extension read through getattr by the shared PPOVEL
    # implementation. It makes the zero residual part of the isolation test.
    cfg.perception_depth_residual = True
    dimensions = {
        "policy": (10,),
        "priv": (16,),
        "command": (8,),
        "vel_command": (5,),
        "object_": (12,),
        "object_geo_": (384,),
        "depth": (1, 36, 64),
        "ref_joint_pos_": (2,),
    }
    spec = Composite(
        {key: Unbounded((2, *shape)) for key, shape in dimensions.items()},
        shape=(2,),
    )
    return PPOVEL(
        cfg,
        spec,
        Unbounded((2, 2)),
        Unbounded((2, 1)),
        "cpu",
        _Env(),
    )


def test_explicit_legacy_contract_is_exactly_the_historical_default():
    """Adding the contract selector must not perturb legacy weights or output."""

    torch.manual_seed(1699)
    explicit = _build_policy(LEGACY_PERCEPTION_PIPELINE_CONTRACT)
    explicit_rng = torch.get_rng_state().clone()
    torch.manual_seed(1699)
    historical = _build_policy(omit_contract=True)

    assert torch.equal(torch.get_rng_state(), explicit_rng)
    for (left_name, left), (right_name, right) in zip(
        explicit.named_parameters(), historical.named_parameters(), strict=True
    ):
        assert left_name == right_name
        assert torch.equal(left, right), left_name
    for (left_name, left), (right_name, right) in zip(
        explicit.named_buffers(), historical.named_buffers(), strict=True
    ):
        assert left_name == right_name
        assert torch.equal(left, right), left_name

    depth = torch.linspace(0.0, 100.0, 2 * 36 * 64).reshape(2, 1, 36, 64)
    with torch.no_grad():
        explicit_feature = explicit.depth_cnn(depth)
        historical_feature = historical.depth_cnn(depth)
    assert torch.equal(explicit_feature, historical_feature)


def _orthogonal_gain(module: nn.Module) -> float:
    weight = module.weight.detach().flatten(1)
    if weight.shape[0] <= weight.shape[1]:
        gram = weight @ weight.T
    else:
        gram = weight.T @ weight
    diagonal = gram.diagonal()
    off_diagonal = gram - torch.diag_embed(diagonal)
    assert torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=2e-5)
    assert torch.allclose(diagonal, diagonal.new_full(diagonal.shape, diagonal[0]), rtol=2e-4)
    return math.sqrt(float(diagonal[0]))


def _depth_cnn_modules(policy):
    convolutions = [
        module for module in policy.depth_cnn.modules() if isinstance(module, nn.Conv2d)
    ]
    linears = [
        module for module in policy.depth_cnn.modules() if isinstance(module, nn.Linear)
    ]
    assert len(convolutions) == 3
    assert len(linears) == 1
    return convolutions, linears


def test_corrected_contract_changes_only_depth_cnn_weights_and_preserves_rng():
    torch.manual_seed(1701)
    legacy = _build_policy(LEGACY_PERCEPTION_PIPELINE_CONTRACT)
    legacy_rng = torch.get_rng_state().clone()

    torch.manual_seed(1701)
    corrected = _build_policy(CORRECTED_PERCEPTION_PIPELINE_CONTRACT)
    assert torch.equal(torch.get_rng_state(), legacy_rng)

    legacy_parameters = dict(legacy.named_parameters())
    corrected_parameters = dict(corrected.named_parameters())
    assert legacy_parameters.keys() == corrected_parameters.keys()
    changed = {
        name
        for name in legacy_parameters
        if not torch.equal(legacy_parameters[name], corrected_parameters[name])
    }
    expected_changed = {
        name
        for name in legacy_parameters
        if name.endswith("weight")
        and (
            name.startswith("depth_cnn.")
            or name.startswith("temporal_depth_gru_ema.depth_cnn.")
        )
        and legacy_parameters[name].ndim >= 2
    }
    assert changed == expected_changed

    convolutions, linears = _depth_cnn_modules(corrected)
    assert all(_orthogonal_gain(module) == pytest.approx(math.sqrt(2.0), rel=2e-4)
               for module in convolutions)
    assert _orthogonal_gain(linears[0]) == pytest.approx(1.0, rel=2e-4)
    assert all(torch.count_nonzero(module.bias) == 0 for module in convolutions + linears)

    assert corrected.temporal_depth_gru.state_dict().keys() == (
        corrected.temporal_depth_gru_ema.state_dict().keys()
    )
    for key, value in corrected.temporal_depth_gru.state_dict().items():
        assert torch.equal(value, corrected.temporal_depth_gru_ema.state_dict()[key])
    core = next(
        module
        for module in corrected.adapt_module.modules()
        if isinstance(module, DepthResidualGRUModule)
    )
    assert torch.count_nonzero(core.depth_projection.weight) == 0


def test_corrected_initialization_restores_a_distinguishable_visual_signal():
    torch.manual_seed(1702)
    legacy = _build_policy(LEGACY_PERCEPTION_PIPELINE_CONTRACT)
    torch.manual_seed(1702)
    corrected = _build_policy(CORRECTED_PERCEPTION_PIPELINE_CONTRACT)
    image_generator = torch.Generator().manual_seed(1704)
    images = torch.stack(
        [torch.zeros(1, 36, 64), torch.rand(1, 36, 64, generator=image_generator)]
    )

    with torch.no_grad():
        legacy_features = legacy.depth_cnn(images)
        corrected_features = corrected.depth_cnn(images)
    legacy_distance = torch.mean(
        (legacy_features[0] - legacy_features[1]).square()
    ).sqrt()
    corrected_distance = torch.mean(
        (corrected_features[0] - corrected_features[1]).square()
    ).sqrt()

    # Broad thresholds lock the conditioning change without depending on an
    # exact BLAS/QR implementation or a particular CPU/GPU backend.
    assert legacy_distance < 1e-5
    assert corrected_distance > 1e-2
    assert corrected_distance > legacy_distance * 1e4


@pytest.mark.parametrize(
    ("left_contract", "right_contract"),
    [
        (LEGACY_PERCEPTION_PIPELINE_CONTRACT, DEPTH_NO_VECNORM_ONLY_CONTRACT),
        (DEPTH_INIT_ONLY_CONTRACT, CORRECTED_PERCEPTION_PIPELINE_CONTRACT),
    ],
)
def test_two_by_two_contracts_select_only_the_expected_initialization(
    left_contract, right_contract
):
    torch.manual_seed(1703)
    left = _build_policy(left_contract)
    left_rng = torch.get_rng_state().clone()
    torch.manual_seed(1703)
    right = _build_policy(right_contract)
    assert torch.equal(torch.get_rng_state(), left_rng)
    for (left_name, left_value), (right_name, right_value) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert left_name == right_name
        assert torch.equal(left_value, right_value), left_name


def test_contract_is_strict_and_missing_field_defaults_to_legacy():
    invalid_cfg = PPOConfig(phase="finetune")
    invalid_cfg.perception_pipeline_contract = "corrected"
    with pytest.raises(ValueError, match="unsupported perception_pipeline_contract"):
        _build_policy(invalid_cfg.perception_pipeline_contract)

    base_cfg = PPOConfig(
        phase="finetune",
        enable_residual_distillation=False,
        latent_dim=8,
    )
    missing_cfg = SimpleNamespace(
        **{
            key: value
            for key, value in vars(base_cfg).items()
            if key != "perception_pipeline_contract"
        }
    )
    missing_cfg.perception_depth_residual = True
    dimensions = {
        "policy": (10,), "priv": (16,), "command": (8,),
        "vel_command": (5,), "object_": (12,), "object_geo_": (384,),
        "depth": (1, 36, 64), "ref_joint_pos_": (2,),
    }
    spec = Composite(
        {key: Unbounded((2, *shape)) for key, shape in dimensions.items()}, shape=(2,)
    )
    policy = PPOVEL(
        missing_cfg, spec, Unbounded((2, 2)), Unbounded((2, 1)), "cpu", _Env()
    )
    assert policy.perception_pipeline_contract == LEGACY_PERCEPTION_PIPELINE_CONTRACT


def test_checkpoint_records_contract_and_rejects_unknown_saved_value():
    policy = _build_policy(CORRECTED_PERCEPTION_PIPELINE_CONTRACT)
    state = policy.state_dict()
    assert state["perception_pipeline_contract"] == CORRECTED_PERCEPTION_PIPELINE_CONTRACT

    damaged = copy.copy(state)
    damaged["perception_pipeline_contract"] = "unknown_v9"
    with pytest.raises(ValueError, match="unsupported perception_pipeline_contract"):
        policy.load_state_dict(damaged)

    # Missing metadata is deliberately interpreted as historical legacy_v1;
    # specialized loaders decide whether a cross-phase warm start is allowed.
    owner = PPOVEL.__new__(PPOVEL)
    nn.Module.__init__(owner)
    owner.env = _Env()
    owner.lr_policy = 3e-4
    owner.opt_policy = None
    assert owner.load_state_dict({}) == []


@pytest.mark.parametrize(
    ("saved_contract", "runtime_contract"),
    [
        (
            LEGACY_PERCEPTION_PIPELINE_CONTRACT,
            CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
        ),
        (
            CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
            LEGACY_PERCEPTION_PIPELINE_CONTRACT,
        ),
    ],
)
def test_generic_full_depth_checkpoint_rejects_pipeline_mismatch(
    saved_contract,
    runtime_contract,
):
    source = _build_policy(saved_contract)
    target = _build_policy(runtime_contract)

    with pytest.raises(
        ValueError,
        match="checkpoint perception_pipeline_contract mismatch",
    ):
        target.load_state_dict(source.state_dict())


def test_generic_depthless_teacher_checkpoint_allows_corrected_warmstart():
    owner = PPOVEL.__new__(PPOVEL)
    nn.Module.__init__(owner)
    owner.perception_pipeline_contract = CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    owner.env = _Env()
    owner.lr_policy = 3e-4
    owner.opt_policy = None

    assert owner.load_state_dict(
        {
            "perception_pipeline_contract": (
                LEGACY_PERCEPTION_PIPELINE_CONTRACT
            ),
            "last_phase": "train",
        }
    ) == []


def test_generic_depthless_finetune_checkpoint_cannot_bypass_contract_guard():
    owner = PPOVEL.__new__(PPOVEL)
    nn.Module.__init__(owner)
    owner.perception_pipeline_contract = CORRECTED_PERCEPTION_PIPELINE_CONTRACT
    owner.env = _Env()
    owner.lr_policy = 3e-4
    owner.opt_policy = None

    with pytest.raises(
        ValueError,
        match="checkpoint perception_pipeline_contract mismatch",
    ):
        owner.load_state_dict(
            {
                "perception_pipeline_contract": (
                    LEGACY_PERCEPTION_PIPELINE_CONTRACT
                ),
                "last_phase": "finetune",
            }
        )


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, float("nan"), float("inf")])
def test_identity_depth_contract_checks_first_real_cnn_input_once(bad_value):
    module = TemporalDepthGRU(nn.Flatten(start_dim=1), hidden_dim=64)
    module.enable_unit_interval_depth_check()

    def input_td(value):
        return TensorDict(
            {
                DEPTH_KEY: torch.full((2, 1, 8, 8), value),
                "is_init": torch.ones(2, 1, dtype=torch.bool),
                "depth_hx": torch.zeros(2, 64),
            },
            [2],
        )

    with pytest.raises(ValueError, match="corrected depth CNN"):
        module(input_td(bad_value))
    assert not module._unit_interval_depth_checked

    module(input_td(0.5))
    assert module._unit_interval_depth_checked
    assert module.first_depth_input_stats == {
        "min": pytest.approx(0.5),
        "max": pytest.approx(0.5),
        "p1": pytest.approx(0.5),
        "p99": pytest.approx(0.5),
        "sample_count": 128,
        "total_count": 128,
    }


def test_depth_diagnostic_quantiles_are_bounded_one_shot_and_rng_free():
    module = TemporalDepthGRU(nn.Flatten(start_dim=1), hidden_dim=64)
    module.enable_unit_interval_depth_check()
    depth = torch.linspace(0.0, 1.0, 10_001)
    torch.manual_seed(1705)
    rng_before = torch.get_rng_state().clone()

    module._validate_first_unit_interval_depth(depth)

    assert torch.equal(torch.get_rng_state(), rng_before)
    stats = copy.deepcopy(module.first_depth_input_stats)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["p1"] == pytest.approx(0.01, abs=2e-3)
    assert stats["p99"] == pytest.approx(0.99, abs=2e-3)
    assert stats["sample_count"] <= 4096
    assert stats["total_count"] == depth.numel()

    # Once accepted, subsequent calls incur neither reductions nor updates.
    module._validate_first_unit_interval_depth(torch.full_like(depth, 100.0))
    assert module.first_depth_input_stats == stats
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_unit_interval_check_is_enabled_after_fake_materialization_only():
    corrected = _build_policy(CORRECTED_PERCEPTION_PIPELINE_CONTRACT)
    assert corrected.temporal_depth_gru._check_unit_interval_depth
    assert not corrected.temporal_depth_gru._unit_interval_depth_checked
    assert corrected.temporal_depth_gru.first_depth_input_stats is None
    assert corrected.temporal_depth_gru_ema._check_unit_interval_depth
    assert not corrected.temporal_depth_gru_ema._unit_interval_depth_checked
    assert corrected.temporal_depth_gru_ema.first_depth_input_stats is None

    legacy = _build_policy(LEGACY_PERCEPTION_PIPELINE_CONTRACT)
    assert not legacy.temporal_depth_gru._check_unit_interval_depth
