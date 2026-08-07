import copy
import functools
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.fastsac_vel import (
    BC_DAGGER_ACTOR_BACKEND,
    BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
    BC_DAGGER_IQL_CRITIC_SEMANTICS,
    BC_DAGGER_LEGACY_TRAINING_ALGORITHM,
    BC_DAGGER_TRAINING_ALGORITHM,
    FASTSAC_BC_DAGGER_ACTOR_BACKEND,
    STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY,
    STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY,
    FastSACTanhNormal,
    FastSACVelFinetune,
    FastSACVelFinetuneConfig,
    OnlineReplay,
    _BCDaggerSACAdapter,
    _BCDaggerSACRolloutActor,
    _validate_fastsac_finetune_config,
)
from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_bc_dagger import (
    _ClippedStudentRolloutPolicy,
)
from active_adaptation.learning.ppo.ppo_vel import PPOVEL


_DELETE = object()


def _stage2_policy(*, load_pretrained_q=True):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        load_pretrained_q=load_pretrained_q,
        q_hidden_dim=8,
        q_num_atoms=11,
        q_v_min=-2.0,
        q_v_max=2.0,
        q_layer_norm=False,
        use_object_adapt=False,
        sac_bc_action_clip=20.0,
        sac_bc_initial_action_std=0.05,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
        q_seed=17,
        sac_alpha_init=0.01,
    )
    policy.actor_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.bc_dagger_actor_anchor = copy.deepcopy(
        policy.actor_adapt
    ).requires_grad_(False)
    policy.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=1,
        initial_log_std=math.log(
            policy.cfg.sac_bc_initial_action_std
            / policy.cfg.sac_bc_action_clip
        ),
        device="cpu",
    )
    policy.qnet = torch.nn.Linear(1, 1, bias=False)
    policy.qnet_target = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.qnet.weight.zero_()
        policy.qnet_target.weight.zero_()
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.q_rng = torch.Generator().manual_seed(1)
    policy.sac_action_rng = torch.Generator().manual_seed(2)
    policy.sac_rollout_rng = torch.Generator().manual_seed(3)
    policy.teacher_replay_id = "stage2-replay"
    policy._replay_vecnorm_fingerprint = "frozen-vecnorm"
    return policy


def test_stage2_checkpoint_before_q_updates_needs_no_teacher_export_state(
    monkeypatch,
):
    policy = _stage2_policy()
    policy.cfg = FastSACVelFinetuneConfig(
        finetune_checkpoint_source="bc_dagger"
    )
    policy.env = SimpleNamespace(current_iter=0)
    policy.actor_backend = FASTSAC_BC_DAGGER_ACTOR_BACKEND
    policy.teacher_replay = None
    policy.offline_replay = None
    policy.q_update_count = 0
    policy.sac_environment_steps = 0
    policy.sac_rollout_count = 0
    policy.sac_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy.num_updates = 0
    monkeypatch.setattr(policy, "_actor_backend_metadata", lambda: {})
    monkeypatch.setattr(policy, "_q_backend_metadata", lambda: {})

    assert not hasattr(policy, "_teacher_export_started")
    assert not hasattr(policy, "_teacher_export_start_seen")

    checkpoint = policy.state_dict()

    assert checkpoint["last_phase"] == "finetune"
    assert checkpoint["q_update_count"] == 0
    assert checkpoint["teacher_export_started"] is False


def test_stage2_loads_pretrained_q_by_default_and_validates_boolean_option():
    cfg = FastSACVelFinetuneConfig()
    assert cfg.load_pretrained_q is True

    cfg.load_pretrained_q = "false"
    with pytest.raises(ValueError, match="load_pretrained_q"):
        _validate_fastsac_finetune_config(cfg)


def test_stochastic_stage2_resume_requires_checkpointed_rollout_rng():
    policy = _stage2_policy()
    policy.cfg.sac_deterministic_rollout = False

    with pytest.raises(ValueError, match="rollout RNG state"):
        policy.load_state_dict({
            "last_phase": "finetune",
            "qnet": {},
        })


@pytest.mark.parametrize(
    ("override", "load_pretrained_q", "message"),
    [
        (
            {"q_action_coordinates": "reference_residual"},
            False,
            "fields",
        ),
        ({"q_reference_dueling": True}, False, "fields"),
        ({"q_condition_on_actuator_state": True}, False, "fields"),
        ({"q_action_fusion": "late"}, True, "early-fusion"),
        ({"sac_q_action_input_gain": 2.0}, True, "unit gain"),
        ({"sac_q_normalize_actions": True}, True, "normalize_actions=false"),
    ],
)
def test_bc_dagger_q_coordinates_must_match_checkpoint_and_replay_schema(
    override, load_pretrained_q, message
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    values = {
        "q_action_coordinates": "absolute",
        "q_reference_dueling": False,
        "q_condition_on_actuator_state": False,
        "q_action_fusion": "early",
        "sac_q_action_input_gain": 1.0,
        "sac_q_normalize_actions": False,
        "load_pretrained_q": load_pretrained_q,
    }
    values.update(override)
    policy.cfg = SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        policy._configure_bc_dagger_actor_backend()


def _iql_checkpoint():
    return {
        "training_algorithm": BC_DAGGER_TRAINING_ALGORITHM,
        "actor_backend": BC_DAGGER_ACTOR_BACKEND,
        "critic_learning_semantics": BC_DAGGER_IQL_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "vecnorm_fingerprint": "frozen-vecnorm",
        "dagger_backend_config": {
            "dagger_action_clip": 20.0,
            "q_hidden_dim": 8,
            "q_num_atoms": 11,
            "q_v_min": -2.0,
            "q_v_max": 2.0,
            "q_layer_norm": False,
            "iql_expectile": 0.7,
        },
        "qnet": {"weight": torch.tensor([[3.0]])},
        "qnet_target": {"weight": torch.tensor([[-4.0]])},
        # Stage 2 requires proof that Stage 1 really trained V, but it does not
        # instantiate or transfer this IQL-only bootstrap network.
        "iql_value": {"net.0.weight": torch.tensor([[9.0]])},
        "q_update_count": 123,
        "iql_value_update_count": 119,
        "teacher_replay_id": "dagger-replay",
        "teacher_replay_state": {"snapshot_id": "dagger-snapshot"},
    }


def test_stage2_rejects_legacy_bc_dagger_checkpoint_before_module_copy(
    monkeypatch,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("legacy checkpoint reached module-copy path")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _iql_checkpoint()
    state["training_algorithm"] = BC_DAGGER_LEGACY_TRAINING_ALGORITHM

    with pytest.raises(ValueError, match="predate IQL critic training"):
        policy.load_state_dict(state)


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        (
            "critic_learning_semantics",
            "plain-bellman-q",
            ValueError,
            "IQL critic semantics mismatch",
        ),
        (
            "actor_learning_semantics",
            "q-weighted-actor",
            ValueError,
            "not trained by pure DAgger BC",
        ),
        (
            "iql_value",
            _DELETE,
            KeyError,
            "missing its IQL value network",
        ),
        (
            "q_update_count",
            0,
            RuntimeError,
            "Q1/Q2 were never updated",
        ),
        (
            "iql_value_update_count",
            0,
            RuntimeError,
            "IQL V was never updated",
        ),
    ],
)
def test_stage2_requires_complete_iql_v2_training_provenance(
    monkeypatch, field, value, exception, message,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("invalid provenance reached module-copy path")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _iql_checkpoint()
    if value is _DELETE:
        del state[field]
    else:
        state[field] = value

    with pytest.raises(exception, match=message):
        policy.load_state_dict(state)


def test_stage2_transfers_iql_q_twins_and_targets_but_discards_stage1_v(
    monkeypatch,
):
    policy = _stage2_policy()
    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )
    ppo_std_before = actor_core.actor_std.detach().clone()
    initial_sac_log_std = policy.bc_dagger_sac_adapter.log_std.detach().clone()
    copied = []

    def copy_ppo_modules(self, state_dict, strict=True):
        copied.append(strict)
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        return []

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_ppo_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )

    failed = policy.load_state_dict(_iql_checkpoint())

    assert failed == []
    assert copied == [True]
    assert policy.qnet.weight.item() == pytest.approx(3.0)
    assert policy.qnet_target.weight.item() == pytest.approx(-4.0)
    assert not policy.qnet_target.weight.requires_grad
    assert torch.equal(actor_core.actor_std, ppo_std_before)
    assert not actor_core.actor_std.requires_grad
    assert torch.equal(
        policy.bc_dagger_sac_adapter.log_std,
        initial_sac_log_std,
    )
    assert policy.bc_dagger_sac_adapter.log_std.item() == pytest.approx(
        math.log(
            policy.cfg.sac_bc_initial_action_std
            / policy.cfg.sac_bc_action_clip
        )
    )
    assert torch.equal(
        policy.bc_dagger_actor_anchor.weight,
        policy.actor_adapt.weight,
    )
    assert not policy.bc_dagger_actor_anchor.weight.requires_grad
    assert not hasattr(policy, "iql_value")
    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0
    assert policy.teacher_replay_id == "dagger-replay"
    assert policy._loaded_teacher_replay_metadata == {
        "snapshot_id": "dagger-snapshot"
    }
    assert policy._bc_dagger_iql_source == {
        "critic_learning_semantics": BC_DAGGER_IQL_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "source_q_updates": 123,
        "source_value_updates": 119,
        "expectile": 0.7,
        "value_network_stage2_usage": "discarded_stage1_bootstrap_only",
    }


def test_stage2_can_transfer_bc_student_but_start_q_and_targets_fresh(
    monkeypatch,
):
    policy = _stage2_policy(load_pretrained_q=False)
    policy.actor_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.encoder_priv = torch.nn.Linear(1, 1, bias=False)
    policy.adapt_module = torch.nn.Linear(1, 1, bias=False)
    policy.adapt_ema = torch.nn.Linear(1, 1, bias=False)
    policy.object_adapt = torch.nn.Linear(1, 1, bias=False)
    policy.object_adapt_ema = torch.nn.Linear(1, 1, bias=False)
    policy.depth_cnn = torch.nn.Linear(1, 1, bias=False)
    policy.temporal_depth_gru = torch.nn.Linear(1, 1, bias=False)
    policy.temporal_depth_gru_ema = torch.nn.Linear(1, 1, bias=False)
    policy.cfg.use_object_adapt = True

    with torch.no_grad():
        for module in (
            policy.actor_adapt,
            policy.encoder_priv,
            policy.adapt_module,
            policy.adapt_ema,
            policy.object_adapt,
            policy.object_adapt_ema,
            policy.depth_cnn,
            policy.temporal_depth_gru,
            policy.temporal_depth_gru_ema,
        ):
            module.weight.zero_()
        policy.qnet.weight.fill_(0.125)
        policy.qnet_target.weight.fill_(-0.75)
    policy.opt_q = torch.optim.AdamW(policy.qnet.parameters(), lr=3e-4)

    # A fresh Stage-2 process must never inherit progress from Stage 1 even if
    # stale values happen to exist before loading.
    policy.q_update_count = 101
    policy.sac_update_count = 102
    policy.sac_actor_update_count = 103
    policy.sac_alpha_update_count = 104
    policy.sac_environment_steps = 105
    policy.sac_rollout_count = 106

    actor_core = SimpleNamespace(
        actor_std=torch.nn.Parameter(torch.tensor([0.25]))
    )
    ppo_std_before = actor_core.actor_std.detach().clone()
    initial_sac_log_std = policy.bc_dagger_sac_adapter.log_std.detach().clone()

    def copy_available_ppo_modules(self, state_dict, strict=True):
        failed = []
        for name, module in self.named_children():
            if name not in state_dict:
                failed.append(name)
                continue
            module.load_state_dict(state_dict[name], strict=strict)
        return failed

    monkeypatch.setattr(PPOVEL, "load_state_dict", copy_available_ppo_modules)
    monkeypatch.setattr(
        FastSACVelFinetune,
        "_ppo_actor_core",
        staticmethod(lambda actor: actor_core),
    )

    state = _iql_checkpoint()
    # Opting out of Q transfer must make the Stage-1-only IQL networks and
    # update provenance irrelevant. This also permits actor-only BC-DAgger
    # checkpoints rather than pretending that an untrained Q is useful.
    for field in (
        "qnet",
        "qnet_target",
        "iql_value",
        "q_update_count",
        "iql_value_update_count",
    ):
        state.pop(field)
    state["optimizers"] = {
        # Deliberately invalid as an Adam state: the Stage-1 optimizer belongs
        # to a different objective and must never be restored in Stage 2.
        "q_optimizer": {"stage1_only": True},
    }
    transferred_modules = (
        "actor_adapt",
        "encoder_priv",
        "adapt_module",
        "adapt_ema",
        "object_adapt",
        "object_adapt_ema",
        "depth_cnn",
        "temporal_depth_gru",
        "temporal_depth_gru_ema",
    )
    for index, name in enumerate(transferred_modules, start=1):
        state[name] = {"weight": torch.tensor([[float(index)]])}

    failed = policy.load_state_dict(state)

    assert "qnet" not in failed
    assert "qnet_target" not in failed
    for index, name in enumerate(transferred_modules, start=1):
        assert getattr(policy, name).weight.item() == pytest.approx(float(index))

    # The online Q remains at its seeded Stage-2 initialization and the target
    # becomes an exact, frozen copy of that fresh online network.
    assert policy.qnet.weight.item() == pytest.approx(0.125)
    assert policy.qnet_target.weight.item() == pytest.approx(0.125)
    assert not policy.qnet_target.weight.requires_grad
    assert torch.equal(actor_core.actor_std, ppo_std_before)
    assert not actor_core.actor_std.requires_grad
    assert torch.equal(
        policy.bc_dagger_sac_adapter.log_std,
        initial_sac_log_std,
    )
    assert torch.equal(
        policy.bc_dagger_actor_anchor.weight,
        policy.actor_adapt.weight,
    )
    assert not policy.bc_dagger_actor_anchor.weight.requires_grad
    assert not hasattr(policy, "iql_value")
    assert policy.opt_q.state == {}
    assert policy.q_update_count == 0
    assert policy.sac_update_count == 0
    assert policy.sac_actor_update_count == 0
    assert policy.sac_alpha_update_count == 0
    assert policy.sac_environment_steps == 0
    assert policy.sac_rollout_count == 0
    assert policy._bc_dagger_iql_source == {
        "critic_learning_semantics": BC_DAGGER_IQL_CRITIC_SEMANTICS,
        "actor_learning_semantics": BC_DAGGER_ACTOR_LEARNING_SEMANTICS,
        "source_q_updates": 0,
        "source_value_updates": 0,
        "expectile": 0.7,
        "value_network_stage2_usage": "not_loaded",
        "q_weights_stage2_usage": "discarded_fresh_q_seed",
        "q_seed": 17,
    }


def test_fresh_q_option_still_requires_pure_bc_actor_provenance(monkeypatch):
    policy = _stage2_policy(load_pretrained_q=False)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("invalid actor provenance reached module copy")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _iql_checkpoint()
    state["actor_learning_semantics"] = "q-weighted-actor"
    for field in (
        "qnet",
        "qnet_target",
        "iql_value",
        "q_update_count",
        "iql_value_update_count",
    ):
        state.pop(field)

    with pytest.raises(ValueError, match="not trained by pure DAgger BC"):
        policy.load_state_dict(state)


def test_stage2_rejects_dagger_action_support_mismatch_before_module_copy(
    monkeypatch,
):
    policy = _stage2_policy()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("action-support mismatch reached module copy")

    monkeypatch.setattr(PPOVEL, "load_state_dict", unexpected_copy)
    state = _iql_checkpoint()
    state["dagger_backend_config"]["dagger_action_clip"] = 10.0

    with pytest.raises(ValueError, match="action clip does not match"):
        policy.load_state_dict(state)


def test_stage2_deterministic_rollout_exactly_matches_dagger_clipping():
    raw_action = torch.tensor(
        [[-25.0, -1.5, 0.25, 30.0, float("nan"), float("inf"), -float("inf")]]
    )
    action_clip = 20.0

    class _RawActionPolicy(torch.nn.Module):
        def forward(self, td):
            td[ACTION_KEY] = raw_action.expand(*td.batch_size, -1)
            return td

    class _MeanDistActor(torch.nn.Module):
        def get_dist(self, td):
            return SimpleNamespace(mean=raw_action.expand(*td.batch_size, -1))

    source_td = TensorDict({}, batch_size=[1])
    _ClippedStudentRolloutPolicy(
        _RawActionPolicy(), action_clip=action_clip
    )(source_td)

    stage2 = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(stage2)
    stage2.cfg = SimpleNamespace(
        sac_bc_action_clip=action_clip,
        sac_bc_initial_action_std=0.05,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
    )
    stage2.actor_adapt = _MeanDistActor()
    stage2.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=raw_action.shape[-1],
        initial_log_std=math.log(
            stage2.cfg.sac_bc_initial_action_std / action_clip
        ),
        device="cpu",
    )
    low = torch.full((raw_action.shape[-1],), -action_clip)
    high = torch.full_like(low, action_clip)
    stage2._fastsac_action_low = low.tolist()
    stage2._fastsac_action_high = high.tolist()
    stage2.dist_cls = functools.partial(
        FastSACTanhNormal,
        low=low,
        high=high,
        event_dims=1,
    )

    stage2_td = TensorDict({}, batch_size=[1])
    _BCDaggerSACRolloutActor(stage2)(stage2_td)

    assert torch.equal(stage2_td[ACTION_KEY], source_td[ACTION_KEY])
    assert torch.equal(
        stage2_td[ACTION_KEY],
        torch.tensor([[-20.0, -1.5, 0.25, 20.0, 0.0, 20.0, -20.0]]),
    )
    assert torch.isfinite(stage2_td["loc"]).all()
    assert torch.isfinite(stage2_td["scale"]).all()
    assert torch.allclose(
        stage2_td["scale"],
        torch.full_like(stage2_td["scale"], 0.05 / action_clip),
    )


def test_stage2_stochastic_rollout_sample_is_exact_online_replay_action():
    mean_action = torch.tensor([[0.2, -0.3]])
    action_clip = 1.0

    class _MeanDistActor(torch.nn.Module):
        def get_dist(self, td):
            return SimpleNamespace(mean=mean_action.expand(*td.batch_size, -1))

    stage2 = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(stage2)
    stage2.cfg = SimpleNamespace(
        train_every=1,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_bc_action_clip=action_clip,
        sac_bc_initial_action_std=0.1,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
    )
    stage2.actor_adapt = _MeanDistActor()
    stage2.bc_dagger_sac_adapter = _BCDaggerSACAdapter(
        action_dim=2,
        initial_log_std=math.log(stage2.cfg.sac_bc_initial_action_std),
        device="cpu",
    )
    low = torch.full((2,), -action_clip)
    high = torch.full_like(low, action_clip)
    stage2._fastsac_action_low = low.tolist()
    stage2._fastsac_action_high = high.tolist()
    stage2.dist_cls = functools.partial(
        FastSACTanhNormal,
        low=low,
        high=high,
        event_dims=1,
    )
    stage2.sac_action_rng = torch.Generator().manual_seed(41)
    stage2.sac_rollout_rng = torch.Generator().manual_seed(42)

    actor_td = TensorDict({}, batch_size=[1])
    _, expected_dist = stage2._bc_dagger_behavior_action_and_dist(actor_td)
    expected_action, _ = expected_dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(42)
    )
    global_rng_before = torch.random.get_rng_state().clone()
    learning_rng_before = stage2.sac_action_rng.get_state().clone()

    _BCDaggerSACRolloutActor(stage2, deterministic=False)(actor_td)

    assert torch.equal(actor_td[ACTION_KEY], expected_action)
    assert not torch.equal(actor_td[ACTION_KEY], mean_action)
    assert ((actor_td[ACTION_KEY] > low) & (actor_td[ACTION_KEY] < high)).all()
    expected_deviation = (expected_action - mean_action).abs()
    assert torch.equal(
        actor_td[STAGE2_BEHAVIOR_MEAN_ABS_DEVIATION_KEY],
        expected_deviation.mean(dim=-1),
    )
    assert torch.equal(
        actor_td[STAGE2_BEHAVIOR_MAX_ABS_DEVIATION_KEY],
        expected_deviation.amax(dim=-1),
    )
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert torch.equal(stage2.sac_action_rng.get_state(), learning_rng_before)

    rollout_rng_before_eval = stage2.sac_rollout_rng.get_state().clone()
    eval_td = TensorDict({}, batch_size=[1])
    _BCDaggerSACRolloutActor(stage2, deterministic=True)(eval_td)
    assert torch.equal(eval_td[ACTION_KEY], mean_action)
    assert torch.equal(
        stage2.sac_rollout_rng.get_state(), rollout_rng_before_eval
    )

    stage2.q_actor_keys = ["actor"]
    stage2.q_critic_keys = ["critic"]
    stage2._q_actor_dim = 1
    stage2._q_critic_dim = 1
    stage2.action_dim = 2
    stage2._rollout_final_batch = {
        "next_observations": torch.tensor([[1.0]]),
        "next_critic_observations": torch.tensor([[2.0]]),
    }
    stage2._truncation_final_batches = []
    stage2._rollout_q_actuator_contexts = []
    rollout = TensorDict(
        {
            "actor": torch.zeros(1, 1, 1),
            "critic": torch.zeros(1, 1, 1),
            ACTION_KEY: actor_td[ACTION_KEY].unsqueeze(1),
            "step_count": torch.full((1, 1, 1), 6),
            "next": TensorDict(
                {
                    "reward": torch.zeros(1, 1, 1),
                    "done": torch.zeros(1, 1, 1, dtype=torch.bool),
                    "terminated": torch.zeros(1, 1, 1, dtype=torch.bool),
                    "discount": torch.ones(1, 1, 1),
                    "stats": TensorDict(
                        {
                            "episode_time_limit": torch.zeros(
                                1, 1, 1, dtype=torch.bool
                            ),
                            "command_finished": torch.zeros(
                                1, 1, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[1, 1],
                    ),
                },
                batch_size=[1, 1],
            ),
        },
        batch_size=[1, 1],
    )

    transition = next(stage2._student_transition_chunks(rollout))
    replay = OnlineReplay(capacity=4, device="cpu")
    replay.extend(transition)

    assert torch.equal(transition["actions"], expected_action)
    assert torch.equal(replay.data["actions"][:1], expected_action)
