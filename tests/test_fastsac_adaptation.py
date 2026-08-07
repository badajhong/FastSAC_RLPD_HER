import functools
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
    FastSACTanhNormal,
    NEXT_TEACHER_REF_ACTION_FIELD,
    PRE_NORMALIZED_REPLAY_KEY,
    STAGE2_OFFLINE_SOURCE_KEY,
    TEACHER_REPLAY_FIELDS,
    TEACHER_REF_ACTION_FIELD,
    TeacherReplayBuffer,
    _validate_fastsac_finetune_config,
)
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import ACTION_KEY, Actor
from active_adaptation.learning.ppo.ppo_vel import REF_JPOS_KEY


def test_student_keeps_ppo_vel_inputs_and_uses_guarded_stage2_defaults():
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
        assert config.q_num_atoms == 501
        assert config.sac_replay_raw_observations is True
        assert config.sac_q_normalize_actions is True
        assert config.sac_q_action_input_gain == 1.0
        assert config.q_action_coordinates == "absolute"
        assert config.q_reference_dueling is False
        assert config.sac_clipped_double_q is True
        assert config.sac_use_autotune is True
    assert fastsac_teacher.sac_updates_per_env_step == 4
    assert fastsac_teacher.sac_policy_frequency == 2
    assert fastsac_teacher.sac_target_entropy_ratio == 0.5
    assert fastsac_teacher.sac_tau == 0.05
    assert fastsac_teacher.sac_learning_starts == 10
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
    # Stage 2 first fills replay from accepted transitions, then trains Q alone
    # through update 8000. The first confidence-gated actor candidate is on the
    # next 128-update cadence boundary, Q update 8064.
    assert fastsac.sac_learning_starts == 98_304
    assert fastsac.sac_batch_size == 512
    assert fastsac.sac_updates_per_env_step == 1
    assert fastsac.q_lr == 3e-5
    assert fastsac.sac_policy_frequency == 128
    assert fastsac.sac_actor_learning_starts_q_updates == 8_000
    assert fastsac.sac_actor_confidence_gate is True
    assert fastsac.sac_actor_gate_disagreement_multiplier == 1.0
    assert fastsac.sac_actor_gate_min_accept_fraction == 0.10
    assert fastsac.sac_actor_gate_absolute_margin == 0.0
    assert fastsac.sac_actor_lr == 3e-7
    assert fastsac.sac_alpha_lr == 2e-5
    assert fastsac.sac_alpha_init == 1e-5
    assert fastsac.sac_alpha_ramp_q_updates == 20_000
    assert fastsac.sac_target_entropy_ratio == 1.0
    assert fastsac.sac_tau == 0.001
    assert fastsac.sac_max_grad_norm == 1.0
    assert fastsac.sac_deterministic_rollout is False
    assert fastsac.sac_freeze_perception is True
    assert fastsac.sac_bc_action_clip == 20.0
    assert fastsac.sac_entropy_reference_scale == 1.0
    assert fastsac.sac_bc_initial_action_std == 0.01
    assert fastsac.sac_bc_log_std_min == -8.0
    assert fastsac.sac_bc_log_std_max == -2.0
    assert fastsac.sac_bc_anchor_coef_start == 0.0
    assert fastsac.sac_bc_anchor_coef_end == 0.0
    assert fastsac.sac_bc_anchor_decay_q_updates == 100_000
    assert fastsac.sac_bc_anchor_huber_delta == 0.1
    assert fastsac.load_pretrained_q is True
    assert fastsac.finetune_checkpoint_source == "auto"
    assert fastsac.teacher_buffer_capacity == 1_048_576
    with pytest.raises(ConfigLoadError):
        store.load("algo/ppo_fastsac_vel_train.yaml")


class _ReplayRecorder:
    def __init__(self, capacity=None):
        self.extended = None
        self.capacity = capacity
        self.size = 0
        self.seen = 0

    def extend(self, transitions):
        self.extended = transitions
        count = next(iter(transitions.values())).shape[0]
        self.size += count
        if self.capacity is not None:
            self.size = min(self.size, int(self.capacity))
        self.seen += count


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


def test_stage2_train_op_uses_seen_gate_and_keeps_perception_frozen():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_updates_per_env_step=2,
        sac_policy_frequency=32,
        sac_actor_learning_starts_q_updates=8_000,
        teacher_buffer_ratio=0.5,
        # Each chunk contributes two accepted rows. Only the third chunk reaches
        # this gate, so exactly one two-update Q burst is run.
        sac_learning_starts=5,
        sac_batch_size=2,
        train_every=3,
        sac_max_grad_norm=0.0,
    )
    policy.sac_update_count = 7_998
    policy.q_update_count = 7_998
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
        policy.q_update_count += 1
        if update_actor:
            policy.sac_actor_update_count += 1
            policy.sac_alpha_update_count += 1
        return (
            torch.tensor(1.0),
            torch.tensor([2.0, 3.0]),
            torch.tensor(4.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    policy._sac_update = sac_update

    policy.train_adapt = lambda td: (_ for _ in ()).throw(
        AssertionError("Stage-2 perception must stay frozen")
    )
    rollout = TensorDict(
        {
            "marker": torch.zeros(2, 3, 1),
            "stats": torch.zeros(2, 3, 1),
        },
        batch_size=[2, 3],
    )

    info = policy.train_op(rollout)

    assert events == ["transitions", "sac:False", "sac:False"]
    assert policy.sac_update_count == 8_000
    assert policy.q_update_count == 8_000
    assert policy.sac_actor_update_count == 0
    assert policy.sac_alpha_update_count == 0
    assert policy.num_updates == 1
    assert info["fastsac/perception_frozen"] == 1.0
    assert info["fastsac/q_only_warmup"] == 1.0
    assert info["fastsac/actor_active"] == 0.0
    assert info["fastsac/online_replay_seen"] == 6
    assert info["fastsac/replay_warmup_ready"] == 1.0
    assert info["fastsac/truncation_finals"] == 3
    assert info["fastsac/actor_loss"] == 0.0
    assert info["fastsac/alpha_active"] == 0.0
    assert info["fastsac/new_online_rows"] == 6
    assert info["fastsac/sampled_total_draws"] == 4
    assert info["fastsac/sampled_online_draws"] == 2
    assert info["fastsac/sampled_offline_draws"] == 2
    assert info["fastsac/sampled_draws_per_new_row_valid"] == 1.0
    assert info["fastsac/sampled_total_draws_per_new_row"] == pytest.approx(
        4.0 / 6.0
    )
    assert info["fastsac/sampled_online_draws_per_new_row"] == pytest.approx(
        2.0 / 6.0
    )
    assert info["fastsac/sampled_offline_draws_per_new_row"] == pytest.approx(
        2.0 / 6.0
    )
    assert set(policy.online_replay.extended) == set(TeacherReplayBuffer.fields)


def test_stage2_actor_first_eligible_cadence_is_q_update_8064():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_actor_learning_starts_q_updates=8_000,
        sac_policy_frequency=128,
    )

    assert policy._stage2_actor_is_active(8_000) is False
    assert policy._stage2_actor_is_active(8_001) is True

    policy.q_update_count = 7_999
    assert policy._sac_actor_update_is_due(0, 0, 1) is False
    policy.q_update_count = 8_000
    assert policy._sac_actor_update_is_due(0, 0, 1) is False
    policy.q_update_count = 8_062
    assert policy._sac_actor_update_is_due(0, 0, 1) is False
    policy.q_update_count = 8_063
    assert policy._sac_actor_update_is_due(0, 0, 1) is True


def _stage2_confidence_gate_policy(frozen_bc_q, min_accept_fraction=0.10):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        sac_actor_confidence_gate=True,
        sac_actor_gate_disagreement_multiplier=1.0,
        sac_actor_gate_min_accept_fraction=min_accept_fraction,
        sac_actor_gate_absolute_margin=0.0,
    )
    policy.bc_dagger_actor_anchor = torch.nn.Identity()
    policy.qnet = SimpleNamespace(values=lambda values: values)
    policy._actor_td_from_flat = lambda observations: observations
    policy._bc_dagger_behavior_action = (
        lambda td, actor=None: torch.zeros(td.shape[0], 1)
    )
    policy._q_forward = lambda *args, **kwargs: frozen_bc_q
    return policy


def test_stage2_confidence_gate_uses_raw_twin_confidence_and_coverage():
    # One of ten rows has a pessimistic +.28 gain and only .02 raw twin
    # disagreement, so the exact 10% coverage boundary passes.
    frozen_bc_q = torch.zeros(2, 10)
    sampled_q = frozen_bc_q - 0.1
    sampled_q[:, 0] = frozen_bc_q[:, 0] + torch.tensor([0.30, 0.28])
    policy = _stage2_confidence_gate_policy(frozen_bc_q)
    batch = {
        "observations": torch.zeros(10, 1),
        "critic_observations": torch.zeros(10, 1),
    }

    passed, row_mask, diagnostics = policy._stage2_actor_confidence_gate(
        batch, torch.ones(10, 1), sampled_q
    )

    assert passed is True
    assert torch.equal(
        row_mask, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert diagnostics["acceptance_fraction"].item() == pytest.approx(0.10)
    assert diagnostics["twin_disagreement"].item() < 0.01
    assert diagnostics["passed"].item() == 1.0

    # The gate is recomputed on every tick; changing only the configured batch
    # coverage makes this same candidate fail instead of permanently unlocking.
    policy.cfg.sac_actor_gate_min_accept_fraction = 0.11
    passed, _, diagnostics = policy._stage2_actor_confidence_gate(
        batch, torch.ones(10, 1), sampled_q
    )
    assert passed is False
    assert diagnostics["skipped"].item() == 1.0

    # The same action-effect ranking is not trusted when the two critics have
    # a large raw value gap. This is the failure signal observed in the run.
    offset_bc_q = torch.stack((
        torch.full((10,), 10.0),
        torch.zeros(10),
    ))
    offset_sampled_q = offset_bc_q - 0.1
    offset_sampled_q[:, 0] = (
        offset_bc_q[:, 0] + torch.tensor([0.30, 0.28])
    )
    offset_policy = _stage2_confidence_gate_policy(offset_bc_q)
    passed, row_mask, diagnostics = (
        offset_policy._stage2_actor_confidence_gate(
            batch, torch.ones(10, 1), offset_sampled_q
        )
    )
    assert passed is False
    assert row_mask.count_nonzero().item() == 0
    assert diagnostics["twin_disagreement"].item() > 9.0


def test_stage2_confidence_gate_rejects_when_one_q_head_disagrees():
    frozen_bc_q = torch.zeros(2, 4)
    sampled_q = torch.tensor([
        [0.30, 0.30, 0.30, 0.30],
        [-0.01, -0.01, -0.01, -0.01],
    ])
    policy = _stage2_confidence_gate_policy(
        frozen_bc_q, min_accept_fraction=0.10
    )
    batch = {
        "observations": torch.zeros(4, 1),
        "critic_observations": torch.zeros(4, 1),
    }

    passed, row_mask, diagnostics = policy._stage2_actor_confidence_gate(
        batch, torch.ones(4, 1), sampled_q
    )

    assert passed is False
    assert row_mask.count_nonzero().item() == 0
    assert diagnostics["clipped_q_gain"].item() < 0.0
    assert diagnostics["confidence_margin"].item() < 0.0


def test_stage2_effective_alpha_starts_at_actual_release_and_ramps_linearly():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_alpha_ramp_q_updates=20_000)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.q_update_count = 20_000
    policy._stage2_actor_release_q_update = None

    # Passing the nominal burn-in by itself does not introduce entropy.
    assert policy._stage2_alpha_ramp_progress() == 0.0
    assert policy._stage2_effective_alpha().item() == 0.0

    assert policy._mark_stage2_actor_released(20_000) == 20_000
    assert policy._stage2_alpha_ramp_progress(20_000) == 0.0
    assert policy._stage2_effective_alpha(20_000).item() == 0.0
    assert policy._stage2_alpha_ramp_progress(30_000) == pytest.approx(0.5)
    assert policy._stage2_effective_alpha(30_000).item() == pytest.approx(0.25)
    assert policy._stage2_alpha_ramp_progress(40_000) == 1.0
    assert policy._stage2_effective_alpha(50_000).item() == pytest.approx(0.5)

    # Later accepted actor ticks keep the original ramp origin.
    assert policy._mark_stage2_actor_released(40_000) == 20_000
    with pytest.raises(RuntimeError, match="cannot move backward"):
        policy._mark_stage2_actor_released(19_999)


def test_stage2_entropy_reference_scale_is_independent_of_safety_clip():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    physical_log_prob = torch.tensor([3.0])
    policy._fastsac_action_log_scale_sum = 2.0 * torch.log(
        torch.tensor(20.0)
    ).item()
    policy._fastsac_entropy_reference_log_scale_sum = 0.0

    # Raw-unit scale 1.0 adds no density constant even though execution is
    # guarded by a much wider +/-20 support.
    assert torch.equal(
        policy._normalized_action_log_prob(physical_log_prob),
        physical_log_prob,
    )

    del policy._fastsac_entropy_reference_log_scale_sum
    assert torch.allclose(
        policy._normalized_action_log_prob(physical_log_prob),
        physical_log_prob + policy._fastsac_action_log_scale_sum,
    )


def test_stage2_stochastic_behavior_samples_q_target_during_actor_freeze():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        sac_deterministic_rollout=False,
        sac_actor_learning_starts_q_updates=8_000,
    )

    assert policy._stage2_actor_is_active(1) is False
    assert policy._stage2_q_target_uses_stochastic_policy(1) is True

    policy.cfg.sac_deterministic_rollout = True
    assert policy._stage2_q_target_uses_stochastic_policy(1) is False
    assert policy._stage2_q_target_uses_stochastic_policy(8_001) is True


def test_stage2_freezes_perception_modules_and_removes_their_optimizers():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    perception_names = (
        "encoder_priv",
        "adapt_module",
        "adapt_ema",
        "object_adapt",
        "object_adapt_ema",
        "depth_cnn",
        "temporal_depth_gru",
        "temporal_depth_gru_ema",
        "dr_estimator",
    )
    for name in perception_names:
        setattr(policy, name, torch.nn.Linear(2, 2))
    policy.actor_adapt = torch.nn.Linear(2, 1)
    policy.opt_adapt = torch.optim.Adam(
        policy.adapt_module.parameters(), lr=1e-3
    )
    policy.opt_dr_estimator = torch.optim.Adam(
        policy.dr_estimator.parameters(), lr=1e-3
    )

    policy._freeze_stage2_perception()

    for name in perception_names:
        assert all(
            parameter.requires_grad is False
            for parameter in getattr(policy, name).parameters()
        )
    assert all(
        parameter.requires_grad for parameter in policy.actor_adapt.parameters()
    )
    assert policy.opt_adapt is None
    assert policy.opt_dr_estimator is None


def test_student_resume_waits_for_accepted_transition_seen_gate():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_updates_per_env_step=1,
        sac_policy_frequency=32,
        sac_actor_learning_starts_q_updates=8_000,
        teacher_buffer_ratio=0.5,
        sac_learning_starts=2,
        sac_batch_size=2,
        train_every=3,
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
    policy.online_replay = _ReplayRecorder(capacity=1)
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
    # The one-row FIFO wraps, but accepted-transition progress remains monotonic.
    policy._student_transition_chunks = lambda td: iter((empty, valid, valid))
    policy._mix_batch = lambda: events.append("mix") or {"batch": torch.ones(1)}

    def sac_update(batch, update_actor):
        assert update_actor is False
        policy.q_update_count += 1
        return (
            torch.tensor(1.0),
            torch.tensor([2.0, 3.0]),
            torch.tensor(4.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    policy._sac_update = sac_update
    rollout = TensorDict(
        {
            "marker": torch.zeros(1, 3, 1),
            "stats": torch.zeros(1, 3, 1),
        },
        batch_size=[1, 3],
    )

    policy.train_op(rollout)

    assert events == ["mix"]
    assert policy.online_replay.size == 1
    assert policy.online_replay.seen == 2
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

    assert set(mixed) == set(TEACHER_REPLAY_FIELDS) | {
        STAGE2_OFFLINE_SOURCE_KEY
    }
    assert all(value.shape[0] == 4 for value in mixed.values())
    assert mixed[STAGE2_OFFLINE_SOURCE_KEY].all()


def test_stage2_replay_mix_counts_use_one_exact_rounding_formula():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        sac_batch_size=5, teacher_buffer_ratio=0.5
    )

    # Python round(2.5) is two; mix and telemetry both consume this one helper.
    assert policy._stage2_replay_mix_counts() == {
        "total": 5,
        "online": 3,
        "offline": 2,
    }


def test_rlpd_marks_only_legacy_bc_dagger_rows_as_pre_normalized():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(sac_batch_size=4, teacher_buffer_ratio=0.5)
    policy.device = torch.device("cpu")
    policy.q_rng = torch.Generator().manual_seed(0)

    class _Replay:
        size = 8

        def __init__(self, pre_normalized):
            self.observations_pre_normalized = pre_normalized

        def sample(self, count, device=None, generator=None):
            return {
                key: value
                for key, value in _fake_transitions(count=count).items()
                if key in TEACHER_REPLAY_FIELDS
            }

    policy.online_replay = _Replay(False)
    policy.offline_replay = _Replay(True)
    mixed = policy._mix_batch()

    marker = mixed[PRE_NORMALIZED_REPLAY_KEY]
    assert marker.dtype is torch.bool
    assert int(marker.sum()) == 2
    assert marker.numel() == 4
    source = mixed[STAGE2_OFFLINE_SOURCE_KEY]
    assert source.dtype is torch.bool
    assert int(source.sum()) == 2
    # VecNorm provenance and replay provenance are independently constructed,
    # even though this legacy test happens to align them.
    assert source.data_ptr() != marker.data_ptr()


def test_stage2_normalizes_online_rows_but_preserves_bc_dagger_rows():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    object.__setattr__(policy, "_replay_vecnorm", SimpleNamespace())
    policy._vecnorm_snapshot = lambda: object()
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_widths = [2]
    policy._q_critic_widths = [3]
    policy._normalize_replay_flat = (
        lambda values, keys, widths, snapshot: values + 10.0
    )
    batch = {
        "observations": torch.zeros(2, 2),
        "next_observations": torch.ones(2, 2),
        "critic_observations": torch.zeros(2, 3),
        "next_critic_observations": torch.ones(2, 3),
        PRE_NORMALIZED_REPLAY_KEY: torch.tensor([False, True]),
    }

    prepared = policy._prepare_student_learning_batch(batch)

    assert torch.equal(prepared["observations"][0], torch.full((2,), 10.0))
    assert torch.equal(prepared["observations"][1], torch.zeros(2))
    assert PRE_NORMALIZED_REPLAY_KEY not in prepared


def test_stage2_normalizes_new_raw_dagger_and_online_rows_together():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    object.__setattr__(policy, "_replay_vecnorm", SimpleNamespace())
    policy._vecnorm_snapshot = lambda: object()
    policy.q_actor_keys = ["actor"]
    policy.q_critic_keys = ["critic"]
    policy._q_actor_widths = [2]
    policy._q_critic_widths = [3]
    policy._normalize_replay_flat = (
        lambda values, keys, widths, snapshot: values + 10.0
    )
    # No PRE_NORMALIZED_REPLAY_KEY: both new DAgger and online rows are raw.
    batch = {
        "observations": torch.zeros(2, 2),
        "next_observations": torch.ones(2, 2),
        "critic_observations": torch.zeros(2, 3),
        "next_critic_observations": torch.ones(2, 3),
    }

    prepared = policy._prepare_student_learning_batch(batch)

    assert torch.equal(prepared["observations"], torch.full((2, 2), 10.0))
    assert torch.equal(
        prepared["critic_observations"], torch.full((2, 3), 10.0)
    )


def test_bc_dagger_actor_adapter_preserves_mean_and_has_sac_gradients():
    class _BCActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Actor(2, init_noise_scale=0.5)
            self.core(torch.zeros(1, 3))
            with torch.no_grad():
                self.core.actor_mean.weight.zero_()
                self.core.actor_mean.bias.copy_(torch.tensor([0.2, -0.3]))
                # Pure DAgger never sampled or optimized this PPO-only state.
                self.core.actor_std.fill_(0.5)
            self.core.actor_std.requires_grad_(False)

        def get_dist(self, td):
            loc, scale = self.core(torch.zeros(*td.batch_size, 3))
            return IndependentNormal(loc, scale)

    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.actor_adapt = _BCActor()
    policy.cfg = SimpleNamespace(
        sac_bc_action_clip=1.0,
        sac_bc_log_std_min=-8.0,
        sac_bc_log_std_max=-2.0,
    )
    policy.bc_dagger_sac_adapter = torch.nn.Module()
    policy.bc_dagger_sac_adapter.log_std = torch.nn.Parameter(
        torch.full((2,), float(torch.tensor(0.05).log()))
    )
    low = torch.tensor([-1.0, -1.0])
    high = torch.tensor([1.0, 1.0])
    policy._fastsac_action_low = low.tolist()
    policy._fastsac_action_high = high.tolist()
    policy.dist_cls = functools.partial(
        FastSACTanhNormal, low=low, high=high, event_dims=1
    )
    td = TensorDict({}, batch_size=[4])

    dist = policy._bc_dagger_actor_dist_from_td(td)
    action, log_prob = dist.rsample_with_log_prob(
        generator=torch.Generator().manual_seed(0)
    )

    assert torch.allclose(
        dist.mean, torch.tensor([0.2, -0.3]).expand(4, -1), atol=1e-6
    )
    assert ((action > low) & (action < high)).all()
    (-log_prob.mean()).backward()
    assert policy.actor_adapt.core.actor_mean.weight.grad is not None
    assert policy.bc_dagger_sac_adapter.log_std.grad is not None
    assert policy.actor_adapt.core.actor_std.grad is None
    assert torch.equal(
        policy.actor_adapt.core.actor_std.detach(), torch.full((2,), 0.5)
    )


def test_fastsac_rejects_second_actor_optimizer_from_distillation():
    assert FastSACVelFinetuneConfig().enable_residual_distillation is False
    cfg = SimpleNamespace(enable_residual_distillation=True)
    with pytest.raises(ValueError, match="requires enable_residual_distillation=false"):
        FastSACVelFinetune(cfg, None, None, None, "cpu", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load_pretrained_q", "true"),
        ("sac_learning_starts", -1),
        ("sac_actor_learning_starts_q_updates", -1),
        ("sac_alpha_ramp_q_updates", 0),
        ("sac_actor_confidence_gate", "true"),
        ("sac_actor_gate_min_accept_fraction", 0.0),
        ("sac_actor_gate_disagreement_multiplier", -1.0),
        ("sac_actor_gate_min_accept_fraction", 1.1),
        ("sac_actor_gate_absolute_margin", -1.0),
        ("sac_bc_anchor_decay_q_updates", 0),
        ("sac_deterministic_rollout", "true"),
        ("sac_freeze_perception", False),
        ("sac_bc_action_clip", 0.0),
        ("sac_entropy_reference_scale", 0.0),
        ("sac_bc_initial_action_std", 0.0),
        ("sac_bc_log_std_min", -1.0),
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
        ("vecnorm", "train"),
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
        self.projection_reward = None

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
        self.projection_reward = reward.clone()
        self.projection_reference = (
            None if reference_actions is None else reference_actions.clone()
        )
        return torch.full(
            (2, observations.shape[0], 2),
            0.5,
            device=observations.device,
        )


class _Stage2DisagreeingQ(torch.nn.Module):
    """Q1 likes positive actions while Q2 predicts the same action is worse."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("support", torch.tensor([-1.0, 1.0]))
        self.register_buffer("slopes", torch.tensor([1.0, -1.0]))
        self.forward_actions = []

    def forward(self, observations, actions, reference_actions=None):
        self.forward_actions.append(actions.detach().clone())
        scores = (
            self.slopes[:, None] * actions[:, 0][None, :]
            + self.dummy * 0.0
        )
        return torch.stack((-scores, scores), dim=-1)

    def values(self, logits):
        return (torch.softmax(logits, dim=-1) * self.support).sum(-1)


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


def test_stage2_q_only_hard_target_matches_pretrained_iql_semantics():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        sac_deterministic_rollout=False,
        sac_actor_learning_starts_q_updates=8_000,
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_q_normalize_actions=False,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.actor_adapt = _Stage2Actor()
    policy._actor_dist_from_flat = lambda obs: policy.actor_adapt.get_dist(
        TensorDict({}, batch_size=obs.shape[:-1])
    )
    policy.qnet = _Stage2QSpy()
    policy.qnet_target = _Stage2QSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._fastsac_action_log_scale_sum = 0.0
    policy._prepare_student_learning_batch = lambda batch: batch
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.tensor([2.0, -4.0]),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
        STAGE2_OFFLINE_SOURCE_KEY: torch.tensor([False, True]),
    }

    result = policy._sac_update(batch, update_actor=False)

    assert torch.equal(
        policy.qnet_target.projection_action, torch.ones(2, 1)
    )
    # The stochastic next action remains in the target, but the pretrained IQL
    # critic sees its original hard reward until an actor update starts SAC.
    assert torch.equal(
        policy.qnet_target.projection_reward, torch.tensor([2.0, -4.0])
    )
    diagnostics = result[10]
    assert diagnostics["q_target_log_pi_mean"].item() == pytest.approx(-2.0)
    assert diagnostics["effective_alpha"].item() == 0.0
    assert diagnostics["entropy_tax_abs_mean"].item() == 0.0
    assert diagnostics["entropy_tax_reward_abs_ratio"].item() == 0.0
    assert diagnostics["reward_mean"].item() == pytest.approx(-1.0)
    assert diagnostics["source_marker_valid"].item() == 1.0
    assert diagnostics["online_rows"].item() == 1.0
    assert diagnostics["offline_rows"].item() == 1.0
    assert diagnostics["online_ratio_valid"].item() == 1.0
    assert diagnostics["offline_ratio_valid"].item() == 1.0
    assert diagnostics["online_entropy_tax_reward_abs_ratio"].item() == 0.0
    assert diagnostics["offline_entropy_tax_reward_abs_ratio"].item() == 0.0
    q_diagnostics = result[12]
    assert q_diagnostics["source_marker_valid"].item() == 1.0
    assert q_diagnostics["online_data_rows"].item() == 1.0
    assert q_diagnostics["offline_data_rows"].item() == 1.0
    assert policy.q_update_count == 1
    assert policy.sac_actor_update_count == 0
    assert policy.sac_alpha_update_count == 0


def test_stage2_confidence_gate_failure_updates_q_but_not_actor_or_alpha():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="bc_dagger",
        sac_deterministic_rollout=False,
        sac_actor_learning_starts_q_updates=0,
        sac_actor_confidence_gate=True,
        sac_actor_gate_disagreement_multiplier=1.0,
        sac_actor_gate_min_accept_fraction=0.10,
        sac_actor_gate_absolute_margin=0.0,
        sac_alpha_ramp_q_updates=20_000,
        sac_use_autotune=True,
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_q_normalize_actions=False,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
    )
    policy.sac_action_rng = torch.Generator().manual_seed(1)
    policy.actor_adapt = _Stage2Actor()
    policy._actor_dist_from_flat = lambda obs: policy.actor_adapt.get_dist(
        TensorDict({}, batch_size=obs.shape[:-1])
    )
    policy.bc_dagger_actor_anchor = torch.nn.Identity()
    policy._actor_td_from_flat = lambda observations: observations
    policy._bc_dagger_behavior_action = (
        lambda td, actor=None: torch.zeros(td.shape[0], 1)
    )
    policy.qnet = _Stage2DisagreeingQ()
    policy.qnet_target = _Stage2QSpy()
    policy.opt_q = torch.optim.SGD(policy.qnet.parameters(), lr=0.0)
    policy.sac_actor_optimizer = torch.optim.SGD(
        policy.actor_adapt.parameters(), lr=1.0
    )
    policy.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.5)))
    policy.alpha_optimizer = torch.optim.SGD([policy.log_alpha], lr=1.0)
    policy.target_entropy = 0.0
    policy.q_update_count = 0
    policy._stage2_actor_release_q_update = None
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    policy._stage2_bc_anchor_loss = lambda observations: (_ for _ in ()).throw(
        AssertionError("zero-coefficient no-anchor SAC must not evaluate BC loss")
    )
    policy._stage2_bc_anchor_coefficient = lambda q_updates=None: 0.0
    batch = {
        "observations": torch.zeros(4, 1),
        "next_observations": torch.zeros(4, 1),
        "critic_observations": torch.zeros(4, 1),
        "next_critic_observations": torch.zeros(4, 1),
        "actions": torch.zeros(4, 1),
        "rewards": torch.zeros(4),
        "dones": torch.zeros(4, dtype=torch.bool),
        "truncations": torch.zeros(4, dtype=torch.bool),
        "discounts": torch.ones(4),
    }
    actor_before = policy.actor_adapt.action.detach().clone()
    alpha_before = policy.log_alpha.detach().clone()

    result = policy._sac_update(batch, update_actor=True)

    assert policy.q_update_count == 1
    assert policy.sac_actor_update_count == 0
    assert policy.sac_alpha_update_count == 0
    assert policy._stage2_actor_release_q_update is None
    assert torch.equal(policy.actor_adapt.action, actor_before)
    assert torch.equal(policy.log_alpha, alpha_before)
    gate = result[11]
    assert gate["attempted"].item() == 1.0
    assert gate["passed"].item() == 0.0
    assert gate["skipped"].item() == 1.0
    assert gate["actor_update_applied"].item() == 0.0
    assert gate["row_count"].item() == 4.0
    assert result[12]["data_valid"].item() == 1.0
    assert result[12]["data_rows"].item() == 4.0
    assert result[12]["source_marker_valid"].item() == 0.0
    # Replay-Q, the exact sampled actor candidate, then frozen BC baseline.
    assert torch.equal(policy.qnet.forward_actions[0], torch.zeros(4, 1))
    assert torch.equal(policy.qnet.forward_actions[1], torch.ones(4, 1))
    assert torch.equal(policy.qnet.forward_actions[2], torch.zeros(4, 1))

    # When both heads agree on the same sampled candidate, the next scheduled
    # tick applies exactly one actor step and records its numbered Q update.
    policy.qnet.slopes.fill_(1.0)
    result = policy._sac_update(batch, update_actor=True)
    assert result[11]["actor_update_applied"].item() == 1.0
    assert policy.q_update_count == 2
    assert policy.sac_actor_update_count == 1
    assert policy.sac_alpha_update_count == 0
    assert policy._stage2_actor_release_q_update == 2
    actor_after_pass = policy.actor_adapt.action.detach().clone()

    # The release marker is telemetry/ramp origin, not a sticky bypass: a
    # later disagreement skips the complete actor+alpha tick again.
    policy.qnet.slopes.copy_(torch.tensor([1.0, -1.0]))
    result = policy._sac_update(batch, update_actor=True)
    assert result[11]["skipped"].item() == 1.0
    assert result[11]["actor_update_applied"].item() == 0.0
    assert policy.q_update_count == 3
    assert policy.sac_actor_update_count == 1
    assert policy.sac_alpha_update_count == 0
    assert policy._stage2_actor_release_q_update == 2
    assert torch.equal(policy.actor_adapt.action, actor_after_pass)


def test_stage2_q_backup_actor_and_dual_share_ramped_effective_alpha():
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.cfg = SimpleNamespace(
        phase="finetune",
        finetune_checkpoint_source="fastsac",
        sac_actor_learning_starts_q_updates=0,
        sac_alpha_ramp_q_updates=20_000,
        sac_use_autotune=True,
        gamma=0.99,
        sac_max_grad_norm=0.0,
        sac_tau=0.05,
        q_action_coordinates="absolute",
        q_reference_dueling=False,
        q_condition_on_actuator_state=False,
        sac_q_normalize_actions=False,
        sac_q_action_input_gain=1.0,
        sac_clipped_double_q=True,
    )
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
    policy.q_update_count = 10_000
    policy._stage2_actor_release_q_update = 1
    policy.sac_actor_update_count = 0
    policy.sac_alpha_update_count = 0
    policy._prepare_student_learning_batch = lambda batch: batch
    batch = {
        "observations": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 1),
        "critic_observations": torch.zeros(2, 1),
        "next_critic_observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, 1),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2, dtype=torch.bool),
        "truncations": torch.zeros(2, dtype=torch.bool),
        "discounts": torch.ones(2),
    }

    result = policy._sac_update(batch, update_actor=True)

    # prospective Q update 10001 is exactly 10000 updates after release 1.
    assert result[10]["alpha_ramp_progress"].item() == pytest.approx(0.5)
    assert result[10]["effective_alpha"].item() == pytest.approx(0.25)
    # Q target: 0 - .99 * .25 * log_pi(-2) = .495.
    assert torch.allclose(
        policy.qnet_target.projection_reward, torch.full((2,), 0.495)
    )
    # Actor uses the identical .25 coefficient: .25 * -2 - Q(=0).
    assert result[3].item() == pytest.approx(-0.5)
    # The dual also uses that same effective coefficient.
    assert result[4].item() == pytest.approx(0.5)
    assert policy.sac_actor_update_count == 1
    assert policy.sac_alpha_update_count == 1


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
    assert policy.sac_actor_update_count == 1
    # The first real actor step defines ramp progress zero and therefore does
    # not perform a meaningless zero-gradient alpha optimizer step.
    assert policy.sac_alpha_update_count == 0

    policy._sac_update(batch, update_actor=False)

    # Temperature is tied to actor cadence, not the more frequent Q cadence.
    assert policy.sac_actor_update_count == 1
    assert policy.sac_alpha_update_count == 0


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
    policy.sac_rollout_rng = torch.Generator().manual_seed(101)
    torch.randint(10, (3,), generator=policy.q_rng)
    torch.randn(3, generator=policy.sac_action_rng)
    torch.randn(3, generator=policy.sac_rollout_rng)

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
    assert torch.equal(
        policy.sac_rollout_rng.get_state(),
        torch.Generator().manual_seed(9).get_state(),
    )


def test_student_resume_does_not_require_previous_offline_replay_manifest(
    monkeypatch,
):
    policy = FastSACVelFinetune.__new__(FastSACVelFinetune)
    torch.nn.Module.__init__(policy)
    policy.cfg = FastSACVelFinetuneConfig()
    policy.cfg.use_object_adapt = False
    policy.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
    policy.sac_update_count = 19

    def load_student(self, state_dict, strict=True):
        self.q_update_count = 77
        self.sac_actor_update_count = 1
        return []

    monkeypatch.setattr(FastSACVEL, "load_state_dict", load_student)
    policy.load_state_dict({
        "qnet": {},
        "last_phase": "finetune",
        "stage2_schedule_config": policy._stage2_schedule_config(),
        "stage2_actor_release_q_update": 12,
    })

    assert policy.q_update_count == 77
    assert policy.sac_update_count == 19
    assert policy._stage2_actor_release_q_update == 12


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

    assert policy._stage2_replay_mix_counts() == {
        "total": 8192,
        "online": 4096,
        "offline": 4096,
    }
    mixed = policy._mix_batch()
    assert policy.online_replay.calls == [4096]
    assert policy.offline_replay.calls == [4096]
    assert mixed["observations"].shape[0] == 8192
    assert mixed["observations"][:, 0].eq(0.0).sum().item() == 4096
    assert mixed["observations"][:, 0].eq(1.0).sum().item() == 4096
    assert torch.equal(
        mixed[STAGE2_OFFLINE_SOURCE_KEY],
        mixed["observations"][:, 0].eq(0.0),
    )
    for key in TEACHER_REPLAY_FIELDS:
        assert mixed[key].shape[0] == 8192
