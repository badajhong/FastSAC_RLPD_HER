# VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands

<div align="center">
<a href="https://vaic-humanoid.github.io/">
	<img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://arxiv.org/abs/2606.09286">
	<img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/ldt29/VAIC/stargazers">
	<img alt="GitHub stars" src="https://img.shields.io/github/stars/ldt29/VAIC?style=social"/>
</a>

</div>

This repository hosts the open-source release for the paper VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands.


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n vaic python=3.11 -y
conda activate vaic

# install isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# install isaaclab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none

# install vaic
cd ..
git clone https://github.com/ldt29/VAIC
cd VAIC
pip install -e .
```

## Verify Your Data
Visualize motions in Isaac Sim with `task.command.replay_motion=true`:

```bash
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea task.command.replay_motion=true
```


## Train and Evaluate

Teacher policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_train task=G1/vaic/skateboard_tea

# True HOI-style FastSAC teacher: stochastic actor, twin distributional Q,
# entropy-temperature update, target Q, and replay actor updates. As in G1 WBT,
# Stage 1 runs four Q/alpha optimizer updates per vector control step, independent
# of task.num_envs. Q training starts after 98,304 accepted replay rows. The actor
# and alpha stay frozen through 8,000 Q updates; the actor then updates every 32 Q
# updates while alpha uses its teacher-specific 2e-5 learning rate.
# VAIC observations/rewards/terminations and teacher->student distillation stay unchanged.
# The teacher mean is centered on VAIC's framewise reference before tanh; the
# student remains a deployable absolute-action actor centered on raw action zero.
# Both share asymmetric action bounds that remain inside the physical joint
# limits over the configured random_joint_offset range.
# Actor, environment, and replay actions stay in those executable coordinates;
# Q1/Q2 alone receive their affine mapping to [-1, 1] (default
# algo.sac_q_normalize_actions=true). This Q-input semantic is checkpointed.
# algo.sac_q_action_input_gain=1.0 preserves that mapping exactly. A different
# positive fixed gain multiplies only the Q input after the optional affine map
# in both stages; it is checkpointed and never changes actor/env/replay actions.
# algo.q_action_coordinates=absolute preserves that historical Q input. The
# opt-in reference_residual backend instead gives Q1/Q2
# (action-frame_reference)/executable_half_range without clamping. Current and
# next references are kept distinct through n-step/timeout replay. Actor,
# environment, reward, termination, and stored actions remain unchanged. Use
# the same coordinate backend for Stage 1, the Stage-2 checkpoint, and its H5.
# algo.q_action_fusion=early is the exact historical concat-before-first-linear
# Q architecture. The opt-in `late` backend independently maps critic
# observations 2341->768 and actions 23->128 (one sixth of q_hidden_dim), then
# concatenates them into the unchanged 384->192->C51 trunk. Teacher/student
# checkpoints and teacher H5 metadata require the same fusion backend.
# Stage 1 uses a compact learning FIFO. It defaults to the policy device
# (the historical GPU-local behavior), but
# algo.teacher_training_replay_device=cpu stores the FIFO in host RAM and moves
# each sampled minibatch field to the policy GPU once before VecNorm/Q. It stores
# the current/next critic observations and learning targets, omits actor
# observations, and keeps the task-wide object geometry once instead of
# repeating it in every row.
# Stage-1 update bursts default to every control step
# (algo.sac_teacher_update_interval_env_steps=1). Increasing the interval still
# inserts every transition but skips optimizer bursts between boundaries; missed
# boundaries during replay warmup are not paid back later. For 1,024 envs, an
# 8-step interval with an 8,192 batch matches HOI's 8,192-env WBT critic/actor
# sample ratios and optimizer-step cadence once the actor is active:
# 4 Q updates and 2 actor updates per 8,192 newly collected transitions.
# Replay observations are captured before VecNorm and normalized with one
# current-stat snapshot after sampling. The SAC reward is the sum of VAIC reward
# groups. Only an episode time limit bootstraps from its real pre-reset final
# observation. Command/motion completion ends the SAC return without bootstrap,
# matching HOI WBT; command completion and true termination win over a
# simultaneous timeout. Stage-1 n-step returns are configurable with
# algo.sac_teacher_n_steps (default 1; use 4 for the delayed skateboard action).
# Rewards and the final bootstrap use the cumulative product of gamma and the
# environment's per-transition discount.
# Stage 1 has only the FastSAC actor/alpha update path: no reference-KL term,
# PPO warmup, or PPO behavior-distillation mode exists. An optional Stage-1-only
# uncertainty gate can
# suppress only the Q part of an actor sample unless the twin-Q mean improvement
# over that row's recorded replay action is positive and larger than the twins'
# disagreement about that improvement. The entropy term remains active on every
# row and the loss retains the full-batch denominator. The gate is disabled by
# default, so Stage 2 and historical Stage-1 behavior are unchanged.
# A separate Stage-1-only conservative-Q experiment is also disabled by
# default (`algo.sac_teacher_conservative_q_coef=0`). When enabled, it adds a
# smooth per-head penalty that increases when the detached deterministic
# teacher action is valued above that replay row's recorded action by more than
# `sac_teacher_conservative_q_margin`. It changes Q learning only: the SAC actor
# and entropy objectives are untouched. A null
# `sac_teacher_conservative_q_starts_q_updates` follows the actor-learning gate,
# including command-line gate overrides. The current reference action is used
# for both policy and replay actions under the reference-residual Q backend.
# By default, clipped
# double-Q selects the lower-expectation target head's complete C51 distribution
# as the common Q1/Q2 target and the actor uses min(Q1,Q2). Set
# algo.sac_clipped_double_q=false only to reproduce HOI's independent C51
# targets/twin-mean actor. Entropy autotuning follows HOI literally: sampled
# log-probability includes the affine action-scale Jacobian while the target is
# still -action_dim*ratio (with no affine offset). algo.sac_use_autotune=false keeps
# algo.sac_alpha_init as a fixed SAC entropy coefficient. SAC learning samples
# use checkpointed RNGs separate from rollout/environment randomization.
# fastsac_vel_train never writes the full actor+critic H5 required by Stage 2;
# that offline dataset must come from the separate collector described below.
python scripts/train.py algo=fastsac_vel_train task=G1/vaic/skateboard_tea
# Reference-residual Q experiment with the existing early-fusion critic:
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.q_action_coordinates=reference_residual
# Delayed-action credit + late action fusion experiment:
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.sac_teacher_n_steps=4 algo.q_action_fusion=late
# Opt-in conservative Q ranking on top of the chosen Q coordinates/fusion:
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.sac_teacher_conservative_q_coef=1.0 \
  algo.sac_teacher_conservative_q_margin=0.002 \
  algo.sac_teacher_conservative_q_temperature=0.002
# Optional teacher-only experiment: Q1/Q2, alpha, target Q, privileged encoder,
# and the teacher actor still train, but VAIC student adaptation/object modules
# and actor distillation do not. This checkpoint is not a pretrained-student
# warm-start for fastsac_vel_finetune; keep the default true for the full pipeline.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.train_student_models=false
# Express the compact learning FIFO as an HOI-style horizon per local vector
# environment. 2048*1024=2,097,152 rows require approximately 37.14 GiB for
# this task. At 1024 envs, the WBT horizon of 1024 steps is about 18.57 GiB for
# replay alone. Put that long FIFO in host RAM when GPU memory is insufficient
# (the sampled SAC minibatch still occupies GPU memory and host transfer can be
# slower than policy-local replay):
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  task.num_envs=1024 task.buffer_steps=1024 \
  algo.teacher_training_replay_device=cpu
# Equal-frame HOI WBT update cadence at 1,024 environments. The update interval
# changes Stage 1 only; Stage-2 RLPD scheduling is unaffected.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  task.num_envs=1024 task.buffer_steps=256 \
  algo.sac_batch_size=8192 \
  algo.sac_teacher_updates_per_env_step=4 \
  algo.sac_teacher_update_interval_env_steps=8 \
  algo.sac_teacher_policy_frequency=2
# Keep all 32 small critic optimizer steps over each 8 control steps, while
# giving each of the two delayed actor updates a larger independent minibatch.
# This retains actor sample UTD=2: 2*8192 actor samples per 8*1024 new rows.
# The default actor batch size of zero instead reuses the Q minibatch exactly.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  task.num_envs=1024 task.buffer_steps=256 \
  algo.sac_batch_size=1024 \
  algo.sac_teacher_actor_batch_size=8192 \
  algo.sac_teacher_updates_per_env_step=4 \
  algo.sac_teacher_update_interval_env_steps=1 \
  algo.sac_teacher_policy_frequency=16
# Optional Stage-1 actor safeguard ablation without a KL/action-distance
# penalty. This is not the original FastSAC path; monitor
# fastsac/actor_uncertainty_gate_acceptance_fraction and the confidence metrics
# when you explicitly enable it.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.sac_teacher_actor_uncertainty_gate=true
# Optionally retain reference-policy support without allocating a second
# replay. The prefix freezes immediately before the first actor update; only
# the suffix is overwritten online, while each minibatch draws 50% from each.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  task.num_envs=1024 task.buffer_steps=1024 \
  algo.teacher_training_replay_device=cpu \
  algo.sac_teacher_seed_storage_ratio=0.25 \
  algo.sac_teacher_seed_sample_ratio=0.5
# Optional Stage-1-only two-phase exploration. Under the unchanged global
# log-std bounds [-5, 0], collect broad behavior at log(std)=-1.5, narrow the
# teacher to log(std)=-2.5 immediately before Q update 8000, let Q adapt for
# another 2000 updates, and only then enable actor/alpha learning. The frozen
# replay prefix remains broad behavior and supplies 50% of each minibatch.
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  task.num_envs=1024 task.buffer_steps=256 \
  algo.sac_teacher_initial_log_std=-1.5 \
  algo.sac_teacher_actor_reset_log_std=-2.5 \
  algo.sac_teacher_actor_std_reset_q_updates=8000 \
  algo.sac_teacher_actor_learning_starts_q_updates=10000 \
  algo.sac_teacher_seed_storage_ratio=0.25 \
  algo.sac_teacher_seed_sample_ratio=0.5
# With 2,048 envs, task.buffer_steps=128 gives 262,144 rows (~4.64 GiB).
python scripts/train.py \
  algo=fastsac_vel_train task=G1/vaic/skateboard_tea \
  algo.train_student_models=false \
  task.num_envs=2048 task.buffer_steps=128

# Same-stage resume restores model weights, active AdamW moments, counters, and
# the dedicated replay/action-sampling RNG states. The simulator starts reset and the
# compact learning FIFO intentionally starts empty; Stage 1 has no replay
# H5 to restore or pair with the checkpoint.
# Old FastSAC checkpoints from before the corrected log-probability, raw replay,
# reward scalarization, timeout/command boundary, n-step-return, and
# normalized-Q-action semantics are
# intentionally incompatible;
# restart stage 1 with the current code.
# total_frames is a new additional training budget after the resume.
python scripts/train.py \
  algo=fastsac_vel_train \
  task=G1/vaic/skateboard_tea \
  task.num_envs=2048 \
  task.buffer_steps=64 \
  checkpoint_path=run:<fastsac_vel_train-wandb-run-path>

# evaluate policy
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/latest-run/files/checkpoint_6000.pt
python scripts/play.py algo=fastsac_vel_train task=G1/vaic/skateboard_tea checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-05/15-54-33-G1Skateboard-fastsac_vel/wandb/latest-run/files/checkpoint_300.pt
```

### PPO teacher DAgger, depth student, and Q replay

```bash
# Frozen PPO teacher + per-environment SafeDAgger control + student BC.
# VAIC depth/object/adaptation supervision and EMA updates remain enabled.
# Stage-2-compatible C51 Q1/Q2 learn from the actions actually executed;
# the student actor remains pure DAgger BC.
python scripts/bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/latest-run/files/checkpoint_6000.pt \
  bc_dagger_iterations=1200
```

`bc_dagger_iterations` is the number of outer DAgger rollout/update iterations;
the entrypoint derives the exact all-rank frame budget from `task.num_envs *
algo.train_every * world_size`. To retain SafeDAgger while also decaying beta to zero at a
specific cumulative DAgger/checkpoint index, use hybrid mode:

```bash
python scripts/bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/ppo_teacher_checkpoint.pt \
  bc_dagger_iterations=1200 \
  algo.dagger_control_mode=hybrid \
  algo.dagger_beta_zero_iteration=900
```

In this example, indices `0..899` use the linearly decaying beta in addition to
SafeDAgger, and indices `900..1199` use pure SafeDAgger. In `safe` mode beta is
unused; choose `beta` mode instead if SafeDAgger itself should be disabled.
W&B logs the zero-based index as `dagger/rollout_index`; the completed-rollout
counter `dagger/rollout_count` is therefore one larger.

The dedicated script defaults to SafeDAgger. It compares the clipped,
deterministic student action with the clipped PPO teacher mean in normalized
action coordinates. Normalized RMS error above `0.006` gives control to the
teacher; after at least eight control steps, error below `0.004` releases it
back to the student. Hysteresis is tracked independently for every environment.
`algo.dagger_control_mode=beta` retains the legacy Bernoulli schedule, while
`hybrid` applies SafeDAgger and uses beta only on otherwise-safe rows.

To deliberately end SafeDAgger control at a cumulative rollout boundary, use
the project-specific cutoff below. This is not part of original SafeDAgger:

```bash
python scripts/bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/ppo_teacher_checkpoint.pt \
  bc_dagger_iterations=2000 \
  algo.dagger_control_mode=safe \
  algo.dagger_safe_zero_iteration=1000
```

Indices `0..999` use SafeDAgger, and index `1000` onward is student-controlled;
the active safety latch is cleared at the boundary. The cutoff is cumulative
across a same-stage resume and must match the saved checkpoint configuration.
The frozen teacher is still evaluated for BC labels and discrepancy logging, so
this makes `dagger/safe_teacher_fraction` and (in `safe` mode)
`dagger/teacher_fraction` exactly zero without ending DAgger supervision. It
also overrides invalid-student takeover, so use it only when a student-only tail
is intentional. In `hybrid` mode it disables only SafeDAgger; beta can still
select the teacher until its separate beta schedule reaches zero. Before an
early cutoff, confirm `dagger/critic_teacher_replay_size` has reached at least
`algo.q_learning_starts_per_source` (default `8192`); otherwise the fixed 50/50
critic cannot begin updating because too few teacher-executed rows exist.

This path performs Huber behavior cloning plus SAC-compatible C51 clipped-double
Q pretraining; it does not call PPO/GAE/PPO-value optimization, a Q-weighted
actor loss, or a SAC actor/entropy optimizer. Q batches are always exactly 50%
teacher-executed and 50% student-executed transitions, independent of the
control mode. Bellman targets sample the student's dedicated small-noise next
action, but use effective alpha zero during this Q-only stage. There is no IQL
V network.
Only valid, actually teacher-executed transitions are exported to
`teacher_replay_buffer.h5`; the learning replay still contains every executed
teacher/student transition. Observations in this H5 are stored before VecNorm
and normalized with the checkpoint's fixed statistics when sampled. The
`fastsac_vel_finetune` loader accepts this paired DAgger schema; it transfers
the BC actor and pretrained Q/Q-target weights before ordinary Stage-2 FastSAC
begins.

BC sampling projects the replay to only the actor observation and teacher
label fields it consumes, and rollout transitions are staged into the learning
FIFOs once per outer iteration. These are automatic training-throughput
optimizations and do not change control decisions, sampled row IDs, optimizer
updates, or batch sizes. Checkpoints whose teacher FIFO has not changed reuse
the existing immutable H5. For fewer full H5 snapshots during SafeDAgger (where
teacher interventions may continue), add `save_interval=500`; this changes only
crash-recovery granularity, not learning.

To test block-coordinate BC-DAgger from a fresh PPO teacher, use the staged
entrypoint:

```bash
python scripts/stage_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/latest-run/files/checkpoint_6000.pt \
  bc_dagger_iterations=3000 \
  algo.dagger_control_mode=beta \
  algo.dagger_beta_zero_iteration=500
```

The default 3,000-rollout schedule is `500` joint warm-up rollouts, seven
coarse cycles of `100` perception-only plus `200` actor-only rollouts, then
`100` final perception rollouts, `172` final actor realignment rollouts, and
`128` frozen actor/perception replay-Q calibration rollouts. The initial joint
block is intentional: a fresh PPO teacher checkpoint has no trained temporal
depth CNN/GRU, so beta protects collection while that path boots up. The
perception blocks update only the depth CNN/GRU, `object_adapt`, and
`adapt_module`; actor blocks update only `actor_adapt`. The joint warm-up may
also warm-start Q, but Q stays frozen throughout the isolated perception/actor
cycles and is recalibrated only after their final representation is frozen.

Every perception/actor handoff clears the actor-learning FIFO because its
stored `priv_pred` belongs to the previous representation. At the final Q
boundary both learning FIFOs are cleared, beta is fixed to the configured
calibration probability (default `0.5`), and a new calibration-only
`teacher_replay_buffer.h5` is built. Intermediate checkpoints are useful for
evaluation but are rejected as Stage-2 sources; only the completed
`checkpoint_final.pt` and the canonical H5 in the same Hydra output root should
be passed to SAC/AWAC (the checkpoint itself lives under
`wandb/latest-run/files/`).
Staged resume is deliberately fail-fast in this first semantics version because
the ephemeral learning FIFOs are not checkpointed. Phase lengths can be
overridden, but `bc_dagger_iterations` must equal their exact sum. If
`joint_warmup_iterations` changes, set `algo.dagger_beta_zero_iteration` to the
same boundary as well.

After a joint BC-DAgger run has converged, finalize its representation and
replay/Q pair before Stage 2:

```bash
python scripts/bc_dagger_finalize.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/joint-bc-dagger/checkpoint_final.pt \
  perception_consolidation_iterations=25 \
  actor_realignment_iterations=0 \
  perception_recheck_iterations=0 \
  replay_q_calibration_iterations=128 \
  calibration_control_mode=beta \
  calibration_teacher_probability=0.5
```

The phases are isolated. Perception consolidation updates only the depth
CNN/GRU, `object_adapt`, and `adapt_module` with the existing supervised
privileged/object targets. Optional actor realignment updates only
`actor_adapt` by DAgger BC. Optional perception recheck again updates only the
three perception modules. At the calibration boundary both learning FIFOs are
cleared; actor and perception are frozen, a fixed beta controller collects a
brand-new teacher-only `teacher_replay_buffer.h5`, and only Q/Q-target are
updated from fresh teacher/student sources. The H5 paired with the input joint
checkpoint is deliberately neither read nor copied.

The example runs 153 outer rollouts (25 + 128), or 2,506,752 frames at 512
environments and `train_every=32`. The new output keeps the canonical
`checkpoint_final.pt` and `teacher_replay_buffer.h5` names; use this new pair,
not the input joint checkpoint, for Stage 2. Finalization currently starts only
from the original completed joint BC-DAgger checkpoint and rejects a partially
finalized checkpoint rather than risking a mixed replay lineage.

For this BC-DAgger bridge, Stage-2 **training** collection samples the same
bounded tanh-Gaussian distribution used by the SAC targets and actor loss. Its
dedicated log-standard-deviation starts at `0.01` raw action units, independently
of the unused PPO/DAgger `actor_std=0.5`. The exact bounded command sent to the
environment is retained in `ACTION_KEY` and therefore becomes the online replay
action paired with its observed reward and next state. Evaluation and deployment
remain the finite, clipped deterministic BC/SAC mean. A dedicated checkpointed
rollout RNG keeps behavior sampling independent of SAC gradient sampling and the
global environment RNG.

The DAgger safety clip and SAC entropy coordinates are deliberately separate.
The offline H5, rollout actor, environment, replay, and pretrained Q retain the
symmetric executable safety support `[-20, 20]` (from `dagger_action_clip`),
whereas entropy uses the fixed raw-action reference scale `1`. The safety bound
therefore cannot inject a `log(20)` density offset into every action dimension
or change the temperature merely because the emergency clip is wide. Set
`algo.sac_deterministic_rollout=true` only for the deterministic collection
ablation.

Stage 2 first collects 98,304 accepted online transitions, then performs 8,000
Q-only updates with actor, alpha, and perception optimizers frozen. The target
still samples the frozen stochastic next action so Q learns on the behavior's
actual support, but effective alpha and the entropy target term are **exactly
zero**. This is a temporary hard-Bellman compatibility bridge for the
BC-DAgger SAC critic, not removal of entropy from the subsequent SAC phase.

After the bridge, actor candidates are considered every 128 Q updates. An actor
tick is applied only when its predicted twin-Q gain over the frozen BC action is
positive and larger than the twins' disagreement. Effective alpha starts at
zero on the first confidence-approved actor tick and increases linearly to its
learned value over the next 20,000 Q updates; the Q target, actor objective, and
temperature update use that same effective alpha. Raw alpha starts at `1e-5`,
and the raw-unit entropy target is the standard `-action_dim`
(`sac_target_entropy_ratio=1`). There is no BC-loss anchor in this no-anchor SAC
path. The guarded Stage-2 defaults are `sac_batch_size=512`,
`q_lr=3e-5`, `sac_tau=0.001`, `sac_actor_lr=3e-7`, and
`sac_policy_frequency=128`.

Stage 2 can instead use the official AWAC actor update from the
[AWAC paper](https://arxiv.org/abs/2006.09359) and the authors'
[RLKit implementation](https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py):

```bash
python scripts/train.py \
  algo=fastsac_vel_finetune \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/bc-dagger/checkpoint_final.pt \
  algo.sac_finetune_actor_objective=awac \
  algo.sac_actor_confidence_gate=false \
  algo.sac_use_autotune=false \
  algo.sac_awac_beta=1.0 \
  algo.sac_updates_per_env_step=4 \
  algo.sac_policy_frequency=64 \
  algo.sac_actor_lr=3e-7
```

For each replay row this path computes
`A = min(Q1(s, a_replay), Q2(s, a_replay)) - E[a~pi] min(Q1(s,a), Q2(s,a))`
with the **online** critics, then minimizes replay-action negative log likelihood
with detached mean-one weights
`batch_size * softmax(A / sac_awac_beta)`. The defaults match the authors'
general implementation: `sac_awac_beta=1`, one current-policy value sample
(`sac_awac_v_samples=1`), and no upper score clamp
(`sac_awac_score_clip=null`). Both actor mean and standard deviation are trained.

AWAC uses a stochastic current-policy next action and a hard, clipped-double-Q
Bellman target: effective alpha, alpha autotuning, the Stage-2 confidence gate,
and the BC anchor are absent. Q/Q-target are **not frozen or restarted**: the
compatible BC-DAgger v3 Q weights warm-start Stage 2, receive the configured
Q-only bridge, and continue training from the mixed offline/online replay while
the AWAC actor improves. The distributional C51 critic and the explicit 50/50
RLPD replay mix are project integrations; the advantage and weighted-likelihood
actor update follow official AWAC rather than this repository's separate
Stage-1 `reference_awac` variant.

Legacy `vaic_ppo_bc_dagger_student_iql_v2` checkpoints are not loaded as AWAC
critics: their 101-atom/raw-action topology and expectile backup do not match the
current 501-atom/normalized-action policy-evaluation critic. For such a legacy
checkpoint use `algo.load_pretrained_q=false` to keep the BC actor but train a
fresh AWAC Q. Current `scripts/bc_dagger.py` checkpoints use the compatible
`vaic_ppo_bc_dagger_student_sac_critic_v3` critic and can retain the default
`algo.load_pretrained_q=true`.

The last three command overrides are a conservative VAIC starting profile, not
paper defaults: UTD 4 yields 128 Q updates and two AWAC actor updates per
32-step rollout while retaining the small BC-preserving actor learning rate.
The authors' much larger `3e-4` actor learning rate and every-Q-step policy
update should not be copied directly into this 23-action recurrent system. Do
not add the earlier `sac_actor_learning_starts_finetune_iteration=32` override;
leaving it null retains the 8,000-Q-update calibration bridge. If AWAC weights
collapse (`fastsac/awac_weight_ess_fraction` near zero), increase beta before
increasing actor learning rate or update frequency.

A v3 BC-DAgger source normally transfers its dedicated `0.01` physical-action
standard deviation exactly. Changing `sac_bc_initial_action_std` does not reset
that saved adapter. For an intentional offline-to-online exploration experiment,
use the Stage-2 boundary option below on a **fresh BC-DAgger transfer**:

```bash
algo.sac_stage2_initial_action_std=0.03
```

`null` (the default) preserves the source adapter bit-for-bit. A non-null value
resets it once after loading the BC-DAgger source; a same-Stage-2 resume restores
the learned adapter state and does not reset it again. The configured reset and
the current effective standard deviation are logged as
`fastsac/stage2_initial_action_std_config` and
`fastsac/bc_sac_center_action_std_mean`. Broader stochastic collection can lower
`train/stats/success` even when the deterministic evaluation mean improves, so
compare deterministic checkpoint evaluations rather than the exploration curve
alone.

The BC adapter's global `log_std` normally shares `sac_actor_lr` with the actor
mean. To let entropy change without accelerating the mean network, set an
independent Stage-2 learning rate:

```bash
algo.sac_actor_lr=1e-7 \
algo.sac_actor_log_std_lr=1e-4
```

The default `null` retains the legacy single AdamW parameter group exactly,
including checkpoint optimizer compatibility. A non-null value is supported
for the BC-DAgger adapter only: the actor mean remains at `sac_actor_lr`, while
only `bc_dagger_sac_adapter.log_std` uses the new rate. This changes optimizer
topology and is recorded in the Stage-2 resume schedule, so start the experiment
from a BC-DAgger checkpoint and repeat the exact value when resuming or
evaluating its Stage-2 checkpoints. Do not combine the first learning-rate
ablation with an action-std reset or a temperature change.

For a human-readable actor-release boundary, set the zero-based, Stage-2-local
finetune iteration instead of converting iterations into Q optimizer updates:

```bash
python scripts/train.py \
  algo=fastsac_vel_finetune \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/bc-dagger/checkpoint_final.pt \
  algo.sac_actor_learning_starts_finetune_iteration=250
```

Iterations `0..249` remain actor-frozen and iteration `250` is the first
eligible iteration. The first actual actor candidate still occurs on the next
`sac_policy_frequency` Q-update boundary and must pass the confidence gate.
The finetune-iteration counter starts at zero for a new Stage-2 transfer and is
restored cumulatively on a same-stage resume. Leaving the option null preserves
the historical `sac_actor_learning_starts_q_updates` gate exactly.

Perception remains frozen for the whole stage because replay stores `priv_pred`,
not the raw recurrent inputs needed to recompute it consistently. Stage-2
checkpoints created before this stochastic-hard-bridge, confidence-gate, and
alpha-ramp revision are intentionally incompatible. Do not resume an old
Stage-2 checkpoint; restart from the original BC-DAgger checkpoint.

To keep the BC student/depth/adaptation/EMA weights but start Stage-2 Q1/Q2
from a fresh initialization, disable only the Q-weight transfer:

```bash
python scripts/train.py \
  algo=fastsac_vel_finetune \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/bc-dagger/checkpoint_final.pt \
  algo.load_pretrained_q=false
```

The default is `true`. With `false`, Stage-1 Q weights and update counts are not
required; both target critics become frozen exact copies of the fresh online Q
networks. The BC actor and all student perception/EMA modules are still restored.

To continue only the model training state from a DAgger checkpoint while
leaving the existing `teacher_replay_buffer.h5` unchanged, use the dedicated
resume argument:

```bash
python scripts/bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  bc_dagger_checkpoint=/home/hcc/research/VAIC/outputs/2026-08-06/19-02-24-G1Skateboard-ppo_bc_dagger/wandb/latest-run/files/checkpoint_800.pt \
  bc_dagger_iterations=399
```

This restores the student/teacher modules, depth/adaptation modules and EMAs,
Q1/Q2 and target Q1/Q2, the BC/Q/adaptation optimizer states, the dedicated
DAgger/Q/SAC-action RNG states, and training counters. At startup it validates the source
H5's replay ID and VecNorm lineage, then makes an atomic, independent read-only
copy at `<new-output>/teacher_replay_buffer.h5`, outside the W&B-watched
`wandb/run-.../files/` directory. The source H5 is never modified, and the new
copy receives no additional transitions or snapshots. This currently duplicates
about 23 GiB; set `bc_dagger_copy_teacher_replay=false` to skip it.

`bc_dagger_iterations` is the number of additional iterations for the new
process: the example adds 399 rollouts to checkpoint 800's 801 completed
rollouts and ends at 1,200. Reusing `bc_dagger_iterations=1200` instead adds
another 1,200 rollouts and ends at 2,001. The in-memory all-transition learning ring and the simulator
episode state are not checkpointed, so they restart and the ring refills during
roughly the first eight rollouts. SafeDAgger's per-environment latch also starts
empty because the simulator starts from a fresh reset. A resumed run creates a
new W&B run.
Stage 2 can auto-discover the copied H5 from a local resumed checkpoint path.
The copy is not automatically uploaded to W&B, so `run:<resumed-run>` still
needs an explicit local `teacher_replay_buffer_path` unless the H5 is uploaded
separately. This applies to newly collected teacher H5 files too: automatic
W&B replay upload is disabled by default because these files are 20+ GiB. Use
`wandb.upload_teacher_replay=true` only for an intentional large upload.

The default CPU capacities are 131,072 learning rows and 1,048,576
teacher-export rows. For a smaller host-memory run, override both capacities with
`algo.dagger_buffer_capacity=65536 algo.teacher_buffer_capacity=131072`.

Student policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
# Train the student with 50% separately collected teacher H5 + 50% new online
# rollout data.
# The checkpoint transfers the FastSAC teacher/Q weights and the already
# distilled same-structure student actor; depth/adaptation + EMA keep VAIC logic.
# Stage-2 accepts a compatible teacher H5 from a different FastSAC run/iteration;
# schema, observation/action dimensions, backend, and action semantics must match.
python scripts/train.py algo=fastsac_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<fastsac_vel_train-wandb-run-path> teacher_replay_buffer_path=<fastsac_vel_buffer-path>
# Add algo.q_action_coordinates=reference_residual here when both the teacher
# checkpoint and compatible H5 were produced with that backend.
# A final fastsac_vel_finetune run also carries the selected compatible offline
# replay. Same-stage optimizer state is restored, while its online FIFO and the
# environment rollout state intentionally begin empty/reset.
# evaluate policy
python scripts/play.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
```
To export trained policies, add `export_policy=true` to the play script.


## Acknowledgments

This repository is built on top of [HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos](https://github.com/LeCAR-Lab/HDMI). We thank the authors for open-sourcing their work.

## Citation

If you find our work useful for your research, please consider citing us:

```bibtex
@article{li2026vaic,
  title = {VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands},
  author = {Li, Dongting and Wu, Qianyang and Chen, Xingyu and Li, Liang and Lin, Yuhang and Wu, Sikai and Zhang, Guoyao and Zhou, Mingliang and Xiang, Diyun and Zhang, Qiang and Xu, Renjing and Ma, Jianzhu},
  journal = {arXiv preprint arXiv:2606.09286},
  year = {2026}
}
```
