# Locked interface contract: TD3 + Teacher-BC DAgger

Status: Phase 0 contract, implementation blocked pending Critic-semantics choice  
Task: `G1/vaic/skateboard_stu`  
Baseline commit: `5d1a6987efbcec77166adbfce76580e65517a417`

This document is normative for subsequent phases. A proposed implementation
that cannot satisfy a clause here must stop, identify the exact clause and
file, and obtain explicit authorization before changing it.

## 1. Algorithm boundary

The new method is TD3 plus the exact existing Teacher BC objective.

It must not contain or construct:

- SAC policy logic or a SAC fallback;
- DDPG fallback or automatic algorithm selection;
- stochastic Actor sampling;
- log-standard-deviation heads or adapters;
- log probabilities, entropy objectives, temperature alpha, or target entropy.

The inherited `actor_std` parameter remains solely for Actor topology and
checkpoint compatibility. The TD3 path always obtains the deterministic Actor
output with `.get_dist(...).mean`; it never reads `actor_std` to choose an
action.

## 2. Locked observation contracts

### Actor

The direct Actor concatenation, order, widths, source, preprocessing, and
normalization are immutable:

```text
[vel_command:20, policy:249, priv_pred:256] -> 525
```

Exact substructure:

- `vel_command:20` = five `[vx, vy, yaw_rate]` future references followed by
  five object-contact flags.
- `policy:249` = root angular velocity 3, projected gravity 3, joint position
  history 174, previous-action history 69.
- `priv_pred:256` = existing temporal depth/object/adaptation encoder output.

The depth image remains `[1, 36, 64]`; the depth GRU, object predictor,
object-geometry transform, adaptation GRU, hidden-state reset behavior, EMA
modules, histories, observation noise, delays, and normalization are all part
of this lock. No privileged state may be added to the Actor.

Actor-normalized raw groups continue to use the frozen VecNorm checkpoint.
Groups ending in `_` stay unnormalized. `priv_pred` remains the encoder output
and receives no additional normalization.

### Critic

The action-value Critic concatenation, order, widths, source, preprocessing,
and normalization are immutable:

```text
[priv:1714, policy:249, command:356, object_:22] -> 2341
```

Exact ordered `priv` fields and widths:

```text
root_ang_vel_history:27
projected_gravity_history:27
joint_pos_history:261
ref_root_pos_future_b:15
ref_root_ori_future_b:30
diff_body_pos_future_local:240
diff_body_ori_future_local:480
diff_body_lin_vel_future_local:240
diff_body_ang_vel_future_local:240
root_linvel_b:3
body_pos_b:6
body_vel_b:6
body_height:4
applied_action:23
applied_torque:29
object_pos_b:3
object_ori_b:9
diff_object_pos_future:15
diff_object_ori_future:45
ref_object_contact_future:5
diff_contact_pos_b:6
```

`command:356` remains ordered body-position future 240, joint-position future
115, and phase 1. `object_:22` remains ordered object XY 2, heading 2,
reference contacts 6, object position 3, and object orientation 9.

The Critic always uses ground-truth `priv` and `object_`; it must not be changed
to `priv_pred` or a Student object prediction. Raw replay observations continue
to be normalized exactly once after sampling. `object_` remains unnormalized.

## 3. Locked network contracts

### Deterministic Actor

The online Actor remains:

```text
525 -> 512/LN/Mish -> 256/LN/Mish -> 256/LN/Mish -> 23-D mean
```

Its 23-D mean is a pre-`tanh` latent. Its noise-free physical command is the
existing affine-`tanh` mapping into the per-joint execution support (currently
`[-20, 20]` for every joint), followed by the existing sanitization/projection.
The Actor body, head, output dimension, output meaning, and state-dict keys are
immutable.

TD3 may add `actor_target` only as an exact parameter copy of this Actor.
Target evaluation also uses only `.mean`. Target Polyak updates may not alter
the online Actor's module names or state-dict layout.

### Critic

The current action-conditioned twins are independent. Each has:

```text
obs stem:    2341 -> 768/LN/SiLU
action stem:   23 -> 128/LN/SiLU
joint trunk:  896 -> 384/LN/SiLU -> 192/LN/SiLU -> 501 logits
C51 support: 501 atoms on [-20, 20]
```

The existing target twin is an exact copy with gradients disabled. Q1 and Q2
must never share parameter storage. Targets must never be included in an
optimizer and must receive no optimizer gradients.

The unresolved choice is whether TD3 retains C51 training semantics or changes
them. Until explicitly authorized, neither scalar heads nor an expectation-only
loss may replace the C51 output contract. See section 11.

## 4. Locked reward contract

The exact emitted reward is a five-vector in this order:

```text
[tracking, object_tracking, loco, feet, debug]
```

Each enabled primitive returns `weight * raw_term`; primitives add within a
group; each group is multiplied by `step_dt=0.02`. The Q replay reward is
exactly `reward_vector.sum(-1)`. There is no other scaling, clipping,
normalization, aggregation, success reward, terminal reward, imitation reward,
intervention penalty, or TD3-specific bonus.

| Group | Term | Weight | Enabled / fixed parameters |
| --- | --- | ---: | --- |
| tracking | `tracking_upper_body_pos` | 0.5 | yes, sigma 0.5 |
| tracking | `tracking_upper_body_ori` | 0.5 | yes, sigma 1.0 |
| tracking | `tracking_lower_body_pos` | 0.5 | yes, sigma 0.5 |
| tracking | `tracking_lower_body_ori` | 0.5 | yes, sigma 1.0 |
| tracking | `tracking_root_pos` | 0.5 | yes, sigma 0.5 |
| tracking | `tracking_root_ori` | 0.5 | yes, sigma 0.5 |
| tracking | `tracking_body_linvel` | 0.5 | yes, sigma 0.5 |
| tracking | `tracking_body_angvel` | 0.5 | yes, sigma 2.5 |
| tracking | `joint_pos_tracking_product` | 0.5 | yes, sigma 0.25 |
| tracking | `joint_vel_tracking_product` | 0.5 | yes, sigma 2.5 |
| object_tracking | `object_pos_tracking` | 1.0 | yes, sigma 0.5 |
| object_tracking | `object_ori_tracking` | 1.0 | yes, sigma 0.5 |
| object_tracking | `object_vel_tracking` | 1.0 | **no**, sigma 0.5 |
| object_tracking | `eef_contact_exp` | 1.0 | yes, pos sigma 0.3, force sigma 40, threshold 10, gain 5 |
| loco | `action_rate_l2` | 0.1 | yes |
| loco | `joint_vel_l2` | 0.0005 | yes, all joints |
| loco | `joint_pos_limits` | 10.0 | yes, soft factor 0.9 |
| loco | `joint_torque_limits` | 0.01 | yes, soft factor 0.6 |
| loco | `survival` | 1.0 | yes |
| feet | `impact_force_l2` | 1.0 | yes, left ankle only |
| feet | `feet_slip` | 0.5 | yes, tolerance 0, left ankle only |
| feet | `feet_air_time` | 1.0 | **no**, threshold 0.3 |
| feet | `feet_air_lift` | 1.0 | **no**, thresholds 0.10/0.18, sigma 1.0 |
| feet | `feet_contact` | 1.0 | **no**, sigma 0.25 |
| feet | `survival` | 1.0 | yes |
| feet | `feet_air_time_skateboard` | 5.0 | yes, both ankles, threshold 0.2, soft discount 1.0 |
| debug | all nine configured debug terms | configured | all disabled; group is zero |

Reward values, the reward tensor, done flags, and next observations must be
stored without mutation. Only the existing five-group sum may produce the
scalar Bellman reward.

## 5. Locked environment/task configuration

Exact governing configuration paths:

- `cfg/task/base/hdmi-base.yaml`
- `cfg/task/G1/vaic/skateboard_stu.yaml`
- `cfg/base/sim_base.yaml`
- `cfg/base/randomization_base.yaml`
- `active_adaptation/assets/g1.py`
- `data/motion/g1/mirobotA/board6/meta.json` and its referenced motion data

The following effective values are immutable:

- task name `G1Skateboard`;
- 512 Student environments by default;
- maximum episode length 1000 control steps;
- control `dt=0.02` seconds;
- Isaac physics `dt=0.005` seconds and decimation 4;
- 23-dimensional action;
- 64x36 depth camera and all camera/sensor randomization;
- every physics, asset, actuator, reset, termination, timeout, command,
  reference, curriculum, and randomization setting.

## 6. Locked action contract

### Controlled joints and scaling

The runtime Isaac action order is:

```text
left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint,
left_hip_roll_joint, right_hip_roll_joint, waist_roll_joint,
left_hip_yaw_joint, right_hip_yaw_joint, waist_pitch_joint,
left_knee_joint, right_knee_joint,
left_shoulder_pitch_joint, right_shoulder_pitch_joint,
left_ankle_pitch_joint, right_ankle_pitch_joint,
left_shoulder_roll_joint, right_shoulder_roll_joint,
left_ankle_roll_joint, right_ankle_roll_joint,
left_shoulder_yaw_joint, right_shoulder_yaw_joint,
left_elbow_joint, right_elbow_joint
```

The corresponding scale vector is:

```text
[0.55, 0.55, 0.55, 0.35, 0.35, 0.44, 0.55, 0.55, 0.44,
 0.35, 0.35, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
 0.44, 0.44, 0.44, 0.44, 0.44]
```

### Teacher, Student, and source selection

- Teacher command: `ref_joint_pos_ + frozen_PPO_residual_mean`, followed by
  existing validity and projection logic.
- Student command: deterministic Actor mean latent, existing affine `tanh`,
  then existing projection.
- SafeDAgger discrepancy uses projected, noise-free candidates and its existing
  RMS/hysteresis thresholds.
- Beta remains the existing rollout-count linear probability and a per-row
  Bernoulli source choice. It is never a numeric blend coefficient.
- `safe`, `beta`, and `hybrid` retain their exact current meanings.
- New exploration must not influence Teacher validity, Teacher label,
  SafeDAgger discrepancy, beta draw, or source choice.

### Exploration and exact executed action

Baseline collector exploration is absent. When explicitly enabled in TD3
training:

1. Compute the noise-free Student command.
2. Perform unchanged safety/beta source selection.
3. On Student-selected rows only, add one exploration-noise sample in the
   documented nominal Q coordinate system, map it back to physical command
   coordinates, and reuse the existing execution projection.
4. On Teacher-selected rows, execute the projected Teacher command unchanged.
5. Store the exact final 23-D command handed to `env.step_and_maybe_reset` as
   `executed_action`/the canonical replay `action`.
6. Store Teacher label, noise-free Student, exploratory Student, source, beta,
   and noise only as separate metadata.
7. Evaluation forces beta/source to Student-only and exploration off.

The environment action manager then applies its locked 2--6 physics-substep
delay and `[0.8, 1.0]` low-pass filter before constructing:

```text
target_radians = default_radians
               + randomized_offset_radians
               + filtered_command * joint_scale
```

The internal filtered command and radian target are not the replay/Critic
action and may not replace it.

### Q action coordinates and TD3 noise

The Q network receives:

```text
q_action = (executed_physical_command - runtime_nominal_center)
         / runtime_nominal_half_range
```

with gain 1 and no clamp. Center/scale and transformed executable bounds must
come from the runtime action contract; they may not be constants.

Target Actor smoothing must:

1. compute the deterministic target Actor command;
2. transform it to Q coordinates;
3. add zero-mean target noise clipped to the configured noise clip;
4. enforce the existing physical execution support, expressed in Q
   coordinates (never an invented `[-1,1]` bound);
5. feed the resulting exact Q coordinate to both target Critics.

Target and collector noise need separate generators/state. Neither may use
`actor_std` or current SAC-compatible sampling code.

## 7. Locked transition and bootstrap contract

Each transition binds all of these fields from the same environment step:

```text
(actor_obs_t,
 critic_obs_t,
 exact_executed_action_t,
 reward_vector_t,
 actor_obs_t+1,
 critic_obs_t+1,
 terminated_t,
 truncated_t,
 custom_timeout_t,
 discount_t)
```

Teacher action is a separate BC label and never substitutes for
`exact_executed_action_t`. No reward or next observation may be paired with a
different candidate or source. True pre-reset next observations remain in use
for pure timeouts.

The bootstrap mask remains:

```text
ordinary transition                 -> 1
pure max-episode timeout             -> 1
physical/other true termination      -> 0
command completion                   -> 0
timeout plus true termination        -> 0
```

## 8. Locked optimization rules

Subject to the unresolved C51 interpretation, the TD3 rules are:

- Q1 and Q2 train independently from exact executed-action transitions.
- The target uses the lower of target Q1 and target Q2 for each row.
- Target actions use clipped target-policy smoothing as specified above.
- Online Actor action is deterministic and noise-free.
- Actor TD3 loss is exactly `-mean(Q1(...))`, never `-min(Q1,Q2)`.
- Actor total loss is exactly:

```text
eta_td3 * standard_td3_actor_loss
+ lambda_bc * exact_existing_teacher_bc_loss
```

- BC stays mean-reduced SmoothL1 in the existing pre-`tanh` latent coordinates
  with the existing inverse transform, validity mask, Huber beta, and noise-free
  Actor output.
- Both Actor terms are combined before one backward/optimizer step.
- Actor and all target networks update only every `policy_delay` Critic updates.
- Target updates use Polyak averaging.
- Critic targets are detached; target networks receive no optimizer gradients.

With all TD3 additions disabled, the existing BC-DAgger update cadence and
outputs must match the baseline under the same seed.

## 9. Checkpoint contract

TD3 uses a distinct algorithm/version marker. It must save and strictly restore:

- online Actor under its compatible existing key;
- target Actor;
- both independent online Q heads;
- both target Q heads;
- Actor and Critic optimizer states;
- Critic-update and delayed-policy counters;
- DAgger rollout/source RNG state;
- collector exploration RNG state;
- target smoothing RNG state;
- action and observation contract fingerprints;
- VecNorm identity/fingerprint;
- gate state when a gate is enabled;
- replay lineage and, if exact numerical resume is promised, complete replay
  contents/cursor in a versioned sidecar.

Fresh compatible PPO Teacher initialization and same-stage TD3 resume are
different modes and must fail fast on the wrong marker. Existing BC-DAgger
checkpoints and command remain functional and are never rewritten in place.

## 10. PROTECTED FILES

No subsequent phase may modify these paths without explicit authorization.
Directory globs protect every current and future file beneath that directory.

### Task, simulation, reward, observation, action, and randomization configs

- `cfg/task/**`
- `cfg/base/**`
- `cfg/train.yaml`
- `cfg/bc_dagger.yaml`
- `cfg/bc_dagger_finalize.yaml`

### Environment, MDP, physics, asset, camera, and sensor implementation

- `active_adaptation/envs/base.py`
- `active_adaptation/envs/locomotion.py`
- `active_adaptation/envs/scene.py`
- `active_adaptation/envs/humanoid.py`
- `active_adaptation/envs/mujoco.py`
- `active_adaptation/envs/mdp/base.py`
- `active_adaptation/envs/mdp/action.py`
- `active_adaptation/envs/mdp/randomizations.py`
- `active_adaptation/envs/mdp/terminations.py`
- `active_adaptation/envs/mdp/rewards/**`
- `active_adaptation/envs/mdp/observations/**`
- `active_adaptation/envs/mdp/commands/hdmi/**`
- `active_adaptation/assets/**`
- `active_adaptation/sensors/**`
- robot/object USD and other simulator asset files
- `data/motion/**`
- `asset_meta.json`

### Existing Actor/Critic/normalization/baseline behavior

- `active_adaptation/learning/ppo/common.py`
- `active_adaptation/learning/modules/distributions.py`
- `active_adaptation/learning/ppo/ppo_vel.py`
- `active_adaptation/learning/ppo/fastsac_vel.py`
- `active_adaptation/learning/ppo/ppo_bc_dagger.py`
- `scripts/helpers.py`
- `scripts/train.py`
- `scripts/bc_dagger.py`
- `scripts/bc_dagger_finalize.py`

### Existing regression tests

- `tests/test_ppo_bc_dagger.py`
- `tests/test_bc_dagger_entrypoint.py`
- `tests/test_fastsac_timeout.py`

New implementation must live in new TD3-side files. Existing helpers may be
imported, but SAC action sampling, entropy, adapter, log-probability, and alpha
paths must not be invoked. If a protected edit becomes necessary, stop before
editing and provide the proposed exact change and why a new-file solution is
not viable.

## 11. Blocking Critic decision

An explicit user decision is required before Phase 1 because the current C51
output contract conflicts with textbook scalar TD3:

- **Recommended for the highest-priority interface lock:** retain C51 output
  and C51 projection, use deterministic smoothed target actions, choose the
  lower expected target head's complete distribution, and optimize the Actor
  with expected Q1. This is interface-preserving distributional TD3.
- **Alternative:** retain the 501 logits but MSE only their expectation against
  the scalar TD3 target. This follows scalar TD3 equations but abandons
  calibrated distribution semantics.
- **Textbook scalar TD3:** change the final Q layer from 501 to 1. This violates
  the protected Critic topology/output/checkpoint contract and needs explicit
  authorization to edit the contract.

No option should be selected silently.

## 12. Minimal regression and smoke-test commands

Baseline must remain green before and after every phase:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/hcc/anaconda3/envs/vaic/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_ppo_bc_dagger.py \
  tests/test_bc_dagger_entrypoint.py \
  tests/test_fastsac_timeout.py
```

After the new files exist, run syntax and focused tests:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python -m py_compile \
  scripts/TD3_bc_dagger.py \
  active_adaptation/learning/ppo/td3_bc_dagger.py

PYTHONDONTWRITEBYTECODE=1 \
/home/hcc/anaconda3/envs/vaic/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_td3_locked_interface.py \
  tests/test_td3_bc_dagger.py \
  tests/test_td3_bc_dagger_entrypoint.py
```

Hydra composition/fail-fast smoke test:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  --cfg job task=G1/vaic/skateboard_stu
```

Minimal simulator smoke test, after a compatible PPO Teacher checkpoint is
supplied and the final TD3 config names are fixed:

```bash
WANDB_MODE=disabled \
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/compatible_PPO_teacher.pt \
  task.num_envs=2 \
  td3_dagger_iterations=4 \
  algo.td3_learning_starts=1 \
  algo.q_batch_size=2 \
  algo.q_updates_per_rollout=1 \
  algo.policy_delay=2 \
  wandb.mode=disabled
```

The focused test suite must assert, at minimum:

- exact observation key order and widths 525/2341;
- exact reward tensor and scalarization parity;
- exact final action identity from collector through replay into both Critics;
- Teacher action/label never receives exploration noise;
- deterministic Student evaluation has beta 0 and noise 0;
- Q1/Q2 and all target parameters are independent;
- target modules have no optimizer gradients;
- smoothing noise is clipped in the correct Q coordinates and honors the
  transformed physical execution bounds;
- lower target Q is used, while Actor loss uses Q1 only;
- exact BC value matches the baseline helper on the same batch;
- Actor/target updates occur only at `policy_delay`;
- Polyak update is numerically correct;
- timeout/bootstrap truth table and true final observations are preserved;
- no SAC/log-probability/entropy/alpha/log-std path is present;
- checkpoint round trip restores all required modules, optimizers, counters,
  RNGs, and gates without changing deterministic evaluation output.
