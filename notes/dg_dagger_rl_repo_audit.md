# TD3 + Teacher-BC DAgger repository audit

Audit date: 2026-08-11  
Audited commit: `5d1a6987efbcec77166adbfce76580e65517a417`  
Task scope: `G1/vaic/skateboard_stu`

This is the Phase 0 audit requested before any TD3 implementation. No training,
environment, task, observation, reward, action, configuration, checkpoint, or
test code was changed. The only Phase 0 changes are this note and
`notes/dg_dagger_rl_locked_interface.md`.

The task name was not explicit in the request. The audit uses
`G1/vaic/skateboard_stu` because the current DAgger command, entrypoint tests,
and saved artifacts all target it. The dimensions and effective rewards below
must not be generalized to an arbitrary Hydra `task=` override.

## 1. Current implementation

There is no TD3 implementation or TD3 entrypoint in the repository. The current
pipeline consists of:

- `scripts/bc_dagger.py`: dedicated validation/launch wrapper.
- `scripts/train.py`: generic rollout, environment stepping, logging,
  checkpoint, and evaluation loop.
- `active_adaptation/learning/ppo/ppo_bc_dagger.py`: deterministic Teacher-BC
  DAgger actor training plus an action-conditioned, stochastic
  SAC-compatible C51 critic calibration path.
- `active_adaptation/learning/ppo/fastsac_vel.py`: shared action-coordinate,
  timeout, C51, and twin-Q utilities used by the current critic.
- `cfg/bc_dagger.yaml`: current BC-DAgger schedule.

The actor is currently optimized only by the existing BC loss. The Q network is
not used for a TD3 actor objective. A thin wrapper around `bc_dagger.py` would
therefore not implement the requested method.

## 2. Student Actor observation and deterministic path

Observation groups and terms retain OmegaConf insertion order. The Student
Actor's direct flat input is exactly:

| Ordered key | Width | Construction |
| --- | ---: | --- |
| `vel_command` | 20 | Five future steps of root velocity `[vx, vy, yaw_rate]` (15), then five object-contact flags (5) |
| `policy` | 249 | Root angular velocity (3), projected gravity (3), six 29-joint position history frames (174), then three 23-action history frames (69) |
| `priv_pred` | 256 | Output of the existing student adaptation/perception stack |
| **Total** | **525** | Ordered concatenation `[vel_command, policy, priv_pred]` |

Source construction is in `active_adaptation/learning/ppo/ppo_vel.py` and the
Q/replay fingerprints are in
`active_adaptation/learning/ppo/ppo_bc_dagger.py:1342-1361`.

The fixed path producing `priv_pred` is also part of the Actor interface:

1. `depth` has shape `[1, 36, 64]` and retains the configured camera noise,
   delay, sanitization, clipping, scaling, quantization, and dropout.
2. The depth encoder is three `Conv2d(5x5, stride=2, padding=2)` layers with
   eight channels and Mish activations, followed by a 64-D projection and a
   64-D temporal GRU.
3. `object_adapt_ema` consumes the existing `policy`, `vel_command`, and depth
   feature path and predicts the 22-D object state.
4. The existing object transform produces `object_pred_trans` (384-D) using
   the predicted object pose and `object_geo_`.
5. `adapt_ema` consumes ordered
   `[policy:249, vel_command:20, object_pred:22,
   object_pred_trans:384]` (675-D), then uses the existing 256-D MLP/GRU path
   to produce `priv_pred:256`.
6. `actor_adapt` consumes the 525-D flat Actor input.

The Actor body is unchanged VAIC architecture:

```text
525 -> Linear(512)/LayerNorm/Mish
    -> Linear(256)/LayerNorm/Mish
    -> Linear(256)/LayerNorm/Mish
    -> actor_mean Linear(256, 23)
```

For legacy checkpoint compatibility, `actor_adapt` remains wrapped as a
`ProbabilisticActor` and retains an inherited 23-D `actor_std` parameter.
DAgger control, BC, and evaluation never sample it: they explicitly use
`actor_adapt.get_dist(...).mean`. The control policy is therefore deterministic,
and TD3 must continue to use only that mean. It must not add or use log standard
deviation, log probability, entropy, temperature, or policy sampling.

The 23-D Actor mean is an unconstrained pre-`tanh` latent `z`, not an executable
joint target. The noise-free Student command is:

```text
a_student = project(20 * tanh(z), low=-20, high=20)
```

These are dimensionless absolute `JointPosition` commands. They are neither
radians nor residuals around the current reference.

### Observation normalization

`scripts/helpers.py` applies VecNorm to non-boolean observation groups whose
names do not end with `_`. Consequently `command`, `policy`, `depth`, `priv`,
and `vel_command` use frozen checkpoint normalization in this pipeline;
`object_`, `object_geo_`, and `ref_joint_pos_` are not VecNorm-normalized.
`priv_pred` is an encoded latent and is not passed through VecNorm.

Replay aliases are copied before VecNorm and normalized exactly once after a
batch is sampled. TD3 must retain this behavior.

## 3. Critic observation and action path

The action-conditioned Critic input is exactly:

| Ordered key | Width | Exact term order |
| --- | ---: | --- |
| `priv` | 1714 | Detailed below |
| `policy` | 249 | Same normalized policy group described above |
| `command` | 356 | Five-step body positions (240), joint references (115), motion phase (1) |
| `object_` | 22 | Object XY (2), heading cos/sin (2), two reference contacts (6), object position (3), object orientation matrix (9) |
| **Total** | **2341** | Ordered concatenation `[priv, policy, command, object_]` |

`priv` is ground-truth privileged state, not `priv_pred`. Its exact ordered
widths are:

| Term | Width |
| --- | ---: |
| `root_ang_vel_history` | 27 |
| `projected_gravity_history` | 27 |
| `joint_pos_history` | 261 |
| `ref_root_pos_future_b` | 15 |
| `ref_root_ori_future_b` | 30 |
| `diff_body_pos_future_local` | 240 |
| `diff_body_ori_future_local` | 480 |
| `diff_body_lin_vel_future_local` | 240 |
| `diff_body_ang_vel_future_local` | 240 |
| `root_linvel_b` | 3 |
| ankle `body_pos_b` | 6 |
| ankle `body_vel_b` | 6 |
| `body_height` | 4 |
| `applied_action` | 23 |
| `applied_torque` | 29 |
| `object_pos_b` | 3 |
| `object_ori_b` | 9 |
| `diff_object_pos_future` | 15 |
| `diff_object_ori_future` | 45 |
| `ref_object_contact_future` | 5 |
| `diff_contact_pos_b` | 6 |
| **Total** | **1714** |

The inherited PPO state-value critic is frozen/unused by BC-DAgger and is not
an action-value network. It must not be mistaken for the TD3 Critic.

### Existing twin Q networks

The repository already has two independently instantiated action-conditioned Q
heads in `TwinDistributionalQ`, plus a deep-copied, gradient-disabled target
twin. Each Q head has the exact topology:

```text
critic_obs 2341 -> Linear(768)/LayerNorm/SiLU
action 23       -> Linear(128)/LayerNorm/SiLU
concat 896      -> Linear(384)/LayerNorm/SiLU
                -> Linear(192)/LayerNorm/SiLU
                -> Linear(501)
```

The output is a categorical distribution over 501 fixed atoms on `[-20, 20]`.
The scalar Q reported by the helper is its expectation. Current target logic
selects the target head with the lower expectation and retains that head's
complete distribution. The current online update is a C51 projection and
cross-entropy update driven by a stochastic SAC-compatible next action.

There is no target Actor. TD3 must add an exact copy of `actor_adapt`, retain
the checkpoint-compatible parameters, feed it the stored next 525-D Actor
input, and use only its deterministic mean.

### Critic action representation

Replay retains the issued physical command `a` in the 23-D dimensionless
absolute command coordinates. Immediately before Q evaluation it becomes:

```text
u = (a - nominal_joint_center) / nominal_joint_half_range
```

The gain is 1.0 and there is deliberately no `[-1, 1]` clamp. Center and scale
are derived at runtime from the robot's soft joint limits, default pose, and
per-joint action scaling, and are serialized in the action contract. They must
not be hard-coded. The executable support `[-20, 20]` and the nominal Q
coordinate system are intentionally separate.

TD3 target-policy noise must be defined in Q coordinates `u`, clipped there,
and constrained by the execution support transformed into those coordinates.
Equivalently, the per-joint Q-coordinate noise can be mapped into physical
command units, followed by the existing physical `[-20, 20]` projection and
the unchanged Q transform. Collector exploration must use the same explicit
coordinate conversion, affect only Student rows, and store the final physical
command in replay.

## 4. Effective reward contract

Reward configuration comes from:

- `cfg/task/base/hdmi-base.yaml`
- `cfg/task/G1/vaic/skateboard_stu.yaml`
- `active_adaptation/envs/base.py`
- `active_adaptation/envs/mdp/base.py`
- `active_adaptation/envs/mdp/rewards/**`
- `active_adaptation/envs/mdp/commands/hdmi/rewards.py`

Each primitive returns `configured_weight * raw_term`. Enabled primitives are
added within their group. The environment emits the exact five-vector group
order `[tracking, object_tracking, loco, feet, debug]`; every group is
multiplied by `step_dt=0.02`. Replay uses an unmodified sum over the five group
values as the scalar Q reward. There is no reward clipping, reward VecNorm,
imitation/intervention bonus, success bonus, terminal bonus/penalty, or extra
TD3 reward. `success` is logging state only. Environment discount is 1.0.

### `tracking` group

All ten terms are enabled and have configured weight `0.5`:

| Term | Parameters |
| --- | --- |
| `tracking_upper_body_pos` | shoulder-pitch/elbow/wrist-yaw bodies, sigma 0.5 |
| `tracking_upper_body_ori` | same bodies, sigma 1.0 |
| `tracking_lower_body_pos` | hip-pitch/knee/ankle-roll bodies, sigma 0.5 |
| `tracking_lower_body_ori` | same bodies, sigma 1.0 |
| `tracking_root_pos` | pelvis, sigma 0.5 |
| `tracking_root_ori` | pelvis, sigma 0.5 |
| `tracking_body_linvel` | configured tracked bodies, sigma 0.5 |
| `tracking_body_angvel` | configured tracked bodies, sigma 2.5 |
| `joint_pos_tracking_product` | configured waist/hip/knee/shoulder/elbow joints, sigma 0.25 |
| `joint_vel_tracking_product` | same joints, sigma 2.5 |

The position, orientation, velocity, and joint errors use their existing
exponential tracking formulas. Their maximum configured group contribution is
5 before the `step_dt` multiplier.

### `object_tracking` group

| Term | Weight | State and parameters |
| --- | ---: | --- |
| `object_pos_tracking` | 1.0 | enabled, sigma 0.5 |
| `object_ori_tracking` | 1.0 | enabled, sigma 0.5 |
| `object_vel_tracking` | 1.0 | **disabled**, sigma 0.5 |
| `eef_contact_exp` | 1.0 | enabled, position sigma 0.3, force sigma 40, force threshold 10, gain 5 |

### `loco` group

| Term | Weight | State and parameters |
| --- | ---: | --- |
| `action_rate_l2` | 0.1 | enabled; negative squared issued-command change |
| `joint_vel_l2` | 0.0005 | enabled; all joints |
| `joint_pos_limits` | 10.0 | enabled; all joints, soft factor 0.9 |
| `joint_torque_limits` | 0.01 | enabled; soft factor 0.6 |
| `survival` | 1.0 | enabled; constant +1 raw term |

### `feet` group

The skateboard task overrides the first two body selections to the left ankle
only and adds the skateboard-specific term.

| Term | Weight | State and parameters |
| --- | ---: | --- |
| `impact_force_l2` | 1.0 | enabled, left ankle only |
| `feet_slip` | 0.5 | enabled, tolerance 0, left ankle only |
| `feet_air_time` | 1.0 | **disabled**, threshold 0.3 |
| `feet_air_lift` | 1.0 | **disabled**, low/high 0.10/0.18, sigma 1.0 |
| `feet_contact` | 1.0 | **disabled**, sigma 0.25 |
| `survival` | 1.0 | enabled; constant +1 raw term |
| `feet_air_time_skateboard` | 5.0 | enabled, both ankles, threshold 0.2, soft discount 1.0 |

### `debug` group

The group is emitted as exactly zero. All configured terms are disabled:
`feet_air_time` (5), `feet_contact_count` (1), `joint_pos_limits` (1),
`eef_contact_all` (1), `root_pos_error` (1), `root_ori_error` (1),
`body_pos_error_local` (1), `body_ori_error_local` (1), and
`joint_pos_error` (1).

## 5. Environment and action pipeline

The skateboard student task uses 512 environments, a maximum of 1000 control
steps (20 seconds), `step_dt=0.02`, and Isaac physics `dt=0.005`, giving four
physics substeps per 50 Hz control step. The task uses a 64x36 tiled depth
camera. Physics, randomization, reset, command completion, and termination stay
entirely in the existing environment.

The 23 controlled joints use the action scales configured in
`cfg/task/base/hdmi-base.yaml`: hip yaw 0.55, hip roll 0.35, hip pitch 0.55,
knee 0.35, ankle pitch/roll 0.44, waist roll/pitch 0.44, waist yaw 0.55,
shoulder pitch/roll/yaw 0.44, and elbow 0.44. Wrist controls are disabled.

The exact rollout pipeline is:

1. Compute the deterministic Teacher and noise-free Student candidates.
2. Teacher candidate is `ref_joint_pos_ + frozen PPO residual mean`.
3. Student candidate is the projected affine-`tanh` Actor mean described above.
4. Check Teacher validity before projection; invalid Teacher rows cannot execute.
5. SafeDAgger computes discrepancy from projected **noise-free** candidates.
6. Current default `safe` control uses takeover RMS 0.006, release RMS 0.004,
   and a minimum eight Teacher steps. `hybrid` adds beta selection to rows not
   already selected by safety.
7. Beta mode computes the unchanged linear schedule and draws one Bernoulli
   source choice per environment and control step. Beta is categorical source
   selection, not numeric action interpolation.
8. Existing behavior performs `torch.where(teacher_source, teacher_action,
   student_action)` and stores the exact result as `action`.
9. `scripts/train.py` passes that exact command to
   `env.step_and_maybe_reset`.
10. `JointPosition` subsequently delays it by 2--6 physics substeps, applies a
    per-environment low-pass alpha sampled in `[0.8, 1.0]` on every physics
    substep, and constructs the radian target:

```text
joint_target_rad = default_joint_pos
                 + randomized_joint_offset
                 + applied_action * per_joint_action_scale
```

`JointPosition` does not perform another action clamp. The random joint offset
is in `[-0.01, 0.01]` radians.

There is no existing collector exploration noise. The surrounding TorchRL
exploration context does not change DAgger because Teacher and Student both use
distribution means. Current stochastic next-action code is SAC-specific Q
calibration logic and must not be reused or stacked with TD3 noise.

For the new collector, exploration is permitted only after source selection on
rows whose selected source is Student. SafeDAgger discrepancy and BC must still
use the noise-free Student candidate. The final projected noisy Student command
must replace `action` only on those Student rows; Teacher and evaluation rows
must remain bitwise unaffected.

### Meaning of `executed_action`

The locked replay/Critic contract defines `executed_action` as the final 23-D
dimensionless command handed to the environment. It is the existing
`current["action"]`, after source choice, optional Student-only exploration,
and the execution projection. Teacher action remains a separate BC label.

The later delayed/filtered `action_manager.applied_action` and radian PD target
are internal, history-dependent environment state. Replacing replay action
with either would change the locked Critic action representation and is not
allowed.

## 6. Replay, BC loss, done, and bootstrap

`_DeviceReplay` in `ppo_bc_dagger.py` is the in-memory circular transition
store. Current transition assembly retains:

- raw Actor and Critic observations and their true next observations;
- the exact selected/projected issued action;
- a separate projected Teacher label and validity mask;
- `is_student` source metadata;
- the exact five-vector reward scalarized only by summation;
- done, custom timeout, and environment discount;
- true pre-reset timeout observations;
- a `step_count > 5` initial-transition filter.

Phase 1 may add backward-compatible audit fields for noise-free Student,
exploratory Student, action source, beta, and noise, but it must not overwrite
the exact issued action or Teacher label.

### Exact existing Teacher BC loss

The authoritative implementation is
`active_adaptation/learning/ppo/ppo_bc_dagger.py:2903-2944`:

1. Select rows with valid Teacher labels.
2. Compute the noise-free Student latent with
   `actor_adapt.get_dist(actor_obs).mean`.
3. Convert the projected physical Teacher command to the Actor's latent using
   the existing inverse affine-`tanh` transform and epsilon behavior.
4. Apply mean-reduced `smooth_l1_loss` with
   `beta=dagger_actor_huber_delta` (currently 1.0).
5. Clip Actor gradients using the existing `max_grad_norm`.

The requested combined objective requires a single delayed Actor optimizer
step over:

```text
eta_td3 * (-mean(Q1(critic_obs, noise_free_actor_action)))
+ lambda_bc * exact_existing_BC_loss
```

A separate TD3 step followed by `_bc_update`, or vice versa, would not be the
specified objective and is forbidden.

### Termination/bootstrap truth table

The environment's `truncated` flag combines time limits and command completion.
The current custom timeout helper distinguishes them. TD3 must reuse the same
mask and pre-reset final observation handling:

| Transition | Bootstrap |
| --- | ---: |
| Ordinary | 1 |
| Pure episode time limit | 1 |
| True termination | 0 |
| Command completion | 0 |
| Time limit plus true termination | 0 |

It must not bootstrap directly from raw environment `truncated`.

## 7. Checkpoint and commands

Current root checkpoints contain W&B, policy, environment, resolved config,
and optionally VecNorm. BC-DAgger adds its algorithm/backend/action-contract
markers, Actor/perception state, twin Q/target Q, optimizers, counters, RNG
streams, and replay provenance.

A TD3 checkpoint needs a distinct, versioned algorithm marker and must include
the online Actor, target Actor, both online Q heads, both target Q heads, Actor
and Critic optimizers, delayed-update counter, beta/source RNG, exploration RNG,
target-noise RNG, action/observation contracts, VecNorm fingerprint, and any
enabled gate state. Exact numerical continuation also requires a complete
transition replay sidecar or serialized replay; the current H5 retains the
Teacher partition rather than every Student transition.

Current BC-DAgger training command:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/path/to/PPO_teacher_checkpoint.pt \
  bc_dagger_iterations=1200
```

Current deterministic Student evaluation command:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/path/to/run/wandb/latest-run/files \
  --config-name=cfg \
  checkpoint_path=/path/to/run/wandb/latest-run/files/checkpoint_final.pt \
  task.num_envs=512
```

`train_command.txt` currently references nonexistent `scripts/rl_bc_dagger.py`
and is stale. It was not modified in Phase 0.

The available completed Aug-10 artifact is an older v3/backend-v1/H5-v2
artifact, while current source expects v6/backend-v4/H5-v5. Its shapes support
the audit, but it is not a valid current resume fixture.

## 8. Blocking design conflict

The existing twin Critic's exact output semantics are 501-atom C51
distributions. Textbook TD3 uses scalar Q heads and a scalar Bellman MSE. The
request simultaneously requires standard TD3 and forbids changing the existing
Critic topology/output semantics. These requirements have no literal
implementation that satisfies both without an explicit interpretation:

1. **Interface-preserving distributional TD3 (recommended locked-interface
   alternative):** retain C51 logits/projection, use deterministic target Actor
   plus clipped smoothing, select the complete target distribution from the
   lower expected target Q, and use expected Q1 for the Actor loss. This
   preserves the existing Critic output semantics but is distributional TD3,
   not textbook scalar-MSE TD3.
2. **Expectation-only TD3:** retain the 501-logit head but optimize only its
   scalar expectation with TD3 MSE. This matches scalar TD3 equations at the Q
   interface, but the 501-atom distribution ceases to have calibrated C51
   semantics.
3. **Scalar-head TD3:** replace each 501-logit output with a scalar. This is the
   textbook implementation but changes protected Critic topology, output
   semantics, and Q checkpoint compatibility.

No TD3 code should be written until the intended interpretation is authorized.

## 9. Proposed file-by-file TD3-side changes (after authorization)

The baseline and protected files can remain untouched by adding a parallel
implementation:

- `scripts/TD3_bc_dagger.py` (new): dedicated Hydra entrypoint, strict source
  checkpoint/algorithm/action-contract validation, schedule handling, and
  dispatch to the unchanged generic training loop. It will reject SAC fields
  and incompatible checkpoints rather than auto-select another algorithm.
- `cfg/TD3_bc_dagger.yaml` (new): inherit `train`, select only the new TD3
  structured config, copy the existing DAgger/SafeDAgger defaults unchanged,
  and add explicit TD3 hyperparameters/feature gates.
- `active_adaptation/learning/ppo/td3_bc_dagger.py` (new): structured config,
  deterministic rollout wrapper, backward-compatible transition metadata,
  existing replay/normalization/timeout logic, existing exact BC computation,
  independent twin/target Critic use, target Actor, TD3 update scheduling,
  Polyak updates, logging, and versioned checkpoint save/load. It must not
  construct the current SAC action adapter or call stochastic policy APIs.
- `tests/test_td3_bc_dagger.py` (new): transition identity, source/noise
  separation, twin independence, target gradient isolation, clipped target
  smoothing, lower-target selection, `-Q1` Actor loss, combined exact BC loss,
  delayed updates, Polyak, timeout truth table, deterministic evaluation, and
  checkpoint round trip.
- `tests/test_td3_bc_dagger_entrypoint.py` (new): Hydra composition,
  fail-fast compatibility checks, forbidden SAC-field checks, and source/resume
  checkpoint validation.
- `tests/test_td3_locked_interface.py` (new): skateboard observation/reward/
  action fingerprints and seeded baseline parity with all TD3 flags disabled.

No edit is proposed to `scripts/train.py`, `scripts/helpers.py`,
`ppo_bc_dagger.py`, task/environment files, existing baseline config, or
existing baseline tests. If implementation proves impossible without one of
those edits, work must stop and request authorization with the exact diff.

## 10. Phase 0 verification

The existing focused baseline suite was run read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/hcc/anaconda3/envs/vaic/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_ppo_bc_dagger.py \
  tests/test_bc_dagger_entrypoint.py \
  tests/test_fastsac_timeout.py
```

Result: `147 passed, 3 warnings`.

Minimal commands proposed after implementation are recorded in the locked
contract note and should be run before any simulator-scale training.
