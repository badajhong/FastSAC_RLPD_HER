# G1 Skateboard FastSAC-BC-DAgger performance evaluation

Evaluation date: 2026-08-13 (Asia/Seoul)

## Outcome

The completed FastSAC-BC-DAgger run is **not competitive with the existing
deployable Student baselines**. Its built-in deterministic final evaluation
completed the motion in **58/128 environments (45.31%)**, with a Wilson 95%
confidence interval of **36.95%-53.95%**.

The dominant observed failure is sustained **torso world-position drift**,
followed by **loss of required ankle-skateboard contact**. From the aggregate
evaluation moments, the 70 failed environments terminated at an estimated mean
reference frame of **391.9 +/- 84.1**, or **7.84 +/- 1.68 seconds** into the
12.42-second motion. The mean lies in the fast, two-feet-on-board glide segment,
about 0.4 seconds after the reference first requires both ankles on the board.
This timing is an aggregate reconstruction, not a measured failure histogram.

The strongest explanation is an undertrained, cold-start Student perception
pipeline rather than a numerically unstable twin critic:

- The run initialized directly from the privileged PPO checkpoint. The startup
  log shows that the depth CNN, temporal-depth GRU and its EMA, and Q networks
  were absent from that source checkpoint.
- The ten forced-Teacher prefill rollouts collected critic replay only and
  performed no perception, Actor, or Critic optimization.
- Every main rollout used `beta=0`, so the newly initialized Student perception
  and stochastic Student policy controlled immediately.
- The run used only 12.33 million environment transitions, approximately 9.3
  times fewer than the recent staged BC-DAgger run.
- Final perception losses remained about twice the recent BC-only baseline,
  whereas the twin-Q, C51-support, and gradient diagnostics remained stable.

Training success was still improving near the end. The result is therefore
better described as **underconverged and poorly initialized** than as a proven
failure of FastSAC itself.

## Evaluated run and checkpoint

| Item | Value |
|---|---|
| Run | `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger` |
| Primary checkpoint | `wandb/latest-run/files/checkpoint_final.pt` |
| Algorithm marker | `distributional_fastsac_teacher_bc_v1` |
| Checkpoint version | 1 |
| Actor backend | `ppo_bc_dagger_mean_plus_global_log_std_fastsac_v1` |
| Main rollouts | 3,000 |
| Teacher-only prefill | 10 rollouts |
| Training environments | 128 |
| Frames per rollout | 4,096 (`128 x 32`) |
| Total physical frames | 12,328,960 |
| Actor/Critic/alpha updates | 11,992 each |
| Training commit | `1c849027edc6abded6871b7dc1bfdbbc0d5e311b` |

The final checkpoint records 3,000 completed main rollouts and 10 completed
prefill rollouts. Training ended cleanly after about 2 hours 31 minutes; there
is no OOM, crash, or incomplete-save indication.

### Effective learning setup

- `dagger_beta_start=0` and `dagger_beta_end=0`: all main behavior was generated
  by the stochastic Student; no Teacher action controlled a main rollout.
- The 10-rollout prefill produced **39,457 valid frozen Teacher transitions**.
- Critic batches mixed frozen Teacher and Student transitions 50:50.
- At this training commit, Actor updates sampled the main replay only, and
  perception updates used the live main rollout only.
- `q_updates_per_rollout=4`, `sac_policy_frequency=1`, `eta_sac=1e-4`,
  `lambda_bc=1`, and initial physical action standard deviation `0.05`.

The later `teacher_actor_replay_fraction` and
`teacher_perception_replay_fraction` features were added after this run. This
result must not be interpreted as an evaluation of those new options.

## Evaluation protocol

The reported result is the automatic post-training evaluation performed on the
in-memory final policy:

- deterministic Student mean action;
- no Teacher action selection and no stochastic SAC sample;
- seed 0;
- 128 parallel environments;
- frame-zero start of the full 622-frame reference motion;
- 1,000-step horizon at 50 Hz;
- the run's Student task, camera, delay, noise, and domain randomization;
- checkpoint VecNorm statistics in evaluation mode;
- rendering disabled.

`success` and `episode_len` use each environment's first episode. Other
evaluator metrics are normalized by that first episode's length before the
across-environment mean and standard deviation are calculated. Consequently,
the small `eval/termination/*` scalars are **not direct failure rates**.

Training `train/stats/success` is not the same protocol: it uses stochastic SAC
actions and randomized reference start phases. It should not be compared
directly with the deterministic frame-zero evaluation success.

## Main deterministic evaluation results

| Metric | Result |
|---|---:|
| Success | **58/128** |
| Success rate | **45.3125%** |
| Wilson 95% CI | **36.95%-53.95%** |
| Failed environments | 70/128 |
| Episode length | 495.70 +/- 130.19 steps |
| Tracking return | 0.079411 +/- 0.001614 |
| Object return | 0.062950 +/- 0.006717 |
| Locomotion return | 0.018834 +/- 0.000704 |
| Feet return | 0.018427 +/- 0.000371 |

### Tracking and control diagnostics

| Metric | Mean +/- SD | Interpretation |
|---|---:|---|
| Object-position tracking | 0.79808 +/- 0.06214 | Relatively strong |
| Object-orientation tracking | 0.88378 +/- 0.06139 | Below recent Student baselines |
| End-effector contact reward | 1.46566 +/- 0.32805 | Clear contact weakness |
| Required-contact fraction | 0.92618 +/- 0.11017 | Below recent Student baselines |
| Root-position error | 0.15377 +/- 0.03351 | Sustained tail errors can still terminate |
| Root-orientation error | 0.10656 +/- 0.03968 | Secondary issue |
| Local body-position error | 0.06655 +/- 0.01473 | Worse than recent BC-only Student |
| Local body-orientation error | 0.19497 +/- 0.03046 | Worse than recent BC-only Student |
| Joint-position error | 0.08862 +/- 0.00606 | Not the leading failure signal |
| Action-rate penalty | -0.03557 +/- 0.01208 | Less smooth than recent baselines |
| Foot-slip penalty | -0.06411 +/- 0.02059 | Non-negligible but not dominant |
| Impact-force penalty | -0.00294 +/- 0.00521 | Worse than recent baselines |

Object position itself is not the primary weakness. Contact quality, torso/body
tracking, and control smoothness degrade more clearly.

## Comparison with existing Student baselines

| Model/checkpoint | Deterministic success | Environments | Important caveat |
|---|---:|---:|---|
| **Current FastSAC-BC-DAgger final** | **45.31%** | 128 | Direct PPO cold start; 12.33M frames |
| 2026-08-12 PPO BC-DAgger final | 83.40% | 512 | Staged training; substantially larger budget |
| 2026-08-11 FastSAC 0%-offline final | 90.43% | 512 | Warm-started from mature BC-DAgger pipeline |
| 2026-08-11 FastSAC 50%-offline final | 90.82% | 512 | Warm-started from mature BC-DAgger pipeline |
| 2026-08-11 PPO checkpoint 42,300 | 94.92% | 512 | Different lineage and algorithm |

Even the upper end of the current model's 95% interval, 53.95%, is far below
the previous deployable Students. However, this is not a clean algorithm-only
comparison: initialization, number of environments, total data, update rules,
Teacher-control schedule, commits, and evaluation sample size differ.

The closest recent BC-only comparison shows where quality was lost:

| Metric | Current FastSAC-BC | 2026-08-12 BC-only | Change |
|---|---:|---:|---:|
| Success | 45.31% | 83.40% | -38.09 points |
| Episode length | 495.70 | 597.11 | -101.41 steps |
| Tracking return | 0.07941 | 0.08024 | -1.0% |
| Object return | 0.06295 | 0.06729 | -6.5% |
| Locomotion return | 0.01883 | 0.01922 | -2.0% |
| Required-contact fraction | 0.92618 | 0.98564 | -0.05946 |
| Contact reward | 1.46566 | 1.69008 | -13.3% |
| Local body-position error | 0.06655 | 0.05995 | +11.0% worse |
| Local body-orientation error | 0.19497 | 0.18651 | +4.5% worse |
| Action-rate penalty magnitude | 0.03557 | 0.03009 | +18.2% worse |
| Impact penalty magnitude | 0.00294 | 0.00192 | +53.5% worse |

## Where failures occur

### Aggregate deterministic-evaluation timing

The evaluator did not save per-environment trajectories, but the first two
moments can be reconstructed exactly from its aggregate outputs:

- 58 successful episodes finish at 621 steps;
- all 128 first episodes total 63,450 steps;
- therefore the 70 failures average **391.89 steps**;
- removing the constant-length successes from the reported variance gives a
  failure-only SD of approximately **84.07 steps**.

At 50 Hz, failures therefore end at **7.84 +/- 1.68 seconds**, approximately
**63.1%** through the reference. A one-SD interval is roughly frames 308-476.

The reference contact and board-motion schedule is:

| Reference interval | Time | Motion state |
|---|---:|---|
| Frames 0-129 | 0.00-2.58 s | No ankle-board contact required; board stationary |
| Frames 130-371 | 2.60-7.42 s | Right ankle required on board; board accelerates to about 1.2 m/s |
| Frames 372-470 | 7.44-9.40 s | **Both ankles required on board; high-speed glide then deceleration** |
| Frames 471-520 | 9.42-10.40 s | Right ankle remains; board decelerates to rest |
| Frames 521-621 | 10.42-12.42 s | No ankle-board contact required; finish |

The mean failure at frame 392 occurs only 20 frames (0.4 seconds) after the
second ankle becomes required, while the skateboard is moving at about 1.12
m/s. Termination conditions require 25 consecutive bad steps, so the underlying
error may begin near the transition itself. The strongest available conclusion
is therefore:

> The vulnerable region is plausibly the transition into, and early part of,
> the two-feet-on-moving-board balance/glide segment.

This is **not proof that frame 392 is the modal failure frame**. Without retained
trajectories, the exact phase histogram cannot be recovered.

### Measured failure causes

The deterministic evaluator retained only episode-length-normalized termination
signals. Their relative signal mass is:

| Cause | Share of normalized termination signal |
|---|---:|
| Sustained torso world-position error | 58.87% |
| Lost required ankle-board contact | 27.53% |
| Local tracked-body orientation error | 10.65% |
| Local tracked-body position error | 1.66% |
| Torso orientation error | 1.29% |

These percentages rank causes but are not recoverable event counts. Stronger
count data are available from the final seven logged stochastic-training
windows (steps 2,816-3,008):

| Termination flag | Incidences | Share of 1,624 failed episodes |
|---|---:|---:|
| Torso world-position drift | 896 | 55.17% |
| Lost ankle-board contact | 482 | 29.68% |
| Local body-orientation error | 143 | 8.81% |
| Local body-position error | 101 | 6.22% |
| Torso-orientation error | 72 | 4.43% |
| Object-position/orientation error | 0 | 0% |

Flags can overlap, so percentages do not sum to 100%. The same ordering remains
stable across the complete training history.

The termination definitions help interpret the result:

- torso position: torso world-position error at least 0.5 m for 25 consecutive
  steps;
- lost contact: any reference-required ankle misses either its 0.2 m board
  target or the 1 N force threshold for 25 steps;
- local orientation: any tracked body exceeds 1.2 rad for 25 steps.

Object-position and object-orientation termination use `min_steps=5000`, much
longer than this 622-frame motion. Their zero incidence is structurally expected
and must not be interpreted as proof of perfect skateboard tracking.

The saved aggregate artifacts cannot identify the exact left/right ankle that
lost contact or the specific tracked body that exceeded the local error limit,
because the termination implementations reduce them with `any`/`max` before
logging.

## Training dynamics and diagnosis

### Progress and checkpoint selection

Mean stochastic training success by 500-main-rollout bin rose approximately as
follows:

| Main-rollout range | Mean training success |
|---|---:|
| 0-500 | 0.4015 |
| 500-1,000 | 0.4737 |
| 1,000-1,500 | 0.4973 |
| 1,500-2,000 | 0.5114 |
| 2,000-2,500 | 0.5352 |
| 2,500-3,000 | 0.5682 |

Among numeric saves, `checkpoint_3000.pt` has the highest trailing-save-window
training-success mean, **0.60234**. The raw maximum, **0.64068**, was also near
the end at logged step 2,976. `checkpoint_final.pt` is nine main rollouts newer
and is the only checkpoint with a deterministic post-training evaluation.
There is no evidence that an earlier checkpoint is better under the actual
deployment protocol.

The upward tail means the run had not convincingly plateaued. Training longer
could improve it, but the initialization and early control-distribution problem
should be addressed before treating extra compute as the primary remedy.

### Perception is the clearest bottleneck

The PPO source checkpoint did not provide loadable depth CNN or temporal-depth
GRU states. Prefill collected data without optimization, and `beta=0` exposed
the cold Student immediately. At the end:

- privileged-feature loss: approximately 0.3480 versus 0.1805 for the recent
  BC-only run (+92.9%);
- object-prediction loss: approximately 0.00332 versus 0.00181 (+83.4%);
- adaptation gradient norm exceeded the configured clip threshold throughout
  the run, and the privileged loss was still trending downward near the end.

The degraded contact and body-tracking evaluation metrics are consistent with
this underconverged representation.

### The critic is not obviously unstable

Final critic diagnostics are numerically well behaved:

- expected Q1/Q2: 7.4586 / 7.4605;
- twin disagreement: 0.03684, about 0.49% of the Q scale;
- C51 support clipping: 0% left and 0.0506% right;
- per-head cross entropy: 5.3413 / 5.3408 with target entropy 5.3296;
- critic gradient norm: 0.6024.

These values do not prove the critic learned ideal action discrimination, but
they rule out obvious twin divergence, support saturation, or exploding critic
gradients as the leading explanation for the 45% success.

### Actor and entropy diagnostics

- exact mean-action BC loss: `4.45e-5`;
- weighted SAC Actor term: `-7.16e-4`;
- weighted BC term: `4.45e-5`;
- total Actor gradient norm: `8.37e-4`;
- alpha: `1.27e-5`;
- entropy contribution relative to reward: about 0.395%;
- mean log standard deviation: `-5.857`.

The logged SAC scalar is about 16 times the BC scalar in absolute magnitude,
but Q contains state/action-independent offsets; scalar loss magnitude alone
does not establish the gradient ratio. Entropy remained weak rather than
excessive. The evidence points more strongly to representation/data coverage
than to uncontrolled stochastic exploration.

## Limitations

1. **Only 128 deterministic environments were evaluated.** The prior report
   used 512 environments. The confidence interval is reported to expose that
   sampling uncertainty.
2. **No evaluation trajectories, videos, or per-environment terminal records
   were saved.** Exact phase histograms, left-versus-right contact loss, and
   failing body identity cannot be reconstructed retrospectively.
3. **Standard fresh reload is currently blocked.** The FastSAC policy's public
   loader intentionally rejects its own same-stage checkpoint because training
   resume is fresh-only. `scripts/eval.py` therefore cannot canonically reload
   this final checkpoint. The 128-env result succeeded because evaluation ran
   on the in-memory policy immediately after training.
4. **No private-loader bypass was used for this report.** This respects the
   request not to change code and avoids presenting an unsupported reload path
   as canonical evaluation.
5. **Single-seed and unequal-budget comparison.** The current run, prior
   FastSAC runs, staged BC-DAgger, and PPO differ in initialization, frame
   budget, control schedule, update cadence, and commit.
6. **This result predates the new Teacher Actor/perception replay options.** It
   cannot evaluate their effect.

## Recommended interpretation

- Do not select this checkpoint over the existing 83%-91% deployable Student
  checkpoints.
- Treat torso translational drift and ankle-board contact through the
  two-feet-on-board transition as the primary observed failure mode.
- Treat perception warm-start/training coverage as the first hypothesis to
  test; the current evidence does not justify blaming C51/twin-Q instability.
- For exact failure localization in a future run, retain only compact
  per-environment first-termination records: reference frame, success, all
  termination flags, and ankle contact status. No such artifact exists for this
  completed run.

## Raw artifacts

- Final checkpoint:
  `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger/wandb/latest-run/files/checkpoint_final.pt`
- Runtime config:
  `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger/wandb/latest-run/files/cfg.yaml`
- W&B summary:
  `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger/wandb/latest-run/files/wandb-summary.json`
- Full W&B history:
  `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger/wandb/latest-run/run-bx3fjgei.wandb`
- Training log:
  `outputs/2026-08-13/00-12-10-G1Skateboard-fastsac_bc_dagger/fastSAC_bc_dagger.log`
- Reference motion:
  `data/motion/g1/mirobotA/board6/motion.npz`

