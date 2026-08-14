# G1 Skateboard FastSAC-BC-DAgger model performance evaluation

Evaluation date: 2026-08-14 (Asia/Seoul)

## Outcome

Among the two predeclared candidates evaluated from the completed run, the
recommended checkpoint is **`checkpoint_final.pt`**. In a fresh evaluation
using deterministic Student actions, it completed the motion in **465/512
environments (90.82%)**, with a Wilson 95% confidence interval of
**88.01%-93.03%**.

The numeric checkpoint selected from the training log, `checkpoint_8400.pt`,
completed **461/512 environments (90.04%)**. The final therefore adds four
successful environments, or 0.78 percentage points. Its normalized tracking,
object, locomotion, and feet returns are also all slightly higher. The
approximate unpaired-binomial 95% interval for the success-rate difference is
-2.82 to +4.38 points, so the run does not establish a statistically material
difference between the two checkpoints; the final is recommended because it
has the better observed deployment result and is the canonical terminal state.

The fresh final result closely reproduces the run's in-memory post-training
evaluation: **465/512 versus 464/512**, while the normalized reward metrics are
nearly identical. This is strong evidence that checkpoint, VecNorm, and Student
inference state reload correctly.

Descriptively, the final ties the 465/512 completion count of the 50%-offline
FastSAC final in the 2026-08-11 report, exceeds the reported PPO BC-DAgger final
by one environment, and remains below PPO checkpoint 42,300's 486/512. Those
are cross-version, single-run comparisons rather than a controlled new
benchmark; the older checkpoints were not rerun for this report.

## Evaluated run and checkpoints

| Item | Value |
|---|---|
| Run | `outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger` |
| W&B run | `wdf9zfis` |
| Runtime config | `wandb/run-20260813_190116-wdf9zfis/files/cfg.yaml` |
| Algorithm marker | `distributional_fastsac_teacher_bc_v1` |
| Checkpoint version | 3 |
| Actor backend | `ppo_vel_normalized_std_tanh_bounded_fastsac_bc_v1` |
| Primary checkpoint | `checkpoint_final.pt` |
| Log-selected numeric checkpoint | `checkpoint_8400.pt` |
| PPO Teacher/partial-perception source | `outputs/15-13-46-G1Skateboard-ppo_vel/wandb/run-20260805_151350-7tsje71w/files/checkpoint_final.pt` |
| Main rollouts | 9,000 |
| Teacher-only prefill | 16 rollouts; 131,072 retained transitions |
| Training environments | 512 |
| Logged env frames per rollout | 16,384 control transitions (`512 x 32`) |
| Logged rollout-training env frames | 147,718,144 control transitions |
| Actor/Critic updates | 36,000 / 36,000 |
| Entropy coefficient | Fixed at approximately `1e-5`; 0 alpha updates |
| Recorded end-to-end W&B runtime | 31,931 seconds (about 8 h 52 min) |

Training and the automatic 1,000-control-step final evaluation ended normally with
exit code 0. The Teacher buffer and Student replay each reached their configured
131,072-row capacity. `checkpoint_9000.pt` and `checkpoint_final.pt` contain
identical deserialized model, environment, configuration, and VecNorm state;
only the final artifact is evaluated here to avoid duplicate work.

The online Teacher and Student replay rings are not serialized in the
checkpoint, and this run did not save a separate HDF5 Teacher buffer. This is
irrelevant to inference, but an exact same-stage training resume is
intentionally unavailable; the checkpoint records
`fresh_only_online_raw_perception_rings_not_serialized_v1`.

### Effective learning setup

- `dagger_beta_start=0` and `dagger_beta_end=0`: every main rollout was executed
  by the stochastic Student. Teacher control was forced only during prefill.
- Actor and Q learning batches used 50% Student-executed replay rows and 50%
  frozen Teacher-executed replay rows. Ten percent of the Teacher quota was
  focused on phases preceding observed Student failures, yielding nominal
  source categories of 50% Student / 45% uniform Teacher / 5% failure-phase-
  focused Teacher. The focused rows are successful Teacher transitions, not
  failed Teacher transitions. Actor rows from both sources retain Teacher
  action labels for BC supervision.
- Perception learning combined its online loss and a separate 128-row Teacher
  replay loss with weights 0.5/0.5; this is a loss mixture, not one 50/50 row
  minibatch. It also completed 128 Teacher-perception warmup updates before main
  training.
- The configured PPO train checkpoint supplied a **partial** perception warm
  start: `object_adapt`, `object_adapt_ema`, `adapt_module`, and `adapt_ema` were
  loaded, while `depth_cnn`, `temporal_depth_gru`, and its EMA were initialized
  fresh. The fresh depth path received the 128 warmup updates and remained
  trainable throughout main training.
- The actor objective used `eta_sac=1e-3`, `lambda_bc=1`; each rollout performed
  four C51 critic and four actor updates. SAC autotuning was disabled.

These are learning-time mixtures only. Evaluation uses no Teacher action,
Teacher policy switch, replay buffer, stochastic SAC sample, or Q critic for
control.

## Checkpoint-selection method

The run contains only one post-training `eval/*` record, produced from the final
policy on the same task and seed, so numeric checkpoints cannot be ranked by
historical deployment evaluation. As
in `MODEL_PERFORMANCE_EVALUATION_2026-08-11.md`, the canonical episodic training
metric `train/stats/success` is used as a proxy.

For each saved numeric checkpoint `k`, all logged success samples in the
preceding save interval `(k - 100, k]` were averaged. The ranking uses
`fastsac/rollout_count`, not W&B `_step`: the 16 Teacher-prefill records offset
the global W&B step counter from the main-rollout checkpoint numbers.

| Selected checkpoint | Save interval | Samples | Mean training success | Supporting samples |
|---|---:|---:|---:|---|
| `checkpoint_8400.pt` | `(8300, 8400]` (width 100) | 3 | **0.763346** | Rollouts 8,305/8,337/8,369: 0.744882, 0.784817, 0.760339 |

The raw maximum, 0.784817 at rollout 8,337, is included in that saved interval
rather than treated as an independently selectable checkpoint. The final tail
interval `(8900, 9000]` averaged 0.761729 from three samples. Final checkpoints
are evaluated separately and excluded from the numeric-save ranking.

Training success rose throughout most of the run and then approximately
plateaued:

| Main-rollout range | Logged samples | Mean stochastic training success |
|---|---:|---:|
| 1-1,000 | 31 | 0.541605 |
| 1,001-2,000 | 31 | 0.660672 |
| 2,001-3,000 | 32 | 0.699801 |
| 3,001-4,000 | 31 | 0.718533 |
| 4,001-5,000 | 31 | 0.732354 |
| 5,001-6,000 | 31 | 0.741439 |
| 6,001-7,000 | 32 | 0.748637 |
| 7,001-8,000 | 31 | 0.757452 |
| 8,001-9,000 | 31 | 0.757947 |

These training statistics use stochastic actions and randomized reference start
phases. They are suitable for within-log checkpoint selection, but their values
must not be compared directly with deterministic, frame-zero evaluation
success.

## Evaluation protocol

- Repository evaluator: `scripts/eval.py` and `scripts/helpers.py:evaluate`
- Evaluation HEAD: `f6476f4524cea6d3ef5bd42204d4597e8400b2d0`
- The run-ID-qualified, fully composed, hash-pinned W&B `cfg.yaml`
- Deterministic Student policy (`ExplorationType.MODE`)
- Seed: 0
- Parallel environments: 512
- Horizon: 1,000 environment/control steps (four physics substeps each)
- Task: G1 skateboard Student task with the run's domain randomization
- Motion: `data/motion/g1/mirobotA/board6`
- Vector normalization: checkpoint statistics in evaluation mode
- Depth cameras: enabled as required Student observations
- Output rendering: disabled
- GPU: NVIDIA GeForce RTX 5090

Success and episode length use each environment's first episode. Every other
evaluator statistic is divided by that episode's length before its
across-environment mean and standard deviation are computed. `episode_cnt`
counts all terminations across the complete 1,000-step rollout and is not the
success denominator.

The complete deployable Student path is evaluated: temporal-depth GRU EMA,
object-adaptation EMA, adaptation GRU EMA, and `actor_adapt`, followed by its
deterministic bounded mean action. Both checkpoint loads reported all expected
policy modules successfully.

## Main results

| Checkpoint | Success | Success rate | 95% Wilson CI | Failed | Episode length, mean +/- SD | `episode_cnt` |
|---|---:|---:|---:|---:|---:|---:|
| **Final** | **465/512** | **90.82%** | **88.01%-93.03%** | **47** | **606.79 +/- 48.54** | 526 |
| 8,400 (log-selected) | 461/512 | 90.04% | 87.14%-92.34% | 51 | 603.85 +/- 55.79 | 536 |

### Normalized reward-group metrics

Higher is better. Values are across-environment mean +/- SD.

| Checkpoint | Tracking return | Object return | Locomotion return | Feet return |
|---|---:|---:|---:|---:|
| **Final** | **0.080889 +/- 0.000912** | **0.067108 +/- 0.003064** | **0.019325 +/- 0.000255** | **0.018406 +/- 0.000162** |
| 8,400 | 0.080777 +/- 0.000904 | 0.066855 +/- 0.003377 | 0.019323 +/- 0.000281 | 0.018390 +/- 0.000180 |

The final-minus-8,400 relative changes are small: +0.14% tracking, +0.38%
object return, +0.01% locomotion return, and +0.08% feet return.

### Tracking and control quality

For scores and contact fraction, higher is better. For errors and negative
penalties, lower magnitude/closer to zero is better.

| Metric | Final | 8,400 | Better observed value |
|---|---:|---:|---|
| Object-position tracking | 0.774739 | 0.769950 | Final |
| Object-orientation tracking | 0.901018 | 0.901818 | 8,400 |
| Required-contact fraction | 0.991439 | 0.987873 | Final |
| End-effector contact reward | 1.679636 | 1.671005 | Final |
| Root-position error | 0.150089 | 0.151372 | Final |
| Root-orientation error | 0.078760 | 0.084673 | Final |
| Local body-position error | 0.057167 | 0.057115 | 8,400, marginally |
| Local body-orientation error | 0.177073 | 0.177903 | Final |
| Joint-position error | 0.086953 | 0.087632 | Final |
| Action-rate penalty | -0.027861 | -0.027850 | 8,400, marginally |
| Foot-slip penalty | -0.072652 | -0.073101 | Final |
| Impact-force penalty | -0.001671 | -0.001719 | Final |

The largest practically visible within-run quality change is root-orientation
error, which is about 7.0% lower at the final checkpoint. Most other differences
are much smaller than their across-environment standard deviations.

## Within-run conclusion

The log proxy correctly identifies a strong late-run checkpoint, but it does
not beat the true final under the deterministic deployment protocol. The final
adds four successes, has a longer and less variable first episode, improves all
four normalized reward groups, and is better on most pose/contact/control
metrics. `checkpoint_final.pt` is therefore the appropriate deployment choice
among these evaluated candidates.

This conclusion is descriptive for one 512-environment rollout per checkpoint,
two fresh candidate-evaluation runs in total. The raw YAML summaries do not retain paired
per-environment success outcomes, so a paired test cannot be recovered. The
confidence interval above treats the two success counts as independent
binomials and does not include simulator, checkpoint-selection, or multi-seed
training uncertainty. The final is preferred within this predefined two-
candidate procedure; the other 89 numeric checkpoints were not each evaluated
under the deployment protocol, so this does not prove a global optimum over
every saved checkpoint.

## Historical post-training reproducibility

The training process evaluated the in-memory final policy immediately after
saving. A fresh process reloaded the saved checkpoint and VecNorm state on the
following metrics:

| Metric | Historical in-memory final | Fresh checkpoint reload | Difference |
|---|---:|---:|---:|
| Success | 464/512 (90.63%) | 465/512 (90.82%) | +1 environment (+0.20 points) |
| Episode length | 606.93 +/- 47.98 | 606.79 +/- 48.54 | -0.14 steps |
| Tracking return | 0.0808894 | 0.0808893 | -0.0000001 |
| Object return | 0.0671181 | 0.0671079 | -0.0000102 |
| Locomotion return | 0.0193218 | 0.0193247 | +0.0000028 |
| Feet return | 0.0184120 | 0.0184061 | -0.0000059 |

The single-success change is exactly the one-environment resolution,
`1/512 = 0.1953125` percentage point, and is well within binomial and simulator
uncertainty. The aggregate reward differences are negligible. There is no sign
of a model-state or normalization reload regression. “Deterministic” describes
the Student action rule; Isaac/GPU execution is not guaranteed to be bitwise
deterministic, so the reproduction commands recreate the protocol rather than
guarantee an exact 465/512 count.

## Context against earlier reports

The following completion counts come from the fresh evaluations recorded in
`MODEL_PERFORMANCE_EVALUATION_2026-08-11.md`; they are included only to place
the new model on the existing local scale.

| Model/checkpoint | Deterministic success | Evaluation commit/source |
|---|---:|---|
| PPO checkpoint 42,300 | 486/512 (94.92%) | 2026-08-11 report |
| **Current FastSAC-BC-DAgger final** | **465/512 (90.82%)** | Current fresh evaluation |
| FastSAC 50%-offline final | 465/512 (90.82%) | 2026-08-11 report |
| PPO BC-DAgger final | 464/512 (90.63%) | 2026-08-11 report |
| FastSAC 0%-offline final | 463/512 (90.43%) | 2026-08-11 report |
| Current FastSAC-BC-DAgger 8,400 | 461/512 (90.04%) | Current fresh evaluation |

The current final is nominally 4.10 points below PPO 42,300; the approximate
unpaired-binomial 95% interval for PPO's advantage is +0.96 to +7.24 points.
The current final's group returns approximately match, and are marginally above,
the rounded values reported for the earlier 50%-offline FastSAC final. The
margins are tiny.

The earlier `MODEL_PERFORMANCE_EVALUATION_2026-08-13.md` covered a different,
undertrained 3,000-rollout FastSAC-BC-DAgger run that achieved 58/128 (45.31%).
The current run used four times as many environments, three times as many main
rollouts, a partial object/adaptation warm start, dedicated depth-perception
warmup, and Teacher replay for actor/perception learning. The later run no
longer exhibits the prior deployment failure, but the simultaneous changes
prevent attributing the gain to any one feature.

## Code-version and provenance caveat

W&B metadata records training commit
`7440cf3dc1c1745a546786cffd12e62a8739a32b`. The run was launched at 19:01,
while the then-working-tree FastSAC/TD3 action-support changes were committed as
`f6476f4524cea6d3ef5bd42204d4597e8400b2d0` at 19:07. The policy source snapshot
saved by W&B is byte-identical to the current `f6476f4` FastSAC policy file.
Thus the metadata commit alone understates the code actually loaded by the
training process. This snapshot covers that policy file only: the exact dirty
contents of inherited `td3_bc_dagger.py`, the entrypoint, config, helpers, and
environment at launch were not archived together, so the complete training
source tree cannot be reconstructed from the recorded commit plus this one
snapshot.

Evaluation ran at HEAD `f6476f4`. The workspace was dirty with pending TVKD
work: additive TVKD routing in `scripts/helpers.py`, a TVKD-only checkpoint-save
guard in `scripts/train.py`, and untracked TVKD policy/config/test files. The
baseline FastSAC evaluator branch used here is unchanged by those pending
edits, but the dirty-tree state is recorded for reproducibility.

Input SHA-256 hashes:

| Input | SHA-256 |
|---|---|
| `cfg.yaml` | `1c016f9662679003039d08abea0bbf75bda8b9de87b4ef160da8fcf145c62012` |
| PPO partial perception/Teacher source `checkpoint_final.pt` | `7b856b6206653850f7b05e1ffc12cd2f545ea390b0b065c30cfea785a58b2dd7` |
| Evaluated run `checkpoint_8400.pt` | `ab0d4add475ab2d9a22b6c4823ef283a8bdd01adbe4ff63e159bfc846d53cb9f` |
| Evaluated run `checkpoint_final.pt` | `0e2c763047b1458e23fb5888d6bda5354d00697aab64d04d1bd220afffa95946` |

## Reproduction commands

Run these commands from `/home/hcc/research/VAIC` and use the VAIC Python
environment explicitly. The run-ID-qualified, hash-pinned W&B config is used
instead of the movable `wandb/latest-run` symlink.

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files \
  --config-name=cfg \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files/checkpoint_final.pt \
  task.num_envs=512 \
  seed=0 \
  eval_render=false
```

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files \
  --config-name=cfg \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files/checkpoint_8400.pt \
  task.num_envs=512 \
  seed=0 \
  eval_render=false
```

## Raw evaluator outputs

- Final: [`scripts/eval/G1Skateboard/G1Skateboard-08-14_10-49.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-14_10-49.yaml)
- Checkpoint 8,400: [`scripts/eval/G1Skateboard/G1Skateboard-08-14_10-51.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-14_10-51.yaml)
- Fully composed config: [`outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files/cfg.yaml`](outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/files/cfg.yaml)
- Local W&B history used for checkpoint selection: [`outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/run-wdf9zfis.wandb`](outputs/2026-08-13/19-01-15-G1Skateboard-fastsac_bc_dagger/wandb/run-20260813_190116-wdf9zfis/run-wdf9zfis.wandb)

`scripts/policy_trajs.pt` is overwritten by each invocation of `scripts/eval.py`
and therefore contains only the most recently run evaluation, checkpoint 8,400.
