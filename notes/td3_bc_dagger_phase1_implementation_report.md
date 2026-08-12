# Distributional TD3 + Teacher-BC DAgger: Phase-1 implementation report

Date: 2026-08-11

Method/checkpoint marker: `distributional_td3_teacher_bc_v1`

## Scope and created files

Phase 1 creates only the six planned implementation/test files and this
permitted report:

- `scripts/TD3_bc_dagger.py`
- `cfg/TD3_bc_dagger.yaml`
- `active_adaptation/learning/ppo/td3_bc_dagger.py`
- `tests/test_td3_bc_dagger.py`
- `tests/test_td3_bc_dagger_entrypoint.py`
- `tests/test_td3_locked_interface.py`
- `notes/td3_bc_dagger_phase1_implementation_report.md`

No tracked file or protected environment, reward, observation, action,
baseline algorithm, or shared-training file was modified. The pre-existing
Phase-0 notes remain untracked and unchanged by this phase.

## Algorithm implemented

The Critic is the explicitly authorized interface-preserving distributional
TD3 option. For each next state, the deterministic target Actor produces an
action, target-policy smoothing is added and clipped in the existing Q-action
coordinates, and the result is constrained by the physical execution bounds
expressed in those coordinates. Both target C51 heads are evaluated. The head
with the lower expected value is selected independently for each row, but its
complete probability distribution is retained; no atom-wise minimum is used.
The Bellman-updated distribution is projected onto the fixed 501-atom
`[-20, 20]` support and detached. Both online Critics minimize categorical
cross entropy against that same projected target.

On every `policy_delay` Critic update, the Actor objective is

`eta_td3 * (-E[online Q1]) + lambda_bc * exact_teacher_bc`.

The BC term uses the authoritative inverse affine-tanh Teacher target and
valid-Teacher masked SmoothL1 loss. The two terms share one Actor
zero-grad/backward/optimizer step. Critic parameters are frozen during this
step while the Q1 action gradient remains live. The target Actor and both
target Critics are then Polyak-updated. No SAC sampling, log-probability,
entropy, alpha/temperature, or Actor-standard-deviation path participates in
training or evaluation.

Collection preserves categorical beta/SafeDAgger source selection. Optional
collector noise applies only to Student-selected rows and is disabled for
deterministic evaluation. Replay stores the exact final command issued to the
environment, with Teacher labels and source/noise metadata stored separately.

## Final verification

Baseline regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/hcc/anaconda3/envs/vaic/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_ppo_bc_dagger.py \
  tests/test_bc_dagger_entrypoint.py \
  tests/test_fastsac_timeout.py
```

Result: `147 passed, 3 warnings in 2.82s`.

Focused Phase-1 command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/hcc/anaconda3/envs/vaic/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_td3_locked_interface.py \
  tests/test_td3_bc_dagger.py \
  tests/test_td3_bc_dagger_entrypoint.py
```

Result: `88 passed, 1 warning in 4.77s`.

Syntax command:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python -m py_compile \
  scripts/TD3_bc_dagger.py \
  active_adaptation/learning/ppo/td3_bc_dagger.py
```

Result: exit status 0 with no output.

Hydra composition command:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  --cfg job task=G1/vaic/skateboard_stu
```

Result: exit status 0; it composes the audited skateboard task and the
`DistributionalTD3TeacherBC` target with 501 atoms, support `[-20, 20]`, and
`policy_delay=2`. The persistent Teacher export and online Teacher learning
ring both compose at 131,072 rows; the online Q batch/update cadence remains
512 rows and 128 updates per rollout.

Fail-fast stochastic-policy rejection command:

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/run-20260805_151350-7tsje71w/files/checkpoint_final.pt \
  td3_dagger_iterations=4 \
  +algo.sac_alpha=0.2 \
  wandb.mode=disabled
```

Result: expected exit status 1 before simulator creation, with
`ValueError: distributional TD3 Teacher-BC forbids stochastic-policy fields: ['sac_alpha']`.

Quality commands:

```bash
/home/hcc/anaconda3/bin/ruff check \
  active_adaptation/learning/ppo/td3_bc_dagger.py \
  scripts/TD3_bc_dagger.py \
  tests/test_td3_bc_dagger.py \
  tests/test_td3_bc_dagger_entrypoint.py \
  tests/test_td3_locked_interface.py

/home/hcc/anaconda3/bin/ruff format --check \
  active_adaptation/learning/ppo/td3_bc_dagger.py \
  scripts/TD3_bc_dagger.py \
  tests/test_td3_bc_dagger.py \
  tests/test_td3_bc_dagger_entrypoint.py \
  tests/test_td3_locked_interface.py

git diff --check
```

Results: `All checks passed!`, `5 files already formatted`, and exit status 0.

The latest real same-stage checkpoint also passes the strengthened read-only
preflight, restoring rollout count 4 and environment-step count 128. Resume
validation covers algorithm/version, module and optimizer state, counters,
independent RNGs, exact Actor/Critic/backend contracts, the 23-joint action
contract and Q-transform fingerprint, and the live VecNorm fingerprint.

The TorchRL optional-C++-extension warning appears in CPU-side tests. Phase 1
uses uniform replay and does not use the unavailable prioritized replay path.

## Minimal two-environment simulator smoke

Command actually run:

```bash
WANDB_MODE=disabled \
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/run-20260805_151350-7tsje71w/files/checkpoint_final.pt \
  td3_dagger_iterations=4 \
  task.num_envs=2 \
  algo.num_minibatches=2 \
  algo.dagger_control_mode=beta \
  algo.dagger_beta_start=0.5 \
  algo.dagger_beta_end=0.5 \
  algo.collector_exploration_noise_std=0.0 \
  algo.td3_learning_starts=1 \
  algo.q_batch_size=2 \
  algo.q_updates_per_rollout=1 \
  algo.dagger_batch_size=2 \
  algo.policy_delay=2 \
  algo.save_teacher_buffer=false \
  save_interval=-1 \
  wandb.mode=disabled
```

Result: exit status 0. The run completed four 32-step rollouts over two
environments (256 collected transitions), four Critic updates, two delayed
Actor/target updates, final checkpoint writing, and the shared 1,000-step
Student-only deterministic evaluation. The final checkpoint is
`/tmp/wandb/run--9j9i0vvy/files/checkpoint_final.pt`.

The fixed beta was a smoke-only choice to exercise both replay partitions;
the observed sampled source fractions were Student `0.546875` and Teacher
`0.453125`. Collector exploration was intentionally disabled, and its observed
noise norm was exactly `0.0`. The target-smoothing noise norm was
`0.9378323554992676`.

Observed pre-projection C51 support clipping fractions from the final update:

- left/below `-20`: `0.0`
- right/above `20`: `0.0009980039903894067` (about `0.0998004%`)

The support was not changed.

## Assumptions and blockers

- The effective task is `G1/vaic/skateboard_stu`; the repository's generic
  train default is not the audited interaction task.
- “Executed action” means the exact 23-D command handed to the environment.
  The environment's downstream delay, low-pass filter, randomized offset,
  per-joint scaling, and radian PD target remain unchanged and are not a
  different Critic action representation.
- Same-stage checkpoints intentionally omit the recent Student transition
  ring. With the production default `save_teacher_buffer=true`, the immutable
  Teacher H5 partition is paired with the checkpoint and the Student ring is
  refilled from new rollouts.
- The production persistent Teacher export is capped at 131,072 rows, matching
  the complete online Teacher learning ring. At the audited 525-D Actor,
  2,341-D Critic, and 23-D action dimensions this is about 2.81 GiB. It is not
  sampled by the online TD3/BC updates; the active all-transition and Teacher
  learning rings remain unchanged at 131,072 rows each.
- Five inherited PPO compatibility fields (three entropy-schedule and two
  standard-deviation initialization fields) remain in Hydra because the
  existing Actor/config topology is locked. The TD3 implementation never
  reads them, and any new SAC/log-probability/temperature field is rejected.
- `algo.num_minibatches=2` is only a two-environment smoke accommodation; the
  audited production task retains 512 environments and its normal batching.
- No unresolved implementation blocker remains. Adaptive beta, Q-filtering,
  imitability gating, and all later phases are intentionally out of scope.

## Recommended 3,000-rollout production command

Run this only when no other training process is using GPU 0. It uses the
explicit VAIC interpreter, explicit runtime environment, a timestamped unique
nohup log, and the bounded export capacity even though that value is also the
production default:

```bash
cd /home/hcc/research/VAIC
run_stamp="$(date +%Y%m%d_%H%M%S_%N)"
run_log="/home/hcc/research/VAIC/td3_bc_dagger_3000_${run_stamp}.log"

nohup /usr/bin/env \
  CUDA_VISIBLE_DEVICES=0 \
  WANDB_MODE=online \
  PYTHONUNBUFFERED=1 \
  /home/hcc/anaconda3/envs/vaic/bin/python \
  /home/hcc/research/VAIC/scripts/TD3_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/run-20260805_151350-7tsje71w/files/checkpoint_final.pt \
  td3_dagger_iterations=3000 \
  algo.dagger_control_mode=beta \
  algo.dagger_beta_start=1 \
  algo.dagger_beta_end=0 \
  algo.collector_exploration_noise_std=0.1 \
  algo.teacher_buffer_capacity=131072 \
  save_interval=100 \
  wandb.mode=online \
  >"${run_log}" 2>&1 &

echo "PID=$! LOG=${run_log}"
```

At the locked 512 environments and 32 collection steps this is 16,384
transitions per rollout and 49,152,000 transitions total. The categorical
Teacher probability is `max(1 - rollout / 1800, 0)`, so the final 1,200
rollouts are Student-only. Checkpoints are requested at rollout indices
100–2,900 plus the final checkpoint. A same-stage resume must retain the same
131,072-row export capacity.

## Suggested later short controlled training run

```bash
WANDB_MODE=online \
/home/hcc/anaconda3/envs/vaic/bin/python scripts/TD3_bc_dagger.py \
  task=G1/vaic/skateboard_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/15-13-46-G1Skateboard-ppo_vel/wandb/run-20260805_151350-7tsje71w/files/checkpoint_final.pt \
  td3_dagger_iterations=50 \
  algo.dagger_control_mode=beta \
  algo.dagger_beta_start=0.5 \
  algo.dagger_beta_end=0.5 \
  algo.collector_exploration_noise_std=0.0 \
  save_interval=10 \
  wandb.mode=online
```

This keeps the audited 512-environment task and production update sizes while
holding the collection mixture fixed and disabling collector noise for a
short, interpretable Phase-1 check.

## Git status

`git diff --stat` is empty because all Phase-1 additions are new, untracked
files and no tracked file changed. The exact final untracked list and line
counts are recorded after final verification in the handoff response.
