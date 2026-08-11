# G1 Skateboard model performance evaluation

Evaluation date: 2026-08-11 (Asia/Seoul)

## Outcome

Among the seven checkpoints tested with the current repository evaluator, the strongest completion result is **PPO checkpoint 42,300: 486/512 successful environments (94.92%)**. The 50%-offline FastSAC final is second at **465/512 (90.82%)**, followed closely by PPO BC-DAgger final at **464/512 (90.63%)** and the new 0%-offline FastSAC final at **463/512 (90.43%)**.

The 0%-offline FastSAC final has the highest normalized tracking, object-return, and locomotion-return metrics, although its margins over the 50%-offline final are very small. It also has the lowest local body and joint pose errors. The BC-DAgger student has the best feet return. The practical choice therefore depends on the deployment objective:

- Use **PPO checkpoint 42,300** when motion-completion success is primary.
- Use **0%-offline FastSAC final** when the tracking/object/locomotion group returns and local body/joint pose fidelity are primary.
- Use **50%-offline FastSAC final** when its two additional successful environments and slightly smoother actions are preferred; the completion difference from 0%-offline is too small to establish a material advantage.
- Use **PPO BC-DAgger final** as a strong directly evaluated imitation-trained student baseline; its completion rate lies between the two FastSAC finals.
- Do **not** use PPO final with the current repository: it regressed to 75.39% success and is substantially worse than PPO checkpoint 42,300.

## Evaluated checkpoints

| Run | Candidate | Checkpoint | Reason |
|---|---|---|---|
| FastSAC, 50% offline replay | Last | `checkpoint_final.pt` | True final state after 3,500 training iterations; newer than numeric checkpoint 3,400 |
| FastSAC, 50% offline replay | Log-selected | `checkpoint_900.pt` | Highest trailing-save-interval mean of logged training success |
| FastSAC, 0% offline replay | Last | `checkpoint_final.pt` | True final state after 3,500 training iterations; newer than numeric checkpoint 3,400 |
| FastSAC, 0% offline replay | Log-selected | `checkpoint_600.pt` | Highest trailing-save-interval mean among saved numeric checkpoints |
| PPO BC-DAgger | Last, student-only | `checkpoint_final.pt` | Requested completed staged checkpoint after 7,000 rollouts |
| PPO | Last | `checkpoint_final.pt` | True final state after 48,828 training iterations; newer than numeric checkpoint 48,600 |
| PPO | Log-selected | `checkpoint_42300.pt` | Highest trailing-save-interval mean of logged training success |

All four run lineages use the G1 skateboard student task. The evaluated configurations are `fastsac_vel_finetune`, `ppo_bc_dagger`, and `ppo_vel_finetune` on `G1/vaic/skateboard_stu`.

The two FastSAC runs share the same BC-DAgger source checkpoint, seed, 3,500-iteration schedule, and every other composed training setting. Their configs differ only in `teacher_buffer_ratio` (`0.5` versus `0.0`) and `teacher_buffer_capacity` (`1,048,576` versus `1`). Capacity 1 only limits the otherwise unused offline replay load to one row when the sampling ratio is zero. Thus this is a clean single-seed Stage-2 replay-mixing ablation, not a claim that the 0%-offline model is teacher-free: both runs inherit the same teacher-trained BC actor, perception stack, and pretrained Q networks.

The BC-DAgger checkpoint identifies itself as `vaic_ppo_bc_dagger_student_sac_critic_v3` with a completed 7,000-rollout staged schedule: 6,000 joint warmup rollouts, 200 final-perception rollouts, 200 final-actor rollouts, and 600 replay/Q-calibration rollouts.

## Checkpoint-selection method for FastSAC and PPO

The training history contains only one held-out `eval/*` record per run, produced after training. Earlier checkpoints therefore cannot be selected from historical evaluation results. I used the canonical episodic training metric `train/stats/success` as a proxy.

For each saved numeric checkpoint at iteration `k`, I averaged all logged success samples in the immediately preceding save interval, `(k - save_interval, k]`. Final checkpoints were evaluated separately and excluded from this ranking. This smoothing avoids choosing an unsaved, single-window spike.

The BC-DAgger final was explicitly requested and was not selected from its mixed-control training log. In particular, its final calibration phase executes approximately 50% teacher actions during training, so those rollout statistics are not a student-only checkpoint ranking.

| Run | Save interval | Selected checkpoint | Samples | Mean training success | Supporting detail |
|---|---:|---:|---:|---:|---|
| FastSAC, 50% offline | 100 | 900 | 3 | **0.782319** | Steps 832/864/896: 0.770755, 0.784421, 0.791781 |
| FastSAC, 0% offline | 100 | 600 | 3 | **0.782177** | Steps 512/544/576: 0.771930, 0.787697, 0.786903 |
| PPO | 300 | 42,300 | 9 | **0.807820** | Steps 42,016-42,272; range 0.792271-0.823434 |

Audit note: the 50%-offline FastSAC raw maximum was 0.792187 at unsaved step 3,168, but the exact step-3,200 sample fell to 0.756471 and checkpoint 3,200's trailing mean was only 0.773623. The 0%-offline raw maximum was 0.788762 at unsaved step 3,040, while checkpoint 3,100's trailing mean was 0.770536. The 0%-offline final tail window averaged 0.784030, slightly above checkpoint 600, but final checkpoints are excluded from numeric-save ranking and evaluated separately. PPO's raw maximum was 0.826944 at unsaved step 43,552; checkpoint 43,500's preceding-window mean was 0.780956. None of the isolated spikes was used as the best-checkpoint criterion.

## Evaluation protocol

- Repository evaluator: `scripts/eval.py` and `scripts/helpers.py:evaluate`
- Current evaluation commit: `0b1dbfd9a36d9ab76f5d96632e8ed82c2c999414`
- Saved, fully composed config for each run
- Deterministic policy mode (`ExplorationType.MODE`)
- Seed: 0
- Parallel environments: 512
- Horizon: 1,000 simulation steps
- Task: G1 skateboard student policy with the run's domain randomization
- Vector normalization: checkpoint statistics in evaluation mode
- Rendering: disabled

Success and episode length are calculated from each environment's first episode. Other evaluator metrics are divided by that episode's length before their across-environment mean and standard deviation are calculated. `episode_cnt` is not the success denominator; it counts every termination during the complete rollout horizon.

The helper assumes each environment produces at least one `done` within the horizon rather than asserting it. This task's 1,000-step episode time limit is intended to ensure that condition.

For PPO BC-DAgger, "student-only" means the complete deployable student inference stack: temporal-depth GRU EMA, object-adaptation EMA, adaptation GRU EMA, and `actor_adapt`. Evaluation uses its deterministic mean action and clips it to the checkpoint's configured range. The privileged PPO teacher, SafeDAgger/beta action switch, and Q critic do not participate in action selection.

Both FastSAC evaluations likewise use the deterministic deployable student mean. “50% offline” and “0% offline” describe the Stage-2 learning minibatch mixture, not evaluation-time control. The 0%-offline run logged zero offline replay draws; all of its Stage-2 update rows came from new student rollouts, while its actor and Q networks still warm-started from the same BC-DAgger checkpoint.

## Main results

| Model/checkpoint | Success | Success rate | 95% Wilson CI | Episode length, mean +/- SD |
|---|---:|---:|---:|---:|
| FastSAC 50%-offline final | 465/512 | 90.82% | 88.01%-93.03% | 606.45 +/- 51.50 |
| FastSAC 50%-offline 900 (log-selected) | 460/512 | 89.84% | 86.92%-92.17% | 604.61 +/- 53.71 |
| FastSAC 0%-offline final | 463/512 | 90.43% | 87.57%-92.69% | 606.14 +/- 50.64 |
| FastSAC 0%-offline 600 (log-selected) | 459/512 | 89.65% | 86.71%-92.00% | 605.11 +/- 52.74 |
| PPO BC-DAgger final (student-only) | 464/512 | 90.63% | 87.79%-92.86% | 605.30 +/- 54.90 |
| PPO final | 386/512 | 75.39% | 71.48%-78.92% | 543.94 +/- 154.72 |
| **PPO 42,300 (log-selected)** | **486/512** | **94.92%** | **92.66%-96.51%** | **610.86 +/- 53.11** |

The nominal unpaired-binomial difference between PPO 42,300 and the completion-leading 50%-offline FastSAC final is **+4.10 percentage points**, with an approximate 95% interval of **+0.96 to +7.24 points**. This is a model-based sampling interval conditional on the current evaluation setup; it does not represent checkpoint-selection, simulator, or multi-seed uncertainty.

### Normalized reward-group metrics

Higher is better. Values are across-environment mean +/- SD.

| Model/checkpoint | Tracking return | Object return | Locomotion return | Feet return |
|---|---:|---:|---:|---:|
| FastSAC 50%-offline final | 0.08065 +/- 0.00104 | 0.06693 +/- 0.00337 | 0.01927 +/- 0.00027 | 0.01840 +/- 0.00018 |
| FastSAC 50%-offline 900 | 0.08053 +/- 0.00097 | 0.06674 +/- 0.00340 | 0.01926 +/- 0.00027 | 0.01840 +/- 0.00018 |
| **FastSAC 0%-offline final** | **0.08065 +/- 0.00099** | **0.06704 +/- 0.00346** | **0.01927 +/- 0.00027** | 0.01839 +/- 0.00018 |
| FastSAC 0%-offline 600 | 0.08048 +/- 0.00099 | 0.06668 +/- 0.00341 | 0.01924 +/- 0.00034 | 0.01840 +/- 0.00018 |
| PPO BC-DAgger final | 0.08037 +/- 0.00118 | 0.06647 +/- 0.00340 | 0.01923 +/- 0.00043 | **0.01843 +/- 0.00018** |
| PPO final | 0.07489 +/- 0.00399 | 0.06107 +/- 0.01134 | 0.01760 +/- 0.00292 | 0.01819 +/- 0.00039 |
| PPO 42,300 | 0.07756 +/- 0.00131 | 0.06641 +/- 0.00419 | 0.01885 +/- 0.00063 | 0.01838 +/- 0.00024 |

Compared with completion-leading PPO 42,300, the 0%-offline FastSAC final is higher by approximately 4.0% in tracking return, 1.0% in object return, and 2.2% in locomotion return. BC-DAgger is close to both FastSAC finals on all four reward groups but does not exceed them except on feet return.

### Tracking-quality comparison of the recommended checkpoints

For tracking scores, higher is better; for errors and negative penalties, lower magnitude/closer to zero is better.

| Metric | FastSAC 50% final | FastSAC 0% final | BC-DAgger student | PPO 42,300 | Better |
|---|---:|---:|---:|---:|---|
| Object-position tracking | 0.77361 | 0.77732 | 0.76956 | 0.77833 | PPO 42,300 |
| Object-orientation tracking | 0.90298 | 0.90418 | 0.90307 | 0.90724 | PPO 42,300 |
| Root-position error | 0.15026 | 0.14848 | 0.15273 | 0.14043 | PPO 42,300 |
| Root-orientation error | 0.08645 | 0.08915 | 0.08617 | 0.14032 | BC-DAgger student |
| Local body-position error | 0.05787 | 0.05733 | 0.06063 | 0.06324 | FastSAC 0% final |
| Local body-orientation error | 0.17881 | 0.17829 | 0.18356 | 0.22982 | FastSAC 0% final |
| Joint-position error | 0.08825 | 0.08768 | 0.08971 | 0.12483 | FastSAC 0% final |
| Action-rate penalty | -0.02972 | -0.03001 | -0.03058 | -0.04409 | FastSAC 50% final |
| Foot-slip penalty | -0.07286 | -0.07330 | -0.07070 | -0.06845 | PPO 42,300 |
| Impact-force penalty | -0.00193 | -0.00181 | -0.00211 | -0.00469 | FastSAC 0% final |

## Within-run conclusions

### FastSAC, 50% offline replay

The 50%-offline final is modestly better than its log-selected checkpoint 900: 465 versus 460 successes, a 0.98-point gain. The approximate success-difference interval includes zero, and their reward metrics are nearly identical, so this is not strong evidence of a material difference. The final checkpoint is the simpler deployment choice.

### FastSAC, 0% offline replay

The 0%-offline final is also better than its log-selected numeric checkpoint: 463 versus 459 successes, a 0.78-point gain. It has higher tracking, object, and locomotion returns and lower local body/joint errors. Checkpoint 600 has a slightly higher feet return and the lowest root-orientation error among all seven evaluated checkpoints, 0.085762 versus 0.085777 for 50%-offline checkpoint 900. Overall, the 0%-offline final is the better checkpoint from this run.

### Stage-2 offline-replay ablation

The two FastSAC finals differ by only two successful environments: 465/512 with 50% offline replay and 463/512 with 0% offline replay, a nominal **+0.39-point** advantage for the 50% mixture. The approximate unpaired-binomial 95% interval for that difference is **-3.18 to +3.96 points**, so these evaluations do not establish a material completion-rate difference.

The 0%-offline final is marginally higher on tracking, object, and locomotion returns and better on most local pose errors, while the 50%-offline final has slightly smoother actions. These differences are descriptive results from one training seed per condition. The same 0%-offline final differed by seven successes between its built-in and fresh evaluations, which is larger than the two-success cross-run gap. The YAML summaries do not retain paired per-environment outcomes, and the experiment does not measure multi-seed training or simulator variance, so it should not be treated as a definitive causal estimate of replay mixing.

### PPO BC-DAgger student

The requested final checkpoint reaches 464/512 successes (90.63%), between the two FastSAC finals. Its normalized returns and pose errors are likewise close to FastSAC, while it has the best feet return and the best root-orientation error among the four recommended deployment candidates. The 0%-offline FastSAC checkpoint 600 has a slightly lower absolute root-orientation error across all seven evaluations. This is a full student-inference result, with no teacher action selection. Only the final BC-DAgger checkpoint was evaluated, so this report does not claim it is the best checkpoint within that run.

Its PPO source teacher is a separate `outputs/15-13-46-G1Skateboard-ppo_vel/.../checkpoint_6000.pt` lineage; PPO checkpoint 42,300 in this report is not that source teacher. The difference between them must not be interpreted as a causal BC-DAgger improvement or regression.

### PPO

PPO checkpoint 42,300 clearly dominates PPO final under the current evaluator: 486 versus 386 successes, a **19.53-point gain**, with markedly better episode-length stability, reward returns, and pose errors. The final PPO checkpoint should not be selected for current-repository deployment.

## Code-version caveat

Both FastSAC runs were trained at the current commit. PPO BC-DAgger was trained at `6bac444f6e4002c8dcb201d74d805ca731392bbc`; that commit differs from the current evaluation commit only in `DGX_SPARK_SETUP.md`, so its policy, task, environment, and evaluator code match. The ordinary PPO comparison run was trained at older commit `44bcb4f26d831e59479eb199570998239f34d431`, before current environment reset and command/contact-alignment changes.

This matters in the final-checkpoint reproducibility check:

| Final checkpoint | Historical post-training success | Fresh current-repo success | Difference |
|---|---:|---:|---:|
| FastSAC 50%-offline final | 90.63% | 90.82% | +0.20 points |
| FastSAC 0%-offline final | 89.06% | 90.43% | +1.37 points |
| PPO BC-DAgger final, student-only | 90.82% | 90.63% | -0.20 points |
| PPO final | 96.68% | 75.39% | -21.29 points |

All checkpoint loads completed their expected policy modules successfully, ruling out an obvious missing-state failure. The ordinary PPO gap strongly suggests evaluation/environment version sensitivity, but this single rerun does not prove that code changes are its sole cause. The fresh table is a valid comparison for compatibility with the **current** repository. Apart from the controlled comparison between the two FastSAC runs, it is not a clean algorithm-only experiment: ordinary PPO was trained against older code, and the run lineages differ in initialization and training protocol. For a publication-quality comparison, retrain all methods on the same commit and evaluate several seeds.

## Reproduction commands

Use the VAIC environment explicitly; the system Python does not contain Torch. The following commands reproduce the two FastSAC finals and PPO 42,300; replacing only a FastSAC checkpoint path reproduces its other candidate.

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/2026-08-11/03-58-50-G1Skateboard-fastsac_vel/.hydra \
  --config-name=config \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-11/03-58-50-G1Skateboard-fastsac_vel/wandb/latest-run/files/checkpoint_final.pt \
  task.num_envs=512
```

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/2026-08-11/10-41-13-G1Skateboard-fastsac_vel/.hydra \
  --config-name=config \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-11/10-41-13-G1Skateboard-fastsac_vel/wandb/latest-run/files/checkpoint_final.pt \
  task.num_envs=512
```

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/10-51-55-G1Skateboard-ppo_vel/.hydra \
  --config-name=config \
  checkpoint_path=/home/hcc/research/VAIC/outputs/10-51-55-G1Skateboard-ppo_vel/wandb/latest-run/files/checkpoint_42300.pt \
  task.num_envs=512
```

PPO BC-DAgger must use its finalized runtime `wandb/.../files/cfg.yaml`; the original `.hydra/config.yaml` predates staging-schedule injection and is checkpoint-incompatible.

```bash
/home/hcc/anaconda3/envs/vaic/bin/python scripts/eval.py \
  --config-path=/home/hcc/research/VAIC/outputs/2026-08-10/18-09-58-G1Skateboard-ppo_bc_dagger/wandb/latest-run/files \
  --config-name=cfg \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-08-10/18-09-58-G1Skateboard-ppo_bc_dagger/wandb/latest-run/files/checkpoint_final.pt \
  task.num_envs=512
```

## Raw evaluator outputs

- FastSAC 50%-offline final: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_10-01.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_10-01.yaml)
- FastSAC 50%-offline checkpoint 900: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_10-03.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_10-03.yaml)
- FastSAC 0%-offline final: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_13-49.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_13-49.yaml)
- FastSAC 0%-offline checkpoint 600: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_13-51.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_13-51.yaml)
- PPO final: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_10-05.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_10-05.yaml)
- PPO checkpoint 42,300: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_10-07.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_10-07.yaml)
- PPO BC-DAgger final, student-only: [`scripts/eval/G1Skateboard/G1Skateboard-08-11_10-18.yaml`](scripts/eval/G1Skateboard/G1Skateboard-08-11_10-18.yaml)

The evaluator also overwrites `scripts/policy_trajs.pt` on every run; that file contains only the most recently run evaluation's policy trajectory selection, 0%-offline FastSAC checkpoint 600 in this evaluation set.
