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
# entropy-temperature update, target Q, replay actor updates, and HOI's
# 8 updates/vector-step cadence all train from the beginning (after its
# default 10-step replay warm-up). Each environment step is inserted and
# trained before the teacher selects the next action, matching HOI ordering.
# VAIC observations/rewards/terminations and teacher->student distillation stay unchanged.
# The GPU replay trains FastSAC immediately and is not cleared at iteration 5100;
# paired H5 snapshots select only rows collected from iteration 5100 onward.
python scripts/train.py algo=fastsac_vel_train task=G1/vaic/skateboard_tea
# Same-stage resume restores model weights, active AdamW moments, counters, and
# the dedicated replay-sampling RNG state (the simulator itself starts reset).
# At/after the H5 gate it restores the exact matching export-eligible FIFO tail.
# A checkpoint before the gate has no H5 by design, so its replay starts empty.
# Old PPO-based fastsac_vel_train checkpoints are rejected by an algorithm marker.
# total_frames is a new additional training budget after the resume.
# For W&B exact replay resume, use the final checkpoint/H5 pair. Periodic H5
# snapshots are overwritten locally and are not uploaded to W&B.
python scripts/train.py algo=fastsac_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<fastsac_vel_train-wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<wandb-run-path>
python scripts/play.py algo=fastsac_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<fastsac_vel_train-wandb-run-path>
```

Student policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
# Train the student with 50% gated teacher H5 + 50% new online rollout data.
# The checkpoint transfers the FastSAC teacher/Q weights and the already
# distilled same-structure student actor; depth/adaptation + EMA keep VAIC logic.
python scripts/train.py algo=fastsac_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<fastsac_vel_train-wandb-run-path>
# A final fastsac_vel_finetune run also carries the paired offline teacher
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
