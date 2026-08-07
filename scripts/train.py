import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

import active_adaptation as aa
from isaaclab.app import AppLauncher
# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict

# Local import that works both for ``python scripts/train.py`` and package
# imports used by dedicated entrypoints/tests.
try:
    from .helpers import (
        EpisodeStats,
        apply_fastsac_buffer_steps,
        apply_teacher_replay_buffer_path_alias,
        copy_frozen_teacher_replay,
        evaluate,
        make_env_policy,
        teacher_replay_storage_dir,
    )
except ImportError:
    from helpers import (
        EpisodeStats,
        apply_fastsac_buffer_steps,
        apply_teacher_replay_buffer_path_alias,
        copy_frozen_teacher_replay,
        evaluate,
        make_env_policy,
        teacher_replay_storage_dir,
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def maybe_upload_teacher_replay(
    run,
    cfg: DictConfig,
    policy,
    replay_snapshot_path,
    *,
    artifact: bool,
):
    """Upload a final H5 only through an explicit W&B opt-in."""
    if not artifact:
        return None
    replay_path = replay_snapshot_path
    if replay_path is None and hasattr(policy, "get_offline_replay_path"):
        replay_path = policy.get_offline_replay_path()
    if replay_path is None:
        return None
    if not bool(cfg.wandb.get("upload_teacher_replay", False)):
        logging.info(
            "Keeping teacher replay local (W&B H5 upload disabled): %s",
            replay_path,
        )
        return None
    run.save(
        replay_path,
        policy="now",
        base_path=os.path.dirname(replay_path),
    )
    return replay_path


def make_wandb_settings(cfg: DictConfig):
    """Prevent W&B's end-of-run directory scan from finding a local H5."""
    replay_filename = cfg.algo.get("teacher_buffer_filename", None)
    upload_enabled = bool(
        cfg.wandb.get("upload_teacher_replay", False)
    )
    ignore_globs = ()
    if replay_filename is not None and not upload_enabled:
        ignore_globs = (str(replay_filename),)
    return wandb.Settings(ignore_globs=ignore_globs)


def run_training(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if (
        cfg.get("bc_dagger_checkpoint", None) is not None
        and not bool(cfg.get("_bc_dagger_model_only_resume", False))
    ):
        raise ValueError(
            "bc_dagger_checkpoint must be validated by scripts/bc_dagger.py; "
            "use that dedicated entrypoint for model-only DAgger resume"
        )
    apply_teacher_replay_buffer_path_alias(cfg)
    apply_fastsac_buffer_steps(cfg)
    freeze_bc_dagger_teacher_replay = bool(
        cfg.get("_bc_dagger_model_only_resume", False)
    )

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
        settings=make_wandb_settings(cfg),
    )
    os.makedirs(run.dir, exist_ok=True)
    hydra_output_dir = (
        HydraConfig.get().runtime.output_dir
        if HydraConfig.initialized()
        else None
    )
    replay_storage_dir = teacher_replay_storage_dir(
        run.dir, hydra_output_dir
    )
    replay_copy_source = cfg.get(
        "_bc_dagger_teacher_replay_copy_source", None
    )
    if (
        freeze_bc_dagger_teacher_replay
        and replay_copy_source is not None
        and aa.is_main_process()
    ):
        print(
            "Copying immutable teacher replay into the new output "
            f"({os.path.getsize(replay_copy_source) / (1024**3):.2f} GiB)..."
        )
        copied_replay = copy_frozen_teacher_replay(
            replay_copy_source,
            replay_storage_dir,
            cfg.algo.get(
                "teacher_buffer_filename", "teacher_replay_buffer.h5"
            ),
        )
        cfg._bc_dagger_teacher_replay_copy_path = copied_replay
        print(f"Teacher replay copy ready: {copied_replay}")
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # Recent W&B offline versions may leave ``run.name`` unset until the user
    # assigns it; the stable run id is a safe local fallback.
    run_idx = (run.name or run.id).split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    # Materialize a requested 20+ GiB frozen replay before launching Isaac so
    # the simulator and GPU are not held idle during filesystem I/O.
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    env, policy, vecnorm = make_env_policy(cfg, configure_replay=True)

    # Replay-producing algorithms need their live FIFO on every training rank.
    # H5 itself is written only by the rank-0 checkpoint hook below.  Capability
    # checks are intentional: PPO-BC/DAgger is a finetune phase but also trains
    # Q from replay and exports teacher-executed transitions.
    requires_training_replay = (
        hasattr(policy, "requires_training_replay")
        and policy.requires_training_replay()
    )
    if (
        not freeze_bc_dagger_teacher_replay
        and hasattr(policy, "configure_teacher_replay")
        and (
            requires_training_replay
            or (
                aa.is_main_process()
                and cfg.algo.get("save_teacher_buffer", False)
            )
        )
    ):
        replay_name = cfg.algo.teacher_buffer_filename
        replay_path = os.path.join(replay_storage_dir, replay_name)
        restore_path = None
        if cfg.checkpoint_path is not None:
            restore_path = cfg.algo.get("teacher_buffer_path")
        elif cfg.algo.get("teacher_buffer_path") is not None:
            raise ValueError(
                "algo.teacher_buffer_path requires checkpoint_path for a "
                "same-stage teacher replay resume."
            )
        policy.configure_teacher_replay(replay_path, restore_path=restore_path)
        if cfg.algo.get("phase") == "train" and not cfg.algo.get(
            "save_teacher_buffer", False
        ):
            logging.info(
                "Stage-1 FastSAC uses a device-only compact learning replay; "
                "no teacher H5 will be written."
            )
        else:
            logging.info(f"Teacher replay buffer: {replay_path}")

    import inspect
    import shutil
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, artifact: bool=False):
        replay_snapshot_path = None
        if (
            not freeze_bc_dagger_teacher_replay
            and hasattr(policy, "snapshot_teacher_replay")
        ):
            replay_snapshot_path = policy.snapshot_teacher_replay(
                env.current_iter, checkpoint_name
            )
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if "vecnorm" in locals():
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if artifact:
            model_artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}",
                type="model"
            )
            model_artifact.add_file(ckpt_path)
            run.log_artifact(model_artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        maybe_upload_teacher_replay(
            run,
            cfg,
            policy,
            replay_snapshot_path,
            artifact=artifact,
        )
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")
    interleaved_updates = (
        hasattr(policy, "uses_interleaved_updates")
        and policy.uses_interleaved_updates()
    )

    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        # This warm-up step fixes rollout buffer shapes, but it also advances
        # the physical simulator.  Keep its returned state as the first real
        # carry so policy observations cannot lag one control step behind the
        # robot/reference state.
        tmp_td, carry = env.step_and_maybe_reset(tmp_carry.clone(False))
        if interleaved_updates:
            # Stage-1 replay has already consumed one-step raw aliases. Keep
            # them only in carry; retaining them in the N x T diagnostics
            # rollout would duplicate a large fraction of observation memory.
            tmp_td = tmp_td.exclude(
                "_fastsac_raw", ("next", "_fastsac_raw")
            )
        tmp_td["next"] = tmp_td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)

    N = env.num_envs
    T = cfg.algo.train_every
    device = env.device

    data_buf = TensorDict({}, batch_size=[N, T], device=device)
    for key, value in tmp_td.items(include_nested=True, leaves_only=True):
        shape_tail = value.shape[1:]
        buf = torch.empty((N, T, *shape_tail), dtype=value.dtype, device=device)
        data_buf.set(key, buf)
    logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    env_frames = 0
    start_iter = env.current_iter
    for i in progress:
        if hasattr(policy, "begin_transition_collection"):
            policy.begin_transition_collection()
        rollout_start = time.perf_counter()
        interleaved_training_time = 0.0
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            env.set_progress(start_iter + i)
            for step in range(cfg.algo.train_every):
                carry = rollout_policy(carry)
                actuator_context = None
                if hasattr(policy, "capture_q_actuator_context"):
                    # Delay/alpha may be resampled by step_and_maybe_reset for
                    # completed rows. Snapshot the Q-only context while it still
                    # belongs to the action and transition being collected.
                    actuator_context = policy.capture_q_actuator_context()
                if (
                    not interleaved_updates
                    and hasattr(policy, "record_rollout_q_actuator_context")
                ):
                    policy.record_rollout_q_actuator_context(actuator_context)
                td, carry = env.step_and_maybe_reset(carry)
                if interleaved_updates:
                    update_start = time.perf_counter()
                    # Replay tensors are ordinary device tensors, so gradients
                    # can run here even though environment collection is under
                    # inference mode.
                    with torch.inference_mode(False), torch.enable_grad():
                        policy.collect_environment_step(
                            td, carry, actuator_context
                        )
                    interleaved_training_time += (
                        time.perf_counter() - update_start
                    )
                elif hasattr(policy, "capture_truncation_final_observations"):
                    # The first return still owns the transformed pre-reset
                    # timeout observation; ``carry`` has already reset that
                    # row. The environment also labels command completion as
                    # truncated, but FastSAC treats it as a non-bootstrapping
                    # task terminal and therefore needs no final-state capture.
                    policy.capture_truncation_final_observations(td, step)
                if interleaved_updates:
                    td = td.exclude(
                        "_fastsac_raw", ("next", "_fastsac_raw")
                    )
                td["next"] = td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)
                data_buf[:, step] = td
            if (
                not interleaved_updates
                and hasattr(policy, "capture_rollout_final_observation")
            ):
                policy.capture_rollout_final_observation(carry)
            requires_value_bootstrap = (
                not hasattr(policy, "requires_value_bootstrap")
                or policy.requires_value_bootstrap()
            )
            if requires_value_bootstrap:
                policy.critic(data_buf)
                values = data_buf["state_value"]
                data_buf["next", "state_value"] = torch.where(
                    data_buf["next", "done"],
                    values, # a walkaround to avoid storing the next states
                    torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
                )
        rollout_time = max(
            time.perf_counter() - rollout_start - interleaved_training_time,
            1e-9,
        )

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data_buf))
        training_time = (
            time.perf_counter() - training_start + interleaved_training_time
        )
        info.update(env.extra)
        info.update(env.stats_ema)

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = env_frames
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = training_time

        checkpoint_index = i
        if (
            bool(cfg.get("_bc_dagger_model_only_resume", False))
            and hasattr(policy, "dagger_rollout_count")
        ):
            # Preserve the original zero-based filename convention while using
            # the cumulative DAgger stage counter in a resumed W&B run.
            checkpoint_index = max(int(policy.dagger_rollout_count) - 1, 0)
        if should_save(checkpoint_index):
            save(policy, f"checkpoint_{checkpoint_index}")

        if aa.is_main_process():
            # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
            run.log(info)

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats, policy_trajs = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    run.log(info)

    wandb.finish()
    os._exit(0)
    env.close()
    simulation_app.close()

    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"
    
    return run_path


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    return run_training(cfg)



if __name__ == "__main__":
    main()
