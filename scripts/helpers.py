import torch
import torch.nn as nn
import hydra
import numpy as np
import copy
import random
import time
import wandb
import logging
import os
import shutil
import stat
import tempfile
import datetime

from collections.abc import Mapping
from dataclasses import fields
from typing import Sequence, List, Tuple, TYPE_CHECKING
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.envs.transforms import VecNorm

from termcolor import colored
from collections import OrderedDict
import imageio
from omegaconf import OmegaConf, DictConfig, open_dict
import active_adaptation.learning
from active_adaptation.utils.wandb import parse_checkpoint_path
import active_adaptation
if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


class ObsNorm(ModBase):
    def __init__(self, in_keys, out_keys, locs, scales):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        
        self.loc = nn.ParameterDict({k: nn.Parameter(locs[k]) for k in in_keys})
        self.scale = nn.ParameterDict({k: nn.Parameter(scales[k]) for k in out_keys})
        self.requires_grad_(False)

    def forward(self, tensordict: TensorDictBase):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            obs = tensordict.get(in_key, None)
            if obs is not None:
                loc = self.loc[in_key]
                scale = self.scale[out_key]
                tensordict.set(out_key, (obs - loc) / scale)
        return tensordict
    
    @classmethod
    def from_vecnorm(cls, vecnorm: VecNorm, keys):
        in_keys = []
        out_keys = []
        for in_key, out_key in zip(vecnorm.in_keys, vecnorm.out_keys):
            if in_key in keys:
                in_keys.append(in_key)
                out_keys.append(out_key)
        return cls(
            in_keys=in_keys,
            out_keys=out_keys,
            locs=vecnorm.loc,
            scales=vecnorm.scale
        )

class ObsOODDetector(ModBase):
    def __init__(self, in_keys, sigma=5.0, ref_tensordict=None):
        super().__init__()
        if ref_tensordict is not None:
            in_keys = [k for k in in_keys if ref_tensordict.get(k, None) is not None and ref_tensordict[k].dtype != torch.bool]
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [("next", k) for k in in_keys]
        self.sigma = sigma

    def forward(self, tensordict: TensorDictBase):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (obs.abs() > self.sigma).float().mean(dim=tuple(range(1, obs.ndim)))
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)
    
    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()


def apply_teacher_replay_buffer_path_alias(cfg: DictConfig):
    """Resolve the user-facing Stage-2 offline H5 path.

    Hydra changes into its run directory before ``main`` executes. Resolve an
    explicit CLI path against Hydra's original working directory so a relative
    path entered from the repository root still points to the intended file.
    Stage-1's compact learning FIFO is ephemeral and cannot be supplied here.
    """
    explicit_path = cfg.get("teacher_replay_buffer_path", None)
    supports_teacher_replay = "teacher_buffer_path" in cfg.algo
    internal_path = (
        cfg.algo.get("teacher_buffer_path", None)
        if supports_teacher_replay
        else None
    )
    if explicit_path is None and internal_path is None:
        return None
    if not supports_teacher_replay:
        raise ValueError(
            "teacher_replay_buffer_path is only supported by an algorithm with "
            "teacher replay support."
        )
    if cfg.algo.get("phase", None) == "train":
        raise ValueError(
            "Stage-1 fastsac_vel_train uses an ephemeral compact learning FIFO "
            "and does not accept teacher_replay_buffer_path. Supply the H5 only "
            "to fastsac_vel_finetune after collecting it in a separate process."
        )
    if cfg.get("checkpoint_path", None) is None:
        raise ValueError(
            "teacher_replay_buffer_path (or algo.teacher_buffer_path) must be "
            "used with checkpoint_path so policy weights and replay provenance "
            "come from the same saved training state."
        )

    def absolute_path(path):
        path = os.path.expanduser(os.fspath(path))
        return os.path.realpath(hydra.utils.to_absolute_path(path))

    resolved_explicit = (
        absolute_path(explicit_path) if explicit_path is not None else None
    )
    resolved_internal = (
        absolute_path(internal_path) if internal_path is not None else None
    )
    selected_path = resolved_explicit or resolved_internal
    if not os.path.isfile(selected_path):
        raise FileNotFoundError(
            f"Teacher replay buffer does not exist or is not a file: {selected_path}"
        )

    if (
        resolved_explicit is not None
        and resolved_internal is not None
        and resolved_explicit != resolved_internal
    ):
        raise ValueError(
            "Conflicting teacher replay paths: teacher_replay_buffer_path="
            f"{resolved_explicit!r}, algo.teacher_buffer_path="
            f"{resolved_internal!r}."
        )

    if explicit_path is not None:
        cfg.teacher_replay_buffer_path = selected_path
    cfg.algo.teacher_buffer_path = selected_path
    return selected_path


def copy_frozen_teacher_replay(source_path, run_dir, filename):
    """Atomically make an independent, read-only replay copy in a new run.

    The source is deliberately never opened for writing.  A source stat check
    before and after the copy rejects a concurrently changing H5 instead of
    publishing a torn offline dataset.
    """
    source_path = os.path.realpath(os.path.abspath(os.fspath(source_path)))
    run_dir = os.path.realpath(os.path.abspath(os.fspath(run_dir)))
    filename = os.fspath(filename)
    if (
        not filename
        or filename in (".", "..")
        or os.path.basename(filename) != filename
    ):
        raise ValueError("Teacher replay copy requires a file basename")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"Frozen teacher replay source does not exist: {source_path}"
        )

    os.makedirs(run_dir, exist_ok=True)
    destination = os.path.join(run_dir, filename)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Refusing to overwrite teacher replay copy: {destination}"
        )
    if os.path.realpath(destination) == source_path:
        raise ValueError("Teacher replay source and destination are identical")

    source_before = os.stat(source_path, follow_symlinks=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".copying", dir=run_dir
    )
    os.close(descriptor)
    try:
        logging.info(
            "Copying immutable teacher replay (%0.2f GiB): %s -> %s",
            source_before.st_size / (1024**3),
            source_path,
            destination,
        )
        shutil.copy2(source_path, temporary)
        source_after = os.stat(source_path, follow_symlinks=True)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(source_before, field) != getattr(source_after, field)
            for field in stable_fields
        ):
            raise RuntimeError(
                "Teacher replay source changed during copy; refusing a torn H5"
            )
        copied = os.stat(temporary, follow_symlinks=True)
        if copied.st_size != source_before.st_size:
            raise RuntimeError(
                "Teacher replay copy size does not match its immutable source"
            )
        # The resumed DAgger run and Stage 2 only consume this offline dataset.
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    logging.info("Immutable teacher replay copy ready: %s", destination)
    return destination


def teacher_replay_storage_dir(wandb_files_dir, hydra_output_dir=None):
    """Return the Hydra output root outside W&B's recursively watched files."""
    files_dir = os.path.realpath(
        os.path.abspath(os.fspath(wandb_files_dir))
    )
    if hydra_output_dir is not None:
        output_dir = os.path.realpath(
            os.path.abspath(os.fspath(hydra_output_dir))
        )
        if os.path.commonpath((files_dir, output_dir)) == files_dir:
            raise ValueError(
                "Teacher replay storage must be outside W&B's files directory"
            )
        return output_dir
    if os.path.basename(files_dir) != "files":
        raise ValueError(
            f"Expected a W&B files directory, got: {files_dir}"
        )
    wandb_dir = os.path.dirname(os.path.dirname(files_dir))
    if os.path.basename(wandb_dir) != "wandb":
        raise ValueError(
            f"Cannot derive Hydra output root from W&B path: {files_dir}"
        )
    return os.path.dirname(wandb_dir)


def find_local_teacher_replay(checkpoint_path, filename):
    """Find one local H5 beside a checkpoint or at its Hydra output root."""
    checkpoint_path = os.path.realpath(
        os.path.abspath(os.fspath(checkpoint_path))
    )
    filename = os.fspath(filename)
    if (
        not filename
        or filename in (".", "..")
        or os.path.basename(filename) != filename
    ):
        raise ValueError("Teacher replay filename must be a plain basename")

    files_dir = os.path.dirname(checkpoint_path)
    candidates = []
    for root, _, files in os.walk(files_dir):
        if filename in files:
            candidates.append(os.path.realpath(os.path.join(root, filename)))
    try:
        output_candidate = os.path.join(
            teacher_replay_storage_dir(files_dir), filename
        )
    except ValueError:
        output_candidate = None
    if output_candidate is not None and os.path.isfile(output_candidate):
        candidates.append(os.path.realpath(output_candidate))

    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple teacher replay buffers were found for the checkpoint; "
            f"set teacher_replay_buffer_path explicitly: {candidates}"
        )
    return candidates[0] if candidates else None


def apply_fastsac_buffer_steps(cfg: DictConfig):
    """Derive the flat Stage-1 FastSAC FIFO size from an env-local horizon.

    HOI expresses replay capacity as steps per vector environment, while VAIC's
    device FIFO is flat. ``task.buffer_steps`` keeps the HOI-facing CLI and this
    helper translates it to ``num_envs * buffer_steps`` before policy creation.
    A null value preserves the explicit/default ``teacher_buffer_capacity``.
    """
    buffer_steps = cfg.task.get("buffer_steps", None)
    if buffer_steps is None:
        return None

    supports_teacher_replay = "teacher_buffer_capacity" in cfg.algo
    if not supports_teacher_replay or cfg.algo.get("phase", None) != "train":
        raise ValueError(
            "task.buffer_steps is only supported by Stage-1 FastSAC teacher "
            "training."
        )

    num_envs = cfg.task.get("num_envs", None)
    replay_shape = (
        ("task.num_envs", num_envs),
        ("task.buffer_steps", buffer_steps),
    )
    for name, value in replay_shape:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if int(value) < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    capacity = int(num_envs) * int(buffer_steps)
    cfg.algo.teacher_buffer_capacity = capacity
    logging.info(
        "FastSAC replay horizon: %d envs x %d steps = %d transitions.",
        int(num_envs),
        int(buffer_steps),
        capacity,
    )
    return capacity


def _apply_direct_sac_dagger_q_transfer(algo_cfg, source_q_backend):
    """Materialize and validate the transferable BC-DAgger Q contract.

    Missing fusion metadata belongs to the legacy early-fusion generation.  It
    must not be interpreted using the new late-fusion topology merely because
    the destination config now defaults to late fusion: tensor shapes and, more
    importantly, the action-conditioning semantics would differ.
    """
    if not isinstance(source_q_backend, dict):
        raise ValueError(
            "BC-DAgger transfer checkpoint is missing its Q backend "
            "configuration"
        )

    source_fusion = str(source_q_backend.get("q_action_fusion", "early"))
    if source_fusion != "late":
        raise ValueError(
            "Pretrained BC-DAgger Q transfer requires a late-fusion source; "
            f"this checkpoint uses {source_fusion!r} fusion (missing metadata "
            "is legacy early fusion). Set algo.load_pretrained_q=false to "
            "transfer the BC actor/perception with a fresh Stage-2 Q."
        )

    transfer_q_fields = {
        "q_action_coordinates": "q_action_coordinates",
        "q_action_fusion": "q_action_fusion",
        "sac_q_normalize_actions": "q_action_normalized",
        "sac_q_action_input_gain": "q_action_input_gain",
        "sac_clipped_double_q": "clipped_double_q",
    }
    missing = [
        source
        for source in transfer_q_fields.values()
        if source not in source_q_backend
    ]
    if missing:
        raise ValueError(
            "SAC-critic BC-DAgger checkpoint lacks Q field(s) "
            f"{missing!r}"
        )
    transferred = {
        destination: source_q_backend[source]
        for destination, source in transfer_q_fields.items()
    }
    incompatible = {
        "q_action_coordinates": transferred["q_action_coordinates"]
        != "absolute",
        "q_action_fusion": transferred["q_action_fusion"] != "late",
        "q_reference_dueling": bool(
            algo_cfg.get("q_reference_dueling", False)
        ),
        "q_condition_on_actuator_state": bool(
            algo_cfg.get("q_condition_on_actuator_state", False)
        ),
        "sac_q_action_input_gain": (
            not np.isfinite(float(transferred["sac_q_action_input_gain"]))
            or float(transferred["sac_q_action_input_gain"]) <= 0.0
        ),
        "sac_q_normalize_actions": not bool(
            transferred["sac_q_normalize_actions"]
        ),
        "sac_clipped_double_q": not bool(
            transferred["sac_clipped_double_q"]
        ),
    }
    enabled = [name for name, invalid in incompatible.items() if invalid]
    if enabled:
        raise ValueError(
            "BC-DAgger FastSAC Q transfer requires normalized absolute "
            "late-fusion, finite-positive-gain, clipped-double-Q semantics; "
            f"incompatible settings: {enabled}"
        )
    for destination, value in transferred.items():
        algo_cfg[destination] = value


def _load_policy_checkpoint(
    policy: ModBase,
    policy_state: dict,
    *,
    inference_only: bool,
):
    """Load a policy checkpoint without conflating evaluation with resume.

    Distributional TD3/FastSAC DAgger checkpoints intentionally reject
    same-stage ``load_state_dict`` because their replay rings are not
    serialized.  During evaluation no replay or optimizer state is needed, so
    those policies expose a separate, guarded model-only loader.  Other
    checkpoints continue to use their existing compatibility loader.
    """
    algorithm = policy_state.get("training_algorithm")
    replayless_inference_algorithms = {
        "distributional_td3_teacher_bc_v1",
        "distributional_fastsac_teacher_bc_v1",
        "distributional_tvkd_fastsac_teacher_bc_v1",
    }
    if inference_only and algorithm in replayless_inference_algorithms:
        loader = getattr(policy, "load_inference_state_dict", None)
        if not callable(loader):
            raise ValueError(
                "The checkpoint is algorithm-specific "
                f"({algorithm!r}), but the selected policy does not support "
                "inference-only reload. Select the matching algo config for "
                "this checkpoint."
            )
        print(colored(
            "[Info]: Load policy model state for inference only "
            "(optimizer, replay, RNG, and training counters are ignored).",
            "green",
        ))
        return loader(policy_state, strict=True)
    return policy.load_state_dict(policy_state)


def _fill_replayless_inference_algo_defaults(
    cfg: DictConfig,
    policy_state: Mapping,
    *,
    inference_only: bool,
) -> dict[str, tuple[str, ...]]:
    """Complete legacy TD3/FastSAC configs only for model-only evaluation.

    Older checkpoints predate fields that their current policy dataclasses
    require during construction.  Preserve the loaded Hydra config exactly
    where it has a value, then source any missing current field from the
    checkpoint's own backend contract before falling back to today's default.
    Training never enters this migration path.
    """
    empty = {"checkpoint": (), "defaults": ()}
    if not inference_only or not isinstance(policy_state, Mapping):
        return empty

    algorithm = policy_state.get("training_algorithm")
    if algorithm == "distributional_td3_teacher_bc_v1":
        from active_adaptation.learning.ppo.td3_bc_dagger import (
            DistributionalTD3TeacherBCConfig,
        )

        default_config = DistributionalTD3TeacherBCConfig()
    elif algorithm == "distributional_fastsac_teacher_bc_v1":
        from active_adaptation.learning.ppo.fastsac_bc_dagger import (
            DistributionalFastSACTeacherBCConfig,
        )

        default_config = DistributionalFastSACTeacherBCConfig()
    elif algorithm == "distributional_tvkd_fastsac_teacher_bc_v1":
        from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import (
            TVKDDistributionalFastSACTeacherBCConfig,
        )

        default_config = TVKDDistributionalFastSACTeacherBCConfig()
    else:
        return empty

    if "algo" not in cfg or cfg.algo is None:
        raise ValueError(
            "inference-only TD3/FastSAC checkpoint reload requires cfg.algo"
        )
    current_fields = {
        field.name: copy.deepcopy(getattr(default_config, field.name))
        for field in fields(default_config)
    }
    backend = policy_state.get("dagger_backend_config")
    if not isinstance(backend, Mapping):
        backend = {}

    filled_checkpoint = []
    filled_defaults = []
    with open_dict(cfg.algo):
        if algorithm == "distributional_tvkd_fastsac_teacher_bc_v1":
            # ValueNorm changes the module type, so it must be selected from
            # the checkpoint before policy construction even when an eval
            # config already carries the structured default ``False``.
            saved_value_norm = backend.get("value_norm")
            if not isinstance(saved_value_norm, bool):
                raise ValueError(
                    "TVKD inference checkpoint lacks boolean value_norm metadata"
                )
            if cfg.algo.get("value_norm") != saved_value_norm:
                cfg.algo.value_norm = saved_value_norm
                filled_checkpoint.append("value_norm")
        for name in current_fields:
            if name not in cfg.algo and name in backend:
                cfg.algo[name] = copy.deepcopy(backend[name])
                filled_checkpoint.append(name)
        for name, value in current_fields.items():
            if name not in cfg.algo:
                cfg.algo[name] = value
                filled_defaults.append(name)

    result = {
        "checkpoint": tuple(filled_checkpoint),
        "defaults": tuple(filled_defaults),
    }
    if filled_checkpoint or filled_defaults:
        print(colored(
            "[Info]: Completed legacy inference algo config without "
            "overwriting existing values: "
            f"checkpoint={len(filled_checkpoint)}, "
            f"current_defaults={len(filled_defaults)}.",
            "green",
        ))
    return result


def make_env_policy(
    cfg: DictConfig,
    configure_replay: bool = False,
    *,
    inference_only: bool = False,
):
    OmegaConf.set_struct(cfg, False)
    cfg._bc_dagger_inference_only = bool(inference_only)

    if configure_replay:
        apply_teacher_replay_buffer_path_alias(cfg)

    # Environment imports and construction may run startup randomization. Seed
    # every RNG first so independent training processes are reproducible.
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    from active_adaptation.envs import SimpleEnv
    from torchrl.envs.transforms import (
        TransformedEnv,
        Compose,
        InitTracker,
        RenameTransform,
        StepCounter,
        VecNorm,
    )
    
    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])

    for obs_group_key in list(cfg.task.observation.keys()):
        if (
            obs_group_key not in policy_in_keys
            and not obs_group_key.endswith("_")
        ):
            cfg.task.observation.pop(obs_group_key)
            print(colored(f"Discard obs group {obs_group_key} as it is not used.", "yellow"))

    base_env = SimpleEnv(cfg.task)

    freeze_bc_dagger_teacher_replay = (
        bool(cfg.get("_bc_dagger_model_only_resume", False))
    )
    fresh_bc_dagger_source = bool(
        cfg.get("_bc_dagger_fresh_source", False)
        or
        cfg.get("_bc_dagger_finalization_source", False)
        or cfg.get("_bc_dagger_staging_source", False)
    )
    replay_consumer = not freeze_bc_dagger_teacher_replay and (
        cfg.algo.get("phase", None) == "finetune"
        or bool(cfg.algo.get("save_teacher_buffer", False))
    )
    auto_resolve_replay = (
        configure_replay
        and replay_consumer
        and not fresh_bc_dagger_source
        and cfg.algo.get("teacher_buffer_path", "__missing__") is None
    )
    checkpoint_path = parse_checkpoint_path(
        cfg.checkpoint_path,
        download_replay=auto_resolve_replay,
        replay_filename=cfg.algo.get(
            "teacher_buffer_filename", "teacher_replay_buffer.h5"
        ),
    )
    if (
        auto_resolve_replay
        and checkpoint_path is not None
    ):
        replay_path = find_local_teacher_replay(
            checkpoint_path, cfg.algo.teacher_buffer_filename
        )
        if replay_path is not None:
            cfg.algo.teacher_buffer_path = replay_path
            print(colored(
                f"[Info]: Found teacher replay buffer: {replay_path}",
                "green",
            ))
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}

    _fill_replayless_inference_algo_defaults(
        cfg,
        state_dict.get("policy", {}),
        inference_only=inference_only,
    )

    # Stage-2 can now warm-start directly from the dedicated PPO-BC DAgger
    # checkpoint. Resolve this before constructing transforms/policy because
    # the source determines the actor adapter and Q action coordinates.
    if (
        cfg.algo.get("phase", None) == "finetune"
        and str(cfg.algo.get("_target_", "")).endswith(".FastSACVelFinetune")
    ):
        policy_state = state_dict.get("policy", {})
        configured_source = str(
            cfg.algo.get("finetune_checkpoint_source", "auto")
        )
        detected_algorithm = policy_state.get("training_algorithm")
        detected_backend = policy_state.get("actor_backend")
        current_dagger_adapter = "vaic_fastsac_bc_dagger_adapter_v6"
        legacy_dagger_adapters = {
            "vaic_fastsac_bc_dagger_adapter_v1",
            "vaic_fastsac_bc_dagger_adapter_v2",
            "vaic_fastsac_bc_dagger_adapter_v3",
            "vaic_fastsac_bc_dagger_adapter_v4",
            "vaic_fastsac_bc_dagger_adapter_v5",
        }
        if detected_algorithm == "vaic_ppo_bc_dagger_student_v1":
            raise ValueError(
                "This BC-DAgger checkpoint predates compatible critic training. "
                "Create a new checkpoint with scripts/bc_dagger.py before "
                "starting fastsac_vel_finetune."
            )
        if detected_algorithm == "vaic_ppo_bc_dagger_student_sac_critic_v3":
            raise ValueError(
                "This BC-DAgger checkpoint predates the separated execution, "
                "Q, and entropy action contract. Start a new BC-DAgger run; "
                "SAC-critic-v3 "
                "checkpoints are intentionally incompatible with Stage 2."
            )
        if detected_backend in legacy_dagger_adapters:
            raise ValueError(
                "This Stage-2 BC-DAgger adapter checkpoint reused PPO's "
                "old action support and/or coupled action/Q/entropy "
                "coordinates. Start a new BC-DAgger run with the separated "
                "safety-envelope, nominal-Q, and bounded-residual contract; "
                "adapter_v1-v5 resumes are "
                "intentionally rejected."
            )
        current_dagger_algorithm = (
            "vaic_ppo_bc_dagger_student_sac_critic_v6"
        )
        iql_dagger_algorithm = "vaic_ppo_bc_dagger_student_iql_v2"
        if configured_source == "auto":
            if detected_algorithm in (
                current_dagger_algorithm,
                iql_dagger_algorithm,
            ):
                configured_source = "bc_dagger"
            elif detected_backend == current_dagger_adapter:
                configured_source = "bc_dagger"
            else:
                configured_source = "fastsac"
        if configured_source not in ("fastsac", "bc_dagger"):
            raise ValueError(
                "algo.finetune_checkpoint_source must be auto, fastsac, or "
                f"bc_dagger; got {configured_source!r}"
            )
        cfg.algo.finetune_checkpoint_source = configured_source
        if configured_source == "bc_dagger":
            direct_sac_dagger_transfer = (
                detected_algorithm == current_dagger_algorithm
            )
            direct_iql_dagger_transfer = (
                detected_algorithm == iql_dagger_algorithm
            )
            direct_dagger_transfer = (
                direct_sac_dagger_transfer or direct_iql_dagger_transfer
            )
            same_stage_resume = (
                detected_backend == current_dagger_adapter
                and policy_state.get("last_phase") == "finetune"
            )
            if not direct_dagger_transfer and not same_stage_resume:
                raise ValueError(
                    "The selected checkpoint is neither a current PPO-BC "
                    "DAgger source nor a current Stage-2 BC adapter resume."
                )
            if same_stage_resume:
                source_schedule = policy_state.get("stage2_schedule_config")
                if not isinstance(source_schedule, dict):
                    raise ValueError(
                        "Stage-2 BC adapter checkpoint lacks its guarded "
                        "schedule configuration"
                    )
                if "load_pretrained_q" not in source_schedule:
                    raise ValueError(
                        "Stage-2 BC adapter checkpoint predates explicit Q "
                        "source provenance; restart from BC-DAgger"
                    )
                cfg.algo.load_pretrained_q = source_schedule[
                    "load_pretrained_q"
                ]

            source_actor_backend = policy_state.get(
                "dagger_backend_config" if direct_dagger_transfer
                else "actor_backend_config"
            )
            if not isinstance(source_actor_backend, dict):
                raise ValueError(
                    "BC-DAgger transfer checkpoint is missing its actor/action "
                    "backend configuration"
                )
            source_action_contract = (
                policy_state.get("action_contract")
                if direct_dagger_transfer
                else source_actor_backend.get("action_contract")
            )
            if not isinstance(source_action_contract, dict):
                raise ValueError(
                    "BC-DAgger transfer checkpoint is missing its executable "
                    "per-joint action contract"
                )
            if source_action_contract.get("semantics") != (
                "separate_execution_support_q_and_entropy_coordinates_v2"
            ):
                raise ValueError(
                    "BC-DAgger transfer checkpoint has incompatible executable "
                    "action-contract semantics"
                )
            action_contract_fingerprint = str(
                source_action_contract.get("fingerprint", "")
            )
            if not action_contract_fingerprint.startswith("sha256:"):
                raise ValueError(
                    "BC-DAgger transfer checkpoint action contract lacks a "
                    "fingerprint"
                )
            # The scalar clip is the BC actor's symmetric tanh support. Q and
            # entropy coordinates are separate fields in the action contract,
            # so a different Stage-2 clip would change actor semantics.
            action_clip_key = (
                "dagger_action_clip"
                if direct_dagger_transfer
                else "action_safety_clip"
            )
            if action_clip_key not in source_actor_backend:
                raise ValueError(
                    "BC-DAgger transfer checkpoint lacks the saved final action "
                    "safety clip"
                )
            action_low = source_action_contract.get("action_low")
            action_high = source_action_contract.get("action_high")
            if (
                not isinstance(action_low, list)
                or not isinstance(action_high, list)
                or not action_low
                or len(action_low) != len(action_high)
            ):
                raise ValueError(
                    "BC-DAgger transfer checkpoint action contract has invalid "
                    "joint-wise bounds"
                )
            required_safety_clip = max(
                max(abs(float(low)), abs(float(high)))
                for low, high in zip(action_low, action_high)
            )
            source_safety_clip = float(source_actor_backend[action_clip_key])
            configured_safety_clip = float(cfg.algo.sac_bc_action_clip)
            if not np.isfinite(source_safety_clip) or not np.isclose(
                source_safety_clip,
                required_safety_clip,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "BC-DAgger checkpoint safety clip does not exactly match "
                    "its actor execution support"
                )
            if not np.isfinite(configured_safety_clip) or not np.isclose(
                configured_safety_clip,
                source_safety_clip,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Configured Stage-2 safety clip does not exactly match "
                    "the checkpoint actor support"
                )
            if direct_sac_dagger_transfer:
                direct_adapter_fields = (
                    "sac_bc_initial_action_std",
                    "sac_bc_log_std_min",
                    "sac_bc_log_std_max",
                    "sac_alpha_init",
                )
                for name in direct_adapter_fields:
                    if name not in source_actor_backend:
                        raise ValueError(
                            "SAC-critic BC-DAgger checkpoint lacks adapter "
                            f"field {name!r}"
                        )
                    cfg.algo[name] = source_actor_backend[name]
                if bool(cfg.algo.get("load_pretrained_q", True)):
                    critic_optimization_fields = {
                        "q_lr": "q_lr",
                        "q_weight_decay": "q_weight_decay",
                        "sac_tau": "q_tau",
                        "sac_max_grad_norm": "q_max_grad_norm",
                    }
                    for destination, source in (
                        critic_optimization_fields.items()
                    ):
                        if source not in source_actor_backend:
                            raise ValueError(
                                "SAC-critic BC-DAgger checkpoint lacks critic "
                                f"field {source!r}"
                            )
                        cfg.algo[destination] = source_actor_backend[source]
            if same_stage_resume:
                # These values define registered adapter tensors and its action
                # distribution, so materialize them before policy construction.
                adapter_fields = {
                    "sac_bc_initial_action_std": "initial_action_std",
                    "sac_bc_log_std_min": "log_std_min",
                    "sac_bc_log_std_max": "log_std_max",
                    "sac_deterministic_rollout": "rollout_behavior",
                    "sac_freeze_perception": "perception_frozen",
                }
                for destination, source in adapter_fields.items():
                    if source not in source_actor_backend:
                        raise ValueError(
                            "Stage-2 BC adapter checkpoint lacks required actor "
                            f"field {source!r}"
                        )
                    value = source_actor_backend[source]
                    if destination == "sac_deterministic_rollout":
                        value = value == "deterministic_executable_bc_mean"
                    cfg.algo[destination] = value

            # A direct actor-only transfer deliberately keeps the current,
            # freshly initialized FastSAC Q topology. A pretrained Q transfer
            # and every same-stage resume must reconstruct the saved Q exactly.
            load_pretrained_q = cfg.algo.get("load_pretrained_q", True)
            if not isinstance(load_pretrained_q, bool):
                raise ValueError("algo.load_pretrained_q must be a boolean")
            if direct_iql_dagger_transfer and load_pretrained_q:
                raise ValueError(
                    "IQL-v2 BC-DAgger Q does not match the current Stage-2 "
                    "SAC/AWAC critic topology or policy-evaluation backup. Set "
                    "algo.load_pretrained_q=false or use a v6 SAC-critic "
                    "BC-DAgger checkpoint."
                )
            inherit_q = load_pretrained_q or same_stage_resume
            source_q_backend = policy_state.get(
                "dagger_backend_config"
                if direct_iql_dagger_transfer
                else "q_backend_config"
            )
            if inherit_q and not isinstance(source_q_backend, dict):
                raise ValueError(
                    "BC-DAgger transfer checkpoint is missing its Q backend "
                    "configuration"
                )
            if inherit_q:
                if source_q_backend.get(
                    "q_action_transform_fingerprint"
                ) != source_action_contract.get(
                    "q_action_transform_fingerprint"
                ):
                    raise ValueError(
                        "BC-DAgger Q action-transform fingerprint does not "
                        "match the checkpoint action contract"
                    )
                # Construct Q1/Q2 with the exact saved topology before loading.
                q_fields = {
                    "q_hidden_dim": (
                        "q_hidden_dim"
                        if direct_iql_dagger_transfer else "hidden_dim"
                    ),
                    "q_num_atoms": (
                        "q_num_atoms"
                        if direct_iql_dagger_transfer else "num_atoms"
                    ),
                    "q_v_min": (
                        "q_v_min" if direct_iql_dagger_transfer else "v_min"
                    ),
                    "q_v_max": (
                        "q_v_max" if direct_iql_dagger_transfer else "v_max"
                    ),
                    "q_layer_norm": (
                        "q_layer_norm"
                        if direct_iql_dagger_transfer else "layer_norm"
                    ),
                }
                for destination, source in q_fields.items():
                    if source not in source_q_backend:
                        raise ValueError(
                            "BC-DAgger transfer checkpoint lacks required Q "
                            f"field {source!r}"
                        )
                    cfg.algo[destination] = source_q_backend[source]

            if direct_sac_dagger_transfer and load_pretrained_q:
                _apply_direct_sac_dagger_q_transfer(
                    cfg.algo, source_q_backend
                )
            elif direct_dagger_transfer:
                # The DAgger H5 stores absolute actor/Q observations and
                # actions, but no framewise reference-action or actuator-state
                # columns. Fresh Q may choose early/late fusion and optional
                # normalization, but cannot request fields absent from replay.
                replay_incompatible = {
                    "q_action_coordinates": cfg.algo.get(
                        "q_action_coordinates", "absolute"
                    ) != "absolute",
                    "q_reference_dueling": bool(cfg.algo.get(
                        "q_reference_dueling", False
                    )),
                    "q_condition_on_actuator_state": bool(cfg.algo.get(
                        "q_condition_on_actuator_state", False
                    )),
                }
                enabled = [
                    name
                    for name, invalid in replay_incompatible.items()
                    if invalid
                ]
                if enabled:
                    raise ValueError(
                        "Fresh-Q BC-DAgger Stage 2 cannot request fields absent "
                        f"from its offline replay: {enabled}"
                    )
            elif same_stage_resume:
                same_stage_q_fields = {
                    "q_action_coordinates": "q_action_coordinates",
                    "q_action_fusion": "q_action_fusion",
                    "q_reference_dueling": "q_reference_dueling",
                    "sac_q_normalize_actions": "q_action_normalized",
                    "sac_q_action_input_gain": "q_action_input_gain",
                    "sac_clipped_double_q": "clipped_double_q",
                    "sac_use_autotune": "alpha_autotune",
                }
                for destination, source in same_stage_q_fields.items():
                    if source not in source_q_backend:
                        raise ValueError(
                            "Stage-2 BC adapter checkpoint lacks required Q "
                            f"field {source!r}"
                        )
                    cfg.algo[destination] = source_q_backend[source]
                actuator_context = source_q_backend.get("q_actuator_context")
                if not isinstance(actuator_context, dict):
                    raise ValueError(
                        "Stage-2 BC adapter checkpoint lacks Q actuator context"
                    )
                cfg.algo.q_condition_on_actuator_state = bool(
                    actuator_context.get("enabled", False)
                )

            if same_stage_resume:
                source_detail = (
                    "Stage-2 model/optimizer resume with online replay refill"
                )
            elif load_pretrained_q:
                source_detail = "normalized SAC-compatible Q transfer"
            else:
                source_detail = (
                    "fresh configured Q; checkpoint Q weights skipped"
                )
            print(colored(
                "[Info]: FastSAC finetune source: PPO-BC DAgger "
                "(frozen-BC-centered bounded-residual stochastic train "
                "behavior, deterministic residual mean evaluation, dedicated "
                "SAC std, exact physical-action "
                f"replay, {source_detail}).",
                "green",
            ))
    
    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True) 
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

    raw_replay_observations = (
        configure_replay
        and bool(
            cfg.algo.get("sac_replay_raw_observations", False)
            or cfg.algo.get("dagger_replay_raw_observations", False)
        )
        and cfg.algo.get("phase", None) in ("train", "finetune")
    )
    if raw_replay_observations:
        configured_raw_keys = cfg.algo.get(
            "replay_raw_observation_keys", None
        )
        if configured_raw_keys is not None:
            requested_raw_keys = set(configured_raw_keys)
            replay_obs_keys = [
                key for key in obs_keys if key in requested_raw_keys
            ]
            missing_raw_keys = requested_raw_keys.difference(obs_keys)
            if missing_raw_keys:
                logging.info(
                    "Configured raw replay keys are absent from this task and "
                    "will be ignored: %s",
                    sorted(missing_raw_keys),
                )
        else:
            replay_obs_keys = obs_keys
        raw_keys = [
            ("_fastsac_raw", *key) if isinstance(key, tuple)
            else ("_fastsac_raw", key)
            for key in replay_obs_keys
        ]
        # Copy before VecNorm. Inverting normalized rollout tensors later is
        # not valid because current, next, and reset states can see different
        # running-stat snapshots.
        transform.append(RenameTransform(
            replay_obs_keys, raw_keys, create_copy=True
        ))

    if "vecnorm" in state_dict.keys():
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
        vecnorm.load_state_dict(state_dict["vecnorm"])
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating obervation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating obervation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())
    elif cfg.vecnorm is not None:
        raise ValueError

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    
    # setup policy
    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    active_adaptation.print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy: ModBase = policy_cls(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device,
        env=env
    )

    if raw_replay_observations:
        if not hasattr(policy, "configure_replay_vecnorm"):
            raise TypeError(
                "raw replay observations require a policy with "
                "configure_replay_vecnorm()."
            )
        policy.configure_replay_vecnorm(vecnorm)
    
    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        _load_policy_checkpoint(
            policy,
            state_dict["policy"],
            inference_only=inference_only,
        )

    if (
        configure_replay
        and cfg.algo.get("phase") == "finetune"
        and hasattr(policy, "configure_offline_replay")
    ):
        policy.configure_offline_replay(cfg.algo.teacher_buffer_path)
    
    if hasattr(policy, "make_tensordict_primer"):
        primer = policy.make_tensordict_primer()
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
        env = TransformedEnv(env.base_env, transform)
    env: _Env

    return env, policy, vecnorm


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm

@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    # exploration_type: ExplorationType=ExplorationType.RANDOM,
    render=False,
    render_mode="rgb_array",
    keys=[("next", "stats")],
    policy_keys=[],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """
    keys = ["ref_motion_phase_", "step_count"]
    keys = set(keys)
    keys.add(("next", "done"))
    keys.add(("next", "stats"))


    env.base_env.eval()
    env.eval()
    policy.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs = []
    frames = []
    policy_trajs = []

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)

            policy_trajs.append(tensordict_.select(*policy_keys, strict=False).cpu())
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(*keys, strict=False).cpu())

            if render:
                frames.append(env.render(mode=render_mode))
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    policy_trajs: TensorDictBase = torch.stack(policy_trajs, dim=1)
    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    episode_len = take_first_episode(trajs["next", "stats", "episode_len"])
    # shape: (num_envs,)
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        if k == "episode_len" or k == "success":
            pass
        else:
            v = v.float() / episode_len.float()

        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        info[key + "_std"] = torch.std(v.float()).item()

    # log video
    if len(frames):
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_array = np.stack(frames)
        frames.clear()
        video_path = os.path.join(os.path.dirname(__file__), f"recording-{time_str}.mp4")
        imageio.mimwrite(
            video_path,
            video_array,
            fps=int(1/env.step_dt),
            codec="libx264"
        )

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats, policy_trajs


def extract_episodes(trajs: TensorDictBase) -> List[TensorDictBase]:
    """
    将一个包含多个环境和时间步的批次化 TensorDict 分割成一个列表,
    其中每个元素都是一个独立的、完整的 episode。

    Args:
        trajs (TensorDictBase): 一个形状为 (N, T, ...) 的 TensorDict,
            其中 N 是环境数量, T 是时间步数。
            这个 TensorDict 必须包含键 ("next", "done")。

    Returns:
        List[TensorDictBase]: 一个 TensorDict 的列表。列表中的每个 TensorDict
            代表一个完整的 episode,其形状为 (t, ...), t 是该 episode 的长度。
    """
    # 验证输入 TensorDict 的维度是否正确 (N, T)
    if trajs.batch_dims != 2:
        raise ValueError(f"输入的 trajs 应该有两个批次维度 (N, T), 但得到了 {trajs.batch_dims} 个。")

    # 获取 done 信号, 形状为 (N, T)
    # 使用 .squeeze() 以防 done 信号的形状是 (N, T, 1)
    dones = trajs.get(("next", "done")).squeeze(-1) 
    if dones.ndim != 2:
        raise ValueError(f"期望 ('next', 'done') 是一个二维张量, 但其形状为 {dones.shape}")

    N, T = dones.shape
    
    all_episodes = []

    # 遍历每一个环境
    for i in range(N):
        # 找到当前环境中所有 done=True 的时间步索引
        # torch.where 返回一个元组, 我们需要第一个元素
        done_indices = torch.where(dones[i])[0]

        start_idx = 0
        # 遍历这些结束点, 切分出每一个 episode
        for end_idx in done_indices:
            # 切片是左闭右开, 所以我们需要 end_idx + 1 来包含结束的那一帧
            episode = trajs[i, start_idx : end_idx + 1]
            all_episodes.append(episode)
            
            # 更新下一个 episode 的起始点
            start_idx = end_idx + 1
            
    return all_episodes

def evaluate_track(trajs: TensorDictBase) -> Tuple[TensorDictBase, TensorDictBase]:
    trajs_tracking_info = trajs.select("ref_motion_phase_", "step_count", ("next", "done"))
    episodes = extract_episodes(trajs_tracking_info)
    init_ref_motion_phase = []
    final_step_count = []
    for episode in episodes:
        init_ref_motion_phase.append(episode["ref_motion_phase_"][1].item())
        final_step_count.append(episode["step_count"][-1].item())
        
    # use matplotlib to plot the init_ref_motion_phase and final_step_count
    # use scatter plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.scatter(init_ref_motion_phase, final_step_count, alpha=0.5)
    plt.xlabel("Initial Reference Motion Phase")
    plt.ylabel("Final Step Count")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "init_ref_motion_phase_vs_final_step_count.png"))
    plt.close()

    breakpoint()

def plot_obs_histogram(
    trajs: TensorDictBase, 
):
    trajs_obs: TensorDictBase = trajs.flatten(0, 1).select("command", "policy")
    policy_obs_np = trajs_obs.numpy()["policy"]
    # use matplotlib to plot the histgram of each dimension of trajs_obs_np
    num_cols = 15
    num_rows = (policy_obs_np.shape[-1] + num_cols - 1) // num_cols
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    for i in range(policy_obs_np.shape[-1]):
        ax = axes[i // num_cols, i % num_cols]
        ax.hist(policy_obs_np[:, i], bins=50)
        # plot mean for this dimension
        mean = np.mean(policy_obs_np[:, i])
        std = np.std(policy_obs_np[:, i])
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1)
        ax.axvline(mean + std, color='green', linestyle='dashed', linewidth=1)
        ax.axvline(mean - std, color='green', linestyle='dashed', linewidth=1)
        ax.set_title(f"Dim {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "trajs_obs_hist.png"))
    plt.close()
