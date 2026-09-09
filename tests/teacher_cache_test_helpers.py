"""Self-contained regression fixtures for exact Teacher cache rebuilding.

The frozen reference below was captured before the synchronization change.
No fixture imports ignored diagnostics or requires checkpoint artifacts.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from active_adaptation.learning.ppo.ppo_vel import (
    DEPTH_KEY, OBS_KEY, VEL_CMD_KEY, OBJECT_GEO_KEY, PPOVEL, set_recurrent_mode,
)
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC as TD3,
    PERCEPTION_DEPTH_U8_KEY, PERCEPTION_POLICY_RAW_KEY,
    PERCEPTION_VEL_COMMAND_RAW_KEY, PERCEPTION_OBJECT_GEO_ID_KEY,
    PERCEPTION_IS_INIT_KEY, _decode_replay_depth_u8,
)
from active_adaptation.learning.ppo.teacher_episode_replay import (
    TeacherActorCacheLineage, TeacherEpisodeSequenceStore,
)
from active_adaptation.learning.ppo.tvkd_fastsac_bc_dagger import TVKDDistributionalFastSACTeacherBCConfig

SOURCE_MODULE_SHA256 = '7bd558895901e119291028635867f7904b1138d6441ab49952c89b6bc1c3b0bb'

@torch.no_grad()
def reference_rebuild_teacher_actor_cache(
    self, lineage: TeacherActorCacheLineage
) -> None:
    """Stream complete Teacher episodes through the current EMA exactly."""
    self._ensure_teacher_episode_cache_state()
    store = self._teacher_episode_store
    if not store.frozen:
        raise RuntimeError("Teacher Actor cache requires a frozen episode store")

    # Keep the derived cache on the learning device.  Copying every
    # 525-wide state back to the CPU ring (and then copying sampled rows
    # to CUDA again) made an exact refresh substantially slower than the
    # legacy ten-frame path.  A single device cache lets the recurrent
    # stream publish without per-chunk D2H synchronization; Q/Actor later
    # gather only the Teacher rows that their unchanged replay indices
    # selected.
    actor_by_node = self._teacher_actor_cache.allocate_build_tensor(
        store, device=self.device
    )
    write_counts = torch.zeros(
        store.node_count, dtype=torch.int16, device=self.device
    )
    snapshot = self._vecnorm_snapshot()
    time_chunk_size = max(1, int(self.cfg.train_every))
    node_budget = max(
        1,
        int(self.cfg.perception_encode_microbatch_size)
        * (int(self.cfg.perception_replay_burn_in) + 2),
    )
    episode_batch_size = max(1, node_budget // time_chunk_size)
    raw_lineage = store.lineage
    device_raw_lineage = (
        raw_lineage.store_id,
        raw_lineage.generation,
        str(torch.device(self.device)),
    )
    if self._teacher_episode_device_raw_lineage != device_raw_lineage:
        # The Teacher store is immutable after prefill.  Upload its
        # compact, frame-once raw tensors once and reuse them across all
        # later EMA generations.  Only the derived Actor cache is
        # invalidated after perception learning.
        self._teacher_episode_device_raw_fields = {
            key: value.to(self.device).contiguous()
            for key, value in store.raw_fields.items()
        }
        self._teacher_episode_device_raw_lineage = device_raw_lineage
    device_raw_fields = self._teacher_episode_device_raw_fields
    if device_raw_fields is None:  # pragma: no cover - guarded above
        raise RuntimeError("Teacher raw device mirror is unavailable")
    current_group = None
    depth_state = None
    adapt_state = None

    with set_recurrent_mode(True):
        for chunk in store.iter_sequence_chunks(
            episode_batch_size=episode_batch_size,
            time_chunk_size=time_chunk_size,
            raw_fields=device_raw_fields,
        ):
            if current_group != chunk.group_id:
                current_group = chunk.group_id
                depth_state = torch.zeros(
                    chunk.group_size,
                    self.depth_feature_dim,
                    device=self.device,
                )
                adapt_state = torch.zeros(
                    chunk.group_size,
                    int(self.cfg.latent_dim),
                    device=self.device,
                )
            if depth_state is None or adapt_state is None:  # pragma: no cover
                raise RuntimeError("Teacher recurrent cache state is unavailable")

            positions = chunk.batch_positions.to(
                device=self.device, dtype=torch.long
            )
            count, sequence_length = chunk.valid.shape
            depth_u8 = chunk.raw_fields[PERCEPTION_DEPTH_U8_KEY].to(
                self.device
            )
            policy_raw = chunk.raw_fields[PERCEPTION_POLICY_RAW_KEY].to(
                self.device
            )
            vel_raw = chunk.raw_fields[PERCEPTION_VEL_COMMAND_RAW_KEY].to(
                self.device
            )
            geometry = self._decode_replay_object_geo(
                chunk.raw_fields[PERCEPTION_OBJECT_GEO_ID_KEY],
                device=self.device,
                dtype=policy_raw.dtype,
            )
            depth = self._normalize_replay_value(
                DEPTH_KEY, _decode_replay_depth_u8(depth_u8), snapshot
            )
            policy = self._normalize_replay_value(
                OBS_KEY, policy_raw, snapshot
            )
            vel = self._normalize_replay_value(VEL_CMD_KEY, vel_raw, snapshot)
            depth_hx = depth_state.index_select(0, positions)
            adapt_hx = adapt_state.index_select(0, positions)
            td = TensorDict(
                {
                    DEPTH_KEY: depth,
                    OBS_KEY: policy,
                    VEL_CMD_KEY: vel,
                    OBJECT_GEO_KEY: geometry.to(dtype=policy.dtype),
                    "is_init": chunk.raw_fields[PERCEPTION_IS_INIT_KEY].to(
                        self.device
                    ),
                    "depth_hx": depth_hx.unsqueeze(1).expand(
                        count, sequence_length, -1
                    ),
                    "adapt_hx": adapt_hx.unsqueeze(1).expand(
                        count, sequence_length, -1
                    ),
                },
                batch_size=(count, sequence_length),
                device=self.device,
            )
            if hasattr(self, "temporal_depth_gru_ema"):
                self.temporal_depth_gru_ema(td)
                if ("next", "depth_hx") not in td.keys(True, True):
                    raise RuntimeError(
                        "Teacher cache requires the locked recurrent depth EMA"
                    )
                depth_state.index_copy_(
                    0, positions, td["next", "depth_hx"][:, -1]
                )
            else:
                td["_depth_feature"] = torch.zeros(
                    count,
                    sequence_length,
                    self.depth_feature_dim,
                    device=self.device,
                    dtype=policy.dtype,
                )
            if bool(self.cfg.use_object_adapt):
                self.object_adapt_ema(td)
                self.object_pred_transform(td)
            self.adapt_ema(td)
            if ("next", "adapt_hx") not in td.keys(True, True):
                raise RuntimeError(
                    "Teacher cache requires the locked recurrent adaptation EMA"
                )
            adapt_state.index_copy_(
                0, positions, td["next", "adapt_hx"][:, -1]
            )
            actor_parts = []
            for key, width in zip(self.q_actor_keys, self._q_actor_widths):
                if key not in td.keys(True, True):
                    raise KeyError(
                        f"Teacher Actor cache is missing input {key!r}"
                    )
                value = td[key]
                if int(value.shape[-1]) != int(width):
                    raise ValueError(
                        f"Teacher Actor cache input {key!r} has width "
                        f"{int(value.shape[-1])}; expected {int(width)}"
                    )
                actor_parts.append(value)
            actor = torch.cat(actor_parts, dim=-1)
            if actor.shape != (*td.batch_size, self._q_actor_dim):
                raise RuntimeError("Teacher Actor cache has an invalid shape")
            valid_device = chunk.valid.to(self.device)
            node_indices = chunk.flat_node_indices[chunk.valid].to(self.device)
            actor_by_node.index_copy_(
                0,
                node_indices,
                actor[valid_device].float(),
            )
            write_counts.index_add_(
                0,
                node_indices,
                torch.ones_like(node_indices, dtype=write_counts.dtype),
            )

    if not bool((write_counts == 1).all()):
        raise RuntimeError(
            "Teacher Actor cache must materialize every node exactly once"
        )
    self._teacher_actor_cache.publish(lineage, store, actor_by_node)


class TeacherCacheDiagnosticPolicy(nn.Module):
    _ensure_teacher_episode_cache_state = TD3._ensure_teacher_episode_cache_state
    _ensure_replay_object_geo_codebook = TD3._ensure_replay_object_geo_codebook
    _replay_object_geo_bank_for = TD3._replay_object_geo_bank_for
    _decode_replay_object_geo = TD3._decode_replay_object_geo
    _teacher_actor_cache_lineage = TD3._teacher_actor_cache_lineage
    _rebuild_teacher_actor_cache = TD3._rebuild_teacher_actor_cache

    def __init__(self, device="cpu", *, microbatch=512, residual=False):
        super().__init__()
        self.device = torch.device(device)
        cfg = TVKDDistributionalFastSACTeacherBCConfig(perception_depth_residual=residual)
        cfg.latent_dim = 256
        cfg.enable_residual_distillation = False
        cfg.train_every = 32
        cfg.perception_encode_microbatch_size = microbatch
        self.cfg = cfg
        dimensions = {"policy": (249,), "priv": (1,), "command": (356,),
                      "vel_command": (20,), "object_": (22,), "object_geo_": (384,),
                      "depth": (1, 36, 64)}
        env = SimpleNamespace(cfg=SimpleNamespace(reward={"tracking": {}}),
                              action_manager=SimpleNamespace(joint_names=list(map(str, range(23)))))
        spec = Composite({key: Unbounded((1, *shape)) for key, shape in dimensions.items()}, shape=(1,))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19501)
            source = PPOVEL(cfg, spec, Unbounded((1, 23)), Unbounded((1, 1)), "cpu", env)
            self._replay_object_geo_bank = torch.randn(4, 384)
        for name in ("temporal_depth_gru_ema", "object_adapt_ema", "object_pred_transform", "adapt_ema"):
            setattr(self, name, getattr(source, name).to(self.device))
        self.requires_grad_(False).eval()
        self.depth_feature_dim = 64
        self.q_actor_keys = ("vel_command", "policy", "priv_pred")
        self._q_actor_widths = (20, 249, 256)
        self._q_actor_dim = 525
        self._perception_ema_generation = 0
        self._replay_vecnorm_fingerprint = "synthetic-fixed-normalization"
        self._replay_object_geo_fingerprint = "synthetic-lossless-geometry"
        self._replay_object_geo_bank_generation = 1
        self._ensure_replay_object_geo_codebook()
        self._ensure_teacher_episode_cache_state()

    def _vecnorm_snapshot(self):
        return None

    def _normalize_replay_value(self, key, value, snapshot):
        return value

def install_store(policy, lengths):
    generator = torch.Generator().manual_seed(19502)
    store = TeacherEpisodeSequenceStore(is_init_key=PERCEPTION_IS_INIT_KEY)
    uids, steps = [], []
    for length in lengths:
        uid = store.allocate_episode_uid()
        reset = torch.zeros(length, 1, dtype=torch.bool)
        reset[0] = True
        fields = {
            PERCEPTION_DEPTH_U8_KEY: torch.randint(0, 101, (length, 1, 36, 64), dtype=torch.uint8, generator=generator),
            PERCEPTION_POLICY_RAW_KEY: torch.randn(length, 249, generator=generator),
            PERCEPTION_VEL_COMMAND_RAW_KEY: torch.randn(length, 20, generator=generator),
            PERCEPTION_OBJECT_GEO_ID_KEY: (torch.arange(length) % 4).int(),
            PERCEPTION_IS_INIT_KEY: reset,
        }
        store.commit_successful_episode(uid, fields)
        uids.append(uid)
        steps.append(length - 1)
    store.freeze(torch.tensor(uids), torch.tensor(steps))
    policy._teacher_episode_store = store
    policy._teacher_episode_device_raw_lineage = None
    policy._teacher_episode_device_raw_fields = None
    return store

def capture_rebuild(policy, method):
    hidden = {"depth_hx": [], "adapt_hx": []}
    hooks = []
    for name, key in (("temporal_depth_gru_ema", "depth_hx"), ("adapt_ema", "adapt_hx")):
        hooks.append(getattr(policy, name).register_forward_hook(
            lambda _module, _args, td, key=key: hidden[key].append(td["next", key][:, -1].detach().clone())
        ))
    try:
        method(policy, policy._teacher_actor_cache_lineage())
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "actor_inputs": policy._teacher_actor_cache._actor_by_node.detach().clone(),
        **{key: torch.cat(values) for key, values in hidden.items()},
    }

def compare(reference, candidate):
    report = {}
    for key in reference:
        expected, actual = reference[key], candidate[key]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report[key] = {"torch_equal": torch.equal(actual, expected),
                       "max_absolute_difference": float((actual - expected).abs().max())}
    return report
