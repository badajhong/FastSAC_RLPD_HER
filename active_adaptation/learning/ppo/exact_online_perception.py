"""Reset-to-state EMA perception for online replay and live recurrent carry.

Only derived features are cached. Raw episode prefixes are retained losslessly
while an online replay row or a live environment references them. Every cache
entry belongs to one EMA generation; an update makes all old hidden states
unusable. Time chunks bound workspace, never the amount of recurrent history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch
from tensordict import TensorDict

from .exact_episode_replay import ExactEpisodePrefixStore, _copy_cpu_tensor_
from .exact_gru_cuda_graph import exact_gru_cuda_graphs
from .ppo_vel import (
    DEPTH_KEY, OBS_KEY, VEL_CMD_KEY, OBJECT_GEO_KEY, PRIV_PRED_KEY,
    exact_recurrent_lengths, set_recurrent_mode,
)
from .td3_bc_dagger import (
    COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS,
    DistributionalTD3TeacherBC,
    PERCEPTION_REPLAY_SEMANTICS,
    PERCEPTION_OBJECT_GEO_ID_KEY,
    REPLAY_ACTOR_OBSERVATIONS_KEY,
    REPLAY_NEXT_ACTOR_OBSERVATIONS_KEY,
    REPLAY_SAMPLE_IS_TEACHER_KEY,
    REPLAY_SAMPLE_IS_DAGGER_ENV_KEY,
    REPLAY_SAMPLE_PHYSICAL_INDEX_KEY,
    _PREFILL_ENV_INDEX_KEY,
    _PREFILL_STEP_INDEX_KEY,
)

EXACT_ONLINE_ACTOR_REPLAY_SEMANTICS = (
    "online_reset_prefix_current_ema_current_next_and_live_carry_v1"
)
EXACT_CURRENT_REF = "exact_online_current_ref"
EXACT_NEXT_REF = "exact_online_next_ref"
_FINAL_INPUT_PREFIX = "exact_input__"
EXACT_LIVE_HIDDEN_KEY = "exact_online_live_hidden"
EXACT_LIVE_GENERATION_KEY = "exact_online_live_generation"


@dataclass
class _EncodedPrefix:
    length: int = 0
    actor_chunks: list[torch.Tensor] = field(default_factory=list)
    depth_hx: torch.Tensor | None = None
    adapt_hx: torch.Tensor | None = None


class ExactOnlinePerceptionReplayMixin:
    """Strict current-EMA machinery, enabled by the algorithm implementation."""

    def _exact_online_replay_enabled(self) -> bool:
        # TVKD overrides this as an invariant, not a Hydra/user-selectable mode.
        # The non-TVKD FastSAC backend keeps its existing replay behavior.
        return False

    def _actor_replay_observation_semantics(self) -> str:
        if self._exact_online_replay_enabled():
            return EXACT_ONLINE_ACTOR_REPLAY_SEMANTICS
        return (
            COLLECTION_EXACT_ACTOR_REPLAY_SEMANTICS
            if self._student_collection_actor_cache_enabled()
            else PERCEPTION_REPLAY_SEMANTICS
        )

    def _ensure_exact_online_state(self) -> None:
        if not hasattr(self, "_exact_online_store"):
            self._exact_online_store = ExactEpisodePrefixStore(is_init_key="is_init")
            self._exact_online_carry_refs = None
            self._exact_online_prefixes = {}
            self._exact_online_actor_bank = None
            self._exact_online_actor_offsets = {}
            self._exact_online_generation = None
            self._exact_online_live_generation = None
            self._exact_online_encoded_nodes = 0
            self._exact_online_cache_hits = 0
            self._exact_online_reused_live_nodes = 0
            self._exact_online_encoder_batches = 0
            self._exact_online_padded_nodes = 0
            self._exact_online_encode_seconds = 0.0

    def _reset_teacher_episode_cache_state(self) -> None:
        super()._reset_teacher_episode_cache_state()
        for name in (
            "_exact_online_store", "_exact_online_carry_refs",
            "_exact_online_prefixes", "_exact_online_generation",
            "_exact_online_live_generation",
            "_exact_online_actor_bank", "_exact_online_actor_offsets",
        ):
            self.__dict__.pop(name, None)

    def _exact_perception_inputs(self, td: TensorDict) -> dict[str, torch.Tensor]:
        # Preserve the actual normalized inputs of the frozen rollout VecNorm.
        # In particular, no depth float->uint8->float rounding is introduced.
        with torch.inference_mode(False):
            return {
                DEPTH_KEY: td[DEPTH_KEY].detach().clone(),
                OBS_KEY: td[OBS_KEY].detach().clone(),
                VEL_CMD_KEY: td[VEL_CMD_KEY].detach().clone(),
                PERCEPTION_OBJECT_GEO_ID_KEY: self._encode_replay_object_geo(td).detach().clone(),
                "is_init": td["is_init"].detach().bool().clone(),
            }

    @torch.no_grad()
    def _capture_exact_online_live_state(self, td: TensorDict) -> None:
        """Preserve EMA outputs before env.step can reset or strip next hx.

        The Student perception stack runs for every environment before choosing
        the executed action, including Teacher-controlled DAgger steps. Keep
        these fields during prefill too so the fixed rollout-buffer schema does
        not change at the prefill/main boundary. Only main rows are reused.
        """
        if not self._exact_online_replay_enabled():
            return
        with torch.inference_mode(False):
            td[EXACT_LIVE_HIDDEN_KEY] = torch.cat(
                (td["next", "depth_hx"].detach(), td["next", "adapt_hx"].detach()),
                dim=-1,
            )
            td[EXACT_LIVE_GENERATION_KEY] = torch.full(
                (*td.batch_size, 1), int(getattr(self, "_perception_ema_generation", 0)),
                dtype=torch.long, device=td[OBS_KEY].device,
            )

    def _live_perception_rollout(self, rollout):
        # These are cache provenance/output snapshots, not supervised inputs.
        # In particular, do not gather both full hidden tensors per minibatch.
        selected = super()._live_perception_rollout(rollout)
        if EXACT_LIVE_HIDDEN_KEY in selected or EXACT_LIVE_GENERATION_KEY in selected:
            return selected.exclude(EXACT_LIVE_HIDDEN_KEY, EXACT_LIVE_GENERATION_KEY)
        return selected

    @torch.no_grad()
    def _reuse_exact_online_live_prefixes(self, td, current, *, generation, live_generation):
        """Seed cache suffixes only from verified same-EMA live computation.

        Missing snapshots, changed generations, or a gap in an existing cache
        simply leave work for the full-reset encoder. No old hidden state or
        finite history substitute is accepted by that fallback.
        """
        actor = td.get(REPLAY_ACTOR_OBSERVATIONS_KEY, None)
        hidden = td.get(EXACT_LIVE_HIDDEN_KEY, None)
        stamps = td.get(EXACT_LIVE_GENERATION_KEY, None)
        n, t = map(int, td.batch_size)
        if stamps is None:
            return
        if stamps.shape != (n, t, 1) or stamps.dtype != torch.long:
            raise RuntimeError("Exact live perception generation stamps are invalid")
        # This is one device-to-host metadata transfer for the complete rollout,
        # never one synchronization per episode. A weight change during a live
        # rollout invalidates reuse of its outputs as one consistent prefix.
        if not bool(stamps.eq(generation[0]).all()):
            return False
        if actor is None or hidden is None:
            return
        hidden_width = int(self.depth_feature_dim) + int(self.cfg.latent_dim)
        if actor.shape != (n, t, self._q_actor_dim) or hidden.shape != (n, t, hidden_width):
            raise RuntimeError("Exact live perception snapshots are batch-misaligned")
        input_depth = td.get("depth_hx", None)
        input_adapt = td.get("adapt_hx", None)
        if input_depth is None or input_adapt is None:
            return False
        if input_depth.shape != (n, t, self.depth_feature_dim) or input_adapt.shape != (n, t, int(self.cfg.latent_dim)):
            raise RuntimeError("Exact live perception input hidden states are batch-misaligned")
        # The one-step live GRU relies on TensorDictPrimer clearing both input
        # states on reset; unlike the sequence GRU, it does not apply is_init.
        # Verify that contract once for the whole rollout before trusting a
        # live reset as an exact zero-state prefix boundary.
        reset = td["is_init"].reshape(n, t, -1).any(-1)
        nonzero_reset = reset & (
            input_depth.ne(0).any(-1) | input_adapt.ne(0).any(-1)
        )
        if bool(nonzero_reset.any()):
            return False
        self._prepare_exact_online_generation()
        cache = self._exact_online_prefixes
        trusted_carry = live_generation == generation
        segments, selected_rows, final_rows = [], [], []
        row_count = 0
        for env, env_refs in enumerate(current.tolist()):
            start = 0
            while start < t:
                uid, position = env_refs[start]
                stop = start + 1
                while stop < t and env_refs[stop][0] == uid:
                    stop += 1
                entry = cache.get(uid)
                cached = 0 if entry is None else entry.length
                # A real reset makes a new episode independent of any stale
                # incoming carry. Continuing episodes require a verified live
                # carry generation and a complete already-cached prefix.
                if (position == 0 or trusted_carry) and position <= cached < position + stop - start:
                    first = start + cached - position
                    length = stop - first
                    selected_rows.extend(range(env * t + first, env * t + stop))
                    final_rows.append(env * t + stop - 1)
                    segments.append((uid, row_count, length, position + stop - start))
                    row_count += length
                start = stop
        if not segments:
            return True
        # Compact in two batch operations. Per-episode entries are views, not
        # hundreds of independent GPU clones or references to a huge rollout.
        with torch.inference_mode(False):
            actors = actor.detach().reshape(n * t, self._q_actor_dim).index_select(
                0, torch.tensor(selected_rows, device=actor.device)
            )
            final_hidden = hidden.detach().reshape(n * t, hidden_width).index_select(
                0, torch.tensor(final_rows, device=hidden.device)
            )
        if not bool(torch.isfinite(actors).all() & torch.isfinite(final_hidden).all()):
            raise RuntimeError("Exact live perception snapshots contain NaN/Inf")
        for index, (uid, offset, length, stop) in enumerate(segments):
            entry = cache.setdefault(uid, _EncodedPrefix())
            entry.actor_chunks.append(actors[offset:offset + length])
            entry.depth_hx = final_hidden[index, :self.depth_feature_dim]
            entry.adapt_hx = final_hidden[index, self.depth_feature_dim:]
            entry.length = stop
        self._exact_online_actor_bank = None
        self._exact_online_reused_live_nodes += row_count
        return True

    def _prepare_raw_final_state(self, td, **kwargs):
        inputs = self._exact_perception_inputs(td) if self._exact_online_replay_enabled() else None
        result = super()._prepare_raw_final_state(td, **kwargs)
        if inputs is not None:
            result.update({_FINAL_INPUT_PREFIX + key: value for key, value in inputs.items()})
        return result

    @staticmethod
    def _exact_final_inputs(final):
        inputs = {
            key[len(_FINAL_INPUT_PREFIX):]: value
            for key, value in final.items() if key.startswith(_FINAL_INPUT_PREFIX)
        }
        if not inputs:
            raise RuntimeError("Exact replay is missing lossless final-observation inputs")
        return inputs

    @torch.no_grad()
    def _journal_exact_online_rollout(self, td):
        self._ensure_exact_online_state()
        generation = self._exact_cache_generation()
        live_generation = self._exact_online_live_generation
        if self._rollout_final_batch is None:
            raise RuntimeError("Exact online replay requires rollout-final observations")
        n, t = map(int, td.batch_size)
        raw = {key: value.cpu() for key, value in self._exact_perception_inputs(td).items()}
        final = {key: value.cpu() for key, value in self._exact_final_inputs(self._rollout_final_batch).items()}
        # Include the pending next observation as the last state. Next rollout
        # overlaps this node and must supply exactly the same inputs.
        sequence = {key: torch.cat((value, final[key].unsqueeze(1)), dim=1) for key, value in raw.items()}
        resets = sequence["is_init"].reshape(n, t + 1, -1).any(-1)
        refs = torch.empty((n, t + 1, 2), dtype=torch.long)
        previous = self._exact_online_carry_refs
        if previous is not None and previous.shape != (n, 2):
            raise RuntimeError("Exact replay environment count changed")
        store = self._exact_online_store
        for env in range(n):
            start = 0
            if previous is None:
                if not bool(resets[env, 0]):
                    raise RuntimeError("Exact replay cannot reconstruct an episode without its real reset")
                uid, offset = store.allocate_episode_uid(), 0
            else:
                uid, offset = map(int, previous[env].tolist())
            boundaries = resets[env, 1:].nonzero().flatten().add(1).tolist() + [t + 1]
            for stop in boundaries:
                store.append(uid, offset, {key: value[env, start:stop] for key, value in sequence.items()})
                refs[env, start:stop, 0] = uid
                refs[env, start:stop, 1] = torch.arange(offset, offset + stop - start)
                if stop < t + 1:
                    uid, offset = store.allocate_episode_uid(), 0
                start = stop
        current, successor = refs[:, :-1].clone(), refs[:, 1:].clone()
        # A timeout bootstrap consumes the pre-reset final input in the old
        # episode. The live reset observation belongs to a different episode.
        for captured in self._truncation_final_batches:
            inputs = {key: value.cpu() for key, value in self._exact_final_inputs(captured).items()}
            for row, flat in enumerate(captured["indices"].cpu().tolist()):
                env, step = divmod(int(flat), t)
                uid, position = map(int, current[env, step].tolist())
                store.append(uid, position + 1, {key: value[row:row + 1] for key, value in inputs.items()})
                successor[env, step] = torch.tensor((uid, position + 1))
        live_outputs_current = self._reuse_exact_online_live_prefixes(
            td, current, generation=generation, live_generation=live_generation,
        )
        self._exact_online_carry_refs = refs[:, -1].clone()
        # If an external caller changed EMA weights without refreshing carry,
        # a reset is the only proof that its live recurrent state is repaired.
        # Leaving this unset forces an exact refresh before further collection.
        self._exact_online_live_generation = (
            generation if live_outputs_current is not False and (
                live_generation == generation or bool(resets.any(-1).all())
            ) else None
        )
        return current.to(td[OBS_KEY].device), successor.to(td[OBS_KEY].device)

    def _dagger_transition_chunks(self, td):
        exact = self._exact_online_replay_enabled() and not self._teacher_prefill_active()
        references = self._journal_exact_online_rollout(td) if exact else None
        for transitions in super()._dagger_transition_chunks(td):
            if references is not None:
                env = transitions[_PREFILL_ENV_INDEX_KEY].long()
                step = transitions[_PREFILL_STEP_INDEX_KEY].long()
                transitions[EXACT_CURRENT_REF] = references[0][env, step]
                transitions[EXACT_NEXT_REF] = references[1][env, step]
            yield transitions

    def _extend_online_replays(self, transitions):
        result = super()._extend_online_replays(transitions)
        if self._exact_online_replay_enabled():
            self._prune_exact_online_history()
        return result

    def _prune_exact_online_history(self):
        self._ensure_exact_online_state()
        referenced = []
        for replay in (self.dagger_replay, self.student_replay):
            for key in (EXACT_CURRENT_REF, EXACT_NEXT_REF):
                if replay.size:
                    if key not in replay.data:
                        raise RuntimeError("Online replay lacks exact episode references")
                    referenced.append(replay.data[key][:replay.size, 0].cpu())
        if self._exact_online_carry_refs is not None:
            referenced.append(self._exact_online_carry_refs[:, 0])
        if referenced and all(
            value.device.type == "cpu" and value.dtype in (torch.int32, torch.int64)
            for value in referenced
        ):
            # Sort owned CPU integer storage, preserving every ID bit without
            # launching a Torch thread team or modifying replay reference views.
            episode_ids = np.concatenate([value.numpy() for value in referenced])
            episode_ids.sort(kind="stable")
            if episode_ids.size > 1:
                distinct = np.empty(episode_ids.size, dtype=np.bool_)
                distinct[0] = True
                np.not_equal(episode_ids[1:], episode_ids[:-1], out=distinct[1:])
                episode_ids = episode_ids[distinct]
            keep = set(episode_ids.tolist())
        else:
            keep = set(torch.cat(referenced).unique().tolist()) if referenced else set()
        self._exact_online_store.retain(keep)
        if set(self._exact_online_prefixes).difference(keep):
            self._exact_online_actor_bank = None
        self._exact_online_prefixes = {uid: value for uid, value in self._exact_online_prefixes.items() if uid in keep}

    def _exact_cache_generation(self):
        # VecNorm/geometry fingerprints are part of the policy input contract.
        # Encoder weights change only at the tracked perception EMA boundary.
        return (
            int(getattr(self, "_perception_ema_generation", 0)),
            getattr(self, "_replay_vecnorm_fingerprint", None),
            getattr(self, "_replay_object_geo_fingerprint", None),
        )

    def _prepare_exact_online_generation(self):
        """Invalidate derived state together; raw histories remain unchanged."""
        self._ensure_exact_online_state()
        generation = self._exact_cache_generation()
        if self._exact_online_generation != generation:
            self._exact_online_prefixes.clear()
            self._exact_online_actor_bank = None
            self._exact_online_actor_offsets.clear()
            self._exact_online_generation = generation
        return generation

    @torch.no_grad()
    def _exact_online_geometry_bank(self, geometry_ids, *, dtype):
        """Validate immutable host IDs before their staged device transfer.

        Only the standard lossless codebook decoder has this fast path. A
        subclass with a different geometry contract keeps its own decoder.
        The caller must transfer these very IDs and gather from the returned
        table before reusing its pinned slot; no device-side bounds sync is
        needed because neither the copied IDs nor the table can change here.
        """
        decoder = getattr(self._decode_replay_object_geo, "__func__", None)
        if decoder is not DistributionalTD3TeacherBC._decode_replay_object_geo:
            return None
        if geometry_ids.device.type != "cpu":
            raise ValueError("Exact geometry bounds must be checked on CPU")
        if geometry_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("object geometry replay IDs must be int32 or int64")
        self._ensure_replay_object_geo_codebook()
        bank = self._replay_object_geo_bank
        if bank is None or int(bank.shape[0]) < 1:
            raise RuntimeError("raw perception replay has no object geometry codebook")
        if bool((geometry_ids < 0).any()) or bool((geometry_ids >= bank.shape[0]).any()):
            raise IndexError("object geometry replay ID is outside the codebook")
        return self._replay_object_geo_bank_for(device=self.device, dtype=dtype)

    @torch.no_grad()
    def _encode_exact_online_prefixes(self, requests: dict[int, int]):
        """Evaluate every real timestep using bounded, padded batched inference."""
        self._prepare_exact_online_generation()
        cache, store = self._exact_online_prefixes, self._exact_online_store
        pending = []
        for uid, stop in requests.items():
            if stop < 0 or stop > store.length(uid):
                raise RuntimeError("Exact replay requested missing recurrent history")
            entry = cache.setdefault(uid, _EncodedPrefix())
            if entry.length < stop:
                pending.append(uid)
            else:
                self._exact_online_cache_hits += 1
        if not pending:
            return
        started = time.perf_counter()
        time_chunk = max(1, int(self.cfg.train_every))
        # Match the established Teacher encoder's frame workspace budget.
        # This multiplier controls batching only; no prefix history is cut.
        frame_budget = int(self.cfg.perception_encode_microbatch_size) * (
            int(getattr(self.cfg, "perception_replay_burn_in", 8)) + 2
        )
        batch_size = max(1, frame_budget // time_chunk)
        pending.sort(key=lambda uid: requests[uid] - cache[uid].length)
        cuda = torch.device(self.device).type == "cuda"
        # Two pinned staging slots overlap host gathering with GPU execution.
        # Before reusing a slot wait only for its H2D copies, not its forward.
        # This bounds pinned memory even when the CPU gets ahead of the GPU.
        staging = [{"buffers": {}, "event": None} for _ in range(2)]
        rounds = 0
        finite_checks = []
        with torch.inference_mode(False), torch.no_grad(), set_recurrent_mode(True):
            for start in range(0, len(pending), batch_size):
                group = pending[start:start + batch_size]
                while group:
                    lengths = [min(time_chunk, requests[uid] - cache[uid].length) for uid in group]
                    length = max(lengths)
                    slot = staging[rounds % len(staging)]
                    if slot["event"] is not None:
                        slot["event"].synchronize()
                    cpu_data = store.batch_slice(
                        [(uid, cache[uid].length, cache[uid].length + valid)
                         for uid, valid in zip(group, lengths)],
                        buffers=slot["buffers"], pin_memory=cuda,
                    )
                    # Transfer compact IDs, not per-frame decoded geometry.
                    # Bounds are checked on CPU against the same codebook.
                    geometry_bank = self._exact_online_geometry_bank(
                        cpu_data[PERCEPTION_OBJECT_GEO_ID_KEY],
                        dtype=cpu_data[OBS_KEY].dtype,
                    )
                    auxiliary = {
                        "_exact_lengths": torch.tensor(lengths, dtype=torch.long),
                        "_exact_valid_indices": torch.cat([
                            torch.arange(row * length, row * length + valid)
                            for row, valid in enumerate(lengths)
                        ]),
                    }
                    if geometry_bank is None:
                        auxiliary[OBJECT_GEO_KEY] = self._decode_replay_object_geo(
                            cpu_data.pop(PERCEPTION_OBJECT_GEO_ID_KEY),
                            device="cpu", dtype=cpu_data[OBS_KEY].dtype,
                        )
                    for key, value in auxiliary.items():
                        buffer = slot["buffers"].get(key)
                        if buffer is None or buffer.numel() < value.numel():
                            buffer = torch.empty(value.numel(), dtype=value.dtype, pin_memory=cuda)
                            slot["buffers"][key] = buffer
                        cpu_data[key] = buffer[:value.numel()].view(value.shape)
                        _copy_cpu_tensor_(cpu_data[key], value)
                    data = {key: value.to(self.device, non_blocking=cuda) for key, value in cpu_data.items()}
                    if cuda:
                        if slot["event"] is None:
                            slot["event"] = torch.cuda.Event()
                        slot["event"].record(torch.cuda.current_stream(self.device))
                    if geometry_bank is not None:
                        geometry_ids = data.pop(PERCEPTION_OBJECT_GEO_ID_KEY)
                        data[OBJECT_GEO_KEY] = geometry_bank.index_select(
                            0, geometry_ids.reshape(-1).long(),
                        ).reshape(*geometry_ids.shape, geometry_bank.shape[-1])
                    device_lengths = data.pop("_exact_lengths")
                    valid_indices = data.pop("_exact_valid_indices")
                    for key, width, attr in (
                        ("depth_hx", self.depth_feature_dim, "depth_hx"),
                        ("adapt_hx", int(self.cfg.latent_dim), "adapt_hx"),
                    ):
                        hidden = torch.stack([
                            getattr(cache[uid], attr) if getattr(cache[uid], attr) is not None
                            else torch.zeros(width, device=self.device, dtype=data[OBS_KEY].dtype)
                            for uid in group
                        ])
                        data[key] = hidden.unsqueeze(1).expand(-1, length, -1)
                    encoded = TensorDict(data, batch_size=(len(group), length), device=self.device)
                    # Padding follows real data only. Both GRUs return the
                    # raw hx at each row's true last state, not the padded end.
                    with exact_recurrent_lengths(lengths, device_lengths), exact_gru_cuda_graphs():
                        self.temporal_depth_gru_ema(encoded)
                        if bool(self.cfg.use_object_adapt):
                            self.object_adapt_ema(encoded)
                            self.object_pred_transform(encoded)
                        self.adapt_ema(encoded)
                    actor = torch.cat([encoded[key] for key in self.q_actor_keys], dim=-1)
                    if actor.shape != (len(group), length, self._q_actor_dim):
                        raise RuntimeError("Exact online replay produced invalid Actor inputs")
                    # Compact owned batch snapshots replace 3*B tiny clones.
                    # Padded values never enter the Actor bank or validation.
                    actor = actor.flatten(0, 1).index_select(0, valid_indices)
                    depth_hx = encoded["next", "depth_hx"][:, -1].detach().clone()
                    adapt_hx = encoded["next", "adapt_hx"][:, -1].detach().clone()
                    finite_checks.extend(torch.isfinite(value).all() for value in (actor, depth_hx, adapt_hx))
                    offset = 0
                    for index, (uid, valid) in enumerate(zip(group, lengths)):
                        entry = cache[uid]
                        entry.actor_chunks.append(actor[offset:offset + valid])
                        entry.depth_hx = depth_hx[index]
                        entry.adapt_hx = adapt_hx[index]
                        entry.length += valid
                        offset += valid
                    self._exact_online_actor_bank = None
                    self._exact_online_encoded_nodes += sum(lengths)
                    self._exact_online_padded_nodes = getattr(self, "_exact_online_padded_nodes", 0) + len(group) * length - sum(lengths)
                    rounds += 1
                    group = [uid for uid in group if cache[uid].length < requests[uid]]
            # One fail-closed synchronization per refresh, not per chunk/row.
            if not bool(torch.stack(finite_checks).all()):
                cache.clear()
                self._exact_online_actor_bank = None
                self._exact_online_actor_offsets.clear()
                raise RuntimeError("Exact online replay produced nonfinite Actor inputs or hidden states")
        self._exact_online_encoder_batches = getattr(self, "_exact_online_encoder_batches", 0) + rounds
        self._exact_online_encode_seconds = getattr(self, "_exact_online_encode_seconds", 0.0) + time.perf_counter() - started

    @staticmethod
    def _exact_prefix_requests(refs: torch.Tensor):
        requests = {}
        for uid, step in refs.detach().cpu().tolist():
            requests[uid] = max(requests.get(uid, 0), step + 1)
        return requests

    def _exact_ring_refs(self, replay, indices, *, next_state):
        key = EXACT_NEXT_REF if next_state else EXACT_CURRENT_REF
        if key not in replay.data:
            raise RuntimeError("Exact replay requires a new collection with complete raw episode histories")
        return replay.data[key].index_select(0, indices.to(replay.device).long()).cpu()

    @torch.inference_mode(False)
    @torch.no_grad()
    def _gather_exact_online_actor(self, refs):
        refs = refs.cpu()
        self._encode_exact_online_prefixes(self._exact_prefix_requests(refs))
        if self._exact_online_actor_bank is None:
            # A single flat GPU bank makes each sampled source one gather,
            # regardless of how many episodes its minibatch spans.
            chunks, offsets, offset = [], {}, 0
            for uid, entry in self._exact_online_prefixes.items():
                if entry.length:
                    offsets[uid] = offset
                    chunks.extend(entry.actor_chunks)
                    offset += entry.length
            bank = torch.cat(chunks, dim=0)
            for uid, offset in offsets.items():
                entry = self._exact_online_prefixes[uid]
                entry.actor_chunks = [bank[offset:offset + entry.length]]
            self._exact_online_actor_bank = bank
            self._exact_online_actor_offsets = offsets
        indices = torch.tensor(
            [self._exact_online_actor_offsets[uid] + step for uid, step in refs.tolist()],
            dtype=torch.long, device=self.device,
        )
        return self._exact_online_actor_bank.index_select(0, indices)

    def _prepare_exact_online_replay_cache(self, sample_plans):
        """Precompute the union of Actor/current and Critic/successor samples."""
        if not self._exact_online_replay_enabled() or sample_plans is None:
            return
        sample_plans = tuple(sample_plans)
        references = []
        for replay, name, next_state in (
            (self.dagger_replay, "student_indices", True),
            (self.student_replay, "pure_student_indices", True),
            (self.dagger_replay, "actor_indices", False),
            (self.student_replay, "actor_pure_student_indices", False),
        ):
            groups = [getattr(plan, name, None) for plan in sample_plans]
            groups = [indices.to(replay.device) for indices in groups
                      if indices is not None and indices.numel()]
            if groups:
                # One metadata transfer per source/purpose, not per update.
                references.append(self._exact_ring_refs(
                    replay, torch.cat(groups), next_state=next_state,
                ))
        if references:
            self._encode_exact_online_prefixes(self._exact_prefix_requests(torch.cat(references)))

    def _prepare_dagger_learning_batch(self, batch):
        prepared = super()._prepare_dagger_learning_batch(batch)
        if not self._exact_online_replay_enabled():
            return prepared
        required = (REPLAY_SAMPLE_IS_TEACHER_KEY, REPLAY_SAMPLE_IS_DAGGER_ENV_KEY, REPLAY_SAMPLE_PHYSICAL_INDEX_KEY)
        if any(key not in batch for key in required):
            raise RuntimeError("Exact online replay requires explicit sampled ring provenance")
        teacher, dagger, physical = (batch[key].detach().cpu().reshape(-1) for key in required)
        for output, next_state in (("observations", False), ("next_observations", True)):
            if output not in prepared:
                continue
            result = prepared[output].clone()
            for replay, mask in ((self.dagger_replay, ~teacher & dagger), (self.student_replay, ~teacher & ~dagger)):
                rows = mask.nonzero().flatten()
                if rows.numel():
                    refs = self._exact_ring_refs(replay, physical[rows], next_state=next_state)
                    result.index_copy_(0, rows.to(self.device), self._gather_exact_online_actor(refs).to(result.dtype))
            prepared[output] = result
        return prepared

    @torch.no_grad()
    def refresh_exact_perception_carry(self, carry):
        """Refresh hx BEFORE consuming carry's pending observation exactly once."""
        if not self._exact_online_replay_enabled() or self._teacher_prefill_active():
            return carry
        self._ensure_exact_online_state()
        refs = self._exact_online_carry_refs
        if refs is None:
            if not bool(carry["is_init"].all()):
                raise RuntimeError("Exact rollout has no reset-to-carry history")
            return carry
        if self._exact_online_live_generation == self._exact_cache_generation():
            return carry
        requests = {int(uid): int(step) for uid, step in refs.tolist()}
        self._encode_exact_online_prefixes(requests)
        carry = carry.clone(False)
        for key, width in (("depth_hx", self.depth_feature_dim), ("adapt_hx", int(self.cfg.latent_dim))):
            hidden = []
            for uid, stop in refs.tolist():
                entry = self._exact_online_prefixes[uid]
                if stop == 0:
                    hidden.append(torch.zeros(width, device=self.device))
                elif entry.length != stop:
                    raise RuntimeError("Exact carry refresh must stop before its pending observation")
                else:
                    hidden.append(getattr(entry, key))
            carry[key] = torch.stack(hidden)
        self._exact_online_live_generation = self._exact_cache_generation()
        return carry
