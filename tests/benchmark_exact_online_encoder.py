"""Bounded standalone CUDA benchmark; never imported by pytest collection.

Run from the repository root with the vaic Python environment, for example::

    python tests/benchmark_exact_online_encoder.py

Only explicit full-path desktop-process exceptions are accepted by the GPU
guard. This measures encoder execution, not end-to-end training throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tensordict import TensorDict
from torch import nn

from active_adaptation.learning.ppo.exact_online_perception import (
    ExactOnlinePerceptionReplayMixin,
    _EncodedPrefix,
)
from active_adaptation.learning.ppo.ppo_vel import DepthResidualGRUModule, set_recurrent_mode
from active_adaptation.learning.ppo.td3_bc_dagger import (
    DistributionalTD3TeacherBC,
    PERCEPTION_OBJECT_GEO_ID_KEY,
)
from test_tvkd_depth_residual import build_full_ppovel


def require_idle_cuda(allowed_process_names=()):
    """Check before creating a CUDA context; never stop another process."""
    def query(kind, fields):
        result = subprocess.run(
            ["nvidia-smi", "--id=0", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    free_mib = int(query("gpu", "memory.free")[0])
    if free_mib < 4096:
        raise RuntimeError(f"CUDA benchmark skipped: only {free_mib} MiB GPU memory is free")
    observed = []
    for row in query("compute-apps", "pid,process_name,used_memory"):
        pid, process_name, used_mib = [value.strip() for value in row.split(",", 2)]
        observed.append({"pid": int(pid), "name": process_name, "memory_mib": used_mib})
        if int(pid) != os.getpid() and process_name not in allowed_process_names:
            raise RuntimeError(f"CUDA benchmark skipped: another compute process is present: {row}")
    return {"free_mib_before_cuda": free_mib, "observed_compute_processes": observed}


class BenchmarkPolicy(ExactOnlinePerceptionReplayMixin, nn.Module):
    _ensure_replay_object_geo_codebook = DistributionalTD3TeacherBC._ensure_replay_object_geo_codebook
    _replay_object_geo_bank_for = DistributionalTD3TeacherBC._replay_object_geo_bank_for
    _decode_replay_object_geo = DistributionalTD3TeacherBC._decode_replay_object_geo

    def __init__(self, *, device="cuda:0", latent_dim=256):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1943)
            source = build_full_ppovel(residual=True, latent_dim=latent_dim)
            self.geometry = torch.randn(40, 384)
        self.cfg = source.cfg
        self.cfg.train_every = 32
        self.cfg.perception_encode_microbatch_size = 128
        self.cfg.perception_replay_burn_in = 8
        self.device = torch.device(device)
        self.depth_feature_dim = source.depth_feature_dim
        self.q_actor_keys = ("vel_command", "policy", "priv_pred")
        self._q_actor_dim = 15 + latent_dim
        self._perception_ema_generation = 0
        self._replay_vecnorm_fingerprint = "synthetic-frozen-normalized-inputs"
        self._replay_object_geo_fingerprint = "synthetic-lossless-geometry"
        for name in ("temporal_depth_gru_ema", "object_adapt_ema", "object_pred_transform", "adapt_ema"):
            setattr(self, name, getattr(source, name).to(self.device))
        core = next(module for module in self.adapt_ema.modules() if isinstance(module, DepthResidualGRUModule))
        generator = torch.Generator().manual_seed(1944)
        with torch.no_grad():
            core.depth_projection.weight.copy_(
                torch.randn(core.depth_projection.weight.shape, generator=generator) * 0.1
            )
        self.requires_grad_(False).eval()
        self._ensure_replay_object_geo_codebook()
        self._replay_object_geo_bank = self.geometry
        self._replay_object_geo_bank_generation = 1
        self._ensure_exact_online_state()

    def run_stack(self, td):
        self.temporal_depth_gru_ema(td)
        self.object_adapt_ema(td)
        self.object_pred_transform(td)
        self.adapt_ema(td)
        return td


def make_requests(policy, lengths=range(129, 169)):
    generator = torch.Generator().manual_seed(1945)
    requests = {}
    for index, length in enumerate(lengths):
        reset = torch.zeros(length, 1, dtype=torch.bool)
        reset[0] = True
        fields = {
            "depth": torch.randint(0, 101, (length, 1, 36, 64), generator=generator).float() / 100,
            "policy": torch.randn(length, 10, generator=generator),
            "vel_command": torch.randn(length, 5, generator=generator),
            PERCEPTION_OBJECT_GEO_ID_KEY: torch.full((length,), index, dtype=torch.long),
            "is_init": reset,
        }
        uid = policy._exact_online_store.allocate_episode_uid()
        policy._exact_online_store.append(uid, 0, fields)
        requests[uid] = length
    return requests


def _legacy_torch_slice(store, uid, start, stop):
    """Freeze the previous Torch-only host-copy path for a fair reference.

    The production store now avoids intra-op thread teams for these copies;
    reusing its slice method would retroactively optimize the old baseline.
    """
    episode = store._episode(uid)
    if not 0 <= start <= stop <= episode.length:
        raise RuntimeError("Legacy benchmark slice requested missing history")
    with torch.inference_mode(False), torch.no_grad():
        result = {
            key: torch.empty((stop - start, *shape), dtype=dtype, device="cpu")
            for key, (dtype, shape) in store._field_specs.items()
        }
        for chunk, offset, left, right in store._pieces(episode, start, stop):
            for key, value in chunk.items():
                result[key][offset:offset + right - left].copy_(value[left:right])
    return result


@torch.no_grad()
def baseline_minimum_length_encoder(policy, requests):
    """Previous exact algorithm: shortest tail, row clones, per-chunk sync."""
    cache = {uid: _EncodedPrefix() for uid in requests}
    pending = sorted(requests, key=requests.get)
    time_chunk = policy.cfg.train_every
    frame_budget = policy.cfg.perception_encode_microbatch_size * (policy.cfg.perception_replay_burn_in + 2)
    batch_size = max(1, frame_budget // time_chunk)
    rounds = 0
    with torch.inference_mode(False), set_recurrent_mode(True):
        for start in range(0, len(pending), batch_size):
            group = pending[start:start + batch_size]
            while group:
                length = min(time_chunk, min(requests[uid] - cache[uid].length for uid in group))
                rows = [_legacy_torch_slice(policy._exact_online_store, uid, cache[uid].length, cache[uid].length + length) for uid in group]
                data = {key: torch.stack([row[key] for row in rows]).to(policy.device) for key in rows[0]}
                data["object_geo_"] = policy._decode_replay_object_geo(
                    data.pop(PERCEPTION_OBJECT_GEO_ID_KEY), device=policy.device, dtype=data["policy"].dtype,
                )
                for key, width in (("depth_hx", policy.depth_feature_dim), ("adapt_hx", policy.cfg.latent_dim)):
                    hx = torch.stack([
                        getattr(cache[uid], key) if getattr(cache[uid], key) is not None
                        else torch.zeros(width, device=policy.device)
                        for uid in group
                    ])
                    data[key] = hx.unsqueeze(1).expand(-1, length, -1)
                encoded = policy.run_stack(TensorDict(data, [len(group), length], device=policy.device))
                actor = torch.cat([encoded[key] for key in policy.q_actor_keys], -1)
                if not bool(torch.isfinite(actor).all()):
                    raise RuntimeError("Baseline produced nonfinite actor inputs")
                for index, uid in enumerate(group):
                    entry = cache[uid]
                    entry.actor_chunks.append(actor[index].detach().clone())
                    entry.depth_hx = encoded["next", "depth_hx"][index, -1].detach().clone()
                    entry.adapt_hx = encoded["next", "adapt_hx"][index, -1].detach().clone()
                    entry.length += length
                group = [uid for uid in group if cache[uid].length < requests[uid]]
                rounds += 1
    return cache, {"encoder_calls": rounds, "valid_frames": sum(requests.values()), "padded_frames": 0}


def optimized_encoder(policy, requests):
    before = (policy._exact_online_encoder_batches, policy._exact_online_encoded_nodes, policy._exact_online_padded_nodes)
    policy._perception_ema_generation += 1  # Invalidate features without changing model weights.
    policy._encode_exact_online_prefixes(requests)
    after = (policy._exact_online_encoder_batches, policy._exact_online_encoded_nodes, policy._exact_online_padded_nodes)
    return policy._exact_online_prefixes, dict(zip(
        ("encoder_calls", "valid_frames", "padded_frames"),
        (new - old for old, new in zip(before, after)),
    ))


def snapshot(cache):
    return {
        uid: (torch.cat(entry.actor_chunks).cpu(), entry.depth_hx.cpu(), entry.adapt_hx.cpu())
        for uid, entry in cache.items()
    }


def value_differences(expected, actual):
    maxima = {"actor_inputs": 0.0, "depth_raw_hidden": 0.0, "adapt_raw_hidden": 0.0}
    assert expected.keys() == actual.keys()
    for uid in expected:
        for name, baseline, candidate in zip(maxima, expected[uid], actual[uid]):
            maxima[name] = max(maxima[name], (candidate - baseline).abs().max().item())
    return maxima


def assert_same_values(expected, actual):
    maxima = value_differences(expected, actual)
    for uid in expected:
        for name, baseline, candidate in zip(maxima, expected[uid], actual[uid]):
            try:
                torch.testing.assert_close(candidate, baseline)
            except AssertionError:
                print(json.dumps({"failed_episode": uid, "failed_field": name,
                                  "all_max_absolute_differences": maxima}), file=sys.stderr)
                raise
    return maxima


def timed_once(policy, requests, function):
    policy._exact_online_prefixes.clear()
    policy._exact_online_actor_bank = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    started = time.perf_counter()
    cache, counters = function(policy, requests)
    torch.cuda.synchronize()
    counters.update({
        "seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "incremental_peak_mib": (torch.cuda.max_memory_allocated() - initial) / 2**20,
    })
    return cache, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-process-name", action="append", default=[], help="Exact desktop executable path allowed by the idle-GPU guard")
    parser.add_argument("--strict-fp32", action="store_true", help="Diagnostic only: disable TF32 for both implementations without changing production settings")
    parser.add_argument("--report-precision-differences", action="store_true", help="Diagnostic only: report failed default-tolerance comparisons and timings without treating them as correctness passes")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Benchmark-process CPU thread count; production settings are not changed")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3, help="Measured repeats after one warm-up per implementation")
    args = parser.parse_args()
    guard = require_idle_cuda(args.allow_process_name)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.strict_fp32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    policy = BenchmarkPolicy()
    requests = make_requests(policy)
    functions = {"previous_exact": baseline_minimum_length_encoder, "optimized_exact": optimized_encoder}
    warmed = {}
    warm_timings = {}
    started = time.perf_counter()
    for name, function in functions.items():
        cache, warm_timings[name] = timed_once(policy, requests, function)
        print(json.dumps({"warmup": name, **warm_timings[name]}), file=sys.stderr)
        warmed[name] = snapshot(cache)
        del cache
    differences = value_differences(warmed["previous_exact"], warmed["optimized_exact"])
    default_tolerance_passed = True
    try:
        assert_same_values(warmed["previous_exact"], warmed["optimized_exact"])
    except AssertionError:
        if not args.report_precision_differences:
            raise
        default_tolerance_passed = False
    measurements = {name: [] for name in functions}
    for repetition in range(args.repeats):
        # Alternate order so neither implementation always gets the first run.
        order = list(functions) if repetition % 2 == 0 else list(reversed(functions))
        for name in order:
            if time.perf_counter() - started > 30:
                raise RuntimeError(f"Bounded benchmark exceeded 30 seconds; warm-ups={warm_timings}, measured={measurements}")
            cache, measured = timed_once(policy, requests, functions[name])
            actual = snapshot(cache)
            if default_tolerance_passed:
                assert_same_values(warmed["previous_exact"], actual)
            else:
                # Deliberately retain the failed comparison in the report;
                # do not silently loosen tolerances to label TF32 identical.
                observed = value_differences(warmed["previous_exact"], actual)
                differences = {key: max(differences[key], observed[key]) for key in differences}
            measurements[name].append(measured)
            del cache
    summaries = {
        name: {
            "median_seconds": statistics.median(value["seconds"] for value in runs),
            "runs": runs,
        }
        for name, runs in measurements.items()
    }
    print(json.dumps({
        "scope": "Isolated exact encoder only; not end-to-end training",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(), "guard": guard,
        "episodes": len(requests), "min_length": min(requests.values()), "max_length": max(requests.values()),
        "latent_dim": policy.cfg.latent_dim, "depth_shape": [1, 36, 64],
        "workspace_frame_budget": 1280, "max_time_chunk": 32,
        "correctness_max_absolute_difference": differences,
        "correctness_passed_default_tolerances": default_tolerance_passed,
        "warmups": warm_timings,
        "results": summaries,
        "median_encoder_speedup": summaries["previous_exact"]["median_seconds"] / summaries["optimized_exact"]["median_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
