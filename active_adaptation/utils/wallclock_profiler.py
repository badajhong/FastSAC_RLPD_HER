"""Explicit, short-window wall-clock profiling for training.

The profiler is deliberately inert unless ``wallclock_profile.enabled`` is
true.  During its configured rollout window it records host submission time
and CUDA-event time for named regions.  CUDA is synchronized exactly once at
the end of the window; no timing result is read inside the training loop.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import functools
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import torch


PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WallClockProfileConfig:
    enabled: bool = False
    label: str = ""
    phase: str = "physical"
    start_rollout: int = 0
    num_rollouts: int = 4
    cuda_events: bool = True
    torch_profiler_sync_audit: bool = False
    nvml_sample_interval_s: float = 0.2
    output_filename: str = "wallclock_profile.json"
    skip_final_evaluation: bool = False
    profile_final_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        value = {} if value is None else value
        config = cls(
            enabled=bool(value.get("enabled", False)),
            label=str(value.get("label", "")),
            phase=str(value.get("phase", "physical")),
            start_rollout=int(value.get("start_rollout", 0)),
            num_rollouts=int(value.get("num_rollouts", 4)),
            cuda_events=bool(value.get("cuda_events", True)),
            torch_profiler_sync_audit=bool(
                value.get("torch_profiler_sync_audit", False)
            ),
            nvml_sample_interval_s=float(
                value.get("nvml_sample_interval_s", 0.2)
            ),
            output_filename=str(
                value.get("output_filename", "wallclock_profile.json")
            ),
            skip_final_evaluation=bool(
                value.get("skip_final_evaluation", False)
            ),
            profile_final_checkpoint=bool(
                value.get("profile_final_checkpoint", False)
            ),
        )
        if config.phase not in ("physical", "teacher_prefill", "main_dagger"):
            raise ValueError(
                "wallclock_profile.phase must be physical, teacher_prefill, "
                "or main_dagger"
            )
        if config.start_rollout < 0:
            raise ValueError("wallclock_profile.start_rollout must be non-negative")
        if config.num_rollouts < 1:
            raise ValueError("wallclock_profile.num_rollouts must be positive")
        if config.nvml_sample_interval_s <= 0.0:
            raise ValueError(
                "wallclock_profile.nvml_sample_interval_s must be positive"
            )
        if Path(config.output_filename).name != config.output_filename:
            raise ValueError(
                "wallclock_profile.output_filename must be a plain filename"
            )
        return config


@dataclass
class _TimingRecord:
    cpu_ms: float
    start_event: Any = None
    end_event: Any = None


class _NvmlSampler:
    """Best-effort host-side utilization sampling with no CUDA API sync."""

    def __init__(self, device_index: int, interval_s: float):
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self.samples: list[tuple[float, float]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        read_sample: Callable[[], tuple[float, float]]
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

            def read_sample() -> tuple[float, float]:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return float(utilization.gpu), float(memory.used)

        except Exception as binding_exc:  # pragma: no cover - host dependent
            # NVIDIA's driver ships libnvidia-ml even when the optional Python
            # bindings are absent. This fallback keeps the training dependency
            # free and, like nvidia-smi, does not touch the CUDA work queues.
            try:
                import ctypes

                library = ctypes.CDLL("libnvidia-ml.so.1")

                class Utilization(ctypes.Structure):
                    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

                class Memory(ctypes.Structure):
                    _fields_ = [
                        ("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong),
                    ]

                library.nvmlInit_v2.restype = ctypes.c_int
                library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                    ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
                library.nvmlDeviceGetUtilizationRates.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(Utilization),
                ]
                library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
                library.nvmlDeviceGetMemoryInfo.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(Memory),
                ]
                library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
                if library.nvmlInit_v2() != 0:
                    raise RuntimeError("nvmlInit_v2 failed")
                ctypes_handle = ctypes.c_void_p()
                if (
                    library.nvmlDeviceGetHandleByIndex_v2(
                        self.device_index, ctypes.byref(ctypes_handle)
                    )
                    != 0
                ):
                    raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")

                def read_sample() -> tuple[float, float]:
                    utilization = Utilization()
                    memory = Memory()
                    if (
                        library.nvmlDeviceGetUtilizationRates(
                            ctypes_handle, ctypes.byref(utilization)
                        )
                        != 0
                    ):
                        raise RuntimeError("nvmlDeviceGetUtilizationRates failed")
                    if (
                        library.nvmlDeviceGetMemoryInfo(
                            ctypes_handle, ctypes.byref(memory)
                        )
                        != 0
                    ):
                        raise RuntimeError("nvmlDeviceGetMemoryInfo failed")
                    return float(utilization.gpu), float(memory.used)

            except Exception as ctypes_exc:
                self.error = (
                    f"bindings={type(binding_exc).__name__}: {binding_exc}; "
                    f"ctypes={type(ctypes_exc).__name__}: {ctypes_exc}"
                )
                return

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(read_sample())
                except Exception as exc:  # pragma: no cover - host dependent
                    self.error = f"{type(exc).__name__}: {exc}"
                    return
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(
            target=sample, name="wallclock-profile-nvml", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))
        result: dict[str, Any] = {
            "sample_count": len(self.samples),
            "error": self.error,
        }
        if self.samples:
            utilization = [sample[0] for sample in self.samples]
            memory = [sample[1] for sample in self.samples]
            result.update(
                {
                    "gpu_utilization_mean_percent": sum(utilization)
                    / len(utilization),
                    "gpu_utilization_peak_percent": max(utilization),
                    "device_memory_peak_bytes": int(max(memory)),
                }
            )
        return result


class WallClockProfiler:
    """Collect named timings only inside one explicit rollout window."""

    def __init__(
        self,
        config: WallClockProfileConfig,
        *,
        device: torch.device | str,
        output_dir: str | os.PathLike[str],
        metadata: Mapping[str, Any] | None = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.output_path = Path(output_dir) / config.output_filename
        self.metadata = dict(metadata or {})
        self.active = False
        self.finished = False
        self._window_started = False
        self._window_complete = False
        self._window_start_time: float | None = None
        self._window_end_time: float | None = None
        self._window_start_epoch_s: float | None = None
        self._window_end_epoch_s: float | None = None
        self._records: dict[str, list[_TimingRecord]] = defaultdict(list)
        self._counters: dict[str, float] = defaultdict(float)
        self._block_stack: list[str] = []
        self._on_start: list[Callable[[], None]] = []
        self._on_finish: list[Callable[[], None]] = []
        self._summary: dict[str, Any] | None = None
        self.explicit_cuda_synchronizations = 0
        self.cuda_event_timing = bool(
            config.enabled
            and config.cuda_events
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._nvml: _NvmlSampler | None = None
        self._nvml_result: dict[str, Any] = {
            "sample_count": 0,
            "error": "profile window did not start",
        }
        self._peak_memory: dict[str, int] = {}
        self._torch_profiler = None
        self._sync_audit: dict[str, Any] = {
            "enabled": bool(config.torch_profiler_sync_audit),
            "implicit_host_scalar_extractions": None,
            "observed_cuda_synchronization_events": None,
        }

    def add_window_callbacks(
        self,
        *,
        on_start: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        if on_start is not None:
            self._on_start.append(on_start)
        if on_finish is not None:
            self._on_finish.append(on_finish)

    def begin_rollout(
        self,
        rollout: int,
        *,
        phase: str = "physical",
        phase_rollout: int | None = None,
    ) -> bool:
        if not self.config.enabled or self.finished:
            self.active = False
            return False
        if self.config.phase == "physical":
            coordinate = int(rollout)
        else:
            if phase != self.config.phase:
                self.active = False
                return False
            if phase_rollout is None:
                raise ValueError("phase-aware profiling requires phase_rollout")
            coordinate = int(phase_rollout)
        stop = self.config.start_rollout + self.config.num_rollouts
        self.active = self.config.start_rollout <= coordinate < stop
        if self.active and not self._window_started:
            self._window_started = True
            self._window_start_time = time.perf_counter()
            self._window_start_epoch_s = time.time()
            if self.config.torch_profiler_sync_audit:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.cuda_event_timing:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                self._torch_profiler = torch.profiler.profile(
                    activities=activities,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                self._torch_profiler.__enter__()
            if self.cuda_event_timing:
                torch.cuda.reset_peak_memory_stats(self.device)
                device_index = (
                    torch.cuda.current_device()
                    if self.device.index is None
                    else self.device.index
                )
                self._nvml = _NvmlSampler(
                    device_index, self.config.nvml_sample_interval_s
                )
                self._nvml.start()
            for callback in self._on_start:
                callback()
        return self.active

    def end_rollout(
        self, rollout: int, *, phase_rollout: int | None = None
    ) -> None:
        if not self.active:
            return
        self.increment("profiled_rollouts")
        stop = self.config.start_rollout + self.config.num_rollouts - 1
        coordinate = (
            int(rollout)
            if self.config.phase == "physical"
            else int(phase_rollout) if phase_rollout is not None else -1
        )
        if coordinate >= stop:
            self._window_complete = True
            self.finish()
        else:
            self.active = False

    def increment(self, name: str, amount: float = 1.0) -> None:
        if self.active:
            self._counters[name] += float(amount)

    def set_metadata(self, name: str, value: Any) -> None:
        """Attach cheap audit state captured outside the timed window."""
        self.metadata[name] = value

    def in_block(self, name: str) -> bool:
        return name in self._block_stack

    @contextmanager
    def block(self, name: str):
        """Record a region; callers should avoid invoking this while inactive."""
        if not self.active:
            yield
            return
        start_event = end_event = None
        if self.cuda_event_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        self._block_stack.append(name)
        try:
            yield
        finally:
            cpu_ms = (time.perf_counter() - start) * 1000.0
            popped = self._block_stack.pop()
            if popped != name:  # pragma: no cover - defensive invariant
                raise RuntimeError("wall-clock profile block stack was corrupted")
            if end_event is not None:
                end_event.record()
            self._records[name].append(
                _TimingRecord(cpu_ms, start_event, end_event)
            )

    @contextmanager
    def external_cpu_block(self, name: str):
        """Measure an explicitly requested I/O probe outside the CUDA window."""
        if not self.config.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._records[name].append(
                _TimingRecord((time.perf_counter() - start) * 1000.0)
            )
            if self.finished:
                self._summary = self._build_summary()
                self._write_summary()

    def finish(self) -> dict[str, Any]:
        if self.finished:
            return dict(self._summary or {})
        if not self.config.enabled:
            self.finished = True
            self._summary = self._build_summary()
            return dict(self._summary)
        self.active = False
        for callback in reversed(self._on_finish):
            callback()
        if self._torch_profiler is not None:
            self._torch_profiler.__exit__(None, None, None)
            host_scalar_extractions = 0
            cuda_sync_events = 0
            sync_event_counts: dict[str, int] = {}
            for event in self._torch_profiler.key_averages():
                key = str(event.key)
                count = int(event.count)
                if "_local_scalar_dense" in key:
                    host_scalar_extractions += count
                if any(
                    token in key
                    for token in (
                        "cudaDeviceSynchronize",
                        "cudaStreamSynchronize",
                        "cudaEventSynchronize",
                    )
                ):
                    cuda_sync_events += count
                    sync_event_counts[key] = count
            self._sync_audit = {
                "enabled": True,
                "implicit_host_scalar_extractions": host_scalar_extractions,
                "observed_cuda_synchronization_events": cuda_sync_events,
                "cuda_event_counts": sync_event_counts,
                "note": (
                    "The optional torch.profiler collection boundary can itself "
                    "synchronize; use a separate one-rollout audit rather than "
                    "the low-overhead A--E timing run."
                ),
            }
        if self._nvml is not None:
            self._nvml_result = self._nvml.stop()
        if self.cuda_event_timing and self._records:
            # This is the sole explicit synchronization in the event-timed
            # window. Event elapsed times are read only after it completes.
            torch.cuda.synchronize(self.device)
            self.explicit_cuda_synchronizations += 1
            self._peak_memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
            }
        self._window_end_time = time.perf_counter()
        self._window_end_epoch_s = time.time()
        self.finished = True
        self._summary = self._build_summary()
        self._write_summary()
        print(
            f"Wall-clock profile JSON: {self.output_path.resolve()}", flush=True
        )
        return dict(self._summary)

    def _write_summary(self) -> None:
        if self._summary is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, self.output_path)

    def _build_summary(self) -> dict[str, Any]:
        duration = None
        if self._window_start_time is not None and self._window_end_time is not None:
            duration = self._window_end_time - self._window_start_time
        window_ms = None if duration is None else duration * 1000.0
        blocks: dict[str, Any] = {}
        for name, records in sorted(self._records.items()):
            cpu = [record.cpu_ms for record in records]
            cuda = [
                float(record.start_event.elapsed_time(record.end_event))
                for record in records
                if record.start_event is not None
            ]
            entry: dict[str, Any] = {
                "calls": len(records),
                "cpu_total_ms": sum(cpu),
                "cpu_mean_ms": sum(cpu) / len(cpu),
                "cpu_max_ms": max(cpu),
            }
            if window_ms:
                entry["cpu_percent_of_window"] = 100.0 * sum(cpu) / window_ms
            if cuda:
                entry.update(
                    {
                        "cuda_total_ms": sum(cuda),
                        "cuda_mean_ms": sum(cuda) / len(cuda),
                        "cuda_max_ms": max(cuda),
                    }
                )
                if window_ms:
                    entry["cuda_percent_of_window"] = (
                        100.0 * sum(cuda) / window_ms
                    )
            blocks[name] = entry
        profiled_rollouts = self._counters.get("profiled_rollouts", 0.0)
        derived: dict[str, float] = {}
        if duration and profiled_rollouts:
            derived["seconds_per_rollout"] = duration / profiled_rollouts
        if duration and self._counters.get("environment_states", 0.0):
            derived["environment_states_per_second"] = (
                self._counters["environment_states"] / duration
            )
        if duration and self._counters.get("environment_control_steps", 0.0):
            derived["environment_control_steps_per_second"] = (
                self._counters["environment_control_steps"] / duration
            )
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "label": self.config.label,
            "metadata": self.metadata,
            "window": {
                "phase": self.config.phase,
                "start_rollout": self.config.start_rollout,
                "num_rollouts": self.config.num_rollouts,
                "completed": self._window_complete,
                "wall_seconds": duration,
                "start_epoch_s": self._window_start_epoch_s,
                "end_epoch_s": self._window_end_epoch_s,
                "start_utc": self._utc_timestamp(self._window_start_epoch_s),
                "end_utc": self._utc_timestamp(self._window_end_epoch_s),
            },
            "blocks": blocks,
            "block_percent_note": (
                "Named regions may be nested or overlap; percentages are not "
                "expected to sum to 100."
            ),
            "counters": dict(sorted(self._counters.items())),
            "derived": derived,
            "cuda": {
                "event_timing": self.cuda_event_timing,
                "explicit_synchronizations": self.explicit_cuda_synchronizations,
                "known_synchronization_count": (
                    self.explicit_cuda_synchronizations
                    + int(
                        self._sync_audit.get(
                            "observed_cuda_synchronization_events", 0
                        )
                        or 0
                    )
                ),
                **self._peak_memory,
            },
            "synchronization_audit": self._sync_audit,
            "nvml": self._nvml_result,
        }

    @staticmethod
    def _utc_timestamp(epoch_s: float | None) -> str | None:
        if epoch_s is None:
            return None
        return datetime.datetime.fromtimestamp(
            epoch_s, tz=datetime.timezone.utc
        ).isoformat()

    @property
    def summary(self) -> Mapping[str, Any] | None:
        return self._summary


@dataclass(frozen=True)
class MethodProfileSpec:
    method: str
    block: str
    counter: str | None = None
    require_parent: str | None = None
    row_counter: str | None = None
    rows_from_result: bool = False


def _batch_rows(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim else 1
    if isinstance(value, Mapping):
        for child in value.values():
            rows = _batch_rows(child)
            if rows:
                return rows
    if isinstance(value, (tuple, list)):
        for child in value:
            rows = _batch_rows(child)
            if rows:
                return rows
    batch_size = getattr(value, "batch_size", None)
    if batch_size is not None and len(batch_size):
        rows = 1
        for dimension in batch_size:
            rows *= int(dimension)
        return rows
    return 0


class MethodInstrumentation:
    """Install instance-local wrappers only while the profile window is live."""

    def __init__(
        self,
        profiler: WallClockProfiler,
        targets: list[tuple[Any, MethodProfileSpec]],
    ):
        self.profiler = profiler
        self.targets = targets
        self._originals: list[tuple[Any, str, Any, bool]] = []

    def install(self) -> None:
        if self._originals:
            return
        for target, spec in self.targets:
            if target is None or not hasattr(target, spec.method):
                continue
            original = getattr(target, spec.method)
            had_instance_attribute = spec.method in getattr(target, "__dict__", {})
            if not callable(original):
                continue

            @functools.wraps(original)
            def profiled(*args, __original=original, __spec=spec, **kwargs):
                should_record = self.profiler.active and (
                    __spec.require_parent is None
                    or self.profiler.in_block(__spec.require_parent)
                )
                if not should_record:
                    return __original(*args, **kwargs)
                with self.profiler.block(__spec.block):
                    result = __original(*args, **kwargs)
                if __spec.counter is not None:
                    self.profiler.increment(__spec.counter)
                if __spec.row_counter is not None:
                    source = (
                        result
                        if __spec.rows_from_result
                        else args[0]
                        if args
                        else None
                    )
                    self.profiler.increment(
                        __spec.row_counter, _batch_rows(source)
                    )
                return result

            setattr(target, spec.method, profiled)
            self._originals.append(
                (target, spec.method, original, had_instance_attribute)
            )

    def restore(self) -> None:
        for target, method, original, had_instance_attribute in reversed(
            self._originals
        ):
            if had_instance_attribute:
                setattr(target, method, original)
            else:
                delattr(target, method)
        self._originals.clear()


def instrument_training_policy(
    policy: Any, profiler: WallClockProfiler
) -> MethodInstrumentation:
    """Register the stable TVKD/FastSAC/TD3 profiling surface.

    Missing methods are intentionally ignored, allowing the same trainer to
    profile baseline PPO and every A--E isolation configuration.
    """
    specs = (
        MethodProfileSpec(
            "train_op",
            "training_operation",
            "training_operation_calls",
        ),
        MethodProfileSpec(
            "_student_raw_action_proposal",
            "student_inference",
            "student_inference_calls",
            row_counter="student_inference_states",
        ),
        MethodProfileSpec(
            "_teacher_action",
            "teacher_inference",
            "teacher_inference_calls",
            row_counter="teacher_inference_states",
        ),
        MethodProfileSpec(
            "_batched_frozen_teacher_value",
            "frozen_teacher_rollout_grid_inference",
            "frozen_teacher_rollout_grid_calls",
            row_counter="frozen_teacher_rollout_grid_states",
        ),
        MethodProfileSpec(
            "get_frozen_teacher_value",
            "teacher_value_inside_tvkd_target",
            "teacher_value_target_calls",
            require_parent="c51_q_forward_backward",
            row_counter="teacher_value_target_states",
        ),
        MethodProfileSpec(
            "_sample_balanced_q_batch",
            "replay_sampling",
            "q_samples",
            row_counter="q_sample_rows",
            rows_from_result=True,
        ),
        MethodProfileSpec(
            "_sample_actor_batch",
            "replay_sampling",
            "actor_samples",
            row_counter="actor_sample_rows",
            rows_from_result=True,
        ),
        MethodProfileSpec(
            "_sample_four_way_perception_batch",
            "replay_sampling",
            "perception_samples",
            row_counter="perception_sample_rows",
            rows_from_result=True,
        ),
        MethodProfileSpec(
            "_prefetch_curriculum_sample_plans",
            "replay_sample_plan_prefetch",
            "replay_sample_plan_prefetch_calls",
        ),
        MethodProfileSpec(
            "_ensure_teacher_actor_cache_current",
            "teacher_actor_cache_ensure",
            "teacher_actor_cache_ensure_calls",
        ),
        MethodProfileSpec(
            "_rebuild_teacher_actor_cache",
            "teacher_actor_cache_rebuild",
            "teacher_actor_cache_rebuild_calls",
        ),
        MethodProfileSpec(
            "_prepare_dagger_learning_batch",
            "replay_batch_preparation",
            "replay_batch_preparation_calls",
            row_counter="replay_batch_preparation_rows",
        ),
        MethodProfileSpec(
            "_record_replay_mix_batch",
            "replay_mix_diagnostics",
            "replay_mix_diagnostic_calls",
        ),
        MethodProfileSpec(
            "_replay_mix_metrics",
            "replay_mix_metrics",
            "replay_mix_metric_calls",
        ),
        MethodProfileSpec(
            "_update_failure_phase_histogram",
            "failure_phase_bookkeeping",
            "failure_phase_bookkeeping_calls",
        ),
        MethodProfileSpec(
            "_student_teacher_td_residual_grid",
            "teacher_value_rollout_grid",
            "teacher_value_rollout_grid_calls",
        ),
        MethodProfileSpec(
            "_record_teacher_phase_match_distances",
            "teacher_phase_match_diagnostics",
            "teacher_phase_match_diagnostic_calls",
        ),
        MethodProfileSpec(
            "_reencode_perception_windows",
            "perception_reencode",
            "perception_reencode_calls",
            row_counter="perception_reencode_windows",
        ),
        MethodProfileSpec(
            "_critic_update", "c51_q_forward_backward", "critic_updates"
        ),
        MethodProfileSpec(
            "_actor_update", "actor_forward_backward", "actor_updates"
        ),
        MethodProfileSpec(
            "train_adapt", "perception_forward_backward", "perception_updates"
        ),
        MethodProfileSpec(
            "_run_teacher_perception_warmup",
            "perception_forward_backward",
            "perception_warmup_runs",
        ),
    )
    targets = [(policy, spec) for spec in specs]
    replay_spec = MethodProfileSpec(
        "extend",
        "replay_insertion",
        "replay_insert_calls",
        row_counter="replay_insert_rows",
    )
    for replay_name in ("dagger_replay", "q_teacher_replay"):
        targets.append((getattr(policy, replay_name, None), replay_spec))
    instrumentation = MethodInstrumentation(profiler, targets)
    profiler.add_window_callbacks(
        on_start=instrumentation.install, on_finish=instrumentation.restore
    )
    return instrumentation


__all__ = [
    "MethodInstrumentation",
    "MethodProfileSpec",
    "PROFILE_SCHEMA_VERSION",
    "WallClockProfileConfig",
    "WallClockProfiler",
    "instrument_training_policy",
]
