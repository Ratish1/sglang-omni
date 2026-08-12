# SPDX-License-Identifier: Apache-2.0
"""Attribute host-blocking CUDA synchronizations in Torch Profiler traces.

The input is a Kineto Chrome trace (plain JSON or ``.json.gz``).  The parser
streams ``traceEvents`` instead of loading the complete trace DOM.  Analysis is
strictly per file/process: timestamps from independent traces are never mixed.

The post-sync bubble is reported only when both endpoints are present:

* ``g0`` is the correlated transfer completion, or the latest event ending on
  an explicitly identified stream before the host wait returns.
* ``g1`` is a GPU event correlated with the first later CUDA launch on the same
  host thread.

The output records the attribution method so correlation-backed measurements
remain distinguishable from time/interval heuristics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

_TRACE_EVENTS_KEY = '"traceEvents"'
_DEFAULT_CHUNK_SIZE = 1024 * 1024
_MAX_EVENT_JSON_SIZE = 64 * 1024 * 1024
_API_TIMESTAMP_EPSILON_US = 1e-3
_SEMANTIC_RANGE_PREFIX = "qwen3_tts."
_TARGET_CPU_OPS = {
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::nonzero",
}
_RUNTIME_LAUNCH_TOKENS = ("launch", "memcpy", "memset")
_NORMALIZE_KEY_RE = re.compile(r"[^a-z0-9]")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    category: str
    pid: int | str
    tid: int | str
    ts_us: float
    dur_us: float
    args: dict[str, Any]

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us

    @property
    def thread_key(self) -> tuple[int | str, int | str]:
        return self.pid, self.tid


@dataclass(frozen=True, slots=True)
class HostLaunch:
    host: TraceEvent
    gpu: TraceEvent


@dataclass(slots=True)
class _Context:
    semantic_ranges: list[TraceEvent]
    cpu_ops: list[TraceEvent]
    python_frames: list[TraceEvent]


@dataclass(frozen=True, slots=True)
class _QueueTimeline:
    launch_times_us: tuple[float, ...]
    prefix_max_gpu_end_us: tuple[float, ...]

    def queued_end_at(self, host_time_us: float) -> float | None:
        index = bisect.bisect_right(self.launch_times_us, host_time_us) - 1
        if index < 0:
            return None
        return self.prefix_max_gpu_end_us[index]


@dataclass(frozen=True, slots=True)
class _BusyTimeline:
    starts_us: tuple[float, ...]
    ends_us: tuple[float, ...]

    def idle_between(self, start_us: float, end_us: float) -> float:
        """Return time not covered by any GPU event in a bounded interval."""
        if end_us <= start_us:
            return 0.0
        index = bisect.bisect_right(self.ends_us, start_us)
        busy_us = 0.0
        cursor = start_us
        while index < len(self.starts_us) and self.starts_us[index] < end_us:
            interval_start = max(start_us, self.starts_us[index])
            interval_end = min(end_us, self.ends_us[index])
            if interval_end > max(cursor, interval_start):
                busy_us += interval_end - max(cursor, interval_start)
                cursor = interval_end
            index += 1
        return max(0.0, end_us - start_us - busy_us)

    def idle_immediately_before(self, end_us: float, *, floor_us: float) -> float:
        """Return the final globally idle segment ending at ``end_us``."""
        if end_us <= floor_us:
            return 0.0
        index = bisect.bisect_left(self.starts_us, end_us) - 1
        if index < 0:
            return end_us - floor_us
        prior_end = self.ends_us[index]
        if prior_end >= end_us:
            return 0.0
        return max(0.0, end_us - max(floor_us, prior_end))


def _open_trace(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def iter_trace_events(
    path: str | Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield objects from the top-level ``traceEvents`` array incrementally."""
    trace_path = Path(path)
    decoder = json.JSONDecoder()
    with _open_trace(trace_path) as stream:
        buffer = ""
        while True:
            key_index = buffer.find(_TRACE_EVENTS_KEY)
            if key_index >= 0:
                buffer = buffer[key_index + len(_TRACE_EVENTS_KEY) :]
                break
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"{trace_path} has no top-level traceEvents array")
            buffer += chunk
            if (
                buffer.find(_TRACE_EVENTS_KEY) < 0
                and len(buffer) > len(_TRACE_EVENTS_KEY) * 2
            ):
                buffer = buffer[-len(_TRACE_EVENTS_KEY) * 2 :]

        while "[" not in buffer:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"{trace_path} has an incomplete traceEvents key")
            buffer += chunk
        buffer = buffer.split("[", 1)[1]

        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                    if len(buffer) > _MAX_EVENT_JSON_SIZE:
                        raise ValueError(
                            f"{trace_path} contains a trace event larger than "
                            f"{_MAX_EVENT_JSON_SIZE} bytes"
                        ) from exc
                    continue
                raise ValueError(
                    f"{trace_path} has malformed or truncated traceEvents JSON"
                ) from exc
            buffer = buffer[end:]
            if isinstance(value, dict):
                yield value


def _as_event(raw: dict[str, Any]) -> TraceEvent | None:
    if raw.get("ph") != "X":
        return None
    try:
        ts_us = float(raw["ts"])
        dur_us = float(raw.get("dur", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    args = raw.get("args")
    return TraceEvent(
        name=str(raw.get("name", "")),
        category=str(raw.get("cat", raw.get("category", ""))),
        pid=raw.get("pid", "unknown"),
        tid=raw.get("tid", "unknown"),
        ts_us=ts_us,
        dur_us=max(dur_us, 0.0),
        args=args if isinstance(args, dict) else {},
    )


def _normalized_args(event: TraceEvent) -> dict[str, Any]:
    return {
        _NORMALIZE_KEY_RE.sub("", str(key).lower()): value
        for key, value in event.args.items()
    }


def _correlation_id(event: TraceEvent) -> int | str | None:
    args = _normalized_args(event)
    for key in ("correlation", "correlationid", "linkedcorrelationid"):
        value = args.get(key)
        if value not in (None, "", 0, "0"):
            return str(value)
    return None


def _stream_id(event: TraceEvent, *, gpu_event: bool = False) -> str | None:
    args = _normalized_args(event)
    for key in ("stream", "streamid"):
        value = args.get(key)
        if value is not None:
            return str(value)
    if gpu_event and event.tid != "unknown":
        return str(event.tid)
    return None


def _device_id(event: TraceEvent) -> str | None:
    args = _normalized_args(event)
    for key in ("device", "deviceid"):
        value = args.get(key)
        if value is not None:
            return str(value)
    return None


def _byte_count(event: TraceEvent) -> int | None:
    args = _normalized_args(event)
    for key, value in args.items():
        if key not in {"bytes", "bytecount", "sizebytes"}:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _is_sync_api(event: TraceEvent) -> bool:
    name = event.name.lower()
    category = event.category.lower()
    return (
        name.startswith(("cuda", "hip"))
        and "synchronize" in name
        and ("runtime" in category or "driver" in category or not category)
    )


def _is_runtime_launch(event: TraceEvent) -> bool:
    name = event.name.lower()
    category = event.category.lower()
    return (
        ("runtime" in category or "driver" in category)
        and not _is_sync_api(event)
        and any(token in name for token in _RUNTIME_LAUNCH_TOKENS)
    )


def _is_runtime_api(event: TraceEvent) -> bool:
    category = event.category.lower()
    return "runtime" in category or "driver" in category


def _is_async_runtime_memcpy(event: TraceEvent) -> bool:
    return "memcpyasync" in event.name.lower()


def _is_gpu_event(event: TraceEvent) -> bool:
    category = event.category.lower()
    name = event.name.lower()
    if category in {"kernel", "gpu_memcpy", "gpu_memset", "cuda_kernel"}:
        return True
    if "gpu" in category and ("kernel" in category or "mem" in category):
        return True
    return name.startswith("memcpy ") and "cpu" not in category


def _is_gpu_copy(event: TraceEvent) -> bool:
    return _is_gpu_event(event) and "memcpy" in event.name.lower()


def _is_target_cpu_op(event: TraceEvent) -> bool:
    return event.name in _TARGET_CPU_OPS or (
        "cpu" in event.category.lower() and "memcpy" in event.name.lower()
    )


def _transfer_direction(event: TraceEvent | None) -> str | None:
    if event is None:
        return None
    compact = event.name.lower().replace(" ", "")
    if "htod" in compact or "hosttodevice" in compact:
        return "HtoD"
    if "dtoh" in compact or "devicetohost" in compact:
        return "DtoH"
    if "dtod" in compact or "devicetodevice" in compact:
        return "DtoD"
    return None


def _event_dict(event: TraceEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "name": event.name,
        "category": event.category,
        "pid": event.pid,
        "tid": event.tid,
        "start_us": event.ts_us,
        "end_us": event.end_us,
        "duration_us": event.dur_us,
        "correlation_id": _correlation_id(event),
        "stream": _stream_id(event, gpu_event=_is_gpu_event(event)),
        "device": _device_id(event),
    }


def _build_queue_timeline(records: Iterable[HostLaunch]) -> _QueueTimeline:
    ordered = sorted(records, key=lambda record: record.host.ts_us)
    times: list[float] = []
    prefix_max: list[float] = []
    max_end = -math.inf
    for record in ordered:
        times.append(record.host.ts_us)
        max_end = max(max_end, record.gpu.end_us)
        prefix_max.append(max_end)
    return _QueueTimeline(tuple(times), tuple(prefix_max))


def _build_busy_timeline(events: Iterable[TraceEvent]) -> _BusyTimeline:
    ordered = sorted(events, key=lambda event: (event.ts_us, event.end_us))
    starts: list[float] = []
    ends: list[float] = []
    for event in ordered:
        if not starts or event.ts_us > ends[-1]:
            starts.append(event.ts_us)
            ends.append(event.end_us)
        else:
            ends[-1] = max(ends[-1], event.end_us)
    return _BusyTimeline(tuple(starts), tuple(ends))


def _choose_correlated_gpu(
    host: TraceEvent,
    gpu_by_correlation: dict[int | str, list[TraceEvent]],
) -> TraceEvent | None:
    correlation = _correlation_id(host)
    candidates = gpu_by_correlation.get(correlation, [])
    if not candidates:
        return None
    return min(candidates, key=lambda event: (event.ts_us, event.dur_us))


def _collect_contexts(
    path: Path,
    syncs: list[TraceEvent],
    *,
    max_python_depth: int,
) -> list[_Context]:
    contexts = [
        _Context(semantic_ranges=[], cpu_ops=[], python_frames=[]) for _ in syncs
    ]
    syncs_by_thread: dict[tuple[int | str, int | str], list[tuple[float, int]]] = (
        defaultdict(list)
    )
    for index, sync in enumerate(syncs):
        syncs_by_thread[sync.thread_key].append((sync.ts_us, index))
    for values in syncs_by_thread.values():
        values.sort()
    sync_starts_by_thread = {
        key: [value[0] for value in values] for key, values in syncs_by_thread.items()
    }

    for raw in iter_trace_events(path):
        event = _as_event(raw)
        if event is None:
            continue
        is_range = event.name.startswith(_SEMANTIC_RANGE_PREFIX)
        is_cpu_op = _is_target_cpu_op(event)
        is_python = event.category == "python_function"
        if not (is_range or is_cpu_op or is_python):
            continue
        thread_syncs = syncs_by_thread.get(event.thread_key)
        if not thread_syncs:
            continue
        starts = sync_starts_by_thread[event.thread_key]
        left = bisect.bisect_left(starts, event.ts_us)
        right = bisect.bisect_right(starts, event.end_us)
        for _, sync_index in thread_syncs[left:right]:
            sync = syncs[sync_index]
            if sync.end_us > event.end_us:
                continue
            context = contexts[sync_index]
            if is_range:
                context.semantic_ranges.append(event)
            if is_cpu_op:
                context.cpu_ops.append(event)
            if is_python and max_python_depth:
                frame_limit = max_python_depth * 8
                if len(context.python_frames) < frame_limit:
                    context.python_frames.append(event)
                else:
                    outermost = max(
                        range(len(context.python_frames)),
                        key=lambda index: context.python_frames[index].dur_us,
                    )
                    if event.dur_us < context.python_frames[outermost].dur_us:
                        context.python_frames[outermost] = event
    return contexts


def _nearest_enclosing(events: list[TraceEvent]) -> TraceEvent | None:
    if not events:
        return None
    return min(events, key=lambda event: (event.dur_us, -event.ts_us))


def _python_stack(events: list[TraceEvent], max_depth: int) -> list[str]:
    ordered = sorted(events, key=lambda event: (event.dur_us, -event.ts_us))
    names: list[str] = []
    for event in ordered:
        if event.name in names:
            continue
        names.append(event.name)
        if len(names) >= max_depth:
            break
    return names


def _fallback_transfer(
    sync: TraceEvent,
    parent: TraceEvent | None,
    semantic_range: TraceEvent | None,
    gpu_copies: list[TraceEvent],
    *,
    max_gap_us: float,
) -> tuple[TraceEvent | None, str | None]:
    if parent is None and semantic_range is None:
        return None, None
    parent_is_transfer = parent is not None and (
        parent.name in {"aten::to", "aten::_to_copy", "aten::copy_"}
        or "memcpy" in parent.name.lower()
    )
    range_is_transfer = semantic_range is not None and any(
        token in semantic_range.name
        for token in (".h2d", ".dtoh", ".host_commit", ".cache")
    )
    if not (parent_is_transfer or range_is_transfer):
        return None, None
    candidate = min(
        gpu_copies,
        key=lambda event: abs(event.end_us - sync.end_us),
        default=None,
    )
    if candidate is None or abs(candidate.end_us - sync.end_us) > max_gap_us:
        return None, None
    return candidate, "nearest_time_heuristic"


def analyze_trace(
    path: str | Path,
    *,
    max_python_depth: int = 16,
    max_transfer_gap_us: float = 10_000.0,
    max_blocking_copy_gap_us: float = 50.0,
) -> list[dict[str, Any]]:
    """Return one attribution record per host-blocking CUDA sync API event.

    A pageable blocking copy is represented by two CUDA runtime events in a
    Kineto trace: ``cudaMemcpyAsync`` followed by
    ``cudaStreamSynchronize``.  The compound interval starts at the memcpy API
    entry, not at the synchronization API entry.
    """
    trace_path = Path(path).expanduser().resolve()
    syncs: list[TraceEvent] = []
    gpu_events: list[TraceEvent] = []
    gpu_copies: list[TraceEvent] = []
    runtime_launches: list[TraceEvent] = []
    runtime_calls: list[TraceEvent] = []
    gpu_by_correlation: dict[int | str, list[TraceEvent]] = defaultdict(list)
    copies_by_correlation: dict[int | str, list[TraceEvent]] = defaultdict(list)

    for raw in iter_trace_events(trace_path):
        event = _as_event(raw)
        if event is None:
            continue
        if _is_sync_api(event):
            syncs.append(event)
        if _is_runtime_api(event):
            runtime_calls.append(event)
        if _is_runtime_launch(event):
            runtime_launches.append(event)
        if _is_gpu_event(event):
            gpu_events.append(event)
            correlation = _correlation_id(event)
            if correlation is not None:
                gpu_by_correlation[correlation].append(event)
            if _is_gpu_copy(event):
                gpu_copies.append(event)
                if correlation is not None:
                    copies_by_correlation[correlation].append(event)

    contexts = _collect_contexts(
        trace_path,
        syncs,
        max_python_depth=max_python_depth,
    )

    host_launches_by_thread: dict[tuple[int | str, int | str], list[HostLaunch]] = (
        defaultdict(list)
    )
    for launch in runtime_launches:
        gpu = _choose_correlated_gpu(launch, gpu_by_correlation)
        if gpu is not None:
            host_launches_by_thread[launch.thread_key].append(HostLaunch(launch, gpu))
    for records in host_launches_by_thread.values():
        records.sort(key=lambda record: record.host.ts_us)

    runtime_calls_by_thread: dict[tuple[int | str, int | str], list[TraceEvent]] = (
        defaultdict(list)
    )
    for event in runtime_calls:
        runtime_calls_by_thread[event.thread_key].append(event)
    runtime_starts_by_thread: dict[tuple[int | str, int | str], list[float]] = {}
    for thread_key, events in runtime_calls_by_thread.items():
        events.sort(key=lambda event: event.ts_us)
        runtime_starts_by_thread[thread_key] = [event.ts_us for event in events]

    queue_records: dict[
        tuple[tuple[int | str, int | str], str | None], list[HostLaunch]
    ] = defaultdict(list)
    for thread_key, records in host_launches_by_thread.items():
        for record in records:
            stream = _stream_id(record.gpu, gpu_event=True)
            queue_records[(thread_key, None)].append(record)
            if stream is not None:
                queue_records[(thread_key, stream)].append(record)
    queue_timelines = {
        key: _build_queue_timeline(records) for key, records in queue_records.items()
    }

    gpu_by_stream: dict[str, list[TraceEvent]] = defaultdict(list)
    gpu_by_device: dict[str | None, list[TraceEvent]] = defaultdict(list)
    for event in gpu_events:
        gpu_by_device[_device_id(event)].append(event)
        stream = _stream_id(event, gpu_event=True)
        if stream is not None:
            gpu_by_stream[stream].append(event)
    gpu_end_times_by_stream: dict[str, list[float]] = {}
    for stream, events in gpu_by_stream.items():
        events.sort(key=lambda event: event.end_us)
        gpu_end_times_by_stream[stream] = [event.end_us for event in events]
    gpu_end_times_by_device: dict[str | None, list[float]] = {}
    for device, events in gpu_by_device.items():
        events.sort(key=lambda event: event.end_us)
        gpu_end_times_by_device[device] = [event.end_us for event in events]
    known_devices = {device for device in gpu_by_device if device is not None}
    busy_timelines_by_device = {
        device: _build_busy_timeline(events) for device, events in gpu_by_device.items()
    }

    occurrences: list[dict[str, Any]] = []
    for index, (sync, context) in enumerate(zip(syncs, contexts, strict=True)):
        parent = _nearest_enclosing(context.cpu_ops)
        semantic_range = _nearest_enclosing(context.semantic_ranges)

        preceding_memcpy = None
        preceding_memcpy_gpu = None
        runtime_events = runtime_calls_by_thread.get(sync.thread_key, [])
        runtime_starts = runtime_starts_by_thread.get(sync.thread_key, [])
        preceding_index = bisect.bisect_left(runtime_starts, sync.ts_us) - 1
        if preceding_index >= 0 and "streamsynchronize" in sync.name.lower():
            candidate = runtime_events[preceding_index]
            api_gap_us = sync.ts_us - candidate.end_us
            if (
                _is_async_runtime_memcpy(candidate)
                and -_API_TIMESTAMP_EPSILON_US <= api_gap_us <= max_blocking_copy_gap_us
            ):
                preceding_memcpy = candidate
                preceding_memcpy_gpu = _choose_correlated_gpu(
                    candidate,
                    copies_by_correlation,
                )

        correlation = _correlation_id(sync)
        correlated_copies = copies_by_correlation.get(correlation, [])
        transfer = min(
            correlated_copies,
            key=lambda event: abs(event.end_us - sync.end_us),
            default=None,
        )
        transfer_method = "correlation" if transfer is not None else None
        if transfer is None and preceding_memcpy_gpu is not None:
            transfer = preceding_memcpy_gpu
            transfer_method = "preceding_runtime_memcpy_correlation"
        if transfer is None:
            transfer, transfer_method = _fallback_transfer(
                sync,
                parent,
                semantic_range,
                gpu_copies,
                max_gap_us=max_transfer_gap_us,
            )

        stream = _stream_id(sync)
        if stream is None and transfer is not None:
            stream = _stream_id(transfer, gpu_event=True)

        prior_gpu = transfer
        prior_method = transfer_method
        if prior_gpu is None and stream is not None:
            stream_events = gpu_by_stream.get(stream, [])
            end_times = gpu_end_times_by_stream.get(stream, [])
            prior_index = bisect.bisect_right(end_times, sync.end_us) - 1
            if prior_index >= 0:
                prior_gpu = stream_events[prior_index]
                prior_method = "latest_completion_on_explicit_stream"
        if prior_gpu is None and "devicesynchronize" in sync.name.lower():
            sync_device = _device_id(sync)
            device_key: str | None = None
            device_is_unambiguous = True
            if sync_device is not None:
                device_key = sync_device
            elif len(known_devices) <= 1:
                device_key = next(iter(known_devices), None)
            else:
                device_is_unambiguous = False
            if device_is_unambiguous:
                device_events = gpu_by_device.get(device_key, [])
                end_times = gpu_end_times_by_device.get(device_key, [])
                prior_index = bisect.bisect_right(end_times, sync.end_us) - 1
                if prior_index >= 0:
                    prior_gpu = device_events[prior_index]
                    prior_method = "latest_completion_before_device_sync_return"

        thread_launches = host_launches_by_thread.get(sync.thread_key, [])
        launch_times = [record.host.ts_us for record in thread_launches]
        next_index = bisect.bisect_left(launch_times, sync.end_us)
        next_launch = (
            thread_launches[next_index] if next_index < len(thread_launches) else None
        )

        bubble_us = None
        if prior_gpu is not None and next_launch is not None:
            bubble_us = max(0.0, next_launch.gpu.ts_us - prior_gpu.end_us)
        host_launch_gap_us = (
            None
            if next_launch is None
            else max(0.0, next_launch.host.ts_us - sync.end_us)
        )

        timeline = queue_timelines.get((sync.thread_key, stream))
        if timeline is None:
            timeline = queue_timelines.get((sync.thread_key, None))
        queued_end_us = None if timeline is None else timeline.queued_end_at(sync.ts_us)
        queue_horizon_us = (
            None if queued_end_us is None else max(0.0, queued_end_us - sync.ts_us)
        )

        blocking_copy = None
        if preceding_memcpy is not None:
            blocking_copy = {
                "mechanism": "cudaMemcpyAsync_then_cudaStreamSynchronize",
                "runtime_memcpy": _event_dict(preceding_memcpy),
                "runtime_memcpy_gpu": _event_dict(preceding_memcpy_gpu),
                "api_gap_us": max(0.0, sync.ts_us - preceding_memcpy.end_us),
                "compound_host_block_us": max(
                    0.0,
                    sync.end_us - preceding_memcpy.ts_us,
                ),
                "runtime_memcpy_call_us": preceding_memcpy.dur_us,
                "sync_wait_us": sync.dur_us,
            }

        gpu_idle = None
        if prior_gpu is not None and next_launch is not None:
            device = _device_id(prior_gpu) or _device_id(next_launch.gpu)
            if device is None and len(known_devices) <= 1:
                device = next(iter(known_devices), None)
            busy_timeline = busy_timelines_by_device.get(device)
            if busy_timeline is not None:
                interval_start = prior_gpu.end_us
                interval_end = next_launch.gpu.ts_us
                gpu_idle = {
                    "device": device,
                    "waited_gpu_end_to_next_causal_gpu_us": max(
                        0.0,
                        interval_end - interval_start,
                    ),
                    "global_idle_in_interval_us": busy_timeline.idle_between(
                        interval_start,
                        interval_end,
                    ),
                    "global_idle_after_sync_return_us": busy_timeline.idle_between(
                        max(interval_start, sync.end_us),
                        interval_end,
                    ),
                    "global_idle_immediately_before_next_causal_gpu_us": (
                        busy_timeline.idle_immediately_before(
                            interval_end,
                            floor_us=interval_start,
                        )
                    ),
                }

        occurrences.append(
            {
                "trace_file": str(trace_path),
                "occurrence_index": index,
                "pid": sync.pid,
                "tid": sync.tid,
                "semantic_range": (
                    None if semantic_range is None else semantic_range.name
                ),
                "parent_cpu_op": None if parent is None else parent.name,
                "python_stack_innermost_first": _python_stack(
                    context.python_frames,
                    max_python_depth,
                ),
                "sync": _event_dict(sync),
                "transfer": (
                    None
                    if transfer is None
                    else {
                        **(_event_dict(transfer) or {}),
                        "direction": _transfer_direction(transfer),
                        "bytes": _byte_count(transfer),
                    }
                ),
                "blocking_copy": blocking_copy,
                "prior_waited_gpu": _event_dict(prior_gpu),
                "next_host_launch": (
                    None if next_launch is None else _event_dict(next_launch.host)
                ),
                "next_causal_gpu": (
                    None if next_launch is None else _event_dict(next_launch.gpu)
                ),
                "metrics": {
                    "sync_wait_us": sync.dur_us,
                    "post_sync_gpu_bubble_us": bubble_us,
                    "host_launch_gap_after_sync_us": host_launch_gap_us,
                    "queue_horizon_at_sync_start_us": queue_horizon_us,
                },
                "gpu_idle": gpu_idle,
                "attribution": {
                    "transfer": transfer_method,
                    "prior_waited_gpu": prior_method,
                    "next_causal_gpu": (
                        None
                        if next_launch is None
                        else "next_correlated_runtime_launch_same_host_thread"
                    ),
                },
            }
        )
    return occurrences


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_occurrences(
    occurrences: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for occurrence in occurrences:
        sync = occurrence["sync"] or {}
        transfer = occurrence["transfer"] or {}
        blocking_copy = occurrence.get("blocking_copy")
        key = (
            str(sync.get("name") or "unknown"),
            str(occurrence.get("semantic_range") or "unscoped"),
            str(occurrence.get("parent_cpu_op") or "direct_wait"),
            str(transfer.get("direction") or "none"),
            (
                "direct_sync"
                if blocking_copy is None
                else str(blocking_copy["mechanism"])
            ),
        )
        groups[key].append(occurrence)

    aggregates: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        waits = [float(row["metrics"]["sync_wait_us"]) for row in rows]
        bubbles = [
            float(value)
            for row in rows
            if (value := row["metrics"]["post_sync_gpu_bubble_us"]) is not None
        ]
        byte_counts = [
            int(value)
            for row in rows
            if row["transfer"] is not None
            and (value := row["transfer"].get("bytes")) is not None
        ]
        transfer_durations = [
            float(row["transfer"]["duration_us"])
            for row in rows
            if row["transfer"] is not None
        ]
        compound_blocks = [
            float(blocking["compound_host_block_us"])
            for row in rows
            if (blocking := row.get("blocking_copy")) is not None
        ]
        runtime_memcpy_calls = [
            float(blocking["runtime_memcpy_call_us"])
            for row in rows
            if (blocking := row.get("blocking_copy")) is not None
        ]
        global_idle = [
            float(idle["global_idle_in_interval_us"])
            for row in rows
            if (idle := row.get("gpu_idle")) is not None
        ]
        post_sync_global_idle = [
            float(idle["global_idle_after_sync_return_us"])
            for row in rows
            if (idle := row.get("gpu_idle")) is not None
        ]
        aggregates.append(
            {
                "sync_name": key[0],
                "semantic_range": key[1],
                "parent_cpu_op": key[2],
                "transfer_direction": key[3],
                "mechanism": key[4],
                "count": len(rows),
                "blocking_copy_count": len(compound_blocks),
                "compound_host_block_total_us": sum(compound_blocks),
                "compound_host_block_mean_us": (
                    None
                    if not compound_blocks
                    else sum(compound_blocks) / len(compound_blocks)
                ),
                "compound_host_block_p50_us": _percentile(compound_blocks, 0.50),
                "compound_host_block_p95_us": _percentile(compound_blocks, 0.95),
                "runtime_memcpy_call_total_us": sum(runtime_memcpy_calls),
                "sync_wait_total_us": sum(waits),
                "sync_wait_mean_us": sum(waits) / len(waits),
                "sync_wait_p50_us": _percentile(waits, 0.50),
                "sync_wait_p95_us": _percentile(waits, 0.95),
                "sync_wait_max_us": max(waits),
                "bubble_measured_count": len(bubbles),
                "post_sync_bubble_total_us": sum(bubbles),
                "post_sync_bubble_p50_us": _percentile(bubbles, 0.50),
                "post_sync_bubble_p95_us": _percentile(bubbles, 0.95),
                "transfer_bytes_total": sum(byte_counts),
                "transfer_duration_total_us": sum(transfer_durations),
                "global_idle_measured_count": len(global_idle),
                "global_idle_total_us": sum(global_idle),
                "global_idle_p50_us": _percentile(global_idle, 0.50),
                "global_idle_p95_us": _percentile(global_idle, 0.95),
                "post_sync_global_idle_total_us": sum(post_sync_global_idle),
            }
        )
    return aggregates


def write_analysis(
    occurrences: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_occurrences(occurrences)
    notes = [
        "All timestamps and causal comparisons are per trace file/process.",
        (
            "The blocking-copy mechanism is the adjacent cudaMemcpyAsync entry "
            "through cudaStreamSynchronize return; compound host time includes both APIs."
        ),
        (
            "A post-sync bubble is emitted only when both g0 and a later "
            "correlation-backed g1 are present."
        ),
        "Global-idle metrics subtract the union of all GPU events on the same device.",
        (
            "Per-occurrence causal intervals may overlap; aggregate bubble/idle sums "
            "are not unique trace wall time."
        ),
        "nearest_time_heuristic transfer attribution must be verified manually.",
        "Sync duration is host wait time, not automatically wasted GPU time.",
    ]
    occurrence_path = directory / "cuda_sync_occurrences.json"
    aggregate_path = directory / "cuda_sync_aggregates.json"
    csv_path = directory / "cuda_sync_aggregates.csv"
    occurrence_path.write_text(
        json.dumps(
            {"schema_version": 1, "notes": notes, "occurrences": occurrences},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(
            {"schema_version": 1, "notes": notes, "aggregates": aggregates},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        list(aggregates[0])
        if aggregates
        else [
            "sync_name",
            "semantic_range",
            "parent_cpu_op",
            "transfer_direction",
            "mechanism",
            "count",
        ]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregates)
    return {
        "occurrences": str(occurrence_path),
        "aggregates_json": str(aggregate_path),
        "aggregates_csv": str(csv_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="Torch Profiler .trace.json or .trace.json.gz files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for occurrence JSON and aggregate JSON/CSV",
    )
    parser.add_argument("--max-python-depth", type=int, default=16)
    parser.add_argument("--max-transfer-gap-us", type=float, default=10_000.0)
    parser.add_argument(
        "--max-blocking-copy-gap-us",
        type=float,
        default=50.0,
        help=(
            "Maximum gap between an immediately preceding cudaMemcpyAsync "
            "return and cudaStreamSynchronize entry for compound-copy attribution"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_python_depth < 0:
        raise SystemExit("--max-python-depth must be >= 0")
    if args.max_transfer_gap_us < 0:
        raise SystemExit("--max-transfer-gap-us must be >= 0")
    if args.max_blocking_copy_gap_us < 0:
        raise SystemExit("--max-blocking-copy-gap-us must be >= 0")
    occurrences: list[dict[str, Any]] = []
    for trace in args.traces:
        occurrences.extend(
            analyze_trace(
                trace,
                max_python_depth=args.max_python_depth,
                max_transfer_gap_us=args.max_transfer_gap_us,
                max_blocking_copy_gap_us=args.max_blocking_copy_gap_us,
            )
        )
    paths = write_analysis(occurrences, args.output_dir)
    print(
        json.dumps(
            {
                "trace_count": len(args.traces),
                "sync_occurrence_count": len(occurrences),
                "outputs": paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
