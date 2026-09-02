# SPDX-License-Identifier: Apache-2.0
"""Per step ledger for OmniScheduler stages.

While a request profile run is active, the scheduler records one row per
batch it runs: the launch to launch cycle, the host wall of the launch call
and of the lookahead resolve, the time the host blocked on a device event,
the device span from the start of the forward to the published tokens, the
batch shape, whether a CUDA graph replayed, the idle sleeps taken before the
step, and the caching allocator's allocation count over the cycle. Together
these say whether a stage is host bound or device bound, and by how much,
without a kernel trace.

Recording follows the request event recorder: a stage joins the run at its
next batch after the recorder becomes active and writes the run when it
handles the profiler stop, so the existing start_profile and
start_request_profile calls are the only switch. The summary is written next
to the events file as step_ledger_<stage>_<pid>.json and logged as one line
per batch shape. A stage that ran no batch during the run writes nothing.

Two guarantees. The ledger adds no device wait of its own: device spans come
from timing event pairs that are read only after the end event reports
complete, pairs still in flight when a run ends are counted, not waited for,
and the one synchronize it performs, wait_for_gpu_end, stands in for a
blocking copy the runner is about to do on the same work. And the ledger
never raises into the scheduler loop or the runner: the first failure inside
it is logged and disables it for the rest of the process, so a request can
never fail because of its accounting.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from sglang_omni.profiler.event_recorder import get_active_stage, get_recorder

logger = logging.getLogger(__name__)


@dataclass
class _Step:
    step_id: int
    mode: str
    rows: int
    extend_tokens: int
    t_launch: float
    cycle: float | None
    idle_sleeps: int
    host: float | None = None
    resolve: float | None = None
    wait: float = 0.0
    graph: bool = False
    gpu_span_ms: float | None = None
    allocations: int | None = None
    gpu_start: Any = None
    gpu_end: Any = None
    closed: bool = False


@dataclass
class _Shape:
    steps: list[_Step] = field(default_factory=list)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        index = min(count - 1, max(0, round(fraction * (count - 1))))
        return ordered[index]

    return {
        "p50": round(at(0.5), 4),
        "p90": round(at(0.9), 4),
        "max": round(ordered[-1], 4),
    }


def _guarded(method: Callable) -> Callable:
    """Turn the first failure inside the ledger into a logged disable."""

    @functools.wraps(method)
    def wrapper(self: "StepLedger", *args: Any, **kwargs: Any) -> Any:
        if self._disabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._disabled = True
            logger.warning(
                "step ledger disabled after a failure in %s",
                method.__name__,
                exc_info=True,
            )
            return None

    return wrapper


class StepLedger:
    """Per step host and device accounting for one scheduler."""

    def __init__(
        self,
        device: Any,
        *,
        clock: Callable[[], float] = time.perf_counter,
        gpu_timing: bool | None = None,
    ) -> None:
        self._disabled = False
        self._clock = clock
        self._lock = threading.Lock()
        self._recorder = get_recorder()
        self._run_id: str | None = None
        self._event_dir: str | None = None
        self._stage: str | None = None
        self._shapes: dict[tuple[str, int], _Shape] = {}
        self._open: dict[int, _Step] = {}
        self._pending_spans: deque[_Step] = deque()
        self._current: _Step | None = None
        self._resolving: _Step | None = None
        self._last_step: _Step | None = None
        # Never reset between runs: a SchedulerOutput stamped in an earlier
        # run must not resolve against a step of a later one.
        self._next_id = 1
        self._last_launch_t: float | None = None
        self._idle_sleeps = 0
        self._dropped_steps = 0
        self._last_allocations: int | None = None
        self._device: Any = None
        self._device_module: Any = None
        self._gpu_timing = False
        self._allocation_stats = False
        try:
            self._device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
            self._device_module = torch.get_device_module(self._device)
            self._gpu_timing = (
                self._probe_gpu_timing() if gpu_timing is None else bool(gpu_timing)
            )
            self._allocation_stats = self._device.type == "cuda"
        except Exception:
            self._disabled = True
            logger.warning(
                "step ledger disabled: no accounting for device %r",
                device,
                exc_info=True,
            )

    # ---- capability -----------------------------------------------------

    def _probe_gpu_timing(self) -> bool:
        if self._device.type == "cpu":
            return False
        try:
            event = self._device_module.Event(enable_timing=True)
        except Exception:
            return False
        return hasattr(event, "elapsed_time") and hasattr(event, "query")

    @property
    def current_id(self) -> int:
        """Id of the step whose launch is open, 0 when not recording."""
        step = self._current
        return 0 if step is None else step.step_id

    # ---- scheduler side -------------------------------------------------

    @_guarded
    def begin(self, batch: Any) -> None:
        """Open a step for the batch about to be launched.

        Called from the scheduler thread once per batch, before the runner
        is entered. Joins or leaves a run as the request recorder toggles.
        """
        # One read: the recorder can be stopped by another thread between
        # is_active and active_run_id, and a run id of None means inactive.
        run_id = self._recorder.active_run_id()
        if run_id is None:
            if self._run_id is not None:
                self.finish()
            return
        if self._run_id is not None and run_id != self._run_id:
            # A new run started before this stage saw the previous stop:
            # write the previous run instead of dropping it.
            self.finish()
        with self._lock:
            if self._run_id is None:
                self._run_id = run_id
                path = self._recorder.active_path()
                self._event_dir = None if path is None else str(Path(path).parent)
                self._stage = get_active_stage()
                self._last_launch_t = None
                self._last_step = None
                self._idle_sleeps = 0
                self._last_allocations = None
            self._drain_spans_unlocked()
            if self._current is not None and not self._current.closed:
                self._discard_unlocked(self._current)
            now = self._clock()
            cycle = None if self._last_launch_t is None else now - self._last_launch_t
            self._last_launch_t = now
            self._note_allocations_unlocked()
            mode = _batch_mode(batch)
            tokens = int(batch.extend_num_tokens or 0) if mode == "extend" else 0
            step = _Step(
                step_id=self._next_id,
                mode=mode,
                rows=len(batch.reqs),
                extend_tokens=tokens,
                t_launch=now,
                cycle=cycle,
                idle_sleeps=self._idle_sleeps,
            )
            self._next_id += 1
            self._idle_sleeps = 0
            self._current = step
            self._last_step = step
            self._open[step.step_id] = step

    @_guarded
    def end_launch(self, *, can_run_cuda_graph: bool, lookahead: bool) -> None:
        """Close the launch call of the current step.

        A synchronous step closes here. A lookahead step stays open until
        end_resolve records its resolve wall.
        """
        step = self._current
        if step is None:
            return
        with self._lock:
            step.host = self._clock() - step.t_launch
            step.graph = bool(can_run_cuda_graph)
            self._current = None
            if not lookahead:
                self._close_unlocked(step)

    @_guarded
    def begin_resolve(self, step_id: int) -> None:
        step = self._open.get(step_id)
        if step is None:
            return
        with self._lock:
            self._resolving = step
            step.resolve = self._clock()

    @_guarded
    def end_resolve(self) -> None:
        step = self._resolving
        if step is None:
            return
        with self._lock:
            self._resolving = None
            if step.resolve is not None:
                step.resolve = self._clock() - step.resolve
            self._close_unlocked(step)

    def note_idle_sleep(self) -> None:
        if self._run_id is not None:
            self._idle_sleeps += 1

    # ---- runner side ----------------------------------------------------

    @_guarded
    def mark_gpu_start(self) -> None:
        """Record the timing event before the forward of the open step."""
        step = self._current
        if step is None or not self._gpu_timing:
            return
        event = self._device_module.Event(enable_timing=True)
        event.record()
        step.gpu_start = event

    @_guarded
    def mark_gpu_end(self) -> None:
        """Record the timing event after the open step published its tokens."""
        step = self._current
        if step is None or step.gpu_start is None:
            return
        event = self._device_module.Event(enable_timing=True)
        event.record()
        step.gpu_end = event
        with self._lock:
            self._pending_spans.append(step)

    @_guarded
    def add_wait(self, seconds: float) -> None:
        """Account host time blocked on a device event.

        Attributed to the step being resolved when a resolve is open,
        otherwise to the step being launched, which is the synchronous
        path's finalize.
        """
        step = self._resolving if self._resolving is not None else self._current
        if step is None:
            return
        step.wait += seconds

    @_guarded
    def wait_for_gpu_end(self) -> None:
        """Block on the open step's end event and book the time as wait.

        The one place the ledger waits on the device: the runner calls it
        right before a blocking copy of the step's tokens, which would wait
        for the same work an instant later. The wall is unchanged, the wait
        becomes visible.
        """
        step = self._resolving if self._resolving is not None else self._current
        if step is None or step.gpu_end is None or step.gpu_end.query():
            return
        waited_from = self._clock()
        step.gpu_end.synchronize()
        step.wait += self._clock() - waited_from

    # ---- readout --------------------------------------------------------

    @_guarded
    def summary(self) -> dict[str, Any]:
        with self._lock:
            self._drain_spans_unlocked()
            return self._summary_unlocked()

    @_guarded
    def finish(self, run_id: str | None = None) -> str | None:
        """Write the run's summary next to its events and reset.

        Safe to call from any thread and when nothing was recorded. With a
        run id, only that run is finished, so a stop for another run is
        ignored the way the recorder ignores it.
        """
        with self._lock:
            if self._run_id is None:
                return None
            if run_id is not None and run_id != self._run_id:
                return None
            self._drain_spans_unlocked()
            for step in list(self._open.values()):
                self._discard_unlocked(step)
            summary = self._summary_unlocked()
            event_dir = self._event_dir
            stage = self._stage or "unknown"
            finished_run = self._run_id
            self._reset_unlocked()
        for line in _format_lines(stage, summary):
            logger.info(line)
        if event_dir is None or summary["steps"] == 0:
            return None
        path = Path(event_dir) / f"step_ledger_{stage}_{os.getpid()}.json"
        try:
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            logger.warning(
                "step ledger for run %s could not be written",
                finished_run,
                exc_info=True,
            )
            return None
        return str(path)

    # ---- internals ------------------------------------------------------

    def _close_unlocked(self, step: _Step) -> None:
        if step.closed:
            return
        step.closed = True
        self._open.pop(step.step_id, None)
        self._shapes.setdefault((step.mode, step.rows), _Shape()).steps.append(step)

    def _discard_unlocked(self, step: _Step) -> None:
        self._open.pop(step.step_id, None)
        if not step.closed:
            step.closed = True
            self._dropped_steps += 1
        if self._current is step:
            self._current = None
        if self._resolving is step:
            self._resolving = None

    def _drain_spans_unlocked(self) -> None:
        pending = self._pending_spans
        while pending:
            step = pending[0]
            end = step.gpu_end
            if end is None:
                pending.popleft()
                continue
            if not end.query():
                return
            pending.popleft()
            try:
                step.gpu_span_ms = float(step.gpu_start.elapsed_time(end))
            except Exception:
                step.gpu_span_ms = None
            step.gpu_start = None
            step.gpu_end = None

    def _note_allocations_unlocked(self) -> None:
        """Sample the allocation counter at launch and charge the delta since
        the previous launch to the previous step: allocations per cycle."""
        if not self._allocation_stats:
            return
        try:
            stats = torch.cuda.memory_stats_as_nested_dict(self._device)
            count = int(stats["allocation"]["all"]["allocated"])
        except Exception:
            self._allocation_stats = False
            return
        previous = self._last_allocations
        self._last_allocations = count
        if previous is not None and self._last_step is not None:
            self._last_step.allocations = count - previous

    def _reset_unlocked(self) -> None:
        self._run_id = None
        self._event_dir = None
        self._stage = None
        self._shapes = {}
        self._open = {}
        self._pending_spans = deque()
        self._current = None
        self._resolving = None
        self._last_step = None
        self._last_launch_t = None
        self._idle_sleeps = 0
        self._dropped_steps = 0
        self._last_allocations = None

    def _summary_unlocked(self) -> dict[str, Any]:
        shapes = []
        unread = 0
        total = 0
        for (mode, rows), shape in sorted(self._shapes.items()):
            steps = shape.steps
            total += len(steps)
            spans = [s.gpu_span_ms for s in steps if s.gpu_span_ms is not None]
            unread += sum(
                1 for s in steps if s.gpu_span_ms is None and s.gpu_end is not None
            )
            cycles = [s.cycle * 1e3 for s in steps if s.cycle is not None]
            # Zero when the device is behind the host: the next launch came
            # before this step's span ended, so the device had no idle.
            idle_floor = [
                max(0.0, s.cycle * 1e3 - s.gpu_span_ms)
                for s in steps
                if s.cycle is not None and s.gpu_span_ms is not None
            ]
            allocations = [s.allocations for s in steps if s.allocations is not None]
            row: dict[str, Any] = {
                "mode": mode,
                "rows": rows,
                "steps": len(steps),
                "cycle_ms": _percentiles(cycles),
                "host_ms": _percentiles(
                    [s.host * 1e3 for s in steps if s.host is not None]
                ),
                "resolve_ms": _percentiles(
                    [s.resolve * 1e3 for s in steps if s.resolve is not None]
                ),
                "wait_ms": _percentiles([s.wait * 1e3 for s in steps]),
                "gpu_span_ms": _percentiles(spans),
                "gpu_idle_floor_ms": _percentiles(idle_floor),
                "graph_share": round(sum(1 for s in steps if s.graph) / len(steps), 4),
                "idle_sleeps_per_step": round(
                    sum(s.idle_sleeps for s in steps) / len(steps), 4
                ),
                "allocations_per_step": (
                    round(sum(allocations) / len(allocations), 2)
                    if allocations
                    else None
                ),
            }
            if mode == "extend":
                row["extend_tokens"] = _percentiles(
                    [float(s.extend_tokens) for s in steps]
                )
            shapes.append(row)
        return {
            "stage": self._stage,
            "pid": os.getpid(),
            "run_id": self._run_id,
            "device": str(self._device),
            "gpu_timing": self._gpu_timing,
            "steps": total,
            "dropped_steps": self._dropped_steps,
            "unread_gpu_spans": unread,
            "shapes": shapes,
        }


def _batch_mode(batch: Any) -> str:
    mode = batch.forward_mode
    if mode is None:
        return "other"
    if mode.is_extend():
        return "extend"
    if mode.is_decode():
        return "decode"
    return "other"


def _fmt(stats: dict[str, float] | None, key: str = "p50") -> str:
    return "n/a" if stats is None else f"{stats[key]:.2f}"


def _format_lines(stage: str, summary: dict[str, Any]) -> list[str]:
    lines = [
        f"step_ledger stage={stage} run={summary['run_id']} steps={summary['steps']} "
        f"dropped={summary['dropped_steps']} unread_gpu_spans={summary['unread_gpu_spans']} "
        f"gpu_timing={summary['gpu_timing']}"
    ]
    for row in summary["shapes"]:
        extra = ""
        if row["mode"] == "extend":
            extra = f" tokens p50 {_fmt(row['extend_tokens'])}"
        allocations = row["allocations_per_step"]
        lines.append(
            f"step_ledger stage={stage} {row['mode']} rows={row['rows']} steps={row['steps']}"
            f" cycle p50/p90 {_fmt(row['cycle_ms'])}/{_fmt(row['cycle_ms'], 'p90')} ms"
            f" host {_fmt(row['host_ms'])} resolve {_fmt(row['resolve_ms'])}"
            f" wait {_fmt(row['wait_ms'])} gpu_span {_fmt(row['gpu_span_ms'])}"
            f" idle_floor {_fmt(row['gpu_idle_floor_ms'])}"
            f" graph {row['graph_share'] * 100:.0f}%"
            f" sleeps/step {row['idle_sleeps_per_step']:.2f}"
            f" allocs/step {'n/a' if allocations is None else allocations}{extra}"
        )
    return lines
