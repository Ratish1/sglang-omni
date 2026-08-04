# SPDX-License-Identifier: Apache-2.0
"""Bounded, owner-thread PyTorch profiling.

PyTorch user ranges are thread-sensitive.  The lifecycle in this module is
therefore deliberately strict: start, step, and stop must all run on the same
thread.  Pipeline control code is responsible for delivering commands to that
owner and waiting for its acknowledgement.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch.profiler import ProfilerActivity, profile, record_function, schedule

from .base_profiler import ProfilerBase

logger = logging.getLogger(__name__)

_MAX_PROFILE_STEPS = 10_000


@dataclass(frozen=True)
class TorchProfilerConfig:
    """Validated configuration for one bounded trace."""

    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 20
    repeat: int = 1
    include_cuda: bool = True
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    compress: bool = True

    def __post_init__(self) -> None:
        for name in ("wait_steps", "warmup_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("active_steps", "repeat"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.repeat != 1:
            raise ValueError(
                "repeat must be 1 because each run owns one finalized trace"
            )
        total = self.total_steps
        if total > _MAX_PROFILE_STEPS:
            raise ValueError(
                f"profiler schedule has {total} steps; maximum is "
                f"{_MAX_PROFILE_STEPS}"
            )

    @property
    def total_steps(self) -> int:
        return (self.wait_steps + self.warmup_steps + self.active_steps) * self.repeat

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TorchProfilerConfig":
        if not data:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown torch profiler config fields: {unknown}")
        return cls(**data)


def _compress_json(json_path: Path) -> Path:
    """Create a durable gzip artifact and remove the uncompressed source."""

    gz_path = json_path.with_suffix(json_path.suffix + ".gz")
    temporary = gz_path.with_suffix(gz_path.suffix + ".tmp")
    with json_path.open("rb") as src, gzip.open(temporary, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(temporary, gz_path)
    json_path.unlink()
    return gz_path


class TorchProfiler(ProfilerBase):
    """Process-local profiler owned by one explicitly named thread."""

    _profiler: profile | None = None
    _trace_template: str = ""
    _active_run_id: str | None = None
    _owner_thread_id: int | None = None
    _owner_native_tid: int | None = None
    _owner_thread_name: str | None = None
    _config: TorchProfilerConfig | None = None
    _step_count: int = 0
    _trace_path: str | None = None
    _pending_json_path: str | None = None
    _export_error: str | None = None
    _lock = threading.RLock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def is_owner_thread(cls) -> bool:
        return cls._owner_thread_id == threading.get_ident()

    @classmethod
    def _require_owner(cls, action: str) -> None:
        if not cls.is_owner_thread():
            raise RuntimeError(
                f"TorchProfiler.{action} must run on owner thread "
                f"{cls._owner_thread_name}/{cls._owner_thread_id}; got "
                f"{threading.current_thread().name}/{threading.get_ident()}"
            )

    @classmethod
    def start(
        cls,
        trace_path_template: str,
        run_id: str | None = None,
        *,
        config: TorchProfilerConfig | None = None,
    ) -> str:
        """Start a bounded trace on the calling thread.

        A same-run call is idempotent only from the owning thread.  Starting a
        second stage or thread while one process-global Kineto session is live
        is rejected rather than silently returning the wrong trace.
        """

        config = config or TorchProfilerConfig()
        with cls._lock:
            rank = cls._get_rank()
            template = str(Path(trace_path_template).expanduser().resolve())
            if cls._profiler is not None:
                if (
                    run_id is not None
                    and cls._active_run_id == run_id
                    and cls.is_owner_thread()
                    and cls._config == config
                    and cls._trace_template == template
                ):
                    return cls._expected_trace_path()
                raise RuntimeError(
                    "torch profiler is already active "
                    f"(run_id={cls._active_run_id}, owner="
                    f"{cls._owner_thread_name}/{cls._owner_thread_id})"
                )

            json_path = Path(f"{template}_rank{rank}.trace.json")
            json_path.parent.mkdir(parents=True, exist_ok=True)

            cls._trace_template = template
            cls._active_run_id = run_id
            cls._owner_thread_id = threading.get_ident()
            cls._owner_native_tid = threading.get_native_id()
            cls._owner_thread_name = threading.current_thread().name
            cls._config = config
            cls._step_count = 0
            cls._trace_path = None
            cls._pending_json_path = None
            cls._export_error = None

            activities = [ProfilerActivity.CPU]
            if config.include_cuda:
                activities.append(ProfilerActivity.CUDA)

            def trace_handler(profiler: profile) -> None:
                try:
                    profiler.export_chrome_trace(str(json_path))
                    # Compression is deliberately deferred to acknowledged
                    # stop.  Gzip can consume substantial scheduler-thread CPU
                    # and would otherwise contaminate requests immediately
                    # following the active Kineto window.
                    if config.compress:
                        cls._pending_json_path = str(json_path)
                    else:
                        cls._trace_path = str(json_path)
                    logger.info(
                        "[Rank %s] profiler trace exported at %s",
                        rank,
                        json_path,
                    )
                except Exception as exc:
                    cls._export_error = str(exc) or type(exc).__name__
                    logger.exception("[Rank %s] profiler trace export failed", rank)

            try:
                cls._profiler = profile(
                    activities=activities,
                    schedule=schedule(
                        wait=config.wait_steps,
                        warmup=config.warmup_steps,
                        active=config.active_steps,
                        repeat=config.repeat,
                    ),
                    on_trace_ready=trace_handler,
                    record_shapes=config.record_shapes,
                    profile_memory=config.profile_memory,
                    with_stack=config.with_stack,
                    with_flops=config.with_flops,
                )
                cls._profiler.start()
            except Exception:
                cls._clear_state()
                raise

            logger.info(
                "[Rank %s] started bounded torch profiler run_id=%s owner=%s/%s "
                "schedule=%s",
                rank,
                run_id,
                cls._owner_thread_name,
                cls._owner_thread_id,
                config.to_dict(),
            )
            return cls._expected_trace_path()

    @classmethod
    def step(cls) -> dict[str, Any] | None:
        """Advance one semantic scheduler step."""

        with cls._lock:
            if cls._profiler is None:
                return None
            cls._require_owner("step")
            # Emit inside every semantic step so at least one canary lands in
            # the scheduled active window (the immediate post-start period may
            # be WAIT/WARMUP and is intentionally discarded).
            with record_function(
                "sglang_omni.profiler.scheduler_owner."
                f"{cls._owner_thread_name or 'unknown'}"
            ):
                pass
            cls._profiler.step()
            cls._step_count += 1
            return cls.snapshot()

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict[str, Any] | None:
        """Stop on the owner thread and return a finalized artifact manifest."""

        with cls._lock:
            if cls._profiler is None:
                return None
            cls._require_owner("stop")
            if (
                run_id is not None
                and cls._active_run_id is not None
                and run_id != cls._active_run_id
            ):
                raise RuntimeError(
                    f"profiler stop run_id={run_id} does not match active "
                    f"run_id={cls._active_run_id}"
                )

            profiler = cls._profiler
            try:
                profiler.stop()
                if (
                    cls._trace_path is None
                    and cls._pending_json_path is None
                    and cls._export_error is None
                ):
                    # A stop before the scheduled active window completed may
                    # not invoke on_trace_ready.  Export the partial trace
                    # explicitly and label it through the recorded step count.
                    rank = cls._get_rank()
                    json_path = Path(f"{cls._trace_template}_rank{rank}.trace.json")
                    profiler.export_chrome_trace(str(json_path))
                    if cls._config is not None and cls._config.compress:
                        cls._pending_json_path = str(json_path)
                    else:
                        cls._trace_path = str(json_path)
                if cls._pending_json_path is not None and cls._export_error is None:
                    cls._trace_path = str(_compress_json(Path(cls._pending_json_path)))
                    cls._pending_json_path = None
            except Exception as exc:
                cls._export_error = str(exc) or type(exc).__name__
                logger.exception("torch profiler stop/export failed")

            result = cls.snapshot()
            cls._clear_state()
            return result

    @classmethod
    def snapshot(cls) -> dict[str, Any]:
        config = cls._config
        return {
            "active": cls._profiler is not None,
            "run_id": cls._active_run_id,
            "owner_tid": cls._owner_native_tid,
            "owner_ident": cls._owner_thread_id,
            "owner_thread": cls._owner_thread_name,
            "step_count": cls._step_count,
            "expected_steps": config.total_steps if config is not None else None,
            "schedule_complete": (
                config is not None and cls._step_count >= config.total_steps
            ),
            "config": config.to_dict() if config is not None else None,
            "trace": cls._trace_path or cls._expected_trace_path(),
            "trace_finalized": cls._trace_path is not None,
            "trace_exported": (
                cls._trace_path is not None or cls._pending_json_path is not None
            ),
            "export_error": cls._export_error,
        }

    @classmethod
    def _expected_trace_path(cls) -> str:
        rank = cls._get_rank()
        config = cls._config
        suffix = (
            ".trace.json.gz" if config is None or config.compress else ".trace.json"
        )
        return f"{cls._trace_template}_rank{rank}{suffix}"

    @classmethod
    def _clear_state(cls) -> None:
        cls._profiler = None
        cls._trace_template = ""
        cls._active_run_id = None
        cls._owner_thread_id = None
        cls._owner_native_tid = None
        cls._owner_thread_name = None
        cls._config = None
        cls._step_count = 0
        cls._trace_path = None
        cls._pending_json_path = None
        cls._export_error = None

    @classmethod
    def get_step_context(cls):
        # Kept for ProfilerBase compatibility.  Semantic ranges are provided by
        # profiler.trace_ranges and steps are advanced by the scheduler.
        from contextlib import nullcontext

        return nullcontext()
