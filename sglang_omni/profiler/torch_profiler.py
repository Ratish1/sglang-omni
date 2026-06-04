# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/torch_profiler.py

import json
import logging
import os
import subprocess
import threading
import time
from contextlib import nullcontext
from types import TracebackType

from torch.profiler import ProfilerActivity, profile
from torch.profiler import record_function as _torch_record_function

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class _NoOpRecordFunction:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


_NOOP_RECORD_FUNCTION = _NoOpRecordFunction()


class _OmniRecordFunction:
    def __init__(self, name: str):
        self.name = name
        self._torch_context = _torch_record_function(name)
        self._start_us: int | None = None

    def __enter__(self):
        self._start_us = time.perf_counter_ns() // 1000
        return self._torch_context.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        end_us = time.perf_counter_ns() // 1000
        torch_result = self._torch_context.__exit__(exc_type, exc, tb)
        start_us = self._start_us
        if start_us is not None:
            TorchProfiler.add_manual_event(
                self.name,
                start_us=start_us,
                duration_us=max(end_us - start_us, 0),
            )
        return bool(torch_result)


def record_function(name: str):
    """Annotate a region in torch profiler traces.

    Keep the helper local to the Omni profiler package so Higgs/model code does
    not import torch profiler internals directly.
    """

    if not TorchProfiler.is_active():
        return _NOOP_RECORD_FUNCTION
    return _OmniRecordFunction(name)


def _manual_trace_event(name: str, *, start_us: int, duration_us: int) -> dict:
    return {
        "ph": "X",
        "cat": "sglang_omni",
        "name": name,
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "ts": start_us,
        "dur": duration_us,
    }


def _append_events_to_chrome_trace(json_file: str, events: list[dict]) -> None:
    if not events:
        return
    with open(json_file, encoding="utf-8") as f:
        trace = json.load(f)
    trace_events = trace.setdefault("traceEvents", [])
    if not isinstance(trace_events, list):
        raise TypeError("Chrome trace 'traceEvents' must be a list")
    trace_events.extend(events)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    _active_run_id: str | None = None
    _lock = threading.Lock()
    _manual_events: list[dict] = []
    _manual_events_lock = threading.Lock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        """
        Start the profiler with the given trace path template.
        """
        with cls._lock:
            rank = cls._get_rank()

            # 1. Cleanup any existing profiler
            if cls._profiler is not None:
                if run_id is not None and cls._active_run_id == run_id:
                    return f"{cls._trace_template}_rank{rank}.trace.json.gz"

                logger.warning(
                    "[Rank %s] Torch profiler already active (run_id=%s), restarting for run_id=%s",
                    rank,
                    cls._active_run_id,
                    run_id,
                )
                try:
                    cls._profiler.stop()
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to stop existing profiler: %s", rank, e
                    )
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""

            # 2. Make path absolute
            trace_path_template = os.path.abspath(trace_path_template)
            cls._trace_template = trace_path_template
            cls._active_run_id = run_id
            cls._clear_manual_events()

            # Expected paths
            json_file = f"{trace_path_template}_rank{rank}.trace.json"

            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            # 3. Define the on_trace_ready handler
            def trace_handler(p):
                nonlocal json_file

                # A. Export JSON Trace
                try:
                    p.export_chrome_trace(json_file)
                    cls._append_manual_events(json_file)
                    logger.info(f"[Rank {rank}] Trace exported to {json_file}")

                    try:
                        subprocess.Popen(["gzip", "-f", json_file])
                        logger.info(
                            f"[Rank {rank}] Triggered background compression for {json_file}"
                        )
                        # Update variable to point to the eventual file
                        json_file = f"{json_file}.gz"
                    except Exception as compress_err:
                        logger.warning(
                            f"[Rank {rank}] Background gzip failed to start: {compress_err}"
                        )

                except Exception as e:
                    logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

            # No ``schedule``: record continuously between start/stop.
            # Expensive flags are env-var opt-in (default off keeps the
            # trace tens of MB; all on can hit multi-GB).
            cls._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=trace_handler,
                record_shapes=os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES")
                == "1",
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1",
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
            )

            # 5. Start profiling
            cls._profiler.start()

            # Return the expected final path
            return f"{trace_path_template}_rank{rank}.trace.json.gz"

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict | None:
        """
        Stop the profiler.

        If run_id is provided:
          - only stop when active_run_id matches (otherwise ignore)
        """
        with cls._lock:
            if cls._profiler is None:
                return None

            rank = cls._get_rank()
            active = cls._active_run_id

            if run_id is not None and active is not None and active != run_id:
                logger.warning(
                    "[Rank %s] Ignoring profiler stop for run_id=%s because active_run_id=%s",
                    rank,
                    run_id,
                    active,
                )
                return None

            base_path = f"{cls._trace_template}_rank{rank}"
            json_path = f"{base_path}.trace.json"
            gz_path = f"{json_path}.gz"

            profiler = cls._profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            if os.path.exists(json_path) or os.path.exists(gz_path):
                logger.info(
                    "[Rank %s] Trace already exported to %s",
                    rank,
                    gz_path if os.path.exists(gz_path) else json_path,
                )
                cls._profiler = None
                cls._active_run_id = None
                cls._trace_template = ""
                return {"trace": gz_path, "table": None}

            # No schedule → on_trace_ready isn't fired on stop, so
            # export here.
            try:
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                profiler.export_chrome_trace(json_path)
                cls._append_manual_events(json_path)
                logger.info("[Rank %s] Trace exported to %s", rank, json_path)
                try:
                    subprocess.Popen(["gzip", "-f", json_path])
                    logger.info(
                        "[Rank %s] Triggered background compression for %s",
                        rank,
                        json_path,
                    )
                except Exception as compress_err:
                    logger.warning(
                        "[Rank %s] Background gzip failed: %s",
                        rank,
                        compress_err,
                    )
            except Exception as e:
                logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""

            return {"trace": gz_path, "table": None}

    @classmethod
    def step(cls):
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def add_manual_event(cls, name: str, *, start_us: int, duration_us: int) -> None:
        if cls._profiler is None:
            return
        event = _manual_trace_event(
            name,
            start_us=start_us,
            duration_us=duration_us,
        )
        with cls._manual_events_lock:
            cls._manual_events.append(event)

    @classmethod
    def _clear_manual_events(cls) -> None:
        with cls._manual_events_lock:
            cls._manual_events = []

    @classmethod
    def _pop_manual_events(cls) -> list[dict]:
        with cls._manual_events_lock:
            events = cls._manual_events
            cls._manual_events = []
        return events

    @classmethod
    def _append_manual_events(cls, json_file: str) -> None:
        events = cls._pop_manual_events()
        if not events:
            return
        _append_events_to_chrome_trace(json_file, events)
        logger.info(
            "Appended %d sglang-omni trace ranges to %s", len(events), json_file
        )

    @classmethod
    def get_step_context(cls):
        return nullcontext()
