# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from torch.profiler import (
    ProfilerActivity,
    _ExperimentalConfig,
    profile,
    supported_activities,
)

from sglang_omni.platforms import current_platform

if current_platform.is_npu():
    import torch_npu

from .base_profiler import ProfilerBase

# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/torch_profiler.py


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def profiler_activities() -> list[ProfilerActivity]:
    """CPU plus whichever device activity this torch build supports."""
    device = sorted(
        (a for a in supported_activities() if a != ProfilerActivity.CPU),
        key=lambda a: a.name,
    )
    return [ProfilerActivity.CPU, *device]


@dataclass(kw_only=True)
class StepWindow:
    """A profile that starts at the counter's next forward and stops after num_steps."""

    counter: object
    trace_path_template: str
    run_id: str
    num_steps: int
    with_stack: bool | None
    record_shapes: bool | None
    forwards_left: int | None = None


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    _active_run_id: str | None = None
    _step_window: StepWindow | None = None
    _lock = threading.RLock()

    @classmethod
    def get_active_run_id(cls) -> str | None:
        return cls._active_run_id

    @classmethod
    def arm_step_window(cls, window: StepWindow) -> None:
        with cls._lock:
            cls._step_window = window

    @classmethod
    def count_forward(cls, counter: object) -> None:
        """Advance the armed window by one forward of counter, before it runs."""
        window = cls._step_window
        if window is None or window.counter is not counter:
            return
        with cls._lock:
            if cls._step_window is not window:
                return
            if window.forwards_left is None:
                cls.start(
                    window.trace_path_template,
                    window.run_id,
                    with_stack=window.with_stack,
                    record_shapes=window.record_shapes,
                )
                window.forwards_left = window.num_steps
            if window.forwards_left == 0:
                cls.stop(run_id=window.run_id)
            else:
                window.forwards_left -= 1

    @classmethod
    def start(
        cls,
        trace_path_template: str,
        run_id: str | None = None,
        *,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ) -> str:
        """Start the profiler; None flags fall back to their env vars."""
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

            # Expected paths
            json_file = f"{trace_path_template}_rank{rank}.trace.json"

            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            if with_stack is None:
                with_stack = os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1"
            if record_shapes is None:
                record_shapes = (
                    os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES") == "1"
                )
            # No ``schedule``: record continuously between start/stop.
            # Expensive flags are opt-in (default off keeps the
            # trace tens of MB; all on can hit multi-GB).
            # note(ratish): without profile_all_threads only the starting thread
            # records cpu ops and spans; the scheduler and worker threads would be
            # kernels without callers. stop() exports, so no on_trace_ready.
            cls._profiler = profile(
                activities=profiler_activities(),
                record_shapes=record_shapes,
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=with_stack,
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
                experimental_config=_ExperimentalConfig(profile_all_threads=True),
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
            window = cls._step_window
            if window is not None and run_id in (None, window.run_id):
                cls._step_window = None
            if cls._profiler is None:
                return None

            rank = cls.get_rank()
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

            try:
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                profiler.export_chrome_trace(json_path)
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
    def get_step_context(cls):
        return nullcontext()


class TorchNPUProfiler(TorchProfiler):

    @classmethod
    def start(
        cls,
        trace_path_template: str,
        run_id: str | None = None,
        *,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ) -> str:
        if with_stack is None:
            with_stack = os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK") == "1"
        if record_shapes is None:
            record_shapes = os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES") == "1"
        with cls._lock:
            trace_path_template = os.path.abspath(trace_path_template)
            rank = cls.get_rank()
            if cls._profiler is not None:
                if run_id is not None and cls._active_run_id == run_id:
                    return trace_path_template

                rank = cls.get_rank()
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

            cls._active_run_id = run_id
            cls._trace_template = trace_path_template

            os.makedirs(trace_path_template, exist_ok=True)

            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            cls._profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    trace_path_template
                ),
                record_shapes=record_shapes,
                profile_memory=os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY")
                == "1",
                with_stack=with_stack,
                with_flops=os.environ.get("SGLANG_TORCH_PROFILER_WITH_FLOPS") == "1",
            )
            cls._profiler.start()

            return trace_path_template

    @classmethod
    def stop(cls, *, run_id: str | None = None) -> dict | None:
        with cls._lock:
            window = cls._step_window
            if window is not None and run_id in (None, window.run_id):
                cls._step_window = None
            if cls._profiler is None:
                return None

            rank = cls.get_rank()
            active = cls._active_run_id
            trace_path = cls._trace_template

            if run_id is not None and active is not None and active != run_id:
                logger.warning(
                    "[Rank %s] Ignoring profiler stop for run_id=%s because active_run_id=%s",
                    rank,
                    run_id,
                    active,
                )
                return None

            profiler = cls._profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""

            return {"trace": trace_path, "table": None}
