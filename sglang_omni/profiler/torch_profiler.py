# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Omni diffusion profiler (Apache 2.0 licensed)
# Original files:
# - https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/torch_profiler.py

import logging
import os
import subprocess
import threading
from contextlib import nullcontext

from torch.profiler import ProfilerActivity, profile, record_function

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses ``on_trace_ready`` to export exactly one finalized gzip artifact.

    ``stop`` intentionally waits for export and compression.  The profiler
    control response is an artifact-completion acknowledgement, so returning
    while a background gzip still owns the JSON would be incorrect.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    _active_run_id: str | None = None
    _export_result: dict | None = None
    _export_error: Exception | None = None
    _trace_handler_called: bool = False
    _lock = threading.Lock()

    @classmethod
    def _export_trace(cls, profiler: profile, json_path: str, rank: int) -> None:
        """Export and compress one trace, recording rather than hiding errors."""
        if cls._trace_handler_called:
            return
        cls._trace_handler_called = True
        try:
            profiler.export_chrome_trace(json_path)
            logger.info("[Rank %s] Trace exported to %s", rank, json_path)
            subprocess.run(["gzip", "-f", json_path], check=True)
            gz_path = f"{json_path}.gz"
            cls._export_result = {"trace": gz_path, "table": None}
            logger.info("[Rank %s] Trace compression completed: %s", rank, gz_path)
        except Exception as exc:
            cls._export_error = exc
            logger.warning("[Rank %s] Failed to export trace: %s", rank, exc)

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
            cls._export_result = None
            cls._export_error = None
            cls._trace_handler_called = False

            # Expected paths
            json_file = f"{trace_path_template}_rank{rank}.trace.json"

            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            logger.info(
                "[Rank %s] Starting End-to-End Torch profiler (run_id=%s)", rank, run_id
            )

            # 3. Define the on_trace_ready handler.  Some Torch versions invoke
            # it from stop() even without a schedule; older versions do not.
            def trace_handler(profiler):
                cls._export_trace(profiler, json_file, rank)

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
                cls._export_error = e

            # Torch 2.11 invokes on_trace_ready from stop() even without a
            # schedule. Older builds may not, so use the handler exactly once
            # as a compatibility fallback instead of unconditionally exporting
            # a second time.
            if not cls._trace_handler_called and cls._export_error is None:
                cls._export_trace(profiler, json_path, rank)

            result = cls._export_result
            error = cls._export_error
            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""
            cls._export_result = None
            cls._export_error = None
            cls._trace_handler_called = False

            if error is not None:
                raise RuntimeError(
                    f"Torch profiler export failed for run_id={active}: {error}"
                ) from error
            if result is None or not os.path.isfile(gz_path):
                raise RuntimeError(
                    f"Torch profiler produced no finalized trace for run_id={active}"
                )
            return result

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

    @classmethod
    def record_function(cls, name: str):
        """Create a semantic range only while this process is being profiled.

        Qwen3-TTS invokes these scopes in request and decode hot paths. Returning
        a null context while inactive keeps production execution free of the
        dispatcher overhead that an unconditional ``record_function`` incurs.
        """
        if cls._profiler is None:
            return nullcontext()
        return record_function(name)
