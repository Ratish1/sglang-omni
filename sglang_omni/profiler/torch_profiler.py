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

import torch
from torch.profiler import (
    ProfilerActivity,
    _ExperimentalConfig,
    profile,
    record_function,
)

from .base_profiler import ProfilerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    _active_run_id: str | None = None
    _host_memory_start: dict | None = None
    _host_memory_start_ns: int | None = None
    _lock = threading.Lock()

    @staticmethod
    def _host_memory_snapshot() -> dict:
        """Read pinned-host allocator counters without initializing CUDA."""

        if not torch.cuda.is_initialized():
            return {
                "available": False,
                "reason": "cuda_not_initialized",
                "stats": {},
            }
        try:
            stats = {
                str(key): value
                for key, value in torch.cuda.host_memory_stats().items()
                if isinstance(value, (int, float))
            }
        except Exception as exc:
            logger.warning("Failed to read CUDA host-memory statistics: %s", exc)
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "stats": {},
            }
        return {"available": True, "reason": None, "stats": stats}

    @staticmethod
    def _host_memory_delta(start: dict, end: dict) -> dict[str, int | float]:
        start_stats = start.get("stats", {})
        end_stats = end.get("stats", {})
        return {
            key: end_stats[key] - start_stats[key]
            for key in sorted(start_stats.keys() & end_stats.keys())
            if isinstance(start_stats[key], (int, float))
            and isinstance(end_stats[key], (int, float))
        }

    @classmethod
    def _write_host_memory_artifact(
        cls,
        *,
        base_path: str,
        rank: int,
        run_id: str | None,
        end_snapshot: dict,
        end_ns: int,
    ) -> str:
        path = f"{base_path}.host_memory.json"
        start_snapshot = cls._host_memory_start or {
            "available": False,
            "reason": "start_snapshot_missing",
            "stats": {},
        }
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "pid": os.getpid(),
            "rank": rank,
            "start_timestamp_ns": cls._host_memory_start_ns,
            "end_timestamp_ns": end_ns,
            "start": start_snapshot,
            "end": end_snapshot,
            "delta": cls._host_memory_delta(start_snapshot, end_snapshot),
            "notes": [
                "Counters are process-local CUDA pinned-host allocator statistics.",
                (
                    "Current-value deltas bound allocator growth during the "
                    "profile window."
                ),
                (
                    "Peak counters are not window-local because profiling does "
                    "not reset global allocator statistics."
                ),
                "Pinned-host statistics do not measure GPU memory.",
            ],
        }
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
        logger.info("[Rank %s] Host-memory statistics exported to %s", rank, path)
        return path

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
                cls._host_memory_start = None
                cls._host_memory_start_ns = None

            # 2. Make path absolute
            trace_path_template = os.path.abspath(trace_path_template)
            cls._trace_template = trace_path_template
            cls._active_run_id = run_id
            cls._host_memory_start_ns = time.time_ns()
            cls._host_memory_start = cls._host_memory_snapshot()

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
            # Omni starts profiling from its control-plane thread after its
            # long-lived scheduler and preprocessing threads already exist.
            # Kineto's CPU operator callbacks are otherwise thread-local, so
            # collect all threads to retain their ATen and semantic ranges.
            cls._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=trace_handler,
                experimental_config=_ExperimentalConfig(profile_all_threads=True),
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
            host_memory_end_ns = time.time_ns()
            host_memory_end = cls._host_memory_snapshot()

            profiler = cls._profiler
            try:
                profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

            # No schedule → on_trace_ready isn't fired on stop, so
            # export here.
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

            host_memory_path = None
            try:
                host_memory_path = cls._write_host_memory_artifact(
                    base_path=base_path,
                    rank=rank,
                    run_id=active,
                    end_snapshot=host_memory_end,
                    end_ns=host_memory_end_ns,
                )
            except Exception as e:
                logger.warning(
                    "[Rank %s] Failed to export host-memory statistics: %s", rank, e
                )

            cls._profiler = None
            cls._active_run_id = None
            cls._trace_template = ""
            cls._host_memory_start = None
            cls._host_memory_start_ns = None

            return {
                "trace": gz_path,
                "table": None,
                "host_memory": host_memory_path,
            }

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
        """Create a semantic range only while this process is profiled."""

        if cls._profiler is None:
            return nullcontext()
        return record_function(name)
