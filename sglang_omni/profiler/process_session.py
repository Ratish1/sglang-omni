# SPDX-License-Identifier: Apache-2.0
"""Process-scoped profiler and CUDA synchronization-debug lifecycle.

Multiple logical stages may share one OS process.  Torch Profiler and
``torch.cuda.set_sync_debug_mode`` are process-global, so stage-local start/stop
calls must join one process session instead of racing independent lifecycles.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from typing import ClassVar

import torch

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.proto.messages import CUDA_SYNC_DEBUG_MODES

logger = logging.getLogger(__name__)


class ProcessProfilerSession:
    """Own one profiling/debug session per process and run id."""

    _lock = threading.RLock()
    _run_id: str | None = None
    _participants: ClassVar[set[str]] = set()
    _enable_torch = False
    _event_dir: str | None = None
    _cuda_capable_process = False
    _cuda_sync_debug_mode = "default"
    _cuda_sync_debug_applied = False

    @classmethod
    def active_run_id(cls) -> str | None:
        with cls._lock:
            return cls._run_id

    @classmethod
    def participants(cls) -> frozenset[str]:
        with cls._lock:
            return frozenset(cls._participants)

    @classmethod
    def start(
        cls,
        *,
        participant: str,
        run_id: str,
        trace_path_template: str,
        event_dir: str | None,
        enable_torch: bool,
        cuda_capable_process: bool,
        cuda_sync_debug_mode: str,
    ) -> dict:
        if cuda_sync_debug_mode not in CUDA_SYNC_DEBUG_MODES:
            raise ValueError(
                "cuda_sync_debug_mode must be one of "
                f"{sorted(CUDA_SYNC_DEBUG_MODES)}, got {cuda_sync_debug_mode!r}"
            )

        with cls._lock:
            if cls._run_id == run_id:
                cls._join_existing_unlocked(
                    participant=participant,
                    event_dir=event_dir,
                    enable_torch=enable_torch,
                    cuda_capable_process=cuda_capable_process,
                    cuda_sync_debug_mode=cuda_sync_debug_mode,
                )
                return cls._status_unlocked(joined=True)

            if cls._run_id is not None:
                logger.warning(
                    "Replacing active process profiler run_id=%s with run_id=%s",
                    cls._run_id,
                    run_id,
                )
                cls._close_unlocked(reason="replaced by a new run")

            cls._run_id = run_id
            cls._participants = {participant}
            cls._enable_torch = bool(enable_torch)
            cls._event_dir = event_dir
            cls._cuda_capable_process = bool(cuda_capable_process)
            cls._cuda_sync_debug_mode = cuda_sync_debug_mode

            try:
                if enable_torch:
                    TorchProfiler.start(trace_path_template, run_id=run_id)
                # Profiler setup may initialize CUDA machinery of its own. Arm
                # the detector only after that setup so warnings/errors belong
                # to target traffic rather than the instrumentation lifecycle.
                cls._set_cuda_sync_debug_unlocked(
                    cuda_sync_debug_mode,
                    cuda_capable_process=cls._cuda_capable_process,
                )
                if event_dir is not None:
                    get_recorder().start(
                        run_id=run_id,
                        event_dir=event_dir,
                        stage=participant,
                    )
            except Exception:
                cls._close_unlocked(reason="start failed")
                raise

            logger.info(
                "Process profiler session started run_id=%s pid=%d rank=%s "
                "process=%s participant=%s torch=%s cuda_capable=%s "
                "cuda_sync_debug_mode=%s applied=%s",
                run_id,
                os.getpid(),
                os.environ.get("RANK", "0"),
                multiprocessing.current_process().name,
                participant,
                enable_torch,
                cls._cuda_capable_process,
                cuda_sync_debug_mode,
                cls._cuda_sync_debug_applied,
            )
            return cls._status_unlocked(joined=False)

    @classmethod
    def _join_existing_unlocked(
        cls,
        *,
        participant: str,
        event_dir: str | None,
        enable_torch: bool,
        cuda_capable_process: bool,
        cuda_sync_debug_mode: str,
    ) -> None:
        if (
            cls._enable_torch != bool(enable_torch)
            or cls._event_dir != event_dir
            or cls._cuda_capable_process != bool(cuda_capable_process)
            or cls._cuda_sync_debug_mode != cuda_sync_debug_mode
        ):
            raise RuntimeError(
                "Conflicting profiler configuration for active run "
                f"{cls._run_id!r}: existing=(torch={cls._enable_torch}, "
                f"event_dir={cls._event_dir!r}, "
                f"cuda_capable={cls._cuda_capable_process}, "
                f"cuda_sync_debug_mode={cls._cuda_sync_debug_mode!r}), "
                f"new=(torch={bool(enable_torch)}, event_dir={event_dir!r}, "
                f"cuda_capable={bool(cuda_capable_process)}, "
                f"cuda_sync_debug_mode={cuda_sync_debug_mode!r})"
            )
        if participant in cls._participants:
            return
        if event_dir is not None:
            get_recorder().start(
                run_id=cls._run_id or "",
                event_dir=event_dir,
                stage=participant,
            )
        cls._participants.add(participant)
        logger.info(
            "Process profiler session joined run_id=%s pid=%d participant=%s "
            "participants=%s",
            cls._run_id,
            os.getpid(),
            participant,
            sorted(cls._participants),
        )

    @classmethod
    def stop(cls, *, participant: str, run_id: str | None = None) -> dict:
        """Leave a stage participant; close after the last joined stage leaves."""
        with cls._lock:
            if cls._run_id is None:
                return {"stopped": False, "reason": "no active process session"}
            if run_id is not None and run_id != cls._run_id:
                logger.warning(
                    "Ignoring process profiler stop for run_id=%s; active run_id=%s",
                    run_id,
                    cls._run_id,
                )
                return {
                    "stopped": False,
                    "reason": "run_id mismatch",
                    "active_run_id": cls._run_id,
                }
            if participant not in cls._participants:
                return {
                    "stopped": False,
                    "reason": "participant is not joined",
                    "active_run_id": cls._run_id,
                    "participants": sorted(cls._participants),
                }
            cls._participants.remove(participant)
            if cls._participants:
                logger.info(
                    "Process profiler participant left run_id=%s pid=%d "
                    "participant=%s remaining=%s",
                    cls._run_id,
                    os.getpid(),
                    participant,
                    sorted(cls._participants),
                )
                return cls._status_unlocked(joined=True) | {"stopped": False}
            return cls._close_unlocked(
                reason=f"last participant {participant} stopped",
                raise_on_error=True,
            )

    @classmethod
    def force_stop(cls, *, reason: str) -> None:
        """Reset process-global state during process failure or teardown."""
        with cls._lock:
            if cls._run_id is not None:
                cls._close_unlocked(reason=reason)

    @classmethod
    def _status_unlocked(cls, *, joined: bool) -> dict:
        recorder = get_recorder()
        return {
            "run_id": cls._run_id,
            "pid": os.getpid(),
            "rank": os.environ.get("RANK", "0"),
            "process": multiprocessing.current_process().name,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "participants": sorted(cls._participants),
            "joined_existing": joined,
            "torch_enabled": cls._enable_torch,
            "torch_active": TorchProfiler.is_active(),
            "event_path": recorder.active_path(),
            "cuda_sync_debug_mode": cls._cuda_sync_debug_mode,
            "cuda_sync_debug_applied": cls._cuda_sync_debug_applied,
        }

    @classmethod
    def _set_cuda_sync_debug_unlocked(
        cls,
        mode: str,
        *,
        cuda_capable_process: bool,
    ) -> None:
        cls._cuda_sync_debug_applied = False
        if mode == "default":
            return
        if not cuda_capable_process:
            logger.info(
                "CUDA sync-debug skipped for non-CUDA process run_id=%s pid=%d "
                "process=%s",
                cls._run_id,
                os.getpid(),
                multiprocessing.current_process().name,
            )
            return
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA sync-debug requested for run_id=%s but CUDA is unavailable "
                "in pid=%d",
                cls._run_id,
                os.getpid(),
            )
            return
        torch.cuda.set_sync_debug_mode(mode)
        cls._cuda_sync_debug_applied = True

    @classmethod
    def _close_unlocked(
        cls,
        *,
        reason: str,
        raise_on_error: bool = False,
    ) -> dict:
        run_id = cls._run_id
        participants = sorted(cls._participants)
        trace_result: dict | None = None
        event_path: str | None = None
        errors: list[str] = []

        # Reset the process-global detector before profiler export or other
        # teardown work can create diagnostic self-hits.
        if cls._cuda_sync_debug_applied:
            try:
                torch.cuda.set_sync_debug_mode("default")
            except Exception:
                errors.append("failed to reset CUDA sync-debug")
                logger.warning(
                    "Failed to reset CUDA sync-debug for run_id=%s pid=%d",
                    run_id,
                    os.getpid(),
                    exc_info=True,
                )

        try:
            if cls._enable_torch:
                if not TorchProfiler.is_active():
                    raise RuntimeError("Torch profiler is not active at session stop")
                trace_result = TorchProfiler.stop(run_id=run_id)
                if trace_result is None:
                    raise RuntimeError("Torch profiler did not finalize a trace")
        except Exception as exc:
            errors.append(str(exc))
            logger.warning(
                "Failed to stop Torch profiler for run_id=%s pid=%d",
                run_id,
                os.getpid(),
                exc_info=True,
            )

        try:
            recorder = get_recorder()
            if recorder.is_active() and (
                run_id is None or recorder.active_run_id() == run_id
            ):
                event_path = recorder.stop(run_id=run_id)
        except Exception as exc:
            errors.append(str(exc))
            logger.warning(
                "Failed to stop request event recorder for run_id=%s pid=%d",
                run_id,
                os.getpid(),
                exc_info=True,
            )
        finally:
            # Clear ownership even when an exporter or recorder fails.  The
            # next run must never inherit stale process-global debug state.
            cls._run_id = None
            cls._participants = set()
            cls._enable_torch = False
            cls._event_dir = None
            cls._cuda_capable_process = False
            cls._cuda_sync_debug_mode = "default"
            cls._cuda_sync_debug_applied = False

        logger.info(
            "Process profiler session stopped run_id=%s pid=%d rank=%s "
            "process=%s participants=%s reason=%s",
            run_id,
            os.getpid(),
            os.environ.get("RANK", "0"),
            multiprocessing.current_process().name,
            participants,
            reason,
        )
        result = {
            "run_id": run_id,
            "pid": os.getpid(),
            "rank": os.environ.get("RANK", "0"),
            "process": multiprocessing.current_process().name,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "participants": participants,
            "stopped": True,
            "reason": reason,
            "trace": None if trace_result is None else trace_result.get("trace"),
            "event_path": event_path,
            "errors": errors,
        }
        if errors and raise_on_error:
            raise RuntimeError("; ".join(errors))
        return result
