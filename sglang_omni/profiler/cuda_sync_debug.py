# SPDX-License-Identifier: Apache-2.0
"""Process-scoped lifecycle for PyTorch's CUDA synchronization detector."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading

import torch

logger = logging.getLogger(__name__)

CUDA_SYNC_DEBUG_MODES = frozenset({"default", "warn", "error"})


class CudaSyncDebug:
    """Arm one synchronization-debug run per CUDA-owning process.

    ``torch.cuda.set_sync_debug_mode`` is process-global. Multiple colocated
    stages therefore join the same run instead of independently owning the
    underlying PyTorch state.
    """

    _lock = threading.Lock()
    _run_id: str | None = None
    _mode = "default"

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        mode: str,
        participant: str,
        cuda_capable_process: bool,
    ) -> bool:
        if mode not in CUDA_SYNC_DEBUG_MODES:
            raise ValueError(
                "cuda_sync_debug_mode must be one of "
                f"{sorted(CUDA_SYNC_DEBUG_MODES)}, got {mode!r}"
            )

        if mode == "default" or not cuda_capable_process:
            return False
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA sync debug requested but CUDA is unavailable "
                "run_id=%s pid=%d participant=%s",
                run_id,
                os.getpid(),
                participant,
            )
            return False

        with cls._lock:
            if cls._run_id == run_id:
                if cls._mode != mode:
                    raise RuntimeError(
                        "Conflicting CUDA sync-debug modes for run "
                        f"{run_id!r}: {cls._mode!r} != {mode!r}"
                    )
                return True
            if cls._run_id is not None:
                cls._reset_unlocked(reason=f"replaced by run {run_id}")

            torch.cuda.set_sync_debug_mode(mode)
            cls._run_id = run_id
            cls._mode = mode
            logger.info(
                "CUDA sync debug enabled run_id=%s mode=%s pid=%d rank=%s "
                "process=%s participant=%s",
                run_id,
                mode,
                os.getpid(),
                os.environ.get("RANK", "0"),
                multiprocessing.current_process().name,
                participant,
            )
            return True

    @classmethod
    def stop(cls, *, run_id: str | None, reason: str) -> bool:
        with cls._lock:
            if cls._run_id is None:
                return False
            if run_id is not None and run_id != cls._run_id:
                return False
            cls._reset_unlocked(reason=reason)
            return True

    @classmethod
    def _reset_unlocked(cls, *, reason: str) -> None:
        run_id = cls._run_id
        mode = cls._mode
        torch.cuda.set_sync_debug_mode("default")

        cls._run_id = None
        cls._mode = "default"
        logger.info(
            "CUDA sync debug disabled run_id=%s mode=%s pid=%d reason=%s",
            run_id,
            mode,
            os.getpid(),
            reason,
        )
