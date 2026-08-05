# SPDX-License-Identifier: Apache-2.0
"""Scheduler-thread profiler lifecycle commands.

Messages in this module are process-local objects placed on a scheduler inbox;
they are never serialized or broadcast through SGLang's request protocol.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable
from typing import Any

from sglang_omni.profiler.torch_profiler import TorchProfiler, TorchProfilerConfig
from sglang_omni.profiler.trace_ranges import (
    nvtx_window_snapshot,
    start_nvtx_window,
    stop_nvtx_window,
)
from sglang_omni.scheduling.messages import IncomingMessage

PROFILER_START = "profiler_start"
PROFILER_STOP = "profiler_stop"
_PROFILER_REQUEST_PREFIX = "__profiler__"


def make_profiler_command(
    *,
    action: str,
    op_id: str,
    run_id: str | None,
    stage: str,
    role: str,
    trace_path_template: str | None = None,
    torch_config: dict[str, Any] | None = None,
    enable_torch: bool = True,
    enable_nvtx: bool = False,
) -> tuple[IncomingMessage, queue.Queue[dict[str, Any]]]:
    if action not in {PROFILER_START, PROFILER_STOP}:
        raise ValueError(f"unsupported profiler action: {action}")
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
    command = IncomingMessage(
        request_id=f"{_PROFILER_REQUEST_PREFIX}:{op_id}",
        type=action,
        data={
            "op_id": op_id,
            "run_id": run_id,
            "stage": stage,
            "role": role,
            "trace_path_template": trace_path_template,
            "torch_config": torch_config,
            "enable_torch": enable_torch,
            "enable_nvtx": enable_nvtx,
            "result_queue": result_queue,
        },
    )
    return command, result_queue


def is_profiler_message(msg: IncomingMessage) -> bool:
    return msg.type in {PROFILER_START, PROFILER_STOP}


def handle_profiler_message(
    msg: IncomingMessage,
    *,
    before_stop: Callable[[], None] | None = None,
) -> bool:
    """Handle a profiler command on the current scheduler thread."""

    if not is_profiler_message(msg):
        return False
    data = msg.data if isinstance(msg.data, dict) else {}
    result_queue = data.get("result_queue")
    result: dict[str, Any] = {
        "op_id": data.get("op_id"),
        "run_id": data.get("run_id"),
        "stage": data.get("stage"),
        "role": data.get("role"),
        "rank": _rank(),
        "pid": os.getpid(),
        "owner_tid": threading.get_native_id(),
        "owner_thread": threading.current_thread().name,
        "action": "start" if msg.type == PROFILER_START else "stop",
        "success": False,
    }
    try:
        if msg.type == PROFILER_START:
            expected_path = None
            torch_state = None
            if data.get("enable_torch", True):
                template = data.get("trace_path_template")
                if not isinstance(template, str) or not template:
                    raise ValueError("trace_path_template is required")
                config = TorchProfilerConfig.from_dict(data.get("torch_config"))
                expected_path = TorchProfiler.start(
                    template,
                    run_id=data.get("run_id"),
                    config=config,
                    owner_label="scheduler",
                )
                torch_state = TorchProfiler.snapshot()
            nvtx_state = None
            if data.get("enable_nvtx", False):
                nvtx_state = start_nvtx_window(data.get("run_id"))
            result.update(
                {
                    "success": True,
                    "status": "active",
                    "trace": expected_path,
                    "torch": torch_state,
                    "nvtx": nvtx_state,
                }
            )
        else:
            if (
                TorchProfiler.is_active() or nvtx_window_snapshot()["active"]
            ) and before_stop is not None:
                before_stop()
            stop_errors: list[str] = []
            nvtx_state = None
            stopped = None
            try:
                nvtx_state = stop_nvtx_window(data.get("run_id"))
            except Exception as exc:
                stop_errors.append(f"NVTX stop failed: {exc}")
            try:
                stopped = TorchProfiler.stop(run_id=data.get("run_id"))
            except Exception as exc:
                stop_errors.append(f"torch stop failed: {exc}")
            result.update(
                {
                    "success": not stop_errors,
                    "status": (
                        "failed"
                        if stop_errors
                        else ("exported" if stopped is not None else "inactive")
                    ),
                    "trace": stopped.get("trace") if stopped is not None else None,
                    "torch": stopped,
                    "nvtx": nvtx_state,
                }
            )
            if stop_errors:
                result["error"] = "; ".join(stop_errors)
            if stopped is not None and stopped.get("export_error"):
                result["success"] = False
                result["status"] = "failed"
                result["error"] = stopped["export_error"]
    except Exception as exc:
        cleanup_errors: list[str] = []
        if msg.type == PROFILER_START:
            try:
                stop_nvtx_window(data.get("run_id"))
            except Exception as cleanup_exc:
                cleanup_errors.append(f"NVTX rollback failed: {cleanup_exc}")
            try:
                TorchProfiler.stop(run_id=data.get("run_id"))
            except Exception as cleanup_exc:
                cleanup_errors.append(f"torch rollback failed: {cleanup_exc}")
        error = str(exc) or type(exc).__name__
        if cleanup_errors:
            error = f"{error}; {'; '.join(cleanup_errors)}"
        result.update(
            {
                "status": "failed",
                "error": error,
            }
        )
    finally:
        if isinstance(result_queue, queue.Queue):
            result_queue.put_nowait(result)
    return True


def profiler_step() -> dict[str, Any] | None:
    """Advance a live owner-thread profiler by one scheduler step."""

    if TorchProfiler.is_active() and not TorchProfiler.is_owner_thread():
        return None
    return TorchProfiler.step()


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


__all__ = [
    "PROFILER_START",
    "PROFILER_STOP",
    "handle_profiler_message",
    "is_profiler_message",
    "make_profiler_command",
    "profiler_step",
]
