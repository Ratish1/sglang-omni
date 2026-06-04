# SPDX-License-Identifier: Apache-2.0
"""Scheduler-thread torch profiler control helpers."""

from __future__ import annotations

import logging
from threading import Event
from typing import Any

from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.profiler.torch_profiler import record_function as _record_function
from sglang_omni.scheduling.messages import IncomingMessage

logger = logging.getLogger(__name__)

_PROFILER_REQUEST_ID = "__profiler__"


def profiler_request_id(action: str, run_id: str | None) -> str:
    return f"{_PROFILER_REQUEST_ID}:{action}:{run_id or ''}"


def is_profiler_message(msg: IncomingMessage) -> bool:
    return msg.type in ("profiler_start", "profiler_stop")


def handle_profiler_message(msg: IncomingMessage) -> bool:
    """Handle a scheduler inbox profiler message on the current thread.

    PyTorch user annotations are thread-sensitive in exported Kineto traces, so
    profiler lifecycle must run on the scheduler thread that emits the Higgs and
    Omni ``record_function`` ranges.
    """

    if msg.type not in ("profiler_start", "profiler_stop"):
        return False

    data = msg.data if isinstance(msg.data, dict) else {}
    done = data.get("done_event")
    if not isinstance(done, Event):
        done = None

    try:
        if msg.type == "profiler_start":
            _handle_profiler_start(data)
        else:
            _handle_profiler_stop(data)
    finally:
        if done is not None:
            done.set()
    return True


def _handle_profiler_start(data: dict[str, Any]) -> None:
    run_id = data.get("run_id")
    trace_path_template = data.get("trace_path_template")
    stage = data.get("stage", "unknown")
    if not isinstance(trace_path_template, str) or not trace_path_template:
        logger.warning("Ignoring profiler_start without trace_path_template")
        return

    TorchProfiler.start(trace_path_template, run_id=run_id)
    with _record_function(f"sglang_omni.profiler.canary_scheduler_start.{stage}"):
        pass


def _handle_profiler_stop(data: dict[str, Any]) -> None:
    run_id = data.get("run_id")
    if TorchProfiler.is_active() and (
        run_id is None or TorchProfiler.get_active_run_id() == run_id
    ):
        TorchProfiler.stop(run_id=run_id)
