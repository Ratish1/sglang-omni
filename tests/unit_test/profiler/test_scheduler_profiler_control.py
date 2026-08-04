# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

import sglang_omni.scheduling.profiler_control as control
from sglang_omni.scheduling.profiler_control import (
    handle_profiler_message,
    make_profiler_command,
)


def test_scheduler_command_runs_lifecycle_on_consuming_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def start(template, run_id=None, config=None):  # noqa: ANN001, ANN202
        calls.append(("start", threading.current_thread().name))
        return f"{template}.json.gz"

    monkeypatch.setattr(control.TorchProfiler, "start", start)
    monkeypatch.setattr(
        control.TorchProfiler,
        "snapshot",
        lambda: {"active": True},
    )

    message, result_queue = make_profiler_command(
        action="profiler_start",
        op_id="op",
        run_id="run",
        stage="asr",
        role="single",
        trace_path_template="/tmp/trace",
        torch_config={"include_cuda": False},
    )

    thread = threading.Thread(
        target=lambda: handle_profiler_message(message),
        name="scheduler-asr",
    )
    thread.start()
    thread.join()
    result = result_queue.get_nowait()

    assert result["success"]
    assert result["owner_thread"] == "scheduler-asr"
    assert calls == [("start", "scheduler-asr")]


def test_stop_flushes_async_work_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(control.TorchProfiler, "is_active", lambda: True)
    monkeypatch.setattr(
        control.TorchProfiler,
        "stop",
        lambda run_id=None: {
            "trace": "/tmp/trace.json.gz",
            "export_error": None,
        },
    )
    message, result_queue = make_profiler_command(
        action="profiler_stop",
        op_id="op",
        run_id="run",
        stage="asr",
        role="single",
    )

    handle_profiler_message(
        message,
        before_stop=lambda: order.append("flush"),
    )
    result = result_queue.get_nowait()
    assert result["success"]
    assert order == ["flush"]
