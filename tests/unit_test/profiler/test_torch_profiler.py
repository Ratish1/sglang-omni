# SPDX-License-Identifier: Apache-2.0
"""TorchProfiler step windows and thread coverage on a real torch profiler."""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

import pytest
import torch

from sglang_omni.profiler.torch_profiler import StepWindow, TorchProfiler


@pytest.fixture(autouse=True)
def stop_profiler():
    yield
    TorchProfiler.stop()


def read_spans(trace_path_template: str) -> list[tuple[str, int]]:
    """Span names and tids from the exported trace, once the background gzip lands."""
    json_path = Path(f"{trace_path_template}_rank0.trace.json")
    gz_path = Path(f"{json_path}.gz")
    deadline = time.monotonic() + 30
    while json_path.exists() or not gz_path.exists():
        assert time.monotonic() < deadline, f"no trace at {gz_path}"
        time.sleep(0.1)
    with gzip.open(gz_path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    return [
        (event["name"], event["tid"])
        for event in events
        if event.get("cat") == "user_annotation"
    ]


def window(counter: object, template: str, num_steps: int) -> StepWindow:
    return StepWindow(
        counter=counter,
        trace_path_template=template,
        run_id="window",
        num_steps=num_steps,
        with_stack=None,
        record_shapes=None,
    )


def test_step_window_records_exactly_its_forwards_then_stops(tmp_path: Path) -> None:
    counter = object()
    template = str(tmp_path / "trace")
    TorchProfiler.arm_step_window(window(counter, template, num_steps=3))

    TorchProfiler.count_forward(object())
    assert not TorchProfiler.is_active()
    for forward in range(1, 6):
        TorchProfiler.count_forward(counter)
        with torch.profiler.record_function(f"forward_{forward}"):
            torch.ones(4).sum()

    assert not TorchProfiler.is_active()
    names = sorted(name for name, _ in read_spans(template))
    assert names == ["forward_1", "forward_2", "forward_3"]


def test_stop_before_the_first_forward_disarms_the_window(tmp_path: Path) -> None:
    counter = object()
    template = str(tmp_path / "trace")
    TorchProfiler.arm_step_window(window(counter, template, num_steps=2))

    TorchProfiler.stop(run_id="another-run")
    TorchProfiler.stop()
    TorchProfiler.count_forward(counter)

    assert not TorchProfiler.is_active()
    assert not list(tmp_path.iterdir())


def test_a_thread_started_before_the_profiler_records_its_spans(
    tmp_path: Path,
) -> None:
    template = str(tmp_path / "trace")
    go = threading.Event()

    def worker() -> None:
        go.wait()
        with torch.profiler.record_function("worker_span"):
            torch.ones(4).sum()

    thread = threading.Thread(target=worker)
    thread.start()
    TorchProfiler.start(template, run_id="threads")
    with torch.profiler.record_function("main_span"):
        torch.ones(4).sum()
    go.set()
    thread.join()
    TorchProfiler.stop()

    spans = dict(read_spans(template))
    assert spans["worker_span"] == thread.native_id
    assert spans["main_span"] == threading.get_native_id()
