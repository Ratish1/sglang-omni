# SPDX-License-Identifier: Apache-2.0
"""Synthetic Kineto trace tests for CUDA synchronization attribution."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from benchmarks.profiling.analyze_cuda_sync_trace import (
    aggregate_occurrences,
    analyze_trace,
    iter_trace_events,
    write_analysis,
)


def _event(
    name: str,
    category: str,
    ts: float,
    dur: float,
    *,
    pid: int = 10,
    tid: int = 20,
    args: dict | None = None,
) -> dict:
    return {
        "ph": "X",
        "name": name,
        "cat": category,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


def _write_trace(path: Path) -> list[dict]:
    events = [
        _event("qwen3_tts.sampling_metadata.h2d", "user_annotation", 90, 80),
        _event("model.py(10): prepare_decode_buffers", "python_function", 95, 70),
        _event("aten::to", "cpu_op", 100, 50),
        _event("aten::_to_copy", "cpu_op", 105, 40),
        _event("aten::copy_", "cpu_op", 110, 30),
        _event(
            "cudaMemcpyAsync",
            "cuda_runtime",
            112,
            3,
            args={"correlation": 7},
        ),
        _event(
            "cudaStreamSynchronize",
            "cuda_runtime",
            115,
            25,
            args={"correlation": 7, "stream": 7},
        ),
        _event(
            "Memcpy HtoD (Pageable -> Device)",
            "gpu_memcpy",
            125,
            10,
            pid=0,
            tid=7,
            args={"correlation": "7", "stream": 7, "device": 0, "bytes": 4096},
        ),
        _event(
            "cudaLaunchKernel",
            "cuda_runtime",
            145,
            2,
            args={"correlation": 8},
        ),
        _event(
            "next_kernel",
            "kernel",
            155,
            10,
            pid=0,
            tid=7,
            args={"correlation": 8, "stream": 7, "device": 0},
        ),
        # A non-duration metadata event verifies that the parser ignores it.
        {"ph": "M", "name": "thread_name", "pid": 10, "tid": 20},
    ]
    document = {"schemaVersion": 1, "traceEvents": events, "displayTimeUnit": "ms"}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(document, stream)
    return events


def test_streaming_parser_handles_gzip_and_tiny_chunks(tmp_path: Path) -> None:
    trace = tmp_path / "synthetic.trace.json.gz"
    expected = _write_trace(trace)
    parsed = list(iter_trace_events(trace, chunk_size=7))
    assert parsed == expected


def test_analyzer_attributes_transfer_stack_and_post_sync_bubble(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "synthetic.trace.json.gz"
    _write_trace(trace)

    occurrences = analyze_trace(trace)

    assert len(occurrences) == 1
    occurrence = occurrences[0]
    assert occurrence["semantic_range"] == "qwen3_tts.sampling_metadata.h2d"
    assert occurrence["parent_cpu_op"] == "aten::copy_"
    assert occurrence["python_stack_innermost_first"] == [
        "model.py(10): prepare_decode_buffers"
    ]
    assert occurrence["transfer"]["direction"] == "HtoD"
    assert occurrence["transfer"]["bytes"] == 4096
    assert occurrence["attribution"]["transfer"] == "correlation"
    assert occurrence["prior_waited_gpu"]["end_us"] == 135
    assert occurrence["next_causal_gpu"]["name"] == "next_kernel"
    assert occurrence["metrics"] == {
        "sync_wait_us": 25,
        "post_sync_gpu_bubble_us": 20,
        "host_launch_gap_after_sync_us": 5,
        "queue_horizon_at_sync_start_us": 20,
    }


def test_aggregate_and_output_files_are_machine_readable(tmp_path: Path) -> None:
    trace = tmp_path / "synthetic.trace.json.gz"
    _write_trace(trace)
    occurrences = analyze_trace(trace)

    aggregates = aggregate_occurrences(occurrences)
    assert aggregates[0]["count"] == 1
    assert aggregates[0]["sync_wait_total_us"] == 25
    assert aggregates[0]["post_sync_bubble_p95_us"] == 20

    outputs = write_analysis(occurrences, tmp_path / "analysis")
    for output in outputs.values():
        assert Path(output).is_file()
    report = json.loads(Path(outputs["occurrences"]).read_text())
    assert report["schema_version"] == 1
    assert len(report["occurrences"]) == 1


def test_device_sync_uses_latest_completion_only_for_unambiguous_device(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "device-sync.trace.json"
    events = [
        _event(
            "previous_kernel",
            "kernel",
            100,
            20,
            pid=0,
            tid=5,
            args={"correlation": 1, "stream": 5, "device": 0},
        ),
        _event("cudaDeviceSynchronize", "cuda_runtime", 122, 8),
        _event(
            "cudaLaunchKernel",
            "cuda_runtime",
            135,
            2,
            args={"correlation": 2},
        ),
        _event(
            "next_kernel",
            "kernel",
            145,
            10,
            pid=0,
            tid=5,
            args={"correlation": 2, "stream": 5, "device": 0},
        ),
    ]
    trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")

    occurrence = analyze_trace(trace)[0]

    assert occurrence["prior_waited_gpu"]["name"] == "previous_kernel"
    assert occurrence["attribution"]["prior_waited_gpu"] == (
        "latest_completion_before_device_sync_return"
    )
    assert occurrence["metrics"]["post_sync_gpu_bubble_us"] == 25


def test_parser_rejects_a_document_without_trace_events(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"events": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="traceEvents"):
        list(iter_trace_events(invalid, chunk_size=3))
