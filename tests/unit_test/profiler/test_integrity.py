# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from sglang_omni.profiler.integrity import (
    validate_event_file,
    validate_request_lifecycle,
    validate_stop_response,
    validate_torch_trace,
)


def _write_trace(path: Path) -> None:
    payload = {
        "traceEvents": [
            {
                "name": "sglang_omni.profiler.scheduler_owner.asr",
                "cat": "cpu_op",
                "pid": 10,
                "tid": 20,
            },
            {
                "name": "kernel",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
            },
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_events(path: Path, run_id: str = "run") -> None:
    event = {
        "request_id": "r1",
        "stage": "asr",
        "event_name": "request_build.audio_load_start",
        "timestamp_ns": 1,
        "monotonic_ns": 2,
        "run_id": run_id,
        "pid": 10,
        "native_tid": 20,
        "thread_name": "worker",
        "metadata": {},
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")


def test_trace_and_event_integrity_accept_complete_artifacts(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json.gz"
    events = tmp_path / "events.jsonl"
    _write_trace(trace)
    _write_events(events)

    assert validate_torch_trace(str(trace)).valid
    assert validate_event_file(str(events), run_id="run").valid


def test_stop_response_requires_finalized_complete_schedule(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json.gz"
    events = tmp_path / "events.jsonl"
    _write_trace(trace)
    _write_events(events)
    response = {
        "run_id": "run",
        "coordinator_event_path": str(events),
        "manifest": {
            "success": True,
            "missing_stages": [],
            "stages": [
                {
                    "stage": "asr",
                    "success": True,
                    "targets": [
                        {
                            "rank": 0,
                            "pid": 10,
                            "success": True,
                            "torch": {
                                "trace": str(trace),
                                "trace_finalized": True,
                                "schedule_complete": True,
                                "step_count": 22,
                                "expected_steps": 22,
                                "export_error": None,
                            },
                            "events": {
                                "path": str(events),
                                "dropped_events": 0,
                            },
                        }
                    ],
                }
            ],
        },
    }

    report = validate_stop_response(response)
    assert report.valid, report.errors

    response["manifest"]["stages"][0]["targets"][0]["torch"][
        "schedule_complete"
    ] = False
    report = validate_stop_response(response)
    assert not report.valid
    assert any("scheduled steps" in error for error in report.errors)


def test_event_integrity_rejects_wrong_run_id(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    _write_events(events, run_id="other")
    report = validate_event_file(str(events), run_id="expected")
    assert not report.valid
    assert any("expected" in error for error in report.errors)


def test_event_integrity_can_reject_cache_hit_workloads(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    _write_events(events)
    payload = json.loads(events.read_text(encoding="utf-8"))
    payload["event_name"] = "pre_lm_cache_hit"
    events.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = validate_event_file(
        str(events),
        run_id="run",
        forbid_event_names={"pre_lm_cache_hit"},
    )
    assert not report.valid
    assert any("forbidden" in error for error in report.errors)


def test_request_lifecycle_accepts_complete_cpu_bottleneck_chain(
    tmp_path: Path,
) -> None:
    names = [
        "http_request_received",
        "scheduler_inbox_receive",
        "request_builder_submitted",
        "scheduler_request_build_start",
        "request_build.pre_lm_wait_start",
        "pre_lm_enqueue",
        "pre_lm_dequeue",
        "pre_lm_batch_start",
        "pre_lm_encode_start",
        "pre_lm_encode_submitted",
        "pre_lm_gpu_wait_start",
        "pre_lm_gpu_complete",
        "pre_lm_split_start",
        "pre_lm_split_end",
        "pre_lm_future_publish",
        "pre_lm_batch_end",
        "pre_lm_waiter_resumed",
        "request_build.pre_lm_wait_end",
        "scheduler_request_build_end",
        "request_builder_future_ready",
        "request_build_capacity_release",
        "request_builder_ready_drained",
        "scheduler_queue_enter",
        "scheduler_prefill_start",
        "scheduler_request_terminal",
        "http_response_ready",
    ]
    events = []
    for sequence, name in enumerate(names, 1):
        metadata = {}
        if name.startswith("pre_lm_"):
            metadata = {"batch_id": "b1", "batch_size": 1}
        events.append(
            {
                "request_id": "r1",
                "stage": "asr",
                "event_name": name,
                "timestamp_ns": sequence,
                "monotonic_ns": sequence,
                "source_sequence": sequence,
                "run_id": "run",
                "pid": 1,
                "native_tid": 2,
                "metadata": metadata,
            }
        )
    path = tmp_path / "events_asr_1.jsonl"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    report = validate_request_lifecycle(
        [str(path)],
        expected_request_ids={"r1"},
    )
    assert report.valid, report.errors


def test_request_lifecycle_rejects_ready_future_never_drained(
    tmp_path: Path,
) -> None:
    events = [
        {
            "request_id": "r1",
            "stage": "asr",
            "event_name": name,
            "timestamp_ns": index,
            "monotonic_ns": index,
            "run_id": "run",
            "pid": 1,
            "native_tid": 2,
            "metadata": {},
        }
        for index, name in enumerate(
            (
                "request_builder_submitted",
                "scheduler_request_build_start",
                "scheduler_request_build_end",
                "request_builder_future_ready",
            ),
            1,
        )
    ]
    path = tmp_path / "events_asr_1.jsonl"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    report = validate_request_lifecycle([str(path)])
    assert not report.valid
    assert any("never drained" in error for error in report.errors)
