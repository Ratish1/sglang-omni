# SPDX-License-Identifier: Apache-2.0
"""Integrity gates for SGLang-Omni profiling artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any


@dataclass
class IntegrityReport:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _open_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("trace root must be an object")
    return data


def validate_torch_trace(
    path: str,
    *,
    require_cuda: bool = True,
    require_canary: bool = True,
    owner_label: str | None = None,
) -> IntegrityReport:
    report = IntegrityReport()
    trace_path = Path(path)
    if not trace_path.is_file():
        report.fail(f"torch trace does not exist: {trace_path}")
        return report
    try:
        trace = _open_json(trace_path)
    except Exception as exc:  # noqa: BLE001 - corrupt traces become gate failures
        report.fail(f"cannot read torch trace {trace_path}: {exc}")
        return report

    events = trace.get("traceEvents")
    if not isinstance(events, list) or not events:
        report.fail(f"torch trace has no traceEvents: {trace_path}")
        return report

    names = {str(event.get("name", "")) for event in events if isinstance(event, dict)}
    categories = {
        str(event.get("cat", "")).lower() for event in events if isinstance(event, dict)
    }
    owner_prefix = (
        f"sglang_omni.profiler.{owner_label}_owner."
        if owner_label
        else "sglang_omni.profiler."
    )
    has_canary = any(
        name.startswith(owner_prefix) and (owner_label is not None or "_owner." in name)
        for name in names
    )
    if require_canary and not has_canary:
        report.fail(
            f"{owner_label or 'profiler'}-owner canary is missing from torch trace"
        )

    has_cuda = any(
        token in category
        for category in categories
        for token in ("cuda", "kernel", "gpu")
    ) or any(
        any(token in name.lower() for token in ("cuda", "kernel")) for name in names
    )
    if require_cuda and not has_cuda:
        report.fail("CUDA activity is missing from a CUDA-required torch trace")
    launch_events = sum(
        1
        for name in names
        if any(
            token in name.lower()
            for token in ("cudalaunch", "cuda launch", "cudagraphlaunch")
        )
    )
    kernel_events = sum(
        1
        for event in events
        if isinstance(event, dict)
        and any(
            token in str(event.get("cat", "")).lower() for token in ("kernel", "gpu")
        )
    )
    if (
        require_cuda
        and owner_label is not None
        and (launch_events == 0 or kernel_events == 0)
    ):
        report.fail(
            "CUDA-required torch trace lacks launch-to-kernel correlation "
            f"(launch_events={launch_events}, kernel_events={kernel_events})"
        )

    pids = {
        event.get("pid")
        for event in events
        if isinstance(event, dict) and event.get("pid") is not None
    }
    tids = {
        event.get("tid")
        for event in events
        if isinstance(event, dict) and event.get("tid") is not None
    }
    if not pids or not tids:
        report.fail("torch trace is missing PID/TID identity")
    report.artifacts.append(
        {
            "type": "torch",
            "path": str(trace_path),
            "events": len(events),
            "pids": len(pids),
            "tids": len(tids),
            "has_cuda": has_cuda,
            "launch_events": launch_events,
            "kernel_events": kernel_events,
            "owner_label": owner_label,
        }
    )
    return report


def validate_event_file(
    path: str,
    *,
    run_id: str | None = None,
    forbid_event_names: set[str] | None = None,
    expected_sha256: str | None = None,
    require_nonempty: bool = True,
) -> IntegrityReport:
    report = IntegrityReport()
    event_path = Path(path)
    if not event_path.is_file():
        report.fail(f"event file does not exist: {event_path}")
        return report

    count = 0
    request_ids: set[str] = set()
    threads: set[tuple[int | None, int | None]] = set()
    event_names: dict[str, int] = {}
    source_last: dict[tuple[Any, Any], tuple[int, int]] = {}
    try:
        with event_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                count += 1
                for field_name in (
                    "request_id",
                    "stage",
                    "event_name",
                    "timestamp_ns",
                    "monotonic_ns",
                    "pid",
                    "native_tid",
                ):
                    if event.get(field_name) is None:
                        report.fail(f"{event_path}:{line_number} missing {field_name}")
                if run_id is not None and event.get("run_id") != run_id:
                    report.fail(
                        f"{event_path}:{line_number} run_id="
                        f"{event.get('run_id')!r}, expected {run_id!r}"
                    )
                request_ids.add(str(event.get("request_id")))
                source = (event.get("pid"), event.get("native_tid"))
                threads.add(source)
                sequence = event.get("source_sequence")
                monotonic_ns = event.get("monotonic_ns")
                if sequence is not None and monotonic_ns is not None:
                    previous = source_last.get(source)
                    if previous is not None:
                        if int(sequence) != previous[0] + 1:
                            report.fail(
                                f"{event_path}:{line_number} source sequence "
                                f"{sequence} does not follow {previous[0]}"
                            )
                        if int(monotonic_ns) < previous[1]:
                            report.fail(
                                f"{event_path}:{line_number} monotonic clock "
                                "moved backwards for one event source"
                            )
                    source_last[source] = (int(sequence), int(monotonic_ns))
                event_name = str(event.get("event_name"))
                event_names[event_name] = event_names.get(event_name, 0) + 1
    except Exception as exc:  # noqa: BLE001 - corrupt events become gate failures
        report.fail(f"cannot read event file {event_path}: {exc}")

    if require_nonempty and count == 0:
        report.fail(f"event file is empty: {event_path}")
    digest = (
        hashlib.sha256(event_path.read_bytes()).hexdigest()
        if event_path.is_file()
        else None
    )
    if expected_sha256 is not None and digest != expected_sha256:
        report.fail(
            f"event file checksum {digest!r} does not match {expected_sha256!r}"
        )
    for event_name in sorted(forbid_event_names or set()):
        if event_names.get(event_name, 0):
            report.fail(
                f"event file {event_path} contains "
                f"{event_names[event_name]} forbidden {event_name!r} events"
            )
    report.artifacts.append(
        {
            "type": "events",
            "path": str(event_path),
            "events": count,
            "requests": len(request_ids),
            "threads": len(threads),
            "event_names": event_names,
            "sha256": digest,
        }
    )
    return report


def _load_event_paths(paths: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
    events.sort(
        key=lambda event: (
            str(event.get("host_boot_id") or ""),
            int(event.get("monotonic_ns") or 0),
            int(event.get("pid") or 0),
            int(event.get("native_tid") or 0),
            int(event.get("source_sequence") or 0),
        )
    )
    return events


def validate_request_lifecycle(
    paths: list[str],
    *,
    expected_request_ids: set[str] | None = None,
) -> IntegrityReport:
    """Validate the request/build/pre-LM state machine across process files."""
    report = IntegrityReport()
    try:
        events = _load_event_paths(paths)
    except Exception as exc:  # noqa: BLE001 - reconstruction is an integrity gate
        report.fail(f"cannot reconstruct event lifecycle: {exc}")
        return report
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        request_id = str(event.get("request_id") or "")
        if request_id and request_id != "__pre_lm__":
            by_request[request_id].append(event)

    if expected_request_ids is not None:
        missing = sorted(expected_request_ids - set(by_request))
        extra = sorted(set(by_request) - expected_request_ids)
        if missing:
            report.fail(f"event capture is missing {len(missing)} expected requests")
        if extra:
            report.warnings.append(
                f"event capture contains {len(extra)} non-measurement requests"
            )

    balanced_pairs = (
        ("scheduler_request_build_start", "scheduler_request_build_end"),
        ("request_build.audio_load_start", "request_build.audio_load_end"),
        ("request_build.feature_extract_start", "request_build.feature_extract_end"),
        (
            "request_build.tokenize_and_pack_start",
            "request_build.tokenize_and_pack_end",
        ),
        ("request_build.pre_lm_wait_start", "request_build.pre_lm_wait_end"),
        ("scheduler_request_build_hol_start", "scheduler_request_build_hol_end"),
        ("pre_lm_encode_start", "pre_lm_encode_submitted"),
        ("pre_lm_split_start", "pre_lm_split_end"),
        ("pre_lm_gpu_wait_start", "pre_lm_gpu_complete"),
    )
    batch_members: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    batch_sizes: dict[str, set[int]] = defaultdict(set)
    for request_id, request_events in by_request.items():
        names = [str(event.get("event_name")) for event in request_events]
        counts = Counter(names)
        for opener, closer in balanced_pairs:
            if counts[opener] != counts[closer]:
                report.fail(
                    f"request {request_id} has unbalanced {opener}/{closer}: "
                    f"{counts[opener]}/{counts[closer]}"
                )

        if counts["http_request_received"] and counts["http_response_ready"] != 1:
            report.fail(
                f"request {request_id} has {counts['http_response_ready']} "
                "HTTP terminal events"
            )
        if (
            counts["scheduler_inbox_receive"]
            and counts["scheduler_request_terminal"] != 1
        ):
            report.fail(
                f"request {request_id} has "
                f"{counts['scheduler_request_terminal']} scheduler terminal events"
            )
        if counts["request_builder_submitted"]:
            if (
                counts["request_builder_future_ready"]
                != counts["request_builder_submitted"]
            ):
                report.fail(
                    f"request {request_id} builder submit/future-ready mismatch"
                )
            if (
                counts["request_build_capacity_release"]
                != counts["request_builder_submitted"]
            ):
                report.fail(
                    f"request {request_id} builder capacity acquire/release mismatch: "
                    f"{counts['request_builder_submitted']}/"
                    f"{counts['request_build_capacity_release']}"
                )
            if (
                counts["request_builder_ready_drained"] == 0
                and counts["scheduler_request_terminal"] == 0
            ):
                report.fail(
                    f"request {request_id} builder future was never drained or failed"
                )
        if counts["pre_lm_enqueue"]:
            for required in (
                "pre_lm_dequeue",
                "pre_lm_future_publish",
                "pre_lm_waiter_resumed",
            ):
                if counts[required] == 0:
                    report.fail(
                        f"request {request_id} enqueued pre-LM work without {required}"
                    )

        positions: dict[str, int] = {}
        for index, name in enumerate(names):
            positions.setdefault(name, index)
        ordered_chain = (
            "scheduler_inbox_receive",
            "request_builder_submitted",
            "scheduler_request_build_start",
            "pre_lm_enqueue",
            "pre_lm_dequeue",
            "pre_lm_future_publish",
            "pre_lm_waiter_resumed",
            "scheduler_request_build_end",
            "request_builder_future_ready",
            "request_builder_ready_drained",
            "scheduler_queue_enter",
            "scheduler_prefill_start",
            "scheduler_request_terminal",
        )
        present_positions = [
            (name, positions[name]) for name in ordered_chain if name in positions
        ]
        for (left_name, left), (right_name, right) in pairwise(present_positions):
            if left > right:
                report.fail(
                    f"request {request_id} lifecycle reordered: "
                    f"{left_name} after {right_name}"
                )

        for event in request_events:
            metadata = event.get("metadata") or {}
            batch_id = metadata.get("batch_id")
            if not batch_id:
                continue
            name = str(event.get("event_name"))
            batch_members[str(batch_id)][name].add(request_id)
            batch_size = metadata.get("batch_size")
            if isinstance(batch_size, int):
                batch_sizes[str(batch_id)].add(batch_size)

    for batch_id, by_name in batch_members.items():
        if len(batch_sizes[batch_id]) > 1:
            report.fail(f"pre-LM batch {batch_id} reports conflicting batch sizes")
        starts = by_name.get("pre_lm_batch_start", set())
        ends = by_name.get("pre_lm_batch_end", set())
        if starts != ends:
            report.fail(
                f"pre-LM batch {batch_id} start/end membership differs "
                f"({len(starts)} vs {len(ends)})"
            )
        if batch_sizes[batch_id]:
            expected_size = next(iter(batch_sizes[batch_id]))
            if starts and len(starts) != expected_size:
                report.fail(
                    f"pre-LM batch {batch_id} membership {len(starts)} "
                    f"does not match batch_size={expected_size}"
                )

    report.artifacts.append(
        {
            "type": "request_lifecycle",
            "paths": paths,
            "requests": len(by_request),
            "events": len(events),
            "batches": len(batch_members),
        }
    )
    return report


def validate_stop_response(
    response: dict[str, Any],
    *,
    require_cuda: bool = True,
    require_events: bool = True,
    require_schedule_complete: bool = True,
    forbid_event_names: set[str] | None = None,
    require_nonempty_events: bool = True,
) -> IntegrityReport:
    """Validate the acknowledged `/stop_profile` response and its files."""

    report = IntegrityReport()
    manifest = response.get("manifest")
    if not isinstance(manifest, dict):
        report.fail("stop response has no profiler manifest")
        return report
    if not manifest.get("success"):
        report.fail("profiler stop manifest reports failure")
    if manifest.get("missing_stages"):
        report.fail(f"missing stage acknowledgements: {manifest['missing_stages']}")

    run_id = response.get("run_id") or manifest.get("run_id")
    event_paths: list[str] = []
    lifecycle_schema = False
    for stage_result in manifest.get("stages", []):
        stage = stage_result.get("stage", "unknown")
        if not stage_result.get("success"):
            report.fail(f"stage {stage} profiler result failed")
        targets = stage_result.get("targets") or []
        if not targets:
            report.fail(f"stage {stage} has no target results")
        for target in targets:
            label = f"{stage}/rank{target.get('rank')}/pid{target.get('pid')}"
            if not target.get("success"):
                report.fail(f"target {label} failed: {target.get('error')}")
                continue
            torch_state = target.get("torch")
            if isinstance(torch_state, dict):
                if torch_state.get("export_error"):
                    report.fail(
                        f"target {label} export failed: {torch_state['export_error']}"
                    )
                if not torch_state.get("trace_finalized"):
                    report.fail(f"target {label} trace was not finalized")
                if require_schedule_complete and not torch_state.get(
                    "schedule_complete"
                ):
                    report.fail(
                        f"target {label} stopped after "
                        f"{torch_state.get('step_count')} of "
                        f"{torch_state.get('expected_steps')} scheduled steps"
                    )
                trace_path = torch_state.get("trace") or target.get("trace")
                if trace_path:
                    _merge(
                        report,
                        validate_torch_trace(
                            str(trace_path),
                            require_cuda=require_cuda,
                            owner_label=torch_state.get("owner_label"),
                        ),
                    )
            events = target.get("events")
            if require_events:
                if not isinstance(events, dict) or not events.get("path"):
                    report.fail(f"target {label} has no event artifact")
                else:
                    event_paths.append(str(events["path"]))
                    lifecycle_schema = (
                        lifecycle_schema or int(events.get("schema_version") or 0) >= 2
                    )
                    if events.get("dropped_events", 0):
                        report.fail(
                            f"target {label} dropped {events['dropped_events']} events"
                        )
                    if "finalized" in events and not events.get("finalized"):
                        report.fail(f"target {label} event artifact was not finalized")
                    if events.get("writer_error"):
                        report.fail(
                            f"target {label} event writer failed: "
                            f"{events['writer_error']}"
                        )
                    if (
                        events.get("enqueued_events") is not None
                        and events.get("written_events") is not None
                        and events["enqueued_events"] != events["written_events"]
                    ):
                        report.fail(
                            f"target {label} enqueued/written event mismatch: "
                            f"{events['enqueued_events']}/"
                            f"{events['written_events']}"
                        )
                    _merge(
                        report,
                        validate_event_file(
                            str(events["path"]),
                            run_id=str(run_id) if run_id is not None else None,
                            forbid_event_names=forbid_event_names,
                            expected_sha256=events.get("sha256"),
                            require_nonempty=require_nonempty_events,
                        ),
                    )

    coordinator_path = response.get("coordinator_event_path")
    if require_events and coordinator_path:
        event_paths.append(str(coordinator_path))
        coordinator_events = response.get("coordinator_events") or {}
        lifecycle_schema = (
            lifecycle_schema or int(coordinator_events.get("schema_version") or 0) >= 2
        )
        if coordinator_events and not coordinator_events.get("finalized"):
            report.fail("coordinator event artifact was not finalized")
        if coordinator_events.get("writer_error"):
            report.fail(
                f"coordinator event writer failed: {coordinator_events['writer_error']}"
            )
        if coordinator_events.get("dropped_events"):
            report.fail(
                f"coordinator dropped {coordinator_events['dropped_events']} events"
            )
        _merge(
            report,
            validate_event_file(
                str(coordinator_path),
                run_id=str(run_id) if run_id is not None else None,
                forbid_event_names=forbid_event_names,
                expected_sha256=coordinator_events.get("sha256"),
                require_nonempty=require_nonempty_events,
            ),
        )
    if require_events and event_paths and lifecycle_schema:
        _merge(report, validate_request_lifecycle(event_paths))
    return report


def _merge(target: IntegrityReport, source: IntegrityReport) -> None:
    target.valid = target.valid and source.valid
    target.errors.extend(source.errors)
    target.warnings.extend(source.warnings)
    target.artifacts.extend(source.artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stop_response", help="JSON response from /stop_profile")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--allow-partial-schedule", action="store_true")
    args = parser.parse_args()
    with Path(args.stop_response).open("r", encoding="utf-8") as handle:
        response = json.load(handle)
    report = validate_stop_response(
        response,
        require_cuda=not args.no_cuda,
        require_events=not args.no_events,
        require_schedule_complete=not args.allow_partial_schedule,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if not report.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
