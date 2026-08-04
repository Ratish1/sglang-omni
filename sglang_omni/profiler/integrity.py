# SPDX-License-Identifier: Apache-2.0
"""Integrity gates for SGLang-Omni profiling artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass, field
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
) -> IntegrityReport:
    report = IntegrityReport()
    trace_path = Path(path)
    if not trace_path.is_file():
        report.fail(f"torch trace does not exist: {trace_path}")
        return report
    try:
        trace = _open_json(trace_path)
    except Exception as exc:
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
    if require_canary and not any(
        name.startswith("sglang_omni.profiler.scheduler_owner.") for name in names
    ):
        report.fail("scheduler-owner canary is missing from torch trace")

    has_cuda = any(
        token in category
        for category in categories
        for token in ("cuda", "kernel", "gpu")
    ) or any(
        any(token in name.lower() for token in ("cuda", "kernel")) for name in names
    )
    if require_cuda and not has_cuda:
        report.fail("CUDA activity is missing from a CUDA-required torch trace")

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
        }
    )
    return report


def validate_event_file(
    path: str,
    *,
    run_id: str | None = None,
    forbid_event_names: set[str] | None = None,
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
                threads.add((event.get("pid"), event.get("native_tid")))
                event_name = str(event.get("event_name"))
                event_names[event_name] = event_names.get(event_name, 0) + 1
    except Exception as exc:
        report.fail(f"cannot read event file {event_path}: {exc}")

    if count == 0:
        report.fail(f"event file is empty: {event_path}")
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
                        f"target {label} export failed: "
                        f"{torch_state['export_error']}"
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
                        ),
                    )
            events = target.get("events")
            if require_events:
                if not isinstance(events, dict) or not events.get("path"):
                    report.fail(f"target {label} has no event artifact")
                else:
                    if events.get("dropped_events", 0):
                        report.fail(
                            f"target {label} dropped "
                            f"{events['dropped_events']} events"
                        )
                    _merge(
                        report,
                        validate_event_file(
                            str(events["path"]),
                            run_id=str(run_id) if run_id is not None else None,
                            forbid_event_names=forbid_event_names,
                        ),
                    )

    coordinator_path = response.get("coordinator_event_path")
    if require_events and coordinator_path:
        _merge(
            report,
            validate_event_file(
                str(coordinator_path),
                run_id=str(run_id) if run_id is not None else None,
                forbid_event_names=forbid_event_names,
            ),
        )
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
