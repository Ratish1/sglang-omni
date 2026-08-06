# SPDX-License-Identifier: Apache-2.0
"""Report worker distribution for one TTS mixed-serving CI run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIXED_STAGE = "mixed-production"
ROUTING_POLICY = "least_request"
EXPECTED_WORKER_COUNT = 2
WORKER_HEADER = "x-sglang-omni-worker"
WS_COMPLETION_RE = re.compile(
    r"tts_websocket_completed request_id=(?P<request_id>\S+) " r"worker=(?P<worker>\S+)"
)


class ObservationError(RuntimeError):
    """Raised when run artifacts cannot support exact worker attribution."""


@dataclass(frozen=True)
class RoutedSample:
    workload: str
    worker_id: str
    latency_s: float
    rtf: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="TTS serving CI output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output stem; defaults to RUN_DIR/distribution_observation and "
            "writes both .json and .md"
        ),
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ObservationError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ObservationError(f"could not read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ObservationError(
                f"invalid JSON in {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ObservationError(f"expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _worker_topology(
    validation: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    workers = validation.get("workers_after", {}).get("workers")
    if not isinstance(workers, list) or len(workers) != EXPECTED_WORKER_COUNT:
        observed = len(workers) if isinstance(workers, list) else 0
        raise ObservationError(
            f"expected {EXPECTED_WORKER_COUNT} workers, observed {observed}"
        )

    worker_ids: list[str] = []
    aliases: dict[str, str] = {}
    for worker in workers:
        if not isinstance(worker, dict):
            raise ObservationError("router validation contains an invalid worker")
        worker_id = worker.get("worker_id")
        display_id = worker.get("display_id")
        if not isinstance(worker_id, str) or not isinstance(display_id, str):
            raise ObservationError("router worker identity is incomplete")
        worker_ids.append(worker_id)
        aliases[worker_id] = worker_id
        aliases[display_id] = worker_id
    return worker_ids, aliases


def _configured_samples(manifest: dict[str, Any]) -> dict[str, int]:
    stages = manifest.get("load_stages")
    if not isinstance(stages, list):
        raise ObservationError("benchmark manifest has no load stages")
    stage = next(
        (
            value
            for value in stages
            if isinstance(value, dict) and value.get("id") == MIXED_STAGE
        ),
        None,
    )
    if stage is None:
        raise ObservationError(f"benchmark manifest has no {MIXED_STAGE!r} stage")
    schedules = stage.get("workload_schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ObservationError(f"{MIXED_STAGE!r} has no workload schedules")

    configured: dict[str, int] = {}
    for schedule in schedules:
        if not isinstance(schedule, dict):
            raise ObservationError("invalid workload schedule")
        workload = schedule.get("workload")
        background = schedule.get("background_offsets_s")
        collisions = schedule.get("collision_offsets_s")
        if (
            not isinstance(workload, str)
            or not isinstance(background, list)
            or not isinstance(collisions, list)
        ):
            raise ObservationError(f"incomplete workload schedule: {schedule}")
        if workload in configured:
            raise ObservationError(f"duplicate workload schedule for {workload!r}")
        configured[workload] = len(background) + len(collisions)
    return configured


def _websocket_workers(
    router_log: Path,
    aliases: dict[str, str],
) -> dict[str, str]:
    try:
        lines = router_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ObservationError(f"could not read {router_log}: {exc}") from exc
    routed: dict[str, str] = {}
    for line in lines:
        match = WS_COMPLETION_RE.search(line)
        if match is None:
            continue
        request_id = match.group("request_id")
        worker_id = aliases.get(match.group("worker"))
        if worker_id is None:
            raise ObservationError("WebSocket log names an unknown worker")
        previous = routed.setdefault(request_id, worker_id)
        if previous != worker_id:
            raise ObservationError(
                f"WebSocket request {request_id!r} completed on multiple workers"
            )
    return routed


def _response_worker(row: dict[str, Any]) -> str | None:
    headers = row.get("response_headers")
    if not isinstance(headers, dict):
        return None
    return next(
        (
            value
            for key, value in headers.items()
            if str(key).lower() == WORKER_HEADER and isinstance(value, str)
        ),
        None,
    )


def _mixed_samples(
    rows: list[dict[str, Any]],
    configured: dict[str, int],
    worker_ids: set[str],
    websocket_workers: dict[str, str],
) -> list[RoutedSample]:
    samples: list[RoutedSample] = []
    for row in rows:
        workload = row.get("workload")
        if row.get("stage_id") != MIXED_STAGE or workload is None:
            continue
        scenario_id = row.get("scenario_id")
        endpoint = row.get("endpoint")
        if (
            not isinstance(scenario_id, str)
            or not isinstance(workload, str)
            or not isinstance(endpoint, str)
        ):
            raise ObservationError("mixed result has incomplete identity")
        if workload not in configured:
            raise ObservationError(f"unconfigured workload {workload!r}")
        if row.get("expected_success") is not True or row.get("success") is not True:
            raise ObservationError(f"mixed result {scenario_id!r} did not pass")

        worker_id = (
            websocket_workers.get(scenario_id)
            if endpoint == "websocket"
            else _response_worker(row)
        )
        if worker_id not in worker_ids:
            raise ObservationError(
                f"mixed result {scenario_id!r} has no valid worker attribution"
            )
        latency_s = row.get("latency_s")
        rtf = row.get("rtf")
        if not isinstance(latency_s, (int, float)) or not isinstance(rtf, (int, float)):
            raise ObservationError(f"mixed result {scenario_id!r} has invalid metrics")
        samples.append(
            RoutedSample(
                workload=workload,
                worker_id=worker_id,
                latency_s=float(latency_s),
                rtf=float(rtf),
            )
        )

    observed = Counter(sample.workload for sample in samples)
    if dict(observed) != configured:
        raise ObservationError(
            f"mixed population mismatch: configured={configured}, "
            f"observed={dict(observed)}"
        )
    return samples


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * pct), len(ordered) - 1)]


def _worker_summary(
    samples: list[RoutedSample],
    worker_ids: list[str],
) -> dict[str, Any]:
    total_count = len(samples)
    total_time_s = sum(sample.latency_s for sample in samples)
    summary: dict[str, Any] = {}
    for worker_id in worker_ids:
        routed = [sample for sample in samples if sample.worker_id == worker_id]
        latencies = [sample.latency_s for sample in routed]
        rtfs = [sample.rtf for sample in routed if sample.rtf > 0]
        request_time_s = sum(latencies)
        summary[worker_id] = {
            "samples": len(routed),
            "sample_share": len(routed) / total_count if total_count else None,
            "client_request_time_s": request_time_s,
            "client_request_time_share": (
                request_time_s / total_time_s if total_time_s > 0 else None
            ),
            "latency_s": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "rtf_p95": _percentile(rtfs, 0.95),
        }
    return summary


def _router_counters(
    validation: dict[str, Any],
    worker_ids: list[str],
) -> dict[str, Any]:
    deltas = validation.get("worker_deltas")
    if not isinstance(deltas, list):
        raise ObservationError("router validation has no worker deltas")
    by_id = {
        value.get("worker_id"): value
        for value in deltas
        if isinstance(value, dict) and isinstance(value.get("worker_id"), str)
    }
    if set(by_id) != set(worker_ids):
        raise ObservationError("router counter workers do not match the topology")
    return {worker_id: by_id[worker_id] for worker_id in worker_ids}


def build_observation(run_dir: Path) -> dict[str, Any]:
    benchmark_dir = run_dir / "benchmark"
    manifest_path = benchmark_dir / "manifest.json"
    events_path = benchmark_dir / "raw" / "events.jsonl"
    validation_path = run_dir / "router_validation.json"
    router_log_path = run_dir / "topology" / "router.log"

    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    worker_ids, aliases = _worker_topology(validation)
    configured = _configured_samples(manifest)
    samples = _mixed_samples(
        _read_jsonl(events_path),
        configured,
        set(worker_ids),
        _websocket_workers(router_log_path, aliases),
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing_policy": ROUTING_POLICY,
        "stage_id": MIXED_STAGE,
        "population": (
            "successful workload-bearing mixed-production requests; "
            "coverage and owner-affine voice stages excluded"
        ),
        "source": {
            "workload_spec_hash": manifest.get("workload_spec_hash"),
            "scenario_set_hash": manifest.get("scenario_set_hash"),
        },
        "configured_samples_by_workload": configured,
        "total_samples": len(samples),
        "workers": _worker_summary(samples, worker_ids),
        "by_workload": {
            workload: _worker_summary(
                [sample for sample in samples if sample.workload == workload],
                worker_ids,
            )
            for workload in configured
        },
        "router_counters_context": {
            "description": (
                "All benchmark router operations. voice_control is owner-affine "
                "and is not part of the mixed performance population."
            ),
            "workers": _router_counters(validation, worker_ids),
        },
    }


def _percent(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.1%}"


def _number(value: Any, digits: int = 3) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TTS Serving Distribution Observation",
        "",
        f"- Routing policy: `{report['routing_policy']}`",
        f"- Stage: `{report['stage_id']}`",
        f"- Measured samples: {report['total_samples']}",
        f"- Population: {report['population']}",
        "",
        "## Mixed performance population",
        "",
        "| Worker | Samples | Sample share | Client request-time (s) | Time share |",
        "|---|---:|---:|---:|---:|",
    ]
    for worker_id, worker in report["workers"].items():
        lines.append(
            f"| `{worker_id}` | {worker['samples']} | "
            f"{_percent(worker['sample_share'])} | "
            f"{_number(worker['client_request_time_s'])} | "
            f"{_percent(worker['client_request_time_share'])} |"
        )

    for workload, workers in report["by_workload"].items():
        lines.extend(
            [
                "",
                f"## `{workload}`",
                "",
                "| Worker | Samples | Share | Latency p50 (s) | "
                "Latency p95 (s) | RTF p95 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for worker_id, worker in workers.items():
            lines.append(
                f"| `{worker_id}` | {worker['samples']} | "
                f"{_percent(worker['sample_share'])} | "
                f"{_number(worker['latency_s']['p50'])} | "
                f"{_number(worker['latency_s']['p95'])} | "
                f"{_number(worker['rtf_p95'], 4)} |"
            )

    lines.extend(
        [
            "",
            "## Router counters (context only)",
            "",
            "These counters include coverage and owner-affine voice operations; "
            "they are not the mixed performance population.",
            "",
            "| Worker | All routed | Speech HTTP | Speech batch | WebSocket | "
            "Voice control |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for worker_id, worker in report["router_counters_context"]["workers"].items():
        classes = worker.get("classes", {})
        lines.append(
            f"| `{worker_id}` | {worker.get('routed', 0)} | "
            f"{classes.get('speech_http', 0)} | "
            f"{classes.get('speech_batch', 0)} | "
            f"{classes.get('tts_websocket', 0)} | "
            f"{classes.get('voice_control', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "distribution_observation"
    )
    try:
        report = build_observation(run_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        json_path = output.with_suffix(".json")
        markdown_path = output.with_suffix(".md")
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    except (ObservationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
