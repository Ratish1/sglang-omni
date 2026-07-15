#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate same-GPU DP layouts and summarize canonical SeedTTS results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

CPU_ITEM_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
KV_PATTERNS = (
    re.compile(r"max_total_num_tokens(?:=|: )\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:KV cache|KV Cache).*?(\d+)\s+tokens", re.IGNORECASE),
    re.compile(r"KV Cache is allocated\.\s*#tokens:\s*([\d,]+)", re.IGNORECASE),
)


def parse_cpu_set(value: str) -> set[int]:
    """Expand Linux CPU-list syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    if not value.strip():
        raise ValueError("CPU set must not be empty")
    for raw_item in value.split(","):
        item = raw_item.strip()
        match = CPU_ITEM_RE.fullmatch(item)
        if match is None:
            raise ValueError(f"invalid CPU-list item {item!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise ValueError(f"CPU range ends before it starts: {item!r}")
        cpus.update(range(start, end + 1))
    return cpus


def split_core_sets(value: str) -> list[str]:
    sets = [item.strip() for item in value.split(";")]
    if not sets or any(not item for item in sets):
        raise ValueError("core sets must be non-empty and separated with ';'")
    return sets


def validate_layout(
    dp: int,
    server_core_sets: Iterable[str],
    client_core_sets: Iterable[str],
    online_cpus: set[int] | None = None,
    allowed_cpus: set[int] | None = None,
    extra_core_sets: Iterable[tuple[str, str]] = (),
) -> dict[str, object]:
    """Validate dedicated, non-overlapping server/client CPU assignments."""
    if dp not in range(1, 11):
        raise ValueError("DP must be between 1 and 10")
    server_raw = list(server_core_sets)
    client_raw = list(client_core_sets)
    if len(server_raw) != dp:
        raise ValueError(f"expected {dp} server core sets, got {len(server_raw)}")
    if len(client_raw) != dp:
        raise ValueError(f"expected {dp} client core sets, got {len(client_raw)}")

    servers = [parse_cpu_set(value) for value in server_raw]
    clients = [parse_cpu_set(value) for value in client_raw]
    extras = [(label, parse_cpu_set(value)) for label, value in extra_core_sets]
    labels = (
        [f"server[{i}]" for i in range(dp)]
        + [f"client[{i}]" for i in range(dp)]
        + [label for label, _ in extras]
    )
    groups = servers + clients + [cpus for _, cpus in extras]
    for i, group in enumerate(groups):
        if online_cpus is not None:
            missing = sorted(group - online_cpus)
            if missing:
                raise ValueError(f"{labels[i]} contains offline CPUs: {missing}")
        if allowed_cpus is not None:
            outside = sorted(group - allowed_cpus)
            if outside:
                raise ValueError(
                    f"{labels[i]} contains CPUs outside the selected NUMA node: "
                    f"{outside}"
                )
        for j in range(i):
            overlap = sorted(group & groups[j])
            if overlap:
                raise ValueError(f"{labels[i]} overlaps {labels[j]} on CPUs {overlap}")

    return {
        "dp": dp,
        "server_core_sets": server_raw,
        "client_core_sets": client_raw,
        "server_cpu_count": sum(map(len, servers)),
        "client_cpu_count": sum(map(len, clients)),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def extract_kv_tokens(text: str) -> int | None:
    values: list[tuple[int, int]] = []
    for pattern in KV_PATTERNS:
        values.extend(
            (match.start(), int(match.group(1).replace(",", "")))
            for match in pattern.finditer(text)
        )
    return max(values)[1] if values else None


def classify_kv_capacity(
    token_counts: dict[str, int | None], expected_tokens: int | None = None
) -> str:
    known = [value for value in token_counts.values() if value is not None]
    if len(known) != len(token_counts):
        return "missing"
    if len(set(known)) != 1:
        return "unequal"
    if expected_tokens is not None and known[0] != expected_tokens:
        return "configured_mismatch"
    return "exact" if expected_tokens is not None else "equal"


def summarize_results(
    result_paths: Iterable[Path], wall_clock_s: float | None = None
) -> dict[str, object]:
    workers: list[dict[str, object]] = []
    successful_rows: list[dict] = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        rows = payload.get("per_request")
        if not isinstance(summary, dict) or not isinstance(rows, list):
            raise ValueError(f"{path} is not a canonical speed_results.json")
        worker = {
            "worker": path.parent.name,
            "path": str(path),
            **summary,
        }
        workers.append(worker)
        successful_rows.extend(row for row in rows if row.get("is_success"))

    latencies = [float(row["latency_s"]) for row in successful_rows]
    rtfs = [float(row["rtf"]) for row in successful_rows if row.get("rtf") is not None]
    audio_durations = [
        float(row["audio_duration_s"])
        for row in successful_rows
        if row.get("audio_duration_s") is not None
    ]
    output_tokens = [
        int(row["completion_tokens"])
        for row in successful_rows
        if row.get("completion_tokens") is not None
    ]
    qps = [float(worker.get("throughput_qps") or 0) for worker in workers]
    completed_requests = sum(
        int(worker.get("completed_requests") or 0) for worker in workers
    )
    throughput_qps = (
        completed_requests / wall_clock_s if wall_clock_s is not None else sum(qps)
    )
    audio_duration_total = sum(audio_durations)
    output_tokens_total = sum(output_tokens)
    aggregate = {
        "workers": len(workers),
        "total_requests": sum(int(w.get("total_requests") or 0) for w in workers),
        "completed_requests": completed_requests,
        "failed_requests": sum(int(w.get("failed_requests") or 0) for w in workers),
        "throughput_qps": round(throughput_qps, 3),
        "measurement_wall_clock_s": (
            round(wall_clock_s, 6) if wall_clock_s is not None else None
        ),
        "worker_qps_sum": round(sum(qps), 3),
        "latency_p50_s": _rounded(percentile(latencies, 0.50), 4),
        "latency_p95_s": _rounded(percentile(latencies, 0.95), 4),
        "latency_p99_s": _rounded(percentile(latencies, 0.99), 4),
        "rtf_p50": _rounded(percentile(rtfs, 0.50), 4),
        "rtf_p95": _rounded(percentile(rtfs, 0.95), 4),
        "rtf_p99": _rounded(percentile(rtfs, 0.99), 4),
        "audio_duration_mean_s": _rounded(
            statistics.fmean(audio_durations) if audio_durations else None, 4
        ),
        "audio_duration_total_s": round(audio_duration_total, 3),
        "audio_throughput_s_per_s": round(
            (
                audio_duration_total / wall_clock_s
                if wall_clock_s is not None
                else sum(float(w.get("audio_throughput_s_per_s") or 0) for w in workers)
            ),
            3,
        ),
        "output_throughput_tok_s": round(
            (
                output_tokens_total / wall_clock_s
                if wall_clock_s is not None
                else sum(float(w.get("output_throughput") or 0) for w in workers)
            ),
            1,
        ),
        "output_tokens_total": output_tokens_total,
        "output_tokens_mean": _rounded(
            statistics.fmean(output_tokens) if output_tokens else None, 1
        ),
        "worker_qps_min": round(min(qps), 3) if qps else None,
        "worker_qps_max": round(max(qps), 3) if qps else None,
        "worker_qps_cv": (
            round(statistics.pstdev(qps) / statistics.fmean(qps), 4)
            if len(qps) > 1 and statistics.fmean(qps) > 0
            else 0.0
        ),
    }
    return {"aggregate": aggregate, "per_worker": workers}


# Two-sided Student-t 97.5th percentiles for df=1..30. For larger samples the
# normal approximation is adequate for this experimental report.
T_975 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def summarize_matrix(matrix_tsv: Path) -> dict[str, object]:
    """Aggregate successful repeated conditions from ``matrix_results.tsv``."""
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    with matrix_tsv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["status"] != "pass":
                failures.append(row)
                continue
            summary_path = Path(row["output_dir"]) / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            key = (int(row["dp"]), int(row["mps"]), int(row["concurrency"]))
            grouped[key].append(float(payload["aggregate"]["throughput_qps"]))

    conditions: list[dict[str, object]] = []
    for (dp, mps, concurrency), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        if len(values) > 1:
            sem = statistics.stdev(values) / math.sqrt(len(values))
            df = len(values) - 1
            critical = T_975[df - 1] if df <= len(T_975) else 1.96
            half_width = critical * sem
        else:
            half_width = None
        conditions.append(
            {
                "dp": dp,
                "mps": bool(mps),
                "concurrency_per_worker": concurrency,
                "repetitions": len(values),
                "throughput_qps_mean": round(mean, 3),
                "throughput_qps_ci95_low": (
                    round(mean - half_width, 3) if half_width is not None else None
                ),
                "throughput_qps_ci95_high": (
                    round(mean + half_width, 3) if half_width is not None else None
                ),
                "throughput_qps_values": values,
            }
        )
    return {"conditions": conditions, "failed_or_incomplete_runs": failures}


def summarize_router_snapshot(snapshot_path: Path) -> dict[str, object]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    workers = payload.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError(f"{snapshot_path} contains no router workers")
    routed = [int(worker.get("routed_requests") or 0) for worker in workers]
    successful = [int(worker.get("successful_requests") or 0) for worker in workers]
    failed = [int(worker.get("failed_requests") or 0) for worker in workers]
    routed_mean = statistics.fmean(routed)
    return {
        "workers": workers,
        "routed_requests_total": sum(routed),
        "successful_requests_total": sum(successful),
        "failed_requests_total": sum(failed),
        "routed_requests_min": min(routed),
        "routed_requests_max": max(routed),
        "routed_requests_cv": (
            round(statistics.pstdev(routed) / routed_mean, 4)
            if len(routed) > 1 and routed_mean > 0
            else 0.0
        ),
    }


def _rounded(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-layout")
    validate.add_argument("--dp", required=True, type=int)
    validate.add_argument("--server-core-sets", required=True)
    validate.add_argument("--client-core-sets", required=True)
    validate.add_argument("--online-cpus")
    validate.add_argument("--numa-cpus")
    validate.add_argument("--router-cores")

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--output", required=True, type=Path)
    summarize.add_argument("--wall-clock-s", type=float)
    summarize.add_argument("results", nargs="+", type=Path)

    kv = sub.add_parser("extract-kv")
    kv.add_argument("logs", nargs="+", type=Path)
    kv.add_argument("--require-equal", action="store_true")

    classify_kv = sub.add_parser("classify-kv")
    classify_kv.add_argument("capacity_json", type=Path)
    classify_kv.add_argument("--expected", type=int)

    matrix = sub.add_parser("summarize-matrix")
    matrix.add_argument("--output", required=True, type=Path)
    matrix.add_argument("matrix_tsv", type=Path)

    router = sub.add_parser("summarize-router")
    router.add_argument("--output", required=True, type=Path)
    router.add_argument("snapshot", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "validate-layout":
        online = parse_cpu_set(args.online_cpus) if args.online_cpus else None
        numa = parse_cpu_set(args.numa_cpus) if args.numa_cpus else None
        result = validate_layout(
            args.dp,
            split_core_sets(args.server_core_sets),
            split_core_sets(args.client_core_sets),
            online,
            numa,
            (("router", args.router_cores),) if args.router_cores else (),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "summarize":
        if args.wall_clock_s is not None and args.wall_clock_s <= 0:
            raise SystemExit("--wall-clock-s must be positive")
        result = summarize_results(args.results, args.wall_clock_s)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["aggregate"], indent=2))
        return
    if args.command == "summarize-matrix":
        result = summarize_matrix(args.matrix_tsv)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if args.command == "summarize-router":
        result = summarize_router_snapshot(args.snapshot)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if args.command == "classify-kv":
        token_counts = json.loads(args.capacity_json.read_text(encoding="utf-8"))
        state = classify_kv_capacity(token_counts, args.expected)
        print(state)
        return

    token_counts: dict[str, int | None] = {}
    for path in args.logs:
        token_counts[str(path)] = extract_kv_tokens(
            path.read_text(encoding="utf-8", errors="replace")
        )
    print(json.dumps(token_counts, indent=2))
    known = [value for value in token_counts.values() if value is not None]
    if args.require_equal and (len(known) != len(token_counts) or len(set(known)) != 1):
        raise SystemExit(
            "KV capacity is missing or differs across replicas; see kv_capacity.json"
        )


if __name__ == "__main__":
    main()
