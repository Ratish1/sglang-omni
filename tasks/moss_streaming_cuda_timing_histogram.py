#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize MOSS streaming vocoder CUDA timing profile events.

Input can be either:
- a request-profiler events directory containing ``events_*.jsonl``
- a request-profiler JSON report containing ``timelines``

The script extracts ``moss_streaming_cuda_timing`` events and groups decode
timing by active participant count and by streaming step length ``T``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl_events(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.glob("events_*.jsonl")) if path.is_dir() else [path]
    for item in paths:
        if not item.is_file():
            continue
        with item.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _iter_profile_json_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    timelines = data.get("timelines")
    if not isinstance(timelines, dict):
        return
    for events in timelines.values():
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    yield event


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir() or path.suffix == ".jsonl":
        yield from _iter_jsonl_events(path)
        return
    yield from _iter_profile_json_events(path)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def row_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def round_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        rounded[key] = round(value, 4) if isinstance(value, float) else value
    return rounded


def summarize(path: Path) -> dict[str, Any]:
    by_participants: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_step_t: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_decode: list[float] = []
    all_d2h_gpu: list[float] = []
    all_d2h_host: list[float] = []

    for event in iter_events(path):
        if event.get("event_name") != "moss_streaming_cuda_timing":
            continue
        metadata = event.get("metadata") or {}
        participants = int(metadata["participants"])
        step_t = int(metadata["step_t"])
        decode_gpu_ms = float(metadata["decode_gpu_ms"])
        d2h_gpu_ms = float(metadata["d2h_gpu_ms"])
        d2h_host_ms = float(metadata["d2h_host_ms"])

        all_decode.append(decode_gpu_ms)
        all_d2h_gpu.append(d2h_gpu_ms)
        all_d2h_host.append(d2h_host_ms)

        by_participants[participants]["decode_gpu_ms"].append(decode_gpu_ms)
        by_participants[participants]["d2h_gpu_ms"].append(d2h_gpu_ms)
        by_participants[participants]["d2h_host_ms"].append(d2h_host_ms)
        by_step_t[step_t]["decode_gpu_ms"].append(decode_gpu_ms)
        by_step_t[step_t]["d2h_gpu_ms"].append(d2h_gpu_ms)
        by_step_t[step_t]["d2h_host_ms"].append(d2h_host_ms)
        by_step_t[step_t]["participants"].append(float(participants))

    participant_rows = []
    for participants, metrics in sorted(by_participants.items()):
        decode = row_stats(metrics["decode_gpu_ms"])
        d2h_gpu = row_stats(metrics["d2h_gpu_ms"])
        d2h_host = row_stats(metrics["d2h_host_ms"])
        participant_rows.append(
            round_row(
                {
                    "participants": participants,
                    "count": decode["count"],
                    "decode_avg_ms": decode["avg"],
                    "decode_p95_ms": decode["p95"],
                    "d2h_gpu_avg_ms": d2h_gpu["avg"],
                    "d2h_host_avg_ms": d2h_host["avg"],
                }
            )
        )

    step_rows = []
    for step_t, metrics in sorted(by_step_t.items()):
        decode = row_stats(metrics["decode_gpu_ms"])
        participants = row_stats(metrics["participants"])
        d2h_host = row_stats(metrics["d2h_host_ms"])
        step_rows.append(
            round_row(
                {
                    "step_t": step_t,
                    "count": decode["count"],
                    "participants_avg": participants["avg"],
                    "participants_p95": participants["p95"],
                    "decode_avg_ms": decode["avg"],
                    "decode_p95_ms": decode["p95"],
                    "d2h_host_avg_ms": d2h_host["avg"],
                }
            )
        )

    return {
        "source": str(path),
        "event_count": len(all_decode),
        "overall": {
            "decode_gpu_ms": round_row(row_stats(all_decode)),
            "d2h_gpu_ms": round_row(row_stats(all_d2h_gpu)),
            "d2h_host_ms": round_row(row_stats(all_d2h_host)),
        },
        "by_participants": participant_rows,
        "by_step_t": step_rows,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)\n"
    columns = list(rows[0])
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    sep = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(
        " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for row in rows
    )
    return f"{header}\n{sep}\n{body}\n"


def write_outputs(summary: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    parts = [
        "# MOSS Streaming CUDA Timing Histogram",
        "",
        f"- source: `{summary['source']}`",
        f"- events: `{summary['event_count']}`",
        "",
        "## By Participants",
        "",
        markdown_table(summary["by_participants"]),
        "",
        "## By Step T",
        "",
        markdown_table(summary["by_step_t"]),
    ]
    md_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("moss_streaming_cuda_timing_histogram"),
    )
    args = parser.parse_args()

    summary = summarize(args.source)
    write_outputs(summary, args.output_prefix)
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {args.output_prefix.with_suffix('.json')}")
    print(f"wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
