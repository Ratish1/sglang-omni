# SPDX-License-Identifier: Apache-2.0
"""Summarize Qwen3-TTS CUDA synchronization evidence from one Kineto trace.

Run :mod:`analyze_cuda_sync_trace` first.  This companion keeps the mechanical
gate reproducible: it proves that each selected semantic range executed, that
an H2D launch occurred in it, and that the range contains no CUDA wait.  It
also reports overlap-aware process-level interval unions alongside the raw
per-occurrence sums.  Neither number is a serving-speedup estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from analyze_cuda_sync_trace import iter_trace_events

_CLEAN_RANGES = (
    "qwen3_tts.preprocess.speaker_mel_h2d",
    "qwen3_tts.preprocess.speaker_embedding_h2d",
    "qwen3_tts.prompt.token_ids_h2d",
    "qwen3_tts.prompt.ref_code_h2d",
    "qwen3_tts.sampling_masks.rebuild",
    "qwen3_tts.sampling_metadata.h2d",
    "qwen3_tts.preprocess.text_tokenizer",
)

_OWNERSHIP_RANGES = (
    "qwen3_tts.preprocess.prompt.build",
    "qwen3_tts.preprocess.reference_tokenizer.encode",
    "qwen3_tts.preprocess.speaker_encoder.forward",
    "qwen3_tts.preprocess.text_tokenizer",
    "qwen3_tts.sampling.base_pipeline",
    "qwen3_tts.vocoder.tokenizer.decode",
)

_NORMALIZE_KEY_RE = re.compile(r"[^a-z0-9]")


def _normalized_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args")
    if not isinstance(args, dict):
        return {}
    return {
        _NORMALIZE_KEY_RE.sub("", str(key).lower()): value
        for key, value in args.items()
    }


def _correlation_id(event: dict[str, Any]) -> str | None:
    args = _normalized_args(event)
    for key in ("correlation", "correlationid", "linkedcorrelationid"):
        value = args.get(key)
        if value not in (None, "", 0, "0"):
            return str(value)
    return None


def _is_complete_event(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur", 0), (int, float))
    )


def _range_activity(
    trace_path: Path,
    expected_ranges: set[str],
) -> tuple[Counter[str], Counter[str]]:
    """Count CPU ranges and correlation-backed H2D launches inside them."""

    ranges_by_thread: dict[tuple[Any, Any], list[tuple[str, float, float]]] = (
        defaultdict(list)
    )
    runtime_memcpys: list[tuple[Any, Any, float, float, str]] = []
    gpu_copy_directions: dict[str, str] = {}

    for event in iter_trace_events(trace_path):
        if not _is_complete_event(event):
            continue
        name = str(event.get("name", ""))
        ts_us = float(event["ts"])
        end_us = ts_us + max(float(event.get("dur", 0.0)), 0.0)
        if event.get("cat") == "user_annotation" and name in expected_ranges:
            ranges_by_thread[(event.get("pid"), event.get("tid"))].append(
                (name, ts_us, end_us)
            )
            continue

        correlation_id = _correlation_id(event)
        if correlation_id is None:
            continue
        lowered = name.lower()
        if name == "cudaMemcpyAsync":
            runtime_memcpys.append(
                (
                    event.get("pid"),
                    event.get("tid"),
                    ts_us,
                    end_us,
                    correlation_id,
                )
            )
        elif "memcpy" in lowered:
            if "htod" in lowered or "host -> device" in lowered:
                gpu_copy_directions[correlation_id] = "HtoD"
            elif "dtoh" in lowered or "device -> host" in lowered:
                gpu_copy_directions[correlation_id] = "DtoH"

    calls: Counter[str] = Counter()
    h2d_launches: Counter[str] = Counter()
    for ranges in ranges_by_thread.values():
        for name, _, _ in ranges:
            calls[name] += 1
        ranges.sort(key=lambda row: (row[1], row[2]))

    for pid, tid, start_us, end_us, correlation_id in runtime_memcpys:
        if gpu_copy_directions.get(correlation_id) != "HtoD":
            continue
        for name, range_start_us, range_end_us in ranges_by_thread.get((pid, tid), ()):
            if range_start_us <= start_us and end_us <= range_end_us:
                h2d_launches[name] += 1
    return calls, h2d_launches


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    rows = sorted((start, end) for start, end in intervals if end > start)
    if not rows:
        return 0.0
    total = 0.0
    active_start, active_end = rows[0]
    for start, end in rows[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
            continue
        total += active_end - active_start
        active_start, active_end = start, end
    return total + active_end - active_start


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get("metrics", {}).get(key) or 0.0) for row in rows)


def _interval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sync_intervals = [
        (float(row["sync"]["start_us"]), float(row["sync"]["end_us"]))
        for row in rows
        if row.get("sync")
    ]
    compound_intervals = [
        (
            float(row["blocking_copy"]["runtime_memcpy"]["start_us"]),
            float(row["sync"]["end_us"]),
        )
        for row in rows
        if row.get("blocking_copy") and row.get("sync")
    ]
    bubble_intervals = [
        (
            float(row["prior_waited_gpu"]["end_us"]),
            float(row["next_causal_gpu"]["start_us"]),
        )
        for row in rows
        if row.get("prior_waited_gpu")
        and row.get("next_causal_gpu")
        and row.get("metrics", {}).get("post_sync_gpu_bubble_us") is not None
    ]
    queue_horizons = [
        float(value)
        for row in rows
        if (value := row.get("metrics", {}).get("queue_horizon_at_sync_start_us"))
        is not None
    ]
    launch_gaps = [
        float(value)
        for row in rows
        if (value := row.get("metrics", {}).get("host_launch_gap_after_sync_us"))
        is not None
    ]
    compound_sum = sum(
        float(row["blocking_copy"]["compound_host_block_us"])
        for row in rows
        if row.get("blocking_copy")
    )
    return {
        "sync_wait_sum_us": _sum_metric(rows, "sync_wait_us"),
        "sync_wait_union_us": _union_duration(sync_intervals),
        "compound_host_block_sum_us": compound_sum,
        "compound_host_block_union_us": _union_duration(compound_intervals),
        "post_sync_gpu_bubble_sum_us": _sum_metric(rows, "post_sync_gpu_bubble_us"),
        "post_sync_gpu_bubble_union_us": _union_duration(bubble_intervals),
        "queue_horizon_p50_us": _percentile(queue_horizons, 0.50),
        "queue_horizon_p95_us": _percentile(queue_horizons, 0.95),
        "host_launch_gap_p50_us": _percentile(launch_gaps, 0.50),
        "host_launch_gap_p95_us": _percentile(launch_gaps, 0.95),
    }


def _group_rows(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        sync = row.get("sync") or {}
        transfer = row.get("transfer") or {}
        key = (
            str(row.get("semantic_range") or "unscoped"),
            str(sync.get("name") or "unknown"),
            str(row.get("parent_cpu_op") or "direct_wait"),
            str(transfer.get("direction") or "none"),
        )
        groups[key].append(row)

    result = []
    for key, rows in groups.items():
        result.append(
            {
                "semantic_range": key[0],
                "sync_name": key[1],
                "parent_cpu_op": key[2],
                "transfer_direction": key[3],
                "count": len(rows),
                "blocking_copy_count": sum(
                    row.get("blocking_copy") is not None for row in rows
                ),
                "transfer_bytes": sum(
                    int((row.get("transfer") or {}).get("bytes") or 0) for row in rows
                ),
                **_interval_metrics(rows),
            }
        )
    result.sort(
        key=lambda row: (
            -row["sync_wait_sum_us"],
            -row["compound_host_block_sum_us"],
            row["semantic_range"],
        )
    )
    return result


def _fmt_ms(value_us: float | int | None) -> str:
    return "-" if value_us is None else f"{float(value_us) / 1000.0:.3f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Qwen3-TTS CUDA synchronization summary",
        "",
        (
            "Mechanical clean-range gate: "
            f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**"
        ),
        "",
        "## Selected ranges",
        "",
        "| Range | Calls | H2D launches | Synchronizations | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["gate"]["ranges"]:
        lines.append(
            f"| `{row['name']}` | {row['calls']} | {row['h2d_launches']} | "
            f"{row['synchronizations']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Trace-wide interval metrics",
            "",
            "Raw sums count every occurrence and may overlap across host threads. "
            "Union values are process-timeline coverage, not recoverable serving time.",
            "",
            "| Metric | Raw sum | Interval union |",
            "|---|---:|---:|",
            "| Host synchronization wait | "
            f"{_fmt_ms(summary['intervals']['sync_wait_sum_us'])} ms | "
            f"{_fmt_ms(summary['intervals']['sync_wait_union_us'])} ms |",
            "| Compound blocking copy | "
            f"{_fmt_ms(summary['intervals']['compound_host_block_sum_us'])} ms | "
            f"{_fmt_ms(summary['intervals']['compound_host_block_union_us'])} ms |",
            "| Correlated post-sync GPU bubble | "
            f"{_fmt_ms(summary['intervals']['post_sync_gpu_bubble_sum_us'])} ms | "
            f"{_fmt_ms(summary['intervals']['post_sync_gpu_bubble_union_us'])} ms |",
            "",
            "## Synchronization owners",
            "",
            "| Semantic range | API | Parent | Direction | Count | Sync wait | "
            "Compound block | Bubble | Bytes |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["groups"]:
        lines.append(
            f"| `{row['semantic_range']}` | `{row['sync_name']}` | "
            f"`{row['parent_cpu_op']}` | {row['transfer_direction']} | "
            f"{row['count']} | {_fmt_ms(row['sync_wait_sum_us'])} ms | "
            f"{_fmt_ms(row['compound_host_block_sum_us'])} ms | "
            f"{_fmt_ms(row['post_sync_gpu_bubble_sum_us'])} ms | "
            f"{row['transfer_bytes']} |"
        )
    lines.extend(
        [
            "",
            "The synchronization detector and trace analyzer are complementary. "
            "A zero detector count does not prove absence of every CUDA wait, and "
            "none of these timing fields alone establishes an end-to-end speedup.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--analysis-dir",
        required=True,
        type=Path,
        help="Directory containing cuda_sync_occurrences.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless every selected range is exercised and wait-free",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    occurrence_path = args.analysis_dir / "cuda_sync_occurrences.json"
    payload = json.loads(occurrence_path.read_text(encoding="utf-8"))
    occurrences = payload["occurrences"]

    expected = set(_CLEAN_RANGES) | set(_OWNERSHIP_RANGES)
    calls, h2d_launches = _range_activity(args.trace, expected)
    sync_counts = Counter(
        str(row.get("semantic_range"))
        for row in occurrences
        if row.get("semantic_range") is not None
    )
    range_rows = []
    for name in _CLEAN_RANGES:
        call_count = calls[name]
        h2d_count = h2d_launches[name]
        sync_count = sync_counts[name]
        if call_count == 0:
            status = "missing_range"
        elif h2d_count == 0:
            status = "missing_h2d_exercise"
        elif sync_count:
            status = "contains_synchronization"
        else:
            status = "pass"
        range_rows.append(
            {
                "name": name,
                "calls": call_count,
                "h2d_launches": h2d_count,
                "synchronizations": sync_count,
                "status": status,
            }
        )

    missing_ownership = sorted(name for name in _OWNERSHIP_RANGES if calls[name] == 0)
    passed = (
        all(row["status"] == "pass" for row in range_rows) and not missing_ownership
    )
    summary = {
        "schema_version": 1,
        "trace": str(args.trace.resolve()),
        "notes": [
            "All causal comparisons are within one process trace.",
            "Per-occurrence sums can overlap across host threads.",
            "Interval unions are timeline coverage, not recoverable serving time.",
            "Synchronization counts do not establish an end-to-end performance claim.",
        ],
        "gate": {
            "passed": passed,
            "ranges": range_rows,
            "ownership_range_calls": {name: calls[name] for name in _OWNERSHIP_RANGES},
            "missing_ownership_ranges": missing_ownership,
        },
        "intervals": _interval_metrics(occurrences),
        "groups": _group_rows(occurrences),
    }
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "qwen3_tts_sync_summary.json"
    markdown_path = args.analysis_dir / "qwen3_tts_sync_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, markdown_path)
    print(
        json.dumps(
            {
                "passed": passed,
                "summary_json": str(json_path.resolve()),
                "summary_markdown": str(markdown_path.resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
