#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize torch Chrome traces for MOSS vocoder profile ranges.

The request-level profiler tells us which Python ranges are expensive. This
script reads the torch profiler Chrome trace and shows which CUDA kernels,
runtime calls, and memcpy events appear inside those same record_function
ranges. It is intentionally a debug aid, not a full Perfetto replacement.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SCOPES = (
    "moss_vocoder_self_attn",
    "moss_vocoder_project_qkv",
    "moss_vocoder_rope",
    "moss_vocoder_rope_cache",
    "moss_vocoder_rope_select",
    "moss_vocoder_rope_view",
    "moss_vocoder_rope_float",
    "moss_vocoder_rope_rotate",
    "moss_vocoder_rope_stack",
    "moss_vocoder_qkv_contiguous",
    "moss_vocoder_flash_window",
    "moss_vocoder_flash_attn",
    "moss_vocoder_attn_output_reshape",
    "moss_vocoder_attn_output_proj",
    "moss_vocoder_ffn",
    "moss_vocoder_ffn_linear_in",
    "moss_vocoder_ffn_activation",
    "moss_vocoder_ffn_linear_out",
)


GPU_CATEGORIES = {
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
    "cuda_runtime",
    "cuda_driver",
}


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    elif isinstance(payload, list):
        events = payload
    else:
        raise TypeError(f"unsupported trace payload type: {type(payload).__name__}")
    return [event for event in events if isinstance(event, dict)]


def _duration_ms(event: dict[str, Any]) -> float:
    return float(event.get("dur") or 0.0) / 1000.0


def _is_complete_event(event: dict[str, Any]) -> bool:
    return event.get("ph") == "X" and event.get("ts") is not None


def _scope_events(
    events: list[dict[str, Any]],
    scope_names: set[str],
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if _is_complete_event(event) and event.get("name") in scope_names
    ]


def _is_gpu_event(event: dict[str, Any]) -> bool:
    if not _is_complete_event(event):
        return False
    cat = str(event.get("cat") or "").lower()
    name = str(event.get("name") or "")
    lower_name = name.lower()
    return (
        cat in GPU_CATEGORIES
        or lower_name.startswith("cuda")
        or "memcpy" in lower_name
        or "memset" in lower_name
    )


def _is_cpu_op_event(event: dict[str, Any], scope_names: set[str]) -> bool:
    if not _is_complete_event(event) or _is_gpu_event(event):
        return False
    name = str(event.get("name") or "")
    cat = str(event.get("cat") or "").lower()
    return name not in scope_names and (
        name.startswith("aten::")
        or name.startswith("torch::")
        or cat in {"cpu_op", "user_annotation"}
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = 0.95 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [_duration_ms(event) for event in events]
    by_name: Counter[str] = Counter()
    by_cat: Counter[str] = Counter()
    duration_by_name: defaultdict[str, float] = defaultdict(float)
    for event in events:
        name = str(event.get("name") or "")
        cat = str(event.get("cat") or "")
        dur = _duration_ms(event)
        by_name[name] += 1
        by_cat[cat] += 1
        duration_by_name[name] += dur
    top_names = [
        {"name": name, "count": by_name[name], "total_ms": round(total_ms, 3)}
        for name, total_ms in sorted(
            duration_by_name.items(), key=lambda item: item[1], reverse=True
        )[:12]
    ]
    return {
        "count": len(events),
        "total_ms": round(sum(durations), 3),
        "avg_ms": round(sum(durations) / len(durations), 6) if durations else 0.0,
        "p95_ms": round(_p95(durations), 6),
        "categories": dict(by_cat.most_common()),
        "top_names": top_names,
    }


def build_summary(
    events: list[dict[str, Any]], scope_names: set[str]
) -> dict[str, Any]:
    scopes = _scope_events(events, scope_names)
    gpu_events = sorted(
        [event for event in events if _is_gpu_event(event)],
        key=lambda event: float(event.get("ts") or 0.0),
    )
    cpu_events = sorted(
        [event for event in events if _is_cpu_op_event(event, scope_names)],
        key=lambda event: float(event.get("ts") or 0.0),
    )
    gpu_starts = [float(event.get("ts") or 0.0) for event in gpu_events]
    cpu_starts = [float(event.get("ts") or 0.0) for event in cpu_events]

    by_scope: dict[str, dict[str, Any]] = {}
    for name in sorted(scope_names):
        matching_scopes = [event for event in scopes if event.get("name") == name]
        scope_durations = [_duration_ms(event) for event in matching_scopes]
        enclosed_gpu: list[dict[str, Any]] = []
        enclosed_cpu: list[dict[str, Any]] = []
        for scope in matching_scopes:
            start = float(scope.get("ts") or 0.0)
            end = start + float(scope.get("dur") or 0.0)
            left = bisect.bisect_left(gpu_starts, start)
            right = bisect.bisect_left(gpu_starts, end)
            enclosed_gpu.extend(gpu_events[left:right])
            left = bisect.bisect_left(cpu_starts, start)
            right = bisect.bisect_left(cpu_starts, end)
            enclosed_cpu.extend(cpu_events[left:right])
        by_scope[name] = {
            "range_count": len(matching_scopes),
            "range_total_ms": round(sum(scope_durations), 3),
            "range_avg_ms": (
                round(sum(scope_durations) / len(scope_durations), 6)
                if scope_durations
                else 0.0
            ),
            "range_p95_ms": round(_p95(scope_durations), 6),
            "enclosed_gpu": _summarize_events(enclosed_gpu),
            "enclosed_cpu_ops": _summarize_events(enclosed_cpu),
        }
    return {
        "scope_names": sorted(scope_names),
        "event_count": len(events),
        "gpu_event_count": len(gpu_events),
        "cpu_op_event_count": len(cpu_events),
        "scopes": by_scope,
    }


def _format_table(rows: list[list[Any]], headers: list[str]) -> str:
    table = [headers, *[[str(cell) for cell in row] for row in rows]]
    widths = [max(len(row[idx]) for row in table) for idx in range(len(headers))]
    out = []
    out.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(table[0])))
    out.append("-+-".join("-" * width for width in widths))
    for row in table[1:]:
        out.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(out)


def format_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for name, data in summary["scopes"].items():
        gpu = data["enclosed_gpu"]
        cpu = data["enclosed_cpu_ops"]
        rows.append(
            [
                name,
                data["range_count"],
                data["range_avg_ms"],
                data["range_p95_ms"],
                gpu["count"],
                gpu["total_ms"],
                gpu["avg_ms"],
                ", ".join(
                    f"{item['name']} ({item['count']})" for item in gpu["top_names"][:3]
                ),
                ", ".join(
                    f"{item['name']} ({item['count']})" for item in cpu["top_names"][:3]
                ),
            ]
        )
    return (
        "# MOSS Vocoder Torch Trace Summary\n\n"
        f"- trace events: `{summary['event_count']}`\n"
        f"- gpu/runtime events: `{summary['gpu_event_count']}`\n\n"
        f"- cpu op events: `{summary['cpu_op_event_count']}`\n\n"
        + _format_table(
            rows,
            [
                "scope",
                "ranges",
                "range_avg_ms",
                "range_p95_ms",
                "gpu_events",
                "gpu_total_ms",
                "gpu_avg_ms",
                "top enclosed gpu/runtime names",
                "top enclosed cpu op names",
            ],
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MOSS vocoder record_function ranges in a torch trace."
    )
    parser.add_argument("trace", type=Path, help="Chrome trace .json or .json.gz")
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Additional exact record_function scope name to summarize.",
    )
    parser.add_argument("--json-out", type=Path, help="Write machine-readable summary")
    parser.add_argument("--md-out", type=Path, help="Write markdown summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope_names = set(DEFAULT_SCOPES)
    scope_names.update(args.scope)
    events = _load_trace(args.trace)
    summary = build_summary(events, scope_names)
    markdown = format_markdown(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
