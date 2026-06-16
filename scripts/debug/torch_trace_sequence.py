# SPDX-License-Identifier: Apache-2.0
"""Print an ordered execution sequence from torch trace + request events.

This is a focused companion to ``torch_trace_summary.py``. It answers:

1. Which MOSS/Omni profiler ranges ran, in timestamp order?
2. For each request, what stage events happened, in timestamp order?

Usage:

    python scripts/debug/torch_trace_sequence.py \
        --trace /tmp/run/trace_pid123_rank0.trace.json.gz \
        --events /tmp/run/events
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang_omni.profiler.views import reconstruct_timelines


@dataclass
class RangeEvent:
    ts_us: float
    dur_us: float
    pid: str
    tid: str
    name: str


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_trace(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as fp:
        obj = json.load(fp)
    events = obj.get("traceEvents", []) if isinstance(obj, dict) else obj
    if not isinstance(events, list):
        raise ValueError(f"{path}: traceEvents is not a list")
    return [ev for ev in events if isinstance(ev, dict)]


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in patterns:
        if any(ch in raw for ch in "*?["):
            paths.extend(Path(p) for p in sorted(glob.glob(raw)))
        else:
            paths.append(Path(raw))
    return paths


def _is_complete_range(ev: dict[str, Any]) -> bool:
    try:
        dur = float(ev.get("dur") or 0.0)
    except (TypeError, ValueError):
        return False
    return ev.get("ph") == "X" and dur > 0.0


def _is_interesting_range(name: str) -> bool:
    return name.startswith(
        (
            "moss.",
            "omni.",
            "sglang.",
            "scheduler.",
            "ar_",
            "moss_ar_",
            "vocoder_",
            "preprocess_",
        )
    )


def _trace_ranges(trace_paths: list[Path], *, min_us: float) -> list[RangeEvent]:
    ranges: list[RangeEvent] = []
    for path in trace_paths:
        for ev in _load_trace(path):
            name = str(ev.get("name") or "")
            if not _is_complete_range(ev) or not _is_interesting_range(name):
                continue
            dur_us = float(ev.get("dur") or 0.0)
            if dur_us < min_us:
                continue
            ranges.append(
                RangeEvent(
                    ts_us=float(ev.get("ts") or 0.0),
                    dur_us=dur_us,
                    pid=str(ev.get("pid") or ""),
                    tid=str(ev.get("tid") or ""),
                    name=name,
                )
            )
    ranges.sort(key=lambda item: (item.ts_us, -item.dur_us, item.name))
    return ranges


def _fmt_ms(value_ms: float) -> str:
    return f"{value_ms:.3f}"


def _metadata_brief(metadata: dict[str, Any], *, max_chars: int) -> str:
    if not metadata:
        return ""
    text = json.dumps(metadata, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def print_trace_range_timeline(
    ranges: list[RangeEvent], *, limit: int, longest: int
) -> None:
    print("# Torch User Range Timeline")
    if not ranges:
        print("(empty: no MOSS/Omni record_function ranges found)")
        return
    t0 = ranges[0].ts_us
    print("rel_ms | dur_ms | pid | tid | name")
    print("---:|---:|---|---|---")
    for item in ranges[:limit]:
        print(
            f"{_fmt_ms((item.ts_us - t0) / 1000.0)} | "
            f"{_fmt_ms(item.dur_us / 1000.0)} | "
            f"{item.pid} | {item.tid} | `{item.name}`"
        )
    if len(ranges) > limit:
        print(f"\n... truncated {len(ranges) - limit} later ranges")

    print("\n# Longest Torch User Ranges")
    print("dur_ms | rel_ms | pid | tid | name")
    print("---:|---:|---|---|---")
    for item in sorted(ranges, key=lambda row: row.dur_us, reverse=True)[:longest]:
        print(
            f"{_fmt_ms(item.dur_us / 1000.0)} | "
            f"{_fmt_ms((item.ts_us - t0) / 1000.0)} | "
            f"{item.pid} | {item.tid} | `{item.name}`"
        )


def print_request_timelines(
    events_dir: Path, *, request_limit: int, event_limit: int, metadata_chars: int
) -> None:
    timelines = reconstruct_timelines(events_dir)
    print("\n# Request Event Timelines")
    print(f"requests: {len(timelines)}")
    if not timelines:
        return
    ordered = sorted(timelines.values(), key=lambda tl: tl.total_ms, reverse=True)[
        :request_limit
    ]
    for timeline in ordered:
        print(
            f"\n## request {timeline.request_id} "
            f"total_ms={_fmt_ms(timeline.total_ms)} events={len(timeline.events)}"
        )
        print("rel_ms | stage | event | metadata")
        print("---:|---|---|---")
        for ev in timeline.to_relative()[:event_limit]:
            print(
                f"{_fmt_ms(float(ev['t_rel_ms']))} | "
                f"{ev.get('stage', '')} | "
                f"`{ev.get('event_name', '')}` | "
                f"{_metadata_brief(ev.get('metadata') or {}, max_chars=metadata_chars)}"
            )
        if len(timeline.events) > event_limit:
            print(f"\n... truncated {len(timeline.events) - event_limit} later events")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--min-us", type=float, default=0.0)
    parser.add_argument("--range-limit", type=int, default=300)
    parser.add_argument("--longest", type=int, default=40)
    parser.add_argument("--request-limit", type=int, default=8)
    parser.add_argument("--event-limit", type=int, default=250)
    parser.add_argument("--metadata-chars", type=int, default=180)
    args = parser.parse_args()

    trace_paths = _expand_paths(args.trace)
    ranges = _trace_ranges(trace_paths, min_us=args.min_us)
    print_trace_range_timeline(ranges, limit=args.range_limit, longest=args.longest)
    print_request_timelines(
        Path(args.events),
        request_limit=args.request_limit,
        event_limit=args.event_limit,
        metadata_chars=args.metadata_chars,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
