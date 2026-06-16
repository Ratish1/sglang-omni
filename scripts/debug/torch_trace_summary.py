# SPDX-License-Identifier: Apache-2.0
"""Summarize a PyTorch/Chrome trace without manually using Perfetto.

Usage:

    python scripts/debug/torch_trace_summary.py /data/run/trace_pid123_rank0.trace.json.gz
    python scripts/debug/torch_trace_summary.py /data/run/trace*.trace.json.gz --events-dir /data/run/events

The script is intentionally heuristic: PyTorch trace categories differ across
versions, but names like ``aten::clone``, ``cudaMemcpyAsync``, and kernel
categories are stable enough to quickly answer whether a trace contains useful
CPU/PyTorch ranges or only GPU kernels.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Agg:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def add(self, dur_us: float) -> None:
        self.count += 1
        self.total_us += dur_us
        self.max_us = max(self.max_us, dur_us)

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_trace(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as fp:
        obj = json.load(fp)
    if isinstance(obj, dict):
        events = obj.get("traceEvents", [])
    else:
        events = obj
    if not isinstance(events, list):
        raise ValueError(f"{path}: traceEvents is not a list")
    return [ev for ev in events if isinstance(ev, dict)]


def _cat(ev: dict[str, Any]) -> str:
    return str(ev.get("cat") or "")


def _name(ev: dict[str, Any]) -> str:
    return str(ev.get("name") or "")


def _dur_us(ev: dict[str, Any]) -> float:
    try:
        return float(ev.get("dur") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_complete(ev: dict[str, Any]) -> bool:
    return ev.get("ph") == "X" and _dur_us(ev) > 0.0


def _is_aten(name: str, cat: str) -> bool:
    del cat
    return name.startswith(("aten::", "prims::", "torch::"))


def _is_user_range(name: str, cat: str) -> bool:
    cat_l = cat.lower()
    if "user_annotation" in cat_l or "python_function" in cat_l:
        return True
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


def _is_cuda_runtime(name: str, cat: str) -> bool:
    cat_l = cat.lower()
    if "cuda_runtime" in cat_l or "cuda_api" in cat_l:
        return True
    return name.startswith(("cuda", "cu")) and "kernel" not in name.lower()


def _is_kernel(name: str, cat: str) -> bool:
    cat_l = cat.lower()
    if "kernel" in cat_l:
        return True
    name_l = name.lower()
    if "kernel" in name_l and not _is_cuda_runtime(name, cat):
        return True
    return False


def _is_memcpy(name: str, cat: str) -> bool:
    hay = f"{name} {cat}".lower()
    return "memcpy" in hay or "memset" in hay or "gpu_memcpy" in hay


def _add(table: dict[str, Agg], key: str, dur_us: float) -> None:
    table[key].add(dur_us)


def _fmt_us(us: float) -> str:
    if us >= 1000.0:
        return f"{us / 1000.0:.3f} ms"
    return f"{us:.3f} us"


def _print_top(title: str, table: dict[str, Agg], limit: int) -> None:
    print(f"\n## {title}")
    if not table:
        print("(empty)")
        return
    print("count | total | avg | max | name")
    print("---:|---:|---:|---:|---")
    rows = sorted(table.items(), key=lambda item: item[1].total_us, reverse=True)
    for name, agg in rows[:limit]:
        print(
            f"{agg.count} | {_fmt_us(agg.total_us)} | {_fmt_us(agg.avg_us)} | "
            f"{_fmt_us(agg.max_us)} | `{name}`"
        )


def _print_event_report(events_dir: Path) -> None:
    try:
        from sglang_omni.profiler.views import build_report, format_table
    except Exception as exc:  # pragma: no cover - debug fallback
        print(f"\n## Request Event Report\ncould not import profiler views: {exc}")
        return
    report = build_report(events_dir)
    print(f"\n# Request Events: {report['request_count']}")
    print("\n## Stage Breakdown")
    print(
        format_table(
            report["stage_breakdown"],
            ["stage", "interval", "count", "total_ms", "avg_ms", "p95_ms"],
        ).rstrip()
    )
    print("\n## Hop Breakdown")
    print(
        format_table(
            report["hop_breakdown"],
            ["src", "dst", "kind", "count", "total_ms", "avg_ms", "p95_ms"],
        ).rstrip()
    )


def summarize_trace(path: Path, *, limit: int, grep: Iterable[str]) -> None:
    events = _load_trace(path)
    complete_events = [ev for ev in events if _is_complete(ev)]

    categories = Counter(_cat(ev) or "(none)" for ev in events)
    phases = Counter(str(ev.get("ph") or "(none)") for ev in events)
    aten: dict[str, Agg] = defaultdict(Agg)
    user: dict[str, Agg] = defaultdict(Agg)
    cuda_runtime: dict[str, Agg] = defaultdict(Agg)
    kernels: dict[str, Agg] = defaultdict(Agg)
    memcpy: dict[str, Agg] = defaultdict(Agg)
    complete: dict[str, Agg] = defaultdict(Agg)
    grep_tables: dict[str, dict[str, Agg]] = {
        pattern: defaultdict(Agg) for pattern in grep
    }

    for ev in complete_events:
        name = _name(ev)
        cat = _cat(ev)
        dur = _dur_us(ev)
        _add(complete, name, dur)
        if _is_aten(name, cat):
            _add(aten, name, dur)
        if _is_user_range(name, cat):
            _add(user, name, dur)
        if _is_cuda_runtime(name, cat):
            _add(cuda_runtime, name, dur)
        if _is_kernel(name, cat):
            _add(kernels, name, dur)
        if _is_memcpy(name, cat):
            _add(memcpy, name, dur)
        name_l = name.lower()
        for pattern, table in grep_tables.items():
            if pattern.lower() in name_l:
                _add(table, name, dur)

    print(f"# Trace: {path}")
    print(f"events: {len(events)}")
    print(f"complete_events: {len(complete_events)}")
    print(f"aten_events: {sum(agg.count for agg in aten.values())}")
    print(f"user_range_events: {sum(agg.count for agg in user.values())}")
    print(f"cuda_runtime_events: {sum(agg.count for agg in cuda_runtime.values())}")
    print(f"kernel_events: {sum(agg.count for agg in kernels.values())}")
    print(f"memcpy_or_memset_events: {sum(agg.count for agg in memcpy.values())}")

    print("\n## Phase Counts")
    for key, value in phases.most_common(20):
        print(f"{key}: {value}")

    print("\n## Category Counts")
    for key, value in categories.most_common(30):
        print(f"{key}: {value}")

    if not aten and not user:
        print(
            "\nWARNING: no aten:: or user annotation ranges were found. This usually "
            "means you opened a kernel-only trace file, torch profiling did not "
            "start in the Python process that ran MOSS, or the current code lacks "
            "record_function labels for the region you want."
        )

    _print_top("Top PyTorch CPU Ops", aten, limit)
    _print_top("Top User / RecordFunction Ranges", user, limit)
    _print_top("Top CUDA Runtime/API Calls", cuda_runtime, limit)
    _print_top("Top Memcpy/Memset Events", memcpy, limit)
    _print_top("Top GPU Kernels", kernels, limit)
    _print_top("Top Complete Events Overall", complete, limit)

    for pattern, table in grep_tables.items():
        _print_top(f"Matches: {pattern}", table, limit)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Chrome trace json/json.gz files")
    parser.add_argument("--events-dir", help="Optional request event JSONL directory")
    parser.add_argument("--limit", type=int, default=30, help="Rows per section")
    parser.add_argument(
        "--grep",
        action="append",
        default=[],
        help="Case-insensitive substring section to add; can be repeated",
    )
    args = parser.parse_args()

    for raw in args.traces:
        if any(ch in raw for ch in "*?["):
            paths = [Path(path) for path in sorted(glob.glob(raw))]
        else:
            paths = [Path(raw)]
        for path in paths:
            summarize_trace(path, limit=args.limit, grep=args.grep)

    if args.events_dir:
        _print_event_report(Path(args.events_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
