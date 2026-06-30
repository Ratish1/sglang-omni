# SPDX-License-Identifier: Apache-2.0
"""Summarize torch-profiler Chrome traces for Phase 2C optimization work.

The input can be one or more ``*.trace.json`` / ``*.trace.json.gz`` files or
directories containing such traces. The output is a compact JSON and Markdown
report that ranks CPU ops, CUDA kernels, runtime calls, annotations, and large
kernel gaps. It intentionally has no third-party dependencies so it can run in
the H100 container after a profiling run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TOP_K = 30
DEFAULT_GAP_THRESHOLD_US = 1000.0
TRACE_SUFFIXES = (".trace.json", ".trace.json.gz")


@dataclass
class AggregateRow:
    name: str
    category: str
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    files: list[str] = field(default_factory=list)


@dataclass
class GapRow:
    trace: str
    pid: int | str
    tid: int | str
    start_ms: float
    gap_ms: float
    prev_name: str
    next_name: str


@dataclass
class TraceRunSummary:
    label: str
    traces: list[str]
    event_count: int
    complete_event_count: int
    total_trace_window_ms: float
    top_cpu_ops: list[AggregateRow]
    top_cuda_kernels: list[AggregateRow]
    top_cuda_runtime: list[AggregateRow]
    top_user_annotations: list[AggregateRow]
    top_memory_events: list[AggregateRow]
    top_all_events: list[AggregateRow]
    top_kernel_gaps: list[GapRow]
    signals: list[str]


def resolve_trace_paths(paths: Iterable[Path]) -> list[Path]:
    traces: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if _is_trace_file(path):
                traces.append(path.resolve())
            else:
                raise FileNotFoundError(f"not a torch trace file: {path}")
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"path does not exist: {path}")
        traces.extend(p.resolve() for p in path.rglob("*") if _is_trace_file(p))
    return sorted(dict.fromkeys(traces))


def summarize_traces(
    traces: list[Path],
    *,
    label: str,
    top_k: int,
    gap_threshold_us: float,
) -> TraceRunSummary:
    buckets: dict[str, dict[str, list[tuple[float, str]]]] = {
        "cpu_op": {},
        "cuda_kernel": {},
        "cuda_runtime": {},
        "user_annotation": {},
        "memory": {},
        "all": {},
    }
    kernel_intervals: dict[
        tuple[str, int | str, int | str], list[tuple[float, float, str]]
    ] = {}
    event_count = 0
    complete_event_count = 0
    min_ts: float | None = None
    max_ts: float | None = None

    for trace_path in traces:
        trace = _load_trace(trace_path)
        for event in _iter_trace_events(trace):
            event_count += 1
            if event.get("ph") != "X":
                continue
            dur_us = _float_or_none(event.get("dur"))
            ts_us = _float_or_none(event.get("ts"))
            if dur_us is None or dur_us <= 0:
                continue
            complete_event_count += 1
            if ts_us is not None:
                min_ts = ts_us if min_ts is None else min(min_ts, ts_us)
                max_ts = (
                    ts_us + dur_us if max_ts is None else max(max_ts, ts_us + dur_us)
                )
            name = _normalize_name(str(event.get("name") or "unknown"))
            category = _classify_event(event)
            _append_duration(buckets["all"], name, category, dur_us, trace_path)
            if category in buckets:
                _append_duration(buckets[category], name, category, dur_us, trace_path)
            if category == "cuda_kernel" and ts_us is not None:
                key = (str(trace_path), event.get("pid", "?"), event.get("tid", "?"))
                kernel_intervals.setdefault(key, []).append(
                    (ts_us, ts_us + dur_us, name)
                )

    top_kernel_gaps = _summarize_kernel_gaps(
        kernel_intervals,
        top_k=top_k,
        gap_threshold_us=gap_threshold_us,
    )
    total_trace_window_ms = 0.0
    if min_ts is not None and max_ts is not None:
        total_trace_window_ms = (max_ts - min_ts) / 1000.0

    summary = TraceRunSummary(
        label=label,
        traces=[str(path) for path in traces],
        event_count=event_count,
        complete_event_count=complete_event_count,
        total_trace_window_ms=total_trace_window_ms,
        top_cpu_ops=_top_rows(buckets["cpu_op"], top_k),
        top_cuda_kernels=_top_rows(buckets["cuda_kernel"], top_k),
        top_cuda_runtime=_top_rows(buckets["cuda_runtime"], top_k),
        top_user_annotations=_top_rows(buckets["user_annotation"], top_k),
        top_memory_events=_top_rows(buckets["memory"], top_k),
        top_all_events=_top_rows(buckets["all"], top_k),
        top_kernel_gaps=top_kernel_gaps,
        signals=[],
    )
    summary.signals = _build_signals(summary)
    return summary


def write_outputs(summary: TraceRunSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "phase2c_trace_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(summary), fp, indent=2)
        fp.write("\n")
    (output_dir / "phase2c_trace_summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def render_markdown(summary: TraceRunSummary) -> str:
    lines = [f"# Qwen3-Omni Phase 2C Torch Trace Summary: {summary.label}", ""]
    lines.extend(
        _table(
            [
                "traces",
                "events",
                "complete_events",
                "trace_window_ms",
            ],
            [
                [
                    len(summary.traces),
                    summary.event_count,
                    summary.complete_event_count,
                    summary.total_trace_window_ms,
                ]
            ],
        )
    )
    if summary.signals:
        lines.extend(["", "## Signals"])
        lines.extend(f"- {signal}" for signal in summary.signals)
    lines.extend(["", "## Trace Files"])
    lines.extend(f"- `{trace}`" for trace in summary.traces)
    for title, rows in [
        ("Top CPU Ops", summary.top_cpu_ops),
        ("Top CUDA Kernels", summary.top_cuda_kernels),
        ("Top CUDA Runtime Calls", summary.top_cuda_runtime),
        ("Top User Annotations", summary.top_user_annotations),
        ("Top Memory Events", summary.top_memory_events),
        ("Top All Complete Events", summary.top_all_events),
    ]:
        lines.extend(["", f"## {title}"])
        lines.extend(_aggregate_table(rows))
    lines.extend(["", "## Top Kernel Gaps"])
    lines.extend(
        _table(
            ["gap_ms", "start_ms", "pid", "tid", "prev", "next", "trace"],
            [
                [
                    row.gap_ms,
                    row.start_ms,
                    row.pid,
                    row.tid,
                    row.prev_name,
                    row.next_name,
                    Path(row.trace).name,
                ]
                for row in summary.top_kernel_gaps
            ],
        )
    )
    return "\n".join(lines) + "\n"


def _aggregate_table(rows: list[AggregateRow]) -> list[str]:
    return _table(
        [
            "name",
            "category",
            "count",
            "total_ms",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "max_ms",
            "files",
        ],
        [
            [
                row.name,
                row.category,
                row.count,
                row.total_ms,
                row.avg_ms,
                row.p50_ms,
                row.p95_ms,
                row.max_ms,
                len(row.files),
            ]
            for row in rows
        ],
    )


def _top_rows(
    bucket: dict[str, list[tuple[float, str]]],
    top_k: int,
) -> list[AggregateRow]:
    rows: list[AggregateRow] = []
    for compound_key, values in bucket.items():
        category, name = compound_key.split("\t", 1)
        durations = sorted(duration for duration, _ in values)
        total_ms = sum(durations) / 1000.0
        count = len(durations)
        rows.append(
            AggregateRow(
                name=name,
                category=category,
                count=count,
                total_ms=total_ms,
                avg_ms=total_ms / count if count else 0.0,
                p50_ms=_percentile(durations, 0.50) / 1000.0,
                p95_ms=_percentile(durations, 0.95) / 1000.0,
                max_ms=max(durations) / 1000.0,
                files=sorted({path for _, path in values}),
            )
        )
    rows.sort(key=lambda row: row.total_ms, reverse=True)
    return rows[:top_k]


def _summarize_kernel_gaps(
    intervals_by_thread: dict[
        tuple[str, int | str, int | str], list[tuple[float, float, str]]
    ],
    *,
    top_k: int,
    gap_threshold_us: float,
) -> list[GapRow]:
    gaps: list[GapRow] = []
    for (trace, pid, tid), intervals in intervals_by_thread.items():
        intervals.sort(key=lambda item: item[0])
        prev_end: float | None = None
        prev_name = ""
        for start_us, end_us, name in intervals:
            if prev_end is not None:
                gap_us = start_us - prev_end
                if gap_us >= gap_threshold_us:
                    gaps.append(
                        GapRow(
                            trace=trace,
                            pid=pid,
                            tid=tid,
                            start_ms=start_us / 1000.0,
                            gap_ms=gap_us / 1000.0,
                            prev_name=prev_name,
                            next_name=name,
                        )
                    )
            if prev_end is None or end_us >= prev_end:
                prev_end = end_us
                prev_name = name
    gaps.sort(key=lambda row: row.gap_ms, reverse=True)
    return gaps[:top_k]


def _build_signals(summary: TraceRunSummary) -> list[str]:
    signals: list[str] = []
    if not summary.top_cuda_kernels:
        signals.append("no CUDA kernel complete events found")
    if not summary.top_cpu_ops:
        signals.append("no torch CPU op complete events found")
    if summary.top_cuda_runtime:
        top_runtime = summary.top_cuda_runtime[0]
        if top_runtime.total_ms >= 100.0:
            signals.append(
                "large CUDA runtime total "
                f"{top_runtime.name} total_ms={top_runtime.total_ms:.3f}"
            )
    if summary.top_kernel_gaps:
        gap = summary.top_kernel_gaps[0]
        signals.append(
            "largest same-thread kernel gap "
            f"{gap.gap_ms:.3f} ms before {gap.next_name}"
        )
    for row in summary.top_cpu_ops[:5]:
        if row.avg_ms >= 10.0:
            signals.append(f"slow CPU op avg {row.name} avg_ms={row.avg_ms:.3f}")
    return signals


def _append_duration(
    bucket: dict[str, list[tuple[float, str]]],
    name: str,
    category: str,
    dur_us: float,
    trace_path: Path,
) -> None:
    compound_key = f"{category}\t{name}"
    bucket.setdefault(compound_key, []).append((dur_us, str(trace_path)))


def _classify_event(event: dict[str, Any]) -> str:
    name = str(event.get("name") or "")
    cat = str(event.get("cat") or "").lower()
    lower_name = name.lower()
    args = event.get("args") or {}
    if "kernel" in cat or cat in {"gpu_kernel", "cuda_kernel"}:
        return "cuda_kernel"
    if "cuda_runtime" in cat or lower_name.startswith("cuda"):
        return "cuda_runtime"
    if "cpu_op" in cat or "operator" in cat:
        return "cpu_op"
    if "user_annotation" in cat or "record_function" in cat:
        return "user_annotation"
    if "memory" in cat or lower_name.startswith("[memory]"):
        return "memory"
    if isinstance(args, dict):
        if "External id" in args or "Input Dims" in args or "Input type" in args:
            return "cpu_op"
    return "other"


def _normalize_name(name: str) -> str:
    name = " ".join(name.strip().split())
    if len(name) <= 180:
        return name
    return name[:177] + "..."


def _iter_trace_events(trace: Any) -> Iterable[dict[str, Any]]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents") or []
    elif isinstance(trace, list):
        events = trace
    else:
        events = []
    for event in events:
        if isinstance(event, dict):
            yield event


def _load_trace(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            return json.load(fp)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _is_trace_file(path: Path) -> bool:
    name = path.name
    return any(name.endswith(suffix) for suffix in TRACE_SUFFIXES)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(math.ceil(percentile * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["(empty)"]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return out


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _default_label(paths: list[Path]) -> str:
    if len(paths) == 1:
        path = paths[0].expanduser()
        if path.is_dir():
            return path.resolve().name
        return path.resolve().parent.name
    return "phase2c-traces"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize torch-profiler Chrome traces for Qwen3-Omni Phase 2C."
    )
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="Trace files or directories containing *.trace.json[.gz].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_omni_phase2c_trace_summary"),
        help="Directory for phase2c_trace_summary.json/md.",
    )
    parser.add_argument("--label", default=None, help="Report label.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--gap-threshold-us",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_US,
        help="Minimum same-thread CUDA kernel gap to include.",
    )
    args = parser.parse_args()

    traces = resolve_trace_paths(args.traces)
    if not traces:
        raise SystemExit("no torch trace files found")
    label = args.label or _default_label(args.traces)
    summary = summarize_traces(
        traces,
        label=label,
        top_k=args.top_k,
        gap_threshold_us=args.gap_threshold_us,
    )
    write_outputs(summary, args.output_dir)
    print(f"Wrote {args.output_dir / 'phase2c_trace_summary.json'}")
    print(f"Wrote {args.output_dir / 'phase2c_trace_summary.md'}")


if __name__ == "__main__":
    main()
