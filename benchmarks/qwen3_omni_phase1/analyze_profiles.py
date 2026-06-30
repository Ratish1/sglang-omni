# SPDX-License-Identifier: Apache-2.0
"""Build a comparable Phase 1 matrix from request-profiler reports.

The input can be a profiler run directory, a ``summary/`` directory, or a
``report.json`` path. The output is a compact JSON/Markdown matrix that ranks
stage, hop, memory, and CUDA graph signals across modality runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TOP_K = 12
LARGE_TRANSFER_BYTES = 64 * 1024 * 1024


@dataclass
class MatrixRun:
    label: str
    report_path: str
    request_count: int
    admin_event_count: int
    hop_row_count: int
    transports: list[str]
    memory_snapshot_count: int
    cuda_graph_audit_count: int
    top_stages: list[dict[str, Any]] = field(default_factory=list)
    top_hops: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    cuda_graph_audits: list[dict[str, Any]] = field(default_factory=list)
    bottleneck_signals: list[str] = field(default_factory=list)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_report_path(path: Path) -> Path:
    path = path.expanduser()
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path / "summary" / "report.json",
                path / "report.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not find report.json under {path}")


def summarize_report(path: Path, *, label: str | None = None, top_k: int) -> MatrixRun:
    report_path = resolve_report_path(path)
    report = load_report(report_path)
    run_label = label or _default_label(path, report_path)

    stage_rows = sorted(
        report.get("stage_breakdown", []),
        key=lambda row: float(row.get("total_ms") or 0.0),
        reverse=True,
    )[:top_k]
    hop_rows = sorted(
        report.get("hop_breakdown", []),
        key=lambda row: float(row.get("total_ms") or 0.0),
        reverse=True,
    )
    top_hop_rows = hop_rows[:top_k]
    memory_snapshots = _collect_memory_snapshots(report)
    cuda_graph_audits = [
        event
        for event in report.get("admin_events", [])
        if event.get("event_name") == "stage_cuda_graph_audit"
    ]
    transports = sorted(
        {
            str(row.get("transport"))
            for row in report.get("hop_breakdown", [])
            if row.get("transport")
        }
    )
    summary = MatrixRun(
        label=run_label,
        report_path=str(report_path),
        request_count=int(report.get("request_count") or 0),
        admin_event_count=len(report.get("admin_events", [])),
        hop_row_count=len(report.get("hop_breakdown", [])),
        transports=transports,
        memory_snapshot_count=len(memory_snapshots),
        cuda_graph_audit_count=len(cuda_graph_audits),
        top_stages=stage_rows,
        top_hops=top_hop_rows,
        memory_snapshots=memory_snapshots,
        cuda_graph_audits=cuda_graph_audits,
    )
    summary.bottleneck_signals = _build_bottleneck_signals(summary, hop_rows)
    return summary


def write_outputs(runs: list[MatrixRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = {"runs": [asdict(run) for run in runs]}
    with (output_dir / "phase1_matrix.json").open("w", encoding="utf-8") as fp:
        json.dump(matrix, fp, indent=2)
        fp.write("\n")
    (output_dir / "phase1_matrix.md").write_text(
        render_markdown(runs), encoding="utf-8"
    )


def render_markdown(runs: list[MatrixRun]) -> str:
    lines = ["# Qwen3-Omni Phase 1 Profiling Matrix", ""]
    lines.extend(
        _table(
            [
                "run",
                "requests",
                "admin_events",
                "hop_rows",
                "transports",
                "memory_snapshots",
                "cuda_audits",
            ],
            [
                [
                    run.label,
                    run.request_count,
                    run.admin_event_count,
                    run.hop_row_count,
                    ", ".join(run.transports) or "-",
                    run.memory_snapshot_count,
                    run.cuda_graph_audit_count,
                ]
                for run in runs
            ],
        )
    )
    for run in runs:
        lines.extend(["", f"## {run.label}", ""])
        if run.bottleneck_signals:
            lines.append("### Signals")
            lines.extend(f"- {signal}" for signal in run.bottleneck_signals)
            lines.append("")
        lines.append("### Top Stages")
        lines.extend(
            _table(
                ["stage", "interval", "count", "total_ms", "avg_ms", "p95_ms"],
                [
                    [
                        row.get("stage", "-"),
                        row.get("interval", "-"),
                        row.get("count", 0),
                        row.get("total_ms", 0),
                        row.get("avg_ms", 0),
                        row.get("p95_ms", 0),
                    ]
                    for row in run.top_stages
                ],
            )
        )
        lines.extend(["", "### Top Hops"])
        lines.extend(
            _table(
                [
                    "src",
                    "dst",
                    "kind",
                    "transport",
                    "modality",
                    "count",
                    "total_ms",
                    "avg_ms",
                    "p95_ms",
                    "total_bytes",
                ],
                [
                    [
                        row.get("src", "-"),
                        row.get("dst", "-"),
                        row.get("kind", "-"),
                        row.get("transport", "-"),
                        row.get("modality") or "-",
                        row.get("count", 0),
                        row.get("total_ms", 0),
                        row.get("avg_ms", 0),
                        row.get("p95_ms", 0),
                        row.get("total_bytes") or 0,
                    ]
                    for row in run.top_hops
                ],
            )
        )
    return "\n".join(lines) + "\n"


def _collect_memory_snapshots(report: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for request_id, events in (report.get("timelines") or {}).items():
        for event in events:
            if event.get("event_name") != "scheduler_request_memory_snapshot":
                continue
            metadata = event.get("metadata") or {}
            snapshots.append(
                {
                    "request_id": request_id,
                    "stage": event.get("stage"),
                    "phase": metadata.get("phase"),
                    "input_tokens": metadata.get("input_tokens"),
                    "max_new_tokens": metadata.get("max_new_tokens"),
                    "required_tokens": metadata.get("required_tokens"),
                    "kv_capacity": metadata.get("kv_capacity"),
                    "kv_available_tokens": metadata.get("kv_available_tokens"),
                    "process_gpu_memory_fraction": metadata.get(
                        "process_gpu_memory_fraction"
                    ),
                    "mem_fraction_static": metadata.get("mem_fraction_static"),
                }
            )
    return snapshots


def _build_bottleneck_signals(
    run: MatrixRun, hop_rows: list[dict[str, Any]]
) -> list[str]:
    signals: list[str] = []
    if not run.cuda_graph_audits:
        signals.append("missing CUDA graph audit events")
    if not run.memory_snapshots:
        signals.append("missing scheduler memory snapshots")
    for row in hop_rows:
        total_bytes = int(row.get("total_bytes") or 0)
        if total_bytes >= LARGE_TRANSFER_BYTES:
            signals.append(
                "large transfer "
                f"{row.get('src')}->{row.get('dst')} {row.get('kind')} "
                f"{row.get('transport')} bytes={total_bytes}"
            )
    for snapshot in run.memory_snapshots:
        required_tokens = snapshot.get("required_tokens")
        kv_capacity = snapshot.get("kv_capacity")
        if isinstance(required_tokens, int) and isinstance(kv_capacity, int):
            if kv_capacity > 0 and required_tokens / kv_capacity >= 0.90:
                signals.append(
                    "near-context admission "
                    f"{snapshot.get('stage')} request={snapshot.get('request_id')} "
                    f"required={required_tokens} capacity={kv_capacity}"
                )
    for audit_event in run.cuda_graph_audits:
        metadata = audit_event.get("metadata") or {}
        if metadata.get("is_valid") is False:
            signals.append(f"invalid CUDA graph audit stage={audit_event.get('stage')}")
    return signals


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
    return str(value).replace("\n", " ")


def _default_label(input_path: Path, report_path: Path) -> str:
    if input_path.is_file():
        parent = report_path.parent
        return parent.parent.name if parent.name == "summary" else parent.name
    resolved = input_path.expanduser().resolve()
    return resolved.parent.name if resolved.name == "summary" else resolved.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Phase 1 comparison matrix from profiler report.json files."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Profiler run dirs, summary dirs, or report.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_omni_phase1_matrix"),
        help="Directory for phase1_matrix.json and phase1_matrix.md.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    runs = [summarize_report(path, top_k=args.top_k) for path in args.reports]
    write_outputs(runs, args.output_dir)
    print(f"Wrote {args.output_dir / 'phase1_matrix.json'}")
    print(f"Wrote {args.output_dir / 'phase1_matrix.md'}")


if __name__ == "__main__":
    main()
