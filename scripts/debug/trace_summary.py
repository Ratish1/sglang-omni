# SPDX-License-Identifier: Apache-2.0
"""Summarize PyTorch Chrome traces for focused MOSS profiling.

This is intentionally dependency-free. It handles ``.json`` and ``.json.gz``
Chrome traces exported by ``torch.profiler`` and writes a compact text/JSON
summary plus a Perfetto SQL helper file.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import quantiles
from typing import Any

_CUDA_RUNTIME_NAMES = {
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaGraphLaunch",
    "cudaLaunchKernel",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
}

_DEFAULT_LABELS = (
    "moss_tts_local",
    "omni_scheduler",
    "omni_model_runner",
    "vocoder",
    "decode_audio_codes",
    "audio_tokenizer",
    "feedback_write",
    "radix_hash",
    "cudaGraphLaunch",
    "cudaStreamSynchronize",
)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        data = json.load(fp)
    events = data.get("traceEvents", data)
    if not isinstance(events, list):
        raise ValueError(f"{path} does not look like a Chrome trace")
    return [ev for ev in events if isinstance(ev, dict)]


def _duration_ms(ev: dict[str, Any]) -> float:
    return float(ev.get("dur") or 0.0) / 1000.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return quantiles(sorted(values), n=100, method="inclusive")[94]


def _summarize_by_name(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        dur = _duration_ms(ev)
        if dur <= 0:
            continue
        name = str(ev.get("name") or "")
        if name:
            durations[name].append(dur)
    rows = []
    for name, vals in durations.items():
        total = sum(vals)
        rows.append(
            {
                "name": name,
                "count": len(vals),
                "total_ms": total,
                "avg_ms": total / len(vals),
                "p95_ms": _p95(vals),
                "max_ms": max(vals),
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


def _label_rows(
    rows: list[dict[str, Any]], labels: list[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped = {}
    for label in labels:
        grouped[label] = [row for row in rows if label in row["name"]][:30]
    return grouped


def _sync_contexts(
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Find the smallest same-thread CPU event containing top sync calls."""
    syncs = [
        ev
        for ev in events
        if ev.get("ph") == "X"
        and str(ev.get("name") or "")
        in {"cudaStreamSynchronize", "cudaDeviceSynchronize"}
        and _duration_ms(ev) > 0
    ]
    syncs.sort(key=_duration_ms, reverse=True)
    syncs = syncs[:limit]

    by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev or "ts" not in ev:
            continue
        name = str(ev.get("name") or "")
        if name.startswith("cuda"):
            continue
        by_thread[(ev.get("pid"), ev.get("tid"))].append(ev)

    starts_by_thread: dict[tuple[Any, Any], list[float]] = {}
    for key, group in by_thread.items():
        group.sort(key=lambda ev: float(ev.get("ts") or 0.0))
        starts_by_thread[key] = [float(ev.get("ts") or 0.0) for ev in group]

    contexts = []
    for sync in syncs:
        key = (sync.get("pid"), sync.get("tid"))
        group = by_thread.get(key, [])
        starts = starts_by_thread.get(key, [])
        sync_start = float(sync.get("ts") or 0.0)
        sync_end = sync_start + float(sync.get("dur") or 0.0)
        idx = bisect.bisect_right(starts, sync_start)
        best = None
        best_dur = float("inf")
        # Search backward from the sync start. A containing parent normally
        # starts close to the sync; cap the scan to avoid quadratic behavior on
        # very large traces.
        for ev in reversed(group[max(0, idx - 5000) : idx]):
            start = float(ev.get("ts") or 0.0)
            end = start + float(ev.get("dur") or 0.0)
            if start <= sync_start and end >= sync_end:
                dur = end - start
                if dur < best_dur:
                    best = ev
                    best_dur = dur
        contexts.append(
            {
                "sync_name": sync.get("name"),
                "sync_ms": _duration_ms(sync),
                "pid": sync.get("pid"),
                "tid": sync.get("tid"),
                "parent_name": None if best is None else best.get("name"),
                "parent_ms": None if best is None else _duration_ms(best),
            }
        )
    return contexts


def _write_text(
    path: Path,
    *,
    top_rows: list[dict[str, Any]],
    cuda_rows: list[dict[str, Any]],
    label_rows: dict[str, list[dict[str, Any]]],
    sync_contexts: list[dict[str, Any]],
) -> None:
    def line(row: dict[str, Any]) -> str:
        return (
            f"{row['name']:<80} count={row['count']:<8} "
            f"total={row['total_ms']:.3f}ms avg={row['avg_ms']:.6f}ms "
            f"p95={row['p95_ms']:.6f}ms max={row['max_ms']:.6f}ms"
        )

    chunks = ["# Trace Summary", "", "## Top Events"]
    chunks.extend(line(row) for row in top_rows[:50])
    chunks.extend(["", "## CUDA Runtime"])
    chunks.extend(line(row) for row in cuda_rows)
    chunks.extend(["", "## Label Matches"])
    for label, rows in label_rows.items():
        chunks.append("")
        chunks.append(f"### {label}")
        if not rows:
            chunks.append("(none)")
        else:
            chunks.extend(line(row) for row in rows[:20])
    chunks.extend(["", "## Top Sync Contexts"])
    for ctx in sync_contexts:
        chunks.append(
            f"{ctx['sync_name']} {ctx['sync_ms']:.3f}ms "
            f"pid={ctx['pid']} tid={ctx['tid']} parent={ctx['parent_name']} "
            f"parent_ms={ctx['parent_ms']}"
        )
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _write_perfetto_sql(path: Path, labels: list[str]) -> None:
    label_filter = " OR ".join(f"name GLOB '*{label}*'" for label in labels)
    path.write_text(
        f"""-- Load the trace with trace_processor_shell or ui.perfetto.dev, then run:

-- Top CUDA runtime calls.
SELECT name, COUNT(*) AS count, SUM(dur)/1e6 AS total_ms,
       AVG(dur)/1e6 AS avg_ms, MAX(dur)/1e6 AS max_ms
FROM slice
WHERE name GLOB 'cuda*'
GROUP BY name
ORDER BY total_ms DESC
LIMIT 50;

-- MOSS / Omni / vocoder labels if present.
SELECT name, COUNT(*) AS count, SUM(dur)/1e6 AS total_ms,
       AVG(dur)/1e6 AS avg_ms, MAX(dur)/1e6 AS max_ms
FROM slice
WHERE {label_filter}
GROUP BY name
ORDER BY total_ms DESC
LIMIT 100;

-- Longest synchronizations.
SELECT ts/1e9 AS ts_s, dur/1e6 AS dur_ms, name, track_id
FROM slice
WHERE name IN ('cudaStreamSynchronize', 'cudaDeviceSynchronize')
ORDER BY dur DESC
LIMIT 100;

-- Immediate ancestor-style context for long syncs often needs visual
-- inspection because PyTorch Chrome traces do not always preserve all user
-- ranges as nested Perfetto slices.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--sync-context-limit", type=int, default=50)
    args = parser.parse_args()

    out_dir = args.out_dir or args.trace.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.label or list(_DEFAULT_LABELS)
    events = _load_trace(args.trace)
    rows = _summarize_by_name(events)
    cuda_rows = [
        row
        for row in rows
        if row["name"] in _CUDA_RUNTIME_NAMES or row["name"].startswith("cuda")
    ]
    sync_contexts = _sync_contexts(events, limit=args.sync_context_limit)
    labels_grouped = _label_rows(rows, labels)

    summary = {
        "trace": str(args.trace),
        "event_count": len(events),
        "top_events": rows[:100],
        "cuda_runtime": cuda_rows,
        "label_matches": labels_grouped,
        "sync_contexts": sync_contexts,
    }
    (out_dir / "trace_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_text(
        out_dir / "trace_summary.txt",
        top_rows=rows,
        cuda_rows=cuda_rows,
        label_rows=labels_grouped,
        sync_contexts=sync_contexts,
    )
    _write_perfetto_sql(out_dir / "perfetto_sync_queries.sql", labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
