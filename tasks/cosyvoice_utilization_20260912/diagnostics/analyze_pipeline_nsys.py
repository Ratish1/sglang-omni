#!/usr/bin/env python3
"""Read-only Nsight SQLite stage timing; standard library, integer nanoseconds.

First use --catalog. Analysis requires explicit device/start/end. Kernel coverage
is NOT SM activity. GPU attribution uses process-scoped CUDA launch correlation
and containing host NVTX ranges, never GPU-time containment or kernel-name guesses.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PREFIX = "omni.pipeline:"
KERNEL_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")


def union(intervals):
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            out.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(start, end, lo, hi):
    return max(start, lo), min(end, hi)


def request_timings(marks):
    grouped = defaultdict(list)
    for event in marks:
        if event.get("request_id"):
            grouped[event["request_id"]].append(event)
    out = {}
    pairs = {
        "http_validation_ns": ("received", "validated"),
        "admission_to_terminal_ns": ("request_admission", "terminal_response"),
        "http_to_first_pcm_yield_ns": ("received", "first_pcm_yield"),
        "http_to_buffered_body_ready_ns": ("received", "buffered_body_ready"),
        "http_to_stream_body_complete_ns": ("received", "stream_body_complete"),
    }
    for rid, events in grouped.items():
        events.sort(key=lambda e: e["start_ns"])
        timestamps = defaultdict(list)
        for event in events:
            timestamps[event["op"]].append(event["start_ns"])
        timings = {}
        for label, (start, end) in pairs.items():
            if (
                len(timestamps[start]) == len(timestamps[end]) == 1
                and timestamps[end][0] >= timestamps[start][0]
            ):
                timings[label] = timestamps[end][0] - timestamps[start][0]
            else:
                timings[label] = None
        out[rid] = {"durations": timings, "events": events}
    return out


def rows(db, table):
    # Names are obtained only from sqlite_master, never interpolated user input.
    return [
        dict(r) for r in db.execute('SELECT * FROM "' + table.replace('"', '""') + '"')
    ]


def process_id(global_tid):
    # Nsight GlobalId packs hardware/vm/process/thread; retain all upper bits.
    return int(global_tid) & ~((1 << 24) - 1)


def load_annotations(db, tables, strings):
    ranges, marks = [], []
    ignored = Counter()
    if "NVTX_EVENTS" not in tables:
        return ranges, marks, {"missing_NVTX_EVENTS": 1}
    for row in rows(db, "NVTX_EVENTS"):
        message = row.get("text") or strings.get(row.get("textId"), "")
        if not message.startswith(PREFIX):
            continue
        try:
            meta = json.loads(message[len(PREFIX) :])
        except (ValueError, TypeError):
            ignored["invalid_metadata"] += 1
            continue
        if not isinstance(meta, dict) or "stage" not in meta or "op" not in meta:
            ignored["invalid_metadata"] += 1
            continue
        start, end, tid = row.get("start"), row.get("end"), row.get("globalTid")
        if start is None or tid is None:
            ignored["missing_start_or_thread"] += 1
            continue
        item = {"start_ns": int(start), "end_ns": end, "global_tid": tid, **meta}
        if meta.get("kind") == "mark":
            marks.append(item)
            continue
        if end is None or end == start:
            ignored["open_or_zero_range"] += 1
            continue
        if end < start or row.get("endGlobalTid") not in (None, 0, tid):
            ignored["invalid_or_cross_thread_range"] += 1
            continue
        ranges.append(item)
    return ranges, marks, dict(ignored)


class HostRanges:
    def __init__(self, ranges):
        self.by_thread = defaultdict(list)
        for r in ranges:
            self.by_thread[r["global_tid"]].append(r)
        self.starts, self.max_ends = {}, {}
        for tid, group in self.by_thread.items():
            group.sort(key=lambda r: (r["start_ns"], -r["end_ns"]))
            self.starts[tid] = [r["start_ns"] for r in group]
            maximum = -1
            self.max_ends[tid] = []
            for r in group:
                maximum = max(maximum, r["end_ns"])
                self.max_ends[tid].append(maximum)

    def containing(self, tid, start, end):
        group = self.by_thread.get(tid, [])
        i = bisect_right(self.starts.get(tid, []), start) - 1
        matches = []
        while i >= 0 and self.max_ends[tid][i] >= end:
            r = group[i]
            if r["end_ns"] >= end:
                matches.append(r)
            i -= 1
        return sorted(matches, key=lambda r: (r["start_ns"], -r["end_ns"]))


def analyze(db, args):
    tables = {
        r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    strings = (
        {r["id"]: r["value"] for r in rows(db, "StringIds")}
        if "StringIds" in tables
        else {}
    )
    populated = [
        t
        for t in KERNEL_TABLES
        if t in tables and db.execute(f'SELECT 1 FROM "{t}" LIMIT 1').fetchone()
    ]
    ranges, marks, ignored = load_annotations(db, tables, strings)
    catalog = {
        "sqlite": str(args.sqlite.resolve()),
        "kernel_tables": populated,
        "gpu_identity": (
            rows(db, "TARGET_INFO_GPU") if "TARGET_INFO_GPU" in tables else []
        ),
        "metrics": (
            rows(db, "TARGET_INFO_GPU_METRICS")
            if "TARGET_INFO_GPU_METRICS" in tables
            else []
        ),
        "annotation_counts": {
            "ranges": len(ranges),
            "marks": len(marks),
            "ignored": ignored,
        },
        "point_events": marks,
        "graph_trace_rows": (
            db.execute(
                "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            ).fetchone()[0]
            if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables
            else 0
        ),
    }
    catalog["kernel_extents"] = {
        t: [
            dict(r)
            for r in db.execute(
                f'SELECT deviceId, min(start) AS start_ns, max(end) AS end_ns, count(*) AS kernels FROM "{t}" GROUP BY deviceId'
            )
        ]
        for t in populated
    }
    if args.catalog:
        return catalog
    if (
        args.start_ns is None
        or args.end_ns is None
        or args.device is None
        or args.start_ns >= args.end_ns
    ):
        raise ValueError(
            "Analysis requires --device and --start-ns < --end-ns; inspect --catalog first"
        )
    table = args.kernel_table or (populated[0] if len(populated) == 1 else None)
    if table not in populated:
        raise ValueError(
            f"Select exactly one populated --kernel-table from {populated}"
        )
    required = {"start", "end", "deviceId", "globalPid", "correlationId"}
    columns = {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}
    if not required <= columns:
        raise ValueError(
            f"Unsupported kernel schema: missing {sorted(required - columns)}"
        )
    lo, hi = args.start_ns, args.end_ns
    runtimes = defaultdict(list)
    all_apis = []
    for source in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if source not in tables:
            continue
        for r in rows(db, source):
            if any(
                r.get(k) is None for k in ("start", "end", "globalTid", "correlationId")
            ):
                continue
            r["source"] = source
            r["name"] = strings.get(r.get("nameId"), "unknown")
            runtimes[(process_id(r["globalTid"]), r["correlationId"], source)].append(r)
            all_apis.append(r)
    index = HostRanges(ranges)
    by_stage, all_intervals = defaultdict(list), []
    attributed, counters, kernels = [], Counter(), []
    query = (
        f'SELECT * FROM "{table}" WHERE deviceId=? AND start<? AND end>? ORDER BY start'
    )
    for row in db.execute(query, (args.device, hi, lo)):
        k = dict(row)
        if k["end"] <= k["start"]:
            counters["invalid_intervals"] += 1
            continue
        interval = clip(k["start"], k["end"], lo, hi)
        all_intervals.append(interval)
        candidates = []
        for source in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            candidates = [
                r
                for r in runtimes.get((k["globalPid"], k["correlationId"], source), [])
                if r["start"] <= k["start"]
            ]
            if candidates:
                break
        chain = []
        reason = "missing_launch"
        if len(candidates) == 1:
            api = candidates[0]
            chain = index.containing(api["globalTid"], api["start"], api["end"])
            reason = "attributed" if chain else "launch_without_annotation"
        elif len(candidates) > 1:
            reason = "ambiguous_launch"
        # Intersecting non-nested host ranges cannot establish a unique owner.
        if any(
            a["end_ns"] < b["end_ns"]
            or (
                (a["start_ns"], a["end_ns"]) == (b["start_ns"], b["end_ns"])
                and a["stage"] != b["stage"]
            )
            for a, b in zip(chain, chain[1:])
        ):
            chain, reason = [], "ambiguous_annotation"
        stage = chain[-1]["stage"] if chain else "unattributed"
        by_stage[stage].append(interval)
        counters[reason] += 1
        kernel = {
            "start_ns": interval[0],
            "end_ns": interval[1],
            "stage": stage,
            "global_pid": k["globalPid"],
            "stream_id": k.get("streamId"),
            "correlation_id": k["correlationId"],
            "graph_id": k.get("graphId"),
            "graph_node_id": k.get("graphNodeId"),
            "name": strings.get(
                k.get("demangledName"), strings.get(k.get("shortName"), "unknown")
            ),
            "attribution": reason,
        }
        if args.kernel_details:
            kernel["host_scopes"] = chain
            kernels.append(kernel)
        if chain:
            attributed.append(interval)
    merged = union(all_intervals)
    stage_unions = {s: union(v) for s, v in by_stage.items()}
    gaps, cursor = [], lo
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < hi:
        gaps.append((cursor, hi))
    largest = []
    for start, end in sorted(gaps, key=lambda p: p[1] - p[0], reverse=True)[
        : args.top_gaps
    ]:
        host = [r for r in ranges if r["start_ns"] < end and r["end_ns"] > start]
        api = [
            {
                "name": r["name"],
                "global_tid": r["globalTid"],
                "overlap_ns": min(end, r["end"]) - max(start, r["start"]),
            }
            for r in all_apis
            if r["start"] < end and r["end"] > start
        ]
        largest.append(
            {
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "overlapping_host_scopes": host,
                "overlapping_cuda_apis": sorted(
                    api, key=lambda r: r["overlap_ns"], reverse=True
                )[:20],
            }
        )
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for r in rows(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            if r.get("deviceId") == args.device and r["start"] < hi and r["end"] > lo:
                copies.append(clip(r["start"], r["end"], lo, hi))
    metrics = []
    if (
        args.metric_type is not None
        and args.metric_id is not None
        and "GPU_METRICS" in tables
    ):
        metrics = [
            dict(r)
            for r in db.execute(
                "SELECT timestamp, value FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
                (args.metric_type, args.metric_id, lo, hi),
            )
        ]
    return {
        "schema_version": 1,
        "catalog": catalog,
        "window": {
            "device_id": args.device,
            "start_ns": lo,
            "end_ns": hi,
            "duration_ns": hi - lo,
        },
        "kernel_table": table,
        "kernel_count": len(all_intervals),
        "kernel_coverage_percent": 100 * duration(merged) / (hi - lo),
        "kernel_union_ns": duration(merged),
        "summed_kernel_ns": duration(all_intervals),
        "uncovered_ns": hi - lo - duration(merged),
        "attribution_counts": dict(counters),
        "attributed_union_ns": duration(union(attributed)),
        "stages": {
            s: {
                "kernel_count": len(by_stage[s]),
                "union_ns": duration(v),
                "exclusive_ns": duration(v)
                - duration(
                    intersection(
                        v,
                        union(
                            [
                                p
                                for other, ps in stage_unions.items()
                                if other != s
                                for p in ps
                            ]
                        ),
                    )
                ),
            }
            for s, v in stage_unions.items()
        },
        "pairwise_overlap_ns": {
            f"{a}|{b}": duration(intersection(stage_unions[a], stage_unions[b]))
            for a, b in combinations(sorted(stage_unions), 2)
        },
        "memcpy_union_ns": duration(union(copies)),
        "compute_or_copy_union_ns": duration(union(all_intervals + copies)),
        "gap_count": len(gaps),
        "largest_gaps": largest,
        "host_ranges": [r for r in ranges if r["start_ns"] < hi and r["end_ns"] > lo],
        "request_lifecycles": request_timings(
            [r for r in marks if lo <= r["start_ns"] < hi]
        ),
        "realized_batches": {
            stage
            + "/"
            + op: dict(
                Counter(
                    str(r["batch_size"])
                    for r in ranges
                    if r["stage"] == stage
                    and r["op"] == op
                    and "batch_size" in r
                    and lo <= r["start_ns"] < hi
                )
            )
            for stage, op in sorted(
                {(r["stage"], r["op"]) for r in ranges if "batch_size" in r}
            )
        },
        "metric_selection": {"type_id": args.metric_type, "metric_id": args.metric_id},
        "metric_samples_raw": metrics,
        "kernels": kernels,
        "limitations": [
            "Kernel coverage is not SM utilization or occupancy.",
            "All captured processes on this device contribute to coverage; inspect process attribution for unrelated work.",
            "Graph nodes require --cuda-graph-trace=node; absent node records can undercount compute. Check the capture command and dropped-event warnings.",
            "Correlation ambiguity remains unattributed; no kernel-name inference.",
            "Host range duration includes enqueue, Python work and waits; overlapping a gap is evidence, not proof of its cause.",
            "Metric values/units are preserved; select the correct GPU metric type and metric ID from catalog. No automatic SM metric substitution.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--kernel-table", choices=KERNEL_TABLES)
    parser.add_argument("--kernel-details", action="store_true")
    parser.add_argument("--top-gaps", type=int, default=20)
    parser.add_argument("--metric-type", type=int)
    parser.add_argument("--metric-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_gaps < 0:
        parser.error("--top-gaps must be nonnegative")
    if not args.sqlite.is_file():
        parser.error("SQLite input does not exist")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh artifact path")
    if (args.metric_type is None) != (args.metric_id is None):
        parser.error("--metric-type and --metric-id must be supplied together")
    with sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        result = analyze(db, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
