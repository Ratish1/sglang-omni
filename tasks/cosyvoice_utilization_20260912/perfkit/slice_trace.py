#!/usr/bin/env python3
"""Slice one CosyVoice Nsight SQLite export into stage, thread and request ledgers.

Read-only, standard library, integer nanoseconds on the exported Nsight clock.
Every number is derived from recorded intervals: NVTX host ranges and marks
from the pipeline annotations, CUPTI kernel intervals joined to their launching
thread through (process, correlation id), and raw GPU metric samples.

The ledgers answer, per capture:

1. AR step ledger: for every ar/execute range, host wall time, GPU kernel union
   of the kernels that range launched, nested sampling and codec D2H host time,
   and the scheduler gap to the next step, grouped by forward mode and batch.
2. Vocoder thread budget: exclusive time of the innermost annotated activity on
   the vocoder thread (flow native, flow packed, hift, d2h, peer waits, glue,
   idle) and the GPU kernel union each activity launched.
3. Flow and HiFT call table: host duration versus launched GPU union per call,
   grouped by op, batch and frame bucket. A ratio well below one is launch bound.
4. Request hop timeline: for every streaming hop, when its tokens became ready
   (AR chunk sent), when the vocoder started it, how long it ran, and when the
   HTTP layer yielded PCM. Queueing delay is start minus ready.
5. GPU idle gaps attributed to the joint (AR thread state, vocoder thread state)
   at every instant, by exact interval intersection.
6. SM Active samples conditioned on which stage's kernels cover the sample time.

Nothing here is a benchmark result. Profiler-on timings describe the traced
cohort only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "omni.pipeline:"
PID_MASK = ~((1 << 24) - 1)
AR_FIRST_FLUSH = 28
AR_FOLLOWUP_FLUSH = 25
LOOKAHEAD = 3

VOCODER_ORDER = [
    ("d2h", lambda r: r["stage"] == "d2h"),
    ("hift", lambda r: r["stage"] == "hift"),
    ("flow_estimator", lambda r: r["stage"] == "flow" and r["op"] == "estimator"),
    ("flow_euler_glue", lambda r: r["stage"] == "flow" and r["op"] == "euler"),
    (
        "flow_packed_glue",
        lambda r: r["stage"] == "flow" and r["op"].startswith("packed"),
    ),
    ("flow_native", lambda r: r["stage"] == "flow" and r["op"] == "native"),
    (
        "peer_wait",
        lambda r: r["stage"] == "scheduler" and r["op"].endswith("peer_wait"),
    ),
    (
        "payload_collection",
        lambda r: r["stage"] == "scheduler" and r["op"] == "payload_collection",
    ),
    ("vocoder_glue", lambda r: r["stage"] == "vocoder"),
]


# ---------------------------------------------------------------- intervals


def union(intervals):
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def total(intervals):
    return sum(e - s for s, e in intervals)


def intersect(left, right):
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        s = max(left[i][0], right[j][0])
        e = min(left[i][1], right[j][1])
        if s < e:
            out.append((s, e))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(left, right):
    """left minus right; both are sorted disjoint unions."""
    out = []
    j = 0
    for s, e in left:
        cur = s
        while j < len(right) and right[j][1] <= cur:
            j += 1
        k = j
        while k < len(right) and right[k][0] < e:
            rs, re = right[k]
            if rs > cur:
                out.append((cur, rs))
            cur = max(cur, re)
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def clip(intervals, lo, hi):
    out = []
    for s, e in intervals:
        s2, e2 = max(s, lo), min(e, hi)
        if s2 < e2:
            out.append((s2, e2))
    return out


def gaps(intervals, lo, hi):
    out, cursor = [], lo
    for s, e in intervals:
        if s > cursor:
            out.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        out.append((cursor, hi))
    return out


def describe(values):
    if not values:
        return {"n": 0}
    values = sorted(values)

    def pct(q):
        i = (len(values) - 1) * q
        lo, hi = int(i), min(int(i) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)

    return {
        "n": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": values[-1],
    }


def ms(ns):
    return None if ns is None else round(ns / 1e6, 3)


# ---------------------------------------------------------------- loading


def load_annotations(db, strings):
    ranges, marks = [], []
    for start, end, tid, end_tid, text_id, text in db.execute(
        "SELECT start, end, globalTid, endGlobalTid, textId, text FROM NVTX_EVENTS"
    ):
        message = text or strings.get(text_id) or ""
        if not message.startswith(PREFIX):
            continue
        meta = json.loads(message[len(PREFIX) :])
        item = {"start": int(start), "end": end, "tid": tid, **meta}
        if meta.get("kind") == "mark":
            marks.append(item)
            continue
        if end is None or end <= start or end_tid not in (None, 0, tid):
            continue
        item["end"] = int(end)
        ranges.append(item)
    ranges.sort(key=lambda r: (r["start"], -r["end"]))
    marks.sort(key=lambda m: m["start"])
    return ranges, marks


class ThreadRanges:
    """Innermost-range lookup for host ranges on one thread."""

    def __init__(self, ranges):
        self.ranges = sorted(ranges, key=lambda r: (r["start"], -r["end"]))
        self.starts = [r["start"] for r in self.ranges]
        self.max_end = []
        m = -1
        for r in self.ranges:
            m = max(m, r["end"])
            self.max_end.append(m)

    def containing(self, t):
        i = bisect_right(self.starts, t) - 1
        best = None
        while i >= 0 and self.max_end[i] > t:
            r = self.ranges[i]
            if r["end"] > t and (
                best is None or r["end"] - r["start"] < best["end"] - best["start"]
            ):
                best = r
            i -= 1
        return best

    def inside(self, outer, stage=None, op=None):
        i = bisect_right(self.starts, outer["start"] - 1)
        out = []
        while i < len(self.ranges) and self.starts[i] < outer["end"]:
            r = self.ranges[i]
            if r is not outer and r["end"] <= outer["end"]:
                if (stage is None or r["stage"] == stage) and (
                    op is None or r["op"] == op
                ):
                    out.append(r)
            i += 1
        return out


def load_kernels(db, device, lo, hi):
    # One launch row per (process, correlation id). Graph-replayed node kernels
    # all carry the correlation id of their cudaGraphLaunch call, so the join
    # must be one-to-many from launch to kernels, never deduplicated by kernel.
    db.execute(
        "CREATE TEMP TABLE launches AS SELECT correlationId, (globalTid & ?) AS pid, "
        "min(globalTid) AS globalTid, min(start) AS start, max(end) AS end "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY correlationId, (globalTid & ?)",
        (PID_MASK, PID_MASK),
    )
    db.execute("CREATE INDEX temp.launch_corr ON launches(correlationId, pid)")
    query = (
        "SELECT k.start, k.end, k.correlationId, k.globalPid, k.graphNodeId, "
        "k.streamId, k.demangledName, l.globalTid, l.start, l.end "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN launches l "
        "ON l.correlationId = k.correlationId AND l.pid = k.globalPid "
        "WHERE k.deviceId = ? AND k.start < ? AND k.end > ? ORDER BY k.start"
    )
    kernels = []
    for row in db.execute(query, (device, hi, lo)):
        kernels.append(
            {
                "start": row[0],
                "end": row[1],
                "corr": row[2],
                "pid": row[3],
                "graph_node": row[4],
                "stream": row[5],
                "name": row[6],
                "launch_tid": row[7],
                "api_start": row[8],
                "api_end": row[9],
            }
        )
    return kernels


# ---------------------------------------------------------------- analysis


def pick_threads(ranges, marks):
    ar = Counter(
        r["tid"] for r in ranges if r["stage"] == "ar" and r["op"] == "execute"
    )
    voc = Counter(
        r["tid"] for r in ranges if r["stage"] in ("flow", "hift", "vocoder", "d2h")
    )
    prep = Counter(r["tid"] for r in ranges if r["stage"] == "preprocessing")
    http = Counter(m["tid"] for m in marks if m["stage"] == "http")
    return {
        "ar": ar.most_common(1)[0][0] if ar else None,
        "vocoder": voc.most_common(1)[0][0] if voc else None,
        "preprocessing": sorted(prep),
        "http": http.most_common(1)[0][0] if http else None,
    }


def default_window(marks):
    received = [
        m["start"] for m in marks if m["stage"] == "http" and m["op"] == "received"
    ]
    ends = [
        m["start"]
        for m in marks
        if (
            m["stage"] == "http"
            and m["op"] in ("stream_body_complete", "buffered_body_ready")
        )
        or (m["stage"] == "coordinator" and m["op"] == "terminal_response")
    ]
    if not received or not ends:
        raise SystemExit("no HTTP request markers; pass --start-ns/--end-ns")
    return min(received), max(ends) + 1


def kernels_by_thread(kernels):
    out = defaultdict(list)
    for k in kernels:
        out[k["launch_tid"]].append(k)
    for lst in out.values():
        lst.sort(key=lambda k: k["api_start"])
    return out


def kernels_launched_in(thread_kernels, api_starts, start, end):
    lo = bisect_right(api_starts, start - 1)
    hi = bisect_right(api_starts, end - 1)
    return thread_kernels[lo:hi]


def gpu_summary(ks, lo, hi):
    if not ks:
        return {"kernels": 0, "gpu_union_ns": 0, "gpu_sum_ns": 0, "gpu_last_end": None}
    u = union(clip([(k["start"], k["end"]) for k in ks], lo, hi))
    return {
        "kernels": len(ks),
        "gpu_union_ns": total(u),
        "gpu_sum_ns": sum(min(k["end"], hi) - max(k["start"], lo) for k in ks),
        "gpu_last_end": max(k["end"] for k in ks),
    }


def ar_ledger(ranges, marks, threads, tk, lo, hi):
    tid = threads["ar"]
    if tid is None:
        return {}
    thread = ThreadRanges([r for r in ranges if r["tid"] == tid])
    kernels = tk.get(tid, [])
    api_starts = [k["api_start"] for k in kernels]
    executes = [
        r
        for r in thread.ranges
        if r["stage"] == "ar" and r["op"] == "execute" and lo <= r["start"] < hi
    ]
    rows = []
    for i, ex in enumerate(executes):
        inner = thread.inside(ex, stage="ar")
        sampling = sum(r["end"] - r["start"] for r in inner if r["op"] == "sampling")
        d2h = sum(r["end"] - r["start"] for r in inner if r["op"] == "codec_ids_d2h")
        prefill = sum(r["end"] - r["start"] for r in inner if r["op"] == "prefill")
        ks = kernels_launched_in(kernels, api_starts, ex["start"], ex["end"])
        g = gpu_summary(ks, lo, hi)
        samp_ranges = [r for r in inner if r["op"] == "sampling"]
        samp_k = []
        for sr in samp_ranges:
            samp_k += kernels_launched_in(kernels, api_starts, sr["start"], sr["end"])
        sg = gpu_summary(samp_k, lo, hi)
        graph_nodes = sum(1 for k in ks if k["graph_node"])
        rows.append(
            {
                "start": ex["start"],
                "end": ex["end"],
                "wall_ns": ex["end"] - ex["start"],
                "mode": "prefill" if prefill else "decode",
                "batch": int(ex.get("batch_size", 0)),
                "sampling_host_ns": sampling,
                "d2h_host_ns": d2h,
                "prefill_host_ns": prefill,
                "gpu_kernels": g["kernels"],
                "gpu_graph_node_kernels": graph_nodes,
                "gpu_union_ns": g["gpu_union_ns"],
                "gpu_tail_ns": max(0, (g["gpu_last_end"] or ex["end"]) - ex["end"]),
                "sampling_gpu_union_ns": sg["gpu_union_ns"],
                "sampling_gpu_kernels": sg["kernels"],
                "gap_to_next_ns": (
                    (executes[i + 1]["start"] - ex["end"])
                    if i + 1 < len(executes)
                    else None
                ),
            }
        )
    # AR active periods: requests between queue enter and stage complete.
    per_req = defaultdict(dict)
    for m in marks:
        if m["stage"] == "tts_engine" and m.get("request_id"):
            per_req[m["request_id"]].setdefault(m["op"], m["start"])
    active = union(
        [
            (v["scheduler_queue_enter"], v.get("stage_complete", hi))
            for v in per_req.values()
            if "scheduler_queue_enter" in v
        ]
    )
    active = clip(active, lo, hi)
    exec_union = clip(union([(r["start"], r["end"]) for r in executes]), lo, hi)
    active_gap = subtract(active, exec_union)
    inactive = subtract([(lo, hi)], active)
    ar_gpu = union(clip([(k["start"], k["end"]) for k in kernels], lo, hi))
    groups = defaultdict(list)
    for r in rows:
        bucket = r["batch"] if r["batch"] <= 4 else (8 if r["batch"] <= 8 else 16)
        groups[(r["mode"], bucket)].append(r)
    table = []
    for (mode, bucket), rs in sorted(groups.items()):
        table.append(
            {
                "mode": mode,
                "batch_bucket": bucket,
                "steps": len(rs),
                "wall_ms": ms(statistics.fmean(r["wall_ns"] for r in rs)),
                "gpu_union_ms": ms(statistics.fmean(r["gpu_union_ns"] for r in rs)),
                "gpu_tail_ms": ms(statistics.fmean(r["gpu_tail_ns"] for r in rs)),
                "sampling_host_ms": ms(
                    statistics.fmean(r["sampling_host_ns"] for r in rs)
                ),
                "sampling_gpu_ms": ms(
                    statistics.fmean(r["sampling_gpu_union_ns"] for r in rs)
                ),
                "d2h_host_ms": ms(statistics.fmean(r["d2h_host_ns"] for r in rs)),
                "gap_to_next_ms": ms(
                    statistics.fmean(
                        r["gap_to_next_ns"]
                        for r in rs
                        if r["gap_to_next_ns"] is not None
                    )
                    if any(r["gap_to_next_ns"] is not None for r in rs)
                    else None
                ),
                "host_us_per_eager_kernel": (
                    round(
                        statistics.fmean(
                            (r["wall_ns"] - r["d2h_host_ns"])
                            / (r["gpu_kernels"] - r["gpu_graph_node_kernels"])
                            for r in rs
                            if r["gpu_kernels"] > r["gpu_graph_node_kernels"]
                        )
                        / 1e3,
                        1,
                    )
                    if any(r["gpu_kernels"] > r["gpu_graph_node_kernels"] for r in rs)
                    else None
                ),
                "kernels_per_step": round(
                    statistics.fmean(r["gpu_kernels"] for r in rs), 1
                ),
                "graph_node_kernels_per_step": round(
                    statistics.fmean(r["gpu_graph_node_kernels"] for r in rs), 1
                ),
                "sampling_kernels_per_step": round(
                    statistics.fmean(r["sampling_gpu_kernels"] for r in rs), 1
                ),
            }
        )
    return {
        "thread": tid,
        "steps": len(rows),
        "active_span_ns": total(active),
        "execute_union_ns": total(exec_union),
        "active_gap_ns": total(active_gap),
        "inactive_ns": total(inactive),
        "ar_gpu_union_ns": total(ar_gpu),
        "gpu_busy_fraction_while_active": (
            (total(intersect(ar_gpu, active)) / total(active)) if active else None
        ),
        "by_mode_batch": table,
        "steps_detail": rows,
        "states": {
            "execute": exec_union,
            "active_gap": active_gap,
            "inactive": inactive,
        },
    }


def vocoder_budget(ranges, threads, tk, lo, hi):
    tid = threads["vocoder"]
    if tid is None:
        return {}
    thread_ranges = [
        r for r in ranges if r["tid"] == tid and r["start"] < hi and r["end"] > lo
    ]
    kernels = tk.get(tid, [])
    api_starts = [k["api_start"] for k in kernels]
    covered = []
    exclusive = {}
    cat_of_range = {}
    for name, pred in VOCODER_ORDER:
        members = [r for r in thread_ranges if pred(r)]
        for r in members:
            cat_of_range.setdefault(id(r), name)
        u = clip(union([(r["start"], r["end"]) for r in members]), lo, hi)
        ex = subtract(u, covered)
        exclusive[name] = ex
        covered = union(covered + u)
    exclusive["idle"] = subtract([(lo, hi)], covered)
    # GPU union of kernels launched while the thread was in each exclusive state.
    gpu_by_state = {}
    for name, ivs in exclusive.items():
        ks = []
        for s, e in ivs:
            ks += kernels_launched_in(kernels, api_starts, s, e)
        gpu_by_state[name] = gpu_summary(ks, lo, hi)
    voc_gpu = union(clip([(k["start"], k["end"]) for k in kernels], lo, hi))
    budget = {
        name: {
            "host_exclusive_ns": total(ivs),
            "host_share": total(ivs) / (hi - lo),
            "launched_gpu_union_ns": gpu_by_state[name]["gpu_union_ns"],
            "launched_kernels": gpu_by_state[name]["kernels"],
        }
        for name, ivs in exclusive.items()
    }
    return {
        "thread": tid,
        "budget": budget,
        "vocoder_gpu_union_ns": total(voc_gpu),
        "states": exclusive,
    }


SYNC_APIS = (
    "Synchronize",
    "cudaMemcpy_",
    "cudaMemcpy2D",
    "cudaMemcpyToSymbol",
    "StreamWaitEvent",
    "EventQuery",
)


def load_thread_apis(db, strings, tid, lo, hi):
    """Runtime API calls on one thread: (start, end, name), sorted by start."""
    out = []
    for start, end, name_id in db.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=? AND start<? AND end>? ORDER BY start",
        (tid, hi, lo),
    ):
        out.append((start, end, strings.get(name_id, "?")))
    return out


def apis_in(apis, api_starts, start, end):
    lo_i = bisect_right(api_starts, start - 1)
    hi_i = bisect_right(api_starts, end - 1)
    return apis[lo_i:hi_i]


def sync_summary(calls):
    counts = Counter()
    host = Counter()
    for s, e, name in calls:
        if any(tag in name for tag in SYNC_APIS):
            short = name.split("_v")[0]
            counts[short] += 1
            host[short] += e - s
    return {
        "count": sum(counts.values()),
        "host_ns": sum(host.values()),
        "by_api": dict(counts),
        "host_by_api": dict(host),
    }


def call_table(ranges, threads, tk, lo, hi, stage, ops, frame_key, bucket, apis=None):
    tid = threads["vocoder"]
    kernels = tk.get(tid, [])
    api_starts = [k["api_start"] for k in kernels]
    apis = apis or []
    api_list_starts = [a[0] for a in apis]
    rows = []
    for r in ranges:
        if (
            r["tid"] != tid
            or r["stage"] != stage
            or r["op"] not in ops
            or not (lo <= r["start"] < hi)
        ):
            continue
        ks = kernels_launched_in(kernels, api_starts, r["start"], r["end"])
        g = gpu_summary(ks, lo, hi)
        frames = frame_key(r)
        sync = sync_summary(apis_in(apis, api_list_starts, r["start"], r["end"]))
        rows.append(
            {
                "op": r["op"],
                "batch": int(r.get("batch_size", 1)),
                "frames": frames,
                "streaming": r.get("streaming"),
                "finalize": r.get("finalize"),
                "host_ns": r["end"] - r["start"],
                "gpu_union_ns": g["gpu_union_ns"],
                "gpu_sum_ns": g["gpu_sum_ns"],
                "kernels": g["kernels"],
                "gpu_tail_ns": max(0, (g["gpu_last_end"] or r["end"]) - r["end"]),
                "sync_calls": sync["count"],
                "sync_host_ns": sync["host_ns"],
                "sync_by_api": sync["by_api"],
                "start": r["start"],
            }
        )
    groups = defaultdict(list)
    for r in rows:
        groups[
            (
                r["op"],
                r["batch"],
                (r["frames"] // bucket) * bucket if r["frames"] is not None else None,
            )
        ].append(r)
    table = []
    for (op, batch, fb), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)
    ):
        host = statistics.fmean(r["host_ns"] for r in rs)
        gpu = statistics.fmean(r["gpu_union_ns"] for r in rs)
        table.append(
            {
                "op": op,
                "batch": batch,
                "frames_bucket": fb,
                "calls": len(rs),
                "host_ms": ms(host),
                "gpu_union_ms": ms(gpu),
                "gpu_over_host": round(gpu / host, 3) if host else None,
                "host_us_per_kernel": (
                    round(
                        statistics.fmean(
                            r["host_ns"] / r["kernels"] for r in rs if r["kernels"]
                        )
                        / 1e3,
                        1,
                    )
                    if any(r["kernels"] for r in rs)
                    else None
                ),
                "kernels_per_call": round(
                    statistics.fmean(r["kernels"] for r in rs), 1
                ),
                "mean_kernel_us": (
                    round(
                        statistics.fmean(
                            r["gpu_sum_ns"] / r["kernels"] for r in rs if r["kernels"]
                        )
                        / 1e3,
                        2,
                    )
                    if any(r["kernels"] for r in rs)
                    else None
                ),
                "gpu_tail_ms": ms(statistics.fmean(r["gpu_tail_ns"] for r in rs)),
                "sync_calls_per_call": round(
                    statistics.fmean(r["sync_calls"] for r in rs), 1
                ),
                "sync_host_ms": ms(statistics.fmean(r["sync_host_ns"] for r in rs)),
                "sync_by_api": dict(
                    sum((Counter(r["sync_by_api"]) for r in rs), Counter())
                ),
            }
        )
    return {"table": table, "calls": rows}


def preprocessing_table(ranges, threads, lo, hi):
    rows = []
    prep = [
        r
        for r in ranges
        if r["stage"] == "preprocessing"
        and r["op"] == "request"
        and lo <= r["start"] < hi
    ]
    by_tid = {}
    for r in ranges:
        if r["stage"] == "preprocessing":
            by_tid.setdefault(r["tid"], []).append(r)
    index = {tid: ThreadRanges(rs) for tid, rs in by_tid.items()}
    for req in prep:
        inner = index[req["tid"]].inside(req, stage="preprocessing")
        d = defaultdict(int)
        for r in inner:
            d[r["op"]] += r["end"] - r["start"]
        rows.append(
            {
                "request_id": req.get("request_id"),
                "total_ns": req["end"] - req["start"],
                "reference_lookup_ns": d["reference_lookup_or_encode"],
                "reference_miss": d["reference_cache_miss"] > 0,
                "finalize_lock_wait_ns": d["finalize_including_lock"]
                - d["prepare_embeddings"],
                "prepare_embeddings_ns": d["prepare_embeddings"],
                "embedding_cache_key_ns": d["embedding_cache_key"],
            }
        )

    def col(name, pred=lambda r: True):
        return {
            k: ms(v) if k != "n" else v
            for k, v in describe([r[name] for r in rows if pred(r)]).items()
        }

    return {
        "requests": len(rows),
        "reference_misses": sum(1 for r in rows if r["reference_miss"]),
        "total_ms": col("total_ns"),
        "reference_lookup_ms_miss": col(
            "reference_lookup_ns", lambda r: r["reference_miss"]
        ),
        "reference_lookup_ms_hit": col(
            "reference_lookup_ns", lambda r: not r["reference_miss"]
        ),
        "finalize_lock_wait_ms": col("finalize_lock_wait_ns"),
        "prepare_embeddings_ms": col("prepare_embeddings_ns"),
        "embedding_cache_key_ms": col("embedding_cache_key_ns"),
        "rows": rows,
    }


def flow_frames(r):
    if r["op"] == "native":
        return (int(r.get("prompt_tokens", 0)) + int(r.get("tokens", 0))) * 2
    tl, pl = r.get("token_lengths"), r.get("prompt_lengths")
    if tl and pl:
        return max(t + p for t, p in zip(tl, pl)) * 2
    return None


def hift_frames(r):
    return int(r.get("frames", 0))


def request_timeline(ranges, marks, threads, lo, hi):
    per = defaultdict(lambda: defaultdict(list))
    for m in marks:
        rid = m.get("request_id")
        if rid:
            per[rid][(m["stage"], m["op"])].append(m)
    voc_tid = threads["vocoder"]
    steps = [
        r
        for r in ranges
        if r["tid"] == voc_tid and r["stage"] == "vocoder" and r["op"] == "stream_step"
    ]
    finals = [
        r
        for r in ranges
        if r["tid"] == voc_tid
        and r["stage"] == "vocoder"
        and r["op"] == "stream_delta"
        and r.get("finalize")
    ]
    hops = defaultdict(list)
    for s in steps:
        for rid in s.get("request_ids", []):
            hops[rid].append(
                {
                    "kind": "batched" if int(s.get("batch_size", 1)) > 1 else "single",
                    "batch": int(s.get("batch_size", 1)),
                    "token_offset": int(s.get("token_offset", 0)),
                    "hop": int(s.get("hop", 0)),
                    "start": s["start"],
                    "end": s["end"],
                }
            )
    for f in finals:
        hops[f["request_id"]].append(
            {
                "kind": "final",
                "batch": 1,
                "token_offset": int(f.get("token_offset", 0)),
                "hop": None,
                "start": f["start"],
                "end": f["end"],
            }
        )
    requests = []
    hop_rows = []
    for rid, ev in per.items():
        first = lambda key: (ev[key][0]["start"] if ev.get(key) else None)  # noqa: E731
        chunks = sorted(
            (
                (m["metadata"].get("chunk_id"), m["start"])
                for m in ev.get(("tts_engine", "stage_stream_chunk_sent"), [])
            ),
            key=lambda x: (x[0] if x[0] is not None else 1 << 30),
        )
        chunk_time = {cid: t for cid, t in chunks}
        ar_done = first(("tts_engine", "stage_complete"))
        pcm = sorted(
            m["start"]
            for m in ev.get(("http", "pcm_yield"), [])
            + ev.get(("http", "first_pcm_yield"), [])
        )
        rec = {
            "request_id": rid,
            "http_received": first(("http", "received")),
            "queue_enter": first(("tts_engine", "scheduler_queue_enter")),
            "prefill_start": first(("tts_engine", "scheduler_prefill_start")),
            "prefill_end": first(("tts_engine", "scheduler_prefill_end")),
            "chunk0_sent": chunk_time.get(0),
            "ar_complete": ar_done,
            "first_pcm_yield": first(("http", "first_pcm_yield")),
            "stream_body_complete": first(("http", "stream_body_complete")),
            "buffered_body_ready": first(("http", "buffered_body_ready")),
            "chunks_sent": len(chunks),
            "hops": 0,
        }
        rid_hops = sorted(hops.get(rid, []), key=lambda h: h["start"])
        rec["hops"] = len(rid_hops)
        for i, h in enumerate(rid_hops):
            if h["kind"] == "final":
                ready = ar_done
            else:
                needed = h["token_offset"] + h["hop"] + LOOKAHEAD
                n = (
                    0
                    if needed <= AR_FIRST_FLUSH
                    else -(-(needed - AR_FIRST_FLUSH) // AR_FOLLOWUP_FLUSH)
                )
                ready = chunk_time.get(n)
            yields = [t for t in pcm if t >= h["end"]]
            hop_rows.append(
                {
                    "request_id": rid,
                    "hop_index": i,
                    "kind": h["kind"],
                    "batch": h["batch"],
                    "token_offset": h["token_offset"],
                    "hop": h["hop"],
                    "ready_ns": ready,
                    "start_ns": h["start"],
                    "end_ns": h["end"],
                    "queue_delay_ns": (
                        (h["start"] - ready) if ready is not None else None
                    ),
                    "run_ns": h["end"] - h["start"],
                    "to_pcm_yield_ns": (yields[0] - h["end"]) if yields else None,
                }
            )
        requests.append(rec)

    def delta(a, b):
        return [
            r[b] - r[a]
            for r in requests
            if r.get(a) is not None and r.get(b) is not None
        ]

    summary = {
        "requests": len(requests),
        "http_to_queue_enter_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("http_received", "queue_enter")).items()
        },
        "queue_enter_to_prefill_start_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("queue_enter", "prefill_start")).items()
        },
        "prefill_start_to_chunk0_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("prefill_start", "chunk0_sent")).items()
        },
        "chunk0_to_first_pcm_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("chunk0_sent", "first_pcm_yield")).items()
        },
        "http_to_first_pcm_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("http_received", "first_pcm_yield")).items()
        },
        "ar_complete_to_body_complete_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("ar_complete", "stream_body_complete")).items()
        },
        "prefill_start_to_ar_complete_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("prefill_start", "ar_complete")).items()
        },
        "ar_complete_to_buffered_body_ready_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("ar_complete", "buffered_body_ready")).items()
        },
        "http_to_buffered_body_ready_ms": {
            k: ms(v) if k != "n" else v
            for k, v in describe(delta("http_received", "buffered_body_ready")).items()
        },
    }
    ar_span = [r["queue_enter"] for r in requests if r.get("queue_enter")], [
        r["ar_complete"] for r in requests if r.get("ar_complete")
    ]
    voc_ranges = [
        r
        for r in ranges
        if r["tid"] == voc_tid and r["stage"] in ("vocoder", "flow", "hift")
    ]
    summary["phases_ms_from_window_start"] = {
        "ar_first_queue_enter": ms(min(ar_span[0]) - lo) if ar_span[0] else None,
        "ar_last_complete": ms(max(ar_span[1]) - lo) if ar_span[1] else None,
        "vocoder_first_range": (
            ms(min(r["start"] for r in voc_ranges) - lo) if voc_ranges else None
        ),
        "vocoder_last_range": (
            ms(max(r["end"] for r in voc_ranges) - lo) if voc_ranges else None
        ),
        "window_end": ms(hi - lo),
    }
    by_kind = defaultdict(list)
    for h in hop_rows:
        by_kind[(h["kind"], h["batch"] > 1, h["token_offset"] == 0)].append(h)
    hop_table = []
    for (kind, batched, first_hop), hs in sorted(
        by_kind.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        q = [h["queue_delay_ns"] for h in hs if h["queue_delay_ns"] is not None]
        hop_table.append(
            {
                "kind": kind,
                "batched": batched,
                "first_hop": first_hop,
                "hops": len(hs),
                "queue_delay_ms": {
                    k: ms(v) if k != "n" else v for k, v in describe(q).items()
                },
                "run_ms": {
                    k: ms(v) if k != "n" else v
                    for k, v in describe([h["run_ns"] for h in hs]).items()
                },
                "to_pcm_yield_ms": {
                    k: ms(v) if k != "n" else v
                    for k, v in describe(
                        [
                            h["to_pcm_yield_ns"]
                            for h in hs
                            if h["to_pcm_yield_ns"] is not None
                        ]
                    ).items()
                },
            }
        )
    return {
        "summary": summary,
        "hop_table": hop_table,
        "requests": requests,
        "hops": hop_rows,
    }


def replay_sequence(marks, lo, hi):
    """Vocoder inbox arrival order reconstructed from the AR side marks.

    Each stream chunk is the tts_engine stage_stream_chunk_sent mark (28 tokens
    for chunk 0, 25 for later chunks by the producer contract); stream_done is
    the tts_engine stage_complete mark. This is the fixture a scheduler unit
    test replays through the vocoder scheduler with a fake vocoder to assert
    per-request token order and final-decode liveness.
    """
    events = []
    for m in marks:
        if not (lo <= m["start"] < hi) or m["stage"] != "tts_engine":
            continue
        rid = m.get("request_id")
        if m["op"] == "stage_stream_chunk_sent":
            cid = int(m["metadata"].get("chunk_id", 0))
            events.append(
                {
                    "t_ns": m["start"],
                    "request_id": rid,
                    "type": "stream_chunk",
                    "chunk_id": cid,
                    "tokens": AR_FIRST_FLUSH if cid == 0 else AR_FOLLOWUP_FLUSH,
                }
            )
        elif m["op"] == "stage_complete":
            events.append(
                {"t_ns": m["start"], "request_id": rid, "type": "stream_done"}
            )
    events.sort(key=lambda e: e["t_ns"])
    return events


def gap_attribution(kernels, ar, voc, lo, hi, min_gap_ns):
    all_union = union(clip([(k["start"], k["end"]) for k in kernels], lo, hi))
    g = [x for x in gaps(all_union, lo, hi) if x[1] - x[0] >= min_gap_ns]
    gap_union = union(g)
    ar_states = ar.get("states", {"unknown": [(lo, hi)]})
    voc_states = voc.get("states", {"unknown": [(lo, hi)]})
    table = []
    for an, aiv in ar_states.items():
        a = intersect(gap_union, aiv)
        if not a:
            continue
        for vn, viv in voc_states.items():
            t = total(intersect(a, viv))
            if t:
                table.append(
                    {
                        "ar_state": an,
                        "vocoder_state": vn,
                        "gap_ns": t,
                        "share_of_window": t / (hi - lo),
                    }
                )
    table.sort(key=lambda r: -r["gap_ns"])
    return {
        "kernel_union_ns": total(all_union),
        "coverage": total(all_union) / (hi - lo),
        "gap_count": len(g),
        "gap_total_ns": total(gap_union),
        "min_gap_ns": min_gap_ns,
        "table": table,
    }


def stage_of_kernel(k, thread_index, threads):
    tid = k["launch_tid"]
    if tid is None:
        return "unattributed"
    tr = thread_index.get(tid)
    r = tr.containing(k["api_start"]) if tr else None
    if r is None:
        if tid == threads["ar"]:
            return "ar_unannotated"
        if tid == threads["vocoder"]:
            return "vocoder_unannotated"
        return "other"
    if r["stage"] == "ar":
        return "ar_sampling" if r["op"] == "sampling" else "ar_forward"
    if r["stage"] == "flow":
        return "flow_native" if r["op"] == "native" else "flow_packed"
    if r["stage"] == "vocoder":
        return "vocoder_glue"
    return r["stage"]


def sm_by_stage(db, kernels, thread_index, threads, lo, hi, metric):
    if metric is None:
        return {}
    samples = [
        (t, v)
        for t, v in db.execute(
            "SELECT timestamp, value FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
            (metric["typeId"], metric["metricId"], lo, hi),
        )
    ]
    if not samples:
        return {}
    stage_union = defaultdict(list)
    for k in kernels:
        stage_union[stage_of_kernel(k, thread_index, threads)].append(
            (k["start"], k["end"])
        )
    stage_union = {s: union(clip(v, lo, hi)) for s, v in stage_union.items()}
    labels = [set() for _ in samples]
    times = [t for t, _ in samples]
    # A sample describes the sampling interval [t, t + period); label it with
    # every stage whose kernels overlap that interval, not the instant t.
    period = (
        int((times[-1] - times[0]) / max(len(times) - 1, 1)) if len(times) > 1 else 0
    )
    for s, ivs in stage_union.items():
        for a, b in ivs:
            i = bisect_right(times, a - period)
            while i < len(times) and times[i] < b:
                labels[i].add(s)
                i += 1
    agg = defaultdict(list)
    for (t, v), lab in zip(samples, labels):
        agg["+".join(sorted(lab)) if lab else "idle"].append(v)
    rows = [
        {
            "kernels_present": name,
            "samples": len(vals),
            "share_of_samples": len(vals) / len(samples),
            "sm_active_mean": statistics.fmean(vals),
        }
        for name, vals in agg.items()
    ]
    rows.sort(key=lambda r: -r["samples"])
    return {
        "metric": metric,
        "samples": len(samples),
        "overall_mean": statistics.fmean(v for _, v in samples),
        "by_kernels_present": rows,
        "stage_gpu_union_ns": {s: total(v) for s, v in stage_union.items()},
    }


def find_sm_metric(db):
    for row in db.execute(
        "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
    ):
        if row[2] == "SMs Active [Throughput %]":
            return {"typeId": row[0], "metricId": row[1], "name": row[2]}
    return None


# ---------------------------------------------------------------- report


def fmt_table(rows, cols):
    if not rows:
        return "(none)\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return head + body


def markdown(result):
    w = result["window"]
    out = [f"# {result['name']}\n"]
    out.append(
        f"window {w['start_ns']}..{w['end_ns']} ({ms(w['duration_ns'])} ms), device {w['device']}, "
        f"threads ar={result['threads']['ar']} vocoder={result['threads']['vocoder']}\n"
    )
    ar = result["ar"]
    if ar:
        out.append("## AR thread\n")
        out.append(
            f"steps {ar['steps']}, active span {ms(ar['active_span_ns'])} ms, execute union {ms(ar['execute_union_ns'])} ms, "
            f"scheduler gap while active {ms(ar['active_gap_ns'])} ms, inactive {ms(ar['inactive_ns'])} ms, "
            f"AR GPU union {ms(ar['ar_gpu_union_ns'])} ms, GPU busy while active {ar['gpu_busy_fraction_while_active']:.3f}\n"
        )
        out.append(
            fmt_table(
                ar["by_mode_batch"],
                [
                    "mode",
                    "batch_bucket",
                    "steps",
                    "wall_ms",
                    "gpu_union_ms",
                    "gpu_tail_ms",
                    "sampling_host_ms",
                    "sampling_gpu_ms",
                    "d2h_host_ms",
                    "gap_to_next_ms",
                    "host_us_per_eager_kernel",
                    "kernels_per_step",
                    "graph_node_kernels_per_step",
                    "sampling_kernels_per_step",
                ],
            )
        )
    voc = result["vocoder"]
    if voc:
        out.append("## Vocoder thread budget (exclusive innermost host state)\n")
        rows = [
            {
                "state": k,
                "host_ms": ms(v["host_exclusive_ns"]),
                "host_share": round(v["host_share"], 3),
                "launched_gpu_union_ms": ms(v["launched_gpu_union_ns"]),
                "launched_kernels": v["launched_kernels"],
            }
            for k, v in voc["budget"].items()
        ]
        out.append(
            fmt_table(
                rows,
                [
                    "state",
                    "host_ms",
                    "host_share",
                    "launched_gpu_union_ms",
                    "launched_kernels",
                ],
            )
        )
        out.append(f"vocoder GPU union {ms(voc['vocoder_gpu_union_ns'])} ms\n")
    call_cols = [
        "op",
        "batch",
        "frames_bucket",
        "calls",
        "host_ms",
        "gpu_union_ms",
        "gpu_over_host",
        "host_us_per_kernel",
        "kernels_per_call",
        "mean_kernel_us",
        "gpu_tail_ms",
        "sync_calls_per_call",
        "sync_host_ms",
        "sync_by_api",
    ]
    out.append("## Flow calls\n")
    out.append(fmt_table(result["flow"]["table"], call_cols))
    out.append("## HiFT calls\n")
    out.append(fmt_table(result["hift"]["table"], call_cols))
    pp = result["preprocessing"]
    out.append("## Preprocessing per request\n")
    out.append(
        "```\n"
        + json.dumps({k: v for k, v in pp.items() if k != "rows"}, indent=1)
        + "\n```\n"
    )
    rt = result["requests"]
    out.append("## Request timeline\n")
    out.append("```\n" + json.dumps(rt["summary"], indent=1) + "\n```\n")
    out.append("## Streaming hops\n")
    hrows = []
    for h in rt["hop_table"]:
        hrows.append(
            {
                "kind": h["kind"],
                "batched": h["batched"],
                "first_hop": h["first_hop"],
                "hops": h["hops"],
                "queue_delay_p50_ms": h["queue_delay_ms"].get("p50"),
                "queue_delay_p90_ms": h["queue_delay_ms"].get("p90"),
                "queue_delay_max_ms": h["queue_delay_ms"].get("max"),
                "run_p50_ms": h["run_ms"].get("p50"),
                "run_max_ms": h["run_ms"].get("max"),
                "to_pcm_p50_ms": h["to_pcm_yield_ms"].get("p50"),
            }
        )
    out.append(
        fmt_table(
            hrows,
            [
                "kind",
                "batched",
                "first_hop",
                "hops",
                "queue_delay_p50_ms",
                "queue_delay_p90_ms",
                "queue_delay_max_ms",
                "run_p50_ms",
                "run_max_ms",
                "to_pcm_p50_ms",
            ],
        )
    )
    ga = result["gaps"]
    out.append("## GPU idle gaps by joint thread state\n")
    out.append(
        f"kernel coverage {ga['coverage']:.3f}, gaps >= {ms(ga['min_gap_ns'])} ms: {ga['gap_count']} totaling {ms(ga['gap_total_ns'])} ms\n"
    )
    out.append(
        fmt_table(
            [
                {
                    "ar_state": r["ar_state"],
                    "vocoder_state": r["vocoder_state"],
                    "gap_ms": ms(r["gap_ns"]),
                    "share_of_window": round(r["share_of_window"], 3),
                }
                for r in ga["table"][:14]
            ],
            ["ar_state", "vocoder_state", "gap_ms", "share_of_window"],
        )
    )
    sm = result["sm"]
    if sm:
        out.append("## SM Active conditioned on kernels present\n")
        out.append(
            f"overall mean {sm['overall_mean']:.2f} over {sm['samples']} samples\n"
        )
        out.append(
            fmt_table(
                [
                    {
                        "kernels_present": r["kernels_present"],
                        "share_of_samples": round(r["share_of_samples"], 3),
                        "sm_active_mean": round(r["sm_active_mean"], 2),
                    }
                    for r in sm["by_kernels_present"][:12]
                ],
                ["kernels_present", "share_of_samples", "sm_active_mean"],
            )
        )
        out.append(
            "stage GPU unions (ms): "
            + json.dumps({k: ms(v) for k, v in sm["stage_gpu_union_ns"].items()})
            + "\n"
        )
    return "\n".join(out)


def analyze(sqlite_path, device, start_ns, end_ns, min_gap_ms, name):
    db = sqlite3.connect(Path(sqlite_path).resolve().as_uri() + "?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges, marks = load_annotations(db, strings)
    threads = pick_threads(ranges, marks)
    lo, hi = (
        (start_ns, end_ns)
        if start_ns is not None and end_ns is not None
        else default_window(marks)
    )
    kernels = load_kernels(db, device, lo, hi)
    tk = kernels_by_thread(kernels)
    thread_index = {
        tid: ThreadRanges([r for r in ranges if r["tid"] == tid])
        for tid in {r["tid"] for r in ranges}
    }
    ar = ar_ledger(ranges, marks, threads, tk, lo, hi)
    voc = vocoder_budget(ranges, threads, tk, lo, hi)
    voc_apis = (
        load_thread_apis(db, strings, threads["vocoder"], lo, hi)
        if threads["vocoder"]
        else []
    )
    flow = call_table(
        ranges,
        threads,
        tk,
        lo,
        hi,
        "flow",
        ("native", "packed_causal", "packed_buffered"),
        flow_frames,
        200,
        voc_apis,
    )
    hift = call_table(
        ranges, threads, tk, lo, hi, "hift", ("inference",), hift_frames, 200, voc_apis
    )
    prep = preprocessing_table(ranges, threads, lo, hi)
    reqs = request_timeline(ranges, marks, threads, lo, hi)
    ga = gap_attribution(kernels, ar, voc, lo, hi, int(min_gap_ms * 1e6))
    sm = sm_by_stage(db, kernels, thread_index, threads, lo, hi, find_sm_metric(db))
    unattributed = sum(1 for k in kernels if k["launch_tid"] is None)
    result = {
        "name": name,
        "sqlite": str(sqlite_path),
        "window": {
            "device": device,
            "start_ns": lo,
            "end_ns": hi,
            "duration_ns": hi - lo,
        },
        "threads": {
            k: (v if not isinstance(v, list) else v) for k, v in threads.items()
        },
        "kernels_in_window": len(kernels),
        "kernels_without_launch": unattributed,
        "ar": {k: v for k, v in ar.items() if k != "states"},
        "vocoder": {k: v for k, v in voc.items() if k != "states"},
        "flow": flow,
        "hift": hift,
        "preprocessing": prep,
        "requests": reqs,
        "gaps": ga,
        "sm": sm,
        "replay": replay_sequence(marks, lo, hi),
    }
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--min-gap-ms", type=float, default=0.5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument(
        "--replay-json",
        type=Path,
        help="write the reconstructed vocoder inbox sequence for scheduler replay tests",
    )
    args = parser.parse_args()
    name = args.name or args.sqlite.parent.name
    result = analyze(
        args.sqlite, args.device, args.start_ns, args.end_ns, args.min_gap_ms, name
    )
    if args.replay_json:
        args.replay_json.write_text(json.dumps(result["replay"], indent=1) + "\n")
    text = markdown(result)
    if args.md:
        args.md.write_text(text)
    else:
        print(text)
    if args.json:
        slim = dict(result)
        slim.pop("replay", None)
        slim["ar"] = {k: v for k, v in result["ar"].items() if k != "steps_detail"}
        slim["flow"] = {"table": result["flow"]["table"]}
        slim["hift"] = {"table": result["hift"]["table"]}
        slim["preprocessing"] = {
            k: v for k, v in result["preprocessing"].items() if k != "rows"
        }
        slim["requests"] = {
            "summary": result["requests"]["summary"],
            "hop_table": result["requests"]["hop_table"],
            "requests": result["requests"]["requests"],
        }
        args.json.write_text(json.dumps(slim, indent=1) + "\n")


if __name__ == "__main__":
    main()
