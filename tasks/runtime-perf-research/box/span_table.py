"""Per stage and span: wall, device time it launched, syncs inside it, and other processes' device time over it.

usage: python3 span_table.py TRACE.sqlite SERVE.LOG [--top 40] [--labels]

Reads the in-code spans of a trace marks tree (SGLANG_ENABLE_NVTX_SCHEDULER=1, SGLANG_ENABLE_NVTX_OPERATIONS=1,
nsys --trace=cuda,nvtx,osrt). A process is named by its sglang_ar_startup line in the serve log, else by the span
family it emits. Per span instance:

  wall    the range's own duration
  dev     union of the device intervals (kernels, graph launches, copies) whose launching CUDA call sits inside
          the range on the same thread
  lag     first device start of the instance minus the range start
  sync    time inside cudaEventSynchronize / cudaStreamSynchronize / cudaDeviceSynchronize / blocking cudaMemcpy
  other   other processes' device busy over the range (on one card without MPS this is the time slice the range
          shared or waited for)

Names are grouped with the batch numbers dropped (bs, toks, rows); --labels adds the exact labels of omni.forward.
"""

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=(\d+) .*pid=(\d+)")
NUMBERS = re.compile(r"(bs|toks|rows)=\d+")
SYNC_CALLS = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy_v", "cudaMemcpy2D_v")
FAMILY_STAGE = (("omni.code2wav", "code2wav"), ("omni.vocoder", "vocoder"), ("omni.threaded_run", "preprocessing"),
                ("omni.simple_run", "simple"))


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def covered(spans, start, end, index_from=0):
    total = 0
    for a, b in spans[index_from:]:
        if a >= end:
            break
        else:
            total += max(0, min(b, end) - max(a, start))
    return total


def first_at(spans, start):
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    return lo


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))] if ordered else 0.0


def tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def device_records(db, names):
    """(correlationId, globalPid, start, end) for kernels, graph launches and memory work."""
    rows = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in names:
            rows.extend(db.execute(f"SELECT correlationId, globalPid, start, end FROM {table}"))
        else:
            pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("serve_log")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()

    stages = {}
    with open(args.serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[3])] = match[1]
            else:
                pass

    db = sqlite3.connect(args.trace)
    names = tables(db)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    session = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
    window_ns = session[1] - session[0]

    device_by_corr = {}
    device_by_pid = defaultdict(list)
    for corr, gpid, start, end in device_records(db, names):
        pid = pid_of(gpid)
        device_by_corr[(pid, corr)] = device_by_corr.get((pid, corr), []) + [(start, end)]
        device_by_pid[pid].append((start, end))
    busy = {pid: merged(spans) for pid, spans in device_by_pid.items()}

    calls_by_tid = defaultdict(list)
    for start, end, name_id, gtid, corr in db.execute(
        "SELECT start, end, nameId, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        calls_by_tid[gtid].append((start, end, strings.get(name_id, ""), corr))
    for calls in calls_by_tid.values():
        calls.sort()

    ranges = []
    text_column = "coalesce(text, (SELECT value FROM StringIds WHERE id = textId))"
    for start, end, text, gtid in db.execute(
        f"SELECT start, end, {text_column}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"
    ):
        if text and (text.startswith("omni.") or text.startswith("scheduler.")):
            ranges.append((start, end, text, gtid))
        else:
            pass

    families = defaultdict(set)
    for _, _, text, gtid in ranges:
        families[pid_of(gtid)].add(text)
    for pid, texts in families.items():
        if pid not in stages:
            stages[pid] = next((stage for prefix, stage in FAMILY_STAGE if any(t.startswith(prefix) for t in texts)),
                               f"pid{pid}")
        else:
            pass

    stats = defaultdict(lambda: defaultdict(list))
    labels = defaultdict(lambda: defaultdict(list))
    for start, end, text, gtid in ranges:
        pid = pid_of(gtid)
        calls = calls_by_tid.get(gtid, [])
        lo, hi = 0, len(calls)
        while lo < hi:
            mid = (lo + hi) // 2
            if calls[mid][0] < start:
                lo = mid + 1
            else:
                hi = mid
        launched, sync = [], 0
        for call_start, call_end, call_name, corr in calls[lo:]:
            if call_start >= end:
                break
            else:
                launched.extend(device_by_corr.get((pid, corr), ()))
                if call_name.startswith(SYNC_CALLS):
                    sync += call_end - call_start
                else:
                    pass
        own = merged(launched)
        dev = sum(b - a for a, b in own)
        lag = (own[0][0] - start) if own else None
        other = 0
        for other_pid, spans in busy.items():
            if other_pid != pid:
                other += covered(spans, start, end, first_at(spans, start))
            else:
                pass
        record = (end - start, dev, lag, sync, other)
        stats[stages[pid]][NUMBERS.sub(r"\1=N", text)].append(record)
        if args.labels and text.startswith("omni.forward"):
            labels[stages[pid]][text].append(record)
        else:
            pass

    print(f"window {window_ns / 1e9:.2f} s from the first to the last CUDA call")
    print("\nA. device busy per process (union, ms) and share of the window")
    for pid, spans in sorted(busy.items(), key=lambda item: -sum(b - a for a, b in item[1])):
        total = sum(b - a for a, b in spans)
        print(f"  {stages.get(pid, f'pid{pid}'):16s} pid {pid:7d} {total / 1e6:10.1f} ms {100 * total / window_ns:6.1f} %")

    print("\nB. spans per stage: n, wall sum ms and % of window, wall p50/p90/p99 ms, dev p50 ms, lag p50 ms, "
          "sync p50 ms, other p50 ms")
    for stage, by_name in sorted(stats.items()):
        print(f"\n  [{stage}]")
        rows = sorted(by_name.items(), key=lambda item: -sum(r[0] for r in item[1]))[: args.top]
        for name, records in rows:
            walls = [r[0] / 1e6 for r in records]
            lags = [r[2] / 1e6 for r in records if r[2] is not None]
            print(f"    {name[:58]:58s} n {len(records):6d} sum {sum(walls):9.1f} {100 * sum(walls) * 1e6 / window_ns:5.1f}%"
                  f" p50 {statistics.median(walls):7.3f} p90 {pct(walls, 0.9):7.3f} p99 {pct(walls, 0.99):8.3f}"
                  f" dev {statistics.median(r[1] / 1e6 for r in records):7.3f}"
                  f" lag {statistics.median(lags) if lags else 0:7.3f}"
                  f" sync {statistics.median(r[3] / 1e6 for r in records):7.3f}"
                  f" other {statistics.median(r[4] / 1e6 for r in records):7.3f}")

    if args.labels:
        print("\nC. omni.forward by exact label: n, wall p50/p90 ms, dev p50 ms, lag p50 ms, other p50 ms")
        for stage, by_label in sorted(labels.items()):
            print(f"\n  [{stage}]")
            for label, records in sorted(by_label.items(), key=lambda item: -len(item[1]))[: args.top]:
                walls = [r[0] / 1e6 for r in records]
                lags = [r[2] / 1e6 for r in records if r[2] is not None]
                print(f"    {label[:58]:58s} n {len(records):6d} p50 {statistics.median(walls):7.3f}"
                      f" p90 {pct(walls, 0.9):7.3f} dev {statistics.median(r[1] / 1e6 for r in records):7.3f}"
                      f" lag {statistics.median(lags) if lags else 0:7.3f}"
                      f" other {statistics.median(r[4] / 1e6 for r in records):7.3f}")
    else:
        pass

    switch_tables = sorted(name for name in names if "CONTEXT_SWITCH" in name or "CTXSW" in name.upper())
    print(f"\nD. context switch tables: {', '.join(switch_tables) or 'none'}")
    for table in switch_tables:
        print(f"  {table}: {db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]} rows")


if __name__ == "__main__":
    main()
