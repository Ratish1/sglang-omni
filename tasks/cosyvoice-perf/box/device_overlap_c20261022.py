"""Whether the speech tokenizer's device work slows the vocoder's and the AR's graph replays (10-22, L3's c16 gaps).

For each label=sqlite (a cosy_profile_run trace with graph level CUDA graph tracing): the tokenizer threads are the
threads of pre.tokenize ranges, the vocoder's those of voc.step ranges, the AR's that of the most loop.step DECODE
ranges. Each thread's device work is the graph replays, kernels and copies whose correlation id matches one of its
CUDA runtime calls. For every graph the vocoder or the AR replays, its replays' device durations split by overlap
with the tokenizer's device work (the same graph, so the same shape and kernels), the share overlapped, and over
all of a thread's replays the device time added in the overlapped split against the clear mean.
usage: python3 device_overlap_c20261022.py l2=<c16.sqlite> l3=<c16.sqlite> [--skip-head-s 1] [--skip-tail-s 1]
"""

import bisect
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

DEVICE_TABLES = ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY")


def option(name, default):
    if name in sys.argv:
        return float(sys.argv[sys.argv.index(name) + 1])
    else:
        return default


def threads_of(db, where):
    rows = db.execute(
        "select n.globalTid from NVTX_EVENTS n left join StringIds s on s.id = n.textId "
        f"where n.end is not null and {where}"
    ).fetchall()
    return Counter(thread for (thread,) in rows)


def correlations(db, threads):
    marks = ",".join("?" * len(threads))
    return {
        correlation
        for (correlation,) in db.execute(
            f"select correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid in ({marks})", tuple(threads)
        )
    }


def device_work(db, table, correlation_ids):
    columns = "start, end, graphExecId" if table.endswith("GRAPH_TRACE") else "start, end, null"
    if table.endswith("KERNEL"):
        # note (ratish): a replayed graph's kernels carry the graph's correlation id; the graph row covers them
        where = "graphId is null or graphId = 0"
    else:
        where = "1"
    return [row for row in db.execute(f"select correlationId, {columns} from {table} where {where}") if row[0] in correlation_ids]


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def overlap_ns(start, end, intervals, starts):
    i = max(0, bisect.bisect_right(starts, start) - 1)
    total = 0
    while i < len(intervals) and intervals[i][0] < end:
        total += max(0, min(end, intervals[i][1]) - max(start, intervals[i][0]))
        i += 1
    return total


skip_head = option("--skip-head-s", 1.0) * 1e9
skip_tail = option("--skip-tail-s", 1.0) * 1e9
for argument in (a for a in sys.argv[1:] if "=" in a):
    label, path = argument.split("=", 1)
    db = sqlite3.connect(path)
    tokenizer_threads = threads_of(db, "coalesce(n.text, s.value) = 'pre.tokenize'")
    vocoder_threads = threads_of(db, "coalesce(n.text, s.value) like 'voc.step %'")
    ar_threads = threads_of(db, "coalesce(n.text, s.value) like 'loop.step DECODE%'")
    print(f"== {label} {path}: tokenizer threads {len(tokenizer_threads)} ({sum(tokenizer_threads.values())} calls)")
    tokenizer_ids = correlations(db, set(tokenizer_threads))
    tokenizer_work = [(start, end) for table in DEVICE_TABLES for _, start, end, _ in device_work(db, table, tokenizer_ids)]
    if not tokenizer_work:
        print("  no tokenizer device work")
        continue
    else:
        pass
    window_start = min(start for start, _ in tokenizer_work) + skip_head
    window_end = max(end for _, end in tokenizer_work) - skip_tail
    tokenizer_intervals = merged(tokenizer_work)
    tokenizer_starts = [start for start, _ in tokenizer_intervals]
    busy = sum(min(end, window_end) - max(start, window_start) for start, end in tokenizer_intervals if end > window_start and start < window_end)
    print(f"  tokenizer device busy {busy / 1e6:.0f} ms of {(window_end - window_start) / 1e6:.0f} ms ({100 * busy / (window_end - window_start):.1f} %)")
    for name, threads in (("vocoder", vocoder_threads), ("AR", Counter(dict(ar_threads.most_common(1))))):
        ids = correlations(db, set(threads))
        replays = defaultdict(lambda: {"clear": [], "overlapped": [], "shares": []})
        for _, start, end, graph in device_work(db, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", ids):
            if start < window_start or end > window_end:
                continue
            else:
                pass
            covered = overlap_ns(start, end, tokenizer_intervals, tokenizer_starts)
            split = "overlapped" if covered > 0 else "clear"
            replays[graph][split].append((end - start) / 1e3)
            if covered > 0:
                replays[graph]["shares"].append(covered / (end - start))
            else:
                pass
        added_us = 0.0
        overlapped_count = 0
        total_count = 0
        print(f"  {name}: {len(replays)} graphs")
        for graph, splits in sorted(replays.items(), key=lambda item: -(len(item[1]["clear"]) + len(item[1]["overlapped"]))):
            clear, hit = splits["clear"], splits["overlapped"]
            total_count += len(clear) + len(hit)
            overlapped_count += len(hit)
            if clear and hit:
                added_us += (statistics.mean(hit) - statistics.mean(clear)) * len(hit)
            else:
                pass
            if len(clear) + len(hit) >= 20:
                hit_text = (
                    f"overlapped n {len(hit)} mean {statistics.mean(hit):.0f} us (covered {100 * statistics.mean(splits['shares']):.0f} %)"
                    if hit
                    else "overlapped n 0"
                )
                clear_text = f"clear n {len(clear)} mean {statistics.mean(clear):.0f} us" if clear else "clear n 0"
                ratio = f", {statistics.mean(hit) / statistics.mean(clear):.3f}x" if clear and hit else ""
                print(f"    graph {graph}: {clear_text}; {hit_text}{ratio}")
            else:
                pass
        print(
            f"  {name} replays {total_count}, overlapped {overlapped_count} ({100 * overlapped_count / max(1, total_count):.1f} %), "
            f"device time added in the overlapped split {added_us / 1e3:.1f} ms over {(window_end - window_start) / 1e9:.1f} s"
        )
