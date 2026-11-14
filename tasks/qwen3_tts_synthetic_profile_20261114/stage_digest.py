"""Per-stage digest of one cell's report database (the .sqlite nsys stats exports).

Prints, for the capture window: NVTX ranges by name with their main thread, host waits and copies
by calling thread, CUDA graph replays by graph and stream, eager kernels by stream, and device busy
time with its idle gaps.

Usage: python3 stage_digest.py <cell.sqlite> [--rows N]
"""

import argparse
import collections
import sqlite3
from pathlib import Path

NS_PER_MS = 1e6
HOST_WAIT_APIS = (
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpyAsync",
    "cudaMemcpy",
    "cudaStreamWaitEvent",
    "cudaEventQuery",
    "cudaGraphLaunch",
    "cudaLaunchKernel",
)
GAP_BOUNDS_US = (10, 100, 1000)


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def quantile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def print_rows(title: str, header: str, rows: list[str]) -> None:
    print(f"## {title}")
    print(header)
    for row in rows:
        print(row)


def nvtx_ranges(db, strings, thread_names, rows):
    domains = {
        domain_id: text
        for domain_id, text in db.execute("select domainId, text from NVTX_EVENTS where eventType = 75")
    }
    groups = collections.defaultdict(list)
    threads = collections.defaultdict(collections.Counter)
    for start, end, text, text_id, domain_id, tid in db.execute(
        "select start, end, text, textId, domainId, globalTid from NVTX_EVENTS "
        "where eventType in (59, 60) and end is not null"
    ):
        name = text if text is not None else strings.get(text_id, "?")
        key = f"{domains.get(domain_id, 'default')}:{name}"
        groups[key].append(end - start)
        threads[key][thread_names.get(tid, str(tid))] += 1
    ranked = sorted(groups.items(), key=lambda item: -sum(item[1]))[:rows]
    print_rows(
        "NVTX ranges (wall)",
        "total ms | count | mean ms | p50 ms | p90 ms | max ms | main thread | range",
        [
            f"{sum(d) / NS_PER_MS:9.2f} | {len(d):6d} | {sum(d) / len(d) / NS_PER_MS:8.3f} | "
            f"{quantile(d, 0.5) / NS_PER_MS:7.3f} | {quantile(d, 0.9) / NS_PER_MS:7.3f} | "
            f"{max(d) / NS_PER_MS:7.3f} | {threads[key].most_common(1)[0][0][:22]} | {key[:110]}"
            for key, d in ranked
        ],
    )


def host_waits(db, strings, thread_names, rows):
    groups = collections.defaultdict(list)
    for start, end, name_id, tid in db.execute(
        "select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        name = strings.get(name_id, "?").split("_v")[0]
        if name.startswith(HOST_WAIT_APIS):
            groups[(name, thread_names.get(tid, str(tid)))].append(end - start)
        else:
            pass
    ranked = sorted(groups.items(), key=lambda item: -sum(item[1]))[:rows]
    print_rows(
        "CUDA API host time by thread",
        "total ms | count | mean us | p90 us | max us | thread | api",
        [
            f"{sum(d) / NS_PER_MS:9.2f} | {len(d):6d} | {sum(d) / len(d) / 1e3:8.1f} | "
            f"{quantile(d, 0.9) / 1e3:8.1f} | {max(d) / 1e3:9.1f} | {thread[:24]} | {name}"
            for (name, thread), d in ranked
        ],
    )


def copies(db, thread_names, rows, window_start):
    kinds = {value: label for value, _, label in db.execute("select * from ENUM_CUDA_MEMCPY_OPER")}
    memory = {value: label for value, _, label in db.execute("select * from ENUM_CUDA_MEM_KIND")}
    callers = dict(db.execute("select correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"))
    groups = collections.defaultdict(lambda: [0, 0, 0])
    for start, end, size, kind, source, target, correlation in db.execute(
        "select start, end, bytes, copyKind, srcKind, dstKind, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where start >= ?",
        (window_start,),
    ):
        thread = thread_names.get(callers.get(correlation), "graph or unknown")
        group = groups[(kinds.get(kind, kind), memory.get(source, source), memory.get(target, target), thread)]
        group[0] += 1
        group[1] += size
        group[2] += end - start
    ranked = sorted(groups.items(), key=lambda item: -item[1][0])[:rows]
    print_rows(
        "Copies by kind, memory and calling thread",
        "count | MB | device ms | kind | source | target | thread",
        [
            f"{count:6d} | {size / 1e6:8.3f} | {time / NS_PER_MS:8.3f} | {kind} | {source} | {target} | {thread[:24]}"
            for (kind, source, target, thread), (count, size, time) in ranked
        ],
    )


def graphs_and_kernels(db, strings, thread_names, rows, window_start):
    callers = dict(db.execute("select correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"))
    graph_groups = collections.defaultdict(list)
    intervals = []
    if table_exists(db, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        for start, end, graph_id, stream, correlation in db.execute(
            "select start, end, graphId, streamId, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "where start >= ?",
            (window_start,),
        ):
            thread = thread_names.get(callers.get(correlation), "?")
            graph_groups[(graph_id, stream, thread)].append(end - start)
            intervals.append((start, end))
    else:
        pass
    print_rows(
        "CUDA graph replays (device span)",
        "total ms | count | mean us | p90 us | graph | stream | launching thread",
        [
            f"{sum(d) / NS_PER_MS:9.2f} | {len(d):6d} | {sum(d) / len(d) / 1e3:8.1f} | {quantile(d, 0.9) / 1e3:8.1f} | "
            f"{graph} | {stream} | {thread[:24]}"
            for (graph, stream, thread), d in sorted(graph_groups.items(), key=lambda item: -sum(item[1]))[:rows]
        ],
    )
    kernel_groups = collections.defaultdict(list)
    for start, end, stream, correlation, graph_node in db.execute(
        "select start, end, streamId, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL where start >= ?",
        (window_start,),
    ):
        origin = "graph node" if graph_node else "eager"
        thread = thread_names.get(callers.get(correlation), "graph" if graph_node else "?")
        kernel_groups[(origin, stream, thread)].append(end - start)
        intervals.append((start, end))
    print_rows(
        "Kernels by origin, stream and launching thread",
        "total ms | count | mean us | origin | stream | thread",
        [
            f"{sum(d) / NS_PER_MS:9.2f} | {len(d):7d} | {sum(d) / len(d) / 1e3:7.2f} | {origin} | {stream} | {thread[:24]}"
            for (origin, stream, thread), d in sorted(kernel_groups.items(), key=lambda item: -sum(item[1]))[:rows]
        ],
    )
    for start, end in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ?", (window_start,)):
        intervals.append((start, end))
    return intervals


def device_busy(intervals, window_start, window_end):
    busy = 0
    gaps = []
    cursor = window_start
    for start, end in sorted(intervals):
        if start > cursor:
            gaps.append(start - cursor)
        else:
            pass
        if end > cursor:
            busy += end - max(start, cursor)
            cursor = end
        else:
            pass
    window = window_end - window_start
    print("## Device busy (union of kernels, graph spans and copies)")
    print(f"window {window / NS_PER_MS:.1f} ms, busy {busy / NS_PER_MS:.1f} ms ({100 * busy / window:.1f} %)")
    for bound in GAP_BOUNDS_US:
        selected = [gap for gap in gaps if gap >= bound * 1e3]
        print(f"idle gaps >= {bound} us: {len(selected)}, {sum(selected) / NS_PER_MS:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--rows", type=int, default=30)
    args = parser.parse_args()
    db = sqlite3.connect(args.database)
    strings = dict(db.execute("select id, value from StringIds"))
    thread_names = {tid: strings.get(name_id, str(tid)) for name_id, tid in db.execute("select nameId, globalTid from ThreadNames")}
    # note (ratish): the graph trace also holds replays stamped before the capture opened; the CUDA API calls bound the window.
    window_start, window_end = db.execute(
        "select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    nvtx_ranges(db, strings, thread_names, args.rows)
    host_waits(db, strings, thread_names, args.rows)
    copies(db, thread_names, args.rows, window_start)
    intervals = graphs_and_kernels(db, strings, thread_names, args.rows, window_start)
    device_busy(intervals, window_start, window_end)


if __name__ == "__main__":
    main()
