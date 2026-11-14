"""Kernel digest of one attribution cell database (node-level CUDA graph trace).

For each CUDA graph replayed in the window: the marked range its launch sat in, kernels per replay,
device time per replay, the share of time in kernels under 5 us, the idle time between a replay's
kernels, and its top kernels. Eager kernels are grouped by the marked range that launched them.

Usage: python3 kernel_digest.py <cell.sqlite> [--rows N]
"""

import argparse
import bisect
import collections
import sqlite3
from pathlib import Path

MARKED_DOMAINS = {"talker", "vocoder", "preprocessing", "capture"}
SMALL_KERNEL_NS = 5000
NS_PER_MS = 1e6
WAIT_APIS = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy")


class MarkedRanges:
    """Innermost marked NVTX range on a thread at a time."""

    def __init__(self, db: sqlite3.Connection, strings: dict[int, str]) -> None:
        domains = {
            domain_id: text
            for domain_id, text in db.execute("select domainId, text from NVTX_EVENTS where eventType = 75")
        }
        per_thread = collections.defaultdict(list)
        for start, end, text, text_id, domain_id, tid in db.execute(
            "select start, end, text, textId, domainId, globalTid from NVTX_EVENTS "
            "where eventType in (59, 60) and end is not null"
        ):
            domain = domains.get(domain_id)
            if domain in MARKED_DOMAINS:
                name = text if text is not None else strings.get(text_id, "?")
                per_thread[tid].append((start, end, f"{domain}:{'.'.join(name.split('.')[-2:])}"))
            else:
                pass
        self.ranges = {tid: sorted(rows) for tid, rows in per_thread.items()}
        self.starts = {tid: [row[0] for row in rows] for tid, rows in self.ranges.items()}

    def at(self, tid: int, time: int) -> str:
        rows = self.ranges.get(tid)
        if not rows:
            return "unmarked"
        else:
            pass
        index = bisect.bisect_right(self.starts[tid], time) - 1
        for candidate in range(index, max(-1, index - 400), -1):
            start, end, name = rows[candidate]
            if start <= time <= end:
                return name
            else:
                pass
        return "unmarked"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()
    db = sqlite3.connect(args.database)
    strings = dict(db.execute("select id, value from StringIds"))
    window_start = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
    launches = {
        correlation: (start, tid)
        for correlation, start, tid in db.execute(
            "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    marks = MarkedRanges(db, strings)
    graph_kernels = collections.defaultdict(list)
    eager = collections.defaultdict(lambda: collections.defaultdict(list))
    for start, end, name_id, correlation, graph_id in db.execute(
        "select start, end, shortName, correlationId, graphId from CUPTI_ACTIVITY_KIND_KERNEL where start >= ?",
        (window_start,),
    ):
        if graph_id:
            graph_kernels[graph_id].append((correlation, start, end, strings.get(name_id, "?")))
        else:
            launch = launches.get(correlation)
            site = "unknown" if launch is None else marks.at(launch[1], launch[0])
            eager[site][strings.get(name_id, "?")].append(end - start)
    print("## CUDA graphs (node-level)")
    for graph_id, kernels in sorted(graph_kernels.items(), key=lambda item: -sum(k[2] - k[1] for k in item[1])):
        replays = collections.defaultdict(list)
        for correlation, start, end, name in kernels:
            replays[correlation].append((start, end, name))
        replay_count = len(replays)
        busy = sum(end - start for _, start, end, _ in kernels)
        span = sum(max(e for _, e, _ in rows) - min(s for s, _, _ in rows) for rows in replays.values())
        small = sum(end - start for _, start, end, _ in kernels if end - start < SMALL_KERNEL_NS)
        first = next(iter(replays))
        launch = launches.get(first)
        site = "unknown" if launch is None else marks.at(launch[1], launch[0])
        print(
            f"graph {graph_id}: {replay_count} replays, {len(kernels) / replay_count:.0f} kernels a replay, "
            f"kernel time {busy / replay_count / 1e3:.1f} us a replay, span {span / replay_count / 1e3:.1f} us, "
            f"idle inside {100 * (span - busy) / max(span, 1):.1f} %, kernels under 5 us {100 * small / max(busy, 1):.1f} % of time, "
            f"launched in {site}"
        )
        by_name = collections.defaultdict(list)
        for _, start, end, name in kernels:
            by_name[name].append(end - start)
        for name, durations in sorted(by_name.items(), key=lambda item: -sum(item[1]))[: args.rows]:
            print(
                f"    {100 * sum(durations) / busy:5.1f} % | {len(durations) / replay_count:6.1f} a replay | "
                f"{sum(durations) / len(durations) / 1e3:7.2f} us | {name[:100]}"
            )
    print("## Host waits and copies by marked call site")
    waits = collections.defaultdict(list)
    for start, end, name_id, tid in db.execute(
        "select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ?",
        (window_start,),
    ):
        name = strings.get(name_id, "?").split("_v")[0]
        if name.startswith(WAIT_APIS):
            waits[(name, marks.at(tid, start))].append(end - start)
        else:
            pass
    for (name, site), durations in sorted(waits.items(), key=lambda item: -sum(item[1]))[: 2 * args.rows]:
        print(
            f"{sum(durations) / NS_PER_MS:8.2f} ms | {len(durations):6d} | mean {sum(durations) / len(durations) / 1e3:7.1f} us | "
            f"max {max(durations) / 1e3:8.1f} us | {name} | {site}"
        )
    kinds = {value: label for value, _, label in db.execute("select * from ENUM_CUDA_MEMCPY_OPER")}
    memory = {value: label for value, _, label in db.execute("select * from ENUM_CUDA_MEM_KIND")}
    copy_sites = collections.defaultdict(lambda: [0, 0])
    for size, kind, source, target, correlation in db.execute(
        "select bytes, copyKind, srcKind, dstKind, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ?",
        (window_start,),
    ):
        launch = launches.get(correlation)
        site = "graph or unknown" if launch is None else marks.at(launch[1], launch[0])
        group = copy_sites[(f"{kinds.get(kind)} {memory.get(source)}->{memory.get(target)}", site)]
        group[0] += 1
        group[1] += size
    for (kind, site), (count, size) in sorted(copy_sites.items(), key=lambda item: -item[1][0])[: 2 * args.rows]:
        print(f"copies {count:6d} | {size / 1e6:9.3f} MB | {kind} | {site}")
    print("## Eager kernels by marked launch site")
    for site, names in sorted(eager.items(), key=lambda item: -sum(sum(d) for d in item[1].values()))[: args.rows]:
        total = sum(sum(d) for d in names.values())
        count = sum(len(d) for d in names.values())
        print(f"{total / NS_PER_MS:8.2f} ms | {count:7d} kernels | {site}")
        for name, durations in sorted(names.items(), key=lambda item: -sum(item[1]))[:5]:
            print(f"        {sum(durations) / NS_PER_MS:7.2f} ms | {len(durations):6d} | {sum(durations) / len(durations) / 1e3:6.2f} us | {name[:90]}")


if __name__ == "__main__":
    main()
