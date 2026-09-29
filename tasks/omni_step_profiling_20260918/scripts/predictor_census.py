"""Kernel by kernel census of the Qwen3-TTS code predictor's graph replays.

Needs a node trace (--cuda-graph-trace=node) of a server with the pipeline_nvtx probe,
ideally with OMNI_PIPE_LINES=capture so every node resolves to the op and line that
captured it. A replay is one cudaGraphLaunch inside an mr.predictor range; its batch
size is the enclosing sched.batch range's bs. Per batch size: replays, kernels per
replay, the sum of kernel time, the replay span and the idle time between nodes. For
the most common batch size: the kernels of the median replay in node order with the
gap before each, grouped by op and line, and each kernel's CTA count against the SM
count, the one occupancy fact a trace without GPU metrics carries.

usage: python predictor_census.py REPORT.sqlite [--sms 132] [--bench-log bench.log]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from nsys_metrics import bench_window

PUSH_POP = 59


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--bench-log")
    parser.add_argument("--sequence", type=int, default=75)
    parser.add_argument(
        "--bs", type=int, help="batch size to detail; default the most common"
    )
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    if args.bench_log:
        t0, t1 = bench_window(db, args.bench_log)
    else:
        t0, t1 = 0, 1 << 62

    predictor, batches = collections.defaultdict(list), collections.defaultdict(list)
    capture = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS "
        "where eventType = ? and end is not null order by start",
        (PUSH_POP,),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(("op ", "cap ")):
            capture[tid].append((start, end, label))
        elif t0 <= start <= t1 and label.startswith("mr.predictor"):
            predictor[tid].append((start, end))
        elif t0 <= start <= t1 and label.startswith("sched.batch"):
            match = re.search(r"bs=(\d+)", label)
            batches[tid].append((start, end, int(match.group(1)) if match else -1))
    capture_starts = {tid: [c[0] for c in items] for tid, items in capture.items()}

    predictor_starts = {tid: [r[0] for r in items] for tid, items in predictor.items()}
    batch_starts = {tid: [r[0] for r in items] for tid, items in batches.items()}

    def enclosing(items, starts, t):
        """The range containing t among one thread's non overlapping ranges."""
        index = bisect.bisect_right(starts, t) - 1
        if index >= 0 and items[index][0] <= t <= items[index][1]:
            return items[index]
        return None

    graph_launch = {i for i, v in strings.items() if v.startswith("cudaGraphLaunch")}
    replay_bs = {}
    for start, tid, corr, name_id in db.execute(
        "select start, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where start between ? and ?",
        (t0, t1),
    ):
        if name_id not in graph_launch or tid not in predictor:
            continue
        if enclosing(predictor[tid], predictor_starts[tid], start) is None:
            continue
        batch = enclosing(batches.get(tid, []), batch_starts.get(tid, []), start)
        replay_bs[corr] = batch[2] if batch else -1

    original, created = {}, {}
    for start, tid, node, orig in db.execute(
        "select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS"
    ):
        if orig is not None:
            original[node] = orig
        elif node not in created:
            created[node] = (start, tid)

    label_cache = {}

    def op_label(node):
        if node in label_cache:
            return label_cache[node]
        root = node
        while root in original:
            root = original[root]
        origin = created.get(root)
        label = "?"
        if origin is not None:
            start, tid = origin
            items, starts = capture.get(tid, []), capture_starts.get(tid, [])
            last = bisect.bisect_right(starts, start) - 1
            index = last
            best = None
            while index >= 0 and index > last - 60:
                s, e, text = items[index]
                if s <= start <= e and text.startswith("op "):
                    best = text
                    break
                index -= 1
            if best:
                match = re.match(r"op (\S+) @(\S+) <(\S+)>", best)
                label = (
                    f"{match.group(1)} {match.group(3).split('sglang_omni/')[-1]}"
                    if match
                    else best
                )
        label_cache[node] = label
        return label

    replays = collections.defaultdict(list)
    for start, end, corr, name_id, node, gx, gy, gz, bx, by, bz in db.execute(
        "select start, end, correlationId, demangledName, graphNodeId, gridX, gridY, "
        "gridZ, blockX, blockY, blockZ from CUPTI_ACTIVITY_KIND_KERNEL "
        "where graphNodeId is not null and start between ? and ?",
        (t0, t1),
    ):
        if corr in replay_bs:
            replays[corr].append(
                (
                    start,
                    end,
                    strings.get(name_id, str(name_id)),
                    node,
                    gx * gy * gz,
                    bx * by * bz,
                )
            )

    by_bs = collections.defaultdict(list)
    for corr, kernels in replays.items():
        kernels.sort()
        busy = sum(e - s for s, e, *_ in kernels)
        span = kernels[-1][1] - kernels[0][0]
        by_bs[replay_bs[corr]].append((span, busy, len(kernels), corr))
    print(f"predictor replays {len(replays)} in the window, SMs {args.sms}")
    print(
        f"{'bs':>4}{'replays':>9}{'kernels':>9}{'span us':>10}{'kernel us':>11}{'idle us':>9}{'idle %':>8}"
    )
    for bs in sorted(by_bs):
        rows = by_bs[bs]
        span = statistics.median(r[0] for r in rows) / 1e3
        busy = statistics.median(r[1] for r in rows) / 1e3
        count = statistics.median(r[2] for r in rows)
        print(
            f"{bs:>4}{len(rows):>9}{count:>9.0f}{span:>10.0f}{busy:>11.0f}{span - busy:>9.0f}{100 * (span - busy) / span:>7.1f}%"
        )

    bs = args.bs if args.bs in by_bs else max(by_bs, key=lambda b: len(by_bs[b]))
    rows = sorted(by_bs[bs])
    corr = rows[len(rows) // 2][3]
    kernels = replays[corr]
    print(
        f"\nmedian replay at bs {bs}: {len(kernels)} kernels, span {(kernels[-1][1] - kernels[0][0]) / 1e3:.0f} us"
    )
    groups = collections.defaultdict(lambda: [0, 0, 0, 0])
    for index, (s, e, name, node, ctas, threads) in enumerate(kernels):
        gap = s - kernels[index - 1][1] if index else 0
        key = (op_label(node), name[:48])
        groups[key][0] += 1
        groups[key][1] += e - s
        groups[key][2] += gap
        groups[key][3] += ctas
    total = sum(e - s for s, e, *_ in kernels)
    print(
        f"by op and kernel (per replay): count, kernel us, idle before us, mean CTAs, SM fill"
    )
    for (label, name), (count, busy, gap, ctas) in sorted(
        groups.items(), key=lambda kv: -kv[1][1] - kv[1][2]
    ):
        mean_ctas = ctas / count
        print(
            f"  {label[:60]:<60} {name:<48} {count:>4} {busy / 1e3:>8.1f} {gap / 1e3:>8.1f} "
            f"{mean_ctas:>7.0f} {min(1.0, mean_ctas / args.sms):>6.2f}"
        )
    fill = sum((e - s) * min(1.0, c / args.sms) for s, e, _, _, c, _ in kernels) / max(
        total, 1
    )
    print(f"  kernel time weighted SM fill (CTAs over SMs, capped at 1): {fill:.2f}")
    print(
        f"\nfirst {args.sequence} kernels of that replay in node order (us: gap before, duration, CTAs)"
    )
    for index, (s, e, name, node, ctas, threads) in enumerate(kernels[: args.sequence]):
        gap = s - kernels[index - 1][1] if index else 0
        print(
            f"  {index:>3} {gap / 1e3:>6.1f} {(e - s) / 1e3:>7.1f} {ctas:>6}  {op_label(node)[:58]:<58} {name[:60]}"
        )


if __name__ == "__main__":
    main()
