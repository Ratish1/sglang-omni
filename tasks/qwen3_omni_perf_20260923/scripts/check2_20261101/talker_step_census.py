"""The talker's work per decode step on its own card, from a node-traced nsys window (check 2, 11-01).

Every kernel on --device is the talker's (the two-card speech default puts only talker_ar there). Graph node
kernels carry their cudaGraphLaunch's correlation id, so one id is one replay; replays are grouped by their
graph (graphId when the export has it, else the replay's kernel-name signature). Per graph: replays a second,
kernels a replay, span and kernel sum a replay (p50), and the kernel names with their count and median device
time a replay for the four graphs with the most kernel time. Kernels outside any graph are the eager work, by name.

usage: python3 talker_step_census.py REPORT.sqlite [--device 1] [--top 25]
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import Counter, defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("select id, value from StringIds"))
    columns = {row[1] for row in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    graph_column = "graphId" if "graphId" in columns else "null"
    rows = db.execute(
        f"select start, end, correlationId, graphNodeId, {graph_column}, shortName "
        "from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ? order by start",
        (args.device,),
    ).fetchall()
    if not rows:
        print(f"no kernels on device {args.device}")
        return
    else:
        pass
    window_s = (rows[-1][1] - rows[0][0]) / 1e9
    replays: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    replay_graph: dict[int, object] = {}
    eager: list[tuple[int, int, str]] = []
    for start, end, correlation, node, graph, short in rows:
        name = names.get(short, "?")
        if node is None:
            eager.append((start, end, name))
        else:
            replays[correlation].append((start, end, name))
            replay_graph[correlation] = graph
    groups: dict[object, list[int]] = defaultdict(list)
    for correlation, kernels in replays.items():
        graph = replay_graph[correlation]
        key = graph if graph is not None else tuple(sorted(Counter(k[2] for k in kernels).items()))
        groups[key].append(correlation)
    busy_s = sum(end - start for start, end, *_ in rows) / 1e9
    print(
        f"device {args.device}: window {window_s:.2f} s, {len(rows)} kernels, kernel time {busy_s:.2f} s "
        f"({busy_s / window_s:.1%} of the window, overlap counted twice), {len(replays)} graph replays in "
        f"{len(groups)} graphs, {len(eager)} eager kernels"
    )
    summaries = []
    for key, correlations in groups.items():
        counts = [len(replays[c]) for c in correlations]
        spans = [
            (max(e for _, e, _ in replays[c]) - min(s for s, _, _ in replays[c])) / 1e3
            for c in correlations
        ]
        sums = [sum(e - s for s, e, _ in replays[c]) / 1e3 for c in correlations]
        first_start = min(min(s for s, _, _ in replays[c]) for c in correlations)
        summaries.append((key, correlations, counts, spans, sums, first_start))
    summaries.sort(key=lambda item: -len(item[1]) * statistics.median(item[4]))
    print(f"{'graph':>6} {'replays':>7} {'/s':>6} {'kernels':>7} {'span us':>8} {'sum us':>8} {'share of kernel time':>20}")
    for index, (key, correlations, counts, spans, sums, _) in enumerate(summaries):
        share = sum(sums) / 1e6 / busy_s
        print(
            f"{index:6d} {len(correlations):7d} {len(correlations) / window_s:6.1f} "
            f"{statistics.median(counts):7.0f} {statistics.median(spans):8.0f} {statistics.median(sums):8.0f} {share:20.1%}"
        )
    for index, (key, correlations, counts, spans, sums, _) in enumerate(summaries[:4]):
        per_name: dict[str, list[float]] = defaultdict(list)
        for correlation in correlations:
            for start, end, name in replays[correlation]:
                per_name[name].append((end - start) / 1e3)
        replay_count = len(correlations)
        table = sorted(
            (
                (statistics.median(d) * len(d) / replay_count, len(d) / replay_count, statistics.median(d), name)
                for name, d in per_name.items()
            ),
            key=lambda row: -row[0],
        )
        total = sum(row[0] for row in table)
        print(f"\ngraph {index}: {replay_count} replays, kernels per replay {statistics.median(counts):.0f}, median-based sum {total:.0f} us")
        print(f"{'us/replay':>10} {'share':>6} {'n/replay':>8} {'us each':>8}  kernel")
        for us, n, each, name in table[: args.top]:
            print(f"{us:10.1f} {us / total:6.1%} {n:8.1f} {each:8.2f}  {name[:110]}")
    eager_by_name: dict[str, list[float]] = defaultdict(list)
    for start, end, name in eager:
        eager_by_name[name].append((end - start) / 1e3)
    eager_total = sum(sum(d) for d in eager_by_name.values())
    print(f"\neager kernels: {len(eager)} in {window_s:.2f} s ({len(eager) / window_s:.0f} a second), {eager_total / 1e3:.1f} ms")
    print(f"{'count':>7} {'/s':>7} {'us each':>8} {'ms total':>9}  kernel")
    for name, d in sorted(eager_by_name.items(), key=lambda item: -sum(item[1]))[: args.top]:
        print(f"{len(d):7d} {len(d) / window_s:7.1f} {statistics.median(d):8.2f} {sum(d) / 1e3:9.2f}  {name[:110]}")


if __name__ == "__main__":
    main()
