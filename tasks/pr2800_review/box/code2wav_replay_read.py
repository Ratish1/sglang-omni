"""Reader for gil_wait_check.py traces: code2wav's graph replays in serving, graph by graph, with the talker beside them.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/code2wav_replay_read.py /data/pr2800/<run>
Per arm, the code2wav thread's graph executions (CUPTI graph trace, the thread with the second most launches in the
talker process) grouped by executable graph; graphs are matched across arms by capture order (the rank of their
graph exec id), since every arm captures the same keys in the same order. Per graph: replays, the median device span,
and the median latency from the cudaGraphLaunch call's start to the replay's end. Then the replay-weighted means over
the graphs every arm replayed, so arms compare on the same graphs.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def process_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def read_arm(path: Path, talker_pid: int) -> dict[int, list[tuple[float, float]]]:
    connection = sqlite3.connect(path)
    launches: dict[int, int] = defaultdict(int)
    call_start: dict[int, tuple[int, int]] = {}
    for correlation, tid, start in connection.execute(
        "SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if process_of(tid) == talker_pid:
            launches[tid] += 1
            call_start[correlation] = (tid, start)
        else:
            pass
    code2wav = sorted(launches, key=lambda tid: -launches[tid])[1]
    by_exec: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for start, end, correlation, graph_exec in connection.execute(
        "SELECT start, end, correlationId, graphExecId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    ):
        caller = call_start.get(correlation)
        if caller is not None and caller[0] == code2wav:
            by_exec[graph_exec].append(((end - start) / 1e3, (end - caller[1]) / 1e3))
        else:
            pass
    ranks = {graph_exec: rank for rank, graph_exec in enumerate(sorted(by_exec))}
    return {ranks[graph_exec]: replays for graph_exec, replays in by_exec.items()}


def main() -> None:
    out = Path(sys.argv[1])
    arms = {}
    for record in json.loads((out / "summary.json").read_text()):
        arms[f"{record['arm']} (card {record['card']})"] = read_arm(
            out / record["arm"] / "trace.sqlite", record["pids"]["talker_ar"]
        )
    common = set.intersection(*(set(graphs) for graphs in arms.values()))
    print(f"graphs replayed by every arm: {len(common)}")
    for rank in sorted(common, key=lambda key: -sum(len(arms[arm][key]) for arm in arms))[:12]:
        cells = []
        for arm, graphs in arms.items():
            replays = graphs[rank]
            cells.append(f"{arm}: {len(replays):4d} x span {statistics.median(s for s, _ in replays):7.1f} "
                         f"latency {statistics.median(l for _, l in replays):7.1f}")
        print(f"  graph {rank:3d}  " + " | ".join(cells))
    for arm, graphs in arms.items():
        replays = [replay for rank in common for replay in graphs[rank]]
        print(f"{arm}: {len(replays)} replays on common graphs, mean span {statistics.fmean(s for s, _ in replays):.1f} "
              f"us, mean launch-to-end {statistics.fmean(l for _, l in replays):.1f} us, summed span "
              f"{sum(s for s, _ in replays) / 1e3:.1f} ms")


if __name__ == "__main__":
    main()
