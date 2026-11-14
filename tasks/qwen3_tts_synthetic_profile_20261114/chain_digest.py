"""Kernel-by-kernel timeline of the code predictor graph from a node-level attribution database.

The predictor graph is the replayed graph whose kernels include PREDICTOR_MARKER. For each kernel
name in replay order: its duration and the gap from the previous kernel's end to its start
(negative when programmatic dependent launch overlaps them), averaged over replays.

Usage: python3 chain_digest.py <cell.sqlite> [--marker NAME]
"""

import argparse
import collections
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--marker", default="norm_qkv_rope_store_kernel")
    args = parser.parse_args()
    db = sqlite3.connect(args.database)
    strings = dict(db.execute("select id, value from StringIds"))
    window_start = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
    replays = collections.defaultdict(list)
    graph_of = {}
    for start, end, name_id, correlation, graph_id in db.execute(
        "select start, end, shortName, correlationId, graphId from CUPTI_ACTIVITY_KIND_KERNEL "
        "where start >= ? and graphId is not null and graphId != 0",
        (window_start,),
    ):
        replays[correlation].append((start, end, strings.get(name_id, "?")))
        graph_of[correlation] = graph_id
    predictor = [
        sorted(kernels)
        for kernels in replays.values()
        if any(args.marker in name for _, _, name in kernels)
    ]
    if not predictor:
        print("no predictor replay found")
        return
    else:
        pass
    count = len(predictor[0])
    predictor = [kernels for kernels in predictor if len(kernels) == count]
    spans = [kernels[-1][1] - kernels[0][0] for kernels in predictor]
    busy = [sum(end - start for start, end, _ in kernels) for kernels in predictor]
    gaps = [
        [kernels[i][0] - kernels[i - 1][1] for i in range(1, count)] for kernels in predictor
    ]
    positive = [sum(max(gap, 0) for gap in replay) for replay in gaps]
    replay_count = len(predictor)
    print(
        f"{replay_count} replays of {count} kernels: span {sum(spans) / replay_count / 1e3:.1f} us, "
        f"kernel time {sum(busy) / replay_count / 1e3:.1f} us, idle between kernels "
        f"{sum(positive) / replay_count / 1e3:.1f} us"
    )
    by_name = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for kernels, replay_gaps in zip(predictor, gaps):
        for index, (start, end, name) in enumerate(kernels):
            entry = by_name[name[:60]]
            entry[0] += 1
            entry[1] += end - start
            if index > 0:
                entry[2] += replay_gaps[index - 1]
                entry[3] += 1
            else:
                pass
    print("per replay: count | mean duration us | mean gap before us | total duration us | total gap us | kernel")
    for name, (seen, duration, gap, gapped) in sorted(by_name.items(), key=lambda item: -item[1][1]):
        per_replay = seen / replay_count
        print(
            f"{per_replay:6.1f} | {duration / seen / 1e3:7.2f} | {gap / max(gapped, 1) / 1e3:7.2f} | "
            f"{duration / replay_count / 1e3:8.1f} | {gap / replay_count / 1e3:8.1f} | {name}"
        )
    print("one codebook step in replay order (median replay), duration us / gap before us:")
    middle = predictor[replay_count // 2]
    step_start = next(i for i, (_, _, name) in enumerate(middle) if args.marker in name and i > count // 2)
    for index in range(step_start, min(step_start + 32, count)):
        start, end, name = middle[index]
        print(f"    {(end - start) / 1e3:6.2f} / {(start - middle[index - 1][1]) / 1e3:6.2f}  {name[:70]}")


if __name__ == "__main__":
    main()
