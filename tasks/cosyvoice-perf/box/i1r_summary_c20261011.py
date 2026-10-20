# SPDX-License-Identifier: Apache-2.0
"""Item 1 repeats (queue_i1r_c20261011.sh): per tree and concurrency, req/s and mean latency of every run
(three per server, two servers per tree, one per card), their mean and spread, and the same time delta of
item 1 against main per repeat (pair a and pair b each run main and item 1 at the same time on the two
cards).

    python3 i1r_summary_c20261011.py /data/c20261011/i1r
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    runs: dict[tuple[str, str], list[tuple[str, str, float, float, int]]] = {}
    same_time: dict[str, list[float]] = {}
    for pair in ("a", "b"):
        by_tree = {}
        for arm in sorted((root / pair).glob("card*")):
            tree = arm.name.split("_", 1)[1]
            by_tree[tree] = arm
            for cell in sorted(path.name for path in arm.iterdir() if path.is_dir()):
                summary = json.loads((arm / cell / "speed_results.json").read_text())["summary"]
                runs.setdefault((tree, cell[:3]), []).append(
                    (pair, cell, summary["throughput_qps"], summary["latency_mean_s"], summary["max_token_hits"])
                )
        cells = sorted(path.name for path in by_tree["t_m2"].iterdir() if path.is_dir())
        for cell in cells:
            main_qps = json.loads((by_tree["t_m2"] / cell / "speed_results.json").read_text())["summary"]
            item_qps = json.loads((by_tree["t_i1"] / cell / "speed_results.json").read_text())["summary"]
            delta = 100 * (item_qps["throughput_qps"] / main_qps["throughput_qps"] - 1)
            same_time.setdefault(cell[:3], []).append(delta)
            print(f"pair {pair} {cell}: main {main_qps['throughput_qps']:.3f} item 1 {item_qps['throughput_qps']:.3f}"
                  f" ({delta:+.1f} %), capped {main_qps['max_token_hits']} / {item_qps['max_token_hits']}")
    print()
    for (tree, concurrency), rows in sorted(runs.items()):
        qps = [row[2] for row in rows]
        print(
            f"{tree} {concurrency}: req/s mean {statistics.fmean(qps):.3f} sd {statistics.stdev(qps):.3f}"
            f" ({100 * statistics.stdev(qps) / statistics.fmean(qps):.1f} %) min {min(qps):.3f} max {max(qps):.3f}"
            f" over {len(qps)} runs"
        )
    for concurrency, deltas in sorted(same_time.items()):
        print(
            f"{concurrency} same time delta of item 1: mean {statistics.fmean(deltas):+.2f} %"
            f" sd {statistics.stdev(deltas):.2f} % over {len(deltas)}: "
            + ", ".join(f"{delta:+.1f}" for delta in deltas)
        )


if __name__ == "__main__":
    main()
