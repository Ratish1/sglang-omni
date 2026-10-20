# SPDX-License-Identifier: Apache-2.0
"""Item 1 pairs (main against float32 logits): what moves req/s in the short cells. Per arm and cell, from
speed_results.json: req/s, the wall time it implies, the mean latency and the req/s that Little's law gives at
the cell's concurrency (concurrency / mean latency, the rate while every slot is full), the output tokens,
the requests at the token cap and their tokens and latencies, and per sample the token difference between the
arms (same text and reference in both).

    python3 i1_cells_compare_c20261011.py /data/c20261011/i1
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

CONCURRENCY = {"c1": 1, "c16": 16, "c64": 64, "L16": 16}


def load(arm: Path, cell: str) -> dict:
    return json.loads((arm / cell / "speed_results.json").read_text())


def main() -> None:
    root = Path(sys.argv[1])
    for pair in ("a", "b"):
        arms = sorted((root / pair).glob("card*"))
        for cell in ("c1", "c16", "c64", "L16"):
            print(f"pair {pair} {cell}")
            per_arm = {}
            for arm in arms:
                data = load(arm, cell)
                summary, rows = data["summary"], data["per_request"]
                qps = summary["throughput_qps"]
                wall = summary["total_requests"] / qps
                latencies = [row["latency_s"] for row in rows]
                mean_latency = statistics.fmean(latencies)
                little = CONCURRENCY[cell] / mean_latency
                capped = [row for row in rows if row.get("finish_reason") == "length"]
                tokens = [row["completion_tokens"] for row in rows]
                per_arm[arm.name] = {row["id"]: row for row in rows}
                print(
                    f"  {arm.name:12s} req/s {qps:7.3f} wall {wall:6.1f} s | mean latency {mean_latency:6.3f} s"
                    f" little {little:7.3f} | tokens total {sum(tokens)} mean {statistics.fmean(tokens):.1f}"
                    f" p99 {sorted(tokens)[int(0.99 * len(tokens))]} max {max(tokens)}"
                    f" | sum latency {sum(latencies):8.1f} s"
                )
                print(
                    f"  {'':12s} capped {len(capped)}: "
                    + ", ".join(f"{row['completion_tokens']} tok {row['latency_s']:.1f} s" for row in capped)
                )
                slowest = sorted(rows, key=lambda row: row["latency_s"], reverse=True)[:5]
                print(
                    f"  {'':12s} slowest: "
                    + ", ".join(f"{row['completion_tokens']} tok {row['latency_s']:.1f} s" for row in slowest)
                )
            names = list(per_arm)
            if len(names) == 2:
                first, second = per_arm[names[0]], per_arm[names[1]]
                diffs = [second[key]["completion_tokens"] - first[key]["completion_tokens"] for key in first]
                same = sum(1 for diff in diffs if diff == 0)
                print(
                    f"  tokens {names[1]} minus {names[0]}: equal in {same} of {len(diffs)},"
                    f" mean {statistics.fmean(diffs):+.2f}, mean abs {statistics.fmean(abs(d) for d in diffs):.1f}"
                )


if __name__ == "__main__":
    main()
