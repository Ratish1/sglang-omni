"""MMSU served pair as table rows: per load and measured pass (clean, events), each arm's boot values and the change of
the arms' means; accuracy per boot.

usage: python3 mmsu_pairs.py <root> <title> <base boots a1,a2> <branch boots b1,b2> [loads c96,c16]
"""

import json
import statistics
import sys
from pathlib import Path

root, title = Path(sys.argv[1]), sys.argv[2]
arms = [sys.argv[3].split(","), sys.argv[4].split(",")]
loads = sys.argv[5].split(",") if len(sys.argv) > 5 else ["c96", "c16"]
ROWS = [("req/s", "throughput_qps", 2), ("latency mean (ms)", "latency_mean_s", 0)]


def result(boot: str, cell: str) -> dict:
    return json.loads((root / boot / cell / "bench" / "mmsu_results.json").read_text())


print(f"| {title} | base | branch | change |\n|---|---|---|---|")
for load in loads:
    for measured in ("clean", "events"):
        cell = f"{load}_{measured}"
        for label, key, digits in ROWS:
            scale = 1000 if key.endswith("_s") else 1
            values = [
                [result(boot, cell)["speed_metrics"][key] * scale for boot in arm]
                for arm in arms
            ]
            means = [statistics.mean(arm) for arm in values]
            cells = [
                " / ".join(f"{value:.{digits}f}" for value in arm) for arm in values
            ]
            print(
                f"| {load} {measured} {label} | {cells[0]} | {cells[1]} | "
                f"{(means[1] / means[0] - 1) * 100:+.1f} % |"
            )
for arm in arms:
    for boot in arm:
        accuracies = [
            result(boot, f"{load}_{measured}")["summary"]["overall_accuracy"]
            for load in loads
            for measured in ("clean", "events")
        ]
        print(f"  {boot}: accuracy {' '.join(f'{value:.4f}' for value in accuracies)}")
