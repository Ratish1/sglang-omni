"""Video-MME served pair from one cell's two benchmark repeats: rep0 is the first pass (encoder cache cold), rep1 the
second (encoder cache hot). Per arm, each boot's value and the change of the arms' means; accuracy, unparsed and failed
answers per boot.

usage: python3 video_reps.py <root> <cell> <title> a1,a2 b1,b2
"""

import json
import statistics
import sys
from pathlib import Path

root, cell, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
arms = [sys.argv[4].split(","), sys.argv[5].split(",")]
ROWS = [("req/s", "throughput_qps", 3), ("latency mean (s)", "latency_mean_s", 2),
        ("latency median (s)", "latency_median_s", 2), ("latency p95 (s)", "latency_p95_s", 2),
        ("latency p99 (s)", "latency_p99_s", 2)]


def load(boot: str, rep: str) -> dict:
    return json.loads((root / boot / cell / "bench" / rep / "videomme_results.json").read_text())


print(f"| {title} | base | branch | change |")
print("|---|---|---|---|")
for rep, label in (("rep0", "first pass"), ("rep1", "second pass")):
    results = [[load(boot, rep) for boot in arm] for arm in arms]
    for name, key, digits in ROWS:
        values = [[r["speed"][key] for r in arm] for arm in results]
        means = [statistics.mean(v) for v in values]
        cells = [" / ".join(f"{x:.{digits}f}" for x in v) for v in values]
        print(f"| {label} {name} | {cells[0]} | {cells[1]} | {100 * (means[1] - means[0]) / means[0]:+.1f} % |")
    for arm, boots in zip(results, arms):
        for result, boot in zip(arm, boots):
            s = result["summary"]
            print(f"  {label} {boot}: accuracy {s['accuracy']:.2f}, unparsed {s.get('unparseable')}, failed {s['failed']}")
