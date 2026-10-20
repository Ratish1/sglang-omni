# The A/A pair's same time deltas (card 1 against card 0, one tree on both) per cell and pair, for every metric row,
# then the mean and spread of each row's delta over the c16 cells, which is what a same time pair resolves.
# usage: python3 aa_summary_c20261013.py <aa dir>
import json
import statistics
import sys
from pathlib import Path

ROWS = (
    ("throughput_qps", "req/s"),
    ("rtf_mean", "RTF"),
    ("audio_ttfp_mean_s", "first audio mean"),
    ("audio_ttfp_p95_s", "first audio p95"),
    ("inter_chunk_mean_s", "inter chunk mean"),
    ("inter_chunk_p95_s", "inter chunk p95"),
    ("inter_chunk_p99_s", "inter chunk p99"),
    ("latency_mean_s", "latency mean"),
    ("latency_p99_s", "latency p99"),
)


def summary(arm: Path, cell: str) -> dict:
    return json.loads((arm / cell / "speed_results.json").read_text())["summary"]


root = Path(sys.argv[1])
deltas: dict[str, list[float]] = {name: [] for _, name in ROWS}
for pair in sorted(path.name for path in root.glob("p*")):
    for cell in ("c16a", "c16b", "c16c", "c64"):
        card0 = summary(root / pair / "card0_t_m", cell)
        card1 = summary(root / pair / "card1_t_m", cell)
        parts = []
        for key, name in ROWS:
            delta = 100 * (card1[key] - card0[key]) / card0[key]
            parts.append(f"{name} {card0[key]:.4g}/{card1[key]:.4g} ({delta:+.1f} %)")
            if cell != "c64":
                deltas[name].append(delta)
        print(f"{pair} {cell}: " + ", ".join(parts))
        print(f"    failed {card0['failed_requests']}/{card1['failed_requests']}, token cap {card0['max_token_hits']}/{card1['max_token_hits']}")
print("c16 same time delta (card 1 against card 0), mean and sd over the six c16 cells:")
for _, name in ROWS:
    values = deltas[name]
    print(f"    {name}: mean {statistics.mean(values):+.2f} %, sd {statistics.stdev(values):.2f} %, range {min(values):+.1f} to {max(values):+.1f} %")
