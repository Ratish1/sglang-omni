"""Compare sequential boots of baseline.py (alternating arms) on one cell: per boot the benchmark numbers of each
repeat and their mean, and the busiest server threads.

usage: python3 seq_report.py --cell c16_clean /data/runtime-perf/omp_a1 /data/runtime-perf/omp_b1 ...
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_report import SPEECH_FIELDS  # noqa: E402
from report import thread_names  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="c16_clean")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run in (Path(path) for path in args.runs):
        cell = run / args.cell
        identity = (run / "identity.txt").read_text()
        serve_env = next((line for line in identity.splitlines() if line.startswith("serve env")), "serve env")
        print(f"== {run.name} ({serve_env})")
        values = {}
        for path in sorted((cell / "bench").rglob("speed_results.json")):
            summary = json.loads(path.read_text())["summary"]
            for label, key, scale in SPEECH_FIELDS:
                values.setdefault(label, []).append(summary[key] * scale)
        for label, series in values.items():
            print(f"    {label:20s} mean {statistics.mean(series):9.2f}  per repeat " + " ".join(f"{v:.2f}" for v in series))
        cpu = json.loads((cell / "thread_cpu.json").read_text())
        window = json.loads((cell / "window.json").read_text())
        span = window["end"] - window["start"]
        names = thread_names(cell)
        rows = []
        for key, end in cpu["end"].items():
            start = cpu["start"].get(key, {"cpu_s": 0.0})["cpu_s"]
            rows.append(((end["cpu_s"] - start) / span, key.split("/")[0], names.get(key.split("/")[1], end["name"])))
        totals = {}
        for share, pid, _ in rows:
            totals[pid] = totals.get(pid, 0.0) + share
        print("    process CPU over the cell window: " + ", ".join(f"pid {pid} {100 * share:.0f}%"
                                                              for pid, share in sorted(totals.items(), key=lambda i: -i[1])))
        for share, pid, name in sorted(rows, reverse=True)[:8]:
            print(f"      {100 * share:5.0f}%  pid {pid} {name}")


if __name__ == "__main__":
    main()
