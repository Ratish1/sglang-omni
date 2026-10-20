"""Per call GPU kernel time, kernel count and window per call of formal traces (10 calls each).
usage: python3 formal_summary.py <trace dir with case/formal/*.trace.json.gz>... --cases a b"""

import argparse
import gzip
import json
from pathlib import Path

CALLS = 10


def summarize(trace_path: Path) -> tuple[float, float, float]:
    with gzip.open(trace_path) as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    busy_us = sum(e["dur"] for e in kernels)
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    return busy_us / CALLS / 1e3, len(kernels) / CALLS, (end - start) / CALLS / 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    print("| case | " + " | ".join(f"{d} GPU ms, kernels, window ms" for d in args.dirs) + " |")
    for case in args.cases:
        cells = []
        for directory in args.dirs:
            traces = list(Path(directory, case, "formal").glob("*.trace.json.gz"))
            if traces:
                gpu_ms, kernels, window_ms = summarize(traces[0])
                cells.append(f"{gpu_ms:.2f}, {kernels:.0f}, {window_ms:.2f}")
            else:
                cells.append("n/a")
        print(f"| {case} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
