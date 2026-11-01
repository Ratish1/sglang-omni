"""Talker decode cadence in one events cell per boot: decode steps from the probe's tb records inside the cell's
window, their rows and the gaps between consecutive steps (a gap holds the step itself plus any prefill between).

usage: python3 talker_cadence.py --cell c16_events <boot> [<boot> ...]
"""

import argparse
import glob
import json
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        with open(f"{boot}/{args.cell}/window.json", encoding="utf-8") as source:
            window = json.load(source)
        start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
        steps = []
        for path in glob.glob(f"{boot}/probe/*.jsonl"):
            with open(path, encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if (
                        isinstance(record, dict)
                        and record["k"] == "tb"
                        and start_ns <= record["ns"] <= end_ns
                    ):
                        steps.append((record["ns"], record["rows"]))
        steps.sort()
        gaps = sorted((b[0] - a[0]) / 1e6 for a, b in zip(steps, steps[1:]))
        rows = [row for _, row in steps]
        seconds = (end_ns - start_ns) / 1e9

        def share(low: float) -> float:
            return sum(1 for gap in gaps if gap > low) / len(gaps) * 100

        print(
            f"{boot}: {len(steps)} decode steps in {seconds:.1f} s ({len(steps) / seconds:.1f}/s), rows mean "
            f"{statistics.mean(rows):.2f}, gap p50 {gaps[len(gaps) // 2]:.1f} p90 {gaps[int(len(gaps) * 0.9)]:.1f} "
            f"p99 {gaps[int(len(gaps) * 0.99)]:.1f} ms, gaps over 20 ms {share(20):.1f} %, over 30 ms {share(30):.1f} %, "
            f"decode busy {sum(gaps) / seconds / 10:.1f} %"
        )


if __name__ == "__main__":
    main()
