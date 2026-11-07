"""Read arm_cells.py runs labelled <model>_<arm>_<run>: per run and arm the cell's req/s, RTF and latency, then the
seeded c1 cell paired by prompt between arms and between an arm's own runs, and its byte identity.

usage: python3 rowwait_read.py [--root /data/rowwait_20261107/arms] [--pairs N-H,N-M,H-M]
"""

import argparse
import csv
import hashlib
import re
import statistics
from collections import defaultdict
from pathlib import Path

CAP_TOKENS = 2048
SUMMARY = {
    "req/s": re.compile(r"Throughput \(req/s\):\s+([\d.]+)"),
    "RTF": re.compile(r"RTF mean:\s+([\d.]+)"),
}


def results(cell: Path) -> dict[str, dict[str, str]]:
    path = cell / "bench" / "results.csv"
    if not path.exists():
        return {}
    else:
        pass
    with open(path, encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def run_line(cell: Path) -> str:
    log = (
        (cell / "bench.log").read_text(errors="replace")
        if (cell / "bench.log").exists()
        else ""
    )
    values = {key: pattern.findall(log) for key, pattern in SUMMARY.items()}
    rows = results(cell).values()
    done = [row for row in rows if row["is_success"] == "True"]
    latency = sorted(1000 * float(row["latency_s"]) for row in done)
    if not latency:
        return f"no results ({len(rows)} rows)"
    else:
        pass
    p95 = latency[min(len(latency) - 1, round(0.95 * (len(latency) - 1)))]
    return (
        f"{len(done)}/{len(rows)} ok, req/s {values['req/s'][-1] if values['req/s'] else '?'}, "
        f"RTF {values['RTF'][-1] if values['RTF'] else '?'}, latency mean {statistics.mean(latency):.1f} "
        f"p50 {statistics.median(latency):.1f} p95 {p95:.1f} ms"
    )


def latency_by_prompt(cells: list[Path]) -> dict[str, float]:
    """Mean latency per prompt over runs, on prompts every run finished below the cap."""
    tables = [results(cell) for cell in cells]
    keys = set.intersection(
        *(
            {
                key
                for key, row in table.items()
                if row["is_success"] == "True"
                and int(row["completion_tokens"]) < CAP_TOKENS
            }
            for table in tables
        )
    )
    return {
        key: statistics.mean(1000 * float(table[key]["latency_s"]) for table in tables)
        for key in keys
    }


def paired(first: dict[str, float], second: dict[str, float]) -> str:
    keys = first.keys() & second.keys()
    deltas = sorted(second[key] - first[key] for key in keys)
    if not deltas:
        return "no common prompts"
    else:
        pass
    lower, upper = deltas[len(deltas) // 10], deltas[(9 * len(deltas)) // 10]
    return (
        f"{len(keys)} prompts, median {statistics.median(deltas):+.2f}, mean {statistics.mean(deltas):+.2f} ms, "
        f"q10 {lower:+.1f} q90 {upper:+.1f}, later {100 * sum(d > 0 for d in deltas) / len(deltas):.0f} %"
    )


def hashes(cell: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((cell / "bench" / "audio").glob("*.wav"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/data/rowwait_20261107/arms")
    )
    parser.add_argument("--pairs", default="N-H,N-M,H-M")
    args = parser.parse_args()
    runs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for label in sorted(path.name for path in args.root.iterdir() if path.is_dir()):
        model, arm, _ = label.split("_")
        runs[model][arm].append(label)
    for model, arms in runs.items():
        cells = sorted(
            {
                cell.name
                for label in sum(arms.values(), [])
                for cell in (args.root / label).iterdir()
                if cell.is_dir()
            }
        )
        for cell in cells:
            print(f"== {model} {cell}")
            for arm, labels in sorted(arms.items()):
                for label in labels:
                    print(f"  {label}: {run_line(args.root / label / cell)}")
            if cell != "n1":
                continue
            else:
                pass
            by_arm = {
                arm: latency_by_prompt([args.root / label / cell for label in labels])
                for arm, labels in arms.items()
            }
            for pair in args.pairs.split(","):
                second, first = pair.split("-")
                if first in by_arm and second in by_arm:
                    print(
                        f"  paired {second} - {first}: {paired(by_arm[first], by_arm[second])}"
                    )
                else:
                    pass
            for arm, labels in sorted(arms.items()):
                if len(labels) >= 2:
                    one, two = (
                        latency_by_prompt([args.root / label / cell])
                        for label in labels[:2]
                    )
                    print(f"  same arm {labels[1]} - {labels[0]}: {paired(one, two)}")
                else:
                    pass
            heads = {
                label: hashes(args.root / label / cell)
                for labels in arms.values()
                for label in labels
            }
            reference = sorted(heads)
            for label in reference[1:]:
                same = sum(
                    heads[reference[0]].get(name) == digest
                    for name, digest in heads[label].items()
                )
                print(
                    f"  bytes {label} against {reference[0]}: {same} of {len(heads[label])} identical"
                )


if __name__ == "__main__":
    main()
