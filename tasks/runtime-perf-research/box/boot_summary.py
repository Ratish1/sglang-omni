"""First audio, p99, latency and throughput per boot and pass from bench.log files, and each arm's mean over its boots.

usage: python3 boot_summary.py <run dir> <cell> arm=boot,boot arm=boot,boot ...
       cell is c16 or c64 (every pass directory starting with it is read: warm is skipped).

Boots of one arm alternate with the other arm's on the one card (a b b a), so the arm means cancel a linear drift.
"""

import re
import statistics
import sys
from pathlib import Path

FIELDS = {
    "ttfc_ms": r"TTFC mean \(s\):\s+([\d.]+)",
    "ttfc_p99_ms": r"TTFC p99 \(s\):\s+([\d.]+)",
    "latency_ms": r"Latency mean \(s\):\s+([\d.]+)",
    "rps": r"Throughput \(req/s\):\s+([\d.]+)",
}


def pass_logs(boot: Path, cell: str) -> list[Path]:
    logs = []
    for directory in sorted(boot.glob(f"{cell}_*")):
        if directory.name.endswith("_warm"):
            continue
        else:
            pass
        logs.extend(
            sorted(directory.glob("bench/rep*/bench.log"))
            or sorted(directory.glob("bench.log"))
        )
    return logs


def read_log(path: Path) -> list[dict[str, float]]:
    """One row per benchmark run in the log; a clean pass with repeats prints one summary per repeat."""
    text = path.read_text(errors="replace")
    found = {name: re.findall(pattern, text) for name, pattern in FIELDS.items()}
    rows = []
    for index in range(len(found["ttfc_ms"])):
        rows.append(
            {
                name: (
                    float(values[index])
                    if name == "rps"
                    else float(values[index]) * 1e3
                )
                for name, values in found.items()
                if index < len(values)
            }
        )
    return rows


def main() -> None:
    run_dir, cell = Path(sys.argv[1]), sys.argv[2]
    arm_means: dict[str, list[dict[str, float]]] = {}
    for spec in sys.argv[3:]:
        arm, boots = spec.split("=")
        arm_means[arm] = []
        for boot in boots.split(","):
            rows = [
                row
                for path in pass_logs(run_dir / boot, cell)
                for row in read_log(path)
            ]
            if not rows:
                print(f"{arm} {boot}: no passes")
                continue
            else:
                pass
            mean = {
                name: statistics.fmean(row[name] for row in rows if name in row)
                for name in FIELDS
            }
            arm_means[arm].append(mean)
            passes = " ".join(f"{row['ttfc_ms']:.1f}" for row in rows)
            print(
                f"{arm} {boot}: first audio by pass {passes} | mean {mean['ttfc_ms']:.1f} p99 {mean['ttfc_p99_ms']:.0f} "
                f"latency {mean['latency_ms']:.0f} req/s {mean['rps']:.3f}"
            )
    print()
    for arm, means in arm_means.items():
        if not means:
            continue
        else:
            pass
        values = [mean["ttfc_ms"] for mean in means]
        spread = (
            statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        )
        print(
            f"{arm:10s} first audio {statistics.fmean(values):8.2f} (se {spread:.2f})  p99 {statistics.fmean(m['ttfc_p99_ms'] for m in means):7.1f}  "
            f"latency {statistics.fmean(m['latency_ms'] for m in means):7.1f}  req/s {statistics.fmean(m['rps'] for m in means):.3f}  boots {len(means)}"
        )


if __name__ == "__main__":
    main()
