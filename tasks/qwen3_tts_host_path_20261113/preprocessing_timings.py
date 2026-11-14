"""Per-call wall and thread CPU time of each probed preprocessing step, base against head, per order.

Reads <root>/order<k>/<side>/probe/timings_*.json written by the boot probe; the first calls of each
step (warmup and the opening burst) are dropped.

usage: python3 preprocessing_timings.py --root /data/.../events_c64_timed
"""

import argparse
import json
import statistics
from pathlib import Path


def calls_by_step(probe: Path) -> dict[str, list[tuple[float, int, int]]]:
    merged: dict[str, list[tuple[float, int, int]]] = {}
    for path in probe.glob("timings_*.json"):
        for name, calls in json.loads(path.read_text()).items():
            merged.setdefault(name, []).extend(tuple(call) for call in calls)
    return merged


def summary(
    calls: list[tuple[float, int, int]], skip: int
) -> tuple[int, float, float, float, float]:
    kept = sorted(calls)[skip:]
    walls = sorted(call[1] / 1e6 for call in kept)
    cpus = [call[2] / 1e6 for call in kept]
    return (
        len(kept),
        statistics.fmean(walls),
        walls[len(walls) // 2],
        walls[int(len(walls) * 0.9)],
        statistics.fmean(cpus),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--skip", type=int, default=65)
    arguments = parser.parse_args()
    for order_directory in sorted(Path(arguments.root).glob("order*")):
        steps = {
            side: calls_by_step(order_directory / side / "probe")
            for side in ("base", "head")
        }
        print(order_directory.name)
        print(
            f"  {'step':32s}{'side':>6s}{'calls':>7s}{'wall mean':>11s}{'wall p50':>10s}"
            f"{'wall p90':>10s}{'cpu mean':>10s}"
        )
        for name in sorted(steps["base"]):
            for side in ("base", "head"):
                if name not in steps[side]:
                    continue
                else:
                    pass
                count, mean, median, p90, cpu = summary(
                    steps[side][name], arguments.skip
                )
                print(
                    f"  {name:32s}{side:>6s}{count:7d}{mean:11.2f}{median:10.2f}{p90:10.2f}{cpu:10.2f}"
                )


if __name__ == "__main__":
    main()
