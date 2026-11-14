"""Per-call wall and thread CPU time of each probed preprocessing step, base against head, per order.

Reads <root>/order<k>/<side>/probe/timings_*.json written by the boot probe. With --cell, only calls
inside that cell's admission window (from its request events) count. The first calls of each step
(warmup and the opening burst) are dropped.

usage: python3 preprocessing_timings.py --root /data/.../events_c64_timed [--cell stream_c64]
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

GIL_WAIT = "gil_wait"
STEADY_REFERENCE = "preprocess_qwen3_tts_payload"
ADMISSION = "request_admission"


def calls_by_step(probe: Path) -> dict[str, list[tuple[float, int, int]]]:
    merged: dict[str, list[tuple[float, int, int]]] = {}
    for path in probe.glob("timings_*.json"):
        for name, calls in json.loads(path.read_text()).items():
            merged.setdefault(name, []).extend(tuple(call) for call in calls)
    return merged


def admission_window(cell_directory: Path) -> tuple[float, float]:
    stamps = []
    for path in glob.glob(str(cell_directory / "events" / "*.jsonl")):
        with open(path) as source:
            for line in source:
                record = json.loads(line)
                if record.get("event_name") == ADMISSION:
                    stamps.append(record["timestamp_ns"] / 1e9)
                else:
                    pass
    return min(stamps), max(stamps) + 1.0


def in_window(
    calls: list[tuple[float, int, int]], window: tuple[float, float] | None
) -> list[tuple[float, int, int]]:
    if window is None:
        return calls
    else:
        return [call for call in calls if window[0] <= call[0] <= window[1]]


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
    parser.add_argument("--cell", default=None)
    parser.add_argument("--skip", type=int, default=65)
    arguments = parser.parse_args()
    for order_directory in sorted(Path(arguments.root).glob("order*")):
        windows = {
            side: (
                None
                if arguments.cell is None
                else admission_window(order_directory / side / arguments.cell)
            )
            for side in ("base", "head")
        }
        steps = {
            side: {
                name: in_window(calls, windows[side])
                for name, calls in calls_by_step(
                    order_directory / side / "probe"
                ).items()
            }
            for side in ("base", "head")
        }
        print(order_directory.name)
        print(
            f"  {'step':32s}{'side':>6s}{'calls':>7s}{'wall mean':>11s}{'wall p50':>10s}"
            f"{'wall p90':>10s}{'cpu mean':>10s}"
        )
        for name in sorted(steps["base"]):
            if name == GIL_WAIT:
                continue
            else:
                pass
            for side in ("base", "head"):
                if not steps[side].get(name):
                    continue
                else:
                    pass
                count, mean, median, p90, cpu = summary(
                    steps[side][name], arguments.skip
                )
                print(
                    f"  {name:32s}{side:>6s}{count:7d}{mean:11.2f}{median:10.2f}{p90:10.2f}{cpu:10.2f}"
                )
        for side in ("base", "head"):
            if not steps[side].get(GIL_WAIT):
                continue
            else:
                pass
            calls = sorted(steps[side][STEADY_REFERENCE])[arguments.skip :]
            first, last = calls[0][0], calls[-1][0]
            seconds = [
                entry for entry in steps[side][GIL_WAIT] if first <= entry[0] <= last
            ]
            print(
                f"  gil wait after a 1 ms sleep, {side}: {len(seconds)} s, mean "
                f"{statistics.fmean(entry[1] for entry in seconds) / 1e3:.0f} us, p90 "
                f"{statistics.fmean(entry[2] for entry in seconds) / 1e3:.0f} us"
            )


if __name__ == "__main__":
    main()
