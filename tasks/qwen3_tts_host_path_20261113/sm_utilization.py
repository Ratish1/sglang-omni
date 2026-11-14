"""Mean SM utilization per arm from an events pair's nvidia-smi dmon samples, per order and cell.

Base runs on card k and head on card 1 - k in order k; samples with both cards idle are dropped.

usage: python3 sm_utilization.py --root /data/.../events_rate45
"""

import argparse
import statistics
from pathlib import Path


def samples(path: Path) -> dict[int, list[int]]:
    by_card: dict[int, list[int]] = {0: [], 1: []}
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 3:
            continue
        else:
            pass
        if not fields[2].isdigit():
            continue
        else:
            pass
        rows.append((fields[0], int(fields[1]), int(fields[2])))
    busy_times = {time for time, _, sm in rows if sm > 0}
    for time, card, sm in rows:
        if time in busy_times and card in by_card:
            by_card[card].append(sm)
        else:
            pass
    return by_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    arguments = parser.parse_args()
    for order_directory in sorted(Path(arguments.root).glob("order*")):
        order = int(order_directory.name.removeprefix("order"))
        for path in sorted(order_directory.glob("*_dmon.txt")):
            by_card = samples(path)
            base, head = by_card[order], by_card[1 - order]
            print(
                f"{order_directory.name} {path.name.removesuffix('_dmon.txt')}: SM % mean base "
                f"{statistics.fmean(base):.1f}, head {statistics.fmean(head):.1f} "
                f"({len(base)} s)"
            )


if __name__ == "__main__":
    main()
