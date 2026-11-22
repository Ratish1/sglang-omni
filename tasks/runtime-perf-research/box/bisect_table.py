"""Part 2 bisect verdicts from q1123_b's same-time pairs: each arm against F per row, pair by pair, against the F/F band.

usage: python3 bisect_table.py <run dir> <arm> [<arm> ...]

Boots: ff_f0 and ff_f1 (F on both cards at once), then per arm <arm>_p1_f0 / <arm>_p1_x1 and <arm>_p2_x0 / <arm>_p2_f1.
A pair's delta is the arm's boot against F's boot run beside it. The band for a row is the F/F pair's absolute delta.
A row moves when both pairs move it the same way past the band: "worse" or "better" by the row's direction (QPS up is
better, every other row down is better), else "level". Lower is better for every row but QPS.
"""

import sys
from pathlib import Path

from body_table import ROWS, boot_values

CELLS = ("c16", "c64")


def delta(arm: dict[str, float], reference: dict[str, float], key: str) -> float:
    return (arm[key] / reference[key] - 1) * 100


def main() -> None:
    run_dir = Path(sys.argv[1])
    band = {
        cell: (boot_values(run_dir / "ff_f1", cell), boot_values(run_dir / "ff_f0", cell))
        for cell in CELLS
    }
    for arm in sys.argv[2:]:
        print(f"## {arm} against F (pair 1, pair 2, band, verdict)")
        for cell in CELLS:
            pairs = [
                (boot_values(run_dir / f"{arm}_p1_x1", cell), boot_values(run_dir / f"{arm}_p1_f0", cell)),
                (boot_values(run_dir / f"{arm}_p2_x0", cell), boot_values(run_dir / f"{arm}_p2_f1", cell)),
            ]
            for label, key, _, _ in ROWS:
                deltas = [delta(x, f, key) for x, f in pairs]
                width = abs(delta(*band[cell], key))
                sign = 1 if key == "throughput_qps" else -1
                if all(sign * value > width for value in deltas):
                    verdict = "better"
                elif all(-sign * value > width for value in deltas):
                    verdict = "worse"
                else:
                    verdict = "level"
                print(f"  {cell} {label}: {deltas[0]:+.1f} %, {deltas[1]:+.1f} %, band {width:.1f} %, {verdict}")


if __name__ == "__main__":
    main()
