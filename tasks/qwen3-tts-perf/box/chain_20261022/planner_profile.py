"""Where the vocoder's initial worker spends its time, from pyspy_sidecar.py's collapsed stacks: per arm, the samples on
the stacks under run_initial_worker, split into waiting for a batch (collect_async_batch), planning
(build_incremental_plan, and within it first_chunk_reference_frames and tail_frames), launching
(launch_incremental_group) and finishing (finish_initial_group), as sample counts and milliseconds at the record rate.

usage: PAIR_ROOT=<pairs> python3 planner_profile.py LABEL [LABEL ...] [--rate 100]
"""

import argparse
import os
from pathlib import Path

PARTS = (
    ("initial worker", "run_initial_worker"),
    ("waiting for a batch", "collect_async_batch"),
    ("run_initial_batch", "run_initial_batch"),
    ("planning", "build_incremental_plan"),
    ("reach", "first_chunk_reference_frames"),
    ("tail_frames walk", "tail_frames"),
    ("launching", "launch_incremental_group"),
    ("finishing", "finish_initial_group"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--rate", type=int, default=100)
    args = parser.parse_args()
    root = Path(os.environ["PAIR_ROOT"])
    for label in args.labels:
        for arm in ("a", "b"):
            path = root / label / arm / "pyspy.txt"
            if not path.exists():
                print(f"{label} {arm}: no profile")
                continue
            else:
                pass
            counts = {name: 0 for name, _ in PARTS}
            for line in path.read_text(errors="ignore").splitlines():
                stack, _, count = line.rpartition(" ")
                for name, function in PARTS:
                    if f"{function} (" in stack:
                        counts[name] += int(count)
                    else:
                        pass
            tree = (root / label / arm / "tree.txt").read_text().splitlines()[1][:9]
            print(f"{label} {arm} ({tree}): " + ", ".join(
                f"{name} {count} ({1000 * count / args.rate:.0f} ms)" for name, count in counts.items()
            ))


if __name__ == "__main__":
    main()
