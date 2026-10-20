"""Unit suites of one box tree on one card, pinned to the card's cores, as a detached job with a pgid file.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/suite_run.py --tree /workspace/sglang-omni/.tmp/wt/<tree> --card 0
    --out /data/pr2800/<run>/<name>.log tests/...
Prints the tree's sglang_omni hash first, so the run can be matched to the Mac's tree.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CARD_CPUS = {
    int(card): cpus
    for card, cpus in (
        entry.split("=", 1)
        for entry in os.environ.get("PR2800_CARD_CPUS", "0=56-63,168-175;1=64-71,176-183").split(";")
    )
}
NODE = os.environ.get("PR2800_NODE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree_hash = subprocess.run(
        ["git", "rev-parse", "HEAD:sglang_omni"], cwd=args.tree, capture_output=True, text=True
    ).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True).stdout.strip()
    with open(args.out, "w") as handle:
        handle.write(f"head {head} sglang_omni {tree_hash}\n")
        handle.flush()
        process = subprocess.Popen(
            ["numactl", f"--physcpubind={CARD_CPUS[args.card]}", f"--membind={NODE}", sys.executable, "-m", "pytest",
             "-q", "-p", "no:cacheprovider", "-rs", *args.tests],
            cwd=args.tree,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.card), "PYTHONPATH": str(args.tree)},
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        args.out.with_suffix(".pgid").write_text(str(process.pid))
        process.wait()
        handle.write(f"exit {process.returncode}\n")


if __name__ == "__main__":
    main()
