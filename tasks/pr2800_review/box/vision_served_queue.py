"""MMMU and Video-AMME served, main 6e75cba89 (upx) against the device position interpolation branch.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/vision_served_queue.py --run run318 --arm vdk
For each benchmark (MMMU in CI stage 3's shape, Video-AMME in CI stage 9's shape), three same-time rounds on the two
cards through videoamme_pair.py, c1 then c16, each pass on a fresh server: upx@0 with the arm on card 1, then
swapped, then upx@0 with upxb@1 (main against main, the band). Writes /data/pr2800/<run>/<benchmark>_<round> and
<run>.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
EXPECTED_TREES = {"upx": "8c3034da", "vdk": "bf85c3b1"}
OUT = DATA
LOG = DATA / "unset.log"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    global OUT, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(set(EXPECTED_TREES) - {"upx"}))
    args = parser.parse_args()
    OUT, LOG = DATA / args.run, DATA / f"{args.run}.log"
    rounds = (("r1", f"upx@0,{args.arm}@1"), ("r2", f"{args.arm}@0,upx@1"), ("aa", "upx@0,upxb@1"))
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    # note (ratish): pair_check read the pins at import, before they were known; pinned() reads these.
    pair_check.CARD_CPUS = {int(card): card_cpus for card, card_cpus in (entry.split("=", 1) for entry in cpus.split(";"))}
    pair_check.NODE = node
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    for arm in ("upx", args.arm):
        path = tree(arm)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
        log(f"{arm} sglang_omni {tree_hash} expected {EXPECTED_TREES[arm]} dirty {dirty}")
        if not tree_hash.startswith(EXPECTED_TREES[arm]) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    for benchmark in ("mmmu", "videoamme"):
        for round_name, arms in rounds:
            run_out = OUT / f"{benchmark}_{round_name}"
            run_out.mkdir(parents=True, exist_ok=True)
            log(f"{benchmark} {round_name} {arms}")
            rc = subprocess.run(
                [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", benchmark, "--arms", arms, "--passes",
                 "c1,c16", "--out", str(run_out)],
                cwd=BOX, env=os.environ.copy(), stdout=open(run_out / "pair.log", "w"), stderr=subprocess.STDOUT,
            ).returncode
            for line in (run_out / "driver.log").read_text().splitlines():
                if "accuracy" in line or "rc" in line:
                    log(f"  {line}")
                else:
                    pass
            log(f"{benchmark} {round_name} rc {rc}")
    log("queue done")


if __name__ == "__main__":
    main()
