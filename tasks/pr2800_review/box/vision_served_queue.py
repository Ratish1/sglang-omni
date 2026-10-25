"""MMMU and Video-AMME served, a base tree against an arm, two same-time pairs at once on the lease's four cards.

Run on the box:
    python3 /workspace/box/tasks/pr2800_review/box/vision_served_queue.py --run run321 --base ipr --arm itq
        --expected c05cbb4d,8b623e1d [--benchmarks mmmu-full,videoamme] [--config ...] [--model ...]
For each benchmark through videoamme_pair.py, c1 then c16, each pass on a fresh server: phase "pair" runs base@0 with
arm@1 and arm@2 with base@3 at the same time (each pair same-time, the arms' cards swapped between the two); phase
"aa" runs base@0 with base@1 and arm@2 with arm@3 (each tree against itself, the band). Both trees must carry the
Mac's sglang_omni tree hashes (--expected) and be clean. Writes /data/pr2800/<run>/<benchmark>_<phase>_<pair>/ and
/data/pr2800/<run>.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
LOG = DATA / "unset.log"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def run_pair(out: Path, benchmark: str, arms: str, config: str, model: str, passes: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log(f"{out.name} {arms}")
    rc = subprocess.run(
        [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", benchmark, "--arms", arms, "--passes",
         passes, "--config", config, "--model", model, "--out", str(out)],
        cwd=BOX, env=os.environ.copy(), stdout=open(out / "pair.log", "w"), stderr=subprocess.STDOUT,
    ).returncode
    for line in (out / "driver.log").read_text().splitlines():
        if "accuracy" in line or "rc" in line:
            log(f"  {out.name} {line}")
        else:
            pass
    log(f"{out.name} rc {rc}")


def main() -> None:
    global LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expected", required=True, help="base and arm sglang_omni tree hash prefixes")
    parser.add_argument("--benchmarks", default="mmmu-full,videoamme")
    parser.add_argument("--config", default=pair_check.CONFIG)
    parser.add_argument("--model", default=pair_check.MODEL)
    parser.add_argument(
        "--phases", default="pair,aa", help="pair runs every tree twice at once already; aa adds a second band"
    )
    parser.add_argument("--passes", default="c1,c16", help="videoamme_pair.py passes, each on a fresh server")
    args = parser.parse_args()
    out, LOG = DATA / args.run, DATA / f"{args.run}.log"
    out.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    # note (ratish): pair_check read the pins at import, before they were known; pinned() reads these.
    pair_check.CARD_CPUS = {int(card): card_cpus for card, card_cpus in (entry.split("=", 1) for entry in cpus.split(";"))}
    pair_check.NODE = node
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}; config {args.config}; model {args.model}")
    for arm, expected in zip((args.base, args.arm), args.expected.split(",")):
        path = tree(arm)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
        log(f"{arm} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    base, arm = args.base, args.arm
    phases = (
        ("pair", (("r1", f"{base}@0,{arm}@1"), ("r2", f"{arm}@2,{base}@3"))),
        ("aa", ((base, f"{base}@0,{base}b@1"), (arm, f"{arm}@2,{arm}b@3"))),
    )
    for phase, pairs in (entry for entry in phases if entry[0] in args.phases.split(",")):
        for benchmark in args.benchmarks.split(","):
            threads = [
                threading.Thread(
                    target=run_pair,
                    args=(out / f"{benchmark}_{phase}_{name}", benchmark, arms, args.config, args.model, args.passes),
                )
                for name, arms in pairs
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    log("queue done")


if __name__ == "__main__":
    main()
