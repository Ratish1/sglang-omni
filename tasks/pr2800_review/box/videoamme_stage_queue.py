"""run320: where Video-AMME c16 spends each request's time, main 6e75cba89 (upx) against the device interpolation
branch's final head 47315b507 (vdm), to place run319's c16 throughput (vdk 0.839 and 0.827 against main's 0.839 to
0.843 over four boots).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/videoamme_stage_queue.py
Video-AMME in CI stage 9's shape at c16 with request events recorded, both arms at the same time, four rounds with the
cards alternating; then stage_times_read.py prints every stage interval's mean per request for each arm; then vdm's
tests/unit_test/qwen3_omni on card 1. Writes /data/pr2800/run320 and run320.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import card_env, tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run320"
LOG = DATA / "run320.log"
EXPECTED_TREES = {"upx": "8c3034da", "vdm": "57c85a35"}
VIDEOAMME = ["--benchmark", "videoamme", "--passes", "c16", "--request-events"]
ROUNDS = ("upx@0,vdm@1", "vdm@0,upx@1", "upx@0,vdm@1", "vdm@0,upx@1")


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], name: str, tail: int) -> int:
    with open(OUT / f"{name}.log", "w") as handle:
        rc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    for line in (OUT / f"{name}.log").read_text().splitlines()[-tail:]:
        log(f"  {line}")
    log(f"{name} rc {rc}")
    return rc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    paths = {}
    for name, expected in EXPECTED_TREES.items():
        paths[name] = tree(name)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=paths[name], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=paths[name]).returncode
        log(f"{name} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: the tree is not the Mac's")
            return
        else:
            pass
    for index, arms in enumerate(ROUNDS, start=1):
        run_logged(
            [sys.executable, str(BOX / "videoamme_pair.py"), *VIDEOAMME, "--arms", arms, "--out",
             str(OUT / f"round{index}")],
            cwd=BOX, env=os.environ.copy(), name=f"round{index}", tail=3,
        )
    run_logged(
        [sys.executable, str(BOX / "stage_times_read.py"), "--run", str(OUT), "--arms", "upx,vdm", "--pass", "c16"],
        cwd=paths["upx"], env=card_env(0, paths["upx"]), name="stage_times", tail=60,
    )
    run_logged(
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(paths["vdm"]), "--card", "1", "--out",
         str(OUT / "units.log"), "tests/unit_test/qwen3_omni"],
        cwd=BOX, env=os.environ.copy(), name="units_driver", tail=1,
    )
    for line in (OUT / "units.log").read_text().splitlines()[-3:]:
        log(f"  units {line}")
    log("queue done")


if __name__ == "__main__":
    main()
