"""run321: the #3131 review fixes' measurement head (pam) and the device interpolation's crossover on this GPU.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/packed_metadata_queue.py
At the same time: pam's tests/unit_test/qwen3_omni on card 0 (suite_run.py, accelerator cases included; the new
public-encoder cases carry tolerance 0 so their failures report the measured differences), and
vision_interp_crossover.py from the device interpolation branch (vdm) on card 1 over run316's MMMU calls.
Writes /data/pr2800/run321 and run321.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import card_env, pinned, tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run321"
LOG = DATA / "run321.log"
RUN316 = DATA / "run316"
EXPECTED_TREES = {"pam": "60db7677", "vdm": "57c85a35"}


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    # note (ratish): pair_check read the pins at import, before they were known; pinned() reads these.
    pair_check.CARD_CPUS = {int(card): card_cpus for card, card_cpus in (entry.split("=", 1) for entry in cpus.split(";"))}
    pair_check.NODE = node
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    paths = {}
    for arm, expected in EXPECTED_TREES.items():
        paths[arm] = tree(arm)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=paths[arm], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni", "tests"], cwd=paths[arm]).returncode
        log(f"{arm} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    units = subprocess.Popen(
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(paths["pam"]), "--card", "0", "--out",
         str(OUT / "units.log"), "tests/unit_test/qwen3_omni"],
        cwd=BOX, env=os.environ.copy(),
    )
    with open(OUT / "crossover.log", "w") as handle:
        rc = subprocess.run(
            pinned(1, [sys.executable, str(BOX / "vision_interp_crossover.py"), "--grids",
                       str(RUN316 / "mmmu" / "c1" / "upx" / "grids"), str(RUN316 / "mmmu" / "c16" / "upx" / "grids")]),
            cwd=paths["vdm"], env=card_env(1, paths["vdm"]), stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    for line in (OUT / "crossover.log").read_text().splitlines():
        if line.startswith(("same", "mixed", "Traceback")) or "distinct image rows" in line:
            log(f"  {line}")
        else:
            pass
    log(f"crossover rc {rc}")
    units.wait()
    for line in (OUT / "units.log").read_text().splitlines():
        if line.startswith(("FAILED", "ERROR", "E   ", "____", "head ")) or " passed" in line:
            log(f"  units {line}")
        else:
            pass
    log("queue done")


if __name__ == "__main__":
    main()
