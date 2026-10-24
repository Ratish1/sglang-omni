"""run316: the vision position interpolation forms on the vision encoder calls MMMU and Video-AMME make on main
6e75cba89 (upx).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/vision_interp_queue.py
1. MMMU (CI stage 3's shape, upx card 0) and Video-AMME (CI stage 9's shape, upxb card 1) at the same time, c1 then
   c16, each server recording every vision encoder call's grids (videoamme_pair.py --encoder-grids).
2. vision_interp_probe.py on card 0 over the four recorded sets.
Writes /data/pr2800/run316 and run316.log.
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
OUT = DATA / "run316"
LOG = DATA / "run316.log"
EXPECTED_TREE = "8c3034da"


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
    path = tree("upx")
    tree_hash = subprocess.run(["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
    log(f"upx sglang_omni {tree_hash} expected {EXPECTED_TREE} dirty {dirty}")
    if not tree_hash.startswith(EXPECTED_TREE) or dirty:
        log("stopped: the tree is not the Mac's")
        return
    else:
        pass
    served = {
        "mmmu": subprocess.Popen(
            [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", "mmmu", "--arms", "upx@0", "--passes",
             "c1,c16", "--encoder-grids", "--out", str(OUT / "mmmu")],
            cwd=BOX, env=os.environ.copy(), stdout=open(OUT / "mmmu.log", "w"), stderr=subprocess.STDOUT,
        ),
        "videoamme": subprocess.Popen(
            [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", "videoamme", "--arms", "upxb@1",
             "--passes", "c1,c16", "--encoder-grids", "--out", str(OUT / "videoamme")],
            cwd=BOX, env=os.environ.copy(), stdout=open(OUT / "videoamme.log", "w"), stderr=subprocess.STDOUT,
        ),
    }
    for name, process in served.items():
        process.wait()
        for line in (OUT / f"{name}.log").read_text().splitlines()[-6:]:
            log(f"  {line}")
        log(f"{name} rc {process.returncode}")
    calls = [
        f"{benchmark}_{pass_name}={OUT / benchmark / pass_name / arm / 'grids'}"
        for benchmark, arm in (("mmmu", "upx"), ("videoamme", "upxb"))
        for pass_name in ("c1", "c16")
    ]
    with open(OUT / "probe.log", "w") as handle:
        rc = subprocess.run(
            pinned(0, [sys.executable, str(BOX / "vision_interp_probe.py"), "--calls", *calls]),
            cwd=path, env=card_env(0, path), stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    for line in (OUT / "probe.log").read_text().splitlines():
        if line.startswith(("==", "table", "legacy", "loop", "vectorized", "dispatch", "hf519", "Traceback")):
            log(f"  {line}")
        else:
            pass
    log(f"probe rc {rc}")
    log("queue done")


if __name__ == "__main__":
    main()
