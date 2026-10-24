"""run313: whether the stacked q, k, v projection (64f8eeea9, vpq) is what slows MMSU c16 against main, measured
against the branch without it (e4bf36e88, e4b), and e4b against main 6e75cba89 (upx).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/stacking_isolation_queue.py
MMSU in CI stage 5's shape at c16, no request events, two arms at the same time with the cards alternating: four rounds
of e4b against vpq, then two of upx against e4b. Writes /data/pr2800/run313 and run313.log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run313"
LOG = DATA / "run313.log"
EXPECTED_TREES = {"upx": "8c3034da", "e4b": "887d4257", "vpq": "78c53e6b"}
MMSU = ["--benchmark", "mmsu", "--config", "examples/configs/qwen3_omni_mmmu_h100.yaml", "--model",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", "--passes", "c16"]
ROUNDS = ("e4b@0,vpq@1", "vpq@0,e4b@1", "e4b@0,vpq@1", "vpq@0,e4b@1", "upx@0,e4b@1", "e4b@0,upx@1")


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    for name, expected in EXPECTED_TREES.items():
        path = tree(name)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
        log(f"{name} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: the tree is not the Mac's")
            return
        else:
            pass
    for index, arms in enumerate(ROUNDS, start=1):
        out = OUT / f"round{index}"
        with open(OUT / f"round{index}.log", "w") as handle:
            rc = subprocess.run(
                [sys.executable, str(BOX / "videoamme_pair.py"), *MMSU, "--arms", arms, "--out", str(out)],
                cwd=BOX, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT,
            ).returncode
        speeds = {}
        for placement in arms.split(","):
            arm, card = placement.split("@")
            speed = json.loads((out / "c16" / arm / "mmsu.json").read_text())["speed"]
            speeds[arm] = speed["throughput_qps"]
            log(f"round {index} {arm} card {card} throughput {speed['throughput_qps']} latency "
                f"{speed['latency_mean_s']}")
        candidate, reference = ("vpq", "e4b") if "vpq" in speeds else ("e4b", "upx")
        log(f"round {index} rc {rc}: {candidate} against {reference} "
            f"{100 * (speeds[candidate] / speeds[reference] - 1):+.2f} %")
    log("queue done")


if __name__ == "__main__":
    main()
