"""run214: the final conv kernel branch (5a6430a8e, fcv) against the merged conv-inputs branch (0d422f290, cim) and
main (3d7d2c306, upm), on one card.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/final_conv_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's or a failed suite stops the queue.
1. final_conv_kernel_probe.py on fcv: the shipped kernel against its launch alternatives, the prototype and cuDNN.
2. fcv's suites: kernels, the channels-last conv and Qwen3-Omni.
3. The slot identity probe on fcv, logged beside run213's hashes (the decoder's last conv changes them).
4. The kit on upm, cim and fcv: 1 and 8 rows, a stream's first window and a later one, 10 frames.
Writes /data/pr2800/run214 and final_conv.log.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run214"
LOG = DATA / "final_conv.log"
EXPECTED_TREES = {"upm": "f91be487", "cim": "3c0b68ee", "fcv": "7b168ede"}
HASHES_RUN213 = ("d614cbcf5f3bc882", "09cf5eb3de4e1045", "c034920400fe7268")
SUITES = ["tests/unit_test/kernels", "tests/unit_test/utils/test_channels_last_conv.py", "tests/unit_test/qwen3_omni"]


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    paths = {name: tree(name) for name in EXPECTED_TREES}
    for name, path in paths.items():
        tree_hash = subprocess.run(["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True,
                                   text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
        log(f"{name} sglang_omni {tree_hash} expected {EXPECTED_TREES[name]} dirty {dirty}")
        if not tree_hash.startswith(EXPECTED_TREES[name]) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    kernel_tree = paths["fcv"]
    card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(kernel_tree)}
    rc = subprocess.run(
        ["numactl", f"--physcpubind={cpus.split(';')[0].split('=')[1]}", f"--membind={node}", "python3",
         str(BOX / "final_conv_kernel_probe.py"), "--out", str(OUT)],
        cwd=kernel_tree, env=card0, stdout=open(OUT / "probe.log", "w"), stderr=subprocess.STDOUT).returncode
    for line in (OUT / "probe.log").read_text().splitlines():
        if line.startswith(("next", "first", "registers")):
            log(f"probe {line}")
        else:
            pass
    if rc != 0:
        log(f"stopped: probe rc {rc}")
        return
    else:
        pass
    rc = subprocess.run(
        ["python3", str(BOX / "suite_run.py"), "--tree", str(kernel_tree), "--card", "0",
         "--out", str(OUT / "units_card0.log"), *SUITES],
        env=env, stdout=open(OUT / "units_card0.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"units card 0 rc {rc}")
    for line in (OUT / "units_card0.log").read_text().splitlines()[-3:]:
        log(f"  {line}")
    if rc != 0:
        log("stopped: a suite failed")
        return
    else:
        pass
    rc = subprocess.run(["python3", str(BOX / "omni_slot_identity_probe.py")], cwd=kernel_tree, env=card0,
                        stdout=open(OUT / "identity_fcv.log", "w"), stderr=subprocess.STDOUT).returncode
    hashes = tuple(re.findall(r"sha256 ([0-9a-f]{16})", (OUT / "identity_fcv.log").read_text()))
    log(f"identity fcv rc {rc} hashes {list(hashes)} equal to run213's {hashes == HASHES_RUN213}")
    kit = []
    for name, label in (("upm", "main"), ("cim", "branch"), ("fcv", "kernel")):
        for rows in (1, 8):
            for decode in ("first", "next"):
                kit += ["--kit", f"{label}_{rows}{decode},{paths[name].name},{rows},{decode},10"]
    rc = subprocess.run(["python3", str(BOX / "split_kit_queue.py"), "--out", str(OUT / "kit"), *kit],
                        env=env, stdout=open(OUT / "kit.out", "w"), stderr=subprocess.STDOUT).returncode
    log(f"kit rc {rc}")
    for line in (OUT / "kit.out").read_text().splitlines()[-14:]:
        log(f"  {line}")
    log("done")


if __name__ == "__main__":
    main()
