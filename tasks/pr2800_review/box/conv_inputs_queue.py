"""run187 to run189: the conv-inputs branch (1c and 1d as one, 333d3cda8) against main (bcd0d8c68).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/conv_inputs_queue.py
Trees are pair_check.py's, fetched by full commit: main (upq, from upstream) and the branch (cip, from origin). Each
step gates the next: a tree hash unlike the Mac's, unequal identity hashes or a failed suite stops the queue.
1. run187: the slot identity probe on both trees (card 0); bf16 decodes keep 1a's three hashes (d614cbcf5f3bc882,
   09cf5eb3de4e1045, c034920400fe7268). Then the branch's suites: kernels and the channels-last conv (card 0) beside
   Qwen3-Omni (card 1), the bf16 and fp16 kernel and slot decode cases included.
2. run188: the kit on main and the branch (card 0): 1 and 8 rows, a stream's first window and a later one, 10 frames.
3. run189: main against the branch, four rounds swapped, c16 and c64 unseeded and the seeded c1 pass over the full
   corpus in each arm's first round; c16 and the seeded c1 pass scored.
Writes /data/pr2800/run187 to run189 and /data/pr2800/conv_inputs.log.
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
LOG = DATA / "conv_inputs.log"
EXPECTED_TREES = {"upq": "ed42481d", "cip": "7d7f27c6"}
EXPECTED_IDENTITY = ("d614cbcf5f3bc882", "09cf5eb3de4e1045", "c034920400fe7268")
SUITES = {
    "0": ["tests/unit_test/kernels", "tests/unit_test/utils/test_channels_last_conv.py"],
    "1": ["tests/unit_test/qwen3_omni"],
}


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
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
    out = DATA / "run187"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        rc = subprocess.run(
            ["python3", str(BOX / "omni_slot_identity_probe.py")],
            cwd=path,
            env={**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(path)},
            stdout=open(out / f"identity_{name}.log", "w"),
            stderr=subprocess.STDOUT,
        ).returncode
        hashes = re.findall(r"sha256 ([0-9a-f]{16})", (out / f"identity_{name}.log").read_text())
        log(f"identity {name} rc {rc} hashes {hashes} equal {tuple(hashes) == EXPECTED_IDENTITY}")
        if tuple(hashes) != EXPECTED_IDENTITY:
            log("stopped: identity differs")
            return
        else:
            pass
    suites = {
        card: subprocess.Popen(
            ["python3", str(BOX / "suite_run.py"), "--tree", str(paths["cip"]), "--card", card,
             "--out", str(out / f"units_card{card}.log"), *tests],
            env=env, stdout=open(out / f"units_card{card}.out", "w"), stderr=subprocess.STDOUT,
        )
        for card, tests in SUITES.items()
    }
    failed = False
    for card, process in suites.items():
        rc = process.wait()
        log(f"units card {card} rc {rc}")
        for line in (out / f"units_card{card}.log").read_text().splitlines()[-3:]:
            log(f"  {line}")
        failed = failed or rc != 0
    if failed:
        log("stopped: a suite failed")
        return
    else:
        pass
    kit = []
    for name, label in (("upq", "main"), ("cip", "branch")):
        for rows in (1, 8):
            for decode in ("first", "next"):
                kit += ["--kit", f"{label}_{rows}{decode},{paths[name].name},{rows},{decode},10"]
    rc = subprocess.run(
        ["python3", str(BOX / "split_kit_queue.py"), "--out", str(DATA / "run188"), *kit],
        env=env, stdout=open(DATA / "run188.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"kit rc {rc}")
    rc = subprocess.run(
        ["python3", str(BOX / "pair_check.py"), "--out", str(DATA / "run189"), "--rounds",
         "upq@0,cip@1;cip@0,upq@1;upq@0,cip@1;cip@0,upq@1", "--passes", "c16,c64,c1_seeded_full",
         "--score-passes", "c16,c1_seeded_full"],
        env=env, stdout=open(DATA / "run189.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"run189 rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
