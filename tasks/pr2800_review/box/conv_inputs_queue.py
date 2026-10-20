"""run190 to run192: the conv-inputs branch (1c and 1d as one, ca00439a1) against main (bcd0d8c68).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/conv_inputs_queue.py [--stop-before-pair | --pair-only]
Trees are pair_check.py's, fetched by full commit: main (upq, from upstream) and the branch (cip, from origin). Each
step gates the next: a tree hash unlike the Mac's, unequal identity hashes or a failed suite stops the queue.
1. run190: the slot identity probe on both trees (card 0); bf16 decodes keep 1a's three hashes (d614cbcf5f3bc882,
   09cf5eb3de4e1045, c034920400fe7268). Then the branch's suites: kernels and the channels-last conv (card 0) beside
   Qwen3-Omni (card 1), the bf16 and fp16 kernel and slot decode cases included.
2. run191: the kit on main and the branch (card 0): 1 and 8 rows, a stream's first window and a later one, 10 frames.
3. run192: main against the branch, four rounds swapped, c16 and c64 unseeded and the seeded c1 pass over the full
   corpus in each arm's first round; c16 and the seeded c1 pass scored.
--stop-before-pair runs steps 1 and 2, --pair-only step 3. Writes /data/pr2800/run190 to run192 and conv_inputs.log.
"""

from __future__ import annotations

import argparse
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
EXPECTED_TREES = {"upq": "ed42481d", "cip": "d938636c"}
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


def checks(paths: dict[str, Path], env: dict[str, str]) -> bool:
    out = DATA / "run190"
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
            return False
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
        return False
    else:
        pass
    kit = []
    for name, label in (("upq", "main"), ("cip", "branch")):
        for rows in (1, 8):
            for decode in ("first", "next"):
                kit += ["--kit", f"{label}_{rows}{decode},{paths[name].name},{rows},{decode},10"]
    rc = subprocess.run(
        ["python3", str(BOX / "split_kit_queue.py"), "--out", str(DATA / "run191"), *kit],
        env=env, stdout=open(DATA / "run191.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"kit rc {rc}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-before-pair", action="store_true", help="identity, suites and the kit only")
    parser.add_argument("--pair-only", action="store_true", help="the pair only")
    args = parser.parse_args()
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
    if args.pair_only:
        pass
    elif not checks(paths, env):
        return
    else:
        pass
    if args.stop_before_pair:
        log("done before the pair")
        return
    else:
        pass
    rc = subprocess.run(
        ["python3", str(BOX / "pair_check.py"), "--out", str(DATA / "run192"), "--rounds",
         "upq@0,cip@1;cip@0,upq@1;upq@0,cip@1;cip@0,upq@1", "--passes", "c16,c64,c1_seeded_full",
         "--score-passes", "c16,c1_seeded_full"],
        env=env, stdout=open(DATA / "run192.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"run192 rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
