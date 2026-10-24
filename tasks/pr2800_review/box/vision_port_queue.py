"""The device position interpolation branch against main 6e75cba89 (upx), unit and component gates.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/vision_port_queue.py --run run317 --arm vdi
1. upx's and the arm's sglang_omni tree hashes against the Mac's.
2. The arm's tests/unit_test/qwen3_omni on card 1 (suite_run.py) while vision_port_probe.py runs from the arm on card 0
   over run316's four recorded sets.
3. vision_boot_probe.py --encode over MMMU-50's 51 images (run309's inputs): upx on card 0 and the arm on card 1 at the
   same time, then swapped; each against run309's fp32 encode (--distance) and the arms against each other
   (--compare).
Writes /data/pr2800/<run> and <run>.log.
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
from pair_check import card_env, pinned, tree  # noqa: E402

DATA = Path("/data/pr2800")
RUN316 = DATA / "run316"
RUN309_PROBE = DATA / "run309" / "probe"
EXPECTED_TREES = {"upx": "8c3034da", "vdi": "d4c87537", "vdk": "bf85c3b1"}
OUT = DATA
LOG = DATA / "unset.log"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def encode(arm: str, card: int, tag: str, encode_dir: Path) -> subprocess.Popen:
    path = tree(arm)
    return subprocess.Popen(
        pinned(card, [sys.executable, str(BOX / "vision_boot_probe.py"), "--encode", tag, "--out", str(encode_dir)]),
        cwd=path, env=card_env(card, path), stdout=open(encode_dir / f"{tag}.log", "w"), stderr=subprocess.STDOUT,
    )


def main() -> None:
    global OUT, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(set(EXPECTED_TREES) - {"upx"}))
    args = parser.parse_args()
    OUT, LOG = DATA / args.run, DATA / f"{args.run}.log"
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
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni", "tests"], cwd=path).returncode
        log(f"{arm} sglang_omni {tree_hash} expected {EXPECTED_TREES[arm]} dirty {dirty}")
        if not tree_hash.startswith(EXPECTED_TREES[arm]) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    branch = tree(args.arm)
    units = subprocess.Popen(
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(branch), "--card", "1", "--out",
         str(OUT / "units.log"), "tests/unit_test/qwen3_omni"],
        cwd=BOX, env=os.environ.copy(),
    )
    calls = [
        f"{benchmark}_{pass_name}={RUN316 / benchmark / pass_name / arm / 'grids'}"
        for benchmark, arm in (("mmmu", "upx"), ("videoamme", "upxb"))
        for pass_name in ("c1", "c16")
    ]
    with open(OUT / "port_probe.log", "w") as handle:
        rc = subprocess.run(
            pinned(0, [sys.executable, str(BOX / "vision_port_probe.py"), "--calls", *calls]),
            cwd=branch, env=card_env(0, branch), stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    for line in (OUT / "port_probe.log").read_text().splitlines():
        if line.startswith(("unit", "served", "==", "sglang", "loop", "vectorized", "dispatch", "Traceback")):
            log(f"  {line}")
        else:
            pass
    log(f"port probe rc {rc}")
    units.wait()
    for line in (OUT / "units.log").read_text().splitlines()[-8:]:
        log(f"  units {line}")
    encode_dir = OUT / "encode"
    encode_dir.mkdir(exist_ok=True)
    for name in ("inputs.pt", "native_fp32.pt"):
        if not (encode_dir / name).exists():
            (encode_dir / name).symlink_to(RUN309_PROBE / name)
        else:
            pass
    for round_arms in (
        (("upx", 0, "main_r1"), (args.arm, 1, "branch_r1")), ((args.arm, 0, "branch_r2"), ("upx", 1, "main_r2")),
    ):
        processes = [(tag, encode(arm, card, tag, encode_dir)) for arm, card, tag in round_arms]
        for tag, process in processes:
            process.wait()
            for line in (encode_dir / f"{tag}.log").read_text().splitlines()[-3:]:
                log(f"  {line}")
            log(f"encode {tag} rc {process.returncode}")
    upx = tree("upx")
    with open(encode_dir / "read.log", "w") as handle:
        for arguments in (
            ["--compare", "main_r1", "main_r2"], ["--compare", "branch_r1", "branch_r2"],
            ["--compare", "main_r1", "branch_r1"],
            ["--distance", "native_fp32", "main_r1", "main_r2", "branch_r1", "branch_r2"],
        ):
            subprocess.run(
                [sys.executable, str(BOX / "vision_boot_probe.py"), *arguments, "--out", str(encode_dir)],
                cwd=upx, env=card_env(0, upx), stdout=handle, stderr=subprocess.STDOUT,
            )
    for line in (encode_dir / "read.log").read_text().splitlines():
        log(f"  {line}")
    log("queue done")


if __name__ == "__main__":
    main()
