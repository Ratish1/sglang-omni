"""run213: the conv-inputs branch with main 3d7d2c306 merged (0d422f290, cim) against that main (upm), on one card.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/merge_check_queue.py
Main moved 12 commits past the branch's base; #3029 gave channels_last_conv1d an explicit bias, which replaces the
branch's add_bias. Trees are pair_check.py's, fetched by full commit. Each step gates the next: a tree hash unlike the
Mac's, slot identity hashes that differ between main and the branch, or a failed suite stops the queue.
1. The slot identity probe on upm and on cim, logged against 0.5.21's and run205's hashes.
2. cim's suites: kernels, the channels-last conv and Qwen3-Omni.
Writes /data/pr2800/run213 and merge_check.log.
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
OUT = DATA / "run213"
LOG = DATA / "merge_check.log"
EXPECTED_TREES = {"upm": "f91be487", "cim": "3c0b68ee"}
HASHES_0521 = ("d614cbcf5f3bc882", "09cf5eb3de4e1045", "c034920400fe7268")
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
    versions = subprocess.run(
        ["python3", "-c", "import sglang, torch, transformers, triton, flashinfer; print(sglang.__version__, "
         "torch.__version__, transformers.__version__, triton.__version__, flashinfer.__version__, "
         "torch.backends.cudnn.version())"],
        capture_output=True, text=True).stdout.strip()
    log(f"stack sglang torch transformers triton flashinfer cudnn: {versions}")
    hashes = {}
    for name, path in paths.items():
        card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(path)}
        rc = subprocess.run(["python3", str(BOX / "omni_slot_identity_probe.py")], cwd=path, env=card0,
                            stdout=open(OUT / f"identity_{name}.log", "w"), stderr=subprocess.STDOUT).returncode
        hashes[name] = tuple(re.findall(r"sha256 ([0-9a-f]{16})", (OUT / f"identity_{name}.log").read_text()))
        log(f"identity {name} rc {rc} hashes {list(hashes[name])} equal to 0.5.21's {hashes[name] == HASHES_0521}")
    if len(hashes["upm"]) != 3 or hashes["upm"] != hashes["cim"]:
        log("stopped: identity differs between main and the branch")
        return
    else:
        pass
    rc = subprocess.run(
        ["python3", str(BOX / "suite_run.py"), "--tree", str(paths["cim"]), "--card", "0",
         "--out", str(OUT / "units_card0.log"), *SUITES],
        env=env, stdout=open(OUT / "units_card0.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"units card 0 rc {rc}")
    for line in (OUT / "units_card0.log").read_text().splitlines()[-3:]:
        log(f"  {line}")
    log("done")


if __name__ == "__main__":
    main()
