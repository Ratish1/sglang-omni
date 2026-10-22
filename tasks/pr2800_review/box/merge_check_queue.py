"""run222: #2999 with main 1b726ab64 merged (fc7f8e03f, cmg) against its measured head d24f347a7 (fcw).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/merge_check_queue.py
Main moved two commits past the branch's last merge: SGLang draft workers on AR stages (#2937) and the Qwen3-TTS
vocoder's fused transformer (#3051), which moves code2wav's q/k/v and gate/up weight views into
merge_projection_weights. Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the
queue.
1. The slot identity probe on cmg (card 0), against run214's hashes, which fcw repeated in run215.
2. cmg's suites at the same time (card 1): kernels, the channels-last conv and Qwen3-Omni.
Writes /data/pr2800/run222 and merge_check.log.
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
OUT = DATA / "run222"
LOG = DATA / "merge_check.log"
EXPECTED_TREE = "dc2a4275"
HASHES_RUN214 = ("92ae661558cb956a", "839020dd9ab51548", "4a79694f36bdc482")
SUITES = [
    "tests/unit_test/kernels",
    "tests/unit_test/utils/test_channels_last_conv.py",
    "tests/unit_test/qwen3_omni",
]


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True
    ).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    path = tree("cmg")
    tree_hash = subprocess.run(
        ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path
    ).returncode
    log(f"cmg sglang_omni {tree_hash} expected {EXPECTED_TREE} dirty {dirty}")
    if not tree_hash.startswith(EXPECTED_TREE) or dirty:
        log("stopped: the tree is not the Mac's")
        return
    else:
        pass
    suites = subprocess.Popen(
        [
            "python3",
            str(BOX / "suite_run.py"),
            "--tree",
            str(path),
            "--card",
            "1",
            "--out",
            str(OUT / "units_card1.log"),
            *SUITES,
        ],
        env=env,
        stdout=open(OUT / "units_card1.out", "w"),
        stderr=subprocess.STDOUT,
    )
    card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(path)}
    rc = subprocess.run(
        ["python3", str(BOX / "omni_slot_identity_probe.py")],
        cwd=path,
        env=card0,
        stdout=open(OUT / "identity_cmg.log", "w"),
        stderr=subprocess.STDOUT,
    ).returncode
    hashes = tuple(
        re.findall(r"sha256 ([0-9a-f]{16})", (OUT / "identity_cmg.log").read_text())
    )
    log(
        f"identity cmg rc {rc} hashes {list(hashes)} equal to run214's "
        f"{hashes == HASHES_RUN214}"
    )
    rc = suites.wait()
    log(f"units card 1 rc {rc}")
    for line in (OUT / "units_card1.log").read_text().splitlines()[-3:]:
        log(f"  {line}")
    log("done")


if __name__ == "__main__":
    main()
