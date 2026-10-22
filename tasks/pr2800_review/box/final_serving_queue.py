"""run215: the final conv kernel branch's head (d24f347a7, fcw) against main (3d7d2c306, upm), served on one card.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/final_serving_queue.py
Trees are pair_check.py's, fetched by full commit. A tree hash unlike the Mac's, a failed suite or slot identity
unlike run214's fcv stops the queue (d24f347a7 adds only an assert to 5a6430a8e).
1. fcw's suites: kernels, the channels-last conv and Qwen3-Omni.
2. The slot identity probe on fcw, gated on run214's fcv hashes.
3. pair_check.py, four rounds on card 0 with the arms in turn (upm>fcw, fcw>upm, upm>fcw, fcw>upm): c16 and c64
   unseeded, c16 and c64 seeded (identical requests), seeded c1 on the full corpus every round; WER and SIM scored on
   the c16 and seeded c1 passes.
Writes /data/pr2800/run215 and final_serving.log.
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
OUT = DATA / "run215"
LOG = DATA / "final_serving.log"
EXPECTED_TREES = {"upm": "f91be487", "fcw": "ada34dac"}
HASHES_RUN214 = ("92ae661558cb956a", "839020dd9ab51548", "4a79694f36bdc482")
SUITES = ["tests/unit_test/kernels", "tests/unit_test/utils/test_channels_last_conv.py", "tests/unit_test/qwen3_omni"]
ROUNDS = "upm@0>fcw@0;fcw@0>upm@0;upm@0>fcw@0;fcw@0>upm@0"
PASSES = "c16,c64,c16_seeded,c64_seeded,c1_seeded_full"
SCORED = "c16,c1_seeded_full"


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
    head_tree = paths["fcw"]
    rc = subprocess.run(
        ["python3", str(BOX / "suite_run.py"), "--tree", str(head_tree), "--card", "0",
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
    card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(head_tree)}
    rc = subprocess.run(["python3", str(BOX / "omni_slot_identity_probe.py")], cwd=head_tree, env=card0,
                        stdout=open(OUT / "identity_fcw.log", "w"), stderr=subprocess.STDOUT).returncode
    hashes = tuple(re.findall(r"sha256 ([0-9a-f]{16})", (OUT / "identity_fcw.log").read_text()))
    log(f"identity fcw rc {rc} hashes {list(hashes)} equal to run214's fcv {hashes == HASHES_RUN214}")
    if hashes != HASHES_RUN214:
        log("stopped: identity differs from run214's fcv")
        return
    else:
        pass
    rc = subprocess.run(
        ["python3", str(BOX / "pair_check.py"), "--out", str(OUT), "--rounds", ROUNDS, "--passes", PASSES,
         "--c1-every-round", "--score-passes", SCORED],
        cwd=BOX, env=env, stdout=open(OUT / "pairs.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"pair_check rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
