"""run232: #2861's head with band-limited resampling (ddde3a0aa, phx) against main ca9e8789f (ups), for its body.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/final_pair_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue. The model downloads
once before any server starts, then two pairs run at the same time:
1. SeedTTS on cards 0 and 1, ups@0,phx@1 then phx@0,ups@1: c16, c64 and the seeded full c1 every round, WER and SIM
   scored on c16 and the seeded full c1 (pair_check.py).
2. MMSU talker on cards 2 and 3, ups@2,phx@3, full mmsu-ci-2000, scored on card 2 (mmsu_talker_pair.py).
Writes /data/pr2800/run232 and final_pair.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import MODEL, tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run232"
LOG = DATA / "final_pair.log"
EXPECTED_TREES = {"ups": "4743b286", "phx": "6ffbdb09"}


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
    log(f"downloading {MODEL}")
    snapshot_download(MODEL)
    log("model ready")
    seedtts = subprocess.Popen(
        ["python3", str(BOX / "pair_check.py"), "--out", str(OUT / "seedtts"), "--rounds", "ups@0,phx@1;phx@0,ups@1",
         "--passes", "c16,c64,c1_seeded_full", "--c1-every-round", "--score-passes", "c16,c1_seeded_full"],
        env=env,
        stdout=open(OUT / "seedtts.out", "w"),
        stderr=subprocess.STDOUT,
    )
    mmsu = subprocess.Popen(
        ["python3", str(BOX / "mmsu_talker_pair.py"), "--arms", "ups@2,phx@3", "--out", str(OUT / "mmsu_talker"),
         "--score-card", "2"],
        env=env,
        stdout=open(OUT / "mmsu_talker.out", "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"mmsu talker rc {mmsu.wait()}")
    log(f"seedtts rc {seedtts.wait()}")
    log("done")


if __name__ == "__main__":
    main()
