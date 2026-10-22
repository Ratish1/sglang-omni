"""run219: the talker's prompt rows rebuilt (#2861 on our branch, 5f33ec1c8, phs) against main (272845a9c, upk).
run218 passed phs's suites and recorded the rows the talker's prompt prefill receives on stage 6's first 40 MMSU
requests; its HF reference paired the server's startup warmup record with the first sample, so the reference runs
again here on run218's records.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_rows_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
1. prompt_rows_probe.py on run218's records: the rows against HF's thinker in fp32 and bf16 (card 1).
2. videoamme_talker_tp2.py: the CI's FP8 thinker TP2 Video-AMME talker, upk then phs.
3. mmsu_talker_pair.py: the full mmsu-ci-2000 with the talker, upk on card 0 and phs on card 1 at once.
4. pair_check.py, four rounds on both cards with the arms swapped: SeedTTS c16 and c64 unseeded, c16 and c64 seeded,
   seeded c1 on the full corpus every round; WER and SIM scored on the c16 and seeded c1 passes.
Writes /data/pr2800/run219 and talker_rows.log.
"""

from __future__ import annotations

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
OUT = DATA / "run219"
LOG = DATA / "talker_rows.log"
EXPECTED_TREES = {"upk": "1620e868", "phs": "d107853b"}
ROUNDS = "upk@0,phs@1;phs@0,upk@1;upk@0,phs@1;phs@0,upk@1"
PASSES = "c16,c64,c16_seeded,c64_seeded,c1_seeded_full"
SCORED = "c16,c1_seeded_full"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def step(
    name: str, command: list[str], env: dict[str, str], keep: tuple[str, ...]
) -> None:
    rc = subprocess.run(
        command,
        cwd=BOX,
        env=env,
        stdout=open(OUT / f"{name}.log", "w"),
        stderr=subprocess.STDOUT,
    ).returncode
    log(f"{name} rc {rc}")
    for line in (OUT / f"{name}.log").read_text().splitlines():
        if any(text in line for text in (*keep, "Traceback")):
            log(f"  {line[:300]}")
        else:
            pass


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True
    ).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    for name, expected in EXPECTED_TREES.items():
        path = tree(name)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path
        ).returncode
        log(f"{name} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    step(
        "probe",
        [
            "python3",
            str(BOX / "prompt_rows_probe.py"),
            "--arm",
            "phs",
            "--records",
            str(DATA / "run218" / "probe" / "dump"),
            "--out",
            str(OUT / "probe"),
        ],
        env,
        ("summary", "talker prompt prefills"),
    )
    step(
        "videoamme",
        [
            "python3",
            str(BOX / "videoamme_talker_tp2.py"),
            "--arms",
            "upk,phs",
            "--out",
            str(OUT / "videoamme"),
        ],
        env,
        ("client rc", "score rc", "accuracy"),
    )
    step(
        "mmsu",
        [
            "python3",
            str(BOX / "mmsu_talker_pair.py"),
            "--arms",
            "upk@0,phs@1",
            "--out",
            str(OUT / "mmsu"),
        ],
        env,
        ("client rc", "score rc", "accuracy"),
    )
    step(
        "seedtts",
        [
            "python3",
            str(BOX / "pair_check.py"),
            "--out",
            str(OUT / "seedtts"),
            "--rounds",
            ROUNDS,
            "--passes",
            PASSES,
            "--c1-every-round",
            "--score-passes",
            SCORED,
        ],
        env,
        ("failed",),
    )
    log("done")


if __name__ == "__main__":
    main()
