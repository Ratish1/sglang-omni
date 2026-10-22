"""run228: band-limited audio resampling (86f163ed1, abr) against its base, main 1b726ab64 (upl).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/resample_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
1. abr's preprocessing and speech voice suites (card 1), and abr's new resampling test against upl's code (card 0),
   which must fail there. The test file runs from a copy outside the trees, so the tree's own package is the one
   imported.
2. MMSU understanding in the CI's request shape on the full mmsu-ci-2000, upl on card 0 and abr on card 1.
3. SeedTTS, upl@0,abr@1 then abr@0,upl@1: c16 in both rounds, the seeded full c1 once per arm, WER and SIM scored on
   the seeded full c1.
Writes /data/pr2800/run228 and resample.log.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run228"
LOG = DATA / "resample.log"
EXPECTED_TREES = {"upl": "6ffdfe50", "abr": "b4d1aee0"}
SUITES = ["tests/unit_test/preprocessing", "tests/unit_test/serve/test_speech_voices.py"]
NEW_TEST = "test_audio_at_another_rate_loads_as_the_reference_loader"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def log_tail(path: Path, lines: int) -> None:
    for line in path.read_text().splitlines()[-lines:]:
        log(f"  {line}")


def suite(env: dict[str, str], path: Path, card: int, name: str, tests: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        ["python3", str(BOX / "suite_run.py"), "--tree", str(path), "--card", str(card), "--out",
         str(OUT / f"{name}.log"), *tests],
        env=env,
        stdout=open(OUT / f"{name}.out", "w"),
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    paths = {}
    for name, expected in EXPECTED_TREES.items():
        paths[name] = tree(name)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=paths[name], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=paths[name]).returncode
        log(f"{name} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: the tree is not the Mac's")
            return
        else:
            pass
    prefix = OUT / "prefix"
    prefix.mkdir(exist_ok=True)
    shutil.copy(paths["abr"] / "tests/unit_test/preprocessing/test_audio.py", prefix / "test_audio_on_main.py")
    units = suite(env, paths["abr"], 1, "units_abr", SUITES)
    before = suite(env, paths["upl"], 0, "new_test_on_upl", [f"{prefix / 'test_audio_on_main.py'}::{NEW_TEST}"])
    log(f"new test on upl rc {before.wait()}")
    log_tail(OUT / "new_test_on_upl.log", 4)
    log(f"units abr rc {units.wait()}")
    log_tail(OUT / "units_abr.log", 3)
    rc = subprocess.run(
        ["python3", str(BOX / "mmsu_understanding_pair.py"), "--arms", "upl@0,abr@1", "--out", str(OUT / "mmsu")],
        env=env,
        stdout=open(OUT / "mmsu.out", "w"),
        stderr=subprocess.STDOUT,
    ).returncode
    log(f"mmsu rc {rc}")
    log_tail(OUT / "mmsu" / "driver.log", 4)
    rc = subprocess.run(
        ["python3", str(BOX / "pair_check.py"), "--out", str(OUT / "seedtts"), "--rounds", "upl@0,abr@1;abr@0,upl@1",
         "--passes", "c16,c1_seeded_full", "--score-passes", "c1_seeded_full"],
        env=env,
        stdout=open(OUT / "seedtts.out", "w"),
        stderr=subprocess.STDOUT,
    ).returncode
    log(f"seedtts rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
