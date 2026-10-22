"""run218: the talker's prompt rows rebuilt (#2861 on our branch, 5f33ec1c8, phs), its suites and its rows against HF.
run217 ran 51cfa3f6c (phr): three test setups failed (a list radix key against the request's array, and the thinker
test's text model patched under the base class name), fixed in 5f33ec1c8 with the same sglang_omni tree.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_rows_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's or a failed suite stops the queue.
1. phs's suites on both cards at once: the prompt hidden request, the shared thinker runner and its MiniCPM-o
   subclass, the engine builder (card 0); Qwen3-Omni (card 1).
2. prompt_rows_probe.py on phs: the rows the talker's prompt prefill receives on stage 6's first 40 MMSU requests,
   against HF's thinker in fp32 and bf16.
Writes /data/pr2800/run217 and talker_rows.log.
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
OUT = DATA / "run218"
LOG = DATA / "talker_rows.log"
EXPECTED_TREES = {"phs": "d107853b"}
SUITES = {
    "0": ["tests/unit_test/scheduling/test_prompt_hidden.py", "tests/unit_test/model_runner",
          "tests/unit_test/minicpm_o/test_thinker_model_runner.py", "tests/unit_test/cpu/test_engine_builder.py"],
    "1": ["tests/unit_test/qwen3_omni"],
}


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
    suites = {
        card: subprocess.Popen(
            ["python3", str(BOX / "suite_run.py"), "--tree", str(paths["phs"]), "--card", card,
             "--out", str(OUT / f"units_card{card}.log"), *tests],
            env=env, stdout=open(OUT / f"units_card{card}.out", "w"), stderr=subprocess.STDOUT,
        )
        for card, tests in SUITES.items()
    }
    failed = False
    for card, process in suites.items():
        rc = process.wait()
        log(f"units card {card} rc {rc}")
        for line in (OUT / f"units_card{card}.log").read_text().splitlines()[-3:]:
            log(f"  {line}")
        failed = failed or rc != 0
    if failed:
        log("stopped: a suite failed")
        return
    else:
        pass
    rc = subprocess.run(["python3", str(BOX / "prompt_rows_probe.py"), "--arm", "phs", "--out", str(OUT / "probe")],
                        cwd=BOX, env=env, stdout=open(OUT / "probe.log", "w"), stderr=subprocess.STDOUT).returncode
    log(f"probe rc {rc}")
    for line in (OUT / "probe.log").read_text().splitlines():
        if line.startswith(("summary", "client rc", "Traceback")) or "talker prompt prefills" in line:
            log(f"  {line}")
        else:
            pass
    log("done")


if __name__ == "__main__":
    main()
