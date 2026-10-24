"""run311: the audio encoder's attention through the packed attention with stacked q, k, v (64f8eeea9, vpq) against
main 6e75cba89 (upx).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/audio_attention_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
1. vpq's Qwen3-Omni unit suites (card 1) while audio_attention_probe.py runs on card 0: mmsu-ci-2000's clips through
   the processor, main's encoder in fp32 with math attention as the reference, then main's and vpq's served encoders
   (graph alone, eager alone, groups of 16 and 32), compared.
2. MMSU in CI stage 5's shape, upx against vpq at the same time, then swapped, c1 and c16: accuracy and speed.
Writes /data/pr2800/run311 and run311.log.
"""

from __future__ import annotations

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
OUT = DATA / "run311"
LOG = DATA / "run311.log"
EXPECTED_TREES = {"upx": "8c3034da", "vpq": "78c53e6b"}
SUITES = ["tests/unit_test/qwen3_omni"]
MMSU = ["--benchmark", "mmsu", "--config", "examples/configs/qwen3_omni_mmmu_h100.yaml", "--model",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", "--passes", "c1,c16"]


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], name: str) -> int:
    with open(OUT / f"{name}.log", "w") as handle:
        rc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    for line in (OUT / f"{name}.log").read_text().splitlines()[-12:]:
        log(f"  {line}")
    log(f"{name} rc {rc}")
    return rc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    # note (ratish): pair_check read the pins at import, before they were known; pinned() reads these.
    pair_check.CARD_CPUS = {int(card): card_cpus for card, card_cpus in (entry.split("=", 1) for entry in cpus.split(";"))}
    pair_check.NODE = node
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
    suites = subprocess.Popen(
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(paths["vpq"]), "--card", "1", "--out",
         str(OUT / "suites_vpq.log"), *SUITES],
        env=os.environ.copy(), stdout=open(OUT / "suites_vpq.out", "w"), stderr=subprocess.STDOUT,
    )
    probe = [sys.executable, str(BOX / "audio_attention_probe.py"), "--out", str(OUT / "probe")]
    for name, arm, step in (("prepare", "upx", ["--prepare"]), ("reference", "upx", ["--reference"]),
                            ("encode_main", "upx", ["--encode", "main"]),
                            ("encode_branch", "vpq", ["--encode", "branch"])):
        if run_logged(pinned(0, [*probe, *step]), cwd=paths[arm], env=card_env(0, paths[arm]), name=name):
            suites.wait()
            return
        else:
            pass
    run_logged([*probe, "--compare", "main", "branch"], cwd=paths["vpq"], env=card_env(0, paths["vpq"]),
               name="compare")
    suites.wait()
    for line in (OUT / "suites_vpq.out").read_text().splitlines()[-3:]:
        log(f"  {line}")
    log(f"suites rc {suites.returncode}")
    for round_name, arms in (("round1", "upx@0,vpq@1"), ("round2", "vpq@0,upx@1")):
        run_logged(
            [sys.executable, str(BOX / "videoamme_pair.py"), *MMSU, "--arms", arms, "--out",
             str(OUT / "mmsu" / round_name)],
            cwd=BOX, env=os.environ.copy(), name=f"mmsu_{round_name}",
        )
    log("queue done")


if __name__ == "__main__":
    main()
