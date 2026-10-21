"""run205: the conv-inputs branch with main's SGLang 0.5.22 stack merged (b42d0b332, cit) against that main (upr).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/bump_queue.py
Trees are pair_check.py's, fetched by full commit. Each step gates the next: a tree hash unlike the Mac's, slot
identity hashes that differ between main and the branch, a kernel output unequal to eager or a failed suite stops
the queue. The stack moved (torch 2.14.1, transformers 5.19.0, SGLang 0.5.22, FlashInfer 0.7.1), so the identity
gate compares the two trees on this stack; whether 0.5.21's hashes still hold is logged, not gated.
1. The slot identity probe on upr and on cit (card 0).
2. snake_kernel_time.py on cit (card 0): the 29 decode launches at 1 and 8 rows against eager SnakeBeta.
3. cit's suites: kernels and the channels-last conv (card 0) beside Qwen3-Omni (card 1).
4. The kit on upr and cit (card 0): 1 and 8 rows, a stream's first window and a later one, 10 frames.
Writes /data/pr2800/run205 and bump.log.
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
OUT = DATA / "run205"
LOG = DATA / "bump.log"
EXPECTED_TREES = {"upr": "3a187e37", "cit": "41f7f2ab"}
HASHES_0521 = ("d614cbcf5f3bc882", "09cf5eb3de4e1045", "c034920400fe7268")
SUITES = {
    "0": ["tests/unit_test/kernels", "tests/unit_test/utils/test_channels_last_conv.py"],
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
    versions = subprocess.run(
        ["python3", "-c", "import sglang, torch, transformers, triton, flashinfer; print(sglang.__version__, "
         "torch.__version__, transformers.__version__, triton.__version__, flashinfer.__version__)"],
        capture_output=True, text=True).stdout.strip()
    log(f"stack sglang torch transformers triton flashinfer: {versions}")
    hashes = {}
    for name, path in paths.items():
        card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(path)}
        rc = subprocess.run(["python3", str(BOX / "omni_slot_identity_probe.py")], cwd=path, env=card0,
                            stdout=open(OUT / f"identity_{name}.log", "w"), stderr=subprocess.STDOUT).returncode
        hashes[name] = tuple(re.findall(r"sha256 ([0-9a-f]{16})", (OUT / f"identity_{name}.log").read_text()))
        log(f"identity {name} rc {rc} hashes {list(hashes[name])} equal to 0.5.21's {hashes[name] == HASHES_0521}")
    if len(hashes["upr"]) != 3 or hashes["upr"] != hashes["cit"]:
        log("stopped: identity differs between main and the branch")
        return
    else:
        pass
    branch = paths["cit"]
    card0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(branch)}
    rc = subprocess.run(["python3", str(BOX / "snake_kernel_time.py"), "--out", str(OUT)], cwd=branch, env=card0,
                        stdout=open(OUT / "kernel_time.out", "w"), stderr=subprocess.STDOUT).returncode
    summary = [line for line in (OUT / "kernel_time.out").read_text().splitlines() if "29 launches" in line]
    for line in summary:
        log(f"kernel {line}")
    if rc != 0 or len(summary) != 2 or not all(line.endswith("True") for line in summary):
        log(f"stopped: kernel time rc {rc}")
        return
    else:
        pass
    suites = {
        card: subprocess.Popen(
            ["python3", str(BOX / "suite_run.py"), "--tree", str(branch), "--card", card,
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
    kit = []
    for name, label in (("upr", "main"), ("cit", "branch")):
        for rows in (1, 8):
            for decode in ("first", "next"):
                kit += ["--kit", f"{label}_{rows}{decode},{paths[name].name},{rows},{decode},10"]
    rc = subprocess.run(["python3", str(BOX / "split_kit_queue.py"), "--out", str(OUT / "kit"), *kit],
                        env=env, stdout=open(OUT / "kit.out", "w"), stderr=subprocess.STDOUT).returncode
    log(f"kit rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
