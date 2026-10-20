"""run180 to run183: 1c and 1d with upstream main merged (#2949 in main): identity, suites, kit, then both pairs.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/followups_merge_queue.py
Trees are pair_check.py's, fetched by full commit: main e7339958b (upn, from upstream), 1c d0da0626f (s3z) and 1d
e3ab1c533 (s4z, both from origin). Each step gates the next: a tree hash unlike the Mac's, unequal identity hashes or a
failed suite stops the queue before the pairs.
1. run180: the slot identity probe on each tree (card 0); #2949, 1c and 1d each decode bit identical to 1a, so all
   three should give d614cbcf5f3bc882, 09cf5eb3de4e1045 and c034920400fe7268. Then the suites: 1c's kernels and
   Qwen3-Omni (card 0) beside 1d's kernels, channels-last conv and Qwen3-Omni (card 1).
2. run181: the kit on main, 1c and 1d (card 0): 1 and 8 rows, a stream's first window and a later one, 10 new frames.
3. run182: main against 1c, four rounds swapped, c16 and c64 unseeded and the seeded c1 pass over the full corpus in
   each arm's first round; c16 and the seeded c1 pass scored.
4. run183: 1c against 1d, the same.
Writes /data/pr2800/run180 to run183 and /data/pr2800/followups_merge.log.
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
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
LOG = DATA / "followups_merge.log"
EXPECTED_TREES = {"upn": "9aef3160", "s3z": "55b2543f", "s4z": "994ee2ba"}
EXPECTED_IDENTITY = ("d614cbcf5f3bc882", "09cf5eb3de4e1045", "c034920400fe7268")
SUITES = {
    "s3z": ("0", ["tests/unit_test/kernels", "tests/unit_test/qwen3_omni"]),
    "s4z": ("1", ["tests/unit_test/kernels", "tests/unit_test/utils/test_channels_last_conv.py",
                  "tests/unit_test/qwen3_omni"]),
}
PAIRS = {"run182": ("upn", "s3z"), "run183": ("s3z", "s4z")}


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def card_pins() -> tuple[str, str]:
    """PR2800_CARD_CPUS and PR2800_NODE for cards 0 and 1: eight cores of each card's affinity and their siblings."""
    lines = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout.splitlines()
    header = lines[0].split("\t")
    cpu_column, node_column = header.index("CPU Affinity"), header.index("NUMA Affinity")
    taken: dict[str, int] = {}
    pins, nodes = [], []
    for card in (0, 1):
        fields = next(line for line in lines if line.startswith(f"GPU{card}\t")).split("\t")
        affinity, node = fields[cpu_column].strip(), fields[node_column].strip()
        offset = taken.get(node, 0)
        taken[node] = offset + 8
        ranges = [int(span.split("-")[0]) for span in affinity.split(",")]
        pins.append(f"{card}=" + ",".join(f"{start + offset}-{start + offset + 7}" for start in ranges))
        nodes.append(node)
    return ";".join(pins), ",".join(sorted(set(nodes)))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    versions = subprocess.run(
        ["python3", "-c", "import torch, sglang, flashinfer, transformers; print(torch.__version__, sglang.__version__, "
         "flashinfer.__version__, transformers.__version__)"],
        capture_output=True, text=True,
    )
    log(f"torch sglang flashinfer transformers: {versions.stdout.strip()}")
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
    out = DATA / "run180"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        rc = subprocess.run(
            ["python3", str(BOX / "omni_slot_identity_probe.py")],
            cwd=path,
            env={**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(path)},
            stdout=open(out / f"identity_{name}.log", "w"),
            stderr=subprocess.STDOUT,
        ).returncode
        hashes = re.findall(r"sha256 ([0-9a-f]{16})", (out / f"identity_{name}.log").read_text())
        log(f"identity {name} rc {rc} hashes {hashes} equal {tuple(hashes) == EXPECTED_IDENTITY}")
        if tuple(hashes) != EXPECTED_IDENTITY:
            log("stopped: identity differs")
            return
        else:
            pass
    suites = {
        name: subprocess.Popen(
            ["python3", str(BOX / "suite_run.py"), "--tree", str(paths[name]), "--card", card,
             "--out", str(out / f"units_{name}.log"), *tests],
            env=env, stdout=open(out / f"units_{name}.out", "w"), stderr=subprocess.STDOUT,
        )
        for name, (card, tests) in SUITES.items()
    }
    failed = False
    for name, process in suites.items():
        rc = process.wait()
        log(f"units {name} rc {rc}")
        for line in (out / f"units_{name}.log").read_text().splitlines()[-3:]:
            log(f"  {line}")
        failed = failed or rc != 0
    if failed:
        log("stopped: a suite failed")
        return
    else:
        pass
    kit = []
    for name, label in (("upn", "main"), ("s3z", "1c"), ("s4z", "1d")):
        for rows in (1, 8):
            for decode in ("first", "next"):
                kit += ["--kit", f"{label}_{rows}{decode},{paths[name].name},{rows},{decode},10"]
    rc = subprocess.run(
        ["python3", str(BOX / "split_kit_queue.py"), "--out", str(DATA / "run181"), *kit],
        env=env, stdout=open(DATA / "run181.out", "w"), stderr=subprocess.STDOUT,
    ).returncode
    log(f"kit rc {rc}")
    for run_name, (a, b) in PAIRS.items():
        rounds = f"{a}@0,{b}@1;{b}@0,{a}@1;{a}@0,{b}@1;{b}@0,{a}@1"
        rc = subprocess.run(
            ["python3", str(BOX / "pair_check.py"), "--out", str(DATA / run_name), "--rounds", rounds,
             "--passes", "c16,c64,c1_seeded_full", "--score-passes", "c16,c1_seeded_full"],
            env=env, stdout=open(DATA / f"{run_name}.out", "w"), stderr=subprocess.STDOUT,
        ).returncode
        log(f"{run_name} rc {rc}")
    log("done")


if __name__ == "__main__":
    main()
