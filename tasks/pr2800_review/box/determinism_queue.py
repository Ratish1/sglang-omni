"""run310: what moves MMSU's c16 answers between two boots of main 6e75cba89 (upx) with the deterministic flag on.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/determinism_queue.py (waits for run309's queue)
The tree is pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
MMSU in the CI's shape at c16, the thinker's deterministic flag on, two boots at once (upx card 0, upxb card 1) with
layer dumps of each request's prompt prefill: first as served (mixed chunk on), then with the thinker's mixed chunk off.
Each pair is read by boot_identity_read.py (the prompt chunk's input embeddings, audio rows and first-token logits)
and by its answers, request by request.
Writes /data/pr2800/run310 and run310.log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import card_env, tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run310"
LOG = DATA / "run310.log"
WAIT_FOR = DATA / "run309.log"
EXPECTED_TREE = "8c3034da"
DETERMINISTIC = ["--serve-arg=--thinker.engine.enable_deterministic_inference", "--serve-arg=true"]
VARIANTS = (
    ("mixed_on", []),
    ("mixed_off", ["--serve-arg=--thinker.engine.enable_mixed_chunk", "--serve-arg=false"]),
)


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], name: str) -> int:
    with open(OUT / f"{name}.log", "w") as handle:
        rc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    for line in (OUT / f"{name}.log").read_text().splitlines()[-4:]:
        log(f"  {line}")
    log(f"{name} rc {rc}")
    return rc


def answers(pass_dir: Path) -> None:
    first, second = (
        {row["sample_id"]: row for row in json.loads((pass_dir / arm / "mmsu.json").read_text())["per_sample"]}
        for arm in ("upx", "upxb")
    )
    same_response = sum(first[sample]["raw_response"] == second[sample]["raw_response"] for sample in first)
    same_choice = sum(first[sample]["predicted_choice"] == second[sample]["predicted_choice"] for sample in first)
    log(f"  {pass_dir.parent.name}: {len(first)} samples, responses identical {same_response}, choices identical "
        f"{same_choice}, correct {sum(row['is_correct'] for row in first.values())} and "
        f"{sum(row['is_correct'] for row in second.values())}")


def main() -> None:
    while "queue done" not in (WAIT_FOR.read_text() if WAIT_FOR.exists() else ""):
        time.sleep(30)
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    box_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    path = tree("upx")
    tree_hash = subprocess.run(
        ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path).returncode
    log(f"upx sglang_omni {tree_hash} expected {EXPECTED_TREE} dirty {dirty}")
    if not tree_hash.startswith(EXPECTED_TREE) or dirty:
        log("stopped: the tree is not the Mac's")
        return
    else:
        pass
    for name, extra in VARIANTS:
        out = OUT / name
        run_logged(
            [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", "mmsu", "--config",
             "examples/configs/qwen3_omni_mmmu_h100.yaml", "--model", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "--arms",
             "upx@0,upxb@1", "--passes", "c16", "--layer-dump", *DETERMINISTIC, *extra, "--out", str(out)],
            cwd=BOX, env=os.environ.copy(), name=f"served_{name}",
        )
        run_logged(
            [sys.executable, str(BOX / "boot_identity_read.py"), str(out / "c16/upx/layers"),
             str(out / "c16/upxb/layers")],
            cwd=path, env=card_env(0, path), name=f"boot_identity_{name}",
        )
        answers(out / "c16")
    log("queue done")


if __name__ == "__main__":
    main()
