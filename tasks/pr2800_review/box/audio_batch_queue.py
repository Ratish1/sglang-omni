"""run308: where MMSU's c16 answers move between boots with the deterministic flag on, main 6e75cba89 (upx).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/audio_batch_queue.py
The tree is pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
1. audio_batch_probe.py on card 0, one process a variant: the layer graphs with batch-invariant mode off, turned on
   after the encoder is built (the served text pipeline's order), and turned on before it (twice, two processes);
   then eager with it on before, twice. Each variant compares every clip alone against the same clip batched with 15
   others; the repeated variants compare across processes.
2. MMSU in the CI's shape, c16, the thinker's deterministic flag on, two boots at once (upx card 0, upxb card 1) with
   layer dumps, then boot_identity_read.py on the two dumps.
Writes /data/pr2800/run308 and run308.log.
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
OUT = DATA / "run308"
LOG = DATA / "run308.log"
EXPECTED_TREE = "8c3034da"
VARIANTS = (
    ("graphs_off", {"AUDIO_PROBE_GRAPHS": "1", "AUDIO_PROBE_BATCH_INVARIANT": ""}),
    ("graphs_after", {"AUDIO_PROBE_GRAPHS": "1", "AUDIO_PROBE_BATCH_INVARIANT": "after"}),
    ("graphs_before", {"AUDIO_PROBE_GRAPHS": "1", "AUDIO_PROBE_BATCH_INVARIANT": "before"}),
    ("graphs_before_again", {"AUDIO_PROBE_GRAPHS": "1", "AUDIO_PROBE_BATCH_INVARIANT": "before"}),
    ("eager_before", {"AUDIO_PROBE_GRAPHS": "0", "AUDIO_PROBE_BATCH_INVARIANT": "before"}),
    ("eager_before_again", {"AUDIO_PROBE_GRAPHS": "0", "AUDIO_PROBE_BATCH_INVARIANT": "before"}),
)
COMPARISONS = (
    ["graphs_off"], ["graphs_after"], ["graphs_before"], ["eager_before"],
    ["graphs_before", "graphs_before_again"], ["eager_before", "eager_before_again"],
)
DETERMINISTIC = ["--serve-arg=--thinker.engine.enable_deterministic_inference", "--serve-arg=true"]


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    os.environ.update(PR2800_CARD_CPUS=cpus, PR2800_NODE=node)
    # note (ratish): pair_check read the pins at import, before they were known; pinned() reads these.
    pair_check.CARD_CPUS = {int(card): card_cpus for card, card_cpus in (entry.split("=", 1) for entry in cpus.split(";"))}
    pair_check.NODE = node
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
    probe = [sys.executable, str(BOX / "audio_batch_probe.py"), "--out", str(OUT / "probe")]
    env = card_env(0, path)
    if run_logged(pinned(0, [*probe, "--prepare"]), cwd=path, env=env, name="prepare"):
        return
    else:
        pass
    for name, variant_env in VARIANTS:
        run_logged(pinned(0, [*probe, "--encode", name]), cwd=path, env={**env, **variant_env}, name=name)
    for tags in COMPARISONS:
        run_logged([*probe, "--compare", *tags], cwd=path, env=env, name="compare_" + "_vs_".join(tags))
    served = OUT / "served"
    run_logged(
        [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", "mmsu", "--config",
         "examples/configs/qwen3_omni_mmmu_h100.yaml", "--model", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "--arms",
         "upx@0,upxb@1", "--passes", "c16", "--layer-dump", *DETERMINISTIC, "--out", str(served)],
        cwd=BOX, env=os.environ.copy(), name="served",
    )
    run_logged(
        [sys.executable, str(BOX / "boot_identity_read.py"), str(served / "c16/upx/layers"),
         str(served / "c16/upxb/layers")],
        cwd=path, env=env, name="boot_identity",
    )
    log("queue done")


if __name__ == "__main__":
    main()
