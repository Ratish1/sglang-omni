"""run322: the #3131 review fixes (rvf, d18506df4) against #3131's head (vpr, 3b8f90244) on the real model.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/review_fix_gates_queue.py
Card 1, vision (vision_boot_probe.py, MMMU-50's 51 images from run309): vpr and rvf served encoders, rvf again with
SGLANG_VIT_ENABLE_CUDA_GRAPH=1, rvf through its public constructor in fp32; compared vpr against rvf and rvf against
rvf with the flag, and every encode's distance to run309's fp32 encode.
Card 0: rvf's tests/unit_test/qwen3_omni (suite_run.py), then audio (audio_attention_probe.py, mmsu-ci-2000 from
run311): vpr and rvf served encoders (graph, eager, groups of 16 and 32), rvf again with the flag, compared; then
rvf's fp32 encoder with math attention against run311's fp32 reference built on main, row by row.
Writes /data/pr2800/run322 and run322.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import card_env, pinned, tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run322"
LOG = DATA / "run322.log"
EXPECTED_TREES = {"vpr": "6f56d9eb", "rvf": "60db7677"}
FLAG = {"SGLANG_VIT_ENABLE_CUDA_GRAPH": "1"}
LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with LOG_LOCK:
        print(line, flush=True)
        with open(LOG, "a") as handle:
            handle.write(line + "\n")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], name: str, tail: int) -> int:
    with open(OUT / f"{name}.log", "w") as handle:
        rc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    for line in (OUT / f"{name}.log").read_text().splitlines()[-tail:]:
        if "rope_parameters" not in line:
            log(f"  {line}")
        else:
            pass
    log(f"{name} rc {rc}")
    return rc


def link(directory: Path, source: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (directory / name).exists():
            (directory / name).symlink_to(source / name)
        else:
            pass


def vision_chain(paths: dict[str, Path]) -> None:
    probe = BOX / "vision_boot_probe.py"
    directory = OUT / "vision"
    link(directory, DATA / "run309" / "probe", ("inputs.pt", "native_fp32.pt"))
    for tag, arm, extra in (("pr_head", "vpr", {}), ("fix", "rvf", {}), ("fix_flag", "rvf", FLAG),
                            ("fix_fp32", "rvf", {"VISION_PROBE_DTYPE": "fp32"})):
        run_logged(pinned(1, [sys.executable, str(probe), "--encode", tag, "--out", str(directory)]),
                   cwd=paths[arm], env={**card_env(1, paths[arm]), **extra}, name=f"vision_{tag}", tail=2)
    for name, arguments in (("compare_pr_fix", ["--compare", "pr_head", "fix"]),
                            ("compare_fix_flag", ["--compare", "fix", "fix_flag"]),
                            ("distance", ["--distance", "native_fp32", "pr_head", "fix", "fix_flag", "fix_fp32"])):
        run_logged([sys.executable, str(probe), *arguments, "--out", str(directory)], cwd=paths["rvf"],
                   env=card_env(1, paths["rvf"]), name=f"vision_{name}", tail=5)


def audio_chain(paths: dict[str, Path]) -> None:
    probe = BOX / "audio_attention_probe.py"
    directory = OUT / "audio"
    link(directory, DATA / "run311" / "probe", ("inputs.pt", "reference.pt"))
    for tag, arm, extra in (("pr_head", "vpr", {}), ("fix", "rvf", {}), ("fix_flag", "rvf", FLAG)):
        run_logged(pinned(0, [sys.executable, str(probe), "--encode", tag, "--out", str(directory)]),
                   cwd=paths[arm], env={**card_env(0, paths[arm]), **extra}, name=f"audio_{tag}", tail=4)
    for name, tags in (("compare_pr_fix", ["pr_head", "fix"]), ("compare_fix_flag", ["fix", "fix_flag"])):
        run_logged([sys.executable, str(probe), "--compare", *tags, "--out", str(directory)], cwd=paths["rvf"],
                   env=card_env(0, paths["rvf"]), name=f"audio_{name}", tail=14)
    fp32_directory = OUT / "audio_fp32"
    link(fp32_directory, DATA / "run311" / "probe", ("inputs.pt",))
    run_logged(pinned(0, [sys.executable, str(probe), "--reference", "--out", str(fp32_directory)]),
               cwd=paths["rvf"], env=card_env(0, paths["rvf"]), name="audio_fp32_reference", tail=2)
    run_logged([sys.executable, str(probe), "--out", str(fp32_directory), "--reference-equal",
                str(DATA / "run311" / "probe")], cwd=paths["rvf"], env=card_env(0, paths["rvf"]),
               name="audio_fp32_equal", tail=2)


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
    for arm, expected in EXPECTED_TREES.items():
        paths[arm] = tree(arm)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=paths[arm], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sglang_omni", "tests"], cwd=paths[arm]).returncode
        log(f"{arm} sglang_omni {tree_hash} expected {expected} dirty {dirty}")
        if not tree_hash.startswith(expected) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    vision = threading.Thread(target=vision_chain, args=(paths,))
    vision.start()
    subprocess.run(
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(paths["rvf"]), "--card", "0", "--out",
         str(OUT / "units.log"), "tests/unit_test/qwen3_omni"],
        cwd=BOX, env=os.environ.copy(),
    )
    for line in (OUT / "units.log").read_text().splitlines():
        if line.startswith(("FAILED", "ERROR", "E   ", "____", "head ")) or " passed" in line:
            log(f"  units {line}")
        else:
            pass
    audio_chain(paths)
    vision.join()
    log("queue done")


if __name__ == "__main__":
    main()
