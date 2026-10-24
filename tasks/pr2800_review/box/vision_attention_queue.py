"""run309: the vision encoder's attention through the packed attention (4dbb97218, vpa) against main 6e75cba89 (upx).

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/vision_attention_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's stops the queue.
1. vpa's Qwen3-Omni unit suites (card 1) while vision_boot_probe.py runs on card 0: MMMU-50's 51 images through
   main's encoder once, vpa's encoder in two processes, the tree's Transformers-attention encoder in bf16 and in fp32
   with math attention; then vpa's two processes compared, and every encoder's distance to fp32. Each encode also times
   three passes over the 51 images.
2. MMMU in CI stage 3's shape with the thinker's deterministic flag: two vpa boots at once (vpa card 0, vpab card 1)
   with layer dumps, c1 then c16, each pass read by boot_identity_read.py.
3. MMMU and Video-AMME in their CI shapes (c16), upx against vpa at the same time, then swapped: accuracy and speed.
Writes /data/pr2800/run309 and run309.log.
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
OUT = DATA / "run309"
LOG = DATA / "run309.log"
EXPECTED_TREES = {"upx": "8c3034da", "vpa": "315b8056"}
SUITES = ["tests/unit_test/qwen3_omni"]
ENCODES = (
    ("main", "upx", {}),
    ("packed", "vpa", {}),
    ("packed_again", "vpa", {}),
    ("native_bf16", "vpa", {"VISION_PROBE_NATIVE": "bf16"}),
    ("native_fp32", "vpa", {"VISION_PROBE_NATIVE": "fp32", "VISION_PROBE_SDPA": "math"}),
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


def pair(benchmark: str, arms: str, out: Path, extra: list[str]) -> None:
    run_logged(
        [sys.executable, str(BOX / "videoamme_pair.py"), "--benchmark", benchmark, "--arms", arms, *extra,
         "--out", str(out)],
        cwd=BOX, env=os.environ.copy(), name=out.relative_to(OUT).as_posix().replace("/", "_"),
    )


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
        [sys.executable, str(BOX / "suite_run.py"), "--tree", str(paths["vpa"]), "--card", "1", "--out",
         str(OUT / "suites_vpa.log"), *SUITES],
        env=os.environ.copy(), stdout=open(OUT / "suites_vpa.out", "w"), stderr=subprocess.STDOUT,
    )
    probe = [sys.executable, str(BOX / "vision_boot_probe.py"), "--out", str(OUT / "probe")]
    if run_logged(pinned(0, [*probe, "--prepare"]), cwd=paths["vpa"], env=card_env(0, paths["vpa"]), name="prepare"):
        return
    else:
        pass
    for tag, arm, variant_env in ENCODES:
        run_logged(
            pinned(0, [*probe, "--encode", tag]), cwd=paths[arm], env={**card_env(0, paths[arm]), **variant_env},
            name=f"encode_{tag}",
        )
    run_logged([*probe, "--compare", "packed", "packed_again"], cwd=paths["vpa"], env=card_env(0, paths["vpa"]),
               name="compare_packed")
    run_logged([*probe, "--distance", "native_fp32", "main", "packed", "native_bf16"], cwd=paths["vpa"],
               env=card_env(0, paths["vpa"]), name="distance")
    suites.wait()
    for line in (OUT / "suites_vpa.out").read_text().splitlines()[-3:]:
        log(f"  {line}")
    log(f"suites rc {suites.returncode}")
    identity = OUT / "identity"
    pair("mmmu", "vpa@0,vpab@1", identity, ["--passes", "c1,c16", "--layer-dump", *DETERMINISTIC])
    for pass_name in ("c1", "c16"):
        run_logged(
            [sys.executable, str(BOX / "boot_identity_read.py"), str(identity / pass_name / "vpa" / "layers"),
             str(identity / pass_name / "vpab" / "layers")],
            cwd=paths["vpa"], env=card_env(0, paths["vpa"]), name=f"boot_identity_{pass_name}",
        )
    for round_name, arms in (("round1", "upx@0,vpa@1"), ("round2", "vpa@0,upx@1")):
        pair("mmmu", arms, OUT / "mmmu" / round_name, ["--passes", "c16"])
        pair("videoamme", arms, OUT / "videoamme" / round_name, ["--passes", "c16"])
    log("queue done")


if __name__ == "__main__":
    main()
