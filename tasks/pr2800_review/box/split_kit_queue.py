"""Code2wav kit pairs on card 0, for the split code2wav PRs.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/split_kit_queue.py --out /data/pr2800/<run>
    --kit name,tree,rows,decode,frames [--kit ...]
Card 0 runs every kit job in order (code2wav_kit_slots.py beside this file), so each shape is compared on one card,
pinned to the card's cores as pair_check.py pins servers, with the tree's own kit (.claude/skills/omni-gpu-deep-dive
/scripts) on PYTHONPATH. Writes <out>/queue.log and one directory per kit job; then the median replay of every kit
job (kit_span_table).
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from kit_replay_span_read import replays, span  # noqa: E402

WORKTREES = Path("/workspace/sglang-omni/.tmp/wt")
KIT = ".claude/skills/omni-gpu-deep-dive/scripts"
CARD_CPUS = {
    int(card): cpus
    for card, cpus in (
        entry.split("=", 1)
        for entry in os.environ.get("PR2800_CARD_CPUS", "0=56-63,168-175;1=64-71,176-183").split(";")
    )
}
NODE = os.environ.get("PR2800_NODE", "1")


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with open(out / "queue.log", "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def run(card: int, tree: Path, command: list[str], log_path: Path) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONPATH": f"{tree}:{tree / KIT}"}
    with open(log_path, "w") as handle:
        handle.write(f"tree {tree.name} {subprocess.run(['git', 'rev-parse', 'HEAD:sglang_omni'], cwd=tree, capture_output=True, text=True).stdout.strip()}\n")
        handle.flush()
        return subprocess.run(
            ["numactl", f"--physcpubind={CARD_CPUS[card]}", f"--membind={NODE}", *command],
            cwd=tree,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def kit_jobs(out: Path, jobs: list[str]) -> None:
    for job in jobs:
        name, tree, rows, decode, frames = job.split(",")
        job_dir = out / f"kit_{name}"
        job_dir.mkdir(parents=True, exist_ok=True)
        log(out, f"card 0 kit {name} start")
        rc = run(
            0,
            WORKTREES / tree,
            [sys.executable, str(BOX / "code2wav_kit_slots.py"), "--rows", rows, "--decode", decode, "--frames",
             frames, "--output-dir", str(job_dir)],
            job_dir / "driver.log",
        )
        log(out, f"card 0 kit {name} rc {rc}")


def kit_span_table(out: Path, jobs: list[str]) -> None:
    for job in jobs:
        name = job.split(",")[0]
        try:
            groups = replays(out / f"kit_{name}")
        except StopIteration:
            log(out, f"kit {name}: no formal trace")
            continue
        spans = [span(group) for group in groups]
        busy = [sum(event["dur"] for event in group) for group in groups]
        log(out, f"kit {name}: {len(groups)} replays, span median {statistics.median(spans):.1f} us, device time "
                 f"median {statistics.median(busy):.1f} us, events {statistics.median(len(group) for group in groups)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kit", action="append", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "driver.pgid").write_text(str(os.getpgid(0)))
    kit_jobs(args.out, args.kit)
    kit_span_table(args.out, args.kit)
    log(args.out, "done")


if __name__ == "__main__":
    main()
