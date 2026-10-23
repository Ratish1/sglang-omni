"""HF's Qwen3-Omni on SocialOmni level 1 (hf_standard_reference.py --task socialomni), scored and paired item by item
with omni's arms.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/socialomni_hf.py --cards 0,1 --out /data/pr2800/<run>
    [--sampled-fps] [--omni <arm>=<result json> ...]
1. Samples: every level 1 item from the benchmark's own loader, with its own prompt (build_level1_prompt), in main's
   tree.
2. The reference, sharded over the cards, in the isolated environment: the cookbook with the video's audio
   interleaved; --sampled-fps hands the processor each video's sampled fps instead.
3. Each response's last line parsed by the benchmark's parse_choice; accuracy, and against each --omni arm the items
   only one side gets right and the exact McNemar p. Writes <out>/socialomni_samples.json, <out>/scored.json and
   <out>/summary.txt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from hf_standard_queue import REFERENCE_PATH  # noqa: E402

DATASET_ROOT = "/data/pr2800/socialomni"


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def write_samples(path: Path) -> None:
    from benchmarks.dataset.socialomni import load_socialomni_level1_samples
    from benchmarks.tasks.socialomni_protocol import build_level1_prompt

    samples = load_socialomni_level1_samples(DATASET_ROOT)
    path.write_text(json.dumps({"samples": [
        dict(sample_id=sample.sample_id, video_path=sample.video_path, prompt=build_level1_prompt(sample),
             gold_answer=sample.gold_answer, visibility=sample.visibility)
        for sample in samples]}, indent=1))


def score(out: Path, omni: list[str]) -> list[str]:
    from benchmarks.tasks.socialomni_protocol import parse_choice

    spec = {sample["sample_id"]: sample for sample in json.loads((out / "socialomni_samples.json").read_text())["samples"]}
    records = [record for path in sorted(out.glob("socialomni_[0-9]*.json")) for record in json.loads(path.read_text())]
    hf = {}
    for record in records:
        lines = record["response"].strip().splitlines()
        hf[record["sample_id"]] = parse_choice(lines[-1], ("A", "B", "C", "D")) if lines else ""
    correct = {sample_id: choice == spec[sample_id]["gold_answer"] for sample_id, choice in hf.items()}
    (out / "scored.json").write_text(json.dumps(
        {sample_id: dict(choice=choice, is_correct=correct[sample_id]) for sample_id, choice in hf.items()}, indent=1))
    lines = [f"HF: {sum(correct.values())} of {len(correct)} correct, {sum(not c for c in hf.values())} unparsed"]
    for entry in omni:
        arm, path = entry.split("=", 1)
        arm_choice = {record["sample_id"]: record["predicted_answer"]
                      for record in json.loads(Path(path).read_text())["per_sample"]["level1"]}
        arm_correct = {sample_id: bool(choice) and choice == spec[sample_id]["gold_answer"]
                       for sample_id, choice in arm_choice.items()}
        shared = sorted(set(arm_correct) & set(correct))
        hf_only = sum(correct[i] and not arm_correct[i] for i in shared)
        arm_only = sum(arm_correct[i] and not correct[i] for i in shared)
        agree = sum(hf[i] == arm_choice[i] for i in shared)
        discordant = hf_only + arm_only
        p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(hf_only, arm_only) + 1))
                      / 2 ** discordant) if discordant else 1.0
        lines.append(f"{arm}: {sum(arm_correct[i] for i in shared)} correct; HF right only {hf_only}, {arm} right "
                     f"only {arm_only}, McNemar p {p_value:.4f}; same choice as HF on {agree} of {len(shared)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampled-fps", action="store_true")
    parser.add_argument("--omni", nargs="*", default=[], help="arm=<socialomni result json> to pair with")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tree = pair_check.tree(pair_check.ASR_TREE)
    if args.samples_only:
        write_samples(args.out / "socialomni_samples.json")
        return
    elif args.score_only:
        lines = score(args.out, args.omni)
        (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            log(args.out, line)
        return
    else:
        pass
    env = {**os.environ, "PYTHONPATH": str(tree)}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--samples-only", "--out", str(args.out)],
                   cwd=tree, env=env, check=True)
    log(args.out, "samples written")
    cards = [int(card) for card in args.cards.split(",")]

    def run_shard(shard: int) -> int:
        card = cards[shard]
        log(args.out, f"shard {shard} card {card} start")
        rc = subprocess.run(
            [sys.executable, str(BOX / "hf_standard_reference.py"), "--task", "socialomni", "--samples",
             str(args.out / "socialomni_samples.json"), "--shard", str(shard), "--shards", str(len(cards)), "--out",
             str(args.out), *(["--sampled-fps"] if args.sampled_fps else [])],
            cwd="/tmp", env={**os.environ, "PYTHONPATH": REFERENCE_PATH, "CUDA_VISIBLE_DEVICES": str(card)},
            stdout=open(args.out / f"socialomni_{shard}.log", "w"), stderr=subprocess.STDOUT).returncode
        log(args.out, f"shard {shard} card {card} rc {rc}")
        return rc

    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        codes = list(pool.map(run_shard, range(len(cards))))
    if any(codes):
        log(args.out, f"stopped: rcs {codes}")
        return
    else:
        pass
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--score-only", "--out", str(args.out),
                    "--omni", *args.omni], cwd=tree, env=env, check=True)
    log(args.out, "done")


if __name__ == "__main__":
    main()
