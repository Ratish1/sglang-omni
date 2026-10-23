"""HF's Qwen3-Omni at batch 1 on a text-answer benchmark, scored by the benchmark's own parser and paired item by item
with omni's arms.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/hf_text_pairing.py --benchmark videoamme
    --cards 0,1 --out /data/pr2800/<run> [--sampled-fps] [--omni <arm>=<result json> ...]
Benchmarks:
- videoamme: CI stage 9's requests (the first 50 of videoamme-ci-50, the video at 2 fps with at most 128 frames of
  401408 pixels, the spoken question, the benchmark's request text), parsed by parse_multi_choice_response; an omni
  result is a videoamme.json.
- socialomni: SocialOmni level 1 (2000 items, the video's own audio interleaved), parsed by parse_choice on the last
  line; an omni result is a socialomni-*.json.
1. Samples from the benchmark's own loader, in main's tree.
2. The reference (hf_standard_reference.py, task videotext or socialomni), sharded over the cards, in the isolated
   environment; --sampled-fps hands the processor each video's sampled fps (float positions) where the cookbook passes
   none.
3. Accuracy, and against each --omni arm: the items whose parsed answers agree, the items only one side gets right and
   the exact McNemar p. Writes <out>/samples.json, <out>/scored.json and <out>/summary.txt.
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

MAIN_TREE = "upv"
SOCIALOMNI_ROOT = "/data/pr2800/socialomni"
VIDEOAMME_SAMPLES = 50
REFERENCE_TASKS = {"videoamme": "videotext", "socialomni": "socialomni"}


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def write_samples(benchmark: str, path: Path) -> None:
    if benchmark == "videoamme":
        from benchmarks.dataset.prepare import DATASETS
        from benchmarks.dataset.videomme import load_videoamme_samples
        from benchmarks.tasks.video_understanding import VIDEOAMME_REQUEST_TEXT

        samples = load_videoamme_samples(VIDEOAMME_SAMPLES, repo_id=DATASETS["videoamme-ci-50"])
        rows = [dict(sample_id=sample.sample_id, video_path=sample.video_path, audio_path=sample.audio_path)
                for sample in samples]
        path.write_text(json.dumps({"prompt": VIDEOAMME_REQUEST_TEXT, "samples": rows}, indent=1))
    else:
        from benchmarks.dataset.socialomni import load_socialomni_level1_samples
        from benchmarks.tasks.socialomni_protocol import build_level1_prompt

        samples = load_socialomni_level1_samples(SOCIALOMNI_ROOT)
        rows = [dict(sample_id=sample.sample_id, video_path=sample.video_path, prompt=build_level1_prompt(sample))
                for sample in samples]
        path.write_text(json.dumps({"samples": rows}, indent=1))


def reference_answers(benchmark: str, out: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Each item's parsed reference answer and its gold answer."""
    records = [record for path in sorted(out.glob(f"{REFERENCE_TASKS[benchmark]}_[0-9]*.json"))
               for record in json.loads(path.read_text())]
    if benchmark == "videoamme":
        from benchmarks.dataset.prepare import DATASETS
        from benchmarks.dataset.videomme import load_videoamme_samples
        from benchmarks.tasks.video_understanding import parse_multi_choice_response

        samples = {sample.sample_id: sample
                   for sample in load_videoamme_samples(VIDEOAMME_SAMPLES, repo_id=DATASETS["videoamme-ci-50"])}
        answers = {record["sample_id"]: parse_multi_choice_response(
            record["response"], samples[record["sample_id"]].all_choices, samples[record["sample_id"]].index2ans)[0]
            for record in records}
        return answers, {sample_id: sample.answer for sample_id, sample in samples.items()}
    else:
        from benchmarks.dataset.socialomni import load_socialomni_level1_samples
        from benchmarks.tasks.socialomni_protocol import parse_choice

        answers = {}
        for record in records:
            lines = record["response"].strip().splitlines()
            answers[record["sample_id"]] = parse_choice(lines[-1], ("A", "B", "C", "D")) if lines else ""
        return answers, {sample.sample_id: sample.gold_answer
                         for sample in load_socialomni_level1_samples(SOCIALOMNI_ROOT)}


def omni_answers(benchmark: str, path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text())
    if benchmark == "videoamme":
        return {record["sample_id"]: record["predicted"] for record in payload["per_sample"]}
    else:
        return {record["sample_id"]: record["predicted_answer"] for record in payload["per_sample"]["level1"]}


def score(benchmark: str, out: Path, omni: list[str]) -> list[str]:
    hf, gold = reference_answers(benchmark, out)
    hf_correct = {item: bool(answer) and answer == gold[item] for item, answer in hf.items()}
    (out / "scored.json").write_text(json.dumps(
        {item: dict(answer=answer, is_correct=hf_correct[item]) for item, answer in hf.items()}, indent=1))
    lines = [f"HF: {sum(hf_correct.values())} of {len(hf)} correct, {sum(not answer for answer in hf.values())} "
             f"unparsed"]
    for entry in omni:
        arm, path = entry.split("=", 1)
        answers = omni_answers(benchmark, path)
        shared = sorted(set(answers) & set(hf))
        arm_correct = {item: bool(answers[item]) and answers[item] == gold[item] for item in shared}
        hf_only = sum(hf_correct[item] and not arm_correct[item] for item in shared)
        arm_only = sum(arm_correct[item] and not hf_correct[item] for item in shared)
        discordant = hf_only + arm_only
        p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(hf_only, arm_only) + 1))
                      / 2 ** discordant) if discordant else 1.0
        differ = [item for item in shared if answers[item] != hf[item]]
        lines.append(f"{arm}: {sum(arm_correct.values())} correct; same answer as HF on {len(shared) - len(differ)} of "
                     f"{len(shared)}; HF right only {hf_only}, {arm} right only {arm_only}, McNemar p {p_value:.4f}; "
                     f"differing {differ[:20]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(REFERENCE_TASKS), required=True)
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampled-fps", action="store_true")
    parser.add_argument("--omni", nargs="*", default=[], help="arm=<omni result json> to pair with")
    parser.add_argument("--samples-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.samples_only:
        write_samples(args.benchmark, args.out / "samples.json")
        return
    elif args.score_only:
        lines = score(args.benchmark, args.out, args.omni)
        (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            log(args.out, line)
        return
    else:
        pass
    tree = pair_check.tree(MAIN_TREE)
    env = {**os.environ, "PYTHONPATH": str(tree)}
    this = [sys.executable, str(Path(__file__).resolve()), "--benchmark", args.benchmark, "--out", str(args.out)]
    subprocess.run([*this, "--samples-only"], cwd=tree, env=env, check=True)
    log(args.out, "samples written")
    cards = [int(card) for card in args.cards.split(",")]

    def run_shard(shard: int) -> int:
        card = cards[shard]
        log(args.out, f"shard {shard} card {card} start")
        rc = subprocess.run(
            [sys.executable, str(BOX / "hf_standard_reference.py"), "--task", REFERENCE_TASKS[args.benchmark],
             "--samples", str(args.out / "samples.json"), "--shard", str(shard), "--shards", str(len(cards)),
             "--out", str(args.out), *(["--sampled-fps"] if args.sampled_fps else [])],
            cwd="/tmp", env={**os.environ, "PYTHONPATH": REFERENCE_PATH, "CUDA_VISIBLE_DEVICES": str(card)},
            stdout=open(args.out / f"reference_{shard}.log", "w"), stderr=subprocess.STDOUT).returncode
        log(args.out, f"shard {shard} card {card} rc {rc}")
        return rc

    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        codes = list(pool.map(run_shard, range(len(cards))))
    if any(codes):
        log(args.out, f"stopped: rcs {codes}")
        return
    else:
        pass
    subprocess.run([*this, "--score-only", "--omni", *args.omni], cwd=tree, env=env, check=True)
    log(args.out, "done")


if __name__ == "__main__":
    main()
