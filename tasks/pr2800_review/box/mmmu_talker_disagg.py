"""The CI's MMMU talker stage in its bf16 disaggregated shape: one arm on one card pair, several passes on one boot.

Run on the box, two at once on the two card pairs (PR2800_NODE the pair's NUMA node), then swapped:
    python3 /workspace/box/tasks/pr2800_review/box/mmmu_talker_disagg.py --arm upy --cards 0+1 --passes 4
        --out /data/<run>/r1
    python3 mmmu_talker_disagg.py --read /data/<run> --arms upy,vpg
The arm serves the CI's bf16 disaggregated speech server (videomme_disagg.SERVE_ARGS: the thinker and encoders on the
first card, the talker and code2wav on the second) once and sends test_qwen3_omni_mmmu_talker_ci.py's requests
--passes times: its MAX_SAMPLES mmmu-ci-50 samples with its prompt, MAX_TOKENS and CONCURRENCY, text and audio out,
greedy. The client runs in the arm's tree and imports those constants from the arm's own test (this file in its
--client mode). Writes <out>/<arm>/p<index>/mmmu.json. --read pools every round under the directory: per arm the pass
scores (correct of the stage's samples), the passes under the stage's 0.75, and per sample the correct passes of each
arm, listing the samples whose rates differ.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
import videoamme_talker_tp2  # noqa: E402
import videomme_disagg  # noqa: E402

RESULTS_NAME = "mmmu.json"
STAGE_ACCURACY = 0.75


def client(port: int, out: Path) -> None:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
    from tests.test_model import test_qwen3_omni_mmmu_talker_ci as stage

    config = MMMUEvalConfig(
        model="qwen3-omni",
        host="127.0.0.1",
        port=port,
        max_samples=stage.MAX_SAMPLES,
        max_tokens=stage.MAX_TOKENS,
        max_concurrency=stage.CONCURRENCY,
        output_dir=str(out),
        enable_audio=True,
        disable_tqdm=True,
        repo_id=DATASETS["mmmu-ci-50"],
        prompt_override=stage.MMMU_TTS_PROMPT,
        timeout_s=500,
    )
    results = asyncio.run(run_mmmu_eval(config, compute_wer=False))
    (out / RESULTS_NAME).write_text(json.dumps(results, indent=1, default=str))


def run_arm(out: Path, arm: str, cards: tuple[int, int], passes: int) -> None:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = videomme_disagg.PORT + 200 * cards[0]
    videoamme_talker_tp2.log(out, f"{arm} cards {cards} serving")
    server = videoamme_talker_tp2.serve(
        arm_tree,
        arm_out / "server.log",
        cards,
        port,
        None,
        videomme_disagg.MODEL,
        videomme_disagg.SERVE_ARGS,
    )
    try:
        for index in range(passes):
            pass_out = arm_out / f"p{index}"
            pass_out.mkdir(parents=True, exist_ok=True)
            with open(pass_out / "client.log", "w") as client_log:
                rc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--client",
                        "--port",
                        str(port),
                        "--out",
                        str(pass_out),
                    ],
                    cwd=arm_tree,
                    env={**os.environ, "PYTHONPATH": str(arm_tree)},
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                ).returncode
            summary = (
                json.loads((pass_out / RESULTS_NAME).read_text())["summary"]
                if rc == 0
                else {}
            )
            videoamme_talker_tp2.log(
                out,
                f"{arm} p{index} rc {rc} accuracy {summary.get('accuracy')} correct {summary.get('correct')} "
                f"failed {summary.get('failed')}",
            )
    finally:
        videoamme_talker_tp2.stop(server, cards)


def read(root: Path, arms: list[str]) -> None:
    scores: dict[str, list[float]] = {arm: [] for arm in arms}
    correct: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    letters: dict[str, dict[str, dict[str, int]]] = {arm: {} for arm in arms}
    for arm in arms:
        for path in sorted(root.glob(f"r*/{arm}/p*/{RESULTS_NAME}")):
            record = json.loads(path.read_text())
            scores[arm].append(record["summary"]["accuracy"])
            for sample in record["per_sample"]:
                key = str(sample["sample_id"])
                correct[arm][key] = correct[arm].get(key, 0) + int(
                    bool(sample["is_correct"])
                )
                predicted = letters[arm].setdefault(key, {})
                predicted[sample["predicted"]] = (
                    predicted.get(sample["predicted"], 0) + 1
                )
        values = scores[arm]
        print(
            f"{arm}: {len(values)} passes, scores {sorted(values)}, mean {sum(values) / max(len(values), 1):.4f}, "
            f"under {STAGE_ACCURACY}: {sum(value < STAGE_ACCURACY for value in values)}"
        )
    samples = sorted(set().union(*(correct[arm] for arm in arms)))
    differing = [
        key
        for key in samples
        if len({correct[arm].get(key, 0) / max(len(scores[arm]), 1) for arm in arms})
        > 1
    ]
    print(
        f"samples {len(samples)}, with different correct rates across arms {len(differing)}"
    )
    for key in differing:
        print(
            f"  {key}: "
            + ", ".join(
                f"{arm} {correct[arm].get(key, 0)}/{len(scores[arm])} {letters[arm].get(key, {})}"
                for arm in arms
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm")
    parser.add_argument(
        "--cards", default="0+1", help="first card (thinker) + second card (talker)"
    )
    parser.add_argument(
        "--passes", type=int, default=1, help="client passes per server boot"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--read", type=Path)
    parser.add_argument("--arms")
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out)
    elif args.read is not None:
        read(args.read, args.arms.split(","))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        first_card, second_card = (int(card) for card in args.cards.split("+"))
        run_arm(args.out, args.arm, (first_card, second_card), args.passes)


if __name__ == "__main__":
    main()
