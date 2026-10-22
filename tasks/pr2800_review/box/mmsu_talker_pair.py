"""MMSU talker in stage 6's request shape on the full mmsu-ci-2000, two arms at the same time, one a card.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/mmsu_talker_pair.py --arms upk@0,phr@1
    --out /data/pr2800/<run>
Each arm serves the cookbook's colocated bf16 config on its card (pair_check.serve) and gets the 2000 requests with the
CI's talker prompt: text and audio, greedy thinker, 256 tokens, c16, audio saved. Both servers stop; one Qwen3-ASR
server on card 0 then transcribes each arm's audio and scores it against the arm's text as the CI does (text-audio
consistency: WER below 50 % over the corpus and the outputs above 50 %). The client and the scoring run in each tree's
own code (this file in its --client and --score modes). Writes <out>/<arm>/mmsu.json, <out>/<arm>/wer.json and
<out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REPO_ID = "zhaochenyang20/mmsu-ci-2000"
MAX_TOKENS = 256
CONCURRENCY = 16
ASR_CONCURRENCY = 4
PORT = 27000
MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def client(port: int, out: Path) -> None:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.eval.benchmark_omni_mmsu import run

    args = argparse.Namespace(
        base_url=None, host="127.0.0.1", port=port, model="qwen3-omni", modalities="text+audio",
        output_dir=str(out), max_samples=None, task_names=None, categories=None, prompt=MMSU_TTS_PROMPT,
        max_tokens=MAX_TOKENS, temperature=0.0, warmup=0, max_concurrency=CONCURRENCY, request_rate=float("inf"),
        save_audio=True, disable_tqdm=True, seed=None, fingerprint=False, lang="en", asr_device="cuda:0",
        asr_concurrency=ASR_CONCURRENCY, timeout_s=500)
    results = asyncio.run(run(args, samples=load_mmsu_samples(repo_id=REPO_ID), compute_wer=False))
    (out / "mmsu.json").write_text(json.dumps(results, indent=1))


def score(port: int, out: Path) -> None:
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    per_sample = json.loads((out / "mmsu.json").read_text())["per_sample"]
    wer = compute_text_audio_consistency_from_records(
        per_sample, "en", "cuda:0", audio_dir=str(out / "audio"), text_key="raw_response",
        asr_router_port=port, asr_concurrency=ASR_CONCURRENCY)
    (out / "wer.json").write_text(json.dumps(wer, indent=1))


def run_arm(out: Path, arm: str, card: int) -> int:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    log(out, f"{arm} card {card} serving")
    server = pair_check.serve(card, arm_tree, pair_check.MODEL, port, ["--config", pair_check.CONFIG, "--colocate"],
                              arm_out / "server.log")
    try:
        log(out, f"{arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--client", "--port", str(port), "--out", str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    log(out, f"{arm} card {card} client rc {rc}")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arm@card,arm@card")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out)
        return
    elif args.score:
        score(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    with ThreadPoolExecutor(max_workers=len(placements)) as pool:
        codes = list(pool.map(lambda placement: run_arm(args.out, *placement), placements))
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = pair_check.ASR_PORT
    asr = pair_check.serve(0, asr_tree, pair_check.ASR_MODEL, asr_port, [], args.out / "asr_server.log")
    summary = {}
    try:
        for (arm, _), code in zip(placements, codes):
            if code != 0:
                continue
            else:
                pass
            rc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--score", "--port", str(asr_port), "--out",
                 str(args.out / arm)],
                cwd=asr_tree, env=pair_check.card_env(0, asr_tree), stdout=open(args.out / arm / "score.log", "w"),
                stderr=subprocess.STDOUT).returncode
            log(args.out, f"{arm} score rc {rc}")
            mmsu = json.loads((args.out / arm / "mmsu.json").read_text())
            wer = json.loads((args.out / arm / "wer.json").read_text())["summary"] if rc == 0 else None
            summary[arm] = dict(accuracy=mmsu["accuracy"], speed=mmsu["speed"], wer=wer)
            log(args.out, f"{arm} accuracy {mmsu['accuracy'].get('overall_accuracy')} wer {wer}")
    finally:
        pair_check.stop(0, asr)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(args.out, "done")


if __name__ == "__main__":
    main()
