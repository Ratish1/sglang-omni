"""Video-MME in CI stages 7 and 8's shape, arm after arm on one card pair, each stage on a fresh server.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/videomme_disagg.py --arms upv,vrt --cards 0+1
    --stages 8,7 --passes 5 --out /data/pr2800/<run>
Each arm serves the CI's bf16 disaggregated speech server (examples/run_qwen3_omni_speech_server.py: the thinker and
the encoders on the first card, the talker and code2wav on the second, mem fractions 0.82 and 0.40) once per stage,
and sends that stage's requests --passes times on it: stage 7 test_qwen3_omni_videomme_ci.py's (videomme-ci-50, 50
samples, text out, c16), stage 8 test_qwen3_omni_videomme_talker_ci.py's (its first 20 samples with the short-answer
prompt, text and audio out, 256 tokens, c16). The client runs in each tree's own code (this file in its --client
mode). After the arms, one Qwen3-ASR server on the first card scores every stage 8 pass as the CI does. Writes
<out>/<arm>/s<stage>/p<index>/videomme.json (and wer.json for stage 8) and <out>/summary.json.
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

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVE_ARGS = [
    "--model-name", "qwen3-omni", "--thinker-max-seq-len", "32768", "--gpu-thinker", "0", "--gpu-image-encoder", "0",
    "--gpu-audio-encoder", "0", "--gpu-talker", "1", "--gpu-code2wav", "1", "--thinker-mem-fraction-static", "0.82",
    "--talker-mem-fraction-static", "0.40",
]
# stage: (samples, whether the talker speaks)
STAGES = {7: (50, False), 8: (20, True)}
SHORT_ANSWER_PROMPT = (
    "For the audio response, answer briefly in one sentence and end with 'Answer: $LETTER'. Do not include "
    "step-by-step reasoning."
)
RESULTS_NAME = "videomme.json"
PORT = 30000


def client(port: int, out: Path, stage: int) -> None:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.dataset.videomme import load_videomme_samples
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval

    max_samples, is_speech = STAGES[stage]
    config = VideoEvalConfig(
        model="qwen3-omni", host="127.0.0.1", port=port, max_samples=max_samples, max_concurrency=16,
        output_dir=str(out), repo_id=DATASETS["videomme-ci-50"], video_fps=2, video_max_frames=128,
        video_max_pixels=401408, enable_audio=is_speech, disable_tqdm=True, timeout_s=500)
    samples = None
    if is_speech:
        samples = load_videomme_samples(max_samples=max_samples, repo_id=DATASETS["videomme-ci-50"])
        for sample in samples:
            sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
    else:
        pass
    results = asyncio.run(run_video_eval(
        config, samples=samples, task_label="Video-MME", output_filename="videomme_results.json",
        audio_output_dir_default="results/videomme_audio", compute_wer=False))
    (out / RESULTS_NAME).write_text(json.dumps(results, indent=1, default=str))


def run_stage(out: Path, arm: str, cards: tuple[int, int], stage: int, passes: int) -> list[tuple[Path, int]]:
    stage_out = out / arm / f"s{stage}"
    stage_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * cards[0]
    videoamme_talker_tp2.log(out, f"{arm} stage {stage} cards {cards} serving")
    server = videoamme_talker_tp2.serve(
        arm_tree, stage_out / "server.log", cards, port, None, MODEL, SERVE_ARGS)
    codes = []
    try:
        for index in range(passes):
            pass_out = stage_out / f"p{index}"
            pass_out.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--client", "--port", str(port), "--stage",
                 str(stage), "--out", str(pass_out)],
                cwd=arm_tree, env={**os.environ, "PYTHONPATH": str(arm_tree)},
                stdout=open(pass_out / "client.log", "w"), stderr=subprocess.STDOUT).returncode
            summary = json.loads((pass_out / RESULTS_NAME).read_text())["summary"] if rc == 0 else {}
            videoamme_talker_tp2.log(
                out, f"{arm} s{stage}/p{index} rc {rc} accuracy {summary.get('accuracy')} failed "
                f"{summary.get('failed')}")
            codes.append((pass_out, rc))
    finally:
        videoamme_talker_tp2.stop(server, cards)
    return codes


def score_speech_passes(out: Path, card: int, speech_passes: list[Path]) -> None:
    """One Qwen3-ASR server on the card scores every stage 8 pass as the CI does."""
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = pair_check.ASR_PORT + 1000 + 200 * card
    asr = pair_check.serve(card, asr_tree, pair_check.ASR_MODEL, asr_port, [], out / "asr_server.log")
    try:
        for pass_out in speech_passes:
            rc = subprocess.run(
                [sys.executable, str(BOX / "videoamme_talker_tp2.py"), "--score", "--port", str(asr_port), "--out",
                 str(pass_out), "--results-name", RESULTS_NAME],
                cwd=asr_tree, env=pair_check.card_env(card, asr_tree),
                stdout=open(pass_out / "score.log", "w"), stderr=subprocess.STDOUT).returncode
            wer = json.loads((pass_out / "wer.json").read_text())["summary"] if rc == 0 else None
            videoamme_talker_tp2.log(
                out, f"{pass_out.relative_to(out)} score rc {rc} wer below 50 "
                f"{None if wer is None else wer['wer_below_50_corpus']} above 50 "
                f"{None if wer is None else wer['n_above_50_pct_wer']}")
    finally:
        pair_check.stop(card, asr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arms in order, each on the card pair in turn")
    parser.add_argument("--cards", default="0+1", help="first card (thinker) + second card (talker)")
    parser.add_argument("--stages", default="8,7")
    parser.add_argument("--passes", type=int, default=1, help="client passes per server boot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out, args.stage)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    first_card, second_card = (int(card) for card in args.cards.split("+"))
    cards = (first_card, second_card)
    stages = [int(stage) for stage in args.stages.split(",")]
    speech_passes = []
    for arm in args.arms.split(","):
        for stage in stages:
            codes = run_stage(args.out, arm, cards, stage, args.passes)
            if STAGES[stage][1]:
                speech_passes += [pass_out for pass_out, rc in codes if rc == 0]
            else:
                pass
    if speech_passes:
        score_speech_passes(args.out, first_card, speech_passes)
    else:
        pass
    summary = {}
    for path in sorted(args.out.glob(f"*/s*/p*/{RESULTS_NAME}")):
        record = json.loads(path.read_text())
        wer_path = path.parent / "wer.json"
        summary[str(path.parent.relative_to(args.out))] = dict(
            summary=record["summary"], speed=record.get("speed"),
            wer=json.loads(wer_path.read_text())["summary"] if wer_path.exists() else None)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    videoamme_talker_tp2.log(args.out, "done")


if __name__ == "__main__":
    main()
