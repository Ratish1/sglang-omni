# SPDX-License-Identifier: Apache-2.0
"""Every Qwen3-Omni input and output pairing against one running server, scored after it stops.

usage: python3 modality_matrix.py serve --port 20000 --out /data/runtime-perf/l28/q_a1/matrix [--cells mmsu_talker,...]
           [--concurrency 64]
       python3 modality_matrix.py score --asr-port 21000 --out /data/runtime-perf/l28/q_a1/matrix

Run only the cells a change can move (read from the code first; an exact change needs none). serve runs the family
benchmarks the Qwen3-Omni CI runs, as the CI calls them, at c64 by default: text output for image (MMMU,
full validation), audio (MMSU, full), video (Video-MME CI sample) and video with its sound (Video-AMME CI sample);
audio output for the same inputs on the CI's talker sets (MMMU CI sample with its reasoning prompt, so replies run
long; the MMSU CI set and Video-MME CI sample with the CI's short answer prompt; Video-AMME CI sample with its fixed
prompt), and SeedTTS EN in full with and without the reference voice. Talker cells save their audio and skip WER, as the CI does, so the
card holds only the server. score transcribes every saved audio with a Qwen3-ASR server at 64 in flight: SeedTTS
against its text (WER, then speaker similarity on the voice-clone cell), the talker cells against the model's own
text reply (text-audio consistency). Each cell writes <out>/<cell>/result.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

CONCURRENCY = 64
ASR_CONCURRENCY = 64
TALKER_MAX_TOKENS = 256
VIDEO_PRESETS = {"video_fps": 2, "video_max_frames": 128, "video_max_pixels": 401408}
REASONING_TALKER_PROMPT = (
    "{intro}\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)
SHORT_ANSWER_PROMPT = (
    "For the audio response, answer briefly in one sentence and end with "
    "'Answer: $LETTER'. Do not include step-by-step reasoning."
)
TEXT_CELLS = ("mmmu", "mmsu", "videomme", "videoamme")
TALKER_CELLS = ("mmmu_talker", "mmsu_talker", "videomme_talker", "videoamme_talker")
SEEDTTS_CELLS = ("seedtts_voice_clone", "seedtts_text")


def plain(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    elif isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    else:
        return value


def write_result(out: Path, cell: str, result: dict, seconds: float) -> None:
    (out / cell).mkdir(parents=True, exist_ok=True)
    (out / cell / "result.json").write_text(json.dumps({**plain(result), "cell_seconds": seconds}, indent=1))
    print(f"{cell} done in {seconds:.0f} s", flush=True)


def run_mmmu(port: int, out: Path, talker: bool) -> dict:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval

    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=port,
        max_concurrency=CONCURRENCY,
        output_dir=str(out),
        repo_id=DATASETS["mmmu-ci-50"] if talker else None,
        enable_audio=talker,
        max_tokens=TALKER_MAX_TOKENS if talker else 2048,
        prompt_override=(
            REASONING_TALKER_PROMPT.format(intro="Look at the image and answer the multiple-choice question.")
            if talker
            else None
        ),
        timeout_s=500,
        disable_tqdm=True,
    )
    return asyncio.run(run_mmmu_eval(config, compute_wer=False))


def run_mmsu(port: int, out: Path, talker: bool) -> dict:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu_eval

    args = argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model="qwen3-omni",
        modalities="text+audio" if talker else "text",
        output_dir=str(out),
        max_samples=None,
        task_names=None,
        categories=None,
        prompt=(
            f"Listen to the audio and answer the multiple-choice question.\n{SHORT_ANSWER_PROMPT}"
            if talker
            else None
        ),
        max_tokens=TALKER_MAX_TOKENS if talker else 32,
        temperature=0.0,
        warmup=None,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        save_audio=talker,
        disable_tqdm=True,
        seed=None,
        fingerprint=False,
        lang="en",
        asr_device="cuda:0",
        asr_concurrency=ASR_CONCURRENCY,
        timeout_s=500,
        repo_id=DATASETS["mmsu-ci-2000"] if talker else DATASETS["mmsu"],
    )
    samples = load_mmsu_samples(repo_id=args.repo_id)
    return asyncio.run(run_mmsu_eval(args, samples=samples, compute_wer=False))


def run_video(port: int, out: Path, talker: bool, with_sound: bool) -> dict:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.dataset.videomme import load_videoamme_samples, load_videomme_samples
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval

    repo_id = DATASETS["videoamme-ci-50" if with_sound else "videomme-ci-50"]
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=port,
        max_tokens=TALKER_MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(out),
        repo_id=repo_id,
        enable_audio=talker,
        timeout_s=500,
        disable_tqdm=True,
        **VIDEO_PRESETS,
    )
    if with_sound:
        return asyncio.run(run_videoamme_eval(config, samples=load_videoamme_samples(repo_id=repo_id), compute_wer=False))
    else:
        samples = load_videomme_samples(repo_id=repo_id)
        if talker:
            for sample in samples:
                sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
        else:
            pass
        return asyncio.run(
            run_video_eval(
                config,
                samples=samples,
                task_label="Video-MME",
                output_filename="videomme_results.json",
                audio_output_dir_default=str(out / "audio"),
                compute_wer=False,
            )
        )


def run_seedtts(port: int, out: Path, voice_clone: bool) -> dict:
    command = [
        sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", "--host", "127.0.0.1", "--port", str(port),
        "--generate-only", "--stream", *(["--voice-clone"] if voice_clone else []), "--lang", "en", "--warmup", "1",
        "--max-concurrency", str(CONCURRENCY), "--output-dir", str(out), "--disable-tqdm",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (out / "generate.log").write_text(completed.stdout + completed.stderr)
    return {"returncode": completed.returncode}


def serve(args: argparse.Namespace) -> None:
    global CONCURRENCY
    CONCURRENCY = args.concurrency
    out = Path(args.out)
    for cell in args.cells.split(","):
        cell_out = out / cell
        cell_out.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        if cell in ("mmmu", "mmmu_talker"):
            result = run_mmmu(args.port, cell_out, talker=cell == "mmmu_talker")
        elif cell in ("mmsu", "mmsu_talker"):
            result = run_mmsu(args.port, cell_out, talker=cell == "mmsu_talker")
        elif cell in ("videomme", "videomme_talker", "videoamme", "videoamme_talker"):
            result = run_video(
                args.port, cell_out, talker=cell.endswith("_talker"), with_sound=cell.startswith("videoamme")
            )
        elif cell in SEEDTTS_CELLS:
            result = run_seedtts(args.port, cell_out, voice_clone=cell == "seedtts_voice_clone")
        else:
            raise ValueError(f"unknown cell {cell}")
        write_result(out, cell, result, time.monotonic() - start)


def score(args: argparse.Namespace) -> None:
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    out = Path(args.out)
    for cell in TALKER_CELLS:
        result_path = out / cell / "result.json"
        if not result_path.exists():
            continue
        else:
            pass
        result = json.loads(result_path.read_text())
        wer = compute_text_audio_consistency_from_records(
            result["per_sample"],
            "en",
            "cuda:0",
            audio_dir=str(out / cell / "audio"),
            asr_router_port=args.asr_port,
            asr_concurrency=ASR_CONCURRENCY,
        )
        (out / cell / "wer.json").write_text(json.dumps(plain(wer), indent=1))
        print(f"{cell} wer {wer.get('summary')}", flush=True)
    for cell in SEEDTTS_CELLS:
        cell_out = out / cell
        if not cell_out.exists():
            continue
        else:
            pass
        command = [
            sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", "--transcribe-only", "--lang", "en",
            *(["--voice-clone"] if cell == "seedtts_voice_clone" else []), "--port", str(args.asr_port),
            "--asr-concurrency", str(ASR_CONCURRENCY), "--output-dir", str(cell_out),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (cell_out / "transcribe.log").write_text(completed.stdout + completed.stderr)
        print(f"{cell} transcribe rc {completed.returncode}", flush=True)


def similarity(args: argparse.Namespace) -> None:
    cell_out = Path(args.out) / "seedtts_voice_clone"
    command = [
        sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", "--similarity-only", "--lang", "en",
        "--voice-clone", "--device", "cuda:0", "--output-dir", str(cell_out),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (cell_out / "similarity.log").write_text(completed.stdout + completed.stderr)
    print(f"seedtts_voice_clone similarity rc {completed.returncode}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--port", type=int, required=True)
    serve_parser.add_argument("--out", required=True)
    serve_parser.add_argument("--cells", default=",".join(SEEDTTS_CELLS + TEXT_CELLS + TALKER_CELLS))
    serve_parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--asr-port", type=int, required=True)
    score_parser.add_argument("--out", required=True)
    similarity_parser = commands.add_parser("similarity")
    similarity_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    elif args.command == "score":
        score(args)
    else:
        similarity(args)


if __name__ == "__main__":
    main()
