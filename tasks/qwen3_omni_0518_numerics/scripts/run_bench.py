# SPDX-License-Identifier: Apache-2.0
"""Run one CI stage's benchmark against a running server with the CI settings.

Usage:
    python run_bench.py videoamme --port P --out DIR [--concurrency 16] [--top-logprobs 5]
    python run_bench.py videomme-talker --port P --out DIR [--concurrency 16] [--max-samples 20]
    python run_bench.py mmsu --port P --out DIR [--concurrency 16] [--max-samples N]

videoamme mirrors tests/test_model/test_qwen3_omni_videoamme_ci.py (stage 9),
videomme-talker mirrors test_qwen3_omni_videomme_talker_ci.py (stage 8) with
the same short-answer prompt and speech output but without the inline WER
pass (the CLI would load Qwen3-ASR on cuda:0 next to the server), and mmsu
mirrors test_qwen3_omni_mmsu_ci.py (stage 5, text only). Each run writes the
same result JSON the CI artifacts contain, so ci_artifacts.py compare-local
reads it. --top-logprobs K (default 5) asks the chat endpoint for per-token
logprobs with K alternatives, which fills answer_margin and min_margin in
every per-sample record. Pass --top-logprobs 0 for CI-identical requests
without logprobs. Run from the omni checkout root with PYTHONPATH=$PWD.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def run_videoamme(args) -> None:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

    config = VideoEvalConfig(
        model="qwen3-omni",
        port=args.port,
        max_samples=args.max_samples or 50,
        max_concurrency=args.concurrency,
        output_dir=str(Path(args.out) / "videoamme"),
        repo_id=DATASETS["videoamme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=True,
        timeout_s=500,
        top_logprobs=args.top_logprobs,
    )
    results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
    s = results["summary"]
    print(
        f"videoamme accuracy={s['accuracy']} failed={s['failed']} qps={results['speed']['throughput_qps']} out_tokens_mean={results['speed']['output_tokens_mean']}"
    )


def run_videomme_talker(args) -> None:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.dataset.videomme import load_videomme_samples
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval

    short_answer_prompt = (
        "For the audio response, answer briefly in one sentence and end with "
        "'Answer: $LETTER'. Do not include step-by-step reasoning."
    )
    samples = load_videomme_samples(
        max_samples=args.max_samples or 20, repo_id=DATASETS["videomme-ci-50"]
    )
    for sample in samples:
        sample.prompt = f"{sample.prompt}\n{short_answer_prompt}"
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=args.port,
        max_samples=args.max_samples or 20,
        max_tokens=256,
        max_concurrency=args.concurrency,
        output_dir=str(Path(args.out) / "videomme_audio"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        disable_tqdm=True,
        timeout_s=500,
        top_logprobs=args.top_logprobs,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            samples=samples,
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default=str(Path(args.out) / "videomme_audio"),
            compute_wer=False,
        )
    )
    s, sp = results["summary"], results["speed"]
    print(
        f"videomme-talker accuracy={s['accuracy']} rtf_mean={sp['rtf_mean']} latency_mean_s={sp['latency_mean_s']} qps={sp['throughput_qps']}"
    )


def run_mmsu(args) -> None:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu_eval

    ns = argparse.Namespace(
        base_url=None,
        host="localhost",
        port=args.port,
        model="qwen3-omni",
        modalities="text",
        output_dir=str(Path(args.out) / "mmsu"),
        max_samples=args.max_samples,
        task_names=None,
        categories=None,
        prompt=None,
        max_tokens=32,
        temperature=0.0,
        warmup=0,
        max_concurrency=args.concurrency,
        request_rate=float("inf"),
        timeout_s=300,
        save_audio=False,
        disable_tqdm=True,
        seed=None,
        repo_id=DATASETS["mmsu-ci-2000"],
        lang="en",
        asr_device="cuda:0",
        top_logprobs=args.top_logprobs,
    )
    results = asyncio.run(run_mmsu_eval(ns, compute_wer=False))
    a, sp = results["accuracy"], results["speed"]
    print(
        f"mmsu accuracy={a['overall_accuracy']} failed={a.get('failed_samples', 0)} qps={sp['throughput_qps']} latency_mean_s={sp['latency_mean_s']}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("videoamme", "videomme-talker", "mmsu"))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--top-logprobs", type=int, default=5)
    args = p.parse_args(argv)
    if args.top_logprobs <= 0:
        args.top_logprobs = None
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {
        "videoamme": run_videoamme,
        "videomme-talker": run_videomme_talker,
        "mmsu": run_mmsu,
    }[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
