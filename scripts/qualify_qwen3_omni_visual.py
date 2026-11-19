# SPDX-License-Identifier: Apache-2.0
"""Exercise the visual CI inputs on a pinned native BF16 colocated worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from sglang_omni.utils import find_available_port
from tests.test_model import (
    test_qwen3_omni_mmmu_talker_ci,
    test_qwen3_omni_videoamme_talker_tp2_ci,
    test_qwen3_omni_videomme_talker_ci,
)

SEED = 20261119


async def run_workloads(port: int, output: Path) -> dict:
    mmmu = test_qwen3_omni_mmmu_talker_ci
    video = test_qwen3_omni_videomme_talker_ci
    mixed = test_qwen3_omni_videoamme_talker_tp2_ci
    reports = {}
    reports["image-text"] = await run_mmmu_eval(
        MMMUEvalConfig(
            model="qwen3-omni",
            host="127.0.0.1",
            port=port,
            max_samples=50,
            max_concurrency=mmmu.CONCURRENCY,
            warmup=2,
            seed=SEED,
            repo_id=DATASETS["mmmu-ci-50"],
            output_dir=str(output / "image-text"),
            disable_tqdm=True,
        ),
        compute_wer=False,
    )
    reports["image-speech"] = await run_mmmu_eval(
        MMMUEvalConfig(
            model="qwen3-omni",
            host="127.0.0.1",
            port=port,
            max_samples=mmmu.MAX_SAMPLES,
            max_tokens=mmmu.MAX_TOKENS,
            max_concurrency=mmmu.CONCURRENCY,
            seed=SEED,
            repo_id=DATASETS["mmmu-ci-50"],
            output_dir=str(output / "image-speech"),
            prompt_override=mmmu.MMMU_TTS_PROMPT,
            enable_audio=True,
            disable_tqdm=True,
            timeout_s=500,
        ),
        compute_wer=False,
    )
    video_options = {
        "model": "qwen3-omni",
        "host": "127.0.0.1",
        "port": port,
        "seed": SEED,
        "video_fps": 2,
        "video_max_frames": 128,
        "video_max_pixels": 401408,
        "enable_audio": True,
        "disable_tqdm": True,
        "timeout_s": 500,
    }
    reports["video-speech"] = await run_video_eval(
        VideoEvalConfig(
            **video_options,
            max_samples=video.MAX_SAMPLES,
            max_tokens=video.MAX_TOKENS,
            max_concurrency=video.CONCURRENCY,
            repo_id=DATASETS["videomme-ci-50"],
            output_dir=str(output / "video-speech"),
        ),
        samples=video.load_short_answer_samples(),
        task_label="Video-MME",
        output_filename="videomme_results.json",
        audio_output_dir_default=str(output / "video-speech"),
        compute_wer=False,
    )
    reports["video-audio-speech"] = await run_videoamme_eval(
        VideoEvalConfig(
            **video_options,
            max_samples=mixed.MAX_SAMPLES,
            max_tokens=mixed.MAX_TOKENS,
            max_concurrency=mixed.CONCURRENCY,
            repo_id=DATASETS["videoamme-ci-50"],
            output_dir=str(output / "video-audio-speech"),
        ),
        compute_wer=False,
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tree = arguments.tree.resolve(strict=True)
    checkpoint = arguments.checkpoint.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir()

    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(tree / "examples/configs/qwen3_omni_colocated_h200.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
        ],
        output / "server.log",
        port,
        timeout=600,
        env={"PYTHONPATH": str(tree)},
        strip_proxy=True,
    )
    try:
        reports = asyncio.run(run_workloads(port, output))
        (output / "visual-results.json").write_text(
            json.dumps(
                {
                    "serving_revision": subprocess.check_output(
                        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "checkpoint": checkpoint.name,
                    "reports": reports,
                },
                indent=2,
            )
        )
        for name, report in reports.items():
            assert (
                report["summary"]["total_samples"] == report["config"]["max_samples"]
            ), name
            assert report["summary"]["failed"] == 0, (name, report["summary"])
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
