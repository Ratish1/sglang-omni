"""Measure the CI Video-MME workload and its request-stage timing on one server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import torch
import transformers

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from sglang_omni.profiler.views import build_report
from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
    load_short_answer_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile", choices=("none", "events", "torch"), default="events"
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    samples = load_short_answer_samples()
    metadata = {
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "seed": arguments.seed,
        "profile": arguments.profile,
    }
    (arguments.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    total_passes = arguments.repeats + int(arguments.profile != "none")
    with httpx.Client(timeout=120) as client:
        for repetition in range(total_passes):
            is_profile = repetition == arguments.repeats
            pass_name = "profile" if is_profile else f"pass-{repetition}"
            output_dir = arguments.output / pass_name
            output_dir.mkdir(parents=True, exist_ok=True)
            if is_profile:
                response = client.post(
                    f"{arguments.base_url}/start_profile",
                    json={
                        "run_id": arguments.output.name,
                        "event_dir": str(output_dir / "events"),
                        "enable_torch": arguments.profile == "torch",
                        "trace_path_template": str(output_dir / "traces" / "{stage}"),
                    },
                )
                response.raise_for_status()
                time.sleep(1)
            else:
                pass
            config = VideoEvalConfig(
                model="qwen3-omni",
                base_url=arguments.base_url,
                max_samples=20,
                max_tokens=256,
                max_concurrency=16,
                warmup=0 if is_profile else 16,
                seed=arguments.seed,
                output_dir=str(output_dir),
                repo_id=DATASETS["videomme-ci-50"],
                video_fps=2,
                video_max_frames=128,
                video_max_pixels=401408,
                enable_audio=True,
                timeout_s=500,
                disable_tqdm=True,
            )
            try:
                result = asyncio.run(
                    run_video_eval(
                        config,
                        samples=samples,
                        task_label="Video-MME",
                        output_filename="videomme_results.json",
                        audio_output_dir_default=str(output_dir / "audio"),
                        compute_wer=False,
                    )
                )
            finally:
                if is_profile:
                    response = client.post(
                        f"{arguments.base_url}/stop_profile",
                        json={"run_id": arguments.output.name},
                    )
                    response.raise_for_status()
                    time.sleep(2)
                else:
                    pass
            print(
                json.dumps(
                    {
                        "pass": pass_name,
                        "summary": result["summary"],
                        "speed": result["speed"],
                    }
                ),
                flush=True,
            )
            if result["summary"]["failed"]:
                raise RuntimeError(f"Incomplete benchmark: {result['summary']}")
            else:
                pass
            if is_profile:
                report = build_report(output_dir / "events")
                (output_dir / "stage-report.json").write_text(
                    json.dumps(report, indent=2)
                )
                print(
                    json.dumps(
                        {
                            "stage_breakdown": report["stage_breakdown"],
                            "hop_breakdown": report["hop_breakdown"],
                        }
                    ),
                    flush=True,
                )
            else:
                pass


if __name__ == "__main__":
    main()
