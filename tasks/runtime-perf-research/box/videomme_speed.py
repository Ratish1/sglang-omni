"""Video-MME benchmark of benchmarks.eval.benchmark_omni_videomme without its text-audio WER pass.

usage: python3 videomme_speed.py <the benchmark_omni_videomme arguments>

The WER pass transcribes every reply through a separate Qwen3-ASR server after the load ends, so it adds no serving
work to the measured window; the requests, the audio output and the saved videomme_results.json are the benchmark's.
"""

import argparse
import asyncio
import logging

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.eval.benchmark_omni_videomme import _config_from_args, add_video_eval_args, run_video_eval
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary


async def benchmark(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    results = await run_video_eval(config, task_label="Video-MME", output_filename="videomme_results.json",
                                   audio_output_dir_default="results/videomme_audio", compute_wer=False)
    print_videomme_accuracy_summary(results["summary"], config.model)
    print_speed_summary(results["speed"], config.model, config.max_concurrency, title="Video-MME Speed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    add_video_eval_args(parser, repo_help="Video-MME dataset repo")
    args = parser.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
