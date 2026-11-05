"""Host memory and build time of one more Qwen3-Omni preprocessor, which the stage keeps per video thread.

usage: PYTHONPATH=<tree> python3 preprocessor_cost.py --count 9
"""

import argparse
import resource
import statistics
import time

from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def current_rss_mb() -> float:
    with open("/proc/self/statm") as source:
        return int(source.read().split()[1]) * resource.getpagesize() / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=9)
    args = parser.parse_args()
    kept = []
    build_seconds = []
    rss_mb = [current_rss_mb()]
    for _ in range(args.count):
        started = time.perf_counter()
        kept.append(Qwen3OmniPreprocessor(MODEL, video_decode_threads=8))
        build_seconds.append(time.perf_counter() - started)
        rss_mb.append(current_rss_mb())
    print(
        f"build seconds first {build_seconds[0]:.2f} rest mean {statistics.fmean(build_seconds[1:]):.2f}"
    )
    print(
        f"rss MB before {rss_mb[0]:.0f} after first {rss_mb[1]:.0f} after all {rss_mb[-1]:.0f}"
    )
    print(
        f"MB per further preprocessor {(rss_mb[-1] - rss_mb[1]) / (args.count - 1):.1f}"
    )


if __name__ == "__main__":
    main()
