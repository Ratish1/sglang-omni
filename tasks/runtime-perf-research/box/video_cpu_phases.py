"""CPU seconds and wall time of each phase of a Qwen3-Omni video request, one request at a time.

usage: OMP_NUM_THREADS=8 PYTHONPATH=<tree> taskset -c <cpus> python3 video_cpu_phases.py --decode-threads 16 --out cell.json

On the 50 Video-MME CI videos at the CI's settings, through Qwen3OmniPreprocessor.call_impl. Requests run serially, so the
process CPU clock across a phase is that phase's CPU, spin included. The video hash runs beside the processor call, so its
thread's own CPU clock is read and taken off the processor. The first five videos are left out of the means.
"""

import argparse
import asyncio
import json
import os
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.models.qwen3_omni.components import preprocessor as preprocessor_module
from sglang_omni.preprocessing import video as video_module
from sglang_omni.proto.request import OmniRequest, StagePayload

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"
VIDEO_SETTINGS = {"video_fps": 2.0, "video_max_frames": 128, "video_max_pixels": 401408}
SKIP = 5

phases: dict[str, float] = defaultdict(float)
hash_thread_seconds = threading.local()


def timed(name: str, call: object) -> object:
    def wrapper(*args: object, **kwargs: object) -> object:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            return call(*args, **kwargs)
        finally:
            phases[f"{name} wall"] += time.perf_counter() - wall_start
            phases[f"{name} cpu"] += time.process_time() - cpu_start

    return wrapper


def timed_hash(call: object) -> object:
    def wrapper(*args: object, **kwargs: object) -> object:
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            return call(*args, **kwargs)
        finally:
            phases["hash wall"] += time.perf_counter() - wall_start
            phases["hash cpu"] += time.thread_time() - cpu_start

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-threads", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    video_module.read_video_torchcodec = timed(
        "decode", video_module.read_video_torchcodec
    )
    video_module.load_video_path = timed("load", video_module.load_video_path)
    preprocessor_module.compute_video_cache_key = timed_hash(
        preprocessor_module.compute_video_cache_key
    )
    preprocessor = preprocessor_module.Qwen3OmniPreprocessor(
        MODEL, max_seq_len=32768, video_decode_threads=args.decode_threads
    )
    preprocessor.processor = type(
        "TimedProcessor",
        (),
        {
            "__call__": staticmethod(timed("processor", preprocessor.processor)),
            "apply_chat_template": staticmethod(
                preprocessor.processor.apply_chat_template
            ),
        },
    )()

    samples = list(load_videomme_samples(50, repo_id=VIDEOMME_CI))
    per_video: dict[str, list[float]] = defaultdict(list)
    for index, sample in enumerate(samples):
        phases.clear()
        inputs = {
            "messages": [{"role": "user", "content": sample.prompt or sample.question}],
            "videos": [sample.video_path],
            **VIDEO_SETTINGS,
        }
        payload = StagePayload(
            request_id=f"video:{index}",
            request=OmniRequest(inputs=inputs, params={}),
            data={},
        )
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        asyncio.run(preprocessor.call_impl(payload))
        phases["request wall"] = time.perf_counter() - wall_start
        phases["request cpu"] = time.process_time() - cpu_start
        phases["resize wall"] = phases["load wall"] - phases["decode wall"]
        phases["resize cpu"] = phases["load cpu"] - phases["decode cpu"]
        phases["processor cpu"] -= phases["hash cpu"]
        if index >= SKIP:
            for name, value in phases.items():
                per_video[name].append(value)
        else:
            pass

    record = {
        "cpus": len(os.sched_getaffinity(0)),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "wait_policy": os.environ.get("OMP_WAIT_POLICY", ""),
        "spincount": os.environ.get("GOMP_SPINCOUNT", ""),
        "decode_threads": args.decode_threads,
        **{
            name: round(statistics.fmean(values) * 1e3, 1)
            for name, values in sorted(per_video.items())
        },
    }
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")
    print(json.dumps(record), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
