"""Qwen3-Omni preprocessing stage alone, closed loop, on the Video-MME CI videos: throughput, CPU and peak memory per setting.

usage: OMP_NUM_THREADS=8 PYTHONPATH=<tree> taskset -c <cpus> python3 preprocess_stage_bench.py --concurrency 16 --processors 2 \
           --videos-in-flight 6 --passes 2 --out cell.json

One fresh process per setting, started under taskset. The stage is built by its factory and driven through its scheduler's
inbox and outbox, with no server and no GPU. The first pass warms the page cache; the last pass's numbers are reported.
"""

import argparse
import json
import os
import resource
import statistics
import threading
import time
from pathlib import Path

import sglang_omni.models.qwen3_omni.stages as qwen_stages
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"
VIDEO_SETTINGS = {"video_fps": 2.0, "video_max_frames": 128, "video_max_pixels": 401408}


def payload_of(request_id: str, prompt: str, video_path: str) -> StagePayload:
    inputs = {
        "messages": [{"role": "user", "content": prompt}],
        "videos": [video_path],
        **VIDEO_SETTINGS,
    }
    return StagePayload(
        request_id=request_id, request=OmniRequest(inputs=inputs, params={}), data={}
    )


def run_pass(
    scheduler: object,
    samples: list[tuple[str, str]],
    concurrency: int,
    pass_name: str,
) -> dict[str, float]:
    started: dict[str, float] = {}
    latencies: list[float] = []
    next_index = 0

    def submit() -> None:
        nonlocal next_index
        request_id = f"{pass_name}:{next_index}"
        prompt, video_path = samples[next_index]
        next_index += 1
        started[request_id] = time.perf_counter()
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=payload_of(request_id, prompt, video_path),
            )
        )

    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    pass_start = time.perf_counter()
    for _ in range(min(concurrency, len(samples))):
        submit()
    for _ in range(len(samples)):
        message = scheduler.outbox.get(timeout=300)
        assert message.type == "result", repr(message.data)
        latencies.append(time.perf_counter() - started[message.request_id])
        if next_index < len(samples):
            submit()
        else:
            pass
    wall_seconds = time.perf_counter() - pass_start
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (
        usage_after.ru_utime
        + usage_after.ru_stime
        - usage_before.ru_utime
        - usage_before.ru_stime
    )
    ordered = sorted(latencies)
    return {
        "requests_per_second": len(samples) / wall_seconds,
        "latency_mean_ms": statistics.fmean(ordered) * 1e3,
        "latency_p95_ms": ordered[int(0.95 * len(ordered))] * 1e3,
        "cores_used": cpu_seconds / wall_seconds,
        "cpu_ms_per_request": cpu_seconds / len(samples) * 1e3,
        "peak_rss_gb": usage_after.ru_maxrss / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--processors", type=int, required=True)
    parser.add_argument("--videos-in-flight", type=int, default=0)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    samples = [
        (sample.prompt or sample.question, sample.video_path)
        for sample in load_videomme_samples(50, repo_id=VIDEOMME_CI)
    ] * 2
    scheduler = qwen_stages.create_preprocessing_executor(
        MODEL,
        max_concurrency=args.processors,
        max_seq_len=32768,
        max_videos_in_flight=args.videos_in_flight or None,
    )
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    results = [
        run_pass(scheduler, samples, args.concurrency, f"pass{index}")
        for index in range(args.passes)
    ]
    scheduler.stop()
    record = {
        "cpus": len(os.sched_getaffinity(0)),
        "concurrency": args.concurrency,
        "processors": args.processors,
        "videos_in_flight": args.videos_in_flight,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        **results[-1],
    }
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")
    print(json.dumps(record), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
