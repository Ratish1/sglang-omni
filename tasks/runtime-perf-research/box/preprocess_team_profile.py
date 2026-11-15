"""The preprocessing stage alone over the 50 Video-MME CI videos, closed loop, by video threads, OpenMP team and FFmpeg
threads a video decodes on, at a client concurrency.

Each configuration runs in a fresh process: two warm requests, then the 50 videos with --concurrency in flight. Per
configuration: the pass's seconds, the mean wall of one request, and the mean of its media load and its HF processor
call, timed by wrapping the preprocessor module's loader and the processor in place.

usage: PYTHONPATH=<tree> taskset -c <cpus> python3 preprocess_team_profile.py --configs 8:14:14:16,8:112:16:16,... --out t.json
       a configuration is video_threads:omp:decode_threads:concurrency[:video_team]; video_team sizes each video
       thread's own OpenMP team as the thread starts, the process team (omp) staying with every other thread
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI


def run_one(
    threads: int, decode_threads: int, concurrency: int, video_team: int
) -> dict[str, float]:
    import torch

    import sglang_omni.models.qwen3_omni.components.preprocessor as preprocessor_mod
    import sglang_omni.models.qwen3_omni.stages as qwen_stages
    from benchmarks.dataset.videomme import load_videomme_samples
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

    if video_team:
        name_thread = qwen_stages.name_native_thread

        def start_video_thread() -> None:
            name_thread()
            torch.get_num_threads()
            torch.set_num_threads(video_team)

        qwen_stages.name_native_thread = start_video_thread
    else:
        pass

    spans: dict[str, list[float]] = {"load": [], "processor": []}
    load_video = preprocessor_mod.ensure_video_list_async

    async def timed_load(*args, **kwargs):
        started = time.perf_counter()
        result = await load_video(*args, **kwargs)
        spans["load"].append(time.perf_counter() - started)
        return result

    preprocessor_mod.ensure_video_list_async = timed_load
    samples = load_videomme_samples(50, repo_id=VIDEOMME_CI)
    scheduler = qwen_stages.create_preprocessing_executor(
        MODEL,
        max_concurrency=threads,
        max_seq_len=32768,
        video_decode_threads=decode_threads,
    )
    processor_class = type(scheduler.fn.processor)
    call_processor = processor_class.__call__

    def timed_processor(self, *args, **kwargs):
        started = time.perf_counter()
        result = call_processor(self, *args, **kwargs)
        spans["processor"].append(time.perf_counter() - started)
        return result

    processor_class.__call__ = timed_processor
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    submitted_at: dict[str, float] = {}

    def submit(name: str, video_path: str) -> None:
        inputs = {
            "messages": [{"role": "user", "content": "Describe the video."}],
            "videos": [video_path],
            **VIDEO_SETTINGS,
        }
        submitted_at[name] = time.perf_counter()
        scheduler.inbox.put(
            IncomingMessage(
                request_id=name,
                type="new_request",
                data=StagePayload(
                    request_id=name,
                    request=OmniRequest(inputs=inputs, params={}),
                    data={},
                ),
            )
        )

    def finish() -> float:
        result = scheduler.outbox.get(timeout=600)
        assert result.type == "result", repr(result.data)
        return time.perf_counter() - submitted_at[result.request_id]

    for index in range(2):
        submit(f"warm-{index}", samples[index].video_path)
        finish()
    for values in spans.values():
        values.clear()
    started = time.perf_counter()
    walls = []
    submitted = 0
    for _ in range(min(concurrency, len(samples))):
        submit(f"video-{submitted}", samples[submitted].video_path)
        submitted += 1
    for _ in samples:
        walls.append(finish())
        if submitted < len(samples):
            submit(f"video-{submitted}", samples[submitted].video_path)
            submitted += 1
        else:
            pass
    seconds = time.perf_counter() - started
    scheduler.stop()
    return {
        "seconds": seconds,
        "request_ms": statistics.mean(walls) * 1000,
        "load_ms": statistics.mean(spans["load"]) * 1000,
        "processor_ms": statistics.mean(spans["processor"]) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--one")
    args = parser.parse_args()
    if args.one is not None:
        threads, decode_threads, concurrency, video_team = (
            int(item) for item in args.one.split(":")
        )
        print(
            json.dumps(run_one(threads, decode_threads, concurrency, video_team)),
            flush=True,
        )
        return
    else:
        pass
    rows = []
    for config in args.configs.split(","):
        threads, omp_threads, decode_threads, concurrency, *video_team = config.split(
            ":"
        )
        video_team = video_team[0] if video_team else "0"
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--configs",
                config,
                "--out",
                "-",
                "--one",
                f"{threads}:{decode_threads}:{concurrency}:{video_team}",
            ],
            env=dict(os.environ, OMP_NUM_THREADS=omp_threads),
            capture_output=True,
            text=True,
            check=True,
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        row["config"] = config
        rows.append(row)
        print(
            f"threads {threads} omp {omp_threads} video team {video_team} decode {decode_threads} c{concurrency}: "
            f"{row['seconds']:.2f} s, request {row['request_ms']:.0f} ms, load {row['load_ms']:.0f} ms, "
            f"processor {row['processor_ms']:.0f} ms",
            flush=True,
        )
    with open(args.out, "w") as sink:
        json.dump(rows, sink, indent=1)


if __name__ == "__main__":
    main()
