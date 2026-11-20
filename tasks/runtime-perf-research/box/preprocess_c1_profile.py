"""Where a lone video's preprocessing time moves between main's serial stage and the video lane, the stage alone.

One video at a time over the 50 Video-MME CI videos at the CI settings, after two warm requests, each configuration in
a fresh process: video threads (1 runs main's serial path, 8 the lane), the stage's OpenMP team, and the FFmpeg threads
a video decodes on. Per configuration: mean and median wall of one request, and of its media load and its HF processor
call, timed by wrapping the preprocessor module's loader and the processor in place.

usage: PYTHONPATH=<tree> taskset -c <cpus> python3 preprocess_c1_profile.py --configs 1:112:16,8:14:14,... --out c1.json
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


def run_one(threads: int, decode_threads: int) -> dict[str, object]:
    import sglang_omni.models.qwen3_omni.components.preprocessor as preprocessor_mod
    import sglang_omni.models.qwen3_omni.stages as qwen_stages
    from benchmarks.dataset.videomme import load_videomme_samples
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

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
    processor_class = preprocessor_mod.Qwen3OmniMoeProcessor
    call_processor = processor_class.__call__

    def timed_processor(self, *args, **kwargs):
        started = time.perf_counter()
        result = call_processor(self, *args, **kwargs)
        spans["processor"].append(time.perf_counter() - started)
        return result

    processor_class.__call__ = timed_processor
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()

    def run(name: str, video_path: str) -> float:
        inputs = {
            "messages": [{"role": "user", "content": "Describe the video."}],
            "videos": [video_path],
            **VIDEO_SETTINGS,
        }
        started = time.perf_counter()
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
        result = scheduler.outbox.get(timeout=600)
        assert result.type == "result", repr(result.data)
        return time.perf_counter() - started

    for index in range(2):
        run(f"warm-{index}", samples[index].video_path)
    for values in spans.values():
        values.clear()
    walls = [
        run(f"video-{index}", sample.video_path) for index, sample in enumerate(samples)
    ]
    scheduler.stop()
    return {
        name: {
            "mean_ms": statistics.mean(values) * 1000,
            "median_ms": statistics.median(values) * 1000,
        }
        for name, values in (("request", walls), *spans.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--one")
    args = parser.parse_args()
    if args.one is not None:
        threads, decode_threads = (int(item) for item in args.one.split(":"))
        print(json.dumps(run_one(threads, decode_threads)), flush=True)
        return
    else:
        pass
    rows = []
    for config in args.configs.split(","):
        threads, omp_threads, decode_threads = config.split(":")
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--configs",
                config,
                "--out",
                "-",
                "--one",
                f"{threads}:{decode_threads}",
            ],
            env=dict(os.environ, OMP_NUM_THREADS=omp_threads),
            capture_output=True,
            text=True,
            check=True,
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        row["config"] = {
            "video_threads": int(threads),
            "omp": int(omp_threads),
            "decode_threads": int(decode_threads),
        }
        rows.append(row)
        print(
            f"video threads {threads} omp {omp_threads} decode {decode_threads}: "
            + ", ".join(
                f"{name} {row[name]['mean_ms']:.1f} ms (median {row[name]['median_ms']:.1f})"
                for name in ("request", "load", "processor")
            ),
            flush=True,
        )
    with open(args.out, "w") as sink:
        json.dump(rows, sink, indent=1)


if __name__ == "__main__":
    main()
