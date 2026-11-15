"""Resident memory of the Qwen3-Omni preprocessing stage alone over video requests, by video threads.

Each thread count runs in a fresh process: the stage preprocesses one video to warm up, its resident peak restarts,
then the 50 Video-MME CI videos run at the CI settings with --concurrency in flight. Per thread count: the pass's
resident peak, the size after the pass, the size after malloc_trim (what the allocator kept but no longer used), and
the bytes of pixel values one video carries (the resized frames hold as many float32 values), so the peak's growth
per video in flight reads against what a video has to hold.

usage: PYTHONPATH=<tree> taskset -c <cpus> python3 preprocess_memory.py --threads 1,2,4,8 --concurrency 16 --out mem.json \
           [--video-cpus N]   (passes the factory the CPUs videos share, as the config does when it splits the team)
"""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI

MIB = 1024 * 1024


def status_mib(field: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) / 1024
        else:
            pass
    raise KeyError(field)


def run_one(threads: int, concurrency: int, video_cpus: int) -> dict[str, object]:
    import sglang_omni.models.qwen3_omni.stages as qwen_stages
    from benchmarks.dataset.videomme import load_videomme_samples
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

    samples = load_videomme_samples(50, repo_id=VIDEOMME_CI)
    scheduler = qwen_stages.create_preprocessing_executor(
        MODEL,
        max_concurrency=threads,
        max_seq_len=32768,
        **({"video_cpus": video_cpus} if video_cpus and threads > 1 else {}),
    )
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()

    def message(name: str, video_path: str) -> IncomingMessage:
        inputs = {
            "messages": [{"role": "user", "content": "Describe the video."}],
            "videos": [video_path],
            **VIDEO_SETTINGS,
        }
        return IncomingMessage(
            request_id=name,
            type="new_request",
            data=StagePayload(
                request_id=name,
                request=OmniRequest(inputs=inputs, params={}),
                data={},
            ),
        )

    scheduler.inbox.put(message("warm", samples[0].video_path))
    assert scheduler.outbox.get(timeout=600).type == "result"
    Path("/proc/self/clear_refs").write_text("5")
    start_mib = status_mib("VmRSS")
    started = time.perf_counter()
    submitted = 0
    for _ in range(min(concurrency, len(samples))):
        scheduler.inbox.put(
            message(f"video-{submitted}", samples[submitted].video_path)
        )
        submitted += 1
    pixel_mib = []
    for _ in samples:
        result = scheduler.outbox.get(timeout=600)
        assert result.type == "result", repr(result.data)
        pixels = result.data.data["encoder_inputs"]["image_encoder"][
            "pixel_values_videos"
        ]
        pixel_mib.append(pixels.numel() * pixels.element_size() / MIB)
        del result
        if submitted < len(samples):
            scheduler.inbox.put(
                message(f"video-{submitted}", samples[submitted].video_path)
            )
            submitted += 1
        else:
            pass
    seconds = time.perf_counter() - started
    peak_mib = status_mib("VmHWM")
    end_mib = status_mib("VmRSS")
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    trimmed_mib = status_mib("VmRSS")
    os_threads = len(os.listdir("/proc/self/task"))
    scheduler.stop()
    return {
        "threads": threads,
        "concurrency": concurrency,
        "seconds": seconds,
        "start_mib": start_mib,
        "peak_mib": peak_mib,
        "end_mib": end_mib,
        "trimmed_mib": trimmed_mib,
        "os_threads": os_threads,
        "pixel_mib_mean": statistics.mean(pixel_mib),
        "pixel_mib_max": max(pixel_mib),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--one", type=int)
    parser.add_argument("--video-cpus", type=int, default=0)
    args = parser.parse_args()
    if args.one is not None:
        print(
            json.dumps(run_one(args.one, args.concurrency, args.video_cpus)),
            flush=True,
        )
        return
    else:
        pass
    cpus = len(os.sched_getaffinity(0))
    rows = []
    for threads in (int(item) for item in args.threads.split(",")):
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--one",
                str(threads),
                "--concurrency",
                str(args.concurrency),
                "--video-cpus",
                str(args.video_cpus),
                "--out",
                "-",
            ],
            env=dict(os.environ, OMP_NUM_THREADS=str(max(cpus // threads, 1))),
            capture_output=True,
            text=True,
            check=True,
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(
            f"threads {threads}: {row['seconds']:.1f} s, rss start {row['start_mib']:.0f} MiB, "
            f"peak {row['peak_mib']:.0f}, end {row['end_mib']:.0f}, after trim {row['trimmed_mib']:.0f}, "
            f"threads {row['os_threads']}, "
            f"pixels a video {row['pixel_mib_mean']:.0f} (max {row['pixel_mib_max']:.0f})",
            flush=True,
        )
    base = rows[0]
    for row in rows[1:]:
        extra = row["threads"] - base["threads"]
        print(
            f"threads {row['threads']} against {base['threads']}: peak "
            f"+{(row['peak_mib'] - base['peak_mib']) / extra:.0f} MiB a further video thread",
            flush=True,
        )
    with open(args.out, "w") as sink:
        json.dump(rows, sink, indent=1)


if __name__ == "__main__":
    main()
