"""The threads the Qwen3-Omni preprocessing stage runs, by name, while eight Video-MME CI videos stay in flight, CPU only.

Samples /proc/self/task every 10 ms over 40 video requests at the CI settings with --max-concurrency video requests.
A thread's OpenMP workers and the threads it starts carry its name, so a name's count is that thread's own pool.
Prints the peak count by name, the total thread count's max and median, and the counts left after the load.

usage: PYTHONPATH=<tree> OMP_NUM_THREADS=<n> taskset -c <cpus> python3 lane_thread_census.py --max-concurrency 8
"""

import argparse
import collections
import os
import threading
import time

from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI

import sglang_omni
import sglang_omni.models.qwen3_omni.stages as qwen_stages
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

VIDEOS_IN_FLIGHT = 8
VIDEO_REQUESTS = 40


def thread_names() -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for task in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{task}/comm") as source:
                counts[source.read().strip()] += 1
        except OSError:
            continue
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, required=True)
    args = parser.parse_args()
    print(f"sglang_omni from {sglang_omni.__file__}", flush=True)
    paths = [
        sample.video_path for sample in load_videomme_samples(50, repo_id=VIDEOMME_CI)
    ]
    scheduler = qwen_stages.create_preprocessing_executor(
        MODEL, max_concurrency=args.max_concurrency, max_seq_len=32768
    )
    threading.Thread(target=scheduler.start, daemon=True).start()
    peak: collections.Counter[str] = collections.Counter()
    totals: list[int] = []
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            counts = thread_names()
            totals.append(sum(counts.values()))
            for name, count in counts.items():
                peak[name] = max(peak[name], count)
            time.sleep(0.01)

    def submit(index: int) -> None:
        request_id = f"video-{index}"
        inputs = {
            "messages": [{"role": "user", "content": "Describe the video."}],
            "videos": [paths[index % len(paths)]],
            **VIDEO_SETTINGS,
        }
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=StagePayload(
                    request_id=request_id,
                    request=OmniRequest(inputs=inputs, params={}),
                    data={},
                ),
            )
        )

    sampler = threading.Thread(target=sample)
    sampler.start()
    for index in range(VIDEOS_IN_FLIGHT):
        submit(index)
    for index in range(VIDEOS_IN_FLIGHT, VIDEO_REQUESTS):
        message = scheduler.outbox.get(timeout=600)
        assert message.type == "result", repr(message.data)
        submit(index)
    for _ in range(VIDEOS_IN_FLIGHT):
        assert scheduler.outbox.get(timeout=600).type == "result"
    done.set()
    sampler.join()
    ordered = sorted(totals)
    print(f"peak by name {dict(peak.most_common(10))}", flush=True)
    print(f"total threads max {ordered[-1]} median {ordered[len(ordered) // 2]}", flush=True)
    print(f"after the load {dict(thread_names().most_common(10))}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
