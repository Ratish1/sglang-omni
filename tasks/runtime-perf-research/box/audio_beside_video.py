"""How long a request without video waits beside eight looping videos, the Qwen3-Omni preprocessing stage alone, CPU only.

The stage is built by its factory with --max-concurrency video requests and driven through its inbox and outbox. Short
requests go one at a time, alternating an audio request (a WAV data URL, decoded on the threads its connector uses) and
an image request (a PNG data URL): first with no video in flight, then while eight Video-MME CI video requests stay in
flight at the CI settings. Each short request's wall from inbox to answer is timed. Mid-load the process's threads are
counted by name, which also counts each OpenMP thread under the name of the thread that started it.

usage: PYTHONPATH=<tree> OMP_NUM_THREADS=<n> taskset -c <cpus> python3 audio_beside_video.py --max-concurrency 8 --out x.json
"""

import argparse
import base64
import collections
import io
import json
import os
import statistics
import threading
import time
import wave

import numpy as np
from PIL import Image
from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI

import sglang_omni
import sglang_omni.models.qwen3_omni.stages as qwen_stages
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

VIDEOS_IN_FLIGHT = 8
SHORT_REQUESTS = 40
SAMPLE_RATE = 16000
ANSWER_TIMEOUT_S = 600.0


def wav_url(seconds: float, frequency: float) -> str:
    phase = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples = (np.sin(2 * np.pi * frequency * phase) * 3000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, SAMPLE_RATE, len(samples), "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def png_url(seed: int) -> str:
    pixels = np.random.default_rng(seed).integers(0, 256, (448, 448, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def short_inputs(index: int) -> tuple[str, dict[str, object]]:
    if index % 2 == 0:
        return "audio", {
            "messages": [{"role": "user", "content": "Say what you hear."}],
            "audio": [wav_url(2.0 + index % 9, 220.0 + 20.0 * index)],
        }
    else:
        return "image", {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_url(index)}},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ]
        }


def thread_names() -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for task in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{task}/comm") as source:
                counts[source.read().strip()] += 1
        except OSError:
            continue
    return dict(counts.most_common())


class Stage:
    def __init__(self, max_concurrency: int, video_paths: list[str]) -> None:
        self.scheduler = qwen_stages.create_preprocessing_executor(
            MODEL, max_concurrency=max_concurrency, max_seq_len=32768
        )
        self.video_paths = video_paths
        self.sent: dict[str, float] = {}
        self.answered: dict[str, tuple[float, str]] = {}
        self.condition = threading.Condition()
        self.videos_on = False
        self.videos_sent = 0
        self.video_answers: list[float] = []
        threading.Thread(target=self.scheduler.start, daemon=True).start()
        threading.Thread(target=self.collect, daemon=True).start()

    def submit(self, request_id: str, inputs: dict[str, object]) -> None:
        with self.condition:
            self.sent[request_id] = time.perf_counter()
        self.scheduler.inbox.put(
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

    def submit_video(self) -> None:
        with self.condition:
            self.videos_sent += 1
            number = self.videos_sent
        path = self.video_paths[number % len(self.video_paths)]
        self.submit(
            f"video-{number}",
            {
                "messages": [{"role": "user", "content": "Describe the video."}],
                "videos": [path],
                **VIDEO_SETTINGS,
            },
        )

    def collect(self) -> None:
        while True:
            message = self.scheduler.outbox.get()
            now = time.perf_counter()
            with self.condition:
                self.answered[message.request_id] = (now, message.type)
                if message.request_id.startswith("video-"):
                    self.video_answers.append(now)
                    resubmit = self.videos_on
                else:
                    resubmit = False
                self.condition.notify_all()
            if resubmit:
                self.submit_video()
            else:
                pass

    def wait(self, request_id: str) -> float:
        with self.condition:
            assert self.condition.wait_for(
                lambda: request_id in self.answered, timeout=ANSWER_TIMEOUT_S
            ), request_id
            answered_at, message_type = self.answered[request_id]
            assert message_type == "result", request_id
            return answered_at - self.sent[request_id]

    def run_short(self, window: str) -> tuple[dict[str, list[float]], dict[str, int]]:
        walls: dict[str, list[float]] = {"audio": [], "image": []}
        census: dict[str, int] = {}
        for index in range(SHORT_REQUESTS):
            modality, inputs = short_inputs(index)
            request_id = f"{window}-{modality}-{index}"
            self.submit(request_id, inputs)
            walls[modality].append(self.wait(request_id))
            if index == SHORT_REQUESTS // 2:
                census = thread_names()
            else:
                pass
        return walls, census


def summary(walls: list[float]) -> dict[str, float]:
    ordered = sorted(walls)
    return {
        "mean_ms": statistics.mean(ordered) * 1000,
        "median_ms": statistics.median(ordered) * 1000,
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))] * 1000,
        "max_ms": ordered[-1] * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(f"sglang_omni from {sglang_omni.__file__}", flush=True)
    video_paths = [
        sample.video_path
        for sample in load_videomme_samples(50, repo_id=VIDEOMME_CI)
    ]
    stage = Stage(args.max_concurrency, video_paths)
    for index in range(2):
        stage.submit_video()
        stage.wait(f"video-{stage.videos_sent}")
    stage.run_short("warm")
    alone, census_alone = stage.run_short("alone")
    stage.videos_on = True
    for _ in range(VIDEOS_IN_FLIGHT):
        stage.submit_video()
    with stage.condition:
        stage.condition.wait_for(
            lambda: len(stage.video_answers) >= 2 + VIDEOS_IN_FLIGHT,
            timeout=ANSWER_TIMEOUT_S,
        )
    load_started = time.perf_counter()
    answers_before = len(stage.video_answers)
    beside, census_beside = stage.run_short("beside")
    load_seconds = time.perf_counter() - load_started
    videos_answered = len(stage.video_answers) - answers_before
    stage.videos_on = False
    report = {
        "max_concurrency": args.max_concurrency,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "alone": {name: summary(values) for name, values in alone.items()},
        "beside_videos": {name: summary(values) for name, values in beside.items()},
        "videos_per_second_beside": videos_answered / load_seconds,
        "threads_alone": census_alone,
        "threads_beside_videos": census_beside,
    }
    for window in ("alone", "beside_videos"):
        print(
            f"{window}: "
            + ", ".join(
                f"{name} mean {row['mean_ms']:.0f} median {row['median_ms']:.0f} "
                f"p90 {row['p90_ms']:.0f} max {row['max_ms']:.0f} ms"
                for name, row in report[window].items()
            ),
            flush=True,
        )
    print(
        f"videos answered {report['videos_per_second_beside']:.2f}/s beside; threads beside videos: "
        + ", ".join(
            f"{name} {count}"
            for name, count in list(census_beside.items())[:12]
        ),
        flush=True,
    )
    with open(args.out, "w") as sink:
        json.dump(report, sink, indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
