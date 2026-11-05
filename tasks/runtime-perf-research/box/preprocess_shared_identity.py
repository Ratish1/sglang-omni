"""Whether one shared HF processor stays byte-identical under concurrent speech and video requests.

Runs the Qwen3-Omni preprocessing stage alone, CPU only, first serially, then with video threads at two client concurrencies. Requests
mix Video-MME CI videos with text and audio of varied length, and every output tensor and cache key is hashed by request.

usage: PYTHONPATH=<tree> OMP_WAIT_POLICY=PASSIVE taskset -c <cpus> python3 preprocess_shared_identity.py --threads 8 --out identity.json
"""

import argparse
import json
import random
import threading
import time

import numpy as np
from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI, tensor_hashes

import sglang_omni.models.qwen3_omni.stages as qwen_stages
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

SPEECH_PER_VIDEO = 4
SAMPLE_RATE = 16000


def build_requests() -> dict[str, dict[str, object]]:
    generator = random.Random(3452)
    requests: dict[str, dict[str, object]] = {}
    for index, sample in enumerate(load_videomme_samples(50, repo_id=VIDEOMME_CI)):
        requests[f"video-{index}"] = {
            "messages": [{"role": "user", "content": sample.prompt or sample.question}],
            "videos": [sample.video_path],
            **VIDEO_SETTINGS,
        }
        for speech in range(SPEECH_PER_VIDEO):
            seconds = generator.uniform(0.5, 25.0)
            samples = int(seconds * SAMPLE_RATE)
            phase = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
            frequency = generator.uniform(120.0, 900.0)
            requests[f"speech-{index}-{speech}"] = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Say this back number {index} {speech} "
                        + "word " * generator.randint(1, 120),
                    }
                ],
                "audio": [np.sin(2 * np.pi * frequency * phase) * 0.1],
            }
    return requests


def run_requests(
    threads: int, concurrency: int, requests: dict[str, dict[str, object]]
) -> tuple[dict[str, dict[str, str]], float]:
    scheduler = qwen_stages.create_preprocessing_executor(
        MODEL, max_concurrency=threads, max_seq_len=32768
    )
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    names = list(requests)
    next_index = 0

    def submit() -> None:
        nonlocal next_index
        name = names[next_index]
        next_index += 1
        scheduler.inbox.put(
            IncomingMessage(
                request_id=name,
                type="new_request",
                data=StagePayload(
                    request_id=name,
                    request=OmniRequest(inputs=dict(requests[name]), params={}),
                    data={},
                ),
            )
        )

    started = time.perf_counter()
    for _ in range(min(concurrency, len(names))):
        submit()
    hashes: dict[str, dict[str, str]] = {}
    for _ in names:
        message = scheduler.outbox.get(timeout=600)
        assert message.type == "result", repr(message.data)
        hashes[message.request_id] = tensor_hashes(message.data.data)
        if next_index < len(names):
            submit()
        else:
            pass
    elapsed = time.perf_counter() - started
    scheduler.stop()
    worker.join(timeout=5)
    return hashes, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests = build_requests()
    reference, reference_seconds = run_requests(1, 1, requests)
    report = {"requests": len(requests), "serial_seconds": reference_seconds}
    for concurrency in (16, 64, 64):
        hashes, seconds = run_requests(args.threads, concurrency, requests)
        differing = [name for name in reference if hashes.get(name) != reference[name]]
        tensors = sum(len(value) for value in reference.values())
        print(
            f"threads {args.threads} concurrency {concurrency}: {seconds:.1f} s, "
            f"{len(hashes)} requests, {tensors} hashed values, {len(differing)} requests differ {differing[:5]}",
            flush=True,
        )
        report[f"c{concurrency}"] = {"seconds": seconds, "differing": differing}
    with open(args.out, "w") as sink:
        json.dump(report, sink, indent=1)


if __name__ == "__main__":
    main()
