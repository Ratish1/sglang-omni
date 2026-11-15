"""Whether one shared HF processor stays byte-identical under concurrent speech and video requests.

Runs the Qwen3-Omni preprocessing stage alone, CPU only, first serially, then with video threads at two client concurrencies. Requests
mix Video-MME CI videos with text and audio of varied length, plus one request per input form the stage reads (video key, video and
image parts, images and audios keys, a message list), and every output tensor and cache key is hashed by request. With --threads 1 it
runs the serial stage alone; --against compares this tree's serial hashes with another tree's report.

usage: PYTHONPATH=<tree> taskset -c <cpus> python3 preprocess_shared_identity.py --threads 8 --out identity.json [--against main.json]
"""

import argparse
import json
import random
import threading
import time

import numpy as np
from PIL import Image
from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI, tensor_hashes

import sglang_omni
import sglang_omni.models.qwen3_omni.stages as qwen_stages
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

SPEECH_PER_VIDEO = 4
SAMPLE_RATE = 16000
IMAGE_SIDE = 448


def tone(seconds: float, frequency: float) -> np.ndarray:
    phase = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * phase) * 0.1


def input_form_requests(video_path: str, image_path: str) -> dict[str, object]:
    def ask(text: str) -> list[dict[str, object]]:
        return [{"role": "user", "content": text}]

    return {
        "form-video-key": {
            "messages": ask("Describe the clip."),
            "video": [video_path],
            **VIDEO_SETTINGS,
        },
        "form-video-part": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_path}},
                        {"type": "text", "text": "Describe the clip."},
                    ],
                }
            ],
            **VIDEO_SETTINGS,
        },
        "form-image-part": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ]
        },
        "form-images-key": {
            "messages": ask("Describe the image."),
            "images": [image_path],
        },
        "form-audios-key": {
            "messages": ask("Say what you hear."),
            "audios": [tone(3.0, 440.0)],
        },
        "form-message-list": ask("Name three colors."),
    }


def build_requests(image_path: str) -> dict[str, object]:
    generator = random.Random(3452)
    pixels = np.random.default_rng(3452).integers(
        0, 256, (IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(image_path)
    requests: dict[str, object] = {}
    samples = load_videomme_samples(50, repo_id=VIDEOMME_CI)
    for index, sample in enumerate(samples):
        requests[f"video-{index}"] = {
            "messages": [{"role": "user", "content": sample.prompt or sample.question}],
            "videos": [sample.video_path],
            **VIDEO_SETTINGS,
        }
        for speech in range(SPEECH_PER_VIDEO):
            seconds = generator.uniform(0.5, 25.0)
            frequency = generator.uniform(120.0, 900.0)
            requests[f"speech-{index}-{speech}"] = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Say this back number {index} {speech} "
                        + "word " * generator.randint(1, 120),
                    }
                ],
                "audio": [tone(seconds, frequency)],
            }
    requests.update(input_form_requests(samples[0].video_path, image_path))
    return requests


def run_requests(
    threads: int, concurrency: int, requests: dict[str, object]
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
        inputs = requests[name]
        scheduler.inbox.put(
            IncomingMessage(
                request_id=name,
                type="new_request",
                data=StagePayload(
                    request_id=name,
                    request=OmniRequest(
                        inputs=(
                            list(inputs) if isinstance(inputs, list) else dict(inputs)
                        ),
                        params={},
                    ),
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--against")
    args = parser.parse_args()
    print(f"sglang_omni from {sglang_omni.__file__}", flush=True)
    requests = build_requests(args.image)
    reference, reference_seconds = run_requests(1, 1, requests)
    report = {
        "requests": len(requests),
        "serial_seconds": reference_seconds,
        "serial_hashes": reference,
    }
    print(f"serial: {reference_seconds:.1f} s, {len(reference)} requests", flush=True)
    if args.against is not None:
        with open(args.against) as source:
            other = json.load(source)["serial_hashes"]
        differing = [name for name in reference if other.get(name) != reference[name]]
        print(
            f"serial against {args.against}: {len(other)} requests there, "
            f"{len(differing)} differ {differing[:5]}",
            flush=True,
        )
        report["against"] = {"path": args.against, "differing": differing}
    else:
        pass
    for concurrency in (16, 64, 64) if args.threads > 1 else ():
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
