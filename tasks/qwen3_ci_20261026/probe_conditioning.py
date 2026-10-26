"""Compare real and zero multimodal Talker conditioning in one seeded server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import httpx
import soundfile

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import load_videomme_samples
from tests.test_model.test_qwen3_omni_videomme_talker_ci import SHORT_ANSWER_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    samples = load_videomme_samples(max_samples=20, repo_id=DATASETS["videomme-ci-50"])
    results: list[dict[str, str | int | float]] = []
    with httpx.Client(timeout=300) as client:
        for sample in samples:
            if sample.sample_id not in ("001-3", "006-3"):
                continue
            else:
                pass
            for seed in (42, 43):
                for trial, condition in enumerate(("real", "zero", "zero", "real")):
                    payload = {
                        "model": "qwen3-omni",
                        "request_id": f"diagnostic-{condition}-prompt-{sample.sample_id}-{seed}-{trial}",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}",
                            }
                        ],
                        "videos": [sample.video_path],
                        "modalities": ["text", "audio"],
                        "audio": {"format": "wav"},
                        "max_tokens": 256,
                        "temperature": 0,
                        "seed": seed,
                        "video_fps": 2,
                        "video_max_frames": 128,
                        "video_max_pixels": 401408,
                        "stream": False,
                    }
                    start_seconds = time.perf_counter()
                    response = client.post(arguments.url, json=payload)
                    response.raise_for_status()
                    elapsed_seconds = time.perf_counter() - start_seconds
                    message = response.json()["choices"][0]["message"]
                    audio_bytes = base64.b64decode(message["audio"]["data"])
                    waveform, sample_rate = soundfile.read(io.BytesIO(audio_bytes))
                    results.append(
                        {
                            "sample_id": sample.sample_id,
                            "seed": seed,
                            "condition": condition,
                            "latency_seconds": elapsed_seconds,
                            "audio_duration_seconds": len(waveform) / sample_rate,
                            "text": message["content"],
                            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                        }
                    )
                    print(json.dumps(results[-1]), flush=True)
    arguments.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
