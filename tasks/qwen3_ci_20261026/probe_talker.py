"""Measure speech duration repeatability with and without a request seed."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results: list[dict[str, str | int | float | None]] = []
    with httpx.Client(timeout=120) as client:
        for seed in (42, 42, 42, None, None, None):
            payload = {
                "model": arguments.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Repeat exactly this sentence and nothing else: The word Guitar is written in black marker on the whiteboard. Answer C.",
                    }
                ],
                "modalities": ["text", "audio"],
                "audio": {"format": "wav"},
                "max_tokens": 128,
                "temperature": 0,
                "stream": False,
            }
            if seed is not None:
                payload["seed"] = seed
            else:
                pass
            start_seconds = time.perf_counter()
            response = client.post(arguments.url, json=payload)
            response.raise_for_status()
            elapsed_seconds = time.perf_counter() - start_seconds
            message = response.json()["choices"][0]["message"]
            audio_bytes = base64.b64decode(message["audio"]["data"])
            waveform, sample_rate = soundfile.read(io.BytesIO(audio_bytes))
            results.append(
                {
                    "seed": seed,
                    "text": message["content"],
                    "latency_seconds": elapsed_seconds,
                    "audio_duration_seconds": len(waveform) / sample_rate,
                    "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                }
            )
            print(json.dumps(results[-1]), flush=True)
    arguments.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
