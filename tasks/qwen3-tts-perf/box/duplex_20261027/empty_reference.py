"""An ICL request whose reference has no samples, then an ordinary request and the health route, against one running
Qwen3-TTS Base server: whether the empty reference fails alone or takes the server's device context with it.

usage: python3 empty_reference.py --base-url URL --model PATH --reference WAV --reference-text TEXT --out DIR
"""

import argparse
from pathlib import Path

import httpx
import numpy as np
import soundfile

REQUEST_TIMEOUT_S = 120.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    empty = args.out / "empty.wav"
    soundfile.write(str(empty), np.zeros(0, dtype=np.float32), 24000, subtype="PCM_16")
    with httpx.Client(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        for label, reference in (("empty", str(empty)), ("ordinary", args.reference), ("ordinary again", args.reference)):
            body = {
                "model": args.model, "voice": "default", "input": "Good morning, how can I help you today?",
                "response_format": "wav", "language": "English",
                "references": [{"audio_path": reference, "text": args.reference_text}],
            }
            try:
                response = client.post(f"{args.base_url}/v1/audio/speech", json=body)
                detail = response.text[:160] if response.status_code != 200 else f"{len(response.content)} bytes"
                print(f"{label}: HTTP {response.status_code}, {detail}", flush=True)
            except httpx.HTTPError as error:
                print(f"{label}: {type(error).__name__}: {error}", flush=True)
        try:
            health = client.get(f"{args.base_url}/health")
            print(f"health: HTTP {health.status_code}", flush=True)
        except httpx.HTTPError as error:
            print(f"health: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
