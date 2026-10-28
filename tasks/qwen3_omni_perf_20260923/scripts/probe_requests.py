"""Representative requests for one model's IPC payload probe run, with a fixed count and concurrency.

kinds: asr (POST /v1/audio/transcriptions of tests/data/query_to_cars.wav), image (POST
/v1/images/generations, the LLaDA2.0-Uni cookbook request with 8 dLLM steps), music (POST
/v1/audio/speech, the MiniMax-Music3 cookbook request with 250 new tokens). Prints one line per
request with its status and seconds, then the count of successes.

usage: python probe_requests.py --kind asr --port 8000 --model M --count 16 --concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time

import requests

WAV = "tests/data/query_to_cars.wav"


def one_request(kind: str, port: int, model: str, index: int) -> tuple[int, int, float]:
    url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    if kind == "asr":
        with open(WAV, "rb") as handle:
            response = requests.post(
                f"{url}/v1/audio/transcriptions",
                data={"model": model, "response_format": "json"},
                files={"file": handle},
                timeout=600,
            )
    elif kind == "image":
        response = requests.post(
            f"{url}/v1/images/generations",
            json={
                "model": model,
                "prompt": "A sailboat on a calm lake.",
                "size": "1024x1024",
                "response_format": "b64_json",
                "decode_mode": "decoder-turbo",
                "num_inference_steps": 8,
                "dllm_steps": 8,
                "guidance_scale": 4.0,
                "seed": 42 + index,
            },
            timeout=900,
        )
    elif kind == "music":
        response = requests.post(
            f"{url}/v1/audio/speech",
            json={
                "model": model,
                "input": "[Verse]\nSunday morning, quiet and slow\n[Chorus]\nTake it easy, take it slow",
                "instructions": "A mellow lo-fi track at 85 BPM",
                "seed": 42 + index,
                "max_new_tokens": 250,
            },
            timeout=900,
        )
    else:
        raise ValueError(f"unknown kind {kind}")
    return index, response.status_code, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("asr", "image", "music"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.kind, args.port, args.model, index)
            for index in range(args.count)
        ]
        results = [future.result() for future in futures]
    for index, status, seconds in results:
        print(f"request {index} status {status} seconds {seconds:.2f}")
    print(f"ok {sum(status == 200 for _, status, _ in results)} of {len(results)}")


if __name__ == "__main__":
    main()
