"""Streams every SeedTTS EN prompt and drops most streams at set points, so each abort path releases its row."""

import argparse
import asyncio
import json
import random
import time
from collections import Counter

import aiohttp

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.tts import _build_tts_payload

DROP_POINTS = ("cancel_early", "first_read", "third_read", "cancel_late", "complete")
EARLY_CANCEL_S = (0.0, 0.05)
LATE_CANCEL_S = (0.1, 1.0)
REQUEST_TIMEOUT_S = 300


async def read_stream(session, url, payload, reads):
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            return f"status_{response.status}"
        else:
            pass
        count = 0
        async for _ in response.content.iter_any():
            count += 1
            if reads is not None and count >= reads:
                return "dropped"
            else:
                pass
        return "completed"


async def run_one(session, url, payload, drop_point, rng):
    if drop_point == "first_read":
        stream = read_stream(session, url, payload, 1)
    elif drop_point == "third_read":
        stream = read_stream(session, url, payload, 3)
    else:
        stream = read_stream(session, url, payload, None)
    if drop_point == "cancel_early":
        deadline = rng.uniform(*EARLY_CANCEL_S)
    elif drop_point == "cancel_late":
        deadline = rng.uniform(*LATE_CANCEL_S)
    else:
        deadline = REQUEST_TIMEOUT_S
    try:
        return await asyncio.wait_for(stream, timeout=deadline)
    except asyncio.TimeoutError:
        return "cancelled"
    except aiohttp.ClientError as error:
        return f"client_error_{type(error).__name__}"


async def run(arguments):
    samples = load_seedtts_samples(arguments.meta, None, split="en")
    rng = random.Random(arguments.seed)
    url = f"http://127.0.0.1:{arguments.port}/v1/audio/speech"
    limit = asyncio.Semaphore(arguments.concurrency)
    outcomes = Counter()

    async def limited(index, sample, session):
        drop_point = DROP_POINTS[index % len(DROP_POINTS)]
        payload = _build_tts_payload(sample, arguments.model, stream=True)
        async with limit:
            outcome = await run_one(session, url, payload, drop_point, rng)
        outcomes[f"{drop_point}/{outcome}"] += 1

    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(
            *(limited(index, sample, session) for index, sample in enumerate(samples))
        )
    return {
        "requests": len(samples),
        "seconds": round(time.monotonic() - started, 1),
        "outcomes": dict(sorted(outcomes.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summary = asyncio.run(run(arguments))
    with open(arguments.output, "w") as sink:
        json.dump(summary, sink, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
