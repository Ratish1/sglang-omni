"""Time Qwen3OmniPreprocessor.call_impl per dispatch form on the SeedTTS EN reference audios.

usage: OMP_NUM_THREADS=32 PYTHONPATH=<tree> python3 preprocess_dispatch.py --out dispatch.json

Forms: run_one (asyncio.run per call, the threaded scheduler's dispatch), persistent (one event loop per thread with
run_until_complete per call, the serial scheduler's dispatch), and each on two threads with a preprocessor each and the
inputs split between them. Prints each form's time per call; a CPU-only process with nothing else running.
"""

import argparse
import asyncio
import json
import statistics
import threading
import time
from pathlib import Path

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS = "zhaochenyang20/seed-tts-eval-arrow"


def payload_of(index: int, text: str, audio_path: str) -> StagePayload:
    inputs = {"messages": [{"role": "user", "content": text}], "audios": [audio_path]}
    return StagePayload(
        request_id=f"speech:{index}",
        request=OmniRequest(inputs=inputs, params={}),
        data={},
    )


def run_inputs(
    preprocessor: Qwen3OmniPreprocessor,
    inputs: list[tuple[int, str, str]],
    persistent: bool,
    times: list[float],
) -> None:
    loop = asyncio.new_event_loop() if persistent else None
    for index, text, audio_path in inputs:
        payload = payload_of(index, text, audio_path)
        start = time.perf_counter()
        if loop is None:
            asyncio.run(preprocessor.call_impl(payload))
        else:
            loop.run_until_complete(preprocessor.call_impl(payload))
        times.append((time.perf_counter() - start) * 1e3)
    if loop is not None:
        loop.close()
    else:
        pass


def summary(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "calls": len(ordered),
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * len(ordered))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    samples = load_seedtts_samples(SEEDTTS, None, split="en")
    inputs = [
        (index, sample.target_text, sample.ref_audio)
        for index, sample in enumerate(samples)
    ]
    preprocessors = [Qwen3OmniPreprocessor(MODEL), Qwen3OmniPreprocessor(MODEL)]
    run_inputs(preprocessors[0], inputs[:64], True, [])
    results: dict[str, dict[str, float]] = {}
    for persistent in (False, True):
        name = "persistent" if persistent else "run_one"
        times: list[float] = []
        run_inputs(preprocessors[0], inputs, persistent, times)
        results[f"{name} one thread"] = summary(times)
        halves = [inputs[0::2], inputs[1::2]]
        thread_times: list[list[float]] = [[], []]
        threads = [
            threading.Thread(
                target=run_inputs,
                args=(preprocessors[i], halves[i], persistent, thread_times[i]),
            )
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        results[f"{name} two threads"] = summary(thread_times[0] + thread_times[1])
    for name, values in results.items():
        print(
            f"{name:24s} calls {values['calls']} mean {values['mean_ms']:.2f} p50 {values['p50_ms']:.2f} "
            f"p95 {values['p95_ms']:.2f} ms"
        )
    Path(args.out).write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
