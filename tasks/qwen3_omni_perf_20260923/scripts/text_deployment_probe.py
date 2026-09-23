"""Requests without output modalities against the text-only Qwen3-Omni pipeline.

Starts Qwen3OmniPipelineConfig (no talker stage) in process and sends, through the
native client that leaves output modalities unset:
  1. no modalities, stream off
  2. no modalities, stream on
  3. output modalities ["text"], stream off (the pipeline still serves after 1 and 2)
Each request has a timeout, so a stage that died shows as a failed request, not a hang.
Prints one line per request and exits 1 if any failed.

usage: python text_deployment_probe.py [--model-path ...] [--timeout 180]
"""

from __future__ import annotations

import argparse
import asyncio
import time


async def run(model_path: str, timeout: float) -> int:
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.client.types import Message
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    config = Qwen3OmniPipelineConfig(model_path=model_path)
    runner = MultiProcessPipelineRunner(config)
    began = time.perf_counter()
    await runner.start(timeout=900)
    print(f"pipeline ready in {time.perf_counter() - began:.0f} s", flush=True)

    cases = [
        ("no-modalities-stream-off", None, False),
        ("no-modalities-stream-on", None, True),
        ("text-modality-stream-off", ["text"], False),
    ]
    failures = 0
    client = Client(runner.coordinator)
    try:
        for name, output_modalities, stream in cases:
            request = GenerateRequest(
                model=config.name,
                messages=[Message(role="user", content="Name three primary colors.")],
                sampling=SamplingParams(temperature=0.0, max_new_tokens=32),
                output_modalities=output_modalities,
                stream=stream,
            )

            async def collect() -> str:
                pieces = []
                async for chunk in client.generate(request, request_id=name):
                    pieces.append(chunk.text or "")
                return "".join(pieces)

            began = time.perf_counter()
            try:
                text = await asyncio.wait_for(collect(), timeout)
                print(
                    f"{name}: ok {time.perf_counter() - began:.1f} s {text!r}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
    finally:
        await runner.stop()
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="marksverdhei/Qwen3-Omni-30B-A3B-FP8")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.model_path, args.timeout)))


if __name__ == "__main__":
    main()
