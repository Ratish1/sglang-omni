# SPDX-License-Identifier: Apache-2.0
"""Run the CI MMSU Talker workload with paired request identities."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import aiohttp

from benchmarks.benchmarker.data import RequestResult, finish_reason_from_server
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.mmsu import MmsuSample, load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.metrics.mmsu import compute_mmsu_metrics
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.audio_understanding import (
    _build_request_payload,
    build_mmsu_results,
)
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import (
    CONCURRENCY,
    MAX_SAMPLES,
    MAX_TOKENS,
    MMSU_TTS_PROMPT,
)


async def run(base_url: str, output_directory: Path, sample_file: Path) -> None:
    if sample_file.exists():
        samples = [
            MmsuSample(**sample) for sample in json.loads(sample_file.read_text())
        ]
    else:
        samples = load_mmsu_samples(
            max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
        )
        sample_file.write_text(
            json.dumps([asdict(sample) for sample in samples], indent=2)
        )
    output_directory.mkdir(exist_ok=False)
    (output_directory / "samples.json").write_text(
        json.dumps([asdict(sample) for sample in samples], indent=2)
    )

    async def send_request(
        session: aiohttp.ClientSession, sample: MmsuSample
    ) -> RequestResult:
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"qwen3-omni-feedback-mmsu/{sample.sample_id}"
            )
        )
        payload = _build_request_payload(
            sample,
            model_name="qwen3-omni",
            prompt=MMSU_TTS_PROMPT,
            modalities=["text", "audio"],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        payload["request_id"] = request_id
        request_bytes = json.dumps(payload, sort_keys=True).encode()
        (output_directory / f"{sample.sample_id}.request.json").write_bytes(
            request_bytes
        )
        started = time.monotonic()
        async with session.post(
            f"{base_url}/v1/chat/completions",
            data=request_bytes,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            response_payload = await response.json()
        elapsed_seconds = time.monotonic() - started
        choice = response_payload["choices"][0]
        message = choice["message"]
        audio_bytes = base64.b64decode(message["audio"].pop("data"), validate=True)
        waveform_path = output_directory / f"{sample.sample_id}.wav"
        waveform_path.write_bytes(audio_bytes)
        audio_seconds = get_wav_duration(audio_bytes)
        result = RequestResult(
            request_id=sample.sample_id,
            text=message["content"],
            is_success=audio_seconds > 0,
            latency_s=elapsed_seconds,
            audio_duration_s=audio_seconds,
            rtf=elapsed_seconds / audio_seconds,
            prompt_tokens=response_payload["usage"]["prompt_tokens"],
            completion_tokens=response_payload["usage"]["completion_tokens"],
            finish_reason=finish_reason_from_server(choice["finish_reason"]),
            wav_path=str(waveform_path),
        )
        record = {
            "sample_id": sample.sample_id,
            "request_id": request_id,
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "waveform_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "response": response_payload,
            "result": asdict(result),
        }
        (output_directory / f"{sample.sample_id}.response.json").write_text(
            json.dumps(record, indent=2)
        )
        print(
            json.dumps({"sample_id": sample.sample_id, "audio_seconds": audio_seconds}),
            flush=True,
        )
        return result

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=CONCURRENCY, warmup=0, disable_tqdm=True, timeout_s=500
        )
    )
    request_results = await runner.run(samples, send_request)
    assert len(request_results) == MAX_SAMPLES
    assert all(result.is_success for result in request_results)
    results = build_mmsu_results(request_results, samples, ["text", "audio"])
    summary = {
        "accuracy": compute_mmsu_metrics(results),
        "speed": compute_speed_metrics(
            request_results, wall_clock_s=runner.wall_clock_s
        ),
        "requests": [asdict(result) for result in request_results],
        "per_sample": [asdict(result) for result in results],
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "max_samples": MAX_SAMPLES,
    }
    (output_directory / "results.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                name: value
                for name, value in summary.items()
                if name not in ("requests", "per_sample")
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(
        run(arguments.base_url, arguments.output_directory, arguments.sample_file)
    )


if __name__ == "__main__":
    main()
