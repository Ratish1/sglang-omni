# SPDX-License-Identifier: Apache-2.0
"""Replay SeedTTS chat requests and retain text, audio and terminal metadata."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import itertools
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

import aiohttp

from benchmarks.benchmarker.data import RequestResult, finish_reason_from_server
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import save_generated_audio_metadata, save_speed_results


async def replay_requests(
    base_urls: list[str],
    output_directory: Path,
    samples: list[SampleInput],
    concurrency: int,
    rounds: int,
    request_namespace: uuid.UUID | None,
    fixed_request_id: uuid.UUID | None,
) -> None:
    worker_urls = itertools.cycle(base_urls)
    for round_index in range(rounds):
        round_directory = output_directory / f"round-{round_index + 1}"
        round_directory.mkdir()

        async def send_request(
            session: aiohttp.ClientSession, sample: SampleInput
        ) -> RequestResult:
            if fixed_request_id is not None:
                request_id = str(fixed_request_id)
            elif request_namespace is not None:
                request_id = str(
                    uuid.uuid5(request_namespace, f"{round_index}/{sample.sample_id}")
                )
            else:
                request_id = str(uuid.uuid4())
            base_url = next(worker_urls)
            prompt_text = (
                f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                f"Now please read the following text out loud in the same voice and style: "
                f"{sample.target_text}"
            )
            payload: dict[str, object] = {
                "model": "qwen3-omni",
                "messages": [{"role": "user", "content": prompt_text}],
                "modalities": ["text", "audio"],
                "audio": {"format": "wav"},
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False,
                "audios": [sample.ref_audio],
                "request_id": request_id,
            }
            request_path = round_directory / f"{request_id}.request.json"
            request_path.write_text(json.dumps(payload, indent=2))
            result = RequestResult(request_id=sample.sample_id, text=sample.target_text)
            started_at = time.perf_counter()
            async with session.post(
                f"{base_url}/v1/chat/completions", json=payload
            ) as response:
                response_body = await response.text()
                result.latency_s = time.perf_counter() - started_at
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {response_body}"
                    response_json: dict[str, object] = {"error": response_body}
                else:
                    response_json = json.loads(response_body)
                    choices = response_json["choices"]
                    assert isinstance(choices, list) and choices
                    message = choices[0]["message"]
                    audio = message["audio"]
                    audio_bytes = base64.b64decode(audio.pop("data"))
                    audio_path = round_directory / f"{request_id}.wav"
                    audio_path.write_bytes(audio_bytes)
                    result.wav_path = str(audio_path)
                    result.audio_duration_s = get_wav_duration(audio_bytes)
                    result.is_success = result.audio_duration_s > 0
                    result.rtf = (
                        result.latency_s / result.audio_duration_s
                        if result.is_success
                        else 0.0
                    )
                    result.finish_reason = finish_reason_from_server(
                        choices[0]["finish_reason"]
                    )
                    usage = response_json["usage"]
                    result.prompt_tokens = usage["prompt_tokens"]
                    result.completion_tokens = usage["completion_tokens"]
                    audio["sha256"] = hashlib.sha256(audio_bytes).hexdigest()
                    audio["wav_path"] = result.wav_path
                record = {
                    "sample_id": sample.sample_id,
                    "request_id": request_id,
                    "base_url": base_url,
                    "target_text": sample.target_text,
                    "http_status": response.status,
                    "latency_s": result.latency_s,
                    "audio_duration_s": result.audio_duration_s,
                    "request_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest(),
                    "response": response_json,
                }
            response_path = round_directory / f"{request_id}.response.json"
            response_path.write_text(json.dumps(record, indent=2))
            print(json.dumps(record), flush=True)
            return result

        runner = BenchmarkRunner(
            RunConfig(max_concurrency=concurrency, warmup=0, disable_tqdm=True)
        )
        results = await runner.run(samples, send_request)
        save_generated_audio_metadata(results, samples, str(round_directory))
        save_speed_results(
            results,
            compute_speed_metrics(results, runner.wall_clock_s),
            {
                "concurrency": concurrency,
                "request_namespace": (
                    str(request_namespace) if request_namespace is not None else None
                ),
                "warmup": 0,
            },
            str(round_directory),
        )
        summary = {
            "round": round_index + 1,
            "requests": len(results),
            "successes": sum(result.is_success for result in results),
            "wall_clock_s": runner.wall_clock_s,
            "max_audio_duration_s": max(result.audio_duration_s for result in results),
        }
        (round_directory / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--dataset", choices=("seedtts", "seedtts-50"), default="seedtts-50"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--request-namespace", type=uuid.UUID)
    parser.add_argument("--fixed-request-id", type=uuid.UUID)
    parser.add_argument("--reference-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.fixed_request_id is not None and (
        arguments.sample_id is None or arguments.rounds != 1
    ):
        parser.error("--fixed-request-id requires one selected sample and one round")
    else:
        pass
    logging.basicConfig(level=logging.INFO)
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    samples = load_seedtts_samples(
        DATASETS[arguments.dataset], arguments.max_samples, split="en"
    )
    if arguments.sample_id is not None:
        samples = [
            sample for sample in samples if sample.sample_id == arguments.sample_id
        ]
        assert samples, arguments.sample_id
    else:
        pass
    reference_directory = (
        arguments.reference_directory or arguments.output_directory / "references"
    )
    reference_directory.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        reference_path = reference_directory / f"{sample.sample_id}.wav"
        shutil.copyfile(sample.ref_audio, reference_path)
        sample.ref_audio = str(reference_path.resolve())
    asyncio.run(
        replay_requests(
            arguments.base_url,
            arguments.output_directory,
            samples,
            arguments.concurrency,
            arguments.rounds,
            arguments.request_namespace,
            arguments.fixed_request_id,
        )
    )


if __name__ == "__main__":
    main()
