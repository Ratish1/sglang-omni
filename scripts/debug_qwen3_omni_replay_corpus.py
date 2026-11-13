"""Replay preserved request bytes with the shared benchmark runner and scorers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import httpx
from pydantic import BaseModel

from benchmarks.benchmarker.data import RequestResult, finish_reason_from_server
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.metrics.mmsu import compute_mmsu_metrics
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer, load_router_asr, transcribe
from benchmarks.tasks.audio_understanding import build_mmsu_results
from benchmarks.tasks.tts import SPEAKER_SIMILARITY_BATCH_SIZE, run_seedtts_utmos


class SavedRequest(BaseModel):
    request_id: str
    sample_id: str
    request_sha256: str
    target_text: str = ""


class ResponseAudio(BaseModel):
    data: str


class ResponseMessage(BaseModel):
    content: str
    audio: ResponseAudio


class ResponseChoice(BaseModel):
    message: ResponseMessage
    finish_reason: str


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    id: str
    choices: list[ResponseChoice]
    usage: ResponseUsage


class CompletedSpeech(BaseModel):
    request_id: str
    sample_id: str
    target_text: str
    text: str
    waveform_sha256: str


@dataclass(kw_only=True)
class ReplayRequest:
    record: SavedRequest
    request_bytes: bytes


@dataclass(kw_only=True)
class QualityConfiguration:
    model: str
    output_dir: str
    device: str


async def replay(arguments: argparse.Namespace) -> None:
    metadata = json.loads(arguments.reference_results.read_text())
    order = metadata["per_request" if arguments.workload == "tts" else "per_sample"]
    sample_ids = [
        record["id" if arguments.workload == "tts" else "sample_id"] for record in order
    ]
    requests: dict[str, ReplayRequest] = {}
    for response_path in arguments.input_directory.glob("*.response.json"):
        record = SavedRequest.model_validate_json(response_path.read_bytes())
        if (
            arguments.request_id is not None
            and record.request_id not in arguments.request_id
        ):
            continue
        else:
            pass
        request_bytes = response_path.with_name(
            response_path.name.removesuffix(".response.json") + ".request.json"
        ).read_bytes()
        payload = json.loads(request_bytes)
        assert payload["request_id"] == record.request_id
        assert (
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            == record.request_sha256
        )
        assert record.sample_id not in requests
        requests[record.sample_id] = ReplayRequest(
            record=record, request_bytes=request_bytes
        )
    if arguments.request_id is not None:
        assert {request.record.request_id for request in requests.values()} == set(
            arguments.request_id
        )
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in requests]
    else:
        pass
    assert len(sample_ids) == len(set(sample_ids)) == len(requests)
    assert set(sample_ids) == set(requests)
    httpx.get(f"{arguments.base_url}/health", timeout=5).raise_for_status()
    arguments.output_directory.mkdir(exist_ok=False)

    async def send_request(
        session: aiohttp.ClientSession, request: ReplayRequest
    ) -> RequestResult:
        record = request.record
        result = RequestResult(request_id=record.sample_id)
        destination = arguments.output_directory / record.request_id
        destination.with_suffix(".request.json").write_bytes(request.request_bytes)
        started = time.perf_counter()
        try:
            async with session.post(
                f"{arguments.base_url}/v1/chat/completions",
                data=request.request_bytes,
                headers={"Content-Type": "application/json"},
            ) as response:
                response_bytes = await response.read()
                destination.with_suffix(".response.json").write_bytes(response_bytes)
                response.raise_for_status()
            chat = ChatResponse.model_validate_json(response_bytes)
            assert chat.id == f"chatcmpl-{record.request_id}"
            message = chat.choices[0].message
            waveform = base64.b64decode(message.audio.data, validate=True)
            result.audio_duration_s = get_wav_duration(waveform)
            result.text = message.content
            result.prompt_tokens = chat.usage.prompt_tokens
            result.completion_tokens = chat.usage.completion_tokens
            result.finish_reason = finish_reason_from_server(
                chat.choices[0].finish_reason
            )
            result.wav_path = str(destination.with_suffix(".wav"))
            Path(result.wav_path).write_bytes(waveform)
            result.is_success = result.audio_duration_s > 0
            if result.is_success:
                result.rtf = (time.perf_counter() - started) / result.audio_duration_s
            else:
                result.error = "Empty generated audio"
            completed = CompletedSpeech(
                request_id=record.request_id,
                sample_id=record.sample_id,
                target_text=record.target_text,
                text=result.text,
                waveform_sha256=hashlib.sha256(waveform).hexdigest(),
            )
            destination.with_suffix(".speech.json").write_text(
                completed.model_dump_json(indent=2)
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            result.error = str(error)
        finally:
            result.latency_s = time.perf_counter() - started
            result.engine_time_s = result.latency_s
        return result

    runner = BenchmarkRunner(RunConfig(max_concurrency=arguments.concurrency, warmup=0))
    results = await runner.run(
        [requests[sample_id] for sample_id in sample_ids], send_request
    )
    report = {
        "wall_clock_s": runner.wall_clock_s,
        "concurrency": arguments.concurrency,
        "warmup": "server startup only, matching the saved attribution runs",
        "summary": compute_speed_metrics(results, runner.wall_clock_s),
        "per_request": [asdict(result) for result in results],
    }
    if arguments.workload == "mmsu":
        samples = [
            MmsuSample(**sample)
            for sample in json.loads(
                (arguments.input_directory / "samples.json").read_text()
            )
        ]
        predictions = build_mmsu_results(results, samples, ["text", "audio"])
        assert len(predictions) == len(results)
        report["mmsu_accuracy"] = compute_mmsu_metrics(predictions)
        report["mmsu_predictions"] = [asdict(prediction) for prediction in predictions]
    else:
        pass
    (arguments.output_directory / "results.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report["summary"]), flush=True)


def score(arguments: argparse.Namespace) -> None:
    arguments.output_directory.mkdir(exist_ok=False)
    asr = load_router_asr(arguments.asr_port, arguments.asr_model)
    requested_outputs: list[SampleOutput] = []
    spoken_outputs: list[SampleOutput] = []
    for source in sorted(arguments.input_directory.glob("*.speech.json")):
        record = CompletedSpeech.model_validate_json(source.read_bytes())
        waveform = source.with_name(f"{record.request_id}.wav")
        assert (
            hashlib.sha256(waveform.read_bytes()).hexdigest() == record.waveform_sha256
        )
        transcript = transcribe(asr, str(waveform), "en", "cuda:0")
        spoken = apply_wer(
            SampleOutput(sample_id=record.sample_id, target_text=record.text),
            transcript,
            "en",
        )
        assert spoken.is_success, spoken.error
        spoken_outputs.append(spoken)
        if arguments.workload == "tts":
            requested = apply_wer(
                SampleOutput(
                    sample_id=record.sample_id, target_text=record.target_text
                ),
                transcript,
                "en",
            )
            assert requested.is_success, requested.error
            requested_outputs.append(requested)
            requested_score = asdict(requested)
        else:
            requested_score = None
        (arguments.output_directory / f"{record.request_id}.json").write_text(
            json.dumps(
                {
                    "speech": record.model_dump(),
                    "spoken_score": asdict(spoken),
                    "requested_score": requested_score,
                },
                indent=2,
            )
        )
    expected = json.loads((arguments.input_directory / "results.json").read_text())
    assert len(spoken_outputs) == len(expected["per_request"])
    summary = {
        "spoken_summary": calculate_wer_metrics(spoken_outputs, "en"),
        "requested_summary": (
            calculate_wer_metrics(requested_outputs, "en")
            if requested_outputs
            else None
        ),
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


def quality(arguments: argparse.Namespace) -> None:
    assert arguments.workload == "tts"
    arguments.output_directory.mkdir(exist_ok=False)
    generation = json.loads((arguments.input_directory / "results.json").read_text())
    speeches = {
        speech.sample_id: speech
        for source in arguments.input_directory.glob("*.speech.json")
        for speech in [CompletedSpeech.model_validate_json(source.read_bytes())]
    }
    records = [speeches[result["request_id"]] for result in generation["per_request"]]
    assert records
    assert len(records) == len(speeches) == generation["summary"]["completed_requests"]
    assert generation["summary"]["failed_requests"] == 0
    generated = []
    reference_paths = []
    for record, result in zip(records, generation["per_request"], strict=True):
        waveform = arguments.input_directory / f"{record.request_id}.wav"
        assert (
            hashlib.sha256(waveform.read_bytes()).hexdigest() == record.waveform_sha256
        )
        payload = json.loads(
            (
                arguments.input_directory / f"{record.request_id}.request.json"
            ).read_bytes()
        )
        assert len(payload["audios"]) == 1
        reference_paths.append(payload["audios"][0])
        generated.append({**result, "sample_id": record.sample_id})
    (arguments.output_directory / "generated.json").write_text(
        json.dumps(generated, indent=2)
    )
    configuration = QualityConfiguration(
        model=payload["model"],
        output_dir=str(arguments.output_directory),
        device="cuda:0",
    )
    utmos = run_seedtts_utmos(configuration)
    assert utmos["summary"]["evaluated"] == len(records)
    assert utmos["summary"]["skipped"] == 0
    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=configuration.device,
    )
    similarities = []
    for start in range(0, len(records), SPEAKER_SIMILARITY_BATCH_SIZE):
        batch = records[start : start + SPEAKER_SIMILARITY_BATCH_SIZE]
        scores = scorer.score_batch(
            reference_paths[start : start + SPEAKER_SIMILARITY_BATCH_SIZE],
            [
                str(arguments.input_directory / f"{record.request_id}.wav")
                for record in batch
            ],
        )
        assert len(scores) == len(batch)
        similarities.extend(
            {
                "id": record.sample_id,
                "speaker_similarity": similarity,
                "is_success": True,
            }
            for record, similarity in zip(batch, scores, strict=True)
        )
    report = {
        "summary": {
            "speaker_similarity_mean": sum(
                result["speaker_similarity"] for result in similarities
            )
            / len(similarities),
            "total_samples": len(records),
            "evaluated": len(similarities),
            "skipped": 0,
        },
        "per_sample": similarities,
    }
    (arguments.output_directory / "similarity_results.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report["summary"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "score", "quality"), required=True)
    parser.add_argument("--workload", choices=("tts", "mmsu"), required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--asr-port", type=int)
    parser.add_argument("--asr-model")
    parser.add_argument("--request-id", action="append")
    arguments = parser.parse_args()
    if arguments.mode == "replay":
        assert arguments.reference_results and arguments.base_url
        assert arguments.concurrency and arguments.concurrency > 0
        asyncio.run(replay(arguments))
    elif arguments.mode == "score":
        assert arguments.asr_port and arguments.asr_model
        score(arguments)
    else:
        quality(arguments)


if __name__ == "__main__":
    main()
