# SPDX-License-Identifier: Apache-2.0
"""Trace the same native samples alone and with neighbors from their dataset."""

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TypedDict

import httpx
import torch

from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.accuracy import extract_answer_letter
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import apply_wer, load_qwen3_asr, transcribe
from benchmarks.tasks.audio_understanding import _build_request_payload
from scripts.debug_qwen3_omni_norm_population import post_seedtts_sample
from sglang_omni.models.qwen3_omni.boundary_witness import WitnessControl

SEEDTTS_WITNESSES = (
    "common_voice_en_18229682-common_voice_en_18229679",
    "common_voice_en_21555942-common_voice_en_21555950",
    "common_voice_en_19560742-common_voice_en_19560741",
    "common_voice_en_17256676-common_voice_en_17256678",
)
MMSU_WITNESS = "speech_stress_perception_f12a907a-291d-467a-9bbb-bd6ef2f2b09a"
REQUEST_TIMEOUT_SECONDS = 500
TRACE_BOUNDARIES = (
    "thinker-prefill-input",
    "thinker-publish",
    "thinker-decode-input",
    "talker-prompt",
    "backbone-prefill",
    "first-decode-prefix-cache",
    "backbone-decode",
    "codec",
)
GEOMETRY_FIELDS = {
    "active_rows",
    "batch_request_ids",
    "is_mixed_forward",
    "request_token_lengths",
    "request_row",
}
NON_SEMANTIC_FIELDS = {"request_pool_index"}


class WitnessResult(TypedDict):
    sample_id: str
    request_count: int
    directory: str
    waveform_sha256: str
    target_text: str
    returned_text: str
    correct_choice: str | None
    predicted_choice: str | None
    completion_tokens: int


def run_rounding_witnesses(
    server_port: int,
    run_directory: Path,
    control_path: Path,
    mmsu_samples: list[MmsuSample],
    mmsu_prompt: str,
    seed: int,
    client_concurrency: int,
) -> None:
    seedtts_samples = load_seedtts_samples(SEEDTTS_DATASET_ID)
    endpoint = f"http://127.0.0.1:{server_port}/v1/chat/completions"
    results: list[WitnessResult] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        for samples, witness_ids in (
            (seedtts_samples, SEEDTTS_WITNESSES),
            (mmsu_samples, (MMSU_WITNESS,)),
        ):
            sample_index_by_id = {
                sample.sample_id: index for index, sample in enumerate(samples)
            }
            for witness_id in witness_ids:
                witness_index = sample_index_by_id[witness_id]
                witness_sample = samples[witness_index]
                for request_count in (1, client_concurrency):
                    request_id = f"witness-{request_count}-{witness_id}"
                    trace_directory = run_directory / request_id
                    trace_directory.mkdir()
                    control: WitnessControl = {
                        "request_id": request_id,
                        "output_directory": str(trace_directory),
                        "norm_scope": "none",
                        "held_input_directory": None,
                    }
                    pending_control = control_path.with_suffix(".pending")
                    pending_control.write_text(json.dumps(control))
                    pending_control.replace(control_path)
                    start_index = min(witness_index, len(samples) - request_count)
                    group = samples[start_index : start_index + request_count]
                    assert len(group) == request_count and witness_sample in group
                    start_barrier = Barrier(request_count)

                    def send(sample: SampleInput | MmsuSample) -> httpx.Response:
                        sample_request_id = (
                            request_id
                            if sample.sample_id == witness_id
                            else f"peer-{request_id}-{sample.sample_id}"
                        )
                        if isinstance(sample, SampleInput):
                            return post_seedtts_sample(
                                client,
                                endpoint,
                                sample,
                                sample_request_id,
                                start_barrier,
                                seed=seed,
                            ).raise_for_status()
                        else:
                            payload = _build_request_payload(
                                sample,
                                model_name="qwen3-omni",
                                prompt=mmsu_prompt,
                                modalities=["text", "audio"],
                                max_tokens=256,
                                temperature=0.0,
                                seed=seed,
                            )
                            payload["request_id"] = sample_request_id
                            start_barrier.wait(timeout=REQUEST_TIMEOUT_SECONDS)
                            return client.post(
                                endpoint, json=payload
                            ).raise_for_status()

                    with ThreadPoolExecutor(max_workers=request_count) as executor:
                        responses = list(executor.map(send, group))
                    response = responses[group.index(witness_sample)].json()
                    message = response["choices"][0]["message"]
                    waveform = base64.b64decode(message["audio"]["data"])
                    (trace_directory / "speech.wav").write_bytes(waveform)
                    returned_text = message.get("content") or ""
                    if isinstance(witness_sample, SampleInput):
                        target_text = witness_sample.target_text
                        correct_choice = predicted_choice = None
                    else:
                        target_text = returned_text
                        assert witness_sample.answer_index is not None
                        correct_choice = chr(ord("A") + witness_sample.answer_index)
                        predicted_index = extract_answer_letter(returned_text)
                        predicted_choice = (
                            None
                            if predicted_index is None
                            else chr(ord("A") + predicted_index)
                        )
                    results.append(
                        {
                            "sample_id": witness_id,
                            "request_count": request_count,
                            "directory": trace_directory.name,
                            "waveform_sha256": hashlib.sha256(waveform).hexdigest(),
                            "target_text": target_text,
                            "returned_text": returned_text,
                            "correct_choice": correct_choice,
                            "predicted_choice": predicted_choice,
                            "completion_tokens": response["usage"]["completion_tokens"],
                        }
                    )
                    print(
                        f"{run_directory.name}: witnessed {witness_id} c{request_count}",
                        flush=True,
                    )
    (run_directory / "witnesses.json").write_text(json.dumps(results, indent=2) + "\n")


def score_rounding_witnesses(
    asr_port: int, output: Path, run_directories: list[Path]
) -> None:
    asr_service = load_qwen3_asr(asr_port)
    transcripts_by_digest: dict[str, str] = {}
    baseline_results: list[WitnessResult] = []
    for run_directory in run_directories:
        records: list[WitnessResult] = json.loads(
            (run_directory / "witnesses.json").read_bytes()
        )
        scored_records = []
        for record in records:
            trace_directory = run_directory / record["directory"]
            waveform_path = trace_directory / "speech.wav"
            digest = record["waveform_sha256"]
            if digest not in transcripts_by_digest:
                transcripts_by_digest[digest] = transcribe(
                    asr_service, str(waveform_path), "en", "cuda:0"
                )
            else:
                pass
            transcript = transcripts_by_digest[digest]
            scored = apply_wer(
                SampleOutput(
                    sample_id=record["sample_id"], target_text=record["target_text"]
                ),
                transcript,
                "en",
            )
            assert scored.is_success
            scored_records.append(
                {**record, "asr_transcript": transcript, "wer": scored.wer}
            )
        (run_directory / "scored-witnesses.json").write_text(
            json.dumps(scored_records, indent=2) + "\n"
        )
        if run_directory == run_directories[0]:
            baseline_results = records
        else:
            comparisons = []
            for baseline, candidate in zip(baseline_results, records, strict=True):
                assert baseline["directory"] == candidate["directory"]
                baseline_trace = run_directories[0] / baseline["directory"]
                candidate_trace = run_directory / candidate["directory"]
                comparisons.append(
                    {
                        "sample_id": candidate["sample_id"],
                        "request_count": candidate["request_count"],
                        "boundaries": compare_witnesses(
                            baseline_trace, candidate_trace
                        ),
                    }
                )
            (output / f"comparison-{run_directory.name}.json").write_text(
                json.dumps(comparisons, indent=2) + "\n"
            )
        for record in records:
            (run_directory / record["directory"] / "speech.wav").unlink()


def compare_witnesses(
    baseline_trace: Path, candidate_trace: Path
) -> dict[str, dict[str, int | bool | str | list[str] | None]]:
    report: dict[str, dict[str, int | bool | str | list[str] | None]] = {}
    for boundary in TRACE_BOUNDARIES:
        baseline_frames = {
            path.name: path for path in baseline_trace.glob(f"{boundary}-*.pt")
        }
        candidate_frames = {
            path.name: path for path in candidate_trace.glob(f"{boundary}-*.pt")
        }
        first_value_difference = first_geometry_difference = None
        first_value_difference_geometry_matched = None
        differing_fields: list[str] = []
        matched_frame_count = 0
        for filename in sorted(baseline_frames.keys() & candidate_frames.keys()):
            baseline = torch.load(baseline_frames[filename], weights_only=True)
            candidate = torch.load(candidate_frames[filename], weights_only=True)
            assert baseline.keys() == candidate.keys()
            geometry_fields = baseline.keys() & GEOMETRY_FIELDS
            if geometry_fields:
                geometry_matched = all(
                    baseline[field] == candidate[field] for field in geometry_fields
                )
                if geometry_matched:
                    matched_frame_count += 1
                elif first_geometry_difference is None:
                    first_geometry_difference = filename
                else:
                    pass
            else:
                geometry_matched = None
            if first_value_difference is None:
                for field in sorted(
                    baseline.keys() - GEOMETRY_FIELDS - NON_SEMANTIC_FIELDS
                ):
                    before = baseline[field]
                    after = candidate[field]
                    equal = (
                        isinstance(after, torch.Tensor) and torch.equal(before, after)
                        if isinstance(before, torch.Tensor)
                        else not isinstance(after, torch.Tensor) and before == after
                    )
                    if not equal:
                        differing_fields.append(field)
                    else:
                        pass
                if differing_fields:
                    first_value_difference = filename
                    first_value_difference_geometry_matched = geometry_matched
                else:
                    pass
            else:
                pass
        report[boundary] = {
            "baseline_frames": len(baseline_frames),
            "candidate_frames": len(candidate_frames),
            "matching_geometry_frames": matched_frame_count,
            "first_geometry_difference": first_geometry_difference,
            "first_value_difference": first_value_difference,
            "first_value_difference_geometry_matched": (
                first_value_difference_geometry_matched
            ),
            "first_differing_fields": differing_fields,
        }
    return report
