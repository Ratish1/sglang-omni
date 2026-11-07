# SPDX-License-Identifier: Apache-2.0
"""Attribute paired speech errors using immutable audio and independent ASR scoring."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer, normalize_text, run_asr_transcription


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--asr-concurrency", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for arm_directory in arguments.arm:
        original_scores = {
            score["id"]: score
            for score in json.loads((arm_directory / "wer_results.json").read_text())[
                "per_sample"
            ]
        }
        records = sorted(
            (
                json.loads(response_file.read_text())
                for response_file in arm_directory.glob("*.response.json")
            ),
            key=lambda record: record["sample_id"],
        )
        assert len(records) == len(original_scores)
        samples = []
        evidence = []
        for record in records:
            sample_id = record["sample_id"]
            assert record["http_status"] == 200
            assert original_scores[sample_id]["is_success"]
            message = record["response"]["choices"][0]["message"]
            waveform_path = Path(message["audio"]["wav_path"])
            waveform_hash = hashlib.sha256(waveform_path.read_bytes()).hexdigest()
            assert waveform_hash == message["audio"]["sha256"]
            request = json.loads(
                (arm_directory / f"{record['request_id']}.request.json").read_text()
            )
            assert (
                hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                == record["request_sha256"]
            )
            with wave.open(str(waveform_path), "rb") as waveform:
                assert waveform.getnchannels() == 1
                assert waveform.getsampwidth() == 2
                assert waveform.getframerate() == 24000
                waveform_samples = waveform.getnframes()
                pcm = np.frombuffer(waveform.readframes(waveform_samples), dtype="<i2")
            assert len(pcm) == waveform_samples
            assert waveform_samples > 0
            samples.append(
                SampleInput(
                    sample_id=sample_id,
                    ref_text=record["target_text"],
                    ref_audio=str(waveform_path),
                    target_text=record["target_text"],
                )
            )
            evidence.append(
                {
                    "sample_id": sample_id,
                    "request_sha256": record["request_sha256"],
                    "waveform_sha256": waveform_hash,
                    "waveform_samples": waveform_samples,
                    "absolute_peak_pcm": int(np.abs(pcm.astype(np.int32)).max()),
                    "saturated_samples": int(
                        np.count_nonzero((pcm == -32768) | (pcm == 32767))
                    ),
                    "thinker_text": message["content"],
                    "thinker_normalized": normalize_text(message["content"], "en"),
                    "target_normalized": original_scores[sample_id]["ref_norm"],
                    "previous_transcript": original_scores[sample_id]["whisper_text"],
                    "previous_errors": sum(
                        original_scores[sample_id][name]
                        for name in ("substitutions", "deletions", "insertions")
                    ),
                }
            )
        results, wall_clock_s = asyncio.run(
            run_asr_transcription(
                samples,
                port=arguments.asr_port,
                model_path=arguments.asr_model,
                concurrency=arguments.asr_concurrency,
                warmup=0,
            )
        )
        result_by_id = {result.request_id: result for result in results}
        scores = []
        for sample, sample_evidence in zip(samples, evidence, strict=True):
            result = result_by_id[sample.sample_id]
            assert result.is_success, result.error
            score = apply_wer(
                SampleOutput(
                    sample_id=sample.sample_id,
                    target_text=sample.target_text,
                ),
                result.text,
                "en",
            )
            assert score.is_success, score.error
            sample_evidence["score"] = asdict(score)
            scores.append(score)
        report = {
            "arm": str(arm_directory),
            "asr_model": arguments.asr_model,
            "asr_concurrency": arguments.asr_concurrency,
            "asr_wall_clock_s": wall_clock_s,
            "summary": calculate_wer_metrics(scores, "en"),
            "per_sample": evidence,
        }
        destination = arguments.output_directory / f"{arm_directory.parent.name}.json"
        destination.write_text(json.dumps(report, indent=2))
        reports.append(
            {name: value for name, value in report.items() if name != "per_sample"}
        )
        print(json.dumps(reports[-1]), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
