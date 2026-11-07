# SPDX-License-Identifier: Apache-2.0
"""Score immutable paired TTS and MMSU audio with a running ASR server."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer, load_router_asr, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--arm", action="append")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    asr = load_router_asr(arguments.asr_port, arguments.asr_model)
    summaries = []
    for arm in arguments.arm or ("baseline", "candidate"):
        for workload in ("tts", "mmsu"):
            directory = arguments.evidence_directory / f"qualification-{arm}-{workload}"
            if workload == "tts":
                directory = directory / "round-1"
            else:
                pass
            records = sorted(
                (
                    json.loads(path.read_text())
                    for path in directory.glob("*.response.json")
                ),
                key=lambda record: record["sample_id"],
            )
            assert records
            requested_scores: list[SampleOutput] = []
            spoken_scores: list[SampleOutput] = []
            evidence = []
            for index, record in enumerate(records):
                message = record["response"]["choices"][0]["message"]
                if workload == "tts":
                    assert record["http_status"] == 200
                    waveform_path = Path(message["audio"]["wav_path"])
                    expected_hash = message["audio"]["sha256"]
                    requested_text = record["target_text"]
                else:
                    assert record["result"]["is_success"]
                    waveform_path = Path(record["result"]["wav_path"])
                    expected_hash = record["waveform_sha256"]
                    requested_text = None
                waveform_hash = hashlib.sha256(waveform_path.read_bytes()).hexdigest()
                assert waveform_hash == expected_hash
                transcript = transcribe(asr, str(waveform_path), "en", "cuda:0")
                spoken_score = apply_wer(
                    SampleOutput(
                        sample_id=record["sample_id"], target_text=message["content"]
                    ),
                    transcript,
                    "en",
                )
                assert spoken_score.is_success, spoken_score.error
                spoken_scores.append(spoken_score)
                if requested_text is not None:
                    requested_score = apply_wer(
                        SampleOutput(
                            sample_id=record["sample_id"], target_text=requested_text
                        ),
                        transcript,
                        "en",
                    )
                    assert requested_score.is_success, requested_score.error
                    requested_scores.append(requested_score)
                    requested_record = asdict(requested_score)
                else:
                    requested_record = None
                evidence.append(
                    {
                        "sample_id": record["sample_id"],
                        "request_sha256": record["request_sha256"],
                        "waveform_sha256": waveform_hash,
                        "returned_text": message["content"],
                        "transcript": transcript,
                        "requested_score": requested_record,
                        "spoken_score": asdict(spoken_score),
                    }
                )
                if (index + 1) % 128 == 0:
                    print(
                        f"{arm} {workload}: {index + 1}/{len(records)} scored",
                        flush=True,
                    )
                else:
                    pass
            summary = {
                "arm": arm,
                "workload": workload,
                "asr_model": arguments.asr_model,
                "asr_concurrency": 1,
                "spoken_summary": calculate_wer_metrics(spoken_scores, "en"),
                "requested_summary": (
                    calculate_wer_metrics(requested_scores, "en")
                    if requested_scores
                    else None
                ),
            }
            report = {**summary, "per_sample": evidence}
            (arguments.output_directory / f"{arm}-{workload}.json").write_text(
                json.dumps(report, indent=2)
            )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summaries, indent=2)
    )


if __name__ == "__main__":
    main()
