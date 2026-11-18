"""Score saved diagnostic waveforms after the vocoder releases its GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import apply_wer, load_router_asr, transcribe


class DecodedWaveform(BaseModel):
    trace_directory: Path
    reference_dtype: Literal["bfloat16", "float32"]
    allow_tf32_convolutions: bool
    non_eos_frames: int
    waveform_samples: int
    sample_rate: int
    waveform_file: Path
    waveform_sha256: str
    score: SampleOutput | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", type=Path, action="append", required=True)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    target_text_by_request = TypeAdapter(dict[str, str]).validate_json(
        arguments.targets_file.read_bytes()
    )
    reports: list[DecodedWaveform] = []
    for summary_file in arguments.summary_file:
        waveforms = TypeAdapter(list[DecodedWaveform]).validate_json(
            summary_file.read_bytes()
        )
        for waveform in waveforms:
            assert waveform.score is None
            assert (
                hashlib.sha256(waveform.waveform_file.read_bytes()).hexdigest()
                == waveform.waveform_sha256
            )
            hypothesis = transcribe(
                load_router_asr(arguments.asr_port, arguments.asr_model),
                str(waveform.waveform_file),
                "en",
                f"cuda:{arguments.gpu}",
            )
            waveform.score = apply_wer(
                SampleOutput(
                    sample_id=waveform.trace_directory.name,
                    target_text=target_text_by_request[waveform.trace_directory.name],
                    audio_duration_s=waveform.waveform_samples / waveform.sample_rate,
                ),
                hypothesis,
                "en",
            )
            assert waveform.score.is_success, waveform.score.error
            reports.append(waveform)
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(
                    [report.model_dump(mode="json") for report in reports],
                    indent=2,
                )
            )
            print(waveform.model_dump_json(), flush=True)
    (arguments.output_directory / "provenance.json").write_text(
        json.dumps(
            {
                "asr_model": arguments.asr_model,
                "summary_sha256": {
                    str(filename): hashlib.sha256(filename.read_bytes()).hexdigest()
                    for filename in arguments.summary_file
                },
                "targets_sha256": hashlib.sha256(
                    arguments.targets_file.read_bytes()
                ).hexdigest(),
                "scored_waveforms": len(reports),
                "scope": "Unchanged benchmark transcription and WER normalization, saved waveform bytes verified before scoring. Generation/vocoding and recognition are separate GPU phases. No performance result is inferred from WER timing fields.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
