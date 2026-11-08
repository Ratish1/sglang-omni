"""Independently transcribe immutable diagnostic waveforms with Whisper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jiwer
import torch
from pydantic import BaseModel, TypeAdapter
from transformers import pipeline
from whisper.normalizers import EnglishTextNormalizer


class WaveformControl(BaseModel):
    label: str
    request_id: str
    waveform_file: Path
    waveform_sha256: str
    reference: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    controls = TypeAdapter(list[WaveformControl]).validate_json(
        arguments.manifest.read_text()
    )
    for control in controls:
        assert (
            hashlib.sha256(control.waveform_file.read_bytes()).hexdigest()
            == control.waveform_sha256
        ), control.waveform_file
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=arguments.model_path,
        dtype=torch.bfloat16,
        device=0,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    normalizer = EnglishTextNormalizer()
    results = []
    for control in controls:
        transcription = recognizer(
            str(control.waveform_file),
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "do_sample": False,
            },
        )["text"]
        reference = normalizer(control.reference)
        hypothesis = normalizer(transcription)
        scores = jiwer.process_words(reference, hypothesis)
        results.append(
            {
                **control.model_dump(mode="json"),
                "transcript": transcription,
                "reference_normalized": reference,
                "hypothesis_normalized": hypothesis,
                "substitutions": scores.substitutions,
                "deletions": scores.deletions,
                "insertions": scores.insertions,
                "wer": scores.wer,
            }
        )
        arguments.output_file.write_text(
            json.dumps(
                {
                    "model_path": arguments.model_path,
                    "dtype": "bfloat16",
                    "attention_implementation": "sdpa",
                    "language": "english",
                    "task": "transcribe",
                    "do_sample": False,
                    "manifest_sha256": hashlib.sha256(
                        arguments.manifest.read_bytes()
                    ).hexdigest(),
                    "results": results,
                },
                indent=2,
            )
        )
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
