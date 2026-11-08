"""Independently transcribe corpus failures and their saved ordinary-run controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel

from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import apply_wer, load_router_asr, transcribe


@dataclass(kw_only=True)
class AudioCase:
    label: str
    waveform_path: Path
    waveform_sha256: str
    returned_text: str


class ControlRecord(BaseModel):
    request_id: str
    waveform_sha256: str
    text: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    controls = [
        ControlRecord.model_validate(record)
        for record in json.loads(
            (arguments.control_directory / "summary.json").read_text()
        )
    ]
    cases = {
        arguments.control_directory
        / f"{record.request_id}.wav": AudioCase(
            label=f"control:{record.request_id}",
            waveform_path=arguments.control_directory / f"{record.request_id}.wav",
            waveform_sha256=record.waveform_sha256,
            returned_text=record.text,
        )
        for record in controls
    }
    control_ids = {record.request_id for record in controls}
    for workload in ("tts", "mmsu"):
        selected = {
            record["sample_id"]
            for arm in arguments.arm
            for record in json.loads(
                (arguments.score_directory / f"{arm}-{workload}.json").read_text()
            )["per_sample"]
            if record["spoken_score"]["wer"] > 0.5
        }
        for arm in arguments.arm:
            directory = arguments.evidence_directory / f"qualification-{arm}-{workload}"
            if workload == "tts":
                directory = directory / "round-1"
            else:
                pass
            for path in directory.glob("*.response.json"):
                record = json.loads(path.read_text())
                if (
                    record["sample_id"] not in selected
                    and record["request_id"] not in control_ids
                ):
                    continue
                else:
                    pass
                message = record["response"]["choices"][0]["message"]
                if workload == "tts":
                    waveform_path = Path(message["audio"]["wav_path"])
                    expected_hash = message["audio"]["sha256"]
                else:
                    waveform_path = Path(record["result"]["wav_path"])
                    expected_hash = record["waveform_sha256"]
                cases[waveform_path] = AudioCase(
                    label=f"{arm}:{workload}:{record['sample_id']}",
                    waveform_path=waveform_path,
                    waveform_sha256=expected_hash,
                    returned_text=message["content"],
                )
    asr = load_router_asr(arguments.asr_port, arguments.asr_model)
    results = []
    for case in cases.values():
        assert (
            hashlib.sha256(case.waveform_path.read_bytes()).hexdigest()
            == case.waveform_sha256
        )
        transcript = transcribe(asr, str(case.waveform_path), "en", "cuda:0")
        score = apply_wer(
            SampleOutput(sample_id=case.label, target_text=case.returned_text),
            transcript,
            "en",
        )
        assert score.is_success, score.error
        result = {
            "label": case.label,
            "waveform_path": str(case.waveform_path),
            "waveform_sha256": case.waveform_sha256,
            "asr_model": arguments.asr_model,
            "returned_text": case.returned_text,
            "transcript": transcript,
            "score": asdict(score),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
