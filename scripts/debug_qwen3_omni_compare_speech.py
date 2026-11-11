"""Compare paired population speech errors without changing their official scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel


class SpeechIdentity(BaseModel):
    request_id: str
    sample_id: str
    target_text: str
    text: str
    waveform_sha256: str


class SpeechErrors(BaseModel):
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_norm: str
    hyp_norm: str
    whisper_text: str
    is_success: bool
    is_transcribed: bool

    @property
    def error_count(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_words(self) -> int:
        return self.substitutions + self.deletions + self.hits


class SpeechRecord(BaseModel):
    speech: SpeechIdentity
    spoken_score: SpeechErrors
    requested_score: SpeechErrors | None


@dataclass(kw_only=True)
class PairedSpeech:
    request_id: str
    sample_id: str
    same_returned_text: bool
    same_normalized_reference: bool
    baseline_errors: int
    candidate_errors: int
    error_delta: int
    baseline_reference_words: int
    candidate_reference_words: int
    baseline_wer: float
    candidate_wer: float
    baseline_transcript: str
    candidate_transcript: str
    baseline_reference: str
    candidate_reference: str


@dataclass(kw_only=True)
class PopulationSubset:
    samples: int
    baseline_wer: float
    candidate_wer: float


@dataclass(kw_only=True)
class SpeechComparison:
    subsets: dict[str, PopulationSubset]
    worse: int
    better: int
    equal: int
    baseline_above_50_percent: list[str]
    candidate_above_50_percent: list[str]
    per_request: list[PairedSpeech]


def compare_population(
    directory: Path, workload: str, recognizer: str
) -> dict[str, SpeechComparison]:
    populations: dict[str, dict[str, SpeechRecord]] = {}
    for arm in ("baseline", "candidate"):
        score_directory = directory / f"{arm}-{workload}-{recognizer}-scores"
        assert (score_directory / "summary.json").is_file()
        populations[arm] = {
            source.stem: SpeechRecord.model_validate_json(source.read_bytes())
            for source in score_directory.glob("*.json")
            if source.name != "summary.json"
        }
    baseline = populations["baseline"]
    candidate = populations["candidate"]
    assert baseline and baseline.keys() == candidate.keys()
    for request_id, record in baseline.items():
        paired = candidate[request_id]
        assert record.speech.request_id == paired.speech.request_id == request_id
        assert record.speech.sample_id == paired.speech.sample_id
        assert record.speech.target_text == paired.speech.target_text

    comparisons = {}
    for target in ("spoken", "requested"):
        if target == "requested" and workload != "tts":
            continue
        else:
            pass
        rows = []
        for request_id, record in baseline.items():
            paired = candidate[request_id]
            before = (
                record.spoken_score if target == "spoken" else record.requested_score
            )
            after = (
                paired.spoken_score if target == "spoken" else paired.requested_score
            )
            assert before is not None and after is not None
            assert before.is_success and after.is_success
            assert before.is_transcribed and after.is_transcribed
            rows.append(
                PairedSpeech(
                    request_id=request_id,
                    sample_id=record.speech.sample_id,
                    same_returned_text=record.speech.text == paired.speech.text,
                    same_normalized_reference=before.ref_norm == after.ref_norm,
                    baseline_errors=before.error_count,
                    candidate_errors=after.error_count,
                    error_delta=after.error_count - before.error_count,
                    baseline_reference_words=before.reference_words,
                    candidate_reference_words=after.reference_words,
                    baseline_wer=before.wer,
                    candidate_wer=after.wer,
                    baseline_transcript=before.whisper_text,
                    candidate_transcript=after.whisper_text,
                    baseline_reference=before.ref_norm,
                    candidate_reference=after.ref_norm,
                )
            )
        subsets = {}
        for selection, selected in (
            ("all", rows),
            ("same_returned_text", [row for row in rows if row.same_returned_text]),
            (
                "same_normalized_reference",
                [row for row in rows if row.same_normalized_reference],
            ),
        ):
            subsets[selection] = PopulationSubset(
                samples=len(selected),
                baseline_wer=sum(row.baseline_errors for row in selected)
                / sum(row.baseline_reference_words for row in selected),
                candidate_wer=sum(row.candidate_errors for row in selected)
                / sum(row.candidate_reference_words for row in selected),
            )
        comparisons[target] = SpeechComparison(
            subsets=subsets,
            worse=sum(row.candidate_wer > row.baseline_wer for row in rows),
            better=sum(row.candidate_wer < row.baseline_wer for row in rows),
            equal=sum(row.candidate_wer == row.baseline_wer for row in rows),
            baseline_above_50_percent=[
                row.request_id for row in rows if row.baseline_wer > 0.5
            ],
            candidate_above_50_percent=[
                row.request_id for row in rows if row.candidate_wer > 0.5
            ],
            per_request=sorted(rows, key=lambda row: row.error_delta, reverse=True),
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = {
        f"{workload}-{recognizer}": {
            target: asdict(comparison)
            for target, comparison in compare_population(
                arguments.directory, workload, recognizer
            ).items()
        }
        for workload in ("tts", "mmsu")
        for recognizer in ("qwen", "whisper-whole")
    }
    with arguments.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                population: {
                    target: {
                        name: value
                        for name, value in comparisons.items()
                        if name != "per_request"
                    }
                    for target, comparisons in targets.items()
                }
                for population, targets in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
