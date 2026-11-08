"""Aggregate complete traces by request order, frame position and speech quality."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pydantic import BaseModel

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace


class CorpusStep(BaseModel):
    step: int
    phase: str
    captured_graph: bool
    request_ids: list[str]
    frame_indices: list[int]
    predictor_batch: int
    live_rows: int


class RequestIdentity(BaseModel):
    request_id: str
    sample_id: str


class SpeechScore(BaseModel):
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int


class ScoredSample(BaseModel):
    sample_id: str
    spoken_score: SpeechScore


class ScoreReport(BaseModel):
    per_sample: list[ScoredSample]


@dataclass(kw_only=True)
class MetricAggregate:
    count: int = 0
    finite_count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: list[float]) -> None:
        self.count += len(values)
        finite = [value for value in values if math.isfinite(value)]
        self.finite_count += len(finite)
        self.total += sum(finite)
        if finite:
            self.minimum = min(self.minimum, min(finite))
            self.maximum = max(self.maximum, max(finite))
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--measurement-directory", type=Path, required=True)
    parser.add_argument("--response-directory", type=Path, required=True)
    parser.add_argument("--score-file", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    identities = [
        RequestIdentity.model_validate_json(path.read_text())
        for path in arguments.response_directory.glob("*.response.json")
    ]
    assert identities
    samples = {identity.request_id: identity.sample_id for identity in identities}
    completions = {
        request_id: RequestTrace.model_validate_json(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_text()
        )
        for request_id in samples
    }
    scores = {}
    if arguments.score_file is not None:
        scores = {
            score.sample_id: score.spoken_score
            for score in ScoreReport.model_validate_json(
                arguments.score_file.read_text()
            ).per_sample
        }
        assert set(scores) == set(samples.values())
    else:
        pass
    actual_order = {
        request.request_id: index
        for index, request in enumerate(
            sorted(completions.values(), key=lambda completion: completion.ordinal)
        )
    }
    aggregates: dict[str, dict[str, MetricAggregate]] = defaultdict(dict)
    per_request: dict[str, Counter[str]] = defaultdict(Counter)
    observed_steps: dict[str, list[int]] = defaultdict(list)
    numeric_totals: dict[str, MetricAggregate] = {}
    predictor_positions: dict[str, MetricAggregate] = {}
    ignored_rows = 0
    with (arguments.capture_directory / "steps.jsonl").open() as manifest:
        for line in manifest:
            metadata = CorpusStep.model_validate_json(line)
            if metadata.predictor_batch == 0:
                continue
            else:
                pass
            measurements = torch.load(
                arguments.measurement_directory / f"step-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            row_values = {
                name: tensor.double()
                .movedim(-1, 0)
                .reshape(metadata.live_rows, -1)
                .tolist()
                for name, tensor in measurements.items()
            }
            selected_rows = [
                row
                for row, request_id in enumerate(metadata.request_ids)
                if request_id in samples
            ]
            for name, tensor in measurements.items():
                if tensor.ndim == 2:
                    for position in range(tensor.shape[0]):
                        predictor_positions.setdefault(
                            f"{name}.position.{position}", MetricAggregate()
                        ).update(
                            [row_values[name][row][position] for row in selected_rows]
                        )
                else:
                    pass
            for row, (request_id, frame_index) in enumerate(
                zip(metadata.request_ids, metadata.frame_indices, strict=True)
            ):
                if request_id not in samples:
                    ignored_rows += 1
                    continue
                else:
                    pass
                completion = completions[request_id]
                assert len(completion.codes) == len(completion.steps)
                assert completion.steps[frame_index] == metadata.step
                observed_steps[request_id].append(metadata.step)
                score = scores.get(samples[request_id])
                if score is None:
                    quality = "unscored"
                elif score.wer == 0:
                    quality = "zero_wer"
                elif score.wer <= 0.5:
                    quality = "positive_wer_at_most_50_percent"
                else:
                    quality = "wer_above_50_percent"
                buckets = (
                    f"order_tenth:{actual_order[request_id] * 10 // len(samples)}",
                    f"frame_quarter:{min(3, frame_index * 4 // len(completion.codes))}",
                    f"geometry:{metadata.live_rows}/{metadata.predictor_batch}/{metadata.phase}/{metadata.captured_graph}",
                    f"quality:{quality}",
                )
                for name, rows in row_values.items():
                    values = rows[row]
                    for bucket in buckets:
                        aggregate = aggregates[bucket].setdefault(
                            name, MetricAggregate()
                        )
                        aggregate.update(values)
                    if name.endswith("fp64_changes_argmax") or name.endswith(
                        "fp64_changes_set"
                    ):
                        per_request[request_id][name] += int(sum(values))
                    elif (
                        name.endswith("selected_is_argmax")
                        or name.endswith("selected_set_valid")
                        or name.endswith(".finite")
                        or name.endswith(".exact")
                        or name.endswith("_finite")
                        or name.endswith("_exact")
                    ):
                        per_request[request_id][f"invalid:{name}"] += len(values) - int(
                            sum(values)
                        )
                    else:
                        pass
                    numeric_totals.setdefault(name, MetricAggregate()).update(values)
                per_request[request_id]["frames"] += 1
    coverage = {
        request_id: completion.main_codes_match
        and observed_steps[request_id] == completion.steps
        for request_id, completion in completions.items()
    }
    (arguments.output_directory / "requests.json").write_text(
        json.dumps(
            [
                {
                    "request_id": request_id,
                    "sample_id": samples[request_id],
                    "ordinal": completions[request_id].ordinal,
                    "finish_reason": completions[request_id].finish_reason,
                    "coverage_exact": coverage[request_id],
                    "spoken_score": (
                        scores[samples[request_id]].model_dump() if scores else None
                    ),
                    "measurements": counts,
                }
                for request_id, counts in per_request.items()
            ],
            indent=2,
        )
    )
    summary = {
        "requests": len(samples),
        "frames": sum(counts["frames"] for counts in per_request.values()),
        "coverage_failures": [
            request_id for request_id, valid in coverage.items() if not valid
        ],
        "ignored_other_workload_rows": ignored_rows,
        "global_measurements": {
            name: asdict(aggregate) for name, aggregate in numeric_totals.items()
        },
        "predictor_positions": {
            name: asdict(aggregate) for name, aggregate in predictor_positions.items()
        },
        "buckets": {
            bucket: {
                name: asdict(aggregate) for name, aggregate in measurements.items()
            }
            for bucket, measurements in aggregates.items()
        },
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        json.dumps(
            {
                name: value
                for name, value in summary.items()
                if name not in ("global_measurements", "buckets", "predictor_positions")
            }
        )
    )


if __name__ == "__main__":
    main()
