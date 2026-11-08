"""Separate rounded-logit ties from other decision changes over complete traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--measurement-directory", type=Path, required=True)
    parser.add_argument(
        "--response-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    identities: set[str] = set()
    for directory in arguments.response_directory:
        for path in directory.glob("*.response.json"):
            record = json.loads(path.read_text())
            assert isinstance(record, dict)
            request_id = record["request_id"]
            assert isinstance(request_id, str)
            identities.add(request_id)
    traces = {
        request_id: RequestTrace.model_validate_json(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_text()
        )
        for request_id in identities
    }
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    with (arguments.capture_directory / "steps.jsonl").open() as manifest:
        for line in manifest:
            metadata = StepMetadata.model_validate_json(line)
            if metadata.predictor_batch == 0:
                continue
            else:
                pass
            rows = [
                row
                for row, request_id in enumerate(metadata.request_ids)
                if request_id in identities
            ]
            if not rows:
                continue
            else:
                pass
            measurements = torch.load(
                arguments.measurement_directory / f"step-{metadata.step:06d}.pt",
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            buckets = {"all": rows}
            for quarter in range(4):
                buckets[f"{metadata.phase}:quarter{quarter}"] = [
                    row
                    for row in rows
                    if min(
                        3,
                        metadata.frame_indices[row]
                        * 4
                        // len(traces[metadata.request_ids[row]].codes),
                    )
                    == quarter
                ]
            for name, changes in measurements.items():
                if name.startswith("router.") and name.endswith("fp64_changes_set"):
                    prefix = name.removesuffix("fp64_changes_set")
                    ties = measurements[f"{prefix}cutoff_margin"] == 0
                elif name.startswith("head.") and name.endswith("fp64_changes_argmax"):
                    prefix = name.removesuffix("fp64_changes_argmax")
                    ties = measurements[f"{prefix}tie_count"] > 1
                else:
                    continue
                for bucket, bucket_rows in buckets.items():
                    selected_changes = changes[bucket_rows]
                    selected_ties = ties[bucket_rows]
                    counts = totals[f"{prefix}{bucket}"]
                    counts["decisions"] += len(bucket_rows)
                    counts["ties"] += int(selected_ties.sum())
                    counts["projection_changes"] += int(selected_changes.sum())
                    counts["changes_on_ties"] += int(
                        (selected_changes & selected_ties).sum()
                    )
                    counts["changes_without_ties"] += int(
                        (selected_changes & ~selected_ties).sum()
                    )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(totals, indent=2)
    )
    print(
        json.dumps(
            {name: counts for name, counts in totals.items() if name.endswith(".all")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
