# SPDX-License-Identifier: Apache-2.0
"""Prepare a verified Qwen3-Omni SIM input packet from archived requests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from uuid import UUID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-selection", type=Path, required=True)
    parser.add_argument("--request-archive", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    assert arguments.runtime_directory.is_absolute()

    with gzip.open(arguments.catalog, "rt", encoding="utf-8") as catalog_file:
        archive_catalog = json.load(catalog_file)
    records = {
        record["request_id"]: record
        for record in archive_catalog["records"]
        if record["workload"] == "tts"
    }
    selected_cases = json.loads(arguments.trace_selection.read_text())["cases"]
    selected_by_sample_id = {
        case["sample_id"]: case["request_id"] for case in selected_cases
    }
    assert len(selected_by_sample_id) == len(selected_cases) > 1
    archived_order = json.loads(
        (arguments.request_archive / "speed_results.json").read_text()
    )["per_request"]
    ordered_cases = [
        {"id": row["id"]}
        for row in archived_order
        if row["id"] in selected_by_sample_id
    ]
    assert len(ordered_cases) == len(selected_cases)
    assert {case["id"] for case in ordered_cases} == set(selected_by_sample_id)
    cases = []
    for sample_id, request_id in selected_by_sample_id.items():
        assert str(UUID(request_id)) == request_id
        record = records[request_id]
        assert record["sample_id"] == sample_id
        request_bytes = (
            arguments.request_archive / f"{request_id}.request.json"
        ).read_bytes()
        assert hashlib.sha256(request_bytes).hexdigest() == record["raw_request_sha256"]
        request = json.loads(request_bytes)
        assert request["request_id"] == request_id and request.get("seed") is None
        assert (
            hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            == record["request_sha256"]
        ), request_id
        (original_reference,) = request["audios"]
        reference_filename = Path(original_reference).name
        assert reference_filename == f"{sample_id}.wav"
        reference_bytes = (
            arguments.reference_archive / reference_filename
        ).read_bytes()
        assert reference_bytes[:4] == b"RIFF" and reference_bytes[8:12] == b"WAVE"
        request["seed"] = arguments.seed
        request["audios"] = [
            str(arguments.runtime_directory / "audio" / reference_filename)
        ]
        seeded_request_bytes = json.dumps(request, sort_keys=True).encode()
        cases.append(
            (
                {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "target_text": record["target_text"],
                    "request_sha256": hashlib.sha256(seeded_request_bytes).hexdigest(),
                    "original_request_sha256": record["raw_request_sha256"],
                    "original_reference": original_reference,
                    "reference_filename": reference_filename,
                    "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                },
                seeded_request_bytes,
                reference_bytes,
            )
        )

    destination = arguments.output_directory
    destination.mkdir()
    (destination / "audio").mkdir()
    for case, request_bytes, reference_bytes in cases:
        request_file = destination / case["request_id"]
        request_file.with_suffix(".request.json").write_bytes(request_bytes)
        request_file.with_suffix(".response.json").write_text(json.dumps(case))
        (destination / "audio" / case["reference_filename"]).write_bytes(
            reference_bytes
        )
    (destination / "speed_results.json").write_text(
        json.dumps({"per_request": ordered_cases})
    )
    (destination / "manifest.json").write_text(
        json.dumps(
            {"seed": arguments.seed, "cases": [case for case, _, _ in cases]}, indent=2
        )
    )
    print(f"Prepared {len(cases)} verified original requests and references")


if __name__ == "__main__":
    main()
