"""Restore original request bytes and ordering for targeted speech diagnostics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

INPUT_ARCHIVE_SHA256 = (
    "e12df44faca9171e6416bcab13d88d5bdc89e944bb7d340fcecec4fb21f9210f"
)
RESULT_ARCHIVE_SHA256 = (
    "4204c6fe6f207836acd21109d8b0b307490091b7940be6bad6fdfd78de2197dd"
)
POPULATION_PREFIX = (
    "dense-restored-native-raw-scale-qualification/raw-scale/"
    "original-population/native-weight-contributions/"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    input_archive = arguments.transfer_directory / (
        "reusable-original-stimuli-and-disputes-clean-20261110.tar.gz"
    )
    result_archive = arguments.transfer_directory / (
        "dense-restored-native-raw-scale-complete-20261116-0000.jsonl.gz"
    )
    with input_archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == INPUT_ARCHIVE_SHA256
    with result_archive.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest() == RESULT_ARCHIVE_SHA256
        )
    with tarfile.open(input_archive) as archive:
        archive.extractall(arguments.output_directory / "assets", filter="data")
    with gzip.open(
        arguments.transfer_directory
        / "restored-original-request-catalog-20261115.json.gz",
        "rt",
    ) as stream:
        catalog = json.load(stream)["records"]
    assert len(catalog) == 1128
    archived_files: dict[str, bytes] = {}
    with gzip.open(result_archive, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["path"].startswith(POPULATION_PREFIX) or record["path"] == (
                "dense-restored-native-raw-scale-qualification/"
                "raw-scale/full-mmsu/generation/samples.json"
            ):
                content = record["content"].encode()
                assert len(content) == record["size"]
                assert hashlib.sha256(content).hexdigest() == record["sha256"]
                archived_files[record["path"]] = content
            else:
                pass
    fingerprints = {}
    samples = json.loads(
        archived_files[
            "dense-restored-native-raw-scale-qualification/"
            "raw-scale/full-mmsu/generation/samples.json"
        ]
    )
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    for workload in ("tts", "mmsu"):
        destination = arguments.output_directory / "corpus-inputs" / workload
        destination.mkdir(parents=True)
        workload_catalog = [
            record for record in catalog if record["workload"] == workload
        ]
        assert len(workload_catalog) == (1088 if workload == "tts" else 40)
        selected_samples = []
        for record in workload_catalog:
            request_bytes = archived_files[
                f"{POPULATION_PREFIX}{workload}/{record['request_id']}.request.json"
            ]
            assert (
                hashlib.sha256(request_bytes).hexdigest()
                == record["raw_request_sha256"]
            )
            payload = json.loads(request_bytes)
            assert payload["request_id"] == record["request_id"]
            audio_path = Path(payload["audios"][0])
            asset = (
                arguments.output_directory
                / "assets"
                / (
                    "restored-reference-audio"
                    if workload == "tts"
                    else "restored-mmsu-inputs"
                )
                / audio_path.name
            )
            assert asset.is_file()
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            if audio_path.exists():
                assert audio_path.resolve() == asset.resolve()
            else:
                audio_path.symlink_to(asset)
            fingerprints[record["request_id"]] = {
                "request_sha256": record["raw_request_sha256"],
                "audio_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "audio_bytes": asset.stat().st_size,
            }
            filename = destination / record["request_filename"]
            filename.write_bytes(request_bytes)
            saved_request = {
                name: record[name]
                for name in ("request_id", "sample_id", "request_sha256", "target_text")
            }
            filename.with_name(
                filename.name.replace(".request.json", ".response.json")
            ).write_text(json.dumps(saved_request, indent=2))
            if workload == "mmsu":
                sample = samples_by_id[record["sample_id"]].copy()
                sample["audio_path"] = str(audio_path)
                selected_samples.append(sample)
            else:
                pass
        previous = json.loads(
            archived_files[f"{POPULATION_PREFIX}{workload}/results.json"]
        )
        sample_ids = [record["request_id"] for record in previous["per_request"]]
        assert set(sample_ids) == {record["sample_id"] for record in workload_catalog}
        order = (
            {"per_request": [{"id": sample_id} for sample_id in sample_ids]}
            if workload == "tts"
            else {"per_sample": [{"sample_id": sample_id} for sample_id in sample_ids]}
        )
        (
            destination
            / ("speed_results.json" if workload == "tts" else "results.json")
        ).write_text(json.dumps(order, indent=2))
        if workload == "mmsu":
            (destination / "samples.json").write_text(
                json.dumps(selected_samples, indent=2)
            )
        else:
            pass
    (arguments.output_directory / "restoration.json").write_text(
        json.dumps(
            {
                "input_archive_sha256": INPUT_ARCHIVE_SHA256,
                "result_archive_sha256": RESULT_ARCHIVE_SHA256,
                "requests": fingerprints,
                "scope": "Original request bytes and request ordering. response.json files are SavedRequest input records, not generated worker responses.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
