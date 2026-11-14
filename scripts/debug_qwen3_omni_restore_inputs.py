"""Restore original corpus audio with byte-exact fingerprint verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import load_seedtts_samples
from scripts.debug_qwen3_omni_replay_corpus import CompletedSpeech, SavedRequest


class AudioFingerprint(BaseModel):
    bytes: int
    sha256: str


class OriginalAudio(BaseModel):
    source_audio_files: dict[str, AudioFingerprint]


class OriginalRequest(BaseModel):
    request_id: str
    audios: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--mmsu-repository", required=True)
    arguments = parser.parse_args()
    fingerprints = OriginalAudio.model_validate_json(
        (arguments.input_directory / "input-audio-fingerprints.json").read_bytes()
    ).source_audio_files
    restored: dict[str, AudioFingerprint] = {}
    for workload in ("tts", "mmsu"):
        inputs = arguments.input_directory / "corpus-inputs" / workload
        if workload == "tts":
            samples = load_seedtts_samples(
                SEEDTTS_DATASET_ID, revision=SEEDTTS_DATASET_REVISION
            )
            source_paths = {
                sample.sample_id: Path(sample.ref_audio) for sample in samples
            }
            targets = {sample.sample_id: sample.target_text for sample in samples}
        else:
            audio_samples = load_mmsu_samples(repo_id=arguments.mmsu_repository)
            source_paths = {
                sample.sample_id: Path(sample.audio_path) for sample in audio_samples
            }
            targets = {}
        speeches = list(inputs.glob("*.speech.json"))
        assert len(speeches) == (1088 if workload == "tts" else 40)
        for source in speeches:
            speech = CompletedSpeech.model_validate_json(source.read_bytes())
            filename = speech.request_id if workload == "tts" else speech.sample_id
            request_path = inputs / f"{filename}.request.json"
            request_bytes = request_path.read_bytes()
            request = OriginalRequest.model_validate_json(request_bytes)
            assert request.request_id == speech.request_id
            assert len(request.audios) == 1
            if workload == "tts":
                assert targets[speech.sample_id] == speech.target_text
            else:
                pass
            destination = Path(request.audios[0])
            expected = fingerprints[str(destination)]
            audio_bytes = source_paths[speech.sample_id].read_bytes()
            actual = AudioFingerprint(
                bytes=len(audio_bytes), sha256=hashlib.sha256(audio_bytes).hexdigest()
            )
            assert actual == expected, destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(audio_bytes)
            assert (
                hashlib.sha256(destination.read_bytes()).hexdigest() == expected.sha256
            )
            restored[str(destination)] = actual
            record = SavedRequest(
                request_id=speech.request_id,
                sample_id=speech.sample_id,
                target_text=speech.target_text,
                request_sha256=hashlib.sha256(
                    json.dumps(json.loads(request_bytes), sort_keys=True).encode()
                ).hexdigest(),
            )
            with (inputs / f"{filename}.response.json").open("x") as output:
                output.write(record.model_dump_json(indent=2))
        print(f"Restored {len(speeches)} {workload} inputs", flush=True)
    assert restored == fingerprints
    report = {
        "seedtts_repository": SEEDTTS_DATASET_ID,
        "seedtts_revision": SEEDTTS_DATASET_REVISION,
        "mmsu_repository": arguments.mmsu_repository,
        "source_audio_files": {
            path: fingerprint.model_dump() for path, fingerprint in restored.items()
        },
    }
    with (arguments.input_directory / "restoration-proof.json").open("x") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
else:
    pass
