"""Replay the saved audio controls unchanged, then an ordinary short spoken reply."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import uuid
from pathlib import Path

import httpx
from pydantic import BaseModel


class ControlRecord(BaseModel):
    request_id: str
    source_request: Path
    waveform_sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--include-visual", action="store_true")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    records = [
        ControlRecord.model_validate(record)
        for record in json.loads(
            (arguments.control_directory / "summary.json").read_text()
        )
    ]
    selected_controls = {
        record.request_id: record
        for record in records
        if json.loads(record.source_request.read_text()).get("audios")
        or arguments.include_visual
    }
    request_order = [
        request_id
        for line in (arguments.capture_directory / "steps.jsonl")
        .read_text()
        .splitlines()
        for metadata in [json.loads(line)]
        if metadata["phase"] == "EXTEND"
        for request_id in metadata["request_ids"]
        if request_id in selected_controls
    ]
    assert len(request_order) == len(set(request_order)) == len(selected_controls)
    requests = [
        (
            request_id,
            selected_controls[request_id].source_request.read_bytes(),
            selected_controls[request_id].waveform_sha256,
        )
        for request_id in request_order
    ]
    short_request_id = str(
        uuid.uuid5(uuid.UUID(request_order[0]), "prefill-short-text-control")
    )
    short_request = {
        "model": json.loads(requests[0][1])["model"],
        "request_id": short_request_id,
        "messages": [{"role": "user", "content": "Say hello."}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    requests.append((short_request_id, json.dumps(short_request).encode(), None))
    observations = []
    with httpx.Client(timeout=300) as client:
        health = client.get(f"{arguments.base_url}/health")
        health.raise_for_status()
        for request_id, request_bytes, expected_waveform_hash in requests:
            assert json.loads(request_bytes)["request_id"] == request_id
            (arguments.output_directory / f"{request_id}.request.json").write_bytes(
                request_bytes
            )
            response = client.post(
                f"{arguments.base_url}/v1/chat/completions",
                content=request_bytes,
                headers={"Content-Type": "application/json"},
            )
            (arguments.output_directory / f"{request_id}.response.json").write_bytes(
                response.content
            )
            response.raise_for_status()
            result = response.json()
            assert result["id"] == f"chatcmpl-{request_id}"
            message = result["choices"][0]["message"]
            waveform = base64.b64decode(message["audio"]["data"], validate=True)
            (arguments.output_directory / f"{request_id}.wav").write_bytes(waveform)
            waveform_hash = hashlib.sha256(waveform).hexdigest()
            observation = {
                "request_id": request_id,
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "waveform_sha256": waveform_hash,
                "waveform_matches_saved": (
                    waveform_hash == expected_waveform_hash
                    if expected_waveform_hash is not None
                    else None
                ),
                "text": message["content"],
                "finish_reason": result["choices"][0]["finish_reason"],
            }
            observations.append(observation)
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(observations, indent=2)
            )
            print(json.dumps(observation), flush=True)


if __name__ == "__main__":
    main()
