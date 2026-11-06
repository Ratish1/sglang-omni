# SPDX-License-Identifier: Apache-2.0
"""Exercise all input modality subsets through a live serial model pipeline."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import uuid
from itertools import combinations
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument(
        "--output-mode", choices=("text", "audio", "both"), required=True
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    inputs = {
        "image": arguments.input_image.resolve(strict=True),
        "audio": arguments.input_audio.resolve(strict=True),
        "video": arguments.input_video.resolve(strict=True),
    }
    input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in inputs.items()
    }
    output_modalities = (
        ["text", "audio"]
        if arguments.output_mode == "both"
        else [arguments.output_mode]
    )
    reports = []
    with httpx.Client(timeout=300) as client:
        for modality_count in range(1, 5):
            for modalities in combinations(
                ("text", "audio", "image", "video"), modality_count
            ):
                case = f"{'-'.join(modalities)}-{arguments.output_mode}-stream-{arguments.stream}-video-audio-{arguments.use_audio_in_video}"
                request_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"qwen3-omni-flow-audit/{case}")
                )
                payload = {
                    "model": arguments.model_name,
                    "request_id": request_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Describe the supplied input in one short sentence."
                                if "text" in modalities
                                else ""
                            ),
                        }
                    ],
                    "modalities": output_modalities,
                    "audio": {"format": "wav"},
                    "max_tokens": arguments.max_output_tokens,
                    "temperature": 0,
                    "stream": arguments.stream,
                }
                for modality, filename in inputs.items():
                    if modality in modalities:
                        payload[f"{modality}s"] = [str(filename)]
                    else:
                        pass
                if "video" in modalities:
                    payload["use_audio_in_video"] = arguments.use_audio_in_video
                else:
                    pass
                (arguments.output_directory / f"{request_id}.request.json").write_text(
                    json.dumps(payload, indent=2)
                )
                started = time.monotonic()
                record = {
                    "case": case,
                    "request_id": request_id,
                    "input_modalities": modalities,
                    "audio_bytes": 0,
                    "stream_done": False,
                    "finish_reasons": [],
                    "events": [],
                }
                with client.stream(
                    "POST", f"{arguments.base_url}/v1/chat/completions", json=payload
                ) as response:
                    record["http_status"] = response.status_code
                    if response.status_code != 200:
                        record["error"] = response.read().decode()
                        record["success"] = False
                    elif arguments.stream:
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            else:
                                event_text = line.removeprefix("data:").strip()
                            if event_text == "[DONE]":
                                record["stream_done"] = True
                            else:
                                event = json.loads(event_text)
                                for choice in event.get("choices", []):
                                    if choice.get("finish_reason") is not None:
                                        record["finish_reasons"].append(
                                            choice["finish_reason"]
                                        )
                                    else:
                                        pass
                                    audio = choice.get("delta", {}).get("audio")
                                    if audio is not None and "data" in audio:
                                        raw = base64.b64decode(
                                            audio.pop("data"), validate=True
                                        )
                                        record["audio_bytes"] += len(raw)
                                        audio["sha256"] = hashlib.sha256(
                                            raw
                                        ).hexdigest()
                                        audio["byte_count"] = len(raw)
                                    else:
                                        pass
                                record["events"].append(event)
                        record["success"] = record["stream_done"] and bool(
                            record["finish_reasons"]
                        )
                    else:
                        event = json.loads(response.read())
                        for choice in event["choices"]:
                            record["finish_reasons"].append(choice["finish_reason"])
                            audio = choice["message"].get("audio")
                            if audio is not None and "data" in audio:
                                raw = base64.b64decode(audio.pop("data"), validate=True)
                                record["audio_bytes"] += len(raw)
                                audio["sha256"] = hashlib.sha256(raw).hexdigest()
                                audio["byte_count"] = len(raw)
                                (
                                    arguments.output_directory / f"{request_id}.wav"
                                ).write_bytes(raw)
                            else:
                                pass
                        record["events"].append(event)
                        record["success"] = bool(record["finish_reasons"])
                if "audio" in output_modalities:
                    record["success"] = record["success"] and record["audio_bytes"] > 0
                else:
                    pass
                record["elapsed_seconds"] = time.monotonic() - started
                (arguments.output_directory / f"{request_id}.response.json").write_text(
                    json.dumps(record, indent=2)
                )
                reports.append(record)
                print(
                    json.dumps(
                        {
                            name: value
                            for name, value in record.items()
                            if name != "events"
                        }
                    ),
                    flush=True,
                )
    summary = {
        "input_files": {name: str(filename) for name, filename in inputs.items()},
        "input_sha256": input_hashes,
        "cases": len(reports),
        "successes": sum(record["success"] for record in reports),
        "scope": "Serial HTTP transport, preprocessing, media scatter, stage handoff and terminal delivery smoke coverage. Generation is explicitly token-bounded. This is not a semantic accuracy, WER, quality, concurrency or performance certification. The video-audio branch is covered only when the supplied video contains an audio track and extraction was requested.",
        "results": [
            {name: value for name, value in record.items() if name != "events"}
            for record in reports
        ],
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if summary["successes"] != summary["cases"]:
        raise SystemExit("One or more model flow cases failed; see the saved responses")
    else:
        print(
            json.dumps({"cases": summary["cases"], "successes": summary["successes"]})
        )


if __name__ == "__main__":
    main()
