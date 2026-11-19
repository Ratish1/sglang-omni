# SPDX-License-Identifier: Apache-2.0
"""Record public chat voice behavior without retaining PCM."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import requests

from scripts.profile_qwen3_omni_mmsu import SEED

REQUEST_TIMEOUT_SECONDS = 120


class StreamVoiceResult(TypedDict):
    http_status: int
    text: str
    audio_bytes: int
    audio_sha256: str
    finish_reasons: list[str]
    error_codes: list[int]
    done_count: int
    events_after_done: int


def probe_voice(
    server_url: str, audio_config: dict[str, str | int | None]
) -> StreamVoiceResult:
    result: StreamVoiceResult = {
        "http_status": 0,
        "text": "",
        "audio_bytes": 0,
        "audio_sha256": "",
        "finish_reasons": [],
        "error_codes": [],
        "done_count": 0,
        "events_after_done": 0,
    }
    audio_hash = hashlib.sha256()
    with requests.Session() as http:
        http.trust_env = False
        with http.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "qwen3-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": "Repeat exactly: This is a short voice check.",
                    }
                ],
                "modalities": ["text", "audio"],
                "temperature": 0,
                "max_tokens": 48,
                "seed": SEED,
                "stream": True,
                "audio": audio_config,
            },
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            result["http_status"] = response.status_code
            for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                else:
                    pass
                if result["done_count"]:
                    result["events_after_done"] += 1
                else:
                    pass
                event_bytes = line.removeprefix(b"data: ")
                if event_bytes == b"[DONE]":
                    result["done_count"] += 1
                else:
                    event = json.loads(event_bytes)
                    if "error" in event:
                        result["error_codes"].append(event["error"]["code"])
                    else:
                        for choice in event["choices"]:
                            delta = choice["delta"]
                            result["text"] += delta.get("content", "")
                            if audio := delta.get("audio"):
                                audio_bytes = base64.b64decode(
                                    audio["data"], validate=True
                                )
                                result["audio_bytes"] += len(audio_bytes)
                                audio_hash.update(audio_bytes)
                            else:
                                pass
                            if finish_reason := choice.get("finish_reason"):
                                result["finish_reasons"].append(finish_reason)
                            else:
                                pass
    result["audio_sha256"] = audio_hash.hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    requests_by_case = {
        "default": {"format": "pcm"},
        "ethan": {"format": "pcm", "voice": "Ethan"},
        "chelsie": {"format": "pcm", "voice": "Chelsie"},
        "chelsie_case": {"format": "pcm", "voice": "cHeLsIe"},
        "null": {"format": "pcm", "voice": None},
        "invalid": {"format": "pcm", "voice": "invalid"},
        "numeric": {"format": "pcm", "voice": 123},
        "default_repeat": {"format": "pcm"},
    }
    arguments.output.write_text(
        json.dumps(
            {
                case: probe_voice(arguments.server_url, audio_config)
                for case, audio_config in requests_by_case.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
