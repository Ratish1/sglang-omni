# SPDX-License-Identifier: Apache-2.0
"""Check speech usage headers through the real Client and HTTP adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import create_app


class ResultCoordinator(Coordinator):
    def __init__(self, usage: dict[str, int | float | None] | None) -> None:
        super().__init__(
            completion_endpoint="inproc://review-complete",
            abort_endpoint="inproc://review-abort",
            entry_stage="preprocess",
            terminal_stages=["vocoder"],
        )
        self.usage = usage

    async def submit(self, request_id: str, request: OmniRequest) -> dict[str, object]:
        assert request_id.startswith("speech-")
        assert not request.params["stream"]
        return {
            "audio_data": [0.0, 0.1, -0.1, 0.0],
            "sample_rate": 24000,
            "finish_reason": "stop",
            "usage": self.usage,
        }


def main() -> None:
    cases = (
        ("positive", {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        ("zero", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        ("absent", None),
        (
            "partial",
            {"prompt_tokens": 5, "completion_tokens": None, "total_tokens": None},
        ),
        ("timing_only", {"engine_time_s": 1.25}),
        ("total_only", {"total_tokens": 0}),
    )
    observations = {}
    audio_hashes = set()
    for name, usage in cases:
        client = Client(ResultCoordinator(usage))
        with TestClient(create_app(client, model_name="review-tts")) as http_client:
            response = http_client.post(
                "/v1/audio/speech", json={"input": "hello", "response_format": "wav"}
            )
        assert response.status_code == 200, response.text
        assert response.content.startswith(b"RIFF")
        assert response.headers["x-finish-reason"] == "stop"
        audio_hashes.add(hashlib.sha256(response.content).hexdigest())
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.endswith("tokens") or key == "x-engine-time"
        }
        if name == "positive":
            assert headers["x-sglang-omni-total-tokens"] == "7"
            assert (
                headers["x-prompt-tokens"]
                == headers["x-sglang-omni-input-tokens"]
                == "5"
            )
            assert (
                headers["x-completion-tokens"]
                == headers["x-sglang-omni-output-tokens"]
                == "2"
            )
        elif name == "zero":
            assert len(headers) == 5
            assert set(headers.values()) == {"0"}
        elif name == "absent":
            assert headers == {}
        elif name == "partial":
            assert headers == {
                "x-prompt-tokens": "5",
                "x-sglang-omni-input-tokens": "5",
            }
        elif name == "timing_only":
            assert headers == {"x-engine-time": "1.25"}
        else:
            assert headers == {"x-sglang-omni-total-tokens": "0"}
        observations[name] = headers
    assert len(audio_hashes) == 1
    evidence = {"headers": observations, "same_audio_bytes_across_usage_cases": True}
    Path("/data/pr2418-review/header-probe.json").write_text(
        json.dumps(evidence, indent=2)
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
