# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app


class RecordingBatchSpeechClient:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request_id, speed, allow_format_fallback
        self.requests.append(request)
        return SpeechResult(
            audio_bytes=f"audio:{request.prompt}".encode(),
            mime_type=f"audio/{response_format}",
            format=response_format,
        )


def test_batch_speech_preserves_order_and_item_errors() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(
        create_app(client_impl, model_name="tts", tts_batch_max_items=4)
    )

    response = client.post(
        "/v1/audio/speech/batch",
        json={
            "response_format": "wav",
            "items": [
                {"input": "first"},
                {"input": "   "},
                {"input": "third", "response_format": "pcm"},
                {"input": 123},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["succeeded"] == 2
    assert body["failed"] == 2
    assert [item["index"] for item in body["results"]] == [0, 1, 2, 3]
    assert body["results"][0]["status"] == "success"
    assert body["results"][1]["status"] == "error"
    assert body["results"][1]["error"]["param"] == "input"
    assert body["results"][2]["media_type"] == "audio/pcm"
    assert body["results"][3]["error"]["param"] == "input"
    assert [request.prompt for request in client_impl.requests] == ["first", "third"]


def test_batch_speech_rejects_invalid_envelope_before_item_work() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(
        create_app(client_impl, model_name="tts", tts_batch_max_items=1)
    )

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one"}, {"input": "two"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "items"
    assert client_impl.requests == []


def test_batch_speech_rejects_streaming_items() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one", "stream": True}]},
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["status"] == "error"
    assert item["error"]["param"] == "items.0.stream"
    assert client_impl.requests == []
