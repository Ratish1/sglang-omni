# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app
from sglang_omni.serve.speech_service import SpeechService


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


class BlockingBatchSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def test_batch_speech_preserves_order_and_item_errors() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(
        create_app(client_impl, model_name="tts", tts_batch_max_items=5)
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
                {"response_format": "wav"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert body["succeeded"] == 2
    assert body["failed"] == 3
    assert [item["index"] for item in body["results"]] == [0, 1, 2, 3, 4]
    assert body["results"][0]["status"] == "success"
    assert body["results"][1]["status"] == "error"
    assert body["results"][1]["error"]["param"] == "items.1.input"
    assert body["results"][2]["media_type"] == "audio/pcm"
    assert body["results"][3]["error"]["param"] == "items.3.input"
    assert body["results"][4]["error"]["param"] == "items.4.input"
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


def test_batch_speech_cancellation_aborts_started_items() -> None:
    async def run() -> None:
        service = SpeechService(default_model="tts")
        batch = service.parse_batch_request({"items": [{"input": "one"}]})
        client_impl = BlockingBatchSpeechClient()

        task = asyncio.create_task(
            service.create_speech_batch(client_impl, batch, request_id="batch")
        )
        await client_impl.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client_impl.aborted == ["batch-0"]

    asyncio.run(run())
