# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client import GenerateChunk
from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app


class StreamingSpeechClient:
    def __init__(self) -> None:
        self.generated_prompts: list[str] = []
        self.speech_prompts: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        self.generated_prompts.append(request.prompt)
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )

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
        self.speech_prompts.append(request.prompt)
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type=f"audio/{response_format}",
            format=response_format,
        )


def test_speech_websocket_streams_sentences_as_binary_frames() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "session": {
                    "response_format": "pcm",
                    "stream_audio": True,
                    "split_granularity": "sentence",
                },
            }
        )
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"

        websocket.send_json({"type": "input.text", "text": "Hello. Second"})
        first_start = websocket.receive_json()
        first_audio = websocket.receive_bytes()
        first_done = websocket.receive_json()

        assert first_start["type"] == "audio.start"
        assert first_start["sentence_text"] == "Hello."
        assert first_audio
        assert first_done["type"] == "audio.done"
        assert first_done["error"] is False

        websocket.send_json({"type": "input.done"})
        second_start = websocket.receive_json()
        second_audio = websocket.receive_bytes()
        second_done = websocket.receive_json()
        session_done = websocket.receive_json()

        assert second_start["sentence_text"] == "Second"
        assert second_audio
        assert second_done["error"] is False
        assert session_done["type"] == "session.done"
        assert session_done["total_sentences"] == 2

    assert client_impl.generated_prompts == ["Hello.", "Second"]


def test_speech_websocket_rejects_missing_initial_config() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "input.text", "text": "hello"})
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert event["error"]["param"] == "type"


def test_speech_websocket_supports_non_streaming_sentence_frames() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "response_format": "wav",
                "stream_audio": False,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"

        websocket.send_json({"type": "input.text", "text": "Hello."})
        assert websocket.receive_json()["type"] == "audio.start"
        assert websocket.receive_bytes() == b"RIFF"
        assert websocket.receive_json()["type"] == "audio.done"
        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"

    assert client_impl.speech_prompts == ["Hello."]


def test_speech_websocket_unknown_message_type_is_recoverable() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json({"type": "unexpected"})
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["error"]["param"] == "type"
        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"
