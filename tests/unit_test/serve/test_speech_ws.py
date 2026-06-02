# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import GenerateChunk
from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app
from sglang_omni.serve.protocol import SpeechStreamSessionConfig
from sglang_omni.serve.speech_service import SpeechService
from sglang_omni.serve.speech_ws import SpeechWebSocketSession


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


class BlockingStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        self.started.set()
        await asyncio.Future()
        yield GenerateChunk(request_id=request_id or "speech-ws")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class TwoChunkStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class InvalidAudioStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=object(),
            sample_rate=24000,
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class CompletedSpeechClient:
    def __init__(self) -> None:
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
        return SpeechResult(audio_bytes=b"RIFF", mime_type="audio/wav", format="wav")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class RecordingWebSocket:
    def __init__(self, *, fail_bytes: bool = False) -> None:
        self.fail_bytes = fail_bytes
        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED
        self.sent_text: list[dict[str, Any]] = []
        self.sent_bytes: list[bytes] = []

    async def send_text(self, payload: str) -> None:
        self.sent_text.append(json.loads(payload))

    async def send_bytes(self, payload: bytes) -> None:
        if self.fail_bytes:
            self.application_state = WebSocketState.DISCONNECTED
            self.client_state = WebSocketState.DISCONNECTED
            raise RuntimeError("client disconnected")
        self.sent_bytes.append(payload)

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


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


def test_speech_websocket_rejects_binary_client_frames() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"

        websocket.send_bytes(b"not-json-text")
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert "text frames" in event["error"]["message"]

        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"


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


def test_speech_websocket_stream_audio_defaults_to_non_streaming() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "wav"})
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"
        assert configured["stream_audio"] is False

        websocket.send_json({"type": "input.text", "text": "Hello."})
        assert websocket.receive_json()["type"] == "audio.start"
        assert websocket.receive_bytes() == b"RIFF"
        assert websocket.receive_json()["type"] == "audio.done"

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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("stream_audio", "true"),
        ("speed", "1.2"),
        ("max_new_tokens", "5"),
        ("token_count", "5"),
        ("duration_tokens", "5"),
    ],
)
def test_speech_websocket_rejects_stringified_config_types(
    field_name: str, value: str
) -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                field_name: value,
                "response_format": "pcm",
            }
        )
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert event["error"]["param"] == field_name


def test_speech_websocket_streaming_accepts_speed() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
                "speed": 1.1,
            }
        )
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"


def test_speech_websocket_cancellation_aborts_active_request() -> None:
    async def run() -> None:
        client_impl = BlockingStreamingSpeechClient()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechService(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        task = asyncio.create_task(session._generate_sentence("Hello."))
        await client_impl.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client_impl.aborted == [websocket.sent_text[0]["id"]]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_send_failure_aborts_active_stream() -> None:
    async def run() -> None:
        client_impl = TwoChunkStreamingSpeechClient()
        websocket = RecordingWebSocket(fail_bytes=True)
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechService(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == [websocket.sent_text[0]["id"]]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_stream_exception_aborts_active_request() -> None:
    async def run() -> None:
        client_impl = InvalidAudioStreamingSpeechClient()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechService(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        await session._generate_sentence("Hello.")

        assert client_impl.aborted == [websocket.sent_text[0]["id"]]
        assert websocket.sent_text[-2]["type"] == "error"
        assert websocket.sent_text[-1]["type"] == "audio.done"
        assert websocket.sent_text[-1]["error"] is True
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_completed_send_failure_does_not_abort() -> None:
    async def run() -> None:
        client_impl = CompletedSpeechClient()
        websocket = RecordingWebSocket(fail_bytes=True)
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechService(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=False)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == []
        assert session.active_request_id is None

    asyncio.run(run())
