# SPDX-License-Identifier: Apache-2.0
"""Stateful WebSocket serving for text-to-speech streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, ClientError
from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_audio
from sglang_omni.serve.protocol import CreateSpeechRequest, SpeechStreamSessionConfig
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    internal_error,
    openai_error_payload,
)
from sglang_omni.serve.speech_service import SpeechService, speech_audio_delta

logger = logging.getLogger(__name__)

CONFIG_TIMEOUT_S = 10.0
IDLE_TIMEOUT_S = 30.0
MAX_CONFIG_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_TEXT_MESSAGE_BYTES = 128 * 1024
MAX_BUFFERED_TEXT_CHARS = 256 * 1024
SENTENCE_BOUNDARIES = frozenset(".!?。！？")
CLAUSE_BOUNDARIES = frozenset(".!?。！？,，;；")
SUPPORTED_SPLIT_GRANULARITIES = frozenset({"sentence", "clause"})


def new_speech_ws_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class SpeechWebSocketSession:
    """Own one `/v1/audio/speech/stream` WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        speech_service: SpeechService,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.speech_service = speech_service
        self.session_id = new_speech_ws_id("speech_ws")
        self.closed = False
        self.config: SpeechStreamSessionConfig | None = None
        self.buffer = ""
        self.sentence_index = 0
        self.active_request_id: str | None = None

    async def run(self) -> None:
        try:
            configured = await self._receive_config()
            if not configured:
                return
            await self._message_loop()
        finally:
            await self.teardown()

    async def _receive_config(self) -> bool:
        try:
            raw = await self._receive_text_frame(
                timeout_s=CONFIG_TIMEOUT_S,
                max_bytes=MAX_CONFIG_MESSAGE_BYTES,
                message_kind="session",
            )
            payload = self._parse_message(raw)
            if payload.get("type") != "session.config":
                await self._send_error(
                    bad_request(
                        "first WebSocket message must be session.config",
                        param="type",
                    )
                )
                return False
            self.config = self._parse_config(payload)
            await self._send_json(
                {
                    "type": "session.configured",
                    "session_id": self.session_id,
                    "response_format": self.config.response_format,
                    "stream_audio": self.config.stream_audio,
                    "split_granularity": self.config.split_granularity,
                }
            )
            return True
        except asyncio.TimeoutError:
            await self._send_error(
                bad_request("session.config was not received before timeout")
            )
        except (SpeechAPIError, ValidationError) as exc:
            await self._send_error(_speech_error_from_exception(exc))
        except (json.JSONDecodeError, ValueError) as exc:
            await self._send_error(bad_request(str(exc)))
        except WebSocketDisconnect:
            pass
        return False

    async def _message_loop(self) -> None:
        while not self.closed:
            try:
                raw = await self._receive_text_frame(
                    timeout_s=IDLE_TIMEOUT_S,
                    max_bytes=MAX_TEXT_MESSAGE_BYTES,
                    message_kind="text",
                )
                payload = self._parse_message(raw)
            except asyncio.TimeoutError:
                await self._send_error(bad_request("speech WebSocket idle timeout"))
                return
            except json.JSONDecodeError as exc:
                await self._send_error(bad_request(str(exc)))
                continue
            except ValueError as exc:
                await self._send_error(bad_request(str(exc)))
                continue
            except WebSocketDisconnect:
                return

            message_type = payload.get("type")
            if message_type == "input.text":
                await self._handle_input_text(payload)
            elif message_type == "input.done":
                await self._handle_input_done()
                return
            else:
                await self._send_error(
                    bad_request(
                        f"unsupported speech WebSocket message type: {message_type!r}",
                        param="type",
                    )
                )

    async def _handle_input_text(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if not isinstance(text, str):
            await self._send_error(bad_request("input.text text must be a string"))
            return
        if not text:
            return
        if len(self.buffer) + len(text) > MAX_BUFFERED_TEXT_CHARS:
            self.buffer = ""
            await self._send_error(
                bad_request(
                    f"buffered speech text exceeds {MAX_BUFFERED_TEXT_CHARS} characters",
                    param="text",
                )
            )
            self.closed = True
            return
        self.buffer += text
        for sentence in self._pop_complete_segments():
            await self._generate_sentence(sentence)

    async def _handle_input_done(self) -> None:
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            await self._generate_sentence(remaining)
        await self._send_json(
            {
                "type": "session.done",
                "session_id": self.session_id,
                "total_sentences": self.sentence_index,
            }
        )

    def _parse_config(
        self,
        payload: dict[str, Any],
    ) -> SpeechStreamSessionConfig:
        raw_config = payload.get("session")
        if raw_config is None:
            raw_config = {key: value for key, value in payload.items() if key != "type"}
        if not isinstance(raw_config, dict):
            raise bad_request(
                "session.config session must be an object",
                param="session",
            )
        config = SpeechStreamSessionConfig.model_validate(raw_config)
        if config.split_granularity not in SUPPORTED_SPLIT_GRANULARITIES:
            supported = ", ".join(sorted(SUPPORTED_SPLIT_GRANULARITIES))
            raise bad_request(
                f"split_granularity must be one of: {supported}",
                param="split_granularity",
            )
        if config.stream_audio and config.response_format.lower() != "pcm":
            raise bad_request(
                "stream_audio=true requires response_format='pcm'",
                param="response_format",
            )
        if config.stream_audio and config.speed != 1.0:
            raise bad_request(
                "stream_audio=true requires speed=1.0",
                param="speed",
            )
        self.speech_service.prepare_request(
            self._speech_request_from_config(config, "probe")
        )
        return config

    async def _generate_sentence(self, sentence: str) -> None:
        assert self.config is not None
        sentence_index = self.sentence_index
        self.sentence_index += 1
        request_id = f"{self.session_id}-{sentence_index}"
        self.active_request_id = request_id
        await self._send_json(
            {
                "type": "audio.start",
                "id": request_id,
                "sentence_index": sentence_index,
                "sentence_text": sentence,
                "format": self.config.response_format,
                "sample_rate": DEFAULT_SAMPLE_RATE,
            }
        )
        total_bytes = 0
        failed = False
        try:
            if self.config.stream_audio:
                total_bytes = await self._stream_sentence_audio(
                    sentence,
                    request_id=request_id,
                )
            else:
                total_bytes = await self._send_sentence_audio(
                    sentence,
                    request_id=request_id,
                )
        except WebSocketDisconnect:
            failed = True
            await self._abort_active_request()
            raise
        except Exception as exc:
            failed = True
            if isinstance(exc, SpeechAPIError):
                error = exc
            else:
                error = internal_error(str(exc))
                logger.exception("TTS WebSocket sentence failed: %s", request_id)
            await self._send_error(error)
        finally:
            self.active_request_id = None
            if self.websocket.application_state == WebSocketState.CONNECTED:
                await self._send_json(
                    {
                        "type": "audio.done",
                        "id": request_id,
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": failed,
                    }
                )

    async def _stream_sentence_audio(self, sentence: str, *, request_id: str) -> int:
        assert self.config is not None
        request = self._speech_request_from_config(sentence=sentence, stream=True)
        gen_req = self.speech_service.build_generate_request(request)
        emitted_samples = 0
        total_bytes = 0
        chunk_count = 0
        async for chunk in self.client.generate(gen_req, request_id=request_id):
            if chunk.audio_data is None:
                continue
            sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
            audio_data, emitted_samples = speech_audio_delta(
                chunk.audio_data,
                emitted_samples=emitted_samples,
                is_terminal=chunk.finish_reason is not None,
            )
            if audio_data is None:
                continue
            audio_bytes, _ = encode_audio(
                audio_data,
                response_format="pcm",
                sample_rate=sample_rate,
                speed=1.0,
                allow_format_fallback=False,
            )
            if not audio_bytes:
                continue
            await self.websocket.send_bytes(audio_bytes)
            total_bytes += len(audio_bytes)
            chunk_count += 1
        if chunk_count == 0:
            raise ClientError("No audio output generated from the pipeline.")
        return total_bytes

    async def _send_sentence_audio(self, sentence: str, *, request_id: str) -> int:
        assert self.config is not None
        request = self._speech_request_from_config(sentence=sentence, stream=False)
        gen_req = self.speech_service.build_generate_request(request)
        result = await self.client.speech(
            gen_req,
            request_id=request_id,
            response_format=request.response_format,
            speed=request.speed,
            allow_format_fallback=False,
        )
        await self.websocket.send_bytes(result.audio_bytes)
        return len(result.audio_bytes)

    def _speech_request_from_config(
        self,
        config: SpeechStreamSessionConfig | None = None,
        sentence: str = "",
        *,
        stream: bool | None = None,
    ) -> CreateSpeechRequest:
        config = config or self.config
        assert config is not None
        payload = config.model_dump(exclude={"stream_audio", "split_granularity"})
        payload["input"] = sentence
        payload["stream"] = config.stream_audio if stream is None else stream
        return CreateSpeechRequest.model_validate(payload)

    def _pop_complete_segments(self) -> list[str]:
        assert self.config is not None
        boundaries = (
            CLAUSE_BOUNDARIES
            if self.config.split_granularity == "clause"
            else SENTENCE_BOUNDARIES
        )
        segments: list[str] = []
        start = 0
        for index, char in enumerate(self.buffer):
            if char in boundaries:
                segment = self.buffer[start : index + 1].strip()
                if segment:
                    segments.append(segment)
                start = index + 1
        self.buffer = self.buffer[start:]
        return segments

    def _parse_message(self, raw: str) -> dict[str, Any]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("speech WebSocket messages must be JSON objects")
        return payload

    async def _receive_text_frame(
        self,
        *,
        timeout_s: float,
        max_bytes: int,
        message_kind: str,
    ) -> str:
        message = await asyncio.wait_for(self.websocket.receive(), timeout=timeout_s)
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            raise WebSocketDisconnect
        if message_type != "websocket.receive":
            raise ValueError(
                f"unsupported speech WebSocket ASGI message: {message_type}"
            )

        raw = message.get("text")
        if raw is None:
            frame_bytes = message.get("bytes")
            if frame_bytes is not None and len(frame_bytes) > max_bytes:
                raise ValueError(
                    f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
                )
            raise ValueError("speech WebSocket client messages must be text frames")
        self._validate_message_size(raw, max_bytes, message_kind)
        return raw

    @staticmethod
    def _validate_message_size(raw: str, max_bytes: int, message_kind: str) -> None:
        if len(raw.encode("utf-8")) > max_bytes:
            raise ValueError(
                f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
            )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        await self.websocket.send_text(json.dumps(payload))

    async def _send_error(self, error: SpeechAPIError) -> None:
        await self._send_json(
            {
                "type": "error",
                **openai_error_payload(
                    error.message,
                    error_type=error.error_type,
                    param=error.param,
                    code=error.code,
                ),
            }
        )

    async def _abort_active_request(self) -> None:
        if self.active_request_id is not None and hasattr(self.client, "abort"):
            await self.client.abort(self.active_request_id)

    async def teardown(self) -> None:
        self.closed = True
        await self._abort_active_request()
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()


def _speech_error_from_exception(exc: Exception) -> SpeechAPIError:
    if isinstance(exc, SpeechAPIError):
        return exc
    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg") or "invalid speech WebSocket config"
        location = ".".join(str(item) for item in first_error.get("loc", ()))
        return bad_request(f"{location}: {message}" if location else str(message))
    return bad_request(str(exc))
