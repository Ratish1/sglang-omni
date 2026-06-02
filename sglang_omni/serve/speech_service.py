# SPDX-License-Identifier: Apache-2.0
"""Shared service layer for TTS speech API requests."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import ValidationError

from sglang_omni.client import ClientError, GenerateRequest, SamplingParams
from sglang_omni.client.audio import FORMAT_MIME_TYPES, to_numpy
from sglang_omni.serve.protocol import (
    SUPPORTED_TTS_LANGUAGES,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SUPPORTED_TTS_TASK_TYPES,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    CreateSpeechBatchRequest,
    CreateSpeechRequest,
    SpeechBatchItem,
    SpeechBatchResponse,
    SpeechBatchResult,
    SpeechReference,
)
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    internal_error,
    openai_error_payload,
)

if TYPE_CHECKING:
    from sglang_omni.client import Client
    from sglang_omni.serve.speech_voices import (
        SpeakerSampleStore,
        UploadedVoiceReference,
    )

_LANGUAGE_CANONICAL = {
    language.lower(): language for language in SUPPORTED_TTS_LANGUAGES
}
_TASK_TYPE_CANONICAL = {
    task_type.replace("_", "").replace("-", "").lower(): task_type
    for task_type in SUPPORTED_TTS_TASK_TYPES
}


class SpeechService:
    """Validate and lower OpenAI-compatible TTS requests."""

    def __init__(
        self,
        *,
        default_model: str,
        requires_uploaded_voice_for_named_voice: bool = False,
        allowed_local_media_paths: list[str | Path] | None = None,
        voice_store: SpeakerSampleStore | None = None,
        tts_batch_max_items: int = 32,
    ) -> None:
        if tts_batch_max_items < 1:
            raise ValueError("tts_batch_max_items must be greater than 0")
        self.default_model = default_model
        self.requires_uploaded_voice_for_named_voice = (
            requires_uploaded_voice_for_named_voice
        )
        self.voice_store = voice_store
        self.tts_batch_max_items = int(tts_batch_max_items)
        self.allowed_local_media_paths = tuple(
            _resolve_allowed_local_media_path(path)
            for path in (allowed_local_media_paths or [])
            if str(path).strip()
        )

    def parse_request(self, payload: Any) -> CreateSpeechRequest:
        """Parse and validate a raw HTTP payload."""

        if not isinstance(payload, dict):
            raise bad_request("speech request body must be a JSON object")
        self._validate_raw_payload(payload)
        try:
            request = CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        return self.prepare_request(request)

    def parse_batch_request(self, payload: Any) -> CreateSpeechBatchRequest:
        """Parse and validate a raw batch speech payload."""

        if not isinstance(payload, dict):
            raise bad_request("speech batch request body must be a JSON object")
        try:
            request = CreateSpeechBatchRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        if not request.items:
            raise bad_request("items must contain at least one request", param="items")
        if len(request.items) > self.tts_batch_max_items:
            raise bad_request(
                f"items must contain at most {self.tts_batch_max_items} requests",
                param="items",
            )
        if request.stream:
            raise bad_request(
                "stream is not supported for batch speech requests",
                param="stream",
            )
        self._validate_batch_defaults(request)
        return request

    def prepare_request(self, request: CreateSpeechRequest) -> CreateSpeechRequest:
        """Validate and normalize a request that was already parsed."""

        updates: dict[str, Any] = {}

        input_text = request.input
        if not isinstance(input_text, str) or not input_text.strip():
            raise bad_request("input must be a non-empty string", param="input")

        response_format = _normalize_response_format(request.response_format)
        _validate_encoder_dependency(response_format)
        updates["response_format"] = response_format

        if not TTS_SPEED_MIN <= float(request.speed) <= TTS_SPEED_MAX:
            raise bad_request(
                f"speed must be between {TTS_SPEED_MIN} and {TTS_SPEED_MAX}",
                param="speed",
            )

        if request.stream and response_format != "pcm":
            raise bad_request(
                "stream=true requires response_format='pcm'",
                param="response_format",
            )

        if request.task_type is not None:
            updates["task_type"] = _normalize_task_type(request.task_type)
        if request.language is not None:
            updates["language"] = _normalize_language(request.language)

        _validate_positive_int(request.max_new_tokens, param="max_new_tokens")
        _validate_positive_int(
            request.initial_codec_chunk_frames,
            param="initial_codec_chunk_frames",
        )
        _validate_non_negative_int(request.seed, param="seed")

        ref_audio = request.ref_audio
        if ref_audio is not None:
            updates["ref_audio"] = self._normalize_media_reference(
                ref_audio, param="ref_audio"
            )

        if request.references:
            updates["references"] = [
                self._normalize_speech_reference(reference)
                for reference in request.references
            ]

        return request.model_copy(update=updates)

    def build_generate_request(
        self,
        request: CreateSpeechRequest,
        *,
        validate: bool = True,
    ) -> GenerateRequest:
        """Convert a validated speech request into a client GenerateRequest."""

        if validate:
            request = self.prepare_request(request)
        explicit_generation_params = sorted(
            field
            for field in (
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
            )
            if field in request.model_fields_set
        )

        tts_params: dict[str, Any] = {
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        uploaded_voice = self._resolve_uploaded_voice_reference(request)
        if explicit_generation_params:
            tts_params["explicit_generation_params"] = explicit_generation_params
        if request.task_type is not None:
            tts_params["task_type"] = request.task_type
        if request.language is not None:
            tts_params["language"] = request.language
        if request.instructions is not None:
            tts_params["instructions"] = request.instructions
        if request.ref_audio is not None:
            tts_params["ref_audio"] = request.ref_audio
        if request.ref_text is not None:
            tts_params["ref_text"] = request.ref_text
        if uploaded_voice is not None:
            tts_params["task_type"] = "Base"
            tts_params["ref_audio"] = uploaded_voice.ref_audio
            if uploaded_voice.voice.ref_text is not None:
                tts_params["ref_text"] = uploaded_voice.voice.ref_text
            tts_params["uploaded_voice_name"] = uploaded_voice.voice.normalized_name
            tts_params["uploaded_voice_created_at"] = uploaded_voice.voice.created_at
            tts_params["uploaded_voice_fingerprint"] = uploaded_voice.voice.fingerprint
        if request.x_vector_only_mode is not None:
            tts_params["x_vector_only_mode"] = request.x_vector_only_mode
        if request.initial_codec_chunk_frames is not None:
            tts_params["initial_codec_chunk_frames"] = (
                request.initial_codec_chunk_frames
            )
        if request.token_count is not None:
            tts_params["token_count"] = request.token_count
        if request.duration_tokens is not None:
            tts_params["duration_tokens"] = request.duration_tokens
        if request.seed is not None:
            tts_params["seed"] = request.seed

        sampling = SamplingParams(
            temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
        )
        if request.max_new_tokens is not None:
            sampling.max_new_tokens = request.max_new_tokens
        if request.temperature is not None:
            sampling.temperature = request.temperature
        if request.top_p is not None:
            sampling.top_p = request.top_p
        if request.top_k is not None:
            sampling.top_k = request.top_k
        if request.repetition_penalty is not None:
            sampling.repetition_penalty = request.repetition_penalty
        if request.seed is not None:
            sampling.seed = request.seed

        prompt: Any = request.input
        references: list[dict[str, Any]] = []
        if request.references:
            references.extend(
                reference.model_dump(exclude_none=True)
                for reference in request.references
            )
        if request.ref_audio is not None:
            ref: dict[str, Any] = {"audio_path": request.ref_audio}
            if request.ref_text is not None:
                ref["text"] = request.ref_text
            references.append(ref)
        elif uploaded_voice is not None:
            ref = _uploaded_voice_reference_dict(uploaded_voice)
            if uploaded_voice.voice.ref_text is not None:
                ref["text"] = uploaded_voice.voice.ref_text
            references.append(ref)
        if references:
            prompt = {"text": request.input, "references": references}

        extra_params: dict[str, Any] = {}
        if request.initial_codec_chunk_frames is not None:
            extra_params["initial_codec_chunk_frames"] = (
                request.initial_codec_chunk_frames
            )

        return GenerateRequest(
            model=request.model or self.default_model,
            prompt=prompt,
            sampling=sampling,
            stage_params=request.stage_params,
            extra_params=extra_params,
            stream=request.stream,
            output_modalities=["audio"],
            metadata={
                "task": "tts",
                "tts_params": tts_params,
            },
        )

    async def create_speech_batch(
        self,
        client: "Client",
        batch: CreateSpeechBatchRequest,
        *,
        request_id: str,
    ) -> SpeechBatchResponse:
        """Run a batch speech request through the normal speech path."""

        results: list[SpeechBatchResult | None] = [None] * len(batch.items)
        tasks: list[asyncio.Task[SpeechBatchResult]] = []
        task_indexes: list[int] = []

        for index, item in enumerate(batch.items):
            try:
                request = self._build_batch_item_request(batch, item, index=index)
            except SpeechAPIError as exc:
                results[index] = _batch_error_result(index, exc)
                continue
            task = asyncio.create_task(
                self._run_batch_item(
                    client,
                    request,
                    request_id=f"{request_id}-{index}",
                    index=index,
                )
            )
            tasks.append(task)
            task_indexes.append(index)

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for index, task_result in zip(task_indexes, task_results, strict=True):
                if isinstance(task_result, SpeechBatchResult):
                    results[index] = task_result
                elif isinstance(task_result, SpeechAPIError):
                    results[index] = _batch_error_result(index, task_result)
                else:
                    results[index] = _batch_error_result(
                        index, internal_error(str(task_result))
                    )

        final_results = [result for result in results if result is not None]
        succeeded = sum(1 for result in final_results if result.success)
        failed = len(final_results) - succeeded
        return SpeechBatchResponse(
            id=request_id,
            results=final_results,
            total=len(final_results),
            succeeded=succeeded,
            failed=failed,
        )

    async def _run_batch_item(
        self,
        client: "Client",
        request: CreateSpeechRequest,
        *,
        request_id: str,
        index: int,
    ) -> SpeechBatchResult:
        gen_req = self.build_generate_request(request, validate=False)
        try:
            result = await client.speech(
                gen_req,
                request_id=request_id,
                response_format=request.response_format,
                speed=request.speed,
                allow_format_fallback=False,
            )
        except ClientError as exc:
            raise internal_error(str(exc)) from exc
        return SpeechBatchResult(
            index=index,
            status="success",
            success=True,
            audio_data=base64.b64encode(result.audio_bytes).decode("ascii"),
            format=result.format,
            media_type=result.mime_type,
        )

    def _build_batch_item_request(
        self,
        batch: CreateSpeechBatchRequest,
        item: SpeechBatchItem,
        *,
        index: int,
    ) -> CreateSpeechRequest:
        payload = batch.model_dump(exclude={"items"}, exclude_none=True)
        item_payload = item.model_dump(exclude_none=True)
        payload.update(item_payload)
        self._validate_raw_payload(payload)
        if item_payload.get("stream"):
            raise bad_request(
                "stream is not supported for batch speech requests",
                param=f"items.{index}.stream",
            )
        payload.pop("stream", None)
        try:
            request = CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        return self.prepare_request(request)

    def _validate_batch_defaults(self, batch: CreateSpeechBatchRequest) -> None:
        payload = batch.model_dump(exclude={"items"}, exclude_none=True)
        payload["input"] = "__batch_defaults_validation__"
        try:
            request = CreateSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise bad_request(_validation_error_message(exc)) from exc
        self.prepare_request(request)

    def _resolve_uploaded_voice_reference(
        self, request: CreateSpeechRequest
    ) -> UploadedVoiceReference | None:
        if (
            self.voice_store is None
            or request.ref_audio is not None
            or request.references
        ):
            return None
        if not request.voice or request.voice.lower() == "default":
            return None
        uploaded_voice = self.voice_store.resolve_reference(request.voice)
        if uploaded_voice is None and self.requires_uploaded_voice_for_named_voice:
            raise bad_request(
                f"Unknown voice '{request.voice}'. Upload a voice first via "
                "POST /v1/audio/voices, or use ref_audio + ref_text.",
                param="voice",
            )
        return uploaded_voice

    def _validate_raw_payload(self, payload: dict[str, Any]) -> None:
        for field_name in (
            "model",
            "input",
            "voice",
            "speaker",
            "response_format",
            "task_type",
            "language",
            "instructions",
            "ref_audio",
            "ref_text",
        ):
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], str):
                    raise bad_request(
                        f"{field_name} must be a string", param=field_name
                    )
        for field_name in ("max_new_tokens", "initial_codec_chunk_frames", "seed"):
            if field_name in payload and payload[field_name] is not None:
                value = payload[field_name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise bad_request(
                        f"{field_name} must be an integer", param=field_name
                    )
        if "top_k" in payload and payload["top_k"] is not None:
            value = payload["top_k"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise bad_request("top_k must be an integer", param="top_k")
        for field_name in ("speed", "temperature", "top_p", "repetition_penalty"):
            if field_name in payload and payload[field_name] is not None:
                value = payload[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise bad_request(
                        f"{field_name} must be a number", param=field_name
                    )
        for field_name in ("stream", "x_vector_only_mode"):
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], bool):
                    raise bad_request(
                        f"{field_name} must be a boolean", param=field_name
                    )

    def _normalize_speech_reference(
        self, reference: SpeechReference
    ) -> SpeechReference:
        updates: dict[str, Any] = {}
        for field_name in ("audio_path", "ref_audio", "audio"):
            value = getattr(reference, field_name)
            if isinstance(value, str):
                updates[field_name] = self._normalize_media_reference(
                    value, param=f"references.{field_name}"
                )
        return reference.model_copy(update=updates)

    def _normalize_media_reference(self, value: str, *, param: str) -> str:
        url = urlparse(value)
        if url.scheme in {"http", "https"}:
            return value
        if url.scheme == "data":
            if "," not in (url.path or ""):
                raise bad_request(
                    "ref_audio data URL must include base64 media data",
                    param=param,
                )
            return value
        if url.scheme != "file":
            raise bad_request(
                "ref_audio must be an http, https, data, or file:// URL",
                param=param,
            )
        if not self.allowed_local_media_paths:
            raise bad_request(
                "file:// ref_audio requires --allowed-local-media-path",
                param=param,
            )
        netloc = url.netloc or ""
        if netloc and netloc != "localhost":
            raise bad_request(
                f"file:// ref_audio netloc is not supported: {netloc}",
                param=param,
            )
        file_path = Path(url2pathname(url.path)).expanduser().resolve()
        for allowed_path in self.allowed_local_media_paths:
            if _is_relative_to(file_path, allowed_path):
                if not file_path.exists():
                    raise bad_request(
                        f"file:// ref_audio path does not exist: {file_path}",
                        param=param,
                    )
                if not file_path.is_file():
                    raise bad_request(
                        f"file:// ref_audio path must be a file: {file_path}",
                        param=param,
                    )
                return str(file_path)
        raise bad_request(
            f"file:// ref_audio path is outside allowed local media paths: {file_path}",
            param=param,
        )


def _normalize_response_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt not in SUPPORTED_TTS_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TTS_RESPONSE_FORMATS))
        raise bad_request(
            f"response_format must be one of: {supported}",
            param="response_format",
        )
    return fmt


def _uploaded_voice_reference_dict(
    uploaded_voice: "UploadedVoiceReference",
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "audio_path": uploaded_voice.ref_audio,
        "uploaded_voice_name": uploaded_voice.voice.normalized_name,
        "uploaded_voice_created_at": uploaded_voice.voice.created_at,
        "uploaded_voice_fingerprint": uploaded_voice.voice.fingerprint,
    }
    parsed = urlparse(uploaded_voice.ref_audio)
    if parsed.scheme == "data" and "," in parsed.path:
        header, data = parsed.path.split(",", 1)
        media_type = header.split(";", 1)[0] or "audio/wav"
        ref["media_type"] = media_type
        ref["data"] = data
    return ref


def _batch_error_result(index: int, error: SpeechAPIError) -> SpeechBatchResult:
    return SpeechBatchResult(
        index=index,
        status="error",
        success=False,
        error=openai_error_payload(
            error.message,
            error_type=error.error_type,
            param=error.param,
            code=error.code,
        )["error"],
    )


def speech_audio_delta(
    audio_data: Any,
    *,
    emitted_samples: int,
    is_terminal: bool,
) -> tuple[Any | None, int]:
    """Return the not-yet-emitted samples from a streaming TTS chunk."""

    audio = to_numpy(audio_data)
    if audio.ndim > 1:
        audio = audio.squeeze()
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[-1]:
            audio = audio[0]
        else:
            audio = audio[:, 0]

    total_samples = int(audio.shape[-1]) if audio.ndim else 0
    if not is_terminal:
        return audio, emitted_samples + total_samples
    if total_samples <= emitted_samples:
        return None, emitted_samples
    return audio[emitted_samples:], total_samples


def audio_format_from_mime_type(mime_type: str, default: str) -> str:
    for audio_format, known_mime_type in FORMAT_MIME_TYPES.items():
        if known_mime_type == mime_type:
            return audio_format
    return default


def _validate_positive_int(value: int | None, *, param: str) -> None:
    if value is not None and value <= 0:
        raise bad_request(f"{param} must be greater than 0", param=param)


def _validate_non_negative_int(value: int | None, *, param: str) -> None:
    if value is not None and value < 0:
        raise bad_request(f"{param} must be greater than or equal to 0", param=param)


def _validate_encoder_dependency(response_format: str) -> None:
    if response_format == "flac":
        _require_module("soundfile", "soundfile is required for response_format='flac'")
    elif response_format in {"mp3", "aac", "opus"}:
        _require_module(
            "pydub",
            f"pydub is required for response_format={response_format!r}",
        )
        if shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
            raise internal_error(
                "ffmpeg or avconv is required for "
                f"response_format={response_format!r}"
            )


def _require_module(module_name: str, message: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise internal_error(message)


def _normalize_language(value: str) -> str:
    normalized = _LANGUAGE_CANONICAL.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_TTS_LANGUAGES))
        raise bad_request(f"language must be one of: {supported}", param="language")
    return normalized


def _normalize_task_type(value: str) -> str:
    normalized = _TASK_TYPE_CANONICAL.get(
        value.strip().replace("_", "").replace("-", "").lower()
    )
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_TTS_TASK_TYPES))
        raise bad_request(f"task_type must be one of: {supported}", param="task_type")
    return normalized


def _validation_error_message(exc: ValidationError) -> str:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(item) for item in first_error.get("loc", ()))
    message = first_error.get("msg") or "invalid speech request"
    return f"{location}: {message}" if location else str(message)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_allowed_local_media_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"allowed local media path must be a directory: {path}")
    return resolved


def build_speech_generate_request(
    req: CreateSpeechRequest,
    default_model: str,
) -> GenerateRequest:
    """Compatibility wrapper for existing callers."""

    return SpeechService(default_model=default_model).build_generate_request(req)
