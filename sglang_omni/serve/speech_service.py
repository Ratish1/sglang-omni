# SPDX-License-Identifier: Apache-2.0
"""Shared service layer for TTS speech API requests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import ValidationError

from sglang_omni.client import GenerateRequest, SamplingParams
from sglang_omni.serve.protocol import (
    SUPPORTED_TTS_LANGUAGES,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SUPPORTED_TTS_TASK_TYPES,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    CreateSpeechRequest,
    SpeechReference,
)
from sglang_omni.serve.speech_errors import bad_request, internal_error

_GENERATION_FIELDS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "seed",
)
_OPTIONAL_INT_FIELDS = ("max_new_tokens", "initial_codec_chunk_frames")
_OPTIONAL_FLOAT_FIELDS = ("speed", "temperature", "top_p", "repetition_penalty")
_OPTIONAL_BOOL_FIELDS = ("stream", "x_vector_only_mode")
_STRING_FIELDS = (
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
        allowed_local_media_paths: list[str | Path] | None = None,
    ) -> None:
        self.default_model = default_model
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

        for field_name in _OPTIONAL_INT_FIELDS:
            value = getattr(request, field_name)
            if value is not None and int(value) <= 0:
                raise bad_request(
                    f"{field_name} must be greater than 0", param=field_name
                )

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

    def build_generate_request(self, request: CreateSpeechRequest) -> GenerateRequest:
        """Convert a validated speech request into a client GenerateRequest."""

        request = self.prepare_request(request)
        explicit_generation_params = sorted(
            field for field in _GENERATION_FIELDS if field in request.model_fields_set
        )

        tts_params: dict[str, Any] = {
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        if explicit_generation_params:
            tts_params["explicit_generation_params"] = explicit_generation_params
        _set_if_present(tts_params, "task_type", request.task_type)
        _set_if_present(tts_params, "language", request.language)
        _set_if_present(tts_params, "instructions", request.instructions)
        _set_if_present(tts_params, "ref_audio", request.ref_audio)
        _set_if_present(tts_params, "ref_text", request.ref_text)
        _set_if_present(tts_params, "x_vector_only_mode", request.x_vector_only_mode)
        _set_if_present(
            tts_params,
            "initial_codec_chunk_frames",
            request.initial_codec_chunk_frames,
        )
        _set_if_present(tts_params, "token_count", request.token_count)
        _set_if_present(tts_params, "duration_tokens", request.duration_tokens)
        _set_if_present(tts_params, "seed", request.seed)

        sampling = SamplingParams(
            temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
        )
        _set_if_present(sampling, "max_new_tokens", request.max_new_tokens)
        _set_if_present(sampling, "temperature", request.temperature)
        _set_if_present(sampling, "top_p", request.top_p)
        _set_if_present(sampling, "top_k", request.top_k)
        _set_if_present(sampling, "repetition_penalty", request.repetition_penalty)
        _set_if_present(sampling, "seed", request.seed)

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
        if references:
            prompt = {"text": request.input, "references": references}

        return GenerateRequest(
            model=request.model or self.default_model,
            prompt=prompt,
            sampling=sampling,
            stage_params=request.stage_params,
            stream=request.stream,
            output_modalities=["audio"],
            metadata={
                "task": "tts",
                "tts_params": tts_params,
            },
        )

    def _validate_raw_payload(self, payload: dict[str, Any]) -> None:
        for field_name in _STRING_FIELDS:
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], str):
                    raise bad_request(
                        f"{field_name} must be a string", param=field_name
                    )
        for field_name in _OPTIONAL_INT_FIELDS:
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
        for field_name in _OPTIONAL_FLOAT_FIELDS:
            if field_name in payload and payload[field_name] is not None:
                value = payload[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise bad_request(
                        f"{field_name} must be a number", param=field_name
                    )
        for field_name in _OPTIONAL_BOOL_FIELDS:
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
        if url.scheme != "file":
            return value
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


def _validate_encoder_dependency(response_format: str) -> None:
    if response_format == "flac":
        _require_module("soundfile", "soundfile is required for response_format='flac'")
    elif response_format in {"mp3", "aac", "opus"}:
        _require_module(
            "pydub",
            f"pydub is required for response_format={response_format!r}",
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


def _set_if_present(target: Any, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(target, dict):
        target[name] = value
        return
    setattr(target, name, value)


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
