# SPDX-License-Identifier: Apache-2.0
"""HTTP and SSE clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

from benchmarks.tts_serving.audio_validation import (
    EXPECTED_AUDIO_CONTENT_TYPES,
    PCM_CONTENT_TYPES,
    validate_audio_response,
    validate_pcm_chunk,
)
from benchmarks.tts_serving.error_contract import is_openai_error_response
from benchmarks.tts_serving.metrics import (
    PCM_SAMPLE_RATE,
    SSE_DONE_MARKER,
    ScenarioResult,
    classify_http_status,
    finish_timing,
    parse_sse_audio_event,
)
from benchmarks.tts_serving.scenarios import (
    VOICE_SMALL_UPLOAD_BYTES,
    VOICE_UPLOAD_SUCCESS_FORMATS,
    Scenario,
)
from benchmarks.tts_serving.spec import BenchmarkSpec
from benchmarks.tts_serving.urls import api_url
from benchmarks.tts_serving.voice_upload_fixtures import (
    get_near_limit_voice_upload_fixture,
    get_voice_upload_fixture,
    get_wav_upload_fixture,
)

AUDIO_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "flac", "aac", "opus"}
MIN_SPEECH_SPEED = 0.25
MAX_SPEECH_SPEED = 4.0
UNSUPPORTED_HTTP_STATUSES = {404, 405, 501}
SUCCESS_BATCH_STATUSES = {"ok", "success", "succeeded"}
FAILED_BATCH_STATUSES = {"error", "failed"}
VALID_BATCH_TASK_TYPES = {"Base", "CustomVoice", "VoiceDesign"}
EXPECTED_BATCH_MEDIA_TYPES = EXPECTED_AUDIO_CONTENT_TYPES
RawVoiceResponse = tuple[int, bytes, dict[str, str]]


@dataclass(frozen=True)
class BatchItemValidation:
    error: str | None = None
    audio_bytes: int = 0
    audio_duration_s: float = 0.0


async def run_http_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        expected_success=scenario.expect_success,
        response_format=_scenario_response_format(scenario),
        batch_size=scenario.planned_metadata.get("batch_size"),
    )
    url = api_url(spec.base_url, scenario.path)
    start = time.perf_counter()
    try:
        if scenario.method == "VOICE_LIFECYCLE":
            await _run_voice_lifecycle(
                session,
                spec,
                scenario,
                result,
            )
        elif scenario.method == "VOICE_OVERWRITE":
            await _run_voice_overwrite(session, spec, scenario, result)
        elif scenario.method == "VOICE_UPLOAD_DELETE_RACE":
            await _run_voice_upload_delete_race(session, spec, scenario, result)
        elif scenario.method == "VOICE_SPEAKER_CAP_SEQUENCE":
            await _run_voice_speaker_cap_sequence(session, spec, scenario, result)
        elif scenario.method == "VOICE_UPLOAD_METADATA_SEQUENCE":
            await _run_voice_upload_metadata_sequence(session, spec, scenario, result)
        elif scenario.method == "VOICE_CACHE_PRESSURE_SEQUENCE":
            await _run_voice_cache_pressure_sequence(session, spec, scenario, result)
        elif scenario.method == "GET":
            async with session.get(url) as response:
                await _handle_probe_response(response, result, scenario)
        elif scenario.method == "DELETE":
            async with session.delete(url) as response:
                await _handle_binary_response(response, result, scenario)
        else:
            body = _request_body(scenario)
            kwargs = (
                {"data": body}
                if scenario.body_type == "multipart"
                else {"json": scenario.payload}
            )
            result.request_bytes = _request_size(scenario)
            async with session.post(url, **kwargs) as response:
                if scenario.endpoint == "speech_sse":
                    await _handle_sse_response(
                        response,
                        result,
                        start,
                        scenario,
                    )
                else:
                    await _handle_binary_response(response, result, scenario)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "transport_error"
        result.error = str(exc)
    finally:
        finish_timing(result, start)
    return result


async def _handle_probe_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    body, body_text = await _response_body_and_text(response)
    result.response_bytes = len(body)
    if _is_unsupported_http_status(response.status, scenario):
        _mark_unsupported_contract(
            result,
            scenario,
            body=body_text,
        )
        return
    if 200 <= response.status < 300:
        if scenario.endpoint == "voices":
            _handle_voice_success(body, result, scenario)
            return
        _mark_success(result, capability="pass")
        return
    _classify_http_failure(response.status, body_text, result, scenario)


async def _handle_binary_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    body = await response.read()
    result.response_bytes = len(body)
    if _is_unsupported_http_status(response.status, scenario):
        _mark_unsupported_contract(
            result,
            scenario,
            body=body.decode("utf-8", errors="replace"),
        )
        return
    if 200 <= response.status < 300:
        if not scenario.expect_success:
            _mark_unexpected_success(result, scenario)
            return
        if scenario.endpoint == "batch":
            _handle_batch_success(body, result, scenario)
            return
        if scenario.endpoint == "voices":
            _handle_voice_success(body, result, scenario)
            return
        response_format = str(scenario.payload.get("response_format", ""))
        validation = validate_audio_response(
            body,
            response_format=response_format,
            content_type=response.headers.get("Content-Type"),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="invalid_audio_response",
                error=(
                    "speech endpoint returned 2xx without the requested audio "
                    f"contract (format={response_format!r}, "
                    f"content-type={response.headers.get('Content-Type')!r}, "
                    f"bytes={len(body)}, validation_error={validation.error})"
                ),
            )
            return
        result.audio_bytes = len(body)
        result.audio_duration_s = validation.duration_s
        _mark_success(result)
        return
    _classify_http_failure(
        response.status, body.decode("utf-8", errors="replace"), result, scenario
    )


async def _handle_sse_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    start: float,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    if response.status != 200:
        body, body_text = await _response_body_and_text(response)
        result.response_bytes = len(body)
        if _is_unsupported_http_status(response.status, scenario):
            _mark_unsupported_contract(result, scenario, body=body_text)
            return
        _classify_http_failure(response.status, body_text, result, scenario)
        return
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "text/event-stream" not in content_type:
        body = await response.read()
        result.response_bytes = len(body)
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE speech endpoint returned 2xx without text/event-stream "
                f"content-type: {response.headers.get('Content-Type')!r}"
            ),
        )
        return

    buffer = bytearray()
    audio_buffer = bytearray()
    chunk_times: list[float] = []
    saw_done = False
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw_line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            saw_done = (
                _merge_sse_line(
                    raw_line.decode("utf-8", errors="replace").strip(),
                    result,
                    start,
                    chunk_times,
                    scenario,
                    audio_buffer,
                )
                or saw_done
            )
            if _result_has_terminal_error(result):
                break
        if _result_has_terminal_error(result):
            break
    if not _result_has_terminal_error(result) and buffer.strip():
        saw_done = (
            _merge_sse_line(
                bytes(buffer).decode("utf-8", errors="replace").strip(),
                result,
                start,
                chunk_times,
                scenario,
                audio_buffer,
            )
            or saw_done
        )
    if _result_has_terminal_error(result):
        result.success = False
        return
    if not scenario.expect_success:
        _mark_unexpected_success(result, scenario)
        return
    if result.audio_bytes <= 0:
        _mark_protocol_error(
            result,
            status="empty_stream",
            error="SSE speech endpoint completed without audio bytes",
        )
        result.response_bytes = result.audio_bytes
        return
    if not saw_done:
        _mark_protocol_error(
            result,
            status="incomplete_sse_stream",
            error="SSE speech endpoint completed without terminal data: [DONE]",
        )
        result.response_bytes = result.audio_bytes
        return
    validation = validate_audio_response(
        bytes(audio_buffer),
        response_format="pcm",
        content_type="audio/pcm",
    )
    if not validation.ok:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE stream completed with invalid aggregate PCM: {validation.error}",
        )
        result.response_bytes = result.audio_bytes
        return
    result.audio_duration_s = validation.duration_s
    _mark_success(result)
    result.response_bytes = result.audio_bytes


async def _response_body_and_text(
    response: aiohttp.ClientResponse,
) -> tuple[bytes, str]:
    body = await response.read()
    return body, body.decode("utf-8", errors="replace")


def _merge_sse_line(
    line: str,
    result: ScenarioResult,
    start: float,
    chunk_times: list[float],
    scenario: Scenario,
    audio_buffer: bytearray,
) -> bool:
    if not line or line.startswith(":"):
        return False
    if line == SSE_DONE_MARKER:
        return True
    try:
        audio_bytes, event = parse_sse_audio_event(line)
    except (TypeError, ValueError, binascii.Error) as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "protocol_error"
        result.error = f"malformed SSE audio event: {exc}"
        return False
    if event is None:
        return False
    if audio_bytes is None:
        if _is_terminal_sse_json_event(event):
            return False
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE event did not include base64 audio payload: {event}",
        )
        return False
    audio = event.get("audio") if isinstance(event, dict) else None
    if not isinstance(audio, dict):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio event has invalid audio metadata: {event}",
        )
        return False
    audio_format = audio.get("format")
    expected_format = str(scenario.payload.get("response_format", ""))
    if not isinstance(audio_format, str) or audio_format != expected_format:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE audio.format must match requested response_format "
                f"(expected={expected_format!r}, observed={audio_format!r})"
            ),
        )
        return False
    mime_type = audio.get("mime_type")
    if mime_type is not None and not isinstance(mime_type, str):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio.mime_type must be a string when present: {event}",
        )
        return False
    normalized_mime_type = (
        (mime_type or "application/octet-stream").lower().split(";", 1)[0]
    )
    if audio_format != "pcm" or normalized_mime_type not in PCM_CONTENT_TYPES:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE stream=true audio chunks must be PCM "
                f"(format={audio_format!r}, mime_type={mime_type!r})"
            ),
        )
        return False
    sample_rate = audio.get("sample_rate", PCM_SAMPLE_RATE)
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=f"SSE audio.sample_rate must be a positive integer: {event}",
        )
        return False
    chunk_validation = validate_pcm_chunk(
        audio_bytes,
        sample_rate=sample_rate,
    )
    if not chunk_validation.ok:
        _mark_protocol_error(
            result,
            status="invalid_sse_response",
            error=(
                "SSE audio.data must decode to valid 16-bit PCM chunk "
                f"(decoded_bytes={len(audio_bytes)}, "
                f"validation_error={chunk_validation.error})"
            ),
        )
        return False
    now = time.perf_counter()
    if result.ttfa_s is None:
        result.ttfa_s = now - start
    elif chunk_times:
        result.inter_chunk_s.append(now - chunk_times[-1])
    chunk_times.append(now)
    audio_buffer.extend(audio_bytes)
    result.audio_bytes += len(audio_bytes)
    result.response_bytes += len(audio_bytes)
    result.audio_duration_s += chunk_validation.duration_s
    return False


def _is_terminal_sse_json_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("audio") is not None:
        return False
    finish_reason = event.get("finish_reason")
    return isinstance(finish_reason, str) and bool(finish_reason)


def _mark_unexpected_success(result: ScenarioResult, scenario: Scenario) -> None:
    result.success = False
    result.status = "unexpected_success"
    result.capability = "fail"
    result.error_class = "unexpected_success"
    result.error = (
        f"scenario {scenario.id} expected an error but received HTTP "
        f"{result.http_status}"
    )


def _result_has_terminal_error(result: ScenarioResult) -> bool:
    return result.error_class is not None or (
        result.status not in {"error", "ok"} and not result.success
    )


def _mark_success(result: ScenarioResult, *, capability: str | None = "pass") -> None:
    result.success = True
    result.status = "ok"
    if capability is not None:
        result.capability = capability


def _mark_unsupported_contract(
    result: ScenarioResult,
    scenario: Scenario,
    *,
    body: str,
    path: str | None = None,
) -> None:
    result.success = False
    result.status = "unsupported_contract"
    result.capability = "fail"
    result.error_class = "unsupported_endpoint"
    result.error = (
        "enabled benchmark contract is unsupported: "
        f"endpoint={scenario.endpoint}, operation={scenario.capability_key}, "
        f"path={path or scenario.path}, http_status={result.http_status}, body={body}"
    )


def _classify_http_failure(
    status: int,
    body: str,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.success = False
    result.error = body
    if scenario.capability_key == "voices.delete" and not scenario.expect_success:
        if _is_valid_missing_voice_delete_response(status, body):
            result.status = "expected_error"
            result.error_class = "expected_client_error"
            result.capability = "pass"
            return
        if _is_unsupported_http_status(status, scenario):
            _mark_unsupported_contract(result, scenario, body=body)
            return
        result.status = "invalid_voice_response"
        result.error_class = "protocol_error"
        result.capability = "fail"
        result.error = (
            "missing voice delete must return HTTP 404 JSON with "
            f"success=false and error details; received status={status}, body={body}"
        )
        return
    if 400 <= status < 500 and _is_expected_client_error_scenario(scenario):
        expected_status = _expected_client_error_status(scenario)
        if status != expected_status:
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "expected client-error scenario returned wrong HTTP status "
                    f"(expected={expected_status}, observed={status}): {body}"
                ),
            )
            return
        if not _is_valid_error_response(
            status,
            body,
            expected_status=expected_status,
        ):
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "expected client-error scenario returned HTTP "
                    f"{status} without OpenAI-compatible error JSON: {body}"
                ),
            )
            return
        result.status = "expected_error"
        result.error_class = "expected_client_error"
        result.capability = "pass"
        return
    if _is_unsupported_http_status(status, scenario):
        _mark_unsupported_contract(result, scenario, body=body)
        return
    if 500 <= status:
        result.status = "failed"
        result.error_class = "server_error"
        result.capability = "fail"
        return
    result.status = "failed"
    result.error_class = "http_error"
    if scenario.expect_success:
        result.capability = "fail"


def _is_unsupported_http_status(status: int, scenario: Scenario) -> bool:
    if status not in UNSUPPORTED_HTTP_STATUSES:
        return False
    if scenario.capability_key == "voices.delete" and not scenario.expect_success:
        return status != 404
    if _is_expected_client_error_scenario(scenario):
        return False
    return True


def _is_expected_client_error_scenario(scenario: Scenario) -> bool:
    return (
        not scenario.expect_success and scenario.expected_status_class == "client_error"
    )


def _expected_client_error_status(scenario: Scenario) -> int:
    return scenario.expected_http_status or 400


def _request_body(
    scenario: Scenario, *, form_fields: Mapping[str, str] | None = None
) -> aiohttp.FormData:
    return _voice_upload_body(
        scenario,
        form_fields=form_fields or scenario.form_fields,
        upload_format=str(scenario.planned_metadata.get("upload_format", "wav")),
        content_type=scenario.upload_content_type or "audio/wav",
        upload_size=scenario.upload_size_bytes,
        upload_case=str(scenario.planned_metadata.get("upload_case", "format")),
        filename=scenario.upload_filename or "audio.wav",
    )


def _voice_upload_body(
    scenario: Scenario,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    filename: str,
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for key, value in form_fields.items():
        form.add_field(key, value)
    if scenario.upload_field:
        form.add_field(
            scenario.upload_field,
            _synthetic_upload_bytes_for(
                upload_case=upload_case,
                upload_format=upload_format,
                upload_size=upload_size,
            ),
            filename=filename,
            content_type=content_type,
        )
    return form


def _synthetic_upload_bytes_for(
    *,
    upload_case: str,
    upload_format: str,
    upload_size: int,
) -> bytes:
    if upload_case == "corrupt_audio":
        return _pad_bytes(b"not-a-valid-audio-upload", upload_size)
    if upload_case in {"near_limit", "cache_eviction"}:
        return get_near_limit_voice_upload_fixture(upload_format, upload_size)
    if upload_case == "format":
        return get_voice_upload_fixture(upload_format)
    return _synthetic_audio_bytes(upload_size, upload_format)


def _synthetic_audio_bytes(size: int, upload_format: str = "wav") -> bytes:
    if size <= 0:
        return b""
    if upload_format == "mp3":
        return _pad_bytes(b"ID3", size)
    if upload_format == "flac":
        return _pad_bytes(b"fLaC", size)
    if upload_format == "ogg":
        return _pad_bytes(b"OggS", size)
    if upload_format == "aac":
        return _pad_bytes(b"\xff\xf1", size)
    if upload_format == "webm":
        return _pad_bytes(b"\x1a\x45\xdf\xa3", size)
    if upload_format == "mp4":
        return _pad_bytes(b"\x00\x00\x00\x18ftypmp42", size)
    return get_wav_upload_fixture(size)


def _pad_bytes(prefix: bytes, size: int) -> bytes:
    if size <= len(prefix):
        return prefix[:size]
    return prefix + (b"\0" * (size - len(prefix)))


def _request_size(scenario: Scenario) -> int:
    if scenario.body_type == "multipart":
        return _voice_request_size(scenario, form_fields=scenario.form_fields)
    try:
        return len(json.dumps(scenario.payload, ensure_ascii=False).encode("utf-8"))
    except TypeError:
        return 0


def _voice_request_size(
    scenario: Scenario,
    *,
    form_fields: Mapping[str, str],
) -> int:
    return _voice_request_size_for(
        form_fields=form_fields,
        upload_size=scenario.upload_size_bytes,
    )


def _voice_request_size_for(
    *,
    form_fields: Mapping[str, str],
    upload_size: int,
) -> int:
    return (
        sum(len(key) + len(value) for key, value in form_fields.items()) + upload_size
    )


def _voice_sequence_upload_size(upload_format: str) -> int:
    if upload_format == "wav":
        return VOICE_SMALL_UPLOAD_BYTES
    return len(get_voice_upload_fixture(upload_format))


def _voice_sequence_form_fields(
    scenario: Scenario,
    *,
    voice_name: str,
    ref_text: str,
    speaker_description: str,
) -> dict[str, str]:
    form_fields = dict(scenario.form_fields)
    form_fields["name"] = voice_name
    form_fields["ref_text"] = ref_text
    form_fields["speaker_description"] = speaker_description
    return form_fields


def _cache_revisit_voice_names(voice_names: list[str]) -> list[str]:
    if len(voice_names) <= 2:
        return voice_names
    midpoint = len(voice_names) // 2
    return [voice_names[0], voice_names[midpoint], voice_names[-1], voice_names[0]]


def _speaker_cap_form_fields(
    scenario: Scenario,
    voice_name: str,
) -> dict[str, str]:
    form_fields = dict(scenario.form_fields)
    form_fields["name"] = voice_name
    return form_fields


def _metadata_positive_int(scenario: Scenario, key: str) -> int | None:
    value = scenario.planned_metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


async def _run_voice_lifecycle(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    body = _request_body(scenario)
    result.request_bytes = _request_size(scenario)
    async with session.post(upload_url, data=body) as upload_response:
        result.http_status = upload_response.status
        result.http_status_class = classify_http_status(upload_response.status)
        result.response_headers = dict(upload_response.headers)
        upload_body = await upload_response.read()
        result.response_bytes += len(upload_body)
        if upload_response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=upload_body.decode("utf-8", errors="replace"),
            )
            return
        if not 200 <= upload_response.status < 300:
            _classify_http_failure(
                upload_response.status,
                upload_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return
        _handle_voice_success(upload_body, result, scenario)
        if not result.success:
            return

    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
    async with session.delete(delete_url) as delete_response:
        result.http_status = delete_response.status
        result.http_status_class = classify_http_status(delete_response.status)
        result.response_headers = dict(delete_response.headers)
        delete_body = await delete_response.read()
        result.response_bytes += len(delete_body)
        if not 200 <= delete_response.status < 300:
            _classify_http_failure(
                delete_response.status,
                delete_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return
        if not _is_valid_voice_delete_success(delete_body):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error="voice lifecycle delete response must be success JSON",
            )
            return
    _mark_success(result, capability="pass")


async def _run_voice_overwrite(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    result.request_bytes = _request_size(scenario) * 2

    before_description = "Synthetic benchmark voice before overwrite."
    after_description = "Synthetic benchmark voice after overwrite."
    first_fields = dict(scenario.form_fields)
    second_fields = dict(scenario.form_fields)
    first_fields["speaker_description"] = before_description
    second_fields["speaker_description"] = after_description

    first_payload = await _post_voice_upload(
        session,
        upload_url,
        scenario,
        result,
        form_fields=first_fields,
    )
    if first_payload is None:
        return
    if not _require_voice_upload_identifier(
        first_payload,
        result,
        error="first same-name voice upload response must include an identifier",
    ):
        return

    second_payload = await _post_voice_upload(
        session,
        upload_url,
        scenario,
        result,
        form_fields=second_fields,
    )
    if second_payload is None:
        return
    if not _require_voice_upload_identifier(
        second_payload,
        result,
        error="second same-name voice upload response must include an identifier",
    ):
        return
    if not _is_voice_overwrite_ack(second_payload):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "second same-name upload must include an overwrite warning or "
                f"replacement indicator: {second_payload}"
            ),
        )
        return

    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return
    entries = _uploaded_voice_entries(voice_list, voice_name)
    if not _validate_overwritten_voice_entry(
        entries,
        result,
        voice_name=voice_name,
        expected_speaker_description=after_description,
    ):
        return
    if not await _delete_voice_by_name(session, spec, scenario, result, voice_name):
        return
    _mark_success(result, capability="pass")


async def _run_voice_upload_delete_race(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    upload_url = api_url(spec.base_url, scenario.path)
    voice_name = str(scenario.planned_metadata.get("voice_name", ""))
    result.request_bytes = _request_size(scenario) * 2

    initial_payload = await _post_voice_upload(
        session,
        upload_url,
        scenario,
        result,
        form_fields=scenario.form_fields,
    )
    if initial_payload is None:
        return
    if not _require_voice_upload_identifier(
        initial_payload,
        result,
        error="initial race voice upload response must include an identifier",
    ):
        return

    race_upload_fields = dict(scenario.form_fields)
    race_upload_fields["speaker_description"] = (
        "Synthetic benchmark voice uploaded concurrently with delete."
    )
    upload_body = _request_body(scenario, form_fields=race_upload_fields)
    delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
    upload_response, delete_response = await asyncio.gather(
        _raw_post(session, upload_url, upload_body),
        _raw_delete(session, delete_url),
    )
    _merge_raw_voice_response(upload_response, result)
    _merge_raw_voice_response(delete_response, result)
    if not _classify_voice_race_response(
        upload_response,
        result,
        scenario,
        operation="concurrent voice upload",
        requires_voice_identifier=True,
    ):
        return
    if not _classify_voice_race_response(
        delete_response,
        result,
        scenario,
        operation="concurrent voice delete",
        requires_delete_success=True,
    ):
        return

    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return
    entries = _uploaded_voice_entries(voice_list, voice_name)
    if len(entries) > 1:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "same-name upload/delete race must not leave duplicate uploaded "
                f"voices named {voice_name!r}; observed={len(entries)}"
            ),
        )
        return
    if entries and not await _delete_voice_by_name(
        session, spec, scenario, result, voice_name
    ):
        return
    _mark_success(result, capability="pass")


async def _run_voice_speaker_cap_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    sequence_config = _speaker_cap_sequence_config(scenario, result)
    if sequence_config is None:
        return
    attempt_count, voice_name_prefix, speaker_max_uploaded = sequence_config

    upload_url = api_url(spec.base_url, scenario.path)
    created_voice_names: list[str] = []
    try:
        uploaded_voices = await _speaker_cap_uploaded_voices_after_stale_cleanup(
            session,
            spec,
            scenario,
            result,
            voice_name_prefix=voice_name_prefix,
        )
        if uploaded_voices is None:
            return

        remaining_before_cap = max(speaker_max_uploaded - len(uploaded_voices), 0)
        if not _speaker_cap_attempts_cross_cap(
            result,
            uploaded_count=len(uploaded_voices),
            attempt_count=attempt_count,
            remaining_before_cap=remaining_before_cap,
            speaker_max_uploaded=speaker_max_uploaded,
        ):
            return

        if not await _fill_speaker_cap(
            session,
            upload_url,
            scenario,
            result,
            voice_name_prefix=voice_name_prefix,
            upload_count=remaining_before_cap,
            created_voice_names=created_voice_names,
        ):
            return
        overflow_name = f"{voice_name_prefix}_overflow"
        created_voice_names.append(overflow_name)
        if not await _expect_speaker_cap_rejection(
            session,
            upload_url,
            scenario,
            result,
            voice_name=overflow_name,
            speaker_max_uploaded=speaker_max_uploaded,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(
            session,
            spec,
            created_voice_names,
        )
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def _run_voice_upload_metadata_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="voice metadata sequence requires voice_name_prefix",
        )
        return

    upload_url = api_url(spec.base_url, scenario.path)
    created_voice_names: list[str] = []
    expected_entries: dict[str, dict[str, str]] = {}
    try:
        for upload_format, content_type in VOICE_UPLOAD_SUCCESS_FORMATS:
            voice_name = f"{voice_name_prefix}_{upload_format}"
            fields = _voice_sequence_form_fields(
                scenario,
                voice_name=voice_name,
                ref_text=f"Voice metadata reference text for {upload_format}.",
                speaker_description=(
                    f"Synthetic metadata sequence voice in {upload_format} format."
                ),
            )
            created_voice_names.append(voice_name)
            expected_entries[voice_name] = {
                "ref_text": fields["ref_text"],
                "speaker_description": fields["speaker_description"],
            }
            if not await _post_expected_voice_upload(
                session,
                upload_url,
                scenario,
                result,
                form_fields=fields,
                upload_format=upload_format,
                content_type=content_type,
                upload_size=_voice_sequence_upload_size(upload_format),
                upload_case="format",
                operation=f"metadata upload {upload_format}",
            ):
                return

        voice_list = await _get_voice_list(session, spec, scenario, result)
        if voice_list is None:
            return
        if not _validate_uploaded_voice_metadata_sequence(
            voice_list,
            expected_entries,
            result,
        ):
            return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


async def _run_voice_cache_pressure_sequence(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> None:
    voice_count = _metadata_positive_int(scenario, "voice_count")
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if voice_count is None or not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="voice cache pressure sequence requires voice_count and voice_name_prefix",
        )
        return

    upload_url = api_url(spec.base_url, scenario.path)
    created_voice_names: list[str] = []
    try:
        for voice_index in range(voice_count):
            voice_name = f"{voice_name_prefix}_{voice_index:04d}"
            fields = _voice_sequence_form_fields(
                scenario,
                voice_name=voice_name,
                ref_text=f"Voice cache pressure reference text {voice_index}.",
                speaker_description=(
                    f"Synthetic cache pressure voice number {voice_index}."
                ),
            )
            created_voice_names.append(voice_name)
            if not await _post_expected_voice_upload(
                session,
                upload_url,
                scenario,
                result,
                form_fields=fields,
                upload_format="wav",
                content_type="audio/wav",
                upload_size=VOICE_SMALL_UPLOAD_BYTES,
                upload_case="format",
                operation=f"cache pressure upload {voice_index}",
            ):
                return
            if not await _post_speech_with_uploaded_voice(
                session,
                spec,
                scenario,
                result,
                voice_name=voice_name,
                prompt=f"Cache pressure synthesis request {voice_index}.",
            ):
                return

        for voice_name in _cache_revisit_voice_names(created_voice_names):
            if not await _post_speech_with_uploaded_voice(
                session,
                spec,
                scenario,
                result,
                voice_name=voice_name,
                prompt=f"Cache revisit synthesis request for {voice_name}.",
            ):
                return
        _mark_success(result, capability="pass")
    finally:
        cleanup_error = await _cleanup_voice_names(session, spec, created_voice_names)
        if cleanup_error is not None:
            _mark_cleanup_error_if_primary_path_passed(result, cleanup_error)


def _speaker_cap_sequence_config(
    scenario: Scenario,
    result: ScenarioResult,
) -> tuple[int, str, int] | None:
    attempt_count = _metadata_positive_int(scenario, "attempt_count")
    if attempt_count is None:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires positive integer attempt_count",
        )
        return None
    voice_name_prefix = str(scenario.planned_metadata.get("voice_name_prefix", ""))
    if not voice_name_prefix:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires voice_name_prefix",
        )
        return None
    speaker_max_uploaded = _metadata_positive_int(scenario, "speaker_max_uploaded")
    if speaker_max_uploaded is None:
        _mark_protocol_error(
            result,
            status="invalid_benchmark_scenario",
            error="speaker cap sequence requires positive integer speaker_max_uploaded",
        )
        return None
    return attempt_count, voice_name_prefix, speaker_max_uploaded


async def _speaker_cap_uploaded_voices_after_stale_cleanup(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
) -> list[dict[str, Any]] | None:
    uploaded_voices = await _get_uploaded_voices(session, spec, scenario, result)
    if uploaded_voices is None:
        return None

    stale_voice_names = _uploaded_voice_names_with_prefix(
        uploaded_voices,
        voice_name_prefix,
    )
    cleanup_error = await _cleanup_voice_names(session, spec, stale_voice_names)
    if cleanup_error is not None:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=cleanup_error,
        )
        return None
    if not stale_voice_names:
        return uploaded_voices
    return await _get_uploaded_voices(session, spec, scenario, result)


def _mark_cleanup_error_if_primary_path_passed(
    result: ScenarioResult,
    cleanup_error: str,
) -> None:
    if result.error_class is None:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=cleanup_error,
        )


def _speaker_cap_attempts_cross_cap(
    result: ScenarioResult,
    *,
    uploaded_count: int,
    attempt_count: int,
    remaining_before_cap: int,
    speaker_max_uploaded: int,
) -> bool:
    if attempt_count > remaining_before_cap:
        return True
    _mark_protocol_error(
        result,
        status="invalid_benchmark_scenario",
        error=(
            "speaker cap sequence did not include enough upload attempts to "
            f"cross the cap (uploaded={uploaded_count}, cap={speaker_max_uploaded}, "
            f"attempts={attempt_count})"
        ),
    )
    return False


async def _fill_speaker_cap(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name_prefix: str,
    upload_count: int,
    created_voice_names: list[str],
) -> bool:
    for cap_index in range(upload_count):
        voice_name = f"{voice_name_prefix}_{cap_index:04d}"
        created_voice_names.append(voice_name)
        if not await _upload_expected_speaker_cap_voice(
            session,
            upload_url,
            scenario,
            result,
            voice_name=voice_name,
        ):
            return False
    return True


async def _upload_expected_speaker_cap_voice(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
) -> bool:
    form_fields = _speaker_cap_form_fields(scenario, voice_name)
    result.request_bytes += _voice_request_size(scenario, form_fields=form_fields)
    payload = await _post_voice_upload(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
    )
    if payload is None:
        return False
    return _require_voice_upload_identifier(
        payload,
        result,
        error=f"speaker cap upload response must include an identifier for {voice_name!r}",
    )


async def _expect_speaker_cap_rejection(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
    speaker_max_uploaded: int,
) -> bool:
    form_fields = _speaker_cap_form_fields(scenario, voice_name)
    body = _request_body(scenario, form_fields=form_fields)
    result.request_bytes += _voice_request_size(scenario, form_fields=form_fields)
    status, response_body, headers = await _raw_post(session, upload_url, body)
    result.http_status = status
    result.http_status_class = classify_http_status(status)
    result.response_headers = headers
    result.response_bytes += len(response_body)
    body_text = response_body.decode("utf-8", errors="replace")

    if status in UNSUPPORTED_HTTP_STATUSES:
        _mark_unsupported_contract(result, scenario, body=body_text)
        return False
    if 200 <= status < 300:
        _mark_protocol_error(
            result,
            status="unexpected_success",
            error=(
                "speaker cap overflow upload unexpectedly succeeded after "
                f"{speaker_max_uploaded} uploaded voices"
            ),
        )
        result.error_class = "unexpected_success"
        return False
    if status == 400:
        if not _is_valid_error_response(status, body_text, expected_status=400):
            _mark_protocol_error(
                result,
                status="invalid_error_response",
                error=(
                    "speaker cap overflow returned HTTP "
                    f"{status} without structured error JSON: {body_text}"
                ),
            )
            return False
        return True
    if 400 <= status < 500:
        _mark_protocol_error(
            result,
            status="invalid_error_response",
            error=(
                "speaker cap overflow must return HTTP 400 with structured error "
                f"JSON, got HTTP {status}: {body_text}"
            ),
        )
        return False

    result.status = "failed"
    result.success = False
    result.error_class = "server_error" if status >= 500 else "http_error"
    result.capability = "fail"
    result.error = body_text
    return False


async def _post_voice_upload(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
) -> dict[str, Any] | None:
    return await _post_voice_upload_audio(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
        upload_format=str(scenario.planned_metadata.get("upload_format", "wav")),
        content_type=scenario.upload_content_type or "audio/wav",
        upload_size=scenario.upload_size_bytes,
        upload_case=str(scenario.planned_metadata.get("upload_case", "format")),
        filename=scenario.upload_filename or "audio.wav",
    )


async def _post_expected_voice_upload(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    operation: str,
) -> bool:
    result.request_bytes += _voice_request_size_for(
        form_fields=form_fields,
        upload_size=upload_size,
    )
    payload = await _post_voice_upload_audio(
        session,
        upload_url,
        scenario,
        result,
        form_fields=form_fields,
        upload_format=upload_format,
        content_type=content_type,
        upload_size=upload_size,
        upload_case=upload_case,
        filename=f"{form_fields.get('name', 'voice')}.{upload_format}",
    )
    if payload is None:
        return False
    return _require_voice_upload_identifier(
        payload,
        result,
        error=f"{operation} response must include an identifier",
    )


async def _post_voice_upload_audio(
    session: aiohttp.ClientSession,
    upload_url: str,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    form_fields: Mapping[str, str],
    upload_format: str,
    content_type: str,
    upload_size: int,
    upload_case: str,
    filename: str,
) -> dict[str, Any] | None:
    body = _voice_upload_body(
        scenario,
        form_fields=form_fields,
        upload_format=upload_format,
        content_type=content_type,
        upload_size=upload_size,
        upload_case=upload_case,
        filename=filename,
    )
    async with session.post(upload_url, data=body) as response:
        result.http_status = response.status
        result.http_status_class = classify_http_status(response.status)
        result.response_headers = dict(response.headers)
        response_body = await response.read()
        result.response_bytes += len(response_body)
        if response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=response_body.decode("utf-8", errors="replace"),
            )
            return None
        if not 200 <= response.status < 300:
            _classify_http_failure(
                response.status,
                response_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return None
    return _json_object_from_bytes(
        response_body,
        result,
        status="invalid_voice_response",
        error_prefix="voice upload response returned invalid JSON",
    )


async def _post_speech_with_uploaded_voice(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    *,
    voice_name: str,
    prompt: str,
) -> bool:
    payload = {
        "model": spec.model_name,
        "input": prompt,
        "voice": voice_name,
        "response_format": "pcm",
        "speed": 1.0,
    }
    result.request_bytes += len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    speech_url = api_url(spec.base_url, "/v1/audio/speech")
    async with session.post(speech_url, json=payload) as response:
        result.http_status = response.status
        result.http_status_class = classify_http_status(response.status)
        result.response_headers = dict(response.headers)
        body = await response.read()
        result.response_bytes += len(body)
        body_text = body.decode("utf-8", errors="replace")
        if response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=body_text,
                path="/v1/audio/speech",
            )
            return False
        if not 200 <= response.status < 300:
            _classify_http_failure(response.status, body_text, result, scenario)
            return False
        validation = validate_audio_response(
            body,
            response_format="pcm",
            content_type=response.headers.get("Content-Type"),
        )
        if not validation.ok:
            _mark_protocol_error(
                result,
                status="invalid_audio_response",
                error=(
                    "speech endpoint returned 2xx without PCM audio while using "
                    f"uploaded voice {voice_name!r}: {validation.error}"
                ),
            )
            return False
        result.audio_bytes += len(body)
        result.audio_duration_s += validation.duration_s
        return True


async def _raw_post(
    session: aiohttp.ClientSession,
    url: str,
    body: aiohttp.FormData,
) -> RawVoiceResponse:
    async with session.post(url, data=body) as response:
        return response.status, await response.read(), dict(response.headers)


async def _raw_delete(
    session: aiohttp.ClientSession,
    url: str,
) -> RawVoiceResponse:
    async with session.delete(url) as response:
        return response.status, await response.read(), dict(response.headers)


def _merge_raw_voice_response(
    response: RawVoiceResponse,
    result: ScenarioResult,
) -> None:
    status, body, headers = response
    result.http_status = status
    result.http_status_class = classify_http_status(status)
    result.response_headers = headers
    result.response_bytes += len(body)


def _classify_voice_race_response(
    response: RawVoiceResponse,
    result: ScenarioResult,
    scenario: Scenario,
    *,
    operation: str,
    requires_voice_identifier: bool = False,
    requires_delete_success: bool = False,
) -> bool:
    status, body, _ = response
    body_text = body.decode("utf-8", errors="replace")
    if status in UNSUPPORTED_HTTP_STATUSES:
        _mark_unsupported_contract(result, scenario, body=body_text)
        return False
    if not 200 <= status < 300:
        _classify_http_failure(status, body_text, result, scenario)
        if result.error_class == "http_error":
            result.error = f"{operation} failed: {body_text}"
        return False
    if requires_voice_identifier:
        payload = _json_object_from_bytes(
            body,
            result,
            status="invalid_voice_response",
            error_prefix=f"{operation} response returned invalid JSON",
        )
        if payload is None:
            return False
        if not _voice_upload_response_identifier(payload):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=f"{operation} response must include an identifier",
            )
            return False
    if requires_delete_success and not _is_valid_voice_delete_success(body):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"{operation} response must be success JSON",
        )
        return False
    return True


def _require_voice_upload_identifier(
    payload: dict[str, Any],
    result: ScenarioResult,
    *,
    error: str,
) -> bool:
    if _voice_upload_response_identifier(payload):
        return True
    _mark_protocol_error(
        result,
        status="invalid_voice_response",
        error=error,
    )
    return False


def _validate_overwritten_voice_entry(
    entries: list[dict],
    result: ScenarioResult,
    *,
    voice_name: str,
    expected_speaker_description: str,
) -> bool:
    if len(entries) != 1:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "same-name voice overwrite must leave exactly one uploaded "
                f"voice named {voice_name!r}; observed={len(entries)}"
            ),
        )
        return False
    if entries[0].get("speaker_description") != expected_speaker_description:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "same-name voice overwrite must expose the second upload metadata "
                f"for {voice_name!r}"
            ),
        )
        return False
    return True


def _validate_uploaded_voice_metadata_sequence(
    payload: dict[str, Any],
    expected_entries: Mapping[str, Mapping[str, str]],
    result: ScenarioResult,
) -> bool:
    if not _is_valid_voice_list_response(payload):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice metadata sequence requires voice list response with "
                "valid preset and uploaded voice metadata"
            ),
        )
        return False
    for voice_name, expected_fields in expected_entries.items():
        entries = _uploaded_voice_entries(payload, voice_name)
        if len(entries) != 1:
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error=(
                    "voice metadata sequence must expose exactly one uploaded "
                    f"voice named {voice_name!r}; observed={len(entries)}"
                ),
            )
            return False
        entry = entries[0]
        for key, expected_value in expected_fields.items():
            if entry.get(key) != expected_value:
                _mark_protocol_error(
                    result,
                    status="invalid_voice_response",
                    error=(
                        "voice metadata sequence did not preserve "
                        f"{key} for {voice_name!r}"
                    ),
                )
                return False
    return True


async def _get_voice_list(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> dict[str, Any] | None:
    list_url = api_url(spec.base_url, "/v1/audio/voices")
    async with session.get(list_url) as list_response:
        result.http_status = list_response.status
        result.http_status_class = classify_http_status(list_response.status)
        result.response_headers = dict(list_response.headers)
        list_body = await list_response.read()
        result.response_bytes += len(list_body)
        if list_response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=list_body.decode("utf-8", errors="replace"),
            )
            return None
        if not 200 <= list_response.status < 300:
            _classify_http_failure(
                list_response.status,
                list_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return None
    return _json_object_from_bytes(
        list_body,
        result,
        status="invalid_voice_response",
        error_prefix="voice list response returned invalid JSON",
    )


async def _get_uploaded_voices(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
) -> list[dict[str, Any]] | None:
    voice_list = await _get_voice_list(session, spec, scenario, result)
    if voice_list is None:
        return None
    if not _is_valid_voice_list_response(voice_list):
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice list response must be an object with voices and "
                "uploaded_voices before speaker cap validation"
            ),
        )
        return None
    uploaded_voices = voice_list["uploaded_voices"]
    return [voice for voice in uploaded_voices if isinstance(voice, dict)]


async def _delete_voice_by_name(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
    result: ScenarioResult,
    voice_name: str,
) -> bool:
    delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
    async with session.delete(delete_url) as delete_response:
        result.http_status = delete_response.status
        result.http_status_class = classify_http_status(delete_response.status)
        result.response_headers = dict(delete_response.headers)
        delete_body = await delete_response.read()
        result.response_bytes += len(delete_body)
        if delete_response.status in UNSUPPORTED_HTTP_STATUSES:
            _mark_unsupported_contract(
                result,
                scenario,
                body=delete_body.decode("utf-8", errors="replace"),
            )
            return False
        if not 200 <= delete_response.status < 300:
            _classify_http_failure(
                delete_response.status,
                delete_body.decode("utf-8", errors="replace"),
                result,
                scenario,
            )
            return False
        if not _is_valid_voice_delete_success(delete_body):
            _mark_protocol_error(
                result,
                status="invalid_voice_response",
                error="voice cleanup delete response must be success JSON",
            )
            return False
    return True


async def _cleanup_voice_names(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    voice_names: list[str],
) -> str | None:
    for voice_name in reversed(voice_names):
        delete_url = api_url(spec.base_url, f"/v1/audio/voices/{voice_name}")
        try:
            status, body, _ = await _raw_delete(session, delete_url)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return f"voice cleanup failed for {voice_name!r}: {exc}"
        body_text = body.decode("utf-8", errors="replace")
        if status == 404:
            continue
        if not 200 <= status < 300 or not _is_valid_voice_delete_success(body):
            return (
                f"voice cleanup failed for {voice_name!r}: "
                f"status={status}, body={body_text}"
            )
    return None


def _scenario_response_format(scenario: Scenario) -> str | None:
    response_format = scenario.planned_metadata.get("response_format")
    if response_format is None:
        response_format = scenario.payload.get("response_format")
    return str(response_format) if response_format is not None else None


def _handle_batch_success(
    body: bytes, result: ScenarioResult, scenario: Scenario
) -> None:
    payload = _json_from_bytes(
        body,
        result,
        status="invalid_batch_response",
        error_prefix="batch endpoint returned invalid JSON",
        default_empty={},
    )
    if payload is None:
        return
    required_keys = {"id", "results", "total", "succeeded", "failed"}
    if not isinstance(payload, dict) or not required_keys <= set(payload):
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint returned JSON without id/results/total/succeeded/failed",
        )
        return
    batch_size = int(scenario.planned_metadata.get("batch_size") or 0)
    results = payload.get("results")
    total = payload.get("total")
    succeeded = payload.get("succeeded")
    failed = payload.get("failed")
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint id must be a non-empty string",
        )
        return
    if (
        not isinstance(results, list)
        or not isinstance(total, int)
        or not isinstance(succeeded, int)
        or not isinstance(failed, int)
    ):
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint returned non-integer counts or non-list results",
        )
        return
    if total != batch_size or len(results) != batch_size:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint result count mismatch "
                f"(expected={batch_size}, total={total}, results={len(results)})"
            ),
        )
        return
    if succeeded + failed != total:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint succeeded + failed does not equal total",
        )
        return
    expected_item_failures = _expected_batch_item_failures(scenario)
    expected_failed = len(expected_item_failures)
    if failed != expected_failed or succeeded != total - expected_failed:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint did not report the exact expected item-level "
                f"outcome counts (expected_succeeded={total - expected_failed}, "
                f"succeeded={succeeded}, expected_failed={expected_failed}, "
                f"failed={failed})"
            ),
        )
        return
    observed_success = 0
    observed_failed = 0
    successful_audio_bytes = 0
    successful_audio_duration_s = 0.0
    for index, item in enumerate(results):
        expect_item_failure = index in expected_item_failures
        expected_format = _expected_batch_response_format(scenario, index)
        validation_error = _validate_batch_item(
            item,
            expected_index=index,
            expected_format=expected_format,
            expect_failure=expect_item_failure,
        )
        if validation_error.error is not None:
            _mark_protocol_error(
                result,
                status="invalid_batch_response",
                error=f"batch endpoint result item {index}: {validation_error.error}",
            )
            return
        if expect_item_failure:
            observed_failed += 1
        else:
            observed_success += 1
            successful_audio_bytes += validation_error.audio_bytes
            successful_audio_duration_s += validation_error.audio_duration_s
    if observed_success != succeeded or observed_failed != failed:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint item statuses do not match top-level counts "
                f"(item_success={observed_success}, succeeded={succeeded}, "
                f"item_failed={observed_failed}, failed={failed})"
            ),
        )
        return
    result.audio_bytes += successful_audio_bytes
    result.audio_duration_s += successful_audio_duration_s
    _mark_success(result, capability="pass")


def _expected_batch_item_failures(scenario: Scenario) -> set[int]:
    items = scenario.payload.get("items", [])
    if not isinstance(items, list):
        return set()
    return {
        index
        for index, item in enumerate(items)
        if _is_expected_batch_item_failure(item)
    }


def _is_expected_batch_item_failure(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    input_text = item.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        return True
    response_format = item.get("response_format")
    if response_format is not None and response_format not in AUDIO_RESPONSE_FORMATS:
        return True
    task_type = item.get("task_type")
    if task_type is not None and task_type not in VALID_BATCH_TASK_TYPES:
        return True
    speed = item.get("speed")
    return speed is not None and (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or speed < MIN_SPEECH_SPEED
        or speed > MAX_SPEECH_SPEED
    )


def _expected_batch_response_format(scenario: Scenario, index: int) -> str:
    items = scenario.payload.get("items", [])
    item_format = None
    if (
        isinstance(items, list)
        and index < len(items)
        and isinstance(items[index], dict)
    ):
        item_format = items[index].get("response_format")
    return str(item_format or scenario.payload.get("response_format") or "wav")


def _handle_voice_success(
    body: bytes, result: ScenarioResult, scenario: Scenario
) -> None:
    payload = _json_from_bytes(
        body,
        result,
        status="invalid_voice_response",
        error_prefix="voice endpoint returned invalid JSON",
        default_empty={},
    )
    if payload is None:
        return
    if scenario.capability_key == "voices.list":
        if _is_valid_voice_list_response(payload):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice list response must be an object with voices and "
                "uploaded_voices; uploaded entries require name, consent, "
                "created_at, file_size, and mime_type"
            ),
        )
        return
    if scenario.capability_key in {"voices.upload", "voices.lifecycle"}:
        if _voice_upload_response_identifier(payload):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error="voice upload response must include id, voice_id, or name",
        )
        return
    _mark_success(result, capability="pass")


def _validate_batch_item(
    item: Any,
    *,
    expected_index: int,
    expected_format: str,
    expect_failure: bool,
) -> BatchItemValidation:
    if not isinstance(item, dict):
        return BatchItemValidation(error="result item must be a JSON object")
    if item.get("index") != expected_index:
        return BatchItemValidation(
            error=(
                "index mismatch "
                f"(expected={expected_index}, observed={item.get('index')})"
            )
        )
    status = item.get("status")
    if expect_failure:
        if status not in FAILED_BATCH_STATUSES:
            return BatchItemValidation(
                error=f"expected failed status, observed={status!r}"
            )
        error = item.get("error")
        if not isinstance(error, (dict, str)) or not error:
            return BatchItemValidation(
                error="failed item must include non-empty error details"
            )
        return BatchItemValidation()
    if status not in SUCCESS_BATCH_STATUSES:
        return BatchItemValidation(
            error=f"expected success status, observed={status!r}"
        )
    audio_data = item.get("audio_data")
    media_type = item.get("media_type")
    if not isinstance(audio_data, str) or not audio_data:
        return BatchItemValidation(
            error="successful item must include non-empty base64 audio_data"
        )
    if not isinstance(media_type, str) or not _is_valid_batch_media_type(
        media_type, expected_format=expected_format
    ):
        return BatchItemValidation(
            error=(
                "successful item media_type does not match requested format "
                f"(format={expected_format!r}, media_type={media_type!r})"
            )
        )
    try:
        decoded = base64.b64decode(audio_data, validate=True)
    except binascii.Error as exc:
        return BatchItemValidation(
            error=f"successful item audio_data is not valid base64: {exc}"
        )
    if not decoded:
        return BatchItemValidation(
            error="successful item audio_data decoded to empty bytes"
        )
    validation = validate_audio_response(
        decoded,
        response_format=expected_format,
        content_type=media_type,
    )
    if not validation.ok:
        return BatchItemValidation(
            error=(
                "successful item audio_data does not match requested audio "
                f"contract (format={expected_format!r}, media_type={media_type!r}, "
                f"decoded_bytes={len(decoded)}, validation_error={validation.error})"
            )
        )
    return BatchItemValidation(
        audio_bytes=len(decoded),
        audio_duration_s=validation.duration_s,
    )


def _is_valid_voice_list_response(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    voices = payload.get("voices")
    uploaded_voices = payload.get("uploaded_voices")
    if not isinstance(voices, list) or not isinstance(uploaded_voices, list):
        return False
    return all(_is_valid_preset_voice(item) for item in voices) and all(
        _is_valid_uploaded_voice_metadata(item) for item in uploaded_voices
    )


def _json_object_from_bytes(
    body: bytes,
    result: ScenarioResult,
    *,
    status: str,
    error_prefix: str,
) -> dict[str, Any] | None:
    payload = _json_from_bytes(
        body,
        result,
        status=status,
        error_prefix=error_prefix,
        default_empty={},
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        _mark_protocol_error(
            result,
            status=status,
            error=f"{error_prefix}: response must be a JSON object",
        )
        return None
    return payload


def _json_from_bytes(
    body: bytes,
    result: ScenarioResult,
    *,
    status: str,
    error_prefix: str,
    default_empty: Any = None,
) -> Any | None:
    try:
        return (
            json.loads(body.decode("utf-8", errors="replace"))
            if body
            else default_empty
        )
    except json.JSONDecodeError as exc:
        _mark_protocol_error(
            result,
            status=status,
            error=f"{error_prefix}: {exc}",
        )
        result.error_type = exc.__class__.__name__
        return None


def _is_valid_preset_voice(item: Any) -> bool:
    if isinstance(item, str):
        return bool(item)
    return isinstance(item, dict) and any(
        isinstance(item.get(key), str) and item[key] for key in ("name", "voice", "id")
    )


def _is_valid_uploaded_voice_metadata(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required = ("name", "consent", "created_at", "file_size", "mime_type")
    return (
        all(key in item for key in required)
        and isinstance(item["name"], str)
        and bool(item["name"])
        and isinstance(item["consent"], (bool, str))
        and bool(str(item["consent"]))
        and isinstance(item["created_at"], (str, int, float))
        and bool(str(item["created_at"]))
        and _is_nonnegative_file_size(item["file_size"])
        and isinstance(item["mime_type"], str)
        and bool(item["mime_type"])
    )


def _voice_upload_response_identifier(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("success") is False:
        return None
    for key in ("id", "voice_id", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    voice = payload.get("voice")
    if isinstance(voice, str) and voice:
        return voice
    if isinstance(voice, dict):
        for key in ("id", "voice_id", "name"):
            value = voice.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _is_voice_overwrite_ack(payload: dict[str, Any]) -> bool:
    if payload.get("overwritten") is True or payload.get("replaced") is True:
        return True
    for key in ("warning", "message"):
        value = payload.get(key)
        if isinstance(value, str) and "overwrit" in value.lower():
            return True
    return False


def _uploaded_voice_entries(payload: dict[str, Any], voice_name: str) -> list[dict]:
    uploaded_voices = payload.get("uploaded_voices")
    if not isinstance(uploaded_voices, list):
        return []
    return [
        item
        for item in uploaded_voices
        if isinstance(item, dict) and item.get("name") == voice_name
    ]


def _uploaded_voice_names_with_prefix(
    uploaded_voices: list[dict[str, Any]],
    voice_name_prefix: str,
) -> list[str]:
    names: list[str] = []
    for voice in uploaded_voices:
        name = voice.get("name")
        if isinstance(name, str) and name.startswith(voice_name_prefix):
            names.append(name)
    return names


def _is_valid_voice_delete_success(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace")) if body else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return False
    if payload.get("deleted") is True:
        return True
    return payload.get("success") is True or any(
        isinstance(payload.get(key), str) and payload[key]
        for key in ("id", "voice_id", "name")
    )


def _is_valid_missing_voice_delete_response(status: int, body: str) -> bool:
    if status != 404:
        return False
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("success") is not False:
        return False
    error = payload.get("error")
    return isinstance(error, (dict, str)) and bool(error)


def _is_valid_error_response(
    status: int,
    body: str,
    *,
    expected_status: int,
) -> bool:
    return status == expected_status and is_openai_error_response(
        body,
        expected_status=expected_status,
    )


def _is_nonnegative_file_size(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value >= 0
    if isinstance(value, str) and value.strip().isdigit():
        return int(value) >= 0
    return False


def _is_valid_batch_media_type(media_type: str, *, expected_format: str) -> bool:
    normalized = media_type.lower().split(";", 1)[0]
    return normalized in EXPECTED_BATCH_MEDIA_TYPES.get(expected_format.lower(), set())


def _mark_protocol_error(result: ScenarioResult, *, status: str, error: str) -> None:
    result.status = status
    result.success = False
    result.capability = "fail"
    result.error_class = "protocol_error"
    result.error = error
