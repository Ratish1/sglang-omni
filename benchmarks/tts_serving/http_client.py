# SPDX-License-Identifier: Apache-2.0
"""HTTP and SSE clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Mapping
from typing import Any

import aiohttp

from benchmarks.tts_serving.metrics import (
    SSE_DONE_MARKER,
    ScenarioResult,
    classify_http_status,
    duration_from_audio_bytes,
    finish_timing,
    parse_sse_audio_event,
)
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec
from benchmarks.tts_serving.urls import api_url
from benchmarks.tts_serving.voice_upload_fixtures import get_voice_upload_fixture

AUDIO_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "flac", "aac", "opus"}
SUCCESS_BATCH_STATUSES = {"ok", "success", "succeeded"}
FAILED_BATCH_STATUSES = {"error", "failed"}
VALID_BATCH_TASK_TYPES = {"Base", "CustomVoice", "VoiceDesign"}
EXPECTED_BATCH_MEDIA_TYPES = {
    "wav": {"audio/wav", "audio/x-wav", "application/octet-stream", ""},
    "pcm": {"application/octet-stream", "audio/pcm", "audio/raw"},
    "mp3": {"audio/mpeg", "audio/mp3"},
    "flac": {"audio/flac"},
    "aac": {"audio/aac", "audio/aacp"},
    "opus": {"audio/opus", "audio/ogg"},
}


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
    if response.status == 404:
        _mark_unsupported_contract(
            result,
            scenario,
            body=await response.text(),
        )
        return
    if 200 <= response.status < 300:
        body = await response.read()
        result.response_bytes = len(body)
        if scenario.endpoint == "voices":
            _handle_voice_success(body, result, scenario)
            return
        _mark_success(result, capability="pass")
        return
    body = await response.text()
    _classify_http_failure(response.status, body, result, scenario)


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
    if response.status == 404 and scenario.capability_key != "voices.delete":
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
        if not _is_audio_response(body, response.headers, response_format):
            _mark_protocol_error(
                result,
                status="invalid_audio_response",
                error=(
                    "speech endpoint returned 2xx without the requested audio "
                    f"contract (format={response_format!r}, "
                    f"content-type={response.headers.get('Content-Type')!r}, "
                    f"bytes={len(body)})"
                ),
            )
            return
        result.audio_bytes = len(body)
        result.audio_duration_s = duration_from_audio_bytes(
            body,
            content_type=response.headers.get("Content-Type"),
            response_format=response_format,
        )
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
        body = await response.text()
        result.response_bytes = len(body.encode("utf-8"))
        if response.status == 404:
            _mark_unsupported_contract(result, scenario, body=body)
            return
        _classify_http_failure(response.status, body, result, scenario)
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
                )
                or saw_done
            )
            if result.status == "failed":
                break
        if result.status == "failed":
            break
    if result.status != "failed" and buffer.strip():
        saw_done = (
            _merge_sse_line(
                bytes(buffer).decode("utf-8", errors="replace").strip(),
                result,
                start,
                chunk_times,
                scenario,
            )
            or saw_done
        )
    if result.status == "failed":
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
    _mark_success(result)
    result.response_bytes = result.audio_bytes


def _merge_sse_line(
    line: str,
    result: ScenarioResult,
    start: float,
    chunk_times: list[float],
    scenario: Scenario,
) -> bool:
    if not line or line.startswith(":"):
        return False
    if line == SSE_DONE_MARKER:
        return True
    try:
        audio_bytes, event = parse_sse_audio_event(line)
    except (ValueError, binascii.Error) as exc:
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
    now = time.perf_counter()
    if result.ttfa_s is None:
        result.ttfa_s = now - start
    elif chunk_times:
        result.inter_chunk_s.append(now - chunk_times[-1])
    chunk_times.append(now)
    result.audio_bytes += len(audio_bytes)
    result.response_bytes += len(audio_bytes)
    audio = event.get("audio") if isinstance(event, dict) else None
    result.audio_duration_s += duration_from_audio_bytes(
        audio_bytes,
        content_type=(
            (audio or {}).get("mime_type") if isinstance(audio, dict) else None
        ),
        response_format=(
            (audio or {}).get("format")
            if isinstance(audio, dict)
            else str(scenario.payload.get("response_format", ""))
        ),
        sample_rate=(
            (audio or {}).get("sample_rate", 24000)
            if isinstance(audio, dict)
            else 24000
        ),
    )
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
) -> None:
    result.success = False
    result.status = "unsupported_contract"
    result.capability = "fail"
    result.error_class = "unsupported_endpoint"
    result.error = (
        "enabled benchmark contract is unsupported: "
        f"endpoint={scenario.endpoint}, operation={scenario.capability_key}, "
        f"path={scenario.path}, http_status={result.http_status}, body={body}"
    )


def _classify_http_failure(
    status: int,
    body: str,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.success = False
    result.error = body
    if 500 <= status:
        result.status = "failed"
        result.error_class = "server_error"
        result.capability = "fail"
        return
    if scenario.capability_key == "voices.delete" and not scenario.expect_success:
        if _is_valid_missing_voice_delete_response(status, body):
            result.status = "expected_error"
            result.error_class = "expected_client_error"
            result.capability = "pass"
            return
        result.status = "invalid_voice_response"
        result.error_class = "protocol_error"
        result.capability = "fail"
        result.error = (
            "missing voice delete must return HTTP 404 JSON with "
            f"success=false and error details; received status={status}, body={body}"
        )
        return
    if status == 404:
        _mark_unsupported_contract(result, scenario, body=body)
        return
    if (
        400 <= status < 500
        and not scenario.expect_success
        and scenario.expected_status_class == "client_error"
    ):
        result.status = "expected_error"
        result.error_class = "expected_client_error"
        result.capability = "pass"
        return
    result.status = "failed"
    result.error_class = "http_error"
    if scenario.expect_success:
        result.capability = "fail"


def _is_audio_response(
    body: bytes,
    headers: Mapping[str, str],
    response_format: str,
) -> bool:
    if not body:
        return False
    content_type = str(headers.get("Content-Type", "")).lower().split(";", 1)[0]
    fmt = response_format.lower()
    if fmt == "wav":
        return body.startswith(b"RIFF") and content_type in {
            "audio/wav",
            "audio/x-wav",
            "application/octet-stream",
            "",
        }
    if fmt == "pcm":
        return content_type in {"application/octet-stream", "audio/pcm", "audio/raw"}
    if fmt == "mp3":
        return (
            content_type in {"audio/mpeg", "audio/mp3"}
            or body.startswith(b"ID3")
            or (len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0)
        )
    if fmt == "flac":
        return body.startswith(b"fLaC") or content_type == "audio/flac"
    if fmt == "opus":
        return body.startswith(b"OggS") or content_type in {"audio/opus", "audio/ogg"}
    if fmt == "aac":
        return content_type in {"audio/aac", "audio/aacp"} or (
            len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xF0) == 0xF0
        )
    return False


def _request_body(scenario: Scenario) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for key, value in scenario.form_fields.items():
        form.add_field(key, value)
    if scenario.upload_field:
        form.add_field(
            scenario.upload_field,
            _synthetic_upload_bytes(scenario),
            filename=scenario.upload_filename or "audio.wav",
            content_type=scenario.upload_content_type or "audio/wav",
        )
    return form


def _synthetic_upload_bytes(scenario: Scenario) -> bytes:
    upload_case = str(scenario.planned_metadata.get("upload_case", "format"))
    if upload_case == "corrupt_audio":
        return _pad_bytes(b"not-a-valid-audio-upload", scenario.upload_size_bytes)
    upload_format = str(scenario.planned_metadata.get("upload_format", "wav"))
    if upload_case == "format" and upload_format != "wav":
        return get_voice_upload_fixture(upload_format)
    return _synthetic_audio_bytes(scenario.upload_size_bytes, upload_format)


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
    if size < 44:
        return _wav_header(0)[:size]
    payload_size = size - 44
    return _wav_header(payload_size) + b"\0" * payload_size


def _pad_bytes(prefix: bytes, size: int) -> bytes:
    if size <= len(prefix):
        return prefix[:size]
    return prefix + (b"\0" * (size - len(prefix)))


def _wav_header(payload_size: int) -> bytes:
    return (
        b"RIFF"
        + (36 + payload_size).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (24000).to_bytes(4, "little")
        + (48000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + payload_size.to_bytes(4, "little")
    )


def _request_size(scenario: Scenario) -> int:
    if scenario.body_type == "multipart":
        return (
            sum(len(key) + len(value) for key, value in scenario.form_fields.items())
            + scenario.upload_size_bytes
        )
    try:
        return len(json.dumps(scenario.payload, ensure_ascii=False).encode("utf-8"))
    except TypeError:
        return 0


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
        if upload_response.status == 404:
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


def _scenario_response_format(scenario: Scenario) -> str | None:
    response_format = scenario.planned_metadata.get("response_format")
    if response_format is None:
        response_format = scenario.payload.get("response_format")
    return str(response_format) if response_format is not None else None


def _handle_batch_success(
    body: bytes, result: ScenarioResult, scenario: Scenario
) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=f"batch endpoint returned invalid JSON: {exc}",
        )
        result.error_type = exc.__class__.__name__
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
    for index, item in enumerate(results):
        expect_item_failure = index in expected_item_failures
        expected_format = _expected_batch_response_format(scenario, index)
        validation_error = _validate_batch_item(
            item,
            expected_index=index,
            expected_format=expected_format,
            expect_failure=expect_item_failure,
        )
        if validation_error is not None:
            _mark_protocol_error(
                result,
                status="invalid_batch_response",
                error=f"batch endpoint result item {index}: {validation_error}",
            )
            return
        if expect_item_failure:
            observed_failed += 1
        else:
            observed_success += 1
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
        isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0
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
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError as exc:
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=f"voice endpoint returned invalid JSON: {exc}",
        )
        result.error_type = exc.__class__.__name__
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
) -> str | None:
    if not isinstance(item, dict):
        return "result item must be a JSON object"
    if item.get("index") != expected_index:
        return (
            "index mismatch "
            f"(expected={expected_index}, observed={item.get('index')})"
        )
    status = item.get("status")
    if expect_failure:
        if status not in FAILED_BATCH_STATUSES:
            return f"expected failed status, observed={status!r}"
        error = item.get("error")
        if not isinstance(error, (dict, str)) or not error:
            return "failed item must include non-empty error details"
        return None
    if status not in SUCCESS_BATCH_STATUSES:
        return f"expected success status, observed={status!r}"
    audio_data = item.get("audio_data")
    media_type = item.get("media_type")
    if not isinstance(audio_data, str) or not audio_data:
        return "successful item must include non-empty base64 audio_data"
    if not isinstance(media_type, str) or not _is_valid_batch_media_type(
        media_type, expected_format=expected_format
    ):
        return (
            "successful item media_type does not match requested format "
            f"(format={expected_format!r}, media_type={media_type!r})"
        )
    try:
        decoded = base64.b64decode(audio_data, validate=True)
    except binascii.Error as exc:
        return f"successful item audio_data is not valid base64: {exc}"
    if not decoded:
        return "successful item audio_data decoded to empty bytes"
    return None


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


def _is_valid_voice_delete_success(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
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
