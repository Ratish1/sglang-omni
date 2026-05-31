# SPDX-License-Identifier: Apache-2.0
"""HTTP and SSE clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import binascii
import json
import time
from collections.abc import Mapping
from typing import Any

import aiohttp

from benchmarks.tts_serving.metrics import (
    ScenarioResult,
    classify_http_status,
    duration_from_audio_bytes,
    finish_timing,
    parse_sse_audio_event,
)
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec

AUDIO_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "flac", "aac", "opus"}
OPTIONAL_ENDPOINTS = {"voices", "batch", "websocket"}
SUCCESS_BATCH_STATUSES = {"ok", "success", "succeeded"}
FAILED_BATCH_STATUSES = {"error", "failed"}


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
        batch_size=scenario.planned_metadata.get("batch_size"),
    )
    url = f"{spec.base_url}{scenario.path}"
    start = time.perf_counter()
    try:
        if scenario.method == "GET":
            async with session.get(url) as response:
                await _handle_probe_response(
                    response,
                    result,
                    scenario,
                    allow_missing=spec.params.allow_missing_optional_endpoints,
                )
        elif scenario.method == "DELETE":
            async with session.delete(url) as response:
                await _handle_binary_response(
                    response,
                    result,
                    scenario,
                    allow_missing=spec.params.allow_missing_optional_endpoints,
                )
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
                        allow_missing=spec.params.allow_missing_optional_endpoints,
                    )
                else:
                    await _handle_binary_response(
                        response,
                        result,
                        scenario,
                        allow_missing=spec.params.allow_missing_optional_endpoints,
                    )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
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
    *,
    allow_missing: bool = False,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    if response.status == 404:
        _mark_missing_contract(
            result,
            scenario,
            body=await response.text(),
            allow_missing=allow_missing,
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
    *,
    allow_missing: bool = False,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    body = await response.read()
    result.response_bytes = len(body)
    if (
        response.status == 404
        and scenario.expect_success
        and _is_optional_endpoint(scenario)
    ):
        _mark_missing_contract(
            result,
            scenario,
            body=body.decode("utf-8", errors="replace"),
            allow_missing=allow_missing,
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
    *,
    allow_missing: bool = False,
) -> None:
    result.http_status = response.status
    result.http_status_class = classify_http_status(response.status)
    result.response_headers = dict(response.headers)
    if response.status != 200:
        body = await response.text()
        result.response_bytes = len(body.encode("utf-8"))
        if (
            response.status == 404
            and scenario.expect_success
            and _is_optional_endpoint(scenario)
        ):
            _mark_missing_contract(
                result,
                scenario,
                body=body,
                allow_missing=allow_missing,
            )
            return
        _classify_http_failure(response.status, body, result, scenario)
        return

    buffer = bytearray()
    chunk_times: list[float] = []
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw_line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            _merge_sse_line(
                raw_line.decode("utf-8", errors="replace").strip(),
                result,
                start,
                chunk_times,
                scenario,
            )
    if buffer.strip():
        _merge_sse_line(
            bytes(buffer).decode("utf-8", errors="replace").strip(),
            result,
            start,
            chunk_times,
            scenario,
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
    _mark_success(result)
    result.response_bytes = result.audio_bytes


def _merge_sse_line(
    line: str,
    result: ScenarioResult,
    start: float,
    chunk_times: list[float],
    scenario: Scenario,
) -> None:
    try:
        audio_bytes, event = parse_sse_audio_event(line)
    except (ValueError, binascii.Error) as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "protocol_error"
        result.error = f"malformed SSE audio event: {exc}"
        return
    if event is None:
        return
    if audio_bytes is None:
        return
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


def _mark_missing_contract(
    result: ScenarioResult,
    scenario: Scenario,
    *,
    body: str,
    allow_missing: bool,
) -> None:
    result.success = False
    result.error_class = "missing_contract"
    result.error = (
        "required benchmark contract is missing: "
        f"endpoint={scenario.endpoint}, operation={scenario.capability_key}, "
        f"path={scenario.path}, http_status={result.http_status}, body={body}"
    )
    if allow_missing:
        result.status = "missing"
        result.capability = "missing"
        return
    result.status = "missing_contract"
    result.capability = "fail"


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
    if scenario.expect_success or _is_optional_endpoint(scenario):
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
            _synthetic_audio_bytes(scenario.upload_size_bytes),
            filename=scenario.upload_filename or "audio.wav",
            content_type=scenario.upload_content_type or "audio/wav",
        )
    return form


def _synthetic_audio_bytes(size: int) -> bytes:
    if size <= 0:
        return b""
    if size < 44:
        return _wav_header(0)[:size]
    payload_size = size - 44
    return _wav_header(payload_size) + b"\0" * payload_size


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


def _is_optional_endpoint(scenario: Scenario) -> bool:
    return scenario.endpoint in OPTIONAL_ENDPOINTS


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
    required_keys = {"results", "total", "succeeded", "failed"}
    if not isinstance(payload, dict) or not required_keys <= set(payload):
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error="batch endpoint returned JSON without results/total/succeeded/failed",
        )
        return
    batch_size = int(scenario.planned_metadata.get("batch_size") or 0)
    results = payload.get("results")
    total = payload.get("total")
    succeeded = payload.get("succeeded")
    failed = payload.get("failed")
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
    expected_item_failures = sum(
        1
        for item in scenario.payload.get("items", [])
        if not isinstance(item, dict)
        or not isinstance(item.get("input"), str)
        or not item.get("input", "").strip()
        or item.get("response_format") == "bogus"
    )
    if failed < expected_item_failures:
        _mark_protocol_error(
            result,
            status="invalid_batch_response",
            error=(
                "batch endpoint did not report expected item-level failures "
                f"(expected_at_least={expected_item_failures}, failed={failed})"
            ),
        )
        return
    observed_success = 0
    observed_failed = 0
    for index, item in enumerate(results):
        if not _is_valid_batch_item(item, expected_index=index):
            _mark_protocol_error(
                result,
                status="invalid_batch_response",
                error=f"batch endpoint result item has invalid schema at index {index}",
            )
            return
        item_status = str(item["status"])
        if item_status in SUCCESS_BATCH_STATUSES:
            observed_success += 1
        else:
            observed_failed += 1
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
        voices = _voice_list_items(payload)
        if voices is not None and all(
            _is_valid_voice_metadata(item) for item in voices
        ):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error=(
                "voice list response must be a list or object with voices/data list "
                "whose items include voice metadata"
            ),
        )
        return
    if scenario.capability_key == "voices.upload":
        if isinstance(payload, dict) and any(
            isinstance(payload.get(key), str) and payload[key]
            for key in ("id", "voice_id", "name")
        ):
            _mark_success(result, capability="pass")
            return
        _mark_protocol_error(
            result,
            status="invalid_voice_response",
            error="voice upload response must include id, voice_id, or name",
        )
        return
    _mark_success(result, capability="pass")


def _is_valid_batch_item(item: Any, *, expected_index: int) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("index") != expected_index:
        return False
    status = item.get("status")
    if status in SUCCESS_BATCH_STATUSES:
        return (
            isinstance(item.get("audio_data"), str)
            and bool(item["audio_data"])
            and isinstance(item.get("media_type"), str)
            and bool(item["media_type"])
        )
    if status in FAILED_BATCH_STATUSES:
        error = item.get("error")
        return isinstance(error, (dict, str)) and bool(error)
    return False


def _voice_list_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    voices = payload.get("voices", payload.get("data"))
    return voices if isinstance(voices, list) else None


def _is_valid_voice_metadata(item: Any) -> bool:
    return isinstance(item, dict) and any(
        isinstance(item.get(key), str) and item[key]
        for key in ("id", "voice_id", "name")
    )


def _mark_protocol_error(result: ScenarioResult, *, status: str, error: str) -> None:
    result.status = status
    result.success = False
    result.capability = "fail"
    result.error_class = "protocol_error"
    result.error = error
