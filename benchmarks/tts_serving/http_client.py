# SPDX-License-Identifier: Apache-2.0
"""HTTP and SSE clients for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import binascii
import time
from collections.abc import Mapping

import aiohttp

from benchmarks.tts_serving.metrics import (
    ScenarioResult,
    duration_from_audio_bytes,
    finish_timing,
    parse_sse_audio_event,
)
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec

AUDIO_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "flac", "aac", "opus"}


async def run_http_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        expected_success=scenario.expect_success,
    )
    url = f"{spec.base_url}{scenario.path}"
    start = time.perf_counter()
    try:
        if scenario.method == "GET":
            async with session.get(url) as response:
                await _handle_probe_response(response, result)
        else:
            async with session.post(url, json=scenario.payload) as response:
                if scenario.endpoint == "speech_sse":
                    await _handle_sse_response(response, result, start, scenario)
                else:
                    await _handle_binary_response(response, result, scenario)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.error_type = exc.__class__.__name__
        result.error = str(exc)
    finally:
        finish_timing(result, start)
    return result


async def _handle_probe_response(
    response: aiohttp.ClientResponse, result: ScenarioResult
) -> None:
    result.http_status = response.status
    result.response_headers = dict(response.headers)
    if response.status == 404:
        result.status = "missing"
        result.capability = "missing"
        result.success = False
        result.error = await response.text()
        return
    if 200 <= response.status < 300:
        result.status = "ok"
        result.capability = "pass"
        result.success = True
        await response.read()
        return
    result.status = "failed"
    result.capability = "fail"
    result.error = await response.text()


async def _handle_binary_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.response_headers = dict(response.headers)
    body = await response.read()
    result.audio_bytes = len(body)
    if response.status == 404 and scenario.category == "capability_probe":
        result.status = "missing"
        result.capability = "missing"
        result.error = body.decode("utf-8", errors="replace")
        return
    if scenario.category == "capability_probe":
        if 200 <= response.status < 300:
            result.status = "ok"
            result.capability = "pass"
            result.success = True
            return
        result.status = "failed"
        result.capability = "fail"
        result.error = body.decode("utf-8", errors="replace")
        return
    if 200 <= response.status < 300:
        if not scenario.expect_success:
            _mark_unexpected_success(result, scenario)
            return
        response_format = str(scenario.payload.get("response_format", ""))
        if not _is_audio_response(body, response.headers, response_format):
            result.success = False
            result.status = "invalid_audio_response"
            result.error = (
                "speech endpoint returned 2xx without an audio-like response "
                f"(content-type={response.headers.get('Content-Type')!r}, "
                f"bytes={len(body)})"
            )
            return
        result.audio_duration_s = duration_from_audio_bytes(
            body,
            content_type=response.headers.get("Content-Type"),
            response_format=response_format,
        )
        result.success = True
        result.status = "ok"
        return
    result.success = False
    result.status = "expected_error" if not scenario.expect_success else "failed"
    result.error = body.decode("utf-8", errors="replace")


async def _handle_sse_response(
    response: aiohttp.ClientResponse,
    result: ScenarioResult,
    start: float,
    scenario: Scenario,
) -> None:
    result.http_status = response.status
    result.response_headers = dict(response.headers)
    if response.status != 200:
        body = await response.text()
        result.status = "expected_error" if not scenario.expect_success else "failed"
        result.error = body
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
    result.success = result.audio_bytes > 0
    result.status = "ok" if result.success else "empty_stream"


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
        result.error_type = exc.__class__.__name__
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
    result.error = (
        f"scenario {scenario.id} expected an error but received HTTP "
        f"{result.http_status}"
    )


def _is_audio_response(
    body: bytes,
    headers: Mapping[str, str],
    response_format: str,
) -> bool:
    if not body:
        return False
    content_type = str(headers.get("Content-Type", "")).lower().split(";", 1)[0]
    fmt = response_format.lower()
    if content_type.startswith("audio/"):
        return True
    if fmt == "wav":
        return body.startswith(b"RIFF")
    if fmt == "pcm":
        return content_type == "application/octet-stream"
    if fmt == "mp3":
        return body.startswith(b"ID3") or (
            len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0
        )
    if fmt == "flac":
        return body.startswith(b"fLaC")
    if fmt == "opus":
        return body.startswith(b"OggS")
    if fmt == "aac":
        return len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xF0) == 0xF0
    if fmt in AUDIO_RESPONSE_FORMATS:
        return content_type == "application/octet-stream"
    return False
