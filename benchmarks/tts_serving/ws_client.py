# SPDX-License-Identifier: Apache-2.0
"""WebSocket capability probe for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time

import aiohttp

from benchmarks.tts_serving.metrics import ScenarioResult, finish_timing
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec

WS_AUDIO_EVENT_TYPES = {
    "audio",
    "audio.delta",
    "audio.speech.chunk",
    "output_audio.delta",
    "response.audio.delta",
    "speech.audio.delta",
}
WS_CONTROL_EVENT_TYPES = {
    "session.created",
    "session.updated",
    "response.created",
    "input.ack",
}


async def run_ws_scenario(
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
    url = _ws_url(spec.base_url, scenario.path)
    start = time.perf_counter()
    try:
        async with session.ws_connect(url) as ws:
            await _run_ws_script(
                ws,
                result,
                scenario.script or _default_script(spec),
                timeout_s=spec.params.timeout_s,
                expect_success=scenario.expect_success,
            )
    except aiohttp.WSServerHandshakeError as exc:
        result.http_status = exc.status
        if exc.status == 404:
            result.status = "missing"
            result.capability = "missing"
        else:
            result.status = "failed"
            result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error = str(exc)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.error_type = exc.__class__.__name__
        result.error_class = "transport_error"
        result.error = str(exc)
    finally:
        finish_timing(result, start)
    return result


async def _run_ws_script(
    ws: aiohttp.ClientWebSocketResponse,
    result: ScenarioResult,
    script: list[dict],
    *,
    timeout_s: int,
    expect_success: bool,
) -> None:
    async with asyncio.timeout(timeout_s):
        for action in script:
            action_type = str(action.get("action"))
            if action_type == "send_json":
                await ws.send_json(action["payload"])
            elif action_type == "send_text":
                await ws.send_str(str(action.get("text", "")))
            elif action_type == "close":
                await ws.close()
                result.status = "ok"
                result.success = True
                result.capability = "pass"
                result.ws_close_reason = "client_closed"
                return
            elif action_type == "expect":
                matched = await _receive_until(
                    ws,
                    result,
                    expected_event=str(action.get("event", "")),
                    expect_success=expect_success,
                )
                if not matched:
                    return
            else:
                result.status = "failed"
                result.capability = "fail"
                result.error = f"unknown WebSocket benchmark action: {action_type}"
                return

    if result.status in {"error", "ok"}:
        result.status = "ok" if expect_success else "expected_error"
        result.success = expect_success
        result.capability = "pass"


async def _receive_until(
    ws: aiohttp.ClientWebSocketResponse,
    result: ScenarioResult,
    *,
    expected_event: str,
    expect_success: bool,
) -> bool:
    while True:
        msg = await ws.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            _record_ws_event(result, "binary")
            result.audio_bytes += len(msg.data)
            result.response_bytes += len(msg.data)
            if expected_event in {"audio", "binary"}:
                result.status = "ok"
                result.success = True
                result.capability = "pass"
                return True
            continue
        if msg.type == aiohttp.WSMsgType.TEXT:
            event_type = _merge_text_event(
                msg.data, result, expect_success=expect_success
            )
            if result.status in {"failed", "expected_error"}:
                return event_type == expected_event or expected_event == "error"
            if _event_matches(event_type, expected_event):
                return True
            continue
        if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
            result.ws_close_reason = "server_closed"
            if expected_event == "close":
                result.status = "ok"
                result.success = True
                result.capability = "pass"
                return True
            result.status = "failed"
            result.capability = "fail"
            result.error = "WebSocket closed before expected event"
            return False
        if msg.type == aiohttp.WSMsgType.ERROR:
            result.status = "failed"
            result.capability = "fail"
            result.error = str(ws.exception())
            return False
    return False


def _merge_text_event(
    data: str,
    result: ScenarioResult,
    *,
    expect_success: bool = True,
) -> str | None:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        result.status = "failed" if expect_success else "expected_error"
        result.capability = "fail" if expect_success else "pass"
        result.error_type = exc.__class__.__name__
        result.error_class = "protocol_error"
        result.error = f"malformed WebSocket JSON event: {exc}"
        return "error"
    if not isinstance(event, dict):
        result.status = "failed" if expect_success else "expected_error"
        result.capability = "fail" if expect_success else "pass"
        result.error = "WebSocket event is not a JSON object"
        return "error"

    event_type = str(event.get("type", ""))
    _record_ws_event(result, event_type or "text")
    if "error" in event_type or "error" in event:
        result.status = "failed" if expect_success else "expected_error"
        result.capability = "fail" if expect_success else "pass"
        result.error_class = "server_error_event"
        result.error = data
        return "error"
    if event_type in WS_CONTROL_EVENT_TYPES:
        return event_type

    audio = event.get("audio")
    if event_type in WS_AUDIO_EVENT_TYPES or isinstance(audio, (dict, str)):
        result.status = "ok"
        result.success = True
        result.capability = "pass"
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            audio_len = _encoded_audio_len(audio["data"])
            result.audio_bytes += audio_len
            result.response_bytes += audio_len
        elif isinstance(audio, str):
            audio_len = _encoded_audio_len(audio)
            result.audio_bytes += audio_len
            result.response_bytes += audio_len
        return "audio"
    if event_type in {"audio.start", "audio.done", "session.done"}:
        result.status = "ok"
        result.capability = "pass"
        return event_type

    result.status = "failed"
    result.capability = "fail"
    result.error = f"unexpected WebSocket event: {data}"
    return event_type


def _encoded_audio_len(value: str) -> int:
    try:
        return len(base64.b64decode(value, validate=True))
    except binascii.Error:
        return len(value)


def _ws_url(base_url: str, path: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :] + path
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :] + path
    return base_url.rstrip("/") + path


def _record_ws_event(result: ScenarioResult, event_type: str) -> None:
    result.ws_event_counts[event_type] = result.ws_event_counts.get(event_type, 0) + 1


def _event_matches(event_type: str | None, expected_event: str) -> bool:
    if expected_event == "audio":
        return event_type == "audio"
    return event_type == expected_event


def _default_script(spec: BenchmarkSpec) -> list[dict]:
    return [
        {
            "action": "send_json",
            "payload": {
                "type": "session.config",
                "model": spec.model_name,
                "voice": "default",
                "response_format": "pcm",
                "stream_audio": True,
                "split_granularity": "sentence",
            },
        },
        {"action": "send_json", "payload": {"type": "input.text", "text": "Hello."}},
        {"action": "send_json", "payload": {"type": "input.done"}},
        {"action": "expect", "event": "audio"},
    ]
