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
            await ws.send_json(scenario.payload)
            await ws.send_json({"type": "input.text", "text": "Hello from benchmark."})
            await ws.send_json({"type": "input.done"})
            await _receive_audio_event(ws, result, timeout_s=spec.params.timeout_s)
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
        result.error = str(exc)
    finally:
        finish_timing(result, start)
    return result


async def _receive_audio_event(
    ws: aiohttp.ClientWebSocketResponse,
    result: ScenarioResult,
    *,
    timeout_s: int,
) -> None:
    async with asyncio.timeout(timeout_s):
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.BINARY:
                result.status = "ok"
                result.success = True
                result.capability = "pass"
                result.audio_bytes += len(msg.data)
                return
            if msg.type == aiohttp.WSMsgType.TEXT:
                if _merge_text_event(msg.data, result):
                    return
                continue
            if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                result.status = "failed"
                result.capability = "fail"
                result.error = "WebSocket closed before an audio event"
                return
            if msg.type == aiohttp.WSMsgType.ERROR:
                result.status = "failed"
                result.capability = "fail"
                result.error = str(ws.exception())
                return


def _merge_text_event(data: str, result: ScenarioResult) -> bool:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        result.status = "failed"
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error = f"malformed WebSocket JSON event: {exc}"
        return True
    if not isinstance(event, dict):
        result.status = "failed"
        result.capability = "fail"
        result.error = "WebSocket event is not a JSON object"
        return True

    event_type = str(event.get("type", ""))
    if "error" in event_type or "error" in event:
        result.status = "failed"
        result.capability = "fail"
        result.error = data
        return True
    if event_type in WS_CONTROL_EVENT_TYPES:
        return False

    audio = event.get("audio")
    if event_type in WS_AUDIO_EVENT_TYPES or isinstance(audio, (dict, str)):
        result.status = "ok"
        result.success = True
        result.capability = "pass"
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            result.audio_bytes += _encoded_audio_len(audio["data"])
        elif isinstance(audio, str):
            result.audio_bytes += _encoded_audio_len(audio)
        return True

    result.status = "failed"
    result.capability = "fail"
    result.error = f"unexpected WebSocket event: {data}"
    return True


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
