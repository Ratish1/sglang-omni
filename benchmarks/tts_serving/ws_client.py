# SPDX-License-Identifier: Apache-2.0
"""WebSocket capability probe for the TTS serving benchmark."""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from benchmarks.tts_serving.metrics import ScenarioResult, finish_timing
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec

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
        capability_key=scenario.capability_key,
        expected_success=scenario.expect_success,
        response_format="pcm",
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
        if scenario.capability_key == "ws.disconnect" and result.status == "ok":
            await _probe_speech_after_disconnect(session, spec, result)
    except aiohttp.WSServerHandshakeError as exc:
        result.http_status = exc.status
        if exc.status == 404:
            _mark_missing_ws_contract(
                result,
                scenario,
                error=str(exc),
                allow_missing=spec.params.allow_missing_optional_endpoints,
            )
        else:
            result.status = "failed"
            result.capability = "fail"
            result.error_class = "http_error"
        result.error_type = exc.__class__.__name__
        if result.error is None:
            result.error = str(exc)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.capability = "fail"
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
                result.was_cancelled = True
                result.ws_close_reason = "client_closed"
                return
            elif action_type == "expect":
                matched = await _expect_next_event(
                    ws,
                    result,
                    expected_event=str(action.get("event", "")),
                    expect_success=expect_success,
                )
                if not matched:
                    return
            elif action_type == "expect_audio_until_done":
                matched = await _expect_audio_until_done(
                    ws,
                    result,
                    min_binary_frames=int(action.get("min_binary_frames", 1)),
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


async def _expect_next_event(
    ws: aiohttp.ClientWebSocketResponse,
    result: ScenarioResult,
    *,
    expected_event: str,
    expect_success: bool,
) -> bool:
    while True:
        msg = await ws.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            if expected_event not in {"audio", "binary"}:
                _mark_ws_protocol_error(
                    result,
                    f"received binary audio while expecting {expected_event}",
                )
                return False
            return _record_binary_audio(msg.data, result)
        if msg.type == aiohttp.WSMsgType.TEXT:
            event_type = _merge_text_event(
                msg.data, result, expect_success=expect_success
            )
            if result.status in {"failed", "expected_error"}:
                return event_type == expected_event or expected_event == "error"
            if _event_matches(event_type, expected_event):
                return True
            if event_type in WS_CONTROL_EVENT_TYPES:
                continue
            _mark_ws_protocol_error(
                result,
                f"received WebSocket event {event_type!r} while expecting {expected_event!r}",
            )
            return False
        if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
            result.ws_close_reason = "server_closed"
            if expected_event == "close":
                result.status = "ok"
                result.success = True
                result.capability = "pass"
                return True
            _mark_ws_protocol_error(result, "WebSocket closed before expected event")
            return False
        if msg.type == aiohttp.WSMsgType.ERROR:
            result.status = "failed"
            result.capability = "fail"
            result.error_class = "transport_error"
            result.error = str(ws.exception())
            return False
        _mark_ws_protocol_error(result, f"unexpected WebSocket frame type: {msg.type}")
        return False


async def _expect_audio_until_done(
    ws: aiohttp.ClientWebSocketResponse,
    result: ScenarioResult,
    *,
    min_binary_frames: int,
    expect_success: bool,
) -> bool:
    binary_frames = 0
    while True:
        msg = await ws.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            if not _record_binary_audio(msg.data, result):
                return False
            binary_frames += 1
            continue
        if msg.type == aiohttp.WSMsgType.TEXT:
            event_type = _merge_text_event(
                msg.data, result, expect_success=expect_success
            )
            if result.status in {"failed", "expected_error"}:
                return event_type == "error"
            if event_type in WS_CONTROL_EVENT_TYPES:
                continue
            if event_type == "audio.done":
                if binary_frames < min_binary_frames:
                    _mark_ws_protocol_error(
                        result,
                        "stream_audio=true completed before the required "
                        f"incremental audio chunks (expected>={min_binary_frames}, "
                        f"observed={binary_frames})",
                    )
                    return False
                return True
            _mark_ws_protocol_error(
                result,
                "received WebSocket event "
                f"{event_type!r} while streaming binary audio until audio.done",
            )
            return False
        if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
            result.ws_close_reason = "server_closed"
            _mark_ws_protocol_error(result, "WebSocket closed before audio.done")
            return False
        if msg.type == aiohttp.WSMsgType.ERROR:
            result.status = "failed"
            result.capability = "fail"
            result.error_class = "transport_error"
            result.error = str(ws.exception())
            return False
        _mark_ws_protocol_error(result, f"unexpected WebSocket frame type: {msg.type}")
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
    if _is_ws_error_event(event_type):
        result.status = "failed" if expect_success else "expected_error"
        result.capability = "fail" if expect_success else "pass"
        result.error_class = "server_error_event"
        result.error = data
        return "error"
    if event_type in WS_CONTROL_EVENT_TYPES:
        return event_type

    if event_type == "audio.start":
        if not _is_valid_audio_start(event):
            _mark_ws_protocol_error(result, f"invalid audio.start event: {data}")
            return event_type
        if result.ws_active_sentence_index is not None:
            _mark_ws_protocol_error(
                result,
                "received audio.start before the previous sentence completed",
            )
            return event_type
        result.ws_active_sentence_index = event["sentence_index"]
        result.ws_active_sentence_bytes = 0
        result.status = "ok"
        result.capability = "pass"
        return event_type
    if event_type == "audio.done":
        if not _is_valid_audio_done(event):
            _mark_ws_protocol_error(result, f"invalid audio.done event: {data}")
            return event_type
        if event.get("error") is True:
            result.status = "failed" if expect_success else "expected_error"
            result.capability = "fail" if expect_success else "pass"
            result.error_class = "server_error_event"
            result.error = data
            return "error"
        if event["sentence_index"] != result.ws_active_sentence_index:
            _mark_ws_protocol_error(
                result,
                "audio.done sentence_index does not match active sentence",
            )
            return event_type
        if event["total_bytes"] != result.ws_active_sentence_bytes:
            _mark_ws_protocol_error(
                result,
                "audio.done total_bytes does not match received binary audio bytes",
            )
            return event_type
        result.ws_completed_sentences += 1
        result.ws_active_sentence_index = None
        result.ws_active_sentence_bytes = 0
        result.status = "ok"
        result.capability = "pass"
        return event_type
    if event_type == "session.done":
        if not _is_valid_session_done(event):
            _mark_ws_protocol_error(result, f"invalid session.done event: {data}")
            return event_type
        if result.ws_active_sentence_index is not None:
            _mark_ws_protocol_error(
                result,
                "session.done arrived before active sentence completed",
            )
            return event_type
        if event["total_sentences"] != result.ws_completed_sentences:
            _mark_ws_protocol_error(
                result,
                "session.done total_sentences does not match completed sentences",
            )
            return event_type
        result.status = "ok"
        result.capability = "pass"
        return event_type

    _mark_ws_protocol_error(result, f"unexpected WebSocket event: {data}")
    return event_type


def _is_ws_error_event(event_type: str) -> bool:
    return event_type == "error" or event_type.endswith(".error")


def _is_valid_audio_start(event: dict) -> bool:
    return (
        isinstance(event.get("sentence_index"), int)
        and event["sentence_index"] >= 0
        and isinstance(event.get("sentence_text"), str)
        and bool(event["sentence_text"])
        and isinstance(event.get("format"), str)
        and event["format"] == "pcm"
        and isinstance(event.get("sample_rate"), int)
        and event["sample_rate"] == 24000
    )


def _is_valid_audio_done(event: dict) -> bool:
    return (
        isinstance(event.get("sentence_index"), int)
        and event["sentence_index"] >= 0
        and isinstance(event.get("total_bytes"), int)
        and event["total_bytes"] >= 0
        and isinstance(event.get("error"), bool)
    )


def _is_valid_session_done(event: dict) -> bool:
    return (
        isinstance(event.get("total_sentences"), int) and event["total_sentences"] >= 0
    )


def _ws_url(base_url: str, path: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :] + path
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :] + path
    return base_url.rstrip("/") + path


def _record_ws_event(result: ScenarioResult, event_type: str) -> None:
    result.ws_event_counts[event_type] = result.ws_event_counts.get(event_type, 0) + 1


def _record_binary_audio(data: bytes, result: ScenarioResult) -> bool:
    _record_ws_event(result, "binary")
    if result.ws_active_sentence_index is None:
        _mark_ws_protocol_error(result, "received binary audio before audio.start")
        return False
    result.audio_bytes += len(data)
    result.ws_active_sentence_bytes += len(data)
    result.response_bytes += len(data)
    result.status = "ok"
    result.success = True
    result.capability = "pass"
    return True


def _event_matches(event_type: str | None, expected_event: str) -> bool:
    if expected_event == "audio":
        return event_type == "audio"
    return event_type == expected_event


def _mark_ws_protocol_error(result: ScenarioResult, error: str) -> None:
    result.status = "failed"
    result.success = False
    result.capability = "fail"
    result.error_class = "protocol_error"
    result.error = error


def _mark_missing_ws_contract(
    result: ScenarioResult,
    scenario: Scenario,
    *,
    error: str,
    allow_missing: bool,
) -> None:
    result.success = False
    result.error_class = "missing_contract"
    result.error = (
        "required benchmark contract is missing: "
        f"endpoint={scenario.endpoint}, operation={scenario.capability_key}, "
        f"path={scenario.path}, http_status={result.http_status}, error={error}"
    )
    if allow_missing:
        result.status = "missing"
        result.capability = "missing"
        return
    result.status = "missing_contract"
    result.capability = "fail"


async def _probe_speech_after_disconnect(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    result: ScenarioResult,
) -> None:
    url = f"{spec.base_url}/v1/audio/speech"
    try:
        async with session.post(
            url,
            json={
                "model": spec.model_name,
                "input": "Post-disconnect liveness probe.",
                "voice": "default",
                "response_format": "wav",
            },
        ) as response:
            await response.read()
            if 200 <= response.status < 300:
                result.capability = "pass"
                return
            result.status = "failed"
            result.success = False
            result.capability = "fail"
            result.http_status = response.status
            result.error_class = (
                "server_error" if response.status >= 500 else "http_error"
            )
            result.error = "post-disconnect speech liveness probe failed"
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.status = "transport_error"
        result.success = False
        result.capability = "fail"
        result.error_type = exc.__class__.__name__
        result.error_class = "transport_error"
        result.error = f"post-disconnect speech liveness probe failed: {exc}"


def _default_script(spec: BenchmarkSpec) -> list[dict]:
    return [
        {
            "action": "send_json",
            "payload": {
                "type": "session.config",
                "model": spec.model_name,
                "voice": "default",
                "response_format": "pcm",
                "stream_audio": False,
                "split_granularity": "sentence",
            },
        },
        {"action": "send_json", "payload": {"type": "input.text", "text": "Hello."}},
        {"action": "send_json", "payload": {"type": "input.done"}},
        {"action": "expect", "event": "audio.start"},
        {"action": "expect", "event": "audio"},
        {"action": "expect", "event": "audio.done"},
        {"action": "expect", "event": "session.done"},
    ]
