from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web

from benchmarks.eval.benchmark_tts_serving import _run_benchmark
from benchmarks.tts_serving.http_client import _handle_binary_response
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.scenarios import (
    Scenario,
    build_scenarios,
    scenario_set_hash,
)
from benchmarks.tts_serving.spec import BenchmarkSpec, SpecError
from benchmarks.tts_serving.ws_client import _merge_text_event, run_ws_scenario


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")


def _result_for(scenario: Scenario) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        expected_success=scenario.expect_success,
    )


def _spec(params: dict | None = None) -> BenchmarkSpec:
    return BenchmarkSpec.from_obj(
        {
            "base_url": "http://127.0.0.1:8000",
            "model_name": "higgs",
            "test_type": "external",
            "seed": 601,
            "params": params or {},
        }
    )


def _wav_bytes(payload_size: int = 16) -> bytes:
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
        + b"\0" * payload_size
    )


async def _start_app(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    assert sockets
    host, port = sockets[0].getsockname()[:2]
    return runner, f"http://{host}:{port}"


def test_spec_builds_default_stage_for_concurrency_sweep() -> None:
    spec = _spec({"total_requests": 7, "concurrency_levels": [1, 4, 16, 16]})

    assert [stage.id for stage in spec.params.load_stages] == ["c1", "c4", "c16"]
    assert [stage.max_concurrency for stage in spec.params.load_stages] == [1, 4, 16]
    assert all(stage.mode == "closed_loop" for stage in spec.params.load_stages)
    assert all(stage.request_count == 7 for stage in spec.params.load_stages)
    assert spec.params.max_concurrency == 16


def test_spec_parses_explicit_load_stages() -> None:
    spec = _spec(
        {
            "profile": "stress",
            "load_stages": [
                {
                    "id": "ramp-128",
                    "mode": "ramp",
                    "request_count": 128,
                    "max_concurrency": 128,
                    "start_request_rate": 2,
                    "request_rate": 64,
                },
                {
                    "id": "burst-128",
                    "mode": "burst",
                    "request_count": 256,
                    "max_concurrency": 128,
                },
            ],
        }
    )

    assert [stage.mode for stage in spec.params.load_stages] == ["ramp", "burst"]
    assert spec.params.load_stages[0].start_request_rate == 2.0
    assert spec.params.load_stages[0].request_rate == 64.0
    assert spec.params.max_concurrency == 128


def test_spec_rejects_invalid_load_stage() -> None:
    with pytest.raises(SpecError, match="start_request_rate"):
        _spec(
            {
                "load_stages": [
                    {
                        "id": "bad-ramp",
                        "mode": "ramp",
                        "request_count": 8,
                        "max_concurrency": 4,
                        "request_rate": 8,
                    }
                ]
            }
        )


def test_stress_scenarios_include_required_serving_families() -> None:
    spec = _spec(
        {
            "profile": "stress",
            "load_stages": [
                {
                    "id": "burst-128",
                    "mode": "burst",
                    "request_count": 48,
                    "max_concurrency": 128,
                }
            ],
        }
    )

    scenarios = build_scenarios(spec)
    ids = [scenario.id for scenario in scenarios]
    categories = {scenario.category for scenario in scenarios}
    batch_sizes = {
        scenario.planned_metadata.get("batch_size")
        for scenario in scenarios
        if scenario.endpoint == "batch"
    }

    assert len(ids) == len(set(ids))
    assert len(scenarios) == 48
    assert scenario_set_hash(scenarios) == scenario_set_hash(build_scenarios(spec))
    assert {"speech_baseline", "speech_malformed", "batch", "voices"} <= categories
    assert {"websocket", "websocket_malformed", "websocket_disconnect"} <= categories
    assert {1, 2, 32} <= batch_sizes
    assert any(
        scenario.upload_size_bytes >= (10 * 1024 * 1024) - 1 for scenario in scenarios
    )


def test_report_includes_tail_stage_endpoint_and_error_taxonomy() -> None:
    spec = _spec(
        {
            "load_stages": [
                {
                    "id": "burst",
                    "mode": "burst",
                    "request_count": 2,
                    "max_concurrency": 2,
                }
            ],
        }
    )
    scenarios = build_scenarios(spec)[:2]
    results = [
        ScenarioResult(
            scenario_id=scenarios[0].id,
            endpoint="speech",
            category="speech_baseline",
            stage_id="burst",
            load_mode="burst",
            load_concurrency=2,
            status="ok",
            success=True,
            latency_s=0.2,
            actual_start_s=10.0,
            completed_s=10.2,
            queue_wait_s=0.0,
        ),
        ScenarioResult(
            scenario_id=scenarios[1].id,
            endpoint="batch",
            category="batch",
            stage_id="burst",
            load_mode="burst",
            load_concurrency=2,
            status="missing",
            success=False,
            latency_s=0.4,
            actual_start_s=10.1,
            completed_s=10.5,
            queue_wait_s=0.1,
            http_status=404,
            http_status_class="4xx",
            error_class="http_error",
            capability="missing",
        ),
    ]

    report = build_results_report(spec, results, scenarios=scenarios)

    assert report["schema_version"] == 2
    assert report["scenario_set_hash"] == scenario_set_hash(scenarios)
    assert report["overall"]["passed"] is True
    assert report["metrics"]["latency_s"]["p99"] == 0.4
    assert report["metrics"]["latency_s"]["p99_9"] == 0.4
    assert report["metrics"]["by_stage"]["burst"]["achieved_rps"] is not None
    assert report["metrics"]["by_endpoint"]["batch"]["status_counts"]["missing"] == 1
    assert report["metrics"]["error_class_counts"]["http_error"] == 1


def test_http_batch_success_requires_batch_result_schema() -> None:
    scenario = Scenario(
        id="batch-1",
        endpoint="batch",
        category="batch",
        stage_id="burst",
        path="/v1/audio/speech/batch",
        payload={"items": [{"input": "hello"}]},
        planned_metadata={"batch_size": 1},
    )
    result = _result_for(scenario)
    response = _FakeResponse(
        status=200,
        body=b'{"results": [], "total": 1, "succeeded": 1, "failed": 0}',
        headers={"Content-Type": "application/json"},
    )

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "ok"
    assert result.success is True
    assert result.capability == "pass"


def test_http_optional_success_404_is_missing() -> None:
    scenario = Scenario(
        id="batch-missing",
        endpoint="batch",
        category="batch",
        stage_id="burst",
        path="/v1/audio/speech/batch",
    )
    result = _result_for(scenario)
    response = _FakeResponse(status=404, body=b"not found")

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "missing"
    assert result.capability == "missing"


def test_http_expected_voice_delete_404_is_expected_error() -> None:
    scenario = Scenario(
        id="voice-delete",
        endpoint="voices",
        category="voices",
        stage_id="burst",
        method="DELETE",
        path="/v1/audio/voices/missing",
        expect_success=False,
        expected_status_class="client_error",
    )
    result = _result_for(scenario)
    response = _FakeResponse(status=404, body=b"voice not found")

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "expected_error"
    assert result.capability is None


def test_http_speech_2xx_without_audio_is_invalid() -> None:
    scenario = Scenario(
        id="speech-valid",
        endpoint="speech",
        category="speech_baseline",
        stage_id="burst",
        payload={"response_format": "wav"},
    )
    result = _result_for(scenario)
    response = _FakeResponse(
        status=200,
        body=b'{"ok": true}',
        headers={"Content-Type": "application/json"},
    )

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "invalid_audio_response"
    assert result.success is False


def test_http_malformed_request_2xx_is_unexpected_success() -> None:
    scenario = Scenario(
        id="speech-malformed",
        endpoint="speech",
        category="speech_malformed",
        stage_id="burst",
        payload={"response_format": "wav"},
        expect_success=False,
    )
    result = _result_for(scenario)
    response = _FakeResponse(
        status=200,
        body=_wav_bytes(),
        headers={"Content-Type": "audio/wav"},
    )

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "unexpected_success"
    assert result.error_class == "unexpected_success"


def test_websocket_expected_error_does_not_fail_endpoint_capability() -> None:
    result = ScenarioResult(
        scenario_id="ws-malformed",
        endpoint="websocket",
        category="websocket_malformed",
        expected_success=False,
    )

    event_type = _merge_text_event(
        '{"type": "error", "error": {"message": "bad request"}}',
        result,
        expect_success=False,
    )

    assert event_type == "error"
    assert result.status == "expected_error"
    assert result.capability == "pass"


def test_websocket_audio_event_records_audio_bytes() -> None:
    result = ScenarioResult(
        scenario_id="ws-normal",
        endpoint="websocket",
        category="websocket",
    )
    encoded_audio = base64.b64encode(b"audio-bytes").decode("ascii")

    event_type = _merge_text_event(
        f'{{"type": "response.audio.delta", "audio": {{"data": "{encoded_audio}"}}}}',
        result,
    )

    assert event_type == "audio"
    assert result.status == "ok"
    assert result.success is True
    assert result.capability == "pass"
    assert result.audio_bytes == len(b"audio-bytes")


@pytest.mark.asyncio
async def test_websocket_script_exercises_stateful_sequence() -> None:
    received: list[dict] = []

    async def handle_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = msg.json()
                received.append(payload)
                if payload.get("type") == "input.done":
                    await ws.send_json({"type": "audio.start"})
                    await ws.send_bytes(b"pcm")
                    await ws.send_json({"type": "audio.done"})
                    await ws.send_json({"type": "session.done"})
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/speech/stream", handle_ws)
    runner, base_url = await _start_app(app)
    try:
        spec = BenchmarkSpec.from_obj(
            {
                "base_url": base_url,
                "model_name": "higgs",
                "params": {"timeout_s": 5},
            }
        )
        scenario = next(
            scenario for scenario in build_scenarios(spec) if scenario.method == "WS"
        )
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await run_ws_scenario(session, spec, scenario)
    finally:
        await runner.cleanup()

    assert [payload["type"] for payload in received] == [
        "session.config",
        "input.text",
        "input.done",
    ]
    assert result.status == "ok"
    assert result.success is True
    assert result.ws_event_counts["binary"] == 1


@pytest.mark.asyncio
async def test_runner_burst_stage_records_queue_wait_with_fake_server() -> None:
    async def handle_speech(request: web.Request) -> web.Response:
        payload = await request.json()
        await asyncio.sleep(0.02)
        if (
            not isinstance(payload.get("input"), str)
            or not payload.get("input", "").strip()
            or payload.get("ref_audio")
        ):
            return web.json_response({"error": "bad request"}, status=400)
        return web.Response(body=_wav_bytes(), content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handle_speech)
    runner, base_url = await _start_app(app)
    try:
        spec = BenchmarkSpec.from_obj(
            {
                "base_url": base_url,
                "model_name": "higgs",
                "params": {
                    "profile": "stress",
                    "enabled_endpoints": ["speech"],
                    "load_stages": [
                        {
                            "id": "burst-serial",
                            "mode": "burst",
                            "request_count": 4,
                            "max_concurrency": 1,
                        }
                    ],
                    "timeout_s": 5,
                },
            }
        )
        scenarios = build_scenarios(spec)
        harness_log: list[str] = []
        results = await _run_benchmark(spec, scenarios, harness_log)
    finally:
        await runner.cleanup()

    report = build_results_report(spec, results, scenarios=scenarios)

    assert len(results) == 4
    assert report["overall"]["passed"] is True
    assert report["metrics"]["by_stage"]["burst-serial"]["total"] == 4
    assert max(result.queue_wait_s or 0.0 for result in results) > 0.0
    assert "stage=burst-serial mode=burst completed 4 scenarios" in harness_log[-1]
