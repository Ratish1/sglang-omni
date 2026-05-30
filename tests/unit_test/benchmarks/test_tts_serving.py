from __future__ import annotations

import asyncio
import base64

import pytest

from benchmarks.tts_serving.http_client import _handle_binary_response
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.scenarios import Scenario, build_scenarios
from benchmarks.tts_serving.spec import BenchmarkSpec, SpecError
from benchmarks.tts_serving.ws_client import _merge_text_event


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


def _result_for(scenario: Scenario) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        expected_success=scenario.expect_success,
    )


def test_tts_serving_spec_rejects_smoke_profile() -> None:
    with pytest.raises(SpecError, match="params.profile"):
        BenchmarkSpec.from_obj(
            {
                "base_url": "http://localhost:8000",
                "model_name": "higgs",
                "params": {"profile": "smoke"},
            }
        )


def test_report_separates_traffic_requests_from_capability_probes() -> None:
    spec = BenchmarkSpec.from_obj(
        {
            "base_url": "http://localhost:8000",
            "model_name": "higgs",
            "params": {
                "profile": "ci",
                "total_requests": 3,
                "enabled_endpoints": ["voices", "websocket"],
            },
        }
    )

    results = [
        ScenarioResult(
            scenario_id=scenario.id,
            endpoint=scenario.endpoint,
            category=scenario.category,
            status="ok",
            success=True,
        )
        for scenario in build_scenarios(spec)
    ]
    report = build_results_report(spec, results)

    assert report["overall"]["traffic_total"] == 3
    assert report["overall"]["capability_probe_total"] == 2
    assert report["overall"]["total"] == 5


def test_http_speech_2xx_without_audio_is_invalid() -> None:
    scenario = Scenario(
        id="speech-valid",
        endpoint="speech",
        category="well_formed",
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
        category="malformed",
        payload={"response_format": "wav"},
        expect_success=False,
    )
    result = _result_for(scenario)
    response = _FakeResponse(
        status=200,
        body=b"RIFF" + b"\0" * 44,
        headers={"Content-Type": "audio/wav"},
    )

    asyncio.run(_handle_binary_response(response, result, scenario))

    assert result.status == "unexpected_success"
    assert result.success is False


def test_websocket_error_text_event_fails_capability_probe() -> None:
    result = ScenarioResult(
        scenario_id="ws-capability",
        endpoint="websocket",
        category="capability_probe",
    )

    done = _merge_text_event(
        '{"type": "error", "error": {"message": "bad request"}}',
        result,
    )

    assert done is True
    assert result.status == "failed"
    assert result.capability == "fail"


def test_websocket_control_event_waits_for_audio() -> None:
    result = ScenarioResult(
        scenario_id="ws-capability",
        endpoint="websocket",
        category="capability_probe",
    )

    done = _merge_text_event('{"type": "session.created"}', result)

    assert done is False
    assert result.status == "error"
    assert result.capability is None


def test_websocket_audio_event_passes_capability_probe() -> None:
    result = ScenarioResult(
        scenario_id="ws-capability",
        endpoint="websocket",
        category="capability_probe",
    )
    encoded_audio = base64.b64encode(b"audio-bytes").decode("ascii")

    done = _merge_text_event(
        f'{{"type": "response.audio.delta", "audio": {{"data": "{encoded_audio}"}}}}',
        result,
    )

    assert done is True
    assert result.status == "ok"
    assert result.success is True
    assert result.capability == "pass"
    assert result.audio_bytes == len(b"audio-bytes")


def test_websocket_unknown_text_event_fails_capability_probe() -> None:
    result = ScenarioResult(
        scenario_id="ws-capability",
        endpoint="websocket",
        category="capability_probe",
    )

    done = _merge_text_event('{"type": "unexpected.event"}', result)

    assert done is True
    assert result.status == "failed"
    assert result.capability == "fail"
