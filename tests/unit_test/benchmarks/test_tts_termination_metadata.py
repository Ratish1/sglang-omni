from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.performance import build_speed_results
from benchmarks.tasks.tts import (
    _handle_raw_pcm_streaming_response,
    _parse_response_headers,
    _request_result_to_generated_entry,
)


def test_speech_headers_record_worker_finish_reason() -> None:
    result = RequestResult(request_id="sample")

    _parse_response_headers(
        result,
        {
            "X-Prompt-Tokens": "7",
            "X-Completion-Tokens": "2048",
            "X-Engine-Time": "1.25",
            "X-Finish-Reason": "length",
        },
    )

    assert result.finish_reason == "length"
    assert result.completion_tokens == 2048


def test_finish_reason_is_optional_in_saved_artifacts() -> None:
    result = RequestResult(request_id="sample", is_success=True, finish_reason="stop")
    speed_results = build_speed_results([result], {}, {})
    sample = SampleInput(
        sample_id="sample",
        ref_text="reference",
        ref_audio="reference.wav",
        target_text="hello",
    )

    assert speed_results["per_request"][0]["finish_reason"] == "stop"
    assert _request_result_to_generated_entry(result, sample)["finish_reason"] == "stop"
    assert RequestResult(request_id="old-artifact").finish_reason is None


@pytest.mark.asyncio
async def test_streaming_speech_records_finish_reason() -> None:
    class Content:
        async def iter_chunks(self):
            yield b"\x00\x00", True

    response = SimpleNamespace(
        headers={
            "Content-Type": "audio/pcm",
            "x-sample-rate": "24000",
            "x-channels": "1",
            "x-bit-depth": "16",
            "X-Finish-Reason": "stop",
        },
        content=Content(),
    )
    result = RequestResult(request_id="sample")

    await _handle_raw_pcm_streaming_response(
        response,
        result,
        start_time=0.0,
        save_audio_dir=None,
    )

    assert result.is_success
    assert result.finish_reason == "stop"
