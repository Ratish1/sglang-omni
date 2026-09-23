# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ThinkerModelRunner.lookahead_eligible."""

from __future__ import annotations

import types

import pytest

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.request_builders import should_generate_audio_output
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def _runner(output_processor: SGLangOutputProcessor) -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner.output_processor = output_processor
    return runner


def _hidden_capturing_processor() -> SGLangOutputProcessor:
    return SGLangOutputProcessor(
        capture_hidden=True,
        should_emit_hidden=lambda request_data: should_generate_audio_output(
            request_data.stage_payload
        ),
    )


def _sampling_params(**overrides):
    values = dict(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_new_tokens=0,
        sampling_seed=None,
        logit_bias=None,
        custom_params=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _request(output_modalities: list[str], return_logprob=False, **sampling):
    stage_payload = StagePayload(
        request_id="req",
        request=OmniRequest(
            inputs=[], params={}, metadata={"output_modalities": output_modalities}
        ),
        data={},
    )
    return types.SimpleNamespace(
        rid="req",
        sampling_params=_sampling_params(**sampling),
        _omni_data=SGLangARRequestData(
            stage_payload=stage_payload, return_logprob=return_logprob
        ),
    )


def _batch(*requests):
    return types.SimpleNamespace(reqs=list(requests))


def test_speech_batch_without_hidden_capture_is_eligible():
    batch = _batch(_request(["text", "audio"]), _request(["text"]))
    assert _runner(SGLangOutputProcessor()).lookahead_eligible(batch) is True


def test_request_that_emits_hidden_states_keeps_batch_synchronous():
    runner = _runner(_hidden_capturing_processor())
    assert runner.lookahead_eligible(_batch(_request(["text"]))) is True
    speech_batch = _batch(_request(["text"]), _request(["text", "audio"]))
    assert runner.lookahead_eligible(speech_batch) is False


def test_return_logprob_disables_lookahead():
    batch = _batch(_request(["text"], return_logprob=True))
    assert _runner(SGLangOutputProcessor()).lookahead_eligible(batch) is False


@pytest.mark.parametrize(
    "sampling",
    [
        dict(repetition_penalty=1.3),
        dict(presence_penalty=0.5),
        dict(frequency_penalty=0.5),
        dict(min_new_tokens=5),
        dict(sampling_seed=42),
        dict(logit_bias={1: 2.0}),
        dict(custom_params={"x": 1}),
    ],
)
def test_history_scored_or_unsupported_sampling_disables_lookahead(sampling):
    batch = _batch(_request(["text"]), _request(["text", "audio"], **sampling))
    assert _runner(SGLangOutputProcessor()).lookahead_eligible(batch) is False
