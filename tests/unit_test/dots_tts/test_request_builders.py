# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.dots_tts.native_adapter import DotsTTSPreparedInputs
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.request_builders import (
    DotsTTSSGLangRequestData,
    build_sglang_dots_tts_request,
    make_dots_tts_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


class FakeAdapter:
    llm_vocab_size = 32000
    model = type("FakeModel", (), {"config": type("FakeConfig", (), {})()})()

    def prepare_inputs(self, state):
        assert state.text == "Hello."
        return DotsTTSPreparedInputs(
            raw_inputs={"ok": True, "prefill_end": 2, "audio_placeholder_ids": {99}},
            input_ids=torch.tensor([[11, 12]]),
            generation_schedule=torch.tensor([[11, 12, 99, 99]]),
            audio_span_positions=torch.tensor([2]),
            prefill_end=2,
            audio_placeholder_ids={99},
            prompt_patch_embeddings=torch.ones(1, 2, 128),
            prompt_conditioning={"speaker": "x"},
        )


def test_build_sglang_dots_tts_request_from_payload() -> None:
    state = DotsTTSState(
        text="Hello.",
        template_name="text_to_audio",
        max_generate_length=8,
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "Hello."}, params={}),
        data=state.to_dict(),
    )

    data = build_sglang_dots_tts_request(payload, adapter=FakeAdapter())

    assert isinstance(data, DotsTTSSGLangRequestData)
    assert data.stage_payload is payload
    assert data.state.text == "Hello."
    assert data.input_ids.tolist() == [[11, 12]]
    assert data.generation_schedule.tolist() == [[11, 12, 99, 99]]
    assert data.span_positions.tolist() == [2]
    assert data.prefill_end == 2
    assert data.position == 2
    assert data.audio_placeholder_ids == {99}
    assert data.prompt_conditioning == {"speaker": "x"}
    assert data.prefill_input_embeds.shape == (1, 2, 128)
    assert data.raw_native_inputs["ok"] is True
    assert data.max_generate_length == 8
    assert data.req.rid == "rid"
    assert data.req.origin_input_ids == [11, 12]
    assert data.req.sampling_params.max_new_tokens == 8
    assert data.req.sampling_params.stop_token_ids is None
    assert data.req.eos_token_ids == set()
    assert data.req.vocab_size == 32000
    assert data.control_token_id == 0
    assert data.output_ids is data.req.output_ids


def test_make_scheduler_adapters_round_trip_payload() -> None:
    state = DotsTTSState(
        text="Hello.",
        template_name="text_to_audio",
        max_generate_length=4,
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "Hello."}, params={}),
        data=state.to_dict(),
    )
    request_builder, result_adapter = make_dots_tts_scheduler_adapters(
        adapter=FakeAdapter()
    )

    data = request_builder(payload)
    data.latent_patches.extend([torch.ones(1, 4, 128), torch.zeros(1, 4, 128)])
    result = result_adapter(data)

    assert result is payload
    assert result.data["modality"] == "audio_latents"
    assert len(result.data["latent_patches"]) == 2
    assert result.data["state"]["text"] == "Hello."


def test_build_request_uses_prepared_audio_span_count_as_default_limit() -> None:
    class ShortScheduleAdapter(FakeAdapter):
        def prepare_inputs(self, state):
            del state
            return DotsTTSPreparedInputs(
                raw_inputs={"ok": True},
                input_ids=torch.tensor([[11, 12]]),
                generation_schedule=torch.tensor([[11, 12, 99, 99, 99, 99]]),
                audio_span_positions=torch.tensor([2, 3, 4, 5]),
                prefill_end=2,
                audio_placeholder_ids={99},
            )

    state = DotsTTSState(text="Hello.", template_name="text_to_audio")
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "Hello."}, params={}),
        data=state.to_dict(),
    )

    data = build_sglang_dots_tts_request(payload, adapter=ShortScheduleAdapter())

    assert data.req.sampling_params.max_new_tokens == 4
    assert data.max_generate_length == 4


def test_build_request_rejects_interleave_template() -> None:
    state = DotsTTSState(
        text="Hello.",
        template_name="tts_interleave",
        max_generate_length=4,
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "Hello."}, params={}),
        data=state.to_dict(),
    )

    import pytest

    with pytest.raises(NotImplementedError, match="interleave"):
        build_sglang_dots_tts_request(payload, adapter=FakeAdapter())
