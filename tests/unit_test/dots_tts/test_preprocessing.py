# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload
from sglang_omni.proto import OmniRequest, StagePayload


class _RecordingTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        from dots_tts.utils.tokenizer import (
            AUDIO_COMP_SPAN_TOKEN,
            AUDIO_GEN_SPAN_TOKEN,
            AUDIO_GEN_START_TOKEN,
        )

        self.encoded_text: list[str] = []
        self._tokens = {
            AUDIO_GEN_START_TOKEN: 101,
            AUDIO_GEN_SPAN_TOKEN: 102,
            AUDIO_COMP_SPAN_TOKEN: 103,
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_text.append(text)
        return [10] if text else []

    def decode(self, token_ids: list[int], **_kwargs) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens[token]

    def __len__(self) -> int:
        return 256


def _payload(**tts_params) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs={
                "input": "hello",
                "references": [
                    {
                        "audio_path": "data:audio/wav;base64,UklGRg==",
                        "text": "reference",
                    }
                ],
            },
            params={},
            metadata={"tts_params": tts_params},
        ),
    )


def _preprocess(payload: StagePayload, tokenizer: _RecordingTokenizer) -> DotsTTSState:
    result = preprocess_dots_tts_payload(
        payload,
        tokenizer=tokenizer,
        model_config=SimpleNamespace(
            patch_size=4,
            vocoder=SimpleNamespace(sample_rate=48000),
        ),
        max_generate_length=20,
        max_sequence_length=128,
    )
    return DotsTTSState.from_dict(result.data)


@pytest.mark.parametrize("language", ["Auto", "auto", "AUTO_DETECT"])
def test_preprocessing_maps_public_base_and_automatic_language(language: str) -> None:
    tokenizer = _RecordingTokenizer()

    state = _preprocess(
        _payload(task_type="Base", language=language, max_new_tokens=3),
        tokenizer,
    )

    assert state.max_new_tokens == 3
    assert state.prompt_audio_path == "data:audio/wav;base64,UklGRg=="
    assert state.use_prompt_prefill is True
    assert any("[EN]reference" in text for text in tokenizer.encoded_text)


def test_preprocessing_rejects_non_base_public_task_before_schedule_build() -> None:
    tokenizer = _RecordingTokenizer()

    with pytest.raises(ValueError, match="only task_type='Base'"):
        _preprocess(_payload(task_type="VoiceDesign"), tokenizer)

    assert tokenizer.encoded_text == []
