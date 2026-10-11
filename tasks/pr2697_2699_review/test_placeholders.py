import asyncio

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    WhisperFeatureExtractor,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error


@pytest.fixture
def preprocessor() -> Qwen3OmniPreprocessor:
    tokens = {
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    tokenizer = Qwen2Tokenizer(extra_special_tokens=tokens)
    template = (
        "{% for message in messages %}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for part in message['content'] %}"
        "{% if part['type'] == 'audio' %}<|audio_start|><|audio_pad|><|audio_end|>"
        "{% elif part['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ part['text'] }}{% endif %}"
        "{% endfor %}{% endif %}{% endfor %}"
    )
    preprocessor = object.__new__(Qwen3OmniPreprocessor)
    preprocessor.processor = Qwen3OmniMoeProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(feature_size=128),
        chat_template=template,
    )
    preprocessor.max_seq_len = None
    preprocessor.default_video_fps = None
    preprocessor.default_video_max_frames = None
    preprocessor.default_video_min_pixels = None
    preprocessor.default_video_max_pixels = None
    preprocessor.default_video_total_pixels = None
    return preprocessor


@pytest.mark.parametrize(
    "marker",
    [
        "<|audio_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_start|><|audio_pad|><|audio_end|>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<|vision_start|><|video_pad|><|vision_end|>",
    ],
)
def test_literal_marker_retains_tokenizer_ids(
    preprocessor: Qwen3OmniPreprocessor, marker: str
) -> None:
    text = f"Explain {marker} as text."
    expected = preprocessor.processor.tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    )
    payload = StagePayload(
        request_id="literal",
        request=OmniRequest(inputs={"messages": [{"role": "user", "content": text}]}),
        data={},
    )
    state = Qwen3OmniPipelineState.from_dict(
        asyncio.run(preprocessor.call_impl(payload)).data
    )
    assert torch.equal(state.prompt["input_ids"], expected["input_ids"][0])
    assert torch.equal(state.prompt["attention_mask"], expected["attention_mask"][0])
    assert state.encoder_inputs["audio_encoder"]["_skip"] is True
    assert state.encoder_inputs["image_encoder"]["_skip"] is True


@pytest.mark.parametrize("media_kind", ["audio", "image"])
@pytest.mark.parametrize("extra_marker", [False, True])
def test_real_media_cardinality(
    preprocessor: Qwen3OmniPreprocessor, media_kind: str, extra_marker: bool
) -> None:
    text = f"Explain <|{media_kind}_pad|>" if extra_marker else "Explain this input"
    inputs = {"messages": [{"role": "user", "content": text}]}
    if media_kind == "audio":
        inputs["audios"] = [np.sin(np.arange(3200, dtype=np.float32) / 8)]
    else:
        inputs["images"] = [Image.new("RGB", (28, 28))]
    payload = StagePayload(
        request_id="with-media", request=OmniRequest(inputs=inputs), data={}
    )
    if extra_marker:
        with pytest.raises(
            ValueError, match=f"2 <\\|{media_kind}_pad\\|> for 1 {media_kind} inputs"
        ) as failure:
            asyncio.run(preprocessor.call_impl(payload))
        assert is_bad_request_error(failure.value)
    else:
        state = Qwen3OmniPipelineState.from_dict(
            asyncio.run(preprocessor.call_impl(payload)).data
        )
        encoder_name = "audio_encoder" if media_kind == "audio" else "image_encoder"
        assert state.encoder_inputs[encoder_name].get("_skip") is not True
