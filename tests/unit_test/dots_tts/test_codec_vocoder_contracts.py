# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.dots_tts.codec import DotsReferenceEncoder
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
    _DotsStreamState,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(state: DotsTTSState) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "target"}, params={}),
        data=state.to_dict(),
    )


def test_inline_reference_reuses_real_reference_service_cache() -> None:
    class _Codec:
        sample_rate = 48000
        patch_size = 4
        latent_dim = 16

        def __init__(self) -> None:
            self.encode_calls = 0

        def encode_reference(self, path: str) -> dict[str, torch.Tensor]:
            assert path == "data:audio/wav;base64,UklGRg=="
            self.encode_calls += 1
            return {
                "speaker_embedding": torch.arange(4, dtype=torch.float32),
                "latent_distribution": torch.arange(8, dtype=torch.float32).reshape(
                    1, 2, 4
                ),
            }

    codec = _Codec()
    encoder = DotsReferenceEncoder(codec, model_id="dots")
    state = DotsTTSState(
        prompt_audio_path="data:audio/wav;base64,UklGRg==",
        use_prompt_prefill=False,
    )

    first = DotsTTSState.from_dict(encoder.encode_payload(_payload(state)).data)
    second = DotsTTSState.from_dict(encoder.encode_payload(_payload(state)).data)

    assert codec.encode_calls == 1
    torch.testing.assert_close(first.speaker_embedding, second.speaker_embedding)
    assert first.speaker_embedding.data_ptr() != second.speaker_embedding.data_ptr()


def test_terminal_vocoder_results_exclude_internal_pipeline_state() -> None:
    codec = SimpleNamespace(sample_rate=48000)
    state = DotsTTSState(
        prompt_audio_path="data:audio/wav;base64,UklGRg==",
        prompt_latents=torch.ones(1, 4, 2),
        speaker_embedding=torch.ones(1, 8),
        generated_latents=torch.ones(1, 8, 2),
        generation_schedule=torch.arange(16).reshape(1, 16),
        completion_tokens=2,
    )
    batch_vocoder = DotsTTSBatchVocoder(codec)
    batch_result = batch_vocoder.store_result(
        _payload(state),
        state,
        torch.zeros(1, 64),
        48000,
    )
    streaming_vocoder = DotsTTSStreamingVocoder(codec, optimize=False)
    stream_result = streaming_vocoder.final_result_data(
        "rid",
        _payload(state),
        _DotsStreamState(codec_state=object()),
    )

    assert set(batch_result.data) == {
        "audio_waveform",
        "audio_waveform_shape",
        "audio_waveform_dtype",
        "sample_rate",
        "modality",
        "usage",
    }
    assert batch_result.data["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 2,
        "total_tokens": 2,
    }
    assert stream_result == {
        "modality": "audio",
        "sample_rate": 48000,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 2,
            "total_tokens": 2,
        },
    }
