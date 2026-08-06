# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.dots_tts.codec import DotsReferenceEncoder
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
    _DotsStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(
    state: DotsTTSState,
    *,
    request_id: str = "rid",
    stream: bool = False,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"text": "target"}, params={"stream": stream}),
        data=state.to_dict(),
    )


def _drain(scheduler: DotsTTSStreamingVocoder) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _waveform(data: dict) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    return np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(
        data["audio_waveform_shape"]
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


def test_nonstream_vocoder_batches_equal_lengths_in_request_order() -> None:
    class _Inference:
        def __init__(self) -> None:
            self.decode_shapes: list[tuple[int, ...]] = []

        def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
            self.decode_shapes.append(tuple(latents.shape))
            identifiers = latents[:, 0, 0]
            return identifiers[:, None, None].expand(-1, 1, 3).contiguous()

    inference = _Inference()
    codec = SimpleNamespace(
        sample_rate=48000,
        patch_size=2,
        latent_dim=1,
        device=torch.device("cpu"),
        lock=threading.RLock(),
        inference=inference,
    )
    scheduler = DotsTTSStreamingVocoder(
        codec,
        optimize=False,
        max_batch_size=8,
        max_batch_wait_ms=2,
    )
    frame_counts = (4, 8, 4)
    payloads = [
        _payload(
            DotsTTSState(
                generated_latents=torch.full((1, frames, 1), float(index + 1))
            ),
            request_id=f"r{index}",
        )
        for index, frames in enumerate(frame_counts)
    ]

    assert scheduler._batch_fn is not None
    results = asyncio.run(scheduler._batch_fn(payloads))

    assert inference.decode_shapes == [(2, 4, 1), (1, 8, 1)]
    assert scheduler._max_batch_size == 8
    assert [result.request_id for result in results] == ["r0", "r1", "r2"]
    for index, result in enumerate(results):
        np.testing.assert_array_equal(
            _waveform(result.data),
            np.full(3, index + 1, dtype=np.float32),
        )


def test_streaming_vocoder_preserves_latents_across_boundaries() -> None:
    class _Inference:
        def __init__(self) -> None:
            self.init_chunk_sizes: list[int] = []
            self.stream_calls: list[tuple[int, bool, bool]] = []

        def init_stream_state(self, *, batch_size: int, chunk_size: int) -> object:
            assert batch_size == 1
            self.init_chunk_sizes.append(chunk_size)
            return object()

        def stream_step(
            self,
            latents: torch.Tensor,
            *,
            stream_state: object,
            optimize: bool,
            use_compiled: bool,
        ) -> torch.Tensor:
            assert stream_state is not None
            self.stream_calls.append((int(latents.shape[1]), optimize, use_compiled))
            return latents.flatten(start_dim=1)

        @staticmethod
        def flush(stream_state: object) -> torch.Tensor:
            assert stream_state is not None
            return torch.tensor([[999.0]])

        @staticmethod
        def decode_latents(latents: torch.Tensor) -> torch.Tensor:
            tail = torch.full((latents.shape[0], 1), 999.0)
            return torch.cat((latents.flatten(start_dim=1), tail), dim=1)

    inference = _Inference()
    codec = SimpleNamespace(
        sample_rate=48000,
        patch_size=2,
        latent_dim=1,
        device=torch.device("cpu"),
        lock=threading.RLock(),
        inference=inference,
    )
    scheduler = DotsTTSStreamingVocoder(codec, optimize=True, merge_steps=3)
    patches = [
        torch.full((1, codec.patch_size, codec.latent_dim), float(index))
        for index in range(1, 7)
    ]
    for chunk_id, patch in enumerate(patches):
        scheduler._on_chunk(
            "rid",
            StreamItem(
                chunk_id=chunk_id,
                data=patch,
                from_stage="tts_engine",
                metadata={"stream": True, "modality": "audio_latents"},
            ),
        )
    scheduler._on_done("rid")
    scheduler._on_streaming_new_request(
        "rid",
        _payload(
            DotsTTSState(generated_latents=torch.cat(patches, dim=1)),
            stream=True,
        ),
    )

    messages = _drain(scheduler)
    stream_audio = np.concatenate(
        [_waveform(message.data) for message in messages if message.type == "stream"]
    )
    expected = inference.decode_latents(torch.cat(patches, dim=1)).numpy().reshape(-1)

    assert inference.init_chunk_sizes == [6]
    assert inference.stream_calls == [
        (2, True, True),
        (2, True, True),
        (6, True, True),
        (2, True, False),
    ]
    np.testing.assert_array_equal(stream_audio, expected)
    assert [message.type for message in messages] == [
        "stream",
        "stream",
        "stream",
        "stream",
        "result",
    ]
    assert scheduler._stream_states == {}
