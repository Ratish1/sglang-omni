# SPDX-License-Identifier: Apache-2.0
"""dots.tts AudioVAE adapters for Omni's shared vocoder schedulers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.payload_types import DotsTTSState, load_dots_tts_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


class DotsTTSBatchVocoder(BatchVocoderBase):
    def __init__(self, codec: DotsAudioCodec) -> None:
        self.codec = codec

    def prepare_item(self, payload: StagePayload) -> tuple[DotsTTSState, torch.Tensor]:
        state = load_dots_tts_state(payload)
        if state.generated_latents is None:
            raise RuntimeError("dots.tts vocoder received no generated latents")
        return state, state.generated_latents

    async def decode_batch(
        self, items: list[tuple[DotsTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        if not items:
            return []
        groups: dict[int, list[int]] = defaultdict(list)
        for item_index, (_state, latents) in enumerate(items):
            groups[int(latents.size(1))].append(item_index)
        outputs: dict[int, tuple[torch.Tensor, int]] = {}
        with self.codec.lock:
            for indices in groups.values():
                batch_latents = torch.cat(
                    [items[index][1] for index in indices],
                    dim=0,
                ).to(self.codec.device)
                waveforms = self.codec.inference.decode_latents(batch_latents)
                for row, item_index in enumerate(indices):
                    outputs[item_index] = (
                        waveforms[row : row + 1],
                        self.codec.sample_rate,
                    )
        return [outputs[index] for index in range(len(items))]

    async def decode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self.prepare_item(payload) for payload in payloads]
        results = await self.decode_batch(items)
        return [
            self.store_result(payload, state, waveform, sample_rate)
            for payload, (state, _latents), (waveform, sample_rate) in zip(
                payloads,
                items,
                results,
                strict=True,
            )
        ]

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="dots.tts",
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class _DotsStreamState:
    codec_state: Any
    pending: list[torch.Tensor] = field(default_factory=list)
    received_patches: int = 0


class DotsTTSStreamingVocoder(StreamingVocoderBase[_DotsStreamState, None]):
    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        optimize: bool,
        merge_steps: int = 4,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if merge_steps < 1:
            raise ValueError("dots.tts vocoder merge_steps must be positive")
        if max_batch_size < 1:
            raise ValueError("dots.tts vocoder max_batch_size must be positive")
        if max_batch_wait_ms < 0:
            raise ValueError("dots.tts vocoder max_batch_wait_ms must be non-negative")
        self.codec = codec
        self.optimize = bool(optimize)
        self.merge_steps = int(merge_steps) if optimize else 1
        self._batch_vocoder = DotsTTSBatchVocoder(codec)
        super().__init__(
            self._batch_vocoder.decode_payload,
            batch_compute_fn=self._batch_vocoder.decode_payloads,
            sample_rate=codec.sample_rate,
            stream_source_hint="dots.tts",
            stream_input_modality="audio_latents",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def warmup_now(self) -> None:
        if not self.optimize or self.codec.device.type != "cuda":
            return
        steady_frames = self.codec.patch_size * self.merge_steps
        chunk_frames = tuple(dict.fromkeys((self.codec.patch_size, steady_frames)))
        with self.codec.lock:
            for frames in chunk_frames:
                codec_state = self.codec.inference.init_stream_state(
                    batch_size=1,
                    chunk_size=steady_frames,
                )
                latents = torch.zeros(
                    (1, frames, self.codec.latent_dim),
                    device=self.codec.device,
                    dtype=torch.float32,
                )
                self.codec.inference.stream_step(
                    latents,
                    stream_state=codec_state,
                    optimize=True,
                    use_compiled=True,
                )
            torch.cuda.synchronize(self.codec.device)

    def create_stream_state(self, request_id: str) -> _DotsStreamState:
        del request_id
        with self.codec.lock:
            codec_state = self.codec.inference.init_stream_state(
                batch_size=1,
                chunk_size=self.codec.patch_size * self.merge_steps,
            )
        return _DotsStreamState(codec_state=codec_state)

    def validate_chunk(
        self,
        request_id: str,
        state: _DotsStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "dots.tts latent chunks must have shape [1, frames, latent_dim]"
            )
        if codes.shape[-1] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {codes.shape[-1]}"
            )
        return codes.to(self.codec.device)

    def ingest(
        self, request_id: str, state: _DotsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.received_patches += 1

    def should_decode(self, state: _DotsStreamState, *, is_final: bool) -> bool:
        if is_final:
            return bool(state.pending)
        if state.received_patches <= 2:
            return bool(state.pending)
        return len(state.pending) >= self.merge_steps

    def decode_delta(
        self, request_id: str, state: _DotsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        chunks: list[torch.Tensor] = []
        if state.pending:
            take = (
                len(state.pending)
                if is_final
                else (1 if state.received_patches <= 2 else self.merge_steps)
            )
            patches, state.pending = state.pending[:take], state.pending[take:]
            with self.codec.lock:
                chunk = self.codec.inference.stream_step(
                    torch.cat(patches, dim=1),
                    stream_state=state.codec_state,
                    optimize=self.optimize,
                    use_compiled=not is_final,
                )
            if chunk.numel():
                chunks.append(chunk)
        if is_final:
            with self.codec.lock:
                tail = self.codec.inference.flush(state.codec_state)
            if tail.numel():
                chunks.append(tail)
        if not chunks:
            return None
        return torch.cat(chunks, dim=-1)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> dict[str, Any]:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _DotsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        if tts_state.generated_latents is None:
            return None
        with self.codec.lock:
            return self.codec.inference.decode_latents(
                tts_state.generated_latents.to(self.codec.device)
            )


__all__ = ["DotsTTSBatchVocoder", "DotsTTSStreamingVocoder"]
