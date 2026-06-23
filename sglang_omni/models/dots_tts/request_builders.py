# SPDX-License-Identifier: Apache-2.0
"""SGLang request helpers for the native dots TTS latent engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import torch

from sglang_omni.models.dots_tts.payload_types import DotsTTSState

if TYPE_CHECKING:
    from sglang_omni.models.dots_tts.native_adapter import DotsTTSPreparedInputs
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class DotsTTSSGLangRequestData(SGLangARRequestData):
    """Per-request state for SGLang-backed dots latent generation."""

    state: DotsTTSState = field(default_factory=DotsTTSState)
    generation_schedule: torch.Tensor | None = None
    position: int = 0
    span_positions: torch.Tensor | None = None
    prefill_end: int = 0
    audio_placeholder_ids: set[int] = field(default_factory=set)
    prompt_conditioning: Any = None
    fm_state: Any = None
    latest_latent_patch: torch.Tensor | None = None
    latent_patches: list[torch.Tensor] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    stream_metadata: dict[str, Any] | None = None
    chunk_id: int = 0
    control_token_id: int = 0
    max_generate_length: int = 500
    input_ids: torch.Tensor | None = None
    raw_native_inputs: dict[str, Any] = field(default_factory=dict)
    latest_hidden_state: torch.Tensor | None = None
    eos_score: torch.Tensor | None = None
    finish_reason: str | None = None


def build_sglang_dots_tts_request(
    payload: StagePayload,
    *,
    adapter: Any,
) -> DotsTTSSGLangRequestData:
    """Build per-request dots data before constructing the SGLang Req."""

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = DotsTTSState.from_dict(payload.data)
    if state.template_name == "tts_interleave":
        raise NotImplementedError(
            "dots TTS SGLang-native v1 does not support text/audio interleave"
        )
    prepared = adapter.prepare_inputs(state)
    input_ids_tensor = _prefill_input_ids(prepared)
    if input_ids_tensor is None:
        input_ids_list: list[int] = []
    else:
        input_ids_list = (
            input_ids_tensor.reshape(-1).detach().cpu().to(dtype=torch.long).tolist()
        )
    cfg = adapter.model.config
    control_token_id = 0
    vocab_size = int(
        getattr(cfg, "vocab_size", max(input_ids_list, default=control_token_id) + 1)
    )
    max_generate_length = _resolve_max_generate_length(state, prepared)
    sampling_params = SamplingParams(
        max_new_tokens=max_generate_length,
        temperature=0.0,
        stop_token_ids=[],
    )
    sampling_params.normalize(None)
    sampling_params.verify(vocab_size)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=vocab_size,
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None
    return DotsTTSSGLangRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=prepared.input_ids,
        generation_schedule=prepared.generation_schedule,
        span_positions=prepared.audio_span_positions,
        prefill_end=_resolve_prefill_end(prepared),
        position=_resolve_prefill_end(prepared),
        audio_placeholder_ids=set(prepared.audio_placeholder_ids),
        prompt_conditioning=prepared.prompt_conditioning,
        fm_state=prepared.fm_state,
        generation_kwargs=dict(prepared.generation_kwargs),
        prefill_input_embeds=prepared.prompt_patch_embeddings,
        raw_native_inputs=dict(prepared.raw_inputs),
        max_generate_length=max_generate_length,
        stream_metadata={"modality": "audio_latents"},
        control_token_id=control_token_id,
    )


def _resolve_prefill_end(prepared: DotsTTSPreparedInputs) -> int:
    if prepared.prefill_end is not None:
        return int(prepared.prefill_end)
    if prepared.input_ids is not None:
        return int(prepared.input_ids.reshape(-1).numel())
    if prepared.generation_schedule is not None:
        return int(prepared.generation_schedule.reshape(-1).numel())
    return 0


def _prefill_input_ids(prepared: DotsTTSPreparedInputs) -> torch.Tensor | None:
    generation_schedule = prepared.generation_schedule
    prefill_end = _resolve_prefill_end(prepared)
    if generation_schedule is not None and prefill_end > 0:
        if generation_schedule.ndim == 1:
            return generation_schedule[:prefill_end].unsqueeze(0)
        return generation_schedule[:, :prefill_end]
    return prepared.input_ids


def _resolve_max_generate_length(
    state: DotsTTSState, prepared: DotsTTSPreparedInputs
) -> int:
    if state.max_generate_length is not None:
        return int(state.max_generate_length)
    span_positions = prepared.audio_span_positions
    if span_positions is not None:
        prefill_end = _resolve_prefill_end(prepared)
        generated_spans = (span_positions >= prefill_end).sum().item()
        if generated_spans > 0:
            return int(generated_spans)
    return 500


def build_stream_output(
    request_id: str,
    data: DotsTTSSGLangRequestData,
    req_output: Any,
) -> Iterator[OutgoingMessage]:
    """Emit the latest latent patch through Omni's stream side channel."""

    del req_output
    latent_patch = data.latest_latent_patch
    if latent_patch is None:
        return
    metadata = dict(data.stream_metadata or {})
    metadata.setdefault("modality", "audio_latents")
    metadata.setdefault("chunk_id", data.chunk_id)
    data.chunk_id += 1
    data.latest_latent_patch = None
    yield OutgoingMessage(
        request_id=request_id,
        type="stream",
        target="vocoder",
        data=latent_patch,
        metadata=metadata,
    )


def apply_latent_result(
    payload: StagePayload,
    data: DotsTTSSGLangRequestData,
) -> StagePayload:
    """Store collected latent patches back into the StagePayload."""

    state = data.state
    payload.data = {
        "modality": "audio_latents",
        "latent_patches": list(data.latent_patches),
        "state": state.to_dict(),
    }
    return payload


def make_dots_tts_scheduler_adapters(*, adapter: Any):
    """Build request/result adapter closures for ``OmniScheduler``."""

    def request_builder(payload: StagePayload) -> DotsTTSSGLangRequestData:
        return build_sglang_dots_tts_request(payload, adapter=adapter)

    def result_adapter(data: DotsTTSSGLangRequestData) -> StagePayload:
        return apply_latent_result(data.stage_payload, data)

    return request_builder, result_adapter


__all__ = [
    "DotsTTSSGLangRequestData",
    "apply_latent_result",
    "build_sglang_dots_tts_request",
    "build_stream_output",
    "make_dots_tts_scheduler_adapters",
]
