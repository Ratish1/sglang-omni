# SPDX-License-Identifier: Apache-2.0
"""Adapter from vendored dots runtime primitives to SGLang runner hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.dots_tts.payload_types import DotsTTSState


@dataclass
class DotsTTSPreparedInputs:
    """Prepared dots request inputs consumed by the SGLang request builder."""

    raw_inputs: dict[str, Any]
    input_ids: torch.Tensor | None = None
    generation_schedule: torch.Tensor | None = None
    audio_span_positions: torch.Tensor | None = None
    prefill_end: int | None = None
    audio_placeholder_ids: set[int] = field(default_factory=set)
    prompt_patch_embeddings: torch.Tensor | None = None
    prompt_conditioning: Any = None
    fm_state: Any = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DotsTTSAudioStepResult:
    """One dots audio decode step produced from a SGLang hidden state."""

    latent_patch: torch.Tensor
    feedback_embedding: torch.Tensor
    eos_score: torch.Tensor | None


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _torch_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    return torch.float32


class DotsTTSNativeAdapter:
    """Small boundary object around vendored dots inference primitives."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.precision = runtime.precision
        self.llm_vocab_size = int(self.model.llm_config.vocab_size)

    def prepare_inputs(self, state: DotsTTSState) -> DotsTTSPreparedInputs:
        requested_max_generate_length = state.max_generate_length
        runtime_max_generate_length = int(self.runtime.max_generate_length or 0)
        if (
            requested_max_generate_length is not None
            and runtime_max_generate_length > 0
            and int(requested_max_generate_length) > runtime_max_generate_length
        ):
            raise ValueError(
                "dots TTS request max_generate_length exceeds the latent engine "
                f"limit: requested={int(requested_max_generate_length)} "
                f"limit={runtime_max_generate_length}."
            )
        raw_inputs = self.runtime._prepare_inputs(
            text=state.text,
            prompt_audio_path=state.prompt_audio_path,
            prompt_text=state.prompt_text,
            template_name=state.template_name,
            language=state.language,
            normalize_text=state.normalize_text,
        )
        generation_schedule = raw_inputs.get("generation_schedule")
        if generation_schedule is None:
            raise RuntimeError(
                "dots TTS native runtime did not produce a generation_schedule; "
                "the SGLang-native latent path requires one."
            )
        prepared = self._prepare_native_state(
            state,
            raw_inputs=raw_inputs,
            generation_schedule=generation_schedule,
        )
        return DotsTTSPreparedInputs(
            raw_inputs=raw_inputs,
            input_ids=prepared["input_ids"],
            generation_schedule=prepared["generation_schedule"],
            audio_span_positions=prepared["audio_span_positions"],
            prefill_end=prepared["prefill_end"],
            audio_placeholder_ids=set(prepared["audio_placeholder_ids"]),
            prompt_patch_embeddings=prepared["prompt_input_embeds"],
            prompt_conditioning=prepared["prompt_conditioning"],
            fm_state=prepared["fm_state"],
            generation_kwargs=self._generation_kwargs(
                state,
                prompt_conditioning=prepared["prompt_conditioning"],
                fm_state=prepared["fm_state"],
            ),
        )

    def _prepare_native_state(
        self,
        state: DotsTTSState,
        *,
        raw_inputs: dict[str, Any],
        generation_schedule: torch.Tensor,
    ) -> dict[str, Any]:
        device = next(self.model.core.parameters()).device
        dtype = _torch_dtype(self.precision)
        generation_schedule = generation_schedule.to(device=device, dtype=torch.long)
        if generation_schedule.ndim == 1:
            generation_schedule = generation_schedule.unsqueeze(0)

        use_prompt_prefill = raw_inputs.get("prompt_audio") is not None and bool(
            raw_inputs.get("prompt_text")
        )
        prompt_conditioning = self.model._prepare_prompt_conditioning(
            raw_inputs.get("prompt_audio"),
            use_prompt_prefill=use_prompt_prefill,
            speaker_scale=state.speaker_scale,
        )
        prompt_patches = prompt_conditioning.prompt_patches
        prompt_patch_count = 0 if prompt_patches is None else int(prompt_patches.size(1))
        audio_placeholder_ids = set(self.model.core.audio_span_token_ids)
        span_positions = self.model._find_audio_span_positions(
            generation_schedule,
            audio_placeholder_ids=audio_placeholder_ids,
        )
        span_count = int(span_positions.numel())
        minimum_required_spans = prompt_patch_count + 1
        if span_count < minimum_required_spans:
            raise ValueError(
                f"generation_schedule provides {span_count} audio spans, but "
                f"prompt prefill requires {prompt_patch_count} spans and generation "
                "requires at least one additional decode span."
            )
        fm_state = self.model._allocate_generate_state(
            max_audio_patch_count=span_count,
            device=device,
            dtype=dtype,
        )
        prompt_latents = prompt_conditioning.prompt_latents
        if prompt_latents is not None:
            prompt_latents = prompt_latents.to(dtype=fm_state.fm_sequence.dtype)
        prompt_patch_embeddings = self.model._prefill_prompt_latents(
            prompt_latents,
            state=fm_state,
        )
        prefill_end, prompt_span_positions = self.model._locate_prefill_boundary(
            span_positions=span_positions,
            prompt_patch_count=prompt_patch_count,
        )
        input_ids = generation_schedule[:, :prefill_end]
        prompt_input_embeds = None
        if prompt_span_positions.numel() > 0:
            prompt_input_embeds = self.model._build_prefill_inputs_embeds(
                input_ids,
                prompt_patch_embeddings=prompt_patch_embeddings,
                prompt_span_positions=prompt_span_positions,
            )
        raw_inputs["prompt_conditioning"] = prompt_conditioning
        raw_inputs["fm_state"] = fm_state
        raw_inputs["audio_span_positions"] = span_positions
        raw_inputs["prefill_end"] = prefill_end
        raw_inputs["prompt_span_positions"] = prompt_span_positions
        raw_inputs["audio_placeholder_ids"] = audio_placeholder_ids
        return {
            "generation_schedule": generation_schedule,
            "input_ids": input_ids,
            "prompt_input_embeds": prompt_input_embeds,
            "prompt_conditioning": prompt_conditioning,
            "fm_state": fm_state,
            "audio_span_positions": span_positions,
            "prefill_end": prefill_end,
            "audio_placeholder_ids": audio_placeholder_ids,
        }

    def _generation_kwargs(
        self,
        state: DotsTTSState,
        *,
        prompt_conditioning: Any,
        fm_state: Any,
    ) -> dict[str, Any]:
        device = None
        if fm_state is not None and fm_state.fm_sequence is not None:
            device = fm_state.fm_sequence.device
        g_cond = getattr(prompt_conditioning, "g_cond", None)
        return {
            "device": device,
            "g_cond": g_cond,
            "ode_method": state.ode_method,
            "num_steps": state.num_steps,
            "guidance_scale": state.guidance_scale,
            "speaker_scale": state.speaker_scale,
            "eos_threshold": 0.8,
        }


__all__ = [
    "DotsTTSAudioStepResult",
    "DotsTTSNativeAdapter",
    "DotsTTSPreparedInputs",
]
