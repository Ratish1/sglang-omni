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
        self.precision = getattr(runtime, "precision", "bfloat16")

    def prepare_inputs(self, state: DotsTTSState) -> DotsTTSPreparedInputs:
        raw_inputs = self.runtime._prepare_inputs(
            text=state.text,
            prompt_audio_path=state.prompt_audio_path,
            prompt_text=state.prompt_text,
            template_name=state.template_name,
            language=state.language,
            normalize_text=state.normalize_text,
        )
        input_ids = raw_inputs.get("input_ids")
        generation_schedule = raw_inputs.get("generation_schedule")
        prompt_input_embeds = raw_inputs.get("prompt_input_embeds")
        prompt_conditioning = raw_inputs.get("prompt_conditioning")
        fm_state = raw_inputs.get("fm_state")
        audio_span_positions = raw_inputs.get("audio_span_positions")
        prefill_end = raw_inputs.get("prefill_end")
        audio_placeholder_ids = raw_inputs.get("audio_placeholder_ids") or set()
        if input_ids is None:
            input_ids = generation_schedule
        if self._can_prepare_native_state(raw_inputs):
            prepared = self._prepare_native_state(
                state,
                raw_inputs=raw_inputs,
                generation_schedule=generation_schedule,
            )
            input_ids = prepared["input_ids"]
            generation_schedule = prepared["generation_schedule"]
            prompt_input_embeds = prepared["prompt_input_embeds"]
            prompt_conditioning = prepared["prompt_conditioning"]
            fm_state = prepared["fm_state"]
            audio_span_positions = prepared["audio_span_positions"]
            prefill_end = prepared["prefill_end"]
            audio_placeholder_ids = prepared["audio_placeholder_ids"]
        return DotsTTSPreparedInputs(
            raw_inputs=raw_inputs,
            input_ids=input_ids,
            generation_schedule=generation_schedule,
            audio_span_positions=audio_span_positions,
            prefill_end=prefill_end,
            audio_placeholder_ids=set(audio_placeholder_ids),
            prompt_patch_embeddings=prompt_input_embeds,
            prompt_conditioning=prompt_conditioning,
            fm_state=fm_state,
            generation_kwargs=self._generation_kwargs(
                state,
                prompt_conditioning=prompt_conditioning,
                fm_state=fm_state,
            ),
        )

    def _can_prepare_native_state(self, raw_inputs: dict[str, Any]) -> bool:
        return (
            raw_inputs.get("generation_schedule") is not None
            and hasattr(self.model, "_allocate_generate_state")
            and hasattr(self.model, "_prepare_prompt_conditioning")
            and hasattr(self.model, "_find_audio_span_positions")
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
        prompt_patch_embeddings = self.model._prefill_prompt_latents(
            prompt_conditioning.prompt_latents,
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
        if fm_state is not None and getattr(fm_state, "fm_sequence", None) is not None:
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

    def generate_audio_step(
        self,
        *,
        hidden_state: torch.Tensor,
        fm_state: Any,
        generation_kwargs: dict[str, Any],
    ) -> DotsTTSAudioStepResult:
        decode_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key in {"device", "g_cond", "ode_method", "num_steps", "guidance_scale"}
        }
        if hidden_state is not None and hasattr(self.model, "_append_hidden_chunk"):
            self.model._append_hidden_chunk(fm_state, hidden_state)
        device = decode_kwargs.get("device")
        dtype = _torch_dtype(self.precision)
        use_amp = (
            isinstance(device, torch.device)
            and device.type == "cuda"
            and dtype in {torch.float16, torch.bfloat16}
        )
        with torch.autocast(
            device_type=device.type if isinstance(device, torch.device) else "cuda",
            dtype=dtype,
            enabled=use_amp,
        ):
            latent_patch = self.model._decode_next_audio(
                state=fm_state,
                **decode_kwargs,
            )
            latent_patch = _as_tensor(latent_patch)
            feedback = getattr(self.model, "_encode_audio_patch_feedback", None)
            if feedback is not None:
                feedback_embedding = feedback(fm_state, audio_patch=latent_patch)
            else:
                feedback_embedding = self.model._encode_audio_patch(latent_patch)

        io_helper = getattr(getattr(self.model, "core", None), "io_helper", None)
        payload_patch = (
            io_helper.denormalize(latent_patch)
            if io_helper is not None
            else latent_patch
        )

        stop_predicate = getattr(self.model, "_should_stop_after_current_audio", None)
        if stop_predicate is not None:
            eos_threshold = float(generation_kwargs.get("eos_threshold", 0.8))
            eos_score = torch.tensor(
                [1.0 if stop_predicate(fm_state, eos_threshold=eos_threshold) else 0.0],
                device=latent_patch.device,
            )
        else:
            eos_predictor = getattr(self.model, "_predict_eos", None)
            eos_score = (
                eos_predictor(hidden_state, latent_patch)
                if eos_predictor is not None
                else None
            )
        return DotsTTSAudioStepResult(
            latent_patch=_as_tensor(payload_patch),
            feedback_embedding=_as_tensor(feedback_embedding),
            eos_score=_as_tensor(eos_score) if eos_score is not None else None,
        )


__all__ = [
    "DotsTTSAudioStepResult",
    "DotsTTSNativeAdapter",
    "DotsTTSPreparedInputs",
]
