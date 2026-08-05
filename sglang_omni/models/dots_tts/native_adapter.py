# SPDX-License-Identifier: Apache-2.0
"""Adapter from vendored dots runtime primitives to SGLang runner hooks."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.serving_types import (
    DotsTTSAudioStepResult,
    DotsTTSPreparedInputs,
)


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
        return self.model.prepare_request(
            raw_inputs,
            state,
            generation_schedule=generation_schedule,
            precision=self.precision,
        )


__all__ = [
    "DotsTTSAudioStepResult",
    "DotsTTSNativeAdapter",
    "DotsTTSPreparedInputs",
]
