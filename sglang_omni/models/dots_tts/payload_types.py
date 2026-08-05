# SPDX-License-Identifier: Apache-2.0
"""Payload state for the dots.tts wrapper pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class DotsTTSState(DeclarativeStateBase):
    """Normalized per-request inputs for the native dots TTS pipeline."""

    sample_rate: int = wire(48000, codec="int")
    text: str = wire("", codec="str")
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    template_name: str | None = None
    language: str | None = None
    speaker_scale: float = wire(1.5, codec="float")
    ode_method: str = wire("euler", codec="str_or")
    num_steps: int = wire(10, codec="int_or")
    guidance_scale: float = wire(1.2, codec="float")
    normalize_text: bool = wire(False, codec="bool")
    profile_inference: bool = wire(False, codec="bool")
    max_generate_length: int | None = wire(None, codec="opt_int")
    seed: int | None = wire(None, codec="opt_int")
    stream: bool = wire(False, codec="bool")
    rtf: float | None = wire(None, codec="float")
