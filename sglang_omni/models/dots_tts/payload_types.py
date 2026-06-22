# SPDX-License-Identifier: Apache-2.0
"""Payload state for the dots.tts wrapper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DotsTTSState:
    """Normalized per-request inputs for ``DotsTtsRuntime.generate``."""

    text: str = ""
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    template_name: str | None = None
    language: str | None = None
    speaker_scale: float = 1.5
    ode_method: str = "euler"
    num_steps: int = 10
    guidance_scale: float = 1.2
    normalize_text: bool = False
    profile_inference: bool = False
    max_generate_length: int | None = None
    seed: int | None = None
    stream: bool = False
    sample_rate: int | None = None
    engine_time_s: float = 0.0
    rtf: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "speaker_scale": float(self.speaker_scale),
            "ode_method": self.ode_method,
            "num_steps": int(self.num_steps),
            "guidance_scale": float(self.guidance_scale),
            "normalize_text": bool(self.normalize_text),
            "profile_inference": bool(self.profile_inference),
            "stream": bool(self.stream),
        }
        if self.prompt_audio_path is not None:
            data["prompt_audio_path"] = self.prompt_audio_path
        if self.prompt_text is not None:
            data["prompt_text"] = self.prompt_text
        if self.template_name is not None:
            data["template_name"] = self.template_name
        if self.language is not None:
            data["language"] = self.language
        if self.max_generate_length is not None:
            data["max_generate_length"] = int(self.max_generate_length)
        if self.seed is not None:
            data["seed"] = int(self.seed)
        if self.sample_rate is not None:
            data["sample_rate"] = int(self.sample_rate)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        if self.rtf is not None:
            data["rtf"] = float(self.rtf)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "DotsTTSState":
        if not isinstance(data, dict):
            data = {}
        return cls(
            text=str(data.get("text", "") or ""),
            prompt_audio_path=data.get("prompt_audio_path"),
            prompt_text=data.get("prompt_text"),
            template_name=data.get("template_name"),
            language=data.get("language"),
            speaker_scale=float(data.get("speaker_scale", 1.5) or 1.5),
            ode_method=str(data.get("ode_method", "euler") or "euler"),
            num_steps=int(data.get("num_steps", 10) or 10),
            guidance_scale=float(data.get("guidance_scale", 1.2) or 1.2),
            normalize_text=bool(data.get("normalize_text", False)),
            profile_inference=bool(data.get("profile_inference", False)),
            max_generate_length=(
                int(data["max_generate_length"])
                if data.get("max_generate_length") is not None
                else None
            ),
            seed=(int(data["seed"]) if data.get("seed") is not None else None),
            stream=bool(data.get("stream", False)),
            sample_rate=(
                int(data["sample_rate"]) if data.get("sample_rate") is not None else None
            ),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
            rtf=(float(data["rtf"]) if data.get("rtf") is not None else None),
        )
