# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for dots.tts.

dots.tts generates continuous audio latents before AudioVAE turns those
latents into waveform samples.  The pipeline keeps that boundary visible:
``latent_engine`` owns the autoregressive/flow latent path and ``vocoder`` owns
AudioVAE decode.  This is still not a Qwen2/SGLang KV-cache rewrite; it is the
native Omni stage split around dots.tts' own generation primitives.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.dots_tts"


class DotsTTSPipelineConfig(PipelineConfig):
    """3-stage dots.tts pipeline: preprocessing -> latent_engine -> vocoder."""

    architecture: ClassVar[str] = "DotsTTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "dots_tts",
        "DotsTtsModel",
    )

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="latent_engine",
        ),
        StageConfig(
            name="latent_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_latent_engine_executor",
            factory_args={"device": "cuda", "precision": "bfloat16"},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda", "precision": "bfloat16"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


EntryClass = DotsTTSPipelineConfig
