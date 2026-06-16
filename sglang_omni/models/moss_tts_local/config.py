# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.moss_tts_local"
_COLOCATED_TOTAL_GPU_MEMORY_FRACTION = 0.90
_COLOCATED_CODEC_MEM_RESERVE = 0.05


def _stages(*, codec_device: str) -> list[StageConfig]:
    colocated_codec = codec_device == "cuda:0"
    vocoder_factory_args = {"device": codec_device}
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "device": codec_device,
                "ref_audio_cache": True,
                "ref_audio_cache_max_items": 8192,
                "ref_audio_cache_max_bytes": 64 * 1024 * 1024,
            },
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={
                "gpu_id": 0,
                "dtype": "bfloat16",
                "codec_mem_reserve": (
                    _COLOCATED_CODEC_MEM_RESERVE if colocated_codec else 0.0
                ),
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=_COLOCATED_TOTAL_GPU_MEMORY_FRACTION
                ),
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args=vocoder_factory_args,
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSLocalPipelineConfig(PipelineConfig):
    """Single-GPU MOSS-TTS Local pipeline."""

    architecture: ClassVar[str] = "MossTTSLocalModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSLocal",
        "MossTTSLocalForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0")
    )

    def supports_uploaded_voice_references(self) -> bool:
        return True


class MossTTSLocalColocatedPipelineConfig(MossTTSLocalPipelineConfig):
    """Backward-compatible alias for the default single-GPU pipeline."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0")
    )


class MossTTSLocalSplitPipelineConfig(MossTTSLocalPipelineConfig):
    """Two-GPU variant that places codec work on the second visible GPU."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1")
    )


EntryClass = MossTTSLocalPipelineConfig

Variants = {
    "default": MossTTSLocalPipelineConfig,
    "colocated": MossTTSLocalColocatedPipelineConfig,
    "split": MossTTSLocalSplitPipelineConfig,
}
