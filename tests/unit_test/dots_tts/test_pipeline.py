# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_dots_tts_uses_framework_stage_boundaries() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig
    from sglang_omni.models.dots_tts.payload_types import DotsTTSState

    config = DotsTTSPipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encode",
        "latent_engine",
        "vocoder",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.terminal_stages == ["vocoder"]
    assert config.generation_sglang_role_to_stage() == {"generation": "latent_engine"}
    assert config.additional_speech_languages == frozenset({"auto_detect"})
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("DotsTTSForConditionalGeneration")
        is DotsTTSPipelineConfig
    )
    assert DotsTTSState().num_steps == 4


def test_public_speech_boundary_accepts_dots_auto_detect_alias() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(model_path="dots-studio/dots.tts-mf")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        additional_speech_languages=config.additional_speech_languages,
    )

    request = validator.parse_generation_request(
        {"input": "hello", "voice": "default", "language": "AUTO_DETECT"}
    )

    assert request.request.language == "auto_detect"


def test_dots_tts_rejects_tp() -> None:
    import pytest

    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    raw = DotsTTSPipelineConfig(model_path="model").model_dump()
    stage = next(item for item in raw["stages"] if item["name"] == "latent_engine")
    stage["tp_size"] = 2
    stage["parallelism"] = {"tp": 2}
    stage["gpu"] = [0, 1]
    with pytest.raises(ValueError, match="tp_size=1"):
        DotsTTSPipelineConfig(**raw)
