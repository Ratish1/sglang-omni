# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator


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
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_required is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("DotsTTSForConditionalGeneration")
        is DotsTTSPipelineConfig
    )
    assert DotsTTSState().num_steps == 4


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


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"input": "target"}, "ref_audio"),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
            },
            "ref_text",
        ),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
                "ref_text": "reference",
                "references": [
                    {
                        "data": "UklGRg==",
                        "media_type": "audio/wav",
                        "text": "reference",
                    }
                ],
            },
            "references",
        ),
    ],
)
def test_public_speech_boundary_enforces_dots_conditioning_contract(
    payload: dict[str, Any], param: str
) -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(model_path="dots-studio/dots.tts-mf")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_generation_request(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == param


def test_public_speech_boundary_preserves_valid_dots_reference() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(model_path="dots-studio/dots.tts-mf")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
    )
    prepared = validator.parse_generation_request(
        {
            "input": "target",
            "task_type": "Base",
            "language": "Auto",
            "max_new_tokens": 3,
            "ref_audio": "data:audio/wav;base64,UklGRg==",
            "ref_text": "reference",
        }
    )

    assert prepared.request.task_type == "Base"
    assert prepared.request.language == "Auto"
    assert prepared.request.max_new_tokens == 3
    assert prepared.reference_descriptors == [
        {
            "data": "UklGRg==",
            "media_type": "audio/wav",
            "text": "reference",
        }
    ]
