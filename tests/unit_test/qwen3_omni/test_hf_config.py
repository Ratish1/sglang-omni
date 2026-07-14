from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTalkerConfig

_MROPE_CONFIG = {
    "interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_theta": 1_000_000,
    "rope_type": "default",
    "type": "default",
}


def test_transformers_v5_talker_rope_parameters_are_exposed_to_sglang() -> None:
    config = Qwen3OmniMoeTalkerConfig(
        text_config={"rope_parameters": _MROPE_CONFIG},
        code_predictor_config={
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "rope_type": "default",
            }
        },
    )

    assert config.text_config.rope_theta == 1_000_000
    assert config.text_config.rope_scaling == _MROPE_CONFIG
    assert config.text_config.rope_scaling is not _MROPE_CONFIG
    assert config.code_predictor_config.rope_theta == 1_000_000
    assert config.code_predictor_config.rope_scaling == {
        "rope_theta": 1_000_000,
        "rope_type": "default",
    }


def test_transformers_v5_code_predictor_rope_theta_is_preserved() -> None:
    config = Qwen3OmniMoeTalkerConfig(
        code_predictor_config={
            "rope_theta": 10_000,
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "rope_type": "default",
            },
        }
    )

    assert config.code_predictor_config.rope_theta == 1_000_000
    assert config.code_predictor_config.rope_scaling == {
        "rope_theta": 1_000_000,
        "rope_type": "default",
    }


def test_legacy_talker_rope_scaling_is_unchanged() -> None:
    rope_scaling = dict(_MROPE_CONFIG)
    rope_scaling.pop("rope_theta")
    config = Qwen3OmniMoeTalkerConfig(
        text_config={"rope_theta": 1_000_000, "rope_scaling": rope_scaling}
    )

    assert config.text_config.rope_theta == 1_000_000
    assert config.text_config.rope_scaling == rope_scaling
    assert config.text_config.rope_scaling is not rope_scaling
