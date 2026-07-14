# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTalkerConfig


def test_talker_config_preserves_transformers_5_mrope_parameters() -> None:
    rope_parameters = {
        "interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_theta": 1_000_000,
        "rope_type": "default",
        "type": "default",
    }

    config = Qwen3OmniMoeTalkerConfig(text_config={"rope_parameters": rope_parameters})

    assert config.text_config.rope_scaling == rope_parameters
    assert config.text_config.rope_theta == 1_000_000
