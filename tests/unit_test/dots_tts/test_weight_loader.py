# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.dots_tts.weight_loader import (
    map_dots_qwen2_key,
    validate_checkpoint_files,
)


def test_map_dots_qwen2_key_to_sglang_qwen2_key() -> None:
    assert (
        map_dots_qwen2_key("llm.model.embed_tokens.weight")
        == "qwen2.model.embed_tokens.weight"
    )
    assert (
        map_dots_qwen2_key("llm.model.layers.0.self_attn.q_proj.weight")
        == "qwen2.model.layers.0.self_attn.q_proj.weight"
    )
    assert map_dots_qwen2_key("llm.lm_head.weight") == "qwen2.lm_head.weight"
    assert map_dots_qwen2_key("core.dit.blocks.0.weight") is None


def test_validate_checkpoint_files_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_checkpoint_files(tmp_path)

    message = str(exc_info.value)
    assert "config.json" in message
    assert "llm_config.json" in message
    assert "model.safetensors" in message
