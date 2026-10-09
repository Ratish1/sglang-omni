# SPDX-License-Identifier: Apache-2.0
"""Serving-layer validation for the precomputed speaker_embedding field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import (
    PreparedSpeechReferences,
    SpeechRequestValidator,
    build_tts_params,
)

DIM = 1024
EMBEDDING = [0.01] * DIM
BASE_REQUEST = {"model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "input": "hello"}


def make_validator(**overrides: object) -> SpeechRequestValidator:
    kwargs: dict[str, object] = {
        "default_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "speaker_embedding_dim": DIM,
    }
    kwargs.update(overrides)
    return SpeechRequestValidator(**kwargs)


@pytest.mark.parametrize(
    ("payload_extra", "validator_kwargs", "expected_param", "expected_fragment"),
    [
        ({"speaker_embedding": []}, {}, "speaker_embedding", None),
        ({"speaker_embedding": [0.01, "x"]}, {}, "speaker_embedding", None),
        (
            {"speaker_embedding": EMBEDDING[:-1]},
            {},
            "speaker_embedding",
            "must contain 1024 values",
        ),
        (
            {"speaker_embedding": EMBEDDING, "x_vector_only_mode": False},
            {},
            "x_vector_only_mode",
            None,
        ),
        (
            {"speaker_embedding": EMBEDDING, "task_type": "CustomVoice"},
            {},
            "task_type",
            None,
        ),
        (
            {"speaker_embedding": EMBEDDING},
            {"speaker_embedding_dim": None},
            "speaker_embedding",
            None,
        ),
    ],
)
def test_speaker_embedding_rejections(
    payload_extra: dict,
    validator_kwargs: dict,
    expected_param: str,
    expected_fragment: str | None,
) -> None:
    validator = make_validator(**validator_kwargs)
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request({**BASE_REQUEST, **payload_extra})
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == expected_param
    if expected_fragment is not None:
        assert expected_fragment in exc_info.value.message


def test_valid_speaker_embedding_lowers_into_tts_params() -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    tts_params = build_tts_params(request)
    assert tts_params["speaker_embedding"] == EMBEDDING
    assert tts_params["task_type"] == "Base"
    explicit = validator.parse_request(
        {**BASE_REQUEST, "speaker_embedding": EMBEDDING, "x_vector_only_mode": True}
    )
    assert build_tts_params(explicit)["x_vector_only_mode"] is True


@pytest.mark.parametrize(
    "conflict",
    [{"ref_audio": "/tmp/ref.wav"}, {"references": [{"text": "transcript"}]}],
)
def test_speaker_embedding_conflicts_with_reference_audio(conflict: dict) -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    prepared = PreparedSpeechReferences(request_updates={}, reference_descriptors=[])
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.validate_speech_references(
            request.model_copy(update=conflict), prepared
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "speaker_embedding"


def test_speaker_embedding_dim_resolution(tmp_path: Path) -> None:
    base = PipelineConfig(
        model_path="/tmp/unused",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path="x:y",
                terminal=True,
            )
        ],
    )
    assert base.resolve_speaker_embedding_dim() is None

    def checkpoint_config(name: str, config: dict) -> Qwen3TTSPipelineConfig:
        path = Path(tmp_path) / name
        path.mkdir()
        path.joinpath("config.json").write_text(json.dumps(config), encoding="utf-8")
        return Qwen3TTSPipelineConfig(model_path=str(path))

    base_checkpoint = checkpoint_config(
        "base", {"tts_model_type": "base", "speaker_encoder_config": {"enc_dim": 1024}}
    )
    assert base_checkpoint.resolve_speaker_embedding_dim() == 1024
    custom_checkpoint = checkpoint_config("custom", {"tts_model_type": "custom_voice"})
    assert custom_checkpoint.resolve_speaker_embedding_dim() is None
