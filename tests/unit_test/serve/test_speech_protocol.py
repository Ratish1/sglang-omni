# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechService


def test_speech_service_rejects_non_string_input() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": 123})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


def test_speech_service_requires_pcm_for_http_streaming() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "stream": True, "response_format": "wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "response_format"


def test_speech_service_normalizes_issue_fields_into_tts_params() -> None:
    service = SpeechService(default_model="tts")
    request = CreateSpeechRequest.model_validate(
        {
            "input": "hello",
            "speaker": "alloy",
            "response_format": "WAV",
            "task_type": "voice-design",
            "language": "english",
            "instructions": "calm",
            "x_vector_only_mode": True,
            "initial_codec_chunk_frames": 8,
            "max_new_tokens": 128,
        }
    )

    gen_req = service.build_generate_request(request)
    tts_params = gen_req.metadata["tts_params"]

    assert gen_req.model == "tts"
    assert tts_params["voice"] == "alloy"
    assert tts_params["response_format"] == "wav"
    assert tts_params["task_type"] == "VoiceDesign"
    assert tts_params["language"] == "English"
    assert tts_params["instructions"] == "calm"
    assert tts_params["x_vector_only_mode"] is True
    assert tts_params["initial_codec_chunk_frames"] == 8
    assert gen_req.sampling.max_new_tokens == 128
    assert tts_params["explicit_generation_params"] == ["max_new_tokens"]


def test_file_reference_requires_allowlist() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "file:///tmp/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_allowed_local_media_path_must_be_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(
        ValueError, match="allowed local media path must be a directory"
    ):
        SpeechService(default_model="tts", allowed_local_media_paths=[missing])


def test_file_reference_resolves_inside_allowlist(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )

    request = service.parse_request(
        {"input": "hello", "ref_audio": audio_path.as_uri()}
    )
    gen_req = service.build_generate_request(request)

    assert request.ref_audio == str(audio_path.resolve())
    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"audio_path": str(audio_path.resolve())}],
    }


def test_file_reference_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside.wav"
    allowed.mkdir()
    outside.write_bytes(b"RIFF")
    link = allowed / "escape.wav"
    link.symlink_to(outside)
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[allowed],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": link.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"
