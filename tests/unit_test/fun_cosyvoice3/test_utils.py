# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import utils
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    FILL_ID,
    SOS_ID,
    TASK_ID,
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
)
from sglang_omni.models.fun_cosyvoice3.utils import build_llm_prompt_embeddings


def speech_embed(ids: torch.Tensor) -> torch.Tensor:
    return ids.to(dtype=torch.float32).unsqueeze(-1).expand(*ids.shape, 4)


def test_cosyvoice3_prompt_embeddings_use_speech_control_tokens_and_reference_tokens() -> (
    None
):
    text_embed = torch.tensor([[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]])
    prompt_tokens = torch.tensor([[30, 31]], dtype=torch.int32)

    result = build_llm_prompt_embeddings(
        text_token=torch.tensor([[1, 2]]),
        text_embed=text_embed,
        prompt_speech_token=prompt_tokens,
        speech_embed=speech_embed,
        embedding=torch.full((1, 192), 999.0),
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.shape == (1, 6, 4)
    assert torch.equal(result[0, 0], torch.full((4,), float(SOS_ID)))
    assert torch.equal(result[0, 1:3], text_embed[0])
    assert torch.equal(result[0, 3], torch.full((4,), float(TASK_ID)))
    assert torch.equal(result[0, 4], torch.full((4,), 30.0))
    assert torch.equal(result[0, 5], torch.full((4,), 31.0))
    assert not torch.any(result == 999.0)


def test_cosyvoice3_prompt_embeddings_keep_empty_reference_shape_and_dtype() -> None:
    result = build_llm_prompt_embeddings(
        text_token=torch.tensor([[1]]),
        text_embed=torch.ones(1, 1, 3, dtype=torch.float16),
        prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        speech_embed=lambda ids: torch.ones(*ids.shape, 3, dtype=torch.float32),
        embedding=torch.zeros(0, 192),
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=3,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert result.shape == (1, 3, 3)
    assert result.dtype == torch.float32
    assert result[:, 3:].numel() == 0


def test_cosyvoice3_speech_vocabulary_layout_is_explicit() -> None:
    assert SOS_ID == VOCAB_SIZE
    assert EOS_ID == VOCAB_SIZE + 1
    assert TASK_ID == VOCAB_SIZE + 2
    assert FILL_ID == VOCAB_SIZE + 3
    assert TOTAL_VOCAB_SIZE == VOCAB_SIZE + 200


def test_cosyvoice3_prompt_mel_uses_flow_layout_and_fixed_configuration(
    monkeypatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_mel(waveform: torch.Tensor) -> torch.Tensor:
        captured["waveform"] = waveform
        return torch.arange(1 * 80 * 3, dtype=torch.float32).reshape(1, 80, 3)

    monkeypatch.setattr(utils, "run_cosyvoice3_mel_spectrogram", fake_mel)

    result = utils.extract_prompt_speech_feat(np.zeros(12, dtype=np.float64))

    assert captured["waveform"].shape == (1, 12)
    assert captured["waveform"].dtype == torch.float32
    assert result.shape == (1, 3, 80)
    assert torch.equal(result[0, 0], torch.arange(0, 80 * 3, 3, dtype=torch.float32))


def test_cosyvoice3_reference_encoders_pin_onnx_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str | tuple[str, dict[str, str]]]] = []
    session_configs: list[dict[str, str]] = []

    class SessionOptions:
        def __init__(self) -> None:
            self.config_entries: dict[str, str] = {}

        def add_session_config_entry(self, name: str, value: str) -> None:
            self.config_entries[name] = value

    def fake_session(
        model_path: str,
        sess_options: SessionOptions,
        providers: list[str | tuple[str, dict[str, str]]],
    ) -> types.SimpleNamespace:
        captured.append(providers)
        session_configs.append(sess_options.config_entries)
        return types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    fake_onnxruntime = types.SimpleNamespace(
        SessionOptions=SessionOptions,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        InferenceSession=fake_session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cuda:0")
    utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cpu")
    utils.SpeakerEncoder("campplus.onnx", device="cuda:0")

    assert captured == [
        [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ],
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert session_configs == [{}, {}, {"session.intra_op.allow_spinning": "0"}]


@pytest.mark.accelerator
def test_cuda_speech_tokenizer_runs_on_a_top_priority_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str | tuple[str, dict[str, str]]]] = []

    def fake_session(
        model_path: str,
        sess_options: types.SimpleNamespace,
        providers: list[str | tuple[str, dict[str, str]]],
    ) -> types.SimpleNamespace:
        captured.append(providers)
        return types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            SessionOptions=types.SimpleNamespace,
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
            InferenceSession=fake_session,
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )

    tokenizer = utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cuda:0")

    assert tokenizer.compute_stream is not None
    assert tokenizer.compute_stream.priority == torch.cuda.Stream.priority_range()[1]
    assert captured == [
        [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "user_compute_stream": str(tokenizer.compute_stream.cuda_stream),
                },
            ),
            "CPUExecutionProvider",
        ]
    ]


@pytest.mark.parametrize("is_on_cuda", [True, False])
def test_speech_tokenizer_on_cuda_runs_its_longest_input_first_and_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch, is_on_cuda: bool
) -> None:
    run_frames: list[int] = []
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    class GatedSession:
        def get_providers(self) -> list[str]:
            if is_on_cuda:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                return ["CPUExecutionProvider"]

        def get_inputs(self) -> list[types.SimpleNamespace]:
            return [
                types.SimpleNamespace(name="feats"),
                types.SimpleNamespace(name="feats_length"),
            ]

        def run(
            self, output_names: None, feeds: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            frames = int(feeds["feats_length"][0])
            run_frames.append(frames)
            if frames == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            elif frames == 2:
                second_started.set()
            else:
                pass
            return [np.zeros(frames, dtype=np.int64)]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            SessionOptions=types.SimpleNamespace,
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
            InferenceSession=lambda model_path, sess_options, providers: (
                GatedSession()
            ),
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(
            log_mel_spectrogram=lambda audio, n_mels: torch.zeros(
                1, n_mels, audio.shape[1] // 160
            )
        ),
    )
    tokenizer = utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cuda:0")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            tokenizer.extract_speech_token, np.zeros(160, dtype=np.float32)
        )
        assert first_started.wait(timeout=2)
        second = pool.submit(
            tokenizer.extract_speech_token, np.zeros(320, dtype=np.float32)
        )
        if is_on_cuda:
            assert not second_started.wait(timeout=0.2)
        else:
            assert second_started.wait(timeout=2)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert run_frames == ([3000] if is_on_cuda else []) + [1, 2]
