# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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


def _speech_embed(ids: torch.Tensor) -> torch.Tensor:
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
        speech_embed=_speech_embed,
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


def test_cosyvoice3_reference_encoders_pin_onnx_providers(monkeypatch) -> None:
    captured: list[object] = []

    def fake_session(model_path, sess_options, providers):
        captured.append(providers)

    fake_onnxruntime = types.SimpleNamespace(
        SessionOptions=types.SimpleNamespace,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        InferenceSession=fake_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cuda:0")
    utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cpu")
    utils.SpeakerEncoder("campplus.onnx", device="cuda:0")

    assert captured == [
        [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "use_ep_level_unified_stream": "1",
                },
            ),
            "CPUExecutionProvider",
        ],
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_speech_tokenizer_runs_one_call_at_a_time(monkeypatch) -> None:
    in_flight = 0
    most_in_flight = 0
    counter_lock = threading.Lock()

    class _SlowSession:
        def get_inputs(self):
            return [
                types.SimpleNamespace(name="feats"),
                types.SimpleNamespace(name="len"),
            ]

        def run(self, outputs, feeds):
            nonlocal in_flight, most_in_flight
            with counter_lock:
                in_flight += 1
                most_in_flight = max(most_in_flight, in_flight)
            time.sleep(0.02)
            with counter_lock:
                in_flight -= 1
            return [np.arange(4, dtype=np.int64)]

    fake_onnxruntime = types.SimpleNamespace(
        SessionOptions=types.SimpleNamespace,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        InferenceSession=lambda model_path, sess_options, providers: _SlowSession(),
    )
    fake_whisper = types.SimpleNamespace(
        log_mel_spectrogram=lambda audio, n_mels: torch.zeros(1, n_mels, 8)
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    tokenizer = utils.SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cpu")

    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(
            pool.map(
                lambda _: tokenizer.extract_speech_token(
                    np.zeros(1600, dtype=np.float32), 16000
                ),
                range(8),
            )
        )

    assert most_in_flight == 1
    assert all(token.tolist() == [[0, 1, 2, 3]] for token in tokens)
