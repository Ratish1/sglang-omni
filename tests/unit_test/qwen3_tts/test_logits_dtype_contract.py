# SPDX-License-Identifier: Apache-2.0
"""The talker owes the sampler float32 logits, as SGLang's LogitsProcessor does."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

VOCAB = 3072
HIDDEN = 32
CODEC_EOS = 2150


def _talker(hidden_states: torch.Tensor, weight: torch.Tensor) -> Qwen3TTSTalker:
    talker = object.__new__(Qwen3TTSTalker)
    talker.model = lambda **kwargs: hidden_states
    talker.codec_head = lambda x: (torch.nn.functional.linear(x, weight), None)
    return talker


def _forward_batch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        mrope_positions=None,
        forward_mode=types.SimpleNamespace(is_extend=lambda: False),
    )


def _runner() -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(vocab_size=VOCAB, codec_eos_token_id=CODEC_EOS)
    )
    return runner


def _forward(rows: int) -> tuple:
    torch.manual_seed(0)
    hidden_states = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    out = Qwen3TTSTalker.forward(
        _talker(hidden_states, weight),
        input_ids=torch.zeros(rows, dtype=torch.long),
        positions=torch.zeros(rows, dtype=torch.long),
        forward_batch=_forward_batch(),
    )
    return out, hidden_states


def test_forward_returns_float32_logits_and_keeps_hidden_states_in_model_dtype():
    out, hidden_states = _forward(2)

    assert out.next_token_logits.dtype is torch.float32
    assert out.hidden_states.dtype is hidden_states.dtype
    assert out.next_token_logits.shape == (2, VOCAB)


def test_logit_shaping_chain_hands_the_sampler_float32():
    out, _ = _forward(2)
    requests = [types.SimpleNamespace() for _ in range(2)]
    runner = _runner()

    runner._apply_repetition_penalty(out, requests)
    runner._apply_codec_suppress_tokens(out, requests)

    logits = out.next_token_logits
    assert logits.dtype is torch.float32

    suppressed = torch.ones(VOCAB, dtype=torch.bool)
    suppressed[: VOCAB - 1024] = False
    suppressed[CODEC_EOS] = False
    assert bool(suppressed.sum()) == 1023
    assert torch.isneginf(logits[:, suppressed]).all()
    assert torch.isfinite(logits[:, ~suppressed]).all()
