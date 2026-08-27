# SPDX-License-Identifier: Apache-2.0
"""The talker owes the sampler float32 logits, as SGLang's LogitsProcessor does."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

VOCAB = 128
HIDDEN = 32


def _talker(hidden_states: torch.Tensor, weight: torch.Tensor) -> Qwen3TTSTalker:
    talker = object.__new__(Qwen3TTSTalker)
    talker.model = lambda **kwargs: hidden_states
    talker.codec_head = lambda x: (torch.nn.functional.linear(x, weight), None)
    return talker


def _forward_batch(
    logits_buffer: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        mrope_positions=None,
        forward_mode=types.SimpleNamespace(is_extend=lambda: False),
        next_token_logits_buffer=logits_buffer,
    )


def _forward(
    rows: int,
    logits_buffer: torch.Tensor | None = None,
) -> tuple:
    torch.manual_seed(0)
    hidden_states = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    expected_logits = torch.nn.functional.linear(hidden_states, weight).float()
    out = Qwen3TTSTalker.forward(
        _talker(hidden_states, weight),
        input_ids=torch.zeros(rows, dtype=torch.long),
        positions=torch.zeros(rows, dtype=torch.long),
        forward_batch=_forward_batch(logits_buffer),
    )
    return out, hidden_states, expected_logits


def test_forward_casts_logits_without_graph_buffer():
    out, hidden_states, expected_logits = _forward(2)

    assert out.next_token_logits.dtype is torch.float32
    assert torch.equal(out.next_token_logits, expected_logits)
    assert out.hidden_states.dtype is hidden_states.dtype


def test_forward_writes_logits_into_graph_buffer():
    logits_buffer = torch.empty(2, VOCAB, dtype=torch.float32)
    out, hidden_states, expected_logits = _forward(2, logits_buffer)

    assert out.next_token_logits is logits_buffer
    assert torch.equal(out.next_token_logits, expected_logits)
    assert out.hidden_states.dtype is hidden_states.dtype
