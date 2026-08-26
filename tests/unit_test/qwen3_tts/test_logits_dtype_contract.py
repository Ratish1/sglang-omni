# SPDX-License-Identifier: Apache-2.0
"""The talker owes the sampler float32 logits, as SGLang's LogitsProcessor does."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

VOCAB = 128
HIDDEN = 32


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
    runner._mask_last_sampled = None
    runner._mask_prep_rids = None
    runner._mask_rep_active = False
    runner._mask_sup_active = False
    return runner


def _request(penalty: float, output_ids: list[int]) -> types.SimpleNamespace:
    sp = types.SimpleNamespace(repetition_penalty=penalty)
    req = types.SimpleNamespace(sampling_params=sp, output_ids=output_ids)
    data = types.SimpleNamespace(req=req, suppress_tokens=None, _qwen3_tts_prep_epoch=1)
    return types.SimpleNamespace(request_id="a", data=data)


def test_forward_returns_float32_logits_and_keeps_hidden_states_in_model_dtype():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)

    out = Qwen3TTSTalker.forward(
        _talker(hidden_states, weight),
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        forward_batch=_forward_batch(),
    )

    assert out.next_token_logits.dtype is torch.float32
    assert out.hidden_states.dtype is torch.bfloat16
    assert out.next_token_logits.shape == (2, VOCAB)


def test_repetition_penalty_reaches_the_sampler_without_a_bfloat16_round_trip():
    torch.manual_seed(1)
    hidden_states = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    penalty = 1.05
    output_ids = [3, 9, 41]

    out = Qwen3TTSTalker.forward(
        _talker(hidden_states, weight),
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        forward_batch=_forward_batch(),
    )
    head_output = out.next_token_logits.clone()

    _runner()._apply_repetition_penalty(out, [_request(penalty, output_ids)])

    idx = torch.tensor(output_ids, dtype=torch.long)
    scored = head_output[0, idx]
    expected = torch.where(scored > 0, scored / penalty, scored * penalty)
    assert torch.equal(out.next_token_logits[0, idx], expected)
    assert not torch.equal(out.next_token_logits[0, idx], expected.bfloat16().float())
