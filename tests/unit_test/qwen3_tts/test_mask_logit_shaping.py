# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS logit shaping keeps one owner for each model policy."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


def _runner(*, vocab_size: int, codec_eos_token_id: int) -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vocab_size=vocab_size,
            codec_eos_token_id=codec_eos_token_id,
        )
    )
    runner._repetition_mask = None
    runner._repetition_penalty_column = None
    runner._repetition_mask_last_sampled = None
    runner._repetition_mask_prep_rids = None
    runner._repetition_mask_active = False
    runner._qwen_repetition_enabled = True
    return runner


def _sampling_request(
    sglang_penalty: float,
    *,
    qwen_penalty: float = 1.0,
) -> types.SimpleNamespace:
    sampling_params = types.SimpleNamespace(
        repetition_penalty=sglang_penalty,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_new_tokens=0,
    )
    req = types.SimpleNamespace(
        sampling_params=sampling_params,
        custom_logit_processor=None,
    )
    req._omni_data = types.SimpleNamespace(
        qwen_repetition_penalty=qwen_penalty,
    )
    return req


def _penalty_request(
    rid: str,
    epoch: int,
    qwen_penalty: float,
    output_ids: list[int],
) -> types.SimpleNamespace:
    req = _sampling_request(1.0, qwen_penalty=qwen_penalty)
    req.output_ids = output_ids
    data = req._omni_data
    data.req = req
    data._qwen3_tts_prep_epoch = epoch
    return types.SimpleNamespace(request_id=rid, data=data)


def _qwen_penalty_reference(
    logits: torch.Tensor,
    requests: list[types.SimpleNamespace],
) -> torch.Tensor:
    output = logits.clone()
    vocab = output.shape[1]
    for row_idx, sched_req in enumerate(requests):
        data = sched_req.data
        penalty = float(data.qwen_repetition_penalty)
        unique = {
            int(token_id)
            for token_id in data.req.output_ids
            if 0 <= int(token_id) < vocab
        }
        if penalty == 1.0 or not unique:
            continue
        token_ids = torch.tensor(sorted(unique), dtype=torch.long)
        scores = output[row_idx, token_ids].to(torch.float32)
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        output[row_idx, token_ids] = scores.to(output.dtype)
    return output


def _apply_qwen_penalty(
    runner: Qwen3TTSModelRunner,
    logits: torch.Tensor,
    requests: list[types.SimpleNamespace],
) -> torch.Tensor:
    logits_output = types.SimpleNamespace(next_token_logits=logits)
    runner._apply_repetition_penalty(logits_output, requests)
    return logits_output.next_token_logits


def test_qwen3_tts_leaves_repetition_penalty_to_sglang() -> None:
    runner = _runner(vocab_size=128, codec_eos_token_id=127)
    runner._qwen_repetition_enabled = False
    logits = torch.randn(2, 128)
    original = logits.clone()
    requests = [
        _penalty_request("a", 1, 1.3, [3, 7]),
        _penalty_request("b", 2, 1.2, [4]),
    ]

    got = _apply_qwen_penalty(runner, logits, requests)

    assert torch.equal(got, original)


def test_qwen3_tts_missing_diagnostic_owner_state_defaults_to_sglang() -> None:
    runner = _runner(vocab_size=128, codec_eos_token_id=127)
    del runner._qwen_repetition_enabled
    logits = torch.randn(1, 128)
    original = logits.clone()

    got = _apply_qwen_penalty(
        runner,
        logits,
        [_penalty_request("a", 1, 1.3, [3, 7])],
    )

    assert torch.equal(got, original)


def test_qwen3_tts_public_penalty_disables_async_lookahead() -> None:
    runner = _runner(vocab_size=128, codec_eos_token_id=127)

    assert runner.lookahead_eligible(
        types.SimpleNamespace(reqs=[_sampling_request(1.0)])
    )
    assert not runner.lookahead_eligible(
        types.SimpleNamespace(reqs=[_sampling_request(1.05)])
    )
    assert not runner.lookahead_eligible(
        types.SimpleNamespace(reqs=[_sampling_request(1.0, qwen_penalty=1.05)])
    )


def test_qwen3_tts_qwen_penalty_matches_reference_across_steady_steps() -> None:
    torch.manual_seed(3)
    runner = _runner(vocab_size=128, codec_eos_token_id=127)
    requests = [
        _penalty_request("a", 1, 1.3, [3, 7]),
        _penalty_request("b", 2, 1.0, [4]),
    ]

    for step in range(6):
        logits = torch.randn(2, 128, dtype=torch.bfloat16)
        got = _apply_qwen_penalty(runner, logits.clone(), requests)
        expected = _qwen_penalty_reference(logits, requests)
        assert torch.equal(got, expected), step

        new_tokens = [10 + step, 40 + step]
        for request, token_id in zip(requests, new_tokens):
            request.data.req.output_ids.append(token_id)
        runner._repetition_mask_last_sampled = torch.tensor(new_tokens)


def test_qwen3_tts_qwen_penalty_rebuilds_for_batch_and_request_changes() -> None:
    torch.manual_seed(4)
    runner = _runner(vocab_size=64, codec_eos_token_id=63)
    requests = [
        _penalty_request("a", 1, 1.5, [3, 9, 11]),
        _penalty_request("b", 2, 1.2, [4, 8]),
    ]
    logits = torch.randn(2, 64)
    assert torch.equal(
        _apply_qwen_penalty(runner, logits.clone(), requests),
        _qwen_penalty_reference(logits, requests),
    )

    survivor = [requests[1]]
    survivor_logits = torch.randn(1, 64)
    got = _apply_qwen_penalty(runner, survivor_logits.clone(), survivor)
    assert torch.equal(got, _qwen_penalty_reference(survivor_logits, survivor))
    assert got[0, 3] == survivor_logits[0, 3]

    reused = [_penalty_request("b", 7, 1.2, [30])]
    reused_logits = torch.randn(1, 64)
    got = _apply_qwen_penalty(runner, reused_logits.clone(), reused)
    assert torch.equal(got, _qwen_penalty_reference(reused_logits, reused))
    assert got[0, 4] == reused_logits[0, 4]


def test_qwen3_tts_qwen_penalty_clears_retracted_history() -> None:
    torch.manual_seed(9)
    runner = _runner(vocab_size=64, codec_eos_token_id=63)
    requests = [_penalty_request("a", 1, 1.5, [3, 9])]
    _apply_qwen_penalty(runner, torch.randn(1, 64), requests)

    requests[0].data.req.output_ids = [3]
    runner._repetition_mask_last_sampled = torch.tensor([3])
    logits = torch.randn(1, 64)
    got = _apply_qwen_penalty(runner, logits.clone(), requests)
    assert torch.equal(got, _qwen_penalty_reference(logits, requests))
    assert got[0, 9] == logits[0, 9]

    requests[0].data.req.output_ids = []
    empty_logits = torch.randn(1, 64)
    got = _apply_qwen_penalty(runner, empty_logits.clone(), requests)
    assert torch.equal(got, empty_logits)


def test_qwen3_tts_suppresses_configured_codec_tail_with_basic_slices() -> None:
    configured_vocab = 3072
    codec_eos = 2150
    materialized_vocab = 6144
    runner = _runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=codec_eos,
    )
    logits = torch.randn(3, materialized_vocab)
    original = logits.clone()
    logits_output = types.SimpleNamespace(next_token_logits=logits)

    runner._apply_codec_suppress_tokens(logits_output, [object(), object()])

    suppress_start = configured_vocab - 1024
    assert torch.equal(logits[:2, :suppress_start], original[:2, :suppress_start])
    assert torch.isneginf(logits[:2, suppress_start:codec_eos]).all()
    assert torch.equal(logits[:2, codec_eos], original[:2, codec_eos])
    assert torch.isneginf(logits[:2, codec_eos + 1 : configured_vocab]).all()
    assert torch.equal(logits[:2, configured_vocab:], original[:2, configured_vocab:])
    assert torch.equal(logits[2], original[2])


def test_qwen3_tts_suppression_skips_empty_request_batch() -> None:
    runner = _runner(vocab_size=3072, codec_eos_token_id=2150)
    logits = torch.randn(1, 6144)
    original = logits.clone()

    runner._apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), []
    )

    assert torch.equal(logits, original)
