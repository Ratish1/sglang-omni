# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS logit shaping keeps one owner for each model policy."""

from __future__ import annotations

import types

import pytest
import torch
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData


def make_runner(
    *,
    vocab_size: int,
    codec_eos_token_id: int,
    leading_silence_mask_frames: int = 0,
    silence_codec_ids: tuple[int, ...] = (),
) -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vocab_size=vocab_size,
            codec_eos_token_id=codec_eos_token_id,
        )
    )
    runner.leading_silence_mask_frames = leading_silence_mask_frames
    runner.silence_codec_ids = torch.tensor(silence_codec_ids, dtype=torch.long)
    return runner


def make_scheduled_request(
    *, mask_leading_silence: bool, generated_frames: int
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            mask_leading_silence=mask_leading_silence,
            output_codes=[torch.zeros(16)] * generated_frames,
        )
    )


def test_qwen3_tts_suppresses_configured_codec_tail_with_basic_slices() -> None:
    configured_vocab = 3072
    codec_eos = 2150
    materialized_vocab = 6144
    runner = make_runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=codec_eos,
    )
    logits = torch.randn(3, materialized_vocab)
    original = logits.clone()
    logits_output = types.SimpleNamespace(next_token_logits=logits)

    runner.apply_codec_suppress_tokens(logits_output, [object(), object()])

    suppress_start = configured_vocab - 1024
    assert torch.equal(logits[:2, :suppress_start], original[:2, :suppress_start])
    assert torch.isneginf(logits[:2, suppress_start:codec_eos]).all()
    assert torch.equal(logits[:2, codec_eos], original[:2, codec_eos])
    assert torch.isneginf(logits[:2, codec_eos + 1 : configured_vocab]).all()
    assert torch.equal(logits[:2, configured_vocab:], original[:2, configured_vocab:])
    assert torch.equal(logits[2], original[2])


def test_qwen3_tts_suppression_skips_empty_request_batch() -> None:
    runner = make_runner(vocab_size=3072, codec_eos_token_id=2150)
    logits = torch.randn(1, 6144)
    original = logits.clone()

    runner.apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), []
    )

    assert torch.equal(logits, original)


def test_qwen3_tts_masks_silence_ids_only_in_opening_frames_of_flagged_requests() -> (
    None
):
    configured_vocab = 3072
    silence_ids = (5, 7, 11)
    runner = make_runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=2150,
        leading_silence_mask_frames=2,
        silence_codec_ids=silence_ids,
    )
    requests = [
        make_scheduled_request(mask_leading_silence=True, generated_frames=0),
        make_scheduled_request(mask_leading_silence=True, generated_frames=1),
        make_scheduled_request(mask_leading_silence=True, generated_frames=2),
        make_scheduled_request(mask_leading_silence=False, generated_frames=0),
    ]
    logits = torch.randn(len(requests), 6144)
    original = logits.clone()

    runner.apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), requests
    )

    silence = list(silence_ids)
    speech = [
        token for token in range(configured_vocab - 1024) if token not in silence_ids
    ]
    assert torch.isneginf(logits[:2, silence]).all()
    assert torch.equal(logits[:2, speech], original[:2, speech])
    assert torch.equal(logits[2:, silence], original[2:, silence])


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qwen3_tts_scales_each_step_by_the_ids_its_request_generated() -> None:
    device = torch.device("cuda")
    vocab_size = 1040
    unsuppressed_tokens = 16
    runner = make_runner(vocab_size=vocab_size, codec_eos_token_id=1030)
    runner.device = device
    runner.repetition_scales = torch.ones(8, vocab_size, device=device)
    runner.repetition_penalties = torch.ones(8, device=device)
    runner.model.prepare_decode_buffers = lambda requests: None
    runner.build_prefill_input_embeds = lambda forward_batch, requests: torch.zeros(
        len(forward_batch.input_ids), 4, device=device
    )
    sampled_ids = iter([[4, 6], [7], [4, 2], [9, 4], [1]])
    sampled_logits: list[torch.Tensor] = []

    def sample(logits_output, forward_batch):
        sampled_logits.append(logits_output.next_token_logits.clone())
        return torch.tensor(next(sampled_ids), dtype=torch.int32, device=device)

    runner.tp_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(sample=sample)
    )
    generated_ids = {"a": [], "b": []}

    def scheduled(request_id, repetition_penalty, output_ids, middle_chunks):
        return types.SimpleNamespace(
            request_id=request_id,
            data=Qwen3TTSSGLangRequestData(
                repetition_penalty=repetition_penalty,
                req=types.SimpleNamespace(
                    output_ids=output_ids,
                    inflight_middle_chunks=middle_chunks,
                    sampling_params=types.SimpleNamespace(sampling_seed=None),
                ),
            ),
        )

    def run_step(requests, request_slots, is_extend):
        forward_batch = types.SimpleNamespace(
            req_pool_indices=torch.tensor(request_slots, device=device),
            forward_mode=types.SimpleNamespace(is_extend=lambda: is_extend),
            input_ids=torch.zeros(len(requests), dtype=torch.long),
            replace_embeds=None,
            sampling_info=types.SimpleNamespace(sampling_seed=None),
        )
        if is_extend:
            runner.before_prefill(forward_batch, None, requests)
        else:
            pass
        logits = torch.randn(len(requests), vocab_size, device=device)
        expected = logits.clone()
        scales = torch.ones(len(requests), vocab_size, device=device)
        for row_index, request in enumerate(requests):
            scales[row_index, generated_ids[request.request_id]] = (
                request.data.repetition_penalty
            )
        apply_scaling_penalties(expected, scales)

        token_ids = runner.sample_next_token_ids(
            types.SimpleNamespace(next_token_logits=logits),
            forward_batch,
            None,
            requests,
        )

        assert torch.equal(
            sampled_logits[-1][:, :unsuppressed_tokens],
            expected[:, :unsuppressed_tokens],
        )
        for request, token_id in zip(requests, token_ids.tolist(), strict=True):
            if request.data.req.inflight_middle_chunks <= 0:
                generated_ids[request.request_id].append(token_id)
            else:
                pass

    first = scheduled("a", 1.05, [], 0)
    second = scheduled("b", 1.2, [], 1)
    run_step([first, second], [3, 5], True)
    second.data.req.inflight_middle_chunks = 0
    run_step([second], [5], True)
    run_step([first, second], [3, 5], False)
    run_step([first, second], [3, 5], False)
    run_step([scheduled("a", 1.05, list(generated_ids["a"]), 0)], [7], True)

    assert generated_ids == {"a": [4, 4, 9, 1], "b": [7, 2, 4]}
