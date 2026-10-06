# SPDX-License-Identifier: Apache-2.0
"""Tests for compiling SGLang's sampler calls before serving."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from torch._dynamo.utils import counters

from sglang_omni.sampling.compiled_calls import (
    scaling_penalty_calls,
    seeded_sampling_call,
    warm_compiled_sampler_calls,
)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warmed_calls_serve_every_batch_size_without_a_compile() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    vocab_size = 3072
    seed_buffer = torch.zeros(64, dtype=torch.long, device=device)
    warm_compiled_sampler_calls(
        (
            seeded_sampling_call(
                device=device,
                log_probability_dtype=torch.float64,
                vocab_size=vocab_size,
                seed_buffer=seed_buffer,
                grad_enabled=True,
            ),
            *scaling_penalty_calls(
                device=device, logits_dtype=torch.bfloat16, vocab_size=vocab_size
            ),
        )
    )
    compiled_frames = counters["frames"]["ok"]

    for rows in (1, 2, 5, 13):
        multinomial_with_seed(
            torch.randn(rows, vocab_size, dtype=torch.float64, device=device),
            seed_buffer[:rows],
            torch.arange(rows, device=device),
        )
        logits = torch.randn(64, vocab_size, dtype=torch.bfloat16, device=device)
        penalties = torch.ones(rows, vocab_size, device=device)
        apply_scaling_penalties(logits[:rows], penalties)
        apply_scaling_penalties(logits[:rows].clone(), penalties)
    assert counters["frames"]["ok"] == compiled_frames

    apply_scaling_penalties(
        torch.randn(2, vocab_size, dtype=torch.float16, device=device),
        torch.ones(2, vocab_size, device=device),
    )
    assert counters["frames"]["ok"] == compiled_frames + 1
