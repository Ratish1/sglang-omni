# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.sampling.repetition_aware import repetition_aware_redraw

ROWS = 4096


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seeded_redraw_reproduces_without_the_masked_token() -> None:
    probs = torch.zeros((ROWS, 8), device="cuda")
    probs[:, 3], probs[:, 5], probs[:, 6] = 0.9, 0.06, 0.04
    sampled_ids = torch.full((ROWS,), 3, dtype=torch.int32, device="cuda")
    rows = torch.arange(ROWS, device="cuda")
    seeds = torch.arange(1, ROWS + 1, dtype=torch.int64, device="cuda")
    positions = torch.arange(ROWS, device="cuda")

    first = repetition_aware_redraw(probs, sampled_ids, rows, seeds, positions)
    second = repetition_aware_redraw(probs, sampled_ids, rows, seeds, positions)

    assert torch.equal(first, second)
    assert not bool((first == 3).any())
