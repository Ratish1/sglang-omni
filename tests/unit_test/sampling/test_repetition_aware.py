# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.sampling.repetition_aware import repetition_aware_redraw

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

ROWS = 4096


def redraw_repeated_rows(
    sampling_seeds: torch.Tensor | None,
) -> torch.Tensor:
    probs = torch.zeros((ROWS, 8), device="cuda")
    probs[:, 3], probs[:, 5], probs[:, 6] = 0.9, 0.06, 0.04
    return repetition_aware_redraw(
        probs,
        torch.full((ROWS,), 3, dtype=torch.int32, device="cuda"),
        torch.arange(ROWS, device="cuda"),
        sampling_seeds,
        torch.arange(ROWS, device="cuda"),
    )


def test_repeated_token_is_redrawn_from_the_rest_of_the_distribution() -> None:
    torch.manual_seed(0)

    token_ids = redraw_repeated_rows(None)

    assert not bool((token_ids == 3).any())
    assert (token_ids == 5).double().mean().item() == pytest.approx(0.6, abs=0.03)


def test_seeded_redraw_reproduces() -> None:
    seeds = torch.arange(1, ROWS + 1, dtype=torch.int64, device="cuda")

    first = redraw_repeated_rows(seeds)
    second = redraw_repeated_rows(seeds)

    assert torch.equal(first, second)
