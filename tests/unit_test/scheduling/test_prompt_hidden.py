# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.scheduling.sglang_backend.prompt_hidden import PromptHiddenStore


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host rows need CUDA")
def test_a_read_returns_the_rows_its_slots_were_last_computed_with() -> None:
    req_to_token = torch.zeros((2, 6), dtype=torch.int32, device="cuda")
    store = PromptHiddenStore(
        slot_count=16,
        hidden_size=2,
        dtype=torch.float32,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        device="cuda",
    )
    first_rows = torch.arange(6.0, device="cuda").view(3, 2)
    store.stage(torch.tensor([3, 4, 5], device="cuda"), first_rows)
    second_rows = torch.full((1, 2), 7.0, device="cuda")
    store.stage(torch.tensor([9], device="cuda"), second_rows)
    reused_rows = torch.full((1, 2), 8.0, device="cuda")
    store.stage(torch.tensor([5], device="cuda"), reused_rows)
    req_to_token[1, 1:4] = torch.tensor([3, 4, 9], dtype=torch.int32)
    req_to_token[0, 2] = 5

    assert torch.equal(
        store.read(1, torch.tensor([1, 2, 3])),
        torch.cat([first_rows[:2], second_rows]).cpu(),
    )
    assert torch.equal(store.read(0, torch.tensor([2])), reused_rows.cpu())
