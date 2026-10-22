# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array

import pytest
import torch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.prompt_hidden import (
    PromptHiddenCapture,
    PromptHiddenRequest,
)

PROMPT_LENGTH = 10
PINNED_ROWS = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned host rows need CUDA"
)


class FreeAllocator:
    device = "cpu"

    def free(self, value: torch.Tensor) -> None:
        pass

    def available_size(self) -> int:
        return 1 << 30


def matched_prefix_length(capture: PromptHiddenCapture) -> int:
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=FreeAllocator(),
            page_size=1,
        )
    )
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=array("q", range(PROMPT_LENGTH))),
            value=torch.arange(PROMPT_LENGTH),
        )
    )
    req = PromptHiddenRequest(
        "speech", "", list(range(PROMPT_LENGTH)), SamplingParams(max_new_tokens=1)
    )
    req.prompt_hidden = capture
    OmniScheduler.normalize_req_token_arrays(req)
    req.init_next_round_input(cache)
    return req.prefix_len


@pytest.mark.accelerator
@PINNED_ROWS
def test_prompt_rows_land_at_their_positions_and_hand_off_once() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([1, 4, 5]))
    hidden_states = torch.arange(18.0, device="cuda").view(6, 3)

    capture.add_chunk(0, hidden_states[:3])
    with pytest.raises(RuntimeError, match="missing"):
        capture.take()
    capture.add_chunk(3, hidden_states[3:])
    torch.cuda.synchronize()

    assert torch.equal(capture.take(), hidden_states[[1, 4, 5]].cpu())
    assert capture.first_owed_position() is None
    assert not capture.is_owed_in(0, 6)


@pytest.mark.accelerator
@PINNED_ROWS
def test_a_request_matches_no_cached_prefix_past_the_first_row_it_owes() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([4, 7]))
    assert matched_prefix_length(capture) == 4

    capture.add_chunk(4, torch.zeros(3, 2, device="cuda"))
    assert matched_prefix_length(capture) == 7

    capture.add_chunk(7, torch.zeros(1, 2, device="cuda"))
    capture.take()
    assert matched_prefix_length(capture) == PROMPT_LENGTH - 1
