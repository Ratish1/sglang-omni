# SPDX-License-Identifier: Apache-2.0

from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler


class TokenAllocator:
    device = "cpu"
    page_size = 1

    def __init__(self, capacity: int):
        self.free_tokens = set(range(1, capacity + 1))

    def available_size(self):
        return len(self.free_tokens)

    def alloc(self, count: int):
        assert count <= self.available_size()
        indices = sorted(self.free_tokens)[:count]
        self.free_tokens.difference_update(indices)
        return torch.tensor(indices, dtype=torch.int32)

    def free_segments(self, segments):
        for indices, _ in segments:
            tokens = indices.tolist()
            assert len(set(tokens)) == len(tokens)
            assert self.free_tokens.isdisjoint(tokens)
            self.free_tokens.update(tokens)

    def check_decode_capacity(self, *, num_tokens, tree_cache):
        return self.available_size() >= num_tokens


@pytest.fixture
def runtime():
    with get_context().override_server_args(
        disable_radix_cache=True, page_size=1, retraction_policy="length"
    ):
        yield


def make_pair(index=0, *, prompt=8, generated=4, max_new=100):
    params = SamplingParams(max_new_tokens=max_new)
    pair = [
        Req(
            rid=f"song{index}{suffix}",
            origin_input_text="",
            origin_input_ids=[1] * prompt,
            sampling_params=params,
        )
        for suffix in ("", "-cfg")
    ]
    for req in pair:
        req.output_ids = array("q", [2] * generated)
        req._omni_data = SimpleNamespace()
    pair[0]._omni_data.cfg_uncond = pair[1]._omni_data
    return pair


def make_batch(reqs, *, free_tokens=0):
    lengths = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
    allocator = TokenAllocator(sum(lengths) + free_tokens)
    pool = ReqToTokenPool(len(reqs), max(lengths), "cpu", False)
    cache = ChunkCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    rows = pool.alloc(reqs)
    for req, row, length in zip(reqs, rows, lengths, strict=True):
        pool.req_to_token[row, :length] = allocator.alloc(length)
        req.kv.kv_committed_len = length
        req.kv.kv_allocated_len = length
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.req_pool_indices = torch.tensor(rows)
    batch.req_pool_indices_cpu = torch.tensor(rows)
    batch.seq_lens = torch.tensor(lengths)
    batch.seq_lens_cpu = torch.tensor(lengths)
    batch.orig_seq_lens = torch.tensor(lengths)
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, 16)
    return batch


def test_retraction_releases_a_whole_pair_and_filters_real_batch(runtime):
    first = make_pair(0, generated=4)
    second = make_pair(1, generated=8)
    batch = make_batch(first + second)
    retained_rows = batch.req_pool_indices[2:].clone()
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)

    retracted, aborted = scheduler._retract_decode_pairs(batch)

    assert retracted == [tuple(first)]
    assert aborted is None
    assert batch.reqs == second
    assert torch.equal(batch.req_pool_indices, retained_rows)
    assert len(batch.sampling_info) == 2
    assert batch.token_to_kv_pool_allocator.available_size() == 22
    assert batch.req_to_token_pool.available_size() == 2
    for req in first:
        assert req.is_retracted and req.retraction_count == 1
        assert not req.kv.holds_kv
        assert list(req.output_ids) == [2] * 4
    assert batch.check_decode_mem()


def test_last_pair_failure_releases_both_rows(runtime):
    pair = make_pair()
    batch = make_batch(pair)
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)

    retracted, aborted = scheduler._retract_decode_pairs(batch)

    assert retracted == []
    assert aborted == tuple(pair)
    assert batch.is_empty()
    assert batch.token_to_kv_pool_allocator.available_size() == 22
    assert batch.req_to_token_pool.available_size() == 2
    assert all(not req.kv.holds_kv for req in pair)
