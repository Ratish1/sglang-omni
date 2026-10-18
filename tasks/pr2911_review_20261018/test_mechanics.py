"""Reviewer probes for chunk output and private KV ownership."""

import pytest
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput
from tests.unit_test.qwen3_tts.test_decode_step_staging import make_runner, step


def test_middle_chunk_runs_through_real_finalize() -> None:
    """An unfinished prompt cannot consume a generated step or emit codec audio."""
    token_ids = torch.tensor([7, 9], device="cuda")
    result, forward_batch, requests = step(token_ids)
    for request in requests:
        request.data.req = Req(
            request.request_id, "", [1, 2, 3], SamplingParams(max_new_tokens=40)
        )
    requests[0].data.req.inflight_middle_chunks = 1

    def predict(
        layer0_codes: torch.Tensor,
        hidden: torch.Tensor,
        semantic_positions: torch.Tensor,
    ) -> None:
        runner.model.output_codes[:2].copy_(layer0_codes.expand(-1, 3))
        runner.model.output_embeds[:2].copy_(hidden[:, 0])

    runner = make_runner(predict, torch.device("cuda"))
    runner.output_processor = SGLangOutputProcessor()
    runner.codes_event_module = torch.cuda
    result.can_run_cuda_graph = False
    runner.collect_codes(result, forward_batch, None, requests)
    runner.finalize(
        result, forward_batch, None, SchedulerOutput(requests=requests, batch_data=None)
    )
    torch.cuda.synchronize()
    assert requests[0].data.generation_steps == 0
    assert requests[0].data.output_codes == []
    assert not requests[0].data.pending_feedback_queue
    assert requests[0].data.latest_stream_code_chunk is None
    assert requests[1].data.generation_steps == 1
    assert len(requests[1].data.output_codes) == 1


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("release_at_middle", [False, True])
def test_chunk_progress_and_private_slots_are_released(
    page_size: int, release_at_middle: bool
) -> None:
    """Re-prefill advances past private KV without making its tail shareable."""
    with get_context().override_server_args(page_size=page_size):
        token_pool = MHATokenToKVPool(
            size=128,
            page_size=page_size,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        if page_size == 1:
            allocator = TokenToKVPoolAllocator(
                size=128,
                dtype=torch.float16,
                device="cuda",
                kvcache=token_pool,
                need_sort=False,
            )
        else:
            allocator = PagedTokenToKVPoolAllocator(
                size=128,
                page_size=page_size,
                dtype=torch.float16,
                device="cuda",
                kvcache=token_pool,
                need_sort=False,
            )
        request_pool = ReqToTokenPool(
            size=4,
            max_context_len=128,
            device="cuda",
            enable_memory_saver=False,
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
            )
        )
        total = allocator.available_size()
        request = Req("chunk", "", list(range(20)), SamplingParams(max_new_tokens=40))
        OmniScheduler.normalize_req_token_arrays(request)
        request_pool.alloc([request])
        request.init_next_round_input(cache)
        prompt_slots = allocator.alloc(16)
        request_pool.write((request.kv.req_pool_idx, slice(0, 16)), prompt_slots)
        request.set_extend_range(0, 16)
        request.kv.kv_committed_len = request.kv.kv_allocated_len = 16
        cache.cache_unfinished_req(request, chunked=True)
        request.output_ids.extend(range(30, 51))
        request.init_next_round_input()
        request.skip_radix_cache_insert = True
        request._omni_prompt_only_radix = True
        private_count = 17 if release_at_middle else 25
        allocation = -(-private_count // page_size) * page_size
        private_slots = allocator.alloc(allocation)
        request_pool.write(
            (request.kv.req_pool_idx, slice(16, 16 + private_count)),
            private_slots[:private_count],
        )
        request.set_extend_range(16, 33)
        request.kv.kv_committed_len = request.kv.kv_allocated_len = 33
        scheduler = object.__new__(OmniScheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = cache
        scheduler.req_to_token_pool = request_pool
        scheduler.stash_chunked_request(request)
        adder = PrefillAdder(
            page_size=page_size,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
            new_token_ratio=1.0,
            rem_input_tokens=64,
            rem_chunk_tokens=32,
        )
        assert adder.add_chunked_req(request) is None
        assert request.extend_range.start == 33
        assert request.extend_range.end == 41
        if release_at_middle:
            request.set_extend_range(16, 33)
        else:
            request.kv.kv_committed_len = request.kv.kv_allocated_len = 41
        assert request.kv.cache_protected_len == 16
        assert cache.total_size() == 16
        release_kv_cache(request, cache)
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(request.origin_input_ids + request.output_ids)
            )
        )
        assert len(match.device_indices) == 16
        assert allocator.available_size() + cache.evictable_size() == total
        cache.evict(EvictParams(num_tokens=128))
        assert allocator.available_size() == total
        all_slots = allocator.alloc(128)
        assert all_slots.unique().numel() == 128
        allocator.free(all_slots)
