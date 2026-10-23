"""Review-only CUDA checks for the target adapter's KV-plan handoff.

The unified allocator fixture follows SGLang's test_kv_loc_plan setup. Model
execution is a controlled boundary; batch construction, plans, allocation,
translation and draft ID consumers are real production implementations.
"""

from array import array
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import UnifiedSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.layout.fused_draft import DenseDraftRegion, FusedDraftPlacement
from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_kv_inject import TargetHiddenKvInjector
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass
class CacheBoundary:
    allocator: SimpleNamespace | None = None

    def attach_allocator(self, allocator: SimpleNamespace) -> None:
        self.allocator = allocator


class HybridCacheBoundary:
    def __init__(self) -> None:
        self.full_kv_pool = CacheBoundary()
        self.swa_kv_pool = CacheBoundary()
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator) -> None:
        self.full_allocator = full_allocator
        self.swa_allocator = swa_allocator


class PrefillBoundary(ModelRunner):
    def __init__(self, model_worker: SimpleNamespace, has_token_ids: bool) -> None:
        self.tp_worker = model_worker
        self.has_token_ids = has_token_ids
        self.sample_calls = 0
        self.hook_calls: list[str] = []
        self.last_forward_batch: ForwardBatch | None = None
        self.token_ids = torch.tensor([7, 8], device="cuda")

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        self.hook_calls.append("before")
        self.last_forward_batch = forward_batch
        assert forward_batch.kv_loc_plan is not None
        assert [request.request_id for request in requests] == ["a", "b"]

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        self.hook_calls.append("forward")
        return GenerationBatchResult(
            next_token_ids=self.token_ids if self.has_token_ids else None,
            can_run_cuda_graph=True,
        )

    def cleanup_prefill(self, forward_batch, schedule_batch, requests) -> None:
        self.hook_calls.append("cleanup")
        assert forward_batch.kv_loc_plan is self.last_forward_batch.kv_loc_plan

    def sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests):
        self.sample_calls += 1
        return self.token_ids


@pytest.fixture
def allocated_readers():
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        full = MHASubPoolSpec(
            name="full", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.bfloat16, grow_direction="down",
            draft_region=DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16,
            ),
        )
        window = MHASubPoolSpec(
            name="swa", layer_num=1, head_num=2, head_dim=4,
            store_dtype=torch.bfloat16, grow_direction="up",
        )
        pool = UnifiedKVPool(
            total_bytes=32 * full.entry_bytes() + 16 * window.entry_bytes(),
            sub_pool_specs=[full, window], device="cuda",
            enable_memory_saver=False, page_size=2,
            fused_draft=FusedDraftPlacement(region=full.draft_region, runner_lane_counts=(1,)),
        )
        cache = HybridCacheBoundary()
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool, kvcache=cache, device="cuda",
            full_max_total_num_tokens=32, swa_max_total_num_tokens=16,
            page_size=2, need_sort=False, forward_stream=None,
        )
        fused_pool = UnifiedDraftKVPool(
            unified_buffer=pool, host_sub_pool_name="full", host_allocator=allocator,
            layer_lanes={0: 0}, page_size=2,
        )
        request_to_token = torch.zeros((4, 16), dtype=torch.int32, device="cuda")
        for row in (1, 2):
            token_ids = allocator.alloc(4)
            assert token_ids is not None
            request_to_token[row, :4] = token_ids.to(torch.int32)
        readers = [
            KVIndexTranslator(
                req_to_token=request_to_token, token_to_kv_pool_allocator=allocator,
                token_to_kv_pool=cache_view, page_size=2, device="cuda",
            )
            for cache_view in (cache, fused_pool, CacheBoundary())
        ]
        yield readers, request_to_token, allocator
        torch.cuda.synchronize()
    finally:
        reset_context()


def make_batch(request_to_token: torch.Tensor, is_prefill_only: bool) -> ScheduleBatch:
    requests = [
        Req(request_id, "prompt", array("q", [1, 2, 3, 4]), SamplingParams())
        for request_id in ("a", "b")
    ]
    for request in requests:
        request.omni_data = None
    rows = torch.tensor([2, 1], device="cuda", dtype=torch.int64)
    return ScheduleBatch(
        reqs=requests, device="cuda", forward_mode=ForwardMode.EXTEND,
        req_pool_indices=rows, req_pool_indices_cpu=rows.cpu(),
        seq_lens=torch.tensor([4, 4], device="cuda"), seq_lens_cpu=torch.tensor([4, 4]),
        seq_lens_sum=8, orig_seq_lens=torch.tensor([4, 4], device="cuda"),
        out_cache_loc=request_to_token[rows, 2:4].to(torch.int64).reshape(-1),
        input_ids=torch.tensor([3, 4, 3, 4], device="cuda"),
        extend_lens=[2, 2], prefix_lens=[2, 2], extend_num_tokens=4,
        extend_logprob_start_lens=[0, 0], spec_algorithm=SpeculativeAlgorithm.NONE,
        is_prefill_only=is_prefill_only,
        model_config=SimpleNamespace(is_matryoshka=False),
    )


def make_worker(translator: KVIndexTranslator) -> SimpleNamespace:
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_index_translator=translator, is_draft_worker=False, device="cuda",
            model_config=SimpleNamespace(requires_mm_token_modalities=False, model_is_mrope=False),
            lora_manager=None, ngram_embedding_manager=SimpleNamespace(enabled=False),
            prefill_attention_backend_str="fa3",
        ),
        model_config=None, device="cuda", random_seed=0,
    )


@pytest.mark.parametrize("is_translated", [False, True])
@pytest.mark.parametrize("has_token_ids,is_prefill_only", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("requests_plan", [False, True])
def test_real_prefill_plan_reaches_both_draft_id_consumers(
    allocated_readers, is_translated: bool, has_token_ids: bool,
    is_prefill_only: bool, requests_plan: bool,
) -> None:
    readers, request_to_token, allocator = allocated_readers
    target, fused, private = readers
    reader = target if is_translated else private
    batch = make_batch(request_to_token, is_prefill_only)
    virtual_ids = batch.out_cache_loc.clone()
    worker = make_worker(reader)
    runner = PrefillBoundary(worker, has_token_ids)
    adapter = SpeculativeTargetWorker(worker)
    adapter.attach_runner(runner)
    output = adapter.forward_batch_generation(
        batch, return_kv_loc_plan=requests_plan, capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    forward_batch = runner.last_forward_batch
    plan = forward_batch.kv_loc_plan
    assert output.kv_loc_plan is (plan if requests_plan else None)
    assert runner.hook_calls == ["before", "forward", "cleanup"]
    assert runner.sample_calls == int(not has_token_ids and not is_prefill_only)
    assert forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
    torch.testing.assert_close(batch.out_cache_loc, virtual_ids, rtol=0, atol=0)
    if is_prefill_only:
        torch.testing.assert_close(output.next_token_ids, torch.zeros(2, device="cuda", dtype=torch.int64))
    else:
        assert output.next_token_ids is runner.token_ids
    if requests_plan:
        for draft_reader in ((fused, private) if is_translated else (private,)):
            draft_runner = SimpleNamespace(kv_index_translator=draft_reader)
            consumer = SimpleNamespace(draft_model_runner=draft_runner)
            dflash_ids = DFlashWorkerV2._draft_write_ids(consumer, output.kv_loc_plan)
            dspark_ids = TargetHiddenKvInjector.ids_for(consumer, output.kv_loc_plan)
            assert dflash_ids is dspark_ids
            expected = allocator.translate_write_loc(virtual_ids) if draft_reader is fused else virtual_ids
            torch.testing.assert_close(dflash_ids, expected, rtol=0, atol=0)
            if draft_reader is fused:
                assert dflash_ids is forward_batch.out_cache_loc
                assert not torch.equal(dflash_ids, virtual_ids)
            else:
                assert dflash_ids is batch.out_cache_loc
    else:
        pass


@pytest.mark.parametrize("requests_plan", [False, True])
def test_verify_returns_the_post_forward_plan_without_sampling(allocated_readers, requests_plan: bool) -> None:
    readers, request_to_token, allocator = allocated_readers
    worker = make_worker(readers[0])
    batch = make_batch(request_to_token, False)
    forward_batch = ForwardBatch.init_new(batch, worker.model_runner)
    old_plan = forward_batch.kv_loc_plan
    replacement = readers[0].own_plan(batch)
    output = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros((4, 8), device="cuda")),
        can_run_graph=True, expert_distribution_metrics={"metric": 1},
        routed_experts_output=torch.tensor([1], device="cuda"),
        indexer_topk_output=torch.tensor([2], device="cuda"),
    )

    def forward(*, forward_batch: ForwardBatch) -> SimpleNamespace:
        replacement.bind(forward_batch, readers[0])
        return output

    worker.model_runner.forward = forward
    adapter = SpeculativeTargetWorker(worker)
    result = adapter.forward_batch_generation(None, forward_batch=forward_batch, is_verify=True, return_kv_loc_plan=requests_plan)
    assert replacement is not old_plan
    assert result.kv_loc_plan is (replacement if requests_plan else None)
    assert result.logits_output is output.logits_output
    assert result.expert_distribution_metrics is output.expert_distribution_metrics
    assert result.routed_experts_output is output.routed_experts_output
    assert result.indexer_topk_output is output.indexer_topk_output
    assert result.can_run_cuda_graph is True
    assert result.next_token_ids is None
