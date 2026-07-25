# SPDX-License-Identifier: Apache-2.0
"""KV slot ownership across the async fast path, against the REAL sglang
ScheduleBatch / Req / ReqToTokenPool / TokenToKVPoolAllocator / RadixCache on CPU.

Every KV slot must have exactly one owner at all times: the radix tree, the
allocator's free list, or a live request. The stub-based tests in
test_async_decode.py pin the batch bookkeeping; these pin the invariant that
bookkeeping exists to protect, which no stub can express — a slot that is both
tree-owned and free is handed to a second request as soon as that tree node is
evicted.

The timeline reproduced is the fast path of _event_loop_async_decode with one
step in flight (is_mixed_chunk, i.e. the Qwen3-Omni thinker):

    iter K  : running.prepare_for_decode  -> launch          pending = step K
    iter K+1: new.prepare_for_extend
              running.prepare_for_decode                     <- overrun slot
              new.mix_with_running(running)                  <- MIXED batch
              drain: resolve step K, a folded req finishes
"""

from __future__ import annotations

import types

import pytest
import torch

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

POOL_SIZE = 256
MAX_CTX = 64


@pytest.fixture()
def server_args():
    """Real ServerArgs installed as the process-global, restored afterwards."""
    from sglang.srt import server_args as sa_mod

    previous = getattr(sa_mod, "_global_server_args", None)
    args = sa_mod.ServerArgs(model_path="dummy", tokenizer_path="dummy", page_size=1)
    # CPU: take the non-triton req_to_token write path in write_cache_indices
    args.attention_backend = "torch_native"
    sa_mod.set_global_server_args_for_scheduler(args)
    try:
        yield args
    finally:
        sa_mod._global_server_args = previous


MODEL_CONFIG = types.SimpleNamespace(
    is_encoder_decoder=False,
    is_matryoshka=False,
    is_multimodal=False,
    vocab_size=128,
    context_len=MAX_CTX,
    think_end_id=None,
    hf_config=types.SimpleNamespace(),
)


def _pools():
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache

    req_to_token = ReqToTokenPool(
        size=8, max_context_len=MAX_CTX, device="cpu", enable_memory_saver=False
    )
    kv = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=torch.float16,
        head_num=1,
        head_dim=4,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    alloc = TokenToKVPoolAllocator(
        size=POOL_SIZE, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False
    )
    tree = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token,
            token_to_kv_pool_allocator=alloc,
            page_size=1,
        )
    )
    return req_to_token, alloc, tree


def _req(rid, ids, tree, max_new_tokens=16):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    params = SamplingParams(max_new_tokens=max_new_tokens)
    params.normalize(tokenizer=None)
    r = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(ids),
        sampling_params=params,
        vocab_size=128,
    )
    r.fill_ids = list(ids)
    r.extend_input_len = len(ids)
    r.prefix_indices = torch.empty((0,), dtype=torch.int64)
    r.last_node = tree.root_node
    tree.inc_lock_ref(r.last_node)
    return r


def _batch(reqs, req_to_token, alloc, tree):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=req_to_token,
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        model_config=MODEL_CONFIG,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


class _Result:
    """The GenerationBatchResult fields upstream's processors read."""

    def __init__(self, n):
        self.copy_done = None
        self.routed_experts_output = None
        self.indexer_topk_output = None
        self.logits_output = None
        self.next_token_ids = torch.full((n,), 55, dtype=torch.long)
        self.can_run_cuda_graph = False
        self.num_correct_drafts = None
        self.extend_input_len_per_req = None
        self.extend_logprob_start_len_per_req = None


def _scheduler(req_to_token, alloc, tree, server_args):
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    s = OmniScheduler.__new__(OmniScheduler)
    s.page_size = 1
    s.server_args = server_args
    s.model_config = MODEL_CONFIG
    s.token_to_kv_pool_allocator = alloc
    s.tree_cache = tree
    s.req_to_token_pool = req_to_token
    s.enable_overlap = False
    s.enable_overlap_mlx = False
    s.enable_metrics = False
    s.enable_hisparse = False
    s.enable_hierarchical_cache = False
    s.is_generation = True
    s.num_generated_tokens = 0
    s.forward_ct_decode = 0
    s.decode_offload_manager = None
    s.is_stats_logging_rank = False
    s.spec_algorithm = SpeculativeAlgorithm.NONE
    for name in (
        "stream_output",
        "report_decode_stats",
        "report_prefill_stats",
        "maybe_collect_routed_experts",
        "maybe_collect_indexer_topk",
        "maybe_collect_customized_info",
        "_mamba_prefix_cache_update",
    ):
        setattr(s, name, lambda *a, **k: None)
    return s


def _tree_slots(tree):
    out, stack = [], [tree.root_node]
    while stack:
        node = stack.pop()
        if node.value is not None:
            vals = node.value
            out.extend(
                int(v) for v in (vals.tolist() if hasattr(vals, "tolist") else vals)
            )
        stack.extend(node.children.values())
    return out


def _free_slots(alloc):
    return [int(v) for v in alloc.free_pages.tolist()] + [
        int(v) for v in alloc.release_pages.tolist()
    ]


def _build_mixed(server_args):
    """Run the real prefill -> decode -> mixed-fold sequence; return the state
    just before the drain, with the pending (in-flight) step's batch."""
    from sglang.srt.managers.scheduler import Scheduler as Upstream

    req_to_token, alloc, tree = _pools()
    sched = _scheduler(req_to_token, alloc, tree, server_args)

    r0 = _req("r0", [1, 2, 3, 4], tree, max_new_tokens=2)  # finishes in the drain
    r1 = _req("r1", [5, 6, 7], tree)
    running = _batch([r0, r1], req_to_token, alloc, tree)
    running.prepare_for_extend()
    Upstream.process_batch_result_prefill(sched, running, _Result(2))

    running.output_ids = torch.tensor([31, 32], dtype=torch.int64)
    running.prepare_for_decode()  # step K
    pending = running.copy()  # launched, not yet resolved

    p0 = _req("p0", [20, 21, 22, 23, 24], tree)
    new = _batch([p0], req_to_token, alloc, tree)
    new.prepare_for_extend()
    running.output_ids = torch.tensor([41, 42], dtype=torch.int64)
    running.prepare_for_decode()  # the overrun slot
    new.mix_with_running(running)
    new.decoding_reqs = list(running.reqs)
    return sched, new, pending, (req_to_token, alloc, tree), (r0, r1, p0)


def test_drain_leaves_the_overrun_step_slot_owned_by_the_radix_tree(server_args):
    """prepare_for_decode counts the step it allocates in kv_committed_len, so at
    drain-release kv_committed_len == len(origin) + len(output) and
    cache_finished_req INSERTS that slot into the radix tree — it does not
    truncate below it. Nothing may hand that slot back to the allocator."""
    from sglang.srt.managers.scheduler import Scheduler as Upstream

    sched, mixed, pending, (r2t, alloc, tree), (r0, _r1, _p0) = _build_mixed(
        server_args
    )
    overrun = int(mixed.out_cache_loc[mixed.extend_lens[0]])  # r0's folded row

    Upstream.process_batch_result_decode(sched, pending, _Result(2))

    assert r0.finished()
    assert r0.kv_committed_len == len(r0.origin_input_ids) + len(r0.output_ids)
    assert overrun in _tree_slots(tree)
    assert overrun not in _free_slots(alloc)


def test_compensating_free_of_the_overrun_slot_aliases_it(server_args):
    """Characterization of the hazard above.

    Freeing the overrun slot after the drain — on the assumption that
    cache_finished_req truncated below it — leaves the slot owned by the radix
    tree AND queued in the allocator. Evicting that node queues it a second
    time, so two requests decode into the same KV slot.
    """
    from sglang.srt.managers.scheduler import Scheduler as Upstream
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    sched, mixed, pending, (r2t, alloc, tree), (r0, _r1, _p0) = _build_mixed(
        server_args
    )
    overrun = int(mixed.out_cache_loc[mixed.extend_lens[0]])
    Upstream.process_batch_result_decode(sched, pending, _Result(2))

    alloc.free(torch.tensor([overrun], dtype=torch.int64))

    assert overrun in _tree_slots(tree) and overrun in _free_slots(alloc)
    tree.evict(EvictParams(num_tokens=len(_tree_slots(tree))))
    free = _free_slots(alloc)
    assert free.count(overrun) == 2, "the same slot is now free twice over"
