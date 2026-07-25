# SPDX-License-Identifier: Apache-2.0
"""Phase 0 semantic guards: the pinned-SGLang lag-accounting contract.

The async-generation design (tasks/async_generation_scheduler_first_principles
_plan_20260725.md) rests on running upstream ``enable_overlap=True`` batch
accounting under Omni's own single-worker machinery. Each test here pins one
upstream behavior the design substitutes into or depends on, against the REAL
pinned ScheduleBatch / Req / pools / RadixCache / result processors on CPU.

If a dependency bump changes any of these, the corresponding test fails BEFORE
the scheduler invariant can silently weaken — this file is the version guard
the plan requires.

Guarded call sites (the substitution list):

  G1  mix_with_running        delta = 0 iff enable_overlap: folded prefixes are
                              correct exactly when host output lags one step
  G2  process_batch_result_*  the finished/retracted row skip is native under
                              enable_overlap and absent without it
  G3  prepare_for_decode      non-in-place seq_lens updates under overlap (a
                              pending snapshot must not see +1 in place)
  G4  prepare_for_decode      consumes batch.output_ids as next input_ids and
                              resets it to None — the relay substitution point
  G5  filter_batch            slices output_ids positionally on the device —
                              a device-tensor relay value filters correctly
  G6  barrier-resume wart     with HOST-CURRENT rows, overlap accounting
                              over-counts folded prefixes by exactly one; pins
                              why the per-row delta adaptation must exist
"""

from __future__ import annotations

import types

import pytest
import torch

POOL_SIZE = 256
MAX_CTX = 64


@pytest.fixture()
def server_args():
    from sglang.srt import server_args as sa_mod

    previous = getattr(sa_mod, "_global_server_args", None)
    args = sa_mod.ServerArgs(model_path="dummy", tokenizer_path="dummy", page_size=1)
    args.attention_backend = "torch_native"  # CPU req_to_token write path
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


def _batch(reqs, pools, *, enable_overlap):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    req_to_token, alloc, tree = pools
    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=req_to_token,
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        model_config=MODEL_CONFIG,
        enable_overlap=enable_overlap,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _decoded_running_batch(pools, *, enable_overlap, lagged):
    """A 2-req running batch one decode step in. ``lagged=True`` leaves the last
    sampled token uncommitted on the host (the in-flight step's publication),
    which is the steady state a lagged scheduler plans from; ``lagged=False``
    commits everything (the barrier-resume / sync state)."""
    r0 = _req("r0", [1, 2, 3, 4], pools[2])
    r1 = _req("r1", [5, 6, 7], pools[2])
    running = _batch([r0, r1], pools, enable_overlap=enable_overlap)
    running.prepare_for_extend()
    # prefill commit: token x_a lands on the host
    r0.output_ids.append(11)
    r1.output_ids.append(12)
    running.output_ids = torch.tensor([11, 12], dtype=torch.int64)
    running.prepare_for_decode()  # step F1: input x_a, allocates x_a's slot
    # F1 publishes its sample x_b to the batch (device relay); host commit of
    # x_b happens only if the pipeline is drained.
    running.output_ids = torch.tensor([21, 22], dtype=torch.int64)
    if not lagged:
        r0.output_ids.append(21)
        r1.output_ids.append(22)
    return running, (r0, r1)


def _mixed(pools, running, *, enable_overlap):
    p0 = _req("p0", [40, 41, 42], pools[2])
    new = _batch([p0], pools, enable_overlap=enable_overlap)
    new.prepare_for_extend()
    running.prepare_for_decode()  # folds need the decode-prepared running batch
    new.mix_with_running(running)
    return new


def _folded_consistency(mixed):
    """prefix + extend == seq_len per folded row; the forward's attention
    metadata is built from exactly these three."""
    return [
        int(p) + int(e) - int(s)
        for p, e, s in zip(
            mixed.prefix_lens[1:], mixed.extend_lens[1:], mixed.seq_lens.tolist()[1:]
        )
    ]


# ---------------------------------------------------------------------------
# G1 / G6 — mix_with_running delta
# ---------------------------------------------------------------------------


def test_g1_overlap_delta_is_correct_exactly_when_host_lags(server_args):
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=True, lagged=True)
    mixed = _mixed(pools, running, enable_overlap=True)
    assert _folded_consistency(mixed) == [
        0,
        0,
    ], "enable_overlap=True folded prefixes must be exact under one-step host lag"


def test_g1_sync_delta_is_correct_exactly_when_host_is_current(server_args):
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=False, lagged=False)
    mixed = _mixed(pools, running, enable_overlap=False)
    assert _folded_consistency(mixed) == [
        0,
        0,
    ], "enable_overlap=False folded prefixes must be exact with host-current rows"


def test_g6_overlap_delta_overcounts_host_current_rows_by_one(server_args):
    """The barrier-resume wart: after a drain commits everything, rows are
    host-current, and the static delta=0 over-counts each folded prefix by
    exactly one. This is WHY the design keys delta on per-row relay state
    (resolved -> -1, unresolved -> 0) instead of the batch flag. If this test
    ever starts passing with [0, 0], upstream changed the convention and the
    adaptation must be re-audited."""
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=True, lagged=False)
    mixed = _mixed(pools, running, enable_overlap=True)
    assert _folded_consistency(mixed) == [
        1,
        1,
    ], "expected the documented +1 over-count on host-current rows"


# ---------------------------------------------------------------------------
# G2 — native finished-row skip in result processing
# ---------------------------------------------------------------------------


class _Result:
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


def _scheduler(pools, server_args, *, enable_overlap):
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    req_to_token, alloc, tree = pools
    s = OmniScheduler.__new__(OmniScheduler)
    s.page_size = 1
    s.server_args = server_args
    s.model_config = MODEL_CONFIG
    s.token_to_kv_pool_allocator = alloc
    s.tree_cache = tree
    s.req_to_token_pool = req_to_token
    s.enable_overlap = enable_overlap
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


def _finished_row_in_next_step(pools, server_args, *, enable_overlap):
    """Drive the real processors: r0 finishes at step N's commit; step N+1's
    batch (already prepared) still carries r0. Returns (scheduler, batch)."""
    from sglang.srt.managers.scheduler import Scheduler as Upstream

    sched = _scheduler(pools, server_args, enable_overlap=enable_overlap)
    r0 = _req("r0", [1, 2, 3, 4], pools[2], max_new_tokens=2)
    r1 = _req("r1", [5, 6, 7], pools[2])
    batch = _batch([r0, r1], pools, enable_overlap=enable_overlap)
    batch.prepare_for_extend()
    Upstream.process_batch_result_prefill(sched, batch, _Result(2))
    batch.output_ids = torch.tensor([21, 22], dtype=torch.int64)
    batch.prepare_for_decode()  # step N
    pending = batch.copy()
    batch.output_ids = torch.tensor([31, 32], dtype=torch.int64)
    batch.prepare_for_decode()  # step N+1, prepared before N committed
    Upstream.process_batch_result_decode(sched, pending, _Result(2))  # N commits
    assert r0.finished()
    return sched, batch


def test_g2_overlap_result_processing_skips_the_finished_row(server_args):
    from sglang.srt.managers.scheduler import Scheduler as Upstream

    pools = _pools()
    sched, batch = _finished_row_in_next_step(pools, server_args, enable_overlap=True)
    Upstream.process_batch_result_decode(sched, batch, _Result(2))  # must not raise


def test_g2_sync_result_processing_double_releases_the_finished_row(server_args):
    """The absence guard: without enable_overlap the processor re-finalizes the
    finished row and release asserts — at the req_pool guard (req_pool_idx was
    nulled by the first release) or, with a still-live pool row, at
    pop_committed_kv_cache. Either way the sync path cannot process a
    prior-finished row: this is the double-free every generation of stale-row
    patch existed to dodge; native overlap semantics do not have the problem."""
    from sglang.srt.managers.scheduler import Scheduler as Upstream

    pools = _pools()
    sched, batch = _finished_row_in_next_step(pools, server_args, enable_overlap=False)
    with pytest.raises(AssertionError, match="already freed|freeing before alloc"):
        Upstream.process_batch_result_decode(sched, batch, _Result(2))


# ---------------------------------------------------------------------------
# G3 / G4 — prepare_for_decode under overlap
# ---------------------------------------------------------------------------


def test_g3_overlap_seq_lens_update_is_not_in_place(server_args):
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=True, lagged=True)
    snapshot = running.seq_lens  # a pending step's snapshot aliases this tensor
    before = snapshot.clone()
    running.prepare_for_decode()
    assert running.seq_lens is not snapshot, "overlap must allocate new seq_lens"
    assert torch.equal(snapshot, before), "the pending snapshot must be unchanged"


def test_g3_sync_seq_lens_update_is_in_place(server_args):
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=False, lagged=False)
    snapshot = running.seq_lens
    running.prepare_for_decode()
    assert (
        running.seq_lens is snapshot
    ), "sync accounting mutates in place — a lagged stage must never rely on it"


def test_g4_prepare_for_decode_consumes_output_ids_as_input(server_args):
    """The relay substitution point: the next forward's input IS whatever the
    launch published into batch.output_ids, and the field is reset to None.
    OmniGenerationRelay slots in exactly here — it becomes the single owner of
    that published value instead of the overloaded output_ids field."""
    pools = _pools()
    running, _ = _decoded_running_batch(pools, enable_overlap=True, lagged=True)
    published = running.output_ids
    running.prepare_for_decode()
    assert torch.equal(running.input_ids, published.to(torch.int64))
    assert running.output_ids is None


# ---------------------------------------------------------------------------
# G5 — filter_batch slices the published value positionally
# ---------------------------------------------------------------------------


def test_g5_filter_batch_keeps_relay_rows_aligned(server_args):
    pools = _pools()
    running, (r0, _r1) = _decoded_running_batch(pools, enable_overlap=True, lagged=True)
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    r0.finished_reason = FINISH_LENGTH(length=1)
    running.filter_batch()
    assert [r.rid for r in running.reqs] == ["r1"]
    assert running.output_ids.tolist() == [
        22
    ], "the surviving row's published value must follow it through the filter"
