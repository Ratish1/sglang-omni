# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS re-prefill after KV-pressure retraction (#1555)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace

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
from sglang.srt.sampling.penaltylib import BatchedRepetitionPenalizer
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

HIDDEN = 2
CODEBOOKS = 3


class FeedbackModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_feedback_embedding = torch.nn.Embedding(4, HIDDEN)
        self.decode_feedback_embedding.weight.requires_grad_(False)
        self.frame_weights = torch.nn.Parameter(
            torch.arange(CODEBOOKS * HIDDEN, dtype=torch.float32).reshape(
                CODEBOOKS, HIDDEN
            ),
            requires_grad=False,
        )

    def feedback_rows_from_codes(self, frame_codes: torch.Tensor) -> torch.Tensor:
        return frame_codes.to(torch.float32) @ self.frame_weights


def make_runner() -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.row_ids_cache = None
    runner.model = FeedbackModel()
    runner.feedback_pool = torch.zeros(8, HIDDEN)
    return runner


def make_sched_req(
    *,
    prompt: torch.Tensor,
    extend_len: int,
    prefix_len: int = 0,
    frame_codes: list[list[int]] | None = None,
    text_rows: torch.Tensor | None = None,
    rid: str = "req",
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        data=SimpleNamespace(
            req=SimpleNamespace(
                rid=rid,
                prefix_indices=list(range(prefix_len)),
                extend_range=SimpleNamespace(length=extend_len),
                origin_input_ids=list(range(prompt.shape[0])),
            ),
            prompt_input_embeds=prompt,
            prefill_input_embeds=prompt,
            output_codes=[torch.tensor(codes) for codes in frame_codes or []],
            text_rows=(
                torch.zeros(1, prompt.shape[-1]) if text_rows is None else text_rows
            ),
        ),
    )


def test_decode_rows_add_each_slot_feedback_to_the_text_row_of_its_position() -> None:
    runner = make_runner()
    runner.feedback_pool[2] = torch.tensor([1.0, 2.0])
    runner.feedback_pool[5] = torch.tensor([3.0, 4.0])
    text_rows = torch.tensor([[10.0, 20.0], [30.0, 40.0], [0.5, 0.5]])
    in_text = make_sched_req(
        prompt=torch.zeros(4, HIDDEN), extend_len=0, text_rows=text_rows
    )
    past_text = make_sched_req(
        prompt=torch.zeros(6, HIDDEN), extend_len=0, text_rows=text_rows, rid="late"
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([9, 9], dtype=torch.long),
        req_pool_indices=torch.tensor([2, 5]),
        seq_lens_cpu=torch.tensor([4 + 2, 6 + 7]),
    )

    runner.write_feedback_buffers(forward_batch, [in_text, past_text])

    assert torch.equal(
        runner.model.decode_feedback_embedding.weight[:2],
        torch.tensor([[31.0, 42.0], [3.5, 4.5]]),
    )
    assert forward_batch.input_ids.tolist() == [0, 1]


def test_reprefill_after_retract_replays_prompt_plus_generated() -> None:
    # note (Richard Wang): 460 plus 134 is the #1555 594 vs 460 mismatch
    prompt_len, generated_len = 460, 134
    prompt = torch.arange(prompt_len * HIDDEN, dtype=torch.float32).reshape(
        prompt_len, HIDDEN
    )
    frame_codes = [[step, step + 1, step + 2] for step in range(generated_len)]
    text_rows = torch.tensor([[100.0, 200.0], [300.0, 400.0], [0.0, 0.0]])
    runner = make_runner()
    extend_len = prompt_len + generated_len
    sched_req = make_sched_req(
        prompt=prompt,
        extend_len=extend_len,
        frame_codes=frame_codes,
        text_rows=text_rows,
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(extend_len, dtype=torch.long))

    out = runner.build_prefill_input_embeds(forward_batch, [sched_req])

    feedback_rows = runner.model.feedback_rows_from_codes(torch.tensor(frame_codes))
    assert out.shape[0] == extend_len
    assert torch.equal(out[:prompt_len], prompt)
    assert torch.equal(
        out[prompt_len : prompt_len + 2], feedback_rows[:2] + text_rows[:2]
    )
    assert torch.equal(out[prompt_len + 2 :], feedback_rows[2:] + text_rows[2])


def test_reprefill_restores_retained_repetition_penalty_history() -> None:
    retained_req = SimpleNamespace(
        output_ids=[2, 5, 2],
        sampling_params=SimpleNamespace(repetition_penalty=1.05),
    )
    fresh_req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(repetition_penalty=1.05),
    )
    identity_req = SimpleNamespace(
        output_ids=[3],
        sampling_params=SimpleNamespace(repetition_penalty=1.0),
    )
    reqs = [retained_req, fresh_req, identity_req]

    class PenaltyOrchestrator:
        vocab_size = 8
        device = "cpu"

        def __init__(self) -> None:
            self.penalizers = {}

        def reqs(self):
            return reqs

    orchestrator = PenaltyOrchestrator()
    penalizer = BatchedRepetitionPenalizer(orchestrator)
    penalizer.prepare()
    orchestrator.penalizers[BatchedRepetitionPenalizer] = penalizer
    schedule_batch = SimpleNamespace(
        reqs=reqs,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        sampling_info=SimpleNamespace(penalizer_orchestrator=orchestrator),
    )

    scaling = penalizer.get_scaling_penalties()
    expected = torch.ones(3, 8)
    expected[0, [2, 5]] = 1.05

    class ExecutionBridge:
        def forward_context(self, batch, *, isolate_sampling):
            assert batch is schedule_batch
            assert isolate_sampling
            assert torch.equal(scaling, expected)
            return contextlib.nullcontext()

    runner = make_runner()
    runner.execution_bridge = ExecutionBridge()
    with runner.execution_context(schedule_batch, isolate_sampling=True):
        pass

    assert torch.equal(scaling, expected)

    logits = torch.tensor(
        [
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
            [1.0, 2.0, 10.0, 4.0, 5.0, -10.0, 7.0, 8.0],
        ]
    )
    original_logits = logits.clone()
    penalizer.apply(logits)

    assert logits[0, 2].item() == pytest.approx(10.0 / 1.05)
    assert logits[0, 5].item() == pytest.approx(-10.0 * 1.05)
    assert torch.equal(logits[1], original_logits[1])
    assert torch.equal(logits[2], original_logits[2])


@pytest.mark.parametrize(
    ("prefix_len", "extend_len", "replayed_frames"),
    [(8, 5, slice(0, 3)), (11, 2, slice(1, 3))],
)
def test_reprefill_replays_the_frames_of_its_extend_window(
    prefix_len: int, extend_len: int, replayed_frames: slice
) -> None:
    prompt = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    frame_codes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    runner = make_runner()
    sched_req = make_sched_req(
        prompt=prompt,
        prefix_len=prefix_len,
        extend_len=extend_len,
        frame_codes=frame_codes,
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(extend_len, dtype=torch.long))

    out = runner.build_prefill_input_embeds(forward_batch, [sched_req])

    expected = torch.cat(
        [
            prompt[prefix_len:],
            runner.model.feedback_rows_from_codes(
                torch.tensor(frame_codes[replayed_frames])
            ),
        ],
        dim=0,
    )
    assert torch.equal(out, expected)


def test_reprefill_without_its_committed_frames_fails_loudly() -> None:
    prompt = torch.zeros(4, HIDDEN)
    sched_req = make_sched_req(prompt=prompt, extend_len=7, frame_codes=[[1, 2, 3]])
    forward_batch = SimpleNamespace(input_ids=torch.zeros(7, dtype=torch.long))

    with pytest.raises(RuntimeError, match="prefill embed mismatch"):
        make_runner().build_prefill_input_embeds(forward_batch, [sched_req])


def test_a_replayed_row_equals_the_row_its_decode_staged() -> None:
    prompt_len = 4
    frame_codes = [[2, 1, 0], [0, 3, 1], [1, 1, 1]]
    runner = make_runner()
    sched_req = make_sched_req(
        prompt=torch.zeros(prompt_len, HIDDEN),
        extend_len=0,
        text_rows=torch.tensor([[10.0, 20.0], [0.0, 0.0]]),
    )
    staged_rows = []
    for step, codes in enumerate(frame_codes):
        sched_req.data.output_codes.append(torch.tensor(codes))
        runner.feedback_pool[3] = runner.model.feedback_rows_from_codes(
            torch.tensor([codes])
        )[0]
        decode_batch = SimpleNamespace(
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([prompt_len + step + 1]),
        )
        runner.write_feedback_buffers(decode_batch, [sched_req])
        staged_rows.append(runner.model.decode_feedback_embedding.weight[0].clone())

    replayed_length = prompt_len + len(frame_codes)
    sched_req.data.req.extend_range = SimpleNamespace(length=replayed_length)
    prefill_batch = SimpleNamespace(
        input_ids=torch.zeros(replayed_length, dtype=torch.long)
    )
    out = runner.build_prefill_input_embeds(prefill_batch, [sched_req])

    assert torch.equal(out[prompt_len:], torch.stack(staged_rows))


def test_fresh_prefill_still_uses_prompt_only_buffer() -> None:
    prompt = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    sched_req = make_sched_req(prompt=prompt, prefix_len=1, extend_len=4)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(4, dtype=torch.long))

    out = make_runner().build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.equal(out, prompt[1:5])


@dataclass
class RadixChunkState:
    page_size: int
    allocator: TokenToKVPoolAllocator | PagedTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    cache: RadixCache
    scheduler: OmniScheduler


@pytest.fixture(params=[1, 16])
def radix_chunk_state(request: pytest.FixtureRequest) -> Iterator[RadixChunkState]:
    page_size = request.param
    with get_context().override_server_args(page_size=page_size):
        kv_pool = MHATokenToKVPool(
            size=128,
            page_size=page_size,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        if page_size == 1:
            allocator = TokenToKVPoolAllocator(
                size=128,
                dtype=torch.float16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
        else:
            allocator = PagedTokenToKVPoolAllocator(
                size=128,
                page_size=page_size,
                dtype=torch.float16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=128, device="cpu", enable_memory_saver=False
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
            )
        )
        scheduler = object.__new__(OmniScheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = cache
        scheduler.req_to_token_pool = req_to_token_pool
        yield RadixChunkState(page_size, allocator, req_to_token_pool, cache, scheduler)


def prompt_only_req(state: RadixChunkState, prompt_len: int) -> Req:
    req = Req("chunk", "", list(range(prompt_len)), SamplingParams(max_new_tokens=40))
    req._omni_prompt_only_radix = True  # noqa: leading-underscore  # production name
    OmniScheduler.normalize_req_token_arrays(req)
    state.req_to_token_pool.alloc([req])
    req.init_next_round_input(state.cache)
    return req


def write_slots(state: RadixChunkState, req: Req, start: int, end: int) -> None:
    slots = state.allocator.alloc(
        -(-(end - start) // state.page_size) * state.page_size
    )
    state.req_to_token_pool.write(
        (req.kv.req_pool_idx, slice(start, end)), slots[: end - start]
    )


def compute_chunk(state: RadixChunkState, req: Req, start: int, end: int) -> None:
    req.set_extend_range(start, end)
    req.kv.kv_committed_len = req.kv.kv_allocated_len = end


def admit_next_chunk(state: RadixChunkState, req: Req) -> None:
    adder = PrefillAdder(
        page_size=state.page_size,
        tree_cache=state.cache,
        token_to_kv_pool_allocator=state.allocator,
        running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
        new_token_ratio=1.0,
        rem_input_tokens=64,
        rem_chunk_tokens=32,
    )
    adder.add_chunked_req(req)


def assert_every_slot_returns_once(state: RadixChunkState, total: int) -> None:
    assert state.allocator.available_size() + state.cache.evictable_size() == total
    state.cache.evict(EvictParams(num_tokens=total))
    slots = state.allocator.alloc(total)
    assert slots.unique().numel() == total


@pytest.mark.parametrize("release_at_middle", [False, True])
def test_a_retracted_reprefill_advances_past_its_private_chunk(
    radix_chunk_state: RadixChunkState, release_at_middle: bool
) -> None:
    """A retracted re-prefill's next chunk starts after its private chunk, which stays out of the radix tree and frees once."""
    state = radix_chunk_state
    total = state.allocator.available_size()
    req = prompt_only_req(state, prompt_len=20)
    write_slots(state, req, 0, 16)
    compute_chunk(state, req, 0, 16)
    state.cache.cache_unfinished_req(req, chunked=True)
    req.output_ids.extend(range(30, 51))
    req.init_next_round_input()
    req.skip_radix_cache_insert = True
    write_slots(state, req, 16, 33 if release_at_middle else 41)
    compute_chunk(state, req, 16, 33)

    state.scheduler.stash_chunked_request(req)
    admit_next_chunk(state, req)

    assert (req.extend_range.start, req.extend_range.end) == (33, 41)
    assert req.kv.cache_protected_len == 16
    if release_at_middle:
        pass
    else:
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 41
    release_kv_cache(req, state.cache)
    match = state.cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req.origin_input_ids + req.output_ids))
    )
    assert len(match.device_indices) == 16
    assert_every_slot_returns_once(state, total)


def test_a_fresh_prompt_chunk_enters_the_radix_tree(
    radix_chunk_state: RadixChunkState,
) -> None:
    state = radix_chunk_state
    total = state.allocator.available_size()
    req = prompt_only_req(state, prompt_len=41)
    write_slots(state, req, 0, 32)
    compute_chunk(state, req, 0, 32)

    state.scheduler.stash_chunked_request(req)
    admit_next_chunk(state, req)

    assert (req.extend_range.start, req.extend_range.end) == (32, 41)
    assert state.cache.total_size() == 32
    release_kv_cache(req, state.cache)
    assert_every_slot_returns_once(state, total)
