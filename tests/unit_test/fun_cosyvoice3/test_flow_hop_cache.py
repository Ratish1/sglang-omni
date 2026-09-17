# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.flow_hop_cache import (
    CACHE_DTYPE,
    CFG_LANES,
    FlowHopCache,
    hop_layout,
)

CHUNK = 50


def _cache(frames: int) -> FlowHopCache:
    layers, heads, head_dim = 4, 2, 8
    return FlowHopCache(
        budget_bytes=frames * 2 * layers * heads * head_dim * CACHE_DTYPE.itemsize,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        chunk=CHUNK,
        device="cpu",
    )


def test_a_hop_inside_one_chunk_reads_to_that_chunks_end() -> None:
    layout = hop_layout([(100, 120)], CHUNK)

    assert layout.lanes == (0,)
    assert layout.cache_seqlens == (120,)
    assert layout.cu_seqlens_q == (0, 20)
    assert layout.max_seqlen_q == 20


def test_a_hop_crossing_a_chunk_boundary_is_cut_at_it() -> None:
    layout = hop_layout([(40, 110)], CHUNK)

    assert layout.lanes == (0, 0, 0)
    assert layout.cache_seqlens == (50, 100, 110)
    assert layout.cu_seqlens_q == (0, 10, 60, 70)
    assert layout.max_seqlen_q == 50


def test_a_first_hop_gets_one_segment_per_whole_chunk() -> None:
    layout = hop_layout([(0, 150)], CHUNK)

    assert layout.cache_seqlens == (50, 100, 150)
    assert layout.cu_seqlens_q == (0, 50, 100, 150)


def test_ragged_rows_accumulate_query_offsets_in_row_order() -> None:
    layout = hop_layout([(0, 50), (100, 130), (90, 100)], CHUNK)

    assert layout.lanes == (0, 1, 2)
    assert layout.cache_seqlens == (50, 130, 100)
    assert layout.cu_seqlens_q == (0, 50, 80, 90)


def test_a_span_that_adds_no_frames_is_rejected() -> None:
    with pytest.raises(ValueError, match="forward span"):
        hop_layout([(120, 120)], CHUNK)


def test_a_budget_below_one_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        FlowHopCache(
            budget_bytes=16,
            layers=4,
            heads=2,
            head_dim=8,
            chunk=CHUNK,
            device="cpu",
        )


def test_a_stream_holds_both_cfg_lanes_and_returns_them_on_release() -> None:
    cache = _cache(frames=400)
    free = cache.allocator.available_size()
    stream = cache.open_stream()

    cache.begin_call([(stream, 0, 60)])

    assert stream.frames == 60
    assert cache.allocator.available_size() == free - 60 * CFG_LANES
    assert [slots.numel() for slots in stream.slots] == [60] * CFG_LANES

    stream.release()

    assert stream.frames == 0
    assert cache.allocator.available_size() == free


def test_a_second_hop_continues_where_the_first_stopped() -> None:
    cache = _cache(frames=400)
    stream = cache.open_stream()
    cache.begin_call([(stream, 0, 50)])

    call = cache.begin_call([(stream, 50, 100)])

    assert stream.frames == 100
    assert call.page_table.shape == (len(call.cache_seqlens), 100)
    assert torch.equal(call.page_table[0, :50], stream.slots[0][:50].to(torch.int32))
    assert call.positions.tolist() == list(range(50, 100)) * CFG_LANES


def test_a_hop_that_skips_frames_is_rejected() -> None:
    cache = _cache(frames=400)
    stream = cache.open_stream()
    cache.begin_call([(stream, 0, 50)])

    with pytest.raises(RuntimeError, match="the hop starts at"):
        cache.begin_call([(stream, 60, 110)])


def test_the_pool_reports_the_slots_it_has_left() -> None:
    cache = _cache(frames=100)
    stream = cache.open_stream()

    assert cache.free_slots == 100

    cache.begin_call([(stream, 0, 30)])

    assert cache.free_slots == 40


def _scheduler(cache: FlowHopCache | None):
    from tests.unit_test.fun_cosyvoice3.test_streaming import _FakeFlow, _FakeHiFT

    from sglang_omni.models.fun_cosyvoice3 import stages
    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
        FunCosyVoice3StreamingVocoderScheduler,
    )

    flow = _FakeFlow()
    return FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(
            stages.FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator),
            _FakeHiFT(),
            flow_hop_cache=cache,
        )
    )


def _state(scheduler, request_id="r", *, token_offset: int, prompt_frames: int = 50):
    state = scheduler.create_stream_state(request_id)
    state.prompt_feat = torch.zeros(1, prompt_frames, 80)
    state.token_offset = token_offset
    return state


def test_a_first_hop_opens_a_stream_in_the_pool() -> None:
    cache = _cache(frames=400)
    scheduler = _scheduler(cache)
    state = _state(scheduler, token_offset=0)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([("r", state)], [])
    assert state.flow_cache is not None


def test_a_stream_that_already_ran_a_hop_uncached_never_starts_caching() -> None:
    cache = _cache(frames=400)
    scheduler = _scheduler(cache)
    state = _state(scheduler, token_offset=25)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])
    assert state.flow_cache is None


def test_a_stream_the_pool_cannot_grow_gives_its_slots_back() -> None:
    cache = _cache(frames=250)
    scheduler = _scheduler(cache)
    state = _state(scheduler, token_offset=0)
    scheduler.split_hop_participants([("r", state)])
    cache.begin_call([(state.flow_cache, 0, 100)])
    state.token_offset, state.hop_len = 25, 50

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])
    assert state.flow_cache is None
    assert cache.allocator.available_size() == cache.slots
    assert cache.fallback_hops == 1


def test_without_a_pool_every_row_recomputes_its_window() -> None:
    scheduler = _scheduler(None)
    state = _state(scheduler, token_offset=0)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])


def test_rows_of_one_step_share_the_pool_instead_of_each_taking_it_all() -> None:
    cache = _cache(frames=300)
    scheduler = _scheduler(cache)
    first = _state(scheduler, "a", token_offset=0)
    second = _state(scheduler, "b", token_offset=0)

    cached, plain = scheduler.split_hop_participants([("a", first), ("b", second)])

    assert cached == [("a", first)]
    assert plain == [("b", second)]
    assert second.flow_cache is None
