# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from bisect import bisect_left

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.flow_hop_cache import (
    CFG_LANES,
    CachedDiT,
    FlowHopCache,
    step_sizes,
)
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from tests.unit_test.fun_cosyvoice3.test_streaming import _FakeFlow, _FakeHiFT

CHUNK = 50
STEPS = 2
BLOCKS = 3
HEADS = 2
HEAD_DIM = 64
CONV_CONTEXT = 30
CONV_CHANNELS = 128
MAX_FRAMES = 600
DTYPE = torch.bfloat16
BYTES_PER_SLOT = 2 * STEPS * BLOCKS * HEADS * HEAD_DIM * DTYPE.itemsize
BYTES_PER_ROW = (
    STEPS * 2 * CONV_CONTEXT * CONV_CHANNELS * DTYPE.itemsize + MAX_FRAMES * 4
)


def _cache(slots: int, *, chunk_size: int = CHUNK, device: str = "cpu") -> FlowHopCache:
    return FlowHopCache(
        budget_bytes=slots * BYTES_PER_SLOT + slots // chunk_size * BYTES_PER_ROW,
        steps=STEPS,
        blocks=BLOCKS,
        heads=HEADS,
        head_dim=HEAD_DIM,
        conv_context=CONV_CONTEXT,
        conv_channels=CONV_CHANNELS,
        chunk_size=chunk_size,
        max_frames=MAX_FRAMES,
        dtype=DTYPE,
        device=device,
    )


def test_the_budget_pays_for_slots_rows_and_conv_tails() -> None:
    cache = _cache(400)

    assert cache.slots == 400
    assert cache.rows.size == 400 // CHUNK
    assert cache.conv_tails.shape == (
        STEPS,
        2,
        400 // CHUNK + 1,
        CONV_CONTEXT,
        CONV_CHANNELS,
    )


def test_a_budget_below_one_chunk_per_lane_is_rejected() -> None:
    with pytest.raises(ValueError, match="below one chunk"):
        _cache(CFG_LANES * CHUNK - 1)


def test_a_stream_holds_both_cfg_lanes_and_returns_them_on_release() -> None:
    cache = _cache(400)
    stream = cache.open_stream()

    assert cache.reserve(stream, 100)

    held = cache.rows.req_to_token[stream.lanes, :100]
    assert held.unique().numel() == 100 * CFG_LANES
    assert cache.allocator.available_size() == 400 - 100 * CFG_LANES

    cache.release(stream)

    assert cache.allocator.available_size() == 400
    assert cache.rows.available_size() == 400 // CHUNK


def test_a_reservation_the_pool_cannot_hold_changes_nothing() -> None:
    cache = _cache(400)
    stream = cache.open_stream()
    cache.reserve(stream, 150)

    assert not cache.reserve(stream, 250)

    assert stream.reserved == 150
    assert cache.allocator.available_size() == 400 - 150 * CFG_LANES


def test_a_new_stream_starts_from_the_conv_left_padding() -> None:
    cache = _cache(400)
    stream = cache.open_stream()
    cache.conv_tails[:, :, stream.lanes] = 1.0
    cache.release(stream)

    reopened = cache.open_stream()

    assert sorted(reopened.lanes) == sorted(stream.lanes)
    assert not cache.conv_tails[:, :, reopened.lanes].any()


def test_a_follow_up_hop_appends_each_chunk_after_the_frames_its_lane_holds() -> None:
    cache = _cache(800)
    first, second = cache.open_stream(), cache.open_stream()
    cache.reserve(first, 100)
    cache.reserve(second, 50)
    first.frames, second.frames = 100, 50
    cache.reserve(first, 200)
    cache.reserve(second, 100)

    hop = cache.begin_hop([first, second])

    table = cache.rows.req_to_token
    assert hop.lanes.tolist() == [
        first.lanes[0],
        second.lanes[0],
        first.lanes[1],
        second.lanes[1],
    ]
    assert hop.lengths.tolist() == [100, 50, 100, 50]
    assert hop.positions.tolist() == (list(range(100, 200)) + list(range(50, 100))) * 2
    assert hop.cache_seqlens.tolist() == [100, 150, 50, 100, 150, 50]
    assert hop.cu_seqlens_q.tolist() == [0, 50, 100, 150, 200, 250, 300]
    assert hop.max_seqlen_q == 50
    assert hop.max_end == 200
    assert torch.equal(hop.page_table[0], table[first.lanes[0], :200])
    assert torch.equal(hop.page_table[2], table[second.lanes[0], :200])
    assert torch.equal(hop.page_table[5], table[second.lanes[1], :200])


def _scheduler(
    cache: FlowHopCache | None, **scheduler_kwargs
) -> FunCosyVoice3StreamingVocoderScheduler:
    flow = _FakeFlow()
    return FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(
            stages.FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator),
            _FakeHiFT(),
            flow_hop_cache=cache,
        ),
        **scheduler_kwargs,
    )


def _state(scheduler, request_id: str = "r", *, token_offset: int = 0):
    state = scheduler.create_stream_state(request_id)
    state.prompt_feat = torch.zeros(1, 50, 80)
    state.token_offset = token_offset
    return state


def test_a_first_hop_reserves_the_prompt_and_the_hop() -> None:
    cache = _cache(400)
    scheduler = _scheduler(cache)
    state = _state(scheduler)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([("r", state)], [])
    assert state.flow_cache.reserved == 100
    assert state.flow_cache.frames == 0


def test_a_stream_that_ran_a_hop_uncached_never_starts_caching() -> None:
    cache = _cache(400)
    scheduler = _scheduler(cache)
    state = _state(scheduler, token_offset=25)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])
    assert state.flow_cache is None
    assert cache.allocator.available_size() == 400


def test_a_stream_the_pool_cannot_grow_gives_its_slots_back() -> None:
    cache = _cache(250)
    scheduler = _scheduler(cache)
    state = _state(scheduler)
    scheduler.split_hop_participants([("r", state)])
    state.token_offset, state.hop_len = 25, 50

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])
    assert state.flow_cache is None
    assert cache.allocator.available_size() == 250
    assert cache.rows.available_size() == 250 // CHUNK
    assert cache.fallback_hops == 1


def test_rows_of_one_step_share_the_pool() -> None:
    cache = _cache(300)
    scheduler = _scheduler(cache)
    first, second = _state(scheduler, "a"), _state(scheduler, "b")

    cached, plain = scheduler.split_hop_participants([("a", first), ("b", second)])

    assert cached == [("a", first)]
    assert plain == [("b", second)]
    assert second.flow_cache is None
    assert cache.rows.available_size() == 300 // CHUNK - CFG_LANES


def test_without_a_pool_every_row_recomputes_its_prefix() -> None:
    scheduler = _scheduler(None)
    state = _state(scheduler)

    cached, plain = scheduler.split_hop_participants([("r", state)])

    assert (cached, plain) == ([], [("r", state)])


def test_a_released_stream_state_returns_its_slots() -> None:
    cache = _cache(400)
    scheduler = _scheduler(cache)
    state = _state(scheduler)
    scheduler.split_hop_participants([("r", state)])

    scheduler.release_stream_resources("r", state)

    assert state.flow_cache is None
    assert cache.allocator.available_size() == 400


@pytest.mark.parametrize(
    "hops", [{"token_hop_len": 20}, {"token_hop_len": 25, "token_max_hop_len": 90}]
)
def test_hops_that_end_inside_a_chunk_are_rejected_with_the_cache(hops) -> None:
    with pytest.raises(ValueError, match="whole Flow chunks"):
        _scheduler(_cache(400), **hops)


def test_every_step_the_pool_holds_has_a_size_within_a_quarter_of_it() -> None:
    unit, slots = CFG_LANES * CHUNK, 6950
    sizes = step_sizes(unit, slots)

    assert sizes[-1] == slots // unit * unit
    assert all(size % unit == 0 for size in sizes)
    for total in range(unit, sizes[-1] + 1, unit):
        size = sizes[bisect_left(sizes, total)]
        assert total <= size < total * 1.25


TINY_CHUNK = 4
TINY_CHANNELS = 8


class _TinyFlow:
    def __init__(self, rows: int) -> None:
        from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

        cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")
        if not _is_fa3_supported():
            pytest.skip("FA3 is unavailable on this device")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.dit = (
            cosyvoice_dit.DiT(
                dim=CONV_CHANNELS,
                depth=BLOCKS,
                heads=HEADS,
                dim_head=HEAD_DIM,
                ff_mult=2,
                mel_dim=TINY_CHANNELS,
                mu_dim=TINY_CHANNELS,
                spk_dim=TINY_CHANNELS,
                out_channels=TINY_CHANNELS,
                static_chunk_size=TINY_CHUNK,
                num_decoding_left_chunks=-1,
                long_skip_connection=True,
            )
            .to(self.device)
            .eval()
        )
        with torch.no_grad():
            for parameter in self.dit.parameters():
                parameter.normal_(0, 0.1)
        as_input = {"device": self.device, "dtype": DTYPE}
        self.noise, self.mu, self.cond = (
            torch.randn(rows, 16, TINY_CHANNELS, **as_input) for _ in range(3)
        )
        self.spks = torch.randn(rows, TINY_CHANNELS, **as_input)
        self.time_span = torch.linspace(0, 1, STEPS + 1, **as_input)

    def cached_dit(self, *, capture_steps: bool) -> CachedDiT:
        return CachedDiT(
            self.dit,
            _cache(256, chunk_size=TINY_CHUNK, device="cuda"),
            device=self.device,
            capture_steps=capture_steps,
        )

    def solve(self, estimator, spans, rows_of) -> torch.Tensor:
        def take(part):
            return torch.cat(
                [part[row, start:end] for row, (start, end) in zip(rows_of, spans)]
            ).unsqueeze(0)

        return solve_flow_euler_packed(
            estimator,
            take(self.noise),
            self.time_span,
            take(self.mu),
            self.spks[rows_of],
            take(self.cond),
            pack_rows([end - start for start, end in spans], self.device),
            cfg_rate=0.7,
            streaming=True,
        )

    def cached_hop(self, estimator, streams, ends, rows_of) -> torch.Tensor:
        for stream, end in zip(streams, ends, strict=True):
            assert estimator.cache.reserve(stream, end)
        estimator.begin_hop(streams)
        spans = [(stream.frames, stream.reserved) for stream in streams]
        out = self.solve(estimator, spans, rows_of)
        for stream in streams:
            stream.frames = stream.reserved
        return out


def _snr_db(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(reference.norm() / (value - reference).norm())


@pytest.mark.accelerator
def test_cached_hops_match_the_hop_over_the_whole_prefix() -> None:
    flow = _TinyFlow(rows=2)
    cached_dit = flow.cached_dit(capture_steps=False)
    cache = cached_dit.cache

    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        packed_dit = PackedDiT(flow.dit, device=flow.device)
        whole = flow.solve(packed_dit, [(0, 16), (0, 16)], [0, 1])
        first, second = cache.open_stream(), cache.open_stream()
        first_hop = flow.cached_hop(cached_dit, [first], [8], [0])
        flow.cached_hop(cached_dit, [second], [12], [1])
        new = flow.cached_hop(cached_dit, [first, second], [16, 16], [0, 1])
        uncached_first_hop = flow.solve(packed_dit, [(0, 8)], [0])

    assert torch.equal(first_hop, uncached_first_hop)
    assert _snr_db(new[:, :8], whole[:, 8:16]) > 30
    assert _snr_db(new[:, 8:], whole[:, 28:]) > 30


@pytest.mark.accelerator
def test_a_hop_replayed_from_its_captured_step_matches_the_eager_hop() -> None:
    flow = _TinyFlow(rows=4)
    eager = flow.cached_dit(capture_steps=False)
    graphed = flow.cached_dit(capture_steps=True)
    hops: dict[str, list[torch.Tensor]] = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        for size in (*reversed(graphed.sizes), None):
            graphed.capture_size = size
            stream = graphed.cache.open_stream()
            flow.cached_hop(graphed, [stream], [TINY_CHUNK], [0])
            graphed.cache.release(stream)
        for name, estimator in (("eager", eager), ("graphed", graphed)):
            streams = [estimator.cache.open_stream() for _ in range(4)]
            hops[name] = [
                flow.cached_hop(estimator, streams[:1], [8], [0]),
                flow.cached_hop(estimator, streams[1:], [12, 12, 12], [1, 2, 3]),
                flow.cached_hop(estimator, streams, [16] * 4, [0, 1, 2, 3]),
            ]

    assert len(graphed.graphs.entries) == len(graphed.sizes)
    assert torch.equal(hops["graphed"][0], hops["eager"][0])
    for replayed, reference in zip(hops["graphed"][1:], hops["eager"][1:]):
        assert _snr_db(replayed, reference) > 30
