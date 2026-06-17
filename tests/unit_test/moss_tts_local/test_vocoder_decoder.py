# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

import sglang_omni.models.moss_tts_local.vocoder_decoder as vocoder_decoder
from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossTTSLocalAttention,
    MossTTSLocalProjectedTransformer,
    MossTTSLocalTransformerLayer,
    MossTTSLocalVocoderDecoder,
    profile_moss_tts_local_vocoder_attention,
)
from sglang_omni.profiler.event_recorder import get_recorder


class _FakeLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = 2
        self.head_dim = hidden_size // self.num_heads
        self.causal = True
        self.context = 4
        self.attention_implementation = "sdpa"
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        return "sdpa"

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x


class _FakeStreamingState:
    def __init__(self, batch_size: int = 2) -> None:
        self.offset = torch.zeros(batch_size, dtype=torch.long)
        self.exec_mask = torch.ones(batch_size, dtype=torch.bool)
        self.cached_keys: torch.Tensor | None = None
        self.cached_values: torch.Tensor | None = None
        self.cached_positions: torch.Tensor | None = None


class _StreamingAttention(_FakeAttention):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self._streaming_state = _FakeStreamingState()
        self.streaming_sdpa_calls = 0

    def _forward_streaming_sdpa(
        self,
        x: torch.Tensor,
        state: _FakeStreamingState,
    ) -> torch.Tensor:
        assert state is self._streaming_state
        self.streaming_sdpa_calls += 1
        return x + 2


class _StreamingFlashAttention(_FakeAttention):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self._streaming_state = _FakeStreamingState()
        self.cache_updates = 0
        self.streaming_flash_calls = 0

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        assert is_streaming
        return "flash_attention_2"

    def _forward_streaming_flash(
        self,
        _: torch.Tensor,
        __: _FakeStreamingState,
    ) -> torch.Tensor:
        self.streaming_flash_calls += 1
        raise AssertionError("wrapper should own streaming flash attention")

    def _ensure_streaming_cache(
        self,
        state: _FakeStreamingState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert state is self._streaming_state
        cache_shape = (batch_size, self.num_heads, self.context, self.head_dim)
        if state.cached_keys is None:
            state.cached_keys = torch.zeros(cache_shape, device=device, dtype=dtype)
            state.cached_values = torch.zeros_like(state.cached_keys)
            state.cached_positions = torch.full(
                (batch_size, self.context),
                -1,
                device=device,
                dtype=torch.long,
            )
        assert state.cached_values is not None
        assert state.cached_positions is not None
        return state.cached_keys, state.cached_values, state.cached_positions

    def _build_streaming_kv(
        self,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        k_cur: torch.Tensor,
        v_cur: torch.Tensor,
        pos_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.cat([cached_k, k_cur], dim=2),
            torch.cat([cached_v, v_cur], dim=2),
            torch.cat([cached_pos, pos_q], dim=1),
        )

    def _update_streaming_cache(
        self,
        state: _FakeStreamingState,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        pos_k: torch.Tensor,
    ) -> None:
        self.cache_updates += 1
        raise AssertionError("wrapper should own streaming cache update")


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.layer_scale_1 = _FakeLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.layer_scale_2 = _FakeLayerScale(hidden_size)


class _FallbackTransformer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_size)])
        self.positional_embedding = "rope"
        self.positional_scale = 1.0
        self.max_period = 10000.0

    def resolve_attention_implementation(self, _: torch.Tensor) -> str:
        return "sdpa"


class _FallbackProjectedStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, 6)
        self.transformer = _FallbackTransformer(6)
        self.output_proj = nn.Linear(6, 7)
        self.is_streaming = False
        self.seen_input_shape: tuple[int, ...] | None = None

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_input_shape = tuple(x.shape)
        return x + 10, input_lengths + 1


class _PatchStage(nn.Module):
    def __init__(self, *, patch_size: int = 2, is_downsample: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = patch_size
        self.is_downsample = is_downsample
        self.module_type = "PatchedPretransform"

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, input_lengths


class _CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayerScale(_FakeLayerScale):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayer(_FakeLayer):
    def __init__(self, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.norm1 = _CountingLayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.layer_scale_1 = _CountingLayerScale(hidden_size)
        self.norm2 = _CountingLayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            _CountingLinear(hidden_size, hidden_size * 2),
            nn.GELU(),
            _CountingLinear(hidden_size * 2, hidden_size),
        )
        self.layer_scale_2 = _CountingLayerScale(hidden_size)


def test_projected_transformer_sdpa_path_does_not_reenter_source_stage() -> None:
    source = _FallbackProjectedStage()
    wrapper = MossTTSLocalProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_projected_transformer_uses_sglang_flash_fallback() -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    wrapper = MossTTSLocalProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._attention_kernel = "sglang"
    attn._supports_sglang_flash_attention = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k, window_size))
        return q

    attn._sglang_flash_attn_varlen_func = fake_flash_attn
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape is None
    assert len(calls) == 1
    cu_q, cu_k, max_q, max_k, window_size = calls[0]
    assert cu_q.tolist() == [0, 4, 7]
    assert cu_k.tolist() == [0, 4, 7]
    assert max_q == 4
    assert max_k == 4
    assert window_size == (source.transformer.layers[0].self_attn.context, 0)
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_projected_transformer_uses_single_unpadded_pack_fast_path(
    monkeypatch,
) -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    wrapper = MossTTSLocalProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._attention_kernel = "sglang"
    attn._supports_sglang_flash_attention = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fail_masked_pack(_: torch.Tensor, __: torch.Tensor) -> None:
        raise AssertionError("single unpadded input should not use masked pack")

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((q.shape, cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q

    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", fail_masked_pack)
    attn._sglang_flash_attn_varlen_func = fake_flash_attn
    x = torch.randn(1, 3, 4)
    lengths = torch.tensor([4])

    out, out_lengths = wrapper(x, lengths)

    assert len(calls) == 1
    q_shape, cu_q, cu_k, max_q, max_k = calls[0]
    assert q_shape[0] == 4
    assert cu_q.tolist() == [0, 4]
    assert cu_k.tolist() == [0, 4]
    assert max_q == 4
    assert max_k == 4
    assert out.shape == (1, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_cached_packed_rope_matches_moss_interleaved_reference() -> None:
    q = torch.randn(5, 2, 6)
    k = torch.randn(5, 2, 6)
    position_ids = torch.tensor([0, 1, 2, 3, 4])
    max_period = 10000.0
    cache = vocoder_decoder._MossPackedRopeCache(max_period=max_period)

    out_q, out_k = vocoder_decoder._apply_cached_packed_rope(
        q,
        k,
        position_ids,
        max_positions=5,
        cache=cache,
    )

    half_dim = q.shape[-1] // 2
    ds = torch.arange(half_dim, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / q.shape[-1]))
    phase = position_ids.float().view(-1, 1, 1) * freqs.view(1, 1, -1)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    q_pair = q.view(*q.shape[:-1], half_dim, 2)
    k_pair = k.view(*k.shape[:-1], half_dim, 2)
    qr, qi = q_pair[..., 0].float(), q_pair[..., 1].float()
    kr, ki = k_pair[..., 0].float(), k_pair[..., 1].float()
    ref_q = torch.stack(
        [
            (qr * cos - qi * sin).to(q.dtype),
            (qr * sin + qi * cos).to(q.dtype),
        ],
        dim=-1,
    ).view_as(q)
    ref_k = torch.stack(
        [
            (kr * cos - ki * sin).to(k.dtype),
            (kr * sin + ki * cos).to(k.dtype),
        ],
        dim=-1,
    ).view_as(k)

    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)
    assert cache._cos is not None
    cos_ptr = cache._cos.data_ptr()
    _ = cache.get(device=q.device, head_dim=q.shape[-1], max_positions=3)
    assert cache._cos.data_ptr() == cos_ptr


def test_projected_transformer_streaming_path_does_not_reenter_source_stage() -> None:
    source = _FallbackProjectedStage()
    source.is_streaming = True
    wrapper = MossTTSLocalProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_attention_uses_source_streaming_state_when_active() -> None:
    source = _StreamingAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)
    x = torch.randn(2, 4, 6)

    out = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert source.streaming_sdpa_calls == 1
    assert torch.allclose(out, source.out_proj(x + 2))


def test_attention_owns_streaming_flash_path() -> None:
    source = _StreamingFlashAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)
    calls: list[tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, int]]] = []

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        assert causal
        calls.append((cu_q, cu_k, max_q, max_k, window_size))
        assert k.shape == v.shape
        return q

    wrapper._attention_kernel = "sglang"
    wrapper._sglang_flash_attn_varlen_func = fake_flash_attn
    x = torch.randn(2, 4, 6)

    out = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert source.streaming_flash_calls == 0
    assert source.cache_updates == 0
    assert source._streaming_state.offset.tolist() == [4, 4]
    assert source._streaming_state.cached_positions is not None
    assert source._streaming_state.cached_positions.tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    assert len(calls) == 1
    cu_q, cu_k, max_q, max_k, window_size = calls[0]
    assert cu_q.tolist() == [0, 4, 8]
    assert cu_k.tolist() == [0, 4, 8]
    assert max_q == 4
    assert max_k == 4
    assert window_size == (source.context, 0)
    assert out.shape == x.shape


def test_attention_streaming_flash_cache_update_preserves_storage() -> None:
    source = _StreamingFlashAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        return q

    wrapper._attention_kernel = "sglang"
    wrapper._sglang_flash_attn_varlen_func = fake_flash_attn
    x = torch.randn(2, 4, 6)

    _ = wrapper(x, input_lengths=torch.tensor([4, 4]))
    state = source._streaming_state
    assert state.cached_keys is not None
    assert state.cached_values is not None
    assert state.cached_positions is not None
    keys_ptr = state.cached_keys.data_ptr()
    values_ptr = state.cached_values.data_ptr()
    positions_ptr = state.cached_positions.data_ptr()

    state.exec_mask = torch.tensor([True, False])
    _ = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert state.cached_keys is not None
    assert state.cached_values is not None
    assert state.cached_positions is not None
    assert state.cached_keys.data_ptr() == keys_ptr
    assert state.cached_values.data_ptr() == values_ptr
    assert state.cached_positions.data_ptr() == positions_ptr
    assert state.offset.tolist() == [8, 4]
    assert state.cached_positions.tolist() == [
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ]


def test_attention_profile_context_attributes_every_request_in_batch(
    tmp_path: Path,
) -> None:
    source = _StreamingFlashAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        return q

    wrapper._attention_kernel = "sglang"
    wrapper._sglang_flash_attn_varlen_func = fake_flash_attn
    x = torch.randn(2, 4, 6)
    recorder = get_recorder()
    path = recorder.start(run_id="run", event_dir=str(tmp_path), stage="vocoder")
    try:
        with profile_moss_tts_local_vocoder_attention(
            ["req-a", "req-b"],
            {"active_vocoder_backend": "owned_pytorch", "batch_size": 2},
        ):
            _ = wrapper(x, input_lengths=torch.tensor([4, 4]))
    finally:
        recorder.stop()

    with Path(path).open("r", encoding="utf-8") as fp:
        events = [json.loads(line) for line in fp if line.strip()]
    flash_events = [
        event
        for event in events
        if event["event_name"] == "moss_vocoder_attn_flash_sglang_start"
    ]
    assert {event["request_id"] for event in flash_events} == {"req-a", "req-b"}
    assert all(
        event["metadata"]["active_vocoder_backend"] == "owned_pytorch"
        for event in flash_events
    )


def test_attention_kernel_defaults_to_remote(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_MOSS_LOCAL_VOCODER_ATTENTION_KERNEL", raising=False)
    source = _FakeAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)

    assert wrapper._attention_kernel == "remote"


def test_transformer_layer_uses_source_modules_for_primitive_ops() -> None:
    source = _CountingLayer(hidden_size=6)
    wrapper = MossTTSLocalTransformerLayer(source)
    x = torch.randn(2, 4, 6)

    _ = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert source.norm1.calls == 1
    assert source.norm2.calls == 1
    assert source.layer_scale_1.calls == 1
    assert source.layer_scale_2.calls == 1
    assert source.ffn[0].calls == 1
    assert source.ffn[2].calls == 1


def test_vocoder_decoder_wraps_supported_stage_types() -> None:
    patch_stage = _PatchStage(patch_size=2, is_downsample=False)
    decoder = nn.ModuleList([_FallbackProjectedStage(), patch_stage])
    wrapped = MossTTSLocalVocoderDecoder(decoder)

    assert len(wrapped) == 2
    assert isinstance(wrapped[0], MossTTSLocalProjectedTransformer)
    assert wrapped[1] is patch_stage
