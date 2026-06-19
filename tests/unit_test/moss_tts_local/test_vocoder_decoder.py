# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossTTSLocalAttention,
    MossTTSLocalProjectedTransformer,
    MossTTSLocalTransformerLayer,
    MossTTSLocalVocoderDecoder,
)


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
        self.calls = 0
        self.last_qkv_same_object = False
        self.last_input_lengths: torch.Tensor | None = None

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        return self.attention_implementation

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        self.last_qkv_same_object = query is key and key is value
        self.last_input_lengths = input_lengths
        return query


class _LengthOnlyAttention(_FakeAttention):
    def forward(
        self,
        query: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        self.last_qkv_same_object = False
        self.last_input_lengths = input_lengths
        return query


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


class _LengthOnlyLayer(_FakeLayer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.self_attn = _LengthOnlyAttention(hidden_size)


class _LengthOnlyTransformer(_FallbackTransformer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.layers = nn.ModuleList([_LengthOnlyLayer(hidden_size)])


class _LengthOnlyProjectedStage(_FallbackProjectedStage):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _LengthOnlyTransformer(6)


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


class _FakeSglangWorkspace:
    def __init__(
        self,
        *,
        max_batch_size: int,
        max_chunk_len: int,
        context: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_chunk_len = max_chunk_len
        self.context = context
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_pack = torch.empty(
            max_batch_size * max_chunk_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def create(cls, **kwargs: object) -> "_FakeSglangWorkspace":
        return cls(**kwargs)  # type: ignore[arg-type]


def _enable_sglang_flash_path(attn: MossTTSLocalAttention) -> None:
    attn._remote_has_flash_attn = True
    attn._remote_flash_attn_varlen_func = object()
    attn._workspace_cls = _FakeSglangWorkspace
    attn._supports_sglang_flash_attention = (  # type: ignore[method-assign]
        lambda _: True
    )


def test_projected_transformer_sdpa_path_delegates_to_source_stage() -> None:
    source = _FallbackProjectedStage()
    wrapper = MossTTSLocalProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape == tuple(x.shape)
    assert torch.equal(out, x + 10)
    assert torch.equal(out_lengths, lengths + 1)


def test_non_streaming_projected_transformer_delegates_even_if_flash_requested() -> (
    None
):
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    wrapper = MossTTSLocalProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._remote_has_flash_attn = False
    attn._remote_flash_attn_varlen_func = None
    attn._supports_sglang_flash_attention = (  # type: ignore[method-assign]
        lambda _: True
    )

    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape == tuple(x.shape)
    assert torch.equal(out, x + 10)
    assert torch.equal(out_lengths, lengths + 1)


def test_attention_uses_remote_flash_window_when_remote_flash_exists() -> None:
    source = _FakeAttention(hidden_size=6)
    source.attention_implementation = "flash_attention_2"
    wrapper = MossTTSLocalAttention(source)
    _enable_sglang_flash_path(wrapper)

    assert wrapper._flash_window_size() == (source.context, 0)


def test_attention_uses_sdpa_equivalent_window_without_remote_flash() -> None:
    source = _FakeAttention(hidden_size=6)
    source.attention_implementation = "flash_attention_2"
    wrapper = MossTTSLocalAttention(source)
    wrapper._remote_has_flash_attn = False
    wrapper._remote_flash_attn_varlen_func = None
    wrapper._supports_sglang_flash_attention = (  # type: ignore[method-assign]
        lambda _: True
    )

    assert wrapper._flash_window_size() == (source.context - 1, 0)


def test_attention_supports_length_only_source_signature() -> None:
    source = _LengthOnlyAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)
    x = torch.randn(2, 4, 6)
    lengths = torch.tensor([4, 3])

    out = wrapper(x, input_lengths=lengths)

    assert source.calls == 1
    assert not source.last_qkv_same_object
    assert source.last_input_lengths is lengths
    assert torch.equal(out, x)


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


def test_attention_streaming_flash_request_uses_local_workspace_and_sglang_varlen() -> (
    None
):
    source = _StreamingFlashAttention(hidden_size=6)
    wrapper = MossTTSLocalAttention(source)
    _enable_sglang_flash_path(wrapper)
    source._streaming_state.exec_mask = torch.tensor([True, False])
    calls = []
    flash_token = object()

    def fake_local_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        offset: torch.Tensor,
        exec_mask: torch.Tensor,
        workspace: _FakeSglangWorkspace,
        *,
        context: int,
        flash_attn_varlen_func: object,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append(
            {
                "q_shape": q.shape,
                "k_shape": k.shape,
                "v_shape": v.shape,
                "cached_k_shape": cached_k.shape,
                "cached_v_shape": cached_v.shape,
                "cached_pos_shape": cached_pos.shape,
                "offset": offset.clone(),
                "exec_mask": exec_mask.clone(),
                "workspace": workspace,
                "context": context,
                "flash_attn_varlen_func": flash_attn_varlen_func,
                "window_size": window_size,
            }
        )
        cached_pos[0].copy_(torch.tensor([0, 1, 2, 3]))
        offset.copy_(torch.where(exec_mask, offset + q.shape[2], offset))
        return q

    wrapper._sglang_flash_attn_varlen_func = flash_token
    wrapper._local_attention_func = fake_local_attention
    x = torch.randn(2, 4, 6)

    out = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert source.streaming_flash_calls == 0
    assert source.cache_updates == 0
    assert source.calls == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["q_shape"] == (2, source.num_heads, 4, source.head_dim)
    assert call["k_shape"] == (2, source.num_heads, 4, source.head_dim)
    assert call["v_shape"] == (2, source.num_heads, 4, source.head_dim)
    assert call["cached_k_shape"] == (2, source.num_heads, 4, source.head_dim)
    assert call["cached_v_shape"] == (2, source.num_heads, 4, source.head_dim)
    assert call["cached_pos_shape"] == (2, 4)
    assert call["offset"].tolist() == [0, 0]
    assert call["exec_mask"].tolist() == [True, False]
    assert isinstance(call["workspace"], _FakeSglangWorkspace)
    assert call["context"] == source.context
    assert call["flash_attn_varlen_func"] is flash_token
    assert call["window_size"] == (source.context, 0)
    assert source._streaming_state.offset.tolist() == [4, 0]
    assert source._streaming_state.cached_positions is not None
    assert source._streaming_state.cached_positions[0].tolist() == [0, 1, 2, 3]
    assert source._streaming_state.cached_positions[1].tolist() == [-1, -1, -1, -1]
    assert out.shape == x.shape
    assert wrapper._workspace is not None
    assert wrapper._workspace is call["workspace"]


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
    assert dict(wrapped.named_children())["source"] is decoder
    assert isinstance(wrapped[0], MossTTSLocalProjectedTransformer)
    assert wrapped[1] is patch_stage
