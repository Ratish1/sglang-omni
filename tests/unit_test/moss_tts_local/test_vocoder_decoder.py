# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossTTSLocalPatchTransform,
    MossTTSLocalProjectedTransformer,
    MossTTSLocalVocoderDecoder,
)


class _FakeLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = 2
        self.head_dim = hidden_size // self.num_heads
        self.causal = True
        self.context = 4
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        return "sdpa"

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x


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


def test_patch_transform_decode_matches_moss_layout() -> None:
    stage = MossTTSLocalPatchTransform(_PatchStage(patch_size=2, is_downsample=False))
    x = torch.arange(1 * 4 * 3).reshape(1, 4, 3)
    lengths = torch.tensor([3])

    out, out_lengths = stage(x, lengths)

    expected = torch.tensor([[[0, 3, 1, 4, 2, 5], [6, 9, 7, 10, 8, 11]]])
    assert torch.equal(out, expected)
    assert torch.equal(out_lengths, torch.tensor([6]))


def test_projected_transformer_fallback_receives_original_layout() -> None:
    source = _FallbackProjectedStage()
    wrapper = MossTTSLocalProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape == (2, 3, 4)
    assert torch.equal(out, x + 10)
    assert torch.equal(out_lengths, lengths + 1)


def test_vocoder_decoder_wraps_supported_stage_types() -> None:
    decoder = nn.ModuleList(
        [_FallbackProjectedStage(), _PatchStage(patch_size=2, is_downsample=False)]
    )
    wrapped = MossTTSLocalVocoderDecoder(decoder)

    assert len(wrapped) == 2
    assert isinstance(wrapped[0], MossTTSLocalProjectedTransformer)
    assert isinstance(wrapped[1], MossTTSLocalPatchTransform)
