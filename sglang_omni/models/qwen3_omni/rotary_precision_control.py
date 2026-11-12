"""Run the measured interleaved rotary kernel without arithmetic contraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from sglang.kernels.ops.attention.rotary_triton import _triton_mrope_forward_fused
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding


def mrope_without_contraction(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine_sine_cache: torch.Tensor,
    positions: torch.Tensor,
    sections: list[int],
    head_features: int,
) -> None:
    query_head_count = query.shape[1] // head_features
    key_head_count = key.shape[1] // head_features
    _triton_mrope_forward_fused[(len(query),)](
        query,
        key,
        cosine_sine_cache,
        positions,
        query.stride(0),
        key.stride(0),
        positions.stride(0),
        query_head_count,
        key_head_count,
        head_features,
        head_features,
        triton.next_power_of_2(query_head_count),
        triton.next_power_of_2(key_head_count),
        triton.next_power_of_2(head_features),
        sections[0],
        sections[1],
        sections[2],
        True,
        False,
        True,
        None,
        enable_fp_fusion=False,
    )


@dataclass(kw_only=True)
class RotaryPrecisionControl:
    enabled: bool
    token_rows: int
    is_active: bool = False

    def install(self, rotary: MRotaryEmbedding) -> None:
        assert rotary.is_neox_style and rotary.mrope_interleaved
        assert not rotary.mrope_interleaved_glm
        assert rotary.head_size == rotary.rotary_dim
        assert rotary.mrope_section is not None
        original = rotary.forward_triton

        def forward_triton(
            positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.enabled and self.is_active and len(query) == self.token_rows:
                mrope_without_contraction(
                    query,
                    key,
                    rotary.cos_sin_cache,
                    positions,
                    rotary.mrope_section,
                    rotary.head_size,
                )
                return query, key
            else:
                return original(positions, query, key)

        rotary.forward_triton = forward_triton
