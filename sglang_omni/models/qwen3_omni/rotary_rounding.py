"""Control rotary contraction without changing native in-place storage."""

from __future__ import annotations

import torch
import triton
from sglang.kernels.ops.attention.rotary_triton import _triton_mrope_forward_fused
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding


def forward_rotary_without_contraction(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    rotary: MRotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary._match_cos_sin_cache_dtype(
        query
    )  # noqa: leading-underscore -- external rotary cache contract
    query_heads = query.shape[1] // rotary.head_size
    key_heads = key.shape[1] // rotary.head_size
    _triton_mrope_forward_fused[(query.shape[0],)](
        query,
        key,
        rotary.cos_sin_cache,
        positions,
        query.stride(0),
        key.stride(0),
        positions.stride(0),
        query_heads,
        key_heads,
        rotary.head_size,
        rotary.rotary_dim,
        triton.next_power_of_2(query_heads),
        triton.next_power_of_2(key_heads),
        triton.next_power_of_2(rotary.head_size),
        *rotary.mrope_section,
        rotary.mrope_interleaved,
        rotary.mrope_interleaved_glm,
        rotary.is_neox_style,
        rotary.axis_map if rotary.mrope_interleaved_glm else None,
        enable_fp_fusion=False,
    )
    return query, key
