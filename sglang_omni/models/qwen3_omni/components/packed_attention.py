# SPDX-License-Identifier: Apache-2.0
"""The packed varlen attention Qwen3-Omni's audio and vision encoders run on a CUDA device."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

HOPPER_CAPABILITY_MAJOR = 9


class PackedAttentionMetadata(Protocol):
    """Every sequence's bounds and the longest one, built once per encoder call for all its layers."""

    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen: int


def packed_attention_backend(capability: tuple[int, int], *, is_hip: bool) -> str:
    """FA3 on NVIDIA Hopper, Triton on every other supported device.

    sglang's VisionAttention rule also picks FA4 on Blackwell. Omni validates
    Hopper only, and Triton is the arm sglang itself uses on the other parts.
    """
    if is_hip:
        # Note (zijiecode): HIP reports gfx950 as (9, 5), which the Hopper check
        # would send to FA3; FA3 does not exist on HIP.
        return "triton_attn"
    else:
        pass
    major, _ = capability
    return "fa3" if major == HOPPER_CAPABILITY_MAJOR else "triton_attn"


def resolve_packed_attention(
    device: torch.device, dtype: torch.dtype
) -> nn.Module | None:
    """The packed attention for an encoder on this device in this dtype, or None to keep Transformers' attention."""
    # note (ratish): FA3 rejects float32 and Triton's dot runs float32 as TF32, so a float32 encoder keeps float32 attention.
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return None
    else:
        pass
    # Local import: only a process that builds an encoder on a CUDA device loads these kernels.
    from sglang.srt.layers.attention import vision

    backend = packed_attention_backend(
        torch.cuda.get_device_capability(device),
        is_hip=torch.version.hip is not None,
    )
    if backend == "fa3":
        impl_class = vision.VisionFlash3Attention
    else:
        impl_class = vision.VisionTritonAttention
    # The encoder process holds no sglang parallel state to read.
    return impl_class(use_data_parallel=True)


def prepare_packed_attention_metadata(
    cu_seqlens: torch.Tensor, max_seqlen: int
) -> PackedAttentionMetadata:
    """The packed attention's metadata for one encoder call, with the longest sequence from the host."""
    from sglang.srt.layers.attention import vision

    # note (ratish): given metadata, the kernels ignore SGLANG_VIT_ENABLE_CUDA_GRAPH, which changes how they read cu_seqlens.
    return vision.prepare_vision_attention_metadata(
        cu_seqlens, cu_seqlens.device, max_seqlen=max_seqlen
    )
