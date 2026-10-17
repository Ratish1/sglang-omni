# SPDX-License-Identifier: Apache-2.0
"""HiFT's Snake activation as a Triton kernel that rounds each step as CosyVoice's eager
ops do."""

from __future__ import annotations

import torch
from sglang.srt.utils.custom_op import register_custom_op

from sglang_omni.models.fun_cosyvoice3.causal_conv import HAS_TRITON

SAMPLES_PER_PROGRAM = 1024

if HAS_TRITON:
    import triton
    import triton.language as tl
    from sglang.kernels.ops.diffusion.common.numerics import mul_rn_f32
    from triton.language.extra import libdevice

    # note (ratish): sample counts change every step, one binary serves them all.
    @triton.jit(do_not_specialize=["samples", "channels"])
    def snake_kernel(
        x_ptr,
        alpha_ptr,
        inverse_alpha_ptr,
        out_ptr,
        samples,
        channels,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        is_sample = offsets < samples
        alpha = tl.load(alpha_ptr + row % channels)
        inverse_alpha = tl.load(inverse_alpha_ptr + row % channels)
        x = tl.load(x_ptr + row * samples + offsets, mask=is_sample)
        sine = libdevice.sin(mul_rn_f32(x, alpha))
        # note (ratish): each product rounds alone, as in the eager ops; an FMA would not.
        out = x + mul_rn_f32(inverse_alpha, mul_rn_f32(sine, sine))
        tl.store(out_ptr + row * samples + offsets, out, mask=is_sample)

else:
    pass


def fake_snake(
    x: torch.Tensor, alpha: torch.Tensor, inverse_alpha: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="fun_cosyvoice3_snake",
    mutates_args=[],
    fake_impl=fake_snake,
)
def snake(
    x: torch.Tensor, alpha: torch.Tensor, inverse_alpha: torch.Tensor
) -> torch.Tensor:
    """x: (batch, channels, samples) float32; alpha and inverse_alpha: (channels,), the
    latter 1 / (alpha + 1e-9) as the module computes it."""
    batch, channels, samples = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    out = torch.empty_like(x)
    grid = (triton.cdiv(samples, SAMPLES_PER_PROGRAM), batch * channels)
    snake_kernel[grid](
        x,
        alpha,
        inverse_alpha,
        out,
        samples,
        channels,
        BLOCK_SAMPLES=SAMPLES_PER_PROGRAM,
    )
    return out
