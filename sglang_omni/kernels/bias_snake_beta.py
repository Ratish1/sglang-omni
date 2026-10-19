# SPDX-License-Identifier: Apache-2.0
"""SnakeBeta of a conv's output plus its bias, and a residual when given, written into the caller's buffer: the
activation of a channels-last conv stack fused with the adds in front of it, bit for bit the eager adds and
SnakeBeta in bf16."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang_omni.utils.snake_beta import FusedSnakeBeta

BIAS_BLOCK_POSITIONS = 64
BIAS_BLOCK_CHANNELS = 32


@triton.jit
def snake_beta_coefficients(alpha_raw, beta_raw, eps):
    """exp(alpha) and 1 / (exp(beta) + eps), each rounded to bf16 where eager rounds it."""
    a = libdevice.exp(alpha_raw).to(tl.bfloat16).to(tl.float32)
    b = libdevice.exp(beta_raw).to(tl.bfloat16).to(tl.float32)
    t = (b + eps).to(tl.bfloat16).to(tl.float32)
    r = libdevice.div_rn(1.0, t).to(tl.bfloat16).to(tl.float32)
    return a, r


@triton.jit
def snake_beta_values(xv, a, r):
    """x + r * sin(x * a) ** 2, each intermediate rounded to bf16 where eager rounds it."""
    s = (xv * a).to(tl.bfloat16).to(tl.float32)
    sn = libdevice.sin(s).to(tl.bfloat16).to(tl.float32)
    p = (sn * sn).to(tl.bfloat16).to(tl.float32)
    m = (r * p).to(tl.bfloat16).to(tl.float32)
    return (xv + m).to(tl.bfloat16)


@triton.jit(do_not_specialize=["x_batch_stride", "y_batch_stride", "C", "length"])
def bias_snake_beta_channels_last_kernel(
    x_ptr,
    x_batch_stride,
    bias_ptr,
    residual_ptr,
    sum_ptr,
    y_ptr,
    y_batch_stride,
    alpha_ptr,
    beta_ptr,
    C,
    length,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    # SnakeBeta of x + bias (+ residual) over [B, L, C]: one program covers BLOCK_POSITIONS
    # positions by BLOCK_CHANNELS channels of one batch row, so every tensor shares one
    # offset tile from its own row base; x and y rows may sit inside longer buffers. The
    # sums round to bf16 in the eager order, (x + bias) then + residual.
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    c = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    batch = tl.program_id(2).to(tl.int64)
    channel_mask = c < C
    a, r = snake_beta_coefficients(
        tl.load(alpha_ptr + c, mask=channel_mask, other=0).to(tl.float32),
        tl.load(beta_ptr + c, mask=channel_mask, other=0).to(tl.float32),
        eps,
    )
    bias = tl.load(bias_ptr + c, mask=channel_mask, other=0).to(tl.float32)

    mask = (positions < length)[:, None] & channel_mask[None, :]
    offsets = positions[:, None] * C + c[None, :]
    contiguous_base = batch * length * C
    xv = tl.load(x_ptr + batch * x_batch_stride + offsets, mask=mask, other=0).to(
        tl.float32
    )
    total = (xv + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + contiguous_base + offsets, mask=mask, other=0)
        total = (total + residual.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        pass
    if WRITE_SUM:
        tl.store(sum_ptr + contiguous_base + offsets, total.to(tl.bfloat16), mask=mask)
    else:
        pass
    tl.store(
        y_ptr + batch * y_batch_stride + offsets,
        snake_beta_values(total, a[None, :], r[None, :]),
        mask=mask,
    )


def fused_bias_snake_beta(
    conv_output: torch.Tensor,
    bias: torch.Tensor,
    activation: FusedSnakeBeta,
    out: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    sum_out: torch.Tensor | None = None,
) -> None:
    """activation(conv_output + bias, + residual if given) into out, the sum into sum_out if
    given, bit for bit the eager adds and SnakeBeta. [B, L, C] bf16 on CUDA; the rows of
    conv_output and out may sit in longer buffers, residual and sum_out are contiguous.
    """
    batch_size, length, channels = conv_output.shape
    grid = (
        triton.cdiv(length, BIAS_BLOCK_POSITIONS),
        triton.cdiv(channels, BIAS_BLOCK_CHANNELS),
        batch_size,
    )
    bias_snake_beta_channels_last_kernel[grid](
        conv_output,
        conv_output.stride(0),
        bias,
        conv_output if residual is None else residual,
        out if sum_out is None else sum_out,
        out,
        out.stride(0),
        activation.alpha,
        activation.beta,
        channels,
        length,
        activation.no_div_by_zero,
        HAS_RESIDUAL=residual is not None,
        WRITE_SUM=sum_out is not None,
        BLOCK_POSITIONS=BIAS_BLOCK_POSITIONS,
        BLOCK_CHANNELS=BIAS_BLOCK_CHANNELS,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
