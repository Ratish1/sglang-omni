# SPDX-License-Identifier: Apache-2.0
"""The code predictor's residual add and RMSNorm in one launch, bit for bit the plain path.

sglang's fused add-norm normalizes the unrounded fp32 sum; the predictor's reference path
rounds the sum to bf16 first (a torch add) and normalizes the rounded value (flashinfer's
RMSNorm). This kernel keeps that rounding point and reproduces flashinfer 0.6.18's CuTe
RMSNorm reduction: one warp per row, lane t holds columns 8t + 256b + j, a sequential
per-lane fold in (b, j) order, then a butterfly with offsets 1, 2, 4, 8, 16, an approximate
rsqrt, and x * rstd * w rounded once. The equality test pins it to the installed kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MATCHED_FLASHINFER_VERSION = "0.6.18"
HIDDEN_SIZE = 1024
LANES = 32
VEC = 8
VEC_BLOCKS = HIDDEN_SIZE // (LANES * VEC)


@triton.jit
def butterfly_sum(lane_partials):
    """Sum 32 lane partials pairing lane i with i ^ 1, then i ^ 2, up to i ^ 16."""
    partials = tl.sum(tl.reshape(lane_partials, [16, 2]), 1)
    partials = tl.sum(tl.reshape(partials, [8, 2]), 1)
    partials = tl.sum(tl.reshape(partials, [4, 2]), 1)
    partials = tl.sum(tl.reshape(partials, [2, 2]), 1)
    return tl.sum(partials, 0)


@triton.jit
def add_rmsnorm_rounded_kernel(
    X,
    RESIDUAL,
    WEIGHT,
    eps,
    HIDDEN: tl.constexpr,
    BLOCKS: tl.constexpr,
    LANE_VEC: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    j = tl.arange(0, LANE_VEC)
    row_base = row * HIDDEN
    acc = tl.zeros([32], dtype=tl.float32)
    for block in tl.static_range(BLOCKS):
        # One 16-byte vector per lane; the lane's eight squares are then added in
        # element order, each through a masked sum, which is exact.
        col = LANE_VEC * lane[:, None] + 32 * LANE_VEC * block + j[None, :]
        total = tl.load(X + row_base + col).to(tl.float32) + tl.load(
            RESIDUAL + row_base + col
        ).to(tl.float32)
        rounded = total.to(tl.bfloat16)
        tl.store(RESIDUAL + row_base + col, rounded)
        value = rounded.to(tl.float32)
        squares = value * value
        for element in tl.static_range(LANE_VEC):
            acc += tl.sum(tl.where(j[None, :] == element, squares, 0.0), 1)
    sum_sq = butterfly_sum(acc)
    rstd = libdevice.rsqrt(sum_sq / HIDDEN + eps)
    block = tl.arange(0, BLOCKS)
    j = tl.arange(0, LANE_VEC)
    col = (
        LANE_VEC * lane[:, None, None]
        + 32 * LANE_VEC * block[None, :, None]
        + j[None, None, :]
    )
    value = tl.load(RESIDUAL + row_base + col).to(tl.float32)
    weight = tl.load(WEIGHT + col).to(tl.float32)
    tl.store(X + row_base + col, (value * rstd * (weight + 0.0)).to(tl.bfloat16))


def supports_exact_add_rmsnorm(
    hidden_size: int, dtype: torch.dtype, device: torch.device
) -> bool:
    """The kernel reproduces flashinfer's tree for the predictor's shape only."""
    return (
        hidden_size == HIDDEN_SIZE and dtype == torch.bfloat16 and device.type == "cuda"
    )


def add_rmsnorm_rounded(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """In place: residual becomes bf16(x + residual), x becomes RMSNorm(residual) * weight.

    Returns (normed, residual) with sglang's residual norm contract.
    """
    rows = x.shape[0]
    add_rmsnorm_rounded_kernel[(rows,)](
        x,
        residual,
        weight,
        eps,
        HIDDEN=HIDDEN_SIZE,
        BLOCKS=VEC_BLOCKS,
        LANE_VEC=VEC,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return x, residual
