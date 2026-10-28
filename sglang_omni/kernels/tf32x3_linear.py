# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
"""A float32 linear layer whose products run on tensor cores as three TF32 passes, accurate to
float32 rounding, for activations whose downstream choices float32 math decides."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_SIZE_M = 8


@triton.jit
def tile_coordinates(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=["row_count", "programs"])
def tf32x3_linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    row_count,
    programs,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # note (ratish): tf32x3 splits each float32 operand into a TF32 high part and a TF32 low part and
    # sums three tensor-core products, which keeps the products' error at float32 rounding.
    num_pid_m = tl.cdiv(row_count, BLOCK_M)
    num_pid_n = tl.cdiv(OUT_FEATURES, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    columns_k = tl.arange(0, BLOCK_K)
    for tile_id in tl.range(
        tl.program_id(0), num_pid_m * num_pid_n, programs, flatten=True
    ):
        pid_m, pid_n = tile_coordinates(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        outputs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        safe_rows = tl.where(rows < row_count, rows, 0)
        safe_outputs = tl.where(outputs < OUT_FEATURES, outputs, 0)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, IN_FEATURES, BLOCK_K):
            columns = k_start + columns_k
            is_column = columns < IN_FEATURES
            inputs = tl.load(
                input_ptr + safe_rows[:, None] * IN_FEATURES + columns[None, :],
                mask=is_column[None, :],
                other=0.0,
            )
            weights = tl.load(
                weight_ptr + safe_outputs[None, :] * IN_FEATURES + columns[:, None],
                mask=is_column[:, None],
                other=0.0,
            )
            accumulator = tl.dot(inputs, weights, accumulator, input_precision="tf32x3")
        tl.store(
            output_ptr + rows[:, None] * OUT_FEATURES + outputs[None, :],
            accumulator,
            mask=(rows < row_count)[:, None] & (outputs < OUT_FEATURES)[None, :],
        )


def tf32x3_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """hidden_states @ weight.T for a float32 (..., in_features) activation and a contiguous
    float32 (out_features, in_features) weight, in a persistent grid of one program per SM.
    """
    out_features, in_features = weight.shape
    assert (
        hidden_states.dtype == torch.float32 and weight.dtype == torch.float32
    ), "float32 operands"
    assert weight.is_contiguous(), "a row-major weight"
    rows = hidden_states.reshape(-1, in_features).contiguous()
    output = rows.new_empty(rows.shape[0], out_features)
    tiles = triton.cdiv(rows.shape[0], block_m) * triton.cdiv(out_features, block_n)
    programs = min(
        tiles, torch.cuda.get_device_properties(rows.device).multi_processor_count
    )
    tf32x3_linear_kernel[(programs,)](
        rows,
        weight,
        output,
        rows.shape[0],
        programs,
        OUT_FEATURES=out_features,
        IN_FEATURES=in_features,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.view(*hidden_states.shape[:-1], out_features)
