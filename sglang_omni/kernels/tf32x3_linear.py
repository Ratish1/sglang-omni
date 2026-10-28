# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
"""A float32 linear layer whose products run on tensor cores as three TF32 passes, accurate to
float32 rounding, for activations whose downstream choices float32 math decides."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_SIZE_M = 8
VECTOR_BYTES = 16
THREADS_PER_PROGRAM = 256
# note (ratish): measured on H100 over the reference encoder's GEMMs (64 to 512 rows by 512 or 2048): eight
# K splits fill the SMs that so few row tiles leave idle, and beat FFMA at every row count.
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
SPLITS = 8
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def tile_coordinates(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=["row_count"])
def tf32x3_linear_kernel(
    input_ptr,
    weight_ptr,
    partial_ptr,
    row_count,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    SPLIT_FEATURES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # note (ratish): tf32x3 splits each float32 operand into a TF32 high part and a TF32 low part and
    # sums three tensor-core products, which keeps the products' error at float32 rounding.
    num_pid_m = tl.cdiv(row_count, BLOCK_M)
    num_pid_n = tl.cdiv(OUT_FEATURES, BLOCK_N)
    pid_m, pid_n = tile_coordinates(
        tl.program_id(0), GROUP_SIZE_M * num_pid_n, num_pid_m, GROUP_SIZE_M
    )
    split = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    outputs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    safe_rows = tl.where(rows < row_count, rows, 0)
    safe_outputs = tl.where(outputs < OUT_FEATURES, outputs, 0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, SPLIT_FEATURES, BLOCK_K):
        columns = split * SPLIT_FEATURES + k_start + tl.arange(0, BLOCK_K)
        inputs = tl.load(
            input_ptr + safe_rows[:, None] * IN_FEATURES + columns[None, :]
        )
        weights = tl.load(
            weight_ptr + safe_outputs[None, :] * IN_FEATURES + columns[:, None]
        )
        accumulator = tl.dot(inputs, weights, accumulator, input_precision="tf32x3")
    tl.store(
        partial_ptr
        + (split * row_count + rows[:, None]) * OUT_FEATURES
        + outputs[None, :],
        accumulator,
        mask=(rows < row_count)[:, None] & (outputs < OUT_FEATURES)[None, :],
    )


@triton.jit(do_not_specialize=["element_count"])
def sum_splits_kernel(
    partial_ptr, output_ptr, element_count, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    # note (ratish): the splits add in their fixed order, so a row's output never depends on the run.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_element = offsets < element_count
    total = tl.load(partial_ptr + offsets, mask=is_element)
    for split in tl.static_range(1, SPLITS):
        total += tl.load(partial_ptr + split * element_count + offsets, mask=is_element)
    tl.store(output_ptr + offsets, total, mask=is_element)


def tf32x3_linear(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """hidden_states @ weight.T for a float32 (..., in_features) activation and a contiguous float32
    (out_features, in_features) weight, the in_features cut into splits summed in order.
    """
    out_features, in_features = weight.shape
    block_m, block_n, block_k, splits = BLOCK_M, BLOCK_N, BLOCK_K, SPLITS
    assert (
        hidden_states.dtype == torch.float32 and weight.dtype == torch.float32
    ), "float32 operands"
    assert weight.is_contiguous(), "a row-major weight"
    assert in_features % (splits * block_k) == 0, "every split holds whole K tiles"
    rows = hidden_states.reshape(-1, in_features).contiguous()
    row_count = rows.shape[0]
    output = rows.new_empty(row_count, out_features)
    partials = (
        output if splits == 1 else rows.new_empty(splits, row_count, out_features)
    )
    tiles = triton.cdiv(row_count, block_m) * triton.cdiv(out_features, block_n)
    tf32x3_linear_kernel[(tiles, splits)](
        rows,
        weight,
        partials,
        row_count,
        OUT_FEATURES=out_features,
        IN_FEATURES=in_features,
        SPLIT_FEATURES=in_features // splits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    if splits > 1:
        element_count = row_count * out_features
        block = THREADS_PER_PROGRAM * VECTOR_BYTES // output.element_size()
        sum_splits_kernel[(triton.cdiv(element_count, block),)](
            partials,
            output,
            element_count,
            SPLITS=splits,
            BLOCK=block,
            num_warps=THREADS_PER_PROGRAM // 32,
        )
    else:
        pass
    return output.view(*hidden_states.shape[:-1], out_features)


class Tf32x3Linear(torch.nn.Module):
    """A bias-free float32 nn.Linear whose products run as three TF32 passes on tensor cores."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        assert linear.bias is None, "a bias-free projection"
        self.weight = linear.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return tf32x3_linear(hidden_states, self.weight)
