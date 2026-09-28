# SPDX-License-Identifier: Apache-2.0
"""The code predictor's decoder layer in three Triton launches at decode batch sizes.

Each launch streams one projection's bf16 weight once and folds the plain path's
elementwise work around it: the input norm into the qkv and gate_up launches; the q and
k norms, the rope and the K and V store into the qkv epilogue; the residual add into the
o_proj and down_proj epilogues; the activation into the gate_up epilogue. Every
projection output, the residual sum, the normed and the rotated q and k, and the
activation are rounded to bf16 where the plain path rounds them, and the elementwise
arithmetic is fp32 as in the plain path's kernels. Two things differ from the plain
path: the projections accumulate in fp32 in the tensor cores over BLOCK_K-wide K blocks,
in their own order against cuBLAS's; and the input norm's row scale is applied to the
accumulated product in fp32, with the norm weight applied to the input before its bf16
rounding, where the plain path rounds the fully normed input once. Both are single
roundings of the same products, so the distance to fp32 is the plain path's. A row's
result depends on its own values only, so it does not change with the batch.

The launches are latency bound: a program streams its weight tiles through a short
pipeline, so the time is the number of K blocks per program times the memory latency
over the pipeline depth. The qkv and the residual launches therefore split K across
programs, with the last program of an output tile summing the fp32 partials in index
order, which keeps the result deterministic; the norm's sum of squares is the diagonal
of the input tile against itself on the tensor cores in the same loop, so no pass
precedes the pipeline. The input is read head by head, a head being the K span whose
columns are contiguous, so the loads stay vectorized whatever the layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from triton.language.extra import libdevice

# The accumulation group of every projection but gate_up; it must divide the head so
# an o_proj K block never crosses a head of the attention output.
BLOCK_K = 128
# One MMA n-tile; the head launch uses the head as its tile.
BLOCK_N = 32
# The gate_up launch has no split, so its wider K block keeps the per-block cost of the
# norm scale small; measured on the H100 and the H200, the same choice on both.
MLP_BLOCK_K = 256
MIN_BLOCK_M = 16
NUM_WARPS = 4
NUM_STAGES = 3


@dataclass(frozen=True)
class PredictorLayerShape:
    """The predictor layer's dimensions and the K split of its residual launches."""

    hidden_size: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    intermediate_size: int
    split_qkv: int
    split_hidden: int


@triton.jit
def row_offsets(stride_row, stride_t, T: tl.constexpr, BLOCK_M: tl.constexpr):
    """Offsets of the rows laid out (row group, token) major."""
    rm = tl.arange(0, BLOCK_M)
    return rm // T * stride_row + rm % T * stride_t


@triton.jit
def diagonal(square):
    """The diagonal of a [BLOCK_M, BLOCK_M] tile as a [BLOCK_M] vector."""
    rm = tl.arange(0, square.shape[0])
    return tl.sum(tl.where(rm[:, None] == rm[None, :], square, 0.0), 1)


@triton.jit
def scaled_tile(x, NORM_W, k0, BLOCK_K: tl.constexpr):
    """The input tile times the norm weight, rounded to bf16 for the MMA."""
    rk = tl.arange(0, BLOCK_K)
    w = tl.load(NORM_W + k0 + rk).to(tl.float32)
    return (x.to(tl.float32) * w[None, :]).to(tl.bfloat16)


@triton.jit
def weight_tile(W, cols, k0, K: tl.constexpr, BLOCK_K: tl.constexpr):
    rk = tl.arange(0, BLOCK_K)
    return tl.load(W + cols[:, None] * K + (k0 + rk)[None, :])


@triton.jit
def norm_qkv_rope_store_kernel(
    X,
    x_stride_row,
    x_stride_t,
    rows,
    NORM_W,
    norm_eps,
    W,
    Q_OUT,
    QN_W,
    KN_W,
    qk_eps,
    COS_SIN,
    POS,
    K_CACHE,
    V_CACHE,
    cache_stride_row,
    cache_stride_head,
    PARTIALS,
    SUM_SQ_PARTIALS,
    COUNTERS,
    T: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per head and K split: the weighted input against the head's qkv
    columns over the split's K range, with the rows' sum of squares alongside; the
    last program of the head sums the partials, applies the norm scale, then norms
    and rotates q and k; q to Q_OUT, k and v to the row's slot."""
    head = tl.program_id(0)
    pid_s = tl.program_id(1)
    HALF: tl.constexpr = HEAD_DIM // 2
    N_TOTAL: tl.constexpr = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    BLOCKS_PER_SPLIT: tl.constexpr = K // BLOCK_K // SPLIT
    rm = tl.arange(0, BLOCK_M)
    rh = tl.arange(0, HALF)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    lo_cols = head * HEAD_DIM + rh
    hi_cols = lo_cols + HALF
    square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    for block in range(pid_s * BLOCKS_PER_SPLIT, (pid_s + 1) * BLOCKS_PER_SPLIT):
        k0 = block * BLOCK_K
        x = tl.load(
            X + rows_off[:, None] + (k0 + rk)[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        square = tl.dot(x, tl.trans(x), square)
        scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
        acc_lo = tl.dot(
            scaled, tl.trans(weight_tile(W, lo_cols, k0, K, BLOCK_K)), acc_lo
        )
        acc_hi = tl.dot(
            scaled, tl.trans(weight_tile(W, hi_cols, k0, K, BLOCK_K)), acc_hi
        )
    sum_sq = diagonal(square)
    if SPLIT > 1:
        partial_rows = (pid_s * BLOCK_M + rm)[:, None] * N_TOTAL
        tl.store(PARTIALS + partial_rows + lo_cols[None, :], acc_lo)
        tl.store(PARTIALS + partial_rows + hi_cols[None, :], acc_hi)
        tl.store(SUM_SQ_PARTIALS + pid_s * BLOCK_M + rm, sum_sq)
        tl.debug_barrier()
        arrived = tl.atomic_add(COUNTERS + head, 1, sem="acq_rel", scope="gpu")
        is_last = arrived == SPLIT - 1
    else:
        is_last = pid_s == 0
    if is_last:
        if SPLIT > 1:
            acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
            acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
            sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
            for split in tl.static_range(SPLIT):
                partial_rows = (split * BLOCK_M + rm)[:, None] * N_TOTAL
                acc_lo += tl.load(
                    PARTIALS + partial_rows + lo_cols[None, :], cache_modifier=".cg"
                )
                acc_hi += tl.load(
                    PARTIALS + partial_rows + hi_cols[None, :], cache_modifier=".cg"
                )
                sum_sq += tl.load(
                    SUM_SQ_PARTIALS + split * BLOCK_M + rm, cache_modifier=".cg"
                )
            tl.atomic_xchg(COUNTERS + head, 0)
        else:
            pass
        rstd = libdevice.rsqrt(sum_sq / K + norm_eps)
        lo = (acc_lo * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        hi = (acc_hi * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + rm, mask=row_mask, other=0)
        cache_rows = rm // T * cache_stride_row + pos * HEAD_DIM
        if head < NUM_Q_HEADS + NUM_KV_HEADS:
            if head < NUM_Q_HEADS:
                w_lo = tl.load(QN_W + rh).to(tl.float32)
                w_hi = tl.load(QN_W + HALF + rh).to(tl.float32)
            else:
                w_lo = tl.load(KN_W + rh).to(tl.float32)
                w_hi = tl.load(KN_W + HALF + rh).to(tl.float32)
            head_sum_sq = tl.sum(lo * lo, 1) + tl.sum(hi * hi, 1)
            head_rstd = libdevice.rsqrt(head_sum_sq / HEAD_DIM + qk_eps)
            lo = (lo * head_rstd[:, None] * w_lo[None, :]).to(tl.bfloat16)
            hi = (hi * head_rstd[:, None] * w_hi[None, :]).to(tl.bfloat16)
            lo = lo.to(tl.float32)
            hi = hi.to(tl.float32)
            cos = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + rh[None, :])
            sin = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + HALF + rh[None, :])
            out_lo = (lo * cos - hi * sin).to(tl.bfloat16)
            out_hi = (lo * sin + hi * cos).to(tl.bfloat16)
            if head < NUM_Q_HEADS:
                q_rows = rm * (NUM_Q_HEADS * HEAD_DIM)
                tl.store(
                    Q_OUT + q_rows[:, None] + lo_cols[None, :],
                    out_lo,
                    mask=row_mask[:, None],
                )
                tl.store(
                    Q_OUT + q_rows[:, None] + hi_cols[None, :],
                    out_hi,
                    mask=row_mask[:, None],
                )
            else:
                slots = cache_rows + (head - NUM_Q_HEADS) * cache_stride_head
                tl.store(
                    K_CACHE + slots[:, None] + rh[None, :],
                    out_lo,
                    mask=row_mask[:, None],
                )
                tl.store(
                    K_CACHE + slots[:, None] + HALF + rh[None, :],
                    out_hi,
                    mask=row_mask[:, None],
                )
        else:
            v_head = head - NUM_Q_HEADS - NUM_KV_HEADS
            slots = cache_rows + v_head * cache_stride_head
            tl.store(
                V_CACHE + slots[:, None] + rh[None, :],
                lo.to(tl.bfloat16),
                mask=row_mask[:, None],
            )
            tl.store(
                V_CACHE + slots[:, None] + HALF + rh[None, :],
                hi.to(tl.bfloat16),
                mask=row_mask[:, None],
            )
    else:
        pass


@triton.jit
def gemv_add_kernel(
    X,
    x_stride_row,
    x_stride_t,
    x_stride_h,
    rows,
    W,
    RES_IN,
    res_in_stride,
    RES_OUT,
    PARTIALS,
    COUNTERS,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per output tile and K split: X against the tile's weight rows
    over the split's K range; the last program of the tile sums the partials in
    index order, rounds the product, then writes the rounded sum with RES_IN."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    HEADS_PER_SPLIT: tl.constexpr = K // D // SPLIT
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for head in range(pid_s * HEADS_PER_SPLIT, (pid_s + 1) * HEADS_PER_SPLIT):
        for kk in tl.static_range(0, D, BLOCK_K):
            k0 = head * D + kk
            x = tl.load(
                X + rows_off[:, None] + (head * x_stride_h + kk + rk)[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            acc = tl.dot(x, tl.trans(weight_tile(W, cols, k0, K, BLOCK_K)), acc)
    if SPLIT > 1:
        partial_offsets = (pid_s * BLOCK_M + rm)[:, None] * N + cols[None, :]
        tl.store(PARTIALS + partial_offsets, acc)
        tl.debug_barrier()
        arrived = tl.atomic_add(COUNTERS + pid_n, 1, sem="acq_rel", scope="gpu")
        is_last = arrived == SPLIT - 1
    else:
        is_last = pid_s == 0
    if is_last:
        if SPLIT > 1:
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for split in tl.static_range(SPLIT):
                acc += tl.load(
                    PARTIALS + (split * BLOCK_M + rm)[:, None] * N + cols[None, :],
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(COUNTERS + pid_n, 0)
        else:
            pass
        product = acc.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(
            RES_IN + rm[:, None] * res_in_stride + cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            RES_OUT + rm[:, None] * N + cols[None, :],
            (product + residual).to(tl.bfloat16),
            mask=row_mask[:, None],
        )
    else:
        pass


@triton.jit
def norm_gate_up_silu_kernel(
    X,
    x_stride_row,
    x_stride_t,
    rows,
    NORM_W,
    norm_eps,
    W,
    OUT,
    T: tl.constexpr,
    K: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per activation tile: the weighted input against the tile's gate
    and up rows with the rows' sum of squares alongside, the norm scale applied to
    both products, both rounded, then silu(gate) * up rounded once."""
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    gate_cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    up_cols = INTERMEDIATE + gate_cols
    square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for block in range(0, K // BLOCK_K):
        k0 = block * BLOCK_K
        x = tl.load(
            X + rows_off[:, None] + (k0 + rk)[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        square = tl.dot(x, tl.trans(x), square)
        scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
        acc_gate = tl.dot(
            scaled, tl.trans(weight_tile(W, gate_cols, k0, K, BLOCK_K)), acc_gate
        )
        acc_up = tl.dot(
            scaled, tl.trans(weight_tile(W, up_cols, k0, K, BLOCK_K)), acc_up
        )
    rstd = libdevice.rsqrt(diagonal(square) / K + norm_eps)
    gate = (acc_gate * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    up = (acc_up * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    activated = gate / (1.0 + libdevice.exp(-gate)) * up
    offsets = rm[:, None] * INTERMEDIATE + gate_cols[None, :]
    tl.store(OUT + offsets, activated.to(tl.bfloat16), mask=row_mask[:, None])


def is_plain_bf16_linear(linear: torch.nn.Module) -> bool:
    return (
        isinstance(linear.quant_method, UnquantizedLinearMethod)
        and linear.bias is None
        and linear.weight.dtype == torch.bfloat16
    )


def split_count(output_tiles: int, k_blocks: int, sm_count: int) -> int:
    """The power-of-two K split whose program count lands nearest the SM count."""
    candidates = [split for split in (1, 2, 4, 8) if k_blocks % split == 0]
    return min(candidates, key=lambda split: abs(output_tiles * split - sm_count))


def resolve_predictor_layer_shape(
    code_predictor: torch.nn.Module, predictor_len: int, device: torch.device
) -> PredictorLayerShape | None:
    """Plain bf16 layers with a neox rope on an fp32 cache and dimensions the blocks
    divide; None keeps the plain path."""
    if device.type != "cuda":
        return None
    else:
        pass
    layers = code_predictor.model.layers
    attention = layers[0].self_attn
    mlp = layers[0].mlp
    hidden_size = attention.hidden_size
    intermediate_size = mlp.down_proj.weight.shape[1]
    dims_divide = (
        attention.head_dim % BLOCK_K == 0
        and hidden_size % BLOCK_K == 0
        and hidden_size % BLOCK_N == 0
        and hidden_size % MLP_BLOCK_K == 0
        and intermediate_size % BLOCK_K == 0
        and intermediate_size % BLOCK_N == 0
    )
    if not dims_divide:
        return None
    else:
        pass
    for layer in layers:
        attention = layer.self_attn
        rope = attention.rotary_emb
        linears = (
            attention.qkv_proj,
            attention.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        )
        rope_matches = (
            type(rope) is RotaryEmbedding
            and rope.is_neox_style
            and rope.rotary_dim == attention.head_dim
            and rope.cos_sin_cache.dtype == torch.float32
            and rope.cos_sin_cache.shape[0] >= predictor_len
        )
        if not (
            rope_matches and all(is_plain_bf16_linear(linear) for linear in linears)
        ):
            return None
        else:
            pass
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    heads = attention.num_heads + 2 * attention.num_kv_heads
    split_hidden = min(
        split_count(hidden_size // BLOCK_N, attention.num_heads, sm_count),
        split_count(hidden_size // BLOCK_N, intermediate_size // BLOCK_K, sm_count),
    )
    return PredictorLayerShape(
        hidden_size=hidden_size,
        head_dim=attention.head_dim,
        num_q_heads=attention.num_heads,
        num_kv_heads=attention.num_kv_heads,
        intermediate_size=intermediate_size,
        split_qkv=split_count(heads, hidden_size // BLOCK_K, sm_count),
        split_hidden=split_hidden,
    )


def block_rows(rows: int) -> int:
    return max(MIN_BLOCK_M, triton.next_power_of_2(rows))


def partials_numel(shape: PredictorLayerShape, max_rows: int) -> int:
    """The fp32 partials the split launches need for max_rows rows, as one flat run
    that each launch strides by its own output width."""
    block_m = block_rows(max_rows)
    qkv_width = (shape.num_q_heads + 2 * shape.num_kv_heads) * shape.head_dim
    return block_m * max(
        shape.split_qkv * qkv_width, shape.split_hidden * shape.hidden_size
    )


def sum_sq_partials_numel(shape: PredictorLayerShape, max_rows: int) -> int:
    return shape.split_qkv * block_rows(max_rows)


def counters_numel(shape: PredictorLayerShape) -> int:
    return max(shape.num_q_heads + 2 * shape.num_kv_heads, shape.hidden_size // BLOCK_N)


def predictor_attention_inputs(
    *,
    x: torch.Tensor,
    tokens_per_row: int,
    norm: torch.nn.Module,
    attention: torch.nn.Module,
    q_out: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    partials: torch.Tensor,
    sum_sq_partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """Norm x's rows, project them, norm and rotate q and k; q into q_out and k and v
    into the caches at each row's position. x is (rows, hidden) with unit column stride;
    the caches are (rows // tokens_per_row, kv heads, slots, head dim)."""
    rows = x.shape[0]
    heads = shape.num_q_heads + 2 * shape.num_kv_heads
    norm_qkv_rope_store_kernel[(heads, shape.split_qkv)](
        x,
        tokens_per_row * x.stride(0),
        x.stride(0),
        rows,
        norm.weight,
        norm.variance_epsilon,
        attention.qkv_proj.weight,
        q_out,
        attention.q_norm.weight,
        attention.k_norm.weight,
        attention.q_norm.variance_epsilon,
        attention.rotary_emb.cos_sin_cache,
        positions,
        k_cache,
        v_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        partials,
        sum_sq_partials,
        counters,
        T=tokens_per_row,
        K=shape.hidden_size,
        HEAD_DIM=shape.head_dim,
        NUM_Q_HEADS=shape.num_q_heads,
        NUM_KV_HEADS=shape.num_kv_heads,
        SPLIT=shape.split_qkv,
        BLOCK_M=block_rows(rows),
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def predictor_o_proj_add(
    *,
    attention_output: torch.Tensor,
    tokens_per_row: int,
    weight: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """residual_out = bf16(bf16(attention_output @ weight.T) + residual_in), reading the
    attention output in its (row group, head, token, head dim) layout and residual_in
    with its own row stride."""
    rows = residual_out.shape[0]
    gemv_add_kernel[(shape.hidden_size // BLOCK_N, shape.split_hidden)](
        attention_output,
        attention_output.stride(0),
        attention_output.stride(2),
        attention_output.stride(1),
        rows,
        weight,
        residual_in,
        residual_in.stride(0),
        residual_out,
        partials,
        counters,
        T=tokens_per_row,
        D=shape.head_dim,
        K=shape.num_q_heads * shape.head_dim,
        N=shape.hidden_size,
        SPLIT=shape.split_hidden,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def predictor_mlp_up(
    *,
    residual: torch.Tensor,
    norm: torch.nn.Module,
    weight: torch.Tensor,
    activated: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """activated = bf16(silu(bf16(normed @ gate.T)) * bf16(normed @ up.T)) with the
    normed residual, gate and up being the halves of the merged weight."""
    rows = residual.shape[0]
    norm_gate_up_silu_kernel[(shape.intermediate_size // BLOCK_N,)](
        residual,
        residual.stride(0),
        residual.stride(0),
        rows,
        norm.weight,
        norm.variance_epsilon,
        weight,
        activated,
        T=1,
        K=shape.hidden_size,
        INTERMEDIATE=shape.intermediate_size,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=MLP_BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def predictor_down_add(
    *,
    activated: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """residual = bf16(bf16(activated @ weight.T) + residual), in place."""
    rows = residual.shape[0]
    gemv_add_kernel[(shape.hidden_size // BLOCK_N, shape.split_hidden)](
        activated,
        activated.stride(0),
        activated.stride(0),
        BLOCK_K,
        rows,
        weight,
        residual,
        residual.stride(0),
        residual,
        partials,
        counters,
        T=1,
        D=BLOCK_K,
        K=shape.intermediate_size,
        N=shape.hidden_size,
        SPLIT=shape.split_hidden,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
