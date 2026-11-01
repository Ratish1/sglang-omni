"""Prototype of design 15 (option O4): one predictor pass, the 5 decoder layers and the final norm, as one cooperative
persistent Triton kernel, against FusedPredictorLayers.forward on upstream main (sglang_omni/utils/predictor_layers.py).

One program per SM, launched with launch_cooperative_grid=True. A pass is 5 phases per layer (norm + qkv + rope +
cache store; attention; o_proj + residual; norm + gate_up + SiLU; down + residual), then the final norm: 26 phases,
25 device-wide barriers. Each phase body is main's kernel body verbatim (same tiles, BLOCK_M, BLOCK_K, K split, split
partials summed in index order by the last arriving program, same roundings); only the tile index comes from a
strided loop over the phase's tiles instead of the program id, and data written inside the pass is read with .cg
(L2, not the SM's L1, which a later phase could otherwise read stale). The final norm reproduces main's add-norm
kernel's reduction tree without the add, with explicit round-to-nearest products and sums, since this kernel compiles
with fp fusion on (main's norm kernel compiles with it off).

PREFETCH asks L2 for the next weight phase's lines at the start of each phase (o_proj during qkv and attention,
gate_up during o_proj, down during gate_up, the next layer's qkv during down; the last down asks for layer 0's qkv,
which the next pass reads first), each program a contiguous 1/programs share, so the HBM stream runs under the
current phase's compute and the barriers.

Weights are stacked per kind (layer major) for the prototype; the caches are main's (layer, row, slot, kv head, head
dim) tensors. Rows are packed (row group, token) major, as both callers pass them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang_omni.utils.predictor_layers import (
    ATTENTION_ROWS,
    BLOCK_K,
    BLOCK_N,
    LANES,
    MLP_BLOCK_K,
    NUM_STAGES,
    NUM_WARPS,
    REDUCED_NUM_STAGES,
    VEC,
    allocate_split_scratch,
    block_rows,
    butterfly_sum,
    diagonal,
    row_offsets,
    scaled_tile,
    weight_tile,
)

PHASES_PER_LAYER = 5
# note (ratish): one bf16 L2 line is 128 bytes.
LINE_ELEMENTS = 64
LINE_BLOCK = 1024


@triton.jit
def grid_barrier(BARRIERS, index, barrier_count, program, programs):
    """Every program arrives at counter index (release), spins until all arrived, then acquires. Program 0 resets the
    previous barrier's counter, which every program has left; the last one is reset at the next call's first.
    """
    tl.debug_barrier()
    tl.atomic_add(BARRIERS + index, 1, sem="release", scope="gpu")
    arrived = tl.load(BARRIERS + index, volatile=True)
    while arrived < programs:
        arrived = tl.load(BARRIERS + index, volatile=True)
    tl.atomic_add(BARRIERS + index, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()
    if program == 0:
        tl.store(BARRIERS + (index + barrier_count - 1) % barrier_count, 0)
    else:
        pass
    return index + 1


@triton.jit
def prefetch_l2(
    W, numel, program, programs, LINE_BLOCK: tl.constexpr, LINE_ELEMENTS: tl.constexpr
):
    """Ask L2 for this program's contiguous share of W's lines; nothing is loaded into registers."""
    lines = numel // LINE_ELEMENTS
    share = tl.cdiv(lines, programs)
    first = program * share
    last = tl.minimum(first + share, lines) - 1
    for start in range(first, last + 1, LINE_BLOCK):
        line = tl.minimum(start + tl.arange(0, LINE_BLOCK), last)
        tl.inline_asm_elementwise(
            "mov.u32 $0, 0;\n\tprefetch.global.L2 [$1];",
            "=r,l",
            [W + line.to(tl.int64) * LINE_ELEMENTS],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def qkv_tile(
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
    cache_stride_slot,
    cache_stride_head,
    PARTIALS,
    SUM_SQ_PARTIALS,
    COUNTERS,
    head,
    pid_s,
    T: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """norm_qkv_rope_store_kernel's body for one (head, K split)."""
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
            cache_modifier=".cg",
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
        cache_rows = rm // T * cache_stride_row + pos * cache_stride_slot
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
def attention_tile(
    Q,
    K_CACHE,
    V_CACHE,
    cache_stride_row,
    cache_stride_slot,
    cache_stride_head,
    OUT,
    end,
    scale,
    row_group,
    kv_head,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_ROW_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """decode_attention_kernel's body for one (row group, kv head)."""
    GROUP: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    rq = tl.arange(0, BLOCK_Q)
    rs = tl.arange(0, BLOCK_S)
    rd = tl.arange(0, HEAD_DIM)
    token = rq // GROUP
    is_query = rq < T * GROUP
    q_offsets = (
        (row_group * T + token)[:, None] * Q_ROW_STRIDE
        + (kv_head * GROUP + rq % GROUP)[:, None] * HEAD_DIM
        + rd[None, :]
    )
    out_offsets = (
        (row_group * T + token)[:, None] * (NUM_Q_HEADS * HEAD_DIM)
        + (kv_head * GROUP + rq % GROUP)[:, None] * HEAD_DIM
        + rd[None, :]
    )
    cache_offsets = (
        row_group * cache_stride_row
        + kv_head * cache_stride_head
        + rs[:, None] * cache_stride_slot
        + rd[None, :]
    )
    q = tl.load(Q + q_offsets, mask=is_query[:, None], other=0.0, cache_modifier=".cg")
    k = tl.load(
        K_CACHE + cache_offsets,
        mask=(rs < end)[:, None],
        other=0.0,
        cache_modifier=".cg",
    )
    v = tl.load(
        V_CACHE + cache_offsets,
        mask=(rs < end)[:, None],
        other=0.0,
        cache_modifier=".cg",
    )
    scores = tl.dot(q, tl.trans(k)) * scale
    visible = rs[None, :] < end - T + token[:, None] + 1
    scores = tl.where(visible, scores, float("-inf"))
    weights = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    attended = tl.dot(weights.to(tl.bfloat16), v) / tl.sum(weights, 1)[:, None]
    tl.store(OUT + out_offsets, attended.to(tl.bfloat16), mask=is_query[:, None])


@triton.jit
def gemv_add_tile(
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
    pid_n,
    pid_s,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """gemv_add_kernel's body for one (output tile, K split)."""
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
                cache_modifier=".cg",
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
            cache_modifier=".cg",
        ).to(tl.float32)
        tl.store(
            RES_OUT + rm[:, None] * N + cols[None, :],
            (product + residual).to(tl.bfloat16),
            mask=row_mask[:, None],
        )
    else:
        pass


@triton.jit
def gate_up_tile(
    X,
    x_stride_row,
    x_stride_t,
    rows,
    NORM_W,
    norm_eps,
    W,
    OUT,
    pid,
    T: tl.constexpr,
    K: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """norm_gate_up_silu_kernel's body for one activation tile."""
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rn = tl.arange(0, 2 * BLOCK_N)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    gate_cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    stacked_cols = pid * BLOCK_N + tl.where(
        rn < BLOCK_N, rn, INTERMEDIATE - BLOCK_N + rn
    )
    square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 2 * BLOCK_N], dtype=tl.float32)
    for block in range(0, K // BLOCK_K):
        k0 = block * BLOCK_K
        x = tl.load(
            X + rows_off[:, None] + (k0 + rk)[None, :],
            mask=row_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        square = tl.dot(x, tl.trans(x), square)
        scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
        acc = tl.dot(
            scaled, tl.trans(weight_tile(W, stacked_cols, k0, K, BLOCK_K)), acc
        )
    rstd = libdevice.rsqrt(diagonal(square) / K + norm_eps)
    products = (acc * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(
        tl.permute(tl.reshape(products, [BLOCK_M, 2, BLOCK_N]), [0, 2, 1])
    )
    activated = gate / (1.0 + libdevice.exp(-gate)) * up
    offsets = rm[:, None] * INTERMEDIATE + gate_cols[None, :]
    tl.store(OUT + offsets, activated.to(tl.bfloat16), mask=row_mask[:, None])


@triton.jit
def norm_row(
    X,
    WEIGHT,
    OUT,
    eps,
    row,
    HIDDEN: tl.constexpr,
    BLOCKS: tl.constexpr,
    LANE_VEC: tl.constexpr,
):
    """add_rmsnorm_rounded_kernel's tree on one stored bf16 row, every product and sum rounded on its own."""
    lane = tl.arange(0, 32)
    j = tl.arange(0, LANE_VEC)
    row_base = row * HIDDEN
    acc = tl.zeros([32], dtype=tl.float32)
    for block in tl.static_range(BLOCKS):
        col = LANE_VEC * lane[:, None] + 32 * LANE_VEC * block + j[None, :]
        value = tl.load(X + row_base + col, cache_modifier=".cg").to(tl.float32)
        squares = libdevice.mul_rn(value, value)
        for element in tl.static_range(LANE_VEC):
            acc = libdevice.add_rn(
                acc, tl.sum(tl.where(j[None, :] == element, squares, 0.0), 1)
            )
    sum_sq = butterfly_sum(acc)
    rstd = libdevice.rsqrt(libdevice.add_rn(sum_sq / HIDDEN, eps))
    block = tl.arange(0, BLOCKS)
    col = (
        LANE_VEC * lane[:, None, None]
        + 32 * LANE_VEC * block[None, :, None]
        + j[None, None, :]
    )
    value = tl.load(X + row_base + col, cache_modifier=".cg").to(tl.float32)
    weight = tl.load(WEIGHT + col).to(tl.float32)
    tl.store(
        OUT + row_base + col,
        libdevice.mul_rn(
            libdevice.mul_rn(value, rstd), libdevice.add_rn(weight, 0.0)
        ).to(tl.bfloat16),
    )


@triton.jit
def predictor_pass_kernel(
    X,
    rows,
    end,
    attention_scale,
    POS,
    COS_SIN,
    IN_NORM,
    POST_NORM,
    Q_NORM,
    K_NORM,
    FINAL_NORM_W,
    norm_eps,
    qk_eps,
    final_eps,
    W_QKV,
    W_O,
    W_GATE_UP,
    W_DOWN,
    K_CACHE,
    V_CACHE,
    cache_stride_layer,
    cache_stride_row,
    cache_stride_slot,
    cache_stride_head,
    Q_BUF,
    ATTN_OUT,
    RESIDUAL,
    ACTIVATED,
    OUT,
    PARTIALS,
    SUM_SQ_PARTIALS,
    COUNTERS,
    BARRIERS,
    T: tl.constexpr,
    LAYERS: tl.constexpr,
    HIDDEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    SPLIT_QKV: tl.constexpr,
    SPLIT_HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MLP_BLOCK_K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_S: tl.constexpr,
    LANES: tl.constexpr,
    VEC: tl.constexpr,
    PREFETCH: tl.constexpr,
    LINE_BLOCK: tl.constexpr,
    LINE_ELEMENTS: tl.constexpr,
):
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    Q_WIDTH: tl.constexpr = NUM_Q_HEADS * HEAD_DIM
    QKV_NUMEL: tl.constexpr = HEADS * HEAD_DIM * HIDDEN
    O_NUMEL: tl.constexpr = HIDDEN * Q_WIDTH
    GATE_UP_NUMEL: tl.constexpr = 2 * INTERMEDIATE * HIDDEN
    DOWN_NUMEL: tl.constexpr = HIDDEN * INTERMEDIATE
    HIDDEN_TILES: tl.constexpr = HIDDEN // BLOCK_N
    BARRIER_COUNT: tl.constexpr = LAYERS * 5
    barrier = 0
    for layer in range(LAYERS):
        if layer == 0:
            x_in = X
        else:
            x_in = RESIDUAL
        layer_k = K_CACHE + layer * cache_stride_layer
        layer_v = V_CACHE + layer * cache_stride_layer
        if PREFETCH:
            prefetch_l2(
                W_O + layer * O_NUMEL,
                O_NUMEL,
                program,
                programs,
                LINE_BLOCK,
                LINE_ELEMENTS,
            )
        else:
            pass
        for tile in range(program, HEADS * SPLIT_QKV, programs):
            qkv_tile(
                x_in,
                T * HIDDEN,
                HIDDEN,
                rows,
                IN_NORM + layer * HIDDEN,
                norm_eps,
                W_QKV + layer * QKV_NUMEL,
                Q_BUF,
                Q_NORM + layer * HEAD_DIM,
                K_NORM + layer * HEAD_DIM,
                qk_eps,
                COS_SIN,
                POS,
                layer_k,
                layer_v,
                cache_stride_row,
                cache_stride_slot,
                cache_stride_head,
                PARTIALS,
                SUM_SQ_PARTIALS,
                COUNTERS,
                tile // SPLIT_QKV,
                tile % SPLIT_QKV,
                T,
                HIDDEN,
                HEAD_DIM,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                SPLIT_QKV,
                BLOCK_M,
                BLOCK_K,
            )
        barrier = grid_barrier(BARRIERS, barrier, BARRIER_COUNT, program, programs)
        for tile in range(program, rows // T * NUM_KV_HEADS, programs):
            attention_tile(
                Q_BUF,
                layer_k,
                layer_v,
                cache_stride_row,
                cache_stride_slot,
                cache_stride_head,
                ATTN_OUT,
                end,
                attention_scale,
                tile // NUM_KV_HEADS,
                tile % NUM_KV_HEADS,
                T,
                HEAD_DIM,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                Q_WIDTH,
                BLOCK_Q,
                BLOCK_S,
            )
        barrier = grid_barrier(BARRIERS, barrier, BARRIER_COUNT, program, programs)
        if PREFETCH:
            prefetch_l2(
                W_GATE_UP + layer * GATE_UP_NUMEL,
                GATE_UP_NUMEL,
                program,
                programs,
                LINE_BLOCK,
                LINE_ELEMENTS,
            )
        else:
            pass
        for tile in range(program, HIDDEN_TILES * SPLIT_HIDDEN, programs):
            gemv_add_tile(
                ATTN_OUT,
                T * Q_WIDTH,
                Q_WIDTH,
                HEAD_DIM,
                rows,
                W_O + layer * O_NUMEL,
                x_in,
                HIDDEN,
                RESIDUAL,
                PARTIALS,
                COUNTERS,
                tile // SPLIT_HIDDEN,
                tile % SPLIT_HIDDEN,
                T,
                HEAD_DIM,
                Q_WIDTH,
                HIDDEN,
                SPLIT_HIDDEN,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )
        barrier = grid_barrier(BARRIERS, barrier, BARRIER_COUNT, program, programs)
        if PREFETCH:
            prefetch_l2(
                W_DOWN + layer * DOWN_NUMEL,
                DOWN_NUMEL,
                program,
                programs,
                LINE_BLOCK,
                LINE_ELEMENTS,
            )
        else:
            pass
        for tile in range(program, INTERMEDIATE // BLOCK_N, programs):
            gate_up_tile(
                RESIDUAL,
                HIDDEN,
                HIDDEN,
                rows,
                POST_NORM + layer * HIDDEN,
                norm_eps,
                W_GATE_UP + layer * GATE_UP_NUMEL,
                ACTIVATED,
                tile,
                1,
                HIDDEN,
                INTERMEDIATE,
                BLOCK_M,
                BLOCK_N,
                MLP_BLOCK_K,
            )
        barrier = grid_barrier(BARRIERS, barrier, BARRIER_COUNT, program, programs)
        if PREFETCH:
            prefetch_l2(
                W_QKV + (layer + 1) % LAYERS * QKV_NUMEL,
                QKV_NUMEL,
                program,
                programs,
                LINE_BLOCK,
                LINE_ELEMENTS,
            )
        else:
            pass
        for tile in range(program, HIDDEN_TILES * SPLIT_HIDDEN, programs):
            gemv_add_tile(
                ACTIVATED,
                INTERMEDIATE,
                INTERMEDIATE,
                BLOCK_K,
                rows,
                W_DOWN + layer * DOWN_NUMEL,
                RESIDUAL,
                HIDDEN,
                RESIDUAL,
                PARTIALS,
                COUNTERS,
                tile // SPLIT_HIDDEN,
                tile % SPLIT_HIDDEN,
                1,
                BLOCK_K,
                INTERMEDIATE,
                HIDDEN,
                SPLIT_HIDDEN,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )
        barrier = grid_barrier(BARRIERS, barrier, BARRIER_COUNT, program, programs)
    for row in range(program, rows, programs):
        norm_row(
            RESIDUAL,
            FINAL_NORM_W,
            OUT,
            final_eps,
            row,
            HIDDEN,
            HIDDEN // (LANES * VEC),
            VEC,
        )


class PersistentPredictorPass:
    """One cooperative launch per predictor pass over stacked layer weights, with FusedPredictorLayers' buffers."""

    def __init__(
        self,
        layers,
        final_norm,
        shape,
        max_rows: int,
        device: torch.device,
        prefetch: bool,
    ) -> None:
        self.shape = shape
        self.max_rows = max_rows
        self.prefetch = prefetch
        self.programs = torch.cuda.get_device_properties(device).multi_processor_count
        attention = [layer.self_attn for layer in layers]
        self.w_qkv = torch.stack([a.qkv_proj.weight for a in attention]).contiguous()
        self.w_o = torch.stack([a.o_proj.weight for a in attention]).contiguous()
        self.w_gate_up = torch.stack(
            [layer.mlp.gate_up_proj.weight for layer in layers]
        ).contiguous()
        self.w_down = torch.stack(
            [layer.mlp.down_proj.weight for layer in layers]
        ).contiguous()
        self.in_norm = torch.stack(
            [layer.input_layernorm.weight for layer in layers]
        ).contiguous()
        self.post_norm = torch.stack(
            [layer.post_attention_layernorm.weight for layer in layers]
        ).contiguous()
        self.q_norm = torch.stack([a.q_norm.weight for a in attention]).contiguous()
        self.k_norm = torch.stack([a.k_norm.weight for a in attention]).contiguous()
        self.norm_eps = layers[0].input_layernorm.variance_epsilon
        self.qk_eps = attention[0].q_norm.variance_epsilon
        self.final_norm = final_norm
        self.cos_sin = attention[0].rotary_emb.cos_sin_cache
        self.num_layers = len(layers)
        dtype = torch.bfloat16
        self.q = torch.zeros(
            max_rows, shape.num_q_heads * shape.head_dim, device=device, dtype=dtype
        )
        self.attention_output = torch.zeros_like(self.q)
        self.residual = torch.zeros(
            max_rows, shape.hidden_size, device=device, dtype=dtype
        )
        self.activated = torch.zeros(
            max_rows, shape.intermediate_size, device=device, dtype=dtype
        )
        self.out = torch.zeros(max_rows, shape.hidden_size, device=device, dtype=dtype)
        self.partials, self.sum_sq_partials, self.counters = allocate_split_scratch(
            shape, max_rows, device
        )
        self.barriers = torch.zeros(
            PHASES_PER_LAYER * self.num_layers, device=device, dtype=torch.int32
        )
        self.num_stages: dict[int, int] = {}
        self.compiled = {}

    def forward(
        self, *, token_embeds, batch_size, cache_len, positions, k_cache, v_cache
    ) -> torch.Tensor:
        shape = self.shape
        num_tokens, hidden_size = token_embeds.shape[1:]
        rows = batch_size * num_tokens
        if rows > self.max_rows or not token_embeds.is_contiguous():
            raise ValueError(f"rows {rows} over {self.max_rows} or input not packed")
        else:
            pass
        block_m = block_rows(rows)
        while True:
            stages = self.num_stages.get(block_m, NUM_STAGES)
            try:
                handle = predictor_pass_kernel[(self.programs,)](
                    token_embeds,
                    rows,
                    cache_len + num_tokens,
                    shape.head_dim**-0.5,
                    positions,
                    self.cos_sin,
                    self.in_norm,
                    self.post_norm,
                    self.q_norm,
                    self.k_norm,
                    self.final_norm.weight,
                    self.norm_eps,
                    self.qk_eps,
                    self.final_norm.variance_epsilon,
                    self.w_qkv,
                    self.w_o,
                    self.w_gate_up,
                    self.w_down,
                    k_cache,
                    v_cache,
                    k_cache.stride(0),
                    k_cache.stride(1),
                    k_cache.stride(2),
                    k_cache.stride(3),
                    self.q,
                    self.attention_output,
                    self.residual,
                    self.activated,
                    self.out,
                    self.partials,
                    self.sum_sq_partials,
                    self.counters,
                    self.barriers,
                    T=num_tokens,
                    LAYERS=self.num_layers,
                    HIDDEN=hidden_size,
                    HEAD_DIM=shape.head_dim,
                    NUM_Q_HEADS=shape.num_q_heads,
                    NUM_KV_HEADS=shape.num_kv_heads,
                    INTERMEDIATE=shape.intermediate_size,
                    SPLIT_QKV=shape.split_qkv,
                    SPLIT_HIDDEN=shape.split_hidden,
                    BLOCK_M=block_m,
                    BLOCK_K=BLOCK_K,
                    BLOCK_N=BLOCK_N,
                    MLP_BLOCK_K=MLP_BLOCK_K,
                    BLOCK_Q=ATTENTION_ROWS,
                    BLOCK_S=triton.next_power_of_2(k_cache.shape[2]),
                    LANES=LANES,
                    VEC=VEC,
                    PREFETCH=self.prefetch,
                    LINE_BLOCK=LINE_BLOCK,
                    LINE_ELEMENTS=LINE_ELEMENTS,
                    num_warps=NUM_WARPS,
                    num_stages=stages,
                    launch_cooperative_grid=True,
                )
            except triton.runtime.OutOfResources:
                if stages == REDUCED_NUM_STAGES:
                    raise
                else:
                    self.num_stages[block_m] = REDUCED_NUM_STAGES
            else:
                self.compiled[(block_m, num_tokens)] = handle
                break
        return self.out[:rows].view(batch_size, num_tokens, hidden_size)
