"""Controlled experiments on the Triton code predictor chain: what costs time inside its kernels.

E1: a chain of 29 empty kernels with programmatic dependent launch, the launch and dependency floor.
E2: one codebook step's chain with each fused kernel changed in one mechanism:
    prefetch, its weight rows into L2 before waiting on the previous kernel;
    early, releasing its dependents at its start instead of after its weight loop;
    both.
E3: split, block and warp sweeps of the unchanged kernels.

Usage: python3 predictor_experiments.py --out <json>
"""

import argparse
import json

import torch
import triton
import triton.language as tl
from predictor_bench import (
    CACHED,
    EPS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    LAYERS,
    Q_HEADS,
    SLOTS,
    load_predictor,
    rope_cache,
    time_call,
)
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from sglang_omni.utils import predictor_layers as fused
from sglang_omni.utils.predictor_layers import (
    diagonal,
    row_offsets,
    scaled_tile,
    weight_tile,
)

BATCHES = (1, 16, 64)


@triton.jit
def empty_kernel(X, USE_GDC: tl.constexpr):
    if USE_GDC:
        gdc_wait()
    else:
        pass
    if tl.program_id(0) == 0:
        tl.store(X, tl.load(X) + 1)
    else:
        pass
    if USE_GDC:
        gdc_launch_dependents()
    else:
        pass


@triton.jit
def prefetch_rows(W, rows, K: tl.constexpr, k_start, SPAN: tl.constexpr):
    """One L2 prefetch for every 128-byte line of the rows' K span."""
    LINES: tl.constexpr = SPAN // 64
    line = tl.minimum(tl.arange(0, 16), LINES - 1)
    pointers = W + rows[:, None] * K + k_start + line[None, :] * 64
    tl.inline_asm_elementwise(
        "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;",
        "=r,l",
        [pointers],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def norm_qkv_rope_store_variant(
    X, x_stride_row, x_stride_t, rows, NORM_W, norm_eps, W, Q_OUT, QN_W, KN_W, qk_eps, COS_SIN, POS,
    K_CACHE, V_CACHE, cache_stride_row, cache_stride_slot, cache_stride_head, PARTIALS, SUM_SQ_PARTIALS,
    COUNTERS, T: tl.constexpr, K: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr, PREFETCH: tl.constexpr, EARLY: tl.constexpr,
):
    head = tl.program_id(0)
    pid_s = tl.program_id(1)
    HALF: tl.constexpr = HEAD_DIM // 2
    N_TOTAL: tl.constexpr = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    BLOCKS_PER_SPLIT: tl.constexpr = K // BLOCK_K // SPLIT
    rh = tl.arange(0, HALF)
    lo_cols = head * HEAD_DIM + rh
    hi_cols = lo_cols + HALF
    if PREFETCH:
        k_start = pid_s * BLOCKS_PER_SPLIT * BLOCK_K
        prefetch_rows(W, lo_cols, K, k_start, BLOCKS_PER_SPLIT * BLOCK_K)
        prefetch_rows(W, hi_cols, K, k_start, BLOCKS_PER_SPLIT * BLOCK_K)
    else:
        pass
    if USE_GDC and EARLY:
        gdc_launch_dependents()
    else:
        pass
    if USE_GDC:
        gdc_wait()
    else:
        pass
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    for block in range(pid_s * BLOCKS_PER_SPLIT, (pid_s + 1) * BLOCKS_PER_SPLIT):
        k0 = block * BLOCK_K
        x = tl.load(X + rows_off[:, None] + (k0 + rk)[None, :], mask=row_mask[:, None], other=0.0)
        square = tl.dot(x, tl.trans(x), square)
        scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
        acc_lo = tl.dot(scaled, tl.trans(weight_tile(W, lo_cols, k0, K, BLOCK_K)), acc_lo)
        acc_hi = tl.dot(scaled, tl.trans(weight_tile(W, hi_cols, k0, K, BLOCK_K)), acc_hi)
    if USE_GDC and not EARLY:
        gdc_launch_dependents()
    else:
        pass
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
                acc_lo += tl.load(PARTIALS + partial_rows + lo_cols[None, :], cache_modifier=".cg")
                acc_hi += tl.load(PARTIALS + partial_rows + hi_cols[None, :], cache_modifier=".cg")
                sum_sq += tl.load(SUM_SQ_PARTIALS + split * BLOCK_M + rm, cache_modifier=".cg")
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
            lo = (lo * head_rstd[:, None] * w_lo[None, :]).to(tl.bfloat16).to(tl.float32)
            hi = (hi * head_rstd[:, None] * w_hi[None, :]).to(tl.bfloat16).to(tl.float32)
            cos = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + rh[None, :])
            sin = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + HALF + rh[None, :])
            out_lo = (lo * cos - hi * sin).to(tl.bfloat16)
            out_hi = (lo * sin + hi * cos).to(tl.bfloat16)
            if head < NUM_Q_HEADS:
                q_rows = rm * (NUM_Q_HEADS * HEAD_DIM)
                tl.store(Q_OUT + q_rows[:, None] + lo_cols[None, :], out_lo, mask=row_mask[:, None])
                tl.store(Q_OUT + q_rows[:, None] + hi_cols[None, :], out_hi, mask=row_mask[:, None])
            else:
                slots = cache_rows + (head - NUM_Q_HEADS) * cache_stride_head
                tl.store(K_CACHE + slots[:, None] + rh[None, :], out_lo, mask=row_mask[:, None])
                tl.store(K_CACHE + slots[:, None] + HALF + rh[None, :], out_hi, mask=row_mask[:, None])
        else:
            v_head = head - NUM_Q_HEADS - NUM_KV_HEADS
            slots = cache_rows + v_head * cache_stride_head
            tl.store(V_CACHE + slots[:, None] + rh[None, :], lo.to(tl.bfloat16), mask=row_mask[:, None])
            tl.store(V_CACHE + slots[:, None] + HALF + rh[None, :], hi.to(tl.bfloat16), mask=row_mask[:, None])
    else:
        pass


@triton.jit
def gemv_add_variant(
    X, x_stride_row, x_stride_t, x_stride_h, rows, W, RES_IN, res_in_stride, RES_OUT, PARTIALS, COUNTERS,
    T: tl.constexpr, D: tl.constexpr, K: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, USE_GDC: tl.constexpr,
    PREFETCH: tl.constexpr, EARLY: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    HEADS_PER_SPLIT: tl.constexpr = K // D // SPLIT
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if PREFETCH:
        prefetch_rows(W, cols, K, pid_s * HEADS_PER_SPLIT * D, HEADS_PER_SPLIT * D)
    else:
        pass
    if USE_GDC and EARLY:
        gdc_launch_dependents()
    else:
        pass
    if USE_GDC:
        gdc_wait()
    else:
        pass
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for head in range(pid_s * HEADS_PER_SPLIT, (pid_s + 1) * HEADS_PER_SPLIT):
        for kk in tl.static_range(0, D, BLOCK_K):
            k0 = head * D + kk
            x = tl.load(
                X + rows_off[:, None] + (head * x_stride_h + kk + rk)[None, :], mask=row_mask[:, None], other=0.0
            )
            acc = tl.dot(x, tl.trans(weight_tile(W, cols, k0, K, BLOCK_K)), acc)
    if USE_GDC and not EARLY:
        gdc_launch_dependents()
    else:
        pass
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
                acc += tl.load(PARTIALS + (split * BLOCK_M + rm)[:, None] * N + cols[None, :], cache_modifier=".cg")
            tl.atomic_xchg(COUNTERS + pid_n, 0)
        else:
            pass
        product = acc.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(RES_IN + rm[:, None] * res_in_stride + cols[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        tl.store(RES_OUT + rm[:, None] * N + cols[None, :], (product + residual).to(tl.bfloat16), mask=row_mask[:, None])
    else:
        pass


@triton.jit
def norm_gate_up_silu_variant(
    X, x_stride_row, x_stride_t, rows, NORM_W, norm_eps, W, OUT, T: tl.constexpr, K: tl.constexpr,
    INTERMEDIATE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr, PREFETCH: tl.constexpr, EARLY: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = tl.arange(0, 2 * BLOCK_N)
    stacked_cols = pid * BLOCK_N + tl.where(rn < BLOCK_N, rn, INTERMEDIATE - BLOCK_N + rn)
    if PREFETCH:
        prefetch_rows(W, stacked_cols, K, 0, K)
    else:
        pass
    if USE_GDC and EARLY:
        gdc_launch_dependents()
    else:
        pass
    if USE_GDC:
        gdc_wait()
    else:
        pass
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < rows
    rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
    gate_cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 2 * BLOCK_N], dtype=tl.float32)
    for block in range(0, K // BLOCK_K):
        k0 = block * BLOCK_K
        x = tl.load(X + rows_off[:, None] + (k0 + rk)[None, :], mask=row_mask[:, None], other=0.0)
        square = tl.dot(x, tl.trans(x), square)
        scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
        acc = tl.dot(scaled, tl.trans(weight_tile(W, stacked_cols, k0, K, BLOCK_K)), acc)
    if USE_GDC and not EARLY:
        gdc_launch_dependents()
    else:
        pass
    rstd = libdevice.rsqrt(diagonal(square) / K + norm_eps)
    products = (acc * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.permute(tl.reshape(products, [BLOCK_M, 2, BLOCK_N]), [0, 2, 1]))
    activated = gate / (1.0 + libdevice.exp(-gate)) * up
    tl.store(OUT + rm[:, None] * INTERMEDIATE + gate_cols[None, :], activated.to(tl.bfloat16), mask=row_mask[:, None])


class WithConstants:
    """A kernel whose launches carry extra constexpr arguments the shared launchers do not pass."""

    def __init__(self, kernel, **constants):
        self.kernel = kernel
        self.constants = constants

    def __getitem__(self, grid):
        launcher = self.kernel[grid]
        return lambda *args, **kwargs: launcher(*args, **kwargs, **self.constants)


ORIGINAL = (fused.norm_qkv_rope_store_kernel, fused.gemv_add_kernel, fused.norm_gate_up_silu_kernel)


def install(prefetch: bool | None, early: bool) -> None:
    """prefetch None restores the shared kernels."""
    if prefetch is None:
        fused.norm_qkv_rope_store_kernel, fused.gemv_add_kernel, fused.norm_gate_up_silu_kernel = ORIGINAL
    else:
        fused.norm_qkv_rope_store_kernel = WithConstants(norm_qkv_rope_store_variant, PREFETCH=prefetch, EARLY=early)
        fused.gemv_add_kernel = WithConstants(gemv_add_variant, PREFETCH=prefetch, EARLY=early)
        fused.norm_gate_up_silu_kernel = WithConstants(norm_gate_up_silu_variant, PREFETCH=prefetch, EARLY=early)


def build_layers(model, cos_sin):
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=layer.input_norm, variance_epsilon=EPS),
            post_attention_layernorm=SimpleNamespace(weight=layer.post_norm, variance_epsilon=EPS),
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=layer.qkv),
                q_norm=SimpleNamespace(weight=layer.q_norm, variance_epsilon=EPS),
                k_norm=SimpleNamespace(weight=layer.k_norm, variance_epsilon=EPS),
                rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
                o_proj=SimpleNamespace(weight=layer.o),
            ),
            mlp=SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=layer.gate_up), down_proj=SimpleNamespace(weight=layer.down)
            ),
        )
        for layer in model.layers
    ]


def layer_chain(layers, shape, rows, k_cache, v_cache, token, device):
    """One codebook step's five fused layers, no lm_head or sampler."""
    fused_layers = fused.FusedPredictorLayers(shape, 64, device, torch.bfloat16)
    positions = torch.full((rows,), CACHED, device=device, dtype=torch.long)

    def run():
        return fused_layers.forward(
            layers=layers, final_norm=lambda value: value, token_embeds=token, batch_size=rows, cache_len=CACHED,
            positions=positions, k_cache=k_cache, v_cache=v_cache,
        )

    return run


def make_shape(split_qkv: int, split_hidden: int) -> fused.PredictorLayerShape:
    return fused.PredictorLayerShape(
        hidden_size=HIDDEN, head_dim=HEAD_DIM, num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE, split_qkv=split_qkv, split_hidden=split_hidden,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    cos_sin = rope_cache(device)
    layers = build_layers(model, cos_sin)
    k_cache = torch.randn(LAYERS, 64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    results = {}
    counter = torch.zeros(1, device=device, dtype=torch.int32)
    for use_gdc in (True, False):
        results[f"E1 29 empty kernels, pdl={use_gdc}"] = time_call(
            lambda: [empty_kernel[(128,)](counter, USE_GDC=use_gdc, launch_pdl=use_gdc) for _ in range(29)]
        )
    print(json.dumps(results, indent=1), flush=True)
    for rows in BATCHES:
        token = torch.randn(rows, 1, HIDDEN, device=device, dtype=torch.bfloat16)
        cell = {}
        reference = None
        for label, prefetch, early in (
            ("baseline", None, False),
            ("variant, no change", False, False),
            ("prefetch before wait", True, False),
            ("early release", False, True),
            ("prefetch and early release", True, True),
        ):
            install(prefetch, early)
            run = layer_chain(layers, make_shape(4, 4), rows, k_cache, v_cache, token, device)
            cell[f"E2 5 layers, {label}"] = time_call(run)
            output = run().float()
            if reference is None:
                reference = output
            else:
                cell[f"E2 {label} max abs vs baseline"] = (output - reference).abs().max().item()
        install(None, False)
        for split_qkv, split_hidden in ((1, 1), (2, 2), (4, 4), (8, 8), (8, 4), (4, 8)):
            run = layer_chain(layers, make_shape(split_qkv, split_hidden), rows, k_cache, v_cache, token, device)
            cell[f"E3 5 layers, split qkv {split_qkv} hidden {split_hidden}"] = time_call(run)
        for block_n in (16, 64):
            fused.BLOCK_N = block_n
            run = layer_chain(layers, make_shape(4, 4), rows, k_cache, v_cache, token, device)
            cell[f"E3 5 layers, block_n {block_n}"] = time_call(run)
        fused.BLOCK_N = 32
        for warps in (2, 8):
            fused.NUM_WARPS = warps
            run = layer_chain(layers, make_shape(4, 4), rows, k_cache, v_cache, token, device)
            cell[f"E3 5 layers, num_warps {warps}"] = time_call(run)
        fused.NUM_WARPS = 4
        results[rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
