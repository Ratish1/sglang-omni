"""The code predictor's step with decode attention variants swapped into the fused layer of the tree on PYTHONPATH
(perf/qwen3-tts-predictor-triton-attention), one process, step replay per variant and batch.

usage: PYTHONPATH=<P6 tree> python3 predictor_attention_variants.py [--batch 1 16 64] [--variant NAME]

Only the first variant a process times is comparable at batch 1 (later ones in the same process ran 0.15 ms slower
per step, the same kernels included), so a comparison runs one variant per process with --variant.

Variants with o_proj under PDL as the branch launches it: p6 (the branch's kernel, one program per row and q head,
releasing o_proj right after its wait), p6 released at end (the same, releasing after its store), grouped released at
end (one program per row and kv head over its q heads, releasing after its store), grouped released early; and the
per-head and grouped kernels with o_proj launched plainly, as V4 launches it after SDPA. Each
variant's step is timed by predictor_layers_bench.bench, which captures one served step in a graph; V4's own run of
that bench is the SDPA baseline.
"""

import argparse

import torch
import triton
import triton.language as tl
from predictor_layers_bench import bench
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from sglang_omni.utils import predictor_layers


@triton.jit
def grouped_attention_kernel(
    Q,
    K_CACHE,
    V_CACHE,
    cache_stride_row,
    cache_stride_slot,
    cache_stride_head,
    OUT,
    end,
    scale,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    GROUPED: tl.constexpr,
    RELEASE_EARLY: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    GROUP: tl.constexpr = (NUM_Q_HEADS // NUM_KV_HEADS) * GROUPED + (1 - GROUPED)
    gdc_wait()
    if RELEASE_EARLY:
        gdc_launch_dependents()
    else:
        pass
    if GROUPED:
        kv_head = head
        first_q_head = head * GROUP
    else:
        kv_head = head // (NUM_Q_HEADS // NUM_KV_HEADS)
        first_q_head = head
    rg = tl.arange(0, GROUP)
    rs = tl.arange(0, BLOCK_S)
    rd = tl.arange(0, HEAD_DIM)
    visible = rs < end - T + row % T + 1
    q_offsets = row * (NUM_Q_HEADS * HEAD_DIM) + (first_q_head + rg)[:, None] * HEAD_DIM + rd[None, :]
    cache_offsets = (
        row // T * cache_stride_row + kv_head * cache_stride_head + rs[:, None] * cache_stride_slot + rd[None, :]
    )
    q = tl.load(Q + q_offsets).to(tl.float32)
    k = tl.load(K_CACHE + cache_offsets, mask=visible[:, None], other=0.0).to(tl.float32)
    v = tl.load(V_CACHE + cache_offsets, mask=visible[:, None], other=0.0).to(tl.float32)
    scores = tl.sum(q[:, None, :] * k[None, :, :], 2) * scale
    scores = tl.where(visible[None, :], scores, float("-inf"))
    weights = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    attended = tl.sum(weights[:, :, None] * v[None, :, :], 1) / tl.sum(weights, 1)[:, None]
    tl.store(OUT + q_offsets, attended.to(tl.bfloat16))
    if RELEASE_EARLY:
        pass
    else:
        gdc_launch_dependents()


@triton.jit
def mma_attention_kernel(
    Q,
    K_CACHE,
    V_CACHE,
    cache_stride_row,
    cache_stride_slot,
    cache_stride_head,
    OUT,
    end,
    scale,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    RELEASE_EARLY: tl.constexpr,
):
    """One program per row group and kv head: its tokens' q heads of that kv head as the MMA rows (padded to 16), the
    scores and the weighted values on the tensor cores, the softmax in fp32, its unnormalized weights rounded to bf16
    for the second product and the sum divided after."""
    group_index = tl.program_id(0)
    kv_head = tl.program_id(1)
    GROUP: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    gdc_wait()
    if RELEASE_EARLY:
        gdc_launch_dependents()
    else:
        pass
    rq = tl.arange(0, 16)
    rs = tl.arange(0, BLOCK_S)
    rd = tl.arange(0, HEAD_DIM)
    token = rq // GROUP
    is_query = rq < T * GROUP
    q_offsets = (
        (group_index * T + token)[:, None] * (NUM_Q_HEADS * HEAD_DIM)
        + (kv_head * GROUP + rq % GROUP)[:, None] * HEAD_DIM
        + rd[None, :]
    )
    cache_offsets = (
        group_index * cache_stride_row + kv_head * cache_stride_head + rs[:, None] * cache_stride_slot + rd[None, :]
    )
    q = tl.load(Q + q_offsets, mask=is_query[:, None], other=0.0)
    k = tl.load(K_CACHE + cache_offsets, mask=(rs < end)[:, None], other=0.0)
    v = tl.load(V_CACHE + cache_offsets, mask=(rs < end)[:, None], other=0.0)
    scores = tl.dot(q, tl.trans(k)) * scale
    visible = rs[None, :] < end - T + token[:, None] + 1
    scores = tl.where(visible, scores, float("-inf"))
    weights = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    attended = tl.dot(weights.to(tl.bfloat16), v) / tl.sum(weights, 1)[:, None]
    tl.store(OUT + q_offsets, attended.to(tl.bfloat16), mask=is_query[:, None])
    if RELEASE_EARLY:
        pass
    else:
        gdc_launch_dependents()


def mma_variant(release: str):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    def decode_attention(*, q, tokens_per_row, k_cache, v_cache, end, attention_output, shape) -> None:
        programs = q.shape[0] // tokens_per_row * shape.num_kv_heads
        release_early = release == "early" or (release == "one wave" and programs <= sm_count)
        mma_attention_kernel[(q.shape[0] // tokens_per_row, shape.num_kv_heads)](
            q, k_cache, v_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), attention_output, end,
            shape.head_dim**-0.5, T=tokens_per_row, HEAD_DIM=shape.head_dim, NUM_Q_HEADS=shape.num_q_heads,
            NUM_KV_HEADS=shape.num_kv_heads, BLOCK_S=max(16, triton.next_power_of_2(k_cache.shape[1])),
            RELEASE_EARLY=release_early, num_warps=predictor_layers.NUM_WARPS, launch_pdl=True,
        )

    return decode_attention


def attention_variant(grouped: bool, release_early: bool):
    def decode_attention(*, q, tokens_per_row, k_cache, v_cache, end, attention_output, shape) -> None:
        heads = shape.num_kv_heads if grouped else shape.num_q_heads
        grouped_attention_kernel[(q.shape[0], heads)](
            q, k_cache, v_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), attention_output, end,
            shape.head_dim**-0.5, T=tokens_per_row, HEAD_DIM=shape.head_dim, NUM_Q_HEADS=shape.num_q_heads,
            NUM_KV_HEADS=shape.num_kv_heads, BLOCK_S=triton.next_power_of_2(k_cache.shape[1]), GROUPED=grouped,
            RELEASE_EARLY=release_early, num_warps=predictor_layers.NUM_WARPS, launch_pdl=True,
        )

    return decode_attention


def plain_o_proj_add(*, attention_output, tokens_per_row, weight, residual_in, residual_out, partials, counters, shape):
    rows = residual_out.shape[0]
    predictor_layers.gemv_add_kernel[(shape.hidden_size // predictor_layers.BLOCK_N, shape.split_hidden)](
        attention_output, attention_output.stride(0), attention_output.stride(2), attention_output.stride(1), rows,
        weight, residual_in, residual_in.stride(0), residual_out, partials, counters, T=tokens_per_row,
        D=shape.head_dim, K=shape.num_q_heads * shape.head_dim, N=shape.hidden_size, SPLIT=shape.split_hidden,
        BLOCK_M=predictor_layers.block_rows(rows), BLOCK_N=predictor_layers.BLOCK_N, BLOCK_K=predictor_layers.BLOCK_K,
        USE_GDC=True, num_warps=predictor_layers.NUM_WARPS, num_stages=predictor_layers.NUM_STAGES,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    pdl_o_proj_add = predictor_layers.o_proj_add
    variants = {
        "mma released early": (mma_variant("early"), pdl_o_proj_add),
        "mma released at end": (mma_variant("end"), pdl_o_proj_add),
        "mma released early in one wave": (mma_variant("one wave"), pdl_o_proj_add),
        "mma, o_proj plain": (mma_variant("end"), plain_o_proj_add),
        "p6": (predictor_layers.decode_attention, pdl_o_proj_add),
        "p6 kernel, o_proj plain": (attention_variant(False, False), plain_o_proj_add),
    }
    with torch.inference_mode():
        for name, (attention, o_proj) in variants.items():
            if args.variant is not None and name != args.variant:
                continue
            else:
                pass
            predictor_layers.decode_attention = attention
            predictor_layers.o_proj_add = o_proj
            print(f"== {name}", flush=True)
            for batch_size in args.batch:
                torch.manual_seed(0)
                bench(batch_size, 50, device)


if __name__ == "__main__":
    main()
