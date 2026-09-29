"""Single-process timing of the kernel families that slow down under MPS at c1: a cuBLAS
GEMM at the thinker's decode shape (1 x 2048 -> 5120), a chain of tiny elementwise adds,
the sglang FA3 decode call at a thinker-like shape, and the sglang fused_experts call at
M=1, each captured alone in one CUDA graph of many launches and timed per launch, then
all four interleaved in one graph as a decode step interleaves them, timed per round with
each kernel's mean duration inside the mix from the torch profiler. Run once with
CUDA_MPS_PIPE_DIRECTORY pointing at a running MPS control daemon and once without to tell
whether MPS by itself changes these kernels. PROBE_FOOTPRINT_GB holds that much device
memory in the process first, as a stage process holds its weights and KV pool. Prints the
SM count the process sees, and the SMs a spinning kernel of the process lands on.

usage: [PROBE_FOOTPRINT_GB=100] python mps_kernel_probe.py
"""

from __future__ import annotations

import os
import re

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile
from triton.language.extra.cuda import (
    gdc_launch_dependents,
    gdc_wait,
    globaltimer,
    smid,
)

LAUNCHES = 200
REPLAYS = 20
CUTE_CONSTANT = re.compile(r"cute::C<(?:\(int\))?(\d+)>")


def capture(body, launches: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            body()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def per_launch_us(graph: torch.cuda.CUDAGraph, launches: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * launches)


def kernel_means(graph: torch.cuda.CUDAGraph, launches: int) -> str:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    rows = [
        row
        for row in prof.key_averages()
        if row.self_device_time_total > 0 and row.count >= 3 * launches
    ]
    rows.sort(key=lambda row: -row.self_device_time_total)
    lines = []
    for row in rows[:10]:
        # an FA3 forward name lists its cluster shape, then its tile shape
        constants = CUTE_CONSTANT.findall(row.key)
        tile = (
            f" tile {'x'.join(constants[3:6])}"
            if "FlashAttnFwdSm90" in row.key and len(constants) >= 6
            else ""
        )
        lines.append(
            f"    {row.count // (3 * launches)}/round mean us {row.self_device_time_total / row.count:7.2f}  {row.key[:60]}{tile}"
        )
    return "\n".join(lines)


def main() -> None:
    device = torch.device("cuda")
    footprint_gb = float(os.environ.get("PROBE_FOOTPRINT_GB", "0"))
    held = torch.empty(int(footprint_gb * 2**30), dtype=torch.uint8, device=device)
    props = torch.cuda.get_device_properties(device)
    print(
        f"mps pipe {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'none')} sms {props.multi_processor_count} "
        f"held GB {held.numel() / 2**30:.0f}"
    )
    x = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(5120, 2048, device=device, dtype=torch.bfloat16)
    out = torch.empty(1, 5120, device=device, dtype=torch.bfloat16)
    small = torch.randn(2048, device=device, dtype=torch.bfloat16)
    from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache

    pages, heads_q, heads_kv, dim = 4096, 32, 4, 128
    k_cache = torch.randn(pages, 1, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, 1, heads_q, dim, device=device, dtype=torch.bfloat16)
    page_table = torch.arange(pages, device=device, dtype=torch.int32)[None, :512]
    cache_seqlens = torch.tensor([512], device=device, dtype=torch.int32)
    decode_attention = sglang_decode_attention_body(device)
    bodies = {
        "cublas 1x2048x5120": lambda: torch.matmul(x, weight.t(), out=out),
        "elementwise add 2048": lambda: small.add_(1.0),
        "fa3 decode 1 x 512": lambda: flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            causal=True,
        ),
        "sglang fused_experts M=1, a PDL chain of seven kernels": moe_body(),
        "fa3 as the sglang decode graph calls it, split kv and combine": decode_attention,
    }
    for name, body in bodies.items():
        graph = capture(body, LAUNCHES)
        print(f"{name} us {per_launch_us(graph, LAUNCHES):.2f}")
        print(kernel_means(graph, LAUNCHES))

    def mixed() -> None:
        for body in bodies.values():
            body()

    rounds = LAUNCHES // len(bodies)
    graph = capture(mixed, rounds)
    print(f"mixed round of all us {per_launch_us(graph, rounds):.2f}")
    print(kernel_means(graph, rounds))
    # one byte from each of 8192 pages 2 MB apart over 16 GB, against the same gather
    # from contiguous bytes: equal work, only the address translations touched differ
    sweep = torch.empty(16 * 2**30, dtype=torch.uint8, device=device)
    strided = torch.arange(0, sweep.numel(), 2 * 2**20, device=device)
    contiguous = torch.arange(strided.numel(), device=device)
    predecessors = {name: bodies[name] for name in list(bodies)[:4]}
    predecessors["gather over 8192 pages 2 MB apart"] = lambda: sweep.index_select(
        0, strided
    )
    predecessors["gather over contiguous bytes"] = lambda: sweep.index_select(
        0, contiguous
    )
    # the split decode attention right after one other body, pair after pair: which
    # predecessor makes it slow
    for name in predecessors:

        def pair(before=predecessors[name]) -> None:
            before()
            decode_attention()

        rounds = LAUNCHES // 2
        graph = capture(pair, rounds)
        print(f"after {name}: pair us {per_launch_us(graph, rounds):.2f}")
        print(kernel_means(graph, rounds))
    print(sm_reach(props.multi_processor_count))
    print(pdl_early_start(props.multi_processor_count))


@triton.jit
def spin_record(sm_ptr, start_ptr, spin_ns):
    start = globaltimer()
    now = start
    while now - start < spin_ns:
        now = globaltimer()
    tl.store(sm_ptr + tl.program_id(0), smid())
    tl.store(start_ptr + tl.program_id(0), start)


@triton.jit
def pdl_primary(start_ptr, spin_ns):
    start = globaltimer()
    gdc_launch_dependents()
    now = start
    while now - start < spin_ns:
        now = globaltimer()
    tl.store(start_ptr + tl.program_id(0), start)


@triton.jit
def pdl_secondary(start_ptr):
    tl.store(start_ptr + tl.program_id(0), globaltimer())
    gdc_wait()


def pdl_early_start(sms: int) -> str:
    """A primary kernel releases its dependents at once and then spins 10 us; a secondary
    launched with programmatic dependent launch records when its CTAs start. With PDL
    working the secondary starts about 1 us after the primary, without it after 10 us.
    Measured eager and inside a CUDA graph."""
    primary = torch.empty(sms, device="cuda", dtype=torch.int64)
    secondary = torch.empty(sms, device="cuda", dtype=torch.int64)

    def pair() -> None:
        pdl_primary[(sms,)](primary, 10_000, num_warps=4)
        pdl_secondary[(sms,)](secondary, num_warps=4, launch_pdl=True)

    def lead_us() -> float:
        return (secondary.min() - primary.min()).item() / 1e3

    for _ in range(3):
        pair()
    torch.cuda.synchronize()
    eager = lead_us()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pair()
    graph.replay()
    torch.cuda.synchronize()
    return f"pdl: secondary starts after the primary by us, eager {eager:.1f}, graph {lead_us():.1f} (spin 10)"


def sm_reach(sms: int) -> str:
    """Four CTAs per SM spin 20 us each and record their SM id and start time: the SMs a
    kernel of this process can reach, and the spread of CTA start times (one wave or two).
    """
    ctas = 4 * sms
    sm_ids = torch.empty(ctas, device="cuda", dtype=torch.int32)
    starts = torch.empty(ctas, device="cuda", dtype=torch.int64)
    for _ in range(3):
        spin_record[(ctas,)](sm_ids, starts, 20_000, num_warps=4)
    torch.cuda.synchronize()
    spread_us = (starts.max() - starts.min()).item() / 1e3
    return (
        f"sm reach: {ctas} ctas on {torch.unique(sm_ids).numel()} distinct SMs of {sms}, "
        f"start spread us {spread_us:.1f}"
    )


def sglang_decode_attention_body(device: torch.device):
    """One thinker decode attention call as sglang's FA3 backend makes it in the decode
    graph: page size 1 over a 200k token pool, a varlen query of one token, automatic
    split count, and scheduler metadata computed before the call against the thinker's
    32768 context."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

    pool, heads_q, heads_kv, dim, seqlen = 200_000, 32, 4, 128, 300
    max_seqlen_k = 32768
    k_cache = torch.randn(pool, 1, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, heads_q, dim, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    # the graph's page table spans the whole context, which sizes the split count
    page_table = torch.zeros(1, max_seqlen_k, device=device, dtype=torch.int32)
    page_table[0, :seqlen] = torch.arange(
        150_000, 150_000 + seqlen, device=device, dtype=torch.int32
    )
    cache_seqlens = torch.tensor([seqlen], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    scheduler_metadata = get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        num_heads=heads_q,
        num_heads_k=heads_kv,
        headdim=dim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=1,
        causal=True,
        num_splits=0,
    )
    return lambda: flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=dim**-0.5,
        causal=True,
        num_splits=0,
        out=out,
        scheduler_metadata=scheduler_metadata,
    )


def moe_body():
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29652")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    device = torch.device("cuda")
    w13 = (torch.randn(128, 1536, 2048, device=device) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(128, 2048, 768, device=device) * 0.02).to(torch.bfloat16)
    x = torch.randn(1, 2048, device=device).to(torch.bfloat16)
    logits = torch.randn(1, 128, device=device).to(torch.bfloat16)
    top, ids = torch.topk(logits.float(), 8, dim=-1)
    topk_output = StandardTopKOutput(
        topk_weights=torch.softmax(top, dim=-1),
        topk_ids=ids.to(torch.int32),
        router_logits=logits,
    )
    config = MoeRunnerConfig(
        num_experts=128,
        top_k=8,
        hidden_size=2048,
        intermediate_size_per_partition=768,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
    )
    return lambda: fused_experts(x, w13, w2, topk_output, config)


if __name__ == "__main__":
    main()
