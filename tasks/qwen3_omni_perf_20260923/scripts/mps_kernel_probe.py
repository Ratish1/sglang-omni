"""Single-process timing of the kernel families that slow down under MPS at c1: a cuBLAS
GEMM at the thinker's decode shape (1 x 2048 -> 5120), a chain of tiny elementwise adds,
and the sglang FA3 decode call at a thinker-like shape, each captured in one CUDA graph
of many launches and timed per launch. Run once with CUDA_MPS_PIPE_DIRECTORY pointing at
a running MPS control daemon and once without, alone on the card, to tell whether MPS by
itself changes these kernels. Prints the SM count the process sees.

usage: python mps_kernel_probe.py
"""

from __future__ import annotations

import os

import torch

LAUNCHES = 200
REPLAYS = 20


def per_launch_us(body) -> float:
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAUNCHES):
            body()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * LAUNCHES)


def main() -> None:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"mps pipe {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'none')} sms {props.multi_processor_count}"
    )
    x = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(5120, 2048, device=device, dtype=torch.bfloat16)
    out = torch.empty(1, 5120, device=device, dtype=torch.bfloat16)
    print(
        f"cublas 1x2048x5120 us {per_launch_us(lambda: torch.matmul(x, weight.t(), out=out)):.2f}"
    )
    small = torch.randn(2048, device=device, dtype=torch.bfloat16)
    print(f"elementwise add 2048 us {per_launch_us(lambda: small.add_(1.0)):.2f}")
    from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache

    pages, heads_q, heads_kv, dim = 4096, 32, 4, 128
    k_cache = torch.randn(pages, 1, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, 1, heads_q, dim, device=device, dtype=torch.bfloat16)
    page_table = torch.arange(pages, device=device, dtype=torch.int32)[None, :512]
    cache_seqlens = torch.tensor([512], device=device, dtype=torch.int32)
    print(
        "fa3 decode 1 x 512 us "
        f"{per_launch_us(lambda: flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens, causal=True)):.2f}"
    )
    print(
        f"sglang fused_experts M=1, a PDL chain of seven kernels, us {moe_chain_us():.2f}"
    )


def moe_chain_us() -> float:
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
    return per_launch_us(lambda: fused_experts(x, w13, w2, topk_output, config))


if __name__ == "__main__":
    main()
