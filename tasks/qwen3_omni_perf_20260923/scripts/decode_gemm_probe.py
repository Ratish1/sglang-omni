"""The thinker's decode GEMMs (qkv 2048 -> 5120, o_proj 4096 -> 2048, router gate
2048 -> 128) at the small graph buckets through cuBLAS (F.linear, the path sglang takes),
sglang's Hopper bf16 GEMV (batch 1 only) and sglang's tiny GEMM (batch up to 16), each over
48 distinct weights in one CUDA graph as the 48 layers would read them, so the weights stream
from DRAM. Prints per-call time and kernels per path, how many outputs differ from cuBLAS,
and whether the tiny GEMM gives row 0 the same bits at every batch size.

usage: python decode_gemm_probe.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

LAYERS = 48
REPLAYS = 20
BATCHES = (1, 2, 4, 8)
SHAPES = {"qkv": (2048, 5120), "o_proj": (4096, 2048), "gate": (2048, 128)}


def capture(body) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def per_call_us(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * LAYERS)


def kernel_names(graph: torch.cuda.CUDAGraph) -> str:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    rows = [row for row in prof.key_averages() if row.self_device_time_total > 0]
    return ", ".join(
        f"{row.key[:40]} x{row.count // LAYERS} {row.self_device_time_total / row.count:.2f} us"
        for row in sorted(rows, key=lambda row: -row.self_device_time_total)
    )


def timed(label: str, gemm, x: torch.Tensor, weights: list[torch.Tensor]) -> None:
    def body() -> None:
        for w in weights:
            gemm(x, w)

    graph = capture(body)
    print(f"    {label:6s} us {per_call_us(graph):6.2f}  [{kernel_names(graph)}]")


def differ_from_cublas(gemm, x: torch.Tensor, weights: list[torch.Tensor]) -> str:
    # eager, since a split-K graph replayed after later captures may read a stale
    # cuBLAS workspace
    differ = largest = 0
    for w in weights:
        reference, candidate = F.linear(x, w), gemm(x, w)
        differ += (reference != candidate).sum().item()
        largest = max(
            largest, (reference.float() - candidate.float()).abs().max().item()
        )
    total = x.shape[0] * weights[0].shape[0] * len(weights)
    return f"{differ} of {total} differ from cuBLAS, max {largest:.1e}"


def main() -> None:
    from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
        hopper_bf16_gemv,
        use_hopper_bf16_gemv,
    )
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

    device = torch.device("cuda")
    torch.manual_seed(0)
    for name, (k, n) in SHAPES.items():
        weights = [
            (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
            for _ in range(LAYERS)
        ]
        rows = torch.randn(max(BATCHES), k, device=device).to(torch.bfloat16)
        print(f"{name} ({k} -> {n}), tiny GEMM eligible {can_use_tiny_gemm(n, k)}")
        for batch in BATCHES:
            x = rows[:batch].contiguous()
            print(f"  batch {batch}")
            timed("cublas", F.linear, x, weights)
            if batch == 1 and use_hopper_bf16_gemv(1, n, k):
                timed("gemv", hopper_bf16_gemv, x, weights)
                print(f"    gemv   {differ_from_cublas(hopper_bf16_gemv, x, weights)}")
            else:
                pass
            if can_use_tiny_gemm(n, k):
                timed("tiny", tiny_gemm_bf16, x, weights)
                print(f"    tiny   {differ_from_cublas(tiny_gemm_bf16, x, weights)}")
            else:
                pass
        if can_use_tiny_gemm(n, k):
            first = [tiny_gemm_bf16(rows[:1].contiguous(), w) for w in weights]
            stable = all(
                torch.equal(tiny_gemm_bf16(rows[:batch].contiguous(), w)[:1], one)
                for batch in BATCHES
                for w, one in zip(weights, first)
            )
            print(f"  tiny GEMM row 0 identical at every batch: {stable}")
        else:
            pass


if __name__ == "__main__":
    main()
