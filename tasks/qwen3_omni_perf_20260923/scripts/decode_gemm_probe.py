"""The thinker's batch-1 decode GEMMs (qkv 2048 -> 5120, o_proj 4096 -> 2048, router gate
2048 -> 128) through cuBLAS (F.linear, the path sglang takes) and through sglang's Hopper
bf16 GEMV, each over 48 distinct weights in one CUDA graph as the 48 layers would read them,
so the weights stream from DRAM. Prints per-call time, the kernels each path launches, and
how many output elements differ between the two paths.

usage: [CUBLASLT_LOG_LEVEL=5] python decode_gemm_probe.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

LAYERS = 48
REPLAYS = 20
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
        f"{row.key[:48]} x{row.count // LAYERS} {row.self_device_time_total / row.count:.2f} us"
        for row in sorted(rows, key=lambda row: -row.self_device_time_total)
    )


def main() -> None:
    from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
        hopper_bf16_gemv,
        use_hopper_bf16_gemv,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    for name, (k, n) in SHAPES.items():
        weights = [
            (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
            for _ in range(LAYERS)
        ]
        x = torch.randn(1, k, device=device).to(torch.bfloat16)
        cublas_out = [torch.empty(1, n, device=device, dtype=torch.bfloat16)]
        gemv_out = [torch.empty(1, n, device=device, dtype=torch.bfloat16)]

        def cublas() -> None:
            for w in weights:
                cublas_out[0] = F.linear(x, w)

        def gemv() -> None:
            for w in weights:
                gemv_out[0] = hopper_bf16_gemv(x, w)

        graph_cublas = capture(cublas)
        print(
            f"{name} ({k} -> {n}) cublas us {per_call_us(graph_cublas):.2f}  [{kernel_names(graph_cublas)}]"
        )
        if use_hopper_bf16_gemv(1, n, k):
            graph_gemv = capture(gemv)
            print(
                f"{name} ({k} -> {n}) gemv   us {per_call_us(graph_gemv):.2f}  [{kernel_names(graph_gemv)}]"
            )
            # eager, since a split-K graph replayed after later captures may read a stale
            # cuBLAS workspace
            differ = largest = 0
            for w in weights:
                reference, candidate = F.linear(x, w), hopper_bf16_gemv(x, w)
                differ += (reference != candidate).sum().item()
                largest = max(
                    largest,
                    (reference.float() - candidate.float()).abs().max().item(),
                )
            print(
                f"{name} over {LAYERS} layers: {differ} of {n * LAYERS} outputs differ, max abs difference {largest:.3e}"
            )
        else:
            print(f"{name} ({k} -> {n}) not eligible for the Hopper GEMV")


if __name__ == "__main__":
    main()
