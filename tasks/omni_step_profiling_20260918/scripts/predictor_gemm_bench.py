"""The code predictor's six GEMM shapes against SGLang's small-M kernels, in CUDA graphs.

Shapes from the Qwen3-TTS 1.7B checkpoint: hidden 1024, q 16 x 128, kv 8 x 128, MLP
3072, a 2048 wide codec head, and the 2048 to 1024 input projection. For each M, each
kernel is replayed from a CUDA graph of 50 back to back launches (so launch overhead is
the graph's, as in serving) and timed with events; bandwidth is weight bytes over time,
against the card's HBM peak. Output difference against cuBLAS is printed, since any
swap changes the predictor's bits.

usage: python predictor_gemm_bench.py [--peak-tbs 3.35]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

SHAPES = {
    "qkv": (1024, 4096),
    "o_proj": (2048, 1024),
    "gate_up": (1024, 6144),
    "down": (3072, 1024),
    "head": (1024, 2048),
    "input_proj": (2048, 1024),
}
REPEAT = 50


def graph_time_us(fn) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPEAT):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    times = []
    for _ in range(20):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / REPEAT)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-tbs", type=float, default=3.35)
    args = parser.parse_args()
    from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
        hopper_bf16_gemv,
        use_hopper_bf16_gemv,
    )
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

    torch.manual_seed(0)
    device = torch.device("cuda")
    print(
        f"{'shape':<11}{'M':>4}{'floor us':>9}{'cuBLAS us':>10}{'BW':>6}{'gemv us':>9}{'BW':>6}{'tiny us':>9}{'BW':>6}  max abs vs cuBLAS"
    )
    for name, (k, n) in SHAPES.items():
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
        floor_us = n * k * 2 / (args.peak_tbs * 1e12) * 1e6
        for m in (1, 2, 4, 8, 16, 32, 64):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            reference = F.linear(x, w)
            cublas = graph_time_us(lambda: F.linear(x, w))
            cells = [f"{cublas:>10.2f}{100 * floor_us / cublas:>5.0f}%"]
            diffs = []
            if use_hopper_bf16_gemv(m, n, k):
                gemv = graph_time_us(lambda: hopper_bf16_gemv(x, w))
                cells.append(f"{gemv:>9.2f}{100 * floor_us / gemv:>5.0f}%")
                diffs.append(
                    f"gemv {float((hopper_bf16_gemv(x, w) - reference).abs().max()):.2e}"
                )
            else:
                cells.append(f"{'-':>9}{'':>6}")
            if m <= 16 and can_use_tiny_gemm(n, k, 16):
                out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
                tiny = graph_time_us(lambda: tiny_gemm_bf16(x, w, out, max_m=16))
                cells.append(f"{tiny:>9.2f}{100 * floor_us / tiny:>5.0f}%")
                diffs.append(
                    f"tiny {float((tiny_gemm_bf16(x, w, max_m=16) - reference).abs().max()):.2e}"
                )
            else:
                cells.append(f"{'-':>9}{'':>6}")
            print(
                f"{name:<11}{m:>4}{floor_us:>9.2f}"
                + "".join(cells)
                + "  "
                + " ".join(diffs)
            )


if __name__ == "__main__":
    main()
