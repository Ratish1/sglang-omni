"""The code predictor's GEMM shapes against SGLang's small-M kernels and #2413's split
GEMV, each read from HBM as in serving.

Shapes from the Qwen3-TTS 1.7B checkpoint: hidden 1024, q 16 x 128, kv 8 x 128, MLP
3072, a 2048 wide codec head, and the 2048 to 1024 input projection. Serving streams
about 157 MB of predictor weights per pass, so no weight is in L2 when its GEMM runs;
here each kernel cycles through enough copies of its weight to exceed L2 (256 MB), all
captured in one CUDA graph and timed with events. Bandwidth is weight bytes over time
against the card's HBM peak. The served per-call times at bs 16 (predictor census) are
the known answer for the cuBLAS column. Output difference against cuBLAS is printed,
since any swap changes the predictor's bits.

usage: python predictor_gemm_bench.py [--peak-tbs 3.35] [--kernels-2413 FILE]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys

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
RESIDUAL_SHAPES = ("o_proj", "down")
POOL_BYTES = 256 << 20


def graph_time_us(fns) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in fns[:3]:
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in fns:
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(20):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / len(fns))
    times.sort()
    return times[len(times) // 2]


def load_2413(path: str):
    spec = importlib.util.spec_from_file_location("predictor_kernels_2413", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gemv_add_2413(pk, x, w, residual, out, partials, counters, split):
    rows, k = x.shape
    pk.gemv_add_kernel[(w.shape[0] // pk.BLOCK_N, split)](
        x,
        x.stride(0),
        x.stride(0),
        pk.BLOCK_K,
        rows,
        w,
        residual,
        residual.stride(0),
        out,
        partials,
        counters,
        T=1,
        D=pk.BLOCK_K,
        K=k,
        N=w.shape[0],
        SPLIT=split,
        BLOCK_M=pk.block_rows(rows),
        BLOCK_N=pk.BLOCK_N,
        BLOCK_K=pk.BLOCK_K,
        num_warps=pk.NUM_WARPS,
        num_stages=pk.NUM_STAGES,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-tbs", type=float, default=3.35)
    parser.add_argument("--kernels-2413")
    args = parser.parse_args()
    from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
        hopper_bf16_gemv,
        use_hopper_bf16_gemv,
    )
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

    pk = load_2413(args.kernels_2413) if args.kernels_2413 else None
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(0)
    device = torch.device("cuda")
    print(
        "us per call, weights from HBM; BW = weight bytes over time against peak; "
        "diff = max abs against cuBLAS"
    )
    for name, (k, n) in SHAPES.items():
        weight_bytes = n * k * 2
        copies = max(8, math.ceil(POOL_BYTES / weight_bytes))
        weights = [
            torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
            for _ in range(copies)
        ]
        floor_us = weight_bytes / (args.peak_tbs * 1e12) * 1e6
        print(f"\n{name} K {k} N {n}: {copies} weight copies, floor {floor_us:.2f} us")
        for m in (1, 2, 4, 8, 16, 32):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            residual = torch.randn(m, n, device=device, dtype=torch.bfloat16)
            cells = []

            def add(label, time_us, diff=None):
                text = f"{label} {time_us:6.2f} ({100 * floor_us / time_us:3.0f}%)"
                if diff is not None:
                    text += f" diff {diff:.1e}"
                else:
                    pass
                cells.append(text)

            reference = F.linear(x, weights[0])
            add("cuBLAS", graph_time_us([lambda w=w: F.linear(x, w) for w in weights]))
            if name in RESIDUAL_SHAPES:
                add(
                    "addmm",
                    graph_time_us(
                        [lambda w=w: torch.addmm(residual, x, w.t()) for w in weights]
                    ),
                )
            else:
                pass
            if use_hopper_bf16_gemv(m, n, k):
                diff = float((hopper_bf16_gemv(x, weights[0]) - reference).abs().max())
                add(
                    "gemv",
                    graph_time_us(
                        [lambda w=w: hopper_bf16_gemv(x, w) for w in weights]
                    ),
                    diff,
                )
            else:
                pass
            if m <= 16 and can_use_tiny_gemm(n, k, 16):
                out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
                diff = float(
                    (tiny_gemm_bf16(x, weights[0], max_m=16) - reference).abs().max()
                )
                add(
                    "tiny",
                    graph_time_us(
                        [
                            lambda w=w: tiny_gemm_bf16(x, w, out, max_m=16)
                            for w in weights
                        ]
                    ),
                    diff,
                )
            else:
                pass
            if pk is not None and name in RESIDUAL_SHAPES:
                split = pk.split_count(n // pk.BLOCK_N, k // pk.BLOCK_K, sms)
                block_m = pk.block_rows(m)
                partials = torch.zeros(block_m * split * n, device=device)
                counters = torch.zeros(
                    n // pk.BLOCK_N, device=device, dtype=torch.int32
                )
                out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
                gemv_add_2413(
                    pk, x, weights[0], residual, out, partials, counters, split
                )
                expected = (reference.float() + residual.float()).to(torch.bfloat16)
                diff = float((out.float() - expected.float()).abs().max())
                add(
                    f"2413 split {split}",
                    graph_time_us(
                        [
                            lambda w=w: gemv_add_2413(
                                pk, x, w, residual, out, partials, counters, split
                            )
                            for w in weights
                        ]
                    ),
                    diff,
                )
            else:
                pass
            print(f"  M {m:>2}  " + " | ".join(cells))
        del weights
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
