"""G-e1: the decode GEMMs of Qwen3-TTS 1.7B at small M, per backend, against the floor.

A decode GEMM at M rows reads its whole weight and does 2 * M flops per weight element, so
at M = 16 it is far below the card's flops per byte ridge: its floor is weight bytes over
memory bandwidth. This bench times every GEMM shape of the talker and the predictor inside
a CUDA graph (8 distinct weights per shape, median replay) at M = 1 to 32 and prints us per
GEMM, the floor, and the per decode step total at M = 16.

backends, one process each:
  default    what the server runs (cuBLAS through F.linear)
  cublaslt   torch.backends.cuda.preferred_blas_library("cublaslt")
  tunable    PyTorch TunableOp: times the cuBLAS and cuBLASLt algorithms per shape on
             this card and keeps the fastest
  triton     torch.compile max-autotune of F.linear (M = 16 and 32 only)

usage: python gemm_shapes_bench.py --backend default|cublaslt|tunable|triton [--bandwidth-gbs 1008]
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

# (name, in features, out features, calls per decode step at one position)
SHAPES = (
    ("talker qkv", 2048, 4096, 28),
    ("talker o", 2048, 2048, 28),
    ("talker gate_up", 2048, 12288, 28),
    ("talker down", 6144, 2048, 28),
    ("predictor qkv", 1024, 4096, 75),
    ("predictor o", 2048, 1024, 75),
    ("predictor gate_up", 1024, 6144, 75),
    ("predictor down", 3072, 1024, 75),
    ("predictor lm_head", 1024, 2048, 15),
    ("predictor input projection", 2048, 1024, 16),
)
WEIGHTS_PER_SHAPE = 8


def graph_us(fn, inputs, weights):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for weight in weights:
                fn(inputs, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for weight in weights:
            fn(inputs, weight)
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    graph.reset()
    return statistics.median(times) * 1000.0 / len(weights)


def kernel_name(fn, inputs, weight):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn(inputs, weight)
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]
    return (names[0] if names else "?")[:60]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--bandwidth-gbs", type=float, default=1008.0)
    parser.add_argument("--out", default="gemm_shapes.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    fn = F.linear
    rows_list = (1, 2, 4, 8, 16, 32)
    if args.backend == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")
    elif args.backend == "tunable":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
    elif args.backend == "triton":
        fn = torch.compile(F.linear, mode="max-autotune-no-cudagraphs", dynamic=False)
        rows_list = (16, 32)
    print(
        f"backend {args.backend}, {torch.cuda.get_device_name()}, torch {torch.__version__}"
    )
    results = {}
    with torch.inference_mode():
        for name, fan_in, fan_out, calls in SHAPES:
            weights = [
                torch.randn(fan_out, fan_in, device=device, dtype=torch.bfloat16)
                for _ in range(WEIGHTS_PER_SHAPE)
            ]
            floor_us = fan_in * fan_out * 2 / (args.bandwidth_gbs * 1e9) * 1e6
            line = f"{name:<28} floor {floor_us:6.1f} us |"
            for rows in rows_list:
                inputs = torch.randn(rows, fan_in, device=device, dtype=torch.bfloat16)
                us = graph_us(fn, inputs, weights)
                results[f"{name}|{rows}"] = {
                    "us": us,
                    "floor_us": floor_us,
                    "calls": calls,
                }
                line += f" M{rows} {us:6.1f}"
            inputs = torch.randn(16, fan_in, device=device, dtype=torch.bfloat16)
            print(line + f" | M16 kernel {kernel_name(fn, inputs, weights[0])}")
            del weights
    for rows in (16, 32):
        if f"talker qkv|{rows}" not in results:
            continue
        total = sum(
            v["us"] * v["calls"] for k, v in results.items() if k.endswith(f"|{rows}")
        )
        floor = sum(
            v["floor_us"] * v["calls"]
            for k, v in results.items()
            if k.endswith(f"|{rows}")
        )
        print(
            f"per decode step at M = {rows}: {total / 1000:.2f} ms of GEMM, floor "
            f"{floor / 1000:.2f} ms, {total / floor:.2f}x"
        )
    with open(args.out, "w") as handle:
        json.dump({"backend": args.backend, "results": results}, handle, indent=1)


if __name__ == "__main__":
    main()
