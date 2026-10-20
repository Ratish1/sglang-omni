"""Candidates for the Qwen3-TTS talker's and predictor's dense GEMMs at 1 to 64 rows, against their weight floors.

usage: python3 skinny_gemm_bench.py [--rows 1 16 64] [--replays 30]

Per shape and row count: one CUDA graph of 28 calls, each on its own weight (more than L2 holds, so every call streams
its weight from HBM as served), per call time (median replay over 28) and its share of the floor at 3.35 TB/s, for
cuBLAS (F.linear), SGLang's Hopper GEMV (one row only), and a Triton split-K GEMM (fp32 partials summed in split order
by a second launch, so deterministic) over a small grid of tile and split choices, the best of which is printed with
its relative distance to an fp32 product.
"""

import argparse
import itertools
import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv

LAYERS = 28
HBM_BYTES_PER_SECOND = 3.35e12
SHAPES = {
    "talker qkv": (4096, 2048),
    "talker o": (2048, 2048),
    "talker gate_up": (12288, 2048),
    "talker down": (2048, 6144),
    "predictor qkv": (4096, 1024),
    "predictor o": (1024, 2048),
    "predictor gate_up": (6144, 1024),
    "predictor down": (1024, 3072),
}


@triton.jit
def split_k_kernel(
    X,
    W,
    PARTIALS,
    OUT,
    rows,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(pid_s * (K // SPLIT), (pid_s + 1) * (K // SPLIT), BLOCK_K):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=(rm < rows)[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
    if SPLIT == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm < rows)[:, None])
    else:
        tl.store(PARTIALS + (pid_s * BLOCK_M + rm)[:, None] * N + rn[None, :], acc)


@triton.jit
def split_sum_kernel(PARTIALS, OUT, rows, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // N
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for split in tl.static_range(SPLIT):
        total += tl.load(PARTIALS + split * BLOCK_M * N + offsets, mask=row < rows, other=0.0)
    tl.store(OUT + offsets, total.to(tl.bfloat16), mask=row < rows)


def split_k_gemm(x: torch.Tensor, w: torch.Tensor, partials: torch.Tensor, out: torch.Tensor, config: tuple) -> None:
    block_n, block_k, split, warps, stages = config
    rows, k = x.shape
    n = w.shape[0]
    block_m = max(16, triton.next_power_of_2(rows))
    split_k_kernel[(n // block_n, split)](
        x, w, partials, out, rows, N=n, K=k, SPLIT=split, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=stages,
    )
    if split == 1:
        pass
    else:
        split_sum_kernel[(triton.cdiv(rows * n, 1024),)](partials, out, rows, N=n, SPLIT=split, BLOCK_M=block_m, BLOCK=1024)


def graph_us(calls) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            calls()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        calls()
    times = []
    for _ in range(30):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / LAYERS)
    return statistics.median(times)


def configs(n: int, k: int, sm_count: int) -> list[tuple]:
    grid = []
    for block_n, block_k, warps, stages in itertools.product((32, 64, 128), (128, 256), (4,), (4,)):
        if n % block_n or k % block_k:
            continue
        else:
            pass
        for split in (1, 2, 4, 8, 16):
            programs = n // block_n * split
            if (k // split) % block_k or programs > 4 * sm_count:
                continue
            else:
                pass
            grid.append((block_n, block_k, split, warps, stages))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64])
    args = parser.parse_args()
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    torch.manual_seed(0)
    with torch.inference_mode():
        for name, (n, k) in SHAPES.items():
            weights = [(torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16) for _ in range(LAYERS)]
            floor_us = n * k * 2 / HBM_BYTES_PER_SECOND * 1e6
            for rows in args.rows:
                x = torch.randn(rows, k, device=device).to(torch.bfloat16)
                reference = x.float() @ weights[0].float().t()
                results = [("cublas", graph_us(lambda: [F.linear(x, w) for w in weights]))]
                if rows == 1:
                    results.append(("sglang gemv", graph_us(lambda: [hopper_bf16_gemv(x, w) for w in weights])))
                else:
                    pass
                block_m = max(16, triton.next_power_of_2(rows))
                partials = torch.empty(16 * block_m * n, device=device)
                out = torch.empty(rows, n, device=device, dtype=torch.bfloat16)
                best = None
                for config in configs(n, k, sm_count):
                    us = graph_us(lambda: [split_k_gemm(x, w, partials, out, config) for w in weights])
                    if best is None or us < best[1]:
                        best = (config, us)
                    else:
                        pass
                split_k_gemm(x, weights[0], partials, out, best[0])
                error = ((out.float() - reference).norm() / reference.norm()).item()
                cublas_error = ((F.linear(x, weights[0]).float() - reference).norm() / reference.norm()).item()
                results.append((f"triton split-k {best[0]} (error {error:.2e}, cublas {cublas_error:.2e})", best[1]))
                line = "; ".join(f"{label} {us:.2f} us ({floor_us / us:.0%})" for label, us in results)
                print(f"{name} {n}x{k} rows {rows}, floor {floor_us:.2f} us: {line}", flush=True)


if __name__ == "__main__":
    main()
