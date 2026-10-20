"""The Qwen3-TTS talker's dense GEMMs at the served row counts, against their weight floors.

usage: python3 talker_gemm_bench.py [--rows 1 8 16 32 64] [--replays 30]

Qwen3-TTS-12Hz-1.7B-Base's talker (hidden 2048, intermediate 6144, 16 q and 8 kv heads of 128, 28 layers) as SGLang's
unquantized linear runs it, F.linear on a bf16 (N, K) weight. Per projection and row count: one CUDA graph of 28 calls,
each on its own layer's weight (2.82 GB over the four projections, far past L2, so every call streams its weight from
HBM as served); the per-call time (median replay over 28), the HBM rate its weight bytes imply, the floor at 3.35 TB/s,
and the kernels cuBLAS picked.
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

HIDDEN = 2048
INTERMEDIATE = 6144
HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 8
LAYERS = 28
HBM_BYTES_PER_SECOND = 3.35e12
PROJECTIONS = {
    "qkv": ((Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, HIDDEN),
    "o": (HIDDEN, Q_HEADS * HEAD_DIM),
    "gate_up": (2 * INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}


def bench(name: str, rows: int, replays: int, device: torch.device) -> None:
    out_features, in_features = PROJECTIONS[name]
    weights = [
        (torch.randn(out_features, in_features, device=device) * 0.02).to(torch.bfloat16) for _ in range(LAYERS)
    ]
    x = torch.randn(rows, in_features, device=device).to(torch.bfloat16)

    def calls() -> None:
        for weight in weights:
            F.linear(x, weight)

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
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / LAYERS)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    kernels: dict[str, list[float]] = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels.setdefault(event.name, []).append(event.time_range.elapsed_us())
        else:
            pass
    per_call_us = statistics.median(times)
    weight_bytes = out_features * in_features * 2
    floor_us = weight_bytes / HBM_BYTES_PER_SECOND * 1e6
    picked = "; ".join(
        f"{kernel[:60]} x{len(values) / LAYERS:.0f} {statistics.mean(values):.2f} us" for kernel, values in kernels.items()
    )
    print(
        f"{name:<8} rows {rows:>3} ({out_features} x {in_features}): {per_call_us:7.2f} us per call, "
        f"{weight_bytes / per_call_us / 1e6:5.2f} TB/s, floor {floor_us:6.2f} us ({floor_us / per_call_us:5.1%}); {picked}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    with torch.inference_mode():
        for rows in args.rows:
            for name in PROJECTIONS:
                bench(name, rows, args.replays, device)
            step_floor_us = sum(n * k * 2 for n, k in PROJECTIONS.values()) * LAYERS / HBM_BYTES_PER_SECOND * 1e6
            print(f"rows {rows}: the four projections' weight floor per talker step {step_floor_us:.0f} us", flush=True)


if __name__ == "__main__":
    main()
