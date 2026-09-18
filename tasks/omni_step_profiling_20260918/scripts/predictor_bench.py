"""Predictor micro-benchmarks for slices P1 and P2, on the checkpoint's shapes.

P1a: the predictor's GEMM chain as the replay runs it (16 one-token passes over 5
layers, 15 heads, 17 input projections) against the folded chain (one two-token pass,
then 14 one-token passes, 16 projections), each captured in a CUDA graph with distinct
weights per layer so every pass streams from DRAM. The full chain at bs 16 should land
near the traced 4.92 ms of GEMM time per replay.
P2a: the fused seeded top-k sampler at its production call, and tl.topk latency by
input width, to size a split selection.

usage: python predictor_bench.py --batch 1 16 --reps 50
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_logits_with_seed_top_k_top_p,
)

TALKER_HIDDEN = 2048
HIDDEN = 1024
INTERMEDIATE = 3072
Q_WIDTH = 16 * 128
KV_WIDTH = 8 * 128
LAYERS = 5
GROUPS = 16
VOCAB = 2048


def make_weights(device: torch.device) -> dict:
    def w(out_features: int, in_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16) * 0.02

    return {
        "layers": [
            {
                "qkv": w(Q_WIDTH + 2 * KV_WIDTH, HIDDEN),
                "o": w(HIDDEN, Q_WIDTH),
                "gate_up": w(2 * INTERMEDIATE, HIDDEN),
                "down": w(HIDDEN, INTERMEDIATE),
            }
            for _ in range(LAYERS)
        ],
        "heads": [w(VOCAB, HIDDEN) for _ in range(GROUPS - 1)],
        "projection": w(HIDDEN, TALKER_HIDDEN),
    }


def one_pass(weights: dict, x: torch.Tensor) -> torch.Tensor:
    for layer in weights["layers"]:
        F.linear(x, layer["qkv"])
        x = x + F.linear(torch.empty(x.shape[0], Q_WIDTH, device=x.device, dtype=x.dtype), layer["o"])
        up = F.linear(x, layer["gate_up"])
        x = x + F.linear(up[:, :INTERMEDIATE], layer["down"])
    return x


def chain(weights: dict, batch: int, folded: bool) -> None:
    device = weights["projection"].device
    talker = torch.randn(batch, TALKER_HIDDEN, device=device, dtype=torch.bfloat16)
    embed = torch.randn(batch, TALKER_HIDDEN, device=device, dtype=torch.bfloat16)
    if folded:
        pair = F.linear(torch.cat((talker, embed)), weights["projection"])
        hidden = one_pass(weights, pair)[batch:]
    else:
        one_pass(weights, F.linear(talker, weights["projection"]))
        hidden = one_pass(weights, F.linear(embed, weights["projection"]))
    for group in range(GROUPS - 1):
        F.linear(hidden, weights["heads"][group])
        step = F.linear(embed, weights["projection"])
        if group < GROUPS - 2:
            hidden = one_pass(weights, step)


def graph_ms(fn, reps: int) -> tuple[float, float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def kernel_names(fn) -> dict[str, tuple[int, float]]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    table: dict[str, tuple[int, float]] = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            count, total = table.get(event.name, (0, 0.0))
            table[event.name] = (count + 1, total + event.time_range.elapsed_us())
    return table


@triton.jit
def topk_latency_kernel(keys, out, width: tl.constexpr, block_k: tl.constexpr):
    row = tl.program_id(0)
    values = tl.load(keys + row * width + tl.arange(0, width))
    top = tl.topk(values, k=block_k)
    tl.store(out + row * block_k + tl.arange(0, block_k), top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    weights = make_weights(device)
    layer_bytes = sum(t.numel() * 2 for layer in weights["layers"] for t in layer.values())
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    print(f"predictor layer weights per pass {layer_bytes / 1e6:.1f} MB")

    print("\n== P1a GEMM chain, CUDA graph, median / min ms per replay")
    for batch in args.batch:
        for folded in (False, True):
            median, best = graph_ms(lambda: chain(weights, batch, folded), args.reps)
            label = "folded (one 2-token pass + 14)" if folded else "current (16 one-token passes)"
            print(f"bs {batch:>3} {label:34s} {median:8.3f} {best:8.3f}")
        for folded in (False, True):
            table = kernel_names(lambda: chain(weights, batch, folded))
            top = sorted(table.items(), key=lambda item: item[1][1], reverse=True)[:4]
            print(f"  bs {batch} {'folded' if folded else 'current'} top kernels:")
            for name, (count, total_us) in top:
                print(f"    {count:5d} x {total_us / count:8.2f} us  {name[:90]}")

    print("\n== P2a fused seeded sampler, 15 calls in one graph, us per call")
    for batch in args.batch:
        logits = torch.randn(batch, VOCAB, device=device, dtype=torch.bfloat16)
        temperatures = torch.full((batch,), 0.9, device=device, dtype=torch.float32)
        top_ks = torch.full((batch,), 50, device=device, dtype=torch.long)
        top_ps = torch.ones(batch, device=device, dtype=torch.float32)
        seeds = torch.arange(batch, device=device, dtype=torch.long)
        positions = torch.arange(batch, device=device, dtype=torch.long)

        def sample() -> None:
            for _ in range(GROUPS - 1):
                assert sample_from_logits_with_seed_top_k_top_p(
                    logits, temperatures, top_ks, top_ps, seeds, positions,
                    max_top_k=64, has_top_p=False,
                ) is not None

        median, best = graph_ms(sample, args.reps)
        print(f"bs {batch:>3} {1e3 * median / (GROUPS - 1):8.2f} {1e3 * best / (GROUPS - 1):8.2f}")

    print("\n== P2a tl.topk latency, one program per row, k=64, us per launch (median)")
    for batch in args.batch:
        for width in (256, 512, 1024, 2048):
            keys = torch.randint(0, 2**62, (batch, width), device=device, dtype=torch.int64).view(torch.uint64)
            out = torch.empty(batch, 64, device=device, dtype=torch.uint64)

            def run() -> None:
                for _ in range(GROUPS - 1):
                    topk_latency_kernel[(batch,)](keys, out, width, 64, num_warps=8)

            median, _ = graph_ms(run, args.reps)
            print(f"bs {batch:>3} width {width:5d} {1e3 * median / (GROUPS - 1):8.2f}")


if __name__ == "__main__":
    main()
