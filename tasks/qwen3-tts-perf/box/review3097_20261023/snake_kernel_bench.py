"""Device time of Qwen3-Omni code2wav's 29 bias SnakeBeta launches of one 10-frame slot decode, as the tree on PYTHONPATH
serves them (fused_bias_snake_beta with its default arithmetic). The shapes follow code2wav's decoder: decoder.0's
activation at 1,536 channels over 40 positions, then per decoder block at (768, 320), (384, 1,600), (192, 6,400) and
(96, 19,200) the unit-0 act1 writing the running sum, three act2, two act1 adding the residual into the sum and the
block's last activation adding the residual. bf16, rows 1 and 8, all 29 launches in one CUDA graph replayed 200 times
per trial, 7 trials; prints the median microseconds per decode.

usage: PYTHONPATH=<tree> python3 snake_kernel_bench.py [--card 2]
"""

import argparse
import json
import statistics

import torch

from sglang_omni.kernels.bias_snake_beta import SnakeBetaCoefficients, fused_bias_snake_beta

LEVELS = ((768, 320), (384, 1600), (192, 6400), (96, 19200))
REPLAYS = 200
TRIALS = 7


def coefficients(channels: int, device: torch.device) -> SnakeBetaCoefficients:
    alpha = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.5
    beta = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.5
    return SnakeBetaCoefficients(alpha_scale=torch.exp(alpha), inverse_beta=1.0 / (torch.exp(beta) + 1e-9))


def launches(rows: int, device: torch.device) -> list:
    def tensor(channels: int, length: int) -> torch.Tensor:
        return torch.randn(rows, length, channels, device=device, dtype=torch.bfloat16)

    calls = []
    first = tensor(1536, 40)
    calls.append((first, torch.randn(1536, device=device, dtype=torch.bfloat16), coefficients(1536, device),
                  torch.empty_like(first), {}))
    for channels, length in LEVELS:
        bias = torch.randn(channels, device=device, dtype=torch.bfloat16)
        scales = coefficients(channels, device)
        conv, out, running = tensor(channels, length), tensor(channels, length), tensor(channels, length)
        calls.append((conv, bias, scales, out, {"sum_out": running}))
        calls.extend((conv, bias, scales, out, {}) for _ in range(3))
        calls.extend((conv, bias, scales, out, {"residual": running, "sum_out": running}) for _ in range(2))
        calls.append((conv, bias, scales, out, {"residual": running}))
    return calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    torch.manual_seed(0)
    result = {}
    for rows in (1, 8):
        calls = launches(rows, device)
        assert len(calls) == 29
        stream = torch.cuda.Stream(device)
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                for conv, bias, scales, out, extra in calls:
                    fused_bias_snake_beta(conv, bias, scales, out, **extra)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
            for conv, bias, scales, out, extra in calls:
                fused_bias_snake_beta(conv, bias, scales, out, **extra)
        times = []
        for _ in range(TRIALS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / REPLAYS)
        result[f"rows{rows}_us"] = round(statistics.median(times), 2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
