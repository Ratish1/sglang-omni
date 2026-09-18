"""P2-e2: the sampler with precomputed noise (branch perf/qwen3-tts-sampler-noise).

Per decode position, as the predictor chain runs it: one seeded_gumbel_noise launch
for the 15 sub-steps, then 15 sampler calls that read their noise row. Timed in CUDA
graphs at bs 1, 2, 4, 8, 16 (checkpoint signature: top_k 50, no top-p): the noise
launch alone, the 15 sampler calls alone, and both together, us per decode position.

usage: python sampler_noise_bench.py
"""

from __future__ import annotations

import statistics

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_logits_with_seed_top_k_top_p,
    seeded_gumbel_noise,
)

VOCAB = 2048
MAX_TOP_K = 50
SUB_STEPS = 15
REPS = 50


def graph_us(fn) -> float:
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
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return 1e3 * statistics.median(times)


def main() -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    print(
        f"{'bs':>3} {'noise us':>9} {'15 samplers us':>15} {'per call us':>12} {'position total us':>18}"
    )
    for batch in (1, 2, 4, 8, 16):
        logits = (torch.randn(batch, VOCAB, device=device) * 3).to(torch.bfloat16)
        temperatures = torch.full((batch,), 0.9, device=device, dtype=torch.float32)
        top_ks = torch.full((batch,), MAX_TOP_K, device=device, dtype=torch.long)
        top_ps = torch.ones(batch, device=device, dtype=torch.float32)
        seeds = torch.arange(1000, 1000 + batch, device=device, dtype=torch.long)
        positions = torch.arange(
            SUB_STEPS * batch, device=device, dtype=torch.long
        ).view(SUB_STEPS, batch)
        noise = seeded_gumbel_noise(seeds, positions, max_top_k=MAX_TOP_K)

        def noise_only():
            seeded_gumbel_noise(seeds, positions, max_top_k=MAX_TOP_K)

        def samplers_only():
            for step in range(SUB_STEPS):
                sample_from_logits_with_seed_top_k_top_p(
                    logits,
                    temperatures,
                    top_ks,
                    top_ps,
                    noise[step],
                    max_top_k=MAX_TOP_K,
                    has_top_p=False,
                )

        def position():
            step_noise = seeded_gumbel_noise(seeds, positions, max_top_k=MAX_TOP_K)
            for step in range(SUB_STEPS):
                sample_from_logits_with_seed_top_k_top_p(
                    logits,
                    temperatures,
                    top_ks,
                    top_ps,
                    step_noise[step],
                    max_top_k=MAX_TOP_K,
                    has_top_p=False,
                )

        noise_us, samplers_us, total_us = (
            graph_us(noise_only),
            graph_us(samplers_only),
            graph_us(position),
        )
        print(
            f"{batch:>3} {noise_us:>9.2f} {samplers_us:>15.2f} {samplers_us / SUB_STEPS:>12.2f} {total_us:>18.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
