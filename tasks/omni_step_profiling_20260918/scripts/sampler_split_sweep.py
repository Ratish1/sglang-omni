"""Every launch configuration of the split seeded sampler, per batch, against the single
kernel, with the noise of each cell.

Uses the branch's kernels (sampling_kernels.py): seeded_top_k_chunk_kernel and
seeded_top_k_merge_sample_kernel, launched directly with each (chunks, chunk warps,
merge warps), and seeded_top_k_top_p_sample_kernel at 4, 8 and 16 warps as the baseline.
Each cell is the median us per call of 15 calls in one CUDA graph over 30 replays, taken
from 3 separate captures; the spread of the 3 is the cell's noise. Every configuration's
tokens are checked against the single kernel on 64 rows first.

Prints per block_k the cell table, then per configuration its worst ratio to the best
configuration of each batch (the regret) over all batches.

usage: python sampler_split_sweep.py [--out FILE]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    FUSED_SAMPLER_VOCAB_SIZE,
    seeded_top_k_chunk_kernel,
    seeded_top_k_merge_sample_kernel,
    seeded_top_k_top_p_sample_kernel,
)

CALLS = 15
REPLAYS = 30
CAPTURES = 3
BATCHES = (1, 2, 4, 8, 16, 32, 64)
BLOCK_KS = (64, 128)
CHUNKS = (2, 4, 8, 16)
CHUNK_WARPS = (2, 4, 8)
MERGE_WARPS = (4, 8, 16)
SINGLE_WARPS = (4, 8, 16)


def inputs(rows: int, block_k: int, generator: torch.Generator):
    logits = (torch.randn(rows, FUSED_SAMPLER_VOCAB_SIZE, generator=generator) * 3).to(
        device="cuda", dtype=torch.bfloat16
    )
    temperatures = (0.5 + torch.rand(rows, generator=generator)).cuda()
    top_ks = torch.full((rows,), block_k - 14 if block_k == 64 else block_k).cuda()
    top_ps = torch.full((rows,), 0.9).cuda()
    seeds = torch.randint(0, 2**62, (rows,), generator=generator).cuda()
    positions = torch.randint(0, 1 << 20, (rows,), generator=generator).cuda()
    return logits, temperatures, top_ks, top_ps, seeds, positions


def split_runner(rows, block_k, chunks, chunk_warps, merge_warps, data):
    logits, temperatures, top_ks, top_ps, seeds, positions = data
    keys = torch.empty((rows, chunks * block_k), device="cuda", dtype=torch.int64)
    out = torch.empty(rows, device="cuda", dtype=torch.long)

    def run():
        seeded_top_k_chunk_kernel[(rows, chunks)](
            logits,
            temperatures,
            keys,
            logits.stride(0),
            FUSED_SAMPLER_VOCAB_SIZE // chunks,
            block_k,
            num_warps=chunk_warps,
        )
        seeded_top_k_merge_sample_kernel[(rows,)](
            keys,
            top_ks,
            top_ps,
            seeds,
            positions,
            out,
            chunks * block_k,
            block_k,
            True,
            num_warps=merge_warps,
        )
        return out

    return run


def single_runner(rows, block_k, warps, data):
    logits, temperatures, top_ks, top_ps, seeds, positions = data
    out = torch.empty(rows, device="cuda", dtype=torch.long)

    def run():
        seeded_top_k_top_p_sample_kernel[(rows,)](
            logits,
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            out,
            logits.stride(0),
            block_k,
            block_k,
            True,
            num_warps=warps,
        )
        return out

    return run


def graph_times(run) -> list[float]:
    times = []
    for _ in range(CAPTURES):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CALLS):
                run()
        graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        replays = []
        for _ in range(REPLAYS):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            replays.append(start.elapsed_time(end) * 1000 / CALLS)
        times.append(statistics.median(replays))
        del graph
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    device_name = torch.cuda.get_device_name(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{device_name}, {sms} SMs")
    generator = torch.Generator().manual_seed(0)
    results = {"device": device_name, "sms": sms, "cells": []}
    for block_k in BLOCK_KS:
        configs = [
            ("split", chunks, chunk_warps, merge_warps)
            for chunks, chunk_warps, merge_warps in itertools.product(
                CHUNKS, CHUNK_WARPS, MERGE_WARPS
            )
            if chunks * block_k < FUSED_SAMPLER_VOCAB_SIZE
            and FUSED_SAMPLER_VOCAB_SIZE // chunks >= block_k
        ] + [("single", 1, warps, 0) for warps in SINGLE_WARPS]

        check = inputs(64, block_k, generator)
        reference = single_runner(64, block_k, 8, check)().clone()
        mismatched = [
            config
            for config in configs
            if config[0] == "split"
            and not torch.equal(
                split_runner(64, block_k, *config[1:], check)(), reference
            )
        ]
        print(
            f"\nblock_k {block_k}: {len(configs)} configs, token mismatches {mismatched}"
        )

        table = {}
        for rows in BATCHES:
            data = inputs(rows, block_k, generator)
            for config in configs:
                if config[0] == "split":
                    run = split_runner(rows, block_k, *config[1:], data)
                else:
                    run = single_runner(rows, block_k, config[2], data)
                times = graph_times(run)
                table[(config, rows)] = times
                results["cells"].append(
                    {
                        "block_k": block_k,
                        "config": list(config),
                        "rows": rows,
                        "us": times,
                    }
                )
        header = "".join(f"{rows:>14}" for rows in BATCHES)
        print(f"  {'config (kind, chunks, chunk warps, merge warps)':<48}{header}")
        for config in configs:
            cells = "".join(
                f"{statistics.median(table[(config, rows)]):>8.2f}+-"
                f"{(max(table[(config, rows)]) - min(table[(config, rows)])) / 2:<4.2f}"
                for rows in BATCHES
            )
            print(f"  {str(config):<48}{cells}")
        best = {
            rows: min(statistics.median(table[(config, rows)]) for config in configs)
            for rows in BATCHES
        }
        regrets = sorted(
            (
                max(
                    statistics.median(table[(config, rows)]) / best[rows]
                    for rows in BATCHES
                ),
                config,
            )
            for config in configs
        )
        print("  worst ratio to the best config of each batch (lower is better):")
        for regret, config in regrets[:8]:
            print(f"    {regret:6.3f}  {config}")
        noise = statistics.median(
            (max(times) - min(times)) / statistics.median(times)
            for times in table.values()
        )
        print(f"  median relative spread of a cell across captures: {noise:.3f}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle)
    else:
        pass


if __name__ == "__main__":
    main()
