"""Every launch configuration of the split seeded sampler, per batch, against the single
kernel, with the noise of each cell.

Uses the branch's kernels (sampling_kernels.py): seeded_top_k_chunk_kernel and
seeded_top_k_merge_sample_kernel (the merge finishes the bitonic rounds on the chunks'
sorted runs), launched directly with each (chunks, chunk warps, merge warps), and
seeded_top_k_top_p_sample_kernel at 2, 4, 8 and 16 warps; the served single kernel runs at
8 warps. A "topk_merge" variant re-sorts the chunk runs with tl.topk, as the first split did.

Each cell is the median us per call of 15 calls in one CUDA graph over 30 replays, taken
from 3 separate captures; the spread of the 3 is the cell's noise. A GPU sleep queued ahead
of each replay's start event keeps the graph launch's host latency out of the time. Rows use the model's
default sampling (temperature 0.9, top_k 50 at width 64, top_p 1.0) unless --top-p is set.
Every configuration's tokens are checked against the single kernel first on 512 rows with
normal logits, integer logits (ties) and signed zeros, at random per-row top_k.

Prints per width the cell table of the best configurations, then per configuration its
worst ratio to the best configuration of each batch (the regret) and its worst ratio to
the served single kernel, and the chunk and merge times of the best split.

usage: python sampler_split_sweep.py [--top-p 0.9] [--block-ks 64,128] [--out FILE]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch
import triton
import triton.language as tl

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    FUSED_SAMPLER_VOCAB_SIZE,
    sample_sorted_top_k,
    seeded_top_k_chunk_kernel,
    seeded_top_k_merge_sample_kernel,
    seeded_top_k_top_p_sample_kernel,
    unpack_top_keys,
)

CALLS = 15
REPLAYS = 30
CAPTURES = 3
SLEEP_CYCLES = 1_000_000
BATCHES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
BLOCK_KS = (64, 128, 256, 512, 1024)
CHUNKS = (2, 4, 8, 16, 32)
CHUNK_WARPS = (1, 2, 4, 8)
MERGE_WARPS = (1, 2, 4, 8)
SINGLE_WARPS = (2, 4, 8, 16)
SERVED_SINGLE = ("single", 1, 8, 0)
TOPK_MERGE_CONFIGS = ((4, 4, 8), (8, 4, 8))
CHECK_ROWS = 512
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50


@triton.jit
def topk_merge_sample_kernel(
    chunk_keys,
    top_ks,
    top_ps,
    seeds,
    positions,
    out,
    num_candidates: tl.constexpr,
    block_k: tl.constexpr,
    has_top_p: tl.constexpr,
):
    row = tl.program_id(0)
    candidates = tl.load(
        chunk_keys + row * num_candidates + tl.arange(0, num_candidates)
    ).to(tl.uint64)
    sorted_scores, sorted_token_ids = unpack_top_keys(
        tl.topk(candidates, k=block_k), block_k
    )
    sample_sorted_top_k(
        sorted_scores,
        sorted_token_ids,
        row,
        top_ks,
        top_ps,
        seeds,
        positions,
        out,
        block_k,
        has_top_p,
    )


def served_inputs(rows, block_k, top_p, generator):
    logits = (torch.randn(rows, FUSED_SAMPLER_VOCAB_SIZE, generator=generator) * 3).to(
        device="cuda", dtype=torch.bfloat16
    )
    temperatures = torch.full((rows,), DEFAULT_TEMPERATURE).cuda()
    top_k = DEFAULT_TOP_K if block_k == 64 else block_k
    top_ks = torch.full((rows,), top_k, dtype=torch.long).cuda()
    top_ps = torch.full((rows,), top_p).cuda()
    seeds = torch.randint(0, 2**62, (rows,), generator=generator).cuda()
    positions = torch.randint(0, 1 << 20, (rows,), generator=generator).cuda()
    return logits, temperatures, top_ks, top_ps, seeds, positions


def check_inputs(block_k, top_p, generator):
    quarter = CHECK_ROWS // 4
    normal = (
        torch.randn(
            CHECK_ROWS - 2 * quarter, FUSED_SAMPLER_VOCAB_SIZE, generator=generator
        )
        * 3
    )
    ties = torch.randint(
        -4, 5, (quarter, FUSED_SAMPLER_VOCAB_SIZE), generator=generator
    ).float()
    zeros = torch.where(
        torch.rand(quarter, FUSED_SAMPLER_VOCAB_SIZE, generator=generator) < 0.5,
        0.0,
        -0.0,
    )
    zeros[:, ::97] = 1.0
    logits = torch.cat([normal, ties, zeros]).to(device="cuda", dtype=torch.bfloat16)
    temperatures = (0.5 + torch.rand(CHECK_ROWS, generator=generator)).cuda()
    top_ks = torch.randint(1, block_k + 1, (CHECK_ROWS,), generator=generator).cuda()
    top_ps = torch.full((CHECK_ROWS,), top_p).cuda()
    seeds = torch.randint(0, 2**62, (CHECK_ROWS,), generator=generator).cuda()
    positions = torch.randint(0, 1 << 20, (CHECK_ROWS,), generator=generator).cuda()
    return logits, temperatures, top_ks, top_ps, seeds, positions


def split_launchers(rows, block_k, config, has_top_p, data):
    kind, chunks, chunk_warps, merge_warps = config
    logits, temperatures, top_ks, top_ps, seeds, positions = data
    num_candidates = chunks * block_k
    keys = torch.empty((rows, num_candidates), device="cuda", dtype=torch.int64)
    out = torch.empty(rows, device="cuda", dtype=torch.long)

    def chunk():
        seeded_top_k_chunk_kernel[(rows, chunks)](
            logits,
            temperatures,
            keys,
            logits.stride(0),
            FUSED_SAMPLER_VOCAB_SIZE // chunks,
            block_k,
            num_warps=chunk_warps,
        )

    def merge():
        if kind == "split":
            seeded_top_k_merge_sample_kernel[(rows,)](
                keys,
                top_ks,
                top_ps,
                seeds,
                positions,
                out,
                num_candidates,
                block_k,
                num_candidates.bit_length() - 1,
                block_k.bit_length() - 1,
                has_top_p,
                num_warps=merge_warps,
            )
        else:
            topk_merge_sample_kernel[(rows,)](
                keys,
                top_ks,
                top_ps,
                seeds,
                positions,
                out,
                num_candidates,
                block_k,
                has_top_p,
                num_warps=merge_warps,
            )
        return out

    def run():
        chunk()
        return merge()

    return run, chunk, merge


def single_runner(rows, block_k, warps, has_top_p, data):
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
            has_top_p,
            num_warps=warps,
        )
        return out

    return run


def runner(rows, block_k, config, has_top_p, data):
    if config[0] == "single":
        return single_runner(rows, block_k, config[2], has_top_p, data)
    else:
        return split_launchers(rows, block_k, config, has_top_p, data)[0]


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
            torch.cuda._sleep(SLEEP_CYCLES)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            replays.append(start.elapsed_time(end) * 1000 / CALLS)
        times.append(statistics.median(replays))
        del graph
    return times


def configs_for(block_k):
    split = [
        ("split", chunks, chunk_warps, merge_warps)
        for chunks, chunk_warps, merge_warps in itertools.product(
            CHUNKS, CHUNK_WARPS, MERGE_WARPS
        )
        if chunks * block_k <= FUSED_SAMPLER_VOCAB_SIZE
    ]
    topk_merge = [
        ("topk_merge", *config)
        for config in TOPK_MERGE_CONFIGS
        if config[0] * block_k < FUSED_SAMPLER_VOCAB_SIZE
    ]
    single = [("single", 1, warps, 0) for warps in SINGLE_WARPS]
    return split + topk_merge + single


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--block-ks", default=",".join(map(str, BLOCK_KS)))
    parser.add_argument("--out")
    args = parser.parse_args()
    has_top_p = 0.0 < args.top_p < 1.0
    device_name = torch.cuda.get_device_name(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{device_name}, {sms} SMs, triton {triton.__version__}, top_p {args.top_p}")
    generator = torch.Generator().manual_seed(0)
    results = {"device": device_name, "sms": sms, "top_p": args.top_p, "cells": []}
    for block_k in (int(value) for value in args.block_ks.split(",")):
        configs = configs_for(block_k)

        check = check_inputs(block_k, args.top_p, generator)
        reference = single_runner(CHECK_ROWS, block_k, 8, has_top_p, check)().clone()
        mismatched = [
            config
            for config in configs
            if not torch.equal(
                runner(CHECK_ROWS, block_k, config, has_top_p, check)(), reference
            )
        ]
        print(
            f"\nblock_k {block_k}: {len(configs)} configs, token mismatches {mismatched}"
        )

        table = {}
        for rows in BATCHES:
            data = served_inputs(rows, block_k, args.top_p, generator)
            for config in configs:
                times = graph_times(runner(rows, block_k, config, has_top_p, data))
                table[(config, rows)] = times
                results["cells"].append(
                    {
                        "block_k": block_k,
                        "config": list(config),
                        "rows": rows,
                        "us": times,
                    }
                )

        def cell(config, rows):
            return statistics.median(table[(config, rows)])

        best = {rows: min(cell(config, rows) for config in configs) for rows in BATCHES}
        regret = {
            config: max(cell(config, rows) / best[rows] for rows in BATCHES)
            for config in configs
        }
        against_served = {
            config: max(
                cell(config, rows) / cell(SERVED_SINGLE, rows) for rows in BATCHES
            )
            for config in configs
        }
        ranked = sorted(configs, key=lambda config: regret[config])
        shown = ranked[:10] + [
            config
            for config in configs
            if config[0] != "split" and config not in ranked[:10]
        ]
        header = "".join(f"{rows:>13}" for rows in BATCHES)
        print(f"  {'config (kind, chunks, chunk warps, merge warps)':<44}{header}")
        for config in shown:
            cells = "".join(
                f"{cell(config, rows):>8.2f}+-"
                f"{(max(table[(config, rows)]) - min(table[(config, rows)])) / 2:<3.2f}"
                for rows in BATCHES
            )
            print(f"  {str(config):<44}{cells}")
        print(
            "  regret (worst ratio to each batch's best) and worst ratio to the served kernel:"
        )
        for config in ranked[:15]:
            print(f"    {regret[config]:6.3f}  {against_served[config]:6.3f}  {config}")
        never_slower = [config for config in configs if against_served[config] < 1.0]
        print(
            f"  configs faster than the served kernel at every batch: {len(never_slower)}"
        )
        noise = statistics.median(
            (max(times) - min(times)) / statistics.median(times)
            for times in table.values()
        )
        print(f"  median relative spread of a cell across captures: {noise:.3f}")

        best_split = next(config for config in ranked if config[0] == "split")
        print(f"  chunk and merge alone for {best_split}:")
        for rows in BATCHES:
            data = served_inputs(rows, block_k, args.top_p, generator)
            _, chunk, merge = split_launchers(
                rows, block_k, best_split, has_top_p, data
            )
            chunk()
            chunk_us = statistics.median(graph_times(chunk))
            merge_us = statistics.median(graph_times(merge))
            print(
                f"    rows {rows:>4}: chunk {chunk_us:6.2f}  merge {merge_us:6.2f}  "
                f"pair {cell(best_split, rows):6.2f}  served {cell(SERVED_SINGLE, rows):6.2f}"
            )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle)
    else:
        pass


if __name__ == "__main__":
    main()
