"""Launch configurations of the code2wav slot kernels, swept on every shape a 10-frame decode launches.

Run on the box from the branch's tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/code2wav_kernel_sweep.py --out /data/pr2800/<run>
Shapes come from the checkpoint's code2wav config: the 16 history exchanges (two ConvNeXt depthwise inputs, laid out
channels first, decoder.0, conv1 of the 12 residual units, the final conv), the 4 overlap exchanges (the decoder
blocks' transposed convs) and the 29 bias SnakeBeta launches (decoder.0's activation and seven per decoder block), at
1 and 8 rows, a later window (warm slots). Each configuration launches the kernel 20 times back to back inside one
CUDA graph, replayed 50 times; the time per launch is the replays' event time over 1000, as the decode replays them.
The floor is a one-program Triton kernel timed the same way. Registers and spills come from the compiled kernel.
Writes <out>/sweep.json and prints, per shape, the shipped configuration, the fastest, and the floor.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import AutoConfig

from sglang_omni.kernels.bias_snake_beta import (
    TRITON_DTYPES,
    VECTOR_BYTES,
    bias_snake_beta_channels_last_kernel,
)
from sglang_omni.kernels.codec_state import CHANNEL_BLOCK, history_exchange_kernel, overlap_exchange_kernel

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LAUNCHES = 20
REPLAYS = 50
DTYPE = torch.bfloat16


@triton.jit
def empty_kernel(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 0.0)


def graph_time_us(launch) -> float:
    """Microseconds per launch of launch(), LAUNCHES back to back in one graph, replayed REPLAYS times."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(LAUNCHES):
            launch()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * LAUNCHES)


def compiled_facts(compiled) -> dict[str, int | None]:
    return {
        "registers": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared": getattr(getattr(compiled, "metadata", None), "shared", None),
    }


def shapes(config) -> tuple[list[dict], list[dict], list[dict]]:
    """The decode's history, overlap and bias SnakeBeta launch shapes for 10 new frames."""
    hidden, decoder_dim = int(config.hidden_size), int(config.decoder_dim)
    histories, overlaps, snakes = [], [], []
    samples = 10
    for ratio in config.upsampling_ratios:
        samples *= int(ratio)
        histories.append(dict(name=f"upsample dwconv {hidden}", channels=hidden, history=6, new=samples, dilation=1,
                              channels_first=True))
    histories.append(dict(name="decoder.0", channels=hidden, history=6, new=samples, dilation=1, channels_first=False))
    snakes.append(dict(name="decoder.0 activation", channels=decoder_dim, length=samples, residual=False, total=False))
    for block, rate in enumerate(config.upsample_rates):
        channels = decoder_dim // 2 ** (block + 1)
        samples *= int(rate)
        overlaps.append(dict(name=f"block {block + 1} transconv", channels=channels, overlap=int(rate), emitted=samples))
        snakes.append(dict(name=f"block {block + 1} transconv bias", channels=channels, length=samples, residual=False,
                           total=True))
        for dilation in (1, 3, 9):
            histories.append(dict(name=f"block {block + 1} conv1 dilation {dilation}", channels=channels,
                                  history=6 * dilation, new=samples, dilation=dilation, channels_first=False))
            snakes.append(dict(name=f"block {block + 1} act2", channels=channels, length=samples, residual=False,
                               total=False))
            snakes.append(dict(name=f"block {block + 1} residual add", channels=channels, length=samples,
                               residual=True, total=dilation != 9))
    histories.append(dict(name="final conv", channels=decoder_dim // 2 ** len(config.upsample_rates), history=6,
                          new=samples, dilation=1, channels_first=False))
    return histories, overlaps, snakes


def history_launcher(shape: dict, rows: int, channel_block: int, num_warps: int):
    channels, history, new = shape["channels"], shape["history"], shape["new"]
    tail = (-new) % shape["dilation"]
    if shape["channels_first"]:
        padded = torch.randn(rows, channels, history + new, device="cuda", dtype=DTYPE).transpose(1, 2)
    else:
        padded = torch.randn(rows, history + new + tail, channels, device="cuda", dtype=DTYPE)
    pool = torch.randn(rows, history, channels, device="cuda", dtype=DTYPE)
    slots = torch.arange(rows, device="cuda")

    def launch():
        return history_exchange_kernel[(rows, triton.cdiv(channels, channel_block))](
            padded, pool, slots, history, new, padded.shape[1] - history - new, channels, *padded.stride(),
            *pool.stride(), IS_COLD=False, IS_SHIFT=new < history,
            HISTORY_BLOCK=triton.next_power_of_2(history), TAIL_BLOCK=triton.next_power_of_2(max(tail, 1)),
            CHANNEL_BLOCK=channel_block, num_warps=num_warps,
        )

    return launch


def overlap_launcher(shape: dict, rows: int, channel_block: int, num_warps: int):
    channels, overlap, emitted = shape["channels"], shape["overlap"], shape["emitted"]
    output = torch.randn(rows, emitted + overlap, channels, device="cuda", dtype=DTYPE)
    pool = torch.randn(rows, overlap, channels, device="cuda", dtype=DTYPE)
    slots = torch.arange(rows, device="cuda")

    def launch():
        return overlap_exchange_kernel[(rows, triton.cdiv(channels, channel_block))](
            output, pool, slots, overlap, emitted, channels, *output.stride(), *pool.stride(), IS_COLD=False,
            OVERLAP_BLOCK=triton.next_power_of_2(overlap), CHANNEL_BLOCK=channel_block, num_warps=num_warps,
        )

    return launch


def snake_launcher(shape: dict, rows: int, block_positions: int, num_warps: int):
    channels, length = shape["channels"], shape["length"]
    conv = torch.randn(rows, length + 7, channels, device="cuda", dtype=DTYPE)[:, 3 : 3 + length]
    out = torch.zeros(rows, length + 60, channels, device="cuda", dtype=DTYPE)[:, 54 : 54 + length]
    residual = torch.randn(rows, length, channels, device="cuda", dtype=DTYPE)
    bias, alpha, beta = (torch.randn(channels, device="cuda", dtype=DTYPE) * 0.5 for _ in range(3))
    block_channels = triton.next_power_of_2(channels)

    def launch():
        return bias_snake_beta_channels_last_kernel[(triton.cdiv(length, block_positions), rows)](
            conv, conv.stride(0), bias, residual, residual, out, out.stride(0), alpha, beta, channels, length, 1e-9,
            DTYPE=TRITON_DTYPES[DTYPE], HAS_RESIDUAL=shape["residual"], WRITE_SUM=shape["total"],
            BLOCK_POSITIONS=block_positions, BLOCK_CHANNELS=block_channels, num_warps=num_warps,
            enable_fp_fusion=False, enable_reflect_ftz=False,
        )

    return launch


def shipped_snake(shape: dict) -> tuple[int, int]:
    block_channels = triton.next_power_of_2(shape["channels"])
    vector_elements = VECTOR_BYTES // torch.empty(0, dtype=DTYPE).element_size()
    num_warps = max(4, block_channels // (32 * vector_elements))
    return max(1, 32 * num_warps * vector_elements // block_channels), num_warps


def sweep(kind: str, shape: dict, rows: int, configs, launcher, shipped) -> dict:
    results = []
    for first, num_warps in configs:
        launch = launcher(shape, rows, first, num_warps)
        compiled = launch()
        results.append(dict(config=[first, num_warps], us=round(graph_time_us(launch), 3), **compiled_facts(compiled)))
    best = min(results, key=lambda row: row["us"])
    current = next(row for row in results if row["config"] == list(shipped))
    return dict(kind=kind, rows=rows, **shape, shipped=current, best=best, all=results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    histories, overlaps, snakes = shapes(config)
    flag = torch.zeros(1, device="cuda")
    floor = graph_time_us(lambda: empty_kernel[(1,)](flag))
    print(f"floor: one-program kernel {floor:.3f} us a launch")
    exchange_configs = list(itertools.product((16, 32, 64, 128, 256), (1, 2, 4, 8)))
    records = []
    for rows in (1, 8):
        for shape in histories:
            records.append(sweep("history", shape, rows, exchange_configs, history_launcher, (CHANNEL_BLOCK, 4)))
        for shape in overlaps:
            records.append(sweep("overlap", shape, rows, exchange_configs, overlap_launcher, (CHANNEL_BLOCK, 4)))
        for shape in snakes:
            block_channels = triton.next_power_of_2(shape["channels"])
            configs = [(positions, warps) for positions, warps in itertools.product((1, 2, 4, 8, 16, 32, 64), (1, 2, 4, 8))
                       if positions * block_channels <= 16384]
            records.append(sweep("snake", shape, rows, configs, snake_launcher, shipped_snake(shape)))
    (args.out / "sweep.json").write_text(json.dumps(dict(floor_us=floor, records=records), indent=1))
    for rows in (1, 8):
        for kind in ("history", "overlap", "snake"):
            group = [record for record in records if record["rows"] == rows and record["kind"] == kind]
            shipped_total = sum(record["shipped"]["us"] for record in group)
            best_total = sum(record["best"]["us"] for record in group)
            print(f"rows {rows} {kind}: {len(group)} launches, shipped {shipped_total:.1f} us, best per shape "
                  f"{best_total:.1f} us, floor {floor * len(group):.1f} us")
            for record in group:
                print(f"  {record['name']:34s} shipped {record['shipped']['config']} {record['shipped']['us']:6.2f} us "
                      f"r{record['shipped']['registers']} s{record['shipped']['spills']} | best {record['best']['config']} "
                      f"{record['best']['us']:6.2f} us r{record['best']['registers']} s{record['best']['spills']}")


if __name__ == "__main__":
    main()
