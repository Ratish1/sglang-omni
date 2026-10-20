"""The bias SnakeBeta kernel's two layout costs, measured apart and together on every decode shape.

Run on the box from the branch's tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/snake_layout_probe.py --out /data/pr2800/<run>
Every code2wav channel count is 3 x 2^k, so the shipped tile of next_power_of_2(channels) lanes a position leaves a
quarter of them masked, computing sines on zeros. Each program also recomputes exp(alpha) and 1 / (exp(beta) + eps)
for the channels its threads hold. "padded" keeps the shipped tile, "unpadded" walks C / lowbit(C) chunks of
lowbit(C) channels over the same whole positions (one contiguous span, no masked lane); "computed" derives the
coefficients in the kernel as shipped, "hoisted" loads them, computed once with the eager ops SnakeBeta runs. Each of
the 17 distinct shapes of a 10-frame decode, at 1 and 8 rows, is timed as code2wav_kernel_sweep.py times it over
(positions, warps); every output is checked bit for bit against the shipped launch. The memory floor is the shape's
bytes at the bandwidth a large copy reaches in the same graph timing. Writes <out>/layout.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import AutoConfig

from sglang_omni.kernels.bias_snake_beta import (
    TRITON_DTYPES,
    VECTOR_BYTES,
    fused_bias_snake_beta,
    snake_beta_coefficients,
    snake_beta_values,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import DTYPE, MODEL, compiled_facts, graph_time_us, shapes  # noqa: E402

EPSILON = 1e-9
MAX_TILE = 16384
MAX_THREAD_ELEMENTS = 64


@triton.jit(do_not_specialize=["length"])
def layout_kernel(
    conv_ptr, conv_batch_stride, bias_ptr, residual_ptr, sum_ptr, out_ptr, out_batch_stride, alpha_ptr, beta_ptr,
    channels, length, epsilon, DTYPE: tl.constexpr, HAS_RESIDUAL: tl.constexpr, WRITE_SUM: tl.constexpr,
    HOISTED: tl.constexpr, CHUNKS: tl.constexpr, BLOCK_POSITIONS: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    batch = tl.program_id(1).to(tl.int64)
    contiguous_base = batch * length * channels
    for chunk in tl.static_range(CHUNKS):
        channel = chunk * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        channel_mask = channel < channels
        alpha = tl.load(alpha_ptr + channel, mask=channel_mask, other=0).to(tl.float32)
        beta = tl.load(beta_ptr + channel, mask=channel_mask, other=0).to(tl.float32)
        if HOISTED:
            alpha_scale = alpha
            inverse_beta = beta
        else:
            alpha_scale, inverse_beta = snake_beta_coefficients(alpha, beta, epsilon, DTYPE)
        bias = tl.load(bias_ptr + channel, mask=channel_mask, other=0).to(tl.float32)
        mask = (positions < length)[:, None] & channel_mask[None, :]
        offsets = positions[:, None] * channels + channel[None, :]
        conv_values = tl.load(conv_ptr + batch * conv_batch_stride + offsets, mask=mask, other=0).to(tl.float32)
        total = (conv_values + bias[None, :]).to(DTYPE).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + contiguous_base + offsets, mask=mask, other=0)
            total = (total + residual.to(tl.float32)).to(DTYPE).to(tl.float32)
        else:
            pass
        if WRITE_SUM:
            tl.store(sum_ptr + contiguous_base + offsets, total.to(DTYPE), mask=mask)
        else:
            pass
        tl.store(out_ptr + batch * out_batch_stride + offsets,
                 snake_beta_values(total, alpha_scale[None, :], inverse_beta[None, :], DTYPE), mask=mask)


class Activation:
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        self.alpha, self.beta, self.no_div_by_zero = alpha, beta, EPSILON


def vector_elements() -> int:
    return VECTOR_BYTES // torch.empty(0, dtype=DTYPE).element_size()


def tile_channels(layout: str, channels: int) -> tuple[int, int]:
    """(lanes a chunk, chunks) of a position: one padded chunk, or lowbit(C) lanes C / lowbit(C) times."""
    lowbit = channels & -channels
    if layout == "unpadded" and lowbit >= vector_elements():
        return lowbit, channels // lowbit
    else:
        return triton.next_power_of_2(channels), 1


def rule(layout: str, channels: int) -> tuple[int, int]:
    """The shipped launch rule on a chunk: one 16-byte vector a thread a chunk, four warps or one per vector of it."""
    block_channels, _ = tile_channels(layout, channels)
    num_warps = max(4, block_channels // (32 * vector_elements()))
    return max(1, 32 * num_warps * vector_elements() // block_channels), num_warps


def configs(layout: str, channels: int) -> list[tuple[int, int]]:
    block_channels, chunks = tile_channels(layout, channels)
    found = []
    for positions, num_warps in itertools.product((1, 2, 4, 8, 16, 32, 64, 128), (1, 2, 4, 8)):
        chunk_elements = positions * block_channels
        if (chunk_elements * chunks <= MAX_TILE and chunk_elements >= 32 * num_warps
                and chunk_elements * chunks // (32 * num_warps) <= MAX_THREAD_ELEMENTS):
            found.append((positions, num_warps))
        else:
            pass
    if rule(layout, channels) not in found:
        found.append(rule(layout, channels))
    else:
        pass
    return found


def copy_bandwidth() -> float:
    """Bytes a second a 128 MB copy reaches, timed as the kernels are."""
    source = torch.randn(64 * 1024 * 1024, device="cuda", dtype=DTYPE)
    target = torch.empty_like(source)
    us = graph_time_us(lambda: target.copy_(source))
    return 2 * source.numel() * source.element_size() / (us * 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    _, _, snakes = shapes(config)
    counts = Counter((s["channels"], s["length"], s["residual"], s["total"]) for s in snakes)
    names = {}
    for s in snakes:
        names.setdefault((s["channels"], s["length"], s["residual"], s["total"]), s["name"])
    bandwidth = copy_bandwidth()
    element_size = torch.empty(0, dtype=DTYPE).element_size()
    variants = [(layout, hoisted) for layout in ("padded", "unpadded") for hoisted in (False, True)]
    records = []
    for rows in (1, 8):
        for key, count in counts.items():
            channels, length, has_residual, write_sum = key
            conv = torch.randn(rows, length + 7, channels, device="cuda", dtype=DTYPE)[:, 3 : 3 + length]
            residual = torch.randn(rows, length, channels, device="cuda", dtype=DTYPE)
            bias, alpha, beta = (torch.randn(channels, device="cuda", dtype=DTYPE) * 0.5 for _ in range(3))
            alpha_scale = torch.exp(alpha)
            inverse_beta = 1.0 / (torch.exp(beta) + EPSILON)
            activation = Activation(alpha, beta)

            def new_out() -> torch.Tensor:
                return torch.zeros(rows, length + 60, channels, device="cuda", dtype=DTYPE)[:, 54 : 54 + length]

            reference_out, reference_sum = new_out(), residual.clone()
            fused_bias_snake_beta(conv, bias, activation, reference_out,
                                  residual=reference_sum if has_residual else None,
                                  sum_out=reference_sum if write_sum else None)
            shipped_out, shipped_sum = new_out(), residual.clone()
            shipped_us = graph_time_us(lambda: fused_bias_snake_beta(
                conv, bias, activation, shipped_out, residual=shipped_sum if has_residual else None,
                sum_out=shipped_sum if write_sum else None))
            bytes_moved = rows * length * channels * element_size * (2 + has_residual + write_sum)
            record = dict(rows=rows, name=names[key], channels=channels, length=length, residual=has_residual,
                          total=write_sum, count=count, shipped_us=round(shipped_us, 3),
                          memory_floor_us=round(bytes_moved / bandwidth * 1e6, 3), variants={})
            for layout, hoisted in variants:
                block_channels, chunks = tile_channels(layout, channels)
                results = []
                for positions, num_warps in configs(layout, channels):
                    out, total = new_out(), residual.clone()

                    def launch(out=out, total=total, positions=positions, num_warps=num_warps):
                        return layout_kernel[(triton.cdiv(length, positions), rows)](
                            conv, conv.stride(0), bias, total, total, out, out.stride(0),
                            alpha_scale if hoisted else alpha, inverse_beta if hoisted else beta, channels, length,
                            EPSILON, DTYPE=TRITON_DTYPES[DTYPE], HAS_RESIDUAL=has_residual, WRITE_SUM=write_sum,
                            HOISTED=hoisted, CHUNKS=chunks, BLOCK_POSITIONS=positions, BLOCK_CHANNELS=block_channels,
                            num_warps=num_warps, enable_fp_fusion=False, enable_reflect_ftz=False,
                        )

                    compiled = launch()
                    equal = bool(torch.equal(out, reference_out)
                                 and (not write_sum or torch.equal(total, reference_sum)))
                    results.append(dict(config=[positions, num_warps], us=round(graph_time_us(launch), 3),
                                        equal=equal, **compiled_facts(compiled)))
                best = min(results, key=lambda row: row["us"])
                at_rule = next(row for row in results if row["config"] == list(rule(layout, channels)))
                record["variants"][f"{layout} {'hoisted' if hoisted else 'computed'}"] = dict(
                    block_channels=block_channels, chunks=chunks, rule=at_rule, best=best,
                    all_equal=all(row["equal"] for row in results), all=results)
            records.append(record)
            print(f"rows {rows} {names[key]:26s} C {channels:5d} L {length:6d} x{count} shipped {shipped_us:6.2f} "
                  f"floor {record['memory_floor_us']:6.2f} | " + " | ".join(
                      f"{name} rule {v['rule']['us']:6.2f} best {v['best']['config']} {v['best']['us']:6.2f}"
                      f"{'' if v['all_equal'] else ' UNEQUAL'}" for name, v in record["variants"].items()),
                  flush=True)
    (args.out / "layout.json").write_text(json.dumps(dict(bandwidth=bandwidth, records=records), indent=1))
    print(f"copy bandwidth {bandwidth / 1e12:.2f} TB/s")
    for rows in (1, 8):
        group = [r for r in records if r["rows"] == rows]
        line = (f"rows {rows} (29 launches): shipped {sum(r['shipped_us'] * r['count'] for r in group):.1f} us, "
                f"memory floor {sum(r['memory_floor_us'] * r['count'] for r in group):.1f}")
        for name in group[0]["variants"]:
            line += (f"; {name} rule {sum(r['variants'][name]['rule']['us'] * r['count'] for r in group):.1f} best "
                     f"{sum(r['variants'][name]['best']['us'] * r['count'] for r in group):.1f}")
        print(line)
        print(f"  all outputs equal: {all(v['all_equal'] for r in group for v in r['variants'].values())}")


if __name__ == "__main__":
    main()
