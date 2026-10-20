"""SGLang's bias_gelu launch carried to the bias SnakeBeta kernel, prototyped against the shipped launch.

Run on the box from the branch's tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/snake_launch_prototype.py --out /data/pr2800/<run> --sweep <sweep.json>
bias_gelu (v0.5.21 jit/csrc/elementwise/bias_gelu.cuh:80-91) loads kMaxVecBytes a thread (16, 32 from Blackwell on),
runs 256 threads a block, launches min(SM count x the occupancy API's blocks per SM, blocks needed) and walks the
vectors with a grid-stride loop. The prototype keeps the branch's tile (whole positions, so a tile is one contiguous
span) and its rounding chain, with 256 threads moving one vector each a tile: "tiles" launches one program a tile,
"persistent" launches min(SM count x occupancy, tiles) programs looping over the tiles. Every 29 bias SnakeBeta shapes
of a 10-frame decode at 1 and 8 rows, timed as code2wav_kernel_sweep.py times them, each output checked bit for bit
against the shipped launch. Writes <out>/prototype.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import AutoConfig

from sglang_omni.kernels.bias_snake_beta import (
    TRITON_DTYPES,
    fused_bias_snake_beta,
    snake_beta_coefficients,
    snake_beta_values,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import DTYPE, MODEL, graph_time_us, shapes  # noqa: E402

THREADS = 256


@triton.jit(do_not_specialize=["length", "tiles_per_row", "total_tiles"])
def persistent_kernel(
    conv_ptr, conv_batch_stride, bias_ptr, residual_ptr, sum_ptr, out_ptr, out_batch_stride, alpha_ptr, beta_ptr,
    channels, length, epsilon, tiles_per_row, total_tiles, DTYPE: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr, BLOCK_POSITIONS: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    channel = tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channel < channels
    alpha_scale, inverse_beta = snake_beta_coefficients(
        tl.load(alpha_ptr + channel, mask=channel_mask, other=0).to(tl.float32),
        tl.load(beta_ptr + channel, mask=channel_mask, other=0).to(tl.float32),
        epsilon,
        DTYPE,
    )
    bias = tl.load(bias_ptr + channel, mask=channel_mask, other=0).to(tl.float32)
    for tile in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        batch = (tile // tiles_per_row).to(tl.int64)
        positions = (tile % tiles_per_row) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
        mask = (positions < length)[:, None] & channel_mask[None, :]
        offsets = positions[:, None] * channels + channel[None, :]
        contiguous_base = batch * length * channels
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


def vector_bytes() -> int:
    """kMaxVecBytes: 16 bytes a load before Blackwell, 32 from it on."""
    return 32 if torch.cuda.get_device_capability()[0] >= 10 else 16


def blocks_per_sm(compiled) -> tuple[int, str]:
    """The occupancy API's resident blocks per SM for the compiled kernel at THREADS, or the ISA arithmetic."""
    shared = int(getattr(compiled.metadata, "shared", 0))
    try:
        from cuda.bindings import driver

        error, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            driver.CUfunction(compiled.function), THREADS, shared
        )
        if int(error) == 0:
            return int(blocks), "occupancy API"
        else:
            pass
    except (ImportError, TypeError, AttributeError):
        pass
    properties = torch.cuda.get_device_properties(0)
    registers = -(-int(compiled.n_regs) // 8) * 8
    limits = [properties.max_threads_per_multi_processor // THREADS, 32]
    limits.append(properties.regs_per_multiprocessor // (registers * THREADS))
    if shared:
        limits.append(properties.shared_memory_per_multiprocessor // shared)
    else:
        pass
    return max(1, min(limits)), "ISA arithmetic"


def tensors(shape: dict, rows: int):
    channels, length = shape["channels"], shape["length"]
    conv = torch.randn(rows, length + 7, channels, device="cuda", dtype=DTYPE)[:, 3 : 3 + length]
    residual = torch.randn(rows, length, channels, device="cuda", dtype=DTYPE)
    bias, alpha, beta = (torch.randn(channels, device="cuda", dtype=DTYPE) * 0.5 for _ in range(3))
    return conv, residual, bias, alpha, beta


class Activation:
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        self.alpha, self.beta, self.no_div_by_zero = alpha, beta, 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    _, _, snakes = shapes(config)
    best = {(r["rows"], r["name"], r["channels"], r["length"]): r["best"]["us"]
            for r in json.loads(args.sweep.read_text())["records"] if r["kind"] == "snake"}
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    vector_elements = vector_bytes() // torch.empty(0, dtype=DTYPE).element_size()
    records = []
    for rows in (1, 8):
        for shape in snakes:
            conv, residual, bias, alpha, beta = tensors(shape, rows)
            activation = Activation(alpha, beta)
            channels, length = shape["channels"], shape["length"]
            shipped_out = torch.zeros(rows, length + 60, channels, device="cuda", dtype=DTYPE)[:, 54 : 54 + length]
            shipped_sum = residual.clone()

            def shipped():
                fused_bias_snake_beta(conv, bias, activation, shipped_out,
                                      residual=shipped_sum if shape["residual"] else None,
                                      sum_out=shipped_sum if shape["total"] else None)

            block_channels = triton.next_power_of_2(channels)
            block_positions = max(1, THREADS * vector_elements // block_channels)
            tiles_per_row = triton.cdiv(length, block_positions)
            total_tiles = tiles_per_row * rows
            variants = {}
            for name in ("tiles", "persistent"):
                out = torch.zeros(rows, length + 60, channels, device="cuda", dtype=DTYPE)[:, 54 : 54 + length]
                total = residual.clone()
                state = {"grid": total_tiles}

                def launch(out=out, total=total, state=state):
                    return persistent_kernel[(state["grid"],)](
                        conv, conv.stride(0), bias, total, total, out, out.stride(0), alpha, beta, channels, length,
                        1e-9, tiles_per_row, total_tiles, DTYPE=TRITON_DTYPES[DTYPE], HAS_RESIDUAL=shape["residual"],
                        WRITE_SUM=shape["total"], BLOCK_POSITIONS=block_positions, BLOCK_CHANNELS=block_channels,
                        num_warps=THREADS // 32, enable_fp_fusion=False, enable_reflect_ftz=False,
                    )

                compiled = launch()
                if name == "persistent":
                    occupancy, source = blocks_per_sm(compiled)
                    state["grid"] = min(sm_count * occupancy, total_tiles)
                else:
                    occupancy, source = None, None
                variants[name] = dict(launch=launch, out=out, total=total, grid=state["grid"], occupancy=occupancy,
                                      source=source, registers=compiled.n_regs, spills=compiled.n_spills)
            timings = {"shipped": graph_time_us(shipped)}
            for name, variant in variants.items():
                timings[name] = graph_time_us(variant["launch"])
            # one fresh launch of each from the same inputs for the bit check
            shipped_sum.copy_(residual)
            shipped()
            equal = {}
            for name, variant in variants.items():
                variant["total"].copy_(residual)
                variant["launch"]()
                equal[name] = bool(torch.equal(variant["out"], shipped_out)
                                   and (not shape["total"] or torch.equal(variant["total"], shipped_sum)))
            torch.cuda.synchronize()
            record = dict(rows=rows, **shape, us={k: round(v, 3) for k, v in timings.items()},
                          best_swept=best.get((rows, shape["name"], channels, length)), equal=equal,
                          block_positions=block_positions, tiles=total_tiles,
                          persistent_grid=variants["persistent"]["grid"],
                          occupancy=variants["persistent"]["occupancy"], occupancy_source=variants["persistent"]["source"],
                          registers=variants["persistent"]["registers"], spills=variants["persistent"]["spills"])
            records.append(record)
    (args.out / "prototype.json").write_text(json.dumps(dict(sm_count=sm_count, vector_bytes=vector_bytes(),
                                                             records=records), indent=1))
    print(f"SMs {sm_count}, vector {vector_bytes()} bytes, threads {THREADS}")
    for rows in (1, 8):
        group = [r for r in records if r["rows"] == rows]
        print(f"rows {rows}: shipped {sum(r['us']['shipped'] for r in group):.1f} us, tiles "
              f"{sum(r['us']['tiles'] for r in group):.1f}, persistent {sum(r['us']['persistent'] for r in group):.1f}, "
              f"best swept {sum(r['best_swept'] or 0 for r in group):.1f}; all equal "
              f"{all(all(r['equal'].values()) for r in group)}")
        for r in group:
            print(f"  {r['name']:30s} C {r['channels']:5d} P {r['block_positions']:3d} tiles {r['tiles']:6d} grid "
                  f"{r['persistent_grid']:5d} occ {r['occupancy']} ({r['occupancy_source']}) r{r['registers']} "
                  f"| shipped {r['us']['shipped']:6.2f} tiles {r['us']['tiles']:6.2f} persistent "
                  f"{r['us']['persistent']:6.2f} best {r['best_swept']}")


if __name__ == "__main__":
    main()
