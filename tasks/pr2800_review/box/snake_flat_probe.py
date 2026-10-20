"""SGLang's elementwise layout carried to the bias SnakeBeta kernel with hoisted coefficients.

Run on the box from the branch's tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/snake_flat_probe.py --out /data/pr2800/<run>
SGLang's CUDA elementwise kernels (v0.5.21 kernels/jit/csrc/elementwise/activation.cuh, bias_gelu.cuh) walk a
row-major tensor as flat kMaxVecBytes vectors, one a thread, 256 threads a block, and read the per-channel vector at
vec_id % row_vecs: no tile padding. activation.cuh launches cdiv(vectors, 256) blocks ("flat"); bias_gelu.cuh
launches min(SM count x the occupancy API's blocks per SM, blocks needed) and grid-strides ("flat persistent", here
per row: each row's programs cap at cdiv(SM count x occupancy, rows)). Both read exp(alpha) and
1 / (exp(beta) + eps) computed once with the eager ops, as Mamba2's weight loader stores -exp(A_log) (v0.5.21
srt/layers/attention/mamba/mamba.py:419). Against them: the shipped launch and the shipped tile with the same hoisted
coefficients at the shipped rule ("padded hoisted"). The 17 distinct shapes of a 10-frame decode at 1 and 8 rows,
timed as code2wav_kernel_sweep.py times them, every output checked bit for bit against the shipped launch. Writes
<out>/flat.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import AutoConfig

from sglang_omni.kernels.bias_snake_beta import TRITON_DTYPES, VECTOR_BYTES, fused_bias_snake_beta, snake_beta_values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import DTYPE, MODEL, compiled_facts, graph_time_us, shapes  # noqa: E402
from snake_launch_prototype import blocks_per_sm  # noqa: E402
from snake_layout_probe import EPSILON, Activation, layout_kernel, rule, tile_channels  # noqa: E402

THREADS = 256


@triton.jit
def flat_vectors(
    conv_row, bias_ptr, residual_row, sum_row, out_row, alpha_scale_ptr, inverse_beta_ptr, start, row_vectors,
    channel_vectors, DTYPE: tl.constexpr, HAS_RESIDUAL: tl.constexpr, WRITE_SUM: tl.constexpr,
    BLOCK_VECTORS: tl.constexpr, VECTOR: tl.constexpr,
):
    vector = start + tl.arange(0, BLOCK_VECTORS)
    lane = tl.arange(0, VECTOR)
    mask = (vector[:, None] < row_vectors) & (lane[None, :] < VECTOR)
    offsets = vector[:, None] * VECTOR + lane[None, :]
    channel = (vector % channel_vectors)[:, None] * VECTOR + lane[None, :]
    alpha_scale = tl.load(alpha_scale_ptr + channel, mask=mask, other=0).to(tl.float32)
    inverse_beta = tl.load(inverse_beta_ptr + channel, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0).to(tl.float32)
    conv_values = tl.load(conv_row + offsets, mask=mask, other=0).to(tl.float32)
    total = (conv_values + bias).to(DTYPE).to(tl.float32)
    if HAS_RESIDUAL:
        total = (total + tl.load(residual_row + offsets, mask=mask, other=0).to(tl.float32)).to(DTYPE).to(tl.float32)
    else:
        pass
    if WRITE_SUM:
        tl.store(sum_row + offsets, total.to(DTYPE), mask=mask)
    else:
        pass
    tl.store(out_row + offsets, snake_beta_values(total, alpha_scale, inverse_beta, DTYPE), mask=mask)


@triton.jit(do_not_specialize=["row_vectors", "channel_vectors"])
def flat_kernel(
    conv_ptr, conv_batch_stride, bias_ptr, residual_ptr, sum_ptr, out_ptr, out_batch_stride, alpha_scale_ptr,
    inverse_beta_ptr, row_vectors, channel_vectors, DTYPE: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr, PERSISTENT: tl.constexpr, BLOCK_VECTORS: tl.constexpr, VECTOR: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    contiguous_base = batch * row_vectors * VECTOR
    conv_row = conv_ptr + batch * conv_batch_stride
    out_row = out_ptr + batch * out_batch_stride
    if PERSISTENT:
        for start in range(tl.program_id(0) * BLOCK_VECTORS, row_vectors, tl.num_programs(0) * BLOCK_VECTORS):
            flat_vectors(conv_row, bias_ptr, residual_ptr + contiguous_base, sum_ptr + contiguous_base, out_row,
                         alpha_scale_ptr, inverse_beta_ptr, start, row_vectors, channel_vectors, DTYPE, HAS_RESIDUAL,
                         WRITE_SUM, BLOCK_VECTORS, VECTOR)
    else:
        flat_vectors(conv_row, bias_ptr, residual_ptr + contiguous_base, sum_ptr + contiguous_base, out_row,
                     alpha_scale_ptr, inverse_beta_ptr, tl.program_id(0) * BLOCK_VECTORS, row_vectors,
                     channel_vectors, DTYPE, HAS_RESIDUAL, WRITE_SUM, BLOCK_VECTORS, VECTOR)


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
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    vector = VECTOR_BYTES // torch.empty(0, dtype=DTYPE).element_size()
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
            timings = {"shipped": graph_time_us(lambda: fused_bias_snake_beta(
                conv, bias, activation, shipped_out, residual=shipped_sum if has_residual else None,
                sum_out=shipped_sum if write_sum else None))}
            launches = {}
            positions, num_warps = rule("padded", channels)
            block_channels, chunks = tile_channels("padded", channels)
            padded_out, padded_sum = new_out(), residual.clone()
            launches["padded hoisted"] = (padded_out, padded_sum, lambda: layout_kernel[
                (triton.cdiv(length, positions), rows)](
                conv, conv.stride(0), bias, padded_sum, padded_sum, padded_out, padded_out.stride(0), alpha_scale,
                inverse_beta, channels, length, EPSILON, DTYPE=TRITON_DTYPES[DTYPE], HAS_RESIDUAL=has_residual,
                WRITE_SUM=write_sum, HOISTED=True, CHUNKS=chunks, BLOCK_POSITIONS=positions,
                BLOCK_CHANNELS=block_channels, num_warps=num_warps, enable_fp_fusion=False, enable_reflect_ftz=False))
            row_vectors = length * channels // vector
            facts = {}
            for name, persistent in (("flat", False), ("flat persistent", True)):
                out, total = new_out(), residual.clone()
                state = {"grid": triton.cdiv(row_vectors, THREADS)}

                def launch(out=out, total=total, state=state, persistent=persistent):
                    return flat_kernel[(state["grid"], rows)](
                        conv, conv.stride(0), bias, total, total, out, out.stride(0), alpha_scale, inverse_beta,
                        row_vectors, channels // vector, DTYPE=TRITON_DTYPES[DTYPE], HAS_RESIDUAL=has_residual,
                        WRITE_SUM=write_sum, PERSISTENT=persistent, BLOCK_VECTORS=THREADS, VECTOR=vector,
                        num_warps=THREADS // 32, enable_fp_fusion=False, enable_reflect_ftz=False,
                    )

                compiled = launch()
                if persistent:
                    occupancy, source = blocks_per_sm(compiled)
                    state["grid"] = min(state["grid"], max(1, triton.cdiv(sm_count * occupancy, rows)))
                else:
                    occupancy, source = None, None
                facts[name] = dict(grid=state["grid"], occupancy=occupancy, occupancy_source=source,
                                   **compiled_facts(compiled))
                launches[name] = (out, total, launch)
            equal = {}
            for name, (out, total, launch) in launches.items():
                total.copy_(residual)
                launch()
                equal[name] = bool(torch.equal(out, reference_out)
                                   and (not write_sum or torch.equal(total, reference_sum)))
                timings[name] = graph_time_us(launch)
            record = dict(rows=rows, name=names[key], channels=channels, length=length, residual=has_residual,
                          total=write_sum, count=count, us={k: round(v, 3) for k, v in timings.items()}, equal=equal,
                          facts=facts)
            records.append(record)
            print(f"rows {rows} {names[key]:26s} C {channels:5d} L {length:6d} x{count} | "
                  + " | ".join(f"{k} {v:6.2f}" for k, v in timings.items())
                  + f" | grid {facts['flat']['grid']} / {facts['flat persistent']['grid']} occ "
                  f"{facts['flat persistent']['occupancy']} ({facts['flat persistent']['occupancy_source']}) "
                  f"r{facts['flat']['registers']} | equal {all(equal.values())}", flush=True)
    (args.out / "flat.json").write_text(json.dumps(dict(sm_count=sm_count, records=records), indent=1))
    for rows in (1, 8):
        group = [r for r in records if r["rows"] == rows]
        print(f"rows {rows} (29 launches): " + ", ".join(
            f"{name} {sum(r['us'][name] * r['count'] for r in group):.1f} us" for name in group[0]["us"]))
        print(f"  all outputs equal: {all(all(r['equal'].values()) for r in group)}")


if __name__ == "__main__":
    main()
