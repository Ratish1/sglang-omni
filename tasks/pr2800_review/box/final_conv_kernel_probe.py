"""run214: the shipped final conv kernel (clamped_single_output_conv) against its launch alternatives, run212's
prototype and cuDNN, on the checkpoint's decoder[-1], timed and checked.

Run in the container from the kernel's tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/final_conv_kernel_probe.py --out /data/pr2800/<run>
For rows 1, 2, 4, 8, 16 at a later window (19200 new samples) and a stream's first (18645), after 6 history samples:
  shipped        clamped_single_output_conv as both code2wav paths call it (64 positions and 8 warps at 96 bf16 channels)
  positions128   the same kernel, 128 positions a program, 8 warps
  positions256   256 positions, 8 warps
  warps4         64 positions, 4 warps
  prototype      run212's single_output_conv_kernel, then the clamp
  cudnn          channels_last_conv1d with the bias, then the clamp, as main runs the final conv
Each is timed as code2wav_kernel_sweep.py times kernels (20 calls in one CUDA graph, 50 replays): warm on one input,
and cold over inputs that together exceed 128 MiB so each call reads from HBM; the bytes read over the cold time. Each
is checked against an fp64 conv of the same values, clamped: relative L2, max abs, and the share of outputs equal to
that result rounded once. The shipped kernel's outputs are compared bit for bit with the window split at 7777 samples
and with each row decoded alone. Writes <out>/final_conv_kernel.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang_omni.kernels.single_output_conv import clamped_single_output_conv, clamped_single_output_conv_kernel
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.utils.channels_last_conv import channels_last_conv1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import graph_time_us  # noqa: E402
from final_conv_probe import single_output_conv_kernel  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
HISTORY = 6
LENGTHS = {"next": 19200, "first": 18645}
COLD_BYTES = 128 * 2**20
SPLIT = 7777
VARIANTS = {"positions128": (128, 8), "positions256": (256, 8), "warps4": (64, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    conv = load_code2wav_model(MODEL, device="cuda", dtype="bfloat16").decoder[-1].conv
    channels = conv.in_channels
    taps_weight = conv.weight.data[0].contiguous()
    registers = {}
    records = []
    for decode, length in LENGTHS.items():
        for rows in (1, 2, 4, 8, 16):
            buffer_bytes = rows * (length + HISTORY) * channels * 2
            inputs = [
                torch.randn(rows, length + HISTORY, channels, device="cuda", dtype=torch.bfloat16) * 0.3
                for _ in range(max(1, math.ceil(COLD_BYTES / buffer_bytes)))
            ]
            padded = inputs[0]
            reference = F.conv1d(padded.double().transpose(1, 2), conv.weight.double(), conv.bias.double()).clamp(
                min=-1, max=1)[:, 0]
            rounded = reference.to(torch.bfloat16)
            prototype_out = torch.empty(rows, length, device="cuda", dtype=torch.bfloat16)

            def variant(block_positions: int, num_warps: int):
                def launch(source: torch.Tensor) -> torch.Tensor:
                    output = source.new_empty(rows, 1, length)
                    handle = clamped_single_output_conv_kernel[(triton.cdiv(length, block_positions), rows)](
                        source, source.stride(0), conv.weight, conv.weight.stride(1), conv.weight.stride(2), conv.bias,
                        output, length, CHANNELS=channels, TAPS=7, BLOCK_POSITIONS=block_positions,
                        BLOCK_CHANNELS=channels & -channels, num_warps=num_warps,
                    )
                    registers[f"{block_positions}x{num_warps}"] = dict(regs=handle.n_regs, spills=handle.n_spills)
                    return output
                return launch

            def prototype(source: torch.Tensor) -> torch.Tensor:
                single_output_conv_kernel[(triton.cdiv(length, 256), rows)](
                    source, source.stride(0), taps_weight, conv.bias, prototype_out, length, channels,
                    DTYPE=tl.bfloat16, TAPS=7, BLOCK_POSITIONS=256, BLOCK_CHANNELS=128, num_warps=8,
                )
                return prototype_out.clamp(min=-1, max=1)

            def cudnn(source: torch.Tensor) -> torch.Tensor:
                return channels_last_conv1d(source, conv, conv.weight, conv.bias, length).transpose(1, 2).clamp(
                    min=-1, max=1)

            launches = {
                "shipped": lambda source: clamped_single_output_conv(source, conv, length),
                **{name: variant(*launch) for name, launch in VARIANTS.items()},
                "prototype": prototype,
                "cudnn": cudnn,
            }
            variant(64, 8)(padded)
            result = dict(decode=decode, rows=rows, length=length, variants={})
            for name, launch in launches.items():
                value = launch(padded).reshape(rows, length)
                turn = iter(range(10**9))
                cold_us = graph_time_us(lambda: launch(inputs[next(turn) % len(inputs)]))
                result["variants"][name] = dict(
                    warm_us=round(graph_time_us(lambda: launch(padded)), 3),
                    cold_us=round(cold_us, 3),
                    cold_gbps=round(buffer_bytes / cold_us / 1e3, 1),
                    rel_l2=float((value.double() - reference).norm() / reference.norm()),
                    max_abs=float((value.double() - reference).abs().max()),
                    rounded_share=float((value == rounded).float().mean()),
                )
            whole = clamped_single_output_conv(padded, conv, length)
            parts = torch.cat([clamped_single_output_conv(padded[:, : SPLIT + HISTORY], conv, SPLIT),
                               clamped_single_output_conv(padded[:, SPLIT:], conv, length - SPLIT)], dim=-1)
            result["split_equal"] = bool(torch.equal(parts, whole))
            result["rows_alone_equal"] = all(
                torch.equal(clamped_single_output_conv(padded[row : row + 1], conv, length), whole[row : row + 1])
                for row in range(rows))
            records.append(result)
            print(f"{decode} rows {rows:2d} split {result['split_equal']} alone {result['rows_alone_equal']}: " + " | ".join(
                f"{name} warm {v['warm_us']:7.2f} cold {v['cold_us']:7.2f} us {v['cold_gbps']:6.0f} GB/s "
                f"rel {v['rel_l2']:.2e} rounded {v['rounded_share']:.4f}"
                for name, v in result["variants"].items()), flush=True)
    print(f"registers {registers}", flush=True)
    (args.out / "final_conv_kernel.json").write_text(json.dumps(dict(
        cudnn=torch.backends.cudnn.version(), torch=torch.__version__, triton=triton.__version__,
        registers=registers, records=records), indent=1))


if __name__ == "__main__":
    main()
