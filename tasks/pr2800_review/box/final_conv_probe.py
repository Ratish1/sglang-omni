"""code2wav's final conv (96 channels to 1, kernel 7) on the slot decode's padded input: four ways, timed and checked.

Run in the container from a tree: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/final_conv_probe.py --out /data/pr2800/<run>
cuDNN 9.24 runs this conv on its SIMT convolve_common_engine_float_NHWC (412 us at 8 rows against 38 us for 9.20's
sm80 xmma engine, run205 against run211). For rows 1, 2, 4, 8, 16 and a 10-frame window (19200 new samples after 6
history samples), on the checkpoint's own decoder[-1] weight and bias:
  cudnn            channels_last_conv1d as the slot decode calls it (cuDNN's heuristics)
  cudnn_benchmark  the same call with cudnn.benchmark on (its autotuned engine)
  cudnn_nchw       a contiguous (B, 96, L) copy and F.conv1d
  triton           one program a row block, the 7 taps over the 96 channels accumulated in fp32, the bias added in
                   fp32, rounded once to the output dtype
Each is timed as code2wav_kernel_sweep.py times kernels (20 calls in one CUDA graph, 50 replays) and checked against an
fp32 conv of the same bf16 values: relative L2, max abs, and whether its bits equal the cudnn call's.
Writes <out>/final_conv.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.utils.channels_last_conv import channels_last_conv1d, channels_last_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import graph_time_us  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
NEW_SAMPLES = 19200
HISTORY = 6


@triton.jit(do_not_specialize=["length"])
def single_output_conv_kernel(
    x_ptr, x_batch_stride, w_ptr, bias_ptr, out_ptr, length, channels, DTYPE: tl.constexpr, TAPS: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    channel = tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channel < channels
    position_mask = positions < length
    accumulator = tl.zeros((BLOCK_POSITIONS,), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        weight = tl.load(w_ptr + channel * TAPS + tap, mask=channel_mask, other=0).to(tl.float32)
        values = tl.load(x_ptr + batch * x_batch_stride + (positions + tap)[:, None] * channels + channel[None, :],
                         mask=position_mask[:, None] & channel_mask[None, :], other=0).to(tl.float32)
        accumulator += tl.sum(values * weight[None, :], axis=1)
    accumulator += tl.load(bias_ptr).to(tl.float32)
    tl.store(out_ptr + batch * length + positions, accumulator.to(DTYPE), mask=position_mask)


def load_final_conv() -> torch.nn.Conv1d:
    model = load_code2wav_model(MODEL, device="cuda", dtype="bfloat16")
    return model.decoder[-1].conv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    conv = load_final_conv()
    weight = conv.weight.data
    cudnn_weight = channels_last_weight(conv)
    taps_weight = weight[0].contiguous()  # (96, 7)
    records = []
    for rows in (1, 2, 4, 8, 16):
        padded = torch.randn(rows, HISTORY + NEW_SAMPLES, weight.shape[1], device="cuda", dtype=weight.dtype) * 0.3
        reference = F.conv1d(padded.transpose(1, 2).float(), weight.float(), conv.bias.float())[:, 0]
        out = torch.empty(rows, NEW_SAMPLES, device="cuda", dtype=weight.dtype)

        def cudnn():
            return channels_last_conv1d(padded, conv, cudnn_weight, NEW_SAMPLES)[..., 0]

        def cudnn_nchw():
            return F.conv1d(padded.transpose(1, 2).contiguous(), weight, conv.bias)[:, 0]

        def run_triton():
            single_output_conv_kernel[(triton.cdiv(NEW_SAMPLES, 256), rows)](
                padded, padded.stride(0), taps_weight, conv.bias, out, NEW_SAMPLES, weight.shape[1],
                DTYPE=tl.bfloat16, TAPS=weight.shape[2], BLOCK_POSITIONS=256, BLOCK_CHANNELS=128, num_warps=8,
            )
            return out

        baseline = cudnn().clone()
        variants = {"cudnn": cudnn, "cudnn_nchw": cudnn_nchw, "triton": run_triton}
        result = dict(rows=rows, variants={})
        for name, launch in variants.items():
            value = launch().clone()
            result["variants"][name] = dict(
                us=round(graph_time_us(launch), 3),
                rel_l2=float((value.float() - reference).norm() / reference.norm()),
                max_abs=float((value.float() - reference).abs().max()),
                bits_equal_cudnn=bool(torch.equal(value, baseline)),
            )
        with torch.backends.cudnn.flags(enabled=True, benchmark=True):
            value = cudnn().clone()
            result["variants"]["cudnn_benchmark"] = dict(
                us=round(graph_time_us(cudnn), 3),
                rel_l2=float((value.float() - reference).norm() / reference.norm()),
                max_abs=float((value.float() - reference).abs().max()),
                bits_equal_cudnn=bool(torch.equal(value, baseline)),
            )
        records.append(result)
        print(f"rows {rows:2d}: " + " | ".join(
            f"{name} {v['us']:7.2f} us rel {v['rel_l2']:.2e} max {v['max_abs']:.2e} same bits {v['bits_equal_cudnn']}"
            for name, v in result["variants"].items()), flush=True)
    (args.out / "final_conv.json").write_text(json.dumps(dict(
        cudnn=torch.backends.cudnn.version(), torch=torch.__version__, records=records), indent=1))


if __name__ == "__main__":
    main()
