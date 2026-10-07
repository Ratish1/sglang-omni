# SPDX-License-Identifier: Apache-2.0
"""The DiT positional conv (grouped Conv1d, kernel 31, 16 groups, then Mish) per call at the
frame counts the packed and prefix paths feed it: the module on a transposed view, as the
packed paths call it, against the fused Triton kernel on the channels last frames. CUDA
event time per call, median of 50 after 10 warmup calls, and each call's relative error
against float64.

    cd <tree with causal_conv.py> && python conv_bench.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.causal_conv import (
    group_conv_mish,
    pack_group_conv_weight,
)


def event_ms(call, *, calls: int, warmup: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(calls):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[90, 160, 260, 760, 1660, 3260, 6460, 12160],
    )
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    conv = (
        torch.nn.Sequential(torch.nn.Conv1d(1024, 1024, 31, groups=16), torch.nn.Mish())
        .cuda()
        .to(torch.bfloat16)
    )
    packed = pack_group_conv_weight(conv[0])
    rows = []
    with torch.inference_mode():
        for frames in args.frames:
            x = torch.randn(frames, 1024, device="cuda", dtype=torch.bfloat16)
            float64 = F.mish(
                F.conv1d(
                    x.double().T.unsqueeze(0),
                    conv[0].weight.double(),
                    conv[0].bias.double(),
                    groups=16,
                )
            )[0].T

            def module():
                return conv(x.T.unsqueeze(0))[0].T

            def fused():
                return group_conv_mish(x.unsqueeze(0), packed, conv[0].bias)[0]

            row = {"frames": frames}
            for name, call in (("module", module), ("fused", fused)):
                row[f"{name}_ms"] = event_ms(call, calls=args.calls, warmup=args.warmup)
                out = call()
                row[f"{name}_error"] = float(
                    torch.linalg.vector_norm(out.double() - float64)
                    / torch.linalg.vector_norm(float64)
                )
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
