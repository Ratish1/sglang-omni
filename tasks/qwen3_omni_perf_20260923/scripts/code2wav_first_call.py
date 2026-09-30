"""First and second eager call of Qwen3-Omni code2wav at window lengths it has not seen (the
tail windows that miss every graph key), main's HF module against the tree's load_code2wav_model,
both with the fused SnakeBeta. A first call pays whatever per-shape setup the convs need (cuDNN
execution plans, benchmark searches); a second call at the same length is the steady cost.

--per-conv: every conv of a window at the first length, each form timed on its first and second
call (served NCL, channels last (1 x k), phase width (k x 1) over (B, C, L / d, d)), summed per form
and per conv kind, to say which setup the first-call time is.

--arm served|candidate loads one arm only, so no setup one arm pays (the shared transformer's, a
shared conv shape's) is charged to the other; compare two processes.

usage: PYTHONPATH=<tree> python3 code2wav_first_call.py [--frames 23 27 31] [--benchmark] [--per-conv]
       [--arm served|candidate|both]
"""

from __future__ import annotations

import argparse
import collections
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)
from sglang_omni.models.weight_loader import load_module
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

QUANTIZERS = 16


def timed_ms(model, codes: torch.Tensor) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        model(codes)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def timed_call_ms(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def per_conv(model, frames: int) -> None:
    """First and second call per conv form at the inputs of one unseen window length."""
    calls = []

    def record(module, inputs):
        calls.append((module, inputs[0].shape))

    handles = [
        module.register_forward_pre_hook(record)
        for module in model.modules()
        if isinstance(module, torch.nn.Conv1d)
    ]
    codebook = int(model.config.codebook_size)
    with torch.inference_mode():
        model(torch.randint(0, codebook, (1, QUANTIZERS, frames), device="cuda"))
    for handle in handles:
        handle.remove()
    totals = collections.defaultdict(float)
    with torch.inference_mode():
        for module, shape in calls:
            weight, bias = module.weight, module.bias
            dilation, groups = module.dilation[0], module.groups
            kind = (
                "dilated"
                if dilation > 1
                else ("depthwise" if groups > 1 else "undilated")
            )
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            x_blc = x.transpose(1, 2).contiguous()
            weight_cl = weight.transpose(1, 2).contiguous().transpose(1, 2)
            length = shape[-1]
            forms = {
                "served": lambda: F.conv1d(x, weight, bias, 1, 0, dilation, groups),
                "1xk": lambda: F.conv2d(
                    x_blc.transpose(1, 2).unsqueeze(2),
                    weight_cl.unsqueeze(2),
                    bias,
                    dilation=(1, dilation),
                    groups=groups,
                ),
            }
            if dilation > 1:
                padded = F.pad(x_blc, (0, 0, 0, (-length) % dilation))
                forms["kx1"] = lambda: F.conv2d(
                    padded.view(1, -1, dilation, shape[1]).permute(0, 3, 1, 2),
                    weight_cl.unsqueeze(3),
                    bias,
                    groups=groups,
                )
            else:
                pass
            for form, fn in forms.items():
                totals[(form, kind, "first")] += timed_call_ms(fn)
                totals[(form, kind, "second")] += timed_call_ms(fn)
    print(
        f"per conv at an unseen {frames}-frame window (ms summed over the window's convs):"
    )
    for (form, kind, call), value in sorted(totals.items()):
        print(f"  {form:<8}{kind:<11}{call:<8}{value:8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--frames", type=int, nargs="+", default=[23, 27, 31, 26])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--per-conv", action="store_true")
    parser.add_argument(
        "--arm", choices=("served", "candidate", "both"), default="both"
    )
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.benchmark
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    served = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)
    served = load_module(
        served,
        args.model_path,
        prefix="code2wav.",
        dtype=torch.bfloat16,
        device="cuda",
        strict=False,
    ).eval()
    fuse_vocoder_decoder(served.decoder)
    candidate = load_code2wav_model(args.model_path, device="cuda", dtype="bfloat16")
    fuse_vocoder_decoder(candidate.decoder)
    codebook = int(served.config.codebook_size)
    if args.per_conv:
        per_conv(served, args.frames[0])
        return
    else:
        pass
    arms = {"served": served, "candidate": candidate}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}
    else:
        arms = {"candidate": candidate, "served": served}
    # a warm call at a length outside the list, so one-time process setup is not charged
    warm = torch.randint(0, codebook, (1, QUANTIZERS, 10), device="cuda")
    for model in arms.values():
        timed_ms(model, warm)
    print(torch.cuda.get_device_name(), "cudnn.benchmark", args.benchmark)
    for frames in args.frames:
        codes = torch.randint(0, codebook, (1, QUANTIZERS, frames), device="cuda")
        row = [f"frames {frames:3d}"]
        for name, model in arms.items():
            first = timed_ms(model, codes)
            second = timed_ms(model, codes)
            row.append(f"{name} first {first:8.1f} ms second {second:6.1f} ms")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
