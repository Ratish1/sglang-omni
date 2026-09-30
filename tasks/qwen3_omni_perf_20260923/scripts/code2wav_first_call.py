"""First and second eager call of Qwen3-Omni code2wav at window lengths it has not seen (the
tail windows that miss every graph key), main's HF module against the tree's load_code2wav_model,
both with the fused SnakeBeta. A first call pays whatever per-shape setup the convs need (cuDNN
execution plans, benchmark searches); a second call at the same length is the steady cost.

usage: PYTHONPATH=<tree> python3 code2wav_first_call.py [--frames 23 27 31] [--benchmark]
"""

from __future__ import annotations

import argparse
import time

import torch
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--frames", type=int, nargs="+", default=[23, 27, 31, 26])
    parser.add_argument("--benchmark", action="store_true")
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
    # a warm call at a length outside the list, so one-time process setup is not charged
    warm = torch.randint(0, codebook, (1, QUANTIZERS, 10), device="cuda")
    timed_ms(served, warm)
    timed_ms(candidate, warm)
    print(torch.cuda.get_device_name(), "cudnn.benchmark", args.benchmark)
    for frames in args.frames:
        codes = torch.randint(0, codebook, (1, QUANTIZERS, frames), device="cuda")
        row = [f"frames {frames:3d}"]
        for name, model in (("served", served), ("candidate", candidate)):
            first = timed_ms(model, codes)
            second = timed_ms(model, codes)
            row.append(f"{name} first {first:8.1f} ms second {second:6.1f} ms")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
