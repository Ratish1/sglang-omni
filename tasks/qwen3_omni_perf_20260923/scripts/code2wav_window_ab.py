"""Whole-window A/B of Qwen3-Omni code2wav on the tree in PYTHONPATH (run on the box): the HF
module as main serves it against the tree's load_code2wav_model, both with the fused SnakeBeta
(the served default), each captured in a CUDA graph per window as the graph runner serves it.

Per window (batch 1 at the serial windows 10, 20, 30, 35; batch 2 at 35 for the batched path):
median replay time of 20, kernels per replay, and each arm's distance to an fp32 eager HF
forward of the same weights and codes (max abs, and L2 relative to the reference's norm), and
the distance between the arms.

usage: PYTHONPATH=<tree> python3 code2wav_window_ab.py [--model-path Qwen/Qwen3-Omni-30B-A3B-Instruct]
"""

from __future__ import annotations

import argparse

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
WINDOWS = ((1, 10), (1, 20), (1, 30), (1, 35), (2, 35))


def load_hf(model_path: str, dtype: torch.dtype) -> Qwen3OmniMoeCode2Wav:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)
    return load_module(
        model, model_path, prefix="code2wav.", dtype=dtype, device="cuda", strict=False
    ).eval()


def replay(model, codes: torch.Tensor):
    """Capture model(codes) as the graph runner does; returns (output, median us, kernels)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(codes)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(codes)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    times.sort()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        graph.replay()
        torch.cuda.synchronize()
    kernels = sum(event.device_type.name == "CUDA" for event in profile.events())
    return output.float().clone(), times[len(times) // 2], kernels


def distance(output: torch.Tensor, reference: torch.Tensor) -> str:
    max_abs = float((output - reference).abs().max())
    relative = float((output - reference).norm() / reference.norm())
    return f"max {max_abs:.1e} rel {relative:.1e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    args = parser.parse_args()
    # the fp32 reference convolves in fp32, not TF32; the bf16 arms are unaffected
    torch.backends.cudnn.allow_tf32 = False
    served = load_hf(args.model_path, torch.bfloat16)
    fuse_vocoder_decoder(served.decoder)
    candidate = load_code2wav_model(args.model_path, device="cuda", dtype="bfloat16")
    fuse_vocoder_decoder(candidate.decoder)
    reference = load_hf(args.model_path, torch.float32)
    codebook = int(served.config.codebook_size)
    print(torch.cuda.get_device_name(), type(candidate).__name__)
    torch.manual_seed(0)
    for batch, frames in WINDOWS:
        codes = torch.randint(0, codebook, (batch, QUANTIZERS, frames), device="cuda")
        with torch.inference_mode():
            expected = reference(codes).float()
            served_out, served_us, served_kernels = replay(served, codes)
            candidate_out, candidate_us, candidate_kernels = replay(candidate, codes)
        print(
            f"batch {batch} frames {frames:3d} | served {served_us:7.1f} us "
            f"{served_kernels:4d} kernels | candidate {candidate_us:7.1f} us "
            f"{candidate_kernels:4d} kernels ({(candidate_us / served_us - 1) * 100:+.1f} %)"
        )
        print(
            f"    vs fp32: served {distance(served_out, expected)}, candidate "
            f"{distance(candidate_out, expected)}; candidate vs served "
            f"{distance(candidate_out, served_out)}"
        )


if __name__ == "__main__":
    main()
