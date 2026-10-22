"""The Qwen3-TTS vocoder's last conv (96 channels to 1, kernel 7) at the shapes the conv engine probe found it on
cuDNN's fallback engine, with the trained weight in bf16, each variant captured in a CUDA graph and timed by CUDA
events between back-to-back replays (median, p10, p90), its kernels named, and its bf16 output against the fp64 conv:
max abs error, mean abs error, and rows that differ from today's call.

Variants: today (the (B, L, C) activation as a channels-last conv2d, the chain's call), pad8 (the weight zero-padded
to 8 output channels, channel 0 read back), today and pad8 under cudnn.benchmark, and ncl (main's call: a contiguous
(B, C, L) conv1d).

usage: PYTHONPATH=<tree> python3 final_conv_bench.py [--card 2] [--replays 200]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.channels_last_conv import channels_last_strides

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SHAPES = ((1, 1926), (1, 15366), (4, 15366), (8, 15366), (1, 21126), (8, 21126))


def channels_last_call(weight: torch.Tensor):
    def call(hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.conv2d(hidden_states.transpose(1, 2).unsqueeze(2), weight.unsqueeze(2))
        return output.squeeze(2).transpose(1, 2)[..., :1]

    return call


def ncl_call(weight: torch.Tensor):
    def call(hidden_states: torch.Tensor) -> torch.Tensor:
        return F.conv1d(hidden_states, weight).transpose(1, 2)

    return call


def time_variant(call, hidden_states, replays: int, device) -> tuple[torch.Tensor, float, float, float, list[str]]:
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(hidden_states)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call(hidden_states)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(replays + 1)]
    events[0].record()
    for index in range(replays):
        graph.replay()
        events[index + 1].record()
    torch.cuda.synchronize(device)
    times_us = sorted(1000 * events[i].elapsed_time(events[i + 1]) for i in range(replays))
    quantiles = statistics.quantiles(times_us, n=10)
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize(device)
    kernels = [event.name[:70] for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    return output.clone(), statistics.median(times_us), quantiles[0], quantiles[-1], kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    weight = tokenizer.model.decoder.decoder[-1].conv.weight.data.to(torch.bfloat16)
    out_channels, in_channels, kernel_size = weight.shape
    print(f"final conv weight {tuple(weight.shape)}")
    padded = torch.zeros(8, in_channels, kernel_size, device=device, dtype=weight.dtype)
    padded[:out_channels].copy_(weight)
    variants = (
        ("today", False, "blc", channels_last_call(channels_last_strides(weight))),
        ("pad8", False, "blc", channels_last_call(channels_last_strides(padded))),
        ("today+benchmark", True, "blc", channels_last_call(channels_last_strides(weight))),
        ("pad8+benchmark", True, "blc", channels_last_call(channels_last_strides(padded))),
        ("ncl", False, "bcl", ncl_call(weight.contiguous())),
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.inference_mode():
        for batch, padded_length in SHAPES:
            hidden_states = torch.randn(batch, padded_length, in_channels, generator=generator).to(device, torch.bfloat16)
            exact = F.conv1d(hidden_states.double().transpose(1, 2), weight.double()).transpose(1, 2)
            baseline = None
            print(f"== batch {batch} padded length {padded_length}")
            for name, benchmark, layout, call in variants:
                torch.backends.cudnn.benchmark = benchmark
                inputs = hidden_states if layout == "blc" else hidden_states.transpose(1, 2).contiguous()
                output, median, p10, p90, kernels = time_variant(call, inputs, args.replays, device)
                error = (output.double() - exact).abs()
                if baseline is None:
                    baseline = output
                else:
                    pass
                differ = int((output != baseline).sum())
                print(
                    f"  {name:16s} median {median:7.1f} us p10 {p10:7.1f} p90 {p90:7.1f}  "
                    f"max err {error.max().item():.3e} mean err {error.mean().item():.3e}  "
                    f"rows differing from today {differ} of {output.numel()}"
                )
                print(f"    kernels: {' | '.join(kernels)}")
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
