"""Every convolution of one Qwen3-Omni code2wav window: its shape, its cuDNN kernels, and what a
channels-last layout or a polyphase dilated conv changes (run on the box).

Known answer first: the whole window, model(codes) on [1, 16 quantizers, frames], captured in a
CUDA graph as the code2wav graph runner serves it, at the serial windows the default walk visits
(10, 20, 30 and 35 frames: chunk 10, left context 25), timed per replay; compare with the census's
code2wav replay times before reading anything else.

Then each Conv1d and ConvTranspose1d of the steady 35-frame window is recorded with its input
shape and run, in a CUDA graph of 10 calls, as served (NCL contiguous), as a 4D channels-last conv
over (N, C, 1, L), and for dilated convs with one group as a polyphase batch: the input's d phases
x[..., p::d] stacked on the batch and convolved undilated in one call, the outputs interleaved
back. Each variant: time, whether it equals the served output bit for bit, and its distance to an
fp32 conv of the same input.

usage: python3 code2wav_conv_census.py [--model-path Qwen/Qwen3-Omni-30B-A3B-Instruct]
"""

from __future__ import annotations

import argparse
import collections

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
    serial_window_frames,
)

CALLS = 10
QUANTIZERS = 16


def graph_time_us(fn, calls: int = CALLS) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
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
        times.append(start.elapsed_time(end) * 1000 / calls)
    times.sort()
    return times[len(times) // 2]


def kernel_names(fn) -> list[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        if event.device_type.name != "CUDA":
            continue
        name = event.name
        for short in (
            "nchwToNhwc",
            "nhwcToNchw",
            "implicit_convolve_sgemm",
            "xmma_fprop",
            "xmma_dgrad",
            "cutlass",
            "conv_depthwise",
            "elementwise",
        ):
            if short in name:
                name = short
                break
        names.append(name[:32])
    return names


def polyphase_conv(x, weight, bias, dilation: int):
    """A dilated conv (stride 1, no padding) as one undilated conv over the input's phases."""
    batch, channels, length = x.shape
    taps = weight.shape[-1]
    output_length = length - (taps - 1) * dilation
    phase_length = -(-length // dilation)
    padded = F.pad(x, (0, phase_length * dilation - length))
    phases = (
        padded.view(batch, channels, phase_length, dilation)
        .permute(0, 3, 1, 2)
        .reshape(batch * dilation, channels, phase_length)
    )
    out = F.conv1d(phases, weight, bias)
    out_channels, out_phase_length = out.shape[1], out.shape[2]
    return (
        out.view(batch, dilation, out_channels, out_phase_length)
        .permute(0, 2, 3, 1)
        .reshape(batch, out_channels, out_phase_length * dilation)[..., :output_length]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--left-context", type=int, default=25)
    args = parser.parse_args()
    model = load_code2wav_model(args.model_path, device="cuda", dtype="bfloat16")
    codebook = int(model.config.codebook_size)
    print(torch.cuda.get_device_name(), "codebook", codebook)
    windows = serial_window_frames(args.chunk, args.left_context)
    print(f"known answer, whole window replay (batch 1), windows {windows}:")
    with torch.inference_mode():
        for frames in windows:
            codes = torch.randint(0, codebook, (1, QUANTIZERS, frames), device="cuda")
            print(
                f"  frames {frames:3d}: {graph_time_us(lambda: model(codes), calls=1):8.1f} us"
            )

    names = {module: name for name, module in model.named_modules()}
    calls = collections.OrderedDict()

    def record(module, inputs):
        kind = "transconv" if isinstance(module, torch.nn.ConvTranspose1d) else "conv"
        calls.setdefault(names[module], (kind, module, tuple(inputs[0].shape)))

    handles = [
        module.register_forward_pre_hook(record)
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
    ]
    steady = windows[-1]
    codes = torch.randint(0, codebook, (1, QUANTIZERS, steady), device="cuda")
    with torch.inference_mode():
        model(codes)
    for handle in handles:
        handle.remove()

    print(f"\nconvolutions of the {steady}-frame window ({len(calls)}); us per call")
    per_conv = []
    with torch.inference_mode():
        for name, (kind, module, shape) in calls.items():
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            weight, bias = module.weight, module.bias
            stride, padding = module.stride[0], module.padding[0]
            dilation, groups = module.dilation[0], module.groups
            x4 = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
            w4 = weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
            if kind == "conv":
                variants = {
                    "served": lambda: F.conv1d(
                        x, weight, bias, stride, padding, dilation, groups
                    ),
                    "nhwc 4d": lambda: F.conv2d(
                        x4, w4, bias, (1, stride), (0, padding), (1, dilation), groups
                    ).squeeze(2),
                }
                reference = F.conv1d(
                    x.float(),
                    weight.float(),
                    None if bias is None else bias.float(),
                    stride,
                    padding,
                    dilation,
                    groups,
                )
                if dilation > 1 and groups == 1 and stride == 1 and padding == 0:
                    variants["polyphase"] = lambda: polyphase_conv(
                        x, weight, bias, dilation
                    )
            else:
                output_padding = module.output_padding[0]
                variants = {
                    "served": lambda: F.conv_transpose1d(
                        x,
                        weight,
                        bias,
                        stride,
                        padding,
                        output_padding,
                        groups,
                        dilation,
                    ),
                    "nhwc 4d": lambda: F.conv_transpose2d(
                        x4,
                        w4,
                        bias,
                        (1, stride),
                        (0, padding),
                        (0, output_padding),
                        groups,
                        (1, dilation),
                    ).squeeze(2),
                }
                reference = F.conv_transpose1d(
                    x.float(),
                    weight.float(),
                    None if bias is None else bias.float(),
                    stride,
                    padding,
                    output_padding,
                    groups,
                    dilation,
                )
            print(
                f"{name:<44}{kind:<10} in {module.in_channels:>5} out {module.out_channels:>5} "
                f"k {module.kernel_size[0]} s {stride} d {dilation} g {groups:>5} x {shape}"
            )
            served = variants["served"]()
            times = {}
            for label, fn in variants.items():
                output = fn()
                time_us = graph_time_us(fn)
                times[label] = time_us
                identical = torch.equal(output, served)
                error = float((output.float() - reference).abs().max())
                print(
                    f"    {label:<12}{time_us:9.1f}  identical {identical!s:<5} "
                    f"max abs vs fp32 {error:.1e}  {' '.join(kernel_names(fn))}"
                )
            per_conv.append(times)
    served_total = sum(times["served"] for times in per_conv)
    nhwc_total = sum(times["nhwc 4d"] for times in per_conv)
    best_total = sum(min(times.values()) for times in per_conv)
    print(
        f"\nsum over the window's convs: served {served_total:.1f} us, all nhwc 4d "
        f"{nhwc_total:.1f} us, fastest variant per conv {best_total:.1f} us"
    )


if __name__ == "__main__":
    main()
