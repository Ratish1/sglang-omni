"""Every conv of one Qwen3-TTS incremental decode: its shape, the cuDNN kernels it runs,
and what a channels-last layout or another algorithm changes.

Loads the checkpoint's speech tokenizer as the vocoder stage does, runs two chained
incremental decodes of random codes (so conv histories exist), and records each
Conv1d call and each conv_transpose1d call with its input shape. Each recorded conv is
then run, in a CUDA graph of 10 calls, as served (NCL contiguous input), as a 4D
channels-last conv over (N, C, 1, L), and for dilated convs as one GEMM over the
concatenated taps and under cuDNN's benchmark search; each with its kernels, whether
it matches the served output bit for bit, and its distance to an fp32 conv.

usage: python vocoder_conv_census.py [--rows 16] [--frames 8]
"""

from __future__ import annotations

import argparse
import collections

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CALLS = 10


def graph_time_us(fn) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(20):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / CALLS)
    times.sort()
    return times[len(times) // 2]


def kernels(fn) -> list[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        if event.device_type.name == "CUDA":
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
                else:
                    pass
            names.append(name[:40])
        else:
            pass
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    from sglang_omni.models.qwen3_tts.incremental_codec import (
        Qwen3TTSIncrementalCodecState,
        Qwen3TTSIncrementalDecoder,
    )
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    names = {module: name for name, module in decoder.named_modules()}
    calls = collections.OrderedDict()

    def record(module, inputs):
        if names[module] not in calls:
            calls[names[module]] = ("conv", module, inputs[0].shape)
        else:
            pass

    handles = [
        module.register_forward_pre_hook(record)
        for module in decoder.modules()
        if isinstance(module, torch.nn.Conv1d)
    ]
    transpose = F.conv_transpose1d
    transconv_modules = {
        module.weight.data_ptr(): (name, module)
        for name, module in decoder.named_modules()
        if isinstance(module, torch.nn.ConvTranspose1d)
    }

    def recording_transpose(x, weight, *rest, **kwargs):
        name, module = transconv_modules[weight.data_ptr()]
        if name not in calls:
            calls[name] = ("transconv", module, x.shape)
        else:
            pass
        return transpose(x, weight, *rest, **kwargs)

    config = getattr(tokenizer.model.config, "decoder_config", tokenizer.model.config)
    num_quantizers = int(config.num_quantizers)
    codebook = int(config.codebook_size)
    codes = torch.randint(
        0, codebook, (args.rows, num_quantizers, args.frames), device="cuda"
    )
    state = Qwen3TTSIncrementalCodecState()
    F.conv_transpose1d = recording_transpose
    with torch.inference_mode():
        incremental.decode(codes, state)
        calls.clear()
        incremental.decode(codes, state)
    F.conv_transpose1d = transpose
    for handle in handles:
        handle.remove()

    print(
        f"rows {args.rows}, frames {args.frames}; us per call; kernels in launch order"
    )
    totals = collections.Counter()
    with torch.inference_mode():
        for name, (kind, module, shape) in calls.items():
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            weight = module.weight
            bias = module.bias
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
            else:
                output_padding = module.output_padding[0]
                variants = {
                    "served": lambda: F.conv_transpose1d(
                        x,
                        weight,
                        None,
                        stride,
                        padding,
                        output_padding,
                        groups,
                        dilation,
                    ),
                    "nhwc 4d": lambda: F.conv_transpose2d(
                        x4,
                        w4,
                        None,
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
                    None,
                    stride,
                    padding,
                    output_padding,
                    groups,
                    dilation,
                )
            if kind == "conv" and dilation > 1 and groups == 1:
                taps = module.kernel_size[0]
                length = shape[-1] - (taps - 1) * dilation
                folded = (
                    weight.permute(0, 2, 1).reshape(weight.shape[0], -1).contiguous()
                )

                def unfolded_gemm():
                    columns = torch.cat(
                        [
                            x[:, :, t * dilation : t * dilation + length]
                            for t in range(taps)
                        ],
                        dim=1,
                    )
                    return torch.matmul(folded, columns) + bias.view(1, -1, 1)

                variants["unfold gemm"] = unfolded_gemm

                def benchmarked():
                    torch.backends.cudnn.benchmark = True
                    output = F.conv1d(
                        x, weight, bias, stride, padding, dilation, groups
                    )
                    torch.backends.cudnn.benchmark = False
                    return output

                variants["cudnn benchmark"] = benchmarked
            else:
                pass
            print(
                f"{name:<34}{kind:<10} in {module.in_channels:>5} out {module.out_channels:>5} "
                f"k {module.kernel_size[0]} d {dilation} g {groups:>4} x {tuple(shape)}"
            )
            served = variants["served"]()
            for label, fn in variants.items():
                output = fn()
                time_us = graph_time_us(fn)
                totals[label] += time_us
                identical = torch.equal(output, served)
                error = float((output.float() - reference).abs().max())
                print(
                    f"    {label:<16}{time_us:9.1f}  identical {identical!s:<5} "
                    f"max abs vs fp32 {error:.1e}  {' '.join(kernels(fn))}"
                )
    print(
        "\nsum over the decode's convs: "
        + ", ".join(
            f"{label} {value:.1f} us" for label, value in totals.items() if value
        )
    )


if __name__ == "__main__":
    main()
