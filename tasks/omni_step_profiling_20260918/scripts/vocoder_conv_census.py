"""Every conv of one Qwen3-TTS incremental decode: its shape, the cuDNN kernels it runs,
and what a channels-last input and weight change.

Loads the checkpoint's speech tokenizer as the vocoder stage does, runs two chained
incremental decodes of random codes (so conv histories exist), and records each
Conv1d call and each conv_transpose1d call with its input shape. Each recorded conv
is then timed as served (NCL contiguous input, the module's weight) and with the input
and weight laid out channels last (strides of NLC over the same shape), in a CUDA
graph of 10 calls, with the kernels each launches and whether the outputs match bit
for bit.

usage: python vocoder_conv_census.py [--rows 16] [--frames 8]
"""

from __future__ import annotations

import argparse
import collections

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CALLS = 10


def channels_last(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2).contiguous().transpose(1, 2)


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
            if kind == "conv":

                def run(x, w):
                    return F.conv1d(
                        x,
                        w,
                        bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )

            else:

                def run(x, w):
                    return F.conv_transpose1d(
                        x,
                        w,
                        None,
                        module.stride,
                        module.padding,
                        module.output_padding,
                        module.groups,
                        module.dilation,
                    )

            x_cl = channels_last(x)
            w_cl = channels_last(weight)
            served = run(x, weight)
            layout = run(x_cl, w_cl)
            identical = torch.equal(served, layout)
            served_us = graph_time_us(lambda: run(x, weight))
            layout_us = graph_time_us(lambda: run(x_cl, w_cl))
            totals["served"] += served_us
            totals["channels last"] += layout_us
            print(
                f"{name:<34}{kind:<10} in {module.in_channels:>5} out {module.out_channels:>5} "
                f"k {module.kernel_size[0]} d {module.dilation[0]} g {module.groups:>4} "
                f"x {tuple(shape)}"
            )
            print(
                f"    served        {served_us:8.1f}  {' '.join(kernels(lambda: run(x, weight)))}"
            )
            print(
                f"    channels last {layout_us:8.1f}  {' '.join(kernels(lambda: run(x_cl, w_cl)))}"
                f"  bit identical {identical}"
            )
    print(
        f"\nsum over the decode's convs: served {totals['served']:.1f} us, "
        f"channels last {totals['channels last']:.1f} us"
    )


if __name__ == "__main__":
    main()
