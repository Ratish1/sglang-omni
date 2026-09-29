"""Every vocoder conv at the served decode shapes, as served and as a channels-last
(B, L, C) conv, with the dilated convs' alternatives.

The served vocoder captures widths 1, 2 and 8 frames at cohort buckets 1, 2, 4 and 8.
For each of those twelve shapes this records every Conv1d and conv_transpose1d input
of a real incremental decode, then runs each conv:

- served: conv1d / conv_transpose1d on the NCL input, as today;
- nlc: the (B, L, C) input viewed as (B, C, 1, L) channels last against a channels-last
  weight, conv2d / conv_transpose2d, the layout the channels-last decoder runs;
- for dilated convs also:
  - polyphase: the d decimated sequences as a batch of non-dilated convs;
  - unfold: one GEMM over the k shifted copies of the input;
  - nlc search: cuDNN's benchmark over every engine (benchmark_limit 0).

Each line gives us per call from a CUDA graph of 10 calls, the main kernel, and the
max abs difference against an fp32 conv. A cuDNN direct kernel (tens of ms) is timed
with one call.

usage: python vocoder_nlc_sweep.py
"""

from __future__ import annotations

import collections

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WIDTHS = (1, 2, 8)
BUCKETS = (1, 2, 4, 8)
CALLS = 10
SLOW_KERNELS = ("direct", "implicit_convolve")


def main_kernel(fn) -> str:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in prof.events()
        if event.device_type.name == "CUDA"
        and "elementwise" not in event.name
        and "Memset" not in event.name
        and "Cat" not in event.name
        and "copy" not in event.name.lower()
    ]
    return max(names, key=len)[:30] if names else "-"


def time_us(fn, slow: bool) -> float:
    if slow:
        fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000
    else:
        pass
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
    for _ in range(10):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / CALLS)
    times.sort()
    return times[len(times) // 2]


def channels_last_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)


def main() -> None:
    from sglang_omni.models.qwen3_tts.incremental_codec import (
        Qwen3TTSIncrementalCodecState,
        Qwen3TTSIncrementalDecoder,
    )
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    config = getattr(tokenizer.model.config, "decoder_config", tokenizer.model.config)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    names = {module: name for name, module in decoder.named_modules()}
    transconvs = {
        module.weight.data_ptr(): (name, module)
        for name, module in decoder.named_modules()
        if isinstance(module, torch.nn.ConvTranspose1d)
    }
    transpose = F.conv_transpose1d
    totals = collections.defaultdict(float)
    torch.manual_seed(0)
    for frames in WIDTHS:
        for rows in BUCKETS:
            calls = collections.OrderedDict()

            def record(module, inputs):
                calls.setdefault(names[module], ("conv", module, inputs[0].shape))

            def recording_transpose(x, weight, *rest, **kwargs):
                name, module = transconvs[weight.data_ptr()]
                calls.setdefault(name, ("transconv", module, x.shape))
                return transpose(x, weight, *rest, **kwargs)

            handles = [
                module.register_forward_pre_hook(record)
                for module in decoder.modules()
                if isinstance(module, torch.nn.Conv1d)
            ]
            codes = torch.randint(
                0,
                int(config.codebook_size),
                (rows, int(config.num_quantizers), frames),
                device="cuda",
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

            print(f"\nwidth {frames} frames, bucket {rows}")
            with torch.inference_mode():
                for name, (kind, module, shape) in calls.items():
                    if name.startswith("quantizer"):
                        continue
                    else:
                        pass
                    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    x_nlc = x.transpose(1, 2).contiguous()
                    x4 = x_nlc.transpose(1, 2).unsqueeze(2)
                    weight, bias = module.weight, module.bias
                    w4 = channels_last_weight(weight)
                    stride, padding = module.stride[0], module.padding[0]
                    dilation, groups = module.dilation[0], module.groups
                    taps = module.kernel_size[0]
                    if kind == "conv":
                        reference = F.conv1d(
                            x.float(),
                            weight.float(),
                            None if bias is None else bias.float(),
                            stride,
                            padding,
                            dilation,
                            groups,
                        )
                        variants = {
                            "served": lambda: F.conv1d(
                                x, weight, bias, stride, padding, dilation, groups
                            ),
                            "nlc": lambda: F.conv2d(
                                x4, w4, bias, 1, (0, padding), (1, dilation), groups
                            )
                            .squeeze(2)
                            .transpose(1, 2),
                        }
                    else:
                        output_padding = module.output_padding[0]
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
                            "nlc": lambda: F.conv_transpose2d(
                                x4,
                                w4,
                                None,
                                (1, stride),
                                (0, padding),
                                (0, output_padding),
                                groups,
                                (1, dilation),
                            )
                            .squeeze(2)
                            .transpose(1, 2),
                        }
                    if kind == "conv" and dilation > 1:
                        length = shape[-1]
                        out_len = length - (taps - 1) * dilation
                        pad = (-length) % dilation
                        cout = weight.shape[0]
                        folded = weight.permute(0, 2, 1).reshape(cout, -1).t()

                        def polyphase():
                            padded = F.pad(x_nlc, (0, 0, 0, pad))
                            phases = (
                                padded.view(rows, -1, dilation, shape[1])
                                .transpose(1, 2)
                                .reshape(rows * dilation, -1, shape[1])
                            )
                            y = (
                                F.conv2d(phases.transpose(1, 2).unsqueeze(2), w4, bias)
                                .squeeze(2)
                                .transpose(1, 2)
                            )
                            y = (
                                y.reshape(rows, dilation, -1, cout)
                                .transpose(1, 2)
                                .reshape(rows, -1, cout)
                            )
                            return y[:, :out_len]

                        def unfold():
                            columns = torch.cat(
                                [
                                    x_nlc[:, t * dilation : t * dilation + out_len]
                                    for t in range(taps)
                                ],
                                dim=2,
                            )
                            return torch.addmm(
                                bias, columns.view(-1, columns.shape[-1]), folded
                            ).view(rows, out_len, cout)

                        def searched():
                            torch.backends.cudnn.benchmark = True
                            torch.backends.cudnn.benchmark_limit = 0
                            output = variants["nlc"]()
                            torch.backends.cudnn.benchmark = False
                            torch.backends.cudnn.benchmark_limit = 10
                            return output

                        variants["polyphase"] = polyphase
                        variants["unfold"] = unfold
                        variants["nlc search"] = searched
                    else:
                        pass
                    cells = []
                    for label, fn in variants.items():
                        kernel = main_kernel(fn)
                        slow = any(word in kernel for word in SLOW_KERNELS)
                        elapsed = time_us(fn, slow)
                        output = fn().float()
                        if label == "served":
                            error = (output - reference).abs().max()
                        else:
                            error = (output - reference.transpose(1, 2)).abs().max()
                        totals[(frames, rows, label)] += elapsed
                        flag = " SLOW" if slow else ""
                        cells.append(
                            f"{label} {elapsed:9.1f}{flag} {kernel} err {float(error):.1e}"
                        )
                    print(
                        f"  {name:<32} {kind:<9} d{dilation} g{groups:<5} "
                        + " | ".join(cells)
                    )
    print("\nsum per shape (us), served against nlc")
    for frames in WIDTHS:
        for rows in BUCKETS:
            print(
                f"  width {frames} bucket {rows}: served {totals[(frames, rows, 'served')]:.0f}"
                f" nlc {totals[(frames, rows, 'nlc')]:.0f}"
                f" polyphase {totals[(frames, rows, 'polyphase')]:.0f}"
                f" unfold {totals[(frames, rows, 'unfold')]:.0f}"
                f" nlc search {totals[(frames, rows, 'nlc search')]:.0f}"
            )


if __name__ == "__main__":
    main()
