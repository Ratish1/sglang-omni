"""Where cuDNN leaves its tensor-core kernel for the decoder's dilated convs, and
whether a row's output depends on the batch it runs in.

The speech tokenizer decoder's residual units run a dilated k 7 conv; at 768 channels
cuDNN runs xmma_fprop at batch 1 and implicit_convolve_sgemm at batch 16. For each
dilated conv of the decoder (the shapes a width 8 decode gives), for batch 1 to 16:
the kernel and time as served (NCL) and as a 4D channels-last conv, and whether every
row equals that row convolved alone at batch 1 as served.

usage: python vocoder_dilated_sweep.py [--frames 8]
"""

from __future__ import annotations

import argparse

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
    for _ in range(10):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / CALLS)
    times.sort()
    return times[len(times) // 2]


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
        and "nchwToNhwc" not in event.name
        and "nhwcToNchw" not in event.name
    ]
    return max(names, key=len)[:34] if names else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    # Fresh samples at decoder.0: frames x the two x2 upsample stages.
    samples = args.frames * 4
    torch.manual_seed(0)
    with torch.inference_mode():
        for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
            samples *= int(block.block[1].conv.stride[0])
            for unit in block.block[2:]:
                conv = unit.conv1.conv
                dilation = conv.dilation[0]
                if dilation == 1:
                    continue
                else:
                    pass
                history = (conv.kernel_size[0] - 1) * dilation
                length = history + samples
                full = torch.randn(
                    16, conv.in_channels, length, device="cuda", dtype=torch.bfloat16
                )
                w4 = conv.weight.unsqueeze(2).contiguous(
                    memory_format=torch.channels_last
                )
                print(
                    f"decoder.{block_index} in {conv.in_channels} d {dilation} "
                    f"L {length}"
                )
                alone = torch.cat(
                    [
                        F.conv1d(
                            full[i : i + 1], conv.weight, conv.bias, 1, 0, dilation
                        )
                        for i in range(16)
                    ]
                )
                for rows in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16):
                    x = full[:rows].contiguous()
                    x4 = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)

                    def served():
                        return F.conv1d(x, conv.weight, conv.bias, 1, 0, dilation)

                    def nhwc():
                        return F.conv2d(x4, w4, conv.bias, 1, 0, (1, dilation)).squeeze(
                            2
                        )

                    served_invariant = torch.equal(served(), alone[:rows])
                    nhwc_invariant = torch.equal(nhwc(), alone[:rows])
                    nhwc_kernel = main_kernel(nhwc)
                    # note(ratish): cuDNN's direct grouped kernel takes about 70 ms here.
                    if "direct" in nhwc_kernel:
                        nhwc_us = float("nan")
                    else:
                        nhwc_us = graph_time_us(nhwc)
                    print(
                        f"  bs {rows:>2}  served {graph_time_us(served):8.1f} "
                        f"{main_kernel(served):<35} rows as alone {served_invariant!s:<5} "
                        f"| nhwc {nhwc_us:8.1f} {nhwc_kernel:<35} "
                        f"rows as alone {nhwc_invariant}"
                    )


if __name__ == "__main__":
    main()
