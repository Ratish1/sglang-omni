"""V-e7: the in-tree fused SnakeBeta kernel inside plain CUDA graphs, no torch.compile.

Every captured vocoder key (widths 1-8, 16, 32, 64 x batch 1, 2, 4, 8): the eager decode
and the same decode after fuse_vocoder_decoder, each captured as a plain CUDA graph on
the same codes and a zero state. Per key: ms per replay of both, kernels per replay of
both, and whether the two waveforms are bitwise equal. Then the kernel-category split of
the fused decode for a few keys, to show what elementwise work is left.

usage: python vocoder_fused_snake_bench.py --model DIR [--reps 30]
"""

from __future__ import annotations

import argparse
import time

import torch
from vocoder_attribution_bench import SPLIT_KEYS, category
from vocoder_resident_bench import BATCHES, WIDTHS, load, random_codes, replay_ms

from sglang_omni.models.qwen3_tts.vocoder_kernels import fuse_vocoder_decoder


def capture(fn, codes, state):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            fn(codes, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        waveform = fn(codes, state)
    return graph, waveform


def measure(incremental, device, reps):
    rows = {}
    for width in WIDTHS:
        for batch in BATCHES:
            torch.manual_seed(width * 100 + batch)
            codes = random_codes(batch, width, device)
            state = incremental.init_state(batch, device=device, dtype=torch.bfloat16)
            graph, waveform = capture(incremental._decode_tensors, codes, state)
            with torch.inference_mode():
                for tensors in (
                    state.conv_histories,
                    state.transconv_overlaps,
                    state.transformer_keys,
                    state.transformer_values,
                ):
                    for value in tensors.values():
                        value.zero_()
            graph.replay()
            torch.cuda.synchronize()
            first = waveform.clone()
            ms = replay_ms(graph, reps)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            kernels = [
                e
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            rows[(width, batch)] = (ms, len(kernels), first, kernels)
            del graph
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    tokenizer, incremental = load(args.model, device)
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    eager = measure(incremental, device, args.reps)
    started = time.perf_counter()
    replaced = fuse_vocoder_decoder(tokenizer.model.decoder)
    print(
        f"fuse_vocoder_decoder replaced {replaced} modules in {time.perf_counter() - started:.2f} s (includes the Triton prewarm)"
    )
    fused = measure(incremental, device, args.reps)
    print(
        f"{'width':>5} {'batch':>5} {'eager ms':>9} {'fused ms':>9} {'delta':>7} {'eager k':>8} {'fused k':>8} {'bitwise':>8} {'max abs':>9}"
    )
    for key in eager:
        e_ms, e_k, e_wave, _ = eager[key]
        f_ms, f_k, f_wave, _ = fused[key]
        same = torch.equal(e_wave, f_wave)
        diff = (e_wave.float() - f_wave.float()).abs().max().item()
        print(
            f"{key[0]:>5} {key[1]:>5} {e_ms:>9.3f} {f_ms:>9.3f} {100 * (f_ms - e_ms) / e_ms:>6.1f}% "
            f"{e_k:>8} {f_k:>8} {str(same):>8} {diff:>9.2e}"
        )
    for key in SPLIT_KEYS:
        for label, rows in (("eager", eager), ("fused", fused)):
            groups: dict[str, list[float]] = {}
            for event in rows[key][3]:
                groups.setdefault(category(event.name), []).append(
                    event.time_range.elapsed_us()
                )
            total = sum(sum(v) for v in groups.values())
            cells = ", ".join(
                f"{name} {len(times)} / {sum(times):.0f} us"
                for name, times in sorted(
                    groups.items(), key=lambda item: -sum(item[1])
                )
            )
            print(
                f"w{key[0]} b{key[1]} {label}: kernel sum {total / 1e3:.3f} ms: {cells}"
            )


if __name__ == "__main__":
    main()
