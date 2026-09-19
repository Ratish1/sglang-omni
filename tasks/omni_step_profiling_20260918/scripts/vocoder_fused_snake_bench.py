"""V-e7 / V-e9: full vocoder decodes in plain CUDA graphs with the fused snake activation.

Every captured vocoder key (widths 1-8, 16, 32, 64 x batch 1, 2, 4, 8), three arms, each a
plain CUDA graph of one decode on the same codes and a zero state: the eager decoder,
the decoder fused by the kernel module at --old-kernels (upstream main's
vocoder_kernels.py, loaded from its file), and the decoder fused by the tree under test.
Per key: ms per replay and kernels per replay of each arm, the new arm against the old,
and whether each fused waveform is bitwise equal to eager. Three alternating rounds per
key, the lowest median of each arm.

usage: python vocoder_fused_snake_bench.py --model DIR --old-kernels FILE [--reps 30]
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import statistics

import torch
from vocoder_resident_bench import BATCHES, WIDTHS, load, random_codes, replay_ms

from sglang_omni.models.qwen3_tts import vocoder_kernels
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder


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


def state_tensors(state) -> list[torch.Tensor]:
    return [
        value
        for tensors in (
            state.conv_histories,
            state.transconv_overlaps,
            state.transformer_keys,
            state.transformer_values,
        )
        for value in tensors.values()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--old-kernels", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--arms", default="eager,old,new")
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    spec = importlib.util.spec_from_file_location(
        "old_vocoder_kernels", args.old_kernels
    )
    old_kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_kernels)

    # note(ratish): the tokenizer loader caches per process, so every arm fuses its
    # own copy of the decoder
    tokenizer, _ = load(args.model, device)
    pristine = tokenizer.model.decoder
    snake_cls = next(
        type(m) for m in pristine.modules() if type(m).__name__ == "SnakeBeta"
    )
    arms = {}
    for name in args.arms.split(","):
        decoder = copy.deepcopy(pristine)
        if name == "old":
            print(
                f"old kernel fused {old_kernels.fuse_vocoder_decoder(decoder)} modules"
            )
        if name == "new":
            print(
                f"new kernel fused {vocoder_kernels.fuse_vocoder_decoder(decoder, snake_cls)} modules"
            )
        arms[name] = (decoder, Qwen3TTSIncrementalDecoder(decoder))
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    print(
        f"{'width':>5} {'batch':>5} {'eager ms':>9} {'old ms':>9} {'new ms':>9} {'new/old':>8} "
        f"{'new/eager':>9} {'eager k':>8} {'old k':>6} {'new k':>6} {'old bits':>9} {'new bits':>9}"
    )
    ratios = []
    for width in WIDTHS:
        for batch in BATCHES:
            torch.manual_seed(width * 100 + batch)
            codes = random_codes(batch, width, device)
            graphs, waves, kernels, states = {}, {}, {}, {}
            for name, (_, incremental) in arms.items():
                # note(ratish): the decode rebinds the state's tensors and the next
                # capture empties the allocator cache, so the inputs, the rebound state
                # and the output of a graph are all held for as long as it lives
                state = incremental.init_state(
                    batch, device=device, dtype=torch.bfloat16
                )
                inputs = state_tensors(state)
                graph, waveform = capture(incremental._decode_tensors, codes, state)
                states[name] = (inputs, state, waveform)
                with torch.inference_mode():
                    for tensor in inputs:
                        tensor.zero_()
                graph.replay()
                torch.cuda.synchronize()
                waves[name] = waveform.clone()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                kernels[name] = sum(
                    e.device_type == torch.autograd.DeviceType.CUDA
                    for e in prof.events()
                )
                graphs[name] = graph
            times = {name: [] for name in arms}
            for _ in range(3):
                for name in arms:
                    times[name].append(replay_ms(graphs[name], args.reps))
            best = {name: min(values) for name, values in times.items()}
            if len(arms) < 3:
                print(
                    f"{width:>5} {batch:>5} "
                    + " ".join(f"{n} {v:.3f}" for n, v in best.items()),
                    flush=True,
                )
                del graphs, states
                torch.cuda.empty_cache()
                continue
            ratios.append(best["new"] / best["old"])
            print(
                f"{width:>5} {batch:>5} {best['eager']:>9.3f} {best['old']:>9.3f} {best['new']:>9.3f} "
                f"{ratios[-1]:>8.3f} {best['new'] / best['eager']:>9.3f} "
                f"{kernels['eager']:>8} {kernels['old']:>6} {kernels['new']:>6} "
                f"{str(torch.equal(waves['old'], waves['eager'])):>9} "
                f"{str(torch.equal(waves['new'], waves['eager'])):>9}",
                flush=True,
            )
            del graphs, states
            torch.cuda.empty_cache()
    if not ratios:
        print("\nnew against old: not measured (fewer than three arms)")
        return
    geo = math.exp(statistics.fmean(math.log(v) for v in ratios))
    print(
        f"\nnew against old: geo {geo:.3f}, worst {max(ratios):.3f}, best {min(ratios):.3f}; "
        f"above 1.00: {sum(v > 1.0 for v in ratios)}, above 1.01: {sum(v > 1.01 for v in ratios)} of {len(ratios)}"
    )


if __name__ == "__main__":
    main()
