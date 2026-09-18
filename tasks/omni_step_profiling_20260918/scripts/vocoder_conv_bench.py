"""Vocoder conv attribution and algorithm headroom for slice V, on the real decoder.

V1: one eager incremental decode per (batch, fresh_frames) shape under the torch
profiler with record_shapes; every cuDNN conv call is listed with its input and weight
shapes and the kernels it launched (including layout transposes), time per call.
V2: the same decodes with torch.backends.cudnn.benchmark on; per shape, decode time
(CUDA events, median) and the waveform against the heuristic run on the same codes and
the same initial state.

usage: python vocoder_conv_bench.py --model DIR --shapes 1:5 16:5 --reps 20
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

import torch

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder

NUM_QUANTIZERS = 16
CODE_RANGE = 1024


def decode_once(decoder, codes, initial_state):
    state = initial_state.clone()
    with torch.inference_mode():
        return decoder.decode(codes, state)


def timed_ms(decoder, codes, initial_state, reps: int) -> float:
    for _ in range(3):
        decode_once(decoder, codes, initial_state)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        decode_once(decoder, codes, initial_state)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def conv_table(decoder, codes, initial_state) -> list[tuple]:
    decode_once(decoder, codes, initial_state)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        decode_once(decoder, codes, initial_state)
        torch.cuda.synchronize()
    rows = []
    total_us = 0.0
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            total_us += event.time_range.elapsed_us()
        if "conv" not in event.name or not event.kernels:
            continue
        kernels = [(k.name, k.duration) for k in event.kernels]
        rows.append((event.name, str(event.input_shapes[:2]), kernels))
    return rows, total_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shapes", nargs="+", required=True, help="batch:fresh_frames")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        args.model, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
    shapes = [tuple(int(v) for v in item.split(":")) for item in args.shapes]
    torch.manual_seed(0)
    inputs = {}
    for batch, frames in shapes:
        codes = torch.randint(0, CODE_RANGE, (batch, NUM_QUANTIZERS, frames), device=device)
        state = decoder.init_state(batch, device=device, dtype=torch.bfloat16)
        inputs[(batch, frames)] = (codes, state)
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}")

    results = {}
    for benchmark in (False, True):
        torch.backends.cudnn.benchmark = benchmark
        mode = "benchmark" if benchmark else "heuristic"
        for shape, (codes, state) in inputs.items():
            waveform = decode_once(decoder, codes, state).float()
            median = timed_ms(decoder, codes, state, args.reps)
            rows, total_us = conv_table(decoder, codes, state)
            results[(mode, shape)] = (waveform, median, rows, total_us)

    for (mode, shape), (_, median, rows, total_us) in results.items():
        batch, frames = shape
        print(f"\n== {mode} batch {batch} fresh_frames {frames}: decode {median:.3f} ms, profiled GPU {total_us / 1e3:.3f} ms, conv calls {len(rows)}")
        grouped: dict[tuple, list] = defaultdict(list)
        for name, shapes_text, kernels in rows:
            grouped[(name, shapes_text, tuple(k for k, _ in kernels))].append(sum(d for _, d in kernels))
        ranked = sorted(grouped.items(), key=lambda item: sum(item[1]), reverse=True)
        for (name, shapes_text, kernel_names), times in ranked:
            print(f"  {len(times):3d} x {statistics.fmean(times):8.1f} us  {name.split('::')[-1]:28s} {shapes_text}")
            for kernel in kernel_names:
                print(f"        {kernel[:110]}")

    print("\n== waveform, benchmark vs heuristic on the same codes and state")
    for _, shape in [key for key in results if key[0] == "heuristic"]:
        reference = results[("heuristic", shape)][0]
        candidate = results[("benchmark", shape)][0]
        diff = (candidate - reference).abs()
        print(
            f"batch {shape[0]} fresh_frames {shape[1]}: equal {torch.equal(candidate, reference)}, "
            f"max abs {diff.max().item():.3e}, mean abs {diff.mean().item():.3e}, "
            f"decode ms {results[('heuristic', shape)][1]:.3f} -> {results[('benchmark', shape)][1]:.3f}"
        )


if __name__ == "__main__":
    main()
