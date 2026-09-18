"""V1-e5 and V2-e1: where one vocoder decode's device time goes, and what compiling buys.

split    one CUDA graph replay per key, kernels grouped by category (count, us), for the
         current eager decode, the resident eager decode and the current decode compiled
         for that key; plus the conv modules whose output is not channels-last in the
         resident chain
compile  every captured key (widths 1-8, 16, 32, 64 x batch 1, 2, 4, 8), one arm per process (--arm):
         eager, one static compile per key (dynamic=False, as the runner does for width 8), or one
         dynamic compile shared by all keys; compile and capture seconds, graph memory,
         replay ms

usage: python vocoder_attribution_bench.py <split|compile> --model DIR
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict

import torch
from vocoder_resident_bench import (
    BATCHES,
    WIDTHS,
    ResidentDecoder,
    graph_of,
    load,
    random_codes,
    replay_ms,
)

SPLIT_KEYS = ((1, 1), (8, 1), (8, 8), (16, 2), (32, 4), (64, 1))
CATEGORIES = (
    ("transpose", r"nchwToNhwc|nhwcToNchw"),
    ("conv", r"convolve|fprop|dgrad|conv_|_conv|Conv"),
    ("triton", r"^triton_"),
    ("gemm", r"gemm|gemv|cutlass|cublas|sm\d+_xmma|ampere_"),
    ("copy_cat", r"copy|Copy|CatArray|cat_"),
    ("index", r"index|gather|scatter|Index"),
    ("reduce_norm_softmax", r"reduce|softmax|norm|Norm|Reduce"),
    ("elementwise", r"elementwise|vectorized|unrolled|Elementwise"),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "other"


def split_replay(graph) -> tuple[dict[str, list[float]], dict[str, float]]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        graph.replay()
        torch.cuda.synchronize()
    groups: dict[str, list[float]] = defaultdict(list)
    names: dict[str, float] = defaultdict(float)
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = event.time_range.elapsed_us()
        groups[category(event.name)].append(us)
        names[event.name] += us
    return groups, names


def mode_split(args, incremental, device) -> None:
    resident = ResidentDecoder(incremental, depthwise_resident=True)
    names = {module: name for name, module in incremental._decoder.named_modules()}
    for width, batch in SPLIT_KEYS:
        codes = random_codes(batch, width, device)
        arms = {
            "current": (
                incremental._decode_tensors,
                incremental.init_state(batch, device=device, dtype=torch.bfloat16),
            ),
            "resident": (
                resident.decode_tensors,
                resident.init_state(batch, device, torch.bfloat16),
            ),
            "current_compiled": (
                torch.compile(
                    incremental._decode_tensors, dynamic=False, fullgraph=True
                ),
                incremental.init_state(batch, device=device, dtype=torch.bfloat16),
            ),
        }
        print(f"\n== width {width} batch {batch}")
        for label, (fn, state) in arms.items():
            graph = graph_of(fn, codes, state)
            ms = replay_ms(graph, args.reps)
            groups, per_name = split_replay(graph)
            total = sum(sum(v) for v in groups.values())
            print(
                f"-- {label}: replay {ms:.3f} ms, profiled kernel sum {total / 1e3:.3f} ms, kernels {sum(len(v) for v in groups.values())}"
            )
            for group, times in sorted(groups.items(), key=lambda item: -sum(item[1])):
                print(
                    f"   {group:>20} {len(times):>5} kernels {sum(times):>10.1f} us {100 * sum(times) / total:>6.1f}%"
                )
            for name, us in sorted(per_name.items(), key=lambda item: -item[1])[
                : args.top
            ]:
                print(f"      {us:>9.1f} us  {name[:140]}")
            del graph
        strided = []
        original_conv = resident.conv

        def recording_conv(conv, x):
            y = original_conv(conv, x)
            if not y.is_contiguous():
                strided.append(
                    f"{names.get(conv, '?')} w{tuple(conv.weight.shape)} groups {conv.groups}"
                )
            return y

        resident.conv = recording_conv
        with torch.inference_mode():
            resident.decode_tensors(
                codes, resident.init_state(batch, device, torch.bfloat16)
            )
        resident.conv = original_conv
        print(f"-- resident conv outputs not channels-last: {len(strided)}")
        for entry in strided:
            print(f"      {entry}")
        torch.cuda.empty_cache()


def mode_compile(args, incremental, device) -> None:
    fn = {
        "eager": incremental._decode_tensors,
        "static": torch.compile(
            incremental._decode_tensors, dynamic=False, fullgraph=True
        ),
        "dynamic": torch.compile(
            incremental._decode_tensors, dynamic=True, fullgraph=True
        ),
    }[args.arm]
    print(
        f"{'width':>5} {'batch':>5} {'arm':>8} {'compile_s':>9} {'capture_s':>9} {'graph_MiB':>9} {'ms':>8}"
    )
    for width in WIDTHS:
        for batch in BATCHES:
            codes = random_codes(batch, width, device)
            state = incremental.init_state(batch, device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                fn(codes, state)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - started
            reserved = torch.cuda.memory_reserved(device)
            started = time.perf_counter()
            graph = graph_of(fn, codes, state)
            capture_s = time.perf_counter() - started
            graph_mib = (torch.cuda.memory_reserved(device) - reserved) / 2**20
            ms = replay_ms(graph, args.reps)
            print(
                f"{width:>5} {batch:>5} {args.arm:>8} {compile_s:>9.2f} {capture_s:>9.2f} {graph_mib:>9.1f} {ms:>8.3f}",
                flush=True,
            )
            del graph
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("split", "compile"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--arm", choices=("eager", "static", "dynamic"), default="eager"
    )
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    torch._dynamo.config.recompile_limit = 256
    _, incremental = load(args.model, device)
    print(
        f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}"
    )
    {"split": mode_split, "compile": mode_compile}[args.mode](args, incremental, device)


if __name__ == "__main__":
    main()
