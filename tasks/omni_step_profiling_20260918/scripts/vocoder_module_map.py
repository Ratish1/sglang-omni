"""V-e10: one vocoder decode, every CUDA kernel attributed to the module that launched it.

The incremental decoder's functions and the decoder's modules are wrapped in profiler
ranges (conv and transposed conv per state key, snake per channel count, the transformer's
attention, MLP and norms per layer, ConvNeXt, the quantizer, residual units). One eager
decode per key runs under the torch profiler; each kernel goes to the innermost range that
contains its launching op on the CPU thread, so nothing is inferred from kernel names.
Kernel durations are what a graph replay executes; launch gaps are not counted.

Per key: device time and kernel count per module group and per range, the kernel families
inside each group, and what is left outside every range.

usage: python vocoder_module_map.py --model DIR [--keys 1x1,8x1,8x8,32x4,64x1] [--fused]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import tempfile
from collections import Counter, defaultdict

import torch
from vocoder_resident_bench import load, random_codes

from sglang_omni.models.qwen3_tts import incremental_codec

FAMILIES = (
    ("cudnn transpose", r"nchwToNhwc|nhwcToNchw"),
    ("conv", r"convolve|fprop|dgrad|conv_|_conv|Conv|xmma"),
    ("gemm", r"gemm|gemv|cutlass|cublas"),
    ("snake triton", r"snake"),
    ("copy cat", r"copy|Copy|CatArray|cat_"),
    ("index", r"index|gather|scatter|Index"),
    ("softmax norm reduce", r"reduce|softmax|norm|Norm|Reduce"),
    ("elementwise", r"elementwise|vectorized|unrolled|Elementwise"),
)


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name):
            return label
    return "other"


def ranged(label_of, fn):
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(label_of(*args, **kwargs)):
            return fn(*args, **kwargs)

    return wrapper


def instrument(decoder) -> None:
    codec = incremental_codec
    codec.incremental_causal_conv1d = ranged(
        lambda module, x, state, key: f"conv|{key}", codec.incremental_causal_conv1d
    )
    codec.incremental_causal_transconv1d = ranged(
        lambda module, x, state, key: f"transconv|{key}",
        codec.incremental_causal_transconv1d,
    )
    codec._incremental_attention = ranged(
        lambda attention, hidden, pos, state, layer_index, *rest: f"attention|layer{layer_index}",
        codec._incremental_attention,
    )
    codec._incremental_convnext = ranged(
        lambda module, x, state, key: f"convnext rest|{key}",
        codec._incremental_convnext,
    )
    codec._incremental_residual_unit = ranged(
        lambda module, x, state, key: f"residual unit rest|{key}",
        codec._incremental_residual_unit,
    )
    codec._incremental_transformer = ranged(
        lambda *args: "transformer rest|pre_transformer", codec._incremental_transformer
    )
    decoder.quantizer.decode = ranged(
        lambda codes: "quantizer|decode", decoder.quantizer.decode
    )

    def hook(label):
        def pre(module, inputs):
            module._range = torch.profiler.record_function(label)
            module._range.__enter__()

        def post(module, inputs, output):
            module._range.__exit__(None, None, None)

        return pre, post

    for name, module in decoder.named_modules():
        kind = type(module).__name__
        if kind in ("SnakeBeta", "FusedSnakeBeta"):
            label = f"snake|C{module.alpha.shape[0]}"
        elif name.endswith(".mlp"):
            label = f"mlp|{name}"
        elif (
            name.endswith(("input_layernorm", "post_attention_layernorm"))
            or name == "pre_transformer.norm"
        ):
            label = f"transformer norm|{name}"
        elif name.endswith(("self_attn_layer_scale", "mlp_layer_scale")):
            label = f"layer scale|{name}"
        elif name.endswith("rotary_emb"):
            label = f"rotary|{name}"
        else:
            continue
        pre, post = hook(label)
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)


def attribute(trace_path: str):
    with open(trace_path) as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    ranges = sorted(
        (
            (float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"])
            for e in events
            if e.get("cat") == "user_annotation" and "|" in e["name"]
        ),
        key=lambda item: (item[0], -item[1]),
    )
    range_starts = [r[0] for r in ranges]
    launch_ts = {}
    for event in events:
        args = event.get("args") or {}
        if (
            event.get("cat") in ("cuda_runtime", "cuda_driver")
            and args.get("correlation") is not None
        ):
            launch_ts[args["correlation"]] = float(event["ts"])
    per_range = defaultdict(lambda: [0, 0.0])
    per_group_family = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        ts = launch_ts.get((event.get("args") or {}).get("correlation"))
        label = "outside every range|-"
        if ts is not None:
            index = bisect.bisect_right(range_starts, ts) - 1
            while index >= 0:
                start, end, name = ranges[index]
                if start <= ts <= end:
                    label = name
                    break
                index -= 1
        duration = float(event["dur"])
        per_range[label][0] += 1
        per_range[label][1] += duration
        entry = per_group_family[label.split("|")[0]][family(event["name"])]
        entry[0] += 1
        entry[1] += duration
    return per_range, per_group_family


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--keys", default="1x1,8x1,8x8,32x4,64x1")
    parser.add_argument(
        "--fused",
        action="store_true",
        help="fuse SnakeBeta with the tree's kernel first",
    )
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    tokenizer, incremental = load(args.model, device)
    decoder = tokenizer.model.decoder
    if args.fused:
        from sglang_omni.models.qwen3_tts.vocoder_kernels import fuse_vocoder_decoder

        print(f"fused {fuse_vocoder_decoder(decoder)} SnakeBeta modules")
    instrument(decoder)
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    for key in args.keys.split(","):
        width, batch = (int(part) for part in key.split("x"))
        codes = random_codes(batch, width, device)
        with torch.inference_mode():
            for _ in range(2):
                state = incremental.init_state(
                    batch, device=device, dtype=torch.bfloat16
                )
                incremental._decode_tensors(codes, state)
            torch.cuda.synchronize()
            state = incremental.init_state(batch, device=device, dtype=torch.bfloat16)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                incremental._decode_tensors(codes, state)
                torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "trace.json")
            prof.export_chrome_trace(path)
            per_range, per_group_family = attribute(path)
        total = sum(v[1] for v in per_range.values())
        kernels = sum(v[0] for v in per_range.values())
        print(
            f"\n==== width {width} batch {batch}: {kernels} kernels, {total / 1e3:.3f} ms of device time"
        )
        groups = Counter()
        group_kernels = Counter()
        for label, (count, us) in per_range.items():
            groups[label.split("|")[0]] += us
            group_kernels[label.split("|")[0]] += count
        print(
            f"{'module group':26s}{'kernels':>9s}{'ms':>10s}{'share':>8s}   kernel families (count / us)"
        )
        for group, us in groups.most_common():
            cells = ", ".join(
                f"{name} {c}/{t:.0f}"
                for name, (c, t) in sorted(
                    per_group_family[group].items(), key=lambda item: -item[1][1]
                )
            )
            print(
                f"{group:26s}{group_kernels[group]:>9d}{us / 1e3:>10.3f}{100 * us / total:>7.1f}%   {cells}"
            )
        print("largest single ranges:")
        for label, (count, us) in sorted(
            per_range.items(), key=lambda item: -item[1][1]
        )[:10]:
            print(
                f"  {label:58s}{count:>6d}{us / 1e3:>9.3f} ms{100 * us / total:>6.1f}%"
            )


if __name__ == "__main__":
    main()
