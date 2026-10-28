"""Tile sweep for D's three-pass TF32 linear on the reference encoder transformer's GEMMs.

Shapes: rows are the transformer's tokens (two per codec frame, buckets 32 to 256 frames), the weights the layer's
q, k, v, o (512 x 512), fc1 (2048 x 512) and fc2 (512 x 2048). Per tile config and shape: device time of the
kernel against torch's float32 F.linear (FFMA, as D serves today), and each one's largest relative error against a
float64 product; then each config's time over one encode's mix (4 x qkvo + fc1 + fc2 per layer, 8 layers) per
bucket.

usage: PYTHONPATH=D_TREE python3 tf32x3_linear_sweep.py [--card 2]
"""

import argparse
import itertools

import torch
import torch.nn.functional as F

from sglang_omni.kernels.tf32x3_linear import tf32x3_linear

BUCKET_FRAMES = (32, 64, 128, 256)
SHAPES = {"qkvo": (512, 512), "fc1": (2048, 512), "fc2": (512, 2048)}
LAYER_MIX = {"qkvo": 4, "fc1": 1, "fc2": 1}
LAYERS = 8
CONFIGS = [
    dict(block_m=bm, block_n=bn, block_k=bk, splits=splits, num_warps=4, num_stages=3)
    for bm, bn, bk, splits in itertools.product((16, 32, 64), (32, 64), (32, 64), (1, 2, 4, 8))
]
REPEATS = 50


def time_us(function, *args) -> float:
    """Device time per call, the calls replayed from one CUDA graph as the encoder's graphs replay them, so no host
    launch time enters."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPEATS):
            function(*args)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return 1e3 * start.elapsed_time(end) / REPEATS


def relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(((output.double() - reference).abs().max() / reference.abs().max()).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(0)
    cases = {}
    for frames, (name, (out_features, in_features)) in itertools.product(BUCKET_FRAMES, SHAPES.items()):
        rows = 2 * frames
        hidden = torch.randn(rows, in_features, device="cuda", generator=generator)
        weight = torch.randn(out_features, in_features, device="cuda", generator=generator) / in_features**0.5
        reference = hidden.double() @ weight.double().T
        ffma_us = time_us(F.linear, hidden, weight)
        cases[frames, name] = (hidden, weight, reference, ffma_us, relative_error(F.linear(hidden, weight), reference))
    totals = {}
    for config in CONFIGS:
        label = " ".join(f"{key}={value}" for key, value in config.items())
        try:
            per_case = {}
            for key, (hidden, weight, reference, _, _) in cases.items():
                output = tf32x3_linear(hidden, weight, **config)
                per_case[key] = (time_us(lambda h, w: tf32x3_linear(h, w, **config), hidden, weight),
                                 relative_error(output, reference))
        except Exception as error:
            print(f"{label}: failed {type(error).__name__}: {str(error)[:120]}", flush=True)
            continue
        totals[label] = per_case
    for frames in BUCKET_FRAMES:
        ffma_encode = LAYERS * sum(LAYER_MIX[name] * cases[frames, name][3] for name in SHAPES)
        ranked = sorted(
            totals.items(),
            key=lambda item: LAYERS * sum(LAYER_MIX[name] * item[1][frames, name][0] for name in SHAPES),
        )
        print(f"== {frames} frames ({2 * frames} tokens): FFMA {ffma_encode:.1f} us per encode, "
              f"FFMA error {max(cases[frames, name][4] for name in SHAPES):.2e}", flush=True)
        for label, per_case in ranked[:4]:
            encode_us = LAYERS * sum(LAYER_MIX[name] * per_case[frames, name][0] for name in SHAPES)
            error = max(per_case[frames, name][1] for name in SHAPES)
            shape_us = ", ".join(f"{name} {per_case[frames, name][0]:.1f}" for name in SHAPES)
            print(f"   {encode_us:7.1f} us per encode, error {error:.2e}: {label} ({shape_us})", flush=True)
    for name in SHAPES:
        by_shape = sorted(
            totals.items(), key=lambda item: sum(item[1][frames, name][0] for frames in BUCKET_FRAMES)
        )
        ffma = ", ".join(f"{cases[frames, name][3]:.1f}" for frames in BUCKET_FRAMES)
        print(f"== {name}: FFMA per bucket {ffma} us", flush=True)
        for label, per_case in by_shape[:3]:
            kernel = ", ".join(f"{per_case[frames, name][0]:.1f}" for frames in BUCKET_FRAMES)
            print(f"   {kernel} us: {label}", flush=True)
    overall = sorted(
        totals.items(),
        key=lambda item: sum(
            LAYERS * sum(LAYER_MIX[name] * item[1][frames, name][0] for name in SHAPES) for frames in BUCKET_FRAMES
        ),
    )
    print(f"best over every bucket: {overall[0][0]}", flush=True)


if __name__ == "__main__":
    main()
