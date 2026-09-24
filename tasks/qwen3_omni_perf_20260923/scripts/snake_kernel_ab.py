"""Two fused SnakeBeta kernel files against eager on the real Qwen3-Omni code2wav decoder.

Loads the code2wav model three times (eager, kernel file A, kernel file B), fuses A and B
with each file's fuse_vocoder_decoder, and for every window shape checks torch.equal of
the decoder output against eager and between A and B, then times a CUDA graph replay of
each arm (median of rounds, alternating arms). Every SnakeBeta activation shape seen in the
forward is also timed through each file's fused call alone.

usage: python snake_kernel_ab.py --kernel-a PR.py --kernel-b OURS.py --model PATH
       [--rounds 7 --iters 50]
Run with PYTHONPATH set to an omni tree that has load_code2wav_model.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import statistics
from types import ModuleType

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)

WINDOWS = ((1, 10), (1, 20), (1, 30), (1, 35), (8, 20), (16, 20))


def load_kernel_file(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fused_call(kernel: ModuleType, x: torch.Tensor, snake: torch.nn.Module):
    """Call the file's fused_snake_beta with the arguments its signature takes."""
    if len(inspect.signature(kernel.fused_snake_beta).parameters) == 4:
        return kernel.fused_snake_beta(x, snake.alpha, snake.beta, snake.no_div_by_zero)
    else:
        return kernel.fused_snake_beta(x, snake.alpha, snake.beta)


def graph_time_ms(model: torch.nn.Module, codes: torch.Tensor, iters: int) -> float:
    static = codes.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            model(static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(static)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-a", required=True)
    parser.add_argument("--kernel-b", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    kernels = {
        "a": load_kernel_file("snake_kernel_a", args.kernel_a),
        "b": load_kernel_file("snake_kernel_b", args.kernel_b),
    }
    models = {
        arm: load_code2wav_model(args.model, device="cuda:0", dtype="bfloat16").eval()
        for arm in ("eager", "a", "b")
    }
    for arm, kernel in kernels.items():
        print(
            f"{arm} fused modules: {kernel.fuse_vocoder_decoder(models[arm].decoder)}"
        )

    shapes: dict[tuple[int, int, int], torch.nn.Module] = {}

    def record_shape(module, inputs, output):
        shapes.setdefault(tuple(inputs[0].shape), module)

    hooks = [
        module.register_forward_hook(record_shape)
        for module in models["eager"].decoder.modules()
        if type(module).__name__ == "SnakeBeta"
    ]
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    config = models["eager"].config
    with torch.inference_mode():
        for batch, frames in WINDOWS:
            codes = torch.randint(
                config.codebook_size,
                (batch, config.num_quantizers, frames),
                device="cuda:0",
                generator=generator,
            )
            outputs = {arm: model(codes) for arm, model in models.items()}
            print(
                f"window B={batch} F={frames}: a==eager {torch.equal(outputs['a'], outputs['eager'])} "
                f"b==eager {torch.equal(outputs['b'], outputs['eager'])} "
                f"a==b {torch.equal(outputs['a'], outputs['b'])}"
            )
            timings: dict[str, list[float]] = {arm: [] for arm in models}
            for _ in range(args.rounds):
                for arm, model in models.items():
                    timings[arm].append(graph_time_ms(model, codes, args.iters))
            medians = {
                arm: statistics.median(values) for arm, values in timings.items()
            }
            print(
                f"  graph replay ms median eager {medians['eager']:.3f} "
                f"a {medians['a']:.3f} b {medians['b']:.3f}"
            )
        for hook in hooks:
            hook.remove()

        print(f"activation shapes seen: {len(shapes)}")
        for shape, snake in sorted(shapes.items()):
            x = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16)
            results = {
                arm: fused_call(kernel, x, snake) for arm, kernel in kernels.items()
            }
            times: dict[str, list[float]] = {arm: [] for arm in kernels}
            for _ in range(args.rounds):
                for arm, kernel in kernels.items():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iters):
                        fused_call(kernel, x, snake)
                    end.record()
                    end.synchronize()
                    times[arm].append(start.elapsed_time(end) * 1000 / args.iters)
            fused = {arm: value is not None for arm, value in results.items()}
            same = fused["a"] and fused["b"] and torch.equal(results["a"], results["b"])
            print(
                f"shape {shape}: fused a {fused['a']} b {fused['b']} a==b {same} "
                f"us median a {statistics.median(times['a']):.1f} "
                f"b {statistics.median(times['b']):.1f}"
            )


if __name__ == "__main__":
    main()
