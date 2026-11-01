"""Design 15's prototype gates 1 and 2 on one card: bit identity of the persistent pass (persistent_pass.py) to main's
fused layers, then the step replay time of both, at the served row counts.

A step is the served predictor step without its lm_heads and codebook steps (predictor_layers_bench.py's step): the
opening pair pass at slots 0 and 1 where 2 * batch fits MAX_FUSED_ROWS, then 14 single-token passes at slots 2 to 15,
each over the 5 layers, here each pass with its own random input. Main's path ends each pass with the served final
norm, sgl_kernel's rmsnorm (what RMSNorm.forward_cuda calls for a 2D bf16 input without the HF cast); the
persistent pass ends with its own.

Identity: both paths from zeroed caches over the same inputs, eager; every pass's output and, after the step, both
caches compared with torch.equal (mismatching elements and the largest difference printed otherwise). Then each step
captured in a CUDA graph, replayed IDENTITY_REPLAYS times, every replay's outputs and caches compared with the eager
result. Timing: ROUNDS rounds, each timing main's graph, the persistent graph with and without the L2 prefetch, in
rotating order, each the median of --replays replays; the medians of the rounds printed. Also the persistent kernel's
registers, spills and shared memory per compiled variant.

usage: PYTHONPATH=<main tree> python3 persistent_pass_bench.py [--card 0] [--batch 1 12 31 64] [--replays 50]
"""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace

import torch
from persistent_pass import PersistentPredictorPass
from predictor_layers_bench import (
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    SLOTS,
    clear_slow_mode,
    make_layers,
    make_shape,
    replay_median_ms,
)
from sgl_kernel import rmsnorm

from sglang_omni.utils.predictor_layers import MAX_FUSED_ROWS, FusedPredictorLayers

IDENTITY_REPLAYS = 30
ROUNDS = 5


def compare(label: str, expected: torch.Tensor, actual: torch.Tensor) -> bool:
    if torch.equal(expected, actual):
        return True
    else:
        differs = expected != actual
        largest = (expected.float() - actual.float()).abs().max().item()
        print(
            f"    {label}: {int(differs.sum())} of {differs.numel()} elements differ, largest {largest:.3e}"
        )
        return False


class Step:
    """One predictor step on one path with its own caches; outputs cloned per pass when recording."""

    def __init__(
        self, run_pass, batch_size: int, has_pair: bool, inputs, device: torch.device
    ) -> None:
        self.run_pass = run_pass
        self.batch_size = batch_size
        self.has_pair = has_pair
        self.inputs = inputs
        self.k_cache = torch.zeros(
            LAYERS,
            batch_size,
            SLOTS,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.outputs: list[torch.Tensor] = []

    def __call__(self, record: bool) -> None:
        self.outputs = []
        pair_embeds, pair_positions, single_embeds, slot_positions = self.inputs
        passes = []
        if self.has_pair:
            passes.append((pair_embeds, 0, pair_positions))
        else:
            pass
        for index, cache_len in enumerate(range(2, SLOTS - 1)):
            passes.append((single_embeds[index], cache_len, slot_positions[cache_len]))
        for embeds, cache_len, positions in passes:
            out = self.run_pass(
                token_embeds=embeds,
                batch_size=self.batch_size,
                cache_len=cache_len,
                positions=positions,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
            )
            if record:
                self.outputs.append(out.clone())
            else:
                pass

    def snapshot(self) -> list[torch.Tensor]:
        return [*self.outputs, self.k_cache.clone(), self.v_cache.clone()]


def bench(batch_size: int, replays: int, device: torch.device) -> None:
    torch.manual_seed(batch_size)
    layers = make_layers(device)
    shape = make_shape(device)
    final_norm = SimpleNamespace(
        weight=(1 + 0.1 * torch.randn(HIDDEN, device=device)).to(torch.bfloat16),
        variance_epsilon=1e-6,
    )
    max_rows = min(2 * batch_size, MAX_FUSED_ROWS)
    fused = FusedPredictorLayers(shape, max_rows, device, torch.bfloat16)
    has_pair = fused.covers(2 * batch_size)
    inputs = (
        torch.randn(batch_size, 2, HIDDEN, device=device).to(torch.bfloat16),
        torch.arange(2, device=device).repeat(batch_size),
        [
            torch.randn(batch_size, 1, HIDDEN, device=device).to(torch.bfloat16)
            for _ in range(SLOTS - 3)
        ],
        torch.arange(SLOTS, device=device)[:, None]
        .expand(SLOTS, batch_size)
        .contiguous(),
    )

    def main_pass(**kwargs):
        return fused.forward(
            layers=layers,
            final_norm=lambda hidden: rmsnorm(
                hidden, final_norm.weight, final_norm.variance_epsilon
            ),
            **kwargs,
        )

    paths = {"main": main_pass}
    persistent = {}
    for name, prefetch in (("persistent", True), ("persistent no prefetch", False)):
        persistent[name] = PersistentPredictorPass(
            layers, final_norm, shape, max_rows, device, prefetch
        )
        paths[name] = persistent[name].forward
    steps = {
        name: Step(run_pass, batch_size, has_pair, inputs, device)
        for name, run_pass in paths.items()
    }
    print(
        f"bs {batch_size}: pair pass {'in' if has_pair else 'left out'}, {len(inputs[2]) + has_pair} passes a step"
    )

    reference = None
    for name, step in steps.items():
        step(record=True)
        torch.cuda.synchronize()
        snapshot = step.snapshot()
        if reference is None:
            reference = snapshot
            continue
        else:
            pass
        labels = [f"pass {index} output" for index in range(len(snapshot) - 2)] + [
            "k cache",
            "v cache",
        ]
        same = [
            compare(f"{name} {label}", expected, actual)
            for label, expected, actual in zip(labels, reference, snapshot)
        ]
        print(
            f"  identity eager, {name} against main: {sum(same)} of {len(same)} tensors equal"
        )

    graphs = {}
    for name, step in steps.items():
        step(record=False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(record=True)
        graphs[name] = graph
        wrong = 0
        for _ in range(IDENTITY_REPLAYS):
            graph.replay()
            torch.cuda.synchronize()
            wrong += sum(
                not torch.equal(e, a) for e, a in zip(reference, step.snapshot())
            )
        print(
            f"  identity graph, {name}: {wrong} unequal tensors over {IDENTITY_REPLAYS} replays against main's eager"
        )

    for name, runner in persistent.items():
        for (block_m, tokens), handle in sorted(runner.compiled.items()):
            print(
                f"  {name} kernel block_m {block_m} T {tokens}: {handle.n_regs} registers, {handle.n_spills} spills, "
                f"{handle.metadata.shared} bytes shared, stages {runner.num_stages.get(block_m, 'default')}"
            )

    clear_slow_mode(device)
    names = list(graphs)
    times = {name: [] for name in names}
    for round_index in range(ROUNDS):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        for name in order:
            times[name].append(replay_median_ms(graphs[name], replays))
    base = statistics.median(times["main"])
    for name in names:
        value = statistics.median(times[name])
        rounds = " ".join(f"{t:.3f}" for t in times[name])
        print(
            f"  step replay {name:<24} {value:.3f} ms ({value / base - 1:+.1%} vs main), rounds {rounds}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 12, 31, 64])
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda", args.card)
    torch.cuda.set_device(device)
    print(f"{torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    with torch.inference_mode():
        for batch_size in args.batch:
            bench(batch_size, args.replays, device)


if __name__ == "__main__":
    main()
