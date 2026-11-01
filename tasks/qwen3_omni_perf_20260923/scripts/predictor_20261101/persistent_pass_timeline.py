"""Where the persistent pass's time goes (persistent_pass.py with timeline_calls): every program stores the global
timer at kernel start, at its arrival at and departure from each of the 25 barriers, and at its end.

Per barrier b of a pass: phase span = the last program's arrival at b minus the earliest departure from b - 1 (the
kernel's earliest start for b = 0), barrier latency = the first departure from b minus the last arrival at b, and
departure spread = the last departure minus the first. Per phase kind (qkv, attention, o_proj, gate_up, down, then
the final norm from the last departure to the last program's end) the means over layers, passes and replays. Per
step: the replay time, the kernels' spans summed, and the gaps between consecutive launches (one launch's last end to
the next one's first start). The timer's smallest nonzero step is printed, as it bounds the resolution.

usage: PYTHONPATH=<main tree> python3 persistent_pass_timeline.py [--card 0] [--batch 1 12 31 64] [--prefetch 0 1]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from types import SimpleNamespace

import torch
from persistent_pass import PHASES_PER_LAYER, PersistentPredictorPass
from persistent_pass_bench import Step
from predictor_layers_bench import (
    HIDDEN,
    LAYERS,
    SLOTS,
    clear_slow_mode,
    make_layers,
    make_shape,
    replay_median_ms,
)

from sglang_omni.utils.predictor_layers import MAX_FUSED_ROWS

KINDS = ("qkv", "attention", "o_proj", "gate_up", "down")
REPLAYS = 20


def read(
    times: torch.Tensor, barriers: int
) -> tuple[dict[str, list[float]], float, float, float]:
    """One pass's per-barrier figures from its (programs, 2 * barriers + 2) timer table, in microseconds."""
    start = times[:, 0]
    arrive = times[:, 1 : 1 + 2 * barriers : 2]
    leave = times[:, 2 : 2 + 2 * barriers : 2]
    end = times[:, 2 * barriers + 1]
    figures: dict[str, list[float]] = defaultdict(list)
    previous_leave = start.min()
    for b in range(barriers):
        kind = KINDS[b % PHASES_PER_LAYER]
        last_arrival = arrive[:, b].max()
        figures[f"{kind} span"].append((last_arrival - previous_leave).item() / 1e3)
        figures[f"{kind} barrier latency"].append(
            (leave[:, b].min() - last_arrival).item() / 1e3
        )
        figures[f"{kind} departure spread"].append(
            (leave[:, b].max() - leave[:, b].min()).item() / 1e3
        )
        previous_leave = leave[:, b].min()
    figures["final norm span"].append((end.max() - previous_leave).item() / 1e3)
    return (
        figures,
        start.min().item(),
        end.max().item(),
        (end.max() - start.min()).item() / 1e3,
    )


def timeline(batch_size: int, prefetch: bool, device: torch.device) -> None:
    torch.manual_seed(batch_size)
    layers = make_layers(device)
    shape = make_shape(device)
    final_norm = SimpleNamespace(
        weight=(1 + 0.1 * torch.randn(HIDDEN, device=device)).to(torch.bfloat16),
        variance_epsilon=1e-6,
    )
    max_rows = min(2 * batch_size, MAX_FUSED_ROWS)
    has_pair = 2 * batch_size <= max_rows
    passes = SLOTS - 3 + has_pair
    runner = PersistentPredictorPass(
        layers, final_norm, shape, max_rows, device, prefetch, timeline_calls=passes
    )
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
    step = Step(runner.forward, batch_size, has_pair, inputs, device)
    step(record=False)
    torch.cuda.synchronize()
    runner.calls = 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(record=False)
    clear_slow_mode(device)
    replay_ms = replay_median_ms(graph, 50)
    barriers = PHASES_PER_LAYER * LAYERS
    figures: dict[str, list[float]] = defaultdict(list)
    spans, gaps, steps = [], [], []
    resolution = None
    for _ in range(REPLAYS):
        graph.replay()
        torch.cuda.synchronize()
        tables = runner.times.cpu()
        bounds = []
        step_span = 0.0
        for call in range(passes):
            pass_figures, first, last, span = read(tables[call], barriers)
            for key, values in pass_figures.items():
                figures[key].extend(values)
            bounds.append((first, last))
            step_span += span
        steps.append(step_span)
        spans.append((bounds[-1][1] - bounds[0][0]) / 1e3)
        gaps.extend((b[0] - a[1]) / 1e3 for a, b in zip(bounds, bounds[1:]))
        steps_ns = tables.flatten().sort().values.diff()
        nonzero = steps_ns[steps_ns > 0]
        smallest = int(nonzero.min()) if nonzero.numel() else 0
        resolution = smallest if resolution is None else min(resolution, smallest)
    print(
        f"bs {batch_size} prefetch {int(prefetch)}: {passes} passes, step replay {replay_ms:.3f} ms; first start to "
        f"last end {statistics.median(spans):.1f} us, kernel spans summed {statistics.median(steps):.1f} us, gap "
        f"between launches mean {statistics.mean(gaps):.2f} us; timer step {resolution} ns"
    )
    total = 0.0
    for kind in (*KINDS, "final norm"):
        row = []
        for figure in ("span", "barrier latency", "departure spread"):
            values = figures.get(f"{kind} {figure}")
            if values:
                row.append(f"{figure} {statistics.mean(values):6.2f}")
            else:
                pass
        per_pass = sum(
            statistics.mean(figures[f"{kind} {figure}"])
            for figure in ("span", "barrier latency")
            if f"{kind} {figure}" in figures
        ) * (LAYERS if kind != "final norm" else 1)
        total += per_pass
        print(f"  {kind:<10} " + ", ".join(row) + f" us; per pass {per_pass:6.1f} us")
    print(f"  sum of spans and latencies per pass {total:.1f} us")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 12, 31, 64])
    parser.add_argument("--prefetch", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    device = torch.device("cuda", args.card)
    torch.cuda.set_device(device)
    with torch.inference_mode():
        for batch_size in args.batch:
            for prefetch in args.prefetch:
                timeline(batch_size, bool(prefetch), device)


if __name__ == "__main__":
    main()
