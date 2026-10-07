"""Device time of the Qwen3-Omni code predictor's decode step at a served row count, replayed
from a CUDA graph as the talker's decode graph replays it.

The predictor is built at the checkpoint's shapes from the unit test's layer fixture (seeded
weights; the fused layers built as the model loader builds them), plus the codebook glue the
talker runs per step: the layer-0 codec embedding, 15 lm_heads (hidden to 2048), argmax, the code
copy, 15 codebook embeddings and the embedding sum. The step is the talker's own
code_predictor_forward_incremental_eager, the function the talker's decode graph captures. No
kernel depends on weight values (argmax over 2048 logits, no routing), so seeded weights time
like the checkpoint's.

time: per batch, the replay wall from CUDA events over 200 replays (mean and p50 of single
replays), then the torch profiler over 20 replays: kernels per step and device us per step by
kernel name. The served per-kernel times (graph_launch_kernels.py on a probed serve) are the
known answer this reproduces before its ncu counters are used.
ncu: per batch, one eager step after warmup inside the NVTX range ncu.b<batch>, for
  ncu --nvtx --nvtx-include "ncu.b<batch>/" --metrics ... --csv python3 predictor_step_probe.py ncu --batches <batch>

usage, from the tools tree with PYTHONPATH set to the tree under test:
  python3 predictor_step_probe.py time [--batches 1,4,8,16,32]
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from dataclasses import replace
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch import nn
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.utils import predictor_layers
from tests.unit_test.fixtures.qwen_predictor import TupleLinear
from tests.unit_test.qwen3_omni.test_predictor_kernels import (
    DTYPE,
    HIDDEN,
    MAX_BS,
    NUM_CODE_GROUPS,
    build_talker,
    fuse,
)

PREDICTOR_VOCAB = 2048
TALKER_VOCAB = 3072


def build_predictor_step(device: torch.device) -> Qwen3OmniTalker:
    """The talker with its predictor layers fused and the per-step glue attached."""
    talker = fuse(build_talker(device, seed=3, max_bs=MAX_BS))
    talker.code_predictor.lm_head = [
        TupleLinear(HIDDEN, PREDICTOR_VOCAB).to(device, DTYPE)
        for _ in range(NUM_CODE_GROUPS - 1)
    ]
    talker.code_predictor.model.codec_embedding = [
        nn.Embedding(PREDICTOR_VOCAB, HIDDEN).to(device, DTYPE)
        for _ in range(NUM_CODE_GROUPS - 1)
    ]
    layer0_embedding = nn.Embedding(TALKER_VOCAB, HIDDEN).to(device, DTYPE)
    talker.get_input_embeddings = lambda: layer0_embedding
    talker.config = SimpleNamespace(num_code_groups=NUM_CODE_GROUPS)
    talker.predictor_input_buffer = torch.zeros(
        MAX_BS, 2, HIDDEN, device=device, dtype=DTYPE
    )
    talker.output_codes = torch.zeros(
        MAX_BS, NUM_CODE_GROUPS, device=device, dtype=torch.long
    )
    talker.output_embeds = torch.zeros(MAX_BS, HIDDEN, device=device, dtype=DTYPE)
    # a tree with the fused codebook step gates it at load as the talker does
    supports_codebook_step = getattr(predictor_layers, "supports_codebook_step", None)
    if supports_codebook_step is not None:
        talker.predictor_fused_codebook_step = supports_codebook_step(
            talker.code_predictor.model.codec_embedding[0].weight, DTYPE
        )
    else:
        pass
    return talker


def step_inputs(device: torch.device, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(batch)
    layer0_codes = torch.randint(
        0, TALKER_VOCAB, (batch, 1), device=device, generator=generator
    )
    talker_hidden = torch.randn(
        batch, 1, HIDDEN, device=device, generator=generator
    ).to(DTYPE)
    return layer0_codes, talker_hidden


def capture_step(
    talker: Qwen3OmniTalker, device: torch.device, batch: int
) -> torch.cuda.CUDAGraph:
    layer0_codes, talker_hidden = step_inputs(device, batch)

    def step() -> None:
        talker.code_predictor_forward_incremental_eager(
            layer0_codes=layer0_codes, talker_hidden=talker_hidden
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        step()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_p50_us(graph: torch.cuda.CUDAGraph, replays: int = 100) -> float:
    singles = []
    for _ in range(replays):
        one_begin = torch.cuda.Event(enable_timing=True)
        one_end = torch.cuda.Event(enable_timing=True)
        one_begin.record()
        graph.replay()
        one_end.record()
        torch.cuda.synchronize()
        singles.append(one_begin.elapsed_time(one_end) * 1000)
    return statistics.median(singles)


def kernel_table(graph: torch.cuda.CUDAGraph) -> list[tuple[str, float, float]]:
    """(name, launches per step, device us per step), largest first."""
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
    events = [e for e in trace.key_averages() if e.device_type.name == "CUDA"]
    return sorted(
        ((e.key, e.count / 20, e.self_device_time_total / 20) for e in events),
        key=lambda item: -item[2],
    )


def time_batch(talker: Qwen3OmniTalker, device: torch.device, batch: int) -> None:
    graph = capture_step(talker, device, batch)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    begin.record()
    for _ in range(200):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    p50 = replay_p50_us(graph)
    kernels = kernel_table(graph)
    device_us = sum(us for _, _, us in kernels)
    print(
        f"batch {batch:2d}: replay {begin.elapsed_time(end) * 1000 / 200:7.1f} us mean,"
        f" {p50:7.1f} us p50 single; kernels {sum(c for _, c, _ in kernels):5.0f}"
        f" per step, device sum {device_us:7.1f} us ({device_us / p50:.0%} of p50)"
    )
    for name, count, us in kernels:
        print(f"    {us:7.1f} us  {count:4.0f} x {us / count:6.2f}  {name[:110]}")


def sweep(talker: Qwen3OmniTalker, device: torch.device, batches: list[int]) -> None:
    """Replay p50 per batch over the fused layers' launch constants: warps, stages,
    the output tile, and the K splits of the qkv and residual launches."""
    shape = talker.predictor_fused_layers.shape
    max_rows = talker.predictor_fused_layers.max_rows
    results = []
    for warps, stages, block_n, split_qkv, split_hidden in itertools.product(
        (4, 8), (2, 3, 4), (16, 32), (2, 4, 8), (2, 4, 8)
    ):
        predictor_layers.NUM_WARPS = warps
        predictor_layers.NUM_STAGES = stages
        predictor_layers.BLOCK_N = block_n
        talker.predictor_fused_layers = predictor_layers.FusedPredictorLayers(
            replace(shape, split_qkv=split_qkv, split_hidden=split_hidden),
            max_rows,
            device,
            DTYPE,
        )
        config = (warps, stages, block_n, split_qkv, split_hidden)
        try:
            times = [
                replay_p50_us(capture_step(talker, device, b), 50) for b in batches
            ]
        except (
            Exception
        ) as exc:  # noqa: BLE001  # a launch configuration the kernels refuse is a sweep result
            print(
                f"config {config}: failed {type(exc).__name__}: {str(exc)[:120]}",
                flush=True,
            )
            continue
        results.append((config, times))
        print(f"config {config}:" + "".join(f" {t:7.1f}" for t in times), flush=True)
    print(
        "best by the sum over batches (warps, stages, block_n, split_qkv, split_hidden):"
    )
    for config, times in sorted(results, key=lambda r: sum(r[1]))[:10]:
        print(f"  {config}:" + "".join(f" {t:7.1f}" for t in times))
    best = min(results, key=lambda r: sum(r[1]))[0]
    warps, stages, block_n, split_qkv, split_hidden = best
    predictor_layers.NUM_WARPS = warps
    predictor_layers.NUM_STAGES = stages
    predictor_layers.BLOCK_N = block_n
    talker.predictor_fused_layers = predictor_layers.FusedPredictorLayers(
        replace(shape, split_qkv=split_qkv, split_hidden=split_hidden),
        max_rows,
        device,
        DTYPE,
    )
    for batch in batches:
        print(f"best config {best}, batch {batch}:")
        for name, count, us in kernel_table(capture_step(talker, device, batch))[:8]:
            print(f"    {us:7.1f} us  {count:4.0f} x {us / count:6.2f}  {name[:90]}")


def dump_batches(
    talker: Qwen3OmniTalker, device: torch.device, batches: list[int], out: str
) -> None:
    """One eager step per batch; the codes and summed embeddings saved for a bitwise
    comparison between trees."""
    results = {}
    with torch.no_grad():
        for batch in batches:
            layer0_codes, talker_hidden = step_inputs(device, batch)
            codes, embeds = talker.code_predictor_forward_incremental_eager(
                layer0_codes=layer0_codes, talker_hidden=talker_hidden
            )
            results[batch] = (codes.cpu().clone(), embeds.cpu().clone())
    torch.save(results, out)
    print(f"dumped batches {batches} to {out}")


def ncu_batch(talker: Qwen3OmniTalker, device: torch.device, batch: int) -> None:
    layer0_codes, talker_hidden = step_inputs(device, batch)
    with torch.no_grad():
        for _ in range(3):
            talker.code_predictor_forward_incremental_eager(
                layer0_codes=layer0_codes, talker_hidden=talker_hidden
            )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"ncu.b{batch}")
        talker.code_predictor_forward_incremental_eager(
            layer0_codes=layer0_codes, talker_hidden=talker_hidden
        )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("time", "ncu", "sweep", "dump"))
    parser.add_argument("--batches", default="1,4,8,16,32")
    parser.add_argument("--out", default="predictor_step.pt")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="NAME=VALUE: a predictor_layers module constant for this run (True, False or an int)",
    )
    parser.add_argument(
        "--talker",
        action="append",
        default=[],
        help="NAME=True|False: a talker attribute after the build (predictor_fused_codebook_step)",
    )
    args = parser.parse_args()
    for assignment in args.set:
        name, value = assignment.split("=", 1)
        parsed = {"True": True, "False": False}.get(value)
        setattr(predictor_layers, name, int(value) if parsed is None else parsed)
        print(f"predictor_layers.{name} = {getattr(predictor_layers, name)}")
    parser_talker = [assignment.split("=", 1) for assignment in args.talker]
    device = torch.device("cuda")
    talker = build_predictor_step(device)
    for name, value in parser_talker:
        setattr(talker, name, {"True": True, "False": False}[value])
        print(f"talker.{name} = {getattr(talker, name)}")
    print(
        torch.cuda.get_device_name(device),
        "fused shape",
        talker.predictor_fused_layers.shape,
    )
    batches = [int(value) for value in args.batches.split(",")]
    if args.mode == "sweep":
        sweep(talker, device, batches)
    elif args.mode == "dump":
        dump_batches(talker, device, batches, args.out)
    else:
        for batch in batches:
            if args.mode == "time":
                time_batch(talker, device, batch)
            else:
                ncu_batch(talker, device, batch)


if __name__ == "__main__":
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        main()
