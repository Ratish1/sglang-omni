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
import statistics
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


def time_batch(talker: Qwen3OmniTalker, device: torch.device, batch: int) -> None:
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
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    begin.record()
    for _ in range(200):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    singles = []
    for _ in range(100):
        one_begin = torch.cuda.Event(enable_timing=True)
        one_end = torch.cuda.Event(enable_timing=True)
        one_begin.record()
        graph.replay()
        one_end.record()
        torch.cuda.synchronize()
        singles.append(one_begin.elapsed_time(one_end) * 1000)
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
    events = [e for e in trace.key_averages() if e.device_type.name == "CUDA"]
    kernels = sorted(
        ((e.key, e.count / 20, e.self_device_time_total / 20) for e in events),
        key=lambda item: -item[2],
    )
    device_us = sum(us for _, _, us in kernels)
    print(
        f"batch {batch:2d}: replay {begin.elapsed_time(end) * 1000 / 200:7.1f} us mean,"
        f" {statistics.median(singles):7.1f} us p50 single; kernels {sum(c for _, c, _ in kernels):5.0f}"
        f" per step, device sum {device_us:7.1f} us ({device_us / statistics.median(singles):.0%} of p50)"
    )
    for name, count, us in kernels:
        print(f"    {us:7.1f} us  {count:4.0f} x {us / count:6.2f}  {name[:110]}")


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
    parser.add_argument("mode", choices=("time", "ncu"))
    parser.add_argument("--batches", default="1,4,8,16,32")
    args = parser.parse_args()
    device = torch.device("cuda")
    talker = build_predictor_step(device)
    print(
        torch.cuda.get_device_name(device),
        "fused shape",
        talker.predictor_fused_layers.shape,
    )
    for batch in (int(value) for value in args.batches.split(",")):
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
