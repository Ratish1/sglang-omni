"""The thinker MoE call with and without SGLang's fused SwiGLU epilogue, per token count (item 6, design 03 M1).

Both arms are SGLang's fused_experts as the thinker serves it (128 experts, top 8, hidden 2048, intermediate 768,
bf16): plain W13 (gate rows, then up rows) with the separate SwiGLU launch, against W13 permuted as
UnquantizedFusedMoEMethod._maybe_interleave_w13_for_fused_swiglu permutes it (gate and up of a channel in adjacent
rows) with fuse_swiglu_interleaved=True. Each call is captured in a CUDA graph and replayed with a 256 MB buffer
written before every replay (the median of 50 replays), on the same routing. Printed per token count: both times, the
byte floor at the given bandwidth, and whether the two outputs are equal bit for bit.

usage: python3 moe_swiglu_probe.py [--bandwidth-tbps 3.10] [--tokens 1 2 4 8 16 32]
"""

from __future__ import annotations

import argparse
import os

import torch
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import publish_build_topology

EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 768
EXPERT_BYTES = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
TIMED_REPLAYS = 50


def replay_time_us(fn, flush: torch.Tensor) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(TIMED_REPLAYS):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth-tbps", type=float, default=3.10)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29657")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    publish_build_topology()
    initialize_model_parallel(backend="gloo")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=device)
    torch.manual_seed(0)
    w13 = (torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=device) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device=device) * 0.02).to(
        torch.bfloat16
    )
    index = torch.empty(2 * INTERMEDIATE, dtype=torch.long, device=device)
    index[0::2] = torch.arange(0, INTERMEDIATE, device=device)
    index[1::2] = torch.arange(INTERMEDIATE, 2 * INTERMEDIATE, device=device)
    w13_interleaved = w13[:, index].contiguous()
    config = MoeRunnerConfig(
        num_experts=EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
    )
    for tokens in args.tokens:
        hidden_states = torch.randn(tokens, HIDDEN, device=device).to(torch.bfloat16)
        router_logits = torch.randn(tokens, EXPERTS, device=device).to(torch.bfloat16)
        top_logits, topk_ids = torch.topk(router_logits.float(), TOP_K, dim=-1)
        topk_weights = torch.softmax(top_logits, dim=-1)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            router_logits=router_logits,
        )
        experts = int(torch.unique(topk_ids).numel())
        floor_us = experts * EXPERT_BYTES / (args.bandwidth_tbps * 1e12) * 1e6

        def plain():
            return fused_experts(hidden_states, w13, w2, topk_output, config)

        def fused():
            return fused_experts(
                hidden_states,
                w13_interleaved,
                w2,
                topk_output,
                config,
                fuse_swiglu_interleaved=True,
            )

        equal = bool(torch.equal(plain(), fused()))
        plain_us = replay_time_us(plain, flush)
        fused_us = replay_time_us(fused, flush)
        print(
            f"tokens {tokens:4d} experts {experts:3d} floor {floor_us:6.1f} us | plain {plain_us:7.1f} us "
            f"| fused swiglu {fused_us:7.1f} us ({(fused_us / plain_us - 1) * 100:+5.1f} %) | bit equal {equal}",
            flush=True,
        )


if __name__ == "__main__":
    main()
