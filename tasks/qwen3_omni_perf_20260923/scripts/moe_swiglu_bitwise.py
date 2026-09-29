"""Bit equality and device time of sglang's fused SwiGLU up-GEMM epilogue against the
standalone activation, through sglang's fused_experts at the Qwen3-Omni thinker's MoE shape
(128 experts, top 8, hidden 2048, intermediate 768, bf16), at every decode graph bucket and
at prefill sizes, with the MoE config table the box resolves for each M.

usage: python3 moe_swiglu_bitwise.py
"""

from __future__ import annotations

import os

import torch
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
SIZES = (1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 128, 300, 512, 2048, 8192)
SEEDS = (0, 1, 2)
TIMED_REPLAYS = 50


def interleave_rows(w13: torch.Tensor) -> torch.Tensor:
    idx = torch.empty(w13.shape[1], dtype=torch.long, device=w13.device)
    idx[0::2] = torch.arange(0, INTER, device=w13.device)
    idx[1::2] = torch.arange(INTER, 2 * INTER, device=w13.device)
    return w13[:, idx].contiguous()


def run(x, w13, w2, weights, ids, logits, fuse: bool) -> torch.Tensor:
    topk_output = StandardTopKOutput(
        topk_weights=weights, topk_ids=ids, router_logits=logits
    )
    config = MoeRunnerConfig(
        num_experts=EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
    )
    return fused_experts(x, w13, w2, topk_output, config, fuse_swiglu_interleaved=fuse)


def graph_time_us(fn, flush: torch.Tensor) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    total = 0.0
    for _ in range(TIMED_REPLAYS):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    return total / TIMED_REPLAYS * 1000


def main() -> None:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=device)
    torch.manual_seed(0)
    w13 = (torch.randn(EXPERTS, 2 * INTER, HIDDEN, device=device) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(EXPERTS, HIDDEN, INTER, device=device) * 0.02).to(torch.bfloat16)
    w13_interleaved = interleave_rows(w13)
    for m in SIZES:
        up, (down, _) = try_get_optimal_moe_config(
            w13.shape, w2.shape, TOP_K, None, m, return_down_config=True
        )
        differ = 0
        for seed in SEEDS:
            torch.manual_seed(seed)
            x = torch.randn(m, HIDDEN, device=device).to(torch.bfloat16)
            logits = torch.randn(m, EXPERTS, device=device).to(torch.bfloat16)
            top_logits, ids = torch.topk(logits.float(), TOP_K, dim=-1)
            weights = torch.softmax(top_logits, dim=-1)
            ids = ids.to(torch.int32)
            ref = run(x.clone(), w13, w2, weights, ids, logits, False)
            got = run(x.clone(), w13_interleaved, w2, weights, ids, logits, True)
            differ += int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        plain_us = graph_time_us(
            lambda: run(x, w13, w2, weights, ids, logits, False), flush
        )
        fused_us = graph_time_us(
            lambda: run(x, w13_interleaved, w2, weights, ids, logits, True), flush
        )
        print(
            f"M {m:5d} config BM {up['BLOCK_SIZE_M']} BN {up['BLOCK_SIZE_N']} "
            f"BK {up['BLOCK_SIZE_K']} down {'same' if down is None or down == up else down} "
            f"| elements differing {differ} of {len(SEEDS) * m * HIDDEN} "
            f"| plain {plain_us:7.1f} us fused {fused_us:7.1f} us "
            f"({(fused_us / plain_us - 1) * 100:+.1f} %)"
        )


if __name__ == "__main__":
    main()
