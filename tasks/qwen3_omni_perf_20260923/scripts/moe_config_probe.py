"""Device time of the Qwen3-Omni thinker's MoE call per token count under the MoE config table
the process resolves (run once per table, with SGLANG_MOE_CONFIG_DIR pointing at an overlay or
unset for sglang's own), against the byte floor of the experts the routing touches.

The call is sglang's fused_experts as serving runs it (128 experts, top 8, hidden 2048,
intermediate 768, bf16, filter_expert off), captured in a CUDA graph and replayed with a 256 MB
buffer written before every replay, so no weight is served from L2 (the expert weights, 1.2 GB,
exceed it anyway). Routing is random top 8 of 128 per token (serving's routing is skewed, so
the floor here is the uniform case). Floor: distinct experts touched times their gate, up and
down bytes, at the card's HBM bandwidth.

usage: [SGLANG_MOE_CONFIG_DIR=overlay] python3 moe_config_probe.py --label NAME [--bandwidth-tbps 4.8]
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
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 768
EXPERT_BYTES = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
TIMED_REPLAYS = 50


def moe_call(hidden_states, w13, w2, topk_weights, topk_ids, router_logits):
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    config = MoeRunnerConfig(
        num_experts=EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=False,
    )
    return fused_experts(hidden_states, w13, w2, topk_output, config)


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
    parser.add_argument("--label", required=True)
    parser.add_argument("--bandwidth-tbps", type=float, default=4.8)
    args = parser.parse_args()
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29652")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    device = torch.device("cuda")
    print(
        f"{args.label}: {torch.cuda.get_device_name()}, config dir "
        f"{os.environ.get('SGLANG_MOE_CONFIG_DIR', 'sglang default')}"
    )
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=device)
    torch.manual_seed(0)
    w13 = (torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=device) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device=device) * 0.02).to(
        torch.bfloat16
    )
    for tokens in TOKEN_COUNTS:
        up, (down, _) = try_get_optimal_moe_config(
            w13.shape, w2.shape, TOP_K, None, tokens, return_down_config=True
        )
        hidden_states = torch.randn(tokens, HIDDEN, device=device).to(torch.bfloat16)
        router_logits = torch.randn(tokens, EXPERTS, device=device).to(torch.bfloat16)
        top_logits, topk_ids = torch.topk(router_logits.float(), TOP_K, dim=-1)
        topk_weights = torch.softmax(top_logits, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        experts = int(torch.unique(topk_ids).numel())
        floor_us = experts * EXPERT_BYTES / (args.bandwidth_tbps * 1e12) * 1e6
        time_us = replay_time_us(
            lambda: moe_call(
                hidden_states, w13, w2, topk_weights, topk_ids, router_logits
            ),
            flush,
        )
        up_tile = f"{up['BLOCK_SIZE_M']}x{up['BLOCK_SIZE_N']}x{up['BLOCK_SIZE_K']}"
        down_tile = (
            "same"
            if down is None or down == up
            else f"{down['BLOCK_SIZE_M']}x{down['BLOCK_SIZE_N']}x{down['BLOCK_SIZE_K']}"
        )
        print(
            f"tokens {tokens:5d} experts {experts:3d} up {up_tile} down {down_tile} "
            f"| {time_us:8.1f} us, floor {floor_us:7.1f} us ({floor_us / time_us * 100:5.1f} % of HBM)"
        )


if __name__ == "__main__":
    main()
