"""The Qwen3-Omni thinker's decode MoE call per kernel (run on the box): sglang's served Triton path
(fused_topk, then fused_experts: align, gate-up GEMM, SwiGLU, down GEMM, top-k sum) at 1 to 8
tokens, each kernel's device time from the profiler over eager calls with a 256 MB buffer written
before each call, so no weight comes from L2. Each GEMM is set against the byte floor of the experts
its tokens touch at the card's HBM bandwidth; the rest is set against nothing (it moves little).

usage: python3 moe_decode_breakdown.py [--bandwidth-tbps 4.8] [--tokens 1 2 4 8]
"""

from __future__ import annotations

import argparse
import collections
import os

import torch
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput, fused_topk
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 768
UP_BYTES = 2 * INTERMEDIATE * HIDDEN * 2
DOWN_BYTES = HIDDEN * INTERMEDIATE * 2
CALLS = 30


def family(name: str) -> str:
    for key, label in (
        ("fused_moe_kernel", "gemm"),
        ("align", "align"),
        ("sort", "align"),
        ("act_and_mul", "swiglu"),
        ("silu", "swiglu"),
        ("sum_reduce", "topk sum"),
        ("moe_sum", "topk sum"),
        ("gate", "router"),
        ("topk", "router"),
    ):
        if key in name:
            return label
        else:
            pass
    return name[:40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth-tbps", type=float, default=4.8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29655")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
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

        def call():
            topk_weights, topk_ids = fused_topk(
                hidden_states, router_logits, TOP_K, renormalize=True
            )
            output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
            return fused_experts(hidden_states, w13, w2, output, config), topk_ids

        _, topk_ids = call()
        experts = int(torch.unique(topk_ids).numel())
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        totals = collections.defaultdict(float)
        counts = collections.Counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(CALLS):
                flush.zero_()
                call()
            torch.cuda.synchronize()
        gemms = 0
        device_events = [e for e in profile.events() if e.device_type.name == "CUDA"]
        for event in sorted(device_events, key=lambda e: e.time_range.start):
            label = family(event.name)
            if label == "gemm":
                # each call runs the gate-up GEMM, then the down GEMM
                label = "gemm up" if gemms % 2 == 0 else "gemm down"
                gemms += 1
            else:
                pass
            totals[label] += event.device_time
            counts[label] += 1
        up_floor = experts * UP_BYTES / (args.bandwidth_tbps * 1e12) * 1e6
        down_floor = experts * DOWN_BYTES / (args.bandwidth_tbps * 1e12) * 1e6
        print(
            f"tokens {tokens} experts {experts}: gemm floors up {up_floor:.1f} us, down "
            f"{down_floor:.1f} us"
        )
        for label in sorted(totals, key=lambda k: -totals[k]):
            per_call = totals[label] / CALLS
            print(
                f"    {label:<12}{per_call:7.1f} us per call  "
                f"({counts[label] / CALLS:.0f} kernels)"
            )


if __name__ == "__main__":
    main()
