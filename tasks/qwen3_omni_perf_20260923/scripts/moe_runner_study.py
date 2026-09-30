"""The Qwen3-Omni thinker's whole MoE call per token count on each MoE runner the pinned sglang
offers, each in the weight layout that runner loads, with routing included (run on the box).

Runners: the served Triton path (sglang's fused router top-k, then fused_experts on w13 [E, 2I, H]
gate half first and w2 [E, H, I]); triton_kernels (sglang's routing(), then
triton_kernel_fused_experts on the transposed w13 [E, H, 2I] and w2 [E, I, H], then the top-k
combine the runner applies); flashinfer's
cutlass fused MoE (torch top-k, then cutlass_fused_moe on w13 up half first, its tactic autotuned
per token count first, as serving does at startup). Every path is
captured in a CUDA graph and replayed with a 256 MB buffer written before every replay; the
median of 50 replays is kept. Each path's output is compared with the served path's on the same
input (max absolute difference and the relative Frobenius error), so a faster path that computes
something else is caught. Routing is random logits over 128 experts, top 8, renormalized.

usage: python3 moe_runner_study.py [--bandwidth-tbps 4.8] [--tokens 1 2 4 ...]
"""

from __future__ import annotations

import argparse
import os

import torch
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
    triton_kernel_fused_experts,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput, fused_topk, routing
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

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
    parser.add_argument("--bandwidth-tbps", type=float, default=4.8)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 8, 16, 64, 128, 512, 2048, 8192],
    )
    args = parser.parse_args()
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29653")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import cutlass_fused_moe

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
    w13_transposed = w13.transpose(1, 2).contiguous()
    w2_transposed = w2.transpose(1, 2).contiguous()
    w13_up_first = torch.cat([w13[:, INTERMEDIATE:], w13[:, :INTERMEDIATE]], dim=1)
    runner_config = MoeRunnerConfig(
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

        def served():
            topk_weights, topk_ids = fused_topk(
                hidden_states, router_logits, TOP_K, renormalize=True
            )
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
            return fused_experts(hidden_states, w13, w2, topk_output, runner_config)

        def triton_kernels_path():
            ragged, gather, scatter, gate, active = routing(
                router_logits, TOP_K, sm_first=False
            )
            # the runner combines the top-k rows after the kernel
            # (moe_runner/triton_kernels.py, TritonKernelsRunner.run)
            return (
                triton_kernel_fused_experts(
                    hidden_states,
                    w13_transposed,
                    w2_transposed,
                    ragged,
                    gather,
                    scatter,
                    gate,
                    active,
                )
                .view(tokens, TOP_K, HIDDEN)
                .sum(dim=1)
            )

        cutlass_output = torch.empty(
            tokens, HIDDEN, device=device, dtype=torch.bfloat16
        )

        def cutlass_path():
            top_logits, topk_ids = torch.topk(router_logits.float(), TOP_K, dim=-1)
            topk_weights = torch.softmax(top_logits, dim=-1)
            return cutlass_fused_moe(
                input=hidden_states,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=w13_up_first,
                fc2_expert_weights=w2,
                output_dtype=torch.bfloat16,
                quant_scales=None,
                output=cutlass_output,
            )

        # flashinfer picks the cutlass tactic by autotuning, as serving does at startup
        with autotune(True):
            cutlass_path()
        torch.cuda.synchronize()
        reference = served().float()
        _, topk_ids = torch.topk(router_logits.float(), TOP_K, dim=-1)
        experts = int(torch.unique(topk_ids).numel())
        floor_us = experts * EXPERT_BYTES / (args.bandwidth_tbps * 1e12) * 1e6
        row = [f"tokens {tokens:5d} experts {experts:3d} floor {floor_us:6.1f} us"]
        for name, fn in (
            ("served", served),
            ("triton_kernels", triton_kernels_path),
            ("cutlass", cutlass_path),
        ):
            try:
                out = fn()
                out = (out[0] if isinstance(out, (list, tuple)) else out).float()
                error = float((out - reference).norm() / reference.norm())
                time_us = replay_time_us(fn, flush)
                row.append(
                    f"{name} {time_us:7.1f} us ({floor_us / time_us * 100:4.0f} %) err {error:.1e}"
                )
            except Exception as exc:
                row.append(f"{name} failed {type(exc).__name__}: {str(exc)[:60]}")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
