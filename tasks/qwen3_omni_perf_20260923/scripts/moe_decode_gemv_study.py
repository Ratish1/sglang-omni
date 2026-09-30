"""Can a decode-shaped MoE reach the byte floor where sglang's Triton MoE does not (run on the box)?

The thinker's MoE at 1 to 16 tokens (128 experts, top 8, hidden 2048, intermediate 768, bf16) two
ways, each captured in a CUDA graph and replayed with a 256 MB buffer written before every replay:
  - served: sglang's fused_experts (align, gate-up GEMM, SwiGLU, down GEMM, top-k sum) on the routing
    of fused_topk;
  - gemv: two Triton kernels on the same routing. gate_up_swiglu: one program per (token, expert) pair
    and intermediate tile reads the gate and up rows of that tile and writes silu(gate) * up.
    down_combine: one program per (token, hidden tile) runs the 8 experts of its token in order and
    writes the weighted sum; no atomics, a fixed order. Rounding follows the served path: each GEMM
    output rounded to bf16, SwiGLU in fp32 rounded to bf16, the routed weight applied in fp32 and the
    product rounded to bf16 per expert, the 8 products summed in fp32.
Each gemv shape runs over a small grid of tile sizes and warps; the fastest is kept. Printed: time
of each path and its share of the byte floor (the distinct experts' bytes at the HBM bandwidth), and
the gemv output's distance to the served one and both distances to an fp32 reference.

usage: python3 moe_decode_gemv_study.py [--bandwidth-tbps 4.8] [--tokens 1 2 3 4 8 16]
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
import triton
import triton.language as tl
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
EXPERT_BYTES = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
TIMED_REPLAYS = 50


@triton.jit
def gate_up_swiglu_kernel(
    x_ptr,
    w13_ptr,
    topk_ids_ptr,
    h_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    token = pair // TOPK
    expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
    rows = tile * BLOCK_I + tl.arange(0, BLOCK_I)
    base = w13_ptr + expert * (2 * INTERMEDIATE_SIZE * HIDDEN_SIZE)
    gate = tl.zeros([BLOCK_I], dtype=tl.float32)
    up = tl.zeros([BLOCK_I], dtype=tl.float32)
    for start in range(0, HIDDEN_SIZE, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + token * HIDDEN_SIZE + cols).to(tl.float32)
        w_gate = tl.load(base + rows[:, None] * HIDDEN_SIZE + cols[None, :])
        w_up = tl.load(
            base + (INTERMEDIATE_SIZE + rows)[:, None] * HIDDEN_SIZE + cols[None, :]
        )
        gate += tl.sum(w_gate.to(tl.float32) * x[None, :], axis=1)
        up += tl.sum(w_up.to(tl.float32) * x[None, :], axis=1)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    h = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(h_ptr + pair * INTERMEDIATE_SIZE + rows, h.to(tl.bfloat16))


@triton.jit
def down_combine_kernel(
    h_ptr,
    w2_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    rows = tile * BLOCK_H + tl.arange(0, BLOCK_H)
    total = tl.zeros([BLOCK_H], dtype=tl.float32)
    for slot in tl.static_range(TOPK):
        pair = token * TOPK + slot
        expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
        weight = tl.load(topk_weights_ptr + pair)
        base = w2_ptr + expert * (HIDDEN_SIZE * INTERMEDIATE_SIZE)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for start in range(0, INTERMEDIATE_SIZE, BLOCK_I):
            cols = start + tl.arange(0, BLOCK_I)
            h = tl.load(h_ptr + pair * INTERMEDIATE_SIZE + cols).to(tl.float32)
            w = tl.load(base + rows[:, None] * INTERMEDIATE_SIZE + cols[None, :])
            acc += tl.sum(w.to(tl.float32) * h[None, :], axis=1)
        total += (acc * weight).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + token * HIDDEN_SIZE + rows, total.to(tl.bfloat16))


def gemv_moe(x, w13, w2, topk_ids, topk_weights, h, out, shape):
    block_i, block_k, warps_up, block_h, block_i_down, warps_down = shape
    tokens = x.shape[0]
    gate_up_swiglu_kernel[(tokens * TOP_K, INTERMEDIATE // block_i)](
        x,
        w13,
        topk_ids,
        h,
        HIDDEN,
        INTERMEDIATE,
        TOP_K,
        block_i,
        block_k,
        num_warps=warps_up,
    )
    down_combine_kernel[(tokens, HIDDEN // block_h)](
        h,
        w2,
        topk_ids,
        topk_weights,
        out,
        HIDDEN,
        INTERMEDIATE,
        TOP_K,
        block_h,
        block_i_down,
        num_warps=warps_down,
    )
    return out


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


def fp32_reference(x, w13, w2, topk_ids, topk_weights) -> torch.Tensor:
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(TOP_K):
            expert = int(topk_ids[token, slot])
            gate_up = w13[expert].float() @ x[token].float()
            gate, up = gate_up[:INTERMEDIATE], gate_up[INTERMEDIATE:]
            h = torch.nn.functional.silu(gate) * up
            out[token] += float(topk_weights[token, slot]) * (w2[expert].float() @ h)
    return out


def distance(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"rel {float((a.float() - b).norm() / b.norm()):.1e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth-tbps", type=float, default=4.8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16])
    args = parser.parse_args()
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29657")
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
    # tile rows, K per load and warps: more programs and longer loads put more bytes in flight
    shapes = [
        (block_i, block_k, warps_up, block_h, block_k_down, warps_down)
        for block_i, block_k, warps_up, block_h, block_k_down, warps_down in itertools.product(
            (4, 8), (256, 1024), (4, 8), (4, 8), (256,), (4, 8)
        )
    ]
    for tokens in args.tokens:
        x = torch.randn(tokens, HIDDEN, device=device).to(torch.bfloat16)
        router_logits = torch.randn(tokens, EXPERTS, device=device).to(torch.bfloat16)
        topk_weights, topk_ids = fused_topk(x, router_logits, TOP_K, renormalize=True)
        topk_ids = topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_weights.float().contiguous()
        routing = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        h = torch.empty(
            tokens * TOP_K, INTERMEDIATE, device=device, dtype=torch.bfloat16
        )
        out = torch.empty(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        experts = int(torch.unique(topk_ids).numel())
        floor_us = experts * EXPERT_BYTES / (args.bandwidth_tbps * 1e12) * 1e6

        def served():
            return fused_experts(x, w13, w2, routing, config)

        served_us = replay_time_us(served, flush)
        best = None
        for shape in shapes:
            time_us = replay_time_us(
                lambda shape=shape: gemv_moe(
                    x, w13, w2, topk_ids, topk_weights, h, out, shape
                ),
                flush,
            )
            if best is None or time_us < best[0]:
                best = (time_us, shape)
            else:
                pass
        gemv_us, shape = best
        reference = fp32_reference(x, w13, w2, topk_ids, topk_weights)
        served_out = served().float()
        gemv_out = gemv_moe(x, w13, w2, topk_ids, topk_weights, h, out, shape).float()
        print(
            f"tokens {tokens:2d} experts {experts:3d} floor {floor_us:6.1f} us | served "
            f"{served_us:6.1f} us ({floor_us / served_us * 100:3.0f} %) | gemv {gemv_us:6.1f} "
            f"us ({floor_us / gemv_us * 100:3.0f} %) shape {shape} | gemv vs served "
            f"{distance(gemv_out, served_out)}, vs fp32 served {distance(served_out, reference)} "
            f"gemv {distance(gemv_out, reference)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
