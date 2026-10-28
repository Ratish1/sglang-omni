"""Shared memory the fused predictor's launches need, by target architecture, row block and pipeline stages: each
launch wrapper of the tree's predictor_layers is called with the code predictor's shapes, Triton compiles it for the
chosen target without launching (so a target with a smaller block budget compiles on an H100), and the compiled
kernel's metadata gives the bytes. A target below sm90 compiles without programmatic dependent launch, as there.

usage: PYTHONPATH=TREE python3 predictor_smem.py --arch 89     (one process per architecture; prints one line a kernel)
"""

import argparse
from types import SimpleNamespace

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import jit

from sglang_omni.utils import predictor_layers
from sglang_omni.utils.predictor_layers import PredictorLayerShape

HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE = 3072
SLOTS = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=int, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    target = GPUTarget("cuda", args.arch, 32)
    triton.runtime.driver.active.get_current_target = lambda: target
    predictor_layers.is_arch_support_pdl = lambda: args.arch >= 90
    compiled = []
    original_run = jit.JITFunction.run

    def compile_only(self, *run_args, grid, warmup, **kwargs):
        kernel = original_run(self, *run_args, grid=grid, warmup=True, **kwargs)
        compiled.append(
            (self.fn.__name__, kwargs["K"], kwargs["BLOCK_M"], kwargs["T"], kwargs["num_stages"], kernel.metadata.shared)
        )
        return kernel

    jit.JITFunction.run = compile_only
    bf16 = dict(device=device, dtype=torch.bfloat16)
    heads = NUM_HEADS + 2 * NUM_KV_HEADS
    norm = SimpleNamespace(weight=torch.ones(HIDDEN, **bf16), variance_epsilon=1e-6)
    attention = SimpleNamespace(
        qkv_proj=SimpleNamespace(weight=torch.zeros(heads * HEAD_DIM, HIDDEN, **bf16)),
        q_norm=SimpleNamespace(weight=torch.ones(HEAD_DIM, **bf16), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(HEAD_DIM, **bf16), variance_epsilon=1e-6),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(SLOTS, HEAD_DIM, device=device, dtype=torch.float32)),
    )
    for num_stages in (2, 3):
        shape = PredictorLayerShape(
            hidden_size=HIDDEN, head_dim=HEAD_DIM, num_q_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
            intermediate_size=INTERMEDIATE, split_qkv=1, split_hidden=1, num_stages=num_stages,
        )
        for rows in (16, 32, 64):
            partials, sum_sq_partials, counters = predictor_layers.allocate_split_scratch(shape, rows, device)
            for tokens_per_row in (1, 2):
                k_cache = torch.zeros(rows // tokens_per_row, SLOTS, NUM_KV_HEADS, HEAD_DIM, **bf16)
                predictor_layers.attention_inputs(
                    x=torch.zeros(rows, HIDDEN, **bf16), tokens_per_row=tokens_per_row, norm=norm,
                    attention=attention, q_out=torch.zeros(rows, NUM_HEADS * HEAD_DIM, **bf16),
                    positions=torch.zeros(rows, device=device, dtype=torch.int64), k_cache=k_cache,
                    v_cache=torch.zeros_like(k_cache), partials=partials, sum_sq_partials=sum_sq_partials,
                    counters=counters, shape=shape,
                )
                attention_output = torch.zeros(rows // tokens_per_row, tokens_per_row, NUM_HEADS, HEAD_DIM, **bf16)
                predictor_layers.o_proj_add(
                    attention_output=attention_output.transpose(1, 2), tokens_per_row=tokens_per_row,
                    weight=torch.zeros(HIDDEN, NUM_HEADS * HEAD_DIM, **bf16),
                    residual_in=torch.zeros(rows, HIDDEN, **bf16), residual_out=torch.zeros(rows, HIDDEN, **bf16),
                    partials=partials, counters=counters, shape=shape,
                )
            residual = torch.zeros(rows, HIDDEN, **bf16)
            activated = torch.zeros(rows, INTERMEDIATE, **bf16)
            predictor_layers.mlp_up(
                residual=residual, norm=norm, weight=torch.zeros(2 * INTERMEDIATE, HIDDEN, **bf16),
                activated=activated, shape=shape,
            )
            predictor_layers.down_add(
                activated=activated, weight=torch.zeros(HIDDEN, INTERMEDIATE, **bf16), residual=residual,
                partials=partials, counters=counters, shape=shape,
            )
    jit.JITFunction.run = original_run
    launches = {"norm_qkv_rope_store_kernel": "attention_inputs", "norm_gate_up_silu_kernel": "mlp_up"}
    seen = set()
    for name, k, block_m, tokens_per_row, num_stages, shared in compiled:
        if name in launches:
            launch = launches[name]
        elif k == NUM_HEADS * HEAD_DIM:
            launch = "o_proj_add"
        else:
            launch = "down_add"
        key = (launch, block_m, tokens_per_row, num_stages, shared)
        if key in seen:
            continue
        else:
            seen.add(key)
        print(f"sm{args.arch} {launch} BLOCK_M {block_m} T {tokens_per_row} stages {num_stages}: {shared} bytes", flush=True)


if __name__ == "__main__":
    main()
