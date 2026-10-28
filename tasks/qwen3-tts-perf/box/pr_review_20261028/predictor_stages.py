"""The fused predictor's launches at two and three pipeline stages on this GPU: whether the outputs are bit identical
and what each launch costs, per row block, with the code predictor's shapes, random weights and inputs, and one K
split so the launches need no cross-program reduction. Times are CUDA-graph replays (triton.testing).

usage: PYTHONPATH=TREE python3 predictor_stages.py
"""

from types import SimpleNamespace

import torch
from triton.testing import do_bench_cudagraph

from sglang_omni.utils import predictor_layers
from sglang_omni.utils.predictor_layers import PredictorLayerShape

HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE = 3072
SLOTS = 32


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    bf16 = dict(device=device, dtype=torch.bfloat16)
    heads = NUM_HEADS + 2 * NUM_KV_HEADS
    norm = SimpleNamespace(weight=torch.rand(HIDDEN, **bf16) + 0.5, variance_epsilon=1e-6)
    attention = SimpleNamespace(
        qkv_proj=SimpleNamespace(weight=torch.randn(heads * HEAD_DIM, HIDDEN, **bf16) * 0.02),
        q_norm=SimpleNamespace(weight=torch.rand(HEAD_DIM, **bf16) + 0.5, variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.rand(HEAD_DIM, **bf16) + 0.5, variance_epsilon=1e-6),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.rand(SLOTS, HEAD_DIM, device=device, dtype=torch.float32)),
    )
    o_weight = torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM, **bf16) * 0.02
    gate_up_weight = torch.randn(2 * INTERMEDIATE, HIDDEN, **bf16) * 0.02
    down_weight = torch.randn(HIDDEN, INTERMEDIATE, **bf16) * 0.02
    for rows in (16, 32, 64):
        x = torch.randn(rows, HIDDEN, **bf16)
        attention_output = torch.randn(rows, 1, NUM_HEADS, HEAD_DIM, **bf16)
        activated_in = torch.randn(rows, INTERMEDIATE, **bf16)
        positions = torch.randint(0, SLOTS, (rows,), device=device)
        results = {}
        for num_stages in (3, 2):
            shape = PredictorLayerShape(
                hidden_size=HIDDEN, head_dim=HEAD_DIM, num_q_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                intermediate_size=INTERMEDIATE, split_qkv=1, split_hidden=1, num_stages=num_stages,
            )
            partials, sum_sq_partials, counters = predictor_layers.allocate_split_scratch(shape, rows, device)
            q_out = torch.zeros(rows, NUM_HEADS * HEAD_DIM, **bf16)
            k_cache = torch.zeros(rows, SLOTS, NUM_KV_HEADS, HEAD_DIM, **bf16)
            v_cache = torch.zeros_like(k_cache)
            residual_out = torch.zeros(rows, HIDDEN, **bf16)
            activated = torch.zeros(rows, INTERMEDIATE, **bf16)
            down_residual = x.clone()
            launches = {
                "attention_inputs": lambda: predictor_layers.attention_inputs(
                    x=x, tokens_per_row=1, norm=norm, attention=attention, q_out=q_out, positions=positions,
                    k_cache=k_cache, v_cache=v_cache, partials=partials, sum_sq_partials=sum_sq_partials,
                    counters=counters, shape=shape,
                ),
                "o_proj_add": lambda: predictor_layers.o_proj_add(
                    attention_output=attention_output.transpose(1, 2), tokens_per_row=1, weight=o_weight,
                    residual_in=x, residual_out=residual_out, partials=partials, counters=counters, shape=shape,
                ),
                "mlp_up": lambda: predictor_layers.mlp_up(
                    residual=x, norm=norm, weight=gate_up_weight, activated=activated, shape=shape,
                ),
                "down_add": lambda: predictor_layers.down_add(
                    activated=activated_in, weight=down_weight, residual=down_residual, partials=partials,
                    counters=counters, shape=shape,
                ),
            }
            for name, launch in launches.items():
                down_residual.copy_(x)
                launch()
                torch.cuda.synchronize()
                outputs = {
                    "attention_inputs": (q_out, k_cache, v_cache),
                    "o_proj_add": (residual_out,),
                    "mlp_up": (activated,),
                    "down_add": (down_residual,),
                }[name]
                first_outputs = [output.clone() for output in outputs]
                time_us = 1000 * do_bench_cudagraph(launch, rep=200)
                results[(name, num_stages)] = (first_outputs, time_us)
        for name in ("attention_inputs", "o_proj_add", "mlp_up", "down_add"):
            three, time_three = results[(name, 3)]
            two, time_two = results[(name, 2)]
            identical = all(torch.equal(a, b) for a, b in zip(three, two))
            print(
                f"rows {rows} {name}: bit identical {identical}, 3 stages {time_three:.2f} us, "
                f"2 stages {time_two:.2f} us ({100 * (time_two / time_three - 1):+.1f} %)",
                flush=True,
            )


if __name__ == "__main__":
    main()
