"""One predictor operator at one batch inside a cudaProfilerStart/Stop window, for Nsight Compute.

Usage: ncu --profile-from-start off ... python3 ncu_op.py --op {o_proj,qkv,gate_up,cutedsl_post} --rows N
"""

import argparse

import torch
from predictor_bench import CACHED, HEAD_DIM, HIDDEN, KV_HEADS, Q_HEADS, SLOTS, load_predictor, rope_cache
from predictor_experiments import build_layers, make_shape

from sglang_omni.utils import predictor_layers as fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    model = load_predictor(device)
    cos_sin = rope_cache(device)
    layers = build_layers(model, cos_sin)
    module = layers[0]
    layer = model.layers[0]
    rows = args.rows
    shape = make_shape(4, 4)
    scratch = fused.FusedPredictorLayers(shape, 64, device, torch.bfloat16)
    x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    residual_out = torch.empty_like(x)
    attention_output = torch.randn(rows, Q_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
    q = torch.empty_like(attention_output)
    k_cache = torch.randn(64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    positions = torch.full((rows,), CACHED, device=device, dtype=torch.long)
    activated = torch.empty(rows, 3072, device=device, dtype=torch.bfloat16)
    if args.op == "o_proj":
        run = lambda: fused.o_proj_add(
            attention_output=attention_output.view(rows, 1, Q_HEADS, HEAD_DIM).transpose(1, 2), tokens_per_row=1,
            weight=layer.o, residual_in=residual, residual_out=residual_out, partials=scratch.partials,
            counters=scratch.counters, shape=shape, fitted_stages=scratch.fitted_stages,
        )
    elif args.op == "qkv":
        run = lambda: fused.attention_inputs(
            x=x, tokens_per_row=1, norm=module.input_layernorm, attention=module.self_attn, q_out=q,
            positions=positions, k_cache=k_cache[:rows], v_cache=v_cache[:rows], partials=scratch.partials,
            sum_sq_partials=scratch.sum_sq_partials, counters=scratch.counters, shape=shape,
            fitted_stages=scratch.fitted_stages,
        )
    elif args.op == "gate_up":
        run = lambda: fused.mlp_up(
            residual=residual, norm=module.post_attention_layernorm, weight=layer.gate_up, activated=activated,
            shape=shape, fitted_stages=scratch.fitted_stages,
        )
    elif args.op == "cutedsl_post":
        from sglang_omni.models.qwen3_tts.cutedsl.api import PackedPostAttention

        post = PackedPostAttention(layer.o, *layer.gate_up.chunk(2, dim=0), layer.down, layer.post_norm, 1e-6, is_predictor=True)
        run = lambda: post(attention_output, residual)
    else:
        raise ValueError(args.op)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
