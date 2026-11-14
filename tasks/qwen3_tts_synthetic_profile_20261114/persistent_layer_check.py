"""Check of the persistent predictor layers against the Triton layers and an fp32 reference.

Same real weights, inputs and caches for all; the token sits at slot CACHED. Prints max and mean abs
error against fp32 for one layer and for the 5-layer stack, both kernels, then times in CUDA graphs:
one layer, the 5 layers as one launch, the 5 layers as 5 single-layer launches, and the Triton
5 layers. Run with PYTHONPATH on the tree that holds sglang_omni/models/qwen3_tts/predictor_persistent.py.

Usage: python3 persistent_layer_check.py --out <json>
"""

import argparse
import json

import torch
import torch.nn.functional as F
from predictor_bench import (
    CACHED,
    EPS,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    Q_HEADS,
    SLOTS,
    load_predictor,
    rope_cache,
    time_call,
)
from predictor_experiments import build_layers, make_shape

from sglang_omni.models.qwen3_tts.predictor_persistent import (
    PersistentPredictorLayers,
    pack_predictor_layer,
    stack_predictor_layers,
)
from sglang_omni.utils import predictor_layers as fused


def reference_layer(layer, x, k_cache, v_cache, cos_sin, rows):
    """fp32 forward of one layer, one token a row at slot CACHED over the caches' first CACHED slots."""
    hidden = x.float()
    cos = cos_sin[CACHED, : HEAD_DIM // 2]
    sin = cos_sin[CACHED, HEAD_DIM // 2 :]

    def rms(value, weight):
        return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()

    def rotate(value):
        low, high = value[..., : HEAD_DIM // 2], value[..., HEAD_DIM // 2 :]
        return torch.cat((low * cos - high * sin, low * sin + high * cos), dim=-1)

    qkv = rms(hidden, layer.input_norm) @ layer.qkv.float().t()
    q, k, v = qkv.split([Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    q = rotate(rms(q.view(rows, Q_HEADS, HEAD_DIM), layer.q_norm))
    k = rotate(rms(k.view(rows, KV_HEADS, HEAD_DIM), layer.k_norm))
    keys = torch.cat((k_cache[:rows, :CACHED].float(), k[:, None]), dim=1)
    values = torch.cat((v_cache[:rows, :CACHED].float(), v.view(rows, 1, KV_HEADS, HEAD_DIM)), dim=1)
    attended = F.scaled_dot_product_attention(
        q[:, :, None], keys.transpose(1, 2), values.transpose(1, 2), enable_gqa=True
    ).reshape(rows, Q_HEADS * HEAD_DIM)
    hidden = hidden + attended @ layer.o.float().t()
    gate, up = (rms(hidden, layer.post_norm) @ layer.gate_up.float().t()).chunk(2, dim=-1)
    return hidden + (F.silu(gate) * up) @ layer.down.float().t()


def errors(cell, name, output, reference):
    error = (output.float() - reference).abs()
    cell[f"{name} vs fp32 max abs"] = error.max().item()
    cell[f"{name} vs fp32 mean abs"] = error.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    cos_sin = rope_cache(device)
    modules = build_layers(model, cos_sin)
    packed = [
        pack_predictor_layer(
            qkv_weight=layer.qkv, output_projection_weight=layer.o, gate_up_weight=layer.gate_up,
            down_weight=layer.down, input_norm_weight=layer.input_norm, post_norm_weight=layer.post_norm,
            query_norm_weight=layer.q_norm, key_norm_weight=layer.k_norm,
        )
        for layer in model.layers
    ]
    single = [stack_predictor_layers([layer]) for layer in packed]
    stacked = stack_predictor_layers(packed)
    persistent = PersistentPredictorLayers(device, cos_sin, EPS)
    k_cache = torch.randn(LAYERS, 64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    results = {}
    for rows in (1, 8, 16):
        cell = {}
        x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        positions = torch.full((rows,), CACHED, device=device, dtype=torch.long)
        triton_layers = fused.FusedPredictorLayers(make_shape(4, 4), 64, device, torch.bfloat16)

        reference = reference_layer(model.layers[0], x, k_cache[0], v_cache[0], cos_sin, rows)
        persistent_out = torch.empty_like(x)
        persistent.forward(single[0], x, persistent_out, CACHED, k_cache[:1].clone(), v_cache[:1].clone())
        triton_keys = k_cache[:1].clone()
        triton_out = triton_layers.forward(
            layers=modules[:1], final_norm=lambda value: value, token_embeds=x.view(rows, 1, HIDDEN), batch_size=rows,
            cache_len=CACHED, positions=positions, k_cache=triton_keys, v_cache=v_cache[:1].clone(),
        ).view(rows, HIDDEN).clone()
        errors(cell, "persistent one layer", persistent_out, reference)
        errors(cell, "triton one layer", triton_out, reference)
        cell["one layer persistent vs triton max abs"] = (persistent_out.float() - triton_out.float()).abs().max().item()

        reference = x
        for index in range(LAYERS):
            reference = reference_layer(model.layers[index], reference, k_cache[index], v_cache[index], cos_sin, rows)
        persistent_keys = k_cache.clone()
        persistent.forward(stacked, x, persistent_out, CACHED, persistent_keys, v_cache.clone())
        triton_keys = k_cache.clone()
        triton_out = triton_layers.forward(
            layers=modules, final_norm=lambda value: value, token_embeds=x.view(rows, 1, HIDDEN), batch_size=rows,
            cache_len=CACHED, positions=positions, k_cache=triton_keys, v_cache=v_cache.clone(),
        ).view(rows, HIDDEN).clone()
        errors(cell, "persistent 5 layers", persistent_out, reference)
        errors(cell, "triton 5 layers", triton_out, reference)
        cell["5 layers persistent vs triton max abs"] = (persistent_out.float() - triton_out.float()).abs().max().item()
        cell["5 layers key cache slot max abs (persistent vs triton)"] = (
            persistent_keys[:, :rows, CACHED].float() - triton_keys[:, :rows, CACHED].float()
        ).abs().max().item()

        cell["persistent one layer us"] = time_call(
            lambda: persistent.forward(single[0], x, persistent_out, CACHED, k_cache[:1], v_cache[:1])
        )
        cell["persistent 5 layers one launch us"] = time_call(
            lambda: persistent.forward(stacked, x, persistent_out, CACHED, k_cache, v_cache)
        )
        buffers = [torch.empty_like(x), torch.empty_like(x)]

        def single_launches():
            source = x
            for index in range(LAYERS):
                persistent.forward(
                    single[index], source, buffers[index % 2], CACHED,
                    k_cache[index : index + 1], v_cache[index : index + 1],
                )
                source = buffers[index % 2]

        cell["persistent 5 single-layer launches us"] = time_call(single_launches)
        cell["triton 5 layers us"] = time_call(
            lambda: triton_layers.forward(
                layers=modules, final_norm=lambda value: value, token_embeds=x.view(rows, 1, HIDDEN),
                batch_size=rows, cache_len=CACHED, positions=positions, k_cache=k_cache, v_cache=v_cache,
            )
        )
        results[rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
