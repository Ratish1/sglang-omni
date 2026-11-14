"""The Qwen3-TTS code predictor alone: each operator's backends and the whole codebook step.

Real checkpoint weights, synthetic inputs, one token a row at a late codebook step (15 cached
slots). Every candidate runs inside a CUDA graph of REPS back-to-back calls; the time is per call.
The floor is the bytes the operator must read once at 3.35 TB/s.

Usage: python3 predictor_bench.py --tree <sglang-omni tree on PYTHONPATH> --out <json>
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots"
PREFIX = "talker.code_predictor."
LAYERS = 5
HIDDEN = 1024
TALKER_HIDDEN = 2048
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 3072
VOCAB = 2048
SLOTS = 17
CACHED = 15
EPS = 1e-6
ROPE_THETA = 1e6
HBM_BYTES_PER_S = 3.35e12
REPS = 20
ROUNDS = 20
BATCHES = (1, 16, 64)
TOP_K = 50
TEMPERATURE = 0.9


def load_predictor(device):
    weights = {}
    for path in glob.glob(os.path.join(MODEL_DIR, "*", "*.safetensors")):
        with safe_open(path, "pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(PREFIX):
                    weights[key[len(PREFIX):]] = handle.get_tensor(key).to(device)
                else:
                    pass
    layers = []
    for index in range(LAYERS):
        base = f"model.layers.{index}."
        attention = base + "self_attn."
        layers.append(
            SimpleNamespace(
                input_norm=weights[base + "input_layernorm.weight"],
                post_norm=weights[base + "post_attention_layernorm.weight"],
                qkv=torch.cat([weights[attention + name + "_proj.weight"] for name in ("q", "k", "v")]).contiguous(),
                q_norm=weights[attention + "q_norm.weight"],
                k_norm=weights[attention + "k_norm.weight"],
                o=weights[attention + "o_proj.weight"].contiguous(),
                gate_up=torch.cat([weights[base + "mlp.gate_proj.weight"], weights[base + "mlp.up_proj.weight"]]).contiguous(),
                down=weights[base + "mlp.down_proj.weight"].contiguous(),
            )
        )
    projection_weight = weights["small_to_mtp_projection.weight"]
    projection_bias = weights["small_to_mtp_projection.bias"]
    codec_table = weights["model.codec_embedding.3.weight"].contiguous()
    return SimpleNamespace(
        layers=layers,
        final_norm=weights["model.norm.weight"],
        lm_head=weights["lm_head.3.weight"].contiguous(),
        codec_table=codec_table,
        projected_table=(codec_table @ projection_weight.t() + projection_bias).contiguous(),
    )


def rope_cache(device):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM))
    freqs = torch.outer(torch.arange(SLOTS, dtype=torch.float32, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def time_call(fn):
    """Microseconds per call of fn, from a CUDA graph of REPS calls replayed ROUNDS times."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(REPS):
            fn()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ROUNDS):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / (REPS * ROUNDS)


def reference_substep(model, x, k_cache, v_cache, cos_sin, rows):
    """fp32 forward of one token a row at slot CACHED over the caches' first CACHED slots."""
    hidden = x.float()
    cos = cos_sin[CACHED, : HEAD_DIM // 2]
    sin = cos_sin[CACHED, HEAD_DIM // 2 :]

    def rms(value, weight):
        return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()

    def rotate(value):
        low, high = value[..., : HEAD_DIM // 2], value[..., HEAD_DIM // 2 :]
        return torch.cat((low * cos - high * sin, low * sin + high * cos), dim=-1)

    for index, layer in enumerate(model.layers):
        qkv = rms(hidden, layer.input_norm) @ layer.qkv.float().t()
        q, k, v = qkv.split([Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
        q = rotate(rms(q.view(rows, Q_HEADS, HEAD_DIM), layer.q_norm))
        k = rotate(rms(k.view(rows, KV_HEADS, HEAD_DIM), layer.k_norm))
        keys = torch.cat((k_cache[index, :rows, :CACHED].float(), k[:, None]), dim=1)
        values = torch.cat((v_cache[index, :rows, :CACHED].float(), v.view(rows, 1, KV_HEADS, HEAD_DIM)), dim=1)
        attended = F.scaled_dot_product_attention(
            q[:, :, None], keys.transpose(1, 2), values.transpose(1, 2), enable_gqa=True
        ).reshape(rows, Q_HEADS * HEAD_DIM)
        hidden = hidden + attended @ layer.o.float().t()
        gate, up = (rms(hidden, layer.post_norm) @ layer.gate_up.float().t()).chunk(2, dim=-1)
        hidden = hidden + (F.silu(gate) * up) @ layer.down.float().t()
    return rms(hidden, model.final_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    from sglang_omni.models.qwen3_tts.sampling_kernels import (
        sample_codes_and_gather_codec_embedding,
    )
    from sglang_omni.utils import predictor_layers as fused

    model = load_predictor(device)
    cos_sin = rope_cache(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    shape = fused.PredictorLayerShape(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_q_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE,
        split_qkv=fused.split_count(Q_HEADS + 2 * KV_HEADS, HIDDEN // fused.BLOCK_K, sm_count),
        split_hidden=min(
            fused.split_count(HIDDEN // fused.BLOCK_N, Q_HEADS, sm_count),
            fused.split_count(HIDDEN // fused.BLOCK_N, INTERMEDIATE // fused.BLOCK_K, sm_count),
        ),
    )
    max_rows = max(BATCHES)
    k_cache = torch.randn(LAYERS, max_rows, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    layer_modules = [
        SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=layer.input_norm, variance_epsilon=EPS),
            post_attention_layernorm=SimpleNamespace(weight=layer.post_norm, variance_epsilon=EPS),
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=layer.qkv),
                q_norm=SimpleNamespace(weight=layer.q_norm, variance_epsilon=EPS),
                k_norm=SimpleNamespace(weight=layer.k_norm, variance_epsilon=EPS),
                rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
                o_proj=SimpleNamespace(weight=layer.o),
            ),
            mlp=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=layer.gate_up), down_proj=SimpleNamespace(weight=layer.down)),
        )
        for layer in model.layers
    ]
    fused_layers = fused.FusedPredictorLayers(shape, max_rows, device, torch.bfloat16)
    try:
        from sglang_omni.models.qwen3_tts.cutedsl.api import PackedPostAttention, SlotPreAttention

        cute_layers = [
            SimpleNamespace(
                pre=SlotPreAttention(layer.input_norm, layer.qkv, layer.q_norm, layer.k_norm, EPS, is_predictor=True),
                post=PackedPostAttention(
                    layer.o, *layer.gate_up.chunk(2, dim=0), layer.down, layer.post_norm, EPS, is_predictor=True
                ),
            )
            for layer in model.layers
        ]
    except Exception as exc:
        print("cutedsl unavailable:", repr(exc))
        cute_layers = None
    results = {"split_qkv": shape.split_qkv, "split_hidden": shape.split_hidden, "batches": {}}
    layer = model.layers[0]
    module = layer_modules[0]
    floors = {
        "qkv": layer.qkv.numel() * 2,
        "o_proj": layer.o.numel() * 2,
        "gate_up": layer.gate_up.numel() * 2,
        "down": layer.down.numel() * 2,
        "lm_head": model.lm_head.numel() * 2,
    }
    for rows in BATCHES:
        cell = {}
        x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        positions = torch.full((rows,), CACHED, device=device, dtype=torch.long)
        cache_slots = torch.arange(rows, device=device) * SLOTS + CACHED
        q = torch.empty(rows, Q_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
        attention_output = torch.empty_like(q)
        qkv_out = torch.empty(rows, layer.qkv.shape[0], device=device, dtype=torch.bfloat16)
        gate_up_out = torch.empty(rows, layer.gate_up.shape[0], device=device, dtype=torch.bfloat16)
        activated = torch.empty(rows, INTERMEDIATE, device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(x)
        normed = torch.empty_like(x)
        logits = torch.empty(rows, VOCAB, device=device, dtype=torch.bfloat16)
        layer_k = k_cache[0, :rows]
        layer_v = v_cache[0, :rows]
        cell["qkv triton fused (norm, qkv, qk norm, rope, store)"] = time_call(
            lambda: fused.attention_inputs(
                x=x, tokens_per_row=1, norm=module.input_layernorm, attention=module.self_attn, q_out=q,
                positions=positions, k_cache=layer_k, v_cache=layer_v, partials=fused_layers.partials,
                sum_sq_partials=fused_layers.sum_sq_partials, counters=fused_layers.counters, shape=shape,
                fitted_stages=fused_layers.fitted_stages,
            )
        )
        cell["qkv cublas gemm only"] = time_call(lambda: torch.mm(x, layer.qkv.t(), out=qkv_out))
        cell["attention triton"] = time_call(
            lambda: fused.decode_attention(
                q=q, tokens_per_row=1, k_cache=layer_k, v_cache=layer_v, end=CACHED + 1,
                attention_output=attention_output,
            )
        )
        cache_seqlens = torch.full((rows,), CACHED + 1, device=device, dtype=torch.int32)
        cell["attention fa3"] = time_call(
            lambda: flash_attn_with_kvcache(
                q.view(rows, 1, Q_HEADS, HEAD_DIM), layer_k, layer_v, cache_seqlens=cache_seqlens, causal=True
            )
        )
        cell["attention sdpa"] = time_call(
            lambda: F.scaled_dot_product_attention(
                q.view(rows, Q_HEADS, 1, HEAD_DIM), layer_k[:, : CACHED + 1].transpose(1, 2),
                layer_v[:, : CACHED + 1].transpose(1, 2), enable_gqa=True,
            )
        )
        cell["o_proj triton gemv_add"] = time_call(
            lambda: fused.o_proj_add(
                attention_output=attention_output.view(rows, 1, Q_HEADS, HEAD_DIM).transpose(1, 2), tokens_per_row=1,
                weight=layer.o, residual_in=residual, residual_out=residual_out, partials=fused_layers.partials,
                counters=fused_layers.counters, shape=shape, fitted_stages=fused_layers.fitted_stages,
            )
        )
        cell["o_proj cublas addmm"] = time_call(lambda: torch.addmm(residual, attention_output, layer.o.t(), out=residual_out))
        cell["gate_up triton fused (norm, gemm, silu)"] = time_call(
            lambda: fused.mlp_up(
                residual=residual, norm=module.post_attention_layernorm, weight=layer.gate_up, activated=activated,
                shape=shape, fitted_stages=fused_layers.fitted_stages,
            )
        )
        cell["gate_up cublas gemm only"] = time_call(lambda: torch.mm(normed, layer.gate_up.t(), out=gate_up_out))
        cell["gate_up rmsnorm + cublas + silu_and_mul"] = time_call(
            lambda: (rmsnorm(residual, layer.post_norm, normed, EPS), torch.mm(normed, layer.gate_up.t(), out=gate_up_out), silu_and_mul(gate_up_out, activated))
        )
        cell["down triton gemv_add"] = time_call(
            lambda: fused.down_add(
                activated=activated, weight=layer.down, residual=residual_out, partials=fused_layers.partials,
                counters=fused_layers.counters, shape=shape, fitted_stages=fused_layers.fitted_stages,
            )
        )
        cell["down cublas addmm"] = time_call(lambda: torch.addmm(residual, activated, layer.down.t(), out=residual_out))
        cell["lm_head cublas"] = time_call(lambda: torch.mm(normed, model.lm_head.t(), out=logits))
        temperatures = torch.full((rows,), TEMPERATURE, device=device)
        top_ks = torch.full((rows,), TOP_K, device=device, dtype=torch.long)
        top_ps = torch.ones(rows, device=device)
        seeds = torch.arange(rows, device=device, dtype=torch.long) + 7
        codes = torch.zeros(rows, device=device, dtype=torch.long)
        gathered = torch.empty(rows, TALKER_HIDDEN, device=device, dtype=torch.bfloat16)
        accumulated = torch.zeros(rows, TALKER_HIDDEN, device=device, dtype=torch.bfloat16)
        projected = torch.empty(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        logits.copy_(torch.randn_like(logits) * 4)
        cell["sampler seeded top-k + gather (current)"] = time_call(
            lambda: sample_codes_and_gather_codec_embedding(
                logits, temperatures, top_ks, top_ps, seeds, positions, max_top_k=TOP_K, has_top_p=False, codes=codes,
                embedding_weight=model.codec_table, gathered=gathered, accumulated=accumulated,
                projected_weight=model.projected_table, projected=projected,
            )
        )
        if cute_layers is not None and rows <= 32:
            k_rows = k_cache[0].view(-1, KV_HEADS, HEAD_DIM)
            v_rows = v_cache[0].view(-1, KV_HEADS, HEAD_DIM)
            cell["qkv cutedsl pre_attention"] = time_call(
                lambda: cute_layers[0].pre(x, positions, cos_sin, cache_slots, k_rows, v_rows)
            )
            cell["o_proj + mlp cutedsl post_attention"] = time_call(lambda: cute_layers[0].post(attention_output, residual))
        else:
            pass
        token = torch.randn(rows, 1, HIDDEN, device=device, dtype=torch.bfloat16)
        final_norm = lambda value: (rmsnorm(value, model.final_norm, normed, EPS), normed)[1]

        def triton_substep():
            hidden = fused_layers.forward(
                layers=layer_modules, final_norm=final_norm, token_embeds=token, batch_size=rows, cache_len=CACHED,
                positions=positions, k_cache=k_cache, v_cache=v_cache,
            )
            torch.mm(hidden.view(rows, HIDDEN), model.lm_head.t(), out=logits)
            sample_codes_and_gather_codec_embedding(
                logits, temperatures, top_ks, top_ps, seeds, positions, max_top_k=TOP_K, has_top_p=False, codes=codes,
                embedding_weight=model.codec_table, gathered=gathered, accumulated=accumulated,
                projected_weight=model.projected_table, projected=projected,
            )

        cell["substep triton (5 layers, norm, lm_head, sampler)"] = time_call(triton_substep)
        reference = reference_substep(model, token.view(rows, HIDDEN), k_cache, v_cache, cos_sin, rows)
        triton_out = fused_layers.forward(
            layers=layer_modules, final_norm=final_norm, token_embeds=token, batch_size=rows, cache_len=CACHED,
            positions=positions, k_cache=k_cache.clone(), v_cache=v_cache.clone(),
        ).view(rows, HIDDEN).float()
        cell["triton vs fp32 max abs"] = (triton_out - reference).abs().max().item()
        if cute_layers is not None and rows <= 32:
            k_work = k_cache.clone()
            v_work = v_cache.clone()

            def cute_substep(k_store=k_cache, v_store=v_cache):
                hidden_states = token.view(rows, HIDDEN)
                for index, cute in enumerate(cute_layers):
                    q_cute = cute.pre(
                        hidden_states, positions, cos_sin, cache_slots, k_store[index].view(-1, KV_HEADS, HEAD_DIM),
                        v_store[index].view(-1, KV_HEADS, HEAD_DIM),
                    )
                    attended = torch.empty_like(q_cute)
                    fused.decode_attention(
                        q=q_cute, tokens_per_row=1, k_cache=k_store[index, :rows], v_cache=v_store[index, :rows],
                        end=CACHED + 1, attention_output=attended,
                    )
                    hidden_states = cute.post(attended, hidden_states)
                return final_norm(hidden_states)

            def cute_full():
                hidden = cute_substep()
                torch.mm(hidden, model.lm_head.t(), out=logits)
                sample_codes_and_gather_codec_embedding(
                    logits, temperatures, top_ks, top_ps, seeds, positions, max_top_k=TOP_K, has_top_p=False,
                    codes=codes, embedding_weight=model.codec_table, gathered=gathered, accumulated=accumulated,
                    projected_weight=model.projected_table, projected=projected,
                )

            cell["substep cutedsl (5 layers, norm, lm_head, sampler)"] = time_call(cute_full)
            cell["cutedsl vs fp32 max abs"] = (cute_substep(k_work, v_work).float() - reference).abs().max().item()
        else:
            pass
        cell["floors us"] = {name: size / HBM_BYTES_PER_S * 1e6 for name, size in floors.items()}
        cell["substep floor us"] = (sum(floors.values()) - floors["lm_head"]) * LAYERS / HBM_BYTES_PER_S * 1e6 + floors["lm_head"] / HBM_BYTES_PER_S * 1e6
        results["batches"][rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
