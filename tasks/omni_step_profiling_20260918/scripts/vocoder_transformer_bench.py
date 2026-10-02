"""The Qwen3-TTS vocoder's decoder transformer at the served shapes: today's incremental
path against a fused composition, timed from CUDA graphs, with kernel counts and distance
to fp32.

The real decoder of Qwen/Qwen3-TTS-12Hz-1.7B-Base (8 layers, hidden 512, 16 heads of 64,
SwiGLU 1024, sliding window 72). Per (fresh frames, rows): a warm state (every row 200
frames in, 71 retained K/V frames) and a random latent input. Arms:

  current  incremental_codec.incremental_transformer, as the decode runs it
  fused    the same layers composed of one qkv and one gate-up GEMM on merged weights,
           SGLang's rmsnorm and silu_and_mul, the platform's rope kernel on a cached fp32
           cos/sin table, the mask built once per call, SDPA with that mask, LayerScale
           plus residual as one addcmul, and the retained K/V as views of the window

Each arm is captured in a CUDA graph and replayed (median of 200 replays, us); kernels are
counted with the profiler on one eager call; outputs are compared with the current path run
in fp32 (weights and state cast to fp32).

usage: python vocoder_transformer_bench.py [--widths 1 2 4 8] [--rows 1 4 8 16]
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.layernorm import rmsnorm

from sglang_omni.models.qwen3_tts import incremental_codec
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.platforms import current_platform

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FRAMES_IN = 200


class FusedTransformer:
    """The decoder transformer on merged weights and fused ops (a bench arm)."""

    def __init__(self, transformer) -> None:
        self.t = transformer
        self.rope = current_platform.get_joint_rope_inplace_kernel()
        rotary = transformer.rotary_emb
        positions = torch.arange(
            transformer.config.max_position_embeddings,
            device=rotary.inv_freq.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, rotary.inv_freq.float())
        self.cos_sin = (
            torch.cat((angles.cos(), angles.sin()), -1) * rotary.attention_scaling
        )
        self.qkv = [
            torch.cat(
                (
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                )
            )
            for layer in transformer.layers
        ]
        self.gate_up = [
            torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight))
            for layer in transformer.layers
        ]

    def __call__(self, hidden_states, keys, values, frame_positions):
        t = self.t
        attention = t.layers[0].self_attn
        head_dim = int(attention.head_dim)
        heads = attention.q_proj.out_features // head_dim
        kv_heads = attention.k_proj.out_features // head_dim
        window = int(attention.sliding_window)
        retained = window - 1
        x = t.input_proj(hidden_states)
        batch, fresh, hidden = x.shape
        prior = keys[0].shape[-2]
        key_positions = frame_positions[:, None] + torch.arange(
            -prior, fresh, device=x.device
        )
        query_positions = frame_positions[:, None] + torch.arange(
            fresh, device=x.device
        )
        allowed = key_positions[:, None, :] <= query_positions[:, :, None]
        allowed &= key_positions[:, None, :] > query_positions[:, :, None] - window
        allowed &= (key_positions >= 0)[:, None, :]
        mask = allowed[:, None]
        flat_positions = query_positions.reshape(-1)
        x = x.reshape(batch * fresh, hidden)
        next_keys, next_values = [], []
        for index, layer in enumerate(t.layers):
            normed = rmsnorm(
                x, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon
            )
            q, k, v = F.linear(normed, self.qkv[index]).split(
                [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], -1
            )
            q = q.view(batch * fresh, heads, head_dim)
            k = k.view(batch * fresh, kv_heads, head_dim)
            self.rope(q, k, self.cos_sin, flat_positions, is_neox=True)
            k_all = torch.cat(
                (keys[index], k.view(batch, fresh, kv_heads, head_dim).transpose(1, 2)),
                2,
            )
            v_all = torch.cat(
                (
                    values[index],
                    v.reshape(batch, fresh, kv_heads, head_dim).transpose(1, 2),
                ),
                2,
            )
            attended = F.scaled_dot_product_attention(
                q.view(batch, fresh, heads, head_dim).transpose(1, 2),
                k_all,
                v_all,
                attn_mask=mask,
                scale=float(attention.scaling),
                enable_gqa=heads != kv_heads,
            )
            attended = attended.transpose(1, 2).reshape(batch * fresh, heads * head_dim)
            x = torch.addcmul(
                x,
                layer.self_attn_layer_scale.scale,
                F.linear(attended, layer.self_attn.o_proj.weight),
            )
            normed = rmsnorm(
                x,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            )
            activated = silu_and_mul(F.linear(normed, self.gate_up[index]))
            x = torch.addcmul(
                x,
                layer.mlp_layer_scale.scale,
                F.linear(activated, layer.mlp.down_proj.weight),
            )
            next_keys.append(k_all[:, :, -retained:])
            next_values.append(v_all[:, :, -retained:])
        x = rmsnorm(x, t.norm.weight, t.norm.variance_epsilon)
        return t.output_proj(x.view(batch, fresh, hidden)), next_keys, next_values


def warm_state(transformer, rows, dtype, device, generator):
    attention = transformer.layers[0].self_attn
    head_dim = int(attention.head_dim)
    kv_heads = attention.k_proj.out_features // head_dim
    retained = int(attention.sliding_window) - 1
    keys = [
        torch.randn(
            rows, kv_heads, retained, head_dim, device=device, generator=generator
        ).to(dtype)
        for _ in transformer.layers
    ]
    values = [
        torch.randn(
            rows, kv_heads, retained, head_dim, device=device, generator=generator
        ).to(dtype)
        for _ in transformer.layers
    ]
    positions = torch.full((rows,), FRAMES_IN, device=device, dtype=torch.long)
    return keys, values, positions


def current_call(transformer, x, keys, values, positions):
    state = incremental_codec.Qwen3TTSIncrementalCodecState(frame_positions=positions)
    state.frame_position = FRAMES_IN
    state.transformer_context_length = keys[0].shape[-2]
    state.transformer_keys = dict(enumerate(keys))
    state.transformer_values = dict(enumerate(values))
    return incremental_codec.incremental_transformer(transformer, x, state)


def graph_time(fn) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(200):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def kernel_count(fn) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )


def compiled_arm(args) -> None:
    device = torch.device("cuda")
    transformer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    ).model.decoder.pre_transformer.eval()
    fused = FusedTransformer(transformer)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 256
    )
    compiled = torch.compile(
        lambda x, keys, values, positions: current_call(
            transformer, x, keys, values, positions
        ),
        dynamic=False,
        fullgraph=True,
    )
    print(f"{'width':>5} {'rows':>4} {'compiled us':>12} {'fused us':>9}")
    with torch.inference_mode():
        for width in args.widths:
            for rows in args.rows:
                generator = torch.Generator(device=device).manual_seed(
                    width * 100 + rows
                )
                keys, values, positions = warm_state(
                    transformer, rows, torch.bfloat16, device, generator
                )
                x = torch.randn(
                    rows,
                    width,
                    transformer.input_proj.in_features,
                    device=device,
                    generator=generator,
                ).to(torch.bfloat16)
                compiled_us = graph_time(lambda: compiled(x, keys, values, positions))
                fused_us = graph_time(lambda: fused(x, keys, values, positions))
                print(
                    f"{width:>5} {rows:>4} {compiled_us:>12.1f} {fused_us:>9.1f}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="time today's path under torch.compile(dynamic=False, fullgraph=True), as "
        "the decoder's precompile builds it, instead of the eager and fused arms",
    )
    args = parser.parse_args()
    if args.compiled:
        compiled_arm(args)
        return
    else:
        pass
    device = torch.device("cuda")
    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    transformer = tokenizer.model.decoder.pre_transformer.eval()
    config = transformer.config
    print(
        f"layers {len(transformer.layers)}, hidden {config.hidden_size}, latent "
        f"{transformer.input_proj.in_features}, window {transformer.window_size}"
    )
    fused = FusedTransformer(transformer)
    reference = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="float32", attn_implementation=None
    ).model.decoder.pre_transformer.eval()
    print(
        f"{'width':>5} {'rows':>4} {'current us':>11} {'fused us':>9} {'speedup':>8} "
        f"{'kernels':>12} {'err current':>12} {'err fused':>10}"
    )
    with torch.inference_mode():
        for width in args.widths:
            for rows in args.rows:
                generator = torch.Generator(device=device).manual_seed(
                    width * 100 + rows
                )
                keys, values, positions = warm_state(
                    transformer, rows, torch.bfloat16, device, generator
                )
                x = torch.randn(
                    rows,
                    width,
                    transformer.input_proj.in_features,
                    device=device,
                    generator=generator,
                ).to(torch.bfloat16)
                current_us = graph_time(
                    lambda: current_call(transformer, x, keys, values, positions)
                )
                fused_us = graph_time(lambda: fused(x, keys, values, positions))
                counts = (
                    kernel_count(
                        lambda: current_call(transformer, x, keys, values, positions)
                    ),
                    kernel_count(lambda: fused(x, keys, values, positions)),
                )
                truth = current_call(
                    reference,
                    x.float(),
                    [k.float() for k in keys],
                    [v.float() for v in values],
                    positions,
                )
                current_out = current_call(transformer, x, keys, values, positions)
                fused_out, _, _ = fused(x, keys, values, positions)
                errors = [
                    float((out.float() - truth).norm() / truth.norm())
                    for out in (current_out, fused_out)
                ]
                print(
                    f"{width:>5} {rows:>4} {current_us:>11.1f} {fused_us:>9.1f} "
                    f"{current_us / fused_us:>7.2f}x {counts[0]:>5} -> {counts[1]:<5} "
                    f"{errors[0]:>12.5f} {errors[1]:>10.5f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
