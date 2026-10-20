"""The Qwen3-TTS vocoder's decoder transformer at the served shapes: today's incremental path against the production
FusedIncrementalTransformer (V3, perf/qwen3-tts-fused-decoder-transformer), timed from CUDA graphs, with kernel counts
and distance to fp32.

The real decoder of Qwen/Qwen3-TTS-12Hz-1.7B-Base (8 layers, hidden 512, 16 heads of 64, SwiGLU 1024, sliding window
72). Per (fresh frames, rows): a warm state (every row 200 frames in, 71 retained K/V frames) and a random latent input.

  current  incremental_codec.incremental_transformer, as the decode runs it on main
  fused    incremental_codec.FusedIncrementalTransformer on the same weights (run with the V3 tree on PYTHONPATH)

Each arm is captured in a CUDA graph and replayed (median of 200 replays, us); kernels are counted with the profiler on
one eager call; outputs are compared with the current path run in fp32 (weights and state cast to fp32). --compiled
times both arms under torch.compile(dynamic=False, fullgraph=True), as the decoder's precompile builds them.

usage: PYTHONPATH=<V3 tree> python vocoder_transformer_bench.py [--widths 1 2 4 8] [--rows 1 4 8 16] [--compiled]
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang_omni.models.qwen3_tts import incremental_codec
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.platforms import current_platform

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FRAMES_IN = 200


def warm_state(transformer, rows, dtype, device, generator):
    attention = transformer.layers[0].self_attn
    head_dim = int(attention.head_dim)
    kv_heads = attention.k_proj.out_features // head_dim
    retained = int(attention.sliding_window) - 1
    keys = [
        torch.randn(rows, kv_heads, retained, head_dim, device=device, generator=generator).to(dtype)
        for _ in transformer.layers
    ]
    values = [
        torch.randn(rows, kv_heads, retained, head_dim, device=device, generator=generator).to(dtype)
        for _ in transformer.layers
    ]
    positions = torch.full((rows,), FRAMES_IN, device=device, dtype=torch.long)
    return keys, values, positions


def decode_state(keys, values, positions):
    state = incremental_codec.Qwen3TTSIncrementalCodecState(frame_positions=positions)
    state.frame_position = FRAMES_IN
    state.transformer_context_length = keys[0].shape[-2]
    state.transformer_keys = dict(enumerate(keys))
    state.transformer_values = dict(enumerate(values))
    return state


def current_call(transformer, x, keys, values, positions):
    return incremental_codec.incremental_transformer(transformer, x, decode_state(keys, values, positions))


def fused_call(fused, x, keys, values, positions):
    return fused(x, decode_state(keys, values, positions))


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
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def kernel_count(fn) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    transformer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    ).model.decoder.pre_transformer.eval()
    fused = incremental_codec.FusedIncrementalTransformer(
        transformer, current_platform.get_joint_rope_inplace_kernel()
    )
    reference = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="float32", attn_implementation=None
    ).model.decoder.pre_transformer.eval()
    config = transformer.config
    print(
        f"layers {len(transformer.layers)}, hidden {config.hidden_size}, latent "
        f"{transformer.input_proj.in_features}, window {transformer.window_size}, compiled {args.compiled}"
    )
    if args.compiled:
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 256)
        run_current = torch.compile(
            lambda x, keys, values, positions: current_call(transformer, x, keys, values, positions),
            dynamic=False,
            fullgraph=True,
        )
        run_fused = torch.compile(
            lambda x, keys, values, positions: fused_call(fused, x, keys, values, positions),
            dynamic=False,
            fullgraph=True,
        )
    else:
        run_current = lambda x, keys, values, positions: current_call(transformer, x, keys, values, positions)
        run_fused = lambda x, keys, values, positions: fused_call(fused, x, keys, values, positions)
    print(
        f"{'width':>5} {'rows':>4} {'current us':>11} {'fused us':>9} {'speedup':>8} "
        f"{'kernels':>12} {'err current':>12} {'err fused':>10}"
    )
    with torch.inference_mode():
        for width in args.widths:
            for rows in args.rows:
                generator = torch.Generator(device=device).manual_seed(width * 100 + rows)
                keys, values, positions = warm_state(transformer, rows, torch.bfloat16, device, generator)
                x = torch.randn(
                    rows, width, transformer.input_proj.in_features, device=device, generator=generator
                ).to(torch.bfloat16)
                current_us = graph_time(lambda: run_current(x, keys, values, positions))
                fused_us = graph_time(lambda: run_fused(x, keys, values, positions))
                counts = (
                    kernel_count(lambda: run_current(x, keys, values, positions)),
                    kernel_count(lambda: run_fused(x, keys, values, positions)),
                )
                truth = current_call(
                    reference, x.float(), [k.float() for k in keys], [v.float() for v in values], positions
                )
                errors = [
                    float((out.float() - truth).norm() / truth.norm())
                    for out in (run_current(x, keys, values, positions), run_fused(x, keys, values, positions))
                ]
                print(
                    f"{width:>5} {rows:>4} {current_us:>11.1f} {fused_us:>9.1f} "
                    f"{current_us / fused_us:>7.2f}x {counts[0]:>5} -> {counts[1]:<5} "
                    f"{errors[0]:>12.5f} {errors[1]:>10.5f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
