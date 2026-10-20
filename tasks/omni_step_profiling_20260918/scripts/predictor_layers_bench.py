"""The code predictor's fused layers at the served shapes, timed per kernel from a CUDA graph with programmatic dependent
launch, on the implementation of the tree on PYTHONPATH (main's or a branch's).

usage: PYTHONPATH=<tree> python3 predictor_layers_bench.py [--batch 1 16 64] [--replays 50]

Qwen3-TTS-12Hz-1.7B-Base's predictor dimensions (hidden 1024, 16 q and 8 kv heads of 128, intermediate 3072, 5 layers,
17 cache slots) with random weights: the dense layers' kernels depend on shapes, not values. One step is the served
one, captured in one graph: the opening pair pass at slots 0 and 1 (fused while it is within MAX_FUSED_ROWS rows, as
served; otherwise left out here), then 14 single-token passes at slots 2 to 15, each over the 5 layers. The layers'
weights (about 150 MB) exceed L2, so every pass streams them from HBM as served. Per batch: the replay time (median of
the replays), and per kernel family its launches per step and mean duration from the profiler's CUDA activity.
Served references (formal traces, 10-20): at bs 1 qkv 9.3 us, gate-up 8.0, gemv_add 5.85; at bs 64 15.1, 11.7, 7.7.
"""

import argparse
import statistics
from collections import defaultdict
from types import SimpleNamespace

import torch

from sglang_omni.utils.predictor_layers import (
    BLOCK_K,
    BLOCK_N,
    MAX_FUSED_ROWS,
    FusedPredictorLayers,
    PredictorLayerShape,
    split_count,
)

HIDDEN = 1024
HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 8
INTERMEDIATE = 3072
LAYERS = 5
SLOTS = 17
FAMILIES = ("norm_qkv_rope_store", "norm_gate_up_silu", "gemv_add", "sdpa", "flash")


def make_layers(device: torch.device) -> list[SimpleNamespace]:
    def weight(rows: int, columns: int) -> torch.Tensor:
        return (torch.randn(rows, columns, device=device) * 0.02).to(torch.bfloat16)

    def norm(size: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(size, device=device)).to(torch.bfloat16), variance_epsilon=1e-6
        )

    inverse_frequencies = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    angles = torch.arange(64, device=device).float()[:, None] * inverse_frequencies[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return [
        SimpleNamespace(
            input_layernorm=norm(HIDDEN),
            post_attention_layernorm=norm(HIDDEN),
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=weight((Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, HIDDEN)),
                o_proj=SimpleNamespace(weight=weight(HIDDEN, Q_HEADS * HEAD_DIM)),
                q_norm=norm(HEAD_DIM),
                k_norm=norm(HEAD_DIM),
                rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
            ),
            mlp=SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=weight(2 * INTERMEDIATE, HIDDEN)),
                down_proj=SimpleNamespace(weight=weight(HIDDEN, INTERMEDIATE)),
            ),
        )
        for _ in range(LAYERS)
    ]


def make_shape(device: torch.device) -> PredictorLayerShape:
    """resolve_predictor_layer_shape's splits for these dimensions on this card."""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return PredictorLayerShape(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_q_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE,
        split_qkv=split_count(Q_HEADS + 2 * KV_HEADS, HIDDEN // BLOCK_K, sm_count),
        split_hidden=min(
            split_count(HIDDEN // BLOCK_N, Q_HEADS, sm_count),
            split_count(HIDDEN // BLOCK_N, INTERMEDIATE // BLOCK_K, sm_count),
        ),
    )


def family_of(name: str) -> str:
    for family in FAMILIES:
        if family in name:
            return family
        else:
            pass
    return "other"


def bench(batch_size: int, replays: int, device: torch.device) -> None:
    layers = make_layers(device)
    shape = make_shape(device)
    fused = FusedPredictorLayers(shape, min(2 * batch_size, MAX_FUSED_ROWS), device, torch.bfloat16)
    k_cache = torch.zeros(LAYERS, batch_size, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    pair_embeds = torch.randn(batch_size, 2, HIDDEN, device=device).to(torch.bfloat16)
    single_embeds = torch.randn(batch_size, 1, HIDDEN, device=device).to(torch.bfloat16)
    pair_positions = torch.arange(2, device=device).repeat(batch_size)
    slot_positions = torch.arange(SLOTS, device=device)[:, None].expand(SLOTS, batch_size).contiguous()
    has_pair = fused.covers(2 * batch_size)

    def step() -> None:
        if has_pair:
            fused.forward(
                layers=layers, final_norm=lambda hidden: hidden, token_embeds=pair_embeds, batch_size=batch_size,
                cache_len=0, positions=pair_positions, k_cache=k_cache, v_cache=v_cache,
            )
        else:
            pass
        for cache_len in range(2, SLOTS - 1):
            fused.forward(
                layers=layers, final_norm=lambda hidden: hidden, token_embeds=single_embeds, batch_size=batch_size,
                cache_len=cache_len, positions=slot_positions[cache_len], k_cache=k_cache, v_cache=v_cache,
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    profiled = 20
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(profiled):
            graph.replay()
        torch.cuda.synchronize()
    durations: dict[str, list[float]] = defaultdict(list)
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            durations[family_of(event.name)].append(event.time_range.elapsed_us())
        else:
            pass
    print(f"bs {batch_size}: pair pass {'fused' if has_pair else 'left out'}, step replay median {statistics.median(times):.3f} ms")
    for family in (*FAMILIES, "other"):
        values = durations.get(family, [])
        if values:
            print(
                f"  {family:<22} {len(values) / profiled:6.1f} per step  mean {statistics.mean(values):7.2f} us  "
                f"total {sum(values) / profiled / 1e3:6.3f} ms per step"
            )
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    with torch.inference_mode():
        for batch_size in args.batch:
            bench(batch_size, args.replays, device)


if __name__ == "__main__":
    main()
