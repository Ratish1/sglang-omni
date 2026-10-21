"""Size the K/V rings for the Qwen3-TTS vocoder (state_in_place_20261021/PLAN.md 3, step 1).

usage: python3 ring_step_bench.py --card 2 --variant gathered|ring --rows 8 --width 8 [--compiled] [--first]

One decoder transformer step of the real 1.7B tokenizer's decoder through V3's FusedIncrementalTransformer, as a
served decode runs it, from a CUDA graph replay, one variant and shape per process:

- gathered: today's path. The arena's per-layer K/V buffers (slots, kv heads, 71, head dim) are gathered for the
  cohort, the fused step concatenates and attends, keeps the last 71 frames, and the K/V are scattered back.
- ring: the rope kernel rotates q and k and stores k and v in per-slot rings of 79 frames; a later decode attends
  its window through the ring's page table with FA3 (eager) or gathers it for SDPA (compiled, since FA3 does not
  trace); a first decode (--first) attends its own frames.

Before timing, both variants run one step from the same state and the script prints how far apart their outputs
and their kept K/V are. Then 30 warmup and 300 timed replays; prints the median and p90 per replay in us.
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.rope import apply_rope_inplace_with_kvcache

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

CHECKPOINT = "/data/hf/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
RING = 79
START = 200


def ring_transformer(fused, hidden_states, key_rings, value_rings, slot_index, start_positions, first_decode):
    """V3's fused step with the K/V in per-slot rings, as PLAN.md 1 lays it out."""
    transformer = fused.transformer
    retained = int(transformer.window_size) - 1
    hidden_states = transformer.input_proj(hidden_states)
    batch_size, width, hidden_size = hidden_states.shape
    rows = batch_size * width
    device = hidden_states.device
    frame_offsets = torch.arange(width, device=device)
    positions = start_positions[:, None] + frame_offsets
    write_rows = (slot_index[:, None] * RING + positions % RING).reshape(-1)
    angles = positions.reshape(rows, 1).float() * fused.inverse_frequencies
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1) * fused.attention_scaling
    rotary_rows = torch.arange(rows, device=device)
    compiling = torch.compiler.is_dynamo_compiling()
    if not first_decode:
        window_starts = (start_positions - retained).clamp(min=0)
        window_rows = slot_index[:, None] * RING + (window_starts[:, None] + torch.arange(RING, device=device)) % RING
        window_lengths = start_positions - window_starts + width
        query_offsets = (window_lengths - width)[:, None] + frame_offsets
        distances = query_offsets[:, :, None] - torch.arange(RING, device=device)
        window_mask = ((distances >= 0) & (distances <= retained))[:, None]
    else:
        pass
    hidden_states = hidden_states.reshape(rows, hidden_size)
    for layer_index, layer in enumerate(transformer.layers):
        attention = layer.self_attn
        head_dim = int(attention.head_dim)
        residual = hidden_states
        normalized = fused.normalize(hidden_states, layer.input_layernorm)
        query, key, value = F.linear(normalized, attention.qkv_weight).split(
            [attention.q_proj.out_features, attention.k_proj.out_features, attention.v_proj.out_features], dim=-1
        )
        query = query.view(rows, -1, head_dim)
        key = key.view(rows, -1, head_dim)
        apply_rope_inplace_with_kvcache(
            query, key, value.view(rows, -1, head_dim), key_rings[layer_index], value_rings[layer_index],
            cos_sin, rotary_rows, write_rows, is_neox=True,
        )
        if first_decode:
            attended = F.scaled_dot_product_attention(
                query.view(batch_size, width, -1, head_dim).transpose(1, 2),
                key.view(batch_size, width, -1, head_dim).transpose(1, 2),
                value.view(batch_size, width, -1, head_dim).transpose(1, 2),
                is_causal=True, scale=float(attention.scaling),
            ).transpose(1, 2).reshape(rows, -1)
        elif compiling:
            keys = key_rings[layer_index][window_rows].view(batch_size, RING, -1, head_dim).transpose(1, 2)
            values = value_rings[layer_index][window_rows].view(batch_size, RING, -1, head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                query.view(batch_size, width, -1, head_dim).transpose(1, 2), keys, values,
                attn_mask=window_mask, scale=float(attention.scaling),
            ).transpose(1, 2).reshape(rows, -1)
        else:
            attended = flash_attn_with_kvcache(
                query.view(batch_size, width, -1, head_dim),
                key_rings[layer_index].view(-1, 1, key.shape[1], head_dim),
                value_rings[layer_index].view(-1, 1, key.shape[1], head_dim),
                page_table=window_rows.to(torch.int32),
                cache_seqlens=window_lengths.to(torch.int32),
                softmax_scale=float(attention.scaling),
                causal=True,
                window_size=(retained, 0),
            ).reshape(rows, -1)
        hidden_states = torch.addcmul(
            residual, layer.self_attn_layer_scale.scale, F.linear(attended, attention.o_proj.weight)
        )
        residual = hidden_states
        normalized = fused.normalize(hidden_states, layer.post_attention_layernorm)
        gate_up = F.linear(normalized, layer.mlp.gate_up_weight)
        if compiling:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
        else:
            activated = silu_and_mul(gate_up)
        hidden_states = torch.addcmul(
            residual, layer.mlp_layer_scale.scale, F.linear(activated, layer.mlp.down_proj.weight)
        )
    hidden_states = fused.normalize(hidden_states, transformer.norm)
    return transformer.output_proj(hidden_states.view(batch_size, width, hidden_size))


class Setup:
    def __init__(self, fused, device, rows, width, first_decode):
        transformer = fused.transformer
        attention = transformer.layers[0].self_attn
        self.fused = fused
        self.layers = len(transformer.layers)
        self.retained = int(transformer.window_size) - 1
        self.kv_heads = int(attention.k_proj.out_features) // int(attention.head_dim)
        self.head_dim = int(attention.head_dim)
        self.width = width
        self.first_decode = first_decode
        slots = 2 * rows + 2
        generator = torch.Generator(device=device).manual_seed(5)
        self.index = torch.arange(rows, device=device) * 2 + 1
        start = 0 if first_decode else START
        self.positions = torch.full((slots,), start, dtype=torch.long, device=device)
        self.hidden = torch.randn(
            rows, width, transformer.input_proj.in_features, device=device, generator=generator
        ).bfloat16()
        shape = (slots, self.kv_heads, self.retained, self.head_dim)
        self.key_store = [torch.randn(shape, device=device, generator=generator).bfloat16() for _ in range(self.layers)]
        self.value_store = [torch.randn(shape, device=device, generator=generator).bfloat16() for _ in range(self.layers)]
        if first_decode:
            for store in (*self.key_store, *self.value_store):
                store.zero_()
        else:
            pass
        width_kv = self.kv_heads * self.head_dim
        self.key_rings = [torch.zeros(slots * RING, width_kv, device=device).bfloat16() for _ in range(self.layers)]
        self.value_rings = [torch.zeros(slots * RING, width_kv, device=device).bfloat16() for _ in range(self.layers)]
        retained_positions = start - self.retained + torch.arange(self.retained, device=device)
        for slot in range(slots):
            ring_rows = slot * RING + retained_positions.clamp(min=0) % RING
            for layer_index in range(self.layers):
                if start >= self.retained:
                    self.key_rings[layer_index][ring_rows] = self.key_store[layer_index][slot].transpose(0, 1).reshape(
                        self.retained, width_kv
                    )
                    self.value_rings[layer_index][ring_rows] = self.value_store[layer_index][slot].transpose(
                        0, 1
                    ).reshape(self.retained, width_kv)
                else:
                    pass

    def gathered_step(self, step):
        state = Qwen3TTSIncrementalCodecState(
            transformer_context_length=self.retained,
            frame_positions=self.positions.index_select(0, self.index),
        )
        for layer_index in range(self.layers):
            state.transformer_keys[layer_index] = self.key_store[layer_index].index_select(0, self.index)
            state.transformer_values[layer_index] = self.value_store[layer_index].index_select(0, self.index)
        output = step(self.hidden, state)
        for layer_index in range(self.layers):
            self.key_store[layer_index].index_copy_(0, self.index, state.transformer_keys[layer_index].contiguous())
            self.value_store[layer_index].index_copy_(0, self.index, state.transformer_values[layer_index].contiguous())
        if not self.first_decode:
            self.positions.index_copy_(0, self.index, state.frame_positions + self.width)
        else:
            pass
        return output

    def ring_step(self, step):
        start = self.positions.index_select(0, self.index)
        output = step(self.fused, self.hidden, self.key_rings, self.value_rings, self.index, start, self.first_decode)
        if not self.first_decode:
            self.positions.index_copy_(0, self.index, start + self.width)
        else:
            pass
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--variant", choices=("gathered", "ring"), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", args.card)
    torch.cuda.set_device(device)
    tokenizer = load_qwen3_tts_tokenizer(CHECKPOINT, device=str(device), dtype="bfloat16", attn_implementation="sdpa")
    fused = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder.eval()).fused_transformer
    with torch.inference_mode():
        check_gathered = Setup(fused, device, args.rows, args.width, args.first)
        check_ring = Setup(fused, device, args.rows, args.width, args.first)
        gathered_output = check_gathered.gathered_step(fused)
        ring_output = check_ring.ring_step(ring_transformer)
        print(
            f"check: output max abs difference {(gathered_output.float() - ring_output.float()).abs().max().item():.3e} "
            f"(output max {gathered_output.float().abs().max().item():.3e})",
            flush=True,
        )
        kept = 0.0
        end = (0 if args.first else START) + args.width
        kept_positions = end - check_ring.retained + torch.arange(check_ring.retained, device=device)
        for slot in check_ring.index.tolist():
            for layer_index in range(check_ring.layers):
                if args.first and args.width < check_ring.retained:
                    continue
                else:
                    pass
                ring_keys = check_ring.key_rings[layer_index][slot * RING + kept_positions % RING]
                store_keys = check_gathered.key_store[layer_index][slot].transpose(0, 1).reshape(check_ring.retained, -1)
                kept = max(kept, (ring_keys.float() - store_keys.float()).abs().max().item())
        print(f"check: kept keys max abs difference {kept:.3e}", flush=True)
        setup = Setup(fused, device, args.rows, args.width, args.first)
        if args.variant == "gathered":
            step = torch.compile(fused.__call__, dynamic=False, fullgraph=True) if args.compiled else fused
            run = lambda: setup.gathered_step(step)  # noqa: E731
        else:
            step = torch.compile(ring_transformer, dynamic=False, fullgraph=True) if args.compiled else ring_transformer
            run = lambda: setup.ring_step(step)  # noqa: E731
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(30):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(300):
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            finish.record()
            finish.synchronize()
            times.append(1e3 * begin.elapsed_time(finish))
        times.sort()
        print(
            f"{args.variant} rows {args.rows} width {args.width} {'compiled' if args.compiled else 'eager'}"
            f"{' first' if args.first else ''}: median {statistics.median(times):.1f} us, p90 {times[269]:.1f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()
