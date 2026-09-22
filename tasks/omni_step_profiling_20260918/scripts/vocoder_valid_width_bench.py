"""V-e11: rows of different widths in one vocoder decode, by a per row valid width.

The stateful decoder is causal in time, so frames to the right of a row's real width
cannot change its real samples. This bench carries a [B] tensor of valid widths through
the decode and moves the four right edge reads to it: conv history, transposed conv
overlap (inputs past the valid width zeroed first), the K/V kept, the position advance.
Nothing in the runtime is changed; the variant lives here as patched module functions.

exact   rows of random valid widths inside a padded decode, 3 chained steps, against the
        exact width decode of each row alone: real samples and every state tensor.
        Also today's own spread: the same exact width row alone against inside a batch.
grid    captured graph replay ms for every (width, rows) key, today's decode and the
        variant, so the census can be replayed offline under a merge rule.

usage: python vocoder_valid_width_bench.py --model DIR --mode exact|grid [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts import incremental_codec as codec
from sglang_omni.models.qwen3_tts import stages as qwen3_stages

ORIGINAL_CONV = codec.incremental_causal_conv1d
ORIGINAL_TRANSCONV = codec.incremental_causal_transconv1d
ORIGINAL_TRANSFORMER = codec.incremental_transformer


def edge_index(valid: torch.Tensor, rate: int, length: int) -> torch.Tensor:
    """[B, length] indices of the `length` entries that start at valid * rate."""
    return (valid * rate)[:, None] + torch.arange(length, device=valid.device)


def conv_valid(module, hidden_states, state, key):
    valid = getattr(state, "valid_frames", None)
    if valid is None:
        return ORIGINAL_CONV(module, hidden_states, state, key)
    history_size = int(module.padding)
    combined = torch.cat((state.conv_histories[key], hidden_states), dim=-1)
    output = module.conv(combined).contiguous()
    rate = hidden_states.shape[-1] // state.padded_frames
    index = edge_index(valid, rate, history_size)
    state.conv_histories[key] = combined.gather(
        -1, index[:, None, :].expand(-1, combined.shape[1], -1)
    )
    return output


def transconv_valid(module, hidden_states, state, key):
    valid = getattr(state, "valid_frames", None)
    if valid is None:
        return ORIGINAL_TRANSCONV(module, hidden_states, state, key)
    conv = module.conv
    stride = int(conv.stride[0])
    right_pad = int(module.right_pad)
    rate = hidden_states.shape[-1] // state.padded_frames
    steps = torch.arange(hidden_states.shape[-1], device=hidden_states.device)
    real = steps[None, :] < (valid * rate)[:, None]
    hidden_states = torch.where(real[:, None, :], hidden_states, 0)
    output = F.conv_transpose1d(
        hidden_states,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        dilation=conv.dilation,
    )
    overlap = state.transconv_overlaps[key]
    output[..., : int(overlap.shape[-1])] += overlap
    index = edge_index(valid, rate * stride, right_pad)
    state.transconv_overlaps[key] = output.gather(
        -1, index[:, None, :].expand(-1, output.shape[1], -1)
    )
    emitted = output[..., : int(hidden_states.shape[-1]) * stride]
    if conv.bias is not None:
        emitted = emitted + conv.bias.view(1, -1, 1)
    return emitted.contiguous()


def transformer_valid(transformer, hidden_states, state):
    valid = getattr(state, "valid_frames", None)
    if valid is None:
        return ORIGINAL_TRANSFORMER(transformer, hidden_states, state)
    hidden_states = transformer.input_proj(hidden_states)
    batch_size, fresh_frames = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    device = hidden_states.device
    frame_positions = state.row_frame_positions(batch_size, device)
    prior_length = int(state.transformer_keys[0].shape[-2])
    key_offsets = torch.arange(-prior_length, fresh_frames, device=device)
    query_offsets = torch.arange(fresh_frames, device=device)
    key_positions = frame_positions.unsqueeze(1) + key_offsets.unsqueeze(0)
    query_positions = frame_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
    position_embeddings = transformer.rotary_emb(hidden_states, query_positions)
    retained = max(0, int(transformer.window_size) - 1)
    index = edge_index(valid, 1, retained)
    next_keys, next_values = {}, {}
    for layer_index, layer in enumerate(transformer.layers):
        residual = hidden_states
        attended, key, value = codec.incremental_attention(
            layer.self_attn,
            layer.input_layernorm(hidden_states),
            position_embeddings,
            state,
            layer_index,
            key_positions,
            query_positions,
        )
        hidden_states = residual + layer.self_attn_layer_scale(attended)
        residual = hidden_states
        hidden_states = layer.mlp(layer.post_attention_layernorm(hidden_states))
        hidden_states = residual + layer.mlp_layer_scale(hidden_states)
        kept = index[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        next_keys[layer_index] = key.gather(-2, kept)
        next_values[layer_index] = value.gather(-2, kept)
    state.transformer_keys, state.transformer_values = next_keys, next_values
    return transformer.output_proj(transformer.norm(hidden_states))


codec.incremental_causal_conv1d = conv_valid
codec.incremental_causal_transconv1d = transconv_valid
codec.incremental_transformer = transformer_valid


def decode_valid(decoder, codes, state, valid):
    state.valid_frames, state.padded_frames = valid, int(codes.shape[-1])
    waveform = decoder.decode_tensors(codes, state)
    state.frame_positions = state.frame_positions + valid
    return waveform


def state_tensors(state):
    for name in ("conv_histories", "transconv_overlaps"):
        for key, value in getattr(state, name).items():
            yield f"{name}.{key}", value
    for layer, value in state.transformer_keys.items():
        yield f"keys.{layer}", value
    for layer, value in state.transformer_values.items():
        yield f"values.{layer}", value
    yield "frame_positions", state.frame_positions


def compare(label, got, want, results):
    got, want = got.float(), want.float()
    diff = (got - want).abs().max().item() if got.numel() else 0.0
    scale = want.abs().max().item() if want.numel() else 0.0
    results.append((label, bool(torch.equal(got, want)), diff, scale))


def snr_db(got, want):
    noise = (got.float() - want.float()).pow(2).sum().item()
    signal = want.float().pow(2).sum().item()
    return (
        float("inf")
        if noise == 0
        else 10 * torch.log10(torch.tensor(signal / noise)).item()
    )


def run_exact(decoder, device, dtype, padded, rows, steps, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upsample = decoder.total_upsample
    batch_state = decoder.init_state(rows, device=device, dtype=dtype)
    alone = [decoder.init_state(1, device=device, dtype=dtype) for _ in range(rows)]
    inside = decoder.init_state(rows, device=device, dtype=dtype)
    results, snrs, spread = [], [], []
    for step in range(steps):
        valid = torch.randint(1, padded + 1, (rows,), generator=generator)
        if step == 0:
            valid[0], valid[-1] = padded, 1
        codes = torch.randint(0, 2048, (rows, 16, padded), generator=generator)
        codes, valid = codes.to(device), valid.to(device)
        waveform = decode_valid(decoder, codes, batch_state, valid)
        for row in range(rows):
            n = int(valid[row])
            want = decoder.decode(codes[row : row + 1, :, :n], alone[row])
            got = waveform[row : row + 1, ..., : n * upsample]
            compare(f"step{step} row{row} n={n} waveform", got, want, results)
            snrs.append(snr_db(got, want))
            for (name, a), (_, b) in zip(
                state_tensors(alone[row]), state_tensors(batch_state)
            ):
                compare(f"step{step} row{row} {name}", b[row : row + 1], a, results)
        # today's own spread: every row at the full width, alone against inside a batch
        full = torch.randint(0, 2048, (rows, 16, padded), generator=generator).to(
            device
        )
        batch_wave = decoder.decode(full, inside)
        lone_state = decoder.init_state(1, device=device, dtype=dtype)
        if step == 0:
            lone_wave = decoder.decode(full[:1], lone_state)
            spread.append(snr_db(batch_wave[:1], lone_wave))
    wave = [r for r in results if r[0].endswith("waveform")]
    states = [r for r in results if not r[0].endswith("waveform")]
    worst_state = max(states, key=lambda r: r[2])
    print(
        f"padded {padded:>2} rows {rows} steps {steps}: waveform bitwise "
        f"{sum(r[1] for r in wave)}/{len(wave)}, worst max abs "
        f"{max(r[2] for r in wave):.3e}, SNR min {min(snrs):.1f} dB median "
        f"{statistics.median(snrs):.1f} dB | state bitwise "
        f"{sum(r[1] for r in states)}/{len(states)}, worst {worst_state[0]} "
        f"{worst_state[2]:.3e} (scale {worst_state[3]:.3e}) | today alone vs in batch "
        f"SNR {spread[0]:.1f} dB"
    )
    return results


def replay_ms(fn, codes, state, reps=30):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            fn(codes, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        fn(codes, state)
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    graph.reset()
    return statistics.median(times)


def run_grid(decoder, device, dtype, out):
    grid = {}
    for width in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64):
        for rows in (1, 2, 4, 8):
            codes = torch.randint(0, 2048, (rows, 16, width), device=device)
            today_state = decoder.init_state(rows, device=device, dtype=dtype)
            today = replay_ms(decoder.decode_tensors, codes, today_state)
            valid_state = decoder.init_state(rows, device=device, dtype=dtype)
            valid_state.valid_frames = torch.full((rows,), width, device=device)
            valid_state.padded_frames = width
            variant = replay_ms(decoder.decode_tensors, codes, valid_state)
            grid[f"{width}x{rows}"] = {"today_ms": today, "valid_ms": variant}
            print(
                f"width {width:>2} rows {rows}: today {today:7.3f} ms, valid width {variant:7.3f} ms"
            )
            del today_state, valid_state
            torch.cuda.empty_cache()
    with open(out, "w") as handle:
        json.dump(grid, handle, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("exact", "grid"), required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default="valid_width_grid.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    # TF32 convs and matmuls round at about 1e-3, which would hide a
    # mechanism error in float32; float64 (loaded as float32, then widened) is the
    # reference that separates the mechanism from kernel rounding
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    load_dtype = "float32" if args.dtype == "float64" else args.dtype
    tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        args.model, device=str(device), dtype=load_dtype, attn_implementation=None
    )
    raw_decoder = tokenizer.model.decoder.to(dtype)
    if args.mode == "grid":
        # the server fuses SnakeBeta before it captures its graphs
        from sglang_omni.models.qwen3_tts.vocoder_kernels import fuse_vocoder_decoder

        print(f"fused SnakeBeta modules: {fuse_vocoder_decoder(raw_decoder)}")
    decoder = codec.Qwen3TTSIncrementalDecoder(raw_decoder)
    print(f"dtype {args.dtype}, total upsample {decoder.total_upsample}")
    with torch.inference_mode():
        if args.mode == "exact":
            for padded, rows in ((8, 4), (8, 8), (64, 2), (64, 4)):
                run_exact(
                    decoder, device, dtype, padded, rows, steps=3, seed=padded + rows
                )
        else:
            run_grid(decoder, device, dtype, args.out)


if __name__ == "__main__":
    main()
