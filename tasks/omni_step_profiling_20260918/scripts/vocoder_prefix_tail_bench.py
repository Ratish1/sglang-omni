"""The reference-prefixed first chunk: decode the conv stack over the tail only.

A Base voice-clone first chunk is R reference frames plus g generated frames. The
reference frames are decoded only to build the stream's state; their audio is trimmed.
Everything after the decoder transformer is causal with a finite history, so the
generated frames' samples and the conv stack's state depend on the transformer's last C
outputs only, C = g + the conv stack's history in frames. The quantizer, pre_conv and
transformer (frame rate, 8 layers, sliding window 72) still run over all R + g frames.

  field     the conv stack's history in frames, derived from the module tree, per g
  exact     float64: the tail decode (rows of different R padded to one width W) against
            each row's full decode alone: generated samples and every state tensor, then
            two warm follow-ups from both states; also the smallest tail that is exact
  cost      bf16, fused SnakeBeta, captured graphs, ms per replay: today's window chain
            for width R + g (the served ladder 1..64, greedy split), against one tail
            replay at width W (reference ladder) with tail C, rows 1, 2, 4, 8

usage: python vocoder_prefix_tail_bench.py --model ID --mode field|exact|cost
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from sglang_omni.models.qwen3_tts import incremental_codec as codec
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    split_frames_by_width,
)

WINDOW_LADDER = (1, 2, 4, 8, 16, 32, 64)
REFERENCE_LADDER = (32, 48, 64, 96, 128, 192, 256)


def conv_need(module, need: int) -> int:
    """Input samples a causal conv needs for its last `need` outputs and its state."""
    return need + int(module.padding)


def transconv_need(module, need: int) -> int:
    """Input samples a transposed conv needs for its last `need` emitted samples and
    its overlap tail."""
    conv = module.conv
    kernel, stride = int(conv.kernel_size[0]), int(conv.stride[0])
    return (need + kernel - 1) // stride


def tail_frames(decoder, generated: int) -> int:
    """Transformer outputs the conv stack reads for `generated` new frames, walked back
    from the waveform through every conv, transposed conv and ConvNeXt."""
    need = generated * int(decoder.total_upsample)
    need = conv_need(decoder.decoder[-1], need)
    for block in reversed(decoder.decoder[1:-2]):
        for residual in reversed(block.block[2:]):
            need = conv_need(residual.conv1, conv_need(residual.conv2, need))
        need = transconv_need(block.block[1], need)
    need = conv_need(decoder.decoder[0], need)
    for blocks in reversed(decoder.upsample):
        need = conv_need(blocks[1].dwconv, need)
        need = transconv_need(blocks[0], need)
    return need


def frame_part(incremental, codes, valid):
    """Quantizer, pre_conv and transformer from a cold state over padded width W;
    returns (B, W, latent) outputs and the state those three own, gathered at n."""
    decoder = incremental.decoder
    batch, width = int(codes.shape[0]), int(codes.shape[-1])
    device, dtype = codes.device, decoder.pre_conv.conv.weight.dtype
    state = incremental.init_state(batch, device=device, dtype=dtype)
    hidden = decoder.quantizer.decode(codes)
    channels_last = incremental.channels_last_weights is not None
    if channels_last:
        hidden = hidden.transpose(1, 2)
        history = state.conv_histories["pre_conv"].transpose(1, 2)
        combined = torch.cat((history, hidden), dim=1)
    else:
        combined = torch.cat((state.conv_histories["pre_conv"], hidden), dim=-1)
    out = codec.causal_conv1d(
        decoder.pre_conv, hidden, state, "pre_conv", incremental.channels_last_weights
    )
    size = int(decoder.pre_conv.padding)
    index = valid[:, None] + torch.arange(size, device=device)
    if channels_last:
        kept = combined.gather(1, index[:, :, None].expand(-1, -1, combined.shape[2]))
        pre_history = kept.transpose(1, 2).contiguous()
    else:
        pre_history = combined.gather(
            2, index[:, None, :].expand(-1, combined.shape[1], -1)
        )
    if not channels_last:
        out = out.transpose(1, 2)
    transformer = decoder.pre_transformer
    hidden = transformer.input_proj(out)
    positions = torch.arange(width, device=device)[None, :].expand(batch, -1)
    prior = int(state.transformer_keys[0].shape[-2])
    key_positions = torch.arange(-prior, width, device=device)[None, :].expand(
        batch, -1
    )
    embeddings = transformer.rotary_emb(hidden, positions)
    retained = max(0, int(transformer.window_size) - 1)
    kv_index = valid[:, None] + torch.arange(retained, device=device)
    keys, values = {}, {}
    for layer_index, layer in enumerate(transformer.layers):
        residual = hidden
        attended, key, value = codec.incremental_attention(
            layer.self_attn,
            layer.input_layernorm(hidden),
            embeddings,
            state,
            layer_index,
            key_positions,
            positions,
        )
        hidden = residual + layer.self_attn_layer_scale(attended)
        residual = hidden
        hidden = residual + layer.mlp_layer_scale(
            layer.mlp(layer.post_attention_layernorm(hidden))
        )
        kept = kv_index[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        keys[layer_index] = key.gather(-2, kept)
        values[layer_index] = value.gather(-2, kept)
    hidden = transformer.output_proj(transformer.norm(hidden))
    state.conv_histories["pre_conv"] = pre_history
    state.transformer_keys, state.transformer_values = keys, values
    state.frame_positions = valid.clone()
    return hidden, state


def conv_part(incremental, hidden, state):
    """decode_tensors after the transformer, on (B, L, latent), against `state`."""
    decoder = incremental.decoder
    weights = incremental.channels_last_weights
    channels_last = weights is not None
    if not channels_last:
        hidden = hidden.transpose(1, 2)
    for stage_index, blocks in enumerate(decoder.upsample):
        hidden = codec.causal_transconv1d(
            blocks[0], hidden, state, f"upsample.{stage_index}.transconv", weights
        )
        hidden = codec.incremental_convnext(
            blocks[1], hidden, state, f"upsample.{stage_index}.convnext", channels_last
        )
    wave = codec.causal_conv1d(decoder.decoder[0], hidden, state, "decoder.0", weights)
    for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
        wave = codec.apply_activation(block.block[0], wave, channels_last)
        wave = codec.causal_transconv1d(
            block.block[1], wave, state, f"decoder.{block_index}.transconv", weights
        )
        for residual_index, residual in enumerate(block.block[2:]):
            wave = codec.incremental_residual_unit(
                residual,
                wave,
                state,
                f"decoder.{block_index}.residual.{residual_index}",
                weights,
            )
    wave = codec.apply_activation(decoder.decoder[-2], wave, channels_last)
    wave = codec.causal_conv1d(
        decoder.decoder[-1], wave, state, "decoder.final", weights
    )
    if channels_last:
        wave = wave.transpose(1, 2)
    return wave.clamp(min=-1, max=1)


def tail_decode(incremental, codes, valid, tail):
    """One padded call: frame part over W, the conv stack over the C outputs ending at n."""
    hidden, state = frame_part(incremental, codes, valid)
    index = (valid - tail)[:, None] + torch.arange(tail, device=codes.device)
    window = hidden.gather(1, index[:, :, None].expand(-1, -1, hidden.shape[2]))
    wave = conv_part(incremental, window, state)
    return wave, state


def state_tensors(state):
    for name in ("conv_histories", "transconv_overlaps"):
        for key, value in getattr(state, name).items():
            yield f"{name}.{key}", value
    for layer, value in state.transformer_keys.items():
        yield f"keys.{layer}", value
    for layer, value in state.transformer_values.items():
        yield f"values.{layer}", value
    yield "frame_positions", state.frame_positions


def max_rel(got, want):
    assert got.shape == want.shape, (got.shape, want.shape)
    if want.numel() == 0:
        return 0.0
    scale = want.abs().max().item()
    diff = (got.double() - want.double()).abs().max().item()
    return diff / scale if scale else diff


def run_field(incremental):
    decoder = incremental.decoder
    print(f"total upsample {int(decoder.total_upsample)}")
    for generated in (1, 2, 4, 8, 16):
        samples = tail_frames(decoder, generated)
        frames = math.ceil(samples - 1e-9)
        print(
            f"g {generated:>2}: conv stack reads {frames} transformer outputs "
            f"(history {frames - generated} frames)"
        )


def run_exact(incremental, device, seed):
    decoder = incremental.decoder
    upsample = int(decoder.total_upsample)
    generator = torch.Generator().manual_seed(seed)
    worst = {"wave": 0.0, "state": 0.0, "follow": 0.0}
    cases = 0
    for generated in (1, 2, 4):
        tail = tail_frames(decoder, generated)
        for width in (48, 64, 96):
            rows = 4
            low = max(tail, generated + 1)
            valid = torch.randint(low, width + 1, (rows,), generator=generator)
            valid[0] = width
            codes = torch.randint(0, 2048, (rows, 16, width), generator=generator)
            codes, valid = codes.to(device), valid.to(device)
            wave, state = tail_decode(incremental, codes, valid, tail)
            for row in range(rows):
                n = int(valid[row])
                alone = incremental.init_state(1, device=device, dtype=torch.float64)
                full = decoder_decode(incremental, codes[row : row + 1, :, :n], alone)
                got = wave[row, ..., -generated * upsample :]
                want = full[0, ..., -generated * upsample :]
                worst["wave"] = max(worst["wave"], max_rel(got, want))
                for (name, a), (_, b) in zip(
                    state_tensors(alone), state_tensors(state)
                ):
                    if name == "frame_positions":
                        assert int(b[row]) == int(a[0]), name
                        continue
                    worst["state"] = max(worst["state"], max_rel(b[row : row + 1], a))
                follow_codes = torch.randint(
                    0, 2048, (1, 16, 4), generator=generator
                ).to(device)
                mine = single_row(state, row)
                for _ in range(2):
                    a_wave = decoder_decode(incremental, follow_codes, alone)
                    b_wave = decoder_decode(incremental, follow_codes, mine)
                    worst["follow"] = max(worst["follow"], max_rel(b_wave, a_wave))
                cases += 1
        print(
            f"g {generated}: tail {tail}, {cases} rows so far; worst relative error "
            f"waveform {worst['wave']:.2e}, state {worst['state']:.2e}, "
            f"two follow-ups {worst['follow']:.2e}",
            flush=True,
        )
    generated = 1
    tail = tail_frames(decoder, generated)
    codes = torch.randint(0, 2048, (1, 16, 64), generator=generator).to(device)
    alone = incremental.init_state(1, device=device, dtype=torch.float64)
    full = decoder_decode(incremental, codes, alone)
    valid = torch.tensor([64], device=device)
    for short in range(tail - 3, tail + 1):
        wave, _ = tail_decode(incremental, codes, valid, short)
        error = max_rel(wave[0, ..., -upsample:], full[0, ..., -upsample:])
        print(f"tail {short} (derived {tail}): waveform relative error {error:.2e}")


def decoder_decode(incremental, codes, state):
    return incremental.decode(codes, state)


def single_row(state, row):
    picked = codec.Qwen3TTSIncrementalCodecState(
        transformer_context_length=state.transformer_context_length,
        frame_positions=state.frame_positions[row : row + 1].clone(),
    )
    for name in (
        "conv_histories",
        "transconv_overlaps",
        "transformer_keys",
        "transformer_values",
    ):
        source = getattr(state, name)
        setattr(picked, name, {k: v[row : row + 1].clone() for k, v in source.items()})
    picked.frame_position = int(state.frame_positions[row])
    return picked


def graph_ms(fn, reps=30):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return graph, statistics.median(times)


def chain_ms(runner, rows, width, device, reps=30):
    """The served chain: the window runner's decode_slots per greedy window."""
    split = runner.split_frames(width)
    codes = torch.randint(0, 2048, (rows, 16, width), device=device)
    slots = list(range(rows))

    def chain():
        offset = 0
        for piece in split:
            runner.decode_slots(codes[:, :, offset : offset + piece], slots)
            offset += piece

    for _ in range(3):
        chain()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        chain()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return split, statistics.median(times)


def run_cost(incremental, device):
    from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        Qwen3TTSIncrementalCodecCudaGraphRunner,
    )

    decoder = incremental.decoder
    dtype = decoder.pre_conv.conv.weight.dtype
    device = torch.device("cuda", torch.cuda.current_device())
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=64, device=device, dtype=dtype
    )
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=dtype,
        num_quantizers=16,
        mode="window",
        fresh_frames=WINDOW_LADDER,
        batch_sizes=(1, 2, 4, 8),
        compile_fresh_frames=(8,),
        arena=arena,
    )
    runner.capture()
    print(f"window runner enabled {runner.enabled}, keys {len(runner.graphs)}")
    generated = 1
    tail = tail_frames(decoder, generated)
    print(f"g {generated}, tail {tail}")
    print(
        f"{'R+g':>4} {'rows':>4} {'chain':>22} {'chain ms':>9} {'W':>4} {'tail ms':>8} {'one width ms':>12}"
    )
    for width in (33, 45, 57, 64, 65, 97):
        padded = next(b for b in REFERENCE_LADDER if b >= width)
        for rows in (1, 2, 4, 8):
            split, today = chain_ms(runner, rows, width, device)
            codes = torch.randint(0, 2048, (rows, 16, padded), device=device)
            valid = torch.full((rows,), width, device=device)
            index = torch.arange(rows, device=device)

            def tail_step():
                wave, state = tail_decode(incremental, codes, valid, tail)
                arena.scatter_by_index(index, state)
                return wave

            _, mine = graph_ms(tail_step)
            full_codes = torch.randint(0, 2048, (rows, 16, width), device=device)

            def one_width():
                state = incremental.init_state(rows, device=device, dtype=dtype)
                return incremental.decode_tensors(full_codes, state)

            _, full = graph_ms(one_width)
            print(
                f"{width:>4} {rows:>4} {str(split):>22} {today:>9.3f} {padded:>4} "
                f"{mine:>8.3f} {full:>12.3f}",
                flush=True,
            )
            torch.cuda.empty_cache()


def float64_norm(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def varlen_metadata(lengths, width, packed, device):
    """Static varlen metadata built outside the graph, as SGLang's FA3 backend keeps its
    graph metadata in static buffers."""
    if packed:
        starts = [0]
        for n in lengths:
            starts.append(starts[-1] + n)
        token_index = torch.cat(
            [i * width + torch.arange(n, device=device) for i, n in enumerate(lengths)]
        )
        positions = torch.cat([torch.arange(n, device=device) for n in lengths])
        cu = torch.tensor(starts, dtype=torch.int32, device=device)
        return token_index, positions, cu, max(lengths)
    rows = len(lengths)
    positions = torch.arange(width, device=device).repeat(rows)
    cu = torch.arange(0, (rows + 1) * width, width, dtype=torch.int32, device=device)
    return None, positions, cu, width


def frame_part_varlen(incremental, codes, metadata):
    """frame_part with SGLang's FA3 varlen kernel for the attention (causal, window 72).
    packed=False: rows padded to W, cu_seqlens of W each. packed=True: the transformer
    runs on the real frames only, concatenated, cu_seqlens of each row's n (host list
    `lengths`, static per graph)."""
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    decoder = incremental.decoder
    transformer = decoder.pre_transformer
    batch, width = int(codes.shape[0]), int(codes.shape[-1])
    device = codes.device
    hidden = decoder.quantizer.decode(codes).transpose(1, 2)
    history = torch.zeros(
        batch,
        int(decoder.pre_conv.padding),
        hidden.shape[2],
        device=device,
        dtype=hidden.dtype,
    )
    combined = torch.cat((history, hidden), dim=1)
    out = codec.channels_last_conv1d(
        combined,
        decoder.pre_conv.conv,
        incremental.channels_last_weights["pre_conv"],
        width,
    )
    hidden = transformer.input_proj(out).reshape(batch * width, -1)
    token_index, positions, cu, max_len = metadata
    if token_index is not None:
        hidden = hidden.index_select(0, token_index)
    tokens = int(hidden.shape[0])
    cos, sin = transformer.rotary_emb(hidden.unsqueeze(0), positions.unsqueeze(0))
    window = int(transformer.window_size)
    for layer in transformer.layers:
        attention = layer.self_attn
        head_dim = int(attention.head_dim)
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        query = (
            attention.q_proj(normalized).view(1, tokens, -1, head_dim).transpose(1, 2)
        )
        key = attention.k_proj(normalized).view(1, tokens, -1, head_dim).transpose(1, 2)
        value = attention.v_proj(normalized).view(tokens, -1, head_dim)
        query, key = codec.apply_rotary_pos_emb(query, key, cos, sin)
        attended = flash_attn_varlen_func(
            query[0].transpose(0, 1).contiguous(),
            key[0].transpose(0, 1).contiguous(),
            value,
            cu,
            cu,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            softmax_scale=float(attention.scaling),
            causal=True,
            window_size=(window - 1, 0),
        )
        attended = attention.o_proj(attended.reshape(tokens, -1))
        hidden = residual + layer.self_attn_layer_scale(attended)
        residual = hidden
        hidden = residual + layer.mlp_layer_scale(
            layer.mlp(layer.post_attention_layernorm(hidden))
        )
    return transformer.output_proj(transformer.norm(hidden))


def run_varlen(incremental, device):
    """Frame part ms: today's manual attention (padded), FA3 varlen padded, FA3 varlen
    packed; rows of n spread over [W/2, W] as first chunks are; and the bf16 distance of
    the varlen frame outputs to today's at the real frames."""
    print(
        f"{'W':>4} {'rows':>4} {'today ms':>9} {'fa padded':>10} {'fa packed':>10} {'max rel':>9}"
    )
    for padded in (48, 64, 128, 256):
        for rows in (1, 2, 4, 8):
            lengths = [
                padded - (padded // 2) * i // max(rows - 1, 1) for i in range(rows)
            ]
            codes = torch.randint(0, 2048, (rows, 16, padded), device=device)
            valid = torch.tensor(lengths, device=device)
            today = graph_ms(lambda: frame_part(incremental, codes, valid)[0])[1]
            padded_meta = varlen_metadata(lengths, padded, False, device)
            packed_meta = varlen_metadata(lengths, padded, True, device)
            fa_padded = graph_ms(
                lambda: frame_part_varlen(incremental, codes, padded_meta)
            )[1]
            fa_packed = graph_ms(
                lambda: frame_part_varlen(incremental, codes, packed_meta)
            )[1]
            reference = frame_part(incremental, codes, valid)[0]
            got = frame_part_varlen(incremental, codes, packed_meta)
            worst, start = 0.0, 0
            for row, n in enumerate(lengths):
                worst = max(worst, max_rel(got[start : start + n], reference[row, :n]))
                start += n
            print(
                f"{padded:>4} {rows:>4} {today:>9.3f} {fa_padded:>10.3f} {fa_packed:>10.3f} {worst:>9.2e}",
                flush=True,
            )
            torch.cuda.empty_cache()


class VarlenMetadata:
    """Live attention metadata read by the eager break at every replay."""

    def __init__(self, cu, max_len, scale, window):
        self.cu, self.max_len, self.scale, self.window = cu, max_len, scale, window


def bcg_attention(query, key, value, meta):
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query,
        key,
        value,
        meta.cu,
        meta.cu,
        max_seqlen_q=meta.max_len,
        max_seqlen_k=meta.max_len,
        softmax_scale=meta.scale,
        causal=True,
        window_size=(meta.window - 1, 0),
    )


def frame_part_bcg(incremental, codes, metadata, live, attention):
    """frame_part_varlen packed, with the attention as a BCG break."""
    decoder = incremental.decoder
    transformer = decoder.pre_transformer
    batch, width = int(codes.shape[0]), int(codes.shape[-1])
    hidden = decoder.quantizer.decode(codes).transpose(1, 2)
    history = torch.zeros(
        batch,
        int(decoder.pre_conv.padding),
        hidden.shape[2],
        device=codes.device,
        dtype=hidden.dtype,
    )
    out = codec.channels_last_conv1d(
        torch.cat((history, hidden), dim=1),
        decoder.pre_conv.conv,
        incremental.channels_last_weights["pre_conv"],
        width,
    )
    token_index, positions, _, _ = metadata
    hidden = transformer.input_proj(out).reshape(batch * width, -1)
    hidden = hidden.index_select(0, token_index)
    tokens = int(hidden.shape[0])
    cos, sin = transformer.rotary_emb(hidden.unsqueeze(0), positions.unsqueeze(0))
    for layer in transformer.layers:
        attn = layer.self_attn
        head_dim = int(attn.head_dim)
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        query = attn.q_proj(normalized).view(1, tokens, -1, head_dim).transpose(1, 2)
        key = attn.k_proj(normalized).view(1, tokens, -1, head_dim).transpose(1, 2)
        value = attn.v_proj(normalized).view(tokens, -1, head_dim)
        query, key = codec.apply_rotary_pos_emb(query, key, cos, sin)
        attended = attention(
            query[0].transpose(0, 1).contiguous(),
            key[0].transpose(0, 1).contiguous(),
            value,
            live,
        )
        attended = attn.o_proj(attended.reshape(tokens, -1))
        hidden = residual + layer.self_attn_layer_scale(attended)
        residual = hidden
        hidden = residual + layer.mlp_layer_scale(
            layer.mlp(layer.post_attention_layernorm(hidden))
        )
    return transformer.output_proj(transformer.norm(hidden))


def timed_replay(replay, reps=30):
    """Median device span (events) and host wall per replay, ms."""
    spans, walls = [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        start.record()
        replay()
        end.record()
        launched = time.perf_counter()
        end.synchronize()
        spans.append(start.elapsed_time(end))
        walls.append(1000 * (launched - began))
    return statistics.median(spans), statistics.median(walls)


def run_bcg(incremental, device):
    """Frame part, rows packed: a full graph with static FA3 metadata against a breakable
    graph whose 8 attention calls run eagerly between segments (SGLang's BCG), and
    today's full graph with the manual attention. Device span and host launch ms."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
        eager_on_graph,
    )

    window = int(incremental.decoder.pre_transformer.window_size)
    scale = float(incremental.decoder.pre_transformer.layers[0].self_attn.scaling)
    broken = eager_on_graph(True)(bcg_attention)
    print(
        f"{'W':>4} {'rows':>4} {'today span/host':>16} {'full fa span/host':>18} {'bcg span/host':>16}"
    )
    for padded in (64, 128):
        for rows in (1, 4, 8):
            lengths = [
                padded - (padded // 2) * i // max(rows - 1, 1) for i in range(rows)
            ]
            codes = torch.randint(0, 2048, (rows, 16, padded), device=device)
            valid = torch.tensor(lengths, device=device)
            meta = varlen_metadata(lengths, padded, True, device)
            live = VarlenMetadata(meta[2], meta[3], scale, window)
            cells = []
            for fn in (
                lambda: frame_part(incremental, codes, valid)[0],
                lambda: frame_part_varlen(incremental, codes, meta),
            ):
                graph, _ = graph_ms(fn)
                cells.append(timed_replay(graph.replay))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    frame_part_bcg(incremental, codes, meta, live, bcg_attention)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = BreakableCUDAGraph()
            with BreakableCUDAGraphCapture(
                graph, pool=torch.cuda.graph_pool_handle(), stream=stream
            ):
                frame_part_bcg(incremental, codes, meta, live, broken)
            torch.cuda.synchronize()
            cells.append(timed_replay(graph.replay))
            print(
                f"{padded:>4} {rows:>4} "
                + " ".join(f"{s:>8.3f}/{w:<7.3f}" for s, w in cells),
                flush=True,
            )
            torch.cuda.empty_cache()


def run_split(incremental, device):
    """Captured ms of the frame part at W against the conv stack at the tail, and the
    attention's share of the frame part (graph of the frame part with attention removed).
    """
    decoder = incremental.decoder
    dtype = decoder.pre_conv.conv.weight.dtype
    tail = tail_frames(decoder, 1)
    print(f"{'W':>4} {'rows':>4} {'frame ms':>9} {'conv ms':>8} {'whole ms':>9}")
    for padded in (48, 64, 128, 256):
        for rows in (1, 2, 4, 8):
            codes = torch.randint(0, 2048, (rows, 16, padded), device=device)
            valid = torch.full((rows,), padded, device=device)
            hidden = torch.randn(rows, tail, 1024, device=device, dtype=dtype)

            def conv_only():
                state = incremental.init_state(rows, device=device, dtype=dtype)
                return conv_part(incremental, hidden, state)

            frame = graph_ms(lambda: frame_part(incremental, codes, valid)[0])[1]
            conv = graph_ms(conv_only)[1]
            whole = graph_ms(lambda: tail_decode(incremental, codes, valid, tail))[1]
            print(
                f"{padded:>4} {rows:>4} {frame:>9.3f} {conv:>8.3f} {whole:>9.3f}",
                flush=True,
            )
            torch.cuda.empty_cache()


def run_capture(incremental, device):
    """The tail runner's graphs as the runner would hold them: one pool, largest key
    first, 3 warmups per key; seconds and memory for the whole set."""
    from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena

    decoder = incremental.decoder
    dtype = decoder.pre_conv.conv.weight.dtype
    device = torch.device("cuda", torch.cuda.current_device())
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=64, device=device, dtype=dtype
    )
    tail = tail_frames(decoder, 1)
    torch.cuda.synchronize()
    before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    began = time.perf_counter()
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    graphs = []
    for rows in (8, 4, 2, 1):
        for padded in reversed(REFERENCE_LADDER):
            codes = torch.zeros((rows, 16, padded), dtype=torch.long, device=device)
            valid = torch.full((rows,), padded, device=device)
            index = torch.full(
                (rows,), arena.scratch_slot, dtype=torch.long, device=device
            )

            def step():
                arena.gather_by_index(index)
                wave, out = tail_decode(incremental, codes, valid, tail)
                arena.scatter_by_index(index, out)
                return wave

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = step()
            graphs.append((graph, codes, valid, index, output))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    after = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    print(
        f"{len(graphs)} tail graphs: {seconds:.1f} s, allocated +{(after[0] - before[0]) / 2**20:.0f} MiB, "
        f"reserved +{(after[1] - before[1]) / 2**20:.0f} MiB"
    )


def stream_decode(incremental, codes, reference, first, ramp, first_chunk):
    """Decode one stream: a first chunk of reference + `first` frames by `first_chunk`,
    then follow-ups of `ramp` frames; returns the emitted samples, one tensor per chunk.
    """
    upsample = int(incremental.decoder.total_upsample)
    dtype = incremental.decoder.pre_conv.conv.weight.dtype
    state = incremental.init_state(1, device=codes.device, dtype=dtype)
    wave = first_chunk(incremental, codes[:, :, : reference + first], state)
    chunks = [wave[..., -first * upsample :].double().flatten()]
    offset = reference + first
    for width in ramp:
        piece = codes[:, :, offset : offset + width]
        chunks.append(incremental.decode(piece, state).double().flatten())
        offset += width
    return chunks


def chain_first_chunk(incremental, codes, state):
    """Today's served first chunk: the greedy window split, one decode per window."""
    pieces = []
    offset = 0
    for width in split_frames_by_width(int(codes.shape[-1]), WINDOW_LADDER):
        pieces.append(incremental.decode(codes[:, :, offset : offset + width], state))
        offset += width
    return torch.cat(pieces, dim=-1)


def tail_first_chunk(incremental, codes, state):
    width = int(codes.shape[-1])
    padded = next(b for b in REFERENCE_LADDER if b >= width)
    tail = tail_frames(incremental.decoder, 1)
    padded_codes = torch.nn.functional.pad(codes, (0, padded - width))
    valid = torch.tensor([width], device=codes.device)
    wave, tail_state = tail_decode(incremental, padded_codes, valid, tail)
    for name in (
        "conv_histories",
        "transconv_overlaps",
        "transformer_keys",
        "transformer_values",
    ):
        setattr(state, name, getattr(tail_state, name))
    state.frame_positions = tail_state.frame_positions
    state.frame_position = width
    return wave


def full_first_chunk(incremental, codes, state):
    return incremental.decode(codes, state)


def run_numerics(raw, device, references):
    import copy

    from benchmarks.dataset.seedtts import load_seedtts_samples
    from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

    truth = codec.Qwen3TTSIncrementalDecoder(copy.deepcopy(raw).to(torch.float64))
    fuse_vocoder_decoder(raw)
    served = codec.Qwen3TTSIncrementalDecoder(raw)
    tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device=str(device),
        dtype="bfloat16",
        attn_implementation=None,
    )
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    distinct = list({s.ref_audio: s for s in samples}.values())[:references]
    encoded = tokenizer.encode([s.ref_audio for s in distinct]).audio_codes
    ramp = (2, 4, 8, 8)
    follow = sum(ramp)
    rows = {"chain": [], "tail": []}
    for codes in encoded:
        codes = codes.to(device).transpose(0, 1).unsqueeze(0)
        reference = int(codes.shape[-1]) - 1 - follow
        if reference < tail_frames(raw, 1):
            continue
        want = stream_decode(truth, codes, reference, 1, ramp, full_first_chunk)
        for arm, first_chunk in (
            ("chain", chain_first_chunk),
            ("tail", tail_first_chunk),
        ):
            got = stream_decode(served, codes, reference, 1, ramp, first_chunk)
            rows[arm].append([snr(g, w) for g, w in zip(got, want)])
    print(f"{len(rows['tail'])} references, first chunk then follow-ups {ramp}")
    for arm, values in rows.items():
        table = torch.tensor(values)
        print(
            f"{arm:>5}: SNR to float64 dB, median per chunk "
            f"{[round(v, 1) for v in table.median(dim=0).values.tolist()]}, "
            f"min per chunk {[round(v, 1) for v in table.min(dim=0).values.tolist()]}"
        )


def snr(got, want):
    noise = (got - want).pow(2).sum().item()
    return (
        float("inf")
        if noise == 0
        else 10 * math.log10(want.pow(2).sum().item() / noise)
    )


class Float64Softmax:
    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    def softmax(self, scores, dim, dtype=None):
        return torch.nn.functional.softmax(scores, dim=dim, dtype=scores.dtype)


def run_width(incremental, device):
    """Tail replay ms against the padded width W, rows of n = 33 and n = W."""
    decoder = incremental.decoder
    dtype = decoder.pre_conv.conv.weight.dtype
    device = torch.device("cuda", torch.cuda.current_device())
    tail = tail_frames(decoder, 1)
    print(f"{'W':>4} {'rows':>4} {'n=33 ms':>8} {'n=W ms':>8}")
    for padded in (16, 32, 48, 64, 96, 128, 192, 256, 257):
        for rows in (1, 2, 4, 8):
            codes = torch.randint(0, 2048, (rows, 16, padded), device=device)
            cells = []
            for width in (min(33, padded), padded):
                valid = torch.full((rows,), width, device=device)

                cells.append(
                    graph_ms(lambda v=valid: tail_decode(incremental, codes, v, tail))[
                        1
                    ]
                )
            print(
                f"{padded:>4} {rows:>4} {cells[0]:>8.3f} {cells[1]:>8.3f}", flush=True
            )
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument(
        "--mode",
        choices=(
            "field",
            "exact",
            "cost",
            "width",
            "numerics",
            "capture",
            "split",
            "varlen",
            "bcg",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--references", type=int, default=24)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    load_dtype = "float32" if args.mode == "exact" else "bfloat16"
    tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        args.model, device=str(device), dtype=load_dtype, attn_implementation=None
    )
    raw = tokenizer.model.decoder
    if args.mode == "numerics":
        with torch.inference_mode():
            run_numerics(raw, device, args.references)
        return
    elif args.mode == "exact":
        raw = raw.to(torch.float64)
        # the model's attention softmax and RMSNorm are float32 by its code, and their
        # reductions round by tensor shape; float64 here separates the mechanism
        codec.F = Float64Softmax()
        norm = type(raw.pre_transformer.norm)
        norm.forward = float64_norm
    else:
        from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

        print(f"fused SnakeBeta modules: {fuse_vocoder_decoder(raw)}")
    incremental = codec.Qwen3TTSIncrementalDecoder(raw)
    print(
        f"{torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"channels last {incremental.channels_last_weights is not None}"
    )
    with torch.inference_mode():
        {
            "field": lambda: run_field(incremental),
            "exact": lambda: run_exact(incremental, device, args.seed),
            "cost": lambda: run_cost(incremental, device),
            "width": lambda: run_width(incremental, device),
            "capture": lambda: run_capture(incremental, device),
            "split": lambda: run_split(incremental, device),
            "varlen": lambda: run_varlen(incremental, device),
            "bcg": lambda: run_bcg(incremental, device),
        }[args.mode]()


if __name__ == "__main__":
    main()
