#!/usr/bin/env python3
"""E5: whether a hop can reuse the finished chunks instead of recomputing them.

Run on the H100 venv from the branch worktree, alone on the GPU, no server:

  python tasks/cosyvoice_utilization_20260912/stage0/e5_hop_prefix_exactness.py \
      --device cuda:0 --dtype float64 --json e5_float64.json
  python tasks/cosyvoice_utilization_20260912/stage0/e5_hop_prefix_exactness.py \
      --device cuda:0 --dtype bfloat16 --json e5_bfloat16.json

Table 1, prefix stability: the mel frames hop k emitted against the same frames
recomputed by hop k+1, for the production packed call and the padded DiT call.
Table 2, schedule independence: the emitted mel of the 25, 50, 100 schedule
against a fixed 25 token schedule over the same frames. Table 3, cached hop:
a hop that runs only its new frames over K/V cached per Euler step and block
from the previous hops, against the production packed call of the full window.

Both paths end in the solver's float32 cast, so the float64 criterion is the
count of elements that differ by more than two float32 ulps of their magnitude.
Generated tokens longer than one reference are the concatenation of several
references' tokens; the causal structure under test does not depend on values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_ID,
    Stream,
    build_streams,
    compare,
    flow_input,
    hop_window,
    load_vocoder,
    provenance,
)

from sglang_omni.models.fun_cosyvoice3.stages import (  # noqa: E402
    FunCosyVoice3Flow,
    generate_flow,
    pack_flow_inputs,
    prepare_flow_conditioning,
    split_generated_mels,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
)

PROMPT_TOKENS = 2 * TOKEN_HOP_LEN
SCHEDULES = {
    "growth 25,50,100": ((0, 25), (25, 50), (75, 100)),
    "fixed 25": tuple((offset, 25) for offset in range(0, 175, 25)),
}


def extend_tokens(streams: list[Stream], count: int, needed: int) -> list[Stream]:
    extended = []
    for index in range(count):
        parts = [streams[index].tokens]
        cursor = index + 1
        while sum(part.shape[1] for part in parts) < needed:
            parts.append(streams[cursor % len(streams)].tokens)
            cursor += 1
        extended.append(
            replace(streams[index], tokens=torch.cat(parts, dim=1)[:, :needed])
        )
    return extended


def ulp_mismatches(value: torch.Tensor, reference: torch.Tensor) -> int:
    tolerance = (
        2
        * torch.finfo(torch.float32).eps
        * torch.maximum(value.abs(), reference.abs()).to(torch.float64)
    )
    diff = (value.to(torch.float64) - reference.to(torch.float64)).abs()
    return int((diff > tolerance).sum())


def packed_hop(vocoder, stream: Stream, offset: int, hop: int) -> torch.Tensor:
    return vocoder.hop_batch([flow_input(stream, hop_window(offset, hop))])[0]


def padded_hop(vocoder, stream: Stream, offset: int, hop: int) -> torch.Tensor:
    flow = vocoder.flow
    item = flow_input(stream, hop_window(offset, hop))
    packed = pack_flow_inputs(flow.flow, [item])
    with torch.autocast(
        device_type="cuda",
        dtype=vocoder.autocast_dtype,
        enabled=vocoder.autocast_dtype is not None,
    ):
        generated = generate_flow(flow, packed, streaming=True, finalize=False)
    return split_generated_mels(
        flow.flow,
        packed,
        generated,
        token_lengths=tuple(
            length - PRE_LOOKAHEAD_LEN for length in packed.combined_token_lengths
        ),
        target_token_lengths=tuple(
            length - PRE_LOOKAHEAD_LEN for length in packed.target_token_lengths
        ),
    )[0]


def cached_forward(dit, x, mu, cond, spks, t, freqs, scale, visible, cache, step):
    from x_transformers.x_transformers import apply_rotary_pos_emb

    time_embedding = dit.time_embed(t)
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    conv = dit.input_embed.conv_pos_embed
    context = conv.kernel_size - 1
    tails = cache.get(("conv", step))
    if tails is None:
        zeros = h.new_zeros(h.shape[0], context, h.shape[2])
        tails = (zeros, zeros)
    conv1_in = torch.cat((tails[0], h), dim=1)
    conv1_out = conv.conv1(conv1_in.transpose(1, 2)).transpose(1, 2)
    conv2_in = torch.cat((tails[1], conv1_out), dim=1)
    conv2_out = conv.conv2(conv2_in.transpose(1, 2)).transpose(1, 2)
    cache[("conv", step)] = (conv1_in[:, -context:], conv2_in[:, -context:])
    h = conv2_out + h
    residual = h
    for index, block in enumerate(dit.transformer_blocks):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(
            h, emb=time_embedding
        )
        attn = block.attn
        query = apply_rotary_pos_emb(attn.to_q(norm), freqs, scale)
        key = apply_rotary_pos_emb(attn.to_k(norm), freqs, scale**-1.0)
        value = attn.to_v(norm)
        past = cache.get(("kv", step, index))
        if past is not None:
            key = torch.cat((past[0], key), dim=1)
            value = torch.cat((past[1], value), dim=1)
        cache[("kv", step, index)] = (key, value)
        rows, frames = query.shape[0], query.shape[1]
        out = F.scaled_dot_product_attention(
            query.view(rows, frames, attn.heads, -1).transpose(1, 2),
            key.view(rows, key.shape[1], attn.heads, -1).transpose(1, 2),
            value.view(rows, value.shape[1], attn.heads, -1).transpose(1, 2),
            attn_mask=visible,
        )
        out = out.transpose(1, 2).reshape(rows, frames, -1).to(query.dtype)
        h = h + gate_msa.unsqueeze(1) * attn.to_out[1](attn.to_out[0](out))
        ff_norm = block.ff_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    return dit.proj_out(dit.norm_out(h, time_embedding))


def cached_hop(
    flow: FunCosyVoice3Flow, stream: Stream, offset: int, hop: int, start: int, cache
) -> tuple[torch.Tensor, int]:
    item = flow_input(stream, hop_window(offset, hop))
    conditioning = prepare_flow_conditioning(
        flow, pack_flow_inputs(flow.flow, [item]), finalize=False
    )
    length = conditioning.mel_lengths[0]
    dit = flow.decoder.estimator
    new = slice(start, length)
    x = conditioning.noisy_mel[:, :, new].transpose(1, 2)
    mu = conditioning.token_condition[:, :, new].transpose(1, 2)
    cond = conditioning.prompt_mel[:, :, new].transpose(1, 2)
    spks = conditioning.speaker_embedding[:, None, :].expand(-1, x.shape[1], -1)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    positions = torch.arange(start, length, device=x.device)
    freqs, scale = dit.rotary_embed.forward_from_seq_len(length)
    freqs = freqs[:, positions]
    if isinstance(scale, torch.Tensor):
        scale = scale[:, positions]
    chunk = int(dit.static_chunk_size)
    visible = (
        torch.arange(length, device=x.device)[None, :]
        < ((positions // chunk + 1) * chunk)[:, None]
    )
    cfg = flow.decoder.inference_cfg_rate
    time_span = conditioning.time_span
    flow_time = torch.zeros(1, device=x.device, dtype=time_span.dtype)
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        flow_time[:] = t
        field = cached_forward(
            dit,
            torch.cat((x, x), dim=0),
            mu_cfg,
            cond_cfg,
            spks_cfg,
            flow_time,
            freqs,
            scale,
            visible,
            cache,
            step - 1,
        )
        x = x + dt * ((1.0 + cfg) * field[:1] - cfg * field[1:])
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
    return x.transpose(1, 2).float(), length


def row(label: str, value: torch.Tensor, reference: torch.Tensor) -> dict:
    result = compare(value, reference)
    result["float32_ulp_mismatches"] = ulp_mismatches(value, reference)
    result["elements"] = value.numel()
    result["row"] = label
    return result


def print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'row':<64}{'equal':>7}{'ulp_mismatch':>14}{'max_abs':>12}{'snr_db':>9}")
    for item in rows:
        print(
            f"{item['row']:<64}{str(item['equal']):>7}"
            f"{item['float32_ulp_mismatches']:>8}/{item['elements']:<6}"
            f"{item['max_abs']:>11.2e}{item['snr_db']:>9.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "bfloat16"), default="float64")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--json")
    args = parser.parse_args()

    info = provenance(args.device)
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    checkpoint, vocoder = load_vocoder(args.model, args.device, autocast_dtype)
    if args.dtype == "float64":
        vocoder.flow.to(torch.float64)
    needed = max(hop_window(*schedule[-1]) for schedule in SCHEDULES.values())
    references = build_streams(
        checkpoint,
        args.device,
        count=4 * args.streams,
        prompt_tokens=PROMPT_TOKENS,
        min_generated=TOKEN_HOP_LEN,
        samples=args.samples,
    )
    streams = extend_tokens(references, args.streams, needed)
    print(f"streams {[stream.sample_id for stream in streams]}, {needed} tokens each")

    def autocast():
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_dtype is not None,
        )

    prefix_rows, schedule_rows, cached_rows = [], [], []
    kv_bytes_per_frame = None
    with torch.inference_mode():
        for stream in streams:
            emitted: dict[tuple[str, str], torch.Tensor] = {}
            for path_name, path in (("packed", packed_hop), ("padded", padded_hop)):
                for schedule_name, schedule in SCHEDULES.items():
                    mels = [path(vocoder, stream, o, h) for o, h in schedule]
                    for index in range(len(schedule) - 1):
                        end = (
                            schedule[index][0] + schedule[index][1]
                        ) * TOKEN_MEL_RATIO
                        prefix_rows.append(
                            row(
                                f"{stream.sample_id} {path_name} {schedule_name} "
                                f"hop {index} vs {index + 1}",
                                mels[index + 1][:, :, :end],
                                mels[index],
                            )
                        )
                    emitted[(path_name, schedule_name)] = torch.cat(
                        [
                            mel[:, :, offset * TOKEN_MEL_RATIO :]
                            for mel, (offset, _) in zip(mels, schedule, strict=True)
                        ],
                        dim=2,
                    )
                schedule_rows.append(
                    row(
                        f"{stream.sample_id} {path_name} growth vs fixed 25",
                        emitted[(path_name, "growth 25,50,100")],
                        emitted[(path_name, "fixed 25")],
                    )
                )
            for schedule_name, schedule in SCHEDULES.items():
                cache: dict = {}
                start = 0
                for index, (offset, hop) in enumerate(schedule):
                    with autocast():
                        cached, length = cached_hop(
                            vocoder.flow, stream, offset, hop, start, cache
                        )
                    if start == 0:
                        cached = cached[
                            :, :, (PROMPT_TOKENS + offset) * TOKEN_MEL_RATIO :
                        ]
                    reference = packed_hop(vocoder, stream, offset, hop)[
                        :, :, offset * TOKEN_MEL_RATIO :
                    ]
                    cached_rows.append(
                        row(
                            f"{stream.sample_id} {schedule_name} hop {index} "
                            f"new frames {start}..{length}",
                            cached,
                            reference,
                        )
                    )
                    start = length
                kv = sum(
                    tensor.numel() * tensor.element_size()
                    for key, pair in cache.items()
                    if key[0] == "kv"
                    for tensor in pair
                )
                kv_bytes_per_frame = kv / start

    print_rows("1. prefix stability: hop k frames recomputed by hop k+1", prefix_rows)
    print_rows("2. emitted mel, hop schedule growth against fixed 25", schedule_rows)
    print_rows(
        "3. cached hop (new frames only) against the packed full window", cached_rows
    )
    print(
        f"\ncached K/V bytes per frame at {args.dtype}: {kv_bytes_per_frame:.0f} "
        f"(10 Euler steps, 22 blocks, K and V, 2 CFG rows)"
    )
    if args.json:
        with open(args.json, "w") as out:
            json.dump(
                {
                    "provenance": info,
                    "dtype": args.dtype,
                    "prefix": prefix_rows,
                    "schedule": schedule_rows,
                    "cached": cached_rows,
                    "kv_bytes_per_frame": kv_bytes_per_frame,
                },
                out,
                indent=1,
            )


if __name__ == "__main__":
    main()
