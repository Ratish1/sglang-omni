#!/usr/bin/env python3
"""G0: does a hop that reuses cached K/V match the full window recompute.

Gate for roadmap rows 1.1 and 1.2, defined in
`../plans/11_hop_prefix_cache.md` section 6. No runtime code changes; run on
the H100 venv, alone on the GPU, no server. Steps: `README.md`.

Every hop of a staggered multi row schedule runs three ways over the same
inputs:

  truth       the packed full window causal call in float32
  production  the same call in bfloat16 autocast, what main serves today
  cached      the new frames only, K/V of the earlier frames read from an
              SGLang MHATokenToKVPool through FA3 with a page table, in
              bfloat16 autocast

Reported per hop and row, for production and for cached against truth: the
emitted mel, the magnitude spectrum of the HiFT waveform delta, and that raw
waveform. The mel and the spectrum are gated, and cached passes when it stays
within --gate-margin-db of production at the minimum and at the median, with
no NaN or Inf and equal lengths. The raw waveform is a diagnostic only: HiFT's
excitation phase is a cumulative sum of the predicted F0, so it decorrelates
for the shipped path too (2026-09-16: -1.6 dB, against 36.4 dB on the mel).

The cached path drives production's own solver, DiT forward and Euler update
(`solve_flow_euler_packed` over a `PackedDiT` subclass). Only attention, the
causal conv position embedding and the RoPE positions are replaced, so the
module order cannot drift and `flow_time` keeps production's dtype, which the
stage 0 E5 cached rows did not.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import time
from dataclasses import dataclass, field, replace

import torch
from common import (
    MODEL_ID,
    Stream,
    build_streams,
    compare,
    extend_tokens,
    load_vocoder,
    provenance,
)
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    gather_rows,
    pack_rows,
    scatter_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    pack_flow_inputs,
    prepare_flow_conditioning,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)

# Prompt lengths in speech tokens, none a hop multiple, so the serving prompt
# padding runs and the batch carries mixed row widths.
PROMPT_TOKENS = (40, 55, 63, 78, 91, 110, 127, 144)
CACHE_DTYPE = torch.bfloat16
STFT_SIZE, STFT_HOP = 1024, 256
METRICS = ("mel", "spectrum", "waveform")
GATED = ("mel", "spectrum")


@dataclass
class StreamCache:
    """One stream's cached K/V slots and causal conv tails, per CFG lane."""

    slots: list[torch.Tensor]
    frames: int = 0
    tails: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


@dataclass
class CachedCall:
    """The layout one cached hop needs: where its new frames live in the pool,
    which slots every query segment may read, and each row's conv tails."""

    pool: MHATokenToKVPool
    heads: int
    head_dim: int
    entries: list[tuple[StreamCache, int]]
    lengths_tensor: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    max_frames: int
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int

    def conv_tails(
        self, step: int, like: torch.Tensor, context: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = like.new_zeros(context, like.shape[2])
        pairs = [cache.tails.get((lane, step)) for cache, lane in self.entries]
        return (
            torch.stack([zeros if pair is None else pair[0] for pair in pairs]),
            torch.stack([zeros if pair is None else pair[1] for pair in pairs]),
        )

    def store_conv_tails(
        self, step: int, conv1_in: torch.Tensor, conv2_in: torch.Tensor, context: int
    ) -> None:
        index = self.lengths_tensor.unsqueeze(1) + torch.arange(
            context, device=conv1_in.device
        ).unsqueeze(0)
        index = index.unsqueeze(-1).expand(-1, -1, conv1_in.shape[2])
        tails1 = torch.gather(conv1_in, 1, index)
        tails2 = torch.gather(conv2_in, 1, index)
        for row, (cache, lane) in enumerate(self.entries):
            cache.tails[(lane, step)] = (tails1[row], tails2[row])


class CachedRowAttention:
    """Attention for the new frames against the pool: the row's new K/V are
    written to its slots, then one FA3 call reads [0, chunk end) of every
    (row, chunk) query segment through the page table."""

    def __init__(self, call: CachedCall) -> None:
        self.call = call
        self.layer = 0

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        call = self.call
        shape = (-1, call.heads, call.head_dim)
        call.pool.set_kv_buffer(
            None,
            call.slots,
            key[0].reshape(shape).to(CACHE_DTYPE).contiguous(),
            value[0].reshape(shape).to(CACHE_DTYPE).contiguous(),
            layer_id_override=self.layer,
        )
        out = flash_attn_with_kvcache(
            query[0].reshape(shape).to(CACHE_DTYPE),
            call.pool.get_key_buffer(self.layer).view(-1, 1, call.heads, call.head_dim),
            call.pool.get_value_buffer(self.layer).view(
                -1, 1, call.heads, call.head_dim
            ),
            cache_seqlens=call.cache_seqlens,
            page_table=call.page_table,
            cu_seqlens_q=call.cu_seqlens_q,
            max_seqlen_q=call.max_seqlen_q,
            causal=False,
        )
        self.layer += 1
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(1, -1, call.heads * call.head_dim).to(query.dtype)


class CachedDiT(PackedDiT):
    """PackedDiT over the new frames only. Same modules in the same order;
    attention reads the pool, the conv position embedding starts from the
    previous hop's tails and RoPE uses absolute frame positions."""

    def __init__(self, dit: torch.nn.Module, call: CachedCall) -> None:
        super().__init__(dit)
        self.call = call
        self.step = 0

    def row_attention(self, rows, *, streaming: bool) -> CachedRowAttention:
        return CachedRowAttention(self.call)

    def _rope(self, rows):
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(self.call.max_frames)
        freqs = freqs[:, self.call.positions]
        if isinstance(scale, torch.Tensor):
            scale = scale[:, self.call.positions]
        return freqs, scale

    def _conv_pos_embed(self, h: torch.Tensor, rows) -> torch.Tensor:
        conv = self.dit.input_embed.conv_pos_embed
        context = conv.kernel_size - 1
        padded = scatter_rows(h, rows, rows.width)
        tail_h, tail_conv1 = self.call.conv_tails(self.step, padded, context)
        conv1_in = torch.cat((tail_h, padded), dim=1)
        conv1_out = conv.conv1(conv1_in.transpose(1, 2)).transpose(1, 2)
        conv2_in = torch.cat((tail_conv1, conv1_out), dim=1)
        conv2_out = conv.conv2(conv2_in.transpose(1, 2)).transpose(1, 2)
        self.call.store_conv_tails(self.step, conv1_in, conv2_in, context)
        self.step += 1
        return gather_rows(conv2_out, rows)


class FlowKVCache:
    """The SGLang pool and allocator behind the cached hops, and the per stream
    state the next hop reads."""

    def __init__(
        self,
        *,
        slots: int,
        layers: int,
        heads: int,
        head_dim: int,
        chunk: int,
        device: str,
    ) -> None:
        self.pool = MHATokenToKVPool(
            size=slots,
            page_size=1,
            dtype=CACHE_DTYPE,
            head_num=heads,
            head_dim=head_dim,
            layer_num=layers,
            device=device,
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            slots, CACHE_DTYPE, device, self.pool, need_sort=False
        )
        self.heads = heads
        self.head_dim = head_dim
        self.chunk = chunk
        self.device = device
        self.streams: dict[int, StreamCache] = {}
        self.bytes_per_frame = 2 * layers * heads * head_dim * CACHE_DTYPE.itemsize

    def begin_call(self, rows: list[tuple[int, int, int]]) -> CachedCall:
        """rows: (stream index, first new frame, end frame) in row order. The
        CFG twins of a row are separate lanes with their own slots."""
        entries: list[tuple[StreamCache, int]] = []
        lengths: list[int] = []
        new_slots: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        segment_entry: list[int] = []
        cache_seqlens: list[int] = []
        cu_seqlens_q: list[int] = [0]
        max_frames = max(end for _, _, end in rows)
        caches: list[StreamCache] = []
        for index, start, _ in rows:
            cache = self.streams.get(index)
            if cache is None:
                cache = StreamCache(
                    slots=[
                        torch.empty(0, dtype=torch.int64, device=self.device)
                        for _ in range(2)
                    ]
                )
                self.streams[index] = cache
            if cache.frames != start:
                raise RuntimeError(
                    f"stream {index} holds {cache.frames} cached frames, "
                    f"the hop starts at {start}"
                )
            caches.append(cache)
        for lane in (0, 1):
            for row, (_, start, end) in enumerate(rows):
                cache = caches[row]
                fresh = self.allocator.alloc(end - start)
                if fresh is None:
                    raise RuntimeError(
                        f"Flow K/V pool exhausted: {self.allocator.available_size()} "
                        f"slots free, {end - start} needed"
                    )
                cache.slots[lane] = torch.cat((cache.slots[lane], fresh))
                entries.append((cache, lane))
                lengths.append(end - start)
                new_slots.append(fresh)
                positions.append(
                    torch.arange(start, end, device=self.device, dtype=torch.int64)
                )
                frame = start
                while frame < end:
                    segment_end = min((frame // self.chunk + 1) * self.chunk, end)
                    segment_entry.append(len(entries) - 1)
                    cache_seqlens.append(segment_end)
                    cu_seqlens_q.append(cu_seqlens_q[-1] + segment_end - frame)
                    frame = segment_end
        for cache, (_, _, end) in zip(caches, rows, strict=True):
            cache.frames = end
        entry_table = torch.zeros(
            len(entries), max_frames, dtype=torch.int32, device=self.device
        )
        for row, (cache, lane) in enumerate(entries):
            slots = cache.slots[lane]
            entry_table[row, : slots.numel()] = slots.to(torch.int32)
        segments = torch.tensor(segment_entry, dtype=torch.int64, device=self.device)
        as_int32 = {"dtype": torch.int32, "device": self.device}
        return CachedCall(
            pool=self.pool,
            heads=self.heads,
            head_dim=self.head_dim,
            entries=entries,
            lengths_tensor=torch.tensor(lengths, dtype=torch.int64, device=self.device),
            slots=torch.cat(new_slots),
            positions=torch.cat(positions),
            max_frames=max_frames,
            page_table=entry_table[segments],
            cache_seqlens=torch.tensor(cache_seqlens, **as_int32),
            cu_seqlens_q=torch.tensor(cu_seqlens_q, **as_int32),
            max_seqlen_q=max(
                cu_seqlens_q[index + 1] - cu_seqlens_q[index]
                for index in range(len(cache_seqlens))
            ),
        )

    def used_slots(self) -> int:
        return sum(cache.frames * 2 for cache in self.streams.values())

    def warm_kernels(self) -> None:
        """Pay the store_cache JIT build and the FA3 first call before the
        measured hops, on a slot that is returned straight after."""
        slot = self.allocator.alloc(1)
        zeros = torch.zeros(
            1, self.heads, self.head_dim, dtype=CACHE_DTYPE, device=self.device
        )
        ones = torch.ones(1, dtype=torch.int32, device=self.device)
        self.pool.set_kv_buffer(None, slot, zeros, zeros, layer_id_override=0)
        flash_attn_with_kvcache(
            zeros,
            self.pool.get_key_buffer(0).view(-1, 1, self.heads, self.head_dim),
            self.pool.get_value_buffer(0).view(-1, 1, self.heads, self.head_dim),
            cache_seqlens=ones,
            page_table=slot.to(torch.int32).view(1, 1),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            max_seqlen_q=1,
            causal=False,
        )
        torch.cuda.synchronize()
        self.allocator.free(slot)


@dataclass
class Row:
    """One stream of the schedule, with the prompt the serving path latches."""

    stream: Stream
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    start_step: int
    hift: dict[str, tuple[torch.Tensor | None, int]] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_token.shape[1])

    def flow_input(self, offset: int, hop: int) -> FlowBatchInput:
        window = offset + hop + PRE_LOOKAHEAD_LEN
        return FlowBatchInput(
            token=self.stream.tokens[:, :window],
            prompt_token=self.prompt_token,
            prompt_feat=self.prompt_feat,
            embedding=self.stream.embedding,
        )


def plan_schedule(
    rows: list[Row], steps: int
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    """The hops each step runs, as (row, token offset, hop length), and each
    row's final mel frame count. Rows join staggered and then grow their hop
    the way `next_stream_hop_len` does, so a step mixes hop sizes."""
    plan: list[list[tuple[int, int, int]]] = []
    state = [(0, TOKEN_HOP_LEN) for _ in rows]
    frames = [0 for _ in rows]
    for step in range(steps):
        participants: list[tuple[int, int, int]] = []
        for index, row in enumerate(rows):
            offset, hop = state[index]
            if step < row.start_step:
                continue
            if offset + hop + PRE_LOOKAHEAD_LEN > int(row.stream.tokens.shape[1]):
                continue
            participants.append((index, offset, hop))
            frames[index] = (row.prompt_len + offset + hop) * TOKEN_MEL_RATIO
            state[index] = (offset + hop, next_stream_hop_len(hop))
        plan.append(participants)
    return plan, frames


def autocast(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=CACHE_DTYPE, enabled=enabled)


def cached_hop(
    flow,
    items: list[FlowBatchInput],
    spans: list[tuple[int, int, int]],
    cache: FlowKVCache,
    repeats: int,
) -> tuple[list[torch.Tensor], float, float]:
    """The hop over each row's new frames only. `spans` is (stream index,
    first new frame, end frame) per row. Returns the mels, the median solve
    wall and the wall the host spent laying out slots and page table."""
    packed = pack_flow_inputs(flow.flow, items)
    with autocast(True):
        conditioning = prepare_flow_conditioning(flow, packed, finalize=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        call = cache.begin_call(spans)
        torch.cuda.synchronize()
        metadata_ms = (time.perf_counter() - started) * 1e3
        for row, (_, start, end) in enumerate(spans):
            if conditioning.mel_lengths[row] != end:
                raise RuntimeError(
                    f"row {row} conditioning holds {conditioning.mel_lengths[row]} "
                    f"frames, the hop ends at {end}"
                )

        def slice_rows(padded: torch.Tensor) -> torch.Tensor:
            parts = [
                padded[row, :, start:end].transpose(0, 1)
                for row, (_, start, end) in enumerate(spans)
            ]
            return torch.cat(parts, dim=0).unsqueeze(0)

        rows = pack_rows([end - start for _, start, end in spans], packed.token.device)
        estimator = CachedDiT(flow.decoder.estimator, call)
        # note(ratish): a repeat rewrites the same slots with the same K/V, so
        # only the conv tails have to be rolled back between them.
        snapshot = {id(entry): dict(entry.tails) for entry, _ in call.entries}
        samples = []
        for _ in range(repeats):
            for entry, _ in call.entries:
                entry.tails = dict(snapshot[id(entry)])
            torch.cuda.synchronize()
            started = time.perf_counter()
            estimator.step = 0
            generated = solve_flow_euler_packed(
                estimator,
                slice_rows(conditioning.noisy_mel),
                conditioning.time_span,
                slice_rows(conditioning.token_condition),
                conditioning.speaker_embedding,
                slice_rows(conditioning.prompt_mel),
                rows,
                cfg_rate=flow.decoder.inference_cfg_rate,
                streaming=True,
            )
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1e3)
    mels = []
    cursor = 0
    for length in rows.lengths:
        mels.append(generated[0, cursor : cursor + length].transpose(0, 1).unsqueeze(0))
        cursor += length
    return mels, statistics.median(samples), metadata_ms


def timed(call, repeats: int) -> tuple[list[torch.Tensor], float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return result, statistics.median(samples)


def conditioning_receptive_field(flow, item: FlowBatchInput) -> dict:
    """Which mel frames of a hop's conditioning one generated token reaches,
    so the cached hop knows how much token context its new frames need."""
    packed = pack_flow_inputs(flow.flow, [item])
    base = prepare_flow_conditioning(flow, packed, finalize=False).token_condition
    prompt = int(item.prompt_token.shape[1])
    position = int(item.token.shape[1]) // 2
    token = item.token.clone()
    token[0, position] = (int(token[0, position]) + 1) % int(
        flow.input_embedding.num_embeddings
    )
    moved = prepare_flow_conditioning(
        flow,
        pack_flow_inputs(flow.flow, [replace(item, token=token)]),
        finalize=False,
    ).token_condition
    frames = (moved != base).any(dim=1)[0].nonzero().flatten()
    first, last = int(frames.min()), int(frames.max())
    index = prompt + position
    return {
        "token_index": index,
        "first_frame": first,
        "last_frame": last,
        "left_context_tokens": index - first // TOKEN_MEL_RATIO,
        "right_context_tokens": last // TOKEN_MEL_RATIO - index,
    }


def spectrum(waveform: torch.Tensor) -> torch.Tensor:
    """Magnitude STFT of one emitted delta. HiFT drives its excitation from a
    phase that is a cumulative sum of the predicted F0, so a bfloat16 level mel
    difference drifts the phase and the raw waveform decorrelates while sounding
    the same; the 2026-09-16 run measures today's own path at -1.6 dB on the raw
    waveform and 36.4 dB on the mel. Magnitude is what survives that drift."""
    return torch.stft(
        waveform.reshape(-1),
        STFT_SIZE,
        hop_length=STFT_HOP,
        window=torch.hann_window(STFT_SIZE),
        return_complex=True,
    ).abs()


def measure(value: torch.Tensor, truth: torch.Tensor) -> dict:
    if tuple(value.shape) != tuple(truth.shape):
        return {
            "snr_db": float("nan"),
            "max_abs": float("nan"),
            "finite": bool(torch.isfinite(value).all()),
            "shape": list(value.shape),
            "truth_shape": list(truth.shape),
        }
    result = compare(value.float().cpu(), truth.float().cpu())
    result.pop("equal")
    result["finite"] = bool(torch.isfinite(value).all())
    result["shape"] = list(value.shape)
    result["truth_shape"] = list(truth.shape)
    return result


def summarize(records: list[dict], key: str) -> dict[str, dict]:
    summary = {}
    for path in ("production", "cached"):
        values = [record[key][path]["snr_db"] for record in records]
        summary[path] = {
            "calls": len(values),
            "min_snr_db": min(values),
            "median_snr_db": statistics.median(values),
            "finite": all(record[key][path]["finite"] for record in records),
            "shapes_match": all(
                record[key][path]["shape"] == record[key][path]["truth_shape"]
                for record in records
            ),
        }
    return summary


def verdict(summary: dict[str, dict], margin: float) -> dict:
    cached, production = summary["cached"], summary["production"]
    return {
        "min_margin_db": cached["min_snr_db"] - production["min_snr_db"],
        "median_margin_db": cached["median_snr_db"] - production["median_snr_db"],
        "pass": bool(
            cached["min_snr_db"] >= production["min_snr_db"] - margin
            and cached["median_snr_db"] >= production["median_snr_db"] - margin
            and cached["finite"]
            and cached["shapes_match"]
        ),
    }


def write_report(destination: str, report: dict) -> None:
    provenance = report["provenance"]
    kernel = next(
        (
            provenance[name]
            for name in ("sgl-kernel", "sglang-kernel")
            if provenance.get(name, "absent") != "absent"
        ),
        "absent",
    )
    lines = [
        "# G0: cached hop against the full window recompute",
        "",
        f"head {provenance['head']}, {provenance['device']}, "
        f"sglang {provenance['sglang']}, kernel {kernel}",
        "",
        f"streams {report['streams']}, steps {report['steps']}, "
        f"gate margin {report['gate_margin_db']} dB, repeats {report['repeats']}",
        "",
        "## 1. Every hop against the float32 truth",
        "",
        "Mel and spectrum are gated. The raw waveform is a diagnostic: HiFT's "
        "excitation phase is a cumulative sum of the predicted F0, so it "
        "decorrelates for the shipped path too.",
        "",
        "| step | rows | row | hop | new frames | window frames | mel prod dB | "
        "mel cached dB | spec prod dB | spec cached dB | wav prod dB | "
        "wav cached dB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in report["hops"]:
        cells = "".join(
            f"{record[name][path]['snr_db']:.1f} | "
            for name in METRICS
            for path in ("production", "cached")
        )
        lines.append(
            f"| {record['step']} | {record['rows']} | {record['sample_id']} | "
            f"{record['hop']} | {record['new_frames']} | {record['window_frames']} | "
            f"{cells.rstrip()}"
        )
    titles = {
        "mel": "Emitted mel",
        "spectrum": "Emitted spectrum magnitude",
        "waveform": "Raw waveform, diagnostic only",
    }
    for section, name in enumerate(METRICS, start=2):
        summary = report["summary"][name]
        lines += [
            "",
            f"## {section}. {titles[name]}",
            "",
            "| path | calls | min dB | median dB | finite | shapes match |",
            "|---|---|---|---|---|---|",
        ]
        for path in ("production", "cached"):
            cell = summary[path]
            lines.append(
                f"| {path} | {cell['calls']} | {cell['min_snr_db']:.1f} | "
                f"{cell['median_snr_db']:.1f} | {cell['finite']} | "
                f"{cell['shapes_match']} |"
            )
        gate = report["verdict"].get(name)
        if gate is None:
            continue
        lines += [
            "",
            f"cached minus production: {gate['min_margin_db']:+.2f} dB at the "
            f"minimum, {gate['median_margin_db']:+.2f} dB at the median; "
            f"gate {'pass' if gate['pass'] else 'FAIL'}",
        ]
    lines += [
        "",
        "## 5. Cost per hop call",
        "",
        "Synchronized median of the repeats, one step per row. The cached solve "
        "excludes the host layout, which is timed beside it; the float32 truth "
        "is not timed.",
        "",
        "| step | rows | new frames | window frames | production ms | cached ms | "
        "cached layout ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for step in report["steps_detail"]:
        lines.append(
            f"| {step['step']} | {step['rows']} | {step['new_frames']} | "
            f"{step['window_frames']} | {step['production_ms']:.1f} | "
            f"{step['cached_ms']:.1f} | {step['cached_metadata_ms']:.1f} |"
        )
    field_info = report["receptive_field"]
    pool = report["pool"]
    lines += [
        "",
        "## 6. Cache layout",
        "",
        f"- bytes per cached frame: {pool['bytes_per_frame']} "
        f"({pool['bytes_per_frame'] * 2} per mel frame with its CFG twin)",
        f"- slots allocated: {pool['slots']}, used {pool['used_slots']}, "
        f"{pool['used_gib']:.1f} GiB of K/V at the end of the schedule",
        f"- hops chunk aligned: {report['chunk_aligned']}, "
        f"chunk {report['chunk']} frames",
        f"- conditioning receptive field of one token: frames "
        f"{field_info['first_frame']} to {field_info['last_frame']}, "
        f"{field_info['left_context_tokens']} tokens left and "
        f"{field_info['right_context_tokens']} right",
        "",
    ]
    with open(destination, "w") as out:
        out.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--stagger", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gate-margin-db", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    info = provenance(args.device)
    info["sgl_kernel_flash_attn"] = inspect.getsourcefile(flash_attn_with_kvcache)
    checkpoint, vocoder = load_vocoder(args.model, args.device, CACHE_DTYPE)
    info["sglang_omni"] = inspect.getsourcefile(type(vocoder))
    for key, value in info.items():
        print(f"{key} {value}")
    flow = vocoder.flow

    prompts = tuple(
        PROMPT_TOKENS[index % len(PROMPT_TOKENS)] for index in range(args.streams)
    )
    hop, tokens = TOKEN_HOP_LEN, 0
    for _ in range(args.steps):
        tokens += hop
        hop = next_stream_hop_len(hop)
    needed = tokens + PRE_LOOKAHEAD_LEN
    references = build_streams(
        checkpoint,
        args.device,
        count=4 * args.streams,
        prompt_tokens=prompts * 4,
        min_generated=TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN,
        samples=args.samples,
    )
    streams = extend_tokens(references, args.streams, needed)
    rows = []
    for index, stream in enumerate(streams):
        prompt_token, prompt_feat = pad_flow_prompt_to_hop(
            stream.prompt_token, stream.prompt_feat, hop_len=TOKEN_HOP_LEN
        )
        rows.append(
            Row(
                stream=stream,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                start_step=index % args.stagger,
            )
        )
    plan, frames = plan_schedule(rows, args.steps)
    for index, row in enumerate(rows):
        print(
            f"row {index} {row.stream.sample_id} prompt "
            f"{int(row.stream.prompt_token.shape[1])} padded to {row.prompt_len} "
            f"tokens, joins at step {row.start_step}, {frames[index]} mel frames"
        )

    dit = flow.decoder.estimator
    attention = dit.transformer_blocks[0].attn
    heads = int(attention.heads)
    head_dim = int(attention.to_q.out_features) // heads
    chunk = flow.packed_estimator.chunk_size
    with torch.inference_mode():
        probe = prepare_flow_conditioning(
            flow,
            pack_flow_inputs(flow.flow, [rows[0].flow_input(0, TOKEN_HOP_LEN)]),
            finalize=False,
        )
    layers = (len(probe.time_span) - 1) * len(dit.transformer_blocks)
    cache = FlowKVCache(
        slots=2 * sum(frames),
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        chunk=chunk,
        device=args.device,
    )
    print(
        f"{layers} cache layers, {cache.bytes_per_frame} bytes per frame, "
        f"{2 * sum(frames)} slots, "
        f"{2 * sum(frames) * cache.bytes_per_frame / 2**30:.1f} GiB"
    )

    records: list[dict] = []
    steps_detail: list[dict] = []
    chunk_aligned = True
    with torch.inference_mode():
        receptive_field = conditioning_receptive_field(
            flow, rows[0].flow_input(0, TOKEN_HOP_LEN)
        )
        cache.warm_kernels()
        for step, participants in enumerate(plan):
            if not participants:
                continue
            items = [
                rows[index].flow_input(offset, hop)
                for index, offset, hop in participants
            ]
            spans = []
            for index, offset, hop in participants:
                emit_from = (rows[index].prompt_len + offset) * TOKEN_MEL_RATIO
                end = (rows[index].prompt_len + offset + hop) * TOKEN_MEL_RATIO
                held = cache.streams[index].frames if index in cache.streams else 0
                spans.append((index, held, end))
                # note(ratish): the cached boundary has to fall on a chunk edge
                # or a new frame would attend keys the cache never wrote.
                chunk_aligned = (
                    chunk_aligned and emit_from % chunk == 0 and held % chunk == 0
                )
            with autocast(False):
                truth = flow.inference_causal(items)
            production, production_ms = timed(
                lambda: vocoder.hop_batch(items), args.repeats
            )
            cached, cached_ms, metadata_ms = cached_hop(
                flow, items, spans, cache, args.repeats
            )
            for row, (index, offset, hop) in enumerate(participants):
                emit_from = (rows[index].prompt_len + offset) * TOKEN_MEL_RATIO
                _, start, end = spans[row]
                mels = {
                    "truth": truth[row][:, :, offset * TOKEN_MEL_RATIO :],
                    "production": production[row][:, :, offset * TOKEN_MEL_RATIO :],
                    "cached": cached[row][:, :, emit_from - start :],
                }
                waves = {}
                for path, mel in mels.items():
                    history, samples = rows[index].hift.get(path, (None, 0))
                    delta, history, samples = vocoder.hift_delta(
                        mel, hift_mel=history, speech_offset=samples, finalize=False
                    )
                    rows[index].hift[path] = (history, samples)
                    waves[path] = delta
                spectra = {name: spectrum(wave) for name, wave in waves.items()}
                values = {"mel": mels, "spectrum": spectra, "waveform": waves}
                records.append(
                    {
                        "step": step,
                        "rows": len(participants),
                        "row": index,
                        "sample_id": rows[index].stream.sample_id,
                        "hop": hop,
                        "token_offset": offset,
                        "new_frames": end - start,
                        "window_frames": end,
                        **{
                            name: {
                                path: measure(taken[path], taken["truth"])
                                for path in ("production", "cached")
                            }
                            for name, taken in values.items()
                        },
                    }
                )
            steps_detail.append(
                {
                    "step": step,
                    "rows": len(participants),
                    "new_frames": sum(end - start for _, start, end in spans),
                    "window_frames": sum(end for _, _, end in spans),
                    "production_ms": production_ms,
                    "cached_ms": cached_ms,
                    "cached_metadata_ms": metadata_ms,
                }
            )
            print(
                f"step {step}: {len(participants)} rows, "
                f"{steps_detail[-1]['new_frames']} new of "
                f"{steps_detail[-1]['window_frames']} frames, "
                f"production {production_ms:.1f} ms, cached {cached_ms:.1f} ms"
            )

    summary = {name: summarize(records, name) for name in METRICS}
    report = {
        "provenance": info,
        "streams": args.streams,
        "steps": args.steps,
        "stagger": args.stagger,
        "repeats": args.repeats,
        "gate_margin_db": args.gate_margin_db,
        "chunk": chunk,
        "chunk_aligned": chunk_aligned,
        "receptive_field": receptive_field,
        "pool": {
            "layers": layers,
            "bytes_per_frame": cache.bytes_per_frame,
            "slots": 2 * sum(frames),
            "used_slots": cache.used_slots(),
            "used_gib": cache.used_slots() * cache.bytes_per_frame / 2**30,
        },
        "rows": [
            {
                "row": index,
                "sample_id": row.stream.sample_id,
                "prompt_tokens": int(row.stream.prompt_token.shape[1]),
                "padded_prompt_tokens": row.prompt_len,
                "start_step": row.start_step,
                "final_frames": frames[index],
            }
            for index, row in enumerate(rows)
        ],
        "steps_detail": steps_detail,
        "hops": records,
        "summary": summary,
        "verdict": {
            name: verdict(summary[name], args.gate_margin_db) for name in GATED
        },
    }
    with open(os.path.join(args.out, "g0.json"), "w") as out:
        json.dump(report, out, indent=1)
    write_report(os.path.join(args.out, "g0.md"), report)
    for name in GATED:
        gate = report["verdict"][name]
        print(
            f"{name}: cached {gate['min_margin_db']:+.2f} dB at the minimum, "
            f"{gate['median_margin_db']:+.2f} dB at the median, "
            f"{'pass' if gate['pass'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
