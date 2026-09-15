#!/usr/bin/env python3
"""E6: which attention kernel can run the packed DiT, on the model's own
activations, against a float32 truth.

Run on the H100 venv from the branch worktree, alone on the GPU, no server:

  python tasks/cosyvoice_utilization_20260912/stage0/e6_attention_kernels.py \
      --device cuda:0 --json e6.json --save-dir e6_worst

A packed hop batch and a packed final batch run through the serving path in
bfloat16 autocast. Every RowAttention call (10 Euler steps by 22 blocks per
solve) is intercepted after it runs; its query, key and value are fed to:

  truth                 SDPA math backend in float32 on the padded rows
  sdpa_bf16_production  the output the serving call produced
  sdpa_bf16_efficient   SDPA mem efficient backend, padded rows and mask
  sdpa_bf16_cudnn       SDPA cuDNN backend, padded rows and mask
  fa3_varlen_bf16       SGLang FA3 varlen, one sequence per row (finals only)
  fa3_paged_bf16        SGLang FA3 with a page table: one query segment per
                        (row, chunk), cache_seqlens at the chunk end (hops),
                        one segment per row at the row length (finals)
  flex_bf16             FlexAttention, compiled, row and chunk block mask

Per call it reports the SNR against the truth for the whole call and for the
worst row, the raw q.k range over the visible keys, and the share of query rows
whose largest visible raw logit is below minus 5e4. At two (step, block) points
per solve it times every candidate. The worst calls' tensors are saved.
"""

from __future__ import annotations

import argparse
import heapq
import inspect
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_ID,
    build_streams,
    flow_input,
    hop_window,
    load_vocoder,
    provenance,
)

from sglang_omni.models.fun_cosyvoice3 import packed_dit  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.streaming import TOKEN_HOP_LEN  # noqa: E402

PROMPT_TOKENS = 2 * TOKEN_HOP_LEN
HOPS = ((0, TOKEN_HOP_LEN), (TOKEN_HOP_LEN, 2 * TOKEN_HOP_LEN))
SENTINEL = -5e4
BENCH_POINTS = {(0, 0), (5, 11)}
BENCH_REPEATS = 5
ORIGINAL_CALL = packed_dit.RowAttention.__call__


def fa3_provenance() -> dict[str, str]:
    import sgl_kernel.flash_attn

    return {"sgl_kernel.flash_attn": inspect.getsourcefile(sgl_kernel.flash_attn)}


def production(attention, query, key, value):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return ORIGINAL_CALL(attention, query, key, value)


def padded_sdpa(attention, query, key, value, backend, dtype):
    with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel([backend]):
        return ORIGINAL_CALL(
            attention, query.to(dtype), key.to(dtype), value.to(dtype)
        ).float()


def fa3_varlen(rows, query, key, value, heads):
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    shape = (-1, heads, query.shape[-1] // heads)
    cu_seqlens = rows.starts_host.to(device=query.device, dtype=torch.int32)
    out = flash_attn_varlen_func(
        query[0].reshape(shape).to(torch.bfloat16),
        key[0].reshape(shape).to(torch.bfloat16),
        value[0].reshape(shape).to(torch.bfloat16),
        cu_seqlens,
        cu_seqlens,
        max_seqlen_q=rows.width,
        max_seqlen_k=rows.width,
        causal=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(1, -1, query.shape[-1]).float()


def paged_layout(rows, chunk, device):
    starts = rows.starts_host.tolist()
    cu_seqlens_q, cache_seqlens, page_table = [0], [], []
    for start, length in zip(starts, rows.lengths):
        pages = list(range(start, start + length)) + [0] * (rows.width - length)
        step = chunk if chunk is not None else length
        for begin in range(0, length, step):
            end = min(begin + step, length)
            cu_seqlens_q.append(cu_seqlens_q[-1] + end - begin)
            cache_seqlens.append(end)
            page_table.append(pages)

    def as_tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return (
        as_tensor(cu_seqlens_q),
        as_tensor(cache_seqlens),
        as_tensor(page_table),
        chunk if chunk is not None else rows.width,
    )


def fa3_paged(layout, query, key, value, heads):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    cu_seqlens_q, cache_seqlens, page_table, max_seqlen_q = layout
    head_dim = query.shape[-1] // heads
    out = flash_attn_with_kvcache(
        query[0].reshape(-1, heads, head_dim).to(torch.bfloat16),
        key[0].reshape(-1, 1, heads, head_dim).to(torch.bfloat16).contiguous(),
        value[0].reshape(-1, 1, heads, head_dim).to(torch.bfloat16).contiguous(),
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        causal=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(1, -1, query.shape[-1]).float()


class FlexCandidate:
    def __init__(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        self.create_block_mask = create_block_mask
        self.flex = torch.compile(flex_attention, dynamic=False)

    def block_mask(self, rows, chunk, device):
        row_ids, positions = rows.row_ids, rows.positions
        total = rows.total

        def mask_mod(batch, head, q_index, kv_index):
            same = row_ids[q_index] == row_ids[kv_index]
            if chunk is None:
                return same
            return same & (
                positions[kv_index] < (positions[q_index] // chunk + 1) * chunk
            )

        return self.create_block_mask(mask_mod, None, None, total, total, device=device)

    def __call__(self, block_mask, query, key, value, heads):
        total, width = query.shape[1], query.shape[-1]

        def split(tensor):
            return (
                tensor.reshape(1, total, heads, width // heads)
                .transpose(1, 2)
                .to(torch.bfloat16)
            )

        out = self.flex(split(query), split(key), split(value), block_mask=block_mask)
        return out.transpose(1, 2).reshape(1, total, width).float()


def logit_stats(rows, chunk, heads, query, key):
    low, high, below, counted = math.inf, -math.inf, 0, 0
    for start, length in zip(rows.starts_host.tolist(), rows.lengths):
        q = query[0, start : start + length].float().reshape(length, heads, -1)
        k = key[0, start : start + length].float().reshape(length, heads, -1)
        scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(1, 2)
        if chunk is not None:
            visible = packed_dit.chunk_causal_mask(length, chunk, query.device)
            row_max = scores.masked_fill(~visible, -math.inf).amax(dim=-1)
            row_min = scores.masked_fill(~visible, math.inf).amin(dim=-1)
        else:
            row_max, row_min = scores.amax(dim=-1), scores.amin(dim=-1)
        low = min(low, float(row_min.min()))
        high = max(high, float(row_max.max()))
        below += int((row_max < SENTINEL).sum())
        counted += row_max.numel()
    return {
        "raw_logit_min": low,
        "raw_logit_max": high,
        "query_rows_below_minus_5e4": below / counted,
    }


def snr(value, truth):
    diff = float((value.double() - truth.double()).norm())
    if diff == 0.0:
        return math.inf
    return 20 * math.log10(float(truth.double().norm()) / diff)


def worst_row_snr(value, truth, rows):
    return min(
        snr(value[:, start : start + length], truth[:, start : start + length])
        for start, length in zip(rows.starts_host.tolist(), rows.lengths)
    )


def timed_ms(call):
    call()
    samples = []
    for _ in range(BENCH_REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


class Recorder:
    def __init__(self, depth, save_count):
        self.depth = depth
        self.kind = None
        self.chunk = None
        self.counters = defaultdict(int)
        self.layouts = {}
        self.records = []
        self.errors = {}
        self.worst = []
        self.save_count = save_count
        try:
            self.flex = FlexCandidate()
        except Exception as exc:
            self.flex = None
            self.errors["flex_bf16"] = f"{type(exc).__name__}: {exc}"

    def begin(self, kind, chunk):
        self.kind, self.chunk = kind, chunk
        self.counters.clear()
        self.layouts.clear()

    def candidates(self, attention, query, key, value):
        rows, heads, chunk = attention.rows, attention.heads, self.chunk
        solve = id(attention)
        if solve not in self.layouts:
            block_mask = None
            if self.flex is not None and "flex_bf16" not in self.errors:
                try:
                    block_mask = self.flex.block_mask(rows, chunk, query.device)
                except Exception as exc:
                    self.errors["flex_bf16"] = f"{type(exc).__name__}: {exc}"
            self.layouts[solve] = (paged_layout(rows, chunk, query.device), block_mask)
        layout, block_mask = self.layouts[solve]
        calls = {
            "sdpa_bf16_efficient": lambda: padded_sdpa(
                attention,
                query,
                key,
                value,
                SDPBackend.EFFICIENT_ATTENTION,
                torch.bfloat16,
            ),
            "sdpa_bf16_cudnn": lambda: padded_sdpa(
                attention, query, key, value, SDPBackend.CUDNN_ATTENTION, torch.bfloat16
            ),
            "fa3_paged_bf16": lambda: fa3_paged(layout, query, key, value, heads),
        }
        if chunk is None:
            calls["fa3_varlen_bf16"] = lambda: fa3_varlen(
                rows, query, key, value, heads
            )
        if block_mask is not None:
            calls["flex_bf16"] = lambda: self.flex(block_mask, query, key, value, heads)
        return {name: call for name, call in calls.items() if name not in self.errors}

    def observe(self, attention, query, key, value, output):
        solve = id(attention)
        index = self.counters[solve]
        self.counters[solve] += 1
        step, block = divmod(index, self.depth)
        rows = attention.rows
        record = {
            "kind": self.kind,
            "step": step,
            "block": block,
            "rows": len(rows.lengths),
            "total": rows.total,
            "width": rows.width,
            "dtypes": [str(query.dtype), str(key.dtype), str(value.dtype)],
            **logit_stats(rows, self.chunk, attention.heads, query, key),
            "snr_db": {},
            "worst_row_snr_db": {},
            "bench_ms": {},
        }
        truth = padded_sdpa(
            attention, query, key, value, SDPBackend.MATH, torch.float32
        )
        outputs = {"sdpa_bf16_production": output.float()}
        candidate_calls = self.candidates(attention, query, key, value)
        for name, call in candidate_calls.items():
            try:
                outputs[name] = call()
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        for name, value_out in outputs.items():
            record["snr_db"][name] = snr(value_out, truth)
            record["worst_row_snr_db"][name] = worst_row_snr(value_out, truth, rows)
        if (step, block) in BENCH_POINTS:
            record["bench_ms"]["sdpa_bf16_production"] = timed_ms(
                lambda: production(attention, query, key, value)
            )
            for name, call in candidate_calls.items():
                if name not in self.errors:
                    record["bench_ms"][name] = timed_ms(call)
        self.records.append(record)
        challenger = min(
            (
                snr_value
                for name, snr_value in record["worst_row_snr_db"].items()
                if name.startswith(("fa3", "flex"))
            ),
            default=math.inf,
        )
        if len(self.worst) >= self.save_count and -challenger <= self.worst[0][0]:
            return
        entry = (
            -challenger,
            len(self.records),
            {
                "record": record,
                "query": query.detach().cpu(),
                "key": key.detach().cpu(),
                "value": value.detach().cpu(),
                "lengths": list(rows.lengths),
                "chunk": self.chunk,
            },
        )
        if len(self.worst) < self.save_count:
            heapq.heappush(self.worst, entry)
        else:
            heapq.heapreplace(self.worst, entry)


def summarize(records):
    table = defaultdict(lambda: {"snr": [], "worst_row": [], "bench": []})
    for record in records:
        for name, value in record["snr_db"].items():
            cell = table[(record["kind"], name)]
            cell["snr"].append((value, record["step"], record["block"]))
            cell["worst_row"].append(record["worst_row_snr_db"][name])
            if name in record["bench_ms"]:
                cell["bench"].append(record["bench_ms"][name])
    print(
        f"\n{'kind':<7}{'candidate':<22}{'calls':>6}{'min snr':>9}{'step,block':>12}"
        f"{'<40 dB':>8}{'worst row':>11}{'bench ms':>10}"
    )
    summary = []
    for (kind, name), cell in sorted(table.items()):
        low = min(cell["snr"])
        under = sum(1 for value, _, _ in cell["snr"] if value < 40)
        bench = statistics.median(cell["bench"]) if cell["bench"] else float("nan")
        where = f"{low[1]},{low[2]}"
        print(
            f"{kind:<7}{name:<22}{len(cell['snr']):>6}{low[0]:>9.1f}{where:>12}"
            f"{under:>8}{min(cell['worst_row']):>11.1f}{bench:>10.3f}"
        )
        summary.append(
            {
                "kind": kind,
                "candidate": name,
                "calls": len(cell["snr"]),
                "min_snr_db": low[0],
                "min_at": [low[1], low[2]],
                "calls_under_40_db": under,
                "worst_row_snr_db": min(cell["worst_row"]),
                "bench_ms_median": bench,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--save-count", type=int, default=4)
    parser.add_argument("--save-dir")
    parser.add_argument("--json")
    args = parser.parse_args()

    info = provenance(args.device)
    fa3 = fa3_provenance()
    for key, value in fa3.items():
        print(f"{key} {value}")
    checkpoint, vocoder = load_vocoder(args.model, args.device, torch.bfloat16)
    estimator = vocoder.flow.packed_estimator
    streams = build_streams(
        checkpoint,
        args.device,
        count=args.rows,
        prompt_tokens=PROMPT_TOKENS,
        min_generated=hop_window(*HOPS[-1]),
        samples=args.samples,
    )
    print(f"streams {[stream.sample_id for stream in streams]}")
    recorder = Recorder(len(estimator.dit.transformer_blocks), args.save_count)

    def patched(self, query, key, value):
        output = ORIGINAL_CALL(self, query, key, value)
        if recorder.kind is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                recorder.observe(self, query, key, value, output)
        return output

    packed_dit.RowAttention.__call__ = patched
    hop_items = [
        flow_input(stream, hop_window(*HOPS[index % len(HOPS)]))
        for index, stream in enumerate(streams)
    ]
    final_items = [flow_input(stream, stream.tokens.shape[1]) for stream in streams]
    with torch.inference_mode():
        vocoder.hop_batch(hop_items)
        vocoder.leftover_batch(final_items)
        recorder.begin("hop", estimator.chunk_size)
        vocoder.hop_batch(hop_items)
        recorder.begin("final", None)
        vocoder.leftover_batch(final_items)
        recorder.begin(None, None)

    for name, error in recorder.errors.items():
        print(f"unavailable {name}: {error}")
    low = min(record["raw_logit_min"] for record in recorder.records)
    high = max(record["raw_logit_max"] for record in recorder.records)
    below = max(record["query_rows_below_minus_5e4"] for record in recorder.records)
    print(
        f"raw q.k over visible keys: min {low:.3e}, max {high:.3e}; largest share of "
        f"query rows whose row max is below -5e4 in one call: {below:.3f}"
    )
    summary = summarize(recorder.records)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for rank, (_, _, payload) in enumerate(sorted(recorder.worst, reverse=True)):
            torch.save(payload, os.path.join(args.save_dir, f"worst_{rank}.pt"))
    if args.json:
        with open(args.json, "w") as out:
            json.dump(
                {
                    "provenance": info,
                    "fa3": fa3,
                    "errors": recorder.errors,
                    "summary": summary,
                    "calls": recorder.records,
                },
                out,
                indent=1,
            )


if __name__ == "__main__":
    main()
