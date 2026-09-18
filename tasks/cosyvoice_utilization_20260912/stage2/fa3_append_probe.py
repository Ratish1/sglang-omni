#!/usr/bin/env python3
"""Can FA3 append a hop's new K and V into the paged pool itself.

The shipped cached hop writes K and V with one store call per layer and then
attends (220 extra launches and their Python per hop). `flash_attn_with_kvcache`
also takes `k`, `v` and `cu_seqlens_k_new` and appends them at `cache_seqlens`
before attending. A row of a hop is several query segments, one per chunk,
that share a page table row, and a later segment must read what an earlier one
appended in the same call, so this checks the two paths against each other on
that layout: attention output and the cache contents, bit for bit, then times
220 calls of each. No model, random tensors, the DiT's head shape.

  python fa3_append_probe.py --out .tmp/out/fa3-append
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache

HEADS, HEAD_DIM, CHUNK, LAYERS = 16, 64, 50, 220
# (first new frame, end frame) per lane row; every boundary on a chunk edge.
LAYOUTS = {
    "one_first_hop": [(0, 100)] * 2,
    "follow_up_rows": [(0, 150), (100, 300), (250, 300), (400, 600)] * 2,
    "sixteen_rows": [(50 * row, 50 * row + 200) for row in range(16)] * 2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None, help="unused, run_box passes it")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    report = {"device": torch.cuda.get_device_name(device), "layouts": {}}

    for name, spans in LAYOUTS.items():
        slots_needed = sum(end for _, end in spans)
        order = torch.randperm(slots_needed, device=device) + 1
        table = torch.zeros(
            len(spans), max(end for _, end in spans), dtype=torch.int32, device=device
        )
        cursor = 0
        for row, (_, end) in enumerate(spans):
            table[row, :end] = order[cursor : cursor + end].to(torch.int32)
            cursor += end
        cache_shape = (slots_needed + 1, 1, HEADS, HEAD_DIM)
        prefix_k = torch.randn(cache_shape, device=device, dtype=torch.bfloat16)
        prefix_v = torch.randn(cache_shape, device=device, dtype=torch.bfloat16)

        segment_rows, starts, ends, offsets = [], [], [], [0]
        for row, (start, end) in enumerate(spans):
            for frame in range(start, end, CHUNK):
                segment_rows.append(row)
                starts.append(frame)
                ends.append(frame + CHUNK)
                offsets.append(offsets[-1] + CHUNK)
        total = offsets[-1]
        as_int32 = {"dtype": torch.int32, "device": device}
        page_table = table[torch.tensor(segment_rows, device=device)]
        seqlens_before = torch.tensor(starts, **as_int32)
        seqlens_after = torch.tensor(ends, **as_int32)
        cu_seqlens = torch.tensor(offsets, **as_int32)
        new_slots = torch.cat(
            [table[row, start:end] for row, (start, end) in enumerate(spans)]
        ).to(torch.int64)
        query, key, value = (
            torch.randn(total, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )

        def stored(k_cache, v_cache):
            store_cache(
                key.view(total, -1),
                value.view(total, -1),
                k_cache.view(-1, HEADS * HEAD_DIM),
                v_cache.view(-1, HEADS * HEAD_DIM),
                new_slots,
            )
            return flash_attn_with_kvcache(
                q=query,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=seqlens_after,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=CHUNK,
                causal=False,
            )

        def appended(k_cache, v_cache):
            return flash_attn_with_kvcache(
                q=query,
                k_cache=k_cache,
                v_cache=v_cache,
                k=key,
                v=value,
                cache_seqlens=seqlens_before,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k_new=cu_seqlens,
                max_seqlen_q=CHUNK,
                causal=False,
            )

        result = {"rows": len(spans), "segments": len(starts), "new_frames": total}
        try:
            k_stored, v_stored = prefix_k.clone(), prefix_v.clone()
            k_appended, v_appended = prefix_k.clone(), prefix_v.clone()
            out_stored = stored(k_stored, v_stored)
            out_appended = appended(k_appended, v_appended)
            torch.cuda.synchronize()
            result["output_equal"] = bool(torch.equal(out_stored, out_appended))
            result["output_max_abs"] = float((out_stored - out_appended).abs().max())
            result["k_cache_equal"] = bool(torch.equal(k_stored, k_appended))
            result["v_cache_equal"] = bool(torch.equal(v_stored, v_appended))
            for label, call in (("stored", stored), ("appended", appended)):
                host, wall = [], []
                for _ in range(7):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(LAYERS):
                        call(k_stored, v_stored)
                    host.append((time.perf_counter() - started) * 1e3)
                    torch.cuda.synchronize()
                    wall.append((time.perf_counter() - started) * 1e3)
                result[f"{label}_host_ms"] = statistics.median(host)
                result[f"{label}_wall_ms"] = statistics.median(wall)
        except Exception as error:  # the report is the point of the run
            result["error"] = f"{type(error).__name__}: {error}"
        report["layouts"][name] = result
        print(name, json.dumps(result))

    with open(os.path.join(args.out, "fa3_append.json"), "w") as out:
        json.dump(report, out, indent=1)


if __name__ == "__main__":
    main()
