#!/usr/bin/env python3
"""FA3 paged attention time for the row shapes of a Qwen3-Omni thinker step, one layer, on one card alone.

usage: python3 fa3_mixed_bench.py [--iters 200]

The thinker (32 query heads, 4 KV heads, head dim 128, bf16, page size 1) runs a decode step's attention with one query
token a row and a mixed step's attention through the extend call, where the admitted prompt's rows set max_seqlen_q
and every decode row rides as a one-token extend. Each case calls sgl_kernel's flash_attn_with_kvcache as SGLang's FA3
backend does for that mode (num_splits 0, FA3 sizes split-KV itself; a few cases force splits) and reports the mean time
of one call over --iters calls after warmup, from CUDA events. Times are one layer; the thinker has 48.
"""

import argparse

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 128


def build(rows: list[tuple[int, int]], device: torch.device):
    """rows: (query tokens, context tokens including them) per request."""
    total_kv = sum(context for _, context in rows)
    k_cache = torch.randn(
        total_kv, 1, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    max_context = max(context for _, context in rows)
    page_table = torch.zeros(len(rows), max_context, dtype=torch.int32, device=device)
    start = 0
    for index, (_, context) in enumerate(rows):
        page_table[index, :context] = torch.arange(
            start, start + context, dtype=torch.int32, device=device
        )
        start += context
    cache_seqlens = torch.tensor(
        [context for _, context in rows], dtype=torch.int32, device=device
    )
    query_lens = [query for query, _ in rows]
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn(
        sum(query_lens), Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max(query_lens)


def time_case(rows, num_splits: int, iters: int, device: torch.device) -> float:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q = build(
        rows, device
    )

    def call():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            num_splits=num_splits,
        )

    for _ in range(20):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    speech = (1, 400)
    long_row = (1, 14000)
    admitted = (100, 500)
    cases = [
        ("decode, 4 speech rows", [speech] * 4),
        ("decode, 3 speech rows and one 14k row", [speech] * 3 + [long_row]),
        ("decode, one 14k row", [long_row]),
        ("mixed, a 100-token admission and 4 speech rows", [admitted] + [speech] * 4),
        (
            "mixed, a 100-token admission, 3 speech rows and one 14k row",
            [admitted] + [speech] * 3 + [long_row],
        ),
        (
            "mixed, a 1-token cached admission, 3 speech rows and one 14k row",
            [(1, 500)] + [speech] * 3 + [long_row],
        ),
        (
            "mixed, a 100-token admission, 3 speech rows and two 14k rows",
            [admitted] + [speech] * 3 + [long_row] * 2,
        ),
    ]
    for name, rows in cases:
        print(
            f"{name}: {time_case(rows, 0, args.iters, device) * 1000:8.1f} us a layer (num_splits 0)"
        )
    for splits in (4, 16):
        rows = [admitted] + [speech] * 3 + [long_row]
        print(
            f"mixed, a 100-token admission, 3 speech rows and one 14k row: "
            f"{time_case(rows, splits, args.iters, device) * 1000:8.1f} us a layer (num_splits {splits})"
        )


if __name__ == "__main__":
    main()
