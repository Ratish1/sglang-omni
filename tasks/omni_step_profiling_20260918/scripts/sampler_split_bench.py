"""A split exact top-k for the predictor's seeded sampler, against the served kernel.

The served kernel (sampling_kernels.py, seeded_top_k_top_p_sample_kernel) runs one
program per row and selects the top block_k of 2048 packed keys (ordered score bits
over the complemented index) with one tl.topk. The keys are unique, so the top
block_k of the union of per-chunk top block_k sets is the same set in the same
order. Stage 1 runs a program per (row, chunk) over chunk-wide slices; stage 2 merges
the chunks' keys per row and runs the served kernel's code from the unpacked scores
on, with its own jit helpers. The two-level variant keeps one program per row and
selects per chunk inside it.

Checks, for top_k above 32 (block_k 64, the served Qwen3-TTS setting): the selected
keys and the sampled tokens against the served kernel on normal logits, heavy ties,
signed zeros and -inf; then the time of 15 back to back calls (one predictor replay)
in a CUDA graph at bs 1, 16 and 64.

usage: python sampler_split_bench.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang_omni.models.qwen3_tts import sampling_kernels as sk
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    fmix32,
    gumbel_from_hash,
    murmur3_mix,
)

VOCAB = 2048
CALLS = 15


@triton.jit
def pack_keys(scores, vocab_offsets):
    score_bits = scores.to(tl.uint32, bitcast=True)
    one = tl.full(vocab_offsets.shape, 1, tl.uint32)
    high = one << 31
    all_ones = high | (high - one)
    ordered = score_bits ^ tl.where((score_bits >> 31) != 0, all_ones, high)
    return (ordered.to(tl.uint64) << 32) | (all_ones - vocab_offsets.to(tl.uint32)).to(
        tl.uint64
    )


@triton.jit
def full_keys_kernel(logits, temperatures, keys_out, block_k: tl.constexpr):
    row = tl.program_id(0)
    vocab_offsets = tl.arange(0, 2048)
    scores = tl.load(logits + row * 2048 + vocab_offsets).to(tl.float32)
    temperature = tl.maximum(tl.load(temperatures + row).to(tl.float32), 1e-5)
    top = tl.topk(pack_keys(scores / temperature, vocab_offsets), k=block_k)
    tl.store(keys_out + row * block_k + tl.arange(0, block_k), top)


@triton.jit
def chunk_keys_kernel(
    logits,
    temperatures,
    keys_out,
    chunk: tl.constexpr,
    chunks: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    vocab_offsets = part * chunk + tl.arange(0, chunk)
    scores = tl.load(logits + row * 2048 + vocab_offsets).to(tl.float32)
    temperature = tl.maximum(tl.load(temperatures + row).to(tl.float32), 1e-5)
    top = tl.topk(pack_keys(scores / temperature, vocab_offsets), k=block_k)
    tl.store(keys_out + (row * chunks + part) * block_k + tl.arange(0, block_k), top)


@triton.jit
def sample_sorted(
    top_packed,
    row,
    top_ks,
    top_ps,
    seeds,
    positions,
    out,
    block_k: tl.constexpr,
    has_top_p: tl.constexpr,
):
    # The served kernel from its unpack on, for max_top_k above 32.
    ranks = tl.arange(0, block_k)
    one_rank = tl.full(ranks.shape, 1, tl.uint32)
    high_bit_rank = one_rank << 31
    all_ones_rank = high_bit_rank | (high_bit_rank - one_rank)
    ordered_top_score_bits = (top_packed >> 32).to(tl.uint32)
    top_score_bits = ordered_top_score_bits ^ tl.where(
        (ordered_top_score_bits >> 31) != 0,
        high_bit_rank,
        all_ones_rank,
    )
    sorted_scores = top_score_bits.to(tl.float32, bitcast=True)
    sorted_token_ids = all_ones_rank - (top_packed & all_ones_rank.to(tl.uint64)).to(
        tl.uint32
    )
    keep_top_k = ranks < tl.load(top_ks + row)
    masked_scores = tl.where(keep_top_k, sorted_scores, -float("inf"))
    max_score = tl.max(masked_scores, axis=0)
    probs = tl.exp(masked_scores - max_score)
    probs = probs / tl.sum(probs, axis=0)
    if has_top_p:
        top_p = tl.load(top_ps + row).to(tl.float32)
        active_top_p = (top_p > 0.0) & (top_p < 1.0)
        cdf = tl.cumsum(probs, axis=0)
        remove = (cdf - probs >= top_p) & active_top_p
        remove = remove & (ranks != 0)
        keep_top_k = keep_top_k & ~remove
    else:
        pass
    logprobs = tl.where(keep_top_k, tl.log(probs), -float("inf"))
    seed = tl.load(seeds + row).to(tl.uint64)
    pos = tl.load(positions + row).to(tl.uint32)
    col = ranks.to(tl.uint32)
    h: tl.uint32 = 0
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, pos)
    h = murmur3_mix(h, col)
    h ^= 16
    h = fmix32(h)
    gumbel = gumbel_from_hash(h)
    sampled_scores = logprobs.to(tl.float64) + gumbel
    max_sampled_score = tl.max(sampled_scores, axis=0)
    candidates = tl.where(sampled_scores == max_sampled_score, ranks, block_k)
    sampled_rank = tl.min(candidates, axis=0)
    token = tl.max(
        tl.where(ranks == sampled_rank, sorted_token_ids.to(tl.int64), 0), axis=0
    )
    tl.store(out + row, token)


@triton.jit
def merge_sample_kernel(
    keys,
    top_ks,
    top_ps,
    seeds,
    positions,
    out,
    keys_out,
    candidates: tl.constexpr,
    block_k: tl.constexpr,
    has_top_p: tl.constexpr,
    store_keys: tl.constexpr,
):
    row = tl.program_id(0)
    top_packed = tl.topk(
        tl.load(keys + row * candidates + tl.arange(0, candidates)).to(tl.uint64),
        k=block_k,
    )
    if store_keys:
        tl.store(keys_out + row * block_k + tl.arange(0, block_k), top_packed)
    else:
        pass
    sample_sorted(
        top_packed, row, top_ks, top_ps, seeds, positions, out, block_k, has_top_p
    )


@triton.jit
def two_level_kernel(
    logits,
    temperatures,
    top_ks,
    top_ps,
    seeds,
    positions,
    out,
    keys_out,
    chunks: tl.constexpr,
    block_k: tl.constexpr,
    has_top_p: tl.constexpr,
    store_keys: tl.constexpr,
):
    row = tl.program_id(0)
    vocab_offsets = tl.arange(0, 2048)
    scores = tl.load(logits + row * 2048 + vocab_offsets).to(tl.float32)
    temperature = tl.maximum(tl.load(temperatures + row).to(tl.float32), 1e-5)
    packed = pack_keys(scores / temperature, vocab_offsets)
    per_chunk = tl.topk(tl.reshape(packed, (chunks, 2048 // chunks)), k=block_k)
    top_packed = tl.topk(tl.reshape(per_chunk, (chunks * block_k,)), k=block_k)
    if store_keys:
        tl.store(keys_out + row * block_k + tl.arange(0, block_k), top_packed)
    else:
        pass
    sample_sorted(
        top_packed, row, top_ks, top_ps, seeds, positions, out, block_k, has_top_p
    )


def make_inputs(rows: int, kind: str, generator: torch.Generator):
    device = torch.device("cuda")
    if kind == "normal":
        logits = torch.randn(rows, VOCAB, generator=generator) * 3
    elif kind == "ties":
        logits = torch.randint(-3, 4, (rows, VOCAB), generator=generator).float()
    else:
        logits = torch.randn(rows, VOCAB, generator=generator)
        logits[:, ::7] = 0.0
        logits[:, 3::11] = -0.0
        logits[:, 5::13] = float("-inf")
    logits = logits.to(torch.bfloat16).to(device)
    temperatures = (0.5 + torch.rand(rows, generator=generator)).to(device)
    top_ks = torch.randint(1, 51, (rows,), generator=generator).to(device)
    top_ks[0] = 50
    top_ps = torch.where(
        torch.rand(rows, generator=generator) < 0.5,
        torch.ones(rows),
        0.5 + 0.5 * torch.rand(rows, generator=generator),
    ).to(device)
    seeds = torch.randint(0, 2**62, (rows,), generator=generator).to(device)
    positions = torch.randint(0, 1 << 20, (rows,), generator=generator).to(device)
    return logits, temperatures, top_ks, top_ps, seeds, positions


class Split:
    def __init__(self, rows: int, chunks: int, block_k: int):
        self.chunks = chunks
        self.block_k = block_k
        self.keys = torch.empty(
            rows * chunks * block_k, device="cuda", dtype=torch.int64
        )
        self.out = torch.empty(rows, device="cuda", dtype=torch.long)

    def __call__(self, inputs, has_top_p, keys_out=None):
        logits, temperatures, top_ks, top_ps, seeds, positions = inputs
        rows = logits.shape[0]
        chunk_keys_kernel[(rows, self.chunks)](
            logits,
            temperatures,
            self.keys,
            chunk=VOCAB // self.chunks,
            chunks=self.chunks,
            block_k=self.block_k,
            num_warps=4,
        )
        merge_sample_kernel[(rows,)](
            self.keys,
            top_ks,
            top_ps,
            seeds,
            positions,
            self.out,
            self.keys if keys_out is None else keys_out,
            candidates=self.chunks * self.block_k,
            block_k=self.block_k,
            has_top_p=has_top_p,
            store_keys=keys_out is not None,
            num_warps=8,
        )
        return self.out


class TwoLevel:
    def __init__(self, rows: int, chunks: int, block_k: int):
        self.chunks = chunks
        self.block_k = block_k
        self.out = torch.empty(rows, device="cuda", dtype=torch.long)

    def __call__(self, inputs, has_top_p, keys_out=None):
        logits, temperatures, top_ks, top_ps, seeds, positions = inputs
        two_level_kernel[(logits.shape[0],)](
            logits,
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            self.out,
            self.out if keys_out is None else keys_out,
            chunks=self.chunks,
            block_k=self.block_k,
            has_top_p=has_top_p,
            store_keys=keys_out is not None,
            num_warps=8,
        )
        return self.out


def served(inputs, has_top_p):
    logits, temperatures, top_ks, top_ps, seeds, positions = inputs
    return sk.sample_from_logits_with_seed_top_k_top_p(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        max_top_k=50,
        has_top_p=has_top_p,
    )


def graph_time_us(fn) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / CALLS)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    block_k = sk.fused_raw_logit_block_k(50)
    generator = torch.Generator().manual_seed(0)
    variants = {f"split {chunks}": (Split, chunks) for chunks in (4, 8, 16)} | {
        f"two-level {chunks}": (TwoLevel, chunks) for chunks in (8, 16)
    }
    print(f"block_k {block_k}; identity against the served kernel")
    failed = set()
    for name, (cls, chunks) in variants.items():
        key_mismatch = token_mismatch = rows_checked = 0
        try:
            for kind in ("normal", "ties", "zeros_inf"):
                for trial in range(20):
                    rows = 64
                    inputs = make_inputs(rows, kind, generator)
                    has_top_p = bool(trial % 2)
                    reference_keys = torch.empty(
                        rows * block_k, device="cuda", dtype=torch.int64
                    )
                    full_keys_kernel[(rows,)](
                        inputs[0],
                        inputs[1],
                        reference_keys,
                        block_k=block_k,
                        num_warps=8,
                    )
                    keys = torch.empty_like(reference_keys)
                    runner = cls(rows, chunks, block_k)
                    tokens = runner(inputs, has_top_p, keys).clone()
                    reference = served(inputs, has_top_p)
                    key_mismatch += int(
                        (keys.view(rows, block_k) != reference_keys.view(rows, block_k))
                        .any(dim=1)
                        .sum()
                    )
                    token_mismatch += int((tokens != reference).sum())
                    rows_checked += rows
        except Exception as error:
            failed.add(name)
            print(f"  {name:<14} failed: {type(error).__name__}: {str(error)[:200]}")
            continue
        print(
            f"  {name:<14} rows {rows_checked}  key rows differing {key_mismatch}  "
            f"tokens differing {token_mismatch}"
        )
    print(f"\nus per call, {CALLS} calls in one graph")
    print(f"  {'bs':>4} {'served':>8}" + "".join(f"{name:>15}" for name in variants))
    for rows in (1, 16, 64):
        inputs = make_inputs(rows, "normal", generator)
        cells = [f"{graph_time_us(lambda: served(inputs, False)):8.2f}"]
        for name, (cls, chunks) in variants.items():
            if name in failed:
                cells.append(f"{'-':>15}")
                continue
            else:
                pass
            runner = cls(rows, chunks, block_k)
            cells.append(f"{graph_time_us(lambda: runner(inputs, False)):15.2f}")
        print(f"  {rows:>4} " + "".join(cells))


if __name__ == "__main__":
    main()
