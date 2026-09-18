"""P2-e1: where the fused seeded top-k sampler spends its 40 us (slices/P2_SAMPLER_LATENCY.md).

A copy of _seeded_top_k_top_p_sample_kernel (sampling_kernels.py:304) for the
checkpoint signature (max_top_k 50, block_k 64, no top-p) that stops after STAGE:
  1 load and scale, 2 + pack and topk, 3 + unpack, softmax and log, 4 + hash,
  5 + Gumbel, argmax and token (the full kernel).
Each stage stores a value that depends on all its work. Stage 5 must reproduce the
shipped kernel's tokens. FP32_GUMBEL prices the fp64 noise (attribution only).
Timed at num_warps 1, 2, 4, 8, 15 calls per CUDA graph, bs 1 and 16.

usage: python sampler_stage_bench.py
"""

from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    _fmix32,
    _gumbel_from_hash,
    _murmur3_mix,
    sample_from_logits_with_seed_top_k_top_p,
)

VOCAB = 2048
MAX_TOP_K = 50
BLOCK_K = 64
CALLS = 15
REPS = 50


@triton.jit
def _staged_kernel(
    logits,
    temperatures,
    top_ks,
    seeds,
    positions,
    out,
    logits_stride_b: tl.constexpr,
    block_k: tl.constexpr,
    STAGE: tl.constexpr,
    FP32_GUMBEL: tl.constexpr,
):
    row = tl.program_id(0)
    vocab_offsets = tl.arange(0, 2048)
    scores = tl.load(logits + row * logits_stride_b + vocab_offsets).to(tl.float32)
    temperature = tl.maximum(tl.load(temperatures + row).to(tl.float32), 1e-5)
    scores = scores / temperature
    result = tl.max(scores, axis=0).to(tl.int64)
    if STAGE >= 2:
        score_bits = scores.to(tl.uint32, bitcast=True)
        one_vocab = tl.full(vocab_offsets.shape, 1, tl.uint32)
        high_bit_vocab = one_vocab << 31
        all_ones_vocab = high_bit_vocab | (high_bit_vocab - one_vocab)
        ordered_score_bits = score_bits ^ tl.where(
            (score_bits >> 31) != 0, all_ones_vocab, high_bit_vocab
        )
        packed = (ordered_score_bits.to(tl.uint64) << 32) | (
            all_ones_vocab - vocab_offsets.to(tl.uint32)
        ).to(tl.uint64)
        top_packed = tl.topk(packed, k=block_k)
        result = tl.sum((top_packed & 0xFFFF).to(tl.int64), axis=0)
    if STAGE >= 3:
        ranks = tl.arange(0, block_k)
        one_rank = tl.full(ranks.shape, 1, tl.uint32)
        high_bit_rank = one_rank << 31
        all_ones_rank = high_bit_rank | (high_bit_rank - one_rank)
        ordered_top_score_bits = (top_packed >> 32).to(tl.uint32)
        top_score_bits = ordered_top_score_bits ^ tl.where(
            (ordered_top_score_bits >> 31) != 0, high_bit_rank, all_ones_rank
        )
        sorted_scores = top_score_bits.to(tl.float32, bitcast=True)
        sorted_token_ids = all_ones_rank - (
            top_packed & all_ones_rank.to(tl.uint64)
        ).to(tl.uint32)
        keep_top_k = ranks < tl.load(top_ks + row)
        masked_scores = tl.where(keep_top_k, sorted_scores, -float("inf"))
        max_score = tl.max(masked_scores, axis=0)
        probs = tl.exp(masked_scores - max_score)
        probs = probs / tl.sum(probs, axis=0)
        logprobs = tl.where(keep_top_k, tl.log(probs), -float("inf"))
        result = tl.max(tl.where(keep_top_k, logprobs, 0.0), axis=0).to(
            tl.int64
        ) + tl.sum(sorted_token_ids.to(tl.int64), axis=0)
    if STAGE >= 4:
        seed = tl.load(seeds + row).to(tl.uint64)
        pos = tl.load(positions + row).to(tl.uint32)
        h: tl.uint32 = 0
        h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, pos)
        h = _murmur3_mix(h, ranks.to(tl.uint32))
        h ^= 16
        h = _fmix32(h)
        result = result + tl.sum(h.to(tl.int64), axis=0)
    if STAGE >= 5:
        if FP32_GUMBEL:
            u = h.to(tl.float32) / 4294967295.0
            log_u = tl.minimum(tl.log(u), -2.3283064365386963e-10)
            sampled_scores = logprobs - tl.log(-log_u)
        else:
            sampled_scores = logprobs.to(tl.float64) + _gumbel_from_hash(h)
        max_sampled_score = tl.max(sampled_scores, axis=0)
        candidates = tl.where(sampled_scores == max_sampled_score, ranks, block_k)
        sampled_rank = tl.min(candidates, axis=0)
        result = tl.max(
            tl.where(ranks == sampled_rank, sorted_token_ids.to(tl.int64), 0), axis=0
        )
    tl.store(out + row, result)


def graph_us_per_call(fn) -> float:
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
    times = []
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return 1e3 * statistics.median(times) / CALLS


def main() -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    print(
        f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, triton {triton.__version__}"
    )
    for batch in (1, 16):
        logits = (torch.randn(batch, VOCAB, device=device) * 3).to(torch.bfloat16)
        temperatures = torch.full((batch,), 0.9, device=device, dtype=torch.float32)
        top_ks = torch.full((batch,), MAX_TOP_K, device=device, dtype=torch.long)
        top_ps = torch.ones(batch, device=device, dtype=torch.float32)
        seeds = torch.arange(1000, 1000 + batch, device=device, dtype=torch.long)
        positions = torch.arange(batch, device=device, dtype=torch.long) * 7
        out = torch.empty(batch, device=device, dtype=torch.long)

        def shipped():
            return sample_from_logits_with_seed_top_k_top_p(
                logits,
                temperatures,
                top_ks,
                top_ps,
                seeds,
                positions,
                max_top_k=MAX_TOP_K,
                has_top_p=False,
            )

        reference = shipped()
        print(
            f"\n== bs {batch}: shipped kernel {graph_us_per_call(shipped):.2f} us per call"
        )
        print(
            f"{'stage':>6} {'gumbel':>7} "
            + " ".join(f"{'w' + str(w):>8}" for w in (1, 2, 4, 8))
        )
        for stage, fp32 in (
            (1, False),
            (2, False),
            (3, False),
            (4, False),
            (5, False),
            (5, True),
        ):
            cells = []
            for warps in (1, 2, 4, 8):

                def run(stage=stage, fp32=fp32, warps=warps):
                    _staged_kernel[(batch,)](
                        logits,
                        temperatures,
                        top_ks,
                        seeds,
                        positions,
                        out,
                        logits.stride(0),
                        BLOCK_K,
                        stage,
                        fp32,
                        num_warps=warps,
                    )

                run()
                if stage == 5 and not fp32:
                    assert torch.equal(
                        out, reference
                    ), f"stage 5 copy diverged at num_warps {warps}"
                cells.append(graph_us_per_call(run))
            print(
                f"{stage:>6} {'fp32' if fp32 else 'fp64':>7} "
                + " ".join(f"{c:>8.2f}" for c in cells),
                flush=True,
            )


if __name__ == "__main__":
    main()
