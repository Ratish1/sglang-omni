# Qwen3-TTS code predictor study (11-14, host-62, H100)

## What runs per frame

- One opening pass of two tokens a request: the projected talker hidden state and the layer 0 code embedding.
- Then 15 codebook steps. Each is:
  - lm_head `g` (1024 to 2048);
  - seeded top-k 50 sampling at temperature 0.9, which matches `torch.topk` tie order and the murmur3 Gumbel draw;
  - the codec embedding gather into the talker feedback sum, and the projected embedding gather;
  - for steps 0 to 13, a 5-layer pass of one token a request over a cache of up to 17 slots.

Shapes:
- Hidden 1024, 16 q heads and 8 kv heads of dim 128, MLP 3072.
- A layer is 31.5 MB of bf16 weights, so 5 layers are 157 MB, plus 4.2 MB of lm_head per codebook.
- A frame reads about 2.4 GB, so its HBM floor is 0.73 ms at 3.35 TB/s at any batch up to 64. At batch 64 the MMA work is about 155 GFLOP, about 0.2 ms.

Default path (`sglang_omni/utils/predictor_layers.py`, shared with Qwen3-Omni):
- 5 Triton launches a layer with split-K and programmatic dependent launch, so about 29 dependent kernels a codebook step and about 443 a frame.
- Passes of more than 64 rows run plain, so the 128-row opening pass at batch 64 does.

Opt-in CuTe DSL path (#3317, `SGLANG_OMNI_QTTS_BACKEND=cutedsl`):
- 3 launches a layer: `pre_attention`, the Triton `decode_attention`, and `post_attention`.
- `post_attention` is a fully resident cooperative-grid kernel for `o_proj` and the MLP, with TMA weight prefetch and published activation words.
- It covers 32 rows at most.

## In-server graph span of the predictor a frame (synthetic decode cells)

| | bs 1 | bs 16 | bs 64 |
|---|---|---|---|
| default (Triton) | 2.41 ms | 2.51 ms | 4.22 ms |
| CuTe DSL | 2.03 ms | 2.20 ms | not covered |

Decode ms per frame with no profiler: default 4.46 / 5.70 / 10.07, CuTe DSL 3.85 / 4.99 / 10.06.

## Isolated operators (`predictor_bench.py`, real weights, 15 cached slots)

Setup:
- Times are µs a call, from a CUDA graph of 20 back-to-back calls of the same operator, so they are throughput.
- Programmatic dependent launch overlaps consecutive Triton calls, so in a dependent chain each operator costs more.

| Operator | bs 1 | bs 16 | bs 64 | HBM floor |
|---|---|---|---|---|
| qkv, Triton fused (norm, qkv, qk norm, rope, store) | 6.70 | 7.29 | 11.58 | 2.50 |
| qkv, CuTe DSL `pre_attention` | 5.82 | 6.13 | n/a | 2.50 |
| qkv, cuBLAS GEMM alone | 3.79 | 3.72 | 4.15 | 2.50 |
| attention, Triton | 1.78 | 2.14 | 3.28 | |
| attention, SDPA | 2.90 | 3.11 | 6.48 | |
| attention, FA3 `flash_attn_with_kvcache` | 7.39 | 8.40 | 16.82 | |
| o_proj + residual, Triton | 4.26 | 4.28 | 5.07 | 1.25 |
| o_proj + residual, cuBLAS addmm | 6.11 | 5.10 | 5.26 | 1.25 |
| gate_up, Triton fused (norm, gemm, silu) | 4.63 | 4.68 | 7.09 | 3.76 |
| gate_up, cuBLAS GEMM alone | 5.01 | 4.43 | 5.12 | 3.76 |
| gate_up, rmsnorm + cuBLAS + silu_and_mul | 7.43 | 7.02 | 7.98 | 3.76 |
| down + residual, Triton | 4.87 | 4.97 | 5.75 | 1.88 |
| down + residual, cuBLAS addmm | 7.37 | 6.30 | 6.51 | 1.88 |
| o_proj + MLP, CuTe DSL `post_attention` | 11.07 | 11.15 | n/a | 6.89 |
| lm_head, cuBLAS | 3.66 | 3.63 | 3.70 | 1.25 |
| seeded top-k sampler + gathers | 6.49 | 6.93 | 7.77 | |
| **codebook step, Triton** | **157.4** | **162.9** | **221.6** | **48.2** |
| **codebook step, CuTe DSL** | **137.4** | **139.3** | n/a | **48.2** |

The bench matches serving: 15 codebook steps make 2.36, 2.44 and 3.32 ms for Triton against 2.41, 2.51 and 4.22 in the server. At batch 64 the server also runs the plain opening pass and shares the card with the vocoder.

Against an fp32 reference of one codebook step, the max abs difference is 0.19, 0.23 and 0.52 for Triton and the same for CuTe DSL.

Reading:
- The best existing backend for each operator is CuTe DSL `pre_attention` (qkv), Triton attention, and CuTe DSL `post_attention` (o_proj and MLP), or Triton per operator at batch 64. Together they give the CuTe DSL chain, 137 µs, 2.8 times the 48 µs floor.
- Every operator is 2 to 4 times its floor, because a 4 to 13 MB matrix is too small for one launch to ramp to full bandwidth. The codebook step then adds about 29 dependent launches. Swapping backends per operator cannot close that.
- FA3 is the wrong backend for a 16-slot cache: four times slower than the Triton kernel.
- Nsight Compute shows the Triton GEMV kernels at about one CTA per SM, 3 to 6 % occupancy, long-scoreboard stalls and 16 to 40 % of DRAM peak.

## Precedents for fusing the whole chain (primary sources)

- **Hazy Research Llama-1B megakernel** (https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).
  - An interpreter on every SM runs a static schedule, over shared-memory pages.
  - Weights for an instruction load as soon as a page is free, before its inputs are ready.
  - Global counters hand off activations, with no grid barriers.
  - Result: under 1 ms a forward pass on H100 at 78 % of memory bandwidth, against about 50 % for vLLM and SGLang.
  - On B200, activation handoffs, not weights, are the largest share of time.
- **Mirage Persistent Kernel** (arXiv 2512.22219): a task graph with event counters, worker and scheduler SMs, and shared-memory pages; 1.0 to 1.7 times over SGLang and vLLM.
- **Kernel launch cost:** about 1.3 µs per kernel inside a CUDA graph (Hazy's measurement).
- **In repo:** CuTe DSL `post_attention` already uses TMA weight prefetch into shared memory, clusters, and published activation words with empty sentinels.
