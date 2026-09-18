# Slice P2: seeded top-k sampler latency

Status 2026-09-18: dropped as a PR by the user's decision. The seeded sampler's logic is
referenced against SGLang's sampler and kernels instead; branch
`perf/qwen3-tts-sampler-noise` stays as a record only.

## 1. What is measured

`_seeded_top_k_top_p_sample_kernel` (`sampling_kernels.py:304`) runs 15 times per decode
step, one program per batch row, `num_warps=8` (`:699`). Bench 01: 40.8 us per call at
bs 1 and at bs 16 (trace: 40.4), 0.61 ms per step. `tl.topk(k=64)` alone over 2048 keys
costs 10.4 us, so selection is a quarter of the kernel.

## 2. The kernel, stage by stage (checkpoint signature: max_top_k 50, block_k 64, no top-p)

```
load 2048 bf16 logits, /temperature (fp32)                        :318-321
pack (ordered score bits << 32 | ~index) as uint64, tl.topk k=64  :327-339
unpack scores and ids                                             :341-354
(max_top_k <= 32 only: threshold gather + 32-entry bitonic sort)  :356-405
softmax over the 64 kept ranks (fp32)                             :407-411
(top-p cumsum)                                                    :413-419
log                                                               :421
murmur3 hash of (seed, position, rank)                            :423-433
Gumbel in fp64: two fp64 logs per rank                            :57-67, :435
fp64 add, max, argmin over ties, gather the token                 :436-448
```

Contract: bit-exact against SGLang `multinomial_with_seed`, whose Gumbel noise is fp64
(`:58-61`). fp64 is therefore required; how it maps onto threads is not.

## 3. Hypothesis (unmeasured)

The 64-rank tensors are smaller than the program's 256 threads, so Triton replicates
them across the 8 warps and each warp evaluates every fp64 log. sm89 runs fp64 at 1/64
of fp32; the H100 at 1/2. If this holds, the cost is consumer and L-series specific and
a thread mapping that evaluates each rank once removes it on every card, with identical
values.

## 4. Experiment P2-e1 (script `scripts/sampler_stage_bench.py`, TESTING.md section 3)

Copies of the kernel inside the bench, each ending at one stage and storing a value that
depends on everything before it: load and scale; + pack and topk; + unpack and softmax;
+ hash; + Gumbel and argmax (the full kernel). Each at `num_warps` 1, 2, 4, 8, bs 1 and
16, 15 calls per CUDA graph. The full copy's tokens must equal the shipped kernel's
tokens on the same inputs (the copy is faithful). An attribution-only copy with an fp32
Gumbel prices the fp64 part.

## 5a. Result (READOUT_03.md section 1) and the design it selects

At `num_warps=8` the fp64 Gumbel adds 35.2 us (bs 1) and 31.7 us (bs 16) of the 43 us;
an fp32 copy adds 0.8 us; the cost grows with the warp count. Selection is 6.8 us.

The noise is a function of (seed, sub-step position, rank) only, not of the logits
(`sampling_kernels.py:423-435`), and the 15 sub-step positions of a decode position are
known before the first sub-step (`_sub_seed_positions`, `sglang_model.py:1644`, one
(15, B) tensor per position). So:

```
_code_predictor_forward_incremental                      sglang_model.py:1518
  sub_positions = _sub_seed_positions(...)   (15, B)     :1581
  noise = seeded_gumbel_noise(seeds, sub_positions, block_k)   one launch, grid 15 x B,
          (15, B, block_k) fp64, _gumbel_from_hash on the same murmur3 hash as today
  15 x sample(logits, ..., noise[layer_idx])             the kernel loads its row of noise
                                                         instead of computing it
```

Values are bit-identical by construction (same hash, same fp64 function, same ranks), so
the tokens are. The fp64 work runs once per step on 15 x B programs spread over the
SMs instead of 15 times serially inside one program per row. No constant depends on
the card; on the H100 (full-rate fp64) the change removes less but adds only one small
launch per step inside the captured graph. The torch fallback path is unchanged.

P2-e2 (with the code): sampler us per call and the noise kernel's us at bs 1, 2, 4, 8,
16 in a graph, tokens equal to the shipped kernel on the same inputs.

## 5. Design options considered before P2-e1

- fp64 Gumbel dominant: compute the 64 noise values once, not per warp (a layout where
  the rank axis spans the threads, or `num_warps` from a startup measurement on the
  device, since every width gives the same values). Bit identical by construction.
- selection dominant: a two-stage top-k over vocab slices on several programs. The packed
  (score, index) keys are unique, so the top 64 of the per-slice top 64s is the global
  top 64; bit identical by construction.
- The kernel hard-codes vocab 2048 (`:318`) and the caller rejects anything else
  (`:658`); the redesign takes the vocab as a constexpr, so another checkpoint gets the
  fused path.

## 6. Gates

`tests/unit_test/qwen3_tts/test_sampling_kernels.py` unchanged and passing on the box,
plus a sweep of recorded logits (P1-e1's recording) against the reference path, all
tokens equal. Then the matrix A/B and census as for P1.
