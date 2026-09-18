# Slice P2: seeded top-k sampler latency

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

## 5. Design options, chosen by P2-e1

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
