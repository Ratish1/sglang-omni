# Run 04 readout: P1 unit tests, P1-e1 numerics, P1-e2 graph timing

moss card 1, RTX 4090 D. Trees: base `.tmp/wt/step-prof` (144bd6399), pair `.tmp/wt/p1`
(144bd6399 + `p1_pair_pass.patch` md5 735945eacf7c1d2330ef0e43d0181c8b, branch
`perf/qwen3-tts-predictor-pair-pass`). Import paths checked from inside each tree.
Text outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run04/` (tgz md5
a63f5ca800afce0691dfd0813a5f5715). Not copied (too slow over the link that day, kept on
the box): `predictor_inputs.pt` (6.5 MB) and the three 394 MB logits files.

## 1. Unit tests (qwen3_tts suite, pair tree)

531 passed, 1 failed, 3 skipped. The failure is
`test_eager_predictor_accepts_a_strided_input_and_leaves_its_neighbours`, which fails on
the base tree too (523 passed, 1 failed): max abs 0.0117 on 2 of 16 elements, sm89.
The first attempt's failure of the new pair test at bs 16 (max abs 0.0117, token 0
included) was one bf16 ulp from a different GEMM kernel at M = 32; the test now allows
one bf16 ulp at unit scale.

## 2. Inputs

1,600 rows (layer-0 code, talker hidden) recorded by
`scripts/record_predictor_inputs/sitecustomize.py` from the served talker during a decode
bs 16 driver run (client 12.87 ms per frame). The qwen-tts reference model does not run
under the box's transformers 5.12.1 (`pad_token_id` missing on its configs; after that,
an attention shape error), so the served talker is the source.

## 3. P1-e1: numerics (320 rows per batch size, greedy and seeded top-k 50 sampling)

Whole-row code agreement (15 predicted codes) and logits error from `p1e1_compare.txt`:

| bs | mode | base == fp32 | pair == fp32 | pair == base | logit mean abs base | pair |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | greedy | 0.4125 | 0.4125 | 0.5531 | 4.1224e-02 | 4.1505e-02 |
| 1 | sampled | 0.0406 | 0.0406 | 0.0656 | 3.6296e-02 | 3.5217e-02 |
| 2 | greedy | 0.4344 | 0.4344 | 1.0000 | 4.2045e-02 | 4.2045e-02 |
| 2 | sampled | 0.0500 | 0.0500 | 1.0000 | 3.5914e-02 | 3.5914e-02 |
| 4 | greedy | 0.4344 | 0.4344 | 1.0000 | 4.2045e-02 | 4.2045e-02 |
| 4 | sampled | 0.0500 | 0.0500 | 1.0000 | 3.5914e-02 | 3.5914e-02 |
| 8 | greedy | 0.4344 | 0.4094 | 0.5125 | 4.2045e-02 | 4.2253e-02 |
| 8 | sampled | 0.0500 | 0.0469 | 0.0562 | 3.5914e-02 | 3.5131e-02 |
| 16 | greedy | 0.4219 | 0.4219 | 0.5125 | 4.1310e-02 | 4.1746e-02 |
| 16 | sampled | 0.0500 | 0.0437 | 0.0719 | 3.5604e-02 | 3.6799e-02 |

(Mean abs is over each arm's own agreeing prefix, so the two arms average different
rows; the sub-step 0 read below has no such bias.)

Sub-step 0 logits (the first pass's output, identical inputs in every arm) against fp32,
all 320 rows, and paired code distance on the rows where base and pair differ
(`p1e1_paired.txt`):

| bs | mode | sub0 mean abs base | pair | sub0 max abs base | pair | rows differ | pair closer | base closer | tie |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | greedy | 2.8688e-02 | 2.8912e-02 | 0.2400 | 0.2460 | 143 | 52 | 56 | 35 |
| 1 | sampled | 2.8688e-02 | 2.8912e-02 | 0.2400 | 0.2460 | 299 | 72 | 71 | 156 |
| 2 | both | 2.8912e-02 | 2.8912e-02 | 0.2460 | 0.2460 | 0 | | | |
| 4 | both | 2.8912e-02 | 2.8912e-02 | 0.2460 | 0.2460 | 0 | | | |
| 8 | greedy | 2.8912e-02 | 2.8602e-02 | 0.2460 | 0.2192 | 156 | 56 | 68 | 32 |
| 8 | sampled | 2.8912e-02 | 2.8602e-02 | 0.2460 | 0.2192 | 302 | 74 | 79 | 149 |
| 16 | greedy | 2.8602e-02 | 3.0299e-02 | 0.2192 | 0.2307 | 156 | 68 | 67 | 21 |
| 16 | sampled | 2.8602e-02 | 3.0299e-02 | 0.2192 | 0.2307 | 297 | 71 | 81 | 145 |

Reads:

- At bs 2 and 4 the pair pass is bit-identical to the base chain (every code, every
  logit).
- The pair pass at batch B runs its first-pass GEMMs at M = 2B and reproduces the base
  chain's sub-step 0 error at batch 2B exactly: pair bs 1 = base bs 2 = 2.8912e-02,
  pair bs 8 = base bs 16 = 2.8602e-02. The rounding follows M (the GEMM kernel cuBLAS
  picks), not the fold.
- At bs 16 the pair pass runs M = 32, which the base chain never runs at the default
  `max_running_requests=16`: sub-step 0 mean abs 3.0299e-02, 5.9 percent above base bs
  16 and 4.8 percent above the base's largest value in this run. Codes: 68 vs 67 rows
  closer to fp32 (greedy), 71 vs 81 (sampled), out of about 300 differing rows.
- Seeded sampling turns any logit change into different codes for most rows (base vs
  fp32: 4 to 5 percent whole-row agreement; base vs pair at bs 1: 6.6 percent), so code
  identity is not a usable gate across kernel choices; distribution-level quality is,
  which is the census.
- Also measured: the bf16 predictor matches the fp32 predictor's greedy codes on only
  41 to 43 percent of rows, at every batch size, in both arms.

Status: P1-e1 passes at bs 1 to 8 (the pair pass reproduces base numbers at M = 2B). At
bs 16 the M = 32 kernel adds 5 to 6 percent sub-step 0 error. P1-e1b checks whether the
base chain at bs 32 (M = 32) gives the same number; if it does, that error belongs to
the M = 32 kernel, which serving at 32 concurrent requests already runs.

## 4. P1-e2: predictor graph replay (bench talker)

| bucket | base ms | pair ms | delta ms | base kernels | pair kernels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.081 | 6.923 | -0.158 | 2902 | 2741 |
| 2 | 10.174 | 9.549 | -0.625 | 3142 | 2952 |
| 4 | 10.196 | 9.616 | -0.580 | 3142 | 2952 |
| 8 | 10.274 | 9.672 | -0.602 | 3142 | 2947 |
| 12 | 10.323 | 9.294 | -1.029 | 3142 | 2968 |
| 16 | 11.064 | 9.929 | -1.135 | 3062 | 2898 |

The bench talker runs rope and qk-norm as plain torch ops (several kernels each) where
the server runs fused kernels, so its absolute times and its per-pass savings are larger
than the server's (the server's predictor replay measured 6.58 ms at bs 16 in run01).
The server delta comes from the matrix A/B ledger.
