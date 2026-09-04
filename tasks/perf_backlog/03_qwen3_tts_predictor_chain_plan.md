# 03. Qwen3-TTS predictor chain: fewer kernels in the replay, a shorter tail (T22)

Status: Conditional. Slices S1 and B1 are fully specified from the census
and the code. S2 and S3 carry a numerics gate whose band is measured in
S1. B3 is a measurement that decides whether the graph launch cost is a
third target. Evidence: doc 02 sections 6 and 8 (the H100 census and
timeline of 2026-09-04, branch `perf/qwen3-tts-profiling`), and the code
at `perf/qwen3-tts-predictor-startup-capture` head `fff86552c`.

## 1. Requirement

Cut the decode step of the Qwen3-TTS talker without changing the audio
of any request beyond the variation batching already produces, on H100,
in the default non streaming serving mode. The replay is one
`cudaGraphLaunch` of 1371 kernels whose time is the sum of kernel
durations plus a 0.45 us gap per kernel, and the host tail after it is
35 to 45 serial launches for 60 us of device work. Both are counted
items, not estimated ones.

## 2. The three targets and their sizes

Per step, at 1 row unless noted, from doc 02:

| Target | Today | Reachable | How |
| --- | ---: | ---: | --- |
| A. kernels in the predictor replay | 1371 kernels, 4.74 ms wall | about 860 kernels, about 3.7 ms | A1 to A4 below, omni code and one Triton kernel |
| B. host tail after the replay | 1.10 ms (1.40 at 16 rows) | about 0.7 ms | B1, B2 below, omni code |
| C. the GEMMs of the replay | 432 calls, 2.07 ms, 40% of bandwidth | sized after A | a GEMV path for M up to 16 |

The reachable column for A adds the rows of section 3. C is not sized
here: its floor is 0.75 ms of weight traffic per replay, and whether a
custom path beats cuBLAS at M 1 to 16 is a measurement.

## 3. Target A, the replay

Each removed kernel saves its duration plus the gap. Counts are per replay
of 16 sub-steps.

- A1 The sampler prologue. `_sample_subtalker_token_seeded` selects
  temperatures, top k, top p and seeds with four `index_select` on
  `row_indices`, which since the eager and graph paths were unified is
  always the identity `arange[:batch]` (sglang_model.py, the call in
  `_sample_subtalker_token`), clamps the temperatures, and computes the
  sub positions with three elementwise kernels per sub-step. Change: drop
  the `row_indices` argument, read the four parameters as slices of the
  staged tensors (views, no kernel), clamp the temperatures once where
  `prepare_decode_buffers` stages them, and compute the sub position
  table `[num_code_groups, batch]` once per predictor call and slice it
  per sub-step. Removes 8 kernels per sub-step, 128 per replay, about
  0.20 ms. Bit identical by construction: the same values reach the
  sampler kernel.
- A2 All rows sample. `_sample_subtalker_token` always computes the
  argmax and the `where` so that mixed batches get argmax rows. When every
  row samples, which the default checkpoint does, both are dead. Change:
  a fifth signature term, whether any row is argmax, so the graph of the
  default signature has no argmax and no `where`, and a mixed batch keeps
  today's graph under its own key. Removes 2 kernels per sub-step, 32 per
  replay, 0.09 ms at 1 row and 0.22 ms at 16 (the argmax is the kernel
  that grows with rows). Bit identical: the selected token is the sampled
  one in both cases.
- A3 One kernel for qk norm, rope and the two cache writes. Today per
  layer: `sglang::fused_qknorm_warp`, `sglang::fused_rope_kernel`, and
  two elementwise kernels writing k and v into the predictor's private
  cache. One Triton kernel that normalises q and k, applies rope, and
  writes k and v into the cache slot removes 3 kernels per layer, 240 per
  replay, about 0.30 ms. Numerics: the norm and rope arithmetic stays in
  fp32 as sglang's kernels do it, but the reduction order of the norm
  changes, so this is gated by G2.
- A4 The residual add into the next RMSNorm. The MLP output's residual
  add is a separate elementwise kernel before the next layer's input
  norm. sglang's `fused_add_rmsnorm` does both. Removes 1 kernel per
  layer, 80 per replay, about 0.10 ms. Same gate as A3.
- A5 The split K reduce of the down projection: 80 kernels, 0.13 ms. A
  different cuBLAS algorithm or a fused epilogue removes the reduce but
  may lengthen the main GEMM. Measured with the GEMV work of target C.
- A6 The sampler kernel itself, 20.6 us per call for a 2048 wide row, 15
  per replay, 0.30 ms. A second version of the kernel is its own item,
  sized by a micro benchmark against the reference.

## 4. Target B, the tail

- B1 `_write_feedback_buffers` (model_runner.py) runs a Python loop over
  rows that launches one kernel per row and then stacks. Change: take the
  next decode input embeds of all rows as one gather from the staged
  feedback queue into the embedding rows, one launch per step. Removes
  `rows` launches, 0.19 ms at 16 rows, more at 64. Bit identical: the
  same rows land in the same slots.
- B2 `prepare_decode_buffers` uploads six `torch.tensor(list)` per step
  for the sampling parameters (sglang_model.py). At steady state the
  batch does not change between steps, so the uploads rewrite the same
  values. Change: key the staged parameters by the ordered request ids
  and skip the uploads when the key is unchanged, upload only the rows
  that changed otherwise. Removes 6 host to device copies and their
  launches per steady step, about 0.1 ms of host time. Bit identical.
- B3 Measure the graph launch cost without the profiler: the ledger
  records the host duration of the backbone replay call and of
  `graph.replay` per step. If a 1371 node launch costs more than the
  0.1 ms a host can hide while the device drains the eager sampling, the
  node count of target A is also a host item, and the ledger shows the
  gain per slice.
- B4 Overlapping the next step's batch build with the replay is sglang's
  overlap schedule, which this model disables by default. It is not part
  of this plan: the Qwen3-Omni talker's attempt was held on a TTFA
  regression (doc 05 section 7.4), and B1 to B3 first show how much tail
  is left to hide.

## 5. Proof

- G1 Bit identity for S1: the existing predictor graph tests (every
  bucket, startup ladder against eager) plus the full corpus c1 A/B with
  1088 of 1088 WAVs equal to the base. The A2 term has a signature test
  for the mixed batch key.
- G2 Numerics band for the fusions: the reference band is the variation
  the current batching produces (doc 24 section 3, two boots of one
  revision at c16, and one reference encoded alone against in a batch).
  A fusion passes when its c1 output differs from the base by no more
  than that band on the same rows, and its WER and similarity sit inside
  the run to run band of doc 24 section 3.2, on the four server c16
  replication of doc 24.
- G3 The census diff: `perfkit.py diff` between the base census and the
  slice's census at 1 and 16 rows, kernels per replay down by the counts
  of section 3, busy down by at least the durations removed. The timeline
  at 16 rows for B1 and B2, the per row run of launches gone.
- G4 The A/B of doc 15, full corpus, c1 and c16, fresh servers, arms
  alternated, `--seed 1234`, the same protocol as the startup capture.

## 6. Slices

- S0 B3 on the profiling branch, one census run. Decides whether the
  launch cost enters the accounting. Box time only.
- S1 A1, A2, B1, B2. Omni Python only, no new kernel, expected bit
  identical. About 0.3 ms off the replay and 0.3 ms off the tail at 16
  rows, 7% of the step. Its census diff also fixes the base for S2.
- S2 A3, one Triton kernel in `predictor_kernels.py`, G2 gate.
- S3 A4 with `fused_add_rmsnorm`, G2 gate.
- S4 A5, A6 and target C, each behind its own micro benchmark.

Branch: `perf/qwen3-tts-predictor-chain` from the startup capture head,
rebased onto main when #1947 merges. Measurement runs on
`perf/qwen3-tts-profiling` with the chain branch merged in.

## 7. Open until measured

- B3, the unprofiled launch cost.
- The G2 band, produced by the S1 base run.
- Whether A5 pays, and the GEMV floor of target C.
