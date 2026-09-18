# Run 07 readout: server profiles of base, PF, P1, P2 (base twice)

moss card 0, RTX 4090 D, Qwen3-TTS-12Hz-1.7B-Base, `mem_fraction_static` 0.85 (shipped
default). Each boot: `run_profile_boot.sh` (md5 6e69d8e891f8016576b701fb9939108b), the
same 12 captures in the same order, ledger on every trace. Trees: upstream 144bd6399 +
profiler (`step_profiler_v2.patch`) + the slice (P1 `ff725039e`, P2 `44672e5d7`, PF
`2f60fc1a8` carry the same diffs). Order base1, pf, p1, p2, base2; boots 14:39 to 15:08.
Text outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run07/` (tgz md5
135de3ca4e21d64d9c9112f7630c66b5); traces (172 MB) stay on the box. The nsys boot failed
on a script flag (`--cpuctxsw` belongs to `nsys start`, not `nsys launch`); fixed, re-run
in run 08's queue.

## 1. Measurement validity

- Every capture window is contiguous (`fwd range ... contiguous True`), owner resolution
  0 mismatches, armed marker present in every boot.
- base1 and base2 agree within 1.5 percent on every decode read (table 2), so drift across
  the 30 minutes is below the slice deltas.
- The in-window ledger step of decode bs 1 (8.77 ms p50) matches the unprofiled client
  cadence (8.64 ms per frame). The client cadence of the profiled decode windows at bs 1 to
  8 (26.7 to 34.2 ms per frame) is not a step time: those streams are short (57 to 82
  frames) and include the trace export; the ledger is the measurement.

## 2. Decode: wall ms per step, p50 (mean) over 39 steady steps

| bs | base1 | pf | p1 | p2 | base2 | p1 vs base1 | p2 vs base1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.769 (8.938) | 8.748 (8.941) | 8.497 (8.698) | 8.295 (8.515) | 8.762 (8.948) | -0.272 (-3.1%) | -0.474 (-5.4%) |
| 2 | 9.977 (10.204) | 10.086 (10.240) | 9.719 (9.942) | 9.553 (9.779) | 10.129 (10.256) | -0.258 (-2.6%) | -0.424 (-4.2%) |
| 4 | 10.182 (10.449) | 10.259 (10.473) | 9.945 (10.162) | 9.729 (10.012) | 10.172 (10.437) | -0.237 (-2.3%) | -0.453 (-4.4%) |
| 8 | 10.313 (11.023) | 10.489 (11.184) | 10.062 (10.641) | 9.887 (10.648) | 10.257 (11.027) | -0.251 (-2.4%) | -0.426 (-4.1%) |
| 16 | 11.146 (12.990) | 11.109 (13.068) | 10.814 (13.004) | 10.725 (12.600) | 11.250 (12.812) | -0.332 (-3.0%) | -0.421 (-3.8%) |

Predictor replay span p50 (ms): base1 4.914 / 5.697 / 5.754 / 5.776 / 6.530 at bs 1 / 2 /
4 / 8 / 16; p1 4.669 / 5.398 / 5.447 / 5.513 / 6.169; p2 4.456 / 5.227 / 5.301 / 5.316 /
6.077. Talker replay unchanged in every arm (3.60 to 4.29 ms). Kernels per step: p1 39 to
60 fewer than base; p2 one more (the noise launch).

- P1 removes 0.24 to 0.36 ms of predictor time per step at every batch size; P2 removes
  0.42 to 0.47 ms. P2 matches its micro-bench (15 calls x about 31 us of fp64 noise).
- PF leaves decode unchanged, as it should.
- Composition at bs 1 (base1): device busy 8.64 of 8.77 ms wall: talker graph 3.61 ms
  (113 gemv, 3.25 ms, at about 87 percent of 1008 GB/s), predictor graph 4.91 ms (352
  gemv 3.51 ms; sampler 15 x 39.4 us = 0.59 ms), talker layer-0 sampling kernel
  `triton_red_fused__to_copy_add_argmax_clamp_div_log_neg_2` 64.7 us per step. Device
  idle 0.13 ms. The step is device-bound at every batch size, so host-side overlap
  (feature inventory) has at most 0.13 ms per step to take at bs 1.
- bs 16 mean vs p50 (13.0 vs 11.1 ms) is vocoder contention (other threads concurrent 1.0
  ms mean), as in run01.

## 3. Prefill (formal captures: bs 1 = 10 single-request prefills, bs 8 = one burst)

| read, p50 | base1 | pf | p1 | p2 | base2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs 1 span ms (the scheduler step) | 26.064 | 13.732 | 25.365 | 25.473 | 25.317 |
| bs 1 scheduler device busy ms | about 10.4 | 10.215 | | | |
| bs 8 span ms | 84.678 (n = 1) | 22.660 | 30.957 | 30.476 | 32.522 |

- PF halves the bs 1 prefill step (26.06 to 13.73 ms; -47 percent against base1, -46
  percent against base2). serve.log: PF replays 91 prefill batches and runs 2 eager;
  base 0 of 95. Capture on the 4090: 31 buckets, 3.58 s, 0.20 GB (2.33 GB free after).
- PF prefill step composition: 29 graph segments of the Talker layers (0.034 to 0.19 ms
  each, about 4.8 ms), the predictor graph 4.92 ms, the rest (about 3.5 ms) host work
  between segments (eager attention metadata, the sidecar path).
- In every prefill step the vocoder's initial worker thread is busy 22.1 ms p50 (PF trace,
  tid 651217): the reference-prefix bootstrap decode through the window graphs. After PF
  it is the largest single piece of the first-audio path (vocoder slice).
- bs 8 prefill has one to four coalesced extends per capture; the p50 comparison is noisy
  (base1 n = 1). PF's 22.66 vs base2 32.52 ms agrees in direction.

## 4. Memory (nvidia-smi every second, whole boot)

Peak: base 21,988 MiB, pf 22,204 MiB, p1 22,000 MiB, p2 21,976 MiB of 24,564. KV pool
12.22 GiB (114,384 tokens) at 0.85; live KV at 16 requests is about 1,600 tokens.
Headroom about 2.4 GB at this workload. The vocoder slice's all-width compiled graphs
add 0.1 to 1.3 GB per large key (READOUT_05 section 3), so the fraction is sized when
that slice lands, from KV demand (16 running x (prompt + 2,048 new tokens), about 37k
tokens) plus measured peaks, not before.

## 5. Where a decode step and a first audio go now

- Decode (bs 1, 8.77 ms): talker gemv 3.25, predictor gemv 3.51, predictor sampler 0.59
  (P2 removes about 0.47), predictor non-GEMM layer kernels (P1 removes one pass),
  talker sampling 0.065.
- First audio (bs 1): preprocessing and reference encode, the prefill step (26.1 ms, 13.7
  with PF), the vocoder bootstrap decode (22 ms).

Next: P1 + P2 + PF stacked on one branch, then the matrix A/B and census; the vocoder
slice from run 08 (V-e6); the talker layer-0 sampler (64.7 us per step, fused Triton
argmax with the same fp64 noise pattern) to be read in SGLang's sampler before any
change.
