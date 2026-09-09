# 13 — RoPE store box readout, 2026-09-09

A: upstream main `4562a7ef3`. B: `79cdfe185`. Root venv, sglang 0.5.18. Full corpus points on physical GPU 1, fresh boot each, seed 1234, warmup 0. Streaming uses the separate-vocoder two-worker CI layout, warmup 1. No production-code patches.

## Suites

- Qwen3-TTS: 451 passed.
- CPU, CUDA hidden: 6319 passed, 8 skipped, 365 deselected.
- Accelerator: 302 passed, 8 skipped, 6382 deselected.
- `test_rope_store_writes_the_cache_the_copy_path_writes`: batch 1 and 16 both passed.

## Census

Both B row counts resolve to fused_rope_store_kernel. Replay count 1222 -> 1062 at c1 and c16, exactly 160 fewer elementwise kernels, no family count grows. The c1 cuDNN attention name is unchanged. See census_diff_c1.md and census_diff_c16.md for timings.

A c16 initial export was interrupted by the inherited 20-second script delay. The valid replacement is A/census-repair/census_c16. B c16 complete raw JSON was ingested directly. Colocated process traces carry the preprocessing stage label; they contain the predictor kernels. Interrupted traces are not used.

## Full corpus

| Arm / point | Complete | QPS | Lat p50 s | Lat p95 s | Lat p99 s | WER | SIM | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A/ab_c16_r1 | 1088/1088 | 15.093 | 1.028 | 1.53 | 1.812 | 1.047% | 71.1584 | 76145 |
| A/ab_c16_r2 | 1088/1088 | 15.99 | 0.979 | 1.421 | 1.769 | 1.038% | 71.1885 | 76455 |
| A/ab_c1_r1 | 1088/1088 | 2.259 | 0.434 | 0.639 | 0.753 | 0.996% | 71.3052 | 71097 |
| A/ab_c1_r2 | 1088/1088 | 2.274 | 0.43 | 0.633 | 0.744 | 1.022% | 71.3052 | 71097 |
| B/ab_c16_r1 | 1088/1088 | 16.103 | 0.965 | 1.415 | 1.781 | 0.971% | 71.1960 | 76801 |
| B/ab_c16_r2 | 1088/1088 | 16.254 | 0.962 | 1.387 | 1.835 | 0.971% | 71.1797 | 77037 |
| B/ab_c1_r1 | 1088/1088 | 2.376 | 0.413 | 0.605 | 0.704 | 1.013% | 71.3052 | 71095 |
| B/ab_c1_r2 | 1088/1088 | 2.355 | 0.417 | 0.615 | 0.723 | 1.022% | 71.3052 | 71095 |

## c1 byte identity

- c1_identity_r1: 1088/1088 identical; 0 differences.
- c1_identity_r2: 1088/1088 identical; 0 differences.

## Streaming

| Arm / pass | Complete | TTFC mean ms | TTFC p99 ms | ITL p99 ms | WER |
|---|---:|---:|---:|---:|---:|
| A/attempt1_c16 | 1088/1088 | 188.3 | 881.6 | 213.4 | 2.671% |
| A/attempt2_c16 | 1088/1088 | 145.9 | 470.1 | 218.5 | 1.733% |
| A/attempt3_c16 | 1088/1088 | 140.1 | 590.4 | 212.3 | 1.114% |
| B/attempt1_c16 | 1088/1088 | 181.8 | 794.3 | 182.3 | 1.030% |
| B/attempt2_c16 | 1088/1088 | 135.1 | 344.3 | 201.3 | 0.946% |
| B/attempt3_c16 | 1088/1088 | 125.7 | 279.6 | 191.6 | 1.072% |

## Verdict and open items

All requested suites, both-row censuses, eight non-streaming full-corpus boots, six streaming passes, fourteen WER scores and eight similarity scores completed. All 15232 measured generation requests succeeded (8704 non-streaming plus 6528 streaming). Both c1 pairs matched all 1088 WAVs. The kernel gate and c16 quality bands pass.

Paired B−A QPS: c1 +5.18% and +3.56%; c16 +6.69% and +1.65%. Median latency improves in every pair. However, c16 p99 is −1.71% in pair 1 and +3.73% in pair 2 (1.835 vs 1.769 s); that 66 ms paired increase exceeds the 43 ms spread of the two A boots. B sampled peak memory is +656 MiB and +582 MiB at c16; c1 differs by only −2 MiB. These prevent an unconditional all-metric performance/memory pass. They do not by themselves prove a regression on a shared host. No extra allocation or cause is inferred from sampled peaks.

Non-streaming c16 WER errors: A 125/124, B 116/116, all within 114–135. Similarity is within 71.12–71.34 on every c16 boot. c1 similarity is exactly 71.30515434286174 on all four boots. The first c1 pair gets 119 vs 121 ASR word errors despite byte-identical input WAVs; the second pair gets 122/122. This is scoring variability, not a TTS audio difference. Scoring used one shared Qwen3-ASR-1.7B endpoint, asynchronous decode disabled, benchmark-default ASR concurrency. No serial-ASR rerun was performed.

Streaming: each arm completed 3264/3264; routed equals successful for both workers in each pass, including warmup. Both arms have zero codec-range hits and tracebacks. B TTFC mean/p99 and ITL p99 are within A's observed range or better. B WER errors are 123/113/128 (113 is better than the historical band lower edge). A has 319/207/133 errors, with 2/1/0 samples above 50% WER. Those baseline outliers are retained in audit.json, not hidden or attributed to RoPE.

No identifiable #2042 c1_wav_sha256.json manifest was found among the retained artifacts; the optional historical-hash comparison was not performed. All new c1 hash manifests are retained. Full raw speed and quality distributions, per-worker counters, and log audits are in the JSON artifacts.

All-GPU CSVs preserve neighbouring load. CPU contention was not controlled. Raw bootstrap-to-bootstrap deltas on this shared host are not proof of a small regression or gain. This readout supports the mechanism and output correctness, but retains the p99/memory review items rather than declaring every gate green. No PR was opened.

The compact archive excludes WAVs, raw traces, temporary exports and ingested trace pickles; they remain on the box.

## Verification of the archive, 2026-09-09, Mac

Read from the archive files, not from the readout above.

Residency: all eight non streaming boots log the same pool, 573280 tokens, 61.23 GiB, and the
same 11.55 GB free after the pool. No boot logs an allocator retry, a lazy capture, a fallback,
a retraction, a traceback or a CUDA error. The c1 sampled peaks are 71097 MiB on A and 71095
on B in both pairs, so the slice adds no resident allocation.

Neighbouring load from the one second samples of the seven other GPUs, mean utilization during
each point, in run order: A c1 42.6, B c1 40.5, B c16 14.9, A c16 8.7, B c1 17.3, A c1 25.4,
A c16 19.8, B c16 21.6 percent. Pair 2 at c16 ran under equal neighbour load.

The c16 tail. In every c16 boot the top of the latency list holds a group of eight requests at
one latency: A pair 2 has eight at 1.77 s, B pair 2 eight at 1.99 s, B pair 1 five at 1.80 s.
Eight is the non streaming vocoder's whole utterance batch, doc 04 item M1. Eleven requests set
p99 of 1088, so p99 reads the plateau of whichever batch of eight was the slowest vocoder batch
of that run, and that batch's composition changes from boot to boot. Median and p95 improve in
both pairs. The c16 sampled peak sits at its maximum for about 65 of 170 samples in every
boot, a level, not a spike, and the same vocoder batch composition moves it, as the S1 allocator
snapshot of doc 03 section 12 attributed for the previous run.

Streaming ran on a different GPU from the non streaming points; its GPU 1 column is empty and
the per worker counters in audit.json are the record. A's first two passes carry two and one
samples above 50 percent WER on the baseline arm, retained as they are.
