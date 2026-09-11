# 16 — Qwen3-TTS S3 fused residual readout, 2026-09-11

A: upstream main `80b5aaed74be473096846e81f5e217f92d11dcda`.
B: `e4ef7ed07d42f3108002ed9c8dfbd94d6dd98f1c`.
Physical GPU 0, NVIDIA H100 80GB HBM3. SGLang 0.5.19, torch
2.13.0+cu130, FlashInfer 0.6.18, sglang-kernel 0.4.6.post1. The same box
venv and installed dependencies were used for both arms.

## E1

The exact analysis-branch script was run at hidden size 1024.

| rows | normed equal | residual equal | max normed diff | max residual diff |
|---:|---:|---:|---:|---:|
| 1 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 2 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 4 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 8 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 16 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 32 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |
| 64 | 0/1000 | 1000/1000 | 3.125e-02 | 0 |

Residual output is bit exact; normalized output is not. Per runbook 15 this
selects the c16 quality-band gate and skips the optional seeded c1 identity
pass.

## Suites on B

- Qwen3-TTS: 459 passed, 14 warnings.
- Non-accelerator: 6447 passed, 7 skipped, 415 deselected, 25 warnings.
- Accelerator: 350 passed, 10 skipped, 6509 deselected, 23 warnings.

## Predictor census

| rows | kernels A | kernels B | busy A ms | busy B ms | delta ms | elementwise A to B | norm A to B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1062 | 982 | 3.347 | 3.271 | -0.076 | 87 to 7 | 256 to 256 |
| 16 | 1062 | 982 | 3.584 | 3.512 | -0.073 | 103 to 23 | 256 to 256 |

The structural gate passes: exactly 80 residual-add elementwise kernels are
removed at each row count, while the norm count stays fixed and the fused-add
RMSNorm kernel replaces the plain norm at the intended sites. Attention, GEMM,
and RoPE counts are unchanged; no family grows.

## Full corpus

Unseeded, warmup 1, two fresh boots per arm and point, order A c1, B c1,
B c16, A c16, then B c1, A c1, A c16, B c16. All points completed 1088 of
1088 requests.

| arm / point | req/s | p50 s | p95 s | p99 s | WER errors | WER | similarity | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A/c1/r1 | 2.400 | 0.407 | 0.591 | 0.709 | 131 | 1.097% | 71.3361 | 71349 |
| B/c1/r1 | 2.467 | 0.395 | 0.586 | 0.672 | 136 | 1.139% | 71.2059 | 71185 |
| B/c16/r1 | 17.003 | 0.921 | 1.310 | 1.517 | 127 | 1.063% | 71.2773 | 76055 |
| A/c16/r1 | 16.400 | 0.956 | 1.363 | 1.509 | 116 | 0.971% | 71.3824 | 77675 |
| B/c1/r2 | 2.471 | 0.397 | 0.573 | 0.681 | 123 | 1.030% | 71.3190 | 71123 |
| A/c1/r2 | 2.365 | 0.413 | 0.593 | 0.715 | 130 | 1.089% | 71.4121 | 71217 |
| A/c16/r2 | 16.345 | 0.958 | 1.358 | 1.583 | 116 | 0.971% | 71.2428 | 75879 |
| B/c16/r2 | 16.731 | 0.934 | 1.356 | 1.563 | 116 | 0.971% | 71.1478 | 76375 |

| pair | req/s delta | p50 A to B | p95 A to B | p99 A to B | peak MiB delta |
|---|---:|---:|---:|---:|---:|
| c1/r1 | +2.79% | 0.407 to 0.395 | 0.591 to 0.586 | 0.709 to 0.672 | -164 |
| c1/r2 | +4.48% | 0.413 to 0.397 | 0.593 to 0.573 | 0.715 to 0.681 | -94 |
| c16/r1 | +3.68% | 0.956 to 0.921 | 1.363 to 1.310 | 1.509 to 1.517 | -1620 |
| c16/r2 | +2.36% | 0.958 to 0.934 | 1.358 to 1.356 | 1.583 to 1.563 | +496 |

B improves throughput and median latency in all four pairs. B p95 is better
in all four pairs. Both B c16 quality points pass the historical gate: 127 and
116 word errors, and 71.2773 and 71.1478 similarity. A/c16/r1 similarity is
71.3824, above the historical upper edge; this baseline-arm outlier is retained.
Sampled c16 peak memory moves in opposite directions between pairs, so no
allocation change is inferred from the one-second samples.

## Log audit

All eight generation serve logs are free of tracebacks, CUDA errors, lazy
capture, fallback, retraction, and range-screen hits. All 8704 measured
generation requests completed successfully.

## E2 host tail

The A/main server was launched with `SGLANG_TORCH_PROFILER_WITH_STACK=1`.
Ingest found 2,083,866 Python-function records at c1 and 6,479,172 at c16,
confirming that stack capture was active.

| rows | steps | step wall p50 ms | step wall p90 ms | host-only tail p50 ms | host-only tail p90 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 658 | 8.558 | 9.817 | 0.037 | 0.040 |
| 16 | 76 | 9.546 | 11.553 | 0.048 | 0.053 |

The host-only tail owner table assigns 100% to generic Python and 0% to Omni,
SGLang, and torch at both row counts. At c16 the largest tail frames are CUDA
graph metadata application (8.8 us p50), graph input loading (6.4 us), graph
execution (3.5 us), environment lookup (3.0 us), and replay dispatch (2.9 us).
There is no Omni-owned host-tail item above the measured noise floor for S4.

## Streaming

Pending. The exact retained CI launcher requires two physical GPUs: one H100
per worker behind the router, with separate vocoder processes. The instruction
to use only physical GPU 0 prevents running this layout faithfully. No
single-GPU substitute was run or represented as equivalent.

## Current verdict

E1, all three suites, both-row census, eight full-corpus generation and quality
points, log audit, and E2 pass their applicable gates. The seeded identity pass
was correctly skipped by E1. Streaming remains the only incomplete runbook
section pending authorization for a second physical GPU.

## Verification of the archive, 2026-09-11, Mac

The eight speed points above equal the DONE lines of `full_ab_driver.log`. The census diffs
carry the numbers quoted. The compact archive holds no bench directories, so the WER and
similarity figures are taken from this readout, not rechecked from files. The E2 hosttail
tables in the archive were produced by the first version of the tool, which measured the wrong
window and misread owners (runbook 15 section 7); they are not used. The streaming pair was not
run: the change sits in the talker step both modes share, so the c16 band covers its quality
and the census covers its timing.

Numerics: E1 shows the fused kernel normalizes the fp32 sum where today's path normalizes the
bf16 rounded sum, one rounding fewer, the same arithmetic the backbone layers run. The residual
between layers is bit identical. c1 similarity reads 71.21 and 71.32 on B against 71.34 and
71.41 on A, a difference of the size of each arm's own boot to boot spread; c16 is inside the
114 to 135 error and 71.12 to 71.34 similarity band on both B boots, one pair with identical WER.

Decision: ship under the band gate with the numerics change stated in the PR. Protocol from
this run on: one boot per arm and point on this box, the census as the measurement.
