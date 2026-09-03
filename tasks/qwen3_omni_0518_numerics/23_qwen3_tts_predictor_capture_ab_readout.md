# 23. Final A/B of the predictor startup capture

Source: `artifacts/qwen3tts-predictor-capture-validation-20260903-final-961e608f5-slim.tar.gz`,
extracted to the scratchpad as `run6/`. Arm A is upstream main `fa1ea43dc`,
arm B is `961e608f5` on `perf/qwen3-tts-predictor-capture` (Slice A of doc
21 before the review cleanup, which changed no runtime behaviour except that
a startup capture failure now raises). Physical H100 GPU 2, one server per
point, order A c1, B c1, B c16, A c16, `--warmup 0`, `--seed 1234` per
request (T39), full seed-tts-eval English split, 1088 samples.

## Startup

From `logs/serve_B_*.log`: the pipeline process reported ready 4.5 s later
on B at c16 (11.2 s against 6.7 s from "Building scheduler"), of which the
six predictor captures took 3.0 s (c16) and 3.1 s (c1) by the startup log
line. No lazy capture line appears in either B serving log. Arm A captured
lazily, one key at c1 and six at c16, inside serving steps.

## Speed

| Point | Latency mean s | Median s | p95 s | p99 s | RTF mean | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A c1 | 0.444 | 0.430 | 0.656 | 0.768 | 0.1092 | 2.251 |
| B c1 | 0.444 | 0.432 | 0.648 | 0.767 | 0.1089 | 2.252 |
| A c16 | 1.106 | 1.029 | 1.577 | 4.120 | 0.2733 | 14.386 |
| B c16 | 1.056 | 1.033 | 1.507 | 1.954 | 0.2598 | 15.048 |

B minus A: the first c1 request is 0.855 s faster and the first c16 batch
2.609 s per request faster. Past the first requests c1 is flat (paired
+0.49 ms). At c16 mean latency is 50 ms lower, p99 2.17 s lower, QPS 4.6%
higher, and the paired latency past the first batch is 11 ms lower on B.

## Output identity and quality

| Point | Equal WAV SHA-256 | WER | Similarity |
| --- | ---: | ---: | ---: |
| A c1 | reference | 1.00477% | 71.3052 |
| B c1 | 1088 of 1088 | 1.00477% | 71.3052 |
| A c16 | reference | 0.98803% | 71.2999 |
| B c16 | 197 of 1088 | 0.97128% | 71.1264 |

c1 is byte identical, which is the proof that the startup graphs replay the
kernels main replays. At c16 neither arm reproduces its own output across
boots (0 of 16 same arm matches in the earlier diagnostic), because batch
composition selects GEMM algorithms and reductions, so the c16 hash
mismatch carries no information about the change. WER and similarity are
within run to run variation.

## Verdict

Slice A does what doc 21 set out: no capture inside a serving step for the
default signature, the first requests after boot faster by the capture
cost they used to carry, steady state unchanged at c1 and better at c16,
output identical at c1. The cost is 3 s of startup.
