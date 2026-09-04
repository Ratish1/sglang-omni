# [Qwen3-TTS] Capture the code predictor CUDA graphs at startup

## Summary

The Qwen3-TTS code predictor runs 16 one-token forwards per talker token over its own five-layer KV cache, replayed through one CUDA graph per (batch bucket, sampling signature). Those graphs were captured lazily, inside the first serving step that produced each batch size, and each capture stalled the running batch by about 350 ms on H100, most of it cuDNN building an attention plan for the new shape. A single warmup request only covers batch size 1, so the other buckets kept capturing under live traffic.

This branch captures the graphs of the checkpoint's default sampling signature at startup, after SGLang's own graphs and before the stage reports ready, and cleans up the capture mechanics on the way. Steady state is unchanged: the replayed kernels are the same, the eager fallback now runs the same kernels as the graph, and the startup cost is about 3 s.

## Changes

- `_predictor_signature_terms` holds the one rule that maps a batch's sampling values to its graph signature. `prepare_decode_buffers` and the new startup path both call it.
- The eager predictor path runs the same three kernels the graph replays: the fused embedding gather, the fused seeded sampler with its ATen fallback, and the fused output projection with the residual add. The capture-only gating, the second eager sampler and its three staging fields are removed. When the talker and predictor hidden sizes match, the input projection is the identity, so the first layer's input is cloned before the fused epilogue writes into it.
- The fused output projection is used only where SGLang's linear would run torch's GEMM: not under batch-invariant mode, which overrides `aten::addmm` but not its out variant, and not when the cuteDSL bf16 backend is selected. This replaces the sm90 pin. A request under `--enable-deterministic-inference` used to fail with a CUBLAS error at this call.
- `_record_predictor_graph_failure` holds the failure accounting shared by every capture path.
- Capture moves onto the talker. `_PredictorDecodeGraph` holds the buffers, the graph and its outputs, and no reference to the talker.
- One capture stream per talker serves the warmups and the captures, as torch requires when several captures share one memory pool. The cyclic collector is off for the capture window so a graph finalizer cannot run inside a capture. The current stream is restored if `capture_end` raises, which torch's context does not do on its own.
- `resolve_subtalker_sampling` holds the subtalker sampling fallbacks used when a checkpoint ships no generation config. The request path and the builder share it.
- `Qwen3TtsEngineBuilder.setup_model_resources` derives the default signature from the merged generation defaults and calls `capture_predictor_graphs`, which captures the bucket ladder in descending order into one pool and logs `Captured 6 Qwen3-TTS predictor CUDA graphs for signature=(...) in 2.9 s`. A startup capture failure fails the boot. Batches whose signature differs from the default still capture lazily, bounded as before.

## Test results

H100, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, SGLang 0.5.18, torch 2.13.0. `pytest tests/unit_test/qwen3_tts -q`: 348 passed.

Full-corpus A/B on the seed-tts-eval English split (1088 samples), one fresh server per arm, `--seed 1234`, no warmup requests, arms alternated. A is main, B is this branch.

c1:

| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Mean latency | 0.443 s | 0.440 s | -0.003 s (-0.7%) |
| Median latency | 0.430 s | 0.427 s | -0.003 s |
| p95 latency | 0.644 s | 0.641 s | -0.003 s |
| p99 latency | 0.766 s | 0.763 s | -0.003 s |
| First request latency | | | -0.672 s |
| QPS | 2.254 | 2.270 | +0.016 (+0.7%) |
| WER | 1.01315% | 1.01315% | 0 |
| Speaker similarity | 71.30515 | 71.30515 | 0 |

c16:

| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Mean latency | 1.077 s | 1.035 s | -0.042 s (-3.9%) |
| Median latency | 1.006 s | 1.008 s | +0.002 s |
| p95 latency | 1.509 s | 1.475 s | -0.034 s (-2.3%) |
| p99 latency | 4.260 s | 1.881 s | -2.379 s (-55.9%) |
| First batch mean latency | 4.503 s | 1.992 s | -2.511 s |
| QPS | 14.778 | 15.357 | +0.579 (+3.9%) |
| WER | 0.94616% | 0.95453% | +0.008 pp |

Output identity and capture logs:

| Check | A | B |
| --- | ---: | ---: |
| c1 WAVs byte-identical to A | reference | 1088 of 1088 |
| Lazy captures inside serving steps, c1 | 1 | 0 |
| Lazy captures inside serving steps, c16 | 6 | 0 |
| Captures before ready | 0 | 6, in 2.9 s |
| Request under `--enable-deterministic-inference` | CUBLAS error | completes |
