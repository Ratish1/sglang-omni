# [Qwen3-TTS] Capture the code predictor CUDA graphs at startup

## Summary

The Qwen3-TTS code predictor runs 16 one-token forwards per talker token
over its own five-layer KV cache, replayed through one CUDA graph per
(batch bucket, sampling signature). The graphs were captured lazily, inside
the first serving step that produced each batch size, and each capture
stalled the running batch by about 350 ms on H100, most of it cuDNN
building an attention plan for the new shape. This PR captures the
graphs of the checkpoint's default sampling signature at startup, after
SGLang's own graphs and before the stage reports ready, and cleans up the
capture mechanics on the way. Eight commits, each mechanical:

1. `_predictor_signature_terms` holds the one rule that maps a batch's
   sampling values to its graph signature. `prepare_decode_buffers` and
   the new startup path both call it.
2. The eager predictor path runs the same three kernels the graph
   replays: the fused embedding gather, the fused seeded sampler with its
   ATen fallback, and the fused output projection with the residual add.
   The capture-only gating, the second eager sampler and its three staging
   fields are removed. When the talker and predictor hidden sizes match
   the input projection is the identity, so the first layer's input is
   cloned before the fused epilogue writes into it.
3. The fused output projection is used only where SGLang's linear would
   run torch's GEMM: not under batch-invariant mode, which overrides
   `aten::addmm` but not its out variant, and not when the cuteDSL bf16
   backend is selected. This replaces the sm90 pin. A request under
   `--enable-deterministic-inference` used to fail with a CUBLAS error at
   this call.
4. `_record_predictor_graph_failure` holds the failure accounting shared
   by every capture path.
5. Capture moves onto the talker. `_PredictorDecodeGraph` holds the
   buffers, the graph and its outputs and no reference to the talker.
6. One capture stream per talker serves the warmups and the captures,
   as torch requires when several captures share one memory pool. The
   cyclic collector is off for the capture window so a graph finalizer
   cannot run inside a capture. The current stream is restored if
   `capture_end` raises, which torch's context does not do on its own.
7. `resolve_subtalker_sampling` holds the subtalker sampling fallbacks
   used when a checkpoint ships no generation config. The request path
   and the builder share it.
8. `Qwen3TtsEngineBuilder.setup_model_resources` derives the default
   signature from the merged generation defaults and calls
   `capture_predictor_graphs`, which captures the bucket ladder in
   descending order into one pool and logs
   `Captured 6 Qwen3-TTS predictor CUDA graphs for signature=(...) in 2.9 s`.
   A startup capture failure fails the boot. Batches whose signature
   differs from the default still capture lazily, bounded as before.

Steady state is unchanged: the replayed kernels are the same, the eager
fallback runs the same kernels as the graph, and the startup cost is
about 3 s.

## Test results

H100, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, SGLang 0.5.18, torch 2.13.0.
`pytest tests/unit_test/qwen3_tts -q`: 348 passed.

Full-corpus A/B on the seed-tts-eval English split (1088 samples), one
fresh server per arm, `--seed 1234`, no warmup requests, arms alternated.
A is main, B is this branch.

c1:

| | Mean | Median | p95 | p99 | QPS | WER | Similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.443 s | 0.430 s | 0.644 s | 0.766 s | 2.254 | 1.01315% | 71.30515 |
| B | 0.440 s | 0.427 s | 0.641 s | 0.763 s | 2.270 | 1.01315% | 71.30515 |
| B minus A | -0.003 s | -0.003 s | -0.003 s | -0.003 s | +0.016 (+0.7%) | 0 | 0 |

c16:

| | Mean | Median | p95 | p99 | QPS | First batch mean | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1.077 s | 1.006 s | 1.509 s | 4.260 s | 14.778 | 4.503 s | 0.94616% |
| B | 1.035 s | 1.008 s | 1.475 s | 1.881 s | 15.357 | 1.992 s | 0.95453% |
| B minus A | -0.042 s (-3.9%) | +0.002 s | -0.034 s (-2.3%) | -2.379 s (-55.9%) | +0.579 (+3.9%) | -2.511 s | +0.008 pp |

Output identity and capture logs:

| | A | B |
| --- | ---: | ---: |
| c1 WAVs byte-identical across arms | 1088 of 1088 | 1088 of 1088 |
| c1 first request, B minus A | | -0.672 s |
| c1 lazy captures inside serving steps | 1 | 0 |
| c16 lazy captures inside serving steps | 6 | 0 |
| Startup captures before ready | 0 | 6, in 2.9 s |
| Request under `--enable-deterministic-inference` | CUBLAS error | completes |
