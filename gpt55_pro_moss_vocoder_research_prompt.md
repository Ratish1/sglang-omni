# gpt-5.5 pro research prompt: best production design for moss-tts local vocoder acceleration

You are acting as a senior ML systems architect reviewing how to accelerate the MOSS-TTS Local non-streaming vocoder in SGLang-Omni. Treat this as an independent research and design task. Do not assume the current implementation direction is correct. Derive the best strategy from first principles, code inspection, kernel semantics, production risk, and empirical evidence.

The primary question:

> What is the best production-quality way to make the MOSS-TTS Local vocoder faster by reusing or patching SGLang internals, while preserving the correct vocoder semantics and making the implementation maintainable?

## Environment And Repositories

- SGLang-Omni repository: `/Users/ratish/sglang-omni`
- Experimental SGLang-Omni worktree: `/Users/ratish/sglang-omni/.worktrees/moss-vocoder-monkey-patch`
- Experimental branch: `perf/moss-vocoder-monkey-patch`
- Current commit with prompt: check `git rev-parse HEAD`
- Local SGLang repository: `/Users/ratish/sglang`
- Target model: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- Target use case: non-streaming MOSS-TTS generation, concurrency around 8, H100 GPU

## Research Objective

Investigate all reasonable acceleration strategies and recommend the best one. Do not restrict yourself to the currently attempted monkey-patch or owned-decoder approaches. Consider:

- preserving the Hugging Face remote MOSS vocoder implementation and changing only backend hooks
- adding a narrow adapter around SGLang attention or sgl-kernel functions
- patching SGLang core with a reusable helper if that is architecturally justified
- optimizing the exact SDPA/cuDNN path rather than forcing FlashAttention
- owning a small amount of model-specific code in SGLang-Omni if it is the cleanest boundary
- accepting small numerical/audio drift only if that is a defensible product decision with objective quality gates
- rejecting an approach entirely if it creates correctness risk or poor maintainability

## Files To Read Completely

Read the relevant files fully. Segment large files if needed, but do not rely on small snippets only.

### SGLang-Omni

- `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
- `sglang_omni/models/moss_tts_local/vocoder_sglang_patch.py`
- `sglang_omni/models/moss_tts_local/vocoder_decoder.py`
- `sglang_omni/models/moss_tts_local/vocoder_introspection.py`
- `benchmarks/eval/inspect_moss_tts_local_vocoder.py`
- `tests/unit_test/moss_tts_local/test_vocoder_sglang_patch.py`
- `tests/unit_test/moss_tts_local/test_vocoder_decoder.py`
- `tests/unit_test/moss_tts_local/test_streaming_vocoder.py`
- `tasks/` files related to MOSS vocoder planning if present

### SGLang

- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v3.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v4.py` if present
- `/Users/ratish/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`
- `/Users/ratish/sglang/python/sglang/srt/layers/attention/vision.py`
- Any local SGLang attention, kernel, or multimodal files that clarify how production code selects FA2, FA3, FA4, FlashInfer, SDPA, cuDNN SDPA, or sgl-kernel paths

### MOSS Remote Code

Read the real Hugging Face remote code for `MOSS-Audio-Tokenizer-v2` if available in cache. If it is not available, use the local introspection JSON artifacts in the repository root, including any of:

- `vocoder.json`
- `vocoder_custom.json`
- `local_codec.json`
- `local_codec_b8.json`
- `digest.json`

Focus on:

- decoder stage structure
- projected transformer stages
- attention implementation selection
- non-streaming vs streaming decode differences
- RoPE implementation and positional semantics
- packed sequence path
- SDPA path
- FlashAttention path
- cache/state behavior
- dtype and tensor layout contracts

## Known Empirical Observations

Treat these as observations to explain, not conclusions to accept.

### Baseline and optimized benchmark context

The broad serving goal is to reduce MOSS-TTS Local real-time factor for non-streaming generation, ideally below `0.3`, without changing generated audio semantics unless explicitly accepted.

Representative strong full-run result from a custom owned decoder direction:

| metric | value |
|---|---:|
| completed | 1088 |
| failed | 0 |
| qps | about 6.42 |
| rtf_mean | about 0.294 |
| latency_mean_s | about 1.24 |
| output_throughput | about 353 tok/s |

However, this path later failed strict waveform parity stress tests.

### Owned decoder stress result

A repo-owned decoder replacement was much faster in some cases but not waveform-exact versus `processor.decode_audio_codes`:

| case | batch | total frames | max frames | processor ms | owned ms | owned max delta |
|---|---:|---:|---:|---:|---:|---:|
| single_short | 1 | 25 | 25 | 76.546 | 51.418 | 0.0732422 |
| single_typical | 1 | 100 | 100 | 83.922 | 50.836 | 0.123779 |
| single_long | 1 | 300 | 300 | 225.708 | 137.256 | 1.42578 |
| mixed_8 | 8 | 1025 | 300 | 754.969 | 331.375 | 0.941406 |
| mixed_16 | 16 | 2449 | 400 | 2005.005 | 766.758 | 1.53906 |

### Fused RoPE result

An experimental fused RoPE path reduced time inside the attention path but produced nonzero waveform deltas:

| probe | max_abs | mean_abs | snr |
|---|---:|---:|---:|
| bs=1, frames=25 | 0.06543 | 0.000906 | 31.80 dB |
| bs=1, frames=100 | 0.01807 | 0.000525 | 37.09 dB |
| bs=8, frames=100 | 0.03397 | 0.000543 | 34.78 dB |

### Monkey-patch result

An experiment preserved the Hugging Face processor/vocoder path and patched remote module globals so the MOSS FlashAttention hook used SGLang:

```python
module.flash_attn_varlen_func = sglang.jit_kernel.flash_attention.flash_attn_varlen_func
module.HAS_FLASH_ATTN = True
```

The patch installed successfully and found `attention_modules=92`, but waveform parity was not exact and longer/mixed stress cases slowed down:

| case | processor ms | sglang patch ms | max_abs_delta |
|---|---:|---:|---:|
| 1x25 | 173.240 | 104.982 | 0.0537109 |
| 1x100 | 164.564 | 100.018 | 0.0488281 |
| 1x300 | 225.327 | 278.221 | 0.224854 |
| 8x100 | 194.551 | 206.353 | 0.116211 |
| 8x300 | 547.350 | 607.493 | 0.573242 |
| single_short | 79.697 | 100.816 | 0.0732422 |
| single_typical | 85.374 | 102.283 | 0.123779 |
| single_long | 244.350 | 287.124 | 0.202881 |
| mixed_8 | 758.759 | 1064.330 | 0.513184 |
| mixed_16 | 1995.862 | 3716.965 | 0.674805 |

## Key Research Questions

Answer these independently from the code and from external ML systems knowledge where useful.

1. What is the actual MOSS vocoder computation graph for non-streaming decode?
   - Which parts are transformer attention?
   - Which parts are projection, RoPE, normalization, feed-forward, upsampling, convolution, quantizer/dequantizer, or post-processing?
   - Which parts are likely bottlenecks under batch size 1, mixed lengths, and concurrency 8?

2. What is the exact semantic contract of the current correct processor path?
   - Does it use SDPA, cuDNN SDPA, upstream flash-attn, SGLang FA3/FA4, or something else in the observed environment?
   - Are outputs expected to be bit-exact across attention backends?
   - If bit-exactness is unrealistic, what quality gate is appropriate for TTS audio?

3. Is a monkey patch a valid production strategy here?
   - If yes, what must it patch exactly?
   - Should it patch module globals, class methods, a narrow adapter, or model construction?
   - How should lifecycle, thread safety, process-global mutation, rollback, and testability be handled?
   - Is there a safer alternative to monkey patching while still avoiding a full decoder reimplementation?

4. Is SGLang `flash_attn_varlen_func` semantically compatible with the Hugging Face remote-code expectation?
   - Compare function signatures, defaults, accepted kwargs, and behavior with upstream `flash_attn.flash_attn_varlen_func`.
   - Inspect FA2 vs FA3 vs FA4 selection.
   - Inspect `window_size`, `causal`, bottom-right mask alignment, `softmax_scale`, `softcap`, dropout, deterministic behavior, `return_attn_probs`, `return_softmax_lse`, `num_splits`, `pack_gqa`, and dtype accumulation.
   - Explain whether an adapter can make the SGLang path remote-code-compatible.

5. How should we isolate the source of waveform deltas?
   - What tensor-level probes should be added?
   - Should we compare attention outputs before waveform generation?
   - Should we run each backend in a fresh process?
   - Should we compare SDPA vs cuDNN SDPA vs upstream FA2 vs SGLang FA3/FA4 on the same captured `q/k/v/cu_seqlens`?
   - What exact commands or scripts should be used?

6. What is the best next implementation strategy?
   - Continue with a cleaner monkey patch?
   - Replace the SGLang wrapper with an upstream-compatible adapter?
   - Add a SGLang core helper?
   - Optimize exact SDPA/cuDNN path?
   - Keep owned decoder but repair semantic drift?
   - Drop strict bit-exactness and use objective audio-quality gates?

7. What would a top-tier production codebase do?
   - How would it stage experiments?
   - What would it ship behind flags?
   - What would it reject?
   - What tests would block merge?
   - What telemetry/profiling would it require?

## Required Output

Give a rigorous answer with the following structure:

1. **Executive Recommendation**
   - One clear recommended direction
   - Whether the current monkey patch should be kept, modified, or discarded
   - Whether the owned decoder direction should be kept, modified, or discarded

2. **System Mechanics**
   - Explain the MOSS vocoder path and where SGLang can realistically help
   - Identify the exact backend semantics that matter

3. **Findings**
   - Use severity labels for correctness, performance, maintainability, and process risks
   - Reference files and line numbers where possible
   - Separate observed facts from hypotheses

4. **Experiment Plan**
   - Minimal experiments in priority order
   - Exact measurements and pass/fail criteria
   - What result would prove or disprove each strategy

5. **Implementation Plan**
   - Concrete code design
   - File boundaries
   - API or env flag shape
   - Test plan
   - Rollback behavior

6. **Final Decision Matrix**
   - Compare monkey patch, adapter, owned decoder, SGLang core helper, SDPA optimization, and approximate-quality paths across correctness, speed, maintainability, and risk
