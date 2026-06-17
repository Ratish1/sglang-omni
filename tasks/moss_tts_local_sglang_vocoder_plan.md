# MOSS-TTS Local SGLang-Backed Vocoder Transformer Plan

## Objective

Improve non-streaming MOSS-TTS Local generation speed by replacing the pure
PyTorch transformer work inside the vocoder decoder with mechanically equivalent
SGLang-backed modules and kernels.

This is not a new public vocoder backend abstraction. The serving boundary stays
the current MOSS-TTS Local vocoder stage:

```text
tts_engine audio codes
  -> MossTTSLocalStreamingVocoderScheduler
  -> processor/audio_tokenizer decode path
  -> waveform response
```

The patch target is the decoder transformer internals below that boundary. The
first success condition is exact or tightly bounded waveform parity against the
current processor decode path. Performance changes only count after parity and
zero failed benchmark requests are established.

## Current Evidence

### Runtime boundary in sglang-omni

Files read:

- `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
- `sglang_omni/models/moss_tts_local/stages.py`
- `sglang_omni/models/fishaudio_s2_pro/sglang_model.py`
- `sglang_omni/vendor/sglang/layers.py`

Current behavior:

- `create_vocoder_executor(...)` loads the MOSS processor, moves
  `processor.audio_tokenizer` to the vocoder GPU, and constructs
  `MossTTSLocalStreamingVocoderScheduler`.
- The vocoder scheduler owns request aggregation, decode events, audio format
  conversion, and the final response payload.
- Non-streaming decode currently uses `_decode_codes_rows(...)`.
  - If no streaming session exists, it calls
    `processor.decode_audio_codes(codes_list)` directly.
  - If a streaming session exists, it uses `_CodecStreamSession.decode_offline`
    under the session state lock.
- Streaming decode remains tied to `codec.streaming(batch_size)` state and slot
  management. This plan does not change streaming behavior.

### Codec decoder shape

Files read:

- `local_codec.json`
- `local_codec_b8.json`
- `digest.json`

The decoder is a staged audio decoder, not one ordinary LLM block:

| stage | type | input | hidden | output | layers | heads | head dim | context |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Transformer | 768 | 1280 | 1280 | 32 | 20 | 64 | 10.0 |
| 1 | PatchedPretransform | - | - | - | - | - | - | patch 2 |
| 2 | Transformer | 640 | 768 | 768 | 12 | 12 | 64 | 10.0 |
| 3 | PatchedPretransform | - | - | - | - | - | - | patch 2 |
| 4 | Transformer | 384 | 768 | 768 | 12 | 12 | 64 | 8.0 |
| 5 | PatchedPretransform | - | - | - | - | - | - | patch 2 |
| 6 | Transformer | 384 | 768 | 768 | 12 | 12 | 64 | 4.0 |
| 7 | PatchedPretransform | - | - | - | - | - | - | patch 2 |
| 8 | Transformer | 384 | 768 | 768 | 12 | 12 | 64 | 2.0 |
| 9 | PatchedPretransform | - | - | - | - | - | - | patch 2 |
| 10 | Transformer | 384 | 768 | 240 | 12 | 12 | 64 | 1.0 |
| 11 | PatchedPretransform | - | - | - | - | - | - | patch 240 |

Totals:

- 12 decoder stages
- 6 projected transformer stages
- 6 reshape-only patch stages
- 92 transformer layers
- LayerNorm, not RMSNorm
- plain GELU FFN, not gated GEGLU/SwiGLU
- RoPE positional encoding
- local causal attention windows from the stage `context_duration`

Important remote-module semantics:

- `MossAudioTokenizerProjectedTransformer.forward(...)` transposes
  `(B, D, T) -> (B, T, D)`, applies input projection, packs padded sequences
  for flash attention, runs the transformer, unpacks, applies output
  projection, then transposes back.
- That path includes per-stage Python synchronizations:
  `input_lengths.any().item()` and `input_lengths.max().item()`.
- `MossAudioTokenizerTransformerLayer.forward(...)` is:
  - LayerNorm
  - self-attention
  - learned layer scale on attention output
  - residual add
  - LayerNorm
  - `Linear -> GELU -> Linear`
  - learned layer scale on FFN output
  - residual add
- `MossAudioTokenizerMultiheadAttention._forward_non_streaming_flash(...)`:
  - projects packed QKV
  - applies packed RoPE using `position_ids`
  - calls flash attention with packed sequence metadata
  - reshapes back to packed hidden dimension
- `MossAudioTokenizerPatchedPretransform.decode(...)` is reshape/permutation:
  `(B, D*patch, L) -> (B, D, L*patch)`.

### Probe results

Current processor decode and offline session decode are numerically identical in
the captured probes, but the offline session wrapper is not a performance win:

| batch | frames | processor mean ms | session mean ms | max abs delta |
|---:|---:|---:|---:|---:|
| 1 | 25 | 79.717 | 111.375 | 0.0 |
| 1 | 100 | 100.260 | 108.663 | 0.0 |
| 1 | 300 | 269.980 | 284.825 | 0.0 |
| 8 | 100 | 158.213 | 194.695 | 0.0 |
| 8 | 300 | 486.572 | 484.792 | 0.0 |

The streaming-session offline lane is therefore useful as a parity reference,
not as the optimization target.

### Invalidated experiment

The direct SGLang attention-function patch is invalidated and must not be
reintroduced:

| case | completed / failed | qps | rtf mean | latency mean | p95 | p99 | output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| processor baseline | 1088 / 0 | 4.925 | 0.3881 | 1.622 | 2.152 | 2.716 | 271.0 |
| direct sglang patch | 1084 / 4 | 2.295 | 0.8403 | 3.480 | 4.085 | 6.967 | 126.4 |

That patch changed a low-level attention call while leaving the remote PyTorch
control flow, packing, syncs, allocation pattern, and layer loop in place. It
also introduced failures. The new plan must own the transformer implementation
before replacing internals with SGLang primitives.

### SGLang source findings

Files read from `/Users/ratish/sglang`:

- `python/sglang/jit_kernel/flash_attention.py`
- `python/sglang/srt/layers/attention/vision.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/models/registry.py`

Relevant reusable pieces:

- `sglang.jit_kernel.flash_attention.flash_attn_varlen_func(...)` supports
  packed varlen QKV, `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`,
  `max_seqlen_k`, `causal`, and `window_size`.
- `VisionAttention` is the closest SGLang precedent for full-sequence,
  no-KV-cache multimodal attention. It uses SGLang QKV/projection layers and
  a selected QKV backend, but it is not a drop-in MOSS layer.
- `ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`, and
  `QKVParallelLinear` can be used once weight loading and tensor-parallel
  semantics are proven for this decoder. For TP=1, correctness can be proven
  first before adding sharded loading.
- `ModelRegistry` supports external model packages, but the normal SGLang model
  runner expects LLM-style `input_ids`, positions, and `ForwardBatch`.

Pieces that do not semantically match MOSS vocoder:

- `RadixAttention` is KV-cache serving attention and requires `ForwardBatch`.
  The MOSS non-streaming vocoder is a dense/packed decoder over codec frames,
  not a token decode loop with SGLang scheduler-owned KV cache.
- SGLang RMSNorm and fused residual RMSNorm kernels do not match MOSS
  LayerNorm.
- `SiluAndMul`, `GeluAndMul`, and gated activation kernels do not match the
  MOSS plain `Linear -> GELU -> Linear` FFN.
- Forcing the whole vocoder through `ModelRunner.forward(input_ids, positions,
  forward_batch, ...)` would introduce a fake token-serving interface before
  the decoder mechanics are proven. That should not be Phase 1.

## Correct Patch Boundary

The clean boundary is not a top-level `processor/session/sglang` backend switch.

The clean boundary is:

```text
MOSS vocoder scheduler and audio response code
  unchanged

MOSS audio tokenizer decode frame setup
  initially reused for codec/code embedding and waveform plumbing

MOSS decoder transformer chain
  replaced with an owned, parity-tested implementation
  then optimized stage by stage with SGLang kernels/layers
```

The implementation should use names that describe MOSS decoder mechanics, not a
generic backend abstraction:

- `MossTTSLocalVocoderDecoder`
- `MossTTSLocalProjectedTransformer`
- `MossTTSLocalTransformerLayer`
- `MossTTSLocalAttention`
- `MossTTSLocalPatchTransform`

Avoid names like:

- `CodecDecoderContract`
- `ProcessorDecodeBackend`
- `SessionDecodeBackend`
- `SGLangCodecBackend`

The first implementation should live under
`sglang_omni/models/moss_tts_local/`, because the current work is a MOSS-TTS
Local integration patch. If the owned implementation proves useful and clean, we
can later split reusable pieces into the SGLang source tree or a vendor wrapper.

## Mechanical Invariants

These invariants must be enforced by construction-time validation and parity
tests, not by hardcoded assumptions in the hot path:

1. Stage topology must be read from the loaded remote model modules or config.
2. Every projected transformer stage must preserve:
   - input projection weights and bias
   - output projection weights and bias
   - transformer layer order
   - LayerNorm weights and bias
   - layer-scale tensors
   - QKV fused projection layout
   - output projection layout
   - FFN `Linear -> GELU -> Linear` layout
   - RoPE settings
   - causal/local attention window settings
3. Patch stages must preserve exact reshape/permutation behavior and length
   updates.
4. Packed and dense paths must produce the same output shape as the processor
   path:
   - stage input: `(B, D, T)`
   - transformer input: `(B, T, D)`
   - packed attention input: `(total_valid_tokens, D)`
   - stage output: `(B, D_out, T_out)`
5. Non-streaming decode must not use streaming chunk emission or streaming
   state mutation.
6. Any SGLang kernel substitution must have an exact semantic match or remain
   out of scope.

## Execution Plan

### Phase 0: Source extraction and parity harness

Goal: make the decoder mechanics inspectable and testable before writing the
replacement module.

Tasks:

1. Add a development-only extractor that records the loaded decoder topology:
   - stage index and type
   - projection dimensions
   - layer count
   - head count and head dim
   - FFN size
   - context window
   - attention implementation choice
   - tensor dtypes and devices
2. Extract the source or bytecode-derived behavior for the functions we must
   reproduce:
   - `_decode_frame` call shape
   - projected transformer forward
   - transformer forward
   - layer forward
   - attention packed flash path
   - RoPE helper
   - patch transform encode/decode
3. Add a local H100 parity script that compares:
   - processor decode output
   - current session offline decode output
   - future owned decoder output
4. Run probes for:
   - batch 1, frames 25
   - batch 1, frames 100
   - batch 1, frames 300
   - batch 8, frames 100
   - batch 8, frames 300

Exit criteria:

- Extractor output matches the known 12-stage, 92-layer topology.
- Processor and session outputs remain `max_abs_delta == 0.0` in the probe set.
- No production serving code changes yet.

### Phase 1: Owned PyTorch-equivalent decoder

Goal: own the decoder transformer chain without changing numerics or serving
behavior.

Tasks:

1. Add a MOSS-specific decoder module under
   `sglang_omni/models/moss_tts_local/`, for example
   `vocoder_decoder.py`.
2. Build module objects from the loaded remote decoder modules, not from
   hardcoded constants:
   - copy or reference weights from the loaded `audio_tokenizer.decoder`
   - validate the stage types and dimensions
   - fail fast on unknown stage shapes
3. Implement:
   - projected transformer stage
   - transformer layer
   - packed attention path
   - dense fallback path only if the loaded processor can select it
   - patch transform stage
4. Keep plain PyTorch math first:
   - `torch.nn.functional.linear`
   - `torch.nn.functional.layer_norm`
   - `torch.nn.functional.gelu`
   - current flash attention path only if it is exactly reproduced
5. Wire this only behind an internal experimental constructor option or local
   dev flag, not as a public backend API.

Exit criteria:

- Owned decoder produces matching waveform for the Phase 0 probe set.
- Full 1088 generate-only run has 0 failures.
- RTF is not used as the success metric yet; this phase proves ownership and
  correctness.
- Existing streaming tests still pass.

### Phase 2: Replace attention internals with SGLang varlen attention

Goal: use SGLang attention primitives inside the owned MOSS attention module,
with exact MOSS semantics.

Tasks:

1. Generate packed sequence metadata once per stage input:
   - valid mask
   - packed tensor
   - `cu_seqlens`
   - `position_ids`
   - max valid sequence length
2. Remove per-layer remote-module packing and Python syncs from the hot path.
3. Implement QKV projection and RoPE in the owned attention module.
4. Call `sglang.jit_kernel.flash_attention.flash_attn_varlen_func(...)` with:
   - packed QKV
   - identical `cu_seqlens_q` and `cu_seqlens_k`
   - stage max sequence length
   - `causal=True`
   - `window_size=(left_context, 0)` when the MOSS local context is finite
   - matching softmax scale
5. Prove that the local causal window interpretation matches MOSS SDPA and
   flash paths. Do not assume the same `context_duration` unit until it is
   traced to frame positions.

Exit criteria:

- Stage-level parity for every transformer stage.
- End-to-end waveform parity within accepted tolerance.
- No failed requests on 1088 generate-only.
- RTF must beat Phase 1 and be competitive with processor baseline before this
  is promoted beyond experiment.

### Phase 3: Adopt SGLang linear layers where they help

Goal: use SGLang linear/loading patterns only after attention parity is stable.

Tasks:

1. Try TP=1 SGLang layer wrappers first:
   - `ReplicatedLinear` for simple projection if useful
   - `ColumnParallelLinear` for fused QKV
   - `RowParallelLinear` for output projection
2. Write explicit weight loaders from MOSS remote module names:
   - `self_attn.in_proj.weight` -> Q/K/V shards
   - `self_attn.out_proj.weight` -> output projection
   - `ffn.0.weight`, `ffn.2.weight`
3. Do not use tensor parallelism until TP=1 parity and performance are known.
4. Keep PyTorch LayerNorm and plain GELU unless an exact SGLang or sgl-kernel
   equivalent is found and proven.

Exit criteria:

- TP=1 parity preserved.
- Weight loading has clear assertions for shape, dtype, and device.
- Performance improves or this phase is reverted.

### Phase 4: Static-shape runtime optimization

Goal: reduce launch overhead and allocation churn after the math path is owned.

Tasks:

1. Bucket by batch size and frame count:
   - batch buckets: 1, 2, 4, 8
   - frame buckets: current decode chunk sizes, including 100 and 300 probe
     shapes
2. Preallocate scratch tensors for:
   - packed buffers
   - valid masks
   - position ids
   - stage outputs
3. Consider CUDA graphs only for shapes that are stable and already parity
   tested.
4. Consider `torch.compile` only around owned blocks where it does not hide
   unsupported graph breaks or increase startup cost in the benchmark.

Exit criteria:

- No regression in cold-start behavior that invalidates the benchmark.
- 1088 generate-only run improves over processor baseline.
- WER run remains within expected noise.

### Phase 5: Production integration

Goal: make the optimized path the internal non-streaming decoder path only
after correctness and speed are proven.

Tasks:

1. Replace the non-streaming processor decoder internals with the owned
   decoder when capability validation succeeds.
2. Keep a narrow fallback to the processor path for unsupported model shapes.
3. Keep streaming decode on the existing streaming session implementation.
4. Add structured logging once at startup:
   - decoder path selected
   - stage topology
   - attention implementation
   - graph/compile mode if enabled
5. Remove development-only extractor/logging before PR unless it is explicitly
   useful as a test utility.

Exit criteria:

- Full SeedTTS generate+WER run passes.
- No extra repeated hot-path logging.
- The code reads as a MOSS decoder implementation, not a generic experimental
  backend framework.

## Validation Plan

### Unit and parity checks

Minimum local/unit checks:

- Existing `tests/unit_test/moss_tts_local/test_streaming_vocoder.py`
- Shape validation for the 12-stage decoder topology
- Patch transform encode/decode shape and length behavior
- Weight-loading shape assertions for every projected transformer stage

H100 parity checks:

- Synthetic code rows:
  - batch 1, frames 25
  - batch 1, frames 100
  - batch 1, frames 300
  - batch 8, frames 100
  - batch 8, frames 300
- Real SeedTTS code rows sampled from the 1088 run.
- Compare:
  - waveform shape
  - `max_abs_delta`
  - `mean_abs_delta`
  - SNR
  - audio length

### Performance checks

Run generate-only first:

- model: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- dataset: `zhaochenyang20/seed-tts-eval-arrow`
- language: `en`
- samples: 1088
- concurrency: 8
- warmup: 1
- non-streaming

Then run full generate+WER:

- same generation settings
- ASR: `Qwen/Qwen3-ASR-1.7B`
- preserve WER within expected run-to-run noise

Required comparison table for any PR:

| branch | completed | failed | qps | rtf mean | rtf median | p95 latency | p99 latency | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream/main | | | | | | | | |
| optimized branch | | | | | | | | |

## Risk Register

| risk | impact | mitigation |
|---|---|---|
| Flash attention local-window semantics differ from MOSS mask | audio drift or WER regression | stage-level parity against SDPA and processor path before integration |
| LayerNorm/GELU replaced with non-equivalent SGLang kernels | mechanical correctness bug | do not use RMSNorm or gated activation kernels |
| Fake SGLang `ForwardBatch` integration adds scheduler overhead | worse latency | stay inside MOSS vocoder stage until an owned decoder is proven |
| CUDA graph capture hides shape or state mutation bugs | intermittent failures | graph only after static bucket parity |
| Weight layout mismatch for fused QKV | severe audio corruption | explicit per-layer shape checks and output parity |
| Existing streaming behavior changes accidentally | user-facing regression | keep streaming session path unchanged and run streaming vocoder tests |

## Immediate Next Steps

1. Implement Phase 0 extractor/parity harness if the existing local JSON is not
   enough for code generation.
2. Implement Phase 1 owned PyTorch-equivalent decoder with no performance claim.
3. Run probe parity on H100.
4. Only then replace attention internals with SGLang varlen attention.

Do not reintroduce:

- direct monkeypatches of remote `flash_attn_varlen_func`
- a public `processor/session/sglang` backend selector
- hardcoded decoder shape files that pretend to define model semantics
- streaming chunk semantics in the non-streaming path
- RMSNorm or gated activation kernels for this LayerNorm/GELU model
