# MOSS-TTS Local Streaming SGLang Vocoder Plan

## Goal

Support the MOSS-TTS Local streaming vocoder with SGLang-backed attention while keeping the merged non-streaming packed vocoder path clean, reusable, and mechanically correct.

This is not a request to make streaming and non-streaming share one giant decoder class. The important unification is at the right level:

- shared MOSS attention math utilities;
- shared route/profiling/validation language;
- separate execution engines for the two different runtime contracts.

## Success Criteria

Functional:

- Streaming requests preserve current chunk semantics: initial threshold, steady threshold, join/coalescing behavior, final flush, abort cleanup, and slot reuse.
- Streaming output preserves the codec streaming contract: one persistent `codec.streaming(batch_size)` session, stable slot ownership, `exec_mask`-guarded state advancement, and reset masks for released slots.
- Non-streaming pure requests continue to use the merged packed SGLang full-sequence path when no live streaming session exists.
- Non-streaming requests that arrive while a streaming session is live continue to use offline slots inside that session, not nested codec streaming.

Quality:

- No NaNs, shape mismatches, empty audio, silent audio, or duration runaway.
- Streaming eager baseline and streaming SGLang candidate are compared on real MOSS-Audio-Tokenizer-v2, not only fake modules.
- Per-attention oracle deltas are captured before any full benchmark claim.
- If bit exactness is not possible under FlashAttention, the PR documents the numerical boundary and proves real-audio quality is unchanged within an agreed gate.

Performance:

- Streaming latency improves or stays neutral for normal chunk sizes.
- CUDA graph capture remains available for the streaming codec path, or we explicitly prove that eager SGLang streaming is faster enough to justify graph changes.
- No Python-side per-token or per-slot hot-loop growth beyond existing scheduler bookkeeping.

Code quality:

- No hidden runtime fallback spaghetti in hot paths.
- Capability decisions happen at construction or a named backend boundary.
- Shared helpers are only extracted when both streaming and non-streaming actually use them.
- SGLang kernels are reused through their public Python APIs; no vendored or duplicated kernel code in Omni unless we later choose a separate kernel PR.

## Current-State Evidence

### Existing Streaming Flow

Files:

- [streaming_vocoder.py](/Users/ratish/sglang-omni/sglang_omni/models/moss_tts_local/streaming_vocoder.py)
- [vocoder_cuda_graph.py](/Users/ratish/sglang-omni/sglang_omni/models/moss_tts_local/vocoder_cuda_graph.py)
- [stages.py](/Users/ratish/sglang-omni/sglang_omni/models/moss_tts_local/stages.py)
- [test_streaming_vocoder.py](/Users/ratish/sglang-omni/tests/unit_test/moss_tts_local/test_streaming_vocoder.py)

Current streaming execution:

```text
AR frame decode
  -> StreamItem row [text_token, code_0, ..., code_n]
  -> MossTTSLocalStreamingVocoderScheduler
     -> _LocalStreamState pending frame rows
     -> _CodecStreamSession persistent codec.streaming(B)
        -> step(slot -> [n_vq, T])
           -> dense codes_step [n_vq, B_full, T]
           -> exec_mask [B_full]
           -> MossVocoderCudaGraphRunner.decode_step(...) if captured
           -> codec._decode_frame(codes_step, codes_lengths) otherwise
           -> per-slot CPU float32 audio chunks
```

Key streaming invariants:

| Invariant | Why it matters |
| --- | --- |
| One persistent codec streaming session | The codec owns incremental causal state and offsets. |
| Stable slot per request | Cached key/value/state belongs to a slot across chunks. |
| `exec_mask` gates advancement | Inactive slots must not advance state during coalesced steps. |
| Reset masks on release/abort | Slot reuse must not leak previous request state. |
| Uniform `T` per `step()` | Existing batching and CUDA graphs are exact-frame-count buckets. |
| Offline slots during live streaming | Avoid illegal nested `codec.streaming()` contexts. |

Existing CUDA graph mechanics:

- Captures one graph per exact frame count `T`.
- Fixed full slot batch width.
- Captures largest `T` first for shared graph pool address stability.
- Patches codec attention cache updates to in-place writes for graph pointer stability.
- Falls back to eager on uncaptured `T`, low VRAM, capture failure, or replay failure.

### Existing Non-Streaming Packed SGLang Flow

Files:

- [vocoder_decoder.py](/Users/ratish/sglang-omni/sglang_omni/models/moss_tts_local/vocoder_decoder.py)
- [test_vocoder_decoder.py](/Users/ratish/sglang-omni/tests/unit_test/moss_tts_local/test_vocoder_decoder.py)

Current non-streaming packed execution:

```text
Pure non-streaming request, no live streaming session
  -> _decode_codes_rows_nonstream(...)
     -> dense audio_codes [n_vq, B, max_T]
     -> padding_mask [B, max_T]
     -> temporarily replace codec.decoder with MossTTSLocalVocoderDecoder
     -> codec.decode(..., chunk_duration=None)
        -> projected transformer stage [B, C, T]
        -> pack valid frames to [sum_T, E]
        -> MOSS interleaved RoPE
        -> sglang.flash_attn_varlen_func(...)
        -> unpack to [B, T, E]
        -> upstream output projection/pretransform/waveform projection
```

Reusable pieces:

| Piece | Reusable for streaming? | Notes |
| --- | --- | --- |
| MOSS local causal window conversion | Yes | `context` includes current token; FA left window is `context - 1`. |
| MOSS interleaved RoPE reference/cache | Yes, with care | Prior SGLang fused RoPE regressed quality; keep MOSS math unless proven. |
| SGLang FlashAttention import/capability boundary | Yes | Streaming may use varlen or kvcache APIs. |
| Per-call route/profile names | Yes | Needed for proof and profile comparability. |
| Full-sequence packed decoder | No | Stateless full sequence, not a live streaming cache engine. |
| Pack/unpack full `[B,T,E]` by `input_lengths` | Partially | Useful for offline/full chunks, but streaming state is cache-based. |

### SGLang Kernel Surface

Relevant SGLang files read:

- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v3.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v4.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/rope.py`
- `/Users/ratish/sglang/python/sglang/srt/layers/attention/triton_backend.py`

Relevant kernel APIs:

| API | Fit for MOSS non-streaming | Fit for MOSS streaming | Notes |
| --- | --- | --- | --- |
| `flash_attn_varlen_func` | Strong | Possible for packed active-slot context | Already used in non-streaming. Needs explicit packed K/V context for streaming. |
| `flash_attn_with_kvcache` | Not needed | Strong candidate | Updates K/V cache and attends in one kernel. Must match cache layout and streaming positions exactly. |
| FA4 varlen | Possible | Limited | FA4 kvcache wrapper does not support in-place KV updates or rotary embedding. |
| SGLang fused RoPE | Risky | Risky | Prior MOSS vocoder RoPE attempt improved speed but hurt WER. Must be a later isolated proof. |
| SGLang LLM attention backends | Not directly | Not directly | Tied to `ForwardBatch`, KV pools, radix/cache metadata, TP, LoRA, speculative paths. |

SGLang’s own sliding-window rule also matches the conversion we already use: when a layer has finite sliding window, the effective window passed to flash-style kernels is derived from the visible key count and usually maps causal current-token-inclusive windows to left-window counts.

### Research Inputs Integrated

Three read-only code research passes were used:

| Area | Finding |
| --- | --- |
| Streaming MOSS vocoder | `_CodecStreamSession` is the critical owner. It owns `codec.streaming(B)`, slots, offline lanes, `exec_mask`, slot resets, and CUDA graph replay. Any SGLang streaming path must preserve this ownership. |
| Packed non-streaming vocoder | The merged decoder replaces only projected transformer self-attention in full-sequence decode. It intentionally does not model streaming cache state. |
| SGLang/SGLang-Omni analogues | Fish/Higgs provide scheduler patterns, but MOSS should keep its stateful codec session design. SGLang LLM attention internals are too tied to `ForwardBatch`/KV pools/radix state to copy wholesale. |

One important negative finding: a source `chunked_local_attention.py` helper is not present as normal source in the current `/Users/ratish/sglang/python/sglang/jit_kernel` checkout, so the plan should not rely on a local patched helper as a required dependency. The stable reusable surface is the public SGLang FlashAttention API.

## Current Flow Diagram

```text
                               MOSS-TTS Local AR server
                                         |
                                         v
                            rows [text_token, code_0..code_n]
                                         |
                         +---------------+----------------+
                         |                                |
                         v                                v
              streaming request                    pure non-stream request
                         |                                |
                         v                                v
          _CodecStreamSession.step(...)        _decode_codes_rows_nonstream(...)
                         |                                |
             persistent codec.streaming(B)                |
                         |                                v
             codec._decode_frame(...)          codec.decode(chunk_duration=None)
                         |                                |
               upstream streaming attention     MossTTSLocalVocoderDecoder
                         |                                |
              CUDA graph or eager SDPA          packed SGLang varlen FA
                         |                                |
                         v                                v
                    audio chunks                     full waveform
```

Important boundary: streaming currently depends on `codec._decode_frame()` and the live `codec.streaming()` state. The non-streaming decoder bypasses that state by running full-sequence decode.

## Proposed Architecture

```text
sglang_omni/models/moss_tts_local/

  vocoder_attention.py              (new, only when we implement shared use)
    - moss_flash_window(context, causal) -> tuple[int, int]
    - MossRopeCache / apply_moss_rope(...)
    - SGLangFlashAttention binding helpers
    - route/profile naming constants only if reused

  vocoder_decoder.py
    - non-stream packed full-sequence decoder
    - consumes shared attention helpers
    - no streaming state awareness

  streaming_vocoder.py
    - request lifecycle, thresholds, slots, offline lanes
    - no low-level attention kernels
    - selects streaming codec attention backend at construction

  vocoder_streaming_attention.py    (new, implementation phase only)
    - streaming attention adapter around codec attention modules
    - owns q/k/v packing or kvcache call bridge
    - preserves source streaming cache contract

  vocoder_cuda_graph.py
    - graph capture/replay for streaming _decode_frame
    - remains responsible for pointer-stability patching
```

Target streaming flow:

```text
StreamItem rows
  -> MossTTSLocalStreamingVocoderScheduler
     -> _CodecStreamSession.step(...)
        -> codec._decode_frame(...)
           -> source codec transformer layer
              -> Moss streaming attention adapter
                 -> source qkv projection
                 -> source/MOSS-compatible RoPE
                 -> SGLang kvcache or packed varlen FlashAttention
                 -> source output projection
        -> CUDA graph replay if captured, eager otherwise
        -> audio chunks
```

Target non-streaming flow stays separate:

```text
Non-stream payload and no live session
  -> codec.decode(chunk_duration=None)
     -> MossTTSLocalVocoderDecoder
        -> packed full-sequence SGLang varlen FlashAttention
     -> full waveform
```

### Why Two Engines Stay Separate

| Dimension | Non-stream packed decoder | Streaming vocoder |
| --- | --- | --- |
| Input shape | Whole request `[B,T]` code matrix | Incremental chunks per slot |
| State | Stateless full-sequence decode | Persistent codec streaming state |
| Cache ownership | None | Codec attention modules own `_streaming_state` |
| Backend target | Projected transformer stage | Individual attention modules inside `_decode_frame` |
| Correctness target | Quality/tolerance vs upstream | Chunk continuity, slot isolation, reset correctness |
| CUDA graph | Not required for vocoder decode | Existing exact-`T` graph buckets matter |

A single class would hide these differences and create more invalid states. The correct unification is shared attention primitives and shared validation, not shared execution ownership.

## Streaming SGLang Design Options

### Option A: `flash_attn_with_kvcache` Streaming Adapter

Use SGLang’s kvcache API for each codec attention module.

```text
source attention forward
  -> project q/k/v for current chunk [B,T,H,D]
  -> apply MOSS RoPE with absolute streaming positions
  -> write k/v into per-slot cache
  -> flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, ...)
  -> output projection
```

Pros:

- Mechanically closest to streaming decode.
- Can avoid rebuilding packed K/V context per step.
- SGLang API explicitly supports incremental K/V update plus local window.

Risks:

- Requires exact cache layout alignment with remote MOSS codec attention state.
- FA4 path does not support in-place KV update or rotary, so FA3/sgl-kernel constraints matter.
- CUDA graph compatibility depends on static cache tensors and no dynamic allocation in captured path.

This is the preferred implementation direction after tracing proves the per-call context semantics.

### Option B: Active-Slot Packed Varlen Streaming Adapter

Build per-step packed Q and packed K/V context for active slots, then call `flash_attn_varlen_func`.

```text
active slots only
  -> gather cache window + new keys for each slot
  -> build packed q/k/v + cu_seqlens
  -> flash_attn_varlen_func(...)
  -> scatter outputs to full batch
  -> update source cache state
```

Pros:

- Reuses the non-streaming packed varlen mental model.
- Easier oracle comparison against SDPA per packed call.

Risks:

- More Python-side packing and scatter.
- Harder to CUDA-graph because active slot count and sequence lengths vary.
- Previous chunked workspace attempts produced NaNs or quality failure when the state/window contract was not exactly right.

This is useful as a debug/oracle bridge, not the first production implementation.

### Option C: Keep Streaming Source Attention, Share Only Infrastructure

Keep current streaming codec attention and focus on graph/profile/refactor while non-streaming remains packed SGLang.

Pros:

- Lowest correctness risk.
- Current streaming tests already encode this contract.

Risks:

- Does not satisfy the goal of SGLang-backed streaming attention.
- Leaves streaming performance limited by upstream attention implementation.

This remains the fallback architecture if Options A/B fail quality gates.

## Detailed Execution Plan

### Phase 0: Trace Streaming Attention Semantics

Goal: prove exactly what streaming attention is doing before replacing it.

Add a debug-only harness, not server hot-path code:

- Run real `MOSS-Audio-Tokenizer-v2` under `codec.streaming(B)`.
- Feed deterministic chunks with these cases:
  - single slot: `T=1,4,5,8,13,25,100`;
  - multi-slot uniform: `B=2,4,8`, same T;
  - ragged timeline: staggered starts, abort/release/reuse, offline slot while streams live;
  - boundary lengths around every captured graph T: `4,5,8,9,10,11,12,13,20,22,24,25`;
  - long stream crossing multiple context windows.
- For selected attention modules and layers, record:
  - module path;
  - input chunk T;
  - active slot ids;
  - `exec_mask`;
  - cache lengths before/after;
  - absolute positions used by RoPE;
  - q/k/v shapes;
  - effective key context length;
  - local window tuple;
  - output max/mean delta against source SDPA oracle.

Deliverable:

- `tasks/moss_streaming_sglang_vocoder_trace_findings.md`
- JSON trace artifact generated on H100.

Gate:

- Prepared inputs and cache lengths must match source streaming behavior exactly.
- Any SGLang per-call oracle must stay in the established BF16-scale envelope before we benchmark audio.

### Phase 1: Extract Shared Attention Utilities

Only after Phase 0 confirms real reuse.

Candidate shared module:

```text
sglang_omni/models/moss_tts_local/vocoder_attention.py
```

Move or introduce:

| Helper | Source today | Consumer after extraction |
| --- | --- | --- |
| `moss_flash_window(context, causal)` | `MossTTSLocalAttention._flash_window_size` | non-stream + streaming adapter |
| `MossRopeCache` | `_MossPackedRopeCache` | non-stream + streaming adapter if using MOSS RoPE |
| `apply_moss_interleaved_rope` | `_apply_cached_packed_rope` | non-stream + streaming adapter |
| SGLang FA binding | local import in `vocoder_decoder.py` | non-stream + streaming adapter |

Do not extract:

- pack/unpack code unless streaming uses it;
- stage wrappers;
- scheduler slot code;
- CUDA graph runner code.

Gate:

- Existing unit tests pass without weakening assertions.
- Non-stream full 1088 result remains in the merged PR envelope.
- No behavior change in streaming tests.

### Phase 2: Streaming Attention Adapter Prototype

Implement a named adapter module that wraps source attention modules without changing scheduler semantics.

Candidate file:

```text
sglang_omni/models/moss_tts_local/vocoder_streaming_attention.py
```

Candidate classes/functions:

| Name | Responsibility |
| --- | --- |
| `MossStreamingAttentionBackend` | Construction-time backend object for source attention modules. |
| `install_moss_streaming_attention(codec)` | Finds codec decoder attention modules and replaces/patches only their attention forward path. |
| `MossStreamingAttentionStateView` | Thin view over source `_streaming_state`; no duplicated owner state. |
| `forward_streaming_flash(...)` | Calls SGLang `flash_attn_with_kvcache` or packed varlen once shape/caches are proven. |

Design rules:

- The codec still owns streaming state.
- Scheduler still owns slots and `exec_mask`.
- Adapter must respect inactive slots and reset masks.
- Adapter must not call the non-streaming `MossTTSLocalVocoderDecoder`.
- Adapter must not use SGLang fused RoPE initially.
- No dynamic fallback inside the attention hot path. Select a backend at construction; if unsupported, do not install it.

Phase 2A should be eager-only:

- Disable graph capture for adapter validation or run graph off.
- Prove local oracle and audio quality first.

Phase 2B should restore graph support:

- Ensure all buffers used by adapter are static for captured T.
- Capture largest T first as current graph runner does.
- Validate graph replay equals eager adapter output for the same chunk schedule.

### Phase 3: Integration Into Scheduler

Keep scheduler changes narrow.

Expected changes:

- Add constructor argument describing streaming vocoder attention backend.
- Create codec attention adapter before `warmup_now()` so graph capture sees final attention implementation.
- Preserve existing non-streaming route:
  - no live session: packed full-sequence decoder;
  - live session: offline slots in persistent session.
- Add route proof logging:
  - streaming backend: source SDPA, SGLang kvcache, or SGLang packed;
  - graph mode: captured/eager;
  - captured T set.

Avoid:

- global monkey-patching;
- per-request backend decisions;
- hidden env-only behavior;
- unifying non-stream and streaming by forcing all decode through one path.

### Phase 4: Validation And Benchmark Gate

Unit tests:

- Existing streaming tests remain unchanged where possible.
- Add tests only for real contracts:
  - adapter preserves reset-mask semantics;
  - inactive slots do not advance state;
  - graph capture uses installed adapter;
  - unsupported backend is not installed at construction.

H100 mechanical probes:

| Probe | Purpose |
| --- | --- |
| Single stream, chunk sizes 1/4/8/25/100 | Basic streaming continuity. |
| Two concurrent streams staggered | Slot/cache isolation. |
| Abort then reuse slot | Reset correctness. |
| Offline request while stream live | Offline slot correctness. |
| Boundary captured T set | Graph bucket coverage. |
| Long stream beyond context | Window/cache correctness. |

H100 quality gate:

- Streaming audio comparison against source streaming, not non-streaming full decode only.
- SeedTTS or equivalent real-audio subset with streaming enabled if benchmark path supports it.
- WER, UTMOS, speaker similarity, audio integrity.
- Bucket by duration and by emitted chunk count.

Performance gate:

- Compare:
  - source streaming eager;
  - source streaming CUDA graph;
  - SGLang streaming eager;
  - SGLang streaming CUDA graph if implemented.
- Request profile rows:
  - attention projection;
  - RoPE;
  - cache update;
  - FlashAttention;
  - output projection;
  - D2H materialization.
- Torch/Nsight profile only after request profile shows a real target.

## Risks

| Risk | Why it is real | Mitigation |
| --- | --- | --- |
| Streaming state mismatch | Earlier attempts showed missing context created wrong K shape and large drift. | Phase 0 traces cache length/position/context before replacement. |
| FlashAttention numerical drift | Prior streaming FA quality failed when attention/window was not exactly right. | Per-call oracle first, then real-audio gate. |
| RoPE mismatch | Prior SGLang JIT RoPE improved speed but regressed WER. | Keep MOSS RoPE math first; optimize later in isolation. |
| CUDA graph pointer instability | Current graph code has explicit largest-first and reset-before-capture safeguards. | Adapter must use static buffers before graph capture. |
| Hidden live-session regression | Non-streaming offline-lane behavior is easy to break. | Preserve `_CodecStreamSession.decode_offline` route and tests. |
| Over-unification | One class for streaming and non-streaming would create invalid states. | Share kernels/utilities, not execution ownership. |

## Open Questions

1. Does the remote MOSS codec attention expose enough stable cache tensors to use `flash_attn_with_kvcache` without duplicating cache ownership?
2. Can SGLang kvcache FA handle MOSS local causal window and MOSS interleaved RoPE with exact enough output under BF16?
3. Is graph capture still worth it once SGLang streaming attention is installed, or does eager SGLang beat graphed SDPA for normal chunk sizes?
4. Should streaming SGLang support be default once validated, or should it be a named backend selected at construction until maintainers accept the quality envelope?
5. Does streaming benchmark coverage already exist for MOSS-TTS Local, or do we need a dedicated benchmark harness?

## Recommendation

Proceed in this order:

1. Build the Phase 0 streaming attention trace harness.
2. Use its evidence to choose `flash_attn_with_kvcache` or packed varlen for streaming.
3. Extract only the attention helpers that the chosen streaming path actually uses.
4. Implement an eager streaming attention adapter.
5. Re-enable CUDA graph capture only after eager streaming quality is stable.

This keeps the final architecture high quality: non-streaming remains the fast packed full-sequence path that already merged, while streaming gets a cache-aware SGLang backend designed around the real codec streaming state instead of trying to force the non-streaming decoder into a different runtime contract.
