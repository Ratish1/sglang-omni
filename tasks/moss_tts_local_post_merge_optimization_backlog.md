# MOSS-TTS Local Post-Merge Optimization Backlog

## Scope

This backlog starts after the packed non-streaming vocoder, owned codec loader, and full streaming vocoder CUDA graph coverage have merged. The next work should be evidence-driven: profile the merged baseline first, then apply one low-risk mechanical optimization at a time.

## Current Runtime Map

```text
/v1/audio/speech
  -> preprocessing
     -> reference audio encode/cache
  -> tts_engine
     -> Qwen backbone decode
     -> MOSS local frame decode CUDA graphs
     -> optional streaming rows to vocoder
  -> vocoder
     -> streaming: persistent codec.streaming() session, exact-T CUDA graphs
     -> non-streaming: full-sequence decode with packed SGLang FlashAttention
  -> CPU audio payload / streaming chunks
```

## Known Hotspots From Previous Profiles

### Streaming, Before Full Graph Coverage

- `eager_decode` dominated when common `T` values were not captured.
- `graph_decode` was small once the graph path was hit.
- `output_d2h`, `pump_step`, `on_done`, and step preparation were still visible.

After full `T=1..25` capture, eager decode should no longer dominate. The next profile must prove where time moved.

### Non-Streaming

- Packed FlashAttention is already comparatively cheap.
- The largest detailed attention-side cost was MOSS RoPE, especially float conversion, rotate, and stack.
- Prior SGLang/native RoPE attempts were faster but caused WER regressions, so RoPE work is high-risk until exact MOSS semantics are proven.
- Pack/unpack and metadata setup are measurable but smaller.

## Risk Tiers

- **Low**: no math change, no scheduler contract change, no new output tolerance.
- **Medium**: preserves math but changes state lifetime, memory pressure, batching, or copy ordering.
- **High**: changes numerical path, chunking semantics, graph/compile scope, or stage payload contracts.
- **Research**: requires new kernel, new scheduling policy, or new correctness harness before implementation.

## Optimization Candidates

| # | Area | Candidate | Risk | Validation gate |
|---:|---|---|---|---|
| 1 | Shared | Re-profile merged streaming and non-streaming paths with request events and torch/Nsight traces | Low | Reports identify new top rows after full graph coverage |
| 2 | Streaming | Preallocate per-session `codes_step`, `codes_lengths`, and `exec_mask` tensors | Low | Graph-vs-eager identity, streaming c8 speed, no duration runaway |
| 3 | Streaming | Preallocate per-session reset mask | Low | Slot release/reuse tests and streaming c8 |
| 4 | Streaming | Add request-profile markers for step prepare, graph/eager decode, D2H, and output slicing | Low | Profile rows appear only when request profiler is active |
| 5 | Streaming | Cache streaming-state reset modules after proving they are not lazy-created | Medium | Trace reset coverage before/after and release/reuse identity |
| 6 | Streaming | Pinned CPU staging buffers for audio D2H | Medium | Same waveform bytes, lower D2H/profile time |
| 7 | Streaming | Nonblocking D2H overlap with next graph replay | Medium-High | Strict ordering tests, no stale static graph output |
| 8 | Streaming | Tune `initial_chunk_frames` and `stream_chunk_frames` under full graph coverage | Low | c1/c8/c16 TTFC, ITL, RTF, WER |
| 9 | Streaming | Emit structured graph coverage metrics instead of log-only CG stats | Low | Benchmark parser consumes graph/eager T directly |
| 10 | Streaming | Adaptive graph capture policy based on measured `T` histogram | Medium | Memory A/B, no eager regression in default benchmark |
| 11 | Streaming | Optional graphs for offline-lane `T > stream_chunk_frames` if mixed traffic shows eager decode | Medium | Mixed stream + non-stream benchmark |
| 12 | Streaming | Same-process GPU tensor handoff from AR rows to vocoder | High | Stage payload contract audit, D2H/H2D profile, WER |
| 13 | AR | Experiment with async decode only for graph-covered MOSS batches | High | Finish/retract/repetition parity and throughput |
| 14 | AR | Compile more of `_decode_frame_graphable` | Medium-High | Compile cache count, startup cost, graph replay parity |
| 15 | AR | Fuse local transformer step and seeded sampler | Research | Kernel-level parity for seeded sampling |
| 16 | AR | Retune frame-decode CUDA graph batch buckets for c8/c16/c32 | Medium | Memory A/B and c8/c16/c32 speed |
| 17 | Non-stream | Re-profile packed vocoder on merged main | Low | Updated top rows before code changes |
| 18 | Non-stream | MOSS-specific safe RoPE fusion | High | All-layer RoPE oracle, direct vocoder probe, WER/UTMOS/similarity |
| 19 | Non-stream | Remove packed decoder `.item()` host sync where max length is already known | Medium | No shape/mask regressions, profile sync reduction |
| 20 | Non-stream | Cache cu-seqlens and position metadata for repeated length patterns | Low-Medium | Direct probe and pack profile rows |
| 21 | Non-stream | Static pack/unpack workspace | Medium | Ragged/boundary direct probe and memory profile |
| 22 | Non-stream | CUDA graph common full-sequence vocoder shapes | High | Memory guard, exact shape coverage, quality gate |
| 23 | Non-stream | Compile stable transformer submodules only | Medium-High | Compile graph count, WER, speed after warmup |
| 24 | Non-stream | Cost-based batching by total vocoder frames | Medium | Mixed short/long throughput and tail latency |
| 25 | Shared | Paired outlier audit automation for WER variance | Low | Same generated audio retranscribe variance report |
| 26 | Shared | NVTX/torch profiler integration with MOSS event names | Low | Trace contains AR and vocoder ranges in the correct process/thread |
| 27 | Shared | c1/c8/c16/c32 benchmark matrix for stream and non-stream | Low | Prevents overfitting to c8 SeedTTS |
| 28 | Shared | Graph/compile memory accounting at startup and peak | Low | Per-option memory delta table |

## First Debug Branch Slice

1. Add profiling markers around streaming vocoder step phases.
2. Emit profiling markers once per batched vocoder step, not once per participating request.
3. Reuse per-session step/reset tensors to remove repeated hot-path allocations.
4. Do not change chunking, codec math, graph capture policy, output formatting, or non-streaming math.
5. Validate with unit tests locally, then H100 streaming profile:
   - graph coverage remains 100% for `T=1..25`
   - graph-vs-eager identity remains bit-exact
   - no duration runaway
   - request profile shows actual per-step prepare/decode/D2H/output-slice intervals
