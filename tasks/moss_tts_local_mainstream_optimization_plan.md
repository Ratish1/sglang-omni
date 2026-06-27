# MOSS-TTS Local Mainstream Optimization Plan

## Goal

Optimize the merged MOSS-TTS Local runtime with the same mechanics SGLang uses for LLM serving: static buffers, CUDA graph replay, compile only around stable graph contexts, ordered torch/NVTX ranges, and strict graph-vs-eager correctness gates.

This plan excludes config tuning as the primary path. Chunk-size tuning, small copy cleanups, and payload formatting are secondary unless profiling proves they dominate.

## Current Read

The merged full streaming CUDA graph work moved streaming vocoder coverage to `T=1..25` and removed the earlier eager-vocoder bottleneck. The latest profiling shows:

- Streaming vocoder graph coverage is clean.
- The apparent D2H interval is mostly synchronization wait. The actual GPU D2H copy is small.
- Chunk-frame tuning did not improve throughput.
- GPU row-handoff style changes did not improve throughput.
- Prior frame-decode `torch.compile` attempts gave speed signal but failed direct parity.

So the next serious target is not streaming chunking. It is the AR frame decode and its graph boundary.

## Runtime Flow

```text
preprocessing/reference encode
  -> SGLang Qwen backbone forward
  -> MOSS AR frame decode
       hidden state
       local transformer step 0
       binary stop sample
       12 sequential RVQ samples
       feedback embedding write
       radix id hash
  -> streaming vocoder
       per-step codec decode graph replay
       CPU audio materialization
```

## SGLang Patterns To Reuse

| SGLang pattern | Source | MOSS transfer |
|---|---|---|
| Static graph input buffers | `/Users/ratish/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` | Keep AR/vocoder graph inputs address-stable and copy into preallocated buffers. |
| Grouped device copies | `DecodeInputBuffers.populate_from_forward_batch` | Batch copy groups where we prove copy overhead dominates. |
| Compile only in capture-safe contexts | `patch_model`, `install_torch_compiled`, piecewise graph context | Do not blindly compile frame decode. Compile only after fixed-batch parity proves stable. |
| Graph/eager hit accounting | SGLang graph runner and MOSS vocoder CG stats | Add AR frame-decode graph/eager stats by batch size. |
| Ordered torch/NVTX ranges | SGLang scheduler/profile hooks | Add explicit `moss_ar_*` ranges so torch traces show kernel sequence, not just aggregate request timing. |

## Invariants

### AR Frame Decode

Any AR optimization must preserve, for fixed hidden states, request order, seeds, base positions, and pool rows:

- binary stop choice
- all 12 RVQ codes
- full generated row
- feedback embedding
- radix token id
- sampling step writes
- audio repetition penalty fallback
- chunked-prefill non-final suppression
- padded bucket rows ignored by callers

Cross-run c8 hash equality is not a hard gate because normal serving batch composition can change hidden/logit numerics. Fixed-batch graph-vs-eager parity is the hard gate.

### Streaming Vocoder

Any streaming-vocoder optimization must preserve:

- persistent `codec.streaming()` state ownership
- slot reset/reuse semantics
- exact-T graph replay identity against eager for all captured `T`
- no duration runaway
- graph/eager stats showing no unexpected fallback

### Non-Streaming Vocoder

Packed non-streaming work is already fast. Changes here must preserve:

- packed local-window attention oracle
- no NaNs
- shape parity
- WER/UTMOS/speaker-sim in baseline envelope

RoPE fusion stays high risk until an all-layer MOSS RoPE oracle proves q/k and attention output parity.

## Phase 1: Profiling-Only Slice

Add ordered torch profiler ranges and AR graph/eager counters:

- `moss_ar_collect_frame`
- `moss_ar_gather_sampling_params`
- `moss_ar_frame_decode_graph`
- `moss_ar_frame_graph_copy_inputs`
- `moss_ar_frame_graph_replay`
- `moss_ar_frame_graph_outputs`
- `moss_ar_frame_decode_eager`
- `moss_ar_build_rows`
- `moss_ar_pool_write`
- `moss_ar_radix_hash`
- `moss_ar_post_decode_launch`
- `moss_ar_post_decode_resolve`

This phase changes no math and no policy. It only tells us whether the next bottleneck is graph input copy, graph replay, pool writes, radix hash, or scheduler resolve.

## Phase 2: Native Pool-Resident Frame Decode

Only after Phase 1 confirms AR boundary cost:

- Keep generated rows, feedback embeddings, and radix ids on device.
- Avoid changing local transformer math or sampler math.
- Reuse existing graph-safe static buffers.
- Preserve eager fallback for repetition penalty and oversized batches.

Validation:

- fixed-batch parity for `bs=1,2,4,8`
- padded bucket parity, including raw `bs=1,2,4` into bucket `8`
- non-contiguous pool rows
- chunked-prefill path
- repetition-penalty eager fallback unchanged

## Phase 3: Graph Bucket Retuning

Use Phase 1 stats to see actual AR frame-decode batch-size distribution under c8/c16/c32.

Only add or remove buckets if:

- graph hit rate improves
- memory increase is measured
- startup capture remains acceptable
- graph-vs-eager identity passes

## Phase 4: Compile/Fusion Research

Do not ship compile/fusion unless fixed-batch parity passes first.

Allowed experiments:

- compile isolated sampler only, if sampler parity is exact
- compile a pool-resident wrapper only if output tensors are exact
- custom op wrapper for copy/fill patterns if torch trace shows many small kernels

Rejected unless new evidence changes it:

- full `_decode_frame_graphable` compile as production path
- logits-only compile as production path
- MOSS RoPE replacement without all-layer oracle

## H100 Profiling Gate

For the profiling branch:

1. Run streaming c8 generate-only with request profile.
2. Run stage-targeted torch profiler for the TTS engine process.
3. Confirm the trace contains `moss_ar_*` ranges and ordered kernels under them.
4. Confirm AR frame graph stats log graph/eager batch-size counts.
5. Compare speed only as a sanity check, not as an acceptance gate, because profiler ranges can add overhead.

Expected useful output:

```text
MOSS-TTS Local frame-decode graph stats: ... graph bs={...} eager bs={...}
```

Then inspect:

- time under `moss_ar_frame_graph_copy_inputs`
- time under `moss_ar_frame_graph_replay`
- time under `moss_ar_pool_write`
- time under `moss_ar_radix_hash`
- time between AR launch and streaming vocoder step

Only after that should we implement the next actual optimization.
