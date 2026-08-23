# Qwen3-TTS performance PR ledger

## Current decision

**Status: conditional.** Seven Qwen3-TTS ranges are mechanically free of the
selected blocking H2D mechanism. No repeatable end-to-end speedup has been
established for this hidden-sync branch. The Qwen repetition-mask candidate is
not semantically ready to land because it optimizes a repetition transform that
is already applied by SGLang's sampler.

- Repository branch: `perf/qwen3-tts-hidden-h2d-sync-v2`
- Immutable comparison base: `2cac60e8ac38cf5d3c7091ec3dd15782bc8b1f41`
- Audited branch head: `c4203e662b3977ab655ad1b9b9cb4eff6ca0d4f4`
- Pinned serving dependency: SGLang `0.5.16`
- H100 evidence model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- Shared-code model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Supported sizes are 0.6B and 1.7B, not 0.7B and 1.5B.
- Numerical contract: one repetition transform, unchanged public sampling
  defaults, unchanged codec suppression, and unchanged token/audio semantics.

## Evidence truth table

| Claim | Result | Evidence |
|---|---|---|
| Selected pageable H2D and scalar-assignment synchronizations were removed | Proved | c1 detector plus all-thread c16 traces |
| Text-tokenizer H2D remains exercised after the rewrite | Proved | 128 range calls and 128 correlated H2D launches |
| Selected rewrites introduced no replacement stream/event/device wait | Proved | Zero synchronization in all seven selected ranges |
| Pinned-host allocation was bounded in the text trace | Bounded in one run | Current allocated bytes increased by 17,249; allocator current was 1,601,606 bytes at stop |
| Hidden-sync branch improves end-to-end serving | Not proved | Prior matched trials were neutral; the one full-corpus pair was distorted by one unseeded 2,048-token tail |
| Qwen repetition is applied once | False on current branch/base | Omni shapes logits, then SGLang shapes the same logits again |
| 1.7B receives the same mechanism changes | Source-proved | Both sizes use the same request builder, talker runner, and vocoder paths |
| 1.7B performance benefit | Not measured | Only 0.6B has current H100 trace/A-B evidence |

The latest full 1,088-prompt pair is not a speedup proof. Candidate mean and p95
request latency were about 2% lower, but candidate throughput was reduced by one
unseeded max-length generation. Earlier alternating trials measured no material
c16 change. Report the current work as synchronization cleanup only.

The already-merged Qwen3-TTS performance work is separate. SGLang-Omni PR
[#1462](https://github.com/sgl-project/sglang-omni/pull/1462) measured larger
host-loop and scheduling gains, but it also introduced the device repetition
mask that this ledger now identifies as duplicate ownership. Its reported
performance does not establish an incremental gain for this branch.

## Branch code ledger

| Commit | Mechanism | Mechanical status | Semantic status | Performance status | Landing action |
|---|---|---|---|---|---|
| `67f075bc` | Restage semantic/subtalker metadata through pinned CPU tensors into persistent CUDA buffers | Pass | Ready | Neutral/not isolated | Keep in H2D PR |
| `315ebb3d` | Speaker mel, cached speaker embedding, prompt token rows, and reference-code same-stream H2D | Pass | Ready, subject to per-path scope below | Neutral/not isolated | Split into model-input H2D PR |
| `86884103` | Replace advanced scalar assignment with Scalar `index_fill_`/`scatter_`; stage mask metadata pinned | Pass | Mixed: suppression is needed; repetition half is duplicate | Not isolated | Do not land intact; retain suppression-only mechanics after ownership fix |
| `2ff0329a` | Preserve official text processor and copy immutable token IDs through pinned host memory | Pass | Ready | No end-to-end A/B | Small independent H2D PR |
| `eaa4dd97`, `167a2573` | Process sync detector and all-thread Torch profiling | Diagnostic pass | Independent of inference semantics | n/a | Profiler PR |
| `35b913ec`, `0a30cb5a` | Semantic attribution and reusable trace analyzers | Diagnostic pass | Independent | n/a | Profiler/analyzer PR |
| documentation commits | Runbook and evidence | Current through `c4203e66` plus this ledger | n/a | n/a | Travel with corresponding PRs |

Seven mechanically clean ranges at `c4203e66`:

| Range | Calls | Correlated H2D | Synchronizations |
|---|---:|---:|---:|
| `qwen3_tts.preprocess.speaker_mel_h2d` | 44 | 44 | 0 |
| `qwen3_tts.preprocess.speaker_embedding_h2d` | 64 | 64 | 0 |
| `qwen3_tts.prompt.token_ids_h2d` | 192 | 192 | 0 |
| `qwen3_tts.prompt.ref_code_h2d` | 64 | 64 | 0 |
| `qwen3_tts.sampling_masks.rebuild` | 116 | 313 | 0 |
| `qwen3_tts.sampling_metadata.h2d` | 324 | 696 | 0 |
| `qwen3_tts.preprocess.text_tokenizer` | 128 | 128 | 0 |

The sampling-mask row proves clean mechanics, not that both masks should exist.

## Repetition-penalty ownership

### Current call chain

1. `sglang_omni/model_runner/base.py::_sample_next_token_ids` calls Omni
   `_apply_repetition_penalty`.
2. `Qwen3TTSModelRunner` builds an output-token boolean mask `[B, V]`, a
   suppression mask `[B, V]`, and penalty column `[B, 1]`.
3. Omni applies sign-aware scaling to every previously generated token.
4. `tp_worker.model_runner.sample` enters SGLang.
5. SGLang `SamplingBatchInfo.apply_logits_bias` calls
   `BatchedRepetitionPenalizer`, which applies the same sign-aware scaling from
   its generated-output state.

For a seen token, a requested penalty `r=1.05` currently acts like
`r^2=1.1025`; `r=1.1` acts like `1.21`. This is a semantic bug, not merely host
overhead.

### Correct owner

**Keep SGLang as the single owner of standard repetition penalty. Keep codec
vocabulary suppression in the Qwen model adapter.**

Reasons:

- SGLang owns sampling, batch filter/merge, and emitted-token accumulation.
- Official Qwen passes `repetition_penalty=1.05` once to Transformers generate.
  With decoder-only `inputs_embeds`, the processor history is generated IDs.
- SGLang's penalizer is also generated-output-only.
- vLLM-Omni Qwen3-TTS masks invalid codec IDs in the model and passes
  `repetition_penalty=1.05` to the vLLM sampler once. It does not add another
  Qwen repetition transform.
- vLLM-Omni is not a numerical oracle here: its artificial prompt consists of
  placeholder token ID 1, while vLLM's generic penalty includes prompt tokens.
  That can pre-penalize codec ID 1 and should be filed separately upstream.

### Required design before removal

Do not delete the Qwen method alone. It currently rebuilds both repetition and
codec-suppression masks, and the suppression method assumes that rebuild ran.

The focused Qwen3-TTS correction is:

1. Split suppression-mask construction from repetition state.
2. Preserve a device-resident Qwen codec-suppression mask only.
3. Make Qwen's Omni repetition hook a no-op, allowing the shared base sampling
   and logprob path to continue into SGLang unchanged.
4. Delete Qwen repetition-only state: `rep_mask`, `pen_col`,
   `_mask_last_sampled`, `_mask_rep_active`, incremental generated-ID sets, and
   repetition-only rebuild logic where no other model uses it.
5. Restore the complete generated-token penalty set when a retracted request is
   re-prefilled and SGLang constructs fresh `SamplingBatchInfo`.

Step 5 is the blocker. A fresh SGLang penalizer starts with `[B,V]` ones and its
normal decode path scatters one latest token per step. Omni currently rebuilds
from all `req.output_ids`, so removing it without a restoration protocol can
forget pre-retraction history. Prefer fixing/seeding the SGLang state at the
batch construction contract; avoid reaching into private penalty tensors from
the Qwen model runner.

Do not remove the generic base repetition hook for all Omni models in this PR.
Audit every AR runner first; Qwen3-Omni has a separate model-owned talker
sampling path.

### Repetition proof gates

- Spy/call-order proof: every seen logit is transformed once, not twice.
- Positive, negative, and zero logits; duplicate token IDs; mixed row penalties;
  fp16, bf16, and fp32.
- Parity with Transformers `RepetitionPenaltyLogitsProcessor` for generated-only
  history.
- First prefill and token-by-token decode parity.
- Retract/re-prefill with nonempty history; merge, filter, shrink, and request-ID
  reuse; no stale row state.
- Codec suppression parity including EOS policy and out-of-range IDs.
- Fixed-seed codec-token comparison before audio/WER comparison.
- Profile one full-vocabulary penalty application removed, then benchmark
  B=1/16/64. Performance is a measured result, not an acceptance assumption.

## PyTorch mechanism ledger

| Situation | Current cost/risk | Correct primitive | Shape and lifetime contract |
|---|---|---|---|
| Python list or pageable CPU tensor passed to `torch.tensor(..., device="cuda")` or `.to(cuda)` | PyTorch enqueues `cudaMemcpyAsync`, then synchronizes the stream for a blocking copy | Create pinned CPU source and use `.to(cuda, non_blocking=True)` | Source immutable; first consumer on same stream; caching host allocator defers reuse until copy event completes |
| Scalar/regular data known without a host tensor | Avoidable allocation and H2D | CUDA `full`, `zeros`, `ones`, `arange`, or basic slicing | Exact dtype/device/value; cache only after device/dtype/model lifetime is stable |
| Python scalar assigned through CUDA advanced indexing | Python scalar is wrapped as a CPU tensor and `index_put_` can copy it synchronously | `index_fill_(..., Scalar)` or `scatter_(..., Scalar)` | Flatten `(row, token)` as `row * V + token`; preserve duplicate and bounds behavior |
| Dynamic small metadata reused by a CUDA consumer | Repeated host construction and blocking H2D | Pinned staging plus `copy_(..., non_blocking=True)` into persistent CUDA destination | Do not overwrite source/destination before prior same-stream consumers; use ping-pong only when producer can lap consumer |
| CUDA value immediately converted with `.item()`, `bool(tensor)`, or a tensor slice bound | Device-to-host synchronization serializes Python control flow | Retain the length/decision as a Python/CPU value before H2D, or express control on device | Derive with the exact padding/downsample formula; prove all batch rows and empty input |
| D2H result whose CPU consumer can wait | Pageable D2H or immediate `.cpu()` drains the stream | Explicit pinned host destination, `copy_(non_blocking=True)`, record event, consume after event | Buffer cannot be read/reused before event; bounded pool; cancellation/shutdown retires in-flight slots |
| Cross-stream CUDA handoff | `record_stream` alone does not order execution | Producer records event; consumer stream calls `wait_event`; then `record_stream` for allocator lifetime | `wait_event` orders future consumer work without blocking host; retain tensor until consumer completes |
| CUDA-graph static output | Storage is overwritten on the next replay | Enqueue D2H on the replay stream before another replay, or clone when ownership requires it | Never publish a view of graph-static output across iterations |

`pin_memory()` is itself a CPU allocation/copy. It is appropriate for the
small one-shot metadata already qualified. For large or variable audio tensors,
use bounded grow-only pools rather than pinning a new allocation per request.
Setting `non_blocking=True` on pageable memory is not the design fix.

The source debugging command remains:

```python
torch.cuda.set_sync_debug_mode("error")
```

Use `"warn"` for inventory. The installed Torch build detects scalar reads,
pageable H2D/D2H, and explicit stream synchronization, but its microprobe did
not warn for `cudaDeviceSynchronize`; therefore pair it with a clean Kineto
trace. The reference is the PyTorch article
[Understanding GPU Memory 2: Finding and Removing Hidden H2D Synchronizations](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/).

## Next synchronization owners

Current trace priority after the seven clean ranges:

| Owner | Observed mechanism | Design, not patch | Serving scenarios to preserve |
|---|---|---|---|
| Vocoder tokenizer decode | 138 scalar D2H; 37 pageable H2D; 64 waveform D2H | Retain code lengths as CPU integers, batch `[B,Q,T]`, call decoder model directly, then publish waveform through a bounded pinned-D2H/event pool | deterministic B=1, batched nonstreaming, streaming private streams, invalid-row isolation, reference trimming, shutdown |
| Reference tokenizer encode | 881 scalar D2H; 54 large pageable H2D | Retain waveform lengths before padding and use exact encode downsample rule; return producer event with CUDA code; redesign cache publication before removing host wait | variable sample rate/length, x-vector-only, cache hit/miss, concurrent readers, cancellation, cross-thread consumer |
| Cache key D2H | 64 immediate CPU BLAKE2 consumers | Change key ownership/algorithm or compute from stable host inputs | Cache identity and collision contract |
| Speaker artifact cache D2H | 88 immediate CPU publications | Async pinned artifact carrying completion event, or bounded device cache | Concurrent cache readers and process-safe fallback |
| Final engine code D2H | 64 stage handoffs | Device-local same-process fast path plus explicit CPU fallback | Colocated and isolated-stage deployments |
| SGLang repetition setup | 35 tiny H2D in trace | SGLang PR #28076-style pinned/nonblocking metadata construction; remove only after repetition semantics are single-owner | merge/filter/retract and non-CUDA fallback |

The strongest production precedent for the vocoder is vLLM-Omni's
`Qwen3TTSCode2Wav`: it bypasses the HF tokenizer decode wrapper specifically to
avoid GPU-CPU-GPU round trips, retains `request_lengths` as Python integers,
pads CUDA codes as `[B,Q,max_frames]`, and calls the decoder directly. Within
SGLang-Omni, Qwen3-Omni Code2Wav provides the bounded pinned slot/event
publication protocol. Reuse the ownership design, not the classes verbatim,
because Qwen3-TTS has two private decode streams and different state locks.

SGLang PR [#28076](https://github.com/sgl-project/sglang/pull/28076) is the exact
metadata precedent: pinned CPU construction followed by nonblocking H2D for
temperatures/top-p/top-k/min-p/seeds and pointer metadata.

## Model applicability

| Path | 0.6B Base | 1.7B Base | 0.6B CustomVoice | 1.7B VoiceDesign |
|---|---|---|---|---|
| Text tokenizer | Shared | Shared | Shared | Shared |
| Sampling metadata and repetition ownership | Shared | Shared | Shared | Shared |
| Talker prompt constants | Shared implementation | Shared implementation | Variant prompt path | Variant prompt path |
| Reference tokenizer/speaker encoder | Used | Used | Not the normal text-only path | Not the normal text-only path |
| Vocoder decode/publication | Shared | Shared | Shared | Shared |

Mechanically, shared paths improve both 0.6B and 1.7B. The fixed host cost is
likely a larger percentage of 0.6B latency because 1.7B performs more GPU work;
that is an inference until each size is profiled. Do one bounded 1.7B mechanical
trace after the implementation stabilizes, not a repeated full corpus for every
small PR.

## PR split and order

1. **Profiler plumbing and analyzers.** Detector lifecycle, all-thread profiling,
   semantic ranges, trace analyzer, no inference behavior.
2. **Single-owner Qwen3-TTS repetition correction.** Decouple suppression,
   restore SGLang history on re-prefill, remove the second full-vocabulary
   transform. Correctness first; performance measured second.
3. **Already-qualified model-input H2D.** Speaker mel/embedding, prompt IDs,
   reference code, sampling metadata. Exclude obsolete repetition-mask code.
4. **Text-tokenizer pinned H2D.** Small isolated patch with existing mechanical
   trace and host-memory evidence.
5. **Vocoder direct-decoder/CPU-length path.** Remove input H2D and scalar length
   control without yet changing final publication.
6. **Vocoder async waveform publication.** Bounded pinned slots/events and
   cancellation-safe ownership.
7. **Reference encoder and cache publication.** Producer-event contract plus
   async/cache redesign; do not merely remove the current host synchronize.
8. **Style-only cleanup.** Run after behavior PRs to avoid obscuring their diffs.

## Style and duplicate-code queue

- The current branch mixes inference changes, detector plumbing, analyzers, and
  runbooks: split before review.
- `sample_before_post_prefill` and `sample_before_post_decode` have identical
  bodies but are distinct required hooks; retain names and optionally delegate
  to one private predicate only if it improves the interface implementation.
- `_prepare_qwen3_tts_base_request`, `_prepare_qwen3_tts_custom_voice_request`,
  and `_prepare_qwen3_tts_voice_design_request` repeat text/instruction
  tokenization and the profiler/no-grad wrapper. Extract only the truly common
  prompt preparation; keep variant model calls explicit.
- `_vocode_payloads` and `_decode_state_audio` repeat tokenizer invocation and
  reference trimming. Factor a common decode/trim helper only after the direct
  decoder PR defines the final tensor and error contracts.
- `_copy_cpu_tensor_to_cuda_consumer` and `_tokenize_qwen3_tts_text` deliberately
  share a pinned-H2D mechanism but have different API/shape contracts. Do not
  introduce a generic helper until a third stable use proves the abstraction.
- Constant prompt embeddings and token rows are rebuilt in several prompt
  builders. A later allocation PR may cache them after model load, but must
  invalidate on device/dtype moves or weight reload.
- The vLLM-Omni placeholder prompt/repetition discrepancy is a separate upstream
  correctness issue, not part of SGLang-Omni cleanup.

`git diff --check` passes at the audited head. Ruff was not available in the
local shell during this audit, so no formatter/linter pass is claimed.

## Immediate next action

Design and implement PR 2 first. It determines which portion of the current
sampling-mask candidate survives. Do not run another full 1,088-prompt A/B until
single-owner repetition passes the fixed-seed token/retract gates and a bounded
profile confirms one full-vocabulary transform is gone. After that, one 0.6B
full-corpus pair and one bounded 1.7B mechanism trace are sufficient for the
first qualification round.
